//! The forward calls the bench workflow measures: for each text checkpoint it lists, one call of
//! each nightly serving workload of `.github/benchmarks.json`, repeated, in a Chrome trace that CI
//! reads forward and kernel metrics from.
//!
//! - A workload's calls are the histories its steps reach: the prompt's last chunk, and a decode
//!   step halfway through the output for one request and for each concurrency every nightly
//!   workload runs. A workload past the checkpoint's context is not served and not measured.
//! - A case's measured calls are `forward` spans of its `case` and `step`, in
//!   `trace/<org>--<name>/forward.json`; kernel spans join them under `FURIOSA_OPT_PROFILE`.

use std::io;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use furiosa_opt_engine::decoder::{Cache, Keep, Request, Rows};
use furiosa_opt_engine::{Checkpoint, LoadOptions};
use furiosa_opt_std::prelude::{Device, HostBufferSlice, Topology};
use tracing::Instrument;

/// Unmeasured calls before a case's samples, while the device and the host's caches settle.
const WARMUPS: usize = 3;
/// Measured calls of a case, whose median CI records.
const REPETITIONS: usize = 20;

/// Each checkpoint, at the hub revision these tests read, measured as `measure` says.
macro_rules! forward {
    ($($test:ident: $hub:expr;)*) => {$(
        #[test]
        #[ignore = "opens a device"]
        fn $test() {
            measure($hub);
        }
    )*};
}

forward! {
    qwen3_0_6b_forward: ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca");
    qwen3_1_7b_forward: ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e");
    qwen3_4b_forward: ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c");
    qwen3_8b_forward: ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218");
    llama3_8b_forward: ("NousResearch/Meta-Llama-3-8B", "315b20096dc791d381d514deb5f8bd9c8d6d3061");
    mistral_7b_v0_3_forward:
        ("mistralai/Mistral-7B-v0.3", "caa1feb0e54d415e2df31207e5f4e273e33509b1");
}

/// One call of each case the workloads reach on the decoder that the checkpoint of hub `repo` at
/// `revision` selects, repeated. Timing reads no cache values, so the histories stay unwritten.
fn measure((repo, revision): (&str, &str)) {
    let checkpoint = Checkpoint::new(snapshot((repo, revision)), None).unwrap();
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let model = LoadOptions::new()
        .load(&mut device, &checkpoint)
        .unwrap()
        .decoder()
        .unwrap();
    let limits = model.limits();
    let (chunk, block) = (
        limits.rows,
        limits.cache.pools[limits.cache.groups[0].pool].block,
    );
    let (lengths, widths) = workloads();
    // Each case: its name, the positions cached before the call, a request's tokens, the requests.
    let mut cases: Vec<(String, usize, usize, usize)> = lengths
        .iter()
        .flat_map(|&(input, output)| {
            let decode = std::iter::once(1)
                .chain(widths.iter().copied())
                .map(move |rows| {
                    let case = format!("in-{input}-out-{output}-decode-rows-{rows}");
                    (case, input + output / 2, 1, rows)
                });
            let prefill = format!("in-{input}-prefill-chunk-{chunk}");
            std::iter::once((prefill, input - chunk, chunk, 1)).chain(decode)
        })
        .filter(|&(_, cached, tokens, rows)| cached + tokens <= limits.window && rows <= chunk)
        .collect();
    // Workloads of one input length share their prompt.
    cases.sort();
    cases.dedup();
    let used = |cached: usize, tokens: usize| (cached + tokens).div_ceil(block);
    let blocks = (cases.iter())
        .map(|&(_, cached, tokens, _)| used(cached, tokens))
        .max()
        .unwrap();
    let rows = cases.iter().map(|&(.., rows)| rows).max().unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    // Profiling attaches to functions as they load, at their first call.
    let _trace = trace(repo);
    runtime.block_on(async {
        let cache = Cache::new(&device, &limits.cache, blocks * rows).unwrap();
        let zeros = vec![0_u8; chunk * size_of::<i32>()];
        let ids = HostBufferSlice::from(&zeros[..])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        for (case, cached, tokens, count) in cases {
            let requests: Vec<Request> = (0..count)
                .map(|row| Request {
                    rows: vec![
                        (0..used(cached, tokens))
                            .map(|block| (block + row * blocks) as u32)
                            .collect(),
                    ],
                    cached,
                    tokens: Rows::new(&ids, row * tokens..(row + 1) * tokens).unwrap(),
                    keep: Keep::Last,
                    adapter: None,
                    positions: None,
                    mask: None,
                    overrides: Vec::new(),
                })
                .collect();
            for sample in 0..WARMUPS + REPETITIONS {
                let span = match sample.checked_sub(WARMUPS) {
                    Some(step) => tracing::info_span!("forward", case, step = step + 1),
                    None => tracing::Span::none(),
                };
                async {
                    let _rows = model.forward(&mut device, &cache, &requests, &[]).await?;
                    // Profile readbacks record a launch's kernel spans after it ends, so the
                    // call's span closes only once the device and those readbacks are done.
                    device.synchronize().await.map_err(io::Error::other)
                }
                .instrument(span)
                .await
                .unwrap();
            }
        }
    });
}

/// The nightly serving workloads of the inventory as input and output tokens, and the
/// concurrencies every one of them runs.
fn workloads() -> (Vec<(usize, usize)>, Vec<usize>) {
    let inventory = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.github/benchmarks.json"
    ));
    let inventory: serde_json::Value = serde_json::from_str(inventory).unwrap();
    let nightly = inventory["tasks"]["text"]["serving"]["nightly"]
        .as_array()
        .unwrap();
    let size = |value: &serde_json::Value| value.as_u64().unwrap() as usize;
    let widths = |workload: &serde_json::Value| -> Vec<usize> {
        workload["widths"]
            .as_array()
            .unwrap()
            .iter()
            .map(size)
            .collect()
    };
    let lengths = nightly
        .iter()
        .map(|workload| (size(&workload["input"]), size(&workload["output"])))
        .collect();
    let common = widths(&nightly[0])
        .into_iter()
        .filter(|width| {
            nightly
                .iter()
                .all(|workload| widths(workload).contains(width))
        })
        .collect();
    (lengths, common)
}

/// Records spans as a Chrome trace in `trace/<org>--<name>/forward.json` until both guards drop;
/// CI reads forward and kernel metrics from it.
fn trace(
    repo: &str,
) -> (
    tracing::subscriber::DefaultGuard,
    tracing_chrometrace::Trace<std::fs::File>,
) {
    use tracing_subscriber::prelude::*;
    let directory = std::path::Path::new("trace").join(repo.to_lowercase().replace('/', "--"));
    std::fs::create_dir_all(&directory).unwrap();
    let file = std::fs::File::create(directory.join("forward.json")).unwrap();
    let (layer, trace) = tracing_chrometrace::ChromeLayer::with_writer(file);
    (
        tracing::subscriber::set_default(tracing_subscriber::registry().with(layer)),
        trace,
    )
}

/// The directory of the weights and configs of hub checkpoint `repo` at `revision`, fetched into
/// the hub cache once.
fn snapshot((repo, revision): (&str, &str)) -> PathBuf {
    let mut command = Command::new("uvx");
    command.args(["--from", "huggingface-hub", "hf", "download"]);
    command.args([repo, "--revision", revision, "--quiet"]);
    command.args(["--include", "*.json", "--include", "*.safetensors"]);
    let output = command
        .stderr(Stdio::inherit())
        .output()
        .expect("uvx fetches the test checkpoint");
    assert!(output.status.success(), "the {repo} download failed");
    String::from_utf8(output.stdout).unwrap().trim().into()
}
