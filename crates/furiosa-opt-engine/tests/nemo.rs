//! Mistral NeMo's published static-scale E4M3 checkpoint, which the registry serves on two chips
//! over E4M3 records: chosen from its config by the chips and KV cache a load names, then loaded
//! whole and called through the production entry as vLLM's reference workloads of it call it.
//!
//! - `FURIOSA_OPT_REFERENCES` holds those outputs as the checkpoint's inventory row names them.
//! - Under the batch-invariant tiling every token passes the 5% rule against one of the
//!   reference's two oracles, and a split or shared call changes no token and no record bit; the
//!   adaptive tiling's tokens have no approved tolerance and are judged on request.

mod common;

use std::io;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use furiosa_opt_engine::decoder::{Cache, Keep, Request, Rows, Served};
use furiosa_opt_engine::registry::{Choice, Implementation};
use furiosa_opt_engine::{Checkpoint, LoadError, LoadOptions, Tiling};
use furiosa_opt_std::prelude::*;
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};

use common::Published;

const KV: usize = 8;
const HEAD: usize = 128;
const CHIPS: usize = 2;
/// The coarsest step between two E4M3 records relative to them: three mantissa bits.
const STEP: f64 = 0.125;

/// Records against the ones they should be: their squared distance and those ones' squared size,
/// summed, and how many are exact.
#[derive(Clone, Copy, Default)]
struct Distance {
    squared: f64,
    size: f64,
    exact: usize,
    total: usize,
}

impl Distance {
    fn add(&mut self, found: f32, want: f32) {
        self.squared += f64::from(found - want).powi(2);
        self.size += f64::from(want).powi(2);
        self.exact += usize::from(found == want);
        self.total += 1;
    }

    /// The distance relative to the size of the records they should be.
    fn relative(&self) -> f64 {
        (self.squared / self.size).sqrt()
    }
}

/// Token IDs on every chip of `device`, a copy each.
async fn ids(device: &mut Device, tokens: &[usize]) -> HbmBuffer {
    let bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|&t| (t as i32).to_le_bytes())
        .collect();
    HostBufferSlice::from(&bytes.repeat(CHIPS)[..])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

/// A request appending `tokens` after `cached` positions in `blocks`, keeping `keep`.
async fn request(
    device: &mut Device,
    blocks: &[u32],
    cached: usize,
    tokens: &[usize],
    keep: Keep,
) -> Request {
    let buffer = ids(device, tokens).await;
    Request {
        rows: vec![blocks.to_vec()],
        cached,
        tokens: Rows::new(&buffer, 0..tokens.len()).unwrap(),
        keep,
        adapter: None,
        positions: None,
        mask: None,
        overrides: Vec::new(),
    }
}

/// Each chip's I32 tokens of a call's kept rows.
async fn tokens(
    device: &mut Device,
    kept: furiosa_opt_engine::decoder::HbmTensor,
) -> Vec<Vec<usize>> {
    let host = kept.buffer.to_host(&mut device.pdma).await.unwrap();
    let words: Vec<usize> = (host.chunks_exact(4))
        .map(|word| i32::from_le_bytes(word.try_into().unwrap()) as usize)
        .collect();
    let rows = words.len() / CHIPS;
    (0..CHIPS)
        .map(|chip| words[chip * rows..][..rows].to_vec())
        .collect()
}

// # The published checkpoint

/// The hub repository of the published checkpoint the registry serves, and the revision its
/// reference outputs were recorded at.
const REPO: &str = "RedHatAI/Mistral-Nemo-Instruct-2407-FP8";
const REVISION: &str = "2545394475682e98b69596fde20b7fa7b43789e1";
const NEMO: Published = Published::new(REPO, REVISION);

/// The reference's oracles: vLLM over E4M3 KV records, each projection at its own scales, on one
/// GPU, and split over two GPUs as the decoder splits it over two chips, each rounding its share of
/// the row-parallel sums to BF16 before they are added.
const ORACLES: [&str; 2] = ["fp8-flashinfer-exact", "fp8-flashinfer-exact-tp2"];
/// The tokens each reference workload continues a prompt by, and the likeliest tokens it records
/// of each step.
const CONTINUATION: usize = 64;
const TOP: usize = 20;
/// The tokens A had selected when B arrived in the reference's staggered workload.
const STAGGER: usize = 8;
/// The cache blocks the reference workloads take, each generation its own.
const BLOCKS: usize = 100;
/// The rows of a latency case's prompt chunk.
const CHUNK: usize = 512;
/// Untimed steps before a latency case's timed ones, whose median it reports.
const WARMUPS: usize = 3;
const REPETITIONS: usize = 20;

/// The published config chooses the registry's NeMo implementation on two chips over E4M3
/// records, from the config alone; a load naming no KV cache, which asks for BF16 records, and one
/// naming another count of chips are refused by their names, and a config whose head width derives
/// from its hidden width chooses none.
#[test]
fn selects_published_size() -> Result<(), LoadError> {
    let checkpoint = NEMO.checkpoint();
    let named = LoadOptions::new().chips(CHIPS)?.kv_cache(Dtype::F8_E4M3);
    let chosen = named.parse(&checkpoint)?;
    assert_eq!(
        (chosen.architecture(), chosen.size(), chosen.chips()),
        ("MistralForCausalLM", "Nemo", CHIPS)
    );
    assert_eq!(
        (chosen.kv_cache(), chosen.precision()),
        (Dtype::F8_E4M3, Dtype::F8_E4M3)
    );
    let refused = |options: LoadOptions| match options.parse(&checkpoint) {
        Err(LoadError::Unsupported { name, .. }) => name,
        other => panic!("{other:?} is not unsupported"),
    };
    assert_eq!(refused(LoadOptions::new()), "kv_cache");
    assert_eq!(
        refused(LoadOptions::new().chips(1)?.kv_cache(Dtype::F8_E4M3)),
        "chips"
    );
    let mut config = NEMO.config();
    config.as_object_mut().unwrap().remove("head_dim");
    let directory = tempfile::tempdir().unwrap();
    let derived = Checkpoint::new(directory.path(), Some(config)).unwrap();
    assert!(matches!(
        named.parse(&derived),
        Err(LoadError::InvalidData { .. })
    ));
    Ok(())
}

/// The NeMo implementation refuses a config whose heads rotate by another spectrum than its step
/// holds: another base, or the base scaled.
#[test]
fn refuses_another_rotary_spectrum() -> Result<(), LoadError> {
    let (name, implementations) = Implementation::listed(&["MistralForCausalLM"])?;
    let nemo = (implementations.iter()).find(|implementation| implementation.size() == "Nemo");
    let nemo = std::slice::from_ref(nemo.expect("the registry lists the NeMo implementation"));
    let choice = Choice {
        chips: NonZeroUsize::new(CHIPS),
        kv_cache: Some(Dtype::F8_E4M3),
    };
    for (key, value) in [
        ("rope_theta", json!(500000.0)),
        (
            "rope_scaling",
            json!({"rope_type": "linear", "factor": 2.0}),
        ),
    ] {
        let mut config = NEMO.config();
        config[key] = value;
        let directory = tempfile::tempdir().unwrap();
        let edited = Checkpoint::new(directory.path(), Some(config)).unwrap();
        match Implementation::choose(name, nemo, &edited, choice) {
            Err(LoadError::Unsupported { name, .. }) => {
                assert_eq!(name, "config.json:rope_parameters", "{key}")
            }
            other => panic!("{key}: {:?}", other.map(Implementation::size)),
        }
    }
    Ok(())
}

/// The published checkpoint, loaded strictly whole onto two chips over E4M3 records through the
/// production entry and timed at the cases of its envelope, adaptively tiled
/// (`Session::measure`); then loaded batch-invariant and called as vLLM's reference workloads call
/// it, every token within the 5% rule of one of the two oracles (`Session::accepts`). Fed the same
/// history, the one-GPU and the two-GPU oracle disagree at a few steps by up to 1.6 nat, so either
/// is an admissible oracle.
#[test]
#[ignore = "opens a device"]
fn nemo_forward() -> io::Result<()> {
    let reference = Reference::open();
    let mut device = Device::new(Topology { chips: 2, pes: 8 }).map_err(io::Error::other)?;
    // Profiling attaches to functions as they load, at their first call.
    let _trace = trace();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    for tiling in [Tiling::Adaptive, Tiling::Fixed] {
        let mut session = Session::load(device, tiling)?;
        runtime.block_on(async {
            match tiling {
                Tiling::Adaptive => session.measure().await,
                Tiling::Fixed => session.accepts(&reference).await,
            }
        })?;
        device = session.device;
    }
    Ok(())
}

/// The published checkpoint adaptively tiled, the default, judged as `nemo_forward` judges the
/// batch-invariant tiling but for the bitwise equalities only that tiling holds. Open decision:
/// whether a tolerance for this tiling's batch-dependent sums is accepted; none is approved, so the
/// test runs only when asked for and fails listing each step both oracles' 5% rule refuses.
#[test]
#[ignore = "opens a device; the adaptive tiling's tolerance is an open decision"]
fn adaptive_tiling_meets_either_oracle() -> io::Result<()> {
    let reference = Reference::open();
    let device = Device::new(Topology { chips: 2, pes: 8 }).map_err(io::Error::other)?;
    let mut session = Session::load(device, Tiling::Adaptive)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(session.accepts(&reference))
}

/// The directory of the published checkpoint's configs and weights, fetched into the hub cache
/// once.
fn snapshot() -> PathBuf {
    let mut command = Command::new("uvx");
    command.args(["--from", "huggingface-hub", "hf", "download"]);
    command.args([REPO, "--revision", REVISION, "--quiet"]);
    command.args(["--include", "*.json", "--include", "*.safetensors"]);
    let output = command
        .stderr(Stdio::inherit())
        .output()
        .expect("uvx fetches the test checkpoint");
    assert!(output.status.success(), "the {REPO} download failed");
    String::from_utf8(output.stdout).unwrap().trim().into()
}

/// Records spans as a Chrome trace in `trace/<org>--<name>/forward.json` until both guards drop;
/// CI reads forward and kernel metrics from it.
fn trace() -> (
    tracing::subscriber::DefaultGuard,
    tracing_chrometrace::Trace<std::fs::File>,
) {
    use tracing_subscriber::prelude::*;
    let directory = std::path::Path::new("trace").join(REPO.to_lowercase().replace('/', "--"));
    std::fs::create_dir_all(&directory).unwrap();
    let file = std::fs::File::create(directory.join("forward.json")).unwrap();
    let (layer, trace) = tracing_chrometrace::ChromeLayer::with_writer(file);
    (
        tracing::subscriber::set_default(tracing_subscriber::registry().with(layer)),
        trace,
    )
}

/// vLLM's reference outputs of the published checkpoint, in the release its inventory row names
/// under `FURIOSA_OPT_REFERENCES`: the prompts' token IDs, and each oracle's steps of each
/// workload.
struct Reference {
    inputs: Vec<u8>,
    outputs: Vec<u8>,
}

impl Reference {
    fn open() -> Self {
        let inventory = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../.github/benchmarks.json"
        ));
        let inventory: Value = serde_json::from_str(inventory).unwrap();
        let row = (inventory["models"].as_array().unwrap().iter())
            .find(|row| row["id"] == REPO)
            .expect("the inventory lists the checkpoint");
        let references = std::env::var_os("FURIOSA_OPT_REFERENCES")
            .expect("FURIOSA_OPT_REFERENCES holds the downloaded references");
        let release = PathBuf::from(references).join(row["reference"].as_str().unwrap());
        let read = |file: &str| {
            let path = release.join(file);
            std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        };
        Self {
            inputs: read("inputs.safetensors"),
            outputs: read("outputs.safetensors"),
        }
    }

    /// The token IDs of prompt `name`.
    fn prompt(&self, name: &str) -> Vec<usize> {
        words(&self.inputs, &format!("{name}/prompt_ids"), id)
    }

    /// Each oracle's steps of workload `workload` fed the reference's history (`@ref`, a one-GPU
    /// run's continuation of each prompt), which every teacher-forced generation is fed.
    fn forced(&self, workload: &str) -> [Steps; 2] {
        let steps = ORACLES.map(|oracle| {
            let name = |field: &str| format!("{oracle}/{workload}@ref/{field}");
            Steps {
                oracle,
                tokens: words(&self.outputs, &name("tokens"), id),
                top: words(&self.outputs, &name("top_ids"), id),
                logprobs: words(&self.outputs, &name("top_logprobs"), f32::from_le_bytes),
            }
        });
        assert_eq!(
            steps[0].tokens, steps[1].tokens,
            "{workload}: the oracles' histories"
        );
        steps
    }

    /// Each oracle's tokens continuing workload `workload`'s prompt, each fed the one before.
    fn continuations(&self, workload: &str) -> [Vec<usize>; 2] {
        ORACLES.map(|oracle| words(&self.outputs, &format!("{oracle}/{workload}/tokens"), id))
    }
}

/// The 4-byte little-endian elements of tensor `name` of the safetensors file `file`.
fn words<T>(file: &[u8], name: &str, from: fn([u8; 4]) -> T) -> Vec<T> {
    let tensors = SafeTensors::deserialize(file).expect("a safetensors file");
    let tensor = (tensors.tensor(name)).unwrap_or_else(|error| panic!("{name}: {error}"));
    (tensor.data().chunks_exact(4))
        .map(|word| from(word.try_into().unwrap()))
        .collect()
}

/// The token ID a little-endian I32 holds.
fn id(word: [u8; 4]) -> usize {
    i32::from_le_bytes(word) as usize
}

/// An oracle's steps of one workload: each step's token fed, and its `TOP` likeliest tokens with
/// their log-probabilities, in rank order.
struct Steps {
    oracle: &'static str,
    tokens: Vec<usize>,
    top: Vec<usize>,
    logprobs: Vec<f32>,
}

impl Steps {
    /// The log-probability `token` lies below the likeliest token at `step`, if it is among the
    /// `TOP` likeliest.
    fn gap(&self, step: usize, token: usize) -> Option<f32> {
        let logprobs = &self.logprobs[step * TOP..][..TOP];
        let rank = self.top[step * TOP..][..TOP]
            .iter()
            .position(|&id| id == token)?;
        Some(logprobs[0] - logprobs[rank])
    }

    /// Whether `token` at `step` passes the 5% rule: within 5% of the likeliest token's
    /// probability. One past the `TOP` likeliest is refused when the last of them already fails
    /// the rule, and lacks the evidence otherwise.
    fn admits(&self, step: usize, token: usize) -> Result<(), String> {
        let tie = 0.95f32.recip().ln();
        let logprobs = &self.logprobs[step * TOP..][..TOP];
        let oracle = self.oracle;
        match self.gap(step, token) {
            Some(gap) if gap <= tie => Ok(()),
            Some(gap) => Err(format!(
                "{oracle}: {gap:.4} nat below the likeliest {}",
                self.top[step * TOP]
            )),
            None if logprobs[0] - logprobs[TOP - 1] > tie => {
                Err(format!("{oracle}: below the likeliest {TOP}"))
            }
            None => Err(format!(
                "{oracle}: past the likeliest {TOP}, which do not bound it: the evidence is missing"
            )),
        }
    }
}

/// A prompt of a workload and the tokens a call each selects after it, each fed to the next call:
/// the reference's continuation when it is given, else the selection itself; over its own blocks.
struct Generation {
    prompt: Vec<usize>,
    blocks: Vec<u32>,
    fed: Option<Vec<usize>>,
    selected: Vec<usize>,
}

impl Generation {
    /// The token at `position`: the prompt's, then the one fed.
    fn token(&self, position: usize) -> usize {
        match position.checked_sub(self.prompt.len()) {
            None => self.prompt[position],
            Some(step) => self.fed.as_ref().unwrap_or(&self.selected)[step],
        }
    }

    /// The positions a generation of a `prompt`-token prompt appends, a call each: the prompt in
    /// chunks that end at `ends`, then each continuation token fed.
    fn steps(prompt: usize, ends: &[usize]) -> Vec<Range<usize>> {
        let ends: Vec<usize> = (ends.iter().copied())
            .chain(prompt..prompt + CONTINUATION)
            .collect();
        (0..ends.len())
            .map(|at| at.checked_sub(1).map_or(0, |last| ends[last])..ends[at])
            .collect()
    }

    /// The positions the generation wrote.
    fn written(&self) -> Range<usize> {
        0..self.prompt.len() + CONTINUATION - 1
    }
}

/// The calls of generations whose steps start at the calls given: each call appends the step of
/// every generation it reaches, in the order given.
fn calls(order: &[(usize, usize, Vec<Range<usize>>)]) -> Vec<Vec<(usize, Range<usize>)>> {
    let length = (order.iter())
        .map(|(_, first, steps)| first + steps.len())
        .max()
        .unwrap_or(0);
    (0..length)
        .map(|call| {
            (order.iter())
                .filter_map(|(generation, first, steps)| {
                    let step = steps.get(call.checked_sub(*first)?)?;
                    Some((*generation, step.clone()))
                })
                .collect()
        })
        .collect()
}

/// Every layer's pool of `cache`, on the host.
async fn pools(device: &mut Device, cache: &Cache) -> io::Result<Vec<Vec<u8>>> {
    let mut pools = Vec::new();
    for pool in cache.layers() {
        let host = pool.to_host(&mut device.pdma).await;
        pools.push(host.map_err(io::Error::other)?.to_vec());
    }
    Ok(pools)
}

/// The records chip `chip` holds in `pool` of position `position` of a generation's `blocks`: its
/// KV heads' keys, then their values.
fn record<'p>(pool: &'p [u8], blocks: &[u32], position: usize, chip: usize) -> &'p [u8] {
    let record = 2 * (KV / CHIPS) * HEAD;
    let at = blocks[position / 16] as usize * 16 * record + position % 16 * record;
    &pool[chip * pool.len() / CHIPS + at..][..record]
}

/// Every layer's records of `positions` of a generation's `blocks`, on every chip.
fn records(pools: &[Vec<u8>], blocks: &[u32], positions: Range<usize>) -> Vec<u8> {
    let mut records = Vec::new();
    for pool in pools {
        for position in positions.clone() {
            for chip in 0..CHIPS {
                records.extend_from_slice(record(pool, blocks, position, chip));
            }
        }
    }
    records
}

/// The published checkpoint's decoder on its device, tiled as it was loaded.
struct Session {
    device: Device,
    model: Box<dyn Served>,
    tiling: Tiling,
}

impl Session {
    /// The published checkpoint, loaded strictly whole onto `device`'s two chips over E4M3 records
    /// through the production entry, tiled as `tiling` says.
    fn load(mut device: Device, tiling: Tiling) -> io::Result<Self> {
        let checkpoint = Checkpoint::new(snapshot(), None).map_err(io::Error::other)?;
        let options = (LoadOptions::new().chips(CHIPS))
            .map_err(io::Error::other)?
            .kv_cache(Dtype::F8_E4M3);
        let options = match tiling {
            Tiling::Adaptive => options,
            Tiling::Fixed => options.batch_invariant(),
        };
        let chosen = options.parse(&checkpoint).map_err(io::Error::other)?;
        assert_eq!((chosen.size(), chosen.tiling()), ("Nemo", tiling));
        let started = Instant::now();
        let model = (options.load(&mut device, &checkpoint))
            .map_err(io::Error::other)?
            .decoder()
            .expect("the NeMo implementation is a decoder");
        println!(
            "{tiling:?}: loaded in {:.1} s",
            started.elapsed().as_secs_f64()
        );
        Ok(Self {
            device,
            model,
            tiling,
        })
    }

    /// The calls `calls` of `generations` on `cache`: each request appends a step of a generation
    /// and keeps its last row once the generation's prompt is whole, and every chip selects the
    /// same tokens.
    async fn run(
        &mut self,
        cache: &Cache,
        generations: &mut [Generation],
        calls: &[Vec<(usize, Range<usize>)>],
    ) -> io::Result<()> {
        let started = Instant::now();
        for call in calls {
            let mut requests = Vec::with_capacity(call.len());
            for (at, positions) in call {
                let generation = &generations[*at];
                let tokens: Vec<usize> = (positions.clone())
                    .map(|position| generation.token(position))
                    .collect();
                let keep = match positions.end < generation.prompt.len() {
                    true => Keep::None,
                    false => Keep::Last,
                };
                let blocks = &generation.blocks;
                let request = request(&mut self.device, blocks, positions.start, &tokens, keep);
                requests.push(request.await);
            }
            let rows = (self.model.forward(&mut self.device, cache, &requests, &[])).await?;
            let chips = tokens(&mut self.device, rows).await;
            assert!(
                chips.iter().all(|chip| *chip == chips[0]),
                "every chip selects the same tokens"
            );
            let mut kept = chips[0].iter();
            for (at, positions) in call {
                if positions.end >= generations[*at].prompt.len() {
                    let token = *kept.next().expect("a kept row's token");
                    generations[*at].selected.push(token);
                }
            }
        }
        let seconds = started.elapsed().as_secs_f64();
        println!("{} calls in {seconds:.1} s", calls.len());
        Ok(())
    }

    /// The decoder's envelope, and vLLM's reference workloads called on it, each generation over
    /// its own blocks, judged by each oracle's 5% rule (`Steps::admits`), a token passing when
    /// either admits it: C fed its own tokens selects one oracle's 64; C, A, B and D, whose prompt
    /// is 492 rows, fed the reference's history select at every step a token an oracle's rule
    /// admits after the same history, as do A in two chunks, A and B together, and B arriving once
    /// A selected eight tokens, its requests first. Under a fixed tiling those select A's and B's
    /// tokens alone and write their records bit for bit; the split's records are A's, exactly in
    /// the first layer, and B's calls leave A's earlier records as they were. Every step one oracle
    /// alone admits and every refusal is printed before any refusal fails the test.
    async fn accepts(&mut self, reference: &Reference) -> io::Result<()> {
        let tiling = self.tiling;
        let limits = self.model.limits();
        assert_eq!(
            (limits.rows, limits.window, limits.adapters),
            (4096, 40960, 0)
        );
        let (pool, group) = (&limits.cache.pools[0], &limits.cache.groups[0]);
        let stride = 2 * (KV / CHIPS) * HEAD;
        assert_eq!((pool.block, pool.bytes), (16, 16 * stride));
        assert_eq!(group.layers, 0..40);
        let cache = Cache::new(&self.device, &limits.cache, BLOCKS)?;
        // A call past the rows, or a request past the positions, the step is compiled for is
        // refused before anything moves.
        let past = |error: io::Error| error.kind() == io::ErrorKind::InvalidInput;
        let blocks: Vec<u32> = (0..2561).map(|block| (block % BLOCKS) as u32).collect();
        let long = request(&mut self.device, &blocks[..257], 0, &[0; 4097], Keep::Last).await;
        let late = request(&mut self.device, &blocks, 40960, &[0], Keep::Last).await;
        for refused in [long, late] {
            let call = (self
                .model
                .forward(&mut self.device, &cache, &[refused], &[]))
            .await;
            assert!(call.is_err_and(past));
        }
        let mut free = (0..BLOCKS).map(|block| (block * 37 % BLOCKS) as u32);
        let mut generation = |prompt: &str, fed: Option<&str>| {
            let prompt = reference.prompt(prompt);
            let positions = prompt.len() + CONTINUATION - 1;
            Generation {
                blocks: free.by_ref().take(positions.div_ceil(16)).collect(),
                prompt,
                fed: fed.map(|workload| {
                    let [steps, _] = reference.forced(workload);
                    steps.tokens
                }),
                selected: Vec::new(),
            }
        };
        // A and B are fed one continuation each in every workload, so the selections of a
        // workload and of A or B alone have the same inputs.
        let mut alone = [
            generation("C", None),
            generation("C", Some("d.C")),
            generation("A", Some("a")),
            generation("B", Some("c_separate.B")),
            generation("D", Some("e.D")),
        ];
        let mut split = [generation("A", Some("a"))];
        let mut together = [
            generation("A", Some("a")),
            generation("B", Some("c_separate.B")),
        ];
        let mut staggered = [
            generation("A", Some("a")),
            generation("B", Some("c_separate.B")),
        ];
        let (a, b) = (alone[2].prompt.len(), alone[3].prompt.len());
        for at in 0..alone.len() {
            let steps = Generation::steps(alone[at].prompt.len(), &[]);
            self.run(&cache, &mut alone, &calls(&[(at, 0, steps)]))
                .await?;
        }
        let halves = calls(&[(0, 0, Generation::steps(a, &[32]))]);
        self.run(&cache, &mut split, &halves).await?;
        let both = calls(&[
            (0, 0, Generation::steps(a, &[])),
            (1, 0, Generation::steps(b, &[])),
        ]);
        self.run(&cache, &mut together, &both).await?;
        let arriving = calls(&[
            (1, STAGGER, Generation::steps(b, &[])),
            (0, 0, Generation::steps(a, &[])),
        ]);
        self.run(&cache, &mut staggered, &arriving[..STAGGER])
            .await?;
        let before = pools(&mut self.device, &cache).await?;
        self.run(&cache, &mut staggered, &arriving[STAGGER..])
            .await?;
        let pools = pools(&mut self.device, &cache).await?;
        let mut refusals = Vec::new();
        let own = &alone[0].selected;
        let departures = reference.continuations("d.C").map(|c| {
            (0..CONTINUATION)
                .find(|&step| own[step] != c[step])
                .map(|step| (step, c[step]))
        });
        match departures.iter().position(Option::is_none) {
            Some(at) => println!(
                "{tiling:?} d.C, each token fed the one before: {}'s {CONTINUATION} tokens",
                ORACLES[at]
            ),
            None => refusals.push(format!(
                "{tiling:?} d.C, each token fed the one before: {}",
                (ORACLES.iter().zip(departures.into_iter().flatten()))
                    .map(|(oracle, (step, token))| format!(
                        "step {step}: {}, {oracle}'s {token}",
                        own[step]
                    ))
                    .collect::<Vec<_>>()
                    .join("; ")
            )),
        }
        let forced = [
            ("d.C", &alone[1]),
            ("a", &alone[2]),
            ("c_separate.B", &alone[3]),
            ("e.D", &alone[4]),
            ("b_split", &split[0]),
            ("c_batch.A", &together[0]),
            ("c_batch.B", &together[1]),
            ("c_staggered.A", &staggered[0]),
            ("c_staggered.B", &staggered[1]),
        ];
        let mut outside = 0;
        for (workload, generation) in forced {
            let [one, two] = &reference.forced(workload);
            let mut refused = Vec::new();
            for (step, &token) in generation.selected.iter().enumerate() {
                let admitted = |by: &Steps, refusal: String| {
                    println!(
                        "{tiling:?} {workload}: step {step}: {token}, admitted by {} alone; \
                         {refusal}",
                        by.oracle
                    )
                };
                match (one.admits(step, token), two.admits(step, token)) {
                    (Ok(()), Ok(())) => {}
                    (Ok(()), Err(refusal)) => admitted(one, refusal),
                    (Err(refusal), Ok(())) => admitted(two, refusal),
                    (Err(first), Err(second)) => {
                        refused.push(format!("step {step}: {token}, {first}; {second}"))
                    }
                }
            }
            println!(
                "{tiling:?} {workload}: {} of {CONTINUATION} steps within the 5% rule of an oracle",
                CONTINUATION - refused.len()
            );
            outside += refused.len();
            refusals.extend(
                (refused.iter()).map(|refusal| format!("{tiling:?} {workload}: {refusal}")),
            );
        }
        for (workload, generation, alone) in [
            ("b_split", &split[0], &alone[2]),
            ("c_batch.A", &together[0], &alone[2]),
            ("c_batch.B", &together[1], &alone[3]),
            ("c_staggered.A", &staggered[0], &alone[2]),
            ("c_staggered.B", &staggered[1], &alone[3]),
        ] {
            let differ: Vec<usize> = (0..CONTINUATION)
                .filter(|&step| generation.selected[step] != alone.selected[step])
                .collect();
            println!("{tiling:?} {workload}: selects another token than alone at {differ:?}");
            if let (Tiling::Fixed, Some(step)) = (tiling, differ.first()) {
                refusals.push(format!(
                    "{tiling:?} {workload}: step {step}: {}, alone {}",
                    generation.selected[*step], alone.selected[*step]
                ));
            }
            let (found, want) = (
                records(&pools, &generation.blocks, generation.written()),
                records(&pools, &alone.blocks, alone.written()),
            );
            if tiling == Tiling::Fixed && found != want {
                refusals.push(format!("{tiling:?} {workload}: records other than alone"));
            }
        }
        for (layer, pool) in pools.iter().enumerate() {
            let mut distance = Distance::default();
            for position in alone[2].written() {
                for chip in 0..CHIPS {
                    let whole = record(pool, &alone[2].blocks, position, chip);
                    let halves = record(pool, &split[0].blocks, position, chip);
                    for at in 0..whole.len() {
                        let value = |stored| <f8e4m3 as Scalar>::load(stored, at).to_f32();
                        distance.add(value(halves), value(whole));
                    }
                }
            }
            let found = format!(
                "{tiling:?} layer {layer}: {} of {} split records A's, off by {:.4}",
                distance.exact,
                distance.total,
                distance.relative()
            );
            println!("{found}");
            if (layer == 0 && distance.exact != distance.total) || distance.relative() > STEP {
                refusals.push(found);
            }
        }
        let written = 0..a + STAGGER - 1;
        if records(&pools, &staggered[0].blocks, written.clone())
            != records(&before, &staggered[0].blocks, written)
        {
            refusals.push(format!(
                "{tiling:?} c_staggered: B's calls change the records A wrote"
            ));
        }
        let judged = forced.len() * CONTINUATION;
        let summary = format!(
            "{tiling:?}: {outside} of {judged} teacher-forced steps outside the 5% rule of both \
             oracles"
        );
        println!("{summary}");
        assert!(
            refusals.is_empty(),
            "{summary}; {} refusals:\n{}",
            refusals.len(),
            refusals.join("\n")
        );
        Ok(())
    }

    /// The time of a step of each case of the decoder's envelope: a 1024-token prompt as two
    /// 512-row chunks, and one decode row after it, alone and as 32 rows; each case's steps
    /// repeated, and a decode row at the last position the step is compiled for selecting the same
    /// token on every chip.
    async fn measure(&mut self) -> io::Result<()> {
        let limits = self.model.limits();
        let (rows, window) = (CHUNK, limits.window);
        let (prompt, decodes) = (2 * rows, 32);
        let cache = Cache::new(&self.device, &limits.cache, window / 16 + decodes)?;
        let shared: Vec<u32> = (0..window / 16).map(|block| block as u32).collect();
        let zeros = vec![0; prompt];
        let mut cases = Vec::new();
        for (case, cached) in [
            ("in-512-prefill-chunk-512", 0),
            ("in-1024-prefill-chunk-512", rows),
        ] {
            let chunk = request(
                &mut self.device,
                &shared,
                cached,
                &zeros[..rows],
                Keep::Last,
            );
            cases.push((case, vec![chunk.await]));
        }
        for (case, count) in [
            ("in-1024-decode-rows-1", 1),
            ("in-1024-decode-rows-32", decodes),
        ] {
            let mut requests = Vec::with_capacity(count);
            for row in 0..count {
                let own = [&shared[..prompt / 16], &[(window / 16 + row) as u32]].concat();
                requests.push(request(&mut self.device, &own, prompt, &[0], Keep::Last).await);
            }
            cases.push((case, requests));
        }
        for (case, requests) in &cases {
            let [median, least, most] = self.time(&cache, case, requests).await?;
            println!(
                "{case}: {:.3} ms a step (least {:.3}, most {:.3}) over {REPETITIONS} steps",
                median.as_secs_f64() * 1e3,
                least.as_secs_f64() * 1e3,
                most.as_secs_f64() * 1e3
            );
        }
        let last = request(&mut self.device, &shared, window - 1, &[0], Keep::Last).await;
        let rows = (self.model.forward(&mut self.device, &cache, &[last], &[])).await?;
        let chips = tokens(&mut self.device, rows).await;
        assert!(
            chips
                .iter()
                .all(|chip| chip.len() == 1 && *chip == chips[0])
        );
        Ok(())
    }

    /// The median, least and most time of a step of the call `requests` on `cache`, from its call
    /// until the device is done, over `REPETITIONS` steps after `WARMUPS`; each timed step is a
    /// `forward` span of `case`.
    async fn time(
        &mut self,
        cache: &Cache,
        case: &str,
        requests: &[Request],
    ) -> io::Result<[Duration; 3]> {
        use tracing::Instrument;
        let mut steps = Vec::with_capacity(REPETITIONS);
        for sample in 0..WARMUPS + REPETITIONS {
            let span = match sample.checked_sub(WARMUPS) {
                Some(step) => tracing::info_span!("forward", case, step = step + 1),
                None => tracing::Span::none(),
            };
            let started = Instant::now();
            async {
                let _rows = (self.model.forward(&mut self.device, cache, requests, &[])).await?;
                self.device.synchronize().await.map_err(io::Error::other)
            }
            .instrument(span)
            .await?;
            if sample >= WARMUPS {
                steps.push(started.elapsed());
            }
        }
        steps.sort_unstable();
        Ok([
            steps[(REPETITIONS - 1) / 2],
            steps[0],
            steps[REPETITIONS - 1],
        ])
    }
}
