//! Qwen3 in BF16 through the engine's Rust API: the model each published checkpoint selects, loaded
//! and decoded, a request a call and several in one; a dropped model's memory freed and a new load
//! serving its cache; each size's smaller root and a layer loaded alone; each reranker's score and
//! each embedding checkpoint's last row; a batch-invariant load's rows; and the choices refused
//! before any weight is read.

#![feature(adt_const_params, unsized_const_params)]

use std::future::IntoFuture;
use std::ops::Range;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use furiosa_opt_engine::decoder::{Cache, HbmTensor, Instance, Keep, Request, Rows, Served};
use furiosa_opt_engine::{Checkpoint, Hbm, Load, LoadError, LoadOptions, Tiling};
use furiosa_opt_models::load::Context;
use furiosa_opt_models::models::decoder::attention::{Geometry, qwen3};
use furiosa_opt_models::models::decoder::form;
use furiosa_opt_models::tensor::cache;
use furiosa_opt_models::tensor::parallel::Parallel;
use furiosa_opt_std::prelude::{Device, HostBufferSlice, Symbol, Topology, bf16, m};
use safetensors::Dtype;

/// "Instruct: Given a web search query, retrieve relevant passages that answer the query
/// Query:What is the capital of China?" in Qwen3's tokens.
const QWEN: [i32; 25] = [
    641, 1235, 25, 16246, 264, 3482, 2711, 3239, 11, 17179, 9760, 46769, 429, 4226, 279, 3239,
    11361, 25, 3838, 374, 279, 6722, 315, 5616, 30,
];

/// A published Qwen3 checkpoint at the hub revision these tests read, and its greedy continuation
/// of `QWEN` in BF16: Transformers' `generate(max_new_tokens=8, do_sample=False)` on the CPU of
/// `AutoModelForCausalLM.from_pretrained(.., dtype=torch.bfloat16)`.
struct Published {
    repo: &'static str,
    revision: &'static str,
    continuation: [i32; 8],
}

const QWEN3_0_6B: Published = Published {
    repo: "Qwen/Qwen3-0.6B",
    revision: "c1899de289a04d12100db370d81485cdf75e47ca",
    continuation: [576, 3482, 2711, 3239, 374, 330, 22282, 594],
};

const QWEN3_1_7B: Published = Published {
    repo: "Qwen/Qwen3-1.7B",
    revision: "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    continuation: [576, 4226, 1265, 387, 304, 279, 1352, 315],
};

const QWEN3_4B: Published = Published {
    repo: "Qwen/Qwen3-4B",
    revision: "1cfa9a7208912126459214e8b04321603b3df60c",
    continuation: [4710, 12187, 1134, 25, 715, 16, 13, 576],
};

/// A 4B checkpoint of another training, with a longer declared context and another `rope_theta`.
const QWEN3_4B_INSTRUCT: Published = Published {
    repo: "Qwen/Qwen3-4B-Instruct-2507",
    revision: "cdbee75f17c01a7cc42f958dc650907174af0554",
    continuation: [4710, 12187, 424, 220, 16, 25, 576, 6722],
};

const QWEN3_4B_THINKING: Published = Published {
    repo: "Qwen/Qwen3-4B-Thinking-2507",
    revision: "768f209d9ea81521153ed38c47d515654e938aea",
    continuation: [4710, 1654, 525, 2661, 264, 3239, 25, 330],
};

const QWEN3_8B: Published = Published {
    repo: "Qwen/Qwen3-8B",
    revision: "b968826d9c46dd6066d109eabc6255188de91218",
    continuation: [98475, 220, 16, 25, 5616, 374, 264, 3146],
};

/// `QWEN`, then `<|endoftext|>`, whose last hidden row Qwen3's embedding checkpoints embed with.
const EMBEDDED: [i32; 26] = [
    641, 1235, 25, 16246, 264, 3482, 2711, 3239, 11, 17179, 9760, 46769, 429, 4226, 279, 3239,
    11361, 25, 3838, 374, 279, 6722, 315, 5616, 30, 151643,
];

/// A published Qwen3 embedding checkpoint at the hub revision these tests read, and the eight
/// largest components of its normalized embedding of `EMBEDDED`, by feature: the last row of
/// Transformers' `AutoModel.from_pretrained(.., dtype=torch.bfloat16)` `last_hidden_state` on the
/// CPU.
struct Embedder {
    repo: &'static str,
    revision: &'static str,
    components: [(usize, f32); 8],
}

const QWEN3_EMBEDDING_0_6B: Embedder = Embedder {
    repo: "Qwen/Qwen3-Embedding-0.6B",
    revision: "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    components: [
        (17, 0.0955),
        (26, -0.0955),
        (30, 0.0955),
        (370, -0.1072),
        (487, 0.0912),
        (702, -0.1034),
        (758, 0.0886),
        (972, 0.0912),
    ],
};

const QWEN3_EMBEDDING_4B: Embedder = Embedder {
    repo: "Qwen/Qwen3-Embedding-4B",
    revision: "5cf2132abc99cad020ac570b19d031efec650f2b",
    components: [
        (17, -0.06),
        (24, -0.0772),
        (291, 0.0581),
        (1172, 0.0646),
        (1182, 0.0565),
        (2174, -0.0614),
        (2279, 0.0586),
        (2500, -0.0579),
    ],
};

const QWEN3_EMBEDDING_8B: Embedder = Embedder {
    repo: "Qwen/Qwen3-Embedding-8B",
    revision: "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    components: [
        (14, 0.0656),
        (1348, 0.0568),
        (2242, -0.0642),
        (2469, -0.091),
        (3564, 0.0563),
        (3691, -0.0618),
        (3863, -0.0781),
        (3878, -0.0714),
    ],
};

/// The request's history blocks, out of the seven the cache holds, in history order: 96 positions.
const BLOCKS: [u32; 6] = [2, 0, 5, 3, 6, 1];

/// The blocks the cache holds.
const CACHED: usize = 7;

/// Times `QWEN` repeats in a prompt past the 40960 positions Qwen3's config declares.
const REPEATS: usize = 1660;

/// Transformers 5.14.1's greedy BF16 continuation of `QWEN` x `REPEATS` under Qwen's YaRN recipe
/// (factor 4 over 32768 positions, 131072 declared), each token >= 0.99 of its step's probability.
const EXTENDED: [i32; 8] = [641, 1235, 25, 16246, 264, 3482, 2711, 3239];

/// The BF16 Qwen3 layer of the size `N` over the memory `X`.
type Layer<const N: &'static str, X> =
    qwen3::DecoderLayer<qwen3::Size<N>, form::Affine, Parallel<1>, cache::Bf16, X>;

/// The BF16 Qwen3 causal model of the size `N`, on one chip.
type Causal<const N: &'static str> =
    qwen3::ForCausalLM<qwen3::Size<N>, form::Affine, Parallel<1>, cache::Bf16, Hbm>;

/// The directory of the files of hub checkpoint `repo` at `revision` that `include` matches,
/// fetched into the hub cache once.
fn snapshot((repo, revision): (&str, &str), include: &[&str]) -> PathBuf {
    let mut command = Command::new("uvx");
    command.args(["--from", "huggingface-hub", "hf", "download"]);
    command.args([repo, "--revision", revision, "--quiet"]);
    for pattern in include {
        command.args(["--include", pattern]);
    }
    let output = command
        .stderr(Stdio::inherit())
        .output()
        .expect("uvx fetches the test checkpoint");
    assert!(output.status.success(), "the {repo} download failed");
    String::from_utf8(output.stdout).unwrap().trim().into()
}

/// The checkpoint of hub `repo` at `revision`, weights included.
fn checkpoint(hub: (&str, &str)) -> Checkpoint {
    Checkpoint::new(snapshot(hub, &["*.json", "*.safetensors"]), None).unwrap()
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn device() -> Device {
    Device::new(Topology { chips: 1, pes: 8 }).unwrap()
}

/// The decoder `checkpoint` selects, loaded onto `device` by the options' selection.
fn causal(checkpoint: &Checkpoint, device: &mut Device) -> Box<dyn Served> {
    LoadOptions::new()
        .load(device, checkpoint)
        .unwrap()
        .decoder()
        .unwrap()
}

/// One request a call: `tokens` appended after the `cached` positions of the history in `blocks`,
/// keeping `keep`.
async fn request(
    device: &mut Device,
    blocks: &[u32],
    (cached, tokens): (usize, &[i32]),
    keep: Keep,
) -> Request {
    let bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect();
    let ids = HostBufferSlice::from(&bytes[..])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    Request {
        rows: vec![blocks.to_vec()],
        cached,
        tokens: Rows::new(&ids, 0..tokens.len()).unwrap(),
        keep,
        adapter: None,
        positions: None,
        mask: None,
        overrides: Vec::new(),
    }
}

/// The I32 token of a one-row output, read back once its call completed.
async fn token(device: &mut Device, tokens: HbmTensor) -> i32 {
    let mut host = [0; 4];
    tokens
        .buffer
        .to_host(&mut device.pdma)
        .output(&mut host[..])
        .await
        .unwrap();
    i32::from_le_bytes(host)
}

/// The values of a call's kept BF16 rows, read back once the call completed, a vector a row.
async fn hidden(device: &mut Device, rows: HbmTensor) -> Vec<Vec<f32>> {
    assert_eq!(rows.dtype, Dtype::BF16);
    let width = rows.shape[1];
    let mut host = vec![0; rows.buffer.size()];
    rows.buffer
        .to_host(&mut device.pdma)
        .output(&mut host[..])
        .await
        .unwrap();
    let values: Vec<f32> = (host.as_chunks::<2>().0.iter())
        .map(|&value| bf16::from_bits(u16::from_le_bytes(value)).to_f32())
        .collect();
    values.chunks(width).map(<[f32]>::to_vec).collect()
}

/// The token `model` selects at each of `positions`, one position a call on `cache`: a prompt
/// position appends the prompt's token, a later one the token selected before it.
async fn greedy(
    device: &mut Device,
    model: &dyn Served,
    cache: &Cache,
    positions: Range<usize>,
) -> Vec<i32> {
    assert!(
        positions.start < QWEN.len(),
        "decoding starts within the prompt"
    );
    let mut selected = Vec::new();
    for position in positions {
        let input = QWEN.get(position).or(selected.last()).copied().unwrap();
        let request = request(device, &BLOCKS, (position, &[input]), Keep::Last).await;
        let tokens = model
            .forward(device, cache, std::slice::from_ref(&request), &[])
            .await
            .unwrap();
        selected.push(token(device, tokens).await);
    }
    selected
}

// # The model a checkpoint selects

/// Each published checkpoint, loaded by the options' selection and decoded one position a call
/// through a scattered block table, continues the prompt with Transformers' greedy tokens.
macro_rules! continues {
    ($($test:ident: $published:ident;)*) => {$(
        #[test]
        #[ignore = "opens a device"]
        fn $test() {
            let checkpoint = checkpoint(($published.repo, $published.revision));
            let mut device = device();
            let model = LoadOptions::new().load(&mut device, &checkpoint).unwrap().decoder().unwrap();
            runtime().block_on(async {
                let cache = Cache::new(&mut device, &model.limits().cache, CACHED).unwrap();
                let (mut selected, mut continued) = (0, Vec::new());
                for position in 0..QWEN.len() + $published.continuation.len() - 1 {
                    let input = QWEN.get(position).copied().unwrap_or(selected);
                    let request =
                        request(&mut device, &BLOCKS, (position, &[input]), Keep::Last).await;
                    let tokens = model
                        .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                        .await
                        .unwrap();
                    selected = token(&mut device, tokens).await;
                    if position + 1 >= QWEN.len() {
                        continued.push(selected);
                    }
                }
                assert_eq!(continued, $published.continuation);
            });
        }
    )*};
}

continues! {
    qwen3_0_6b_continues_the_prompt: QWEN3_0_6B;
    qwen3_1_7b_continues_the_prompt: QWEN3_1_7B;
    qwen3_4b_continues_the_prompt: QWEN3_4B;
    qwen3_4b_instruct_continues_the_prompt: QWEN3_4B_INSTRUCT;
    qwen3_4b_thinking_continues_the_prompt: QWEN3_4B_THINKING;
    qwen3_8b_continues_the_prompt: QWEN3_8B;
}

/// A call of two requests, each appending rows after a history of its own, returns the body rows
/// each request gives fed one row a call, and writes the cache as those calls do: each request's
/// next row agrees too.
#[test]
#[ignore = "opens a device"]
fn a_call_of_requests_returns_each_alone() {
    let checkpoint = checkpoint((QWEN3_0_6B.repo, QWEN3_0_6B.revision));
    let mut device = device();
    let causal: Causal<"0.6B"> = Load::load(&mut Context::new(&mut device), &checkpoint).unwrap();
    let body = Instance::new(causal.model, Tiling::Adaptive);
    let body: &dyn Served = &body;
    // Each request: its sequence, the positions cached before the call, the rows it appends, those
    // it keeps and its blocks; the first appends across a block, the second's history crosses one.
    let requests: [(&[i32], usize, usize, Range<usize>, [u32; 2]); 2] = [
        (&QWEN[..20], 14, 5, 0..4, [3, 1]),
        (&QWEN[2..], 17, 5, 0..5, [0, 2]),
    ];
    // The paths differ only in the order of their sums, less than this device differs from
    // Transformers, whose final hidden rows the embedding contract test holds to a cosine of 0.999.
    let agreement = |got: &[f32], want: &[f32]| {
        let dot = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum()
        };
        let error: f64 = (got.iter().zip(want))
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum();
        let finite = got.iter().chain(want).all(|value| value.is_finite());
        let cosine = dot(got, want) / (dot(got, got) * dot(want, want)).sqrt();
        let relative = (error / dot(want, want)).sqrt();
        // The same budget in length: a difference within the chord of the bound's angle.
        let agrees = finite && cosine >= 0.999 && relative <= (2.0 * (1.0 - 0.999_f64)).sqrt();
        (agrees, cosine, relative)
    };
    let agree = |got: &[f32], want: &[f32]| {
        let (agrees, cosine, relative) = agreement(got, want);
        assert!(agrees, "cosine {cosine}, relative L2 {relative}");
    };
    runtime().block_on(async {
        // Every row of each sequence, fed one row a call.
        let cache = Cache::new(&mut device, &body.limits().cache, CACHED).unwrap();
        let mut alone = Vec::new();
        for (tokens, .., blocks) in &requests {
            let mut rows = Vec::new();
            for position in 0..tokens.len() {
                let token = &tokens[position..=position];
                let request = request(&mut device, blocks, (position, token), Keep::Last).await;
                let row = body
                    .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                    .await
                    .unwrap();
                rows.extend(hidden(&mut device, row).await);
            }
            alone.push(rows);
        }
        let row = &alone[0][0];
        let doubled: Vec<f32> = row.iter().map(|value| 2.0 * value).collect();
        assert!(agreement(row, row).0 && !agreement(&doubled, row).0);
        // The same histories, then one call of both requests' rows, then each one's next row.
        let cache = Cache::new(&mut device, &body.limits().cache, CACHED).unwrap();
        let mut call = Vec::new();
        for (tokens, cached, appended, kept, blocks) in &requests {
            for position in 0..*cached {
                let token = &tokens[position..=position];
                let request = request(&mut device, blocks, (position, token), Keep::Last).await;
                body.forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                    .await
                    .unwrap();
            }
            let rows = &tokens[*cached..cached + appended];
            let keep = Keep::Range(kept.start, kept.end);
            call.push(request(&mut device, blocks, (*cached, rows), keep).await);
        }
        let rows = body.forward(&mut device, &cache, &call, &[]).await.unwrap();
        let packed = hidden(&mut device, rows).await;
        let want: Vec<&Vec<f32>> = (requests.iter().zip(&alone))
            .flat_map(|((_, cached, _, kept, _), rows)| {
                &rows[cached + kept.start..cached + kept.end]
            })
            .collect();
        assert_eq!(packed.len(), want.len());
        for (got, want) in packed.iter().zip(want) {
            agree(got, want);
        }
        for ((tokens, cached, appended, _, blocks), rows) in requests.iter().zip(&alone) {
            let next = cached + appended;
            let token = &tokens[next..=next];
            let request = request(&mut device, blocks, (next, token), Keep::Last).await;
            let row = body
                .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                .await
                .unwrap();
            agree(&hidden(&mut device, row).await[0], &rows[next]);
        }
    });
}

/// Qwen3-0.6B under Qwen's YaRN recipe, overridden as a framework passes it, continues a prompt
/// past the 40960 positions its own config declares with Transformers' greedy tokens.
#[test]
#[ignore = "opens a device"]
fn qwen3_0_6b_continues_past_its_trained_context() {
    let directory = snapshot(
        (QWEN3_0_6B.repo, QWEN3_0_6B.revision),
        &["*.json", "*.safetensors"],
    );
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json")).unwrap()).unwrap();
    config["rope_scaling"] = serde_json::json!({
        "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768,
    });
    config["max_position_embeddings"] = 131072.into();
    let checkpoint = Checkpoint::new(directory, Some(config)).unwrap();
    let mut device = device();
    let model = causal(&checkpoint, &mut device);
    let prompt = QWEN.repeat(REPEATS);
    assert!(prompt.len() > 40960);
    let rows = model.limits().rows;
    runtime().block_on(async {
        let blocks = (prompt.len() + EXTENDED.len()).div_ceil(16);
        let cache = Cache::new(&mut device, &model.limits().cache, blocks).unwrap();
        let table: Vec<u32> = (0..blocks as u32).collect();
        let calls = prompt.chunks(rows).enumerate();
        let mut selected = Vec::new();
        for (call, tokens) in calls {
            let cached = call * rows;
            let last = cached + tokens.len() == prompt.len();
            let keep = if last { Keep::Last } else { Keep::None };
            let request = request(&mut device, &table, (cached, tokens), keep).await;
            let kept = model
                .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                .await
                .unwrap();
            if last {
                selected.push(token(&mut device, kept).await);
            }
        }
        while selected.len() < EXTENDED.len() {
            let position = prompt.len() + selected.len() - 1;
            let input = [selected[selected.len() - 1]];
            let request = request(&mut device, &table, (position, &input), Keep::Last).await;
            let kept = model
                .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                .await
                .unwrap();
            selected.push(token(&mut device, kept).await);
        }
        assert_eq!(selected, EXTENDED);
    });
}

// # Freeing and loading again

/// Dropping a model that served a call frees every byte of device memory its load took: freeing
/// the weights is a drop.
#[test]
#[ignore = "opens a device"]
fn dropping_a_model_frees_its_load() {
    let checkpoint = checkpoint((QWEN3_0_6B.repo, QWEN3_0_6B.revision));
    let mut device = device();
    let free = device.memory().usage().available;
    let model = causal(&checkpoint, &mut device);
    let loaded = free - device.memory().usage().available;
    assert!(loaded > 0, "the weights take device memory");
    runtime().block_on(async {
        let cache = Cache::new(&mut device, &model.limits().cache, CACHED).unwrap();
        greedy(&mut device, &*model, &cache, 0..1).await;
        device.synchronize().await.unwrap();
        let serving = device.memory().usage().available;
        drop(model);
        assert_eq!(device.memory().usage().available - serving, loaded);
    });
}

/// A new load from the checkpoint continues the prompt on the cache a dropped load filled, with
/// Transformers' greedy tokens: the weights come back by a new load, and the cache serves it.
#[test]
#[ignore = "opens a device"]
fn a_new_load_continues_on_the_cache_of_a_dropped_one() {
    let checkpoint = checkpoint((QWEN3_0_6B.repo, QWEN3_0_6B.revision));
    let mut device = device();
    runtime().block_on(async {
        let model = causal(&checkpoint, &mut device);
        let cache = Cache::new(&mut device, &model.limits().cache, CACHED).unwrap();
        let prompt = QWEN.len() - 1;
        greedy(&mut device, &*model, &cache, 0..prompt).await;
        drop(model);
        let model = causal(&checkpoint, &mut device);
        let positions = prompt..prompt + QWEN3_0_6B.continuation.len();
        assert_eq!(
            greedy(&mut device, &*model, &cache, positions).await,
            QWEN3_0_6B.continuation
        );
    });
}

// # A smaller root and a subtree

/// The base model of each size, selected by name on a causal checkpoint, reads its names under
/// `model` and returns the rows the causal model's body computes, byte for byte.
macro_rules! body_rows {
    ($($test:ident: $published:ident selects $size:literal of $hidden:literal;)*) => {$(
        #[test]
        #[ignore = "opens a device"]
        fn $test() {
            let checkpoint = checkpoint(($published.repo, $published.revision));
            let mut device = device();
            let options = LoadOptions::new().architecture("Qwen3Model").unwrap();
            let base = options.load(&mut device, &checkpoint).unwrap().decoder().unwrap();
            let causal: Causal<$size> =
                Load::load(&mut Context::new(&mut device), &checkpoint).unwrap();
            let body = Instance::new(causal.model, Tiling::Adaptive);
            runtime().block_on(async {
                let mut hidden = Vec::new();
                let bodies: [&dyn Served; 2] = [&*base, &body];
                for body in bodies {
                    let cache = Cache::new(&mut device, &body.limits().cache, CACHED).unwrap();
                    let request = request(&mut device, &BLOCKS, (0, &QWEN[..1]), Keep::Last).await;
                    let rows = body
                        .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                        .await
                        .unwrap();
                    let mut host = vec![0; rows.buffer.size()];
                    rows.buffer
                        .to_host(&mut device.pdma)
                        .output(&mut host[..])
                        .await
                        .unwrap();
                    hidden.push(host);
                }
                assert_eq!(hidden[0].len(), $hidden * size_of::<bf16>());
                assert_eq!(hidden[0], hidden[1]);
            });
        }
    )*};
}

body_rows! {
    qwen3_model_returns_the_body_rows: QWEN3_0_6B selects "0.6B" of 1024;
    qwen3_1_7b_model_returns_the_body_rows: QWEN3_1_7B selects "1.7B" of 2048;
    qwen3_4b_model_returns_the_body_rows: QWEN3_4B selects "4B" of 2560;
    qwen3_8b_model_returns_the_body_rows: QWEN3_8B selects "8B" of 4096;
}

/// A layer loads alone from the checkpoint under its own names, as a streamed layer does: strict
/// within `model.layers.5`, with the checkpoint's values.
#[test]
#[ignore = "opens a device"]
fn a_layer_loads_alone() {
    let checkpoint = checkpoint((QWEN3_0_6B.repo, QWEN3_0_6B.revision));
    let mut device = device();
    let mut context = Context::new(&mut device);
    let layer = context
        .field("layer", "model.layers.5", |cx| {
            <Layer<"0.6B", Hbm> as Load<_>>::load(cx, &checkpoint)
        })
        .unwrap();
    drop(context);
    // The layer's first child is its pre-norm residual, whose norm holds `input_layernorm.weight`.
    type Width = <qwen3::Size<"0.6B"> as Geometry>::Hidden;

    let norm = layer.0.0.0.weight().tensor();
    let values = runtime()
        .block_on(norm.to_host::<m![Width]>(&mut device.pdma).into_future())
        .unwrap()
        .into_vec();
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect();
    let stored = checkpoint
        .get("model.layers.5.input_layernorm.weight")
        .unwrap();
    assert_eq!(bytes, stored.data());
}

/// Qwen3-Reranker's chat template of the query "What is the capital of China?" and the document
/// "The capital of China is Beijing.", in Qwen3's tokens.
const RERANKED: [i32; 85] = [
    151644, 8948, 198, 60256, 3425, 279, 11789, 20027, 279, 8502, 3118, 389, 279, 11361, 323, 279,
    758, 1235, 3897, 13, 7036, 429, 279, 4226, 646, 1172, 387, 330, 9693, 1, 476, 330, 2152, 3263,
    151645, 198, 151644, 872, 198, 27, 641, 1235, 26818, 16246, 264, 3482, 2711, 3239, 11, 17179,
    9760, 46769, 429, 4226, 279, 3239, 198, 27, 2859, 26818, 3555, 374, 279, 6722, 315, 5616, 5267,
    75692, 26818, 576, 6722, 315, 5616, 374, 26549, 13, 151645, 198, 151644, 77091, 198, 151667,
    271, 151668, 271,
];

/// A published Qwen3 sequence classifier at the hub revision these tests read, whether it is an
/// original reranker, which vLLM serves as a classifier of its `no` and `yes` tokens, and the
/// score of its last row of `RERANKED` in BF16 on the CPU: Transformers' causal logit of `yes`
/// less that of `no` for a reranker, else its sequence classifier's logit.
struct Scorer {
    repo: &'static str,
    revision: &'static str,
    reranker: bool,
    score: f32,
}

const QWEN3_RERANKER_0_6B: Scorer = Scorer {
    repo: "Qwen/Qwen3-Reranker-0.6B",
    revision: "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    reranker: true,
    score: 7.625,
};

const QWEN3_RERANKER_4B: Scorer = Scorer {
    repo: "Qwen/Qwen3-Reranker-4B",
    revision: "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
    reranker: true,
    score: 6.25,
};

const QWEN3_RERANKER_8B: Scorer = Scorer {
    repo: "Qwen/Qwen3-Reranker-8B",
    revision: "77d193c791ed757ca307ee72715aa132723da912",
    reranker: true,
    score: 5.1875,
};

/// The 0.6B reranker converted to a score head, its weights stored in FP32.
const QWEN3_RERANKER_0_6B_SEQ_CLS: Scorer = Scorer {
    repo: "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
    revision: "6a5829f5079c66e78d911e06fe21931cc00232f7",
    reranker: false,
    score: 7.6562,
};

/// The overrides vLLM serves an original Qwen3 reranker with.
const RERANKER: &str = r#"{"architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}"#;

// # Scores

/// Each published classifier, under the config vLLM serves it with, loads as the sequence
/// classifier its config selects; the score of its last row of `RERANKED` is Transformers'.
macro_rules! scores {
    ($($test:ident: $scorer:ident;)*) => {$(
        #[test]
        #[ignore = "opens a device"]
        fn $test() {
            let directory = snapshot(($scorer.repo, $scorer.revision), &["*.json", "*.safetensors"]);
            let mut config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(directory.join("config.json")).unwrap())
                    .unwrap();
            if $scorer.reranker {
                let overrides: serde_json::Value = serde_json::from_str(RERANKER).unwrap();
                for (key, value) in overrides.as_object().unwrap() {
                    config[key] = value.clone();
                }
            }
            let checkpoint = Checkpoint::new(directory, Some(config)).unwrap();
            let mut device = device();
            let model = LoadOptions::new().load(&mut device, &checkpoint).unwrap().decoder().unwrap();
            let score = runtime().block_on(async {
                let cache = Cache::new(&mut device, &model.limits().cache, CACHED).unwrap();
                let mut last = [0; 4];
                for (position, &token) in RERANKED.iter().enumerate() {
                    let request =
                        request(&mut device, &BLOCKS, (position, &[token]), Keep::Last).await;
                    let scores = model
                        .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                        .await
                        .unwrap();
                    let mut row = vec![0; scores.buffer.size()];
                    scores
                        .buffer
                        .to_host(&mut device.pdma)
                        .output(&mut row[..])
                        .await
                        .unwrap();
                    last.copy_from_slice(&row[..4]);
                }
                f32::from_le_bytes(last)
            });
            let expected = $scorer.score;
            assert!((score - expected).abs() < 0.25, "{score} vs {expected}");
        }
    )*};
}

scores! {
    qwen3_reranker_0_6b_scores_the_document: QWEN3_RERANKER_0_6B;
    qwen3_reranker_4b_scores_the_document: QWEN3_RERANKER_4B;
    qwen3_reranker_8b_scores_the_document: QWEN3_RERANKER_8B;
    qwen3_reranker_seq_cls_scores_the_document: QWEN3_RERANKER_0_6B_SEQ_CLS;
}

// # Embeddings

/// Each published embedding checkpoint, stored without the causal class's `model` scope, loads as
/// the base model a pooling server selects; its last row of `EMBEDDED`, normalized, has
/// Transformers' largest components.
macro_rules! embeds {
    ($($test:ident: $embedder:ident;)*) => {$(
        #[test]
        #[ignore = "opens a device"]
        fn $test() {
            let checkpoint = checkpoint(($embedder.repo, $embedder.revision));
            let mut device = device();
            let options = LoadOptions::new().architecture("Qwen3Model").unwrap();
            let model = options.load(&mut device, &checkpoint).unwrap().decoder().unwrap();
            let last = runtime().block_on(async {
                let cache = Cache::new(&mut device, &model.limits().cache, CACHED).unwrap();
                let mut last = Vec::new();
                for (position, &token) in EMBEDDED.iter().enumerate() {
                    let request =
                        request(&mut device, &BLOCKS, (position, &[token]), Keep::Last).await;
                    let rows = model
                        .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                        .await
                        .unwrap();
                    last = vec![0; rows.buffer.size()];
                    rows.buffer
                        .to_host(&mut device.pdma)
                        .output(&mut last[..])
                        .await
                        .unwrap();
                }
                last
            });
            let hidden: Vec<f32> = last
                .chunks_exact(2)
                .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
                .collect();
            let norm = hidden.iter().map(|value| value * value).sum::<f32>().sqrt();
            for (feature, expected) in $embedder.components {
                let actual = hidden[feature] / norm;
                assert!((actual - expected).abs() < 0.01, "feature {feature}: {actual} vs {expected}");
            }
        }
    )*};
}

embeds! {
    qwen3_embedding_0_6b_embeds_the_last_row: QWEN3_EMBEDDING_0_6B;
    qwen3_embedding_4b_embeds_the_last_row: QWEN3_EMBEDDING_4B;
    qwen3_embedding_8b_embeds_the_last_row: QWEN3_EMBEDDING_8B;
}

// # Batch invariance

/// Under a batch-invariant load the base model returns a request's rows with the same bits
/// whatever shares its calls: its 40-token prompt alone in 16-row calls, then in chunks of 5, 10,
/// 3, 12 and 10 rows at the end, start, middle and second row of calls beside other prompts'
/// rows and decode rows, one of them over a 1,100-position history.
#[test]
#[ignore = "opens a device"]
fn batch_invariant_rows_keep_their_bits() {
    let checkpoint = checkpoint((QWEN3_0_6B.repo, QWEN3_0_6B.revision));
    let mut device = device();
    let options = (LoadOptions::new().architecture("Qwen3Model").unwrap()).batch_invariant();
    let model = options
        .load(&mut device, &checkpoint)
        .unwrap()
        .decoder()
        .unwrap();
    // The followed request's blocks, two other prompts', and the long history's.
    let (followed, first, second) = ([0, 1, 2], [3, 4], [5, 6]);
    let long: Vec<u32> = (7..77).collect();
    let tokens: Vec<i32> = (0..40).map(|at| 1000 + 37 * at).collect();
    let prompt: Vec<i32> = (0..20).map(|at| 5000 + 11 * at).collect();
    runtime().block_on(async {
        let cache = Cache::new(&mut device, &model.limits().cache, 80).unwrap();
        let call = async |device: &mut Device, requests: Vec<Request>| {
            let rows = model.forward(device, &cache, &requests, &[]).await.unwrap();
            let mut host = vec![0; rows.buffer.size()];
            (rows.buffer.to_host(&mut device.pdma))
                .output(&mut host[..])
                .await
                .unwrap();
            host
        };
        let own = async |device: &mut Device, rows: Range<usize>| {
            request(device, &followed, (rows.start, &tokens[rows]), Keep::All).await
        };
        let other = async |device: &mut Device, blocks: &[u32], rows: Range<usize>| {
            request(device, blocks, (rows.start, &prompt[rows]), Keep::None).await
        };
        let decode = async |device: &mut Device, position: usize| {
            request(device, &long, (position, &[7]), Keep::None).await
        };
        let mut alone = Vec::new();
        for start in (0..40).step_by(16) {
            let request = own(&mut device, start..(start + 16).min(40)).await;
            alone.extend(call(&mut device, vec![request]).await);
        }
        for start in (0..1100).step_by(16) {
            let history = vec![7; (1100 - start).min(16)];
            let request = request(&mut device, &long, (start, &history), Keep::Last).await;
            call(&mut device, vec![request]).await;
        }
        let mut shared = Vec::new();
        let requests = vec![
            other(&mut device, &first, 0..11).await,
            own(&mut device, 0..5).await,
        ];
        shared.extend(call(&mut device, requests).await);
        let requests = vec![
            own(&mut device, 5..15).await,
            decode(&mut device, 1100).await,
            other(&mut device, &second, 0..5).await,
        ];
        shared.extend(call(&mut device, requests).await);
        let requests = vec![
            other(&mut device, &second, 5..11).await,
            own(&mut device, 15..18).await,
            other(&mut device, &first, 11..18).await,
        ];
        shared.extend(call(&mut device, requests).await);
        let requests = vec![
            decode(&mut device, 1101).await,
            own(&mut device, 18..30).await,
            other(&mut device, &first, 18..19).await,
        ];
        shared.extend(call(&mut device, requests).await);
        let requests = vec![
            own(&mut device, 30..40).await,
            other(&mut device, &second, 11..17).await,
        ];
        shared.extend(call(&mut device, requests).await);
        let width = 1024 * size_of::<bf16>();
        assert_eq!((alone.len(), shared.len()), (40 * width, 40 * width));
        let differs = (alone.chunks(width).zip(shared.chunks(width))).position(|(a, b)| a != b);
        assert_eq!(differs, None, "the request's first row whose bits differ");
    });
}

// # Refusals, before any weight is read

/// Every choice the options refuse is reported from the config alone: the checkpoint here holds no
/// tensor, so a read would be `Missing`.
#[test]
fn refusals_read_no_weight() {
    let directory = tempfile::tempdir().unwrap();
    let config = std::fs::read(
        snapshot((QWEN3_0_6B.repo, QWEN3_0_6B.revision), &["config.json"]).join("config.json"),
    )
    .unwrap();
    std::fs::write(directory.path().join("config.json"), &config).unwrap();
    let checkpoint = Checkpoint::new(directory.path(), None).unwrap();

    let selected = LoadOptions::new().parse(&checkpoint).unwrap();
    assert_eq!(
        (selected.architecture(), selected.size(), selected.chips()),
        ("Qwen3ForCausalLM", "0.6B", 1)
    );

    let refused = |result: Result<(), LoadError>| match result {
        Err(LoadError::InvalidData { name, reason }) => ("invalid", name, reason),
        Err(LoadError::Unsupported { name, reason }) => ("unsupported", name, reason),
        other => panic!("{other:?} is no refusal of a choice"),
    };
    let named = |name| LoadOptions::new().architecture(name).map(drop);
    let parse = |options: LoadOptions| options.parse(&checkpoint).map(drop);
    let edited = |edit: fn(&mut serde_json::Value)| {
        let mut edited: serde_json::Value = serde_json::from_slice(&config).unwrap();
        edit(&mut edited);
        let checkpoint = Checkpoint::new(directory.path(), Some(edited)).unwrap();
        LoadOptions::new().parse(&checkpoint).map(drop)
    };
    // Every implementation of the name refuses the edited width before its form, the BF16 sizes
    // and then those a load quantizes; the registry reports each refusal in its order.
    let widths = [
        ("0.6B", 1024),
        ("1.7B", 2048),
        ("4B", 2560),
        ("8B", 4096),
        ("0.6B", 1024),
        ("1.7B", 2048),
        ("4B", 2560),
    ]
    .map(|(size, width)| format!("config.json: hidden_size: 1000, the size {size} holds {width}"));
    let cases = [
        (
            named("NoSuchModel"),
            ("invalid", "NoSuchModel", "unknown architecture".into()),
        ),
        (
            named("ApertusForCausalLM"),
            (
                "unsupported",
                "ApertusForCausalLM",
                "no model type is implemented".into(),
            ),
        ),
        (
            LoadOptions::new().chips(0).map(drop),
            (
                "invalid",
                "chips",
                "a model loads onto at least one chip".into(),
            ),
        ),
        (
            parse(LoadOptions::new().chips(4).unwrap()),
            (
                "unsupported",
                "chips",
                "every implementation runs on one chip".into(),
            ),
        ),
        (
            parse(LoadOptions::new().kv_cache(Dtype::F8_E4M3)),
            (
                "unsupported",
                "kv_cache",
                "every implementation caches keys and values in BF16".into(),
            ),
        ),
        (
            parse(LoadOptions::new().layers(0..14)),
            (
                "unsupported",
                "layers",
                "every implementation loads whole".into(),
            ),
        ),
        (
            edited(|config| config["hidden_size"] = 1000.into()),
            (
                "invalid",
                "Qwen3ForCausalLM",
                format!(
                    "no implementation serves the checkpoint: {}",
                    widths.join("; ")
                ),
            ),
        ),
        (
            edited(|config| drop(config.as_object_mut().unwrap().remove("architectures"))),
            (
                "invalid",
                "config.json:architectures",
                "the config names no architecture".into(),
            ),
        ),
    ];
    for (result, (kind, name, reason)) in cases {
        assert_eq!(refused(result), (kind, name.to_string(), reason));
    }
}
