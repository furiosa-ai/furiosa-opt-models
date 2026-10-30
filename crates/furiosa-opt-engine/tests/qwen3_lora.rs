//! Qwen3 in BF16 with LoRA adapters through the engine's Rust API: PEFT checkpoints loaded as
//! adapters and run beside rows without one, against the base model with each adapter merged into
//! its weights; and a load for adapters of a line that runs none, refused from its config.

#![feature(adt_const_params, unsized_const_params)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use furiosa_opt_engine::decoder::{Cache, Decoder, Instance, Keep, Request, Rows};
use furiosa_opt_engine::{AsyncModule, Checkpoint, Hbm, Load, LoadError, LoadOptions, Tiling};
use furiosa_opt_models::load::Context;
use furiosa_opt_models::models::decoder::attention::qwen3;
use furiosa_opt_models::models::decoder::form;
use furiosa_opt_models::tensor::cache;
use furiosa_opt_models::tensor::parallel::Parallel;
use furiosa_opt_std::prelude::{Device, HostBufferSlice, M, Topology, bf16};
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use serde_json::{Value, json};

/// The hub revisions of Qwen/Qwen3-0.6B and of the block-scaled E4M3 Qwen3-8B these tests read.
const REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";
const BLOCK_SCALED: &str = "6124ba90c39c8e2299ddd57f4b4f0df5845480c8";

/// Qwen3-0.6B's adapter tree.
type Adapter = qwen3::Adapter<qwen3::Size<"0.6B">, Hbm>;

/// Qwen3-0.6B's base model, which returns the kept rows' hidden values.
type Base = qwen3::Model<qwen3::Size<"0.6B">, form::Affine, Parallel<1>, cache::Bf16, Hbm>;

/// Qwen3-0.6B's causal model, whose base model a test serves.
type Causal = qwen3::ForCausalLM<qwen3::Size<"0.6B">, form::Affine, Parallel<1>, cache::Bf16, Hbm>;

const HIDDEN: usize = 1024;

/// The layers the test adapters update: the last four. BF16 rounding of the merged weights and
/// of the update differ, and 28 layers amplify that at a prompt's first position to the size of
/// the update itself; four layers keep it small, and other layers hold zero terms.
const ADAPTED: std::ops::Range<usize> = 24..28;

/// Each module an adapter may update, by its path in a layer, with its outputs and inputs: the
/// query, key, value and output projections and the feed-forward's gate, up and down projections.
const MODULES: [(&str, usize, usize); 7] = [
    ("self_attn.q_proj", 2048, HIDDEN),
    ("self_attn.k_proj", 1024, HIDDEN),
    ("self_attn.v_proj", 1024, HIDDEN),
    ("self_attn.o_proj", HIDDEN, 2048),
    ("mlp.gate_proj", 3072, HIDDEN),
    ("mlp.up_proj", 3072, HIDDEN),
    ("mlp.down_proj", HIDDEN, 3072),
];

/// Three prompts: two of five tokens, the first of which runs with no adapter, and one of one
/// token, whose row a call lays out before the others'; the last two run with an adapter each.
const PROMPTS: [&[i32]; 3] = [
    &[641, 1235, 25, 16246, 264],
    &[3838, 374, 279, 6722, 315],
    &[3482],
];

/// The checkpoint directory of `model` at `revision`, its files that `include` matches, fetched
/// into the hub cache once.
fn snapshot(model: &str, revision: &str, include: &[&str]) -> PathBuf {
    let mut command = Command::new("uvx");
    command.args(["--from", "huggingface-hub", "hf", "download"]);
    command.args([model, "--revision", revision, "--quiet"]);
    for pattern in include {
        command.args(["--include", pattern]);
    }
    let output = command
        .stderr(Stdio::inherit())
        .output()
        .expect("uvx fetches the test checkpoint");
    assert!(output.status.success(), "the {model} download failed");
    String::from_utf8(output.stdout).unwrap().trim().into()
}

/// Pseudo-random values in `[-1, 1)` times `scale`.
fn values(count: usize, mut state: u64, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
        })
        .collect()
}

/// A PEFT adapter of rank `rank` updating the modules `targets` of the `ADAPTED` layers: its
/// factors by module name, `lora_A` `[rank, inputs]` and `lora_B` `[outputs, rank]` in FP32, and
/// its scale.
struct Peft {
    config: Value,
    factors: Vec<(String, [Vec<f32>; 2])>,
    scale: f32,
}

impl Peft {
    fn new(seed: u64, rank: usize, alpha: f32, rslora: bool, targets: &[&str]) -> Self {
        let factors = ADAPTED
            .flat_map(|layer| {
                (MODULES.iter().enumerate())
                    .filter(|(_, (path, ..))| {
                        path.split_once('.')
                            .is_some_and(|(_, module)| targets.contains(&module))
                    })
                    .map(move |(at, &(path, outputs, inputs))| {
                        let seed = seed + 97 * layer as u64 + 7919 * at as u64;
                        let a = values(rank * inputs, seed, 1.0 / (inputs as f32).sqrt());
                        let b = values(outputs * rank, seed + 1, 0.06);
                        (format!("model.layers.{layer}.{path}"), [a, b])
                    })
            })
            .collect();
        let scale = alpha
            / if rslora {
                (rank as f32).sqrt()
            } else {
                rank as f32
            };
        let config = json!({
            "peft_type": "LORA", "r": rank, "lora_alpha": alpha, "use_rslora": rslora,
            "target_modules": targets, "bias": "none", "use_dora": false,
            "modules_to_save": null, "rank_pattern": {}, "alpha_pattern": {},
        });
        Self {
            config,
            factors,
            scale,
        }
    }

    /// The adapter as PEFT saves it in `directory`, opened as a checkpoint with its
    /// `adapter_config.json`.
    fn save(&self, directory: &Path) -> Checkpoint {
        let bytes = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        let rank = self.config["r"].as_u64().unwrap() as usize;
        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        for (module, [a, b]) in &self.factors {
            let name = |factor: &str| format!("base_model.model.{module}.{factor}.weight");
            tensors.push((name("lora_A"), vec![rank, a.len() / rank], bytes(a)));
            tensors.push((name("lora_B"), vec![b.len() / rank, rank], bytes(b)));
        }
        let views = (tensors.iter())
            .map(|(name, shape, data)| (name, TensorView::new(Dtype::F32, shape.clone(), data)));
        let views: Vec<_> = views.map(|(name, view)| (name, view.unwrap())).collect();
        let file = directory.join("adapter_model.safetensors");
        std::fs::write(file, safetensors::serialize(views, &None).unwrap()).unwrap();
        let config = directory.join("adapter_config.json");
        std::fs::write(config, self.config.to_string()).unwrap();
        Checkpoint::new(directory, Some(self.config.clone())).unwrap()
    }

    /// `base` with this adapter merged into its weights: `W + scale * B A` of each adapted module
    /// in FP32, rounded once to BF16, every other tensor as `base` holds it.
    fn merged(&self, base: &Checkpoint, directory: &Path) -> Checkpoint {
        std::fs::write(directory.join("config.json"), base.config().to_string()).unwrap();
        let mut merged = Checkpoint::new(directory, None).unwrap();
        for name in base.names() {
            let view = base.get(&name).unwrap();
            let module = name.strip_suffix(".weight");
            let update = (self.factors.iter()).find(|(path, _)| Some(path.as_str()) == module);
            let data = match update {
                None => view.data().to_vec(),
                Some((_, [a, b])) => {
                    let &[_, inputs] = view.shape() else {
                        panic!("{name} is a matrix")
                    };
                    let rank = a.len() / inputs;
                    let weight = view.data().as_chunks::<2>().0.iter();
                    (weight.enumerate())
                        .flat_map(|(at, &bytes)| {
                            let (row, column) = (at / inputs, at % inputs);
                            let update: f32 = (0..rank)
                                .map(|r| b[row * rank + r] * a[r * inputs + column])
                                .sum();
                            let value = bf16::from_bits(u16::from_le_bytes(bytes)).to_f32();
                            bf16::from_f32(value + self.scale * update)
                                .to_bits()
                                .to_le_bytes()
                        })
                        .collect()
                }
            };
            merged
                .insert(&name, view.dtype(), view.shape(), data)
                .unwrap();
        }
        merged
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// The base model of `checkpoint`, a causal one or a merged copy, loaded on `device` and served as
/// a load serves it.
fn base(checkpoint: &Checkpoint, device: &mut Device) -> Instance<Base> {
    let causal: Causal = Load::load(&mut Context::new(device), checkpoint).unwrap();
    Instance::new(causal.model, Tiling::Adaptive)
}

/// Prompt `index` as a request of every row's hidden values, on blocks of its own.
async fn request(device: &mut Device, index: usize, adapter: Option<usize>) -> Request {
    let bytes: Vec<u8> = (PROMPTS[index].iter())
        .flat_map(|token| token.to_le_bytes())
        .collect();
    let ids = HostBufferSlice::from(&bytes[..])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    Request {
        rows: vec![vec![index as u32]],
        cached: 0,
        tokens: Rows::new(&ids, 0..PROMPTS[index].len()).unwrap(),
        keep: Keep::All,
        adapter,
        positions: None,
        mask: None,
        overrides: Vec::new(),
    }
}

/// The BF16 values of a call's hidden rows, read once the call completed.
async fn hidden<R: M>(device: &mut Device, rows: Rows<bf16, R>) -> Vec<f32> {
    let mut host = vec![0; rows.buffer().size()];
    (rows.into_buffer().to_host(&mut device.pdma))
        .output(&mut host[..])
        .await
        .unwrap();
    (host.as_chunks::<2>().0.iter())
        .map(|&bytes| bf16::from_bits(u16::from_le_bytes(bytes)).to_f32())
        .collect()
}

/// The Euclidean distance between two rows.
fn distance(left: &[f32], right: &[f32]) -> f32 {
    (left.iter().zip(right))
        .map(|(l, r)| (l - r) * (l - r))
        .sum::<f32>()
        .sqrt()
}

/// One call of three prompts, the first without an adapter and the others with one each of two
/// adapters, one of every projection and one of three, returns for each prompt, wherever the call
/// lays out its rows, the rows it returns alone with its adapter, bit for bit, and for the prompt
/// without one the base model's rows.
/// Alone, a prompt's rows lie at most a quarter as far from the rows of the base model with its
/// adapter merged into the weights as those lie from the base model's.
#[test]
#[ignore = "opens a device"]
fn adapted_rows_match_merged_weights() {
    let directory = snapshot("Qwen/Qwen3-0.6B", REVISION, &["*.json", "*.safetensors"]);
    let checkpoint = Checkpoint::new(directory, None).unwrap();
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let every = MODULES.map(|(path, ..)| path.split_once('.').unwrap().1);
    let pefts = [
        Peft::new(11, 16, 32.0, false, &every),
        Peft::new(23, 8, 16.0, true, &["q_proj", "v_proj", "up_proj"]),
    ];
    let directories = [(); 4].map(|()| tempfile::tempdir().unwrap());
    let adapters: Vec<Adapter> = (pefts.iter().zip(&directories))
        .map(|(peft, directory)| {
            let saved = peft.save(directory.path());
            Base::adapter(&saved, &mut device).unwrap()
        })
        .collect();
    let model = base(&checkpoint, &mut device);
    let merged: Vec<Instance<Base>> = (pefts.iter().zip(&directories[2..]))
        .map(|(peft, directory)| base(&peft.merged(&checkpoint, directory.path()), &mut device))
        .collect();
    runtime().block_on(async {
        let cache = Cache::new(&mut device, &model.decoder().limits().cache, 4).unwrap();
        let mut requests = Vec::new();
        for (index, adapter) in [None, Some(0), Some(1)].into_iter().enumerate() {
            requests.push(request(&mut device, index, adapter).await);
        }
        let slots = [Some(&adapters[0]), Some(&adapters[1])];
        let rows = model
            .forward(&mut device, (), (&cache, &slots[..]), &requests[..])
            .await
            .unwrap();
        let mixed = hidden(&mut device, rows).await;
        // Prompt `index` alone through `model`, with `adapter` in the only slot of its call.
        let alone = async |device: &mut Device,
                           model: &Instance<Base>,
                           index,
                           adapter: Option<&Adapter>| {
            let cache = Cache::new(device, &model.decoder().limits().cache, 4).unwrap();
            let request = request(device, index, adapter.map(|_| 0)).await;
            let slots = [adapter];
            let rows = model
                .forward(device, (), (&cache, &slots[..adapter.iter().len()]), &[request][..])
                .await
                .unwrap();
            hidden(device, rows).await
        };
        // The rows of prompt `index` among the mixed call's, which returns them in request order.
        let range = |index: usize| {
            let start = PROMPTS[..index].iter().map(|prompt| prompt.len()).sum::<usize>();
            start * HIDDEN..(start + PROMPTS[index].len()) * HIDDEN
        };
        let plain = alone(&mut device, &model, 0, None).await;
        assert_eq!(mixed[range(0)], plain[..], "the prompt without an adapter");
        for (index, (adapter, merged)) in [(1, (&adapters[0], &merged[0])), (2, (&adapters[1], &merged[1]))] {
            let adapted = alone(&mut device, &model, index, Some(adapter)).await;
            assert_eq!(mixed[range(index)], adapted[..], "prompt {index} beside the others");
            let base = alone(&mut device, &model, index, None).await;
            let merged = alone(&mut device, merged, index, None).await;
            let rows = adapted.chunks_exact(HIDDEN).zip(merged.chunks_exact(HIDDEN));
            for (row, ((adapted, merged), base)) in rows.zip(base.chunks_exact(HIDDEN)).enumerate() {
                let (near, far) = (distance(adapted, merged), distance(merged, base));
                assert!(
                    4.0 * near <= far,
                    "prompt {index} row {row}: {near} from the merged rows, which lie {far} from the base"
                );
            }
        }
    });
}

/// A load for calls that run adapters parses a line whose projections take them, and refuses
/// from the config alone a line whose projections take none, naming the choice.
#[test]
fn loads_for_adapters_only_lines_that_take_them() {
    let config = |model: &str, revision: &str| {
        Checkpoint::new(snapshot(model, revision, &["config.json"]), None).unwrap()
    };
    let options = LoadOptions::new().adapters();
    let small = options.parse(&config("Qwen/Qwen3-0.6B", REVISION)).unwrap();
    assert_eq!(small.size(), "0.6B");
    match options.parse(&config("furiosa-ai/Qwen3-8B-FP8", BLOCK_SCALED)) {
        Err(LoadError::Unsupported { name, .. }) => assert_eq!(name, "adapters"),
        other => panic!("block-scaled Qwen3-8B for adapters: {other:?}"),
    }
}
