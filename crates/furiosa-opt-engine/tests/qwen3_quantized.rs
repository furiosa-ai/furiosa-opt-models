//! Qwen3 quantized to FP8 at load through the engine's Rust API: a BF16 checkpoint loaded with each
//! quantization holds the codes and scales the frameworks' quantizers give it, and decodes; the
//! quantizations it refuses are refused before any weight is read.

#![feature(adt_const_params, unsized_const_params)]

use std::future::IntoFuture;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use furiosa_opt_engine::decoder::{self, Cache, Keep, Request, Rows};
use furiosa_opt_engine::{Checkpoint, Hbm, Load, LoadError, LoadOptions, Quantization};
use furiosa_opt_models::load::Context;
use furiosa_opt_models::models::Implements;
use furiosa_opt_models::models::decoder::attention::{Geometry, KvHeads, qwen3};
use furiosa_opt_models::models::decoder::{Form, form};
use furiosa_opt_models::specialize::attention::Decode;
use furiosa_opt_models::tensor::cache;
use furiosa_opt_models::tensor::form::fp8::Granularity;
use furiosa_opt_models::tensor::parallel::Parallel;
use furiosa_opt_std::prelude::*;
use safetensors::Dtype;

/// The hub revision of Qwen/Qwen3-0.6B these tests read.
const REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";

/// "Instruct: Given a web search query, retrieve relevant passages that answer the query
/// Query:What is the capital of China?" in Qwen3's tokens.
const QWEN: [i32; 25] = [
    641, 1235, 25, 16246, 264, 3482, 2711, 3239, 11, 17179, 9760, 46769, 429, 4226, 279, 3239,
    11361, 25, 3838, 374, 279, 6722, 315, 5616, 30,
];

/// Qwen3-0.6B's greedy continuation of `QWEN` under either quantization: Transformers' greedy
/// tokens with every projection restored from torch's codes and scales. At each step the second
/// token is at most 0.69 times as likely as the first, so only the first is within 5% of it.
const CONTINUATION: [i32; 8] = [576, 3482, 2711, 3239, 374, 330, 22282, 594];

/// The FNV-1a 64 digests of layer 0's codes and of its row scales' FP32 bytes under each
/// granularity, as torch's quantizer of each framework gives them: of the joined query, key and
/// value projection (rows joined head by head), the output, gate, up and down projections.
const LAYER: [[(u64, u64); 5]; 2] = [
    [
        (9294998295025189955, 11282888862993146661),
        (8538269687551517631, 9586696893038224165),
        (9750773197714196190, 16690636256519172901),
        (6311568669608634275, 16690636256519172901),
        (17378256406554666565, 9711946953892844325),
    ],
    [
        (7689942180726729397, 312498989827055998),
        (18129886945720467249, 10671502234385792847),
        (15669421129021200064, 6572636871194389929),
        (18030556371742531049, 6349145977851510309),
        (12342923209435713515, 671045855503459192),
    ],
];

/// The request's history blocks, out of the four the cache holds, in history order.
const BLOCKS: [u32; 3] = [2, 0, 3];

type Size = qwen3::Size<"0.6B">;
type Hidden = <Size as Geometry>::Hidden;
type Intermediate = <Size as Geometry>::Intermediate;
type Queries = <Size as Geometry>::Queries;
type Width = <Size as Decode>::Width;

/// The Qwen3-0.6B causal line whose projections hold form `F`.
type Line<F> = qwen3::ForCausalLM<Size, F, Parallel<1>, cache::Bf16, Hbm>;

/// A channel-scaled projection of `In`-wide rows to `Out` features, in HBM.
type Projection<In, Out> =
    furiosa_opt_models::models::decoder::Projection<form::ChannelScaled, In, Out>;

/// The channel-scaled query, key and value projection joined into one.
type Packed = <form::ChannelScaled as Form>::Of<
    Hidden,
    Width,
    bf16,
    m![1],
    Hbm,
    furiosa_opt_models::ops::attention::Packed<KvHeads, <Size as Decode>::Packed, Hbm, f8e4m3>,
>;

/// The directory of Qwen3-0.6B's files that `include` matches, fetched into the hub cache once.
fn snapshot(include: &[&str]) -> PathBuf {
    let mut command = Command::new("uvx");
    command.args(["--from", "huggingface-hub", "hf", "download"]);
    command.args(["Qwen/Qwen3-0.6B", "--revision", REVISION, "--quiet"]);
    for pattern in include {
        command.args(["--include", pattern]);
    }
    let output = command
        .stderr(Stdio::inherit())
        .output()
        .expect("uvx fetches the test checkpoint");
    assert!(output.status.success(), "the Qwen3-0.6B download failed");
    String::from_utf8(output.stdout).unwrap().trim().into()
}

fn checkpoint() -> Checkpoint {
    Checkpoint::new(snapshot(&["*.json", "*.safetensors"]), None).unwrap()
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

/// FNV-1a 64 of `bytes`.
fn digest(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The digests of a channel-scaled weight's codes and of its row scales, read back from HBM.
fn digests<Out: AxisName, In: AxisName, W, T>(
    device: &mut Device,
    projection: &furiosa_opt_models::tensor::form::BlockScaled<
        In,
        Out,
        f8e4m3,
        f32,
        m![Out, 1 # 8],
        (),
        m![1],
        W,
        T,
    >,
) -> (u64, u64)
where
    W: core::borrow::Borrow<HbmTensor<f8e4m3, m![1], m![Out, In]>>,
    T: core::borrow::Borrow<HbmTensor<f32, m![1], m![Out, 1 # 8]>>,
{
    let runtime = runtime();
    let codes = runtime
        .block_on(
            projection
                .codes()
                .to_host::<m![Out, In]>(&mut device.pdma)
                .into_future(),
        )
        .unwrap();
    let scales = runtime
        .block_on(
            projection
                .scales()
                .to_host::<m![Out, 1 # 8]>(&mut device.pdma)
                .into_future(),
        )
        .unwrap();
    let scales: Vec<u8> = (scales.into_vec().chunks_exact(8))
        .flat_map(|lanes| lanes[0].to_le_bytes())
        .collect();
    (
        digest(&Vec::<u8>::from(codes.into_buffer())),
        digest(&scales),
    )
}

/// One request a call: the token at `position` appended after the `position` cached ones.
async fn request(device: &mut Device, token: i32, position: usize) -> Request {
    let ids = HostBufferSlice::from(&token.to_le_bytes()[..])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    Request {
        rows: vec![BLOCKS.to_vec()],
        cached: position,
        tokens: Rows::new(&ids, 0..1).unwrap(),
        keep: Keep::Last,
        adapter: None,
        positions: None,
        mask: None,
        overrides: Vec::new(),
    }
}

/// The I32 token of a one-row output, read back once its call completed.
async fn token(device: &mut Device, tokens: decoder::HbmTensor) -> i32 {
    let mut host = [0; 4];
    tokens
        .buffer
        .to_host(&mut device.pdma)
        .output(&mut host[..])
        .await
        .unwrap();
    i32::from_le_bytes(host)
}

/// The projection `P` at `name` of layer 0, loaded alone from `quantized`; the names under it
/// that `siblings` start are other leaves'.
fn layer<P: for<'a> Load<&'a Checkpoint>>(
    device: &mut Device,
    quantized: &Checkpoint,
    name: &str,
    siblings: &[&str],
) -> P {
    let mut context = Context::new(device);
    let position = format!("model.layers.0.{name}");
    let loaded = context.field(name, &position, |cx| {
        cx.checkpoint(quantized, |cx| {
            siblings.iter().for_each(|sibling| cx.ignore(sibling));
            P::load(cx, quantized)
        })
    });
    loaded.unwrap()
}

/// Layer 0's projections, loaded alone from the checkpoint read quantized, hold torch's codes and
/// scales: vLLM's tensor scale shared by the joined query, key and value rows and by gate and up,
/// SGLang's channel scales.
#[test]
#[ignore = "opens a device"]
fn holds_the_codes_and_scales_of_each_quantizer() {
    let checkpoint = checkpoint();
    let mut device = device();
    for (granularity, expected) in [Granularity::Tensor, Granularity::Channel]
        .into_iter()
        .zip(LAYER)
    {
        let quantized = checkpoint.quantized(granularity).unwrap();
        let attention = ["o_proj", "q_norm", "k_norm"];
        let joined: Packed = layer(&mut device, &quantized, "self_attn", &attention);
        let output: Projection<Queries, Hidden> =
            layer(&mut device, &quantized, "self_attn.o_proj", &[]);
        let gate: Projection<Hidden, Intermediate> =
            layer(&mut device, &quantized, "mlp.gate_proj", &[]);
        let up: Projection<Hidden, Intermediate> =
            layer(&mut device, &quantized, "mlp.up_proj", &[]);
        let down: Projection<Intermediate, Hidden> =
            layer(&mut device, &quantized, "mlp.down_proj", &[]);
        let held = [
            digests(&mut device, &joined),
            digests(&mut device, &output),
            digests(&mut device, &gate),
            digests(&mut device, &up),
            digests(&mut device, &down),
        ];
        assert_eq!(held, expected, "{granularity:?}");
    }
}

/// Qwen3-0.6B, loaded from its BF16 checkpoint with each quantization and decoded one position a
/// call, continues the prompt with the greedy tokens of Transformers on the restored weights.
#[test]
#[ignore = "opens a device"]
fn continues_the_prompt_quantized_at_load() {
    let checkpoint = checkpoint();
    let mut device = device();
    for quantization in [Quantization::Fp8PerTensor, Quantization::Fp8PerChannel] {
        let model = LoadOptions::new()
            .quantization(quantization)
            .load(&mut device, &checkpoint)
            .unwrap()
            .decoder()
            .unwrap();
        runtime().block_on(async {
            let cache = Cache::new(&mut device, &model.limits().cache, 4).unwrap();
            let (mut selected, mut continued) = (0, Vec::new());
            for position in 0..QWEN.len() + CONTINUATION.len() - 1 {
                let input = QWEN.get(position).copied().unwrap_or(selected);
                let request = request(&mut device, input, position).await;
                let tokens = model
                    .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                    .await
                    .unwrap();
                selected = token(&mut device, tokens).await;
                if position + 1 >= QWEN.len() {
                    continued.push(selected);
                }
            }
            assert_eq!(continued, CONTINUATION, "{quantization}");
        });
    }
}

/// A quantization selects the FP8 line of the checkpoint's size, which serves only the checkpoint
/// read quantized, and no BF16 line of a base model or classifier, from the config alone, an absent
/// or null `quantization_config` alike; a declared one is refused with and without a quantization.
#[test]
fn selects_the_line_of_the_quantization_from_the_config() {
    let directory = tempfile::tempdir().unwrap();
    let config = std::fs::read(snapshot(&["config.json"]).join("config.json")).unwrap();
    let config: serde_json::Value = serde_json::from_slice(&config).unwrap();
    let declaring = |declared: serde_json::Value| {
        let mut config = config.clone();
        config["quantization_config"] = declared;
        Checkpoint::new(directory.path(), Some(config)).unwrap()
    };
    let online = |quantization| LoadOptions::new().quantization(quantization);
    let precision = |options: &LoadOptions, checkpoint: &Checkpoint| {
        let selected = options.parse(checkpoint).unwrap();
        (selected.size(), selected.precision())
    };
    let refusal = |options: &LoadOptions, checkpoint: &Checkpoint| match options.parse(checkpoint) {
        Err(LoadError::InvalidData { name, .. }) => name,
        other => panic!("{:?} is not `InvalidData`", other.map(drop)),
    };
    for stored in [
        Checkpoint::new(directory.path(), Some(config.clone())).unwrap(),
        declaring(serde_json::Value::Null),
    ] {
        assert_eq!(
            precision(&LoadOptions::new(), &stored),
            ("0.6B", Dtype::BF16)
        );
        assert_eq!(Line::<form::Affine>::scope(&stored).unwrap(), "");
        assert!(Line::<form::ChannelScaled>::scope(&stored).is_err());
        for granularity in [Granularity::Tensor, Granularity::Channel] {
            let quantized = stored.quantized(granularity).unwrap();
            assert!(Line::<form::Affine>::scope(&quantized).is_err());
            assert_eq!(Line::<form::ChannelScaled>::scope(&quantized).unwrap(), "");
        }
        for quantization in [Quantization::Fp8PerTensor, Quantization::Fp8PerChannel] {
            let options = online(quantization);
            assert_eq!(precision(&options, &stored), ("0.6B", Dtype::F8_E4M3));
            for architecture in ["Qwen3Model", "Qwen3ForSequenceClassification"] {
                let other = options.clone().architecture(architecture).unwrap();
                assert_eq!(refusal(&other, &stored), architecture);
            }
        }
    }
    let blocks = declaring(serde_json::json!({
        "quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128],
    }));
    assert_eq!(refusal(&LoadOptions::new(), &blocks), "Qwen3ForCausalLM");
    let quantized = refusal(&online(Quantization::Fp8PerTensor), &blocks);
    assert_eq!(quantized, "config.json:quantization_config");
}
