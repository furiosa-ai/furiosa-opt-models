//! Llama's tree in BF16 through the engine's Rust API, Llama-3-8B and Mistral-7B: the
//! implementation each checkpoint selects, loaded and decoded one position a call.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use furiosa_opt_engine::decoder::{Cache, HbmTensor, Keep, Request, Rows};
use furiosa_opt_engine::{Checkpoint, LoadOptions};
use furiosa_opt_std::prelude::{Device, HostBufferSlice, Topology};

/// The hub repository and revision of Llama-3-8B this test reads.
const LLAMA: (&str, &str) = (
    "NousResearch/Meta-Llama-3-8B",
    "315b20096dc791d381d514deb5f8bd9c8d6d3061",
);

/// The hub repository and revision of Mistral-7B v0.3 this test reads.
const MISTRAL: (&str, &str) = (
    "mistralai/Mistral-7B-v0.3",
    "caa1feb0e54d415e2df31207e5f4e273e33509b1",
);

/// "Instruct: Given a web search query, retrieve relevant passages that answer the query
/// Query:What is the capital of China?" in Llama 3's tokens, after its begin-of-text token.
const PROMPT: [i32; 27] = [
    128000, 644, 1257, 25, 16644, 264, 3566, 2778, 3319, 11, 17622, 9959, 47869, 430, 4320, 279,
    3319, 198, 2929, 25, 3923, 374, 279, 6864, 315, 5734, 30,
];

/// Llama-3-8B's greedy continuation of `PROMPT` in BF16: Transformers' `generate(max_new_tokens=8,
/// do_sample=False)` of `AutoModelForCausalLM.from_pretrained(.., dtype=torch.bfloat16)` on CPU.
const CONTINUATION: [i32; 8] = [22559, 25, 27647, 198, 2929, 25, 3923, 374];

/// `PROMPT`'s text in Mistral's tokens, after its begin-of-sequence token.
const MISTRAL_PROMPT: [i32; 28] = [
    1, 1328, 2124, 29515, 13396, 1032, 5454, 4240, 6477, 29493, 20900, 9366, 2223, 2059, 1137,
    5140, 1040, 6477, 781, 4062, 29515, 3963, 1117, 1040, 6333, 1070, 5845, 29572,
];

/// Mistral-7B v0.3's greedy continuation of `MISTRAL_PROMPT`, as Transformers gives it for
/// `CONTINUATION`.
const MISTRAL_CONTINUATION: [i32; 8] = [781, 15536, 29496, 1208, 2223, 2059, 29515, 781];

/// The request's history blocks, out of the four the cache holds, in history order.
const BLOCKS: [u32; 3] = [2, 0, 3];

/// The directory of a hub checkpoint's files that `include` matches, fetched into the hub cache
/// once.
fn snapshot((repository, revision): (&str, &str), include: &[&str]) -> PathBuf {
    let mut command = Command::new("uvx");
    command.args(["--from", "huggingface-hub", "hf", "download"]);
    command.args([repository, "--revision", revision, "--quiet"]);
    for pattern in include {
        command.args(["--include", pattern]);
    }
    let output = command
        .stderr(Stdio::inherit())
        .output()
        .expect("uvx fetches the test checkpoint");
    assert!(output.status.success(), "the {repository} download failed");
    String::from_utf8(output.stdout).unwrap().trim().into()
}

/// A hub checkpoint, weights included.
fn checkpoint(hub: (&str, &str)) -> Checkpoint {
    Checkpoint::new(snapshot(hub, &["*.json", "*.safetensors"]), None).unwrap()
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

/// The `count` tokens the checkpoint of `hub`, loaded by the options' selection and decoded one
/// position a call through a scattered block table, selects after `prompt`.
fn continuation(hub: (&str, &str), prompt: &[i32], count: usize) -> Vec<i32> {
    let checkpoint = checkpoint(hub);
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let model = LoadOptions::new()
        .load(&mut device, &checkpoint)
        .unwrap()
        .decoder()
        .unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let cache = Cache::new(&mut device, &model.limits().cache, 4).unwrap();
        let (mut selected, mut continued, mut steps) = (0, Vec::new(), Vec::new());
        for position in 0..prompt.len() + count - 1 {
            let input = prompt.get(position).copied().unwrap_or(selected);
            let request = request(&mut device, input, position).await;
            let start = Instant::now();
            let tokens = model
                .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                .await
                .unwrap();
            selected = token(&mut device, tokens).await;
            steps.push(start.elapsed());
            if position + 1 >= prompt.len() {
                continued.push(selected);
            }
        }
        // The wall time of a call, its token read back included, past the first call's builds.
        let mut later = steps[1..].to_vec();
        later.sort();
        eprintln!(
            "call: first {:?}, median {:?}, min {:?}",
            steps[0],
            later[later.len() / 2],
            later[0]
        );
        continued
    })
}

/// Llama-3-8B continues the prompt with Transformers' greedy tokens.
#[test]
#[ignore = "opens a device"]
fn llama3_8b_continues_the_prompt() {
    assert_eq!(
        continuation(LLAMA, &PROMPT, CONTINUATION.len()),
        CONTINUATION
    );
}

/// Mistral-7B v0.3 continues the prompt with Transformers' greedy tokens; its snapshot also holds
/// Mistral's own consolidated file, which the index leaves out.
#[test]
#[ignore = "opens a device"]
fn mistral_7b_continues_the_prompt() {
    assert_eq!(
        continuation(MISTRAL, &MISTRAL_PROMPT, MISTRAL_CONTINUATION.len()),
        MISTRAL_CONTINUATION
    );
}

/// The published config selects the 8B implementation from the config alone: the checkpoint here
/// holds no tensor, so a read would be `Missing`.
#[test]
fn the_config_selects_the_8b_implementation() {
    let directory = tempfile::tempdir().unwrap();
    let config = std::fs::read(snapshot(LLAMA, &["config.json"]).join("config.json")).unwrap();
    std::fs::write(directory.path().join("config.json"), &config).unwrap();
    let checkpoint = Checkpoint::new(directory.path(), None).unwrap();
    let selected = LoadOptions::new().parse(&checkpoint).unwrap();
    assert_eq!(
        (selected.architecture(), selected.size(), selected.chips()),
        ("LlamaForCausalLM", "8B", 1)
    );
}

/// Mistral-7B v0.3's published config selects the Mistral implementation from the config alone.
#[test]
fn the_config_selects_the_mistral_implementation() {
    let directory = tempfile::tempdir().unwrap();
    let config = std::fs::read(snapshot(MISTRAL, &["config.json"]).join("config.json")).unwrap();
    std::fs::write(directory.path().join("config.json"), &config).unwrap();
    let checkpoint = Checkpoint::new(directory.path(), None).unwrap();
    let selected = LoadOptions::new().parse(&checkpoint).unwrap();
    assert_eq!(
        (selected.architecture(), selected.size(), selected.chips()),
        ("MistralForCausalLM", "Mistral-7B", 1)
    );
}
