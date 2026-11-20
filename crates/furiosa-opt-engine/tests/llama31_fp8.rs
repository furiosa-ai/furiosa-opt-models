//! A short native W8A8 continuation through the engine's checkpoint and request boundaries.
use furiosa_opt_engine::decoder::{Cache, Instance, Keep, Request, Rows, Served};
use furiosa_opt_engine::{Checkpoint, Tiling};
use furiosa_opt_models::models::decoder::attention::llama::fp8::ForCausalLM;
use furiosa_opt_std::prelude::{Device, HostBufferSlice, Topology};
use std::process::{Command, Stdio};

const REPO: &str = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic";
const REVISION: &str = "442e7f522277df12f53d63e8384087af31fbfc4b";
/// "Instruct: Given a web search query, retrieve relevant passages that answer the query
/// Query:What is the capital of China?" in Llama's tokens, after begin-of-text.
const PROMPT: [i32; 27] = [
    128000, 644, 1257, 25, 16644, 264, 3566, 2778, 3319, 11, 17622, 9959, 47869, 430, 4320, 279,
    3319, 198, 2929, 25, 3923, 374, 279, 6864, 315, 5734, 30,
];
/// Independent PyTorch W8A8 calculation: the frozen vLLM v0.8.4 token quantizer and the
/// specified BF16 O/MLP partial-storage boundaries. A full FP32 dot-product oracle differs;
/// these are not a recording of this engine's output or an upstream CUDA execution.
const CONTINUATION: [i32; 8] = [720, 12465, 425, 25, 27647, 374, 279, 6864];

fn checkpoint() -> Checkpoint {
    let output = Command::new("uvx")
        .args([
            "--from",
            "huggingface-hub",
            "hf",
            "download",
            REPO,
            "--revision",
            REVISION,
            "--quiet",
            "--include",
            "*.json",
            "--include",
            "*.safetensors",
        ])
        .stderr(Stdio::inherit())
        .output()
        .expect("hf downloads the pinned checkpoint");
    assert!(output.status.success());
    Checkpoint::new(String::from_utf8(output.stdout).unwrap().trim(), None).unwrap()
}

#[test]
#[ignore = "opens a device and loads the pinned checkpoint"]
fn continues_the_prompt_with_token_dynamic_fp8() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let model = Instance::new(
        checkpoint().load::<ForCausalLM>(&mut device).unwrap(),
        Tiling::Adaptive,
    );
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            // "The capital of France is Paris. " repeated 80 times, then
            // "Question: What is the capital of France? Answer:". This crosses a span boundary.
            let mut long = vec![128000, 791, 6864, 315, 9822, 374, 12366, 13];
            long.extend([578, 6864, 315, 9822, 374, 12366, 13].repeat(79));
            long.extend([16225, 25, 3639, 374, 279, 6864, 315, 9822, 30, 22559, 25]);
            for (prompt, expected) in [
                (PROMPT.to_vec(), CONTINUATION),
                (long, [12366, 13, 578, 6864, 315, 9822, 374, 12366]),
            ] {
                let count = (prompt.len() + expected.len()).div_ceil(16);
                let blocks: Vec<u32> = (0..count as u32).rev().collect();
                let cache = Cache::new(&device, &model.limits().cache, count).unwrap();
                let (mut ids, mut cached, mut generated) = (prompt, 0, Vec::new());
                for _ in expected {
                    let bytes: Vec<_> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
                    let tokens = HostBufferSlice::from(&bytes[..])
                        .to_hbm(&mut device.pdma)
                        .await
                        .unwrap();
                    let request = Request {
                        rows: vec![blocks.clone()],
                        cached,
                        tokens: Rows::new(&tokens, 0..ids.len()).unwrap(),
                        keep: Keep::Last,
                        adapter: None,
                        positions: None,
                        mask: None,
                        overrides: Vec::new(),
                    };
                    let output = model
                        .forward(&mut device, &cache, &[request], &[])
                        .await
                        .unwrap();
                    let mut bytes = [0; 4];
                    output
                        .buffer
                        .to_host(&mut device.pdma)
                        .output(&mut bytes[..])
                        .await
                        .unwrap();
                    let selected = i32::from_le_bytes(bytes);
                    cached += ids.len();
                    generated.push(selected);
                    ids = vec![selected];
                }
                assert_eq!(generated, expected);
            }
        });
}
