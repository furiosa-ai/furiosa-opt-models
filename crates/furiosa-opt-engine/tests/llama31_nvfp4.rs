//! A short W4A16 continuation through the engine, with inline input and expected tokens.
use furiosa_opt_engine::decoder::{Cache, Instance, Keep, Request, Rows, Served};
use furiosa_opt_engine::{Checkpoint, Tiling};
use furiosa_opt_models::models::decoder::attention::llama::nvfp4::ForCausalLM;
use furiosa_opt_std::prelude::{Device, HostBufferSlice, Topology};
use std::{
    path::PathBuf,
    process::{Command, Stdio},
};

const REPO: &str = "nvidia/Llama-3.1-8B-Instruct-NVFP4";
const REVISION: &str = "bdb54e24298451af785c0ac63c1b485e9b7400a2";
/// "Hello" after Llama's begin-of-text token.
const PROMPT: [i32; 2] = [128000, 9906];
/// ", I", from the independent PyTorch W4A16 reference for the pinned checkpoint.
const CONTINUATION: [i32; 2] = [11, 358];

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
    let path: PathBuf = String::from_utf8(output.stdout).unwrap().trim().into();
    Checkpoint::new(path, None).unwrap()
}
async fn request(device: &mut Device, ids: &[i32], cached: usize, blocks: &[u32]) -> Request {
    let bytes: Vec<_> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    let ids = HostBufferSlice::from(&bytes[..])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    Request {
        rows: vec![blocks.to_vec()],
        cached,
        tokens: Rows::new(&ids, 0..bytes.len() / 4).unwrap(),
        keep: Keep::Last,
        adapter: None,
        positions: None,
        mask: None,
        overrides: Vec::new(),
    }
}
async fn tokens(device: &mut Device, buffer: furiosa_opt_engine::decoder::HbmTensor) -> Vec<i32> {
    let mut bytes = vec![0; buffer.buffer.size()];
    buffer
        .buffer
        .to_host(&mut device.pdma)
        .output(&mut bytes[..])
        .await
        .unwrap();
    bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
#[test]
#[ignore = "opens a device and loads the pinned checkpoint"]
fn continues_hello_with_w4a16_tokens() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let model = Instance::new(
        checkpoint().load::<ForCausalLM>(&mut device).unwrap(),
        Tiling::Adaptive,
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        for chunk in [1, 2] {
            let cache = Cache::new(&device, &model.limits().cache, 1).unwrap();
            let (mut cached, mut selected) = (0, 0);
            for ids in PROMPT.chunks(chunk) {
                let request = request(&mut device, ids, cached, &[0]).await;
                let output = model
                    .forward(&mut device, &cache, &[request], &[])
                    .await
                    .unwrap();
                selected = tokens(&mut device, output).await[0];
                cached += ids.len();
            }
            assert_eq!(selected, CONTINUATION[0], "prefill chunk {chunk}");
            let request = request(&mut device, &[selected], cached, &[0]).await;
            let output = model
                .forward(&mut device, &cache, &[request], &[])
                .await
                .unwrap();
            assert_eq!(
                tokens(&mut device, output).await,
                [CONTINUATION[1]],
                "decode after chunk {chunk}"
            );
        }
    });
}
