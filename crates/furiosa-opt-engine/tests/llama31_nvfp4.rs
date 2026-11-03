//! Pinned PR #457 W4A16 oracles through the composition engine. The reference uses
//! independent PyTorch projections with a late tensor scale, BF16 activations and BF16 KV.
use furiosa_opt_engine::decoder::{Cache, Instance, Keep, Request, Rows, Served};
use furiosa_opt_engine::{Checkpoint, Tiling};
use furiosa_opt_models::models::decoder::attention::llama::nvfp4::ForCausalLM;
use furiosa_opt_std::prelude::{Device, HostBufferSlice, Topology};
use serde_json::Value;
use std::{
    path::PathBuf,
    process::{Command, Stdio},
};

const REPO: &str = "nvidia/Llama-3.1-8B-Instruct-NVFP4";
const REVISION: &str = "bdb54e24298451af785c0ac63c1b485e9b7400a2";
const SHORT: &str = include_str!("fixtures/llama31_nvfp4_short.json");
const LONG: &str = include_str!("fixtures/llama31_nvfp4_long.json");
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
fn reference(fixture: &str) -> (Vec<i32>, Vec<i32>) {
    let reference: Value = serde_json::from_str(fixture).unwrap();
    let prompt = reference["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let expected = reference["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["token"].as_i64().unwrap() as i32)
        .collect();
    (prompt, expected)
}

#[test]
#[ignore = "opens a device and loads the pinned checkpoint"]
fn matches_w4a16_references_across_prefill_boundaries() {
    let checkpoint = checkpoint();
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let model = Instance::new(
        checkpoint.load::<ForCausalLM>(&mut device).unwrap(),
        Tiling::Adaptive,
    );
    assert_eq!((model.limits().window, model.limits().rows), (40960, 1024));
    assert_eq!(model.limits().cache.pools[0].bytes, 16 * 2 * 8 * 128 * 2);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        for (fixture, chunk_sizes) in [(SHORT, &[1, 2][..]), (LONG, &[128, 512, 1024][..])] {
            let (prompt, expected) = reference(fixture);
            for &chunk in chunk_sizes {
                let count = (prompt.len() + expected.len()).div_ceil(16);
                let blocks: Vec<u32> = (0..count as u32).rev().collect();
                let cache = Cache::new(&mut device, &model.limits().cache, count).unwrap();
                let (mut cached, mut selected) = (0, 0);
                for ids in prompt.chunks(chunk) {
                    let request = request(&mut device, ids, cached, &blocks).await;
                    let output = model
                        .forward(&mut device, &cache, &[request], &[])
                        .await
                        .unwrap();
                    selected = tokens(&mut device, output).await[0];
                    cached += ids.len();
                }
                let mut got = vec![selected];
                for _ in 1..expected.len() {
                    let request = request(&mut device, &[selected], cached, &blocks).await;
                    let output = model
                        .forward(&mut device, &cache, &[request], &[])
                        .await
                        .unwrap();
                    selected = tokens(&mut device, output).await[0];
                    got.push(selected);
                    cached += 1;
                }
                assert_eq!(got, expected, "{} prompt rows, chunk {chunk}", prompt.len());
            }
        }
        // A chunk followed by a decode item exercises host reordering, retained prefill
        // scratch, and live-row-only cache writes in the same model call.
        let (short, short_expected) = reference(SHORT);
        let (long, long_expected) = reference(LONG);
        let short_blocks = (short.len() + short_expected.len()).div_ceil(16) as u32;
        let long_blocks = (long.len() + long_expected.len()).div_ceil(16) as u32;
        let short_table: Vec<_> = (0..short_blocks).rev().collect();
        let long_table: Vec<_> = (short_blocks..short_blocks + long_blocks).rev().collect();
        let cache = Cache::new(
            &mut device,
            &model.limits().cache,
            (short_blocks + long_blocks) as usize,
        )
        .unwrap();
        for (prefix, table) in [(&short[..1], &short_table), (&long[..1023], &long_table)] {
            let request = request(&mut device, prefix, 0, table).await;
            model
                .forward(&mut device, &cache, &[request], &[])
                .await
                .unwrap();
        }
        let requests = [
            request(&mut device, &long[1023..], 1023, &long_table).await,
            request(&mut device, &short[1..], 1, &short_table).await,
        ];
        let output = model
            .forward(&mut device, &cache, &requests, &[])
            .await
            .unwrap();
        assert_eq!(
            tokens(&mut device, output).await,
            [long_expected[0], short_expected[0]]
        );
        let requests = [
            request(&mut device, &long_expected[..1], long.len(), &long_table).await,
            request(&mut device, &short_expected[..1], short.len(), &short_table).await,
        ];
        let output = model
            .forward(&mut device, &cache, &requests, &[])
            .await
            .unwrap();
        assert_eq!(
            tokens(&mut device, output).await,
            [long_expected[1], short_expected[1]]
        );
        // Seventeen different sequences exercise full cache-store packets and
        // a partial tail. Compare batching with separate requests, including the next decode
        // step so a misplaced cache write cannot hide behind the current output.
        let mut separate = Vec::new();
        for row in 0..17 {
            let cache = Cache::new(&mut device, &model.limits().cache, 1).unwrap();
            let prompt = [128000, 9906 + row];
            let initial = request(&mut device, &prompt, 0, &[0]).await;
            let output = model
                .forward(&mut device, &cache, &[initial], &[])
                .await
                .unwrap();
            let first = tokens(&mut device, output).await[0];
            let next = request(&mut device, &[first], 2, &[0]).await;
            let output = model
                .forward(&mut device, &cache, &[next], &[])
                .await
                .unwrap();
            separate.push((first, tokens(&mut device, output).await[0]));
        }
        let cache = Cache::new(&mut device, &model.limits().cache, 17).unwrap();
        for cached in 0..3 {
            let mut requests = Vec::new();
            for (row, &(first, _)) in separate.iter().enumerate() {
                let id = match cached {
                    0 => 128000,
                    1 => 9906 + row as i32,
                    _ => first,
                };
                requests.push(request(&mut device, &[id], cached, &[16 - row as u32]).await);
            }
            let output = model
                .forward(&mut device, &cache, &requests, &[])
                .await
                .unwrap();
            let found = tokens(&mut device, output).await;
            if cached > 0 {
                let expected: Vec<_> = separate
                    .iter()
                    .map(|&(first, second)| if cached == 1 { first } else { second })
                    .collect();
                assert_eq!(found, expected, "17-sequence batch, cached={cached}");
            }
        }
    });
}
