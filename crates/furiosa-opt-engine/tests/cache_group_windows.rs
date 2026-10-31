//! Calls over several cache groups: each group's attention operands planned under that group's
//! sliding window, each group's history and the rows a launch returns staged as Gemma 4's launch
//! reads them, each group's window as its paging binds it from the config, and Gemma 4's sliding
//! and full layers served in one launch a call, whose greedy tokens continue a prompt as the
//! reference's do, under the adaptive tiling a load holds it to.

use std::io;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use furiosa_opt_engine::decoder::{
    Batch, Cache, CacheSpec, Group, Groups, Instance, Keep, Kind, Limits, Pool, Request, Root,
    Rows, Served, Table,
};
use furiosa_opt_engine::registry::Implementation;
use furiosa_opt_engine::{Checkpoint, LoadError, Tiling};
use furiosa_opt_models::load::Context;
use furiosa_opt_models::models::decoder::attention::gemma4;
use furiosa_opt_models::models::decoder::launch::Tables;
use furiosa_opt_models::ops::attention::full::paged::{Options, Plan, Waves};
use furiosa_opt_models::ops::attention::gqa;
use furiosa_opt_models::ops::index::table::{BlockIndex, FIELDS};
use furiosa_opt_models::specialize::attention::WAVES;
use furiosa_opt_models::tensor::{cache::Block, parallel::Chip};
use furiosa_opt_std::prelude::*;

/// Each cache group's operands attend that group's window: a sliding group's decode rows and
/// prompt chunks read only the blocks from the one holding each row's first window key, where
/// a whole-history group's, over the same blocks, read from block 0.
#[test]
fn plans_each_group_under_its_window() -> io::Result<()> {
    axes![
        Capacity = 64,
        Reach = 128,
        Sequences = 2,
        KvHeads = 8,
        Groups = 4,
        Head = 128,
        Windows = 80
    ];
    const WINDOW: usize = 1024;
    let device = Device::new(Topology { chips: 1, pes: 8 }).map_err(io::Error::other)?;
    let group = |layer, window| Group {
        pool: 0,
        layers: vec![layer],
        kind: Kind::Paged,
        window,
    };
    let limits = Limits {
        window: 4096,
        rows: 8,
        cache: CacheSpec {
            pools: vec![Pool {
                block: 16,
                bytes: 64,
            }],
            groups: vec![group(0, None), group(1, Some(WINDOW))],
        },
        adapters: 0,
    };
    let cache = Cache::new(&device, &limits.cache, 400)?;
    // A decode row at position 2063, then a prompt of four rows after 2000 cached positions,
    // each naming the same blocks for both groups.
    let requests = [
        ((0..129).collect::<Vec<u32>>(), 2063, 1),
        ((200..326).collect(), 2000, 4),
    ];
    let requests = (requests.iter())
        .map(|(blocks, cached, count)| {
            let bytes = count * Rows::<i32, m![1]>::bytes();
            let buffer = HbmBuffer::alloc(&device, bytes).map_err(io::Error::other)?;
            Ok(Request {
                rows: vec![blocks.clone(), blocks.clone()],
                cached: *cached,
                tokens: Rows::new(&buffer, 0..*count)?,
                keep: Keep::Last,
                adapter: None,
                positions: None,
                mask: None,
                overrides: Vec::new(),
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    let batch = Batch::<()>::new(&requests, &limits, &cache, &[])?;
    let operands = |group| {
        batch.operands::<Capacity, Reach, Sequences, KvHeads, Groups, Head, Waves<WAVES>, Windows>(
            group,
            Tiling::Adaptive,
        )
    };
    let (whole, sliding) = (operands(0)?, operands(1)?);
    let mut history: Vec<i32> = (0..129).collect();
    history.resize(144, 128);
    let items = |window| {
        let options = Options {
            tiling: Tiling::Adaptive,
            window,
        };
        let plan = Plan::new::<KvHeads, Groups, Head, Waves<WAVES>, Reach>(
            &[[0, 1, 2063, 0, 129]],
            &history,
            options,
        );
        (plan.reads())
            .map(|(values, words)| (values.to_vec(), words))
            .collect::<Vec<_>>()
    };
    let (all, windowed) = (items(None), items(Some(WINDOW)));
    assert_ne!(all, windowed, "the decode row's window binds");
    let count = all.len();
    assert_eq!(whole[2..2 + count], all);
    assert_eq!(sliding[2..2 + count], windowed);
    // The chunk's history: all 2000 keys, or from block 61, holding its first row's first
    // window key 977, on.
    let history = |arrays: &[(Vec<i32>, usize)]| arrays[2 + count + 1].0[0];
    assert_eq!((history(&whole), history(&sliding)), (2000, 2000 - 61 * 16));
    Ok(())
}

/// What a call of Gemma 4's decoder is planned within: 64 rows up to the full group's context, the
/// sliding group attending the positions of all but one of its cache windows, both paged in blocks
/// of 16 positions; one layer a group, of pages no plan reads.
fn limits() -> Limits {
    let positions = |windows: usize| windows * gqa::Cache::SIZE;
    let group = |layer, window| Group {
        pool: 0,
        layers: vec![layer],
        kind: Kind::Paged,
        window,
    };
    Limits {
        window: positions(gemma4::full::Windows::SIZE),
        rows: gemma4::Rows::SIZE,
        cache: CacheSpec {
            pools: vec![Pool {
                block: Block::SIZE,
                bytes: 64,
            }],
            groups: vec![
                group(0, Some(positions(gemma4::sliding::Windows::SIZE - 1))),
                group(1, None),
            ],
        },
        adapters: 0,
    }
}

/// A request that appends `count` rows after `cached` positions on `blocks` in both groups,
/// keeping its last; a plan reads none of its token IDs.
fn appending(
    device: &Device,
    blocks: Vec<u32>,
    (cached, count): (usize, usize),
) -> io::Result<Request> {
    let bytes = count * Rows::<i32, m![1]>::bytes();
    let buffer = HbmBuffer::alloc(device, bytes).map_err(io::Error::other)?;
    Ok(Request {
        rows: vec![blocks.clone(), blocks],
        cached,
        tokens: Rows::new(&buffer, 0..count)?,
        keep: Keep::Last,
        adapter: None,
        positions: None,
        mask: None,
        overrides: Vec::new(),
    })
}

/// Checks that group `group` of `limits` stages the history of `batch`, a call of `requests` in
/// call row order, in a table of `B` block IDs: each segment keeps its rows and cached positions,
/// and every block its rows read lies at its table start plus the block's index in its request.
fn reads<B: AxisName>(
    batch: &Batch<'_, ()>,
    (group, limits): (usize, &Limits),
    requests: &[Request],
) -> io::Result<()> {
    let [(segments, _), (blocks, _), ..] =
        batch.history::<gemma4::Rows, B>((group, Table::Blocks))?;
    let window = limits.cache.groups[group].window;
    let word = |value: usize| i32::try_from(value).unwrap();
    let mut start = 0;
    for (request, segment) in requests.iter().zip(segments.chunks_exact(FIELDS)) {
        let rows = request.tokens.len();
        let fields = [start, rows, request.cached].map(word);
        assert_eq!(segment[..3], fields, "group {group}, row {start}");
        // The device reads from the cache window holding the first row's first visible position.
        let low = window.map_or(0, |window| (request.cached + 1).saturating_sub(window));
        let first = low / gqa::Cache::SIZE * BlockIndex::SIZE;
        for block in first..(request.cached + rows).div_ceil(Block::SIZE) {
            let entry = usize::try_from(segment[3] + word(block)).unwrap();
            let id = word(request.rows[group][block] as usize);
            assert_eq!(
                blocks[entry], id,
                "group {group}, row {start}, block {block}"
            );
        }
        start += rows;
    }
    Ok(())
}

/// Each group's history of a call the decoder admits fits the table its launch is compiled for,
/// at any position: 64 decode rows past the positions of the sliding table's windows, and one
/// request whose rows end at the context's last position.
#[test]
fn stages_each_admitted_history_within_its_table() -> io::Result<()> {
    let device = Device::new(Topology { chips: 1, pes: 8 }).map_err(io::Error::other)?;
    let limits = limits();
    let (rows, blocks) = (limits.rows, limits.window / Block::SIZE);
    let cache = Cache::new(&device, &limits.cache, blocks)?;
    let ids = |range: std::ops::Range<usize>| -> Vec<u32> { range.map(|id| id as u32).collect() };
    let position = gemma4::sliding::Windows::SIZE * gqa::Cache::SIZE;
    let used = (position + 1).div_ceil(Block::SIZE);
    let decode = (0..rows)
        .map(|row| appending(&device, ids(row * used..(row + 1) * used), (position, 1)))
        .collect::<io::Result<Vec<_>>>()?;
    let prompt = appending(&device, ids(0..blocks), (limits.window - rows, rows))?;
    for requests in [decode, vec![prompt]] {
        let batch = Batch::<()>::new(&requests, &limits, &cache, &[])?;
        reads::<Tables<gemma4::Rows, gemma4::sliding::Windows>>(&batch, (0, &limits), &requests)?;
        reads::<Tables<gemma4::Rows, gemma4::full::Windows>>(&batch, (1, &limits), &requests)?;
    }
    Ok(())
}

/// The I32 words of `buffer`, read back once the writes submitted before it completed.
async fn words(device: &mut Device, buffer: &HbmBuffer) -> io::Result<Vec<i32>> {
    let mut host = vec![0; buffer.size()];
    (buffer.to_host(&mut device.pdma).output(&mut host[..]).await).map_err(io::Error::other)?;
    Ok((host.as_chunks::<4>().0.iter())
        .map(|&word| i32::from_le_bytes(word))
        .collect())
}

/// A call stages a row's byte offset for each of the rows its launch returns, over memory that
/// held other words: the kept rows' in order, then the first row's.
#[test]
fn stages_a_row_offset_for_every_returned_row() -> io::Result<()> {
    const ROW: usize = gemma4::Hidden::SIZE * size_of::<bf16>();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).map_err(io::Error::other)?;
        let limits = limits();
        let cache = Cache::new(&device, &limits.cache, 8)?;
        let keeping = |block, rows, keep| -> io::Result<Request> {
            Ok(Request {
                keep,
                ..appending(&device, vec![block], rows)?
            })
        };
        // A decode row and a prompt keeping their last rows, a prompt keeping none, and one
        // keeping two of its rows.
        let calls = [
            (
                vec![
                    keeping(3, (10, 1), Keep::Last)?,
                    keeping(5, (0, 4), Keep::Last)?,
                ],
                vec![0, 4],
            ),
            (vec![keeping(5, (0, 3), Keep::None)?], vec![]),
            (vec![keeping(5, (0, 4), Keep::Range(1, 3))?], vec![1, 2]),
        ];
        for (requests, kept) in calls {
            let batch = Batch::<()>::new(&requests, &limits, &cache, &[])?;
            let returned = batch.returned::<gemma4::Rows>(ROW)?;
            // Memory the staging buffer takes next, first fit, holding words that are no offset.
            let poison = vec![0xa5_u8; 1 << 20];
            let held = (HostBufferSlice::from(&poison[..])
                .to_hbm(&mut device.pdma)
                .await)
                .map_err(io::Error::other)?;
            drop(held);
            let (_, views) = batch
                .stage::<gemma4::Rows, Tables<gemma4::Rows, gemma4::sliding::Windows>, Chip>(
                    &mut device,
                    (&[returned], &[]),
                    (0, Table::Blocks),
                )
                .await?;
            let offset = |row: &usize| i32::try_from(row * ROW).unwrap();
            let mut expected: Vec<i32> = kept.iter().map(offset).collect();
            expected.resize(gemma4::Rows::SIZE, 0);
            assert_eq!(words(&mut device, &views[0]).await?, expected, "{kept:?}");
        }
        Ok(())
    })
}

/// Gemma 4's pagings of its two cache groups, bound from a config whose sliding layers attend back
/// `window` positions and whose groups turn as the published checkpoint's do.
fn pagings(device: &mut Device, window: usize) -> <gemma4::ForCausalLM as Root>::Groups {
    let config = serde_json::json!({
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
            "full_attention": {
                "rope_type": "proportional", "rope_theta": 1e6, "partial_rotary_factor": 0.25,
            },
        },
        "sliding_window": window, "max_position_embeddings": 262144,
    });
    let empty = tempfile::tempdir().unwrap();
    Checkpoint::new(empty.path(), Some(config))
        .unwrap()
        .load(device)
        .unwrap()
}

/// Each cache group reports the window its paging binds from the config: the sliding group the
/// config's `sliding_window`, the published one or one below what its cache windows hold, the full
/// group none.
#[test]
fn reports_each_group_window_its_config_binds() -> io::Result<()> {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).map_err(io::Error::other)?;
    for window in [1024, 512] {
        let pagings = pagings(&mut device, window);
        let limits = Groups::<gemma4::ForCausalLM>::limits(&pagings, Tiling::Adaptive);
        let windows: Vec<_> = (limits.cache.groups.iter())
            .map(|group| group.window)
            .collect();
        assert_eq!(windows, [Some(window), None], "sliding_window {window}");
    }
    Ok(())
}

/// The published Gemma 4 checkpoint at the hub revision this test reads.
const GEMMA4: (&str, &str) = (
    "unsloth/gemma-4-12b-it-NVFP4",
    "b1f649734b34aa5575b03d186abd1b9be3d0d5c4",
);

/// "What is 2 + 2? Answer with just the number." in Gemma 4's chat template, in its tokens.
const PROMPT: [i32; 27] = [
    2, 105, 2364, 107, 3689, 563, 236743, 236778, 900, 236743, 236778, 236881, 25685, 607, 1164,
    506, 1548, 236761, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
];

/// The quantization reference's greedy continuation of `PROMPT`: the vLLM plugin's
/// `test_generation.reference_model` of the checkpoint, `generate(max_new_tokens=2, do_sample=False)`.
const CONTINUATION: [i32; 2] = [236812, 106];

/// Each group's history blocks, out of the eight the cache holds, in history order.
const BLOCKS: [u32; 3] = [5, 2, 7];

/// A second sequence's history blocks, none of which `BLOCKS` holds.
const OTHER: [u32; 3] = [1, 3, 4];

/// The directory of the hub checkpoint's files that `include` matches, fetched into the hub cache
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

/// A request of `tokens` appended after the `cached` positions of both groups' histories on
/// `blocks`, keeping the last.
async fn request(
    device: &mut Device,
    (blocks, cached): (&[u32], usize),
    tokens: &[i32],
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
        rows: vec![blocks.to_vec(), blocks.to_vec()],
        cached,
        tokens: Rows::new(&ids, 0..tokens.len()).unwrap(),
        keep: Keep::Last,
        adapter: None,
        positions: None,
        mask: None,
        overrides: Vec::new(),
    }
}

/// The tokens `model` selects for the last rows of `requests`, appended in one call.
async fn kept(
    device: &mut Device,
    model: &dyn Served,
    cache: &Cache,
    requests: &[Request],
) -> Vec<i32> {
    let output = model.forward(device, cache, requests, &[]).await.unwrap();
    assert_eq!(output.shape, [requests.len()]);
    words(device, &output.buffer).await.unwrap()
}

/// The token `model` selects for the last of `tokens` appended after `cached` positions, the
/// call's one request.
async fn next(
    device: &mut Device,
    model: &dyn Served,
    cache: &Cache,
    (cached, tokens): (usize, &[i32]),
) -> i32 {
    let request = request(device, (&BLOCKS, cached), tokens).await;
    kept(device, model, cache, &[request]).await[0]
}

/// A load of Gemma 4's decoder for calls under the fixed tiling, each row's values whatever rows
/// share its call, is refused before it reads a weight: from the published config alone, a load
/// under the adaptive tiling goes on to find no weight.
#[test]
fn refuses_a_fixed_tiling_before_reading_a_weight() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let config = std::fs::read(snapshot(GEMMA4, &["config.json"]).join("config.json")).unwrap();
    let empty = tempfile::tempdir().unwrap();
    let config = serde_json::from_slice(&config).unwrap();
    let source = Checkpoint::new(empty.path(), Some(config)).unwrap();
    let implementation = Implementation::of::<gemma4::ForCausalLM>();
    let mut load = |tiling| {
        let context = &mut Context::new(&mut device);
        implementation.load(context, &source, tiling).map(drop)
    };
    match load(Tiling::Fixed) {
        Err(LoadError::Unsupported { name, .. }) => assert_eq!(name, "batch_invariant"),
        other => panic!("{other:?} is not `Unsupported`"),
    }
    match load(Tiling::Adaptive) {
        Err(LoadError::Missing(_)) => {}
        other => panic!("{other:?} is not `Missing`: the checkpoint holds no weight"),
    }
}

/// Gemma 4's decoder, its sliding layers cached in one group and its full layers in another,
/// continues the prompt with the reference's greedy tokens: after the prompt in one call and a
/// decode step, after the same tokens one position a call, and as each of two sequences a call.
#[test]
#[ignore = "reads the published checkpoint from the local hub cache"]
fn continues_a_prompt_over_both_groups_as_the_reference() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let source = Checkpoint::new(snapshot(GEMMA4, &["*.json", "*.safetensors"]), None).unwrap();
    let root: gemma4::ForCausalLM = source.load(&mut device).unwrap();
    let instance = Instance::new(root, Tiling::Adaptive);
    let model: &dyn Served = &instance;
    let limits = model.limits();
    let [sliding, full] = &limits.cache.groups[..] else {
        panic!("{:?}", limits.cache.groups);
    };
    assert_eq!(
        (sliding.layers.clone(), sliding.window),
        (
            (0..48).filter(|layer| (layer + 1) % 6 != 0).collect(),
            Some(1024)
        )
    );
    assert_eq!(
        (full.layers.clone(), full.window),
        ((0..48).filter(|layer| (layer + 1) % 6 == 0).collect(), None)
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let cache = Cache::new(&device, &limits.cache, 8).unwrap();
        let first = next(&mut device, model, &cache, (0, &PROMPT)).await;
        let second = next(&mut device, model, &cache, (PROMPT.len(), &[first])).await;
        assert_eq!([first, second], CONTINUATION, "the prompt in one call");
        let mut stepped = Vec::new();
        for position in 0..PROMPT.len() + 1 {
            let token = PROMPT.get(position).copied().unwrap_or_else(|| stepped[0]);
            let token = next(&mut device, model, &cache, (position, &[token])).await;
            if position + 1 >= PROMPT.len() {
                stepped.push(token);
            }
        }
        assert_eq!(stepped, CONTINUATION, "one position a call");
        let mut prompts = Vec::new();
        for blocks in [BLOCKS, OTHER] {
            prompts.push(request(&mut device, (&blocks, 0), &PROMPT).await);
        }
        let firsts = kept(&mut device, model, &cache, &prompts).await;
        let mut steps = Vec::new();
        for (blocks, &token) in [BLOCKS, OTHER].iter().zip(&firsts) {
            steps.push(request(&mut device, (blocks, PROMPT.len()), &[token]).await);
        }
        let seconds = kept(&mut device, model, &cache, &steps).await;
        let continued: Vec<_> = firsts
            .into_iter()
            .zip(seconds)
            .map(|(a, b)| [a, b])
            .collect();
        assert_eq!(continued, [CONTINUATION; 2], "two sequences a call");
    });
}
