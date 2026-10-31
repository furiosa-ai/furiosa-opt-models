//! A call plans each cache group's attention operands under that group's sliding window. Opens a
//! device for the cache and the token rows; launches nothing.

use std::io;

use furiosa_opt_engine::decoder::{
    Batch, Cache, CacheSpec, Group, Keep, Kind, Limits, Pool, Request, Rows,
};
use furiosa_opt_models::ops::attention::full::paged::{Options, Plan, Tiling, Waves};
use furiosa_opt_models::specialize::attention::WAVES;
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
