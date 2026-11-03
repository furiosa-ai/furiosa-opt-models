//! Values that specialize attention, today's choices: how a decode attends each row's history,
//! and the capacities a decode launch's work items are compiled for; a launch's counts only bound
//! its loops within them.

#![deny(clippy::arbitrary_source_item_ordering)]

use std::marker::ConstParamTy;

use crate::tensor::cache::KeyValue;
use furiosa_opt_std::prelude::{AxisName, bf16};

/// How a decode call of a model is compiled today, by the model's size: the rows of a call, the
/// history each row attends, and the envelope one launch of the catalog model is compiled for.
pub trait Decode {
    /// The rows a call runs at most when each attends its whole history as a work item.
    type Rows: AxisName;
    /// The 512-key windows of history a row attends at most: the context the model is compiled
    /// for.
    type Windows: AxisName;
    /// The rows of a launch; a runtime count of them is live.
    type Capacity: AxisName;
    /// The rows attention reads and writes: a chunk's last query tile reaches up to one tile past
    /// the launch's rows.
    type Reach: AxisName;
    /// The prompt chunks of a launch.
    type Sequences: AxisName;
    /// The rotary tiles of a launch's rows.
    type Slots: AxisName;
}

/// How a decode attends each row's history in the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum Strategy {
    /// Over the windows its call's device preparation pages, each window split into shards whose
    /// softmax parts merge.
    Windows,
    /// As the work items its call's host plans, over the whole history.
    Items,
}

/// Waves of each domain of a decode launch's items.
pub const WAVES: usize = 128;

/// Wide waves of a launch whose 64 rows may each attend the whole 40,960-key history: three wide
/// items of one wave a row.
pub const HISTORY_WAVES: usize = 192;

/// Wide item slots, each an item's f32 output and log-sum-exp a merge reads.
pub const SLOTS: usize = 2048;

/// Wide rows a launch holds, each a merge job.
pub const DECODES: usize = 1024;

/// Items a merge combines into one row.
pub const INPUTS: usize = 64;

/// Rows of a row group, the lanes of a vector packet.
const LANES: usize = 8;

/// Row lanes a cluster holds: a row group's lanes split over the chip's two clusters.
const CLUSTER_LANES: usize = LANES / 2;

/// Slots each slice holds of a work item's landed history, a key and a value of `heads` KV heads
/// of `head` BF16 channels a slot: 64, or fewer so a landing takes at most a quarter of the slice's
/// data memory and the two a domain keeps in flight leave half of it to the rest of the layer.
pub(crate) const fn landing(head: usize, heads: usize) -> usize {
    let mut slots = 64;
    while slots * KeyValue::SIZE * heads * head * size_of::<bf16>() > super::DM / 4 {
        slots /= 2;
    }
    assert!(
        slots >= 16,
        "a landing holds whole 16-slot probability packets"
    );
    slots
}

/// Channels of a `head`-channel head one register fill of `keys` keys holds: all of them, or the
/// widest halving whose fill takes half the tensor registers, so the next fill lands beside it.
pub(crate) const fn channels(head: usize, keys: usize) -> usize {
    let mut width = head;
    while keys * width * size_of::<bf16>() > super::TRF / 2 {
        width /= 2;
    }
    assert!(
        width >= 32 && head.is_multiple_of(width),
        "a part holds whole 32-channel packets of the head"
    );
    assert!(
        LANES * width * size_of::<f32>() <= super::VRF / 2,
        "a row group's FP32 part fits half the vector registers"
    );
    width
}

/// Rows one pass of windowed attention attends at most: 32, or fewer so a slice's tensor registers
/// hold the `keys` keys of `head` BF16 channels it scores for each row group of the pass, its
/// vector registers the mask and the bias of those scores, `groups` query groups a row, and one
/// data memory page the FP32 attention of those groups, of which a window's fold keeps four.
pub(crate) const fn pass(head: usize, groups: usize, keys: usize) -> usize {
    let mut rows = 32;
    while rows / LANES * keys * head * size_of::<bf16>() > super::TRF
        || 2 * (rows / LANES) * groups * keys * size_of::<f32>() > super::VRF
        || rows / LANES * groups * head * size_of::<f32>() > super::DM / 16
    {
        rows /= 2;
    }
    assert!(rows >= LANES, "a pass attends a whole row group");
    rows
}

/// Whether a pass of `rows` rows over an `envelope` lands a window while the one before it is
/// attended: when two landings, a fold and the envelope's rows leave a quarter of DM to the layer.
pub(crate) const fn overlaps(
    head: usize,
    groups: usize,
    keys: usize,
    (rows, envelope): (usize, usize),
) -> bool {
    let landings = 2 * (rows / LANES) * KeyValue::SIZE * keys * head * size_of::<bf16>();
    let folds = 4 * (rows / LANES) * groups * head * size_of::<f32>();
    let envelope = 2 * envelope.div_ceil(LANES) * groups * head * size_of::<bf16>();
    landings + folds + envelope <= 3 * super::DM / 4
}

/// Shards a window of `keys` keys splits into, one a slice: as many as a cluster's slices hold
/// for each of `heads` heads and each of its row lanes, each shard at least one `block` of keys.
pub(crate) const fn shards(heads: usize, keys: usize, block: usize) -> usize {
    assert!(
        heads > 0 && super::SLICES.is_multiple_of(heads * CLUSTER_LANES),
        "a cluster's slices hold whole heads of every row lane"
    );
    let fill = super::SLICES / (heads * CLUSTER_LANES);
    let whole = keys / block;
    let shards = if fill < whole { fill } else { whole };
    assert!(
        shards > 0 && keys.is_multiple_of(shards * block),
        "a shard holds whole blocks of the window"
    );
    shards
}

/// Heads a cluster's slices hold at `shards` shards of each of its row lanes.
pub(crate) const fn heads(shards: usize) -> usize {
    super::SLICES / (CLUSTER_LANES * shards)
}
