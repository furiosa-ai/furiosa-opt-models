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
pub(crate) const LANES: usize = 8;

/// Row lanes a cluster holds: a row group's lanes split over the chip's two clusters.
pub(crate) const CLUSTER_LANES: usize = LANES / 2;

/// Shard slices a ring carries a pass's query rows over at most: a ring takes about half a
/// microsecond a slice whatever it carries, so a head of more shards lands a copy a ring.
pub(crate) const RING: usize = 8;

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

/// Rows a slice attends in one pass of windowed attention at most: 32, or fewer so its tensor
/// registers hold the `keys` keys of `head` BF16 channels it scores for each of its row groups, its
/// vector registers the mask and the bias of those scores, `groups` query groups a row, and one
/// data memory page the FP32 attention of those groups, of which a window's fold keeps four.
pub(crate) const fn pass(head: usize, groups: usize, keys: usize) -> usize {
    let mut rows = 32;
    while rows / LANES * keys * head * size_of::<bf16>() > super::TRF
        || 2 * (rows / LANES) * groups * keys * size_of::<f32>() > super::VRF
        || rows / LANES * groups * head * size_of::<f32>() > super::PAGE
    {
        rows /= 2;
    }
    assert!(rows >= LANES, "a pass attends a whole row group");
    rows
}

/// The copies of a head's slot, of `copies`, that a pass of `groups` row groups fills, a row group
/// a copy; each of its slices holds the rest of its row groups.
pub(crate) const fn slots(groups: usize, copies: usize) -> usize {
    if groups < copies { groups } else { copies }
}

/// Whether a history of `windows` windows holds every window's softmax terms until one merge
/// combines them, rather than folding each into a running attention: when it has a fold to save and
/// the FP32 terms of `count` row groups of `groups` query groups, each window's, fit one data memory
/// page.
pub(crate) const fn holds(head: usize, groups: usize, count: usize, windows: usize) -> bool {
    windows > 1 && windows * count * groups * (head + LANES) * size_of::<f32>() <= super::PAGE
}

/// Whether a window's keys and values stream past its `groups` query rows of `head` channels and
/// their softmax weights in the tensor registers, rather than fill the registers themselves: when
/// the groups fit the register lanes and the weighted sum's accumulators hold each lane's whole
/// head while the keys stream.
pub(crate) const fn streams(head: usize, groups: usize) -> bool {
    groups.is_power_of_two() && groups <= LANES && groups * head <= super::ACCUMULATOR
}

/// Whether a pass of `rows` rows over an `envelope` lands a window while the one before it is
/// attended: when two landings, a fold, the envelope's attended rows and the `copies` of its query
/// rows that data memory holds leave a quarter of DM to the layer.
pub(crate) const fn overlaps(
    head: usize,
    groups: usize,
    keys: usize,
    pass: (usize, usize, usize),
) -> bool {
    let (landing, small) = held(head, groups, keys, pass, size_of::<bf16>());
    2 * landing + small <= 3 * super::DM / 4
}

/// Whether such a pass, streaming windows of `element`-byte codes, leaves data memory for `places`
/// beside what it holds, counted in pages as the allocator gives them: each landing and each of the
/// places takes whole pages, while the pass's smaller places share pages with the layer's, which
/// take an eighth of DM.
pub(crate) const fn beside(
    head: usize,
    groups: usize,
    keys: usize,
    pass: (usize, usize, usize),
    (element, places): (usize, &[usize]),
) -> bool {
    let (landing, small) = held(head, groups, keys, pass, element);
    let mut pages =
        2 * landing.div_ceil(super::PAGE) + (small + super::DM / 8).div_ceil(super::PAGE);
    let mut place = 0;
    while place < places.len() {
        pages += places[place].div_ceil(super::PAGE);
        place += 1;
    }
    streams(head, groups) && pages <= super::DM / super::PAGE
}

/// Data memory such a pass holds with windows of `element`-byte codes: a window's landing, of which
/// it keeps two, and its smaller places: a fold, the envelope's attended rows and the copies of its
/// query rows.
const fn held(
    head: usize,
    groups: usize,
    keys: usize,
    (rows, envelope, copies): (usize, usize, usize),
    element: usize,
) -> (usize, usize) {
    let landing = (rows / LANES) * KeyValue::SIZE * keys * head * element;
    let folds = 4 * (rows / LANES) * groups * head * size_of::<f32>();
    let envelope = (1 + copies) * envelope.div_ceil(LANES) * groups * head * size_of::<bf16>();
    (landing, folds + envelope)
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

/// Windows of a lone row its cluster's cells hold at once, one a cell: all of its `windows` where
/// whole rows' windows fill the cluster's lanes; else, where a history may span as many, its first
/// [`LANES`], which the lanes of a head's two slots hold; else one.
pub(crate) const fn lanes(windows: usize) -> usize {
    if windows > 1 && CLUSTER_LANES.is_multiple_of(windows) {
        windows
    } else if windows >= LANES {
        LANES
    } else {
        1
    }
}

/// Whether a pass holds a lone row's every window at once, one a lane: where the lanes hold them
/// and each cell is `element` bytes, narrower than BF16, as landing them together then costs less
/// than attending them in turn saves; a BF16 window's landing costs about as much.
pub(crate) const fn spreads(windows: usize, element: usize) -> bool {
    lanes(windows) > 1 && element < size_of::<bf16>()
}
