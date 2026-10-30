//! Values that specialize attention, today's choices: how a decode attends each row's history,
//! and the capacities a decode launch's work items are compiled for; a launch's counts only bound
//! its loops within them.

#![deny(clippy::arbitrary_source_item_ordering)]

use std::marker::ConstParamTy;

use furiosa_opt_std::prelude::{AxisName, M};

/// How a decode call of a model is compiled today, by the model's size: the rows of a call, the
/// history each row attends, the packed query, key and value weight's rows and layout, the work a
/// call's host plans, and the envelope one launch of the catalog model is compiled for.
pub trait Decode {
    /// The rows a call runs at most when each attends its whole history as a work item.
    type Rows: AxisName;
    /// The 512-key windows of history a row attends at most: the context the model is compiled
    /// for.
    type Windows: AxisName;
    /// The rows of the packed query, key and value weight.
    type Width: AxisName;
    /// The packed weight's layout: `[Width, Hidden]`, or each key and value head's rows together.
    type Packed: M;
    /// The work items a call's host plans.
    type Work;
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
