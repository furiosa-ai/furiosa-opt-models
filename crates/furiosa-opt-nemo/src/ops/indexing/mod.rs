//! Select and store complete rows using device-resident byte offsets.

pub mod rows;

use crate::ops::layout::axes::Single;
use furiosa_opt_std::prelude::*;

/// Runtime list of individually retained contiguous rows.
pub type RowStorage<Width, D = f32> = Slice<HbmTensor<D, m![1], m![Width]>>;

/// Byte offset from the source allocation or tensor base.
pub type Offset<Chips = m![1]> = HbmTensor<i32, Chips, m![Single]>;
