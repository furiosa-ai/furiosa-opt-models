//! One served four-chip text layer with its causal positions and caller-owned KV cache.

use super::super::packing::DEFAULT_MAX_TEXT;
use super::{self as text, Head, KvHeads};
use crate::Module;
use crate::ops::attention::gqa;
use crate::tensor::{
    axis::{Rows, Single},
    cache::Blocks,
    memory::Hbm,
    parallel::{Parallel, Parallelism},
};
use furiosa_opt_std::prelude::*;

/// One layer borrows its weights and the shared rotary position constants.
#[derive(DeviceSend)]
pub struct Layer<'a, 'w: 'a> {
    /// The partitioned pre-norm layer weights.
    pub layer: &'a text::Layer<P, &'w Hbm>,
    /// The full-head rotary position calculation shared by every layer.
    pub positions: &'a text::position::Positions<&'w Hbm>,
}

impl Module<Hidden, &Row, &mut Cache> for Layer<'_, '_> {
    type Output = Hidden;

    fn forward(&self, device: &mut Device, row: &Row, cache: &mut Cache, input: Hidden) -> Hidden {
        let rotation = self.positions.forward(device, (), &mut (), &row.position);
        let mut state = (
            (
                ((((), ()), ((), ()), ((), ())), (), ()),
                (cache, &row.slot, &row.history),
                (),
            ),
            ((((), ()), ()), ((), ())),
        );
        self.layer
            .forward(device, (&rotation, &()), &mut state, input)
    }
}

/// A row's metadata aliases the prompt's one allocation. Every table entry, including rounded
/// overfetch, addresses allocated cache storage; the causal bias hides unfilled token rows.
/// Constructed only by the text request's checked preparation and upload.
#[derive(DeviceSend)]
pub struct Row {
    /// The row's replicated full-head position coordinates.
    pub(crate) position: HbmTensor<f32, Chips, m![Single, Head]>,
    /// The row's write address in the layer's KV cache.
    pub(crate) slot: HbmTensor<i32, Chips, m![1]>,
    /// The allocated cache blocks and causal visibility for this row.
    pub(crate) history: gqa::Read<Windows, Chips>,
    /// The replicated vocabulary ID read by the embedding launch.
    pub(crate) token: HbmTensor<i32, Chips, m![1]>,
}

impl Row {
    /// Borrow the checked vocabulary ID for the separate embedding launch.
    pub fn token(&self) -> &HbmTensor<i32, Chips, m![1]> {
        &self.token
    }
}

/// Paged BF16 KV storage partitioned over the four served chips.
pub type Cache = Blocks<LocalKv, Head, bf16, Chips>;

/// One BF16 hidden row replicated over the four served chips.
pub type Hidden = HbmTensor<bf16, Chips, m![{ Symbol<text::Hidden> }]>;

/// Causal attention windows covering the served maximum text capacity.
pub type Windows = Rows<{ DEFAULT_MAX_TEXT.div_ceil(gqa::Cache::SIZE) }>;

type P = Parallel<4>;
type Chips = <P as Parallelism>::Chip;
type LocalKv = <P as Parallelism>::Local<KvHeads>;
