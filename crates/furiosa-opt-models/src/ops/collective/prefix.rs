//! Whole compact-prefix exchanges inside one capacity and one runtime operation.

use super::{ChipMap, Exchange, O, Pieces, Rows};
use crate::tensor::{
    live::Live,
    parallel::{Parallel, Part, Shard},
};
use furiosa_opt_std::prelude::*;

/// A nonempty, increasing list of fixed DMA extents. `(S,)` is the final extent;
/// `(S, Tail)` prepends one extent. The model supplies this type once for its envelope.
pub trait Extents {
    /// The first extent in this list.
    type First: AxisName;
    /// The remaining extents, or this same terminal list.
    type Rest: Extents;
    /// Whether this list contains only its final extent.
    const LAST: bool;
}

impl<S: AxisName> Extents for (S,) {
    type First = S;
    type Rest = Self;
    const LAST: bool = true;
}

impl<S: AxisName, Tail: Extents> Extents for (S, Tail) {
    type First = S;
    type Rest = Tail;
    const LAST: bool = false;
}

impl<const N: usize> Exchange<Parallel<N>> {
    /// Exchange one initialized prefix with one whole TDMA copy, preserving destination tails.
    /// The Live interval must start at zero and its row count must be one of `G`'s extents.
    /// Source parts use that compact row count as their stride, within allocation capacity `C`.
    /// Only the DMA mapping varies internally; callers use the same operation for every request.
    pub fn write_prefix<D: MaterializableScalar, C: AxisName, W: AxisName, G: Extents>(
        &self,
        device: &mut Device,
        input: &Live<&Rows<D, N, Pieces<C, N>, Part<W, N>>, C>,
        output: &mut Rows<D, N, C, W>,
    ) {
        const {
            assert!(G::First::SIZE > 0 && G::First::SIZE <= C::SIZE);
            assert!(G::LAST || G::First::SIZE < <G::Rest as Extents>::First::SIZE);
        }
        if G::LAST || input.rows() == G::First::SIZE {
            self.write_extent::<D, G::First, C, W>(device, input.value(), output);
        } else {
            self.write_prefix::<D, C, W, G::Rest>(device, input, output);
        }
    }

    fn write_extent<D: MaterializableScalar, S: AxisName, C: AxisName, W: AxisName>(
        &self,
        device: &mut Device,
        input: &Rows<D, N, Pieces<C, N>, Part<W, N>>,
        output: &mut Rows<D, N, C, W>,
    ) {
        const { assert!(N > 0 && W::SIZE > 0 && S::SIZE > 0 && S::SIZE <= C::SIZE) };
        type L<W, const N: usize> = Symbol<Part<W, N>>;
        // SAFETY: the first alias keeps the full physical extent and declares explicit outer
        // padding. The second names only that initialized contiguous prefix, preserving byte
        // order. Neither alias writes, and the original input owner stays borrowed throughout.
        let padded: PaddedRows<D, N, S, C, W> = unsafe { input.view().alias() };
        let compact: Rows<D, N, Pieces<S, N>, Part<W, N>> = unsafe { padded.view().alias() };
        // SAFETY: the compact parts are in chip order, each its rows in order.
        let source = unsafe {
            compact
                .view()
                .reshape::<ChipMap<N>, m![{ O<N> }, S, { L<W, N> }]>()
        };
        // SAFETY: this declares only outer padding; the physical extent and row-major byte
        // order remain C * W. The selected prefix keeps the original mutable borrow.
        let destination = unsafe { output.view_mut().reshape::<ChipMap<N>, m![S # C, W]>() };
        // SAFETY: a row's columns are the source chips' contiguous parts. Retain the outer
        // allocation padding so the view keeps the original destination's writable provenance.
        let destination = unsafe {
            destination.reshape::<m![{ O<N> }], m![S # C, { Symbol<Shard<N>> }, { L<W, N> }]>()
        };
        source.to_hbm_view(&mut device.tdma, destination);
    }
}

type PaddedRows<D, const N: usize, S, C, W> = HbmTensor<
    D,
    ChipMap<N>,
    m![{ Symbol<Pieces<S, N>> } # ({ Symbol<Pieces<C, N>> }), { Symbol<Part<W, N>> }],
>;

/// Whether every DMA extent preserves a consumer's block alignment.
pub(crate) const fn aligned<G: Extents>(block: usize) -> bool {
    G::First::SIZE.is_multiple_of(block) && (G::LAST || aligned::<G::Rest>(block))
}
