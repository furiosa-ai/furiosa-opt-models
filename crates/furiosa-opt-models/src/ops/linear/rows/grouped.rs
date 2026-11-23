//! K-grouped BF16 slabs for the register-resident row contraction.
//!
//! Coefficients use `[O / 32, I / K, O % 32, I % K]`, with `K` chosen for a
//! 256-row projection. Each group holds 32 contiguous K-element output chunks;
//! each pass lands its output-column slab in DM through TDMA. The contraction
//! is shared with ordinary row projections.

use super::{
    B, Blocked, Bs, Chunked, Cluster, Hr, KeptOutputs, Ks, Np, PassOutputs, Result, Rows, Sink,
    Slab, Weight,
};
use crate::tensor::form::grouped::Layout;
use furiosa_opt_std::prelude::*;

type Tile = crate::tensor::axis::Rows<256>;

/// A packed tensor borrowed for the duration of a projection.
pub(crate) struct Packed<'a, I: AxisName, O: AxisName, Chips: M>(
    pub(crate) &'a HbmTensor<bf16, Chips, Layout<I, O>>,
);

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M> Slab<I, O, R, Chips>
    for Packed<'_, I, O, Chips>
{
    const PIPELINED: bool = true;

    fn slab(&self, device: &mut Device, pass: usize) -> Weight<I, O, R, Chips> {
        const {
            assert!(
                <Chunked<I, O, R> as AxisName>::SIZE == <Chunked<I, O, Tile> as AxisName>::SIZE,
                "the rows' K groups differ from the packed weight's"
            );
            assert!(<Np<I, O, R> as M>::SIZE.is_multiple_of(32));
        };
        self.0
            .view()
            .tile::<m![O / ({ Np<I, O, R> })], 1, m![
                O / ({ Np<I, O, R> }) = 1 # (O / ({ Np<I, O, R> })),
                O % ({ Np<I, O, R> }) / 32,
                I / ({ Ks<I, O, Tile> }),
                O % 32,
                I % ({ Ks<I, O, Tile> })
            ]>(pass)
            .to_dm(&mut device.tdma)
    }
}

/// Row-major grouped outputs, staged into whole HBM write units when a slice's output run is
/// shorter than a unit. Ordinary projections retain their existing sink.
pub(crate) struct Output<'a, D: Scalar, Chips: M, R: AxisName, O: AxisName>(
    pub(crate) &'a mut HbmTensor<D, Chips, m![R, O]>,
);

impl<D: MaterializableScalar, I: AxisName, O: AxisName, R: AxisName, Chips: M>
    Sink<D, I, O, R, Chips> for Output<'_, D, Chips, R, O>
{
    type Order = Rows<I, O, R>;

    fn store(
        &mut self,
        device: &mut Device,
        kept: &Result<D, I, O, R, Chips, Self::Order>,
        (block, pass): (usize, usize),
    ) {
        const {
            assert!(
                (<PassOutputs<I, O, R> as AxisName>::SIZE * D::BITS / 8).is_multiple_of(256),
                "a pass row ends inside an HBM write unit"
            )
        };
        if const { (<KeptOutputs<I, O, R> as AxisName>::SIZE * D::BITS / 8).is_multiple_of(256) } {
            self.0.store(device, kept, (block, pass));
        } else {
            // SAFETY: the blocks partition the rows in order; the pass selects its output columns.
            let blocked = unsafe { self.0.view_mut().reshape::<Chips, Blocked<R, O>>() };
            let tile = blocked
                .tile::<m![{ Bs<R> }], 1, m![{ Bs<R> } = 1 #{!} ({ Bs<R> }), { B<R> }, O]>(block)
                .tile::<m![O / ({ Np<I, O, R> })], 1, m![
                    { Bs<R> } = 1 #{!} ({ Bs<R> }),
                    { B<R> },
                    O / ({ Np<I, O, R> }) = 1 #{!} (O / ({ Np<I, O, R> })),
                    O % ({ Np<I, O, R> })
                ]>(pass);
            let mut rows: DmTensor<
                D,
                Chips,
                Cluster<R>,
                m![{ B<R> } % 4, { B<R> } % ({ Hr<R> }) / 4 # 64],
                m![O % ({ Np<I, O, R> })],
            > = DmTensor::new();
            kept.view().to_dm_view(&mut device.tdma, rows.view_mut());
            rows.view().to_hbm_view(&mut device.tdma, tile);
        }
    }
}
