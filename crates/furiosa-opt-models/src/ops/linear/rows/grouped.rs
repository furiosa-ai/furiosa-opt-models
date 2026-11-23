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
use super::{InDm, Offset, Parts, Pass, Passes, Projection, Rounding, Source, tiled};
use crate::Op;
use crate::tensor::form::grouped::Layout;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

type Tile = crate::tensor::axis::Rows<256>;

/// A packed tensor borrowed for the duration of a projection.
pub(crate) struct Packed<'a, I: AxisName, O: AxisName, Chips: M>(
    pub(crate) &'a HbmTensor<bf16, Chips, Layout<I, O>>,
);

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M> Slab<I, O, R, Chips>
    for Packed<'_, I, O, Chips>
{
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

impl<'s, 't, 'w, I, O, R, Chips, Bias, Out, SourceOrder, SinkOrder, S, T> Op<(&'s S, &'t mut T)>
    for Projection<'_, I, O, R, Chips, Bias, Out, Packed<'w, I, O, Chips>>
where
    I: AxisName,
    O: AxisName,
    R: AxisName,
    Chips: M,
    Bias: Offset<I, O, R, Chips>,
    Out: Rounding,
    SourceOrder: M,
    SinkOrder: M,
    S: Source<I, O, R, Chips, Order = SourceOrder>,
    T: Sink<Out, I, O, R, Chips, Order = SinkOrder>,
{
    type Output = ();

    fn forward(&self, device: &mut Device, (rows, output): (&'s S, &'t mut T)) {
        for pass in 0..<Passes<I, O, R> as AxisName>::SIZE {
            Pass::<_, Out, SinkOrder>(self, PhantomData).forward(
                device,
                (
                    pass,
                    rows,
                    |device: &mut Device, kept: &Result<Out, I, O, R, Chips, SinkOrder>, block| {
                        output.store(device, kept, (block, pass))
                    },
                ),
            );
        }
    }
}

impl<'i, 'o, 'w, I: AxisName, O: AxisName, R: AxisName, Chips: M, W: AxisName>
    Op<(
        &'i HbmTensor<bf16, Chips, m![R, I]>,
        (
            &'o mut HbmTensor<bf16, Chips, m![R, W]>,
            &'o mut HbmTensor<bf16, Chips, m![R, W]>,
            &'o mut HbmTensor<bf16, Chips, m![R, W]>,
        ),
    )> for Projection<'_, I, O, R, Chips, (), bf16, Packed<'w, I, O, Chips>>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, (first, second, third)): (
            &'i HbmTensor<bf16, Chips, m![R, I]>,
            (
                &'o mut HbmTensor<bf16, Chips, m![R, W]>,
                &'o mut HbmTensor<bf16, Chips, m![R, W]>,
                &'o mut HbmTensor<bf16, Chips, m![R, W]>,
            ),
        ),
    ) {
        const { assert!(O::SIZE == 3 * W::SIZE) };
        if const {
            <Passes<I, O, R> as AxisName>::SIZE == 3
                && <PassOutputs<I, O, R> as AxisName>::SIZE == W::SIZE
        } {
            let store_part = |device: &mut Device,
                              kept: &Result<bf16, I, O, R, Chips, Rows<I, O, R>>,
                              output: &mut HbmTensor<bf16, Chips, m![R, W]>,
                              block: usize| {
                // SAFETY: the blocks split the rows in order, and a part's columns are its pass's
                // outputs.
                let blocked = unsafe {
                    output
                        .view_mut()
                        .reshape::<Chips, m![{ Bs<R> }, { B<R> }, O % ({ Np<I, O, R> })]>()
                };
                let tile = blocked.tile::<m![{ Bs<R> }], 1, m![
                    { Bs<R> } = 1 #{!} ({ Bs<R> }), { B<R> }, O % ({ Np<I, O, R> })
                ]>(block);
                kept.view().to_hbm_view(&mut device.tdma, tile);
            };
            Pass::<_, bf16, Rows<I, O, R>>(self, PhantomData).forward(
                device,
                (
                    0,
                    input,
                    |device: &mut Device,
                     kept: &Result<bf16, I, O, R, Chips, Rows<I, O, R>>,
                     block| store_part(device, kept, first, block),
                ),
            );
            Pass::<_, bf16, Rows<I, O, R>>(self, PhantomData).forward(
                device,
                (
                    1,
                    input,
                    |device: &mut Device,
                     kept: &Result<bf16, I, O, R, Chips, Rows<I, O, R>>,
                     block| store_part(device, kept, second, block),
                ),
            );
            Pass::<_, bf16, Rows<I, O, R>>(self, PhantomData).forward(
                device,
                (
                    2,
                    input,
                    |device: &mut Device,
                     kept: &Result<bf16, I, O, R, Chips, Rows<I, O, R>>,
                     block| store_part(device, kept, third, block),
                ),
            );
        } else {
            let mut joint: HbmTensor<bf16, Chips, m![R, O]> = HbmTensor::new();
            self.forward(&mut *device, (input, &mut joint));
            // SAFETY: a row's outputs are its three parts in order.
            let parts = unsafe { joint.view().reshape::<Chips, m![R, Parts, W]>() };
            parts
                .tile::<m![Parts], 1, m![R, Parts = 1 # 3, W]>(0)
                .to_hbm_view(&mut device.tdma, first.view_mut());
            parts
                .tile::<m![Parts], 1, m![R, Parts = 1 # 3, W]>(1)
                .to_hbm_view(&mut device.tdma, second.view_mut());
            parts
                .tile::<m![Parts], 1, m![R, Parts = 1 # 3, W]>(2)
                .to_hbm_view(&mut device.tdma, third.view_mut());
        }
    }
}

impl<
    's,
    'w,
    I: AxisName,
    O: AxisName,
    R: AxisName,
    Chips: M,
    Bias: Offset<I, O, R, Chips>,
    Scalar,
    D,
    SourceOrder,
    SinkOrder,
    S,
    Store,
> Op<(usize, &'s S, Store)>
    for Pass<
        '_,
        Projection<'_, I, O, R, Chips, Bias, Scalar, Packed<'w, I, O, Chips>>,
        D,
        SinkOrder,
    >
where
    D: Rounding,
    SourceOrder: M,
    SinkOrder: M,
    S: Source<I, O, R, Chips, Order = SourceOrder>,
    Store: FnMut(&mut Device, &Result<D, I, O, R, Chips, SinkOrder>, usize),
{
    type Output = ();

    fn forward(&self, device: &mut Device, (pass, rows, mut store): (usize, &'s S, Store)) {
        let mut first: InDm<I, O, R, Chips, SourceOrder> = DmTensor::new();
        let mut second: InDm<I, O, R, Chips, SourceOrder> = DmTensor::new();
        rows.to_dm(device, 0, &mut first);
        let slab = self.0.weight.slab(device, pass);
        if const { <Bs<R> as M>::SIZE > 1 } {
            rows.to_dm(device, 1, &mut second);
        }
        let offsets = self.0.bias.pass(device, pass);
        let (_, (last, block)) = tiled::trips(
            device,
            (rows, |block| block, PhantomData::<Bs<R>>),
            (first, second),
            (),
            |_: &mut Device, _, _: &mut InDm<I, O, R, Chips, SourceOrder>| {},
            |device, register, kept| Bias::contract(device, (&slab, &offsets), register, kept),
            &mut store,
        );
        store(device, &last, block);
    }
}
