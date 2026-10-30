//! Rows in HBM projected by a register-resident row tile.
//!
//! - Each cluster holds its half of a block of rows in the register file, one K group a slice,
//!   while the pass's weight, resident in data memory, streams past it in 64-byte packets of
//!   eight outputs by four features, each packet over every row group of the tile.
//! - Slices are output groups by K groups; the K groups' FP32 partials sum in the inter-slice
//!   reduction of the same command, and each output rounds to BF16 once.
//! - A block's rows are written in every output group's slices; the next block arrives, and the
//!   register of the next fills, while the current block contracts.
//! - A bias adds `bias / parts` to every K group's partial sums before the reduction, so each
//!   output still rounds once; FP32 outputs hold the reduced sums unrounded.

use core::marker::PhantomData;

use super::Matrix;
use super::kernel::{Contraction, Grid, Offset, Rounding, Tiling};
use crate::Op;
use crate::specialize::contraction;
use furiosa_opt_std::prelude::*;

/// Rows projected by `weight` plus `bias`, block by block, into `D` rows.
pub(crate) struct Projection<'a, I: AxisName, O: AxisName, R: AxisName, Chips: M, F, D = bf16> {
    weight: &'a HbmTensor<bf16, Chips, m![O, I]>,
    bias: &'a F,
    rows: PhantomData<(R, D)>,
}

impl<'a, I: AxisName, O: AxisName, R: AxisName, Chips: M, F, D>
    Projection<'a, I, O, R, Chips, F, D>
{
    pub(crate) const fn new(weight: &'a HbmTensor<bf16, Chips, m![O, I]>, bias: &'a F) -> Self {
        Self {
            weight,
            bias,
            rows: PhantomData,
        }
    }
}

/// BF16 elements one register fill holds a slice: one bank, so the next fill takes the other.
const REGISTER: usize = 16_384;

#[derive(Clone, Copy)]
struct Config {
    /// K groups: the slices whose partials one inter-slice reduction sums.
    parts: usize,
    /// Outputs one slice holds after the reduction, for each of its rows.
    kept: usize,
    passes: usize,
}

/// The row tile of `(input, output)` at a cluster's rows of `rows`, checked against the register,
/// slab and slice geometry so a tile that cannot hold fails to compile.
impl Config {
    const fn new(input: usize, output: usize, rows: usize) -> Self {
        let (parts, kept, passes) = contraction::row_tile(input, output, contraction::half(rows));
        let tiling = Config {
            parts,
            kept,
            passes,
        };
        let chunk = input / parts;
        let outputs = tiling.outputs(rows);
        assert!(input.is_multiple_of(parts) && chunk.is_multiple_of(4));
        assert!(contraction::half(rows) * chunk <= REGISTER);
        assert!(kept.is_multiple_of(8) && outputs * chunk * 2 <= contraction::SLAB);
        assert!(256 / parts * outputs * passes == output);
        tiling
    }

    /// Outputs a slice computes in a pass: the reduction spreads them over its K groups, kept a
    /// slice.
    const fn outputs(self, rows: usize) -> usize {
        self.parts / (contraction::half(rows) / 8) * self.kept
    }
}

/// Features of one K group.
#[derive(Clone, Debug)]
pub(crate) struct Chunked<I, O, R>(PhantomData<(I, O, R)>);

impl<I: AxisName, O: AxisName, R: AxisName> AxisName for Chunked<I, O, R> {
    const NAME: Ident = Ident::new("RowChunk");
    const SIZE: usize = I::SIZE / Config::new(I::SIZE, O::SIZE, R::SIZE).parts;
}

/// Each slice keeps its output group's outputs over one K group; tile `at` is pass `at`.
impl<I: AxisName, O: AxisName, R: AxisName> Tiling<I, O> for Chunked<I, O, R> {
    type Cluster = Cluster<R>;
    type Slices = Slices<I, O, R>;
    type Rows = m![O % ({ Ns<I, O, R> })];
    type Inputs = m![I % ({ Ks<I, O, R> })];
    type Restorations = m![O % ({ Ns<I, O, R> }) / ({ Ns<I, O, R> })];
    type Restored = m![O % ({ Ns<I, O, R> })];

    fn to_dm<D: MaterializableScalar, X: M, K: M, E: M>(
        device: &mut Device,
        tensor: &HbmTensor<D, X, m![O, { K }]>,
        at: usize,
    ) -> DmTensor<D, X, Cluster<R>, Slices<I, O, R>, E> {
        tensor
            .view()
            .tile::<m![O / ({ Np<I, O, R> })], 1, m![O % ({ Np<I, O, R> }) # O, { K }]>(at)
            .to_dm(&mut device.tdma)
    }
}

/// Outputs a slice computes in a pass.
#[derive(Clone, Debug)]
pub(crate) struct SliceOutputs<I, O, R>(PhantomData<(I, O, R)>);

impl<I: AxisName, O: AxisName, R: AxisName> AxisName for SliceOutputs<I, O, R> {
    const NAME: Ident = Ident::new("RowSliceOutputs");
    const SIZE: usize = Config::new(I::SIZE, O::SIZE, R::SIZE).outputs(R::SIZE);
}

/// Outputs a pass computes.
#[derive(Clone, Debug)]
pub(crate) struct PassOutputs<I, O, R>(PhantomData<(I, O, R)>);

impl<I: AxisName, O: AxisName, R: AxisName> AxisName for PassOutputs<I, O, R> {
    const NAME: Ident = Ident::new("RowPassOutputs");
    const SIZE: usize = {
        let tiling = Config::new(I::SIZE, O::SIZE, R::SIZE);
        256 / tiling.parts * tiling.outputs(R::SIZE)
    };
}

/// Outputs a slice keeps of each row after the reduction.
#[derive(Clone, Debug)]
pub(crate) struct KeptOutputs<I, O, R>(PhantomData<(I, O, R)>);

impl<I: AxisName, O: AxisName, R: AxisName> AxisName for KeptOutputs<I, O, R> {
    const NAME: Ident = Ident::new("RowKeptOutputs");
    const SIZE: usize = Config::new(I::SIZE, O::SIZE, R::SIZE).kept;
}

/// Weight passes.
#[derive(Clone, Debug)]
pub(crate) struct Passes<I, O, R>(PhantomData<(I, O, R)>);

impl<I: AxisName, O: AxisName, R: AxisName> AxisName for Passes<I, O, R> {
    const NAME: Ident = Ident::new("RowPasses");
    const SIZE: usize = Config::new(I::SIZE, O::SIZE, R::SIZE).passes;
}

/// Rows a cluster holds of a block.
#[derive(Clone, Debug)]
pub(crate) struct Half<R>(PhantomData<(R,)>);

impl<R: AxisName> AxisName for Half<R> {
    const NAME: Ident = Ident::new("RowHalf");
    const SIZE: usize = contraction::half(R::SIZE);
}

/// The rows of one block, both clusters'.
#[derive(Clone, Debug)]
pub(crate) struct RowBlock<R>(PhantomData<(R,)>);

impl<R: AxisName> AxisName for RowBlock<R> {
    const NAME: Ident = Ident::new("RowBlock");
    const SIZE: usize = 2 * Half::<R>::SIZE;
}

/// The blocks of the rows, in row order.
#[derive(Clone, Debug)]
pub(crate) struct RowBlocks<R>(PhantomData<(R,)>);

impl<R: AxisName> AxisName for RowBlocks<R> {
    const NAME: Ident = Ident::new("RowBlocks");
    const SIZE: usize = R::SIZE / (2 * Half::<R>::SIZE);
}

type Ks<I, O, R> = Symbol<Chunked<I, O, R>>;
type Ns<I, O, R> = Symbol<SliceOutputs<I, O, R>>;
type Np<I, O, R> = Symbol<PassOutputs<I, O, R>>;
type Nr<I, O, R> = Symbol<KeptOutputs<I, O, R>>;
type Hr<R> = Symbol<Half<R>>;
type B<R> = Symbol<RowBlock<R>>;
type Bs<R> = Symbol<RowBlocks<R>>;

/// The rows as blocks.
type Blocked<R, W> = m![{ Bs<R> }, { B<R> }, W];

type Cluster<R> = m![{ B<R> } / ({ Hr<R> })];
/// Output groups by K groups.
type Slices<I, O, R> = m![O % ({ Np<I, O, R> }) / ({ Ns<I, O, R> }), I / ({ Ks<I, O, R> })];
/// After the reduction: output groups, the outputs the K groups spread over, row groups.
type Kept<I, O, R> = m![
    O % ({ Np<I, O, R> }) / ({ Ns<I, O, R> }),
    O % ({ Ns<I, O, R> }) / ({ Nr<I, O, R> }),
    { B<R> } % ({ Hr<R> }) / 8
];

type Weight<I, O, R, Chips> = DmTensor<
    bf16,
    Chips,
    Cluster<R>,
    Slices<I, O, R>,
    m![O % ({ Ns<I, O, R> }), I % ({ Ks<I, O, R> })],
>;
/// A block's rows, a copy in every output group, in the order `E` the rows arrive in.
type InDm<I, O, R, Chips, E> = DmTensor<bf16, Chips, Cluster<R>, Slices<I, O, R>, E>;
/// A cluster's rows of a block, row-major.
type Lines<I, O, R> = m![{ B<R> } % ({ Hr<R> }), I % ({ Ks<I, O, R> })];
/// A cluster's rows of a block, head-major as heads of `H` features.
type Columns<I, O, R, H> = m![
    I % ({ Ks<I, O, R> }) / ({ Symbol<H> }),
    { B<R> } % ({ Hr<R> }),
    I % ({ Symbol<H> })
];
/// A cluster's rows of a block, eight a lane: K quads outermost, then row groups.
type Register<I, O, R, Chips> = TrfTensor<
    bf16,
    Chips,
    Cluster<R>,
    Slices<I, O, R>,
    m![{ B<R> } % 8],
    m![I % ({ Ks<I, O, R> }) / 4, { B<R> } % ({ Hr<R> }) / 32, { B<R> } % 32 / 8, I % 4],
>;
/// Eight rows of a slice and their kept outputs, in the order `E` the rows leave in.
type Result<D, I, O, R, Chips, E> = DmTensor<D, Chips, Cluster<R>, Kept<I, O, R>, E>;
/// Eight rows of a slice, each with its kept outputs in order.
type Rows<I, O, R> = m![{ B<R> } % 8, O % ({ Nr<I, O, R> })];
/// The kept outputs of eight rows head by head, each head's rows in order.
type Heads<I, O, R, H> = m![
    O % ({ Nr<I, O, R> }) / ({ Symbol<H> }),
    { B<R> } % 8,
    O % ({ Symbol<H> })
];

impl<'s, 't, I, O, R, Chips, F, D, SourceOrder, SinkOrder, S, T> Op<(&'s S, &'t mut T)>
    for Projection<'_, I, O, R, Chips, F, D>
where
    I: AxisName,
    O: AxisName,
    R: AxisName,
    Chips: M,
    F: Bias<I, O, R, Chips>,
    D: Rounding,
    SourceOrder: M,
    SinkOrder: M,
    S: Source<I, O, R, Chips, Order = SourceOrder>,
    T: Sink<D, I, O, R, Chips, Order = SinkOrder>,
{
    type Output = ();

    fn forward(&self, device: &mut Device, (rows, output): (&'s S, &'t mut T)) {
        for pass in 0..<Passes<I, O, R> as AxisName>::SIZE {
            Pass::<_, D, SinkOrder>(self, PhantomData).forward(
                device,
                (
                    pass,
                    rows,
                    |device: &mut Device, kept: &Result<D, I, O, R, Chips, SinkOrder>, block| {
                        output.store(device, kept, (block, pass))
                    },
                ),
            );
        }
    }
}

axes![Parts = 3];

impl<'i, 'o, I: AxisName, O: AxisName, R: AxisName, Chips: M, W: AxisName>
    Op<(
        &'i HbmTensor<bf16, Chips, m![R, I]>,
        (
            &'o mut HbmTensor<bf16, Chips, m![R, W]>,
            &'o mut HbmTensor<bf16, Chips, m![R, W]>,
            &'o mut HbmTensor<bf16, Chips, m![R, W]>,
        ),
    )> for Projection<'_, I, O, R, Chips, ()>
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

/// Weight pass `pass` over every block of `rows`, each block's result through `store`.
struct Pass<'a, P, D, E>(&'a P, PhantomData<(D, E)>);

impl<
    's,
    I: AxisName,
    O: AxisName,
    R: AxisName,
    Chips: M,
    F: Bias<I, O, R, Chips>,
    Scalar,
    D,
    SourceOrder,
    SinkOrder,
    S,
    Store,
> Op<(usize, &'s S, Store)> for Pass<'_, Projection<'_, I, O, R, Chips, F, Scalar>, D, SinkOrder>
where
    D: Rounding,
    SourceOrder: M,
    SinkOrder: M,
    S: Source<I, O, R, Chips, Order = SourceOrder>,
    Store: FnMut(&mut Device, &Result<D, I, O, R, Chips, SinkOrder>, usize),
{
    type Output = ();

    fn forward(&self, device: &mut Device, (pass, rows, mut store): (usize, &'s S, Store)) {
        let fill = |device: &mut Device, in_dm: &InDm<I, O, R, Chips, SourceOrder>| {
            device.sub.begin(in_dm.view())
                .fetch::<
                    m![{ B<R> } % 8, I % ({ Ks<I, O, R> }) / 4, { B<R> } % ({ Hr<R> }) / 32],
                    m![{ B<R> } % 32 / 8, I % 4],
                >()
                .collect::<
                    m![{ B<R> } % 8, I % ({ Ks<I, O, R> }) / 4, { B<R> } % ({ Hr<R> }) / 32],
                    m![{ B<R> } % 32 / 8, I % 4],
                >()
                .to_trf()
        };
        let blocks = <Bs<R> as M>::SIZE;
        let mut first: InDm<I, O, R, Chips, SourceOrder> = DmTensor::new();
        let mut second: InDm<I, O, R, Chips, SourceOrder> = DmTensor::new();
        // Two results, so that a block contracts while the block before it is stored.
        let mut kept: Result<D, I, O, R, Chips, SinkOrder> = DmTensor::new();
        let mut held: Result<D, I, O, R, Chips, SinkOrder> = DmTensor::new();
        // The first rows are written ahead of the weight, which the first contraction waits for
        // anyway.
        rows.to_dm(device, 0, &mut first);
        if const { <Bs<R> as M>::SIZE > 1 } {
            rows.to_dm(device, 1, &mut second);
        }
        let slab: Weight<I, O, R, Chips> = self.0.weight.tile::<Chunked<I, O, R>>(device, pass);
        let offsets = self.0.bias.pass(device, pass);
        // One block's rows by the pass's weight: every output of the pass for the cluster's rows.
        let contract = |device: &mut Device,
                        register: &Register<I, O, R, Chips>,
                        kept: &mut Result<D, I, O, R, Chips, SinkOrder>| {
            D::finish::<
                Kept<I, O, R>,
                m![O % ({ Nr<I, O, R> }) / 8, O % 8],
                m![{ B<R> } % 8],
                m![{ D::Packets<m![O % ({ Nr<I, O, R> })]> }, { B<R> } % 8],
                D::Packet<O>,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            >(
                &offsets,
                Chunked::<I, O, R>::packets(device, &slab, register),
                kept.view_mut(),
            );
        };
        // Block b contracts while block b + 1 fills and block b + 2 arrives.
        for pair in 0..blocks / 2 {
            let block = 2 * pair;
            let register = fill(device, &first);
            contract(device, &register, &mut kept);
            let next = fill(device, &second);
            rows.to_dm(device, (block + 2).min(blocks - 1), &mut first);
            store(device, &kept, block);
            contract(device, &next, &mut held);
            rows.to_dm(device, (block + 3).min(blocks - 1), &mut second);
            store(device, &held, block + 1);
        }
        if const { <Bs<R> as M>::SIZE % 2 == 1 } {
            let register = fill(device, &first);
            contract(device, &register, &mut kept);
            store(device, &kept, blocks - 1);
        }
    }
}

/// Rows a projection reads: row-major, or head-major as heads of `H` features.
pub(crate) trait Source<I: AxisName, O: AxisName, R: AxisName, Chips: M> {
    /// The order a block's rows are written in: the order they arrive in.
    type Order: M;

    /// Block `block`'s rows into `in_dm`.
    fn to_dm(
        &self,
        device: &mut Device,
        block: usize,
        in_dm: &mut InDm<I, O, R, Chips, Self::Order>,
    );
}

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M> Source<I, O, R, Chips>
    for HbmTensor<bf16, Chips, m![R, I]>
{
    type Order = Lines<I, O, R>;

    fn to_dm(
        &self,
        device: &mut Device,
        block: usize,
        in_dm: &mut InDm<I, O, R, Chips, Self::Order>,
    ) {
        // SAFETY: the blocks split the rows in order; every cell keeps its address.
        unsafe { self.view().reshape::<Chips, Blocked<R, I>>() }
            .tile::<m![{ Bs<R> }], 1, m![{ Bs<R> } = 1 # ({ Bs<R> }), { B<R> }, I]>(block)
            .to_dm_view(&mut device.tdma, in_dm.view_mut());
    }
}

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M, H: AxisName> Source<I, O, R, Chips>
    for HbmTensor<bf16, Chips, m![I / ({ Symbol<H> }), R, I % ({ Symbol<H> })]>
{
    type Order = Columns<I, O, R, H>;

    fn to_dm(
        &self,
        device: &mut Device,
        block: usize,
        in_dm: &mut InDm<I, O, R, Chips, Self::Order>,
    ) {
        // SAFETY: within each head, the blocks split the rows in order; every cell keeps its
        // address.
        unsafe {
            self.view().reshape::<Chips, m![
                I / ({ Symbol<H> }),
                { Bs<R> },
                { B<R> },
                I % ({ Symbol<H> })
            ]>()
        }
        .tile::<m![{ Bs<R> }], 1, m![
            I / ({ Symbol<H> }),
            { Bs<R> } = 1 # ({ Bs<R> }),
            { B<R> },
            I % ({ Symbol<H> })
        ]>(block)
        .to_dm_view(&mut device.tdma, in_dm.view_mut());
    }
}

/// A projection's bias: none, or an FP32 value of each output, as each weight pass adds it.
pub(crate) trait Bias<I: AxisName, O: AxisName, R: AxisName, Chips: M> {
    /// What a pass adds to its K groups' FP32 sums before they meet.
    type Pass: Offset<Chips, Cluster<R>, Slices<I, O, R>>;

    /// The offset of pass `pass`.
    fn pass(&self, device: &mut Device, pass: usize) -> Self::Pass;
}

// The outputs stay one stream digit until the lane fold: the accumulator check reads a split
// digit outside the reduced one as reduced, and counts every output as a slot.
type Folded<I, O, R> = m![
    O % ({ Ns<I, O, R> }) / ({ Nr<I, O, R> }),
    O % ({ Nr<I, O, R> }) / 8,
    { B<R> } % ({ Hr<R> }) / 8,
    O % 8
];

impl<I: AxisName, O: AxisName, R: AxisName> Chunked<I, O, R> {
    /// A pass's weight against one block's register: every output of each slice for the
    /// cluster's rows, eight outputs by four features a packet, before the K groups meet.
    fn packets<'l, Chips: M>(
        device: &'l mut Device,
        weight: &'l Weight<I, O, R, Chips>,
        register: &'l Register<I, O, R, Chips>,
    ) -> ContractTensor<
        'l,
        { Tu::Main },
        f32,
        Chips,
        Cluster<R>,
        Slices<I, O, R>,
        Folded<I, O, R>,
        m![{ B<R> } % 8],
    > {
        Contraction::<I, O>::contract::<
            Grid<m![O % 8], Folded<I, O, R>, 4>,
            bf16,
            m![O % ({ Ns<I, O, R> }) / 8],
            m![I % ({ Ks<I, O, R> }) / 4],
            m![{ B<R> } % ({ Hr<R> }) / 8],
            m![{ B<R> } % 8],
            _,
            _,
            _,
            _,
            _,
            _,
        >(device, weight, register)
    }
}

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M> Bias<I, O, R, Chips> for () {
    type Pass = ();

    fn pass(&self, _: &mut Device, _: usize) {}
}

/// A pass's outputs' bias divided by the K groups, which is exact, in every K group's slices: the
/// groups' sums carry the whole bias once they are reduced.
type Offsets<I, O, R, Chips> = VrfTensor<
    f32,
    Chips,
    Cluster<R>,
    Slices<I, O, R>,
    m![O % ({ Ns<I, O, R> }) / ({ Nr<I, O, R> }), O % ({ Nr<I, O, R> }) / 8, O % 8],
>;

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M> Bias<I, O, R, Chips>
    for HbmTensor<f32, Chips, m![O]>
{
    type Pass = Offsets<I, O, R, Chips>;

    fn pass(&self, device: &mut Device, pass: usize) -> Offsets<I, O, R, Chips> {
        type Group<I, O, R> =
            m![O % ({ Ns<I, O, R> }) / ({ Nr<I, O, R> }), O % ({ Nr<I, O, R> }) / 8];
        let in_dm: DmTensor<f32, Chips, Cluster<R>, Slices<I, O, R>, m![O % ({ Ns<I, O, R> })]> =
            self.view()
                .tile::<m![O / ({ Np<I, O, R> })], 1, m![O % ({ Np<I, O, R> }) # O]>(pass)
                .to_dm(&mut device.tdma);
        device
            .main
            .begin(in_dm.view())
            .fetch::<Group<I, O, R>, m![O % 8]>()
            .collect::<Group<I, O, R>, m![O % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Group<I, O, R> }, O % 8 / 4 % 2], m![O % 4]>()
            .vector_fp_binary(
                FpBinaryOp::MulF(FpMulAlu::Mul0),
                const { 1.0 / Config::new(I::SIZE, O::SIZE, R::SIZE).parts as f32 },
            )
            .vector_widen_concat::<Group<I, O, R>, m![O % 8]>()
            .vector_final()
            .to_vrf(&mut device.sub)
    }
}

/// Rows a projection writes: row-major, or head-major as heads of `H` features.
pub(crate) trait Sink<D: MaterializableScalar, I: AxisName, O: AxisName, R: AxisName, Chips: M> {
    /// The order a block's projected rows wait in: the order they leave in.
    type Order: M;

    /// A block's projected rows into their pass's columns.
    fn store(
        &mut self,
        device: &mut Device,
        kept: &Result<D, I, O, R, Chips, Self::Order>,
        at: (usize, usize),
    );
}

impl<D: MaterializableScalar, I: AxisName, O: AxisName, R: AxisName, Chips: M>
    Sink<D, I, O, R, Chips> for HbmTensor<D, Chips, m![R, O]>
{
    type Order = Rows<I, O, R>;

    fn store(
        &mut self,
        device: &mut Device,
        kept: &Result<D, I, O, R, Chips, Self::Order>,
        (block, pass): (usize, usize),
    ) {
        // SAFETY: the blocks split the rows in order; every cell keeps its address.
        let blocked = unsafe { self.view_mut().reshape::<Chips, Blocked<R, O>>() };
        let tile = blocked
            .tile::<m![{ Bs<R> }], 1, m![{ Bs<R> } = 1 #{!} ({ Bs<R> }), { B<R> }, O]>(block)
            .tile::<m![O / ({ Np<I, O, R> })], 1, m![
                { Bs<R> } = 1 #{!} ({ Bs<R> }),
                { B<R> },
                O / ({ Np<I, O, R> }) = 1 #{!} (O / ({ Np<I, O, R> })),
                O % ({ Np<I, O, R> })
            ]>(pass);
        kept.view().to_hbm_view(&mut device.tdma, tile);
    }
}

impl<D, I, O, R, Chips, H> Sink<D, I, O, R, Chips>
    for HbmTensor<D, Chips, m![O / ({ Symbol<H> }), R, O % ({ Symbol<H> })]>
where
    D: MaterializableScalar,
    I: AxisName,
    O: AxisName,
    R: AxisName,
    Chips: M,
    H: AxisName,
{
    type Order = Heads<I, O, R, H>;

    fn store(
        &mut self,
        device: &mut Device,
        kept: &Result<D, I, O, R, Chips, Self::Order>,
        (block, pass): (usize, usize),
    ) {
        // SAFETY: within each head, the blocks split the rows in order, and a pass computes whole
        // heads.
        let blocked = unsafe {
            self.view_mut().reshape::<Chips, m![
                O / ({ Np<I, O, R> }),
                O % ({ Np<I, O, R> }) / ({ Symbol<H> }),
                { Bs<R> },
                { B<R> },
                O % ({ Symbol<H> })
            ]>()
        };
        let tile = blocked
            .tile::<m![O / ({ Np<I, O, R> })], 1, m![
                O / ({ Np<I, O, R> }) = 1 #{!} (O / ({ Np<I, O, R> })),
                O % ({ Np<I, O, R> }) / ({ Symbol<H> }),
                { Bs<R> },
                { B<R> },
                O % ({ Symbol<H> })
            ]>(pass)
            .tile::<m![{ Bs<R> }], 1, m![
                O / ({ Np<I, O, R> }) = 1 #{!} (O / ({ Np<I, O, R> })),
                O % ({ Np<I, O, R> }) / ({ Symbol<H> }),
                { Bs<R> } = 1 #{!} ({ Bs<R> }),
                { B<R> },
                O % ({ Symbol<H> })
            ]>(block);
        kept.view().to_hbm_view(&mut device.tdma, tile);
    }
}
