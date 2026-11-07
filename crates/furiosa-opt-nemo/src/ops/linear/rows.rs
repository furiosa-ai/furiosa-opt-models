//! Rows in HBM projected by a register-resident row tile.
//!
//! - Each cluster holds its half of a block of rows in the register file, one K group a slice,
//!   while the pass's weight, resident in data memory, streams past it in 64-byte packets of
//!   eight outputs by four features, each packet over every row group of the tile.
//! - Slices are output groups by K groups; the K groups' FP32 partials sum in the inter-slice
//!   reduction of the same command, and each output rounds to BF16 once.
//! - A block's rows land in every output group's slices; the next block lands, and the register
//!   of the next fills, while the current block contracts.
//! - A bias adds `bias / parts` to every K group's partial sums before the reduction, so each
//!   output still rounds once; FP32 outputs land the reduced sums unrounded.

use core::marker::PhantomData;

use crate::Module;
use furiosa_opt_std::prelude::*;

/// Rows projected by `weight` plus `bias`, block by block, into `Out` rows.
pub(crate) struct Projection<'a, I: AxisName, O: AxisName, R: AxisName, Chips: M, Bias, Out = bf16>
{
    weight: &'a HbmTensor<bf16, Chips, m![O, I]>,
    bias: &'a Bias,
    rows: PhantomData<(R, Out)>,
}

impl<'a, I: AxisName, O: AxisName, R: AxisName, Chips: M, Bias, Out>
    Projection<'a, I, O, R, Chips, Bias, Out>
{
    pub(crate) const fn new(weight: &'a HbmTensor<bf16, Chips, m![O, I]>, bias: &'a Bias) -> Self {
        Self {
            weight,
            bias,
            rows: PhantomData,
        }
    }
}

/// BF16 elements one register fill holds a slice: one bank, so the next fill takes the other.
const REGISTER: usize = 16_384;
/// Data memory a slice gives the weight slab, in bytes.
const SLAB_BYTES: usize = 301_056;

/// The rows a cluster holds of a block: 128 when the rows come in blocks of 256, else 64.
impl Tiling {
    const fn half(rows: usize) -> usize {
        if rows.is_multiple_of(256) {
            128
        } else {
            assert!(rows.is_multiple_of(128), "rows come in blocks of 128");
            64
        }
    }
}

#[derive(Clone, Copy)]
struct Tiling {
    /// K groups: the slices whose partials one inter-slice reduction sums.
    parts: usize,
    /// Outputs one slice holds after the reduction, for each of its rows.
    kept: usize,
    passes: usize,
}

/// The row tile of each served `(input, output)` at 128 or 64 rows a cluster: K groups, outputs
/// kept a slice, weight passes. `new` checks each against the register, slab and slice geometry,
/// so a value that cannot hold fails to compile rather than lands wrong.
impl Tiling {
    // TODO: no rule selects these values and no measurement records them, as for the contraction
    // tiling of `kernel.rs`; choose among the legal tiles by a recorded measurement per shape.
    const fn new(input: usize, output: usize, rows: usize) -> Self {
        let (parts, kept, passes) = match (input, output, Self::half(rows)) {
            (128, 2048, 128) => (16, 128, 1),
            (1792, 5376, 128) => (64, 336, 1),
            (2048, 2048, 128) => (16, 128, 1),
            (2048, 3072, 128) => (16, 192, 1),
            (2048, 6144, 128) => (16, 384, 1),
            (2048, 16384, 128) => (16, 1024, 1),
            (3584, 5376, 128) => (64, 336, 1),
            (5120, 5376, 128) => (64, 336, 1),
            (5376, 1792, 128) => (64, 112, 1),
            (5376, 5376, 128) => (64, 336, 1),
            (5376, 7168, 128) => (64, 448, 1),
            (5376, 21504, 128) => (64, 448, 3),
            (5376, 28672, 128) => (64, 448, 4),
            (7168, 5376, 128) => (128, 336, 1),
            (8192, 2048, 128) => (64, 128, 1),
            (14336, 5376, 128) => (128, 168, 2),
            (5120, 5376, 64) => (64, 168, 1),
            (5376, 7168, 64) => (64, 224, 1),
            (5376, 21504, 64) => (64, 224, 3),
            (5376, 28672, 64) => (64, 224, 4),
            (7168, 5376, 64) => (128, 168, 1),
            (14336, 5376, 64) => (128, 56, 3),
            _ => panic!("no row tile for this projection"),
        };
        let tiling = Tiling {
            parts,
            kept,
            passes,
        };
        let chunk = input / parts;
        let outputs = tiling.outputs(rows);
        assert!(input.is_multiple_of(parts) && chunk.is_multiple_of(4));
        assert!(Self::half(rows) * chunk <= REGISTER);
        assert!(kept.is_multiple_of(8) && outputs * chunk * 2 <= SLAB_BYTES);
        assert!(256 / parts * outputs * passes == output);
        tiling
    }

    /// Outputs a slice computes in a pass: the reduction spreads them over its K groups, kept a slice.
    const fn outputs(self, rows: usize) -> usize {
        self.parts / (Self::half(rows) / 8) * self.kept
    }
}

/// Features of one K group.
#[derive(Clone, Debug)]
pub struct Chunked<I, O, R>(PhantomData<(I, O, R)>);

impl<I: AxisName, O: AxisName, R: AxisName> AxisName for Chunked<I, O, R> {
    const NAME: Ident = Ident::new("RowChunk");
    const SIZE: usize = I::SIZE / Tiling::new(I::SIZE, O::SIZE, R::SIZE).parts;
}

/// Outputs a slice computes in a pass.
#[derive(Clone, Debug)]
pub struct SliceOutputs<I, O, R>(PhantomData<(I, O, R)>);

impl<I: AxisName, O: AxisName, R: AxisName> AxisName for SliceOutputs<I, O, R> {
    const NAME: Ident = Ident::new("RowSliceOutputs");
    const SIZE: usize = Tiling::new(I::SIZE, O::SIZE, R::SIZE).outputs(R::SIZE);
}

/// Outputs a pass computes.
#[derive(Clone, Debug)]
pub struct PassOutputs<I, O, R>(PhantomData<(I, O, R)>);

impl<I: AxisName, O: AxisName, R: AxisName> AxisName for PassOutputs<I, O, R> {
    const NAME: Ident = Ident::new("RowPassOutputs");
    const SIZE: usize = {
        let tiling = Tiling::new(I::SIZE, O::SIZE, R::SIZE);
        256 / tiling.parts * tiling.outputs(R::SIZE)
    };
}

/// Outputs a slice keeps of each row after the reduction.
#[derive(Clone, Debug)]
pub struct KeptOutputs<I, O, R>(PhantomData<(I, O, R)>);

impl<I: AxisName, O: AxisName, R: AxisName> AxisName for KeptOutputs<I, O, R> {
    const NAME: Ident = Ident::new("RowKeptOutputs");
    const SIZE: usize = Tiling::new(I::SIZE, O::SIZE, R::SIZE).kept;
}

/// Weight passes.
#[derive(Clone, Debug)]
pub struct Passes<I, O, R>(PhantomData<(I, O, R)>);

impl<I: AxisName, O: AxisName, R: AxisName> AxisName for Passes<I, O, R> {
    const NAME: Ident = Ident::new("RowPasses");
    const SIZE: usize = Tiling::new(I::SIZE, O::SIZE, R::SIZE).passes;
}

/// Rows a cluster holds of a block.
#[derive(Clone, Debug)]
pub struct Half<R>(PhantomData<(R,)>);

impl<R: AxisName> AxisName for Half<R> {
    const NAME: Ident = Ident::new("RowHalf");
    const SIZE: usize = Tiling::half(R::SIZE);
}

/// The rows of one block, both clusters'.
#[derive(Clone, Debug)]
pub struct RowBlock<R>(PhantomData<(R,)>);

impl<R: AxisName> AxisName for RowBlock<R> {
    const NAME: Ident = Ident::new("RowBlock");
    const SIZE: usize = 2 * Half::<R>::SIZE;
}

/// The blocks of the rows, in row order.
#[derive(Clone, Debug)]
pub struct RowBlocks<R>(PhantomData<(R,)>);

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
type Landed<I, O, R, Chips, E> = DmTensor<bf16, Chips, Cluster<R>, Slices<I, O, R>, E>;
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

impl<'s, 't, I, O, R, Chips, B, D, LE, RE, S, T> Module<(&mut Device, (&'s S, &'t mut T))>
    for Projection<'_, I, O, R, Chips, B, D>
where
    I: AxisName,
    O: AxisName,
    R: AxisName,
    Chips: M,
    B: Offset<I, O, R, Chips>,
    D: Rounding,
    LE: M,
    RE: M,
    S: Source<I, O, R, Chips, Order = LE>,
    T: Sink<D, I, O, R, Chips, Order = RE>,
{
    type Output = ();

    fn forward(&self, (device, (rows, output)): (&mut Device, (&'s S, &'t mut T))) {
        for pass in 0..<Passes<I, O, R> as AxisName>::SIZE {
            Pass::<_, D, RE>(self, PhantomData).forward((
                device,
                (
                    pass,
                    rows,
                    |device: &mut Device, kept: &Result<D, I, O, R, Chips, RE>, block| {
                        output.store(device, kept, (block, pass))
                    },
                ),
            ));
        }
    }
}

axes![Parts = 3];

impl<'i, 'o, I: AxisName, O: AxisName, R: AxisName, Chips: M, W: AxisName>
    Module<(
        &mut Device,
        (
            &'i HbmTensor<bf16, Chips, m![R, I]>,
            (
                &'o mut HbmTensor<bf16, Chips, m![R, W]>,
                &'o mut HbmTensor<bf16, Chips, m![R, W]>,
                &'o mut HbmTensor<bf16, Chips, m![R, W]>,
            ),
        ),
    )> for Projection<'_, I, O, R, Chips, ()>
{
    type Output = ();

    fn forward(
        &self,
        (device, (input, (first, second, third))): (
            &mut Device,
            (
                &'i HbmTensor<bf16, Chips, m![R, I]>,
                (
                    &'o mut HbmTensor<bf16, Chips, m![R, W]>,
                    &'o mut HbmTensor<bf16, Chips, m![R, W]>,
                    &'o mut HbmTensor<bf16, Chips, m![R, W]>,
                ),
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
                // SAFETY: the blocks split the rows in order, and a part's columns are its pass's outputs.
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
            Pass::<_, bf16, Rows<I, O, R>>(self, PhantomData).forward((
                device,
                (
                    0,
                    input,
                    |device: &mut Device,
                     kept: &Result<bf16, I, O, R, Chips, Rows<I, O, R>>,
                     block| store_part(device, kept, first, block),
                ),
            ));
            Pass::<_, bf16, Rows<I, O, R>>(self, PhantomData).forward((
                device,
                (
                    1,
                    input,
                    |device: &mut Device,
                     kept: &Result<bf16, I, O, R, Chips, Rows<I, O, R>>,
                     block| store_part(device, kept, second, block),
                ),
            ));
            Pass::<_, bf16, Rows<I, O, R>>(self, PhantomData).forward((
                device,
                (
                    2,
                    input,
                    |device: &mut Device,
                     kept: &Result<bf16, I, O, R, Chips, Rows<I, O, R>>,
                     block| store_part(device, kept, third, block),
                ),
            ));
        } else {
            let mut joint: HbmTensor<bf16, Chips, m![R, O]> = HbmTensor::new();
            self.forward((&mut *device, (input, &mut joint)));
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
    'd,
    's,
    I: AxisName,
    O: AxisName,
    R: AxisName,
    Chips: M,
    Bias: Offset<I, O, R, Chips>,
    Scalar,
    D,
    LE,
    RE,
    S,
    Store,
> Module<(&'d mut Device, (usize, &'s S, Store))>
    for Pass<'_, Projection<'_, I, O, R, Chips, Bias, Scalar>, D, RE>
where
    D: Rounding,
    LE: M,
    RE: M,
    S: Source<I, O, R, Chips, Order = LE>,
    Store: FnMut(&mut Device, &Result<D, I, O, R, Chips, RE>, usize),
{
    type Output = ();

    fn forward(&self, (device, (pass, rows, mut store)): (&'d mut Device, (usize, &'s S, Store))) {
        let fill = |device: &mut Device, landed: &Landed<I, O, R, Chips, LE>| {
            device.sub.begin(landed.view())
                .fetch::<m![{ B<R> } % 8, I % ({ Ks<I, O, R> }) / 4, { B<R> } % ({ Hr<R> }) / 32], m![{ B<R> } % 32 / 8, I % 4]>()
                .collect::<m![{ B<R> } % 8, I % ({ Ks<I, O, R> }) / 4, { B<R> } % ({ Hr<R> }) / 32], m![{ B<R> } % 32 / 8, I % 4]>()
                .to_trf()
        };
        let blocks = <Bs<R> as M>::SIZE;
        let mut first: Landed<I, O, R, Chips, LE> = DmTensor::new();
        let mut second: Landed<I, O, R, Chips, LE> = DmTensor::new();
        // Two results, so that a block contracts while the block before it is stored.
        let mut kept: Result<D, I, O, R, Chips, RE> = DmTensor::new();
        let mut held: Result<D, I, O, R, Chips, RE> = DmTensor::new();
        // The first rows land ahead of the weight, which the first contraction waits for anyway.
        rows.land(device, 0, &mut first);
        if const { <Bs<R> as M>::SIZE > 1 } {
            rows.land(device, 1, &mut second);
        }
        let slab: Weight<I, O, R, Chips> = self
            .0
            .weight
            .view()
            .tile::<m![O / ({ Np<I, O, R> })], 1, m![O % ({ Np<I, O, R> }) # O, I]>(pass)
            .to_dm(&mut device.tdma);
        let offsets = self.0.bias.pass(device, pass);
        // Block b contracts while block b + 1 fills and block b + 2 lands.
        for pair in 0..blocks / 2 {
            let block = 2 * pair;
            let register = fill(device, &first);
            Bias::contract::<D, RE>(device, (&slab, &offsets), &register, &mut kept);
            let next = fill(device, &second);
            rows.land(device, (block + 2).min(blocks - 1), &mut first);
            store(device, &kept, block);
            Bias::contract::<D, RE>(device, (&slab, &offsets), &next, &mut held);
            rows.land(device, (block + 3).min(blocks - 1), &mut second);
            store(device, &held, block + 1);
        }
        if const { <Bs<R> as M>::SIZE % 2 == 1 } {
            let register = fill(device, &first);
            Bias::contract::<D, RE>(device, (&slab, &offsets), &register, &mut kept);
            store(device, &kept, blocks - 1);
        }
    }
}

/// Rows a projection reads: row-major, or head-major as heads of `H` features.
pub(crate) trait Source<I: AxisName, O: AxisName, R: AxisName, Chips: M> {
    /// The order a block's rows land in: the order they arrive in.
    type Order: M;

    /// Block `block`'s rows into `landed`.
    fn land(
        &self,
        device: &mut Device,
        block: usize,
        landed: &mut Landed<I, O, R, Chips, Self::Order>,
    );
}

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M> Source<I, O, R, Chips>
    for HbmTensor<bf16, Chips, m![R, I]>
{
    type Order = Lines<I, O, R>;

    fn land(
        &self,
        device: &mut Device,
        block: usize,
        landed: &mut Landed<I, O, R, Chips, Self::Order>,
    ) {
        // SAFETY: the blocks split the rows in order; every cell keeps its address.
        unsafe { self.view().reshape::<Chips, Blocked<R, I>>() }
            .tile::<m![{ Bs<R> }], 1, m![{ Bs<R> } = 1 # ({ Bs<R> }), { B<R> }, I]>(block)
            .to_dm_view(&mut device.tdma, landed.view_mut());
    }
}

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M, H: AxisName> Source<I, O, R, Chips>
    for HbmTensor<bf16, Chips, m![I / ({ Symbol<H> }), R, I % ({ Symbol<H> })]>
{
    type Order = Columns<I, O, R, H>;

    fn land(
        &self,
        device: &mut Device,
        block: usize,
        landed: &mut Landed<I, O, R, Chips, Self::Order>,
    ) {
        // SAFETY: as for row-major rows, within each head.
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
        .to_dm_view(&mut device.tdma, landed.view_mut());
    }
}

/// What a pass adds to its FP32 sums before they round: nothing, or every output's bias.
pub(crate) trait Offset<I: AxisName, O: AxisName, R: AxisName, Chips: M> {
    /// The operand a pass reads.
    type Pass;

    /// The operand of pass `pass`.
    fn pass(&self, device: &mut Device, pass: usize) -> Self::Pass;

    /// One block's rows by the pass's weight: every output of the pass for the cluster's rows.
    fn contract<D: Rounding, E: M>(
        device: &mut Device,
        pass: (&Weight<I, O, R, Chips>, &Self::Pass),
        register: &Register<I, O, R, Chips>,
        kept: &mut Result<D, I, O, R, Chips, E>,
    );
}

/// The reduced FP32 sums of a block, one stream a slice of eight rows by its kept outputs.
type Reduced<'l, I, O, R, C> = VectorFinalTensor<
    'l,
    { Tu::Main },
    f32,
    C,
    Cluster<R>,
    Kept<I, O, R>,
    m![O % ({ Nr<I, O, R> }) / 8, O % 8],
    m![{ B<R> } % 8],
>;

/// How a block's FP32 sums land: rounded once to BF16, or as they are.
pub(crate) trait Rounding: MaterializableScalar {
    fn land<I: AxisName, O: AxisName, R: AxisName, Chips: M, E: M>(
        sums: Reduced<'_, I, O, R, Chips>,
        kept: &mut Result<Self, I, O, R, Chips, E>,
    );
}

impl Rounding for bf16 {
    fn land<I: AxisName, O: AxisName, R: AxisName, Chips: M, E: M>(
        sums: Reduced<'_, I, O, R, Chips>,
        kept: &mut Result<bf16, I, O, R, Chips, E>,
    ) {
        sums.cast::<bf16, m![{ B<R> } % 8 # 16]>()
            .transpose::<m![O % ({ Nr<I, O, R> }) / 4, { B<R> } % 8], m![O % 4 # 16]>()
            .commit_trim::<m![O % 4]>()
            .commit_view(kept.view_mut());
    }
}

impl Rounding for f32 {
    fn land<I: AxisName, O: AxisName, R: AxisName, Chips: M, E: M>(
        sums: Reduced<'_, I, O, R, Chips>,
        kept: &mut Result<f32, I, O, R, Chips, E>,
    ) {
        sums.transpose::<m![O % ({ Nr<I, O, R> }) / 2, { B<R> } % 8], m![O % 2 # 8]>()
            .commit_trim::<m![O % 2]>()
            .commit_view(kept.view_mut());
    }
}

// The outputs stay one stream digit until the lane fold: the accumulator check reads a split
// digit outside the reduced one as reduced, and counts every output as a slot.
type Feed<I, O, R> = m![O % ({ Ns<I, O, R> }) / 8, I % ({ Ks<I, O, R> }) / 4];
type Steps<I, O, R> = m![
    O % ({ Ns<I, O, R> }) / 8,
    I % ({ Ks<I, O, R> }) / 4,
    { B<R> } % ({ Hr<R> }) / 8
];
type Sums<I, O, R> = m![O % ({ Ns<I, O, R> }) / 8, { B<R> } % ({ Hr<R> }) / 8];
type Folded<I, O, R> = m![
    O % ({ Ns<I, O, R> }) / ({ Nr<I, O, R> }),
    O % ({ Nr<I, O, R> }) / 8,
    { B<R> } % ({ Hr<R> }) / 8,
    O % 8
];

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M> Offset<I, O, R, Chips> for () {
    type Pass = ();

    fn pass(&self, _: &mut Device, _: usize) {}

    fn contract<D: Rounding, E: M>(
        device: &mut Device,
        (weight, ()): (&Weight<I, O, R, Chips>, &()),
        register: &Register<I, O, R, Chips>,
        kept: &mut Result<D, I, O, R, Chips, E>,
    ) {
        let sums = device
            .main
            .begin(weight.view())
            .fetch::<Feed<I, O, R>, m![O % 8, I % 4]>()
            .collect::<m![{ Feed<I, O, R> }, O % 8 / 4], m![O % 4, I % 4]>()
            .contract_outer::<Steps<I, O, R>, m![O % 8, I % 4], _, _, _>(register)
            .contract_packet::<m![O % 8]>()
            .contract_time::<Sums<I, O, R>>()
            .contract_lane::<Folded<I, O, R>, m![{ B<R> } % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Kept<I, O, R>, m![O % ({ Nr<I, O, R> }) / 8, O % 8]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final();
        D::land(sums, kept);
    }
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

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M> Offset<I, O, R, Chips>
    for HbmTensor<f32, Chips, m![O]>
{
    type Pass = Offsets<I, O, R, Chips>;

    fn pass(&self, device: &mut Device, pass: usize) -> Offsets<I, O, R, Chips> {
        type Group<I, O, R> =
            m![O % ({ Ns<I, O, R> }) / ({ Nr<I, O, R> }), O % ({ Nr<I, O, R> }) / 8];
        let landed: DmTensor<f32, Chips, Cluster<R>, Slices<I, O, R>, m![O % ({ Ns<I, O, R> })]> =
            self.view()
                .tile::<m![O / ({ Np<I, O, R> })], 1, m![O % ({ Np<I, O, R> }) # O]>(pass)
                .to_dm(&mut device.tdma);
        device
            .main
            .begin(landed.view())
            .fetch::<Group<I, O, R>, m![O % 8]>()
            .collect::<Group<I, O, R>, m![O % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Group<I, O, R> }, O % 8 / 4 % 2], m![O % 4]>()
            .vector_fp_binary(
                FpBinaryOp::MulF(FpMulAlu::Mul0),
                const { 1.0 / Tiling::new(I::SIZE, O::SIZE, R::SIZE).parts as f32 },
            )
            .vector_widen_concat::<Group<I, O, R>, m![O % 8]>()
            .vector_final()
            .to_vrf(&mut device.sub)
    }

    fn contract<D: Rounding, E: M>(
        device: &mut Device,
        (weight, offsets): (&Weight<I, O, R, Chips>, &Offsets<I, O, R, Chips>),
        register: &Register<I, O, R, Chips>,
        kept: &mut Result<D, I, O, R, Chips, E>,
    ) {
        let sums = device
            .main
            .begin(weight.view())
            .fetch::<Feed<I, O, R>, m![O % 8, I % 4]>()
            .collect::<m![{ Feed<I, O, R> }, O % 8 / 4], m![O % 4, I % 4]>()
            .contract_outer::<Steps<I, O, R>, m![O % 8, I % 4], _, _, _>(register)
            .contract_packet::<m![O % 8]>()
            .contract_time::<Sums<I, O, R>>()
            .contract_lane::<Folded<I, O, R>, m![{ B<R> } % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, offsets)
            .vector_inter_slice_reduce::<Kept<I, O, R>, m![O % ({ Nr<I, O, R> }) / 8, O % 8]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final();
        D::land(sums, kept);
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
        // SAFETY: as for the input.
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
        // SAFETY: as for row-major rows, within each head; a pass computes whole heads.
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
