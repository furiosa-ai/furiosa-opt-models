use crate::tensor::mapping;
use core::borrow::Borrow;
use core::marker::PhantomData;

use crate::Op;
use crate::specialize::contraction::{self, BAND, Routing};
use furiosa_opt_std::prelude;
use furiosa_opt_std::prelude::*;

use crate::ops::linear::Groups;
use crate::ops::linear::{Affine, Linear, Matrix};
use crate::tensor::axis;
use crate::tensor::live::Live;
use crate::tensor::mapping::Rows as Layout;

/// A projection's tiling: the measured choices and the geometry they fix.
struct Config {
    /// Input features one contraction part holds.
    chunk: usize,
    /// Output features one block holds.
    block: usize,
    /// The odd factor of the blocks per pass and cluster, carved as the most significant output
    /// digit.
    top: usize,
    /// Slices the top digit occupies, its next power of two.
    padded: usize,
    /// Weight slabs loaded one after another.
    passes: usize,
    routing: Routing,
}

impl Config {
    /// The tiling of `(chunk, block, passes, routing)` for `output` features: the remaining
    /// geometry follows from them, and a choice the slice layout cannot hold is refused.
    const fn new(
        output: usize,
        (chunk, block, passes, routing): (usize, usize, usize, Routing),
    ) -> Self {
        assert!(
            output.is_multiple_of(2 * passes * block),
            "the passes and blocks split the outputs over both clusters"
        );
        let top = contraction::odd(output / (2 * passes * block));
        assert!(
            top == 1 || passes == 1,
            "the top digit and the slab index are one digit"
        );
        Self {
            chunk,
            block,
            top,
            padded: top.next_power_of_two(),
            passes,
            routing,
        }
    }
}

impl<In: AxisName, Out: AxisName> Contraction<In, Out> {
    /// The tiling of `(In, Out)` and the geometry it fixes.
    const fn config() -> Config {
        Config::new(Out::SIZE, contraction::tiling(In::SIZE, Out::SIZE))
    }
}

#[derive(Clone, Debug)]
pub struct Contraction<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for Contraction<In, Out> {
    const NAME: Ident = Ident::new("Contraction");

    const SIZE: usize = {
        let c = Contraction::<In, Out>::config();
        c.chunk
    };
}

pub(crate) type Chunk<In, Out> = Symbol<Contraction<In, Out>>;

#[derive(Clone, Debug)]
pub struct OutputBlock<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for OutputBlock<In, Out> {
    const NAME: Ident = Ident::new("OutputBlock");

    const SIZE: usize = {
        let c = Contraction::<In, Out>::config();
        c.block
    };
}

pub(crate) type Block<In, Out> = Symbol<OutputBlock<In, Out>>;

#[derive(Clone, Debug)]
pub struct Carving<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for Carving<In, Out> {
    const NAME: Ident = Ident::new("Carving");

    const SIZE: usize = {
        let c = Contraction::<In, Out>::config();
        c.top * c.passes
    };
}

pub(crate) type Carve<In, Out> = Symbol<Carving<In, Out>>;

#[derive(Clone, Debug)]
pub struct TopDigit<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for TopDigit<In, Out> {
    const NAME: Ident = Ident::new("TopDigit");

    const SIZE: usize = {
        let c = Contraction::<In, Out>::config();
        c.top
    };
}

pub(crate) type Top<In, Out> = Symbol<TopDigit<In, Out>>;

#[derive(Clone, Debug)]
pub struct PaddingCount<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for PaddingCount<In, Out> {
    const NAME: Ident = Ident::new("PaddingCount");

    const SIZE: usize = {
        let c = Contraction::<In, Out>::config();
        c.padded
    };
}

pub(crate) type Padded<In, Out> = Symbol<PaddingCount<In, Out>>;

#[derive(Clone, Debug)]
pub struct PassCount<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for PassCount<In, Out> {
    const NAME: Ident = Ident::new("PassCount");

    const SIZE: usize = {
        let c = Contraction::<In, Out>::config();
        c.passes
    };
}

pub(crate) type Passes<In, Out> = Symbol<PassCount<In, Out>>;

pub(crate) type Input<In, Out, Rows> =
    m![In / ({ Chunk<In, Out> }), Rows, In % ({ Chunk<In, Out> })];
pub(crate) type Output<In, Out, Rows> =
    m![Out / ({ Block<In, Out> }), Rows, Out % ({ Block<In, Out> })];
type Parts<In, Out> = m![In / ({ Chunk<In, Out> })];
type Chunked<In, Out> = m![In % ({ Chunk<In, Out> })];

pub(crate) type Cluster<In, Out> =
    m![Out % (Out / ({ Carve<In, Out> })) / ((Out / ({ Carve<In, Out> })) / 2)];

type Blocks<In, Out> = m![Out % ((Out / ({ Carve<In, Out> })) / 2) / ({ Block<In, Out> })];
type Idle<In, Out> = m![256 / ({ Blocks<In, Out> }) / ({ Parts<In, Out> }) / ({ Padded<In, Out> })];
type ProjectionSlices<In, Out> = m![
    Out / (Out / ({ Top<In, Out> })) # ({ Padded<In, Out> }),
    { Blocks<In, Out> },
    1 # ({ Idle<In, Out> }),
    { Parts<In, Out> }
];
pub(crate) type Reduced<In, Out> = m![
    Out / (Out / ({ Top<In, Out> })) # ({ Padded<In, Out> }),
    { Blocks<In, Out> },
    1 # ({ Idle<In, Out> }),
    1 # ({ Parts<In, Out> })
];
type SlabIndex<In, Out> = m![Out / (Out / ({ Passes<In, Out> }))];
type SlabSelected<In, Out> = m![{ SlabIndex<In, Out> } = 1 #{!} ({ SlabIndex<In, Out> })];

pub(crate) type Element<In, Out, Rows> =
    m![{ SlabIndex<In, Out> }, Rows, Out % ({ Block<In, Out> }) / 4, Out % 4];

pub(crate) type Result<In, Out, Rows, X = m![1]> =
    DmTensor<bf16, X, Cluster<In, Out>, Reduced<In, Out>, Element<In, Out, Rows>>;

type Resident<In, Out, X = m![1]> = DmTensor<
    bf16,
    X,
    Cluster<In, Out>,
    ProjectionSlices<In, Out>,
    m![Out % ({ Block<In, Out> }), In % ({ Chunk<In, Out> })],
>;

/// Rows one contraction pass carries: every row while their lanes fit the TRF beside the
/// weight, otherwise one lane group.
#[derive(Clone, Debug)]
pub(crate) struct Batch<In, Out, Rows>(PhantomData<(In, Out, Rows)>);

impl<In: AxisName, Out: AxisName, Rows: AxisName> AxisName for Batch<In, Out, Rows> {
    const NAME: Ident = Ident::new("Batch");

    const SIZE: usize = contraction::batch(
        Rows::SIZE,
        <Chunked<In, Out> as M>::SIZE,
        axis::Lanes::<Rows>::SIZE,
    );
}

/// The rows of an HBM envelope one staging is written beside a resident weight slab.
#[derive(Clone, Debug)]
pub(crate) struct Band<Rows>(PhantomData<Rows>);

impl<Rows: AxisName> AxisName for Band<Rows> {
    const NAME: Ident = Ident::new("Band");

    const SIZE: usize = {
        assert!(Rows::SIZE.is_multiple_of(BAND));
        BAND
    };
}

/// The bands of an HBM envelope, in row order.
#[derive(Clone, Debug)]
pub(crate) struct Bands<Rows>(PhantomData<Rows>);

impl<Rows: AxisName> AxisName for Bands<Rows> {
    const NAME: Ident = Ident::new("Bands");
    const SIZE: usize = Rows::SIZE / Band::<Rows>::SIZE;
}

/// An HBM envelope's rows as its bands.
type Banded<Rows, Width> = m![{ Symbol<Bands<Rows>> }, { Symbol<Band<Rows>> }, Width];

/// Lane groups in one batch of `B` rows.
type Count<Rows, B> = m![Rows % B / ({ Symbol<axis::Lanes<Rows>> })];

type Lanes<In, Out, Rows, B, X = m![1]> = TrfTensor<
    bf16,
    X,
    Cluster<In, Out>,
    ProjectionSlices<In, Out>,
    m![{ Layout<Rows>::Lanes }],
    m![{ Count<Rows, B> }, { Chunked<In, Out> }],
>;

type Packet<Rows> = m![(16, { Layout<Rows>::Lanes }) / 1];
/// The elements of `D` in one DMA packet.
type Word<D> = Symbol<mapping::Packet<D>>;

/// The elements of `D` one contraction step takes: two DMA packets.
#[derive(Clone, Debug)]
struct Step<D>(PhantomData<D>);

impl<D: Scalar> AxisName for Step<D> {
    const NAME: Ident = Ident::new("Step");
    const SIZE: usize = 2 * <mapping::Packet<D> as AxisName>::SIZE;
}

type Replicated<In, Out, Rows, X = m![1]> =
    DmTensor<bf16, X, Cluster<In, Out>, ProjectionSlices<In, Out>, m![Rows, { Chunked<In, Out> }]>;

type Transposed<In, Out, Rows, X = m![1]> = DmTensor<
    bf16,
    X,
    Cluster<In, Out>,
    m![{ Parts<In, Out> }, { Layout<Rows>::Lanes }, { Chunked<In, Out> } / ({ Packet<Rows> })],
    m![1, { Groups<Rows> }, In % ({ Packet<Rows> })],
>;

type Ring<In, Out, Rows, X = m![1]> =
    DmTensor<bf16, X, Cluster<In, Out>, m![In / 16 # 256], m![1, Rows, In % 16]>;

type ChunkRows<In, Out, Rows, X = m![1]> = DmTensor<
    bf16,
    X,
    Cluster<In, Out>,
    m![{ Parts<In, Out> } # 256],
    m![1, Rows, { Chunked<In, Out> }],
>;

enum Staged<In: AxisName, Out: AxisName, Rows: AxisName, X: M = m![1]> {
    Replicated(Replicated<In, Out, Rows, X>),
    Transposed(Transposed<In, Out, Rows, X>),
    Ring(Ring<In, Out, Rows, X>),
    Chunk(ChunkRows<In, Out, Rows, X>),
}

/// The route a batch of `B` of the `Rows` takes. A transposed broadcast serves one lane group
/// only, and a chunk fetch grows with the lanes, so either takes the ring past its reach.
impl<In: AxisName, Out: AxisName, Rows: AxisName, X: M> Staged<In, Out, Rows, X> {
    const fn route<B: AxisName>() -> Routing {
        match Contraction::<In, Out>::config().routing {
            Routing::Transposed if B::SIZE > axis::Lanes::<Rows>::SIZE => Routing::Ring,
            Routing::Chunk if axis::Lanes::<Rows>::SIZE > 1 => Routing::Ring,
            routing => routing,
        }
    }
}

/// A view of a contraction's input rows, on chip or in HBM, that a stage copies from.
pub(crate) trait Source {
    type Chips: M;

    fn copy<'a, C: M, S: M, E: M>(
        self,
        device: &mut Device,
        into: DmTensorViewMut<'a, bf16, Self::Chips, C, S, E>,
    ) where
        Self: 'a;
}

impl<'v, X: M, ViewCluster: M, ViewSlices: M, ViewElements: M> Source
    for DmTensorView<'v, bf16, X, ViewCluster, ViewSlices, ViewElements>
{
    type Chips = X;

    fn copy<'a, C: M, S: M, E: M>(
        self,
        device: &mut Device,
        into: DmTensorViewMut<'a, bf16, X, C, S, E>,
    ) where
        Self: 'a,
    {
        self.to_dm_view(&mut device.tdma, into);
    }
}

impl<'v, X: M, ViewElements: M> Source for HbmTensorView<'v, bf16, X, ViewElements> {
    type Chips = X;

    fn copy<'a, C: M, S: M, E: M>(
        self,
        device: &mut Device,
        into: DmTensorViewMut<'a, bf16, X, C, S, E>,
    ) where
        Self: 'a,
    {
        self.to_dm_view(&mut device.tdma, into);
    }
}

struct Stage<In, Out, Rows, B>(PhantomData<(In, Out, Rows, B)>);

impl<In: AxisName, Out: AxisName, Rows: AxisName, B: AxisName, V: Source> crate::Op<V>
    for Stage<In, Out, Rows, B>
{
    type Output = Staged<In, Out, Rows, V::Chips>;

    fn forward(&self, device: &mut Device, input: V) -> Self::Output {
        match const { Staged::<In, Out, Rows, V::Chips>::route::<B>() } {
            Routing::Replicated => {
                let mut rows = DmTensor::new();
                input.copy(device, rows.view_mut());
                Staged::Replicated(rows)
            }

            Routing::Transposed => {
                let mut rows = DmTensor::new();
                input.copy(device, rows.view_mut());
                Staged::Transposed(rows)
            }

            Routing::Ring => {
                let mut rows = DmTensor::new();
                input.copy(device, rows.view_mut());
                Staged::Ring(rows)
            }

            Routing::Chunk => {
                let mut rows = DmTensor::new();
                input.copy(device, rows.view_mut());
                Staged::Chunk(rows)
            }
        }
    }
}

impl<In: AxisName, Out: AxisName, Rows: AxisName, X: M, B: AxisName> Op<(usize, PhantomData<B>)>
    for Staged<In, Out, Rows, X>
{
    type Output = Lanes<In, Out, Rows, B, X>;

    fn forward(&self, device: &mut Device, (at, _): (usize, PhantomData<B>)) -> Self::Output {
        match self {
            Self::Replicated(input) => device
                .sub
                .begin(input.view().tile::<m![Rows / B], 1, m![
                    Rows / B = 1 # (Rows / B), Rows % B, { Chunked<In, Out> }
                ]>(at))
                .fetch::<
                    m![{ Layout<Rows>::Lanes }, { Count<Rows, B> }, { Chunked<In, Out> } / 16],
                    m![In % 16],
                >()
                .collect::<
                    m![{ Layout<Rows>::Lanes }, { Count<Rows, B> }, { Chunked<In, Out> } / 16],
                    m![In % 16],
                >()
                .to_trf(),

            Self::Transposed(input) => device
                .sub
                .begin(input.view().tile::<
                    Layout<Rows>::Groups,
                    1,
                    m![1, { Layout<Rows>::Selected }, In % ({ Packet<Rows> })],
                >(at))
                .fetch::<m![1], m![In % ({ Packet<Rows> })]>()
                .switch::<ProjectionSlices<In, Out>, m![
                    { Layout<Rows>::Lanes },
                    { Count<Rows, B> },
                    { Chunked<In, Out> } / ({ Packet<Rows> })
                ]>(SwitchConfig::TransposedBroadcast1 {
                    slice1: <Parts<In, Out> as M>::SIZE,
                    slice0: <Chunk<In, Out> as M>::SIZE / 16,
                })
                .collect::<
                    m![{ Layout<Rows>::Lanes }, { Count<Rows, B> }, { Chunked<In, Out> } / 16],
                    m![In % 16],
                >()
                .to_trf(),

            Self::Ring(input) => device
                .sub
                .begin(input.view().tile::<m![Rows / B], 1, m![
                    1, Rows / B = 1 # (Rows / B), Rows % B, In % 16
                ]>(at))
                .fetch::<m![{ Layout<Rows>::Lanes }, { Count<Rows, B> }], m![In % 16]>()
                .switch::<ProjectionSlices<In, Out>, m![
                    { Layout<Rows>::Lanes },
                    { Count<Rows, B> },
                    { Chunked<In, Out> } / 16
                ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
                .collect::<
                    m![{ Layout<Rows>::Lanes }, { Count<Rows, B> }, { Chunked<In, Out> } / 16],
                    m![In % 16],
                >()
                .to_trf(),

            Self::Chunk(input) => device
                .sub
                .begin(input.view().tile::<m![Rows / B], 1, m![
                    1, Rows / B = 1 # (Rows / B), Rows % B, { Chunked<In, Out> }
                ]>(at))
                .fetch::<
                    m![{ Layout<Rows>::Lanes }, { Count<Rows, B> }, { Chunked<In, Out> } / 16],
                    m![In % 16],
                >()
                .switch::<ProjectionSlices<In, Out>, m![
                    { Layout<Rows>::Lanes },
                    { Count<Rows, B> },
                    { Chunked<In, Out> } / 16
                ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
                .collect::<
                    m![{ Layout<Rows>::Lanes }, { Count<Rows, B> }, { Chunked<In, Out> } / 16],
                    m![In % 16],
                >()
                .to_trf(),
        }
    }
}

/// How a contraction holds a weight tile: the clusters and slices it spreads over, the rows and
/// inputs each slice keeps, and the restorations that scale those rows.
pub(crate) trait Tiling<In: AxisName, Out: AxisName>: 'static {
    type Cluster: M;
    type Slices: M;
    /// Output rows of a tile one slice keeps.
    type Rows: M;
    /// Inputs of a row one slice keeps.
    type Inputs: M;
    /// The restorations of a slice's rows, each few enough rows that their FP32 scales fit the VRF.
    type Restorations: prelude::Count;
    /// Rows one restoration scales.
    type Restored: M;

    /// Tile `at` of an `[Out, K]` tensor, `K` a row's inputs or its scales, each slice keeping
    /// `E`: its rows' share of `K`.
    fn to_dm<D: MaterializableScalar, X: M, K: M, E: M>(
        device: &mut Device,
        tensor: &HbmTensor<D, X, m![Out, { K }]>,
        at: usize,
    ) -> DmTensor<D, X, Self::Cluster, Self::Slices, E>;
}

impl<In, Out, Rows, V, U, W, Group> Op<(Live<PhantomData<Rows>, Rows>, V, Group)> for Affine<W, ()>
where
    In: AxisName,
    Out: AxisName,
    Rows: AxisName,
    V: Source<Chips = m![1]>,
    U: Source<Chips = m![1]>,
    W: Matrix<In = In, Out = Out>,
    Group: Fn(usize) -> U,
{
    type Output = Result<In, Out, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        (live, input, group): (Live<PhantomData<Rows>, Rows>, V, Group),
    ) -> Self::Output {
        if live.count::<axis::Lanes<Rows>>() == <Groups<Rows> as M>::SIZE {
            return Contraction::<In, Out>::resident(device, &self.weight, input, |_, _| Rounded);
        }
        let rows = {
            match const { Staged::<In, Out, Rows>::route::<axis::Lanes<Rows>>() } {
                Routing::Replicated => {
                    let mut rows: Replicated<In, Out, Rows> = DmTensor::new();
                    for at in 0..live.count::<axis::Lanes<Rows>>() {
                        group(at).copy(
                            &mut *device,
                            rows.view_mut().tile::<
                                Layout<Rows>::Groups,
                                1,
                                Layout<Rows>::GroupMut<Chunked<In, Out>>,
                            >(at),
                        );
                    }
                    Staged::<In, Out, Rows>::Replicated(rows)
                }
                Routing::Transposed => {
                    let mut rows: Transposed<In, Out, Rows> = DmTensor::new();
                    for at in 0..live.count::<axis::Lanes<Rows>>() {
                        group(at).copy(
                            &mut *device,
                            rows.view_mut().tile::<
                                Layout<Rows>::Groups,
                                1,
                                m![1, { Layout<Rows>::SelectedMut }, In % ({ Packet<Rows> })],
                            >(at),
                        );
                    }
                    Staged::<In, Out, Rows>::Transposed(rows)
                }
                Routing::Ring => {
                    let mut rows: Ring<In, Out, Rows> = DmTensor::new();
                    for at in 0..live.count::<axis::Lanes<Rows>>() {
                        group(at).copy(
                            &mut *device,
                            rows.view_mut().tile::<
                                Layout<Rows>::Groups,
                                1,
                                m![1, { Layout<Rows>::GroupMut<m![In % 16]> }],
                            >(at),
                        );
                    }
                    Staged::<In, Out, Rows>::Ring(rows)
                }
                Routing::Chunk => {
                    let mut rows: ChunkRows<In, Out, Rows> = DmTensor::new();
                    for at in 0..live.count::<axis::Lanes<Rows>>() {
                        group(at).copy(
                            &mut *device,
                            rows.view_mut().tile::<
                                Layout<Rows>::Groups,
                                1,
                                m![1, { Layout<Rows>::GroupMut<Chunked<In, Out>> }],
                            >(at),
                        );
                    }
                    Staged::<In, Out, Rows>::Chunk(rows)
                }
            }
        };
        let mut result = DmTensor::new();
        #[unroll]
        for slab in 0..<Passes<In, Out> as M>::SIZE {
            let weight = self.weight.tile::<Contraction<In, Out>>(device, slab);
            for group in 0..live.count::<axis::Lanes<Rows>>() {
                let lanes = rows.forward(&mut *device, (group, PhantomData::<axis::Lanes<Rows>>));
                Rounded.store(
                    Contraction::<In, Out>::sums(device, &weight, &lanes),
                    &mut result,
                    slab,
                    group,
                );
            }
        }
        result
    }
}

/// Rows in HBM as a contraction reads and writes them a band of [`BAND`] rows at a time: a band
/// whole, which it stages at once when all its rows are live, or one lane group of it.
pub(crate) trait Stored {
    /// The envelope's rows.
    type Rows: AxisName;
    /// A band's cells.
    type Band: M;
    /// A lane group's cells.
    type Group: M;
    /// A band's cells, to write.
    type BandMut: M;

    /// Band `at` of the envelope.
    fn band(&self, at: usize) -> HbmTensorView<'_, bf16, m![1], Self::Band>;

    /// Lane group `at` of band `band`.
    fn group(&self, band: usize, at: usize) -> HbmTensorView<'_, bf16, m![1], Self::Group>;

    /// Band `at` of the envelope, to write.
    fn band_mut(&mut self, at: usize) -> HbmTensorViewMut<'_, bf16, m![1], Self::BandMut>;
}

/// The rows of a band, the same axis in every envelope.
#[derive(Clone, Debug)]
pub(crate) struct Line;

impl AxisName for Line {
    const NAME: Ident = Ident::new("Line");
    const SIZE: usize = BAND;
}

/// The bands of an envelope of `Rows`.
type Of<Rows> = Symbol<Bands<Rows>>;

/// Row-major rows: a lane group's rows, each with all its features.
impl<Rows: AxisName, In: AxisName> Stored for HbmTensor<bf16, m![1], m![Rows, In]> {
    type Rows = Rows;
    type Band = m![{ Of<Rows> } = 1 # ({ Of<Rows> }), Line, In];
    type BandMut = m![{ Of<Rows> } = 1 #{!} ({ Of<Rows> }), Line, In];
    type Group = m![
        { Of<Rows> } = 1 # ({ Of<Rows> }),
        { Layout<Line>::Selected },
        { Layout<Line>::Lanes },
        In
    ];

    fn band(&self, at: usize) -> HbmTensorView<'_, bf16, m![1], Self::Band> {
        // SAFETY: the bands split the row axis in row order; every cell keeps its address.
        unsafe { self.view().reshape::<m![1], m![{ Of<Rows> }, Line, In]>() }
            .tile::<m![{ Of<Rows> }], 1, Self::Band>(at)
    }

    fn group(&self, band: usize, at: usize) -> HbmTensorView<'_, bf16, m![1], Self::Group> {
        // SAFETY: as for a band, its lane groups in row order.
        unsafe {
            self.view().reshape::<m![1], m![
                { Of<Rows> },
                { Layout<Line>::Groups },
                { Layout<Line>::Lanes },
                In
            ]>()
        }
        .tile::<m![{ Of<Rows> }], 1, m![
            { Of<Rows> } = 1 # ({ Of<Rows> }),
            { Layout<Line>::Groups },
            { Layout<Line>::Lanes },
            In
        ]>(band)
        .tile::<Layout<Line>::Groups, 1, Self::Group>(at)
    }

    fn band_mut(&mut self, at: usize) -> HbmTensorViewMut<'_, bf16, m![1], Self::BandMut> {
        // SAFETY: as for reading a band.
        unsafe {
            self.view_mut()
                .reshape::<m![1], m![{ Of<Rows> }, Line, In]>()
        }
        .tile::<m![{ Of<Rows> }], 1, Self::BandMut>(at)
    }
}

/// Rows stored in groups of eight, sixteen features a block: a lane group's rows of one block
/// are contiguous.
impl<Rows: AxisName, In: AxisName> Stored
    for HbmTensor<bf16, m![1], m![Rows / 8, In / 16, Rows % 8, In % 16]>
{
    type Rows = Rows;
    type Band = m![
        { Of<Rows> } = 1 # ({ Of<Rows> }),
        { Layout<Line>::Groups },
        In / 16,
        { Layout<Line>::Lanes },
        In % 16
    ];
    type BandMut = m![
        { Of<Rows> } = 1 #{!} ({ Of<Rows> }),
        { Layout<Line>::Groups },
        In / 16,
        { Layout<Line>::Lanes },
        In % 16
    ];
    type Group = m![
        { Of<Rows> } = 1 # ({ Of<Rows> }),
        { Layout<Line>::Selected },
        In / 16,
        { Layout<Line>::Lanes },
        In % 16
    ];

    fn band(&self, at: usize) -> HbmTensorView<'_, bf16, m![1], Self::Band> {
        const { assert!(axis::Lanes::<Line>::SIZE == 8) };
        // SAFETY: a band's groups of eight rows are consecutive groups; every cell keeps its address.
        unsafe {
            self.view().reshape::<m![1], m![
                { Of<Rows> },
                { Layout<Line>::Groups },
                In / 16,
                { Layout<Line>::Lanes },
                In % 16
            ]>()
        }
        .tile::<m![{ Of<Rows> }], 1, Self::Band>(at)
    }

    fn group(&self, band: usize, at: usize) -> HbmTensorView<'_, bf16, m![1], Self::Group> {
        self.band(band)
            .tile::<Layout<Line>::Groups, 1, Self::Group>(at)
    }

    fn band_mut(&mut self, at: usize) -> HbmTensorViewMut<'_, bf16, m![1], Self::BandMut> {
        // SAFETY: as for reading a band.
        unsafe {
            self.view_mut().reshape::<m![1], m![
                { Of<Rows> },
                { Layout<Line>::Groups },
                In / 16,
                { Layout<Line>::Lanes },
                In % 16
            ]>()
        }
        .tile::<m![{ Of<Rows> }], 1, Self::BandMut>(at)
    }
}

/// The live rows of an HBM envelope projected into BF16 rows of a window of it, a band of rows at
/// a time: the dense live contraction stages each band's live lane groups from where they are
/// stored, and the projected rows are written at their rows of the window, which starts with the
/// live rows' first band. The live rows start at a band, and a band past them costs nothing, so a
/// few live rows of a wide envelope cost one band.
impl<'i, 'o, In, Out, Rows, Window, S, O, W> Op<(Live<(), Rows>, &'i S, &'o mut O)>
    for Affine<W, ()>
where
    In: AxisName,
    Out: AxisName,
    Rows: AxisName,
    Window: AxisName,
    S: Stored<Rows = Rows>,
    O: Stored<Rows = Window>,
    W: Matrix<In = In, Out = Out>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (live, input, output): (Live<(), Rows>, &'i S, &'o mut O),
    ) {
        let (first, bands) = live.groups::<Band<Rows>>();
        for band in 0..bands {
            let rows = live.subrange(band * BAND, BAND).rows();
            let projected = self.forward(
                &mut *device,
                (
                    Live::<(), Line>::all().subrange(0, rows).with(PhantomData),
                    input.band(first + band),
                    |at| input.group(first + band, at),
                ),
            );
            projected
                .view()
                .to_hbm_view(&mut device.tdma, output.band_mut(band));
        }
    }
}

/// Every row of an HBM envelope into `output` as unrounded FP32 sums: each weight tile `tile(at)`
/// loads once, and every band of rows is written beside it.
impl<'i, 'o, In: AxisName, Out: AxisName, Chips: M, Rows: AxisName, T>
    Op<(
        &'i HbmTensor<bf16, Chips, m![Rows, In]>,
        &'o mut HbmTensor<f32, Chips, m![Rows, Out]>,
    )> for Linear<In, Out, Chips, T>
where
    T: Borrow<HbmTensor<bf16, Chips, m![Out, In]>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, output): (
            &'i HbmTensor<bf16, Chips, m![Rows, In]>,
            &'o mut HbmTensor<f32, Chips, m![Rows, Out]>,
        ),
    ) {
        type B<Rows> = Band<Rows>;
        type Slab<In, Out, Rows> = m![
            { SlabIndex<In, Out> } = 1 # ({ SlabIndex<In, Out> }),
            Rows,
            Out % ({ Block<In, Out> }) / 2,
            Out % 2
        ];
        // SAFETY: the bands split the row axis in row order; every cell keeps its address.
        let input = unsafe { input.view().reshape::<Chips, Banded<Rows, In>>() };
        for at in 0..<Passes<In, Out> as M>::SIZE {
            let weight = self.tensor().tile::<Contraction<In, Out>>(device, at);
            for band in 0..Bands::<Rows>::SIZE {
                let rows = Stage::<In, Out, B<Rows>, Batch<In, Out, B<Rows>>>(PhantomData).forward(
                    &mut *device,
                    input.tile::<m![{ Symbol<Bands<Rows>> }], 1, m![
                        { Symbol<Bands<Rows>> } = 1 # ({ Symbol<Bands<Rows>> }),
                        { Symbol<B<Rows>> },
                        In
                    ]>(band),
                );
                let mut result: Wide<In, Out, B<Rows>, Chips> = DmTensor::new();
                Contraction::<In, Out>::batches::<_, Batch<In, Out, B<Rows>>, _, _>(
                    device,
                    &rows,
                    &weight,
                    &Unrounded,
                    &mut result,
                    at,
                );
                result
                    .view()
                    .tile::<SlabIndex<In, Out>, 1, Slab<In, Out, B<Rows>>>(at)
                    .to_hbm_view(
                        &mut device.tdma,
                        // SAFETY: the bands split the row axis in row order; every cell keeps its
                        // address.
                        unsafe { output.view_mut().reshape::<Chips, Banded<Rows, Out>>() }
                            .tile::<m![{ Symbol<Bands<Rows>> }], 1, m![
                                { Symbol<Bands<Rows>> } = 1 #{!} ({ Symbol<Bands<Rows>> }),
                                { Symbol<B<Rows>> },
                                Out
                            ]>(band)
                            .tile::<SlabIndex<In, Out>, 1, m![
                                { Symbol<Bands<Rows>> } = 1 #{!} ({ Symbol<Bands<Rows>> }),
                                { Symbol<B<Rows>> },
                                { SlabSelected<In, Out> },
                                Out % (Out / ({ Passes<In, Out> }))
                            ]>(at),
                    );
            }
        }
    }
}

/// The contraction tiles the tiling names: they divide the widths, fill the slices and fit a
/// slice's registers and data memory.
impl<In: AxisName, Out: AxisName> Contraction<In, Out> {
    const TILED: () = {
        let c = Self::config();
        assert!(contraction::legal(
            In::SIZE,
            Out::SIZE,
            (c.chunk, c.block, c.passes)
        ));
    };
}

/// Each slice keeps one output block of a tile over one contraction part; tile `at` is the
/// `at`-th of the passes.
impl<In: AxisName, Out: AxisName> Tiling<In, Out> for Contraction<In, Out> {
    type Cluster = Cluster<In, Out>;
    type Slices = ProjectionSlices<In, Out>;
    type Rows = m![Out % ({ Block<In, Out> })];
    type Inputs = Chunked<In, Out>;
    type Restorations = Restorations<In, Out>;
    type Restored = m![Out % ({ Symbol<RestoredRows<In, Out>> })];

    fn to_dm<D: MaterializableScalar, X: M, K: M, E: M>(
        device: &mut Device,
        tensor: &HbmTensor<D, X, m![Out, { K }]>,
        at: usize,
    ) -> DmTensor<D, X, Cluster<In, Out>, ProjectionSlices<In, Out>, E> {
        const { Self::TILED };
        tensor
            .view()
            .tile::<SlabIndex<In, Out>, 1, m![Out % (Out / ({ Passes<In, Out> })) # Out, { K }]>(at)
            .to_dm(&mut device.tdma)
    }
}

/// Output rows of a block one restoration scales: a power of two dividing the block, at most 64,
/// so their FP32 scales take at most 2 KiB of the VRF.
#[derive(Clone, Debug)]
pub(crate) struct RestoredRows<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for RestoredRows<In, Out> {
    const NAME: Ident = Ident::new("RestoredRows");
    const SIZE: usize = contraction::restored(Contraction::<In, Out>::config().block);
}

/// The restorations of a slice's block.
type Restorations<In, Out> = m![Out % ({ Block<In, Out> }) / ({ Symbol<RestoredRows<In, Out>> })];

/// One pass's FP32 sums of a batch of `B` of the `Rows`: every output of each slice's block for
/// every row lane, before the contraction parts meet.
type Sums<'l, In, Out, Rows, B, X> = ContractTensor<
    'l,
    { Tu::Main },
    f32,
    X,
    Cluster<In, Out>,
    ProjectionSlices<In, Out>,
    m![Out % ({ Block<In, Out> }), { Count<Rows, B> }],
    m![{ Layout<Rows>::Lanes } # 8],
>;

impl<In: AxisName, Out: AxisName> Contraction<In, Out> {
    /// A weight slab against one batch's lanes: each slice's `Rows` outputs for each `Time` group
    /// of `Lane` lanes before the parts meet, streamed a `D` packet and contracted two at a time.
    pub(crate) fn sums<
        'l,
        D: MaterializableScalar + ContractionCast<Output = f32>,
        C: M,
        S: M,
        Rows: M,
        Inputs: M,
        Time: M,
        Lane: M,
        X: M,
    >(
        device: &'l mut Device,
        weight: &'l DmTensor<D, X, C, S, m![{ Rows }, { Inputs }]>,
        lanes: &'l TrfTensor<D, X, C, S, Lane, m![{ Time }, { Inputs }]>,
    ) -> ContractTensor<'l, { Tu::Main }, f32, X, C, S, m![{ Rows }, { Time }], m![{ Lane } # 8]>
    {
        device
            .main
            .begin(weight.view())
            .fetch::<m![{ Rows }, { Inputs } / ({ Word<D> })], m![In % ({ Word<D> })]>()
            .collect::<m![{ Rows }, { Inputs } / ({ Word<D> })], m![In % ({ Word<D> })]>()
            .contract_outer::<
                m![{ Rows }, { Inputs } / ({ Symbol<Step<D>> }), { Time }],
                m![In % ({ Symbol<Step<D>> })],
                _,
                _,
                _,
            >(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Rows }, { Time }]>()
            .contract_lane::<m![{ Rows }, { Time }], m![{ Lane } # 8]>(LaneMode::Interleaved)
    }

    /// Every batch of `B` of the staged rows against weight slab `slab`, each pass's sums written
    /// by `epilogue`.
    fn batches<Rows: AxisName, B: AxisName, X: M, E: Epilogue<In, Out, Rows, B, X>>(
        device: &mut Device,
        rows: &Staged<In, Out, Rows, X>,
        weight: &Resident<In, Out, X>,
        epilogue: &E,
        result: &mut E::Result,
        slab: usize,
    ) {
        #[unroll]
        for at in 0..<m![Rows / ({ Symbol<B> })] as M>::SIZE {
            let lanes = rows.forward(&mut *device, (at, PhantomData::<B>));
            epilogue.store(Self::sums(device, weight, &lanes), result, slab, at);
        }
    }

    /// Every row of `input`, staged once, against every slab of `weight`; `epilogue(slab)` writes
    /// that slab's sums.
    fn resident<Rows: AxisName, V: Source, W, E>(
        device: &mut Device,
        weight: &W,
        input: V,
        mut epilogue: impl FnMut(&mut Device, usize) -> E,
    ) -> Result<In, Out, Rows, V::Chips>
    where
        W: Matrix<V::Chips, In = In, Out = Out>,
        E: Epilogue<
                In,
                Out,
                Rows,
                Batch<In, Out, Rows>,
                V::Chips,
                Result = Result<In, Out, Rows, V::Chips>,
            >,
    {
        let rows =
            Stage::<In, Out, Rows, Batch<In, Out, Rows>>(PhantomData).forward(&mut *device, input);
        let mut pass = |device: &mut Device,
                        weight: &Resident<In, Out, V::Chips>,
                        result: &mut Result<In, Out, Rows, V::Chips>,
                        slab: usize| {
            let epilogue = epilogue(device, slab);
            Self::batches(device, &rows, weight, &epilogue, result, slab);
        };
        let mut result = DmTensor::new();
        #[unroll]
        for slab in 0..<Passes<In, Out> as M>::SIZE {
            let weight = weight.tile::<Self>(device, slab);
            pass(device, &weight, &mut result, slab);
        }
        result
    }
}

/// What a pass does with its sums: sum them over the contraction parts and write them in its
/// result.
trait Epilogue<In: AxisName, Out: AxisName, Rows: AxisName, B: AxisName, X: M> {
    type Result;

    /// Writes the sums of slab `slab` and batch `at`.
    fn store(
        &self,
        sums: Sums<'_, In, Out, Rows, B, X>,
        result: &mut Self::Result,
        slab: usize,
        at: usize,
    );
}

/// The sums rounded once to BF16.
pub(crate) struct Rounded;

impl Rounded {
    /// Reduced sums rounded once to BF16 into `into`, `Packet` outputs a packet in `Grouped` order.
    pub(super) fn finish<
        C: M,
        Reduced: M,
        Rows: M,
        Time: M,
        Lane: M,
        Grouped: M,
        Packet: M,
        E: M,
        X: M,
        const O: stage::VeOrder,
    >(
        reduced: VectorInterSliceReduceTensor<
            '_,
            { Tu::Main },
            f32,
            X,
            C,
            Reduced,
            m![{ Rows }, { Time }],
            m![{ Lane } # 8],
            O,
        >,
        into: DmTensorViewMut<'_, bf16, X, C, Reduced, E>,
    ) {
        Self::tail::<_, _, _, _, Lane, Grouped, Packet, _, _>(reduced.vector_final(), into);
    }

    /// Final FP32 sums rounded once to BF16 into `into`, `Packet` outputs a packet in `Grouped`
    /// order; `L` is the row lanes a packet held before rounding.
    pub(crate) fn tail<C: M, S: M, Time: M, P: M, L: M, Grouped: M, Packet: M, E: M, X: M>(
        sums: VectorFinalTensor<'_, { Tu::Main }, f32, X, C, S, Time, P>,
        into: DmTensorViewMut<'_, bf16, X, C, S, E>,
    ) {
        Self::rounded::<bf16, _, _, _, _, L, Grouped, Packet, _>(sums).commit_view(into);
    }

    /// Final FP32 sums rounded once to `T`, `Packet` outputs a packet in `Grouped` order, as the
    /// stream a caller commits; `L` is the row lanes a packet held before rounding.
    pub(crate) fn rounded<
        'l,
        T: MaterializableScalar,
        C: M,
        S: M,
        Time: M,
        P: M,
        L: M,
        Grouped: M,
        Packet: M,
        X: M,
    >(
        sums: VectorFinalTensor<'l, { Tu::Main }, f32, X, C, S, Time, P>,
    ) -> CommitTrimTensor<'l, { Tu::Main }, T, X, C, S, Grouped, Packet>
    where
        f32: CastEngineCast<T>,
    {
        sums.cast::<T, m![{ L } # ({ Word<T> })]>()
            .transpose::<Grouped, m![{ Packet } # ({ Word<T> })]>()
            .commit_trim::<Packet>()
    }

    /// The table's reduced sums rounded into the cells of slab `slab` and batch `at`, four outputs
    /// a packet.
    fn round<
        In: AxisName,
        Out: AxisName,
        Rows: AxisName,
        B: AxisName,
        X: M,
        const O: stage::VeOrder,
    >(
        reduced: VectorInterSliceReduceTensor<
            '_,
            { Tu::Main },
            f32,
            X,
            Cluster<In, Out>,
            Reduced<In, Out>,
            m![Out % ({ Block<In, Out> }), { Count<Rows, B> }],
            m![{ Layout<Rows>::Lanes } # 8],
            O,
        >,
        result: &mut Result<In, Out, Rows, X>,
        slab: usize,
        at: usize,
    ) {
        Self::finish::<
            _,
            _,
            _,
            _,
            _,
            m![Out % ({ Block<In, Out> }) / 4, { Count<Rows, B> }, { Layout<Rows>::Lanes }],
            m![Out % 4],
            _,
            _,
            O,
        >(
            reduced,
            Contraction::<In, Out>::cells::<Rows, B, bf16, X, 4>(result.view_mut(), slab, at),
        );
    }
}

impl<In: AxisName, Out: AxisName, Rows: AxisName, B: AxisName, X: M> Epilogue<In, Out, Rows, B, X>
    for Rounded
{
    type Result = Result<In, Out, Rows, X>;

    fn store(
        &self,
        sums: Sums<'_, In, Out, Rows, B, X>,
        result: &mut Self::Result,
        slab: usize,
        at: usize,
    ) {
        Self::round(
            sums.vector_init()
                .vector_inter_slice_reduce::<
                    Reduced<In, Out>,
                    m![Out % ({ Block<In, Out> }), { Count<Rows, B> }],
                >(InterSliceReduceOpF32::Add),
            result,
            slab,
            at,
        );
    }
}

/// Each contraction part's share of an FP32 offset added to its sums, then rounded once to BF16.
struct Offset<In: AxisName, Out: AxisName, X: M>(Offsets<In, Out, X>);

impl<In: AxisName, Out: AxisName, Rows: AxisName, B: AxisName, X: M> Epilogue<In, Out, Rows, B, X>
    for Offset<In, Out, X>
{
    type Result = Result<In, Out, Rows, X>;

    fn store(
        &self,
        sums: Sums<'_, In, Out, Rows, B, X>,
        result: &mut Self::Result,
        slab: usize,
        at: usize,
    ) {
        Rounded::round(
            sums.vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, &self.0)
                .vector_inter_slice_reduce::<
                    Reduced<In, Out>,
                    m![Out % ({ Block<In, Out> }), { Count<Rows, B> }],
                >(InterSliceReduceOpF32::Add),
            result,
            slab,
            at,
        );
    }
}

/// The FP32 sums, two outputs a packet.
pub(crate) struct Unrounded;

impl Unrounded {
    /// Sums reduced over the parts, `Packet` outputs a packet in `Grouped` order, ready to commit.
    pub(crate) fn finish<
        'l,
        C: M,
        S: M,
        Reduced: M,
        Rows: M,
        Time: M,
        Lane: M,
        Grouped: M,
        Packet: M,
        X: M,
    >(
        sums: ContractTensor<
            'l,
            { Tu::Main },
            f32,
            X,
            C,
            S,
            m![{ Rows }, { Time }],
            m![{ Lane } # 8],
        >,
    ) -> CommitTrimTensor<'l, { Tu::Main }, f32, X, C, Reduced, Grouped, Packet> {
        Self::tail(
            sums.vector_init()
                .vector_inter_slice_reduce::<Reduced, m![{ Rows }, { Time }]>(
                    InterSliceReduceOpF32::Add,
                )
                .vector_final(),
        )
    }

    /// Final FP32 sums, `Packet` outputs a packet in `Grouped` order, ready to commit.
    pub(crate) fn tail<'l, C: M, S: M, Time: M, P: M, Grouped: M, Packet: M, X: M>(
        sums: VectorFinalTensor<'l, { Tu::Main }, f32, X, C, S, Time, P>,
    ) -> CommitTrimTensor<'l, { Tu::Main }, f32, X, C, S, Grouped, Packet> {
        sums.transpose::<Grouped, m![{ Packet } # 8]>()
            .commit_trim::<Packet>()
    }
}

impl<In: AxisName, Out: AxisName, Rows: AxisName, B: AxisName, X: M> Epilogue<In, Out, Rows, B, X>
    for Unrounded
{
    type Result = Wide<In, Out, Rows, X>;

    fn store(
        &self,
        sums: Sums<'_, In, Out, Rows, B, X>,
        result: &mut Self::Result,
        slab: usize,
        at: usize,
    ) {
        Self::finish::<
            _,
            _,
            Reduced<In, Out>,
            _,
            _,
            _,
            m![Out % ({ Block<In, Out> }) / 2, { Count<Rows, B> }, { Layout<Rows>::Lanes }],
            m![Out % 2],
            _,
        >(sums)
        .commit_view(Contraction::<In, Out>::cells::<Rows, B, f32, X, 2>(
            result.view_mut(),
            slab,
            at,
        ));
    }
}

/// A weight slab's per-output offsets, each part's share in every slice of that part: the parts'
/// sums carry the whole offset once they are reduced.
type Offsets<In, Out, X = m![1]> = VrfTensor<
    f32,
    X,
    Cluster<In, Out>,
    ProjectionSlices<In, Out>,
    m![Out % ({ Block<In, Out> }), 1 # 8],
>;

pub(crate) struct Biased<Rows>(pub(crate) PhantomData<Rows>);

impl<'i, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, S: M, E: M, X: M>
    Op<(
        (
            &HbmTensor<bf16, X, m![Out, In]>,
            &HbmTensor<f32, X, m![Out, 1 # 8]>,
        ),
        DmTensorView<'i, bf16, X, Chips, S, E>,
    )> for Biased<Rows>
{
    type Output = Result<In, Out, Rows, X>;

    /// Each output `bf16(x . w + bias)` with the bias added to the FP32 sum, so the output rounds
    /// once; each offset is an eight-element FP32 vector.
    fn forward(
        &self,
        device: &mut Device,
        ((weight, bias), input): (
            (
                &HbmTensor<bf16, X, m![Out, In]>,
                &HbmTensor<f32, X, m![Out, 1 # 8]>,
            ),
            DmTensorView<'i, bf16, X, Chips, S, E>,
        ),
    ) -> Self::Output {
        Contraction::<In, Out>::resident(device, weight, input, |device, slab| {
            let in_dm: DmTensor<
                f32,
                X,
                Cluster<In, Out>,
                ProjectionSlices<In, Out>,
                m![Out % ({ Block<In, Out> }), 1 # 8],
            > = Contraction::<In, Out>::to_dm(device, bias, slab);
            Offset(
                device
                    .main
                    .begin(in_dm.view())
                    .fetch::<m![Out % ({ Block<In, Out> })], m![1 # 8]>()
                    .collect::<m![Out % ({ Block<In, Out> })], m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_trim::<m![1 # 4]>()
                    .vector_fp_binary(
                        FpBinaryOp::MulF(FpMulAlu::Mul0),
                        const { 1.0 / <Parts<In, Out> as M>::SIZE as f32 },
                    )
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .to_vrf(&mut device.sub),
            )
        })
    }
}

/// The result cells weight slab `slab` writes for batch `at` of `B` rows, `P` outputs a packet.
impl<In: AxisName, Out: AxisName> Contraction<In, Out> {
    fn cells<Rows: AxisName, B: AxisName, D: Scalar, X: M, const P: usize>(
        output: DmTensorViewMut<
            '_,
            D,
            X,
            Cluster<In, Out>,
            Reduced<In, Out>,
            m![{ SlabIndex<In, Out> }, Rows, Out % ({ Block<In, Out> }) / { P }, Out % { P }],
        >,
        slab: usize,
        at: usize,
    ) -> DmTensorViewMut<
        '_,
        D,
        X,
        Cluster<In, Out>,
        Reduced<In, Out>,
        m![
            { SlabSelected<In, Out> },
            Rows / B = 1 #{!} (Rows / B),
            Rows % B,
            Out % ({ Block<In, Out> }) / { P },
            Out % { P }
        ],
    > {
        output
            .tile::<SlabIndex<In, Out>, 1, m![
                { SlabSelected<In, Out> },
                Rows,
                Out % ({ Block<In, Out> }) / { P },
                Out % { P }
            ]>(slab)
            .tile::<m![Rows / B], 1, _>(at)
    }
}

/// FP32 sums of a weight slab's outputs, two to a packet.
type Wide<In, Out, Rows, X = m![1]> = DmTensor<
    f32,
    X,
    Cluster<In, Out>,
    Reduced<In, Out>,
    m![{ SlabIndex<In, Out> }, Rows, Out % ({ Block<In, Out> }) / 2, Out % 2],
>;

impl<In: AxisName, Out: AxisName> Linear<In, Out> {
    pub type Input<Rows: AxisName> = Input<In, Out, Rows>;
    pub type Output<Rows: AxisName> = Output<In, Out, Rows>;
    pub type Chunk = Chunk<In, Out>;
    pub type Block = Block<In, Out>;

    /// Row group `at` of a projection's rows.
    pub fn group<Rows: AxisName>(
        projected: &Result<In, Out, Rows>,
        at: usize,
    ) -> DmTensorView<
        '_,
        bf16,
        m![1],
        Cluster<In, Out>,
        Reduced<In, Out>,
        m![
            { SlabIndex<In, Out> },
            { Layout<Rows>::Selected },
            { Layout<Rows>::Lanes },
            Out % ({ Block<In, Out> }) / 4,
            Out % 4
        ],
    > {
        projected.view().tile::<Layout<Rows>::Groups, 1, _>(at)
    }
}

/// The step shape beside `sums`: several outputs a packet.
impl<In: AxisName, Out: AxisName> Contraction<In, Out> {
    /// A weight slab against one batch's lanes, `Packet` by `Q` a packet: each slice's `Rows`
    /// outputs over `Inputs` inputs for each `Time` group, `Folded` ordering the sums to reduce.
    pub(crate) fn packets<
        'l,
        Rows: M,
        Inputs: M,
        Packet: M,
        Time: M,
        Folded: M,
        Lane: M,
        const Q: usize,
        C: M,
        S: M,
        E: M,
        L: M,
        T: M,
        X: M,
    >(
        device: &'l mut Device,
        weight: &'l DmTensor<bf16, X, C, S, E>,
        lanes: &'l TrfTensor<bf16, X, C, S, L, T>,
    ) -> ContractTensor<'l, { Tu::Main }, f32, X, C, S, Folded, Lane> {
        device
            .main
            .begin(weight.view())
            .fetch::<m![{ Rows }, { Inputs }], m![{ Packet }, In % { Q }]>()
            .collect::<
                m![({ Rows }, { Inputs }), { Packet } / ({ Packet } / 2)],
                m![{ Packet } % ({ Packet } / 2), In % { Q }],
            >()
            .contract_outer::<
                m![{ Rows }, { Inputs }, { Time }],
                m![{ Packet }, In % { Q }],
                _,
                _,
                _,
            >(
                lanes,
            )
            .contract_packet::<Packet>()
            .contract_time::<m![{ Rows }, { Time }]>()
            .contract_lane::<Folded, Lane>(LaneMode::Interleaved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::normalization::tests::random;
    use crate::tensor::live::tests::count;
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![Envelope = 256, Hidden = 4096, Narrow = 1024, Wide = 6144];

    type RowMajor<W> = HbmTensor<bf16, m![1], m![Envelope, W]>;
    type Grouped<W> = HbmTensor<bf16, m![1], m![Envelope / 8, W / 16, Envelope % 8, W % 16]>;
    type Weight<In, Out> = Affine<HbmTensor<bf16, m![1], m![Out, In]>, ()>;

    /// The live rows from row `first` of `rows` projected over `before`'s rows.
    #[device(chip = 1)]
    fn grouped(
        device: &mut Device,
        weight: &Weight<Hidden, Narrow>,
        first: &Value,
        rows: &Value,
        input: &Grouped<Hidden>,
        before: &RowMajor<Narrow>,
    ) -> RowMajor<Narrow> {
        let live = Live::<(), Envelope>::all().subrange(
            Live::<(), Envelope>::read(first).rows(),
            Live::<(), Envelope>::read(rows).rows(),
        );
        let mut output = RowMajor::new();
        before
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
        weight.forward(&mut *device, (live, input, &mut output));
        output
    }

    /// As [`grouped`], from row-major rows into grouped ones.
    #[device(chip = 1)]
    fn row_major(
        device: &mut Device,
        weight: &Weight<Narrow, Wide>,
        first: &Value,
        rows: &Value,
        input: &RowMajor<Narrow>,
        before: &Grouped<Wide>,
    ) -> Grouped<Wide> {
        let live = Live::<(), Envelope>::all().subrange(
            Live::<(), Envelope>::read(first).rows(),
            Live::<(), Envelope>::read(rows).rows(),
        );
        let mut output = Grouped::new();
        before
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
        weight.forward(&mut *device, (live, input, &mut output));
        output
    }

    /// Each live row projected over `weight` (`[out, in]`) in FP64, at its row of the window that
    /// starts with the live rows' first band.
    fn expected(x: &[bf16], weight: &[bf16], (first, rows): (usize, usize)) -> Vec<f64> {
        let inputs = x.len() / Envelope::SIZE;
        x.chunks_exact(inputs)
            .skip(first)
            .take(rows)
            .flat_map(|x| {
                weight.chunks_exact(inputs).map(move |w| {
                    w.iter()
                        .zip(x)
                        .map(|(w, x)| f64::from(w.to_f32()) * f64::from(x.to_f32()))
                        .sum()
                })
            })
            .collect()
    }

    /// The live rows of an envelope, read and written in either stored layout, project as the
    /// FP64 reference does within one BF16 rounding into the window that starts with their first
    /// band, and only their bands of it are written: one row, rows from a later band into the
    /// next, and the whole envelope.
    #[tokio::test(flavor = "current_thread")]
    async fn projects_only_the_live_bands() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        async fn upload<Out: AxisName, In: AxisName>(
            device: &mut Device,
            values: &[bf16],
        ) -> Weight<In, Out> {
            Affine::new(
                HostTensor::<bf16, m![Out, In]>::from_vec(values.to_vec())
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                (),
            )
        }
        let sentinel = 7.0;
        // The window's rows from `offset` hold the reference, its rows past `written` the sentinel.
        let check = |got: &[bf16], width: usize, reference: Vec<f64>, (offset, written)| {
            let rows = reference.len() / width;
            for (index, (got, expected)) in got[offset * width..].iter().zip(&reference).enumerate()
            {
                let got = f64::from(got.to_f32());
                assert!(
                    (got - expected).abs() <= expected.abs() / 128.0 + 1e-2,
                    "{rows} live rows from {offset}, cell {index}: {got} vs {expected}"
                );
            }
            for row in written..Envelope::SIZE {
                assert!(
                    got[row * width..][..width]
                        .iter()
                        .all(|v| v.to_f32() == sentinel),
                    "{rows} live rows from {offset}: row {row} past their bands was written"
                );
            }
        };
        let narrow = random(Narrow::SIZE * Hidden::SIZE, 0x51);
        let wide = random(Wide::SIZE * Narrow::SIZE, 0x52);
        let x = random(Envelope::SIZE * Hidden::SIZE, 0x53);
        let y = random(Envelope::SIZE * Narrow::SIZE, 0x54);
        let narrow_weight = upload::<Narrow, Hidden>(&mut device, &narrow).await;
        let wide_weight = upload::<Wide, Narrow>(&mut device, &wide).await;
        let grouped_x: Grouped<Hidden> =
            HostTensor::<bf16, m![Envelope, Hidden]>::from_vec(x.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let rows_y: RowMajor<Narrow> =
            HostTensor::<bf16, m![Envelope, Narrow]>::from_vec(y.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let filled = |width| vec![bf16::from_f32(sentinel); Envelope::SIZE * width];
        let before_narrow: RowMajor<Narrow> =
            HostTensor::<bf16, m![Envelope, Narrow]>::from_vec(filled(Narrow::SIZE))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let before_wide: Grouped<Wide> =
            HostTensor::<bf16, m![Envelope, Wide]>::from_vec(filled(Wide::SIZE))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        for live in [(0, 1), (64, 70), (0, Envelope::SIZE)] {
            let first = count(live.0).to_hbm(&mut device.pdma).await.unwrap();
            let rows_live = count(live.1).to_hbm(&mut device.pdma).await.unwrap();
            let got = launch(
                grouped,
                (
                    &mut device,
                    &narrow_weight,
                    &first,
                    &rows_live,
                    &grouped_x,
                    &before_narrow,
                ),
            )
            .await
            .unwrap()
            .to_host::<m![Envelope, Narrow]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
            let offset = live.0 % BAND;
            let written = (offset + live.1).div_ceil(BAND) * BAND;
            check(
                &got,
                Narrow::SIZE,
                expected(&x, &narrow, live),
                (offset, written),
            );
            let got = launch(
                row_major,
                (
                    &mut device,
                    &wide_weight,
                    &first,
                    &rows_live,
                    &rows_y,
                    &before_wide,
                ),
            )
            .await
            .unwrap()
            .to_host::<m![Envelope, Wide]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
            check(
                &got,
                Wide::SIZE,
                expected(&y, &wide, live),
                (offset, written),
            );
        }
    }
}
