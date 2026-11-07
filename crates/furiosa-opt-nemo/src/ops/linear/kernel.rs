use core::marker::PhantomData;

use crate::Module;
use furiosa_opt_std::prelude::*;

use crate::ops::layout::Live;
use crate::ops::layout::Rows as Layout;
use crate::ops::layout::axes;
use crate::ops::linear::Groups;
use crate::ops::linear::{Affine, Linear};

/// How a projection's input rows reach the slices that hold its contraction parts, and how the
/// parts' partial sums meet.
#[derive(Clone, Copy)]
enum Routing {
    /// Every slice holds every row.
    Replicated,
    /// Rows stream to the slices transposed, a packet a slice.
    Transposed,
    /// Rows pass around the slices in a ring.
    Ring,
    /// Each part's slice holds that part of the rows.
    Chunk,
}

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
        let mut top = output / (2 * passes * block);
        while top.is_multiple_of(2) {
            top /= 2;
        }
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
    /// The tiling of each served `(In, Out)`: contraction chunk, output block, slab passes and
    /// routing. They change the emitted operations, so a shape without a tiling is refused.
    // TODO: no rule selects these values and no measurement records them; choose among the legal
    // tilings `Config::new` admits, by a recorded measurement or the oracle's tiling per shape.
    const fn config() -> Config {
        use Routing::*;
        let choice = match (In::SIZE, Out::SIZE) {
            (64, 10752) => (64, 84, 1, Replicated),
            (192, 10752) => (192, 84, 1, Replicated),
            (512, 10752) => (512, 84, 1, Replicated),
            (1024, 4096) => (512, 32, 1, Ring),
            (1024, 6144) => (256, 96, 1, Replicated),
            (1152, 1152) => (288, 36, 1, Replicated),
            (1152, 3456) => (288, 108, 1, Replicated),
            (1152, 4608) => (288, 144, 1, Replicated),
            (2048, 1024) => (128, 32, 1, Chunk),
            (2048, 2048) => (128, 64, 1, Chunk),
            (2048, 4096) => (256, 64, 1, Ring),
            (2048, 12288) => (256, 192, 1, Replicated),
            (2560, 6144) => (640, 64, 1, Ring),
            (2560, 19456) => (320, 304, 1, Replicated),
            (2688, 10752) => (672, 84, 1, Replicated),
            (2688, 24192) => (672, 108, 7, Replicated),
            (2688, 96768) => (672, 108, 7, Replicated),
            (3072, 1024) => (768, 32, 1, Replicated),
            (4096, 2560) => (128, 160, 1, Chunk),
            (4096, 4096) => (128, 512, 1, Chunk),
            (4096, 5120) => (128, 320, 1, Chunk),
            (4096, 6144) => (1024, 64, 1, Transposed),
            (4096, 24576) => (128, 384, 4, Replicated),
            (4096, 32768) => (128, 512, 8, Replicated),
            (4608, 1152) => (576, 36, 1, Replicated),
            (5120, 1024) => (320, 32, 1, Replicated),
            (5120, 4096) => (320, 128, 1, Replicated),
            (5120, 5376) => (320, 168, 1, Replicated),
            (5120, 10240) => (320, 320, 1, Replicated),
            (5120, 16384) => (320, 128, 4, Replicated),
            (5120, 51200) => (640, 80, 10, Replicated),
            (5120, 131072) => (320, 128, 32, Replicated),
            (5376, 64) => (672, 4, 1, Replicated),
            (5376, 192) => (672, 12, 1, Replicated),
            (6144, 2048) => (384, 64, 1, Replicated),
            (8192, 5120) => (512, 160, 1, Replicated),
            (9728, 2560) => (608, 80, 1, Replicated),
            (10240, 2560) => (640, 80, 1, Replicated),
            (10752, 5376) => (672, 168, 1, Replicated),
            (12288, 4096) => (384, 256, 1, Replicated),
            (16384, 4096) => (512, 128, 2, Replicated),
            (16384, 5120) => (512, 160, 2, Replicated),
            (25600, 5120) => (3200, 32, 5, Replicated),
            _ => panic!("unsupported linear dimensions"),
        };
        Config::new(Out::SIZE, choice)
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

pub type Chunk<In, Out> = Symbol<Contraction<In, Out>>;

#[derive(Clone, Debug)]
pub struct OutputBlock<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for OutputBlock<In, Out> {
    const NAME: Ident = Ident::new("OutputBlock");

    const SIZE: usize = {
        let c = Contraction::<In, Out>::config();
        c.block
    };
}

pub type Block<In, Out> = Symbol<OutputBlock<In, Out>>;

#[derive(Clone, Debug)]
pub struct Carving<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for Carving<In, Out> {
    const NAME: Ident = Ident::new("Carving");

    const SIZE: usize = {
        let c = Contraction::<In, Out>::config();
        c.top * c.passes
    };
}

pub type Carve<In, Out> = Symbol<Carving<In, Out>>;

#[derive(Clone, Debug)]
pub struct TopDigit<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for TopDigit<In, Out> {
    const NAME: Ident = Ident::new("TopDigit");

    const SIZE: usize = {
        let c = Contraction::<In, Out>::config();
        c.top
    };
}

pub type Top<In, Out> = Symbol<TopDigit<In, Out>>;

#[derive(Clone, Debug)]
pub struct PaddingCount<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for PaddingCount<In, Out> {
    const NAME: Ident = Ident::new("PaddingCount");

    const SIZE: usize = {
        let c = Contraction::<In, Out>::config();
        c.padded
    };
}

pub type Padded<In, Out> = Symbol<PaddingCount<In, Out>>;

#[derive(Clone, Debug)]
pub struct PassCount<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for PassCount<In, Out> {
    const NAME: Ident = Ident::new("PassCount");

    const SIZE: usize = {
        let c = Contraction::<In, Out>::config();
        c.passes
    };
}

pub type Passes<In, Out> = Symbol<PassCount<In, Out>>;

pub type Input<In, Out, Rows> = m![In / ({ Chunk<In, Out> }), Rows, In % ({ Chunk<In, Out> })];
pub type Output<In, Out, Rows> = m![Out / ({ Block<In, Out> }), Rows, Out % ({ Block<In, Out> })];
type Parts<In, Out> = m![In / ({ Chunk<In, Out> })];
type Chunked<In, Out> = m![In % ({ Chunk<In, Out> })];

pub type Cluster<In, Out> =
    m![Out % (Out / ({ Carve<In, Out> })) / ((Out / ({ Carve<In, Out> })) / 2)];

type Blocks<In, Out> = m![Out % ((Out / ({ Carve<In, Out> })) / 2) / ({ Block<In, Out> })];
type Idle<In, Out> = m![256 / ({ Blocks<In, Out> }) / ({ Parts<In, Out> }) / ({ Padded<In, Out> })];
type ProjectionSlices<In, Out> = m![Out / (Out / ({ Top<In, Out> })) # ({ Padded<In, Out> }), { Blocks<In, Out> }, 1 # ({ Idle<In, Out> }), { Parts<In, Out> }];
pub type Reduced<In, Out> = m![Out / (Out / ({ Top<In, Out> })) # ({ Padded<In, Out> }), { Blocks<In, Out> }, 1 # ({ Idle<In, Out> }), 1 # ({ Parts<In, Out> })];
type SlabIndex<In, Out> = m![Out / (Out / ({ Passes<In, Out> }))];
type SlabWeight<In, Out> = m![Out % (Out / ({ Passes<In, Out> })) # Out, In];
type SlabSelected<In, Out> = m![{ SlabIndex<In, Out> } = 1 #{!} ({ SlabIndex<In, Out> })];

pub type Element<In, Out, Rows> =
    m![{ SlabIndex<In, Out> }, Rows, Out % ({ Block<In, Out> }) / 4, Out % 4];

pub type Result<In, Out, Rows, X = m![1]> =
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
pub struct Batch<In, Out, Rows>(PhantomData<(In, Out, Rows)>);

impl<In: AxisName, Out: AxisName, Rows: AxisName> AxisName for Batch<In, Out, Rows> {
    const NAME: Ident = Ident::new("Batch");

    const SIZE: usize =
        if Rows::SIZE <= 32 && Rows::SIZE * <Chunked<In, Out> as M>::SIZE * 2 <= 65_536 {
            Rows::SIZE
        } else {
            axes::Lanes::<Rows>::SIZE
        };
}

/// The rows of an HBM envelope one staging lands beside a resident weight slab.
#[derive(Clone, Debug)]
pub struct Band<Rows>(PhantomData<Rows>);

impl<Rows: AxisName> AxisName for Band<Rows> {
    const NAME: Ident = Ident::new("Band");

    const SIZE: usize = {
        assert!(Rows::SIZE.is_multiple_of(BAND));
        BAND
    };
}

/// The bands of an HBM envelope, in row order.
#[derive(Clone, Debug)]
pub struct Bands<Rows>(PhantomData<Rows>);

impl<Rows: AxisName> AxisName for Bands<Rows> {
    const NAME: Ident = Ident::new("Bands");
    const SIZE: usize = Rows::SIZE / Band::<Rows>::SIZE;
}

/// Rows a band holds: with the widest resident slab of the table its staged copy still fits a
/// slice's data memory.
const BAND: usize = 64;

/// An HBM envelope's rows as its bands.
type Banded<Rows, Width> = m![{ Symbol<Bands<Rows>> }, { Symbol<Band<Rows>> }, Width];

/// Lane groups in one batch of `B` rows.
type Count<Rows, B> = m![Rows % B / ({ Symbol<axes::Lanes<Rows>> })];

type Lanes<In, Out, Rows, B, X = m![1]> = TrfTensor<
    bf16,
    X,
    Cluster<In, Out>,
    ProjectionSlices<In, Out>,
    m![{ Layout<Rows>::Lanes }],
    m![{ Count<Rows, B> }, { Chunked<In, Out> }],
>;

type Packet<Rows> = m![(16, { Layout<Rows>::Lanes }) / 1];

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
            Routing::Transposed if B::SIZE > axes::Lanes::<Rows>::SIZE => Routing::Ring,
            Routing::Chunk if axes::Lanes::<Rows>::SIZE > 1 => Routing::Ring,
            routing => routing,
        }
    }
}

struct Stage<In, Out, Rows, B>(PhantomData<(In, Out, Rows, B)>);

impl<'i, In: AxisName, Out: AxisName, Rows: AxisName, B: AxisName, X: M, Chips: M, S: M, E: M>
    Module<(&mut Device, DmTensorView<'i, bf16, X, Chips, S, E>)> for Stage<In, Out, Rows, B>
{
    type Output = Staged<In, Out, Rows, X>;

    fn forward(
        &self,
        (device, input): (&mut Device, DmTensorView<'i, bf16, X, Chips, S, E>),
    ) -> Self::Output {
        match const { Staged::<In, Out, Rows, X>::route::<B>() } {
            Routing::Replicated => {
                let mut rows = DmTensor::new();
                input.to_dm_view(&mut device.tdma, rows.view_mut());
                Staged::Replicated(rows)
            }

            Routing::Transposed => {
                let mut rows = DmTensor::new();
                input.to_dm_view(&mut device.tdma, rows.view_mut());
                Staged::Transposed(rows)
            }

            Routing::Ring => {
                let mut rows = DmTensor::new();
                input.to_dm_view(&mut device.tdma, rows.view_mut());
                Staged::Ring(rows)
            }

            Routing::Chunk => {
                let mut rows = DmTensor::new();
                input.to_dm_view(&mut device.tdma, rows.view_mut());
                Staged::Chunk(rows)
            }
        }
    }
}

impl<In: AxisName, Out: AxisName, Rows: AxisName, X: M, B: AxisName>
    Module<(&mut Device, (usize, PhantomData<B>))> for Staged<In, Out, Rows, X>
{
    type Output = Lanes<In, Out, Rows, B, X>;

    fn forward(&self, (device, (at, _)): (&mut Device, (usize, PhantomData<B>))) -> Self::Output {
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
                .begin(input.view().tile::<Layout<Rows>::Groups, 1, m![1, { Layout<Rows>::Selected }, In % ({ Packet<Rows> })]>(at))
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

/// A weight form the contraction loads as BF16 slabs: BF16 coefficients as stored, or E4M3
/// coefficients each output row restores by its FP32 scale.
pub(crate) trait Matrix {
    type In: AxisName;
    type Out: AxisName;

    /// The whole weight, for a contraction of one pass.
    fn resident(&self, device: &mut Device) -> Resident<Self::In, Self::Out>;

    /// Weight slab `slab` of a contraction that streams the weight in passes.
    fn slab(&self, device: &mut Device, slab: usize) -> Resident<Self::In, Self::Out>;
}

impl<'v, In, Out, Rows, Chips, S, E, G, W, Group>
    Module<(
        &mut Device,
        (
            Live<Rows>,
            DmTensorView<'v, bf16, m![1], Chips, S, E>,
            Group,
        ),
    )> for Affine<W, ()>
where
    In: AxisName,
    Out: AxisName,
    Rows: AxisName,
    Chips: M,
    S: M,
    E: M,
    G: M,
    W: Matrix<In = In, Out = Out>,
    Group: Fn(usize) -> DmTensorView<'v, bf16, m![1], Chips, S, G>,
{
    type Output = Result<In, Out, Rows>;

    fn forward(
        &self,
        (device, (live, input, group)): (
            &mut Device,
            (
                Live<Rows>,
                DmTensorView<'v, bf16, m![1], Chips, S, E>,
                Group,
            ),
        ),
    ) -> Self::Output {
        if live.count() == <Groups<Rows> as M>::SIZE {
            let batches = |device: &mut Device,
                           rows: &Staged<In, Out, Rows>,
                           weight: &Resident<In, Out>,
                           result: &mut Result<In, Out, Rows>,
                           slab: usize| {
                #[unroll]
                for at in 0..<m![Rows / ({ Symbol<Batch<In, Out, Rows>> })] as M>::SIZE {
                    let lanes =
                        rows.forward((&mut *device, (at, PhantomData::<Batch<In, Out, Rows>>)));
                    Pass.forward((
                        &mut *device,
                        (
                            weight,
                            &lanes,
                            Contraction::<In, Out>::landing::<Rows, Batch<In, Out, Rows>, _>(
                                result.view_mut(),
                                slab,
                                at,
                            ),
                        ),
                    ));
                }
            };
            let rows = Stage::<In, Out, Rows, Batch<In, Out, Rows>>(PhantomData)
                .forward((&mut *device, input));
            // A weight that fits DM loads once; a wider one streams slab by slab.
            return if const { <Passes<In, Out> as M>::SIZE == 1 } {
                let weight = self.weight.resident(device);
                let mut result = DmTensor::new();
                batches(device, &rows, &weight, &mut result, 0);
                result
            } else {
                let mut result = DmTensor::new();
                #[unroll]
                for slab in 0..<Passes<In, Out> as M>::SIZE {
                    let weight = self.weight.slab(device, slab);
                    batches(device, &rows, &weight, &mut result, slab);
                }
                result
            };
        }
        let rows = {
            match const { Staged::<In, Out, Rows>::route::<axes::Lanes<Rows>>() } {
                Routing::Replicated => {
                    let mut rows: Replicated<In, Out, Rows> = DmTensor::new();
                    for at in 0..live.count() {
                        group(at).to_dm_view(
                            &mut device.tdma,
                            rows.view_mut().tile::<Layout<Rows>::Groups, 1, Layout<Rows>::GroupMut<Chunked<In, Out>>>(at),
                        );
                    }
                    Staged::<In, Out, Rows>::Replicated(rows)
                }
                Routing::Transposed => {
                    let mut rows: Transposed<In, Out, Rows> = DmTensor::new();
                    for at in 0..live.count() {
                        group(at).to_dm_view(
                            &mut device.tdma,
                            rows.view_mut().tile::<Layout<Rows>::Groups, 1, m![1, { Layout<Rows>::SelectedMut }, In % ({ Packet<Rows> })]>(at),
                        );
                    }
                    Staged::<In, Out, Rows>::Transposed(rows)
                }
                Routing::Ring => {
                    let mut rows: Ring<In, Out, Rows> = DmTensor::new();
                    for at in 0..live.count() {
                        group(at).to_dm_view(
                            &mut device.tdma,
                            rows.view_mut().tile::<Layout<Rows>::Groups, 1, m![1, { Layout<Rows>::GroupMut<m![In % 16]> }]>(at),
                        );
                    }
                    Staged::<In, Out, Rows>::Ring(rows)
                }
                Routing::Chunk => {
                    let mut rows: ChunkRows<In, Out, Rows> = DmTensor::new();
                    for at in 0..live.count() {
                        group(at).to_dm_view(
                            &mut device.tdma,
                            rows.view_mut().tile::<Layout<Rows>::Groups, 1, m![1, { Layout<Rows>::GroupMut<Chunked<In, Out>> }]>(at),
                        );
                    }
                    Staged::<In, Out, Rows>::Chunk(rows)
                }
            }
        };
        let mut result = DmTensor::new();
        #[unroll]
        for slab in 0..<Passes<In, Out> as M>::SIZE {
            let weight = self.weight.slab(device, slab);
            for group in 0..live.count() {
                let lanes = rows.forward((&mut *device, (group, PhantomData::<axes::Lanes<Rows>>)));
                Pass.forward((
                    &mut *device,
                    (
                        &weight,
                        &lanes,
                        Contraction::<In, Out>::landing::<Rows, axes::Lanes<Rows>, _>(
                            result.view_mut(),
                            slab,
                            group,
                        ),
                    ),
                ));
            }
        }
        result
    }
}

/// Every row of an HBM envelope into `output` as unrounded FP32 sums: each weight slab `slab(at)`
/// loads once, and every band of rows lands beside it.
impl<'i, 'o, In: AxisName, Out: AxisName, X: M, Rows: AxisName>
    Module<(
        &mut Device,
        (
            &'i HbmTensor<bf16, X, m![Rows, In]>,
            &'o mut HbmTensor<f32, X, m![Rows, Out]>,
        ),
    )> for Linear<In, Out, X>
{
    type Output = ();

    fn forward(
        &self,
        (device, (input, output)): (
            &mut Device,
            (
                &'i HbmTensor<bf16, X, m![Rows, In]>,
                &'o mut HbmTensor<f32, X, m![Rows, Out]>,
            ),
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
        let input = unsafe { input.view().reshape::<X, Banded<Rows, In>>() };
        for at in 0..<Passes<In, Out> as M>::SIZE {
            let weight = Slab.forward((&mut *device, (self.tensor(), at)));
            for band in 0..Bands::<Rows>::SIZE {
                let rows = {
                    let input = input.tile::<m![{ Symbol<Bands<Rows>> }], 1, m![
                        { Symbol<Bands<Rows>> } = 1 # ({ Symbol<Bands<Rows>> }),
                        { Symbol<B<Rows>> },
                        In
                    ]>(band);
                    match const { Staged::<In, Out, B<Rows>, X>::route::<Batch<In, Out, B<Rows>>>() }
                    {
                        Routing::Replicated => {
                            let mut rows = DmTensor::new();
                            input.to_dm_view(&mut device.tdma, rows.view_mut());
                            Staged::<In, Out, B<Rows>, X>::Replicated(rows)
                        }
                        Routing::Transposed => {
                            let mut rows = DmTensor::new();
                            input.to_dm_view(&mut device.tdma, rows.view_mut());
                            Staged::<In, Out, B<Rows>, X>::Transposed(rows)
                        }
                        Routing::Ring => {
                            let mut rows = DmTensor::new();
                            input.to_dm_view(&mut device.tdma, rows.view_mut());
                            Staged::<In, Out, B<Rows>, X>::Ring(rows)
                        }
                        Routing::Chunk => {
                            let mut rows = DmTensor::new();
                            input.to_dm_view(&mut device.tdma, rows.view_mut());
                            Staged::<In, Out, B<Rows>, X>::Chunk(rows)
                        }
                    }
                };
                let mut result: Wide<In, Out, B<Rows>, X> = DmTensor::new();
                {
                    type R<Rows> = Band<Rows>;
                    type B<In, Out, Rows> = Batch<In, Out, Band<Rows>>;
                    let slab = at;
                    #[unroll]
                    for at in
                        0..<m![{ Symbol<R<Rows>> } / ({ Symbol<B<In, Out, Rows>> })] as M>::SIZE
                    {
                        let lanes =
                            rows.forward((&mut *device, (at, PhantomData::<B<In, Out, Rows>>)));
                        {
                            let result = result
                                .view_mut()
                                .tile::<SlabIndex<In, Out>, 1, m![
                                    { SlabSelected<In, Out> },
                                    { Symbol<R<Rows>> },
                                    Out % ({ Block<In, Out> }) / 2,
                                    Out % 2
                                ]>(slab)
                                .tile::<m![{ Symbol<R<Rows>> } / ({ Symbol<B<In, Out, Rows>> })], 1, m![
                                    { SlabSelected<In, Out> },
                                    { Symbol<R<Rows>> } / ({ Symbol<B<In, Out, Rows>> }) = 1 #{!} ({ Symbol<R<Rows>> } / ({ Symbol<B<In, Out, Rows>> })),
                                    { Symbol<R<Rows>> } % ({ Symbol<B<In, Out, Rows>> }),
                                    Out % ({ Block<In, Out> }) / 2,
                                    Out % 2
                                ]>(at);
                            device.main.begin(weight.view())
                                .fetch::<m![Out % ({ Block<In, Out> }), In % ({ Chunk<In, Out> }) / 16], m![In % 16]>()
                                .collect::<m![Out % ({ Block<In, Out> }), In % ({ Chunk<In, Out> }) / 16], m![In % 16]>()
                                .contract_outer::<m![Out % ({ Block<In, Out> }), In % ({ Chunk<In, Out> }) / 32, { Count<R<Rows>, B<In, Out, Rows>> }], m![In % 32], _, _, _>(&lanes)
                                .contract_packet::<m![1]>()
                                .contract_time::<m![Out % ({ Block<In, Out> }), { Count<R<Rows>, B<In, Out, Rows>> }]>()
                                .contract_lane::<m![Out % ({ Block<In, Out> }), { Count<R<Rows>, B<In, Out, Rows>> }], m![{ Layout<R<Rows>>::Lanes } # 8]>(LaneMode::Interleaved)
                                .vector_init()
                                .vector_inter_slice_reduce::<Reduced<In, Out>, m![Out % ({ Block<In, Out> }), { Count<R<Rows>, B<In, Out, Rows>> }]>(InterSliceReduceOpF32::Add)
                                .vector_final()
                                .transpose::<m![Out % ({ Block<In, Out> }) / 2, { Count<R<Rows>, B<In, Out, Rows>> }, { Layout<R<Rows>>::Lanes }], m![Out % 2 # 8]>()
                                .commit_trim::<m![Out % 2]>()
                                .commit_view(result);
                        }
                    }
                }
                result
                    .view()
                    .tile::<SlabIndex<In, Out>, 1, Slab<In, Out, B<Rows>>>(at)
                    .to_hbm_view(
                        &mut device.tdma,
                        // SAFETY: as for the input.
                        unsafe { output.view_mut().reshape::<X, Banded<Rows, Out>>() }
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

/// The contraction tiles the table names: they divide the widths and fill the slices.
impl<In: AxisName, Out: AxisName> Contraction<In, Out> {
    const TILED: () = {
        let c = Self::config();
        assert!(In::SIZE.is_multiple_of(c.chunk));
        assert!(Out::SIZE.is_multiple_of(2 * c.top * c.passes * c.block));
        assert!(256usize.is_multiple_of(
            Out::SIZE / (2 * c.top * c.passes * c.block) * (In::SIZE / c.chunk) * c.padded
        ));
    };
}

impl<In: AxisName, Out: AxisName> Matrix for HbmTensor<bf16, m![1], m![Out, In]> {
    type In = In;
    type Out = Out;
    fn resident(&self, device: &mut Device) -> Resident<In, Out> {
        const { Contraction::<In, Out>::TILED };
        self.to_dm(&mut device.tdma)
    }

    fn slab(&self, device: &mut Device, slab: usize) -> Resident<In, Out> {
        Slab.forward((device, (self, slab)))
    }
}

/// Weight slab `slab` of a BF16 weight on each chip of `X`.
struct Slab;

impl<In: AxisName, Out: AxisName, X: M>
    Module<(&mut Device, (&HbmTensor<bf16, X, m![Out, In]>, usize))> for Slab
{
    type Output = Resident<In, Out, X>;

    fn forward(
        &self,
        (device, (weight, slab)): (&mut Device, (&HbmTensor<bf16, X, m![Out, In]>, usize)),
    ) -> Self::Output {
        const { Contraction::<In, Out>::TILED };
        weight
            .view()
            .tile::<SlabIndex<In, Out>, 1, SlabWeight<In, Out>>(slab)
            .to_dm(&mut device.tdma)
    }
}

/// Output rows of a block one restoration scales: a power of two dividing the block, at most 64,
/// so their FP32 scales take at most 2 KiB of the VRF.
#[derive(Clone, Debug)]
struct RestoredRows<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for RestoredRows<In, Out> {
    const NAME: Ident = Ident::new("RestoredRows");

    const SIZE: usize = {
        let block = Contraction::<In, Out>::config().block;
        let mut rows = 64;
        while !block.is_multiple_of(rows) {
            rows /= 2;
        }
        rows
    };
}

type Restored<In, Out> = Symbol<RestoredRows<In, Out>>;

/// The restorations of a slice's block.
type Restorations<In, Out> = m![Out % ({ Block<In, Out> }) / ({ Restored<In, Out> })];

type Coded<In, Out> = DmTensor<
    f8e4m3,
    m![1],
    Cluster<In, Out>,
    ProjectionSlices<In, Out>,
    m![Out % ({ Block<In, Out> }), In % ({ Chunk<In, Out> })],
>;

type Scales<In, Out> = DmTensor<
    f32,
    m![1],
    Cluster<In, Out>,
    ProjectionSlices<In, Out>,
    m![Out % ({ Block<In, Out> }), 1 # 8],
>;

/// BF16 coefficients of E4M3 ones: `bf16(coefficient * scale)` with the FP32 scale of the output
/// row, as the one-row projection restores a tile.
struct Restore;

impl<In: AxisName, Out: AxisName> Module<(&mut Device, (&Coded<In, Out>, &Scales<In, Out>))>
    for Restore
{
    type Output = Resident<In, Out>;

    fn forward(
        &self,
        (device, (coded, scales)): (&mut Device, (&Coded<In, Out>, &Scales<In, Out>)),
    ) -> Self::Output {
        type Rows<In, Out> = m![Out % ({ Restored<In, Out> })];
        type Selected<In, Out> = m![{ Restorations<In, Out> } = 1 # ({ Restorations<In, Out> })];
        type SelectedMut<In, Out> =
            m![{ Restorations<In, Out> } = 1 #{!} ({ Restorations<In, Out> })];
        let mut restored: Resident<In, Out> = DmTensor::new();
        for part in 0..<Restorations<In, Out> as M>::SIZE {
            let decoded: DmTensor<
                bf16,
                m![1],
                Cluster<In, Out>,
                ProjectionSlices<In, Out>,
                m![{ Rows<In, Out> }, In % ({ Chunk<In, Out> })],
            > = device
                .main
                .begin(coded.view().tile::<Restorations<In, Out>, 1, m![
                    { Selected<In, Out> }, { Rows<In, Out> }, In % ({ Chunk<In, Out> })
                ]>(part))
                .fetch::<m![{ Rows<In, Out> }, In % ({ Chunk<In, Out> }) / 16], m![In % 16]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![{ Rows<In, Out> }, In % ({ Chunk<In, Out> }) / 16], m![In % 16]>()
                .commit_trim::<m![In % 16]>()
                .commit();
            let scale: VrfTensor<
                f32,
                m![1],
                Cluster<In, Out>,
                ProjectionSlices<In, Out>,
                m![{ Rows<In, Out> }, 1 # 8],
            > = device
                .sub
                .begin(scales.view().tile::<Restorations<In, Out>, 1, m![
                    { Selected<In, Out> }, { Rows<In, Out> }, 1 # 8
                ]>(part))
                .fetch::<m![{ Rows<In, Out> }], m![1 # 8]>()
                .collect::<m![{ Rows<In, Out> }], m![1 # 8]>()
                .to_vrf();
            device
            .main
            .begin(decoded.view())
            .fetch::<m![{ Rows<In, Out> }, In % ({ Chunk<In, Out> }) / 16], m![In % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Rows<In, Out> }, In % ({ Chunk<In, Out> }) / 8], m![In % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Rows<In, Out> }, In % ({ Chunk<In, Out> }) / 4], m![In % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![{ Rows<In, Out> }, In % ({ Chunk<In, Out> }) / 8], m![In % 8]>()
            .vector_final()
            .cast::<bf16, m![In % 8 # 16]>()
            .commit_trim::<m![In % 8]>()
            .commit_view(restored.view_mut().tile::<Restorations<In, Out>, 1, m![
                { SelectedMut<In, Out> }, { Rows<In, Out> }, In % ({ Chunk<In, Out> })
            ]>(part));
        }
        restored
    }
}

/// E4M3 coefficients in [output, input] order with one FP32 scale per output row.
impl<In: AxisName, Out: AxisName> Matrix
    for (
        HbmTensor<f8e4m3, m![1], m![Out, In]>,
        HbmTensor<f32, m![1], m![Out, 1 # 8]>,
    )
{
    type In = In;
    type Out = Out;

    fn resident(&self, device: &mut Device) -> Resident<In, Out> {
        const { Contraction::<In, Out>::TILED };
        let coded = self.0.to_dm(&mut device.tdma);
        let scales = self.1.to_dm(&mut device.tdma);
        Restore.forward((device, (&coded, &scales)))
    }

    fn slab(&self, device: &mut Device, slab: usize) -> Resident<In, Out> {
        const { Contraction::<In, Out>::TILED };
        let coded = self
            .0
            .view()
            .tile::<SlabIndex<In, Out>, 1, SlabWeight<In, Out>>(slab)
            .to_dm(&mut device.tdma);
        let scales = self
            .1
            .view()
            .tile::<SlabIndex<In, Out>, 1, m![Out % (Out / ({ Passes<In, Out> })) # Out, 1 # 8]>(
                slab,
            )
            .to_dm(&mut device.tdma);
        Restore.forward((device, (&coded, &scales)))
    }
}

struct Pass;

impl<'r, In: AxisName, Out: AxisName, Rows: AxisName, B: AxisName, Element: M>
    Module<(
        &mut Device,
        (
            &Resident<In, Out>,
            &Lanes<In, Out, Rows, B>,
            DmTensorViewMut<'r, bf16, m![1], Cluster<In, Out>, Reduced<In, Out>, Element>,
        ),
    )> for Pass
{
    type Output = ();

    fn forward(
        &self,
        (device, (weight, lanes, result)): (
            &mut Device,
            (
                &Resident<In, Out>,
                &Lanes<In, Out, Rows, B>,
                DmTensorViewMut<'r, bf16, m![1], Cluster<In, Out>, Reduced<In, Out>, Element>,
            ),
        ),
    ) {
        device
        .main
        .begin(weight.view())
        .fetch::<
            m![Out % ({ Block<In, Out> }), In % ({ Chunk<In, Out> }) / 16],
            m![In % 16],
        >()
        .collect::<
            m![Out % ({ Block<In, Out> }), In % ({ Chunk<In, Out> }) / 16],
            m![In % 16],
        >()
        .contract_outer::<
            m![Out % ({ Block<In, Out> }), In % ({ Chunk<In, Out> }) / 32, { Count<Rows, B> }],
            m![In % 32],
            _,
            _,
            _,
        >(lanes)
        .contract_packet::<m![1]>()
        .contract_time::<m![Out % ({ Block<In, Out> }), { Count<Rows, B> }]>()
        .contract_lane::<
            m![Out % ({ Block<In, Out> }), { Count<Rows, B> }],
            m![{ Layout<Rows>::Lanes } # 8],
        >(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<
            Reduced<In, Out>,
            m![Out % ({ Block<In, Out> }), { Count<Rows, B> }],
        >(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
        .transpose::<
            m![Out % ({ Block<In, Out> }) / 4, { Count<Rows, B> }, { Layout<Rows>::Lanes }],
            m![Out % 4 # 16],
        >()
        .commit_trim::<m![Out % 4]>()
        .commit_view(result);
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
    Module<(
        &mut Device,
        (
            (
                &HbmTensor<bf16, X, m![Out, In]>,
                &HbmTensor<f32, X, m![Out, 1 # 8]>,
            ),
            DmTensorView<'i, bf16, X, Chips, S, E>,
        ),
    )> for Biased<Rows>
{
    type Output = Result<In, Out, Rows, X>;

    /// `input` rows of at most a batch, each output `bf16(x . w + bias)` with the bias added to the
    /// FP32 sum, so the output rounds once; each offset is an eight-element FP32 vector.
    fn forward(
        &self,
        (device, ((weight, bias), input)): (
            &mut Device,
            (
                (
                    &HbmTensor<bf16, X, m![Out, In]>,
                    &HbmTensor<f32, X, m![Out, 1 # 8]>,
                ),
                DmTensorView<'i, bf16, X, Chips, S, E>,
            ),
        ),
    ) -> Self::Output {
        const { assert!(Batch::<In, Out, Rows>::SIZE == Rows::SIZE) };
        let rows = Stage::<In, Out, Rows, Rows>(PhantomData).forward((&mut *device, input));
        let mut result = DmTensor::new();
        for at in 0..<Passes<In, Out> as M>::SIZE {
            let slab_weight = Slab.forward((&mut *device, (weight, at)));
            let offsets: Offsets<In, Out, X> = {
                let landed: DmTensor<f32, X, Cluster<In, Out>, ProjectionSlices<In, Out>, m![Out % ({ Block<In, Out> }), 1 # 8]> = bias.view()
                    .tile::<SlabIndex<In, Out>, 1, m![Out % (Out / ({ Passes<In, Out> })) # Out, 1 # 8]>(at)
                    .to_dm(&mut device.tdma);
                device
                    .main
                    .begin(landed.view())
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
                    .to_vrf(&mut device.sub)
            };
            let lanes = rows.forward((&mut *device, (0, PhantomData::<Rows>)));
            {
                type B<Rows> = Rows;
                let weight = &slab_weight;
                let lanes = &lanes;
                let bias = &offsets;
                let result =
                    Contraction::<In, Out>::landing::<Rows, Rows, X>(result.view_mut(), at, 0);
                device
                    .main
                    .begin(weight.view())
                    .fetch::<
                        m![Out % ({ Block<In, Out> }), In % ({ Chunk<In, Out> }) / 16],
                        m![In % 16],
                    >()
                    .collect::<
                        m![Out % ({ Block<In, Out> }), In % ({ Chunk<In, Out> }) / 16],
                        m![In % 16],
                    >()
                    .contract_outer::<
                        m![Out % ({ Block<In, Out> }), In % ({ Chunk<In, Out> }) / 32, { Count<Rows, B<Rows>> }],
                        m![In % 32], _, _, _,
                    >(lanes)
                    .contract_packet::<m![1]>()
                    .contract_time::<m![Out % ({ Block<In, Out> }), { Count<Rows, B<Rows>> }]>()
                    .contract_lane::<
                        m![Out % ({ Block<In, Out> }), { Count<Rows, B<Rows>> }],
                        m![{ Layout<Rows>::Lanes } # 8],
                    >(LaneMode::Interleaved)
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_clip(ClipBinaryOpF32::Add, bias)
                    .vector_inter_slice_reduce::<
                        Reduced<In, Out>,
                        m![Out % ({ Block<In, Out> }), { Count<Rows, B<Rows>> }],
                    >(InterSliceReduceOpF32::Add)
                    .vector_final()
                    .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
                    .transpose::<
                        m![Out % ({ Block<In, Out> }) / 4, { Count<Rows, B<Rows>> }, { Layout<Rows>::Lanes }],
                        m![Out % 4 # 16],
                    >()
                    .commit_trim::<m![Out % 4]>()
                    .commit_view(result);
            }
        }
        result
    }
}

/// The result cells weight slab `slab` writes for batch `at` of `B` rows.
impl<In: AxisName, Out: AxisName> Contraction<In, Out> {
    fn landing<Rows: AxisName, B: AxisName, X: M>(
        output: DmTensorViewMut<
            '_,
            bf16,
            X,
            Cluster<In, Out>,
            Reduced<In, Out>,
            Element<In, Out, Rows>,
        >,
        slab: usize,
        at: usize,
    ) -> DmTensorViewMut<
        '_,
        bf16,
        X,
        Cluster<In, Out>,
        Reduced<In, Out>,
        m![
            { SlabSelected<In, Out> },
            Rows / B = 1 #{!} (Rows / B),
            Rows % B,
            Out % ({ Block<In, Out> }) / 4,
            Out % 4
        ],
    > {
        type Slab<In, Out, Rows> =
            m![{ SlabSelected<In, Out> }, Rows, Out % ({ Block<In, Out> }) / 4, Out % 4];
        output
            .tile::<SlabIndex<In, Out>, 1, Slab<In, Out, Rows>>(slab)
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
