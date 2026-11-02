//! A launch's live rows through NVFP4 weights under the summed contract: each slice owns
//! consecutive output rows over one part of the inputs, as the late contract's rows do, and the
//! codes meet the rows unrestored, a lane group of eight rows at a time.
//!
//! - A weight begun lands its tensor scale, block scales and codes, one transfer each, before any
//!   rows arrive, so a schedule can land it under earlier work; its projection widens the block
//!   scales to FP32 once, then runs every lane group.
//! - A lane group fills half of the tensor registers through the switch ring once and meets every
//!   pass there; a pass covers the most rows whose FP32 block scales fit the vector registers.
//! - The projected rows end where the contract places them: in the output's feature slices once
//!   the last pass ends, or [`Kept`] where the passes summed them.
//! - Rows another summed projection kept land in both clusters as they lie, an owner's run of
//!   features a slice, and its lane groups stage from there; rows whose runs cannot tile both
//!   clusters' slices that way move to the feature slices first.

use crate::ops::linear::kernel::Rounded;
use crate::ops::normalization::{self, FeatureSlices, Spread};
use crate::ops::pointwise::Kept;
use crate::specialize::{TRF, VRF};
use crate::tensor::form::BlockScaled;
use crate::tensor::form::nvfp4::{Features, Held, Scale, Scales, Summed, W4A16};
use crate::tensor::memory::Memory;
use crate::tensor::{axis::Lanes, live::Live, mapping::Rows as Layout, parallel::Chip, scalar};
use crate::{Begin, Op};
use core::{borrow::Borrow, marker::PhantomData};
use furiosa_opt_std::prelude::*;

/// Inputs of a row one slice holds: the share of the fewest power-of-two parts whose eight rows fill
/// at most half of a slice's tensor registers, so a lane group fills beside the one in use.
#[derive(Clone, Debug)]
pub struct Chunk<I>(PhantomData<I>);

impl<I: AxisName> AxisName for Chunk<I> {
    const NAME: Ident = Ident::new("Chunk");
    const SIZE: usize = {
        let mut parts = 1;
        while 8 * 2 * (I::SIZE / parts) > TRF / 2 {
            parts *= 2;
        }
        I::SIZE / parts
    };
}

/// Output rows one slice owns: its cluster's half of the outputs over the cluster's owners.
#[derive(Clone, Debug)]
pub struct Owned<I, O>(PhantomData<(I, O)>);

impl<I: AxisName, O: AxisName> AxisName for Owned<I, O> {
    const NAME: Ident = Ident::new("Owned");
    const SIZE: usize = O::SIZE / 2 / (256 / (I::SIZE / Chunk::<I>::SIZE));
}

/// Output rows a pass covers: the most, a multiple of four dividing the owned rows, whose FP32 block
/// scales over the slice's chunk take at most three quarters of its vector registers.
#[derive(Clone, Debug)]
struct Height<I, O>(PhantomData<(I, O)>);

impl<I: AxisName, O: AxisName> AxisName for Height<I, O> {
    const NAME: Ident = Ident::new("Height");
    const SIZE: usize = {
        let owned = Owned::<I, O>::SIZE;
        let mut rows = owned - owned % 4;
        while rows > 4
            && !(owned.is_multiple_of(rows) && rows * Chunk::<I>::SIZE / 16 * 4 <= VRF * 3 / 4)
        {
            rows -= 4;
        }
        rows
    };
}

type Width<I> = Symbol<Chunk<I>>;
type Owner<I, O> = Symbol<Owned<I, O>>;
type Tall<I, O> = Symbol<Height<I, O>>;
/// The first half of the outputs in cluster 0, the second in cluster 1.
type Cluster<O> = m![O / (O / 2)];
/// A slice owns consecutive output rows over one part of the inputs; the parts of a row's owners
/// are adjacent, innermost, for the inter-slice reducer.
type Owners<I, O> = m![O % (O / 2) / ({ Owner<I, O> }), I / ({ Width<I> })];
type Reduced<I, O> = m![O % (O / 2) / ({ Owner<I, O> }), 1 # (I / ({ Width<I> }))];
/// A slice's part of a row's inputs.
type Part<I> = m![I % ({ Width<I> })];
/// A slice's passes over its rows, and the rows of one pass.
type Tiles<I, O> = m![O % ({ Owner<I, O> }) / ({ Tall<I, O> })];
type Elements<I, O> = m![O % ({ Tall<I, O> })];
/// A lane group's rows, the eight lanes of one contraction.
type Lane<R> = m![{ Layout<R>::Lanes }];

/// A launch's rows as the passes of a projection of `I` inputs into `O` outputs sum them: each
/// cluster half of the outputs, each owner slice its consecutive outputs of every row.
type Summing<I, O, R> =
    DmTensor<bf16, Chip, Cluster<O>, Reduced<I, O>, m![R, O % ({ Owner<I, O> })]>;

/// The rows a projection of `H` inputs into `I` outputs summed, in both clusters: each half of the
/// outputs on half of the slices, an owner's run on a slice of its own, as the passes left them.
type Both<H, I, R> =
    DmTensor<bf16, Chip, normalization::Cluster, Runs<H, I>, m![R, I % ({ Owner<H, I> })]>;
type Runs<H, I> = m![I / (I / 2), I % (I / 2) / ({ Owner<H, I> })];

/// Whether [`Both`] holds the rows a projection of `H` inputs into `I` outputs summed: the runs of
/// both halves fit a cluster's slices and a part of `I` inputs is whole runs.
const fn fits_both<H: AxisName, I: AxisName>() -> bool {
    H::SIZE / Chunk::<H>::SIZE >= 2
        && I::SIZE / Chunk::<I>::SIZE >= 2
        && Chunk::<I>::SIZE.is_multiple_of(Owned::<H, I>::SIZE)
}

/// Inputs a switch packet takes from one owner's run of `H`-input rows: the most, at most a flit,
/// that divide the run, so a packet never spans two owners.
#[derive(Clone, Debug)]
struct Piece<H, I>(PhantomData<(H, I)>);

impl<H: AxisName, I: AxisName> AxisName for Piece<H, I> {
    const NAME: Ident = Ident::new("Piece");
    const SIZE: usize = {
        let run = Owned::<H, I>::SIZE;
        let mut piece = 16;
        while !run.is_multiple_of(piece) {
            piece -= 4;
        }
        piece
    };
}

/// A weight's tensor scale, E4M3 block scales and codes on the slices of its passes: the weight
/// begun.
pub struct Landed<I: AxisName, O: AxisName> {
    global: DmTensor<f32, Chip, Cluster<O>, Owners<I, O>, m![1 # 8]>,
    blocks: DmTensor<
        f8e4m3,
        Chip,
        Cluster<O>,
        Owners<I, O>,
        m![O % ({ Owner<I, O> }), { Part<I> } / 16],
    >,
    codes: DmTensor<f4e2m1, Chip, Cluster<O>, Owners<I, O>, m![O % ({ Owner<I, O> }), { Part<I> }]>,
}

impl<I: AxisName, O: AxisName> Landed<I, O> {
    /// The weight's tensor scale, block scales and codes, in that order, one transfer each: a
    /// transfer into part of an owner would hold every later command until the part before it
    /// lands.
    fn land<Y: Memory, P, K, V>(
        device: &mut Device,
        weight: &BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P>, Chip, K, V>,
    ) -> Self
    where
        K: Borrow<HbmTensor<f4e2m1, Chip, m![O, I]>>,
        V: Borrow<HbmTensor<f8e4m3, Chip, Scales<I, O>>>,
        Held<Y>: Borrow<Scale>,
    {
        const {
            assert!(
                I::SIZE.is_multiple_of(Chunk::<I>::SIZE)
                    && Chunk::<I>::SIZE.is_multiple_of(128)
                    && 256usize.is_multiple_of(I::SIZE / Chunk::<I>::SIZE)
                    && O::SIZE.is_multiple_of(2 * 256 / (I::SIZE / Chunk::<I>::SIZE))
                    && Owned::<I, O>::SIZE.is_multiple_of(Height::<I, O>::SIZE)
                    && Height::<I, O>::SIZE.is_multiple_of(4)
            )
        };
        let global: &Scale = weight.contract.global.borrow();
        Self {
            global: global.to_dm(&mut device.tdma),
            blocks: weight.scales().to_dm(&mut device.tdma),
            codes: weight.codes().to_dm(&mut device.tdma),
        }
    }

    /// Every live row's outputs `bf16(sum_b scale_b (sum_{k in b} code_k x_k) / global)`, a lane
    /// group at a time, each in the slices that sum it.
    fn project<R: AxisName, X: Stage<I, O, R>>(
        &self,
        device: &mut Device,
        input: &Live<X, R>,
    ) -> Summing<I, O, R> {
        let global = scalar::register(device, &self.global);
        // The block scales widen to FP32 once, so each pass of every group loads its rows' as is.
        let mut scales: DmTensor<
            f32,
            Chip,
            Cluster<O>,
            Owners<I, O>,
            m![O % ({ Owner<I, O> }), { Part<I> } / 16],
        > = DmTensor::new();
        device
            .sub
            .begin(self.blocks.view())
            .fetch::<m![O % ({ Owner<I, O> })], m![{ Part<I> } / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![O % ({ Owner<I, O> }), { Part<I> } / 128], m![I / 16 % 8]>()
            .commit_trim::<m![I / 16 % 8]>()
            .commit_view(scales.view_mut());
        let mut projected = Summing::<I, O, R>::new();
        let (first, count) = input.groups::<Lanes<R>>();
        for at in 0..count {
            let group = first + at;
            let lanes = input.value().stage(device, group);
            for tile in 0..<Tiles<I, O> as M>::SIZE {
                let scale: VrfTensor<
                    f32,
                    Chip,
                    Cluster<O>,
                    Owners<I, O>,
                    m![{ Elements<I, O> }, { Part<I> } / 16],
                > = device
                    .sub
                    .begin(scales.view().tile::<Tiles<I, O>, 1, m![
                        { Elements<I, O> } # ({ Owner<I, O> }), { Part<I> } / 16
                    ]>(tile))
                    .fetch::<m![{ Elements<I, O> }, { Part<I> } / 128], m![I / 16 % 8]>()
                    .collect::<m![{ Elements<I, O> }, { Part<I> } / 128], m![I / 16 % 8]>()
                    .to_vrf();
                // Each lane's block sums meet their row's block scales, the same for every lane.
                let sums = device
                    .main
                    .begin(self.codes.view().tile::<Tiles<I, O>, 1, m![
                        { Elements<I, O> } # ({ Owner<I, O> }), { Part<I> }
                    ]>(tile))
                    .fetch::<m![{ Elements<I, O> }, { Part<I> } / 64], m![I % 64]>()
                    .fetch_table_lookup::<bf16>()
                    .collect::<m![{ Elements<I, O> }, { Part<I> } / 16], m![I % 16]>()
                    .contract_outer::<m![{ Elements<I, O> }, { Part<I> } / 16, 1], m![I % 16], _, _, _>(
                        &lanes,
                    )
                    .contract_packet::<m![1]>()
                    .contract_time::<m![{ Elements<I, O> }, { Part<I> } / 16, 1]>()
                    .contract_lane::<m![{ Elements<I, O> }, { Part<I> } / 16, 1], m![{ Lane<R> } # 8]>(
                        LaneMode::Interleaved,
                    )
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<
                        m![{ Elements<I, O> }, { Part<I> } / 16, 1, { Lane<R> } # 8 / 4],
                        m![{ Lane<R> } # 8 % 4],
                    >()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
                    .vector_intra_slice_reduce::<
                        I,
                        m![{ Elements<I, O> }, 1, { Lane<R> } # 8 / 4],
                        m![{ Lane<R> } # 8 % 4],
                    >(IntraSliceReduceOpF32::Add)
                    .vector_fp_div(&global)
                    .vector_widen_concat::<m![{ Elements<I, O> }, 1], m![{ Lane<R> } # 8]>()
                    .vector_inter_slice_reduce::<Reduced<I, O>, m![{ Elements<I, O> }, 1]>(
                        InterSliceReduceOpF32::Add,
                    )
                    .vector_final();
                Rounded::rounded::<
                    bf16,
                    _,
                    _,
                    _,
                    _,
                    Lane<R>,
                    m![{ Elements<I, O> } / 4, 1, { Lane<R> }],
                    m![O % 4],
                    _,
                >(sums)
                .commit_view(
                    projected
                        .view_mut()
                        .tile::<Layout<R>::Groups, 1, Layout<R>::GroupMut<m![O % ({ Owner<I, O> })]>>(
                            group,
                        )
                        .tile::<Tiles<I, O>, 1, m![
                            { Layout<R>::SelectedMut },
                            { Lane<R> },
                            { Elements<I, O> } #{!} ({ Owner<I, O> })
                        ]>(tile),
                );
            }
        }
        projected
    }
}

/// Where a projection's rows end once its passes summed them: the rows `place` gives of `R` rows
/// projected from `I` inputs into `O` outputs.
pub trait Place<I: AxisName, O: AxisName, R: AxisName> {
    type Rows;
    fn place(device: &mut Device, rows: Summing<I, O, R>) -> Self::Rows;
}

impl<I: AxisName, O: AxisName, R: AxisName> Place<I, O, R> for Features {
    type Rows = Spread<O, R>;

    fn place(device: &mut Device, rows: Summing<I, O, R>) -> Self::Rows {
        spread(device, &rows)
    }
}

impl<I: AxisName, O: AxisName, R: AxisName> Place<I, O, R> for Kept {
    type Rows = Kept<Summing<I, O, R>>;

    fn place(_: &mut Device, rows: Summing<I, O, R>) -> Self::Rows {
        Kept(rows)
    }
}

/// The live rows of a launch in HBM through a weight under the summed contract, into HBM.
impl<I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory>
    Op<(Live<(), R>, &HbmTensor<bf16, Chip, m![R, I]>)>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, m![O, I]>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Scales<I, O>>>,
    Held<Y>: Borrow<Scale>,
{
    type Output = HbmTensor<bf16, Chip, m![R, O]>;

    fn forward(
        &self,
        device: &mut Device,
        (live, input): (Live<(), R>, &HbmTensor<bf16, Chip, m![R, I]>),
    ) -> Self::Output {
        let rows: Live<Spread<I, R>, R> = live.with(input.to_dm(&mut device.tdma));
        self.forward(&mut *device, &rows)
            .value()
            .to_hbm(&mut device.tdma)
    }
}

/// A weight under the summed contract begun: its scales and codes land before any rows arrive.
impl<I: AxisName, O: AxisName, K, V, Y: Memory, P> Op<Begin<()>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, m![O, I]>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Scales<I, O>>>,
    Held<Y>: Borrow<Scale>,
{
    type Output = Landed<I, O>;

    fn forward(&self, device: &mut Device, _: Begin<()>) -> Self::Output {
        Landed::land(device, self)
    }
}

/// The live rows of a launch in their feature slices through a weight under the summed contract,
/// into where its placement `P` puts them; the projected rows keep the live rows.
impl<'x, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P: Place<I, O, R>>
    Op<&'x Live<Spread<I, R>, R>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, m![O, I]>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Scales<I, O>>>,
    Held<Y>: Borrow<Scale>,
{
    type Output = Live<P::Rows, R>;

    fn forward(&self, device: &mut Device, input: &'x Live<Spread<I, R>, R>) -> Self::Output {
        let landed = self.forward(&mut *device, Begin(()));
        self.forward(device, (landed, input))
    }
}

/// The same rows through the weight as it landed.
impl<'x, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P: Place<I, O, R>>
    Op<(Landed<I, O>, &'x Live<Spread<I, R>, R>)>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P>, Chip, K, V>
{
    type Output = Live<P::Rows, R>;

    fn forward(
        &self,
        device: &mut Device,
        (landed, input): (Landed<I, O>, &'x Live<Spread<I, R>, R>),
    ) -> Self::Output {
        let projected = landed.project(device, input);
        input.with(P::place(device, projected))
    }
}

/// The live rows a summed projection of `H` inputs kept where its passes summed them, through a
/// weight under the summed contract, into where its placement `P` puts them.
impl<'x, H: AxisName, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P>
    Op<&'x Live<Kept<Summing<H, I, R>>, R>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, m![O, I]>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Scales<I, O>>>,
    Held<Y>: Borrow<Scale>,
    P: Place<I, O, R>,
{
    type Output = Live<P::Rows, R>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<Kept<Summing<H, I, R>>, R>,
    ) -> Self::Output {
        let landed = self.forward(&mut *device, Begin(()));
        self.forward(device, (landed, input))
    }
}

/// The same rows through the weight as it landed: one transfer lands them in both clusters as they
/// lie, or in the feature slices where they cannot lie so, then they project.
impl<'x, H: AxisName, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P>
    Op<(Landed<I, O>, &'x Live<Kept<Summing<H, I, R>>, R>)>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P>, Chip, K, V>
where
    P: Place<I, O, R>,
{
    type Output = Live<P::Rows, R>;

    fn forward(
        &self,
        device: &mut Device,
        (landed, input): (Landed<I, O>, &'x Live<Kept<Summing<H, I, R>>, R>),
    ) -> Self::Output {
        let rows = &input.value().0;
        let projected = if const { fits_both::<H, I>() } {
            let mut both = Both::<H, I, R>::new();
            rows.view().to_dm_view(&mut device.tdma, both.view_mut());
            landed.project(device, &input.with(both))
        } else {
            let spread = input.with(spread(device, rows));
            landed.project(device, &spread)
        };
        input.with(P::place(device, projected))
    }
}

/// Every row a projection's passes summed, in the output's feature slices: one transfer, as a
/// transfer costs about the same for one row as for the envelope.
fn spread<I: AxisName, O: AxisName, R: AxisName>(
    device: &mut Device,
    rows: &Summing<I, O, R>,
) -> Spread<O, R> {
    let mut spread = Spread::<O, R>::new();
    rows.view().to_dm_view(&mut device.tdma, spread.view_mut());
    spread
}

/// Rows a projection of `I` inputs into `O` outputs reads a lane group of as the tensor-register
/// lanes of every slice, each slice the group's inputs in its part.
trait Stage<I: AxisName, O: AxisName, R: AxisName> {
    fn stage(
        &self,
        device: &mut Device,
        group: usize,
    ) -> TrfTensor<bf16, Chip, Cluster<O>, Owners<I, O>, Lane<R>, m![1, { Part<I> }]>;
}

/// Rows in their feature slices: the switch ring takes each feature slice's inputs to the slices
/// that hold them.
impl<I: AxisName, O: AxisName, R: AxisName> Stage<I, O, R> for Spread<I, R> {
    fn stage(
        &self,
        device: &mut Device,
        group: usize,
    ) -> TrfTensor<bf16, Chip, Cluster<O>, Owners<I, O>, Lane<R>, m![1, { Part<I> }]> {
        type Feature<I> = Symbol<normalization::Chunk<I>>;
        type Group<I, R> = Layout<R>::Elements<m![I % ({ Feature<I> })]>;
        let input = Layout::<R>::group(self.view(), group);
        if const { I::SIZE <= 16 * 256 } {
            // SAFETY: both physical clusters hold every row, sixteen inputs a slice; only the
            // logical cluster axis changes.
            let input =
                unsafe { input.reshape::<Chip, Cluster<O>, m![I / 16 # 256], Group<I, R>>() };
            device
                .sub
                .begin(input)
                .fetch::<Lane<R>, m![I % 16]>()
                .switch::<Owners<I, O>, m![{ Lane<R> }, { Part<I> } / 16]>(
                    SwitchConfig::CustomBroadcast { ring_size: 256 },
                )
                .collect::<m![{ Lane<R> }, { Part<I> } / 16], m![I % 16]>()
                .to_trf()
        } else {
            // A slice holds several switch packets, delivered with other slices' inputs
            // innermost, so the parts are written in DM before the lanes fill in input order.
            // SAFETY: as above, only the logical cluster axis changes.
            let input =
                unsafe { input.reshape::<Chip, Cluster<O>, FeatureSlices<I>, Group<I, R>>() };
            let mut staged: DmTensor<
                bf16,
                Chip,
                Cluster<O>,
                Owners<I, O>,
                m![{ Lane<R> }, { Part<I> }],
            > = DmTensor::new();
            device
                .sub
                .begin(input)
                .fetch::<m![{ Lane<R> }, I % ({ Feature<I> }) / 16], m![I % 16]>()
                .switch::<Owners<I, O>, m![
                    { Lane<R> },
                    I % ({ Feature<I> }) / 16,
                    I % ({ Width<I> }) / ({ Feature<I> })
                ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
                .collect::<m![
                    { Lane<R> },
                    I % ({ Feature<I> }) / 16,
                    I % ({ Width<I> }) / ({ Feature<I> })
                ], m![I % 16]>()
                .commit_trim::<m![I % 16]>()
                .commit_view(staged.view_mut());
            device
                .sub
                .begin(staged.view())
                .fetch::<m![{ Lane<R> }, { Part<I> } / 16], m![I % 16]>()
                .collect::<m![{ Lane<R> }, { Part<I> } / 16], m![I % 16]>()
                .to_trf()
        }
    }
}

/// Rows a summed projection of `H` inputs kept, in both clusters: the switch ring takes each
/// owner's run, a piece at a time, to the slices whose part holds it, and the parts are written in
/// DM before the lanes fill in input order.
impl<H: AxisName, I: AxisName, O: AxisName, R: AxisName> Stage<I, O, R> for Both<H, I, R> {
    fn stage(
        &self,
        device: &mut Device,
        group: usize,
    ) -> TrfTensor<bf16, Chip, Cluster<O>, Owners<I, O>, Lane<R>, m![1, { Part<I> }]> {
        type Run<H, I> = Symbol<Piece<H, I>>;
        type Group<H, I, R> = Layout<R>::Elements<m![I % ({ Owner<H, I> })]>;
        const { assert!(fits_both::<H, I>()) };
        let input = Layout::<R>::group(self.view(), group);
        // SAFETY: both physical clusters hold every row; only the logical cluster axis changes.
        let input = unsafe { input.reshape::<Chip, Cluster<O>, Runs<H, I>, Group<H, I, R>>() };
        let mut staged: DmTensor<
            bf16,
            Chip,
            Cluster<O>,
            Owners<I, O>,
            m![{ Lane<R> }, { Part<I> }],
        > = DmTensor::new();
        device
            .sub
            .begin(input)
            .fetch::<m![{ Lane<R> }, I % ({ Owner<H, I> }) / ({ Run<H, I> })], m![
                I % ({ Run<H, I> })
            ]>()
            .switch::<Owners<I, O>, m![
                { Lane<R> },
                I % ({ Owner<H, I> }) / ({ Run<H, I> }),
                I % ({ Width<I> }) / ({ Owner<H, I> })
            ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
            .collect::<m![
                { Lane<R> },
                I % ({ Owner<H, I> }) / ({ Run<H, I> }),
                I % ({ Width<I> }) / ({ Owner<H, I> })
            ], m![I % ({ Run<H, I> }) # 16]>()
            .commit_trim::<m![I % ({ Run<H, I> })]>()
            .commit_view(staged.view_mut());
        device
            .sub
            .begin(staged.view())
            .fetch::<m![{ Lane<R> }, { Part<I> } / 16], m![I % 16]>()
            .collect::<m![{ Lane<R> }, { Part<I> } / 16], m![I % 16]>()
            .to_trf()
    }
}
