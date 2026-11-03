//! A launch's live rows through NVFP4 weights under the summed contract: each slice owns
//! consecutive output rows over one part of the inputs, as the late contract's rows do, and the
//! codes meet the rows unrestored, a lane group of eight rows at a time.
//!
//! - A weight begun lands its tensor scale, block scales and codes, one transfer each, before any
//!   rows arrive, so a schedule can land it under earlier work; its projection widens the block
//!   scales to FP32 once, then runs every lane group.
//! - A lane group fills half of the tensor registers through the switch ring once and meets every
//!   pass there; a pass covers the most rows whose FP32 block scales fit the vector registers. The
//!   next group fills the other half under the passes when its switch takes one part a pass.
//! - A pass's vector engine takes each block's sums four lanes a step, so the last group, when its
//!   live rows lie in its first four lanes, switches, fills and folds those four alone: one step a
//!   block.
//! - The projected rows end where the contract places them: in the output's feature slices once
//!   the last pass ends, or [`Kept`] where the passes summed them.
//! - Rows another summed projection kept land in both clusters as they lie, an owner's run of
//!   features a slice, and its lane groups stage from there; rows whose runs cannot tile both
//!   clusters' slices that way move to the feature slices first. A projection begun with such
//!   rows issues their landing alone, so a schedule can issue later transfers behind it.

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
/// The first four lanes of a lane group.
type Front<R> = m![{ Lane<R> } % 4];

/// A launch's rows as the passes of a projection of `I` inputs into `O` outputs sum them: each
/// cluster half of the outputs, each owner slice its consecutive outputs of every row.
type Summing<I, O, R> =
    DmTensor<bf16, Chip, Cluster<O>, Reduced<I, O>, m![R, O % ({ Owner<I, O> })]>;

/// The rows of a lane group's lanes `L` in the tensor registers of every slice of a projection of
/// `I` inputs into `O` outputs, each slice the group's inputs in its part.
type Registers<I, O, L> = TrfTensor<bf16, Chip, Cluster<O>, Owners<I, O>, L, m![1, { Part<I> }]>;

/// The tensor scale in the vector registers of every slice of a projection.
type Global<I, O> = VrfTensor<f32, Chip, Cluster<O>, Owners<I, O>, m![1 # 8]>;

/// The FP32 block scales of every row a slice owns, widened once from E4M3.
type Widened<I, O> =
    DmTensor<f32, Chip, Cluster<O>, Owners<I, O>, m![O % ({ Owner<I, O> }), { Part<I> } / 16]>;

/// The FP32 block scales of one pass's rows, in the vector registers.
type Scaling<I, O> =
    VrfTensor<f32, Chip, Cluster<O>, Owners<I, O>, m![{ Elements<I, O> }, { Part<I> } / 16]>;

/// A lane group's rows in each slice's part, in input order, as the switch leaves them in DM.
type Staging<I, O, R> =
    DmTensor<bf16, Chip, Cluster<O>, Owners<I, O>, m![{ Lane<R> }, { Part<I> }]>;

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
        let mut scales = Widened::<I, O>::new();
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
        let rows = input.value();
        // The last group folds its first four lanes alone when its live rows end there: one group
        // or none, counted by arithmetic, as a device cannot cast a comparison to a count.
        let front = if const { Lanes::<R>::SIZE == 8 } {
            count.min(1) * ((11 - (input.start() + input.rows() + 7) % 8) / 8)
        } else {
            0
        };
        let whole = count - front;
        let last = first + count.max(1) - 1;
        let operands = (&scales, &global);
        if const { X::PARTS == 0 || X::PARTS == <Tiles<I, O> as M>::SIZE } {
            // Each pass of a group before the last switches one part of the next group after its
            // own loads, beside its contraction; the next group's lanes fill the other half of the
            // TRF under the last pass. The last group's passes run alone.
            let mut staged = X::staged();
            for _ in 0..whole.min(1) {
                rows.switch(device, first, &mut staged);
                let mut lanes = rows.fill(device, first, &staged);
                for next in 1..whole {
                    for tile in 0..<Tiles<I, O> as M>::SIZE {
                        self.pass::<R, Split>(
                            device,
                            (&lanes, operands),
                            (first + next - 1, tile),
                            &mut projected,
                        );
                        if const { X::PARTS > 0 } {
                            rows.part(device, (first + next, tile), &mut staged);
                        }
                    }
                    lanes = rows.fill(device, first + next, &staged);
                }
                for tile in 0..<Tiles<I, O> as M>::SIZE {
                    self.pass::<R, Split>(
                        device,
                        (&lanes, operands),
                        (first + whole - 1, tile),
                        &mut projected,
                    );
                    if const { X::PARTS > 0 } {
                        for _ in 0..front {
                            rows.part(device, (last, tile), &mut staged);
                        }
                    }
                }
            }
            for _ in 0..front {
                // Behind whole groups its parts came under their passes; alone, it switches its
                // four lanes.
                if const { X::PARTS > 0 } {
                    for _ in 0..1 - whole.min(1) {
                        rows.switch_front(device, last, &mut staged);
                    }
                }
                let lanes = rows.front(device, last, &staged);
                for tile in 0..<Tiles<I, O> as M>::SIZE {
                    self.pass::<R, Trim>(device, (&lanes, operands), (last, tile), &mut projected);
                }
            }
        } else {
            for at in 0..whole {
                let group = first + at;
                let mut staged = X::staged();
                rows.switch(device, group, &mut staged);
                let lanes = rows.fill(device, group, &staged);
                for tile in 0..<Tiles<I, O> as M>::SIZE {
                    self.pass::<R, Split>(
                        device,
                        (&lanes, operands),
                        (group, tile),
                        &mut projected,
                    );
                }
            }
            for _ in 0..front {
                let mut staged = X::staged();
                rows.switch_front(device, last, &mut staged);
                let lanes = rows.front(device, last, &staged);
                for tile in 0..<Tiles<I, O> as M>::SIZE {
                    self.pass::<R, Trim>(device, (&lanes, operands), (last, tile), &mut projected);
                }
            }
        }
        projected
    }

    /// Pass `tile` of lane group `group` over the lanes `lanes` holds: their block sums meet their
    /// rows' block scales, the same for every lane, as `F` folds the lanes.
    fn pass<R: AxisName, F: Fold<I, O, R>>(
        &self,
        device: &mut Device,
        (lanes, (scales, global)): (&Registers<I, O, F::Filled>, (&Widened<I, O>, &Global<I, O>)),
        (group, tile): (usize, usize),
        projected: &mut Summing<I, O, R>,
    ) {
        let scale: Scaling<I, O> = device
            .sub
            .begin(scales.view().tile::<Tiles<I, O>, 1, m![
                { Elements<I, O> } # ({ Owner<I, O> }), { Part<I> } / 16
            ]>(tile))
            .fetch::<m![{ Elements<I, O> }, { Part<I> } / 128], m![I / 16 % 8]>()
            .collect::<m![{ Elements<I, O> }, { Part<I> } / 128], m![I / 16 % 8]>()
            .to_vrf();
        let sums = device
            .main
            .begin(self.codes.view().tile::<Tiles<I, O>, 1, m![
                { Elements<I, O> } # ({ Owner<I, O> }), { Part<I> }
            ]>(tile))
            .fetch::<m![{ Elements<I, O> }, { Part<I> } / 64], m![I % 64]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![{ Elements<I, O> }, { Part<I> } / 16], m![I % 16]>()
            .contract_outer::<m![{ Elements<I, O> }, { Part<I> } / 16, 1], m![I % 16], _, _, _>(
                lanes,
            )
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Elements<I, O> }, { Part<I> } / 16, 1]>()
            .contract_lane::<m![{ Elements<I, O> }, { Part<I> } / 16, 1], m![{ F::Filled } # 8]>(
                LaneMode::Interleaved,
            )
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero);
        let sums = F::fold(sums, (&scale, global))
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
            F::Filled,
            m![{ Elements<I, O> } / 4, 1, { F::Filled }],
            m![O % 4],
            _,
        >(sums)
        .commit_view(F::sums(projected, (group, tile)));
    }
}

/// How a pass folds the lanes it holds into the vector engine, four lanes a step, and where their
/// sums land.
trait Fold<I: AxisName, O: AxisName, R: AxisName> {
    /// The lanes of a group the tensor registers hold.
    type Filled: M;
    /// The rows of a pass's outputs these lanes write.
    type Rows: M;

    /// The lanes' block sums met with their block scales, added over the blocks and divided by
    /// the tensor scale.
    fn fold<'l>(
        sums: Tagged<'l, I, O, Self::Filled>,
        operands: (&Scaling<I, O>, &Global<I, O>),
    ) -> Folded<'l, I, O, Self::Filled>;

    /// Where pass `tile` of group `group` writes these lanes' sums.
    fn sums(
        projected: &mut Summing<I, O, R>,
        at: (usize, usize),
    ) -> DmTensorViewMut<'_, bf16, Chip, Cluster<O>, Reduced<I, O>, Self::Rows>;
}

/// A pass's block sums leaving the lane folder, lanes `L` a packet of eight.
type Tagged<'l, I, O, L> = VectorBranchTensor<
    'l,
    { Tu::Main },
    f32,
    Chip,
    Cluster<O>,
    Owners<I, O>,
    m![{ Elements<I, O> }, { Part<I> } / 16, 1],
    m![{ L } # 8],
    Fresh,
    { stage::VeOrder::IntraFirst },
>;

/// A pass's sums of every output once its blocks added, lanes `L` a packet of eight.
type Folded<'l, I, O, L> = VectorWidenTensor<
    'l,
    { Tu::Main },
    f32,
    Chip,
    Cluster<O>,
    Owners<I, O>,
    m![{ Elements<I, O> }, 1],
    m![{ L } # 8],
    Fresh,
    { stage::VeOrder::IntraFirst },
>;

/// Every lane of a group: two vector steps a block.
struct Split;

impl<I: AxisName, O: AxisName, R: AxisName> Fold<I, O, R> for Split {
    type Filled = Lane<R>;
    type Rows = m![
        { Layout<R>::SelectedMut },
        { Lane<R> },
        { Elements<I, O> } #{!} ({ Owner<I, O> })
    ];

    fn fold<'l>(
        sums: Tagged<'l, I, O, Lane<R>>,
        (scale, global): (&Scaling<I, O>, &Global<I, O>),
    ) -> Folded<'l, I, O, Lane<R>> {
        sums.vector_narrow_split::<
            m![{ Elements<I, O> }, { Part<I> } / 16, 1, { Lane<R> } # 8 / 4],
            m![{ Lane<R> } # 8 % 4],
        >()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
        .vector_intra_slice_reduce::<
            I,
            m![{ Elements<I, O> }, 1, { Lane<R> } # 8 / 4],
            m![{ Lane<R> } # 8 % 4],
        >(IntraSliceReduceOpF32::Add)
        .vector_fp_div(global)
        .vector_widen_concat::<m![{ Elements<I, O> }, 1], m![{ Lane<R> } # 8]>()
    }

    fn sums(
        projected: &mut Summing<I, O, R>,
        (group, tile): (usize, usize),
    ) -> DmTensorViewMut<'_, bf16, Chip, Cluster<O>, Reduced<I, O>, Self::Rows> {
        projected
            .view_mut()
            .tile::<Layout<R>::Groups, 1, Layout<R>::GroupMut<m![O % ({ Owner<I, O> })]>>(group)
            .tile::<Tiles<I, O>, 1, Self::Rows>(tile)
    }
}

/// The first four lanes of a group, which hold every live row of it: one vector step a block.
struct Trim;

impl<I: AxisName, O: AxisName, R: AxisName> Fold<I, O, R> for Trim {
    type Filled = Front<R>;
    type Rows = m![
        { Layout<R>::SelectedMut },
        { Lane<R> } % 4 #{!} ({ Symbol<Lanes<R>> }),
        { Elements<I, O> } #{!} ({ Owner<I, O> })
    ];

    fn fold<'l>(
        sums: Tagged<'l, I, O, Front<R>>,
        (scale, global): (&Scaling<I, O>, &Global<I, O>),
    ) -> Folded<'l, I, O, Front<R>> {
        sums.vector_narrow_trim::<Front<R>>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
            .vector_intra_slice_reduce::<I, m![{ Elements<I, O> }, 1], Front<R>>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_fp_div(global)
            .vector_widen_pad::<m![{ Front<R> } # 8]>()
    }

    fn sums(
        projected: &mut Summing<I, O, R>,
        (group, tile): (usize, usize),
    ) -> DmTensorViewMut<'_, bf16, Chip, Cluster<O>, Reduced<I, O>, Self::Rows> {
        projected
            .view_mut()
            .tile::<Layout<R>::Groups, 1, Layout<R>::GroupMut<m![O % ({ Owner<I, O> })]>>(group)
            .tile::<m![{ Lane<R> } / 4], 1, m![
                { Layout<R>::SelectedMut },
                { Lane<R> } % 4 #{!} ({ Symbol<Lanes<R>> }),
                O % ({ Owner<I, O> })
            ]>(0)
            .tile::<Tiles<I, O>, 1, Self::Rows>(tile)
    }
}

/// Where a projection's rows end once its passes summed them: the rows `place` gives of `R` rows
/// projected from `I` inputs into `O` outputs.
pub trait Place<I: AxisName, O: AxisName, R: AxisName> {
    type Rows;
    fn place(device: &mut Device, rows: Live<Summing<I, O, R>, R>) -> Live<Self::Rows, R>;
}

impl<I: AxisName, O: AxisName, R: AxisName> Place<I, O, R> for Features {
    type Rows = Spread<O, R>;

    fn place(device: &mut Device, rows: Live<Summing<I, O, R>, R>) -> Live<Self::Rows, R> {
        let mut spread = Spread::<O, R>::new();
        transfer(device, rows.with(rows.value()), &mut spread);
        rows.with(spread)
    }
}

impl<I: AxisName, O: AxisName, R: AxisName> Place<I, O, R> for Kept {
    type Rows = Kept<Summing<I, O, R>>;

    fn place(_: &mut Device, rows: Live<Summing<I, O, R>, R>) -> Live<Self::Rows, R> {
        let live = rows.with(());
        live.with(Kept(rows.into_value()))
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
        P::place(device, input.with(projected))
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

/// The same rows through the weight as it landed: they land in both clusters as they lie, or in
/// the feature slices where they cannot lie so, then they project.
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
        if const { fits_both::<H, I>() } {
            let begun = self.forward(&mut *device, Begin((landed, input)));
            self.forward(device, begun)
        } else {
            let mut spread = Spread::<I, R>::new();
            transfer(device, input.with(&input.value().0), &mut spread);
            self.forward(device, (landed, &input.with(spread)))
        }
    }
}

/// The same rows begun with the weight as it landed: one transfer lands them in both clusters as
/// they lie, so a schedule can issue it ahead of other transfers; given back, they project.
impl<'x, H: AxisName, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P>
    Op<Begin<(Landed<I, O>, &'x Live<Kept<Summing<H, I, R>>, R>)>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P>, Chip, K, V>
{
    type Output = (Landed<I, O>, Live<Both<H, I, R>, R>);

    fn forward(
        &self,
        device: &mut Device,
        Begin((landed, input)): Begin<(Landed<I, O>, &'x Live<Kept<Summing<H, I, R>>, R>)>,
    ) -> Self::Output {
        const { assert!(fits_both::<H, I>()) };
        let mut both = Both::<H, I, R>::new();
        transfer(device, input.with(&input.value().0), &mut both);
        (landed, input.with(both))
    }
}

/// Rows landed in both clusters through the weight as it landed.
impl<H: AxisName, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P>
    Op<(Landed<I, O>, Live<Both<H, I, R>, R>)>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P>, Chip, K, V>
where
    P: Place<I, O, R>,
{
    type Output = Live<P::Rows, R>;

    fn forward(
        &self,
        device: &mut Device,
        (landed, rows): (Landed<I, O>, Live<Both<H, I, R>, R>),
    ) -> Self::Output {
        let projected = landed.project(device, &rows);
        P::place(device, rows.with(projected))
    }
}

/// The live rows of `rows` into `into`'s mapping in one transfer: the lane group holding them, or
/// the envelope when they span groups, as a transfer's setup costs about as much as all its rows.
fn transfer<R: AxisName, C: M, S: M, X: M, D: M, T: M, Y: M>(
    device: &mut Device,
    rows: Live<&DmTensor<bf16, Chip, C, S, m![R, { X }]>, R>,
    into: &mut DmTensor<bf16, Chip, D, T, m![R, { Y }]>,
) {
    let (first, count) = rows.groups::<Lanes<R>>();
    if count > 1 {
        rows.value()
            .view()
            .to_dm_view(&mut device.tdma, into.view_mut());
    } else {
        Layout::<R>::group(rows.value().view(), first).to_dm_view(
            &mut device.tdma,
            (into.view_mut()).tile::<Layout<R>::Groups, 1, Layout<R>::GroupMut<Y>>(first),
        );
    }
}

/// Rows a projection of `I` inputs into `O` outputs reads a lane group of as the tensor-register
/// lanes of every slice, each slice the group's inputs in its part: the switch ring writes a group
/// into its staging DM, whole or in `PARTS` parts, and the group fills the TRF from there.
trait Stage<I: AxisName, O: AxisName, R: AxisName> {
    /// The parts the switch can write a group in, none when it fills the TRF itself.
    const PARTS: usize;

    fn staged() -> Staging<I, O, R>;
    fn switch(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R>);
    /// The first four lanes of `group` alone, as `switch` switches every lane.
    fn switch_front(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R>);
    fn part(&self, device: &mut Device, at: (usize, usize), staged: &mut Staging<I, O, R>);
    fn fill(
        &self,
        device: &mut Device,
        group: usize,
        staged: &Staging<I, O, R>,
    ) -> Registers<I, O, Lane<R>>;
    /// The first four lanes of `group` alone, as `fill` fills every lane.
    fn front(
        &self,
        device: &mut Device,
        group: usize,
        staged: &Staging<I, O, R>,
    ) -> Registers<I, O, Front<R>>;
}

/// Rows in their feature slices: the switch ring takes each feature slice's inputs to the slices
/// that hold them, straight into the TRF where a slice holds one packet of a row.
impl<I: AxisName, O: AxisName, R: AxisName> Stage<I, O, R> for Spread<I, R> {
    const PARTS: usize = if I::SIZE <= 16 * 256 { 0 } else { 1 };

    fn staged() -> Staging<I, O, R> {
        DmTensor::new()
    }

    fn switch(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R>) {
        type Feature<I> = Symbol<normalization::Chunk<I>>;
        type Group<I, R> = Layout<R>::Elements<m![I % ({ Feature<I> })]>;
        if const { I::SIZE > 16 * 256 } {
            // A slice holds several switch packets, delivered with other slices' inputs
            // innermost, so the parts are written in DM before the lanes fill in input order.
            // SAFETY: both physical clusters hold every row; only the logical cluster axis changes.
            let input = unsafe {
                Layout::<R>::group(self.view(), group)
                    .reshape::<Chip, Cluster<O>, FeatureSlices<I>, Group<I, R>>()
            };
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
        }
    }

    fn switch_front(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R>) {
        type Feature<I> = Symbol<normalization::Chunk<I>>;
        type Group<I, R> = Layout<R>::Elements<m![I % ({ Feature<I> })]>;
        if const { I::SIZE > 16 * 256 } {
            // SAFETY: as in `switch`.
            let input = unsafe {
                Layout::<R>::group(self.view(), group)
                    .reshape::<Chip, Cluster<O>, FeatureSlices<I>, Group<I, R>>()
            };
            device
                .sub
                .begin(input.tile::<m![{ Lane<R> } / 4], 1, m![
                    { Layout<R>::Selected },
                    { Lane<R> } % 4 # ({ Symbol<Lanes<R>> }),
                    I % ({ Feature<I> })
                ]>(0))
                .fetch::<m![{ Front<R> }, I % ({ Feature<I> }) / 16], m![I % 16]>()
                .switch::<Owners<I, O>, m![
                    { Front<R> },
                    I % ({ Feature<I> }) / 16,
                    I % ({ Width<I> }) / ({ Feature<I> })
                ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
                .collect::<m![
                    { Front<R> },
                    I % ({ Feature<I> }) / 16,
                    I % ({ Width<I> }) / ({ Feature<I> })
                ], m![I % 16]>()
                .commit_trim::<m![I % 16]>()
                .commit_view(staged_front(staged));
        }
    }

    fn part(&self, device: &mut Device, (group, _): (usize, usize), staged: &mut Staging<I, O, R>) {
        self.switch(device, group, staged);
    }

    fn fill(
        &self,
        device: &mut Device,
        group: usize,
        staged: &Staging<I, O, R>,
    ) -> Registers<I, O, Lane<R>> {
        type Feature<I> = Symbol<normalization::Chunk<I>>;
        type Group<I, R> = Layout<R>::Elements<m![I % ({ Feature<I> })]>;
        if const { I::SIZE <= 16 * 256 } {
            // SAFETY: both physical clusters hold every row, sixteen inputs a slice; only the
            // logical cluster axis changes.
            let input = unsafe {
                Layout::<R>::group(self.view(), group)
                    .reshape::<Chip, Cluster<O>, m![I / 16 # 256], Group<I, R>>()
            };
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
            filled(device, staged)
        }
    }

    fn front(
        &self,
        device: &mut Device,
        group: usize,
        staged: &Staging<I, O, R>,
    ) -> Registers<I, O, Front<R>> {
        type Feature<I> = Symbol<normalization::Chunk<I>>;
        type Group<I, R> = Layout<R>::Elements<m![I % ({ Feature<I> })]>;
        if const { I::SIZE <= 16 * 256 } {
            // SAFETY: as in `fill`.
            let input = unsafe {
                Layout::<R>::group(self.view(), group)
                    .reshape::<Chip, Cluster<O>, m![I / 16 # 256], Group<I, R>>()
            };
            device
                .sub
                .begin(input.tile::<m![{ Lane<R> } / 4], 1, m![
                    { Layout<R>::Selected },
                    { Lane<R> } % 4 # ({ Symbol<Lanes<R>> }),
                    I % ({ Feature<I> })
                ]>(0))
                .fetch::<Front<R>, m![I % 16]>()
                .switch::<Owners<I, O>, m![{ Front<R> }, { Part<I> } / 16]>(
                    SwitchConfig::CustomBroadcast { ring_size: 256 },
                )
                .collect::<m![{ Front<R> }, { Part<I> } / 16], m![I % 16]>()
                .to_trf()
        } else {
            filled_front(device, staged)
        }
    }
}

/// Rows a summed projection of `H` inputs kept, in both clusters: the switch ring takes each
/// owner's run, a piece at a time, to the slices whose part holds it, all pieces at once or one a
/// part, and the parts are written in DM before the lanes fill in input order.
impl<H: AxisName, I: AxisName, O: AxisName, R: AxisName> Stage<I, O, R> for Both<H, I, R> {
    const PARTS: usize = Owned::<H, I>::SIZE / Piece::<H, I>::SIZE;

    fn staged() -> Staging<I, O, R> {
        DmTensor::new()
    }

    fn switch(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R>) {
        type Run<H, I> = Symbol<Piece<H, I>>;
        type Group<H, I, R> = Layout<R>::Elements<m![I % ({ Owner<H, I> })]>;
        const { assert!(fits_both::<H, I>()) };
        // SAFETY: both physical clusters hold every row; only the logical cluster axis changes.
        let input = unsafe {
            Layout::<R>::group(self.view(), group)
                .reshape::<Chip, Cluster<O>, Runs<H, I>, Group<H, I, R>>()
        };
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
    }

    fn switch_front(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R>) {
        type Run<H, I> = Symbol<Piece<H, I>>;
        type Group<H, I, R> = Layout<R>::Elements<m![I % ({ Owner<H, I> })]>;
        const { assert!(fits_both::<H, I>()) };
        // SAFETY: as in `switch`.
        let input = unsafe {
            Layout::<R>::group(self.view(), group)
                .reshape::<Chip, Cluster<O>, Runs<H, I>, Group<H, I, R>>()
        };
        device
            .sub
            .begin(input.tile::<m![{ Lane<R> } / 4], 1, m![
                { Layout<R>::Selected },
                { Lane<R> } % 4 # ({ Symbol<Lanes<R>> }),
                I % ({ Owner<H, I> })
            ]>(0))
            .fetch::<m![{ Front<R> }, I % ({ Owner<H, I> }) / ({ Run<H, I> })], m![
                I % ({ Run<H, I> })
            ]>()
            .switch::<Owners<I, O>, m![
                { Front<R> },
                I % ({ Owner<H, I> }) / ({ Run<H, I> }),
                I % ({ Width<I> }) / ({ Owner<H, I> })
            ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
            .collect::<m![
                { Front<R> },
                I % ({ Owner<H, I> }) / ({ Run<H, I> }),
                I % ({ Width<I> }) / ({ Owner<H, I> })
            ], m![I % ({ Run<H, I> }) # 16]>()
            .commit_trim::<m![I % ({ Run<H, I> })]>()
            .commit_view(staged_front(staged));
    }

    fn part(
        &self,
        device: &mut Device,
        (group, piece): (usize, usize),
        staged: &mut Staging<I, O, R>,
    ) {
        type Run<H, I> = Symbol<Piece<H, I>>;
        type Pieces<H, I> = m![I % ({ Owner<H, I> }) / ({ Run<H, I> })];
        type Group<H, I, R> = Layout<R>::Elements<m![I % ({ Owner<H, I> })]>;
        const { assert!(fits_both::<H, I>()) };
        // SAFETY: as above, only the logical cluster axis changes.
        let input = unsafe {
            Layout::<R>::group(self.view(), group)
                .reshape::<Chip, Cluster<O>, Runs<H, I>, Group<H, I, R>>()
        };
        device
            .sub
            .begin(input.tile::<Pieces<H, I>, 1, m![
                { Layout<R>::Selected },
                { Lane<R> },
                I % ({ Run<H, I> }) # ({ Owner<H, I> })
            ]>(piece))
            .fetch::<Lane<R>, m![I % ({ Run<H, I> })]>()
            .switch::<Owners<I, O>, m![{ Lane<R> }, I % ({ Width<I> }) / ({ Owner<H, I> })]>(
                SwitchConfig::CustomBroadcast { ring_size: 256 },
            )
            .collect::<m![{ Lane<R> }, I % ({ Width<I> }) / ({ Owner<H, I> })], m![
                I % ({ Run<H, I> }) # 16
            ]>()
            .commit_trim::<m![I % ({ Run<H, I> })]>()
            .commit_view(staged.view_mut().tile::<Pieces<H, I>, 1, m![
                { Lane<R> },
                I % ({ Width<I> }) / ({ Owner<H, I> }),
                I % ({ Run<H, I> }) #{!} ({ Owner<H, I> })
            ]>(piece));
    }

    fn fill(
        &self,
        device: &mut Device,
        _: usize,
        staged: &Staging<I, O, R>,
    ) -> Registers<I, O, Lane<R>> {
        filled(device, staged)
    }

    fn front(
        &self,
        device: &mut Device,
        _: usize,
        staged: &Staging<I, O, R>,
    ) -> Registers<I, O, Front<R>> {
        filled_front(device, staged)
    }
}

/// A lane group's staged rows into the TRF, in input order.
fn filled<I: AxisName, O: AxisName, R: AxisName>(
    device: &mut Device,
    staged: &Staging<I, O, R>,
) -> Registers<I, O, Lane<R>> {
    device
        .sub
        .begin(staged.view())
        .fetch::<m![{ Lane<R> }, { Part<I> } / 16], m![I % 16]>()
        .collect::<m![{ Lane<R> }, { Part<I> } / 16], m![I % 16]>()
        .to_trf()
}

/// The first four lanes of a lane group's staging, where its front switches and fills.
fn staged_front<I: AxisName, O: AxisName, R: AxisName>(
    staged: &mut Staging<I, O, R>,
) -> DmTensorViewMut<
    '_,
    bf16,
    Chip,
    Cluster<O>,
    Owners<I, O>,
    m![{ Lane<R> } % 4 #{!} ({ Symbol<Lanes<R>> }), { Part<I> }],
> {
    staged.view_mut().tile::<m![{ Lane<R> } / 4], 1, _>(0)
}

/// A lane group's first four staged rows into the TRF, in input order.
fn filled_front<I: AxisName, O: AxisName, R: AxisName>(
    device: &mut Device,
    staged: &Staging<I, O, R>,
) -> Registers<I, O, Front<R>> {
    device
        .sub
        .begin(staged.view().tile::<m![{ Lane<R> } / 4], 1, m![
            { Lane<R> } % 4 # ({ Symbol<Lanes<R>> }),
            { Part<I> }
        ]>(0))
        .fetch::<m![{ Front<R> }, { Part<I> } / 16], m![I % 16]>()
        .collect::<m![{ Front<R> }, { Part<I> } / 16], m![I % 16]>()
        .to_trf()
}
