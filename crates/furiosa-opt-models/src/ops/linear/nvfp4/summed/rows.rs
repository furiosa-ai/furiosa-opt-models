//! A launch's live rows through NVFP4 weights under the summed contract: each slice owns
//! consecutive output rows over one part of the inputs, as the late contract's rows do, and the
//! codes meet the rows unrestored, a lane group of eight rows at a time.
//!
//! - A weight begun lands its tensor scale, block scales and codes, one transfer each, before any
//!   rows arrive, so a schedule can land it under earlier work; its projection widens the block
//!   scales to FP32 once, then runs every lane group.
//! - A lane group fills half of the tensor registers through the switch ring once and meets every
//!   pass there; a pass covers the most rows whose FP32 block scales fit the vector registers. The
//!   next group fills the other half under the passes when its switch takes at least one part a
//!   pass, the parts past the passes after the last.
//! - A pass's vector engine takes each block's sums four lanes a step, so the last group, when its
//!   live rows lie in its first four lanes, switches, fills and folds those four alone: one step a
//!   block.
//! - From three live lane groups the weight decodes once instead, each code times its block scale
//!   in BF16, which is exact, and every group meets it as the summed passes meet the codes: the
//!   contraction adds a row's blocks and the vector engine divides each sum by the tensor scale.
//!   Fewer groups keep the summed passes, which the decode would cost more than it saves. The two
//!   add a row's products in another order, so its outputs may differ in their last bits.
//! - The projected rows end where the contract places them: in the output's feature slices once
//!   the last pass ends, or [`Kept`] where the passes summed them. Rows bound for the feature
//!   slices are owned in whole chunks of them, the owners' odd factor the top output digit, so
//!   they move there in whole chunks.
//! - Rows another summed projection kept land in both clusters as they lie, an owner's run of
//!   features a slice, and its lane groups stage from there; rows whose runs cannot tile both
//!   clusters' slices that way move to the feature slices first. A projection begun with such
//!   rows issues their landing alone, so a schedule can issue later transfers behind it.

use crate::ops::linear::kernel::Rounded;
use crate::ops::normalization::{self, FeatureSlices, Spread};
use crate::ops::pointwise::Kept;
use crate::ops::vector::Restore;
use crate::specialize::{PACKET, SLICES, TRF, VRF, contraction};
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

/// Output rows one slice owns: its cluster's half of the outputs over the cluster's owners, in whole
/// feature chunks of them when the placement `P` moves the rows to the feature slices.
#[derive(Clone, Debug)]
pub struct Owned<I, O, P>(PhantomData<(I, O, P)>);

impl<I: AxisName, O: AxisName, P: Ownership> AxisName for Owned<I, O, P> {
    const NAME: Ident = Ident::new("Owned");
    const SIZE: usize = {
        let grain = if P::CHUNKED {
            normalization::Chunk::<O>::SIZE
        } else {
            1
        };
        contraction::owned(O::SIZE, I::SIZE / Chunk::<I>::SIZE, grain)
    };
}

/// The odd factor of a cluster's owners, carved as the most significant output digit: its padding
/// to a power of two then lies past the last output.
#[derive(Clone, Debug)]
pub struct TopDigit<I, O, P>(PhantomData<(I, O, P)>);

impl<I: AxisName, O: AxisName, P: Ownership> AxisName for TopDigit<I, O, P> {
    const NAME: Ident = Ident::new("TopDigit");
    const SIZE: usize = contraction::odd(O::SIZE / 2 / Owned::<I, O, P>::SIZE);
}

/// Owner slots the top digit takes: its next power of two.
#[derive(Clone, Debug)]
pub struct TopSlots<I, O, P>(PhantomData<(I, O, P)>);

impl<I: AxisName, O: AxisName, P: Ownership> AxisName for TopSlots<I, O, P> {
    const NAME: Ident = Ident::new("TopSlots");
    const SIZE: usize = TopDigit::<I, O, P>::SIZE.next_power_of_two();
}

/// Output rows a pass covers: the most, a multiple of four dividing the owned rows, whose FP32 block
/// scales over the slice's chunk fit its vector registers beside the tensor scale's eight lanes.
#[derive(Clone, Debug)]
struct Height<I, O, P>(PhantomData<(I, O, P)>);

impl<I: AxisName, O: AxisName, P: Ownership> AxisName for Height<I, O, P> {
    const NAME: Ident = Ident::new("Height");
    const SIZE: usize = {
        let owned = Owned::<I, O, P>::SIZE;
        let mut rows = owned - owned % 4;
        while rows > 4
            && !(owned.is_multiple_of(rows) && rows * Chunk::<I>::SIZE / 16 * 4 + 8 * 4 <= VRF)
        {
            rows -= 4;
        }
        rows
    };
}

type Width<I> = Symbol<Chunk<I>>;
type Owner<I, O, P> = Symbol<Owned<I, O, P>>;
type Top<I, O, P> = Symbol<TopDigit<I, O, P>>;
type Slots<I, O, P> = Symbol<TopSlots<I, O, P>>;
type Tall<I, O, P> = Symbol<Height<I, O, P>>;
/// Each top digit's first half of the outputs in cluster 0, its second in cluster 1.
type Cluster<I, O, P> = m![O % (O / ({ Top<I, O, P> })) / ((O / ({ Top<I, O, P> })) / 2)];
/// A slice owns consecutive output rows over one part of the inputs; the parts of a row's owners
/// are adjacent, innermost, for the inter-slice reducer.
type Owners<I, O, P> = m![
    O / (O / ({ Top<I, O, P> })) # ({ Slots<I, O, P> }),
    O % ((O / ({ Top<I, O, P> })) / 2) / ({ Owner<I, O, P> }),
    I / ({ Width<I> })
];
type Reduced<I, O, P> = m![
    O / (O / ({ Top<I, O, P> })) # ({ Slots<I, O, P> }),
    O % ((O / ({ Top<I, O, P> })) / 2) / ({ Owner<I, O, P> }),
    1 # (I / ({ Width<I> }))
];
/// A slice's part of a row's inputs.
type Part<I> = m![I % ({ Width<I> })];
/// A slice's block scales: one for each 16 inputs of each row it owns.
type Scaled<I, O, P> = m![O % ({ Owner<I, O, P> }), { Part<I> } / 16];

/// A slice's codes: its rows over its part of the inputs.
type Coded<I, O, P> = m![O % ({ Owner<I, O, P> }), { Part<I> }];

/// A slice's weights in BF16, each code times its block scale.
type Decoded<I, O, P> = DmTensor<bf16, Chip, Cluster<I, O, P>, Owners<I, O, P>, Coded<I, O, P>>;

/// Values of a slice's run of `R` values of `D` in whole packets, an odd number of them, so that
/// the slices' runs start alternately in the HBM's two stacks.
#[derive(Clone, Debug)]
pub struct Bursts<R, D>(PhantomData<(R, D)>);

impl<R: M + 'static, D: Scalar> AxisName for Bursts<R, D> {
    const NAME: Ident = Ident::new("Bursts");
    const SIZE: usize = {
        let packets = (R::SIZE * D::BITS / 8).div_ceil(PACKET);
        (packets | 1) * PACKET * 8 / D::BITS
    };
}

/// [`Scaled`] padded to [`Bursts`].
type Padded<I, O, P> = m![{ Scaled<I, O, P> } # ({ Symbol<Bursts<Scaled<I, O, P>, f8e4m3>> })];

/// [`Coded`] padded to [`Bursts`].
type Packed<I, O, P> = m![{ Coded<I, O, P> } # ({ Symbol<Bursts<Coded<I, O, P>, f4e2m1>> })];

/// A weight's codes in HBM as the slices that sum them hold them, a slice's rows after the slice
/// before, in output order: a slice's run lands whole, in whole packets, and no idle slot has one.
pub type Codes<I, O, P> = m![O / ({ Owner<I, O, P> }), I / ({ Width<I> }), { Packed<I, O, P> }];

/// A weight's E4M3 block scales in HBM as the slices that sum them hold them, a slice's run padded
/// to whole packets.
pub type Blocks<I, O, P> = m![O / ({ Owner<I, O, P> }), I / ({ Width<I> }), { Padded<I, O, P> }];

/// A slice's passes over its rows, and the rows of one pass.
type Tiles<I, O, P> = m![O % ({ Owner<I, O, P> }) / ({ Tall<I, O, P> })];
type Elements<I, O, P> = m![O % ({ Tall<I, O, P> })];
/// A lane group's rows, the eight lanes of one contraction.
type Lane<R> = m![{ Layout<R>::Lanes }];
/// The first four lanes of a lane group.
type Front<R> = m![{ Lane<R> } % 4];

/// A launch's rows as the passes of a projection of `I` inputs into `O` outputs sum them: each
/// cluster its half of every top digit's outputs, each owner slice its consecutive outputs of every
/// row.
type Summing<I, O, R, P> =
    DmTensor<bf16, Chip, Cluster<I, O, P>, Reduced<I, O, P>, m![R, O % ({ Owner<I, O, P> })]>;

/// The rows of a lane group's lanes `L` in the tensor registers of every slice of a projection of
/// `I` inputs into `O` outputs, each slice the group's inputs in its part.
type Registers<I, O, L, P> =
    TrfTensor<bf16, Chip, Cluster<I, O, P>, Owners<I, O, P>, L, m![1, { Part<I> }]>;

/// The tensor scale in the vector registers of every slice of a projection.
type Global<I, O, P> = VrfTensor<f32, Chip, Cluster<I, O, P>, Owners<I, O, P>, m![1 # 8]>;

/// The FP32 block scales of every row a slice owns, widened once from E4M3.
type Widened<I, O, P> = DmTensor<
    f32,
    Chip,
    Cluster<I, O, P>,
    Owners<I, O, P>,
    m![O % ({ Owner<I, O, P> }), { Part<I> } / 16],
>;

/// The FP32 block scales of one pass's rows, in the vector registers.
type Scaling<I, O, P> = VrfTensor<
    f32,
    Chip,
    Cluster<I, O, P>,
    Owners<I, O, P>,
    m![{ Elements<I, O, P> }, { Part<I> } / 16],
>;

/// A lane group's rows in each slice's part, in input order, as the switch leaves them in DM.
type Staging<I, O, R, P> =
    DmTensor<bf16, Chip, Cluster<I, O, P>, Owners<I, O, P>, m![{ Lane<R> }, { Part<I> }]>;

/// The rows a projection of `H` inputs into `I` outputs summed, in both clusters: each half of the
/// outputs on half of the slices, an owner's run on a slice of its own, as the passes left them.
type Both<H, I, R> =
    DmTensor<bf16, Chip, normalization::Cluster, Runs<H, I>, m![R, I % ({ Owner<H, I, Kept> })]>;
type Runs<H, I> = m![I / (I / 2), I % (I / 2) / ({ Owner<H, I, Kept> })];

/// Whether [`Both`] holds the rows a projection of `H` inputs into `I` outputs summed: the runs of
/// both halves fit a cluster's slices and a part of `I` inputs is whole runs.
const fn fits_both<H: AxisName, I: AxisName>() -> bool {
    H::SIZE / Chunk::<H>::SIZE >= 2
        && I::SIZE / Chunk::<I>::SIZE >= 2
        && Chunk::<I>::SIZE.is_multiple_of(Owned::<H, I, Kept>::SIZE)
}

/// Inputs a switch packet takes from one owner's run of `H`-input rows: the most, at most a flit,
/// that divide the run, so a packet never spans two owners.
#[derive(Clone, Debug)]
struct Piece<H, I>(PhantomData<(H, I)>);

impl<H: AxisName, I: AxisName> AxisName for Piece<H, I> {
    const NAME: Ident = Ident::new("Piece");
    const SIZE: usize = {
        let run = Owned::<H, I, Kept>::SIZE;
        let mut piece = 16;
        while !run.is_multiple_of(piece) {
            piece -= 4;
        }
        piece
    };
}

/// A weight's tensor scale, E4M3 block scales and codes on the slices of its passes: the weight
/// begun. Its codes are `C`: its own, borrowed from where a schedule landed them ahead, or none
/// while they wait there.
pub struct Landed<I: AxisName, O: AxisName, P: Ownership, C = InDm<I, O, P>> {
    global: DmTensor<f32, Chip, Cluster<I, O, P>, Owners<I, O, P>, m![1 # 8]>,
    blocks: DmTensor<f8e4m3, Chip, Cluster<I, O, P>, Owners<I, O, P>, Padded<I, O, P>>,
    codes: C,
}

/// A weight's codes on the slices of its passes.
type InDm<I, O, P> = DmTensor<f4e2m1, Chip, Cluster<I, O, P>, Owners<I, O, P>, Packed<I, O, P>>;

/// The place a weight lands in, before it lands.
impl<I: AxisName, O: AxisName, P: Ownership> Default for Landed<I, O, P> {
    fn default() -> Self {
        Self {
            global: DmTensor::new(),
            blocks: DmTensor::new(),
            codes: DmTensor::new(),
        }
    }
}

impl<I: AxisName, O: AxisName, P: Ownership> Landed<I, O, P> {
    /// The slices hold the weight: whole parts of its inputs, each owner's run of outputs whole
    /// in the passes, the owners and parts filling every slice.
    const TILED: () = {
        let parts = I::SIZE / Chunk::<I>::SIZE;
        let owners = O::SIZE / 2 / Owned::<I, O, P>::SIZE;
        assert!(
            I::SIZE.is_multiple_of(Chunk::<I>::SIZE)
                && Chunk::<I>::SIZE.is_multiple_of(128)
                && SLICES.is_multiple_of(parts)
                && O::SIZE.is_multiple_of(2 * Owned::<I, O, P>::SIZE)
                && TopSlots::<I, O, P>::SIZE * (owners / TopDigit::<I, O, P>::SIZE) * parts
                    == SLICES
                && Owned::<I, O, P>::SIZE.is_multiple_of(Height::<I, O, P>::SIZE)
                && Height::<I, O, P>::SIZE.is_multiple_of(4)
        )
    };

    /// The weight's tensor scale, block scales and codes, in that order, one transfer each: a
    /// transfer into part of an owner would hold every later command until the part before it
    /// lands.
    fn land<Y: Memory, S: M + 'static, K, V>(
        device: &mut Device,
        weight: &BlockScaled<
            I,
            O,
            f4e2m1,
            f8e4m3,
            Scales<I, O>,
            Summed<W4A16, Y, P, S>,
            Chip,
            K,
            V,
        >,
    ) -> Self
    where
        K: Borrow<HbmTensor<f4e2m1, Chip, Codes<I, O, P>>>,
        V: Borrow<HbmTensor<f8e4m3, Chip, Blocks<I, O, P>>>,
        Held<Y, S>: Borrow<Scale<S>>,
    {
        const { Self::TILED };
        let global: &Scale<S> = weight.contract.global.borrow();
        Self {
            global: global.to_dm(&mut device.tdma),
            blocks: weight.weight_scale.borrow().to_dm(&mut device.tdma),
            codes: weight.weight.borrow().to_dm(&mut device.tdma),
        }
    }
}

impl<I: AxisName, O: AxisName, P: Ownership, C: Borrow<InDm<I, O, P>>> Landed<I, O, P, C> {
    /// Every live row's outputs `bf16(sum_b scale_b (sum_{k in b} code_k x_k) / global)`, a lane
    /// group at a time, each in the slices that sum it.
    fn project<R: AxisName, X: Stage<I, O, R, P>>(
        &self,
        device: &mut Device,
        input: &Live<X, R>,
    ) -> Summing<I, O, R, P> {
        let global = scalar::register(device, &self.global);
        // The block scales widen to FP32 once, so each pass of every group loads its rows' as is.
        let mut scales = Widened::<I, O, P>::new();
        device
            .sub
            .begin(self.blocks.view().unpad::<Scaled<I, O, P>>())
            .fetch::<m![O % ({ Owner<I, O, P> })], m![{ Part<I> } / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![O % ({ Owner<I, O, P> }), { Part<I> } / 128], m![I / 16 % 8]>()
            .commit_trim::<m![I / 16 % 8]>()
            .commit_view(scales.view_mut());
        let mut projected = Summing::<I, O, R, P>::new();
        let (first, count) = input.groups::<Lanes<R>>();
        let rows = input.value();
        // One from three live groups, counted by arithmetic (a device cannot cast a comparison to a
        // count) in a form the compiler bounds by one, so the loop over it frees what it read.
        let groups = R::SIZE / Lanes::<R>::SIZE;
        let dense = if const { R::SIZE / Lanes::<R>::SIZE >= 3 } {
            (count + groups - 3) / groups
        } else {
            0
        };
        // The summed passes' last group folds its first four lanes alone when its live rows end
        // there: one group or none.
        let front = if const { Lanes::<R>::SIZE == 8 } {
            (1 - dense) * count.min(1) * ((11 - (input.start() + input.rows() + 7) % 8) / 8)
        } else {
            0
        };
        let whole = (1 - dense) * count - front;
        let last = first + count.max(1) - 1;
        let operands = (&scales, &global);
        let tiles = <Tiles<I, O, P> as M>::SIZE;
        let mut staged = X::staged();
        let alone = |_: &mut Device, _: usize, _: &mut Staging<I, O, R, P>| {};
        if const { X::PARTS == 0 || X::PARTS >= <Tiles<I, O, P> as M>::SIZE } {
            for _ in 0..whole.min(1) {
                let summed = |device: &mut Device, lanes: &Registers<I, O, Lane<R>, P>, at| {
                    self.pass::<R, Split>(device, (lanes, operands), at, &mut projected);
                };
                // The last group's parts come beside the passes of the whole groups' last.
                let beside = |device: &mut Device, part, staged: &mut _| {
                    for _ in 0..front {
                        rows.part(device, (last, part), staged);
                    }
                };
                Self::meet(
                    device,
                    (rows, &mut staged),
                    (first, whole),
                    (summed, beside),
                );
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
                for tile in 0..tiles {
                    self.pass::<R, Trim>(device, (&lanes, operands), (last, tile), &mut projected);
                }
            }
        } else {
            let summed = |device: &mut Device, lanes: &Registers<I, O, Lane<R>, P>, at| {
                self.pass::<R, Split>(device, (lanes, operands), at, &mut projected);
            };
            Self::meet(device, (rows, &mut staged), (first, whole), (summed, alone));
            for _ in 0..front {
                let mut staged = X::staged();
                rows.switch_front(device, last, &mut staged);
                let lanes = rows.front(device, last, &staged);
                for tile in 0..tiles {
                    self.pass::<R, Trim>(device, (&lanes, operands), (last, tile), &mut projected);
                }
            }
        }
        // Last, so that nothing after it holds the summed passes' FP32 scales.
        for _ in 0..dense {
            let weights = self.decode(device);
            let decoded = |device: &mut Device, lanes: &Registers<I, O, Lane<R>, P>, at| {
                Self::decoded_pass(device, (lanes, (&weights, &global)), at, &mut projected);
            };
            let mut staged = X::staged();
            Self::meet(
                device,
                (rows, &mut staged),
                (first, count),
                (decoded, alone),
            );
        }
        projected
    }

    /// Lane groups `first..first + count` of `rows`, each filled into the TRF once and met there by
    /// every pass `pass` runs. Where the switch writes a group a part a pass or more, each pass of a
    /// group switches a part of the next group beside its contraction, the parts past the passes
    /// after the last, so the next group fills the other half of the TRF; `beside` takes each part
    /// beside the last group's passes. Else a group switches whole before its own.
    fn meet<R: AxisName, X: Stage<I, O, R, P>>(
        device: &mut Device,
        (rows, staged): (&X, &mut Staging<I, O, R, P>),
        (first, count): (usize, usize),
        (mut pass, mut beside): (
            impl FnMut(&mut Device, &Registers<I, O, Lane<R>, P>, (usize, usize)),
            impl FnMut(&mut Device, usize, &mut Staging<I, O, R, P>),
        ),
    ) {
        let tiles = <Tiles<I, O, P> as M>::SIZE;
        if const { X::PARTS == 0 || X::PARTS >= <Tiles<I, O, P> as M>::SIZE } {
            rows.switch(device, first, staged);
            let mut lanes = rows.fill(device, first, staged);
            for next in 1..count {
                for tile in 0..tiles {
                    pass(device, &lanes, (first + next - 1, tile));
                    if const { X::PARTS > 0 } {
                        rows.part(device, (first + next, tile), staged);
                    }
                }
                if const { X::PARTS > <Tiles<I, O, P> as M>::SIZE } {
                    for part in 0..X::PARTS - tiles {
                        rows.part(device, (first + next, tiles + part), staged);
                    }
                }
                lanes = rows.fill(device, first + next, staged);
            }
            for tile in 0..tiles {
                pass(device, &lanes, (first + count - 1, tile));
                if const { X::PARTS > 0 } {
                    beside(device, tile, staged);
                }
            }
            if const { X::PARTS > <Tiles<I, O, P> as M>::SIZE } {
                for part in tiles..X::PARTS {
                    beside(device, part, staged);
                }
            }
        } else {
            for at in 0..count {
                rows.switch(device, first + at, staged);
                let lanes = rows.fill(device, first + at, staged);
                for tile in 0..tiles {
                    pass(device, &lanes, (first + at, tile));
                }
            }
        }
    }

    /// Pass `tile` of lane group `group` over the lanes `lanes` holds: their block sums meet their
    /// rows' block scales, the same for every lane, as `F` folds the lanes.
    fn pass<R: AxisName, F: Fold<I, O, R, P>>(
        &self,
        device: &mut Device,
        (lanes, (scales, global)): (
            &Registers<I, O, F::Filled, P>,
            (&Widened<I, O, P>, &Global<I, O, P>),
        ),
        (group, tile): (usize, usize),
        projected: &mut Summing<I, O, R, P>,
    ) {
        let scale: Scaling<I, O, P> = device
            .sub
            .begin(scales.view().tile::<Tiles<I, O, P>, 1, m![
                { Elements<I, O, P> } # ({ Owner<I, O, P> }), { Part<I> } / 16
            ]>(tile))
            .fetch::<m![{ Elements<I, O, P> }, { Part<I> } / 128], m![I / 16 % 8]>()
            .collect::<m![{ Elements<I, O, P> }, { Part<I> } / 128], m![I / 16 % 8]>()
            .to_vrf();
        let sums = device
            .main
            .begin(
                (self.codes.borrow().view())
                    .unpad::<Coded<I, O, P>>()
                    .tile::<Tiles<I, O, P>, 1, m![
                        { Elements<I, O, P> } # ({ Owner<I, O, P> }), { Part<I> }
                    ]>(tile),
            )
            .fetch::<m![{ Elements<I, O, P> }, { Part<I> } / 64], m![I % 64]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![{ Elements<I, O, P> }, { Part<I> } / 16], m![I % 16]>()
            .contract_outer::<m![{ Elements<I, O, P> }, { Part<I> } / 16, 1], m![I % 16], _, _, _>(
                lanes,
            )
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Elements<I, O, P> }, { Part<I> } / 16, 1]>()
            .contract_lane::<m![{ Elements<I, O, P> }, { Part<I> } / 16, 1], m![{ F::Filled } # 8]>(
                LaneMode::Interleaved,
            )
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero);
        let sums = F::fold(sums, (&scale, global))
            .vector_inter_slice_reduce::<Reduced<I, O, P>, m![{ Elements<I, O, P> }, 1]>(
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
            m![{ Elements<I, O, P> } / 4, 1, { F::Filled }],
            m![O % 4],
            _,
        >(sums)
        .commit_view(F::sums(projected, (group, tile)));
    }

    /// The weights in BF16, each code times its block scale, a pass's rows at a time.
    fn decode(&self, device: &mut Device) -> Decoded<I, O, P> {
        let mut weights = Decoded::<I, O, P>::new();
        for tile in 0..<Tiles<I, O, P> as M>::SIZE {
            let scale: Scaling<I, O, P> = device
                .sub
                .begin(
                    self.blocks
                        .view()
                        .unpad::<Scaled<I, O, P>>()
                        .tile::<Tiles<I, O, P>, 1, m![
                            { Elements<I, O, P> } # ({ Owner<I, O, P> }), { Part<I> } / 16
                        ]>(tile),
                )
                .fetch::<Elements<I, O, P>, m![{ Part<I> } / 16]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Elements<I, O, P> }, { Part<I> } / 128], m![I / 16 % 8]>()
                .to_vrf();
            device
                .main
                .begin(
                    (self.codes.borrow().view())
                        .unpad::<Coded<I, O, P>>()
                        .tile::<Tiles<I, O, P>, 1, m![
                            { Elements<I, O, P> } # ({ Owner<I, O, P> }), { Part<I> }
                        ]>(tile),
                )
                .fetch::<m![{ Elements<I, O, P> }, { Part<I> } / 64], m![I % 64]>()
                .fetch_table_lookup::<bf16>()
                .fetch_cast::<f32>()
                .collect::<m![{ Elements<I, O, P> }, { Part<I> } / 8], m![I % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Elements<I, O, P> }, { Part<I> } / 4], m![I % 4]>()
                .vector_restore::<m![{ Elements<I, O, P> }, { Part<I> } / 8], m![I % 8], _, _>(
                    &scale,
                    |values| values,
                )
                .commit_view(weights.view_mut().tile::<Tiles<I, O, P>, 1, m![
                    { Elements<I, O, P> } #{!} ({ Owner<I, O, P> }), { Part<I> }
                ]>(tile));
        }
        weights
    }

    /// Pass `tile` of lane group `group` over weights that carry their block scales: the
    /// contraction adds a row's blocks, the vector engine divides each sum by the tensor scale.
    fn decoded_pass<R: AxisName>(
        device: &mut Device,
        (lanes, (weights, global)): (
            &Registers<I, O, Lane<R>, P>,
            (&Decoded<I, O, P>, &Global<I, O, P>),
        ),
        (group, tile): (usize, usize),
        projected: &mut Summing<I, O, R, P>,
    ) {
        let sums = device
            .main
            .begin(weights.view().tile::<Tiles<I, O, P>, 1, m![
                { Elements<I, O, P> } # ({ Owner<I, O, P> }), { Part<I> }
            ]>(tile))
            .fetch::<m![{ Elements<I, O, P> }, { Part<I> } / 16], m![I % 16]>()
            .collect::<m![{ Elements<I, O, P> }, { Part<I> } / 16], m![I % 16]>()
            .contract_outer::<m![{ Elements<I, O, P> }, { Part<I> } / 16, 1], m![I % 16], _, _, _>(
                lanes,
            )
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Elements<I, O, P> }, 1]>()
            .contract_lane::<m![{ Elements<I, O, P> }, 1], m![{ Lane<R> } # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![{ Elements<I, O, P> }, 1, { Lane<R> } # 8 / 4],
                m![{ Lane<R> } # 8 % 4],
            >()
            .vector_fp_div(global)
            .vector_widen_concat::<m![{ Elements<I, O, P> }, 1], m![{ Lane<R> } # 8]>()
            .vector_inter_slice_reduce::<Reduced<I, O, P>, m![{ Elements<I, O, P> }, 1]>(
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
            m![{ Elements<I, O, P> } / 4, 1, { Lane<R> }],
            m![O % 4],
            _,
        >(sums)
        .commit_view(<Split as Fold<I, O, R, P>>::sums(projected, (group, tile)));
    }
}

/// How a pass folds the lanes it holds into the vector engine, four lanes a step, and where their
/// sums land.
trait Fold<I: AxisName, O: AxisName, R: AxisName, P: Ownership> {
    /// The lanes of a group the tensor registers hold.
    type Filled: M;
    /// The rows of a pass's outputs these lanes write.
    type Rows: M;

    /// The lanes' block sums met with their block scales, added over the blocks and divided by
    /// the tensor scale.
    fn fold<'l>(
        sums: Tagged<'l, I, O, Self::Filled, P>,
        operands: (&Scaling<I, O, P>, &Global<I, O, P>),
    ) -> Folded<'l, I, O, Self::Filled, P>;

    /// Where pass `tile` of group `group` writes these lanes' sums.
    fn sums(
        projected: &mut Summing<I, O, R, P>,
        at: (usize, usize),
    ) -> DmTensorViewMut<'_, bf16, Chip, Cluster<I, O, P>, Reduced<I, O, P>, Self::Rows>;
}

/// A pass's block sums leaving the lane folder, lanes `L` a packet of eight.
type Tagged<'l, I, O, L, P> = VectorBranchTensor<
    'l,
    { Tu::Main },
    f32,
    Chip,
    Cluster<I, O, P>,
    Owners<I, O, P>,
    m![{ Elements<I, O, P> }, { Part<I> } / 16, 1],
    m![{ L } # 8],
    Fresh,
    { stage::VeOrder::IntraFirst },
>;

/// A pass's sums of every output once its blocks added, lanes `L` a packet of eight.
type Folded<'l, I, O, L, P> = VectorWidenTensor<
    'l,
    { Tu::Main },
    f32,
    Chip,
    Cluster<I, O, P>,
    Owners<I, O, P>,
    m![{ Elements<I, O, P> }, 1],
    m![{ L } # 8],
    Fresh,
    { stage::VeOrder::IntraFirst },
>;

/// Every lane of a group: two vector steps a block.
struct Split;

impl<I: AxisName, O: AxisName, R: AxisName, P: Ownership> Fold<I, O, R, P> for Split {
    type Filled = Lane<R>;
    type Rows = m![
        { Layout<R>::SelectedMut },
        { Lane<R> },
        { Elements<I, O, P> } #{!} ({ Owner<I, O, P> })
    ];

    fn fold<'l>(
        sums: Tagged<'l, I, O, Lane<R>, P>,
        (scale, global): (&Scaling<I, O, P>, &Global<I, O, P>),
    ) -> Folded<'l, I, O, Lane<R>, P> {
        sums.vector_narrow_split::<
            m![{ Elements<I, O, P> }, { Part<I> } / 16, 1, { Lane<R> } # 8 / 4],
            m![{ Lane<R> } # 8 % 4],
        >()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
        .vector_intra_slice_reduce::<
            I,
            m![{ Elements<I, O, P> }, 1, { Lane<R> } # 8 / 4],
            m![{ Lane<R> } # 8 % 4],
        >(IntraSliceReduceOpF32::Add)
        .vector_fp_div(global)
        .vector_widen_concat::<m![{ Elements<I, O, P> }, 1], m![{ Lane<R> } # 8]>()
    }

    fn sums(
        projected: &mut Summing<I, O, R, P>,
        (group, tile): (usize, usize),
    ) -> DmTensorViewMut<'_, bf16, Chip, Cluster<I, O, P>, Reduced<I, O, P>, Self::Rows> {
        projected
            .view_mut()
            .tile::<Layout<R>::Groups, 1, Layout<R>::GroupMut<m![O % ({ Owner<I, O, P> })]>>(group)
            .tile::<Tiles<I, O, P>, 1, Self::Rows>(tile)
    }
}

/// The first four lanes of a group, which hold every live row of it: one vector step a block.
struct Trim;

impl<I: AxisName, O: AxisName, R: AxisName, P: Ownership> Fold<I, O, R, P> for Trim {
    type Filled = Front<R>;
    type Rows = m![
        { Layout<R>::SelectedMut },
        { Lane<R> } % 4 #{!} ({ Symbol<Lanes<R>> }),
        { Elements<I, O, P> } #{!} ({ Owner<I, O, P> })
    ];

    fn fold<'l>(
        sums: Tagged<'l, I, O, Front<R>, P>,
        (scale, global): (&Scaling<I, O, P>, &Global<I, O, P>),
    ) -> Folded<'l, I, O, Front<R>, P> {
        sums.vector_narrow_trim::<Front<R>>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
            .vector_intra_slice_reduce::<I, m![{ Elements<I, O, P> }, 1], Front<R>>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_fp_div(global)
            .vector_widen_pad::<m![{ Front<R> } # 8]>()
    }

    fn sums(
        projected: &mut Summing<I, O, R, P>,
        (group, tile): (usize, usize),
    ) -> DmTensorViewMut<'_, bf16, Chip, Cluster<I, O, P>, Reduced<I, O, P>, Self::Rows> {
        projected
            .view_mut()
            .tile::<Layout<R>::Groups, 1, Layout<R>::GroupMut<m![O % ({ Owner<I, O, P> })]>>(group)
            .tile::<m![{ Lane<R> } / 4], 1, m![
                { Layout<R>::SelectedMut },
                { Lane<R> } % 4 #{!} ({ Symbol<Lanes<R>> }),
                O % ({ Owner<I, O, P> })
            ]>(0)
            .tile::<Tiles<I, O, P>, 1, Self::Rows>(tile)
    }
}

/// How a placement has a projection's slices own its outputs: in whole feature chunks of them when
/// its rows move to the feature slices, so the move takes whole chunks; else as the passes split them.
pub trait Ownership: Clone + core::fmt::Debug + 'static {
    /// Whether the owners hold whole feature chunks of the outputs.
    const CHUNKED: bool;
}

impl Ownership for Features {
    const CHUNKED: bool = true;
}

impl Ownership for Kept {
    const CHUNKED: bool = false;
}

/// Where a projection's rows end once its passes summed them: the rows `place` gives of `R` rows
/// projected from `I` inputs into `O` outputs.
pub trait Place<I: AxisName, O: AxisName, R: AxisName>: Ownership + Sized {
    type Rows;
    fn place(device: &mut Device, rows: Live<Summing<I, O, R, Self>, R>) -> Live<Self::Rows, R>;
}

impl<I: AxisName, O: AxisName, R: AxisName> Place<I, O, R> for Features {
    type Rows = Spread<O, R>;

    fn place(device: &mut Device, rows: Live<Summing<I, O, R, Self>, R>) -> Live<Self::Rows, R> {
        let mut spread = Spread::<O, R>::new();
        transfer(device, rows.with(rows.value()), &mut spread);
        rows.with(spread)
    }
}

impl<I: AxisName, O: AxisName, R: AxisName> Place<I, O, R> for Kept {
    type Rows = Kept<Summing<I, O, R, Self>>;

    fn place(_: &mut Device, rows: Live<Summing<I, O, R, Self>, R>) -> Live<Self::Rows, R> {
        let live = rows.with(());
        live.with(Kept(rows.into_value()))
    }
}

/// The live rows of a launch in HBM through a weight under the summed contract, into HBM.
impl<I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, S: M + 'static>
    Op<(Live<(), R>, &HbmTensor<bf16, Chip, m![R, I]>)>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, Features, S>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, Codes<I, O, Features>>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Blocks<I, O, Features>>>,
    Held<Y, S>: Borrow<Scale<S>>,
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
impl<I: AxisName, O: AxisName, K, V, Y: Memory, P, S: M + 'static> Op<Begin<()>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, Codes<I, O, P>>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Blocks<I, O, P>>>,
    Held<Y, S>: Borrow<Scale<S>>,
    P: Ownership,
{
    type Output = Landed<I, O, P>;

    fn forward(&self, device: &mut Device, _: Begin<()>) -> Self::Output {
        Landed::land(device, self)
    }
}

/// The same weight's codes landed into a place made before, so a schedule can land them a step
/// ahead of the rows they meet and hold them across other work; its scales land when it is begun
/// beside them.
impl<'l, I: AxisName, O: AxisName, K, V, Y: Memory, P, S: M + 'static> Op<&'l mut Landed<I, O, P>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, Codes<I, O, P>>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Blocks<I, O, P>>>,
    Held<Y, S>: Borrow<Scale<S>>,
    P: Ownership,
{
    type Output = ();

    fn forward(&self, device: &mut Device, landed: &'l mut Landed<I, O, P>) {
        const { Landed::<I, O, P>::TILED };
        (self.weight.borrow().view()).to_dm_view(&mut device.tdma, landed.codes.view_mut());
    }
}

/// The same weight begun beside its codes landed in a place: its tensor scale and block scales,
/// one transfer each.
impl<'l, I: AxisName, O: AxisName, K, V, Y: Memory, P, S: M + 'static>
    Op<Begin<&'l Landed<I, O, P>>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
where
    V: Borrow<HbmTensor<f8e4m3, Chip, Blocks<I, O, P>>>,
    Held<Y, S>: Borrow<Scale<S>>,
    P: Ownership,
{
    type Output = Landed<I, O, P, ()>;

    fn forward(&self, device: &mut Device, _: Begin<&'l Landed<I, O, P>>) -> Self::Output {
        let global: &Scale<S> = self.contract.global.borrow();
        Landed {
            global: global.to_dm(&mut device.tdma),
            blocks: self.weight_scale.borrow().to_dm(&mut device.tdma),
            codes: (),
        }
    }
}

/// The live rows of a launch in their feature slices through a weight under the summed contract,
/// into where its placement `P` puts them; the projected rows keep the live rows.
impl<'x, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P: Place<I, O, R>, S: M + 'static>
    Op<&'x Live<Spread<I, R>, R>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, Codes<I, O, P>>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Blocks<I, O, P>>>,
    Held<Y, S>: Borrow<Scale<S>>,
{
    type Output = Live<P::Rows, R>;

    fn forward(&self, device: &mut Device, input: &'x Live<Spread<I, R>, R>) -> Self::Output {
        let landed = self.forward(&mut *device, Begin(()));
        self.forward(device, (landed, input))
    }
}

/// The same rows through the weight as it landed.
impl<'x, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P: Place<I, O, R>, S: M + 'static>
    Op<(Landed<I, O, P>, &'x Live<Spread<I, R>, R>)>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
{
    type Output = Live<P::Rows, R>;

    fn forward(
        &self,
        device: &mut Device,
        (landed, input): (Landed<I, O, P>, &'x Live<Spread<I, R>, R>),
    ) -> Self::Output {
        let projected = landed.project(device, input);
        P::place(device, input.with(projected))
    }
}

/// The same rows through the weight begun beside its codes, which lie in their place.
impl<
    'l,
    'x,
    I: AxisName,
    O: AxisName,
    R: AxisName,
    K,
    V,
    Y: Memory,
    P: Place<I, O, R>,
    S: M + 'static,
>
    Op<(
        (Landed<I, O, P, ()>, &'l Landed<I, O, P>),
        &'x Live<Spread<I, R>, R>,
    )> for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
{
    type Output = Live<P::Rows, R>;

    fn forward(
        &self,
        device: &mut Device,
        ((scales, place), input): (
            (Landed<I, O, P, ()>, &'l Landed<I, O, P>),
            &'x Live<Spread<I, R>, R>,
        ),
    ) -> Self::Output {
        let Landed { global, blocks, .. } = scales;
        let landed = Landed {
            global,
            blocks,
            codes: &place.codes,
        };
        let projected = landed.project(device, input);
        P::place(device, input.with(projected))
    }
}

/// The live rows a summed projection of `H` inputs kept where its passes summed them, through a
/// weight under the summed contract, into where its placement `P` puts them.
impl<'x, H: AxisName, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P, S: M + 'static>
    Op<&'x Live<Kept<Summing<H, I, R, Kept>>, R>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, Codes<I, O, P>>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Blocks<I, O, P>>>,
    Held<Y, S>: Borrow<Scale<S>>,
    P: Place<I, O, R>,
{
    type Output = Live<P::Rows, R>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<Kept<Summing<H, I, R, Kept>>, R>,
    ) -> Self::Output {
        let landed = self.forward(&mut *device, Begin(()));
        self.forward(device, (landed, input))
    }
}

/// The same rows through the weight as it landed: they land in both clusters as they lie, or in
/// the feature slices where they cannot lie so, then they project.
impl<'x, H: AxisName, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P, S: M + 'static>
    Op<(Landed<I, O, P>, &'x Live<Kept<Summing<H, I, R, Kept>>, R>)>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
where
    P: Place<I, O, R>,
{
    type Output = Live<P::Rows, R>;

    fn forward(
        &self,
        device: &mut Device,
        (landed, input): (Landed<I, O, P>, &'x Live<Kept<Summing<H, I, R, Kept>>, R>),
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
impl<'x, H: AxisName, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P, S: M + 'static>
    Op<Begin<(Landed<I, O, P>, &'x Live<Kept<Summing<H, I, R, Kept>>, R>)>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
where
    P: Ownership,
{
    type Output = (Landed<I, O, P>, Live<Both<H, I, R>, R>);

    fn forward(
        &self,
        device: &mut Device,
        Begin((landed, input)): Begin<(Landed<I, O, P>, &'x Live<Kept<Summing<H, I, R, Kept>>, R>)>,
    ) -> Self::Output {
        const { assert!(fits_both::<H, I>()) };
        let mut both = Both::<H, I, R>::new();
        transfer(device, input.with(&input.value().0), &mut both);
        (landed, input.with(both))
    }
}

/// Rows landed in both clusters through the weight as it landed.
impl<H: AxisName, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P, S: M + 'static>
    Op<(Landed<I, O, P>, Live<Both<H, I, R>, R>)>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
where
    P: Place<I, O, R>,
{
    type Output = Live<P::Rows, R>;

    fn forward(
        &self,
        device: &mut Device,
        (landed, rows): (Landed<I, O, P>, Live<Both<H, I, R>, R>),
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
trait Stage<I: AxisName, O: AxisName, R: AxisName, P: Ownership> {
    /// The parts the switch can write a group in, none when it fills the TRF itself.
    const PARTS: usize;

    fn staged() -> Staging<I, O, R, P>;
    fn switch(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R, P>);
    /// The first four lanes of `group` alone, as `switch` switches every lane.
    fn switch_front(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R, P>);
    fn part(&self, device: &mut Device, at: (usize, usize), staged: &mut Staging<I, O, R, P>);
    fn fill(
        &self,
        device: &mut Device,
        group: usize,
        staged: &Staging<I, O, R, P>,
    ) -> Registers<I, O, Lane<R>, P>;
    /// The first four lanes of `group` alone, as `fill` fills every lane.
    fn front(
        &self,
        device: &mut Device,
        group: usize,
        staged: &Staging<I, O, R, P>,
    ) -> Registers<I, O, Front<R>, P>;
}

/// Rows in their feature slices: the switch ring takes each feature slice's inputs to the slices
/// that hold them, straight into the TRF where a slice holds one packet of a row.
impl<I: AxisName, O: AxisName, R: AxisName, P: Ownership> Stage<I, O, R, P> for Spread<I, R> {
    const PARTS: usize = if I::SIZE <= 16 * 256 { 0 } else { 1 };

    fn staged() -> Staging<I, O, R, P> {
        DmTensor::new()
    }

    fn switch(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R, P>) {
        type Feature<I> = Symbol<normalization::Chunk<I>>;
        type Group<I, R> = Layout<R>::Elements<m![I % ({ Feature<I> })]>;
        if const { I::SIZE > 16 * 256 } {
            // A slice holds several switch packets, delivered with other slices' inputs
            // innermost, so the parts are written in DM before the lanes fill in input order.
            // SAFETY: both physical clusters hold every row; only the logical cluster axis changes.
            let input = unsafe {
                Layout::<R>::group(self.view(), group)
                    .reshape::<Chip, Cluster<I, O, P>, FeatureSlices<I>, Group<I, R>>()
            };
            device
                .sub
                .begin(input)
                .fetch::<m![{ Lane<R> }, I % ({ Feature<I> }) / 16], m![I % 16]>()
                .switch::<Owners<I, O, P>, m![
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

    fn switch_front(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R, P>) {
        type Feature<I> = Symbol<normalization::Chunk<I>>;
        type Group<I, R> = Layout<R>::Elements<m![I % ({ Feature<I> })]>;
        if const { I::SIZE > 16 * 256 } {
            // SAFETY: as in `switch`.
            let input = unsafe {
                Layout::<R>::group(self.view(), group)
                    .reshape::<Chip, Cluster<I, O, P>, FeatureSlices<I>, Group<I, R>>()
            };
            device
                .sub
                .begin(input.tile::<m![{ Lane<R> } / 4], 1, m![
                    { Layout<R>::Selected },
                    { Lane<R> } % 4 # ({ Symbol<Lanes<R>> }),
                    I % ({ Feature<I> })
                ]>(0))
                .fetch::<m![{ Front<R> }, I % ({ Feature<I> }) / 16], m![I % 16]>()
                .switch::<Owners<I, O, P>, m![
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

    fn part(
        &self,
        device: &mut Device,
        (group, _): (usize, usize),
        staged: &mut Staging<I, O, R, P>,
    ) {
        self.switch(device, group, staged);
    }

    fn fill(
        &self,
        device: &mut Device,
        group: usize,
        staged: &Staging<I, O, R, P>,
    ) -> Registers<I, O, Lane<R>, P> {
        type Feature<I> = Symbol<normalization::Chunk<I>>;
        type Group<I, R> = Layout<R>::Elements<m![I % ({ Feature<I> })]>;
        if const { I::SIZE <= 16 * 256 } {
            // SAFETY: both physical clusters hold every row, sixteen inputs a slice; only the
            // logical cluster axis changes.
            let input = unsafe {
                Layout::<R>::group(self.view(), group)
                    .reshape::<Chip, Cluster<I, O, P>, m![I / 16 # 256], Group<I, R>>()
            };
            device
                .sub
                .begin(input)
                .fetch::<Lane<R>, m![I % 16]>()
                .switch::<Owners<I, O, P>, m![{ Lane<R> }, { Part<I> } / 16]>(
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
        staged: &Staging<I, O, R, P>,
    ) -> Registers<I, O, Front<R>, P> {
        type Feature<I> = Symbol<normalization::Chunk<I>>;
        type Group<I, R> = Layout<R>::Elements<m![I % ({ Feature<I> })]>;
        if const { I::SIZE <= 16 * 256 } {
            // SAFETY: as in `fill`.
            let input = unsafe {
                Layout::<R>::group(self.view(), group)
                    .reshape::<Chip, Cluster<I, O, P>, m![I / 16 # 256], Group<I, R>>()
            };
            device
                .sub
                .begin(input.tile::<m![{ Lane<R> } / 4], 1, m![
                    { Layout<R>::Selected },
                    { Lane<R> } % 4 # ({ Symbol<Lanes<R>> }),
                    I % ({ Feature<I> })
                ]>(0))
                .fetch::<Front<R>, m![I % 16]>()
                .switch::<Owners<I, O, P>, m![{ Front<R> }, { Part<I> } / 16]>(
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
impl<H: AxisName, I: AxisName, O: AxisName, R: AxisName, P: Ownership> Stage<I, O, R, P>
    for Both<H, I, R>
{
    const PARTS: usize = Owned::<H, I, Kept>::SIZE / Piece::<H, I>::SIZE;

    fn staged() -> Staging<I, O, R, P> {
        DmTensor::new()
    }

    fn switch(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R, P>) {
        type Run<H, I> = Symbol<Piece<H, I>>;
        type Group<H, I, R> = Layout<R>::Elements<m![I % ({ Owner<H, I, Kept> })]>;
        const { assert!(fits_both::<H, I>()) };
        // SAFETY: both physical clusters hold every row; only the logical cluster axis changes.
        let input = unsafe {
            Layout::<R>::group(self.view(), group)
                .reshape::<Chip, Cluster<I, O, P>, Runs<H, I>, Group<H, I, R>>()
        };
        device
            .sub
            .begin(input)
            .fetch::<m![{ Lane<R> }, I % ({ Owner<H, I, Kept> }) / ({ Run<H, I> })], m![
                I % ({ Run<H, I> })
            ]>()
            .switch::<Owners<I, O, P>, m![
                { Lane<R> },
                I % ({ Owner<H, I, Kept> }) / ({ Run<H, I> }),
                I % ({ Width<I> }) / ({ Owner<H, I, Kept> })
            ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
            .collect::<m![
                { Lane<R> },
                I % ({ Owner<H, I, Kept> }) / ({ Run<H, I> }),
                I % ({ Width<I> }) / ({ Owner<H, I, Kept> })
            ], m![I % ({ Run<H, I> }) # 16]>()
            .commit_trim::<m![I % ({ Run<H, I> })]>()
            .commit_view(staged.view_mut());
    }

    fn switch_front(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R, P>) {
        type Run<H, I> = Symbol<Piece<H, I>>;
        type Group<H, I, R> = Layout<R>::Elements<m![I % ({ Owner<H, I, Kept> })]>;
        const { assert!(fits_both::<H, I>()) };
        // SAFETY: as in `switch`.
        let input = unsafe {
            Layout::<R>::group(self.view(), group)
                .reshape::<Chip, Cluster<I, O, P>, Runs<H, I>, Group<H, I, R>>()
        };
        device
            .sub
            .begin(input.tile::<m![{ Lane<R> } / 4], 1, m![
                { Layout<R>::Selected },
                { Lane<R> } % 4 # ({ Symbol<Lanes<R>> }),
                I % ({ Owner<H, I, Kept> })
            ]>(0))
            .fetch::<m![{ Front<R> }, I % ({ Owner<H, I, Kept> }) / ({ Run<H, I> })], m![
                I % ({ Run<H, I> })
            ]>()
            .switch::<Owners<I, O, P>, m![
                { Front<R> },
                I % ({ Owner<H, I, Kept> }) / ({ Run<H, I> }),
                I % ({ Width<I> }) / ({ Owner<H, I, Kept> })
            ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
            .collect::<m![
                { Front<R> },
                I % ({ Owner<H, I, Kept> }) / ({ Run<H, I> }),
                I % ({ Width<I> }) / ({ Owner<H, I, Kept> })
            ], m![I % ({ Run<H, I> }) # 16]>()
            .commit_trim::<m![I % ({ Run<H, I> })]>()
            .commit_view(staged_front(staged));
    }

    fn part(
        &self,
        device: &mut Device,
        (group, piece): (usize, usize),
        staged: &mut Staging<I, O, R, P>,
    ) {
        type Run<H, I> = Symbol<Piece<H, I>>;
        type Pieces<H, I> = m![I % ({ Owner<H, I, Kept> }) / ({ Run<H, I> })];
        type Group<H, I, R> = Layout<R>::Elements<m![I % ({ Owner<H, I, Kept> })]>;
        const { assert!(fits_both::<H, I>()) };
        // SAFETY: as above, only the logical cluster axis changes.
        let input = unsafe {
            Layout::<R>::group(self.view(), group)
                .reshape::<Chip, Cluster<I, O, P>, Runs<H, I>, Group<H, I, R>>()
        };
        device
            .sub
            .begin(input.tile::<Pieces<H, I>, 1, m![
                { Layout<R>::Selected },
                { Lane<R> },
                I % ({ Run<H, I> }) # ({ Owner<H, I, Kept> })
            ]>(piece))
            .fetch::<Lane<R>, m![I % ({ Run<H, I> })]>()
            .switch::<Owners<I, O, P>, m![{ Lane<R> }, I % ({ Width<I> }) / ({ Owner<H, I, Kept> })]>(
                SwitchConfig::CustomBroadcast { ring_size: 256 },
            )
            .collect::<m![{ Lane<R> }, I % ({ Width<I> }) / ({ Owner<H, I, Kept> })], m![
                I % ({ Run<H, I> }) # 16
            ]>()
            .commit_trim::<m![I % ({ Run<H, I> })]>()
            .commit_view(staged.view_mut().tile::<Pieces<H, I>, 1, m![
                { Lane<R> },
                I % ({ Width<I> }) / ({ Owner<H, I, Kept> }),
                I % ({ Run<H, I> }) #{!} ({ Owner<H, I, Kept> })
            ]>(piece));
    }

    fn fill(
        &self,
        device: &mut Device,
        _: usize,
        staged: &Staging<I, O, R, P>,
    ) -> Registers<I, O, Lane<R>, P> {
        filled(device, staged)
    }

    fn front(
        &self,
        device: &mut Device,
        _: usize,
        staged: &Staging<I, O, R, P>,
    ) -> Registers<I, O, Front<R>, P> {
        filled_front(device, staged)
    }
}

/// A lane group's staged rows into the TRF, in input order.
fn filled<I: AxisName, O: AxisName, R: AxisName, P: Ownership>(
    device: &mut Device,
    staged: &Staging<I, O, R, P>,
) -> Registers<I, O, Lane<R>, P> {
    device
        .sub
        .begin(staged.view())
        .fetch::<m![{ Lane<R> }, { Part<I> } / 16], m![I % 16]>()
        .collect::<m![{ Lane<R> }, { Part<I> } / 16], m![I % 16]>()
        .to_trf()
}

/// The first four lanes of a lane group's staging, where its front switches and fills.
fn staged_front<I: AxisName, O: AxisName, R: AxisName, P: Ownership>(
    staged: &mut Staging<I, O, R, P>,
) -> DmTensorViewMut<
    '_,
    bf16,
    Chip,
    Cluster<I, O, P>,
    Owners<I, O, P>,
    m![{ Lane<R> } % 4 #{!} ({ Symbol<Lanes<R>> }), { Part<I> }],
> {
    staged.view_mut().tile::<m![{ Lane<R> } / 4], 1, _>(0)
}

/// A lane group's first four staged rows into the TRF, in input order.
fn filled_front<I: AxisName, O: AxisName, R: AxisName, P: Ownership>(
    device: &mut Device,
    staged: &Staging<I, O, R, P>,
) -> Registers<I, O, Front<R>, P> {
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
