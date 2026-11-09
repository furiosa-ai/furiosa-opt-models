//! A launch's live rows through NVFP4 weights under the summed contract: each slice owns
//! consecutive output rows over one part of the inputs, as the late contract's rows do, and the
//! codes meet the rows unrestored, a lane group of eight rows at a time.
//!
//! - A weight begun lands its tensor scale, block scales and codes, one transfer each, before any
//!   rows arrive, so a schedule can land it under earlier work.
//! - A lane group fills half of the tensor registers through the switch ring once and meets every
//!   pass there; a pass covers the most rows whose FP32 block scales fit the vector registers, and
//!   loads them there from E4M3 as it begins. The next group fills the other half under the passes
//!   when its switch takes at least one part a pass, the parts past the passes after the last.
//! - A pass's vector engine takes each block's sums four lanes a step, so the last group, when its
//!   live rows lie in its first four lanes, switches, fills and folds those four alone: one step a
//!   block.
//! - From three live lane groups the weight decodes once instead, each code times its block scale
//!   in BF16, which is exact, and every group meets it as the summed passes meet the codes: the
//!   contraction adds a row's blocks and the vector engine divides each sum by the tensor scale.
//!   Fewer groups keep the summed passes, which the decode would cost more than it saves. The two
//!   add a row's products in another order, so its outputs may differ in their last bits.
//! - The decoded weight lies in the order the switch delivers a part's inputs from the feature
//!   slices ([`Switched`]), so each group switches from there straight into the TRF, and the next
//!   group switches into the other half beside the passes of the one before. Rows another
//!   projection kept land in the feature slices for it first, beside the decode.
//! - The projected rows end where the contract places them: in the output's feature slices once
//!   the last pass ends, or [`Kept`] where the passes summed them. Rows bound for the feature
//!   slices are owned in whole chunks of them, the owners' odd factor the top output digit, so
//!   they move there in whole chunks.
//! - Rows another summed projection kept land in both clusters as they lie, an owner's run of
//!   features a slice, and its lane groups stage from there; rows whose runs cannot tile both
//!   clusters' slices that way move to the feature slices first. A projection begun with such
//!   rows issues their landing alone, so a schedule can issue later transfers behind it.
//! - A projection whose clusters halve its inputs ([`Exchanged`]) reads rows another summed
//!   projection kept where they lie, each cluster its own half, with no transfer first (from three
//!   live groups each cluster lays its half in its feature slices once); one transfer gathers both
//!   clusters' FP32 partial sums into the output's feature slices, and a pass adds them there.
//!   Over a decoded weight an owner's feature chunks are as many as its parts, so the inter-slice
//!   reduce scatters them, each chunk's sums onto its own feature slice, and the clusters swap
//!   their partial sums there instead.

use crate::ops::linear::kernel::Rounding;
use crate::ops::normalization::{self, FeatureSlices, Spread};
use crate::ops::pointwise::Kept;
use crate::ops::vector::Restore;
use crate::specialize::{PACKET, SLICES, TRF, VRF, contraction};
use crate::tensor::form::BlockScaled;
use crate::tensor::form::nvfp4::{Exchanged, Features, Held, Scale, Scales, Summed, W4A16};
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

/// Output rows one slice owns: its cluster's outputs over the cluster's owners, in whole feature
/// chunks of them when the placement `P` moves the rows to the feature slices.
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
        let parts = I::SIZE / Chunk::<I>::SIZE;
        // `owned` shares half of the outputs it is given among the seats of its parts.
        if P::HALVES_INPUTS {
            contraction::owned(2 * O::SIZE, parts / 2, grain)
        } else {
            contraction::owned(O::SIZE, parts, grain)
        }
    };
}

/// The odd factor of a cluster's owners, carved as the most significant output digit: its padding
/// to a power of two then lies past the last output.
#[derive(Clone, Debug)]
pub struct TopDigit<I, O, P>(PhantomData<(I, O, P)>);

impl<I: AxisName, O: AxisName, P: Ownership> AxisName for TopDigit<I, O, P> {
    const NAME: Ident = Ident::new("TopDigit");
    const SIZE: usize = contraction::odd(outputs::<O, P>() / Owned::<I, O, P>::SIZE);
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
type Cluster<I, O, P> = <P as Ownership>::Cluster<I, O>;
type Owners<I, O, P> = <P as Ownership>::Owners<I, O>;
type Reduced<I, O, P> = <P as Ownership>::Reduced<I, O>;

/// Each top digit's first half of the outputs in cluster 0, its second in cluster 1.
type OutputCluster<I, O, P> = m![O % (O / ({ Top<I, O, P> })) / ((O / ({ Top<I, O, P> })) / 2)];
/// A slice owns consecutive output rows of its cluster's half over one part of the inputs; the
/// parts of a row's owners are adjacent, innermost, for the inter-slice reducer.
type OutputOwners<I, O, P> = m![
    O / (O / ({ Top<I, O, P> })) # ({ Slots<I, O, P> }),
    O % ((O / ({ Top<I, O, P> })) / 2) / ({ Owner<I, O, P> }),
    I / ({ Width<I> })
];
type OutputReduced<I, O, P> = m![
    O / (O / ({ Top<I, O, P> })) # ({ Slots<I, O, P> }),
    O % ((O / ({ Top<I, O, P> })) / 2) / ({ Owner<I, O, P> }),
    1 # (I / ({ Width<I> }))
];

/// The first half of the inputs in cluster 0, the second in cluster 1.
type InputCluster<I> = m![I / (I / 2)];
/// A slice owns consecutive output rows over one part of its cluster's half of the inputs.
type InputOwners<I, O, P> = m![
    O / (O / ({ Top<I, O, P> })) # ({ Slots<I, O, P> }),
    O % (O / ({ Top<I, O, P> })) / ({ Owner<I, O, P> }),
    I % (I / 2) / ({ Width<I> })
];
type InputReduced<I, O, P> = m![
    O / (O / ({ Top<I, O, P> })) # ({ Slots<I, O, P> }),
    O % (O / ({ Top<I, O, P> })) / ({ Owner<I, O, P> }),
    1 # (I / 2 / ({ Width<I> }))
];
/// A slice's part of a row's inputs.
type Part<I> = m![I % ({ Width<I> })];
/// A slice's block scales: one for each 16 inputs of each row it owns.
type Scaled<I, O, P> = m![O % ({ Owner<I, O, P> }), { Part<I> } / 16];

/// A slice's codes: its rows over its part of the inputs.
type Coded<I, O, P> = m![O % ({ Owner<I, O, P> }), { Part<I> }];

/// A feature slice's run of a row's inputs.
type Feature<I> = Symbol<normalization::Chunk<I>>;

/// A part's 16-input packets in the order the switch delivers them from the feature slices: a
/// slice's own packets outer, the slices inner. The input order where a slice holds one packet.
type Packets<I> = m![I % ({ Feature<I> }) / 16, I % ({ Width<I> }) / ({ Feature<I> })];

/// A part's inputs in the order the switch delivers them from the feature slices.
type Switched<I> = m![{ Packets<I> }, I % 16];

/// A slice's weights in BF16, each code times its block scale, its inputs in [`Switched`] order.
type Decoded<I, O, P> = DmTensor<
    bf16,
    Chip,
    Cluster<I, O, P>,
    Owners<I, O, P>,
    m![O % ({ Owner<I, O, P> }), { Switched<I> }],
>;

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

/// A launch's rows as the passes of a projection of `I` inputs into `O` outputs sum them, each
/// owner slice its consecutive outputs of every row over its cluster's inputs.
type Summing<I, O, R, P> = DmTensor<
    <P as Ownership>::Sum,
    Chip,
    Cluster<I, O, P>,
    Reduced<I, O, P>,
    m![R, O % ({ Owner<I, O, P> })],
>;

/// The rows of a lane group's lanes `L` in the tensor registers of every slice of a projection of
/// `I` inputs into `O` outputs, each slice the group's inputs in its part.
type Registers<I, O, L, P> =
    TrfTensor<bf16, Chip, Cluster<I, O, P>, Owners<I, O, P>, L, m![1, { Part<I> }]>;

/// [`Registers`] of a whole lane group, its inputs in [`Switched`] order.
type Switching<I, O, R, P> =
    TrfTensor<bf16, Chip, Cluster<I, O, P>, Owners<I, O, P>, Lane<R>, m![1, { Switched<I> }]>;

/// The tensor scale in the vector registers of every slice of a projection.
type Global<I, O, P> = VrfTensor<f32, Chip, Cluster<I, O, P>, Owners<I, O, P>, m![1 # 8]>;

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

/// Rows in the feature slices of each cluster's half of the features: a slice's run of every row,
/// in the cluster whose passes read that half, the runs of a part of a weight's inputs on slices
/// of their own.
type HalfSpread<I, R> = DmTensor<
    bf16,
    Chip,
    InputCluster<I>,
    m![
        I % (I / 2) / ({ Width<I> }),
        I % ({ Width<I> }) / ({ Feature<I> }) # ({ Symbol<Seats<I>> })
    ],
    m![R, I % ({ Feature<I> })],
>;

/// A part of a cluster's half of a projection's inputs, and an input within its part, as axes of
/// their own: the runs and the feature chunks a regroup moves between cut a part differently, so a
/// transfer keeps each side's coordinates of a part consecutive only with the parts named apart.
#[derive(Clone, Debug)]
struct Section<I>(PhantomData<I>);

impl<I: AxisName> AxisName for Section<I> {
    const NAME: Ident = Ident::new("Section");
    const SIZE: usize = I::SIZE / 2 / Chunk::<I>::SIZE;
}

/// An input within its part of a cluster's half of a projection's inputs: see [`Section`].
#[derive(Clone, Debug)]
struct Within<I>(PhantomData<I>);

impl<I: AxisName> AxisName for Within<I> {
    const NAME: Ident = Ident::new("Within");
    const SIZE: usize = Chunk::<I>::SIZE;
}

/// Slices each part of a cluster's half of the inputs takes in [`HalfSpread`]: an equal share of
/// the cluster, so that no part's runs straddle the slices of two processing elements, which a
/// transfer splits.
#[derive(Clone, Debug)]
pub struct Seats<I>(PhantomData<I>);

impl<I: AxisName> AxisName for Seats<I> {
    const NAME: Ident = Ident::new("Seats");
    const SIZE: usize = SLICES / (I::SIZE / 2 / Chunk::<I>::SIZE);
}

/// Rows a summed projection of `H` inputs into `I` outputs kept, each cluster the half of `I` its
/// passes summed, and the same rows in each cluster's half of the feature slices, written from
/// three live lane groups for the passes over a decoded weight.
type Halved<H, I, R> = (Summing<H, I, R, Kept>, HalfSpread<I, R>);

/// The slices of the rows a summed projection of `H` inputs into `I` outputs kept, in a cluster
/// that holds a half of `I`: an owner's run on the first slice of its parts.
type HalfRuns<H, I> = m![I % (I / 2) / ({ Owner<H, I, Kept> }), 1 # (H / ({ Width<H> }))];

/// The rows a summed projection of `H` inputs into `I` outputs kept, as the clusters of a
/// projection that halves its inputs name them.
type HalfKept<'k, H, I, R> = DmTensorView<
    'k,
    bf16,
    Chip,
    InputCluster<I>,
    HalfRuns<H, I>,
    m![R, I % ({ Owner<H, I, Kept> })],
>;

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
    /// in the passes, the owners and parts filling every slice; where the clusters halve the
    /// inputs, an owner's feature chunks are as many as its parts, each its share of every pass.
    const TILED: () = {
        let halves = if P::HALVES_INPUTS { 2 } else { 1 };
        let parts = I::SIZE / Chunk::<I>::SIZE / halves;
        let owners = outputs::<O, P>() / Owned::<I, O, P>::SIZE;
        let chunk = normalization::Chunk::<O>::SIZE;
        assert!(
            I::SIZE.is_multiple_of(halves * Chunk::<I>::SIZE)
                && Chunk::<I>::SIZE.is_multiple_of(128)
                && SLICES.is_multiple_of(parts)
                && outputs::<O, P>().is_multiple_of(Owned::<I, O, P>::SIZE)
                && O::SIZE.is_multiple_of(outputs::<O, P>())
                && TopSlots::<I, O, P>::SIZE * (owners / TopDigit::<I, O, P>::SIZE) * parts
                    == SLICES
                && Owned::<I, O, P>::SIZE.is_multiple_of(Height::<I, O, P>::SIZE)
                && Height::<I, O, P>::SIZE.is_multiple_of(4)
                && (!P::HALVES_INPUTS
                    || (Owned::<I, O, P>::SIZE == parts * chunk
                        && Height::<I, O, P>::SIZE.is_multiple_of(parts)
                        && chunk.is_multiple_of(Height::<I, O, P>::SIZE / parts)))
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
    /// group at a time, each in the slices that sum it, as `S` holds them.
    fn project<R: AxisName, X: Stage<I, O, R, P>, S: Sums<I, O, R, P>>(
        &self,
        device: &mut Device,
        input: &Live<X, R>,
    ) -> S
    where
        P::Sum: Rounding,
    {
        let global = scalar::register(device, &self.global);
        let mut projected = S::new();
        let (first, count) = input.groups::<Lanes<R>>();
        let rows = input.value();
        let dense = decoded::<R>(count);
        // The summed passes' last group folds its first four lanes alone when its live rows end
        // there: one group or none.
        let front = if const { Lanes::<R>::SIZE == 8 } {
            (1 - dense) * count.min(1) * ((11 - (input.start() + input.rows() + 7) % 8) / 8)
        } else {
            0
        };
        let whole = (1 - dense) * count - front;
        let last = first + count.max(1) - 1;
        let tiles = <Tiles<I, O, P> as M>::SIZE;
        let mut staged = X::staged();
        let alone = |_: &mut Device, _: usize, _: &mut Staging<I, O, R, P>| {};
        if const { X::PARTS == 0 || X::PARTS >= <Tiles<I, O, P> as M>::SIZE } {
            for _ in 0..whole.min(1) {
                let summed = |device: &mut Device, lanes: &Registers<I, O, Lane<R>, P>, at| {
                    self.pass::<R, Split>(device, (lanes, &global), at, projected.summed());
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
                    self.pass::<R, Trim>(
                        device,
                        (&lanes, &global),
                        (last, tile),
                        projected.summed(),
                    );
                }
            }
        } else {
            let summed = |device: &mut Device, lanes: &Registers<I, O, Lane<R>, P>, at| {
                self.pass::<R, Split>(device, (lanes, &global), at, projected.summed());
            };
            Self::meet(device, (rows, &mut staged), (first, whole), (summed, alone));
            for _ in 0..front {
                let mut staged = X::staged();
                rows.switch_front(device, last, &mut staged);
                let lanes = rows.front(device, last, &staged);
                for tile in 0..tiles {
                    self.pass::<R, Trim>(
                        device,
                        (&lanes, &global),
                        (last, tile),
                        projected.summed(),
                    );
                }
            }
        }
        for _ in 0..dense {
            rows.densely(device, |device, rows| {
                let weights = self.decode(device);
                let mut met = |device: &mut Device, lanes: &Switching<I, O, R, P>, group| {
                    for tile in 0..tiles {
                        projected.decoded(device, (lanes, (&weights, &global)), (group, tile));
                    }
                };
                // Pairs of groups, the second in the TRF's other half, so a switch runs beside the
                // passes before it; an even count's last pair has no group to switch beside its own.
                let mut held = rows.switched(device, first);
                for pair in 0..(count - 1) / 2 {
                    let group = first + 2 * pair;
                    let next = rows.switched(device, group + 1);
                    met(device, &held, group);
                    held = rows.switched(device, group + 2);
                    met(device, &next, group + 1);
                }
                for _ in 0..1 - count % 2 {
                    let next = rows.switched(device, last);
                    met(device, &held, last - 1);
                    met(device, &next, last);
                }
                for _ in 0..count % 2 {
                    met(device, &held, last);
                }
            });
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
        (lanes, global): (&Registers<I, O, F::Filled, P>, &Global<I, O, P>),
        (group, tile): (usize, usize),
        projected: &mut Summing<I, O, R, P>,
    ) where
        P::Sum: Rounding,
    {
        let scale = self.scaling(device, tile);
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
        <P::Sum as Rounding>::round::<
            _,
            _,
            _,
            _,
            F::Filled,
            m![{ <P::Sum as Rounding>::Packets<Elements<I, O, P>> }, 1, { F::Filled }],
            <P::Sum as Rounding>::Packet<O>,
            _,
        >(sums)
        .commit_view(F::sums(projected, (group, tile)));
    }

    /// The weights in BF16, each code times its block scale, a pass's rows at a time, each row's
    /// inputs written in [`Switched`] order.
    fn decode(&self, device: &mut Device) -> Decoded<I, O, P> {
        let mut weights = Decoded::<I, O, P>::new();
        for tile in 0..<Tiles<I, O, P> as M>::SIZE {
            let scale = self.scaling(device, tile);
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
                    { Elements<I, O, P> } #{!} ({ Owner<I, O, P> }), { Switched<I> }
                ]>(tile));
        }
        weights
    }

    /// The block scales of pass `tile`'s rows in the vector registers, widened to FP32 as they load.
    fn scaling(&self, device: &mut Device, tile: usize) -> Scaling<I, O, P> {
        device
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
            .to_vrf()
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
    ) -> DmTensorViewMut<'_, P::Sum, Chip, Cluster<I, O, P>, Reduced<I, O, P>, Self::Rows>;
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
    ) -> DmTensorViewMut<'_, P::Sum, Chip, Cluster<I, O, P>, Reduced<I, O, P>, Self::Rows> {
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
    ) -> DmTensorViewMut<'_, P::Sum, Chip, Cluster<I, O, P>, Reduced<I, O, P>, Self::Rows> {
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

/// What a lane group's pass over a decoded weight reads: the group in the TRF, the weights, which
/// carry their block scales, and the tensor scale.
type Operands<'a, I, O, R, P> = (
    &'a Switching<I, O, R, P>,
    (&'a Decoded<I, O, P>, &'a Global<I, O, P>),
);

/// A pass's sums of rows `T` over a decoded weight, every lane of a group, each divided by the
/// tensor scale: the stream the inter-slice reduce takes.
type Divided<'l, I, O, R, P, T> = VectorWidenTensor<
    'l,
    { Tu::Main },
    f32,
    Chip,
    Cluster<I, O, P>,
    Owners<I, O, P>,
    m![{ T }, 1],
    m![{ Lane<R> } # 8],
    Fresh,
    { stage::VeOrder::IntraFirst },
>;

/// What a projection's passes sum a launch's rows into: the summed passes' sums on the owners'
/// reduced slices, and the sums of the passes over a decoded weight where the placement lands them.
trait Sums<I: AxisName, O: AxisName, R: AxisName, P: Ownership> {
    fn new() -> Self;
    /// Where the summed passes write.
    fn summed(&mut self) -> &mut Summing<I, O, R, P>;
    /// Pass `tile` of lane group `group` over the decoded weights.
    fn decoded(
        &mut self,
        device: &mut Device,
        operands: Operands<'_, I, O, R, P>,
        at: (usize, usize),
    );

    /// Rows `T` of a lane group's pass `at` over the decoded weights, the tile `W` of each slice's
    /// rows at index `X`: the contraction adds a row's blocks, the vector engine divides each sum by
    /// the tensor scale.
    fn divided<'l, T: M, X: M, W: M>(
        device: &'l mut Device,
        (lanes, (weights, global)): Operands<'l, I, O, R, P>,
        at: usize,
    ) -> Divided<'l, I, O, R, P, T> {
        device
            .main
            .begin(weights.view().tile::<X, 1, W>(at))
            .fetch::<m![{ T }, { Packets<I> }], m![I % 16]>()
            .collect::<m![{ T }, { Packets<I> }], m![I % 16]>()
            .contract_outer::<m![{ T }, { Packets<I> }, 1], m![I % 16], _, _, _>(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ T }, 1]>()
            .contract_lane::<m![{ T }, 1], m![{ Lane<R> } # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ T }, 1, { Lane<R> } # 8 / 4], m![{ Lane<R> } # 8 % 4]>()
            .vector_fp_div(global)
            .vector_widen_concat::<m![{ T }, 1], m![{ Lane<R> } # 8]>()
    }
}

/// The decoded passes' sums where the summed passes leave theirs, an owner's on the first slice of
/// its parts.
impl<I: AxisName, O: AxisName, R: AxisName, P: Ownership> Sums<I, O, R, P> for Summing<I, O, R, P>
where
    P::Sum: Rounding,
{
    fn new() -> Self {
        DmTensor::new()
    }

    fn summed(&mut self) -> &mut Summing<I, O, R, P> {
        self
    }

    fn decoded(
        &mut self,
        device: &mut Device,
        operands: Operands<'_, I, O, R, P>,
        (group, tile): (usize, usize),
    ) {
        type Tile<I, O, P> = m![{ Elements<I, O, P> } # ({ Owner<I, O, P> }), { Switched<I> }];
        let sums = <Self as Sums<I, O, R, P>>::divided::<
            Elements<I, O, P>,
            Tiles<I, O, P>,
            Tile<I, O, P>,
        >(device, operands, tile)
        .vector_inter_slice_reduce::<Reduced<I, O, P>, m![{ Elements<I, O, P> }, 1]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final();
        <P::Sum as Rounding>::round::<
            _,
            _,
            _,
            _,
            Lane<R>,
            m![{ <P::Sum as Rounding>::Packets<Elements<I, O, P>> }, 1, { Lane<R> }],
            <P::Sum as Rounding>::Packet<O>,
            _,
        >(sums)
        .commit_view(<Split as Fold<I, O, R, P>>::sums(self, (group, tile)));
    }
}

/// The summed passes' sums on each owner's first part slice, the decoded passes' scattered over
/// its parts: the reduce lands each of the owner's chunks on its own feature slice.
impl<I: AxisName, O: AxisName, R: AxisName> Sums<I, O, R, Exchanged>
    for (Summing<I, O, R, Exchanged>, Scattered<I, O, R>)
{
    fn new() -> Self {
        (DmTensor::new(), DmTensor::new())
    }

    fn summed(&mut self) -> &mut Summing<I, O, R, Exchanged> {
        &mut self.0
    }

    fn decoded(
        &mut self,
        device: &mut Device,
        operands: Operands<'_, I, O, R, Exchanged>,
        (group, share): (usize, usize),
    ) {
        type Each<I, O> = Symbol<Share<I, O>>;
        type Tile<I, O> = m![
            O % ({ Owner<I, O, Exchanged> }) / ({ Feature<O> }),
            O % ({ Each<I, O> }) # ({ Feature<O> }),
            { Switched<I> }
        ];
        let sums =
            <Self as Sums<I, O, R, Exchanged>>::divided::<Shared<I, O>, Shares<I, O>, Tile<I, O>>(
                device, operands, share,
            )
            .vector_inter_slice_reduce::<Scatter<I, O>, m![O % ({ Each<I, O> }), 1]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final();
        <f32 as Rounding>::round::<
            _,
            _,
            _,
            _,
            Lane<R>,
            m![{ <f32 as Rounding>::Packets<m![O % ({ Each<I, O> })]> }, 1, { Lane<R> }],
            <f32 as Rounding>::Packet<O>,
            _,
        >(sums)
        .commit_view(
            (self.1.view_mut())
                .tile::<Layout<R>::Groups, 1, Layout<R>::GroupMut<m![O % ({ Feature<O> })]>>(group)
                .tile::<Shares<I, O>, 1, m![
                    { Layout<R>::SelectedMut },
                    { Lane<R> },
                    O % ({ Each<I, O> }) #{!} ({ Feature<O> })
                ]>(share),
        );
    }
}

/// How a placement has a projection's clusters and slices share its work: each cluster sums every
/// input into its half of the outputs, or its half of the inputs into every output; the owners hold
/// whole feature chunks of the outputs when the rows move to the feature slices, so the move takes
/// whole chunks, else as the passes split them.
pub trait Ownership: Clone + core::fmt::Debug + 'static {
    /// Whether the owners hold whole feature chunks of the outputs.
    const CHUNKED: bool;
    /// Whether each cluster sums its half of the inputs into every output, so that the clusters'
    /// partial sums meet in an exchange.
    const HALVES_INPUTS: bool;
    /// The cluster each part of the weight lies in.
    type Cluster<I: AxisName, O: AxisName>: M;
    /// The slice each owner's part of the inputs lies in.
    type Owners<I: AxisName, O: AxisName>: M;
    /// The slice an owner's sums land in once its parts meet.
    type Reduced<I: AxisName, O: AxisName>: M;
    /// What a pass stores its FP32 sums as.
    type Sum: MaterializableScalar;
}

impl Ownership for Features {
    const CHUNKED: bool = true;
    const HALVES_INPUTS: bool = false;
    type Cluster<I: AxisName, O: AxisName> = OutputCluster<I, O, Self>;
    type Owners<I: AxisName, O: AxisName> = OutputOwners<I, O, Self>;
    type Reduced<I: AxisName, O: AxisName> = OutputReduced<I, O, Self>;
    type Sum = bf16;
}

impl Ownership for Kept {
    const CHUNKED: bool = false;
    const HALVES_INPUTS: bool = false;
    type Cluster<I: AxisName, O: AxisName> = OutputCluster<I, O, Self>;
    type Owners<I: AxisName, O: AxisName> = OutputOwners<I, O, Self>;
    type Reduced<I: AxisName, O: AxisName> = OutputReduced<I, O, Self>;
    type Sum = bf16;
}

/// The FP32 partial sums of each cluster's half of the inputs; their exchange adds them in FP32 and
/// rounds each output once.
impl Ownership for Exchanged {
    const CHUNKED: bool = true;
    const HALVES_INPUTS: bool = true;
    type Cluster<I: AxisName, O: AxisName> = InputCluster<I>;
    type Owners<I: AxisName, O: AxisName> = InputOwners<I, O, Self>;
    type Reduced<I: AxisName, O: AxisName> = InputReduced<I, O, Self>;
    type Sum = f32;
}

/// Where a projection's rows end once its passes summed them in BF16, each cluster its half of the
/// outputs: the rows `place` gives of `R` rows projected from `I` inputs into `O` outputs.
pub trait Place<I: AxisName, O: AxisName, R: AxisName>: Ownership<Sum = bf16> + Sized {
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

/// The same weight's codes landed into the place a schedule carries, which they hold given back,
/// so a schedule can land them a step ahead of the rows they meet and carry them across other
/// work; its scales land when it is begun beside them.
impl<I: AxisName, O: AxisName, K, V, Y: Memory, P, S: M + 'static> Op<Begin<Landed<I, O, P>>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, Codes<I, O, P>>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Blocks<I, O, P>>>,
    Held<Y, S>: Borrow<Scale<S>>,
    P: Ownership,
{
    type Output = Landed<I, O, P>;

    fn forward(&self, device: &mut Device, Begin(place): Begin<Landed<I, O, P>>) -> Self::Output {
        const { Landed::<I, O, P>::TILED };
        // A tensor of their own, not a write into the codes before them, which are dead once read.
        let codes = self.weight.borrow().to_dm(&mut device.tdma);
        Landed { codes, ..place }
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

/// The same scales into a place made before, so a schedule can land them under other work.
impl<'l, I: AxisName, O: AxisName, K, V, Y: Memory, P, S: M + 'static>
    Op<&'l mut Landed<I, O, P, ()>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
where
    V: Borrow<HbmTensor<f8e4m3, Chip, Blocks<I, O, P>>>,
    Held<Y, S>: Borrow<Scale<S>>,
    P: Ownership,
{
    type Output = ();

    fn forward(&self, device: &mut Device, begun: &'l mut Landed<I, O, P, ()>) {
        let global: &Scale<S> = self.contract.global.borrow();
        (global.view()).to_dm_view(&mut device.tdma, begun.global.view_mut());
        (self.weight_scale.borrow().view()).to_dm_view(&mut device.tdma, begun.blocks.view_mut());
    }
}

/// The place a weight's scales land in beside its codes in theirs, before they land.
impl<I: AxisName, O: AxisName, P: Ownership> Default for Landed<I, O, P, ()> {
    fn default() -> Self {
        Self {
            global: DmTensor::new(),
            blocks: DmTensor::new(),
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
            let mut both = Both::<H, I, R>::new();
            transfer(device, input.with(&input.value().0), &mut both);
            self.forward(device, (landed, input.with(both)))
        } else {
            let mut spread = Spread::<I, R>::new();
            transfer(device, input.with(&input.value().0), &mut spread);
            self.forward(device, (landed, &input.with(spread)))
        }
    }
}

/// The same rows begun with the weight as it landed: one transfer lands them in both clusters as
/// they lie, so a schedule can issue it ahead of other transfers; given back, they project.
impl<H: AxisName, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, P, S: M + 'static>
    Op<Begin<(Landed<I, O, P>, Live<Kept<Summing<H, I, R, Kept>>, R>)>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, P, S>, Chip, K, V>
where
    P: Place<I, O, R>,
{
    type Output = (Landed<I, O, P>, Live<Both<H, I, R>, R>);

    fn forward(
        &self,
        device: &mut Device,
        Begin((landed, input)): Begin<(Landed<I, O, P>, Live<Kept<Summing<H, I, R, Kept>>, R>)>,
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

/// The live rows a summed projection of `H` inputs kept, through a weight whose clusters halve
/// its inputs: each cluster's passes read the half its cluster holds, where it lies.
impl<'x, H: AxisName, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, S: M + 'static>
    Op<&'x Live<Kept<Summing<H, I, R, Kept>>, R>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, Exchanged, S>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, Codes<I, O, Exchanged>>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Blocks<I, O, Exchanged>>>,
    Held<Y, S>: Borrow<Scale<S>>,
{
    type Output = Live<Spread<O, R>, R>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<Kept<Summing<H, I, R, Kept>>, R>,
    ) -> Self::Output {
        let landed = self.forward(&mut *device, Begin(()));
        let kept = &input.value().0;
        let spread = halve(&mut *device, input.with(kept));
        let partial = landed.project(
            &mut *device,
            &input.with(Halves {
                kept,
                spread: &spread,
            }),
        );
        exchange(device, input.with(partial))
    }
}

/// The same rows begun with the weight as it landed: they stay where they lie; from three live
/// lane groups one transfer a cluster, issued now so a schedule can issue later transfers behind
/// it, also lays its half in its feature slices for the passes over the decoded weight.
impl<H: AxisName, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, S: M + 'static>
    Op<
        Begin<(
            Landed<I, O, Exchanged>,
            Live<Kept<Summing<H, I, R, Kept>>, R>,
        )>,
    >
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, Exchanged, S>, Chip, K, V>
{
    type Output = (Landed<I, O, Exchanged>, Live<Halved<H, I, R>, R>);

    fn forward(
        &self,
        device: &mut Device,
        Begin((landed, input)): Begin<(
            Landed<I, O, Exchanged>,
            Live<Kept<Summing<H, I, R, Kept>>, R>,
        )>,
    ) -> Self::Output {
        let live = input.with(());
        let Kept(kept) = input.into_value();
        let spread = halve(device, live.with(&kept));
        (landed, live.with((kept, spread)))
    }
}

/// Rows kept in halves through the weight as it landed.
impl<H: AxisName, I: AxisName, O: AxisName, R: AxisName, K, V, Y: Memory, S: M + 'static>
    Op<(Landed<I, O, Exchanged>, Live<Halved<H, I, R>, R>)>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y, Exchanged, S>, Chip, K, V>
{
    type Output = Live<Spread<O, R>, R>;

    fn forward(
        &self,
        device: &mut Device,
        (landed, rows): (Landed<I, O, Exchanged>, Live<Halved<H, I, R>, R>),
    ) -> Self::Output {
        let (kept, spread) = rows.value();
        let partial = landed.project(&mut *device, &rows.with(Halves { kept, spread }));
        exchange(device, rows.with(partial))
    }
}

axes![Term = 2];

/// The FP32 partial sums a projection that halves its `I` inputs leaves in each cluster's reduced
/// slices, the cluster's index naming the term its half of the inputs gives.
type Partials<I, O, R> = DmTensor<
    f32,
    Chip,
    m![Term],
    Reduced<I, O, Exchanged>,
    m![R, O % ({ Owner<I, O, Exchanged> })],
>;

/// Both clusters' FP32 partial sums of every row in the output's feature slices of both clusters,
/// the term of each cluster's half of the inputs beside the other's.
type Terms<O, R> = DmTensor<
    f32,
    Chip,
    normalization::Cluster,
    FeatureSlices<O>,
    m![R, Term, O % ({ Feature<O> })],
>;

/// Rows of each feature chunk an owner holds that one pass over a decoded weight sums, when the
/// reduce scatters the owner's chunks over its parts: the pass's rows shared evenly among them.
#[derive(Clone, Debug)]
struct Share<I, O>(PhantomData<(I, O)>);

impl<I: AxisName, O: AxisName> AxisName for Share<I, O> {
    const NAME: Ident = Ident::new("Share");
    const SIZE: usize = Height::<I, O, Exchanged>::SIZE * normalization::Chunk::<O>::SIZE
        / Owned::<I, O, Exchanged>::SIZE;
}

/// Which share of each chunk a scattering pass sums.
type Shares<I, O> = m![O % ({ Feature<O> }) / ({ Symbol<Share<I, O>> })];

/// A scattering pass's rows: one share of each of an owner's feature chunks.
type Shared<I, O> = m![
    O % ({ Owner<I, O, Exchanged> }) / ({ Feature<O> }),
    O % ({ Symbol<Share<I, O>> })
];

/// The slices a projection that halves its inputs scatters an owner's sums over: the slices of
/// its parts, each the feature slice of one of its chunks.
type Scatter<I, O> = m![
    O / (O / ({ Top<I, O, Exchanged> })) # ({ Slots<I, O, Exchanged> }),
    O % (O / ({ Top<I, O, Exchanged> })) / ({ Owner<I, O, Exchanged> }),
    O % ({ Owner<I, O, Exchanged> }) / ({ Feature<O> })
];

/// A launch's FP32 partial sums over a cluster's half of the inputs, scattered over each owner's
/// parts.
type Scattered<I, O, R> =
    DmTensor<f32, Chip, InputCluster<I>, Scatter<I, O>, m![R, O % ({ Feature<O> })]>;

/// The live rows of `rows` into `into`'s mapping in one transfer, as a setup costs about as much as
/// all the rows: the one live row, else the lane group holding them, else the envelope.
fn transfer<V: Scalar, R: AxisName, C: M, S: M, X: M, D: M, T: M, Y: M>(
    device: &mut Device,
    rows: Live<&DmTensor<V, Chip, C, S, m![R, { X }]>, R>,
    into: &mut DmTensor<V, Chip, D, T, m![R, { Y }]>,
) {
    let (first, count) = rows.groups::<Lanes<R>>();
    if count > 1 {
        rows.value()
            .view()
            .to_dm_view(&mut device.tdma, into.view_mut());
    } else if rows.rows() == 1 {
        let row = rows.start();
        (rows.value().view())
            .tile::<m![R], 1, m![R = 1 # R, { X }]>(row)
            .to_dm_view(
                &mut device.tdma,
                (into.view_mut()).tile::<m![R], 1, m![R = 1 #{!} R, { Y }]>(row),
            );
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
    /// The rows in feature slices the passes over the decoded weights switch from.
    type Dense: Dense<I, O, R, P>;

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
    /// The rows, three lane groups or more of them live, in their feature slices for `dense`.
    fn densely(&self, device: &mut Device, dense: impl FnOnce(&mut Device, &Self::Dense));
}

/// Rows in feature slices whose lane groups switch straight into the TRF, in [`Switched`] order.
trait Dense<I: AxisName, O: AxisName, R: AxisName, P: Ownership> {
    fn switched(&self, device: &mut Device, group: usize) -> Switching<I, O, R, P>;
}

/// Rows in their feature slices: the switch ring takes each feature slice's inputs to the slices
/// that hold them, straight into the TRF where a slice holds one packet of a row.
impl<I: AxisName, O: AxisName, R: AxisName, P: Place<I, O, R>> Stage<I, O, R, P> for Spread<I, R> {
    const PARTS: usize = if I::SIZE <= 16 * 256 { 0 } else { 1 };
    type Dense = Self;

    fn staged() -> Staging<I, O, R, P> {
        DmTensor::new()
    }

    fn switch(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R, P>) {
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
                .commit_view(staged_front::<I, O, R, P>(staged));
        }
    }

    fn part(
        &self,
        device: &mut Device,
        (group, _): (usize, usize),
        staged: &mut Staging<I, O, R, P>,
    ) {
        <Self as Stage<I, O, R, P>>::switch(self, device, group, staged);
    }

    fn fill(
        &self,
        device: &mut Device,
        group: usize,
        staged: &Staging<I, O, R, P>,
    ) -> Registers<I, O, Lane<R>, P> {
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
            filled::<I, O, R, P>(device, staged)
        }
    }

    fn front(
        &self,
        device: &mut Device,
        group: usize,
        staged: &Staging<I, O, R, P>,
    ) -> Registers<I, O, Front<R>, P> {
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
            filled_front::<I, O, R, P>(device, staged)
        }
    }

    fn densely(&self, device: &mut Device, dense: impl FnOnce(&mut Device, &Self::Dense)) {
        dense(device, self);
    }
}

/// Rows a summed projection of `H` inputs kept, in both clusters: the switch ring takes each
/// owner's run, a piece at a time, to the slices whose part holds it, all pieces at once or one a
/// part, and the parts are written in DM before the lanes fill in input order.
impl<H: AxisName, I: AxisName, O: AxisName, R: AxisName, P: Place<I, O, R>> Stage<I, O, R, P>
    for Both<H, I, R>
{
    const PARTS: usize = Owned::<H, I, Kept>::SIZE / Piece::<H, I>::SIZE;
    type Dense = Spread<I, R>;

    fn staged() -> Staging<I, O, R, P> {
        DmTensor::new()
    }

    fn switch(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R, P>) {
        switch_runs::<H, I, O, R, P, Runs<H, I>>(
            device,
            both_runs::<H, I, O, R, P>(self, group),
            staged,
        );
    }

    fn switch_front(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R, P>) {
        switch_front_runs::<H, I, O, R, P, Runs<H, I>>(
            device,
            both_runs::<H, I, O, R, P>(self, group),
            staged,
        );
    }

    fn part(
        &self,
        device: &mut Device,
        (group, piece): (usize, usize),
        staged: &mut Staging<I, O, R, P>,
    ) {
        switch_piece::<H, I, O, R, P, Runs<H, I>>(
            device,
            (both_runs::<H, I, O, R, P>(self, group), piece),
            staged,
        );
    }

    fn fill(
        &self,
        device: &mut Device,
        _: usize,
        staged: &Staging<I, O, R, P>,
    ) -> Registers<I, O, Lane<R>, P> {
        filled::<I, O, R, P>(device, staged)
    }

    fn front(
        &self,
        device: &mut Device,
        _: usize,
        staged: &Staging<I, O, R, P>,
    ) -> Registers<I, O, Front<R>, P> {
        filled_front::<I, O, R, P>(device, staged)
    }

    fn densely(&self, device: &mut Device, dense: impl FnOnce(&mut Device, &Self::Dense)) {
        // Live rows spanning three groups or more move as the whole envelope, as in `transfer`.
        let mut spread = Spread::<I, R>::new();
        self.view().to_dm_view(&mut device.tdma, spread.view_mut());
        dense(device, &spread);
    }
}

/// [`Halved`] rows as the passes of a projection that halves its inputs read them: each cluster
/// switches its own half's runs, a piece at a time, to the slices whose part holds them, as rows in
/// both clusters switch, with no transfer first.
struct Halves<'k, H: AxisName, I: AxisName, R: AxisName> {
    kept: &'k Summing<H, I, R, Kept>,
    spread: &'k HalfSpread<I, R>,
}

impl<'k, H: AxisName, I: AxisName, R: AxisName> Halves<'k, H, I, R> {
    /// Lane group `group` of the kept rows, as the cluster whose passes read its half names it.
    fn runs<O: AxisName>(
        &self,
        group: usize,
    ) -> RunGroup<'k, H, I, O, R, Exchanged, HalfRuns<H, I>> {
        Layout::<R>::group(halved(self.kept), group)
    }
}

impl<H: AxisName, I: AxisName, O: AxisName, R: AxisName> Stage<I, O, R, Exchanged>
    for Halves<'_, H, I, R>
{
    const PARTS: usize = Owned::<H, I, Kept>::SIZE / Piece::<H, I>::SIZE;
    type Dense = HalfSpread<I, R>;

    fn staged() -> Staging<I, O, R, Exchanged> {
        DmTensor::new()
    }

    fn switch(&self, device: &mut Device, group: usize, staged: &mut Staging<I, O, R, Exchanged>) {
        type Slices<H, I> = HalfRuns<H, I>;
        switch_runs::<H, I, O, R, Exchanged, Slices<H, I>>(device, self.runs::<O>(group), staged);
    }

    fn switch_front(
        &self,
        device: &mut Device,
        group: usize,
        staged: &mut Staging<I, O, R, Exchanged>,
    ) {
        type Slices<H, I> = HalfRuns<H, I>;
        switch_front_runs::<H, I, O, R, Exchanged, Slices<H, I>>(
            device,
            self.runs::<O>(group),
            staged,
        );
    }

    fn part(
        &self,
        device: &mut Device,
        (group, piece): (usize, usize),
        staged: &mut Staging<I, O, R, Exchanged>,
    ) {
        type Slices<H, I> = HalfRuns<H, I>;
        let input = (self.runs::<O>(group), piece);
        switch_piece::<H, I, O, R, Exchanged, Slices<H, I>>(device, input, staged);
    }

    fn fill(
        &self,
        device: &mut Device,
        _: usize,
        staged: &Staging<I, O, R, Exchanged>,
    ) -> Registers<I, O, Lane<R>, Exchanged> {
        filled::<I, O, R, Exchanged>(device, staged)
    }

    fn front(
        &self,
        device: &mut Device,
        _: usize,
        staged: &Staging<I, O, R, Exchanged>,
    ) -> Registers<I, O, Front<R>, Exchanged> {
        filled_front::<I, O, R, Exchanged>(device, staged)
    }

    fn densely(&self, device: &mut Device, dense: impl FnOnce(&mut Device, &Self::Dense)) {
        dense(device, self.spread);
    }
}

/// A lane group of the rows a summed projection of `H` inputs into `I` outputs kept, an owner's
/// run a slice of slices `S`, in the cluster whose passes read them.
type RunGroup<'v, H, I, O, R, P, S> = DmTensorView<
    'v,
    bf16,
    Chip,
    Cluster<I, O, P>,
    S,
    Layout<R>::Elements<m![I % ({ Owner<H, I, Kept> })]>,
>;

/// Rows in both clusters' feature slices.
impl<I: AxisName, O: AxisName, R: AxisName, P: Place<I, O, R>> Dense<I, O, R, P> for Spread<I, R> {
    fn switched(&self, device: &mut Device, group: usize) -> Switching<I, O, R, P> {
        type Group<I, R> = Layout<R>::Elements<m![I % ({ Feature<I> })]>;
        // SAFETY: both physical clusters hold every row; only the logical cluster axis changes.
        let input = unsafe {
            Layout::<R>::group(self.view(), group)
                .reshape::<Chip, Cluster<I, O, P>, FeatureSlices<I>, Group<I, R>>()
        };
        device
            .sub
            .begin(input)
            .fetch::<m![{ Lane<R> }, I % ({ Feature<I> }) / 16], m![I % 16]>()
            .switch::<Owners<I, O, P>, m![{ Lane<R> }, { Packets<I> }]>(
                SwitchConfig::CustomBroadcast { ring_size: 256 },
            )
            .collect::<m![{ Lane<R> }, { Packets<I> }], m![I % 16]>()
            .to_trf()
    }
}

/// Rows in the feature slices of each cluster's half of the features, the half its passes read.
impl<I: AxisName, O: AxisName, R: AxisName> Dense<I, O, R, Exchanged> for HalfSpread<I, R> {
    fn switched(&self, device: &mut Device, group: usize) -> Switching<I, O, R, Exchanged> {
        device
            .sub
            .begin(Layout::<R>::group(self.view(), group))
            .fetch::<m![{ Lane<R> }, I % ({ Feature<I> }) / 16], m![I % 16]>()
            .switch::<Owners<I, O, Exchanged>, m![{ Lane<R> }, { Packets<I> }]>(
                SwitchConfig::CustomBroadcast { ring_size: 256 },
            )
            .collect::<m![{ Lane<R> }, { Packets<I> }], m![I % 16]>()
            .to_trf()
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

/// The outputs each cluster sums: every one when the clusters halve the inputs, else half of them.
const fn outputs<O: AxisName, P: Ownership>() -> usize {
    if P::HALVES_INPUTS {
        O::SIZE
    } else {
        O::SIZE / 2
    }
}

/// The kept rows `kept` holds, in the clusters whose passes read their halves.
fn halved<H: AxisName, I: AxisName, R: AxisName>(
    kept: &Summing<H, I, R, Kept>,
) -> HalfKept<'_, H, I, R> {
    const { assert!(halves::<H, I>()) };
    // SAFETY: the kept rows' cluster is the top digit of `I`, which names the halves the clusters
    // read; the slices drop only the unit top digit a single owner block leaves.
    unsafe { kept.view().reshape() }
}

/// Whether the rows a summed projection of `H` inputs kept lie in the halves of its `I` outputs a
/// projection that halves its inputs reads: each cluster one contiguous half, a part whole runs.
const fn halves<H: AxisName, I: AxisName>() -> bool {
    TopDigit::<H, I, Kept>::SIZE == 1
        && I::SIZE.is_multiple_of(2 * Chunk::<I>::SIZE)
        && Chunk::<I>::SIZE.is_multiple_of(Owned::<H, I, Kept>::SIZE)
}

/// Each cluster's half of the rows `kept` holds, laid in its half of the feature slices in one
/// transfer when three lane groups or more are live, as the passes over a decoded weight read them.
fn halve<H: AxisName, I: AxisName, R: AxisName>(
    device: &mut Device,
    kept: Live<&Summing<H, I, R, Kept>, R>,
) -> HalfSpread<I, R> {
    type Which<I> = Symbol<Section<I>>;
    type At<I> = Symbol<Within<I>>;
    let mut spread = HalfSpread::<I, R>::new();
    for _ in 0..decoded::<R>(kept.groups::<Lanes<R>>().1) {
        // SAFETY: both name the same cells in the same order: an input of a half is its part's
        // index times a part's width plus its place in the part.
        let (from, into) = unsafe {
            (
                halved(*kept.value()).reshape::<Chip, InputCluster<I>, m![
                    { Which<I> },
                    { At<I> } / ({ Owner<H, I, Kept> }),
                    1 # (H / ({ Width<H> }))
                ], m![R, { At<I> } % ({ Owner<H, I, Kept> })]>(
                ),
                spread.view_mut().reshape::<Chip, InputCluster<I>, m![
                    { Which<I> },
                    { At<I> } / ({ Feature<I> }) # ({ Symbol<Seats<I>> })
                ], m![R, { At<I> } % ({ Feature<I> })]>(),
            )
        };
        from.to_dm_view(&mut device.tdma, into);
    }
    spread
}

/// The FP32 partial sums of each cluster's half of the inputs `rows` holds, added in the output's
/// feature slices of both clusters, each output rounded once to BF16: the clusters swap the sums
/// the passes over a decoded weight scattered there, else one transfer gathers both there.
fn exchange<I: AxisName, O: AxisName, R: AxisName>(
    device: &mut Device,
    rows: Live<(Summing<I, O, R, Exchanged>, Scattered<I, O, R>), R>,
) -> Live<Spread<O, R>, R> {
    let live = rows.with(());
    let (summed, scattered) = rows.into_value();
    let mut sums = Spread::<O, R>::new();
    let (first, count) = live.groups::<Lanes<R>>();
    if decoded::<R>(count) > 0 {
        let other: Scattered<I, O, R> = scattered.view().cluster_swap().to_dm(&mut device.tdma);
        // SAFETY: both clusters add the same two terms into their own copy of every row, an
        // owner's chunks on their feature slices; only the logical cluster and slice axes change.
        let into = unsafe {
            (sums.view_mut()).reshape::<Chip, InputCluster<I>, Scatter<I, O>, m![
                R,
                O % ({ Feature<O> })
            ]>()
        };
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(scattered.view(), other.view())
            .fetch::<m![R, Term, O % ({ Feature<O> }) / 8], m![O % 8]>()
            .collect::<m![R, Term, O % ({ Feature<O> }) / 8], m![O % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, m![R, O % ({ Feature<O> }) / 8]>()
            .vector_clip_zip(ClipBinaryOpF32::Add)
            .vector_final()
            .cast::<bf16, m![O % 8 # 16]>()
            .commit_trim::<m![O % 8]>()
            .commit_view(into);
    } else {
        // SAFETY: each cluster summed its own half of the inputs, so its index names the term.
        let partial: Partials<I, O, R> = unsafe { summed.reshape() };
        let mut terms = Terms::<O, R>::new();
        transfer(device, live.with(&partial), &mut terms);
        if count > 1 {
            device
                .main
                .begin(terms.view())
                .fetch::<m![R, Term, O % ({ Feature<O> }) / 8], m![O % 8]>()
                .collect::<m![R, Term, O % ({ Feature<O> }) / 8], m![O % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<Term, m![R, O % ({ Feature<O> }) / 8]>()
                .vector_clip_zip(ClipBinaryOpF32::Add)
                .vector_final()
                .cast::<bf16, m![O % 8 # 16]>()
                .commit_trim::<m![O % 8]>()
                .commit_view(sums.view_mut());
        } else if live.rows() == 1 {
            let row = live.start();
            device
                .main
                .begin(
                    (terms.view()).tile::<m![R], 1, m![R = 1 # R, Term, O % ({ Feature<O> })]>(row),
                )
                .fetch::<m![Term, O % ({ Feature<O> }) / 8], m![O % 8]>()
                .collect::<m![Term, O % ({ Feature<O> }) / 8], m![O % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<Term, m![O % ({ Feature<O> }) / 8]>()
                .vector_clip_zip(ClipBinaryOpF32::Add)
                .vector_final()
                .cast::<bf16, m![O % 8 # 16]>()
                .commit_trim::<m![O % 8]>()
                .commit_view(
                    (sums.view_mut()).tile::<m![R], 1, m![R = 1 #{!} R, O % ({ Feature<O> })]>(row),
                );
        } else {
            device
                .main
                .begin(Layout::<R>::group(terms.view(), first))
                .fetch::<m![{ Lane<R> }, Term, O % ({ Feature<O> }) / 8], m![O % 8]>()
                .collect::<m![{ Lane<R> }, Term, O % ({ Feature<O> }) / 8], m![O % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<Term, m![{ Lane<R> }, O % ({ Feature<O> }) / 8]>()
                .vector_clip_zip(ClipBinaryOpF32::Add)
                .vector_final()
                .cast::<bf16, m![O % 8 # 16]>()
                .commit_trim::<m![O % 8]>()
                .commit_view(
                    (sums.view_mut())
                        .tile::<Layout<R>::Groups, 1, Layout<R>::GroupMut<m![O % ({ Feature<O> })]>>(
                            first,
                        ),
                );
        }
    }
    live.with(sums)
}

/// Whether `count` live lane groups of `R` rows meet a decoded weight: one from three groups,
/// counted by arithmetic (a device cannot cast a comparison to a count) in a form the compiler
/// bounds by one, so the loop over it frees what it read.
fn decoded<R: AxisName>(count: usize) -> usize {
    let groups = R::SIZE / Lanes::<R>::SIZE;
    if const { R::SIZE / Lanes::<R>::SIZE >= 3 } {
        (count + groups - 3) / groups
    } else {
        0
    }
}

/// Lane group `group` of rows in both clusters, as the cluster whose passes read them names it.
fn both_runs<H: AxisName, I: AxisName, O: AxisName, R: AxisName, P: Ownership>(
    rows: &Both<H, I, R>,
    group: usize,
) -> RunGroup<'_, H, I, O, R, P, Runs<H, I>> {
    const { assert!(fits_both::<H, I>()) };
    // SAFETY: both physical clusters hold every row; only the logical cluster axis changes.
    unsafe { Layout::<R>::group(rows.view(), group).reshape() }
}

/// A lane group's runs into the staging of the slices whose part holds each, a piece of every run
/// at a time.
fn switch_runs<H: AxisName, I: AxisName, O: AxisName, R: AxisName, P: Ownership, S: M>(
    device: &mut Device,
    input: RunGroup<'_, H, I, O, R, P, S>,
    staged: &mut Staging<I, O, R, P>,
) {
    type Run<H, I> = Symbol<Piece<H, I>>;
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

/// The first four lanes of a lane group's runs alone, as `switch_runs` switches every lane.
fn switch_front_runs<H: AxisName, I: AxisName, O: AxisName, R: AxisName, P: Ownership, S: M>(
    device: &mut Device,
    input: RunGroup<'_, H, I, O, R, P, S>,
    staged: &mut Staging<I, O, R, P>,
) {
    type Run<H, I> = Symbol<Piece<H, I>>;
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
        .commit_view(staged_front::<I, O, R, P>(staged));
}

/// Piece `piece` of every run of a lane group into its place in the staging.
fn switch_piece<H: AxisName, I: AxisName, O: AxisName, R: AxisName, P: Ownership, S: M>(
    device: &mut Device,
    (input, piece): (RunGroup<'_, H, I, O, R, P, S>, usize),
    staged: &mut Staging<I, O, R, P>,
) {
    type Run<H, I> = Symbol<Piece<H, I>>;
    type Pieces<H, I> = m![I % ({ Owner<H, I, Kept> }) / ({ Run<H, I> })];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::decoder::attention::gemma4::{Hidden, Intermediate, Rows};
    use crate::tensor::memory::Hbm;
    use crate::tensor::scalar::{Copies, Value};
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    /// Gemma 4's down projection under the summed contract, its clusters halving its inputs.
    type Down = BlockScaled<
        Intermediate,
        Hidden,
        f4e2m1,
        f8e4m3,
        Scales<Intermediate, Hidden>,
        Summed<W4A16, Hbm, Exchanged, Copies>,
        Chip,
        HbmTensor<f4e2m1, Chip, Codes<Intermediate, Hidden, Exchanged>>,
        HbmTensor<f8e4m3, Chip, Blocks<Intermediate, Hidden, Exchanged>>,
    >;

    /// The live rows `start` and `rows` name, landed from HBM rows where Gemma 4's gate and up
    /// projections keep their outputs, through `down`, back to HBM rows.
    #[device(chip = 1)]
    fn projected(
        device: &mut Device,
        down: &Down,
        input: &HbmTensor<bf16, Chip, m![Rows, Intermediate]>,
        (start, rows): (&Value, &Value),
    ) -> HbmTensor<bf16, Chip, m![Rows, Hidden]> {
        let live =
            Live::<(), Rows>::all().subrange(start.extent::<m![Rows]>(), rows.extent::<m![Rows]>());
        let kept: Summing<Hidden, Intermediate, Rows, Kept> = input.to_dm(&mut device.tdma);
        let output = Op::forward(down, &mut *device, &live.with(Kept(kept)));
        output.value().to_hbm(&mut device.tdma)
    }

    /// Uniform values in `[0, 1)` from a xorshift state.
    fn uniform(count: usize, mut state: u64) -> impl Iterator<Item = f64> {
        (0..count).map(move |_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
    }

    /// Each live row's outputs within one BF16 rounding of its FP64 product plus `2^-9` of its
    /// outputs' RMS, the FP32 sums' error, at one row, a lane group, a group and a row, three
    /// groups and the envelope: the exchange rounds each output once, as the other placements do.
    #[tokio::test(flavor = "current_thread")]
    async fn exchanges_rows_within_one_rounding() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        let (i, o, r) = (Intermediate::SIZE, Hidden::SIZE, Rows::SIZE);
        let codes: Vec<f4e2m1> = uniform(o * i, 4)
            .map(|u| f4e2m1::from_bits((u * 16.0) as u8))
            .collect();
        let scales: Vec<f8e4m3> = uniform(o * i / 16, 5)
            .map(|u| f8e4m3::from_f32((8.0 + 440.0 * u) as f32))
            .collect();
        let global = 2f32.powi(14) * 1.37;
        let row: Vec<bf16> = uniform(r * i, 6)
            .map(|u| bf16::from_f32((4.0 * u - 2.0) as f32))
            .collect();
        let exact: Vec<f64> = (0..r * o)
            .map(|at| {
                let (line, output) = (at / o, at % o);
                (0..i)
                    .map(|input| {
                        let weight = f64::from(codes[output * i + input].to_f32())
                            * f64::from(scales[(output * i + input) / 16].to_f32())
                            / f64::from(global);
                        weight * f64::from(f32::from(row[line * i + input]))
                    })
                    .sum()
            })
            .collect();
        let pdma = &mut device.pdma;
        let down: Down = BlockScaled {
            weight: HostTensor::<f4e2m1, m![Hidden, Intermediate]>::from_vec(codes)
                .to_hbm(pdma)
                .await
                .unwrap(),
            weight_scale: HostTensor::<f8e4m3, Scales<Intermediate, Hidden>>::from_vec(scales)
                .to_hbm(pdma)
                .await
                .unwrap(),
            contract: Summed {
                input: W4A16,
                global: HostTensor::<f32, m![1 # 8]>::from_vec([
                    global, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ])
                .to_hbm(pdma)
                .await
                .unwrap(),
                place: PhantomData,
            },
            shape: PhantomData,
        };
        let input = HostTensor::<bf16, m![Rows, Intermediate]>::from_vec(row)
            .to_hbm(pdma)
            .await
            .unwrap();
        let count = |value: usize| HostTensor::<i32, m![1]>::from_vec([value as i32]);
        for (start, live) in [(0, 1), (0, 8), (3, 6), (0, 9), (7, 17), (0, 64)] {
            let first = count(start).to_hbm(&mut device.pdma).await.unwrap();
            let rows = count(live).to_hbm(&mut device.pdma).await.unwrap();
            let output = launch(projected, (&mut device, &down, &input, (&first, &rows)))
                .await
                .unwrap()
                .to_host::<m![Rows, Hidden]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let mut worst = 0f64;
            for line in start..start + live {
                let want = &exact[line * o..(line + 1) * o];
                let rms = (want.iter().map(|e| e * e).sum::<f64>() / o as f64).sqrt();
                for (got, want) in output[line * o..(line + 1) * o].iter().zip(want) {
                    let error = (f64::from(f32::from(*got)) - want).abs();
                    worst = worst.max(error / (want.abs() / 256.0 + rms / 512.0));
                }
            }
            println!("rows {start}+{live}: worst {worst:.3} of the bound");
            assert!(
                worst <= 1.0,
                "rows {start}+{live}: an output is {worst} of the bound away"
            );
        }
    }
}
