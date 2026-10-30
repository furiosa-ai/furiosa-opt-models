//! Indexed reductions over explicitly supplied device operands.

use crate::Op;
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::ops::linear::Groups;
use crate::ops::linear::stream::{self, Block, Column, Columns, Logits, PassRows, ToDm};
use crate::ops::pointwise;
use crate::tensor::mapping::Rows as Layout;
use crate::tensor::{axis::Single, mapping::BroadcastClusters as Cluster, parallel::Chip, scalar};
use core::borrow::Borrow;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::HostBuffer;

/// Max-reduce nonnegative weights into zero-initialized output bins.
/// An index of -1 excludes that row; other indices must be in 0..Bins::SIZE.
#[derive(DeviceSend)]
pub struct ScatterMax<Bins>(pub PhantomData<Bins>);

impl<Bins: AxisName, Rows: AxisName, Chips: M + 'static>
    Op<(
        &HbmTensor<f32, Chips, m![Rows, Single]>,
        &HbmTensor<i32, Chips, m![Rows]>,
    )> for ScatterMax<Bins>
{
    type Output = HbmTensor<f32, Chips, m![Single, Bins]>;

    fn forward(
        &self,
        _: &mut Device,
        _: (
            &HbmTensor<f32, Chips, m![Rows, Single]>,
            &HbmTensor<i32, Chips, m![Rows]>,
        ),
    ) -> Self::Output {
        todo!("NPU max reduction by output-bin index, excluding -1 rows")
    }
}

impl<Bins: AxisName, Rows: AxisName, Chips: M + 'static>
    Op<&(
        HbmTensor<f32, Chips, m![Rows, Single]>,
        HbmTensor<i32, Chips, m![Rows]>,
    )> for ScatterMax<Bins>
{
    type Output = HbmTensor<f32, Chips, m![Single, Bins]>;

    fn forward(
        &self,
        device: &mut Device,
        (weights, indices): &(
            HbmTensor<f32, Chips, m![Rows, Single]>,
            HbmTensor<i32, Chips, m![Rows]>,
        ),
    ) -> Self::Output {
        self.forward(device, (weights, indices))
    }
}

type Segment<V> = Symbol<Part<V>>;
type Window<V> = Symbol<Extent<V>>;
type Slices<V> = m![V % ({ Window<V> }) / ({ Segment<V> }) # 256];
pub type Elements<V> = m![V / ({ Window<V> }), V % ({ Segment<V> })];
type Time<V> = m![V / ({ Window<V> }), V % ({ Segment<V> }) / 8];
type Segments<V> = m![V / ({ Window<V> }), V % ({ Segment<V> }) / 4];
type Rows<D, V, Chips = Chip> = DmTensor<D, Chips, Cluster, Slices<V>, Elements<V>>;
type Reduced<D, Chips = Chip> = DmTensor<D, Chips, Cluster, m![1 # 256], m![1 # 8]>;

/// First maximum of a finite vocabulary vector, or of each row's candidates over a projection's
/// passes; ties choose the lowest index. The index tensor is bound once.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Argmax<V: AxisName, Chips: M = Chip, Indices = HbmTensor<i32, Chips, m![V]>> {
    /// Reversed token IDs V::SIZE - 1 - token; larger ranks choose the first tied index.
    indices: Indices,
    vocabulary: PhantomData<fn() -> (V, Chips)>,
}

impl<V: AxisName, Chips: M, Indices> Argmax<V, Chips, Indices> {
    const fn new(indices: Indices) -> Self {
        Self {
            indices,
            vocabulary: PhantomData,
        }
    }
}

impl<V: AxisName, Indices: Borrow<HbmTensor<i32, Chip, m![V]>>> Argmax<V, Chip, Indices> {
    /// This argmax with its ranks in data memory, as columns a projection's passes read.
    pub(crate) fn to_dm<Hidden: AxisName>(
        &self,
        device: &mut Device,
    ) -> Argmax<V, Chip, Columns<Hidden, V, i32>> {
        Argmax::new(ToDm::<Hidden>(PhantomData).forward(device, self.indices.borrow()))
    }
}

/// The ranks computed on host. By hand, with the update below: a derived `with` field would
/// rebuild them in every update.
impl<'a, V: AxisName, Chips: M, Indices> Load<&'a Checkpoint> for Argmax<V, Chips, Indices>
where
    Indices: Load<HostTensor<i32, m![{ Chips }, { m![V] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("indices", "indices", |cx| ranks::<V, Chips, _>(from, cx))
                .map(Self::new)
        })
    }
}

/// The ranks are no update target.
impl<'t, 's, V: AxisName, Chips: M, Indices>
    Load<(&'t mut Argmax<V, Chips, Indices>, &'s Checkpoint)>
    for &'t mut Argmax<V, Chips, Indices>
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Argmax<V, Chips, Indices>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |_| Ok(()))?;
        Ok(into)
    }
}

impl<V: AxisName, Chip: M> Op<&HbmTensor<bf16, Chip, m![V]>> for Argmax<V, Chip> {
    type Output = HbmTensor<i32, Chip, m![1 # 4]>;

    fn forward(&self, device: &mut Device, logits: &HbmTensor<bf16, Chip, m![V]>) -> Self::Output {
        let logits: Rows<bf16, V, Chip> = logits.to_dm(&mut device.tdma);
        let maximum: Reduced<f32, Chip> = device
            .main
            .begin(logits.view())
            .fetch::<Time<V>, m![V % 8]>()
            .fetch_cast::<f32>()
            .collect::<Time<V>, m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Segments<V>, m![V % 4]>()
            .vector_intra_slice_reduce::<V, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<m![1 # 256], m![1]>(InterSliceReduceOpF32::Max)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        // Every reduced slice holds the same scalar; restore the original slice labels.
        let maximum: DmTensor<f32, Chip, Cluster, Slices<V>, m![1 # 8]> =
            unsafe { maximum.reshape() };
        let maximum = scalar::register(device, &maximum);
        let shifted: Rows<f32, V, Chip> = device
            .main
            .begin(logits.view())
            .fetch::<Time<V>, m![V % 8]>()
            .fetch_cast::<f32>()
            .collect::<Time<V>, m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Segments<V>, m![V % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &maximum)
            .vector_widen_concat::<Time<V>, m![V % 8]>()
            // Canonicalize -0 so equal signed zeros obey the same first-index rule.
            .vector_clip(ClipBinaryOpF32::Add, 0.0_f32)
            .vector_final()
            .commit_trim::<m![V % 8]>()
            .commit();
        let indices: Rows<i32, V, Chip> = self.indices.to_dm(&mut device.tdma);
        let indices: VrfTensor<i32, Chip, Cluster, Slices<V>, Elements<V>> = device
            .sub
            .begin(indices.view())
            .fetch::<Time<V>, m![V % 8]>()
            .collect::<Time<V>, m![V % 8]>()
            .to_vrf();
        // Zero differences rank by reverse index. Negative differences remain negative even
        // after adding the largest index, including a subtraction that overflows to -infinity.
        let ranked: Rows<i32, V, Chip> = device
            .main
            .begin(shifted.view())
            .fetch::<Time<V>, m![V % 8]>()
            .collect::<Time<V>, m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>()
            .vector_fxp(FxpBinaryOp::AddFxpSat, &indices)
            .vector_final()
            .commit_trim::<m![V % 8]>()
            .commit();
        let rank: Reduced<i32, Chip> = device
            .main
            .begin(ranked.view())
            .fetch::<Time<V>, m![V % 8]>()
            .collect::<Time<V>, m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Segments<V>, m![V % 4]>()
            .vector_intra_slice_reduce::<V, m![1], m![1 # 4]>(IntraSliceReduceOpI32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<m![1 # 256], m![1]>(InterSliceReduceOpI32::Max)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let token: DmTensor<i32, Chip, Cluster, m![1 # 256], m![1 # 4]> = device
            .main
            .begin(rank.view())
            .fetch::<m![1], m![1 # 8]>()
            .collect::<m![1], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp_with_mode(
                FxpBinaryOp::SubFxp,
                BinaryArgMode::Mode10,
                const { V::SIZE as i32 - 1 },
            )
            .vector_final()
            .commit_trim::<m![1 # 4]>()
            .commit();
        token.to_hbm(&mut device.tdma)
    }
}

/// Each row's token in one slice of the first cluster, as the candidates' selection leaves it.
pub type Selected<Rows, Chips = Chip> =
    DmTensor<i32, Chips, m![1 # 2], m![1 # 256], m![Rows # ({ Symbol<Selection<Rows>> })]>;

/// A row's token cells: several rows keep each group's eight token lanes, the cells the selection
/// leaves its tokens in, so it returns those cells in place.
#[derive(Debug, Clone)]
pub struct Selection<Rows>(PhantomData<Rows>);

impl<Rows: AxisName> AxisName for Selection<Rows> {
    const NAME: Ident = Rows::NAME;

    const SIZE: usize = if Rows::SIZE < 2 {
        2
    } else {
        <Groups<Rows> as M>::SIZE * 8
    };
}

type P<Hidden, V> = m![V % ({ PassRows<Hidden, V> })];
type Index<Hidden, V> = m![V % ({ Block<Hidden, V> }) / ({ PassRows<Hidden, V> })];
type Lanes<Rows> = m![{ Layout<Rows>::Lanes } # 8];
type Half<Rows> = m![{ Layout<Rows>::Lanes } # 8 % 4];
type Times<Rows> = m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 8 / 4];
type Batch<Rows> = m![{ Groups<Rows> } % 8];

/// Takes every logit difference of at least 2^-100 in magnitude past -2^26, where the exponential
/// is 0: a smaller nonzero difference needs logits below 2^-76.
const SCALE: f32 = 1.0e38;
type Batches<Rows> = m![{ Groups<Rows> } / 8];

/// Each row's `Count` summaries in every slice of `Slices`.
type Summaries<Chips, D, V, Count, Slices, Rows> = DmTensor<
    D,
    Chips,
    stream::Cluster<V>,
    Slices,
    m![{ Count }, { Groups<Rows> }, { Lanes<Rows> }],
>;
/// Each row's one summary in every slice of `Slices`.
type Summary<Chips, D, V, Slices, Rows> =
    DmTensor<D, Chips, stream::Cluster<V>, Slices, m![{ Groups<Rows> }, { Lanes<Rows> }]>;
type Stored<Chips, D, Hidden, V, Rows> =
    Summaries<Chips, D, V, Index<Hidden, V>, stream::Reduced<Hidden, V>, Rows>;
/// One cluster's vocabulary groups.
type Blocks<Hidden, V> = m![V % (V / 2) / ({ Block<Hidden, V> })];

/// Every group's row summaries, in one slice of each cluster.
type Gathered<Chips, D, Hidden, V, Rows> =
    Summaries<Chips, D, V, Blocks<Hidden, V>, m![1 # 256], Rows>;
type Global<Chips, D, V, Rows> = Summary<Chips, D, V, m![1 # 256], Rows>;

/// Each row's maximum and the rank of its first token in every pass of a vocabulary projection, as
/// the projection's passes push them; the argmax that made them selects among them.
pub struct Candidates<'a, Hidden: AxisName, V: AxisName, Rows: AxisName, Chips: M + 'static = Chip>
{
    indices: &'a Columns<Hidden, V, i32, Chips>,
    maximum: Stored<Chips, f32, Hidden, V, Rows>,
    rank: Stored<Chips, i32, Hidden, V, Rows>,
}

impl<Hidden: AxisName, V: AxisName, Rows: AxisName, Chips: M + 'static>
    Candidates<'_, Hidden, V, Rows, Chips>
{
    /// Pass `at`'s maxima and the ranks of their first tokens, kept in the pass's slot.
    pub(crate) fn push(
        &mut self,
        context: &mut Device,
        logits: &Logits<Hidden, V, Rows, Chips>,
        at: usize,
    ) {
        let candidates = self;
        let iota = candidates
            .indices
            .view()
            .tile::<Index<Hidden, V>, 1, Column<Hidden, V>>(at);
        // Two lane halves remain in the reduction, of an eight-slot accumulator; the tensor unit
        // commits both summaries to their pass's slots, leaving DMA to the weight stream.
        Maximum::<V, P<Hidden, V>, Rows>::of(&mut context.main, logits.view())
            .commit_view(candidates.maximum.view_mut().tile::<Index<Hidden, V>, 1,
                m![1 #{!} ({ Index<Hidden, V> }), { Groups<Rows> }, { Lanes<Rows> }],
            >(at));
        let maximum: VrfTensor<
            f32, Chips, stream::Cluster<V>, stream::Reduced<Hidden, V>,
            m![{ Groups<Rows> }, { Lanes<Rows> }],
        > = context.sub.begin(candidates.maximum.view().tile::<Index<Hidden, V>, 1,
            m![1 # ({ Index<Hidden, V> }), { Groups<Rows> }, { Lanes<Rows> }],
        >(at))
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .to_vrf();
        // The token numbers as floats, the factor of a row's maxima below.
        let numbers: DmTensor<
            f32,
            Chips,
            stream::Cluster<V>,
            stream::Reduced<Hidden, V>,
            P<Hidden, V>,
        > = context
            .main
            .begin(iota)
            .fetch::<m![{ P<Hidden, V> } / 8], m![V % 8]>()
            .collect::<m![{ P<Hidden, V> } / 8], m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp_to_fp(31)
            .vector_final()
            .commit_trim::<m![V % 8]>()
            .commit();
        let numbers: VrfTensor<
            f32,
            Chips,
            stream::Cluster<V>,
            stream::Reduced<Hidden, V>,
            P<Hidden, V>,
        > = context
            .sub
            .begin(numbers.view())
            .fetch::<m![{ P<Hidden, V> } / 8], m![V % 8]>()
            .collect::<m![{ P<Hidden, V> } / 8], m![V % 8]>()
            .to_vrf();
        // Only a maximum's difference is zero, the scale taking the rest past -2^26 or to -inf: exp
        // is 1 at the maxima, 0 elsewhere, and its largest token-number product is the first's.
        context.main.begin(logits.view())
            .fetch::<m![{ Groups<Rows> }, { P<Hidden, V> }], Lanes<Rows>>()
            .collect::<m![{ Groups<Rows> }, { P<Hidden, V> }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![{ Groups<Rows> }, { P<Hidden, V> }, { Layout<Rows>::Lanes } # 8 / 4],
                Half<Rows>,
            >()
            .vector_fp_binary(FpBinaryOp::SubF, &maximum)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), SCALE)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &numbers)
            .vector_intra_slice_reduce::<V, Times<Rows>, Half<Rows>>(IntraSliceReduceOpF32::Max)
            .vector_widen_concat::<Groups<Rows>, Lanes<Rows>>()
            .vector_fp_to_fxp(31)
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit_view(candidates.rank.view_mut().tile::<Index<Hidden, V>, 1,
                m![1 #{!} ({ Index<Hidden, V> }), { Groups<Rows> }, { Lanes<Rows> }],
            >(at));
    }
}

impl<V: AxisName, Hidden: AxisName, Chips: M + 'static>
    Argmax<V, Chips, Columns<Hidden, V, i32, Chips>>
{
    /// Empty candidates of `Rows` rows, for a projection's passes to push.
    pub fn candidates<Rows: AxisName>(&self) -> Candidates<'_, Hidden, V, Rows, Chips> {
        Candidates {
            indices: &self.indices,
            maximum: DmTensor::new(),
            rank: DmTensor::new(),
        }
    }
}

/// Each row's first maximum over every pass's candidates.
impl<Hidden: AxisName, V: AxisName, Rows: AxisName, Chips: M + 'static>
    Op<Candidates<'_, Hidden, V, Rows, Chips>>
    for Argmax<V, Chips, Columns<Hidden, V, i32, Chips>>
{
    type Output = Selected<Rows, Chips>;

    fn forward(
        &self,
        context: &mut Device,
        candidates: Candidates<'_, Hidden, V, Rows, Chips>,
    ) -> Self::Output {
        // Tensor-unit operations run on one live slice per vocabulary group, where each group
        // settles its maximum and rank; tensor DMA gathers the summaries to fold per cluster.
        let local: Summary<Chips, f32, V, stream::Reduced<Hidden, V>, Rows> =
            Maximum::<V, Index<Hidden, V>, Rows>::of(&mut context.main, candidates.maximum.view())
                .commit();
        let rank = Ranks::<V, Index<Hidden, V>, stream::Reduced<Hidden, V>, Rows>(PhantomData)
            .forward(
                &mut *context,
                (&candidates.rank, &candidates.maximum, &local),
            );
        // The groups' maxima and ranks move to one slice of each cluster.
        let mut maxima: Gathered<Chips, f32, Hidden, V, Rows> = DmTensor::new();
        local
            .view()
            .to_dm_view(&mut context.tdma, maxima.view_mut());
        let mut ranks: Gathered<Chips, i32, Hidden, V, Rows> = DmTensor::new();
        rank.view().to_dm_view(&mut context.tdma, ranks.view_mut());
        let maximum: Global<Chips, f32, V, Rows> =
            Maximum::<V, Blocks<Hidden, V>, Rows>::of(&mut context.main, maxima.view()).commit();
        let other: Global<Chips, f32, V, Rows> =
            maximum.view().cluster_swap().to_dm(&mut context.tdma);
        let other: VrfTensor<
            f32,
            Chips,
            stream::Cluster<V>,
            m![1 # 256],
            m![{ Groups<Rows> }, { Lanes<Rows> }],
        > = context
            .sub
            .begin(other.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .to_vrf();
        let maximum: Global<Chips, f32, V, Rows> = context
            .main
            .begin(maximum.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Max, &other)
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit();
        let rank = Ranks::<V, Blocks<Hidden, V>, m![1 # 256], Rows>(PhantomData)
            .forward(&mut *context, (&ranks, &maxima, &maximum));
        let other: Global<Chips, i32, V, Rows> =
            rank.view().cluster_swap().to_dm(&mut context.tdma);
        let other: VrfTensor<
            i32,
            Chips,
            stream::Cluster<V>,
            m![1 # 256],
            m![{ Groups<Rows> }, { Lanes<Rows> }],
        > = context
            .sub
            .begin(other.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .to_vrf();
        let rank: Global<Chips, i32, V, Rows> = context
            .main
            .begin(rank.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpI32::Max, &other)
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit();
        let token: Global<Chips, i32, V, Rows> = context
            .main
            .begin(rank.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp_with_mode(
                FxpBinaryOp::SubFxp,
                BinaryArgMode::Mode10,
                V::SIZE as i32 - 1,
            )
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit();
        Chosen::<V, Rows>(PhantomData).forward(context, token)
    }
}

/// Each row's token as the head returns it: one row moves into its padded slot, more rows already
/// sit in the cells the head returns.
struct Chosen<V, Rows>(PhantomData<(V, Rows)>);

impl<V: AxisName, Rows: AxisName, Chips: M + 'static> Op<Global<Chips, i32, V, Rows>>
    for Chosen<V, Rows>
{
    type Output = Selected<Rows, Chips>;

    fn forward(&self, context: &mut Device, token: Global<Chips, i32, V, Rows>) -> Self::Output {
        if const { Rows::SIZE == 1 } {
            // SAFETY: both clusters hold the same tokens after the symmetric swap.
            let token: DmTensor<
                i32,
                Chips,
                m![1 # 2],
                m![1 # 256],
                m![{ Groups<Rows> }, { Lanes<Rows> }],
            > = unsafe { token.reshape() };
            let mut output = Selected::<Rows, Chips>::new();
            token
                .view()
                .to_dm_view(&mut context.tdma, output.view_mut());
            output
        } else {
            // SAFETY: both clusters hold the same tokens after the symmetric swap.
            unsafe { token.reshape() }
        }
    }
}

/// Each row's maximum over its `Count` summaries in each slice, ready to commit.
struct Maximum<V, Count, Rows>(PhantomData<(V, Count, Rows)>);

impl<V: AxisName, Count: M, Rows: AxisName> Maximum<V, Count, Rows> {
    fn of<'l, Slices: M, E: M, Chips: M + 'static>(
        main: &'l mut TuContext<{ Tu::Main }>,
        summaries: DmTensorView<'l, f32, Chips, stream::Cluster<V>, Slices, E>,
    ) -> CommitTrimTensor<
        'l,
        { Tu::Main },
        f32,
        Chips,
        stream::Cluster<V>,
        Slices,
        Groups<Rows>,
        Lanes<Rows>,
    > {
        main.begin(summaries)
            .fetch::<m![{ Groups<Rows> }, { Count }], Lanes<Rows>>()
            .collect::<m![{ Groups<Rows> }, { Count }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![{ Groups<Rows> }, { Count }, { Layout<Rows>::Lanes } # 8 / 4],
                Half<Rows>,
            >()
            .vector_intra_slice_reduce::<V, Times<Rows>, Half<Rows>>(IntraSliceReduceOpF32::Max)
            .vector_widen_concat::<Groups<Rows>, Lanes<Rows>>()
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
    }
}

/// Each row's largest rank among its `Count` summaries that tie its maximum, in every slice of
/// `Slices`.
struct Ranks<V, Count, Slices, Rows>(PhantomData<(V, Count, Slices, Rows)>);

/// The ranks and maxima of each row's summaries, and the row's maximum.
type Tied<'a, Chips, V, Count, Slices, Rows> = (
    &'a Summaries<Chips, i32, V, Count, Slices, Rows>,
    &'a Summaries<Chips, f32, V, Count, Slices, Rows>,
    &'a Summary<Chips, f32, V, Slices, Rows>,
);

impl<'a, V: AxisName, Count: M, Slices: M, Rows: AxisName, Chips: M + 'static>
    Op<Tied<'a, Chips, V, Count, Slices, Rows>> for Ranks<V, Count, Slices, Rows>
{
    type Output = Summary<Chips, i32, V, Slices, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        (ranks, maxima, maximum): Tied<'a, Chips, V, Count, Slices, Rows>,
    ) -> Self::Output {
        let maximum: VrfTensor<
            f32,
            Chips,
            stream::Cluster<V>,
            Slices,
            m![{ Groups<Rows> }, { Lanes<Rows> }],
        > = context
            .sub
            .begin(maximum.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .to_vrf();
        let shifted: Summaries<Chips, f32, V, Count, Slices, Rows> = context
            .main
            .begin(maxima.view())
            .fetch::<m![{ Count }, { Groups<Rows> }], Lanes<Rows>>()
            .collect::<m![{ Count }, { Groups<Rows> }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Count }, { Times<Rows> }], Half<Rows>>()
            .vector_fp_binary(FpBinaryOp::SubF, &maximum)
            .vector_widen_concat::<m![{ Count }, { Groups<Rows> }], Lanes<Rows>>()
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit();
        let mut rank: Summary<Chips, i32, V, Slices, Rows> = DmTensor::new();
        if const { Rows::SIZE <= 64 } {
            let ranks: VrfTensor<
                i32,
                Chips,
                stream::Cluster<V>,
                Slices,
                m![{ Groups<Rows> }, { Count }, { Lanes<Rows> }],
            > = context
                .sub
                .begin(ranks.view())
                .fetch::<m![{ Groups<Rows> }, { Count }], Lanes<Rows>>()
                .collect::<m![{ Groups<Rows> }, { Count }], Lanes<Rows>>()
                .to_vrf();
            context.main.begin(shifted.view())
                .fetch::<m![{ Groups<Rows> }, { Count }], Lanes<Rows>>()
                .collect::<m![{ Groups<Rows> }, { Count }], Lanes<Rows>>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_reinterpret::<i32>()
                .vector_fxp(FxpBinaryOp::AddFxpSat, &ranks)
                .vector_narrow_split::<
                    m![{ Groups<Rows> }, { Count }, { Layout<Rows>::Lanes } # 8 / 4],
                    Half<Rows>,
                >()
                .vector_intra_slice_reduce::<V, Times<Rows>, Half<Rows>>(IntraSliceReduceOpI32::Max)
                .vector_widen_concat::<Groups<Rows>, Lanes<Rows>>()
                .vector_final()
                .commit_trim::<Lanes<Rows>>()
                .commit_view(rank.view_mut());
        } else {
            #[unroll]
            for batch in 0..<Batches<Rows> as M>::SIZE {
                let ranks: VrfTensor<i32, Chips, stream::Cluster<V>, Slices,
                    m![{ Batch<Rows> }, { Count }, { Lanes<Rows> }]> =
                context.sub.begin(ranks.view().tile::<Batches<Rows>, 1,
                    m![{ Count }, 1 # ({ Batches<Rows> }), { Batch<Rows> }, { Lanes<Rows> }],
                >(batch))
                    .fetch::<m![{ Batch<Rows> }, { Count }], Lanes<Rows>>()
                    .collect::<m![{ Batch<Rows> }, { Count }], Lanes<Rows>>()
                    .to_vrf();
                context.main.begin(shifted.view().tile::<Batches<Rows>, 1,
                    m![{ Count }, 1 # ({ Batches<Rows> }), { Batch<Rows> }, { Lanes<Rows> }],
                >(batch))
                .fetch::<m![{ Batch<Rows> }, { Count }], Lanes<Rows>>()
                .collect::<m![{ Batch<Rows> }, { Count }], Lanes<Rows>>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_reinterpret::<i32>()
                .vector_fxp(FxpBinaryOp::AddFxpSat, &ranks)
                .vector_narrow_split::<
                    m![{ Batch<Rows> }, { Count }, { Layout<Rows>::Lanes } / 4],
                    Half<Rows>,
                >()
                .vector_intra_slice_reduce::<
                    V,
                    m![{ Batch<Rows> }, { Layout<Rows>::Lanes } / 4],
                    Half<Rows>,
                >(IntraSliceReduceOpI32::Max)
                .vector_widen_concat::<Batch<Rows>, Lanes<Rows>>()
                .vector_final()
                .commit_trim::<Lanes<Rows>>()
                .commit_view(rank.view_mut().tile::<Batches<Rows>, 1,
                    m![1 #{!} ({ Batches<Rows> }), { Batch<Rows> }, { Lanes<Rows> }],
                >(batch));
            }
        }
        rank
    }
}

/// One slice's contiguous elements: a 256th of a spread vocabulary, its pointwise part of every
/// window when the windows fill fewer than 256 slices, otherwise its pointwise part of a window.
#[derive(Clone, Debug)]
pub struct Part<V>(PhantomData<V>);

impl<V: AxisName> AxisName for Part<V> {
    const NAME: Ident = Ident::new("Part");

    // A narrow vocabulary spreads when 256 slices of whole eight-element vectors hold it within the
    // index operand's 1024 elements per slice.
    const SIZE: usize =
        if Extent::<V>::NARROW && V::SIZE.is_multiple_of(256 * 8) && V::SIZE / 256 <= 1024 {
            V::SIZE / 256
        } else if Extent::<V>::NARROW {
            pointwise::Part::<m![V]>::SIZE * (V::SIZE / pointwise::Window::<m![V]>::SIZE)
        } else {
            pointwise::Part::<m![V]>::SIZE
        };
}

/// The elements of one transfer window, which every slice shares: the whole vocabulary unless its
/// pointwise windows fill all 256 slices.
#[derive(Clone, Debug)]
pub struct Extent<V>(PhantomData<V>);

impl<V: AxisName> Extent<V> {
    /// A vocabulary whose pointwise windows fill fewer than 256 slices: it transfers as one window,
    /// since the tensor DMA cannot factor a window index beside the padded slice axis.
    const NARROW: bool = pointwise::Window::<m![V]>::SIZE / pointwise::Part::<m![V]>::SIZE < 256;
}

impl<V: AxisName> AxisName for Extent<V> {
    const NAME: Ident = Ident::new("Window");

    const SIZE: usize = if Self::NARROW {
        V::SIZE
    } else {
        pointwise::Window::<m![V]>::SIZE
    };
}

/// `V - 1 - index` for every index of `V` on every chip, the ranks that let an integer maximum
/// choose the lowest of tied indices; a load computes them, as no checkpoint stores them.
pub(crate) fn ranks<V: AxisName, Chips: M, T>(
    _: &Checkpoint,
    context: &mut Context<'_>,
) -> Result<T, LoadError>
where
    T: Load<HostTensor<i32, m![{ Chips }, { m![V] }]>>,
{
    let ranks = (0..V::SIZE).map(|index| (V::SIZE - 1 - index) as i32);
    let ranks = ranks.collect::<Vec<_>>().repeat(Chips::SIZE);
    let host =
        HostTensor::try_from(HostBuffer::from_vec(i32::to_buf(&ranks))).map_err(|error| {
            LoadError::InvalidData {
                name: context.path(),
                reason: error.to_string(),
            }
        })?;
    Load::load(context, host)
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![Vocabulary = 262144];

    #[device(chip = 1)]
    fn argmax(
        device: &mut Device,
        selector: &Argmax<Vocabulary>,
        logits: &HbmTensor<bf16, Chip, m![Vocabulary]>,
    ) -> HbmTensor<i32, Chip, m![1 # 4]> {
        selector.forward(device, logits)
    }

    /// Finite BF16 maxima select the first index across signed-zero ties and window boundaries.
    #[tokio::test(flavor = "current_thread")]
    async fn selects_first_finite_maximum() {
        let mut device = Device::new(argmax.topology()).unwrap();
        let indices = HostTensor::<i32, m![Vocabulary]>::from_vec(
            (0..Vocabulary::SIZE).map(|i| (Vocabulary::SIZE - 1 - i) as i32),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let selector = Argmax::<Vocabulary>::new(indices);
        let cases: &[(&[(usize, f32)], usize)] = &[
            (&[(Vocabulary::SIZE - 1, 3.0)], Vocabulary::SIZE - 1),
            (&[(7, 2.0), (131_079, 2.0)], 7),
            (&[(1, -0.0), (Vocabulary::SIZE - 1, 0.0)], 1),
            (&[(1, 0.0), (Vocabulary::SIZE - 1, -0.0)], 1),
            (
                &[(0, bf16::from_bits(0x3f7f).to_f32()), (131_072, 1.0)],
                131_072,
            ),
            (&[(0, -0.5), (Vocabulary::SIZE - 1, -0.5)], 0),
            (&[(123, bf16::from_bits(0x7f7f).to_f32())], 123),
        ];
        for &(updates, expected) in cases {
            let mut values = vec![bf16::from_bits(0xff7f); Vocabulary::SIZE];
            for &(index, value) in updates {
                values[index] = bf16::from_f32(value);
            }
            let logits = HostTensor::<bf16, m![Vocabulary]>::from_vec(values)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let selected = launch(argmax, (&mut device, &selector, &logits))
                .await
                .unwrap()
                .to_host::<m![1 # 4]>(&mut device.pdma)
                .await
                .unwrap();
            assert_eq!(
                selected.into_vec()[0],
                expected as i32,
                "updates: {updates:?}"
            );
        }
    }
}
