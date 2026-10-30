//! `x * w / sqrt(mean(x²) + epsilon)` per row, with FP32 statistics and one rounding.

use super::{
    Chunk, Cluster, FeatureSlices, Lines, Spread, Tile, TilePart, TileParts, Widen, tiles,
};
use crate::Op;
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::modules::generic::schedule::{self, IntoStream, RowWise};
use crate::tensor::form::Weight;
use crate::tensor::{axis::Single, live::Live, scalar};
use core::borrow::Borrow;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::stage::{Standalone, VeOrder::IntraFirst, Way::Way4};
use furiosa_opt_std::prelude::*;

/// FP32 bits of the epsilon an [`RmsNorm`] adds when its type names none.
pub const EPSILON: usize = 1e-6_f32.to_bits() as usize;

/// BF16 rows times learned weights, divided by sqrt(mean(x²) + epsilon).
/// EPSILON stores the FP32 bit pattern in a usize const parameter.
#[derive(DeviceSend, Load)]
pub struct RmsNorm<
    Width: AxisName,
    const EPSILON: usize = { self::EPSILON },
    Chips: M = m![1],
    Parameter = Weight<m![Width], Chips>,
> {
    weight: Parameter,
    features: PhantomData<(Width, Chips)>,
}

impl<Width: AxisName, const EPSILON: usize, Chips: M, Parameter>
    RmsNorm<Width, EPSILON, Chips, Parameter>
{
    /// Retain one learned BF16 scale per feature on the same chip mapping as the input.
    pub const fn new(weight: Parameter) -> Self {
        Self {
            weight,
            features: PhantomData,
        }
    }

    /// Borrow the retained per-feature scale.
    pub fn weight(&self) -> &Parameter {
        &self.weight
    }

    /// The retained per-feature scale.
    pub fn into_weight(self) -> Parameter {
        self.weight
    }
}

/// A copy reading the same scale; written out, as device code cannot clone a `PhantomData`.
impl<Width: AxisName, const EPSILON: usize, Chips: M, Parameter: Clone> Clone
    for RmsNorm<Width, EPSILON, Chips, Parameter>
{
    fn clone(&self) -> Self {
        Self::new(self.weight.clone())
    }
}

/// Rows `u` whose true values are `c s u` for finite, positive factors, `c` of each row and `s`
/// of each head, with each row's `1 / (c s)²`: an RMS norm of the true rows runs on `u` with
/// epsilon `epsilon / (c s)²`.
pub struct Scaled<X, S> {
    value: X,
    inverse: S,
}

impl<X, Chips: M, C: M, P: M, L: M, R: M> Scaled<X, VrfTensor<f32, Chips, C, P, m![{ L }, { R }]>> {
    /// `value` with the factor `c` of each of its rows in `factors`, which names the rows `R` of a
    /// slice, and `s²` of each of its heads `L` in `heads`: one, or each head's value.
    pub fn new<H>(
        device: &mut Device,
        value: X,
        factors: DmTensorView<'_, bf16, Chips, C, P, R>,
        heads: H,
    ) -> Self
    where
        H: IntoBranchedOperand<
                f32,
                VeTensorShape<Chips, C, P, m![{ L }, { R } / 4], m![{ R } % 4]>,
            > + StashTransition<Spent, f32, { Way4 }>,
    {
        let inverse = device
            .sub
            .begin(factors)
            .fetch::<L, R>()
            .fetch_cast::<f32>()
            .collect::<L, R>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ L }, { R } / 4], m![{ R } % 4]>()
            .vector_stash()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), heads)
            .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
            .vector_widen_concat::<L, R>()
            .vector_final()
            .to_vrf();
        Self { value, inverse }
    }
}

impl<X, S> Scaled<X, S> {
    /// The rows and their inverse squared factors, for a norm that applies them.
    pub fn cancel(self) -> (X, S) {
        (self.value, self.inverse)
    }
}

/// A norm's learned per-feature scale, in a form the norm reads into each feature slice.
pub trait Learned<Width: AxisName, Chips: M> {
    fn lanes(&self, device: &mut Device) -> Features<Width, Chips>;
}

/// A norm gain written in the feature slices of the rows it scales.
pub type InDm<Width, Chips = m![1]> =
    DmTensor<bf16, Chips, Cluster, FeatureSlices<Width>, m![Width % ({ Symbol<Chunk<Width>> })]>;

impl<Width: AxisName, Chips: M> Learned<Width, Chips> for InDm<Width, Chips> {
    fn lanes(&self, device: &mut Device) -> Features<Width, Chips> {
        self.widen::<Width, 8>(&mut device.sub)
    }
}

/// The retained HBM gain, owned or borrowed, is written in the slices each time a norm reads it.
impl<Width: AxisName, Chips: M, T> Learned<Width, Chips> for Weight<m![Width], Chips, T>
where
    T: Borrow<HbmTensor<bf16, Chips, m![Width]>>,
{
    fn lanes(&self, device: &mut Device) -> Features<Width, Chips> {
        let in_dm: InDm<Width, Chips> = self.tensor().to_dm(&mut device.tdma);
        in_dm.lanes(device)
    }
}

/// An [`RmsNorm`] of unit gain, whose checkpoint stores no weight: it multiplies by a vector of
/// ones a load places, held as `T`, so it runs the learned norm's kernel.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct UnitRmsNorm<
    Width: AxisName,
    const EPSILON: usize = { self::EPSILON },
    Chips: M = m![1],
    T = HbmTensor<bf16, Chips, m![Width]>,
>(Unit<Width, EPSILON, Chips, T>);

/// The learned norm a unit norm runs.
type Unit<Width, const EPSILON: usize, Chips, T> =
    RmsNorm<Width, EPSILON, Chips, Weight<m![Width], Chips, T>>;

impl<Width: AxisName, const EPSILON: usize, Chips: M> UnitRmsNorm<Width, EPSILON, Chips> {
    /// The norm multiplying by `ones`, which must hold one in every feature.
    pub const fn new(ones: Weight<m![Width], Chips>) -> Self {
        Self(RmsNorm::new(ones))
    }
}

impl<I, Width: AxisName, const EPSILON: usize, Chips: M, T> crate::Op<I>
    for UnitRmsNorm<Width, EPSILON, Chips, T>
where
    Unit<Width, EPSILON, Chips, T>: crate::Op<I>,
{
    type Output = <Unit<Width, EPSILON, Chips, T> as crate::Op<I>>::Output;

    fn forward(&self, device: &mut Device, input: I) -> Self::Output {
        Op::forward(&self.0, device, input)
    }
}

/// The ones a load places under `weight`, as no checkpoint stores them.
/// By hand, with the update below: a derived `with` field would rebuild them in every update.
impl<'a, Width: AxisName, const EPSILON: usize, Chips: M + 'static, T> Load<&'a Checkpoint>
    for UnitRmsNorm<Width, EPSILON, Chips, T>
where
    T: Load<HostTensor<bf16, m![{ Chips }, { m![Width] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let ones = HostTensor::from_vec(vec![bf16::from_f32(1.0); Chips::SIZE * Width::SIZE]);
            cx.field("weight", "weight", |cx| Load::load(cx, ones))
                .map(|ones| Self(RmsNorm::new(ones)))
        })
    }
}

/// The ones are no update target.
impl<'t, 's, Width: AxisName, const EPSILON: usize, Chips: M, T>
    Load<(
        &'t mut UnitRmsNorm<Width, EPSILON, Chips, T>,
        &'s Checkpoint,
    )> for &'t mut UnitRmsNorm<Width, EPSILON, Chips, T>
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut UnitRmsNorm<Width, EPSILON, Chips, T>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |_| Ok::<(), LoadError>(()))?;
        Ok(into)
    }
}

impl<'x, Width: AxisName, Rows: AxisName, const EPSILON: usize, Chips: M, Parameter>
    Op<&'x Spread<Width, Rows, Chips>> for RmsNorm<Width, EPSILON, Chips, Parameter>
where
    Parameter: Learned<Width, Chips>,
{
    type Output = Spread<Width, Rows, Chips>;

    fn forward(&self, device: &mut Device, input: &'x Spread<Width, Rows, Chips>) -> Self::Output {
        self.normalized::<m![Rows], _>(device, input.view())
            .commit()
    }
}

/// Every row of the envelope is normalized alone, so the live rows give what they give in a
/// launch of their own.
impl<'x, Width: AxisName, Rows: AxisName, const EPSILON: usize, Chips: M, Parameter>
    Op<&'x Live<Spread<Width, Rows, Chips>, Rows>> for RmsNorm<Width, EPSILON, Chips, Parameter>
where
    Parameter: Learned<Width, Chips>,
{
    type Output = Live<Spread<Width, Rows, Chips>, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<Spread<Width, Rows, Chips>, Rows>,
    ) -> Self::Output {
        input.with(self.forward(device, input.value()))
    }
}

impl<Width: AxisName, const EPSILON: usize, Chips: M> Op<&HbmTensor<bf16, Chips, m![Width]>>
    for RmsNorm<Width, EPSILON, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Width]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &HbmTensor<bf16, Chips, m![Width]>,
    ) -> Self::Output {
        let input: Spread<Width, Single, Chips> = input.to_dm(&mut device.tdma);
        self.forward(&mut *device, &input).to_hbm(&mut device.tdma)
    }
}

/// `input * weight / sqrt(mean(input^2) + epsilon)` over each of the `P` head rows of every slice,
/// the squares summed in one pass and scaled in another; `EPSILON` holds epsilon's FP32 bits.
impl<'x, S: M, P: M, Head: AxisName, R: AxisName, const EPSILON: usize, T, Chips: M + 'static>
    Op<&'x Live<DmTensor<bf16, Chips, m![1 # 2], S, m![{ P }, Head]>, R>>
    for RmsNorm<Head, EPSILON, Chips, Weight<m![Head], Chips, T>>
where
    T: Borrow<HbmTensor<bf16, Chips, m![Head]>>,
{
    type Output = Live<DmTensor<bf16, Chips, m![1 # 2], S, m![{ P }, Head]>, R>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<DmTensor<bf16, Chips, m![1 # 2], S, m![{ P }, Head]>, R>,
    ) -> Self::Output {
        let mean: DmTensor<f32, Chips, m![1 # 2], S, m![{ P }, 1 # 8]> =
            Squares::<HeadLanes<P, Head>, Head, m![{ P }], m![1], EPSILON>::of(
                &mut device.main,
                input.value().view(),
            )
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let inverse: VrfTensor<f32, Chips, m![1 # 2], S, m![{ P }, 1 # 8]> =
            Root::<m![{ P }], m![1 # 8]>::of(&mut device.sub, mean.view(), |square| square)
                .to_vrf();
        let weight: DmTensor<bf16, Chips, m![1 # 2], S, m![Head]> =
            self.weight.tensor().to_dm(&mut device.tdma);
        let weight = weight.widen::<Head, 16>(&mut device.sub);
        input.with(
            Scale::<HeadLanes<P, Head>>::of(
                &mut device.main,
                input.value().view(),
                &inverse,
                &weight,
            )
            .commit(),
        )
    }
}

/// The FP32 vectors of the features the scaling pass keeps in its registers: the weight.
const VECTORS: usize = 1;

/// How a slice's part of a row of `R` moves through a norm's passes.
type TileLanes<R, W> = Lanes<
    m![{ TilePart<W, VECTORS> } / 8],
    m![W % 8],
    m![{ TilePart<W, VECTORS> } / 8, W % 8 / 4 % 2],
    m![W % 4],
    m![{ TilePart<W, VECTORS> } / 8],
    m![W % 8],
    m![W % 8],
    m![R % ({ Symbol<Lines<W, VECTORS>> }), { Symbol<TileParts<W, VECTORS>> }],
    m![1 # 8],
>;

/// Rows of `D` normalized by their RMS and the learned weight, rounded once to BF16, a tile at a
/// time.
impl<D, Width: AxisName, R: AxisName, const EPSILON: usize, Chips: M, T>
    Op<&HbmTensor<D, Chips, m![R, Width]>>
    for RmsNorm<Width, EPSILON, Chips, Weight<m![Width], Chips, T>>
where
    D: MaterializableScalar + FetchCast<f32>,
    T: Borrow<HbmTensor<bf16, Chips, m![Width]>>,
{
    type Output = HbmTensor<bf16, Chips, m![R, Width]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &HbmTensor<D, Chips, m![R, Width]>,
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        let weight: Tile<bf16, R, Width, VECTORS, Chips> =
            self.weight.tensor().view().to_dm(&mut device.tdma);
        let weight = weight.widen::<Width, 16>(&mut device.sub);
        tiles(device, input, &mut output, |device, rows, normalized| {
            let inverse = Squares::<
                TileLanes<R, Width>,
                Width,
                m![1],
                m![Width / (Width / ({ Symbol<TileParts<Width, VECTORS>> }))],
                EPSILON,
            >::inverse(device, rows.view(), |square| square);
            Scale::<TileLanes<R, Width>>::of(&mut device.main, rows.view(), &inverse, &weight)
                .commit_view(normalized.view_mut());
        });
        output
    }
}

/// Typed RMS contract with runtime epsilon, FP32 statistics and source-dtype output.
#[derive(DeviceSend)]
pub struct Rms<Weight, Epsilon> {
    /// Per-feature gain: a learned scale, a [`Centered`](super::Centered) offset from one, or `()`
    /// for unit gain.
    pub weight: Weight,
    /// Device-resident scalar added to the mean square.
    pub epsilon: Epsilon,
}

impl<Weight, Epsilon> Rms<Weight, Epsilon> {
    pub fn new(weight: Weight, epsilon: Epsilon) -> Self {
        Self { weight, epsilon }
    }
}

impl<D: Scalar, Source: Scalar, Width: AxisName, Count: AxisName, Chips: M + 'static>
    Op<&HbmTensor<Source, Chips, m![Count, Width]>>
    for Rms<HbmTensor<D, Chips, m![Width]>, HbmTensor<f32, Chips, m![1]>>
{
    type Output = HbmTensor<Source, Chips, m![Count, Width]>;

    fn forward(
        &self,
        _: &mut Device,
        _: &HbmTensor<Source, Chips, m![Count, Width]>,
    ) -> Self::Output {
        todo!("NPU RMS normalization with FP32 statistics and source-dtype output")
    }
}

type Reduced = m![256];

/// A norm's rows as the stream its caller commits.
pub(super) type Commit<'l, Width, Rows, Chips> = CommitTrimTensor<
    'l,
    { Tu::Main },
    bf16,
    Chips,
    Cluster,
    FeatureSlices<Width>,
    Time<Width, Rows>,
    m![Width % 8],
>;

/// Each row's features in a feature slice, eight at a time.
type Time<Width, Rows> = m![{ Rows }, Width % ({ Symbol<Chunk<Width>> }) / 8];

/// [`Time`] narrowed to four lanes.
type Narrow<Width, Rows> = m![{ Rows }, Width % ({ Symbol<Chunk<Width>> }) / 8, Width % 8 / 4 % 2];

/// FP32 rows in four lanes at stage `Stage` of a vector pass in unit `U`.
type Stream<'l, const U: Tu, Stage, X, C, S, T, P> =
    VectorTensor<'l, U, Stage, f32, X, C, S, T, P, Fresh, { IntraFirst }, Standalone, { Way4 }>;

/// Each row's mean square at stage `Stage` of its root pass, one value in four lanes.
type Row<'l, const U: Tu, Stage, X, C, S, Rows> = Stream<'l, U, Stage, X, C, S, Rows, m![1 # 4]>;

/// How a norm's passes move rows: `FetchTime` packets of `FetchPacket` from DM, FP32 `Time` packets
/// of `Packet`, four-lane `Narrow` packets of `Half` for arithmetic, `Stored` values back to DM; a
/// row's sums over its slices reduce into `Reduced`, which leaves its statistic as `Statistic`.
pub(crate) struct Lanes<
    Time,
    Packet,
    Narrow,
    Half,
    FetchTime = Time,
    FetchPacket = Packet,
    Stored = Packet,
    Reduced = (),
    Statistic = (),
>(
    PhantomData<(Time, Packet, Narrow, Half)>,
    PhantomData<(FetchTime, FetchPacket, Stored)>,
    PhantomData<(Reduced, Statistic)>,
);

/// `mean(x²) + EPSILON` of each of the `Rows` rows over `Reduce`, the rows moved in `Lanes`;
/// `EPSILON` holds epsilon's FP32 bits, split over the `Parts` slices a caller's reduction adds.
pub(crate) struct Squares<Lanes, Reduce, Rows, Parts, const EPSILON: usize>(
    PhantomData<(Lanes, Reduce, Rows, Parts)>,
);

impl<
    Time: M,
    Packet: M,
    Narrow: M,
    Half: M,
    FetchTime: M,
    FetchPacket: M,
    Stored: M,
    Reduced,
    Statistic,
    Reduce: AxisName,
    Rows: M,
    Parts: M,
    const EPSILON: usize,
>
    Squares<
        Lanes<Time, Packet, Narrow, Half, FetchTime, FetchPacket, Stored, Reduced, Statistic>,
        Reduce,
        Rows,
        Parts,
        EPSILON,
    >
{
    /// The mean squares of `input`'s rows, as the stream a caller reduces over slices and commits.
    pub(crate) fn of<'l, D: MaterializableScalar + FetchCast<f32>, X: M, C: M, S: M, E: M>(
        main: &'l mut TuContext<{ Tu::Main }>,
        input: DmTensorView<'l, D, X, C, S, E>,
    ) -> VectorClipTensor<'l, { Tu::Main }, f32, X, C, S, Rows, m![1 # 8], Spent, { IntraFirst }>
    {
        let rows = main
            .begin(input)
            .fetch::<FetchTime, FetchPacket>()
            .fetch_cast::<f32>()
            .collect::<Time, Packet>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero);
        if const { Narrow::SIZE == Time::SIZE } {
            rows.vector_narrow_trim::<Half>()
                .vector_mean_square::<Reduce, Rows, Parts, EPSILON>()
        } else {
            rows.vector_narrow_split::<Narrow, Half>()
                .vector_mean_square::<Reduce, Rows, Parts, EPSILON>()
        }
    }

    /// `1 / sqrt(mean(x²) + epsilon)` of each row of `input`, in every slice that holds part of
    /// the row; `epsilon` adds a loaded epsilon, or passes on a square that holds `EPSILON`.
    pub(crate) fn inverse<D, X: M, C: M, S: M, E: M, Added>(
        device: &mut Device,
        input: DmTensorView<'_, D, X, C, S, E>,
        epsilon: impl for<'l> FnOnce(
            Row<'l, { Tu::Sub }, stage::Narrow, X, C, S, Rows>,
        ) -> Row<'l, { Tu::Sub }, Added, X, C, S, Rows>,
    ) -> VrfTensor<f32, X, C, S, Statistic>
    where
        D: MaterializableScalar + FetchCast<f32>,
        Reduced: M,
        Statistic: M,
        Added: stage::Stage + stage::CanTransitionTo<stage::Fp>,
    {
        let squares: DmTensor<f32, X, C, Reduced, Statistic> = Self::of(&mut device.main, input)
            .vector_inter_slice_reduce::<Reduced, Rows>(InterSliceReduceOpF32::Add)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        // SAFETY: the reduction leaves each row's statistic in every slice of the row, in place.
        let squares: DmTensor<f32, X, C, S, Statistic> = unsafe { squares.reshape() };
        Root::<Rows, m![1 # 8]>::of(&mut device.sub, squares.view(), epsilon).to_vrf()
    }
}

/// The statistic an RMS norm takes the square root of, from a stream narrowed to four lanes.
trait MeanSquare<'l, const T: Tu, Chip: M, Cluster: M, Slice: M> {
    /// `mean(x²) + epsilon` over `Reduce` into `Rows`, dividing by `Reduce::SIZE`; `EPSILON` holds
    /// epsilon's FP32 bits, shared over `Slices` whose sums a later inter-slice reduction adds.
    fn vector_mean_square<Reduce: AxisName, Rows: M, Slices: M, const EPSILON: usize>(
        self,
    ) -> VectorTensor<
        'l,
        T,
        stage::Clip,
        f32,
        Chip,
        Cluster,
        Slice,
        Rows,
        m![1 # 8],
        Spent,
        { stage::VeOrder::IntraFirst },
    >;
}

impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    MeanSquare<'l, T, Chip, Cluster, Slice>
    for VectorTensor<
        'l,
        T,
        stage::Narrow,
        f32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        Fresh,
        { stage::VeOrder::IntraFirst },
        stage::Standalone,
        { stage::Way::Way4 },
    >
{
    fn vector_mean_square<Reduce: AxisName, Rows: M, Slices: M, const EPSILON: usize>(
        self,
    ) -> VectorTensor<
        'l,
        T,
        stage::Clip,
        f32,
        Chip,
        Cluster,
        Slice,
        Rows,
        m![1 # 8],
        Spent,
        { stage::VeOrder::IntraFirst },
    > {
        self.vector_stash()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
            .vector_intra_slice_reduce::<Reduce, Rows, m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_fp_div(const { Reduce::SIZE as f32 })
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(
                ClipBinaryOpF32::Add,
                const { f32::from_bits(EPSILON as u32) / Slices::SIZE as f32 },
            )
    }
}

/// The inverse root of each of the `Rows` mean squares, read in `Fetched` packets.
pub(crate) struct Root<Rows, Fetched>(PhantomData<(Rows, Fetched)>);

impl<Rows: M, Fetched: M> Root<Rows, Fetched> {
    /// `1 / sqrt(square + epsilon)` of each mean square in `statistic`, in the eight lanes a scale
    /// pass reads; `epsilon` adds a loaded epsilon, or passes on a square that already holds it.
    pub(crate) fn of<'l, const U: Tu, X: M, C: M, S: M, E: M, Added>(
        tu: &'l mut TuContext<U>,
        statistic: DmTensorView<'l, f32, X, C, S, E>,
        epsilon: impl FnOnce(
            Row<'l, U, stage::Narrow, X, C, S, Rows>,
        ) -> Row<'l, U, Added, X, C, S, Rows>,
    ) -> VectorFinalTensor<'l, U, f32, X, C, S, Rows, m![1 # 8]>
    where
        Added: stage::Stage + stage::CanTransitionTo<stage::Fp>,
    {
        let square = tu
            .begin(statistic)
            .fetch::<Rows, Fetched>()
            .fetch_cast::<f32>()
            .collect::<Rows, m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>();
        epsilon(square)
            .vector_fp_unary(FpUnaryOp::Sqrt)
            .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
    }
}

/// Rows moved in `Lanes`, times their inverse root mean square and the gain, rounded once to BF16.
pub(crate) struct Scale<Lanes>(PhantomData<Lanes>);

impl<
    Time: M,
    Packet: M,
    Narrow: M,
    Half: M,
    FetchTime: M,
    FetchPacket: M,
    Stored: M,
    Reduced,
    Statistic,
> Scale<Lanes<Time, Packet, Narrow, Half, FetchTime, FetchPacket, Stored, Reduced, Statistic>>
{
    /// Each row of `input` times its `inverse` and the `gain`, as the stream its caller commits.
    pub(crate) fn of<'l, D, X: M, C: M, S: M, E: M, R: M, G>(
        main: &'l mut TuContext<{ Tu::Main }>,
        input: DmTensorView<'l, D, X, C, S, E>,
        inverse: &VrfTensor<f32, X, C, S, R>,
        gain: G,
    ) -> CommitTrimTensor<'l, { Tu::Main }, bf16, X, C, S, Time, Stored>
    where
        D: MaterializableScalar + FetchCast<f32>,
        G: IntoBranchedOperand<f32, VeTensorShape<X, C, S, Time, Half>>
            + IntoBranchedOperand<f32, VeTensorShape<X, C, S, Narrow, Half>>
            + StashTransition<Fresh, f32, { Way4 }, Next = Fresh>,
    {
        let rows = main
            .begin(input)
            .fetch::<FetchTime, FetchPacket>()
            .fetch_cast::<f32>()
            .collect::<Time, Packet>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero);
        let scaled = if const { Narrow::SIZE == Time::SIZE } {
            rows.vector_narrow_trim::<Half>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), inverse)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), gain)
                .vector_widen_pad::<Packet>()
        } else {
            rows.vector_narrow_split::<Narrow, Half>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), inverse)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), gain)
                .vector_widen_concat::<Time, Packet>()
        };
        scaled
            .vector_final()
            .cast::<bf16, m![{ Stored } # 16]>()
            .commit_trim::<Stored>()
    }
}

/// How a [`Spread`]'s rows move through a norm's passes.
type SpreadLanes<Width, Rows> = Lanes<
    Time<Width, Rows>,
    m![Width % 8],
    Narrow<Width, Rows>,
    m![Width % 4],
    Time<Width, Rows>,
    m![Width % 8],
    m![Width % 8],
    Reduced,
    m![{ Rows }, 1 # 8],
>;

/// The feature slices a row spans, whose sums the norm reduces.
type Parts<Width> = m![Width / ({ Symbol<Chunk<Width>> })];

impl<Width: AxisName, const EPSILON: usize, Chips: M, Parameter: Learned<Width, Chips>>
    RmsNorm<Width, EPSILON, Chips, Parameter>
{
    /// Each of the `Rows` rows of every feature slice divided by its root mean square and scaled
    /// by the gain, whose lanes are written once the inverse is in the vector register file.
    pub(super) fn normalized<'l, Rows: M, E: M>(
        &self,
        device: &'l mut Device,
        input: DmTensorView<'l, bf16, Chips, Cluster, FeatureSlices<Width>, E>,
    ) -> Commit<'l, Width, Rows, Chips> {
        let inverse =
            Squares::<SpreadLanes<Width, Rows>, Width, Rows, Parts<Width>, EPSILON>::inverse(
                device,
                input.clone(),
                |square| square,
            );
        // The learned scale is shared by every row.
        let scale = self.weight.lanes(device);
        Scale::<SpreadLanes<Width, Rows>>::of(&mut device.main, input, &inverse, &scale)
    }
}

/// How the head rows of a per-head norm move through its passes: read sixteen features a packet.
type HeadLanes<P, Head> = Lanes<
    m![{ P }, Head / 8],
    m![Head % 8],
    m![{ P }, Head / 4],
    m![Head % 4],
    m![{ P }, Head / 16],
    m![Head % 16],
>;

/// Per-feature FP32 operands of the normalized rows' pass.
pub(super) type Features<Width, Chips> =
    VrfTensor<f32, Chips, Cluster, FeatureSlices<Width>, m![Width % ({ Symbol<Chunk<Width>> })]>;

/// The stream shapes a gain multiplies in: each row's features four lanes at a time, split or
/// trimmed.
type Normalized<Width, Rows, Chips> =
    VeTensorShape<Chips, Cluster, FeatureSlices<Width>, Narrow<Width, Rows>, m![Width % 4]>;
type Trimmed<Width, Rows, Chips> =
    VeTensorShape<Chips, Cluster, FeatureSlices<Width>, Time<Width, Rows>, m![Width % 4]>;

/// `input * gain / sqrt(mean(input²) + epsilon)` per row with FP32 statistics and one BF16
/// rounding; epsilon joins the whole row's mean square once, as a device operand.
pub(super) fn normalize<Width: AxisName, Rows: AxisName, Chips: M, Gain>(
    device: &mut Device,
    input: &Spread<Width, Rows, Chips>,
    epsilon: &HbmTensor<f32, Chips, m![1 # 8]>,
    gain: Gain,
) -> Spread<Width, Rows, Chips>
where
    Gain: IntoBranchedOperand<f32, Normalized<Width, m![Rows], Chips>>
        + IntoBranchedOperand<f32, Trimmed<Width, m![Rows], Chips>>
        + StashTransition<Fresh, f32, { Way4 }, Next = Fresh>,
{
    let epsilon: DmTensor<f32, Chips, Cluster, FeatureSlices<Width>, m![1 # 8]> =
        epsilon.to_dm(&mut device.tdma);
    let epsilon = scalar::register(device, &epsilon);
    let inverse =
        Squares::<SpreadLanes<Width, m![Rows]>, Width, m![Rows], Parts<Width>, 0>::inverse(
            device,
            input.view(),
            |square| square.vector_fp_binary(FpBinaryOp::AddF, &epsilon),
        );
    Scale::<SpreadLanes<Width, m![Rows]>>::of(&mut device.main, input.view(), &inverse, gain)
        .commit()
}

/// A norm streams one `forward` a tile.
impl<'m, 's, U: schedule::Stream, E, S, Width: AxisName, const EPSILON: usize, Chips: M, Parameter>
    IntoStream<U, E, &'s mut S> for &'m RmsNorm<Width, EPSILON, Chips, Parameter>
where
    RowWise<'s, &'m RmsNorm<Width, EPSILON, Chips, Parameter>, E, S, U>: schedule::Stream,
{
    type Stream = RowWise<'s, &'m RmsNorm<Width, EPSILON, Chips, Parameter>, E, S, U>;

    fn into_stream(self, env: E, state: &'s mut S, upstream: U) -> Self::Stream {
        RowWise::new(self, env, state, upstream)
    }
}

#[cfg(test)]
mod tests {
    use crate::Op;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use crate::ops::attention::geometry::{Hidden, QueryGroups};
    use crate::ops::attention::{
        Heads,
        tests::{Head, KvHeads},
    };
    use crate::ops::normalization::{
        RowMajor,
        tests::{random, residual},
    };
    use crate::tensor::{
        axis::{Eight, Single, Sixteen},
        live::Live,
    };
    use crate::tensor::{live::tests::count, scalar::Value};
    use crate::tests::DEVICE;
    use crate::{ops::normalization::RmsNorm, tensor::form::Weight};

    type Chip = m![1];
    type Rows<Width, Count> = HbmTensor<bf16, Chip, RowMajor<Width, Count>>;

    axes![
        Quarters = 2048,
        Residual = 2048,
        Samples = 640,
        Embed = 3840,
        Tokens = 256
    ];

    /// The epsilon of the FP32 rows' norms, as FP32 bits.
    const RESIDUAL_EPSILON: usize = 1e-5_f32.to_bits() as usize;

    /// Row `row` scaled by `scale * 2^row`, so a variance shared across rows misnormalizes every
    /// row but one.
    fn values(rows: usize, width: usize, scale: f32) -> Vec<bf16> {
        (0..rows * width)
            .map(|index| {
                let (row, column) = (index / width, index % width);
                bf16::from_f32(
                    (((column * 37) % 101) as f32 / 25.0 - 2.0) * scale * (1 << row) as f32,
                )
            })
            .collect()
    }

    fn scales(width: usize) -> Vec<bf16> {
        (0..width)
            .map(|index| bf16::from_f32(((index * 13) % 29) as f32 / 24.0 - 0.6))
            .collect()
    }

    /// Each row of `weight.len()` values times its weights over the root of its FP32 mean square
    /// plus epsilon.
    fn definition(rows: &[bf16], weight: &[bf16]) -> Vec<f32> {
        rows.chunks_exact(weight.len())
            .flat_map(|row| {
                let mean =
                    row.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>() / weight.len() as f32;
                let inverse = 1.0 / (mean + 1e-6).sqrt();
                row.iter()
                    .zip(weight)
                    .map(move |(v, w)| v.to_f32() * inverse * w.to_f32())
            })
            .collect()
    }

    /// Every value within 2% of the definition's, or of one where that is smaller.
    fn assert_close(actual: &[bf16], expected: &[f32], width: usize, scale: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (got, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (got.to_f32() - expected).abs() <= 0.02 * expected.abs().max(1.0),
                "scale {scale} row {} column {}: {} vs {expected}",
                index / width,
                index % width,
                got.to_f32(),
            );
        }
    }

    #[device(chip = 1)]
    fn normalized<Width: AxisName, Count: AxisName>(
        device: &mut Device,
        norm: &RmsNorm<Width>,
        input: &Rows<Width, Count>,
        output: &mut Rows<Width, Count>,
    ) {
        let input: super::Spread<Width, Count> = input.to_dm(&mut device.tdma);
        norm.forward(&mut *device, &input)
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    /// Against the definition on the host, per row. The small scale shrinks the mean square to
    /// epsilon's order, so an epsilon added once per slice instead of once overall fails loudly.
    async fn matches_the_definition_over<Width: AxisName, Count: AxisName>() {
        let mut device = Device::new(normalized.topology()).unwrap();
        for scale in [1.0f32, 1e-3] {
            let values = values(Count::SIZE, Width::SIZE, scale);
            let scales = scales(Width::SIZE);
            let expected = definition(&values, &scales);
            let input = HostTensor::<bf16, RowMajor<Width, Count>>::from_vec(values)
                .to_hbm::<Chip, RowMajor<Width, Count>>(&mut device.pdma)
                .await
                .unwrap();
            let norm = RmsNorm::new(Weight::new(
                HostTensor::<bf16, m![Width]>::from_vec(scales)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ));
            let mut actual = HostTensor::<bf16, RowMajor<Width, Count>>::zero()
                .to_hbm::<Chip, RowMajor<Width, Count>>(&mut device.pdma)
                .await
                .unwrap();
            launch(normalized, (&mut device, &norm, &input, &mut actual))
                .await
                .unwrap();
            let actual = actual
                .to_host::<RowMajor<Width, Count>>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            assert_close(&actual, &expected, Width::SIZE, scale);
        }
    }

    /// Per-row RMS normalization matches FP32 statistics when epsilon dominates the variance.
    #[tokio::test]
    async fn matches_the_definition() {
        matches_the_definition_over::<Hidden<1024>, Eight>().await;
    }

    /// Sixteen rows preserve independent statistics across all 256 feature slices.
    #[tokio::test]
    async fn sixteen_rows_match_the_definition() {
        matches_the_definition_over::<Hidden<4096>, Sixteen>().await;
    }

    #[device(chip = 1)]
    fn heads<Kv: AxisName, Group: AxisName, H: AxisName>(
        device: &mut Device,
        norm: &RmsNorm<H>,
        live: &Value,
        input: &HbmTensor<bf16, Chip, m![Sixteen, Kv, Group, H]>,
    ) -> HbmTensor<bf16, Chip, m![Sixteen, Kv, Group, H]> {
        let value = input.to_dm(&mut device.tdma);
        let input: Live<Heads<Sixteen, Kv, Group, H>, Sixteen> =
            Live::<(), Sixteen>::read(live).with(value);
        norm.forward(&mut *device, &input)
            .value()
            .to_hbm(&mut device.tdma)
    }

    /// Every `H`-wide head row of `Group` heads for each of `Kv` key/value heads, in the slices a projection writes them
    /// in, against the definition on the host. Mean squares differ from row to row, and the
    /// small scale shrinks them to epsilon's order.
    async fn normalizes_each_head_of<Kv: AxisName, Group: AxisName, H: AxisName>() {
        let mut device = Device::new(heads.topology()).unwrap();
        let live = count(Sixteen::SIZE).to_hbm(&mut device.pdma).await.unwrap();
        for scale in [1.0f32, 1e-3] {
            let values: Vec<bf16> = random(
                Sixteen::SIZE * Kv::SIZE * Group::SIZE * H::SIZE,
                0xaaaa_bbbb,
            )
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let row = index / H::SIZE;
                bf16::from_f32(value.to_f32() * scale * (1 + row % 5) as f32)
            })
            .collect();
            let scales = scales(H::SIZE);
            let expected = definition(&values, &scales);
            let input = HostTensor::<bf16, m![Sixteen, Kv, Group, H]>::from_vec(values)
                .to_hbm::<Chip, m![Sixteen, Kv, Group, H]>(&mut device.pdma)
                .await
                .unwrap();
            let norm = RmsNorm::new(Weight::new(
                HostTensor::<bf16, m![H]>::from_vec(scales)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ));
            let actual = launch(heads, (&mut device, &norm, &live, &input))
                .await
                .unwrap()
                .to_host::<m![Sixteen, Kv, Group, H]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            assert_close(&actual, &expected, H::SIZE, scale);
        }
    }

    /// Head rows 256 wide under eight key/value heads, and 512 wide under one, normalize by their
    /// own statistics.
    #[tokio::test]
    async fn normalizes_each_wide_head() {
        axes![Narrow = 256, Wide = 512];
        normalizes_each_head_of::<KvHeads, QueryGroups<2>, Narrow>().await;
        normalizes_each_head_of::<Single, QueryGroups<16>, Wide>().await;
    }

    /// Query and key head rows each normalize by their own statistics.
    #[tokio::test]
    async fn normalizes_each_head() {
        normalizes_each_head_of::<KvHeads, QueryGroups<2>, Head>().await;
        normalizes_each_head_of::<KvHeads, Single, Head>().await;
    }

    #[device(chip = 1)]
    fn normalized_rows<D: MaterializableScalar + FetchCast<f32>, R: AxisName, W: AxisName>(
        device: &mut Device,
        norm: &RmsNorm<W, RESIDUAL_EPSILON>,
        input: &HbmTensor<D, Chip, m![R, W]>,
    ) -> HbmTensor<bf16, Chip, m![R, W]> {
        norm.forward(device, input)
    }

    /// Each row of `D` is normalized by its own RMS and the learned weight, as the FP64 definition
    /// rounded once to BF16; `cast` stores a value, `round` is the value it stores.
    async fn normalizes<D: MaterializableScalar + FetchCast<f32>, R: AxisName, W: AxisName>(
        cast: fn(f32) -> D,
        round: fn(f32) -> f32,
    ) {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let inputs: Vec<f32> = residual(R::SIZE, W::SIZE).into_iter().map(round).collect();
        let weight: Vec<bf16> = random(W::SIZE, 0x9e37_79b9_7f4a_7c15)
            .iter()
            .map(|v| bf16::from_f32(1.0 + v.to_f32() / 2.0))
            .collect();
        let norm = RmsNorm::<W, RESIDUAL_EPSILON>::new(Weight::new(
            HostTensor::<bf16, m![W]>::from_vec(weight.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        ));
        let input = HostTensor::<D, m![R, W]>::from_vec(
            inputs.iter().map(|v| cast(*v)).collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R, W]>(&mut device.pdma)
        .await
        .unwrap();
        let output = launch(normalized_rows, (&mut device, &norm, &input))
            .await
            .unwrap()
            .to_host::<m![R, W]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (row, values) in inputs.chunks_exact(W::SIZE).enumerate() {
            let mean = values.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / W::SIZE as f64;
            let inverse = 1.0 / (mean + 1e-5).sqrt();
            for (column, value) in values.iter().enumerate() {
                let expected = f64::from(*value) * inverse * f64::from(weight[column].to_f32());
                let got = f64::from(output[row * W::SIZE + column].to_f32());
                assert!(
                    (got - expected).abs() <= expected.abs() / 256.0 + 1e-4,
                    "row {row} column {column}: {got} vs {expected}"
                );
            }
        }
    }

    /// FP32 rows.
    #[tokio::test(flavor = "current_thread")]
    async fn normalizes_fp32_rows_into_bf16() {
        normalizes::<f32, Residual, Quarters>(|v| v, |v| v).await;
    }

    /// BF16 rows at the two widths of a multimodal embedder.
    #[tokio::test(flavor = "current_thread")]
    async fn normalizes_bf16_rows() {
        let round = |v: f32| bf16::from_f32(v).to_f32();
        normalizes::<bf16, Tokens, Samples>(bf16::from_f32, round).await;
        normalizes::<bf16, Tokens, Embed>(bf16::from_f32, round).await;
    }
}
