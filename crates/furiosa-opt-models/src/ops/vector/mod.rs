//! Vector-engine passes that several ops share, each emitting the same stages at every caller.

use crate::ops::pointwise::cast::Finite;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::stage::VeOrder::IntraFirst;
use furiosa_opt_std::prelude::*;

/// Combine the two operands a stream interleaves along one axis.
pub(crate) trait Zip<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Packet: M> {
    /// `op` of the two halves of `A` in FP32. `Time` is the stream's time without `A`, `Narrow`
    /// and `Half` its four-element split; the result returns to `Time` and the stream's packet.
    fn vector_zip<A: AxisName, Time: M, Narrow: M, Half: M>(
        self,
        op: FpBinaryOp,
    ) -> VectorTensor<
        'l,
        T,
        stage::Widen,
        f32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        Fresh,
        { stage::VeOrder::IntraFirst },
        stage::Zipped,
    >;
}

impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Source: M, Packet: M>
    Zip<'l, T, Chip, Cluster, Slice, Packet>
    for VectorInitTensor<'l, T, f32, Chip, Cluster, Slice, Source, Packet>
{
    fn vector_zip<A: AxisName, Time: M, Narrow: M, Half: M>(
        self,
        op: FpBinaryOp,
    ) -> VectorTensor<
        'l,
        T,
        stage::Widen,
        f32,
        Chip,
        Cluster,
        Slice,
        Time,
        Packet,
        Fresh,
        { stage::VeOrder::IntraFirst },
        stage::Zipped,
    > {
        self.vector_intra_slice_unzip::<A, Time>()
            .vector_narrow_split::<Narrow, Half>()
            .vector_fp_zip(op)
            .vector_widen_concat::<Time, Packet>()
    }
}

/// The smallest E4M3 row scale: a row of zeros quantizes to zeros instead of dividing by zero.
pub(crate) const SMALLEST: f32 = 1e-12;

/// A row's E4M3 quantization from a stream narrowed to four lanes: its scale, its largest magnitude
/// over E4M3's largest value and at least [`SMALLEST`], and its values over that scale in E4M3.
pub(crate) trait RowQuantize<'l, const T: Tu, Chip: M, Cluster: M, Slice: M> {
    /// Each row's scale over `Reduce` into `Rows`, in FP32.
    fn vector_row_scale<Reduce: AxisName, Rows: M>(
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
        Fresh,
        { stage::VeOrder::IntraFirst },
    >;

    /// The values divided by their rows' `scale`, cast to E4M3 back in the stream's `Time` and
    /// `Packet`.
    fn vector_quantize<Time: M, Packet: M, Scale: M>(
        self,
        scale: &VrfTensor<f32, Chip, Cluster, Slice, Scale>,
    ) -> CommitTrimTensor<'l, T, f8e4m3, Chip, Cluster, Slice, Time, Packet>;
}

impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    RowQuantize<'l, T, Chip, Cluster, Slice>
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
    fn vector_row_scale<Reduce: AxisName, Rows: M>(
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
        Fresh,
        { stage::VeOrder::IntraFirst },
    > {
        self.vector_fp_binary_with_mode(
            FpBinaryOp::MulF(FpMulAlu::Mul0),
            BinaryArgMode::Mode00,
            0.0f32,
        )
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_intra_slice_reduce::<Reduce, Rows, m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(<f8e4m3 as Finite>::LARGEST)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, SMALLEST)
    }

    fn vector_quantize<Wide: M, Lanes: M, Scale: M>(
        self,
        scale: &VrfTensor<f32, Chip, Cluster, Slice, Scale>,
    ) -> CommitTrimTensor<'l, T, f8e4m3, Chip, Cluster, Slice, Wide, Lanes> {
        self.vector_fp_div(scale)
            .vector_widen_concat::<Wide, Lanes>()
            .vector_final()
            .cast::<f8e4m3, m![{ Lanes } # 32]>()
            .commit_trim::<Lanes>()
    }
}

/// A row quantization's passes in DM: `Time` packets of eight `Reduce` values, fetched as `Fetched`
/// and split as `Narrow`; each of `Rows` takes its largest scale over `Slices`.
pub(crate) struct Quantization<Time, Narrow, Reduce, Rows, Slices, Outer, Inner, Fetched = Time>(
    PhantomData<(Time, Narrow, Reduce, Rows, Slices, Outer, Inner, Fetched)>,
);

impl<Time: M, Narrow: M, Reduce: AxisName, Rows: M, Slices: M, Outer: M, Inner: M, Fetched: M>
    Quantization<Time, Narrow, Reduce, Rows, Slices, Outer, Inner, Fetched>
{
    /// Each row's scale of `input`, as the stream a caller commits.
    pub(crate) fn scales<'l, D: MaterializableScalar + FetchCast<f32>, X: M, C: M, S: M, E: M>(
        main: &'l mut TuContext<{ Tu::Main }>,
        input: DmTensorView<'l, D, X, C, S, E>,
    ) -> CommitTrimTensor<'l, { Tu::Main }, bf16, X, C, Slices, Outer, Inner> {
        Self::stream(main, input)
            .vector_row_scale::<Reduce, Rows>()
            .vector_inter_slice_reduce::<Slices, Rows>(InterSliceReduceOpF32::Max)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<Outer, m![{ Inner } # 16]>()
            .commit_trim::<Inner>()
    }

    /// The rows of `input` over their `scale` in E4M3, as the stream a caller commits.
    pub(crate) fn rows<'l, D, X: M, C: M, S: M, E: M, Scale: M>(
        main: &'l mut TuContext<{ Tu::Main }>,
        input: DmTensorView<'l, D, X, C, S, E>,
        scale: &VrfTensor<f32, X, C, S, Scale>,
    ) -> CommitTrimTensor<'l, { Tu::Main }, f8e4m3, X, C, S, Time, m![Reduce % 8]>
    where
        D: MaterializableScalar + FetchCast<f32>,
    {
        Self::stream(main, input).vector_quantize::<Time, m![Reduce % 8], _>(scale)
    }

    fn stream<'l, D: MaterializableScalar + FetchCast<f32>, X: M, C: M, S: M, E: M>(
        main: &'l mut TuContext<{ Tu::Main }>,
        input: DmTensorView<'l, D, X, C, S, E>,
    ) -> VectorNarrowTensor<
        'l,
        { Tu::Main },
        f32,
        X,
        C,
        S,
        Narrow,
        m![Reduce % 4],
        Fresh,
        { IntraFirst },
    > {
        main.begin(input)
            .fetch::<Fetched, m![Reduce % 8]>()
            .fetch_cast::<f32>()
            .collect::<Fetched, m![Reduce % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Narrow, m![Reduce % 4]>()
    }
}

/// SiLU from a stream narrowed to four lanes: `x * sigmoid(x)` of each FP32 value, `x` held in the
/// stash. A caller rounds where its input does: FP32 sums once after, a BF16 gate also before.
pub(crate) trait Silu<
    'l,
    const T: Tu,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    const O: stage::VeOrder,
>
{
    fn vector_silu(
        self,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Spent, O>;
}

impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, const O: stage::VeOrder>
    Silu<'l, T, Chip, Cluster, Slice, Time, Packet, O>
    for VectorNarrowTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Fresh, O>
{
    fn vector_silu(
        self,
    ) -> VectorFpTensor<'l, T, f32, Chip, Cluster, Slice, Time, Packet, Spent, O> {
        self.vector_stash()
            .vector_fp_unary(FpUnaryOp::Sigmoid)
            .vector_fp_binary_with_mode(
                FpBinaryOp::MulF(FpMulAlu::Mul0),
                BinaryArgMode::Mode10,
                Stash,
            )
    }
}

/// A stream narrowed to four lanes in stage `Stage`, from its FP32 values times their block
/// scales on.
pub(crate) type Scaled<'l, const T: Tu, Stage, Chip, Cluster, Slice, Narrow, Half> = VectorTensor<
    'l,
    T,
    Stage,
    f32,
    Chip,
    Cluster,
    Slice,
    Narrow,
    Half,
    Fresh,
    { IntraFirst },
    stage::Standalone,
    { stage::Way::Way4 },
>;

/// BF16 values of codes decoded to FP32 in a stream narrowed to four lanes: each value times its
/// FP32 `scale` and the `tensor` scale's stage, rounded once, in the stream's `Time` and `Packet`.
pub(crate) trait Restore<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Narrow: M, Half: M> {
    /// `tensor` places a tensor scale: the identity for one the reduced sums take, a division for
    /// one the restored values take before they round.
    fn vector_restore<Time: M, Packet: M, Scale: M, Stage>(
        self,
        scale: &VrfTensor<f32, Chip, Cluster, Slice, Scale>,
        tensor: impl FnOnce(
            Scaled<'l, T, stage::Fp, Chip, Cluster, Slice, Narrow, Half>,
        ) -> Scaled<'l, T, Stage, Chip, Cluster, Slice, Narrow, Half>,
    ) -> CommitTrimTensor<'l, T, bf16, Chip, Cluster, Slice, Time, Packet>
    where
        Stage: stage::Stage + stage::CanTransitionTo<stage::Widen>;
}

impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Narrow: M, Half: M>
    Restore<'l, T, Chip, Cluster, Slice, Narrow, Half>
    for VectorNarrowTensor<'l, T, f32, Chip, Cluster, Slice, Narrow, Half, Fresh, { IntraFirst }>
{
    fn vector_restore<Time: M, Packet: M, Scale: M, Stage>(
        self,
        scale: &VrfTensor<f32, Chip, Cluster, Slice, Scale>,
        tensor: impl FnOnce(
            Scaled<'l, T, stage::Fp, Chip, Cluster, Slice, Narrow, Half>,
        ) -> Scaled<'l, T, Stage, Chip, Cluster, Slice, Narrow, Half>,
    ) -> CommitTrimTensor<'l, T, bf16, Chip, Cluster, Slice, Time, Packet>
    where
        Stage: stage::Stage + stage::CanTransitionTo<stage::Widen>,
    {
        tensor(self.vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale))
            .vector_widen_concat::<Time, Packet>()
            .vector_final()
            .cast::<bf16, m![{ Packet } # 16]>()
            .commit_trim::<Packet>()
    }
}
