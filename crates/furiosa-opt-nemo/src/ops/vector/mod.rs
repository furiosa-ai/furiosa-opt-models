//! Vector-engine passes that several ops share, each emitting the same stages at every caller.

pub(crate) mod scalar;

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

/// The statistic an RMS norm takes the square root of, from a stream narrowed to four lanes.
pub(crate) trait MeanSquare<'l, const T: Tu, Chip: M, Cluster: M, Slice: M> {
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
