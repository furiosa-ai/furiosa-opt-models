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
    /// The token-dynamic FP8 scale, retained in FP32 through quantization and contraction.
    /// Source: <https://github.com/vllm-project/vllm/blob/dc1b4a6f1300003ae27f033afbdff5e2683721ce/csrc/quantization/fp8/common.cuh>.
    pub(crate) fn scales_f32<
        'l,
        D: MaterializableScalar + FetchCast<f32>,
        X: M,
        C: M,
        S: M,
        E: M,
    >(
        main: &'l mut TuContext<{ Tu::Main }>,
        input: DmTensorView<'l, D, X, C, S, E>,
    ) -> CommitTrimTensor<
        'l,
        { Tu::Main },
        f32,
        X,
        C,
        Slices,
        m![{ Outer }, { Inner } / 2],
        m![{ Inner } % 2],
    > {
        main.begin(input)
            .fetch::<Fetched, m![Reduce % 8]>()
            .fetch_cast::<f32>()
            .collect::<Fetched, m![Reduce % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpF32::BitAnd, f32::from_bits(0x7fff_ffff))
            .vector_narrow_split::<Narrow, m![Reduce % 4]>()
            .vector_intra_slice_reduce::<Reduce, Rows, m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_fp_div(448.0_f32)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Max, 1.0_f32 / (448.0 * 512.0))
            .vector_inter_slice_reduce::<Slices, Rows>(InterSliceReduceOpF32::Max)
            .vector_final()
            .transpose::<m![{ Outer }, { Inner } / 2], m![{Inner} % 2 # 8]>()
            .commit_trim::<m![{ Inner } % 2]>()
    }

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

/// Agree token scales when the two clusters own disjoint input features of the same rows.
/// The input already contains a maximum over each cluster's complete input half. Only scale
/// metadata crosses clusters; activations retain their contraction ownership.
pub(crate) fn cluster_max<X: M, C: M, S: M, R: M>(
    device: &mut Device,
    local: &DmTensor<f32, X, C, S, R>,
    rows: usize,
) -> DmTensor<f32, X, C, S, R> {
    const {
        assert!(C::SIZE == 2 && R::SIZE.is_multiple_of(8));
    }
    let remote: DmTensor<f32, X, C, S, R> = local.view().cluster_swap().to_dm(&mut device.tdma);
    let other: VrfTensor<f32, X, C, S, R> = device
        .sub
        .begin(remote.view())
        .fetch::<m![{ R } / 8], m![{ R } % 8]>()
        .collect::<m![{ R } / 8], m![{ R } % 8]>()
        .to_vrf();
    let mut output = DmTensor::new();
    device
        .main
        .begin(local.view())
        .fetch::<m![{ R } / 8], m![{ R } % 8]>()
        .collect::<m![{ R } / 8], m![{ R } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Max, &other)
        .vector_final()
        .commit_trim::<m![{ R } % 8]>()
        .commit_prefix::<m![{ R } / 8], _, _>(output.view_mut(), rows.div_ceil(8));
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![Rows = 8, Hidden = 4096];
    type Q = Quantization<
        m![Rows, Hidden % 64 / 8],
        m![Rows, Hidden % 64 / 4],
        Hidden,
        m![Rows],
        m![4, 1 # 64],
        m![Rows / 8],
        m![Rows % 8],
    >;

    #[device(chip = 1)]
    fn token_quantized(
        device: &mut Device,
        input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
    ) -> (
        HbmTensor<f8e4m3, m![1], m![Rows, Hidden]>,
        HbmTensor<f32, m![1], m![Hidden / 64, Rows]>,
    ) {
        let input: DmTensor<bf16, m![1], m![2], m![4, Hidden / 64], m![Rows, Hidden % 64]> =
            input.to_dm(&mut device.tdma);
        let mut scales: DmTensor<f32, m![1], m![2], m![4, 1 # 64], m![Rows]> = DmTensor::new();
        Q::scales_f32(&mut device.main, input.view()).commit_view(scales.view_mut());
        // SAFETY: the maximum reduction broadcasts each row's scale to all 64 feature slices.
        let copies = unsafe {
            scales
                .view()
                .reshape::<m![1], m![2], m![4, Hidden / 64], m![Rows]>()
        };
        let factors: VrfTensor<f32, m![1], m![2], m![4, Hidden / 64], m![Rows]> = device
            .sub
            .begin(copies.clone())
            .fetch::<m![Rows / 8], m![Rows % 8]>()
            .collect::<m![Rows / 8], m![Rows % 8]>()
            .to_vrf();
        let mut codes: DmTensor<f8e4m3, m![1], m![2], m![4, Hidden / 64], m![Rows, Hidden % 64]> =
            DmTensor::new();
        Q::rows(&mut device.main, input.view(), &factors).commit_view(codes.view_mut());
        let mut stored = HbmTensor::new();
        copies.to_hbm_view(&mut device.tdma, stored.view_mut());
        (codes.to_hbm(&mut device.tdma), stored)
    }

    /// Each token uses its complete input's FP32 scale, including zero/tiny rows and a largest
    /// magnitude in the final feature slice; FP8 codes follow true division at that scale.
    #[tokio::test(flavor = "current_thread")]
    async fn token_scales_preserve_fp32_and_the_floor() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(token_quantized.topology()).unwrap();
        let magnitudes = [0.0, 1e-8, 0.001, 1.25, 7.5, 33.0, 448.0, 1e30];
        let input: Vec<_> = (0..Rows::SIZE * Hidden::SIZE)
            .map(|at| {
                let (row, feature) = (at / Hidden::SIZE, at % Hidden::SIZE);
                bf16::from_f32(
                    magnitudes[row]
                        * if feature == Hidden::SIZE - 1 {
                            -1.0
                        } else {
                            ((feature % 97) as f32 - 48.0) / 64.0
                        },
                )
            })
            .collect();
        let source = HostTensor::<bf16, m![Rows, Hidden]>::from_vec(input.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let (codes, scales) = launch(token_quantized, (&mut device, &source))
            .await
            .unwrap();
        let codes = codes
            .to_host::<m![Rows, Hidden]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let scales = scales
            .to_host::<m![Hidden / 64, Rows]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for row in 0..Rows::SIZE {
            let values = &input[row * Hidden::SIZE..(row + 1) * Hidden::SIZE];
            let maximum = values
                .iter()
                .map(|v| v.to_f32().abs())
                .fold(0.0_f32, f32::max);
            let scale = (maximum / 448.0).max(1.0 / (448.0 * 512.0));
            for slice in 0..64 {
                assert_eq!(
                    scales[slice * Rows::SIZE + row].to_bits(),
                    scale.to_bits(),
                    "row {row}, slice {slice}"
                );
            }
            for (column, value) in values.iter().enumerate() {
                let expected = f8e4m3::from_f32(value.to_f32() / scale);
                assert_eq!(
                    codes[row * Hidden::SIZE + column].to_f32(),
                    expected.to_f32(),
                    "row {row}, column {column}"
                );
            }
        }
    }
}
