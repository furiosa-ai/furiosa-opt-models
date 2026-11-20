use super::*;
use crate::Op;

/// The row scales, gathered into the slices that rescale the partial sums.
pub(super) struct Gather<P>(pub(super) PhantomData<P>);

impl<
    'a,
    Hidden: AxisName,
    QWidth: AxisName,
    Capacity: AxisName,
    D: MaterializableScalar,
    P: M + Count,
> Op<&'a ReplicatedScales<Hidden, QWidth, Capacity, D>> for Gather<P>
{
    type Output = Scales<Hidden, QWidth, Capacity, D>;

    fn forward(
        &self,
        context: &mut Device,
        scales: &'a ReplicatedScales<Hidden, QWidth, Capacity, D>,
    ) -> Self::Output {
        // SAFETY: the same slices; a quarter's replicas are named by the slices they reach.
        let quarters = unsafe {
            scales.view().reshape::<Chip, Cluster<QWidth>,
                m![Hidden / 1024, Capacity / (Capacity / 4), Hidden % 256 / 16],
                m![Capacity % (Capacity / 4)]>()
        };
        let mut gathered = Scales::<Hidden, QWidth, Capacity, D>::new();
        context
            .main
            .begin(quarters)
            .fetch::<m![Capacity % (Capacity / 4) / ({ P })], m![Capacity % ({ P })]>()
            .switch::<m![Hidden / 16], m![
                Capacity % (Capacity / 4) / ({ P }),
                Capacity / (Capacity / 4)
            ]>(SwitchConfig::Broadcast1 {
                slice1: 4,
                slice0: 16,
            })
            .collect::<m![
                Capacity % (Capacity / 4) / ({ P }),
                Capacity / (Capacity / 4)
            ], m![Capacity % ({ P })]>()
            .commit_trim::<m![Capacity % ({ P })]>()
            .commit_view(gathered.view_mut());
        gathered
    }
}

/// The partial sums of the first `waves` row groups times their row and channel scales.
pub(super) struct Rescale;

/// The partial sums, the row and channel scales, and the row groups a [`Rescale`] writes.
type Operands<'a, Hidden, QWidth, Capacity, D, S> = (
    &'a tiles::Partial<Hidden, QWidth, Capacity>,
    (
        &'a Scales<Hidden, QWidth, Capacity, D>,
        &'a Channels<Hidden, QWidth, S>,
    ),
    usize,
);

impl<
    'a,
    Hidden: AxisName,
    QWidth: AxisName,
    Capacity: AxisName,
    D: MaterializableScalar + FetchCast<f32>,
    S: MaterializableScalar + FetchCast<f32>,
> Op<Operands<'a, Hidden, QWidth, Capacity, D, S>> for Rescale
{
    type Output = tiles::Partial<Hidden, QWidth, Capacity>;

    fn forward(
        &self,
        context: &mut Device,
        (partial, (scales, channels), waves): Operands<'a, Hidden, QWidth, Capacity, D, S>,
    ) -> Self::Output {
        let (row, channel) = factors(context, scales, channels);
        let mut output = tiles::Partial::new();
        context
            .main
            .begin(partial.view())
            .fetch::<m![
                Capacity % (Capacity / 4) / 8,
                Capacity / (Capacity / 4),
                Capacity % 8,
                Hidden % 16 / 8
            ], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![
                Capacity % (Capacity / 4) / 8,
                Capacity / (Capacity / 4),
                Capacity % 8,
                Hidden % 16 / 8
            ], m![Hidden % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![
                Capacity % (Capacity / 4) / 8,
                Capacity / (Capacity / 4),
                Capacity % 8,
                Hidden % 16 / 4
            ], m![Hidden % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &row)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &channel)
            .vector_widen_concat::<m![
                Capacity % (Capacity / 4) / 8,
                Capacity / (Capacity / 4),
                Capacity % 8,
                Hidden % 16 / 8
            ], m![Hidden % 8]>()
            .vector_final()
            .cast::<bf16, m![Hidden % 8 # 16]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit_prefix::<m![Capacity % (Capacity / 4) / 8], _, _>(output.view_mut(), waves);
        output
    }
}

fn factors<
    H: AxisName,
    Q: AxisName,
    R: AxisName,
    D: MaterializableScalar + FetchCast<f32>,
    S: MaterializableScalar + FetchCast<f32>,
>(
    context: &mut Device,
    scales: &Scales<H, Q, R, D>,
    channels: &Channels<H, Q, S>,
) -> (
    VrfTensor<f32, Chip, Cluster<Q>, m![H / 16], m![R / (R / 4), R % (R / 4)]>,
    VrfTensor<f32, Chip, Cluster<Q>, m![H / 16], m![H % 16]>,
) {
    let row: VrfTensor<f32, Chip, Cluster<Q>, m![H / 16], m![R / (R / 4), R % (R / 4)]> = context
        .sub
        .begin(scales.view())
        .fetch::<m![R / (R / 4), R % (R / 4) / 8], m![R % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![R / (R / 4), R % (R / 4) / 8], m![R % 8]>()
        .to_vrf();
    let channel: VrfTensor<f32, Chip, Cluster<Q>, m![H / 16], m![H % 16]> = context
        .sub
        .begin(channels.view())
        .fetch::<m![H % 16 / 8], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 16 / 8], m![H % 8]>()
        .to_vrf();
    (row, channel)
}

pub(super) fn decode<H: AxisName, Q: AxisName, R: AxisName>(
    device: &mut Device,
    input: &channel::DecodePartial<H, Q, R>,
    scales: &Scales<H, Q, R, f32>,
    channels: &Channels<H, Q, bf16>,
    waves: usize,
) -> tiles::Partial<H, Q, R> {
    let (row, channel) = factors(device, scales, channels);
    // SAFETY: Wave is the outer local row coordinate in both views.
    let partial = unsafe {
        input.view().reshape::<Chip,Cluster<Q>,m![H/16],m![R%(R/4)/8,H%16,R/(R/4),R%8]>()
    };
    let mut output = tiles::Partial::new();
    device
        .main
        .begin(partial)
        .fetch::<m![R % (R / 4) / 8, R / (R / 4), H % 16], m![R % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![R % (R / 4) / 8, R / (R / 4), H % 16], m![R % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R % (R / 4) / 8, R / (R / 4), H % 16, R % 8 / 4], m![R % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &row)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &channel)
        .vector_widen_concat::<m![R % (R / 4) / 8, R / (R / 4), H % 16], m![R % 8]>()
        .vector_final()
        .cast::<bf16, m![R%8#16]>()
        .transpose::<m![R % (R / 4) / 8, R / (R / 4), H % 16 / 4, R % 8], m![H%4#16]>()
        .commit_trim::<m![H % 4]>()
        .commit_prefix::<m![R % (R / 4) / 8], _, _>(output.view_mut(), waves);
    output
}
