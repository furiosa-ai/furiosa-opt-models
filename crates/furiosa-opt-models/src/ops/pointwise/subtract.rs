//! Elementwise differences, computed in FP32 and rounded once to the source element type.

use super::{Float, Rows, Window};
use crate::Op;
use crate::load::Load;
use crate::tensor::memory::residency::Residency;
use furiosa_opt_std::prelude::*;

/// `left - right` in the operands' common layout and precision.
#[derive(Clone, Default, DeviceSend, Load, Residency)]
pub struct Subtract;

/// Each difference is narrowed before it reaches HBM or another pointwise operation.
impl<D: Float, S: M + 'static, Chips: M> Op<(&HbmTensor<D, Chips, S>, &HbmTensor<D, Chips, S>)>
    for Subtract
{
    type Output = HbmTensor<D, Chips, S>;

    fn forward(
        &self,
        device: &mut Device,
        (left, right): (&HbmTensor<D, Chips, S>, &HbmTensor<D, Chips, S>),
    ) -> Self::Output {
        Window::<S>::combine::<D, Chips>(device, left, right, FpBinaryOp::SubF)
    }
}

/// One window of each operand. The BF16 output is materialized before a following operation
/// reads it; a reduction of magnitudes must not sum unrounded FP32 differences.
impl<D: Float, S: M + 'static, Chips: M> Op<&(Rows<D, S, Chips>, Rows<D, S, Chips>)> for Subtract {
    type Output = Rows<D, S, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (left, right): &(Rows<D, S, Chips>, Rows<D, S, Chips>),
    ) -> Self::Output {
        Window::<S>::combine_tile::<D, Chips>(device, left, right, FpBinaryOp::SubF)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::Subtract;
    use crate::Op;
    use crate::tensor::axis::{Eight, Single};
    use crate::tensor::parallel::Shard;
    use furiosa_opt_std::{prelude::*, runtime::launch};

    // Preserve the original numerical test geometry: three windows of 224 slices x 512 elements.
    axes![Rows = 64, Features = 5_376, Tail = 520];
    pub(crate) type ReferenceRows = Rows;
    pub(crate) type ReferenceFeatures = Features;
    pub(crate) type Chips = m![{ Symbol<Shard<4>> }];
    pub(crate) type Part<R = Rows, F = Features> = HbmTensor<bf16, Chips, m![R, F]>;
    pub(crate) type Host<R = Rows, F = Features> = HostTensor<bf16, m![{ Symbol<Shard<4>> }, R, F]>;

    /// Seeded BF16 operands include controlled equal and opposite values.
    pub(crate) fn terms(elements: usize) -> (Vec<bf16>, Vec<bf16>) {
        let count = 4 * elements;
        let left = crate::ops::normalization::tests::random(count, 7);
        let step = crate::ops::normalization::tests::random(count, 11);
        let right = left
            .iter()
            .zip(&step)
            .enumerate()
            .map(|(index, (left, step))| match index % 5 {
                0 => *left,
                1 => bf16::from_f32(-left.to_f32()),
                _ => bf16::from_f32(left.to_f32() * 4.0 + step.to_f32() / 8.0),
            })
            .collect();
        let left = left.iter().map(|left| bf16::from_f32(left.to_f32() * 4.0));
        (left.collect(), right)
    }

    /// Extra short cases cover rounding ties and distinct per-chip totals independently of
    /// the original seeded workload.
    pub(crate) fn rounding_terms(elements: usize) -> (Vec<bf16>, Vec<bf16>) {
        (0..4 * elements)
            .map(|index| {
                // Powers of two keep the BF16 ties while making every chip's totals distinct.
                let scale = (1 << (index / elements)) as f32;
                let at = index % elements;
                let left = bf16::from_f32(((at * 37 % 8192) as f32 - 4096.0) / 512.0 * scale);
                let right = match index % 5 {
                    0 => left,
                    1 => bf16::from_f32(-left.to_f32()),
                    // Both signs of 1 + 1/256 round to unit magnitude before abs/sum.
                    2 => {
                        let sign = if index % 10 == 2 { 1.0 } else { -1.0 };
                        return (
                            bf16::from_f32(sign * scale),
                            bf16::from_f32(-sign * scale / 256.0),
                        );
                    }
                    3 => bf16::from_f32(left.to_f32() + scale / 512.0),
                    _ => bf16::from_f32(left.to_f32() * 0.875 - scale / 16.0),
                };
                (left, right)
            })
            .unzip()
    }

    #[device(chip = 4)]
    fn difference<R: AxisName, F: AxisName>(
        device: &mut Device,
        left: &Part<R, F>,
        right: &Part<R, F>,
    ) -> Part<R, F> {
        Subtract.forward(device, (left, right))
    }

    async fn matches<R: AxisName, F: AxisName>((left, right): (Vec<bf16>, Vec<bf16>)) {
        let mut device = Device::new(difference.topology()).unwrap();
        let expected: Vec<u16> = left
            .iter()
            .zip(&right)
            .map(|(left, right)| bf16::from_f32(left.to_f32() - right.to_f32()).to_bits())
            .collect();
        let left: Part<R, F> = Host::<R, F>::from_vec(left)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let right: Part<R, F> = Host::<R, F>::from_vec(right)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let got = launch(difference, (&mut device, &left, &right))
            .await
            .unwrap()
            .to_host::<m![{ Symbol<Shard<4>> }, R, F]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (index, (got, expected)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                *expected,
                "{} x {} difference at element {index}",
                R::SIZE,
                F::SIZE
            );
        }
    }

    /// Every element on every chip is its FP32 difference rounded to BF16 once, including a
    /// single vector, a short window ending eight elements past 512, and three full windows.
    #[tokio::test(flavor = "current_thread")]
    async fn rounds_each_difference_once() {
        let _lease = crate::tests::DEVICE.lock().await;
        matches::<Rows, Features>(terms(Rows::SIZE * Features::SIZE)).await;
        matches::<Single, Eight>(rounding_terms(Eight::SIZE)).await;
        matches::<Single, Tail>(rounding_terms(Tail::SIZE)).await;
    }
}
