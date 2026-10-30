//! Gains of an RMS norm whose epsilon is data: a learned offset from one, or none.

use super::{Chunk, Cluster, FeatureSlices, Rms, Spread, rms_norm};
use crate::load::Load;
use crate::{Op, tensor::axis::Single};
use furiosa_opt_std::prelude::*;

/// A learned gain stored as its offset from one: the norm multiplies by `1 + w`, summed in FP32.
#[derive(DeviceSend, Load)]
pub struct Centered<W>(pub W);

/// The FP32 epsilon of a norm whose type does not fix it, in every cell of a padded vector.
pub type Epsilon<Chips = m![1]> = HbmTensor<f32, Chips, m![1 # 8]>;

/// What multiplies each normalized feature, in FP32 before the one rounding to BF16.
pub(crate) trait Gain<Width: AxisName, Chips: M> {
    /// `input / sqrt(mean(input²) + epsilon)` times this gain, per row.
    fn normalize<Rows: AxisName>(
        &self,
        device: &mut Device,
        input: &Spread<Width, Rows, Chips>,
        epsilon: &Epsilon<Chips>,
    ) -> Spread<Width, Rows, Chips>;
}

impl<Width: AxisName, Chips: M> Gain<Width, Chips> for Centered<HbmTensor<bf16, Chips, m![Width]>> {
    fn normalize<Rows: AxisName>(
        &self,
        device: &mut Device,
        input: &Spread<Width, Rows, Chips>,
        epsilon: &Epsilon<Chips>,
    ) -> Spread<Width, Rows, Chips> {
        let offset: DmTensor<
            bf16,
            Chips,
            Cluster,
            FeatureSlices<Width>,
            m![Width % ({ Symbol<Chunk<Width>> })],
        > = self.0.to_dm(&mut device.tdma);
        // The FP32 gain `1 + offset` of every feature, formed before any rounding.
        let gain: rms_norm::Features<Width, Chips> = device
            .sub
            .begin(offset.view())
            .fetch::<m![Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![Width % ({ Symbol<Chunk<Width>> }) / 8, Width % 8 / 4 % 2],
                m![Width % 4],
            >()
            .vector_fp_binary(FpBinaryOp::AddF, 1.0_f32)
            .vector_widen_concat::<m![Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
            .vector_final()
            .to_vrf();
        rms_norm::normalize(device, input, epsilon, &gain)
    }
}

/// Unit gain: the normalized features themselves.
impl<Width: AxisName, Chips: M> Gain<Width, Chips> for () {
    fn normalize<Rows: AxisName>(
        &self,
        device: &mut Device,
        input: &Spread<Width, Rows, Chips>,
        epsilon: &Epsilon<Chips>,
    ) -> Spread<Width, Rows, Chips> {
        rms_norm::normalize(device, input, epsilon, 1.0_f32)
    }
}

impl<Width: AxisName, Chips: M, Weight: Gain<Width, Chips>> Op<&HbmTensor<bf16, Chips, m![Width]>>
    for Rms<Weight, Epsilon<Chips>>
{
    type Output = HbmTensor<bf16, Chips, m![Width]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &HbmTensor<bf16, Chips, m![Width]>,
    ) -> Self::Output {
        let input: Spread<Width, Single, Chips> = input.to_dm(&mut device.tdma);
        self.weight
            .normalize(&mut *device, &input, &self.epsilon)
            .to_hbm(&mut device.tdma)
    }
}

impl<Width: AxisName, Rows: AxisName, Chips: M, Weight: Gain<Width, Chips>>
    Op<&HbmTensor<bf16, Chips, m![Rows, Width]>> for Rms<Weight, Epsilon<Chips>>
{
    type Output = HbmTensor<bf16, Chips, m![Rows, Width]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &HbmTensor<bf16, Chips, m![Rows, Width]>,
    ) -> Self::Output {
        let input: Spread<Width, Rows, Chips> = input.to_dm(&mut device.tdma);
        self.weight
            .normalize(&mut *device, &input, &self.epsilon)
            .to_hbm(&mut device.tdma)
    }
}

#[cfg(test)]
mod tests {
    use super::{Centered, Epsilon};
    use crate::ops::{attention::geometry::Hidden, normalization::tests::random};
    use crate::tests::DEVICE;
    use crate::{Op, ops::normalization::Rms};
    use furiosa_opt_std::{prelude::*, runtime::launch};

    type Width = Hidden<1024>;
    type Row = HbmTensor<bf16, m![1], m![Width]>;

    #[device(chip = 1)]
    fn centered(device: &mut Device, norm: &Rms<Centered<Row>, Epsilon>, input: &Row) -> Row {
        norm.forward(device, input)
    }

    /// `BF16(x * gain(w) / sqrt(mean(x²) + epsilon))` of every feature, in FP32.
    fn reference(
        input: &[bf16],
        offsets: &[bf16],
        gain: &dyn Fn(f32) -> f32,
        epsilon: f32,
    ) -> Vec<u16> {
        let mean = input.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>() / input.len() as f32;
        let inverse = 1.0 / (mean + epsilon).sqrt();
        input
            .iter()
            .zip(offsets)
            .map(|(v, w)| bf16::from_f32(v.to_f32() * inverse * gain(w.to_f32())).to_bits())
            .collect()
    }

    /// A centered norm multiplies by `1 + w` summed in FP32 and adds its own epsilon: rounding
    /// the gain to BF16 first, or adding the other epsilon, gives other features.
    #[tokio::test]
    async fn centered_gain_rounds_once() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(centered.topology()).unwrap();
        let width = Width::SIZE;
        let offsets = random(width, 0x2545_f491_4f6c_dd1d);
        for (epsilon, other, scale) in [(1e-5, 1e-8, 1e-3), (1e-8, 1e-5, 1e-3), (1e-5, 1e-8, 1.0)] {
            let input: Vec<bf16> = random(width, 0x9e37_79b9_7f4a_7c15)
                .into_iter()
                .map(|v| bf16::from_f32(v.to_f32() * scale))
                .collect();
            let norm = Rms::new(
                Centered(
                    HostTensor::<bf16, m![Width]>::from_vec(offsets.clone())
                        .to_hbm(&mut device.pdma)
                        .await
                        .unwrap(),
                ),
                HostTensor::<f32, m![1 # 8]>::from_vec(vec![epsilon; 8])
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            );
            let tensor = HostTensor::<bf16, m![Width]>::from_vec(input.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let output: Vec<u16> = launch(centered, (&mut device, &norm, &tensor))
                .await
                .unwrap()
                .to_host::<m![Width]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec()
                .into_iter()
                .map(bf16::to_bits)
                .collect();
            let same = |gain: &dyn Fn(f32) -> f32, epsilon: f32| {
                let expected = reference(&input, &offsets, gain, epsilon);
                output
                    .iter()
                    .zip(expected)
                    .filter(|(a, b)| **a == *b)
                    .count()
            };
            let summed = same(&|w| 1.0 + w, epsilon);
            assert!(summed >= width - width / 256, "{summed} hold the FP32 gain");
            let rounded = same(&|w| bf16::from_f32(1.0 + w).to_f32(), epsilon);
            assert!(
                rounded <= width - width / 8,
                "{rounded} hold the rounded gain"
            );
            if scale < 1.0 {
                let swapped = same(&|w| 1.0 + w, other);
                assert!(swapped <= width / 8, "{swapped} hold epsilon {other}");
            }
        }
    }
}
