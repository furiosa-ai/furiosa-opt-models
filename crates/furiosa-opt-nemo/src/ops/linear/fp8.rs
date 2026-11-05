//! E4M3 projections, named by the granularity of their scales.

use crate::{
    Module,
    ops::{
        layout::{Packet, axes::Chip},
        vector::scalar,
    },
};
use furiosa_opt_std::prelude::*;

use super::projection::{Block, Cluster, Reduced, Slices};

/// FP32 tensor scale in the first lane of an eight-element padded vector.
pub type Scale<Chips = Chip> = HbmTensor<f32, Chips, m![1 # 8]>;

/// One contiguous feature vector on the selected chip mapping.
pub type Vector<W, D = bf16, Chips = Chip> = HbmTensor<D, Chips, m![W]>;

type Elements<O, const ROWS: usize> = m![O % ({ Block<O> }) = { ROWS }];
type Input<I, O, D, Chips> = DmTensor<D, Chips, Cluster, Slices<I, O>, m![I % (I / 2)]>;

type Tile<I, O, const ROWS: usize, Chips> =
    DmTensor<f8e4m3, Chips, Cluster, Slices<I, O>, m![{ Elements<O, ROWS> }, I % (I / 2)]>;

/// One FP32 weight scale per output channel.
pub type ChannelScaled<In, Out> = super::Affine<
    (
        HbmTensor<f8e4m3, m![1], m![Out, In]>,
        HbmTensor<f32, m![1], m![Out, 1 # 8]>,
    ),
    (),
>;

/// Static per-tensor activation and weight scales.
#[derive(DeviceSend)]
pub struct TensorScaled<In: AxisName, Out: AxisName, D: Scalar = bf16, Chips: M = Chip> {
    /// Compile-time output dtype; the final projection casts its FP32 result to D.
    pub output: std::marker::PhantomData<D>,
    /// E4M3 coefficients in [output, input] order.
    pub weight: HbmTensor<f8e4m3, Chips, m![Out, In]>,
    /// Multiply the contracted result by this scale to restore weight units.
    pub weight_scale: Scale<Chips>,
    /// Divide BF16 activations before E4M3 quantization, then multiply the result by this scale.
    pub input_scale: Scale<Chips>,
}

/// Project one contiguous BF16 input row in output tiles of at least one eight-byte SRAM word.
impl<I: AxisName, O: AxisName, D: MaterializableScalar, Chips: M>
    Module<(&mut Device, HbmTensorView<'_, bf16, Chips, m![I]>)> for TensorScaled<I, O, D, Chips>
where
    f32: CastEngineCast<D>,
{
    type Output = Vector<O, D, Chips>;

    fn forward(
        &self,
        (device, input): (&mut Device, HbmTensorView<'_, bf16, Chips, m![I]>),
    ) -> Self::Output {
        if const { D::BITS != 8 } {
            self.tiled::<D, 4>(device, input)
        } else if const { O::SIZE.is_multiple_of(64 * 8) } {
            self.tiled::<D, 8>(device, input)
        } else {
            // Too few rows per slice for eight-byte tiles of 8-bit values: narrow after the fact.
            let wide: DmTensor<bf16, Chips, Cluster, m![1 # 256], m![O]> =
                self.tiled::<bf16, 4>(device, input).to_dm(&mut device.tdma);
            crate::ops::cast::convert(device, wide).to_hbm(&mut device.tdma)
        }
    }
}

impl<I: AxisName, O: AxisName, D: MaterializableScalar, Chips: M> TensorScaled<I, O, D, Chips>
where
    f32: CastEngineCast<D>,
{
    /// The projection in tiles of `ROWS` outputs of `T` per slice; the forward picks the tiling.
    fn tiled<T: MaterializableScalar, const ROWS: usize>(
        &self,
        device: &mut Device,
        input: HbmTensorView<'_, bf16, Chips, m![I]>,
    ) -> Vector<O, T, Chips>
    where
        f32: CastEngineCast<T>,
    {
        const {
            assert!(
                I::SIZE > 0
                    && O::SIZE > 0
                    && I::SIZE.is_multiple_of(256)
                    && O::SIZE.is_multiple_of(64 * ROWS)
                    && ROWS * T::BITS >= 64
            );
        }
        let input: DmTensor<bf16, Chips, Cluster, m![64, 1 # 2, I / (I / 2)], m![I % (I / 2)]> =
            input.to_dm(&mut device.tdma);
        let input: Input<I, O, bf16, Chips> = unsafe { input.reshape() };
        let scale: DmTensor<f32, Chips, Cluster, Slices<I, O>, m![1 # 8]> =
            self.input_scale.to_dm(&mut device.tdma);
        let scale = scalar::register(device, &scale);
        let input: Input<I, O, f8e4m3, Chips> = device
            .main
            .begin(input.view())
            .fetch::<m![I % (I / 2) / 16], m![I % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![I % (I / 2) / 8], m![I % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![I % (I / 2) / 4], m![I % 4]>()
            .vector_fp_div(&scale)
            .vector_widen_concat::<m![I % (I / 2) / 8], m![I % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![I % 8 # 32]>()
            .commit_trim::<m![I % 8]>()
            .commit();
        let input: TrfTensor<f8e4m3, Chips, Cluster, Slices<I, O>, m![1], m![I % (I / 2)]> = device
            .sub
            .begin(input.view())
            .fetch::<m![I % (I / 2) / 32], m![I % 32]>()
            .collect::<m![I % (I / 2) / 32], m![I % 32]>()
            .to_trf();
        let input_scale: DmTensor<f32, Chips, Cluster, Reduced<O>, m![1 # 8]> =
            self.input_scale.to_dm(&mut device.tdma);
        let input_scale = scalar::register(device, &input_scale);
        let weight_scale: DmTensor<f32, Chips, Cluster, Reduced<O>, m![1 # 8]> =
            self.weight_scale.to_dm(&mut device.tdma);
        let weight_scale = scalar::register(device, &weight_scale);
        let mut output: DmTensor<T, Chips, Cluster, Reduced<O>, m![O % ({ Block<O> })]> =
            DmTensor::new();
        for index in 0..(O::SIZE / (64 * ROWS)) {
            let index = if const { O::SIZE == 64 * ROWS } {
                0
            } else {
                ROWS * index
            };
            let weight: Tile<I, O, ROWS, Chips> = self.weight.view().tile::<m![O % ({ Block<O> })], ROWS, m![O / ({ Block<O> }), { Elements<O, ROWS> } # ({ Block<O> }), I]>(index).to_dm(&mut device.tdma);
            let partial: DmTensor<
                f32,
                Chips,
                Cluster,
                Slices<I, O>,
                m![{ Elements<O, ROWS> }, 1 # 8],
            > = device
                .main
                .begin(weight.view())
                .fetch::<m![{ Elements<O, ROWS> }, I % (I / 2) / 32], m![I % 32]>()
                .collect::<m![{ Elements<O, ROWS> }, I % (I / 2) / 32], m![I % 32]>()
                .contract_outer::<m![{ Elements<O, ROWS> }, I % (I / 2) / 64], m![I % 64], _, _, _>(
                    &input,
                )
                .contract_packet::<m![1]>()
                .contract_time::<Elements<O, ROWS>>()
                .contract_lane::<Elements<O, ROWS>, m![1 # 8]>(LaneMode::Interleaved)
                .commit_trim::<m![1 # 8]>()
                .commit();
            let value: DmTensor<T, Chips, Cluster, Reduced<O>, Elements<O, ROWS>> = device
                .main
                .begin(partial.view())
                .fetch::<Elements<O, ROWS>, m![1 # 8]>()
                .collect::<Elements<O, ROWS>, m![1 # 8]>()
                .vector_init()
                .vector_inter_slice_reduce::<Reduced<O>, Elements<O, ROWS>>(
                    InterSliceReduceOpF32::Add,
                )
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_scale)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .cast::<T, m![1 # ({Symbol<Packet<T>>})]>()
                .transpose::<m![1], m![{ Elements<O, ROWS> } # ({Symbol<Packet<T>>})]>()
                .commit_trim::<Elements<O, ROWS>>()
                .commit();
            value.view().to_dm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![O % ({ Block<O> })], ROWS, m![{ Elements<O, ROWS> } #{!} ({ Block<O> })]>(
                        index,
                    ),
            );
        }
        output.to_hbm(&mut device.tdma)
    }
}

impl<I: AxisName, O: AxisName, D: MaterializableScalar, Chips: M>
    Module<(&mut Device, &Vector<I, bf16, Chips>)> for TensorScaled<I, O, D, Chips>
where
    f32: CastEngineCast<D>,
{
    type Output = Vector<O, D, Chips>;

    fn forward(&self, (device, input): (&mut Device, &Vector<I, bf16, Chips>)) -> Self::Output {
        self.forward((device, input.view()))
    }
}

impl<I: AxisName, O: AxisName, Heads: AxisName, Head: AxisName, D: MaterializableScalar, Chips: M>
    Module<(&mut Device, &HbmTensor<bf16, Chips, m![Heads, Head]>)> for TensorScaled<I, O, D, Chips>
where
    f32: CastEngineCast<D>,
{
    type Output = Vector<O, D, Chips>;

    fn forward(
        &self,
        (device, input): (&mut Device, &HbmTensor<bf16, Chips, m![Heads, Head]>),
    ) -> Self::Output {
        const { assert!(I::SIZE == Heads::SIZE * Head::SIZE) };
        // SAFETY: the heads of a row are its `I` inputs in order.
        let input: HbmTensorView<'_, bf16, Chips, m![I]> = unsafe { input.view().reshape() };
        self.forward((device, input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::{Count, tests::count};
    use crate::ops::attention::geometry;
    use crate::ops::layout::{Dm, Live, axes::Sixteen};
    use crate::ops::normalization::{RowMajor, Spread, tests::random};

    axes![Input = 256, Output = 256];

    #[device(chip = 1)]
    fn projected<I: AxisName, O: AxisName, D: MaterializableScalar>(
        device: &mut Device,
        linear: &TensorScaled<I, O, D>,
        input: &Vector<I>,
    ) -> Vector<O, D>
    where
        f32: CastEngineCast<D>,
    {
        linear.forward((device, input))
    }

    /// E4M3 activation rounding and saturation match the host cast before restoring both scales.
    #[tokio::test(flavor = "current_thread")]
    async fn quantizes_at_saturation() {
        let mut device = Device::new(projected.topology()).unwrap();
        for identity in [true, false] {
            let (input_scale, weight_scale) = if identity {
                (0.25, 0.125)
            } else {
                (0.125, 0.0625)
            };
            // At scale 0.25, magnitudes around 112 straddle the finite E4M3 saturation boundary.
            let boundary = [
                -1000.0,
                -113.0,
                -112.0,
                -104.0,
                -0.015625,
                -0.001953125,
                0.0,
                0.00048828125,
                0.015625,
                0.265625,
                0.296875,
                104.0,
                112.0,
                113.0,
                1000.0,
            ];
            let input: Vec<_> = (0..Input::SIZE)
                .map(|index| {
                    bf16::from_f32(if identity {
                        boundary[index % boundary.len()]
                    } else {
                        ((index * 37 + 11) % 257) as f32 / 17.0 - 128.0 / 17.0
                    })
                })
                .collect();
            let weight: Vec<_> = (0..Output::SIZE * Input::SIZE)
                .map(|index| {
                    f8e4m3::from_f32(if identity {
                        f32::from(index / Input::SIZE == index % Input::SIZE)
                    } else {
                        let hash = (index as u32)
                            .wrapping_mul(1664525)
                            .wrapping_add(1013904223);
                        ((hash >> 24) % 31) as f32 / 16.0 - 15.0 / 16.0
                    })
                })
                .collect();
            let quantized: Vec<_> = input
                .iter()
                .map(|value| f8e4m3::from_f32(value.to_f32() / input_scale).to_f32())
                .collect();
            let expected: Vec<_> = weight
                .chunks_exact(Input::SIZE)
                .map(|row| {
                    bf16::from_f32(
                        row.iter()
                            .zip(&quantized)
                            .map(|(weight, input)| weight.to_f32() * input)
                            .sum::<f32>()
                            * input_scale
                            * weight_scale,
                    )
                })
                .collect();
            let linear: TensorScaled<Input, Output> = TensorScaled {
                output: Default::default(),
                weight: HostTensor::<f8e4m3, m![Output, Input]>::from_vec(weight)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                input_scale: HostTensor::<f32, m![1 # 8]>::from_vec([
                    input_scale,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
                weight_scale: HostTensor::<f32, m![1 # 8]>::from_vec([
                    weight_scale,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            };
            let input = HostTensor::<bf16, m![Input]>::from_vec(input)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(projected, (&mut device, &linear, &input))
                .await
                .unwrap()
                .to_host::<m![Output]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            assert_eq!(output, expected);
        }
    }

    /// Live rows of a partial launch: one whole row group and part of the next.
    const LIVE: usize = 11;

    #[device(chip = 1)]
    fn channel_rows<I: AxisName, O: AxisName>(
        device: &mut Device,
        affine: &ChannelScaled<I, O>,
        live: &Count,
        input: &HbmTensor<bf16, Chip, RowMajor<I, Sixteen>>,
    ) -> HbmTensor<bf16, Chip, RowMajor<O, Sixteen>> {
        let input: Dm<_, Spread<I, Sixteen>> = Dm {
            value: input.to_dm(&mut device.tdma),
            tiles: Live::<Sixteen>::read(live),
        };
        affine
            .forward((&mut *device, &input))
            .value
            .to_hbm(&mut device.tdma)
    }

    /// Every live row is the FP32 sum of its products with the BF16 coefficients each output row
    /// restores by its own scale, `bf16(coefficient * scale)`, within one BF16 rounding of the
    /// sum and the FP32 error of summing in another order.
    async fn lands_the_restored_rows<I: AxisName, O: AxisName>(device: &mut Device) {
        let live = count(LIVE).to_hbm(&mut device.pdma).await.unwrap();
        let inputs = random(Sixteen::SIZE * I::SIZE, 0x0b5e_ed11);
        let coefficients: Vec<f8e4m3> = random(O::SIZE * I::SIZE, 0xc0de_f8e4)
            .into_iter()
            .map(|value| f8e4m3::from_f32(value.to_f32() * 8.0))
            .collect();
        let scales: Vec<f32> = (0..O::SIZE)
            .map(|row| 1.0 / 64.0 + (row * 37 % 101) as f32 * 1.7e-4)
            .collect();
        let restored: Vec<bf16> = coefficients
            .iter()
            .enumerate()
            .map(|(index, value)| bf16::from_f32(value.to_f32() * scales[index / I::SIZE]))
            .collect();
        let affine: ChannelScaled<I, O> = super::super::Affine::new(
            (
                HostTensor::<f8e4m3, m![O, I]>::from_vec(coefficients)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                HostTensor::<f32, m![O, 1 # 8]>::from_vec(
                    scales
                        .iter()
                        .flat_map(|&scale| std::iter::once(scale).chain([0.0; 7])),
                )
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            ),
            (),
        );
        let input = HostTensor::<bf16, RowMajor<I, Sixteen>>::from_vec(inputs.clone())
            .to_hbm::<Chip, RowMajor<I, Sixteen>>(&mut device.pdma)
            .await
            .unwrap();
        let landed = launch(channel_rows, (&mut *device, &affine, &live, &input))
            .await
            .unwrap()
            .to_host::<RowMajor<O, Sixteen>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for row in 0..LIVE {
            let input = &inputs[row * I::SIZE..][..I::SIZE];
            for (column, weights) in restored.chunks_exact(I::SIZE).enumerate() {
                let (sum, magnitude) =
                    input
                        .iter()
                        .zip(weights)
                        .fold((0.0f32, 0.0f32), |(sum, magnitude), (x, w)| {
                            let term = x.to_f32() * w.to_f32();
                            (sum + term, magnitude + term.abs())
                        });
                let tolerance = sum.abs() / 128.0 + magnitude * I::SIZE as f32 * f32::EPSILON;
                let got = landed[row * O::SIZE + column].to_f32();
                assert!(
                    (got - sum).abs() <= tolerance,
                    "{} -> {}, row {row} column {column}: {got} vs {sum}",
                    I::SIZE,
                    O::SIZE,
                );
            }
        }
    }

    /// The live rows of each projection tiling land the host's sums over the restored rows.
    #[tokio::test(flavor = "current_thread")]
    async fn lands_the_restored_rows_at_each_tiling() {
        let mut device = Device::new(channel_rows.topology()).unwrap();
        lands_the_restored_rows::<geometry::Hidden<2048>, geometry::ProjectionWidth<4096>>(
            &mut device,
        )
        .await;
        lands_the_restored_rows::<geometry::Hidden<2560>, geometry::ProjectionWidth<19456>>(
            &mut device,
        )
        .await;
        lands_the_restored_rows::<geometry::Hidden<4096>, geometry::ProjectionWidth<24576>>(
            &mut device,
        )
        .await;
    }
}

mod block;
pub mod panels;

pub use block::{BlockScaled, Blocks, GROUP, Scales};
pub use panels::BlockPanels;
