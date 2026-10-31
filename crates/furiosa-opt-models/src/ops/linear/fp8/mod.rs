//! E4M3 projections, named by the granularity of their scales.

use core::borrow::Borrow;
use core::marker::PhantomData;

use crate::load::Load;
use crate::ops::pointwise::cast::convert;
use crate::tensor::form::BlockScaled;
use crate::{
    Op,
    tensor::{parallel::Chip, scalar},
};
use furiosa_opt_std::prelude::*;

use super::kernel::Contraction;
use super::projection::{Block, Cluster, Partials, Reduced, Slices};

/// FP32 tensor scale in the first lane of an eight-element padded vector.
pub(crate) type Scale<Chips = Chip> = HbmTensor<f32, Chips, m![1 # 8]>;

/// One contiguous feature vector on the selected chip mapping.
pub(crate) type Vector<W, D = bf16, Chips = Chip> = HbmTensor<D, Chips, m![W]>;

type Elements<O, const ROWS: usize> = m![O % ({ Block<O> }) = { ROWS }];
type Input<I, O, D, Chips> = DmTensor<D, Chips, Cluster, Slices<I, O>, m![I % (I / 2)]>;

type Tile<I, O, const ROWS: usize, Chips> =
    DmTensor<f8e4m3, Chips, Cluster, Slices<I, O>, m![{ Elements<O, ROWS> }, I % (I / 2)]>;

/// One FP32 weight scale per output channel: a block of one output row by every input, under no
/// contract; the codes are held as `W` and the scales as `T`.
pub(crate) type ChannelScaled<
    In,
    Out,
    W = HbmTensor<f8e4m3, Chip, m![Out, In]>,
    T = HbmTensor<f32, Chip, m![Out, 1 # 8]>,
> = BlockScaled<In, Out, f8e4m3, f32, m![Out, 1 # 8], (), Chip, W, T>;

/// Static per-tensor activation and weight scales: its checkpoint's `weight`, `weight_scale` and
/// `input_scale`, the coefficients held as `W` and the scales as `S`, owned or borrowed.
#[derive(DeviceSend, Load)]
pub struct TensorScaled<
    In: AxisName,
    Out: AxisName,
    D: Scalar = bf16,
    Chips: M = Chip,
    W = HbmTensor<f8e4m3, Chips, m![Out, In]>,
    S = Scale<Chips>,
> {
    /// Compile-time output dtype; the final projection casts its FP32 result to D.
    pub output: PhantomData<D>,
    /// E4M3 coefficients in [output, input] order.
    pub weight: W,
    /// Multiply the contracted result by this scale to restore weight units.
    pub weight_scale: S,
    /// Divide BF16 activations before E4M3 quantization, then multiply the result by this scale.
    pub input_scale: S,
    pub shape: PhantomData<fn() -> (In, Out, Chips)>,
}

/// Project one contiguous BF16 input row in output tiles of at least one eight-byte SRAM word.
impl<I: AxisName, Out: AxisName, D: MaterializableScalar, Chips: M, W, S>
    Op<HbmTensorView<'_, bf16, Chips, m![I]>> for TensorScaled<I, Out, D, Chips, W, S>
where
    f32: CastEngineCast<D>,
    W: Borrow<HbmTensor<f8e4m3, Chips, m![Out, I]>>,
    S: Borrow<Scale<Chips>>,
{
    type Output = Vector<Out, D, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        input: HbmTensorView<'_, bf16, Chips, m![I]>,
    ) -> Self::Output {
        if const { D::BITS != 8 } {
            self.tiled::<D, 4>(device, input)
        } else if const { Out::SIZE.is_multiple_of(64 * 8) } {
            self.tiled::<D, 8>(device, input)
        } else {
            // Too few rows per slice for eight-byte tiles of 8-bit values: narrow after the fact.
            let wide: DmTensor<bf16, Chips, Cluster, m![1 # 256], m![Out]> =
                self.tiled::<bf16, 4>(device, input).to_dm(&mut device.tdma);
            convert(device, wide).to_hbm(&mut device.tdma)
        }
    }
}

impl<I: AxisName, Out: AxisName, D: MaterializableScalar, Chips: M, W, S>
    TensorScaled<I, Out, D, Chips, W, S>
where
    f32: CastEngineCast<D>,
    W: Borrow<HbmTensor<f8e4m3, Chips, m![Out, I]>>,
    S: Borrow<Scale<Chips>>,
{
    /// The projection in tiles of `ROWS` outputs of `T` per slice; the forward picks the tiling.
    fn tiled<T: MaterializableScalar, const ROWS: usize>(
        &self,
        device: &mut Device,
        input: HbmTensorView<'_, bf16, Chips, m![I]>,
    ) -> Vector<Out, T, Chips>
    where
        f32: CastEngineCast<T>,
    {
        const {
            assert!(
                I::SIZE > 0
                    && Out::SIZE > 0
                    && I::SIZE.is_multiple_of(256)
                    && Out::SIZE.is_multiple_of(64 * ROWS)
                    && ROWS * T::BITS >= 64
            );
        }
        let (weight, weight_scale, input_scale): (
            &HbmTensor<f8e4m3, Chips, m![Out, I]>,
            &Scale<Chips>,
            &Scale<Chips>,
        ) = (
            self.weight.borrow(),
            self.weight_scale.borrow(),
            self.input_scale.borrow(),
        );
        let input: DmTensor<bf16, Chips, Cluster, m![64, 1 # 2, I / (I / 2)], m![I % (I / 2)]> =
            input.to_dm(&mut device.tdma);
        let input: Input<I, Out, bf16, Chips> = unsafe { input.reshape() };
        let scale: DmTensor<f32, Chips, Cluster, Slices<I, Out>, m![1 # 8]> =
            input_scale.to_dm(&mut device.tdma);
        let scale = scalar::register(device, &scale);
        let input: Input<I, Out, f8e4m3, Chips> = device
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
        let input: TrfTensor<f8e4m3, Chips, Cluster, Slices<I, Out>, m![1], m![1, I % (I / 2)]> =
            device
                .sub
                .begin(input.view())
                .fetch::<m![I % (I / 2) / 32], m![I % 32]>()
                .collect::<m![I % (I / 2) / 32], m![I % 32]>()
                .to_trf();
        let input_scale: DmTensor<f32, Chips, Cluster, Reduced<Out>, m![1 # 8]> =
            input_scale.to_dm(&mut device.tdma);
        let input_scale = scalar::register(device, &input_scale);
        let weight_scale: DmTensor<f32, Chips, Cluster, Reduced<Out>, m![1 # 8]> =
            weight_scale.to_dm(&mut device.tdma);
        let weight_scale = scalar::register(device, &weight_scale);
        let mut output: DmTensor<T, Chips, Cluster, Reduced<Out>, m![Out % ({ Block<Out> })]> =
            DmTensor::new();
        for index in 0..(Out::SIZE / (64 * ROWS)) {
            let index = if const { Out::SIZE == 64 * ROWS } {
                0
            } else {
                ROWS * index
            };
            let weight: Tile<
                I,
                Out,
                ROWS,
                Chips,
            > = weight.view().tile::<
                m![Out % ({ Block<Out> })],
                ROWS,
                m![Out / ({ Block<Out> }), { Elements<Out, ROWS> } # ({ Block<Out> }), I],
            >(index).to_dm(&mut device.tdma);
            let partial: DmTensor<
                f32,
                Chips,
                Cluster,
                Slices<I, Out>,
                m![{ Elements<Out, ROWS> }, 1 # 8],
            > = Contraction::<I, Out>::sums(device, &weight, &input)
                .commit_trim::<m![1 # 8]>()
                .commit();
            let value: DmTensor<T, Chips, Cluster, Reduced<Out>, Elements<Out, ROWS>> =
                Partials::<Out, Elements<Out, ROWS>>::restored(
                    &mut device.main,
                    partial.view(),
                    |sums| {
                        sums.vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale)
                            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_scale)
                    },
                )
                .commit();
            value.view().to_dm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<
                        m![Out % ({ Block<Out> })],
                        ROWS,
                        m![{ Elements<Out, ROWS> } #{!} ({ Block<Out> })],
                    >(
                        index,
                    ),
            );
        }
        output.to_hbm(&mut device.tdma)
    }
}

impl<I: AxisName, Out: AxisName, D: MaterializableScalar, Chips: M, W, S>
    Op<&Vector<I, bf16, Chips>> for TensorScaled<I, Out, D, Chips, W, S>
where
    f32: CastEngineCast<D>,
    W: Borrow<HbmTensor<f8e4m3, Chips, m![Out, I]>>,
    S: Borrow<Scale<Chips>>,
{
    type Output = Vector<Out, D, Chips>;

    fn forward(&self, device: &mut Device, input: &Vector<I, bf16, Chips>) -> Self::Output {
        self.forward(device, input.view())
    }
}

impl<
    I: AxisName,
    Out: AxisName,
    Heads: AxisName,
    Head: AxisName,
    D: MaterializableScalar,
    Chips: M,
    W,
    S,
> Op<&HbmTensor<bf16, Chips, m![Heads, Head]>> for TensorScaled<I, Out, D, Chips, W, S>
where
    f32: CastEngineCast<D>,
    W: Borrow<HbmTensor<f8e4m3, Chips, m![Out, I]>>,
    S: Borrow<Scale<Chips>>,
{
    type Output = Vector<Out, D, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        input: &HbmTensor<bf16, Chips, m![Heads, Head]>,
    ) -> Self::Output {
        const { assert!(I::SIZE == Heads::SIZE * Head::SIZE) };
        // SAFETY: the heads of a row are its `I` inputs in order.
        let input: HbmTensorView<'_, bf16, Chips, m![I]> = unsafe { input.view().reshape() };
        self.forward(device, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::attention::geometry;
    use crate::ops::normalization::{RowMajor, Spread, tests::random};
    use crate::ops::swiglu;
    use crate::tensor::form::fp8::{Finite, Granularity};
    use crate::tensor::{axis::Sixteen, live::Live};
    use crate::tensor::{live::tests::count, scalar::Value};
    use std::iter;

    axes![Input = 256, Output = 256];

    /// Live rows of a partial launch: one whole row group and part of the next.
    const LIVE: usize = 11;

    #[device(chip = 1)]
    fn projected<I: AxisName, O: AxisName, D: MaterializableScalar>(
        device: &mut Device,
        linear: &TensorScaled<I, O, D>,
        input: &Vector<I>,
    ) -> Vector<O, D>
    where
        f32: CastEngineCast<D>,
    {
        linear.forward(device, input)
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
                shape: PhantomData,
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

    #[device(chip = 1)]
    fn channel_rows<I: AxisName, O: AxisName>(
        device: &mut Device,
        affine: &ChannelScaled<I, O>,
        live: &Value,
        input: &HbmTensor<bf16, Chip, RowMajor<I, Sixteen>>,
    ) -> HbmTensor<bf16, Chip, RowMajor<O, Sixteen>> {
        let value = input.to_dm(&mut device.tdma);
        let input: Live<Spread<I, Sixteen>, Sixteen> = Live::<(), Sixteen>::read(live).with(value);
        affine
            .forward(&mut *device, &input)
            .value()
            .to_hbm(&mut device.tdma)
    }

    /// Every live row is the FP32 sum of its products with the BF16 coefficients each output row
    /// restores by its own scale, `bf16(coefficient * scale)`, within one BF16 rounding of the
    /// sum and the FP32 error of summing in another order.
    async fn writes_the_restored_rows<I: AxisName, O: AxisName>(
        device: &mut Device,
        coefficients: Vec<f8e4m3>,
        scales: Vec<f32>,
    ) {
        let live = count(LIVE).to_hbm(&mut device.pdma).await.unwrap();
        let inputs = random(Sixteen::SIZE * I::SIZE, 0x0b5e_ed11);
        let restored: Vec<bf16> = coefficients
            .iter()
            .enumerate()
            .map(|(index, value)| bf16::from_f32(value.to_f32() * scales[index / I::SIZE]))
            .collect();
        let affine: ChannelScaled<I, O> = BlockScaled {
            weight: HostTensor::<f8e4m3, m![O, I]>::from_vec(coefficients)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            weight_scale: HostTensor::<f32, m![O, 1 # 8]>::from_vec(
                scales
                    .iter()
                    .flat_map(|&scale| iter::once(scale).chain([0.0; 7])),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
            contract: (),
            shape: PhantomData,
        };
        let input = HostTensor::<bf16, RowMajor<I, Sixteen>>::from_vec(inputs.clone())
            .to_hbm::<Chip, RowMajor<I, Sixteen>>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(channel_rows, (&mut *device, &affine, &live, &input))
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
                let got = output[row * O::SIZE + column].to_f32();
                assert!(
                    (got - sum).abs() <= tolerance,
                    "{} -> {}, row {row} column {column}: {got} vs {sum}",
                    I::SIZE,
                    O::SIZE,
                );
            }
        }
    }

    /// The live rows of each projection tiling give the host's sums over the restored rows.
    #[tokio::test(flavor = "current_thread")]
    async fn writes_the_restored_rows_at_each_tiling() {
        async fn arbitrary<I: AxisName, O: AxisName>(device: &mut Device) {
            let coefficients: Vec<f8e4m3> = random(O::SIZE * I::SIZE, 0xc0de_f8e4)
                .into_iter()
                .map(|value| f8e4m3::from_f32(value.to_f32() * 8.0))
                .collect();
            let scales: Vec<f32> = (0..O::SIZE)
                .map(|row| 1.0 / 64.0 + (row * 37 % 101) as f32 * 1.7e-4)
                .collect();
            writes_the_restored_rows::<I, O>(device, coefficients, scales).await;
        }
        let mut device = Device::new(channel_rows.topology()).unwrap();
        arbitrary::<geometry::Hidden<2048>, geometry::ProjectionWidth<4096>>(&mut device).await;
        arbitrary::<geometry::Hidden<2560>, geometry::ProjectionWidth<19456>>(&mut device).await;
        arbitrary::<geometry::Hidden<4096>, geometry::ProjectionWidth<24576>>(&mut device).await;
        arbitrary::<geometry::Hidden<3840>, geometry::ProjectionWidth<4096>>(&mut device).await;
        arbitrary::<geometry::Hidden<3840>, geometry::ProjectionWidth<2048>>(&mut device).await;
        arbitrary::<geometry::Hidden<3840>, geometry::ProjectionWidth<8192>>(&mut device).await;
        arbitrary::<geometry::Hidden<3840>, geometry::ProjectionWidth<512>>(&mut device).await;
        arbitrary::<geometry::Hidden<4096>, geometry::ProjectionWidth<3840>>(&mut device).await;
        arbitrary::<geometry::Hidden<8192>, geometry::ProjectionWidth<3840>>(&mut device).await;
    }

    /// A BF16 weight quantized at load by each [`Granularity`], at each projection of a Qwen3-0.6B
    /// layer, gives the host's sums over the coefficients its codes and scales restore.
    #[tokio::test(flavor = "current_thread")]
    async fn writes_the_restored_rows_of_weights_quantized_at_load() {
        async fn quantized<I: AxisName, O: AxisName>(device: &mut Device) {
            // Rows up to sixteen times apart, so a tensor scale leaves the small rows subnormal.
            let weight: Vec<f32> = random(O::SIZE * I::SIZE, 0x00f8_0a11)
                .into_iter()
                .enumerate()
                .map(|(index, value)| value.to_f32() / (1 << (index / I::SIZE % 5)) as f32)
                .collect();
            let weight = Finite::new(weight, I::SIZE).unwrap();
            for granularity in [Granularity::Tensor, Granularity::Channel] {
                let (codes, scales) = granularity.quantize(&weight, 0.0);
                writes_the_restored_rows::<I, O>(device, codes, scales).await;
            }
        }
        let mut device = Device::new(channel_rows.topology()).unwrap();
        type Hidden = geometry::Hidden<1024>;
        type Intermediate = swiglu::Intermediate<3072>;
        quantized::<Hidden, geometry::ProjectionWidth<4096>>(&mut device).await;
        quantized::<geometry::QueryWidth<2048>, Hidden>(&mut device).await;
        quantized::<Hidden, Intermediate>(&mut device).await;
        quantized::<Intermediate, Hidden>(&mut device).await;
    }
}

mod block;
