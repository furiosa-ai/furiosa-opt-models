//! The linear of the late NVFP4 contract of [`crate::tensor::form::nvfp4`]: a contiguous row
//! through a [`BlockScaled`] projection, its inputs entering in the contract's input form.

use super::kernel::{Contraction, Tiling};
use super::projection::{Block, Cluster, OneRow, Partials, Reduced, Slices};
use crate::{
    Op,
    tensor::{parallel::Chip, scalar},
};
use furiosa_opt_std::prelude::*;

use super::quantize::{Group, Quantization};
use crate::tensor::form::BlockScaled;
use crate::tensor::form::nvfp4::{Held, Late, Scale, Scales, W4A4, W4A16};
use crate::tensor::memory::Memory;
use core::borrow::Borrow;

mod rows;

/// One contiguous feature vector.
pub(crate) type Vector<W> = HbmTensor<bf16, Chip, m![W]>;

/// BF16 inputs enter the contraction as they are.
impl<I: AxisName> Op<Half<I>> for W4A16 {
    type Output = Half<I>;

    fn forward(&self, _: &mut Device, input: Half<I>) -> Half<I> {
        input
    }
}

/// `input` with each element replaced by its E2M1 code times its block's E4M3 scale.
impl<I: AxisName, Y: Memory> Op<Half<I>> for W4A4<Y>
where
    Held<Y>: Borrow<Scale>,
{
    type Output = Half<I>;

    fn forward(&self, device: &mut Device, input: Half<I>) -> Half<I> {
        // SAFETY: relabels each slice's half row as its consecutive 16-input blocks.
        let input: Grouped<I, bf16> = unsafe { input.reshape() };
        let (global, restore): (&Scale, &Scale) = (self.global.borrow(), self.restore.borrow());
        let global: Packet<I> = global.to_dm(&mut device.tdma);
        let global = scalar::register(device, &global);
        let restore: Packet<I> = restore.to_dm(&mut device.tdma);
        let restore = scalar::register(device, &restore);
        let (codes, scale) = Single::<I>::codes(device, || input.view(), (&global, &restore));
        let quantized: Grouped<I, bf16> =
            Single::<I>::restored(&mut device.main, codes.view(), &scale).commit();
        // SAFETY: the inverse of the relabel above.
        unsafe { quantized.reshape() }
    }
}

/// The NVFP4 quantization of a slice's input half.
type Single<I> = Quantization<
    m![{ Blocks<I> }],
    m![{ Blocks<I> }, Group],
    m![{ Blocks<I> }, Group / 8],
    m![{ Blocks<I> }, Group / 4],
    m![{ Blocks<I> } / 8],
    m![{ Blocks<I> } / 4],
    Blocks<I>,
>;

/// The 16-input blocks of the input half one slice holds.
type Blocks<I> = m![I % (I / 2) / 16];

/// Each slice's input half; every output block's slices hold a copy.
type Halves<I> = m![64, 1 # 2, I / (I / 2)];
type Half<I, D = bf16> = DmTensor<D, Chip, Cluster, Halves<I>, m![I % (I / 2)]>;
type Grouped<I, D> = DmTensor<D, Chip, Cluster, Halves<I>, m![{ Blocks<I> }, Group]>;
type Packet<I> = DmTensor<f32, Chip, Cluster, Halves<I>, m![1 # 8]>;

type Elements<O> = m![O % ({ Block<O> }) = 4];
type Tile<I, O> = DmTensor<bf16, Chip, Cluster, Slices<I, O>, m![{ Elements<O> }, I % (I / 2)]>;

impl<I: AxisName, O: AxisName, A, K, V, Y: Memory> Op<&Vector<I>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Late<A, Y>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, m![O, I]>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Scales<I, O>>>,
    Held<Y>: Borrow<Scale>,
    A: Op<Half<I>, Output = Half<I>>,
{
    type Output = Vector<O>;

    /// Project one contiguous BF16 input row; both widths must be positive multiples of 256, and
    /// two rows' block scales of each slice's input half must fit its 8 KiB of vector registers.
    fn forward(&self, device: &mut Device, input: &Vector<I>) -> Self::Output {
        const {
            assert!(
                I::SIZE > 0
                    && O::SIZE > 0
                    && I::SIZE.is_multiple_of(256)
                    && O::SIZE.is_multiple_of(256)
                    && I::SIZE <= 32768
            )
        };
        // Output rows `index..index + 4` of each slice's block as BF16 code times block scale, two
        // rows a pass so a pass's block scales fit the vector registers.
        let read = |device: &mut Device, index: usize| -> Tile<I, O> {
            let coded = (
                OneRow::<I, O>::to_dm(device, self.codes(), index),
                OneRow::<I, O>::to_dm(device, self.scales(), index),
            );
            Self::restore::<_, _, _, _, m![{ Elements<O> } / 2], m![{ Elements<O> } % 2], _, _>(
                device,
                coded,
                |values| values,
            )
        };
        let input: Half<I> = input.to_dm(&mut device.tdma);
        let input = self.contract.input.forward(&mut *device, input);
        // SAFETY: names each copy of the input half after the output block its slices project.
        let input: DmTensor<bf16, Chip, Cluster, Slices<I, O>, m![I % (I / 2)]> =
            unsafe { input.reshape() };
        let input: TrfTensor<bf16, Chip, Cluster, Slices<I, O>, m![1], m![1, I % (I / 2)]> = device
            .sub
            .begin(input.view())
            .fetch::<m![I % (I / 2) / 16], m![I % 16]>()
            .collect::<m![I % (I / 2) / 16], m![I % 16]>()
            .to_trf();
        let output: &Scale = self.contract.output.borrow();
        let scale: DmTensor<f32, Chip, Cluster, Reduced<O>, m![1 # 8]> =
            output.to_dm(&mut device.tdma);
        let scale = scalar::register(device, &scale);
        let mut output: DmTensor<bf16, Chip, Cluster, Reduced<O>, m![O % ({ Block<O> })]> =
            DmTensor::new();
        for index in 0..(O::SIZE / 256) {
            let tile = read(device, 4 * index);
            let partial: DmTensor<f32, Chip, Cluster, Slices<I, O>, m![{ Elements<O> }, 1 # 8]> =
                Contraction::<I, O>::sums(device, &tile, &input)
                    .commit_trim::<m![1 # 8]>()
                    .commit();
            let value: DmTensor<bf16, Chip, Cluster, Reduced<O>, Elements<O>> =
                Partials::<O, Elements<O>>::restored(&mut device.main, partial.view(), |sums| {
                    sums.vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
                })
                .commit();
            value.view().to_dm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![O % ({ Block<O> })], 4, m![{ Elements<O> } #{!} ({ Block<O> })]>(
                        4 * index,
                    ),
            );
        }
        output.to_hbm(&mut device.tdma)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ops::pointwise::cast::Finite;
    use crate::tensor::form::nvfp4::Scale;
    use core::marker::PhantomData;
    use furiosa_opt_std::runtime::launch;
    use std::iter;

    axes![Input = 512, Output = 512, Wide = 21504, Narrow = 256];

    #[device(chip = 1)]
    fn projected<I: AxisName, O: AxisName>(
        device: &mut Device,
        linear: &BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Late<W4A4>>,
        input: &Vector<I>,
    ) -> Vector<O> {
        linear.forward(device, input)
    }

    /// Round to the nearest E2M1 value, ties to even, from a value clamped to the E2M1 range.
    fn e2m1(value: f32) -> f32 {
        let magnitude = value.abs();
        let rounded: f32 = match magnitude {
            m if m <= 0.25 => 0.0,
            m if m < 0.75 => 0.5,
            m if m <= 1.25 => 1.0,
            m if m < 1.75 => 1.5,
            m if m <= 2.5 => 2.0,
            m if m < 3.5 => 3.0,
            m if m <= 5.0 => 4.0,
            _ => 6.0,
        };
        rounded.copysign(value)
    }

    /// The reference quantizer's code times block scale for each input: block scales count in
    /// units of the input scale's reciprocal, and an input divides by its block scale returned to
    /// input units.
    pub(super) fn quantized(input: &[bf16], input_scale: f32) -> Vec<f32> {
        let global = 1.0 / input_scale;
        let restore = 1.0 / global;
        input
            .chunks_exact(16)
            .flat_map(|block| {
                let maximum = block
                    .iter()
                    .map(|x| x.to_f32().abs())
                    .fold(0.0f32, f32::max);
                let scale =
                    f8e4m3::from_f32((global * (maximum * (1.0 / 6.0))).min(f8e4m3::LARGEST))
                        .to_f32();
                let reciprocal = 1.0 / (scale * restore);
                block.iter().map(move |x| {
                    if scale == 0.0 {
                        0.0
                    } else {
                        e2m1((x.to_f32() * reciprocal).clamp(-f4e2m1::LARGEST, f4e2m1::LARGEST))
                            * scale
                    }
                })
            })
            .collect()
    }

    pub(super) fn noise(count: usize, mut state: u64) -> impl Iterator<Item = f32> {
        (0..count).map(move |_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
    }

    async fn scalar(device: &mut Device, value: f32) -> Scale {
        HostTensor::<f32, m![1 # 8]>::from_vec(iter::once(value).chain([0.0; 7]))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap()
    }

    /// The projection of `codes` under E4M3 block `scales` and a checkpoint's tensor scales.
    pub(super) async fn linear<I: AxisName, O: AxisName>(
        device: &mut Device,
        codes: &[f4e2m1],
        scales: &[f8e4m3],
        input_scale: f32,
        weight_scale_2: f32,
    ) -> BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Late<W4A4>> {
        let global = 1.0 / input_scale;
        let input = W4A4 {
            global: scalar(device, global).await,
            restore: scalar(device, 1.0 / global).await,
        };
        projection(device, codes, scales, input, input_scale * weight_scale_2).await
    }

    /// The projection of `codes` under E4M3 block `scales` whose inputs enter in form `input`,
    /// its sums times `output`.
    pub(crate) async fn projection<I: AxisName, O: AxisName, A>(
        device: &mut Device,
        codes: &[f4e2m1],
        scales: &[f8e4m3],
        input: A,
        output: f32,
    ) -> BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Late<A>> {
        BlockScaled {
            weight: HostTensor::<f4e2m1, m![O, I]>::from_vec(codes.to_vec())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            weight_scale: HostTensor::<f8e4m3, Scales<I, O>>::from_vec(scales.to_vec())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            contract: Late {
                input,
                output: scalar(device, output).await,
            },
            shape: PhantomData,
        }
    }

    async fn run<I: AxisName, O: AxisName>(
        device: &mut Device,
        linear: &BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Late<W4A4>>,
        input: &[bf16],
    ) -> Vec<bf16> {
        let input = HostTensor::<bf16, m![I]>::from_vec(input.to_vec())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        launch(projected, (&mut *device, linear, &input))
            .await
            .unwrap()
            .to_host::<m![O]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }

    /// A weight whose row `o` selects input `o` gives each quantized input times the input scale:
    /// the reference quantizer's code times block scale bit for bit, at E2M1 ties, at zero and
    /// subnormal block scales and where the block scale saturates.
    #[tokio::test(flavor = "current_thread")]
    async fn quantizes_inputs_as_the_reference() {
        let mut device = Device::new(projected.topology()).unwrap();
        let mut codes = vec![f4e2m1::from_bits(0); Output::SIZE * Input::SIZE];
        for row in 0..Output::SIZE {
            codes[row * Input::SIZE + row] = f4e2m1::from_bits(0b0010);
        }
        let scales = vec![f8e4m3::from_f32(1.0); Output::SIZE * Input::SIZE / 16];
        for input_scale in [0.03125, 0.3258928656578064] {
            let linear =
                linear::<Input, Output>(&mut device, &codes, &scales, input_scale, 1.0).await;
            let ties = [
                6.0, 0.5, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75,
            ];
            let input: Vec<bf16> = noise(Input::SIZE, 0x2545_f491_4f6c_dd1d)
                .enumerate()
                .map(|(index, value)| {
                    let (block, member) = (index / 16, index % 16);
                    bf16::from_f32(match block {
                        0 => 0.0,
                        // A zero block scale, and an E4M3 subnormal one.
                        1 => value * 1e-6,
                        2 => value * 1e-3,
                        // `x * reciprocal` falls on the E2M1 ties when the input scale is 2^-5.
                        3 => 4.0 * input_scale * ties.get(member).copied().unwrap_or(-2.5),
                        // A block scale above the largest E4M3 value before it saturates.
                        4 => value * 1e4,
                        _ => value * 100.0 * (block % 5) as f32,
                    })
                })
                .collect();
            let expected: Vec<u16> = quantized(&input, input_scale)
                .into_iter()
                .map(|value| bf16::from_f32(input_scale * value).to_bits())
                .collect();
            let output: Vec<u16> = run(&mut device, &linear, &input)
                .await
                .into_iter()
                .map(|value| {
                    if value.to_f32() == 0.0 {
                        0
                    } else {
                        value.to_bits()
                    }
                })
                .collect();
            let expected: Vec<u16> = expected
                .into_iter()
                .map(|bits| {
                    if bf16::from_bits(bits).to_f32() == 0.0 {
                        0
                    } else {
                        bits
                    }
                })
                .collect();
            assert_eq!(output, expected, "input scale {input_scale}");
        }
    }

    /// Random codes, block scales and inputs give the exact sum of the code-times-scale products
    /// times both tensor scales, within one BF16 rounding and the FP32 accumulation's error.
    async fn sums_exactly<I: AxisName, O: AxisName>(device: &mut Device) {
        let mut seeds = noise(2 * O::SIZE * I::SIZE / 16 + I::SIZE, 0x9e37_79b9_7f4a_7c15);
        let codes: Vec<f4e2m1> = (0..O::SIZE * I::SIZE)
            .map(|index| f4e2m1::from_bits((index * 7 + index / 13) as u8 & 15))
            .collect();
        let scales: Vec<f8e4m3> = (0..O::SIZE * I::SIZE / 16)
            .map(|_| f8e4m3::from_f32(2f32.powf(6.0 * seeds.next().unwrap() + 2.0)))
            .collect();
        let (input_scale, weight_scale_2) = (0.0424107164144516, 0.00017293295240961015);
        let input: Vec<bf16> = seeds
            .by_ref()
            .take(I::SIZE)
            .map(|value| bf16::from_f32(value * 100.0))
            .collect();
        let linear = linear::<I, O>(device, &codes, &scales, input_scale, weight_scale_2).await;
        let results = run(device, &linear, &input).await;
        let operand = quantized(&input, input_scale);
        let output = f64::from(input_scale) * f64::from(weight_scale_2);
        for (row, got) in results.iter().enumerate() {
            let terms = (0..I::SIZE).map(|column| {
                let weight = codes[row * I::SIZE + column].to_f32()
                    * scales[(row * I::SIZE + column) / 16].to_f32();
                weight as f64 * operand[column] as f64
            });
            let (sum, magnitude) = terms.fold((0.0, 0.0), |(sum, magnitude), term| {
                (sum + term, magnitude + term.abs())
            });
            let exact = sum * output;
            let bound = exact.abs() / 128.0 + magnitude * output / f64::from(1 << 14);
            assert!(
                (got.to_f32() as f64 - exact).abs() <= bound,
                "{}x{} row {row}: {} against {exact}",
                I::SIZE,
                O::SIZE,
                got.to_f32()
            );
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn restores_exact_sums_once() {
        let mut device = Device::new(projected.topology()).unwrap();
        sums_exactly::<Input, Output>(&mut device).await;
        sums_exactly::<Wide, Narrow>(&mut device).await;
    }
}
