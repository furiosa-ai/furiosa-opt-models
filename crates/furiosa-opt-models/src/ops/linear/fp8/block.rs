//! The FP8 block linear of one row: E4M3 codes whose activations quantize at run time in groups
//! of 128 inputs, each block's sum scaled before the blocks add, as [`fp8`] states.

use super::super::kernel::Rounded;
use super::super::projection::{Block, Cluster, Reduced, Slices};
use super::Vector;
use crate::specialize::VRF;
use crate::tensor::form::BlockScaled;
use crate::tensor::form::fp8::{self, Blocks, GROUP, Group, Scales};
use crate::{Op, ops::pointwise::cast::Finite};
use core::borrow::Borrow;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

/// Multiplying by the reciprocal, not dividing, rounds each scale as the reference quantizer does.
const RECIPROCAL: f32 = 1.0 / f8e4m3::LARGEST;

/// The scale of a group whose magnitudes all stay below 1e-10.
const FLOOR: f32 = 1e-10 * RECIPROCAL;

/// Clears an FP32 sign bit.
const MAGNITUDE: f32 = f32::from_bits(0x7fff_ffff);

/// The four output rows of a slice that one tile contracts.
type Elements<O> = m![O % ({ Block<O> }) = 4];
/// The four-row tiles of a slice's output rows.
type Tiles<O> = m![O % ({ Block<O> }) / 4];
/// The 128-input blocks of the input half a slice holds.
type Half<I> = m![I % (I / 2) / 128];
/// A half's blocks, padded as the weight scales store them.
type Padded<I> = m![{ Half<I> } # ({ Symbol<Blocks<I>> })];

type Input<I, O, D, Chips> = DmTensor<D, Chips, Cluster, Slices<I, O>, m![I % (I / 2)]>;
type Grouped<I, O, D, Chips> = DmTensor<D, Chips, Cluster, Slices<I, O>, m![{ Half<I> }, Group]>;
type Register<I, O, Chips> = VrfTensor<f32, Chips, Cluster, Slices<I, O>, m![{ Half<I> }, 1 # 8]>;

/// The quantization passes over BF16 activations in DM, read in `Fetched` packets of 16, `Eighths`
/// of eight and `Fourths` of four: each of the `Units` 128-input groups takes one scale.
pub(super) struct Activations<Units, Fetched, Eighths, Fourths>(
    PhantomData<(Units, Fetched, Eighths, Fourths)>,
);

impl<Units: M, Fetched: M, Eighths: M, Fourths: M> Activations<Units, Fetched, Eighths, Fourths> {
    /// Each group's scale, its largest magnitude times [`RECIPROCAL`] and at least [`FLOOR`], in
    /// the first of eight lanes, as the stream a caller lays out.
    pub(super) fn scales<'l, X: M, C: M, S: M, E: M>(
        main: &'l mut TuContext<{ Tu::Main }>,
        input: DmTensorView<'l, bf16, X, C, S, E>,
    ) -> VectorFinalTensor<'l, { Tu::Main }, f32, X, C, S, Units, m![1 # 8]> {
        main.begin(input)
            .fetch::<Fetched, m![Group % 16]>()
            .fetch_cast::<f32>()
            .collect::<Eighths, m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpF32::BitAnd, MAGNITUDE)
            .vector_narrow_split::<Fourths, m![Group % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), RECIPROCAL)
            .vector_intra_slice_reduce::<Group, Units, m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Max, FLOOR)
            .vector_final()
    }

    /// The activations over their groups' `scale` in E4M3, as the stream a caller commits.
    pub(super) fn quantized<'l, X: M, C: M, S: M, E: M, Scale: M>(
        main: &'l mut TuContext<{ Tu::Main }>,
        input: DmTensorView<'l, bf16, X, C, S, E>,
        scale: &VrfTensor<f32, X, C, S, Scale>,
    ) -> CommitTrimTensor<'l, { Tu::Main }, f8e4m3, X, C, S, Eighths, m![Group % 8]> {
        main.begin(input)
            .fetch::<Fetched, m![Group % 16]>()
            .fetch_cast::<f32>()
            .collect::<Eighths, m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Fourths, m![Group % 4]>()
            .vector_fp_div(scale)
            .vector_widen_concat::<Eighths, m![Group % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Group % 8 # 32]>()
            .commit_trim::<m![Group % 8]>()
    }
}

/// A slice's sums of `Rows` weight rows over `Blocks` 128-input blocks of `In` against `Lanes`
/// activations, fetched as `Fetched`, split as `Narrow`/`Half`, reduced into `Reduced`.
pub(super) struct Sums<In, Rows, Blocks, Lanes, Fetched, Narrow, Half, Reduced = Rows>(
    PhantomData<(In, Rows, Blocks, Lanes, Fetched, Narrow, Half, Reduced)>,
);

impl<In: AxisName, Rows: M, Blocks: M, Lanes: M, Fetched: M, Narrow: M, Half: M, Reduced: M>
    Sums<In, Rows, Blocks, Lanes, Fetched, Narrow, Half, Reduced>
{
    /// Each block's FP32 sum of `weight` against `lanes`, times its `activation` scale, then its
    /// weight `scale`, added over the slice's blocks, as the stream a caller commits.
    pub(super) fn of<'l, X: M, C: M, S: M, E: M, L: M, T: M, A: M, W: M>(
        main: &'l mut TuContext<{ Tu::Main }>,
        weight: DmTensorView<'l, f8e4m3, X, C, S, E>,
        lanes: &'l TrfTensor<f8e4m3, X, C, S, L, T>,
        (activation, scale): (&VrfTensor<f32, X, C, S, A>, &VrfTensor<f32, X, C, S, W>),
    ) -> CommitTrimTensor<'l, { Tu::Main }, f32, X, C, S, Rows, Lanes> {
        let sums = main
            .begin(weight)
            .fetch::<Fetched, m![In % 32]>()
            .collect::<Fetched, m![In % 32]>()
            .contract_outer::<m![{ Rows }, { Blocks }, In % 128 / 64], m![In % 64], _, _, _>(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Rows }, { Blocks }]>()
            .contract_lane::<m![{ Rows }, { Blocks }], Lanes>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero);
        if const { Narrow::SIZE == Rows::SIZE * Blocks::SIZE } {
            sums.vector_narrow_trim::<Half>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), activation)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), scale)
                .vector_intra_slice_reduce::<In, Rows, Half>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<Lanes>()
                .vector_final()
                .commit_trim::<Lanes>()
        } else {
            sums.vector_narrow_split::<Narrow, Half>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), activation)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), scale)
                .vector_intra_slice_reduce::<In, Reduced, Half>(IntraSliceReduceOpF32::Add)
                .vector_widen_concat::<Rows, Lanes>()
                .vector_final()
                .commit_trim::<Lanes>()
        }
    }
}

/// The slices' FP32 sums of `Rows` outputs, `Lane` lanes each, added over the slices into
/// `Reduced` and rounded once to BF16, four outputs of `Out` a packet in `Grouped` order.
pub(super) struct Totals<Out, Rows, Lane, Reduced, Grouped>(
    PhantomData<(Out, Rows, Lane, Reduced, Grouped)>,
);

impl<Out: AxisName, Rows: M, Lane: M, Reduced: M, Grouped: M>
    Totals<Out, Rows, Lane, Reduced, Grouped>
{
    /// The `sums` added over the slices and rounded, as the stream a caller commits.
    pub(super) fn of<'l, X: M, C: M, S: M, E: M>(
        main: &'l mut TuContext<{ Tu::Main }>,
        sums: DmTensorView<'l, f32, X, C, S, E>,
    ) -> CommitTrimTensor<'l, { Tu::Main }, bf16, X, C, Reduced, Grouped, m![Out % 4]> {
        let sums = main
            .begin(sums)
            .fetch::<Rows, m![{ Lane } # 8]>()
            .collect::<Rows, m![{ Lane } # 8]>()
            .vector_init()
            .vector_inter_slice_reduce::<Reduced, Rows>(InterSliceReduceOpF32::Add)
            .vector_final();
        Rounded::rounded::<bf16, _, _, _, _, Lane, Grouped, m![Out % 4], _>(sums)
    }
}

/// A slice's block sums of one four-row tile.
type Tile<I, O> = Sums<
    I,
    Elements<O>,
    Half<I>,
    m![1 # 8],
    m![{ Elements<O> }, I % (I / 2) / 32],
    m![{ Elements<O> }, { Half<I> }],
    m![1 # 4],
>;

/// The quantization of a slice's input half.
type Halves<I> = Activations<
    Half<I>,
    m![{ Half<I> }, Group / 16],
    m![{ Half<I> }, Group / 8],
    m![{ Half<I> }, Group / 4],
>;

impl<I: AxisName, O: AxisName, Chips: M, K, V> Op<HbmTensorView<'_, bf16, Chips, m![I]>>
    for BlockScaled<I, O, f8e4m3, f32, Scales<I, O>, fp8::Block, Chips, K, V>
where
    K: Borrow<HbmTensor<f8e4m3, Chips, m![O, I]>>,
    V: Borrow<HbmTensor<f32, Chips, Scales<I, O>>>,
{
    type Output = Vector<O, bf16, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        input: HbmTensorView<'_, bf16, Chips, m![I]>,
    ) -> Self::Output {
        const {
            assert!(
                I::SIZE > 0
                    && O::SIZE > 0
                    && I::SIZE.is_multiple_of(2 * GROUP)
                    && O::SIZE.is_multiple_of(256)
                    // The activation and weight scales of a half share one slice's VRF.
                    && 2 * (I::SIZE / (2 * GROUP)) * 32 <= VRF
            );
        }
        let input: DmTensor<bf16, Chips, Cluster, m![64, 1 # 2, I / (I / 2)], m![I % (I / 2)]> =
            input.to_dm(&mut device.tdma);
        let input: Input<I, O, bf16, Chips> = unsafe { input.reshape() };
        let input: Grouped<I, O, bf16, Chips> = unsafe { input.reshape() };
        let scales: DmTensor<f32, Chips, Cluster, Slices<I, O>, m![{ Half<I> }, 1 # 8]> =
            Halves::<I>::scales(&mut device.main, input.view())
                .commit_trim::<m![1 # 8]>()
                .commit();
        let activation: Register<I, O, Chips> = device
            .sub
            .begin(scales.view())
            .fetch::<Half<I>, m![1 # 8]>()
            .collect::<Half<I>, m![1 # 8]>()
            .to_vrf();
        let quantized: Grouped<I, O, f8e4m3, Chips> =
            Halves::<I>::quantized(&mut device.main, input.view(), &activation).commit();
        let quantized: Input<I, O, f8e4m3, Chips> = unsafe { quantized.reshape() };
        let quantized: TrfTensor<f8e4m3, Chips, Cluster, Slices<I, O>, m![1], m![I % (I / 2)]> =
            device
                .sub
                .begin(quantized.view())
                .fetch::<m![I % (I / 2) / 32], m![I % 32]>()
                .collect::<m![I % (I / 2) / 32], m![I % 32]>()
                .to_trf();
        let weight_scale = unsafe {
            self.scales().view().reshape::<Chips, m![
                O / ({ Block<O> }), { Tiles<O> }, I / (I / 2), { Padded<I> }
            ]>()
        };
        let weight_scale: DmTensor<
            f32,
            Chips,
            Cluster,
            Slices<I, O>,
            m![{ Tiles<O> }, { Padded<I> }],
        > = weight_scale.to_dm(&mut device.tdma);
        let mut halves: DmTensor<f32, Chips, Cluster, Slices<I, O>, m![O % ({ Block<O> }), 1 # 8]> =
            DmTensor::new();
        for tile in 0..(O::SIZE / 256) {
            let weight: DmTensor<
                f8e4m3,
                Chips,
                Cluster,
                Slices<I, O>,
                m![{ Elements<O> }, I % (I / 2)],
            > = self.codes().view().tile::<
                m![O % ({ Block<O> })],
                4,
                m![O / ({ Block<O> }), { Elements<O> } # ({ Block<O> }), I],
            >(4 * tile).to_dm(&mut device.tdma);
            let scale: Register<I, O, Chips> = device
                .sub
                .begin(weight_scale.view().tile::<Tiles<O>, 1, m![
                    { Tiles<O> } = 1 # ({ Tiles<O> }), { Padded<I> }
                ]>(tile))
                .fetch::<Half<I>, m![1 # 8]>()
                .collect::<Half<I>, m![1 # 8]>()
                .to_vrf();
            // The two scale products sit in one vector chain with the block sum; the halves add up
            // after the loop, as a chain that also reduces across slices does not lower.
            Tile::<I, O>::of(
                &mut device.main,
                weight.view(),
                &quantized,
                (&activation, &scale),
            )
            .commit_view(halves.view_mut().tile::<m![O % ({ Block<O> })], 4, m![
                { Elements<O> } #{!} ({ Block<O> }), 1 # 8
            ]>(4 * tile));
        }
        let output: DmTensor<bf16, Chips, Cluster, Reduced<O>, m![O % ({ Block<O> })]> =
            Totals::<O, m![O % ({ Block<O> })], m![1], Reduced<O>, m![O % ({ Block<O> }) / 4]>::of(
                &mut device.main,
                halves.view(),
            )
            .commit();
        output.to_hbm(&mut device.tdma)
    }
}

impl<I: AxisName, O: AxisName, Chips: M, K, V> Op<&Vector<I, bf16, Chips>>
    for BlockScaled<I, O, f8e4m3, f32, Scales<I, O>, fp8::Block, Chips, K, V>
where
    K: Borrow<HbmTensor<f8e4m3, Chips, m![O, I]>>,
    V: Borrow<HbmTensor<f32, Chips, Scales<I, O>>>,
{
    type Output = Vector<O, bf16, Chips>;

    fn forward(&self, device: &mut Device, input: &Vector<I, bf16, Chips>) -> Self::Output {
        self.forward(device, input.view())
    }
}

impl<I: AxisName, O: AxisName, Heads: AxisName, Head: AxisName, Chips: M, K, V>
    Op<&HbmTensor<bf16, Chips, m![Heads, Head]>>
    for BlockScaled<I, O, f8e4m3, f32, Scales<I, O>, fp8::Block, Chips, K, V>
where
    K: Borrow<HbmTensor<f8e4m3, Chips, m![O, I]>>,
    V: Borrow<HbmTensor<f32, Chips, Scales<I, O>>>,
{
    type Output = Vector<O, bf16, Chips>;

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
pub(super) mod tests {
    use super::*;
    use crate::tensor::parallel::Chip;
    use crate::tests::DEVICE;

    axes![
        Narrow = 768,
        Square = 768,
        Middle = 512,
        Keys = 256,
        Wide = 1024
    ];

    /// Activation group exponents of `inputs`: a group's largest input is 448 times two to its
    /// exponent, so its scale is exactly that power of two.
    pub(in super::super) const EXPONENTS: [i32; 6] = [-1, 0, 1, 2, -2, 1];

    #[device(chip = 1)]
    fn projected<I: AxisName, O: AxisName>(
        device: &mut Device,
        linear: &BlockScaled<I, O, f8e4m3, f32, Scales<I, O>, fp8::Block, Chip>,
        input: &Vector<I>,
    ) -> Vector<O> {
        linear.forward(device, input)
    }

    /// The reference quantizer's FP32 outputs (vLLM 0.24.0 `per_token_group_quant_fp8` and
    /// `w8a8_triton_block_scaled_mm`): scale `max(amax, 1e-10) * fl(1/448)` per 128 inputs,
    /// saturating E4M3 `x / scale`, each block's FP32 sum times its activation scale then its
    /// `[output / 128, input / 128]` weight scale, blocks added in order.
    pub(in super::super) fn reference(input: &[bf16], weight: &[f8e4m3], grid: &[f32]) -> Vec<f32> {
        let (inputs, groups) = (input.len(), input.len() / GROUP);
        let (quantized, scales): (Vec<Vec<f32>>, Vec<f32>) = input
            .chunks_exact(GROUP)
            .map(|group| {
                let largest = group.iter().map(|x| x.to_f32().abs()).fold(0.0, f32::max);
                let scale = largest.max(1e-10) * (1.0 / 448.0);
                let codes = group
                    .iter()
                    .map(|x| f8e4m3::from_f32(x.to_f32() / scale).to_f32())
                    .collect();
                (codes, scale)
            })
            .unzip();
        weight
            .chunks_exact(inputs)
            .enumerate()
            .map(|(row, coefficients)| {
                (0..groups)
                    .map(|group| {
                        let sum: f32 = coefficients[group * GROUP..][..GROUP]
                            .iter()
                            .zip(&quantized[group])
                            .map(|(w, x)| w.to_f32() * x)
                            .sum();
                        sum * scales[group] * grid[row / GROUP * groups + group]
                    })
                    .sum()
            })
            .collect()
    }

    async fn run<I: AxisName, O: AxisName>(
        device: &mut Device,
        input: &[bf16],
        weight: &[f8e4m3],
        grid: &[f32],
    ) -> Vec<f32> {
        let linear: BlockScaled<I, O, f8e4m3, f32, Scales<I, O>, fp8::Block, Chip> = BlockScaled {
            weight: HostTensor::<f8e4m3, m![O, I]>::from_vec(weight.to_vec())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            weight_scale: HostTensor::<f32, Scales<I, O>>::from_vec(fp8::layout::<I, O>(grid))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            contract: fp8::Block,
            shape: PhantomData,
        };
        let input = HostTensor::<bf16, m![I]>::from_vec(input.to_vec())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        launch(projected, (&mut *device, &linear, &input))
            .await
            .unwrap()
            .to_host::<m![O]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
            .into_iter()
            .map(|value| value.to_f32())
            .collect()
    }

    /// A hash of `index` under `seed`, for deterministic test values.
    pub(in super::super) fn hash(index: usize, seed: u32) -> u32 {
        (index as u32 ^ seed)
            .wrapping_mul(0x9e37_79b9)
            .rotate_left(13)
            .wrapping_mul(0x85eb_ca6b)
    }

    /// Each input quantizes by its group's scale as the reference does: ties to even, E4M3
    /// subnormals and flushes, the largest input at 448, a scale that is no power of two, an
    /// all-zero group, and a group below the scale floor; each output row reads one input through
    /// its own block scale, across blocks that straddle the slices' row groups.
    #[tokio::test(flavor = "current_thread")]
    async fn quantizes_each_input() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        let (inputs, outputs) = (Narrow::SIZE, Square::SIZE);
        let edges = [
            448.0,
            -448.0,
            9.5,
            -9.5,
            17.0,
            1.0625,
            1.1875,
            -1.1875,
            0.00390625,
            0.0029296875,
            0.00048828125,
            -0.0029296875,
            0.0,
            300.0,
            3.3,
            -0.7,
        ];
        let input: Vec<bf16> = (0..inputs)
            .map(|index| {
                let at = index % GROUP;
                bf16::from_f32(match index / GROUP {
                    0 => edges[at % edges.len()] / 4.0,
                    1 | 4 if at == 77 => 3.7,
                    1 | 4 => ((hash(index, 7) % 1945) as f32 - 972.0) / 270.0,
                    2 => 0.0,
                    3 => ((hash(index, 9) % 201) as f32 - 100.0) * 1e-14,
                    _ => ((hash(index, 11) % 897) as f32 - 448.0) / 64.0,
                })
            })
            .collect();
        let source = |row: usize| (row * 5 + 131) % inputs;
        let weight: Vec<f8e4m3> = (0..outputs * inputs)
            .map(|index| f8e4m3::from_f32(f32::from(index % inputs == source(index / inputs))))
            .collect();
        let grid: Vec<f32> = (0..(outputs / GROUP) * (inputs / GROUP))
            .map(|block| 2f32.powi((block * 7 % 9) as i32 - 4))
            .collect();
        let output = run::<Narrow, Square>(&mut device, &input, &weight, &grid).await;
        let expected = reference(&input, &weight, &grid);
        for (row, (got, expected)) in output.iter().zip(&expected).enumerate() {
            assert_eq!(
                *got,
                bf16::from_f32(*expected).to_f32(),
                "row {row} reads {}",
                input[source(row)].to_f32()
            );
        }
    }

    /// Integers below 16 times their group's power of two, and one input of 448 times it a group.
    pub(in super::super) fn powers(inputs: usize) -> Vec<bf16> {
        (0..inputs)
            .map(|index| {
                let (group, at) = (index / GROUP, index % GROUP);
                let unit = 2f32.powi(EXPONENTS[group]);
                bf16::from_f32(match at == group * 31 % GROUP {
                    true => 448.0 * unit * if group % 2 == 0 { 1.0 } else { -1.0 },
                    false => ((hash(index, 3) % 33) as f32 - 16.0) * unit,
                })
            })
            .collect()
    }

    /// Integer E4M3 weights below 3 and power-of-two block scales: every block sum and scaled
    /// term is exact in FP32.
    pub(in super::super) fn integers(
        outputs: usize,
        inputs: usize,
        seed: u32,
    ) -> (Vec<f8e4m3>, Vec<f32>) {
        let weight = (0..outputs * inputs)
            .map(|index| f8e4m3::from_f32((hash(index, 5) % 5) as f32 - 2.0))
            .collect();
        let grid = (0..(outputs / GROUP) * (inputs / GROUP))
            .map(|block| 2f32.powi(((block + seed as usize) * 3 % 5) as i32 - 2))
            .collect();
        (weight, grid)
    }

    /// Each block's integer sum takes its own activation and weight scale before the blocks add
    /// up: every row gives the reference exactly, and one scale after the whole sum does not.
    #[tokio::test(flavor = "current_thread")]
    async fn scales_each_block_before_the_sum() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        let (inputs, outputs) = (Narrow::SIZE, Middle::SIZE);
        let input = powers(inputs);
        let (weight, grid) = integers(outputs, inputs, 0);
        let output = run::<Narrow, Middle>(&mut device, &input, &weight, &grid).await;
        let expected = reference(&input, &weight, &grid);
        let mut differs = 0;
        for (row, (got, expected)) in output.iter().zip(&expected).enumerate() {
            assert_eq!(*got, bf16::from_f32(*expected).to_f32(), "row {row}");
            let whole: f32 = (0..inputs)
                .map(|index| {
                    let code = input[index].to_f32() / 2f32.powi(EXPONENTS[index / GROUP]);
                    weight[row * inputs + index].to_f32() * code
                })
                .sum();
            let once = whole * 2f32.powi(EXPONENTS[0]) * grid[row / GROUP * (inputs / GROUP)];
            differs += usize::from(bf16::from_f32(once).to_f32() != *got);
        }
        assert!(
            differs > outputs / 2,
            "{differs} rows tell the schemes apart"
        );
    }

    /// Query, key and value projections of one input each give the reference through their own
    /// block scales; key and value share their weights and differ only by those scales.
    #[tokio::test(flavor = "current_thread")]
    async fn keeps_each_projection_scales() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        let input = powers(Narrow::SIZE);
        let (query, query_grid) = integers(Middle::SIZE, Narrow::SIZE, 1);
        let (key, key_grid) = integers(Keys::SIZE, Narrow::SIZE, 2);
        let value_grid: Vec<f32> = key_grid.iter().map(|scale| scale * 4.0).collect();
        let queries = run::<Narrow, Middle>(&mut device, &input, &query, &query_grid).await;
        let keys = run::<Narrow, Keys>(&mut device, &input, &key, &key_grid).await;
        let values = run::<Narrow, Keys>(&mut device, &input, &key, &value_grid).await;
        for (output, weight, grid) in [
            (&queries, &query, &query_grid),
            (&keys, &key, &key_grid),
            (&values, &key, &value_grid),
        ] {
            let expected = reference(&input, weight, grid);
            for (row, (got, expected)) in output.iter().zip(&expected).enumerate() {
                assert_eq!(*got, bf16::from_f32(*expected).to_f32(), "row {row}");
            }
        }
        assert_ne!(keys, values);
    }

    /// Random activations with outliers, E4M3 weights over their whole range and BF16 block
    /// scales fall within one BF16 step of the reference's FP32 outputs.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_the_reference_quantizer() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        let (inputs, outputs) = (Wide::SIZE, Narrow::SIZE);
        let input: Vec<bf16> = (0..inputs)
            .map(|index| {
                let value = ((hash(index, 11) % 20001) as f32 - 10000.0) / 4000.0;
                bf16::from_f32(if index % 97 == 0 { value * 40.0 } else { value })
            })
            .collect();
        let weight: Vec<f8e4m3> = (0..outputs * inputs)
            .map(|index| {
                let value = (hash(index, 13) % 897) as f32 - 448.0;
                f8e4m3::from_f32(value / (1 + index % 7) as f32)
            })
            .collect();
        let grid: Vec<f32> = (0..(outputs / GROUP) * (inputs / GROUP))
            .map(|block| {
                bf16::from_f32(1e-4 * (1.0 + (hash(block, 17) % 1000) as f32 / 100.0)).to_f32()
            })
            .collect();
        let output = run::<Wide, Narrow>(&mut device, &input, &weight, &grid).await;
        let expected = reference(&input, &weight, &grid);
        for (row, (got, expected)) in output.iter().zip(&expected).enumerate() {
            let step = 2f32.powi(expected.abs().max(f32::MIN_POSITIVE).log2().floor() as i32 - 7);
            assert!(
                (got - expected).abs() <= step,
                "row {row}: {got} vs {expected}"
            );
        }
    }
}
