//! E4M3 projections whose activations quantize at run time in groups of 128 inputs and whose
//! weights carry one scale per 128 x 128 block.
//!
//! - A group's scale is its largest magnitude over the largest E4M3 value, never below `FLOOR`,
//!   so an all-zero group quantizes to zeros instead of dividing by zero.
//! - Each 128-input block contracts on its own; its FP32 sum takes the block's activation scale,
//!   then its weight scale, before the blocks add up.

use super::super::projection::{Block, Cluster, Reduced, Slices};
use super::Vector;
use crate::{Module, ops::layout::axes::Chip};
use furiosa_opt_std::prelude::*;

// The inputs of one quantization group, apart from the group they belong to.
axes![Group = 128];

/// Inputs that share one activation scale, and each side of a weight block.
pub const GROUP: usize = Group::SIZE;

/// The largest finite E4M3 magnitude: a group's largest input quantizes to it.
const LARGEST: f32 = 448.0;

/// Multiplying by the reciprocal, not dividing, rounds each scale as the reference quantizer does.
pub(super) const RECIPROCAL: f32 = 1.0 / LARGEST;

/// The scale of a group whose magnitudes all stay below 1e-10.
pub(super) const FLOOR: f32 = 1e-10 * RECIPROCAL;

/// Clears an FP32 sign bit.
pub(super) const MAGNITUDE: f32 = f32::from_bits(0x7fff_ffff);

/// One FP32 weight scale per four output rows and 128 inputs, in [output, input] order, with the
/// blocks of each input half padded to an even count: a checkpoint's 128 x 128 block scale repeats
/// over the 32 four-row tiles of its block.
pub type Scales<In, Out, Chips = Chip> = HbmTensor<
    f32,
    Chips,
    m![Out / 4, In / (In / 2), In % (In / 2) / 128 # ({ Symbol<Blocks<In>> })],
>;

/// The 128-input blocks of one input half, padded to an even count so that a row of FP32 scales
/// fills whole 8-byte SRAM words.
#[derive(Clone, Debug)]
pub struct Blocks<In>(std::marker::PhantomData<In>);

impl<In: AxisName> AxisName for Blocks<In> {
    const NAME: Ident = Ident::new("Blocks");
    const SIZE: usize = (In::SIZE / (2 * GROUP)).next_multiple_of(2);
}

/// E4M3 weights in 128 x 128 blocks of one scale each, contracted with activations quantized per
/// 128-input group at run time.
#[derive(DeviceSend)]
pub struct BlockScaled<In: AxisName, Out: AxisName, Chips: M = Chip> {
    /// E4M3 coefficients in [output, input] order.
    pub weight: HbmTensor<f8e4m3, Chips, m![Out, In]>,
    /// Multiply a block's contracted sum by its scale to restore weight units.
    pub weight_scale: Scales<In, Out, Chips>,
}

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

impl<I: AxisName, O: AxisName, Chips: M>
    Module<(&mut Device, HbmTensorView<'_, bf16, Chips, m![I]>)> for BlockScaled<I, O, Chips>
{
    type Output = Vector<O, bf16, Chips>;

    fn forward(
        &self,
        (device, input): (&mut Device, HbmTensorView<'_, bf16, Chips, m![I]>),
    ) -> Self::Output {
        const {
            assert!(
                I::SIZE > 0
                    && O::SIZE > 0
                    && I::SIZE.is_multiple_of(2 * GROUP)
                    && O::SIZE.is_multiple_of(256)
                    // The activation and weight scales of a half share one slice's 8 KiB VRF.
                    && 2 * (I::SIZE / (2 * GROUP)) * 32 <= 8192
            );
        }
        let input: DmTensor<bf16, Chips, Cluster, m![64, 1 # 2, I / (I / 2)], m![I % (I / 2)]> =
            input.to_dm(&mut device.tdma);
        let input: Input<I, O, bf16, Chips> = unsafe { input.reshape() };
        let input: Grouped<I, O, bf16, Chips> = unsafe { input.reshape() };
        let scales: DmTensor<f32, Chips, Cluster, Slices<I, O>, m![{ Half<I> }, 1 # 8]> = device
            .main
            .begin(input.view())
            .fetch::<m![{ Half<I> }, Group / 16], m![Group % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Half<I> }, Group / 8], m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpF32::BitAnd, MAGNITUDE)
            .vector_narrow_split::<m![{ Half<I> }, Group / 4], m![Group % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), RECIPROCAL)
            .vector_intra_slice_reduce::<Group, Half<I>, m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Max, FLOOR)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let activation: Register<I, O, Chips> = device
            .sub
            .begin(scales.view())
            .fetch::<Half<I>, m![1 # 8]>()
            .collect::<Half<I>, m![1 # 8]>()
            .to_vrf();
        let quantized: Grouped<I, O, f8e4m3, Chips> = device
            .main
            .begin(input.view())
            .fetch::<m![{ Half<I> }, Group / 16], m![Group % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Half<I> }, Group / 8], m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Half<I> }, Group / 4], m![Group % 4]>()
            .vector_fp_div(&activation)
            .vector_widen_concat::<m![{ Half<I> }, Group / 8], m![Group % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Group % 8 # 32]>()
            .commit_trim::<m![Group % 8]>()
            .commit();
        let quantized: Input<I, O, f8e4m3, Chips> = unsafe { quantized.reshape() };
        let quantized: TrfTensor<f8e4m3, Chips, Cluster, Slices<I, O>, m![1], m![I % (I / 2)]> =
            device
                .sub
                .begin(quantized.view())
                .fetch::<m![I % (I / 2) / 32], m![I % 32]>()
                .collect::<m![I % (I / 2) / 32], m![I % 32]>()
                .to_trf();
        let weight_scale = unsafe {
            self.weight_scale.view().reshape::<Chips, m![
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
            let weight: DmTensor<f8e4m3, Chips, Cluster, Slices<I, O>, m![{ Elements<O> }, I % (I / 2)]> = self.weight.view().tile::<m![O % ({ Block<O> })], 4, m![O / ({ Block<O> }), { Elements<O> } # ({ Block<O> }), I]>(4 * tile).to_dm(&mut device.tdma);
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
            device
                .main
                .begin(weight.view())
                .fetch::<m![{ Elements<O> }, I % (I / 2) / 32], m![I % 32]>()
                .collect::<m![{ Elements<O> }, I % (I / 2) / 32], m![I % 32]>()
                .contract_outer::<m![{ Elements<O> }, { Half<I> }, I % 128 / 64], m![I % 64], _, _, _>(
                    &quantized,
                )
                .contract_packet::<m![1]>()
                .contract_time::<m![{ Elements<O> }, { Half<I> }]>()
                .contract_lane::<m![{ Elements<O> }, { Half<I> }], m![1 # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &activation)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
                .vector_intra_slice_reduce::<I, Elements<O>, m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit_view(halves.view_mut().tile::<m![O % ({ Block<O> })], 4, m![
                    { Elements<O> } #{!} ({ Block<O> }), 1 # 8
                ]>(4 * tile));
        }
        let output: DmTensor<bf16, Chips, Cluster, Reduced<O>, m![O % ({ Block<O> })]> = device
            .main
            .begin(halves.view())
            .fetch::<m![O % ({ Block<O> })], m![1 # 8]>()
            .collect::<m![O % ({ Block<O> })], m![1 # 8]>()
            .vector_init()
            .vector_inter_slice_reduce::<Reduced<O>, m![O % ({ Block<O> })]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![O % ({ Block<O> }) / 4], m![O % 4 # 16]>()
            .commit_trim::<m![O % 4]>()
            .commit();
        output.to_hbm(&mut device.tdma)
    }
}

impl<I: AxisName, O: AxisName, Chips: M> Module<(&mut Device, &Vector<I, bf16, Chips>)>
    for BlockScaled<I, O, Chips>
{
    type Output = Vector<O, bf16, Chips>;

    fn forward(&self, (device, input): (&mut Device, &Vector<I, bf16, Chips>)) -> Self::Output {
        self.forward((device, input.view()))
    }
}

impl<I: AxisName, O: AxisName, Heads: AxisName, Head: AxisName, Chips: M>
    Module<(&mut Device, &HbmTensor<bf16, Chips, m![Heads, Head]>)> for BlockScaled<I, O, Chips>
{
    type Output = Vector<O, bf16, Chips>;

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
pub(super) mod tests {
    use super::*;

    axes![
        Narrow = 768,
        Square = 768,
        Middle = 512,
        Keys = 256,
        Wide = 1024
    ];

    #[device(chip = 1)]
    fn projected<I: AxisName, O: AxisName>(
        device: &mut Device,
        linear: &BlockScaled<I, O>,
        input: &Vector<I>,
    ) -> Vector<O> {
        linear.forward((device, input))
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

    /// `grid`'s scales in the device form: each block row once per four output rows, each input
    /// half's blocks padded to an even count.
    fn stored<I: AxisName, O: AxisName>(grid: &[f32]) -> Vec<f32> {
        let (half, padded) = (I::SIZE / (2 * GROUP), Blocks::<I>::SIZE);
        (0..O::SIZE / 4)
            .flat_map(|tile| (0..2 * padded).map(move |at| (tile, at / padded, at % padded)))
            .map(|(tile, side, block)| match block < half {
                true => grid[tile * 4 / GROUP * (2 * half) + side * half + block],
                false => 0.0,
            })
            .collect()
    }

    async fn land<I: AxisName, O: AxisName>(
        device: &mut Device,
        input: &[bf16],
        weight: &[f8e4m3],
        grid: &[f32],
    ) -> Vec<f32> {
        let linear: BlockScaled<I, O> = BlockScaled {
            weight: HostTensor::<f8e4m3, m![O, I]>::from_vec(weight.to_vec())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            weight_scale: HostTensor::<
                f32,
                m![O / 4, I / (I / 2), I % (I / 2) / 128 # ({ Symbol<Blocks<I>> })],
            >::from_vec(stored::<I, O>(grid))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
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
        let _lease = crate::tests::DEVICE.lock().await;
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
        let landed = land::<Narrow, Square>(&mut device, &input, &weight, &grid).await;
        let expected = reference(&input, &weight, &grid);
        for (row, (landed, expected)) in landed.iter().zip(&expected).enumerate() {
            assert_eq!(
                *landed,
                bf16::from_f32(*expected).to_f32(),
                "row {row} reads {}",
                input[source(row)].to_f32()
            );
        }
    }

    /// Activation group exponents of `inputs`: a group's largest input is 448 times two to its
    /// exponent, so its scale is exactly that power of two.
    pub(in super::super) const EXPONENTS: [i32; 6] = [-1, 0, 1, 2, -2, 1];

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
    /// up: every row lands the reference exactly, and one scale after the whole sum does not.
    #[tokio::test(flavor = "current_thread")]
    async fn scales_each_block_before_the_sum() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        let (inputs, outputs) = (Narrow::SIZE, Middle::SIZE);
        let input = powers(inputs);
        let (weight, grid) = integers(outputs, inputs, 0);
        let landed = land::<Narrow, Middle>(&mut device, &input, &weight, &grid).await;
        let expected = reference(&input, &weight, &grid);
        let mut differs = 0;
        for (row, (landed, expected)) in landed.iter().zip(&expected).enumerate() {
            assert_eq!(*landed, bf16::from_f32(*expected).to_f32(), "row {row}");
            let whole: f32 = (0..inputs)
                .map(|index| {
                    let code = input[index].to_f32() / 2f32.powi(EXPONENTS[index / GROUP]);
                    weight[row * inputs + index].to_f32() * code
                })
                .sum();
            let once = whole * 2f32.powi(EXPONENTS[0]) * grid[row / GROUP * (inputs / GROUP)];
            differs += usize::from(bf16::from_f32(once).to_f32() != *landed);
        }
        assert!(
            differs > outputs / 2,
            "{differs} rows tell the schemes apart"
        );
    }

    /// Query, key and value projections of one input each land the reference through their own
    /// block scales; key and value share their weights and differ only by those scales.
    #[tokio::test(flavor = "current_thread")]
    async fn keeps_each_projection_scales() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        let input = powers(Narrow::SIZE);
        let (query, query_grid) = integers(Middle::SIZE, Narrow::SIZE, 1);
        let (key, key_grid) = integers(Keys::SIZE, Narrow::SIZE, 2);
        let value_grid: Vec<f32> = key_grid.iter().map(|scale| scale * 4.0).collect();
        let queries = land::<Narrow, Middle>(&mut device, &input, &query, &query_grid).await;
        let keys = land::<Narrow, Keys>(&mut device, &input, &key, &key_grid).await;
        let values = land::<Narrow, Keys>(&mut device, &input, &key, &value_grid).await;
        for (landed, weight, grid) in [
            (&queries, &query, &query_grid),
            (&keys, &key, &key_grid),
            (&values, &key, &value_grid),
        ] {
            let expected = reference(&input, weight, grid);
            for (row, (landed, expected)) in landed.iter().zip(&expected).enumerate() {
                assert_eq!(*landed, bf16::from_f32(*expected).to_f32(), "row {row}");
            }
        }
        assert_ne!(keys, values);
    }

    /// Random activations with outliers, E4M3 weights over their whole range and BF16 block
    /// scales land within one BF16 step of the reference's FP32 outputs.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_the_reference_quantizer() {
        let _lease = crate::tests::DEVICE.lock().await;
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
        let landed = land::<Wide, Narrow>(&mut device, &input, &weight, &grid).await;
        let expected = reference(&input, &weight, &grid);
        for (row, (landed, expected)) in landed.iter().zip(&expected).enumerate() {
            let step = 2f32.powi(expected.abs().max(f32::MIN_POSITIVE).log2().floor() as i32 - 7);
            assert!(
                (landed - expected).abs() <= step,
                "row {row}: {landed} vs {expected}"
            );
        }
    }
}
