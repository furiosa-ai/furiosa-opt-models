//! NVFP4 projections whose input rows quantize as their weights do.
//!
//! - A weight row holds E2M1 codes with one E4M3 scale per 16 inputs, under one FP32 tensor scale.
//! - Each call quantizes its input row the same way: a 16-input block's E4M3 scale comes from its
//!   largest magnitude under the input's FP32 tensor scale, and its codes round to nearest even,
//!   bit for bit as the reference quantizer `ref_nvfp4_quant` does.
//! - Both operands enter the contraction as code times block scale, which BF16 holds exactly, and
//!   the FP32 sum takes both tensor scales once, as a block-scaled FP4 product does.
//!
//! Source: <https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py>.

use super::projection::{Block, Cluster, Reduced, Slices};
use crate::{
    Module,
    ops::{layout::axes::Chip, vector::scalar},
};
use furiosa_opt_std::prelude::BitReq::{Ignore, One};
use furiosa_opt_std::prelude::*;

mod rows;

/// FP32 scalar in the first lane of an eight-element padded vector.
pub type Scale = HbmTensor<f32, Chip, m![1 # 8]>;

/// One contiguous feature vector.
pub type Vector<W> = HbmTensor<bf16, Chip, m![W]>;

/// E2M1 weights and inputs, each with E4M3 scales per 16 inputs and an FP32 tensor scale; the
/// inputs quantize at run time.
#[derive(DeviceSend)]
pub struct BlockScaled<In: AxisName, Out: AxisName> {
    /// E2M1 coefficients in [output, input] order.
    pub weight: HbmTensor<f4e2m1, Chip, m![Out, In]>,
    /// E4M3 scale of each 16 consecutive coefficients of a row.
    pub weight_scale: HbmTensor<f8e4m3, Chip, m![Out, In / 16]>,
    /// The tensor scales, in the forms the quantizer and the contraction read.
    pub scales: Scales,
}

/// FP32 scalars derived from a checkpoint's input scale (its calibrated input maximum over
/// 6 x 448) and weight tensor scale.
#[derive(DeviceSend)]
pub struct Scales {
    /// `1 / input_scale`: block scales are E4M3 in these units.
    pub global: Scale,
    /// `1 / global`, which returns a block scale to input units.
    pub restore: Scale,
    /// `input_scale * weight_scale_2`, which returns the contracted sum to output units.
    pub output: Scale,
}

/// `input` with each element replaced by its E2M1 code times its block's E4M3 scale.
impl<I: AxisName> Module<(&mut Device, Half<I>)> for Scales {
    type Output = Half<I>;

    fn forward(&self, (device, input): (&mut Device, Half<I>)) -> Half<I> {
        // SAFETY: relabels each slice's half row as its consecutive 16-input blocks.
        let input: Grouped<I, bf16> = unsafe { input.reshape() };
        let global: Packet<I> = self.global.to_dm(&mut device.tdma);
        let global = scalar::register(device, &global);
        let restore: Packet<I> = self.restore.to_dm(&mut device.tdma);
        let restore = scalar::register(device, &restore);
        // Rounding is monotone, so the largest rounded `|x| / 6 * global` of a block is the
        // reference's `global * (amax / 6)`.
        let scale: PerBlock<I, f8e4m3> = device
            .main
            .begin(input.view())
            .fetch::<m![{ Blocks<I> }], m![Group]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Blocks<I> }, Group / 8], m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>()
            .vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX)
            .vector_reinterpret::<f32>()
            .vector_narrow_split::<m![{ Blocks<I> }, Group / 4], m![Group % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), const { 1.0f32 / E2M1 })
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &global)
            .vector_intra_slice_reduce::<Group, m![{ Blocks<I> }], m![1 # 4]>(
                IntraSliceReduceOpF32::Max,
            )
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Min, E4M3)
            .vector_final()
            .cast::<f8e4m3, m![1 # 32]>()
            .transpose::<m![{ Blocks<I> } / 8], m![{ Blocks<I> } % 8 # 32]>()
            .commit_trim::<m![{ Blocks<I> } % 8]>()
            .commit();
        // A zero scale gives an infinite reciprocal; capping it keeps `0 * reciprocal` finite.
        let reciprocal: PerBlock<I, f32> = device
            .main
            .begin(scale.view())
            .fetch::<m![{ Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Blocks<I> } / 4], m![{ Blocks<I> } % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &restore)
            .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
            .vector_widen_concat::<m![{ Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
            .vector_clip(ClipBinaryOpF32::Min, f32::MAX)
            .vector_final()
            .commit_trim::<m![{ Blocks<I> } % 8]>()
            .commit();
        let reciprocal: Register<I> = device
            .sub
            .begin(reciprocal.view())
            .fetch::<m![{ Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
            .collect::<m![{ Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
            .to_vrf();
        let scaled: Grouped<I, f32> = device
            .main
            .begin(input.view())
            .fetch::<m![{ Blocks<I> }], m![Group]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Blocks<I> }, Group / 8], m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Blocks<I> }, Group / 4], m![Group % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &reciprocal)
            .vector_widen_concat::<m![{ Blocks<I> }, Group / 8], m![Group % 8]>()
            .vector_clip(ClipBinaryOpF32::Max, -E2M1)
            .vector_clip(ClipBinaryOpF32::Min, E2M1)
            .vector_final()
            .commit_trim::<m![Group % 8]>()
            .commit();
        // Adding 1.5 * 2^(22 + k) rounds to the E2M1 spacing 2^(k - 1) of the value's binade,
        // to nearest even; subtracting it again is exact.
        let below = TagGuard::matches([One, One, Ignore, Ignore]);
        let within = TagGuard::matches([Ignore, Ignore, One, One]);
        let rounded: Grouped<I, f32> = device
            .main
            .begin(scaled.view())
            .fetch::<m![{ Blocks<I> }, Group / 8], m![Group % 8]>()
            .collect::<m![{ Blocks<I> }, Group / 8], m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Less(2.0f32),
                Cmp::Greater(-2.0f32),
                Cmp::Less(4.0f32),
                Cmp::Greater(-4.0f32),
            ]))
            .vector_narrow_split::<m![{ Blocks<I> }, Group / 4], m![Group % 4]>()
            .vector_fp_binary(
                FpBinaryOp::AddF,
                Branched::imm(below, ROUND[0])
                    .imm(within, ROUND[1])
                    .imm(TagGuard::all(), ROUND[2]),
            )
            .vector_widen_concat::<m![{ Blocks<I> }, Group / 8], m![Group % 8]>()
            .vector_clip(
                ClipBinaryOpF32::Add,
                Branched::imm(below, -ROUND[0])
                    .imm(within, -ROUND[1])
                    .imm(TagGuard::all(), -ROUND[2]),
            )
            .vector_final()
            .commit_trim::<m![Group % 8]>()
            .commit();
        let scale: Register<I> = device
            .sub
            .begin(scale.view())
            .fetch::<m![{ Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
            .to_vrf();
        let quantized: Grouped<I, bf16> = device
            .main
            .begin(rounded.view())
            .fetch::<m![{ Blocks<I> }, Group / 8], m![Group % 8]>()
            .collect::<m![{ Blocks<I> }, Group / 8], m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Blocks<I> }, Group / 4], m![Group % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![{ Blocks<I> }, Group / 8], m![Group % 8]>()
            .vector_final()
            .cast::<bf16, m![Group % 8 # 16]>()
            .commit_trim::<m![Group % 8]>()
            .commit();
        // SAFETY: the inverse of the relabel above.
        unsafe { quantized.reshape() }
    }
}

/// The largest E2M1 magnitude: a block's largest input quantizes to it.
const E2M1: f32 = 6.0;

/// The largest finite E4M3 magnitude, which caps a block scale.
const E4M3: f32 = 448.0;

/// Magic constants rounding magnitudes below 2, below 4 and up to 6 to E2M1 steps.
const ROUND: [f32; 3] = [
    1.5 * (1 << 22) as f32,
    1.5 * (1 << 23) as f32,
    1.5 * (1 << 24) as f32,
];

// The 16 inputs one E4M3 scale covers, apart from the block they belong to, and the four output
// rows a slice decodes at a time, apart from its block of rows.
axes![Group = 16, Row = 4];

/// The 16-input blocks of the input half one slice holds.
type Blocks<I> = m![I % (I / 2) / 16];

/// Each slice's input half; every output block's slices hold a copy.
type Halves<I> = m![64, 1 # 2, I / (I / 2)];
type Half<I, D = bf16> = DmTensor<D, Chip, Cluster, Halves<I>, m![I % (I / 2)]>;
type Grouped<I, D> = DmTensor<D, Chip, Cluster, Halves<I>, m![{ Blocks<I> }, Group]>;
type PerBlock<I, D> = DmTensor<D, Chip, Cluster, Halves<I>, m![{ Blocks<I> }]>;
type Register<I> = VrfTensor<f32, Chip, Cluster, Halves<I>, m![{ Blocks<I> }]>;
type Packet<I> = DmTensor<f32, Chip, Cluster, Halves<I>, m![1 # 8]>;

type Elements<O> = m![O % ({ Block<O> }) = 4];
type Tile<I, O, D = bf16> =
    DmTensor<D, Chip, Cluster, Slices<I, O>, m![{ Elements<O> }, I % (I / 2)]>;
type Rows<I, O, D> = DmTensor<D, Chip, Cluster, Slices<I, O>, m![Row, I % (I / 2)]>;

impl<I: AxisName, O: AxisName> Module<(&mut Device, &Vector<I>)> for BlockScaled<I, O> {
    type Output = Vector<O>;

    /// Project one contiguous BF16 input row; both widths must be positive multiples of 256, and
    /// two rows' block scales of each slice's input half must fit its 8 KiB of vector registers.
    fn forward(&self, (device, input): (&mut Device, &Vector<I>)) -> Self::Output {
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
            let packed: Tile<I, O, f4e2m1> = self.weight.view().tile::<m![O % ({ Block<O> })], 4, m![O / ({ Block<O> }), { Elements<O> } # ({ Block<O> }), I]>(index).to_dm(&mut device.tdma);
            let packed: Tile<I, O, f8e4m3> = device
                .main
                .begin(packed.view())
                .fetch::<Elements<O>, m![I % (I / 2)]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![{ Elements<O> }, I % (I / 2) / 32], m![I % 32]>()
                .commit_trim::<m![I % 32]>()
                .commit();
            // SAFETY: names the tile's four rows apart from the block they belong to.
            let packed: Rows<I, O, f8e4m3> = unsafe { packed.reshape() };
            let scale: DmTensor<f8e4m3, Chip, Cluster, Slices<I, O>, m![{ Elements<O> }, I % (I / 2) / 16]> = self.weight_scale.view().tile::<m![O % ({ Block<O> })], 4, m![O / ({ Block<O> }), { Elements<O> } # ({ Block<O> }), I / 16]>(index).to_dm(&mut device.tdma);
            // SAFETY: as for the codes.
            let scale: DmTensor<f8e4m3, Chip, Cluster, Slices<I, O>, m![Row, I % (I / 2) / 16]> =
                unsafe { scale.reshape() };
            let mut tile: Rows<I, O, bf16> = DmTensor::new();
            for pair in 0..2 {
                let scale: VrfTensor<
                    f32,
                    Chip,
                    Cluster,
                    Slices<I, O>,
                    m![Row % 2, I % (I / 2) / 16],
                > = device
                    .sub
                    .begin(
                        scale
                            .view()
                            .tile::<m![Row / 2], 1, m![Row / 2 = 1 # 2, Row % 2, I % (I / 2) / 16]>(
                                pair,
                            ),
                    )
                    .fetch::<m![Row % 2], m![I % (I / 2) / 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Row % 2, I % (I / 2) / 128], m![I / 16 % 8]>()
                    .to_vrf();
                device
                    .main
                    .begin(
                        packed
                            .view()
                            .tile::<m![Row / 2], 1, m![Row / 2 = 1 # 2, Row % 2, I % (I / 2)]>(
                                pair,
                            ),
                    )
                    .fetch::<m![Row % 2, I % (I / 2) / 32], m![I % 32]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Row % 2, I % (I / 2) / 8], m![I % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Row % 2, I % (I / 2) / 4], m![I % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
                    .vector_widen_concat::<m![Row % 2, I % (I / 2) / 8], m![I % 8]>()
                    .vector_final()
                    .cast::<bf16, m![I % 8 # 16]>()
                    .commit_trim::<m![I % 8]>()
                    .commit_view(
                        tile.view_mut()
                            .tile::<m![Row / 2], 1, m![Row / 2 = 1 #{!} 2, Row % 2, I % (I / 2)]>(
                                pair,
                            ),
                    );
            }
            // SAFETY: the inverse of the relabels above.
            unsafe { tile.reshape() }
        };
        let input: Half<I> = input.to_dm(&mut device.tdma);
        let input = self.scales.forward((device, input));
        // SAFETY: names each copy of the input half after the output block its slices project.
        let input: DmTensor<bf16, Chip, Cluster, Slices<I, O>, m![I % (I / 2)]> =
            unsafe { input.reshape() };
        let input: TrfTensor<bf16, Chip, Cluster, Slices<I, O>, m![1], m![I % (I / 2)]> = device
            .sub
            .begin(input.view())
            .fetch::<m![I % (I / 2) / 16], m![I % 16]>()
            .collect::<m![I % (I / 2) / 16], m![I % 16]>()
            .to_trf();
        let scale: DmTensor<f32, Chip, Cluster, Reduced<O>, m![1 # 8]> =
            self.scales.output.to_dm(&mut device.tdma);
        let scale = scalar::register(device, &scale);
        let mut output: DmTensor<bf16, Chip, Cluster, Reduced<O>, m![O % ({ Block<O> })]> =
            DmTensor::new();
        for index in 0..(O::SIZE / 256) {
            let tile = read(device, 4 * index);
            let partial: DmTensor<f32, Chip, Cluster, Slices<I, O>, m![{ Elements<O> }, 1 # 8]> =
                device
                    .main
                    .begin(tile.view())
                    .fetch::<m![{ Elements<O> }, I % (I / 2) / 16], m![I % 16]>()
                    .collect::<m![{ Elements<O> }, I % (I / 2) / 16], m![I % 16]>()
                    .contract_outer::<m![{ Elements<O> }, I % (I / 2) / 32], m![I % 32], _, _, _>(
                        &input,
                    )
                    .contract_packet::<m![1]>()
                    .contract_time::<Elements<O>>()
                    .contract_lane::<Elements<O>, m![1 # 8]>(LaneMode::Interleaved)
                    .commit_trim::<m![1 # 8]>()
                    .commit();
            let value: DmTensor<bf16, Chip, Cluster, Reduced<O>, Elements<O>> = device
                .main
                .begin(partial.view())
                .fetch::<Elements<O>, m![1 # 8]>()
                .collect::<Elements<O>, m![1 # 8]>()
                .vector_init()
                .vector_inter_slice_reduce::<Reduced<O>, Elements<O>>(InterSliceReduceOpF32::Add)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .cast::<bf16, m![1 # 16]>()
                .transpose::<m![1], m![{ Elements<O> } # 16]>()
                .commit_trim::<Elements<O>>()
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
mod tests {
    use super::*;
    use furiosa_opt_std::runtime::launch;

    axes![Input = 512, Output = 512, Wide = 21504, Narrow = 256];

    #[device(chip = 1)]
    fn projected<I: AxisName, O: AxisName>(
        device: &mut Device,
        linear: &BlockScaled<I, O>,
        input: &Vector<I>,
    ) -> Vector<O> {
        linear.forward((device, input))
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
                let scale = f8e4m3::from_f32((global * (maximum * (1.0 / 6.0))).min(E4M3)).to_f32();
                let reciprocal = 1.0 / (scale * restore);
                block.iter().map(move |x| {
                    if scale == 0.0 {
                        0.0
                    } else {
                        e2m1((x.to_f32() * reciprocal).clamp(-E2M1, E2M1)) * scale
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
        HostTensor::<f32, m![1 # 8]>::from_vec(std::iter::once(value).chain([0.0; 7]))
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
    ) -> BlockScaled<I, O> {
        let global = 1.0 / input_scale;
        let [restore, output] = [1.0 / global, input_scale * weight_scale_2];
        BlockScaled {
            weight: HostTensor::<f4e2m1, m![O, I]>::from_vec(codes.to_vec())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            weight_scale: HostTensor::<f8e4m3, m![O, I / 16]>::from_vec(scales.to_vec())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            scales: Scales {
                global: scalar(device, global).await,
                restore: scalar(device, restore).await,
                output: scalar(device, output).await,
            },
        }
    }

    async fn run<I: AxisName, O: AxisName>(
        device: &mut Device,
        linear: &BlockScaled<I, O>,
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

    /// A weight whose row `o` selects input `o` lands each quantized input times the input scale:
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
            let linear: BlockScaled<Input, Output> =
                linear(&mut device, &codes, &scales, input_scale, 1.0).await;
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
                        // `x * reciprocal` lands on the E2M1 ties when the input scale is 2^-5.
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
            let landed: Vec<u16> = run(&mut device, &linear, &input)
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
            assert_eq!(landed, expected, "input scale {input_scale}");
        }
    }

    /// Random codes, block scales and inputs land the exact sum of the code-times-scale products
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
        let linear: BlockScaled<I, O> =
            linear(device, &codes, &scales, input_scale, weight_scale_2).await;
        let landed = run(device, &linear, &input).await;
        let operand = quantized(&input, input_scale);
        let output = f64::from(input_scale) * f64::from(weight_scale_2);
        for (row, landed) in landed.iter().enumerate() {
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
                (landed.to_f32() as f64 - exact).abs() <= bound,
                "{}x{} row {row}: {} against {exact}",
                I::SIZE,
                O::SIZE,
                landed.to_f32()
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
