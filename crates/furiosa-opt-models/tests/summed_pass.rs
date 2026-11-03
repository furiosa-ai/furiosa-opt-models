//! The passes of one summed NVFP4 projection alone, at the feed-forward's shape: each slice owns 60
//! outputs over a part of 1,920 inputs and sums them in passes over a lane group of rows.
//!
//! - A pass loads its outputs' FP32 block scales into the vector registers on Sub, then on Main
//!   decodes their E2M1 codes through the fetch table, sums each 16-input block against every
//!   lane, scales and adds the blocks, divides by the tensor scale and adds the two parts of a row.
//! - Each form changes one thing: one row instead of eight, the vector engine on four lanes
//!   instead of eight, BF16 weights instead of codes, passes of four outputs instead of twelve,
//!   four outputs' blocks interleaved in time, or the block sums added without their scales.
//! - Every form but the last gives the same sums for the rows it holds.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::ops::normalization::Spread;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_std::{prelude::*, runtime::launch};
use tokio::sync::Mutex;

/// The device tests share one chip, so they run one at a time.
static DEVICE: Mutex<()> = Mutex::const_new(());

axes![
    I = 3840,
    O = 15360,
    One = 1,
    Eight = 8,
    Four = 4,
    Twelve = 12
];

/// Each cluster holds half of the outputs.
type Cluster = m![O / 7680];
/// A slice owns 60 consecutive outputs over one of a row's two parts of 1,920 inputs; the parts
/// of an output are adjacent slices, for the inter-slice reducer.
type Owners = m![O % 7680 / 60, I / 1920];
/// The slices holding a row's sums once its parts add.
type Reduced = m![O % 7680 / 60, 1 # 2];

type Codes = DmTensor<f4e2m1, Chip, Cluster, Owners, m![O % 60, I % 1920]>;
type Weights = DmTensor<bf16, Chip, Cluster, Owners, m![O % 60, I % 1920]>;
type Blocks = DmTensor<f8e4m3, Chip, Cluster, Owners, m![O % 60, I % 1920 / 16]>;
type Widened = DmTensor<f32, Chip, Cluster, Owners, m![O % 60, I % 1920 / 16]>;
type Global = VrfTensor<f32, Chip, Cluster, Owners, m![1 # 8]>;
/// A lane group's rows in every slice's tensor registers, each slice its part's inputs.
type Lanes<R> = TrfTensor<bf16, Chip, Cluster, Owners, m![R], m![1, I % 1920]>;
/// The passes over a slice's outputs, and the outputs of one pass.
type Tiles<H> = m![O % 60 / ({ Symbol<H> })];
type Elements<H> = m![O % ({ Symbol<H> })];
/// The FP32 block scales of one pass's outputs.
type Scaling<H> = VrfTensor<f32, Chip, Cluster, Owners, m![{ Elements<H> }, I % 1920 / 16]>;
type Summing<R> = DmTensor<bf16, Chip, Cluster, Reduced, m![R, O % 60]>;

type HbmCodes = HbmTensor<f4e2m1, Chip, m![O, I]>;
type HbmWeights = HbmTensor<bf16, Chip, m![O, I]>;
type HbmBlocks = HbmTensor<f8e4m3, Chip, m![O, I / 16]>;
type HbmGlobal = HbmTensor<f32, Chip, m![1 # 8]>;
type HbmRows<R> = HbmTensor<bf16, Chip, m![R, I]>;
type HbmSums<R> = HbmTensor<bf16, Chip, m![R, O]>;

/// The block scales widened to FP32 once, so each pass loads its outputs' as they are.
fn widen(device: &mut Device, blocks: &Blocks) -> Widened {
    let mut scales = Widened::new();
    device
        .sub
        .begin(blocks.view())
        .fetch::<m![O % 60], m![I % 1920 / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![O % 60, I % 1920 / 128], m![I / 16 % 8]>()
        .commit_trim::<m![I / 16 % 8]>()
        .commit_view(scales.view_mut());
    scales
}

/// The tensor scale in every slice's vector registers.
fn register(device: &mut Device, global: &HbmGlobal) -> Global {
    let global: DmTensor<f32, Chip, Cluster, Owners, m![1 # 8]> = global.to_dm(&mut device.tdma);
    device
        .sub
        .begin(global.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// A lane group's rows from their feature slices into every slice's tensor registers.
fn fill<R: AxisName>(device: &mut Device, rows: &Spread<I, R, Chip>) -> Lanes<R> {
    // SAFETY: both physical clusters hold every row; only the logical cluster axis changes.
    let rows = unsafe {
        rows.view()
            .reshape::<Chip, Cluster, m![I / 16 # 256], m![R, I % 16]>()
    };
    device
        .sub
        .begin(rows)
        .fetch::<m![R], m![I % 16]>()
        .switch::<Owners, m![R, I % 1920 / 16]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![R, I % 1920 / 16], m![I % 16]>()
        .to_trf()
}

/// One pass's outputs' block scales into the vector registers.
fn load<H: AxisName>(device: &mut Device, scales: &Widened, tile: usize) -> Scaling<H> {
    device
        .sub
        .begin(scales.view().tile::<Tiles<H>, 1, m![
            { Elements<H> } # 60, I % 1920 / 16
        ]>(tile))
        .fetch::<m![{ Elements<H> }, I % 1920 / 128], m![I / 16 % 8]>()
        .collect::<m![{ Elements<H> }, I % 1920 / 128], m![I / 16 % 8]>()
        .to_vrf()
}

/// What one pass reads its weights from, and how its vector engine folds the lanes.
trait Pass<R: AxisName, H: AxisName> {
    fn pass(
        &self,
        device: &mut Device,
        operands: (&Lanes<R>, &Scaling<H>, &Global),
        tile: usize,
        sums: &mut Summing<R>,
    );
}

/// The codes through the fetch table, the vector engine on all eight lanes.
impl<R: AxisName, H: AxisName> Pass<R, H> for Codes {
    fn pass(
        &self,
        device: &mut Device,
        (lanes, scale, global): (&Lanes<R>, &Scaling<H>, &Global),
        tile: usize,
        sums: &mut Summing<R>,
    ) {
        device
            .main
            .begin(
                self.view()
                    .tile::<Tiles<H>, 1, m![{ Elements<H> } # 60, I % 1920]>(tile),
            )
            .fetch::<m![{ Elements<H> }, I % 1920 / 64], m![I % 64]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![{ Elements<H> }, I % 1920 / 16], m![I % 16]>()
            .contract_outer::<m![{ Elements<H> }, I % 1920 / 16, 1], m![I % 16], _, _, _>(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Elements<H> }, I % 1920 / 16, 1]>()
            .contract_lane::<m![{ Elements<H> }, I % 1920 / 16, 1], m![R # 8]>(
                LaneMode::Interleaved,
            )
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Elements<H> }, I % 1920 / 16, 1, R # 8 / 4], m![R # 8 % 4]>(
            )
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
            .vector_intra_slice_reduce::<I, m![{ Elements<H> }, 1, R # 8 / 4], m![R # 8 % 4]>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_fp_div(global)
            .vector_widen_concat::<m![{ Elements<H> }, 1], m![R # 8]>()
            .vector_inter_slice_reduce::<Reduced, m![{ Elements<H> }, 1]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .cast::<bf16, m![R # 16]>()
            .transpose::<m![{ Elements<H> } / 4, 1, R], m![O % 4 # 16]>()
            .commit_trim::<m![O % 4]>()
            .commit_view(sums.view_mut().tile::<Tiles<H>, 1, m![
                R, { Elements<H> } #{!} 60
            ]>(tile));
    }
}

/// The codes through the fetch table, the vector engine on the first four lanes alone.
struct Trimmed(Codes);

impl<R: AxisName, H: AxisName> Pass<R, H> for Trimmed {
    fn pass(
        &self,
        device: &mut Device,
        (lanes, scale, global): (&Lanes<R>, &Scaling<H>, &Global),
        tile: usize,
        sums: &mut Summing<R>,
    ) {
        const { assert!(R::SIZE <= 4) };
        device
            .main
            .begin(
                self.0
                    .view()
                    .tile::<Tiles<H>, 1, m![{ Elements<H> } # 60, I % 1920]>(tile),
            )
            .fetch::<m![{ Elements<H> }, I % 1920 / 64], m![I % 64]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![{ Elements<H> }, I % 1920 / 16], m![I % 16]>()
            .contract_outer::<m![{ Elements<H> }, I % 1920 / 16, 1], m![I % 16], _, _, _>(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Elements<H> }, I % 1920 / 16, 1]>()
            .contract_lane::<m![{ Elements<H> }, I % 1920 / 16, 1], m![R # 8]>(
                LaneMode::Interleaved,
            )
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![R # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
            .vector_intra_slice_reduce::<I, m![{ Elements<H> }, 1], m![R # 4]>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_fp_div(global)
            .vector_widen_pad::<m![R # 8]>()
            .vector_inter_slice_reduce::<Reduced, m![{ Elements<H> }, 1]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .cast::<bf16, m![R # 16]>()
            .transpose::<m![{ Elements<H> } / 4, 1, R], m![O % 4 # 16]>()
            .commit_trim::<m![O % 4]>()
            .commit_view(sums.view_mut().tile::<Tiles<H>, 1, m![
                R, { Elements<H> } #{!} 60
            ]>(tile));
    }
}

/// The codes through the fetch table, four outputs' blocks interleaved in time, so the block sums
/// of eight lane halves accumulate side by side; each output still adds its blocks in order.
struct Slotted(Codes);

impl<R: AxisName, H: AxisName> Pass<R, H> for Slotted {
    fn pass(
        &self,
        device: &mut Device,
        (lanes, scale, global): (&Lanes<R>, &Scaling<H>, &Global),
        tile: usize,
        sums: &mut Summing<R>,
    ) {
        device
            .main
            .begin(
                self.0
                    .view()
                    .tile::<Tiles<H>, 1, m![{ Elements<H> } # 60, I % 1920]>(tile),
            )
            .fetch::<m![{ Elements<H> } / 4, I % 1920 / 64, O % 4], m![I % 64]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![{ Elements<H> } / 4, I % 1920 / 64, O % 4, I % 64 / 16], m![I % 16]>()
            .contract_outer::<
                m![{ Elements<H> } / 4, I % 1920 / 64, O % 4, I % 64 / 16, 1],
                m![I % 16],
                _,
                _,
                _,
            >(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Elements<H> } / 4, I % 1920 / 64, O % 4, I % 64 / 16, 1]>()
            .contract_lane::<
                m![{ Elements<H> } / 4, I % 1920 / 64, O % 4, I % 64 / 16, 1],
                m![R # 8],
            >(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![{ Elements<H> } / 4, I % 1920 / 64, O % 4, I % 64 / 16, 1, R # 8 / 4],
                m![R # 8 % 4],
            >()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
            .vector_intra_slice_reduce::<
                I,
                m![{ Elements<H> } / 4, O % 4, 1, R # 8 / 4],
                m![R # 8 % 4],
            >(IntraSliceReduceOpF32::Add)
            .vector_fp_div(global)
            .vector_widen_concat::<m![{ Elements<H> } / 4, O % 4, 1], m![R # 8]>()
            .vector_inter_slice_reduce::<Reduced, m![{ Elements<H> } / 4, O % 4, 1]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .cast::<bf16, m![R # 16]>()
            .transpose::<m![{ Elements<H> } / 4, 1, R], m![O % 4 # 16]>()
            .commit_trim::<m![O % 4]>()
            .commit_view(sums.view_mut().tile::<Tiles<H>, 1, m![
                R, { Elements<H> } #{!} 60
            ]>(tile));
    }
}

/// The codes through the fetch table, four outputs' blocks interleaved in time and the vector
/// engine on the first four lanes alone: four accumulators side by side.
struct SlottedTrimmed(Codes);

impl<R: AxisName, H: AxisName> Pass<R, H> for SlottedTrimmed {
    fn pass(
        &self,
        device: &mut Device,
        (lanes, scale, global): (&Lanes<R>, &Scaling<H>, &Global),
        tile: usize,
        sums: &mut Summing<R>,
    ) {
        const { assert!(R::SIZE <= 4) };
        device
            .main
            .begin(
                self.0
                    .view()
                    .tile::<Tiles<H>, 1, m![{ Elements<H> } # 60, I % 1920]>(tile),
            )
            .fetch::<m![{ Elements<H> } / 4, I % 1920 / 64, O % 4], m![I % 64]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![{ Elements<H> } / 4, I % 1920 / 64, O % 4, I % 64 / 16], m![I % 16]>()
            .contract_outer::<
                m![{ Elements<H> } / 4, I % 1920 / 64, O % 4, I % 64 / 16, 1],
                m![I % 16],
                _,
                _,
                _,
            >(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Elements<H> } / 4, I % 1920 / 64, O % 4, I % 64 / 16, 1]>()
            .contract_lane::<
                m![{ Elements<H> } / 4, I % 1920 / 64, O % 4, I % 64 / 16, 1],
                m![R # 8],
            >(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![R # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
            .vector_intra_slice_reduce::<I, m![{ Elements<H> } / 4, O % 4, 1], m![R # 4]>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_fp_div(global)
            .vector_widen_pad::<m![R # 8]>()
            .vector_inter_slice_reduce::<Reduced, m![{ Elements<H> } / 4, O % 4, 1]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .cast::<bf16, m![R # 16]>()
            .transpose::<m![{ Elements<H> } / 4, 1, R], m![O % 4 # 16]>()
            .commit_trim::<m![O % 4]>()
            .commit_view(sums.view_mut().tile::<Tiles<H>, 1, m![
                R, { Elements<H> } #{!} 60
            ]>(tile));
    }
}

/// The codes through the fetch table, the block sums added without their scales: the price of the
/// scale multiply alone, not a summed projection.
struct Unscaled(Codes);

impl<R: AxisName, H: AxisName> Pass<R, H> for Unscaled {
    fn pass(
        &self,
        device: &mut Device,
        (lanes, _, global): (&Lanes<R>, &Scaling<H>, &Global),
        tile: usize,
        sums: &mut Summing<R>,
    ) {
        device
            .main
            .begin(
                self.0
                    .view()
                    .tile::<Tiles<H>, 1, m![{ Elements<H> } # 60, I % 1920]>(tile),
            )
            .fetch::<m![{ Elements<H> }, I % 1920 / 64], m![I % 64]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![{ Elements<H> }, I % 1920 / 16], m![I % 16]>()
            .contract_outer::<m![{ Elements<H> }, I % 1920 / 16, 1], m![I % 16], _, _, _>(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Elements<H> }, I % 1920 / 16, 1]>()
            .contract_lane::<m![{ Elements<H> }, I % 1920 / 16, 1], m![R # 8]>(
                LaneMode::Interleaved,
            )
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Elements<H> }, I % 1920 / 16, 1, R # 8 / 4], m![R # 8 % 4]>(
            )
            .vector_intra_slice_reduce::<I, m![{ Elements<H> }, 1, R # 8 / 4], m![R # 8 % 4]>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_fp_div(global)
            .vector_widen_concat::<m![{ Elements<H> }, 1], m![R # 8]>()
            .vector_inter_slice_reduce::<Reduced, m![{ Elements<H> }, 1]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .cast::<bf16, m![R # 16]>()
            .transpose::<m![{ Elements<H> } / 4, 1, R], m![O % 4 # 16]>()
            .commit_trim::<m![O % 4]>()
            .commit_view(sums.view_mut().tile::<Tiles<H>, 1, m![
                R, { Elements<H> } #{!} 60
            ]>(tile));
    }
}

/// The codes decoded beforehand into BF16 weights, read without the fetch table.
impl<R: AxisName, H: AxisName> Pass<R, H> for Weights {
    fn pass(
        &self,
        device: &mut Device,
        (lanes, scale, global): (&Lanes<R>, &Scaling<H>, &Global),
        tile: usize,
        sums: &mut Summing<R>,
    ) {
        device
            .main
            .begin(
                self.view()
                    .tile::<Tiles<H>, 1, m![{ Elements<H> } # 60, I % 1920]>(tile),
            )
            .fetch::<m![{ Elements<H> }, I % 1920 / 16], m![I % 16]>()
            .collect::<m![{ Elements<H> }, I % 1920 / 16], m![I % 16]>()
            .contract_outer::<m![{ Elements<H> }, I % 1920 / 16, 1], m![I % 16], _, _, _>(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Elements<H> }, I % 1920 / 16, 1]>()
            .contract_lane::<m![{ Elements<H> }, I % 1920 / 16, 1], m![R # 8]>(
                LaneMode::Interleaved,
            )
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Elements<H> }, I % 1920 / 16, 1, R # 8 / 4], m![R # 8 % 4]>(
            )
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
            .vector_intra_slice_reduce::<I, m![{ Elements<H> }, 1, R # 8 / 4], m![R # 8 % 4]>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_fp_div(global)
            .vector_widen_concat::<m![{ Elements<H> }, 1], m![R # 8]>()
            .vector_inter_slice_reduce::<Reduced, m![{ Elements<H> }, 1]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .cast::<bf16, m![R # 16]>()
            .transpose::<m![{ Elements<H> } / 4, 1, R], m![O % 4 # 16]>()
            .commit_trim::<m![O % 4]>()
            .commit_view(sums.view_mut().tile::<Tiles<H>, 1, m![
                R, { Elements<H> } #{!} 60
            ]>(tile));
    }
}

/// Every pass of a lane group of `R` rows through weights already in DM, `H` outputs a pass.
fn project<R: AxisName, H: AxisName>(
    device: &mut Device,
    weights: &impl Pass<R, H>,
    (blocks, global, rows): (&Blocks, &HbmGlobal, &HbmRows<R>),
) -> HbmSums<R> {
    let global = register(device, global);
    let rows: Spread<I, R, Chip> = rows.to_dm(&mut device.tdma);
    let scales = widen(device, blocks);
    let lanes = fill(device, &rows);
    let mut sums = Summing::<R>::new();
    for tile in 0..<Tiles<H> as M>::SIZE {
        let scale = load::<H>(device, &scales, tile);
        weights.pass(device, (&lanes, &scale, &global), tile, &mut sums);
    }
    sums.to_hbm(&mut device.tdma)
}

/// Eight rows, twelve outputs a pass: the feed-forward's own pass.
#[device(chip = 1)]
fn eight_rows(
    device: &mut Device,
    (codes, blocks, global, rows): (&HbmCodes, &HbmBlocks, &HbmGlobal, &HbmRows<Eight>),
) -> HbmSums<Eight> {
    let codes: Codes = codes.to_dm(&mut device.tdma);
    let blocks: Blocks = blocks.to_dm(&mut device.tdma);
    project::<Eight, Twelve>(device, &codes, (&blocks, global, rows))
}

/// One row, the vector engine on all eight lanes.
#[device(chip = 1)]
fn one_row(
    device: &mut Device,
    (codes, blocks, global, rows): (&HbmCodes, &HbmBlocks, &HbmGlobal, &HbmRows<One>),
) -> HbmSums<One> {
    let codes: Codes = codes.to_dm(&mut device.tdma);
    let blocks: Blocks = blocks.to_dm(&mut device.tdma);
    project::<One, Twelve>(device, &codes, (&blocks, global, rows))
}

/// One row, the vector engine on four lanes.
#[device(chip = 1)]
fn four_lanes(
    device: &mut Device,
    (codes, blocks, global, rows): (&HbmCodes, &HbmBlocks, &HbmGlobal, &HbmRows<One>),
) -> HbmSums<One> {
    let codes: Codes = codes.to_dm(&mut device.tdma);
    let blocks: Blocks = blocks.to_dm(&mut device.tdma);
    project::<One, Twelve>(device, &Trimmed(codes), (&blocks, global, rows))
}

/// Eight rows through BF16 weights, no fetch table.
#[device(chip = 1)]
fn decoded_weights(
    device: &mut Device,
    (weights, blocks, global, rows): (&HbmWeights, &HbmBlocks, &HbmGlobal, &HbmRows<Eight>),
) -> HbmSums<Eight> {
    let weights: Weights = weights.to_dm(&mut device.tdma);
    let blocks: Blocks = blocks.to_dm(&mut device.tdma);
    project::<Eight, Twelve>(device, &weights, (&blocks, global, rows))
}

/// Eight rows, four outputs a pass: three times the passes for the same sums.
#[device(chip = 1)]
fn short_passes(
    device: &mut Device,
    (codes, blocks, global, rows): (&HbmCodes, &HbmBlocks, &HbmGlobal, &HbmRows<Eight>),
) -> HbmSums<Eight> {
    let codes: Codes = codes.to_dm(&mut device.tdma);
    let blocks: Blocks = blocks.to_dm(&mut device.tdma);
    project::<Eight, Four>(device, &codes, (&blocks, global, rows))
}

/// Eight rows, four outputs' blocks interleaved in time.
#[device(chip = 1)]
fn eight_rows_slotted(
    device: &mut Device,
    (codes, blocks, global, rows): (&HbmCodes, &HbmBlocks, &HbmGlobal, &HbmRows<Eight>),
) -> HbmSums<Eight> {
    let codes: Codes = codes.to_dm(&mut device.tdma);
    let blocks: Blocks = blocks.to_dm(&mut device.tdma);
    project::<Eight, Twelve>(device, &Slotted(codes), (&blocks, global, rows))
}

/// One row, four outputs' blocks interleaved in time, the vector engine on four lanes.
#[device(chip = 1)]
fn four_lanes_slotted(
    device: &mut Device,
    (codes, blocks, global, rows): (&HbmCodes, &HbmBlocks, &HbmGlobal, &HbmRows<One>),
) -> HbmSums<One> {
    let codes: Codes = codes.to_dm(&mut device.tdma);
    let blocks: Blocks = blocks.to_dm(&mut device.tdma);
    project::<One, Twelve>(device, &SlottedTrimmed(codes), (&blocks, global, rows))
}

/// Eight rows, the block sums added without their scales.
#[device(chip = 1)]
fn eight_rows_unscaled(
    device: &mut Device,
    (codes, blocks, global, rows): (&HbmCodes, &HbmBlocks, &HbmGlobal, &HbmRows<Eight>),
) -> HbmSums<Eight> {
    let codes: Codes = codes.to_dm(&mut device.tdma);
    let blocks: Blocks = blocks.to_dm(&mut device.tdma);
    project::<Eight, Twelve>(device, &Unscaled(codes), (&blocks, global, rows))
}

/// Pseudo-random 64-bit words from `state`.
fn words(count: usize, mut state: u64) -> impl Iterator<Item = u64> {
    (0..count).map(move |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    })
}

/// Random NVFP4 weights and eight rows on the device: codes over the sixteen E2M1 values, E4M3
/// block scales from 8 to 448, a tensor scale near 2^14 and inputs in `[-2, 2)`.
struct Drawn {
    codes: HbmCodes,
    weights: HbmWeights,
    blocks: HbmBlocks,
    global: HbmGlobal,
    eight: HbmRows<Eight>,
    one: HbmRows<One>,
    /// Every row's outputs in FP64: `sum_k code_k scale_k x_k / global`.
    exact: Vec<f64>,
}

impl Drawn {
    async fn new(device: &mut Device) -> Self {
        let unit = |word: u64| (word >> 11) as f64 / (1u64 << 53) as f64;
        let codes: Vec<u8> = (words(O::SIZE * I::SIZE / 2, 1))
            .map(|word| (word >> 56) as u8)
            .collect();
        let decoded: Vec<f32> = (0..O::SIZE * I::SIZE)
            .map(|at| f4e2m1::from_bits(codes[at / 2] >> (4 * (at % 2))).to_f32())
            .collect();
        let blocks: Vec<f8e4m3> = (words(O::SIZE * I::SIZE / 16, 2))
            .map(|word| f8e4m3::from_f32((8.0 + 440.0 * unit(word)) as f32))
            .collect();
        let global = 2f32.powi(14) * 1.37;
        let rows: Vec<bf16> = (words(Eight::SIZE * I::SIZE, 3))
            .map(|word| bf16::from_f32((4.0 * unit(word) - 2.0) as f32))
            .collect();
        let mut exact = vec![0.0; Eight::SIZE * O::SIZE];
        for output in 0..O::SIZE {
            let weights: Vec<f64> = (0..I::SIZE)
                .map(|k| {
                    let at = output * I::SIZE + k;
                    f64::from(decoded[at]) * f64::from(blocks[at / 16].to_f32()) / f64::from(global)
                })
                .collect();
            for (row, input) in rows.chunks_exact(I::SIZE).enumerate() {
                exact[row * O::SIZE + output] = (weights.iter().zip(input))
                    .map(|(weight, x)| weight * f64::from(x.to_f32()))
                    .sum();
            }
        }
        let weights: Vec<bf16> = decoded.iter().map(|&value| bf16::from_f32(value)).collect();
        let packet = [global, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let pdma = &mut device.pdma;
        Self {
            codes: (HostTensor::<f4e2m1, m![O, I]>::from_buf(codes)
                .to_hbm(pdma)
                .await)
                .unwrap(),
            weights: (HostTensor::<bf16, m![O, I]>::from_vec(weights)
                .to_hbm(pdma)
                .await)
                .unwrap(),
            blocks: (HostTensor::<f8e4m3, m![O, I / 16]>::from_vec(blocks)
                .to_hbm(pdma)
                .await)
                .unwrap(),
            global: (HostTensor::<f32, m![1 # 8]>::from_vec(packet)
                .to_hbm(pdma)
                .await)
                .unwrap(),
            one: (HostTensor::<bf16, m![One, I]>::from_vec(rows[..I::SIZE].to_vec())
                .to_hbm(pdma)
                .await)
                .unwrap(),
            eight: (HostTensor::<bf16, m![Eight, I]>::from_vec(rows)
                .to_hbm(pdma)
                .await)
                .unwrap(),
            exact,
        }
    }
}

/// How far `found` is from `exact`: the worst output's error over one BF16 rounding of it plus
/// `2^-9` of its row's RMS.
fn worst(found: &[bf16], exact: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for (found, exact) in found.chunks_exact(O::SIZE).zip(exact.chunks_exact(O::SIZE)) {
        let rms = (exact.iter().map(|e| e * e).sum::<f64>() / O::SIZE as f64).sqrt();
        for (got, want) in found.iter().zip(exact) {
            let got = f64::from(got.to_f32());
            worst = worst.max((got - want).abs() / (want.abs() / 256.0 + rms / 512.0));
        }
    }
    worst
}

/// Asserts that `form`'s sums are `want`'s, bit for bit.
fn same(form: &str, found: &[bf16], want: &[bf16]) {
    assert_eq!(found.len(), want.len(), "{form}");
    for (at, (got, want)) in found.iter().zip(want).enumerate() {
        assert_eq!(got.to_bits(), want.to_bits(), "{form}: output {at}");
    }
}

/// The feed-forward's pass sums every row within one rounding of the FP64 product, and every other
/// form gives its sums bit for bit.
#[tokio::test(flavor = "current_thread")]
async fn every_form_sums_as_the_pass() {
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(eight_rows.topology()).unwrap();
    let drawn = Drawn::new(&mut device).await;
    let eight = (&drawn.codes, &drawn.blocks, &drawn.global, &drawn.eight);
    let one = (&drawn.codes, &drawn.blocks, &drawn.global, &drawn.one);
    let decoded = (&drawn.weights, &drawn.blocks, &drawn.global, &drawn.eight);
    let pass = launch(eight_rows, (&mut device, eight)).await.unwrap();
    let pass = (pass
        .to_host::<m![Eight, O]>(&mut device.pdma)
        .await
        .unwrap())
    .into_vec();
    let found = worst(&pass, &drawn.exact);
    println!("eight rows: worst {found:.3} of the bound");
    assert!(
        found <= 1.0,
        "eight rows: an output is {found} of the bound away"
    );
    let sums = launch(short_passes, (&mut device, eight)).await.unwrap();
    let sums = (sums
        .to_host::<m![Eight, O]>(&mut device.pdma)
        .await
        .unwrap())
    .into_vec();
    same("short passes", &sums, &pass);
    let sums = launch(decoded_weights, (&mut device, decoded))
        .await
        .unwrap();
    let sums = (sums
        .to_host::<m![Eight, O]>(&mut device.pdma)
        .await
        .unwrap())
    .into_vec();
    same("decoded weights", &sums, &pass);
    let sums = launch(one_row, (&mut device, one)).await.unwrap();
    let sums = (sums.to_host::<m![One, O]>(&mut device.pdma).await.unwrap()).into_vec();
    same("one row", &sums, &pass[..O::SIZE]);
    let sums = launch(four_lanes, (&mut device, one)).await.unwrap();
    let sums = (sums.to_host::<m![One, O]>(&mut device.pdma).await.unwrap()).into_vec();
    same("four lanes", &sums, &pass[..O::SIZE]);
    let sums = launch(eight_rows_slotted, (&mut device, eight))
        .await
        .unwrap();
    let sums = (sums
        .to_host::<m![Eight, O]>(&mut device.pdma)
        .await
        .unwrap())
    .into_vec();
    same("eight rows slotted", &sums, &pass);
    let sums = launch(four_lanes_slotted, (&mut device, one))
        .await
        .unwrap();
    let sums = (sums.to_host::<m![One, O]>(&mut device.pdma).await.unwrap()).into_vec();
    same("four lanes slotted", &sums, &pass[..O::SIZE]);
}

/// The device spans profiled launches report: name, instruction, first and last cycle.
#[derive(Default)]
struct Spans(std::sync::Mutex<Vec<(String, String, u64, u64)>>);

impl tracing::Subscriber for Spans {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        #[derive(Default)]
        struct Fields(String, String, u64, u64);
        impl tracing::field::Visit for Fields {
            fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                match field.name() {
                    "begin_cycle" => self.2 = value,
                    "end_cycle" => self.3 = value,
                    _ => {}
                }
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                match field.name() {
                    "name" => self.0 = value.into(),
                    "instruction_index" => self.1 = value.into(),
                    _ => {}
                }
            }
            fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
        }
        if span.metadata().name() == "NPU" {
            let mut fields = Fields::default();
            span.record(&mut fields);
            let Fields(name, index, begin, end) = fields;
            self.0.lock().unwrap().push((name, index, begin, end));
        }
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

impl Spans {
    /// A form's `Task` cycles over its launches, and the commands of its last launch.
    fn report(&self, form: &str, tasks: &mut [u64]) {
        tasks.sort_unstable();
        println!(
            "form {form} task median {} min {} max {} cycles",
            tasks[tasks.len() / 2],
            tasks[0],
            tasks[tasks.len() - 1]
        );
        let mut launched = self.0.lock().unwrap().clone();
        launched.sort_by_key(|span| span.2);
        let start = launched.first().map_or(0, |span| span.2);
        for (name, index, begin, end) in launched.iter().filter(|span| span.0 != "Task") {
            println!(
                "span {form} {} {} {name} {index}",
                begin - start,
                end - start
            );
        }
    }

    /// The longest `Task` of the last launch.
    fn task(&self) -> u64 {
        let launched = self.0.lock().unwrap();
        let task = launched.iter().filter(|span| span.0 == "Task");
        task.map(|span| span.3 - span.2)
            .max()
            .expect("run with FURIOSA_OPT_PROFILE")
    }
}

/// Every form's device time over `PASS_PROFILE_STEPS` launches after a warm-up, at the level
/// `FURIOSA_OPT_PROFILE` asks for; at `trace` the commands of each form's last launch follow.
#[tokio::test(flavor = "current_thread")]
#[ignore = "profiles launches on a device"]
async fn pass_profile() {
    let steps: usize =
        std::env::var("PASS_PROFILE_STEPS").map_or(10, |steps| steps.parse().unwrap());
    let spans = std::sync::Arc::new(Spans::default());
    let _spans = tracing::subscriber::set_default(std::sync::Arc::clone(&spans));
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(eight_rows.topology()).unwrap();
    let drawn = Drawn::new(&mut device).await;
    let eight = (&drawn.codes, &drawn.blocks, &drawn.global, &drawn.eight);
    let one = (&drawn.codes, &drawn.blocks, &drawn.global, &drawn.one);
    let decoded = (&drawn.weights, &drawn.blocks, &drawn.global, &drawn.eight);
    macro_rules! profile {
        ($form:literal, $kernel:ident, $arguments:expr) => {{
            let mut tasks = vec![];
            for step in 0..=steps {
                spans.0.lock().unwrap().clear();
                launch($kernel, (&mut device, $arguments)).await.unwrap();
                device.synchronize().await.unwrap();
                if step > 0 {
                    tasks.push(spans.task());
                }
            }
            spans.report($form, &mut tasks);
        }};
    }
    profile!("eight-rows", eight_rows, eight);
    profile!("one-row", one_row, one);
    profile!("four-lanes", four_lanes, one);
    profile!("decoded-weights", decoded_weights, decoded);
    profile!("short-passes", short_passes, eight);
    profile!("eight-rows-slotted", eight_rows_slotted, eight);
    profile!("four-lanes-slotted", four_lanes_slotted, one);
    profile!("eight-rows-unscaled", eight_rows_unscaled, eight);
}
