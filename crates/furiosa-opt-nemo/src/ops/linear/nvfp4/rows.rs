//! The rows of a launch through one NVFP4 projection: each weight tile restores once for all the
//! live rows, and the two clusters own the two sums of the output.
//!
//! - A row splits into input parts across slices, the fewest that let eight rows' parts fill the
//!   tensor registers as the lanes of one contraction; a slice holds one part of every row.
//! - Each slice owns consecutive output rows over its part, restored in tiles of up to twelve
//!   rows, each tile in one pass whose block scales fit the vector registers.
//! - A batch of eight rows quantizes as one row does, each row under its own block scales.
//! - A slice restores each weight tile once and contracts it with every live batch; the parts'
//!   FP32 sums add before the output scale, once, as for one row.

use core::marker::PhantomData;

use super::{BlockScaled, E2M1, E4M3, Group, ROUND};
use crate::ops::layout::{Live, axes::Tile};
use crate::{Module, ops::layout::axes::Chip};
use furiosa_opt_std::prelude::BitReq::{Ignore, One};
use furiosa_opt_std::prelude::*;

/// A launch's rows, each one contiguous feature vector.
type Rows<R, W> = HbmTensor<bf16, Chip, m![R, W]>;

/// The rows one contraction carries as its eight lanes.
type Batch<R> = Tile<R, 8>;

/// Inputs of a row one slice holds: the largest part, at most half a row, whose eight rows fill
/// at most the 64 KiB of tensor registers.
#[derive(Clone, Debug)]
struct Chunk<I>(PhantomData<I>);

impl<I: AxisName> AxisName for Chunk<I> {
    const NAME: Ident = Ident::new("Chunk");
    const SIZE: usize = {
        let mut parts = 2;
        while 8 * 2 * (I::SIZE / parts) > 65_536 {
            parts *= 2;
        }
        I::SIZE / parts
    };
}

/// Output rows one slice owns: its cluster's half of the output over the cluster's owners.
#[derive(Clone, Debug)]
struct Owned<I, O>(PhantomData<(I, O)>);

impl<I: AxisName, O: AxisName> AxisName for Owned<I, O> {
    const NAME: Ident = Ident::new("Owned");
    const SIZE: usize = O::SIZE / 2 / (256 / (I::SIZE / Chunk::<I>::SIZE));
}

/// Output rows a slice restores in one pass: the most, up to twelve, dividing its rows whose block
/// scales fill at most the 8 KiB of vector registers.
#[derive(Clone, Debug)]
struct Height<I, O>(PhantomData<(I, O)>);

impl<I: AxisName, O: AxisName> AxisName for Height<I, O> {
    const NAME: Ident = Ident::new("Height");
    const SIZE: usize = {
        let mut rows = 12;
        while !Owned::<I, O>::SIZE.is_multiple_of(rows) || rows * Chunk::<I>::SIZE / 16 * 4 > 8192 {
            rows -= 1;
        }
        rows
    };
}

type Width<I> = Symbol<Chunk<I>>;
type Block<I, O> = Symbol<Owned<I, O>>;
type Parts<I> = m![I / ({ Width<I> })];
type Part<I> = m![I % ({ Width<I> })];
type Blocks<I> = m![I % ({ Width<I> }) / 16];
type Lanes<R> = m![R % ({ Symbol<Batch<R>> })];
type Index<R> = m![R / ({ Symbol<Batch<R>> })];

/// One output half a cluster, and a cluster's output owners, each over every input part; every
/// owner holds a copy of each row's part.
type Clusters<O> = m![O / (O / 2)];
type Owners<I, O> = m![O % (O / 2) / ({ Block<I, O> }), { Parts<I> }];
type Reduced<I, O> = m![O % (O / 2) / ({ Block<I, O> }), 1 # ({ Parts<I> })];

type Tiles<I, O> = m![O % ({ Block<I, O> }) / ({ Symbol<Height<I, O>> })];
type Elements<I, O> = m![O % ({ Symbol<Height<I, O>> })];
type Restored<I, O, D = bf16> =
    DmTensor<D, Chip, Clusters<O>, Owners<I, O>, m![{ Elements<I, O> }, { Part<I> }]>;
type Operand<I, O, R> = TrfTensor<bf16, Chip, Clusters<O>, Owners<I, O>, Lanes<R>, Part<I>>;
type Sums<I, O, R> = DmTensor<f32, Chip, Clusters<O>, Owners<I, O>, m![O % ({ Block<I, O> }), R]>;

type Staged<I, O, R> = DmTensor<bf16, Chip, Clusters<O>, Owners<I, O>, m![R, { Blocks<I> }, Group]>;
type Selected<R, I> = m![{ Index<R> } = 1 # ({ Index<R> }), { Lanes<R> }, { Blocks<I> }, Group];
type SelectedMut<R, I> =
    m![{ Index<R> } = 1 #{!} ({ Index<R> }), { Lanes<R> }, { Blocks<I> }, Group];
type Grouped<I, O, R, D> =
    DmTensor<D, Chip, Clusters<O>, Owners<I, O>, m![{ Lanes<R> }, { Blocks<I> }, Group]>;
type PerBlock<I, O, R, D> =
    DmTensor<D, Chip, Clusters<O>, Owners<I, O>, m![{ Lanes<R> }, { Blocks<I> }]>;
type Register<I, O, R> =
    VrfTensor<f32, Chip, Clusters<O>, Owners<I, O>, m![{ Lanes<R> }, { Blocks<I> }]>;
type Scalar<I, O> = VrfTensor<f32, Chip, Clusters<O>, Owners<I, O>, m![1 # 8]>;

/// One weight tile as stored: a tile's output rows of E2M1 codes over a slice's input part, with
/// their E4M3 block scales.
type Coded<I, O> = (
    Restored<I, O, f4e2m1>,
    DmTensor<f8e4m3, Chip, Clusters<O>, Owners<I, O>, m![{ Elements<I, O> }, { Blocks<I> }]>,
);

/// The live rows of a launch through one projection: every weight tile restores once for all of
/// them, and a row past them is padding nothing reads.
impl<I: AxisName, O: AxisName, R: AxisName> Module<(&mut Device, (Live<R>, &Rows<R, I>))>
    for BlockScaled<I, O>
{
    type Output = Rows<R, O>;

    fn forward(
        &self,
        (device, (live, input)): (&mut Device, (Live<R>, &Rows<R, I>)),
    ) -> Self::Output {
        // `value` in every slice's vector registers: landed in one slice a cluster, then copied by
        // the switch ring, far cheaper than a DMA into each slice.
        let broadcast = |device: &mut Device,
                         value: &super::Scale|
         -> VrfTensor<f32, Chip, m![2], m![256], m![1 # 8]> {
            let landed: DmTensor<f32, Chip, m![2], m![1 # 256], m![1 # 8]> =
                value.to_dm(&mut device.tdma);
            device
                .sub
                .begin(landed.view())
                .fetch::<m![1], m![1 # 8]>()
                .switch::<m![256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
                .collect::<m![1], m![1 # 8]>()
                .to_vrf()
        };
        // Replaces batch `batch` of `rows` with each element's E2M1 code times its block's E4M3
        // scale, every row under its own block scales, as the single-row quantizer does.
        let quantize = |device: &mut Device,
                        (global, restore): &(Scalar<I, O>, Scalar<I, O>),
                        rows: &mut Staged<I, O, R>,
                        batch: usize| {
            let scale: PerBlock<I, O, R, f8e4m3> = device
                .main
                .begin(rows.view().tile::<Index<R>, 1, Selected<R, I>>(batch))
                .fetch::<m![{ Lanes<R> }, { Blocks<I> }], m![Group]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Lanes<R> }, { Blocks<I> }, Group / 8], m![Group % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_reinterpret::<i32>()
                .vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX)
                .vector_reinterpret::<f32>()
                .vector_narrow_split::<m![{ Lanes<R> }, { Blocks<I> }, Group / 4], m![Group % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), const { 1.0f32 / E2M1 })
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), global)
                .vector_intra_slice_reduce::<Group, m![{ Lanes<R> }, { Blocks<I> }], m![1 # 4]>(
                    IntraSliceReduceOpF32::Max,
                )
                .vector_widen_pad::<m![1 # 8]>()
                .vector_clip(ClipBinaryOpF32::Min, E4M3)
                .vector_final()
                .cast::<f8e4m3, m![1 # 32]>()
                .transpose::<m![{ Lanes<R> }, { Blocks<I> } / 8], m![{ Blocks<I> } % 8 # 32]>()
                .commit_trim::<m![{ Blocks<I> } % 8]>()
                .commit();
            let reciprocal: PerBlock<I, O, R, f32> = device
                .main
                .begin(scale.view())
                .fetch::<m![{ Lanes<R> }, { Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Lanes<R> }, { Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Lanes<R> }, { Blocks<I> } / 4], m![{ Blocks<I> } % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), restore)
                .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
                .vector_widen_concat::<m![{ Lanes<R> }, { Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
                .vector_clip(ClipBinaryOpF32::Min, f32::MAX)
                .vector_final()
                .commit_trim::<m![{ Blocks<I> } % 8]>()
                .commit();
            let reciprocal: Register<I, O, R> = device
                .sub
                .begin(reciprocal.view())
                .fetch::<m![{ Lanes<R> }, { Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
                .collect::<m![{ Lanes<R> }, { Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
                .to_vrf();
            let scaled: Grouped<I, O, R, f32> = device
                .main
                .begin(rows.view().tile::<Index<R>, 1, Selected<R, I>>(batch))
                .fetch::<m![{ Lanes<R> }, { Blocks<I> }], m![Group]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Lanes<R> }, { Blocks<I> }, Group / 8], m![Group % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Lanes<R> }, { Blocks<I> }, Group / 4], m![Group % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &reciprocal)
                .vector_widen_concat::<m![{ Lanes<R> }, { Blocks<I> }, Group / 8], m![Group % 8]>()
                .vector_clip(ClipBinaryOpF32::Max, -E2M1)
                .vector_clip(ClipBinaryOpF32::Min, E2M1)
                .vector_final()
                .commit_trim::<m![Group % 8]>()
                .commit();
            let below = TagGuard::matches([One, One, Ignore, Ignore]);
            let within = TagGuard::matches([Ignore, Ignore, One, One]);
            let rounded: Grouped<I, O, R, f32> = device
                .main
                .begin(scaled.view())
                .fetch::<m![{ Lanes<R> }, { Blocks<I> }, Group / 8], m![Group % 8]>()
                .collect::<m![{ Lanes<R> }, { Blocks<I> }, Group / 8], m![Group % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Comparison([
                    Cmp::Less(2.0f32),
                    Cmp::Greater(-2.0f32),
                    Cmp::Less(4.0f32),
                    Cmp::Greater(-4.0f32),
                ]))
                .vector_narrow_split::<m![{ Lanes<R> }, { Blocks<I> }, Group / 4], m![Group % 4]>()
                .vector_fp_binary(
                    FpBinaryOp::AddF,
                    Branched::imm(below, ROUND[0])
                        .imm(within, ROUND[1])
                        .imm(TagGuard::all(), ROUND[2]),
                )
                .vector_widen_concat::<m![{ Lanes<R> }, { Blocks<I> }, Group / 8], m![Group % 8]>()
                .vector_clip(
                    ClipBinaryOpF32::Add,
                    Branched::imm(below, -ROUND[0])
                        .imm(within, -ROUND[1])
                        .imm(TagGuard::all(), -ROUND[2]),
                )
                .vector_final()
                .commit_trim::<m![Group % 8]>()
                .commit();
            let scale: Register<I, O, R> = device
                .sub
                .begin(scale.view())
                .fetch::<m![{ Lanes<R> }, { Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Lanes<R> }, { Blocks<I> } / 8], m![{ Blocks<I> } % 8]>()
                .to_vrf();
            device
                .main
                .begin(rounded.view())
                .fetch::<m![{ Lanes<R> }, { Blocks<I> }, Group / 8], m![Group % 8]>()
                .collect::<m![{ Lanes<R> }, { Blocks<I> }, Group / 8], m![Group % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Lanes<R> }, { Blocks<I> }, Group / 4], m![Group % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
                .vector_widen_concat::<m![{ Lanes<R> }, { Blocks<I> }, Group / 8], m![Group % 8]>()
                .vector_final()
                .cast::<bf16, m![Group % 8 # 16]>()
                .commit_trim::<m![Group % 8]>()
                .commit_view(
                    rows.view_mut()
                        .tile::<Index<R>, 1, SelectedMut<R, I>>(batch),
                );
        };
        // Batch `batch` of the quantized rows as the lanes of a contraction.
        let fill = |device: &mut Device,
                    rows: &DmTensor<bf16, Chip, Clusters<O>, Owners<I, O>, m![R, { Part<I> }]>,
                    batch: usize|
         -> Operand<I, O, R> {
            device
                .sub
                .begin(rows.view().tile::<Index<R>, 1, m![
                    { Index<R> } = 1 # ({ Index<R> }), { Lanes<R> }, { Part<I> }
                ]>(batch))
                .fetch::<m![{ Lanes<R> }, { Part<I> } / 16], m![I % 16]>()
                .collect::<m![{ Lanes<R> }, { Part<I> } / 16], m![I % 16]>()
                .to_trf()
        };
        // Each slice's FP32 sums of one weight tile and one batch of lanes over its input part,
        // into the cells of `sums` the tile and batch own.
        let contract = |device: &mut Device,
                        tile: &Restored<I, O>,
                        lanes: &Operand<I, O, R>,
                        sums: &mut Sums<I, O, R>,
                        (index, batch): (usize, usize)| {
            device
                .main
                .begin(tile.view())
                .fetch::<m![{ Elements<I, O> }, { Part<I> } / 16], m![I % 16]>()
                .collect::<m![{ Elements<I, O> }, { Part<I> } / 16], m![I % 16]>()
                .contract_outer::<m![{ Elements<I, O> }, { Part<I> } / 32], m![I % 32], _, _, _>(
                    lanes,
                )
                .contract_packet::<m![1]>()
                .contract_time::<Elements<I, O>>()
                .contract_lane::<Elements<I, O>, m![{ Lanes<R> } # 8]>(LaneMode::Interleaved)
                .commit_trim::<m![{ Lanes<R> }]>()
                .commit_view(
                    sums.view_mut()
                        .tile::<Tiles<I, O>, 1, m![{ Elements<I, O> } #{!} ({ Block<I, O> }), R]>(
                            index,
                        )
                        .tile::<Index<R>, 1, m![
                            { Elements<I, O> } #{!} ({ Block<I, O> }),
                            { Index<R> } = 1 #{!} ({ Index<R> }),
                            { Lanes<R> }
                        ]>(batch),
                );
        };
        // Weight tile `tile` of each slice's block over its input part.
        let load = |device: &mut Device, tile: usize| -> Coded<I, O> {
            (
                self.weight.view().tile::<Tiles<I, O>, 1, m![
                    O / (O / 2), O % (O / 2) / ({ Block<I, O> }), { Elements<I, O> } # ({ Block<I, O> }), I
                ]>(tile).to_dm(&mut device.tdma),
                self.weight_scale.view().tile::<Tiles<I, O>, 1, m![
                    O / (O / 2), O % (O / 2) / ({ Block<I, O> }), { Elements<I, O> } # ({ Block<I, O> }), I / 16
                ]>(tile).to_dm(&mut device.tdma),
            )
        };
        // A loaded tile as BF16 code times block scale: the E2M1 table decode, its FP32 widening
        // and the block scales in one pass.
        let restore = |device: &mut Device, (packed, scale): Coded<I, O>| -> Restored<I, O> {
            let scale: VrfTensor<
                f32,
                Chip,
                Clusters<O>,
                Owners<I, O>,
                m![{ Elements<I, O> }, { Blocks<I> }],
            > = device
                .sub
                .begin(scale.view())
                .fetch::<Elements<I, O>, Blocks<I>>()
                .fetch_cast::<f32>()
                .collect::<m![{ Elements<I, O> }, { Blocks<I> } / 8], m![I / 16 % 8]>()
                .to_vrf();
            device
                .main
                .begin(packed.view())
                .fetch::<m![{ Elements<I, O> }, { Part<I> } / 32], m![I % 32]>()
                .fetch_table_lookup::<f8e4m3>()
                .fetch_cast::<f32>()
                .collect::<m![{ Elements<I, O> }, { Part<I> } / 8], m![I % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Elements<I, O> }, { Part<I> } / 4], m![I % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
                .vector_widen_concat::<m![{ Elements<I, O> }, { Part<I> } / 8], m![I % 8]>()
                .vector_final()
                .cast::<bf16, m![I % 8 # 16]>()
                .commit_trim::<m![I % 8]>()
                .commit()
        };
        const {
            assert!(
                I::SIZE > 0
                    && O::SIZE > 0
                    && I::SIZE.is_multiple_of(256)
                    && O::SIZE.is_multiple_of(256)
                    && R::SIZE.is_multiple_of(8)
                    && Chunk::<I>::SIZE.is_multiple_of(128)
                    && Owned::<I, O>::SIZE.is_multiple_of(4)
                    && Height::<I, O>::SIZE * Chunk::<I>::SIZE / 16 * 4 <= 8192
            )
        };
        let batches = live.retile::<Batch<R>>().count();
        // SAFETY: every slice of both clusters holds the same broadcast value, so any partition
        // names it.
        let scales: (Scalar<I, O>, Scalar<I, O>) = unsafe {
            (
                broadcast(device, &self.scales.global).reshape(),
                broadcast(device, &self.scales.restore).reshape(),
            )
        };
        let staged: DmTensor<bf16, Chip, Clusters<O>, Owners<I, O>, m![R, { Part<I> }]> =
            input.to_dm(&mut device.tdma);
        // SAFETY: relabels each slice's input part as its 16-input blocks.
        let mut staged: Staged<I, O, R> = unsafe { staged.reshape() };
        if const { <Index<R> as M>::SIZE == 1 } {
            quantize(device, &scales, &mut staged, 0);
        } else {
            for batch in 0..batches {
                quantize(device, &scales, &mut staged, batch);
            }
        }
        // SAFETY: the inverse of the relabel above.
        let quantized: DmTensor<bf16, Chip, Clusters<O>, Owners<I, O>, m![R, { Part<I> }]> =
            unsafe { staged.reshape() };
        let mut sums: Sums<I, O, R> = DmTensor::new();
        let tiles = <Tiles<I, O> as M>::SIZE;
        // Each tile's load issues before the previous tile restores, so it lands meanwhile.
        let mut next = load(device, 0);
        if const { <Index<R> as M>::SIZE == 1 } {
            let lanes = fill(device, &quantized, 0);
            #[unroll]
            for tile in 0..<Tiles<I, O> as M>::SIZE - 1 {
                let ahead = load(device, tile + 1);
                let current = next;
                next = ahead;
                let restored = restore(device, current);
                contract(device, &restored, &lanes, &mut sums, (tile, 0));
            }
            let restored = restore(device, next);
            contract(device, &restored, &lanes, &mut sums, (tiles - 1, 0));
        } else {
            #[unroll]
            for tile in 0..<Tiles<I, O> as M>::SIZE - 1 {
                let ahead = load(device, tile + 1);
                let current = next;
                next = ahead;
                let restored = restore(device, current);
                for batch in 0..batches {
                    let lanes = fill(device, &quantized, batch);
                    contract(device, &restored, &lanes, &mut sums, (tile, batch));
                }
            }
            let restored = restore(device, next);
            for batch in 0..batches {
                let lanes = fill(device, &quantized, batch);
                contract(device, &restored, &lanes, &mut sums, (tiles - 1, batch));
            }
        }
        // SAFETY: as for the quantizer's scales.
        let scale: VrfTensor<f32, Chip, Clusters<O>, Reduced<I, O>, m![1 # 8]> =
            unsafe { broadcast(device, &self.scales.output).reshape() };
        let output: DmTensor<bf16, Chip, Clusters<O>, Reduced<I, O>, m![R, O % ({ Block<I, O> })]> =
            device
                .main
                .begin(sums.view())
                .fetch::<m![R / 8, O % ({ Block<I, O> })], m![R % 8]>()
                .collect::<m![R / 8, O % ({ Block<I, O> })], m![R % 8]>()
                .vector_init()
                .vector_inter_slice_reduce::<Reduced<I, O>, m![R / 8, O % ({ Block<I, O> })]>(
                    InterSliceReduceOpF32::Add,
                )
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![R / 8, O % ({ Block<I, O> }), R % 8 / 4], m![R % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
                .vector_widen_concat::<m![R / 8, O % ({ Block<I, O> })], m![R % 8]>()
                .vector_final()
                .cast::<bf16, m![R % 8 # 16]>()
                .transpose::<m![R / 8, O % ({ Block<I, O> }) / 4, R % 8], m![O % 4 # 16]>()
                .commit_trim::<m![O % 4]>()
                .commit();
        output.to_hbm(&mut device.tdma)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{linear, noise, quantized};
    use super::*;
    use crate::compose::{Count, Tiles, tests::count};
    use crate::ops::layout::axes::{Eight, Sixteen};
    use furiosa_opt_std::runtime::launch;

    // Two input parts and two tiles of twelve rows a slice; four parts and two tiles of six.
    axes![Input = 512, Output = 6144, Wide = 16384, Narrow = 1536];

    #[device(chip = 1)]
    fn projected<I: AxisName, O: AxisName, R: AxisName>(
        device: &mut Device,
        rows: &Count,
        linear: &BlockScaled<I, O>,
        input: &Rows<R, I>,
    ) -> Rows<R, O> {
        linear.forward((&mut *device, (Tiles::read(rows), input)))
    }

    /// The live rows' outputs, rows past `live` being NaN on input.
    async fn run<I: AxisName, O: AxisName, R: AxisName>(
        device: &mut Device,
        linear: &BlockScaled<I, O>,
        input: &[bf16],
        live: usize,
    ) -> Vec<bf16> {
        let rows: Vec<bf16> = input
            .iter()
            .enumerate()
            .map(|(at, value)| {
                if at / I::SIZE < live {
                    *value
                } else {
                    bf16::from_f32(f32::NAN)
                }
            })
            .collect();
        let rows: Rows<R, I> = HostTensor::<bf16, m![R, I]>::from_vec(rows)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let count: Count = count(live).to_hbm(&mut device.pdma).await.unwrap();
        let mut output = launch(projected, (&mut *device, &count, linear, &rows))
            .await
            .unwrap()
            .to_host::<m![R, O]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        output.truncate(live * O::SIZE);
        output
    }

    /// Each live row quantizes as the reference quantizes it alone, whatever rows pad the launch.
    #[tokio::test(flavor = "current_thread")]
    async fn quantizes_rows_alone() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        // Weight row `o` selects input `o mod In`, so each output is one quantized input.
        let mut codes = vec![f4e2m1::from_bits(0); Output::SIZE * Input::SIZE];
        for row in 0..Output::SIZE {
            codes[row * Input::SIZE + row % Input::SIZE] = f4e2m1::from_bits(0b0010);
        }
        let scales = vec![f8e4m3::from_f32(1.0); Output::SIZE * Input::SIZE / 16];
        let ties = [
            6.0, 0.5, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75,
        ];
        for input_scale in [0.03125, 0.3258928656578064] {
            let linear: BlockScaled<Input, Output> =
                linear(&mut device, &codes, &scales, input_scale, 1.0).await;
            let input: Vec<bf16> = noise(Eight::SIZE * Input::SIZE, 0x2545_f491_4f6c_dd1d)
                .enumerate()
                .map(|(at, value)| {
                    let (row, block, member) = (at / Input::SIZE, at % Input::SIZE / 16, at % 16);
                    // Each row puts zero, zero-scale, subnormal-scale, E2M1-tie and saturating
                    // blocks at its own offset and its own magnitude.
                    bf16::from_f32(match (block + row) % 8 {
                        0 => 0.0,
                        1 => value * 1e-6,
                        2 => value * 1e-3,
                        3 => 4.0 * input_scale * ties.get(member).copied().unwrap_or(-2.5),
                        4 => value * 1e4,
                        _ => value * 100.0 * (1 + row) as f32,
                    })
                })
                .collect();
            for live in [8, 3, 1] {
                let landed = run::<Input, Output, Eight>(&mut device, &linear, &input, live).await;
                for row in 0..live {
                    let reference =
                        quantized(&input[row * Input::SIZE..][..Input::SIZE], input_scale);
                    let expected: Vec<u16> = (0..Output::SIZE)
                        .map(|o| bf16::from_f32(input_scale * reference[o % Input::SIZE]))
                        .map(|value| {
                            if value.to_f32() == 0.0 {
                                0
                            } else {
                                value.to_bits()
                            }
                        })
                        .collect();
                    let landed: Vec<u16> = landed[row * Output::SIZE..][..Output::SIZE]
                        .iter()
                        .map(|value| {
                            if value.to_f32() == 0.0 {
                                0
                            } else {
                                value.to_bits()
                            }
                        })
                        .collect();
                    assert_eq!(
                        landed, expected,
                        "input scale {input_scale}, {live} live, row {row}"
                    );
                }
            }
        }
    }

    /// Random codes, block scales and rows land each live row's exact sum of code-times-scale
    /// products times both tensor scales, within one BF16 rounding and the FP32 accumulation's error.
    async fn sums_exactly<I: AxisName, O: AxisName, R: AxisName>(
        device: &mut Device,
        lives: &[usize],
    ) {
        let mut seeds = noise(
            2 * O::SIZE * I::SIZE / 16 + R::SIZE * I::SIZE,
            0x9e37_79b9_7f4a_7c15,
        );
        let codes: Vec<f4e2m1> = (0..O::SIZE * I::SIZE)
            .map(|index| f4e2m1::from_bits((index * 7 + index / 13) as u8 & 15))
            .collect();
        let scales: Vec<f8e4m3> = (0..O::SIZE * I::SIZE / 16)
            .map(|_| f8e4m3::from_f32(2f32.powf(6.0 * seeds.next().unwrap() + 2.0)))
            .collect();
        let (input_scale, weight_scale_2) = (0.0424107164144516, 0.00017293295240961015);
        let input: Vec<bf16> = seeds
            .by_ref()
            .take(R::SIZE * I::SIZE)
            .enumerate()
            .map(|(at, value)| bf16::from_f32(value * 100.0 / (1 + at / I::SIZE) as f32))
            .collect();
        let linear: BlockScaled<I, O> =
            linear(device, &codes, &scales, input_scale, weight_scale_2).await;
        let output = f64::from(input_scale) * f64::from(weight_scale_2);
        for &live in lives {
            let landed = run::<I, O, R>(device, &linear, &input, live).await;
            for row in 0..live {
                let operand = quantized(&input[row * I::SIZE..][..I::SIZE], input_scale);
                for o in 0..O::SIZE {
                    let (sum, magnitude) = (0..I::SIZE)
                        .map(|k| {
                            let weight = codes[o * I::SIZE + k].to_f32()
                                * scales[(o * I::SIZE + k) / 16].to_f32();
                            weight as f64 * operand[k] as f64
                        })
                        .fold((0.0, 0.0), |(sum, magnitude), term: f64| {
                            (sum + term, magnitude + term.abs())
                        });
                    let exact = sum * output;
                    let bound = exact.abs() / 128.0 + magnitude * output / f64::from(1 << 14);
                    let value = landed[row * O::SIZE + o].to_f32() as f64;
                    assert!(
                        (value - exact).abs() <= bound,
                        "{}x{}, {live} of {} live, row {row}, output {o}: {value} against {exact}",
                        I::SIZE,
                        O::SIZE,
                        R::SIZE
                    );
                }
            }
        }
    }

    /// Each live row lands the exact projection of its quantized inputs within rounding, at every
    /// envelope and live count.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_the_reference() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        sums_exactly::<Input, Output, Eight>(&mut device, &[8, 3, 1]).await;
        sums_exactly::<Input, Output, Sixteen>(&mut device, &[16, 9]).await;
        sums_exactly::<Wide, Narrow, Eight>(&mut device, &[8]).await;
    }
}
