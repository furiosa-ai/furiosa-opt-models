//! The rows of a launch through one NVFP4 projection: each weight tile restores once for all the
//! live rows, and the two clusters own the two sums of the output.
//!
//! - A row splits into input parts across slices, the fewest that let eight rows' parts fill the
//!   tensor registers as the lanes of one contraction; a slice holds one part of every row.
//! - Each slice owns consecutive output rows over its part, restored in tiles of up to twelve
//!   rows, each tile in one pass whose block scales fit the vector registers.
//! - W4A4 inputs quantize a batch of eight rows as one row, each row under its own block
//!   scales; BF16 inputs enter as they are.
//! - A slice restores each weight tile once and contracts it with every live batch; the parts'
//!   FP32 sums add before the output scale, once, as for one row.

use core::marker::PhantomData;

use crate::ops::linear::decode::Coded;
use crate::ops::linear::kernel::{Contraction, Rounded};
use crate::ops::linear::quantize::{Group, Quantization};
use crate::ops::normalization::Spread;
use crate::specialize::{VRF, contraction};
use crate::tensor::form::BlockScaled;
use crate::tensor::form::nvfp4::{Held, Late, Scale, Scales, W4A4, W4A16};
use crate::tensor::memory::Memory;
use crate::tensor::{axis::Tile, live::Live};
use crate::{Op, tensor::parallel::Chip};
use core::borrow::Borrow;
use furiosa_opt_std::prelude::*;

/// A launch's rows, each one contiguous feature vector.
type Rows<R, W> = HbmTensor<bf16, Chip, m![R, W]>;

/// The rows one contraction carries as its eight lanes.
type Batch<R> = Tile<R, 8>;

/// Inputs of a row one slice holds: the largest part, at most half a row, whose eight rows fill
/// at most the slice's tensor registers.
#[derive(Clone, Debug)]
struct Chunk<I>(PhantomData<I>);

impl<I: AxisName> AxisName for Chunk<I> {
    const NAME: Ident = Ident::new("Chunk");
    const SIZE: usize = I::SIZE / contraction::parts(I::SIZE);
}

/// Output rows one slice owns: its cluster's half of the output over the cluster's owners.
#[derive(Clone, Debug)]
struct Owned<I, O>(PhantomData<(I, O)>);

impl<I: AxisName, O: AxisName> AxisName for Owned<I, O> {
    const NAME: Ident = Ident::new("Owned");
    const SIZE: usize = O::SIZE / 2 / (256 / (I::SIZE / Chunk::<I>::SIZE));
}

/// Output rows a slice restores in one pass: the most, up to twelve, dividing its rows whose block
/// scales fill at most the slice's vector registers.
#[derive(Clone, Debug)]
struct Height<I, O>(PhantomData<(I, O)>);

impl<I: AxisName, O: AxisName> AxisName for Height<I, O> {
    const NAME: Ident = Ident::new("Height");
    const SIZE: usize = contraction::height(Owned::<I, O>::SIZE, Chunk::<I>::SIZE);
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
type Restored<I, O> =
    DmTensor<bf16, Chip, Clusters<O>, Owners<I, O>, m![{ Elements<I, O> }, { Part<I> }]>;
type Operand<I, O, R> =
    TrfTensor<bf16, Chip, Clusters<O>, Owners<I, O>, Lanes<R>, m![1, { Part<I> }]>;
type Sums<I, O, R> = DmTensor<f32, Chip, Clusters<O>, Owners<I, O>, m![O % ({ Block<I, O> }), R]>;

type Staged<I, O, R> = DmTensor<bf16, Chip, Clusters<O>, Owners<I, O>, m![R, { Blocks<I> }, Group]>;
type Selected<R, I> = m![{ Index<R> } = 1 # ({ Index<R> }), { Lanes<R> }, { Blocks<I> }, Group];
type SelectedMut<R, I> =
    m![{ Index<R> } = 1 #{!} ({ Index<R> }), { Lanes<R> }, { Blocks<I> }, Group];
/// The NVFP4 quantization of a batch's rows, each under its own block scales.
type Batched<I, R> = Quantization<
    m![{ Lanes<R> }, { Blocks<I> }],
    m![{ Lanes<R> }, { Blocks<I> }, Group],
    m![{ Lanes<R> }, { Blocks<I> }, Group / 8],
    m![{ Lanes<R> }, { Blocks<I> }, Group / 4],
    m![{ Lanes<R> }, { Blocks<I> } / 8],
    m![{ Lanes<R> }, { Blocks<I> } / 4],
    Blocks<I>,
>;
type Scalar<I, O> = VrfTensor<f32, Chip, Clusters<O>, Owners<I, O>, m![1 # 8]>;

/// BF16 rows enter the contraction as they are.
impl<I: AxisName, O: AxisName, R: AxisName> Op<(Live<(), R>, Staged<I, O, R>)> for W4A16 {
    type Output = Staged<I, O, R>;

    fn forward(&self, _: &mut Device, (_, rows): (Live<(), R>, Staged<I, O, R>)) -> Self::Output {
        rows
    }
}

/// The live rows with each element replaced by its E2M1 code times its block's E4M3 scale, every
/// row under its own block scales, as the single-row quantizer does.
impl<I: AxisName, O: AxisName, R: AxisName, Y: Memory> Op<(Live<(), R>, Staged<I, O, R>)>
    for W4A4<Y>
where
    Held<Y>: Borrow<Scale>,
{
    type Output = Staged<I, O, R>;

    fn forward(
        &self,
        device: &mut Device,
        (live, mut rows): (Live<(), R>, Staged<I, O, R>),
    ) -> Self::Output {
        // SAFETY: every slice of both clusters holds the same broadcast value, so any partition
        // names it.
        let (global, restore): (Scalar<I, O>, Scalar<I, O>) = unsafe {
            (
                broadcast(device, self.global.borrow()).reshape(),
                broadcast(device, self.restore.borrow()).reshape(),
            )
        };
        let mut quantize = |device: &mut Device, batch: usize| {
            let (codes, scale) = Batched::<I, R>::codes(
                device,
                || rows.view().tile::<Index<R>, 1, Selected<R, I>>(batch),
                (&global, &restore),
            );
            Batched::<I, R>::restored(&mut device.main, codes.view(), &scale).commit_view(
                rows.view_mut()
                    .tile::<Index<R>, 1, SelectedMut<R, I>>(batch),
            );
        };
        if const { <Index<R> as M>::SIZE == 1 } {
            quantize(device, 0);
        } else {
            for batch in 0..live.count::<Batch<R>>() {
                quantize(device, batch);
            }
        }
        rows
    }
}

/// The live rows of a launch in HBM through one projection, into HBM.
impl<I: AxisName, O: AxisName, R: AxisName, A, K, V, Y: Memory> Op<(Live<(), R>, &Rows<R, I>)>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Late<A, Y>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, m![O, I]>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Scales<I, O>>>,
    Held<Y>: Borrow<Scale>,
    A: Op<(Live<(), R>, Staged<I, O, R>), Output = Staged<I, O, R>>,
{
    type Output = Rows<R, O>;

    fn forward(
        &self,
        device: &mut Device,
        (live, input): (Live<(), R>, &Rows<R, I>),
    ) -> Self::Output {
        let staged = input.to_dm(&mut device.tdma);
        self.project(device, live, staged).to_hbm(&mut device.tdma)
    }
}

/// The live rows of a launch on chip through one projection, into the feature slices the next
/// projection reads; the projected rows keep the launch's live rows.
impl<'x, I: AxisName, O: AxisName, R: AxisName, A, C: M, S: M, F: M, K, V, Y: Memory>
    Op<&'x Live<DmTensor<bf16, Chip, C, S, m![R, { F }]>, R>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Late<A, Y>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, m![O, I]>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Scales<I, O>>>,
    Held<Y>: Borrow<Scale>,
    A: Op<(Live<(), R>, Staged<I, O, R>), Output = Staged<I, O, R>>,
{
    type Output = Live<Spread<O, R>, R>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<DmTensor<bf16, Chip, C, S, m![R, { F }]>, R>,
    ) -> Self::Output {
        let staged = input.value().to_dm(&mut device.tdma);
        self.spread(device, input.with(()), staged)
    }
}

/// The live rows of a launch read in place from any on-chip placement through one projection,
/// into the feature slices the next projection reads.
impl<'v, I: AxisName, O: AxisName, R: AxisName, A, C: M, S: M, E: M, K, V, Y: Memory>
    Op<(
        Live<PhantomData<R>, R>,
        DmTensorView<'v, bf16, Chip, C, S, E>,
    )> for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Late<A, Y>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, m![O, I]>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Scales<I, O>>>,
    Held<Y>: Borrow<Scale>,
    A: Op<(Live<(), R>, Staged<I, O, R>), Output = Staged<I, O, R>>,
{
    type Output = Live<Spread<O, R>, R>;

    fn forward(
        &self,
        device: &mut Device,
        (live, input): (
            Live<PhantomData<R>, R>,
            DmTensorView<'v, bf16, Chip, C, S, E>,
        ),
    ) -> Self::Output {
        let mut staged = DmTensor::new();
        input.to_dm_view(&mut device.tdma, staged.view_mut());
        self.spread(device, live.with(()), staged)
    }
}

impl<I: AxisName, O: AxisName, A, K, V, Y: Memory>
    BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Late<A, Y>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, m![O, I]>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Scales<I, O>>>,
    Held<Y>: Borrow<Scale>,
{
    /// The live rows of `staged` in their output owners, each holding its part of every row: each
    /// weight tile restores once for all, and rows past them are unread padding.
    fn project<R: AxisName>(
        &self,
        device: &mut Device,
        live: Live<(), R>,
        staged: DmTensor<bf16, Chip, Clusters<O>, Owners<I, O>, m![R, { Part<I> }]>,
    ) -> DmTensor<bf16, Chip, Clusters<O>, Reduced<I, O>, m![R, O % ({ Block<I, O> })]>
    where
        A: Op<(Live<(), R>, Staged<I, O, R>), Output = Staged<I, O, R>>,
    {
        // Batch `batch` of the rows, in the form they enter the contraction, as its lanes.
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
            Contraction::<I, O>::sums(device, tile, lanes)
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
        let load = |device: &mut Device,
                    tile: usize|
         -> Coded<Clusters<O>, Owners<I, O>, Elements<I, O>, Part<I>> {
            (
                self.codes()
                    .view()
                    .tile::<Tiles<I, O>, 1, m![
                        O / (O / 2), O % (O / 2) / ({ Block<I, O> }), { Elements<
                            I,
                            O,
                        > } # ({ Block<I, O> }), I
                    ]>(tile)
                    .to_dm(&mut device.tdma),
                self.scales()
                    .view()
                    .tile::<Tiles<I, O>, 1, m![
                        O / (O / 2), O % (O / 2) / ({ Block<I, O> }), { Elements<
                            I,
                            O,
                        > } # ({ Block<I, O> }), I / 16
                    ]>(tile)
                    .to_dm(&mut device.tdma),
            )
        };
        // A loaded tile restores in one pass: `Height` rows' block scales fit the vector registers.
        let restore = |device: &mut Device, coded| {
            Self::restore::<_, _, _, _, m![1], Elements<I, O>, _, _>(device, coded, |values| values)
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
                    && Height::<I, O>::SIZE * Chunk::<I>::SIZE / 16 * 4 <= VRF
            )
        };
        let batches = live.count::<Batch<R>>();
        // SAFETY: relabels each slice's input part as its 16-input blocks.
        let staged: Staged<I, O, R> = self
            .contract
            .input
            .forward(&mut *device, (live, unsafe { staged.reshape() }));
        // SAFETY: the inverse of the relabel above.
        let rows: DmTensor<bf16, Chip, Clusters<O>, Owners<I, O>, m![R, { Part<I> }]> =
            unsafe { staged.reshape() };
        let mut sums: Sums<I, O, R> = DmTensor::new();
        let tiles = <Tiles<I, O> as M>::SIZE;
        // Each tile's load issues before the previous tile restores, so it arrives meanwhile.
        let mut next = load(device, 0);
        if const { <Index<R> as M>::SIZE == 1 } {
            let lanes = fill(device, &rows, 0);
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
                    let lanes = fill(device, &rows, batch);
                    contract(device, &restored, &lanes, &mut sums, (tile, batch));
                }
            }
            let restored = restore(device, next);
            for batch in 0..batches {
                let lanes = fill(device, &rows, batch);
                contract(device, &restored, &lanes, &mut sums, (tiles - 1, batch));
            }
        }
        // SAFETY: every slice holds the same broadcast value, so any partition names it.
        let scale: VrfTensor<f32, Chip, Clusters<O>, Reduced<I, O>, m![1 # 8]> =
            unsafe { broadcast(device, self.contract.output.borrow()).reshape() };
        let scaled = device
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
            .vector_final();
        Rounded::rounded::<
            bf16,
            _,
            _,
            _,
            _,
            m![R % 8],
            m![R / 8, O % ({ Block<I, O> }) / 4, R % 8],
            m![O % 4],
            _,
        >(scaled)
        .commit()
    }

    fn spread<R: AxisName>(
        &self,
        device: &mut Device,
        live: Live<(), R>,
        staged: DmTensor<bf16, Chip, Clusters<O>, Owners<I, O>, m![R, { Part<I> }]>,
    ) -> Live<Spread<O, R>, R>
    where
        A: Op<(Live<(), R>, Staged<I, O, R>), Output = Staged<I, O, R>>,
    {
        let mut output = DmTensor::new();
        self.project(device, live, staged)
            .view()
            .to_dm_view(&mut device.tdma, output.view_mut());
        live.with(output)
    }
}

/// `value` in every slice's vector registers: written in one slice a cluster, then copied by the
/// switch ring, far cheaper than a DMA into each slice.
fn broadcast(
    device: &mut Device,
    value: &Scale,
) -> VrfTensor<f32, Chip, m![2], m![256], m![1 # 8]> {
    let in_dm: DmTensor<f32, Chip, m![2], m![1 # 256], m![1 # 8]> = value.to_dm(&mut device.tdma);
    device
        .sub
        .begin(in_dm.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

#[cfg(test)]
mod tests {
    use super::super::tests::{linear, noise, projection, quantized};
    use super::*;
    use crate::tensor::axis::{Eight, Sixteen};
    use crate::tensor::{
        live::{Live, tests::count},
        scalar::Value,
    };
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    // Two input parts and two tiles of twelve rows a slice; four parts and two tiles of six.
    axes![Input = 512, Output = 6144, Wide = 16384, Narrow = 1536];

    #[device(chip = 1)]
    fn projected<I: AxisName, O: AxisName, R: AxisName>(
        device: &mut Device,
        rows: &Value,
        linear: &BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Late<W4A4>>,
        input: &Rows<R, I>,
    ) -> Rows<R, O> {
        linear.forward(&mut *device, (Live::<(), R>::read(rows), input))
    }

    #[device(chip = 1)]
    fn resident<I: AxisName, O: AxisName, R: AxisName>(
        device: &mut Device,
        rows: &Value,
        linear: &BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Late<W4A16>>,
        input: &Rows<R, I>,
    ) -> Rows<R, O> {
        let value = input.to_dm(&mut device.tdma);
        let input: Live<Spread<I, R>, R> = Live::<(), R>::read(rows).with(value);
        linear
            .forward(&mut *device, &input)
            .value()
            .to_hbm(&mut device.tdma)
    }

    /// `input`'s rows in HBM, rows past `live` NaN, and the count of the live ones.
    async fn padded<I: AxisName, R: AxisName>(
        device: &mut Device,
        input: &[bf16],
        live: usize,
    ) -> (Rows<R, I>, Value) {
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
        (rows, count(live).to_hbm(&mut device.pdma).await.unwrap())
    }

    /// The live rows' outputs, rows past `live` being NaN on input.
    async fn run<I: AxisName, O: AxisName, R: AxisName>(
        device: &mut Device,
        linear: &BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Late<W4A4>>,
        input: &[bf16],
        live: usize,
    ) -> Vec<bf16> {
        let (rows, count) = padded::<I, R>(device, input, live).await;
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
        let _lease = DEVICE.lock().await;
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
            let linear =
                linear::<Input, Output>(&mut device, &codes, &scales, input_scale, 1.0).await;
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
                let output = run::<Input, Output, Eight>(&mut device, &linear, &input, live).await;
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
                    let got: Vec<u16> = output[row * Output::SIZE..][..Output::SIZE]
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
                        got, expected,
                        "input scale {input_scale}, {live} live, row {row}"
                    );
                }
            }
        }
    }

    /// Random codes, block scales and rows give each live row's exact sum of code-times-scale
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
        let linear = linear::<I, O>(device, &codes, &scales, input_scale, weight_scale_2).await;
        let output = f64::from(input_scale) * f64::from(weight_scale_2);
        for &live in lives {
            let results = run::<I, O, R>(device, &linear, &input, live).await;
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
                    let value = results[row * O::SIZE + o].to_f32() as f64;
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

    /// Each live row gives the exact projection of its quantized inputs within rounding, at every
    /// envelope and live count.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_the_reference() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        sums_exactly::<Input, Output, Eight>(&mut device, &[8, 3, 1]).await;
        sums_exactly::<Input, Output, Sixteen>(&mut device, &[16, 9]).await;
        sums_exactly::<Wide, Narrow, Eight>(&mut device, &[8]).await;
    }

    /// BF16 rows on chip enter the contraction as they are and the weight tensor scale multiplies
    /// each live row's sum once, after the reduce: fractional inputs stay unquantized, and a sum
    /// that cancels comes out as the late scale rounds it, not as weights rounded with the scale
    /// would.
    #[tokio::test(flavor = "current_thread")]
    async fn scales_bf16_rows_late() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(resident.topology()).unwrap();
        // Row `o` holds two E2M1 codes, at inputs `o` and `o + 1`, under E4M3 block scales of 0.75
        // to 2; row 0 takes 1 * x[0] + 1.5 * x[1] under a unit block scale.
        let (first, second) = (
            [0b0010u8, 0b0101, 0b1001, 0b0111],
            [0b0011u8, 0b1100, 0b0001],
        );
        let mut codes = vec![f4e2m1::from_bits(0); Output::SIZE * Input::SIZE];
        for o in 0..Output::SIZE {
            codes[o * Input::SIZE + o % Input::SIZE] = f4e2m1::from_bits(first[o % 4]);
            codes[o * Input::SIZE + (o + 1) % Input::SIZE] = f4e2m1::from_bits(second[o % 3]);
        }
        let scales: Vec<f8e4m3> = (0..Output::SIZE * Input::SIZE / 16)
            .map(|at| f8e4m3::from_f32([1.0, 1.25, 0.75, 2.0][at % 4]))
            .collect();
        let weight_scale_2 = 0.1f32;
        let linear =
            projection::<Input, Output, _>(&mut device, &codes, &scales, W4A16, weight_scale_2)
                .await;
        // Sixteenths from -1.875 to 1.875, so every product and sum is exact in FP32; row 0 puts
        // 1 and -1 under the first two codes, whose sum -0.5 cancels.
        let mut input: Vec<f32> = (0..Eight::SIZE * Input::SIZE)
            .map(|at| ((at % Input::SIZE * 13 + at / Input::SIZE * 29) % 61) as f32 / 16.0 - 1.875)
            .collect();
        input[..2].copy_from_slice(&[1.0, -1.0]);
        let input: Vec<bf16> = input.into_iter().map(bf16::from_f32).collect();
        let restored = |o: usize, k: usize| {
            codes[o * Input::SIZE + k].to_f32() * scales[(o * Input::SIZE + k) / 16].to_f32()
        };
        let early = |o: usize, k: usize| bf16::from_f32(restored(o, k) * weight_scale_2).to_f32();
        // A row's outputs from its `operand` under `weight`, each sum times `factor`.
        let outputs = |operand: &[f32], weight: &dyn Fn(usize, usize) -> f32, factor: f32| {
            (0..Output::SIZE)
                .map(|o| {
                    let sum: f32 = [o % Input::SIZE, (o + 1) % Input::SIZE]
                        .into_iter()
                        .map(|k| weight(o, k) * operand[k])
                        .sum();
                    let value = bf16::from_f32(sum * factor);
                    if value.to_f32() == 0.0 {
                        0
                    } else {
                        value.to_bits()
                    }
                })
                .collect::<Vec<u16>>()
        };
        let row = |r: usize| -> Vec<f32> {
            input[r * Input::SIZE..][..Input::SIZE]
                .iter()
                .map(|x| x.to_f32())
                .collect()
        };
        assert_eq!(
            [
                outputs(&row(0), &restored, weight_scale_2)[0],
                outputs(&row(0), &early, 1.0)[0]
            ],
            [-0.050048828125, -0.05029296875].map(|value| bf16::from_f32(value).to_bits())
        );
        for live in [8, 3] {
            let (rows, count) = padded::<Input, Eight>(&mut device, &input, live).await;
            let output: Vec<u16> = launch(resident, (&mut device, &count, &linear, &rows))
                .await
                .unwrap()
                .to_host::<m![Eight, Output]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec()
                .into_iter()
                .map(|value| {
                    if value.to_f32() == 0.0 {
                        0
                    } else {
                        value.to_bits()
                    }
                })
                .collect();
            for r in 0..live {
                let expected = outputs(&row(r), &restored, weight_scale_2);
                let quantized = quantized(
                    &input[r * Input::SIZE..][..Input::SIZE],
                    row(r).iter().fold(0.0f32, |m, x| m.max(x.abs())) / (6.0 * 448.0),
                );
                // The expectation tells both mistakes apart: weights rounded with the scale, and
                // inputs quantized as for a block-scaled product.
                assert_ne!(expected, outputs(&row(r), &early, 1.0), "row {r}");
                assert_ne!(
                    expected,
                    outputs(&quantized, &restored, weight_scale_2),
                    "row {r}"
                );
                assert_eq!(
                    output[r * Output::SIZE..][..Output::SIZE],
                    expected,
                    "{live} live, row {r}"
                );
            }
        }
    }
}
