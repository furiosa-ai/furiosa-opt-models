//! Elementwise sums, in FP32 and rounded once.

use super::{Float, Window};
use crate::Op;
use crate::load::Load;
use crate::modules::generic::schedule::{Fill, IntoStream, Slot, Spill, Stream};
use crate::ops::normalization::span::{self, Owned, Span};
use crate::ops::normalization::{
    Blocked, Chunk, Cluster, Element, FeatureSlices, Offset, Spread, Tile, TilePart, TileRows,
};
use crate::ops::pointwise::gated;
use crate::tensor::{axis::Single, live::Live};
use core::task::Poll;
use furiosa_opt_std::prelude::*;

/// Elementwise sum of two tensors in the same layout.
#[derive(Clone, Default, DeviceSend, Load)]
pub struct Add;

// --8<-- [start:add]
/// HBM inputs and output retain their element type and physical layout.
impl<D: Float, S: M + 'static, Chips: M> Op<(&HbmTensor<D, Chips, S>, &HbmTensor<D, Chips, S>)>
    for Add
{
    type Output = HbmTensor<D, Chips, S>;

    fn forward(
        &self,
        device: &mut Device,
        (left, right): (&HbmTensor<D, Chips, S>, &HbmTensor<D, Chips, S>),
    ) -> Self::Output {
        Window::<S>::binary(device, [left, right], FpBinaryOp::AddF)
    }
}

/// DM rows accumulate in f32 and round the result to bf16.
impl<'x, 'y, Hidden: AxisName, Rows: AxisName, Chips: M + 'static>
    Op<(
        &'x Spread<Hidden, Rows, Chips>,
        &'y Spread<Hidden, Rows, Chips>,
    )> for Add
{
    type Output = Spread<Hidden, Rows, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (lhs, rhs): (
            &'x Spread<Hidden, Rows, Chips>,
            &'y Spread<Hidden, Rows, Chips>,
        ),
    ) -> Self::Output {
        let rhs: VrfTensor<f32, Chips, Cluster, FeatureSlices<Hidden>, Element<Hidden, Rows>> =
            device
                .sub
                .begin(rhs.view())
                .fetch::<m![Rows, Hidden % ({ Symbol<Chunk<Hidden>> }) / 8], m![Hidden % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Rows, Hidden % ({ Symbol<Chunk<Hidden>> }) / 8], m![Hidden % 8]>()
                .to_vrf();
        device
            .main
            .begin(lhs.view())
            .fetch::<m![Rows, Hidden % ({ Symbol<Chunk<Hidden>> }) / 8], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows, Hidden % ({ Symbol<Chunk<Hidden>> }) / 8], m![Hidden % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &rhs)
            .vector_final()
            .cast::<bf16, m![Hidden % 8 # 16]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit()
    }
}

/// Both terms are the same rows of one launch, all of them or one span; the sum keeps their live
/// rows.
impl<'x, 'y, Hidden: AxisName, Rows: AxisName, Chips: M + 'static>
    Op<(
        &'x Live<Spread<Hidden, Rows, Chips>, Rows>,
        &'y Live<Spread<Hidden, Rows, Chips>, Rows>,
    )> for Add
{
    type Output = Live<Spread<Hidden, Rows, Chips>, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        (lhs, rhs): (
            &'x Live<Spread<Hidden, Rows, Chips>, Rows>,
            &'y Live<Spread<Hidden, Rows, Chips>, Rows>,
        ),
    ) -> Self::Output {
        lhs.with(self.forward(device, (lhs.value(), rhs.value())))
    }
}

/// The live rows of each cluster's half, capped at the half; the sum keeps the terms' live rows.
impl<Width: AxisName, Rows: AxisName, R: AxisName, Chips: M + 'static>
    Op<(
        &Live<Owned<Width, Rows, Chips>, R>,
        &Live<Owned<Width, Rows, Chips>, R>,
    )> for Add
{
    type Output = Live<Owned<Width, Rows, Chips>, R>;

    fn forward(
        &self,
        context: &mut Device,
        (lhs, rhs): (
            &Live<Owned<Width, Rows, Chips>, R>,
            &Live<Owned<Width, Rows, Chips>, R>,
        ),
    ) -> Self::Output {
        let reach = span::reach::<Rows>(lhs.rows());
        let mut sum = Owned::<Width, Rows, Chips>::new();
        Add::of::<
            m![Rows % (Rows / 2) / 8],
            m![Rows % 8, { Offset<Width> } / 8],
            m![Width % 8],
            Term,
            _,
            _,
            _,
            _,
        >(context, (lhs.value(), rhs.value()))
        .commit_prefix::<m![Rows % (Rows / 2) / 8], _, _>(sum.view_mut(), reach.div_ceil(8));
        lhs.with(sum)
    }
}

// --8<-- [end:add]

/// Rows an add keeps from its input until the branch tile of the same rows comes, and the place
/// that holds a launch's tiles of them.
pub(crate) trait Kept<R>: Sized {
    type Place: for<'r> Spill<Device, &'r Self, Live<(), R>>
        + Fill<Device, Live<(), R>, Rows = Self>;

    fn place() -> Self::Place;
}

/// Spans wait in the launch's rows in HBM.
impl<Width: AxisName, R: AxisName, Chips: M + 'static> Kept<R> for Owned<Width, Span, Chips> {
    type Place = HbmTensor<bf16, Chips, Blocked<Width, R>>;

    fn place() -> Self::Place {
        HbmTensor::new()
    }
}

/// A launch's rows streamed as one tile wait as a copy in data memory.
impl<Width: AxisName, R: AxisName, Chips: M + 'static> Kept<R> for Spread<Width, R, Chips> {
    type Place = Self;

    fn place() -> Self {
        DmTensor::new()
    }
}

impl<Width: AxisName, R: AxisName, Chips: M + 'static>
    Spill<Device, &Spread<Width, R, Chips>, Live<(), R>> for Spread<Width, R, Chips>
{
    fn spill(&mut self, device: &mut Device, rows: &Spread<Width, R, Chips>, _: Live<(), R>) {
        *self = rows.to_dm(&mut device.tdma);
    }
}

impl<Width: AxisName, R: AxisName, Chips: M + 'static> Fill<Device, Live<(), R>>
    for Spread<Width, R, Chips>
{
    type Rows = Self;

    fn fill(&self, device: &mut Device, _: Live<(), R>) -> Self {
        self.to_dm(&mut device.tdma)
    }
}

/// A residual's sum as a stream: its node keeps each input tile in a place of its own until the
/// branch tile of the same rows comes, so a branch may take every input first.
impl<'s, U, E, X: Kept<R>, R: AxisName> IntoStream<U, E, &'s mut ()> for Add
where
    U: Stream<Item = (Option<Live<X, R>>, Option<Live<X, R>>)>,
    Node<X::Place, U>: Stream,
{
    type Stream = Node<X::Place, U>;

    fn into_stream(self, _: E, _: &'s mut (), upstream: U) -> Self::Stream {
        Node {
            kept: X::place(),
            upstream,
        }
    }
}

impl<'m, 's, U: Stream, E> IntoStream<U, E, &'s mut ()> for &'m Add
where
    Add: IntoStream<U, E, &'s mut ()>,
{
    type Stream = <Add as IntoStream<U, E, &'s mut ()>>::Stream;

    fn into_stream(self, env: E, unit: &'s mut (), upstream: U) -> Self::Stream {
        Add.into_stream(env, unit, upstream)
    }
}

/// An add's node: the place its input tiles wait in, and its upstream, the residual's node.
pub struct Node<K, U> {
    kept: K,
    upstream: U,
}

impl<T, K, U: AsMut<Slot<T>>> AsMut<Slot<T>> for Node<K, U> {
    fn as_mut(&mut self) -> &mut Slot<T> {
        self.upstream.as_mut()
    }
}

/// A branch tile lands before its kept tile, so a branch from far memory overlaps the fill.
impl<X, R: AxisName, K, U> Stream for Node<K, U>
where
    U: Stream<Item = (Option<Live<X, R>>, Option<Live<X, R>>)>,
    K: for<'r> Spill<Device, &'r X, Live<(), R>> + Fill<Device, Live<(), R>, Rows = X>,
    Add: for<'x, 'y> Op<(&'x Live<X, R>, &'y Live<X, R>), Output = Live<X, R>>,
{
    type Item = Live<X, R>;

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Self::Item>> {
        match self.upstream.poll_next(device) {
            Poll::Ready(Some((input, branch))) => {
                if let Some(input) = input {
                    self.kept.spill(device, input.value(), input.with(()));
                }
                let Some(branch) = branch else {
                    return Poll::Pending;
                };
                let kept = branch.with(self.kept.fill(device, branch.with(())));
                Poll::Ready(Some(Add.forward(device, (&kept, &branch))))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// The FP32 vectors of the features the add keeps in its registers: the gate and the residual.
const VECTORS: usize = 2;

/// `x + gate * branch` per row of an FP32 residual and a BF16 branch, FP32 inside and out, a tile
/// at a time: tile t computes while tile t + 1 arrives and tile t - 1 is stored.
impl<W: AxisName, R: AxisName, Chips: M>
    Op<(
        &HbmTensor<f32, Chips, m![R, W]>,
        &HbmTensor<bf16, Chips, m![R, W]>,
        &HbmTensor<f32, Chips, m![Single, W]>,
    )> for gated::Add
{
    type Output = HbmTensor<f32, Chips, m![R, W]>;

    fn forward(
        &self,
        device: &mut Device,
        (input, branch, gate): (
            &HbmTensor<f32, Chips, m![R, W]>,
            &HbmTensor<bf16, Chips, m![R, W]>,
            &HbmTensor<f32, Chips, m![Single, W]>,
        ),
    ) -> Self::Output {
        type T<W> = Symbol<TileRows<W, VECTORS>>;
        let tiles = const {
            assert!(
                R::SIZE.is_multiple_of(TileRows::<W, VECTORS>::SIZE),
                "rows come in whole tiles"
            );
            R::SIZE / TileRows::<W, VECTORS>::SIZE
        };
        // SAFETY: a single row is its features.
        let in_dm: Tile<f32, R, W, VECTORS, Chips> =
            unsafe { gate.view().reshape::<Chips, m![W]>() }.to_dm(&mut device.tdma);
        let gate = device
            .sub
            .begin(in_dm.view())
            .fetch::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
            .collect::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
            .to_vrf::<TilePart<W, VECTORS>>();
        let input_to_dm = |device: &mut Device,
                           at: usize,
                           tile: &mut Tile<f32, R, W, VECTORS, Chips>| {
            input
                .view()
                .tile::<m![R / ({ T<W> })], 1, m![R / ({ T<W> }) = 1 # (R / ({ T<W> })), R % ({ T<W> }), W]>(at)
                .to_dm_view(&mut device.tdma, tile.view_mut());
        };
        let branch_to_dm =
            |device: &mut Device, at: usize, tile: &mut Tile<bf16, R, W, VECTORS, Chips>| {
                branch
                    .view()
                    .tile::<m![R / ({ T<W> })], 1, m![R / ({ T<W> }) = 1 # (R / ({ T<W> })), R % ({ T<W> }), W]>(at)
                    .to_dm_view(&mut device.tdma, tile.view_mut());
            };
        let mut output = HbmTensor::new();
        let mut store = |device: &mut Device, tile: &Tile<f32, R, W, VECTORS, Chips>, at: usize| {
            tile.view().to_hbm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![R / ({ T<W> })], 1, m![R / ({ T<W> }) = 1 #{!} (R / ({ T<W> })), R % ({ T<W> }), W]>(at),
            );
        };
        let add = |device: &mut Device,
                   (rows, update): (
            &Tile<f32, R, W, VECTORS, Chips>,
            &Tile<bf16, R, W, VECTORS, Chips>,
        ),
                   sum: &mut Tile<f32, R, W, VECTORS, Chips>| {
            let held = device
                .sub
                .begin(rows.view())
                .fetch::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
                .collect::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
                .to_vrf::<TilePart<W, VECTORS>>();
            device
                .main
                .begin(update.view())
                .fetch::<m![{ TilePart<W, VECTORS> } / 16], m![W % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ TilePart<W, VECTORS> } / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate)
                .vector_fp_binary(FpBinaryOp::AddF, &held)
                .vector_widen_concat::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
                .vector_final()
                .commit_trim::<m![W % 8]>()
                .commit_view(sum.view_mut());
        };
        let (mut first, mut second): (
            Tile<f32, R, W, VECTORS, Chips>,
            Tile<f32, R, W, VECTORS, Chips>,
        ) = (DmTensor::new(), DmTensor::new());
        let (mut update, mut next): (
            Tile<bf16, R, W, VECTORS, Chips>,
            Tile<bf16, R, W, VECTORS, Chips>,
        ) = (DmTensor::new(), DmTensor::new());
        let (mut kept, mut held): (
            Tile<f32, R, W, VECTORS, Chips>,
            Tile<f32, R, W, VECTORS, Chips>,
        ) = (DmTensor::new(), DmTensor::new());
        input_to_dm(&mut *device, 0, &mut first);
        branch_to_dm(&mut *device, 0, &mut update);
        if const { R::SIZE / TileRows::<W, VECTORS>::SIZE > 1 } {
            input_to_dm(&mut *device, 1, &mut second);
            branch_to_dm(&mut *device, 1, &mut next);
        }
        for pair in 0..tiles / 2 {
            let at = 2 * pair;
            add(&mut *device, (&first, &update), &mut kept);
            input_to_dm(&mut *device, (at + 2).min(tiles - 1), &mut first);
            branch_to_dm(&mut *device, (at + 2).min(tiles - 1), &mut update);
            store(&mut *device, &kept, at);
            add(&mut *device, (&second, &next), &mut held);
            input_to_dm(&mut *device, (at + 3).min(tiles - 1), &mut second);
            branch_to_dm(&mut *device, (at + 3).min(tiles - 1), &mut next);
            store(&mut *device, &held, at + 1);
        }
        if const { R::SIZE / TileRows::<W, VECTORS>::SIZE % 2 == 1 } {
            add(&mut *device, (&first, &update), &mut kept);
            store(&mut *device, &kept, tiles - 1);
        }
        output
    }
}

/// The two terms of an add whose stream reads them interleaved.
#[derive(Debug, Clone)]
pub(crate) struct Term;

impl AxisName for Term {
    const NAME: Ident = Ident::new("Term");
    const SIZE: usize = 2;
}

impl Add {
    /// `left + right` in FP32, rounded once to BF16, the terms read interleaved along `Inputs` in
    /// `Time` packets of `Run` runs of `Packet` values: the stream a caller commits.
    pub(crate) fn of<
        'l,
        Time: M,
        Run: M,
        Packet: M,
        Inputs: AxisName,
        Chip: M,
        Cluster: M,
        Slice: M,
        Element: M,
    >(
        device: &'l mut Device,
        (left, right): (
            &'l DmTensor<bf16, Chip, Cluster, Slice, Element>,
            &'l DmTensor<bf16, Chip, Cluster, Slice, Element>,
        ),
    ) -> CommitTrimTensor<'l, { Tu::Main }, bf16, Chip, Cluster, Slice, m![{ Time }, { Run }], Packet>
    {
        device
            .main
            .begin_interleaved::<Inputs, _, _, _, _, _>(left.view(), right.view())
            .fetch::<m![{ Time }, Inputs, { Run }], Packet>()
            .fetch_cast::<f32>()
            .collect::<m![{ Time }, Inputs, { Run }], Packet>()
            .vector_init()
            .vector_intra_slice_unzip::<Inputs, m![{ Time }, { Run }]>()
            .vector_clip_zip(ClipBinaryOpF32::Add)
            .vector_final()
            .cast::<bf16, m![{ Packet } # 16]>()
            .commit_trim::<Packet>()
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use crate::ops::normalization::{
        RowMajor, Spread,
        tests::{random, residual},
    };

    use super::Add;
    use crate::Op;
    use crate::ops::pointwise::gated;
    use crate::tensor::axis::{Eight, Single};
    use crate::tensor::parallel::Chip;
    use crate::tests::DEVICE;

    axes![Pinned = 1024];

    type Rows = HbmTensor<bf16, Chip, RowMajor<Pinned, Eight>>;

    fn terms(step: usize) -> Vec<bf16> {
        (0..Eight::SIZE * Pinned::SIZE)
            .map(|index| bf16::from_f32(((index * step) % 101) as f32 / 25.0 - 2.0))
            .collect()
    }

    // --8<-- [start:summed]
    /// The DM-resident add, round-tripped through HBM so the host can read it.
    #[device(chip = 1)]
    fn summed(device: &mut Device, lhs: &Rows, rhs: &Rows, output: &mut Rows) {
        let lhs: Spread<Pinned, Eight> = lhs.to_dm(&mut device.tdma);
        let rhs: Spread<Pinned, Eight> = rhs.to_dm(&mut device.tdma);
        Add.forward(&mut *device, (&lhs, &rhs))
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }
    // --8<-- [end:summed]

    #[tokio::test(flavor = "current_thread")]
    async fn adds_elementwise() {
        let mut device = Device::new(summed.topology()).unwrap();
        let (lhs, rhs) = (terms(37), terms(53));
        let expected: Vec<f32> = lhs
            .iter()
            .zip(&rhs)
            .map(|(a, b)| a.to_f32() + b.to_f32())
            .collect();
        let lhs = HostTensor::<bf16, RowMajor<Pinned, Eight>>::from_vec(lhs)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let rhs = HostTensor::<bf16, RowMajor<Pinned, Eight>>::from_vec(rhs)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut output = HostTensor::<bf16, RowMajor<Pinned, Eight>>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        launch(summed, (&mut device, &lhs, &rhs, &mut output))
            .await
            .unwrap();
        let got = output
            .to_host::<RowMajor<Pinned, Eight>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();

        for (index, (got, expected)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (got.to_f32() - expected).abs() <= 0.02,
                "index {index}: {} vs {expected}",
                got.to_f32()
            );
        }
    }

    axes![Quarters = 2048, Residual = 2048];

    #[device(chip = 1)]
    fn gated(
        device: &mut Device,
        input: &HbmTensor<f32, Chip, m![Residual, Quarters]>,
        branch: &HbmTensor<bf16, Chip, m![Residual, Quarters]>,
        gate: &HbmTensor<f32, Chip, m![Single, Quarters]>,
    ) -> HbmTensor<f32, Chip, m![Residual, Quarters]> {
        gated::Add.forward(device, (input, branch, gate))
    }

    /// A BF16 branch scaled by a per-feature gate adds to the FP32 residual in FP32, every row
    /// against the FP64 sum.
    #[tokio::test(flavor = "current_thread")]
    async fn adds_a_scaled_branch_in_fp32() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(gated.topology()).unwrap();
        let inputs = residual(Residual::SIZE, Quarters::SIZE);
        let branch = random(Residual::SIZE * Quarters::SIZE, 0x1405_7b7e_f767_814f);
        let gate: Vec<f32> = random(Quarters::SIZE, 0x9e37_79b9_7f4a_7c15)
            .iter()
            .map(|v| v.to_f32() / 8.0)
            .collect();
        let input = HostTensor::<f32, m![Residual, Quarters]>::from_vec(inputs.clone())
            .to_hbm::<Chip, m![Residual, Quarters]>(&mut device.pdma)
            .await
            .unwrap();
        let update = HostTensor::<bf16, m![Residual, Quarters]>::from_vec(branch.clone())
            .to_hbm::<Chip, m![Residual, Quarters]>(&mut device.pdma)
            .await
            .unwrap();
        let scale = HostTensor::<f32, m![Single, Quarters]>::from_vec(gate.clone())
            .to_hbm::<Chip, m![Single, Quarters]>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(gated, (&mut device, &input, &update, &scale))
            .await
            .unwrap()
            .to_host::<m![Residual, Quarters]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (index, got) in output.iter().enumerate() {
            let column = index % Quarters::SIZE;
            let expected = f64::from(inputs[index])
                + f64::from(gate[column]) * f64::from(branch[index].to_f32());
            assert!(
                (f64::from(*got) - expected).abs() <= expected.abs() * 1e-6 + 1e-6,
                "row {} column {column}: {got} vs {expected}",
                index / Quarters::SIZE
            );
        }
    }
}
