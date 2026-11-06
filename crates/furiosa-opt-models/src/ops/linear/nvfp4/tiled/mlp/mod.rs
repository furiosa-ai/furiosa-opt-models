//! Span MLP: paired gate/up sweeps, FP32 partials, and a blocked product/down handoff.
//! W4A16 arithmetic retains per-16 weight block scales and FP32 sums.

use super::{block, panels::Panels};
use crate::ops::linear::nvfp4::tiled::prefetch::Points;
use crate::ops::normalization::span::Owned;
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;

mod compact;
pub mod down;
mod gate;
mod product;
mod restored_compact;

pub use down::Down;

/// The paired gate/up projections and partitioned down projection of a gated MLP.
#[derive(DeviceSend, crate::load::Load)]
pub struct Mlp<H: AxisName, I: AxisName> {
    pub gate_proj: Panels<H, I>,
    pub up_proj: Panels<H, I>,
    pub down_proj: Down<I, H>,
}
impl<H: AxisName, I: AxisName, const N: usize> crate::modules::generic::collection::View<N>
    for Mlp<H, I>
{
    type Storage = (
        <Panels<H, I> as crate::modules::generic::collection::View<N>>::Storage,
        <Panels<H, I> as crate::modules::generic::collection::View<N>>::Storage,
        <Down<I, H> as crate::modules::generic::collection::View<N>>::Storage,
    );
    type Layer = Self;
    fn view(storage: &Self::Storage, layer: usize) -> Self {
        use crate::modules::generic::collection::View;
        Self {
            gate_proj: <Panels<H, I> as View<N>>::view(&storage.0, layer),
            up_proj: <Panels<H, I> as View<N>>::view(&storage.1, layer),
            down_proj: <Down<I, H> as View<N>>::view(&storage.2, layer),
        }
    }
    fn stack(layers: [Self; N], cx: &mut crate::load::Context<'_>) -> Self::Storage {
        use crate::modules::generic::collection::View;
        let (mut gate, mut up, mut down) = (Vec::new(), Vec::new(), Vec::new());
        for layer in layers {
            gate.push(layer.gate_proj);
            up.push(layer.up_proj);
            down.push(layer.down_proj);
        }
        (
            <Panels<H, I> as View<N>>::stack(gate.try_into().ok().unwrap(), cx),
            <Panels<H, I> as View<N>>::stack(up.try_into().ok().unwrap(), cx),
            <Down<I, H> as View<N>>::stack(down.try_into().ok().unwrap(), cx),
        )
    }
}

type Cluster<I> = m![I / (I / 2)];
type Owners<I> = m![I % (I / 2) / 224];
// The low six row bits are eight slice owners of eight lanes. Keeping bit six in
// local time lets a contraction trim a 128-row wave to its first 64 rows.
type Slices<I, R> = m![{ Owners<I> }, R % 64 / 8];
type LocalRows<R> = m![R % 128 / 64, R % 8];

type Ready<I, R> =
    DmTensor<bf16, Chip, Cluster<I>, Slices<I, R>, m![R / 128, { LocalRows<R> }, I % 224]>;
type Products<I, R> = HbmTensor<bf16, Chip, m![I / 224, I % 224 / 32, R, I % 32]>;
pub(super) trait Product<I: AxisName, R: AxisName> {
    fn stripe<O: AxisName, const N: usize, E: M>(
        &self,
        device: &mut Device,
        stripe: usize,
        start: usize,
        target: DmTensorViewMut<'_, bf16, Chip, Cluster<I>, down::Partition<I, O>, E>,
    );
}

impl<I: AxisName, R: AxisName> Product<I, R> for Products<I, R> {
    fn stripe<O: AxisName, const N: usize, E: M>(
        &self,
        device: &mut Device,
        stripe: usize,
        start: usize,
        target: DmTensorViewMut<'_, bf16, Chip, Cluster<I>, down::Partition<I, O>, E>,
    ) {
        self.view()
            .tile::<m![I % 224 / 32], 1, m![I / 224, I % 224 / 32 = 1 # 7, R, I % 32]>(stripe)
            .tile::<m![R], N, m![I / 224, I % 224 / 32 = 1 # 7, R = {N} # R, I % 32]>(start)
            .to_dm_view(&mut device.tdma, target);
    }
}

type Input<H, R> = HbmTensor<bf16, Chip, m![H / 128, R, H % 128]>;

pub(crate) type Entry<H, I> = (
    super::panels::Packed<H, I>,
    super::panels::Packed<H, I>,
    down::Packed<I, H, down::Wide>,
);

/// Stage the first gate/up slab before the preceding O projection. The returned
/// point loads down panel 0 while O's first contraction is in flight.
pub(crate) fn stage<'a, H: AxisName, I: AxisName>(
    device: &mut Device,
    gate: &Panels<H, I>,
    up: &Panels<H, I>,
    down: &'a Down<I, H>,
) -> impl Points<Device, Output = Entry<H, I>> + use<'a, H, I> {
    let gate = gate.load(device, 0);
    let up = up.load(device, 0);
    crate::ops::linear::nvfp4::tiled::prefetch::Stage(move |device: &mut Device| {
        (gate, up, down.stage(device, 0))
    })
}

/// Gate and up share unquantized BF16 input rows; their weight scales remain independent.
pub fn forward<H: AxisName, I: AxisName, R: AxisName>(
    device: &mut Device,
    gate: &Panels<H, I>,
    up: &Panels<H, I>,
    down: &Down<I, H>,
    input: &Owned<H, R>,
    rows: usize,
) -> Owned<H, R> {
    forward_with_next(device, gate, up, down, input, rows, ()).0
}

/// Host the caller's following operand loads at the first two down-panel boundaries.
pub(crate) fn forward_with_next<H: AxisName, I: AxisName, R: AxisName, N: Points<Device>>(
    device: &mut Device,
    gate: &Panels<H, I>,
    up: &Panels<H, I>,
    down: &Down<I, H>,
    input: &Owned<H, R>,
    rows: usize,
    next: N,
) -> (Owned<H, R>, N::Output) {
    let entry = stage(device, gate, up, down).finish(device);
    forward_prepared(device, gate, up, down, input, rows, entry, next)
}

pub(crate) fn forward_prepared<H: AxisName, I: AxisName, R: AxisName, N: Points<Device>>(
    device: &mut Device,
    gate: &Panels<H, I>,
    up: &Panels<H, I>,
    down: &Down<I, H>,
    input: &Owned<H, R>,
    rows: usize,
    entry: Entry<H, I>,
    next: N,
) -> (Owned<H, R>, N::Output) {
    const {
        assert!(H::SIZE == 4096 && I::SIZE == 14336 && R::SIZE == 512);
    }
    let mut product = Products::new();
    let first = if rows > 16 {
        let (g, u, first) = gate::sweep::<H, I, R>(device, gate, up, down, input, rows, entry);
        product::stage(device, &g, &u, rows, &mut product);
        first
    } else if rows > 8 {
        restored_compact::prepare::<H, I, R, 16>(device, gate, up, down, input, entry, &mut product)
    } else {
        compact::prepare::<H, I, R, 8>(device, gate, up, down, input, entry, &mut product)
    };
    down.forward(device, &product, first, rows, next)
}

/// FP32 panel sums remain in DM; no BF16 rounding occurs between panels.
fn sum_scaled<C: M, S: M, Time: M, Feature: M>(
    device: &mut Device,
    scales: &super::Scales,
    left: &DmTensor<f32, Chip, C, S, m![{ Time }, { Feature }]>,
    right: &DmTensor<f32, Chip, C, S, m![{ Time }, { Feature }]>,
) -> DmTensor<bf16, Chip, C, S, m![{ Time }, { Feature }]> {
    const {
        assert!(Feature::SIZE <= 256 && Feature::SIZE.is_multiple_of(8));
    }
    axes![Term = 2];
    let factor = block::scalar::<C, S>(device, &scales.output);
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(left.view(), right.view())
        .fetch::<m![{ Time }, Term, { Feature } / 8], m![{ Feature } % 8]>()
        .collect::<m![{ Time }, Term, { Feature } / 8], m![{ Feature } % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![{ Time }, { Feature } / 8]>()
        .vector_narrow_split::<m![{ Time }, { Feature } / 4], m![{ Feature } % 4]>()
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factor)
        .vector_widen_concat::<m![{ Time }, { Feature } / 8], m![{ Feature } % 8]>()
        .vector_final()
        .cast::<bf16, m![{ Feature } % 8 # 16]>()
        .commit_trim::<m![{ Feature } % 8]>()
        .commit()
}

/// Add views directly into a caller-owned buffer. No temporary old-sum tensor or
/// copy back into the span is needed; all partials remain FP32.
fn sum_into<C: M, S: M, Time: M, Feature: M, Prefix: M, Input: M, E: M>(
    device: &mut Device,
    left: DmTensorView<'_, f32, Chip, C, S, Input>,
    right: DmTensorView<'_, f32, Chip, C, S, Input>,
    output: DmTensorViewMut<'_, f32, Chip, C, S, E>,
    live: usize,
) {
    axes![Side = 2];
    device
        .main
        .begin_interleaved::<Side, _, _, _, _, _>(left, right)
        .fetch::<m![{ Time }, Side, { Feature } / 8], m![{ Feature } % 8]>()
        .collect::<m![{ Time }, Side, { Feature } / 8], m![{ Feature } % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Side, m![{ Time }, { Feature } / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .commit_trim::<m![{ Feature } % 8]>()
        .commit_prefix::<Prefix, _, _>(output, live);
}
