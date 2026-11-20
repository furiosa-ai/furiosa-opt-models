//! Per-chip FP32 sums of a pointwise module's results, with its element-type rounding preserved.

use super::{Extent, Flat, Float, Rows, Segment, Width, Window};
use crate::load::Load;
use crate::tensor::memory::residency::Residency;
use crate::{Module, Op};
use furiosa_opt_std::prelude::*;

/// Sum every element produced by `F`, independently on each chip.
///
/// `F` acts on one DM window at a time and returns that window in the source element type.
/// `Sum(Abs)` is an L1 norm; `Sum(Then(Subtract, Abs))` is an L1 distance with each difference
/// rounded before taking its magnitude. Each window is reduced in FP32, its sum stored in HBM,
/// and the window sums reduced in a final pass. Only the first lane of each result is defined.
#[derive(Clone, Default, DeviceSend, Load, Residency)]
#[residency(children(F))]
pub struct Sum<F>(#[load(rename = "")] pub F);

/// Each chip's sum in the first element of its eight-element vector.
pub type Total<Chips> = HbmTensor<f32, Chips, m![1 # 8]>;

/// A wrapper of a child module, with that child's state, rather than an op with empty state.
impl<I, F> !Op<I> for Sum<F> {}

impl<D: Float, S: M + 'static, Chips: M, F, Env: Copy, State>
    Module<&HbmTensor<D, Chips, S>, Env, &mut State> for Sum<F>
where
    F: for<'t, 's> Module<&'t Rows<D, S, Chips>, Env, &'s mut State, Output = Rows<D, S, Chips>>,
{
    type Output = Total<Chips>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &mut State,
        input: &HbmTensor<D, Chips, S>,
    ) -> Self::Output {
        Windowed::<S>::sum(input, device, |device, tile| {
            self.0.forward(device, env, &mut *state, tile)
        })
    }
}

impl<D: Float, S: M + 'static, Chips: M, F, Env: Copy, State>
    Module<(&HbmTensor<D, Chips, S>, &HbmTensor<D, Chips, S>), Env, &mut State> for Sum<F>
where
    F: for<'t, 's> Module<
            &'t (Rows<D, S, Chips>, Rows<D, S, Chips>),
            Env,
            &'s mut State,
            Output = Rows<D, S, Chips>,
        >,
{
    type Output = Total<Chips>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &mut State,
        operands: (&HbmTensor<D, Chips, S>, &HbmTensor<D, Chips, S>),
    ) -> Self::Output {
        Windowed::<S>::sum(operands, device, |device, tiles| {
            self.0.forward(device, env, &mut *state, tiles)
        })
    }
}

/// A window's index in the flattened input.
type Index<S, G = S> = m![{ Width<S> } / ({ Extent<G> })];
type Packet<S, G> = m![
    { Width<S> } / ({ Extent<G> }) = 1 # ({ Width<S> } / ({ Extent<G> })),
    { Width<S> } % ({ Extent<G> })
];
type Slot<S, G> = m![
    { Width<S> } / ({ Extent<G> }) = 1 #{!} ({ Width<S> } / ({ Extent<G> })),
    { Width<S> } % ({ Extent<G> })
];

/// One FP32 value per chip, in an eight-element vector on one slice of one cluster.
type LocalTotal<Chips> = DmTensor<f32, Chips, m![1 # 2], m![1 # 256], m![1 # 8]>;

/// A tensor or a tuple of tensors whose corresponding windows land in DM together.
/// `S` is the allocation shape; `G` determines numerical window geometry and defaults to `S`.
/// Checked prefixes use their whole-block geometry so capacity cannot change their tiling.
/// Kept private to pointwise operations; unary HBM mapping already uses `Window::map`.
pub(super) trait Windowed<S: M + 'static, G: M + 'static = S>: Sized {
    type Tile;

    fn land(&self, device: &mut Device, index: usize) -> Self::Tile;

    fn map<D, Chips, F>(self, device: &mut Device, op: &F) -> HbmTensor<D, Chips, S>
    where
        D: MaterializableScalar,
        Chips: M,
        F: for<'t> Op<&'t Self::Tile, Output = Rows<D, G, Chips>>,
    {
        self.map_windows(device, S::SIZE / Window::<G>::SIZE, op)
    }

    /// Map initialized whole windows without reading or filling source allocation tails.
    fn map_windows<D, Chips, F>(
        self,
        device: &mut Device,
        active: usize,
        op: &F,
    ) -> HbmTensor<D, Chips, S>
    where
        D: MaterializableScalar,
        Chips: M,
        F: for<'t> Op<&'t Self::Tile, Output = Rows<D, G, Chips>>,
    {
        const {
            assert!(S::SIZE.is_multiple_of(Window::<G>::SIZE));
            assert!(
                S::SIZE > 0 && S::SIZE.is_multiple_of(8),
                "pointwise tensors require complete eight-element vectors"
            );
        }
        let mut result = HbmTensor::<D, Chips, m![{ Width<S> }]>::new();
        for index in 0..active {
            let tiles = self.land(device, index);
            op.forward(&mut *device, &tiles).view().to_hbm_view(
                &mut device.tdma,
                result.view_mut().tile::<Index<S, G>, 1, Slot<S, G>>(index),
            );
        }
        // SAFETY: flattening preserves the contiguous element order; no cells are padded.
        unsafe { result.reshape() }
    }

    /// The original reduction order: intra-slice, inter-slice, then a separate pass over the
    /// windows' FP32 sums. No HBM tensor of unrounded differences is introduced.
    fn sum<D: Float, Chips: M>(
        self,
        device: &mut Device,
        formula: impl FnMut(&mut Device, &Self::Tile) -> Rows<D, G, Chips>,
    ) -> Total<Chips> {
        self.sum_windows(device, S::SIZE / Window::<G>::SIZE, formula)
    }

    /// Reduce initialized whole windows; callers establish the bound before entering this
    /// private implementation. Inactive scalar slots are zero, without fetching source rows.
    fn sum_windows<D: Float, Chips: M>(
        self,
        device: &mut Device,
        active: usize,
        mut formula: impl FnMut(&mut Device, &Self::Tile) -> Rows<D, G, Chips>,
    ) -> Total<Chips> {
        const {
            assert!(S::SIZE.is_multiple_of(Window::<G>::SIZE));
            assert!(
                S::SIZE > 0 && S::SIZE.is_multiple_of(8),
                "pointwise sums require complete eight-element vectors"
            );
        }
        let mut sums = HbmTensor::<f32, Chips, m![{ Width<S> } / ({ Extent<G> }), 1 # 8]>::new();
        let windows = const { S::SIZE / Window::<G>::SIZE };
        if active < windows {
            let mut zero = LocalTotal::<Chips>::new();
            zero.view_mut().memset(0.0, &mut device.sub);
            for offset in 0..windows - active {
                zero.view().to_hbm_view(
                    &mut device.tdma,
                    sums.view_mut().tile::<Index<S, G>, 1, m![
                        { Width<S> } / ({ Extent<G> }) = 1 #{!} ({ Width<S> } / ({ Extent<G> })),
                        1 # 8
                    ]>(active + offset),
                );
            }
        }
        for index in 0..active {
            let tiles = self.land(device, index);
            let tile = formula(&mut *device, &tiles);
            let sum: LocalTotal<Chips> = device
                .main
                .begin(tile.view())
                .fetch::<m![{ Width<G> } % ({ Segment<G> }) / 8], m![{ Width<G> } % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Width<G> } % ({ Segment<G> }) / 8], m![{ Width<G> } % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Width<G> } % ({ Segment<G> }) / 4], m![{ Width<G> } % 4]>()
                .vector_intra_slice_reduce::<Flat<G>, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<m![1 # 256], m![1]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            sum.view().to_hbm_view(
                &mut device.tdma,
                sums.view_mut().tile::<Index<S, G>, 1, m![
                    { Width<S> } / ({ Extent<G> }) = 1 #{!} ({ Width<S> } / ({ Extent<G> })),
                    1 # 8
                ]>(index),
            );
        }
        let sums: DmTensor<
            f32,
            Chips,
            m![1 # 2],
            m![1 # 256],
            m![{ Width<S> } / ({ Extent<G> }), 1 # 8],
        > = sums.to_dm(&mut device.tdma);
        let total: LocalTotal<Chips> = device
            .main
            .begin(sums.view())
            .fetch::<Index<S, G>, m![1 # 8]>()
            .collect::<Index<S, G>, m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_intra_slice_reduce::<Flat<S>, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        total.to_hbm(&mut device.tdma)
    }
}

impl<D: MaterializableScalar, S: M + 'static, G: M + 'static, Chips: M> Windowed<S, G>
    for &HbmTensor<D, Chips, S>
{
    type Tile = Rows<D, G, Chips>;

    fn land(&self, device: &mut Device, index: usize) -> Self::Tile {
        // SAFETY: flattening preserves contiguous element order; callers require full vectors.
        let window = unsafe { self.view().reshape::<Chips, m![{ Width<S> }]>() }
            .tile::<Index<S, G>, 1, Packet<S, G>>(index);
        window.to_dm(&mut device.tdma)
    }
}

impl<S: M + 'static, G: M + 'static, A: Windowed<S, G>, B: Windowed<S, G>> Windowed<S, G>
    for (A, B)
{
    type Tile = (A::Tile, B::Tile);

    fn land(&self, device: &mut Device, index: usize) -> Self::Tile {
        (self.0.land(device, index), self.1.land(device, index))
    }
}

#[cfg(test)]
mod tests {
    use super::{Sum, Total};
    use crate::ops::pointwise::abs::Abs;
    use crate::ops::pointwise::subtract::Subtract;
    use crate::ops::pointwise::subtract::tests::{
        Chips, Host, Part, ReferenceFeatures, ReferenceRows, rounding_terms, terms,
    };
    use crate::tensor::axis::{Eight, Single};
    use crate::tensor::parallel::Shard;
    use crate::{Module, Then};
    use furiosa_opt_std::{prelude::*, runtime::launch};

    type Distance = (Sum<Then<Subtract, Abs>>, Sum<Abs>);
    axes![Tail = 520];

    #[device(chip = 4)]
    fn magnitudes<R: AxisName, F: AxisName>(
        device: &mut Device,
        distance: &Distance,
        (left, right): (&Part<R, F>, &Part<R, F>),
    ) -> (Total<Chips>, Total<Chips>) {
        distance.forward(device, (), &mut (((), ()), ()), ((left, right), right))
    }

    async fn matches<R: AxisName, F: AxisName>((left, right): (Vec<bf16>, Vec<bf16>)) {
        let mut device = Device::new(magnitudes.topology()).unwrap();
        let part = left.len() / 4;
        let expected: Vec<(f64, f64)> = (0..4)
            .map(|chip| {
                left.iter().zip(&right).skip(chip * part).take(part).fold(
                    (0.0_f64, 0.0_f64),
                    |(distance, norm), (left, right)| {
                        let difference = bf16::from_f32(left.to_f32() - right.to_f32());
                        (
                            distance + f64::from(difference.to_f32().abs()),
                            norm + f64::from(right.to_f32().abs()),
                        )
                    },
                )
            })
            .collect();
        let left: Part<R, F> = Host::<R, F>::from_vec(left)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let right: Part<R, F> = Host::<R, F>::from_vec(right)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let distance = (Sum(Then(Subtract, Abs)), Sum(Abs));
        let (distances, norms) = launch(magnitudes, (&mut device, &distance, (&left, &right)))
            .await
            .unwrap();
        let read = async |device: &mut Device, total: Total<Chips>| {
            total
                .to_host::<m![{ Symbol<Shard<4>> }, 1 # 8]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec()
                .chunks_exact(8)
                .map(|chip| f64::from(chip[0]))
                .collect::<Vec<_>>()
        };
        let (distances, norms) = (
            read(&mut device, distances).await,
            read(&mut device, norms).await,
        );
        for chip in 0..4 {
            let (distance, norm) = expected[chip];
            let close = |got: f64, want: f64| (got - want).abs() <= want * 1e-5;
            assert!(
                close(distances[chip], distance) && close(norms[chip], norm),
                "{} x {}, chip {chip}: ({}, {}) against ({distance}, {norm})",
                R::SIZE,
                F::SIZE,
                distances[chip],
                norms[chip]
            );
        }
    }

    /// Every chip's rounded-difference L1 distance and L1 norm agree with independent FP64 sums
    /// within the original FP32 accumulation tolerance. Only lane zero is part of each total's
    /// contract. Cover the repro's three windows, a single vector, and a short vector-aligned tail.
    #[tokio::test(flavor = "current_thread")]
    async fn sums_each_chips_magnitudes() {
        let _lease = crate::tests::DEVICE.lock().await;
        matches::<ReferenceRows, ReferenceFeatures>(terms(
            ReferenceRows::SIZE * ReferenceFeatures::SIZE,
        ))
        .await;
        matches::<Single, Eight>(rounding_terms(Eight::SIZE)).await;
        matches::<Single, Tail>(rounding_terms(Tail::SIZE)).await;
    }
}
