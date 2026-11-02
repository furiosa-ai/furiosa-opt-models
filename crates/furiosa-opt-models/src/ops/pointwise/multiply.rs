//! Elementwise products: of two tensors, of a tensor and a scalar, and of rows and per-row scalars.

use super::{
    Batch, BatchMut, Cell, Cells, Chunks, Extent, Factors, Float, Kept, Rows, Segment, Slices,
    SpreadPacket, SpreadTime, Walk, Width, Window,
};
use crate::Op;
use crate::load::Load;
use crate::ops::normalization::{Chunk, Cluster, FeatureSlices, Spread};
use crate::ops::vector::Zip;
use crate::tensor::{axis::Single, live::Live, scalar};
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;
use std::num::NonZeroUsize;

/// Multiply a single-chip tensor by an FP32 scalar; the kernel requires eight padded cells.
#[derive(DeviceSend, Load)]
pub struct Scale<Factor = HbmTensor<f32, m![1], m![1]>>(pub Factor);

impl<D: Float, S: M + 'static, Chips: M + 'static> Op<&HbmTensor<D, Chips, S>> for Scale {
    type Output = HbmTensor<D, Chips, S>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<D, Chips, S>) -> Self::Output {
        todo!("NPU scaling by an unpadded scalar")
    }
}

/// Elementwise product of tensors with identical chip and element mappings.
#[derive(Default, DeviceSend, Load)]
pub struct Multiply;

impl<D: Float, S: M + 'static, Chips: M> Op<(&HbmTensor<D, Chips, S>, &HbmTensor<D, Chips, S>)>
    for Multiply
{
    type Output = HbmTensor<D, Chips, S>;

    fn forward(
        &self,
        device: &mut Device,
        (left, right): (&HbmTensor<D, Chips, S>, &HbmTensor<D, Chips, S>),
    ) -> Self::Output {
        Window::<S>::combine::<_, true, Chips>(device, left, right)
    }
}

/// Multiply the queries of a positioned query/key pair by the step's FP32 scalar; the keys,
/// and so the cache, keep their values.
#[derive(Clone, Copy, Default, DeviceSend)]
pub struct QueryScale;

impl<'s, D: Float, S: M + 'static, K, Chips: M + 'static>
    Op<(
        (HbmTensor<D, Chips, S>, K),
        &'s HbmTensor<f32, Chips, m![Single, 1 # 8]>,
    )> for QueryScale
{
    type Output = (HbmTensor<D, Chips, S>, K);

    fn forward(
        &self,
        device: &mut Device,
        ((query, key), factor): (
            (HbmTensor<D, Chips, S>, K),
            &'s HbmTensor<f32, Chips, m![Single, 1 # 8]>,
        ),
    ) -> Self::Output {
        (Scale::rows(device, &query, factor), key)
    }
}

/// The query scale `1 + beta * ln(1 + floor(position / original))` of a token, which grows
/// once per `original` positions past the context the model was trained on.
#[derive(Clone, Copy, Debug)]
pub struct Temperature {
    /// Coefficient of the logarithm.
    pub beta: f32,
    /// Positions per step: the context length before extension.
    pub original: NonZeroUsize,
}

impl Temperature {
    /// The scale at a token's sequence position (not its cache slot), computed in FP32 and
    /// rounded to the BF16 the queries are multiplied by.
    pub fn at(&self, position: usize) -> bf16 {
        let steps = (position as f32 / self.original.get() as f32).floor();
        bf16::from_f32(1.0 + self.beta * (1.0 + steps).ln())
    }
}

/// Multiply each contiguous row by its corresponding scalar.
#[derive(Default, DeviceSend)]
pub struct RowScale;

impl<D: Scalar, Rows: AxisName, Width: AxisName, Chips: M + 'static>
    Op<(
        &HbmTensor<D, Chips, m![Rows, Width]>,
        &HbmTensor<D, Chips, m![Rows]>,
    )> for RowScale
{
    type Output = HbmTensor<D, Chips, m![Rows, Width]>;

    fn forward(
        &self,
        _: &mut Device,
        _: (
            &HbmTensor<D, Chips, m![Rows, Width]>,
            &HbmTensor<D, Chips, m![Rows]>,
        ),
    ) -> Self::Output {
        todo!("NPU multiplication by one scalar per row")
    }
}

impl<D: Float, S: M + 'static, Chips: M + 'static> Op<&HbmTensor<D, Chips, S>>
    for Scale<HbmTensor<f32, Chips, m![1 # 8]>>
{
    type Output = HbmTensor<D, Chips, S>;

    fn forward(&self, device: &mut Device, input: &HbmTensor<D, Chips, S>) -> Self::Output {
        Self::rows(device, input, &self.0)
    }
}

/// `input` times the FP32 scalar in the first cell of `factor`, rounded to `D` once.
impl<F: M, Chips: M + 'static> Scale<HbmTensor<f32, Chips, F>> {
    pub(in crate::ops) fn rows<D: Float, S: M + 'static>(
        device: &mut Device,
        input: &HbmTensor<D, Chips, S>,
        factor: &HbmTensor<f32, Chips, F>,
    ) -> HbmTensor<D, Chips, S> {
        let mut result = HbmTensor::<D, Chips, m![{ Width<S> }]>::new();
        for index in 0..S::SIZE / Window::<S>::SIZE {
            let input: Rows<D, S, Chips> =
                unsafe { input.view().reshape::<Chips, m![{ Width<S> }]>() }
                    .tile::<m![{ Width<S> } / ({ Extent<S> })], 1, Batch<S>>(index)
                    .to_dm(&mut device.tdma);
            let scale: DmTensor<f32, Chips, m![1 # 2], Slices<S>, m![1 # 8]> =
                factor.to_dm(&mut device.tdma);
            let scale = scalar::register(device, &scale);
            let output: Rows<f32, S, Chips> = device
            .main
            .begin(input.view())
            .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Width<S> } % ({ Segment<S> }) / 4], m![{ Width<S> } % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .vector_final()
            .commit_trim::<m![{ Width<S> } % 8]>()
            .commit();
            D::narrow(device, output).view().to_hbm_view(
                &mut device.tdma,
                result
                    .view_mut()
                    .tile::<m![{ Width<S> } / ({ Extent<S> })], 1, BatchMut<S>>(index),
            );
        }
        // SAFETY: flattening preserves the contiguous element order.
        unsafe { result.reshape() }
    }
}

/// Every row of one launch in feature slices times the FP32 scalar in the first cell of the factor,
/// rounded once to BF16; the result keeps their live rows.
impl<'x, W: AxisName, R: AxisName, Chips: M + 'static, F> Op<&'x Live<Spread<W, R, Chips>, R>>
    for Scale<F>
where
    F: Borrow<HbmTensor<f32, Chips, m![1 # 8]>>,
{
    type Output = Live<Spread<W, R, Chips>, R>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<Spread<W, R, Chips>, R>,
    ) -> Self::Output {
        let factor: DmTensor<f32, Chips, Cluster, FeatureSlices<W>, m![1 # 8]> =
            self.0.borrow().to_dm(&mut device.tdma);
        let factor = scalar::register(device, &factor);
        let value = device
            .main
            .begin(input.value().view())
            .fetch::<SpreadTime<W, R>, SpreadPacket<W>>()
            .fetch_cast::<f32>()
            .collect::<SpreadTime<W, R>, SpreadPacket<W>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R, W % ({ Symbol<Chunk<W>> }) / 4], m![W % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factor)
            .vector_widen_concat::<SpreadTime<W, R>, SpreadPacket<W>>()
            .vector_final()
            .cast::<bf16, m![W % 8 # 16]>()
            .commit_trim::<m![W % 8]>()
            .commit();
        input.with(value)
    }
}

/// The FP32 elementwise product of one launch's rows in feature slices, rounded once to BF16, both
/// factors streamed interleaved so no row count fills the registers; it keeps their live rows.
impl<'x, 'y, W: AxisName, R: AxisName, Chips: M + 'static>
    Op<(
        &'x Live<Spread<W, R, Chips>, R>,
        &'y Live<Spread<W, R, Chips>, R>,
    )> for Multiply
{
    type Output = Live<Spread<W, R, Chips>, R>;

    fn forward(
        &self,
        device: &mut Device,
        (left, right): (
            &'x Live<Spread<W, R, Chips>, R>,
            &'y Live<Spread<W, R, Chips>, R>,
        ),
    ) -> Self::Output {
        let factors = (left.value().view(), right.value().view());
        left.with(product::<Chunks<W, R>, _, _, _, _>(device, factors))
    }
}

/// The product of two launches' rows their producers kept in one layout, cell by cell as in feature
/// slices; it keeps their layout and live rows.
impl<'x, 'y, R: AxisName, Chips: M + 'static, C: M, S: M, E: M + 'static>
    Op<(
        &'x Live<Kept<DmTensor<bf16, Chips, C, S, E>>, R>,
        &'y Live<Kept<DmTensor<bf16, Chips, C, S, E>>, R>,
    )> for Multiply
{
    type Output = Live<Kept<DmTensor<bf16, Chips, C, S, E>>, R>;

    fn forward(
        &self,
        device: &mut Device,
        (left, right): (
            &'x Live<Kept<DmTensor<bf16, Chips, C, S, E>>, R>,
            &'y Live<Kept<DmTensor<bf16, Chips, C, S, E>>, R>,
        ),
    ) -> Self::Output {
        // SAFETY: the cells keep their order; only their names merge into one axis.
        let factors = unsafe {
            (
                left.value().0.view().reshape::<Chips, C, S, Cell<E>>(),
                right.value().0.view().reshape::<Chips, C, S, Cell<E>>(),
            )
        };
        let value = product::<Cells<E>, _, _, _, _>(device, factors);
        // SAFETY: as above, back.
        left.with(Kept(unsafe { value.reshape::<Chips, C, S, E>() }))
    }
}

/// The FP32 product of each pair of BF16 cells walked as `K` says, the factors read interleaved,
/// rounded once to BF16.
fn product<K: Walk, Chips: M + 'static, C: M, S: M, E: M>(
    device: &mut Device,
    (left, right): (
        DmTensorView<'_, bf16, Chips, C, S, E>,
        DmTensorView<'_, bf16, Chips, C, S, E>,
    ),
) -> DmTensor<bf16, Chips, C, S, E> {
    device
        .main
        .begin_interleaved::<Factors, _, _, _, _, _>(left, right)
        .fetch::<K::Pairs, K::Packet>()
        .fetch_cast::<f32>()
        .collect::<K::Pairs, K::Packet>()
        .vector_init()
        .vector_zip::<Factors, K::Time, K::Narrow, K::Half>(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_final()
        .cast::<bf16, K::Padded>()
        .commit_trim::<K::Packet>()
        .commit()
}

#[cfg(test)]
mod tests {
    use super::Temperature;
    use std::num::NonZeroUsize;

    /// The scale steps at multiples of the original context and is BF16 in between.
    #[test]
    fn steps_at_each_original_context() {
        let temperature = Temperature {
            beta: 0.1,
            original: NonZeroUsize::new(16384).unwrap(),
        };
        let scales = [0, 16383, 16384, 32767, 32768].map(|at| temperature.at(at).to_f32());
        assert_eq!(scales, [1.0, 1.0, 1.0703125, 1.0703125, 1.109375]);
    }
}
