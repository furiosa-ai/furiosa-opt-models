//! O projection partitioned across clusters by input features. Up to 32 live rows
//! keep BF16 activations in TRF and decode packed codes into 16-input dot products.
//! Larger launches restore one BF16 weight tile per slice and reuse it across
//! 128-row waves. Cluster partials add before tensor scaling and BF16 rounding.
use super::block;
use crate::Op;
use crate::modules::generic::collection::View;
use crate::ops::linear::nvfp4::tiled::Weights;
use crate::ops::linear::nvfp4::tiled::prefetch::Points;
use crate::ops::normalization::span::Owned;
use crate::tensor::{live::Live, parallel::Chip};
use furiosa_opt_std::prelude::*;

#[derive(DeviceSend, crate::load::Load)]
pub struct Output<I: AxisName, O: AxisName> {
    #[load(rename = "")]
    pub weights: Weights<I, O, Stored<I, O>, StoredScales<I, O>>,
}
pub type Stored<I, O> = m![I / (I / 2), O / 256, I % (I / 2) / 128, O % 256, I % 128];

pub type StoredScales<I, O> = m![
    I / (I / 2),
    O / 256,
    I % (I / 2) / 128,
    O % 256,
    I % 128 / 16
];
type Cluster<I> = m![I / (I / 2)];
type Selected<I> = m![I / (I / 2) = 1 # 2];
type Slices<I, O> = m![O / 256, I % (I / 2) / 128];
type Reduced<O> = m![O / 16];
type Weight<I, O, D> = DmTensor<D, Chip, Cluster<I>, Slices<I, O>, m![O % 256, I % 128]>;
type Scales<I, O> = DmTensor<f8e4m3, Chip, Cluster<I>, Slices<I, O>, m![O % 256, I % 128 / 16]>;
type Operand<I, O, R, const P: usize> =
    TrfTensor<bf16, Chip, Cluster<I>, Slices<I, O>, m![R % 8], m![R % { P } / 8, I % 128]>;
type WeightOperand<I, O> =
    TrfTensor<bf16, Chip, Cluster<I>, Slices<I, O>, m![O % 8], m![O % 256 / 8, I % 128]>;
type BlockScales<I, O> = VrfTensor<f32, Chip, Cluster<I>, Slices<I, O>, m![O % 256, I % 128 / 16]>;
type Partial<I, O, R, const P: usize> =
    DmTensor<f32, Chip, Cluster<I>, Reduced<O>, m![R % { P }, O % 16]>;
type OwnerRows<R, const P: usize> = m![R % { P } / 4];
type Factor<I, O> = VrfTensor<f32, Chip, Selected<I>, Reduced<O>, m![1 # 8]>;
axes![Term = 2];
type Rows<R, W> = HbmTensor<bf16, Chip, m![R, W]>;
pub(crate) type Staged<I, O> = (Weight<I, O, f4e2m1>, Scales<I, O>);

/// Stage O's packed weights at the attention's final history transfer.
impl<I: AxisName, O: AxisName> Op<&mut Staged<I, O>> for Output<I, O> {
    type Output = ();
    fn forward(&self, device: &mut Device, staged: &mut Staged<I, O>) {
        self.stage_into(device, staged);
    }
}

impl<I: AxisName, O: AxisName> Output<I, O> {
    #[cfg(test)]
    pub fn reorder<const PACK: usize, T: Copy>(values: &[T]) -> Vec<T> {
        assert_eq!(values.len(), O::SIZE * I::SIZE / PACK);
        let mut result = Vec::with_capacity(values.len());
        for cluster in 0..2 {
            for owner in 0..O::SIZE / 256 {
                for group in 0..I::SIZE / 2 / 128 {
                    for out in owner * 256..(owner + 1) * 256 {
                        let at = (out * I::SIZE + cluster * (I::SIZE / 2) + group * 128) / PACK;
                        result.extend_from_slice(&values[at..at + 128 / PACK]);
                    }
                }
            }
        }
        result
    }

    fn block_scales(device: &mut Device, weight: &Scales<I, O>) -> BlockScales<I, O> {
        device
            .sub
            .begin(weight.view())
            .fetch::<m![O % 256], m![I % 128 / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![O % 256], m![I % 128 / 16]>()
            .to_vrf()
    }

    fn project<R: AxisName, const P: usize, E: M, F: M>(
        device: &mut Device,
        weight: &Weight<I, O, f4e2m1>,
        scales: &BlockScales<I, O>,
        input: DmTensorView<'_, bf16, Chip, Cluster<I>, Slices<I, O>, E>,
        output: DmTensorViewMut<'_, f32, Chip, Cluster<I>, Reduced<O>, F>,
    ) {
        let input: Operand<I, O, R, P> = device
            .sub
            .begin(input)
            .fetch::<m![R % 8, R % { P } / 8, I % 128 / 16], m![I % 16]>()
            .collect::<m![R % 8, R % { P } / 8, I % 128 / 16], m![I % 16]>()
            .to_trf();
        device
            .main
            .begin(weight.view())
            .fetch::<m![O % 256, I % 128 / 64], m![I % 64]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![O % 256, I % 128 / 16], m![I % 16]>()
            .contract_outer::<m![O % 256, I % 128 / 16, R % { P } / 8], m![I % 16], _, _, _>(&input)
            .contract_packet::<m![1]>()
            .contract_time::<m![O % 256, I % 128 / 16, R % { P } / 8]>()
            .contract_lane::<m![O % 256, I % 128 / 16, R % { P } / 8], m![R % 8]>(
                LaneMode::Interleaved,
            )
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![O % 256, I % 128 / 16, R % { P } / 4], m![R % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scales)
            .vector_intra_slice_reduce::<I, m![O % 256, R % { P } / 4], m![R % 4]>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_widen_concat::<m![O % 256, R % { P } / 8], m![R % 8]>()
            .vector_inter_slice_reduce::<Reduced<O>, m![O % 16, R % { P } / 8]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            // FP32 transpose emits two output columns per padded flit.
            .transpose::<m![O % 16 / 2, R % { P }], m![O % 2 # 8]>()
            .commit_trim::<m![O % 2]>()
            .commit_view(output);
    }

    fn pass<R: AxisName, All: AxisName, const PASS: usize, N: Points<Device>>(
        &self,
        device: &mut Device,
        weight: &Staged<I, O>,
        input: &Rows<All, I>,
        output: &mut Owned<O, R>,
        at: usize,
        start: usize,
        next: N,
    ) -> N::Next {
        let window = input
            .view()
            .tile::<m![All], PASS, m![All = {PASS} # All, I]>(start + at * PASS);
        // SAFETY: rename this contiguous pass's row axis, keeping the source stride.
        let window = unsafe { window.reshape::<Chip, m![R % {PASS} # All, I]>() };
        let input: DmTensor<bf16, Chip, Cluster<I>, Slices<I, O>, m![R % { PASS }, I % 128]> =
            window.to_dm(&mut device.tdma);
        let scales = Self::block_scales(device, &weight.1);
        let mut partial: Partial<I, O, R, PASS> = DmTensor::new();
        Self::project::<R, PASS, _, _>(
            device,
            &weight.0,
            &scales,
            input.view(),
            partial.view_mut(),
        );
        // Request following operands before enqueueing the output transfer. Otherwise
        // that DMA waits for this contraction and holds the prefetch behind it.
        let next = next.at(device);
        let factor =
            block::scalar::<Selected<I>, Reduced<O>>(device, &self.weights.contract.output);
        Self::handoff::<R, PASS>(device, &partial, &factor, output, at);
        next
    }

    // Four row owners share BF16 input blocks: 128 rows for full waves,
    // 32 rows for tails. Weight TRF and the output factor are reused.
    fn compact_pass<R: AxisName, All: AxisName, const P: usize, N: Points<Device>>(
        &self,
        device: &mut Device,
        weight: &WeightOperand<I, O>,
        factor: &Factor<I, O>,
        input: &Rows<All, I>,
        output: &mut Owned<O, R>,
        wave: usize,
        start: usize,
        next: N,
    ) -> N::Next {
        let window = input
            .view()
            .tile::<m![All], P, m![All = {P} # All, I]>(start + wave * P);
        // SAFETY: rename the selected row window while retaining its source stride.
        let window = unsafe { window.reshape::<Chip, m![R % {P} # All, I]>() };
        let input: DmTensor<
            bf16,
            Chip,
            Cluster<I>,
            m![O / 1024, R % { P } / ({OwnerRows<R, P>}), I % (I / 2) / 128],
            m![R % ({OwnerRows<R, P>}), I % 128],
        > = window.to_dm(&mut device.tdma);
        let partial: Partial<I, O, R, P> = device
            .main.begin(input.view())
            .fetch::<m![R % ({OwnerRows<R, P>}), I % 128 / 32], m![I % 32]>()
            .switch::<Slices<I, O>, m![R % ({OwnerRows<R, P>}), I % 128 / 32, R % {P} / ({OwnerRows<R, P>})]>(
                SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
            .collect::<m![R % ({OwnerRows<R, P>}), I % 128 / 32, R % {P} / ({OwnerRows<R, P>}), I % 32 / 16], m![I % 16]>()
            .contract_outer::<m![R % ({OwnerRows<R, P>}), I % 128 / 32, R % {P} / ({OwnerRows<R, P>}), O % 256 / 8], m![I % 32], _, _, _>(weight)
            .contract_packet::<m![1]>()
            .contract_time::<m![R % ({OwnerRows<R, P>}), R % {P} / ({OwnerRows<R, P>}), O % 256 / 8]>()
            .contract_lane::<m![R % ({OwnerRows<R, P>}), R % {P} / ({OwnerRows<R, P>}), O % 256 / 8], m![O % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Reduced<O>, m![R % ({OwnerRows<R, P>}), R % {P} / ({OwnerRows<R, P>}), O % 16 / 8]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .commit_trim::<m![O % 8]>()
            .commit();
        let next = next.at(device);
        Self::handoff::<R, P>(device, &partial, factor, output, wave);
        next
    }
}

impl<I: AxisName, O: AxisName> Output<I, O> {
    fn handoff<R: AxisName, const P: usize>(
        device: &mut Device,
        partial: &Partial<I, O, R, P>,
        factor: &Factor<I, O>,
        output: &mut Owned<O, R>,
        wave: usize,
    ) {
        if wave * P < R::SIZE / 2 {
            Self::combine::<R, P, 0>(device, partial, factor, output, wave);
        } else {
            Self::combine::<R, P, 1>(device, partial, factor, output, wave);
        }
    }

    fn combine<R: AxisName, const P: usize, const OWNER: usize>(
        device: &mut Device,
        own: &Partial<I, O, R, P>,
        factor: &Factor<I, O>,
        output: &mut Owned<O, R>,
        wave: usize,
    ) {
        let mut received: Partial<I, O, R, P> = DmTensor::new();
        // Only OWNER consumes this contiguous pass. Copy the remote input-half
        // partial into OWNER, leaving the local partial in its original buffer.
        own.view()
            .cluster_tile::<Cluster<I>, 1, Selected<I>>(1 - OWNER)
            .to_dm_view(
                &mut device.tdma,
                received
                    .view_mut()
                    .cluster_tile::<Cluster<I>, 1, m![I / (I / 2) = 1 #{!} 2]>(OWNER),
            );
        let own = own.view().cluster_tile::<Cluster<I>, 1, Selected<I>>(OWNER);
        let received = received
            .view()
            .cluster_tile::<Cluster<I>, 1, Selected<I>>(OWNER);
        device.main.begin_interleaved::<Term, _, _, _, _, _>(own, received)
            .fetch::<m![R % {P}, Term, O % 16 / 8], m![O % 8]>()
            .collect::<m![R % {P}, Term, O % 16 / 8], m![O % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, m![R % {P}, O % 16 / 8]>()
            .vector_narrow_split::<m![R % {P}, O % 16 / 4], m![O % 4]>()
            .vector_fp_zip(FpBinaryOp::AddF)
            // Scaling shares the mandatory cluster-combine pass. Moving it into
            // each partial would not remove a pass and would round before the add.
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), factor)
            .vector_widen_concat::<m![R % {P}, O % 16 / 8], m![O % 8]>()
            .vector_final()
            .cast::<bf16, m![O % 8 # 16]>()
            .commit_trim::<m![O % 8]>()
            // SAFETY: only rename the selected physical cluster and its padding
            // marker. The row offset, live elements, and allocation stride stay fixed.
            .commit_view(unsafe { output.view_mut()
                .cluster_tile::<m![R / (R / 2)], 1, m![R / (R / 2) = 1 #{!} 2]>(OWNER)
                .tile::<m![R % (R / 2) / {P}], 1,
                    m![R % (R / 2) / {P} = 1 #{!} (R / 2 / {P}), R % {P}, O % 16]>(wave % (R::SIZE / 2 / P))
                .reshape::<Chip, Selected<I>, Reduced<O>,
                    m![R % (R / 2) / {P} = 1 #{!} (R / 2 / {P}), R % {P}, O % 16]>() });
    }

    fn stage_into(&self, device: &mut Device, staged: &mut Staged<I, O>) {
        self.weights
            .weight
            .view()
            .to_dm_view(&mut device.tdma, staged.0.view_mut());
        self.weights
            .weight_scale
            .tensor
            .view()
            .to_dm_view(&mut device.tdma, staged.1.view_mut());
    }

    pub(crate) fn prepare(&self, device: &mut Device) -> Staged<I, O> {
        let codes: Weight<I, O, f4e2m1> = self.weights.weight.to_dm(&mut device.tdma);
        let scales: Scales<I, O> = self.weights.weight_scale.tensor.to_dm(&mut device.tdma);
        (codes, scales)
    }

    pub(crate) fn prepared_window<R: AxisName, All: AxisName>(
        &self,
        device: &mut Device,
        live: Live<(), R>,
        input: &Rows<All, I>,
        start: usize,
        weight: &Staged<I, O>,
    ) -> Owned<O, R> {
        self.prepared_window_with_next(device, live, input, start, weight, ())
            .0
    }

    pub(crate) fn prepared_window_with_next<R: AxisName, All: AxisName, N: Points<Device>>(
        &self,
        device: &mut Device,
        live: Live<(), R>,
        input: &Rows<All, I>,
        start: usize,
        weight: &Staged<I, O>,
        next: N,
    ) -> (Owned<O, R>, N::Output) {
        const {
            assert!(I::SIZE == 4096 && O::SIZE == 4096 && R::SIZE == 512);
        }
        let mut output = Owned::new();
        // Small launches need one live 8/16/32-row pass. Its tensor factor loads
        // after the first contraction and the following-operand prefetch hook.
        let next = if live.rows() <= 8 {
            self.pass::<R, All, 8, _>(device, weight, input, &mut output, 0, start, next)
        } else if live.rows() <= 16 {
            self.pass::<R, All, 16, _>(device, weight, input, &mut output, 0, start, next)
        } else if live.rows() <= 32 {
            self.pass::<R, All, 32, _>(device, weight, input, &mut output, 0, start, next)
        } else {
            // Restore once for all live 128-row waves. Attention's prefetch hook
            // stages packed DM only; the weight TRF becomes resident here.
            let weight = block::restore_view(device, weight.0.view(), weight.1.view());
            let resident: WeightOperand<I, O> = device
                .sub
                .begin(weight.view())
                .fetch::<m![O % 8, O % 256 / 8, I % 128 / 16], m![I % 16]>()
                .collect::<m![O % 8, O % 256 / 8, I % 128 / 16], m![I % 16]>()
                .to_trf();
            let factor =
                block::scalar::<Selected<I>, Reduced<O>>(device, &self.weights.contract.output);
            // Round coverage to 32 rows, then pack four such groups into a
            // 128-row pass, matching Qwen's granularity without extra launches.
            let groups = live.rows().div_ceil(32);
            let full = groups / 4;
            let next = if full > 0 {
                self.compact_pass::<R, All, 128, _>(
                    device,
                    &resident,
                    &factor,
                    input,
                    &mut output,
                    0,
                    start,
                    next,
                )
            } else {
                self.compact_pass::<R, All, 32, _>(
                    device,
                    &resident,
                    &factor,
                    input,
                    &mut output,
                    0,
                    start,
                    next,
                )
            };
            // Full ranges from zero to four; this is max(full - 1, 0),
            // with a zero-based loop for the device compiler.
            for wave in 0..3 * full / 4 {
                self.compact_pass::<R, All, 128, _>(
                    device,
                    &resident,
                    &factor,
                    input,
                    &mut output,
                    wave + 1,
                    start,
                    (),
                );
            }
            for tail in 0..groups % 4 {
                // The first 32-row pass already ran when there was no full wave.
                if full > 0 || tail > 0 {
                    self.compact_pass::<R, All, 32, _>(
                        device,
                        &resident,
                        &factor,
                        input,
                        &mut output,
                        full * 4 + tail,
                        start,
                        (),
                    );
                }
            }
            next
        };
        (output, next.finish(device))
    }

    pub fn owned<R: AxisName>(
        &self,
        device: &mut Device,
        live: Live<(), R>,
        input: &Rows<R, I>,
    ) -> Owned<O, R> {
        let weight = self.prepare(device);
        self.prepared_window(device, live, input, 0, &weight)
    }
}

impl<I: AxisName, O: AxisName, R: AxisName> Op<(Live<(), R>, &Rows<R, I>)> for Output<I, O> {
    type Output = Rows<R, O>;
    fn forward(
        &self,
        device: &mut Device,
        (live, input): (Live<(), R>, &Rows<R, I>),
    ) -> Self::Output {
        self.owned(device, live, input).to_hbm(&mut device.tdma)
    }
}

/// A live window of a larger launch, preserving its starting row and source stride.
impl<I: AxisName, O: AxisName, R: AxisName, All: AxisName> Op<(Live<(), R>, &Rows<All, I>, usize)>
    for Output<I, O>
{
    type Output = Rows<R, O>;
    fn forward(
        &self,
        device: &mut Device,
        (live, input, start): (Live<(), R>, &Rows<All, I>, usize),
    ) -> Self::Output {
        let weight = self.prepare(device);
        self.prepared_window(device, live, input, start, &weight)
            .to_hbm(&mut device.tdma)
    }
}

impl<I: AxisName, O: AxisName, const N: usize> View<N> for Output<I, O> {
    type Storage = <Weights<I, O, Stored<I, O>, StoredScales<I, O>> as View<N>>::Storage;
    type Layer = Self;
    fn view(storage: &Self::Storage, layer: usize) -> Self {
        Self {
            weights: <Weights<I, O, Stored<I, O>, StoredScales<I, O>> as View<N>>::view(
                storage, layer,
            ),
        }
    }
    fn stack(layers: [Self; N], cx: &mut crate::load::Context<'_>) -> Self::Storage {
        <Weights<I, O, Stored<I, O>, StoredScales<I, O>> as View<N>>::stack(
            layers.map(|layer| layer.weights),
            cx,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::linear::nvfp4::tiled::Blocks;
    use crate::ops::linear::nvfp4::tiled::output::Output as OutputProjection;
    use crate::ops::linear::nvfp4::tiled::tests as fixtures;
    use crate::tensor::form::BlockScaled;
    use crate::{
        Op,
        tensor::{live::Live, scalar::Value as Count},
    };
    use fixtures::*;
    use furiosa_opt_std::{prelude::*, runtime::launch};

    axes![Reach = 1024];

    #[device(chip = 1)]
    fn project_output(
        device: &mut Device,
        projection: &OutputProjection<Hidden, OutputWidth>,
        count: &Count,
        input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
    ) -> HbmTensor<bf16, m![1], m![Rows, OutputWidth]> {
        projection.forward(device, (Live::<(), Rows>::read(count), input))
    }

    #[device(chip = 1)]
    fn project_window(
        device: &mut Device,
        projection: &OutputProjection<Hidden, OutputWidth>,
        count: &Count,
        input: &HbmTensor<bf16, m![1], m![Reach, Hidden]>,
    ) -> HbmTensor<bf16, m![1], m![Rows, OutputWidth]> {
        projection.forward(device, (Live::<(), Rows>::read(count), input, 512))
    }
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "opens a device"]
    async fn projects_live_rows_and_selects_the_requested_span() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(project_output.topology()).unwrap();
        let (weight, weight_scale, scales) = weights::<Hidden, OutputWidth, _, _>(
            &mut device,
            OutputProjection::<Hidden, OutputWidth>::reorder::<1, _>,
            OutputProjection::<Hidden, OutputWidth>::reorder::<16, _>,
        )
        .await;
        let projection = OutputProjection {
            weights: BlockScaled::new(weight, Blocks::new(weight_scale), scales),
        };
        let input = input::<Hidden>(&mut device).await;
        let original = fixtures::read::<_, m![Rows, Hidden]>(&mut device.pdma, &input).await;
        for rows in [1, 8, 9, 16, 17, 32, 33, 129, 257, 512] {
            let mut padded = original.clone();
            padded[rows * Hidden::SIZE..].fill(bf16::from_f32(f32::NAN));
            let input =
                fixtures::upload::<bf16, m![Rows, Hidden]>(&mut device.pdma, padded.clone()).await;
            let count = count(&mut device, rows).await;
            let result = launch(project_output, (&mut device, &projection, &count, &input))
                .await
                .unwrap();
            check::<Hidden, OutputWidth>(&mut device, result, rows).await;
            if rows == 257 {
                let mut window = vec![bf16::from_f32(f32::NAN); Rows::SIZE * Hidden::SIZE];
                window.extend(padded);
                let window =
                    fixtures::upload::<bf16, m![Reach, Hidden]>(&mut device.pdma, window).await;
                let result = launch(project_window, (&mut device, &projection, &count, &window))
                    .await
                    .unwrap();
                check::<Hidden, OutputWidth>(&mut device, result, rows).await;
            }
        }
    }
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "opens a device"]
    async fn preserves_fp32_partials_until_the_output_is_scaled() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(project_output.topology()).unwrap();
        let weight = fixtures::upload::<
            f4e2m1,
            crate::ops::linear::nvfp4::tiled::output::Stored<Hidden, OutputWidth>,
        >(
            &mut device.pdma,
            vec![f4e2m1::from_bits(2); Hidden::SIZE * OutputWidth::SIZE],
        )
        .await;
        let weight_scale = fixtures::upload::<
            f8e4m3,
            crate::ops::linear::nvfp4::tiled::output::StoredScales<Hidden, OutputWidth>,
        >(
            &mut device.pdma,
            vec![f8e4m3::from_f32(0.125); Hidden::SIZE * OutputWidth::SIZE / 16],
        )
        .await;
        // The two input halves nearly cancel: +/-1536 plus a 0.125..0.5 residual.
        // Rounding cluster partials to BF16 before exchange would erase that residual.
        let factor = 0.1337_f32;
        let projection = OutputProjection {
            weights: BlockScaled::new(
                weight,
                Blocks::new(weight_scale),
                crate::ops::linear::nvfp4::tiled::Scales {
                    output: fixtures::upload::<f32, m![1 # 8]>(&mut device.pdma, vec![factor; 8])
                        .await,
                },
            ),
        };
        let mut cancellation = Vec::with_capacity(Rows::SIZE * Hidden::SIZE);
        for row in 0..Rows::SIZE {
            let sign = if row % 2 == 0 { 1.0 } else { -1.0 };
            let mut values = vec![6.0 * sign; Hidden::SIZE];
            values[Hidden::SIZE / 2..].fill(-6.0 * sign);
            values[0] += sign * (row % 4 + 1) as f32;
            cancellation.extend(values.into_iter().map(bf16::from_f32));
        }
        let input =
            fixtures::upload::<bf16, m![Rows, Hidden]>(&mut device.pdma, cancellation).await;
        for rows in [32, 33, 257] {
            let count = count(&mut device, rows).await;
            let result = launch(project_output, (&mut device, &projection, &count, &input))
                .await
                .unwrap()
                .to_host::<m![Rows, OutputWidth]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for row in 0..rows {
                let sign = if row % 2 == 0 { 1.0 } else { -1.0 };
                let expected = bf16::from_f32(sign * (row % 4 + 1) as f32 * 0.125 * factor);
                assert!(
                    result[row * OutputWidth::SIZE..(row + 1) * OutputWidth::SIZE]
                        .iter()
                        .all(|value| *value == expected),
                    "cancellation rows={rows}, row={row}"
                );
            }
        }
    }
}
