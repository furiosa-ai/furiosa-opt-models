//! Weight-stationary O projection partitioned across clusters by input features.
//! Each slice restores a [256 output, 128 input] BF16 tile into 64 KiB of TRF.
//! Only the owned input columns land in each cluster. FP32 partials exchange in
//! DM and add before tensor scaling and the single final BF16 rounding.
use super::{Scales as Factors, block};
use crate::Op;
use crate::ops::linear::nvfp4::tiled::prefetch::Points;
use crate::ops::normalization::span::Owned;
use crate::tensor::{live::Live, parallel::Chip};
use furiosa_opt_std::prelude::*;

#[derive(DeviceSend, crate::load::Load)]
#[load(tree)]
pub struct Output<I: AxisName, O: AxisName> {
    pub weight: HbmTensor<f4e2m1, Chip, Stored<I, O>>,
    pub weight_scale: HbmTensor<f8e4m3, Chip, StoredScales<I, O>>,
    #[load(rename = "")]
    pub scales: Factors,
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
type Operand<I, O> =
    TrfTensor<bf16, Chip, Cluster<I>, Slices<I, O>, m![O % 8], m![O % 256 / 8, I % 128]>;
type Partial<I, O, R, const P: usize> =
    DmTensor<f32, Chip, Cluster<I>, Reduced<O>, m![R % { P }, O % 16]>;
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

    fn pass<R: AxisName, All: AxisName, const PASS: usize, N: Points<Device>>(
        &self,
        device: &mut Device,
        weight: &Operand<I, O>,
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
        // Stream BF16 activations directly from their landed DM buffer.
        let partial: Partial<I, O, R, PASS> = device
            .main
            .begin(input.view())
            .fetch::<m![R % { PASS }, O % 256 / 8, I % 128 / 16], m![I % 16]>()
            .collect::<m![R % { PASS }, O % 256 / 8, I % 128 / 16], m![I % 16]>()
            .contract_outer::<m![R % { PASS }, O % 256 / 8, I % 128 / 32], m![I % 32], _, _, _>(
                weight,
            )
            .contract_packet::<m![1]>()
            .contract_time::<m![R % { PASS }, O % 256 / 8]>()
            .contract_lane::<m![R % { PASS }, O % 256 / 8], m![O % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Reduced<O>, m![R % { PASS }, O % 16 / 8]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .commit_trim::<m![O % 8]>()
            .commit();
        // Request following operands before enqueueing the output transfer. Otherwise
        // that DMA waits for this contraction and holds the prefetch behind it.
        let next = next.at(device);
        let factor = block::scalar::<Selected<I>, Reduced<O>>(device, &self.scales.output);
        Self::handoff::<R, PASS>(device, &partial, &factor, output, at);
        next
    }
    // Four 32-row owners share BF16 input blocks, broadcast over the switch.
    fn compact_pass<R: AxisName, All: AxisName, N: Points<Device>>(
        &self,
        device: &mut Device,
        weight: &Operand<I, O>,
        factor: &Factor<I, O>,
        input: &Rows<All, I>,
        output: &mut Owned<O, R>,
        wave: usize,
        start: usize,
        next: N,
    ) -> N::Next {
        let window = input
            .view()
            .tile::<m![All], 128, m![All = 128 # All, I]>(start + wave * 128);
        // SAFETY: rename the selected row window while retaining its source stride.
        let window = unsafe { window.reshape::<Chip, m![R % 128 # All, I]>() };
        let input: DmTensor<
            bf16,
            Chip,
            Cluster<I>,
            m![O / 1024, R % 128 / 32, I % (I / 2) / 128],
            m![R % 32, I % 128],
        > = window.to_dm(&mut device.tdma);
        let partial: Partial<I, O, R, 128> = device
            .main.begin(input.view())
            .fetch::<m![R % 32, I % 128 / 32], m![I % 32]>()
            .switch::<Slices<I, O>, m![R % 32, I % 128 / 32, R % 128 / 32]>(
                SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
            .collect::<m![R % 32, I % 128 / 32, R % 128 / 32, I % 32 / 16], m![I % 16]>()
            .contract_outer::<m![R % 32, I % 128 / 32, R % 128 / 32, O % 256 / 8], m![I % 32], _, _, _>(weight)
            .contract_packet::<m![1]>()
            .contract_time::<m![R % 32, R % 128 / 32, O % 256 / 8]>()
            .contract_lane::<m![R % 32, R % 128 / 32, O % 256 / 8], m![O % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Reduced<O>, m![R % 32, R % 128 / 32, O % 16 / 8]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .commit_trim::<m![O % 8]>()
            .commit();
        let next = next.at(device);
        Self::handoff::<R, 128>(device, &partial, factor, output, wave);
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
        self.weight
            .view()
            .to_dm_view(&mut device.tdma, staged.0.view_mut());
        self.weight_scale
            .view()
            .to_dm_view(&mut device.tdma, staged.1.view_mut());
    }

    pub(crate) fn prepare(&self, device: &mut Device) -> Staged<I, O> {
        let codes: Weight<I, O, f4e2m1> = self.weight.to_dm(&mut device.tdma);
        let scales: Scales<I, O> = self.weight_scale.to_dm(&mut device.tdma);
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
        let weight = block::restore_view(device, weight.0.view(), weight.1.view());
        // Attention's hook stages packed DM only. TRF becomes resident after attention.
        let resident: Operand<I, O> = device
            .sub
            .begin(weight.view())
            .fetch::<m![O % 8, O % 256 / 8, I % 128 / 16], m![I % 16]>()
            .collect::<m![O % 8, O % 256 / 8, I % 128 / 16], m![I % 16]>()
            .to_trf();
        let mut output = Owned::new();
        // Small launches need one live 8/32-row pass. Its tensor factor loads
        // after the first contraction and the following-operand prefetch hook.
        let next = if live.rows() <= 8 {
            self.pass::<R, All, 8, _>(device, &resident, input, &mut output, 0, start, next)
        } else if live.rows() <= 32 {
            self.pass::<R, All, 32, _>(device, &resident, input, &mut output, 0, start, next)
        } else {
            let factor = block::scalar::<Selected<I>, Reduced<O>>(device, &self.scales.output);
            let next = self.compact_pass(
                device,
                &resident,
                &factor,
                input,
                &mut output,
                0,
                start,
                next,
            );
            for following in 0..3 {
                if following + 1 < live.rows().div_ceil(128) {
                    self.compact_pass(
                        device,
                        &resident,
                        &factor,
                        input,
                        &mut output,
                        following + 1,
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

crate::ops::linear::nvfp4::tiled::loading!(Output);

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
