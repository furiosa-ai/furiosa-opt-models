use super::*;
use crate::ops::linear::nvfp4::tiled::Scales;

/// Each cluster owns half the intermediate features and accumulates full-width outputs.
#[derive(DeviceSend, crate::load::Load)]
#[load(tree)]
pub struct Down<I: AxisName, O: AxisName> {
    pub weight: HbmTensor<f4e2m1, Chip, Stored<I, O>>,
    pub weight_scale: HbmTensor<f8e4m3, Chip, StoredScales<I, O>>,
    #[load(rename = "")]
    pub scales: Scales,
}
pub type Stored<I, O> = m![
    I % 224 / 32,
    I / (I / 2),
    O / 512,
    I % (I / 2) / 224,
    O % 512,
    I % 32
];

pub type StoredScales<I, O> = m![
    I % 224 / 32,
    I / (I / 2),
    O / 512,
    I % (I / 2) / 224,
    O % 512,
    I % 32 / 16
];
// Storage retains seven contiguous 32-feature stripes. Two adjacent stripes
// form one 64-feature contraction panel, with no checkpoint padding or repack.
axes![Wide = 64, Tail = 32, Panel = 4];
pub(super) type Partition<I, O> = m![O / 512, I % (I / 2) / 224];
type Weight<I, O, D, K> =
    DmTensor<D, Chip, Cluster<I>, Partition<I, O>, m![{ Symbol<K> } / 32, O % 512, I % 32]>;
type BlockScales<I, O, K> = DmTensor<
    f8e4m3,
    Chip,
    Cluster<I>,
    Partition<I, O>,
    m![{ Symbol<K> } / 32, O % 512, I % 32 / 16],
>;
pub(super) type Packed<I, O, K> = (Weight<I, O, f4e2m1, K>, BlockScales<I, O, K>);
pub(super) type Ahead<I, O> = (Packed<I, O, Wide>, Packed<I, O, Wide>);
type Activation<I, O, R, K> =
    DmTensor<bf16, Chip, Cluster<I>, Partition<I, O>, m![{ Symbol<K> } / 32, R, I % 32]>;
type Lanes<I, O, R, K, const W: usize> = TrfTensor<
    bf16,
    Chip,
    Cluster<I>,
    Partition<I, O>,
    m![R % 8],
    m![R % {W} / 16, { Symbol<K> } / 32, {Symbol<K>} % 32 / 16, R % 16 / 8, {Symbol<K>} % 16],
>;
type RestoredLanes<I, O, R, K, const W: usize> = TrfTensor<
    bf16,
    Chip,
    Cluster<I>,
    Partition<I, O>,
    m![R % 8],
    m![R % {W} / 16, { Symbol<K> } / 32, I % 32 / 16, R % 16 / 8, I % 16],
>;
// Retain all four FP32 panel results until the final ordered reduction.
type Partials<I, O, R> = DmTensor<f32, Chip, Cluster<I>, m![O / 16], m![R, Panel, O % 16]>;
type Accum<I, O, R> = DmTensor<f32, Chip, Cluster<I>, m![O / 16], m![R, O % 16]>;

impl<I: AxisName, O: AxisName> Down<I, O> {
    /// Canonical rows to seven aligned 512-output × 32-input tiles per slice.
    pub fn reorder<const PACK: usize, T: Copy>(values: &[T]) -> Vec<T> {
        assert!(I::SIZE == 14336 && O::SIZE == 4096 && matches!(PACK, 1 | 2 | 16));
        assert_eq!(values.len(), O::SIZE * I::SIZE / PACK);
        let mut packed = Vec::with_capacity(values.len());
        for panel in 0..7 {
            for cluster in 0..2 {
                for owner in 0..O::SIZE / 512 {
                    for group in 0..32 {
                        for output in owner * 512..(owner + 1) * 512 {
                            let at = (output * I::SIZE
                                + cluster * (I::SIZE / 2)
                                + group * 224
                                + panel * 32)
                                / PACK;
                            packed.extend_from_slice(&values[at..at + 32 / PACK]);
                        }
                    }
                }
            }
        }
        packed
    }

    fn load<K: AxisName, const PARTS: usize>(
        &self,
        device: &mut Device,
        stripe: usize,
    ) -> Packed<I, O, K> {
        let mut codes = Weight::<I, O, f4e2m1, K>::new();
        let mut scales = BlockScales::<I, O, K>::new();
        for part in 0..PARTS {
            // The seven-stripe storage digit cannot be tiled by two in DMA.
            // Land each stripe directly into its final half of the wide panel.
            let target = codes.view_mut().tile::<m![{Symbol<K>} / 32], 1,
                m![{Symbol<K>} / 32 = 1 #{!} ({Symbol<K>} / 32), O % 512, I % 32]>(part);
            self.weight.view().tile::<m![I % 224 / 32], 1,
                m![I % 224 / 32 = 1 # 7,I / (I / 2),O / 512,I % (I / 2) / 224,O % 512,I % 32]>(stripe + part)
                .to_dm_view(&mut device.tdma, target);
            let target = scales.view_mut().tile::<m![{Symbol<K>} / 32], 1,
                m![{Symbol<K>} / 32 = 1 #{!} ({Symbol<K>} / 32), O % 512, I % 32 / 16]>(part);
            self.weight_scale.view().tile::<m![I % 224 / 32], 1,
                m![I % 224 / 32 = 1 # 7,I / (I / 2),O / 512,I % (I / 2) / 224,O % 512,I % 32 / 16]>(stripe + part)
                .to_dm_view(&mut device.tdma, target);
        }
        (codes, scales)
    }

    pub(super) fn stage(&self, device: &mut Device, panel: usize) -> Packed<I, O, Wide> {
        self.load::<Wide, 2>(device, panel * 2)
    }

    pub(super) fn restore(
        device: &mut Device,
        packed: Packed<I, O, Wide>,
    ) -> Weight<I, O, bf16, Wide> {
        block::restore(device, packed.0, packed.1)
    }

    fn input<R: AxisName, K: AxisName, const N: usize, const PARTS: usize>(
        device: &mut Device,
        product: &impl Product<I, R>,
        stripe: usize,
    ) -> Activation<I, O, R, K> {
        let mut output = Activation::new();
        #[unroll]
        for part in 0..PARTS {
            let target = output.view_mut()
                .tile::<m![{Symbol<K>} / 32],1,
                    m![{Symbol<K>} / 32 = 1 #{!} ({Symbol<K>} / 32), R, I % 32]>(part)
                .tile::<m![R],N,
                    m![{Symbol<K>} / 32 = 1 #{!} ({Symbol<K>} / 32),R = {N} #{!} R,I % 32]>(0);
            product.stripe::<O, N, _>(device, stripe + part, 0, target);
        }
        output
    }

    fn land<R: AxisName, K: AxisName, const PARTS: usize>(
        device: &mut Device,
        product: &impl Product<I, R>,
        stripe: usize,
        rows: usize,
    ) -> Activation<I, O, R, K> {
        if rows <= 16 {
            Self::input::<R, K, 16, PARTS>(device, product, stripe)
        } else if rows <= 64 {
            Self::input::<R, K, 64, PARTS>(device, product, stripe)
        } else if rows <= 128 {
            Self::input::<R, K, 128, PARTS>(device, product, stripe)
        } else if rows <= 256 {
            Self::input::<R, K, 256, PARTS>(device, product, stripe)
        } else {
            Self::input::<R, K, 512, PARTS>(device, product, stripe)
        }
    }

    fn lanes<R: AxisName, K: AxisName, const W: usize>(
        device: &mut Device,
        input: &Activation<I, O, R, K>,
        wave: usize,
        groups: usize,
    ) -> Lanes<I, O, R, K, W> {
        // SAFETY: only rename the contiguous local 32-input stripe; I owners stay fixed.
        let input = unsafe {
            input.view().reshape::<Chip, Cluster<I>, Partition<I, O>,
            m![{Symbol<K>} / 32, R, {Symbol<K>} % 32]>()
        };
        let stream = device.sub.begin(input.tile::<m![R / {W}],1,
            m![{Symbol<K>} / 32,R / {W} = 1 # (R / {W}),R % {W},{Symbol<K>} % 32]>(wave))
            .fetch::<m![R % 8,R % {W} / 16,{Symbol<K>} / 32,{Symbol<K>} % 32 / 16,R % 16 / 8],m![{Symbol<K>} % 16]>()
            .collect::<m![R % 8,R % {W} / 16,{Symbol<K>} / 32,{Symbol<K>} % 32 / 16,R % 16 / 8],m![{Symbol<K>} % 16]>()
            ;
        if W == 16 {
            stream.to_trf()
        } else {
            stream.to_trf_prefix::<m![R % { W } / 16], _, _>(groups)
        }
    }

    // Pack two future output owners (O bit 4), not two local output features.
    // Row bit 3 is a TRF-only time broadcast. Group the local output bits
    // together so packed fetch and FP32 scale addressing fit eight sequencer entries.
    fn contract<R: AxisName, K: AxisName, const W: usize, E: M>(
        device: &mut Device,
        weight: &Packed<I, O, K>,
        input: &Lanes<I, O, R, K, W>,
        output: DmTensorViewMut<'_, f32, Chip, Cluster<I>, m![O / 16], E>,
        groups: usize,
    ) {
        // SAFETY: K names this panel's local input coordinates without changing storage.
        let codes = unsafe {
            weight.0.view().reshape::<Chip, Cluster<I>, Partition<I, O>,
            m![{Symbol<K>} / 32, O % 512, {Symbol<K>} % 32]>()
        };
        let blocks = unsafe {
            weight.1.view().reshape::<Chip, Cluster<I>, Partition<I, O>,
            m![{Symbol<K>} / 32, O % 512, {Symbol<K>} % 32 / 16]>()
        };
        let scales: VrfTensor<
            f32,
            Chip,
            Cluster<I>,
            Partition<I, O>,
            m![{Symbol<K>} / 32, O % 512, {Symbol<K>} % 32 / 16],
        > = device
            .sub
            .begin(blocks)
            .fetch::<m![{Symbol<K>} / 32,O % 512 / 4], m![O % 4,{Symbol<K>} % 32 / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{Symbol<K>} / 32,O % 512 / 4], m![O % 4,{Symbol<K>} % 32 / 16]>()
            .to_vrf();
        let result = device.main.begin(codes)
            .fetch::<m![R % {W} / 16,O % 512 / 32,O % 16,{Symbol<K>} / 32,{Symbol<K>} % 32 / 16,O % 32 / 16],m![{Symbol<K>} % 16]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![R % {W} / 16,O % 512 / 32,O % 16,{Symbol<K>} / 32,{Symbol<K>} % 32 / 16,O % 32 / 16],m![{Symbol<K>} % 16]>()
            .contract_outer::<m![R % {W} / 16,O % 512 / 32,O % 16,{Symbol<K>} / 32,{Symbol<K>} % 32 / 16,R % 16 / 8],m![O % 32 / 16,{Symbol<K>} % 16],_,_,_>(input)
            .contract_packet::<m![O % 32 / 16]>()
            .contract_time::<m![R % {W} / 16,O % 512 / 32,O % 16,{Symbol<K>} / 32,{Symbol<K>} % 32 / 16,R % 16 / 8]>()
            .contract_lane::<m![R % {W} / 16,O % 512 / 32,O % 16,{Symbol<K>} / 32,{Symbol<K>} % 32 / 16,R % 16 / 8,O % 32 / 16],m![R % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R % {W} / 16,O % 512 / 32,O % 16,{Symbol<K>} / 32,{Symbol<K>} % 32 / 16,R % 16 / 8,O % 32 / 16,R % 8 / 4],m![R % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scales)
            .vector_intra_slice_reduce::<K,m![R % {W} / 16,O % 512 / 32,O % 16,R % 16 / 8,O % 32 / 16,R % 8 / 4],m![R % 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_concat::<m![R % {W} / 16,O % 512 / 32,O % 16,R % 16 / 8,O % 32 / 16],m![R % 8]>()
            .vector_inter_slice_reduce::<m![O / 16],m![R % {W} / 16,O % 16,R % 16 / 8]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .transpose::<m![R % {W} / 16,O % 16 / 2,R % 16],m![O % 2 # 8]>()
            .commit_trim::<m![O % 2]>();
        if W == 16 {
            result.commit_view(output);
        } else {
            result.commit_prefix::<m![R % { W } / 16], _, _>(output, groups);
        }
    }

    fn panel<R: AxisName, K: AxisName, const W: usize, const P: usize>(
        device: &mut Device,
        weight: &Packed<I, O, K>,
        input: &Activation<I, O, R, K>,
        output: &mut Partials<I, O, R>,
        rows: usize,
    ) {
        let groups = rows.div_ceil(16);
        let mut current = Self::lanes::<R, K, W>(device, input, 0, groups);
        // Keep the accumulator views statically indexed. Runtime-indexed views
        // failed the A16 MLP-to-QKV transition when next-layer preparation was
        // live. Static waves preserve the overlap without copies or snapshots.
        // The compact branch has at most 16 rows; prefill has two span waves.
        let waves = if W == 16 { 1 } else { R::SIZE / W };
        #[unroll]
        for wave in 0..waves {
            if wave * W < rows {
                let next = Self::lanes::<R, K, W>(
                    device,
                    input,
                    (wave + 1) % (R::SIZE / W),
                    groups - groups.min(W / 16 * (wave + 1)),
                );
                let live = groups - W / 16 * wave;
                let target = output
                    .view_mut()
                    .tile::<m![Panel], 1, m![R, Panel = 1 #{!} 4, O % 16]>(P)
                    .tile::<m![R / { W }], 1,
                        m![R / {W} = 1 #{!} (R / {W}), R % {W}, Panel = 1 #{!} 4, O % 16]>(wave);
                Self::contract::<R, K, W, _>(device, weight, &current, target, live);
                current = next;
            }
        }
    }

    fn run<R: AxisName, const W: usize, N: Points<Device>>(
        &self,
        device: &mut Device,
        product: &impl Product<I, R>,
        first: Ahead<I, O>,
        rows: usize,
        next: N,
    ) -> (Owned<O, R>, N::Output) {
        let (first, second) = first;
        let mut partials = Partials::new();
        let input0 = Self::land::<R, Wide, 2>(device, product, 0, rows);
        let tail = self.load::<Tail, 1>(device, 6);
        let mut next = next.at(device);
        Self::panel::<R, Wide, W, 0>(device, &first, &input0, &mut partials, rows);
        // Fetch panel 2 during down 0, after gate/up buffers have been released.
        // This bounds SRAM while retaining O-weight reuse across spans.
        let third = self.stage(device, 2);
        // These loads follow command issuance, so they can execute during down 0.
        let input1 = Self::land::<R, Wide, 2>(device, product, 2, rows);
        next.between(device);
        let next = next.at(device);
        Self::panel::<R, Wide, W, 1>(device, &second, &input1, &mut partials, rows);
        let input2 = Self::land::<R, Wide, 2>(device, product, 4, rows);
        Self::panel::<R, Wide, W, 2>(device, &third, &input2, &mut partials, rows);
        let input3 = Self::land::<R, Tail, 1>(device, product, 6, rows);
        Self::panel::<R, Tail, W, 3>(device, &tail, &input3, &mut partials, rows);
        let sums = Self::sum_panels(device, &partials, rows);
        let output = if rows <= 16 {
            self.finish::<R, 16>(device, &sums)
        } else if rows <= 128 {
            self.finish::<R, 128>(device, &sums)
        } else {
            self.finish::<R, 256>(device, &sums)
        };
        (output, next.finish(device))
    }

    fn lanes_restored<R: AxisName, K: AxisName, const W: usize>(
        device: &mut Device,
        input: &Activation<I, O, R, K>,
        wave: usize,
        groups: usize,
    ) -> RestoredLanes<I, O, R, K, W> {
        let stream = device.sub.begin(input.view().tile::<m![R / {W}],1,
            m![{Symbol<K>} / 32,R / {W} = 1 # (R / {W}),R % {W},I % 32]>(wave))
            .fetch::<m![R % 8,R % {W} / 16,{Symbol<K>} / 32,I % 32 / 16,R % 16 / 8],m![I % 16]>()
            .collect::<m![R % 8,R % {W} / 16,{Symbol<K>} / 32,I % 32 / 16,R % 16 / 8],m![I % 16]>()
            ;
        if W == 16 {
            stream.to_trf()
        } else {
            stream.to_trf_prefix::<m![R % { W } / 16], _, _>(groups)
        }
    }

    // Pack two future output owners (O bit 4), not two local output features.
    // Row bit 3 is a TRF-only time broadcast. This reuses streamed weights and
    // preserves the efficient inter-slice reduction order with FP32 output.
    fn contract_restored<R: AxisName, K: AxisName, const W: usize, E: M>(
        device: &mut Device,
        weight: &Weight<I, O, bf16, K>,
        input: &RestoredLanes<I, O, R, K, W>,
        output: DmTensorViewMut<'_, f32, Chip, Cluster<I>, m![O / 16], E>,
        groups: usize,
    ) {
        let result = device.main.begin(weight.view())
            .fetch::<m![R % {W} / 16,O % 16 / 2,O % 512 / 32,O % 2,{Symbol<K>} / 32,I % 32 / 16,O % 32 / 16],m![I % 16]>()
            .collect::<m![R % {W} / 16,O % 16 / 2,O % 512 / 32,O % 2,{Symbol<K>} / 32,I % 32 / 16,O % 32 / 16],m![I % 16]>()
            .contract_outer::<m![R % {W} / 16,O % 16 / 2,O % 512 / 32,O % 2,{Symbol<K>} / 32,I % 32 / 16,R % 16 / 8],m![O % 32 / 16,I % 16],_,_,_>(input)
            .contract_packet::<m![O % 32 / 16]>()
            .contract_time::<m![R % {W} / 16,O % 16 / 2,O % 512 / 32,O % 2,R % 16 / 8]>()
            .contract_lane::<m![R % {W} / 16,O % 16 / 2,O % 512 / 32,O % 2,R % 16 / 8,O % 32 / 16],m![R % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<m![O / 16],m![R % {W} / 16,O % 16 / 2,O % 2,R % 16 / 8]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .transpose::<m![R % {W} / 16,O % 16 / 2,R % 16],m![O % 2 # 8]>()
            .commit_trim::<m![O % 2]>();
        if W == 16 {
            result.commit_view(output);
        } else {
            result.commit_prefix::<m![R % { W } / 16], _, _>(output, groups);
        }
    }

    fn panel_restored<R: AxisName, K: AxisName, const W: usize, const P: usize>(
        device: &mut Device,
        weight: &Weight<I, O, bf16, K>,
        input: &Activation<I, O, R, K>,
        output: &mut Partials<I, O, R>,
        rows: usize,
    ) {
        let groups = rows.div_ceil(16);
        let mut current = Self::lanes_restored::<R, K, W>(device, input, 0, groups);
        // Keep the accumulator views statically indexed. Runtime-indexed views
        // failed the A16 MLP-to-QKV transition when next-layer preparation was
        // live. Static waves preserve the overlap without copies or snapshots.
        // The compact branch has at most 16 rows; prefill has two span waves.
        let waves = if W == 16 { 1 } else { R::SIZE / W };
        #[unroll]
        for wave in 0..waves {
            if wave * W < rows {
                let next = Self::lanes_restored::<R, K, W>(
                    device,
                    input,
                    (wave + 1) % (R::SIZE / W),
                    groups - groups.min(W / 16 * (wave + 1)),
                );
                let live = groups - W / 16 * wave;
                let target = output
                    .view_mut()
                    .tile::<m![Panel], 1, m![R, Panel = 1 #{!} 4, O % 16]>(P)
                    .tile::<m![R / { W }], 1,
                        m![R / {W} = 1 #{!} (R / {W}), R % {W}, Panel = 1 #{!} 4, O % 16]>(wave);
                Self::contract_restored::<R, K, W, _>(device, weight, &current, target, live);
                current = next;
            }
        }
    }

    fn run_restored<R: AxisName, const W: usize, N: Points<Device>>(
        &self,
        device: &mut Device,
        product: &impl Product<I, R>,
        first: Ahead<I, O>,
        rows: usize,
        next: N,
    ) -> (Owned<O, R>, N::Output) {
        let (first, second) = first;
        let first = Self::restore(device, first);
        let mut partials = Partials::new();
        let input0 = Self::land::<R, Wide, 2>(device, product, 0, rows);
        let tail = self.load::<Tail, 1>(device, 6);
        let mut next = next.at(device);
        Self::panel_restored::<R, Wide, W, 0>(device, &first, &input0, &mut partials, rows);
        // Fetch panel 2 during down 0, after gate/up buffers have been released.
        // This bounds SRAM while retaining O-weight reuse across spans.
        let third = self.stage(device, 2);
        // These loads follow command issuance, so they can execute during down 0.
        let input1 = Self::land::<R, Wide, 2>(device, product, 2, rows);
        let second = Self::restore(device, second);
        next.between(device);
        let next = next.at(device);
        Self::panel_restored::<R, Wide, W, 1>(device, &second, &input1, &mut partials, rows);
        let input2 = Self::land::<R, Wide, 2>(device, product, 4, rows);
        let third = Self::restore(device, third);
        Self::panel_restored::<R, Wide, W, 2>(device, &third, &input2, &mut partials, rows);
        let input3 = Self::land::<R, Tail, 1>(device, product, 6, rows);
        let tail = block::restore(device, tail.0, tail.1);
        Self::panel_restored::<R, Tail, W, 3>(device, &tail, &input3, &mut partials, rows);
        let sums = Self::sum_panels(device, &partials, rows);
        let output = if rows <= 16 {
            self.finish::<R, 16>(device, &sums)
        } else if rows <= 128 {
            self.finish::<R, 128>(device, &sums)
        } else {
            self.finish::<R, 256>(device, &sums)
        };
        (output, next.finish(device))
    }

    /// Sum panels in contraction order without intermediate DM read/add/write passes.
    /// Only the 16-row prefix written by every panel may be consumed.
    fn sum_panels<R: AxisName>(
        device: &mut Device,
        partials: &Partials<I, O, R>,
        rows: usize,
    ) -> Accum<I, O, R> {
        let mut sums = Accum::new();
        device
            .main
            .begin(partials.view())
            .fetch::<m![R, Panel, O % 16 / 8], m![O % 8]>()
            .collect::<m![R, Panel, O % 16 / 8], m![O % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R, Panel, O % 16 / 4], m![O % 4]>()
            .vector_intra_slice_reduce::<Panel, m![R, O % 16 / 4], m![O % 4]>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_widen_concat::<m![R, O % 16 / 8], m![O % 8]>()
            .vector_final()
            .commit_trim::<m![O % 8]>()
            .commit_prefix::<m![R / 16], _, _>(sums.view_mut(), rows.div_ceil(16));
        sums
    }

    fn finish<R: AxisName, const N: usize>(
        &self,
        device: &mut Device,
        sums: &Accum<I, O, R>,
    ) -> Owned<O, R> {
        let half = |device: &mut Device, indices| {
            sums.view()
                .tile::<m![R % (R / 2)], N, m![R / (R / 2), R % (R / 2) = {N} # (R / 2), O % 16]>(0)
                .asymmetric_cluster_slice::<m![R / (R / 2)], m![R % (R / 2) = { N }, O % 16]>(
                    &mut device.sub,
                    indices,
                )
        };
        let (send, own) = (half(device, &[1, 0]), half(device, &[0, 1]));
        let exchanged = send.view().cluster_swap().to_dm(&mut device.tdma);
        let values = super::sum_scaled(device, &self.scales, &own, &exchanged);
        let mut output: Owned<O, R> = DmTensor::new();
        // SAFETY: cluster c now owns row half c, with all feature halves summed.
        let values = unsafe {
            values.reshape::<Chip, m![R / (R / 2)], m![O / 16], m![R % (R / 2) = { N }, O % 16]>()
        };
        values.view().to_dm_view(
            &mut device.tdma,
            output
                .view_mut()
                .tile::<m![R % (R / 2)], N, m![R % (R / 2) = {N} #{!} (R / 2), O % 16]>(0),
        );
        output
    }

    pub(super) fn forward<R: AxisName, N: Points<Device>>(
        &self,
        device: &mut Device,
        product: &impl Product<I, R>,
        first: Ahead<I, O>,
        rows: usize,
        next: N,
    ) -> (Owned<O, R>, N::Output) {
        const {
            assert!(I::SIZE == 14336 && O::SIZE == 4096 && R::SIZE == 512);
        }
        if rows <= 16 {
            self.run::<R, 16, N>(device, product, first, rows, next)
        } else {
            self.run_restored::<R, 256, N>(device, product, first, rows, next)
        }
    }
}

crate::ops::linear::nvfp4::tiled::loading!(Down);
