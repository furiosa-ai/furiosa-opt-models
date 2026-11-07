use furiosa_opt_std::prelude::*;

axes![Factor = 8, Copies = 512, Ring = 16];

/// Every scale of one kernel in one 256-byte row, so one DMA lands all of them.
pub type Packed = m![Factor, 1 # 8];
pub type Register<B, C, S> = VrfTensor<f32, B, C, S, m![1 # 8]>;
pub use crate::ops::linear::contraction::InputScale;
/// A packed row as landed: one slice of every 16-slice ring, for [`Scales::spread`].
pub struct Seeds<B: M>(
    DmTensor<f32, B, m![Copies / 256], m![Copies % 256 / 16, 1 # 16], m![Factor, 1 # 8]>,
);

impl<B: M> Seeds<B> {
    pub fn new() -> Self {
        Self(DmTensor::new())
    }

    pub fn land_into<E: M>(&mut self, device: &mut Device, packed: HbmTensorView<'_, f32, B, E>) {
        packed.to_dm_view(&mut device.tdma, self.0.view_mut());
    }
}

/// The packed row in every slice of both clusters.
pub struct Scales<B: M>(
    DmTensor<f32, B, m![Copies / 256], m![Copies % 256 / 16, Ring], m![Factor, 1 # 8]>,
);

impl<B: M> Scales<B> {
    /// Lands one packed row (`E` is [`Packed`], optionally under a one-wide stacking tile).
    pub fn land<E: M>(device: &mut Device, packed: HbmTensorView<'_, f32, B, E>) -> Self {
        let mut scales = Self::new();
        scales.land_into(device, packed);
        scales
    }

    /// Scales not yet landed, for `land_into`.
    pub fn new() -> Self {
        Self(DmTensor::new())
    }

    /// The DMA lands the row in one slice of each ring and the switch spreads it over the ring,
    /// as the oracle's scale loads do, instead of one 256-byte DMA run per slice.
    pub fn land_into<E: M>(&mut self, device: &mut Device, packed: HbmTensorView<'_, f32, B, E>) {
        let mut seeds = Seeds::new();
        seeds.land_into(device, packed);
        self.spread(device, &seeds);
    }

    /// One packed row copied into every slice by the transfer alone: no switch, so no switch
    /// table staged for the step, at the cost of one run a slice.
    pub fn copy_into<E: M>(&mut self, device: &mut Device, packed: HbmTensorView<'_, f32, B, E>) {
        packed.to_dm_view(&mut device.tdma, self.0.view_mut());
    }

    /// Spreads `seeds` over every slice of their rings.
    pub fn spread(&mut self, device: &mut Device, Seeds(seeds): &Seeds<B>) {
        device
            .main
            .begin(seeds.view())
            .fetch::<m![Factor], m![1 # 8]>()
            .switch::<m![Copies % 256 / 16, Ring], m![Factor]>(SwitchConfig::CustomBroadcast {
                ring_size: 16,
            })
            .collect::<m![Factor], m![1 # 8]>()
            .commit_trim::<m![1 # 8]>()
            .commit_view(self.0.view_mut());
    }

    /// The packed row in every slice, under the caller's cluster and slice names.
    pub fn row<C: M, S: M>(&self) -> DmTensorView<'_, f32, B, C, S, m![Factor, 1 # 8]> {
        const { assert!(C::SIZE == 2 && S::SIZE == 256) };
        // SAFETY: every physical cluster and slice holds the same packed scales.
        unsafe { self.0.view().reshape::<B, C, S, m![Factor, 1 # 8]>() }
    }

    pub fn get<C: M, S: M>(&self, device: &mut Device, factor: usize) -> Register<B, C, S> {
        const { assert!(C::SIZE == 2 && S::SIZE == 256) };
        // SAFETY: every physical cluster and slice holds the same packed scales.
        let view = unsafe { self.0.view().reshape::<B, C, S, m![Factor, 1 # 8]>() };
        device
            .sub
            .begin(
                view.tile::<m![Factor], 1, m![Factor = 1 # 8, 1 # 8]>(factor)
                    .unpad::<m![Factor = 1, 1 # 8]>(),
            )
            .fetch::<m![Factor = 1], m![1 # 8]>()
            .collect::<m![Factor = 1], m![1 # 8]>()
            .to_vrf::<m![1 # 8]>()
    }
}

pub use crate::ops::linear::contraction::Scales as Epilogue;

/// Divides by the input scale and rounds once to FP8.
pub fn quantize<C: M, S: M, T: M, P: M, B: M>(
    device: &mut Device,
    input: &DmTensor<bf16, B, C, S, m![{ T }, { P }]>,
    projection: &InputScale<B, C, S>,
) -> DmTensor<f8e4m3, B, C, S, m![{ T }, { P }]> {
    const { assert!(P::SIZE == 8 && T::SIZE.is_multiple_of(2)) };
    device
        .main
        .begin(input.view())
        .fetch::<m![{ T } / 2], m![{ T } % 2, { P }]>()
        .fetch_cast::<f32>()
        .collect::<T, P>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ T }, { P } / 4], m![{ P } % 4]>()
        .vector_fp_div(projection.input_scale())
        .vector_widen_concat::<T, P>()
        .vector_final()
        .cast::<f8e4m3, m![{ P } # 32]>()
        .commit_trim::<P>()
        .commit()
}
