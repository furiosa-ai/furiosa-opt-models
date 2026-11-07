use super::*;
use mlp::Intermediate;
use qkv::QHeads;

axes![Traced = 2, Recorded = 2];
type Layer<D, E> = HbmTensor<D, Chips, m![Traced, { E }]>;
type ProductWave = m![
    Rows / 32,
    Intermediate / 8,
    Rows % 32 / 16,
    Intermediate % 8,
    Rows % 16
];

/// Wave 0's piece outputs of each of the first two layers, for an independent reference check.
#[derive(DeviceSend)]
pub struct Trace {
    pub queries: Layer<bf16, m![Rows, QHeads, Head]>,
    /// Waves 0 and 1's rows.
    pub attended: Layer<bf16, m![Recorded, Rows, Width]>,
    pub residual: Layer<bf16, Blocked>,
    pub activations: Layer<f8e4m3, m![Rows, Hidden]>,
    pub product: Layer<f8e4m3, ProductWave>,
    pub partials: Layer<bf16, Blocked>,
    pub output: Layer<bf16, Blocked>,
}

impl Trace {
    pub(super) fn queries(&mut self, device: &mut Device, (state, layer): (&QueryRows, usize)) {
        let rows = state
            .view()
            .tile::<m![Capacity], 64, m![Capacity = 64 # Capacity, QHeads, Head]>(0);
        // SAFETY: the 64-row tile keeps the capacity stride; only its row axis is renamed.
        let rows = unsafe { rows.reshape::<Chips, m![Rows # Capacity, QHeads, Head]>() };
        rows.to_hbm_view(
            &mut device.tdma,
            self.queries
                .view_mut()
                .tile::<m![Traced], 1, m![Traced = 1 #{!} Traced, Rows, QHeads, Head]>(layer),
        );
    }

    pub(super) fn attended(
        &mut self,
        device: &mut Device,
        (rows, layer, wave): (&output::Input, usize, usize),
    ) {
        for _ in 0..(Recorded::SIZE - wave.min(Recorded::SIZE)).min(1) {
            rows.view().to_hbm_view(
                &mut device.tdma,
                self.attended
                    .view_mut()
                    .tile::<m![Traced], 1, m![Traced = 1 #{!} Traced, Recorded, Rows, Width]>(layer)
                    .tile::<m![Recorded], 1, m![
                        Traced = 1 #{!} Traced,
                        Recorded = 1 #{!} Recorded,
                        Rows,
                        Width
                    ]>(wave.min(1)),
            );
        }
    }

    pub(super) fn residual(&mut self, device: &mut Device, (state, layer): (&HiddenRows, usize)) {
        wave(state, 0).to_hbm_view(&mut device.tdma, blocked(&mut self.residual, layer));
    }

    pub(super) fn output(&mut self, device: &mut Device, (state, layer): (&HiddenRows, usize)) {
        wave(state, 0).to_hbm_view(&mut device.tdma, blocked(&mut self.output, layer));
    }

    pub(super) fn mlp(
        &mut self,
        device: &mut Device,
        (activations, product, partials): (&mlp::Input, &mlp::Product, &mlp::Output),
        layer: usize,
    ) {
        activations
            .view()
            .tile::<m![Waves], 1, m![Waves = 1 # Waves, { mlp::Grouped }]>(0)
            .to_hbm_view(
                &mut device.tdma,
                self.activations
                    .view_mut()
                    .tile::<m![Traced], 1, m![Traced = 1 #{!} Traced, Rows, Hidden]>(layer),
            );
        product
            .view()
            .tile::<m![Waves], 1, m![Waves = 1 # Waves, { ProductWave }]>(0)
            .to_hbm_view(
                &mut device.tdma,
                self.product
                    .view_mut()
                    .tile::<m![Traced], 1, m![Traced = 1 #{!} Traced, { ProductWave }]>(layer),
            );
        partials
            .view()
            .tile::<m![Waves], 1, m![Waves = 1 # Waves, Hidden / 2560, Hidden % 2560 / 80,
                Rows / 8, Rows % 8, Hidden % 80]>(0)
            .to_hbm_view(&mut device.tdma, blocked(&mut self.partials, layer));
    }
}

fn blocked(
    target: &mut Layer<bf16, Blocked>,
    layer: usize,
) -> HbmTensorViewMut<'_, bf16, Chips, m![Traced = 1 #{!} Traced, { Blocked }]> {
    target
        .view_mut()
        .tile::<m![Traced], 1, m![Traced = 1 #{!} Traced, { Blocked }]>(layer)
}
