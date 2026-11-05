//! One wave's attended rows through the O projection, the residual and the post-attention norm,
//! into the wave's FP8 feed-forward activation, split where the residual's exchange is in flight:
//! work issued between `issue` and `complete` runs under the transfer.
use super::*;

/// One layer's O weight and the post-attention norm's gain and divisor, landed once for every wave
/// the layer projects.
pub(super) struct Tail<'a> {
    weight: output::Weight,
    norm: Norm,
    layer: (&'a stack::Layers<'a>, usize),
    /// This chip's index, which places its O weight's two halves within the stored copy.
    rank: usize,
}

/// The post-attention norm's gain and the feed-forward scales it quantizes the wave by.
struct Norm {
    gain: crate::ops::normalization::RmsNorm<
        Hidden,
        { 1e-5_f32.to_bits() as usize },
        Chips,
        norm::eight::Landed,
    >,
    scales: scale::Scales<Chips>,
}

/// The tail's scale rows as landed, alive only from `Tail::land` to `Tail::spread`.
pub(super) struct Seeds {
    output: scale::Seeds<Chips>,
    norm: scale::Seeds<Chips>,
}

impl Seeds {
    pub(super) fn new() -> Self {
        Self {
            output: scale::Seeds::new(),
            norm: scale::Seeds::new(),
        }
    }
}

/// The tail's transfers (`Tail::land` into `seeds`) as a batch's last wave issues them.
pub(super) struct Landing<'a, 'b> {
    pub(super) tail: &'a mut Tail<'b>,
    pub(super) seeds: &'a mut Seeds,
}

impl attention::Lead for Landing<'_, '_> {
    fn land(&mut self, device: &mut Device) {
        self.tail.land(device, self.seeds);
    }
}

/// The tail's scales in registers, the O projection's and the one the norm quantizes a wave by: a
/// load issued behind the wave's O fill holds the in-order stream until that fill finishes.
pub(super) struct Factors {
    output: output::Factors,
    norm: scale::InputScale<Chips, norm::Cluster, norm::eight::Slices>,
}

/// A prompt wave's hidden rows with its O partials added, and their copy on its way to the norm.
pub(super) struct Summed {
    rows: norm::eight::Values,
    hidden: Residual,
    at: usize,
}

/// A wave's O partial, projected and not yet sent.
pub(super) struct Projection {
    partial: output::Partial,
    at: usize,
}

/// A wave whose O partial is on its way to the peer chip, with the hidden rows it adds to.
pub(super) struct Pending {
    sent: residual::Exchanged<Cluster, Slices, Time, Packet>,
    hidden: Residual,
    at: usize,
    live: Live,
}

impl<'a> Tail<'a> {
    /// Weights not yet landed, for `land_into` to fill where the layer first needs them.
    pub(super) fn new(layer: (&'a stack::Layers<'a>, usize), rank: usize) -> Self {
        Self {
            weight: output::Weight::new(),
            norm: Norm {
                gain: crate::ops::normalization::RmsNorm::new(DmTensor::new()),
                scales: scale::Scales::new(),
            },
            layer,
            rank,
        }
    }

    pub(super) fn land_into(&mut self, device: &mut Device) {
        let mut seeds = Seeds::new();
        self.land(device, &mut seeds);
        self.spread(device, &seeds);
    }

    /// The O weight's, the norm gain's and the scale rows' transfers, without the Main work that
    /// spreads the scales, so the transfers can be issued ahead of other Main work.
    pub(super) fn land(&mut self, device: &mut Device, seeds: &mut Seeds) {
        let (layers, index) = self.layer;
        self.weight
            .land(device, (layers, index, self.rank), &mut seeds.output);
        layers
            .post_norm
            .at(index)
            .to_dm_view(&mut device.tdma, self.norm.gain.weight_mut().view_mut());
        seeds.norm.land_into(device, layers.mlp_scales.at(index));
    }

    pub(super) fn spread(&mut self, device: &mut Device, seeds: &Seeds) {
        self.weight.spread(device, &seeds.output);
        self.norm.scales.spread(device, &seeds.norm);
    }

    /// The layer's feed-forward scales the norm quantizes by, spread for the feed-forward itself.
    pub(super) fn scales(&self) -> &scale::Scales<Chips> {
        &self.norm.scales
    }

    /// The spread scales in registers, for the waves projected and normalized while they live.
    pub(super) fn factors(&self, device: &mut Device) -> Factors {
        Factors {
            output: self.weight.factors(device),
            norm: crate::ops::linear::Affine::new((), self.norm.scales.get(device, 0)),
        }
    }

    /// Projects `rows`, decode wave `at`'s attended rows, and sends the partial's `live` groups to
    /// the peer chip; `hidden` is the wave's landed hidden rows the partials add to.
    pub(super) fn issue(
        &self,
        device: &mut Device,
        factors: &Factors,
        (rows, hidden, at, live): (Projected<f32>, Residual, usize, Live),
        #[cfg(feature = "trace")] trace: (&mut Trace, Copied<f32>),
    ) -> Pending {
        let Projection { partial, at } = self.project(
            device,
            factors,
            (rows, at),
            #[cfg(feature = "trace")]
            trace,
        );
        let sent = live.exchange(device, partial);
        Pending {
            sent,
            hidden,
            at,
            live,
        }
    }

    /// Projects `rows`, wave `at`'s attended rows. Both clusters round the rows to BF16 where they
    /// land, so the projection reads them without an HBM round trip.
    fn project(
        &self,
        device: &mut Device,
        factors: &Factors,
        (rows, at): (Projected<f32>, usize),
        #[cfg(feature = "trace")] (trace, copy): (&mut Trace, Copied<f32>),
    ) -> Projection {
        let filled = self.weight.fill(device);
        let rows = round(device, &rows);
        #[cfg(feature = "trace")]
        {
            let copy: output::Input = round(device, &copy).to_hbm(&mut device.tdma);
            trace.attended(device, (&copy, self.layer.1, at));
        }
        Projection {
            partial: filled.project(device, (&rows, &factors.output)),
            at,
        }
    }

    /// `issue` of `rows` the caller keeps.
    #[cfg(not(feature = "trace"))]
    pub(super) fn issue_kept(
        &self,
        device: &mut Device,
        factors: &Factors,
        (rows, hidden, at, live): (&Projected<f32>, Residual, usize, Live),
    ) -> Pending {
        let filled = self.weight.fill(device);
        let rounded = round(device, rows);
        let partial = filled.project(device, (&rounded, &factors.output));
        Pending {
            sent: live.exchange(device, partial),
            hidden,
            at,
            live,
        }
    }

    /// `issue_kept` of a full wave, its partial sent whole outside any guard: a guard's end waits
    /// for the transfer's syncs, which outside one wait for the partial's first reader.
    #[cfg(not(feature = "trace"))]
    pub(super) fn issue_whole(
        &self,
        device: &mut Device,
        factors: &Factors,
        (rows, hidden, at): (&Projected<f32>, Residual, usize),
    ) -> Pending {
        let filled = self.weight.fill(device);
        let rounded = round(device, rows);
        let partial = filled.project(device, (&rounded, &factors.output));
        self.send_landed(device, (Projection { partial, at }, hidden))
    }

    /// `send` with the wave's hidden rows already landed.
    #[cfg(not(feature = "trace"))]
    pub(super) fn send_landed(
        &self,
        device: &mut Device,
        (Projection { partial, at }, hidden): (Projection, Residual),
    ) -> Pending {
        let sent = residual::exchange(device, partial);
        Pending {
            sent,
            hidden,
            at,
            live: Live::of(Rows::SIZE),
        }
    }

    /// Lands the projected wave's hidden rows and sends its partial to the peer chip.
    pub(super) fn send(
        &self,
        device: &mut Device,
        Projection { partial, at }: Projection,
        state: &State,
    ) -> Pending {
        let hidden: Residual = wave(&state.hidden, at).to_dm(&mut device.tdma);
        let sent = residual::exchange(device, partial);
        Pending {
            sent,
            hidden,
            at,
            live: Live::of(Rows::SIZE),
        }
    }

    /// `issue` of prompt wave `wave` (of a chunk starting at wave `start`).
    pub(super) fn issue_prompt(
        &self,
        device: &mut Device,
        factors: &Factors,
        (start, wave): (usize, usize),
        state: &State,
        #[cfg(feature = "trace")] trace: &mut Trace,
    ) -> Pending {
        let projection = self.project_prompt(
            device,
            factors,
            (start, wave),
            state,
            #[cfg(feature = "trace")]
            trace,
        );
        self.send(device, projection, state)
    }

    /// Prompt wave `wave`'s attended rows (of a chunk starting at wave `start`), landed for
    /// `project_into`.
    #[cfg(not(feature = "trace"))]
    pub(super) fn land_prompt(
        &self,
        device: &mut Device,
        (start, wave): (usize, usize),
        prompt: &attention::prompt::Attended,
    ) -> Projected<f32> {
        let attended = prompt.view().tile::<m![Prompt], 64, m![
            Prompt = 64 # Prompt,
            AttentionHeads,
            AttentionGroups,
            AttentionHead
        ]>(64 * (wave - start));
        // SAFETY: the 64-row tile keeps the chunk's row stride, and its row-major heads flatten
        // Heads/Groups/Head in wire order.
        let attended = unsafe { attended.reshape::<Chips, m![Rows # Prompt, Width]>() };
        attended.to_dm(&mut device.tdma)
    }

    /// `land_prompt` into `rows`, a buffer the caller keeps until the wave projects.
    #[cfg(not(feature = "trace"))]
    pub(super) fn land_prompt_into(
        &self,
        device: &mut Device,
        (start, wave): (usize, usize),
        prompt: &attention::prompt::Attended,
        rows: &mut Projected<f32>,
    ) {
        let attended = prompt.view().tile::<m![Prompt], 64, m![
            Prompt = 64 # Prompt,
            AttentionHeads,
            AttentionGroups,
            AttentionHead
        ]>(64 * (wave - start));
        // SAFETY: the 64-row tile keeps the chunk's row stride, and its row-major heads flatten
        // Heads/Groups/Head in wire order.
        let attended = unsafe { attended.reshape::<Chips, m![Rows # Prompt, Width]>() };
        attended.to_dm_view(&mut device.tdma, rows.view_mut());
    }

    /// `issue_prompt` of wave `at`, whose attended rows `land_prompt_into` landed.
    #[cfg(not(feature = "trace"))]
    pub(super) fn issue_landed(
        &self,
        device: &mut Device,
        factors: &Factors,
        (rows, at): (&Projected<f32>, usize),
        state: &State,
    ) -> Pending {
        let filled = self.weight.fill(device);
        let rounded = round(device, rows);
        let projection = Projection {
            partial: filled.project(device, (&rounded, &factors.output)),
            at,
        };
        self.send(device, projection, state)
    }

    /// The O partial of landed `rows`, rounded into `rounded` and written through `output`: a
    /// rounding buffer kept across the chunk's waves cannot take a buffer a DMA still reads.
    #[cfg(not(feature = "trace"))]
    pub(super) fn project_into<E: M>(
        &self,
        device: &mut Device,
        factors: &Factors,
        (rows, rounded): (Projected<f32>, &mut Projected<bf16>),
        output: DmTensorViewMut<'_, bf16, Chips, output::Cluster, output::Reduced, E>,
    ) {
        let filled = self.weight.fill(device);
        round_into(device, &rows, rounded.view_mut());
        filled.project_into(device, (rounded, &factors.output), output);
    }

    /// Wave `at`'s O partial of both chips, sent with another partial, added to its `hidden`
    /// rows, and their copy into the norm's layout issued.
    #[cfg(not(feature = "trace"))]
    pub(super) fn sum_sent(
        &self,
        device: &mut Device,
        (sent, hidden, at): (&super::exchange::Sent, Residual, usize),
    ) -> Summed {
        let hidden = super::exchange::finish(device, sent, super::exchange::O, &hidden);
        Summed {
            rows: hidden.to_dm(&mut device.tdma),
            hidden,
            at,
        }
    }

    /// `project` of prompt wave `wave`, read from the chunk's attended rows.
    pub(super) fn project_prompt(
        &self,
        device: &mut Device,
        factors: &Factors,
        (start, wave): (usize, usize),
        state: &State,
        #[cfg(feature = "trace")] trace: &mut Trace,
    ) -> Projection {
        let attended = state.prompt.view().tile::<m![Prompt], 64, m![
            Prompt = 64 # Prompt,
            AttentionHeads,
            AttentionGroups,
            AttentionHead
        ]>(64 * (wave - start));
        // SAFETY: the 64-row tile keeps the chunk's row stride, and its row-major heads flatten
        // Heads/Groups/Head in wire order.
        let attended = unsafe { attended.reshape::<Chips, m![Rows # Prompt, Width]>() };
        #[cfg(feature = "trace")]
        let copy: Copied<f32> = attended.clone().to_dm(&mut device.tdma);
        let rows: Projected<f32> = attended.to_dm(&mut device.tdma);
        self.project(
            device,
            factors,
            (rows, wave),
            #[cfg(feature = "trace")]
            (trace, copy),
        )
    }

    /// The pending wave's exchanged partial added to its hidden rows, normalized and quantized
    /// into its activation, then stored into `residuals`: that store queues behind the
    /// activation's, which the feed-forward waits for, and only the down sweep reads it.
    pub(super) fn complete(
        &self,
        device: &mut Device,
        factors: &Factors,
        pending: Pending,
        stores: (&mut HiddenRows, &mut mlp::Input),
        #[cfg(feature = "trace")] trace: &mut Trace,
    ) {
        let summed = self.sum(device, pending);
        self.normalize(
            device,
            factors,
            summed,
            stores,
            #[cfg(feature = "trace")]
            trace,
        );
    }

    /// `complete` of a decode wave, whose activation stores only the row groups its `live` holds;
    /// `copied` issues work behind the norm's copy, under the norm.
    pub(super) fn complete_live(
        &self,
        device: &mut Device,
        factors: &Factors,
        (pending, copied): (Pending, impl FnOnce(&mut Device)),
        stores: (&mut HiddenRows, &mut mlp::Input),
        #[cfg(feature = "trace")] trace: &mut Trace,
    ) {
        let Pending {
            sent,
            hidden,
            at,
            live,
        } = pending;
        let hidden = residual::finish(device, sent, &hidden);
        let rows = live.copy(device, &hidden);
        copied(device);
        let (residuals, activations) = stores;
        let activation = norm::eight::quantize(device, &rows, (&self.norm.gain, &factors.norm));
        live.store_activation(device, &activation, (activations, at));
        hidden
            .view()
            .to_hbm_view(&mut device.tdma, wave_mut(residuals, at));
        #[cfg(feature = "trace")]
        trace.residual(device, (residuals, self.layer.1));
    }

    /// The pending prompt wave's exchanged partial added to its hidden rows, and their copy into
    /// the norm's layout issued, so work issued before `normalize` runs under the copy.
    pub(super) fn sum(
        &self,
        device: &mut Device,
        Pending {
            sent, hidden, at, ..
        }: Pending,
    ) -> Summed {
        let hidden = residual::finish(device, sent, &hidden);
        Summed {
            rows: hidden.to_dm(&mut device.tdma),
            hidden,
            at,
        }
    }

    /// A summed wave normalized and quantized into its activation, then its hidden rows stored
    /// into `residuals` (see `complete`).
    pub(super) fn normalize(
        &self,
        device: &mut Device,
        factors: &Factors,
        Summed { rows, hidden, at }: Summed,
        (residuals, activations): (&mut HiddenRows, &mut mlp::Input),
        #[cfg(feature = "trace")] trace: &mut Trace,
    ) {
        norm::eight::quantize(device, &rows, (&self.norm.gain, &factors.norm))
            .view()
            .to_hbm_view(
                &mut device.tdma,
                activations
                    .view_mut()
                    .tile::<m![Waves], 1, m![Waves = 1 #{!} Waves, { mlp::Grouped }]>(at),
            );
        hidden
            .view()
            .to_hbm_view(&mut device.tdma, wave_mut(residuals, at));
        #[cfg(feature = "trace")]
        trace.residual(device, (residuals, self.layer.1));
    }
}
