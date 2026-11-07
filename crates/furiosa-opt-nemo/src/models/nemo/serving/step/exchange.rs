//! One TP2 transfer for two waves' partials: a later wave's O partial and an earlier wave's down
//! partial travel together, so a fused chunk iteration has one rendezvous and one completion.
use super::*;

axes![Kind = 2];

/// Which partial of a pair: the O projection's or the down projection's.
pub(super) const O: usize = 0;
pub(super) const DOWN: usize = 1;
/// Both partials of a pair in the step's residual layout.
pub(super) type Partials = DmTensor<bf16, Chips, Cluster, Slices, m![Kind, { Time }, { Packet }]>;

/// A pair on its way to the peer chip, received in place of the peer's.
pub(super) struct Sent {
    partials: Partials,
    received: Partials,
}

/// Sends `partials` to the peer chip; the sums wait only for `finish`.
pub(super) fn send(device: &mut Device, partials: Partials) -> Sent {
    let mut received = Partials::new();
    crate::ops::collective::AllReduce::<crate::ops::layout::axes::Parallel<2>>::default().stage(
        device,
        crate::ops::layout::Dm::<crate::compose::Tile<Rows, Rows>, _> {
            tiles: crate::compose::Tiles::over(Rows::SIZE),
            value: (partials.view(), received.view_mut()),
        },
    );
    Sent { partials, received }
}

/// Partial `kind` of both chips plus `residual`, rounded as `residual::finish` rounds.
pub(super) fn finish(
    device: &mut Device,
    sent: &Sent,
    kind: usize,
    residual: &Residual,
) -> Residual {
    const { assert!(Time::SIZE % 20 == 0) };
    let partial = sent
        .partials
        .view()
        .tile::<m![Kind], 1, m![Kind = 1 # 2, { Time }, { Packet }]>(kind);
    let received = sent
        .received
        .view()
        .tile::<m![Kind], 1, m![Kind = 1 # 2, { Time }, { Packet }]>(kind);
    let sum = crate::ops::collective::AllReduce::<crate::ops::layout::axes::Parallel<2>>::default()
        .complete::<crate::compose::Tile<Rows, Rows>, Chips, Cluster, Slices, _, Time, Packet>(
            &mut *device,
            crate::ops::layout::Dm::<crate::compose::Tile<Rows, Rows>, _> {
                tiles: crate::compose::Tiles::over(Rows::SIZE),
                value: (partial, received),
            },
        )
        .value;
    crate::ops::add::Add
        .stage::<crate::compose::Tile<Rows, Rows>, Chips, Cluster, Slices, _, Time, Packet>(
            device,
            crate::ops::layout::Dm::<crate::compose::Tile<Rows, Rows>, _> {
                tiles: crate::compose::Tiles::over(Rows::SIZE),
                value: (sum.view(), residual.view()),
            },
        )
        .value
        .commit()
}

/// The view partial `kind` of a pair is written through.
pub(super) fn slot(
    partials: &mut Partials,
    kind: usize,
) -> DmTensorViewMut<'_, bf16, Chips, Cluster, Slices, m![Kind = 1 #{!} 2, { Time }, { Packet }]> {
    partials
        .view_mut()
        .tile::<m![Kind], 1, m![Kind = 1 #{!} 2, { Time }, { Packet }]>(kind)
}

/// A fused chunk's opening wave: the first whose iteration sends the O partial of the wave two
/// ahead. Every wave before it ran its tail with its attention or in a prompt-only step's pair.
pub(super) trait Opening: Copy {
    fn wave(self) -> usize;
}

/// Wave 1, after one decode wave or none: a constant, so the pair loop's transfers address their
/// waves from the loop counter alone.
#[derive(Clone, Copy)]
pub(super) struct Second;

impl Opening for Second {
    fn wave(self) -> usize {
        1
    }
}

/// The chunk's first wave, past this many decode waves (two or more).
#[derive(Clone, Copy)]
pub(super) struct Past(pub(super) usize);

impl Opening for Past {
    fn wave(self) -> usize {
        self.0
    }
}

/// A chunk of four waves or more past its opening wave, after wave 0's gate/up: each wave's down
/// inside the next wave's gate/up and each chunk wave's tail two waves ahead, one TP2 transfer an
/// iteration carrying the O partial of the wave two ahead and the down partial of the wave two
/// behind, so the iteration has one rendezvous and one completion and every Main command before it
/// runs under the link. Decode waves before the opening one have no tail left to send.
pub(super) fn chunk<O: Opening>(
    device: &mut Device,
    (tail, factors, pass, prompt, projection): (
        &Tail<'_>,
        &tail::Factors,
        &mlp::Pass<'_>,
        &attention::prompt::Attended,
        tail::Projection,
    ),
    (residuals, activations, product): (&mut HiddenRows, &mut mlp::Input, &mut mlp::Product),
    (hidden, partials): (Hidden<'_>, &mut mlp::Output),
    (swept, opening, decode, waves): (mlp::Swept<'_>, O, usize, usize),
) {
    let mlp::Swept {
        mut raw,
        mut activation,
        panels,
    } = swept;
    let opening = opening.wave();
    // Decode waves 1 .. opening: gate/up and the previous wave's down, its partial sent alone.
    for wave in 1..opening {
        let first = activation.fill(device, (0, 0));
        let advanced = compute(
            device,
            (pass, &panels),
            (&mut *product, &*activations),
            (&mut activation, &mut raw),
            (first, wave, true),
        );
        let partial = pass.close(device, advanced);
        let sent = pass.send(device, partial, (&*residuals, wave - 1));
        pass.reload(device, (&*product, wave), &mut raw);
        pass.complete(device, sent, (&mut *hidden.output, wave - 1));
    }
    // The opening wave: the O partial of the wave two ahead alone.
    let first = activation.fill(device, (0, 0));
    let rows_in: Residual = hidden.wave(opening + 2).to_dm(&mut device.tdma);
    let rows = tail.land_prompt(device, (decode, opening + 3), prompt);
    let pending = tail.send_landed(device, (projection, rows_in));
    let advanced = compute(
        device,
        (pass, &panels),
        (&mut *product, &*activations),
        (&mut activation, &mut raw),
        (first, opening, true),
    );
    let mut sending = exchange::Partials::new();
    let mut rounded = Projected::<bf16>::new();
    tail.project_into(
        device,
        factors,
        (rows, &mut rounded),
        exchange::slot(&mut sending, exchange::O),
    );
    let summed = tail.sum(device, pending);
    // A wave's residual rows land before the norm's store into `residuals`, which they would wait
    // for.
    let mut residual: Residual = super::wave(&*residuals, opening - 1).to_dm(&mut device.tdma);
    // The product's reload reads nothing the norm writes, so it moves while the norm computes.
    pass.reload(device, (&*product, opening), &mut raw);
    tail.normalize(
        device,
        factors,
        summed,
        (&mut *residuals, &mut *activations),
    );
    pass.close_into(
        device,
        advanced,
        exchange::slot(&mut sending, exchange::DOWN),
    );
    // Waves opening + 1 ..= waves - 4: the O partial of wave + 2 with the down partial of wave - 2.
    // The attended rows land ahead of the pair, so the O projection waits for them and not the link.
    for at in 0..waves - (opening + 4) {
        let wave = opening + 1 + at;
        let first = activation.fill(device, (0, 0));
        let rows = tail.land_prompt(device, (decode, wave + 3), prompt);
        let sent = exchange::send(device, sending);
        let rows_in: Residual = hidden.wave(wave + 2).to_dm(&mut device.tdma);
        // Wave + 3's O projection runs under the pair, ahead of this wave's gate/up and the
        // previous wave's down, so the next pair waits on neither its fill nor its command.
        let mut next = exchange::Partials::new();
        tail.project_into(
            device,
            factors,
            (rows, &mut rounded),
            exchange::slot(&mut next, exchange::O),
        );
        let advanced = compute(
            device,
            (pass, &panels),
            (&mut *product, &*activations),
            (&mut activation, &mut raw),
            (first, wave, true),
        );
        let summed = tail.sum_sent(device, (&sent, rows_in, wave + 2));
        exchange::finish(device, &sent, exchange::DOWN, &residual)
            .view()
            .to_hbm_view(&mut device.tdma, super::wave_mut(hidden.output, wave - 2));
        residual = super::wave(&*residuals, wave - 1).to_dm(&mut device.tdma);
        pass.reload(device, (&*product, wave), &mut raw);
        tail.normalize(
            device,
            factors,
            summed,
            (&mut *residuals, &mut *activations),
        );
        pass.close_into(device, advanced, exchange::slot(&mut next, exchange::DOWN));
        sending = next;
    }
    // Wave `waves - 3`: the last O partial with a down partial.
    let wave = waves - 3;
    let first = activation.fill(device, (0, 0));
    let sent = exchange::send(device, sending);
    let rows_in: Residual = hidden.wave(wave + 2).to_dm(&mut device.tdma);
    let advanced = compute(
        device,
        (pass, &panels),
        (&mut *product, &*activations),
        (&mut activation, &mut raw),
        (first, wave, true),
    );
    let summed = tail.sum_sent(device, (&sent, rows_in, wave + 2));
    exchange::finish(device, &sent, exchange::DOWN, &residual)
        .view()
        .to_hbm_view(&mut device.tdma, super::wave_mut(hidden.output, wave - 2));
    let mut residual: Residual = super::wave(&*residuals, wave - 1).to_dm(&mut device.tdma);
    pass.reload(device, (&*product, wave), &mut raw);
    tail.normalize(
        device,
        factors,
        summed,
        (&mut *residuals, &mut *activations),
    );
    let mut partial = pass.close(device, advanced);
    // Waves `waves - 2` and `waves - 1`: down partials alone; the first lands the last activation.
    #[unroll]
    for at in 0..2 {
        let wave = waves - 2 + at;
        let first = activation.fill(device, (0, 0));
        let sent = pass.send_rows(device, (partial, residual));
        residual = super::wave(&*residuals, wave - 1).to_dm(&mut device.tdma);
        let advanced = compute(
            device,
            (pass, &panels),
            (&mut *product, &*activations),
            (&mut activation, &mut raw),
            (first, wave, at == 0),
        );
        pass.complete(device, sent, (&mut *hidden.output, wave - 2));
        partial = pass.close(device, advanced);
        mlp::down::record(device, &partial, partials, wave - 1);
        pass.reload(device, (&*product, wave), &mut raw);
    }
    let sent = pass.send_rows(device, (partial, residual));
    let last = pass.last(
        device,
        (&panels, &raw),
        Some((sent, &mut *hidden.output, waves - 2)),
    );
    mlp::down::record(device, &last, partials, waves - 1);
    let sent = pass.send(device, last, (&*residuals, waves - 1));
    pass.complete(device, sent, (&mut *hidden.output, waves - 1));
}

/// The MLP work of iteration `wave` after its first register fill: its gate/up with the next
/// wave's activation landed once the last fill has read it (when `land`), then the previous
/// wave's down with this wave's product stored under it.
pub(super) fn compute(
    device: &mut Device,
    (pass, panels): (&mlp::Pass<'_>, &mlp::down::Panels<'_>),
    (product, inputs): (&mut mlp::Product, &mlp::Input),
    (activation, raw): (&mut mlp::producer::Activation, &mut mlp::down::Raw),
    (first, wave, land): (mlp::producer::Register, usize, bool),
) -> mlp::Advanced {
    let opened = pass
        .open::<crate::models::nemo::serving::mlp::producer::Whole>(device, (activation, first));
    let turned = pass.turn(device, opened, activation);
    if land {
        activation.land_into(device, inputs, wave + 1);
    }
    let prefilled = pass.prefill(device, raw);
    let contracted = pass.contract_last(device, turned);
    let lefted = pass.left(device, prefilled, panels);
    let prepared = pass.prepare(device, contracted);
    let advanced = pass.right(device, lefted, panels);
    pass.store_prepared(device, prepared, (product, wave));
    advanced
}

/// Where a fused chunk reads its waves' hidden rows (`input`, or `output` when the layer updates
/// them in place) and writes its layer output.
pub(super) struct Hidden<'a> {
    pub(super) input: Option<&'a HiddenRows>,
    pub(super) output: &'a mut HiddenRows,
}

impl Hidden<'_> {
    fn wave(
        &self,
        wave: usize,
    ) -> HbmTensorView<'_, bf16, Chips, m![Waves = 1 # Waves, { Blocked }]> {
        match self.input {
            Some(input) => super::wave(input, wave),
            None => super::wave(&*self.output, wave),
        }
    }
}
