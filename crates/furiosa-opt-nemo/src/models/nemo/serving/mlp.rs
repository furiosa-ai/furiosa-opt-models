use crate::Module;
use furiosa_opt_std::prelude::*;

use crate::models::nemo::serving::step::{self, All, Four, Groups, HiddenRows, Live, One, Two};
use crate::models::nemo::serving::{Chips, residual, scale, stack};

pub mod block;
pub mod down;
pub mod dpanel;
pub mod producer;
pub mod weight;

axes![
    Rows = 64,
    Hidden = 5120,
    Intermediate = 7168,
    Capacity = 4096,
    Waves = 64,
    Term = 2,
    Branch = 2
];
const _: () = assert!(Capacity::SIZE == Waves::SIZE * Rows::SIZE);
type Cluster = m![Intermediate / 3584];
type Slice = m![Intermediate % 3584 / 56, Rows / 16];
type Time = m![Intermediate % 56, Rows % 16 / 8];
type Packet = m![Rows % 8];
type Values<D> = DmTensor<D, Chips, Cluster, Slice, m![{ Time }, { Packet }]>;
/// One hidden half's gate (branch 0) and up (branch 1) contractions of a wave.
type Branches = DmTensor<f32, Chips, Cluster, Slice, m![Branch, { Time }, { Packet }]>;
/// The branches of both hidden halves.
type Halves = (Branches, Branches);
type BranchView<'a> =
    DmTensorView<'a, f32, Chips, Cluster, Slice, m![Branch = 1 # 2, { Time }, { Packet }]>;

/// Branch `branch` (0 gate, 1 up) of one hidden half's contractions.
fn branch(values: &Branches, branch: usize) -> BranchView<'_> {
    values
        .view()
        .tile::<m![Branch], 1, m![Branch = 1 # 2, { Time }, { Packet }]>(branch)
}
/// Two tensors read in 28-step blocks of each, so the VE cache holds a block: interleaved a packet
/// apart it holds one vector and reads the streams several times slower a flit.
type Interleaved = m![
    Intermediate % 56 / 14,
    Term,
    Intermediate % 14,
    Rows % 16 / 8
];
pub type Count = HbmTensor<i32, Chips, m![1]>;
pub type Input = producer::Input;
pub use producer::Grouped;
pub type Output = down::Output;
/// The FP8 gate·up product of every wave, as the down projection reloads it.
pub type Product = HbmTensor<
    f8e4m3,
    Chips,
    m![
        Waves,
        Rows / 32,
        Intermediate / 8,
        Rows % 32 / 16,
        Intermediate % 8,
        Rows % 16
    ],
>;

/// 1 if a step of `rows` live rows has three waves or more and at most 32 live rows in its last,
/// which then contracts half its gate/up register (`producer::Half`), else 0.
fn half_last(rows: usize) -> usize {
    let waves = (rows + Rows::SIZE - 1) / Rows::SIZE;
    ((waves - 2).min(1) * (1 - (rows - Rows::SIZE * (waves - 1) - 1) / 32)).min(1)
}

/// One layer's TP2 feed-forward shard, reading its weights where the stack holds them.
pub struct Model<'a> {
    pub layers: &'a stack::Layers<'a>,
    pub index: usize,
}

/// The gate halves `begin` finds landed: half 0, and half 1 with its count (0 or 1) where the
/// step may land it ahead; with `None`, `begin` lands half 1 into a buffer of its own.
pub type Gates<'f> = (&'f weight::Landed, Option<(&'f mut weight::Landed, usize)>);

/// The staged gate and up weights of both hidden halves, kept for every wave of a step.
struct Staged([weight::GateUp; 2]);

/// A wave's gate and up contractions over both hidden halves, before their product.
pub struct Contracted(Halves);

/// The next layer's work a decode sweep issues: one wave's K/V weight, or on two waves or more the
/// rows it has finished (every earlier wave's output is final in `hidden` once the last down ran).
pub trait Ahead {
    /// Work that reads nothing the sweep writes, issued as the sweep starts.
    fn prefetch(&mut self, device: &mut Device);

    /// A one-wave step's K/V weight, behind the second down panel: the queue idles there until the
    /// down's cross.
    fn land_pairs(&mut self, device: &mut Device);

    /// Wave 0's finished rows landed once every wave but the last completes, ahead of the last
    /// send: the in-order transfer queue holds whatever follows that send until it completes.
    fn land(&mut self, device: &mut Device, hidden: &HiddenRows);

    /// Work on waves `0..waves - 1`, issued while the last wave's partial is on its way to the
    /// peer chip.
    fn issue(&mut self, device: &mut Device, rows: (&HiddenRows, usize));
}

/// A sweep with nothing to issue for the next layer.
impl Ahead for () {
    fn prefetch(&mut self, _: &mut Device) {}

    fn land_pairs(&mut self, _: &mut Device) {}

    fn land(&mut self, _: &mut Device, _: &HiddenRows) {}

    fn issue(&mut self, _: &mut Device, _: (&HiddenRows, usize)) {}
}

/// One layer's feed-forward weights in data memory once wave 0 has projected: the staged gate and
/// up halves, the scale registers and the first down panel.
pub struct Pass<'a> {
    model: &'a Model<'a>,
    factors: Factors,
    staged: Staged,
    panel: down::Panel,
}

/// The feed-forward's scales in registers, loaded once a layer for every wave: a load per wave
/// queues on Sub behind that wave's register fills, and the in-order stream waits there.
struct Factors {
    gate: scale::Epilogue<Cluster, Slice, Chips>,
    up: scale::Epilogue<Cluster, Slice, Chips>,
    /// The down input scale, which the product is quantized by.
    divisor: scale::InputScale<Chips, Cluster, Slice>,
    down: scale::Epilogue<down::Half, down::Reduced, Chips>,
}

impl Factors {
    fn load(device: &mut Device, scales: &scale::Scales<Chips>) -> Self {
        Self {
            gate: scale::Epilogue {
                input: scales.get(device, 0),
                weight: scales.get(device, 1),
            },
            up: scale::Epilogue {
                input: scales.get(device, 2),
                weight: scales.get(device, 3),
            },
            divisor: crate::ops::linear::Affine::new((), scales.get(device, 4)),
            down: scale::Epilogue {
                input: scales.get(device, 4),
                weight: scales.get(device, 5),
            },
        }
    }
}

impl<'a> Model<'a> {
    /// The first gate half, for `begin`, landed into `first` where the step has the transfer
    /// queue to spare.
    pub fn land_first(&self, device: &mut Device, first: &mut weight::First) {
        self.gate().land_into(device, (0, &mut first.low));
    }

    /// The layer's gate weight.
    pub fn gate(&self) -> weight::Weight<'a> {
        weight::Weight(&self.layers.gate, self.index)
    }

    /// Gate and up of wave 0 through `product`, with the layer's spread feed-forward `scales`. The
    /// weights land once a step: wave 0 projects each half as it lands, and the staged halves serve
    /// every later wave.
    pub fn begin(
        &'a self,
        device: &mut Device,
        (input, first, scales): (&Input, Gates<'_>, &scale::Scales<Chips>),
        product: &mut Product,
    ) -> Pass<'a> {
        let activation = producer::Activation::land(device, input, 0);
        let (mut halves, staged) = self.stream(device, &activation, first);
        // The first down panel streams under the first wave's last up projection and later waves.
        let panel = self.projection().land(device);
        activation.project(device, (&staged.0[1], 1), (1, &mut halves.1));
        let factors = Factors::load(device, scales);
        let prepared = prepare(device, &factors, &halves);
        self.store(device, &factors.divisor, prepared, (product, 0));
        Pass {
            model: self,
            factors,
            staged,
            panel,
        }
    }

    fn projection(&self) -> down::Projection<'_> {
        down::Projection {
            layers: self.layers,
            index: self.index,
        }
    }
}

impl Pass<'_> {
    /// Gate and up of wave `wave`, contracting both against one row fill per hidden half.
    pub fn project(
        &self,
        device: &mut Device,
        input: &Input,
        (product, wave): (&mut Product, usize),
    ) {
        let activation = producer::Activation::land(device, input, wave);
        self.project_landed(device, &activation, (product, wave));
    }

    /// `project` of an activation already landed, so its transfer can be issued ahead of others.
    pub fn project_landed(
        &self,
        device: &mut Device,
        activation: &producer::Activation,
        (product, wave): (&mut Product, usize),
    ) {
        let contracted = self.contract(device, activation);
        self.store(device, contracted, (product, wave));
    }

    /// Gate and up of a landed activation, contracting both against one row fill per hidden
    /// half; `store` turns them into the wave's product.
    pub fn contract(&self, device: &mut Device, activation: &producer::Activation) -> Contracted {
        Contracted(activation.project_halves(device, &self.staged.0))
    }

    /// Wave `wave`'s product of its contracted gate and up into `product`.
    pub fn store(
        &self,
        device: &mut Device,
        Contracted(halves): Contracted,
        (product, wave): (&mut Product, usize),
    ) {
        let prepared = prepare(device, &self.factors, &halves);
        self.model
            .store(device, &self.factors.divisor, prepared, (product, wave));
    }

    /// Down of every live wave, each wave's partials of both chips added to its `residuals` rows
    /// into `hidden`.
    pub fn finish(
        &self,
        device: &mut Device,
        (product, output): (&Product, &mut Output),
        (residuals, hidden, waves): (&HiddenRows, &mut HiddenRows, usize),
    ) {
        self.model.projection().sweep(
            device,
            (product, &self.factors.down),
            (residuals, hidden, output),
            (waves, &self.panel),
        );
    }

    /// Gate and up of waves 1.. and down of every wave into `hidden`: a wave's down runs inside the
    /// next wave's gate/up and its exchange under the one after, so transfers wait under Main work.
    /// A step of one wave sends only the groups holding its `rows` live rows and, if `alone` is 1,
    /// lands `ahead`'s K/V weight; a step of more sends only the last wave's live groups and issues
    /// `ahead` on its earlier waves' finished rows under that exchange.
    pub fn sweep<A: Ahead>(
        &self,
        device: &mut Device,
        (input, product, output): (&Input, &mut Product, &mut Output),
        (residuals, hidden, waves, rows): (&HiddenRows, &mut HiddenRows, usize, usize),
        (ahead, alone): (&mut A, usize),
    ) {
        let Swept {
            mut raw,
            mut activation,
            panels,
        } = self.start(device, (input, &mut *product), waves);
        for _ in 0..(waves - 1).min(1) {
            ahead.prefetch(device);
        }
        for _ in 0..alone {
            ahead.land_pairs(device);
        }
        let single = 1 - (waves - 1).min(1);
        let live = Live::of(rows);
        for _ in 0..single * live.one {
            self.alone::<One>(
                device,
                (&panels, &raw),
                (residuals, &mut *hidden, &mut *output),
            );
        }
        for _ in 0..single * live.two {
            self.alone::<Two>(
                device,
                (&panels, &raw),
                (residuals, &mut *hidden, &mut *output),
            );
        }
        for _ in 0..single * live.four {
            self.alone::<Four>(
                device,
                (&panels, &raw),
                (residuals, &mut *hidden, &mut *output),
            );
        }
        for _ in 0..single * live.all {
            self.alone::<All>(
                device,
                (&panels, &raw),
                (residuals, &mut *hidden, &mut *output),
            );
        }
        for _ in 0..(waves - 1).min(1) {
            let mut partial = self.fused::<producer::Whole>(
                device,
                (&panels, &mut activation, &mut raw),
                (input, &mut *product, 1, waves),
                None,
            );
            down::record(device, &partial, output, 0);
            for wave in 2..waves - 1 {
                partial = self.fused::<producer::Whole>(
                    device,
                    (&panels, &mut activation, &mut raw),
                    (input, &mut *product, wave, waves),
                    Some((partial, residuals, &mut *hidden)),
                );
                down::record(device, &partial, output, wave - 1);
            }
            // The last wave of a step of three or more: half its gate/up register when at most
            // 32 of its rows are live.
            let half = half_last(rows);
            for _ in 0..half {
                partial = self.fused::<producer::Half>(
                    device,
                    (&panels, &mut activation, &mut raw),
                    (input, &mut *product, waves - 1, waves),
                    Some((partial, residuals, &mut *hidden)),
                );
                down::record(device, &partial, output, waves - 2);
            }
            for _ in 0..((waves - 2).min(1) * (1 - half)).min(1) {
                partial = self.fused::<producer::Whole>(
                    device,
                    (&panels, &mut activation, &mut raw),
                    (input, &mut *product, waves - 1, waves),
                    Some((partial, residuals, &mut *hidden)),
                );
                down::record(device, &partial, output, waves - 2);
            }
            // The last wave's down runs under the previous wave's exchange, which completes after it.
            let sent = self.send(device, partial, (residuals, waves - 2));
            let last = self.last(device, (&panels, &raw), None);
            down::record(device, &last, output, waves - 1);
            self.complete(device, sent, (&mut *hidden, waves - 2));
            ahead.land(device, &*hidden);
            let landed = step::wave(residuals, waves - 1).to_dm(&mut device.tdma);
            let next = Sent {
                exchanged: Live::of(rows - Rows::SIZE * (waves - 1)).exchange(device, last),
                rows: landed,
            };
            ahead.issue(device, (&*hidden, waves));
            self.complete(device, next, (hidden, waves - 1));
        }
    }

    /// Wave 0's reloaded product, wave 1's activation and both down panels, the second panel landing
    /// behind that activation so it lands under wave 1's gate/up.
    pub fn start(
        &self,
        device: &mut Device,
        (input, product): (&Input, &mut Product),
        waves: usize,
    ) -> Swept<'_> {
        let raw = down::reload(device, product, 0);
        let mut activation = producer::Activation::new();
        for _ in 0..(waves - 1).min(1) {
            activation.land_into(device, input, 1);
        }
        Swept {
            raw,
            activation,
            panels: self.model.projection().land_right(device, &self.panel),
        }
    }

    /// The only wave's down, its partial's groups `G` sent to the peer chip, into wave 0 of
    /// `hidden`.
    fn alone<G: Groups>(
        &self,
        device: &mut Device,
        (panels, raw): (&down::Panels<'_>, &down::Raw),
        (residuals, hidden, output): (&HiddenRows, &mut HiddenRows, &mut Output),
    ) {
        let partial = self.down(device, (panels, raw));
        down::record(device, &partial, output, 0);
        let mut rows = down::Partial::new();
        G::land_hidden(device, (residuals, 0), &mut rows);
        let mut exchanged = residual::Exchanged::new(partial);
        G::send(device, &mut exchanged);
        let summed = residual::finish(device, exchanged, &rows);
        G::store_hidden(device, &summed, (hidden, 0));
    }

    /// One wave's down projection alone, from its reloaded product.
    fn down(
        &self,
        device: &mut Device,
        (panels, raw): (&down::Panels<'_>, &down::Raw),
    ) -> down::Partial {
        let relaid = down::relayout(device, raw);
        let low = down::fill(device, &relaid, 0);
        let left = down::contract(device, (panels.left, &low));
        let high = down::fill(device, &relaid, 1);
        let right = down::contract(device, (&panels.right, &high));
        // A decode step's one wave waits on its cross, which moves half the bytes in BF16.
        let sum = down::fold_compact(device, (left, right));
        let crossed = down::cross(device, &sum);
        down::finish(&self.factors.down, device, crossed)
    }

    /// One wave's down projection from its reloaded product, completing `done`'s exchange under
    /// the first down MAC.
    pub fn last(
        &self,
        device: &mut Device,
        (panels, raw): (&down::Panels<'_>, &down::Raw),
        done: Option<(Sent, &mut HiddenRows, usize)>,
    ) -> down::Partial {
        let relaid = down::relayout(device, raw);
        let low = down::fill(device, &relaid, 0);
        let left = down::contract(device, (panels.left, &low));
        let high = down::fill(device, &relaid, 1);
        if let Some((sent, hidden, wave)) = done {
            self.complete(device, sent, (hidden, wave));
        }
        let right = down::contract(device, (&panels.right, &high));
        let sum = down::fold(device, (left, right));
        let crossed = down::cross(device, &sum);
        down::finish(&self.factors.down, device, crossed)
    }

    /// Wave `wave`'s partial on its way to the peer chip, with the residual rows it adds to.
    pub fn send(
        &self,
        device: &mut Device,
        partial: down::Partial,
        (residuals, wave): (&HiddenRows, usize),
    ) -> Sent {
        let rows: down::Partial = step::wave(residuals, wave).to_dm(&mut device.tdma);
        Sent {
            exchanged: residual::exchange(device, partial),
            rows,
        }
    }

    /// `send` with the residual rows already landed.
    pub fn send_rows(
        &self,
        device: &mut Device,
        (partial, rows): (down::Partial, down::Partial),
    ) -> Sent {
        Sent {
            exchanged: residual::exchange(device, partial),
            rows,
        }
    }

    /// A sent wave's partials of both chips added to its residual rows into wave `wave` of `hidden`.
    pub fn complete(
        &self,
        device: &mut Device,
        Sent { exchanged, rows }: Sent,
        (hidden, wave): (&mut HiddenRows, usize),
    ) {
        down::complete(device, (exchanged, &rows), (hidden, wave));
    }

    /// Gate and up of wave `wave` from `activation` with the previous wave's down from `raw` placed
    /// between its contractions; `sent`, the partial of the wave before, is exchanged under them and
    /// completed under the first down MAC. The next wave lands in `activation` once its last fill
    /// has read it, and this wave's product in `raw` after the exchange has left data memory.
    fn fused<P: producer::Part>(
        &self,
        device: &mut Device,
        (panels, activation, raw): (&down::Panels<'_>, &mut producer::Activation, &mut down::Raw),
        (input, product, wave, waves): (&Input, &mut Product, usize, usize),
        sent: Option<(down::Partial, &HiddenRows, &mut HiddenRows)>,
    ) -> down::Partial {
        let first = activation.fill(device, (0, 0));
        // The first fill loads while this chip waits at the exchange's rendezvous.
        let done = match sent {
            Some((partial, residuals, hidden)) => Some((
                self.send(device, partial, (residuals, wave - 2)),
                hidden,
                wave - 2,
            )),
            None => None,
        };
        let relaid = P::relay(device, raw);
        let opened = self.open::<P>(device, (activation, first));
        let advanced = self.advance::<P>(
            device,
            opened,
            (panels, activation, (relaid, raw)),
            (Some(input), product, wave, waves),
            done,
        );
        self.close(device, advanced)
    }

    /// The first hidden half of a wave's gate/up from its filled first register.
    pub fn open<P: producer::Part>(
        &self,
        device: &mut Device,
        (activation, first): (&producer::Activation, producer::Register),
    ) -> Opened {
        let Staged(pairs) = &self.staged;
        let mut halves = (Branches::new(), Branches::new());
        P::begin(device, (&pairs[0], &first), (&mut halves.0, 0));
        let second = activation.fill(device, (0, 1));
        P::end(device, (&pairs[0], &first), (&mut halves.0, 0));
        P::begin(device, (&pairs[0], &second), (&mut halves.0, 1));
        let third = activation.fill(device, (1, 0));
        P::end(device, (&pairs[0], &second), (&mut halves.0, 1));
        Opened { halves, third }
    }

    /// The second hidden half of an opened wave `wave`, then the previous wave's down from `raw`
    /// (relaid ahead if `relaid` holds it), completing `done` under the first down MAC; the next
    /// wave lands in `activation` once its last fill has read it, and this wave's product in `raw`.
    pub fn advance<P: producer::Part>(
        &self,
        device: &mut Device,
        Opened { mut halves, third }: Opened,
        (panels, activation, (relaid, raw)): (
            &down::Panels<'_>,
            &mut producer::Activation,
            (Option<down::Activation>, &mut down::Raw),
        ),
        (input, product, wave, waves): (Option<&Input>, &mut Product, usize, usize),
        done: Option<(Sent, &mut HiddenRows, usize)>,
    ) -> Advanced {
        let Staged(pairs) = &self.staged;
        P::begin(device, (&pairs[1], &third), (&mut halves.1, 0));
        let fourth = activation.fill(device, (1, 1));
        P::end(device, (&pairs[1], &third), (&mut halves.1, 0));
        if let Some(input) = input {
            for _ in 0..(waves - wave - 1).min(1) {
                activation.land_into(device, input, wave + 1);
            }
        }
        let relaid = match relaid {
            Some(relaid) => relaid,
            None => down::relayout(device, raw),
        };
        P::begin(device, (&pairs[1], &fourth), (&mut halves.1, 1));
        let low = down::fill(device, &relaid, 0);
        P::end(device, (&pairs[1], &fourth), (&mut halves.1, 1));
        let left = down::contract(device, (panels.left, &low));
        let high = down::fill(device, &relaid, 1);
        if let Some((sent, hidden, wave)) = done {
            self.complete(device, sent, (hidden, wave));
        }
        let prepared = prepare(device, &self.factors, &halves);
        let right = down::contract(device, (&panels.right, &high));
        let sum = down::fold(device, (left, right));
        let crossed = down::cross(device, &sum);
        self.model
            .store(device, &self.factors.divisor, prepared, (product, wave));
        down::reload_into(device, product, wave, raw);
        Advanced(crossed)
    }

    /// The previous wave's partial of an advanced iteration.
    pub fn close(&self, device: &mut Device, Advanced(crossed): Advanced) -> down::Partial {
        down::finish(&self.factors.down, device, crossed)
    }

    /// `close` into `output`, a view of a wider buffer.
    pub fn close_into<E: M>(
        &self,
        device: &mut Device,
        Advanced(crossed): Advanced,
        output: DmTensorViewMut<'_, bf16, Chips, down::Half, down::Reduced, E>,
    ) {
        down::finish_into(&self.factors.down, device, crossed, output);
    }

    /// The second hidden half's first gate and up contraction of an opened wave, its last
    /// register fill issued behind it.
    pub fn turn(
        &self,
        device: &mut Device,
        Opened { mut halves, third }: Opened,
        activation: &producer::Activation,
    ) -> Turned {
        producer::contract(device, (&self.staged.0[1], &third), (&mut halves.1, 0));
        let fourth = activation.fill(device, (1, 1));
        Turned { halves, fourth }
    }

    /// A turned wave's last gate and up contraction, completing its gate/up.
    pub fn contract_last(
        &self,
        device: &mut Device,
        Turned { mut halves, fourth }: Turned,
    ) -> Contracted {
        producer::contract(device, (&self.staged.0[1], &fourth), (&mut halves.1, 1));
        Contracted(halves)
    }

    /// A prefilled wave's first panel contraction, the second panel's fill issued under it.
    pub fn left(
        &self,
        device: &mut Device,
        Prefilled { relaid, low }: Prefilled,
        panels: &down::Panels<'_>,
    ) -> Lefted {
        Lefted {
            left: down::contract(device, (panels.left, &low)),
            high: down::fill(device, &relaid, 1),
        }
    }

    /// The SiLU and scaled up of a contracted wave.
    pub fn prepare(&self, device: &mut Device, Contracted(halves): Contracted) -> Prepared {
        prepare(device, &self.factors, &halves)
    }

    /// A wave's second panel contraction and both panels' sum crossing the clusters.
    pub fn right(
        &self,
        device: &mut Device,
        Lefted { left, high }: Lefted,
        panels: &down::Panels<'_>,
    ) -> Advanced {
        let right = down::contract(device, (&panels.right, &high));
        let sum = down::fold(device, (left, right));
        Advanced(down::cross(device, &sum))
    }

    /// A prepared wave's product into wave `wave` of `product`.
    pub fn store_prepared(
        &self,
        device: &mut Device,
        prepared: Prepared,
        (product, wave): (&mut Product, usize),
    ) {
        self.model
            .store(device, &self.factors.divisor, prepared, (product, wave));
    }

    /// The second hidden half of an opened wave's gate/up.
    pub fn contract_rest(
        &self,
        device: &mut Device,
        opened: Opened,
        activation: &producer::Activation,
    ) -> Contracted {
        let turned = self.turn(device, opened, activation);
        self.contract_last(device, turned)
    }

    /// A reloaded product relaid with its first panel's register filled.
    pub fn prefill(&self, device: &mut Device, raw: &down::Raw) -> Prefilled {
        let relaid = down::relayout(device, raw);
        let low = down::fill(device, &relaid, 0);
        Prefilled { relaid, low }
    }

    /// The down of a prefilled wave with the contracted wave `wave`'s product finished under it
    /// and stored into `product`.
    pub fn down_with(
        &self,
        device: &mut Device,
        (Prefilled { relaid, low }, Contracted(halves)): (Prefilled, Contracted),
        panels: &down::Panels<'_>,
        (product, wave): (&mut Product, usize),
    ) -> Advanced {
        let left = down::contract(device, (panels.left, &low));
        let high = down::fill(device, &relaid, 1);
        let prepared = prepare(device, &self.factors, &halves);
        let right = down::contract(device, (&panels.right, &high));
        let sum = down::fold(device, (left, right));
        let crossed = down::cross(device, &sum);
        self.model
            .store(device, &self.factors.divisor, prepared, (product, wave));
        Advanced(crossed)
    }

    /// Wave `wave`'s product into `raw`, for its down in the next iteration.
    pub fn reload(
        &self,
        device: &mut Device,
        (product, wave): (&Product, usize),
        raw: &mut down::Raw,
    ) {
        down::reload_into(device, product, wave, raw);
    }
}

/// A wave's gate/up with three of its four register fills contracted.
pub struct Turned {
    halves: Halves,
    fourth: producer::Register,
}

/// A wave's first down panel contracted and the second panel's fill issued.
pub struct Lefted {
    left: down::Values,
    high: down::Register,
}

/// A reloaded product relaid for the down fills, the first one issued.
pub struct Prefilled {
    relaid: down::Activation,
    low: down::Register,
}

/// The down sweep's buffers: the reloaded product, the next activation and both down panels.
pub struct Swept<'p> {
    pub raw: down::Raw,
    pub activation: producer::Activation,
    pub panels: down::Panels<'p>,
}

/// A wave's gate/up with its first hidden half contracted and the second's first fill issued.
pub struct Opened {
    halves: Halves,
    third: producer::Register,
}

/// A fused iteration whose previous wave's partial is crossing the clusters.
pub struct Advanced(down::Crossed);

/// A wave's partial on its way to the peer chip, with the residual rows it adds to.
pub struct Sent {
    exchanged:
        residual::Exchanged<down::Half, down::Reduced, down::PartialTime, down::PartialPacket>,
    rows: down::Partial,
}

impl Model<'_> {
    /// Projects the first wave's gate halves 0 and 1 and up half 0 as each lands while the
    /// previous one contracts, and stages up half 1; every staged half is kept. With gate half 1
    /// landed in `first`, both up halves land at once.
    fn stream(
        &self,
        device: &mut Device,
        activation: &producer::Activation,
        (low, ahead): Gates<'_>,
    ) -> (Halves, Staged) {
        let gate = weight::Weight(&self.layers.gate, self.index);
        let up = weight::Weight(&self.layers.up, self.index);
        let mut pairs = [weight::GateUp::new(), weight::GateUp::new()];
        let mut halves = (Branches::new(), Branches::new());
        low.stage_into(device, (&mut pairs[0], 0));
        let mut third = weight::Landed::new();
        let mut fourth = weight::Landed::new();
        let mut own = weight::Landed::new();
        let (high, count) = match ahead {
            Some((high, count)) => (high, Some(count)),
            None => (&mut own, None),
        };
        gate.land_unless(device, (1, &mut *high), count);
        if let Some(count) = count {
            for _ in 0..count {
                up.land_into(device, (0, &mut third));
                up.land_into(device, (1, &mut fourth));
            }
        }
        activation.project(device, (&pairs[0], 0), (0, &mut halves.0));
        high.stage_into(device, (&mut pairs[1], 0));
        up.land_unless(device, (0, &mut third), count);
        activation.project(device, (&pairs[1], 0), (1, &mut halves.1));
        third.stage_into(device, (&mut pairs[0], 1));
        up.land_unless(device, (1, &mut fourth), count);
        activation.project(device, (&pairs[0], 1), (0, &mut halves.0));
        fourth.stage_into(device, (&mut pairs[1], 1));
        (halves, Staged(pairs))
    }

    fn store(
        &self,
        device: &mut Device,
        divisor: &scale::InputScale<Chips, Cluster, Slice>,
        prepared: Prepared,
        (product, wave): (&mut Product, usize),
    ) {
        let value = finalize(device, divisor, prepared);
        publish(device, value, (product, wave));
    }
}

/// One wave's E4M3 product in data memory, laid out as `Product` stores it.
type Stored = DmTensor<
    f8e4m3,
    Chips,
    Cluster,
    m![Intermediate % 3584 / 56, Rows / 32, 2],
    m![
        Intermediate % 56 / 8,
        Rows % 32 / 16,
        Intermediate % 8,
        Rows % 16
    ],
>;

/// The product of a prepared wave, quantized by the down input scale `divisor`.
fn finalize(
    device: &mut Device,
    divisor: &scale::InputScale<Chips, Cluster, Slice>,
    Prepared(gate, up): Prepared,
) -> Stored {
    let value: Values<bf16> = device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(gate.view(), up.view())
        .fetch::<Interleaved, Packet>()
        .fetch_cast::<f32>()
        .collect::<Interleaved, Packet>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, Time>()
        .vector_narrow_split::<m![{ Time }, { Packet } / 4], m![{ Packet } % 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_widen_concat::<Time, Packet>()
        .vector_final()
        .cast::<bf16, m![{ Packet } # 16]>()
        .commit_trim::<Packet>()
        .commit();
    let value = scale::quantize::<Cluster, Slice, Time, Packet, Chips>(device, &value, divisor);
    device
            .main
            .begin(value.view())
            .fetch::<m![Intermediate % 56], m![Rows % 16]>()
            .switch::<m![Intermediate % 3584 / 56, Rows / 32, 2], m![Intermediate % 56, Rows % 32 / 16]>(
                SwitchConfig::Broadcast01 {
                    slice0: 2,
                    slice1: 1,
                    time0: 1,
                },
            )
            .collect::<m![Intermediate % 56, Rows % 32 / 16], m![Rows % 16 # 32]>()
            .commit_trim::<m![Rows % 16]>()
            .commit()
}

/// Wave `wave`'s product into `product`.
fn publish(device: &mut Device, value: Stored, (product, wave): (&mut Product, usize)) {
    value.view().to_hbm_view(
        &mut device.tdma,
        product.view_mut().tile::<m![Waves], 1, m![
            Waves = 1 #{!} Waves,
            Rows / 32,
            Intermediate / 8,
            Rows % 32 / 16,
            Intermediate % 8,
            Rows % 16
        ]>(wave),
    );
}

/// SiLU of the scaled gate and the scaled up of one wave, each rounded to BF16 from its two
/// hidden halves; `Model::store` multiplies them.
pub struct Prepared(Values<bf16>, Values<bf16>);

fn prepare(device: &mut Device, factors: &Factors, halves: &Halves) -> Prepared {
    let gate = activate(device, halves, &factors.gate);
    let value = device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(branch(&halves.0, 1), branch(&halves.1, 1))
        .fetch::<Interleaved, Packet>()
        .collect::<Interleaved, Packet>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, Time>()
        .vector_narrow_split::<m![{ Time }, { Packet } / 4], m![{ Packet } % 4]>()
        .vector_fp_zip(FpBinaryOp::AddF);
    let up = crate::ops::linear::Affine::new((), &factors.up)
        .forward(crate::ops::layout::Dm::<producer::contraction::Single, _> {
            tiles: crate::compose::Tiles::over(Rows::SIZE),
            value,
        })
        .value
        .vector_widen_concat::<Time, Packet>()
        .vector_final()
        .cast::<bf16, m![{ Packet } # 16]>()
        .commit_trim::<Packet>()
        .commit();
    Prepared(gate, up)
}

/// Sums the gate's two hidden-half contractions.
fn fold(device: &mut Device, (left, right): &Halves) -> Values<f32> {
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(branch(left, 0), branch(right, 0))
        .fetch::<Interleaved, Packet>()
        .collect::<Interleaved, Packet>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, Time>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .commit_trim::<Packet>()
        .commit()
}

/// SiLU of the scaled gate projection, rounded to BF16.
fn activate(
    device: &mut Device,
    halves: &Halves,
    scales: &scale::Epilogue<Cluster, Slice, Chips>,
) -> Values<bf16> {
    let folded = fold(device, halves);
    let value = device
        .main
        .begin(folded.view())
        .fetch::<Time, Packet>()
        .collect::<Time, Packet>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Time }, { Packet } / 4], m![{ Packet } % 4]>();
    crate::ops::linear::Affine::new((), scales)
        .forward(crate::ops::layout::Dm::<producer::contraction::Single, _> {
            tiles: crate::compose::Tiles::over(Rows::SIZE),
            value,
        })
        .value
        .vector_stash()
        .vector_fp_unary(FpUnaryOp::Sigmoid)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Fma), Stash)
        .vector_widen_concat::<Time, Packet>()
        .vector_final()
        .cast::<bf16, m![{ Packet } # 16]>()
        .commit_trim::<Packet>()
        .commit()
}
