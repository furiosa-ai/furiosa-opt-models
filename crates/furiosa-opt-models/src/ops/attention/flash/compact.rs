//! Bidirectional attention of a 4,096-row query tile over a long key sequence, one chip's heads,
//! each span of keys resting in the tensor register files while the queries stream past it.
//!
//! - A cluster attends one head at a time; the chip's two clusters take a pair of heads.
//! - A span of 2,048 keys spreads over a cluster's 256 slices as eight query groups of 512 rows by
//!   32 key slices of 64 keys. A slice keeps its keys and its values in its tensor register file;
//!   its group's queries stream past the keys through the switch, and the weights it computed
//!   stream past the values.
//! - The values carry 8 channels of ones after their 128, the register's smallest channel group,
//!   so one product holds a row's weighted values and its mass; the products sum over the key
//!   slices onto the slice that holds the row.
//! - Each span's peaks combine over its key slices into every row's running maximum.
//! - `Scores<C>` round to BF16 before they are scaled; the maximum and the products are FP32.
//! - A call names its live keys, a prefix of the spans' keys; a last span whose live keys fit its
//!   first half takes that half alone, 32 keys a slice.

use furiosa_opt_std::prelude::*;

use super::{Duo, Head, Queries, Rows, Scattered, Span, TailRows, Term, Tokens, Visible};
use crate::Op;
use crate::load::Load;
use crate::tensor::live::Live;
use crate::tensor::memory::residency::Residency;
use crate::tensor::parallel::Chip;
use crate::tensor::scalar::Value as Count;

mod half;

/// Flash attention with eight mass channels and a single-pass running-maximum update.
/// Its query/key layouts match the default flash kernel; callers select this variant explicitly.
#[derive(DeviceSend, Load, Residency)]
pub struct Compact<Heads, const BITS: usize>(std::marker::PhantomData<Heads>);

impl<Heads, const BITS: usize> Default for Compact<Heads, BITS> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

axes![Channel = 136];

/// A pair of heads, one a cluster.
type Cluster = m![Duo];
/// Eight query groups of 512 rows by the span's 32 key slices of 64 keys.
type Slices = m![Rows / 512, Span / 64];
/// A query group's rows, sixteen a slice: each row's products sum onto its slice.
type QuerySlices = m![Rows / 512, Rows % 512 / 16];
/// The span's keys or values, landed in every query group.
type Copies = m![8, Span / 64];
/// A query group's key slices, each holding the statistic they agree on.
type Agreed = m![Rows / 512, 32];

type Query<Chips = Chip> = DmTensor<bf16, Chips, Cluster, QuerySlices, m![Rows % 16, Head]>;
type Key<Chips = Chip> = DmTensor<bf16, Chips, Cluster, Slices, m![Span % 64, Head]>;
/// A slice's values channel-major, the head channels followed by the 8 channels of ones.
type Transposed<Chips = Chip> = DmTensor<bf16, Chips, Cluster, Slices, m![Channel, Span % 64]>;
type KeyRegister<Chips = Chip> = TrfTensor<
    bf16,
    Chips,
    Cluster,
    Slices,
    m![Span % 8],
    m![Span % 64 / 16, Head / 16, Span % 16 / 8, Head % 16],
>;
type ValueRegister<Chips = Chip> = TrfTensor<
    bf16,
    Chips,
    Cluster,
    Slices,
    m![Channel % 8],
    m![Span % 64 / 16, Channel / 8, Span % 16],
>;
/// Every row of the group's scores against the slice's keys, or their weights.
type Scores<Chips = Chip> = DmTensor<bf16, Chips, Cluster, Slices, m![Rows % 512, Span % 64]>;
/// One statistic a row of the group, in every key slice.
type Peak<Chips = Chip> = DmTensor<f32, Chips, Cluster, Slices, m![Rows % 512]>;
type Top<Chips = Chip> = VrfTensor<f32, Chips, Cluster, Slices, m![Rows % 512]>;
/// One statistic a row, on the row's own slice.
type DecayOperand<Chips = Chip> = VrfTensor<f32, Chips, Cluster, QuerySlices, m![Rows % 16]>;
type AgreedTop<Chips = Chip> = VrfTensor<f32, Chips, Cluster, Agreed, m![Rows % 512]>;
/// A row's weighted values, then its mass in the channels of ones.
type Product<Chips = Chip> = DmTensor<f32, Chips, Cluster, QuerySlices, m![Rows % 16, Channel]>;
type Attended<Chips = Chip> = DmTensor<bf16, Chips, Cluster, QuerySlices, m![Rows % 16, Head]>;

/// The running maximum before any key: finite, so a span whose keys are all absent leaves it.
const START: f32 = -1e30;

/// A key visibility and where its absent keys may lie, which decides how each span's values enter
/// the sum.
trait Visibility<Spans: AxisName, Chips: M = Chip> {
    /// Whether the visible keys are a prefix, so that the live keys' count says where they end.
    const PREFIX: bool;

    fn keys(&self) -> &Visible<Spans, Chips>;

    /// `value`, span `span`'s values, channel-major into the value and mass channels of
    /// `transposed` for the span's trip.
    fn stage(
        &self,
        device: &mut Device,
        value: &Key<Chips>,
        span: usize,
        transposed: &mut Transposed<Chips>,
    );
}

/// Only the last attended span holds absent keys, so the spans before it enter whole.
impl<Spans: AxisName, Chips: M> Visibility<Spans, Chips> for Visible<Spans, Chips> {
    const PREFIX: bool = true;

    fn keys(&self) -> &Visible<Spans, Chips> {
        self
    }

    fn stage(
        &self,
        device: &mut Device,
        value: &Key<Chips>,
        _: usize,
        transposed: &mut Transposed<Chips>,
    ) {
        Transpose.forward(device, (value, transposed));
    }
}

impl<Spans: AxisName, Chips: M> Visibility<Spans, Chips> for Scattered<'_, Spans, Chips> {
    const PREFIX: bool = false;

    fn keys(&self) -> &Visible<Spans, Chips> {
        self.0
    }

    fn stage(
        &self,
        device: &mut Device,
        value: &Key<Chips>,
        span: usize,
        transposed: &mut Transposed<Chips>,
    ) {
        let (kept, seen) = Keep.forward(device, (value, self.0, span));
        VisibleValues.forward(device, ((&kept, &seen), transposed));
    }
}

/// What one pass over the spans attends: the query rows of a tile's head pairs that fill the
/// eight query groups, each group over its own pair's keys.
trait Unit<Pairs: AxisName> {
    /// Head pairs one unit holds.
    const PAIRS: usize;

    /// Unit `unit`'s query rows of the tile from row `from`, sixteen a slice.
    fn queries<Q: AxisName, Chips: M>(
        device: &mut Device,
        query: &Queries<Q, Pairs, Chips>,
        from: usize,
        unit: usize,
    ) -> Query<Chips>;

    /// Span `span`'s keys or values of unit `unit`'s pairs, in their query groups.
    fn tokens<Spans: AxisName, Chips: M>(
        device: &mut Device,
        tokens: &Tokens<Spans, Pairs, Chips>,
        unit: usize,
        span: usize,
    ) -> Key<Chips>;

    /// Unit `unit`'s attended rows into the tile from row `to` of `output`, given as
    /// `(output, to)`.
    fn store<O: AxisName, Chips: M>(
        device: &mut Device,
        attended: &Attended<Chips>,
        output: (&mut Queries<O, Pairs, Chips>, usize),
        unit: usize,
    );
}

/// One head pair's whole 4,096-row tile, 512 rows a query group, every group over the pair's keys.
struct Whole;

impl<Pairs: AxisName> Unit<Pairs> for Whole {
    const PAIRS: usize = 1;

    fn queries<Q: AxisName, Chips: M>(
        device: &mut Device,
        query: &Queries<Q, Pairs, Chips>,
        from: usize,
        pair: usize,
    ) -> Query<Chips> {
        let tile = query
            .view()
            .tile::<m![Q], { Rows::SIZE }, m![Q = 4096 # Q, Pairs, Duo, Head]>(from);
        // SAFETY: the tile's rows are the query tile's in order; only the row axis is renamed.
        let tile = unsafe { tile.reshape::<Chips, m![Rows # Q, Pairs, Duo, Head]>() };
        tile.tile::<m![Pairs], 1, m![Rows # Q, Pairs = 1 # Pairs, Duo, Head]>(pair)
            .to_dm(&mut device.tdma)
    }

    fn tokens<Spans: AxisName, Chips: M>(
        device: &mut Device,
        tokens: &Tokens<Spans, Pairs, Chips>,
        pair: usize,
        span: usize,
    ) -> Key<Chips> {
        let landed: DmTensor<bf16, Chips, Cluster, Copies, m![Span % 64, Head]> = tokens
            .view()
            .tile::<m![Pairs], 1, m![Pairs = 1 # Pairs, Duo, Spans, Span, Head]>(pair)
            .tile::<m![Spans], 1, m![Pairs = 1 # Pairs, Duo, Spans = 1 # Spans, Span, Head]>(span)
            .to_dm(&mut device.tdma);
        // SAFETY: every query group holds the same keys; only the broadcast axis is named.
        unsafe { landed.reshape() }
    }

    fn store<O: AxisName, Chips: M>(
        device: &mut Device,
        attended: &Attended<Chips>,
        (output, to): (&mut Queries<O, Pairs, Chips>, usize),
        pair: usize,
    ) {
        let tile = output
            .view_mut()
            .tile::<m![O], { Rows::SIZE }, m![O = 4096 #{!} O, Pairs, Duo, Head]>(to);
        // SAFETY: the tile's rows are the query tile's in order; only the row axis is renamed.
        let tile = unsafe { tile.reshape::<Chips, m![Rows #{!} O, Pairs, Duo, Head]>() };
        let heads =
            tile.tile::<m![Pairs], 1, m![Rows #{!} O, Pairs = 1 #{!} Pairs, Duo, Head]>(pair);
        attended.view().to_hbm_view(&mut device.tdma, heads);
    }
}

/// The last 1,024 rows of four head pairs at once, so a quarter-tile tail takes a quarter of the
/// trips; a last unit short of four pairs repeats pairs the unit before it attended.
struct Tail;

impl<Pairs: AxisName> Unit<Pairs> for Tail {
    const PAIRS: usize = 4;

    fn queries<Q: AxisName, Chips: M>(
        device: &mut Device,
        query: &Queries<Q, Pairs, Chips>,
        from: usize,
        unit: usize,
    ) -> Query<Chips> {
        let tile = query
            .view()
            .tile::<m![Q], { TailRows::SIZE }, m![Q = 1024 # Q, Pairs, Duo, Head]>(from);
        // SAFETY: the tile's rows are the unit's rows of each pair in order; only axes are renamed.
        let tile = unsafe { tile.reshape::<Chips, m![Rows % 1024 # Q, Pairs, Duo, Head]>() };
        let tile = tile.tile::<m![Pairs], 4, m![Rows % 1024 # Q, Pairs = 4 # Pairs, Duo, Head]>(
            (4 * unit).min(Pairs::SIZE - 4),
        );
        // SAFETY: row r of the unit's pair p is query row p * 1024 + r; only the axis is renamed.
        let tile =
            unsafe { tile.reshape::<Chips, m![Rows % 1024 # Q, Rows / 1024 # Pairs, Duo, Head]>() };
        tile.to_dm(&mut device.tdma)
    }

    fn tokens<Spans: AxisName, Chips: M>(
        device: &mut Device,
        tokens: &Tokens<Spans, Pairs, Chips>,
        unit: usize,
        span: usize,
    ) -> Key<Chips> {
        let pairs = tokens
            .view()
            .tile::<m![Pairs], 4, m![Pairs = 4 # Pairs, Duo, Spans, Span, Head]>(
                (4 * unit).min(Pairs::SIZE - 4),
            );
        // SAFETY: pair p of the unit serves query rows p * 1024..; only the axis is renamed.
        let pairs =
            unsafe { pairs.reshape::<Chips, m![Rows / 1024 # Pairs, Duo, Spans, Span, Head]>() };
        let landed: DmTensor<
            bf16,
            Chips,
            Cluster,
            m![Rows / 1024, 2, Span / 64],
            m![Span % 64, Head],
        > = pairs
            .tile::<m![Spans], 1, m![Rows / 1024 # Pairs, Duo, Spans = 1 # Spans, Span, Head]>(span)
            .to_dm(&mut device.tdma);
        // SAFETY: a pair's two query groups hold the same keys; only the broadcast axis is named.
        unsafe { landed.reshape() }
    }

    fn store<O: AxisName, Chips: M>(
        device: &mut Device,
        attended: &Attended<Chips>,
        (output, to): (&mut Queries<O, Pairs, Chips>, usize),
        unit: usize,
    ) {
        let tile = output
            .view_mut()
            .tile::<m![O], { TailRows::SIZE }, m![O = 1024 #{!} O, Pairs, Duo, Head]>(to);
        // SAFETY: as for the queries.
        let tile = unsafe { tile.reshape::<Chips, m![Rows % 1024 #{!} O, Pairs, Duo, Head]>() };
        let tile = tile.tile::<m![Pairs], 4, m![
            Rows % 1024 #{!} O,
            Pairs = 4 #{!} Pairs,
            Duo,
            Head
        ]>((4 * unit).min(Pairs::SIZE - 4));
        // SAFETY: as for the queries.
        let tile = unsafe {
            tile.reshape::<Chips, m![Rows % 1024 #{!} O, Rows / 1024 #{!} Pairs, Duo, Head]>()
        };
        attended.view().to_hbm_view(&mut device.tdma, tile);
    }
}

impl<
    R: AxisName,
    Spans: AxisName,
    Pairs: AxisName,
    V: Visibility<Spans, Chips>,
    Chips: M,
    const SCALE: usize,
    const HALF: bool,
>
    Op<(
        &Queries<R, Pairs, Chips>,
        &mut Queries<R, Pairs, Chips>,
        Live<(), R>,
    )> for Sweep<'_, Spans, Pairs, V, Chips, SCALE, HALF>
{
    /// softmax(scale * QKᵀ)V of every row of `query` over the visible keys of the first `spans` spans,
    /// a 4,096-row tile at a time; a tile's first unit lands and scores under the previous tile's last.
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (query, output, interval): (
            &Queries<R, Pairs, Chips>,
            &mut Queries<R, Pairs, Chips>,
            Live<(), R>,
        ),
    ) {
        let (key, value, visible) = (self.key, self.value, self.visible);
        const {
            assert!(
                R::SIZE >= Rows::SIZE,
                "the query envelope holds one whole tile"
            )
        };
        let rows = interval.rows().max(Rows::SIZE);
        let start = interval.start().min(R::SIZE - rows);
        // A quarter tile left over is the tail of four head pairs at once; any other remainder is a
        // last tile that ends at the last row and repeats rows the tile before it attended.
        let tail = rows % Rows::SIZE == TailRows::SIZE
            && const { Pairs::SIZE >= <Tail as Unit<Pairs>>::PAIRS };
        let tiles = if tail {
            rows / Rows::SIZE
        } else {
            rows.div_ceil(Rows::SIZE)
        };
        let pairs = const { Pairs::SIZE };
        let first_row = |tile: usize| start + (tile * Rows::SIZE).min(rows - Rows::SIZE);
        let mut transposed = Transposed::<Chips>::new();
        transposed
            .view_mut()
            .tile::<m![Channel], 8, m![Channel = 8 #{!} Channel, Span % 64]>(Head::SIZE)
            .memset(const { bf16::from_f32(1.0) }, &mut device.sub);
        let landed = Landing::<Whole>(core::marker::PhantomData)
            .forward(device, ((query, key, value), start, 0));
        let mut ready = {
            let Landed {
                rows,
                keys,
                first,
                later,
                values,
            } = landed;
            let scored = {
                let score = LoadScore.forward(&mut device.main, keys);
                Calculate.forward(device, (score, &rows))
            };
            {
                let first = first;
                visible.stage(device, &first, 0, &mut transposed);
            }
            Ready {
                rows,
                scored,
                later,
                values,
            }
        };
        // Every tile but the last: its last pair lands the next tile's first.
        for tile in 0..tiles - 1 {
            let (from, next) = (first_row(tile), first_row(tile + 1));
            #[unroll]
            for pair in 0..pairs {
                let wraps = (pair + 1) / pairs;
                ready = UnitSweep::<Whole, _>(core::marker::PhantomData, self).forward(
                    device,
                    (
                        (ready, &mut transposed),
                        (pair, |device: &mut Device| {
                            Landing::<Whole>(core::marker::PhantomData).forward(
                                device,
                                (
                                    (query, key, value),
                                    from + (next - from) * wraps,
                                    (pair + 1) % pairs,
                                ),
                            )
                        }),
                        |device: &mut Device, attended: &Attended<Chips>| {
                            Whole::store(device, attended, (&mut *output, from), pair)
                        },
                    ),
                );
            }
        }
        let from = first_row(tiles - 1);
        #[unroll]
        for pair in 0..pairs - 1 {
            ready = UnitSweep::<Whole, _>(core::marker::PhantomData, self).forward(
                device,
                (
                    (ready, &mut transposed),
                    (pair, |device: &mut Device| {
                        Landing::<Whole>(core::marker::PhantomData)
                            .forward(device, ((query, key, value), from, pair + 1))
                    }),
                    |device: &mut Device, attended: &Attended<Chips>| {
                        Whole::store(device, attended, (&mut *output, from), pair)
                    },
                ),
            );
        }
        let last = pairs - 1;
        let store_last = |device: &mut Device, attended: &Attended<Chips>| {
            Whole::store(device, attended, (&mut *output, from), last)
        };
        if tail {
            let at = start + rows - TailRows::SIZE;
            ready = UnitSweep::<Whole, _>(core::marker::PhantomData, self).forward(
                device,
                (
                    (ready, &mut transposed),
                    (last, |device: &mut Device| {
                        Landing::<Tail>(core::marker::PhantomData)
                            .forward(device, ((query, key, value), at, 0))
                    }),
                    store_last,
                ),
            );
            let units = const { Pairs::SIZE.div_ceil(<Tail as Unit<Pairs>>::PAIRS) };
            #[unroll]
            for unit in 0..units {
                let next = (unit + 1).min(units - 1);
                ready = UnitSweep::<Tail, _>(core::marker::PhantomData, self).forward(
                    device,
                    (
                        (ready, &mut transposed),
                        (unit, |device: &mut Device| {
                            Landing::<Tail>(core::marker::PhantomData)
                                .forward(device, ((query, key, value), at, next))
                        }),
                        |device: &mut Device, attended: &Attended<Chips>| {
                            Tail::store(device, attended, (&mut *output, at), unit)
                        },
                    ),
                );
            }
        } else {
            ready = UnitSweep::<Whole, _>(core::marker::PhantomData, self).forward(
                device,
                (
                    (ready, &mut transposed),
                    (last, |device: &mut Device| {
                        Landing::<Whole>(core::marker::PhantomData)
                            .forward(device, ((query, key, value), from, last))
                    }),
                    store_last,
                ),
            );
        }
        drop(ready);
    }
}

/// What every unit of a call sweeps: the keys and values of its spans, their visibility, and the
/// live spans; `SCALE` holds the bits of the FP32 scale, and `HALF` whether no live key lies past
/// the first half of the last span.
struct Sweep<
    'a,
    Spans: AxisName,
    Pairs: AxisName,
    V,
    Chips: M,
    const SCALE: usize,
    const HALF: bool,
> {
    key: &'a Tokens<Spans, Pairs, Chips>,
    value: &'a Tokens<Spans, Pairs, Chips>,
    visible: &'a V,
    live: usize,
}

struct UnitSweep<'a, U, S>(core::marker::PhantomData<U>, &'a S);

impl<
    Spans: AxisName,
    Pairs: AxisName,
    V: Visibility<Spans, Chips>,
    Chips: M,
    const SCALE: usize,
    const HALF: bool,
    U: half::Halves<Pairs>,
    F: FnOnce(&mut Device) -> Landed<Chips>,
    S: FnOnce(&mut Device, &Attended<Chips>),
> Op<((Ready<Chips>, &mut Transposed<Chips>), (usize, F), S)>
    for UnitSweep<'_, U, Sweep<'_, Spans, Pairs, V, Chips, SCALE, HALF>>
{
    type Output = Ready<Chips>;

    /// Unit `unit` of kind `U` from `ready`: its last span runs while `following` lands the next
    /// unit and Main scores that unit's first span; `store` takes the rows, the next unit returns.
    fn forward(
        &self,
        device: &mut Device,
        ((ready, transposed), (unit, following), store): (
            (Ready<Chips>, &mut Transposed<Chips>),
            (usize, F),
            S,
        ),
    ) -> Ready<Chips> {
        let (key, value, visible, live) = (self.1.key, self.1.value, self.1.visible, self.1.live);
        let scale = const { f32::from_bits(SCALE as u32) };
        let Ready {
            rows,
            scored,
            later,
            values,
        } = ready;
        let score = LoadScore.forward(&mut device.main, later);
        let mut held = Peak::<Chips>::new();
        held.view_mut().memset(START, &mut device.sub);
        let previous = Maximum.forward(device, &held);
        let mut sum = Product::<Chips>::new();
        sum.view_mut().memset(0.0, &mut device.sub);
        let mut trip = Trip {
            score,
            scored,
            previous,
            values,
            softmax: OnlineSoftmax { max: held, sum },
        };
        // The trips after the loop read the carried tensors where an even count of trips leaves
        // them; after an odd count the loop's end copies them back.
        for step in 0..live - 3 {
            let landed = (
                U::tokens(device, key, unit, (step + 2).min(Spans::SIZE - 1)),
                U::tokens(device, value, unit, (step + 2).min(Spans::SIZE - 1)),
            );
            let staged = (visible, step + 1, &mut *transposed);
            trip = self
                .1
                .forward(&mut *device, (trip, (&rows, staged), landed));
        }
        if const { HALF } {
            return half::Finish::<U, _>(core::marker::PhantomData, self.1)
                .forward(device, ((trip, rows, transposed), (unit, following), store));
        }
        let landed = (
            U::tokens(device, key, unit, live - 1),
            U::tokens(device, value, unit, live - 1),
        );
        let Trip {
            score,
            scored,
            previous,
            values,
            softmax,
        } = self.1.forward(
            &mut *device,
            (trip, (&rows, (visible, live - 2, &mut *transposed)), landed),
        );
        let following = following(device);
        let top = Raise.forward(device, (&scored, previous, scale));
        let next = Calculate.forward(device, (score, &rows));
        drop(rows);
        let (weights, softmax) = Weigh.forward(device, (softmax, (scored, top), scale));
        let vreg = Register.forward(device, &*transposed);
        let product = Multiply.forward(device, (weights, vreg));
        let cleared = {
            let values = Values(values);
            let (kept, seen) = Keep.forward(device, (&values.0, visible.keys(), live - 1));
            let mut transposed = Transposed::<Chips>::new();
            VisibleValues.forward(device, ((&kept, &seen), &mut transposed));
            transposed
        };
        let (softmax, previous, retired) = Absorb.forward(device, (softmax, product));
        let scored = next;
        // The last span, while Main scores the next unit's first span.
        let Landed {
            rows,
            keys,
            first,
            later,
            values,
        } = following;
        let top = Raise.forward(device, (&scored, previous, scale));
        let next = {
            let score = LoadScore.forward(&mut device.main, keys);
            Calculate.forward(device, (score, &rows))
        };
        let (weights, softmax) = Weigh.forward(device, (softmax, (scored, top), scale));
        let vreg = {
            let cleared = cleared;
            Register.forward(device, &cleared)
        };
        let product = Multiply.forward(device, (weights, vreg));
        {
            let first = first;
            visible.stage(device, &first, 0, transposed);
        }
        let (finished, sum) = {
            let OnlineSoftmax {
                max: (held, top),
                sum,
            } = softmax;
            let raised = Maximum.forward(device, &top);
            let factor = Decay.forward(device, (&held, &raised));
            drop((held, top, raised));
            let next = Accumulate.forward(&mut *device, (&sum, &product, &factor));
            drop((factor, product));
            let finished = {
                let softmax = OnlineSoftmax { max: (), sum: next };
                let sum = &softmax.sum;
                let ones = sum
                    .view()
                    .tile::<m![Channel], 8, m![Rows % 16, Channel = 8 # Channel]>(Head::SIZE);
                // SAFETY: every ones channel holds the row's mass.
                let mass = unsafe {
                    ones.reshape::<Chips, Cluster, QuerySlices, m![Rows % 16, [1 # 8] # Channel]>()
                };
                let total: VrfTensor<f32, Chips, Cluster, QuerySlices, m![Rows % 16, 1 # 8]> =
                    device
                        .sub
                        .begin(mass)
                        .fetch::<m![Rows % 16], m![1 # 8]>()
                        .collect::<m![Rows % 16], m![1 # 8]>()
                        .to_vrf();
                let values = sum
                    .view()
                    .tile::<m![Channel], { Head::SIZE }, m![Rows % 16, Channel = 128 # Channel]>(0);
                // SAFETY: the value channels are consecutive; only the axis is renamed.
                let values = unsafe {
                    values.reshape::<Chips, Cluster, QuerySlices, m![Rows % 16, Head # Channel]>()
                };
                device
                    .main
                    .begin(values)
                    .fetch::<m![Rows % 16, Head / 8], m![Head % 8]>()
                    .collect::<m![Rows % 16, Head / 8], m![Head % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Rows % 16, Head / 4], m![Head % 4]>()
                    .vector_fp_div(&total)
                    .vector_widen_concat::<m![Rows % 16, Head / 8], m![Head % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Head % 8 # 16]>()
                    .commit_trim::<m![Head % 8]>()
                    .commit()
            };
            (finished, sum)
        };
        // Retired sums keep their original allocation lifetimes through the unit's store.
        let _sums = (sum, retired);
        let attended = finished;
        store(device, &attended);
        Ready {
            rows,
            scored: next,
            later,
            values,
        }
    }
}

/// A unit's queries and its first two spans' keys and values.
struct Landed<Chips: M> {
    rows: Query<Chips>,
    keys: Key<Chips>,
    first: Key<Chips>,
    later: Key<Chips>,
    values: Key<Chips>,
}

struct Landing<U>(core::marker::PhantomData<U>);

impl<Chips: M, U: Unit<Pairs>, Q: AxisName, Spans: AxisName, Pairs: AxisName>
    Op<(
        (
            &Queries<Q, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
        ),
        usize,
        usize,
    )> for Landing<U>
{
    type Output = Landed<Chips>;

    fn forward(
        &self,
        device: &mut Device,
        ((query, key, value), from, unit): (
            (
                &Queries<Q, Pairs, Chips>,
                &Tokens<Spans, Pairs, Chips>,
                &Tokens<Spans, Pairs, Chips>,
            ),
            usize,
            usize,
        ),
    ) -> Landed<Chips> {
        Landed {
            rows: U::queries(device, query, from, unit),
            keys: U::tokens(device, key, unit, 0),
            first: U::tokens(device, value, unit, 0),
            later: U::tokens(device, key, unit, 1.min(Spans::SIZE - 1)),
            values: U::tokens(device, value, unit, 1.min(Spans::SIZE - 1)),
        }
    }
}

/// A unit ready for its first trip, all in data memory so it may cross a loop: its queries, its
/// first span's scores, and its second span's keys and values; its first span's values are staged.
struct Ready<Chips: M> {
    rows: Query<Chips>,
    scored: Scores<Chips>,
    later: Key<Chips>,
    values: Key<Chips>,
}

/// What trip c hands the next: span c's scores, the running maximum and sum before span c, and
/// span c + 1's keys in the register file and its values.
struct Trip<Chips: M> {
    score: Score<Chips>,
    scored: Scores<Chips>,
    previous: AgreedTop<Chips>,
    values: Key<Chips>,
    softmax: OnlineSoftmax<Chips>,
}

/// Trip `c`: span c + 1's scores and span c's values weighed into the sum, while `landed` holds
/// span c + 2's keys and values and `staged` the buffers span c + 1 transposes into.
impl<
    Spans: AxisName,
    Pairs: AxisName,
    V: Visibility<Spans, Chips>,
    Chips: M,
    const SCALE: usize,
    const HALF: bool,
>
    Op<(
        Trip<Chips>,
        (&Query<Chips>, (&V, usize, &mut Transposed<Chips>)),
        (Key<Chips>, Key<Chips>),
    )> for Sweep<'_, Spans, Pairs, V, Chips, SCALE, HALF>
{
    type Output = Trip<Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (trip, (rows, staged), (next_keys, next_values)): (
            Trip<Chips>,
            (&Query<Chips>, (&V, usize, &mut Transposed<Chips>)),
            (Key<Chips>, Key<Chips>),
        ),
    ) -> Self::Output {
        let Trip {
            score,
            scored,
            previous,
            values,
            softmax,
        } = trip;
        let (visible, span, transposed) = staged;
        let scale = const { f32::from_bits(SCALE as u32) };
        // Sub raises the maximum first, so the pass starts as soon as the last trip's sum is in.
        let top = Raise.forward(device, (&scored, previous, scale));
        let next = Calculate.forward(device, (score, rows));
        let (weights, softmax) = Weigh.forward(device, (softmax, (scored, top), scale));
        let vreg = Register.forward(device, &*transposed);
        let product = Multiply.forward(device, (weights, vreg));
        {
            let values = values;
            visible.stage(device, &values, span, transposed);
        }
        let (softmax, previous, _sum) = Absorb.forward(device, (softmax, product));
        Trip {
            score: LoadScore.forward(&mut device.main, next_keys),
            scored: next,
            previous,
            values: next_values,
            softmax,
        }
    }
}

struct Score<Chips: M>(KeyRegister<Chips>);

struct LoadScore;

impl LoadScore {
    /// `key` into the register file by `context`: a load by Main holds Sub too, while Sub can load
    /// under a running product.
    fn forward<const T: Tu, Chips: M>(
        &self,
        context: &mut TuContext<T>,
        key: Key<Chips>,
    ) -> Score<Chips> {
        Score(
            context
                .begin(key.view())
                .fetch::<m![Span % 8, Span % 64 / 16, Head / 16, Span % 16 / 8], m![Head % 16]>()
                .collect::<m![Span % 8, Span % 64 / 16, Head / 16, Span % 16 / 8], m![Head % 16]>()
                .to_trf(),
        )
    }
}

struct Calculate;

impl<Chips: M> Op<(Score<Chips>, &Query<Chips>)> for Calculate {
    type Output = Scores<Chips>;

    /// Every row of the group against every key of the slice: each query streams from the slice
    /// that holds it past the group's 32 key slices.
    fn forward(
        &self,
        device: &mut Device,
        (score, query): (Score<Chips>, &Query<Chips>),
    ) -> Self::Output {
        type Stream = m![Rows % 16 / 2, Span % 64 / 16, Rows % 512 / 16, Head / 16];
        type Steps = m![{ Stream }, Span % 16 / 8];
        type Sums = m![
            Rows % 16 / 2,
            Span % 64 / 16,
            Rows % 512 / 16,
            Span % 16 / 8
        ];
        let mut scores = Scores::<Chips>::new();
        device
            .main
            .begin(query.view())
            .fetch::<m![Rows % 16 / 2, Span % 64 / 16, Head / 16], m![Rows % 2, Head % 16]>()
            .switch::<Slices, Stream>(SwitchConfig::Broadcast01 {
                slice1: 32,
                slice0: 1,
                time0: 8,
            })
            .collect::<m![{ Stream }, Rows % 2], m![Head % 16]>()
            .contract_outer::<Steps, m![Rows % 2, Head % 16], _, _, _>(&score.0)
            .contract_packet::<m![Rows % 2]>()
            .contract_time::<Sums>()
            .contract_lane::<m![{ Sums }, Rows % 2], m![Span % 8]>(LaneMode::Interleaved)
            // The commit adapter narrows, so the Vector Engine stays free for Sub's passes.
            .commit_trim::<m![Span % 8]>()
            .commit_cast::<bf16>()
            .commit_view(scores.view_mut());
        scores
    }
}

struct OnlineSoftmax<Chips: M, Max = Peak<Chips>> {
    max: Max,
    sum: Product<Chips>,
}

struct Weights<Chips: M> {
    values: Scores<Chips>,
}

struct Raise;

impl<Chips: M> Op<(&Scores<Chips>, AgreedTop<Chips>, f32)> for Raise {
    type Output = Peak<Chips>;

    /// `held` raised to each row's scaled scores over its group's key slices in one pass; rounding
    /// a product by a positive scale keeps the order, so scaling first keeps the maximum.
    fn forward(
        &self,
        device: &mut Device,
        (scores, held, scale): (&Scores<Chips>, AgreedTop<Chips>, f32),
    ) -> Self::Output {
        // SAFETY: a group's key slices hold the same running maximum; only the slice axis is renamed.
        let held: Top<Chips> = unsafe { held.reshape() };
        let raised: DmTensor<f32, Chips, Cluster, Agreed, m![Rows % 512]> = device
            .sub
            .begin(scores.view())
            .fetch::<m![Rows % 512, Span % 64 / 16], m![Span % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % 512, Span % 64 / 8], m![Span % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows % 512, Span % 64 / 4], m![Span % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
            .vector_intra_slice_reduce::<Span, m![Rows % 512], m![1 # 4]>(
                IntraSliceReduceOpF32::Max,
            )
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Max, &held)
            .vector_inter_slice_reduce::<Agreed, m![Rows % 512]>(InterSliceReduceOpF32::Max)
            .vector_final()
            .transpose::<m![Rows % 512 / 2], m![Rows % 2 # 8]>()
            .commit_trim::<m![Rows % 2]>()
            .commit();
        // SAFETY: every key slice of a group received the same maximum.
        unsafe { raised.reshape() }
    }
}

struct Weigh;

impl<Chips: M> Op<(OnlineSoftmax<Chips>, (Scores<Chips>, Peak<Chips>), f32)> for Weigh {
    type Output = (
        Weights<Chips>,
        OnlineSoftmax<Chips, (Peak<Chips>, Peak<Chips>)>,
    );

    fn forward(
        &self,
        device: &mut Device,
        (softmax, (scores, top), scale): (OnlineSoftmax<Chips>, (Scores<Chips>, Peak<Chips>), f32),
    ) -> Self::Output {
        let values = {
            let scores = scores;
            let operand = Operand.forward(device, &top);
            device
                .sub
                .begin(scores.view())
                .fetch::<m![Rows % 512, Span % 64 / 16], m![Span % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![Rows % 512, Span % 64 / 8], m![Span % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Rows % 512, Span % 64 / 4], m![Span % 4]>()
                .vector_fp_ternary_with_mode(
                    FpTernaryOp::FmaF,
                    TernaryArgMode::Mode021,
                    (&operand, -scale),
                )
                .vector_fp_unary(FpUnaryOp::NegExp)
                .vector_widen_concat::<m![Rows % 512, Span % 64 / 8], m![Span % 8]>()
                .vector_final()
                .cast::<bf16, m![Span % 8 # 16]>()
                .commit_trim::<m![Span % 8]>()
                .commit()
        };
        (
            Weights { values },
            OnlineSoftmax {
                max: (softmax.max, top),
                sum: softmax.sum,
            },
        )
    }
}

struct Decay;

impl<Chips: M> Op<(&Peak<Chips>, &AgreedTop<Chips>)> for Decay {
    type Output = DecayOperand<Chips>;

    /// The factor e^(held - raised) on each row's own slice: every key slice of a group holds the same
    /// factors, so reducing the copies moves each row's factor onto its slice.
    fn forward(
        &self,
        device: &mut Device,
        (held, raised): (&Peak<Chips>, &AgreedTop<Chips>),
    ) -> Self::Output {
        // SAFETY: the held maximum is the same in every key slice of a group; only the axis is named.
        let held = unsafe {
            held.view()
                .reshape::<Chips, Cluster, Agreed, m![Rows % 512]>()
        };
        device
            .main
            .begin(held)
            .fetch::<m![Rows % 512 / 8], m![Rows % 8]>()
            .collect::<m![Rows % 512 / 8], m![Rows % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows % 512 / 4], m![Rows % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, raised)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Rows % 512 / 8], m![Rows % 8]>()
            .vector_inter_slice_reduce::<QuerySlices, m![Rows % 16 / 8]>(InterSliceReduceOpF32::Max)
            .vector_final()
            .to_vrf(&mut device.sub)
    }
}

struct Maximum;

impl<Chips: M> Op<&Peak<Chips>> for Maximum {
    type Output = AgreedTop<Chips>;

    /// The running maximum `held` as the operand of the group's key slices.
    fn forward(&self, device: &mut Device, held: &Peak<Chips>) -> Self::Output {
        // SAFETY: a group's key slices hold the same running maximum; only the broadcast axis is named.
        unsafe { Operand.forward(device, held).reshape() }
    }
}

struct Operand;

impl<Chips: M> Op<&Peak<Chips>> for Operand {
    type Output = Top<Chips>;

    fn forward(&self, device: &mut Device, peak: &Peak<Chips>) -> Self::Output {
        device
            .sub
            .begin(peak.view())
            .fetch::<m![Rows % 512 / 8], m![Rows % 8]>()
            .collect::<m![Rows % 512 / 8], m![Rows % 8]>()
            .to_vrf()
    }
}

struct Values<Chips: M>(Key<Chips>);

struct Register;

impl<Chips: M> Op<&Transposed<Chips>> for Register {
    type Output = ValueRegister<Chips>;

    fn forward(&self, device: &mut Device, value: &Transposed<Chips>) -> Self::Output {
        device
            .main
            .begin(value.view())
            .fetch::<m![Channel % 8, Span % 64 / 16, Channel / 8], m![Span % 16]>()
            .collect::<m![Channel % 8, Span % 64 / 16, Channel / 8], m![Span % 16]>()
            .to_trf()
    }
}

struct Transpose;

impl<Chips: M> Op<(&Key<Chips>, &mut Transposed<Chips>)> for Transpose {
    type Output = ();

    /// `value` channel-major into the value channels of `transposed`.
    fn forward(
        &self,
        device: &mut Device,
        (value, transposed): (&Key<Chips>, &mut Transposed<Chips>),
    ) {
        device
            .sub
            .begin(value.view())
            .fetch::<m![Head / 16, Span % 64], m![Head % 16]>()
            .collect::<m![Head / 16, Span % 64], m![Head % 16]>()
            .transpose::<m![Head / 16, Span % 64 / 4, Head % 16], m![Span % 4 # 16]>()
            .commit_trim::<m![Span % 4]>()
            .commit_view({
                let channels = transposed
                    .view_mut()
                    .tile::<m![Channel], { Head::SIZE }, m![Channel = 128 #{!} Channel, Span % 64]>(
                        0,
                    );
                // SAFETY: the value channels are consecutive; only the axis is renamed.
                unsafe {
                    channels.reshape::<Chips, Cluster, Slices, m![Head #{!} Channel, Span % 64]>()
                }
            });
    }
}

struct Keep;

impl<Chips: M, Spans: AxisName> Op<(&Key<Chips>, &Visible<Spans, Chips>, usize)> for Keep {
    type Output = (Key<Chips>, Seen<Chips>);

    /// `value` of span `span` with every absent key's values cleared, and the span's visibility.
    fn forward(
        &self,
        device: &mut Device,
        (value, visible, span): (&Key<Chips>, &Visible<Spans, Chips>, usize),
    ) -> Self::Output {
        let seen: DmTensor<bf16, Chips, m![2], Copies, m![Span % 64]> = visible
            .view()
            .tile::<m![Spans], 1, m![Spans = 1 # Spans, Span]>(span)
            .to_dm(&mut device.tdma);
        // SAFETY: both clusters and every query group hold the same visibility; only broadcast axes
        // are named.
        let seen: Seen<Chips> = unsafe { seen.reshape() };
        let keep: VrfTensor<f32, Chips, Cluster, Slices, m![Span % 64]> = device
            .sub
            .begin(seen.view())
            .fetch::<m![Span % 64 / 16], m![Span % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Span % 64 / 8], m![Span % 8]>()
            .to_vrf();
        let kept: Key<Chips> = device
            .sub
            .begin(value.view())
            .fetch::<m![Span % 64, Head / 16], m![Head % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Span % 64, Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Span % 64, Head / 4], m![Head % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &keep)
            .vector_widen_concat::<m![Span % 64, Head / 8], m![Head % 8]>()
            .vector_final()
            .cast::<bf16, m![Head % 8 # 16]>()
            .commit_trim::<m![Head % 8]>()
            .commit();
        (kept, seen)
    }
}

struct VisibleValues;

impl<Chips: M> Op<((&Key<Chips>, &Seen<Chips>), &mut Transposed<Chips>)> for VisibleValues {
    type Output = ();

    /// Kept values channel-major into `transposed`, whose mass channels take the keys' visibility in
    /// place of ones.
    fn forward(
        &self,
        device: &mut Device,
        ((kept, seen), transposed): ((&Key<Chips>, &Seen<Chips>), &mut Transposed<Chips>),
    ) {
        Transpose.forward(device, (kept, &mut *transposed));
        let mass = transposed
            .view_mut()
            .tile::<m![Channel], 8, m![Channel = 8 #{!} Channel, Span % 64]>(Head::SIZE);
        // SAFETY: each mass channel holds the same visibility; only the broadcast axis is renamed.
        let mass =
            unsafe { mass.reshape::<Chips, Cluster, Slices, m![[8] #{!} Channel, Span % 64]>() };
        device
            .sub
            .begin(seen.view())
            .fetch::<m![8, Span % 64 / 16], m![Span % 16]>()
            .collect::<m![8, Span % 64 / 16], m![Span % 16]>()
            .commit_trim::<m![Span % 16]>()
            .commit_view(mass);
    }
}

struct Multiply;

impl<Chips: M> Op<(Weights<Chips>, ValueRegister<Chips>)> for Multiply {
    type Output = Product<Chips>;

    /// Every row's weighted values and mass against the slice's keys, summed over the group's key
    /// slices onto the row's own slice.
    fn forward(
        &self,
        device: &mut Device,
        (weights, value): (Weights<Chips>, ValueRegister<Chips>),
    ) -> Self::Output {
        let value = value;
        let weights = weights.values;
        // Each weight packet meets every channel group of the register, so the weights stream once.
        type Fetch = m![Rows % 16 / 2, Rows % 512 / 16, Span % 64 / 16];
        type Steps = m![{ Fetch }, Channel / 8];
        type Sums = m![Rows % 16 / 2, Rows % 512 / 16, Channel / 8];
        let mut product = Product::<Chips>::new();
        device
            .main
            .begin(weights.view())
            .fetch::<Fetch, m![Rows % 2, Span % 16]>()
            .collect::<m![{ Fetch }, Rows % 2], m![Span % 16]>()
            .contract_outer::<Steps, m![Rows % 2, Span % 16], _, _, _>(&value)
            .contract_packet::<m![Rows % 2]>()
            .contract_time::<Sums>()
            .contract_lane::<m![{ Sums }, Rows % 2], m![Channel % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<QuerySlices, m![Rows % 16 / 2, Channel / 8, Rows % 2]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .commit_trim::<m![Channel % 8]>()
            .commit_view(product.view_mut());
        product
    }
}

struct Absorb;

impl<Chips: M>
    Op<(
        OnlineSoftmax<Chips, (Peak<Chips>, Peak<Chips>)>,
        Product<Chips>,
    )> for Absorb
{
    type Output = (OnlineSoftmax<Chips>, AgreedTop<Chips>, Product<Chips>);

    fn forward(
        &self,
        device: &mut Device,
        (softmax, product): (
            OnlineSoftmax<Chips, (Peak<Chips>, Peak<Chips>)>,
            Product<Chips>,
        ),
    ) -> Self::Output {
        let OnlineSoftmax {
            max: (held, top),
            sum,
        } = softmax;
        let raised = Maximum.forward(device, &top);
        let factor = Decay.forward(device, (&held, &raised));
        drop(held);
        let next = Accumulate.forward(device, (&sum, &product, &factor));
        drop((factor, product));
        (
            OnlineSoftmax {
                max: top,
                sum: next,
            },
            raised,
            sum,
        )
    }
}

struct Accumulate;

impl<Chips: M> Op<(&Product<Chips>, &Product<Chips>, &DecayOperand<Chips>)> for Accumulate {
    type Output = Product<Chips>;

    /// `sum` rescaled by `decay` plus `product`.
    fn forward(
        &self,
        device: &mut Device,
        (sum, product, decay): (&Product<Chips>, &Product<Chips>, &DecayOperand<Chips>),
    ) -> Self::Output {
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(sum.view(), product.view())
            .fetch::<m![Rows % 16, Term, Channel / 8], m![Channel % 8]>()
            .collect::<m![Rows % 16, Term, Channel / 8], m![Channel % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, m![Rows % 16, Channel / 8]>()
            .vector_narrow_split::<m![Rows % 16, Channel / 8, Channel % 8 / 4], m![Channel % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), decay, ())
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_widen_concat::<m![Rows % 16, Channel / 8], m![Channel % 8]>()
            .vector_final()
            .commit_trim::<m![Channel % 8]>()
            .commit()
    }
}

/// Span visibility in every slice that holds its keys.
type Seen<Chips = Chip> = DmTensor<bf16, Chips, Cluster, Slices, m![Span % 64]>;

impl<
    R: AxisName,
    Spans: AxisName,
    Pairs: AxisName,
    Heads: AxisName,
    Chips: M,
    V: Visibility<Spans, Chips>,
    const BITS: usize,
>
    Op<(
        &Queries<R, Pairs, Chips>,
        &Tokens<Spans, Pairs, Chips>,
        &Tokens<Spans, Pairs, Chips>,
        (&V, &Count<Chips>),
    )> for Compact<Heads, BITS>
{
    type Output = Queries<R, Pairs, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (query, key, value, (visible, spans)): (
            &Queries<R, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
            (&V, &Count<Chips>),
        ),
    ) -> Self::Output {
        self.forward(
            device,
            (Live::all().with(query), key, value, (visible, spans)),
        )
        .into_value()
    }
}

impl<
    R: AxisName,
    Spans: AxisName,
    Pairs: AxisName,
    Heads: AxisName,
    Chips: M,
    V: Visibility<Spans, Chips>,
    const BITS: usize,
>
    Op<(
        Live<&Queries<R, Pairs, Chips>, R>,
        &Tokens<Spans, Pairs, Chips>,
        &Tokens<Spans, Pairs, Chips>,
        (&V, &Count<Chips>),
    )> for Compact<Heads, BITS>
{
    type Output = Live<Queries<R, Pairs, Chips>, R>;

    fn forward(
        &self,
        device: &mut Device,
        (query, key, value, (visible, spans)): (
            Live<&Queries<R, Pairs, Chips>, R>,
            &Tokens<Spans, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
            (&V, &Count<Chips>),
        ),
    ) -> Self::Output {
        const {
            assert!(
                Heads::SIZE == 2 * Pairs::SIZE,
                "a chip attends its heads in pairs"
            )
        };
        let interval = query.with(());
        let query = *query.value();
        let mut output = HbmTensor::new();
        if interval.rows() == 0 {
            return interval.with(output);
        }
        let keys = spans.extent::<m![Spans, Span]>();
        let live = (keys + Span::SIZE - 1) / Span::SIZE;
        // The last span's live keys within its first half take the half span's geometry.
        let short = V::PREFIX && keys - (live - 1) * Span::SIZE <= half::HalfSpan::SIZE;
        if short {
            Sweep::<_, _, _, _, BITS, true> {
                key,
                value,
                visible,
                live,
            }
            .forward(device, (query, &mut output, interval));
        } else {
            Sweep::<_, _, _, _, BITS, false> {
                key,
                value,
                visible,
                live,
            }
            .forward(device, (query, &mut output, interval));
        }
        interval.with(output)
    }
}
