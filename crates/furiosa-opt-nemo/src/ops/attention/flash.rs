//! Bidirectional attention of a 4,096-row query tile over a long key sequence, one chip's heads,
//! each span of keys resting in the tensor register files while the queries stream past it.
//!
//! - A cluster attends one head at a time; the chip's two clusters take a pair of heads.
//! - A span of 2,048 keys spreads over a cluster's 256 slices as eight query groups of 512 rows by
//!   32 key slices of 64 keys. A slice keeps its keys and its values in its tensor register file;
//!   its group's queries stream past the keys through the switch, and the weights it computed
//!   stream past the values.
//! - The values carry 16 channels of ones after their 128, so one product holds a row's weighted
//!   values and its mass; the products sum over the key slices onto the slice that holds the row.
//! - Each span's peaks combine over its key slices into every row's running maximum.
//! - Scores<C> round to BF16 before they are scaled; the maximum and the products are FP32.

use furiosa_opt_std::prelude::*;

use crate::Module;
use crate::compose::Count;
use crate::ops::layout::axes::Chip;

axes![
    Rows = 4096,
    TailRows = 1024,
    Span = 2048,
    Duo = 2,
    Head = 128,
    Channel = 144,
    Term = 2
];

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
/// A slice's values channel-major, each head channel followed by the 16 channels of ones.
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
    m![Channel / 16, Span % 64 / 16, Channel % 16 / 8, Span % 16],
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

/// The heads of `R` query rows, each attending every key of `Spans` spans; a call attends one tile
/// of 4,096 of them.
pub type Queries<R, Pairs, Chips = Chip> = HbmTensor<bf16, Chips, m![R, Pairs, Duo, Head]>;
/// Every head's keys or values, head-major so that a slice's keys of a span are one run.
pub type Tokens<Spans, Pairs, Chips = Chip> =
    HbmTensor<bf16, Chips, m![Pairs, Duo, Spans, Span, Head]>;
/// Each key's visibility, 1 or 0: the visible keys are a prefix, so only the last span holds absent
/// keys, whose rows may hold any finite values.
pub type Visible<Spans, Chips = Chip> = HbmTensor<bf16, Chips, m![Spans, Span]>;

/// A key visibility and where its absent keys may lie, which decides how each span's values enter
/// the sum.
trait Visibility<Spans: AxisName, Chips: M = Chip> {
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
        Transpose.forward((device, (value, transposed)));
    }
}

/// A key visibility whose absent keys may lie in any span, so every span's values and mass are
/// cleared where its keys are absent.
pub struct Scattered<'a, Spans: AxisName, Chips: M = Chip>(pub &'a Visible<Spans, Chips>);

impl<Spans: AxisName, Chips: M> Visibility<Spans, Chips> for Scattered<'_, Spans, Chips> {
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
        let (kept, seen) = Keep.forward((device, (value, self.0, span)));
        VisibleValues.forward((device, ((&kept, &seen), transposed)));
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

/// The head pairs of rows `W` features wide.
#[derive(Clone, Debug)]
pub struct HeadPairs<W>(std::marker::PhantomData<W>);

impl<W: AxisName> AxisName for HeadPairs<W> {
    const NAME: Ident = Ident::new("HeadPairs");

    const SIZE: usize = {
        assert!(W::SIZE.is_multiple_of(Duo::SIZE * Head::SIZE));
        W::SIZE / (Duo::SIZE * Head::SIZE)
    };
}

impl<
    R: AxisName,
    Spans: AxisName,
    Pairs: AxisName,
    V: Visibility<Spans, Chips>,
    Chips: M,
    const SCALE: usize,
> crate::Module<(&mut Device, &Queries<R, Pairs, Chips>)>
    for Sweep<'_, Spans, Pairs, V, Chips, SCALE>
{
    /// softmax(scale * QKᵀ)V of every row of `query` over the visible keys of the first `spans` spans,
    /// a 4,096-row tile at a time; a tile's first unit lands and scores under the previous tile's last.
    type Output = Queries<R, Pairs, Chips>;

    fn forward(
        &self,
        (device, query): (&mut Device, &Queries<R, Pairs, Chips>),
    ) -> Queries<R, Pairs, Chips> {
        let (key, value, visible) = (self.key, self.value, self.visible);
        let mut output: Queries<R, Pairs, Chips> = HbmTensor::new();
        let rows = R::SIZE;
        // A quarter tile left over is the tail of four head pairs at once; any other remainder is a
        // last tile that ends at the last row and repeats rows the tile before it attended.
        let tail = const {
            R::SIZE % Rows::SIZE == TailRows::SIZE && Pairs::SIZE >= <Tail as Unit<Pairs>>::PAIRS
        };
        let tiles = if tail {
            rows / Rows::SIZE
        } else {
            rows.div_ceil(Rows::SIZE)
        };
        let pairs = const { Pairs::SIZE };
        let first_row = |tile: usize| (tile * Rows::SIZE).min(rows - Rows::SIZE);
        let mut transposed = Transposed::<Chips>::new();
        transposed
            .view_mut()
            .tile::<m![Channel], 16, m![Channel = 16 #{!} Channel, Span % 64]>(Head::SIZE)
            .memset(const { bf16::from_f32(1.0) }, &mut device.sub);
        let landed = Landing::<Whole>(core::marker::PhantomData)
            .forward((device, ((query, key, value), 0, 0)));
        let mut ready = {
            let Landed {
                rows,
                keys,
                first,
                later,
                values,
            } = landed;
            let scored = {
                let score = LoadScore.forward((device, keys));
                Calculate.forward((device, (score, &rows)))
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
                ready = UnitSweep::<Whole, _>(core::marker::PhantomData, self).forward((
                    device,
                    (
                        (ready, &mut transposed),
                        (pair, |device: &mut Device| {
                            Landing::<Whole>(core::marker::PhantomData).forward((
                                device,
                                (
                                    (query, key, value),
                                    from + (next - from) * wraps,
                                    (pair + 1) % pairs,
                                ),
                            ))
                        }),
                        |device: &mut Device, attended: &Attended<Chips>| {
                            Whole::store(device, attended, (&mut output, from), pair)
                        },
                    ),
                ));
            }
        }
        let from = first_row(tiles - 1);
        #[unroll]
        for pair in 0..pairs - 1 {
            ready = UnitSweep::<Whole, _>(core::marker::PhantomData, self).forward((
                device,
                (
                    (ready, &mut transposed),
                    (pair, |device: &mut Device| {
                        Landing::<Whole>(core::marker::PhantomData)
                            .forward((device, ((query, key, value), from, pair + 1)))
                    }),
                    |device: &mut Device, attended: &Attended<Chips>| {
                        Whole::store(device, attended, (&mut output, from), pair)
                    },
                ),
            ));
        }
        let last = pairs - 1;
        let store_last = |device: &mut Device, attended: &Attended<Chips>| {
            Whole::store(device, attended, (&mut output, from), last)
        };
        if tail {
            let at = rows - TailRows::SIZE;
            ready = UnitSweep::<Whole, _>(core::marker::PhantomData, self).forward((
                device,
                (
                    (ready, &mut transposed),
                    (last, |device: &mut Device| {
                        Landing::<Tail>(core::marker::PhantomData)
                            .forward((device, ((query, key, value), at, 0)))
                    }),
                    store_last,
                ),
            ));
            let units = const { Pairs::SIZE.div_ceil(<Tail as Unit<Pairs>>::PAIRS) };
            #[unroll]
            for unit in 0..units {
                let next = (unit + 1).min(units - 1);
                ready = UnitSweep::<Tail, _>(core::marker::PhantomData, self).forward((
                    device,
                    (
                        (ready, &mut transposed),
                        (unit, |device: &mut Device| {
                            Landing::<Tail>(core::marker::PhantomData)
                                .forward((device, ((query, key, value), at, next)))
                        }),
                        |device: &mut Device, attended: &Attended<Chips>| {
                            Tail::store(device, attended, (&mut output, at), unit)
                        },
                    ),
                ));
            }
        } else {
            ready = UnitSweep::<Whole, _>(core::marker::PhantomData, self).forward((
                device,
                (
                    (ready, &mut transposed),
                    (last, |device: &mut Device| {
                        Landing::<Whole>(core::marker::PhantomData)
                            .forward((device, ((query, key, value), from, last)))
                    }),
                    store_last,
                ),
            ));
        }
        drop(ready);
        output
    }
}

/// What every unit of a call sweeps: the keys and values of its spans, their visibility, and the
/// live spans; `SCALE` holds the bits of the FP32 scale.
struct Sweep<'a, Spans: AxisName, Pairs: AxisName, V, Chips: M, const SCALE: usize> {
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
    U: Unit<Pairs>,
    F: FnOnce(&mut Device) -> Landed<Chips>,
    S: FnOnce(&mut Device, &Attended<Chips>),
>
    Module<(
        &mut Device,
        ((Ready<Chips>, &mut Transposed<Chips>), (usize, F), S),
    )> for UnitSweep<'_, U, Sweep<'_, Spans, Pairs, V, Chips, SCALE>>
{
    type Output = Ready<Chips>;

    /// Unit `unit` of kind `U` from `ready`: its last span runs while `following` lands the next
    /// unit and Main scores that unit's first span; `store` takes the rows, the next unit returns.
    fn forward(
        &self,
        (device, ((ready, transposed), (unit, following), store)): (
            &mut Device,
            ((Ready<Chips>, &mut Transposed<Chips>), (usize, F), S),
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
        let score = LoadScore.forward((device, later));
        let mut held = Peak::<Chips>::new();
        held.view_mut().memset(START, &mut device.sub);
        let previous = Maximum.forward((device, &held));
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
                .forward((&mut *device, (trip, (&rows, staged), landed)));
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
        } = self.1.forward((
            &mut *device,
            (trip, (&rows, (visible, live - 2, &mut *transposed)), landed),
        ));
        let following = following(device);
        let next = Calculate.forward((device, (score, &rows)));
        drop(rows);
        let (weights, softmax) = Weigh.forward((device, (softmax, (scored, previous), scale)));
        let vreg = Register.forward((device, &*transposed));
        let product = Multiply.forward((device, (weights, vreg)));
        let cleared = {
            let values = Values(values);
            let (kept, seen) = Keep.forward((device, (&values.0, visible.keys(), live - 1)));
            let mut transposed = Transposed::<Chips>::new();
            VisibleValues.forward((device, ((&kept, &seen), &mut transposed)));
            transposed
        };
        let (softmax, previous, retired) = Absorb.forward((device, (softmax, product)));
        let scored = next;
        // The last span, while Main scores the next unit's first span.
        let Landed {
            rows,
            keys,
            first,
            later,
            values,
        } = following;
        let next = {
            let score = LoadScore.forward((device, keys));
            Calculate.forward((device, (score, &rows)))
        };
        let (weights, softmax) = Weigh.forward((device, (softmax, (scored, previous), scale)));
        let vreg = {
            let cleared = cleared;
            Register.forward((device, &cleared))
        };
        let product = Multiply.forward((device, (weights, vreg)));
        {
            let first = first;
            visible.stage(device, &first, 0, transposed);
        }
        let (finished, sum) = {
            let OnlineSoftmax {
                max: (held, top),
                sum,
            } = softmax;
            let raised = Maximum.forward((device, &top));
            let factor = Decay.forward((device, (&held, &raised)));
            drop((held, top, raised));
            let next = Accumulate.forward((&mut *device, (&sum, &product, &factor)));
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
    Module<(
        &mut Device,
        (
            (
                &Queries<Q, Pairs, Chips>,
                &Tokens<Spans, Pairs, Chips>,
                &Tokens<Spans, Pairs, Chips>,
            ),
            usize,
            usize,
        ),
    )> for Landing<U>
{
    type Output = Landed<Chips>;

    fn forward(
        &self,
        (device, ((query, key, value), from, unit)): (
            &mut Device,
            (
                (
                    &Queries<Q, Pairs, Chips>,
                    &Tokens<Spans, Pairs, Chips>,
                    &Tokens<Spans, Pairs, Chips>,
                ),
                usize,
                usize,
            ),
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
impl<Spans: AxisName, Pairs: AxisName, V: Visibility<Spans, Chips>, Chips: M, const SCALE: usize>
    Module<(
        &mut Device,
        (
            Trip<Chips>,
            (&Query<Chips>, (&V, usize, &mut Transposed<Chips>)),
            (Key<Chips>, Key<Chips>),
        ),
    )> for Sweep<'_, Spans, Pairs, V, Chips, SCALE>
{
    type Output = Trip<Chips>;

    fn forward(
        &self,
        (device, (trip, (rows, staged), (next_keys, next_values))): (
            &mut Device,
            (
                Trip<Chips>,
                (&Query<Chips>, (&V, usize, &mut Transposed<Chips>)),
                (Key<Chips>, Key<Chips>),
            ),
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
        let next = Calculate.forward((device, (score, rows)));
        let (weights, softmax) = Weigh.forward((
            device,
            (
                softmax,
                (scored, previous),
                const { f32::from_bits(SCALE as u32) },
            ),
        ));
        let vreg = Register.forward((device, &*transposed));
        let product = Multiply.forward((device, (weights, vreg)));
        {
            let values = values;
            visible.stage(device, &values, span, transposed);
        }
        let (softmax, previous, _sum) = Absorb.forward((device, (softmax, product)));
        Trip {
            score: LoadScore.forward((device, next_keys)),
            scored: next,
            previous,
            values: next_values,
            softmax,
        }
    }
}

struct Score<Chips: M>(KeyRegister<Chips>);

struct LoadScore;

impl<Chips: M> Module<(&mut Device, Key<Chips>)> for LoadScore {
    type Output = Score<Chips>;

    fn forward(&self, (device, key): (&mut Device, Key<Chips>)) -> Self::Output {
        Score(
            device
                .main
                .begin(key.view())
                .fetch::<m![Span % 8, Span % 64 / 16, Head / 16, Span % 16 / 8], m![Head % 16]>()
                .collect::<m![Span % 8, Span % 64 / 16, Head / 16, Span % 16 / 8], m![Head % 16]>()
                .to_trf(),
        )
    }
}

struct Calculate;

impl<Chips: M> Module<(&mut Device, (Score<Chips>, &Query<Chips>))> for Calculate {
    type Output = Scores<Chips>;

    /// Every row of the group against every key of the slice: each query streams from the slice
    /// that holds it past the group's 32 key slices.
    fn forward(
        &self,
        (device, (score, query)): (&mut Device, (Score<Chips>, &Query<Chips>)),
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

struct Weigh;

impl<Chips: M>
    Module<(
        &mut Device,
        (OnlineSoftmax<Chips>, (Scores<Chips>, AgreedTop<Chips>), f32),
    )> for Weigh
{
    type Output = (
        Weights<Chips>,
        OnlineSoftmax<Chips, (Peak<Chips>, Peak<Chips>)>,
    );

    fn forward(
        &self,
        (device, (softmax, (scores, previous), scale)): (
            &mut Device,
            (OnlineSoftmax<Chips>, (Scores<Chips>, AgreedTop<Chips>), f32),
        ),
    ) -> Self::Output {
        let top = {
            let held = &previous;
            let scores = &scores;
            let local: Peak<Chips> = device
                .sub
                .begin(scores.view())
                .fetch::<m![Rows % 512, Span % 64 / 16], m![Span % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![Rows % 512, Span % 64 / 8], m![Span % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Rows % 512, Span % 64 / 4], m![Span % 4]>()
                .vector_intra_slice_reduce::<Span, m![Rows % 512], m![1 # 4]>(
                    IntraSliceReduceOpF32::Max,
                )
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .transpose::<m![Rows % 512 / 2], m![Rows % 2 # 8]>()
                .commit_trim::<m![Rows % 2]>()
                .commit();
            let raised: DmTensor<f32, Chips, Cluster, Agreed, m![Rows % 512]> = device
                .sub
                .begin(local.view())
                .fetch::<m![Rows % 512 / 8], m![Rows % 8]>()
                .collect::<m![Rows % 512 / 8], m![Rows % 8]>()
                .vector_init()
                .vector_inter_slice_reduce::<Agreed, m![Rows % 512 / 8]>(InterSliceReduceOpF32::Max)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Rows % 512 / 4], m![Rows % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
                .vector_widen_concat::<m![Rows % 512 / 8], m![Rows % 8]>()
                .vector_clip(ClipBinaryOpF32::Max, held)
                .vector_final()
                .commit_trim::<m![Rows % 8]>()
                .commit();
            // SAFETY: every key slice of a group received the same maximum.
            unsafe { raised.reshape() }
        };
        drop(previous);
        let values = {
            let scores = scores;
            let operand = Operand.forward((device, &top));
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

impl<Chips: M> Module<(&mut Device, (&Peak<Chips>, &AgreedTop<Chips>))> for Decay {
    type Output = DecayOperand<Chips>;

    /// The factor e^(held - raised) on each row's own slice: every key slice of a group holds the same
    /// factors, so reducing the copies moves each row's factor onto its slice.
    fn forward(
        &self,
        (device, (held, raised)): (&mut Device, (&Peak<Chips>, &AgreedTop<Chips>)),
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

impl<Chips: M> Module<(&mut Device, &Peak<Chips>)> for Maximum {
    type Output = AgreedTop<Chips>;

    /// The running maximum `held` as the operand of the group's key slices.
    fn forward(&self, (device, held): (&mut Device, &Peak<Chips>)) -> Self::Output {
        // SAFETY: a group's key slices hold the same running maximum; only the broadcast axis is named.
        unsafe { Operand.forward((device, held)).reshape() }
    }
}

struct Operand;

impl<Chips: M> Module<(&mut Device, &Peak<Chips>)> for Operand {
    type Output = Top<Chips>;

    fn forward(&self, (device, peak): (&mut Device, &Peak<Chips>)) -> Self::Output {
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

impl<Chips: M> Module<(&mut Device, &Transposed<Chips>)> for Register {
    type Output = ValueRegister<Chips>;

    fn forward(&self, (device, value): (&mut Device, &Transposed<Chips>)) -> Self::Output {
        device
            .main
            .begin(value.view())
            .fetch::<m![Channel % 8, Channel / 16, Span % 64 / 16, Channel % 16 / 8], m![Span % 16]>()
            .collect::<m![Channel % 8, Channel / 16, Span % 64 / 16, Channel % 16 / 8], m![Span % 16]>()
            .to_trf()
    }
}

struct Transpose;

impl<Chips: M> Module<(&mut Device, (&Key<Chips>, &mut Transposed<Chips>))> for Transpose {
    type Output = ();

    /// `value` channel-major into the value channels of `transposed`.
    fn forward(
        &self,
        (device, (value, transposed)): (&mut Device, (&Key<Chips>, &mut Transposed<Chips>)),
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

impl<Chips: M, Spans: AxisName> Module<(&mut Device, (&Key<Chips>, &Visible<Spans, Chips>, usize))>
    for Keep
{
    type Output = (Key<Chips>, Seen<Chips>);

    /// `value` of span `span` with every absent key's values cleared, and the span's visibility.
    fn forward(
        &self,
        (device, (value, visible, span)): (
            &mut Device,
            (&Key<Chips>, &Visible<Spans, Chips>, usize),
        ),
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

impl<Chips: M>
    Module<(
        &mut Device,
        ((&Key<Chips>, &Seen<Chips>), &mut Transposed<Chips>),
    )> for VisibleValues
{
    type Output = ();

    /// Kept values channel-major into `transposed`, whose mass channels take the keys' visibility in
    /// place of ones.
    fn forward(
        &self,
        (device, ((kept, seen), transposed)): (
            &mut Device,
            ((&Key<Chips>, &Seen<Chips>), &mut Transposed<Chips>),
        ),
    ) {
        Transpose.forward((device, (kept, &mut *transposed)));
        let mass = transposed
            .view_mut()
            .tile::<m![Channel], 16, m![Channel = 16 #{!} Channel, Span % 64]>(Head::SIZE);
        // SAFETY: each mass channel holds the same visibility; only the broadcast axis is renamed.
        let mass =
            unsafe { mass.reshape::<Chips, Cluster, Slices, m![[16] #{!} Channel, Span % 64]>() };
        device
            .sub
            .begin(seen.view())
            .fetch::<m![16, Span % 64 / 16], m![Span % 16]>()
            .collect::<m![16, Span % 64 / 16], m![Span % 16]>()
            .commit_trim::<m![Span % 16]>()
            .commit_view(mass);
    }
}

struct Multiply;

impl<Chips: M> Module<(&mut Device, (Weights<Chips>, ValueRegister<Chips>))> for Multiply {
    type Output = Product<Chips>;

    /// Every row's weighted values and mass against the slice's keys, summed over the group's key
    /// slices onto the row's own slice.
    fn forward(
        &self,
        (device, (weights, value)): (&mut Device, (Weights<Chips>, ValueRegister<Chips>)),
    ) -> Self::Output {
        let value = value;
        let weights = weights.values;
        type Fetch = m![Rows % 16 / 2, Channel / 16, Rows % 512 / 16, Span % 64 / 16];
        type Steps = m![{ Fetch }, Channel % 16 / 8];
        type Sums = m![
            Rows % 16 / 2,
            Channel / 16,
            Rows % 512 / 16,
            Channel % 16 / 8
        ];
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
        .vector_inter_slice_reduce::<
            QuerySlices,
            m![Rows % 16 / 2, Channel / 16, Channel % 16 / 8, Rows % 2],
        >(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![Channel % 8]>()
        .commit_view(product.view_mut());
        product
    }
}

struct Absorb;

impl<Chips: M>
    Module<(
        &mut Device,
        (
            OnlineSoftmax<Chips, (Peak<Chips>, Peak<Chips>)>,
            Product<Chips>,
        ),
    )> for Absorb
{
    type Output = (OnlineSoftmax<Chips>, AgreedTop<Chips>, Product<Chips>);

    fn forward(
        &self,
        (device, (softmax, product)): (
            &mut Device,
            (
                OnlineSoftmax<Chips, (Peak<Chips>, Peak<Chips>)>,
                Product<Chips>,
            ),
        ),
    ) -> Self::Output {
        let OnlineSoftmax {
            max: (held, top),
            sum,
        } = softmax;
        let raised = Maximum.forward((device, &top));
        let factor = Decay.forward((device, (&held, &raised)));
        drop(held);
        let next = Accumulate.forward((device, (&sum, &product, &factor)));
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

impl<Chips: M>
    Module<(
        &mut Device,
        (&Product<Chips>, &Product<Chips>, &DecayOperand<Chips>),
    )> for Accumulate
{
    type Output = Product<Chips>;

    /// `sum` rescaled by `decay` plus `product`.
    fn forward(
        &self,
        (device, (sum, product, decay)): (
            &mut Device,
            (&Product<Chips>, &Product<Chips>, &DecayOperand<Chips>),
        ),
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
    crate::Module<(
        &mut Device,
        (
            &Queries<R, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
            (&V, &Count<Chips>),
        ),
    )> for super::full::Attention<std::marker::PhantomData<Heads>, super::full::Scaled<BITS>>
{
    type Output = Queries<R, Pairs, Chips>;

    fn forward(
        &self,
        (device, (query, key, value, (visible, spans))): (
            &mut Device,
            (
                &Queries<R, Pairs, Chips>,
                &Tokens<Spans, Pairs, Chips>,
                &Tokens<Spans, Pairs, Chips>,
                (&V, &Count<Chips>),
            ),
        ),
    ) -> Self::Output {
        const {
            assert!(
                Heads::SIZE == 2 * Pairs::SIZE,
                "a chip attends its heads in pairs"
            )
        };
        Sweep::<_, _, _, _, BITS> {
            key,
            value,
            visible,
            live: spans.extent::<m![Spans]>(),
        }
        .forward((device, query))
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use super::{Duo, Head, Queries, Rows, Span, Tokens, Visible};
    use crate::Module;
    use crate::compose::tests::count;
    use crate::ops::layout::axes::Single;
    use crate::ops::normalization::tests::random;

    /// A chip's heads, two a pair.
    #[derive(Clone, Debug)]
    struct Heads<Pairs>(std::marker::PhantomData<Pairs>);

    impl<Pairs: AxisName> AxisName for Heads<Pairs> {
        const NAME: Ident = Ident::new("Heads");
        const SIZE: usize = 2 * Pairs::SIZE;
    }

    axes![Three = 3, Twenty = 20, Seven = 7];

    /// The score multiplier of 128-wide heads, `1 / sqrt(128)`.
    const INVERSE_ROOT: usize = (std::f32::consts::FRAC_1_SQRT_2 / 8.0).to_bits() as usize;

    #[device(chip = 1)]
    fn attend_heads<Spans: AxisName, Pairs: AxisName>(
        device: &mut Device,
        query: &Queries<Rows, Pairs>,
        key: &Tokens<Spans, Pairs>,
        value: &Tokens<Spans, Pairs>,
        visible: &Visible<Spans>,
        spans: &crate::compose::Count,
        output: &mut Queries<Rows, Pairs>,
    ) {
        // Through the catalog leaf, as a model calls it.
        let attention = crate::ops::attention::full::Attention {
            heads: std::marker::PhantomData::<Heads<Pairs>>,
            scale: crate::ops::attention::full::Scaled::<INVERSE_ROOT>,
        };
        attention
            .forward((&mut *device, (query, key, value, (visible, spans))))
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    /// `tokens` in token-major order (`[key][head][channel]`) as the kernel takes them, head-major.
    fn head_major(tokens: &[bf16], heads: usize) -> Vec<bf16> {
        let keys = tokens.len() / (heads * Head::SIZE);
        (0..heads)
            .flat_map(|head| {
                (0..keys).flat_map(move |key| {
                    let at = (key * heads + head) * Head::SIZE;
                    tokens[at..at + Head::SIZE].iter().copied()
                })
            })
            .collect()
    }

    /// Runs the kernel on token-major `q`, `k`, `v` over the first `live` keys; returns its output.
    async fn run<Spans: AxisName, Pairs: AxisName>(
        (q, k, v): (&[bf16], &[bf16], &[bf16]),
        live: usize,
    ) -> Vec<bf16> {
        let heads = Pairs::SIZE * Duo::SIZE;
        let keys = Spans::SIZE * Span::SIZE;
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(attend_heads.topology()).unwrap();
        let query: Queries<Rows, Pairs> =
            HostTensor::<bf16, m![Rows, Pairs, Duo, Head]>::from_vec(q.to_vec())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let key: Tokens<Spans, Pairs> =
            HostTensor::<bf16, m![Pairs, Duo, Spans, Span, Head]>::from_vec(head_major(k, heads))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let value: Tokens<Spans, Pairs> =
            HostTensor::<bf16, m![Pairs, Duo, Spans, Span, Head]>::from_vec(head_major(v, heads))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let visible: Visible<Spans> = HostTensor::<bf16, m![Spans, Span]>::from_vec(
            (0..keys)
                .map(|at| bf16::from_f32(if at < live { 1.0 } else { 0.0 }))
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let count = count(live.div_ceil(Span::SIZE))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut output: Queries<Rows, Pairs> =
            HostTensor::<bf16, m![Rows, Pairs, Duo, Head]>::zero()
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        launch(
            attend_heads,
            (
                &mut device,
                &query,
                &key,
                &value,
                &visible,
                &count,
                &mut output,
            ),
        )
        .await
        .unwrap();
        output
            .to_host::<m![Rows, Pairs, Duo, Head]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }

    /// A 4,096-row tile over the first `live` keys, the absent ones four times larger so a leak
    /// shows, against an FP64 softmax with BF16-rounded scores on every `stride`-th row.
    async fn check<Spans: AxisName, Pairs: AxisName>(live: usize, stride: usize) {
        let keys = Spans::SIZE * Span::SIZE;
        let heads = Pairs::SIZE * Duo::SIZE;
        // Scaled queries make the softmax peaked.
        let q: Vec<bf16> = random(Rows::SIZE * heads * Head::SIZE, 0x9e37_79b9_7f4a_7c15)
            .into_iter()
            .map(|x| bf16::from_f32(x.to_f32() * 8.0))
            .collect();
        let k: Vec<bf16> = random(keys * heads * Head::SIZE, 0x2545_f491_4f6c_dd1d)
            .into_iter()
            .enumerate()
            .map(|(at, x)| {
                let absent = at / (heads * Head::SIZE) >= live;
                bf16::from_f32(x.to_f32() * if absent { 4.0 } else { 1.0 })
            })
            .collect();
        let v: Vec<bf16> = random(keys * heads * Head::SIZE, 0x1405_7b7e_f767_814f)
            .into_iter()
            .enumerate()
            .map(|(at, x)| {
                let absent = at / (heads * Head::SIZE) >= live;
                bf16::from_f32(x.to_f32() * if absent { 4.0 } else { 1.0 })
            })
            .collect();
        let got = run::<Spans, Pairs>((&q, &k, &v), live).await;
        let scale = f32::from_bits(INVERSE_ROOT as u32) as f64;
        let (mut error, mut norm, mut worst) = (0.0f64, 0.0f64, 0.0f64);
        for row in (0..Rows::SIZE).step_by(stride) {
            for head in 0..heads {
                let at = |t: &[bf16], r: usize, d: usize| {
                    t[(r * heads + head) * Head::SIZE + d].to_f32() as f64
                };
                // The kernel rounds each score to BF16 before scaling it.
                let scores: Vec<f64> = (0..live)
                    .map(|key| {
                        let dot: f64 = (0..Head::SIZE)
                            .map(|d| at(&q, row, d) * at(&k, key, d))
                            .sum();
                        scale * bf16::from_f32(dot as f32).to_f32() as f64
                    })
                    .collect();
                let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                let mass: f64 = weights.iter().sum();
                for d in 0..Head::SIZE {
                    let want = weights
                        .iter()
                        .enumerate()
                        .map(|(key, w)| w * at(&v, key, d))
                        .sum::<f64>()
                        / mass;
                    let found = at(&got, row, d);
                    assert!(
                        found.is_finite(),
                        "row {row} head {head} channel {d}: {found}"
                    );
                    error += (found - want).powi(2);
                    norm += want.powi(2);
                    worst = worst.max((found - want).abs());
                }
            }
        }
        let relative = (error / norm).sqrt();
        eprintln!(
            "flash attention {heads} heads x {live} keys: relative L2 {relative:.3e}, worst {worst:.3e}"
        );
        assert!(relative < 1e-2, "relative L2 {relative}");
    }

    /// Two heads over three spans, the third's last 300 keys absent.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_reference_softmax() {
        check::<Three, Single>(3 * Span::SIZE - 300, 29).await;
    }

    /// Fourteen heads over 37,787 live keys of twenty spans.
    #[tokio::test(flavor = "current_thread")]
    async fn attends_a_long_sequence() {
        check::<Twenty, Seven>(37_787, 1365).await;
    }
}
