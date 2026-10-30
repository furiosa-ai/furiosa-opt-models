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
//! - A call names its live keys, a prefix of the spans' keys; a last span whose live keys fit its
//!   first half takes that half alone, 32 keys a slice.

use furiosa_opt_std::prelude::*;

use crate::Op;
use crate::ops::attention::Attention;
use crate::specialize::SLICES;
use crate::tensor::parallel::Chip;
use crate::tensor::scalar;
use crate::tensor::scalar::Value;
use core::marker::PhantomData;

axes![
    Rows = 4096,
    TailRows = 1024,
    Span = 2048,
    Duo = 2,
    Head = 128,
    Channel = 144,
    Term = 2
];

/// A span's first half, the keys a last span takes alone when its live keys fit them.
#[derive(Clone, Debug)]
struct HalfSpan;

impl AxisName for HalfSpan {
    const NAME: Ident = Ident::new("HalfSpan");
    const SIZE: usize = Span::SIZE / 2;
}

/// A span step over `H`-wide heads: a span's keys against a cluster's query groups, each group's
/// rows spread over the span's key slices.
pub(super) struct SpanStep<H>(PhantomData<H>);

impl<H: AxisName> SpanStep<H> {
    /// Key slices: a span's 2,048 keys, 64 a slice.
    const KEY_SLICES: usize = 32;
    /// Query elements a slice holds: 4 KiB of BF16 queries.
    const QUERIES: usize = 2048;
    /// Rows a slice holds.
    const SLICE: usize = {
        assert!(Self::QUERIES.is_multiple_of(H::SIZE));
        Self::QUERIES / H::SIZE
    };
    /// Rows of a query group: its key slices' rows.
    const GROUP: usize = Self::KEY_SLICES * Self::SLICE;

    /// Heads of a tile of `rows` query rows that one cluster attends at once: the cluster's query
    /// groups over the groups one head's rows fill.
    pub(super) const fn heads(rows: usize) -> usize {
        let groups = SLICES / Self::KEY_SLICES;
        assert!(rows.is_multiple_of(Self::GROUP) && (groups * Self::GROUP).is_multiple_of(rows));
        groups * Self::GROUP / rows
    }
}

// A span step's tensors default to this tile's geometry: query groups `G` of `GROUP` rows of `R`
// (`SLICE` a slice), heads `H` of `C` channels with 16 mass channels, keys `K` (`KEYS` a slice).

/// A pair of heads, one a cluster.
type Cluster = m![Duo];
/// Eight query groups of 512 rows, one head's: a cluster attends one head of the tile at a time.
type Groups = m![Rows / { GROUP_ROWS }];
const _: () = assert!(SpanStep::<Head>::heads(Rows::SIZE) == 1);
/// Rows of a query group.
const GROUP_ROWS: usize = SpanStep::<Head>::GROUP;
/// Rows of each of a query group's 32 key slices.
const SLICE_ROWS: usize = SpanStep::<Head>::SLICE;
/// The query groups by the span's 32 key slices of `KEYS` keys.
type Slices<G = Groups, K = Span, const KEYS: usize = 64> = m![{ G }, K / { KEYS }];
/// A query group's rows, `SLICE` a slice: each row's products sum onto its slice.
type QuerySlices<
    G = Groups,
    R = Rows,
    const GROUP: usize = { GROUP_ROWS },
    const SLICE: usize = { SLICE_ROWS },
> = m![{ G }, R % { GROUP } / { SLICE }];
/// The span's keys `K` or their values, written in every query group.
type Copies<K = Span, const KEYS: usize = 64> = m![8, K / { KEYS }];
/// A query group's key slices, each holding the statistic they agree on.
type Agreed<G = Groups> = m![{ G }, 32];

type Query<
    Chips = Chip,
    G = Groups,
    R = Rows,
    H = Head,
    const GROUP: usize = { GROUP_ROWS },
    const SLICE: usize = { SLICE_ROWS },
> = DmTensor<bf16, Chips, Cluster, QuerySlices<G, R, GROUP, SLICE>, m![R % { SLICE }, H]>;
type Key<Chips = Chip, G = Groups, H = Head, K = Span, const KEYS: usize = 64> =
    DmTensor<bf16, Chips, Cluster, Slices<G, K, KEYS>, m![K % { KEYS }, H]>;
/// A slice's values channel-major, each head channel followed by the 16 channels of ones.
type ChannelMajor<Chips = Chip, G = Groups, C = Channel, K = Span, const KEYS: usize = 64> =
    DmTensor<bf16, Chips, Cluster, Slices<G, K, KEYS>, m![C, K % { KEYS }]>;
type KeyRegister<Chips = Chip, G = Groups, H = Head, K = Span, const KEYS: usize = 64> = TrfTensor<
    bf16,
    Chips,
    Cluster,
    Slices<G, K, KEYS>,
    m![K % 8],
    m![K % { KEYS } / 16, H / 16, K % 16 / 8, H % 16],
>;
type ValueRegister<Chips = Chip, G = Groups, C = Channel, K = Span, const KEYS: usize = 64> =
    TrfTensor<
        bf16,
        Chips,
        Cluster,
        Slices<G, K, KEYS>,
        m![C % 8],
        m![C / 16, K % { KEYS } / 16, C % 16 / 8, K % 16],
    >;
/// Every row of the group's scores against the slice's keys, or their weights.
type Scores<
    Chips = Chip,
    G = Groups,
    R = Rows,
    const GROUP: usize = { GROUP_ROWS },
    K = Span,
    const KEYS: usize = 64,
> = DmTensor<bf16, Chips, Cluster, Slices<G, K, KEYS>, m![R % { GROUP }, K % { KEYS }]>;
/// One statistic a row of the group, in every key slice.
type Peak<
    Chips = Chip,
    G = Groups,
    R = Rows,
    const GROUP: usize = { GROUP_ROWS },
    K = Span,
    const KEYS: usize = 64,
> = DmTensor<f32, Chips, Cluster, Slices<G, K, KEYS>, m![R % { GROUP }]>;
type Top<
    Chips = Chip,
    G = Groups,
    R = Rows,
    const GROUP: usize = { GROUP_ROWS },
    K = Span,
    const KEYS: usize = 64,
> = VrfTensor<f32, Chips, Cluster, Slices<G, K, KEYS>, m![R % { GROUP }]>;
/// One statistic a row, on the row's own slice.
type DecayOperand<
    Chips = Chip,
    G = Groups,
    R = Rows,
    const GROUP: usize = { GROUP_ROWS },
    const SLICE: usize = { SLICE_ROWS },
> = VrfTensor<f32, Chips, Cluster, QuerySlices<G, R, GROUP, SLICE>, m![R % { SLICE }]>;
/// One statistic a row of the group, in every key slice as the slices agree on it.
type AgreedTop<Chips = Chip, G = Groups, R = Rows, const GROUP: usize = { GROUP_ROWS }> =
    VrfTensor<f32, Chips, Cluster, Agreed<G>, m![R % { GROUP }]>;
/// A row's weighted values, then its mass in the channels of ones.
type Product<
    Chips = Chip,
    G = Groups,
    R = Rows,
    C = Channel,
    const GROUP: usize = { GROUP_ROWS },
    const SLICE: usize = { SLICE_ROWS },
> = DmTensor<f32, Chips, Cluster, QuerySlices<G, R, GROUP, SLICE>, m![R % { SLICE }, C]>;
type Attended<
    Chips = Chip,
    G = Groups,
    R = Rows,
    H = Head,
    const GROUP: usize = { GROUP_ROWS },
    const SLICE: usize = { SLICE_ROWS },
> = DmTensor<bf16, Chips, Cluster, QuerySlices<G, R, GROUP, SLICE>, m![R % { SLICE }, H]>;
/// A group's scaled maxima on the Sub engine at stage `S`, before they leave it.
type Raised<'l, Chips, G, R, const GROUP: usize, S = stage::Widen, Stash = Fresh> = VectorTensor<
    'l,
    { Tu::Sub },
    S,
    f32,
    Chips,
    Cluster,
    Agreed<G>,
    m![R % { GROUP } / 8],
    m![R % 8],
    Stash,
    { stage::VeOrder::InterFirst },
>;

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
trait Visibility<Spans: AxisName, Chips: M = Chip>: Sized {
    fn keys(&self) -> &Visible<Spans, Chips>;

    /// softmax(scale * QKᵀ)V of every row of `query` over these visible keys of `key` and `value`,
    /// the first `keys` keys live, into `output`; `SCALE` holds the bits of the FP32 scale.
    fn sweep<R: AxisName, Pairs: AxisName, const SCALE: usize>(
        &self,
        device: &mut Device,
        tokens: (
            &Queries<R, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
        ),
        keys: usize,
        output: &mut Queries<R, Pairs, Chips>,
    );

    /// `value`, span `span`'s values, channel-major into the value and mass channels of
    /// `transposed` for the span's trip.
    fn stage(
        &self,
        device: &mut Device,
        value: &Key<Chips>,
        span: usize,
        transposed: &mut ChannelMajor<Chips>,
    );
}

/// Only the last attended span holds absent keys, so the spans before it enter whole.
impl<Spans: AxisName, Chips: M> Visibility<Spans, Chips> for Visible<Spans, Chips> {
    fn keys(&self) -> &Visible<Spans, Chips> {
        self
    }

    /// The keys past the live ones are absent, so a last span whose live keys fit its first half
    /// takes that half alone.
    fn sweep<R: AxisName, Pairs: AxisName, const SCALE: usize>(
        &self,
        device: &mut Device,
        (query, key, value): (
            &Queries<R, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
        ),
        keys: usize,
        output: &mut Queries<R, Pairs, Chips>,
    ) {
        // Spelled out, so the device compiler bounds the span loop by the live keys' extent.
        let live = (keys + Span::SIZE - 1) / Span::SIZE;
        // Both sweeps write the same rows, so no tensor crosses the choice.
        if 2 * (keys - (live - 1) * Span::SIZE) <= Span::SIZE {
            Sweep::<_, _, _, _, SCALE, HalfSpan, 32, { HalfSpan::SIZE }> {
                key,
                value,
                visible: self,
                live,
                last: PhantomData,
            }
            .forward(device, (query, output));
        } else {
            Sweep::<_, _, _, _, SCALE> {
                key,
                value,
                visible: self,
                live,
                last: PhantomData,
            }
            .forward(device, (query, output));
        }
    }

    fn stage(
        &self,
        device: &mut Device,
        value: &Key<Chips>,
        _: usize,
        transposed: &mut ChannelMajor<Chips>,
    ) {
        Transpose::<{ Head::SIZE }>.forward(device, (value, transposed));
    }
}

/// A key visibility whose absent keys may lie in any span, so every span's values and mass are
/// cleared where its keys are absent.
pub struct Scattered<'a, Spans: AxisName, Chips: M = Chip>(pub &'a Visible<Spans, Chips>);

impl<Spans: AxisName, Chips: M> Visibility<Spans, Chips> for Scattered<'_, Spans, Chips> {
    fn keys(&self) -> &Visible<Spans, Chips> {
        self.0
    }

    /// Every span whole: an absent key may lie anywhere, so no count of live keys bounds a half.
    fn sweep<R: AxisName, Pairs: AxisName, const SCALE: usize>(
        &self,
        device: &mut Device,
        (query, key, value): (
            &Queries<R, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
        ),
        keys: usize,
        output: &mut Queries<R, Pairs, Chips>,
    ) {
        Sweep::<_, _, _, _, SCALE> {
            key,
            value,
            visible: self,
            live: (keys + Span::SIZE - 1) / Span::SIZE,
            last: PhantomData,
        }
        .forward(device, (query, output));
    }

    fn stage(
        &self,
        device: &mut Device,
        value: &Key<Chips>,
        span: usize,
        transposed: &mut ChannelMajor<Chips>,
    ) {
        let (kept, seen) = Keep::<{ Span::SIZE }>.forward(device, (value, self.0, span));
        VisibleValues::<{ Head::SIZE }>.forward(device, ((&kept, &seen), transposed));
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

    /// The first `WIDTH` keys `K` of span `span`, or their values, of unit `unit`'s pairs, `KEYS`
    /// a key slice, in their query groups.
    fn tokens<Spans: AxisName, Chips: M, K: AxisName, const KEYS: usize, const WIDTH: usize>(
        device: &mut Device,
        tokens: &Tokens<Spans, Pairs, Chips>,
        unit: usize,
        span: usize,
    ) -> Key<Chips, Groups, Head, K, KEYS>;

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

    fn tokens<Spans: AxisName, Chips: M, K: AxisName, const KEYS: usize, const WIDTH: usize>(
        device: &mut Device,
        tokens: &Tokens<Spans, Pairs, Chips>,
        pair: usize,
        span: usize,
    ) -> Key<Chips, Groups, Head, K, KEYS> {
        const { assert!(K::SIZE == WIDTH && WIDTH == 32 * KEYS && WIDTH <= Span::SIZE) };
        let keys = tokens
            .view()
            .tile::<m![Pairs], 1, m![Pairs = 1 # Pairs, Duo, Spans, Span, Head]>(pair)
            .tile::<m![Spans], 1, m![Pairs = 1 # Pairs, Duo, Spans = 1 # Spans, Span, Head]>(span)
            .tile::<m![Span], WIDTH, m![
                Pairs = 1 # Pairs,
                Duo,
                Spans = 1 # Spans,
                Span = { WIDTH } # Span,
                Head
            ]>(0);
        // SAFETY: the tile's keys are the span's first `WIDTH` in order; only the axis is renamed.
        let keys = unsafe {
            keys.reshape::<Chips, m![Pairs = 1 # Pairs, Duo, Spans = 1 # Spans, K # Span, Head]>()
        };
        let in_dm: DmTensor<bf16, Chips, Cluster, Copies<K, KEYS>, m![K % { KEYS }, Head]> =
            keys.to_dm(&mut device.tdma);
        // SAFETY: every query group holds the same keys; only the broadcast axis is named.
        unsafe { in_dm.reshape() }
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

    fn tokens<Spans: AxisName, Chips: M, K: AxisName, const KEYS: usize, const WIDTH: usize>(
        device: &mut Device,
        tokens: &Tokens<Spans, Pairs, Chips>,
        unit: usize,
        span: usize,
    ) -> Key<Chips, Groups, Head, K, KEYS> {
        const { assert!(K::SIZE == WIDTH && WIDTH == 32 * KEYS && WIDTH <= Span::SIZE) };
        let pairs = tokens
            .view()
            .tile::<m![Pairs], 4, m![Pairs = 4 # Pairs, Duo, Spans, Span, Head]>(
                (4 * unit).min(Pairs::SIZE - 4),
            );
        // SAFETY: pair p of the unit serves query rows p * 1024..; only the axis is renamed.
        let pairs =
            unsafe { pairs.reshape::<Chips, m![Rows / 1024 # Pairs, Duo, Spans, Span, Head]>() };
        let keys = pairs
            .tile::<m![Spans], 1, m![Rows / 1024 # Pairs, Duo, Spans = 1 # Spans, Span, Head]>(span)
            .tile::<m![Span], WIDTH, m![
                Rows / 1024 # Pairs,
                Duo,
                Spans = 1 # Spans,
                Span = { WIDTH } # Span,
                Head
            ]>(0);
        // SAFETY: the tile's keys are the span's first `WIDTH` in order; only the axis is renamed.
        let keys = unsafe {
            keys.reshape::<Chips, m![Rows / 1024 # Pairs, Duo, Spans = 1 # Spans, K # Span, Head]>()
        };
        let in_dm: DmTensor<
            bf16,
            Chips,
            Cluster,
            m![Rows / 1024, 2, K / { KEYS }],
            m![K % { KEYS }, Head],
        > = keys.to_dm(&mut device.tdma);
        // SAFETY: a pair's two query groups hold the same keys; only the broadcast axis is named.
        unsafe { in_dm.reshape() }
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
        // SAFETY: the tile's rows are the unit's rows of each pair in order; only axes are renamed.
        let tile = unsafe { tile.reshape::<Chips, m![Rows % 1024 #{!} O, Pairs, Duo, Head]>() };
        let tile = tile.tile::<m![Pairs], 4, m![
            Rows % 1024 #{!} O,
            Pairs = 4 #{!} Pairs,
            Duo,
            Head
        ]>((4 * unit).min(Pairs::SIZE - 4));
        // SAFETY: row r of the unit's pair p is output row p * 1024 + r; only the axis is renamed.
        let tile = unsafe {
            tile.reshape::<Chips, m![Rows % 1024 #{!} O, Rows / 1024 #{!} Pairs, Duo, Head]>()
        };
        attended.view().to_hbm_view(&mut device.tdma, tile);
    }
}

/// The head pairs of rows `W` features wide.
#[derive(Clone, Debug)]
pub struct HeadPairs<W>(PhantomData<W>);

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
    L: AxisName,
    const LAST: usize,
    const WIDTH: usize,
> crate::Op<(&Queries<R, Pairs, Chips>, &mut Queries<R, Pairs, Chips>)>
    for Sweep<'_, Spans, Pairs, V, Chips, SCALE, L, LAST, WIDTH>
{
    /// softmax(scale * QKᵀ)V of each `query` row over the visible keys of the first `live` spans
    /// into `output`, 4,096 rows a tile, each tile arriving as the last one scores.
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (query, output): (&Queries<R, Pairs, Chips>, &mut Queries<R, Pairs, Chips>),
    ) {
        let (key, value, visible) = (self.key, self.value, self.visible);
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
        let mut transposed = ChannelMajor::<Chips>::new();
        transposed
            .view_mut()
            .tile::<m![Channel], 16, m![Channel = 16 #{!} Channel, Span % 64]>(Head::SIZE)
            .memset(const { bf16::from_f32(1.0) }, &mut device.sub);
        let in_dm = ToDm::<Whole>(PhantomData).forward(device, ((query, key, value), 0, 0));
        let mut ready = {
            let InDm {
                rows,
                keys,
                first,
                second,
            } = in_dm;
            let scored = {
                let score = LoadScore.forward(device, keys);
                Calculate.forward(device, (score, &rows))
            };
            {
                let first = first;
                visible.stage(device, &first, 0, &mut transposed);
            }
            Ready {
                rows,
                scored,
                second,
            }
        };
        // Every tile but the last: its last pair writes the next tile's first.
        for tile in 0..tiles - 1 {
            let (from, next) = (first_row(tile), first_row(tile + 1));
            #[unroll]
            for pair in 0..pairs {
                let wraps = (pair + 1) / pairs;
                ready = UnitSweep::<Whole, _>(PhantomData, self).forward(
                    device,
                    (
                        (ready, &mut transposed),
                        (pair, |device: &mut Device| {
                            ToDm::<Whole>(PhantomData).forward(
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
            ready = UnitSweep::<Whole, _>(PhantomData, self).forward(
                device,
                (
                    (ready, &mut transposed),
                    (pair, |device: &mut Device| {
                        ToDm::<Whole>(PhantomData)
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
            let at = rows - TailRows::SIZE;
            ready = UnitSweep::<Whole, _>(PhantomData, self).forward(
                device,
                (
                    (ready, &mut transposed),
                    (last, |device: &mut Device| {
                        ToDm::<Tail>(PhantomData).forward(device, ((query, key, value), at, 0))
                    }),
                    store_last,
                ),
            );
            let units = const { Pairs::SIZE.div_ceil(<Tail as Unit<Pairs>>::PAIRS) };
            #[unroll]
            for unit in 0..units - 1 {
                ready = UnitSweep::<Tail, _>(PhantomData, self).forward(
                    device,
                    (
                        (ready, &mut transposed),
                        (unit, |device: &mut Device| {
                            ToDm::<Tail>(PhantomData)
                                .forward(device, ((query, key, value), at, unit + 1))
                        }),
                        |device: &mut Device, attended: &Attended<Chips>| {
                            Tail::store(device, attended, (&mut *output, at), unit)
                        },
                    ),
                );
            }
            // The call's last unit hands on nothing.
            let unit = units - 1;
            UnitSweep::<Tail, _>(PhantomData, self).forward(
                device,
                (
                    (ready, &mut transposed),
                    (unit, |_: &mut Device| ()),
                    |device: &mut Device, attended: &Attended<Chips>| {
                        Tail::store(device, attended, (&mut *output, at), unit)
                    },
                ),
            );
        } else {
            UnitSweep::<Whole, _>(PhantomData, self).forward(
                device,
                (
                    (ready, &mut transposed),
                    (last, |_: &mut Device| ()),
                    store_last,
                ),
            );
        }
    }
}

/// What every unit of a call sweeps: its spans' keys, values and visibility, and the live spans;
/// `SCALE` holds the FP32 scale's bits, and the last span attends its first `WIDTH` keys `L`.
struct Sweep<
    'a,
    Spans: AxisName,
    Pairs: AxisName,
    V,
    Chips: M,
    const SCALE: usize,
    L = Span,
    const LAST: usize = 64,
    const WIDTH: usize = 2048,
> {
    key: &'a Tokens<Spans, Pairs, Chips>,
    value: &'a Tokens<Spans, Pairs, Chips>,
    visible: &'a V,
    live: usize,
    last: PhantomData<L>,
}

struct UnitSweep<'a, U, S>(PhantomData<U>, &'a S);

impl<
    Spans: AxisName,
    Pairs: AxisName,
    V: Visibility<Spans, Chips>,
    Chips: M,
    const SCALE: usize,
    L: AxisName,
    const LAST: usize,
    const WIDTH: usize,
    U: Unit<Pairs>,
    N: Following<Chips, Groups, Head, Channel>,
    F: FnOnce(&mut Device) -> N,
    S: FnOnce(&mut Device, &Attended<Chips>),
> Op<((Ready<Chips>, &mut ChannelMajor<Chips>), (usize, F), S)>
    for UnitSweep<'_, U, Sweep<'_, Spans, Pairs, V, Chips, SCALE, L, LAST, WIDTH>>
{
    type Output = N::Ready;

    /// Unit `unit` of kind `U` from `ready`: its last span runs while `following` writes the next
    /// unit, if any, and Main scores its first span; `store` takes the rows.
    fn forward(
        &self,
        device: &mut Device,
        ((ready, transposed), (unit, following), store): (
            (Ready<Chips>, &mut ChannelMajor<Chips>),
            (usize, F),
            S,
        ),
    ) -> Self::Output {
        let (key, value, visible, live) = (self.1.key, self.1.value, self.1.visible, self.1.live);
        let scale = const { f32::from_bits(SCALE as u32) };
        let Ready {
            rows,
            scored,
            second: (later, values),
        } = ready;
        let score = LoadScore.forward(device, later);
        let (softmax, previous) = OnlineSoftmax::start(device);
        let mut trip = Trip {
            score,
            scored,
            previous,
            values,
            softmax,
        };
        // The trips after the loop read the carried tensors where an even count of trips leaves
        // them; after an odd count the loop's end copies them back.
        for step in 0..live - 3 {
            let span = (step + 2).min(Spans::SIZE - 1);
            let in_dm = (
                U::tokens::<_, _, Span, 64, { Span::SIZE }>(device, key, unit, span),
                U::tokens::<_, _, Span, 64, { Span::SIZE }>(device, value, unit, span),
            );
            let staged = (visible, step + 1, &mut *transposed);
            trip = self.1.forward(&mut *device, (trip, (&rows, staged), in_dm));
        }
        // The last span's first `WIDTH` keys, which hold its live ones.
        let in_dm = (
            U::tokens::<_, _, L, LAST, WIDTH>(device, key, unit, live - 1),
            U::tokens::<_, _, L, LAST, WIDTH>(device, value, unit, live - 1),
        );
        let Trip {
            score,
            scored,
            previous,
            values,
            softmax,
        } = self.1.forward(
            &mut *device,
            (trip, (&rows, (visible, live - 2, &mut *transposed)), in_dm),
        );
        let following = following(device);
        let next = Calculate.forward(device, (score, &rows));
        drop(rows);
        let (weights, softmax) = Weigh.forward(device, (softmax, (scored, previous), scale));
        let vreg = Register.forward(device, &*transposed);
        let mut product = Product::new();
        Multiply.forward(&mut *device, (weights, vreg, &mut product));
        let cleared = {
            let values = Values(values);
            let (kept, seen) = Keep::<WIDTH>.forward(device, (&values.0, visible.keys(), live - 1));
            let mut transposed = ChannelMajor::<Chips, Groups, Channel, L, LAST>::new();
            VisibleValues::<{ Head::SIZE }>.forward(device, ((&kept, &seen), &mut transposed));
            transposed
        };
        let (softmax, previous, retired) = Absorb.forward(device, (softmax, product));
        Last::<Head, { Head::SIZE }>(PhantomData).forward(
            device,
            (
                (softmax, (next, previous), scale),
                (
                    move |device: &mut Device, _: &ChannelMajor<Chips>| {
                        let cleared = cleared;
                        Register.forward(device, &cleared)
                    },
                    retired,
                ),
                (
                    following,
                    |device: &mut Device,
                     first: &Key<Chips>,
                     transposed: &mut ChannelMajor<Chips>| {
                        visible.stage(device, first, 0, transposed)
                    },
                    transposed,
                ),
                store,
            ),
        )
    }
}

/// A unit's last span weighed into its sum and its rows, heads `H` of `HEAD` channels, stored,
/// while Main scores the first span of the unit it hands on: the whole sweep of a one-span unit.
struct Last<H, const HEAD: usize>(PhantomData<H>);

/// What a unit's last span hands on: the next unit, its first span scored under the last span and
/// its values staged after it; or nothing, after a call's last unit.
trait Following<Chips: M, G: M, H: AxisName, C: AxisName> {
    /// The next unit, ready for its sweep.
    type Ready;
    /// The next unit's first values, until they are staged.
    type First;

    /// The next unit with its first span scored, and its first values.
    fn score(self, device: &mut Device) -> (Self::Ready, Self::First);

    /// `first` laid out by `stage` into `transposed`.
    fn stage<T>(
        device: &mut Device,
        first: Self::First,
        stage: T,
        transposed: &mut ChannelMajor<Chips, G, C>,
    ) where
        T: FnOnce(&mut Device, &Key<Chips, G, H>, &mut ChannelMajor<Chips, G, C>);
}

impl<
    Chips: M,
    Second,
    G: M,
    R: AxisName,
    H: AxisName,
    C: AxisName,
    const GROUP: usize,
    const SLICE: usize,
> Following<Chips, G, H, C> for InDm<Chips, Second, G, R, H, GROUP, SLICE>
{
    type Ready = Ready<Chips, Second, G, R, H, GROUP, SLICE>;
    type First = Key<Chips, G, H>;

    fn score(self, device: &mut Device) -> (Self::Ready, Self::First) {
        let InDm {
            rows,
            keys,
            first,
            second,
        } = self;
        let scored = {
            let score = LoadScore.forward(device, keys);
            Calculate.forward(device, (score, &rows))
        };
        (
            Ready {
                rows,
                scored,
                second,
            },
            first,
        )
    }

    fn stage<T>(
        device: &mut Device,
        first: Self::First,
        stage: T,
        transposed: &mut ChannelMajor<Chips, G, C>,
    ) where
        T: FnOnce(&mut Device, &Key<Chips, G, H>, &mut ChannelMajor<Chips, G, C>),
    {
        let first = first;
        stage(device, &first, transposed);
    }
}

impl<Chips: M, G: M, H: AxisName, C: AxisName> Following<Chips, G, H, C> for () {
    type Ready = ();
    type First = ();

    fn score(self, _: &mut Device) -> ((), ()) {
        ((), ())
    }

    fn stage<T>(_: &mut Device, _: (), _: T, _: &mut ChannelMajor<Chips, G, C>)
    where
        T: FnOnce(&mut Device, &Key<Chips, G, H>, &mut ChannelMajor<Chips, G, C>),
    {
    }
}

impl<
    Chips: M,
    N: Following<Chips, G, H, C>,
    G: M,
    R: AxisName,
    H: AxisName,
    C: AxisName,
    const GROUP: usize,
    const SLICE: usize,
    const HEAD: usize,
    K: AxisName,
    const KEYS: usize,
    F,
    P,
    T,
    S,
>
    Op<(
        (
            OnlineSoftmax<Peak<Chips, G, R, GROUP>, Product<Chips, G, R, C, GROUP, SLICE>>,
            (
                Scores<Chips, G, R, GROUP, K, KEYS>,
                AgreedTop<Chips, G, R, GROUP>,
            ),
            f32,
        ),
        (F, P),
        (N, T, &mut ChannelMajor<Chips, G, C>),
        S,
    )> for Last<H, HEAD>
where
    F: FnOnce(&mut Device, &ChannelMajor<Chips, G, C>) -> ValueRegister<Chips, G, C, K, KEYS>,
    T: FnOnce(&mut Device, &Key<Chips, G, H>, &mut ChannelMajor<Chips, G, C>),
    S: FnOnce(&mut Device, &Attended<Chips, G, R, H, GROUP, SLICE>),
{
    type Output = N::Ready;

    fn forward(
        &self,
        device: &mut Device,
        (
            (softmax, (scored, previous), scale),
            (register, retired),
            (following, stage, transposed),
            store,
        ): (
            (
                OnlineSoftmax<Peak<Chips, G, R, GROUP>, Product<Chips, G, R, C, GROUP, SLICE>>,
                (
                    Scores<Chips, G, R, GROUP, K, KEYS>,
                    AgreedTop<Chips, G, R, GROUP>,
                ),
                f32,
            ),
            (F, P),
            (N, T, &mut ChannelMajor<Chips, G, C>),
            S,
        ),
    ) -> Self::Output {
        let (next, first) = following.score(device);
        let (weights, softmax) = Weigh.forward(device, (softmax, (scored, previous), scale));
        let vreg = register(device, &*transposed);
        let mut product = Product::<Chips, G, R, C, GROUP, SLICE>::new();
        Multiply.forward(&mut *device, (weights, vreg, &mut product));
        N::stage(device, first, stage, transposed);
        let (finished, sum) = {
            let OnlineSoftmax {
                max: (held, top),
                sum,
            } = softmax;
            let raised = Maximum.forward(device, &top);
            let factor = Decay::<SLICE>.forward(device, (&held, &raised));
            drop((held, top, raised));
            let next = Accumulate.forward(&mut *device, (&sum, &product, &factor));
            drop((factor, product));
            let finished = {
                let softmax = OnlineSoftmax { max: (), sum: next };
                Average::<H, HEAD>(PhantomData).forward(&mut *device, &softmax.sum)
            };
            (finished, sum)
        };
        // Retired sums keep their original allocation lifetimes through the unit's store.
        let _sums = (sum, retired);
        let attended = finished;
        store(device, &attended);
        next
    }
}

/// A unit's queries in groups `G` of `GROUP` rows of `R`, `SLICE` a slice, its first span's keys
/// and values of heads `H`, and `Second`, its second span's (none for a unit of one span).
struct InDm<
    Chips: M,
    Second = (Key<Chips>, Key<Chips>),
    G: M = Groups,
    R: AxisName = Rows,
    H: AxisName = Head,
    const GROUP: usize = { GROUP_ROWS },
    const SLICE: usize = { SLICE_ROWS },
> {
    rows: Query<Chips, G, R, H, GROUP, SLICE>,
    keys: Key<Chips, G, H>,
    first: Key<Chips, G, H>,
    second: Second,
}

struct ToDm<U>(PhantomData<U>);

impl<Chips: M, U: Unit<Pairs>, Q: AxisName, Spans: AxisName, Pairs: AxisName>
    Op<(
        (
            &Queries<Q, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
        ),
        usize,
        usize,
    )> for ToDm<U>
{
    type Output = InDm<Chips>;

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
    ) -> InDm<Chips> {
        let second = 1.min(Spans::SIZE - 1);
        InDm {
            rows: U::queries(device, query, from, unit),
            keys: U::tokens::<_, _, Span, 64, { Span::SIZE }>(device, key, unit, 0),
            first: U::tokens::<_, _, Span, 64, { Span::SIZE }>(device, value, unit, 0),
            second: (
                U::tokens::<_, _, Span, 64, { Span::SIZE }>(device, key, unit, second),
                U::tokens::<_, _, Span, 64, { Span::SIZE }>(device, value, unit, second),
            ),
        }
    }
}

/// A unit ready for its first trip, all in data memory so it may cross a loop: its queries, its
/// first span's scores and `Second`, its second span's keys and values; its first values staged.
struct Ready<
    Chips: M,
    Second = (Key<Chips>, Key<Chips>),
    G: M = Groups,
    R: AxisName = Rows,
    H: AxisName = Head,
    const GROUP: usize = { GROUP_ROWS },
    const SLICE: usize = { SLICE_ROWS },
> {
    rows: Query<Chips, G, R, H, GROUP, SLICE>,
    scored: Scores<Chips, G, R, GROUP>,
    second: Second,
}

/// What trip c hands the next: span c's scores, the running maximum and sum before span c, and
/// span c + 1's keys `K` in the register file and its values, `KEYS` a key slice.
struct Trip<Chips: M, K: AxisName = Span, const KEYS: usize = 64> {
    score: Score<Chips, Groups, Head, K, KEYS>,
    scored: Scores<Chips>,
    previous: AgreedTop<Chips>,
    values: Key<Chips, Groups, Head, K, KEYS>,
    softmax: OnlineSoftmax<Peak<Chips>, Product<Chips>>,
}

/// Trip `c`: span c + 1's scores and span c's values weighed into the sum, while `in_dm` holds
/// span c + 2's keys and values and `staged` the buffers span c + 1 transposes into.
impl<
    Spans: AxisName,
    Pairs: AxisName,
    V: Visibility<Spans, Chips>,
    Chips: M,
    const SCALE: usize,
    L: AxisName,
    const LAST: usize,
    const WIDTH: usize,
    K: AxisName,
    const KEYS: usize,
>
    Op<(
        Trip<Chips>,
        (&Query<Chips>, (&V, usize, &mut ChannelMajor<Chips>)),
        (
            Key<Chips, Groups, Head, K, KEYS>,
            Key<Chips, Groups, Head, K, KEYS>,
        ),
    )> for Sweep<'_, Spans, Pairs, V, Chips, SCALE, L, LAST, WIDTH>
{
    type Output = Trip<Chips, K, KEYS>;

    fn forward(
        &self,
        device: &mut Device,
        (trip, (rows, staged), (next_keys, next_values)): (
            Trip<Chips>,
            (&Query<Chips>, (&V, usize, &mut ChannelMajor<Chips>)),
            (
                Key<Chips, Groups, Head, K, KEYS>,
                Key<Chips, Groups, Head, K, KEYS>,
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
        let next = Calculate.forward(device, (score, rows));
        let (weights, softmax) = Weigh.forward(
            device,
            (
                softmax,
                (scored, previous),
                const { f32::from_bits(SCALE as u32) },
            ),
        );
        let vreg = Register.forward(device, &*transposed);
        let mut product = Product::new();
        Multiply.forward(&mut *device, (weights, vreg, &mut product));
        {
            let values = values;
            visible.stage(device, &values, span, transposed);
        }
        let (softmax, previous, _sum) = Absorb.forward(device, (softmax, product));
        Trip {
            score: LoadScore.forward(device, next_keys),
            scored: next,
            previous,
            values: next_values,
            softmax,
        }
    }
}

struct Score<
    Chips: M,
    G: M = Groups,
    H: AxisName = Head,
    K: AxisName = Span,
    const KEYS: usize = 64,
>(KeyRegister<Chips, G, H, K, KEYS>);

struct LoadScore;

impl<Chips: M, G: M, H: AxisName, K: AxisName, const KEYS: usize> Op<Key<Chips, G, H, K, KEYS>>
    for LoadScore
{
    type Output = Score<Chips, G, H, K, KEYS>;

    fn forward(&self, device: &mut Device, key: Key<Chips, G, H, K, KEYS>) -> Self::Output {
        Score(
            device
                .main
                .begin(key.view())
                .fetch::<m![K % 8, K % { KEYS } / 16, H / 16, K % 16 / 8], m![H % 16]>()
                .collect::<m![K % 8, K % { KEYS } / 16, H / 16, K % 16 / 8], m![H % 16]>()
                .to_trf(),
        )
    }
}

struct Calculate;

impl<
    Chips: M,
    G: M,
    R: AxisName,
    H: AxisName,
    const GROUP: usize,
    const SLICE: usize,
    K: AxisName,
    const KEYS: usize,
>
    Op<(
        Score<Chips, G, H, K, KEYS>,
        &Query<Chips, G, R, H, GROUP, SLICE>,
    )> for Calculate
{
    type Output = Scores<Chips, G, R, GROUP, K, KEYS>;

    /// Every row of the group against every key of the slice: each query streams from the slice
    /// that holds it past the group's 32 key slices.
    fn forward(
        &self,
        device: &mut Device,
        (score, query): (
            Score<Chips, G, H, K, KEYS>,
            &Query<Chips, G, R, H, GROUP, SLICE>,
        ),
    ) -> Self::Output {
        type Stream<R, H, K, const GROUP: usize, const SLICE: usize, const KEYS: usize> = m![
            R % { SLICE } / 2,
            K % { KEYS } / 16,
            R % { GROUP } / { SLICE },
            H / 16
        ];
        type Steps<R, H, K, const GROUP: usize, const SLICE: usize, const KEYS: usize> =
            m![{ Stream<R, H, K, GROUP, SLICE, KEYS> }, K % 16 / 8];
        type Sums<R, K, const GROUP: usize, const SLICE: usize, const KEYS: usize> = m![
            R % { SLICE } / 2,
            K % { KEYS } / 16,
            R % { GROUP } / { SLICE },
            K % 16 / 8
        ];
        let mut scores = Scores::<Chips, G, R, GROUP, K, KEYS>::new();
        device
            .main
            .begin(query.view())
            .fetch::<m![R % { SLICE } / 2, K % { KEYS } / 16, H / 16], m![R % 2, H % 16]>()
            .switch::<Slices<G, K, KEYS>, Stream<R, H, K, GROUP, SLICE, KEYS>>(
                SwitchConfig::Broadcast01 {
                    slice1: 32,
                    slice0: 1,
                    time0: H::SIZE / 16,
                },
            )
            .collect::<m![{ Stream<R, H, K, GROUP, SLICE, KEYS> }, R % 2], m![H % 16]>()
            .contract_outer::<Steps<R, H, K, GROUP, SLICE, KEYS>, m![R % 2, H % 16], _, _, _>(
                &score.0,
            )
            .contract_packet::<m![R % 2]>()
            .contract_time::<Sums<R, K, GROUP, SLICE, KEYS>>()
            .contract_lane::<m![{ Sums<R, K, GROUP, SLICE, KEYS> }, R % 2], m![K % 8]>(
                LaneMode::Interleaved,
            )
            // The commit adapter narrows, so the Vector Engine stays free for Sub's passes.
            .commit_trim::<m![K % 8]>()
            .commit_cast::<bf16>()
            .commit_view(scores.view_mut());
        scores
    }
}

/// The running maximum `Max` of the spans so far and the sum `Sum` of their weighted values.
struct OnlineSoftmax<Max, Sum> {
    max: Max,
    sum: Sum,
}

impl<Chips: M, G: M, R: AxisName, C: AxisName, const GROUP: usize, const SLICE: usize>
    OnlineSoftmax<Peak<Chips, G, R, GROUP>, Product<Chips, G, R, C, GROUP, SLICE>>
{
    /// The softmax before any key: the running maximum `START`, also as the operand of the key
    /// slices, and a zero sum.
    fn start(device: &mut Device) -> (Self, AgreedTop<Chips, G, R, GROUP>) {
        let mut max = Peak::<Chips, G, R, GROUP>::new();
        max.view_mut().memset(START, &mut device.sub);
        let previous = Maximum.forward(device, &max);
        let mut sum = Product::<Chips, G, R, C, GROUP, SLICE>::new();
        sum.view_mut().memset(0.0, &mut device.sub);
        (Self { max, sum }, previous)
    }
}

struct Weights<
    Chips: M,
    G: M = Groups,
    R: AxisName = Rows,
    const GROUP: usize = { GROUP_ROWS },
    K: AxisName = Span,
    const KEYS: usize = 64,
> {
    values: Scores<Chips, G, R, GROUP, K, KEYS>,
}

/// Each row's largest score over its group's key slices, times `scale`; `finish` ends the
/// maxima's pass on the Vector Engine, where a running maximum can bound them.
struct Raise;

impl<
    Chips: M,
    G: M,
    R: AxisName,
    const GROUP: usize,
    K: AxisName,
    const KEYS: usize,
    S: stage::Stage,
    Stash: StashState,
    F,
> Op<(&Scores<Chips, G, R, GROUP, K, KEYS>, f32, F)> for Raise
where
    F: for<'l> FnOnce(Raised<'l, Chips, G, R, GROUP>) -> Raised<'l, Chips, G, R, GROUP, S, Stash>,
{
    type Output = Peak<Chips, G, R, GROUP, K, KEYS>;

    fn forward(
        &self,
        device: &mut Device,
        (scores, scale, finish): (&Scores<Chips, G, R, GROUP, K, KEYS>, f32, F),
    ) -> Self::Output {
        let local: Peak<Chips, G, R, GROUP, K, KEYS> = device
            .sub
            .begin(scores.view())
            .fetch::<m![R % { GROUP }, K % { KEYS } / 16], m![K % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![R % { GROUP }, K % { KEYS } / 8], m![K % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R % { GROUP }, K % { KEYS } / 4], m![K % 4]>()
            .vector_intra_slice_reduce::<K, m![R % { GROUP }], m![1 # 4]>(
                IntraSliceReduceOpF32::Max,
            )
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .transpose::<m![R % { GROUP } / 2], m![R % 2 # 8]>()
            .commit_trim::<m![R % 2]>()
            .commit();
        let raised: DmTensor<f32, Chips, Cluster, Agreed<G>, m![R % { GROUP }]> = finish(
            device
                .sub
                .begin(local.view())
                .fetch::<m![R % { GROUP } / 8], m![R % 8]>()
                .collect::<m![R % { GROUP } / 8], m![R % 8]>()
                .vector_init()
                .vector_inter_slice_reduce::<Agreed<G>, m![R % { GROUP } / 8]>(
                    InterSliceReduceOpF32::Max,
                )
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![R % { GROUP } / 4], m![R % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
                .vector_widen_concat::<m![R % { GROUP } / 8], m![R % 8]>(),
        )
        .vector_final()
        .commit_trim::<m![R % 8]>()
        .commit();
        // SAFETY: every key slice of a group received the same maximum.
        unsafe { raised.reshape() }
    }
}

/// The weights e^(scale * score - top) of every row's scores, `top` the row's raised maximum.
struct Exp;

impl<Chips: M, G: M, R: AxisName, const GROUP: usize, K: AxisName, const KEYS: usize>
    Op<(
        &Scores<Chips, G, R, GROUP, K, KEYS>,
        &Top<Chips, G, R, GROUP, K, KEYS>,
        f32,
    )> for Exp
{
    type Output = Scores<Chips, G, R, GROUP, K, KEYS>;

    fn forward(
        &self,
        device: &mut Device,
        (scores, top, scale): (
            &Scores<Chips, G, R, GROUP, K, KEYS>,
            &Top<Chips, G, R, GROUP, K, KEYS>,
            f32,
        ),
    ) -> Self::Output {
        device
            .sub
            .begin(scores.view())
            .fetch::<m![R % { GROUP }, K % { KEYS } / 16], m![K % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![R % { GROUP }, K % { KEYS } / 8], m![K % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R % { GROUP }, K % { KEYS } / 4], m![K % 4]>()
            .vector_fp_ternary_with_mode(FpTernaryOp::FmaF, TernaryArgMode::Mode021, (top, -scale))
            .vector_fp_unary(FpUnaryOp::NegExp)
            .vector_widen_concat::<m![R % { GROUP }, K % { KEYS } / 8], m![K % 8]>()
            .vector_final()
            .cast::<bf16, m![K % 8 # 16]>()
            .commit_trim::<m![K % 8]>()
            .commit()
    }
}

struct Weigh;

impl<Chips: M, G: M, R: AxisName, const GROUP: usize, K: AxisName, const KEYS: usize, Sum>
    Op<(
        OnlineSoftmax<Peak<Chips, G, R, GROUP>, Sum>,
        (
            Scores<Chips, G, R, GROUP, K, KEYS>,
            AgreedTop<Chips, G, R, GROUP>,
        ),
        f32,
    )> for Weigh
{
    type Output = (
        Weights<Chips, G, R, GROUP, K, KEYS>,
        OnlineSoftmax<(Peak<Chips, G, R, GROUP>, Peak<Chips, G, R, GROUP, K, KEYS>), Sum>,
    );

    fn forward(
        &self,
        device: &mut Device,
        (softmax, (scores, previous), scale): (
            OnlineSoftmax<Peak<Chips, G, R, GROUP>, Sum>,
            (
                Scores<Chips, G, R, GROUP, K, KEYS>,
                AgreedTop<Chips, G, R, GROUP>,
            ),
            f32,
        ),
    ) -> Self::Output {
        let top = Raise.forward(
            &mut *device,
            (&scores, scale, |raised: Raised<'_, Chips, G, R, GROUP>| {
                raised.vector_clip(ClipBinaryOpF32::Max, &previous)
            }),
        );
        drop(previous);
        let values = {
            let scores = scores;
            let operand = Operand.forward(&mut *device, &top);
            Exp.forward(device, (&scores, &operand, scale))
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

/// Rescales of the sums onto each row's own slice, `SLICE` rows a slice.
struct Decay<const SLICE: usize = { SLICE_ROWS }>;

impl<Chips: M, G: M, R: AxisName, const GROUP: usize, const SLICE: usize>
    Op<(&Peak<Chips, G, R, GROUP>, &AgreedTop<Chips, G, R, GROUP>)> for Decay<SLICE>
{
    type Output = DecayOperand<Chips, G, R, GROUP, SLICE>;

    /// The factor e^(held - raised) on each row's own slice: every key slice of a group holds the
    /// same factors, so reducing the copies moves each row's factor onto its slice.
    fn forward(
        &self,
        device: &mut Device,
        (held, raised): (&Peak<Chips, G, R, GROUP>, &AgreedTop<Chips, G, R, GROUP>),
    ) -> Self::Output {
        // SAFETY: the held maximum is the same in every key slice of a group; only the axis is
        // named.
        let held = unsafe {
            held.view()
                .reshape::<Chips, Cluster, Agreed<G>, m![R % { GROUP }]>()
        };
        device
            .main
            .begin(held)
            .fetch::<m![R % { GROUP } / 8], m![R % 8]>()
            .collect::<m![R % { GROUP } / 8], m![R % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R % { GROUP } / 4], m![R % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, raised)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![R % { GROUP } / 8], m![R % 8]>()
            .vector_inter_slice_reduce::<QuerySlices<G, R, GROUP, SLICE>, m![R % { SLICE } / 8]>(
                InterSliceReduceOpF32::Max,
            )
            .vector_final()
            .to_vrf(&mut device.sub)
    }
}

struct Maximum;

impl<Chips: M, G: M, R: AxisName, const GROUP: usize, K: AxisName, const KEYS: usize>
    Op<&Peak<Chips, G, R, GROUP, K, KEYS>> for Maximum
{
    type Output = AgreedTop<Chips, G, R, GROUP>;

    /// The running maximum `held` as the operand of the group's key slices.
    fn forward(
        &self,
        device: &mut Device,
        held: &Peak<Chips, G, R, GROUP, K, KEYS>,
    ) -> Self::Output {
        // SAFETY: a group's key slices hold the same running maximum; only the broadcast axis is
        // named.
        unsafe { Operand.forward(device, held).reshape() }
    }
}

struct Operand;

impl<Chips: M, G: M, R: AxisName, const GROUP: usize, K: AxisName, const KEYS: usize>
    Op<&Peak<Chips, G, R, GROUP, K, KEYS>> for Operand
{
    type Output = Top<Chips, G, R, GROUP, K, KEYS>;

    fn forward(
        &self,
        device: &mut Device,
        peak: &Peak<Chips, G, R, GROUP, K, KEYS>,
    ) -> Self::Output {
        device
            .sub
            .begin(peak.view())
            .fetch::<m![R % { GROUP } / 8], m![R % 8]>()
            .collect::<m![R % { GROUP } / 8], m![R % 8]>()
            .to_vrf()
    }
}

struct Values<Chips: M, K: AxisName = Span, const KEYS: usize = 64>(
    Key<Chips, Groups, Head, K, KEYS>,
);

struct Register;

impl<Chips: M, G: M, C: AxisName, K: AxisName, const KEYS: usize>
    Op<&ChannelMajor<Chips, G, C, K, KEYS>> for Register
{
    type Output = ValueRegister<Chips, G, C, K, KEYS>;

    fn forward(
        &self,
        device: &mut Device,
        value: &ChannelMajor<Chips, G, C, K, KEYS>,
    ) -> Self::Output {
        device
            .main
            .begin(value.view())
            .fetch::<m![C % 8, C / 16, K % { KEYS } / 16, C % 16 / 8], m![K % 16]>()
            .collect::<m![C % 8, C / 16, K % { KEYS } / 16, C % 16 / 8], m![K % 16]>()
            .to_trf()
    }
}

/// Values of `HEAD`-wide heads, channel-major.
struct Transpose<const HEAD: usize>;

impl<Chips: M, G: M, H: AxisName, C: AxisName, K: AxisName, const KEYS: usize, const HEAD: usize>
    Op<(
        &Key<Chips, G, H, K, KEYS>,
        &mut ChannelMajor<Chips, G, C, K, KEYS>,
    )> for Transpose<HEAD>
{
    type Output = ();

    /// `value` channel-major into the value channels of `transposed`.
    fn forward(
        &self,
        device: &mut Device,
        (value, transposed): (
            &Key<Chips, G, H, K, KEYS>,
            &mut ChannelMajor<Chips, G, C, K, KEYS>,
        ),
    ) {
        const { assert!(H::SIZE == HEAD, "the head is HEAD channels wide") };
        let channels = transposed
            .view_mut()
            .tile::<m![C], HEAD, m![C = { HEAD } #{!} C, K % { KEYS }]>(0);
        // SAFETY: the value channels are consecutive; only the axis is renamed.
        let channels = unsafe {
            channels.reshape::<Chips, Cluster, Slices<G, K, KEYS>, m![H #{!} C, K % { KEYS }]>()
        };
        device
            .sub
            .begin(value.view())
            .fetch::<m![H / 16, K % { KEYS }], m![H % 16]>()
            .collect::<m![H / 16, K % { KEYS }], m![H % 16]>()
            .transpose::<m![H / 16, K % { KEYS } / 4, H % 16], m![K % 4 # 16]>()
            .commit_trim::<m![K % 4]>()
            .commit_view(channels);
    }
}

/// The first `WIDTH` keys of a span: their values kept where the keys are visible.
struct Keep<const WIDTH: usize>;

impl<Chips: M, Spans: AxisName, K: AxisName, const KEYS: usize, const WIDTH: usize>
    Op<(
        &Key<Chips, Groups, Head, K, KEYS>,
        &Visible<Spans, Chips>,
        usize,
    )> for Keep<WIDTH>
{
    type Output = (
        Key<Chips, Groups, Head, K, KEYS>,
        Seen<Chips, Groups, K, KEYS>,
    );

    /// `value` of span `span` with every absent key's values cleared, and the span's visibility.
    fn forward(
        &self,
        device: &mut Device,
        (value, visible, span): (
            &Key<Chips, Groups, Head, K, KEYS>,
            &Visible<Spans, Chips>,
            usize,
        ),
    ) -> Self::Output {
        const { assert!(K::SIZE == WIDTH && WIDTH == 32 * KEYS && WIDTH <= Span::SIZE) };
        let keys = visible
            .view()
            .tile::<m![Spans], 1, m![Spans = 1 # Spans, Span]>(span)
            .tile::<m![Span], WIDTH, m![Spans = 1 # Spans, Span = { WIDTH } # Span]>(0);
        // SAFETY: the tile is the span's first `WIDTH` keys in order; only the axis is renamed.
        let keys = unsafe { keys.reshape::<Chips, m![Spans = 1 # Spans, K # Span]>() };
        let seen: DmTensor<bf16, Chips, m![2], Copies<K, KEYS>, m![K % { KEYS }]> =
            keys.to_dm(&mut device.tdma);
        // SAFETY: both clusters and every query group hold the same visibility; only broadcast axes
        // are named.
        let seen: Seen<Chips, Groups, K, KEYS> = unsafe { seen.reshape() };
        (Clear.forward(device, (value, &seen)), seen)
    }
}

/// Values with every absent key's values cleared, given the keys' visibility.
struct Clear;

impl<Chips: M, G: M, H: AxisName, K: AxisName, const KEYS: usize>
    Op<(&Key<Chips, G, H, K, KEYS>, &Seen<Chips, G, K, KEYS>)> for Clear
{
    type Output = Key<Chips, G, H, K, KEYS>;

    fn forward(
        &self,
        device: &mut Device,
        (value, seen): (&Key<Chips, G, H, K, KEYS>, &Seen<Chips, G, K, KEYS>),
    ) -> Self::Output {
        let keep: VrfTensor<f32, Chips, Cluster, Slices<G, K, KEYS>, m![K % { KEYS }]> = device
            .sub
            .begin(seen.view())
            .fetch::<m![K % { KEYS } / 16], m![K % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![K % { KEYS } / 8], m![K % 8]>()
            .to_vrf();
        device
            .sub
            .begin(value.view())
            .fetch::<m![K % { KEYS }, H / 16], m![H % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![K % { KEYS }, H / 8], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![K % { KEYS }, H / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &keep)
            .vector_widen_concat::<m![K % { KEYS }, H / 8], m![H % 8]>()
            .vector_final()
            .cast::<bf16, m![H % 8 # 16]>()
            .commit_trim::<m![H % 8]>()
            .commit()
    }
}

/// Kept values of `HEAD`-wide heads channel-major, their mass channels taking the keys' visibility.
struct VisibleValues<const HEAD: usize>;

impl<Chips: M, G: M, H: AxisName, C: AxisName, K: AxisName, const KEYS: usize, const HEAD: usize>
    Op<(
        (&Key<Chips, G, H, K, KEYS>, &Seen<Chips, G, K, KEYS>),
        &mut ChannelMajor<Chips, G, C, K, KEYS>,
    )> for VisibleValues<HEAD>
{
    type Output = ();

    /// Kept values channel-major into `transposed`, whose mass channels take the keys' visibility
    /// in place of ones.
    fn forward(
        &self,
        device: &mut Device,
        ((kept, seen), transposed): (
            (&Key<Chips, G, H, K, KEYS>, &Seen<Chips, G, K, KEYS>),
            &mut ChannelMajor<Chips, G, C, K, KEYS>,
        ),
    ) {
        Transpose::<HEAD>.forward(device, (kept, &mut *transposed));
        let mass = transposed
            .view_mut()
            .tile::<m![C], 16, m![C = 16 #{!} C, K % { KEYS }]>(HEAD);
        // SAFETY: each mass channel holds the same visibility; only the broadcast axis is renamed.
        let mass = unsafe {
            mass.reshape::<Chips, Cluster, Slices<G, K, KEYS>, m![[16] #{!} C, K % { KEYS }]>()
        };
        device
            .sub
            .begin(seen.view())
            .fetch::<m![16, K % { KEYS } / 16], m![K % 16]>()
            .collect::<m![16, K % { KEYS } / 16], m![K % 16]>()
            .commit_trim::<m![K % 16]>()
            .commit_view(mass);
    }
}

struct Multiply;

impl<
    Chips: M,
    G: M,
    R: AxisName,
    C: AxisName,
    const GROUP: usize,
    const SLICE: usize,
    K: AxisName,
    const KEYS: usize,
>
    Op<(
        Weights<Chips, G, R, GROUP, K, KEYS>,
        ValueRegister<Chips, G, C, K, KEYS>,
        &mut Product<Chips, G, R, C, GROUP, SLICE>,
    )> for Multiply
{
    type Output = ();

    /// Every row's weighted values and mass against the slice's keys, summed over the group's key
    /// slices onto the row's own slice of `product`.
    fn forward(
        &self,
        device: &mut Device,
        (weights, value, product): (
            Weights<Chips, G, R, GROUP, K, KEYS>,
            ValueRegister<Chips, G, C, K, KEYS>,
            &mut Product<Chips, G, R, C, GROUP, SLICE>,
        ),
    ) {
        let value = value;
        let weights = weights.values;
        type Fetch<R, C, K, const GROUP: usize, const SLICE: usize, const KEYS: usize> = m![
            R % { SLICE } / 2,
            C / 16,
            R % { GROUP } / { SLICE },
            K % { KEYS } / 16
        ];
        type Steps<R, C, K, const GROUP: usize, const SLICE: usize, const KEYS: usize> =
            m![{ Fetch<R, C, K, GROUP, SLICE, KEYS> }, C % 16 / 8];
        type Sums<R, C, const GROUP: usize, const SLICE: usize> = m![
            R % { SLICE } / 2,
            C / 16,
            R % { GROUP } / { SLICE },
            C % 16 / 8
        ];
        device
        .main
        .begin(weights.view())
        .fetch::<Fetch<R, C, K, GROUP, SLICE, KEYS>, m![R % 2, K % 16]>()
        .collect::<m![{ Fetch<R, C, K, GROUP, SLICE, KEYS> }, R % 2], m![K % 16]>()
        .contract_outer::<Steps<R, C, K, GROUP, SLICE, KEYS>, m![R % 2, K % 16], _, _, _>(&value)
        .contract_packet::<m![R % 2]>()
        .contract_time::<Sums<R, C, GROUP, SLICE>>()
        .contract_lane::<m![{ Sums<R, C, GROUP, SLICE> }, R % 2], m![C % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<
            QuerySlices<G, R, GROUP, SLICE>,
            m![R % { SLICE } / 2, C / 16, C % 16 / 8, R % 2],
        >(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![C % 8]>()
        .commit_view(product.view_mut());
    }
}

/// Every row's weighted values of `HEAD`-wide heads `H` divided by its mass: the attended rows.
struct Average<H, const HEAD: usize>(PhantomData<H>);

impl<
    Chips: M,
    G: M,
    R: AxisName,
    H: AxisName,
    C: AxisName,
    const GROUP: usize,
    const SLICE: usize,
    const HEAD: usize,
> Op<&Product<Chips, G, R, C, GROUP, SLICE>> for Average<H, HEAD>
{
    type Output = Attended<Chips, G, R, H, GROUP, SLICE>;

    fn forward(
        &self,
        device: &mut Device,
        sum: &Product<Chips, G, R, C, GROUP, SLICE>,
    ) -> Self::Output {
        const { assert!(H::SIZE == HEAD, "the head is HEAD channels wide") };
        let ones = sum
            .view()
            .tile::<m![C], 8, m![R % { SLICE }, C = 8 # C]>(HEAD);
        // SAFETY: every ones channel holds the row's mass.
        let mass = unsafe {
            ones.reshape::<
                Chips,
                Cluster,
                QuerySlices<G, R, GROUP, SLICE>,
                m![R % { SLICE }, [1 # 8] # C],
            >()
        };
        let total: VrfTensor<
            f32,
            Chips,
            Cluster,
            QuerySlices<G, R, GROUP, SLICE>,
            m![R % { SLICE }, 1 # 8],
        > = device
            .sub
            .begin(mass)
            .fetch::<m![R % { SLICE }], m![1 # 8]>()
            .collect::<m![R % { SLICE }], m![1 # 8]>()
            .to_vrf();
        let values = sum
            .view()
            .tile::<m![C], HEAD, m![R % { SLICE }, C = { HEAD } # C]>(0);
        // SAFETY: the value channels are consecutive; only the axis is renamed.
        let values = unsafe {
            values.reshape::<
                Chips,
                Cluster,
                QuerySlices<G, R, GROUP, SLICE>,
                m![R % { SLICE }, H # C],
            >()
        };
        device
            .main
            .begin(values)
            .fetch::<m![R % { SLICE }, H / 8], m![H % 8]>()
            .collect::<m![R % { SLICE }, H / 8], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R % { SLICE }, H / 4], m![H % 4]>()
            .vector_fp_div(&total)
            .vector_widen_concat::<m![R % { SLICE }, H / 8], m![H % 8]>()
            .vector_final()
            .cast::<bf16, m![H % 8 # 16]>()
            .commit_trim::<m![H % 8]>()
            .commit()
    }
}

struct Absorb;

impl<Chips: M, G: M, R: AxisName, C: AxisName, const GROUP: usize, const SLICE: usize>
    Op<(
        OnlineSoftmax<
            (Peak<Chips, G, R, GROUP>, Peak<Chips, G, R, GROUP>),
            Product<Chips, G, R, C, GROUP, SLICE>,
        >,
        Product<Chips, G, R, C, GROUP, SLICE>,
    )> for Absorb
{
    type Output = (
        OnlineSoftmax<Peak<Chips, G, R, GROUP>, Product<Chips, G, R, C, GROUP, SLICE>>,
        AgreedTop<Chips, G, R, GROUP>,
        Product<Chips, G, R, C, GROUP, SLICE>,
    );

    fn forward(
        &self,
        device: &mut Device,
        (softmax, product): (
            OnlineSoftmax<
                (Peak<Chips, G, R, GROUP>, Peak<Chips, G, R, GROUP>),
                Product<Chips, G, R, C, GROUP, SLICE>,
            >,
            Product<Chips, G, R, C, GROUP, SLICE>,
        ),
    ) -> Self::Output {
        let OnlineSoftmax {
            max: (held, top),
            sum,
        } = softmax;
        let raised = Maximum.forward(device, &top);
        let factor = Decay::<SLICE>.forward(device, (&held, &raised));
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

impl<Chips: M, G: M, R: AxisName, C: AxisName, const GROUP: usize, const SLICE: usize>
    Op<(
        &Product<Chips, G, R, C, GROUP, SLICE>,
        &Product<Chips, G, R, C, GROUP, SLICE>,
        &DecayOperand<Chips, G, R, GROUP, SLICE>,
    )> for Accumulate
{
    type Output = Product<Chips, G, R, C, GROUP, SLICE>;

    /// `sum` rescaled by `decay` plus `product`.
    fn forward(
        &self,
        device: &mut Device,
        (sum, product, decay): (
            &Product<Chips, G, R, C, GROUP, SLICE>,
            &Product<Chips, G, R, C, GROUP, SLICE>,
            &DecayOperand<Chips, G, R, GROUP, SLICE>,
        ),
    ) -> Self::Output {
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(sum.view(), product.view())
            .fetch::<m![R % { SLICE }, Term, C / 8], m![C % 8]>()
            .collect::<m![R % { SLICE }, Term, C / 8], m![C % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, m![R % { SLICE }, C / 8]>()
            .vector_narrow_split::<m![R % { SLICE }, C / 8, C % 8 / 4], m![C % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), decay, ())
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_widen_concat::<m![R % { SLICE }, C / 8], m![C % 8]>()
            .vector_final()
            .commit_trim::<m![C % 8]>()
            .commit()
    }
}

/// Span visibility in every slice that holds its keys.
type Seen<Chips = Chip, G = Groups, K = Span, const KEYS: usize = 64> =
    DmTensor<bf16, Chips, Cluster, Slices<G, K, KEYS>, m![K % { KEYS }]>;

impl<
    R: AxisName,
    Spans: AxisName,
    Pairs: AxisName,
    Heads: AxisName,
    Chips: M,
    V: Visibility<Spans, Chips>,
    const BITS: usize,
>
    crate::Op<(
        &Queries<R, Pairs, Chips>,
        &Tokens<Spans, Pairs, Chips>,
        &Tokens<Spans, Pairs, Chips>,
        (&V, &Value<Chips>),
    )> for Attention<PhantomData<Heads>, scalar::F32<BITS>>
{
    type Output = Queries<R, Pairs, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (query, key, value, (visible, keys)): (
            &Queries<R, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
            &Tokens<Spans, Pairs, Chips>,
            (&V, &Value<Chips>),
        ),
    ) -> Self::Output {
        const {
            assert!(
                Heads::SIZE == 2 * Pairs::SIZE,
                "a chip attends its heads in pairs"
            )
        };
        let mut output = HbmTensor::new();
        visible.sweep::<R, Pairs, BITS>(
            device,
            (query, key, value),
            keys.extent::<m![Spans, Span]>(),
            &mut output,
        );
        output
    }
}

/// The attention of [`super::single`]'s tiles, whose keys fit one span: each trip is a unit of one
/// span, which the last-span step attends from the softmax's start.
// The compiler opens a profile region wherever consecutive commands change source file, so the
// trips stay in the file of the steps they run.
mod one {
    use core::marker::PhantomData;

    use furiosa_opt_std::prelude::*;

    use super::super::single::{
        Channel, Head, Major, Quad, Rotation, Rows, Tokens, TokensView, Trips, Visible,
    };
    use super::{
        Clear, Cluster, Duo, Following, Last, OnlineSoftmax, Register, Seen, Span, SpanStep,
        VisibleValues,
    };
    use crate::Op;
    use crate::ops::attention::Attention;
    use crate::ops::rotary::Rotary;
    use crate::tensor::scalar::F32;

    /// Four heads by two query groups of 1,024 rows.
    type Groups = m![Quad, Rows / { GROUP }];
    /// Rows of a query group and of each of its 32 key slices.
    const GROUP: usize = SpanStep::<Head>::GROUP;
    const SLICE: usize = SpanStep::<Head>::SLICE;
    /// The span's keys or values, written in both query groups of their head.
    type Copies = m![Quad, 2, Span / 64];

    type Query<Chips> = super::Query<Chips, Groups, Rows, Head, GROUP, SLICE>;
    type Key<Chips> = super::Key<Chips, Groups, Head>;
    /// A slice's values channel-major, each head channel followed by the 16 channels of visibility.
    type ChannelMajor<Chips> = super::ChannelMajor<Chips, Groups, Channel>;
    type Attended<Chips> = super::Attended<Chips, Groups, Rows, Head, GROUP, SLICE>;
    /// A trip, a unit of one span: its queries and its span's keys and values.
    type InDm<Chips> = super::InDm<Chips, (), Groups, Rows, Head, GROUP, SLICE>;
    type Ready<Chips> = super::Ready<Chips, (), Groups, Rows, Head, GROUP, SLICE>;

    /// softmax(scale * QKᵀ)V of every head over the visible keys, `SCALE` the bits of the FP32
    /// scale: each trip of eight heads is a unit of one span, whose last-span step is its sweep.
    impl<Trips: AxisName, Heads: AxisName, Chips: M, const SCALE: usize>
        Op<(
            (
                TokensView<'_, Trips, Chips>,
                TokensView<'_, Trips, Chips>,
                TokensView<'_, Trips, Chips>,
            ),
            &Visible<Chips>,
        )> for Attention<PhantomData<Heads>, F32<SCALE>>
    {
        type Output = Tokens<Trips, Chips>;

        fn forward(
            &self,
            device: &mut Device,
            ((query, key, value), visible): (
                (
                    TokensView<'_, Trips, Chips>,
                    TokensView<'_, Trips, Chips>,
                    TokensView<'_, Trips, Chips>,
                ),
                &Visible<Chips>,
            ),
        ) -> Self::Output {
            let mut output = Tokens::<Trips, Chips>::new();
            let seen: DmTensor<bf16, Chips, m![2], m![4, 2, Span / 64], m![Span % 64]> =
                visible.to_dm(&mut device.tdma);
            let sweep = Sweep {
                // SAFETY: every head and query group holds the same visibility; only broadcast
                // axes are named.
                seen: unsafe { seen.reshape() },
                scale: const { f32::from_bits(SCALE as u32) },
            };
            let queries = |device: &mut Device,
                           tokens: &TokensView<'_, Trips, Chips>,
                           trip: usize|
             -> Query<Chips> {
                tokens
                    .clone()
                    .tile::<m![Trips], 1, m![Trips = 1 # Trips, Duo, Quad, Rows, Head]>(trip)
                    .to_dm(&mut device.tdma)
            };
            let tokens = |device: &mut Device,
                          rows: &TokensView<'_, Trips, Chips>,
                          trip: usize|
             -> Key<Chips> {
                // SAFETY: the span's keys are the tile's rows in order; only the row axis is
                // renamed.
                let in_dm: DmTensor<bf16, Chips, Cluster, Copies, m![Span % 64, Head]> = unsafe {
                    rows.clone()
                        .tile::<m![Trips], 1, m![Trips = 1 # Trips, Duo, Quad, Rows, Head]>(trip)
                        .reshape::<Chips, m![Trips = 1 # Trips, Duo, Quad, Span, Head]>()
                }
                .to_dm(&mut device.tdma);
                // SAFETY: every query group holds the same keys; only the broadcast axis is named.
                unsafe { in_dm.reshape() }
            };
            let to_dm = |device: &mut Device, trip: usize| -> InDm<Chips> {
                super::InDm {
                    rows: queries(device, &query, trip),
                    keys: tokens(device, &key, trip),
                    first: tokens(device, &value, trip),
                    second: (),
                }
            };
            let mut transposed = ChannelMajor::<Chips>::new();
            let (mut ready, first) =
                <InDm<Chips> as Following<Chips, Groups, Head, Channel>>::score(
                    to_dm(device, 0),
                    device,
                );
            sweep.stage(device, &first, &mut transposed);
            drop(first);
            // Each trip scores the next one's span under its own; the last hands on nothing.
            #[unroll]
            for trip in 0..Trips::SIZE - 1 {
                let following = to_dm(device, trip + 1);
                ready = sweep.forward(
                    device,
                    (
                        (ready, following, &mut transposed),
                        |device: &mut Device, attended: &Attended<Chips>| {
                            let heads = output.view_mut().tile::<m![Trips], 1, m![
                                Trips = 1 #{!} Trips, Duo, Quad, Rows, Head
                            ]>(trip);
                            attended.view().to_hbm_view(&mut device.tdma, heads);
                        },
                    ),
                );
            }
            let last = Trips::SIZE - 1;
            sweep.forward(
                device,
                (
                    (ready, (), &mut transposed),
                    |device: &mut Device, attended: &Attended<Chips>| {
                        let heads = output.view_mut().tile::<m![Trips], 1, m![
                            Trips = 1 #{!} Trips, Duo, Quad, Rows, Head
                        ]>(last);
                        attended.view().to_hbm_view(&mut device.tdma, heads);
                    },
                ),
            );
            output
        }
    }

    /// A call's sweep over trips of one span: the span's visibility in every slice that holds its
    /// keys, and the FP32 scale.
    struct Sweep<Chips: M> {
        seen: Seen<Chips, Groups>,
        scale: f32,
    }

    impl<Chips: M> Sweep<Chips> {
        /// `values` with every absent key's cleared, channel-major into `transposed`, whose mass
        /// channels take the keys' visibility.
        fn stage(
            &self,
            device: &mut Device,
            values: &Key<Chips>,
            transposed: &mut ChannelMajor<Chips>,
        ) {
            let kept = Clear.forward(&mut *device, (values, &self.seen));
            VisibleValues::<{ Head::SIZE }>.forward(device, ((&kept, &self.seen), transposed));
        }
    }

    /// A trip of one span from `ready`, its values staged: weighed from the softmax's start, so
    /// its empty sum's decay is a no-op, while the trip it hands on is scored; `store` takes rows.
    impl<Chips: M, N: Following<Chips, Groups, Head, Channel>, S>
        Op<((Ready<Chips>, N, &mut ChannelMajor<Chips>), S)> for Sweep<Chips>
    where
        S: FnOnce(&mut Device, &Attended<Chips>),
    {
        type Output = N::Ready;

        fn forward(
            &self,
            device: &mut Device,
            ((ready, following, transposed), store): (
                (Ready<Chips>, N, &mut ChannelMajor<Chips>),
                S,
            ),
        ) -> Self::Output {
            let super::Ready {
                rows,
                scored,
                second: (),
            } = ready;
            drop(rows);
            let (softmax, previous) = OnlineSoftmax::start(device);
            Last::<Head, { Head::SIZE }>(PhantomData).forward(
                device,
                (
                    (softmax, (scored, previous), self.scale),
                    (
                        |device: &mut Device, transposed: &ChannelMajor<Chips>| {
                            Register.forward(device, transposed)
                        },
                        (),
                    ),
                    (
                        following,
                        |device: &mut Device,
                         values: &Key<Chips>,
                         transposed: &mut ChannelMajor<Chips>| {
                            self.stage(device, values, transposed)
                        },
                        transposed,
                    ),
                    store,
                ),
            )
        }
    }

    /// Bidirectional attention of a head-major 2,048-row tile over its visible keys, eight heads a
    /// trip, each row rotated by its own rotation; an absent key's row gets an output no row reads.
    impl<W: AxisName, Heads: AxisName, Chips: M, const BITS: usize>
        Op<(
            &Major<W, Chips>,
            &Major<W, Chips>,
            &Major<W, Chips>,
            (&Rotation<Chips>, &Visible<Chips>),
        )> for Attention<PhantomData<Heads>, F32<BITS>>
    {
        type Output = Major<W, Chips>;

        fn forward(
            &self,
            device: &mut Device,
            (query, key, value, (rotation, visible)): (
                &Major<W, Chips>,
                &Major<W, Chips>,
                &Major<W, Chips>,
                (&Rotation<Chips>, &Visible<Chips>),
            ),
        ) -> Self::Output {
            const {
                assert!(
                    Heads::SIZE == 8 * Trips::<W>::SIZE,
                    "a chip attends its heads eight a trip"
                )
            };
            let (query, key) = Rotary::<Head, Head, bf16, Chips>::new()
                .forward(&mut *device, ((query, key), rotation));
            // SAFETY: the heads are in order, eight a trip, each its rows in order.
            let output = self.forward(
                device,
                (
                    (
                        unsafe {
                            query
                                .view()
                                .reshape::<Chips, m![{ Symbol<Trips<W>> }, Duo, Quad, Rows, Head]>()
                        },
                        unsafe {
                            key.view()
                                .reshape::<Chips, m![{ Symbol<Trips<W>> }, Duo, Quad, Rows, Head]>()
                        },
                        unsafe {
                            value
                                .view()
                                .reshape::<Chips, m![{ Symbol<Trips<W>> }, Duo, Quad, Rows, Head]>()
                        },
                    ),
                    visible,
                ),
            );
            // SAFETY: the heads are in order, eight a trip, each its rows in order.
            unsafe { output.reshape() }
        }
    }

    /// Bidirectional attention of a head-major tile, its operands lent as a layer's context.
    impl<'c, 'o, W: AxisName, Heads: AxisName, Chips: M, const BITS: usize>
        Op<(
            &Major<W, Chips>,
            &Major<W, Chips>,
            &Major<W, Chips>,
            &'c mut (&'o Rotation<Chips>, &'o Visible<Chips>),
        )> for Attention<PhantomData<Heads>, F32<BITS>>
    {
        type Output = Major<W, Chips>;

        fn forward(
            &self,
            device: &mut Device,
            (query, key, value, context): (
                &Major<W, Chips>,
                &Major<W, Chips>,
                &Major<W, Chips>,
                &'c mut (&'o Rotation<Chips>, &'o Visible<Chips>),
            ),
        ) -> Self::Output {
            let (rotation, visible) = *context;
            self.forward(device, (query, key, value, (rotation, visible)))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Op;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use super::{Duo, Head, Queries, Rows, Span, Tokens, Visible};
    use crate::ops::attention::Attention;
    use crate::ops::normalization::tests::random;
    use crate::tensor::axis::Single;
    use crate::tensor::live::tests::count;
    use crate::tensor::scalar::F32;
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use std::f32::consts::FRAC_1_SQRT_2;
    use std::marker::PhantomData;

    /// A chip's heads, two a pair.
    #[derive(Clone, Debug)]
    struct Heads<Pairs>(PhantomData<Pairs>);

    impl<Pairs: AxisName> AxisName for Heads<Pairs> {
        const NAME: Ident = Ident::new("Heads");
        const SIZE: usize = 2 * Pairs::SIZE;
    }

    axes![Three = 3, Twenty = 20, Seven = 7];

    /// The score multiplier of 128-wide heads, `1 / sqrt(128)`.
    const INVERSE_ROOT: usize = (FRAC_1_SQRT_2 / 8.0).to_bits() as usize;

    #[device(chip = 1)]
    fn attend_heads<Spans: AxisName, Pairs: AxisName>(
        device: &mut Device,
        query: &Queries<Rows, Pairs>,
        key: &Tokens<Spans, Pairs>,
        value: &Tokens<Spans, Pairs>,
        visible: &Visible<Spans>,
        keys: &Value,
        output: &mut Queries<Rows, Pairs>,
    ) {
        // Through the catalog leaf, as a model calls it.
        let attention = Attention {
            heads: PhantomData::<Heads<Pairs>>,
            scale: F32::<INVERSE_ROOT>,
        };
        attention
            .forward(&mut *device, (query, key, value, (visible, keys)))
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
        let _lease = DEVICE.lock().await;
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
        let count = count(live).to_hbm(&mut device.pdma).await.unwrap();
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

    /// Fourteen heads over 37,787 live keys of twenty spans, the last span's 923 live keys within
    /// its first half, which it attends alone.
    #[tokio::test(flavor = "current_thread")]
    async fn attends_a_long_sequence() {
        check::<Twenty, Seven>(37_787, 1365).await;
    }
}
