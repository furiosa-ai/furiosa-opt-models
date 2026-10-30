use super::*;
use crate::Begin;
use crate::Op;

/// The launch's rows spread over the slices, before their quantization for the pairs: `rows`
/// live rows from tile `base`.
#[must_use]
pub struct Spread<Hidden: AxisName, All: AxisName> {
    spread: spread::Spread<Hidden, All>,
    base: usize,
    rows: usize,
}

/// The pairs' quantized rows and row scales staged in HBM, where each pair's rows are copied from.
#[must_use]
pub struct Lanes<Hidden: AxisName, All: AxisName> {
    staged: spread::Staged<Hidden, All>,
    scales: spread::StagedScales<All>,
}

impl<Hidden: AxisName, All: AxisName> Lanes<Hidden, All> {
    /// The places the pairs begun stage the lanes in.
    pub(super) fn new() -> Self {
        Self {
            staged: HbmTensor::new(),
            scales: HbmTensor::new(),
        }
    }
}

/// The last single tile's rows written, before their quantization: `groups` groups of eight.
#[must_use]
pub struct InDm<Hidden: AxisName> {
    rows: activation::InDm<Hidden, Single>,
    groups: usize,
}

/// The launch's rows spread over the slices, for the pairs of `rows` live rows from tile `base`.
impl<'a, Hidden: AxisName, KvHeads: AxisName, All: AxisName>
    Op<(&'a Normalized<Hidden, All>, Begin<(usize, usize)>)> for Quantized<Hidden, KvHeads, All>
{
    type Output = Spread<Hidden, All>;

    fn forward(
        &self,
        context: &mut Device,
        (normalized, Begin((base, rows))): (&'a Normalized<Hidden, All>, Begin<(usize, usize)>),
    ) -> Self::Output {
        Spread {
            spread: normalized.to_dm(&mut context.tdma),
            base,
            rows,
        }
    }
}

/// The spread rows quantized and staged for the pairs, and the first pair's rows written from them.
impl<Hidden: AxisName, KvHeads: AxisName, All: AxisName> Op<Spread<Hidden, All>>
    for Quantized<Hidden, KvHeads, All>
{
    type Output = (Lanes<Hidden, All>, Period<Hidden, TilePair, KvHeads>);

    fn forward(
        &self,
        context: &mut Device,
        Spread { spread, base, rows }: Spread<Hidden, All>,
    ) -> Self::Output {
        // The quantization counts its blocks of 64 rows from the envelope's first row, and the
        // live rows start at tile `base`.
        let quantized = Quantize.forward(
            &mut *context,
            (&spread, (base * Tile::SIZE + rows + 63) / 64),
        );
        let (staged, scales) = Stage.forward(&mut *context, (&quantized.0, &quantized.1));
        let own = PairRows.forward(&mut *context, (&staged, base / 2));
        let rows_scales = PairScales(PhantomData).forward(&mut *context, (&scales, base / 2));
        (
            Lanes { staged, scales },
            Period::new(own, rows_scales, Tile::SIZE / 8),
        )
    }
}

/// Pair `pair` of the launch's rows and its row scales, written from the staged lanes.
impl<Hidden: AxisName, KvHeads: AxisName, All: AxisName> Op<(&Lanes<Hidden, All>, (usize, usize))>
    for Quantized<Hidden, KvHeads, All>
{
    type Output = Period<Hidden, TilePair, KvHeads>;

    fn forward(
        &self,
        context: &mut Device,
        (lanes, (base, pair)): (&Lanes<Hidden, All>, (usize, usize)),
    ) -> Self::Output {
        // A pair past the envelope's last wraps to its first, which the lanes also hold.
        let at = (base / 2 + pair) % (All::SIZE / <TilePair as Group>::Period::SIZE);
        let own = PairRows.forward(&mut *context, (&lanes.staged, at));
        let scales = PairScales(PhantomData).forward(&mut *context, (&lanes.scales, at));
        Period::new(own, scales, Tile::SIZE / 8)
    }
}

/// The last single tile's rows written, and the row it starts at.
impl<'a, Hidden: AxisName, KvHeads: AxisName, All: AxisName>
    Op<(&'a Normalized<Hidden, All>, Begin<Live<(), All>>)> for Quantized<Hidden, KvHeads, All>
{
    type Output = (InDm<Hidden>, usize);

    fn forward(
        &self,
        context: &mut Device,
        (normalized, Begin(live)): (&'a Normalized<Hidden, All>, Begin<Live<(), All>>),
    ) -> Self::Output {
        let first = 2 * (live.count::<Tile>() / 2);
        let groups = live.rows().div_ceil(8) - first * (Tile::SIZE / 8);
        let start = live.start() + first * Tile::SIZE;
        let rows = ToDm::<Single>(PhantomData)
            .forward(&mut *context, (normalized, (start, Tile::SIZE), groups));
        (InDm { rows, groups }, start)
    }
}

/// The single tile's rows quantized with their row scales.
impl<Hidden: AxisName, KvHeads: AxisName, All: AxisName> Op<InDm<Hidden>>
    for Quantized<Hidden, KvHeads, All>
{
    type Output = Period<Hidden, Single, KvHeads>;

    fn forward(&self, context: &mut Device, InDm { rows, groups }: InDm<Hidden>) -> Self::Output {
        let own = Quantize.forward(&mut *context, (&rows, groups));
        let rows = RowScales::<Hidden, KvHeads>(PhantomData).forward(&mut *context, &own.1);
        Period::new(own.0, rows, groups)
    }
}
