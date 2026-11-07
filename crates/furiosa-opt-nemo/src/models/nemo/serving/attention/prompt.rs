//! Attention of one prompt chunk over its sequences' whole histories, one key span at a time:
//! every key of a span rests in some slice's tensor register file while the chunk's queries
//! stream past it, so a history is read once a layer rather than once a row.
//!
//! - A chunk is one or more segments in row order, each one sequence's consecutive positions; a
//!   segment attends only its own sequence, each row up to its own position, so sequences share
//!   a launch's tokenwise work while each keeps its own causal attention.
//! - A cluster owns two KV heads. Its 256 slices pair tiles of query rows with key slices:
//!   [`narrow`] attends 256-row blocks over 8,192-key spans and [`wide`] 1,024-row blocks over
//!   2,048-key spans (128 rows a tile, 64 keys a slice); [`small`] attends 512-row blocks over
//!   512-key spans (32 rows, 32 keys), so a short history's work follows its rows and keys. A
//!   segment takes the geometry whose (span, block) pairs cost least.
//! - A wide segment whose last span holds at most 1,024 of its keys attends that span in [`half`]
//!   (wide's query tiles, 32 keys a slice), so it computes no key of the span's second half.
//! - A block attends only the spans its rows reach, so a fresh prompt's first blocks skip the
//!   spans past their own rows.
//! - Softmax peaks are all-reduced over a group's key slices; the value product and the mass are
//!   reduced onto the slices of the query rows, where the product is divided by the mass.
//! - The values carry channels of ones after their own, so one product holds both the value
//!   product and the mass.
//! - A span after the first takes its exponents above the first span's peak, so it only adds its
//!   product to the earlier spans', and the last span divides once.
use super::*;
use narrow::{Reach as NarrowReach, Spans as NarrowSpans};
use small::Spans as SmallSpans;
use wide::{Reach as WideReach, Spans as WideSpans};

axes![Prompt = 4096, Channel = 144, Segments = 16];

/// Every chunk row's attended heads.
pub type Attended = HbmTensor<f32, Chip, m![Prompt, Heads, Groups, Head]>;
/// One count a segment. A loop reading one-position tiles of such a read-only table finds it
/// staged in scratchpad memory once, ahead of the layer loop.
pub type PerSegment = HbmTensor<i32, Chip, m![Segments]>;

/// One step's prompt chunk: the rows `start..` of its segments, each segment one sequence's
/// consecutive positions attending that sequence's positions up to its own in one geometry.
#[derive(DeviceSend)]
pub struct Plan {
    /// The chunk's 256-row blocks, 0 when the step has no chunk.
    pub blocks: Count,
    /// The chunk's first capacity row.
    pub start: Count,
    /// The chunk's segments, 0 when the step has no chunk.
    pub segments: Count,
    /// Each segment's first chunk row.
    pub first: PerSegment,
    /// Each segment's chunk row past its last.
    pub end: PerSegment,
    /// The segments' spans in 256-row blocks over 8,192 keys.
    pub narrow: narrow::Tables,
    /// The segments' spans in 1,024-row blocks over 2,048 keys.
    pub wide: wide::Tables,
    /// Each segment's last wide span attended in the half geometry: 1 when the segment sees at
    /// most that span's first 1,024 keys, else 0.
    pub half: PerSegment,
    /// The segments' spans in 512-row blocks over 512 keys.
    pub small: small::Tables,
}

/// The first segment's spans in each geometry, 0 in all but the one its plan fills.
pub struct FirstSpans {
    pub narrow: usize,
    pub wide: usize,
    pub small: usize,
}

impl FirstSpans {
    /// The first segment's spans of `plan`, read once ahead of the layer loop.
    pub fn of(plan: &Plan) -> Self {
        Self {
            narrow: at(&plan.narrow.spans, 0).extent::<m![NarrowSpans]>(),
            wide: at(&plan.wide.spans, 0).extent::<m![WideSpans]>(),
            small: at(&plan.small.spans, 0).extent::<m![SmallSpans]>(),
        }
    }
}

/// The first segment's first span in the geometry its plan fills, gathered once the layer's cache
/// writes are issued, so the layer's work before the chunk's attention hides the transfer.
pub struct Landing {
    landed: narrow::Landed,
}

/// Segment `segment`'s entry of `table`.
pub fn at(table: &PerSegment, segment: usize) -> Count {
    // SAFETY: one entry of a table the host wrote before the launch, which no command writes.
    unsafe {
        table
            .view()
            .tile::<m![Segments], 1, m![Segments = 1 # Segments]>(segment)
            .alias()
    }
}

/// Gathers the first segment's first span in the geometry whose spans `first` holds, `count` (0 or
/// 1) times.
pub fn land(
    context: &mut Device,
    (pool, entry): (&Pool, usize),
    (plan, first): (&Plan, &FirstSpans),
    count: usize,
) -> Landing {
    Landing {
        landed: narrow::Landed::new(),
    }
    .land(context, (pool, entry), (plan, first), (0, 0, count))
}

/// The leading spans every row of the first segment sees whole, in 256- and 1,024-row blocks:
/// a first span seen whole lies before the chunk's rows, which no layer of the launch writes, so
/// each layer's may land ahead of the layer's cache writes (`Landing::land_ahead`).
pub fn history(plan: &Plan) -> (usize, usize) {
    (
        at(&plan.narrow.full, 0).extent::<m![NarrowSpans]>(),
        at(&plan.wide.full, 0).extent::<m![WideSpans]>(),
    )
}

impl Landing {
    /// The first segment's first span gathered `count` (0 or 1) times, but in 256- or 1,024-row
    /// blocks when `history` holds a span its rows see whole, which `land_ahead` gathers.
    pub fn land(
        self,
        context: &mut Device,
        (pool, entry): (&Pool, usize),
        (plan, first): (&Plan, &FirstSpans),
        (narrow, wide, count): (usize, usize, usize),
    ) -> Self {
        let mut landed = self.landed;
        // A count of extents alone keeps its range; the layer's `count` nests outside it.
        for _ in 0..count {
            for _ in 0..(1 - narrow.min(1)) * first.narrow.min(1) {
                narrow::land(context, (pool, entry), &plan.narrow.pages[0], &mut landed);
            }
        }
        // SAFETY: every geometry's landing buffer holds 32 KiB in each of a cluster's 256 slices,
        // and a chunk lands and reads only the geometry it attends in.
        let mut landed: wide::Landed = unsafe { landed.reshape() };
        for _ in 0..count {
            for _ in 0..(1 - wide.min(1)) * first.wide.min(1) {
                wide::land(context, (pool, entry), &plan.wide.pages[0], &mut landed);
            }
        }
        // SAFETY: as above.
        let mut landed: small::Landed = unsafe { landed.reshape() };
        for _ in 0..count {
            for _ in 0..first.small.min(1) {
                small::land(context, (pool, entry), &plan.small.pages[0], &mut landed);
            }
        }
        // SAFETY: as above.
        Landing {
            landed: unsafe { landed.reshape() },
        }
    }

    /// The first span `land` leaves to it, `history` holding a span seen whole: in 256- or
    /// 1,024-row blocks, `count` (0 or 1) times.
    pub fn land_ahead(
        &mut self,
        context: &mut Device,
        (pool, entry): (&Pool, usize),
        (plan, first): (&Plan, &FirstSpans),
        (narrow, wide, count): (usize, usize, usize),
    ) {
        for _ in 0..count {
            for _ in 0..narrow.min(1) * first.narrow.min(1) {
                narrow::land_shared(
                    context,
                    (pool, entry),
                    &plan.narrow.pages[0],
                    &mut self.landed,
                );
            }
            for _ in 0..wide.min(1) * first.wide.min(1) {
                wide::land_shared(
                    context,
                    (pool, entry),
                    &plan.wide.pages[0],
                    &mut self.landed,
                );
            }
        }
    }
}

/// The chunk's attended rows into `output`, the chunk's `segments` from capacity row `start`, each
/// in the geometry whose spans its plan fills; the first segment's first span is `landing`. The
/// small geometry's segments attend in a loop of their own after the others'.
pub fn attend(
    context: &mut Device,
    (pool, entry): (&Pool, usize),
    (plan, (start, segments), queries): (&Plan, (usize, usize), &qkv::Queries),
    landing: Landing,
    output: &mut Attended,
) -> Landing {
    let mut landed = landing.landed;
    for segment in 0..segments {
        let first = at(&plan.first, segment).extent::<m![Prompt]>();
        let rows = at(&plan.end, segment).extent::<m![Prompt]>() - first;
        let narrow = (
            at(&plan.narrow.spans, segment).extent::<m![NarrowSpans]>(),
            at(&plan.narrow.full, segment).extent::<m![NarrowSpans]>(),
            at(&plan.narrow.ahead, segment).extent::<m![NarrowReach]>(),
            rows.div_ceil(narrow::Queries::SIZE),
        );
        let wide = (
            at(&plan.wide.spans, segment).extent::<m![WideSpans]>(),
            at(&plan.wide.full, segment).extent::<m![WideSpans]>(),
            at(&plan.wide.ahead, segment).extent::<m![WideReach]>(),
            rows.div_ceil(wide::Queries::SIZE),
        );
        // A later segment's first span lands behind the segment before it.
        let later = segment.min(1);
        for _ in 0..later * narrow.0.min(1) {
            let pages = &plan.narrow.pages[segment * narrow::Spans::SIZE];
            narrow::land(context, (pool, entry), pages, &mut landed);
        }
        // SAFETY: as in `land`.
        let mut wide_landed: wide::Landed = unsafe { landed.reshape() };
        for _ in 0..later * wide.0.min(1) {
            let pages = &plan.wide.pages[segment * wide::Spans::SIZE];
            wide::land(context, (pool, entry), pages, &mut wide_landed);
        }
        let last = at(&plan.half, segment).extent::<m![1]>();
        let rows = (segment, start, first);
        // SAFETY: as in `land`.
        let mut sequence = narrow::Sequence::new(unsafe { wide_landed.reshape() });
        sequence.attend_segment::<false>(
            context,
            (pool, entry),
            (&plan.narrow, rows, queries),
            (narrow, 0),
            output,
        );
        // SAFETY: as in `land`.
        let mut sequence = wide::Sequence::new(unsafe { sequence.into_landed().reshape() });
        sequence.attend_segment::<true>(
            context,
            (pool, entry),
            (&plan.wide, rows, queries),
            (wide, last),
            output,
        );
        let mut sequence = half::Sequence::resume(sequence);
        sequence.attend_last(
            context,
            (&plan.wide, (start, first), queries),
            (wide.0, wide.2, wide.3, last),
            output,
        );
        // SAFETY: as in `land`.
        landed = unsafe { sequence.into_landed().reshape() };
    }
    // SAFETY: as in `land`.
    let mut sequence = small::Sequence::new(unsafe { landed.reshape() });
    for segment in 0..segments {
        let first = at(&plan.first, segment).extent::<m![Prompt]>();
        let rows = at(&plan.end, segment).extent::<m![Prompt]>() - first;
        let small = (
            at(&plan.small.spans, segment).extent::<m![SmallSpans]>(),
            at(&plan.small.full, segment).extent::<m![SmallSpans]>(),
            at(&plan.small.ahead, segment).extent::<m![SmallSpans]>(),
            rows.div_ceil(small::Queries::SIZE),
        );
        // A segment's first span lands here unless it is the chunk's only one, whose first span
        // `land` gathered: the loop before may have landed later segments over it.
        for _ in 0..(segment + segments - 1).min(1) * small.0.min(1) {
            let pages = &plan.small.pages[segment * small::Spans::SIZE];
            sequence.land_first(context, (pool, entry), pages);
        }
        sequence.attend_segment::<false>(
            context,
            (pool, entry),
            (&plan.small, (segment, start, first), queries),
            (small, 0),
            output,
        );
    }
    // SAFETY: as in `land`.
    Landing {
        landed: unsafe { sequence.into_landed().reshape() },
    }
}

/// 256-row blocks over 8,192-key spans: two query groups of 128 key slices.
pub mod narrow {
    use super::*;

    axes![
        Keys = 8192,
        Queries = 256,
        Spans = 5,
        Term = 2,
        QueryBlocks = 8,
        Reach = 160
    ];
    type Row = m![Queries % 4];
    type RowPair = m![Queries % 4 / 2];
    type RowGroup = m![Queries % 128 / 4];
    type Block = m![Prompt = 256 # Prompt];
    type BlockMut = m![Prompt = 256 #{!} Prompt];
    type CapacityBlock = m![Capacity = 256 # Capacity];
    type Copies = m![2];
    const TILE_ROWS: usize = 128;
    const SLICE_KEYS: usize = 64;
    const KEY_SLICE_LEVELS: usize = 7;

    include!("prompt/whole.rs");
    include!("prompt/body.rs");
    include!("prompt/check.rs");
}

/// 1,024-row blocks over 2,048-key spans: eight query groups of 32 key slices.
pub mod wide {
    use super::*;

    axes![
        Keys = 2048,
        Queries = 1024,
        Spans = 20,
        Term = 2,
        QueryBlocks = 2,
        Reach = 40
    ];
    type Row = m![Queries % 16];
    type RowPair = m![Queries % 16 / 2];
    type RowGroup = m![Queries % 128 / 16];
    type Block = m![Prompt = 1024 # Prompt];
    type BlockMut = m![Prompt = 1024 #{!} Prompt];
    type CapacityBlock = m![Capacity = 1024 # Capacity];
    type Copies = m![8];
    const TILE_ROWS: usize = 128;
    const SLICE_KEYS: usize = 64;
    const KEY_SLICE_LEVELS: usize = 5;

    include!("prompt/whole.rs");
    include!("prompt/body.rs");
    include!("prompt/check.rs");
}

/// A wide segment's last span when it holds at most 1,024 of the segment's keys: wide's 1,024-row
/// blocks and 128-row tiles over 32 keys a slice, the wide sequence's peaks and product carried
/// over.
pub mod half {
    use super::*;

    axes![
        Keys = 1024,
        Queries = 1024,
        Spans = 2,
        Term = 2,
        QueryBlocks = 2,
        Reach = 2
    ];
    type Row = m![Queries % 16];
    type RowPair = m![Queries % 16 / 2];
    type RowGroup = m![Queries % 128 / 16];
    type Block = m![Prompt = 1024 # Prompt];
    type BlockMut = m![Prompt = 1024 #{!} Prompt];
    type CapacityBlock = m![Capacity = 1024 # Capacity];
    type Copies = m![8];
    const TILE_ROWS: usize = 128;
    const SLICE_KEYS: usize = 32;
    const KEY_SLICE_LEVELS: usize = 5;

    include!("prompt/half.rs");
    include!("prompt/body.rs");
    include!("prompt/check.rs");
}

/// 512-row blocks over 512-key spans, each slice four rows of one KV head against every key of
/// the span: a span lands once a cluster and each row's softmax stays in its slice (reach: 2,048
/// keys).
pub mod small {
    use super::*;

    axes![
        Keys = 512,
        Queries = 512,
        Spans = 4,
        Term = 2,
        QueryBlocks = 4,
        Reach = 4,
        Part = 32,
        Channel = 160
    ];
    type Block = m![Prompt = 512 # Prompt];
    type BlockMut = m![Prompt = 512 #{!} Prompt];
    type CapacityBlock = m![Capacity = 512 # Capacity];

    include!("prompt/rows.rs");
    include!("prompt/check.rs");
}
