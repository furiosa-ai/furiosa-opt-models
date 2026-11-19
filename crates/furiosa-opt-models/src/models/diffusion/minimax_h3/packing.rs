//! Checked host geometry for one packed text/audio/video request on four parts.
//!
//! Allocation strides, computed rows, and valid tokens are separate. Local buffers and gathered
//! parts retain the allocation stride; compact projection writes computed parts at the
//! runtime work stride. This descriptor specifies addresses, not a kernel or exchange dispatch.
//! Padding has no semantic position. Consumers must initialize computed padding before reading it.

use crate::ops::index::packed::{RowPlan, SpanPlan};
use crate::tensor::{axis::Rows, live::BlockPrefix};
use furiosa_opt_std::prelude::AxisName;
use std::{io, ops::Range};

pub const DEFAULT_MAX_TEXT: usize = 2048;
pub const PARTS: usize = 4;
pub const TEXT_STRIP: usize = 16;
pub const ROW_BLOCK: usize = 256;
pub const GATHER_HEAD: usize = ROW_BLOCK;
pub const GATHER_BODY: usize = 1024;
/// Compiled maximum-text profile: one short first tile, then ten body tiles.
pub const DEFAULT_GATHER_TILES: usize = 11;
/// Per-part allocation shared by the default planner, gather and engine row axis.
pub const DEFAULT_GATHER_ROWS: usize = GATHER_HEAD + (DEFAULT_GATHER_TILES - 1) * GATHER_BODY;
pub const KEY_SPAN: usize = 2048;
pub const AUDIO_TOKENS: usize = 414;
pub const VIDEO_TOKENS: usize = 37_296;
pub const AUDIO_ROWS: usize = 112;
pub const VIDEO_ROWS: usize = 9_328;

/// One compiled allocation's row limits. A larger text limit changes geometry, not model choice.
/// These host bounds do not establish executable kernel support for every configured limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capacity {
    max_text: usize,
    max_work: Work,
    gather_rows: usize,
    gather_tiles: usize,
    gathered_rows: usize,
    staging_rows: usize,
    max_query_rows: usize,
    key_rows: usize,
}

/// Checked runtime work within one capacity; valid text is counted before any padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    capacity: Capacity,
    text_tokens: usize,
    work: Work,
    query_rows: usize,
    key_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Work {
    text: usize,
    packed: usize,
    computed: usize,
}

/// One source of modality counts and offsets for both forward and reverse semantic lookups.
struct Coordinates {
    valid: usize,
    stride: usize,
    local_start: usize,
    upstream_start: usize,
    key_start: usize,
}

/// Semantic modality, independent of its physical or attention-key ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Text,
    Audio,
    Video,
}

/// One part's layout, derived from its request plan rather than a second inventory of its rows.
#[derive(Clone, Copy, Debug)]
pub struct Part<'a> {
    plan: &'a Plan,
    index: usize,
}

/// A modality's local storage interval with a valid prefix; padding is not a token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    start: usize,
    rows: usize,
    valid: usize,
}

/// One valid token's semantic position and addresses in the two device row orders.
/// `upstream_row` indexes supplied positions in text/audio/video order; `key_row` indexes the
/// video/audio/text prefix. No rotary coordinates or coefficients are synthesized here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    modality: Modality,
    modality_row: usize,
    part: usize,
    local_row: usize,
    upstream_row: usize,
    key_row: usize,
    allocated_row: usize,
    query_row: usize,
}

/// One contiguous valid run from compact joined Q/K/V rows into the packed key prefix.
/// No modality padding or guard rows belong to either interval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyRun {
    source: Range<usize>,
    destination: Range<usize>,
}

impl Capacity {
    /// Derive the smallest allocation holding a 256-row first gather and 1024-row bodies.
    pub fn new(max_text: usize) -> io::Result<Self> {
        Self::checked(max_text, None)
    }

    /// Check a caller's compiled per-part allocation against its maximum text limit and gather
    /// equation. The allocation may exceed the minimum, but may not truncate computed rows.
    #[cfg(test)]
    pub(super) fn with_gather_rows(max_text: usize, gather_rows: usize) -> io::Result<Self> {
        Self::checked(max_text, Some(gather_rows))
    }

    pub fn max_text(&self) -> usize {
        self.max_text
    }

    /// Physical stride of a local allocation and of every capacity-strided gathered part.
    #[cfg(test)]
    fn gather_rows(&self) -> usize {
        self.gather_rows
    }

    #[cfg(test)]
    fn gather_tiles(&self) -> usize {
        self.gather_tiles
    }

    /// Gather staging slots: the first slot also has body capacity, though only its head is used.
    #[cfg(test)]
    fn staging_rows(&self) -> usize {
        self.staging_rows
    }

    /// Allocation envelope used for joined QKV/query tensors. Only a request's computed prefix
    /// is initialized by compact stores; allocation capacity is not an active row count.
    pub fn query_capacity(&self) -> usize {
        self.gathered_rows
    }

    /// Maximum active compact query work, which can be smaller than the allocation envelope.
    /// A numerical implementation still has to validate the chosen attention mapping.
    #[cfg(test)]
    fn max_query_rows(&self) -> usize {
        self.max_query_rows
    }

    /// K/V allocation extent rounded to whole key spans; visibility uses the request's key count.
    pub fn key_rows(&self) -> usize {
        self.key_rows
    }

    pub fn key_spans(&self) -> usize {
        self.key_rows / KEY_SPAN
    }

    /// Reject empty or over-capacity requests before deriving any row addresses.
    pub fn plan(&self, text_tokens: usize) -> io::Result<Plan> {
        if text_tokens == 0 || text_tokens > self.max_text {
            return Err(invalid(format!(
                "text_tokens: {text_tokens} is outside 1..={}",
                self.max_text
            )));
        }
        let work = Work::new(text_tokens)?;
        Ok(Plan {
            capacity: *self,
            text_tokens,
            work,
            query_rows: multiply(PARTS, work.computed, "query work rows")?,
            key_count: add(media_tokens(), text_tokens, "valid key count")?,
        })
    }

    fn checked(max_text: usize, gather_rows: Option<usize>) -> io::Result<Self> {
        if max_text == 0 {
            return Err(invalid("max_text: must be greater than zero"));
        }
        let key_count = add(media_tokens(), max_text, "maximum valid key count")?;
        let max_work = Work::new(max_text)?;
        let bodies = round_up(
            max_work.computed - GATHER_HEAD,
            GATHER_BODY,
            "gather bodies",
        )?;
        let minimum = add(GATHER_HEAD, bodies, "gather allocation rows")?;
        let gather_rows = gather_rows.unwrap_or(minimum);
        if gather_rows < max_work.computed {
            return Err(invalid(format!(
                "gather_rows: {gather_rows} cannot hold {} computed rows per part",
                max_work.computed
            )));
        }
        if !(gather_rows - GATHER_HEAD).is_multiple_of(GATHER_BODY) {
            return Err(invalid(format!(
                "gather_rows: {gather_rows} must be {GATHER_HEAD} + n * {GATHER_BODY}"
            )));
        }
        let gather_tiles = add(1, (gather_rows - GATHER_HEAD) / GATHER_BODY, "gather tiles")?;
        Ok(Self {
            max_text,
            max_work,
            gather_rows,
            gather_tiles,
            gathered_rows: multiply(PARTS, gather_rows, "gathered allocation rows")?,
            staging_rows: multiply(
                multiply(gather_tiles, GATHER_BODY, "gather staging slots")?,
                PARTS,
                "gather staging rows",
            )?,
            max_query_rows: multiply(PARTS, max_work.computed, "maximum compact query rows")?,
            key_rows: round_up(key_count, KEY_SPAN, "key capacity")?,
        })
    }
}

impl Default for Capacity {
    fn default() -> Self {
        Self::checked(DEFAULT_MAX_TEXT, Some(DEFAULT_GATHER_ROWS))
            .expect("the default packing capacity has bounded geometry")
    }
}

impl Plan {
    pub fn capacity(&self) -> Capacity {
        self.capacity
    }

    pub fn text_tokens(&self) -> usize {
        self.text_tokens
    }

    pub fn text_rows_per_part(&self) -> usize {
        self.work.text
    }

    pub fn packed_rows_per_part(&self) -> usize {
        self.work.packed
    }

    pub fn work_rows_per_part(&self) -> usize {
        self.work.computed
    }

    /// Runtime modulation work for one local residual allocation. Refuse an envelope from a
    /// different compiled capacity before it can change the gather stride or allocation tail.
    pub fn span_plan<R: AxisName>(&self) -> io::Result<SpanPlan<R, TEXT_STRIP>> {
        if R::SIZE != self.capacity.gather_rows {
            return Err(invalid(format!(
                "span_plan: compiled allocation {} differs from packing allocation {}",
                R::SIZE,
                self.capacity.gather_rows
            )));
        }
        SpanPlan::try_packed([self.work.text, AUDIO_ROWS, VIDEO_ROWS], self.work.computed)
    }

    /// Projection/FFN work for the same local allocation and interval as [`Self::span_plan`].
    /// Derive both from this request before upload; neither text length nor allocation capacity
    /// is a projection block count. No rounding occurs at this boundary.
    pub fn block_prefix<R: AxisName>(&self) -> io::Result<BlockPrefix<(), R, Rows<ROW_BLOCK>>> {
        let spans = self.span_plan::<R>()?;
        BlockPrefix::try_over(spans.work_rows())
            .ok_or_else(|| invalid("block_prefix: work must consist of whole projection blocks"))
    }

    /// Pack prepared per-chip text and media rows into the same allocation used by the blocks.
    /// Sources must initialize each padded modality span, including text padding after its valid
    /// tokens. Join clears the final work guard; Split extracts the same media spans for the heads.
    pub fn row_plan<R: AxisName, T: AxisName, A: AxisName, V: AxisName>(
        &self,
    ) -> io::Result<RowPlan<R, T, A, V>> {
        self.span_plan::<R>()?;
        let spans = [Modality::Text, Modality::Audio, Modality::Video]
            .map(|modality| self.coordinates(modality));
        RowPlan::try_new(
            std::array::from_fn(|i| 0..spans[i].stride),
            std::array::from_fn(|i| (0..spans[i].stride, spans[i].local_start)),
            self.work.computed,
        )
    }

    /// Joined rows written by compact projection, including finite computed padding.
    pub fn query_rows(&self) -> usize {
        self.query_rows
    }

    /// Visible keys only after K/V are packed in video/audio/text order.
    pub fn key_count(&self) -> usize {
        self.key_count
    }

    /// Valid-only transfers in video/audio/text order. Source rows use the compact projection's
    /// runtime part stride; destination rows form a contiguous prefix without modality padding.
    /// The remaining key allocation must be initialized separately before attention reads it.
    pub fn key_runs(&self) -> impl Iterator<Item = KeyRun> + '_ {
        [Modality::Video, Modality::Audio, Modality::Text]
            .into_iter()
            .flat_map(move |modality| {
                let coordinates = self.coordinates(modality);
                (0..PARTS).filter_map(move |part| {
                    let valid = coordinates
                        .valid
                        .saturating_sub(part * coordinates.stride)
                        .min(coordinates.stride);
                    if valid == 0 {
                        return None;
                    }
                    let source = part * self.work.computed + coordinates.local_start;
                    let destination = coordinates.key_start + part * coordinates.stride;
                    Some(KeyRun {
                        source: source..source + valid,
                        destination: destination..destination + valid,
                    })
                })
            })
    }

    pub fn part(&self, index: usize) -> Option<Part<'_>> {
        (index < PARTS).then_some(Part { plan: self, index })
    }

    /// The token at a local row, or none for modality padding, guard rows, or invalid coordinates.
    pub fn token(&self, part: usize, local_row: usize) -> Option<Token> {
        let part = self.part(part)?;
        for modality in [Modality::Text, Modality::Audio, Modality::Video] {
            let span = part.span(modality);
            if span.valid_rows().contains(&local_row) {
                let row = part.index * span.rows + local_row - span.start;
                return self.locate(modality, row);
            }
        }
        None
    }

    /// Locate a valid semantic token. All address arithmetic is bounded by the checked capacity.
    pub fn locate(&self, modality: Modality, row: usize) -> Option<Token> {
        let coordinates = self.coordinates(modality);
        if row >= coordinates.valid {
            return None;
        }
        let part = row / coordinates.stride;
        let local_row = coordinates.local_start + row % coordinates.stride;
        Some(Token {
            modality,
            modality_row: row,
            part,
            local_row,
            upstream_row: coordinates.upstream_start + row,
            key_row: coordinates.key_start + row,
            allocated_row: part * self.capacity.gather_rows + local_row,
            query_row: part * self.work.computed + local_row,
        })
    }

    fn coordinates(&self, modality: Modality) -> Coordinates {
        match modality {
            Modality::Text => Coordinates {
                valid: self.text_tokens,
                stride: self.work.text,
                local_start: 0,
                upstream_start: 0,
                key_start: media_tokens(),
            },
            Modality::Audio => Coordinates {
                valid: AUDIO_TOKENS,
                stride: AUDIO_ROWS,
                local_start: self.work.text,
                upstream_start: self.text_tokens,
                key_start: VIDEO_TOKENS,
            },
            Modality::Video => Coordinates {
                valid: VIDEO_TOKENS,
                stride: VIDEO_ROWS,
                local_start: self.work.text + AUDIO_ROWS,
                upstream_start: self.text_tokens + AUDIO_TOKENS,
                key_start: 0,
            },
        }
    }
}

impl Work {
    fn new(text_tokens: usize) -> io::Result<Self> {
        let text = multiply(
            text_tokens.div_ceil(PARTS * TEXT_STRIP),
            TEXT_STRIP,
            "working text rows per part",
        )?;
        let packed = add(text, AUDIO_ROWS + VIDEO_ROWS, "packed rows per part")?;
        Ok(Self {
            text,
            packed,
            computed: round_up(packed, ROW_BLOCK, "computed rows per part")?,
        })
    }
}

impl Part<'_> {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn span(&self, modality: Modality) -> Span {
        let coordinates = self.plan.coordinates(modality);
        Span {
            start: coordinates.local_start,
            rows: coordinates.stride,
            valid: coordinates
                .valid
                .saturating_sub(self.index * coordinates.stride)
                .min(coordinates.stride),
        }
    }

    /// Finite row-block padding after the packed spans; it belongs to no modality.
    #[cfg(test)]
    fn guard_rows(&self) -> Range<usize> {
        self.plan.work.packed..self.plan.work.computed
    }

    /// Uncomputed allocation tail; no consumer may read it based on this plan.
    #[cfg(test)]
    fn allocation_tail(&self) -> Range<usize> {
        self.plan.work.computed..self.plan.capacity.gather_rows
    }
}

impl Span {
    pub fn rows(&self) -> Range<usize> {
        self.start..self.start + self.rows
    }

    pub fn valid_rows(&self) -> Range<usize> {
        self.start..self.start + self.valid
    }
}

impl Token {
    #[cfg(test)]
    fn modality(&self) -> Modality {
        self.modality
    }

    pub fn modality_row(&self) -> usize {
        self.modality_row
    }

    pub fn part(&self) -> usize {
        self.part
    }

    pub fn local_row(&self) -> usize {
        self.local_row
    }

    pub fn upstream_row(&self) -> usize {
        self.upstream_row
    }

    #[cfg(test)]
    pub(super) fn key_row(&self) -> usize {
        self.key_row
    }

    /// Row in joined capacity-strided parts. A chip's local residual instead uses `local_row`.
    #[cfg(test)]
    fn allocated_row(&self) -> usize {
        self.allocated_row
    }

    /// Destination row once compact projection stores have established the planned work order.
    pub fn query_row(&self) -> usize {
        self.query_row
    }
}

impl KeyRun {
    /// Rows of the compact joined projection, before key feature/head transposition.
    pub fn source(&self) -> Range<usize> {
        self.source.clone()
    }

    /// Rows of the video/audio/text key prefix, before its unused allocation suffix.
    pub fn destination(&self) -> Range<usize> {
        self.destination.clone()
    }
}

fn media_tokens() -> usize {
    AUDIO_TOKENS + VIDEO_TOKENS
}

fn add(left: usize, right: usize, field: &str) -> io::Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid(format!("{field}: row arithmetic overflows usize")))
}

fn multiply(left: usize, right: usize, field: &str) -> io::Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid(format!("{field}: row arithmetic overflows usize")))
}

fn round_up(rows: usize, alignment: usize, field: &str) -> io::Result<usize> {
    multiply(rows.div_ceil(alignment), alignment, field)
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::axis;
    use std::collections::BTreeSet;

    #[test]
    fn matches_the_77_token_row_order_and_distinguishes_storage_from_work() {
        let capacity = Capacity::default();
        let plan = capacity.plan(77).unwrap();
        assert_eq!(
            (capacity.gather_rows(), capacity.gather_tiles()),
            (10496, 11)
        );
        assert_eq!(
            (
                capacity.query_capacity(),
                capacity.staging_rows(),
                capacity.max_query_rows(),
                capacity.key_rows()
            ),
            (41984, 45056, 39936, 40960)
        );
        assert_eq!(
            (
                plan.text_rows_per_part(),
                plan.packed_rows_per_part(),
                plan.work_rows_per_part(),
                plan.query_rows(),
                plan.key_count()
            ),
            (32, 9472, 9472, 37888, 37787)
        );
        assert_eq!(
            [0, 1, 2, 3].map(|part| plan
                .part(part)
                .unwrap()
                .span(Modality::Text)
                .valid_rows()
                .len()),
            [32, 32, 13, 0]
        );
        assert_eq!(
            [0, 1, 2, 3].map(|part| plan
                .part(part)
                .unwrap()
                .span(Modality::Audio)
                .valid_rows()
                .len()),
            [112, 112, 112, 78]
        );
        assert_eq!(
            [0, 1, 2, 3].map(|part| plan
                .part(part)
                .unwrap()
                .span(Modality::Video)
                .valid_rows()
                .len()),
            [9328, 9328, 9328, 9312]
        );
        // Explicit upstream/media-order anchors, independent of the inverse lookup.
        for (modality, row, part, local, upstream, key, query) in [
            (Modality::Text, 76, 2, 12, 76, 37786, 18956),
            (Modality::Audio, 413, 3, 109, 490, 37709, 28525),
            (Modality::Video, 0, 0, 144, 491, 0, 144),
            (Modality::Video, 37295, 3, 9455, 37786, 37295, 37871),
        ] {
            let token = plan.locate(modality, row).unwrap();
            assert_eq!(
                (
                    token.part(),
                    token.local_row(),
                    token.upstream_row(),
                    token.key_row(),
                    token.query_row()
                ),
                (part, local, upstream, key, query)
            );
            assert_eq!(token.allocated_row(), part * 10496 + local);
        }
    }

    #[test]
    fn request_boundaries_select_work_extents_without_a_bucket_enum() {
        let capacity = Capacity::default();
        for (tokens, text, packed, work, queries, keys) in [
            (1, 16, 9456, 9472, 37888, 37711),
            (63, 16, 9456, 9472, 37888, 37773),
            (64, 16, 9456, 9472, 37888, 37774),
            (65, 32, 9472, 9472, 37888, 37775),
            (127, 32, 9472, 9472, 37888, 37837),
            (128, 32, 9472, 9472, 37888, 37838),
            (129, 48, 9488, 9728, 38912, 37839),
            (1152, 288, 9728, 9728, 38912, 38862),
            (1153, 304, 9744, 9984, 39936, 38863),
            (2048, 512, 9952, 9984, 39936, 39758),
        ] {
            let plan = capacity.plan(tokens).unwrap();
            assert_eq!(
                (
                    plan.text_rows_per_part(),
                    plan.packed_rows_per_part(),
                    plan.work_rows_per_part(),
                    plan.query_rows(),
                    plan.key_count()
                ),
                (text, packed, work, queries, keys)
            );
        }
        let extents: BTreeSet<_> = (1..=2048)
            .map(|tokens| capacity.plan(tokens).unwrap().work_rows_per_part())
            .collect();
        assert_eq!(extents, BTreeSet::from([9472, 9728, 9984]));
    }

    #[test]
    fn all_valid_tokens_have_unique_source_key_and_query_addresses() {
        let capacity = Capacity::default();
        for text in [1, 64, 65, 77, 128, 129, 2048] {
            let plan = capacity.plan(text).unwrap();
            let (mut source, mut keys, mut queries) =
                (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
            for part in 0..4 {
                for row in 0..plan.work_rows_per_part() {
                    if let Some(token) = plan.token(part, row) {
                        assert!(source.insert(token.upstream_row()));
                        assert!(keys.insert(token.key_row()));
                        assert!(queries.insert(token.query_row()));
                        assert_eq!(
                            plan.locate(token.modality(), token.modality_row()),
                            Some(token)
                        );
                        assert!(token.query_row() < plan.query_rows());
                        assert!(token.allocated_row() < capacity.query_capacity());
                    }
                }
            }
            assert!(source.into_iter().eq(0..text + 414 + 37296));
            assert!(keys.into_iter().eq(0..text + 414 + 37296));
            assert_eq!(queries.len(), text + 414 + 37296);
        }
    }

    #[test]
    fn guard_padding_and_capacity_tail_never_acquire_semantic_positions() {
        let plan = Capacity::default().plan(129).unwrap();
        let part = plan.part(3).unwrap();
        assert_eq!(part.guard_rows(), 9488..9728);
        assert_eq!(part.allocation_tail(), 9728..10496);
        assert!(plan.token(3, 0).is_none()); // This part has no text tokens.
        assert!(plan.token(3, 48 + 78).is_none()); // Last-part audio padding.
        for row in 9472..10496 {
            assert!(plan.token(3, row).is_none());
        }
        assert!(plan.part(4).is_none());
        assert!(plan.token(usize::MAX, usize::MAX).is_none());
        assert!(plan.locate(Modality::Text, 129).is_none());
        assert!(plan.locate(Modality::Audio, 414).is_none());
        assert!(plan.locate(Modality::Video, 37296).is_none());
    }

    #[test]
    fn modulation_spans_preserve_packing_boundaries_and_refuse_a_different_allocation() {
        for tokens in [1, 64, 65, 77, 128, 129, 1152, 1153, 2048] {
            let plan = Capacity::default().plan(tokens).unwrap();
            let spans = plan.span_plan::<axis::Rows<10496>>().unwrap();
            let part = plan.part(0).unwrap();
            let mut first = 0;
            for (modality, rows) in [Modality::Text, Modality::Audio, Modality::Video]
                .into_iter()
                .zip(spans.modality_rows())
            {
                assert_eq!(first..first + rows, part.span(modality).rows());
                first += rows;
            }
            assert_eq!(spans.packed_rows(), plan.packed_rows_per_part());
            assert_eq!(spans.work_rows(), plan.work_rows_per_part());
            assert_eq!(spans.guards().start()..spans.work_rows(), part.guard_rows());
            assert_eq!(spans.allocation_tail(), part.allocation_tail());
            for error in [
                plan.span_plan::<axis::Rows<9984>>().unwrap_err(),
                plan.span_plan::<axis::Rows<11520>>().unwrap_err(),
            ] {
                assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
                assert!(error.to_string().contains("compiled allocation"));
            }
        }
        let plan = Capacity::with_gather_rows(513, 11520)
            .unwrap()
            .plan(513)
            .unwrap();
        let spans = plan.span_plan::<axis::Rows<11520>>().unwrap();
        assert_eq!(spans.modality_rows(), [144, 112, 9328]);
        assert_eq!(spans.allocation_tail(), 9728..11520);
    }

    #[test]
    fn projection_blocks_and_modulation_use_the_same_checked_request_work() {
        for tokens in 1..=DEFAULT_MAX_TEXT {
            let plan = Capacity::default().plan(tokens).unwrap();
            let prefix = plan.block_prefix::<axis::Rows<10496>>().unwrap();
            let spans = plan.span_plan::<axis::Rows<10496>>().unwrap();
            assert_eq!(prefix.rows(), spans.work_rows());
            assert_eq!(prefix.blocks() * ROW_BLOCK, plan.work_rows_per_part());
            assert_eq!(prefix.interval().start(), 0);
        }
        let plan = Capacity::default().plan(129).unwrap();
        for result in [
            plan.block_prefix::<axis::Rows<9984>>().map(|_| ()),
            plan.block_prefix::<axis::Rows<11520>>().map(|_| ()),
        ] {
            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidInput);
        }
        let plan = Capacity::with_gather_rows(513, 11520)
            .unwrap()
            .plan(513)
            .unwrap();
        assert_eq!(
            plan.block_prefix::<axis::Rows<11520>>().unwrap().rows(),
            9728
        );
    }

    #[test]
    fn key_runs_exclude_padding_when_valid_keys_exactly_fill_the_allocation() {
        let plan = Capacity::default().plan(77).unwrap();
        let runs: Vec<_> = plan.key_runs().collect();
        assert_eq!(runs.len(), 11);
        assert_eq!(runs[0].source(), 144..9472);
        assert_eq!(runs[0].destination(), 0..9328);
        assert_eq!(runs[3].source(), 28560..37872);
        assert_eq!(runs[3].destination(), 27984..37296);
        assert_eq!(runs[7].source(), 28448..28526);
        assert_eq!(runs[7].destination(), 37632..37710);
        assert_eq!(runs[10].source(), 18944..18957);
        assert_eq!(runs[10].destination(), 37774..37787);

        // With this supported capacity, all valid keys fit exactly. Copying the aligned text
        // work instead would require 38926 rows and overrun the 38912-row destination.
        let capacity = Capacity::new(1202).unwrap();
        assert_eq!(capacity.key_rows(), 38912);
        let plan = capacity.plan(1202).unwrap();
        assert_eq!(plan.key_count(), capacity.key_rows());
        assert_eq!(plan.text_rows_per_part() * PARTS + 37710, 38926);
        let runs: Vec<_> = plan.key_runs().collect();
        assert_eq!(runs.last().unwrap().destination(), 38622..38912);
        assert_eq!(runs.last().unwrap().source(), 29952..30242);

        for tokens in [1, 64, 77, 128, 129, 1153, 1202, 2048] {
            let plan = Capacity::default().plan(tokens).unwrap();
            let mut end = 0;
            for run in plan.key_runs() {
                assert_eq!(run.destination().start, end);
                assert_eq!(run.source().len(), run.destination().len());
                assert!(run.source().end <= plan.query_rows());
                assert!(run.destination().end <= plan.capacity().key_rows());
                end = run.destination().end;
                for (source, key) in run.source().zip(run.destination()) {
                    let token = plan
                        .token(
                            source / plan.work_rows_per_part(),
                            source % plan.work_rows_per_part(),
                        )
                        .expect("key transfers contain only semantic tokens");
                    assert_eq!(token.query_row(), source);
                    assert_eq!(token.key_row(), key);
                }
            }
            assert_eq!(end, plan.key_count());
        }
    }

    #[test]
    fn capacity_is_configurable_and_allocation_geometry_is_checked() {
        let capacity = Capacity::new(513).unwrap();
        assert_eq!(capacity.max_text(), 513);
        assert_eq!(capacity.plan(513).unwrap().text_rows_per_part(), 144);
        assert!(capacity.plan(514).is_err());
        let larger = Capacity::with_gather_rows(513, 11520).unwrap();
        assert_eq!(larger.gather_rows(), 11520);
        assert_eq!(larger.max_query_rows(), capacity.max_query_rows());
        assert_eq!(
            larger.plan(513).unwrap().key_count(),
            capacity.plan(513).unwrap().key_count()
        );
        let beyond_default = Capacity::new(4096).unwrap();
        assert_eq!(beyond_default.plan(4096).unwrap().query_rows(), 41984);
        assert_eq!(beyond_default.key_rows(), 43008);
        for rows in [0, 9472, 9952, 9984, 10495] {
            assert_eq!(
                Capacity::with_gather_rows(2048, rows).unwrap_err().kind(),
                io::ErrorKind::InvalidInput
            );
        }
    }

    #[test]
    fn refuses_invalid_lengths_and_row_arithmetic_overflow() {
        assert_eq!(
            Capacity::new(0).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        for tokens in [0, 2049, usize::MAX] {
            let error = Capacity::default().plan(tokens).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(error.to_string().contains("text_tokens"));
        }
        for max_text in [usize::MAX, usize::MAX - 37710] {
            let error = Capacity::new(max_text).unwrap_err();
            assert!(error.to_string().contains("overflows usize"), "{error}");
        }
        let huge_gather = usize::MAX / GATHER_BODY * GATHER_BODY + GATHER_HEAD;
        let error = Capacity::with_gather_rows(2048, huge_gather).unwrap_err();
        assert!(error.to_string().contains("overflows usize"), "{error}");
    }
}
