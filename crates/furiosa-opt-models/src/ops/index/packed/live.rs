//! Checked runtime modulation spans within a fixed allocation of rows.

use super::Join;
use crate::tensor::live::Live;
use crate::tensor::scalar::Value;
use core::{fmt, marker::PhantomData};
use furiosa_opt_std::prelude::*;
use std::{io, ops::Range};

/// Three contiguous modulated spans, followed by unmodulated guard rows, within `R` rows.
///
/// The spans follow [`super::Packed`]'s text/audio/video table order. `EDGE` is 16 or 64 rows;
/// every span and the work endpoint must align to it. This is work geometry, including each
/// modality's intentional padding, not a semantic token count or a promise of initialization.
/// Consumers initialize the guard interval separately; it never acquires a modulation row.
pub struct SpanPlan<R, const EDGE: usize> {
    strips: [usize; 3],
    work: usize,
    envelope: PhantomData<R>,
}

impl<R, const EDGE: usize> Copy for SpanPlan<R, EDGE> {}

impl<R, const EDGE: usize> Clone for SpanPlan<R, EDGE> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: AxisName, const EDGE: usize> fmt::Debug for SpanPlan<R, EDGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpanPlan")
            .field("modality_rows", &self.modality_rows())
            .field("work_rows", &self.work_rows())
            .field("capacity", &R::SIZE)
            .field("edge", &EDGE)
            .finish()
    }
}

impl<R: AxisName, const EDGE: usize> SpanPlan<R, EDGE> {
    const CAPACITY_STRIPS: usize = {
        assert!(EDGE == 16 || EDGE == 64);
        assert!(R::SIZE.is_multiple_of(EDGE));
        R::SIZE / EDGE
    };

    /// Check row counts without rounding or extending any modulated span into the guards.
    /// Empty spans and empty work are allowed. Invalid alignment, overflow, truncated work,
    /// or work outside `R` returns [`io::ErrorKind::InvalidInput`].
    pub fn try_packed(modality_rows: [usize; 3], work_rows: usize) -> io::Result<Self> {
        let capacity = const { Self::CAPACITY_STRIPS };
        let mut total = 0usize;
        for (index, rows) in modality_rows.into_iter().enumerate() {
            if !rows.is_multiple_of(EDGE) {
                return Err(invalid(format!(
                    "modality_rows[{index}]: {rows} is not aligned to {EDGE} rows"
                )));
            }
            total = total
                .checked_add(rows)
                .ok_or_else(|| invalid("modality_rows: sum overflows usize"))?;
        }
        if !work_rows.is_multiple_of(EDGE) {
            return Err(invalid(format!(
                "work_rows: {work_rows} is not aligned to {EDGE} rows"
            )));
        }
        if total > work_rows {
            return Err(invalid(format!(
                "work_rows: {work_rows} truncates {total} modulated rows"
            )));
        }
        if work_rows / EDGE > capacity {
            return Err(invalid(format!(
                "work_rows: {work_rows} exceeds capacity {}",
                R::SIZE
            )));
        }
        Ok(Self {
            strips: modality_rows.map(|rows| rows / EDGE),
            work: work_rows / EDGE,
            envelope: PhantomData,
        })
    }

    /// Each span's `(first strip, strip count, table row)`, in packing order. The coordinates
    /// use `EDGE`-row strips, as the shared block/edge partition planner expects.
    pub fn spans(&self) -> [(usize, usize, usize); 3] {
        Join::spans_in_strips(self.strips)
    }

    /// The text/audio/video work lengths in rows, excluding the separate guard interval.
    pub fn modality_rows(&self) -> [usize; 3] {
        self.strips.map(|strips| strips * EDGE)
    }

    /// All modulated rows, excluding guards and the uncomputed allocation tail.
    pub fn packed_rows(&self) -> usize {
        (self.strips[0] + self.strips[1] + self.strips[2]) * EDGE
    }

    /// Computed work, including the unmodulated guards.
    pub fn work_rows(&self) -> usize {
        self.work * EDGE
    }

    /// The prefix consumers may compute and must initialize before it is read.
    pub fn work(&self) -> Live<(), R> {
        Live::all().subrange(0, self.work_rows())
    }

    /// Unmodulated guard rows inside the work prefix. This does not initialize their contents.
    pub fn guards(&self) -> Live<(), R> {
        Live::all().subrange(self.packed_rows(), self.work_rows() - self.packed_rows())
    }

    /// Allocation rows outside this plan's work; a consumer must not infer they are initialized.
    pub fn allocation_tail(&self) -> Range<usize> {
        self.work_rows()..R::SIZE
    }

    /// Encode `[text, audio, video, work]` in **strip counts**, not row counts, for I32 transport.
    /// Fails if a count does not fit I32; host-only geometry need not have that transport limit.
    pub fn strip_counts(&self) -> io::Result<[i32; 4]> {
        let counts = [self.strips[0], self.strips[1], self.strips[2], self.work];
        let mut encoded = [0; 4];
        for (index, count) in counts.into_iter().enumerate() {
            encoded[index] = i32::try_from(count)
                .map_err(|_| invalid(format!("strip_counts[{index}]: {count} does not fit I32")))?;
        }
        Ok(encoded)
    }

    // A device read bounds every count and their cumulative endpoint. Valid host encodings
    // survive unchanged; malformed memory still cannot make spans overlap or escape work.
    fn bounded_strips([text, audio, video, work]: [usize; 4]) -> Self {
        let work = work.min(const { Self::CAPACITY_STRIPS });
        let text = text.min(work);
        let audio = audio.min(work - text);
        let video = video.min(work - text - audio);
        Self {
            strips: [text, audio, video],
            work,
            envelope: PhantomData,
        }
    }
}

/// Request-owned, immutable transport for a checked plan. All four I32 values count `EDGE`-row
/// strips. Keeping tensors private prevents construction from unchecked headers or unit changes.
#[derive(DeviceSend)]
pub(crate) struct SpanInputs<R, Chips: M, const EDGE: usize> {
    counts: [Value<Chips>; 4],
    envelope: PhantomData<R>,
}

impl<R: AxisName, Chips: M, const EDGE: usize> SpanInputs<R, Chips, EDGE> {
    /// Validate the encoding before submitting any transfer. The caller retains these request
    /// inputs until its launches finish; preparation is outside the per-layer numerical leaves.
    pub(crate) async fn upload(device: &mut Device, plan: &SpanPlan<R, EDGE>) -> io::Result<Self> {
        let counts = plan.strip_counts()?;
        let mut upload = async |value| {
            HostTensor::<i32, m![1]>::from_vec([value])
                .to_hbm::<Chips, m![1]>(&mut device.pdma)
                .await
                .map_err(io::Error::other)
        };
        let counts = [
            upload(counts[0]).await?,
            upload(counts[1]).await?,
            upload(counts[2]).await?,
            upload(counts[3]).await?,
        ];
        Ok(Self {
            counts,
            envelope: PhantomData,
        })
    }

    /// Decode once at the device entry, then share the local plan with all span consumers.
    /// Counts are strips, not rows or the projection's larger row-block units.
    pub(crate) fn read(&self) -> SpanPlan<R, EDGE> {
        SpanPlan::bounded_strips([
            self.counts[0].extent::<m![R / { EDGE }]>(),
            self.counts[1].extent::<m![R / { EDGE }]>(),
            self.counts[2].extent::<m![R / { EDGE }]>(),
            self.counts[3].extent::<m![R / { EDGE }]>(),
        ])
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::index::packed::Packed;
    use crate::ops::normalization::adaptive::blocks::Block;
    use crate::tensor::axis::Rows;

    fn coverage<R: AxisName, const EDGE: usize>(plan: SpanPlan<R, EDGE>) {
        let mut actual = vec![None; R::SIZE];
        for regions in Block::<R>::plan(plan.spans(), EDGE) {
            for (region, (first, count, table)) in regions.into_iter().enumerate() {
                let rows = if region == 1 { Block::<R>::SIZE } else { EDGE };
                for (offset, slot) in actual[first * rows..(first + count) * rows]
                    .iter_mut()
                    .enumerate()
                {
                    let row = first * rows + offset;
                    assert!(slot.replace(table).is_none(), "row {row} written twice");
                }
            }
        }
        let [text, audio, video] = plan.modality_rows();
        for (row, &table) in actual.iter().enumerate() {
            let expected = if row < text {
                Some(1)
            } else if row < text + audio {
                Some(5)
            } else if row < text + audio + video {
                Some(0)
            } else {
                None
            };
            assert_eq!(table, expected, "row {row}");
        }
        assert_eq!(plan.packed_rows(), text + audio + video);
        assert_eq!(
            (plan.work().start(), plan.work().rows()),
            (0, plan.work_rows())
        );
        assert_eq!(
            (plan.guards().start(), plan.guards().rows()),
            (plan.packed_rows(), plan.work_rows() - plan.packed_rows())
        );
        assert_eq!(plan.allocation_tail(), plan.work_rows()..R::SIZE);
    }

    #[test]
    fn runtime_spans_cover_modalities_without_covering_guards() {
        type Capacity = Rows<10496>;
        for text in (16usize..=512).step_by(16) {
            let packed = text + 112 + 9328;
            let work = packed.div_ceil(256) * 256;
            let plan = SpanPlan::<Capacity, 16>::try_packed([text, 112, 9328], work).unwrap();
            assert_eq!(
                plan.spans(),
                [
                    (0, text / 16, 1),
                    (text / 16, 7, 5),
                    (text / 16 + 7, 583, 0)
                ]
            );
            coverage(plan);
        }
        coverage(SpanPlan::<Capacity, 16>::try_packed([0, 0, 0], 0).unwrap());
        coverage(SpanPlan::<Capacity, 16>::try_packed([0, 0, 0], 256).unwrap());
        coverage(SpanPlan::<Capacity, 16>::try_packed([16, 0, 16], 256).unwrap());
    }

    fn static_spans<T: AxisName, A: AxisName, V: AxisName, const EDGE: usize>() {
        type Sequence<T, A, V> = Packed<T, A, V>;
        let plan = SpanPlan::<Sequence<T, A, V>, EDGE>::try_packed(
            [T::SIZE, A::SIZE, V::SIZE],
            Sequence::<T, A, V>::SIZE,
        )
        .unwrap();
        assert_eq!(plan.spans(), Join::spans::<T, A, V>());
        coverage(plan);
    }

    #[test]
    fn static_and_runtime_spans_share_table_order_for_both_block_sizes() {
        assert_eq!(Block::<Rows<1536>>::SIZE, 512);
        assert_eq!(Block::<Rows<1280>>::SIZE, 256);
        static_spans::<Rows<128>, Rows<448>, Rows<960>, 64>();
        static_spans::<Rows<128>, Rows<448>, Rows<704>, 64>();
        static_spans::<Rows<32>, Rows<112>, Rows<9328>, 16>();
    }

    #[test]
    fn invalid_span_geometry_is_refused_before_planning() {
        for (rows, work, reason) in [
            ([17, 16, 16], 256, "modality_rows[0]"),
            ([16, 17, 16], 256, "modality_rows[1]"),
            ([16, 16, 17], 256, "modality_rows[2]"),
            ([16, 16, 16], 255, "work_rows"),
            ([16, 16, 16], 32, "truncates"),
            ([16, 16, 16], 272, "exceeds capacity"),
            ([usize::MAX - 15, 16, 0], 256, "overflows"),
        ] {
            let error = SpanPlan::<Rows<256>, 16>::try_packed(rows, work).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(error.to_string().contains(reason), "{error}");
        }
        assert!(SpanPlan::<Rows<512>, 64>::try_packed([16, 64, 64], 512).is_err());
    }

    #[test]
    fn transport_encodes_strips_and_refuses_i32_overflow() {
        let plan = SpanPlan::<Rows<10496>, 16>::try_packed([48, 112, 9328], 9728).unwrap();
        assert_eq!(plan.strip_counts().unwrap(), [3, 7, 583, 608]);
        let wide = SpanPlan::<Rows<1536>, 64>::try_packed([128, 448, 960], 1536).unwrap();
        assert_eq!(wide.strip_counts().unwrap(), [2, 7, 15, 24]);
        type Large = Rows<{ (i32::MAX as usize + 1) * 16 }>;
        for rows in [[Large::SIZE, 0, 0], [0, 0, 0]] {
            let plan = SpanPlan::<Large, 16>::try_packed(rows, Large::SIZE).unwrap();
            let error = plan.strip_counts().unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(error.to_string().contains("does not fit I32"));
        }
    }

    #[test]
    fn bounded_decode_preserves_checked_plans_and_closes_invalid_extents() {
        type Plan = SpanPlan<Rows<10496>, 16>;
        for text in (16usize..=512).step_by(16) {
            let work = (text + 112 + 9328).div_ceil(256) * 256;
            let checked = Plan::try_packed([text, 112, 9328], work).unwrap();
            let decoded = Plan::bounded_strips(checked.strip_counts().unwrap().map(|v| v as usize));
            assert_eq!(decoded.spans(), checked.spans());
            assert_eq!(decoded.work_rows(), checked.work_rows());
        }
        for counts in [
            [usize::MAX; 4],
            [0, usize::MAX, usize::MAX, 1],
            [1, 2, usize::MAX, 2],
            [usize::MAX, 1, 2, 0],
        ] {
            let decoded = Plan::bounded_strips(counts);
            assert!(decoded.packed_rows() <= decoded.work_rows());
            assert!(decoded.work_rows() <= Rows::<10496>::SIZE);
            coverage(decoded);
        }
        let wide = SpanPlan::<Rows<1536>, 64>::bounded_strips([2, 7, 15, 24]);
        assert_eq!(wide.modality_rows(), [128, 448, 960]);
        coverage(wide);
        fn device_send<T: furiosa_opt_std::runtime::DeviceSend>() {}
        device_send::<SpanInputs<Rows<10496>, crate::tensor::parallel::Chip, 16>>();
        device_send::<SpanInputs<Rows<1536>, crate::tensor::parallel::Chip, 64>>();
        device_send::<SpanInputs<Rows<10496>, m![{ Symbol<crate::tensor::parallel::Shard<4>> }], 16>>(
        );
    }
}
