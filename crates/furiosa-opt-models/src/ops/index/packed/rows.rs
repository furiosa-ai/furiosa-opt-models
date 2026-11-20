//! Checked transfers from three source allocations into one row prefix.

use super::{Join, Rows, Split};
use crate::Op;
use crate::tensor::{live::Live, scalar::Value};
use core::{fmt, marker::PhantomData};
use furiosa_opt_std::prelude::*;
use std::{io, ops::Range};

/// Transfers from text, audio and video allocations into nonoverlapping, ordered destinations
/// within `R`'s work prefix. Source capacities are types; copy lengths and offsets are rows.
///
/// The constructor checks the caller's declared initialized source intervals. Like [`Live`],
/// this describes initialized storage, rather than inspecting its values. Intentional source
/// padding inside a copied interval is preserved, including its exact BF16 bits. Join initializes
/// every gap inside the work prefix to positive zero; rows past work remain untouched.
///
/// This is transfer geometry only. A model derives it from its request plan; it carries no
/// modulation rows, valid-token counts, attention layout, or model-specific rounding rules.
pub struct RowPlan<R, Text, Audio, Video> {
    source: [usize; 3],
    rows: [usize; 3],
    destination: [usize; 3],
    work: usize,
    shape: PhantomData<(R, Text, Audio, Video)>,
}

impl<R, T, A, V> Copy for RowPlan<R, T, A, V> {}

impl<R, T, A, V> Clone for RowPlan<R, T, A, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: AxisName, T: AxisName, A: AxisName, V: AxisName> fmt::Debug for RowPlan<R, T, A, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowPlan")
            .field("copies", &self.copies())
            .field("work_rows", &self.work)
            .field("capacity", &R::SIZE)
            .finish()
    }
}

impl<R: AxisName, T: AxisName, A: AxisName, V: AxisName> RowPlan<R, T, A, V> {
    /// Check initialized intervals and `(source range, destination start)` pairs, in text,
    /// audio, video order. No rounding is performed. Empty copies, gaps, nonzero source starts,
    /// and exact copy endpoints are supported by the checked descriptor. Device execution
    /// additionally requires transfer layouts supported by the compiler.
    ///
    /// Rejects reversed/out-of-capacity ranges, reads outside initialized sources, overlapping
    /// or out-of-order destinations, overflow, and writes outside the bounded work prefix.
    pub fn try_new(
        initialized: [Range<usize>; 3],
        copies: [(Range<usize>, usize); 3],
        work_rows: usize,
    ) -> io::Result<Self> {
        if work_rows > R::SIZE {
            return Err(invalid(format!(
                "work_rows: {work_rows} exceeds capacity {}",
                R::SIZE
            )));
        }
        let capacities = [T::SIZE, A::SIZE, V::SIZE];
        let mut previous = 0;
        for index in 0..3 {
            let ready = &initialized[index];
            let (from, at) = &copies[index];
            if ready.start > ready.end || ready.end > capacities[index] {
                return Err(invalid(format!(
                    "initialized[{index}]: {ready:?} exceeds source capacity {}",
                    capacities[index]
                )));
            }
            if from.start > from.end || from.start < ready.start || from.end > ready.end {
                return Err(invalid(format!(
                    "copies[{index}]: {from:?} is outside initialized source {ready:?}"
                )));
            }
            let end = at.checked_add(from.end - from.start).ok_or_else(|| {
                invalid(format!(
                    "copies[{index}]: destination endpoint overflows usize"
                ))
            })?;
            if *at < previous || end > work_rows {
                return Err(invalid(format!(
                    "copies[{index}]: destination {at}..{end} overlaps preceding rows or exceeds work {work_rows}"
                )));
            }
            previous = end;
        }
        Ok(Self {
            source: std::array::from_fn(|i| copies[i].0.start),
            rows: std::array::from_fn(|i| copies[i].0.end - copies[i].0.start),
            destination: copies.map(|(_, at)| at),
            work: work_rows,
            shape: PhantomData,
        })
    }

    /// The checked source ranges and their destination starts, in text/audio/video order.
    pub fn copies(&self) -> [(Range<usize>, usize); 3] {
        std::array::from_fn(|i| {
            (
                self.source[i]..self.source[i] + self.rows[i],
                self.destination[i],
            )
        })
    }

    /// All initialized destination rows, including gaps cleared by Join.
    pub fn work(&self) -> Live<(), R> {
        Live::all().subrange(0, self.work)
    }

    /// Allocation rows that Join neither reads nor writes.
    pub fn allocation_tail(&self) -> Range<usize> {
        self.work..R::SIZE
    }

    /// The four gaps surrounding the three copies, including empty gaps.
    pub fn gaps(&self) -> [Live<(), R>; 4] {
        let text_end = self.destination[0] + self.rows[0];
        let audio_end = self.destination[1] + self.rows[1];
        let video_end = self.destination[2] + self.rows[2];
        [
            Live::all().subrange(0, self.destination[0]),
            Live::all().subrange(text_end, self.destination[1] - text_end),
            Live::all().subrange(audio_end, self.destination[2] - audio_end),
            Live::all().subrange(video_end, self.work - video_end),
        ]
    }

    fn counts(&self) -> io::Result<[i32; 10]> {
        let mut result = [0; 10];
        for (index, count) in self
            .source
            .into_iter()
            .chain(self.rows)
            .chain(self.destination)
            .chain([self.work])
            .enumerate()
        {
            result[index] = i32::try_from(count)
                .map_err(|_| invalid(format!("row_counts[{index}]: {count} does not fit I32")))?;
        }
        Ok(result)
    }

    // Transport is immutable and only created from checked plans. Bound even malformed counts
    // to physical source/destination allocations; a valid encoding is preserved exactly.
    fn bounded(source: [usize; 3], rows: [usize; 3], destination: [usize; 3], work: usize) -> Self {
        let work = work.min(R::SIZE);
        let text = bounded_copy(source[0], rows[0], destination[0], T::SIZE, work, 0);
        let audio = bounded_copy(
            source[1],
            rows[1],
            destination[1],
            A::SIZE,
            work,
            text.1 + text.2,
        );
        let video = bounded_copy(
            source[2],
            rows[2],
            destination[2],
            V::SIZE,
            work,
            audio.1 + audio.2,
        );
        Self {
            source: [text.0, audio.0, video.0],
            destination: [text.1, audio.1, video.1],
            rows: [text.2, audio.2, video.2],
            work,
            shape: PhantomData,
        }
    }
}

/// Request-owned transport for one checked transfer plan. Decode once at a device entry and
/// pass that local plan to Join/Split. These counts are rows, not strips or projection blocks.
#[derive(DeviceSend)]
pub struct RowInputs<R, Text, Audio, Video, Chips: M = m![1]> {
    counts: [Value<Chips>; 10],
    shape: PhantomData<(R, Text, Audio, Video)>,
}

impl<R: AxisName, T: AxisName, A: AxisName, V: AxisName, C: M> RowInputs<R, T, A, V, C> {
    /// Validate transport sizes before any transfer. Retain the result until its launches finish.
    pub async fn upload(device: &mut Device, plan: &RowPlan<R, T, A, V>) -> io::Result<Self> {
        let counts = plan.counts()?;
        let mut values = Vec::with_capacity(10);
        for count in counts {
            values.push(
                HostTensor::<i32, m![1]>::from_vec([count])
                    .to_hbm::<C, m![1]>(&mut device.pdma)
                    .await
                    .map_err(io::Error::other)?,
            );
        }
        let counts = values
            .try_into()
            .unwrap_or_else(|_| unreachable!("ten row counts were uploaded"));
        Ok(Self {
            counts,
            shape: PhantomData,
        })
    }

    /// Read each count once, with bounds supplied by the typed allocations. No source tensor is
    /// read here and no work is re-rounded. A local plan can be shared by all transfer consumers.
    pub fn read(&self) -> RowPlan<R, T, A, V> {
        RowPlan::bounded(
            [
                self.counts[0].extent::<m![T]>(),
                self.counts[1].extent::<m![A]>(),
                self.counts[2].extent::<m![V]>(),
            ],
            [
                self.counts[3].extent::<m![T]>(),
                self.counts[4].extent::<m![A]>(),
                self.counts[5].extent::<m![V]>(),
            ],
            [
                self.counts[6].extent::<m![R]>(),
                self.counts[7].extent::<m![R]>(),
                self.counts[8].extent::<m![R]>(),
            ],
            self.counts[9].extent::<m![R]>(),
        )
    }
}

impl<R: AxisName, T: AxisName, A: AxisName, V: AxisName, W: AxisName, C: M>
    Op<(
        &(Rows<V, W, C>, Rows<A, W, C>),
        &Rows<T, W, C>,
        &mut Rows<R, W, C>,
        &RowPlan<R, T, A, V>,
    )> for Join
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        ((video, audio), text, output, plan): (
            &(Rows<V, W, C>, Rows<A, W, C>),
            &Rows<T, W, C>,
            &mut Rows<R, W, C>,
            &RowPlan<R, T, A, V>,
        ),
    ) {
        copy(
            device,
            text,
            output,
            plan.source[0],
            plan.destination[0],
            plan.rows[0],
        );
        copy(
            device,
            audio,
            output,
            plan.source[1],
            plan.destination[1],
            plan.rows[1],
        );
        copy(
            device,
            video,
            output,
            plan.source[2],
            plan.destination[2],
            plan.rows[2],
        );
        let [before_text, before_audio, before_video, after_video] = plan.gaps();
        clear(device, output, before_text.start(), before_text.rows());
        clear(device, output, before_audio.start(), before_audio.rows());
        clear(device, output, before_video.start(), before_video.rows());
        clear(device, output, after_video.start(), after_video.rows());
    }
}

impl<R: AxisName, T: AxisName, A: AxisName, V: AxisName, W: AxisName, C: M>
    Op<(
        &(Rows<V, W, C>, Rows<A, W, C>),
        &Rows<T, W, C>,
        &RowPlan<R, T, A, V>,
    )> for Join
{
    type Output = Live<Rows<R, W, C>, R>;

    fn forward(
        &self,
        device: &mut Device,
        (media, text, plan): (
            &(Rows<V, W, C>, Rows<A, W, C>),
            &Rows<T, W, C>,
            &RowPlan<R, T, A, V>,
        ),
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        self.forward(device, (media, text, &mut output, plan));
        plan.work().with(output)
    }
}

/// PackedEmbedding passes its text operand through unchanged. Carry its request plan alongside
/// the text, so the existing composition can use runtime packing without another weight owner.
impl<R: AxisName, T: AxisName, A: AxisName, V: AxisName, W: AxisName, C: M>
    Op<(
        &(Rows<V, W, C>, Rows<A, W, C>),
        (&Rows<T, W, C>, &RowPlan<R, T, A, V>),
    )> for Join
{
    type Output = Live<Rows<R, W, C>, R>;

    fn forward(
        &self,
        device: &mut Device,
        (media, (text, plan)): (
            &(Rows<V, W, C>, Rows<A, W, C>),
            (&Rows<T, W, C>, &RowPlan<R, T, A, V>),
        ),
    ) -> Self::Output {
        self.forward(device, (media, text, plan))
    }
}

/// Invert the two media transfers. The caller supplies initialized packed work described by
/// the same plan. Uncopied output rows are positive zero, so each returned allocation is fully
/// initialized, including a partial last strip. Text is not extracted.
impl<R: AxisName, T: AxisName, A: AxisName, V: AxisName, W: AxisName, C: M>
    Op<(&Rows<R, W, C>, &RowPlan<R, T, A, V>)> for Split
{
    type Output = (Rows<V, W, C>, Rows<A, W, C>);

    fn forward(
        &self,
        device: &mut Device,
        (packed, plan): (&Rows<R, W, C>, &RowPlan<R, T, A, V>),
    ) -> Self::Output {
        let (mut video, mut audio) = (HbmTensor::new(), HbmTensor::new());
        copy(
            device,
            packed,
            &mut video,
            plan.destination[2],
            plan.source[2],
            plan.rows[2],
        );
        clear(device, &mut video, 0, plan.source[2]);
        let end = plan.source[2] + plan.rows[2];
        clear(device, &mut video, end, V::SIZE - end);
        copy(
            device,
            packed,
            &mut audio,
            plan.destination[1],
            plan.source[1],
            plan.rows[1],
        );
        clear(device, &mut audio, 0, plan.source[1]);
        let end = plan.source[1] + plan.rows[1];
        clear(device, &mut audio, end, A::SIZE - end);
        (video, audio)
    }
}

/// Copy-strip rows have their own nominal axis, independent of caller-provided row axes.
#[derive(Clone, Debug)]
struct Batch<const N: usize>;

impl<const N: usize> AxisName for Batch<N> {
    const NAME: Ident = Ident::new("TransferRows");
    const SIZE: usize = N;
}

fn bounded_copy(
    source: usize,
    rows: usize,
    destination: usize,
    capacity: usize,
    work: usize,
    previous: usize,
) -> (usize, usize, usize) {
    let source = source.min(capacity);
    let destination = destination.min(work).max(previous);
    let rows = rows.min(capacity - source).min(work - destination);
    (source, destination, rows)
}

/// Fixed-size strips avoid dependent const arguments while retaining the span-copy layouts.
/// Long copies issue long strips first; no strip reaches outside the checked interval.
fn copy<S: AxisName, R: AxisName, W: AxisName, C: M>(
    device: &mut Device,
    source: &Rows<S, W, C>,
    output: &mut Rows<R, W, C>,
    from: usize,
    at: usize,
    count: usize,
) {
    if const { S::SIZE >= 8192 && R::SIZE >= 8192 } {
        for index in 0..count / 8192 {
            piece::<S, R, W, C, 8192>(
                device,
                source,
                output,
                from + index * 8192,
                at + index * 8192,
            );
        }
    }
    // Each smaller strip divides the preceding one. Derive offsets directly from count,
    // rather than feeding an accumulated runtime cursor into every later address expression.
    macro_rules! strips {
        ($rows:literal, $preceding:literal) => {
            if const { S::SIZE >= $rows && R::SIZE >= $rows } {
                let first = count / $preceding * $preceding;
                let pieces = count % $preceding / $rows;
                for index in 0..pieces {
                    piece::<S, R, W, C, $rows>(
                        device,
                        source,
                        output,
                        from + first + index * $rows,
                        at + first + index * $rows,
                    );
                }
            }
        };
    }
    strips!(1024, 8192);
    strips!(256, 1024);
    strips!(64, 256);
    strips!(16, 64);
    if const { S::SIZE > 0 && R::SIZE > 0 } {
        let first = count / 16 * 16;
        for index in 0..count % 16 {
            piece::<S, R, W, C, 1>(
                device,
                source,
                output,
                from + first + index,
                at + first + index,
            );
        }
    }
}

fn piece<S: AxisName, R: AxisName, W: AxisName, C: M, const N: usize>(
    device: &mut Device,
    source: &Rows<S, W, C>,
    output: &mut Rows<R, W, C>,
    from: usize,
    at: usize,
) {
    type Strip<const N: usize> = Symbol<Batch<N>>;
    let source = source
        .view()
        .tile::<m![S], N, m![S = ({ Strip<N> }) # S, W]>(from);
    let target = output
        .view_mut()
        .tile::<m![R], N, m![R = ({ Strip<N> }) #{!} R, W]>(at);
    // SAFETY: both checked intervals contain N rows; only their row axes are renamed.
    let source = unsafe { source.reshape::<C, m![{ Strip<N> } # S, W]>() };
    // SAFETY: as above; the destination retains its original allocation stride.
    let target = unsafe { target.reshape::<C, m![{ Strip<N> } #{!} R, W]>() };
    source.to_hbm_view(&mut device.tdma, target);
}

/// Overwrite exactly these rows without reading any old destination value, reusing one zero
/// strip for whole strips and individual rows at exact, non-strip capacities and endpoints.
pub(super) fn clear<R: AxisName, W: AxisName, C: M>(
    device: &mut Device,
    output: &mut Rows<R, W, C>,
    first: usize,
    count: usize,
) {
    if count > 0 {
        if const { W::SIZE.is_multiple_of(128) } {
            // Reuse the normal 16-row zero/store implementation. Its DM layout spans the
            // configured clusters and slices; the small HBM strip permits an arbitrary target
            // capacity or endpoint, including one that is not a multiple of sixteen.
            let mut zero: Rows<Batch<16>, W, C> = HbmTensor::new();
            crate::ops::normalization::adaptive::rows::Narrow::<W>::new()
                .clear(device, &mut zero, 0, 1);
            if const { R::SIZE >= 16 } {
                for strip in 0..count / 16 {
                    piece::<Batch<16>, R, W, C, 16>(device, &zero, output, 0, first + strip * 16);
                }
            }
            for row in 0..count % 16 {
                piece::<Batch<16>, R, W, C, 1>(
                    device,
                    &zero,
                    output,
                    0,
                    first + count / 16 * 16 + row,
                );
            }
        } else {
            // Static span copies permit other widths too. Preserve that contract without
            // imposing the normalization strip's whole-256-byte-burst width requirement.
            type Row<W, C> = DmTensor<bf16, C, m![1 # 2], m![1 # 256], m![{ Symbol<Batch<1>> }, W]>;
            let mut zero: Row<W, C> = DmTensor::new();
            zero.view_mut()
                .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
            for row in 0..count {
                let target = output
                    .view_mut()
                    .tile::<m![R], 1, m![R = 1 #{!} R, W]>(first + row);
                // SAFETY: retain the single row and its allocation padding as a row axis.
                // Flattening that padding into W would change the requested feature view.
                let target = unsafe { target.reshape::<C, m![{ Symbol<Batch<1>> } #{!} R, W]>() };
                zero.view().to_hbm_view(&mut device.tdma, target);
            }
        }
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::axis;

    type Plan = RowPlan<axis::Rows<1024>, axis::Rows<1024>, axis::Rows<64>, axis::Rows<128>>;

    #[test]
    fn refuses_uninitialized_sources_and_invalid_destination_geometry() {
        let base = [(0..16, 0), (0..16, 16), (0..16, 32)];
        for (ready, copies, work, reason) in [
            ([0..16, 0..16, 0..16], base.clone(), 1025, "work_rows"),
            ([0..1025, 0..16, 0..16], base.clone(), 64, "initialized[0]"),
            ([0..16, 0..65, 0..16], base.clone(), 64, "initialized[1]"),
            ([0..16, 0..16, 0..129], base.clone(), 64, "initialized[2]"),
            (
                [1..16, 0..16, 0..16],
                base.clone(),
                64,
                "outside initialized",
            ),
            (
                [0..15, 0..16, 0..16],
                base.clone(),
                64,
                "outside initialized",
            ),
            (
                [0..16, 0..16, 0..16],
                [(Range { start: 1, end: 0 }, 0), (0..16, 16), (0..16, 32)],
                64,
                "outside initialized",
            ),
            (
                [0..16, 0..16, 0..16],
                [(0..16, 0), (0..16, 15), (0..16, 32)],
                64,
                "overlaps",
            ),
            (
                [0..16, 0..16, 0..16],
                [(0..16, 0), (0..16, 32), (0..16, 16)],
                64,
                "overlaps",
            ),
            ([0..16, 0..16, 0..16], base.clone(), 47, "exceeds work"),
            (
                [0..16, 0..16, 0..16],
                [(0..16, usize::MAX), (0..16, 16), (0..16, 32)],
                64,
                "overflows",
            ),
        ] {
            let error = Plan::try_new(ready, copies, work).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(error.to_string().contains(reason), "{error}");
        }
        let reversed = Range { start: 16, end: 0 };
        assert!(Plan::try_new([reversed, 0..16, 0..16], base, 64).is_err());
    }
}
