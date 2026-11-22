//! Text, audio and video rows packed into one sequence, and the copies between the sequence and
//! its spans.
//!
//! - The spans are text, audio, video, in that order, each a whole number of 64-row tiles (16-row
//!   ones in a chip's part of a sequence split over chips), so a tile never mixes modalities.
//! - A modulation table holds one row per timestep and modality, row `timestep * 3 + modality`
//!   (video 0, text 1, audio 2). Text and video rows take the video timestep 0, audio rows the
//!   audio timestep 1, so a span's table row is a constant.

use crate::Op;
use crate::load::Load;
use crate::ops::index::sequence::Sum;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

pub use live::SpanPlan;
pub use rows::{RowInputs, RowPlan};

pub(crate) mod live;
mod rows;

axes![Group = 6, Timestep = 2, Modality = 3];

/// Text, audio and video spans, in that order.
pub type Packed<Text, Audio, Video> = Sum<Text, Sum<Audio, Video>>;

impl<Text: AxisName, Audio: AxisName, Video: AxisName> Packed<Text, Audio, Video> {
    /// Rows of the strips the spans come in: 64 when every span holds whole 64-row strips, else 16.
    pub(crate) const STRIP: usize =
        if Text::SIZE % 64 == 0 && Audio::SIZE % 64 == 0 && Video::SIZE % 64 == 0 {
            64
        } else {
            assert!(Text::SIZE % 16 == 0 && Audio::SIZE % 16 == 0 && Video::SIZE % 16 == 0);
            16
        };
}

/// A modality's span: `LIVE` rows of tokens, then pad rows up to `ROWS`, a whole number of 16-row
/// tiles.
#[derive(Clone, Debug, DeviceSend, Load)]
pub struct Span<const LIVE: usize, const ROWS: usize>;

impl<const LIVE: usize, const ROWS: usize> AxisName for Span<LIVE, ROWS> {
    const NAME: Ident = Ident::new("Span");

    const SIZE: usize = {
        assert!(LIVE <= ROWS && ROWS.is_multiple_of(16));
        ROWS
    };
}

/// BF16 rows in HBM, a copy or a part on each chip of `Chips`.
pub type Rows<R, W, Chips = m![1]> = HbmTensor<bf16, Chips, m![R, W]>;

/// One modulation row per timestep and modality.
pub type Table<W, Chips = m![1]> = HbmTensor<bf16, Chips, m![Group, W]>;

/// The rows of `span` as rows `at..` of `packed`.
impl<const LIVE: usize, const ROWS: usize> Span<LIVE, ROWS> {
    pub const fn new() -> Self {
        Self
    }
}

impl<const LIVE: usize, const ROWS: usize, R: AxisName, W: AxisName, Chips: M>
    Op<(
        &Rows<Span<LIVE, ROWS>, W, Chips>,
        (&mut Rows<R, W, Chips>, usize),
    )> for Span<LIVE, ROWS>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (span, (packed, at)): (
            &Rows<Span<LIVE, ROWS>, W, Chips>,
            (&mut Rows<R, W, Chips>, usize),
        ),
    ) {
        type S<const LIVE: usize, const ROWS: usize> = Symbol<Span<LIVE, ROWS>>;
        let rows = packed
            .view_mut()
            .tile::<m![R], ROWS, m![R = ({ S<LIVE, ROWS> }) #{!} R, W]>(at);
        // SAFETY: the tile's rows are the span's in order; only the row axis is renamed.
        let rows = unsafe { rows.reshape::<Chips, m![{ S<LIVE, ROWS> } #{!} R, W]>() };
        span.view().to_hbm_view(&mut device.tdma, rows);
    }
}

impl<const LIVE: usize, const ROWS: usize, R: AxisName, W: AxisName, Chips: M>
    Op<(&Rows<R, W, Chips>, usize)> for Span<LIVE, ROWS>
{
    type Output = Rows<Span<LIVE, ROWS>, W, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (packed, at): (&Rows<R, W, Chips>, usize),
    ) -> Self::Output {
        type S<const LIVE: usize, const ROWS: usize> = Symbol<Span<LIVE, ROWS>>;
        let rows = packed
            .view()
            .tile::<m![R], ROWS, m![R = ({ S<LIVE, ROWS> }) # R, W]>(at);
        // SAFETY: the tile's rows are the span's in order; only the row axis is renamed.
        let rows = unsafe { rows.reshape::<Chips, m![{ S<LIVE, ROWS> } # R, W]>() };
        let mut span = HbmTensor::new();
        rows.to_hbm_view(&mut device.tdma, span.view_mut());
        span
    }
}

/// The span's rows at the start of `packed`.
impl<const LIVE: usize, const ROWS: usize, R: AxisName, W: AxisName, Chips: M>
    Op<&Rows<R, W, Chips>> for Span<LIVE, ROWS>
{
    type Output = Rows<Span<LIVE, ROWS>, W, Chips>;

    fn forward(&self, device: &mut Device, packed: &Rows<R, W, Chips>) -> Self::Output {
        self.forward(device, (packed, 0))
    }
}

/// Video and audio rows with the text rows packed into one sequence: text, audio, then video.
#[derive(DeviceSend, Load)]
pub struct Join;

impl<
    'm,
    't,
    const TEXT_LIVE: usize,
    const TEXT_ROWS: usize,
    const AUDIO_LIVE: usize,
    const AUDIO_ROWS: usize,
    const VIDEO_LIVE: usize,
    const VIDEO_ROWS: usize,
    W: AxisName,
    Chips: M,
>
    Op<(
        &'m (
            Rows<Span<VIDEO_LIVE, VIDEO_ROWS>, W, Chips>,
            Rows<Span<AUDIO_LIVE, AUDIO_ROWS>, W, Chips>,
        ),
        &'t Rows<Span<TEXT_LIVE, TEXT_ROWS>, W, Chips>,
    )> for Join
{
    type Output = Rows<
        Packed<
            Span<TEXT_LIVE, TEXT_ROWS>,
            Span<AUDIO_LIVE, AUDIO_ROWS>,
            Span<VIDEO_LIVE, VIDEO_ROWS>,
        >,
        W,
        Chips,
    >;

    fn forward(
        &self,
        device: &mut Device,
        ((video, audio), text): (
            &'m (
                Rows<Span<VIDEO_LIVE, VIDEO_ROWS>, W, Chips>,
                Rows<Span<AUDIO_LIVE, AUDIO_ROWS>, W, Chips>,
            ),
            &'t Rows<Span<TEXT_LIVE, TEXT_ROWS>, W, Chips>,
        ),
    ) -> Self::Output {
        let mut packed = HbmTensor::new();
        Span::<TEXT_LIVE, TEXT_ROWS>::new().forward(&mut *device, (text, (&mut packed, 0)));
        Span::<AUDIO_LIVE, AUDIO_ROWS>::new()
            .forward(&mut *device, (audio, (&mut packed, TEXT_ROWS)));
        Span::<VIDEO_LIVE, VIDEO_ROWS>::new()
            .forward(&mut *device, (video, (&mut packed, TEXT_ROWS + AUDIO_ROWS)));
        packed
    }
}

/// The video and audio rows of a packed sequence, for the heads.
#[derive(DeviceSend, Load)]
pub struct Split;

impl<
    'x,
    const TEXT_LIVE: usize,
    const TEXT_ROWS: usize,
    const AUDIO_LIVE: usize,
    const AUDIO_ROWS: usize,
    const VIDEO_LIVE: usize,
    const VIDEO_ROWS: usize,
    W: AxisName,
    Chips: M,
>
    Op<
        &'x Rows<
            Packed<
                Span<TEXT_LIVE, TEXT_ROWS>,
                Span<AUDIO_LIVE, AUDIO_ROWS>,
                Span<VIDEO_LIVE, VIDEO_ROWS>,
            >,
            W,
            Chips,
        >,
    > for Split
{
    type Output = (
        Rows<Span<VIDEO_LIVE, VIDEO_ROWS>, W, Chips>,
        Rows<Span<AUDIO_LIVE, AUDIO_ROWS>, W, Chips>,
    );

    fn forward(
        &self,
        device: &mut Device,
        packed: &'x Rows<
            Packed<
                Span<TEXT_LIVE, TEXT_ROWS>,
                Span<AUDIO_LIVE, AUDIO_ROWS>,
                Span<VIDEO_LIVE, VIDEO_ROWS>,
            >,
            W,
            Chips,
        >,
    ) -> Self::Output {
        (
            Span::<VIDEO_LIVE, VIDEO_ROWS>::new()
                .forward(&mut *device, (packed, TEXT_ROWS + AUDIO_ROWS)),
            Span::<AUDIO_LIVE, AUDIO_ROWS>::new().forward(&mut *device, (packed, TEXT_ROWS)),
        )
    }
}

/// A video and an audio span each in rows of its own, `Video` and `Audio` rows long: both are
/// allocated, then the video span is written in the leading rows of the first, then the audio span.
/// Any additional rows are initialized to positive zero before a head reads its full envelope.
#[derive(DeviceSend, Load)]
pub struct Materialize<Video, Audio>(PhantomData<(Video, Audio)>);

impl<Video, Audio> Materialize<Video, Audio> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<
    'm,
    const VIDEO_LIVE: usize,
    const VIDEO_ROWS: usize,
    const AUDIO_LIVE: usize,
    const AUDIO_ROWS: usize,
    Video: AxisName,
    Audio: AxisName,
    W: AxisName,
    Chips: M,
>
    Op<&'m (
        Rows<Span<VIDEO_LIVE, VIDEO_ROWS>, W, Chips>,
        Rows<Span<AUDIO_LIVE, AUDIO_ROWS>, W, Chips>,
    )> for Materialize<Video, Audio>
{
    type Output = (Rows<Video, W, Chips>, Rows<Audio, W, Chips>);

    fn forward(
        &self,
        device: &mut Device,
        (video, audio): &'m (
            Rows<Span<VIDEO_LIVE, VIDEO_ROWS>, W, Chips>,
            Rows<Span<AUDIO_LIVE, AUDIO_ROWS>, W, Chips>,
        ),
    ) -> Self::Output {
        const {
            assert!(VIDEO_ROWS <= Video::SIZE && AUDIO_ROWS <= Audio::SIZE);
        }
        let mut media: Self::Output = (HbmTensor::new(), HbmTensor::new());
        Span::<VIDEO_LIVE, VIDEO_ROWS>::new().forward(&mut *device, (video, (&mut media.0, 0)));
        Span::<AUDIO_LIVE, AUDIO_ROWS>::new().forward(&mut *device, (audio, (&mut media.1, 0)));
        if const { Video::SIZE > VIDEO_ROWS } {
            rows::clear(device, &mut media.0, VIDEO_ROWS, Video::SIZE - VIDEO_ROWS);
        }
        if const { Audio::SIZE > AUDIO_ROWS } {
            rows::clear(device, &mut media.1, AUDIO_ROWS, Audio::SIZE - AUDIO_ROWS);
        }
        media
    }
}

impl Join {
    /// Each span's first strip, its strips and its table row, in packing order, in strips of
    /// the sequence's strip rows.
    pub(crate) const fn spans<T: AxisName, A: AxisName, V: AxisName>() -> [(usize, usize, usize); 3]
    {
        let unit = Packed::<T, A, V>::STRIP;
        Self::spans_in_strips([T::SIZE / unit, A::SIZE / unit, V::SIZE / unit])
    }

    /// The shared table ordering for static and checked runtime spans. Callers establish that
    /// these strip counts sum without overflow and fit their row envelope.
    const fn spans_in_strips([text, audio, video]: [usize; 3]) -> [(usize, usize, usize); 3] {
        [(0, text, 1), (text, audio, 5), (text + audio, video, 0)]
    }
}

/// The embeddings of a step's timesteps out of a time embedding over a whole strip of rows: the
/// strip's leading `Timestep` rows.
#[derive(DeviceSend, Load)]
pub struct Trim;

impl<R: AxisName, W: AxisName, Chips: M> Op<&HbmTensor<f32, Chips, m![R, W]>> for Trim {
    type Output = HbmTensor<f32, Chips, m![Timestep, W]>;

    fn forward(
        &self,
        device: &mut Device,
        embedded: &HbmTensor<f32, Chips, m![R, W]>,
    ) -> Self::Output {
        let mut time = HbmTensor::new();
        let rows = embedded
            .view()
            .tile::<m![R], { Timestep::SIZE }, m![R = (Timestep) # R, W]>(0);
        // SAFETY: the tile is the leading rows in order; only the row axis is renamed.
        let rows = unsafe { rows.reshape::<Chips, m![Timestep # R, W]>() };
        rows.to_hbm_view(&mut device.tdma, time.view_mut());
        time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::normalization::tests::random;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    type TextRows = Span<50, 64>;
    type AudioRows = Span<40, 64>;
    type VideoRows = Span<100, 128>;
    type Joined = Packed<TextRows, AudioRows, VideoRows>;
    axes![Narrow = 256];

    #[device(chip = 1)]
    fn joined(
        device: &mut Device,
        media: &(Rows<VideoRows, Narrow>, Rows<AudioRows, Narrow>),
        text: &Rows<TextRows, Narrow>,
    ) -> (
        Rows<Joined, Narrow>,
        (Rows<VideoRows, Narrow>, Rows<AudioRows, Narrow>),
    ) {
        let packed = Join.forward(&mut *device, (media, text));
        let split = Split.forward(device, &packed);
        (packed, split)
    }

    /// Joined rows hold the text, audio and video spans whole, in that order, and the split gives
    /// the video and audio spans back unchanged.
    #[tokio::test(flavor = "current_thread")]
    async fn joins_and_splits_the_spans() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let width = Narrow::SIZE;
        let text = random(64 * width, 0x2545_f491_4f6c_dd1d);
        let audio = random(64 * width, 0x9e37_79b9_7f4a_7c15);
        let video = random(128 * width, 0x0123_4567_89ab_cdef);
        let media = (
            HostTensor::<bf16, m![VideoRows, Narrow]>::from_vec(video.clone())
                .to_hbm::<m![1], m![VideoRows, Narrow]>(&mut device.pdma)
                .await
                .unwrap(),
            HostTensor::<bf16, m![AudioRows, Narrow]>::from_vec(audio.clone())
                .to_hbm::<m![1], m![AudioRows, Narrow]>(&mut device.pdma)
                .await
                .unwrap(),
        );
        let words = HostTensor::<bf16, m![TextRows, Narrow]>::from_vec(text.clone())
            .to_hbm::<m![1], m![TextRows, Narrow]>(&mut device.pdma)
            .await
            .unwrap();
        let (packed, (video_rows, audio_rows)) =
            launch(joined, (&mut device, &media, &words)).await.unwrap();
        let packed = packed
            .to_host::<m![Joined, Narrow]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        assert_eq!(packed, [text, audio.clone(), video.clone()].concat());
        let video_rows = video_rows
            .to_host::<m![VideoRows, Narrow]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let audio_rows = audio_rows
            .to_host::<m![AudioRows, Narrow]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        assert_eq!((video_rows, audio_rows), (video, audio));
    }
}

#[cfg(test)]
pub(crate) mod row_tests {
    //! Checked row transfers against host addresses, independently for Join and Split.
    //!
    //! No checkpoint data is used. Sources outside the declared initialized intervals are NaN;
    //! copies preserve all BF16 bits, gaps become positive zero, and retained allocation tails keep
    //! their previous values. Head outputs initialize their aligned extensions.

    use crate::Op;
    use crate::ops::index::packed::{Join, Materialize, RowInputs, RowPlan, Rows, Span, Split};
    use crate::tensor::{axis, parallel::Shard};
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;
    use std::ops::Range;

    type Chips = m![{ Symbol<Shard<4>> }];
    axes![Width = 128, HeadWidth = 5376, NarrowHead = 64];
    type Media<A, V, W = Width> = (Rows<V, W, Chips>, Rows<A, W, Chips>);
    type Header<R, T, A, V> = RowInputs<R, T, A, V, Chips>;

    type Video = Span<50, 64>;
    type Audio = Span<10, 16>;
    type VideoHead = axis::Rows<80>;
    type AudioHead = axis::Rows<32>;

    #[derive(Clone, Debug)]
    pub(crate) struct Case {
        pub(crate) initialized: [Range<usize>; 3],
        pub(crate) copies: [(Range<usize>, usize); 3],
        pub(crate) work: usize,
    }

    #[device(chip = 4)]
    fn join<R: AxisName, T: AxisName, A: AxisName, V: AxisName>(
        device: &mut Device,
        media: &Media<A, V>,
        text: &Rows<T, Width, Chips>,
        header: &Header<R, T, A, V>,
        output: &mut Rows<R, Width, Chips>,
    ) {
        let plan = header.read();
        Join.forward(device, (media, text, output, &plan));
    }

    #[device(chip = 4)]
    fn join_owned<R: AxisName, T: AxisName, A: AxisName, V: AxisName>(
        device: &mut Device,
        media: &Media<A, V>,
        text: &Rows<T, Width, Chips>,
        header: &Header<R, T, A, V>,
    ) -> Rows<R, Width, Chips> {
        let plan = header.read();
        Join.forward(device, (media, (text, &plan))).into_value()
    }

    #[device(chip = 4)]
    fn split<R: AxisName, T: AxisName, A: AxisName, V: AxisName>(
        device: &mut Device,
        input: &Rows<R, Width, Chips>,
        header: &Header<R, T, A, V>,
    ) -> Media<A, V> {
        let plan = header.read();
        Split.forward(device, (input, &plan))
    }

    fn source(chip: usize, modality: usize, row: usize, column: usize) -> bf16 {
        // Some copied values are signed zero or NaN payloads: this boundary copies, not computes.
        match (row * 17 + column * 7 + chip * 3 + modality) % 257 {
            0 => bf16::from_bits(0x8000),
            1 => bf16::from_bits(0x7fc5),
            _ => bf16::from_f32(
                ((chip * 73 + modality * 53 + row * 11 + column * 7) % 251) as f32 / 16. - 8.,
            ),
        }
    }

    fn independent_packed(chip: usize, row: usize, column: usize) -> bf16 {
        bf16::from_f32(((chip * 101 + row * 23 + column * 13) % 509) as f32 / 32. - 7.)
    }

    fn untouched(chip: usize, row: usize, column: usize) -> bf16 {
        if (row + column + chip).is_multiple_of(7) {
            bf16::from_bits(0x7fc3)
        } else {
            bf16::from_f32(-32. - ((row + column + chip) % 32) as f32)
        }
    }

    async fn upload<R: AxisName, W: AxisName>(
        device: &mut Device,
        values: impl IntoIterator<Item = bf16>,
    ) -> Rows<R, W, Chips> {
        HostTensor::<bf16, m![{ Chips }, R, W]>::from_vec(values)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap()
    }

    async fn download<R: AxisName, W: AxisName>(
        device: &mut Device,
        values: &Rows<R, W, Chips>,
    ) -> Vec<bf16> {
        values
            .to_host::<m![{ Chips }, R, W]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }

    fn assert_bits(actual: &[bf16], expected: &[bf16], label: &str, case: usize) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{label}, case {case}, cell {index}"
            );
        }
    }

    pub(crate) async fn cases<R: AxisName, T: AxisName, A: AxisName, V: AxisName>(
        device: &mut Device,
        cases: &[(Case, RowPlan<R, T, A, V>)],
    ) {
        let mut expected: Vec<_> = (0..4)
            .flat_map(|chip| {
                (0..R::SIZE).flat_map(move |row| {
                    (0..Width::SIZE).map(move |column| untouched(chip, row, column))
                })
            })
            .collect();
        // Reuse the same allocation while work grows and shrinks, without resetting its tail.
        let mut output = upload::<R, Width>(device, expected.clone()).await;
        for (case_index, (case, plan)) in cases.iter().enumerate() {
            let header = Header::upload(device, plan).await.unwrap();
            let values = |modality: usize, capacity: usize| {
                (0..4).flat_map(move |chip| {
                    (0..capacity).flat_map(move |row| {
                        (0..Width::SIZE).map(move |column| {
                            if case.initialized[modality].contains(&row) {
                                source(chip, modality, row, column)
                            } else {
                                bf16::from_bits(0x7fc1)
                            }
                        })
                    })
                })
            };
            let text = upload::<T, Width>(device, values(0, T::SIZE)).await;
            let media = (
                upload::<V, Width>(device, values(2, V::SIZE)).await,
                upload::<A, Width>(device, values(1, A::SIZE)).await,
            );
            launch(join, (&mut *device, &media, &text, &header, &mut output))
                .await
                .unwrap();
            // Host address oracle: initialize the work prefix, then place each source cell at its
            // destination. This does not use Join, Split, plan accessors or production strip logic.
            for chip in 0..4 {
                let start = chip * R::SIZE * Width::SIZE;
                expected[start..start + case.work * Width::SIZE].fill(bf16::from_bits(0));
                for (modality, (range, at)) in case.copies.iter().enumerate() {
                    for (offset, row) in range.clone().enumerate() {
                        for column in 0..Width::SIZE {
                            expected[start + (at + offset) * Width::SIZE + column] =
                                source(chip, modality, row, column);
                        }
                    }
                }
            }
            assert_bits(
                &download(device, &output).await,
                &expected,
                "Join",
                case_index,
            );
            if case_index == 1 {
                let owned = launch(join_owned, (&mut *device, &media, &text, &header))
                    .await
                    .unwrap();
                // An owned Join deliberately leaves its tail uninitialized: read only work on each
                // chip, keeping the allocation-capacity stride rather than globally truncating.
                let owned = owned.into_buffer();
                for chip in 0..4 {
                    let buffer = owned
                        .slice(chip..chip + 1, ..case.work * Width::SIZE * 2)
                        .unwrap();
                    let bytes = buffer.to_host(&mut device.pdma).await.unwrap();
                    let actual: Vec<_> = bytes
                        .chunks_exact(2)
                        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                        .collect();
                    let at = chip * R::SIZE * Width::SIZE;
                    assert_bits(
                        &actual,
                        &expected[at..at + case.work * Width::SIZE],
                        "owned Join",
                        case_index,
                    );
                }
            }
            // Split receives an independent host tensor, so an inverse Join bug cannot cancel a
            // Split bug. Gaps and the uncomputed tail are NaN and must never appear in its outputs.
            let packed = upload::<R, Width>(
                device,
                (0..4).flat_map(|chip| {
                    (0..R::SIZE).flat_map(move |row| {
                        (0..Width::SIZE).map(move |column| {
                            if case
                                .copies
                                .iter()
                                .any(|(range, at)| (*at..at + range.len()).contains(&row))
                            {
                                independent_packed(chip, row, column)
                            } else {
                                bf16::from_bits(0x7fc7)
                            }
                        })
                    })
                }),
            )
            .await;
            let extracted = launch(split, (&mut *device, &packed, &header))
                .await
                .unwrap();
            for (modality, capacity, actual) in [
                (2, V::SIZE, download(device, &extracted.0).await),
                (1, A::SIZE, download(device, &extracted.1).await),
            ] {
                let (range, at) = &case.copies[modality];
                let mut expected = Vec::with_capacity(4 * capacity * Width::SIZE);
                for chip in 0..4 {
                    for row in 0..capacity {
                        for column in 0..Width::SIZE {
                            expected.push(if range.contains(&row) {
                                independent_packed(chip, at + row - range.start, column)
                            } else {
                                bf16::from_bits(0)
                            });
                        }
                    }
                }
                assert_bits(&actual, &expected, "independent Split", case_index);
            }
        }
    }

    #[device(chip = 4)]
    fn materialize<W: AxisName>(
        device: &mut Device,
        media: &Media<Audio, Video, W>,
    ) -> (Rows<VideoHead, W, Chips>, Rows<AudioHead, W, Chips>) {
        Materialize::<VideoHead, AudioHead>::new().forward(device, media)
    }

    async fn materialize_case<W: AxisName>(device: &mut Device) {
        let values = |modality, capacity| {
            (0..4).flat_map(move |chip| {
                (0..capacity).flat_map(move |row| {
                    (0..W::SIZE).map(move |column| source(chip, modality, row, column))
                })
            })
        };
        let media = (
            upload::<Video, W>(device, values(2, 64)).await,
            upload::<Audio, W>(device, values(1, 16)).await,
        );
        let output = launch(materialize, (&mut *device, &media)).await.unwrap();
        for (modality, copied, capacity, actual) in [
            (2, 64, 80, download(device, &output.0).await),
            (1, 16, 32, download(device, &output.1).await),
        ] {
            let expected: Vec<_> = (0..4)
                .flat_map(|chip| {
                    (0..capacity).flat_map(move |row| {
                        (0..W::SIZE).map(move |column| {
                            if row < copied {
                                source(chip, modality, row, column)
                            } else {
                                bf16::from_bits(0)
                            }
                        })
                    })
                })
                .collect();
            assert_bits(&actual, &expected, "Materialize", 0);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn materialize_preserves_source_padding_and_initializes_head_extensions() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
        materialize_case::<HeadWidth>(&mut device).await;
        materialize_case::<NarrowHead>(&mut device).await;
    }
}
