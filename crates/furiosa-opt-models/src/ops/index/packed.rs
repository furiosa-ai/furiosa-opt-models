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
        let mut media: Self::Output = (HbmTensor::new(), HbmTensor::new());
        Span::<VIDEO_LIVE, VIDEO_ROWS>::new().forward(&mut *device, (video, (&mut media.0, 0)));
        Span::<AUDIO_LIVE, AUDIO_ROWS>::new().forward(&mut *device, (audio, (&mut media.1, 0)));
        media
    }
}

impl Join {
    /// Each span's first strip, its strips and its table row, in packing order, in strips of
    /// the sequence's strip rows.
    pub(crate) const fn spans<T: AxisName, A: AxisName, V: AxisName>() -> [(usize, usize, usize); 3]
    {
        let unit = Packed::<T, A, V>::STRIP;
        let (text, audio) = (T::SIZE / unit, A::SIZE / unit);
        [
            (0, text, 1),
            (text, audio, 5),
            (text + audio, V::SIZE / unit, 0),
        ]
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
