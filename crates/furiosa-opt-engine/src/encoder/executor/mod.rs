use super::{Audio, Encoder, Image};
use crate::decoder::{HbmTensor, Rows};
use furiosa_opt_models::Module;
use furiosa_opt_models::encoder as encoder_model;
use furiosa_opt_models::load::Load;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
use std::io;
use std::pin::Pin;
use std::sync::Arc;

mod execution;
mod planner;

use self::execution::{embed_frames, embed_patches, keep_rows, output_buffers, output_tensors};
use self::planner::{fits_short, launches, validate_positions};

/// `R` rows of `W` BF16 features on one chip: a launch's input or its output.
type Tensor<R, W> = furiosa_opt_std::prelude::HbmTensor<bf16, Chip, m![R, W]>;

/// One offset of each of a launch's `R` patches.
type Offsets<R> = furiosa_opt_std::prelude::HbmTensor<i32, Chip, m![R]>;

/// A launch of `R` rows of `E`'s patches: the patches, their offsets for x and for y, and the
/// launch's rows.
type Patches<'x, E, R> = (
    &'x Tensor<R, <E as encoder_model::Vision>::Patch>,
    &'x Offsets<R>,
    &'x Offsets<R>,
    &'x mut Tensor<R, <E as encoder_model::Encoder>::Text>,
);

/// A launch of `R` rows of `E`'s audio frames, and the launch's rows.
type Frames<'x, E, R> = (
    &'x Tensor<R, <E as encoder_model::Audio>::Samples>,
    &'x mut Tensor<R, <E as encoder_model::Encoder>::Text>,
);

/// A loaded encoder tree that embeds vision launches of `R` rows.
trait EmbedsVision<E, R>:
    DeviceSend
    + Send
    + Sync
    + for<'x, 's> Module<
        Patches<'x, E, R>,
        (),
        &'s mut <E as encoder_model::Vision>::VisionState,
        Output = (),
    >
where
    E: encoder_model::Encoder + encoder_model::Vision,
    R: AxisName,
    for<'a> E::View<'a>: Load<&'a E>,
{
}

impl<T, E, R> EmbedsVision<E, R> for T
where
    E: encoder_model::Encoder + encoder_model::Vision,
    R: AxisName,
    T: DeviceSend
        + Send
        + Sync
        + for<'x, 's> Module<
            Patches<'x, E, R>,
            (),
            &'s mut <E as encoder_model::Vision>::VisionState,
            Output = (),
        >,
{
}

/// A loaded encoder tree that embeds audio launches of `R` rows.
trait EmbedsAudio<E, R>:
    DeviceSend
    + Send
    + Sync
    + for<'x, 's> Module<
        Frames<'x, E, R>,
        (),
        &'s mut <E as encoder_model::Audio>::AudioState,
        Output = (),
    >
where
    E: encoder_model::Encoder + encoder_model::Audio,
    R: AxisName,
    for<'a> E::View<'a>: Load<&'a E>,
{
}

impl<T, E, R> EmbedsAudio<E, R> for T
where
    E: encoder_model::Encoder + encoder_model::Audio,
    R: AxisName,
    T: DeviceSend
        + Send
        + Sync
        + for<'x, 's> Module<
            Frames<'x, E, R>,
            (),
            &'s mut <E as encoder_model::Audio>::AudioState,
            Output = (),
        >,
{
}

trait VisionPath: Send + Sync {
    fn input_width(&self) -> usize;
    fn embed<'a>(
        &'a self,
        device: &'a mut Device,
        images: &'a [Image],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>>;
}

trait AudioPath: Send + Sync {
    fn input_width(&self) -> usize;
    fn embed<'a>(
        &'a self,
        device: &'a mut Device,
        clips: &'a [Audio],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>>;
}

struct VisionExecutor<E>(Arc<E>);
struct AudioExecutor<E>(Arc<E>);

impl<E> VisionPath for VisionExecutor<E>
where
    E: encoder_model::Encoder + encoder_model::Vision + Send + Sync + 'static,
    for<'a> E::View<'a>: EmbedsVision<E, <E as encoder_model::Vision>::Short>
        + EmbedsVision<E, <E as encoder_model::Vision>::Long>,
{
    fn input_width(&self) -> usize {
        <E as encoder_model::Vision>::Patch::SIZE
    }

    fn embed<'a>(
        &'a self,
        device: &'a mut Device,
        images: &'a [Image],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>> {
        Box::pin(embed_vision(self.0.as_ref(), device, images))
    }
}

impl<E> AudioPath for AudioExecutor<E>
where
    E: encoder_model::Encoder + encoder_model::Audio + Send + Sync + 'static,
    for<'a> E::View<'a>: EmbedsAudio<E, <E as encoder_model::Audio>::Short>
        + EmbedsAudio<E, <E as encoder_model::Audio>::Long>,
{
    fn input_width(&self) -> usize {
        <E as encoder_model::Audio>::Samples::SIZE
    }

    fn embed<'a>(
        &'a self,
        device: &'a mut Device,
        clips: &'a [Audio],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>> {
        Box::pin(embed_audio(self.0.as_ref(), device, clips))
    }
}

/// Selects the statically implemented paths a model exposes at runtime.
pub(crate) struct Builder<E> {
    model: Arc<E>,
    vision: Option<Box<dyn VisionPath>>,
    audio: Option<Box<dyn AudioPath>>,
}

impl<E> Builder<E>
where
    E: encoder_model::Encoder + Send + Sync + 'static,
{
    pub(crate) fn new(model: E) -> Self {
        Self {
            model: Arc::new(model),
            vision: None,
            audio: None,
        }
    }

    pub(crate) fn vision(mut self) -> Self
    where
        E: encoder_model::Vision,
        for<'a> E::View<'a>: EmbedsVision<E, <E as encoder_model::Vision>::Short>
            + EmbedsVision<E, <E as encoder_model::Vision>::Long>,
    {
        assert_launch_rows(
            "vision",
            <E as encoder_model::Vision>::Short::SIZE,
            <E as encoder_model::Vision>::Long::SIZE,
        );
        self.vision = Some(Box::new(VisionExecutor(Arc::clone(&self.model))));
        self
    }

    pub(crate) fn audio(mut self) -> Self
    where
        E: encoder_model::Audio,
        for<'a> E::View<'a>: EmbedsAudio<E, <E as encoder_model::Audio>::Short>
            + EmbedsAudio<E, <E as encoder_model::Audio>::Long>,
    {
        assert_launch_rows(
            "audio",
            <E as encoder_model::Audio>::Short::SIZE,
            <E as encoder_model::Audio>::Long::SIZE,
        );
        self.audio = Some(Box::new(AudioExecutor(Arc::clone(&self.model))));
        self
    }

    pub(crate) fn build(self) -> Box<dyn Encoder> {
        assert!(
            self.vision.is_some() || self.audio.is_some(),
            "an encoder must expose at least one supported modality"
        );
        Box::new(EncoderAdapter {
            output_width: <E as encoder_model::Encoder>::Text::SIZE,
            vision: self.vision,
            audio: self.audio,
        })
    }
}

/// Assert a valid compiled launch shape while registering the model path.
fn assert_launch_rows(modality: &str, short: usize, long: usize) {
    assert!(
        short > 0,
        "{modality} short launch row count must be positive"
    );
    assert!(
        short <= long,
        "{modality} short launch row count must not exceed long launch row count"
    );
}

struct EncoderAdapter {
    output_width: usize,
    vision: Option<Box<dyn VisionPath>>,
    audio: Option<Box<dyn AudioPath>>,
}

impl super::sealed::Sealed for EncoderAdapter {}

impl Encoder for EncoderAdapter {
    fn vision_input_width(&self) -> Option<usize> {
        self.vision.as_ref().map(|path| path.input_width())
    }

    fn audio_input_width(&self) -> Option<usize> {
        self.audio.as_ref().map(|path| path.input_width())
    }

    fn output_width(&self) -> usize {
        self.output_width
    }

    fn vision<'a>(
        &'a self,
        device: &'a mut Device,
        images: &'a [Image],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>> {
        match self.vision.as_ref() {
            Some(path) => path.embed(device, images),
            None => unsupported("vision"),
        }
    }

    fn audio<'a>(
        &'a self,
        device: &'a mut Device,
        clips: &'a [Audio],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>> {
        match self.audio.as_ref() {
            Some(path) => path.embed(device, clips),
            None => unsupported("audio"),
        }
    }
}

fn unsupported<'a>(
    modality: &'static str,
) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>> {
    Box::pin(async move {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("this encoder has no {modality} capability"),
        ))
    })
}

/// Execute a model's vision capability with its own compiled row shapes.
async fn embed_vision<E>(
    encoder: &E,
    device: &mut Device,
    images: &[Image],
) -> io::Result<Vec<HbmTensor>>
where
    E: encoder_model::Encoder + encoder_model::Vision + Sync,
    for<'a> E::View<'a>: EmbedsVision<E, <E as encoder_model::Vision>::Short>
        + EmbedsVision<E, <E as encoder_model::Vision>::Long>,
{
    validate_positions(
        images.iter().flat_map(|image| image.positions.iter()),
        <E as encoder_model::Vision>::Positions::SIZE,
    )?;
    let patches = (images.iter())
        .map(Image::patches::<Symbol<<E as encoder_model::Vision>::Patch>>)
        .collect::<io::Result<Vec<_>>>()?;
    let counts: Vec<_> = patches.iter().map(Rows::len).collect();
    let short = <E as encoder_model::Vision>::Short::SIZE;
    let long = <E as encoder_model::Vision>::Long::SIZE;
    let launches = launches(counts.iter().copied(), long);
    let mut output =
        output_buffers::<Symbol<<E as encoder_model::Encoder>::Text>>(device, &counts, &launches)?;
    for launch in &launches {
        let chunks: Vec<_> = launch
            .items
            .iter()
            .map(|item| {
                Rows::<bf16, Symbol<<E as encoder_model::Vision>::Patch>>::new(
                    patches[item.item].buffer(),
                    item.rows.clone(),
                )
            })
            .collect::<io::Result<_>>()?;
        let items: Vec<_> = launch
            .items
            .iter()
            .zip(&chunks)
            .map(|(item, patches)| (patches, &images[item.item].positions[item.rows.clone()]))
            .collect();
        let rows = match fits_short(launch.rows, short) {
            true => {
                embed_patches::<E, <E as encoder_model::Vision>::Short>(encoder, device, &items)
                    .await?
            }
            false => {
                embed_patches::<E, <E as encoder_model::Vision>::Long>(encoder, device, &items)
                    .await?
            }
        };
        keep_rows(device, launch, rows, &mut output)?;
    }
    output_tensors::<Symbol<<E as encoder_model::Encoder>::Text>>(output, &counts)
}

/// Execute a model's audio capability with its own compiled row shapes.
async fn embed_audio<E>(
    encoder: &E,
    device: &mut Device,
    clips: &[Audio],
) -> io::Result<Vec<HbmTensor>>
where
    E: encoder_model::Encoder + encoder_model::Audio + Sync,
    for<'a> E::View<'a>: EmbedsAudio<E, <E as encoder_model::Audio>::Short>
        + EmbedsAudio<E, <E as encoder_model::Audio>::Long>,
{
    let frames = (clips.iter())
        .map(Audio::frames::<Symbol<<E as encoder_model::Audio>::Samples>>)
        .collect::<io::Result<Vec<_>>>()?;
    let counts: Vec<_> = frames.iter().map(Rows::len).collect();
    let short = <E as encoder_model::Audio>::Short::SIZE;
    let long = <E as encoder_model::Audio>::Long::SIZE;
    let launches = launches(counts.iter().copied(), long);
    let mut output =
        output_buffers::<Symbol<<E as encoder_model::Encoder>::Text>>(device, &counts, &launches)?;
    for launch in &launches {
        let chunks: Vec<_> = launch
            .items
            .iter()
            .map(|item| {
                Rows::<bf16, Symbol<<E as encoder_model::Audio>::Samples>>::new(
                    frames[item.item].buffer(),
                    item.rows.clone(),
                )
            })
            .collect::<io::Result<_>>()?;
        let items: Vec<_> = chunks.iter().collect();
        let rows = match fits_short(launch.rows, short) {
            true => {
                embed_frames::<E, <E as encoder_model::Audio>::Short>(encoder, device, &items)
                    .await?
            }
            false => {
                embed_frames::<E, <E as encoder_model::Audio>::Long>(encoder, device, &items)
                    .await?
            }
        };
        keep_rows(device, launch, rows, &mut output)?;
    }
    output_tensors::<Symbol<<E as encoder_model::Encoder>::Text>>(output, &counts)
}
