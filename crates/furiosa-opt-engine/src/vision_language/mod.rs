//! Serve a vision-language model's media encoder: a call turns images, video frames or audio into
//! rows of the language model's width, which a framework places at the media's placeholder tokens.
//!
//! - An [`Image`] is merged patches in HBM, one row each, with each patch's (x, y) position in its
//!   frame; the frames of a video are one image each, or one image of all their patches.
//! - [`Audio`] is frames of samples in HBM, one row each.
//! - A framework serves every media encoder as an [`Encoder`], one object whatever its model: a call
//!   packs its items into launches of the model's two compiled row counts and returns each item's
//!   rows in order, a view of its launch's rows, once the device accepted the launches.

use crate::decoder::{HbmTensor, Rows};
use crate::{AsyncModule, Function, forward};
use furiosa_opt_models::Module;
use furiosa_opt_models::load::{Context, Load};
use furiosa_opt_models::models::vision_language as media;
use furiosa_opt_models::ops::index::copy;
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
use std::io;
use std::marker::PhantomData;
use std::ops::Range;
use std::pin::Pin;

/// An image or a video's frames as a processor gives them: merged patches in HBM, one row each,
/// and each patch's (x, y) position in its frame.
pub struct Image {
    patches: HbmBuffer,
    positions: Vec<[i32; 2]>,
}

impl Image {
    /// The patches stored from the start of `patches`, one for each of `positions`; refuses a
    /// negative position, as a processor's padding patches are not an image's.
    pub fn new(patches: HbmBuffer, positions: Vec<[i32; 2]>) -> io::Result<Self> {
        if positions.iter().flatten().any(|&coordinate| coordinate < 0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a patch position is negative: pass the image's valid patches only",
            ));
        }
        Ok(Self { patches, positions })
    }

    /// The image's patches, of the width `Row` maps.
    fn patches<Row: M>(&self) -> io::Result<Rows<bf16, Row>> {
        rows(&self.patches, self.positions.len())
    }
}

/// Audio as a processor gives it: frames of samples in HBM, one row each.
pub struct Audio {
    frames: HbmBuffer,
    count: usize,
}

impl Audio {
    /// The first `count` frames stored from the start of `frames`.
    pub fn new(frames: HbmBuffer, count: usize) -> Self {
        Self { frames, count }
    }

    /// The clip's frames, of the width `Row` maps.
    fn frames<Row: M>(&self) -> io::Result<Rows<bf16, Row>> {
        rows(&self.frames, self.count)
    }
}

/// The first `count` rows of `buffer`, refused unless the buffer holds exactly them.
fn rows<Row: M>(buffer: &HbmBuffer, count: usize) -> io::Result<Rows<bf16, Row>> {
    if buffer.size() != count * Rows::<bf16, Row>::bytes() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{} bytes do not hold {count} rows of {} BF16 values",
                buffer.size(),
                Row::SIZE
            ),
        ));
    }
    Rows::new(buffer, 0..count)
}

/// A vision-language model's media encoder as a framework serves it, one object whatever its
/// model: each item's rows of the language model's input, `[rows, width]` BF16.
pub trait Encoder: Send + Sync {
    /// Each of `images`' rows, one a patch, in order.
    fn vision<'a>(
        &'a self,
        device: &'a mut Device,
        images: &'a [Image],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>>;

    /// Each of `clips`' rows, one a frame, in order.
    fn audio<'a>(
        &'a self,
        device: &'a mut Device,
        clips: &'a [Audio],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>>;
}

/// `R` rows of `W` BF16 features on one chip: a launch's input or its output.
type Tensor<R, W> = furiosa_opt_std::prelude::HbmTensor<bf16, Chip, m![R, W]>;

/// One offset of each of a launch's `R` patches.
type Offsets<R> = furiosa_opt_std::prelude::HbmTensor<i32, Chip, m![R]>;

/// A launch of `R` rows of `E`'s patches: the patches, their offsets for x and for y, and the
/// launch's rows.
type Patches<'x, E, R> = (
    &'x Tensor<R, <E as media::Encoder>::Patch>,
    &'x Offsets<R>,
    &'x Offsets<R>,
    &'x mut Tensor<R, <E as media::Encoder>::Text>,
);

/// A launch of `R` rows of `E`'s audio frames, and the launch's rows.
type Frames<'x, E, R> = (
    &'x Tensor<R, <E as media::Encoder>::Samples>,
    &'x mut Tensor<R, <E as media::Encoder>::Text>,
);

/// The tree of a media encoder `E` whose launches of `R` rows, of patches and of audio frames,
/// the entry runs.
trait Embeds<E: media::Encoder, R: AxisName>:
    DeviceSend
    + Send
    + Sync
    + for<'x, 's> Module<Patches<'x, E, R>, (), &'s mut E::Vision, Output = ()>
    + for<'x, 's> Module<Frames<'x, E, R>, (), &'s mut E::Audio, Output = ()>
{
}

impl<T, E: media::Encoder, R: AxisName> Embeds<E, R> for T where
    T: DeviceSend
        + Send
        + Sync
        + for<'x, 's> Module<Patches<'x, E, R>, (), &'s mut E::Vision, Output = ()>
        + for<'x, 's> Module<Frames<'x, E, R>, (), &'s mut E::Audio, Output = ()>
{
}

/// Every media encoder: a launch runs at `Short` rows when its items fit, else at `Long`.
impl<E> Encoder for E
where
    E: media::Encoder + Send + Sync,
    for<'a> E::View<'a>: Embeds<E, E::Short> + Embeds<E, E::Long>,
{
    fn vision<'a>(
        &'a self,
        device: &'a mut Device,
        images: &'a [Image],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>> {
        Box::pin(async move {
            let outside = |xy: &[i32; 2]| xy.iter().any(|&at| at as usize >= E::Positions::SIZE);
            if images
                .iter()
                .any(|image| image.positions.iter().any(outside))
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "a patch position reaches past the {} positions",
                        E::Positions::SIZE
                    ),
                ));
            }
            let patches = (images.iter())
                .map(Image::patches::<Symbol<E::Patch>>)
                .collect::<io::Result<Vec<_>>>()?;
            let mut rows = Vec::with_capacity(images.len());
            for launch in launches(patches.iter().map(Rows::len), E::Long::SIZE)? {
                let items: Vec<_> = (launch.iter())
                    .map(|&at| (&patches[at], &images[at].positions[..]))
                    .collect();
                let count: usize = items.iter().map(|(patches, _)| patches.len()).sum();
                rows.extend(match count <= E::Short::SIZE {
                    true => embed_patches::<E, E::Short>(self, device, &items).await?,
                    false => embed_patches::<E, E::Long>(self, device, &items).await?,
                });
            }
            rows.into_iter().map(HbmTensor::try_from).collect()
        })
    }

    fn audio<'a>(
        &'a self,
        device: &'a mut Device,
        clips: &'a [Audio],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>> {
        Box::pin(async move {
            let frames = (clips.iter())
                .map(Audio::frames::<Symbol<E::Samples>>)
                .collect::<io::Result<Vec<_>>>()?;
            let mut rows = Vec::with_capacity(clips.len());
            for launch in launches(frames.iter().map(Rows::len), E::Long::SIZE)? {
                let items: Vec<_> = launch.iter().map(|&at| &frames[at]).collect();
                let count: usize = items.iter().map(|frames| frames.len()).sum();
                rows.extend(match count <= E::Short::SIZE {
                    true => embed_frames::<E, E::Short>(self, device, &items).await?,
                    false => embed_frames::<E, E::Long>(self, device, &items).await?,
                });
            }
            rows.into_iter().map(HbmTensor::try_from).collect()
        })
    }
}

/// The launches of a call of items of `rows` rows each, in order: consecutive items whose rows fit
/// in `limit`, refusing an item larger than that.
fn launches(rows: impl IntoIterator<Item = usize>, limit: usize) -> io::Result<Vec<Vec<usize>>> {
    let packed = rows.into_iter().enumerate().try_fold(
        Vec::<(usize, Vec<usize>)>::new(),
        |mut launches, (item, count)| {
            if count > limit {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("item {item} holds {count} rows; a launch takes at most {limit}"),
                ));
            }
            match launches.last_mut() {
                Some((filled, items)) if *filled + count <= limit => {
                    *filled += count;
                    items.push(item);
                }
                _ => launches.push((count, vec![item])),
            }
            Ok(launches)
        },
    )?;
    Ok(packed.into_iter().map(|(_, items)| items).collect())
}

/// One launch's buffers: its items' rows copied one after another into its input rows, the rows past
/// them computed and read by no one, and its output rows of width `Text`.
struct Launch<Text> {
    input: HbmBuffer,
    output: HbmBuffer,
    items: Vec<Range<usize>>,
    text: PhantomData<Text>,
}

impl<Text: M> Launch<Text> {
    /// The buffers of a launch of `R` rows of `items`.
    fn new<R: AxisName, Input: M>(
        device: &mut Device,
        items: &[&Rows<bf16, Input>],
    ) -> io::Result<Self> {
        let row = Rows::<bf16, Input>::bytes();
        let ranges: Vec<Range<usize>> = (items.iter())
            .scan(0, |at, rows| {
                let start = *at;
                *at += rows.len();
                Some(start..*at)
            })
            .collect();
        let input = HbmBuffer::alloc(device, R::SIZE * row).map_err(io::Error::other)?;
        let mut targets = (ranges.iter())
            .map(|range| input.alias(range.start * row..range.end * row))
            .collect::<Result<Vec<_>, _>>()
            .map_err(io::Error::other)?;
        let sources = items.iter().map(|rows| rows.buffer());
        copy(&mut *device, sources.zip(&mut targets))?;
        let output = HbmBuffer::alloc(device, R::SIZE * Rows::<bf16, Text>::bytes())
            .map_err(io::Error::other)?;
        Ok(Self {
            input,
            output,
            items: ranges,
            text: PhantomData,
        })
    }

    /// Each item's rows, views of the launch's output rows.
    fn rows(&self) -> io::Result<Vec<Rows<bf16, Text>>> {
        (self.items.iter())
            .map(|range| Rows::new(&self.output, range.clone()))
            .collect()
    }
}

/// The rows of one launch of `R` rows of `items`' patches; a row past them reads position 0.
async fn embed_patches<E, R>(
    encoder: &E,
    device: &mut Device,
    items: &[(&Rows<bf16, Symbol<E::Patch>>, &[[i32; 2]])],
) -> io::Result<Vec<Rows<bf16, Symbol<E::Text>>>>
where
    E: media::Encoder + Sync,
    R: AxisName + Send + Sync,
    for<'a> E::View<'a>: Embeds<E, R>,
{
    let offsets = |coordinate: usize| {
        let positions = items.iter().flat_map(|(_, positions)| positions.iter());
        let mut offsets: Vec<i32> = positions
            .map(|xy| E::offset(xy[coordinate] as usize, coordinate))
            .collect();
        offsets.resize(R::SIZE, E::offset(0, coordinate));
        HostTensor::<i32, m![R]>::from_vec(offsets)
    };
    let (x, y) = (offsets(0), offsets(1));
    let inputs: Vec<_> = items.iter().map(|(patches, _)| *patches).collect();
    let launch = Launch::new::<R, _>(device, &inputs)?;
    let x: Offsets<R> = x.to_hbm(&mut device.pdma).await.map_err(io::Error::other)?;
    let y: Offsets<R> = y.to_hbm(&mut device.pdma).await.map_err(io::Error::other)?;
    let patches: Tensor<R, E::Patch> = alias(&launch.input, ..)?;
    let mut rows: Tensor<R, E::Text> = alias(&launch.output, ..)?;
    let model: E::View<'_> =
        Load::load(&mut Context::new(None), encoder).map_err(io::Error::other)?;
    let input = (&patches, &x, &y, &mut rows);
    Function::new(&model, forward)
        .forward(device, (), &mut E::Vision::default(), input)
        .await?;
    launch.rows()
}

/// The rows of one launch of `R` rows of `items`' audio frames.
async fn embed_frames<E, R>(
    encoder: &E,
    device: &mut Device,
    items: &[&Rows<bf16, Symbol<E::Samples>>],
) -> io::Result<Vec<Rows<bf16, Symbol<E::Text>>>>
where
    E: media::Encoder + Sync,
    R: AxisName + Send + Sync,
    for<'a> E::View<'a>: Embeds<E, R>,
{
    let launch = Launch::new::<R, _>(device, items)?;
    let frames: Tensor<R, E::Samples> = alias(&launch.input, ..)?;
    let mut rows: Tensor<R, E::Text> = alias(&launch.output, ..)?;
    let model: E::View<'_> =
        Load::load(&mut Context::new(None), encoder).map_err(io::Error::other)?;
    Function::new(&model, forward)
        .forward(device, (), &mut E::Audio::default(), (&frames, &mut rows))
        .await?;
    launch.rows()
}
