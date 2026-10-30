//! Gemma 4 unified's listed encoder as a media encoder: an image's or frame's patches through the
//! patch path and `embed_vision`, audio frames through `embed_audio`, each launch compiled for the
//! fewest of 512 or 2304 rows that hold its items.

use super::{Audio, Encoder, Image, launches};
use crate::decoder::{HbmTensor, Rows};
use crate::registry::{Listed, Model};
use crate::{AsyncModule, Function, forward};
use furiosa_opt_models::load::{Context, Load, LoadError, Part};
use furiosa_opt_models::models::vision_language::gemma4_unified::{
    self, Frames, Offsets, Patch, Patches, Positions, Samples, Text, offset,
};
use furiosa_opt_models::ops::attention::full::paged::Tiling;
use furiosa_opt_models::ops::index::copy;
use furiosa_opt_models::tensor::memory::{Hbm, alias};
use furiosa_opt_std::prelude::*;
use std::io;
use std::pin::Pin;

axes![Short = 512, Long = 2304];

/// The media encoder reads a load of the encoder part alone, and serves media; its rows are
/// computed alone, so no tiling applies to them.
impl Listed for gemma4_unified::Encoder<Hbm> {
    fn reads(part: Part) -> Result<(), LoadError> {
        match part {
            Part::Encoder => Ok(()),
            Part::Whole => Err(LoadError::Unsupported {
                name: "encoder".into(),
                reason: "a media encoder loads alone, so choose the encoder".into(),
            }),
        }
    }

    fn served(self, _: Tiling) -> Model {
        Model::Encoder(Box::new(self))
    }
}

impl Encoder for gemma4_unified::Encoder<Hbm> {
    fn vision<'a>(
        &'a self,
        device: &'a mut Device,
        images: &'a [Image],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>> {
        Box::pin(async move {
            let outside = |xy: &[i32; 2]| xy.iter().any(|&at| at as usize >= Positions::SIZE);
            if images
                .iter()
                .any(|image| image.positions.iter().any(outside))
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "a patch position reaches past the {} positions",
                        Positions::SIZE
                    ),
                ));
            }
            let patches = (images.iter())
                .map(Image::patches::<m![Patch]>)
                .collect::<io::Result<Vec<_>>>()?;
            let mut rows = Vec::with_capacity(images.len());
            for launch in launches(patches.iter().map(Rows::len), Long::SIZE)? {
                let items: Vec<_> = (launch.iter())
                    .map(|&at| (&patches[at], &images[at].positions[..]))
                    .collect();
                let count: usize = items.iter().map(|(patches, _)| patches.len()).sum();
                rows.extend(match count <= Short::SIZE {
                    true => embed_patches::<Short>(self, device, &items).await?,
                    false => embed_patches::<Long>(self, device, &items).await?,
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
                .map(Audio::frames::<m![Samples]>)
                .collect::<io::Result<Vec<_>>>()?;
            let mut rows = Vec::with_capacity(clips.len());
            for launch in launches(frames.iter().map(Rows::len), Long::SIZE)? {
                let items: Vec<_> = launch.iter().map(|&at| &frames[at]).collect();
                let count: usize = items.iter().map(|frames| frames.len()).sum();
                rows.extend(match count <= Short::SIZE {
                    true => embed_frames::<Short>(self, device, &items).await?,
                    false => embed_frames::<Long>(self, device, &items).await?,
                });
            }
            rows.into_iter().map(HbmTensor::try_from).collect()
        })
    }
}

/// The items' rows of one launch of `R` rows: their inputs copied one after another into the
/// launch's rows, the rows past them computed and read by no one.
async fn launch<R: AxisName, Input: M>(
    device: &mut Device,
    items: &[&Rows<bf16, Input>],
    run: impl AsyncFnOnce(&mut Device, HbmBuffer, &HbmBuffer) -> io::Result<()>,
) -> io::Result<Vec<Rows<bf16, m![Text]>>> {
    let row = Rows::<bf16, Input>::bytes();
    let starts: Vec<usize> = (items.iter())
        .scan(0, |at, rows| {
            let start = *at;
            *at += rows.len();
            Some(start)
        })
        .collect();
    let inputs = HbmBuffer::alloc(device, R::SIZE * row).map_err(io::Error::other)?;
    let mut targets = (items.iter().zip(&starts))
        .map(|(rows, &start)| inputs.alias(start * row..(start + rows.len()) * row))
        .collect::<Result<Vec<_>, _>>()
        .map_err(io::Error::other)?;
    let sources = items.iter().map(|rows| rows.buffer());
    copy(&mut *device, sources.zip(&mut targets))?;
    let output = HbmBuffer::alloc(device, R::SIZE * Rows::<bf16, m![Text]>::bytes())
        .map_err(io::Error::other)?;
    run(device, inputs, &output).await?;
    (items.iter().zip(&starts))
        .map(|(rows, &start)| Rows::new(&output, start..start + rows.len()))
        .collect()
}

/// The rows of one launch of `R` rows of `items`' patches; a row past them reads position 0.
async fn embed_patches<R: AxisName + Send + Sync>(
    encoder: &gemma4_unified::Encoder<Hbm>,
    device: &mut Device,
    items: &[(&Rows<bf16, m![Patch]>, &[[i32; 2]])],
) -> io::Result<Vec<Rows<bf16, m![Text]>>> {
    let offsets = |coordinate: usize| {
        let positions = items.iter().flat_map(|(_, positions)| positions.iter());
        let mut offsets: Vec<i32> = positions
            .map(|xy| offset(xy[coordinate] as usize, coordinate))
            .collect();
        offsets.resize(R::SIZE, offset(0, coordinate));
        HostTensor::<i32, m![R]>::from_vec(offsets)
    };
    let (x, y) = (offsets(0), offsets(1));
    let inputs: Vec<_> = items.iter().map(|(patches, _)| *patches).collect();
    let embed = async move |device: &mut Device, inputs: HbmBuffer, output: &HbmBuffer| {
        let x: Offsets<R> = x.to_hbm(&mut device.pdma).await.map_err(io::Error::other)?;
        let y: Offsets<R> = y.to_hbm(&mut device.pdma).await.map_err(io::Error::other)?;
        let patches: Patches<R> = alias(&inputs, ..)?;
        let mut rows: gemma4_unified::Rows<R> = alias(output, ..)?;
        let model: gemma4_unified::Encoder<&Hbm> =
            Load::load(&mut Context::new(None), encoder).map_err(io::Error::other)?;
        let input = (&patches, &x, &y, &mut rows);
        Function::new(&model, forward)
            .forward(device, (), &mut Default::default(), input)
            .await
    };
    launch::<R, m![Patch]>(device, &inputs, embed).await
}

/// The rows of one launch of `R` rows of `items`' audio frames.
async fn embed_frames<R: AxisName + Send + Sync>(
    encoder: &gemma4_unified::Encoder<Hbm>,
    device: &mut Device,
    items: &[&Rows<bf16, m![Samples]>],
) -> io::Result<Vec<Rows<bf16, m![Text]>>> {
    let embed = async move |device: &mut Device, inputs: HbmBuffer, output: &HbmBuffer| {
        let frames: Frames<R> = alias(&inputs, ..)?;
        let mut rows: gemma4_unified::Rows<R> = alias(output, ..)?;
        let model: gemma4_unified::Encoder<&Hbm> =
            Load::load(&mut Context::new(None), encoder).map_err(io::Error::other)?;
        Function::new(&model, forward)
            .forward(device, (), &mut Default::default(), (&frames, &mut rows))
            .await
    };
    launch::<R, m![Samples]>(device, items, embed).await
}
