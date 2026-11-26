use super::planner::PackedLaunch;
use super::{EmbedsAudio, EmbedsVision, Offsets, Tensor};
use crate::decoder::{HbmTensor, Rows};
use crate::{AsyncModule, Function, forward};
use furiosa_opt_models::encoder as encoder_model;
use furiosa_opt_models::load::{Context, Load};
use furiosa_opt_models::ops::index::copy;
use furiosa_opt_models::tensor::memory::Dtype;
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_std::prelude::*;
use std::io;
use std::marker::PhantomData;
use std::ops::Range;

/// One output allocation per item split across launches; a single launch's rows can be returned
/// directly without copying.
pub(super) fn output_buffers<Text: M>(
    device: &mut Device,
    counts: &[usize],
    launches: &[PackedLaunch],
) -> io::Result<Vec<Option<HbmBuffer>>> {
    let mut pieces = vec![0; counts.len()];
    for item in launches.iter().flat_map(|launch| &launch.items) {
        pieces[item.item] += 1;
    }
    let bytes = Rows::<bf16, Text>::bytes();
    counts
        .iter()
        .zip(pieces)
        .map(|(&count, pieces)| {
            if pieces == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "an encoder item has no launch rows",
                ));
            }
            if pieces == 1 {
                return Ok(None);
            }
            let size = count.checked_mul(bytes).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "media output size overflows usize",
                )
            })?;
            HbmBuffer::alloc(device, size)
                .map(Some)
                .map_err(io::Error::other)
        })
        .collect()
}

/// Keep this launch's rows in their item's output. A single-piece result adopts the launch buffer;
/// split items copy into the row range reserved for that piece.
pub(super) fn keep_rows<Text: M>(
    device: &mut Device,
    launch: &PackedLaunch,
    rows: Vec<Rows<bf16, Text>>,
    output: &mut [Option<HbmBuffer>],
) -> io::Result<()> {
    let row_bytes = Rows::<bf16, Text>::bytes();
    let mut sources = Vec::new();
    let mut targets = Vec::new();
    for (item, rows) in launch.items.iter().zip(rows) {
        let Some(buffer) = output[item.item].as_ref() else {
            output[item.item] = Some(rows.into_buffer());
            continue;
        };
        // Split-item output allocation checked count * row_bytes; launch ranges fit that count.
        let start = item.rows.start * row_bytes;
        let end = item.rows.end * row_bytes;
        targets.push(buffer.alias(start..end).map_err(io::Error::other)?);
        sources.push(rows.into_buffer());
    }
    if !sources.is_empty() {
        copy(&mut *device, sources.iter().zip(&mut targets))?;
    }
    Ok(())
}

/// One output tensor per input item, in item order.
pub(super) fn output_tensors<Text: M>(
    buffers: Vec<Option<HbmBuffer>>,
    counts: &[usize],
) -> io::Result<Vec<HbmTensor>> {
    buffers
        .into_iter()
        .zip(counts)
        .map(|(buffer, &count)| {
            let buffer = buffer.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "an encoder item has no output rows",
                )
            })?;
            let rows = Rows::<bf16, Text>::new(&buffer, 0..count)?;
            Ok(HbmTensor {
                buffer: rows.into_buffer(),
                shape: vec![count, Text::SIZE],
                dtype: Dtype::BF16,
            })
        })
        .collect()
}

/// One launch's buffers: item rows in order, followed by unwritten input rows whose output rows are
/// discarded, and output rows of width `Text`.
struct Launch<Text> {
    input: HbmBuffer,
    output: HbmBuffer,
    output_ranges: Vec<Range<usize>>,
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
            output_ranges: ranges,
            text: PhantomData,
        })
    }

    /// Each item's rows, views of the launch's output rows.
    fn rows(&self) -> io::Result<Vec<Rows<bf16, Text>>> {
        (self.output_ranges.iter())
            .map(|range| Rows::new(&self.output, range.clone()))
            .collect()
    }
}

/// The rows of one launch of `R` rows of `items`' patches; a row past them reads position 0.
pub(super) async fn embed_patches<E, R>(
    encoder: &E,
    device: &mut Device,
    items: &[(
        &Rows<bf16, Symbol<<E as encoder_model::Vision>::Patch>>,
        &[[i32; 2]],
    )],
) -> io::Result<Vec<Rows<bf16, Symbol<<E as encoder_model::Encoder>::Text>>>>
where
    E: encoder_model::Encoder + encoder_model::Vision + Sync,
    R: AxisName + Send + Sync,
    for<'a> E::View<'a>: EmbedsVision<E, R>,
{
    let offsets = |coordinate: encoder_model::Coordinate| {
        let positions = items.iter().flat_map(|(_, positions)| positions.iter());
        let mut offsets: Vec<i32> = positions
            .map(|xy| {
                <E as encoder_model::Vision>::offset(xy[coordinate.index()] as usize, coordinate)
            })
            .collect();
        // The unused patch rows are independent, but their offsets still index the position table.
        offsets.resize(R::SIZE, <E as encoder_model::Vision>::offset(0, coordinate));
        HostTensor::<i32, m![R]>::from_vec(offsets)
    };
    let (x, y) = (
        offsets(encoder_model::Coordinate::X),
        offsets(encoder_model::Coordinate::Y),
    );
    let inputs: Vec<_> = items.iter().map(|(patches, _)| *patches).collect();
    let launch = Launch::new::<R, _>(device, &inputs)?;
    let x: Offsets<R> = x.to_hbm(&mut device.pdma).await.map_err(io::Error::other)?;
    let y: Offsets<R> = y.to_hbm(&mut device.pdma).await.map_err(io::Error::other)?;
    let patches: Tensor<R, <E as encoder_model::Vision>::Patch> = alias(&launch.input, ..)?;
    let mut rows: Tensor<R, <E as encoder_model::Encoder>::Text> = alias(&launch.output, ..)?;
    let model: E::View<'_> =
        Load::load(&mut Context::new(None), encoder).map_err(io::Error::other)?;
    let input = (&patches, &x, &y, &mut rows);
    Function::new(&model, forward)
        .forward(
            device,
            (),
            &mut <E as encoder_model::Vision>::VisionState::default(),
            input,
        )
        .await?;
    launch.rows()
}

/// The rows of one launch of `R` rows of `items`' audio frames.
pub(super) async fn embed_frames<E, R>(
    encoder: &E,
    device: &mut Device,
    items: &[&Rows<bf16, Symbol<<E as encoder_model::Audio>::Samples>>],
) -> io::Result<Vec<Rows<bf16, Symbol<<E as encoder_model::Encoder>::Text>>>>
where
    E: encoder_model::Encoder + encoder_model::Audio + Sync,
    R: AxisName + Send + Sync,
    for<'a> E::View<'a>: EmbedsAudio<E, R>,
{
    let launch = Launch::new::<R, _>(device, items)?;
    let frames: Tensor<R, <E as encoder_model::Audio>::Samples> = alias(&launch.input, ..)?;
    let mut rows: Tensor<R, <E as encoder_model::Encoder>::Text> = alias(&launch.output, ..)?;
    let model: E::View<'_> =
        Load::load(&mut Context::new(None), encoder).map_err(io::Error::other)?;
    Function::new(&model, forward)
        .forward(
            device,
            (),
            &mut <E as encoder_model::Audio>::AudioState::default(),
            (&frames, &mut rows),
        )
        .await?;
    launch.rows()
}
