//! Serve each modality a model's encoder supports: framework inputs become rows of the model's
//! output width, which the framework places at the media's placeholder tokens.
//!
//! - An [`Image`] is merged patches in HBM, one row each, with each patch's (x, y) position in its
//!   frame; the frames of a video are one image each, or one image of all their patches.
//! - [`Audio`] is frames of samples in HBM, one row each.
//! - A framework serves every encoder as one [`Encoder`] object whatever its model. Vision and audio
//!   are independent capabilities; each supported call packs its items using that modality's
//!   compiled row counts and returns each item's rows in order.
//!
//! These input types cover positioned patch features and audio-frame rows. Another processor input
//! form needs its own typed capability; the runtime API does not erase distinct media into a generic
//! container.

mod executor;
pub(crate) use executor::Builder;

mod sealed {
    pub trait Sealed {}
}

use crate::decoder::{HbmTensor, Rows};
use furiosa_opt_std::prelude::*;
use std::io;
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

/// The modalities a framework can run from one loaded encoder model.
///
/// Each supported method returns one output row per input row, in order. Unsupported methods report
/// `io::ErrorKind::Unsupported`. Engine registry builds these facades; Rust callers consume them.
#[allow(private_bounds)]
pub trait Encoder: sealed::Sealed + Send + Sync {
    /// Number of BF16 values in each input patch row, if this model embeds vision.
    fn vision_input_width(&self) -> Option<usize> {
        None
    }

    /// Number of BF16 values in each input audio frame, if this model embeds audio.
    fn audio_input_width(&self) -> Option<usize> {
        None
    }

    /// Number of BF16 values in each output row.
    fn output_width(&self) -> usize;

    /// Each image's rows, one per patch, in order; unsupported if the model has no vision path.
    fn vision<'a>(
        &'a self,
        _device: &'a mut Device,
        _images: &'a [Image],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>> {
        Box::pin(async {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "this encoder has no vision capability",
            ))
        })
    }

    /// Each clip's rows, one per frame, in order; unsupported if the model has no audio path.
    fn audio<'a>(
        &'a self,
        _device: &'a mut Device,
        _clips: &'a [Audio],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<HbmTensor>>> + Send + 'a>> {
        Box::pin(async {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "this encoder has no audio capability",
            ))
        })
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
