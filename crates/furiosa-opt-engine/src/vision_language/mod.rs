//! Serve a vision-language model's media encoder: a call turns images, video frames or audio into
//! rows of the language model's width, which a framework places at the media's placeholder tokens.
//!
//! - An [`Image`] is merged patches in HBM, one row each, with each patch's (x, y) position in its
//!   frame; the frames of a video are one image each, or one image of all their patches.
//! - [`Audio`] is frames of samples in HBM, one row each.
//! - A framework serves every media encoder as an [`Encoder`], one object whatever its model: a call
//!   packs its items into launches of a few compiled row counts and returns each item's rows in
//!   order, a view of its launch's rows, once the device accepted the launches.

mod gemma4_unified;

use crate::decoder::{HbmTensor, Rows};
use furiosa_opt_std::prelude::{Device, HbmBuffer, M, bf16};
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
