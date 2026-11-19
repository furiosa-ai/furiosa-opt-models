//! Host packing and stitching for H3's fixed four-chip video decode profile.
//!
//! Ported from the preserved Furiosa Rust `registry/minimax_h3/video.rs` and `pixels.rs`.
//! A request borrows FP32 `[1,24,37,48,84]` latents, packs 49 groups in clip-major order,
//! and accepts four decoded tiles per group. The caller owns decoder execution and the
//! downloaded group buffer. No launch, device allocation or synchronization occurs here.
//!
//! Stitching retains 28 reusable `[3,28,256,256]` tiles, the final
//! `[1,3,124,768,1344]` FP32 output and one `[3,5,768,1344]` fading buffer. Processing is
//! parallel across independent tiles and planes. Spatial blends read original neighbors,
//! above first and then before. All reads of the previous temporal fade finish before
//! that buffer is overwritten.
//! `finish` moves the output vector without copying the full video. Pixel values are
//! neither clamped nor rescaled. Resolution and batch are fixed; the original optional frame
//! count keeps the last 1..=124 frames without changing the 49 decoder calls.

use core::ops::Range;
use std::io;

use super::super::host::filled;
use super::{Parts, Patch, Rows, TOKENS};
use crate::models::diffusion::minimax_h3::embeddings::Chips;
use furiosa_opt_std::prelude::*;

/// The fixed request's borrowed latent values in channel, time, row, column order.
#[derive(Clone, Copy)]
pub struct Latents<'a> {
    values: &'a [f32],
}

impl<'a> Latents<'a> {
    /// Check the exact element count of contiguous FP32 `[1,24,37,48,84]` storage.
    pub fn new(values: &'a [f32]) -> io::Result<Self> {
        if values.len() != LATENT_VALUES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "video latents require {LATENT_SHAPE:?} ({LATENT_VALUES} FP32 values), got {} values",
                    values.len()
                ),
            ));
        }
        Ok(Self { values })
    }

    /// Pack four spatial tiles of a clip, retaining the decoder's chip and row strides.
    /// Each live row is `[z_hi | z_lo | z_hi | 0]`; unused lanes and all rows after
    /// token 1,791 are finite positive zero, including register and class-token input rows.
    /// Latent statistics are not applied here, matching the preserved Rust path.
    pub fn pack(&self, group: GroupIndex) -> io::Result<Packed> {
        let mut values = filled(Chips::SIZE * Rows::SIZE * Parts::SIZE, bf16::from_f32(0.0))?;
        for chip in 0..Chips::SIZE {
            let tile = group.spatial() * Chips::SIZE + chip;
            let top = LATENT_TOPS[tile / LATENT_LEFTS.len()];
            let left = LATENT_LEFTS[tile % LATENT_LEFTS.len()];
            for token in 0..TOKENS {
                for channel in 0..CHANNELS {
                    let index =
                        ((channel * LATENT_TIME + group.clip() * CLIP_STRIDE + token / 256)
                            * LATENT_HEIGHT
                            + top
                            + token / 16 % 16)
                            * LATENT_WIDTH
                            + left
                            + token % 16;
                    let value = self.values[index];
                    let [high, low] = super::parameters::split(value);
                    for (part, value) in [high, low, high].into_iter().enumerate() {
                        let at =
                            (chip * Rows::SIZE + token) * Parts::SIZE + part * CHANNELS + channel;
                        values[at] = value;
                    }
                }
            }
        }
        Ok(HostTensor::from_vec(values))
    }
}

/// One of 49 calls: seven spatial groups for each of seven clips, in that order.
/// Construction checks `0..GROUPS`; the private representation cannot name another profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupIndex(usize);

impl GroupIndex {
    /// The call index in `0..49`.
    pub const fn index(self) -> usize {
        self.0
    }

    /// The temporal clip in `0..7`, starting at latent frame `5 * clip`.
    pub const fn clip(self) -> usize {
        self.0 / SPATIAL_GROUPS
    }

    /// The spatial group in `0..7`, four consecutive row-major tiles per group.
    pub const fn spatial(self) -> usize {
        self.0 % SPATIAL_GROUPS
    }
}

impl TryFrom<usize> for GroupIndex {
    type Error = io::Error;

    fn try_from(index: usize) -> io::Result<Self> {
        if index >= GROUPS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("video group index must be in 0..{GROUPS}, got {index}"),
            ));
        }
        Ok(Self(index))
    }
}

/// Bounded host state for one request; construction allocates all retained pixel buffers.
/// Large buffers use fallible reservations. Accepted groups do not allocate another video,
/// clip image, tile or fading buffer, and the original tile pixels are never blended in place.
pub struct Stitch {
    video: Video,
    tiles: Vec<Pixels>,
    accepted: usize,
}

impl Stitch {
    /// Reserve 28 reusable tiles, the 124-frame output, and one five-frame fading image.
    /// Their FP32 payloads total 2,214,395,904 bytes, excluding the caller's download buffer.
    pub fn new() -> io::Result<Self> {
        Self::with_frames(OUTPUT_SHAPE[2])
    }

    /// Retain the last `frames` frames, as in the original decoder's `frame_num` argument.
    /// Validate the count before reserving the large output or tile buffers.
    pub fn with_frames(frames: usize) -> io::Result<Self> {
        if !(1..=OUTPUT_SHAPE[2]).contains(&frames) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "video frame_num must be in 1..=124",
            ));
        }
        let mut tiles = Vec::new();
        tiles.try_reserve_exact(TILES).map_err(allocation)?;
        for _ in 0..TILES {
            tiles.push(Pixels::tile()?);
        }
        let video = Video::new(
            Tiling {
                tops: &PIXEL_TOPS,
                lefts: &PIXEL_LEFTS,
                kept: 3..20,
                fade: 23..28,
            },
            CLIPS,
            frames,
            TILE_SHAPE,
        )?;
        Ok(Self {
            video,
            tiles,
            accepted: 0,
        })
    }

    /// Accept contiguous FP32 `[4,2048,3072]` decoder rows, chip first, including padded rows.
    /// The source stride is 2,048 rows per chip; only the first 1,792 rows are unpacked.
    /// Both sequence and exact input length are checked before any retained buffer is mutated.
    /// The caller may reuse or release its group buffer once this method returns.
    pub fn accept(&mut self, group: GroupIndex, decoded: &[f32]) -> io::Result<()> {
        if group.index() != self.accepted {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "video stitch expects group {}, got {}",
                    self.accepted,
                    group.index()
                ),
            ));
        }
        if decoded.len() != DECODED_VALUES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "decoded video group requires [4,2048,3072] ({DECODED_VALUES} FP32 values), got {} values",
                    decoded.len()
                ),
            ));
        }
        let first = group.spatial() * Chips::SIZE;
        std::thread::scope(|scope| {
            for (tile, source) in self.tiles[first..first + Chips::SIZE]
                .iter_mut()
                .zip(decoded.chunks_exact(Rows::SIZE * Patch::SIZE))
            {
                scope.spawn(move || tile.unpack(source));
            }
        });
        if group.spatial() + 1 == SPATIAL_GROUPS {
            self.video.push(&self.tiles);
        }
        self.accepted += 1;
        Ok(())
    }

    /// Require all 49 groups, then move contiguous FP32 `[1,3,frames,768,1344]` values out.
    /// The vector's order is channel, frame, row, column; there is no full-output copy.
    pub fn finish(self) -> io::Result<Vec<f32>> {
        if self.accepted != GROUPS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "video stitch requires {GROUPS} groups, accepted {}",
                    self.accepted
                ),
            ));
        }
        Ok(self.video.values)
    }
}

/// One reusable channel-major decoded tile; small private shapes also exercise the original math.
struct Pixels {
    values: Vec<f32>,
    shape: [usize; 4],
}

impl Pixels {
    fn tile() -> io::Result<Self> {
        Ok(Self {
            values: zeroed_floats(TILE_SHAPE.iter().product())?,
            shape: TILE_SHAPE,
        })
    }

    /// A token is 3 channels × 4 frames × 16 rows × 16 columns, in runs of 16 columns.
    /// Input padding rows are deliberately absent from the unpacked tile.
    fn unpack(&mut self, source: &[f32]) {
        for (token, row) in source.chunks_exact(Patch::SIZE).take(TOKENS).enumerate() {
            for (run, pixels) in row.chunks_exact(16).enumerate() {
                let channel = run / 64;
                let frame = token / 256 * 4 + run / 16 % 4;
                let y = token / 16 % 16 * 16 + run % 16;
                let start = ((channel * 28 + frame) * 256 + y) * 256 + token % 16 * 16;
                self.values[start..start + 16].copy_from_slice(pixels);
            }
        }
    }
}

/// Private geometry of the preserved blend, with small original numerical examples in tests.
struct Tiling {
    tops: &'static [usize],
    lefts: &'static [usize],
    kept: Range<usize>,
    fade: Range<usize>,
}

impl Tiling {
    /// Blend above, then before, always from original neighbors rather than updated tiles.
    fn stitch(&self, grid: &[Pixels], plane: usize, mut write: impl FnMut(usize, f32)) {
        let (tops, lefts) = (self.tops, self.lefts);
        let side = grid[0].shape[2];
        let width = lefts[lefts.len() - 1] + side;
        let tile = |index: usize| &grid[index].values[plane * side * side..][..side * side];
        for (i, &top) in tops.iter().enumerate() {
            for (j, &left) in lefts.iter().enumerate() {
                let index = i * lefts.len() + j;
                let above =
                    (i > 0).then(|| (tile(index - lefts.len()), side - (top - tops[i - 1])));
                let before = (j > 0).then(|| (tile(index - 1), side - (left - lefts[j - 1])));
                let kept_y = tops.get(i + 1).map_or(side, |next| next - top);
                let kept_x = lefts.get(j + 1).map_or(side, |next| next - left);
                for y in 0..kept_y {
                    for x in 0..kept_x {
                        let mut pixel = tile(index)[y * side + x];
                        if let Some((above, extent)) = above.filter(|&(_, e)| y < e) {
                            let weight = y as f32 / extent as f32;
                            pixel = above[(side - extent + y) * side + x] * (1.0 - weight)
                                + pixel * weight;
                        }
                        if let Some((before, extent)) = before.filter(|&(_, e)| x < e) {
                            let weight = x as f32 / extent as f32;
                            pixel = before[y * side + side - extent + x] * (1.0 - weight)
                                + pixel * weight;
                        }
                        write((top + y) * width + left + x, pixel);
                    }
                }
            }
        }
    }
}

/// The original channel/frame output order and causal temporal fade, with reusable storage.
struct Video {
    tiling: Tiling,
    clips: usize,
    pushed: usize,
    frames: usize,
    values: Vec<f32>,
    fading: Vec<f32>,
    shape: [usize; 4],
}

impl Video {
    fn new(tiling: Tiling, clips: usize, frames: usize, tile: [usize; 4]) -> io::Result<Self> {
        let [channels, _, side, _] = tile;
        let height = tiling.tops[tiling.tops.len() - 1] + side;
        let width = tiling.lefts[tiling.lefts.len() - 1] + side;
        let plane = height * width;
        let values = zeroed_floats(channels * frames * plane)?;
        let fading = zeroed_floats(channels * tiling.fade.len() * plane)?;
        Ok(Self {
            tiling,
            clips,
            pushed: 0,
            frames,
            values,
            fading,
            shape: [channels, frames, height, width],
        })
    }

    fn push(&mut self, grid: &[Pixels]) {
        let [_, time, _, _] = grid[0].shape;
        let plane = self.shape[2] * self.shape[3];
        let tiling = &self.tiling;
        let (kept, fade) = (&tiling.kept, &tiling.fade);
        let skipped = self.clips * kept.len() + fade.len() - self.frames;
        let first = self.pushed * kept.len();
        let last = self.pushed + 1 == self.clips;
        let before = &self.fading;
        std::thread::scope(|scope| {
            for (index, output) in self.values.chunks_exact_mut(plane).enumerate() {
                let (channel, frame) = (index / self.frames, index % self.frames + skipped);
                let Some(offset) = frame.checked_sub(first) else {
                    continue;
                };
                let source = match offset.checked_sub(kept.len()) {
                    None => kept.start + offset,
                    Some(after) if last && after < fade.len() => fade.start + after,
                    Some(_) => continue,
                };
                let faded = (offset < fade.len() && self.pushed != 0).then(|| {
                    let weight = offset as f32 / fade.len() as f32;
                    (
                        &before[(channel * fade.len() + offset) * plane..][..plane],
                        weight,
                    )
                });
                scope.spawn(move || {
                    tiling.stitch(grid, channel * time + source, |at, pixel| {
                        let pixel = faded.map_or(pixel, |(before, weight)| {
                            before[at] * (1.0 - weight) + pixel * weight
                        });
                        output[at] = pixel;
                    });
                });
            }
        });
        // All previous fade readers finished above. Overwrite the single fading image only now.
        if !last {
            std::thread::scope(|scope| {
                for (index, values) in self.fading.chunks_exact_mut(plane).enumerate() {
                    let (channel, offset) = (index / fade.len(), index % fade.len());
                    scope.spawn(move || {
                        tiling.stitch(grid, channel * time + fade.start + offset, |at, pixel| {
                            values[at] = pixel;
                        });
                    });
                }
            });
        }
        self.pushed += 1;
    }
}

/// Packed input retains the ordinary decoder's four-chip, 2,048-row, 128-part mapping.
pub type Packed = HostTensor<bf16, m![{ Chips }, { m![Rows, Parts] }]>;

pub const LATENT_SHAPE: [usize; 5] = [1, 24, 37, 48, 84];
pub const OUTPUT_SHAPE: [usize; 5] = [1, 3, 124, 768, 1344];
pub const GROUPS: usize = CLIPS * SPATIAL_GROUPS;
/// Contiguous decoded `[4,2048,3072]` FP32 values, including each chip's padding rows.
pub const DECODED_VALUES: usize = 4 * Rows::SIZE * Patch::SIZE;

const CHANNELS: usize = 24;
const LATENT_TIME: usize = 37;
const LATENT_HEIGHT: usize = 48;
const LATENT_WIDTH: usize = 84;
const LATENT_VALUES: usize = CHANNELS * LATENT_TIME * LATENT_HEIGHT * LATENT_WIDTH;
const CLIPS: usize = 7;
const CLIP_STRIDE: usize = 5;
const SPATIAL_GROUPS: usize = 7;
const TILES: usize = 4 * SPATIAL_GROUPS;
const TILE_SHAPE: [usize; 4] = [3, 28, 256, 256];
const LATENT_TOPS: [usize; 4] = [0, 10, 21, 32];
const LATENT_LEFTS: [usize; 7] = [0, 11, 22, 33, 44, 56, 68];
const PIXEL_TOPS: [usize; 4] = [0, 160, 336, 512];
const PIXEL_LEFTS: [usize; 7] = [0, 176, 352, 528, 704, 896, 1088];

const _: () = {
    assert!(Chips::SIZE == 4 && Rows::SIZE == 2048 && Parts::SIZE == 128);
    assert!(TOKENS == 1792 && Patch::SIZE == 3072);
};

fn zeroed_floats(length: usize) -> io::Result<Vec<f32>> {
    bytemuck::allocation::try_zeroed_vec(length).map_err(|()| io::ErrorKind::OutOfMemory.into())
}

fn allocation(error: std::collections::TryReserveError) -> io::Error {
    io::Error::new(io::ErrorKind::OutOfMemory, error)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins latent channel order, low parts, four-chip tile order and finite input padding.
    #[test]
    fn preserves_video_latents() {
        let latent: Vec<f32> = (0..LATENT_VALUES)
            .map(|index| index as f32 * 0.001)
            .collect();
        assert!(Latents::new(&latent[..latent.len() - 1]).is_err());
        assert!(GroupIndex::try_from(GROUPS).is_err());
        let rows = Latents::new(&latent)
            .unwrap()
            .pack(GroupIndex::try_from(0).unwrap())
            .unwrap()
            .into_vec();
        assert_eq!(
            [0, 1, 25, 49, 2048 * 128].map(|index| rows[index].to_bits()),
            [0, 17173, 15932, 17173, 15412]
        );
        for chip in 0..4 {
            let padding = (chip * Rows::SIZE + TOKENS) * Parts::SIZE;
            let end = (chip + 1) * Rows::SIZE * Parts::SIZE;
            assert!(rows[padding..end].iter().all(|value| value.to_bits() == 0));
        }
    }

    /// Pins the decoder's channel/frame/spatial patch order, without a full-video allocation.
    #[test]
    fn preserves_pixel_order() {
        let source: Vec<f32> = (0..TOKENS * Patch::SIZE)
            .map(|index| index as f32)
            .collect();
        let mut pixels = Pixels::tile().unwrap();
        pixels.unpack(&source);
        assert_eq!(
            [0, 16, 256, 65536, 1835008].map(|index| pixels.values[index]),
            [0., 3072., 16., 256., 1024.]
        );
    }

    /// Pins above-then-before blending from original neighbors with the original small example.
    #[test]
    fn stitches_from_original_neighbors() {
        let grid = (0..4)
            .map(|tile| Pixels {
                values: (0..16).map(|pixel| (tile * 100 + pixel) as f32).collect(),
                shape: [1, 1, 4, 4],
            })
            .collect::<Vec<_>>();
        let tiling = Tiling {
            tops: &[0, 2],
            lefts: &[0, 2],
            kept: 0..1,
            fade: 1..1,
        };
        let mut video = Video::new(tiling, 1, 1, [1, 1, 4, 4]).unwrap();
        video.push(&grid);
        assert_eq!(
            video.values,
            [
                0., 1., 2., 52., 102., 103., 4., 5., 6., 56., 106., 107., 8., 9., 202., 156., 110.,
                111., 108., 109., 206., 208., 210., 211., 208., 209., 210., 260., 310., 311., 212.,
                213., 214., 264., 314., 315.,
            ]
        );
    }

    /// Pins the original causal fade and trim to the last frames on single-pixel frame planes.
    #[test]
    fn fades_clips_and_keeps_the_last_frames() {
        let clip = |values: [f32; 4]| {
            [Pixels {
                values: values.to_vec(),
                shape: [1, 4, 1, 1],
            }]
        };
        let tiling = || Tiling {
            tops: &[0],
            lefts: &[0],
            kept: 0..2,
            fade: 2..4,
        };
        let expected = [1., 2., 3., 12., 30., 40.];
        for frames in [1, 5, 6] {
            let mut video = Video::new(tiling(), 2, frames, [1, 4, 1, 1]).unwrap();
            video.push(&clip([1., 2., 3., 4.]));
            video.push(&clip([10., 20., 30., 40.]));
            assert_eq!(video.values, expected[expected.len() - frames..]);
        }
        for frames in [0, OUTPUT_SHAPE[2] + 1] {
            let error = Stitch::with_frames(frames).err().unwrap();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        }
    }
}
