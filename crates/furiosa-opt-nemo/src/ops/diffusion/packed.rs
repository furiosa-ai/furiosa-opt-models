//! Text, audio and video rows packed into one sequence, and the row operations whose modulation
//! each row's modality and timestep select.
//!
//! - The spans are text, audio, video, in that order, each a whole number of 64-row tiles (16-row
//!   ones in a chip's part of a sequence split over chips), so a tile never mixes modalities.
//! - A modulation table holds one row per timestep and modality, row `timestep * 3 + modality`
//!   (video 0, text 1, audio 2). Text and video rows take the video timestep 0, audio rows the
//!   audio timestep 1, so a span's table row is a constant.
//! - A tile spreads its features over the slices and its two halves over the clusters; its row
//!   statistics reduce across slices, and per-feature operands stay in the slices that use them.

use furiosa_opt_std::prelude::*;

mod blocks;
mod rows;

axes![
    Strip = 64,
    Group = 6,
    Timestep = 2,
    Modality = 3,
    Chunks = 6
];

axes![Part = 2, Frequency = 128];
