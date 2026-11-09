//! Attention of a prompt chunk whose rows see at most 2,048 keys each, in passes of 1,024 rows:
//! a pass's eight 128-row tiles each hold their own 2,048 keys on 32 slices, so a short history
//! fills the slices with rows instead of absent keys.
//!
//! - A cluster owns two KV heads, attended one after the other; tile `t` of a pass is the chunk's
//!   rows `128 * t..` of the pass, and every tile lands the chunk's keys.
//! - Softmax peaks are all-reduced over a tile's 32 slices; the value product and the mass are
//!   reduced onto the tile's query slices, where the product is divided by the mass.
use super::*;
use prompt::Prompt;

axes![Keys = 2048, Queries = 1024];

/// Byte offset of each of the chunk's 128 key pages within its layer's pool pages.
pub type Pages = HbmTensor<i32, Chip, m![Keys / 16]>;

/// The short path of one step's prompt chunk: the rows `start..start + rows` of the prompt plan.
#[derive(DeviceSend)]
pub struct Plan {
    /// The chunk's 1,024-row passes, 0 when the chunk attends span after span instead.
    pub passes: Count,
    /// The chunk's key pages, the same for every tile.
    pub pages: Pages,
    /// The last key of its tile's keys that chunk row `r` sees, in every lane.
    pub limits: HbmTensor<f32, Chip, m![Prompt, Lane]>,
    /// Each key's index within its tile's keys.
    pub positions: HbmTensor<f32, Chip, m![Keys]>,
}
