pub mod residual;

use furiosa_opt_std::prelude::*;
pub use mlp::{Hidden, Rows};
pub use residual::Chips;
axes![Width = 2048];
/// Hidden states between pieces: each 80-wide hidden group keeps a wave's 64 rows contiguous, so
/// a slice that owns 8 rows of one group moves one 1,280-byte run.
pub type Blocked = m![Hidden / 80, Rows, Hidden % 80];

pub mod attention;
pub mod head;
pub mod mlp;
pub mod norm;
pub mod output;
pub mod qkv;
pub mod scale;
pub mod stack;
pub mod step;
