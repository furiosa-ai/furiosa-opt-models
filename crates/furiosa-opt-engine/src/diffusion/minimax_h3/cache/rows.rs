//! TeaCache row-operation types over the request's initialized prefix.
//!
//! The caller retains operands, metadata and outputs until all readers finish, including
//! after cancellation or submission failure, and supplies writable copy state.

pub(super) use furiosa_opt_models::models::diffusion::minimax_h3::denoiser::cache::{
    Add, Copy, Subtract,
};
