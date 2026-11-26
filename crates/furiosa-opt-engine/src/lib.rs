//! Load `furiosa-opt-models` weight trees onto the device and run them for a serving framework.
//!
//! - `Checkpoint` maps a checkpoint directory; a weight tree loads from it into HBM tensor by
//!   tensor.
//! - The `registry` lists every upstream architecture name with the decoders and media encoders
//!   that implement it. `DecoderLoadOptions` loads a decoder; `EncoderLoadOptions` loads a media
//!   encoder.
//!   A choice the registry refuses is reported before any weight is read.
//! - A `Function` runs a module of a loaded tree through a compiled entry, as an `AsyncModule`.
//! - A `decoder` root serves requests on a `decoder::Cache`: a call validates and plans them, then
//!   launches on the views the decoder keeps of the cache.
//! - An `encoder` turns images, video frames and audio into rows of its language
//!   model's width.
//! - The package `furiosa-opt-python` binds this API to Python, as the extension module of the
//!   `furiosa_opt_engine` wheel.
//! - Scheduling, cache block assignment and sampling stay in the framework.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

mod function;

pub mod decoder;
pub mod encoder;
pub mod load;
pub mod registry;

pub use function::{AsyncModule, Function, forward};
pub use furiosa_opt_models::load::{Checkpoint, Load, LoadError};
pub use furiosa_opt_models::ops::attention::full::paged::Tiling;
pub use furiosa_opt_models::tensor::memory::Hbm;
pub use load::{Config, DecoderLoadOptions, EncoderLoadOptions, Quantization};
