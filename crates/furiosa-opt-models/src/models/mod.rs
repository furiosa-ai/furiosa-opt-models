//! Sized model compositions of upstream architectures; the engine's registry lists their names.
//!
//! - Sized families own their compositions and configuration parsing. The engine loads their
//!   weights.
//! - Catalog membership follows the pinned upstream registries and implementations below.
//! - An item's `Source:` names a file of its project at the revision linked below.
//!
//! - [vLLM registry](https://github.com/vllm-project/vllm/blob/153242a314153637999eb6ebe8dd830e63433bb6/vllm/model_executor/models/registry.py)
//! - [SGLang registry](https://github.com/sgl-project/sglang/blob/81cb895b85c1ac3361bfd9a241f362b48b75301f/python/sglang/srt/models/registry.py)
//! - [SGLang generation components](https://github.com/sgl-project/sglang/blob/81cb895b85c1ac3361bfd9a241f362b48b75301f/python/sglang/multimodal_gen/runtime/models/registry.py)
//! - [vLLM-Omni models](https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/registry.py)
//! - [vLLM-Omni diffusion](https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/diffusion/registry.py)
//!
//! - Registered Transformers architectures use their concrete implementation at
//!   [this revision](https://github.com/huggingface/transformers/tree/f324707307757d9c0b8dac1c4462eceff911fa2f/src/transformers/models).
//! - Diffusion wiring follows
//!   [Diffusers](https://github.com/huggingface/diffusers/tree/e0118ade2f60234c41bacf40330a7e2f61108849/src/diffusers/models).

#![allow(non_camel_case_types)]
#![deny(clippy::arbitrary_source_item_ordering)]

use serde::de::DeserializeOwned;

use crate::load::{Checkpoint, LoadError};
use furiosa_opt_std::prelude::{Broadcast, HbmTensor, Symbol, m};
use std::io;

// Declarations and configuration parsing.
pub mod config;
pub mod decoder;

// Sized model families.
pub mod diffusion;
pub mod encoder;
pub mod encoder_decoder;
pub mod vision_language;

/// One FP32 value: a norm's epsilon, an attention scale or a learned gate.
pub type Scalar = HbmTensor<f32, m![1], m![1]>;

/// One signed token, position or segment index per row of `Count`.
pub type Ids<Count> = HbmTensor<i32, m![1], m![Count]>;

/// Checkpoints that one composition serves, told apart only by their configuration.
pub trait Family: Sized {
    /// The checkpoint configuration's schema, which `try_into` narrows to this family.
    type Hf: DeserializeOwned + TryInto<Self, Error = io::Error>;

    /// Parse a checkpoint configuration, refusing values the compiled composition cannot hold.
    fn parse(document: &serde_json::Value) -> io::Result<Self> {
        config::parse::<Self::Hf>(document)?.try_into()
    }
}

/// The class of a type that implements an architecture: its family's module and its name there,
/// such as `qwen3` and `ForCausalLM`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Class {
    pub family: &'static str,
    pub name: &'static str,
}

/// A type that implements an upstream architecture at one size.
pub trait Implements {
    /// The size, as the hub spells it.
    const SIZE: &'static str;

    /// The type's class, which every size of the type shares within a family.
    const CLASS: Class;

    /// The scope a checkpoint is read at, or why the type does not serve it.
    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError>;
}
