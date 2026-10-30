//! CLIP patch features at the configured layer, followed by the learned image projector.

use super::language::Hidden;
use crate::Then;
use crate::models::encoder::vision;
use crate::modules::{feedforward::Mlp, language::SelectRow};
use crate::ops::{
    index::gather::Gather,
    linear::Affine,
    pointwise::{activation::Gelu, cast::Convert as Cast},
};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::io;

/// Selected encoder depth with the class-token row removed.
pub type Encoder = Then<Body, SelectRow<Gather, Offsets>>;

type Body =
    vision::ClipBody<Channels, Height, Width, Patch, Embedding, Intermediate, Heads, SELECTED>;

type Offsets = HbmTensor<i32, m![1], m![Patches]>;

/// Two-layer GELU projector from image width to BF16 language features.
pub type Projector = Then<Mlp<Linear<Embedding, Hidden>, Gelu, Linear<Hidden, Hidden>>, Cast<bf16>>;

type Linear<I, O> = Affine<HbmTensor<f32, m![1], m![O, I]>, [HbmTensor<f32, m![1], m![O]>; 1]>;

/// RGB image in channel, row, column order.
pub type Pixels = vision::Pixels<Channels, Height, Width>;

/// Spatial patch count, excluding the class token.
pub type Patches = vision::Patches<Height, Width, Patch>;

/// Encoder rows, including the leading class token.
pub type Tokens = vision::Tokens<Height, Width, Patch>;

/// Selected image patch features before projection into language width.
pub type Features = vision::Rows<Embedding, Patches>;

/// Pixels and writable captures for the selected encoder depth.
pub type Context = vision::Context<Pixels, Embedding, Tokens, SELECTED>;

/// Projected image patch rows used to replace language placeholders.
pub type Rows = HbmTensor<bf16, m![1], m![Patches, Hidden]>;

/// Total layer count of the checkpoint's image encoder.
pub const LAYERS: usize = 24;

/// Layer count yielding hidden-state index -2, before the final encoder layer.
pub const SELECTED: usize = LAYERS - 1;

axes![
    Channels = 3,
    Height = 336,
    Width = 336,
    Patch = 14,
    Embedding = 1024,
    Intermediate = 4096,
    Heads = 16,
];

/// `layer_norm_eps` of the tower's norms.
const EPSILON: f32 = 1e-5;

/// What the image tower's `vision_config` declares that the size fixes, and the features the
/// tower does not serve; an absent key holds `CLIPVisionConfig`'s default.
#[derive(Deserialize)]
#[serde(default)]
pub struct Config {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_channels: usize,
    image_size: usize,
    patch_size: usize,
    layer_norm_eps: f64,
    hidden_act: String,
}

/// `CLIPVisionConfig`'s defaults, the model type a LLaVA config's image half names when it names
/// none.
impl Default for Config {
    fn default() -> Self {
        Self {
            model_type: "clip_vision_model".into(),
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 224,
            patch_size: 32,
            layer_norm_eps: 1e-5,
            hidden_act: "quick_gelu".into(),
        }
    }
}

impl Config {
    /// Refuses a vision config the tower does not serve: a value it does not hold is invalid, an
    /// activation it does not have unsupported.
    pub fn check(&self) -> io::Result<()> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        if self.model_type != "clip_vision_model" {
            return Err(invalid(format!(
                "model_type: {}, the tower is clip_vision_model",
                self.model_type
            )));
        }
        let fixed = [
            ("hidden_size", self.hidden_size, Embedding::SIZE),
            (
                "intermediate_size",
                self.intermediate_size,
                Intermediate::SIZE,
            ),
            ("num_hidden_layers", self.num_hidden_layers, LAYERS),
            ("num_attention_heads", self.num_attention_heads, Heads::SIZE),
            ("num_channels", self.num_channels, Channels::SIZE),
            ("image_size", self.image_size, Height::SIZE),
            ("patch_size", self.patch_size, Patch::SIZE),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(invalid(format!(
                "{key}: {declared}, the tower holds {held}"
            )));
        }
        if self.layer_norm_eps as f32 != EPSILON {
            return Err(invalid(format!(
                "layer_norm_eps: {}, the tower holds {EPSILON}",
                self.layer_norm_eps
            )));
        }
        match self.hidden_act.as_str() {
            "quick_gelu" => Ok(()),
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "hidden_act: the tower's feed-forward is quick GELU",
            )),
        }
    }
}
