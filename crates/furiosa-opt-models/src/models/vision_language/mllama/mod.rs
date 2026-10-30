//! Mllama, by published size: a [`Size`] fixes what its checkpoint declares, and the catalog
//! composition below it encodes image tiles and decodes text whose gated cross-attention layers
//! read them.
//!
//! - Checkpoint: unsloth/Llama-3.2-11B-Vision-Instruct, 677b0c1b7008230a0fb88708c5550748e72b9a83.
//!   The text model sits at `language_model`, the tiled image tower at `vision_model`, the
//!   projector at `multi_modal_projector`.
//! - The text model's token table has eight rows past its vocabulary, one an image token.

use crate::models::Family;
use crate::models::config::{
    self, Positive,
    scaling::{Defaults, Spectrum},
};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::io;

use crate::modules::{language, multimodal, residual, transformer};
use crate::ops::pointwise::activation::Tanh;
use crate::{Branch, Interleave};
pub use text::Model;

pub mod text;
pub mod vision;

impl<Q: AxisName, const N: usize> crate::Module<&mut Context<Q, N>, (), &mut ()> for Model {
    type Output = text::Scores<Q>;

    fn forward(
        &self,
        device: &mut Device,
        env: (),
        _: &mut (),
        context: &mut Context<Q, N>,
    ) -> Self::Output {
        if N > 0 {
            #[unroll]
            for index in 0..N {
                encoder::Encoder {
                    vision: &self.encoder,
                    text: self.decoder.body(),
                }
                .forward(
                    &mut *device,
                    env,
                    &mut (),
                    (&mut context.images[index], &mut context.text.memory.caches),
                );
            }
        }
        let hidden = self
            .decoder
            .body()
            .forward(&mut *device, env, &mut (), &mut context.text);
        self.decoder
            .head()
            .forward(device, env, &mut Default::default(), &hidden)
    }
}

pub mod encoder;

/// Image chunks to encode and text rows to decode against their cached features.
#[derive(DeviceSend)]
pub struct Context<Q: AxisName, const IMAGES: usize> {
    /// Image inputs and cross-attention cache destinations.
    pub images: [encoder::Input; IMAGES],
    /// Decoder tokens, writable self-attention state, and cached image memory.
    pub text: text::Input<Q>,
}

axes![
    Channels = 3,
    Height = 560,
    Width = 560,
    PatchHeight = 14,
    PatchWidth = 14,
    VisionWidth = 1280,
    VisionIntermediate = 5120,
    VisionHeads = 16,
    Tiles = 4,
    AspectRatios = 9,
];

/// Per-tile token count: 1601 class/patch rows rounded up to eight-row alignment.
#[derive(Clone, Debug)]
pub struct Padded;

impl AxisName for Padded {
    const NAME: Ident = Ident::new("PaddedPatches");
    const SIZE: usize = 1608;
}

axes![
    Hidden = 4096,
    KvWidth = 1024,
    Head = 128,
    Heads = 32,
    Intermediate = 14336,
    Embeddings = 128264,
    Vocabulary = 128256,
];

/// A published Mllama checkpoint size, by its hub spelling: `"11B"` is
/// Llama-3.2-11B-Vision-Instruct.
pub struct Size<const N: &'static str>;

/// The image placeholder's token, `image_token_index`: the first row past the vocabulary.
const IMAGE_TOKEN: usize = Vocabulary::SIZE;

/// Text decoder layers, `num_hidden_layers`, the cross-attention layers among them.
const TEXT_LAYERS: usize = 40;

/// Local image tower layers, `num_hidden_layers`.
const VISION_LAYERS: usize = 32;

/// Global image tower layers, `num_global_layers`.
const GLOBAL_LAYERS: usize = 8;

/// `rms_norm_eps` of the text model and `norm_eps` of the image tower.
const EPSILON: f32 = 1e-5;

/// `max_position_embeddings`: the positions the text model serves.
const POSITIONS: usize = 131072;

/// The tile shapes, as rows by columns of tiles, `supported_aspect_ratios`.
const ASPECT_RATIOS: [[usize; 2]; 8] = [
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [2, 1],
    [2, 2],
    [3, 1],
    [4, 1],
];

/// What an Mllama checkpoint's config declares that its size fixes, and the features the size
/// does not serve; an absent key holds `MllamaConfig`'s default.
#[derive(Deserialize)]
pub struct Config {
    model_type: String,
    image_token_index: Option<usize>,
    text_config: Text,
    vision_config: Vision,
}

/// The text model's `text_config`; an absent key holds `MllamaTextConfig`'s default.
#[derive(Deserialize)]
#[serde(default)]
pub struct Text {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    cross_attention_layers: Vec<usize>,
    rms_norm_eps: f64,
    #[serde(flatten, deserialize_with = "Text::spectrum")]
    rope: Spectrum,
    max_position_embeddings: usize,
    hidden_act: String,
    tie_word_embeddings: bool,
}

/// `MllamaTextConfig`'s: its `default_theta`.
impl Defaults for Text {
    const THETA: Positive = Positive::new(500000.0);
}

/// The image tower's `vision_config`; an absent key holds `MllamaVisionConfig`'s default.
#[derive(Deserialize)]
#[serde(default)]
pub struct Vision {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    attention_heads: usize,
    num_hidden_layers: usize,
    num_global_layers: usize,
    num_channels: usize,
    image_size: usize,
    patch_size: usize,
    max_num_tiles: usize,
    vision_output_dim: usize,
    norm_eps: f64,
    intermediate_layers_indices: Vec<usize>,
    supported_aspect_ratios: Vec<[usize; 2]>,
    hidden_act: String,
}

/// `MllamaTextConfig`'s defaults, the model type an Mllama config's text half names when it names
/// none.
impl Default for Text {
    fn default() -> Self {
        Self {
            model_type: "mllama_text_model".into(),
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 40,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            vocab_size: 128256,
            cross_attention_layers: text::CROSS.to_vec(),
            rms_norm_eps: 1e-5,
            rope: Self::spectrum(serde_json::Value::Object(Default::default()))
                .expect("a configuration naming no rotary field rotates by its class's base"),
            max_position_embeddings: 131072,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
        }
    }
}

/// `MllamaVisionConfig`'s defaults, the model type an Mllama config's image half names when it
/// names none.
impl Default for Vision {
    fn default() -> Self {
        Self {
            model_type: "mllama_vision_model".into(),
            hidden_size: 1280,
            intermediate_size: 5120,
            attention_heads: 16,
            num_hidden_layers: 32,
            num_global_layers: 8,
            num_channels: 3,
            image_size: 448,
            patch_size: 14,
            max_num_tiles: 4,
            vision_output_dim: 7680,
            norm_eps: 1e-5,
            intermediate_layers_indices: vision::CAPTURES.to_vec(),
            supported_aspect_ratios: ASPECT_RATIOS.to_vec(),
            hidden_act: "gelu".into(),
        }
    }
}

impl Text {
    /// Refuses a text config the size does not serve: a value it does not hold is invalid, a
    /// feature it does not have unsupported.
    fn check(&self) -> io::Result<()> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        if self.model_type != "mllama_text_model" {
            return Err(invalid(format!(
                "model_type: {}, the text model is mllama_text_model",
                self.model_type
            )));
        }
        let fixed = [
            ("hidden_size", self.hidden_size, Hidden::SIZE),
            (
                "intermediate_size",
                self.intermediate_size,
                Intermediate::SIZE,
            ),
            ("num_hidden_layers", self.num_hidden_layers, TEXT_LAYERS),
            ("num_attention_heads", self.num_attention_heads, Heads::SIZE),
            (
                "num_key_value_heads",
                self.num_key_value_heads,
                KvWidth::SIZE / Head::SIZE,
            ),
            ("vocab_size", self.vocab_size, Vocabulary::SIZE),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(invalid(format!("{key}: {declared}, the size holds {held}")));
        }
        if self.cross_attention_layers != text::CROSS {
            return Err(invalid(format!(
                "cross_attention_layers: {:?}, the size holds {:?}",
                self.cross_attention_layers,
                text::CROSS
            )));
        }
        if self.rms_norm_eps as f32 != EPSILON {
            return Err(invalid(format!(
                "rms_norm_eps: {}, the size holds {EPSILON}",
                self.rms_norm_eps
            )));
        }
        if self.tie_word_embeddings {
            return Err(invalid(
                "tie_word_embeddings: true, the text model's head is its own".into(),
            ));
        }
        if self.max_position_embeddings > POSITIONS {
            return Err(unsupported(
                "max_position_embeddings: past the text model's 131072 positions",
            ));
        }
        if self.hidden_act != "silu" {
            return Err(unsupported(
                "hidden_act: the text model's feed-forward is SiLU",
            ));
        }
        // The text model's rotation holds each pair's frequency and no amplitude.
        self.rope.bind::<Head, Head>()?;
        config::compiled("attention_factor", self.rope.amplitude(), 1.0).map(drop)
    }
}

impl Vision {
    /// Refuses a vision config the size does not serve: a value it does not hold is invalid, an
    /// activation it does not have unsupported.
    fn check(&self) -> io::Result<()> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        if self.model_type != "mllama_vision_model" {
            return Err(invalid(format!(
                "model_type: {}, the tower is mllama_vision_model",
                self.model_type
            )));
        }
        let fixed = [
            ("hidden_size", self.hidden_size, VisionWidth::SIZE),
            (
                "intermediate_size",
                self.intermediate_size,
                VisionIntermediate::SIZE,
            ),
            ("attention_heads", self.attention_heads, VisionHeads::SIZE),
            ("num_hidden_layers", self.num_hidden_layers, VISION_LAYERS),
            ("num_global_layers", self.num_global_layers, GLOBAL_LAYERS),
            ("num_channels", self.num_channels, Channels::SIZE),
            ("image_size", self.image_size, Height::SIZE),
            ("patch_size", self.patch_size, PatchHeight::SIZE),
            ("max_num_tiles", self.max_num_tiles, Tiles::SIZE),
            (
                "vision_output_dim",
                self.vision_output_dim,
                VisionWidth::SIZE * (vision::CAPTURES.len() + 1),
            ),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(invalid(format!("{key}: {declared}, the size holds {held}")));
        }
        if self.norm_eps as f32 != EPSILON {
            return Err(invalid(format!(
                "norm_eps: {}, the size holds {EPSILON}",
                self.norm_eps
            )));
        }
        if self.intermediate_layers_indices != vision::CAPTURES {
            return Err(invalid(format!(
                "intermediate_layers_indices: {:?}, the size holds {:?}",
                self.intermediate_layers_indices,
                vision::CAPTURES
            )));
        }
        if self.supported_aspect_ratios != ASPECT_RATIOS {
            return Err(invalid(format!(
                "supported_aspect_ratios: {:?}, the size holds {ASPECT_RATIOS:?}",
                self.supported_aspect_ratios
            )));
        }
        match self.hidden_act.as_str() {
            "gelu" => Ok(()),
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "hidden_act: the tower's feed-forward is GELU",
            )),
        }
    }
}

/// A config is of the size when its root, its text half and its image half each declare the
/// size's values; a nested key is named by its dotted path.
impl TryFrom<Config> for Size<"11B"> {
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let nested =
            |key: &str, error: io::Error| io::Error::new(error.kind(), format!("{key}.{error}"));
        if config.model_type != "mllama" {
            return Err(invalid(format!(
                "model_type: {}, the size 11B is mllama",
                config.model_type
            )));
        }
        let image = config.image_token_index.unwrap_or(IMAGE_TOKEN);
        if image != IMAGE_TOKEN {
            return Err(invalid(format!(
                "image_token_index: {image}, the size 11B holds {IMAGE_TOKEN}"
            )));
        }
        config
            .text_config
            .check()
            .map_err(|error| nested("text_config", error))?;
        config
            .vision_config
            .check()
            .map_err(|error| nested("vision_config", error))?;
        Ok(Size)
    }
}

impl Family for Size<"11B"> {
    type Hf = Config;
}

pub(crate) type CrossLanguage<
    Vision,
    Embedding,
    Attention,
    Cross,
    Mlp,
    Norm,
    Gate,
    Mask,
    Add,
    Head,
    const TEXT_LAYERS: usize,
    const CROSS_LAYERS: usize,
    const ORDER: &'static [Branch],
> = multimodal::EncoderDecoder<
    Vision,
    language::Readout<
        language::Normalized<
            language::Backbone<
                Embedding,
                Interleave<
                    (
                        [Attention; TEXT_LAYERS],
                        [GatedCross<Cross, Mlp, Norm, Gate, Mask, Add>; CROSS_LAYERS],
                    ),
                    ORDER,
                >,
            >,
            Norm,
        >,
        Head,
    >,
>;

pub(crate) type GatedCross<Attention, Mlp, Norm, Gate, Mask, Add> =
    transformer::PreNorm<Attention, Mlp, Norm, residual::Gated<Gate, Tanh, Mask, Add>>;
