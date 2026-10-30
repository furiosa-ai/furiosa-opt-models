//! LLaVA, by published size: a [`Size`] fixes what its checkpoint declares, and the catalog
//! composition below it replaces placeholder embeddings with selected image features before its
//! causal decoder.
//!
//! - Checkpoint scopes follow llava-hf/llava-1.5-7b-hf's model.safetensors.index.json: the image
//!   tower at `vision_tower`, the projector at `multi_modal_projector`, the causal language model
//!   at `language_model`.
//! - The size's language model is an attention decoder: the size is its [`Geometry`].
//! - Features are the tower's penultimate layer's patch rows, projected by a biased GELU pair.

use crate::modules::attention;
use crate::modules::{language as text, multimodal};
use crate::ops::attention::gqa::Read;
use crate::ops::embedding;
use crate::ops::index::{gather::Gather, table::Offset};
use crate::ops::rotary::Rotation;
use crate::tensor::mapping::Reshape;
use crate::{Module, Then};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::io;

use crate::models::Family;
use crate::models::decoder::attention::Geometry;
use crate::ops::attention::geometry::QueryGroups;
use crate::ops::normalization::RmsNorm;
use crate::tensor::cache::Blocks;
use crate::tensor::form::Weight;
use crate::tensor::memory::Memory;
pub use language::{Head, Heads, Hidden, LAYERS, Vocabulary, Windows};

pub mod language;
pub mod vision;

/// Image-conditioned causal decoder with a vocabulary projection.
pub type Model = text::CausalLm<
    Embedding,
    crate::Layers<[language::Layer; LAYERS]>,
    language::Norm,
    language::HeadProjection,
>;

impl<const N: usize> Module<&mut Context<'_, [(vision::Context, Offset); N]>, (), &mut ()>
    for Model
{
    type Output = HbmTensor<bf16, m![1], m![Vocabulary]>;

    fn forward(
        &self,
        device: &mut Device,
        _: (),
        _: &mut (),
        context: &mut Context<'_, [(vision::Context, Offset); N]>,
    ) -> Self::Output {
        let (caches, attention) = &mut context.states;
        let layers = caches.each_mut().map(|cache| {
            (
                (Default::default(), cache, Default::default()),
                Default::default(),
            )
        });
        let mut state = (((Default::default(), layers), ()), Default::default());
        self.forward(
            device,
            &*attention,
            &mut state,
            (&context.token, &mut context.images),
        )
    }
}

impl<const N: usize> Module<&mut Context<'_, [(vision::Rows, Offset); N]>, (), &mut ()> for Model {
    type Output = HbmTensor<bf16, m![1], m![Vocabulary]>;

    fn forward(
        &self,
        device: &mut Device,
        _: (),
        _: &mut (),
        context: &mut Context<'_, [(vision::Rows, Offset); N]>,
    ) -> Self::Output {
        let (caches, attention) = &mut context.states;
        let layers = caches.each_mut().map(|cache| {
            (
                (Default::default(), cache, Default::default()),
                Default::default(),
            )
        });
        // Encoded rows skip the image tower and its projector, so their slots are units.
        let mut state = (
            (
                ((Default::default(), (), (), Default::default()), layers),
                (),
            ),
            Default::default(),
        );
        self.forward(
            device,
            &*attention,
            &mut state,
            (&context.token, &context.images),
        )
    }
}

/// Token embeddings with selected image features replacing image placeholders.
pub type Embedding = multimodal::InputEmbedding<Tokens, vision::Encoder, vision::Projector, Merge>;

type Tokens = embedding::Embedding<Vocabulary, Hidden>;

type Merge = multimodal::Replace<Selection>;

type Selection = Then<Gather, Reshape<m![Hidden]>>;

/// Language head projection returning one vocabulary row.
pub type Projection = language::HeadProjection;

/// Each layer's writable KV cache, and the rotary, write and paged-read inputs every layer reads.
pub type States<'a> = (
    &'a mut [Blocks<Heads, Head, bf16, m![1]>; LAYERS],
    attention::Context<Rotation<Head>, HbmTensor<i32, m![1], m![1]>, Read<Windows>>,
);

/// One decoder row and the image feature replacements selected for that row.
#[derive(DeviceSend)]
pub struct Context<'a, Images> {
    /// Byte offset of the token embedding row.
    pub token: Offset,
    /// Image inputs or encoded feature rows paired with replacement offsets.
    pub images: Images,
    /// Writable language-layer caches and their attention operands.
    pub states: States<'a>,
}

/// A published LLaVA checkpoint size, by its hub spelling: `"1.5-7b"` is
/// llava-hf/llava-1.5-7b-hf.
pub struct Size<const N: &'static str>;

impl Geometry for Size<"1.5-7b"> {
    type Hidden = Hidden;
    type Intermediate = language::Intermediate;
    type Vocabulary = Vocabulary;
    type KvHeads = Heads;
    type Groups = QueryGroups<1>;
    type Head = Head;
    type Queries = language::QueryWidth;
    type Layers<L> = [L; LAYERS];
    type Norm<W: AxisName, X: Memory> = RmsNorm<
        W,
        { language::EPSILON.to_bits() as usize },
        m![1],
        Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>,
    >;

    const SIZE: &'static str = "1.5-7b";
    const FAMILY: &'static str = "llava";
    const LAYERS: usize = LAYERS;
    const EPSILON: usize = language::EPSILON.to_bits() as usize;
    const TIED_EMBEDDINGS: bool = false;
}

/// The image placeholder's token, `image_token_index`.
const IMAGE_TOKEN: i64 = 32000;

/// The tower layer whose rows the projector reads, `vision_feature_layer`: the penultimate.
const FEATURE_LAYER: i64 = -2;

/// What a LLaVA checkpoint's config declares that its size fixes, and the features the size does
/// not serve; an absent key holds `LlavaConfig`'s default.
#[derive(Deserialize)]
pub struct Config {
    model_type: String,
    image_token_index: Option<i64>,
    vision_feature_layer: Option<i64>,
    vision_feature_select_strategy: Option<String>,
    projector_hidden_act: Option<String>,
    multimodal_projector_bias: Option<bool>,
    tie_word_embeddings: Option<bool>,
    text_config: language::Config,
    vision_config: vision::Config,
}

/// A config is of a size when its root, its text half and its image half each declare the size's
/// values; a nested key is named by its dotted path.
impl<const N: &'static str> TryFrom<Config> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let nested =
            |key: &str, error: io::Error| io::Error::new(error.kind(), format!("{key}.{error}"));
        if config.model_type != "llava" {
            return Err(invalid(format!(
                "model_type: {}, the size {N} is llava",
                config.model_type
            )));
        }
        let image = config.image_token_index.unwrap_or(IMAGE_TOKEN);
        if image != IMAGE_TOKEN {
            return Err(invalid(format!(
                "image_token_index: {image}, the size {N} holds {IMAGE_TOKEN}"
            )));
        }
        let layer = config.vision_feature_layer.unwrap_or(FEATURE_LAYER);
        if layer != FEATURE_LAYER {
            return Err(invalid(format!(
                "vision_feature_layer: {layer}, the size {N} reads {FEATURE_LAYER}"
            )));
        }
        if !config.multimodal_projector_bias.unwrap_or(true) {
            return Err(invalid(
                "multimodal_projector_bias: false, the size's projector has biases".into(),
            ));
        }
        if config.tie_word_embeddings.unwrap_or(false) {
            return Err(invalid(format!(
                "tie_word_embeddings: true, the size {N} ties: false"
            )));
        }
        match config.vision_feature_select_strategy.as_deref() {
            None | Some("default") => {}
            Some(_) => {
                return Err(unsupported(
                    "vision_feature_select_strategy: the size drops the class row",
                ));
            }
        }
        match config.projector_hidden_act.as_deref() {
            None | Some("gelu") => {}
            Some(_) => return Err(unsupported("projector_hidden_act: the projector is GELU")),
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

impl<const N: &'static str> Family for Size<N>
where
    Self: Geometry,
{
    type Hf = Config;
}
