//! CLIP and SigLIP retrieval: independent normalized encoders followed by pairwise logits.
use super::{clip, vision};
use crate::models::Family;
use crate::models::config::flattened;
use crate::modules::contrastive::{DotProduct, Factor};
use crate::modules::language::{FeatureHead, Readout, SelectRow};
use crate::ops::{
    index::{gather::Gather, table::Offset},
    linear::Coefficients,
    normalization::Normalize,
    pointwise::activation::NewGelu,
};
use crate::{Each, Then};
use furiosa_opt_std::prelude::*;
use serde::{Deserialize, de::IgnoredAny};
use std::{collections::BTreeMap, io};

/// Independent text and image encoders followed by pairwise scaled logits plus `Bias`.
pub type Model<Text, Image, Bias> = FeatureHead<(Each<Text>, Each<Image>), DotProduct<Bias>>;

/// Retrieval composition selected from the CLIP or SigLIP checkpoint family.
pub enum Loaded<
    V: AxisName,
    P: AxisName,
    W: AxisName,
    I: AxisName,
    H: AxisName,
    C: AxisName,
    Y: AxisName,
    X: AxisName,
    K: AxisName,
    D: AxisName,
    J: AxisName,
    A: AxisName,
    O: AxisName,
    const TEXT_LAYERS: usize,
    const IMAGE_LAYERS: usize,
> {
    /// Projected encoders with scaled pairwise logits and no logit bias.
    Clip(Clip<V, P, W, I, H, C, Y, X, K, D, J, A, O, TEXT_LAYERS, IMAGE_LAYERS>),
    /// Attention-pooled image encoder with scaled and biased pairwise logits.
    Siglip(Siglip<V, P, W, I, H, C, Y, X, K, D, J, A, TEXT_LAYERS, IMAGE_LAYERS>),
}

/// Source: vLLM `vllm/model_executor/models/clip.py`.
pub type Clip<
    V,
    P,
    W,
    I,
    H,
    C,
    Y,
    X,
    K,
    D,
    J,
    A,
    O,
    const TEXT_LAYERS: usize,
    const IMAGE_LAYERS: usize,
> = Model<
    ClipText<V, P, W, I, H, O, TEXT_LAYERS>,
    ClipImage<C, Y, X, K, D, J, A, O, IMAGE_LAYERS>,
    (),
>;

/// Source: vLLM `vllm/model_executor/models/siglip.py`.
pub type Siglip<
    V,
    P,
    W,
    I,
    H,
    C,
    Y,
    X,
    K,
    D,
    J,
    A,
    const TEXT_LAYERS: usize,
    const IMAGE_LAYERS: usize,
> = Model<
    SiglipText<V, P, W, I, H, D, TEXT_LAYERS>,
    SiglipImage<C, Y, X, K, D, J, A, IMAGE_LAYERS>,
    Factor,
>;

/// Pooled, projected CLIP text features normalized per input.
pub type ClipText<V, P, W, I, H, O, const N: usize> =
    Readout<clip::Encoder<clip::Text<V, P, W, I, H, N>, Linear<W, O, 0>>, Normalize>;

pub type Linear<I, O, const B: usize> = Coefficients<f32, I, O, B>;

/// CLIP class-token features projected and normalized per image.
pub type ClipImage<C, H, W, P, D, I, A, O, const N: usize> = Readout<
    Readout<vision::ClipBody<C, H, W, P, D, I, A, N>, Then<vision::ClipHead<D>, Linear<D, O, 0>>>,
    Normalize,
>;

/// SigLIP's final text row projected and normalized per input.
pub type SiglipText<V, P, W, I, H, O, const N: usize> = Readout<
    Readout<
        clip::Text<V, P, W, I, H, N, NewGelu>,
        Then<SelectRow<Gather, Offset>, Linear<W, O, 1>>,
    >,
    Normalize,
>;

/// Attention-pooled SigLIP image features normalized per image.
pub type SiglipImage<C, H, W, P, D, I, A, const N: usize> = Readout<
    Readout<vision::SiglipBody<C, H, W, P, D, I, A, N>, vision::SiglipHead<D, I, A>>,
    Normalize,
>;

/// CLIP text operands and the byte offset of the caller-selected pooling row.
pub type ClipInput<W, P, const N: usize> = (clip::Operands<W, P, N>, Offset);

/// SigLIP text operands; its fixed final-row pooling needs no caller offset.
pub type SiglipInput<W, P, const N: usize> = clip::Operands<W, P, N>;

/// Text and image tower configurations with their shared retrieval family.
#[derive(Clone)]
pub struct Config {
    /// Checkpoint family selecting both towers and the logit head.
    pub family: vision::Family,
    /// Text tower configuration.
    pub text: clip::Config,
    /// Image tower configuration.
    pub image: vision::Config,
    /// SigLIP text projection width, required to match the image output width.
    pub text_projection: usize,
}

impl Family for Config {
    type Hf = Hf;
}

/// A full contrastive checkpoint's configuration: its family and both towers.
#[derive(Deserialize)]
pub struct Hf {
    #[serde(flatten, deserialize_with = "flattened")]
    pair: Pair,
    #[serde(flatten, deserialize_with = "flattened")]
    text: clip::Hf,
    #[serde(flatten, deserialize_with = "flattened")]
    image: vision::Hf,
}

/// The fields of a dual encoder's checkpoint configuration that pair its towers.
#[derive(Deserialize)]
struct Pair {
    model_type: Option<String>,
    text_config: Option<Tower>,
    vision_config: Option<BTreeMap<String, IgnoredAny>>,
}

/// The text tower's output width: its projection, else its hidden width.
#[derive(Deserialize)]
struct Tower {
    projection_size: Option<usize>,
    hidden_size: Option<usize>,
}

impl TryFrom<Hf> for Config {
    type Error = io::Error;

    /// Narrows both towers of a full contrastive checkpoint configuration.
    fn try_from(Hf { pair, text, image }: Hf) -> io::Result<Self> {
        let family = match pair.model_type.as_deref() {
            Some("clip") => vision::Family::Clip,
            Some("siglip") => vision::Family::Siglip,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "expected full CLIP or SigLIP configuration",
                ));
            }
        };
        let (Some(tower), Some(_)) = (pair.text_config, pair.vision_config) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing text_config or vision_config",
            ));
        };
        let text_projection = Some(tower.projection_size.or(tower.hidden_size).unwrap_or(768))
            .filter(|&size| size > 0)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "projection_size must be positive",
                )
            })?;
        Ok(Self {
            family,
            text: text.try_into()?,
            image: image.try_into()?,
            text_projection,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// A full CLIP config of both towers at their defaults.
    fn document() -> Value {
        json!({
            "architectures": ["CLIPModel"], "model_type": "clip",
            "text_config": {"model_type": "clip_text_model"},
            "vision_config": {"model_type": "clip_vision_model"},
        })
    }

    /// A full CLIP or SigLIP config of both towers and a positive text output width parses;
    /// another model type, a missing tower or no output width is refused by its reason.
    #[test]
    fn refuses_what_the_pair_does_not_hold() {
        let parse = |edit: &dyn Fn(&mut Value)| {
            let mut document = document();
            edit(&mut document);
            <Config as Family>::parse(&document).map(drop)
        };
        parse(&|_| ()).unwrap();
        use io::ErrorKind::{InvalidData, Unsupported};
        let missing = "missing text_config or vision_config";
        let width = "projection_size must be positive";
        type Edit<'a> = &'a dyn Fn(&mut Value);
        let cases: [(Edit, io::ErrorKind, &str); 5] = [
            (
                &|document| document["model_type"] = json!("align"),
                Unsupported,
                "expected full CLIP or SigLIP configuration",
            ),
            (
                &|document| drop(document.as_object_mut().unwrap().remove("text_config")),
                InvalidData,
                missing,
            ),
            (
                &|document| drop(document.as_object_mut().unwrap().remove("vision_config")),
                InvalidData,
                missing,
            ),
            (
                &|document| document["text_config"]["projection_size"] = json!(0),
                InvalidData,
                width,
            ),
            (
                &|document| document["text_config"]["hidden_size"] = json!(0),
                InvalidData,
                width,
            ),
        ];
        for (edit, kind, reason) in cases {
            let error = parse(edit).unwrap_err();
            assert_eq!((error.kind(), error.to_string()), (kind, reason.to_owned()));
        }
    }
}
