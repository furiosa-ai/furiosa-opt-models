//! CLIP and SigLIP image encoders with independently callable feature readouts.
use crate::models;
use crate::models::{Scalar, config};
use crate::modules::vision::Patch;
use crate::modules::{attention, feedforward, language, residual::Residual, transformer, vision};
use crate::ops;
use crate::ops::{
    index::{
        gather::Gather,
        sequence::{Concat, Sum},
        table::Offset,
    },
    linear::Coefficients,
    normalization::LayerNorm,
    pointwise::{
        activation::{NewGelu, QuickGelu},
        add::Add,
    },
};
use crate::tensor::axis::Single;
use crate::tensor::image;
use crate::tensor::image::Grid;
use crate::tensor::image::Image;
use crate::{Identity, Module, Then};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::{io, marker::PhantomData};

/// Image features with the checkpoint's selected pooling and projection contract.
pub enum Model<
    C: AxisName,
    H: AxisName,
    W: AxisName,
    P: AxisName,
    D: AxisName,
    I: AxisName,
    A: AxisName,
    O: AxisName,
    const N: usize,
> {
    /// CLIP token features and unprojected class-token vector.
    Clip(Clip<C, H, W, P, D, I, A, N>),
    /// CLIP token features and projected class-token vector.
    Projected(Projected<C, H, W, P, D, I, A, O, N>),
    /// SigLIP patch features and attention-pooled vector.
    Siglip(Siglip<C, H, W, P, D, I, A, N>),
    /// SigLIP patch features without its attention-pooling head.
    Encoder(SiglipBody<C, H, W, P, D, I, A, N>),
}

/// CLIP token features paired with their pooled class-token vector.
pub type Clip<C, H, W, P, D, I, A, const N: usize> =
    language::FeatureHead<ClipBody<C, H, W, P, D, I, A, N>, ClipHead<D>>;

/// CLIP patch and class-token embedding followed by pre-normalized encoder layers.
pub type ClipBody<C, H, W, P, D, I, A, const N: usize> = Body<
    vision::Embedding<
        Patch<C, D, P, P, f32, ()>,
        Prefix<D>,
        Rows<D, Tokens<H, W, P>>,
        Then<Identity, Norm<D>>,
        Add,
    >,
    [Layer<D, I, A, QuickGelu>; N],
    Identity,
>;

type Body<E, L, N> = transformer::Encoder<E, L, N>;

/// Unmasked layers: the env is empty, the state the encoder's slots.
impl<
    's,
    I,
    Embedding,
    Layer,
    Norm,
    EmbeddingState,
    LayerState,
    NormState,
    D: AxisName,
    R: AxisName,
    const N: usize,
> Module<&mut Context<I, D, R, N>, (), &'s mut (EmbeddingState, [LayerState; N], NormState)>
    for Body<Embedding, [Layer; N], Norm>
where
    Embedding: for<'x> crate::Module<&'x I, (), &'s mut EmbeddingState, Output = Rows<D, R>>,
    Layer: for<'t> crate::Module<Rows<D, R>, (), &'t mut LayerState, Output = Rows<D, R>>,
    Norm: crate::Module<Rows<D, R>, (), &'s mut NormState, Output = Rows<D, R>>,
{
    type Output = Rows<D, R>;

    fn forward(
        &self,
        device: &mut Device,
        env: (),
        state: &'s mut (EmbeddingState, [LayerState; N], NormState),
        context: &mut Context<I, D, R, N>,
    ) -> Self::Output {
        self.forward(
            device,
            env,
            state,
            (&context.input, &mut context.features, &mut context.last),
        )
    }
}

type Prefix<W> = vision::Prepend<Rows<W, Single>, Concat>;

pub type Norm<W> = LayerNorm<Vector<W>, Scalar>;

type Vector<W> = HbmTensor<f32, m![1], m![W]>;

pub type Layer<W, I, H, A> = transformer::PreNorm<Attention<W, H>, Mlp<W, I, A>, Norm<W>, Add>;

type Attention<W, H> = attention::Full<
    attention::Projection<Linear<W, W>>,
    ops::attention::Attention<PhantomData<H>, Scalar>,
    Linear<W, W>,
>;

pub type Linear<I, O, const B: usize = 1> = Coefficients<f32, I, O, B>;

type Mlp<W, I, A> = feedforward::Mlp<Linear<W, I>, A, Linear<I, W>>;

/// Class-token row selection followed by normalization.
pub type ClipHead<D> = Then<Class, Norm<D>>;

type Class = language::SelectRow<Gather, Offset>;

/// CLIP token features paired with a projected class-token vector.
pub type Projected<C, H, W, P, D, I, A, O, const N: usize> =
    language::FeatureHead<ClipBody<C, H, W, P, D, I, A, N>, Then<ClipHead<D>, Linear<D, O, 0>>>;

/// SigLIP patch features paired with their attention-pooled vector.
pub type Siglip<C, H, W, P, D, I, A, const N: usize> =
    language::FeatureHead<SiglipBody<C, H, W, P, D, I, A, N>, SiglipHead<D, I, A>>;

/// SigLIP patch encoding with no class token and a final normalization.
pub type SiglipBody<C, H, W, P, D, I, A, const N: usize> = Body<
    vision::Embedding<Patch<C, D, P, P>, Identity, Rows<D, Patches<H, W, P>>, Identity, Add>,
    [Layer<D, I, A, NewGelu>; N],
    Then<Identity, Norm<D>>,
>;

/// Learned-probe attention pooling followed by a residual feed-forward layer.
pub type SiglipHead<D, I, A> = vision::Pool<
    Rows<D, Single>,
    Linear<D, D>,
    Linear<D, D>,
    Linear<D, D>,
    ops::attention::Attention<PhantomData<A>, Scalar>,
    Linear<D, D>,
    Residual<Norm<D>, Mlp<D, I, NewGelu>, Add>,
>;

/// F32 image pixels ordered by channel, height and width.
pub type Pixels<C, H, W> = Image<C, H, W>;

/// Number of non-overlapping square patches in an image.
pub type Patches<H, W, P> = Grid<H, W, P, P>;

/// CLIP patch rows plus one prepended class-token row.
pub type Tokens<H, W, P> = Sum<Single, Patches<H, W, P>>;

/// Row-major F32 image features.
pub type Rows<W, N> = image::Rows<W, N>;

/// Image input with writable layer-input captures and a pre-normalization final capture.
#[derive(DeviceSend)]
pub struct Context<I, D: AxisName, R: AxisName, const N: usize> {
    /// Pixel input accepted by the selected image embedding.
    pub input: I,
    /// Input feature rows for each encoder layer.
    pub features: [Rows<D, R>; N],
    /// Final layer output before the encoder's final normalization.
    pub last: Rows<D, R>,
}

/// Image tower family and its embedding and pooling conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Class-token image encoding.
    Clip,
    /// Patch-only encoding with optional attention pooling.
    Siglip,
}

/// Parsed image geometry and feature-readout settings.
#[derive(Clone)]
pub struct Config {
    /// Image tower family.
    pub family: Family,
    /// Input pixel channels.
    pub channels: usize,
    /// Image height and width in pixels.
    pub image: [usize; 2],
    /// Square patch side length in pixels.
    pub patch: usize,
    /// Hidden feature width.
    pub width: usize,
    /// Feed-forward expansion width.
    pub intermediate: usize,
    /// Attention heads per layer.
    pub heads: usize,
    /// Encoder layer count.
    pub layers: usize,
    /// Layer-normalization epsilon.
    pub epsilon: f32,
    /// CLIP projection output width, when requested.
    pub projection: Option<usize>,
    /// Checkpoint `vision_use_head` flag, used to select SigLIP attention pooling.
    pub pool: bool,
}

impl models::Family for Config {
    type Hf = Hf;
}

/// An image tower's checkpoint configuration.
#[derive(Deserialize)]
pub struct Hf {
    architectures: Option<Vec<String>>,
    model_type: Option<String>,
    projection_dim: Option<usize>,
    #[serde(flatten, deserialize_with = "config::vision")]
    tower: Tower,
}

impl TryFrom<Hf> for Config {
    type Error = io::Error;

    /// Narrows an image tower configuration with the projection its architecture declares.
    fn try_from(hf: Hf) -> io::Result<Self> {
        let projected = (hf.architectures.iter().flatten())
            .any(|name| matches!(name.as_str(), "CLIPVisionModelWithProjection" | "CLIPModel"));
        let tower = hf.tower;
        let family = match tower.model_type.as_deref().or(hf.model_type.as_deref()) {
            Some("clip" | "clip_vision_model") => Family::Clip,
            Some("siglip" | "siglip_vision_model") => Family::Siglip,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "expected CLIP or SigLIP vision config",
                ));
            }
        };
        let dimension = |name: &str, value: Option<usize>, default: usize| {
            Some(value.unwrap_or(default))
                .filter(|&value| value > 0)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{name}: expected positive dimension"),
                    )
                })
        };
        let pair = |name: &str, value: Option<Size>, default: usize| match value {
            Some(Size::Pair(axes)) if axes.iter().all(|&axis| axis > 0) => Ok(axes),
            Some(Size::Pair(_)) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{name}: expected two positive dimensions"),
            )),
            Some(Size::Side(side)) => dimension(name, Some(side), default).map(|n| [n, n]),
            None => dimension(name, None, default).map(|n| [n, n]),
        };
        let image = pair("image_size", tower.image_size, 224)?;
        let patch = pair(
            "patch_size",
            tower.patch_size,
            if family == Family::Clip { 32 } else { 16 },
        )?;
        let activation = tower
            .hidden_act
            .as_deref()
            .unwrap_or(if family == Family::Clip {
                "quick_gelu"
            } else {
                "gelu_pytorch_tanh"
            });
        if patch[0] != patch[1]
            || image.iter().any(|size| size % patch[0] != 0)
            || !matches!(
                (family, activation),
                (Family::Clip, "quick_gelu") | (Family::Siglip, "gelu_pytorch_tanh" | "gelu_new")
            )
            || (projected && family != Family::Clip)
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "unsupported vision patch, activation, or projection",
            ));
        }
        let epsilon =
            tower
                .layer_norm_eps
                .unwrap_or(if family == Family::Clip { 1e-5 } else { 1e-6 }) as f32;
        let width = dimension("hidden_size", tower.hidden_size, 768)?;
        let heads = dimension("num_attention_heads", tower.num_attention_heads, 12)?;
        if !epsilon.is_finite() || epsilon <= 0.0 || width % heads != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid vision attention or normalization dimensions",
            ));
        }
        let projection = projected
            .then(|| {
                Some(hf.projection_dim.or(tower.projection_dim).unwrap_or(512))
                    .filter(|&dimension| dimension > 0)
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "projection_dim must be positive",
                        )
                    })
            })
            .transpose()?;
        Ok(Self {
            family,
            channels: dimension("num_channels", tower.num_channels, 3)?,
            image,
            patch: patch[0],
            width,
            intermediate: dimension("intermediate_size", tower.intermediate_size, 3072)?,
            heads,
            layers: dimension("num_hidden_layers", tower.num_hidden_layers, 12)?,
            epsilon,
            projection,
            pool: tower.vision_use_head.unwrap_or(true),
        })
    }
}

/// The image tower's geometry, each field defaulting by the tower's family.
#[derive(Deserialize)]
struct Tower {
    model_type: Option<String>,
    projection_dim: Option<usize>,
    image_size: Option<Size>,
    patch_size: Option<Size>,
    num_channels: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_hidden_layers: Option<usize>,
    hidden_act: Option<String>,
    layer_norm_eps: Option<f64>,
    vision_use_head: Option<bool>,
}

/// A square image or patch side, or its `[height, width]`.
#[derive(Deserialize)]
#[serde(untagged)]
enum Size {
    Side(usize),
    Pair([usize; 2]),
}

impl Config {
    /// Whether the parsed head flag or projection requests a pooled output.
    pub fn pooled(&self) -> bool {
        self.pool || self.projection.is_some()
    }

    /// Image tower family.
    pub fn family(&self) -> Family {
        self.family
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// The CLIP image tower's config with a projection, its other fields at their defaults.
    fn document() -> Value {
        json!({"architectures": ["CLIPVisionModelWithProjection"], "model_type": "clip_vision_model"})
    }

    /// A CLIP or SigLIP tower of square patches that tile the image, its family's activation,
    /// positive dimensions and epsilon, and a projection only on CLIP, parses; each other value is
    /// refused by its kind and reason.
    #[test]
    fn refuses_what_the_tower_does_not_run() {
        let parse = |edit: &[(&str, Value)]| {
            let mut document = document();
            for (key, value) in edit {
                document[*key] = value.clone();
            }
            <Config as models::Family>::parse(&document).map(drop)
        };
        parse(&[]).unwrap();
        use io::ErrorKind::{InvalidData, Unsupported};
        let unsupported = "unsupported vision patch, activation, or projection";
        let invalid = "invalid vision attention or normalization dimensions";
        let positive = |name: &str| format!("{name}: expected positive dimension");
        let pair = |name: &str| format!("{name}: expected two positive dimensions");
        let mut cases = vec![
            (
                vec![("model_type", json!("vit"))],
                Unsupported,
                "expected CLIP or SigLIP vision config".to_owned(),
            ),
            (
                vec![("image_size", json!([0, 224]))],
                InvalidData,
                pair("image_size"),
            ),
            (
                vec![("patch_size", json!([0, 32]))],
                InvalidData,
                pair("patch_size"),
            ),
            (
                vec![("patch_size", json!([16, 32]))],
                Unsupported,
                unsupported.to_owned(),
            ),
            (
                vec![("image_size", json!(100))],
                Unsupported,
                unsupported.to_owned(),
            ),
            (
                vec![("hidden_act", json!("gelu"))],
                Unsupported,
                unsupported.to_owned(),
            ),
            (
                vec![("model_type", json!("siglip_vision_model"))],
                Unsupported,
                unsupported.to_owned(),
            ),
            (
                vec![("num_attention_heads", json!(5))],
                InvalidData,
                invalid.to_owned(),
            ),
            (
                vec![("layer_norm_eps", json!(0.0))],
                InvalidData,
                invalid.to_owned(),
            ),
            (
                vec![("layer_norm_eps", json!(1e39))],
                InvalidData,
                invalid.to_owned(),
            ),
            (
                vec![("projection_dim", json!(0))],
                InvalidData,
                "projection_dim must be positive".to_owned(),
            ),
        ];
        for name in [
            "image_size",
            "patch_size",
            "hidden_size",
            "num_attention_heads",
            "num_channels",
            "intermediate_size",
            "num_hidden_layers",
        ] {
            cases.push((vec![(name, json!(0))], InvalidData, positive(name)));
        }
        for (edit, kind, reason) in cases {
            let error = parse(&edit).unwrap_err();
            assert_eq!(
                (error.kind(), error.to_string()),
                (kind, reason),
                "{edit:?}"
            );
        }
    }
}
