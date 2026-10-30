//! Gemma 4 unified (`gemma4_unified`): the media embedders of `Gemma4UnifiedForConditionalGeneration`,
//! which turn an image, a video frame or audio into rows of the text model's width without a tower;
//! the rows replace the media's placeholder tokens in the language model's input.
//!
//! - An image or a video frame is merged patches, 48 x 48 pixels of three channels a row, each with
//!   its (x, y) position. The patch path normalizes a patch, projects it, normalizes it, adds one row
//!   of a factorized position table per coordinate and normalizes again; `embed_vision` then projects
//!   it into the text width after a unit-gain RMS norm.
//! - Audio is 640 waveform samples (40 ms at 16 kHz) a row, projected by `embed_audio` in the same way.
//! - Every row is computed alone, so one call may hold the patches of several images or frames.
//!   Padding patches follow the valid ones; their rows are computed and read by no caller.
//! - [`Encoder`] is what a load of the media encoder alone reads: `model.vision_embedder.*`,
//!   `model.embed_vision.*`, `model.embed_audio.*`, the language model ignored.

use furiosa_opt_std::prelude::*;
use serde::Deserialize;

use crate::load::{Checkpoint, Load, LoadError, Named};
use crate::models::config::{self, Constant, Literal};
use crate::models::{Class, Implements};
use crate::modules::language::ConditionedEmbedding;
use crate::ops::embedding::Factorized;
use crate::ops::linear::{Affine, Linear};
use crate::ops::normalization::{LayerNorm, UnitRmsNorm};
use crate::ops::pointwise::add::Add;
use crate::tensor::memory::{Hbm, Memory};
use crate::tensor::parallel::Chip;
use crate::{Identity, Module, Sequential, Then};

axes![
    Patch = 6912,
    Embed = 3840,
    Text = 3840,
    Samples = 640,
    Positions = 1120,
    Coordinates = 2,
    Short = 512,
    Long = 2304
];

/// A merged patch's features: 48 x 48 pixels of three channels.
pub type Patches<R> = HbmTensor<bf16, Chip, m![R, Patch]>;

/// Audio frames of 640 samples.
pub type Frames<R> = HbmTensor<bf16, Chip, m![R, Samples]>;

/// Rows of the text model's width, the embedders' output.
pub type Rows<R> = HbmTensor<bf16, Chip, m![R, Text]>;

/// The byte offset of each patch's row in the factorized position table for one coordinate:
/// [`offset`](super::Encoder::offset) of its x, or of its y.
pub type Offsets<R> = HbmTensor<i32, Chip, m![R]>;

/// The factorized position table as stored, `[positions, coordinates, features]`: a table of rows
/// `2 position + coordinate`.
pub type Table = m![{ m![Positions, Coordinates] }, Embed];

/// The media encoder of the published size over the memory `X`.
pub type Encoder<X> = Embedders<
    VisionEmbedder<
        Norm<Patch, X>,
        Norm<Embed, X>,
        Affine<
            <X as Memory>::Tensor<bf16, Chip, m![Embed, Patch]>,
            <X as Memory>::Tensor<f32, Chip, m![Embed]>,
        >,
        <X as Memory>::Tensor<bf16, Chip, Table>,
    >,
    MultimodalEmbedder<Unit<Embed, X>, TextProjection<Embed, X>>,
    MultimodalEmbedder<Unit<Samples, X>, TextProjection<Samples, X>>,
>;

/// The embedders' `nn.LayerNorm`, epsilon 1e-5, over BF16 parameters.
type Norm<W, X> =
    LayerNorm<<X as Memory>::Tensor<bf16, Chip, m![W]>, <X as Memory>::Tensor<f32, Chip, m![1]>>;

/// The embedders' unit-gain RMS norm, of epsilon `rms_norm_eps`.
type Unit<W, X> = UnitRmsNorm<W, EPSILON, Chip, <X as Memory>::Tensor<bf16, Chip, m![W]>>;

/// The bias-free projection of `In` rows into the text width.
type TextProjection<In, X> =
    Linear<In, Text, Chip, <X as Memory>::Tensor<bf16, Chip, m![Text, In]>>;

const EPSILON: usize = 1e-6_f32.to_bits() as usize;

/// `Gemma4UnifiedVisionEmbedder`: each patch's row of the embedding width,
/// `pos_norm(patch_ln2(patch_dense(patch_ln1(patch))) + pos[x, 0] + pos[y, 1])`, every sum rounded
/// to BF16 as Transformers' BF16 forward rounds it; its input is the patches and their x and y
/// offsets.
pub type VisionEmbedder<Wide, Norm, Dense, Position> = Then<
    ConditionedEmbedding<
        Then<Then<Named<"patch_ln1", Wide>, Named<"patch_dense", Dense>>, Named<"patch_ln2", Norm>>,
        Named<"pos_embedding", Factorized<Position>>,
        Add,
    >,
    Named<"pos_norm", Norm>,
>;

/// `Gemma4UnifiedMultimodalEmbedder`: rows normalized by a unit-gain RMS norm, whose checkpoint
/// stores no weight, and projected into the caller's rows of the text width; its input is the rows
/// and that destination, which the pair passes on.
pub type MultimodalEmbedder<Norm, Projection> = Sequential<
    (Named<"embedding_pre_projection_norm", Norm>, Identity),
    Named<"embedding_projection", Projection>,
>;

/// A patch's row of the embedding width, between the two embedders.
type Embedded<R> = HbmTensor<bf16, Chip, m![R, Embed]>;

/// The embedders of the conditional checkpoint without its language model: what a load of the
/// media encoder alone reads, under `model`.
#[derive(DeviceSend, Load)]
#[load(ignore = "language_model")]
pub struct Embedders<Vision, Image, Audio> {
    pub vision_embedder: Vision,
    pub embed_vision: Image,
    pub embed_audio: Audio,
}

/// The embedders read a checkpoint under `model`, the size whose text width they project into, if
/// its config declares their widths and no load quantizes their BF16 projections.
impl<Vision, Image, Audio> Implements for Embedders<Vision, Image, Audio> {
    const SIZE: &'static str = "12B";
    const CLASS: Class = Class {
        family: "gemma4_unified",
        name: "Encoder",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        if source.quantization().is_some() {
            return Err(LoadError::Unsupported {
                name: "quantization".into(),
                reason: "the media encoder's projections stay BF16".into(),
            });
        }
        let widths = || {
            let Hf {
                model_type: Literal,
                text_config:
                    TextHf {
                        hidden_size: config::Size { .. },
                    },
                vision_config:
                    VisionHf {
                        mm_embed_dim: config::Size { .. },
                        mm_posemb_size: config::Size { .. },
                        patch_size: Constant,
                        pooling_kernel_size: Constant,
                        rms_norm_eps: vision,
                    },
                audio_config:
                    AudioHf {
                        audio_embed_dim: config::Size { .. },
                        rms_norm_eps: audio,
                    },
            } = config::parse(source.config())?;
            let compiled = f64::from(f32::from_bits(EPSILON as u32));
            config::compiled("vision_config.rms_norm_eps", vision as f32 as f64, compiled)?;
            config::compiled("audio_config.rms_norm_eps", audio as f32 as f64, compiled)
        };
        widths().map(|_| "model").map_err(|error| {
            let (name, reason) = ("config.json".into(), error.to_string());
            match error.kind() {
                std::io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
                _ => LoadError::InvalidData { name, reason },
            }
        })
    }
}

/// The config values the embedders' widths are fixed to: the text width, the vision embedding
/// width, the position table, a merged patch of 3 x 16 x 16 pixels pooled 3 x 3, and the audio
/// frame.
#[derive(Deserialize)]
struct Hf {
    model_type: Literal<"gemma4_unified">,
    text_config: TextHf,
    vision_config: VisionHf,
    audio_config: AudioHf,
}

#[derive(Deserialize)]
struct TextHf {
    hidden_size: config::Size<Text>,
}

#[derive(Deserialize)]
struct VisionHf {
    mm_embed_dim: config::Size<Embed>,
    mm_posemb_size: config::Size<Positions>,
    patch_size: Constant<16>,
    pooling_kernel_size: Constant<3>,
    rms_norm_eps: f64,
}

#[derive(Deserialize)]
struct AudioHf {
    audio_embed_dim: config::Size<Samples>,
    rms_norm_eps: f64,
}

/// A composition, not an op.
impl<X, V, I, A> !crate::Op<X> for Embedders<V, I, A> {}

/// An image's or a video frame's patches into the caller's rows; the state holds the vision
/// embedder's slot, then `embed_vision`'s.
impl<'x, 's, R: AxisName, V, I, A, SV, SI, Env: Copy>
    Module<
        (
            &'x Patches<R>,
            &'x Offsets<R>,
            &'x Offsets<R>,
            &'x mut Rows<R>,
        ),
        Env,
        &'s mut (SV, SI),
    > for Embedders<V, I, A>
where
    V: Module<
            (&'x Patches<R>, (&'x Offsets<R>, &'x Offsets<R>)),
            Env,
            &'s mut SV,
            Output = Embedded<R>,
        >,
    I: for<'e> Module<(&'e Embedded<R>, &'x mut Rows<R>), Env, &'s mut SI, Output = ()>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (vision, image): &'s mut (SV, SI),
        (patches, x, y, output): (
            &'x Patches<R>,
            &'x Offsets<R>,
            &'x Offsets<R>,
            &'x mut Rows<R>,
        ),
    ) {
        let embedded = self
            .vision_embedder
            .forward(&mut *device, env, vision, (patches, (x, y)));
        self.embed_vision
            .forward(device, env, image, (&embedded, output));
    }
}

/// Audio frames into the caller's rows; the state is `embed_audio`'s.
impl<'x, 's, R: AxisName, V, I, A, SA, Env: Copy>
    Module<(&'x Frames<R>, &'x mut Rows<R>), Env, &'s mut SA> for Embedders<V, I, A>
where
    A: Module<(&'x Frames<R>, &'x mut Rows<R>), Env, &'s mut SA, Output = ()>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &'s mut SA,
        input: (&'x Frames<R>, &'x mut Rows<R>),
    ) {
        self.embed_audio.forward(device, env, state, input);
    }
}

/// Launches of 512 or 2304 rows; a patch's operand is the byte offset of the [`Table`] row its
/// coordinate reads.
impl super::Encoder for Encoder<Hbm> {
    type Short = Short;
    type Long = Long;
    type Patch = Patch;
    type Positions = Positions;
    type Samples = Samples;
    type Text = Text;
    type Vision = (VisionState, ProjectionState);
    type Audio = ProjectionState;
    type View<'a> = Encoder<&'a Hbm>;

    fn offset(position: usize, coordinate: usize) -> i32 {
        ((Coordinates::SIZE * position + coordinate) * Embed::SIZE * size_of::<bf16>()) as i32
    }
}

/// A [`VisionEmbedder`]'s state: its patch path's and its position table's, then its last norm's.
type VisionState = (((((), ()), ()), ()), ());

/// A [`MultimodalEmbedder`]'s state: its norm's and the identity's, then its projection's.
type ProjectionState = (((), ()), ());

#[cfg(test)]
mod tests {
    use super::Encoder;
    use crate::load::{Checkpoint, LoadError};
    use crate::models::Implements;
    use crate::tensor::form::fp8::Granularity;
    use crate::tensor::memory::Hbm;

    /// The encoder reads a checkpoint of its widths under `model`, and refuses one of other widths
    /// and a load that quantizes it.
    #[test]
    fn reads_a_config_of_its_widths() {
        let empty = tempfile::tempdir().unwrap();
        let config = |hidden: usize| {
            let config = serde_json::json!({
                "model_type": "gemma4_unified",
                "text_config": {"hidden_size": hidden},
                "vision_config": {
                    "mm_embed_dim": 3840, "mm_posemb_size": 1120, "patch_size": 16,
                    "pooling_kernel_size": 3, "rms_norm_eps": 1e-6,
                },
                "audio_config": {"audio_embed_dim": 640, "rms_norm_eps": 1e-6},
            });
            Checkpoint::new(empty.path(), Some(config)).unwrap()
        };
        let scope = <Encoder<Hbm> as Implements>::scope;
        assert_eq!(scope(&config(3840)).unwrap(), "model");
        let other = scope(&config(2560));
        assert!(
            matches!(&other, Err(LoadError::Unsupported { name, .. }) if name == "config.json"),
            "{other:?}"
        );
        let quantized = scope(&config(3840).quantized(Granularity::Tensor).unwrap());
        assert!(
            matches!(&quantized, Err(LoadError::Unsupported { name, .. }) if name == "quantization"),
            "{quantized:?}"
        );
    }
}
