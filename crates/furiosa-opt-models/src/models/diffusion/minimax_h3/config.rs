//! Owned checkpoint schemas narrowed to the geometry of the preserved RNGD implementation.

use crate::models::Family;
use crate::models::config::{self as schema, Constant, Literal};
use serde::Deserialize;
use serde_json::Value;
use std::{io, num::NonZeroUsize};

/// Noise schedule shifts read from the pipeline configuration, not a scheduler component.
#[derive(Clone, Debug, PartialEq)]
pub struct Shifts {
    video: f64,
    audio: f64,
}

impl Shifts {
    pub fn video(&self) -> f64 {
        self.video
    }
    pub fn audio(&self) -> f64 {
        self.audio
    }
}

/// Finite latent means and positive finite deviations, retained at their checkpoint precision.
/// These values do not imply an additional normalization operation or a parameter fold.
#[derive(Clone, Debug, PartialEq)]
pub struct Statistics<const CHANNELS: usize> {
    mean: [f64; CHANNELS],
    std: [f64; CHANNELS],
}

impl<const CHANNELS: usize> Statistics<CHANNELS> {
    pub fn mean(&self) -> &[f64; CHANNELS] {
        &self.mean
    }
    pub fn std(&self) -> &[f64; CHANNELS] {
        &self.std
    }
}

#[derive(Deserialize)]
struct StatisticsHf {
    latents_mean: Vec<f64>,
    latents_std: Vec<f64>,
}

impl<const CHANNELS: usize> TryFrom<StatisticsHf> for Statistics<CHANNELS> {
    type Error = io::Error;

    fn try_from(raw: StatisticsHf) -> io::Result<Self> {
        let array = |key: &str, values: Vec<f64>| -> io::Result<[f64; CHANNELS]> {
            let count = values.len();
            values.try_into().map_err(|_| {
                invalid(format!(
                    "{key}: found {count} channels, expected {CHANNELS}",
                ))
            })
        };
        let mean = array("latents_mean", raw.latents_mean)?;
        let std = array("latents_std", raw.latents_std)?;
        for (channel, &value) in mean.iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid(format!("latents_mean[{channel}]: must be finite")));
            }
        }
        for (channel, &value) in std.iter().enumerate() {
            positive(&format!("latents_std[{channel}]"), value)?;
        }
        Ok(Self { mean, std })
    }
}

/// Checked pipeline settings from the actual merged pipeline document.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    shifts: Shifts,
}

impl Config {
    pub fn shifts(&self) -> &Shifts {
        &self.shifts
    }
}

/// The pipeline index fields retained by the public checkpoint's merged configuration.
#[derive(Deserialize)]
pub struct PipelineHf {
    #[serde(rename = "_class_name")]
    class: Literal<"MiniMaxH3Pipeline">,
    #[serde(rename = "_minimax_h3")]
    h3: Section,
    scheduler: Option<Value>,
}

#[derive(Deserialize)]
struct Section {
    #[serde(default, rename = "schema_version")]
    _schema_version: Option<Constant<1>>,
    partition: Literal<"fl2va">,
    sigma_shift_scales: ShiftsHf,
}

#[derive(Deserialize)]
struct ShiftsHf {
    video: f64,
    audio: f64,
}

impl TryFrom<PipelineHf> for Config {
    type Error = io::Error;

    fn try_from(raw: PipelineHf) -> io::Result<Self> {
        let PipelineHf {
            class: Literal,
            h3,
            scheduler,
        } = raw;
        let Section {
            _schema_version: _,
            partition: Literal,
            sigma_shift_scales: shifts,
        } = h3;
        if scheduler.is_some_and(|value| !value.is_null()) {
            return Err(unsupported(
                "scheduler: H3 uses the pipeline's sigma_shift_scales",
            ));
        }
        positive("_minimax_h3.sigma_shift_scales.video", shifts.video)?;
        positive("_minimax_h3.sigma_shift_scales.audio", shifts.audio)?;
        Ok(Self {
            shifts: Shifts {
                video: shifts.video,
                audio: shifts.audio,
            },
        })
    }
}

impl Family for Config {
    type Hf = PipelineHf;
}

/// Checked joint-transformer geometry and rotary base.
#[derive(Clone, Debug, PartialEq)]
pub struct Transformer {
    rope_theta: f32,
}

impl Transformer {
    pub fn rope_theta(&self) -> f32 {
        self.rope_theta
    }
}

/// The transformer's checkpoint configuration. Constants are validated during deserialization.
#[derive(Deserialize)]
pub struct TransformerHf {
    hidden_size: Constant<5376>,
    num_layers: Constant<50>,
    token_refiner_num_layers: Constant<2>,
    num_attention_heads: Constant<56>,
    attention_head_dim: Constant<128>,
    ffn_hidden_size: Constant<14336>,
    latents_dim: Constant<24>,
    audio_latents_dim: Constant<32>,
    patch_size: [usize; 3],
    text_dim: Constant<5120>,
    timestep_input_dim: Constant<256>,
    time_embed_hidden_size: Constant<5376>,
    time_embed_dim: Constant<2688>,
    adaln_out_features: Constant<96768>,
    final_adaln_out_features: Constant<10752>,
    rope_inv_freq_len: Constant<16>,
    norm_eps: f64,
    qk_norm_eps: f64,
    final_norm_eps: f64,
    #[serde(default = "default_theta")]
    rope_theta: f64,
    quantization_config: Option<Value>,
}

impl TryFrom<TransformerHf> for Transformer {
    type Error = io::Error;

    fn try_from(raw: TransformerHf) -> io::Result<Self> {
        let TransformerHf {
            hidden_size: Constant,
            num_layers: Constant,
            token_refiner_num_layers: Constant,
            num_attention_heads: Constant,
            attention_head_dim: Constant,
            ffn_hidden_size: Constant,
            latents_dim: Constant,
            audio_latents_dim: Constant,
            patch_size,
            text_dim: Constant,
            timestep_input_dim: Constant,
            time_embed_hidden_size: Constant,
            time_embed_dim: Constant,
            adaln_out_features: Constant,
            final_adaln_out_features: Constant,
            rope_inv_freq_len: Constant,
            norm_eps,
            qk_norm_eps,
            final_norm_eps,
            rope_theta,
            quantization_config,
        } = raw;
        schema::compiled("patch_size", patch_size, [1, 2, 2])?;
        schema::compiled("norm_eps", norm_eps, 1e-5)?;
        schema::compiled("qk_norm_eps", qk_norm_eps, 1e-5)?;
        schema::compiled("final_norm_eps", final_norm_eps, 1e-5)?;
        positive("rope_theta", rope_theta)?;
        let theta = rope_theta as f32;
        if !theta.is_finite() || theta <= 0. {
            return Err(invalid("rope_theta: must fit a positive finite FP32 value"));
        }
        unquantized(quantization_config)?;
        Ok(Self { rope_theta: theta })
    }
}

impl Family for Transformer {
    type Hf = TransformerHf;
}

/// Qwen3-VL language configuration for the first 50 layers, before final normalization.
#[derive(Clone, Debug, PartialEq)]
pub struct TextEncoder {
    layers: usize,
    positions: Option<NonZeroUsize>,
}

impl TextEncoder {
    pub const READ_LAYERS: usize = 50;
    pub const ROPE_THETA: f32 = 5e6;
    pub fn checkpoint_layers(&self) -> usize {
        self.layers
    }
    pub fn positions(&self) -> Option<NonZeroUsize> {
        self.positions
    }
}

/// The actual text component, with the language model nested or flattened as public config handles.
#[derive(Deserialize)]
pub struct TextHf {
    #[serde(flatten, deserialize_with = "schema::text")]
    language: LanguageHf,
    quantization_config: Option<Value>,
}

#[derive(Deserialize)]
struct LanguageHf {
    hidden_size: Constant<5120>,
    num_attention_heads: Constant<64>,
    num_key_value_heads: Constant<8>,
    head_dim: Constant<128>,
    intermediate_size: Constant<25600>,
    num_hidden_layers: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    hidden_act: Option<String>,
    attention_bias: Option<bool>,
    #[serde(rename = "vocab_size")]
    _vocab_size: Option<Constant<151936>>,
    max_position_embeddings: Option<NonZeroUsize>,
    rope_scaling: Option<RotationHf>,
    quantization_config: Option<Value>,
}

#[derive(Deserialize)]
struct RotationHf {
    mrope_interleaved: bool,
    rope_type: Literal<"default">,
}

impl TryFrom<TextHf> for TextEncoder {
    type Error = io::Error;

    fn try_from(raw: TextHf) -> io::Result<Self> {
        unquantized(raw.quantization_config)?;
        let LanguageHf {
            hidden_size: Constant,
            num_attention_heads: Constant,
            num_key_value_heads: Constant,
            head_dim: Constant,
            intermediate_size: Constant,
            num_hidden_layers,
            rope_theta,
            rms_norm_eps,
            hidden_act,
            attention_bias,
            _vocab_size: _,
            max_position_embeddings,
            rope_scaling,
            quantization_config,
        } = raw.language;
        if num_hidden_layers < Self::READ_LAYERS {
            return Err(invalid("num_hidden_layers: H3 reads the first 50 layers"));
        }
        schema::compiled("rope_theta", rope_theta, f64::from(Self::ROPE_THETA))?;
        schema::compiled("rms_norm_eps", rms_norm_eps, 1e-6)?;
        schema::compiled(
            "hidden_act",
            hidden_act.as_deref().unwrap_or("silu"),
            "silu",
        )?;
        schema::compiled("attention_bias", attention_bias.unwrap_or(false), false)?;
        if let Some(RotationHf {
            mrope_interleaved,
            rope_type: Literal,
        }) = rope_scaling
        {
            schema::compiled("rope_scaling.mrope_interleaved", mrope_interleaved, true)?;
        }
        unquantized(quantization_config)?;
        Ok(Self {
            layers: num_hidden_layers,
            positions: max_position_embeddings,
        })
    }
}

impl Family for TextEncoder {
    type Hf = TextHf;
}

/// Video wrapper settings and latent statistics. Source architecture is checked independently.
#[derive(Clone, Debug, PartialEq)]
pub struct Video {
    statistics: Statistics<24>,
}

impl Video {
    pub fn statistics(&self) -> &Statistics<24> {
        &self.statistics
    }
}

/// Configuration of the video wrapper, distinct from its source model's configuration.
#[derive(Deserialize)]
pub struct VideoHf {
    latent_channels: Constant<24>,
    vae_tile_size: Constant<256>,
    vae_tile_overlap_min: Constant<64>,
    vae_clip_length: Constant<17>,
    #[serde(flatten, deserialize_with = "schema::flattened")]
    statistics: StatisticsHf,
}

impl TryFrom<VideoHf> for Video {
    type Error = io::Error;
    fn try_from(raw: VideoHf) -> io::Result<Self> {
        let VideoHf {
            latent_channels: Constant,
            vae_tile_size: Constant,
            vae_tile_overlap_min: Constant,
            vae_clip_length: Constant,
            statistics,
        } = raw;
        Ok(Self {
            statistics: statistics.try_into()?,
        })
    }
}

impl Family for Video {
    type Hf = VideoHf;
}

/// Proof that a video's source configuration describes the compiled ViT decoder.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoSource {
    _geometry: (),
}

/// Configuration beside the video decoder's weights.
#[derive(Deserialize)]
pub struct VideoSourceHf {
    use_vit_decoder: bool,
    embed_dim: Constant<24>,
    vae_ratio: Constant<16>,
    vae_ratio_t: Constant<4>,
    pixel_norm_type: Literal<"imagenet">,
    vit_decoder_kwargs: VitHf,
    quantization_config: Option<Value>,
}

#[derive(Deserialize)]
struct VitHf {
    dim_head: Constant<64>,
    heads: Constant<32>,
    num_layers: Constant<36>,
    rope_theta: f64,
    rope_dim_ratio: f64,
    norm_type: Literal<"rms_norm">,
    qk_norm_type: Literal<"rms_norm">,
    qk_norm_affine: bool,
    norm_affine: bool,
    ffn_use_gated: bool,
    ffn_activation_fn: Literal<"silu">,
}

impl TryFrom<VideoSourceHf> for VideoSource {
    type Error = io::Error;
    fn try_from(raw: VideoSourceHf) -> io::Result<Self> {
        let VideoSourceHf {
            use_vit_decoder,
            embed_dim: Constant,
            vae_ratio: Constant,
            vae_ratio_t: Constant,
            pixel_norm_type: Literal,
            vit_decoder_kwargs: vit,
            quantization_config,
        } = raw;
        let VitHf {
            dim_head: Constant,
            heads: Constant,
            num_layers: Constant,
            rope_theta,
            rope_dim_ratio,
            norm_type: Literal,
            qk_norm_type: Literal,
            qk_norm_affine,
            norm_affine,
            ffn_use_gated,
            ffn_activation_fn: Literal,
        } = vit;
        schema::compiled("use_vit_decoder", use_vit_decoder, true)?;
        schema::compiled("vit_decoder_kwargs.rope_theta", rope_theta, 100.)?;
        schema::compiled("vit_decoder_kwargs.rope_dim_ratio", rope_dim_ratio, 0.75)?;
        schema::compiled("vit_decoder_kwargs.qk_norm_affine", qk_norm_affine, false)?;
        schema::compiled("vit_decoder_kwargs.norm_affine", norm_affine, true)?;
        schema::compiled("vit_decoder_kwargs.ffn_use_gated", ffn_use_gated, true)?;
        unquantized(quantization_config)?;
        Ok(Self { _geometry: () })
    }
}

impl Family for VideoSource {
    type Hf = VideoSourceHf;
}

/// Stereo 32 kHz audio wrapper settings and latent statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    statistics: Statistics<32>,
}

impl Audio {
    pub fn statistics(&self) -> &Statistics<32> {
        &self.statistics
    }
}

/// Configuration of the audio wrapper, separate from its decoder metadata.
#[derive(Deserialize)]
pub struct AudioHf {
    latent_channels: Constant<32>,
    sample_rate: Constant<32000>,
    output_channel: Constant<2>,
    #[serde(flatten, deserialize_with = "schema::flattened")]
    statistics: StatisticsHf,
}

impl TryFrom<AudioHf> for Audio {
    type Error = io::Error;
    fn try_from(raw: AudioHf) -> io::Result<Self> {
        let AudioHf {
            latent_channels: Constant,
            sample_rate: Constant,
            output_channel: Constant,
            statistics,
        } = raw;
        Ok(Self {
            statistics: statistics.try_into()?,
        })
    }
}

impl Family for Audio {
    type Hf = AudioHf;
}

/// Proof that the separate audio metadata describes the compiled BigVGAN decoder.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioMetadata {
    _geometry: (),
}

/// The actual audio `metadata.json` document; it is not merged into wrapper configuration.
#[derive(Deserialize)]
pub struct AudioMetadataHf {
    metadata: MetadataHf,
}

#[derive(Deserialize)]
struct MetadataHf {
    kwargs: DecoderHf,
}

#[derive(Deserialize)]
struct DecoderHf {
    decoder_type: Literal<"bigvgan">,
    decoder_dim: Constant<1024>,
    decoder_rates: Vec<usize>,
    vae_latent_channels: Constant<32>,
    latent_dim: Constant<2048>,
}

impl TryFrom<AudioMetadataHf> for AudioMetadata {
    type Error = io::Error;
    fn try_from(raw: AudioMetadataHf) -> io::Result<Self> {
        let DecoderHf {
            decoder_type: Literal,
            decoder_dim: Constant,
            decoder_rates,
            vae_latent_channels: Constant,
            latent_dim: Constant,
        } = raw.metadata.kwargs;
        schema::compiled(
            "metadata.kwargs.decoder_rates",
            decoder_rates,
            vec![5, 5, 2, 2, 2, 2, 2],
        )?;
        Ok(Self { _geometry: () })
    }
}

impl Family for AudioMetadata {
    type Hf = AudioMetadataHf;
}

fn default_theta() -> f64 {
    10_000.
}

fn positive(key: &str, value: f64) -> io::Result<()> {
    if !value.is_finite() || value <= 0. {
        return Err(invalid(format!("{key}: must be positive and finite")));
    }
    Ok(())
}

fn unquantized(value: Option<Value>) -> io::Result<()> {
    if value.is_some_and(|value| !value.is_null()) {
        return Err(unsupported(
            "quantization_config: H3 reads unquantized parameters",
        ));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn unsupported(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, message.into())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    /// Synthetic configuration values only: no weights, outputs or captured latent statistics.
    pub(crate) fn documents() -> (Value, Value, Value) {
        let index = json!({
            "_class_name": "MiniMaxH3Pipeline",
            "_minimax_h3": {"schema_version": 1, "partition": "fl2va",
                "sigma_shift_scales": {"video": 2.0, "audio": 1.0}},
            "scheduler": null,
            "transformer": ["diffusers", "MiniMaxH3DiTModel"],
            "text_encoder": ["transformers", "MiniMaxH3Qwen3VLHFEncoder"],
            "video_vae": ["diffusers", "MiniMaxH3VideoVAE"],
            "audio_vae": ["diffusers", "MiniMaxH3AudioVAE"],
        });
        let mut merged = index.clone();
        merged["transformer"] = json!({
            "hidden_size":5376, "num_layers":50, "token_refiner_num_layers":2,
            "num_attention_heads":56, "attention_head_dim":128, "ffn_hidden_size":14336,
            "latents_dim":24, "audio_latents_dim":32, "patch_size":[1,2,2], "text_dim":5120,
            "timestep_input_dim":256, "time_embed_hidden_size":5376, "time_embed_dim":2688,
            "adaln_out_features":96768, "final_adaln_out_features":10752, "rope_inv_freq_len":16,
            "norm_eps":1e-5, "qk_norm_eps":1e-5, "final_norm_eps":1e-5,
        });
        merged["text_encoder"] = json!({"text_config": {
            "hidden_size":5120, "num_attention_heads":64, "num_key_value_heads":8,
            "head_dim":128, "intermediate_size":25600, "num_hidden_layers":64,
            "rope_theta":5e6, "rms_norm_eps":1e-6, "vocab_size":151936,
            "max_position_embeddings":4096,
            "rope_scaling":{"mrope_interleaved":true,"rope_type":"default","mrope_section":[24,20,20]},
        }});
        merged["video_vae"] = json!({
            "source_path":"source", "latent_channels":24, "vae_tile_size":256,
            "vae_tile_overlap_min":64, "vae_clip_length":17,
            "latents_mean":vec![0.;24], "latents_std":vec![1.;24],
            "source": {"use_vit_decoder":true,"embed_dim":24,"vae_ratio":16,"vae_ratio_t":4,
                "pixel_norm_type":"imagenet", "vit_decoder_kwargs":{
                    "dim_head":64,"heads":32,"num_layers":36,"rope_theta":100.,"rope_dim_ratio":0.75,
                    "norm_type":"rms_norm","qk_norm_type":"rms_norm","qk_norm_affine":false,
                    "norm_affine":true,"ffn_use_gated":true,"ffn_activation_fn":"silu"}},
        });
        merged["audio_vae"] = json!({"latent_channels":32,"sample_rate":32000,"output_channel":2,
            "latents_mean":vec![0.;32],"latents_std":vec![1.;32]});
        let metadata = json!({"metadata":{"kwargs":{"decoder_type":"bigvgan","decoder_dim":1024,
            "decoder_rates":[5,5,2,2,2,2,2],"vae_latent_channels":32,"latent_dim":2048}}});
        (merged, index, metadata)
    }

    fn refuses<T: Family>(document: &Value, kind: io::ErrorKind, field: &str) {
        let error = T::parse(document)
            .err()
            .expect("configuration must be refused");
        assert_eq!(error.kind(), kind, "{error}");
        assert!(error.to_string().contains(field), "{error}");
    }

    #[test]
    fn checks_pipeline_partition_scheduler_and_positive_finite_shifts() {
        let (merged, _, _) = documents();
        let checked = Config::parse(&merged).unwrap();
        assert_eq!(
            (checked.shifts().video(), checked.shifts().audio()),
            (2., 1.)
        );
        for (path, value, kind, field) in [
            (
                "/_class_name",
                json!("Other"),
                io::ErrorKind::Unsupported,
                "_class_name",
            ),
            (
                "/_minimax_h3/partition",
                json!("other"),
                io::ErrorKind::Unsupported,
                "partition",
            ),
            (
                "/_minimax_h3/schema_version",
                json!(2),
                io::ErrorKind::Unsupported,
                "schema_version",
            ),
            (
                "/scheduler",
                json!(["lib", "Scheduler"]),
                io::ErrorKind::Unsupported,
                "scheduler",
            ),
            (
                "/_minimax_h3/sigma_shift_scales/video",
                json!(0),
                io::ErrorKind::InvalidData,
                "video",
            ),
            (
                "/_minimax_h3/sigma_shift_scales/audio",
                json!(-1),
                io::ErrorKind::InvalidData,
                "audio",
            ),
            (
                "/_minimax_h3/sigma_shift_scales/video",
                serde_json::from_str("1e9999").unwrap(),
                io::ErrorKind::InvalidData,
                "video",
            ),
        ] {
            let mut document = merged.clone();
            *document.pointer_mut(path).unwrap() = value;
            refuses::<Config>(&document, kind, field);
        }
    }

    #[test]
    fn checks_every_transformer_extent_and_epsilon_before_loading() {
        let (merged, _, _) = documents();
        let source = &merged["transformer"];
        assert_eq!(Transformer::parse(source).unwrap().rope_theta(), 10_000.);
        for key in [
            "hidden_size",
            "num_layers",
            "token_refiner_num_layers",
            "num_attention_heads",
            "attention_head_dim",
            "ffn_hidden_size",
            "latents_dim",
            "audio_latents_dim",
            "text_dim",
            "timestep_input_dim",
            "time_embed_hidden_size",
            "time_embed_dim",
            "adaln_out_features",
            "final_adaln_out_features",
            "rope_inv_freq_len",
        ] {
            let mut document = source.clone();
            document[key] = json!(1);
            refuses::<Transformer>(&document, io::ErrorKind::Unsupported, key);
        }
        for key in ["norm_eps", "qk_norm_eps", "final_norm_eps"] {
            let mut document = source.clone();
            document[key] = json!(1e-6);
            refuses::<Transformer>(&document, io::ErrorKind::Unsupported, key);
        }
        let mut document = source.clone();
        document["patch_size"] = json!([2, 2, 2]);
        refuses::<Transformer>(&document, io::ErrorKind::Unsupported, "patch_size");
        document = source.clone();
        document["hidden_size"] = json!("5376");
        refuses::<Transformer>(&document, io::ErrorKind::InvalidData, "hidden_size");
        for value in [
            json!(0),
            json!(-1),
            json!(1e-300),
            serde_json::from_str("1e9999").unwrap(),
        ] {
            let mut document = source.clone();
            document["rope_theta"] = value;
            refuses::<Transformer>(&document, io::ErrorKind::InvalidData, "rope_theta");
        }
        let mut document = source.clone();
        document["rope_theta"] = json!(100.);
        assert_eq!(Transformer::parse(&document).unwrap().rope_theta(), 100.);
        document["quantization_config"] = json!({"quant_method":"other"});
        refuses::<Transformer>(&document, io::ErrorKind::Unsupported, "quantization_config");
    }

    #[test]
    fn checks_text_geometry_depth_and_interleaved_default_rotation() {
        let (merged, _, _) = documents();
        let source = &merged["text_encoder"]["text_config"];
        let checked = TextEncoder::parse(&merged["text_encoder"]).unwrap();
        assert_eq!(checked.checkpoint_layers(), 64);
        assert_eq!(checked.positions().unwrap().get(), 4096);
        for key in [
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "vocab_size",
        ] {
            let mut document = source.clone();
            document[key] = json!(1);
            refuses::<TextEncoder>(&document, io::ErrorKind::Unsupported, key);
        }
        for (path, value, kind, field) in [
            (
                "/num_hidden_layers",
                json!(49),
                io::ErrorKind::InvalidData,
                "num_hidden_layers",
            ),
            (
                "/rope_theta",
                json!(100.),
                io::ErrorKind::Unsupported,
                "rope_theta",
            ),
            (
                "/rms_norm_eps",
                json!(1e-5),
                io::ErrorKind::Unsupported,
                "rms_norm_eps",
            ),
            (
                "/rope_scaling/mrope_interleaved",
                json!(false),
                io::ErrorKind::Unsupported,
                "mrope_interleaved",
            ),
            (
                "/rope_scaling/rope_type",
                json!("linear"),
                io::ErrorKind::Unsupported,
                "rope_type",
            ),
            (
                "/max_position_embeddings",
                json!(0),
                io::ErrorKind::InvalidData,
                "max_position_embeddings",
            ),
        ] {
            let mut document = source.clone();
            *document.pointer_mut(path).unwrap() = value;
            refuses::<TextEncoder>(&document, kind, field);
        }
        let mut document = source.clone();
        document["num_hidden_layers"] = json!(50);
        document.as_object_mut().unwrap().remove("rope_scaling");
        TextEncoder::parse(&document).unwrap();
        for (field, value) in [
            ("hidden_act", json!("gelu")),
            ("attention_bias", json!(true)),
        ] {
            let mut document = source.clone();
            document[field] = value;
            refuses::<TextEncoder>(&document, io::ErrorKind::Unsupported, field);
        }
        for mut document in [source.clone(), merged["text_encoder"].clone()] {
            document["quantization_config"] = json!({"quant_method":"other"});
            refuses::<TextEncoder>(&document, io::ErrorKind::Unsupported, "quantization_config");
        }
    }

    #[test]
    fn checks_video_wrapper_and_source_geometry_independently() {
        let (merged, _, _) = documents();
        let wrapper = &merged["video_vae"];
        Video::parse(wrapper).unwrap();
        for key in [
            "latent_channels",
            "vae_tile_size",
            "vae_tile_overlap_min",
            "vae_clip_length",
        ] {
            let mut document = wrapper.clone();
            document[key] = json!(1);
            refuses::<Video>(&document, io::ErrorKind::Unsupported, key);
        }
        let source = &wrapper["source"];
        VideoSource::parse(source).unwrap();
        for (path, value) in [
            ("/use_vit_decoder", json!(false)),
            ("/embed_dim", json!(1)),
            ("/vae_ratio", json!(1)),
            ("/vae_ratio_t", json!(1)),
            ("/pixel_norm_type", json!("other")),
            ("/vit_decoder_kwargs/dim_head", json!(1)),
            ("/vit_decoder_kwargs/heads", json!(1)),
            ("/vit_decoder_kwargs/num_layers", json!(1)),
            ("/vit_decoder_kwargs/rope_theta", json!(1.)),
            ("/vit_decoder_kwargs/rope_dim_ratio", json!(1.)),
            ("/vit_decoder_kwargs/norm_type", json!("other")),
            ("/vit_decoder_kwargs/qk_norm_type", json!("other")),
            ("/vit_decoder_kwargs/qk_norm_affine", json!(true)),
            ("/vit_decoder_kwargs/norm_affine", json!(false)),
            ("/vit_decoder_kwargs/ffn_use_gated", json!(false)),
            ("/vit_decoder_kwargs/ffn_activation_fn", json!("other")),
        ] {
            let mut document = source.clone();
            *document.pointer_mut(path).unwrap() = value;
            refuses::<VideoSource>(
                &document,
                io::ErrorKind::Unsupported,
                path.rsplit('/').next().unwrap(),
            );
        }
    }

    #[test]
    fn checks_audio_wrapper_and_separate_decoder_metadata() {
        let (merged, _, metadata) = documents();
        Audio::parse(&merged["audio_vae"]).unwrap();
        AudioMetadata::parse(&metadata).unwrap();
        for key in ["latent_channels", "sample_rate", "output_channel"] {
            let mut document = merged["audio_vae"].clone();
            document[key] = json!(1);
            refuses::<Audio>(&document, io::ErrorKind::Unsupported, key);
        }
        for (key, value) in [
            ("decoder_type", json!("other")),
            ("decoder_dim", json!(1)),
            ("vae_latent_channels", json!(1)),
            ("latent_dim", json!(1)),
            ("decoder_rates", json!([5, 5, 2])),
        ] {
            let mut document = metadata.clone();
            document["metadata"]["kwargs"][key] = value;
            refuses::<AudioMetadata>(&document, io::ErrorKind::Unsupported, key);
        }
    }

    #[test]
    fn latent_statistics_have_fixed_extents_and_checked_numeric_values() {
        let (merged, _, _) = documents();
        let source = &merged["video_vae"];
        let checked = Video::parse(source).unwrap();
        assert_eq!(checked.statistics().mean(), &[0.; 24]);
        assert_eq!(checked.statistics().std(), &[1.; 24]);
        for (key, values) in [
            ("latents_mean", json!(vec![0.; 23])),
            ("latents_std", json!(vec![1.; 25])),
        ] {
            let mut document = source.clone();
            document[key] = values;
            refuses::<Video>(&document, io::ErrorKind::InvalidData, key);
        }
        for (key, value) in [
            ("latents_mean", serde_json::from_str("1e9999").unwrap()),
            ("latents_std", json!(0.)),
            ("latents_std", json!(-1.)),
            ("latents_std", serde_json::from_str("1e9999").unwrap()),
        ] {
            let mut document = source.clone();
            document[key][2] = value;
            refuses::<Video>(&document, io::ErrorKind::InvalidData, key);
        }
        let mut document = merged["audio_vae"].clone();
        document["latents_std"] = json!(vec![1.; 31]);
        refuses::<Audio>(&document, io::ErrorKind::InvalidData, "latents_std");
        document = source.clone();
        document["latents_mean"][0] = serde_json::from_str("0.1234567890123456789").unwrap();
        assert_eq!(
            Video::parse(&document).unwrap().statistics().mean()[0],
            0.12345678901234568
        );
    }
}
