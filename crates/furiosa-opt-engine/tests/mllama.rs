//! The model list refuses Mllama's published checkpoint until a type serves its architecture. The
//! size reads its published config and refuses what it does not hold.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;

use furiosa_opt_engine::{LoadError, LoadOptions};
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::vision_language::mllama;
use serde_json::{Value, json};

use common::Published;

const MLLAMA_11B: Published = Published::new(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    "677b0c1b7008230a0fb88708c5550748e72b9a83",
);

// # The published checkpoint

/// Llama 3.2 11B Vision's config names `MllamaForConditionalGeneration`, which the model list
/// refuses: no type serves the name yet.
#[test]
fn mllama_is_refused_until_its_root_exists() {
    match LoadOptions::new().parse(&MLLAMA_11B.checkpoint()) {
        Err(LoadError::Unsupported { name, reason }) => {
            assert_eq!(name, "MllamaForConditionalGeneration");
            assert_eq!(reason, "no model type is implemented");
        }
        other => panic!("{other:?}"),
    }
}

// # The size by its published config

/// The size parses its published checkpoint's config, and any other spectrum its text model
/// binds: another base, another Llama 3 rescaling, none, or no base, which reads as
/// `MllamaTextConfig`'s.
#[test]
fn size_reads_its_published_config() {
    let published = MLLAMA_11B.config();
    mllama::Size::<"11B">::parse(&published).unwrap();
    for (key, value) in [
        ("rope_theta", json!(10000.0)),
        ("rope_scaling", json!(null)),
        (
            "rope_scaling",
            json!({
                "rope_type": "llama3", "factor": 16.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
            }),
        ),
    ] {
        let mut document = published.clone();
        document["text_config"][key] = value;
        mllama::Size::<"11B">::parse(&document).unwrap();
    }
    let mut baseless = published;
    baseless["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("rope_theta");
    mllama::Size::<"11B">::parse(&baseless).unwrap();
}

/// A config the size does not serve is refused, each case by its kind and reason, which names a
/// nested key by its dotted path.
#[test]
fn refuses_what_the_size_does_not_hold() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = MLLAMA_11B.config();
    let cases: [(&[&str], Value, io::ErrorKind, &str); 34] = [
        (
            &["model_type"],
            json!("llava"),
            InvalidData,
            "model_type: llava, the size 11B is mllama",
        ),
        (
            &["image_token_index"],
            json!(128257),
            InvalidData,
            "image_token_index: 128257, the size 11B holds 128256",
        ),
        (
            &["text_config", "model_type"],
            json!("llama"),
            InvalidData,
            "text_config.model_type: llama, the text model is mllama_text_model",
        ),
        (
            &["text_config", "hidden_size"],
            json!(8192),
            InvalidData,
            "text_config.hidden_size: 8192, the size holds 4096",
        ),
        (
            &["text_config", "intermediate_size"],
            json!(28672),
            InvalidData,
            "text_config.intermediate_size: 28672, the size holds 14336",
        ),
        (
            &["text_config", "num_hidden_layers"],
            json!(100),
            InvalidData,
            "text_config.num_hidden_layers: 100, the size holds 40",
        ),
        (
            &["text_config", "num_attention_heads"],
            json!(64),
            InvalidData,
            "text_config.num_attention_heads: 64, the size holds 32",
        ),
        (
            &["text_config", "num_key_value_heads"],
            json!(16),
            InvalidData,
            "text_config.num_key_value_heads: 16, the size holds 8",
        ),
        (
            &["text_config", "vocab_size"],
            json!(128264),
            InvalidData,
            "text_config.vocab_size: 128264, the size holds 128256",
        ),
        (
            &["text_config", "cross_attention_layers"],
            json!([3, 8, 13, 18]),
            InvalidData,
            "text_config.cross_attention_layers: [3, 8, 13, 18], the size holds [3, 8, 13, 18, 23, 28, 33, 38]",
        ),
        (
            &["text_config", "rms_norm_eps"],
            json!(1e-6),
            InvalidData,
            "text_config.rms_norm_eps: 0.000001, the size holds 0.00001",
        ),
        (
            &["text_config", "tie_word_embeddings"],
            json!(true),
            InvalidData,
            "text_config.tie_word_embeddings: true, the text model's head is its own",
        ),
        (
            &["text_config", "max_position_embeddings"],
            json!(262144),
            Unsupported,
            "text_config.max_position_embeddings: past the text model's 131072 positions",
        ),
        (
            &["text_config", "hidden_act"],
            json!("gelu"),
            Unsupported,
            "text_config.hidden_act: the text model's feed-forward is SiLU",
        ),
        (
            &["text_config", "rope_parameters"],
            json!({"rope_type": "linear"}),
            InvalidData,
            "text_config: missing field `factor`",
        ),
        (
            &["text_config", "rope_theta"],
            json!(0),
            InvalidData,
            "text_config: rope_theta: found 0, expected a positive finite value",
        ),
        (
            &["text_config", "rope_scaling"],
            json!({"rope_type": "dynamic", "factor": 2.0}),
            Unsupported,
            "text_config: found rope_type \"dynamic\", compiled default, proportional, linear, \
             llama3 or yarn",
        ),
        (
            &["text_config", "rope_scaling"],
            json!({
                "rope_type": "yarn", "factor": 4.0, "attention_factor": 2.0,
                "original_max_position_embeddings": 8192,
            }),
            Unsupported,
            "text_config.attention_factor: found 2.0, compiled 1.0",
        ),
        (
            &["text_config", "partial_rotary_factor"],
            json!(0.5),
            Unsupported,
            "text_config.partial_rotary_factor: found 0.5 of a 128-wide head, compiled 128 rotated \
             coordinates",
        ),
        (
            &["vision_config", "model_type"],
            json!("clip_vision_model"),
            InvalidData,
            "vision_config.model_type: clip_vision_model, the tower is mllama_vision_model",
        ),
        (
            &["vision_config", "hidden_size"],
            json!(1024),
            InvalidData,
            "vision_config.hidden_size: 1024, the size holds 1280",
        ),
        (
            &["vision_config", "intermediate_size"],
            json!(4096),
            InvalidData,
            "vision_config.intermediate_size: 4096, the size holds 5120",
        ),
        (
            &["vision_config", "attention_heads"],
            json!(8),
            InvalidData,
            "vision_config.attention_heads: 8, the size holds 16",
        ),
        (
            &["vision_config", "num_hidden_layers"],
            json!(24),
            InvalidData,
            "vision_config.num_hidden_layers: 24, the size holds 32",
        ),
        (
            &["vision_config", "num_global_layers"],
            json!(4),
            InvalidData,
            "vision_config.num_global_layers: 4, the size holds 8",
        ),
        (
            &["vision_config", "image_size"],
            json!(448),
            InvalidData,
            "vision_config.image_size: 448, the size holds 560",
        ),
        (
            &["vision_config", "patch_size"],
            json!(16),
            InvalidData,
            "vision_config.patch_size: 16, the size holds 14",
        ),
        (
            &["vision_config", "num_channels"],
            json!(1),
            InvalidData,
            "vision_config.num_channels: 1, the size holds 3",
        ),
        (
            &["vision_config", "max_num_tiles"],
            json!(8),
            InvalidData,
            "vision_config.max_num_tiles: 8, the size holds 4",
        ),
        (
            &["vision_config", "vision_output_dim"],
            json!(5120),
            InvalidData,
            "vision_config.vision_output_dim: 5120, the size holds 7680",
        ),
        (
            &["vision_config", "norm_eps"],
            json!(1e-6),
            InvalidData,
            "vision_config.norm_eps: 0.000001, the size holds 0.00001",
        ),
        (
            &["vision_config", "intermediate_layers_indices"],
            json!([3, 7, 15, 23]),
            InvalidData,
            "vision_config.intermediate_layers_indices: [3, 7, 15, 23], the size holds [3, 7, 15, 23, 30]",
        ),
        (
            &["vision_config", "supported_aspect_ratios"],
            json!([[1, 1], [2, 2]]),
            InvalidData,
            "vision_config.supported_aspect_ratios: [[1, 1], [2, 2]], the size holds [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]",
        ),
        (
            &["vision_config", "hidden_act"],
            json!("quick_gelu"),
            Unsupported,
            "vision_config.hidden_act: the tower's feed-forward is GELU",
        ),
    ];
    for (path, value, kind, reason) in cases {
        let mut document = published.clone();
        let (key, parents) = path.split_last().unwrap();
        let parent = parents
            .iter()
            .fold(&mut document, |part, key| &mut part[*key]);
        parent[*key] = value;
        let name = path.join(".");
        let error = mllama::Size::<"11B">::parse(&document)
            .map(drop)
            .unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{name}");
    }
}
