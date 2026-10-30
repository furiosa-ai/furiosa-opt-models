//! The model list refuses LLaVA's published checkpoint until a type serves its architecture. The
//! size reads its published config and refuses what it does not hold.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;

use furiosa_opt_engine::{LoadError, LoadOptions};
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::vision_language::llava;
use serde_json::{Value, json};

use common::Published;

const LLAVA_1_5_7B: Published = Published::new(
    "llava-hf/llava-1.5-7b-hf",
    "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
);

// # The published checkpoint

/// LLaVA-1.5-7B's config names `LlavaForConditionalGeneration`, which the model list refuses: no
/// type serves the name yet.
#[test]
fn llava_is_refused_until_its_root_exists() {
    match LoadOptions::new().parse(&LLAVA_1_5_7B.checkpoint()) {
        Err(LoadError::Unsupported { name, reason }) => {
            assert_eq!(name, "LlavaForConditionalGeneration");
            assert_eq!(reason, "no model type is implemented");
        }
        other => panic!("{other:?}"),
    }
}

// # The size by its published config

/// The size parses its published checkpoint's config, whose text half names no rotary field, and
/// any other spectrum its decoder binds: another base, or a rescaled one.
#[test]
fn size_reads_its_published_config() {
    let published = LLAVA_1_5_7B.config();
    llava::Size::<"1.5-7b">::parse(&published).unwrap();
    for (key, value) in [
        ("rope_theta", json!(1000000.0)),
        (
            "rope_scaling",
            json!({"rope_type": "linear", "factor": 2.0}),
        ),
    ] {
        let mut document = published.clone();
        document["text_config"][key] = value;
        llava::Size::<"1.5-7b">::parse(&document).unwrap();
    }
}

/// A config the size does not serve is refused, each case by its kind and reason, which names a
/// nested key by its dotted path.
#[test]
fn refuses_what_the_size_does_not_hold() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = LLAVA_1_5_7B.config();
    let cases: [(&[&str], Value, io::ErrorKind, &str); 36] = [
        (
            &["model_type"],
            json!("llava_next"),
            InvalidData,
            "model_type: llava_next, the size 1.5-7b is llava",
        ),
        (
            &["image_token_index"],
            json!(32001),
            InvalidData,
            "image_token_index: 32001, the size 1.5-7b holds 32000",
        ),
        (
            &["vision_feature_layer"],
            json!(-1),
            InvalidData,
            "vision_feature_layer: -1, the size 1.5-7b reads -2",
        ),
        (
            &["vision_feature_select_strategy"],
            json!("full"),
            Unsupported,
            "vision_feature_select_strategy: the size drops the class row",
        ),
        (
            &["projector_hidden_act"],
            json!("silu"),
            Unsupported,
            "projector_hidden_act: the projector is GELU",
        ),
        (
            &["multimodal_projector_bias"],
            json!(false),
            InvalidData,
            "multimodal_projector_bias: false, the size's projector has biases",
        ),
        (
            &["tie_word_embeddings"],
            json!(true),
            InvalidData,
            "tie_word_embeddings: true, the size 1.5-7b ties: false",
        ),
        (
            &["text_config", "model_type"],
            json!("mistral"),
            InvalidData,
            "text_config.model_type: mistral, the decoder is llama",
        ),
        (
            &["text_config", "hidden_size"],
            json!(5120),
            InvalidData,
            "text_config.hidden_size: 5120, the decoder holds 4096",
        ),
        (
            &["text_config", "intermediate_size"],
            json!(13824),
            InvalidData,
            "text_config.intermediate_size: 13824, the decoder holds 11008",
        ),
        (
            &["text_config", "num_hidden_layers"],
            json!(40),
            InvalidData,
            "text_config.num_hidden_layers: 40, the decoder holds 32",
        ),
        (
            &["text_config", "num_attention_heads"],
            json!(40),
            InvalidData,
            "text_config.num_attention_heads: 40, the decoder holds 32",
        ),
        (
            &["text_config", "num_key_value_heads"],
            json!(8),
            InvalidData,
            "text_config.num_key_value_heads: 8, the decoder holds 32",
        ),
        (
            &["text_config", "head_dim"],
            json!(64),
            InvalidData,
            "text_config.head_dim: 64, the decoder holds 128",
        ),
        (
            &["text_config", "vocab_size"],
            json!(32128),
            InvalidData,
            "text_config.vocab_size: 32128, the decoder compiles 32064 rows",
        ),
        (
            &["text_config", "vocab_size"],
            json!(0),
            InvalidData,
            "text_config.vocab_size: 0, the decoder compiles 32064 rows",
        ),
        (
            &["text_config", "rope_parameters"],
            json!({"rope_type": "linear"}),
            InvalidData,
            "text_config: missing field `factor`",
        ),
        (
            &["text_config", "rms_norm_eps"],
            json!(1e-6),
            InvalidData,
            "text_config.rms_norm_eps: 0.000001, the decoder holds 0.00001",
        ),
        (
            &["text_config", "rope_theta"],
            json!(0),
            InvalidData,
            "text_config: rope_theta: found 0, expected a positive finite value",
        ),
        (
            &["text_config", "tie_word_embeddings"],
            json!(true),
            InvalidData,
            "text_config.tie_word_embeddings: true, the decoder's head is its own",
        ),
        (
            &["text_config", "max_position_embeddings"],
            json!(8192),
            Unsupported,
            "text_config.max_position_embeddings: past the decoder's 4096 positions",
        ),
        (
            &["text_config", "rope_scaling"],
            json!({"rope_type": "dynamic", "factor": 2.0}),
            Unsupported,
            "text_config: found rope_type \"dynamic\", compiled default, proportional, linear, \
             llama3 or yarn",
        ),
        (
            &["text_config", "partial_rotary_factor"],
            json!(0.5),
            Unsupported,
            "text_config.partial_rotary_factor: found 0.5 of a 128-wide head, compiled 128 rotated \
             coordinates",
        ),
        (
            &["text_config", "hidden_act"],
            json!("gelu"),
            Unsupported,
            "text_config.hidden_act: the decoder's feed-forward is SiLU",
        ),
        (
            &["text_config", "attention_bias"],
            json!(true),
            Unsupported,
            "text_config.attention_bias: the decoder projects without biases",
        ),
        (
            &["text_config", "mlp_bias"],
            json!(true),
            Unsupported,
            "text_config.mlp_bias: the decoder projects without biases",
        ),
        (
            &["vision_config", "model_type"],
            json!("siglip_vision_model"),
            InvalidData,
            "vision_config.model_type: siglip_vision_model, the tower is clip_vision_model",
        ),
        (
            &["vision_config", "hidden_size"],
            json!(768),
            InvalidData,
            "vision_config.hidden_size: 768, the tower holds 1024",
        ),
        (
            &["vision_config", "intermediate_size"],
            json!(3072),
            InvalidData,
            "vision_config.intermediate_size: 3072, the tower holds 4096",
        ),
        (
            &["vision_config", "num_hidden_layers"],
            json!(12),
            InvalidData,
            "vision_config.num_hidden_layers: 12, the tower holds 24",
        ),
        (
            &["vision_config", "num_attention_heads"],
            json!(12),
            InvalidData,
            "vision_config.num_attention_heads: 12, the tower holds 16",
        ),
        (
            &["vision_config", "image_size"],
            json!(224),
            InvalidData,
            "vision_config.image_size: 224, the tower holds 336",
        ),
        (
            &["vision_config", "patch_size"],
            json!(32),
            InvalidData,
            "vision_config.patch_size: 32, the tower holds 14",
        ),
        (
            &["vision_config", "num_channels"],
            json!(1),
            InvalidData,
            "vision_config.num_channels: 1, the tower holds 3",
        ),
        (
            &["vision_config", "layer_norm_eps"],
            json!(1e-6),
            InvalidData,
            "vision_config.layer_norm_eps: 0.000001, the tower holds 0.00001",
        ),
        (
            &["vision_config", "hidden_act"],
            json!("gelu"),
            Unsupported,
            "vision_config.hidden_act: the tower's feed-forward is quick GELU",
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
        let error = llava::Size::<"1.5-7b">::parse(&document)
            .map(drop)
            .unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{name}");
    }
}

/// An absent key of the text half holds `LlamaConfig`'s default: without `rms_norm_eps` the text
/// config declares 1e-6, which the size does not hold.
#[test]
fn absent_text_key_holds_llama_default() {
    let mut document = LLAVA_1_5_7B.config();
    let text = document["text_config"].as_object_mut().unwrap();
    text.remove("rms_norm_eps");
    let error = llava::Size::<"1.5-7b">::parse(&document)
        .map(drop)
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
    let reason = "text_config.rms_norm_eps: 0.000001, the decoder holds 0.00001";
    assert_eq!(error.to_string(), reason);
}
