//! The model list refuses Whisper's published checkpoint until a type serves its architecture.
//! Each size reads its own published config alone.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;

use furiosa_opt_engine::{LoadError, LoadOptions};
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::encoder_decoder::whisper;
use serde_json::{Value, json};

use common::Published;

const WHISPER_TINY: Published = Published::new(
    "openai/whisper-tiny",
    "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
);
const WHISPER_TINY_EN: Published = Published::new(
    "openai/whisper-tiny.en",
    "87c7102498dcde7456f24cfd30239ca606ed9063",
);

// # The published checkpoint

/// Whisper-tiny's config names `WhisperForConditionalGeneration`, which the model list refuses: no
/// type serves the name yet.
#[test]
fn whisper_is_refused_until_its_root_exists() {
    match LoadOptions::new().parse(&WHISPER_TINY.checkpoint()) {
        Err(LoadError::Unsupported { name, reason }) => {
            assert_eq!(name, "WhisperForConditionalGeneration");
            assert_eq!(reason, "no model type is implemented");
        }
        other => panic!("{other:?}"),
    }
}

// # Sizes by their published configs

/// `S` parses `config`, the parsed size dropped.
fn parse<S: Family>(config: &Value) -> io::Result<()> {
    S::parse(config).map(drop)
}

/// Every published size: its checkpoint and its parser.
const SIZES: [(Published, fn(&Value) -> io::Result<()>); 2] = [
    (WHISPER_TINY, parse::<whisper::Size<"tiny">>),
    (WHISPER_TINY_EN, parse::<whisper::Size<"tiny.en">>),
];

/// Each size parses its published checkpoint's config, and the other size refuses it by a field
/// they differ in, with both values: one size serves each published checkpoint.
#[test]
fn each_size_reads_its_published_config_alone() {
    let configs: Vec<Value> = SIZES
        .iter()
        .map(|(published, _)| published.config())
        .collect();
    for (served, document) in configs.iter().enumerate() {
        for (size, (published, parse)) in SIZES.iter().enumerate() {
            match (size == served, parse(document)) {
                (true, Ok(())) => {}
                (false, Err(error)) if error.kind() == io::ErrorKind::InvalidData => {
                    refuses_by_difference(&error, document, &configs[size]);
                }
                (_, parsed) => panic!("{published:?}'s size on config {served}: {parsed:?}"),
            }
        }
    }
}

/// That `error`, a size's refusal of `config`, names a field where `config` differs from `own`, the
/// size's own config, with both values.
fn refuses_by_difference(error: &io::Error, config: &serde_json::Value, own: &serde_json::Value) {
    let reason = error.to_string();
    let text = |value: &serde_json::Value| value.as_str().map_or(value.to_string(), str::to_owned);
    let field = reason
        .split_once(": ")
        .map_or(reason.as_str(), |(field, _)| field);
    let (found, held) = (text(&config[field]), text(&own[field]));
    assert_ne!(found, held, "{reason}");
    let named = reason.starts_with(&format!("{field}: {found}, ")) && reason.ends_with(&held);
    assert!(named, "{reason}");
}

/// A config the tiny size does not serve is refused, each case by its kind and reason.
#[test]
fn refuses_what_the_size_does_not_hold() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = WHISPER_TINY.config();
    let cases = [
        (
            "model_type",
            json!("t5"),
            InvalidData,
            "model_type: t5, the size tiny is whisper",
        ),
        (
            "d_model",
            json!(512),
            InvalidData,
            "d_model: 512, the size tiny holds 384",
        ),
        (
            "encoder_layers",
            json!(6),
            InvalidData,
            "encoder_layers: 6, the size tiny holds 4",
        ),
        (
            "decoder_layers",
            json!(6),
            InvalidData,
            "decoder_layers: 6, the size tiny holds 4",
        ),
        (
            "encoder_attention_heads",
            json!(8),
            InvalidData,
            "encoder_attention_heads: 8, the size tiny holds 6",
        ),
        (
            "decoder_attention_heads",
            json!(8),
            InvalidData,
            "decoder_attention_heads: 8, the size tiny holds 6",
        ),
        (
            "encoder_ffn_dim",
            json!(2048),
            InvalidData,
            "encoder_ffn_dim: 2048, the size tiny holds 1536",
        ),
        (
            "decoder_ffn_dim",
            json!(2048),
            InvalidData,
            "decoder_ffn_dim: 2048, the size tiny holds 1536",
        ),
        (
            "num_mel_bins",
            json!(128),
            InvalidData,
            "num_mel_bins: 128, the size tiny holds 80",
        ),
        (
            "max_source_positions",
            json!(3000),
            InvalidData,
            "max_source_positions: 3000, the size tiny holds 1500",
        ),
        (
            "max_target_positions",
            json!(896),
            InvalidData,
            "max_target_positions: 896, the size tiny holds 448",
        ),
        (
            "vocab_size",
            json!(51866),
            InvalidData,
            "vocab_size: 51866, the size tiny holds 51865",
        ),
        (
            "tie_word_embeddings",
            json!(false),
            InvalidData,
            "tie_word_embeddings: false, the size ties: true",
        ),
        (
            "activation_function",
            json!("relu"),
            Unsupported,
            "activation_function: Whisper's feed-forward is GELU",
        ),
        (
            "scale_embedding",
            json!(true),
            Unsupported,
            "scale_embedding: Whisper embeds tokens unscaled",
        ),
    ];
    for (key, value, kind, reason) in cases {
        let mut document = published.clone();
        document[key] = value;
        let error = whisper::Size::<"tiny">::parse(&document)
            .map(drop)
            .unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{key}");
    }
}
