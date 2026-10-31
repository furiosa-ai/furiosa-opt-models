//! The model list refuses the published T5 and mT5 checkpoints until a type serves their
//! architectures. Each size reads its own published config alone.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;

use furiosa_opt_engine::{Config, LoadError, LoadOptions};
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::encoder_decoder::{mt5, t5};
use serde_json::{Value, json};

use common::Published;

const T5_SMALL: Published = Published::new(
    "google-t5/t5-small",
    "df1b051c49625cf57a3d0d8d3863ed4d13564fe4",
);
const T5_V1_1_SMALL: Published = Published::new(
    "google/t5-v1_1-small",
    "8a88af75516269158a3aa488d1abdfd3d5e4ee49",
);
const MT5_SMALL: Published = Published::new(
    "google/mt5-small",
    "73fb5dbe4756edadc8fbe8c769b0a109493acf7a",
);
const MT5_BASE: Published = Published::new(
    "google/mt5-base",
    "2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f",
);
const T5_BASE: Published = Published::new(
    "google-t5/t5-base",
    "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1",
);
const T5_LARGE: Published = Published::new(
    "google-t5/t5-large",
    "150ebc2c4b72291e770f58e6057481c8d2ed331a",
);
const T5_V1_1_BASE: Published = Published::new(
    "google/t5-v1_1-base",
    "b5fc947a416ea3cb079532cb3c2bbadeb7f800fc",
);
const T5_V1_1_LARGE: Published = Published::new(
    "google/t5-v1_1-large",
    "a98b0fcd0b8137ded40cdf0c0cf0ee884e7c9726",
);
const T5_V1_1_XL: Published = Published::new(
    "google/t5-v1_1-xl",
    "e827710dfd12562d3ef5ea28853b2d0b31859bc6",
);
const T5_V1_1_XXL: Published = Published::new(
    "google/t5-v1_1-xxl",
    "3db67ab1af984cf10548a73467f0e5bca2aaaeb2",
);
const MT5_LARGE: Published = Published::new(
    "google/mt5-large",
    "50b7223e98fcd124b0cabb1ec81bc6324c7df107",
);

// # The published checkpoints

/// Assert that `parsed` is the model list's refusal of `architecture`, a name no type serves yet.
fn assert_refused(parsed: Result<Config, LoadError>, architecture: &str) {
    match parsed {
        Err(LoadError::Unsupported { name, reason }) => {
            assert_eq!(name, architecture);
            assert_eq!(reason, "no model type is implemented");
        }
        other => panic!("{other:?}"),
    }
}

/// T5-small (v1.0) and T5 v1.1-small name `T5ForConditionalGeneration`, which the model list
/// refuses: no type serves the name yet.
#[test]
fn t5_is_refused_until_its_root_exists() {
    for published in [T5_SMALL, T5_V1_1_SMALL] {
        let parsed = LoadOptions::new().parse(&published.checkpoint());
        assert_refused(parsed, "T5ForConditionalGeneration");
    }
}

/// mT5-small names `MT5ForConditionalGeneration`, which the model list refuses: no type serves the
/// name yet.
#[test]
fn mt5_is_refused_until_its_root_exists() {
    let parsed = LoadOptions::new().parse(&MT5_SMALL.checkpoint());
    assert_refused(parsed, "MT5ForConditionalGeneration");
}

/// Choosing `T5EncoderModel` for a conditional checkpoint's encoder is refused: no type serves the
/// name yet.
#[test]
fn t5_encoder_is_refused_until_its_root_exists() {
    let options = LoadOptions::new().architecture("T5EncoderModel");
    let parsed = options.and_then(|options| options.parse(&T5_V1_1_SMALL.checkpoint()));
    assert_refused(parsed, "T5EncoderModel");
}

// # Sizes by their published configs

/// `S` parses `config`, the parsed size dropped.
fn parse<S: Family>(config: &Value) -> io::Result<()> {
    S::parse(config).map(drop)
}

/// Every published size: its checkpoint and its parser.
const SIZES: [(Published, fn(&Value) -> io::Result<()>); 11] = [
    (T5_SMALL, parse::<t5::Size<"small">>),
    (T5_BASE, parse::<t5::Size<"base">>),
    (T5_LARGE, parse::<t5::Size<"large">>),
    (T5_V1_1_SMALL, parse::<t5::Size<"v1_1-small">>),
    (T5_V1_1_BASE, parse::<t5::Size<"v1_1-base">>),
    (T5_V1_1_LARGE, parse::<t5::Size<"v1_1-large">>),
    (T5_V1_1_XL, parse::<t5::Size<"v1_1-xl">>),
    (T5_V1_1_XXL, parse::<t5::Size<"v1_1-xxl">>),
    (MT5_SMALL, parse::<mt5::Size<"small">>),
    (MT5_BASE, parse::<mt5::Size<"base">>),
    (MT5_LARGE, parse::<mt5::Size<"large">>),
];

/// Each size parses its published checkpoint's config, and every other size refuses it by a field
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
                (false, Err(error)) if error.kind() == io::ErrorKind::Unsupported => {
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

/// A config the v1.0 small size does not serve is refused, each case by its kind and reason.
#[test]
fn refuses_what_the_size_does_not_hold() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = T5_SMALL.config();
    let cases = [
        (
            "model_type",
            json!("mt5"),
            Unsupported,
            "model_type: mt5, the size small is t5",
        ),
        (
            "d_model",
            json!(768),
            Unsupported,
            "d_model: 768, the size small holds 512",
        ),
        (
            "d_ff",
            json!(1024),
            Unsupported,
            "d_ff: 1024, the size small holds 2048",
        ),
        (
            "d_kv",
            json!(128),
            Unsupported,
            "d_kv: 128, the size small holds 64",
        ),
        (
            "num_heads",
            json!(6),
            Unsupported,
            "num_heads: 6, the size small holds 8",
        ),
        (
            "num_layers",
            json!(8),
            Unsupported,
            "num_layers: 8, the size small holds 6",
        ),
        (
            "num_decoder_layers",
            json!(8),
            Unsupported,
            "num_decoder_layers: 8, the size small holds 6",
        ),
        (
            "vocab_size",
            json!(250112),
            Unsupported,
            "vocab_size: 250112, the size holds 1 to 32128",
        ),
        (
            "vocab_size",
            json!(0),
            InvalidData,
            "vocab_size: 0, the size holds 1 to 32128",
        ),
        (
            "relative_attention_num_buckets",
            json!(64),
            Unsupported,
            "relative_attention_num_buckets: 64, the size small holds 32",
        ),
        (
            "relative_attention_max_distance",
            json!(256),
            Unsupported,
            "relative_attention_max_distance: 256, the size small holds 128",
        ),
        (
            "layer_norm_epsilon",
            json!(1e-5),
            Unsupported,
            "layer_norm_epsilon: 0.00001, the size small holds 1e-6",
        ),
        (
            "feed_forward_proj",
            json!("gated-gelu"),
            Unsupported,
            "feed_forward_proj: gated-gelu, the size small holds relu",
        ),
        (
            "feed_forward_proj",
            json!("gated-silu"),
            Unsupported,
            "feed_forward_proj: gated-silu, T5 serves relu and gated-gelu",
        ),
        (
            "tie_word_embeddings",
            json!(false),
            Unsupported,
            "tie_word_embeddings: false, the size small ties: true",
        ),
        (
            "scale_decoder_outputs",
            json!(false),
            Unsupported,
            "scale_decoder_outputs: false, the size small ties: true",
        ),
    ];
    for (key, value, kind, reason) in cases {
        let mut document = published.clone();
        document[key] = value;
        let error = t5::Size::<"small">::parse(&document).map(drop).unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{key}");
    }
    // `T5Config` ties by default, which the untied v1.1 sizes do not.
    let mut untied = T5_V1_1_SMALL.config();
    untied
        .as_object_mut()
        .unwrap()
        .remove("tie_word_embeddings");
    let error = t5::Size::<"v1_1-small">::parse(&untied)
        .map(drop)
        .unwrap_err();
    let refused = "tie_word_embeddings: true, the size v1_1-small ties: false".to_owned();
    assert_eq!((error.kind(), error.to_string()), (Unsupported, refused));
}

/// An absent key holds the default of the config class `model_type` names: an mT5-base config
/// without `d_ff` declares `MT5Config`'s 1024, which the size does not hold, not `T5Config`'s 2048,
/// which it does.
#[test]
fn absent_key_holds_its_config_class_default() {
    let mut document = MT5_BASE.config();
    document.as_object_mut().unwrap().remove("d_ff");
    let error = mt5::Size::<"base">::parse(&document).map(drop).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::Unsupported, "{error}");
    assert_eq!(error.to_string(), "d_ff: 1024, the size base holds 2048");
}
