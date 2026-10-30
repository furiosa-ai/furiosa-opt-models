//! Mamba2 reads its published checkpoint: the size parses its config and refuses what it does not
//! serve, and the registry chooses the checkpoint's size.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;

use furiosa_opt_engine::Checkpoint;
use furiosa_opt_models::load::LoadError;
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::decoder::recurrent::mamba2::Size;
use serde_json::json;

use common::{Published, size};

const MAMBA2_130M: Published = Published::new(
    "AntonV/mamba2-130m-hf",
    "05e8773fc4ac1cd067e8a18a5c45372ce5178405",
);
const MAMBA2_130M_ORIGINAL: Published = Published::new(
    "state-spaces/mamba2-130m",
    "ea6060f68a4289e9c06f80effa896629ba519216",
);

// # Strict loads of the published checkpoints

/// Mamba2-130m selects its size and loads strictly, its head tied to the embedding.
#[test]
// Fails until the registry lists a type for the name.
#[should_panic(expected = "Unsupported")]
fn mamba2_130m_loads_strictly() {
    let checkpoint = MAMBA2_130M.declared();
    assert_eq!(size("Mamba2ForCausalLM", &checkpoint).unwrap(), "130m");
}

// # Configs

/// The size parses its published checkpoint's config; the original layout's config, which
/// Transformers does not read, is refused: it names no `hidden_size`.
#[test]
fn size_reads_its_published_config() {
    Size::<"130m">::parse(&MAMBA2_130M.config()).unwrap();
    let error = Size::<"130m">::parse(&MAMBA2_130M_ORIGINAL.config())
        .map(drop)
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
    assert_eq!(error.to_string(), "missing field `hidden_size`");
}

/// A config the size does not serve is refused, each case by its kind and reason.
#[test]
fn size_refuses_what_it_does_not_serve() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = MAMBA2_130M.config();
    let cases = [
        (
            "hidden_size",
            json!(1024),
            InvalidData,
            "hidden_size: 1024, the size 130m holds 768",
        ),
        (
            "num_hidden_layers",
            json!(48),
            InvalidData,
            "num_hidden_layers: 48, the size 130m holds 24",
        ),
        (
            "expand",
            json!(4),
            InvalidData,
            "expand: 4, the size 130m holds 2",
        ),
        (
            "num_heads",
            json!(48),
            InvalidData,
            "num_heads: 48, the size 130m holds 24",
        ),
        (
            "head_dim",
            json!(32),
            InvalidData,
            "head_dim: 32, the size 130m holds 64",
        ),
        (
            "n_groups",
            json!(8),
            InvalidData,
            "n_groups: 8, the size 130m holds 1",
        ),
        (
            "state_size",
            json!(64),
            InvalidData,
            "state_size: 64, the size 130m holds 128",
        ),
        (
            "conv_kernel",
            json!(3),
            InvalidData,
            "conv_kernel: 3, the size 130m holds 4",
        ),
        (
            "vocab_size",
            json!(50304),
            InvalidData,
            "vocab_size: 50304, the size compiles 50288 rows",
        ),
        (
            "vocab_size",
            json!(0),
            InvalidData,
            "vocab_size: 0, the size compiles 50288 rows",
        ),
        (
            "layer_norm_epsilon",
            json!(1e-6),
            InvalidData,
            "layer_norm_epsilon: 0.000001, another than the size's",
        ),
        (
            "tie_word_embeddings",
            json!(false),
            InvalidData,
            "tie_word_embeddings: false, the size ties: true",
        ),
        (
            "hidden_act",
            json!("gelu"),
            Unsupported,
            "hidden_act: Mamba2 gates its mixer with SiLU",
        ),
        (
            "use_bias",
            json!(true),
            Unsupported,
            "use_bias: Mamba2 projects without biases",
        ),
        (
            "use_conv_bias",
            json!(false),
            Unsupported,
            "use_conv_bias: Mamba2's convolution adds its bias",
        ),
        (
            "residual_in_fp32",
            json!(false),
            Unsupported,
            "residual_in_fp32: Mamba2 keeps its residual rows in FP32",
        ),
        (
            "rms_norm",
            json!(false),
            Unsupported,
            "rms_norm: Mamba2 normalizes its gated mixer output",
        ),
        (
            "time_step_limit",
            json!([0.0, 0.1]),
            Unsupported,
            "time_step_limit: Mamba2 clamps no time step",
        ),
    ];
    for (field, value, kind, reason) in cases {
        let mut document = published.clone();
        document[field] = value;
        let error = Size::<"130m">::parse(&document).map(drop).unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{field}");
    }
    let mut clamped = published.clone();
    clamped["time_step_limit"][0] = json!(0.5);
    let error = Size::<"130m">::parse(&clamped).map(drop).unwrap_err();
    let refused = "time_step_limit: Mamba2 clamps no time step".to_owned();
    assert_eq!((error.kind(), error.to_string()), (Unsupported, refused));
}

/// A Mamba2 checkpoint is read at its root once its size parses its config; a config the size does
/// not serve is refused under `config.json`, by the kind of its refusal.
#[test]
fn scope_reads_the_config_of_its_size() {
    use furiosa_opt_models::models::Implements;
    use furiosa_opt_models::models::decoder::recurrent::ForCausalLM;
    let Some(published) = MAMBA2_130M.config().as_object().cloned() else {
        panic!("a config object");
    };
    let directory = tempfile::tempdir().unwrap();
    let scope = |edit: &[(&str, serde_json::Value)]| {
        let mut config = published.clone();
        config.extend(
            edit.iter()
                .map(|(key, value)| (key.to_string(), value.clone())),
        );
        let checkpoint = Checkpoint::new(directory.path(), Some(config.into())).unwrap();
        ForCausalLM::<Size<"130m">, (), ()>::scope(&checkpoint)
    };
    assert_eq!(scope(&[]).unwrap(), "");
    match scope(&[("use_bias", json!(true))]) {
        Err(LoadError::Unsupported { name, reason }) => assert_eq!(
            (name.as_str(), reason.as_str()),
            ("config.json", "use_bias: Mamba2 projects without biases")
        ),
        other => panic!("{other:?}"),
    }
    match scope(&[("hidden_size", json!(1024))]) {
        Err(LoadError::InvalidData { name, reason }) => assert_eq!(
            (name.as_str(), reason.as_str()),
            ("config.json", "hidden_size: 1024, the size 130m holds 768")
        ),
        other => panic!("{other:?}"),
    }
}
