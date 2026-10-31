//! Mixtral reads its published checkpoint: the size parses its config and refuses what it does not
//! serve, and the registry chooses the checkpoint's size.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;

use furiosa_opt_engine::Checkpoint;
use furiosa_opt_engine::registry::Implementation;
use furiosa_opt_models::load::LoadError;
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::decoder::attention::mixtral::Size;
use serde_json::json;

use common::Published;

const MIXTRAL_8X7B: Published = Published::new(
    "mistralai/Mixtral-8x7B-v0.1",
    "fc7ac94680e38d7348cfa806e51218e6273104b0",
);

/// The size of the implementation the registry chooses under `architecture` for `source`.
fn load(architecture: &str, source: &Checkpoint) -> Result<&'static str, LoadError> {
    let (name, implementations) = Implementation::listed(&[architecture])?;
    Implementation::choose(name, implementations, source).map(Implementation::size)
}

// # Strict loads of the published checkpoints

/// Mixtral-8x7B selects its size and loads strictly.
#[test]
// Fails until the registry lists a type for the name.
#[should_panic(expected = "Unsupported")]
fn mixtral_8x7b_loads_strictly() {
    let checkpoint = MIXTRAL_8X7B.declared();
    assert_eq!(load("MixtralForCausalLM", &checkpoint).unwrap(), "8x7B");
}

// # Configs

/// The size parses its published checkpoint's config.
#[test]
fn size_reads_its_published_config() {
    Size::<"8x7B">::parse(&MIXTRAL_8X7B.config()).unwrap();
}

/// A config the size does not serve is refused, each case by its kind and reason.
#[test]
fn size_refuses_what_it_does_not_serve() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = MIXTRAL_8X7B.config();
    let cases = [
        (
            "hidden_size",
            json!(6144),
            InvalidData,
            "hidden_size: 6144, the size 8x7B holds 4096",
        ),
        (
            "intermediate_size",
            json!(16384),
            InvalidData,
            "intermediate_size: 16384, the size 8x7B holds 14336",
        ),
        (
            "num_hidden_layers",
            json!(56),
            InvalidData,
            "num_hidden_layers: 56, the size 8x7B holds 32",
        ),
        (
            "num_key_value_heads",
            json!(4),
            InvalidData,
            "num_key_value_heads: 4, the size 8x7B holds 8",
        ),
        (
            "num_attention_heads",
            json!(48),
            InvalidData,
            "num_attention_heads: 48, the size 8x7B holds 32",
        ),
        (
            "head_dim",
            json!(64),
            InvalidData,
            "head_dim: 64, the size 8x7B holds 128",
        ),
        (
            "num_local_experts",
            json!(16),
            InvalidData,
            "num_local_experts: 16, the size 8x7B holds 8",
        ),
        (
            "num_experts_per_tok",
            json!(1),
            InvalidData,
            "num_experts_per_tok: 1, the size 8x7B holds 2",
        ),
        (
            "vocab_size",
            json!(32768),
            Unsupported,
            "vocab_size: 32768, the size holds 1 to 32000",
        ),
        (
            "vocab_size",
            json!(0),
            InvalidData,
            "vocab_size: 0, the size holds 1 to 32000",
        ),
        (
            "rms_norm_eps",
            json!(1e-6),
            InvalidData,
            "rms_norm_eps: 0.000001, another than the size's",
        ),
        (
            "tie_word_embeddings",
            json!(true),
            InvalidData,
            "tie_word_embeddings: true, the size ties: false",
        ),
        (
            "max_position_embeddings",
            json!(65536),
            Unsupported,
            "max_position_embeddings: past the compiled context of 40960",
        ),
        (
            "sliding_window",
            json!(4096),
            Unsupported,
            "sliding_window: Mixtral attends the whole history",
        ),
        (
            "rope_scaling",
            json!({"type": "linear", "factor": 2.0}),
            Unsupported,
            "found Linear { factor: 2.0 } rotating 1 of each head, compiled the unscaled spectrum over the whole head",
        ),
        (
            "hidden_act",
            json!("gelu"),
            Unsupported,
            "hidden_act: Mixtral's experts gate with SiLU",
        ),
    ];
    for (field, value, kind, reason) in cases {
        let mut document = published.clone();
        document[field] = value;
        let error = Size::<"8x7B">::parse(&document).map(drop).unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{field}");
    }
}
