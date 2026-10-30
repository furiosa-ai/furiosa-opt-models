//! Bamba reads its published checkpoint: the size parses its config and refuses what it does not
//! serve, and the registry chooses the checkpoint's size.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;

use furiosa_opt_engine::Checkpoint;
use furiosa_opt_engine::registry::Implementation;
use furiosa_opt_models::load::LoadError;
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::decoder::hybrid::bamba::Size;
use serde_json::json;

use common::Published;

const BAMBA_9B: Published = Published::new(
    "ibm-ai-platform/Bamba-9B",
    "50ec8766310eae056e05e42ba70df45645912e1e",
);

/// The size of the implementation the registry chooses under `architecture` for `source`.
fn load(architecture: &str, source: &Checkpoint) -> Result<&'static str, LoadError> {
    let (name, implementations) = Implementation::listed(&[architecture])?;
    Implementation::choose(name, implementations, source).map(Implementation::size)
}

// # Strict loads of the published checkpoints

/// Bamba-9B selects its size and loads strictly.
#[test]
// Fails until the registry lists a type for the name.
#[should_panic(expected = "Unsupported")]
fn bamba_9b_loads_strictly() {
    let checkpoint = BAMBA_9B.declared();
    assert_eq!(load("BambaForCausalLM", &checkpoint).unwrap(), "9B");
}

// # Configs

/// The size parses its published checkpoint's config, which names no rotary field, the same
/// rotation in Transformers 5's spelling, and another base, which its kernels read at load.
#[test]
fn size_reads_its_published_config() {
    let published = BAMBA_9B.config();
    Size::<"9B">::parse(&published).unwrap();
    let mut respelled = published.clone();
    respelled["rope_parameters"] =
        json!({"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5});
    Size::<"9B">::parse(&respelled).unwrap();
    let mut rebased = published;
    rebased["rope_theta"] = json!(1000000.0);
    Size::<"9B">::parse(&rebased).unwrap();
}

/// A config the size does not serve is refused, each case by its kind and reason.
#[test]
fn size_refuses_what_it_does_not_serve() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = BAMBA_9B.config();
    let cases = [
        (
            "hidden_size",
            json!(2048),
            InvalidData,
            "hidden_size: 2048, the size 9B holds 4096",
        ),
        (
            "intermediate_size",
            json!(8192),
            InvalidData,
            "intermediate_size: 8192, the size 9B holds 14336",
        ),
        (
            "num_hidden_layers",
            json!(40),
            InvalidData,
            "num_hidden_layers: 40, the size 9B holds 32",
        ),
        (
            "num_key_value_heads",
            json!(4),
            InvalidData,
            "num_key_value_heads: 4, the size 9B holds 8",
        ),
        (
            "num_attention_heads",
            json!(16),
            InvalidData,
            "num_attention_heads: 16, the size 9B holds 32",
        ),
        (
            "attn_rotary_emb",
            json!(128),
            InvalidData,
            "attn_rotary_emb: 128, the size 9B holds 64",
        ),
        (
            "mamba_expand",
            json!(1),
            InvalidData,
            "mamba_expand: 1, the size 9B holds 2",
        ),
        (
            "mamba_n_heads",
            json!(64),
            InvalidData,
            "mamba_n_heads: 64, the size 9B holds 128",
        ),
        (
            "mamba_d_head",
            json!(128),
            InvalidData,
            "mamba_d_head: 128, the size 9B holds 64",
        ),
        (
            "mamba_d_head",
            json!("wide"),
            InvalidData,
            "mamba_d_head: not a count, the size 9B holds 64",
        ),
        (
            "mamba_n_groups",
            json!(8),
            InvalidData,
            "mamba_n_groups: 8, the size 9B holds 1",
        ),
        (
            "mamba_d_state",
            json!(64),
            InvalidData,
            "mamba_d_state: 64, the size 9B holds 128",
        ),
        (
            "mamba_d_conv",
            json!(3),
            InvalidData,
            "mamba_d_conv: 3, the size 9B holds 4",
        ),
        (
            "attn_layer_indices",
            json!([8, 18, 27]),
            InvalidData,
            "attn_layer_indices: [8, 18, 27], the size 9B holds [9, 18, 27]",
        ),
        (
            "vocab_size",
            json!(131072),
            InvalidData,
            "vocab_size: 131072, the size compiles 128256 rows",
        ),
        (
            "vocab_size",
            json!(0),
            InvalidData,
            "vocab_size: 0, the size compiles 128256 rows",
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
            json!(262144),
            Unsupported,
            "max_position_embeddings: past the compiled context of 40960",
        ),
        (
            "hidden_act",
            json!("gelu"),
            Unsupported,
            "hidden_act: Bamba gates its feed-forward and mixer with SiLU",
        ),
        (
            "mamba_proj_bias",
            json!(true),
            Unsupported,
            "mamba_proj_bias: Bamba's mixer projects without biases",
        ),
        (
            "mamba_conv_bias",
            json!(false),
            Unsupported,
            "mamba_conv_bias: Bamba's convolution adds its bias",
        ),
        (
            "rope_scaling",
            json!({"rope_type": "linear", "factor": 2.0}),
            Unsupported,
            "found Linear { factor: 2.0 } rotating 0.5 of each head, compiled the unscaled \
             spectrum over the first 64 of 128 coordinates",
        ),
        (
            "rope_parameters",
            json!({"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}),
            Unsupported,
            "found Linear { factor: 2.0 } rotating 0.5 of each head, compiled the unscaled \
             spectrum over the first 64 of 128 coordinates",
        ),
        (
            "partial_rotary_factor",
            json!(1.0),
            Unsupported,
            "found Default rotating 1 of each head, compiled the unscaled spectrum over the first \
             64 of 128 coordinates",
        ),
        (
            "rope_theta",
            json!(0),
            InvalidData,
            "rope_theta: found 0, expected a positive finite value",
        ),
    ];
    for (field, value, kind, reason) in cases {
        let mut document = published.clone();
        document[field] = value;
        let error = Size::<"9B">::parse(&document).map(drop).unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{field}");
    }
}
