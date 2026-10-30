//! OLMoE reads its published checkpoint: the size parses its config and refuses what it does not
//! serve, and the checkpoint loads in BF16 strictly through the model list, on host: every name its
//! line reads is checked by name, dtype and shape, and a name it lacks or does not read is refused.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;

use furiosa_opt_engine::Checkpoint;
use furiosa_opt_engine::registry::Implementation;
use furiosa_opt_models::load::LoadError;
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::decoder::attention::olmoe::Size;
use safetensors::Dtype;
use serde_json::json;

use common::Published;

const OLMOE_1B_7B: Published = Published::new(
    "allenai/OLMoE-1B-7B-0924",
    "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
);

/// The size of the implementation the registry chooses under `architecture` for `source`.
fn load(architecture: &str, source: &Checkpoint) -> Result<&'static str, LoadError> {
    let (name, implementations) = Implementation::listed(&[architecture])?;
    Implementation::choose(name, implementations, source).map(Implementation::size)
}

// # Strict loads of the published checkpoints

/// OLMoE-1B-7B selects its size and loads strictly.
#[test]
// Fails until the registry lists a type for the name.
#[should_panic(expected = "Unsupported")]
fn olmoe_1b_7b_loads_strictly() {
    let checkpoint = OLMOE_1B_7B.declared();
    assert_eq!(load("OlmoeForCausalLM", &checkpoint).unwrap(), "1B-7B");
}

// # Strictness

/// A tensor the line reads and the checkpoint lacks is missing, by its name.
#[test]
// Fails until the registry lists a type for the name.
#[should_panic(expected = "Unsupported")]
fn missing_tensor_is_missing() {
    let mut checkpoint = OLMOE_1B_7B.declared();
    let name = "model.layers.3.mlp.experts.5.up_proj.weight";
    assert!(checkpoint.remove(name));
    match load("OlmoeForCausalLM", &checkpoint) {
        Err(LoadError::Missing(names)) => assert_eq!(names, [name]),
        other => panic!("{other:?}"),
    }
}

/// A tensor no rule of the line reads is unexpected, by its name.
#[test]
// Fails until the registry lists a type for the name.
#[should_panic(expected = "Unsupported")]
fn extra_tensor_is_unexpected() {
    let mut checkpoint = OLMOE_1B_7B.declared();
    let name = "model.layers.3.mlp.experts.5.extra.weight";
    checkpoint
        .insert(name, Dtype::F32, &[1], vec![0; 4])
        .unwrap();
    match load("OlmoeForCausalLM", &checkpoint) {
        Err(LoadError::Unexpected(names)) => assert_eq!(names, [name]),
        other => panic!("{other:?}"),
    }
}

// # Configs

/// The size parses its published checkpoint's config.
#[test]
fn size_reads_its_published_config() {
    Size::<"1B-7B">::parse(&OLMOE_1B_7B.config()).unwrap();
}

/// A config the size does not serve is refused, each case by its kind and reason.
#[test]
fn size_refuses_what_it_does_not_serve() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = OLMOE_1B_7B.config();
    let cases = [
        (
            "hidden_size",
            json!(4096),
            InvalidData,
            "hidden_size: 4096, the size 1B-7B holds 2048",
        ),
        (
            "intermediate_size",
            json!(2048),
            InvalidData,
            "intermediate_size: 2048, the size 1B-7B holds 1024",
        ),
        (
            "num_hidden_layers",
            json!(32),
            InvalidData,
            "num_hidden_layers: 32, the size 1B-7B holds 16",
        ),
        (
            "num_key_value_heads",
            json!(8),
            InvalidData,
            "num_key_value_heads: 8, the size 1B-7B holds 16",
        ),
        (
            "num_attention_heads",
            json!(32),
            InvalidData,
            "num_attention_heads: 32, the size 1B-7B holds 16",
        ),
        (
            "head_dim",
            json!(64),
            InvalidData,
            "head_dim: 64, the size 1B-7B holds 128",
        ),
        (
            "num_experts",
            json!(128),
            InvalidData,
            "num_experts: 128, the size 1B-7B holds 64",
        ),
        (
            "num_experts_per_tok",
            json!(2),
            InvalidData,
            "num_experts_per_tok: 2, the size 1B-7B holds 8",
        ),
        (
            "vocab_size",
            json!(100352),
            InvalidData,
            "vocab_size: 100352, the size compiles 50304 rows",
        ),
        (
            "vocab_size",
            json!(0),
            InvalidData,
            "vocab_size: 0, the size compiles 50304 rows",
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
            "hidden_act",
            json!("gelu"),
            Unsupported,
            "hidden_act: OLMoE's experts gate with SiLU",
        ),
        (
            "norm_topk_prob",
            json!(true),
            Unsupported,
            "norm_topk_prob: OLMoE weighs the selected experts without renormalizing",
        ),
        (
            "attention_bias",
            json!(true),
            Unsupported,
            "attention_bias: OLMoE projects without biases",
        ),
        (
            "clip_qkv",
            json!(8.0),
            Unsupported,
            "clip_qkv: OLMoE projects without clipping",
        ),
        (
            "rope_scaling",
            json!({"type": "linear", "factor": 2.0}),
            Unsupported,
            "found Linear { factor: 2.0 } rotating 1 of each head, compiled the unscaled spectrum over the whole head",
        ),
    ];
    for (field, value, kind, reason) in cases {
        let mut document = published.clone();
        document[field] = value;
        let error = Size::<"1B-7B">::parse(&document).map(drop).unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{field}");
    }
}
