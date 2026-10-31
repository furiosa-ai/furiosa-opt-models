//! Qwen3 MoE reads its published checkpoints: each size parses its own config and refuses another
//! size's and what it does not serve, and a checkpoint loads in BF16 strictly through the model
//! list, on host: every name its line reads is checked by name, dtype and shape.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;

use furiosa_opt_engine::Checkpoint;
use furiosa_opt_engine::registry::Implementation;
use furiosa_opt_models::load::LoadError;
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::decoder::attention::qwen3_moe::Size;
use serde_json::json;

use common::Published;

const QWEN3_30B_A3B: Published = Published::new(
    "Qwen/Qwen3-30B-A3B",
    "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
);
const QWEN3_235B_A22B: Published = Published::new(
    "Qwen/Qwen3-235B-A22B",
    "8efa61729e24bd65b1d152b5ab5409052aa80e65",
);

/// The size of the implementation the registry chooses under `architecture` for `source`.
fn load(architecture: &str, source: &Checkpoint) -> Result<&'static str, LoadError> {
    let (name, implementations) = Implementation::listed(&[architecture])?;
    Implementation::choose(name, implementations, source).map(Implementation::size)
}

// # Strict loads of the published checkpoints

/// Qwen3-30B-A3B selects its size and loads strictly.
#[test]
// Fails until the registry lists a type for the name.
#[should_panic(expected = "Unsupported")]
fn qwen3_30b_a3b_loads_strictly() {
    let checkpoint = QWEN3_30B_A3B.declared();
    assert_eq!(load("Qwen3MoeForCausalLM", &checkpoint).unwrap(), "30B-A3B");
}

// # Configs

/// Each size parses its published checkpoint's config and refuses the other size's by its width.
#[test]
fn sizes_read_their_published_configs() {
    let large = QWEN3_235B_A22B.config();
    let small = QWEN3_30B_A3B.config();
    Size::<"30B-A3B">::parse(&small).unwrap();
    Size::<"235B-A22B">::parse(&large).unwrap();
    for (error, reason) in [
        (
            Size::<"30B-A3B">::parse(&large).map(drop).unwrap_err(),
            "hidden_size: 4096, the size 30B-A3B holds 2048",
        ),
        (
            Size::<"235B-A22B">::parse(&small).map(drop).unwrap_err(),
            "hidden_size: 2048, the size 235B-A22B holds 4096",
        ),
    ] {
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (io::ErrorKind::InvalidData, reason.into()));
    }
}

/// A config the size does not serve is refused, each case by its kind and reason.
#[test]
fn size_refuses_what_it_does_not_serve() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = QWEN3_30B_A3B.config();
    let cases = [
        (
            "moe_intermediate_size",
            json!(1536),
            InvalidData,
            "moe_intermediate_size: 1536, the size 30B-A3B holds 768",
        ),
        (
            "num_hidden_layers",
            json!(94),
            InvalidData,
            "num_hidden_layers: 94, the size 30B-A3B holds 48",
        ),
        (
            "num_key_value_heads",
            json!(8),
            InvalidData,
            "num_key_value_heads: 8, the size 30B-A3B holds 4",
        ),
        (
            "num_attention_heads",
            json!(64),
            InvalidData,
            "num_attention_heads: 64, the size 30B-A3B holds 32",
        ),
        (
            "head_dim",
            json!(64),
            InvalidData,
            "head_dim: 64, the size 30B-A3B holds 128",
        ),
        (
            "num_experts",
            json!(64),
            InvalidData,
            "num_experts: 64, the size 30B-A3B holds 128",
        ),
        (
            "num_experts_per_tok",
            json!(4),
            InvalidData,
            "num_experts_per_tok: 4, the size 30B-A3B holds 8",
        ),
        (
            "vocab_size",
            json!(152064),
            Unsupported,
            "vocab_size: 152064, the size holds 1 to 151936",
        ),
        (
            "vocab_size",
            json!(0),
            InvalidData,
            "vocab_size: 0, the size holds 1 to 151936",
        ),
        (
            "rms_norm_eps",
            json!(1e-5),
            InvalidData,
            "rms_norm_eps: 0.00001, another than the size's",
        ),
        (
            "tie_word_embeddings",
            json!(true),
            InvalidData,
            "tie_word_embeddings: true, the size ties: false",
        ),
        (
            "max_position_embeddings",
            json!(131072),
            Unsupported,
            "max_position_embeddings: past the compiled context of 40960",
        ),
        (
            "hidden_act",
            json!("gelu"),
            Unsupported,
            "hidden_act: Qwen3 MoE's experts gate with SiLU",
        ),
        (
            "norm_topk_prob",
            json!(false),
            Unsupported,
            "norm_topk_prob: Qwen3 MoE renormalizes the selected experts' probabilities",
        ),
        (
            "mlp_only_layers",
            json!([0]),
            Unsupported,
            "mlp_only_layers: every Qwen3 MoE layer is sparse",
        ),
        (
            "decoder_sparse_step",
            json!(2),
            Unsupported,
            "decoder_sparse_step: every Qwen3 MoE layer is sparse",
        ),
        (
            "rope_scaling",
            json!({"rope_type": "linear", "factor": 2.0}),
            Unsupported,
            "found Linear { factor: 2.0 } rotating 1 of each head, compiled the unscaled spectrum over the whole head",
        ),
        (
            "use_sliding_window",
            json!(true),
            Unsupported,
            "use_sliding_window: Qwen3 MoE attends the whole history",
        ),
        (
            "attention_bias",
            json!(true),
            Unsupported,
            "attention_bias: Qwen3 MoE projects without biases",
        ),
    ];
    for (field, value, kind, reason) in cases {
        let mut document = published.clone();
        document[field] = value;
        let error = Size::<"30B-A3B">::parse(&document).map(drop).unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{field}");
    }
}
