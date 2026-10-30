//! DeepSeek-V2 reads its published checkpoint: the size parses its config and refuses what it does
//! not serve, and the registry chooses the checkpoint's size.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;

use furiosa_opt_engine::Checkpoint;
use furiosa_opt_engine::registry::Implementation;
use furiosa_opt_models::load::LoadError;
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::decoder::attention::deepseek_v2::Size;
use serde_json::json;

use common::Published;

const DEEPSEEK_V2_LITE: Published = Published::new(
    "deepseek-ai/DeepSeek-V2-Lite",
    "604d5664dddd88a0433dbae533b7fe9472482de0",
);

/// The size of the implementation the registry chooses under `architecture` for `source`.
fn load(architecture: &str, source: &Checkpoint) -> Result<&'static str, LoadError> {
    let (name, implementations) = Implementation::listed(&[architecture])?;
    Implementation::choose(name, implementations, source).map(Implementation::size)
}

// # Strict loads of the published checkpoints

/// DeepSeek-V2-Lite selects its size and loads strictly.
#[test]
// Fails until the registry lists a type for the name.
#[should_panic(expected = "Unsupported")]
fn deepseek_v2_lite_loads_strictly() {
    let checkpoint = DEEPSEEK_V2_LITE.declared();
    assert_eq!(load("DeepseekV2ForCausalLM", &checkpoint).unwrap(), "Lite");
}

// # Configs

/// The size parses its published checkpoint's config, and the same rotation in Transformers 5's
/// spelling.
#[test]
fn size_reads_its_published_config() {
    let published = DEEPSEEK_V2_LITE.config();
    Size::<"Lite">::parse(&published).unwrap();
    let mut respelled = published.clone();
    let mut rope = respelled["rope_scaling"].take();
    rope["rope_type"] = rope["type"].take();
    rope["rope_theta"] = respelled["rope_theta"].take();
    respelled["rope_parameters"] = rope;
    Size::<"Lite">::parse(&respelled).unwrap();
}

/// A config the size does not serve is refused, each case by its kind and reason.
#[test]
fn size_refuses_what_it_does_not_serve() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = DEEPSEEK_V2_LITE.config();
    let cases = [
        (
            "hidden_size",
            json!(5120),
            InvalidData,
            "hidden_size: 5120, the size Lite holds 2048",
        ),
        (
            "intermediate_size",
            json!(12288),
            InvalidData,
            "intermediate_size: 12288, the size Lite holds 10944",
        ),
        (
            "moe_intermediate_size",
            json!(1536),
            InvalidData,
            "moe_intermediate_size: 1536, the size Lite holds 1408",
        ),
        (
            "num_hidden_layers",
            json!(60),
            InvalidData,
            "num_hidden_layers: 60, the size Lite holds 27",
        ),
        (
            "num_key_value_heads",
            json!(128),
            InvalidData,
            "num_key_value_heads: 128, the size Lite holds 16",
        ),
        (
            "num_attention_heads",
            json!(128),
            InvalidData,
            "num_attention_heads: 128, the size Lite holds 16",
        ),
        (
            "kv_lora_rank",
            json!(256),
            InvalidData,
            "kv_lora_rank: 256, the size Lite holds 512",
        ),
        (
            "qk_nope_head_dim",
            json!(64),
            InvalidData,
            "qk_nope_head_dim: 64, the size Lite holds 128",
        ),
        (
            "qk_rope_head_dim",
            json!(32),
            InvalidData,
            "qk_rope_head_dim: 32, the size Lite holds 64",
        ),
        (
            "v_head_dim",
            json!(64),
            InvalidData,
            "v_head_dim: 64, the size Lite holds 128",
        ),
        (
            "n_routed_experts",
            json!(160),
            InvalidData,
            "n_routed_experts: 160, the size Lite holds 64",
        ),
        (
            "n_shared_experts",
            json!(1),
            InvalidData,
            "n_shared_experts: 1, the size Lite holds 2",
        ),
        (
            "num_experts_per_tok",
            json!(8),
            InvalidData,
            "num_experts_per_tok: 8, the size Lite holds 6",
        ),
        (
            "first_k_dense_replace",
            json!(3),
            InvalidData,
            "first_k_dense_replace: 3, the size Lite holds 1",
        ),
        (
            "vocab_size",
            json!(129280),
            InvalidData,
            "vocab_size: 129280, the size compiles 102400 rows",
        ),
        (
            "vocab_size",
            json!(0),
            InvalidData,
            "vocab_size: 0, the size compiles 102400 rows",
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
            json!(327680),
            Unsupported,
            "max_position_embeddings: past the compiled context of 163840",
        ),
        (
            "rope_scaling",
            json!({"type": "linear", "factor": 2.0}),
            Unsupported,
            "rope_scaling: DeepSeek-V2 rotates with its compiled YaRN",
        ),
        (
            "rope_theta",
            json!(50000),
            Unsupported,
            "rope_theta: DeepSeek-V2 rotates its compiled YaRN at 10000",
        ),
        (
            "q_lora_rank",
            json!(1536),
            Unsupported,
            "q_lora_rank: DeepSeek-V2 projects its queries at full rank",
        ),
        (
            "attention_bias",
            json!(true),
            Unsupported,
            "attention_bias: DeepSeek-V2 projects without biases",
        ),
        (
            "mlp_bias",
            json!(true),
            Unsupported,
            "mlp_bias: DeepSeek-V2's experts project without biases",
        ),
        (
            "hidden_act",
            json!("gelu"),
            Unsupported,
            "hidden_act: DeepSeek-V2's experts gate with SiLU",
        ),
        (
            "scoring_func",
            json!("sigmoid"),
            Unsupported,
            "scoring_func: DeepSeek-V2 routes by softmax probabilities",
        ),
        (
            "topk_method",
            json!("noaux_tc"),
            Unsupported,
            "topk_method: DeepSeek-V2 selects the top experts greedily",
        ),
        (
            "n_group",
            json!(8),
            Unsupported,
            "n_group: DeepSeek-V2 routes over one expert group",
        ),
        (
            "topk_group",
            json!(3),
            Unsupported,
            "topk_group: DeepSeek-V2 routes over one expert group",
        ),
        (
            "norm_topk_prob",
            json!(true),
            Unsupported,
            "norm_topk_prob: DeepSeek-V2 weighs the selected experts without renormalizing",
        ),
        (
            "routed_scaling_factor",
            json!(2.5),
            Unsupported,
            "routed_scaling_factor: DeepSeek-V2 weighs the routed experts unscaled",
        ),
        (
            "moe_layer_freq",
            json!(2),
            Unsupported,
            "moe_layer_freq: every DeepSeek-V2 layer past the dense ones is sparse",
        ),
    ];
    for (field, value, kind, reason) in cases {
        let mut document = published.clone();
        document[field] = value;
        let error = Size::<"Lite">::parse(&document).map(drop).unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{field}");
    }
    let refused = |document: &serde_json::Value| {
        let error = Size::<"Lite">::parse(document).map(drop).unwrap_err();
        (error.kind(), error.to_string())
    };
    let yarn = "rope_scaling: DeepSeek-V2 rotates with its compiled YaRN".to_owned();
    for (key, value) in [
        ("type", json!("linear")),
        ("factor", json!(20.0)),
        ("beta_fast", json!(16)),
        ("beta_slow", json!(2)),
        ("mscale", json!(1.0)),
        ("mscale_all_dim", json!(1.0)),
        ("original_max_position_embeddings", json!(2048)),
    ] {
        let mut document = published.clone();
        document["rope_scaling"][key] = value;
        assert_eq!(refused(&document), (Unsupported, yarn.clone()), "{key}");
    }
    let mut unrotated = published.clone();
    unrotated.as_object_mut().unwrap().remove("rope_scaling");
    assert_eq!(refused(&unrotated), (Unsupported, yarn));
    let mut baseless = published.clone();
    baseless.as_object_mut().unwrap().remove("rope_theta");
    let theta = "rope_theta: DeepSeek-V2 rotates its compiled YaRN at 10000".to_owned();
    assert_eq!(refused(&baseless), (Unsupported, theta));
}
