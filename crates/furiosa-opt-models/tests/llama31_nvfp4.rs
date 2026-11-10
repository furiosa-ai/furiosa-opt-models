//! The published W4A16 checkpoint contract, without opening a device.
use furiosa_opt_models::load::Checkpoint;
use furiosa_opt_models::models::decoder::attention::llama::nvfp4::{ForCausalLM, Size};
use furiosa_opt_models::models::{Family, Implements};
use serde_json::{Value, json};

fn config() -> Value {
    serde_json::from_str(r#"{
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 128256, "hidden_act": "silu", "rms_norm_eps": 1e-5,
        "max_position_embeddings": 131072, "rope_theta": 500000.0,
        "tie_word_embeddings": false, "attention_bias": false, "mlp_bias": false,
        "rope_scaling": {
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192
        },
        "quantization_config": {
            "quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"],
            "config_groups": {"group_0": {
                "targets": ["Linear"],
                "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16}
            }},
            "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"}
        }
    }"#).unwrap()
}

#[test]
fn admits_the_pinned_modelopt_checkpoint() {
    let source = Checkpoint::new(tempfile::tempdir().unwrap().path(), Some(config())).unwrap();
    assert_eq!(ForCausalLM::scope(&source).unwrap(), "");
}

#[test]
fn refuses_incompatible_geometry_and_features() {
    for (field, value) in [
        ("hidden_size", json!(2048)),
        ("intermediate_size", json!(12288)),
        ("num_hidden_layers", json!(36)),
        ("num_attention_heads", json!(16)),
        ("num_key_value_heads", json!(4)),
        ("head_dim", json!(64)),
        ("vocab_size", json!(128000)),
        ("rms_norm_eps", json!(1e-6)),
        ("tie_word_embeddings", json!(true)),
        ("attention_bias", json!(true)),
        ("mlp_bias", json!(true)),
        ("hidden_act", json!("gelu")),
        ("sliding_window", json!(4096)),
    ] {
        let mut changed = config();
        changed[field] = value;
        assert!(Size::parse(&changed).is_err(), "{field}");
    }
}

#[test]
fn refuses_other_weight_and_cache_declarations() {
    for value in [
        Value::Null,
        json!({"quant_method": "modelopt", "quant_algo": "FP8"}),
        {
            let mut q = config()["quantization_config"].clone();
            q["ignore"] = json!(["lm_head", "model.layers.0.self_attn*"]);
            q
        },
        {
            let mut q = config()["quantization_config"].clone();
            q["kv_cache_scheme"] = Value::Null;
            q
        },
        {
            let mut q = config()["quantization_config"].clone();
            q["config_groups"]["group_0"]["weights"]["group_size"] = json!(32);
            q
        },
    ] {
        let mut changed = config();
        changed["quantization_config"] = value;
        assert!(Size::parse(&changed).is_err());
    }
}

#[test]
fn validates_the_scaled_rotary_spectrum() {
    let mut changed = config();
    changed["rope_scaling"]["high_freq_factor"] = json!(0.0);
    assert!(Size::parse(&changed).is_err());
    changed = config();
    changed["rope_scaling"]["rope_type"] = json!("unknown");
    assert!(Size::parse(&changed).is_err());
}
