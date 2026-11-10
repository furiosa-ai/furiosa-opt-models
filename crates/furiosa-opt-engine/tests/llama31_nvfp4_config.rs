//! Registry selection for the pinned NVFP4 checkpoint, without opening a device.
use furiosa_opt_engine::{Checkpoint, LoadOptions};
use safetensors::Dtype;
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
fn source(config: Value) -> Checkpoint {
    Checkpoint::new(tempfile::tempdir().unwrap().path(), Some(config)).unwrap()
}

#[test]
fn selects_nvfp4_with_bf16_cache_from_config_alone() {
    let source = source(config());
    let selected = LoadOptions::new().parse(&source).unwrap();
    assert_eq!(
        (selected.architecture(), selected.size(), selected.chips()),
        ("LlamaForCausalLM", "8B", 1)
    );
    assert_eq!(
        (selected.precision(), selected.kv_cache()),
        (Dtype::U8, Dtype::BF16)
    );
    assert!(
        LoadOptions::new()
            .kv_cache(Dtype::F8_E4M3)
            .parse(&source)
            .is_err()
    );
    assert!(LoadOptions::new().adapters().parse(&source).is_err());
}

#[test]
fn incompatible_nvfp4_does_not_fall_back_to_bf16() {
    for (field, value) in [
        ("intermediate_size", json!(12288)),
        ("attention_bias", json!(true)),
    ] {
        let mut document = config();
        document[field] = value;
        assert!(
            LoadOptions::new().parse(&source(document)).is_err(),
            "{field}"
        );
    }
    let mut document = config();
    document["quantization_config"] = Value::Null;
    assert!(LoadOptions::new().parse(&source(document)).is_err());
}
