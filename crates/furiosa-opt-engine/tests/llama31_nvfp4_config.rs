//! Registry selection for the pinned NVFP4 checkpoint, without opening a device.
use furiosa_opt_engine::{Checkpoint, LoadError, LoadOptions};
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
}

#[test]
fn refuses_unsupported_cache_and_adapter_choices() {
    let source = source(config());
    for (options, field, expected) in [
        (
            LoadOptions::new().kv_cache(Dtype::F8_E4M3),
            "kv_cache",
            "every implementation caches keys and values in BF16",
        ),
        (
            LoadOptions::new().adapters(),
            "adapters",
            "the LlamaForCausalLM 8B implementation takes no adapter",
        ),
    ] {
        match options
            .parse(&source)
            .err()
            .expect("unsupported load choice")
        {
            LoadError::Unsupported { name, reason } => {
                assert_eq!((name.as_str(), reason.as_str()), (field, expected))
            }
            other => panic!("{field}: {other:?}"),
        }
    }
}

#[test]
fn incompatible_nvfp4_does_not_fall_back_to_bf16() {
    // One invalid geometry pins registry selection; the model parser owns field coverage.
    let mut document = config();
    document["intermediate_size"] = json!(12288);
    match LoadOptions::new()
        .parse(&source(document))
        .err()
        .expect("no compatible implementation")
    {
        LoadError::InvalidData { name, reason } => {
            assert_eq!(name, "LlamaForCausalLM");
            assert!(
                reason.starts_with("no implementation serves the checkpoint:"),
                "{reason}"
            );
            assert!(
                reason.contains("intermediate_size: found 12288, compiled 14336"),
                "{reason}"
            );
        }
        other => panic!("{other:?}"),
    }
}
