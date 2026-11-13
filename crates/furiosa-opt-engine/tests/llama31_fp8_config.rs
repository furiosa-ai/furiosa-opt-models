//! Native token-dynamic FP8 selection from the checkpoint declaration, without a device.
use furiosa_opt_engine::{Checkpoint, LoadOptions};
use safetensors::Dtype;
use serde_json::{Value, json};

fn source(mutate: impl FnOnce(&mut Value)) -> Checkpoint {
    let mut config = json!({
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
            "quant_method": "compressed-tensors", "format": "naive-quantized",
            "quantization_status": "compressed", "ignore": ["lm_head"],
            "config_groups": {"group_0": {
                "targets": ["Linear"],
                "weights": {"dynamic": false, "num_bits": 8, "type": "float",
                    "strategy": "channel", "symmetric": true},
                "input_activations": {"dynamic": true, "num_bits": 8, "type": "float",
                    "strategy": "token", "symmetric": true}
            }}
        }
    });
    mutate(&mut config);
    Checkpoint::new(tempfile::tempdir().unwrap().path(), Some(config)).unwrap()
}

#[test]
fn selects_native_fp8_with_bf16_cache() {
    let source = source(|_| {});
    let chosen = LoadOptions::new().parse(&source).unwrap();
    assert_eq!(
        (chosen.architecture(), chosen.size(), chosen.chips()),
        ("LlamaForCausalLM", "8B", 1)
    );
    assert_eq!(
        (chosen.precision(), chosen.kv_cache()),
        (Dtype::F8_E4M3, Dtype::BF16)
    );
}

#[test]
fn a_different_geometry_does_not_select_the_native_fp8_kernel() {
    let source = source(|config| config["intermediate_size"] = json!(12288));
    let error = LoadOptions::new()
        .parse(&source)
        .err()
        .expect("wrong geometry");
    assert!(
        error
            .to_string()
            .contains("intermediate_size: found 12288, compiled 14336"),
        "{error}"
    );
}
