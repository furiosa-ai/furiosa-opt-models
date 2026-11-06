//! Registry selection for the pinned NVFP4 checkpoint, without opening a device.
use furiosa_opt_engine::{Checkpoint, LoadOptions};
use safetensors::Dtype;
use serde_json::{Value, json};

fn config() -> Value {
    serde_json::from_str(include_str!(
        "../../furiosa-opt-models/tests/fixtures/llama_nvfp4_config.json"
    ))
    .unwrap()
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
