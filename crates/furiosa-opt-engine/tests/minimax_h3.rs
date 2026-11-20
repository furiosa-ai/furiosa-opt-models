//! H3 load-option refusals at the engine's checkpoint boundary.

use furiosa_opt_engine::{Checkpoint, LoadError, LoadOptions, Quantization};
use safetensors::Dtype;
use serde_json::json;

/// H3-specific load choices fail before sidecars or weights are needed. The ordinary
/// decoder refusal contract remains covered at the Qwen3 checkpoint boundary.
#[test]
fn refuses_h3_decoder_options_before_documents_and_weights() {
    let empty = tempfile::tempdir().unwrap();
    let source = Checkpoint::new(
        empty.path(),
        Some(json!({"_class_name": "MiniMaxH3Pipeline"})),
    )
    .unwrap();
    for (options, field) in [
        (LoadOptions::new().chips(1).unwrap(), "chips"),
        (LoadOptions::new().chips(8).unwrap(), "chips"),
        (LoadOptions::new().kv_cache(Dtype::BF16), "kv_cache"),
        (LoadOptions::new().layers(0..50), "layers"),
        (LoadOptions::new().batch_invariant(), "batch_invariant"),
        (
            LoadOptions::new().quantization(Quantization::Fp8PerTensor),
            "quantization",
        ),
        (LoadOptions::new().adapters(), "adapters"),
        (LoadOptions::new().encoder(), "encoder"),
    ] {
        for error in [
            options.parse(&source).err(),
            options.prepare_pipeline(&source).err(),
        ] {
            match error.expect("a refused choice must not read absent files") {
                LoadError::Unsupported { name, reason } => {
                    assert_eq!(name, field);
                    let expected = match field {
                        "chips" => "H3 runs on four chips",
                        "kv_cache" => "H3 has no KV cache",
                        "layers" => "every implementation loads whole",
                        "batch_invariant" => "H3 uses its original attention tiling",
                        "quantization" => "H3 loads its checkpoint precision",
                        "adapters" => "the MiniMaxH3Pipeline FL2VA implementation takes no adapter",
                        "encoder" => "H3 loads as one complete pipeline",
                        _ => unreachable!(),
                    };
                    assert_eq!(reason, expected);
                }
                other => panic!("{field}: {other:?}"),
            }
        }
    }
}
