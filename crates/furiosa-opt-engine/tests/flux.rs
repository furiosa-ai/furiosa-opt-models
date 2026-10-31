//! The model list refuses FLUX.1's published transformer until a type serves its architecture.
//! Its config is checked against the size.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io::ErrorKind;

use furiosa_opt_engine::{LoadError, LoadOptions};
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::diffusion::flux;

use common::Published;

/// The transformer of a copy of the FLUX.1-schnell pipeline, whose own repository is gated.
const FLUX_1_SCHNELL: Published = Published::new(
    "frankjoshua/FLUX.1-schnell",
    "741f7c3ce8b383c54771c7003378a50191e9efe9",
)
.folder("transformer");

// # The published transformer

/// FLUX.1-schnell's transformer config names `FluxTransformer2DModel`, which the model list
/// refuses: no type serves the name yet.
#[test]
fn flux_is_refused_until_its_root_exists() {
    match LoadOptions::new().parse(&FLUX_1_SCHNELL.checkpoint()) {
        Err(LoadError::Unsupported { name, reason }) => {
            assert_eq!(name, "FluxTransformer2DModel");
            assert_eq!(reason, "no model type is implemented");
        }
        other => panic!("{other:?}"),
    }
}

// # The transformer's config

/// The schnell size reads its published config; the dev size refuses it, as schnell embeds no
/// guidance.
#[test]
fn sizes_read_their_published_config() {
    let config = FLUX_1_SCHNELL.config();
    flux::Size::<"schnell">::parse(&config).unwrap();
    let error = flux::Size::<"dev">::parse(&config).err().unwrap();
    assert_eq!(error.kind(), ErrorKind::Unsupported, "{error}");
    let reason = "guidance_embeds: false, the size dev embeds guidance: true";
    assert_eq!(error.to_string(), reason);
}

/// A value the size does not hold and a feature it does not serve are unsupported, each refused
/// with its reason, which names the key it is read at.
#[test]
fn size_refuses_what_it_does_not_hold() {
    use serde_json::json;
    let published = FLUX_1_SCHNELL.config();
    let cases = [
        (
            "patch_size",
            json!(2),
            ErrorKind::Unsupported,
            "patch_size: 2, the size schnell holds 1",
        ),
        (
            "in_channels",
            json!(16),
            ErrorKind::Unsupported,
            "in_channels: 16, the size schnell holds 64",
        ),
        (
            "out_channels",
            json!(16),
            ErrorKind::Unsupported,
            "out_channels: 16, the size schnell holds 64",
        ),
        (
            "num_layers",
            json!(8),
            ErrorKind::Unsupported,
            "num_layers: 8, the size schnell holds 19",
        ),
        (
            "num_single_layers",
            json!(20),
            ErrorKind::Unsupported,
            "num_single_layers: 20, the size schnell holds 38",
        ),
        (
            "attention_head_dim",
            json!(64),
            ErrorKind::Unsupported,
            "attention_head_dim: 64, the size schnell holds 128",
        ),
        (
            "num_attention_heads",
            json!(48),
            ErrorKind::Unsupported,
            "num_attention_heads: 48, the size schnell holds 24",
        ),
        (
            "joint_attention_dim",
            json!(2048),
            ErrorKind::Unsupported,
            "joint_attention_dim: 2048, the size schnell holds 4096",
        ),
        (
            "pooled_projection_dim",
            json!(1024),
            ErrorKind::Unsupported,
            "pooled_projection_dim: 1024, the size schnell holds 768",
        ),
        (
            "axes_dims_rope",
            json!([32, 48, 48]),
            ErrorKind::Unsupported,
            "axes_dims_rope: [32, 48, 48], the size schnell rotates [16, 56, 56]",
        ),
        (
            "guidance_embeds",
            json!(true),
            ErrorKind::Unsupported,
            "guidance_embeds: true, the size schnell embeds guidance: false",
        ),
        (
            "quantization_config",
            json!({"quant_method": "bitsandbytes"}),
            ErrorKind::Unsupported,
            "quantization_config: FLUX.1 computes on BF16 weights",
        ),
    ];
    for (key, value, kind, reason) in cases {
        let mut config = published.clone();
        config[key] = value;
        let error = flux::Size::<"schnell">::parse(&config).err().unwrap();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{key}");
    }
}
