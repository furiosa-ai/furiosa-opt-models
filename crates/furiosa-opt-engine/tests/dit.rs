//! The model list refuses DiT's published transformer until a type serves its architecture. Its
//! config is checked against the size.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io::ErrorKind;

use furiosa_opt_engine::{LoadError, LoadOptions};
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::diffusion::dit;

use common::Published;

/// The DiT-XL-2-256 pipeline's transformer, at the revision that stores its weights as safetensors.
const DIT_XL_2_256: Published = Published::new(
    "facebook/DiT-XL-2-256",
    "32d0e3550d47bdf75c5092db378c0f32cd12e100",
)
.folder("transformer");

// # The published transformer

/// DiT-XL-2-256's transformer config names `Transformer2DModel`, which the model list refuses: no
/// type serves the name yet.
#[test]
fn dit_is_refused_until_its_root_exists() {
    match LoadOptions::new().parse(&DIT_XL_2_256.checkpoint()) {
        Err(LoadError::Unsupported { name, reason }) => {
            assert_eq!(name, "Transformer2DModel");
            assert_eq!(reason, "no model type is implemented");
        }
        other => panic!("{other:?}"),
    }
}

// # The transformer's config

/// The XL-2-256 size reads its published config.
#[test]
fn size_reads_its_published_config() {
    dit::Size::<"XL-2-256">::parse(&DIT_XL_2_256.config()).unwrap();
}

/// A value the size does not hold is invalid data and a feature it does not serve is unsupported,
/// each refused with its reason, which names the key it is read at.
#[test]
fn size_refuses_what_it_does_not_hold() {
    use serde_json::json;
    let published = DIT_XL_2_256.config();
    let cases = [
        (
            "num_attention_heads",
            json!(12),
            ErrorKind::InvalidData,
            "num_attention_heads: 12, the size XL-2-256 holds 16",
        ),
        (
            "attention_head_dim",
            json!(64),
            ErrorKind::InvalidData,
            "attention_head_dim: 64, the size XL-2-256 holds 72",
        ),
        (
            "num_layers",
            json!(24),
            ErrorKind::InvalidData,
            "num_layers: 24, the size XL-2-256 holds 28",
        ),
        (
            "in_channels",
            json!(3),
            ErrorKind::InvalidData,
            "in_channels: 3, the size XL-2-256 holds 4",
        ),
        (
            "out_channels",
            json!(4),
            ErrorKind::InvalidData,
            "out_channels: 4, the size XL-2-256 holds 8",
        ),
        (
            "sample_size",
            json!(64),
            ErrorKind::InvalidData,
            "sample_size: 64, the size XL-2-256 holds 32",
        ),
        (
            "patch_size",
            json!(4),
            ErrorKind::InvalidData,
            "patch_size: 4, the size XL-2-256 holds 2",
        ),
        (
            "num_embeds_ada_norm",
            json!(100),
            ErrorKind::InvalidData,
            "num_embeds_ada_norm: 100, the size XL-2-256 holds 1000",
        ),
        (
            "norm_eps",
            json!(1e-6),
            ErrorKind::InvalidData,
            "norm_eps: 0.000001, another than the size's",
        ),
        (
            "interpolation_scale",
            json!(2.0),
            ErrorKind::InvalidData,
            "interpolation_scale: 2, the size XL-2-256 holds 1",
        ),
        (
            "norm_type",
            json!("ada_norm_single"),
            ErrorKind::Unsupported,
            "norm_type: DiT modulates its norms by the time and class embedding",
        ),
        (
            "activation_fn",
            json!("geglu"),
            ErrorKind::Unsupported,
            "activation_fn: DiT's feed-forward is tanh-approximated GELU",
        ),
        (
            "norm_elementwise_affine",
            json!(true),
            ErrorKind::Unsupported,
            "norm_elementwise_affine: DiT's norms hold no weights",
        ),
        (
            "attention_bias",
            json!(false),
            ErrorKind::Unsupported,
            "attention_bias: DiT projects with biases",
        ),
        (
            "only_cross_attention",
            json!(true),
            ErrorKind::Unsupported,
            "only_cross_attention: DiT attends to its own rows",
        ),
        (
            "double_self_attention",
            json!(true),
            ErrorKind::Unsupported,
            "double_self_attention: DiT attends once a layer",
        ),
        (
            "use_linear_projection",
            json!(true),
            ErrorKind::Unsupported,
            "use_linear_projection: DiT embeds patches",
        ),
        (
            "cross_attention_dim",
            json!(1152),
            ErrorKind::Unsupported,
            "cross_attention_dim: DiT attends to no condition rows",
        ),
        (
            "num_vector_embeds",
            json!(1024),
            ErrorKind::Unsupported,
            "num_vector_embeds: DiT predicts continuous latents",
        ),
    ];
    for (key, value, kind, reason) in cases {
        let mut config = published.clone();
        config[key] = value;
        let error = dit::Size::<"XL-2-256">::parse(&config).err().unwrap();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{key}");
    }
}
