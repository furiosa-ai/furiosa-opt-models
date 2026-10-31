//! The model list refuses the KL autoencoder's published checkpoint until a type serves its
//! architecture. Its config is checked against the size.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io::ErrorKind;

use furiosa_opt_engine::{LoadError, LoadOptions};
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::diffusion::autoencoder;

use common::Published;

const SD_VAE_FT_MSE: Published = Published::new(
    "stabilityai/sd-vae-ft-mse",
    "31f26fdeee1355a5c34592e401dd41e45d25a493",
);

// # The published checkpoint

/// sd-vae-ft-mse's config names `AutoencoderKL`, which the model list refuses: no type serves the
/// name yet.
#[test]
fn autoencoder_is_refused_until_its_root_exists() {
    match LoadOptions::new().parse(&SD_VAE_FT_MSE.checkpoint()) {
        Err(LoadError::Unsupported { name, reason }) => {
            assert_eq!(name, "AutoencoderKL");
            assert_eq!(reason, "no model type is implemented");
        }
        other => panic!("{other:?}"),
    }
}

// # The autoencoder's config

/// The 256 size reads the published config; the other sizes refuse it by its sample size.
#[test]
fn sizes_read_their_published_config() {
    let config = SD_VAE_FT_MSE.config();
    autoencoder::Size::<"256">::parse(&config).unwrap();
    for (error, size) in [
        (
            autoencoder::Size::<"512">::parse(&config).err().unwrap(),
            512,
        ),
        (
            autoencoder::Size::<"1024">::parse(&config).err().unwrap(),
            1024,
        ),
    ] {
        let reason = format!("sample_size: 256, the size {size} holds {size}");
        assert_eq!(
            (error.kind(), error.to_string()),
            (ErrorKind::Unsupported, reason)
        );
    }
}

/// A value the size does not hold and a feature it does not serve are unsupported, each refused
/// with its reason, which names the key it is read at.
#[test]
fn size_refuses_what_it_does_not_hold() {
    use serde_json::json;
    let published = SD_VAE_FT_MSE.config();
    let cases = [
        (
            "in_channels",
            json!(4),
            ErrorKind::Unsupported,
            "in_channels: 4, the size 256 holds 3",
        ),
        (
            "out_channels",
            json!(1),
            ErrorKind::Unsupported,
            "out_channels: 1, the size 256 holds 3",
        ),
        (
            "latent_channels",
            json!(16),
            ErrorKind::Unsupported,
            "latent_channels: 16, the size 256 holds 4",
        ),
        (
            "block_out_channels",
            json!([128, 256, 512]),
            ErrorKind::Unsupported,
            "block_out_channels: [128, 256, 512], the size 256 holds [128, 256, 512, 512]",
        ),
        (
            "layers_per_block",
            json!(1),
            ErrorKind::Unsupported,
            "layers_per_block: 1, the size 256 holds 2",
        ),
        (
            "norm_num_groups",
            json!(16),
            ErrorKind::Unsupported,
            "norm_num_groups: 16, the size 256 holds 32",
        ),
        (
            "use_quant_conv",
            json!(false),
            ErrorKind::Unsupported,
            "use_quant_conv: false, the size 256 holds the moment projection",
        ),
        (
            "use_post_quant_conv",
            json!(false),
            ErrorKind::Unsupported,
            "use_post_quant_conv: false, the size 256 holds the latent projection",
        ),
        (
            "act_fn",
            json!("gelu"),
            ErrorKind::Unsupported,
            "act_fn: the encoder and decoder activate by SiLU",
        ),
        (
            "mid_block_add_attention",
            json!(false),
            ErrorKind::Unsupported,
            "mid_block_add_attention: both middle blocks attend",
        ),
        (
            "down_block_types",
            json!(vec!["AttnDownEncoderBlock2D"; 4]),
            ErrorKind::Unsupported,
            "down_block_types: every encoder level is a DownEncoderBlock2D",
        ),
        (
            "up_block_types",
            json!(vec!["AttnUpDecoderBlock2D"; 4]),
            ErrorKind::Unsupported,
            "up_block_types: every decoder level is an UpDecoderBlock2D",
        ),
        (
            "down_block_types",
            json!(vec!["DownEncoderBlock2D"; 3]),
            ErrorKind::Unsupported,
            "down_block_types: every encoder level is a DownEncoderBlock2D",
        ),
        (
            "up_block_types",
            json!(vec!["UpDecoderBlock2D"; 3]),
            ErrorKind::Unsupported,
            "up_block_types: every decoder level is an UpDecoderBlock2D",
        ),
    ];
    for (key, value, kind, reason) in cases {
        let mut config = published.clone();
        config[key] = value;
        let error = autoencoder::Size::<"256">::parse(&config).err().unwrap();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{key}");
    }
}
