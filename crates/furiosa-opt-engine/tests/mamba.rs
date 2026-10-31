//! Mamba reads its published checkpoints: each size parses its own config and refuses another
//! size's and what it does not serve, and a checkpoint loads in BF16 strictly through the model
//! list, on host: every name its line reads is checked by name, dtype and shape, and a name it
//! lacks or does not read is refused.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;

use furiosa_opt_engine::Checkpoint;
use furiosa_opt_engine::registry::Implementation;
use furiosa_opt_models::load::LoadError;
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::decoder::recurrent::mamba::Size;
use safetensors::Dtype;
use serde_json::json;

use common::Published;

const MAMBA_130M: Published = Published::new(
    "state-spaces/mamba-130m-hf",
    "1e76775f628fbf1350fbe4dbb3d971ba64af25a1",
);
const MAMBA_2_8B: Published = Published::new(
    "state-spaces/mamba-2.8b-hf",
    "96c48e0292b63f5346b6d30061af2551f7101e26",
);
const MAMBA_370M: Published = Published::new(
    "state-spaces/mamba-370m-hf",
    "b519127f5bfaaa1c27dd938dad051ec360972b23",
);
const MAMBA_790M: Published = Published::new(
    "state-spaces/mamba-790m-hf",
    "9822dd4b76af2bd9099b6ce2f19efd8329189a7e",
);
const MAMBA_1_4B: Published = Published::new(
    "state-spaces/mamba-1.4b-hf",
    "6e46eae61c27280517feef46f536d16b91076f08",
);

/// The size of the implementation the registry chooses under `architecture` for `source`.
fn load(architecture: &str, source: &Checkpoint) -> Result<&'static str, LoadError> {
    let (name, implementations) = Implementation::listed(&[architecture])?;
    Implementation::choose(name, implementations, source).map(Implementation::size)
}

// # Strict loads of the published checkpoints

/// Mamba-130m selects its size and loads strictly, its head tied to the embedding.
#[test]
// Fails until the registry lists a type for the name.
#[should_panic(expected = "Unsupported")]
fn mamba_130m_loads_strictly() {
    let checkpoint = MAMBA_130M.declared();
    assert_eq!(load("MambaForCausalLM", &checkpoint).unwrap(), "130m");
}

/// Mamba-2.8b selects its size and loads strictly, its head tied to the embedding.
#[test]
// Fails until the registry lists a type for the name.
#[should_panic(expected = "Unsupported")]
fn mamba_2_8b_loads_strictly() {
    let checkpoint = MAMBA_2_8B.declared();
    assert_eq!(load("MambaForCausalLM", &checkpoint).unwrap(), "2.8b");
}

// # Strictness

/// A tensor the line reads and the checkpoint lacks is missing, by its name.
#[test]
// Fails until the registry lists a type for the name.
#[should_panic(expected = "Unsupported")]
fn missing_tensor_is_missing() {
    let mut checkpoint = MAMBA_130M.declared();
    let name = "backbone.layers.3.mixer.x_proj.weight";
    assert!(checkpoint.remove(name));
    match load("MambaForCausalLM", &checkpoint) {
        Err(LoadError::Missing(names)) => assert_eq!(names, [name]),
        other => panic!("{other:?}"),
    }
}

/// A tensor no rule of the line reads is unexpected, by its name.
#[test]
// Fails until the registry lists a type for the name.
#[should_panic(expected = "Unsupported")]
fn extra_tensor_is_unexpected() {
    let mut checkpoint = MAMBA_130M.declared();
    let name = "backbone.layers.3.mixer.extra.weight";
    checkpoint
        .insert(name, Dtype::F32, &[1], vec![0; 4])
        .unwrap();
    match load("MambaForCausalLM", &checkpoint) {
        Err(LoadError::Unexpected(names)) => assert_eq!(names, [name]),
        other => panic!("{other:?}"),
    }
}

// # Configs

/// Whether the size `N` serves `config`.
fn parse<const N: &'static str>(config: &serde_json::Value) -> io::Result<()>
where
    Size<N>: Family,
{
    Size::<N>::parse(config).map(drop)
}

/// Each size parses its published checkpoint's config and refuses every other size's by a field
/// they differ in, with both values.
#[test]
fn sizes_read_their_published_configs() {
    let sizes = [
        (parse::<"130m"> as fn(&_) -> _, MAMBA_130M.config()),
        (parse::<"370m">, MAMBA_370M.config()),
        (parse::<"790m">, MAMBA_790M.config()),
        (parse::<"1.4b">, MAMBA_1_4B.config()),
        (parse::<"2.8b">, MAMBA_2_8B.config()),
    ];
    for (i, (parse, own)) in sizes.iter().enumerate() {
        for (j, (_, config)) in sizes.iter().enumerate() {
            match (i == j, parse(config)) {
                (true, Ok(())) => {}
                (false, Err(error)) if error.kind() == io::ErrorKind::InvalidData => {
                    refuses_by_difference(&error, config, own);
                }
                (own, result) => panic!("size {i}, config {j} (own: {own}): {result:?}"),
            }
        }
    }
}

/// That `error`, a size's refusal of `config`, names a field where `config` differs from `own`, the
/// size's own config, with both values.
fn refuses_by_difference(error: &io::Error, config: &serde_json::Value, own: &serde_json::Value) {
    let reason = error.to_string();
    let text = |value: &serde_json::Value| value.as_str().map_or(value.to_string(), str::to_owned);
    let field = reason
        .split_once(": ")
        .map_or(reason.as_str(), |(field, _)| field);
    let (found, held) = (text(&config[field]), text(&own[field]));
    assert_ne!(found, held, "{reason}");
    let named = reason.starts_with(&format!("{field}: {found}, ")) && reason.ends_with(&held);
    assert!(named, "{reason}");
}

/// A config the size does not serve is refused, each case by its kind and reason; a time step
/// rank spelled `auto` is the size's.
#[test]
fn size_refuses_what_it_does_not_serve() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = MAMBA_130M.config();
    let mut auto = published.clone();
    auto["time_step_rank"] = json!("auto");
    Size::<"130m">::parse(&auto).unwrap();
    let cases = [
        (
            "num_hidden_layers",
            json!(48),
            InvalidData,
            "num_hidden_layers: 48, the size 130m holds 24",
        ),
        (
            "expand",
            json!(4),
            InvalidData,
            "expand: 4, the size 130m holds 2",
        ),
        (
            "state_size",
            json!(64),
            InvalidData,
            "state_size: 64, the size 130m holds 16",
        ),
        (
            "conv_kernel",
            json!(3),
            InvalidData,
            "conv_kernel: 3, the size 130m holds 4",
        ),
        (
            "time_step_rank",
            json!(64),
            InvalidData,
            "time_step_rank: 64, the size 130m holds 48",
        ),
        (
            "time_step_rank",
            json!("wide"),
            InvalidData,
            "time_step_rank: not a count, the size 130m holds 48",
        ),
        (
            "hidden_size",
            json!(1024),
            InvalidData,
            "hidden_size: 1024, the size 130m holds 768",
        ),
        (
            "vocab_size",
            json!(50432),
            Unsupported,
            "vocab_size: 50432, the size holds 1 to 50280",
        ),
        (
            "vocab_size",
            json!(0),
            InvalidData,
            "vocab_size: 0, the size holds 1 to 50280",
        ),
        (
            "layer_norm_epsilon",
            json!(1e-6),
            InvalidData,
            "layer_norm_epsilon: 0.000001, another than the size's",
        ),
        (
            "tie_word_embeddings",
            json!(false),
            InvalidData,
            "tie_word_embeddings: false, the size ties: true",
        ),
        (
            "hidden_act",
            json!("gelu"),
            Unsupported,
            "hidden_act: Mamba gates its mixer with SiLU",
        ),
        (
            "use_bias",
            json!(true),
            Unsupported,
            "use_bias: Mamba projects without biases",
        ),
        (
            "use_conv_bias",
            json!(false),
            Unsupported,
            "use_conv_bias: Mamba's convolution adds its bias",
        ),
        (
            "residual_in_fp32",
            json!(false),
            Unsupported,
            "residual_in_fp32: Mamba keeps its residual rows in FP32",
        ),
    ];
    for (field, value, kind, reason) in cases {
        let mut document = published.clone();
        document[field] = value;
        let error = Size::<"130m">::parse(&document).map(drop).unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{field}");
    }
}
