//! Qwen3-8B in BF16 loads strictly from its published checkpoint, on host: every name its tree reads
//! is checked by name, dtype and shape as an HBM load checks it.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use furiosa_opt_engine::LoadOptions;
use furiosa_opt_models::load::{Context, Load, LoadError, Shapes};
use furiosa_opt_models::models::Implements;
use furiosa_opt_models::models::decoder::attention::qwen3;
use furiosa_opt_models::models::decoder::{form, launch};
use furiosa_opt_std::prelude::bf16;
use safetensors::Dtype;

use common::Published;

const QWEN3_8B: Published =
    Published::new("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218");

/// Qwen3-8B in block-scaled E4M3, as the hub publishes it.
const QWEN3_8B_FP8: Published = Published::new(
    "furiosa-ai/Qwen3-8B-FP8",
    "6124ba90c39c8e2299ddd57f4b4f0df5845480c8",
);

/// The BF16 causal LM of Qwen3-8B over the memory that checks shapes.
type Catalog = qwen3::Catalog<qwen3::Size<"8B">, form::Affine, bf16, Shapes>;

/// The BF16 Qwen3-8B causal line.
type Causal = launch::Model<qwen3::Size<"8B">, form::Affine>;

/// Qwen3-8B loads as its size with its own head.
#[test]
fn qwen3_8b_loads_strictly() {
    let checkpoint = QWEN3_8B.declared();
    <Catalog as Load<_>>::load(&mut Context::new(None), &checkpoint).unwrap();
}

/// The BF16 Qwen3-8B causal line serves only a BF16 checkpoint, and the block-FP8 tree only a
/// block-FP8 one.
#[test]
fn one_line_serves_each_precision() {
    let (bf16, fp8) = (QWEN3_8B.checkpoint(), QWEN3_8B_FP8.checkpoint());
    assert_eq!(Causal::scope(&bf16).unwrap(), "");
    assert!(matches!(
        Causal::scope(&fp8),
        Err(LoadError::Unsupported { .. })
    ));
    assert_eq!(qwen3::fp8::ForCausalLM::scope(&fp8).unwrap(), "");
    assert!(qwen3::fp8::ForCausalLM::scope(&bf16).is_err());
    let config = LoadOptions::new().parse(&bf16).unwrap();
    assert_eq!((config.size(), config.precision()), ("8B", Dtype::BF16));
}
