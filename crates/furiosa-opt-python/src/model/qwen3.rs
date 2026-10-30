//! The Qwen3 classes a load of a Qwen3 checkpoint returns, each a `Decoder`: `ForCausalLM`, whose
//! calls return the greedy token of each kept row, `Model`, whose calls return the kept rows'
//! hidden values, and `ForSequenceClassification`, whose calls return the kept rows' scores.

use super::{Constructor, Decoder};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_stub_gen::derive::gen_stub_pyclass;

/// Adds the Qwen3 classes to `module`.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ForCausalLM>()?;
    module.add_class::<Model>()?;
    module.add_class::<ForSequenceClassification>()?;
    Ok(())
}

/// The Qwen3 classes by name, each as the constructor of its objects.
pub(super) const CLASSES: &[(&str, Constructor)] = &[
    ("ForCausalLM", Decoder::object::<ForCausalLM>),
    ("Model", Decoder::object::<Model>),
    (
        "ForSequenceClassification",
        Decoder::object::<ForSequenceClassification>,
    ),
];

/// HF `Qwen3ForCausalLM`, loaded on a device: the base model, then its greedy vocabulary head.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Decoder, module = "engine.qwen3")]
#[derive(Default)]
pub(super) struct ForCausalLM;

/// HF `Qwen3Model`, loaded on a device: the embedding, the layers and the final norm.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Decoder, module = "engine.qwen3")]
#[derive(Default)]
pub(super) struct Model;

/// HF `Qwen3ForSequenceClassification`, loaded on a device: the base model, then its score head.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Decoder, module = "engine.qwen3")]
#[derive(Default)]
pub(super) struct ForSequenceClassification;
