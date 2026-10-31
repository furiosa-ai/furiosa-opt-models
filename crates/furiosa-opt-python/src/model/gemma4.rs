//! The Gemma 4 class a load of a Gemma 4 checkpoint's language model returns: `ForCausalLM`, a
//! `Decoder` whose calls return the greedy token of each kept row.

use super::{Constructor, Decoder};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_stub_gen::derive::gen_stub_pyclass;

/// Adds the Gemma 4 class to `module`.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ForCausalLM>()
}

/// The Gemma 4 class by name, as the constructor of its objects.
pub(super) const CLASSES: &[(&str, Constructor)] =
    &[("ForCausalLM", Decoder::object::<ForCausalLM>)];

/// HF `Gemma4ForCausalLM`, loaded on a device: the text model, then its greedy vocabulary head.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Decoder, module = "engine.gemma4")]
#[derive(Default)]
pub(super) struct ForCausalLM;
