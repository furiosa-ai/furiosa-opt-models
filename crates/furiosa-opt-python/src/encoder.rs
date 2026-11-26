//! The typed media inputs an encoder takes, as a framework's processor gives them.
//!
//! - An `Image` is merged patches on the model's device, one row each, and each patch's (x, y)
//!   position in its frame; a video's frames are one image each, or one image of all of them.
//! - `Audio` is frames of samples on the model's device, one row each.
//! - A loaded `Encoder`'s `vision(images)` and `audio(clips)` return each item's rows of the
//!   language model's width, in order.
//!
//! These inputs cover positioned patch features and audio-frame rows. Another processor input form
//! needs its own typed API; these classes are not a universal media container.

use super::tensor::HbmTensor;
use super::{Error, Shared};
use furiosa_opt_engine::encoder;
use furiosa_opt_models::tensor::memory::Dtype;
use furiosa_opt_std::prelude::HbmBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::Arc;

/// An image or a video's frames: `patches`, merged patch features as BF16 rows `[n, features]`, and the
/// (x, y) position of each patch in its frame.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.encoder")]
pub(super) struct Image {
    patches: Py<HbmTensor>,
    positions: Vec<[i32; 2]>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Image {
    /// `patches` lie on the model's device; `positions` name the valid patches only.
    ///
    /// Raises:
    ///     ValueError: `patches` are not two-dimensional BF16 rows, one for each position.
    #[new]
    fn new(
        patches: Bound<'_, HbmTensor>,
        #[gen_stub(override_type(type_repr = "collections.abc.Sequence[collections.abc.Sequence[int]]", imports = ("collections.abc",)))]
        positions: Vec<[i32; 2]>,
    ) -> PyResult<Self> {
        rows(&patches, positions.len())?;
        Ok(Self {
            patches: patches.unbind(),
            positions,
        })
    }
}

impl Image {
    /// This image as a call on the device of `context` reads it; refuses patches on another device.
    pub(super) fn on(&self, context: &Shared) -> Result<encoder::Image, Error> {
        let patches = storage(&self.patches, context)?;
        Ok(encoder::Image::new(patches, self.positions.clone())?)
    }
}

/// Audio: `frames`, BF16 rows `[n, samples]` of samples each.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.encoder")]
pub(super) struct Audio {
    frames: Py<HbmTensor>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Audio {
    /// `frames` lie on the model's device.
    ///
    /// Raises:
    ///     ValueError: `frames` are not two-dimensional BF16 rows.
    #[new]
    fn new(frames: Bound<'_, HbmTensor>) -> PyResult<Self> {
        let count = frames.get().shape.first().copied().unwrap_or(0);
        rows(&frames, count)?;
        Ok(Self {
            frames: frames.unbind(),
        })
    }
}

impl Audio {
    /// This clip as a call on the device of `context` reads it; refuses frames on another device.
    pub(super) fn on(&self, context: &Shared) -> Result<encoder::Audio, Error> {
        let count = self.frames.get().shape[0];
        Ok(encoder::Audio::new(storage(&self.frames, context)?, count))
    }
}

/// Adds the media classes to `module`.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Image>()?;
    module.add_class::<Audio>()?;
    Ok(())
}

/// Refuses `tensor` unless it holds `count` two-dimensional BF16 rows.
fn rows(tensor: &Bound<'_, HbmTensor>, count: usize) -> PyResult<()> {
    let tensor = tensor.get();
    match (tensor.dtype, &tensor.shape[..]) {
        (Dtype::BF16, &[rows, _]) if rows == count => Ok(()),
        _ => Err(PyValueError::new_err(format!(
            "media are {count} BF16 rows, not {:?} {:?}",
            tensor.dtype, tensor.shape
        ))),
    }
}

/// The storage of `tensor`, refused unless it lies on the device of `context`.
fn storage(tensor: &Py<HbmTensor>, context: &Shared) -> Result<HbmBuffer, Error> {
    let tensor = tensor.get();
    if !Arc::ptr_eq(&tensor.context, context) {
        return Err(PyValueError::new_err("a media tensor lies on another device").into());
    }
    Ok(tensor.buffer.alias(..)?)
}
