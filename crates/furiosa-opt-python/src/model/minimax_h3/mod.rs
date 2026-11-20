//! One H3 pipeline and component views retaining that same owner and device context.

use crate::tensor::{Device, HostTensor};
use crate::{Error, Shared};
use furiosa_opt_engine::diffusion::minimax_h3 as engine;
use furiosa_opt_engine::residency::block_on;
use furiosa_opt_models::tensor::memory::Dtype;
use numpy::IntoPyArray;
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::Arc;

mod input;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Component>()?;
    module.add_class::<Pipeline>()?;
    module.add_class::<Transformer>()?;
    module.add_class::<TextEncoder>()?;
    module.add_class::<VideoDecoder>()?;
    module.add_class::<AudioDecoder>()?;
    Ok(())
}

pub(super) fn object(
    py: Python<'_>,
    pipeline: Arc<engine::Pipeline>,
    context: Shared,
) -> PyResult<Py<Pipeline>> {
    Py::new(py, (Pipeline, Component { pipeline, context }))
}

/// Shared ownership and placement of a pipeline or any of its callable component views.
#[gen_stub_pyclass]
#[pyclass(frozen, subclass, module = "engine.minimax_h3")]
#[derive(Clone)]
pub(super) struct Component {
    pipeline: Arc<engine::Pipeline>,
    context: Shared,
}

#[gen_stub_pymethods]
#[pymethods]
impl Component {
    /// The original device; every view retains it even after the pipeline object is dropped.
    #[getter]
    fn device(&self) -> Device {
        Device(self.context.clone())
    }

    /// Return this view on the device it already owns.
    #[gen_stub(override_return_type(type_repr = "typing.Self", imports = ("typing",)))]
    fn to_hbm<'py>(slf: Bound<'py, Self>, device: &Device) -> PyResult<Bound<'py, Self>> {
        if !Arc::ptr_eq(&slf.borrow().context, &device.0) {
            return Err(PyNotImplementedError::new_err(
                "these weights stay on the device they loaded onto",
            ));
        }
        Ok(slf)
    }

    /// H3 keeps its resident components on their original device.
    #[gen_stub(override_return_type(type_repr = "typing.Self", imports = ("typing",)))]
    fn to_host(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        Err(PyNotImplementedError::new_err(format!(
            "{} weights stay on the device they loaded onto",
            slf.get_type().name()?,
        )))
    }

    /// The pipeline has already prepared and pinned its streamed host sources.
    #[gen_stub(override_return_type(type_repr = "typing.Self", imports = ("typing",)))]
    fn pin(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }
}

/// The complete four-chip H3 pipeline, with one serialized request memo and streamed encoder.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Component, module = "engine.minimax_h3")]
pub(super) struct Pipeline;

#[gen_stub_pymethods]
#[pymethods]
impl Pipeline {
    #[classattr]
    fn chips() -> usize {
        engine::Pipeline::CHIPS
    }

    #[classattr]
    fn max_text_tokens() -> usize {
        engine::Pipeline::MAX_TEXT_TOKENS
    }

    #[classattr]
    fn names() -> (&'static str, &'static str, &'static str, &'static str) {
        ("transformer", "text_encoder", "video_vae", "audio_vae")
    }

    #[getter]
    fn max_num_seqs(&self) -> usize {
        1
    }

    #[getter]
    fn transformer(slf: PyRef<'_, Self>) -> PyResult<Py<Transformer>> {
        let py = slf.py();
        Py::new(py, (Transformer, slf.into_super().clone()))
    }

    #[getter]
    fn text_encoder(slf: PyRef<'_, Self>) -> PyResult<Py<TextEncoder>> {
        let py = slf.py();
        Py::new(py, (TextEncoder, slf.into_super().clone()))
    }

    #[getter]
    fn video_vae(slf: PyRef<'_, Self>) -> PyResult<Py<VideoDecoder>> {
        let py = slf.py();
        Py::new(py, (VideoDecoder, slf.into_super().clone()))
    }

    #[getter]
    fn audio_vae(slf: PyRef<'_, Self>) -> PyResult<Py<AudioDecoder>> {
        let py = slf.py();
        Py::new(py, (AudioDecoder, slf.into_super().clone()))
    }
}

/// Transformer evaluations and optional retained rows, with solver updates in the framework.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Component, module = "engine.minimax_h3")]
pub(super) struct Transformer;

#[gen_stub_pymethods]
#[pymethods]
impl Transformer {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        slf: PyRef<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Result<(Py<PyAny>, Py<PyAny>), Error> {
        if !args.is_empty() {
            return Err(PyTypeError::new_err("transformer takes forward keyword arguments").into());
        }
        let kwargs = kwargs.ok_or_else(|| {
            PyTypeError::new_err("transformer requires forward keyword arguments")
        })?;
        let values = input::parse(kwargs)?;
        let py = slf.py();
        let pipeline = slf.into_super().pipeline.clone();
        let output = py.detach(|| block_on(pipeline.forward(values)))?;
        Ok((
            fp32(py, output.video, &output.video_shape)?,
            fp32(py, output.audio, &output.audio_shape)?,
        ))
    }

    /// Compute a probe, ask the framework's policy, and read out computed or cached rows.
    /// Retained rows stay inside this pipeline; callback exceptions leave no published cache.
    /// Calls into the same pipeline from `decide` on its thread raise `RuntimeError`.
    /// Calls from other threads remain serialized.
    #[pyo3(signature = (decide, /, **kwargs))]
    fn cached_forward(
        slf: PyRef<'_, Self>,
        #[gen_stub(override_type(type_repr = "collections.abc.Callable[[tuple[float, float] | None], bool]", imports = ("collections.abc",)))]
        decide: Py<PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Result<(Py<PyAny>, Py<PyAny>), Error> {
        let py = slf.py();
        if !decide.bind(py).is_callable() {
            return Err(PyTypeError::new_err("TeaCache decision must be callable").into());
        }
        let kwargs = kwargs.ok_or_else(|| {
            PyTypeError::new_err("transformer requires forward keyword arguments")
        })?;
        let values = input::parse(kwargs)?;
        let pipeline = slf.into_super().pipeline.clone();
        let mut callback_error = None;
        let output = py.detach(|| {
            block_on(pipeline.forward_cached(values, |means| {
                Python::attach(|py| decide.call1(py, (means,))?.extract::<bool>(py)).map_err(
                    |error| {
                        callback_error = Some(error);
                        std::io::Error::other("TeaCache decision callback failed")
                    },
                )
            }))
        });
        if let Some(error) = callback_error {
            return Err(error.into());
        }
        let output = output?;
        let result = (|| {
            Ok((
                fp32(py, output.video, &output.video_shape)?,
                fp32(py, output.audio, &output.audio_shape)?,
            ))
        })();
        if result.is_err() {
            // Preserve the original conversion exception if cleanup itself reports an error.
            let _ = py.detach(|| block_on(pipeline.reset_cache()));
        }
        result
    }

    /// Start a new generation without retaining an earlier generation's probe or residual.
    fn reset_cache(slf: PyRef<'_, Self>) -> Result<(), Error> {
        let py = slf.py();
        let pipeline = slf.into_super().pipeline.clone();
        py.detach(|| block_on(pipeline.reset_cache()))?;
        Ok(())
    }
}

/// The first 50 text layers before final normalization, on the pipeline's same four chips.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Component, module = "engine.minimax_h3")]
pub(super) struct TextEncoder;

#[gen_stub_pymethods]
#[pymethods]
impl TextEncoder {
    #[pyo3(signature = (input_ids, **media))]
    fn __call__(
        slf: PyRef<'_, Self>,
        input_ids: &Bound<'_, PyAny>,
        media: Option<&Bound<'_, PyDict>>,
    ) -> Result<Py<PyAny>, Error> {
        encode(slf, input_ids, media)
    }

    #[pyo3(signature = (input_ids, **media))]
    fn encode_ids(
        slf: PyRef<'_, Self>,
        input_ids: &Bound<'_, PyAny>,
        media: Option<&Bound<'_, PyDict>>,
    ) -> Result<Py<PyAny>, Error> {
        encode(slf, input_ids, media)
    }

    /// Streaming places each layer before its call; the framework has nothing to move here.
    fn load_to_device(&self) {}
}

fn encode(
    slf: PyRef<'_, TextEncoder>,
    input_ids: &Bound<'_, PyAny>,
    media: Option<&Bound<'_, PyDict>>,
) -> Result<Py<PyAny>, Error> {
    if media.is_some_and(|media| media.values().iter().any(|value| !value.is_none())) {
        return Err(PyNotImplementedError::new_err(
            "the engine encodes text; image and video inputs are not served",
        )
        .into());
    }
    let ids = host(input_ids)?;
    if ids.shape.len() != 1 || ids.shape[0] == 0 || !matches!(ids.dtype, Dtype::I32 | Dtype::I64) {
        return Err(
            PyValueError::new_err("input_ids must be a non-empty 1-D integer tensor").into(),
        );
    }
    let tokens = ids
        .buffer
        .chunks_exact(ids.dtype.size())
        .map(|bytes| {
            let value = match ids.dtype {
                Dtype::I32 => i64::from(i32::from_le_bytes(bytes.try_into().unwrap())),
                Dtype::I64 => i64::from_le_bytes(bytes.try_into().unwrap()),
                _ => unreachable!("integer dtype checked above"),
            };
            i32::try_from(value).map_err(|_| PyValueError::new_err("token ID exceeds int32"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let py = slf.py();
    let pipeline = slf.into_super().pipeline.clone();
    let rows = py.detach(|| block_on(pipeline.encode(&tokens)))?;
    // NumPy owns the bits and torch shares that storage; no float conversion changes BF16.
    let bits: Vec<u16> = rows.into_iter().map(|value| value.to_bits()).collect();
    let torch = py.import("torch")?;
    Ok(torch
        .call_method1("from_numpy", (bits.into_pyarray(py),))?
        .call_method1("view", (torch.getattr("bfloat16")?,))?
        .call_method1("reshape", ((ids.shape[0], 5120),))?
        .unbind())
}

/// The original video decoder, tile order and stitching, with optional last-frame trimming.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Component, module = "engine.minimax_h3")]
pub(super) struct VideoDecoder;

#[gen_stub_pymethods]
#[pymethods]
impl VideoDecoder {
    #[pyo3(signature = (z, frame_num=None, process_image=false))]
    fn decode_base(
        slf: PyRef<'_, Self>,
        z: &Bound<'_, PyAny>,
        frame_num: Option<usize>,
        process_image: bool,
    ) -> Result<Py<PyAny>, Error> {
        let host = host(z)?;
        let values = floats(&host)?;
        let py = slf.py();
        let pipeline = slf.into_super().pipeline.clone();
        let decoded = py.detach(|| {
            block_on(pipeline.decode_video(&host.shape, &values, frame_num, process_image))
        })?;
        fp32(py, decoded.values, &decoded.shape).map_err(Into::into)
    }
}

/// The original stereo audio decoder, returning channel-major FP32 waveform samples.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Component, module = "engine.minimax_h3")]
pub(super) struct AudioDecoder;

#[gen_stub_pymethods]
#[pymethods]
impl AudioDecoder {
    fn decode(slf: PyRef<'_, Self>, z: &Bound<'_, PyAny>) -> Result<Py<PyAny>, Error> {
        let host = host(z)?;
        let values = floats(&host)?;
        let py = slf.py();
        let pipeline = slf.into_super().pipeline.clone();
        let decoded = py.detach(|| block_on(pipeline.decode_audio(&host.shape, &values)))?;
        fp32(py, decoded.values, &decoded.shape).map_err(Into::into)
    }
}

fn host(value: &Bound<'_, PyAny>) -> PyResult<HostTensor> {
    let host = if let Ok(host) = value.extract::<PyRef<'_, HostTensor>>() {
        HostTensor {
            buffer: host.buffer.to_vec().into(),
            shape: host.shape.clone(),
            dtype: host.dtype,
        }
    } else {
        HostTensor::from_dlpack(value)?
    };
    let bytes = host
        .shape
        .iter()
        .try_fold(host.dtype.size(), |size, &axis| size.checked_mul(axis));
    if bytes != Some(host.buffer.len()) {
        return Err(PyValueError::new_err(
            "host tensor bytes must match its shape and dtype",
        ));
    }
    Ok(host)
}

fn floats(host: &HostTensor) -> PyResult<Vec<f32>> {
    if host.dtype != Dtype::F32 {
        return Err(PyValueError::new_err(
            "decoder latents must be host float32",
        ));
    }
    Ok(host
        .buffer
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect())
}

fn fp32(py: Python<'_>, values: Vec<f32>, shape: &[usize]) -> PyResult<Py<PyAny>> {
    py.import("torch")?
        .call_method1("from_numpy", (values.into_pyarray(py),))?
        .call_method1("reshape", (shape.to_vec(),))
        .map(Bound::unbind)
}
