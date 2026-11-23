//! One H3 pipeline and component views retaining that same owner and device context.

use crate::tensor::{Device, HostTensor};
use crate::{Error, Shared};
use furiosa_opt_engine::diffusion::minimax_h3 as engine;
use furiosa_opt_engine::residency::block_on;
use furiosa_opt_models::tensor::memory::Dtype;
use numpy::IntoPyArray;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::{Arc, Mutex, PoisonError};

mod input;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Component>()?;
    module.add_class::<Pipeline>()?;
    module.add_class::<Transformer>()?;
    module.add_class::<TextEncoder>()?;
    module.add_class::<VideoDecoder>()?;
    module.add_class::<AudioDecoder>()?;
    module.add_class::<PreparedStep>()?;
    module.add_class::<Output>()?;
    module.add_class::<PreparedText>()?;
    module.add_class::<Encoded>()?;
    module.add_class::<PreparedAudio>()?;
    module.add_class::<AudioOutput>()?;
    module.add_class::<PreparedVideo>()?;
    module.add_class::<VideoDecoding>()?;
    Ok(())
}

/// Prepare checked host weights, then place on the original context outside its device lock.
pub(super) fn load(
    py: Python<'_>,
    options: &furiosa_opt_engine::LoadOptions,
    checkpoint: &furiosa_opt_engine::Checkpoint,
    device: &Device,
) -> Result<Py<PyAny>, Error> {
    let shared = py.detach(|| furiosa_opt_engine::residency::SharedDevice::bind(device.0.clone()));
    let prepared =
        py.detach(|| options.prepare_pipeline::<engine::Pipeline>(checkpoint, shared.placement()))?;
    let pipeline = py.detach(|| block_on(prepared.place(shared)))?;
    Ok(object(py, pipeline, device.0.clone())?.into_any())
}

fn object(
    py: Python<'_>,
    pipeline: Arc<engine::Pipeline>,
    context: Shared,
) -> PyResult<Py<Pipeline>> {
    Py::new(py, (Pipeline, Component { pipeline, context }))
}

/// Shared ownership and placement of a pipeline or any of its component views.
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
    /// Check framework arguments and upload operands explicitly, retaining exclusive staging.
    #[pyo3(signature = (*args, **kwargs))]
    fn prepare(
        slf: PyRef<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Result<PreparedStep, Error> {
        if !args.is_empty() {
            return Err(PyTypeError::new_err("transformer takes forward keyword arguments").into());
        }
        let kwargs = kwargs.ok_or_else(|| {
            PyTypeError::new_err("transformer requires forward keyword arguments")
        })?;
        let values = input::parse(kwargs)?;
        let py = slf.py();
        let pipeline = slf.into_super().pipeline.clone();
        let call = py.detach(|| block_on(pipeline.prepare(values)))?;
        Ok(PreparedStep {
            call: Mutex::new(Some((call, pipeline))),
        })
    }

    /// Start a new generation after the preceding call/result has retired.
    fn reset_cache(slf: PyRef<'_, Self>) -> Result<(), Error> {
        let py = slf.py();
        let pipeline = slf.into_super().pipeline.clone();
        py.detach(|| block_on(pipeline.reset_cache()))?;
        Ok(())
    }
}

/// Uploaded transformer operands. Each prepared call can be consumed exactly once.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.minimax_h3")]
pub(super) struct PreparedStep {
    call: Mutex<Option<(engine::PreparedStep, Arc<engine::Pipeline>)>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PreparedStep {
    fn forward(&self, py: Python<'_>) -> Result<Output, Error> {
        let (call, pipeline) = take(&self.call)?;
        let result = py.detach(|| block_on(call.forward()))?;
        Ok(Output {
            result: Mutex::new(Some((result, pipeline))),
        })
    }

    /// Callback errors retain their Python identity and leave no published cache.
    #[pyo3(signature = (decide, /))]
    fn forward_cached(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "collections.abc.Callable[[tuple[float, float] | None], bool]", imports = ("collections.abc",)))]
        decide: Py<PyAny>,
    ) -> Result<Output, Error> {
        if !decide.bind(py).is_callable() {
            return Err(PyTypeError::new_err("TeaCache decision must be callable").into());
        }
        let (call, pipeline) = take(&self.call)?;
        let mut callback_error = None;
        let result = py.detach(|| {
            block_on(call.forward_cached(|means| {
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
        Ok(Output {
            result: Mutex::new(Some((result?, pipeline))),
        })
    }
}

/// Accepted device velocities; waiting retains both staging admission and unpublished cache.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.minimax_h3")]
pub(super) struct Output {
    result: Mutex<Option<(engine::Output, Arc<engine::Pipeline>)>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Output {
    fn wait(&self, py: Python<'_>) -> Result<(), Error> {
        py.detach(|| {
            let result = self.result.lock().unwrap_or_else(PoisonError::into_inner);
            block_on(result.as_ref().ok_or_else(consumed)?.0.wait()).map_err(Error::from)
        })
    }

    /// Consume device results into host FP32 NumPy tensors using the Rust gather contract.
    #[gen_stub(override_return_type(type_repr = "tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32]]", imports = ("numpy", "numpy.typing")))]
    fn to_host(&self, py: Python<'_>) -> Result<(Py<PyAny>, Py<PyAny>), Error> {
        let (output, pipeline) = take(&self.result)?;
        let output = py.detach(|| block_on(output.to_host()))?;
        let result = (|| {
            Ok((
                fp32(py, output.video, &output.video_shape)?,
                fp32(py, output.audio, &output.audio_shape)?,
            ))
        })();
        if result.is_err() {
            let _ = py.detach(|| block_on(pipeline.reset_cache()));
        }
        result
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
    fn prepare(
        slf: PyRef<'_, Self>,
        input_ids: &Bound<'_, PyAny>,
        media: Option<&Bound<'_, PyDict>>,
    ) -> Result<PreparedText, Error> {
        prepare_text(slf, input_ids, media)
    }

    /// Streaming places each layer before its call; the framework has nothing to move here.
    fn load_to_device(&self) {}
}

fn prepare_text(
    slf: PyRef<'_, TextEncoder>,
    input_ids: &Bound<'_, PyAny>,
    media: Option<&Bound<'_, PyDict>>,
) -> Result<PreparedText, Error> {
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
    let call = py.detach(|| block_on(pipeline.prepare_text(&tokens)))?;
    Ok(PreparedText {
        call: Mutex::new(Some(call)),
        tokens: tokens.len(),
    })
}

#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.minimax_h3")]
pub(super) struct PreparedText {
    call: Mutex<Option<engine::PreparedText>>,
    tokens: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl PreparedText {
    fn forward(&self, py: Python<'_>) -> Result<Encoded, Error> {
        let call = take(&self.call)?;
        let rows = py.detach(|| block_on(call.forward()))?;
        Ok(Encoded {
            rows: Mutex::new(Some(rows)),
            tokens: self.tokens,
        })
    }
}

/// Device encoder rows, before final normalization.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.minimax_h3")]
pub(super) struct Encoded {
    rows: Mutex<Option<engine::Encoded>>,
    tokens: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl Encoded {
    fn wait(&self, py: Python<'_>) -> Result<(), Error> {
        py.detach(|| {
            let rows = self.rows.lock().unwrap_or_else(PoisonError::into_inner);
            block_on(rows.as_ref().ok_or_else(consumed)?.wait()).map_err(Error::from)
        })
    }

    /// Consume into a uint16 NumPy tensor of BF16 bits, preserving every bit for the framework.
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.uint16]", imports = ("numpy", "numpy.typing")))]
    fn to_host(&self, py: Python<'_>) -> Result<Py<PyAny>, Error> {
        let rows = take(&self.rows)?;
        let rows = py.detach(|| block_on(rows.to_host()))?;
        let bits: Vec<u16> = rows.into_iter().map(|value| value.to_bits()).collect();
        Ok(bits
            .into_pyarray(py)
            .call_method1("reshape", ((self.tokens, 5120),))?
            .unbind())
    }
}

/// The original video decoder, tile order and stitching, with optional last-frame trimming.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Component, module = "engine.minimax_h3")]
pub(super) struct VideoDecoder;

#[gen_stub_pymethods]
#[pymethods]
impl VideoDecoder {
    #[pyo3(signature = (z, frame_num=None, process_image=false))]
    fn prepare(
        slf: PyRef<'_, Self>,
        z: &Bound<'_, PyAny>,
        frame_num: Option<usize>,
        process_image: bool,
    ) -> Result<PreparedVideo, Error> {
        let host = host(z)?;
        let values = floats(&host)?;
        let py = slf.py();
        let pipeline = slf.into_super().pipeline.clone();
        let call = py.detach(|| {
            block_on(pipeline.prepare_video(&host.shape, values, frame_num, process_image))
        })?;
        Ok(PreparedVideo {
            call: Mutex::new(Some(call)),
        })
    }
}

/// The original stereo audio decoder, returning channel-major FP32 waveform samples.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Component, module = "engine.minimax_h3")]
pub(super) struct AudioDecoder;

#[gen_stub_pymethods]
#[pymethods]
impl AudioDecoder {
    fn prepare(slf: PyRef<'_, Self>, z: &Bound<'_, PyAny>) -> Result<PreparedAudio, Error> {
        let host = host(z)?;
        let values = floats(&host)?;
        let py = slf.py();
        let pipeline = slf.into_super().pipeline.clone();
        let call = py.detach(|| block_on(pipeline.prepare_audio(&host.shape, &values)))?;
        Ok(PreparedAudio {
            call: Mutex::new(Some(call)),
        })
    }
}

#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.minimax_h3")]
pub(super) struct PreparedAudio {
    call: Mutex<Option<engine::PreparedAudio>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PreparedAudio {
    fn forward(&self, py: Python<'_>) -> Result<AudioOutput, Error> {
        let call = take(&self.call)?;
        let waveform = py.detach(|| block_on(call.forward()))?;
        Ok(AudioOutput {
            waveform: Mutex::new(Some(waveform)),
        })
    }
}

/// Accepted device waveform; host conversion removes only the original per-channel padding.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.minimax_h3")]
pub(super) struct AudioOutput {
    waveform: Mutex<Option<engine::AudioOutput>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl AudioOutput {
    fn wait(&self, py: Python<'_>) -> Result<(), Error> {
        py.detach(|| {
            let waveform = self.waveform.lock().unwrap_or_else(PoisonError::into_inner);
            block_on(waveform.as_ref().ok_or_else(consumed)?.wait()).map_err(Error::from)
        })
    }

    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn to_host(&self, py: Python<'_>) -> Result<Py<PyAny>, Error> {
        let waveform = take(&self.waveform)?;
        let decoded = py.detach(|| block_on(waveform.to_host()))?;
        fp32(py, decoded.values, &decoded.shape).map_err(Into::into)
    }
}

#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.minimax_h3")]
pub(super) struct PreparedVideo {
    call: Mutex<Option<engine::PreparedVideo>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PreparedVideo {
    fn forward(&self, py: Python<'_>) -> Result<VideoDecoding, Error> {
        let call = take(&self.call)?;
        let stream = py.detach(|| block_on(call.forward()))?;
        Ok(VideoDecoding {
            stream: Mutex::new(Some(stream)),
        })
    }
}

/// Ongoing two-slot video decoding. Final frames require ordered CPU download and stitching.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.minimax_h3")]
pub(super) struct VideoDecoding {
    stream: Mutex<Option<engine::VideoDecoding>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl VideoDecoding {
    /// Wait for the currently submitted group; this does not produce or advance final frames.
    fn wait(&self, py: Python<'_>) -> Result<(), Error> {
        py.detach(|| {
            let stream = self.stream.lock().unwrap_or_else(PoisonError::into_inner);
            block_on(stream.as_ref().ok_or_else(consumed)?.wait()).map_err(Error::from)
        })
    }

    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn to_host(&self, py: Python<'_>) -> Result<Py<PyAny>, Error> {
        let stream = take(&self.stream)?;
        let decoded = py.detach(|| block_on(stream.to_host()))?;
        fp32(py, decoded.values, &decoded.shape).map_err(Into::into)
    }
}

fn consumed() -> PyErr {
    PyRuntimeError::new_err("H3 prepared call or device result has already been consumed")
}

fn take<T>(slot: &Mutex<Option<T>>) -> PyResult<T> {
    slot.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
        .ok_or_else(consumed)
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
    values
        .into_pyarray(py)
        .call_method1("reshape", (shape.to_vec(),))
        .map(Bound::unbind)
}
