//! Checkpoints, the options that load a registry model from one onto a device, and the decoder or
//! media encoder a load returns.
//!
//! A framework opens a `Checkpoint`, states its choices in `LoadOptions` and loads once; a refused
//! choice raises before any weight is read. A load returns a `Decoder`, which serves every model
//! alike, as an object of its model type's class in a submodule of its own, such as
//! `qwen3.ForCausalLM`; a load of the media encoder alone returns an `Encoder`, whatever its model.
//! A decoder whose projections take LoRA adapters loads them from PEFT checkpoints too.

use super::decoder::{self, Adapter, AdapterSpec, Cache, CacheSpec, Output, Request};
use super::tensor::{Device, Element, HbmTensor};
use super::vision_language::{Audio, Image};
use super::{Context, Error, Shared};
use furiosa_opt_engine as engine;
use furiosa_opt_engine::decoder::Served;
use furiosa_opt_engine::registry::Model;
use furiosa_opt_engine::vision_language;
use furiosa_opt_models::load::{LoadError, Part};
use furiosa_opt_models::models::Class;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyRange, PyRangeMethods};
use pyo3::{PyClass, PyClassInitializer};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::path::PathBuf;
use std::sync::PoisonError;

mod llama;
mod mistral;
mod qwen3;

/// The submodule of each model type that loads, by family: its name, the function that adds its
/// classes to it, and its decoder classes by name.
pub(super) const FAMILIES: [Family; 3] = [
    ("llama", llama::register, llama::CLASSES),
    ("mistral", mistral::register, mistral::CLASSES),
    ("qwen3", qwen3::register, qwen3::CLASSES),
];

type Family = (
    &'static str,
    fn(&Bound<'_, PyModule>) -> PyResult<()>,
    &'static [(&'static str, Constructor)],
);

/// A decoder as an object of one class.
type Constructor = fn(Decoder, Python<'_>) -> PyResult<Py<Decoder>>;

/// The tensors of a checkpoint directory, mapped or read into host memory, and the config of its
/// model.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine")]
pub(super) struct Checkpoint(pub(super) engine::Checkpoint);

#[gen_stub_pymethods]
#[pymethods]
impl Checkpoint {
    /// Map the checkpoint's indexed weights and configuration at `path`, including pipeline
    /// components, with `config` as the model's config when it is given.
    ///
    /// Raises:
    ///     ValueError: The config or a file is malformed, or two files hold one tensor.
    ///     OSError: A file cannot be read.
    #[new]
    #[pyo3(signature = (path, config=None))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        #[gen_stub(override_type(type_repr = "dict[str, typing.Any] | None", imports = ("typing",)))]
        config: Option<&Bound<'_, PyAny>>,
    ) -> Result<Self, Error> {
        // Python's `json` spells an infinite float as a bare `Infinity`, as the reader expects.
        let config = match config {
            Some(config) => {
                let text: String = py
                    .import("json")?
                    .call_method1("dumps", (config,))?
                    .extract()?;
                let parsed = engine::Checkpoint::parse_config(text.as_bytes());
                Some(parsed.map_err(|error| PyValueError::new_err(format!("config: {error}")))?)
            }
            None => None,
        };
        Ok(Self(py.detach(|| engine::Checkpoint::new(path, config))?))
    }

    /// This checkpoint with every tensor copied into host memory, where neither the page cache's
    /// eviction nor a later change to its files reaches them.
    fn read(&self, py: Python<'_>) -> Self {
        Self(py.detach(|| self.0.read()))
    }
}

/// A load's architecture, chips, KV cache form, layers, batch invariance, quantization and LoRA
/// adapters: a method returns new options or raises for a choice no model holds, and `parse` and
/// `load` for one its model refuses.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine")]
pub(super) struct LoadOptions(engine::LoadOptions);

#[gen_stub_pymethods]
#[pymethods]
impl LoadOptions {
    #[new]
    fn new() -> Self {
        Self(engine::LoadOptions::new())
    }

    /// Load onto `chips` chips instead of the one count the model allows.
    ///
    /// Raises:
    ///     ValueError: `chips` is zero.
    fn chips(&self, chips: usize) -> Result<Self, Error> {
        Ok(Self(self.0.clone().chips(chips)?))
    }

    /// Cache keys and values as `dtype`, a tensor dtype name, instead of `bfloat16`.
    ///
    /// Raises:
    ///     ValueError: `dtype` names no tensor dtype.
    fn kv_cache(&self, dtype: Element) -> Self {
        Self(self.0.clone().kv_cache(dtype.0))
    }

    /// Select the architecture `name` instead of the one the checkpoint's config names.
    ///
    /// Raises:
    ///     ValueError: The model list holds no architecture `name`.
    ///     NotImplementedError: The list refuses `name`.
    fn architecture(&self, name: &str) -> Result<Self, Error> {
        Ok(Self(self.0.clone().architecture(name)?))
    }

    /// Load only the layers `layers`, at their positions in the whole model.
    ///
    /// Raises:
    ///     ValueError: `layers` has a step other than 1 or a negative bound.
    fn layers(
        &self,
        #[gen_stub(override_type(type_repr = "range"))] layers: &Bound<'_, PyRange>,
    ) -> PyResult<Self> {
        let layer = |bound: isize| usize::try_from(bound).ok();
        match (
            layers.step()?,
            layer(layers.start()?),
            layer(layers.stop()?),
        ) {
            (1, Some(start), Some(stop)) => Ok(Self(self.0.clone().layers(start..stop))),
            _ => Err(PyValueError::new_err(
                "layers are a range of consecutive layers",
            )),
        }
    }

    /// Serve calls whose rows keep their values' bits whatever other rows share the calls: each
    /// row's attention is tiled by its own history alone.
    fn batch_invariant(&self) -> Self {
        Self(self.0.clone().batch_invariant())
    }

    /// Quantize a BF16 checkpoint's projections at load as `name` says: `fp8_per_tensor` or
    /// `fp8_per_channel`, E4M3 weights under one scale per tensor or per output row.
    ///
    /// Raises:
    ///     ValueError: `name` names no quantization.
    fn quantization(&self, name: &str) -> Result<Self, Error> {
        Ok(Self(self.0.clone().quantization(name.parse()?)))
    }

    /// Load the model's media encoder alone, whose rows its language model reads: the producer
    /// of an encoder cache, for one.
    fn encoder(&self) -> Self {
        Self(self.0.clone().encoder())
    }

    /// Load for calls that run LoRA adapters: `parse` and `load` refuse a model whose projections
    /// take none.
    fn adapters(&self) -> Self {
        Self(self.0.clone().adapters())
    }

    /// Check the choices against `checkpoint`'s config and names, reading no weight and opening no
    /// device.
    ///
    /// Raises:
    ///     ValueError: The config names no listed architecture, no listed model of it serves the
    ///         checkpoint, or a quantized checkpoint is given a quantization.
    ///     NotImplementedError: The list refuses the architecture, or a choice.
    fn parse(&self, checkpoint: &Checkpoint) -> Result<(), Error> {
        self.0.parse(&checkpoint.0)?;
        Ok(())
    }

    /// Load the model `checkpoint` selects strictly into HBM of `device`: a decoder, as an object of
    /// its model type's class such as `qwen3.ForCausalLM`, or an `Encoder` for a load of the media
    /// encoder alone. Dropping the object frees its device memory, and a new load of the same
    /// model serves the caches the dropped one did.
    ///
    /// Raises:
    ///     ValueError: As `parse` does, or a tensor is missing, unexpected or malformed.
    ///     NotImplementedError: As `parse` does, or no class holds the model yet.
    ///     RuntimeError: The device fails.
    #[gen_stub(override_return_type(type_repr = "Decoder | Encoder"))]
    fn load(
        &self,
        py: Python<'_>,
        checkpoint: &Checkpoint,
        device: &Device,
    ) -> Result<Py<PyAny>, Error> {
        let config = self.0.parse(&checkpoint.0)?;
        // A decoder's class is its model type's, found before a weight is read.
        let unbound = || LoadError::Unsupported {
            name: config.architecture().into(),
            reason: "no Python class holds the model yet".into(),
        };
        let class = match config.part() {
            Part::Encoder => None,
            Part::Whole => Some(Decoder::class(config.class()).ok_or_else(unbound)?),
        };
        let model = py.detach(|| {
            let mut context = device.0.lock().unwrap_or_else(PoisonError::into_inner);
            self.0.load(&mut context.device, &checkpoint.0)
        })?;
        let context = device.0.clone();
        Ok(match (model, class) {
            (Model::Decoder(served), Some(class)) => {
                let decoder = Decoder {
                    served,
                    config,
                    context,
                };
                class(decoder, py)?.into_any()
            }
            (Model::Encoder(encoder), None) => {
                Py::new(py, Encoder { encoder, context })?.into_any()
            }
            _ => {
                unreachable!("a load of the encoder alone serves a media encoder, others a decoder")
            }
        })
    }
}

/// A decoder loaded on a device, whatever its model: what its calls hold and return, the adapters
/// they run with, and a call on a cache. A load returns an object of its model type's class, which
/// derives from this one.
#[gen_stub_pyclass]
#[pyclass(frozen, subclass, module = "engine")]
pub(super) struct Decoder {
    served: Box<dyn Served>,
    config: engine::Config,
    context: Shared,
}

impl Decoder {
    /// The constructor of `class`, which the submodule of its family lists under its name.
    fn class(class: Class) -> Option<Constructor> {
        let (.., classes) = FAMILIES
            .iter()
            .find(|(family, ..)| *family == class.family)?;
        let (_, constructor) = classes.iter().find(|(name, _)| *name == class.name)?;
        Some(*constructor)
    }

    /// This decoder as an object of its model type's class `C`.
    fn object<C: PyClass<BaseType = Self> + Default>(self, py: Python<'_>) -> PyResult<Py<Self>> {
        let object = PyClassInitializer::from(self).add_subclass(C::default());
        Ok(Py::new(py, object)?.into_bound(py).into_super().unbind())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Decoder {
    /// The checkpoint's size, as the hub spells it, such as `0.6B`.
    #[getter]
    fn size(&self) -> &'static str {
        self.config.size()
    }

    /// The dtype of the projection weights.
    #[getter]
    fn precision(&self) -> Element {
        Element(self.config.precision())
    }

    /// The chips the model runs on.
    #[getter]
    fn chips(&self) -> usize {
        self.config.chips()
    }

    /// The dtype the cache stores keys and values in.
    #[getter]
    fn kv_cache(&self) -> Element {
        Element(self.config.kv_cache())
    }

    /// The cache the model's calls read.
    #[getter]
    fn cache(&self) -> CacheSpec {
        CacheSpec::of(&*self.served, &self.context)
    }

    /// Positions a request reaches at most: the context the model attends.
    #[getter]
    fn window(&self) -> usize {
        self.served.limits().window
    }

    /// Rows of one call at most, across its requests.
    #[getter]
    fn rows(&self) -> usize {
        self.served.limits().rows
    }

    /// What a call returns of each kept row.
    #[getter]
    fn output(&self) -> Output {
        self.served.output().into()
    }

    /// The LoRA adapters the model's calls run with, or `None` for a model that runs none.
    #[getter]
    fn adapters(&self) -> Option<AdapterSpec> {
        AdapterSpec::of(&*self.served, &self.context)
    }

    /// The rows `requests` keep, in request order and then position order, as `output` names
    /// them: one-dimensional I32 token IDs, `[rows, hidden]` BF16 hidden values or
    /// `[rows, columns]` FP32 scores with the label columns first. They are returned once the
    /// device accepted the call; reading them waits for it. A request with an `adapter` runs with
    /// that slot of `adapters`, a list of at most `AdapterSpec.SLOTS`.
    ///
    /// Raises:
    ///     ValueError: A request breaks a rule of the call, names an empty slot, or a tensor or an
    ///         adapter lies on another device or belongs to another model.
    ///     NotImplementedError: A request asks for what the model does not serve.
    ///     RuntimeError: The device refuses the call.
    #[pyo3(signature = (requests, cache, adapters=None))]
    fn forward(
        &self,
        py: Python<'_>,
        requests: Vec<PyRef<'_, Request>>,
        cache: &Cache,
        #[gen_stub(override_type(
            type_repr = "collections.abc.Sequence[decoder.Adapter | None] | None",
            imports = ("collections.abc",)
        ))]
        adapters: Option<Vec<Option<PyRef<'_, Adapter>>>>,
    ) -> Result<HbmTensor, Error> {
        let adapters = adapters.unwrap_or_default();
        decoder::forward(py, &*self.served, &self.context, cache, requests, adapters)
    }
}

/// A media encoder loaded on a device, whatever its model: its calls turn images, video frames and
/// audio into rows of its language model's width.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine")]
pub(super) struct Encoder {
    encoder: Box<dyn vision_language::Encoder>,
    context: Shared,
}

#[gen_stub_pymethods]
#[pymethods]
impl Encoder {
    /// The rows of each of `images`, `[patches, width]` BF16 in order, returned once the device
    /// accepted the call; reading them waits for it.
    ///
    /// Raises:
    ///     ValueError: A patch is not of the model's features, a position is outside those the
    ///         model places, or a tensor lies on another device.
    ///     NotImplementedError: An image holds more patches than a call takes.
    ///     RuntimeError: The device refuses the call.
    fn vision(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "collections.abc.Sequence[vision_language.Image]", imports = ("collections.abc",)))]
        images: Vec<PyRef<'_, Image>>,
    ) -> Result<Vec<HbmTensor>, Error> {
        let images = (images.iter())
            .map(|image| image.on(&self.context))
            .collect::<Result<Vec<_>, Error>>()?;
        let rows = py.detach(|| {
            let mut shared = self.context.lock().unwrap_or_else(PoisonError::into_inner);
            let Context { device, runtime } = &mut *shared;
            runtime.block_on(self.encoder.vision(device, &images))
        })?;
        Ok(self.tensors(rows))
    }

    /// The rows of each of `clips`, `[frames, width]` BF16 in order, returned once the device
    /// accepted the call; reading them waits for it.
    ///
    /// Raises:
    ///     ValueError: A frame is not of the model's samples, or a tensor lies on another device.
    ///     NotImplementedError: A clip holds more frames than a call takes.
    ///     RuntimeError: The device refuses the call.
    fn audio(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "collections.abc.Sequence[vision_language.Audio]", imports = ("collections.abc",)))]
        clips: Vec<PyRef<'_, Audio>>,
    ) -> Result<Vec<HbmTensor>, Error> {
        let clips = (clips.iter())
            .map(|clip| clip.on(&self.context))
            .collect::<Result<Vec<_>, Error>>()?;
        let rows = py.detach(|| {
            let mut shared = self.context.lock().unwrap_or_else(PoisonError::into_inner);
            let Context { device, runtime } = &mut *shared;
            runtime.block_on(self.encoder.audio(device, &clips))
        })?;
        Ok(self.tensors(rows))
    }
}

impl Encoder {
    /// Each of a call's `rows` as a tensor on the encoder's device.
    fn tensors(&self, rows: Vec<engine::decoder::HbmTensor>) -> Vec<HbmTensor> {
        (rows.into_iter())
            .map(|rows| HbmTensor {
                buffer: rows.buffer,
                shape: rows.shape,
                dtype: rows.dtype,
                context: self.context.clone(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_engine::registry::{ARCHITECTURES, Implementation};
    use pyo3::types::PyDict;

    /// Every implementation the registry lists loads as a class: a decoder as the class its
    /// family's submodule lists under its name, a media encoder as `Encoder`.
    #[test]
    fn binds_a_class_to_every_listed_implementation() {
        let unbound: Vec<_> = (ARCHITECTURES.iter())
            .flat_map(|&(_, implementations)| implementations)
            .map(Implementation::class)
            .filter(|class| class.name != "Encoder" && Decoder::class(*class).is_none())
            .collect();
        assert!(unbound.is_empty(), "{unbound:?}");
    }

    /// A config that Python's `json` writes with a `NaN` is refused with `ValueError` and where it
    /// stops parsing, and loaded layers are a range of consecutive ones: another step or a negative
    /// bound is refused with `ValueError`.
    #[test]
    fn refuses_configs_and_layers_it_cannot_read() {
        Python::initialize();
        Python::attach(|py| {
            let scope = PyDict::new(py);
            let classes = [
                ("Checkpoint", py.get_type::<Checkpoint>()),
                ("LoadOptions", py.get_type::<LoadOptions>()),
            ];
            for (name, class) in classes {
                scope.set_item(name, class).unwrap();
            }
            let code = c"
cases = (
    (lambda: Checkpoint('.', {'a': float('nan')}), 'config: expected value at line 1 column 7'),
    (lambda: LoadOptions().layers(range(0, 4, 2)), 'layers are a range of consecutive layers'),
    (lambda: LoadOptions().layers(range(-1, 2)), 'layers are a range of consecutive layers'),
    (lambda: LoadOptions().layers(range(0, -1)), 'layers are a range of consecutive layers'),
)
for refused, reason in cases:
    try:
        refused()
    except ValueError as error:
        assert str(error) == reason, error
    else:
        raise AssertionError(reason)
LoadOptions().layers(range(2, 4))
";
            py.run(code, Some(&scope), None).unwrap();
        });
    }
}
