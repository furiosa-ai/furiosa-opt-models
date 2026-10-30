//! The engine's Python binding, the extension module `furiosa_opt_engine._native` that the
//! `furiosa_opt_engine` wheel installs: devices, the tensors a framework moves between its host and
//! a device, checkpoints and the options that load a model from one, and the requests and cache of
//! a decoder's calls.
//!
//! - `Device(placement)` opens chips; every tensor, buffer and cache on it keeps it open. Any
//!   thread may use and drop them; the transfers and calls on one device run one at a time.
//! - A load returns a `Decoder`, which serves every model alike, as an object of its model type's
//!   class: each model type that loads has a submodule of its own for its classes, such as `qwen3`.
//!   A load of a model's media encoder alone returns an `Encoder`, whatever its model.
//! - The submodule `decoder` holds the requests and cache a decoder's calls take, and what they
//!   return; `vision_language` holds the images and audio a media encoder's calls take.
//! - A failure raises `ValueError` for invalid input or data, `NotImplementedError` for what the
//!   engine does not serve, `RuntimeError` for a device failure and `OSError` for a file that
//!   cannot be read.
//!
//! # Implementation
//!
//! Every object on one device holds the same `Shared` context: the opened device and the runtime
//! that drives it, behind one lock.

use furiosa_opt_models::load::LoadError;
use furiosa_opt_std::prelude::Device;
use furiosa_opt_std::runtime::{DeviceError, ToPlacements};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::io::{self, ErrorKind};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};
use tokio::runtime::Runtime;

mod decoder;
mod model;
mod stub;
mod tensor;
mod vision_language;

pub use stub::stub_info;

/// Registers the engine's Python API.
#[pymodule(name = "_native")]
fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = pyo3_log::try_init();
    // Load
    module.add_class::<model::Checkpoint>()?;
    module.add_class::<model::LoadOptions>()?;
    module.add_class::<model::Decoder>()?;
    module.add_class::<model::Encoder>()?;
    module.add_class::<tensor::Placement>()?;
    // Memory
    module.add_class::<tensor::Device>()?;
    module.add_class::<tensor::HbmBuffer>()?;
    module.add_class::<tensor::HostBuffer>()?;
    module.add_class::<tensor::Transfer>()?;
    module.add_class::<tensor::HostTensor>()?;
    module.add_class::<tensor::HbmTensor>()?;
    // Calls
    submodule(module, "decoder", decoder::register)?;
    submodule(module, "vision_language", vision_language::register)?;
    for (name, register, _) in model::FAMILIES {
        submodule(module, name, register)?;
    }
    Ok(())
}

/// Adds the submodule `name`, whose classes `classes` adds, to `module`.
fn submodule(
    module: &Bound<'_, PyModule>,
    name: &str,
    classes: impl FnOnce(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let py = module.py();
    let child = PyModule::new(py, name)?;
    classes(&child)?;
    module.add_submodule(&child)?;
    // PyO3's `add_submodule` makes no package, so an import finds the child only through
    // `sys.modules`, under the package that re-exports this private module.
    let parent = module.name()?;
    let parent = parent.to_str()?;
    let package = parent
        .rsplit_once('.')
        .map_or(parent, |(package, _)| package);
    let qualified = format!("{package}.{name}");
    child.setattr("__name__", &qualified)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item(qualified, child)
}

/// The opened device and the runtime that drives its transfers and calls.
struct Context {
    device: Device,
    runtime: Runtime,
}

impl Context {
    /// Opens a device on the first of `placements` that is free, with the runtime that drives it.
    fn open(placements: impl ToPlacements) -> Result<Shared, Error> {
        let device = Device::new(placements)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        Ok(Arc::new(Mutex::new(Self { device, runtime })))
    }

    /// The context of `shared`, for a thread that holds the GIL: while another thread holds the
    /// context, this one waits for it without the GIL.
    fn lock<'a>(shared: &'a Shared, py: Python<'_>) -> MutexGuard<'a, Self> {
        loop {
            match shared.try_lock() {
                Ok(context) => return context,
                // A panicked call leaves the device and runtime usable, so later calls still run.
                Err(TryLockError::Poisoned(context)) => return context.into_inner(),
                Err(TryLockError::WouldBlock) => py.detach(|| drop(shared.lock())),
            }
        }
    }

    /// Runs `task`, a host copy or a wait for the device, on the device of `shared` without the
    /// GIL.
    fn transfer<T: Send>(
        shared: &Shared,
        py: Python<'_>,
        task: impl AsyncFnOnce(&mut Device) -> Result<T, furiosa_opt_std::Error> + Send,
    ) -> Result<T, Error> {
        Ok(py.detach(move || {
            let mut context = shared.lock().unwrap_or_else(PoisonError::into_inner);
            let Context { device, runtime } = &mut *context;
            runtime.block_on(task(device))
        })?)
    }
}

/// The context every object on one device holds; the device stays open while any of them lives.
/// Lock order: the context, then the GIL; a GIL holder only try-locks it (`Context::lock`).
type Shared = Arc<Mutex<Context>>;

/// An engine error, raised in Python by its cause as this module lists them, and a memory request
/// the runtime refuses as `ValueError`; a Python error stays itself.
#[derive(Debug)]
struct Error(Box<dyn std::error::Error + Send + Sync>);

impl<E: Into<Box<dyn std::error::Error + Send + Sync>>> From<E> for Error {
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

impl From<Error> for PyErr {
    fn from(Error(error): Error) -> Self {
        let message = error.to_string();
        let error = match error.downcast::<PyErr>() {
            Ok(error) => return *error,
            Err(error) => error,
        };
        if let Some(error) = error.downcast_ref::<LoadError>() {
            return match error {
                LoadError::Missing(_)
                | LoadError::Unexpected(_)
                | LoadError::InvalidData { .. } => PyValueError::new_err(message),
                LoadError::Unsupported { .. } => PyNotImplementedError::new_err(message),
                LoadError::Device { .. } => PyRuntimeError::new_err(message),
            };
        }
        let error = match error.downcast::<io::Error>() {
            Ok(error) => match error.kind() {
                ErrorKind::InvalidInput | ErrorKind::InvalidData => {
                    return PyValueError::new_err(message);
                }
                ErrorKind::Unsupported => return PyNotImplementedError::new_err(message),
                // A call reports a device failure as the source of an `Other` error.
                ErrorKind::Other => match error.into_inner() {
                    Some(source) => source,
                    None => return PyRuntimeError::new_err(message),
                },
                _ => return PyErr::from(*error),
            },
            Err(error) => error,
        };
        match error.downcast_ref::<furiosa_opt_std::Error>() {
            Some(
                furiosa_opt_std::Error::Invalid(_)
                | furiosa_opt_std::Error::Device(DeviceError::ForeignBuffer),
            ) => PyValueError::new_err(message),
            _ => PyRuntimeError::new_err(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_std::prelude::Topology;

    /// The context of a one-chip device, for objects that need a device but run no entry.
    pub(super) fn context() -> Shared {
        Context::open(Topology { chips: 1, pes: 8 }).unwrap()
    }

    /// Each refusal reaches Python as the exception its cause names, with its message: invalid
    /// input or data a `ValueError`, what the engine does not serve a `NotImplementedError`, a
    /// device failure a `RuntimeError`, a file error its `OSError`; a Python error stays itself.
    #[test]
    fn raises_each_refusal_as_its_cause() {
        use furiosa_opt_std::Error as Runtime;
        use pyo3::exceptions::{PyFileNotFoundError, PyTypeError};
        Python::initialize();
        Python::attach(|py| {
            let invalid = || LoadError::InvalidData {
                name: "a".into(),
                reason: "b".into(),
            };
            let other = |source: Runtime| io::Error::new(ErrorKind::Other, source);
            let cases: [(Error, fn(Python<'_>, &PyErr) -> bool, &str); 14] = [
                (
                    LoadError::missing(["a".to_owned()]).into(),
                    |py, error| error.is_instance_of::<PyValueError>(py),
                    "missing from the checkpoint: a",
                ),
                (
                    LoadError::Unexpected(vec!["a".into()]).into(),
                    |py, error| error.is_instance_of::<PyValueError>(py),
                    &LoadError::Unexpected(vec!["a".into()]).to_string(),
                ),
                (
                    invalid().into(),
                    |py, error| error.is_instance_of::<PyValueError>(py),
                    "a: b",
                ),
                (
                    LoadError::Unsupported {
                        name: "a".into(),
                        reason: "b".into(),
                    }
                    .into(),
                    |py, error| error.is_instance_of::<PyNotImplementedError>(py),
                    "a: unsupported, b",
                ),
                (
                    LoadError::Device {
                        path: "a".into(),
                        source: Runtime::Poisoned,
                    }
                    .into(),
                    |py, error| error.is_instance_of::<PyRuntimeError>(py),
                    &format!("a: {}", Runtime::Poisoned),
                ),
                (
                    io::Error::new(ErrorKind::InvalidInput, "c").into(),
                    |py, error| error.is_instance_of::<PyValueError>(py),
                    "c",
                ),
                (
                    io::Error::new(ErrorKind::InvalidData, "c").into(),
                    |py, error| error.is_instance_of::<PyValueError>(py),
                    "c",
                ),
                (
                    io::Error::new(ErrorKind::Unsupported, "c").into(),
                    |py, error| error.is_instance_of::<PyNotImplementedError>(py),
                    "c",
                ),
                (
                    other(Runtime::Invalid("d")).into(),
                    |py, error| error.is_instance_of::<PyValueError>(py),
                    "invalid request: d",
                ),
                (
                    other(Runtime::Poisoned).into(),
                    |py, error| error.is_instance_of::<PyRuntimeError>(py),
                    &Runtime::Poisoned.to_string(),
                ),
                (
                    io::Error::from(ErrorKind::Other).into(),
                    |py, error| error.is_instance_of::<PyRuntimeError>(py),
                    "other error",
                ),
                (
                    io::Error::new(ErrorKind::NotFound, "e").into(),
                    |py, error| error.is_instance_of::<PyFileNotFoundError>(py),
                    "e",
                ),
                (
                    Runtime::Device(DeviceError::ForeignBuffer).into(),
                    |py, error| error.is_instance_of::<PyValueError>(py),
                    &Runtime::Device(DeviceError::ForeignBuffer).to_string(),
                ),
                (
                    PyTypeError::new_err("f").into(),
                    |py, error| error.is_instance_of::<PyTypeError>(py),
                    "f",
                ),
            ];
            for (error, raised, message) in cases {
                let error = PyErr::from(error);
                assert!(raised(py, &error), "{error}");
                assert_eq!(error.value(py).to_string(), message);
            }
        });
    }
}
