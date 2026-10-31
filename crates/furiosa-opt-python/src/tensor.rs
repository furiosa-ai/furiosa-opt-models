//! Devices, and the tensors and buffers a framework moves between its host and a device.
//!
//! A framework copies inputs to the device before a call and reads results back after it. A
//! `HostBuffer` holds host bytes and an `HbmBuffer` device bytes; `to_hbm` and `to_host` copy
//! between them and return the copy as a `Transfer` once it is submitted. A `HostTensor` or
//! `HbmTensor` gives such bytes a shape and a dtype. For example,
//! `HostTensor.from_dlpack(rows).to_hbm(device)` copies a PyTorch tensor to `device`. Device
//! memory belongs to one `Device`, which stays open while any buffer on it lives.
//!
//! The device runs a call's launches after those of the calls submitted before it, and after the
//! copies submitted before it on their bytes. A copy runs after the earlier calls and copies that
//! write its bytes, through whichever tensor, slice or buffer names them, and also after those
//! that read the bytes it writes. A copy of pinned `HostBuffer` memory may return before it ends;
//! `Transfer.wait`, every other copy, `HbmTensor.wait` and `Device.synchronize` wait for the
//! device, and nothing else waits for a launch.

use super::{Context, Error, Shared};
use furiosa_opt_models::tensor::memory::Dtype;
use furiosa_opt_std::prelude;
use half::bf16;
use numpy::IntoPyArray;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyMemoryError, PyTypeError, PyValueError};
use pyo3::types::{
    IntoPyDict, PyByteArray, PyInt, PyList, PyMemoryView, PyRange, PyRangeMethods, PySlice,
    PySliceMethods, PyString, PyTuple,
};
use pyo3::{ffi, prelude::*};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use pyo3_stub_gen::{PyStubType, TypeInfo};
use std::env::{self, VarError};
use std::ffi::{c_int, c_void};
use std::ops::Range;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError, TryLockError};
use std::task::{Poll, Waker};
use std::{io, slice};

/// Physical chips, in rank order, and the PEs used on each: where a `Device` opens.
#[gen_stub_pyclass]
#[pyclass(frozen, eq, module = "engine")]
#[derive(Clone, PartialEq)]
pub(super) struct Placement(furiosa_opt_std::Placement);

#[gen_stub_pymethods]
#[pymethods]
impl Placement {
    /// Raises:
    ///     ValueError: `pes` has a step other than 1.
    #[new]
    fn new(
        #[gen_stub(override_type(type_repr = "collections.abc.Sequence[int]", imports = ("collections.abc",)))]
        chips: Vec<u8>,
        #[gen_stub(override_type(type_repr = "range"))] pes: &Bound<'_, PyRange>,
    ) -> PyResult<Self> {
        if pes.step()? != 1 {
            return Err(PyValueError::new_err(
                "pes must be a range of consecutive PEs",
            ));
        }
        let bound = |name: &str| pes.getattr(name)?.extract::<u8>();
        Ok(Self(furiosa_opt_std::Placement {
            chips,
            pes: bound("start")?..bound("stop")?,
        }))
    }

    /// The chips at `positions` of `FURIOSA_VISIBLE_CHIPS`, with `pes` on each: position `i` is
    /// the `i`-th chip the variable lists, or chip `i` while it lists none, as a framework numbers
    /// the devices a process sees.
    ///
    /// Raises:
    ///     ValueError: An entry of the variable is no chip index, or `pes` has a step other
    ///         than 1.
    ///     IndexError: A position lies past the chips the variable lists.
    #[staticmethod]
    fn visible(
        #[gen_stub(override_type(type_repr = "collections.abc.Sequence[int]", imports = ("collections.abc",)))]
        positions: Vec<u8>,
        #[gen_stub(override_type(type_repr = "range"))] pes: &Bound<'_, PyRange>,
    ) -> PyResult<Self> {
        Self::new(
            Self::listed(env::var("FURIOSA_VISIBLE_CHIPS"), &positions)?,
            pes,
        )
    }

    #[getter]
    fn chips(&self) -> Vec<usize> {
        self.0.chips.iter().map(|&chip| chip.into()).collect()
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "range"))]
    fn pes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyRange>> {
        PyRange::new(py, self.0.pes.start.into(), self.0.pes.end.into())
    }

    fn __repr__(&self) -> String {
        let furiosa_opt_std::Placement { chips, pes } = &self.0;
        format!(
            "Placement(chips={chips:?}, pes=range({}, {}))",
            pes.start, pes.end
        )
    }
}

impl Placement {
    /// The chips `Placement.visible` places at `positions` while `FURIOSA_VISIBLE_CHIPS` holds
    /// `value`, read as the runtime reads it: split at commas, entries trimmed, empty ones skipped.
    fn listed(value: Result<String, VarError>, positions: &[u8]) -> PyResult<Vec<u8>> {
        let value = match value {
            Err(VarError::NotPresent) => String::new(),
            value => value
                .map_err(|why| PyValueError::new_err(format!("FURIOSA_VISIBLE_CHIPS: {why}")))?,
        };
        let listed = value
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .map(|entry| {
                entry.parse().map_err(|_| {
                    PyValueError::new_err(format!(
                        "FURIOSA_VISIBLE_CHIPS: `{entry}` is not a chip index"
                    ))
                })
            })
            .collect::<PyResult<Vec<u8>>>()?;
        positions
            .iter()
            .map(|&position| match listed.as_slice() {
                [] => Ok(position),
                chips => chips.get(usize::from(position)).copied().ok_or_else(|| {
                    PyIndexError::new_err(format!(
                        "FURIOSA_VISIBLE_CHIPS lists {} chips, none at position {position}",
                        chips.len()
                    ))
                }),
            })
            .collect()
    }
}

/// An opened device, which every tensor, buffer and cache on it keeps open; its memory sizes are
/// bytes per chip.
#[gen_stub_pyclass]
#[pyclass(module = "engine")]
pub(super) struct Device(pub(super) Shared);

#[gen_stub_pymethods]
#[pymethods]
impl Device {
    /// Open the chips and PEs of `placement`.
    ///
    /// Raises:
    ///     RuntimeError: The chips are taken or fail to open.
    #[new]
    fn new(py: Python<'_>, placement: &Placement) -> Result<Self, Error> {
        let placement = placement.0.clone();
        Ok(Self(py.detach(|| Context::open(placement))?))
    }

    /// Total device memory in bytes per chip.
    #[getter]
    fn capacity(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(Context::lock(&self.0, py).device.memory().capacity())
    }

    /// Unallocated device memory in bytes per chip.
    #[getter]
    fn available(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(Context::lock(&self.0, py).device.memory().usage().available)
    }

    /// Wait until every call submitted on the device before this one has ended.
    ///
    /// Raises:
    ///     RuntimeError: The device failed.
    fn synchronize(&self, py: Python<'_>) -> Result<(), Error> {
        Context::transfer(&self.0, py, async |device| device.synchronize().await)
    }
}

/// Device memory of `len` bytes on each chip, freed once no object refers to it; a copy moves every
/// chip's bytes in chip order, so its host buffer holds `len(buffer)` times the chip count.
#[gen_stub_pyclass]
#[pyclass(module = "engine")]
pub(super) struct HbmBuffer {
    pub(super) inner: prelude::HbmBuffer,
    pub(super) context: Shared,
}

#[gen_stub_pymethods]
#[pymethods]
impl HbmBuffer {
    /// Allocate device memory.
    ///
    /// Args:
    ///     device: Device to allocate on.
    ///     size: Bytes per chip.
    ///
    /// Raises:
    ///     RuntimeError: The allocation fails.
    #[new]
    fn new(py: Python<'_>, device: &Device, size: usize) -> Result<Self, Error> {
        let inner = prelude::HbmBuffer::alloc(&Context::lock(&device.0, py).device, size)?;
        Ok(Self {
            inner,
            context: device.0.clone(),
        })
    }

    /// Size in bytes per chip.
    fn __len__(&self) -> usize {
        self.inner.size()
    }

    /// Return a view of a byte range that shares this storage; the range starts and ends on 8-byte
    /// boundaries, as every device operand does.
    ///
    /// Args:
    ///     offset: Start of the range in bytes per chip.
    ///     size: Length of the range in bytes per chip.
    ///
    /// Raises:
    ///     ValueError: `offset` or `size` is not a multiple of 8, or the range is outside the
    ///         allocation.
    fn slice(&self, offset: usize, size: usize) -> PyResult<Self> {
        if !offset.is_multiple_of(8) || !size.is_multiple_of(8) {
            return Err(PyValueError::new_err(
                "a buffer slice starts and ends at multiples of 8 bytes",
            ));
        }
        // An end past `usize::MAX` saturates, so the runtime refuses it as outside the buffer.
        let bytes = offset..offset.saturating_add(size);
        Ok(Self {
            inner: self.inner.alias(bytes).map_err(Error::from)?,
            context: self.context.clone(),
        })
    }

    /// Copy the bytes on every chip into `host`, chip after chip; return once submitted. It runs
    /// after the earlier calls and copies that write the bytes, before the later ones.
    ///
    /// Raises:
    ///     ValueError: `len(host)` is not this buffer's `len` times the chip count.
    ///     RuntimeError: The transfer, or a call that writes the bytes, failed before this
    ///         returns; the `Transfer` raises a later failure.
    fn to_host(&self, py: Python<'_>, host: &HostBuffer) -> PyResult<Transfer> {
        let source = self.inner.alias(..).map_err(Error::from)?;
        Transfer::new(
            py,
            Context::transfer(&self.context, py, async move |device| {
                let mut memory = host.memory.lock().unwrap_or_else(PoisonError::into_inner);
                Ok::<Running, _>(Box::pin(
                    source
                        .to_host(&mut device.pdma)
                        .output(memory.slice_mut(host.bytes.clone())?)
                        .submit(),
                ))
            })?,
        )
    }
}

/// Host memory that copies read and write, page-locked where the system provides it; a `slice`
/// shares it, so one buffer holds the bytes of many copies. The buffer protocol lets `memoryview`,
/// NumPy and PyTorch read and change its bytes in place once a copy's `Transfer` is done.
///
/// Exporting waits for the copies already submitted on the bytes, and only for those. A later
/// copy through any view of the same bytes reads or writes them while the export is held, so wait
/// for that copy's `Transfer` before using the export.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine")]
pub(super) struct HostBuffer {
    memory: Arc<Mutex<prelude::HostBuffer>>,
    bytes: Range<usize>,
}

/// All of `memory`.
impl From<prelude::HostBuffer> for HostBuffer {
    fn from(memory: prelude::HostBuffer) -> Self {
        Self {
            bytes: 0..memory.len(),
            memory: Arc::new(Mutex::new(memory)),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl HostBuffer {
    /// `source` zeroed bytes, or a copy of the bytes of a C-contiguous buffer such as `bytes` or a
    /// NumPy array.
    ///
    /// Raises:
    ///     OverflowError: `source` is a negative integer.
    ///     TypeError: `source` is neither an integer nor a C-contiguous buffer.
    #[new]
    fn new(
        #[gen_stub(override_type(type_repr = "int | typing_extensions.Buffer", imports = ("typing_extensions",)))]
        source: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        if source.is_instance_of::<PyInt>() {
            return Ok(prelude::HostBuffer::zero(source.extract()?).into());
        }
        let bytes = PyMemoryView::from(source)?.call_method1("cast", ("B",))?;
        let bytes = PyBuffer::<u8>::get(&bytes)?;
        let mut host = prelude::HostBuffer::zero(bytes.item_count());
        bytes.copy_to_slice(source.py(), &mut host)?;
        Ok(host.into())
    }

    /// Size in bytes.
    fn __len__(&self) -> usize {
        self.bytes.len()
    }

    /// Return a view of a byte range that shares this memory; views may overlap, and of two copies
    /// that write the same bytes, the one submitted later lands last.
    ///
    /// Args:
    ///     offset: Start of the range in bytes.
    ///     size: Length of the range in bytes.
    ///
    /// Raises:
    ///     ValueError: `size` is zero, or the range is outside the buffer.
    fn slice(&self, offset: usize, size: usize) -> PyResult<Self> {
        if size == 0 {
            return Err(PyValueError::new_err(
                "a host buffer slice holds at least one byte",
            ));
        }
        let end = offset
            .checked_add(size)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| PyValueError::new_err("the range is outside the host buffer"))?;
        Ok(Self {
            memory: Arc::clone(&self.memory),
            bytes: self.bytes.start + offset..self.bytes.start + end,
        })
    }

    /// Copy the bytes into `buffer`, chip after chip; return once submitted. It runs after the
    /// earlier calls and copies on those bytes, before the later ones.
    ///
    /// Raises:
    ///     ValueError: The size is not `len(buffer)` times the chip count.
    ///     RuntimeError: The transfer failed before this returns; the `Transfer` raises a later
    ///         failure.
    fn to_hbm(&self, py: Python<'_>, buffer: &HbmBuffer) -> PyResult<Transfer> {
        let mut target = buffer.inner.alias(..).map_err(Error::from)?;
        Transfer::new(
            py,
            Context::transfer(&buffer.context, py, async move |device| {
                let memory = self.memory.lock().unwrap_or_else(PoisonError::into_inner);
                Ok::<Running, _>(Box::pin(
                    memory
                        .slice(self.bytes.clone())?
                        .to_hbm(&mut device.pdma)
                        .output(&mut target)
                        .submit(),
                ))
            })?,
        )
    }

    /// Export the bytes, writable, without a copy, once the copies writing or reading them ended.
    ///
    /// Raises:
    ///     RuntimeError: A copy that wrote the bytes failed.
    #[gen_stub(skip)]
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let (buf, len) = {
            let host = slf.get();
            let mut memory = host.memory.lock().unwrap_or_else(PoisonError::into_inner);
            let mut bytes = memory
                .slice_mut(host.bytes.clone())
                .and_then(|bytes| bytes.into_bytes())
                .map_err(Error::from)?;
            (
                bytes.as_mut_ptr().cast::<c_void>(),
                bytes.len() as ffi::Py_ssize_t,
            )
        };
        // SAFETY: the view holds a reference to `slf`, whose memory never moves or resizes; the
        // static `B` format leaves nothing for a release to free.
        match unsafe { ffi::PyBuffer_FillInfo(view, slf.as_ptr(), buf, len, 0, flags) } {
            0 => Ok(()),
            _ => Err(PyErr::fetch(slf.py())),
        }
    }
}

/// A submitted copy between a `HostBuffer` and an `HbmBuffer`. It runs to its end whether or not
/// anyone keeps this or asks it.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine")]
pub(super) struct Transfer(Mutex<State>);

impl Transfer {
    /// The transfer of `copy`, which raises here when the copy already failed.
    fn new(py: Python<'_>, copy: Running) -> PyResult<Self> {
        let transfer = Self(Mutex::new(State::Running(copy)));
        transfer.done(py)?;
        Ok(transfer)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Transfer {
    /// Whether the copy ended, without waiting for it.
    ///
    /// Raises:
    ///     RuntimeError: The copy failed.
    fn done(&self, py: Python<'_>) -> PyResult<bool> {
        let mut state = match self.0.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(state)) => state.into_inner(),
            // A thread that waits holds the state until the copy ended.
            Err(TryLockError::WouldBlock) => return Ok(false),
        };
        let _ = state.poll(&mut std::task::Context::from_waker(Waker::noop()));
        state.ended(py)
    }

    /// Wait until the copy ended.
    ///
    /// Raises:
    ///     RuntimeError: The copy failed.
    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        // A thread that waits holds the state and takes the GIL to end, so this one waits for the
        // state without the GIL.
        let mut state = loop {
            match self.0.try_lock() {
                Ok(state) => break state,
                Err(TryLockError::Poisoned(state)) => break state.into_inner(),
                Err(TryLockError::WouldBlock) => py.detach(|| drop(self.0.lock())),
            }
        };
        if state
            .poll(&mut std::task::Context::from_waker(Waker::noop()))
            .is_pending()
        {
            let running = &mut *state;
            py.detach(|| {
                tokio::runtime::Builder::new_current_thread()
                    .build()?
                    .block_on(std::future::poll_fn(|context| running.poll(context)));
                io::Result::Ok(())
            })?;
        }
        state.ended(py).map(drop)
    }
}

/// A submitted copy until it ended, then its outcome, kept to raise again.
enum State {
    Running(Running),
    Ended(PyResult<()>),
}

/// A submitted copy, which borrows nothing.
type Running = Pin<Box<dyn Future<Output = Result<(), prelude::Error>> + Send>>;

impl State {
    fn poll(&mut self, context: &mut std::task::Context<'_>) -> Poll<()> {
        if let Self::Running(running) = self {
            *self = Self::Ended(
                std::task::ready!(running.as_mut().poll(context))
                    .map_err(|error| Error::from(error).into()),
            );
        }
        Poll::Ready(())
    }

    fn ended(&self, py: Python<'_>) -> PyResult<bool> {
        match self {
            Self::Running(_) => Ok(false),
            Self::Ended(Ok(())) => Ok(true),
            Self::Ended(Err(error)) => Err(error.clone_ref(py)),
        }
    }
}

/// A host tensor, its contents copied on construction, of dtype `bool`, `uint8`, `int8`, `int16`,
/// `int32`, `int64`, `float16`, `bfloat16`, `float32`, `float64`, `float8_e4m3fn` or `float8_e5m2`.
#[gen_stub_pyclass]
#[pyclass(module = "engine")]
pub(super) struct HostTensor {
    pub(super) buffer: prelude::HostBuffer,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: Dtype,
}

#[gen_stub_pymethods]
#[pymethods]
impl HostTensor {
    /// Copy nested Python lists or tuples of numbers as `dtype`, by default int32, the engine's
    /// index type, when every value is an integer and float32 otherwise.
    ///
    /// Raises:
    ///     ValueError: The nesting is ragged, a value does not fit `dtype`, or `dtype` is
    ///         `float8_e4m3fn` (use `from_dlpack`).
    #[new]
    #[pyo3(signature = (values, *, dtype=None))]
    fn new(
        #[gen_stub(override_type(type_repr = "list[object] | tuple[object, ...] | float"))]
        values: &Bound<'_, PyAny>,
        dtype: Option<Element>,
    ) -> PyResult<Self> {
        fn walk<'py>(
            value: &Bound<'py, PyAny>,
            depth: usize,
            shape: &mut Vec<usize>,
            leaves: &mut Vec<Bound<'py, PyAny>>,
        ) -> PyResult<()> {
            let items = match value.cast::<PyList>() {
                Ok(list) => list.iter().collect::<Vec<_>>(),
                Err(_) => match value.cast::<PyTuple>() {
                    Ok(tuple) => tuple.iter().collect(),
                    Err(_) if depth == shape.len() => {
                        leaves.push(value.clone());
                        return Ok(());
                    }
                    Err(_) => return Err(PyValueError::new_err("values must not be ragged")),
                },
            };
            if depth == shape.len() && leaves.is_empty() {
                shape.push(items.len());
            } else if shape.get(depth) != Some(&items.len()) {
                return Err(PyValueError::new_err("values must not be ragged"));
            }
            items
                .iter()
                .try_for_each(|item| walk(item, depth + 1, shape, leaves))
        }
        let (mut shape, mut leaves) = (Vec::new(), Vec::new());
        walk(values, 0, &mut shape, &mut leaves)?;
        let dtype = match dtype {
            Some(Element(dtype)) => dtype,
            None if leaves.iter().all(|leaf| leaf.is_instance_of::<PyInt>()) => Dtype::I32,
            None => Dtype::F32,
        };
        let range = |_| PyValueError::new_err("value does not fit the tensor dtype");
        let bytes = leaves
            .iter()
            .map(|leaf| {
                let integer = || leaf.extract::<i64>();
                Ok(match dtype {
                    Dtype::BOOL => vec![u8::from(leaf.extract::<bool>()?)],
                    Dtype::U8 => vec![u8::try_from(integer()?).map_err(range)?],
                    Dtype::I8 => i8::try_from(integer()?)
                        .map_err(range)?
                        .to_le_bytes()
                        .to_vec(),
                    Dtype::I16 => i16::try_from(integer()?)
                        .map_err(range)?
                        .to_le_bytes()
                        .to_vec(),
                    Dtype::I32 => i32::try_from(integer()?)
                        .map_err(range)?
                        .to_le_bytes()
                        .to_vec(),
                    Dtype::I64 => integer()?.to_le_bytes().to_vec(),
                    Dtype::F16 => half::f16::from_f64(leaf.extract()?).to_le_bytes().to_vec(),
                    Dtype::BF16 => bf16::from_f64(leaf.extract()?).to_le_bytes().to_vec(),
                    Dtype::F32 => (leaf.extract::<f64>()? as f32).to_le_bytes().to_vec(),
                    Dtype::F64 => leaf.extract::<f64>()?.to_le_bytes().to_vec(),
                    _ => return Err(PyValueError::new_err("use from_dlpack for this dtype")),
                })
            })
            .collect::<PyResult<Vec<_>>>()?
            .concat();
        Ok(Self {
            buffer: bytes.into(),
            shape,
            dtype,
        })
    }

    /// Return a zero-filled tensor.
    #[staticmethod]
    #[pyo3(signature = (shape, *, dtype))]
    fn empty(
        #[gen_stub(override_type(type_repr = "list[int]"))] shape: Vec<usize>,
        dtype: Element,
    ) -> PyResult<Self> {
        let size = dtype.bytes(&shape)?;
        let Element(dtype) = dtype;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|error| PyMemoryError::new_err(error.to_string()))?;
        bytes.resize(size, 0);
        Ok(Self {
            buffer: bytes.into(),
            shape,
            dtype,
        })
    }

    /// Copy any host tensor that exports DLPack (NumPy, PyTorch, JAX); a strided source is
    /// copied in logical order.
    ///
    /// Raises:
    ///     ValueError: The tensor is not on the host or its dtype is unsupported.
    #[staticmethod]
    fn from_dlpack(
        #[gen_stub(override_type(type_repr = "object"))] value: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let py = value.py();
        // Versioned capsules also carry read-only arrays; older exporters only know the legacy one.
        let capsule = match value.call_method(
            "__dlpack__",
            (),
            Some(&[("max_version", (1, 0))].into_py_dict(py)?),
        ) {
            Err(error) if error.is_instance_of::<PyTypeError>(py) => {
                value.call_method0("__dlpack__")?
            }
            capsule => capsule?,
        };
        // SAFETY: the capsule names select the DLPack struct layout the exporter promises; the
        // capsule is renamed after copying so ownership passes to this call exactly once.
        unsafe {
            let versioned =
                ffi::PyCapsule_GetPointer(capsule.as_ptr(), c"dltensor_versioned".as_ptr());
            if !versioned.is_null() {
                let managed = versioned.cast::<dlpack::Versioned>();
                let tensor = (*managed).tensor.to_host();
                ffi::PyCapsule_SetName(capsule.as_ptr(), c"used_dltensor_versioned".as_ptr());
                if let Some(deleter) = (*managed).deleter {
                    deleter(managed);
                }
                return tensor;
            }
            ffi::PyErr_Clear();
            let managed = ffi::PyCapsule_GetPointer(capsule.as_ptr(), c"dltensor".as_ptr())
                .cast::<dlpack::Managed>();
            if managed.is_null() {
                return Err(PyErr::fetch(py));
            }
            let tensor = (*managed).tensor.to_host();
            ffi::PyCapsule_SetName(capsule.as_ptr(), c"used_dltensor".as_ptr());
            if let Some(deleter) = (*managed).deleter {
                deleter(managed);
            }
            tensor
        }
    }

    /// Size of each axis.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Element type name.
    #[getter]
    fn dtype(&self) -> Element {
        Element(self.dtype)
    }

    /// Overwrite the contents with raw little-endian `bytes`.
    ///
    /// Raises:
    ///     ValueError: `len(bytes)` differs from the tensor size.
    fn write(
        &mut self,
        #[gen_stub(override_type(type_repr = "bytes"))] bytes: &[u8],
    ) -> PyResult<()> {
        if bytes.len() != self.buffer.len() {
            return Err(PyValueError::new_err("host bytes do not match tensor size"));
        }
        self.buffer.copy_from_slice(bytes);
        Ok(())
    }

    /// Return a host snapshot; NumPy observes BF16 values as lossless F32 values.
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.generic]", imports = ("numpy", "numpy.typing")))]
    fn numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.dtype == Dtype::BF16 {
            return self
                .buffer
                .chunks_exact(2)
                .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
                .collect::<Vec<_>>()
                .into_pyarray(py)
                .call_method1("reshape", (self.shape(),))
                .map(Bound::unbind);
        }
        py.import("numpy")?
            .call_method1(
                "frombuffer",
                (PyByteArray::new(py, &self.buffer), self.dtype().name()?),
            )?
            .call_method1("reshape", (self.shape(),))
            .map(Bound::unbind)
    }

    /// Copy to new device memory, one copy on each chip; the copy ends before this returns,
    /// while earlier calls may still run on the device.
    ///
    /// Args:
    ///     device: Destination device.
    ///
    /// Raises:
    ///     RuntimeError: The transfer fails.
    fn to_hbm(&self, py: Python<'_>, device: &Device) -> PyResult<HbmTensor> {
        let chips = usize::from(Context::lock(&device.0, py).device.topology().chips);
        let copies;
        let source = match chips {
            1 => &self.buffer,
            chips => {
                copies = prelude::HostBuffer::from(self.buffer.repeat(chips));
                &copies
            }
        };
        let inner = Context::transfer(&device.0, py, async move |device| {
            source.to_hbm(&mut device.pdma).await
        })?;
        let buffer = HbmBuffer {
            inner,
            context: device.0.clone(),
        };
        HbmTensor::new(&buffer, self.shape(), self.dtype())
    }
}

/// Dense row-major elements in a `HbmBuffer`, the same on each chip; a call's launches may write it
/// after the call returns, and later calls, `to_host` and `wait` wait for them.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine")]
pub(super) struct HbmTensor {
    pub(super) buffer: prelude::HbmBuffer,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: Dtype,
    pub(super) context: Shared,
}

#[gen_stub_pymethods]
#[pymethods]
impl HbmTensor {
    /// View `buffer` as a tensor of `shape` and `dtype` on each chip.
    ///
    /// Raises:
    ///     ValueError: The buffer is too small for `shape` and `dtype`.
    #[new]
    fn new(
        buffer: &HbmBuffer,
        #[gen_stub(override_type(type_repr = "list[int]"))] shape: Vec<usize>,
        dtype: Element,
    ) -> PyResult<Self> {
        let bytes = dtype.bytes(&shape)?;
        if bytes > buffer.inner.size() {
            return Err(PyValueError::new_err("tensor exceeds device storage"));
        }
        Ok(Self {
            buffer: buffer.inner.alias(..bytes).map_err(Error::from)?,
            shape,
            dtype: dtype.0,
            context: buffer.context.clone(),
        })
    }

    /// Size of each axis.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Element type name.
    #[getter]
    fn dtype(&self) -> Element {
        Element(self.dtype)
    }

    /// The storage backing the tensor.
    #[getter]
    fn buffer(&self) -> Result<HbmBuffer, Error> {
        Ok(HbmBuffer {
            inner: self.buffer.alias(..)?,
            context: self.context.clone(),
        })
    }

    /// Wait for the calls submitted before this one that write the tensor, without copying it.
    ///
    /// Raises:
    ///     RuntimeError: One of them failed.
    fn wait(&self, py: Python<'_>) -> Result<(), Error> {
        let ready = self.buffer.ready();
        Context::transfer(&self.context, py, async move |_| ready.await)
    }

    /// Copy the first chip's elements to host memory; the copy waits for the calls submitted
    /// before it that write these bytes.
    ///
    /// Raises:
    ///     RuntimeError: The transfer fails, or a call that writes the tensor failed.
    fn to_host(&self, py: Python<'_>) -> Result<HostTensor, Error> {
        let source = self.buffer.alias(..)?;
        let mut host = vec![0; source.size()];
        if !host.is_empty() {
            let bytes = &mut host;
            Context::transfer(&self.context, py, async move |device| {
                source
                    .slice(0, ..)?
                    .to_host(&mut device.pdma)
                    .output(bytes)
                    .await
            })?;
        }
        Ok(HostTensor {
            buffer: host.into(),
            shape: self.shape(),
            dtype: self.dtype,
        })
    }

    /// The rows `start:stop` of the leading axis, sharing storage.
    ///
    /// Raises:
    ///     ValueError: The tensor has no axes, or the slice steps by other than one row.
    fn __getitem__(
        &self,
        #[gen_stub(override_type(type_repr = "slice"))] rows: &Bound<'_, PySlice>,
    ) -> PyResult<Self> {
        let leading = *self
            .shape
            .first()
            .ok_or_else(|| PyValueError::new_err("row is out of bounds"))?;
        let rows = rows.indices(isize::try_from(leading)?)?;
        if rows.step != 1 {
            return Err(PyValueError::new_err("a row slice steps by one row"));
        }
        let start = usize::try_from(rows.start)?;
        let row = Element(self.dtype).bytes(&self.shape[1..])?;
        let mut shape = self.shape.clone();
        shape[0] = rows.slicelength;
        Ok(Self {
            buffer: self
                .buffer
                .alias(start * row..(start + rows.slicelength) * row)
                .map_err(Error::from)?,
            shape,
            dtype: self.dtype,
            context: self.context.clone(),
        })
    }
}

/// A tensor element type, which Python names as PyTorch names its dtypes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Element(pub(super) Dtype);

/// Every element type with its PyTorch name and its DLPack `(code, bits)`; the dtypes vLLM
/// exchanges with a platform.
const ELEMENTS: [(&str, Dtype, [u8; 2]); 12] = [
    ("bool", Dtype::BOOL, [6, 8]),
    ("uint8", Dtype::U8, [1, 8]),
    ("int8", Dtype::I8, [0, 8]),
    ("int16", Dtype::I16, [0, 16]),
    ("int32", Dtype::I32, [0, 32]),
    ("int64", Dtype::I64, [0, 64]),
    ("float16", Dtype::F16, [2, 16]),
    ("bfloat16", Dtype::BF16, [4, 16]),
    ("float32", Dtype::F32, [2, 32]),
    ("float64", Dtype::F64, [2, 64]),
    ("float8_e4m3fn", Dtype::F8_E4M3, [10, 8]),
    ("float8_e5m2", Dtype::F8_E5M2, [12, 8]),
];

impl Element {
    fn find(matches: impl Fn(&(&str, Dtype, [u8; 2])) -> bool) -> PyResult<(&'static str, Self)> {
        ELEMENTS
            .iter()
            .find(|element| matches(element))
            .map(|&(name, dtype, _)| (name, Self(dtype)))
            .ok_or_else(|| PyValueError::new_err("unsupported tensor dtype"))
    }

    fn name(self) -> PyResult<&'static str> {
        Self::find(|&(_, dtype, _)| dtype == self.0).map(|(name, _)| name)
    }

    fn from_dlpack(code: [u8; 2]) -> PyResult<Self> {
        Self::find(|&(_, _, element)| element == code).map(|(_, element)| element)
    }

    /// Bytes a contiguous tensor of `shape` spans.
    fn bytes(self, shape: &[usize]) -> PyResult<usize> {
        shape
            .iter()
            .try_fold(self.0.size(), |size, &axis| size.checked_mul(axis))
            .ok_or_else(|| PyValueError::new_err("tensor size overflows"))
    }
}

impl TryFrom<&str> for Element {
    type Error = PyErr;

    fn try_from(name: &str) -> PyResult<Self> {
        Self::find(|&(label, _, _)| label == name).map(|(_, element)| element)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Element {
    type Error = PyErr;

    fn extract(object: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Self::try_from(object.extract::<&str>()?)
    }
}

impl<'py> IntoPyObject<'py> for Element {
    type Target = PyString;
    type Output = Bound<'py, PyString>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Ok(PyString::new(py, self.name()?))
    }
}

impl PyStubType for Element {
    fn type_output() -> TypeInfo {
        TypeInfo::builtin("str")
    }
}

/// The `dlpack.h` C ABI.
mod dlpack {
    use std::ffi::c_void;

    #[repr(C)]
    pub(super) struct Tensor {
        pub(super) data: *const u8,
        pub(super) device: [i32; 2],
        pub(super) ndim: i32,
        pub(super) dtype: [u8; 2],
        pub(super) lanes: u16,
        pub(super) shape: *const i64,
        pub(super) strides: *const i64,
        pub(super) offset: u64,
    }

    #[repr(C)]
    pub(super) struct Managed {
        pub(super) tensor: Tensor,
        context: *mut c_void,
        pub(super) deleter: Option<unsafe extern "C" fn(*mut Managed)>,
    }

    #[repr(C)]
    pub(super) struct Versioned {
        version: [u32; 2],
        context: *mut c_void,
        pub(super) deleter: Option<unsafe extern "C" fn(*mut Versioned)>,
        flags: u64,
        pub(super) tensor: Tensor,
    }
}

impl dlpack::Tensor {
    /// Copy the elements in logical order into a new `HostTensor`.
    ///
    /// # Safety
    /// `self` points at live exporter memory described by its shape and strides.
    unsafe fn to_host(&self) -> PyResult<HostTensor> {
        const CPU: i32 = 1;
        if self.device[0] != CPU {
            return Err(PyValueError::new_err("from_dlpack requires host memory"));
        }
        if self.lanes != 1 {
            return Err(PyValueError::new_err("unsupported tensor dtype"));
        }
        let element = Element::from_dlpack(self.dtype)?;
        let ndim =
            usize::try_from(self.ndim).map_err(|_| PyValueError::new_err("negative rank"))?;
        let axes = |pointer: *const i64| unsafe { slice::from_raw_parts(pointer, ndim) };
        let shape = axes(self.shape)
            .iter()
            .map(|&axis| usize::try_from(axis).map_err(|_| PyValueError::new_err("negative axis")))
            .collect::<PyResult<Vec<_>>>()?;
        let length = element.bytes(&shape)?;
        let Element(dtype) = element;
        if length == 0 {
            return Ok(HostTensor {
                buffer: Vec::<u8>::new().into(),
                shape,
                dtype,
            });
        }
        let width = dtype.size();
        let base = unsafe { self.data.add(self.offset as usize) };
        let contiguous = shape
            .iter()
            .rev()
            .scan(1, |stride, &axis| {
                let current = *stride;
                *stride *= axis;
                Some(current as i64)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>();
        let strides = if self.strides.is_null() {
            &contiguous
        } else {
            axes(self.strides)
        };
        let bytes = if strides == contiguous || shape.iter().all(|&axis| axis <= 1) {
            unsafe { slice::from_raw_parts(base, length) }.to_vec()
        } else {
            (0..length / width)
                .flat_map(|index| {
                    let offset = shape
                        .iter()
                        .zip(strides)
                        .rev()
                        .scan(index, |rest, (&axis, &stride)| {
                            let position = *rest % axis;
                            *rest /= axis;
                            Some(position as isize * stride as isize)
                        })
                        .sum::<isize>();
                    unsafe { slice::from_raw_parts(base.offset(offset * width as isize), width) }
                })
                .copied()
                .collect()
        };
        Ok(HostTensor {
            buffer: bytes.into(),
            shape,
            dtype,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tests::context;
    use furiosa_opt_std::runtime::DeviceError;
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::types::PyDict;
    use std::ffi::CStr;

    /// Run `code` with `HostTensor`, `HostBuffer` and NumPy (`np`) in scope.
    fn run(code: &CStr) {
        Python::initialize();
        Python::attach(|py| {
            let scope = PyDict::new(py);
            scope
                .set_item("HostTensor", py.get_type::<HostTensor>())
                .unwrap();
            scope
                .set_item("HostBuffer", py.get_type::<HostBuffer>())
                .unwrap();
            scope.set_item("np", py.import("numpy").unwrap()).unwrap();
            py.run(code, Some(&scope), None).unwrap();
        });
    }

    /// Run `code` with the `engine` module and an opened single-chip `device` in scope.
    pub(crate) fn run_on_device(code: &CStr) {
        Python::initialize();
        let context = context();
        Python::attach(|py| {
            let engine = PyModule::new(py, "engine").unwrap();
            crate::register(&engine).unwrap();
            let scope = PyDict::new(py);
            scope.set_item("engine", engine).unwrap();
            scope
                .set_item("device", Py::new(py, Device(context)).unwrap())
                .unwrap();
            let result = py.run(code, Some(&scope), None);
            // Functions `code` defines hold `scope` as their globals; clearing it breaks that cycle,
            // so the device closes before the next test opens it.
            scope.clear();
            result.unwrap();
        });
    }

    /// A placement takes a range of consecutive PEs and refuses a strided one with `ValueError`.
    #[test]
    fn refuses_strided_pes() {
        Python::initialize();
        Python::attach(|py| {
            let placement = py.get_type::<Placement>();
            let pes = |step| PyRange::new_with_step(py, 0, 8, step).unwrap();
            assert!(placement.call1((vec![0u8], pes(1))).is_ok());
            let refused = placement.call1((vec![0u8], pes(2))).unwrap_err();
            assert!(refused.is_instance_of::<PyValueError>(py));
            let reason = refused.value(py).to_string();
            assert_eq!(reason, "pes must be a range of consecutive PEs");
        });
    }

    /// A visible placement takes the `i`-th listed chip for position `i`, or chip `i` while none is
    /// listed, reads the list as the runtime does and refuses a bad entry or a position past it.
    #[test]
    fn places_positions_on_visible_chips() {
        Python::initialize();
        let placed = |value: Option<&str>, positions: &[u8]| {
            let value = value.map(str::to_owned).ok_or(VarError::NotPresent);
            Placement::listed(value, positions).map_err(|error| error.to_string())
        };
        let refused = |reason: &str| Err(reason.to_owned());
        assert_eq!(placed(None, &[0, 1]), Ok(vec![0, 1]));
        assert_eq!(placed(None, &[2, 3]), Ok(vec![2, 3]));
        assert_eq!(placed(Some(""), &[1]), Ok(vec![1]));
        assert_eq!(placed(Some("5,3"), &[0, 1]), Ok(vec![5, 3]));
        assert_eq!(placed(Some("5,3"), &[1]), Ok(vec![3]));
        assert_eq!(placed(Some(" 5, ,3,"), &[0, 1]), Ok(vec![5, 3]));
        assert_eq!(
            placed(Some("0-3"), &[0]),
            refused("ValueError: FURIOSA_VISIBLE_CHIPS: `0-3` is not a chip index"),
        );
        assert_eq!(
            placed(Some("5,a"), &[0]),
            refused("ValueError: FURIOSA_VISIBLE_CHIPS: `a` is not a chip index"),
        );
        assert_eq!(
            placed(Some("5,3"), &[2]),
            refused("IndexError: FURIOSA_VISIBLE_CHIPS lists 2 chips, none at position 2"),
        );
    }

    /// Dtypes outside the supported element types are refused with `ValueError` by every way of
    /// making a `HostTensor`.
    #[test]
    fn refuses_unsupported_dtypes() {
        run(c"
for dtype in ('complex64',):
    for make in (
        lambda: HostTensor([1], dtype=dtype),
        lambda: HostTensor.empty([1], dtype=dtype),
        lambda: HostTensor.from_dlpack(np.zeros(1, dtype=dtype)),
    ):
        try:
            make()
        except ValueError as error:
            assert str(error) == 'unsupported tensor dtype', error
        else:
            raise AssertionError(f'{dtype} must be refused')
");
    }

    /// Every NumPy dtype among the supported element types round-trips through lists, DLPack and
    /// `numpy`.
    #[test]
    fn round_trips_numpy_dtypes() {
        run(c"
for dtype in ('bool', 'uint8', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'):
    values = np.array([0, 1, 1], dtype=dtype)
    for host in (HostTensor(values.tolist(), dtype=dtype), HostTensor.from_dlpack(values)):
        assert host.dtype == dtype
        np.testing.assert_array_equal(host.numpy(), values)
");
    }

    /// DLPack exporters are read through the versioned capsule, which carries read-only arrays,
    /// and through the legacy capsule of exporters that do not take `max_version`.
    #[test]
    fn imports_dlpack_capsules() {
        run(c"
class Legacy:
    def __init__(self, array):
        self.array = array
    def __dlpack__(self):
        return self.array.__dlpack__()

frozen = np.frombuffer(np.arange(4, dtype=np.int32).tobytes(), dtype=np.int32)
np.testing.assert_array_equal(HostTensor.from_dlpack(frozen).numpy(), frozen)
legacy = Legacy(np.arange(3, dtype=np.int32))
np.testing.assert_array_equal(HostTensor.from_dlpack(legacy).numpy(), [0, 1, 2])
");
    }

    /// A capsule `from_dlpack` cannot copy is refused with `ValueError` and why: memory off the
    /// host, a negative rank, a negative axis and lanes of a vector dtype; unchanged, it copies.
    #[test]
    fn refuses_capsules_it_cannot_copy() {
        run(c"
import ctypes

class Tensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.c_void_p), ('device', ctypes.c_int32 * 2), ('ndim', ctypes.c_int32),
        ('dtype', ctypes.c_uint8 * 2), ('lanes', ctypes.c_uint16),
        ('shape', ctypes.POINTER(ctypes.c_int64)), ('strides', ctypes.POINTER(ctypes.c_int64)),
        ('offset', ctypes.c_uint64),
    ]

class Managed(ctypes.Structure):
    _fields_ = [('tensor', Tensor), ('context', ctypes.c_void_p), ('deleter', ctypes.c_void_p)]

capsule = ctypes.pythonapi.PyCapsule_New
capsule.restype = ctypes.py_object
capsule.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

class Exporter:
    # Three int32 values on the host as a legacy capsule, `edit` applied to its tensor. The
    # exporter owns everything the capsule points at, and the capsule frees nothing.
    def __init__(self, edit):
        self.values = np.arange(3, dtype=np.int32)
        self.shape = (ctypes.c_int64 * 1)(3)
        self.name = ctypes.create_string_buffer(b'dltensor')
        self.managed = Managed()
        tensor = self.managed.tensor
        tensor.data, tensor.ndim, tensor.lanes = self.values.ctypes.data, 1, 1
        tensor.device[0], tensor.dtype[0], tensor.dtype[1] = 1, 0, 32
        tensor.shape = self.shape
        edit(tensor)

    def __dlpack__(self):
        return capsule(ctypes.addressof(self.managed), ctypes.addressof(self.name), None)

def device(tensor):
    tensor.device[0] = 2
def rank(tensor):
    tensor.ndim = -1
def axis(tensor):
    tensor.shape[0] = -1
def lanes(tensor):
    tensor.lanes = 2

np.testing.assert_array_equal(HostTensor.from_dlpack(Exporter(lambda _: None)).numpy(), [0, 1, 2])
for edit, reason in (
    (device, 'from_dlpack requires host memory'),
    (rank, 'negative rank'),
    (axis, 'negative axis'),
    (lanes, 'unsupported tensor dtype'),
):
    try:
        HostTensor.from_dlpack(Exporter(edit))
    except ValueError as error:
        assert str(error) == reason, error
    else:
        raise AssertionError(f'{edit.__name__} must be refused')
");
    }

    /// A strided DLPack source is copied in logical element order.
    #[test]
    fn copies_strided_dlpack_in_order() {
        run(c"
source = np.arange(12, dtype=np.float32).reshape(3, 4).T
host = HostTensor.from_dlpack(source)
assert host.shape == [4, 3]
np.testing.assert_array_equal(host.numpy(), source)
");
    }

    /// Nested lists take their shape from the nesting; integers default to int32, other numbers
    /// to float32, and bfloat16 snapshots read as float32.
    #[test]
    fn infers_list_shape_and_dtype() {
        run(c"
listed = HostTensor([[1, 2, 3], (4, 5, 6)])
assert listed.shape == [2, 3] and listed.dtype == 'int32'
np.testing.assert_array_equal(listed.numpy(), [[1, 2, 3], [4, 5, 6]])
assert HostTensor([1, 2.5]).dtype == 'float32'
halves = HostTensor([1.0, -2.0], dtype='bfloat16').numpy()
assert halves.dtype == np.float32
np.testing.assert_array_equal(halves, [1.0, -2.0])
");
    }

    /// Ragged nesting, values outside the dtype and a dtype a list cannot spell are refused with
    /// `ValueError`, each with its reason.
    #[test]
    fn refuses_ragged_and_overflowing_lists() {
        run(c"
ragged, wide, dlpack = (
    'values must not be ragged',
    'value does not fit the tensor dtype',
    'use from_dlpack for this dtype',
)
for values, dtype, reason in (
    ([[1], [2, 3]], 'int32', ragged),
    ([[1], 2], 'int32', ragged),
    ([2**31], 'int32', wide),
    ([256], 'uint8', wide),
    ([128], 'int8', wide),
    ([2**15], 'int16', wide),
    ([1.0], 'float8_e4m3fn', dlpack),
    ([1.0], 'float8_e5m2', dlpack),
):
    try:
        HostTensor(values, dtype=dtype)
    except ValueError as error:
        assert str(error) == reason, (values, dtype, error)
    else:
        raise AssertionError(f'{values} as {dtype} must be refused')
for values, dtype in (([255], 'uint8'), ([-128], 'int8'), ([2**15 - 1], 'int16')):
    assert HostTensor(values, dtype=dtype).numpy().tolist() == values
");
    }

    /// An empty tensor holds zeros of its shape; a shape whose bytes overflow the address space is
    /// refused with `ValueError`, and one past what an allocation can hold with `MemoryError`.
    #[test]
    fn refuses_shapes_whose_size_overflows() {
        run(c"
empty = HostTensor.empty([2, 3], dtype='float32')
assert empty.shape == [2, 3] and empty.dtype == 'float32'
np.testing.assert_array_equal(empty.numpy(), np.zeros((2, 3), dtype=np.float32))
try:
    HostTensor.empty([2**62], dtype='float32')
except ValueError as error:
    assert str(error) == 'tensor size overflows', error
else:
    raise AssertionError('a shape of 2**64 bytes must be refused')
try:
    HostTensor.empty([2**61], dtype='float32')
except MemoryError as error:
    reason = \"memory allocation failed because the computed capacity exceeded the collection's maximum\"
    assert str(error) == reason, error
else:
    raise AssertionError('a shape of 2**63 bytes must be refused')
");
    }

    /// A copy or write between storage of unequal lengths is refused with `ValueError` and why,
    /// and a refused write leaves the tensor unchanged.
    #[test]
    fn refuses_transfers_of_unequal_length() {
        run_on_device(
            c"
buffer = engine.HbmBuffer(device, 16)
host = engine.HostTensor([1, 2, 3])
for transfer in (
    lambda: engine.HostBuffer(8).to_hbm(buffer),
    lambda: buffer.to_host(engine.HostBuffer(8)),
):
    try:
        transfer()
    except ValueError as error:
        reason = 'invalid request: the host bytes do not match the memory on its chips'
        assert str(error) == reason, error
    else:
        raise AssertionError('a transfer between unequal lengths must be refused')
try:
    host.write(b'x')
except ValueError as error:
    assert str(error) == 'host bytes do not match tensor size', error
else:
    raise AssertionError('a write of other bytes must be refused')
assert host.numpy().tolist() == [1, 2, 3]
",
        );
    }

    /// `done` returns `False` at once while a copy runs.
    #[test]
    fn reports_pending() {
        Python::initialize();
        Python::attach(|py| {
            assert!(
                !Bound::new(
                    py,
                    Transfer::new(py, Box::pin(std::future::pending())).unwrap(),
                )
                .unwrap()
                .call_method0("done")
                .unwrap()
                .extract::<bool>()
                .unwrap()
            );
        });
    }

    /// A failed copy raises `RuntimeError` with its reason from `wait` and `done` every time they
    /// are asked.
    #[test]
    fn preserves_failure() {
        Python::initialize();
        Python::attach(|py| {
            let mut polled = false;
            let failing = Transfer::new(
                py,
                Box::pin(std::future::poll_fn(move |context| {
                    context.waker().wake_by_ref();
                    if std::mem::replace(&mut polled, true) {
                        Poll::Ready(Err(DeviceError::Poisoned.into()))
                    } else {
                        Poll::Pending
                    }
                })),
            )
            .and_then(|transfer| Bound::new(py, transfer))
            .unwrap();
            for ask in ["wait", "done", "wait"] {
                let failure = failing.call_method0(ask).unwrap_err();
                assert!(
                    failure.is_instance_of::<PyRuntimeError>(py),
                    "{ask}: {failure}"
                );
                assert_eq!(
                    failure.value(py).to_string(),
                    DeviceError::Poisoned.to_string()
                );
            }
        });
    }

    /// A tensor over a buffer views its bytes up to the buffer's end, and one element past it is
    /// refused with `ValueError` and why.
    #[test]
    fn refuses_a_tensor_past_its_storage() {
        run_on_device(
            c"
import struct

buffer = engine.HbmBuffer(device, 16)
engine.HostBuffer(struct.pack('<4i', 4, 3, 2, 1)).to_hbm(buffer)
assert engine.HbmTensor(buffer, [4], 'int32').to_host().numpy().tolist() == [4, 3, 2, 1]
try:
    engine.HbmTensor(buffer, [5], 'int32')
except ValueError as error:
    assert str(error) == 'tensor exceeds device storage', error
else:
    raise AssertionError('a tensor past its buffer must be refused')
",
        );
    }

    /// A buffer slice starts and ends at whole 8-byte words inside the buffer; another range is
    /// refused with `ValueError` and why.
    #[test]
    fn slices_buffers_at_whole_words() {
        run_on_device(
            c"
buffer = engine.HbmBuffer(device, 16)
buffer.slice(8, 8)
aligned = 'a buffer slice starts and ends at multiples of 8 bytes'
outside = 'invalid request: the range is outside the buffer'
for offset, size, reason in ((1, 8, aligned), (0, 4, aligned), (8, 16, outside), (8, 2**64 - 8, outside)):
    try:
        buffer.slice(offset, size)
    except ValueError as error:
        assert str(error) == reason, (offset, size, error)
    else:
        raise AssertionError(f'{offset}, {size} must be refused')
",
        );
    }

    /// A slice of a tensor's leading axis holds those rows; a tensor without axes has no rows and
    /// a slice stepping by other than one row is refused, each with `ValueError` and why.
    #[test]
    fn slices_consecutive_rows() {
        run_on_device(
            c"
rows = engine.HostTensor([[0, 1], [2, 3], [4, 5], [6, 7]]).to_hbm(device)
assert rows[1:3].shape == [2, 2]
assert rows[1:3].to_host().numpy().tolist() == [[2, 3], [4, 5]]
scalar = engine.HbmTensor(engine.HbmBuffer(device, 8), [], 'int64')
for take, reason in (
    (lambda: scalar[0:1], 'row is out of bounds'),
    (lambda: rows[::2], 'a row slice steps by one row'),
):
    try:
        take()
    except ValueError as error:
        assert str(error) == reason, error
    else:
        raise AssertionError(f'{reason}: must be refused')
",
        );
    }

    /// A host slice is a nonempty range inside the buffer or view it is cut from; any other is
    /// refused with `ValueError`.
    #[test]
    fn refuses_empty_and_outside_host_slices() {
        run(c"
host = HostBuffer(32)
view = host.slice(8, 16)
assert (len(host.slice(0, 32)), len(view), len(view.slice(4, 12))) == (32, 16, 12)
for buffer, offset, size in ((host, 0, 0), (host, 24, 16), (host, 2**63, 2**63), (view, 8, 16)):
    try:
        buffer.slice(offset, size)
    except ValueError:
        pass
    else:
        raise AssertionError(f'a slice of {size} bytes at {offset} must be refused')
");
    }

    /// Copies through host slices move exactly the slices' bytes, which the slices' buffer protocol
    /// reads back.
    #[test]
    fn copies_through_host_slices() {
        run_on_device(
            c"
data = engine.HostBuffer(bytes(range(64)))
buffer = engine.HbmBuffer(device, 32)
data.slice(8, 32).to_hbm(buffer).wait()
back = engine.HostBuffer(64)
buffer.slice(8, 16).to_host(back.slice(40, 16)).wait()
assert bytes(back) == bytes(40) + bytes(range(16, 32)) + bytes(8)
assert bytes(back.slice(40, 16)) == bytes(range(16, 32))
",
        );
    }

    /// Of two copies into overlapping host slices, the one submitted later lands last, even when
    /// the earlier one runs longer.
    #[test]
    fn lands_overlapping_host_writes_in_submission_order() {
        run_on_device(
            c"
size = 64 << 20
ones = engine.HbmBuffer(device, size)
engine.HostBuffer(bytes([1]) * size).to_hbm(ones).wait()
twos = engine.HbmBuffer(device, 8)
engine.HostBuffer(bytes([2]) * 8).to_hbm(twos).wait()
host = engine.HostBuffer(size)
copies = [ones.to_host(host), twos.to_host(host.slice(size - 8, 8))]
for copy in copies:
    copy.wait()
assert bytes(host.slice(size - 16, 16)) == bytes([1]) * 8 + bytes([2]) * 8
",
        );
    }
}
