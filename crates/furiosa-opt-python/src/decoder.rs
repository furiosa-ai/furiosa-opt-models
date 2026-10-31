//! The requests and cache of a decoder's calls, as a framework builds them.
//!
//! - A `Request` appends token IDs to one sequence, in the cache blocks its `rows` name.
//! - A loaded decoder describes its cache as a `CacheSpec` on its device; `Cache(spec, blocks)`
//!   allocates `blocks` blocks of every layer there, which the framework assigns to sequences.
//! - A decoder whose projections take LoRA adapters describes them as an `AdapterSpec`;
//!   `Adapter(spec, checkpoint)` loads a PEFT checkpoint onto its device as one.
//! - A decoder's `forward(requests, cache, adapters)` takes a list of requests, the cache they
//!   append to and the adapters they run with, which a request names by index, and returns the
//!   rows they keep as its `Output` names them.

use super::model::Checkpoint;
use super::tensor::{HbmBuffer, HbmTensor};
use super::{Context, Error, Shared};
use furiosa_opt_engine::decoder::{self, Served};
use furiosa_opt_models::tensor::memory::Dtype;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyRange, PyRangeMethods};
use pyo3_stub_gen::derive::{
    gen_stub_pyclass, gen_stub_pyclass_complex_enum, gen_stub_pyclass_enum, gen_stub_pymethods,
};
use pyo3_stub_gen::{PyStubType, TypeInfo};
use std::sync::{Arc, PoisonError};

/// Adds the request and cache classes to `module`.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Request>()?;
    module.add_class::<Keep>()?;
    module.add_class::<Output>()?;
    module.add_class::<CacheSpec>()?;
    module.add_class::<Pool>()?;
    module.add_class::<Group>()?;
    module.add_class::<Kind>()?;
    module.add_class::<Cache>()?;
    module.add_class::<AdapterSpec>()?;
    module.add_class::<Adapter>()?;
    Ok(())
}

/// The rows `model` on `context`'s device keeps of `requests` on `cache` with `adapters`, called
/// without the GIL and returned once the device accepted it: `[rows]` for one-element rows, else
/// `[rows, width]`.
pub(super) fn forward(
    py: Python<'_>,
    model: &dyn Served,
    context: &Shared,
    cache: &Cache,
    requests: Vec<PyRef<'_, Request>>,
    adapters: Vec<Option<PyRef<'_, Adapter>>>,
) -> Result<HbmTensor, Error> {
    if !Arc::ptr_eq(&cache.context, context) {
        return Err(PyValueError::new_err("the cache lies on another device").into());
    }
    let requests = (requests.iter())
        .map(|request| request.on(context))
        .collect::<Result<Vec<_>, Error>>()?;
    let adapters = (adapters.iter())
        .map(|adapter| {
            adapter
                .as_ref()
                .map(|adapter| adapter.on(context))
                .transpose()
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let rows = py.detach(|| {
        let mut shared = context.lock().unwrap_or_else(PoisonError::into_inner);
        let Context { device, runtime } = &mut *shared;
        runtime.block_on(model.forward(device, &cache.inner, &requests, &adapters))
    })?;
    Ok(HbmTensor {
        buffer: rows.buffer,
        shape: rows.shape,
        dtype: rows.dtype,
        context: context.clone(),
    })
}

/// One sequence's `tokens` after the `cached` positions its blocks hold: row `i`, position
/// `p = cached + i`, writes slot `p % block` of block `rows[0][p // block]` and attends `0..=p`.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.decoder")]
pub(super) struct Request {
    /// The sequence's block IDs, a list for each group of the cache.
    #[pyo3(get)]
    rows: Vec<Vec<u32>>,
    /// Positions already in the cache.
    #[pyo3(get)]
    cached: usize,
    /// One-dimensional I32 token IDs.
    #[pyo3(get)]
    tokens: Py<HbmTensor>,
    /// The appended rows the call returns.
    #[pyo3(get)]
    keep: Kept,
    /// The slot of the adapter the sequence runs with.
    #[pyo3(get)]
    adapter: Option<usize>,
    /// I32 positions `[R, n]` of the `n` appended rows, where `R` is 1, or 3 for rotary positions
    /// in three sections; `None` is position `cached + i` for row `i`.
    #[pyo3(get)]
    positions: Option<Py<HbmTensor>>,
    /// `[n, n]` booleans: whether appended row `i` sees appended row `j`; every row sees the cached
    /// positions. `None` is causal.
    #[pyo3(get)]
    mask: Option<Py<HbmTensor>>,
    /// BF16 rows of the body's width that replace the embeddings of the appended rows from each
    /// offset on.
    #[pyo3(get)]
    overrides: Vec<(usize, Py<HbmTensor>)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Request {
    /// `tokens` lie on the model's device. The engine never reads an ID, so the framework keeps
    /// every ID within the model's vocabulary: an ID outside it reads other device memory.
    ///
    /// Raises:
    ///     ValueError: `tokens` are not one-dimensional I32 IDs, or `keep` is a range with a step
    ///         other than 1 or a negative bound.
    #[new]
    #[pyo3(
        signature = (
            rows, cached, tokens, keep=Kept(decoder::Keep::Last),
            *, adapter=None, positions=None, mask=None, overrides=Vec::new(),
        ),
        text_signature = "(rows, cached, tokens, keep=..., *, adapter=None, positions=None, \
                          mask=None, overrides=[])"
    )]
    fn new(
        rows: Vec<Vec<u32>>,
        cached: usize,
        tokens: Bound<'_, HbmTensor>,
        keep: Kept,
        adapter: Option<usize>,
        positions: Option<Py<HbmTensor>>,
        mask: Option<Py<HbmTensor>>,
        overrides: Vec<(usize, Py<HbmTensor>)>,
    ) -> PyResult<Self> {
        let ids = tokens.get();
        if ids.shape.len() != 1 || ids.dtype != Dtype::I32 {
            return Err(PyValueError::new_err("tokens are one-dimensional I32 IDs"));
        }
        Ok(Self {
            rows,
            cached,
            tokens: tokens.unbind(),
            keep,
            adapter,
            positions,
            mask,
            overrides,
        })
    }
}

impl Request {
    /// This request as the call on the device of `context` reads it; refuses a tensor on another
    /// device.
    fn on(&self, context: &Shared) -> Result<decoder::Request, Error> {
        let buffer = |tensor: &Py<HbmTensor>| -> Result<_, Error> {
            let tensor = tensor.get();
            if !Arc::ptr_eq(&tensor.context, context) {
                return Err(
                    PyValueError::new_err("a request tensor lies on another device").into(),
                );
            }
            Ok(tensor.buffer.alias(..)?)
        };
        let tokens = decoder::Rows::new(&buffer(&self.tokens)?, 0..self.tokens.get().shape[0])?;
        Ok(decoder::Request {
            rows: self.rows.clone(),
            cached: self.cached,
            tokens,
            keep: self.keep.0,
            adapter: self.adapter,
            positions: self.positions.as_ref().map(buffer).transpose()?,
            mask: self.mask.as_ref().map(buffer).transpose()?,
            overrides: (self.overrides.iter())
                .map(|(offset, rows)| Ok((*offset, buffer(rows)?)))
                .collect::<Result<_, Error>>()?,
        })
    }
}

/// The appended rows of a request that a call returns; `range(a, b)` returns rows `a` to `b - 1`.
#[gen_stub_pyclass_enum]
#[pyclass(
    frozen,
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE",
    module = "engine.decoder"
)]
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Keep {
    /// None of them; the call only writes their keys and values.
    None,
    /// The last one.
    Last,
    /// All of them, in position order.
    All,
}

/// What a decoder's call returns of each kept row.
#[gen_stub_pyclass_enum]
#[pyclass(
    frozen,
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE",
    module = "engine.decoder"
)]
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Output {
    /// Its greedy token ID.
    Tokens,
    /// Its hidden values.
    Hidden,
    /// Its scores.
    Scores,
}

impl From<decoder::Output> for Output {
    fn from(output: decoder::Output) -> Self {
        match output {
            decoder::Output::Tokens => Self::Tokens,
            decoder::Output::Hidden => Self::Hidden,
            decoder::Output::Scores => Self::Scores,
        }
    }
}

/// A `Keep`, or the `range` of the rows a request keeps.
#[derive(Clone, Copy)]
struct Kept(decoder::Keep);

impl<'a, 'py> FromPyObject<'a, 'py> for Kept {
    type Error = PyErr;

    fn extract(object: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let Ok(rows) = object.cast::<PyRange>() else {
            return Ok(Self(match *object.extract::<PyRef<'py, Keep>>()? {
                Keep::None => decoder::Keep::None,
                Keep::Last => decoder::Keep::Last,
                Keep::All => decoder::Keep::All,
            }));
        };
        let row = |bound: isize| usize::try_from(bound).ok();
        match (rows.step()?, row(rows.start()?), row(rows.stop()?)) {
            (1, Some(start), Some(stop)) => Ok(Self(decoder::Keep::Range(start, stop))),
            _ => Err(PyValueError::new_err(
                "kept rows are a range of consecutive rows",
            )),
        }
    }
}

impl<'py> IntoPyObject<'py> for Kept {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self.0 {
            decoder::Keep::None => Keep::None.into_bound_py_any(py),
            decoder::Keep::Last => Keep::Last.into_bound_py_any(py),
            decoder::Keep::All => Keep::All.into_bound_py_any(py),
            decoder::Keep::Range(start, stop) => {
                PyRange::new(py, start.try_into()?, stop.try_into()?).map(Bound::into_any)
            }
        }
    }
}

impl PyStubType for Kept {
    fn type_output() -> TypeInfo {
        TypeInfo::unqualified("Keep | range")
    }
}

/// The cache a loaded decoder's calls read: block formats, and the groups of layers that use
/// them. `Cache(spec, blocks)` allocates its storage on the decoder's device.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.decoder")]
pub(super) struct CacheSpec {
    inner: decoder::CacheSpec,
    context: Shared,
}

impl CacheSpec {
    /// The cache the calls of `model`, on the device of `context`, read.
    pub(super) fn of(model: &dyn Served, context: &Shared) -> Self {
        Self {
            inner: model.limits().cache,
            context: context.clone(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl CacheSpec {
    /// Block formats, which groups name by index.
    #[getter]
    fn pools(&self) -> Vec<Pool> {
        self.inner.pools.iter().copied().map(Pool).collect()
    }

    /// The cached layers, a kind and a pool each; a request names its blocks of every group.
    #[getter]
    fn groups(&self) -> Vec<Group> {
        self.inner.groups.iter().cloned().map(Group).collect()
    }
}

/// A block format.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.decoder")]
pub(super) struct Pool(decoder::Pool);

#[gen_stub_pymethods]
#[pymethods]
impl Pool {
    /// Tokens one block holds.
    #[getter]
    fn block(&self) -> usize {
        self.0.block
    }

    /// Bytes one block of one layer spans: a page.
    #[getter]
    fn bytes(&self) -> usize {
        self.0.bytes
    }
}

/// Layers whose cache has one kind and takes its blocks from one pool.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.decoder")]
pub(super) struct Group(decoder::Group);

#[gen_stub_pymethods]
#[pymethods]
impl Group {
    /// Index of the pool in `CacheSpec.pools`.
    #[getter]
    fn pool(&self) -> usize {
        self.0.pool
    }

    /// The model layers the group caches.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "range"))]
    fn layers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyRange>> {
        let layers = &self.0.layers;
        PyRange::new(py, layers.start.try_into()?, layers.end.try_into()?)
    }

    /// What the group's blocks hold.
    #[getter]
    fn kind(&self) -> Kind {
        match self.0.kind {
            decoder::Kind::Paged => Kind::Paged {},
            decoder::Kind::State => Kind::State {},
            decoder::Kind::Cross { length } => Kind::Cross { length },
        }
    }

    /// Positions a row attends back to in a sliding window; `None` attends the whole history.
    #[getter]
    fn window(&self) -> Option<usize> {
        self.0.window
    }
}

/// What the blocks of a group hold.
#[gen_stub_pyclass_complex_enum]
#[pyclass(frozen, module = "engine.decoder")]
pub(super) enum Kind {
    /// Keys and values of positions, a block of positions each, in the request's block order.
    Paged {},
    /// A recurrent state a block: the current one, then one for each draft position.
    State {},
    /// Keys and values of `length` encoder positions, which a call writes once and later calls
    /// read.
    Cross { length: usize },
}

/// The storage of a `CacheSpec`: `blocks` blocks of every layer it lists, on the device of the
/// decoder it describes. The framework assigns the blocks to sequences.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.decoder")]
pub(super) struct Cache {
    inner: decoder::Cache,
    context: Shared,
}

#[gen_stub_pymethods]
#[pymethods]
impl Cache {
    /// Allocate `blocks` blocks of every layer `spec` lists.
    ///
    /// Raises:
    ///     ValueError: `blocks` is zero or more than a layer's I32 byte offsets reach.
    ///     RuntimeError: The allocation fails.
    #[new]
    fn new(py: Python<'_>, spec: &CacheSpec, blocks: usize) -> Result<Self, Error> {
        let device = &Context::lock(&spec.context, py).device;
        Ok(Self {
            inner: decoder::Cache::new(device, &spec.inner, blocks)?,
            context: spec.context.clone(),
        })
    }

    /// Blocks per layer.
    #[getter]
    fn blocks(&self) -> usize {
        self.inner.blocks()
    }

    /// Each layer's storage, group by group, for host snapshots of assigned blocks; a copy to the
    /// host waits for the calls submitted before it that write these bytes.
    #[getter]
    fn layers(&self) -> Result<Vec<HbmBuffer>, Error> {
        let layer = |layer: &furiosa_opt_std::prelude::HbmBuffer| -> Result<_, Error> {
            Ok(HbmBuffer {
                inner: layer.alias(..)?,
                context: self.context.clone(),
            })
        };
        self.inner.layers().iter().map(layer).collect()
    }

    /// The spec the cache was allocated for.
    #[getter]
    fn spec(&self) -> CacheSpec {
        CacheSpec {
            inner: self.inner.spec().clone(),
            context: self.context.clone(),
        }
    }
}

/// The LoRA adapters a loaded decoder's calls run their requests with: `RANK`, the rank every
/// update holds, and `SLOTS`, the adapters one call runs with. `Adapter(spec, checkpoint)` loads
/// one onto the decoder's device.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.decoder")]
pub(super) struct AdapterSpec {
    load: decoder::Loader,
    context: Shared,
}

impl AdapterSpec {
    /// The adapters of `model`, on the device of `context`; none when its calls run none.
    pub(super) fn of(model: &dyn Served, context: &Shared) -> Option<Self> {
        model.adapters().map(|load| Self {
            load,
            context: context.clone(),
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl AdapterSpec {
    /// The rank an adapter's updates hold at most: a checkpoint of a lower rank is zero-padded.
    #[classattr]
    const RANK: usize = furiosa_opt_models::modules::lora::RANK;

    /// The adapters one call runs with at most: the length of the list `forward` takes.
    #[classattr]
    const SLOTS: usize = decoder::SLOTS;
}

/// A PEFT LoRA checkpoint loaded onto a decoder's device as its adapter tree: each adapted
/// module's `lora_A` and `lora_B`, the scale folded in. Nothing changes it, so a call reads it in
/// place; a framework makes it active by putting it in a slot of the list a call takes.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.decoder")]
pub(super) struct Adapter {
    inner: decoder::Adapter,
    context: Shared,
}

#[gen_stub_pymethods]
#[pymethods]
impl Adapter {
    /// Load `checkpoint`, a PEFT LoRA checkpoint opened with its `adapter_config.json` as its
    /// config, as an adapter of the decoder `spec` describes.
    ///
    /// Raises:
    ///     ValueError: A factor is missing, malformed or of another shape, or the checkpoint
    ///         holds a tensor no adapted module reads.
    ///     NotImplementedError: The configuration asks for an update a term does not hold, such as
    ///         DoRA, a bias or a rank past `AdapterSpec.RANK`.
    ///     RuntimeError: The device fails.
    #[new]
    fn new(py: Python<'_>, spec: &AdapterSpec, checkpoint: &Checkpoint) -> Result<Self, Error> {
        let inner = py.detach(|| {
            let mut context = spec.context.lock().unwrap_or_else(PoisonError::into_inner);
            (spec.load)(&checkpoint.0, &mut context.device)
        })?;
        Ok(Self {
            inner,
            context: spec.context.clone(),
        })
    }
}

impl Adapter {
    /// This adapter as a call on the device of `context` reads it; refuses one on another device.
    fn on(&self, context: &Shared) -> Result<&decoder::Adapter, Error> {
        if !Arc::ptr_eq(&self.context, context) {
            return Err(PyValueError::new_err("an adapter lies on another device").into());
        }
        Ok(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyTypeError;
    use pyo3::types::PyDict;

    /// Kept rows read as a `Keep` or as a range of consecutive rows; a range with another step or
    /// a negative bound is refused with `ValueError`, and another value with `TypeError`.
    #[test]
    fn reads_kept_rows_as_a_keep_or_a_range() {
        Python::initialize();
        Python::attach(|py| {
            let scope = PyDict::new(py);
            scope.set_item("Keep", py.get_type::<Keep>()).unwrap();
            let keep = |code: &std::ffi::CStr| {
                (py.eval(code, Some(&scope), None).unwrap())
                    .extract::<Kept>()
                    .map(|Kept(keep)| keep)
            };
            assert_eq!(keep(c"Keep.LAST").unwrap(), decoder::Keep::Last);
            assert_eq!(keep(c"range(1, 3)").unwrap(), decoder::Keep::Range(1, 3));
            for refused in [c"range(0, 4, 2)", c"range(-1, 2)", c"range(0, -1)"] {
                let error = keep(refused).unwrap_err();
                assert!(error.is_instance_of::<PyValueError>(py), "{refused:?}");
                let reason = error.value(py).to_string();
                assert_eq!(reason, "kept rows are a range of consecutive rows");
            }
            let error = keep(c"5").unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py), "{error}");
            let reason = error.value(py).to_string();
            assert_eq!(reason, "'int' object cannot be cast as 'Keep'");
        });
    }

    /// A request's tokens are one-dimensional I32 IDs; another rank or dtype is refused with
    /// `ValueError` and why.
    #[test]
    fn takes_one_dimensional_i32_tokens() {
        crate::tensor::tests::run_on_device(
            c"
buffer = engine.HbmBuffer(device, 16)
engine.decoder.Request([[0]], 0, engine.HbmTensor(buffer, [2], 'int32'))
for shape, dtype in (([1, 2], 'int32'), ([2], 'int64')):
    try:
        engine.decoder.Request([[0]], 0, engine.HbmTensor(buffer, shape, dtype))
    except ValueError as error:
        assert str(error) == 'tokens are one-dimensional I32 IDs', (shape, dtype, error)
    else:
        raise AssertionError(f'{shape} {dtype} tokens must be refused')
",
        );
    }
}
