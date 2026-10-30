//! Serve a decoder: each request appends tokens to one sequence in the cache, and a call of several
//! requests returns the rows they keep once the device accepted its launches.
//!
//! - Row `i` of a request is position `p = cached + i`; it writes slot `rows[0][p / B] * B + p % B`
//!   of the paged cache group and attends to `p + 1` positions, where `B` is the group's block of
//!   tokens.
//! - `Batch::new` validates a call's requests and plans its operands, and nothing else validates
//!   them; a request the decoder cannot serve is refused there as unsupported.
//! - A call borrows the views its decoder keeps of the `Cache`; it reads no token ID on the host.
//! - A framework serves every decoder as a `Served`, one object whatever its model: a call returns
//!   the kept rows as an `HbmTensor` of their dtype.
//! - A decoder serves requests as an `Instance`, with the tiling its load chose for its rows'
//!   attention: under `Tiling::Fixed` a row's values do not depend on the other rows of its calls.
//! - A call may run its requests with LoRA adapters: the caller's list of them, which a request
//!   names an adapter of by its index (`adapter`); the batch keeps the list it checked.

mod adapters;
mod batch;
mod cache;
mod qwen3;

pub use adapters::{Adapter, Binding, Loader, SLOTS};
pub use batch::{Batch, Rows, Staged, Table};
pub use cache::{Cache, CacheSpec, Group, Kind, Pool};

use crate::{AsyncModule, Checkpoint, LoadError};
use furiosa_opt_models::ops::attention::full::paged::Tiling;
use furiosa_opt_models::tensor::memory::{Dtype, dtype};
use furiosa_opt_std::prelude::{Broadcast, Device, HbmBuffer, M, Scalar, m};
use std::io;
use std::pin::Pin;

/// One sequence chunk: `tokens` appended after the `cached` positions already in the cache.
pub struct Request {
    /// The sequence's block IDs, a list for each cache group; block `j` of a paged group's list
    /// holds positions `j * B..(j + 1) * B`.
    pub rows: Vec<Vec<u32>>,
    /// Positions already in the cache.
    pub cached: usize,
    /// I32 token IDs, one a row, each below the vocabulary; nothing checks the values.
    pub tokens: Rows<i32, m![1]>,
    /// The appended rows the call returns.
    pub keep: Keep,
    /// The index, in the call's list of adapters, of the adapter the appended rows run with.
    pub adapter: Option<usize>,
    /// I32 positions `[R, n]` of the `n` appended rows, R-major, where `R` is 1 or 3 for rotary
    /// positions in three sections; `None` is position `cached + i` for row `i`.
    pub positions: Option<HbmBuffer>,
    /// `[n, n]` bytes, row-major: whether appended row `i` sees appended row `j`; every row sees
    /// the cached positions. `None` is causal.
    pub mask: Option<HbmBuffer>,
    /// BF16 rows of the body's width that replace the embedding of the appended rows from each
    /// offset on.
    pub overrides: Vec<(usize, HbmBuffer)>,
}

impl Clone for Request {
    fn clone(&self) -> Self {
        let alias = |buffer: &HbmBuffer| {
            (buffer.alias(..)).unwrap_or_else(|_| unreachable!("a buffer aliases itself"))
        };
        Self {
            rows: self.rows.clone(),
            cached: self.cached,
            tokens: self.tokens.clone(),
            keep: self.keep,
            adapter: self.adapter,
            positions: self.positions.as_ref().map(alias),
            mask: self.mask.as_ref().map(alias),
            overrides: (self.overrides.iter())
                .map(|(offset, rows)| (*offset, alias(rows)))
                .collect(),
        }
    }
}

/// The appended rows a call returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keep {
    /// None of them.
    None,
    /// The last one.
    Last,
    /// All of them, in position order.
    All,
    /// Rows `a..b` of them, in position order.
    Range(usize, usize),
}

/// What one call of a decoder may hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Positions a request reaches at most: the context the decoder attends.
    pub window: usize,
    /// Rows of a call, across its requests.
    pub rows: usize,
    /// The cache the calls read.
    pub cache: CacheSpec,
    /// Adapters a call runs at most; none for a decoder whose projections take no adapter terms.
    pub adapters: usize,
}

/// What a call returns of each kept row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// Its greedy token ID.
    Tokens,
    /// Its hidden values.
    Hidden,
    /// Its scores.
    Scores,
}

/// The rows a call returns, their element type erased: `shape` is `[rows]` for one-element rows,
/// else `[rows, width]`, of `dtype` elements.
pub struct HbmTensor {
    pub buffer: HbmBuffer,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

/// Rows of `T` elements as a tensor of `T`'s dtype; refuses a scalar no dtype stores.
impl<T: Scalar, Row: M> TryFrom<Rows<T, Row>> for HbmTensor {
    type Error = io::Error;

    fn try_from(rows: Rows<T, Row>) -> io::Result<Self> {
        let shape = match Row::SIZE {
            1 => vec![rows.len()],
            width => vec![rows.len(), width],
        };
        Ok(Self {
            buffer: rows.into_buffer(),
            shape,
            dtype: dtype::<T>()?,
        })
    }
}

/// A decoder root a framework serves: what its calls hold and return, the views it keeps of its
/// cache, the adapters its calls run with, and the launches of one call; an `Instance` of it
/// serves requests on a cache, with adapters or without, as an `AsyncModule`.
pub trait Decoder: Send + Sync {
    /// What a call returns of each kept row.
    const OUTPUT: Output;
    /// Adapters a call runs at most, which its limits hold: none for a decoder whose projections
    /// take no adapter terms.
    const ADAPTERS: usize = 0;
    /// What a call returns: the kept rows' token IDs, hidden values or scores.
    type Output: TryInto<HbmTensor, Error = io::Error>;
    /// The views a call borrows of a cache's layers.
    type Layers: Send + 'static;
    /// An adapter a call may run its requests with: a LoRA checkpoint loaded as the decoder's
    /// adapter tree, or `Infallible` for a decoder that runs none.
    type Adapter: Send + Sync + 'static;

    /// What one call holds when its rows' attention is tiled as `tiling` says.
    fn limits(&self, tiling: Tiling) -> Limits;

    /// Views of `storage`, the layers of a cache allocated for this decoder.
    fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers>;

    /// `checkpoint`, a PEFT LoRA checkpoint, loaded onto `device` as an adapter of this decoder;
    /// refused for a decoder that runs none.
    fn adapter(_: &Checkpoint, _: &mut Device) -> Result<Self::Adapter, LoadError> {
        Err(LoadError::Unsupported {
            name: "adapters".into(),
            reason: "the decoder's projections take no adapter".into(),
        })
    }

    /// Launches the call `batch` plans on `layers`, with the adapters it binds, its rows'
    /// attention tiled as `tiling` says, returning once the device accepted them.
    fn call(
        &self,
        device: &mut Device,
        layers: &mut Self::Layers,
        batch: Batch<'_, Self::Adapter>,
        tiling: Tiling,
    ) -> impl Future<Output = io::Result<Self::Output>> + Send;
}

/// A decoder and the tiling its load chose for its calls: the instance serves requests as that
/// tiling says, so no caller passes one.
pub struct Instance<D> {
    decoder: D,
    tiling: Tiling,
}

impl<D> Instance<D> {
    pub fn new(decoder: D, tiling: Tiling) -> Self {
        Self { decoder, tiling }
    }

    pub fn decoder(&self) -> &D {
        &self.decoder
    }
}

/// Appends each request's tokens to the cache and returns the rows it keeps, by request and then
/// position, once the device accepted the launches tiled as the instance says; an invalid request
/// fails before any launch.
impl<'r, 'c, D: Decoder> AsyncModule<&'r [Request], (), &'c Cache> for Instance<D> {
    type Output = D::Output;

    async fn forward(
        &self,
        device: &mut Device,
        (): (),
        cache: &'c Cache,
        requests: &'r [Request],
    ) -> io::Result<D::Output> {
        let adapters: &[Option<&D::Adapter>] = &[];
        AsyncModule::forward(self, device, (), (cache, adapters), requests).await
    }
}

/// As the call of the requests alone, which runs them with the adapters of `adapters`, the
/// caller's list of them: a request names its adapter by its index there.
impl<'r, 'c, 'l, 'a, D: Decoder>
    AsyncModule<&'r [Request], (), (&'c Cache, &'l [Option<&'a D::Adapter>])> for Instance<D>
{
    type Output = D::Output;

    async fn forward(
        &self,
        device: &mut Device,
        (): (),
        (cache, adapters): (&'c Cache, &'l [Option<&'a D::Adapter>]),
        requests: &'r [Request],
    ) -> io::Result<D::Output> {
        let limits = Decoder::limits(&self.decoder, self.tiling);
        let batch = Batch::new(requests, &limits, cache, adapters)?;
        let mut layers = cache.lend(D::layers)?;
        self.decoder
            .call(device, &mut layers, batch, self.tiling)
            .await
    }
}

/// A decoder as a framework serves it, one object whatever its model: what its calls hold and
/// return, its adapters, and a call that returns the kept rows as an `HbmTensor`.
pub trait Served: Send + Sync {
    fn limits(&self) -> Limits;

    /// What a call returns of each kept row.
    fn output(&self) -> Output;

    /// How the decoder loads the adapters its calls run with; none for a decoder that runs none.
    fn adapters(&self) -> Option<Loader>;

    /// Appends each request's tokens to `cache` and returns the rows it keeps, by request and then
    /// position, once the device accepted the launches; a request with an `adapter` runs with that
    /// slot of `adapters`. An invalid request, or an adapter of another model, fails before any
    /// launch.
    fn forward<'a>(
        &'a self,
        device: &'a mut Device,
        cache: &'a Cache,
        requests: &'a [Request],
        adapters: &'a [Option<&'a Adapter>],
    ) -> Pin<Box<dyn Future<Output = io::Result<HbmTensor>> + Send + 'a>>;
}

/// An instance serves as itself, its call's rows erased to their dtype and its adapters to
/// `Adapter`s.
impl<D: Decoder> Served for Instance<D> {
    fn limits(&self) -> Limits {
        Decoder::limits(&self.decoder, self.tiling)
    }

    fn output(&self) -> Output {
        D::OUTPUT
    }

    fn adapters(&self) -> Option<Loader> {
        let load: Loader = |checkpoint, device| Ok(Adapter::new(D::adapter(checkpoint, device)?));
        (D::ADAPTERS > 0).then_some(load)
    }

    fn forward<'a>(
        &'a self,
        device: &'a mut Device,
        cache: &'a Cache,
        requests: &'a [Request],
        adapters: &'a [Option<&'a Adapter>],
    ) -> Pin<Box<dyn Future<Output = io::Result<HbmTensor>> + Send + 'a>> {
        Box::pin(async move {
            let adapters = (adapters.iter())
                .map(|adapter| adapter.map(Adapter::of::<D::Adapter>).transpose())
                .collect::<io::Result<Vec<_>>>()?;
            let call = (cache, &adapters[..]);
            let rows = AsyncModule::forward(self, device, (), call, requests).await?;
            rows.try_into()
        })
    }
}
