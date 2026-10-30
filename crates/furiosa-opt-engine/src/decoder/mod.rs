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
//! - A `Root` of the decoder family's catalog is a decoder of one launch a call: the host plans
//!   the rows' positions, cache slots, attention work and selected rows, and the launch writes the
//!   selected rows' outputs in kept order.

mod adapters;
mod batch;
mod cache;
mod qwen3;

pub use adapters::{Adapter, Binding, Loader, SLOTS};
pub use batch::{Batch, Rows, Staged, Table};
pub use cache::{Cache, CacheSpec, Group, Kind, Pool};

use crate::{AsyncModule, Checkpoint, Load, LoadError, forward};
use furiosa_opt_models::Module;
use furiosa_opt_models::models::decoder::attention::qwen3::{LmHead, fp8, launch};
use furiosa_opt_models::models::decoder::attention::{
    ForCausalLM, ForSequenceClassification, Geometry, Head, KvHeads,
};
use furiosa_opt_models::models::decoder::form;
use furiosa_opt_models::modules::attention::key_value;
use furiosa_opt_models::modules::language::{Classifier, Input, Selected};
use furiosa_opt_models::modules::lora::{self, Rank, States};
use furiosa_opt_models::ops::attention::full::paged::{Tiling, chunks::Masks};
use furiosa_opt_models::ops::index::table::BlockIndex;
use furiosa_opt_models::ops::linear::affine::Columns;
use furiosa_opt_models::specialize::attention::Decode;
use furiosa_opt_models::tensor::cache::{Block, Blocks};
use furiosa_opt_models::tensor::memory::{Dtype, Hbm, alias, dtype};
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::DeviceOutput;
use std::convert::Infallible;
use std::io;
use std::marker::PhantomData;
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

/// A root of the decoder family's catalog, which a call launches once: of the size `Size`, writing
/// each selected row's output, `Row` elements, into `Place` in order, its projections running
/// with the adapters `Adapting` gives.
pub trait Root {
    type Size: Geometry + Decode + 'static;
    type Place: Place;
    type Row: M;
    type Adapting: Adapting<Self::Size>;
}

/// The causal model writes the selected rows' tokens; its projections' form runs its adapters.
impl<S: Geometry + Decode + 'static, B, P: Adapting<S>> Root
    for ForCausalLM<S, launch::Over<S, B, P>, LmHead<S, Hbm>, P>
{
    type Size = S;
    type Place = launch::Tokens<S>;
    type Row = m![1];
    type Adapting = P;
}

/// The base model writes the selected rows' hidden values; its projections' form runs its
/// adapters.
impl<S: Geometry + Decode + 'static, B, P: Adapting<S>> Root for launch::Over<S, B, P> {
    type Size = S;
    type Place = launch::Values<S>;
    type Row = m![{ Symbol<<S as Geometry>::Hidden> }];
    type Adapting = P;
}

/// The sequence classifier writes the selected rows' scores, label columns first; it runs no
/// adapter.
impl<S: Geometry + Decode + 'static, B, P> Root
    for ForSequenceClassification<S, launch::Over<S, B, P>, Classifier<<S as Geometry>::Hidden>, P>
{
    type Size = S;
    type Place = launch::Scores<S>;
    type Row = m![{ Symbol<Columns> }];
    type Adapting = Infallible;
}

/// Block-scaled FP8 Qwen3-8B writes the selected rows' tokens; it runs no adapter.
impl Root for fp8::ForCausalLM {
    type Size = Size8B;
    type Place = launch::Tokens<Size8B>;
    type Row = m![1];
    type Adapting = Infallible;
}

type Size8B = furiosa_opt_models::models::decoder::attention::qwen3::Size<"8B">;

/// The adapters a call of a root of the size `S` may run its rows with: LoRA terms of every dense
/// BF16 projection, or none.
pub trait Adapting<S: Geometry + Decode + 'static> {
    /// An adapter: a PEFT checkpoint loaded as the root's adapter tree, or `Infallible`.
    type Adapter: Send + Sync + 'static;
    /// Each layer's projections' states in a call that runs [`SLOTS`] adapters.
    type States<'t>: States;
    /// Adapters a call runs at most.
    const ADAPTERS: usize;

    /// `checkpoint`, a PEFT LoRA checkpoint, loaded onto `device` as an adapter.
    fn adapter(checkpoint: &Checkpoint, device: &mut Device) -> Result<Self::Adapter, LoadError>;

    /// Layer `layer`'s states with the call's adapters `held` and their `selections`.
    fn states<'t>(
        held: [&'t Self::Adapter; SLOTS],
        layer: usize,
        selections: &'t launch::Selections<S, SLOTS>,
    ) -> Self::States<'t>;
}

/// Dense BF16 projections add each adapter's update of the rows it selects.
impl<S: Geometry + Decode + 'static> Adapting<S> for form::Affine
where
    launch::Adapter<S, Hbm>:
        Send + Sync + 'static + for<'h> Load<&'h launch::Adapter<S, lora::Staged>>,
    launch::Adapter<S, lora::Staged>: for<'c> Load<&'c Checkpoint>,
    <S as Geometry>::Layers<launch::Terms<S, Hbm>>: AsRef<[launch::Terms<S, Hbm>]>,
{
    type Adapter = launch::Adapter<S, Hbm>;
    type States<'t> = launch::Adapted<'t, S, SLOTS>;
    const ADAPTERS: usize = SLOTS;

    // A PEFT checkpoint's terms are built on host from the checkpoint read once, then uploaded.
    fn adapter(checkpoint: &Checkpoint, device: &mut Device) -> Result<Self::Adapter, LoadError> {
        lora::load::<launch::Adapter<S, Hbm>, launch::Adapter<S, lora::Staged>>(checkpoint, device)
    }

    fn states<'t>(
        held: [&'t Self::Adapter; SLOTS],
        layer: usize,
        selections: &'t launch::Selections<S, SLOTS>,
    ) -> Self::States<'t> {
        launch::adapted(held, layer, selections)
    }
}

/// A root that runs no adapter refuses a load of one; no call holds one.
impl<S: Geometry + Decode + 'static> Adapting<S> for Infallible {
    type Adapter = Infallible;
    type States<'t> = ();
    const ADAPTERS: usize = 0;

    fn adapter(_: &Checkpoint, _: &mut Device) -> Result<Infallible, LoadError> {
        Err(LoadError::Unsupported {
            name: "adapters".into(),
            reason: "the decoder's projections take no adapter".into(),
        })
    }

    fn states<'t>(
        held: [&'t Infallible; SLOTS],
        _: usize,
        _: &'t launch::Selections<S, SLOTS>,
    ) -> Self::States<'t> {
        match *held[0] {}
    }
}

/// What a launch writes of each selected row, named by its element type: a token in I32, hidden
/// values in BF16, scores in FP32.
pub trait Place: DeviceOutput {
    /// The element type of the rows a call returns.
    type Scalar: Scalar;
    /// What a call returns of each kept row.
    const OUTPUT: Output;

    fn into_buffer(self) -> HbmBuffer;
}

impl<E: M> Place for furiosa_opt_std::prelude::HbmTensor<i32, Chip, E> {
    type Scalar = i32;
    const OUTPUT: Output = Output::Tokens;

    fn into_buffer(self) -> HbmBuffer {
        furiosa_opt_std::prelude::HbmTensor::into_buffer(self)
    }
}

impl<E: M> Place for furiosa_opt_std::prelude::HbmTensor<bf16, Chip, E> {
    type Scalar = bf16;
    const OUTPUT: Output = Output::Hidden;

    fn into_buffer(self) -> HbmBuffer {
        furiosa_opt_std::prelude::HbmTensor::into_buffer(self)
    }
}

impl<E: M> Place for furiosa_opt_std::prelude::HbmTensor<f32, Chip, E> {
    type Scalar = f32;
    const OUTPUT: Output = Output::Scores;

    fn into_buffer(self) -> HbmBuffer {
        furiosa_opt_std::prelude::HbmTensor::into_buffer(self)
    }
}

/// Block IDs of a call of `R` rows whose every row is a request spanning the whole context of `W`
/// windows.
#[derive(Clone, Debug)]
pub(super) struct Tables<R, W>(PhantomData<fn() -> (R, W)>);

impl<R: AxisName, W: AxisName> AxisName for Tables<R, W> {
    const NAME: Ident = Ident::new("Tables");
    const SIZE: usize = R::SIZE * W::SIZE * BlockIndex::SIZE;
}

/// A catalog root as a decoder: a call plans its rows' positions, cache slots, attention work and
/// selected rows on the host, then launches the root once over every layer's own blocks, which
/// writes the selected rows' outputs in kept order; a call that runs adapters passes each layer's
/// projections their terms and selections of the rows. Under a fixed tiling every row is a work
/// item, so a launch holds the size's call rows.
impl<R> Decoder for R
where
    R: Root + DeviceSend + Send + Sync,
    for<'b, 't> <R::Size as Geometry>::Layers<launch::Layer<'b, Held<'t, R>>>:
        TryFrom<Vec<launch::Layer<'b, Held<'t, R>>>> + DeviceSend,
    for<'b> <R::Size as Geometry>::Layers<launch::Layer<'b>>:
        TryFrom<Vec<launch::Layer<'b>>> + DeviceSend,
    for<'i, 'a, 's, 'b> R: Module<
            &'i mut launch::Input<'a, R::Size>,
            (),
            launch::State<'s, R::Size, <R::Size as Geometry>::Layers<launch::Layer<'b>>, R::Place>,
            Output = (),
        >,
    for<'i, 'a, 's, 'b, 't> R: Module<
            &'i mut launch::Input<'a, R::Size>,
            (),
            launch::State<
                's,
                R::Size,
                <R::Size as Geometry>::Layers<launch::Layer<'b, Held<'t, R>>>,
                R::Place,
            >,
            Output = (),
        >,
{
    const OUTPUT: Output = <R::Place as Place>::OUTPUT;
    const ADAPTERS: usize = <R::Adapting as Adapting<R::Size>>::ADAPTERS;
    type Output = Rows<<R::Place as Place>::Scalar, R::Row>;
    type Layers = Vec<Blocks<KvHeads, Head>>;
    type Adapter = <R::Adapting as Adapting<R::Size>>::Adapter;

    fn limits(&self, tiling: Tiling) -> Limits {
        limits::<R::Size>(Self::ADAPTERS, tiling)
    }

    fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
        (storage.iter())
            .map(|layer| Blocks::from_buffer(layer.alias(..).map_err(io::Error::other)?))
            .collect()
    }

    fn adapter(checkpoint: &Checkpoint, device: &mut Device) -> Result<Self::Adapter, LoadError> {
        <R::Adapting as Adapting<R::Size>>::adapter(checkpoint, device)
    }

    // The root is captured and launched in place, never held across an await: that `Send` check
    // erases lifetimes and cannot prove the root's higher-ranked bounds.
    fn call(
        &self,
        device: &mut Device,
        layers: &mut Self::Layers,
        batch: Batch<'_, Self::Adapter>,
        tiling: Tiling,
    ) -> impl Future<Output = io::Result<Self::Output>> + Send {
        let root = self;
        async move {
            let mut reads = batch.operands::<
                launch::Capacity<R::Size>,
                <R::Size as Decode>::Reach,
                <R::Size as Decode>::Sequences,
                KvHeads,
                <R::Size as Geometry>::Groups,
                Head,
                launch::Waves<R::Size>,
                <R::Size as Decode>::Windows,
            >(0, tiling)?;
            if let Some(binding) = batch.adapters() {
                reads.extend(binding.selections::<launch::Capacity<R::Size>, Rank<1>>());
                reads.extend(binding.selections::<launch::Capacity<R::Size>, Rank<2>>());
            }
            // ponytail: placed each call; a load-lifetime place would spare two small uploads a
            // call.
            let masks = Masks::<<R::Size as Decode>::Windows>::new(device).await?;
            // Staged last, so nothing of the size's axes is held across an await.
            let (staged, views) = batch
                .stage::<
                    launch::Capacity<R::Size>,
                    Tables<launch::Capacity<R::Size>, <R::Size as Decode>::Windows>,
                    Chip,
                >(device, &reads, (0, Table::Blocks))
                .await?;
            let mut views = views.into_iter();
            let mut next =
                || (views.next()).ok_or_else(|| io::Error::other("a read array is missing"));
            let operands: launch::Operands<R::Size> = key_value::Operands {
                batch: key_value::Batch {
                    positions: alias(&next()?, ..)?,
                    slot: alias(&next()?, ..)?,
                },
                work: launch::Work::<R::Size>::bind(&mut next, &masks)?,
                selected: Selected {
                    live: alias(&next()?, ..)?,
                    rows: alias(&next()?, ..)?,
                },
                slots: PhantomData,
            };
            let tokens: launch::Ids<R::Size> = alias(&staged.tokens, ..)?;
            let mut input: launch::Input<'_, R::Size> = Input::new(tokens, staged.count, &operands);
            let place = DeviceOutput::alloc(device).map_err(io::Error::other)?;
            let place = match batch.adapters() {
                None => {
                    let slots = (layers.iter_mut())
                        .map(|blocks| ((Default::default(), blocks, ()), ()))
                        .collect();
                    submit(device, root, &mut input, slots, place)?
                }
                Some(binding) => {
                    let selections = (
                        adapters::selected::<launch::Capacity<R::Size>, Rank<1>>(&mut next)?,
                        adapters::selected::<launch::Capacity<R::Size>, Rank<2>>(&mut next)?,
                    );
                    let held = binding.held();
                    let slots = (layers.iter_mut().enumerate())
                        .map(|(at, blocks)| {
                            let states =
                                <R::Adapting as Adapting<R::Size>>::states(held, at, &selections);
                            ((Default::default(), blocks, ()), states)
                        })
                        .collect();
                    submit(device, root, &mut input, slots, place)?
                }
            };
            Rows::new(&place.into_buffer(), 0..batch.kept().len())
        }
    }
}

/// What each layer's projections of the root `R` run with in a call of adapters.
type Held<'t, R> = <<R as Root>::Adapting as Adapting<<R as Root>::Size>>::States<'t>;

/// What a launch of the size `S` holds, its rows tiled as `tiling` says: the context its size is
/// compiled for, one paged group of every layer's blocks, `adapters` adapters, and the launch's
/// rows; a fixed tiling attends every row as a work item, so it holds the size's call rows, which
/// its items hold over the whole context.
fn limits<S: Geometry + Decode + 'static>(adapters: usize, tiling: Tiling) -> Limits {
    let rows = match tiling {
        Tiling::Fixed => S::Rows::SIZE,
        Tiling::Adaptive => launch::Capacity::<S>::SIZE,
    };
    Limits {
        window: launch::Positions::<S>::SIZE,
        rows,
        cache: CacheSpec {
            pools: vec![Pool {
                block: Block::SIZE,
                bytes: Blocks::<KvHeads, Head>::BLOCK_STRIDE as usize,
            }],
            groups: vec![Group {
                pool: 0,
                layers: 0..S::LAYERS,
                kind: Kind::Paged,
                window: None,
            }],
        },
        adapters,
    }
}

/// One launch of `root` over a call's `input` and its layers' `slots`, writing the selected rows'
/// outputs into `place`, which it returns once the device accepted the launch.
fn submit<'i, 'a, R, L>(
    device: &mut Device,
    root: &R,
    input: &'i mut launch::Input<'a, R::Size>,
    slots: Vec<L>,
    place: R::Place,
) -> io::Result<R::Place>
where
    R: Root + DeviceSend,
    <R::Size as Geometry>::Layers<L>: TryFrom<Vec<L>> + DeviceSend,
    for<'s> R: Module<
            &'i mut launch::Input<'a, R::Size>,
            (),
            launch::State<'s, R::Size, <R::Size as Geometry>::Layers<L>, R::Place>,
            Output = (),
        >,
{
    let slots: <R::Size as Geometry>::Layers<L> = (slots.try_into())
        .unwrap_or_else(|_| unreachable!("a cache of the decoder's spec holds each layer"));
    let hidden = DeviceOutput::alloc(device).map_err(io::Error::other)?;
    let mut places = (((hidden, slots), ()), place);
    let state: launch::State<'_, R::Size, _, R::Place> = &mut places;
    launch(forward, (&mut *device, root, (), state, input))
        .output(&mut ())
        .submit()
        .accepted()
        .map(drop)
        .map_err(io::Error::other)?;
    Ok(places.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_models::models::decoder::attention::qwen3;
    use furiosa_opt_std::prelude::Topology;

    /// A launch of Qwen3-4B admits and plans a call past the 40960 positions a Qwen3 checkpoint
    /// declares, up to the context its size is compiled for: decode rows and a prompt that end at
    /// its last position, under a fixed tiling as many as the launch admits.
    #[test]
    fn launches_plan_the_compiled_context() -> io::Result<()> {
        type S = qwen3::Size<"4B">;
        let device = Device::new(Topology { chips: 1, pes: 8 }).map_err(io::Error::other)?;
        // A request of `rows` rows ending at position `end`: its history reads block 0, and its
        // rows write their own blocks from `first`.
        let request = |rows: usize, end: usize, first: usize| -> io::Result<Request> {
            let cached = end - rows;
            let blocks = (0..end.div_ceil(Block::SIZE))
                .map(|at| (at.checked_sub(cached / Block::SIZE)).map_or(0, |own| first + own))
                .map(|block| block as u32)
                .collect();
            let ids = HbmBuffer::alloc(&device, rows * Rows::<i32, m![1]>::bytes())
                .map_err(io::Error::other)?;
            Ok(Request {
                rows: vec![blocks],
                cached,
                tokens: Rows::new(&ids, 0..rows)?,
                keep: Keep::Last,
                adapter: None,
                positions: None,
                mask: None,
                overrides: Vec::new(),
            })
        };
        for tiling in [Tiling::Adaptive, Tiling::Fixed] {
            let limits = limits::<S>(0, tiling);
            let context = limits.window;
            assert!(context > 40960, "{tiling:?}: {context} positions");
            // The rows every row of which the items hold over the whole context.
            let rows = limits.rows.min(<S as Decode>::Rows::SIZE);
            let cache = Cache::new(&device, &limits.cache, 2 + rows)?;
            let mut requests = vec![request(rows / 2, context, 0)?];
            for row in 0..rows / 2 {
                requests.push(request(1, context, 2 + row)?);
            }
            let batch = Batch::<Infallible>::new(&requests, &limits, &cache, &[])?;
            batch.operands::<
                launch::Capacity<S>,
                <S as Decode>::Reach,
                <S as Decode>::Sequences,
                KvHeads,
                <S as Geometry>::Groups,
                Head,
                launch::Waves<S>,
                <S as Decode>::Windows,
            >(0, tiling)?;
        }
        Ok(())
    }
}
