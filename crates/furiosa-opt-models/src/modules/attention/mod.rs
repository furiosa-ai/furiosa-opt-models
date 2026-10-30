//! Attention: a `Qkv` child projects one call's queries, keys and values, a `Cache` child stores
//! them and returns the history, an `Attend` child attends over it and an output child projects it.

pub use appended::Appended;

pub use cache::{Cached, KeyValue};

pub use cross::{Cross, PositionedCross, PositionedKeyValue, PositionedProjection};

pub use full::Full;

pub use joint::Joint;

pub use latent::{Latent, LatentProjection, Sparse};

pub use positioned::Positioned;

pub use selected::Selected;

pub use shared::Shared;

pub use stored::{Source, Stored};

use crate::load::Load;

use crate::Module;

use crate::modules::generic::schedule::{Begin, IntoStream, Slot, Stream};

use crate::tensor::live::Live;

use core::task::Poll;

use furiosa_opt_std::prelude::{AxisName, Device, DeviceSend};

mod appended;

mod cache;

mod cross;

mod full;

mod joint;

pub mod key_value;

mod latent;

mod positioned;

pub mod relative;

mod selected;

mod shared;

mod stored;

/// Attends queries over the history a cache stage returns; a trait beside `Module`, as each kind
/// of history (`History`, `Paged`, `Selection`) flattens into a different attend op input.
pub(crate) trait Attend<Query, History> {
    type Output;

    fn attend(&self, device: &mut Device, input: (&Query, History)) -> Self::Output;
}

/// Keys and values read from the cache, with what attention reads besides them.
pub struct History<K, V, C>(pub K, pub V, pub C);

/// A layer's cache with the read operand that selects each row's history inside it.
pub(crate) struct Paged<C, R>(C, R);

impl<Q, K, V, C, O, M> Attend<Q, History<K, V, C>> for M
where
    M: for<'a, 's> Module<(&'a Q, &'a K, &'a V, C), (), &'s mut (), Output = O>,
{
    type Output = O;

    fn attend(&self, device: &mut Device, (query, history): (&Q, History<K, V, C>)) -> O {
        self.forward(
            device,
            (),
            &mut (),
            (query, &history.0, &history.1, history.2),
        )
    }
}

impl<Q, C, R, O, M> Attend<Q, Paged<C, R>> for M
where
    M: for<'a, 't> Module<(&'a Q, C, R), (), &'t mut (), Output = O>,
{
    type Output = O;

    fn attend(&self, device: &mut Device, (query, history): (&Q, Paged<C, R>)) -> O {
        self.forward(device, (), &mut (), (query, history.0, history.1))
    }
}

/// A call's operands for positioning queries/keys, writing new K/V and reading history.
#[derive(Clone, DeviceSend)]
pub struct Context<Positions, Write, Read> {
    /// Position operand applied to this call's queries and keys.
    pub positions: Positions,
    /// Cache destinations for this call's keys and values.
    pub write: Write,
    /// History selection consumed by the reader or paged attention.
    pub read: Read,
}

/// Projects the same input independently into query, key and value rows; each reads under the
/// name its family gives it.
#[derive(DeviceSend, Load)]
pub struct Projection<Query, Key = Query, Value = Key> {
    #[load(rename = "")]
    pub query: Query,
    #[load(rename = "")]
    pub key: Key,
    #[load(rename = "")]
    pub value: Value,
}

/// The state holds the query projection's slot, then the key's and the value's.
impl<'s, X, Q, K, V, QueryState, KeyState, ValueState, Query, Key, Value, Env: Copy>
    Module<&X, Env, &'s mut (QueryState, KeyState, ValueState)> for Projection<Query, Key, Value>
where
    Query: for<'x> Module<&'x X, Env, &'s mut QueryState, Output = Q>,
    Key: for<'x> Module<&'x X, Env, &'s mut KeyState, Output = K>,
    Value: for<'x> Module<&'x X, Env, &'s mut ValueState, Output = V>,
{
    type Output = (Q, K, V);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (query, key, value): &'s mut (QueryState, KeyState, ValueState),
        input: &X,
    ) -> Self::Output {
        let query = self.query.forward(&mut *device, env, query, input);
        let key = self.key.forward(&mut *device, env, key, input);
        let value = self.value.forward(device, env, value, input);
        (query, key, value)
    }
}

/// Projects QKV, normalizes Q and K independently and leaves V unchanged.
#[derive(DeviceSend, Load)]
pub struct NormalizedQkv<Projection, QueryNorm, KeyNorm> {
    #[load(rename = "")]
    pub projection: Projection,
    #[load(rename = "")]
    pub query_norm: QueryNorm,
    #[load(rename = "")]
    pub key_norm: KeyNorm,
}

/// The state holds the projection's slot, then the query norm's and the key norm's.
impl<
    's,
    X,
    Q,
    K,
    V,
    NormalizedQuery,
    NormalizedKey,
    ProjectionState,
    QueryNormState,
    KeyNormState,
    Projection,
    QueryNorm,
    KeyNorm,
    Env: Copy,
> Module<&X, Env, &'s mut (ProjectionState, QueryNormState, KeyNormState)>
    for NormalizedQkv<Projection, QueryNorm, KeyNorm>
where
    Projection: for<'x> Module<&'x X, Env, &'s mut ProjectionState, Output = (Q, K, V)>,
    QueryNorm: for<'x> Module<&'x Q, Env, &'s mut QueryNormState, Output = NormalizedQuery>,
    KeyNorm: for<'x> Module<&'x K, Env, &'s mut KeyNormState, Output = NormalizedKey>,
{
    type Output = (NormalizedQuery, NormalizedKey, V);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (projection, query_norm, key_norm): &'s mut (ProjectionState, QueryNormState, KeyNormState),
        hidden: &X,
    ) -> Self::Output {
        let (query, key, value) = self
            .projection
            .forward(&mut *device, env, projection, hidden);
        let query = self
            .query_norm
            .forward(&mut *device, env, query_norm, &query);
        let key = self.key_norm.forward(device, env, key_norm, &key);
        (query, key, value)
    }
}

/// The node of a composite of the attention's front, over its upstream's tiles or its child's
/// stream: its module, what it holds across polls (its leaves' written forms, written at its first
/// poll, or the positions of the tile its child works on), the call's env and the state it lends.
pub(crate) struct Front<M, H, E, S, U> {
    module: M,
    held: Option<H>,
    env: E,
    state: S,
    upstream: U,
}

impl<M, H, E, S, U> Front<M, H, E, S, U> {
    const fn new(module: M, env: E, state: S, upstream: U) -> Self {
        Self {
            module,
            held: None,
            env,
            state,
            upstream,
        }
    }
}

impl<T, M, H, E, S, U: AsMut<Slot<T>>> AsMut<Slot<T>> for Front<M, H, E, S, U> {
    fn as_mut(&mut self) -> &mut Slot<T> {
        self.upstream.as_mut()
    }
}

/// The projections stream over their upstream's tiles: each writes its weights on the first poll
/// (`Begin(())`), and the value projection begins the input form all three read. Each upstream
/// tile is staged into that form, which gives the tiles the projections read, as many and of what
/// rows it takes; each is lent to all three.
impl<
    's,
    U: Stream,
    E,
    QueryState,
    KeyState,
    ValueState,
    Query,
    Key,
    Value,
    QueryForm,
    KeyForm,
    ValueForm,
    T,
> IntoStream<U, E, &'s mut (QueryState, KeyState, ValueState)> for Projection<Query, Key, Value>
where
    Query: for<'x> Module<Begin<()>, E, &'x mut QueryState, Output = QueryForm>,
    Key: for<'x> Module<Begin<()>, E, &'x mut KeyState, Output = KeyForm>,
    Value: for<'x> Module<Begin<()>, E, &'x mut ValueState, Output = ValueForm>,
    ValueForm: for<'x> Module<Begin<()>, E, &'x mut ValueState, Output = T>,
    Front<
        Self,
        ((QueryForm, KeyForm, ValueForm), T),
        E,
        &'s mut (QueryState, KeyState, ValueState),
        U,
    >: Stream,
{
    type Stream = Front<
        Self,
        ((QueryForm, KeyForm, ValueForm), T),
        E,
        &'s mut (QueryState, KeyState, ValueState),
        U,
    >;

    fn into_stream(
        self,
        env: E,
        state: &'s mut (QueryState, KeyState, ValueState),
        upstream: U,
    ) -> Self::Stream {
        Front::new(self, env, state, upstream)
    }
}

impl<
    's,
    X,
    U,
    E: Copy,
    QueryState,
    KeyState,
    ValueState,
    Query,
    Key,
    Value,
    QueryForm,
    KeyForm,
    ValueForm,
    T,
    I,
    R,
    Q,
    K,
    V,
> Stream
    for Front<
        Projection<Query, Key, Value>,
        ((QueryForm, KeyForm, ValueForm), T),
        E,
        &'s mut (QueryState, KeyState, ValueState),
        U,
    >
where
    U: Stream<Item = X>,
    R: AxisName,
    Query: for<'x> Module<Begin<()>, E, &'x mut QueryState, Output = QueryForm>,
    Key: for<'x> Module<Begin<()>, E, &'x mut KeyState, Output = KeyForm>,
    Value: for<'x> Module<Begin<()>, E, &'x mut ValueState, Output = ValueForm>,
    ValueForm: for<'x> Module<Begin<()>, E, &'x mut ValueState, Output = T>
        + for<'x, 't> Module<(&'t mut T, X), E, &'x mut ValueState, Output = ()>
        + for<'x, 't> Module<&'t mut T, E, &'x mut ValueState, Output = Poll<Option<Live<I, R>>>>
        + for<'x, 't> Module<&'t I, E, &'x mut ValueState, Output = V>,
    QueryForm: for<'x, 't> Module<&'t I, E, &'x mut QueryState, Output = Q>,
    KeyForm: for<'x, 't> Module<&'t I, E, &'x mut KeyState, Output = K>,
{
    type Item = Live<(Q, K, V), R>;

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Self::Item>> {
        let Self {
            module,
            held,
            env,
            state,
            upstream,
        } = self;
        let (query, key, value) = &mut **state;
        if matches!(held, None) {
            let written = module
                .value
                .forward(&mut *device, *env, &mut *value, Begin(()));
            let keyed = module.key.forward(&mut *device, *env, &mut *key, Begin(()));
            let queried = module
                .query
                .forward(&mut *device, *env, &mut *query, Begin(()));
            let input = written.forward(&mut *device, *env, &mut *value, Begin(()));
            *held = Some(((queried, keyed, written), input));
            // The writing poll reads no count, so the polls that follow, which do, keep its state.
            return Poll::Pending;
        }
        let Some(((queries, keys, values), input)) = held else {
            return Poll::Pending;
        };
        // A poll gives a staged tile, lets the form stage, or takes one upstream tile, so every
        // count a poll reads ends its run of polls at the same trip.
        let tile = match values.forward(&mut *device, *env, &mut *value, &mut *input) {
            Poll::Ready(Some(tile)) => tile,
            Poll::Pending => return Poll::Pending,
            Poll::Ready(None) => {
                return match upstream.poll_next(&mut *device) {
                    Poll::Ready(Some(tile)) => {
                        values.forward(&mut *device, *env, &mut *value, (&mut *input, tile));
                        Poll::Pending
                    }
                    Poll::Ready(None) => Poll::Ready(None),
                    Poll::Pending => Poll::Pending,
                };
            }
        };
        let rows = tile.value();
        Poll::Ready(Some(tile.with((
            queries.forward(&mut *device, *env, &mut *query, rows),
            keys.forward(&mut *device, *env, &mut *key, rows),
            values.forward(device, *env, &mut *value, rows),
        ))))
    }
}

/// The norms stream over their projection's stream: each writes its gain on the first poll, then
/// normalizes the queries or keys of every projected tile; values pass unchanged.
impl<
    's,
    U: Stream,
    E: Copy,
    ProjectionState,
    QueryNormState,
    KeyNormState,
    P,
    QueryNorm,
    KeyNorm,
    QueryForm,
    KeyForm,
    C,
> IntoStream<U, E, &'s mut (ProjectionState, QueryNormState, KeyNormState)>
    for NormalizedQkv<P, QueryNorm, KeyNorm>
where
    P: IntoStream<U, E, &'s mut ProjectionState, Stream = C>,
    QueryNorm: for<'x> Module<Begin<()>, E, &'x mut QueryNormState, Output = QueryForm>,
    KeyNorm: for<'x> Module<Begin<()>, E, &'x mut KeyNormState, Output = KeyForm>,
    Front<
        NormalizedQkv<(), QueryNorm, KeyNorm>,
        (QueryForm, KeyForm),
        E,
        (&'s mut QueryNormState, &'s mut KeyNormState),
        C,
    >: Stream,
{
    type Stream = Front<
        NormalizedQkv<(), QueryNorm, KeyNorm>,
        (QueryForm, KeyForm),
        E,
        (&'s mut QueryNormState, &'s mut KeyNormState),
        C,
    >;

    fn into_stream(
        self,
        env: E,
        (projection, query_norm, key_norm): &'s mut (ProjectionState, QueryNormState, KeyNormState),
        upstream: U,
    ) -> Self::Stream {
        let norms = NormalizedQkv {
            projection: (),
            query_norm: self.query_norm,
            key_norm: self.key_norm,
        };
        let projection = self.projection.into_stream(env, projection, upstream);
        Front::new(norms, env, (query_norm, key_norm), projection)
    }
}

impl<
    'q,
    'k,
    U,
    E: Copy,
    QueryNormState,
    KeyNormState,
    QueryNorm,
    KeyNorm,
    QueryForm,
    KeyForm,
    Q,
    K,
    V,
    NormalizedQuery,
    NormalizedKey,
    R,
> Stream
    for Front<
        NormalizedQkv<(), QueryNorm, KeyNorm>,
        (QueryForm, KeyForm),
        E,
        (&'q mut QueryNormState, &'k mut KeyNormState),
        U,
    >
where
    U: Stream<Item = Live<(Q, K, V), R>>,
    R: AxisName,
    QueryNorm: for<'x> Module<Begin<()>, E, &'x mut QueryNormState, Output = QueryForm>,
    KeyNorm: for<'x> Module<Begin<()>, E, &'x mut KeyNormState, Output = KeyForm>,
    QueryForm: for<'x> Module<Q, E, &'x mut QueryNormState, Output = NormalizedQuery>,
    KeyForm: for<'x> Module<K, E, &'x mut KeyNormState, Output = NormalizedKey>,
{
    type Item = Live<(NormalizedQuery, NormalizedKey, V), R>;

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Self::Item>> {
        let Self {
            module,
            held,
            env,
            state: (query, key),
            upstream,
        } = self;
        if matches!(held, None) {
            let queried = module
                .query_norm
                .forward(&mut *device, *env, &mut **query, Begin(()));
            let keyed = module
                .key_norm
                .forward(&mut *device, *env, &mut **key, Begin(()));
            *held = Some((queried, keyed));
        }
        match (upstream.poll_next(&mut *device), held) {
            (Poll::Ready(Some(tile)), Some((query_norm, key_norm))) => {
                let rows = tile.with(());
                let (queries, keys, values) = tile.into_value();
                Poll::Ready(Some(rows.with((
                    query_norm.forward(&mut *device, *env, &mut **query, queries),
                    key_norm.forward(device, *env, &mut **key, keys),
                    values,
                ))))
            }
            (Poll::Ready(None), _) => Poll::Ready(None),
            _ => Poll::Pending,
        }
    }
}

/// Projects Q/K/V, stores them in the cache, attends over the history and projects the result.
#[derive(DeviceSend, Load)]
pub struct Attention<Qkv, Cache, Attend, Project> {
    #[load(rename = "")]
    pub qkv: Qkv,
    #[load(rename = "")]
    pub cache: Cache,
    #[load(rename = "")]
    pub attend: Attend,
    #[load(rename = "")]
    pub output: Project,
}

/// A composition, not an op.
impl<X, Qkv, Cache, Attend, Project> !crate::Op<&X> for Attention<Qkv, Cache, Attend, Project> {}

/// Projects `input`, takes the queries and history from the cache stage given `context`, attends,
/// and projects under the routing `mask` or `()`; the state holds the three stages' slots in order.
impl<'s, I, C, M, O, Y, R, H, QkvState, CacheState, ProjectState, Qkv, Cache, A, Project, Env: Copy>
    Module<((I, C), M), Env, &'s mut (QkvState, CacheState, ProjectState)>
    for Attention<Qkv, Cache, A, Project>
where
    Qkv: Module<I, Env, &'s mut QkvState, Output = Y>,
    Cache: Module<(Y, C), Env, &'s mut CacheState, Output = (R, H)>,
    A: Attend<R, H>,
    M: Routing<Project, A::Output, &'s mut ProjectState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (qkv, cache, output): &'s mut (QkvState, CacheState, ProjectState),
        ((input, context), mask): ((I, C), M),
    ) -> O {
        let projected = self.qkv.forward(&mut *device, env, qkv, input);
        let (query, history) = self
            .cache
            .forward(&mut *device, env, cache, (projected, context));
        let hidden = self.attend.attend(&mut *device, (&query, history));
        mask.project(device, &self.output, output, &hidden)
    }
}

/// The output projection reads the attended values alone, or with a routing mask.
trait Routing<Project, Hidden, S> {
    type Output;

    fn project(
        self,
        device: &mut Device,
        output: &Project,
        state: S,
        hidden: &Hidden,
    ) -> Self::Output;
}

impl<P: for<'x> Module<&'x Y, (), S, Output = O>, Y, O, S> Routing<P, Y, S> for () {
    type Output = O;

    fn project(self, device: &mut Device, output: &P, state: S, hidden: &Y) -> O {
        output.forward(device, (), state, hidden)
    }
}

impl<M, P: for<'a, 'b> Module<(&'a Y, &'b M), (), S, Output = O>, Y, O, S> Routing<P, Y, S> for &M {
    type Output = O;

    fn project(self, device: &mut Device, output: &P, state: S, hidden: &Y) -> O {
        output.forward(device, (), state, (hidden, self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::{Ident, Topology, axes};
    use std::vec::IntoIter;

    /// Adds its multiple of the input to its slot and returns the slot.
    struct Accumulate(i32);

    impl<E> Module<&i32, E, &mut i32> for Accumulate {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: E, slot: &mut i32, input: &i32) -> i32 {
            *slot += self.0 * input;
            *slot
        }
    }

    /// Stores the key plus the value as the whole history.
    struct Store;

    impl<E> Module<(&mut i32, &(), (&i32, &i32)), E, &mut ()> for Store {
        type Output = ();

        fn forward(
            &self,
            _: &mut Device,
            _: E,
            _: &mut (),
            (cache, _, (key, value)): (&mut i32, &(), (&i32, &i32)),
        ) {
            *cache = key + value;
        }
    }

    /// Scores a query against the history by their product.
    struct Score;

    impl Module<(&i32, &i32, &()), (), &mut ()> for Score {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            _: (),
            _: &mut (),
            (query, history, _): (&i32, &i32, &()),
        ) -> i32 {
            query * history
        }
    }

    /// Stores the key plus the value, times the write operand.
    struct Place;

    impl<E> Module<(&mut i32, &i32, (&i32, &i32)), E, &mut ()> for Place {
        type Output = ();

        fn forward(
            &self,
            _: &mut Device,
            _: E,
            _: &mut (),
            (cache, write, (key, value)): (&mut i32, &i32, (&i32, &i32)),
        ) {
            *cache = write * (key + value);
        }
    }

    /// Scores a query against the history, plus the read operand.
    impl Module<(&i32, &i32, &i32), (), &mut ()> for Score {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            _: (),
            _: &mut (),
            (query, history, read): (&i32, &i32, &i32),
        ) -> i32 {
            query * history + read
        }
    }

    /// A tile of a call's rows: the rows, the tile's number and how many of its rows are live.
    #[derive(Clone, Copy)]
    struct Tile {
        rows: [i64; 4],
        at: usize,
        live: usize,
    }

    /// What a branch read: the weighted sum of a tile's live rows, the tile's number and its live
    /// rows.
    type Read = (i64, usize, usize);

    /// A weight written: it weighs a tile's live rows, or what a branch read.
    struct Written(i64);

    impl<E> Module<&Tile, E, &mut ()> for Written {
        type Output = Read;

        fn forward(&self, _: &mut Device, _: E, _: &mut (), tile: &Tile) -> Read {
            let sum: i64 = tile.rows[..tile.live].iter().sum();
            (self.0 * sum, tile.at, tile.live)
        }
    }

    impl<E> Module<Read, E, &mut ()> for Written {
        type Output = Read;

        fn forward(&self, _: &mut Device, _: E, _: &mut (), (sum, at, live): Read) -> Read {
            (self.0 * sum, at, live)
        }
    }

    impl<E> Module<&Read, E, &mut ()> for Written {
        type Output = Read;

        fn forward(&self, device: &mut Device, env: E, _: &mut (), read: &Read) -> Read {
            self.forward(device, env, &mut (), *read)
        }
    }

    axes![Envelope = 12];

    /// The tiles a value projection staged and has not given yet.
    type Staged = Option<Live<Tile, Envelope>>;

    /// A written value projection begins its siblings' input with nothing staged.
    impl<E> Module<Begin<()>, E, &mut ()> for Written {
        type Output = Staged;

        fn forward(&self, _: &mut Device, _: E, _: &mut (), _: Begin<()>) -> Staged {
            None
        }
    }

    /// Each tile is staged as it is, at its own rows.
    impl<E> Module<(&mut Staged, Tile), E, &mut ()> for Written {
        type Output = ();

        fn forward(&self, _: &mut Device, _: E, _: &mut (), (staged, tile): (&mut Staged, Tile)) {
            let rows = Live::<(), Envelope>::all().subrange(4 * tile.at, tile.live);
            *staged = Some(rows.with(tile));
        }
    }

    impl<E> Module<&mut Staged, E, &mut ()> for Written {
        type Output = Poll<Staged>;

        fn forward(&self, _: &mut Device, _: E, _: &mut (), staged: &mut Staged) -> Poll<Staged> {
            Poll::Ready(staged.take())
        }
    }

    /// A weight stored whole.
    struct Whole(i64);

    /// A weight stored as two parts, which are written as their sum.
    struct Parts(i64, i64);

    impl crate::Op<Begin<()>> for Whole {
        type Output = Written;

        fn forward(&self, _: &mut Device, _: Begin<()>) -> Written {
            Written(self.0)
        }
    }

    impl<X> !crate::Op<&X> for Whole {}

    impl crate::Op<Begin<()>> for Parts {
        type Output = Written;

        fn forward(&self, _: &mut Device, _: Begin<()>) -> Written {
            Written(self.0 + self.1)
        }
    }

    impl<X> !crate::Op<&X> for Parts {}

    impl<'x, X, E> Module<&'x X, E, &mut ()> for Whole
    where
        Written: for<'s> Module<&'x X, E, &'s mut (), Output = Read>,
    {
        type Output = Read;

        fn forward(&self, device: &mut Device, env: E, _: &mut (), input: &'x X) -> Read {
            Written(self.0).forward(device, env, &mut (), input)
        }
    }

    impl<'x, X, E> Module<&'x X, E, &mut ()> for Parts
    where
        Written: for<'s> Module<&'x X, E, &'s mut (), Output = Read>,
    {
        type Output = Read;

        fn forward(&self, device: &mut Device, env: E, _: &mut (), input: &'x X) -> Read {
            Written(self.0 + self.1).forward(device, env, &mut (), input)
        }
    }

    /// Positions a tile's queries up and its keys down by the tile's shift; streamed, by the shift
    /// given with the tile, whose rows it checks.
    struct Shift;

    impl<E> Module<((Read, Read), &(i64, Live<(), Envelope>)), E, &mut ()> for Shift {
        type Output = (Read, Read);

        fn forward(
            &self,
            device: &mut Device,
            env: E,
            _: &mut (),
            (rows, (shift, at)): ((Read, Read), &(i64, Live<(), Envelope>)),
        ) -> (Read, Read) {
            assert_eq!((at.start(), at.rows()), (4 * rows.0.1, rows.0.2));
            self.forward(device, env, &mut (), (rows, shift))
        }
    }

    impl<E> Module<((Read, Read), &i64), E, &mut ()> for Shift {
        type Output = (Read, Read);

        fn forward(
            &self,
            _: &mut Device,
            _: E,
            _: &mut (),
            ((query, key), shift): ((Read, Read), &i64),
        ) -> (Read, Read) {
            (
                (query.0 + shift, query.1, query.2),
                (key.0 - shift, key.1, key.2),
            )
        }
    }

    /// A call's tiles, each with its shift, one a poll.
    struct Given(IntoIter<(Tile, i64)>);

    impl Stream for Given {
        type Item = (Tile, i64);

        fn poll_next(&mut self, _: &mut Device) -> Poll<Option<(Tile, i64)>> {
            Poll::Ready(self.0.next())
        }
    }

    /// The query, key and value projections each read and write only their own slot.
    #[test]
    fn projects_each_on_its_own_slot() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let projection = Projection {
            query: Accumulate(1),
            key: Accumulate(2),
            value: Accumulate(3),
        };
        let mut state = (0, 100, 1000);
        let output = projection.forward(device, (), &mut state, &7);
        assert_eq!((output, state), ((7, 114, 1021), (7, 114, 1021)));
    }

    /// Attention keeps its projections', cache's and output projection's state apart: a call reads
    /// and updates each in its own slot of the state it is lent, and no other.
    #[test]
    fn keeps_attention_state_independent() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let attention = Attention {
            qkv: Projection {
                query: Accumulate(1),
                key: Accumulate(2),
                value: Accumulate(3),
            },
            cache: KeyValue {
                write: Store,
                read: (),
            },
            attend: Score,
            output: Accumulate(10),
        };
        let mut cache = 0;
        let mut state = ((0, 100, 1000), &mut cache, 5);
        let output = attention.forward(device, (), &mut state, ((&7, (&(), &(), &())), ()));
        let (projection, _, projected) = state;
        assert_eq!(
            (output, projection, cache, projected),
            (79455, (7, 114, 1021), 1135, 79455)
        );
    }

    /// A layer that holds its own write and read operands writes and reads its cache with them.
    #[test]
    fn writes_and_reads_with_its_layers_operands() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let attention = Attention {
            qkv: Positioned {
                projection: Projection {
                    query: Accumulate(1),
                    key: Accumulate(2),
                    value: Accumulate(3),
                },
                positions: crate::First,
            },
            cache: KeyValue {
                write: Place,
                read: (),
            },
            attend: Score,
            output: Accumulate(10),
        };
        let (mut cache, write, read) = (0, 2, 5);
        let mut state = ((0, 0, 0), (&mut cache, &write, &read), 0);
        let output = attention.forward(device, (&(), &()), &mut state, &7);
        assert_eq!((output, cache), (4950, 70));
    }

    /// Whichever form stores the weights, the streamed front computes the sequential one, every
    /// branch reads the same tile, a partial last one too, and each weight or gain changes only
    /// its own branch.
    #[test]
    fn routes_each_parameter_to_its_branch() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let tiles = vec![
            (
                Tile {
                    rows: [1, 2, 3, 4],
                    at: 0,
                    live: 4,
                },
                10,
            ),
            (
                Tile {
                    rows: [5, 6, 7, 8],
                    at: 1,
                    live: 4,
                },
                20,
            ),
            (
                Tile {
                    rows: [9, 10, 11, 12],
                    at: 2,
                    live: 3,
                },
                30,
            ),
        ];
        macro_rules! front {
            ($form:expr, [$query:expr, $key:expr, $value:expr, $query_norm:expr, $key_norm:expr]) => {
                Positioned {
                    projection: NormalizedQkv {
                        projection: Projection {
                            query: $form($query),
                            key: $form($key),
                            value: $form($value),
                        },
                        query_norm: $form($query_norm),
                        key_norm: $form($key_norm),
                    },
                    positions: Shift,
                }
            };
        }
        macro_rules! outputs {
            ($form:expr, $weights:expr) => {{
                let [query, key, value, query_norm, key_norm]: [i64; 5] = $weights;
                let mut state = (((), (), ()), (), ());
                let front = front!($form, [query, key, value, query_norm, key_norm]);
                let sequential: Vec<_> = tiles
                    .iter()
                    .map(|(tile, shift)| front.forward(&mut *device, (), &mut state, (tile, shift)))
                    .collect();
                let mut stream = front!($form, [query, key, value, query_norm, key_norm])
                    .into_stream((), &mut state, Given(tiles.clone().into_iter()));
                let mut streamed = Vec::new();
                loop {
                    match stream.poll_next(&mut *device) {
                        Poll::Ready(Some(rows)) => streamed.push(rows.into_value()),
                        Poll::Ready(None) => break,
                        Poll::Pending => {}
                    }
                }
                assert_eq!(streamed, sequential);
                streamed
            }};
        }
        let whole = |weight| Whole(weight);
        let parts = |weight: i64| Parts(weight - 1, 1);
        let weights = [2, 3, 5, 7, 11];
        let reference = outputs!(whole, weights);
        assert_eq!(outputs!(parts, weights), reference);
        for ((query, key, value), (tile, _)) in reference.iter().zip(&tiles) {
            let read = (tile.at, tile.live);
            assert_eq!(
                [(query.1, query.2), (key.1, key.2), (value.1, value.2)],
                [read; 3]
            );
        }
        let branches = [
            [true, false, false],
            [false, true, false],
            [false, false, true],
            [true, false, false],
            [false, true, false],
        ];
        for (changed, branches) in branches.iter().enumerate() {
            let mut changed_weights = weights;
            changed_weights[changed] += 1;
            for outputs in [
                outputs!(whole, changed_weights),
                outputs!(parts, changed_weights),
            ] {
                for (rows, reference) in outputs.iter().zip(&reference) {
                    let differs = [
                        rows.0 != reference.0,
                        rows.1 != reference.1,
                        rows.2 != reference.2,
                    ];
                    assert_eq!(differs, *branches);
                }
            }
        }
    }
}
