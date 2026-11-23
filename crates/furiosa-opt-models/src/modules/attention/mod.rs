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

use furiosa_opt_std::prelude::{Device, DeviceSend};

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
    use furiosa_opt_std::prelude::Topology;

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
}
