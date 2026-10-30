//! The key/value cache stage, which writes this call's keys and values and reads the history
//! back, and the cached attention built on it.
//!
//! - Sequentially, the stage writes every row of the call, then reads the history.
//! - Streamed over a paged cache, it retains the call's rows, stores the rows attended as items
//!   before attention, lends attention a history of the pool and the retained rows, and stores
//!   the other rows once attention no longer reads them; the cached attention's stream orders
//!   these steps among its children.

use super::{Attend, Attention, Context, History, Paged, Positioned};

use crate::Module;
use crate::load::Load;
use crate::modules::generic::schedule::{Begin, IntoStream, Slot, Spill, Stream};
use crate::modules::language::Call;
use crate::ops::attention::full::paged::{self, Retained};
use crate::ops::index::{gather::Gather, scatter};
use crate::tensor::cache::Blocks;
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use core::ops::Deref;
use core::task::Poll;
use furiosa_opt_std::prelude::*;

/// Writes this call's keys and values to the cache and returns the history attention reads.
#[derive(DeviceSend, Load)]
pub struct KeyValue<Write, Read> {
    #[load(rename = "")]
    pub write: Write,
    #[load(rename = "")]
    pub read: Read,
}

/// The state is the layer's view of its cache; the input holds where this call's keys and values
/// are written, which history attention reads and what it reads besides.
impl<'s, 'b, Q, K, V, C, WriteOperand, ReadOperand, AttendOperand, H, Write, Read, Env: Copy>
    Module<((Q, K, V), (WriteOperand, ReadOperand, AttendOperand)), Env, &'s mut &'b mut C>
    for KeyValue<Write, Read>
where
    Write: for<'w, 'k, 'u> Module<
            (&'w mut C, WriteOperand, (&'k K, &'k V)),
            Env,
            &'u mut (),
            Output = (),
        >,
    Read: ReadHistory<&'s C, ReadOperand, AttendOperand, History = H>,
{
    type Output = (Q, H);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        cache: &'s mut &'b mut C,
        ((query, keys, values), (write, read, mask)): (
            (Q, K, V),
            (WriteOperand, ReadOperand, AttendOperand),
        ),
    ) -> Self::Output {
        self.write.forward(
            &mut *device,
            env,
            &mut (),
            (&mut **cache, write, (&keys, &values)),
        );
        (query, self.read.read(device, &**cache, read, mask))
    }
}

/// A layer that holds its own operands: the state holds its cache view beside where this call's
/// keys and values are written and which history attention reads.
impl<'s, 'b, 'c, Q, K, V, C, WriteOperand, ReadOperand, A, Write, Read, Out, Env: Copy>
    Module<((Q, K, V), A), Env, &'s mut (&'b mut C, &'c WriteOperand, &'c ReadOperand)>
    for KeyValue<Write, Read>
where
    Self: Module<
            ((Q, K, V), (&'c WriteOperand, &'c ReadOperand, A)),
            Env,
            &'s mut &'b mut C,
            Output = Out,
        >,
{
    type Output = Out;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (cache, write, read): &'s mut (&'b mut C, &'c WriteOperand, &'c ReadOperand),
        (projected, mask): ((Q, K, V), A),
    ) -> Out {
        self.forward(device, env, cache, (projected, (*write, *read, mask)))
    }
}

/// Reads the history attention attends over from the written cache; `()` lends the cache paged.
pub(crate) trait ReadHistory<Cache, Operand, Mask> {
    type History;

    fn read(&self, device: &mut Device, cache: Cache, read: Operand, mask: Mask) -> Self::History;
}

impl<C, R, A> ReadHistory<C, R, A> for () {
    type History = Paged<C, R>;

    fn read(&self, _: &mut Device, cache: C, read: R, _: A) -> Self::History {
        Paged(cache, read)
    }
}

impl<'a, C, R, A, K, V, M> ReadHistory<&'a C, R, A> for M
where
    M: for<'c, 's> Module<(&'c C, R), (), &'s mut (), Output = (K, V)>,
{
    type History = History<K, V, A>;

    fn read(&self, device: &mut Device, cache: &'a C, read: R, mask: A) -> Self::History {
        let (keys, values) = self.forward(device, (), &mut (), (cache, read));
        History(keys, values, mask)
    }
}

/// A gather reads the pool in bounded windows inside attention: the history is the pool with the
/// call's retained keys and values, read by this gather, and the read operand that selects it.
impl<'a, Reach: AxisName, KvHeads: AxisName, Head: AxisName, R, A>
    ReadHistory<
        (
            &'a Blocks<KvHeads, Head>,
            &'a Retention<Reach, KvHeads, Head>,
        ),
        R,
        A,
    > for &'a Gather
{
    type History = Paged<Retained<'a, Reach, KvHeads, Head>, R>;

    fn read(
        &self,
        _: &mut Device,
        (pool, (key, value)): (
            &'a Blocks<KvHeads, Head>,
            &'a Retention<Reach, KvHeads, Head>,
        ),
        read: R,
        _: A,
    ) -> Self::History {
        let history = Retained {
            pool,
            key,
            value,
            read: *self,
        };
        Paged(history, read)
    }
}

/// The call's keys and values, each row's `KvHeads` heads of `Head` features, as the stage
/// retains them until it stores them.
type Retention<Reach, KvHeads, Head> = (
    HbmTensor<bf16, Chip, m![Reach, KvHeads, Head]>,
    HbmTensor<bf16, Chip, m![Reach, KvHeads, Head]>,
);

/// The call's query rows, `Groups` query heads of `Head` features for each of `KvHeads`, which the
/// stage passes to attention as its sequential form passes the queries through.
type Queries<Reach, KvHeads, Groups, Head> =
    HbmTensor<bf16, Chip, m![Reach, KvHeads, Groups, Head]>;

/// The rows a stage stores once attention no longer reads them as retained rows.
pub struct Deferred<R>(Live<(), R>);

/// The places a call's rows are retained in, before any arrive: the slots its rows go to name
/// their extent, the read operand the query heads each KV head attends with.
impl<
    'w,
    'e,
    's,
    'b,
    Reach: AxisName,
    Sequences: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    const WAVES: usize,
    L: AxisName,
    WI: Deref<Target = HbmTensor<i32, Chip, m![Reach]>>,
    W,
    R,
    Env,
>
    Module<
        Begin<(
            &'w WI,
            &'w paged::Context<'e, Sequences, KvHeads, Groups, Head, WAVES, L>,
        )>,
        Env,
        &'s mut &'b mut Blocks<KvHeads, Head>,
    > for KeyValue<W, R>
{
    type Output = (
        Queries<Reach, KvHeads, Groups, Head>,
        Retention<Reach, KvHeads, Head>,
    );

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &'s mut &'b mut Blocks<KvHeads, Head>,
        _: Begin<(
            &'w WI,
            &'w paged::Context<'e, Sequences, KvHeads, Groups, Head, WAVES, L>,
        )>,
    ) -> Self::Output {
        (HbmTensor::new(), (HbmTensor::new(), HbmTensor::new()))
    }
}

/// A projected tile's queries, keys and values retained at its rows.
impl<'p, 's, 'b, QP, KV, Q, K, V, L: AxisName, C, W, R, Env>
    Module<((&'p mut QP, &'p mut KV), Live<(Q, K, V), L>), Env, &'s mut &'b mut C>
    for KeyValue<W, R>
where
    QP: for<'r> Spill<Device, &'r Q, usize>,
    KV: for<'r> Spill<Device, (&'r K, &'r V), usize>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &'s mut &'b mut C,
        ((queries, retained), tile): ((&'p mut QP, &'p mut KV), Live<(Q, K, V), L>),
    ) {
        let start = tile.start();
        let (query, key, value) = tile.into_value();
        queries.spill(&mut *device, &query, start);
        retained.spill(device, (&key, &value), start);
    }
}

/// The rows attended as items, the rows from row 0 of the call's `live` rows that attention reads
/// from the pool, stored before attention; the others are deferred.
impl<
    'p,
    'w,
    'e,
    's,
    'b,
    Reach: AxisName,
    Sequences: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    const WAVES: usize,
    L: AxisName,
    WI: Deref<Target = HbmTensor<i32, Chip, m![Reach]>>,
    W,
    R,
    Env,
>
    Module<
        (
            &'p Retention<Reach, KvHeads, Head>,
            (
                &'w WI,
                &'w paged::Context<'e, Sequences, KvHeads, Groups, Head, WAVES, L>,
            ),
            Live<(), L>,
        ),
        Env,
        &'s mut &'b mut Blocks<KvHeads, Head>,
    > for KeyValue<W, R>
where
    W: for<'d, 'r> Module<scatter::Rows<'r, Reach, KvHeads, Head, L>, (), &'d mut (), Output = ()>,
{
    type Output = Deferred<L>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        pool: &'s mut &'b mut Blocks<KvHeads, Head>,
        ((key, value), (slot, work), live): (
            &'p Retention<Reach, KvHeads, Head>,
            (
                &'w WI,
                &'w paged::Context<'e, Sequences, KvHeads, Groups, Head, WAVES, L>,
            ),
            Live<(), L>,
        ),
    ) -> Deferred<L> {
        let items = work.items();
        self.write.forward(
            device,
            (),
            &mut (),
            scatter::Rows {
                pair: (key, value),
                slot: &**slot,
                rows: items,
                pool: &mut **pool,
            },
        );
        Deferred(live.subrange(items.rows(), live.rows()))
    }
}

/// The deferred rows stored, once attention no longer reads them as retained rows.
impl<'p, 'w, 's, 'b, Reach, KvHeads, Head, L, WI, W, R, Env>
    Module<
        (Deferred<L>, &'p Retention<Reach, KvHeads, Head>, &'w WI),
        Env,
        &'s mut &'b mut Blocks<KvHeads, Head>,
    > for KeyValue<W, R>
where
    Reach: AxisName,
    KvHeads: AxisName,
    Head: AxisName,
    L: AxisName,
    WI: Deref<Target = HbmTensor<i32, Chip, m![Reach]>>,
    W: for<'d, 'r> Module<scatter::Rows<'r, Reach, KvHeads, Head, L>, (), &'d mut (), Output = ()>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        pool: &'s mut &'b mut Blocks<KvHeads, Head>,
        (Deferred(rows), (key, value), slot): (
            Deferred<L>,
            &'p Retention<Reach, KvHeads, Head>,
            &'w WI,
        ),
    ) {
        self.write.forward(
            device,
            (),
            &mut (),
            scatter::Rows {
                pair: (key, value),
                slot: &**slot,
                rows,
                pool: &mut **pool,
            },
        );
    }
}

impl<W, R> KeyValue<W, R> {
    /// The history attention reads: what the reader reads of the cache and the retained rows.
    fn history<'a, C, KV, RI, A>(
        &'a self,
        device: &mut Device,
        cache: &'a C,
        retained: &'a KV,
        read: RI,
        mask: A,
    ) -> <&'a R as ReadHistory<(&'a C, &'a KV), RI, A>>::History
    where
        &'a R: ReadHistory<(&'a C, &'a KV), RI, A>,
    {
        ReadHistory::read(&&self.read, device, (cache, retained), read, mask)
    }
}

/// Attention over this call's positioned keys and values together with the cached history.
pub type Cached<Q, P, W, R, A, O> = Attention<Positioned<Q, P>, KeyValue<W, R>, A, O>;

/// A cached attention reads its call's context and the mask its history read applies from the env.
impl<
    'x,
    'e,
    's,
    'b,
    X,
    PositionOperand,
    WriteOperand,
    ReadOperand,
    AttendOperand,
    C,
    QkvState,
    ProjectState,
    Qkv,
    Write,
    Read,
    Attend,
    Project,
    Out,
>
    Module<
        &'x X,
        (
            &'e Context<PositionOperand, WriteOperand, ReadOperand>,
            &'e AttendOperand,
        ),
        &'s mut (QkvState, &'b mut C, ProjectState),
    > for Attention<Qkv, KeyValue<Write, Read>, Attend, Project>
where
    Self: Module<
            (
                (
                    (&'x X, &'e PositionOperand),
                    (&'e WriteOperand, &'e ReadOperand, &'e AttendOperand),
                ),
                (),
            ),
            (
                &'e Context<PositionOperand, WriteOperand, ReadOperand>,
                &'e AttendOperand,
            ),
            &'s mut (QkvState, &'b mut C, ProjectState),
            Output = Out,
        >,
{
    type Output = Out;

    fn forward(
        &self,
        device: &mut Device,
        env: (
            &'e Context<PositionOperand, WriteOperand, ReadOperand>,
            &'e AttendOperand,
        ),
        state: &'s mut (QkvState, &'b mut C, ProjectState),
        input: &'x X,
    ) -> Self::Output {
        let (context, mask) = env;
        let operands = (&context.write, &context.read, mask);
        self.forward(
            device,
            env,
            state,
            (((input, &context.positions), operands), ()),
        )
    }
}

/// A cached attention whose layer holds its own write and read operands reads the positions and
/// the mask every layer reads from the env.
impl<
    'x,
    'e,
    's,
    'b,
    'c,
    X,
    P,
    A,
    C,
    WriteOperand,
    ReadOperand,
    QkvState,
    ProjectState,
    Qkv,
    Write,
    Read,
    Attend,
    Project,
    Out,
>
    Module<
        &'x X,
        (&'e P, &'e A),
        &'s mut (
            QkvState,
            (&'b mut C, &'c WriteOperand, &'c ReadOperand),
            ProjectState,
        ),
    > for Attention<Qkv, KeyValue<Write, Read>, Attend, Project>
where
    Self: Module<
            (((&'x X, &'e P), &'e A), ()),
            (&'e P, &'e A),
            &'s mut (
                QkvState,
                (&'b mut C, &'c WriteOperand, &'c ReadOperand),
                ProjectState,
            ),
            Output = Out,
        >,
{
    type Output = Out;

    fn forward(
        &self,
        device: &mut Device,
        env: (&'e P, &'e A),
        state: &'s mut (
            QkvState,
            (&'b mut C, &'c WriteOperand, &'c ReadOperand),
            ProjectState,
        ),
        input: &'x X,
    ) -> Self::Output {
        let (positions, mask) = env;
        self.forward(device, env, state, (((input, positions), mask), ()))
    }
}

// Paged attention reads the layer's cache in bounded windows; no read module materializes K/V.
impl<'x, 'e, X, PositionOperand, WriteOperand, ReadOperand, S, Qkv, Write, Attend, Project, Out>
    Module<&'x X, &'e Context<PositionOperand, WriteOperand, ReadOperand>, S>
    for Attention<Qkv, KeyValue<Write, ()>, Attend, Project>
where
    Self: Module<
            (
                (
                    (&'x X, &'e PositionOperand),
                    (&'e WriteOperand, &'e ReadOperand, &'e ()),
                ),
                (),
            ),
            &'e Context<PositionOperand, WriteOperand, ReadOperand>,
            S,
            Output = Out,
        >,
{
    type Output = Out;

    fn forward(
        &self,
        device: &mut Device,
        context: &'e Context<PositionOperand, WriteOperand, ReadOperand>,
        state: S,
        input: &'x X,
    ) -> Self::Output {
        let operands = (&context.write, &context.read, &());
        self.forward(
            device,
            context,
            state,
            (((input, &context.positions), operands), ()),
        )
    }
}

/// How far a cached attention's node got: it lands its children's weights, takes the call's
/// tiles while its projections give the rows it retains, attends once, then gives its output
/// projection's tiles.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    Landing,
    Taking,
    Attending,
    Giving,
}

/// A cached attention's node: its children, the projections' and the output projection's as
/// their streams, the call's live rows and context, the layer's cache view, the places its cache
/// stage retains the call's rows in, the rows that stage defers, how far it got, and its upstream.
pub(crate) struct Staged<'c, 's, 'b, F, W, R, A, P, PI, WI, RI, L, C, QP, KV, U> {
    attention: Attention<F, KeyValue<W, R>, A, P>,
    live: Live<(), L>,
    context: &'c Context<PI, WI, RI>,
    cache: &'s mut &'b mut C,
    places: Option<(QP, KV)>,
    deferred: Option<Deferred<L>>,
    step: Step,
    upstream: U,
}

impl<T, F, W, R, A, P, PI, WI, RI, L, C, QP, KV, U: AsMut<Slot<T>>> AsMut<Slot<T>>
    for Staged<'_, '_, '_, F, W, R, A, P, PI, WI, RI, L, C, QP, KV, U>
{
    fn as_mut(&mut self) -> &mut Slot<T> {
        self.upstream.as_mut()
    }
}

/// A cached attention streams over its call: its projections over the call's tiles with the
/// call's positions, its output projection over the attended rows, on the layer's cache view.
impl<'c, 's, 'b, U, X, PI, WI, RI, S, T, L, SF, C, SO, Qkv, W, R, A, P, F, G, QP, KV, O>
    IntoStream<U, Call<'c, (Context<PI, WI, RI>, S), T, L>, &'s mut (SF, &'b mut C, SO)>
    for Attention<Qkv, KeyValue<W, R>, A, P>
where
    U: Stream<Item = X>,
    L: AxisName,
    Qkv: IntoStream<Slot<(X, &'c PI)>, Live<(), L>, &'s mut SF, Stream = F>,
    KeyValue<W, R>:
        for<'x> Module<Begin<(&'c WI, &'c RI)>, (), &'x mut &'b mut C, Output = (QP, KV)>,
    for<'a> &'a R: ReadHistory<(&'a C, &'a KV), &'c RI, ()>,
    A: for<'a> Attend<QP, <&'a R as ReadHistory<(&'a C, &'a KV), &'c RI, ()>>::History, Output = O>,
    P: IntoStream<Slot<Live<O, L>>, Live<(), L>, &'s mut SO, Stream = G>,
    Staged<'c, 's, 'b, F, W, R, A, G, PI, WI, RI, L, C, QP, KV, U>: Stream,
{
    type Stream = Staged<'c, 's, 'b, F, W, R, A, G, PI, WI, RI, L, C, QP, KV, U>;

    fn into_stream(
        self,
        call: Call<'c, (Context<PI, WI, RI>, S), T, L>,
        (front, cache, output): &'s mut (SF, &'b mut C, SO),
        upstream: U,
    ) -> Self::Stream {
        let Attention {
            qkv,
            cache: stage,
            attend,
            output: project,
        } = self;
        Staged {
            attention: Attention {
                qkv: qkv.into_stream(call.live, front, Slot::new()),
                cache: stage,
                attend,
                output: project.into_stream(call.live, output, Slot::new()),
            },
            live: call.live,
            context: &call.context.0,
            cache,
            places: None,
            deferred: None,
            step: Step::Landing,
            upstream,
        }
    }
}

impl<'c, 's, 'b, F, W, R, A, P, PI, WI, RI, L, C, QP, KV, U, X, Q, K, V, O> Stream
    for Staged<'c, 's, 'b, F, W, R, A, P, PI, WI, RI, L, C, QP, KV, U>
where
    L: AxisName,
    U: Stream<Item = X>,
    F: Stream<Item = Live<(Q, K, V), L>> + AsMut<Slot<(X, &'c PI)>>,
    P: Stream + AsMut<Slot<Live<O, L>>>,
    KeyValue<W, R>: for<'x> Module<Begin<(&'c WI, &'c RI)>, (), &'x mut &'b mut C, Output = (QP, KV)>
        + for<'x, 'p> Module<
            ((&'p mut QP, &'p mut KV), Live<(Q, K, V), L>),
            (),
            &'x mut &'b mut C,
            Output = (),
        > + for<'x, 'p> Module<
            (&'p KV, (&'c WI, &'c RI), Live<(), L>),
            (),
            &'x mut &'b mut C,
            Output = Deferred<L>,
        > + for<'x, 'p> Module<(Deferred<L>, &'p KV, &'c WI), (), &'x mut &'b mut C, Output = ()>,
    for<'a> &'a R: ReadHistory<(&'a C, &'a KV), &'c RI, ()>,
    A: for<'a> Attend<QP, <&'a R as ReadHistory<(&'a C, &'a KV), &'c RI, ()>>::History, Output = O>,
{
    type Item = P::Item;

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Self::Item>> {
        let Self {
            attention:
                Attention {
                    qkv,
                    cache: stage,
                    attend,
                    output,
                },
            live,
            context,
            cache,
            places,
            deferred,
            step,
            upstream,
        } = self;
        let context: &'c Context<PI, WI, RI> = context;
        let operands = (&context.write, &context.read);
        if *step == Step::Landing {
            *step = Step::Taking;
            // Polled first, the projections' and the output projection's streams find no input
            // and land their weights.
            let _ = qkv.poll_next(&mut *device);
            let _ = output.poll_next(&mut *device);
            *places = Some(stage.forward(&mut *device, (), &mut **cache, Begin(operands)));
            // The landing poll reads no count, so the polls that follow, which do, keep its state.
            return Poll::Pending;
        }
        let Some((queries, retained)) = places else {
            return Poll::Pending;
        };
        if *step == Step::Taking {
            if qkv.as_mut().is_empty() {
                match upstream.poll_next(&mut *device) {
                    Poll::Ready(Some(tile)) => qkv
                        .as_mut()
                        .put(Poll::Ready(Some((tile, &context.positions)))),
                    Poll::Ready(None) => qkv.as_mut().put(Poll::Ready(None)),
                    Poll::Pending => {}
                }
            }
            // One projected tile a poll, so the projections' counts roll the polls that take them.
            match qkv.poll_next(&mut *device) {
                Poll::Ready(Some(rows)) => {
                    stage.forward(
                        &mut *device,
                        (),
                        &mut **cache,
                        ((&mut *queries, &mut *retained), rows),
                    );
                    return Poll::Pending;
                }
                Poll::Ready(None) => *step = Step::Attending,
                Poll::Pending => return Poll::Pending,
            }
        }
        if *step == Step::Attending {
            *step = Step::Giving;
            *deferred = Some(stage.forward(
                &mut *device,
                (),
                &mut **cache,
                (&*retained, operands, *live),
            ));
            let history = stage.history(&mut *device, &***cache, &*retained, &context.read, ());
            let attended = attend.attend(&mut *device, (&*queries, history));
            output.as_mut().put(Poll::Ready(Some(live.with(attended))));
        }
        if output.as_mut().is_empty() {
            output.as_mut().put(Poll::Ready(None));
        }
        let polled = output.poll_next(&mut *device);
        // The first projected tile, or the end of a call with none, releases the retained rows.
        if matches!(polled, Poll::Ready(_))
            && let Some(rows) = deferred.take()
        {
            stage.forward(device, (), &mut **cache, (rows, &*retained, &context.write));
        }
        polled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use core::array;
    use furiosa_opt_std::prelude::Topology;
    use std::vec::IntoIter;

    axes![Quad = 4, Envelope = 8];

    /// Four rows of one column.
    type Tile = [i64; 4];

    /// A pool of slots, each a key, a value and the writer's tag.
    type Pool = [(i64, i64, i64); 16];

    /// A call's read operand: the rows from row 0 that attend as decode rows, each over the pool
    /// slots of its history, its own included; the prompt rows after them attend over the
    /// committed slots and, causally, their own rows, which `own` slots hold.
    struct Work {
        decode: usize,
        histories: Vec<Vec<usize>>,
        committed: Vec<usize>,
        own: Vec<usize>,
    }

    /// Positions queries by its shift; keys double and values triple each row.
    struct Project;

    impl Module<(&Vec<i64>, &i64), (), &mut ()> for Project {
        type Output = (Vec<i64>, Vec<i64>, Vec<i64>);

        fn forward(
            &self,
            _: &mut Device,
            _: (),
            _: &mut (),
            (rows, shift): (&Vec<i64>, &i64),
        ) -> Self::Output {
            let each = |f: &dyn Fn(i64) -> i64| rows.iter().map(|&row| f(row)).collect();
            (
                each(&|row| row + shift),
                each(&|row| 2 * row),
                each(&|row| 3 * row),
            )
        }
    }

    /// The projections' stream: a tile at a time, positioned by the shift given with it.
    struct Projecting<'c>(Slot<(Live<Tile, Envelope>, &'c i64)>);

    impl<'c> Stream for Projecting<'c> {
        type Item = Live<(Tile, Tile, Tile), Envelope>;

        fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Self::Item>> {
            self.0.poll_next(device).map(|tile| {
                tile.map(|(rows, shift)| {
                    let each = |f: &dyn Fn(i64) -> i64| rows.value().map(f);
                    rows.with((
                        each(&|row| row + shift),
                        each(&|row| 2 * row),
                        each(&|row| 3 * row),
                    ))
                })
            })
        }
    }

    impl<'c> AsMut<Slot<(Live<Tile, Envelope>, &'c i64)>> for Projecting<'c> {
        fn as_mut(&mut self) -> &mut Slot<(Live<Tile, Envelope>, &'c i64)> {
            &mut self.0
        }
    }

    impl<'c, 's> IntoStream<Slot<(Live<Tile, Envelope>, &'c i64)>, Live<(), Envelope>, &'s mut ()>
        for Project
    {
        type Stream = Projecting<'c>;

        fn into_stream(
            self,
            _: Live<(), Envelope>,
            _: &'s mut (),
            upstream: Slot<(Live<Tile, Envelope>, &'c i64)>,
        ) -> Projecting<'c> {
            Projecting(upstream)
        }
    }

    /// Stores each row's key and value at its slot, with its tag.
    struct Write(i64);

    impl<E>
        Module<
            (
                &mut Pool,
                &Vec<usize>,
                (&Vec<i64>, &Vec<i64>),
                Live<(), Envelope>,
            ),
            E,
            &mut (),
        > for Write
    {
        type Output = ();

        fn forward(
            &self,
            _: &mut Device,
            _: E,
            _: &mut (),
            (pool, slots, (keys, values), rows): (
                &mut Pool,
                &Vec<usize>,
                (&Vec<i64>, &Vec<i64>),
                Live<(), Envelope>,
            ),
        ) {
            for row in rows.start()..rows.start() + rows.rows() {
                pool[slots[row]] = (keys[row], values[row], self.0);
            }
        }
    }

    impl<E> Module<(&mut Pool, &Vec<usize>, (&Vec<i64>, &Vec<i64>)), E, &mut ()> for Write {
        type Output = ();

        fn forward(
            &self,
            device: &mut Device,
            env: E,
            _: &mut (),
            (pool, slots, rows): (&mut Pool, &Vec<usize>, (&Vec<i64>, &Vec<i64>)),
        ) {
            let live = Live::over(slots.len());
            self.forward(device, env, &mut (), (pool, slots, rows, live));
        }
    }

    /// Reads each row's visible keys plus values, scaled.
    struct Read(i64);

    impl Read {
        /// Row `row`'s scaled sum, the call's prompt rows' own keys and values as `current` gives.
        fn sum(&self, work: &Work, pool: &Pool, current: impl Fn(usize) -> i64, row: usize) -> i64 {
            let slot = |&slot: &usize| pool[slot].0 + pool[slot].1;
            let sum: i64 = if row < work.decode {
                work.histories[row].iter().map(slot).sum()
            } else {
                let current: i64 = (work.decode..=row).map(current).sum();
                work.committed.iter().map(slot).sum::<i64>() + current
            };
            self.0 * sum
        }
    }

    impl Module<(&Pool, &Work), (), &mut ()> for Read {
        type Output = (Vec<i64>, ());

        fn forward(
            &self,
            _: &mut Device,
            _: (),
            _: &mut (),
            (pool, work): (&Pool, &Work),
        ) -> Self::Output {
            let own = |row: usize| {
                let slot = work.own[row - work.decode];
                pool[slot].0 + pool[slot].1
            };
            let rows = work.decode + work.own.len();
            (
                (0..rows)
                    .map(|row| self.sum(work, pool, own, row))
                    .collect(),
                (),
            )
        }
    }

    /// The call's retained rows, before the stage stores the prompt rows' keys and values.
    type Retained = (Vec<i64>, Vec<i64>);

    impl<'a, 'w> ReadHistory<(&'a Pool, &'a Retained), &'w Work, ()> for &'a Read {
        type History = History<Vec<i64>, (), ()>;

        fn read(
            &self,
            _: &mut Device,
            (pool, (keys, values)): (&'a Pool, &'a Retained),
            work: &'w Work,
            _: (),
        ) -> Self::History {
            let retained = |row: usize| keys[row] + values[row];
            let rows = work.decode + work.own.len();
            let sums = (0..rows)
                .map(|row| self.sum(work, pool, retained, row))
                .collect();
            History(sums, (), ())
        }
    }

    impl Spill<Device, &Tile, usize> for Vec<i64> {
        fn spill(&mut self, _: &mut Device, rows: &Tile, start: usize) {
            self[start..start + 4].copy_from_slice(rows);
        }
    }

    impl Spill<Device, (&Tile, &Tile), usize> for Retained {
        fn spill(&mut self, device: &mut Device, (keys, values): (&Tile, &Tile), start: usize) {
            self.0.spill(device, keys, start);
            self.1.spill(device, values, start);
        }
    }

    impl<E> Module<Begin<(&Vec<usize>, &Work)>, E, &mut &mut Pool> for KeyValue<Write, Read> {
        type Output = (Vec<i64>, Retained);

        fn forward(
            &self,
            _: &mut Device,
            _: E,
            _: &mut &mut Pool,
            _: Begin<(&Vec<usize>, &Work)>,
        ) -> Self::Output {
            let rows = || vec![0; Envelope::SIZE];
            (rows(), (rows(), rows()))
        }
    }

    impl<E> Module<(&Retained, (&Vec<usize>, &Work), Live<(), Envelope>), E, &mut &mut Pool>
        for KeyValue<Write, Read>
    {
        type Output = Deferred<Envelope>;

        fn forward(
            &self,
            device: &mut Device,
            env: E,
            pool: &mut &mut Pool,
            ((keys, values), (slots, work), live): (
                &Retained,
                (&Vec<usize>, &Work),
                Live<(), Envelope>,
            ),
        ) -> Deferred<Envelope> {
            let items = live.subrange(0, work.decode);
            let rows = (&mut **pool, slots, (keys, values), items);
            self.write.forward(device, env, &mut (), rows);
            Deferred(live.subrange(work.decode, live.rows()))
        }
    }

    impl<E> Module<(Deferred<Envelope>, &Retained, &Vec<usize>), E, &mut &mut Pool>
        for KeyValue<Write, Read>
    {
        type Output = ();

        fn forward(
            &self,
            device: &mut Device,
            env: E,
            pool: &mut &mut Pool,
            (Deferred(rows), (keys, values), slots): (Deferred<Envelope>, &Retained, &Vec<usize>),
        ) {
            let rows = (&mut **pool, slots, (keys, values), rows);
            self.write.forward(device, env, &mut (), rows);
        }
    }

    /// Attends a query by its row's sum.
    struct Score;

    impl<M> Module<(&Vec<i64>, &Vec<i64>, &(), M), (), &mut ()> for Score {
        type Output = Vec<i64>;

        fn forward(
            &self,
            _: &mut Device,
            _: (),
            _: &mut (),
            (queries, sums, ..): (&Vec<i64>, &Vec<i64>, &(), M),
        ) -> Vec<i64> {
            sums.iter()
                .zip(queries)
                .map(|(sum, query)| sum * query)
                .collect()
        }
    }

    /// Projects the attended rows by its multiple; streamed, gives their tiles or none at all.
    struct Emit(i64, bool);

    impl Module<&Vec<i64>, (), &mut ()> for Emit {
        type Output = Vec<i64>;

        fn forward(&self, _: &mut Device, _: (), _: &mut (), rows: &Vec<i64>) -> Vec<i64> {
            rows.iter().map(|row| self.0 * row).collect()
        }
    }

    /// The output projection's stream: the attended rows once, then their tiles.
    struct Emitting {
        emit: Emit,
        live: Live<(), Envelope>,
        attended: Option<Vec<i64>>,
        given: usize,
        upstream: Slot<Live<Vec<i64>, Envelope>>,
    }

    impl Stream for Emitting {
        type Item = Vec<i64>;

        fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Vec<i64>>> {
            if self.attended.is_none() {
                match self.upstream.poll_next(device) {
                    Poll::Ready(Some(rows)) => self.attended = Some(rows.into_value()),
                    Poll::Ready(None) => return Poll::Ready(None),
                    Poll::Pending => return Poll::Pending,
                }
            }
            if let (true, Some(rows)) = (self.emit.1, &self.attended)
                && self.given < self.live.count::<Quad>()
            {
                let at = self.live.tile::<Quad>(self.given);
                self.given += 1;
                let rows = &rows[at.start()..at.start() + at.rows()];
                return Poll::Ready(Some(rows.iter().map(|row| self.emit.0 * row).collect()));
            }
            match self.upstream.poll_next(device) {
                Poll::Ready(None) => Poll::Ready(None),
                _ => Poll::Pending,
            }
        }
    }

    impl AsMut<Slot<Live<Vec<i64>, Envelope>>> for Emitting {
        fn as_mut(&mut self) -> &mut Slot<Live<Vec<i64>, Envelope>> {
            &mut self.upstream
        }
    }

    impl<'s> IntoStream<Slot<Live<Vec<i64>, Envelope>>, Live<(), Envelope>, &'s mut ()> for Emit {
        type Stream = Emitting;

        fn into_stream(
            self,
            live: Live<(), Envelope>,
            _: &'s mut (),
            upstream: Slot<Live<Vec<i64>, Envelope>>,
        ) -> Emitting {
            Emitting {
                emit: self,
                live,
                attended: None,
                given: 0,
                upstream,
            }
        }
    }

    /// A call's tiles, one a poll.
    struct Given(IntoIter<Live<Tile, Envelope>>);

    impl Stream for Given {
        type Item = Live<Tile, Envelope>;

        fn poll_next(&mut self, _: &mut Device) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.0.next())
        }
    }

    /// A call: its rows, each row's slot and its read operand.
    struct Case {
        rows: Vec<i64>,
        slots: Vec<usize>,
        work: Work,
    }

    type Layer = Attention<Project, KeyValue<Write, Read>, Score, Emit>;

    /// The pool before any call: every slot holds a sentinel of its own.
    fn pool() -> Pool {
        array::from_fn(|slot| (-(slot as i64) - 1, 0, 0))
    }

    /// The call's attended outputs and the pool after it, by the sequential composition.
    fn sequential(device: &mut Device, layer: &Layer, case: &Case, pool: &mut Pool) -> Vec<i64> {
        let operands = (&case.slots, &case.work, &());
        let input = (((&case.rows, &3), operands), ());
        layer.forward(device, (), &mut ((), &mut *pool, ()), input)
    }

    /// The call's attended outputs and the pool after it, streamed tile by tile.
    fn streamed(device: &mut Device, layer: Layer, case: &Case, pool: &mut Pool) -> Vec<i64> {
        let live = Live::<(), Envelope>::over(case.rows.len());
        let tiles = (0..live.count::<Quad>())
            .map(|tile| {
                let at = live.tile::<Quad>(tile);
                at.with(array::from_fn(|row| {
                    case.rows.get(at.start() + row).copied().unwrap_or(0)
                }))
            })
            .collect::<Vec<_>>();
        let context = (
            Context {
                positions: 3,
                write: case.slots.clone(),
                read: Work {
                    decode: case.work.decode,
                    histories: case.work.histories.clone(),
                    committed: case.work.committed.clone(),
                    own: case.work.own.clone(),
                },
            },
            (),
        );
        let call = Call {
            context: &context,
            tokens: &(),
            live,
        };
        let mut state = ((), &mut *pool, ());
        let mut stream = layer.into_stream(call, &mut state, Given(tiles.into_iter()));
        let mut outputs = Vec::new();
        loop {
            match stream.poll_next(device) {
                Poll::Ready(Some(rows)) => outputs.extend(rows),
                Poll::Ready(None) => break,
                Poll::Pending => {}
            }
        }
        outputs
    }

    /// Streamed, a cached attention stores the decode rows before it attends and the prompt rows
    /// after, and computes what the sequential composition does: the same outputs and the same
    /// pool, its untouched sentinels too, for a prompt, then a call of decode and prompt rows
    /// reusing the first call's slots; with no tile given it still stores the prompt rows; each
    /// writer tag and reader scale shows in the pool or the outputs.
    #[test]
    fn streams_the_sequential_cache_stage() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let prompt = Case {
            rows: vec![1, 2, 3, 4, 5, 6],
            slots: (0..6).collect(),
            work: Work {
                decode: 0,
                histories: vec![],
                committed: vec![],
                own: (0..6).collect(),
            },
        };
        let mixed = Case {
            rows: vec![7, 8, 9, 10, 11, 12, 13],
            slots: vec![6, 7, 8, 9, 10, 11, 12],
            work: Work {
                decode: 2,
                histories: vec![(0..7).collect(), vec![2, 3, 7]],
                committed: vec![0, 1],
                own: (8..13).collect(),
            },
        };
        let layer = |tag, scale, gives| Attention {
            qkv: Project,
            cache: KeyValue {
                write: Write(tag),
                read: Read(scale),
            },
            attend: Score,
            output: Emit(5, gives),
        };
        let run = |device: &mut Device, (tag, scale, gives): (i64, i64, bool)| {
            let (mut ours, mut theirs) = (pool(), pool());
            let calls = [&prompt, &mixed].map(|case| {
                let reference = sequential(device, &layer(tag, scale, gives), case, &mut theirs);
                let outputs = streamed(device, layer(tag, scale, gives), case, &mut ours);
                assert_eq!(ours, theirs);
                assert_eq!(outputs, if gives { reference.clone() } else { vec![] });
                reference
            });
            (calls, ours)
        };
        let (outputs, pool) = run(device, (1, 1, true));
        assert_eq!(pool[13..], [(-14, 0, 0), (-15, 0, 0), (-16, 0, 0)]);
        assert_eq!(run(device, (1, 1, false)).1, pool);
        let (tagged, retagged) = (run(device, (2, 1, true)), run(device, (1, 2, true)));
        assert_eq!(tagged.0, outputs);
        assert_ne!(tagged.1, pool);
        assert_ne!(retagged.0, outputs);
        assert_eq!(retagged.1, pool);
    }
}
