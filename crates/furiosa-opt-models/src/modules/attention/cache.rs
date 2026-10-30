//! The key/value cache stage, which writes this call's keys and values and reads the history
//! back, and the cached attention built on it.
//!
//! - Sequentially, the stage writes every row of the call, then reads the history.
//! - In phases over a paged cache, it retains a launch's rows, stores the rows attended as items
//!   before attention and the other rows once attention no longer reads them; a family's schedule
//!   calls the phases in that order.

use super::{Attention, Context, History, Paged, Positioned};

use crate::Module;
use crate::load::Load;
use crate::modules::generic::schedule::Spill;
use crate::ops::attention::full::paged;
use crate::ops::index::scatter;
use crate::tensor::cache::Blocks;
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use core::ops::Deref;
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

/// The call's keys and values, each row's `KvHeads` heads of `Head` features, as the stage
/// retains them until it stores them.
type Retention<Reach, KvHeads, Head> = (
    HbmTensor<bf16, Chip, m![Reach, KvHeads, Head]>,
    HbmTensor<bf16, Chip, m![Reach, KvHeads, Head]>,
);

/// The rows a stage stores once attention no longer reads them as retained rows.
pub struct Deferred<R>(Live<(), R>);

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
    Waves: AxisName,
    Windows: AxisName,
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
                &'w paged::Context<'e, Sequences, KvHeads, Groups, Head, Waves, Windows, L>,
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
                &'w paged::Context<'e, Sequences, KvHeads, Groups, Head, Waves, Windows, L>,
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
