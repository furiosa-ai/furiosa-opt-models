//! The streaming execution form: a node's computation as a stream its downstream polls, a tile
//! at a time, like `futures::Stream`.
//!
//! - A node owns its upstream and polls it for the next input tile when it needs one; a poll does
//!   the work that tile, or the wait, allows and answers `Ready(Some(tile))` with an output tile,
//!   `Pending` when it has none yet, or `Ready(None)` once it is done. Polled again, a done node
//!   answers `Ready(None)` again.
//! - A node busy with a tile over several polls does not poll its upstream until that tile is done;
//!   a node whose upstream answers `Pending` may do work that needs no input, e.g. write its
//!   weights.
//! - A composite builds its children's streams over each other: tile k passes every node before
//!   tile k + 1 arrives. [`Scheduled`] polls a model root this way until it is done.
//! - A node takes the state its module's `forward` takes: a composite gives each child its slot,
//!   and a leaf's node lends its slot to each `forward` it runs.
//! - A node owns its module: a borrowed tree streams as references (`&M` is a module), a layer
//!   built as a view when it starts streams as that view.
//! - A leaf streams by one of two nodes: [`RowWise`] runs its `forward` on each tile, [`CrossRow`]
//!   takes every tile, runs its `forward` once and gives the output tile by tile.
//! - A stream whose input is not one owned upstream (a residual's branch, a layer of a stack) pulls
//!   from a [`Slot`] its owner fills, reached through the stream's `AsMut<Slot<T>>`.

use crate::Module;
use crate::tensor::live::Live;
use core::marker::PhantomData;
use core::mem;
use core::task::Poll;
use furiosa_opt_std::prelude::*;

/// A node's computation as a stream of output tiles, like `futures::Stream` with the device as its
/// context: nothing wakes a node, its downstream polls it until it answers `Ready(None)`.
pub(crate) trait Stream {
    /// An output tile.
    type Item;

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Self::Item>>;
}

/// A module as the stream computing it over its upstream `U`'s tiles, with the env `E` and state
/// `S` its `forward` takes; a trait, like `IntoIterator`, as the stream's type is the module's.
pub(crate) trait IntoStream<U: Stream, E, S> {
    type Stream: Stream;

    fn into_stream(self, env: E, state: S, upstream: U) -> Self::Stream;
}

/// A stream of no tiles: a model root's upstream.
pub struct Empty<T>(PhantomData<T>);

impl<T> Empty<T> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Stream for Empty<T> {
    type Item = T;

    fn poll_next(&mut self, _: &mut Device) -> Poll<Option<T>> {
        Poll::Ready(None)
    }
}

/// The input of a stream its owner fills: polled, it gives what was put in it once, a tile or
/// `Ready(None)` to end the stream, and `Pending` while it is empty.
pub struct Slot<T>(Poll<Option<T>>);

impl<T> Slot<T> {
    pub const fn new() -> Self {
        Self(Poll::Pending)
    }

    /// Whether the stream took what was last put in, so the slot takes the next.
    pub const fn is_empty(&self) -> bool {
        matches!(self.0, Poll::Pending)
    }

    /// Puts the owner's next poll result in: a tile, `Ready(None)` once the input ends, or
    /// `Pending`, which puts nothing. The owner puts a result only while the slot is empty.
    pub fn put(&mut self, polled: Poll<Option<T>>) {
        if !matches!(polled, Poll::Pending) {
            self.0 = polled;
        }
    }
}

impl<T> Stream for Slot<T> {
    type Item = T;

    fn poll_next(&mut self, _: &mut Device) -> Poll<Option<T>> {
        match mem::replace(&mut self.0, Poll::Pending) {
            Poll::Ready(None) => {
                self.0 = Poll::Ready(None);
                Poll::Ready(None)
            }
            polled => polled,
        }
    }
}

impl<T> AsMut<Self> for Slot<T> {
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

/// A place in HBM a node reads on chip: [`Fill::fill`] moves the rows of one tile `T` on chip
/// where the value enters the node.
pub(crate) trait Fill<D, T> {
    /// The tile's rows, on chip.
    type Rows;
    fn fill(&self, context: &mut D, tile: T) -> Self::Rows;
}

/// A place in HBM a node writes from chip: [`Spill::spill`] stores the rows `R` of one tile `T`
/// where they leave the node.
pub(crate) trait Spill<D, R, T> {
    fn spill(&mut self, context: &mut D, rows: R, tile: T);
}

/// The node of a leaf that computes each output tile from one input tile, like `futures`' `Map`
/// with the leaf's `forward` for the closure; it owns its module, a borrowed leaf or a view.
pub struct RowWise<'s, M, E, S, U> {
    module: M,
    env: E,
    state: &'s mut S,
    upstream: U,
}

impl<'s, M, E, S, U> RowWise<'s, M, E, S, U> {
    pub const fn new(module: M, env: E, state: &'s mut S, upstream: U) -> Self {
        Self {
            module,
            env,
            state,
            upstream,
        }
    }
}

impl<E: Copy, S, O, U: Stream, M> Stream for RowWise<'_, M, E, S, U>
where
    M: for<'s, 'x> Module<&'x U::Item, E, &'s mut S, Output = O>,
{
    type Item = O;

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<O>> {
        match self.upstream.poll_next(device) {
            Poll::Ready(Some(tile)) => Poll::Ready(Some(self.module.forward(
                device,
                self.env,
                &mut *self.state,
                &tile,
            ))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T, M, E, S, U: AsMut<Slot<T>>> AsMut<Slot<T>> for RowWise<'_, M, E, S, U> {
    fn as_mut(&mut self) -> &mut Slot<T> {
        self.upstream.as_mut()
    }
}

/// The node of a leaf that reads its input whole: it gathers `P`-row tiles in `W`, writes the
/// place `O` after the last live row of `R`, gives `O` a tile a poll, then ends with its upstream.
pub struct CrossRow<'s, M, E, S, W, O, P, R, U> {
    module: M,
    env: E,
    state: &'s mut S,
    input: W,
    output: O,
    live: Live<(), R>,
    taken: usize,
    given: usize,
    upstream: U,
    tile: PhantomData<P>,
}

impl<'s, M, E: Copy + Into<Live<(), R>>, S, W, O, P, R, U> CrossRow<'s, M, E, S, W, O, P, R, U> {
    pub fn new(module: M, env: E, state: &'s mut S, (input, output): (W, O), upstream: U) -> Self {
        Self {
            module,
            env,
            state,
            input,
            output,
            live: env.into(),
            taken: 0,
            given: 0,
            upstream,
            tile: PhantomData,
        }
    }
}

impl<X, E, S, W, O, P, R, U, M> Stream for CrossRow<'_, M, E, S, W, O, P, R, U>
where
    E: Copy,
    P: AxisName,
    R: AxisName,
    U: Stream<Item = Live<X, R>>,
    W: Spill<Device, X, Live<(), R>>,
    M: for<'s, 'w, 'o> Module<(&'w W, &'o mut O), E, &'s mut S, Output = ()>,
    O: Fill<Device, Live<(), R>>,
{
    type Item = Live<O::Rows, R>;

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Self::Item>> {
        if self.taken < self.live.count::<P>() {
            if let Poll::Ready(Some(tile)) = self.upstream.poll_next(device) {
                let at = tile.with(());
                self.input.spill(device, tile.into_value(), at);
                self.taken += 1;
                if self.taken >= self.live.count::<P>() {
                    let places = (&self.input, &mut self.output);
                    self.module
                        .forward(device, self.env, &mut *self.state, places);
                }
            }
            return Poll::Pending;
        }
        if self.given < self.live.count::<P>() {
            let at = self.live.tile::<P>(self.given);
            self.given += 1;
            return Poll::Ready(Some(at.with(self.output.fill(device, at))));
        }
        // The node ends with its upstream, whose end may still do work.
        match self.upstream.poll_next(device) {
            Poll::Ready(None) => Poll::Ready(None),
            _ => Poll::Pending,
        }
    }
}

impl<T, M, E, S, W, O, P, R, U: AsMut<Slot<T>>> AsMut<Slot<T>>
    for CrossRow<'_, M, E, S, W, O, P, R, U>
{
    fn as_mut(&mut self) -> &mut Slot<T> {
        self.upstream.as_mut()
    }
}

/// A stream of one tile, like `futures`' `stream::once`: its first poll gives its leaf's `forward`
/// on the input it holds, e.g. a call's tokens streamed as one tile of all its rows.
pub struct Once<'m, M, E, X> {
    module: &'m M,
    env: E,
    input: X,
    given: usize,
}

impl<'m, M, E, X> Once<'m, M, E, X> {
    pub const fn new(module: &'m M, env: E, input: X) -> Self {
        Self {
            module,
            env,
            input,
            given: 0,
        }
    }
}

impl<X: Copy, O, E: Copy, M> Stream for Once<'_, M, E, X>
where
    M: for<'s> Module<X, E, &'s mut (), Output = O>,
{
    type Item = O;

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<O>> {
        if self.given < 1 {
            self.given += 1;
            return Poll::Ready(Some(self.module.forward(
                device,
                self.env,
                &mut (),
                self.input,
            )));
        }
        Poll::Ready(None)
    }
}

/// A module's input taken up to its first boundary, where it returns its state; given back, the
/// state runs the next step, and the state types refuse any other order at compile time.
pub struct Begin<T>(pub T);

/// A model root polled until done, computing what its `forward` does on the same state in the fused
/// order; it streams over no input, writes its results to places its state names, and owns `M`.
#[derive(DeviceSend)]
pub struct Scheduled<M>(pub M);

impl<M: Copy + IntoStream<Empty<()>, E, S>, E, S> Module<(), E, S> for Scheduled<M> {
    type Output = ();

    fn forward(&self, device: &mut Device, env: E, state: S, (): ()) {
        let mut root = self.0.into_stream(env, state, Empty::new());
        let mut done = false;
        while !done {
            done = matches!(root.poll_next(device), Poll::Ready(None));
        }
    }
}

#[cfg(test)]
pub(in crate::modules) mod tests {
    use super::*;
    use crate::load::Context;
    use crate::modules::generic::collection::{Collected, Stack};
    use crate::modules::generic::{Layers, Rolled, Stacked, View};
    use crate::tests::DEVICE;
    use crate::{Module, Sequential};
    use furiosa_opt_std::prelude::Topology;

    axes![Window = 40, Rest = 39, Width = 128, One = 1];

    type Count = HbmTensor<i32, m![1], m![1]>;
    type Rows = HbmTensor<bf16, m![1], m![Window, Width]>;
    type Tile = DmTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![Width]>;
    pub(in crate::modules) type Launch = Live<(), Window>;
    type Row = Live<Tile, Window>;
    /// A place a toy leaf stores its tiles into, which the host reads back.
    type Place<'s> = &'s mut Rows;

    impl Spill<Device, Tile, Launch> for Rows {
        fn spill(&mut self, device: &mut Device, rows: Tile, tile: Launch) {
            put(device, &rows, tile.start(), self);
        }
    }

    impl Fill<Device, Launch> for Rows {
        type Rows = Tile;

        fn fill(&self, device: &mut Device, tile: Launch) -> Tile {
            to_dm(device, self, tile.start())
        }
    }

    /// Writes each live row of the rows its state names, one a poll.
    struct ToDm;

    /// Answers `Pending` at its first poll, then writes each live row, one a poll.
    struct Late;

    /// Stores each input tile at its row of the rows its state names and passes it on.
    struct Put;

    /// Copies its whole input: as a leaf, it takes every tile before it gives the first.
    struct Gather;

    /// Writes one weight row at its first poll, stores each input tile at its row and passes it
    /// on, and stores the weight at row 0 of its second place once every tile is through.
    struct Warm;

    /// A hand-written toy node: its places, the live rows, how many tiles it has taken and given,
    /// and its upstream.
    struct Toy<L, S, U> {
        leaf: L,
        places: S,
        live: Launch,
        taken: usize,
        given: usize,
        upstream: U,
    }

    impl<L, S, U> Toy<L, S, U> {
        fn new(leaf: L, live: Launch, places: S, upstream: U) -> Self {
            Self {
                leaf,
                places,
                live,
                taken: 0,
                given: 0,
                upstream,
            }
        }

        /// The tile of the next live row of `rows`, written.
        fn to_dm(&mut self, device: &mut Device, rows: impl Fn(&S) -> &Rows) -> Poll<Option<Row>> {
            if self.given < self.live.rows() {
                let at = self.given;
                self.given += 1;
                Poll::Ready(Some(self.live.subrange(at, 1).with(to_dm(
                    device,
                    rows(&self.places),
                    at,
                ))))
            } else {
                Poll::Ready(None)
            }
        }
    }

    impl<T, L, S, U: AsMut<Slot<T>>> AsMut<Slot<T>> for Toy<L, S, U> {
        fn as_mut(&mut self) -> &mut Slot<T> {
            self.upstream.as_mut()
        }
    }

    impl<'s, 'r, U: Stream> IntoStream<U, Launch, &'s mut &'r Rows> for &ToDm {
        type Stream = Toy<ToDm, &'r Rows, U>;

        fn into_stream(self, live: Launch, rows: &'s mut &'r Rows, upstream: U) -> Self::Stream {
            Toy::new(ToDm, live, *rows, upstream)
        }
    }

    /// A source: it never polls its upstream.
    impl<U> Stream for Toy<ToDm, &Rows, U> {
        type Item = Row;

        fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Row>> {
            self.to_dm(device, |rows| *rows)
        }
    }

    impl<'s, 'r, U: Stream> IntoStream<U, Launch, &'s mut &'r Rows> for &Late {
        type Stream = Toy<Late, &'r Rows, U>;

        fn into_stream(self, live: Launch, rows: &'s mut &'r Rows, upstream: U) -> Self::Stream {
            Toy::new(Late, live, *rows, upstream)
        }
    }

    /// A source; `taken` counts the first poll, which only waits.
    impl<U> Stream for Toy<Late, &Rows, U> {
        type Item = Row;

        fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Row>> {
            if self.taken < 1 {
                self.taken = 1;
                return Poll::Pending;
            }
            self.to_dm(device, |rows| *rows)
        }
    }

    impl<'m, 's, 'p, U: Stream<Item = Row>> IntoStream<U, Launch, &'s mut Place<'p>> for &'m Put {
        type Stream = RowWise<'s, &'m Put, Launch, Place<'p>, U>;

        fn into_stream(self, live: Launch, rows: &'s mut Place<'p>, upstream: U) -> Self::Stream {
            RowWise::new(self, live, rows, upstream)
        }
    }

    impl Module<&Row, Launch, &mut Place<'_>> for Put {
        type Output = Row;

        fn forward(&self, device: &mut Device, _: Launch, rows: &mut Place<'_>, tile: &Row) -> Row {
            put(device, tile.value(), tile.start(), rows);
            tile.with(tile.value().to_dm(&mut device.tdma))
        }
    }

    impl<'m, 's, U: Stream<Item = Row>> IntoStream<U, Launch, &'s mut ()> for &'m Gather {
        type Stream = CrossRow<'s, &'m Gather, Launch, (), Rows, Rows, One, Window, U>;

        fn into_stream(self, live: Launch, unit: &'s mut (), upstream: U) -> Self::Stream {
            let places = (HbmTensor::new(), HbmTensor::new());
            CrossRow::new(self, live, unit, places, upstream)
        }
    }

    /// A gather built as a layer's view streams itself.
    impl<'s, U: Stream<Item = Row>> IntoStream<U, Launch, &'s mut ()> for Gather {
        type Stream = CrossRow<'s, Gather, Launch, (), Rows, Rows, One, Window, U>;

        fn into_stream(self, live: Launch, unit: &'s mut (), upstream: U) -> Self::Stream {
            let places = (HbmTensor::new(), HbmTensor::new());
            CrossRow::new(self, live, unit, places, upstream)
        }
    }

    impl<const N: usize> View<N> for Gather {
        type Storage = ();
        type Layer = Self;

        fn view((): &(), _: usize) -> Self {
            Self
        }

        fn stack(_: [Self; N], _: &mut Context<'_>) {}
    }

    impl Module<(&Rows, &mut Rows), Launch, &mut ()> for Gather {
        type Output = ();

        fn forward(
            &self,
            device: &mut Device,
            _: Launch,
            _: &mut (),
            (rows, copy): (&Rows, &mut Rows),
        ) {
            rows.view().to_hbm_view(&mut device.tdma, copy.view_mut());
        }
    }

    type Places<'s> = (&'s Rows, (&'s mut Rows, &'s mut Rows));

    impl<'s, 'p, U: Stream<Item = Row>> IntoStream<U, Launch, &'s mut Places<'p>> for &Warm {
        type Stream = Toy<(Warm, Option<Tile>), &'s mut Places<'p>, U>;

        fn into_stream(
            self,
            live: Launch,
            places: &'s mut Places<'p>,
            upstream: U,
        ) -> Self::Stream {
            Toy::new((Warm, None), live, places, upstream)
        }
    }

    /// `given` counts the weight's two moves: written at the first poll, after its upstream's, and
    /// stored once every tile is through.
    impl<U: Stream<Item = Row>> Stream for Toy<(Warm, Option<Tile>), &mut Places<'_>, U> {
        type Item = Row;

        fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Row>> {
            let (weights, (rows, rest)) = &mut *self.places;
            if self.taken < self.live.rows() {
                let polled = self.upstream.poll_next(device);
                if self.given < 1 {
                    self.leaf.1 = Some(to_dm(device, weights, 0));
                    self.given = 1;
                }
                return match polled {
                    Poll::Ready(Some(tile)) => {
                        put(device, tile.value(), tile.start(), rows);
                        self.taken += 1;
                        Poll::Ready(Some(tile))
                    }
                    _ => Poll::Pending,
                };
            }
            if let Some(weight) = self.leaf.1.take() {
                put(device, &weight, 0, rest);
            }
            Poll::Ready(None)
        }
    }

    type Layer = Sequential<Put, Put>;

    const STACK: Sequential<ToDm, Sequential<Layer, Layer>> =
        Sequential(ToDm, Sequential(Sequential(Put, Put), Sequential(Put, Put)));

    /// A row's number, a tile a row.
    pub(in crate::modules) type Number = Live<usize, Window>;

    /// The numbers a layer took, which its slot views.
    pub(in crate::modules) type History<'h> = &'h mut Vec<usize>;

    /// Gives each live row's index, a row a poll.
    pub(in crate::modules) struct Indices {
        live: Launch,
        given: usize,
    }

    impl Indices {
        pub(in crate::modules) const fn new(live: Launch) -> Self {
            Self { live, given: 0 }
        }
    }

    impl Stream for Indices {
        type Item = Number;

        fn poll_next(&mut self, _: &mut Device) -> Poll<Option<Number>> {
            if self.given < self.live.rows() {
                self.given += 1;
                let at = self.given - 1;
                return Poll::Ready(Some(self.live.subrange(at, 1).with(at)));
            }
            Poll::Ready(None)
        }
    }

    /// Records each number in the history its slot views and gives the next number.
    pub(in crate::modules) struct Record;

    impl<'v, 'h, E> Module<Number, E, &'v mut History<'h>> for Record {
        type Output = Number;

        fn forward(
            &self,
            _: &mut Device,
            _: E,
            history: &'v mut History<'h>,
            at: Number,
        ) -> Number {
            history.push(*at.value());
            at.with(at.value() + 1)
        }
    }

    impl<'v, 'h, E> Module<&Number, E, &'v mut History<'h>> for Record {
        type Output = Number;

        fn forward(
            &self,
            device: &mut Device,
            env: E,
            history: &'v mut History<'h>,
            at: &Number,
        ) -> Number {
            self.forward(device, env, history, *at)
        }
    }

    impl<'m, 's, 'h, E: Copy, U: Stream<Item = Number>> IntoStream<U, E, &'s mut History<'h>>
        for &'m Record
    {
        type Stream = RowWise<'s, &'m Record, E, History<'h>, U>;

        fn into_stream(self, env: E, history: &'s mut History<'h>, upstream: U) -> Self::Stream {
            RowWise::new(self, env, history, upstream)
        }
    }

    /// A record built as a layer's view streams itself.
    impl<'s, 'h, E: Copy, U: Stream<Item = Number>> IntoStream<U, E, &'s mut History<'h>> for Record {
        type Stream = RowWise<'s, Record, E, History<'h>, U>;

        fn into_stream(self, env: E, history: &'s mut History<'h>, upstream: U) -> Self::Stream {
            RowWise::new(self, env, history, upstream)
        }
    }

    impl<const N: usize> View<N> for Record {
        type Storage = ();
        type Layer = Self;

        fn view((): &(), _: usize) -> Self {
            Self
        }

        fn stack(_: [Self; N], _: &mut Context<'_>) {}
    }

    /// Gives its whole input once it took every number.
    pub(in crate::modules) struct Collect;

    impl Spill<Device, usize, Launch> for Vec<usize> {
        fn spill(&mut self, _: &mut Device, number: usize, at: Launch) {
            self[at.start()] = number;
        }
    }

    impl Fill<Device, Launch> for Vec<usize> {
        type Rows = usize;

        fn fill(&self, _: &mut Device, at: Launch) -> usize {
            self[at.start()]
        }
    }

    impl<E> Module<(&Vec<usize>, &mut Vec<usize>), E, &mut ()> for Collect {
        type Output = ();

        fn forward(
            &self,
            _: &mut Device,
            _: E,
            _: &mut (),
            (numbers, copy): (&Vec<usize>, &mut Vec<usize>),
        ) {
            copy.clone_from(numbers);
        }
    }

    impl<'m, 's, U: Stream<Item = Number>> IntoStream<U, Launch, &'s mut ()> for &'m Collect {
        type Stream = CrossRow<'s, &'m Collect, Launch, (), Vec<usize>, Vec<usize>, One, Window, U>;

        fn into_stream(self, live: Launch, unit: &'s mut (), upstream: U) -> Self::Stream {
            let places = (vec![0; live.rows()], vec![0; live.rows()]);
            CrossRow::new(self, live, unit, places, upstream)
        }
    }

    impl<'s, U: Stream<Item = Number>> IntoStream<U, Launch, &'s mut ()> for Collect {
        type Stream = CrossRow<'s, Collect, Launch, (), Vec<usize>, Vec<usize>, One, Window, U>;

        fn into_stream(self, live: Launch, unit: &'s mut (), upstream: U) -> Self::Stream {
            let places = (vec![0; live.rows()], vec![0; live.rows()]);
            CrossRow::new(self, live, unit, places, upstream)
        }
    }

    impl<const N: usize> View<N> for Collect {
        type Storage = ();
        type Layer = Self;

        fn view((): &(), _: usize) -> Self {
            Self
        }

        fn stack(_: [Self; N], _: &mut Context<'_>) {}
    }

    fn to_dm(device: &mut Device, rows: &Rows, at: usize) -> Tile {
        rows.view()
            .tile::<m![Window], 1, m![Window = 1 # Window, Width]>(at)
            .to_dm(&mut device.tdma)
    }

    fn put(device: &mut Device, tile: &Tile, at: usize, rows: &mut Rows) {
        let row = rows
            .view_mut()
            .tile::<m![Window], 1, m![Window = 1 #{!} Window, Width]>(at);
        tile.view().to_hbm_view(&mut device.tdma, row);
    }

    /// The live rows of a launch holding at least one row, a tile a row: the first, and the rest.
    fn live(count: &Count) -> Launch {
        Live::all().subrange(0, 1 + count.extent::<m![Rest]>())
    }

    /// A two-layer stack of row-wise nodes, run by `Scheduled`.
    #[device(chip = 1)]
    pub fn stack(
        device: &mut Device,
        count: &Count,
        input: &Rows,
        a: &mut Rows,
        b: &mut Rows,
        c: &mut Rows,
        e: &mut Rows,
    ) {
        let mut state = (input, ((a, b), (c, e)));
        Scheduled(&STACK).forward(device, live(count), &mut state, ());
    }

    /// The same stack, each tile through every node's `forward` before the next tile arrives.
    #[device(chip = 1)]
    pub fn stack_forward(
        device: &mut Device,
        count: &Count,
        input: &Rows,
        mut a: &mut Rows,
        mut b: &mut Rows,
        mut c: &mut Rows,
        mut e: &mut Rows,
    ) {
        let live = live(count);
        for at in 0..live.rows() {
            let tile = live.subrange(at, 1).with(to_dm(device, input, at));
            let tile = Put.forward(device, live, &mut a, &tile);
            let tile = Put.forward(device, live, &mut b, &tile);
            let tile = Put.forward(device, live, &mut c, &tile);
            Put.forward(device, live, &mut e, &tile);
        }
    }

    /// A node that answers `Pending` first, polled by one with work that needs no input.
    #[device(chip = 1)]
    pub fn waits(
        device: &mut Device,
        count: &Count,
        input: &Rows,
        weights: &Rows,
        rows: &mut Rows,
        rest: &mut Rows,
    ) {
        let model = Sequential(Late, Warm);
        let mut state = (input, (weights, (rows, rest)));
        Scheduled(&model).forward(device, live(count), &mut state, ());
    }

    /// The order `waits` runs, written by hand: the weight is written on the first `Pending`,
    /// before any tile.
    #[device(chip = 1)]
    pub fn waits_by_hand(
        device: &mut Device,
        count: &Count,
        input: &Rows,
        weights: &Rows,
        rows: &mut Rows,
        rest: &mut Rows,
    ) {
        let live = live(count);
        let weight = to_dm(device, weights, 0);
        for at in 0..live.rows() {
            let tile = to_dm(device, input, at);
            put(device, &tile, at, rows);
        }
        put(device, &weight, 0, rest);
    }

    /// A whole leaf followed by a node with work that needs no input, run by `Scheduled`.
    #[device(chip = 1)]
    pub fn barrier(
        device: &mut Device,
        count: &Count,
        input: &Rows,
        weights: &Rows,
        rows: &mut Rows,
        rest: &mut Rows,
    ) {
        let model = Sequential(ToDm, Sequential(Gather, Warm));
        let mut state = (input, ((), (weights, (rows, rest))));
        Scheduled(&model).forward(device, live(count), &mut state, ());
    }

    /// The order `barrier` runs, written by hand: the weight is written on the first wait, right
    /// after the whole leaf takes the first tile.
    #[device(chip = 1)]
    pub fn barrier_by_hand(
        device: &mut Device,
        count: &Count,
        input: &Rows,
        weights: &Rows,
        rows: &mut Rows,
        rest: &mut Rows,
    ) {
        let live = live(count);
        let mut scratch = Rows::new();
        let tile = to_dm(device, input, 0);
        put(device, &tile, 0, &mut scratch);
        let weight = to_dm(device, weights, 0);
        for at in 1..live.rows() {
            let tile = to_dm(device, input, at);
            put(device, &tile, at, &mut scratch);
        }
        let mut copy = Rows::new();
        scratch
            .view()
            .to_hbm_view(&mut device.tdma, copy.view_mut());
        for at in 0..live.rows() {
            let tile = to_dm(device, &copy, at);
            put(device, &tile, at, rows);
        }
        put(device, &weight, 0, rest);
    }

    /// Three whole layers, two live at a time, between a copy and a store, run by `Scheduled`.
    #[device(chip = 1)]
    pub fn rolled(device: &mut Device, count: &Count, input: &Rows, out: &mut Rows) {
        let layers = Layers {
            layers: [Gather, Gather, Gather],
            kind: Rolled,
        };
        let mut state = (input, ([(); 3], out));
        Scheduled(&Sequential(ToDm, Sequential(layers, Put))).forward(
            device,
            live(count),
            &mut state,
            (),
        );
    }

    /// The same stack, each layer a view of a stack built when it starts.
    #[device(chip = 1)]
    pub fn rolled_views(device: &mut Device, count: &Count, input: &Rows, out: &mut Rows) {
        let layers = Layers {
            layers: Collected::<Stack, Gather, 3>(Stacked { storage: () }),
            kind: Rolled,
        };
        let mut state = (input, ([(); 3], out));
        Scheduled(&Sequential(ToDm, Sequential(layers, Put))).forward(
            device,
            live(count),
            &mut state,
            (),
        );
    }

    /// The order `rolled` runs, written by hand: each layer gives its tiles while the next takes
    /// them.
    #[device(chip = 1)]
    pub fn rolled_by_hand(device: &mut Device, count: &Count, input: &Rows, out: &mut Rows) {
        let live = live(count);
        let mut gathered = Rows::new();
        for at in 0..live.rows() {
            let tile = to_dm(device, input, at);
            put(device, &tile, at, &mut gathered);
        }
        let mut copy = Rows::new();
        gathered
            .view()
            .to_hbm_view(&mut device.tdma, copy.view_mut());
        for _ in 0..2 {
            let mut gathered = Rows::new();
            for at in 0..live.rows() {
                let tile = to_dm(device, &copy, at);
                put(device, &tile, at, &mut gathered);
            }
            gathered
                .view()
                .to_hbm_view(&mut device.tdma, copy.view_mut());
        }
        for at in 0..live.rows() {
            let tile = to_dm(device, &copy, at);
            put(device, &tile, at, out);
        }
    }

    /// Three source layers rolled two at a time: the second gives a tile while the first is live,
    /// breaking the stack's contract; its tiles pass on out of layer order. No test launches it.
    #[device(chip = 1)]
    pub fn rolled_rowwise(
        device: &mut Device,
        count: &Count,
        input: &Rows,
        a: &Rows,
        out: &mut Rows,
    ) {
        let layers = Layers {
            layers: [ToDm, ToDm, ToDm],
            kind: Rolled,
        };
        let mut state = (input, ([a; 3], out));
        Scheduled(&Sequential(ToDm, Sequential(layers, Put))).forward(
            device,
            live(count),
            &mut state,
            (),
        );
    }

    async fn rows(device: &mut Device, seed: usize) -> Rows {
        let values =
            (0..Window::SIZE * Width::SIZE).map(|i| bf16::from_f32(((i * seed) % 97) as f32));
        HostTensor::<bf16, m![Window, Width]>::from_vec(values)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap()
    }

    async fn read(device: &mut Device, rows: &Rows) -> Vec<bf16> {
        rows.to_host::<m![Window, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }

    /// Each scheduled entry stores what its order written by hand stores.
    #[tokio::test(flavor = "current_thread")]
    async fn stores_what_its_order_stores() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let count = HostTensor::<i32, m![1]>::from_vec([8])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let (input, weights) = (rows(&mut device, 3).await, rows(&mut device, 5).await);
        macro_rules! stack_stores {
            ($entry:ident) => {{
                let (mut a, mut b) = (rows(&mut device, 0).await, rows(&mut device, 0).await);
                let (mut c, mut e) = (rows(&mut device, 0).await, rows(&mut device, 0).await);
                let args = (&mut device, &count, &input, &mut a, &mut b, &mut c, &mut e);
                launch($entry, args).await.unwrap();
                let mut stored = Vec::new();
                for rows in [&a, &b, &c, &e] {
                    stored.push(read(&mut device, rows).await);
                }
                stored
            }};
        }
        assert_eq!(stack_stores!(stack), stack_stores!(stack_forward));
        macro_rules! kept_stores {
            ($entry:ident) => {{
                let (mut kept, mut rest) = (rows(&mut device, 0).await, rows(&mut device, 0).await);
                let args = (&mut device, &count, &input, &weights, &mut kept, &mut rest);
                launch($entry, args).await.unwrap();
                (
                    read(&mut device, &kept).await,
                    read(&mut device, &rest).await,
                )
            }};
        }
        assert_eq!(kept_stores!(barrier), kept_stores!(barrier_by_hand));
        assert_eq!(kept_stores!(waits), kept_stores!(waits_by_hand));
        macro_rules! rolled_stores {
            ($entry:ident) => {{
                let mut out = rows(&mut device, 0).await;
                launch($entry, (&mut device, &count, &input, &mut out))
                    .await
                    .unwrap();
                read(&mut device, &out).await
            }};
        }
        assert_eq!(rolled_stores!(rolled), rolled_stores!(rolled_by_hand));
        assert_eq!(rolled_stores!(rolled_views), rolled_stores!(rolled_by_hand));
    }

    /// The numbers a stream gives, polled until it is done; a stream of these tests ends within
    /// this many polls.
    pub(in crate::modules) fn drain(
        device: &mut Device,
        mut stream: impl Stream<Item = Number>,
    ) -> Vec<usize> {
        const POLLS: usize = 1000;
        let mut numbers = Vec::new();
        for _ in 0..POLLS {
            match stream.poll_next(device) {
                Poll::Ready(Some(number)) => numbers.push(number.into_value()),
                Poll::Ready(None) => return numbers,
                Poll::Pending => {}
            }
        }
        panic!("the stream did not end within {POLLS} polls");
    }

    /// Completion drains the input's final effect before a whole-input consumer ends.
    #[test]
    fn drains_front_input() {
        struct End<'a> {
            source: Indices,
            effect: &'a mut usize,
        }

        impl Stream for End<'_> {
            type Item = Number;

            fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Number>> {
                let polled = self.source.poll_next(device);
                if matches!(polled, Poll::Ready(None)) {
                    *self.effect += 1;
                }
                polled
            }
        }

        let _lease = DEVICE.blocking_lock();
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let live = Launch::over(1);
        let mut effect = 0;
        let source = End {
            source: Indices { live, given: 0 },
            effect: &mut effect,
        };
        let output = drain(&mut device, Collect.into_stream(live, &mut (), source));
        assert_eq!(output, vec![0]);
        assert_eq!(
            effect, 1,
            "the consumer must observe actual upstream completion"
        );
    }
}
