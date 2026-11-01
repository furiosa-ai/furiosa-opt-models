//! Low-rank adaptation (LoRA): a projection's output plus the update `B (A x)` of the adapter a
//! row runs with, read from adapters the caller holds.
//!
//! - One adapter's update of one projection is a [`Term`]: PEFT's `lora_A`, `[rank, inputs]`, and
//!   `lora_B`, `[outputs, rank]`, each a linear. [`load`] applies a PEFT checkpoint's configuration
//!   once, the scale into `lora_B`, and builds every term on host, the rank zero-padded; a tree in
//!   HBM loads from it, new or into the storage of one it holds.
//! - In a call that runs adapters, a projection's state is its [`Adapters`]: each adapter's term and
//!   its selection of the call's rows, one on the rows that run with it and zero on every other. The
//!   projection adds each adapter's update of the rows it selects to its own output; a call without
//!   adapters keeps the state `()`.
//! - [`GateUp`] is the update of a feed-forward's packed gate and up projections.
//! - A projection adds each adapter's update to the rows it writes in HBM ([`Adapters::add`]); a
//!   layer's [`Layer`] holds each of its projections' adapters.
//!
//! Numerics: `lora_A x` rounds to BF16, the selection keeps or zeroes it exactly, `lora_B` of it
//! rounds to BF16, and its sum with the projection's BF16 output rounds once.

use std::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use crate::ops::linear::Affine;
use crate::ops::pointwise::{add::Add, multiply::Multiply};
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;

pub use peft::Config;
pub use staged::{Staged, load};

mod peft;
mod staged;

/// The rank a term holds for each module it adapts: an adapter of a lower rank is zero-padded, and
/// a contraction part holds at least 32 inputs (`specialize::contraction`).
pub const RANK: usize = 32;

/// The rank columns of a term adapting `N` modules at once: one, or the gate and up projections a
/// packed projection computes.
#[derive(Clone, Debug)]
pub struct Rank<const N: usize>;

impl<const N: usize> AxisName for Rank<N> {
    const NAME: Ident = Ident::new("Rank");
    const SIZE: usize = N * RANK;
}

/// One adapter's update of a projection, `lora_B (lora_A x)`: `A` holds `[Rank, In]` and `B`
/// `[Out, Rank]`, the adapter's scale folded in.
#[derive(DeviceSend)]
pub struct Term<A, B> {
    pub lora_a: Affine<A, ()>,
    pub lora_b: Affine<B, ()>,
}

/// The `N` adapters of a call as one projection's state: each adapter's term, and its selection of
/// the call's rows, `[rows, rank]` ones on the rows that run with it and zeros on every other.
#[derive(DeviceSend)]
pub struct Adapters<'a, T, S, const N: usize> {
    pub terms: [&'a T; N],
    pub selections: &'a [S; N],
}

/// Written out, as a derive would ask the term and the selection to be `Copy` too.
impl<T, S, const N: usize> Clone for Adapters<'_, T, S, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, S, const N: usize> Copy for Adapters<'_, T, S, N> {}

/// The update of a feed-forward's projection computing its gate and up rows at once, packed as
/// the gate's rows then the up's: its rank holds the gate's, then the up's, and each output row
/// reads only its own part's.
#[derive(DeviceSend)]
pub struct GateUp<T>(T);

impl<T> GateUp<T> {
    /// The term, in the packed projection's output order.
    pub fn term(&self) -> &T {
        &self.0
    }
}

/// The adapters each of a layer's projections runs with in a call: its query, key, value and
/// output projections' and its feed-forward's packed gate/up and down projections'.
#[derive(Clone, Copy, DeviceSend)]
pub struct Layer<Q, K, V, O, G, D> {
    pub query: Q,
    pub key: K,
    pub value: V,
    pub output: O,
    pub gate_up: G,
    pub down: D,
}

/// The states of a layer's projections in a call: `()` for every one when the call runs no
/// adapter, else a [`Layer`] of their adapters.
pub trait States: Copy {
    type Query;
    type Key;
    type Value;
    type Output;
    type GateUp;
    type Down;

    fn query(&mut self) -> &mut Self::Query;
    fn key(&mut self) -> &mut Self::Key;
    fn value(&mut self) -> &mut Self::Value;
    fn output(&mut self) -> &mut Self::Output;
    fn gate_up(&mut self) -> &mut Self::GateUp;
    fn down(&mut self) -> &mut Self::Down;
}

/// A call without adapters: every projection's state is `()`.
impl States for () {
    type Query = ();
    type Key = ();
    type Value = ();
    type Output = ();
    type GateUp = ();
    type Down = ();

    fn query(&mut self) -> &mut () {
        self
    }

    fn key(&mut self) -> &mut () {
        self
    }

    fn value(&mut self) -> &mut () {
        self
    }

    fn output(&mut self) -> &mut () {
        self
    }

    fn gate_up(&mut self) -> &mut () {
        self
    }

    fn down(&mut self) -> &mut () {
        self
    }
}

impl<Q: Copy, K: Copy, V: Copy, O: Copy, G: Copy, D: Copy> States for Layer<Q, K, V, O, G, D> {
    type Query = Q;
    type Key = K;
    type Value = V;
    type Output = O;
    type GateUp = G;
    type Down = D;

    fn query(&mut self) -> &mut Q {
        &mut self.query
    }

    fn key(&mut self) -> &mut K {
        &mut self.key
    }

    fn value(&mut self) -> &mut V {
        &mut self.value
    }

    fn output(&mut self) -> &mut O {
        &mut self.output
    }

    fn gate_up(&mut self) -> &mut G {
        &mut self.gate_up
    }

    fn down(&mut self) -> &mut D {
        &mut self.down
    }
}

/// The windows of `Window` rows of `All` rows.
#[derive(Clone, Debug)]
struct Windows<All, Window>(PhantomData<(All, Window)>);

impl<All: AxisName, Window: AxisName> AxisName for Windows<All, Window> {
    const NAME: Ident = Ident::new("Windows");
    const SIZE: usize = All::SIZE / Window::SIZE;
}

impl<'a, A, B, K: AxisName, All: AxisName, const N: usize>
    Adapters<'a, Term<A, B>, HbmTensor<bf16, Chip, m![All, K]>, N>
{
    /// `lines`, a window of rows a projection wrote from launch row `start` on, the projection of
    /// the live rows `live` of `input`, plus each adapter's update of those rows; `start` is a
    /// multiple of the window's rows.
    pub(crate) fn add<'i, Rows, Window, Out, I>(
        &self,
        device: &mut Device,
        live: Live<(), Rows>,
        input: &'i I,
        start: usize,
        mut lines: HbmTensor<bf16, Chip, m![Window, Out]>,
    ) -> HbmTensor<bf16, Chip, m![Window, Out]>
    where
        Rows: AxisName,
        Window: AxisName,
        Out: AxisName,
        Affine<A, ()>: for<'o> crate::Op<
                (
                    Live<(), Rows>,
                    &'i I,
                    &'o mut HbmTensor<bf16, Chip, m![Window, K]>,
                ),
                Output = (),
            >,
        Affine<B, ()>: for<'s, 'o> crate::Op<
                (
                    Live<(), Window>,
                    &'s HbmTensor<bf16, Chip, m![Window, K]>,
                    &'o mut HbmTensor<bf16, Chip, m![Window, Out]>,
                ),
                Output = (),
            >,
    {
        type Spans<All, Window> = Symbol<Windows<All, Window>>;
        let rows = Live::<(), Window>::all().subrange(0, live.rows());
        #[unroll]
        for slot in 0..N {
            let term = self.terms[slot];
            let mut shrunk = HbmTensor::new();
            crate::Op::forward(&term.lora_a, &mut *device, (live, input, &mut shrunk));
            // SAFETY: a window's rows of a selection are contiguous, and no launch writes one.
            let selection: HbmTensor<bf16, Chip, m![Window, K]> = unsafe {
                self.selections[slot]
                    .view()
                    .reshape::<Chip, m![{ Spans<All, Window> }, Window, K]>()
                    .tile::<m![{ Spans<All, Window> }], 1, m![
                        { Spans<All, Window> } = 1 # ({ Spans<All, Window> }),
                        Window,
                        K
                    ]>(start / Window::SIZE)
                    .alias()
            };
            let selected = crate::Op::forward(&Multiply, &mut *device, (&shrunk, &selection));
            let mut update = HbmTensor::new();
            crate::Op::forward(&term.lora_b, &mut *device, (rows, &selected, &mut update));
            lines = crate::Op::forward(&Add, &mut *device, (&lines, &update));
        }
        lines
    }
}
