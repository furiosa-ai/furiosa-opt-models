//! One launch of a size's step: the causal model it runs in form `P`, the input, operands and
//! state a launch passes it, and the context its rotary table prepares from them once.
//!
//! - The model's rotary table prepares a launch's [`Context`] from its [`Operands`] once (the
//!   work's counts read, the rows' rotation gathered); the body and head then stream over spans of
//!   the live rows, each layer on its own cache blocks, and the head reads the selected rows'
//!   tokens out tile by tile.
//! - The envelope a launch is compiled for is the size's [`Decode`]: its rows, and the context its
//!   rotary table, prompt chunks and work items cover, its windows' [`Positions`].

use super::{CatalogBody, Groups, Hidden, LmHead, QueryHeads, tiles};
use crate::load::Named;
use crate::models::decoder::attention::{
    ForCausalLM, ForSequenceClassification, Geometry, Head, KvHeads,
};
use crate::modules::attention::{self, key_value};
use crate::modules::language::{self, Call, Selected};
use crate::modules::lora::{self, Rank};
use crate::ops::attention::full::paged;
use crate::ops::attention::gqa::Cache;
use crate::ops::embedding;
use crate::ops::linear::affine::Columns;
use crate::ops::linear::query::matrix::Width;
use crate::ops::normalization::Blocked;
use crate::ops::rotary::{
    self, SplitHalf,
    group::{self, Table},
};
use crate::ops::swiglu::GateUp;
use crate::specialize::attention::Decode;
use crate::tensor::cache::Blocks;
use crate::tensor::memory::{Hbm, Memory};
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
use std::fmt;
use std::marker::PhantomData;

/// The rows of a launch of size `S`; a runtime count of them is live.
pub type Capacity<S> = <S as Decode>::Capacity;

/// The positions a launch of size `S` covers: the context its size is compiled for, the keys of
/// its windows.
pub struct Positions<S>(PhantomData<fn() -> S>);

impl<S: Decode + 'static> AxisName for Positions<S> {
    const NAME: Ident = Ident::new("Positions");
    const SIZE: usize = <S as Decode>::Windows::SIZE * Cache::SIZE;
}

/// The wide waves of a launch of size `S`: those its size's rows take when each attends the whole
/// context under either tiling.
pub struct Waves<S>(PhantomData<fn() -> S>);

impl<S: Geometry + Decode + 'static> AxisName for Waves<S> {
    const NAME: Ident = Ident::new("Waves");
    const SIZE: usize = tiles::<S>();
}

impl<S> Clone for Positions<S> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<S> fmt::Debug for Positions<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Positions")
    }
}

impl<S> Clone for Waves<S> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<S> fmt::Debug for Waves<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Waves")
    }
}

pub type Work<S> = paged::Work<
    <S as Decode>::Sequences,
    KvHeads,
    Groups<S>,
    Head,
    Waves<S>,
    <S as Decode>::Windows,
>;
pub type Ids<S> = embedding::Ids<Capacity<S>>;

/// A launch's next tokens, one I32 for each selected row in order; rows past them are unspecified.
pub type Tokens<S> = HbmTensor<i32, Chip, m![{ Symbol<Capacity<S>> }]>;

/// A launch's hidden values, a row for each selected row in order; rows past them are unspecified.
pub type Values<S> = HbmTensor<bf16, Chip, m![{ Symbol<Capacity<S>> }, { Symbol<Hidden<S>> }]>;

/// A launch's scores, label columns first, a row for each selected row in order; rows past them
/// are unspecified.
pub type Scores<S> = HbmTensor<f32, Chip, m![{ Symbol<Capacity<S>> }, { Symbol<Columns> }]>;

/// A launch's input: its token IDs, live count and the operands its preparation reads.
pub type Input<'a, S> = language::Input<Capacity<S>, Ids<S>, &'a Operands<S>>;

/// A launch's operands beside its token IDs and live count: its rows, its attention work and the
/// rows it returns; its rows' rotation is gathered in tiles of the size's `Slots`.
pub type Operands<S> = key_value::Operands<
    Capacity<S>,
    <S as Decode>::Reach,
    <S as Decode>::Sequences,
    <S as Decode>::Slots,
    KvHeads,
    Groups<S>,
    Head,
    Waves<S>,
    <S as Decode>::Windows,
>;

/// The rotation of a launch's rows, gathered from the positions' table it loads from the config.
type Position<S> = SplitHalf<Head, Table<Positions<S>, rotary::Row<Head>>>;

/// The base model over the body `B`, with the rotary table its launches gather their rows'
/// rotation from.
// The decoder tree's `Model` is named by its path: this module's `Model` is the catalog's.
pub type Over<S, B, P> = crate::models::decoder::attention::Model<S, B, Position<S>, P>;

/// The catalog's base model of size `S` in form `P`.
pub type Base<S, P> = Over<S, CatalogBody<S, P, bf16, Hbm>, P>;

/// The catalog's causal model of size `S` in form `P`: the base model and the greedy head.
pub type Model<S, P> = ForCausalLM<S, Base<S, P>, LmHead<S, Hbm>, P>;

/// The catalog's sequence classifier of size `S` in form `P`: the base model and its score head.
pub type Classifier<S, P> =
    ForSequenceClassification<S, Base<S, P>, language::Classifier<Hidden<S>>, P>;

/// The slots of an attention's front: its projections', then its norms'; none holds anything.
type Front = (((), (), ()), (), ());

/// A layer's state slot: its attention's (its front's, its cache blocks, which it writes and
/// reads, and its output projection's), then what its projections run with in the call, `A`:
/// their adapters, none by default.
pub type Layer<'b, A = ()> = ((Front, &'b mut Blocks<KvHeads, Head>, ()), A);

/// A launch's body's state: the hidden rows the embedding writes and the layers' slots `L`, a
/// [`Layer`] each, then the final norm's, which holds none.
pub type Body<'s, S, L> = &'s mut (
    (HbmTensor<bf16, Chip, Blocked<Hidden<S>, Capacity<S>>>, L),
    (),
);

/// A launch's state: its body's, then the place `O` its head writes the selected rows' outputs in,
/// their tokens by default.
pub type State<'s, S, L, O = Tokens<S>> = &'s mut (
    (
        (HbmTensor<bf16, Chip, Blocked<Hidden<S>, Capacity<S>>>, L),
        (),
    ),
    O,
);

/// A LoRA adapter of the catalog's size `S` as PEFT saves it, over memory `X`: each layer's
/// updates of its query, key, value and output projections and of its feed-forward's packed
/// gate/up and down projections.
pub type Adapter<S, X> = Named<
    "base_model",
    Named<"model", Named<"model", Named<"layers", <S as Geometry>::Layers<Terms<S, X>>>>>,
>;

/// One layer's terms of an [`Adapter`].
pub type Terms<S, X> = (
    Named<
        "self_attn",
        (
            (
                Named<"q_proj", Factors<Rank<1>, Hidden<S>, Width<QueryHeads<S>, Head>, X>>,
                Named<"k_proj", Factors<Rank<1>, Hidden<S>, Width<KvHeads, Head>, X>>,
                Named<"v_proj", Factors<Rank<1>, Hidden<S>, Width<KvHeads, Head>, X>>,
            ),
            Named<"o_proj", Factors<Rank<1>, <S as Geometry>::Queries, Hidden<S>, X>>,
        ),
    >,
    Named<
        "mlp",
        (
            lora::GateUp<Factors<Rank<2>, Hidden<S>, GateUp<Intermediate<S>>, X>>,
            Named<"down_proj", Factors<Rank<1>, Intermediate<S>, Hidden<S>, X>>,
        ),
    >,
);

/// An update's `[K, In]` and `[Out, K]` factors in memory `X`.
type Factors<K, In, Out, X> = lora::Term<
    <X as Memory>::Tensor<bf16, Chip, m![K, In]>,
    <X as Memory>::Tensor<bf16, Chip, m![Out, K]>,
>;

type Intermediate<S> = <S as Geometry>::Intermediate;

/// A selection of a launch's rows over `K` rank columns: one on the rows that run with its
/// adapter, zero on every other.
pub type Selection<S, K> = HbmTensor<bf16, Chip, m![{ Symbol<Capacity<S>> }, K]>;

/// The selections of a call's `N` adapters, over one term's rank columns, then over the packed
/// gate/up term's.
pub type Selections<S, const N: usize> = ([Selection<S, Rank<1>>; N], [Selection<S, Rank<2>>; N]);

/// What each layer's projections run with in a call of `N` adapters of the size `S`: each
/// adapter's term of the projection, and its selection of the launch's rows.
pub type Adapted<'t, S, const N: usize> = lora::Layer<
    Projected<'t, S, Width<QueryHeads<S>, Head>, Hidden<S>, Rank<1>, N>,
    Projected<'t, S, Width<KvHeads, Head>, Hidden<S>, Rank<1>, N>,
    Projected<'t, S, Width<KvHeads, Head>, Hidden<S>, Rank<1>, N>,
    Projected<'t, S, Hidden<S>, <S as Geometry>::Queries, Rank<1>, N>,
    Projected<'t, S, GateUp<Intermediate<S>>, Hidden<S>, Rank<2>, N>,
    Projected<'t, S, Hidden<S>, Intermediate<S>, Rank<1>, N>,
>;

/// A projection's adapters from `In` to `Out` features over `K` rank columns.
type Projected<'t, S, Out, In, K, const N: usize> =
    lora::Adapters<'t, Factors<K, In, Out, Hbm>, Selection<S, K>, N>;

/// Layer `layer`'s adapters of `held`, a call's adapters, with their `selections`.
pub fn adapted<'t, S: Geometry + Decode + 'static, const N: usize>(
    held: [&'t Adapter<S, Hbm>; N],
    layer: usize,
    (one, two): &'t Selections<S, N>,
) -> Adapted<'t, S, N>
where
    <S as Geometry>::Layers<Terms<S, Hbm>>: AsRef<[Terms<S, Hbm>]>,
{
    let terms = held.map(|adapter| &adapter.0.0.0.0.as_ref()[layer]);
    let (attention, mlp) = (terms.map(|t| &t.0.0), terms.map(|t| &t.1.0));
    lora::Layer {
        query: lora::Adapters {
            terms: attention.map(|t| &t.0.0.0),
            selections: one,
        },
        key: lora::Adapters {
            terms: attention.map(|t| &t.0.1.0),
            selections: one,
        },
        value: lora::Adapters {
            terms: attention.map(|t| &t.0.2.0),
            selections: one,
        },
        output: lora::Adapters {
            terms: attention.map(|t| &t.1.0),
            selections: one,
        },
        gate_up: lora::Adapters {
            terms: mlp.map(|t| t.0.term()),
            selections: two,
        },
        down: lora::Adapters {
            terms: mlp.map(|t| &t.1.0),
            selections: one,
        },
    }
}

/// What a launch's nodes read beside its token IDs and live rows, prepared once before its first
/// node runs: its rows' rotation and cache slots, its attention work and the rows it returns.
pub type Context<'a, Capacity, Reach, Sequences, Slots, Groups, Waves, W> = (
    attention::Context<
        SplitHalf<Head, group::Rotation<Slots, rotary::Row<Head>>>,
        &'a HbmTensor<i32, Chip, m![Reach]>,
        paged::Context<'a, Sequences, KvHeads, Groups, Head, Waves, W, Capacity>,
    >,
    &'a Selected<Capacity>,
);

/// The rows the readout reads tokens of.
impl<'c, T, Capacity, Reach, Sequences, Slots, Groups, Waves, W>
    From<Call<'c, Context<'_, Capacity, Reach, Sequences, Slots, Groups, Waves, W>, T, Capacity>>
    for &'c Selected<Capacity>
where
    Capacity: AxisName,
    Reach: AxisName,
    Sequences: AxisName,
    Slots: AxisName,
    Groups: AxisName,
    Waves: AxisName,
    W: AxisName,
{
    fn from(
        call: Call<
            'c,
            Context<'_, Capacity, Reach, Sequences, Slots, Groups, Waves, W>,
            T,
            Capacity,
        >,
    ) -> Self {
        call.context.1
    }
}

#[cfg(test)]
mod tests {
    use super::super::Size;
    use super::*;
    use crate::load::{Checkpoint, Load};
    use crate::models::decoder::form::{Affine, BlockScaled};

    /// Each size's catalog model loads from a checkpoint in dense BF16, and Qwen3-8B also in
    /// block-scaled FP8.
    #[test]
    fn loads_from_a_checkpoint() {
        fn loads<T: for<'a> Load<&'a Checkpoint>>() {}
        loads::<Model<Size<"8B">, BlockScaled>>();
        loads::<Model<Size<"8B">, Affine>>();
        loads::<Model<Size<"4B">, Affine>>();
        loads::<Model<Size<"1.7B">, Affine>>();
        loads::<Model<Size<"0.6B">, Affine>>();
    }
}
