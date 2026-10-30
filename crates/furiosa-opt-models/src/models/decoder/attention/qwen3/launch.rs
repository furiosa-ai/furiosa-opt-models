//! One launch of a size's step: the causal model it runs in form `P`, the input, operands and
//! state a launch passes it, and the context its rotary table prepares from them once.
//!
//! - The model's rotary table prepares a launch's [`Context`] from its [`Operands`] once (the
//!   work's counts read, the rows' rotation gathered); the body and head then stream over spans of
//!   the live rows, each layer on its own cache blocks, and the head reads the selected rows'
//!   tokens out tile by tile.
//! - The envelope a launch is compiled for is the size's [`Decode`].

use super::{CatalogBody, Groups, Hidden, LmHead};
use crate::models::decoder::attention::{
    ForCausalLM, ForSequenceClassification, Geometry, Head, KvHeads,
};
use crate::modules::attention::{self, key_value};
use crate::modules::language::{self, Call, Selected};
use crate::ops::attention::full::paged::{
    self,
    chunks::{Cache, Chunks as Windows},
};
use crate::ops::embedding;
use crate::ops::linear::affine::Columns;
use crate::ops::normalization::Blocked;
use crate::ops::rotary::{
    self, SplitHalf,
    group::{self, Table},
};
use crate::specialize::attention::{Decode, WAVES};
use crate::tensor::cache::Blocks;
use crate::tensor::memory::Hbm;
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;

/// The rows of a launch of size `S`; a runtime count of them is live.
pub type Capacity<S> = <S as Decode>::Capacity;

pub type Work<S> = paged::Work<<S as Decode>::Sequences, KvHeads, Groups<S>, Head, WAVES>;
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
    WAVES,
>;

/// The rotation of a launch's rows, gathered from the positions' table it loads from the config.
type Position<S> = SplitHalf<Head, Table<<S as Decode>::Positions, rotary::Row<Head>>>;

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
/// reads, and its output projection's), then its MLP's, which holds none.
pub type Layer<'b> = ((Front, &'b mut Blocks<KvHeads, Head>, ()), ());

/// A launch's body's state: the hidden rows the embedding writes and each layer's slot, then the
/// final norm's, which holds none.
pub type Body<'s, 'b, S> = &'s mut (
    (
        HbmTensor<bf16, Chip, Blocked<Hidden<S>, Capacity<S>>>,
        <S as Geometry>::Layers<Layer<'b>>,
    ),
    (),
);

/// A launch's state: its body's, then the place `O` its head writes the selected rows' outputs in,
/// their tokens by default.
pub type State<'s, 'b, S, O = Tokens<S>> = &'s mut (
    (
        (
            HbmTensor<bf16, Chip, Blocked<Hidden<S>, Capacity<S>>>,
            <S as Geometry>::Layers<Layer<'b>>,
        ),
        (),
    ),
    O,
);

/// What a launch's nodes read beside its token IDs and live rows, prepared once before its first
/// node runs: its rows' rotation and cache slots, its attention work and the rows it returns.
pub type Context<'a, Capacity, Reach, Sequences, Slots, Groups> = (
    attention::Context<
        SplitHalf<Head, group::Rotation<Slots, rotary::Row<Head>>>,
        &'a HbmTensor<i32, Chip, m![Reach]>,
        paged::Context<'a, Sequences, KvHeads, Groups, Head, WAVES, Capacity>,
    >,
    &'a Selected<Capacity>,
);

/// The rows the readout reads tokens of.
impl<'c, T, Capacity, Reach, Sequences, Slots, Groups>
    From<Call<'c, Context<'_, Capacity, Reach, Sequences, Slots, Groups>, T, Capacity>>
    for &'c Selected<Capacity>
where
    Capacity: AxisName,
    Reach: AxisName,
    Sequences: AxisName,
    Slots: AxisName,
    Groups: AxisName,
{
    fn from(
        call: Call<'c, Context<'_, Capacity, Reach, Sequences, Slots, Groups>, T, Capacity>,
    ) -> Self {
        call.context.1
    }
}

/// Positions of the longest context a launch's tables cover.
pub const CONTEXT: usize = Windows::SIZE * Cache::SIZE;

#[cfg(test)]
mod tests {
    use super::super::Size;
    use super::*;
    use crate::load::{Checkpoint, Load};
    use crate::models::decoder::form::{Affine, BlockScaled};

    /// Each size's BF16 catalog model runs its whole tree on its launch's input and state.
    #[test]
    fn runs_a_launch() {
        macro_rules! runs {
            ($($size:literal),+) => {$({
                type S = Size<$size>;
                fn runs<'i, 'a: 'i, 's, 'b: 's>()
                where
                    Model<S, Affine>: crate::Module<&'i mut Input<'a, S>, (), State<'s, 'b, S>>,
                {
                }
                runs();
            })+};
        }
        runs!("8B");
    }

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
