//! One launch of a causal model whose layers interleave in two collections, each attending in its
//! own group over pages of its own cache: the input and state a launch passes it.
//!
//! - Each group's history is the call's requests' rows, cached positions and block tables in that
//!   group's cache; the group's paging prepares the rows' rotations, write slots and pages from it
//!   once, on the device.
//! - The launch writes the greedy token of each row it returns, read from its hidden rows at the
//!   byte offsets its input lists, in that order.
//! - A body's [`Collections`] are its two collections' layers by position in the model, which a
//!   launch's state holds a slot of each, in the collections' storage order.

use super::Tables;
use crate::models::decoder::Attention;
use crate::modules::attention::key_value;
use crate::modules::language::{self, Backbone, Normalized};
use crate::ops::embedding;
use crate::tensor::cache::Blocks;
use crate::tensor::parallel::Chip;
use crate::{Branch, Interleave};
use furiosa_opt_std::prelude::*;

/// A launch's history in group `A`'s cache, at the capacities of `R` rows that each span the
/// group's whole context.
pub type History<R, A> = (
    (),
    key_value::History<R, Tables<R, <A as Attention>::Windows>, R>,
);

/// A launch's input of `R` rows over the caches of groups `A` and `B`: its token IDs, its live
/// count, each group's history, then the byte offset among its hidden rows of each row it returns.
pub type Input<R, A, B> = language::Input<
    R,
    embedding::Ids<R>,
    (History<R, A>, History<R, B>, HbmTensor<i32, Chip, m![R]>),
>;

/// A layer's slot over group `A`'s cache: its blocks, which it writes and reads, then what its
/// leaves run with in the call, nothing.
pub type Layer<'b, A> = (
    &'b mut Blocks<<A as Attention>::Heads, <A as Attention>::Head>,
    (),
);

/// A launch's state of `R` rows: the slots of group `A`'s `FIRST` layers and of group `B`'s
/// `SECOND`, then the place the tokens of the rows it returns are written in.
pub type State<'s, 'b, R, A, B, const FIRST: usize, const SECOND: usize> = (
    ([Layer<'b, A>; FIRST], [Layer<'b, B>; SECOND]),
    &'s mut HbmTensor<i32, Chip, m![R]>,
);

/// A base model's body whose layers interleave in two collections: each collection's layers, by
/// position in the model, and a launch's slots of them.
pub trait Collections {
    /// A slot `P` for each of the first collection's layers, then `Q` for each of the second's.
    type Slots<P, Q>;

    /// The first collection's layers' positions in the model, then the second's, each in storage
    /// order.
    fn positions() -> (Vec<usize>, Vec<usize>);

    /// The slots `first` and `second` give, a slot for each layer; `None` unless each lists one
    /// for each of its collection's layers.
    fn slots<P, Q>(first: Vec<P>, second: Vec<Q>) -> Option<Self::Slots<P, Q>>;
}

/// The embedding, the collections in their order, then the final norm.
impl<E, X, Y, N, const FIRST: usize, const SECOND: usize, const ORDER: &'static [Branch]>
    Collections for Normalized<Backbone<E, Interleave<([X; FIRST], [Y; SECOND]), ORDER>>, N>
{
    type Slots<P, Q> = ([P; FIRST], [Q; SECOND]);

    fn positions() -> (Vec<usize>, Vec<usize>) {
        let (first, second) = Interleave::<([X; FIRST], [Y; SECOND]), ORDER>::POSITIONS;
        (first.to_vec(), second.to_vec())
    }

    fn slots<P, Q>(first: Vec<P>, second: Vec<Q>) -> Option<([P; FIRST], [Q; SECOND])> {
        Some((first.try_into().ok()?, second.try_into().ok()?))
    }
}
