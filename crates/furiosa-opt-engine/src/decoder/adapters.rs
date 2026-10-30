//! The LoRA adapters a decoder's calls run their requests with.
//!
//! - An adapter is a PEFT checkpoint loaded into HBM as the decoder's adapter tree
//!   (`Decoder::Adapter`). Nothing changes it, so a call reads it where it lies: a framework makes
//!   an adapter active by putting it in a slot of the list it calls with, without a copy.
//! - A call holds at most [`SLOTS`] adapters, and each slot's update runs on every row: a slot no
//!   request names reads another slot's adapter, its selection zero on every row.
//! - A slot's selection of the call's rows, one on the rows of the requests that name the slot and
//!   zero on every other, is staged with the call's operands, once for each rank width.
//! - A `Served` decoder loads and reads its adapters as an [`Adapter`], the tree's type erased:
//!   an adapter belongs to the decoder's type on a device, not to one load of it.

use crate::{Checkpoint, LoadError};
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_std::prelude::*;
use std::any::Any;
use std::io;

/// Adapters one call runs at most: a policy, as each adds its update to every adapted projection of
/// every row whether a row names it or not.
pub const SLOTS: usize = 2;

/// An adapter a served decoder loaded, its tree's type erased: a call of a decoder of another type
/// refuses it.
pub struct Adapter(Box<dyn Any + Send + Sync>);

/// Loads a PEFT LoRA checkpoint onto a device as an adapter of one decoder type.
pub type Loader = fn(&Checkpoint, &mut Device) -> Result<Adapter, LoadError>;

impl Adapter {
    pub(super) fn new<A: Send + Sync + 'static>(tree: A) -> Self {
        Self(Box::new(tree))
    }

    /// The tree of a decoder whose adapters are `A`s; refused as an adapter of another model.
    pub(super) fn of<A: 'static>(&self) -> io::Result<&A> {
        self.0.downcast_ref::<A>().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "an adapter was loaded for another model",
            )
        })
    }
}

/// The adapters of a call that `Batch::new` checked, bound to the call: the adapter of each slot,
/// a slot the caller left empty holding the first adapter it holds, and each call row's slot.
pub struct Binding<'a, A> {
    held: [&'a A; SLOTS],
    rows: Vec<Option<usize>>,
}

impl<'a, A> Binding<'a, A> {
    /// The caller's list `adapters` bound to `rows`, the slot of each call row's adapter; none
    /// when the list holds no adapter or no row runs with one.
    pub(super) fn new(adapters: &[Option<&'a A>], rows: Vec<Option<usize>>) -> Option<Self> {
        let first = adapters.iter().flatten().next().copied()?;
        let held =
            std::array::from_fn(|slot| adapters.get(slot).copied().flatten().unwrap_or(first));
        rows.iter()
            .any(Option::is_some)
            .then_some(Self { held, rows })
    }

    /// The adapter of each slot.
    pub fn held(&self) -> [&'a A; SLOTS] {
        self.held
    }

    /// Each slot's selection of the call's rows as `[R, K]` BF16 values in I32 words: a row's `K`
    /// values are one when it runs with the slot's adapter and zero otherwise, and the rows past
    /// the call's are zero.
    pub fn selections<R: AxisName, K: AxisName>(&self) -> [(Vec<i32>, usize); SLOTS] {
        const { assert!((R::SIZE * K::SIZE).is_multiple_of(2)) };
        std::array::from_fn(|slot| {
            let values: Vec<bf16> = (0..R::SIZE)
                .flat_map(|row| {
                    let selected = self.rows.get(row).copied().flatten() == Some(slot);
                    std::iter::repeat_n(bf16::from_f32(f32::from(u8::from(selected))), K::SIZE)
                })
                .collect();
            let words: Vec<i32> = (values.as_chunks::<2>().0.iter())
                .map(|[low, high]| i32::from(low.to_bits()) | (i32::from(high.to_bits()) << 16))
                .collect();
            let capacity = words.len();
            (words, capacity)
        })
    }
}

/// The selections `Binding::selections` staged, a slot's view of `K` rank columns each, as the
/// call's next read arrays `next` gives them.
pub(super) fn selected<R: AxisName, K: AxisName>(
    mut next: impl FnMut() -> io::Result<HbmBuffer>,
) -> io::Result<[HbmTensor<bf16, m![1], m![R, K]>; SLOTS]> {
    let views = (0..SLOTS)
        .map(|_| alias(&next()?, ..))
        .collect::<io::Result<Vec<_>>>()?;
    Ok(views
        .try_into()
        .unwrap_or_else(|_| unreachable!("a view a slot")))
}
