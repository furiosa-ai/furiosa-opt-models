//! Live execution of the checkpoint-named joint attention tree.
//!
//! The family connects existing numerical operations without changing their arithmetic or
//! weight ownership. Request preparation bounds text capacity and derives all consumer plans
//! together; the context binds those readers to the matching positions and visibility.

use super::embeddings::Chips;
use super::packing::{PARTS, ROW_BLOCK};
use super::request::{BlockCall, Prefix, Transfer};
use super::transformer::{Heads, SCALE};
use crate::load::Named;
use crate::modules::attention::Full;
use crate::modules::parallel::Exchanged;
use crate::ops::attention::{flash, full::PackKeys};
use crate::ops::collective::Pieces;
use crate::ops::linear::{Grouped, grouped::Layout};
use crate::ops::rotary::{Rotary, Rotation};
use crate::tensor::axis::Rows;
use crate::tensor::live::{BlockPrefix, Live};
use crate::tensor::parallel::{Parallel, Part};
use crate::{Module, Op};
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;

type Local<R, W> = HbmTensor<bf16, Chips, m![R, W]>;
type Joined<R> = Pieces<R, PARTS>;
type Split<R, A> = Local<Joined<R>, Part<A, PARTS>>;

/// Readers for one request, decoded once before the layer stack. Positions describe the compact
/// joined query order, including initialized work padding; visibility describes the packed valid
/// key prefix. All three operands must come from the same prepared request. The tensor's Live
/// interval must start at zero and equal `call.prefix()`; the surrounding adaptive block and
/// its branches retain that interval. Allocation tails are never part of initialized work.
pub struct Context<'c, 'a, R: AxisName, K: AxisName> {
    call: &'c BlockCall<'a, R, K>,
    rotation: &'c Rotation<flash::Head, Chips, Joined<R>>,
    visible: &'c flash::Visible<K, Chips>,
}

impl<'c, 'a, R: AxisName, K: AxisName> Context<'c, 'a, R, K> {
    pub fn new(
        call: &'c BlockCall<'a, R, K>,
        rotation: &'c Rotation<flash::Head, Chips, Joined<R>>,
        visible: &'c flash::Visible<K, Chips>,
    ) -> Self {
        Self {
            call,
            rotation,
            visible,
        }
    }

    /// The same prefix drives both the attention and the feed-forward branches.
    pub fn environment(&self) -> (&Prefix<R>, &Self) {
        (self.call.prefix(), self)
    }
}

impl<'i, 'p, 'e, 'c, 'a, 's, R, K, W, A, Qkv, QkvState, Storage>
    Module<
        &'i Live<Local<R, W>, R>,
        (&'p Prefix<R>, &'e Context<'c, 'a, R, K>),
        &'s mut (QkvState, ()),
    >
    for Full<
        Qkv,
        flash::Compact<Heads<Part<A, PARTS>>, SCALE>,
        Named<"out_proj", Exchanged<Grouped<A, W, Chips, Storage>, Parallel<PARTS>>>,
    >
where
    R: AxisName,
    K: AxisName,
    W: AxisName,
    A: AxisName,
    Storage: Borrow<HbmTensor<bf16, Chips, Layout<A, W>>>,
    Qkv: for<'x, 'v> Module<
            &'v BlockPrefix<&'x Local<R, W>, R, Rows<ROW_BLOCK>>,
            (),
            &'s mut QkvState,
            Output = Live<(Split<R, A>, Split<R, A>, Split<R, A>), Joined<R>>,
        >,
{
    type Output = Live<Local<R, W>, R>;

    fn forward(
        &self,
        device: &mut Device,
        (prefix, context): (&'p Prefix<R>, &'e Context<'c, 'a, R, K>),
        (qkv_state, output_state): &'s mut (QkvState, ()),
        input: &'i Live<Local<R, W>, R>,
    ) -> Self::Output {
        const {
            // Compact projection initializes only joined work. Both following tiled consumers
            // must end on that boundary, rather than read an uninitialized allocation suffix.
            assert!(
                (PARTS * ROW_BLOCK).is_multiple_of(
                    crate::ops::attention::full::heads::Block::<
                        Joined<R>,
                        Part<A, PARTS>,
                        flash::Head,
                    >::SIZE
                )
            );
            assert!((PARTS * ROW_BLOCK).is_multiple_of(
                crate::ops::attention::full::heads::tiles::Strip::<Part<A, PARTS>, flash::Head>::SIZE
            ));
        }
        let projected = self
            .qkv
            .forward(device, (), qkv_state, &prefix.with(input.value()));
        let rows = projected.with(());
        let (query, key, value) = projected.into_value();
        let rotated = Op::forward(
            &Rotary::<flash::Head, flash::Head, bf16, Chips>::new(),
            device,
            (rows.with((&query, &key)), context.rotation),
        );
        let (query, key) = rotated.into_value();
        let pack = PackKeys::<K>::default();
        let keys = Op::forward(&pack, device, (rows.with(&key), context.call.keys()));
        let values = Op::forward(&pack, device, (rows.with(&value), context.call.keys()));
        // SAFETY: the contiguous local attended width is pairs of consecutive 128-wide heads.
        // Splitting only that axis preserves the compact row order and allocation capacity.
        let query: flash::Queries<Joined<R>, flash::HeadPairs<Part<A, PARTS>>, Chips> =
            unsafe { query.reshape() };
        let attended = Op::forward(
            &self.attend,
            device,
            (
                rows.with(&query),
                &keys,
                &values,
                (context.visible, context.call.key_count()),
            ),
        );
        // SAFETY: inverse of the query view above; every row's head pairs remain consecutive.
        let attended: Split<R, A> = unsafe { attended.into_value().reshape() };
        let input = prefix.with(&attended);
        // BlockPlan refuses capacities beyond the compiled maximum. Its positive text count
        // therefore derives exactly one of these work extents, with no rounding or fallback
        // to a larger text bucket. The final branch is the third checked whole-transfer shape.
        if prefix.rows() == Transfer::<0>::SIZE {
            self.output
                .0
                .prefix::<Transfer<0>>()
                .forward(device, (), output_state, &input)
        } else if prefix.rows() == Transfer::<1>::SIZE {
            self.output
                .0
                .prefix::<Transfer<1>>()
                .forward(device, (), output_state, &input)
        } else {
            self.output
                .0
                .prefix::<Transfer<2>>()
                .forward(device, (), output_state, &input)
        }
    }
}
