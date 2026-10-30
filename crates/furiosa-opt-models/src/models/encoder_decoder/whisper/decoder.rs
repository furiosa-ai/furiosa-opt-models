//! Decoder history is writable; encoded source projections become read-only after creation.

use super::{Ids, Mask};
use crate::{
    Module,
    modules::attention,
    ops::index::{gather::RowStorage, table::KeyValue},
};
use furiosa_opt_std::prelude::*;

/// The decoder over its step's masks: the self attention reads the history mask, the cross
/// attention the source mask beside this call's memory access.
#[derive(DeviceSend)]
pub struct Body<Function>(pub Function);

/// A composition, not an op.
impl<I, Function> !crate::Op<I> for Body<Function> {}

/// The env is the self mask and the source mask, then this call's memory access, a write of the
/// encoded source or a read; the state is the function's slot.
impl<'e, I, A: Copy, FunctionState, Function, Q: AxisName, K: AxisName, S: AxisName, O>
    Module<I, ((&'e Mask<Q, K>, &'e Mask<Q, S>), A), FunctionState> for Body<Function>
where
    Function: Module<I, ((&'e (), &'e Mask<Q, K>), (&'e Mask<Q, S>, A)), FunctionState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        ((mask, cross_mask), access): ((&'e Mask<Q, K>, &'e Mask<Q, S>), A),
        state: FunctionState,
        input: I,
    ) -> O {
        self.0
            .forward(device, ((&(), mask), (cross_mask, access)), state, input)
    }
}

/// Opaque F32 key/value row storage.
pub type Cache<Width> = KeyValue<RowStorage<Width>>;

/// Per-row key/value byte offsets.
pub type Offsets<Count> = KeyValue<Ids<Count>>;

/// Key/value write destinations and number of rows to write.
pub type Write<Count> = (Offsets<Count>, HbmTensor<i32, m![1], m![1]>);

/// Self-history writes and reads without rotary operands.
pub type History<Queries, Keys> = attention::Context<(), Write<Queries>, Offsets<Keys>>;

/// Writable self-attention storage paired with its history operands.
pub type State<Width, Queries, Keys> = (Cache<Width>, History<Queries, Keys>);

/// Projected source cache and the offsets used for source-memory access.
pub type Cross<Width, Source> = (Cache<Width>, Write<Source>);
