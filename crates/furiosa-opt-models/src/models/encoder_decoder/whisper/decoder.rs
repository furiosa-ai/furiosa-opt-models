//! Decoder history is writable; encoded source projections become read-only after creation.

use super::Mask;
use crate::Module;
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
