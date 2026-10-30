//! One module applied independently to every input.

use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Apply one function independently to every input.
#[derive(Clone, DeviceSend)]
pub struct Each<M>(pub M);

impl<'s, I, O, M, MState, const N: usize, Env: Copy> Module<[I; N], Env, &'s mut [MState; N]>
    for Each<M>
where
    M: Module<I, Env, &'s mut MState, Output = O>,
{
    type Output = [O; N];

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &'s mut [MState; N],
        inputs: [I; N],
    ) -> Self::Output {
        let mut slots = state.iter_mut();
        inputs.map(|input| {
            let slot = slots.next().expect("one slot for each input");
            self.0.forward(&mut *device, env, slot, input)
        })
    }
}
