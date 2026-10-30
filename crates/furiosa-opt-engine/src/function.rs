//! Run modules of loaded trees on the device: a `Function` binds a borrowed tree to the compiled
//! entry that runs it, and its `forward` launches that entry.
//!
//! - The entry: `forward`, which runs any module on one chip.
//! - A call returns once the runtime accepted its launch. A refusal at submission is the call's
//!   error; a failure after that surfaces where a result is read, which waits for the launch.
//! - Nothing loads an entry before a call, and loading a tree binds none, so a binary that only
//!   loads reaches no kernel.

use std::io;

use furiosa_opt_models::Module;
use furiosa_opt_std::prelude::{Device, DeviceFn, DeviceSend, device, launch};
use furiosa_opt_std::runtime::DeviceOutput;

/// A computation the host runs by launching device work: `Module`'s arguments, and a result the
/// call returns once the runtime accepted the launches.
pub trait AsyncModule<I, E, S> {
    /// The result; a read of its device values waits for the launches that write them.
    type Output;

    /// Launches the computation of `input` under `env` on `state`.
    fn forward(
        &self,
        device: &mut Device,
        env: E,
        state: S,
        input: I,
    ) -> impl Future<Output = io::Result<Self::Output>> + Send;
}

/// A borrowed tree and the entry that runs it on the device.
pub struct Function<M, F> {
    module: M,
    entry: F,
}

impl<M, F> Function<M, F> {
    /// `module` run by `entry`.
    pub fn new(module: M, entry: F) -> Self {
        Self { module, entry }
    }
}

impl<M, F, I, E, S, O> AsyncModule<I, E, S> for Function<M, F>
where
    M: DeviceSend + Module<I, E, S, Output = O> + Sync,
    I: DeviceSend + Send,
    E: DeviceSend + Send,
    S: DeviceSend + Send,
    O: DeviceOutput,
    F: Copy + Sync + for<'d, 'm> DeviceFn<(&'d mut Device, &'m M, E, S, I), Output = O>,
{
    type Output = O;

    async fn forward(&self, device: &mut Device, env: E, state: S, input: I) -> io::Result<O> {
        // The output takes new memory before the launch, as a launch into returned storage does,
        // so the call returns it at acceptance instead of at completion.
        let mut output = O::alloc(device).map_err(io::Error::other)?;
        launch(self.entry, (device, &self.module, env, state, input))
            .output(&mut output)
            .submit()
            .accepted()
            .map(drop)
            .map_err(io::Error::other)?;
        Ok(output)
    }
}

// --8<-- [start:entry]
/// Runs a module on one chip, on the env, state and input its launch passes.
#[device(chip = 1)]
pub fn forward<M, I, E, S, O>(device: &mut Device, module: &M, env: E, state: S, input: I) -> O
where
    M: DeviceSend + Module<I, E, S, Output = O>,
    I: DeviceSend,
    E: DeviceSend,
    S: DeviceSend,
    O: DeviceOutput,
{
    module.forward(device, env, state, input)
}
// --8<-- [end:entry]

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_std::prelude::{Broadcast, HbmTensor, m};

    type Scalar = HbmTensor<f32, m![1], m![1]>;

    /// Returns a copy of its input; its call also reads an env and borrows a state.
    #[derive(DeviceSend)]
    struct Echo;

    impl<'a> Module<&'a Scalar, &'a Scalar, &'a mut Scalar> for Echo {
        type Output = Scalar;

        fn forward(
            &self,
            device: &mut Device,
            _: &'a Scalar,
            _: &'a mut Scalar,
            input: &'a Scalar,
        ) -> Scalar {
            input.to_hbm(&mut device.tdma)
        }
    }

    // One entry per chip count, since `#[device]` takes a literal.
    /// `forward` on two chips.
    #[device(chip = 2)]
    fn two<M, I, E, S, O>(device: &mut Device, module: &M, env: E, state: S, input: I) -> O
    where
        M: DeviceSend + Module<I, E, S, Output = O>,
        I: DeviceSend,
        E: DeviceSend,
        S: DeviceSend,
        O: DeviceOutput,
    {
        module.forward(device, env, state, input)
    }

    /// `forward` on four chips.
    #[device(chip = 4)]
    fn four<M, I, E, S, O>(device: &mut Device, module: &M, env: E, state: S, input: I) -> O
    where
        M: DeviceSend + Module<I, E, S, Output = O>,
        I: DeviceSend,
        E: DeviceSend,
        S: DeviceSend,
        O: DeviceOutput,
    {
        module.forward(device, env, state, input)
    }

    /// A function launches a module with its env and state through the entry of each chip count.
    const _: () = {
        fn launches<A>(_: A)
        where
            A: AsyncModule<&'static Scalar, &'static Scalar, &'static mut Scalar, Output = Scalar>,
        {
        }
        let _ = || {
            launches(Function::new(&Echo, forward));
            launches(Function::new(&Echo, two));
            launches(Function::new(&Echo, four));
        };
    };
}
