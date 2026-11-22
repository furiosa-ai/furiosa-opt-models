//! Run modules of loaded trees on the device: a `Function` binds a borrowed tree to the compiled
//! entry that runs it, and its `forward` launches that entry.
//!
//! - The entry: `forward`, which runs any module on the chip count its argument type selects.
//! - A call returns once the runtime accepted its launch. A refusal at submission is the call's
//!   error; a failure after that surfaces where a result is read, which waits for the launch.
//! - Nothing loads an entry before a call, and loading a tree binds none, so a binary that only
//!   loads reaches no kernel.

use std::{io, marker::PhantomData};

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
pub struct Function<M, F, const CHIPS: usize = 1> {
    module: M,
    entry: F,
}

impl<M, F> Function<M, F> {
    /// `module` run by `entry` on one chip.
    pub fn new(module: M, entry: F) -> Self {
        Self { module, entry }
    }
}

impl<M, F, const CHIPS: usize> Function<M, F, CHIPS> {
    /// Run the same module and entry on `N` chips.
    pub fn on<const N: usize>(self) -> Function<M, F, N> {
        Function {
            module: self.module,
            entry: self.entry,
        }
    }

    /// Submit a call and retain its completion for weight residency. Submission refusal returns
    /// an error immediately; an accepted call returns its output and work without waiting.
    /// The caller keeps every movable weight it reads until the completion ends.
    /// Returned tensors may still be written. Before passing one to a dependent multi-chip
    /// launch, wait for its `ready()` future outside the device lock; the runtime refuses
    /// overlapping accesses to peer-shared return storage. Independent prefetch can run first.
    pub fn forward_tracked<I, E, S, O>(
        &self,
        device: &mut Device,
        env: E,
        state: S,
        input: I,
    ) -> io::Result<(O, crate::residency::Completion)>
    where
        M: DeviceSend + Module<I, E, S, Output = O> + Sync,
        I: DeviceSend + Send,
        E: DeviceSend + Send,
        S: DeviceSend + Send,
        O: DeviceOutput,
        F: Copy
            + Sync
            + for<'d, 'm> DeviceFn<
                (&'d mut Device, (&'m M, PhantomData<[(); CHIPS]>), E, S, I),
                Output = O,
            >,
    {
        let mut output = O::alloc(device).map_err(io::Error::other)?;
        let work = self.forward_into_tracked(device, env, state, input, &mut output)?;
        Ok((output, work))
    }

    /// Submit into the caller's retained output, returning only its accepted-work completion.
    /// The caller retains inputs and output until completion, including on failure/cancellation.
    /// Explicit local allocations support ordered workspace/ring reuse for entries whose output
    /// writes stay on each chip; the ordinary return path keeps its shared allocation policy.
    pub fn forward_into_tracked<I, E, S, O>(
        &self,
        device: &mut Device,
        env: E,
        state: S,
        input: I,
        output: &mut O,
    ) -> io::Result<crate::residency::Completion>
    where
        M: DeviceSend + Module<I, E, S, Output = O> + Sync,
        I: DeviceSend + Send,
        E: DeviceSend + Send,
        S: DeviceSend + Send,
        O: DeviceOutput,
        F: Copy
            + Sync
            + for<'d, 'm> DeviceFn<
                (&'d mut Device, (&'m M, PhantomData<[(); CHIPS]>), E, S, I),
                Output = O,
            >,
    {
        let accepted = launch(
            self.entry,
            (
                device,
                (&self.module, PhantomData::<[(); CHIPS]>),
                env,
                state,
                input,
            ),
        )
        .output(output)
        .submit()
        .accepted()
        .map_err(io::Error::other)?;
        Ok(crate::residency::Completion::new(accepted))
    }
}

impl<M, F, I, E, S, O, const CHIPS: usize> AsyncModule<I, E, S> for Function<M, F, CHIPS>
where
    M: DeviceSend + Module<I, E, S, Output = O> + Sync,
    I: DeviceSend + Send,
    E: DeviceSend + Send,
    S: DeviceSend + Send,
    O: DeviceOutput,
    F: Copy
        + Sync
        + for<'d, 'm> DeviceFn<
            (&'d mut Device, (&'m M, PhantomData<[(); CHIPS]>), E, S, I),
            Output = O,
        >,
{
    type Output = O;

    async fn forward(&self, device: &mut Device, env: E, state: S, input: I) -> io::Result<O> {
        // The output takes new memory before the launch, as a launch into returned storage does,
        // so the call returns it at acceptance instead of at completion.
        let mut output = O::alloc(device).map_err(io::Error::other)?;
        launch(
            self.entry,
            (
                device,
                (&self.module, PhantomData::<[(); CHIPS]>),
                env,
                state,
                input,
            ),
        )
        .output(&mut output)
        .submit()
        .accepted()
        .map(drop)
        .map_err(io::Error::other)?;
        Ok(output)
    }
}

// --8<-- [start:entry]
/// Runs a module on `CHIPS` chips, on the env, state and input its launch passes.
#[device(chip = CHIPS)]
pub fn forward<const CHIPS: usize, M, I, E, S, O>(
    device: &mut Device,
    module: (&M, PhantomData<[(); CHIPS]>),
    env: E,
    state: S,
    input: I,
) -> O
where
    M: DeviceSend + Module<I, E, S, Output = O>,
    I: DeviceSend,
    E: DeviceSend,
    S: DeviceSend,
    O: DeviceOutput,
{
    module.0.forward(device, env, state, input)
}
// --8<-- [end:entry]

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_std::prelude::{Broadcast, HbmTensor, Topology, m};

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

    /// The shared entry's topology follows the function's selected chip count.
    #[test]
    fn topology_follows_chip_count() {
        fn topology<F, const CHIPS: usize>(_: Function<Echo, F, CHIPS>) -> Topology
        where
            F: DeviceFn<
                    (
                        &'static mut Device,
                        (&'static Echo, PhantomData<[(); CHIPS]>),
                        &'static Scalar,
                        &'static mut Scalar,
                        &'static Scalar,
                    ),
                    Output = Scalar,
                >,
        {
            F::TOPOLOGY
        }

        assert_eq!(
            topology(Function::new(Echo, forward)),
            Topology { chips: 1, pes: 8 }
        );
        assert_eq!(
            topology(Function::new(Echo, forward).on::<2>()),
            Topology { chips: 2, pes: 8 }
        );
        assert_eq!(
            topology(Function::new(Echo, forward).on::<4>()),
            Topology { chips: 4, pes: 8 }
        );
    }
}
