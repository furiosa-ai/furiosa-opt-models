//! Mistral NeMo, the one model this crate serves.

pub mod nemo;

#[cfg(test)]
use crate::Module;
#[cfg(test)]
use furiosa_opt_std::prelude::{Device, device};
#[cfg(test)]
use furiosa_opt_std::runtime::{DeviceOutput, DeviceSend};

/// Device entry the operations' tests run a module through on one chip, with caller-owned input
/// and state.
#[cfg(test)]
#[device(chip = 1)]
pub(crate) fn forward<M, I, O>(device: &mut Device, module: &M, input: I) -> O
where
    M: DeviceSend + for<'d> Module<(&'d mut Device, I), Output = O>,
    I: DeviceSend,
    O: DeviceOutput,
{
    module.forward((device, input))
}
