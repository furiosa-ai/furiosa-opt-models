//! Models as trees of `Module`s: numerical operations, reusable module structures and sized model
//! definitions.

#![feature(register_tool)]
#![feature(proc_macro_hygiene, stmt_expr_attributes)]
#![feature(inherent_associated_types)]
#![feature(trait_alias)]
#![feature(associated_type_defaults)]
#![feature(adt_const_params, unsized_const_params)]
#![feature(const_trait_impl)]
#![feature(negative_impls, with_negative_coherence)]
#![cfg_attr(test, feature(float_erf))]
#![register_tool(furiosa_opt)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(unreachable_pub)]

extern crate self as furiosa_opt_models;

pub mod load;
mod module;
pub use module::{Module, Op};
pub(crate) use modules::generic::Order;
pub use modules::generic::{
    Branch, Each, First, Fork, Identity, Interleave, Layers, Repeat, Rolled, Sequential, Then,
    Unrolled,
};
pub(crate) use modules::generic::{Collection, Either};

pub mod models;
pub mod modules;
pub mod ops;
pub mod schedule;
pub mod specialize;
pub mod tensor;

#[cfg(test)]
mod tests {
    use crate::Module;
    use furiosa_opt_std::prelude::{Device, DeviceSend, device};
    use furiosa_opt_std::runtime::DeviceOutput;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use tokio::sync::Mutex;

    // Device tests share one chip, so they run one at a time.
    pub(crate) static DEVICE: Mutex<()> = Mutex::const_new(());

    /// Runs a module on one chip, on the env, state and input a test launches it with.
    #[device(chip = 1)]
    pub(crate) fn forward<M, I, E, S, O>(
        device: &mut Device,
        module: &M,
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
        module.forward(device, env, state, input)
    }

    /// `forward` on four chips.
    #[device(chip = 4)]
    pub(crate) fn four<M, I, E, S, O>(
        device: &mut Device,
        module: &M,
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
        module.forward(device, env, state, input)
    }

    /// The hub cache's directory of the files of checkpoint `repo` at `revision` that `include`
    /// matches, fetched there once.
    pub(crate) fn snapshot((repo, revision): (&str, &str), include: &[&str]) -> PathBuf {
        let mut command = Command::new("uvx");
        command.args(["--from", "huggingface-hub", "hf", "download"]);
        command.args([repo, "--revision", revision, "--quiet"]);
        for pattern in include {
            command.args(["--include", pattern]);
        }
        let output = command
            .stderr(Stdio::inherit())
            .output()
            .expect("uvx fetches the test checkpoint");
        assert!(output.status.success(), "the {repo} download failed");
        String::from_utf8(output.stdout).unwrap().trim().into()
    }
}
