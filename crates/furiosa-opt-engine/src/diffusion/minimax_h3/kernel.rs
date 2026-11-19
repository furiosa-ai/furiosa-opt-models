//! Four-chip device entry bindings for model-owned H3 numerical stages.

use super::request;
use furiosa_opt_models::Module;
use furiosa_opt_models::models::diffusion::minimax_h3::denoiser::launch::{
    Audio, EncoderRows, Hidden, StepInput, Video,
};
use furiosa_opt_models::models::diffusion::minimax_h3::embeddings::Chips;
use furiosa_opt_models::ops::attention::flash;
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_std::prelude::*;

pub(super) use furiosa_opt_models::models::diffusion::minimax_h3::denoiser::launch::{
    Block, Condition, Embed, EmbeddedTime, Heads, RefinerLayer, RefinerNorm,
};

#[device(chip = 4)]
pub(super) fn condition<R: AxisName>(
    device: &mut Device,
    module: &Condition<'_, '_>,
    env: (),
    state: &mut (),
    source: EncoderRows<'_, R>,
) -> Hidden<R> {
    module.forward(device, env, state, source)
}

#[device(chip = 4)]
pub(super) fn refine<const N: usize, K: AxisName>(
    device: &mut Device,
    module: &RefinerLayer<'_, '_>,
    context: (&flash::Visible<K, Chips>, &Value<Chips>),
    state: &mut (),
    input: Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>>,
) -> Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>> {
    module.forward(device, context, state, input)
}

#[device(chip = 4)]
pub(super) fn refine_norm<const N: usize>(
    device: &mut Device,
    module: &RefinerNorm<'_, '_>,
    env: (),
    state: &mut (),
    input: &Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>>,
) -> Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>> {
    module.forward(device, env, state, input)
}

#[device(chip = 4)]
pub(super) fn embed<R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName>(
    device: &mut Device,
    module: &Embed<'_, '_>,
    inputs: &request::Inputs<R, K, T, A, V>,
    state: &mut (),
    source: StepInput<'_, T, A, V>,
) -> (Hidden<R>, EmbeddedTime) {
    module.forward(device, inputs, state, source)
}

#[device(chip = 4)]
pub(super) fn block<R: AxisName, K: AxisName, T, A, V>(
    device: &mut Device,
    module: &Block<'_, '_>,
    inputs: &request::Inputs<R, K, T, A, V>,
    state: &mut (),
    source: (Hidden<R>, &EmbeddedTime),
) -> Hidden<R> {
    module.forward(device, inputs, state, source)
}

#[device(chip = 4)]
pub(super) fn heads<R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName>(
    device: &mut Device,
    module: &Heads<'_, '_>,
    inputs: &request::Inputs<R, K, T, A, V>,
    state: &mut (),
    source: (&Hidden<R>, &EmbeddedTime),
) -> (Video<V>, Audio<A>) {
    module.forward(device, inputs, state, source)
}
