//! Device entry bindings; numerical work stays in the models' ordinary Module composition.

use super::{Cache, Chips, Hidden, P, request::Row, text};
use furiosa_opt_models::{Module, tensor::memory::Hbm};
use furiosa_opt_std::prelude::*;

pub(super) use text::launch::Layer;

#[device(chip = 4)]
pub(super) fn layer(
    device: &mut Device,
    module: &Layer<'_, '_>,
    row: &Row,
    cache: &mut Cache,
    input: Hidden,
) -> Hidden {
    module.forward(device, row, cache, input)
}

#[device(chip = 4)]
pub(super) fn embedding(
    device: &mut Device,
    module: &&text::Embedding<P, &Hbm>,
    env: (),
    state: &mut (),
    token: &HbmTensor<i32, Chips, m![1]>,
) -> Hidden {
    module.forward(device, env, state, token)
}
