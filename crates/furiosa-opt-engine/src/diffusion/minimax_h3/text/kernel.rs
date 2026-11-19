//! Device entry bindings; numerical work stays in the models' ordinary Module composition.

use super::{Cache, Chips, Hidden, P, request::Row, text};
use furiosa_opt_models::{Module, tensor::memory::Hbm};
use furiosa_opt_std::prelude::*;

#[derive(DeviceSend)]
pub(super) struct Layer<'a, 'w: 'a> {
    pub layer: &'a text::Layer<P, &'w Hbm>,
    pub positions: &'a text::position::Positions<&'w Hbm>,
}

impl Module<Hidden, &Row, &mut Cache> for Layer<'_, '_> {
    type Output = Hidden;

    fn forward(&self, device: &mut Device, row: &Row, cache: &mut Cache, input: Hidden) -> Hidden {
        let rotation = self.positions.forward(device, (), &mut (), &row.position);
        let mut state = (
            (
                ((((), ()), ((), ()), ((), ())), (), ()),
                (cache, &row.slot, &row.history),
                (),
            ),
            ((((), ()), ()), ((), ())),
        );
        self.layer
            .forward(device, (&rotation, &()), &mut state, input)
    }
}

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
