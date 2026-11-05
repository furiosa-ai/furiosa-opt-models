//! E4M3 projection with per-row activation scales and per-channel weight scales.

use super::*;
use crate::ops::swiglu::spans::fp8::{LARGEST, SMALLEST};

mod contract;
mod landing;
mod project;
mod quantize;
mod scale;
use contract::*;
use quantize::*;
use scale::*;

type Resident8<Hidden, QWidth> = m![Hidden / 256, QWidth % 2048 / 128];
type Weight8<Hidden, QWidth> = DmTensor<
    f8e4m3,
    Chip,
    Cluster<QWidth>,
    Resident8<Hidden, QWidth>,
    m![Hidden % 256, QWidth % 128],
>;
type Resident<Hidden, QWidth> = TrfTensor<
    f8e4m3,
    Chip,
    Cluster<QWidth>,
    Resident8<Hidden, QWidth>,
    m![Hidden % 8],
    m![Hidden % 256 / 8, QWidth % 128],
>;
type Rowwise<Hidden, QWidth, Capacity> = m![
    Hidden / 1024,
    Capacity / (Capacity / 4),
    QWidth % 2048 / 128
];
pub(crate) type Landing8<Hidden, QWidth, Capacity> = DmTensor<
    bf16,
    Chip,
    Cluster<QWidth>,
    Rowwise<Hidden, QWidth, Capacity>,
    m![Capacity % (Capacity / 4), QWidth % 128],
>;
type Source8<Hidden, QWidth, Capacity> = DmTensor<
    f8e4m3,
    Chip,
    Cluster<QWidth>,
    Rowwise<Hidden, QWidth, Capacity>,
    m![Capacity % (Capacity / 4), QWidth % 128],
>;
type Scales8<Hidden, QWidth, Capacity> = DmTensor<
    bf16,
    Chip,
    Cluster<QWidth>,
    m![Hidden / 1024, Capacity / (Capacity / 4), Replicas],
    m![Capacity % (Capacity / 4)],
>;
type Scales<Hidden, QWidth, Capacity> = DmTensor<
    bf16,
    Chip,
    Cluster<QWidth>,
    m![Hidden / 16],
    m![Capacity / (Capacity / 4), Capacity % (Capacity / 4)],
>;
type Channels<Hidden, QWidth> =
    DmTensor<f32, Chip, Cluster<QWidth>, m![Hidden / 16], m![Hidden % 16]>;

axes![Replicas = 16];

pub(crate) type Blocked8<Hidden, QWidth> = m![QWidth / 128, Hidden, QWidth % 128];
pub type Output8<Hidden, QWidth> = Affine<
    (
        HbmTensor<f8e4m3, Chip, Blocked8<Hidden, QWidth>>,
        HbmTensor<f32, Chip, m![Hidden]>,
    ),
    (),
>;
pub(crate) type Staged8<Hidden, QWidth> =
    Affine<(Weight8<Hidden, QWidth>, Channels<Hidden, QWidth>), ()>;

impl<Hidden: AxisName, QWidth: AxisName> Output8<Hidden, QWidth> {
    pub(crate) fn empty() -> Staged8<Hidden, QWidth> {
        Affine::new((Weight8::new(), Channels::new()), ())
    }

    pub(crate) fn load(&self, context: &mut Device, staged: &mut Staged8<Hidden, QWidth>) {
        let (weight, channels) = &mut staged.weight;
        self.weight
            .0
            .view()
            .to_dm_view(&mut context.tdma, weight.view_mut());
        self.weight
            .1
            .view()
            .to_dm_view(&mut context.tdma, channels.view_mut());
    }
}
