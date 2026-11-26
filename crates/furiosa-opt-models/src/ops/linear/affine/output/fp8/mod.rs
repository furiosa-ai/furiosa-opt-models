//! E4M3 projection with per-row activation scales and per-channel weight scales.

use super::*;
use crate::Op;
use crate::tensor::form::fp8::Rows;
pub(crate) use project::Landed;
pub(crate) use to_dm::ToDm;

pub use quantize::Tokens;

pub mod channel;
mod contract;
mod project;
mod quantize;
mod scale;
mod to_dm;

type WeightSlices<Hidden, QWidth> = m![Hidden / 256, QWidth % 2048 / 128];
type Weight<Hidden, QWidth> = DmTensor<
    f8e4m3,
    Chip,
    Cluster<QWidth>,
    WeightSlices<Hidden, QWidth>,
    m![Hidden % 256, QWidth % 128],
>;
type Resident<Hidden, QWidth> = TrfTensor<
    f8e4m3,
    Chip,
    Cluster<QWidth>,
    WeightSlices<Hidden, QWidth>,
    m![Hidden % 8],
    m![Hidden % 256 / 8, QWidth % 128],
>;
type RowWise<Hidden, QWidth, Capacity> = m![
    Hidden / 1024,
    Capacity / (Capacity / 4),
    QWidth % 2048 / 128
];
pub(crate) type InDm<Hidden, QWidth, Capacity> = DmTensor<
    bf16,
    Chip,
    Cluster<QWidth>,
    RowWise<Hidden, QWidth, Capacity>,
    m![Capacity % (Capacity / 4), QWidth % 128],
>;
type Source<Hidden, QWidth, Capacity> = DmTensor<
    f8e4m3,
    Chip,
    Cluster<QWidth>,
    RowWise<Hidden, QWidth, Capacity>,
    m![Capacity % (Capacity / 4), QWidth % 128],
>;
type ReplicatedScales<Hidden, QWidth, Capacity, D = bf16> = DmTensor<
    D,
    Chip,
    Cluster<QWidth>,
    m![Hidden / 1024, Capacity / (Capacity / 4), Replicas],
    m![Capacity % (Capacity / 4)],
>;
type Scales<Hidden, QWidth, Capacity, D = bf16> = DmTensor<
    D,
    Chip,
    Cluster<QWidth>,
    m![Hidden / 16],
    m![Capacity / (Capacity / 4), Capacity % (Capacity / 4)],
>;
type Channels<Hidden, QWidth, S = f32> =
    DmTensor<S, Chip, Cluster<QWidth>, m![Hidden / 16], m![Hidden % 16]>;

axes![Replicas = 16];

pub type Blocked<Hidden, QWidth> = m![QWidth / 128, Hidden, QWidth % 128];
pub type Stored<Hidden, QWidth, Format = crate::tensor::form::fp8::Block, S = f32> = Affine<
    Rows<
        m![QWidth],
        m![Hidden],
        HbmTensor<f8e4m3, Chip, Blocked<Hidden, QWidth>>,
        HbmTensor<S, Chip, m![Hidden]>,
        Format,
    >,
    (),
>;
pub(crate) type Staged<Hidden, QWidth, S = f32> =
    Affine<(Weight<Hidden, QWidth>, Channels<Hidden, QWidth, S>), ()>;

impl<Hidden: AxisName, QWidth: AxisName, Format, S: MaterializableScalar>
    Stored<Hidden, QWidth, Format, S>
{
    pub(crate) fn empty() -> Staged<Hidden, QWidth, S> {
        Affine::new((Weight::new(), Channels::new()), ())
    }
}

/// The weight and its channel scales, written in the staged form the projection reads.
impl<Hidden: AxisName, QWidth: AxisName, Format, S: MaterializableScalar>
    Op<&mut Staged<Hidden, QWidth, S>> for Stored<Hidden, QWidth, Format, S>
{
    type Output = ();

    fn forward(&self, context: &mut Device, staged: &mut Staged<Hidden, QWidth, S>) {
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
