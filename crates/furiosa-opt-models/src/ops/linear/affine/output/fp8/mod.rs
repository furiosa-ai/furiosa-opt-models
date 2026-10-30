//! E4M3 projection with per-row activation scales and per-channel weight scales.

use super::*;
use crate::tensor::form::fp8::Rows;
pub(crate) use project::Landed;
pub(crate) use to_dm::ToDm;

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
type ReplicatedScales<Hidden, QWidth, Capacity> = DmTensor<
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

pub type Blocked<Hidden, QWidth> = m![QWidth / 128, Hidden, QWidth % 128];
pub type Stored<Hidden, QWidth> = Affine<
    Rows<
        m![QWidth],
        m![Hidden],
        HbmTensor<f8e4m3, Chip, Blocked<Hidden, QWidth>>,
        HbmTensor<f32, Chip, m![Hidden]>,
    >,
    (),
>;
type Staged<Hidden, QWidth> = Affine<(Weight<Hidden, QWidth>, Channels<Hidden, QWidth>), ()>;
