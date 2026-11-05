use furiosa_opt_std::prelude::*;

use crate::models::nemo::serving::{Hidden, Rows};

axes![Out = 1024];
pub type Cluster = m![Out / 512];
pub type Slice = m![Out % 512 / 128, Hidden / 80];
pub type Reduced = m![Out % 512 / 128, Out % 128 / 64, Out % 64 / 32, Rows / 4];
pub type Lanes = m![Out % 8];
pub type Elements = m![Out % 128 / 32, Hidden % 80 / 16, Out % 32 / 8, Hidden % 16];
type FillTime = m![Out % 8, Out % 128 / 32, Hidden % 80 / 16];
pub type FillPacket = m![Out % 32 / 8, Hidden % 16];
type FillFlits = m![{ FillTime }, Out % 32 / 16];
pub type FillFlit = m![Out % 16 / 8, Hidden % 16];
type FeedTime = m![Out % 128 / 32, Rows / 4, Hidden % 80 / 16];
type Time = m![{ FeedTime }, Out % 32 / 8];
type Packet = m![Rows % 4, Hidden % 16];
type Flits = m![{ FeedTime }, Rows % 4 / 2];
type Flit = m![Rows % 2, Hidden % 16];
type Sums = m![Out % 128 / 32, Rows / 4, Out % 32 / 8];
type Groups = m![{ Sums }, Rows % 4];
pub type Activation<B> = DmTensor<f8e4m3, B, Cluster, Slice, m![Rows, Hidden % 80]>;
pub type Weight<B> = DmTensor<f8e4m3, B, Cluster, Slice, m![{ Lanes }, { Elements }]>;
pub type Stored = m![{ Cluster }, { Slice }, { Lanes }, { Elements }];
pub type Register<B> = TrfTensor<f8e4m3, B, Cluster, Slice, Lanes, Elements>;
pub type Values<B> = DmTensor<bf16, B, Cluster, Reduced, m![Rows % 4, Out % 32 / 8, Out % 8]>;

/// A weight panel's register, from the panel's [`Weight`] or a one-panel view of a wider tensor.
pub fn fill<B: M, E: M>(
    device: &mut Device,
    weight: DmTensorView<'_, f8e4m3, B, Cluster, Slice, E>,
) -> Register<B> {
    device
        .sub
        .begin(weight)
        .fetch::<FillTime, FillPacket>()
        .collect::<FillFlits, FillFlit>()
        .to_trf()
}

impl crate::ops::linear::contraction::Mapping for crate::compose::Tile<Rows, Rows> {
    type Cluster = Cluster;
    type Slices = Slice;
    type Reduced = Reduced;
    type Input = m![Rows, Hidden % 80];
    type Lanes = Lanes;
    type Weight = Elements;
    type Feed = FeedTime;
    type Packet = Packet;
    type Flits = Flits;
    type Flit = Flit;
    type Steps = Time;
    type Contracted = m![Rows % 4];
    type Sums = Sums;
    type Groups = Groups;
    type Output = m![Rows % 4, Out % 32 / 8];
}
