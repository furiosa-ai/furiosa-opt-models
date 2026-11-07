//! E4M3 span projections with row activation scales and folded weight channel scales.

use super::*;
use crate::compose::Points;
use crate::ops::add::clusters as phases;
use crate::ops::normalization::span::Owned;

mod activation;
mod compute;
mod contract;
mod down;
mod input;
mod live;
mod spread;
mod transfer;
use activation::*;
use compute::*;
use contract::*;
use down::*;
use input::*;
pub(crate) use live::Span;
use spread::*;
use transfer::*;

pub(crate) const LARGEST: f32 = 448.0;
pub(crate) const SMALLEST: f32 = 1e-12;
const WAVE: usize = 256;
axes![Copies = 32, Side = 2, Panels = 3];

pub(crate) type Panel<H, I> =
    DmTensor<f8e4m3, Chip, Cluster<I>, Partition<H, I>, m![{ Out<I> }, { K<H> }]>;
pub(crate) type GateUp<H, I> =
    DmTensor<f8e4m3, Chip, Cluster<I>, Partition<H, I>, m![Term, { Out<I> }, { K<H> }]>;
pub(crate) type Ahead<H, I> = GateUp<H, I>;
type Both<I, Rows> =
    DmTensor<bf16, Chip, Cluster<I>, Slice<I, Rows>, m![Rows % (Rows / 8), Term, { Out<I> }]>;
type Input<H, I, Rows> = DmTensor<
    f8e4m3,
    Chip,
    Cluster<I>,
    m![H % 1024 / 32, Rows / (Rows / 8)],
    m![H / 1024, Rows % (Rows / 8), H % 32],
>;
type Lanes<H, I, Rows> = TrfTensor<
    f8e4m3,
    Chip,
    Cluster<I>,
    Partition<H, I>,
    m![Rows % 8],
    m![Rows % 16 / 8, H % 128 / 32, Rows / (Rows / 8), H % 32],
>;
type DownSlice<H, I> = m![H / 512, I % 6144 / 192];
pub(crate) type Down<H, I> =
    DmTensor<f8e4m3, Chip, Cluster<I>, DownSlice<H, I>, m![H % 512, I % 64]>;
type Product<H, I, Rows> = DmTensor<bf16, Chip, Cluster<I>, DownSlice<H, I>, m![Rows, I % 64]>;
type Activation<H, I, Rows> = DmTensor<f8e4m3, Chip, Cluster<I>, DownSlice<H, I>, m![Rows, I % 64]>;
type Spread<I, Rows, E> = DmTensor<
    E,
    Chip,
    Cluster<I>,
    m![I % 6144 / 192, Rows / (Rows / 8)],
    m![Panels, Rows % (Rows / 8), I % 64],
>;
type Scale<H, I, Rows> = DmTensor<bf16, Chip, Cluster<I>, m![H / 16], m![Rows]>;
type RowLanes<Rows> = crate::ops::layout::Rows<Rows>::Lanes;
type Operand<H, I, Rows, Fill, const W: usize> = TrfTensor<
    f8e4m3,
    Chip,
    Cluster<I>,
    DownSlice<H, I>,
    RowLanes<Rows>,
    m![
        { Fill } % { W } / 32,
        { Fill } / { W },
        Rows % 32 / 8,
        I % 64
    ],
>;
pub(crate) type Chunks<I, Rows> = m![I / 6144, I % 6144 / 64, Rows, I % 64];
type ChunkSlices<I, Rows> = m![I / 6144, I % 6144 / 192, I % 192 / 64, Rows, I % 64];
type ChunkPanel<I, Rows> = m![I / 6144, I % 6144 / 192, I % 192 / 64 = 1 # 3, Rows, I % 64];
type Partial<H, I, Rows> =
    DmTensor<bf16, Chip, Cluster<I>, m![H / 16], m![{ Groups<Rows> }, { RowLanes<Rows> }, H % 16]>;
type Scales<Rows> = HbmTensor<bf16, Chip, m![Rows]>;
type Labels<Rows> = m![
    Rows % (Rows / 8) / 8,
    Rows % (Rows / 2) / (Rows / 4),
    Rows % (Rows / 4) / (Rows / 8),
    Rows % 8
];
type RowScale<I, Rows> = VrfTensor<f32, Chip, Cluster<I>, Slice<I, Rows>, m![Rows % (Rows / 8)]>;
type ChannelScale<I, Rows> = VrfTensor<f32, Chip, Cluster<I>, Slice<I, Rows>, m![{ Out<I> }]>;
type PanelScale<H, I, Rows> =
    VrfTensor<f32, Chip, Cluster<I>, m![H / 16], m![{ Groups<Rows> }, { RowLanes<Rows> }]>;
pub type Stored<H, I> = m![
    I % 192 / 64,
    I / 6144,
    H / 512,
    I % 6144 / 192,
    H % 512,
    I % 64
];
type StoredPanel<H, I> = m![I % 192 / 64 = 1 # 3, I / 6144, H / 512,
    I % 6144 / 192, H % 512, I % 64];
pub(crate) type Blocked<H, I> = m![Term, H / 1024, I / 192, H % 1024 / 128, I % 192, H % 128];
pub type Mlp8<H, I> = crate::ops::swiglu::Mlp<
    H,
    I,
    (
        HbmTensor<f8e4m3, Chip, Blocked<H, I>>,
        HbmTensor<f32, Chip, m![Term, I]>,
    ),
    (
        HbmTensor<f8e4m3, Chip, Stored<H, I>>,
        HbmTensor<f32, Chip, m![H]>,
    ),
>;
type Channels<I, Rows> = DmTensor<f32, Chip, Cluster<I>, Slice<I, Rows>, m![{ Out<I> }]>;

impl<Hidden: AxisName, Intermediate: AxisName> Mlp8<Hidden, Intermediate>
where
    super::super::Down<Intermediate>: AxisName,
{
    pub(crate) fn panel(
        &self,
        context: &mut Device,
        term: usize,
        panel: usize,
    ) -> Panel<Hidden, Intermediate> {
        self.gate_up
            .0
            .view()
            .tile::<m![Term], 1, m![Term = 1 # 2, Hidden / 1024, Intermediate / 192,
                Hidden % 1024 / 128, Intermediate % 192, Hidden % 128]>(term)
            .tile::<m![Hidden / 1024], 1, m![Term = 1 # 2, Hidden / 1024 = 1 # 4,
                Intermediate / 192, Hidden % 1024 / 128, Intermediate % 192, Hidden % 128]>(
                panel
            )
            .to_dm(&mut context.tdma)
    }

    pub(crate) fn pair(&self, context: &mut Device, panel: usize) -> GateUp<Hidden, Intermediate> {
        self.gate_up
            .0
            .view()
            .tile::<m![Hidden / 1024], 1, m![Term, Hidden / 1024 = 1 # 4,
                Intermediate / 192, Hidden % 1024 / 128, Intermediate % 192, Hidden % 128]>(
                panel
            )
            .to_dm(&mut context.tdma)
    }

    pub(crate) fn down(&self, context: &mut Device, panel: usize) -> Down<Hidden, Intermediate> {
        let mut down = Down::new();
        self.load_down_into(context, panel, &mut down);
        down
    }

    pub(crate) fn load_down_into(
        &self,
        context: &mut Device,
        panel: usize,
        down: &mut Down<Hidden, Intermediate>,
    ) {
        self.down
            .0
            .view()
            .tile::<m![Intermediate % 192 / 64], 1, StoredPanel<Hidden, Intermediate>>(panel)
            .to_dm_view(&mut context.tdma, down.view_mut());
    }

    fn channels<Rows: AxisName>(
        &self,
        context: &mut Device,
        term: usize,
    ) -> Channels<Intermediate, Rows> {
        self.gate_up
            .1
            .view()
            .tile::<m![Term], 1, m![Term = 1 # 2, Intermediate]>(term)
            .to_dm(&mut context.tdma)
    }
}
