//! The launch's hidden rows in HBM, blocked sixteen features at a time, as a span's rows on chip:
//! a span loop fills them where it enters and spills them where it leaves.

use super::{Owned, Span};
use crate::compose::{Fill, Spill, Tile, Tiles};
use crate::ops::layout::axes::{Chip, Rows};
use crate::ops::normalization::{Blocked, Spread};
use furiosa_opt_std::prelude::*;

type Spans<Capacity> = Tile<Capacity, Span>;
type Waves<Capacity> = Tile<Capacity, Rows<128>>;
type Wave = Rows<128>;

/// A span's rows split into the two cluster halves; a launch of at most eight rows, or a span's
/// single wave, lands alone.
impl<Hidden: AxisName, Capacity: AxisName> Fill<Device, Tiles<Spans<Capacity>>>
    for HbmTensor<bf16, Chip, Blocked<Hidden, Capacity>>
{
    type Rows = Owned<Hidden, Span>;

    fn fill(&self, context: &mut Device, live: Tiles<Spans<Capacity>>) -> Self::Rows {
        const { assert!(Capacity::SIZE.is_multiple_of(Span::SIZE)) };
        // SAFETY: the tiled span retains the hidden features and splits its rows into cluster halves.
        let rows = unsafe {
            self.view().tile::<m![Capacity], { Span::SIZE },
                m![Hidden / 16, Capacity = { Span::SIZE } # Capacity, Hidden % 16]>(live.start())
                .reshape::<Chip, m![Hidden / 16, Span / (Span / 2) # (Capacity / (Span / 2)),
                    Span % (Span / 2), Hidden % 16]>()
        };
        let mut input = Self::Rows::new();
        if live.rows() <= 8 {
            rows.tile::<m![Span % (Span / 2)], 8, m![Hidden / 16,
                Span / (Span / 2) # (Capacity / (Span / 2)),
                Span % (Span / 2) = 8 # (Span / 2), Hidden % 16]>(0)
                .to_dm_view(&mut context.tdma, input.view_mut()
                    .tile::<m![Span % (Span / 2)], 8,
                        m![Span % (Span / 2) = 8 #{!} (Span / 2), Hidden % 16]>(0));
        } else if live.retile::<Waves<Capacity>>().count() == 1 {
            rows.tile::<m![Span % (Span / 2)], 128, m![Hidden / 16,
                Span / (Span / 2) # (Capacity / (Span / 2)),
                Span % (Span / 2) = 128 # (Span / 2), Hidden % 16]>(0)
                .to_dm_view(&mut context.tdma, input.view_mut()
                    .tile::<m![Span % (Span / 2)], 128,
                        m![Span % (Span / 2) = 128 #{!} (Span / 2), Hidden % 16]>(0));
        } else {
            rows.to_dm_view(&mut context.tdma, input.view_mut());
        }
        input
    }
}

/// A span's rows back to their place; a span's single wave moves alone.
impl<'r, Hidden: AxisName, Capacity: AxisName>
    Spill<Device, &'r Owned<Hidden, Span>, Tiles<Spans<Capacity>>>
    for HbmTensor<bf16, Chip, Blocked<Hidden, Capacity>>
{
    fn spill(
        &mut self,
        context: &mut Device,
        rows: &'r Owned<Hidden, Span>,
        live: Tiles<Spans<Capacity>>,
    ) {
        const { assert!(Capacity::SIZE.is_multiple_of(Span::SIZE)) };
        let span = self.view_mut().tile::<m![Capacity], { Span::SIZE },
            m![Hidden / 16, Capacity = { Span::SIZE } #{!} Capacity, Hidden % 16]>(live.start());
        if live.retile::<Waves<Capacity>>().count() == 1 {
            // SAFETY: the same row digits split the tiled span into consecutive cluster halves.
            let halves = unsafe {
                span.reshape::<Chip, m![Hidden / 16, Span / (Span / 2) #{!} (Capacity / (Span / 2)),
                    Span % (Span / 2), Hidden % 16]>()
            };
            rows.view().tile::<m![Span % (Span / 2)], 128,
                m![Span % (Span / 2) = 128 # (Span / 2), Hidden % 16]>(0)
                .to_hbm_view(&mut context.tdma, halves.tile::<m![Span % (Span / 2)], 128,
                    m![Hidden / 16, Span / (Span / 2) #{!} (Capacity / (Span / 2)),
                        Span % (Span / 2) = 128 #{!} (Span / 2), Hidden % 16]>(0));
        } else {
            rows.view().to_hbm_view(&mut context.tdma, span);
        }
    }
}

/// The rows a launch returns tokens of, among its hidden rows: each one's blocked byte offset.
pub(crate) struct Selection<'a, Hidden: AxisName, Capacity: AxisName> {
    pub(crate) hidden: &'a HbmTensor<bf16, Chip, Blocked<Hidden, Capacity>>,
    pub(crate) rows: &'a HbmTensor<i32, Chip, m![Capacity]>,
}

/// A span of the selected rows, gathered a wave of 128 at a time into the feature slices the
/// final norm reads.
impl<Hidden: AxisName, Capacity: AxisName> Fill<Device, Tiles<Spans<Capacity>>>
    for Selection<'_, Hidden, Capacity>
{
    type Rows = Spread<Hidden, Span>;

    fn fill(&self, context: &mut Device, tile: Tiles<Spans<Capacity>>) -> Self::Rows {
        const { assert!(Capacity::SIZE.is_multiple_of(Span::SIZE)) };
        let mut rows: Self::Rows = DmTensor::new();
        for wave in 0..(tile.rows() + Wave::SIZE - 1) / Wave::SIZE {
            let mut offsets = HbmTensor::<i32, Chip, m![Wave]>::new();
            self.rows
                .view()
                .tile::<m![Capacity / Wave], 1, m![
                    Capacity / Wave = 1 # (Capacity / Wave), Capacity % Wave
                ]>(tile.start() / Wave::SIZE + wave)
                .to_hbm_view(&mut context.tdma, offsets.view_mut());
            let gathered: Spread<Hidden, Wave> = self
                .hidden
                .gather::<m![Capacity], m![Hidden]>()
                .by_byte_offsets(&offsets)
                .to_dm(&mut context.tdma);
            gathered.view().to_dm_view(
                &mut context.tdma,
                rows.view_mut().tile::<m![Span / Wave], 1, m![
                    Span / Wave = 1 #{!} (Span / Wave), Span % Wave, Hidden % 16
                ]>(wave),
            );
        }
        rows
    }
}
