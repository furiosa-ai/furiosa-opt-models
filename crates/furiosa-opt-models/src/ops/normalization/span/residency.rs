//! The launch's hidden rows in HBM, blocked sixteen features at a time, as a span's rows on chip:
//! a span loop fills them where it enters and spills them where it leaves.

use super::{Owned, Span};
use crate::modules::generic::schedule::{Fill, Spill};
use crate::ops::normalization::{Blocked, Offset, Part, Spread};
use crate::tensor::live::Live;
use crate::tensor::{axis::Rows, parallel::Chip};
use furiosa_opt_std::prelude::*;

type Tile = Rows<128>;

/// A span's rows split into the two cluster halves; a launch of at most eight rows, or a span's
/// single wave, is written alone.
impl<Hidden: AxisName, Capacity: AxisName, Chips: M + 'static> Fill<Device, Live<(), Capacity>>
    for HbmTensor<bf16, Chips, Blocked<Hidden, Capacity>>
{
    type Rows = Owned<Hidden, Span, Chips>;

    fn fill(&self, context: &mut Device, live: Live<(), Capacity>) -> Self::Rows {
        const { assert!(Capacity::SIZE.is_multiple_of(Span::SIZE)) };
        // SAFETY: the tiled span retains the hidden features and splits its rows into cluster
        // halves.
        let rows = unsafe {
            self.view().tile::<m![Capacity], { Span::SIZE },
                m![{ Part<Hidden> }, Capacity = { Span::SIZE } # Capacity, { Offset<Hidden> }]>(
                    live.start(),
                )
                .reshape::<Chips, m![{ Part<Hidden> }, Span / (Span / 2) # (Capacity / (Span / 2)),
                    Span % (Span / 2), { Offset<Hidden> }]>()
        };
        let mut input = Self::Rows::new();
        if live.rows() <= 8 {
            rows.tile::<m![Span % (Span / 2)], 8, m![{ Part<Hidden> },
                Span / (Span / 2) # (Capacity / (Span / 2)),
                Span % (Span / 2) = 8 # (Span / 2), { Offset<Hidden> }]>(0)
                .to_dm_view(&mut context.tdma, input.view_mut()
                    .tile::<m![Span % (Span / 2)], 8,
                        m![Span % (Span / 2) = 8 #{!} (Span / 2), { Offset<Hidden> }]>(0));
        } else if live.count::<Tile>() == 1 {
            rows.tile::<m![Span % (Span / 2)], 128, m![{ Part<Hidden> },
                Span / (Span / 2) # (Capacity / (Span / 2)),
                Span % (Span / 2) = 128 # (Span / 2), { Offset<Hidden> }]>(0)
                .to_dm_view(&mut context.tdma, input.view_mut()
                    .tile::<m![Span % (Span / 2)], 128,
                        m![Span % (Span / 2) = 128 #{!} (Span / 2), { Offset<Hidden> }]>(0));
        } else {
            rows.to_dm_view(&mut context.tdma, input.view_mut());
        }
        input
    }
}

/// A span's rows back to their place; a span's single wave moves alone.
impl<'r, Hidden: AxisName, Capacity: AxisName, Chips: M + 'static>
    Spill<Device, &'r Owned<Hidden, Span, Chips>, Live<(), Capacity>>
    for HbmTensor<bf16, Chips, Blocked<Hidden, Capacity>>
{
    fn spill(
        &mut self,
        context: &mut Device,
        rows: &'r Owned<Hidden, Span, Chips>,
        live: Live<(), Capacity>,
    ) {
        const { assert!(Capacity::SIZE.is_multiple_of(Span::SIZE)) };
        let span = self.view_mut().tile::<m![Capacity], { Span::SIZE },
            m![{ Part<Hidden> }, Capacity = { Span::SIZE } #{!} Capacity, { Offset<Hidden> }]>(
                live.start(),
            );
        if live.count::<Tile>() == 1 {
            // SAFETY: the same row digits split the tiled span into consecutive cluster halves.
            let halves = unsafe {
                span.reshape::<Chips, m![{ Part<Hidden> },
                    Span / (Span / 2) #{!} (Capacity / (Span / 2)), Span % (Span / 2),
                    { Offset<Hidden> }]>()
            };
            rows.view().tile::<m![Span % (Span / 2)], 128,
                m![Span % (Span / 2) = 128 # (Span / 2), { Offset<Hidden> }]>(0)
                .to_hbm_view(&mut context.tdma, halves.tile::<m![Span % (Span / 2)], 128,
                    m![{ Part<Hidden> }, Span / (Span / 2) #{!} (Capacity / (Span / 2)),
                        Span % (Span / 2) = 128 #{!} (Span / 2), { Offset<Hidden> }]>(0));
        } else {
            rows.view().to_hbm_view(&mut context.tdma, span);
        }
    }
}

/// The rows a launch returns tokens of, among its hidden rows: each one's blocked byte offset.
pub(crate) struct Selection<'a, Hidden: AxisName, Capacity: AxisName, Chips: M + 'static = Chip> {
    pub(crate) hidden: &'a HbmTensor<bf16, Chips, Blocked<Hidden, Capacity>>,
    pub(crate) rows: &'a HbmTensor<i32, Chips, m![Capacity]>,
}

/// A span of the selected rows, gathered a wave of 128 at a time into the feature slices the
/// final norm reads.
impl<Hidden: AxisName, Capacity: AxisName, Chips: M + 'static> Fill<Device, Live<(), Capacity>>
    for Selection<'_, Hidden, Capacity, Chips>
{
    type Rows = Spread<Hidden, Span, Chips>;

    fn fill(&self, context: &mut Device, tile: Live<(), Capacity>) -> Self::Rows {
        const { assert!(Capacity::SIZE.is_multiple_of(Span::SIZE)) };
        let mut rows: Self::Rows = DmTensor::new();
        let (first, waves) = tile.groups::<Tile>();
        for wave in 0..waves {
            let mut offsets = HbmTensor::<i32, Chips, m![Tile]>::new();
            self.rows
                .view()
                .tile::<m![Capacity / Tile], 1, m![
                    Capacity / Tile = 1 # (Capacity / Tile), Capacity % Tile
                ]>(first + wave)
                .to_hbm_view(&mut context.tdma, offsets.view_mut());
            let gathered: Spread<Hidden, Tile, Chips> = self
                .hidden
                .gather::<m![Capacity], m![Hidden]>()
                .by_byte_offsets(&offsets)
                .to_dm(&mut context.tdma);
            gathered.view().to_dm_view(
                &mut context.tdma,
                rows.view_mut().tile::<m![Span / Tile], 1, m![
                    Span / Tile = 1 #{!} (Span / Tile), Span % Tile, { Offset<Hidden> }
                ]>(wave),
            );
        }
        rows
    }
}
