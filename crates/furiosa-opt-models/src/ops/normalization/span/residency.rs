//! The launch's hidden rows in HBM, blocked sixteen features at a time, as a span's rows on chip:
//! a span loop fills them where it enters and spills them where it leaves.

use super::{Owned, Span};
use crate::ops::normalization::{Blocked, Offset, Part, Spread};
use crate::specialize::contraction::BAND;
use crate::tensor::live::{Fill, Live, Spill};
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

/// A span's row-major rows split into the two cluster halves.
type Halves<Hidden> = m![Span / (Span / 2), Span % (Span / 2), Hidden];

/// The first `ROWS` rows of each cluster half of a span's row-major rows.
type First<Hidden, const ROWS: usize> =
    m![Span / (Span / 2), Span % (Span / 2) = { ROWS } # (Span / 2), Hidden];

/// A span's rows in HBM as the dense projections write them, row by row, land on chip a tile of
/// the rows holding the live ones at a time: the first eight or 64 rows of each cluster half when
/// those hold them all, otherwise every row.
impl<Hidden: AxisName, Chips: M + 'static> Fill<Device, Live<(), Span>>
    for HbmTensor<bf16, Chips, m![Span, Hidden]>
{
    type Rows = Owned<Hidden, Span, Chips>;

    fn fill(&self, context: &mut Device, live: Live<(), Span>) -> Self::Rows {
        const { assert!(Span::SIZE.is_multiple_of(2 * BAND)) };
        // SAFETY: the same row digits split the rows into consecutive cluster halves.
        let halves = unsafe { self.view().reshape::<Chips, Halves<Hidden>>() };
        let mut rows = Self::Rows::new();
        if live.rows() <= 8 {
            halves
                .tile::<m![Span % (Span / 2)], 8, First<Hidden, 8>>(0)
                .to_dm_view(&mut context.tdma, rows.view_mut().tile::<m![Span % (Span / 2)], 8,
                    m![Span % (Span / 2) = 8 #{!} (Span / 2), { Offset<Hidden> }]>(0));
        } else if live.rows() <= BAND {
            halves
                .tile::<m![Span % (Span / 2)], BAND, First<Hidden, BAND>>(0)
                .to_dm_view(&mut context.tdma, rows.view_mut().tile::<m![Span % (Span / 2)], BAND,
                    m![Span % (Span / 2) = { BAND } #{!} (Span / 2), { Offset<Hidden> }]>(0));
        } else {
            self.view().to_dm_view(&mut context.tdma, rows.view_mut());
        }
        rows
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
