use super::activation::Landed;
use super::*;
use crate::compose::{Spill, Tiles};
use crate::ops::normalization::span::{Owned, Span};

/// A launch's spans and waves, in its envelope of `Capacity` rows.
type Spans<Capacity> = crate::compose::Tile<Capacity, Span>;
type Waves<Capacity> = crate::compose::Tile<Capacity, Tile>;

type Wave<Hidden> = DmTensor<
    bf16,
    Chip,
    m![2],
    m![4, Hidden / 64],
    m![
        Tile % 32 / 8,
        Tile / 32,
        Hidden % 64 / 16,
        Tile % 8,
        Hidden % 16
    ],
>;

fn land<Hidden: AxisName, All: AxisName>(
    context: &mut Device,
    normalized: &Normalized<Hidden, All>,
    start: usize,
) -> Wave<Hidden> {
    let rows = normalized.view().tile::<m![All / 8], { Tile::SIZE / 8 },
        m![All / 8 = { Tile::SIZE / 8 } # (All / 8), Hidden / 16, All % 8, Hidden % 16]>(start / 8);
    // SAFETY: the window's row groups are the wave's, in order.
    let rows = unsafe {
        rows.reshape::<Chip, m![Tile / 8 # (All / 8), Hidden / 16, Tile % 8, Hidden % 16]>()
    };
    rows.to_dm(&mut context.tdma)
}

fn land_live<Hidden: AxisName, All: AxisName>(
    context: &mut Device,
    normalized: &Normalized<Hidden, All>,
    start: usize,
    groups: usize,
) -> Wave<Hidden> {
    let mut landed = Wave::new();
    if groups == 1 {
        let group = normalized
            .view()
            .tile::<m![All / 8], 1, m![All / 8 = 1 # (All / 8), Hidden / 16, All % 8, Hidden % 16]>(
                start / 8,
            );
        // SAFETY: the window's row group is the wave's first, in order.
        let group = unsafe {
            group.reshape::<Chip, m![All / 8 = 1 # (All / 8), Hidden / 16, Tile % 8, Hidden % 16]>()
        };
        group.to_dm_view(
            &mut context.tdma,
            landed
                .view_mut()
                .tile::<m![Tile % 32 / 8], 1, m![Tile % 32 / 8 = 1 #{!} 4, Tile / 32,
                Hidden % 64 / 16, Tile % 8, Hidden % 16]>(0)
                .tile::<m![Tile / 32], 1, m![Tile % 32 / 8 = 1 #{!} 4, Tile / 32 = 1 #{!} 4,
                Hidden % 64 / 16, Tile % 8, Hidden % 16]>(0),
        );
    } else {
        let rows = normalized.view().tile::<m![All / 8], { Tile::SIZE / 8 },
            m![All / 8 = { Tile::SIZE / 8 } # (All / 8), Hidden / 16, All % 8, Hidden % 16]>(
            start / 8);
        // SAFETY: the window's row groups are the wave's, in order.
        let rows = unsafe {
            rows.reshape::<Chip, m![Tile / 8 # (All / 8), Hidden / 16, Tile % 8, Hidden % 16]>()
        };
        rows.to_dm_view(&mut context.tdma, landed.view_mut());
    }
    landed
}

fn place<Hidden: AxisName, G: Group>(
    context: &mut Device,
    landed: &Wave<Hidden>,
    rows: &mut Landed<Hidden, G>,
    wave: usize,
) {
    context
        .main
        .begin(landed.view())
        .fetch::<m![Tile % 32 / 8, Tile / 32, Hidden % 64 / 16, Tile % 8], m![Hidden % 16]>()
        .collect::<m![Tile % 32 / 8, Tile / 32, Hidden % 64 / 16, Tile % 8], m![Hidden % 16]>()
        .commit_trim::<m![Hidden % 16]>()
        .commit_view(G::slot(rows.view_mut(), wave));
}

pub(crate) fn input<Hidden: AxisName, G: Group, All: AxisName>(
    context: &mut Device,
    normalized: &Normalized<Hidden, All>,
    starts: (usize, usize),
    groups: usize,
) -> Landed<Hidden, G> {
    let mut rows = DmTensor::new();
    let first = land_live(context, normalized, starts.0, groups);
    if const { G::SIZE > 1 } {
        let second = land(context, normalized, starts.0 + starts.1);
        place::<Hidden, G>(context, &first, &mut rows, 0);
        place::<Hidden, G>(context, &second, &mut rows, 1);
    } else {
        place::<Hidden, G>(context, &first, &mut rows, 0);
    }
    rows
}

/// The span rows a norm leaves, by groups of eight rows; the rows of a launch of at most eight
/// rows, or of a span's single wave, move alone.
impl<'r, Hidden: AxisName, Capacity: AxisName>
    Spill<Device, &'r Owned<Hidden, Span>, (Tiles<Spans<Capacity>>, usize)>
    for Normalized<Hidden, Capacity>
{
    fn spill(
        &mut self,
        context: &mut Device,
        normalized: &'r Owned<Hidden, Span>,
        (live, rows): (Tiles<Spans<Capacity>>, usize),
    ) {
        const { assert!(Capacity::SIZE.is_multiple_of(Span::SIZE)) };
        let span = self.view_mut().tile::<m![Capacity / 8], { Span::SIZE / 8 },
            m![Capacity / 8 = { Span::SIZE / 8 } #{!} (Capacity / 8), Hidden / 16, Capacity % 8,
                Hidden % 16]>(live.start() / 8);
        if rows <= 8 {
            // SAFETY: the row groups retain their order when split into the two cluster halves.
            let halves = unsafe {
                span.reshape::<Chip, m![Span / (Span / 2) #{!} (Capacity / (Span / 2)),
                    Span % (Span / 2) / 8, Hidden / 16, Span % 8, Hidden % 16]>(
                )
            };
            normalized.view().tile::<m![Span % (Span / 2)], 16,
                m![Span % (Span / 2) = 16 # (Span / 2), Hidden % 16]>(0)
                .to_hbm_view(&mut context.tdma, halves.tile::<m![Span % (Span / 2) / 8], 2,
                    m![Span / (Span / 2) #{!} (Capacity / (Span / 2)),
                        Span % (Span / 2) / 8 = 2 #{!} (Span % (Span / 2) / 8), Hidden / 16,
                        Span % 8, Hidden % 16]>(0));
        } else if live.retile::<Waves<Capacity>>().count() == 1 {
            // SAFETY: the row groups retain their order when split into the two cluster halves.
            let halves = unsafe {
                span.reshape::<Chip, m![Span / (Span / 2) #{!} (Capacity / (Span / 2)),
                    Span % (Span / 2) / 8, Hidden / 16, Span % 8, Hidden % 16]>(
                )
            };
            normalized.view().tile::<m![Span % (Span / 2)], 128,
                m![Span % (Span / 2) = 128 # (Span / 2), Hidden % 16]>(0)
                .to_hbm_view(&mut context.tdma, halves.tile::<m![Span % (Span / 2) / 8], 16,
                    m![Span / (Span / 2) #{!} (Capacity / (Span / 2)),
                        Span % (Span / 2) / 8 = 16 #{!} (Span % (Span / 2) / 8), Hidden / 16,
                        Span % 8, Hidden % 16]>(0));
        } else {
            normalized.view().to_hbm_view(&mut context.tdma, span);
        }
    }
}
