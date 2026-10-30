use super::activation::InDm;
use super::*;
use crate::Op;
use core::marker::PhantomData;

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

/// A group's normalized rows from `starts.0`, its second wave `starts.1` rows on, written where the
/// quantization reads them; a one-wave group writes its first `groups` row groups alone.
pub(crate) struct ToDm<G>(pub(crate) PhantomData<G>);

impl<'a, Hidden: AxisName, G: Group, All: AxisName>
    Op<(&'a Normalized<Hidden, All>, (usize, usize), usize)> for ToDm<G>
{
    type Output = InDm<Hidden, G>;

    fn forward(
        &self,
        context: &mut Device,
        (normalized, starts, groups): (&'a Normalized<Hidden, All>, (usize, usize), usize),
    ) -> Self::Output {
        let to_dm = |context: &mut Device, start: usize| -> Wave<Hidden> {
            let rows = normalized
                .view()
                .tile::<m![All / 8], { Tile::SIZE / 8 }, m![
                    All / 8 = { Tile::SIZE / 8 } # (All / 8),
                    Hidden / 16,
                    All % 8,
                    Hidden % 16
                ]>(start / 8);
            // SAFETY: the window's row groups are the wave's, in order.
            let rows = unsafe {
                rows.reshape::<Chip, m![Tile / 8 # (All / 8), Hidden / 16, Tile % 8, Hidden % 16]>()
            };
            rows.to_dm(&mut context.tdma)
        };
        let live_to_dm = |context: &mut Device, start: usize, groups: usize| -> Wave<Hidden> {
            let mut in_dm = Wave::new();
            if groups == 1 {
                let group = normalized
                    .view()
                    .tile::<
                        m![All / 8],
                        1,
                        m![All / 8 = 1 # (All / 8), Hidden / 16, All % 8, Hidden % 16],
                    >(
                        start / 8,
                    );
                // SAFETY: the window's row group is the wave's first, in order.
                let group = unsafe {
                    group.reshape::<
                        Chip,
                        m![All / 8 = 1 # (All / 8), Hidden / 16, Tile % 8, Hidden % 16],
                    >()
                };
                group.to_dm_view(
                    &mut context.tdma,
                    in_dm
                        .view_mut()
                        .tile::<m![Tile % 32 / 8], 1, m![Tile % 32 / 8 = 1 #{!} 4, Tile / 32,
                        Hidden % 64 / 16, Tile % 8, Hidden % 16]>(0)
                        .tile::<m![Tile / 32], 1, m![Tile % 32 / 8 = 1 #{!} 4, Tile / 32 = 1 #{!} 4,
                        Hidden % 64 / 16, Tile % 8, Hidden % 16]>(0),
                );
            } else {
                let rows = normalized
                    .view()
                    .tile::<m![All / 8], { Tile::SIZE / 8 }, m![
                        All / 8 = { Tile::SIZE / 8 } # (All / 8),
                        Hidden / 16,
                        All % 8,
                        Hidden % 16
                    ]>(start / 8);
                // SAFETY: the window's row groups are the wave's, in order.
                let rows = unsafe {
                    rows.reshape::<
                        Chip,
                        m![Tile / 8 # (All / 8), Hidden / 16, Tile % 8, Hidden % 16],
                    >()
                };
                rows.to_dm_view(&mut context.tdma, in_dm.view_mut());
            }
            in_dm
        };
        let place = |context: &mut Device,
                     in_dm: &Wave<Hidden>,
                     rows: &mut InDm<Hidden, G>,
                     wave: usize| {
            context
                .main
                .begin(in_dm.view())
                .fetch::<
                    m![Tile % 32 / 8, Tile / 32, Hidden % 64 / 16, Tile % 8],
                    m![Hidden % 16],
                >()
                .collect::<
                    m![Tile % 32 / 8, Tile / 32, Hidden % 64 / 16, Tile % 8],
                    m![Hidden % 16],
                >()
                .commit_trim::<m![Hidden % 16]>()
                .commit_view(G::slot(rows.view_mut(), wave));
        };
        let mut rows = DmTensor::new();
        let first = live_to_dm(context, starts.0, groups);
        if const { G::SIZE > 1 } {
            let second = to_dm(context, starts.0 + starts.1);
            place(context, &first, &mut rows, 0);
            place(context, &second, &mut rows, 1);
        } else {
            place(context, &first, &mut rows, 0);
        }
        rows
    }
}
