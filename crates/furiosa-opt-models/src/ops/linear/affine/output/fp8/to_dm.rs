use super::*;
use crate::Op;
use crate::ops::linear::affine::output;
use core::marker::PhantomData;

/// The attended rows written where a projection of `Hidden` features reads them, whatever its
/// weights.
pub(crate) struct ToDm<Hidden>(pub(crate) PhantomData<Hidden>);

impl<Hidden: AxisName, QWidth: AxisName, Capacity: AxisName, Reach: AxisName>
    Op<output::ToDm<'_, QWidth, Capacity, Reach>> for ToDm<Hidden>
{
    type Output = InDm<Hidden, QWidth, Capacity>;

    fn forward(
        &self,
        context: &mut Device,
        input: output::ToDm<'_, QWidth, Capacity, Reach>,
    ) -> Self::Output {
        let (attended, start, (waves, rows)) = (input.attended, input.start, input.live);
        let mut in_dm = InDm::new();
        if rows <= 8 {
            // SAFETY: the first eight memory rows are the first eight rows, in order.
            let group = unsafe {
                attended
                    .view()
                    .tile::<m![Reach], 8, m![Reach = 8 # Reach, QWidth]>(start)
                    .reshape::<Chip, m![Capacity % 8 # Reach, QWidth]>()
            };
            group.to_dm_view(&mut context.tdma, in_dm.view_mut()
                .slice_tile::<m![Capacity / (Capacity / 4)], 1, m![Hidden / 1024,
                    Capacity / (Capacity / 4) = 1 #{!} 4, QWidth % 2048 / 128]>(0)
                .tile::<m![Capacity % (Capacity / 4)], 8,
                    m![Capacity % (Capacity / 4) = 8 #{!} 128, QWidth % 128]>(0));
        } else if rows <= 64 {
            // SAFETY: the first 64 memory rows are the first two waves' eight-row groups.
            let pair = unsafe {
                attended
                    .view()
                    .tile::<m![Reach], 64, m![Reach = 64 # Reach, QWidth]>(start)
                    .reshape::<Chip, m![Capacity % 16 / 8 # (Reach / 32),
                        Capacity / (Capacity / 4), Capacity % 8, QWidth]>()
            };
            pair.to_dm_view(&mut context.tdma, in_dm.view_mut()
                .tile::<m![Capacity % (Capacity / 4)], 16,
                    m![Capacity % (Capacity / 4) = 16 #{!} 128, QWidth % 128]>(0));
        } else if waves == 1 {
            Op::forward(&Tile, &mut *context, (attended, &mut in_dm, (start, 0)));
        } else {
            Op::forward(&Tile, &mut *context, (attended, &mut in_dm, (start, 0)));
            Op::forward(&Tile, &mut *context, (attended, &mut in_dm, (start, 1)));
            Op::forward(&Tile, &mut *context, (attended, &mut in_dm, (start, 2)));
            Op::forward(&Tile, &mut *context, (attended, &mut in_dm, (start, 3)));
        }
        in_dm
    }
}

/// Wave `wave` of the attended rows from `start`, written in its quarter of the destination rows.
struct Tile;

impl<'a, Hidden: AxisName, QWidth: AxisName, Capacity: AxisName, Reach: AxisName>
    Op<(
        &'a HbmTensor<bf16, Chip, m![Reach, QWidth]>,
        &'a mut InDm<Hidden, QWidth, Capacity>,
        (usize, usize),
    )> for Tile
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (attended, in_dm, (start, wave)): (
            &'a HbmTensor<bf16, Chip, m![Reach, QWidth]>,
            &'a mut InDm<Hidden, QWidth, Capacity>,
            (usize, usize),
        ),
    ) {
        let rows = |start: usize| {
            // SAFETY: the wave's memory rows are its eight-row groups of every quarter, in order.
            unsafe {
                attended
                    .view()
                    .tile::<m![Reach], 128, m![Reach = 128 # Reach, QWidth]>(start)
                    .reshape::<Chip, m![Capacity % 32 / 8 # (Reach / 32), Capacity / (Capacity / 4),
                        Capacity % 8, QWidth]>()
            }
        };
        rows(start + wave * 128).to_dm_view(&mut context.tdma, in_dm.view_mut()
            .tile::<m![Capacity % (Capacity / 4)], 32,
                m![Capacity % (Capacity / 4) = 32 #{!} 128, QWidth % 128]>(32 * wave));
    }
}
