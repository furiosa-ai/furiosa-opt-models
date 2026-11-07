use super::*;

impl<Hidden: AxisName, QWidth: AxisName, Capacity: AxisName, Reach: AxisName>
    Module<(&mut Device, Land<'_, QWidth, Capacity, Reach>)> for Output8<Hidden, QWidth>
{
    type Output = Landing8<Hidden, QWidth, Capacity>;

    fn forward(
        &self,
        (context, input): (&mut Device, Land<'_, QWidth, Capacity, Reach>),
    ) -> Self::Output {
        let (attended, start, (waves, rows)) = (input.attended, input.start, input.live);
        let mut landing = Landing8::new();
        if rows <= 8 {
            // SAFETY: the first eight memory rows are the first eight rows, in order.
            let group = unsafe {
                attended
                    .view()
                    .tile::<m![Reach], 8, m![Reach = 8 # Reach, QWidth]>(start)
                    .reshape::<Chip, m![Capacity % 8 # Reach, QWidth]>()
            };
            group.to_dm_view(&mut context.tdma, landing.view_mut()
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
            pair.to_dm_view(&mut context.tdma, landing.view_mut()
                .tile::<m![Capacity % (Capacity / 4)], 16,
                    m![Capacity % (Capacity / 4) = 16 #{!} 128, QWidth % 128]>(0));
        } else if waves == 1 {
            land_wave::<Hidden, QWidth, Capacity, Reach>(
                context,
                attended,
                &mut landing,
                (start, 0),
            );
        } else {
            land_wave::<Hidden, QWidth, Capacity, Reach>(
                context,
                attended,
                &mut landing,
                (start, 0),
            );
            land_wave::<Hidden, QWidth, Capacity, Reach>(
                context,
                attended,
                &mut landing,
                (start, 1),
            );
            land_wave::<Hidden, QWidth, Capacity, Reach>(
                context,
                attended,
                &mut landing,
                (start, 2),
            );
            land_wave::<Hidden, QWidth, Capacity, Reach>(
                context,
                attended,
                &mut landing,
                (start, 3),
            );
        }
        landing
    }
}

pub(super) fn land_wave<Hidden: AxisName, QWidth: AxisName, Capacity: AxisName, Reach: AxisName>(
    context: &mut Device,
    attended: &HbmTensor<bf16, Chip, m![Reach, QWidth]>,
    landing: &mut Landing8<Hidden, QWidth, Capacity>,
    (start, wave): (usize, usize),
) {
    wave_rows::<QWidth, Capacity, Reach>(attended, start + wave * 128).to_dm_view(&mut context.tdma, landing.view_mut()
        .tile::<m![Capacity % (Capacity / 4)], 32,
            m![Capacity % (Capacity / 4) = 32 #{!} 128, QWidth % 128]>(32 * wave));
}
