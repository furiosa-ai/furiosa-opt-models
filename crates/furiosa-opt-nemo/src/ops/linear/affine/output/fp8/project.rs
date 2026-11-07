use super::*;

impl<Hidden: AxisName, QWidth: AxisName, Capacity: AxisName, P: Points<Device>>
    Module<(&mut Device, Project<'_, Hidden, QWidth, Capacity, P>)> for Staged8<Hidden, QWidth>
{
    type Output = (Owned<Hidden, Capacity>, <P::Next as Points<Device>>::Next);

    fn forward(
        &self,
        (context, input): (&mut Device, Project<'_, Hidden, QWidth, Capacity, P>),
    ) -> Self::Output {
        let (landing, points, (_, rows)) = (input.landing, input.points, input.live);
        let (weight, channels) = &self.weight;
        let mut source = Source8::new();
        let mut scales = Scales8::new();
        let resident = resident::<Hidden, QWidth>(context, weight);
        let waves = (rows + 31) / 32;
        quantize_first::<Hidden, QWidth, Capacity>(context, landing, (&mut source, &mut scales));
        let (whole, tail) = (waves / 4, waves % 4);
        let (first, single) = ((whole + 3) / 4, 1 / (whole + 1));
        let mut partial = waves::Partial::new();
        if first > 0 {
            contract_wave::<Hidden, QWidth, Capacity>(
                context,
                (&resident, &source),
                (&mut partial, 0),
            );
        } else {
            contract::<Hidden, QWidth, Capacity>(context, (&resident, &source), (&mut partial, 0));
        }
        let points = points.at(context);
        if waves > 4 {
            quantize_all::<Hidden, QWidth, Capacity>(
                context,
                landing,
                (&mut source, &mut scales),
                waves,
            );
        }
        for block in 0..3 * whole / 4 {
            contract_wave::<Hidden, QWidth, Capacity>(
                context,
                (&resident, &source),
                (&mut partial, block + 1),
            );
        }
        for wave in 0..(tail + 3 + first) % 4 {
            contract::<Hidden, QWidth, Capacity>(
                context,
                (&resident, &source),
                (&mut partial, 4 * whole + single + wave),
            );
        }
        let scales = gather::<Hidden, QWidth, Capacity>(context, &scales);
        let partial =
            rescale::<Hidden, QWidth, Capacity>(context, &partial, (&scales, channels), waves);
        let half = |context: &mut Device, indices| -> waves::Half<Hidden, QWidth, Capacity> {
            partial
                .view()
                .asymmetric_cluster_slice::<m![Capacity % (Capacity / 4) / 64], _>(
                    &mut context.sub,
                    indices,
                )
        };
        let (send, own) = (half(context, &[1, 0]), half(context, &[0, 1]));
        let few = (64 / rows.max(1)).min(1);
        let mut received = waves::Half::new();
        for wave in 0..waves * few {
            send.view().tile::<m![Capacity % 64 / 8], 1, m![Capacity % 64 / 8 = 1 # 8,
                Capacity / (Capacity / 4), Capacity % 8, Hidden % 16]>(wave)
                .cluster_swap()
                .to_dm_view(&mut context.tdma, received.view_mut().tile::<m![Capacity % 64 / 8], 1,
                    m![Capacity % 64 / 8 = 1 #{!} 8, Capacity / (Capacity / 4), Capacity % 8,
                        Hidden % 16]>(wave));
        }
        let whole = (rows / 129).min(1);
        for _ in 0..(rows / 65).min(1) * (1 / (whole + 1)) {
            send.view().tile::<m![Capacity % 64 / 32], 1, m![Capacity % 64 / 32 = 1 # 2,
                Capacity % 32 / 8, Capacity / (Capacity / 4), Capacity % 8, Hidden % 16]>(0)
                .cluster_swap()
                .to_dm_view(&mut context.tdma, received.view_mut().tile::<m![Capacity % 64 / 32], 1,
                    m![Capacity % 64 / 32 = 1 #{!} 2, Capacity % 32 / 8, Capacity / (Capacity / 4),
                        Capacity % 8, Hidden % 16]>(0));
        }
        for _ in 0..whole {
            send.view()
                .cluster_swap()
                .to_dm_view(&mut context.tdma, received.view_mut());
        }
        let points = points.at(context);
        let (small, big) = ((rows + 767) / 512 % 2, (rows + 255) / 512);
        let mut summed = waves::Half::<Hidden, QWidth, Capacity>::new();
        phases::add_live::<
            m![Capacity % 64 / 8],
            m![Capacity % 64 / 8, Capacity / (Capacity / 4)],
            m![Capacity % 8, Hidden % 16 / 8],
            m![Hidden % 8],
            _,
            _,
            _,
            _,
            _,
        >(
            context,
            (&own, &received),
            (summed.view_mut(), waves * small + 8 * big),
        );
        // SAFETY: each cluster owns its physical row half in order.
        (unsafe { summed.reshape() }, points)
    }
}
