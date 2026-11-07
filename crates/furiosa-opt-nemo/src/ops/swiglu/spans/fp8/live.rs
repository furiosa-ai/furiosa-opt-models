use super::*;
use crate::Module;

pub(crate) struct Span<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, P> {
    pub(crate) normalized: &'a Owned<Hidden, Rows>,
    pub(crate) panels: (
        &'a Ahead<Hidden, Intermediate>,
        &'a Down<Hidden, Intermediate>,
    ),
    pub(crate) next: P,
    pub(crate) live: (usize, usize),
}

impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, P: Points<Device>>
    Module<(&mut Device, Span<'_, Hidden, Intermediate, Rows, P>)> for Mlp8<Hidden, Intermediate>
where
    super::super::super::Down<Intermediate>: AxisName,
{
    type Output = (Owned<Hidden, Rows>, P::Next);

    fn forward(
        &self,
        (context, input): (&mut Device, Span<'_, Hidden, Intermediate, Rows, P>),
    ) -> Self::Output {
        let (mlp, normalized, panels, next) = (self, input.normalized, input.panels, input.next);
        let (live, rows) = input.live;
        let channels = (
            mlp.channels::<Rows>(context, 0),
            mlp.channels::<Rows>(context, 1),
        );
        let down: DmTensor<f32, Chip, Cluster<Intermediate>, m![Hidden / 16], m![Hidden % 16]> =
            mlp.down.1.to_dm(&mut context.tdma);
        let ((gate, up), (second, third)) = gated::<Hidden, Intermediate, Rows>(
            context,
            mlp,
            normalized,
            (panels.0, (&channels.0, &channels.1)),
            (live, rows),
        );
        let mut products = Halves::<Intermediate, Rows>::new();
        let mut stores = HbmTensor::<bf16, Chip, Stores<Intermediate, Rows>>::new();
        if live <= 2 {
            halves::<Intermediate, Rows>(context, (&gate, &up), rows, &mut products);
            store_waves::<Intermediate, Rows>(context, &products, rows, &mut stores);
        }
        // SAFETY: the store's row digits are the span's rows in order.
        let product: HbmTensor<bf16, Chip, Chunks<Intermediate, Rows>> =
            unsafe { stores.reshape() };
        let (mut first, mut input1, mut input2) =
            (Activation::new(), Activation::new(), Activation::new());
        let (mut c0, mut c1, mut c2) = (Scale::new(), Scale::new(), Scale::new());
        let mut quantized = Spread::<Intermediate, Rows, f8e4m3>::new();
        let mut groups = SpreadScale::<Intermediate, Rows>::new();
        let (mut h1, mut h2) = (
            Staged::<Intermediate, Rows>::new(),
            Staged::<Intermediate, Rows>::new(),
        );
        let mut stored_scales = StagedScales::<Intermediate, Rows>::new();
        if live > 2 {
            let mut chunks =
                ChunkMax::<Intermediate, Rows, SpreadSlices<Intermediate, Rows>>::new();
            let activated = activate::<Intermediate, Rows>(context, &gate, rows);
            halves_panel::<Intermediate, Rows, 0>(context, (&activated, &up), &mut products);
            panel_maxima::<Intermediate, Rows, 0>(context, &products, &mut chunks);
            let broadcast = spread_scales::<Intermediate, Rows>(context, &chunks, &mut groups);
            quantize_spread::<Intermediate, Rows, 0>(
                context,
                &products,
                &broadcast,
                &mut quantized,
            );
            let mut h0 = Staged::<Intermediate, Rows>::new();
            stage_panel::<Intermediate, Rows>(context, &quantized, 0, &mut h0);
            copy_panel::<Hidden, Intermediate, Rows>(context, &h0, &mut first);
            halves_panel::<Intermediate, Rows, 1>(context, (&activated, &up), &mut products);
            halves_panel::<Intermediate, Rows, 2>(context, (&activated, &up), &mut products);
            panel_maxima::<Intermediate, Rows, 1>(context, &products, &mut chunks);
            panel_maxima::<Intermediate, Rows, 2>(context, &products, &mut chunks);
            let broadcast = spread_scales::<Intermediate, Rows>(context, &chunks, &mut groups);
            quantize_spread::<Intermediate, Rows, 1>(
                context,
                &products,
                &broadcast,
                &mut quantized,
            );
            quantize_spread::<Intermediate, Rows, 2>(
                context,
                &products,
                &broadcast,
                &mut quantized,
            );
            stage_panel::<Intermediate, Rows>(context, &quantized, 1, &mut h1);
            stage_panel::<Intermediate, Rows>(context, &quantized, 2, &mut h2);
        }
        let product0 =
            load_product::<Hidden, Intermediate, Rows>(context, &product, 0, (live, rows));
        let product1 =
            load_product::<Hidden, Intermediate, Rows>(context, &product, 1, (live, rows));
        let product2 =
            load_product::<Hidden, Intermediate, Rows>(context, &product, 2, (live, rows));
        let mut next = next.at(context);
        if live == 2 {
            next.between(context);
        }
        if live <= 2 {
            quantize_panel::<Hidden, Intermediate, Rows>(
                context,
                &product0,
                rows,
                (&mut first, &mut c0),
            );
        }
        let d0 =
            project_live::<Hidden, Intermediate, Rows>(context, (panels.1, &first), (live, rows));
        if live > 2 {
            stage_scales::<Intermediate, Rows>(context, &groups, &mut stored_scales);
            copy_panel::<Hidden, Intermediate, Rows>(context, &h1, &mut input1);
            copy_panel::<Hidden, Intermediate, Rows>(context, &h2, &mut input2);
            copy_scales::<Hidden, Intermediate, Rows>(context, &stored_scales, 0, &mut c0);
            copy_scales::<Hidden, Intermediate, Rows>(context, &stored_scales, 1, &mut c1);
            copy_scales::<Hidden, Intermediate, Rows>(context, &stored_scales, 2, &mut c2);
            next.between(context);
        }
        if live <= 2 {
            quantize_panel::<Hidden, Intermediate, Rows>(
                context,
                &product1,
                rows,
                (&mut input1, &mut c1),
            );
        }
        let d1 =
            project_live::<Hidden, Intermediate, Rows>(context, (&second, &input1), (live, rows));
        if live <= 2 {
            quantize_panel::<Hidden, Intermediate, Rows>(
                context,
                &product2,
                rows,
                (&mut input2, &mut c2),
            );
        }
        let d2 =
            project_live::<Hidden, Intermediate, Rows>(context, (&third, &input2), (live, rows));
        let (s0, s1, s2) = (
            panel_scale::<Hidden, Intermediate, Rows>(context, &c0),
            panel_scale::<Hidden, Intermediate, Rows>(context, &c1),
            panel_scale::<Hidden, Intermediate, Rows>(context, &c2),
        );
        let down: ChannelScales<Hidden, Intermediate> = context
            .sub
            .begin(down.view())
            .fetch::<m![Hidden % 16 / 8], m![Hidden % 8]>()
            .collect::<m![Hidden % 16 / 8], m![Hidden % 8]>()
            .to_vrf();
        let groups = (rows + 7) / 8;
        let d0 = rescale::<Hidden, Intermediate, Rows>(context, &d0, (&s0, &down), groups);
        let d1 = rescale::<Hidden, Intermediate, Rows>(context, &d1, (&s1, &down), groups);
        let d2 = rescale::<Hidden, Intermediate, Rows>(context, &d2, (&s2, &down), groups);
        let sums = sum::<Hidden, Intermediate, Rows>(context, (&d0, &d1), groups);
        let partial = sum::<Hidden, Intermediate, Rows>(context, (&sums, &d2), groups);
        let mut result: DmTensor<
            bf16,
            Chip,
            Cluster<Intermediate>,
            m![Hidden / 16],
            m![Rows % (Rows / 2), Hidden % 16],
        > = DmTensor::new();
        if rows <= 8 {
            let live_rows = || {
                partial.view().tile::<m![Rows % (Rows / 2)], 8,
                m![Rows / (Rows / 2), Rows % (Rows / 2) = 8 # (Rows / 2), Hidden % 16]>(0)
            };
            let half = |context: &mut Device, indices| {
                live_rows().asymmetric_cluster_slice::<
                    m![Rows / (Rows / 2)],
                    m![Rows % (Rows / 2) = 8, Hidden % 16],
                >(&mut context.sub, indices)
            };
            let (send, own) = (half(context, &[1, 0]), half(context, &[0, 1]));
            let exchanged = send.view().cluster_swap().to_dm(&mut context.tdma);
            next.between(context);
            phases::add_live::<m![Rows % (Rows / 2) = 8], m![Rows % (Rows / 2) = 8, Hidden % 16 / 8],
                m![1], m![Hidden % 8], _, _, _, _, _>(context, (&own, &exchanged),
                    (result.view_mut().tile::<m![Rows % (Rows / 2)], 8,
                        m![Rows % (Rows / 2) = 8 #{!} (Rows / 2), Hidden % 16]>(0), rows));
        } else {
            let half = |context: &mut Device, indices| {
                partial.view().asymmetric_cluster_slice::<
                    m![Rows / (Rows / 2)],
                    m![Rows % (Rows / 2), Hidden % 16],
                >(&mut context.sub, indices)
            };
            let (send, own) = (half(context, &[1, 0]), half(context, &[0, 1]));
            let exchanged = send.view().cluster_swap().to_dm(&mut context.tdma);
            if live == 1 {
                next.between(context);
            }
            let half = Rows::SIZE / 2;
            let reach = rows + rows / (half + 1) * half - rows / (half + 1) * rows;
            phases::add_live::<
                m![Rows % (Rows / 2) / 8],
                m![Rows % (Rows / 2) / 8],
                m![Rows % 8, Hidden % 16 / 8],
                m![Hidden % 8],
                _,
                _,
                _,
                _,
                _,
            >(
                context,
                (&own, &exchanged),
                (result.view_mut(), reach.div_ceil(8)),
            );
        }
        // SAFETY: extraction and exchange give cluster c the complete sum for row half c.
        (unsafe { result.reshape() }, next)
    }
}
