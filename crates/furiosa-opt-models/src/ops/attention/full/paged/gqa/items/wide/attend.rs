use super::*;
use crate::Module;

/// Attends each wide row of `items` over its history in `pool` into `attended`, 16 columns a wave
/// under the last wave's chain; `boundary` runs `guests` times once the last history is in flight.
impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Waves: AxisName,
    Rows: AxisName,
    H,
    S,
    Env: Copy,
>
    Module<
        (
            (
                &HeadRows<Rows, KvHeads, Groups, Head>,
                &mut HeadRows<Rows, KvHeads, Groups, Head>,
            ),
            (
                &Wide<KvHeads, Head, Waves>,
                Shape,
                (&Gather, &Blocks<KvHeads, Head>),
            ),
            ((&H, &mut S), usize),
        ),
        Env,
        &mut (),
    > for Attention<KvHeads, Groups, Head>
where
    H: for<'d, 's> Module<&'s mut S, Env, &'d mut (), Output = ()>,
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        ((query, attended), (items, shape, pool), ((boundary, state), guests)): (
            (
                &HeadRows<Rows, KvHeads, Groups, Head>,
                &mut HeadRows<Rows, KvHeads, Groups, Head>,
            ),
            (
                &Wide<KvHeads, Head, Waves>,
                Shape,
                (&Gather, &Blocks<KvHeads, Head>),
            ),
            ((&H, &mut S), usize),
        ),
    ) {
        // Wave `wave`'s history, in flight.
        let history = |context: &mut Device, wave: usize| -> Self::WideInDm {
            let tables = &items.tables;
            let indices = &Tables::wave(&tables.entries, wave);
            let read = Read {
                pool: pool.1,
                indices,
                output: PhantomData,
            };
            let lengths = &Tables::wave(&tables.lengths, wave);
            let prefix = Prefix {
                read,
                lengths,
                offset: 0,
                sparse: PhantomData::<m![Time]>,
            };
            pool.0.forward(&mut *context, env, &mut (), prefix)
        };
        // Pair `pair`'s queries written in their half of `queries`: a column's query puts one
        // (head, group) row on a token slice, which its chain spreads over the column on chip.
        let to_dm = |context: &mut Device,
                     (queries, pair): (&mut Self::WideQueries<Waves>, usize)| {
            let rows = Tables::pair(&items.tables.queries, pair);
            query
                .gather::<m![Rows], m![KvHeads, Groups, Head]>()
                .by_byte_offsets(&rows)
                .to_dm_view(
                    &mut context.tdma,
                    queries
                        .view_mut()
                        .tile::<m![Waves % 4 / 2], 1, m![1 #{!} 2, Waves % 2, Head]>(pair % 2),
                );
        };
        // Pair `pair`'s rounded rows stored: a group's columns write its row with the same bytes,
        // a column of no item its own sink.
        let store = |context: &mut Device,
                     pair: usize,
                     rounded: &Self::WideRounded<Waves>,
                     attended: HbmTensorViewMut<
            '_,
            bf16,
            Chip,
            m![Rows, KvHeads, Groups, Head],
        >| {
            // SAFETY: each slice of a group holds its item's rows, so each column's first does.
            let leads: DmTensorView<
                '_,
                bf16,
                Chip,
                Self::ItemCluster,
                ColumnSlices<m![1 # 16]>,
                m![Waves % 2, KvHeads % ({ Local<KvHeads> }), Groups, Head],
            > = unsafe { rounded.view().reshape() };
            let outputs = Tables::pair(&items.tables.outputs, pair);
            attended
                .scatter::<m![Rows], m![KvHeads, Groups, Head]>()
                .by_byte_offsets(outputs.view())
                .from_dm_view(&mut context.tdma, leads);
        };
        // Wave `wave`'s items' f32 outputs and log-sum-exps kept in their slots, which only a merge
        // reads, so a launch of one-item rows skips them.
        let keep = |context: &mut Device,
                    wave: usize,
                    (output, mass, top): Self::WideKept,
                    (partials, scales): &mut (
            Self::WidePartials<Columns>,
            Self::WideScales<Columns>,
        )| {
            let tables = &items.tables;
            for _ in 0..shape.merges.min(1) {
                let lse: Self::WideLocal = context
                    .main
                    .begin(mass.view())
                    .fetch::<m![KvHeads % ({ Local<KvHeads> }), Groups], m![1 # 8]>()
                    .collect::<m![KvHeads % ({ Local<KvHeads> }), Groups], m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_trim::<m![1 # 4]>()
                    .vector_fp_unary(FpUnaryOp::Log)
                    .vector_fp_binary(FpBinaryOp::AddF, &top)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                // SAFETY: each slice of a group holds its item's, so each column's first does.
                let (output, lse): (
                    DmTensorView<
                        '_,
                        f32,
                        Chip,
                        Self::ItemCluster,
                        ColumnSlices<m![1 # 16]>,
                        m![KvHeads % ({ Local<KvHeads> }), Groups, Head],
                    >,
                    DmTensorView<
                        '_,
                        f32,
                        Chip,
                        Self::ItemCluster,
                        ColumnSlices<m![1 # 16]>,
                        m![KvHeads % ({ Local<KvHeads> }), Groups, Lane],
                    >,
                ) = unsafe { (output.view().reshape(), lse.view().reshape()) };
                let (to, by) = (
                    Tables::wave(&tables.partials, wave),
                    Tables::wave(&tables.scales, wave),
                );
                let partials = partials
                    .view_mut()
                    .scatter::<m![Columns], m![KvHeads, Groups, Head]>();
                partials
                    .by_byte_offsets(to.view())
                    .from_dm_view(&mut context.tdma, output);
                let scales = scales
                    .view_mut()
                    .scatter::<m![Columns], m![KvHeads, Groups, Lane]>();
                scales
                    .by_byte_offsets(by.view())
                    .from_dm_view(&mut context.tdma, lse);
            }
        };
        // Wave `wave`'s chain: every column's item scored and normalized over its group, its rows
        // rounded into their half of `rounded`; what a merge keeps of it returned.
        let attend =
            |context: &mut Device,
             (shape, (keys, limits)): (Shape, (&Self::WideKeys, &Self::WideLimits<Waves>)),
             ((queries, in_dm), wave): ((&Self::WideQueries<Waves>, &Self::WideInDm), usize),
             rounded: &mut Self::WideRounded<Waves>|
             -> Self::WideKept {
                let heads: Self::WideDm<bf16, m![{ Self::WideRows }, Head]> = Self::spread::<
                    m![Head / 16],
                    m![Head / 16, KvHeads % ({ Local<KvHeads> }), Groups],
                    _,
                    _,
                    _,
                    _,
                    _,
                >(
                    context,
                    (queries, wave),
                    SwitchConfig::CustomBroadcast { ring_size: 16 },
                );
                // A column's limit, which its first slice holds, spread over its token slices.
                let limit: Self::WideLimit = context
                    .main
                    .begin(
                        limits
                            .view()
                            .tile::<m![Waves], 1, m![Waves = 1 # Waves, 1 # 8]>(wave),
                    )
                    .fetch::<m![1], m![1 # 8]>()
                    .switch::<ColumnSlices<m![Block]>, m![1]>(SwitchConfig::CustomBroadcast {
                        ring_size: 16,
                    })
                    .collect::<m![1], m![1 # 8]>()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                let limit: Self::WideVrf<i32, m![1 # 8]> = context
                    .sub
                    .begin(limit.view())
                    .fetch::<m![1], m![1 # 8]>()
                    .collect::<m![1], m![1 # 8]>()
                    .to_vrf();
                self.attend::<m![Groups], m![Time / 8], _, _, _, _, _, _>(
                    context,
                    ((&heads, &limit), (keys, in_dm)),
                    (shape.units, shape.depth),
                    (wave, rounded),
                )
            };
        // Waves after the first; the whole layer's work of a domain without rows runs no trip.
        let (any, rest) = (shape.waves.min(1), shape.waves.max(1) - 1);
        for _ in 0..any {
            let mut kept = (
                <Self::WidePartials<Columns>>::new(),
                <Self::WideScales<Columns>>::new(),
            );
            let mut rounded = <Self::WideRounded<Waves>>::new();
            let mut queries = <Self::WideQueries<Waves>>::new();
            // Only while the layer attends: a launch-long table on chip would hold its memory
            // through every other module.
            let keys: Self::WideKeys = items.keys.to_dm(&mut context.tdma);
            // Every wave's at once: a transfer queued between two histories delays the second.
            let limits: Self::WideLimits<Waves> = items.limits.to_dm(&mut context.tdma);
            to_dm(context, (&mut queries, 0));
            let mut in_dm = history(context, 0);
            let chain =
                |context: &mut Device,
                 queries: &Self::WideQueries<Waves>,
                 in_dm: &Self::WideInDm,
                 (wave, rounded): (usize, &mut Self::WideRounded<Waves>)| {
                    attend(
                        context,
                        (shape, (&keys, &limits)),
                        ((queries, in_dm), wave),
                        rounded,
                    )
                };
            // Two waves a trip into alternating buffers, each under the other's chain; a history
            // issues once the last arrives, behind the next pair's queries or this pair's rows.
            for pair in 0..rest / 2 {
                let next = history(context, 2 * pair + 1);
                let first = chain(context, &queries, &in_dm, (2 * pair, &mut rounded));
                to_dm(context, (&mut queries, pair + 1));
                keep(context, 2 * pair, first, &mut kept);
                in_dm = history(context, 2 * pair + 2);
                let second = chain(context, &queries, &next, (2 * pair + 1, &mut rounded));
                store(context, pair, &rounded, attended.view_mut());
                keep(context, 2 * pair + 1, second, &mut kept);
            }
            // The weight queues behind the last wave's history and the stores issued before it: a
            // store behind the weight would hold the chain that rounds into the stored rows.
            for _ in 0..rest % 2 {
                let other = history(context, rest);
                let first = chain(context, &queries, &in_dm, (rest - 1, &mut rounded));
                keep(context, rest - 1, first, &mut kept);
                for _ in 0..guests {
                    boundary.forward(&mut *context, env, &mut (), &mut *state);
                }
                let second = chain(context, &queries, &other, (rest, &mut rounded));
                store(context, rest / 2, &rounded, attended.view_mut());
                keep(context, rest, second, &mut kept);
            }
            // A last wave alone: its pair's second wave is past the last, whose rows go to the
            // sinks.
            for _ in 0..1 - rest % 2 {
                for _ in 0..guests {
                    boundary.forward(&mut *context, env, &mut (), &mut *state);
                }
                let only = chain(context, &queries, &in_dm, (rest, &mut rounded));
                store(context, rest / 2, &rounded, attended.view_mut());
                keep(context, rest, only, &mut kept);
            }
            for tile in 0..shape.merges {
                let merge = (&kept.0, &kept.1);
                self.forward(
                    &mut *context,
                    env,
                    &mut (),
                    (items, merge, tile, attended.view_mut()),
                );
            }
        }
    }
}
