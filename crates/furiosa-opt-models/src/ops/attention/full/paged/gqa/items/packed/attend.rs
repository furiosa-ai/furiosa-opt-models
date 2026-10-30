use super::*;
use crate::Module;

/// Attends each packed row of `items` over its history in `pool` into `attended`, 64 columns a wave
/// under the last wave's chain; `boundary` runs `guests` times once the last history is in flight.
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, Rows: AxisName, H, S, Env: Copy>
    Module<
        (
            (
                &HeadRows<Rows, KvHeads, Groups, Head>,
                &mut HeadRows<Rows, KvHeads, Groups, Head>,
            ),
            (
                &Packed<KvHeads, Head>,
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
                &Packed<KvHeads, Head>,
                Shape,
                (&Gather, &Blocks<KvHeads, Head>),
            ),
            ((&H, &mut S), usize),
        ),
    ) {
        // Wave `wave`'s history, in flight: a column's block rounds, each block's 16 tokens four
        // on each of its token slices.
        let history = |context: &mut Device, wave: usize| -> Self::PackedInDm {
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
                sparse: PhantomData::<m![Round]>,
            };
            let in_dm: Self::PackedDm<
                bf16,
                m![Round, Block % 4, KeyValue, KvHeads % ({ Local<KvHeads> }), Head],
            > = pool.0.forward(&mut *context, env, &mut (), prefix);
            // SAFETY: a slice's block rounds of four tokens each are its 64 slots in order.
            unsafe { in_dm.reshape() }
        };
        // Pair `pair`'s queries and lane limits into their halves of `lanes`: a column's two rows
        // put a head on a token slice for its chain to spread, and its limits on each slice.
        let to_dm =
            |context: &mut Device,
             ((queries, limits), pair): (&mut Self::PackedLanes<Waves>, usize)| {
                let rows = Tables::pair(&items.tables.queries, pair);
                query
                    .gather::<m![Rows], m![KvHeads, Groups, Head]>()
                    .by_byte_offsets(&rows)
                    .to_dm_view(
                        &mut context.tdma,
                        queries
                            .view_mut()
                            .tile::<
                                m![Waves % 4 / 2],
                                1,
                                m![1 #{!} 2, Waves % 2, Pack, Groups, Head],
                            >(
                                pair % 2,
                            ),
                    );
                Tables::pair(&items.limits, pair).view().to_dm_view(
                    &mut context.tdma,
                    limits
                        .view_mut()
                        .tile::<m![Waves % 4 / 2], 1, m![1 #{!} 2, Waves % 2, Pack, 1 # 8]>(
                            pair % 2,
                        ),
                );
            };
        // Pair `pair`'s rounded rows stored: a group's first column writes its lanes' rows, every
        // other column lane its own sink.
        let store = |context: &mut Device,
                     pair: usize,
                     rounded: &Self::PackedRounded<Waves>,
                     attended: HbmTensorViewMut<
            '_,
            bf16,
            Chip,
            m![Rows, KvHeads, Groups, Head],
        >| {
            // SAFETY: each slice of a group holds its lanes' rows, so each column's first does.
            let leads: DmTensorView<
                '_,
                bf16,
                Chip,
                Self::ItemCluster,
                m![Column, 1 # 4],
                m![Waves % 2, Pack, KvHeads % ({ Local<KvHeads> }), Groups, Head],
            > = unsafe { rounded.view().reshape() };
            let outputs = Tables::pair(&items.tables.outputs, pair);
            attended
                .scatter::<m![Rows], m![KvHeads, Groups, Head]>()
                .by_byte_offsets(outputs.view())
                .from_dm_view(&mut context.tdma, leads);
        };
        // Wave `wave`'s rounded rows stored, the last wave of a launch and the first of its pair: a
        // group's first column writes its lanes' rows, every other column lane its own sink.
        let last =
            |context: &mut Device,
             wave: usize,
             rounded: &Self::PackedRounded<Waves>,
             attended: HbmTensorViewMut<'_, bf16, Chip, m![Rows, KvHeads, Groups, Head]>| {
                // SAFETY: each slice of a group holds its lanes' rows, so each column's first does.
                let leads: DmTensorView<
                    '_,
                    bf16,
                    Chip,
                    Self::ItemCluster,
                    m![Column, 1 # 4],
                    m![Waves % 2, Pack, KvHeads % ({ Local<KvHeads> }), Groups, Head],
                > = unsafe { rounded.view().reshape() };
                let lead =
                    leads.tile::<
                        m![Waves % 2],
                        1,
                        m![1 # 2, Pack, KvHeads % ({ Local<KvHeads> }), Groups, Head],
                    >(wave % 2);
                let outputs = Tables::wave(&items.tables.outputs, wave);
                attended
                    .scatter::<m![Rows], m![KvHeads, Groups, Head]>()
                    .by_byte_offsets(outputs.view())
                    .from_dm_view(&mut context.tdma, lead);
            };
        // Wave `wave`'s chain: both lanes of every group scored and normalized over the group,
        // their rows rounded into their half of `rounded`.
        let attend = |context: &mut Device,
                      (shape, keys): (Shape, &Self::PackedKeys),
                      (((queries, limits), in_dm), wave): (
            (&Self::PackedLanes<Waves>, &Self::PackedInDm),
            usize,
        ),
                      rounded: &mut Self::PackedRounded<Waves>| {
            let heads: Self::PackedDm<bf16, m![{ Self::PackedRows }, Head]> = Self::spread::<
                m![Pack, Groups, Head / 16],
                m![Pack, Groups, Head / 16, KvHeads % ({ Local<KvHeads> })],
                _,
                _,
                _,
                _,
                _,
            >(
                context,
                (queries, wave),
                SwitchConfig::CustomBroadcast { ring_size: 4 },
            );
            let limit: Self::PackedVrf<i32, m![Pack, 1 # 8]> = context
                .sub
                .begin(
                    limits
                        .view()
                        .tile::<m![Waves % 4], 1, m![1 # 4, Pack, 1 # 8]>(wave % 4),
                )
                .fetch::<m![Pack], m![1 # 8]>()
                .collect::<m![Pack], m![1 # 8]>()
                .to_vrf();
            self.attend::<m![Pack, Groups], m![Pack, Slot / 8], _, _, _, _, _, _>(
                context,
                ((&heads, &limit), (keys, in_dm)),
                (shape.units, shape.depth),
                (wave, rounded),
            );
        };
        // Waves after the first; the whole layer's work of a domain without rows runs no trip.
        let (any, rest) = (shape.waves.min(1), shape.waves.max(1) - 1);
        for _ in 0..any {
            let mut rounded = <Self::PackedRounded<Waves>>::new();
            let mut lanes = (
                <Self::PackedQueries<Waves>>::new(),
                <Self::PackedLimits<Waves>>::new(),
            );
            // Only while the layer attends: a launch-long table on chip would hold its memory
            // through every other module.
            let keys: Self::PackedKeys = items.keys.to_dm(&mut context.tdma);
            to_dm(context, (&mut lanes, 0));
            let mut in_dm = history(context, 0);
            let chain =
                |context: &mut Device,
                 lanes: &Self::PackedLanes<Waves>,
                 in_dm: &Self::PackedInDm,
                 (wave, rounded): (usize, &mut Self::PackedRounded<Waves>)| {
                    attend(context, (shape, &keys), ((lanes, in_dm), wave), rounded)
                };
            // Two waves a trip into alternating buffers, each under the other's chain; a history
            // issues once the last arrives, behind the next pair's queries or this pair's rows.
            for pair in 0..rest / 2 {
                let next = history(context, 2 * pair + 1);
                chain(context, &lanes, &in_dm, (2 * pair, &mut rounded));
                to_dm(context, (&mut lanes, pair + 1));
                in_dm = history(context, 2 * pair + 2);
                chain(context, &lanes, &next, (2 * pair + 1, &mut rounded));
                store(context, pair, &rounded, attended.view_mut());
            }
            // The weight queues behind the last wave's history and the stores issued before it: a
            // store behind the weight would hold the chain that rounds into the stored rows.
            for _ in 0..rest % 2 {
                let other = history(context, rest);
                chain(context, &lanes, &in_dm, (rest - 1, &mut rounded));
                for _ in 0..guests {
                    boundary.forward(&mut *context, env, &mut (), &mut *state);
                }
                chain(context, &lanes, &other, (rest, &mut rounded));
                store(context, rest / 2, &rounded, attended.view_mut());
            }
            // A last wave alone: its pair's second wave is past the last, so it stores its own
            // rows.
            for _ in 0..1 - rest % 2 {
                for _ in 0..guests {
                    boundary.forward(&mut *context, env, &mut (), &mut *state);
                }
                chain(context, &lanes, &in_dm, (rest, &mut rounded));
                last(context, rest, &rounded, attended.view_mut());
            }
        }
    }
}
