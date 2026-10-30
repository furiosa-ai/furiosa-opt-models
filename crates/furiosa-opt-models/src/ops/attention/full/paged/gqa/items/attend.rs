use super::*;

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName> Attention<KvHeads, Groups, Head> {
    /// A tensor of every token slice `S` of a cluster's columns, `E` its elements.
    pub(super) type ItemDm<S: M, D: Scalar, E: M> = DmTensor<D, Chip, Self::ItemCluster, S, E>;
    pub(super) type ItemVrf<S: M, D: MaterializableScalar, E: M> =
        VrfTensor<D, Chip, Self::ItemCluster, S, E>;
    /// A head's 32-channel chunk of every slot is contiguous, so the value contraction's fetch
    /// steps slots 64 bytes apart rather than on one bank group.
    type ItemValues<S: M, T: AxisName> =
        Self::ItemDm<S, bf16, m![KvHeads % ({ Local<KvHeads> }), Head / 32, T, Head % 32]>;
    /// What a merge keeps of a wave on slices `S` of rows `R`: each row's f32 output, mass and
    /// largest score.
    pub(super) type Kept<S: M, R: M> = (
        Self::ItemDm<S, f32, m![{ R }, Head]>,
        Self::ItemDm<S, f32, m![{ R }, 1 # 8]>,
        Self::ItemVrf<S, f32, m![{ R }, 1 # 8]>,
    );

    /// Wave `wave`'s queries of `queries`, each query row `R` spread by `switch` over the token
    /// slices `S` of its column: fetched as `F` and switched as `W`.
    pub(super) fn spread<F: M, W: M, Q: M, P: M, Waves: AxisName, S: M, R: M>(
        context: &mut Device,
        (queries, wave): (
            &DmTensor<bf16, Chip, Self::ItemCluster, P, m![Waves % 4, { Q }]>,
            usize,
        ),
        switch: SwitchConfig,
    ) -> Self::ItemDm<S, bf16, m![{ R }, Head]> {
        context
            .main
            .begin(
                queries
                    .view()
                    .tile::<m![Waves % 4], 1, m![1 # 4, { Q }]>(wave % 4),
            )
            .fetch::<F, m![Head % 16]>()
            .switch::<S, W>(switch)
            .collect::<W, m![Head % 16]>()
            .commit_trim::<m![Head % 16]>()
            .commit()
    }

    /// Wave `wave`'s chain on slices `S`: query lanes scored over `in_dm`'s first `units` units
    /// below their `limit`, normalized over `2^depth` slices into `rounded`; returns what merges.
    pub(super) fn attend<
        L: M,
        U: M,
        S: M,
        T: AxisName,
        R: M,
        Keys: M,
        Limit: M,
        Waves: AxisName,
    >(
        &self,
        context: &mut Device,
        ((heads, limit), (keys, in_dm)): (
            (
                &Self::ItemDm<S, bf16, m![{ R }, Head]>,
                &Self::ItemVrf<S, i32, Limit>,
            ),
            (
                &Self::ItemDm<S, i32, Keys>,
                &Self::ItemDm<S, bf16, m![T, KeyValue, KvHeads % ({ Local<KvHeads> }), Head]>,
            ),
        ),
        (units, depth): (usize, usize),
        (wave, rounded): (
            usize,
            &mut Self::ItemDm<S, bf16, m![Waves % 2, { R }, Head]>,
        ),
    ) -> Self::Kept<S, R> {
        // `local`'s `op` over each row's group, on every slice of it.
        let span = |context: &mut Device,
                    local: &Self::ItemDm<S, f32, m![{ R }, 1 # 8]>,
                    op: InterSliceReduceOpF32|
         -> Self::ItemDm<S, f32, m![{ R }, 1 # 8]> {
            context
                .main
                .begin(local.view())
                .fetch::<R, m![1 # 8]>()
                .collect::<R, m![1 # 8]>()
                .vector_init()
                .vector_inter_slice_reduce_span(op, depth)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit()
        };
        let operand = |context: &mut Device,
                       local: &Self::ItemDm<S, f32, m![{ R }, 1 # 8]>|
         -> Self::ItemVrf<S, f32, m![{ R }, 1 # 8]> {
            context
                .sub
                .begin(local.view())
                .fetch::<R, m![1 # 8]>()
                .collect::<R, m![1 # 8]>()
                .to_vrf()
        };
        // The slots each row sees, as tags: those whose key position is below its limit.
        let distance: Self::ItemDm<S, i32, Keys> = context
            .main
            .begin(keys.view())
            .fetch::<U, m![T % 8]>()
            .collect::<U, m![T % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp, limit)
            .vector_final()
            .commit_trim::<m![T % 8]>()
            .commit();
        let present: Self::ItemVrf<S, i4, m![{ R }, T]> = context
            .main
            .begin(distance.view())
            .fetch::<m![{ R }, T / 8], m![T % 8]>()
            .collect::<m![{ R }, T / 8], m![T % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Less(0),
                Cmp::False,
                Cmp::False,
                Cmp::False,
            ]))
            .vector_log(&mut context.sub);
        let lanes: TrfTensor<
            bf16,
            Chip,
            Self::ItemCluster,
            S,
            L,
            m![KvHeads % ({ Local<KvHeads> }), Head],
        > = context
            .sub
            .begin(heads.view())
            .fetch::<m![{ L }, KvHeads % ({ Local<KvHeads> }), Head / 16], m![Head % 16]>()
            .collect::<m![{ L }, KvHeads % ({ Local<KvHeads> }), Head / 16], m![Head % 16]>()
            .to_trf();
        // Every row's scores over each slice's first `units` eight-slot units, from one key
        // stream; later slots keep any bits.
        let mut raw = <Self::ItemDm<S, f32, m![T / 8, { R }, T % 8]>>::new();
        context
            .main
            .begin(in_dm.view().tile::<
                m![KeyValue],
                1,
                m![T, KeyValue = 1 # 2, KvHeads % ({ Local<KvHeads> }), Head],
            >(0))
            .fetch::<m![T / 8, KvHeads % ({ Local<KvHeads> }), T % 8, Head / 16], m![Head % 16]>()
            .collect::<m![T / 8, KvHeads % ({ Local<KvHeads> }), T % 8, Head / 16], m![Head % 16]>()
            .contract_outer::<
                m![T / 8, KvHeads % ({ Local<KvHeads> }), T % 8, Head / 32],
                m![Head % 32],
                _,
                _,
                _,
            >(&lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![T / 8, KvHeads % ({ Local<KvHeads> }), T % 8]>()
            .contract_lane::<
                m![T / 8, KvHeads % ({ Local<KvHeads> }), T % 8],
                m![{ L } # 8],
            >(LaneMode::Interleaved)
            .vector_init()
            .vector_scale(self.scale)
            .vector_final()
            .transpose::<
                m![T / 8, KvHeads % ({ Local<KvHeads> }), T % 8 / 2, { L }],
                m![T % 2 # 8],
            >()
            .commit_trim::<m![T % 2]>()
            .commit_prefix::<m![T / 8], _, _>(raw.view_mut(), units);
        // The values, every non-finite one zero: a slot past a column's blocks may hold any
        // bits, and a zero probability times NaN is NaN.
        let values: Self::ItemValues<S, T> = context
            .main
            .begin(
                in_dm
                    .view()
                    .tile::<
                        m![KeyValue],
                        1,
                        m![T, KeyValue = 1 # 2, KvHeads % ({ Local<KvHeads> }), Head],
                    >(1),
            )
            .fetch::<
                m![T, KvHeads % ({ Local<KvHeads> }), Head / 32, Head % 32 / 8],
                m![Head % 8],
            >()
            .fetch_cast::<f32>()
            .collect::<
                m![T, KvHeads % ({ Local<KvHeads> }), Head / 32, Head % 32 / 8],
                m![Head % 8],
            >()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Less(f32::INFINITY),
                Cmp::Greater(f32::NEG_INFINITY),
                Cmp::False,
                Cmp::False,
            ]))
            .vector_logic(
                LogicBinaryOpF32::BitAnd,
                Branched::imm(CLEARED, 0.0f32).imm(SECOND_CLEARED, 0.0f32),
            )
            .vector_final()
            .cast::<bf16, m![Head % 8 # 16]>()
            .commit_trim::<m![Head % 8]>()
            .commit();
        // Each row's largest score over its group, the scores masked as read.
        let local: Self::ItemDm<S, f32, m![{ R }, 1 # 8]> = context
            .main
            .begin(raw.view())
            .fetch::<m![{ R }, T / 8], m![T % 8]>()
            .collect::<m![{ R }, T / 8], m![T % 8]>()
            .vector_init()
            .vector_tagged(&present)
            .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
            .vector_narrow_split::<m![{ R }, T / 4], m![T % 4]>()
            .vector_fp_binary(FpBinaryOp::AddF, Branched::imm(CLEARED, f32::NEG_INFINITY))
            .vector_intra_slice_reduce::<T, R, m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let top = span(context, &local, InterSliceReduceOpF32::Max);
        let top = operand(context, &top);
        // Unnormalized probabilities, so each row's largest is exactly one in bf16; a masked
        // key's zero less infinity is negative infinity, whose probability is zero.
        let probabilities: Self::ItemDm<S, bf16, m![{ R }, T]> = context
            .main
            .begin(raw.view())
            .fetch::<m![{ R }, T / 8], m![T % 8]>()
            .collect::<m![{ R }, T / 8], m![T % 8]>()
            .vector_init()
            .vector_tagged(&present)
            .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
            .vector_narrow_split::<m![{ R }, T / 4], m![T % 4]>()
            .vector_fp_binary(
                FpBinaryOp::SubF,
                Branched::imm(CLEARED, f32::INFINITY).rf(ADMITTED, &top),
            )
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![{ R }, T / 8], m![T % 8]>()
            .vector_final()
            .cast::<bf16, m![T % 8 # 16]>()
            .commit_trim::<m![T % 8]>()
            .commit();
        // Each row's mass over its group, the rounded probabilities summed so the weights
        // divide to one; an empty row's mass is the epsilon, so its output is zero.
        let local: Self::ItemDm<S, f32, m![{ R }, 1 # 8]> = context
            .main
            .begin(probabilities.view())
            .fetch::<m![{ R }, T / 8], m![T % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ R }, T / 8], m![T % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ R }, T / 4], m![T % 4]>()
            .vector_intra_slice_reduce::<T, R, m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Add, 1e-6f32 / 2.0)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let mass = span(context, &local, InterSliceReduceOpF32::Add);
        let total = operand(context, &mass);
        let weights: TrfTensor<
            bf16,
            Chip,
            Self::ItemCluster,
            S,
            L,
            m![KvHeads % ({ Local<KvHeads> }), T],
        > = context
            .sub
            .begin(probabilities.view())
            .fetch::<m![{ L }, KvHeads % ({ Local<KvHeads> }), T / 16], m![T % 16]>()
            .collect::<m![{ L }, KvHeads % ({ Local<KvHeads> }), T / 16], m![T % 16]>()
            .to_trf();
        // Each row's attention in f32 on every slice of its group: the scored units' values
        // weighted by every lane, reduced over the group and divided by the mass in one pass.
        let output: Self::ItemDm<S, f32, m![{ R }, Head]> = context
            .main
            .begin(values.view())
            .fetch::<m![Head / 32, KvHeads % ({ Local<KvHeads> }), T], m![Head % 32]>()
            .collect::<
                m![Head / 32, KvHeads % ({ Local<KvHeads> }), T, Head % 32 / 16],
                m![Head % 16],
            >()
            .contract_outer::<
                m![Head / 32, KvHeads % ({ Local<KvHeads> }), T],
                m![Head % 32],
                _,
                _,
                _,
            >(&weights)
            .contract_packet::<m![Head % 32]>()
            .contract_time_prefix::<m![Head / 32, KvHeads % ({ Local<KvHeads> })], m![T / 8]>(units)
            .contract_lane::<
                m![Head / 32, KvHeads % ({ Local<KvHeads> }), { L }, Head % 32 / 8],
                m![Head % 8],
            >(
                LaneMode::Sequential,
            )
            .vector_init()
            .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Add, depth)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![Head / 32, KvHeads % ({ Local<KvHeads> }), { L }, Head % 32 / 4],
                m![Head % 4],
            >()
            .vector_fp_div(&total)
            .vector_widen_concat::<
                m![Head / 32, KvHeads % ({ Local<KvHeads> }), { L }, Head % 32 / 8],
                m![Head % 8],
            >()
            .vector_final()
            .commit_trim::<m![Head % 8]>()
            .commit();
        context
            .main
            .begin(output.view())
            .fetch::<m![{ R }, Head / 8], m![Head % 8]>()
            .collect::<m![{ R }, Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_final()
            .cast::<bf16, m![Head % 8 # 16]>()
            .commit_trim::<m![Head % 8]>()
            .commit_view(
                rounded
                    .view_mut()
                    .tile::<m![Waves % 2], 1, m![1 #{!} 2, { R }, Head]>(wave % 2),
            );
        (output, mass, top)
    }
}
