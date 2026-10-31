use super::*;
use crate::Op;
use furiosa_opt_std::prelude;

/// Softmax over one row of `K` keys: the probabilities and the row's log-sum-exp.
pub(crate) struct Normalize;

impl<KvHeads: AxisName, Groups: AxisName, K: AxisName>
    Op<
        DmTensorView<
            '_,
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![Rows % 8, { Symbol<K> }],
        >,
    > for Normalize
{
    type Output = (
        DmTensor<bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Rows % 8, { Symbol<K> }]>,
        Local<KvHeads, Groups>,
    );

    fn forward(
        &self,
        context: &mut Device,
        scores: DmTensorView<
            '_,
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![Rows % 8, { Symbol<K> }],
        >,
    ) -> Self::Output {
        let mut probabilities = DmTensor::new();
        let mut lse = Local::<KvHeads, Groups>::new();
        let maximum = Op::forward(&Maximum::<K>(PhantomData), &mut *context, scores.clone());
        let exponent: DmTensor<
            f32,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![Rows % 8, { Symbol<K> }],
        > = context
            .main
            .begin(scores.clone())
            .fetch::<m![Rows % 8, { Symbol<K> } / 16], m![{ Symbol<K> } % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([Cmp::Equal(f32::NEG_INFINITY); 4]))
            .vector_narrow_split::<m![Rows % 8, { Symbol<K> } / 4], m![{ Symbol<K> } % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, Branched::rf(state::PRESENT, &maximum))
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_final()
            .commit_trim::<m![{ Symbol<K> } % 8]>()
            .commit();
        let total: Local<KvHeads, Groups> = context
            .main
            .begin(exponent.view())
            .fetch::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .collect::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows % 8, { Symbol<K> } / 4], m![{ Symbol<K> } % 4]>()
            .vector_intra_slice_reduce::<K, m![Rows % 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Max, 1e-30f32)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        Op::forward(
            &Lse,
            &mut *context,
            ((total.view(), &maximum), lse.view_mut()),
        );
        let divisor = Op::forward(&state::Operand, &mut *context, total.view());
        context
            .main
            .begin(exponent.view())
            .fetch::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .collect::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows % 8, { Symbol<K> } / 4], m![{ Symbol<K> } % 4]>()
            .vector_fp_div(&divisor)
            .vector_widen_concat::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_final()
            .cast::<bf16, m![{ Symbol<K> } % 8 # 16]>()
            .commit_trim::<m![{ Symbol<K> } % 8]>()
            .commit_view(probabilities.view_mut());
        (probabilities, lse)
    }
}

/// Tile `t` of a group's `K`-key score rows with every key its mask marks nonzero set to −∞.
pub(crate) struct Mask;

impl<KvHeads: AxisName, Groups: AxisName, K: AxisName, T: AxisName>
    Op<(
        (
            DmTensorView<
                '_,
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
            >,
            DmTensorView<
                '_,
                i8,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
            >,
        ),
        &mut DmTensor<
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
        >,
        usize,
    )> for Mask
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (input, output, t): (
            (
                DmTensorView<
                    '_,
                    bf16,
                    Chip,
                    Cluster<KvHeads>,
                    Slice<KvHeads, Groups>,
                    m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
                >,
                DmTensorView<
                    '_,
                    i8,
                    Chip,
                    Cluster<KvHeads>,
                    Slice<KvHeads, Groups>,
                    m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
                >,
            ),
            &mut DmTensor<
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
            >,
            usize,
        ),
    ) {
        let (scores, mask) = input;
        let visible: VrfTensor<
            i4,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![Rows % 8, { Symbol<K> }],
        > = context
            .main
            .begin(mask.tile::<m![{ Symbol<T> }], 1, m![{ TileOf<T> }, Rows % 8, { Symbol<K> }]>(t))
            .fetch::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .fetch_cast::<i32>()
            .collect::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Equal(0i32),
                Cmp::False,
                Cmp::False,
                Cmp::False,
            ]))
            .vector_log(&mut context.sub);
        context
            .main
            .begin(
                scores.tile::<m![{ Symbol<T> }], 1, m![{ TileOf<T> }, Rows % 8, { Symbol<K> }]>(t),
            )
            .fetch::<m![Rows % 8, { Symbol<K> } / 16], m![{ Symbol<K> } % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_init()
            .vector_tagged(&visible)
            .vector_logic(
                LogicBinaryOpF32::BitAnd,
                Branched::imm(state::PRESENT, 0.0f32),
            )
            .vector_clip(
                ClipBinaryOpF32::Add,
                Branched::imm(state::PRESENT, f32::NEG_INFINITY),
            )
            .vector_final()
            .cast::<bf16, m![{ Symbol<K> } % 8 # 16]>()
            .commit_trim::<m![{ Symbol<K> } % 8]>()
            .commit_view(
                output
                    .view_mut()
                    .tile::<m![{ Symbol<T> }], 1, m![{ TileSlot<T> }, Rows % 8, { Symbol<K> }]>(t),
            );
    }
}

/// The maximum of each row of `K` scores.
pub(crate) struct Maximum<K>(pub(crate) PhantomData<K>);

impl<KvHeads: AxisName, Groups: AxisName, K: AxisName, Q: M>
    Op<DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>> for Maximum<K>
{
    type Output = Scalar<KvHeads, Groups>;

    fn forward(
        &self,
        context: &mut Device,
        scores: DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
    ) -> Self::Output {
        context
            .main
            .begin(scores)
            .fetch::<m![Rows % 8, { Symbol<K> } / 16], m![{ Symbol<K> } % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows % 8, { Symbol<K> } / 4], m![{ Symbol<K> } % 4]>()
            .vector_intra_slice_reduce::<K, m![Rows % 8], m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .to_vrf(&mut context.sub)
    }
}

/// The total of each row of `K` exponents into tile `t` of `total`.
struct Total<K>(PhantomData<K>);

impl<KvHeads: AxisName, Groups: AxisName, K: AxisName, T: AxisName, Q: M>
    Op<(
        DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
        &mut state::Running<KvHeads, Groups, T>,
        usize,
    )> for Total<K>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (exponent, total, t): (
            DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
            &mut state::Running<KvHeads, Groups, T>,
            usize,
        ),
    ) {
        context
            .main
            .begin(exponent)
            .fetch::<m![Rows % 8, { Symbol<K> } / 16], m![{ Symbol<K> } % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows % 8, { Symbol<K> } / 4], m![{ Symbol<K> } % 4]>()
            .vector_intra_slice_reduce::<K, m![Rows % 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Max, 1e-30f32)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(
                total
                    .view_mut()
                    .tile::<m![{ Symbol<T> }], 1, TileStatisticSlot<T>>(t),
            );
    }
}

/// Unnormalized causal softmax of tile `t` over its first `K` (`KEYS`) unmasked keys: `exp(score −
/// max)` into `exponent`, the total into `total`, the lse into `lse`, later keys unwritten.
pub(crate) struct CausalSoftmax<K, const KEYS: usize>(pub(crate) PhantomData<K>);

impl<K: AxisName, const KEYS: usize> CausalSoftmax<K, KEYS> {
    /// The first `K` keys of tile `t`'s row of a `W`-key owner.
    fn prefix<
        'a,
        D: prelude::Scalar,
        KvHeads: AxisName,
        Groups: AxisName,
        T: AxisName,
        W: AxisName,
    >(
        row: DmTensorView<
            'a,
            D,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Symbol<T> }, Rows % 8, { Symbol<W> }],
        >,
        t: usize,
    ) -> DmTensorView<
        'a,
        D,
        Chip,
        Cluster<KvHeads>,
        Slice<KvHeads, Groups>,
        m![{ TileOf<T> }, Rows % 8, { Symbol<K> } # ({ Symbol<W> })],
    > {
        type Row<T, W> = m![{ TileOf<T> }, Rows % 8, { Symbol<W> }];
        // SAFETY: the first K keys of the row keep their positions in its W-key footprint.
        unsafe {
            row.tile::<m![{ Symbol<T> }], 1, Row<T, W>>(t)
            .tile::<m![{ Symbol<W> }], KEYS,
                m![{ TileOf<T> }, Rows % 8, { Symbol<W> } = { KEYS } # ({ Symbol<W> })]>(0)
            .reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                m![{ TileOf<T> }, Rows % 8, { Symbol<K> } # ({ Symbol<W> })]>()
        }
    }
}

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    K: AxisName,
    W: AxisName,
    T: AxisName,
    L: M,
    const KEYS: usize,
>
    Op<(
        (
            DmTensorView<
                '_,
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Symbol<W> }],
            >,
            DmTensorView<
                '_,
                i8,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Symbol<W> }],
            >,
        ),
        (
            &mut DmTensor<
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Symbol<W> }],
            >,
            &mut state::Running<KvHeads, Groups, T>,
            DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, L>,
        ),
        usize,
    )> for CausalSoftmax<K, KEYS>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((scores, mask), (exponent, total, lse), t): (
            (
                DmTensorView<
                    '_,
                    bf16,
                    Chip,
                    Cluster<KvHeads>,
                    Slice<KvHeads, Groups>,
                    m![{ Symbol<T> }, Rows % 8, { Symbol<W> }],
                >,
                DmTensorView<
                    '_,
                    i8,
                    Chip,
                    Cluster<KvHeads>,
                    Slice<KvHeads, Groups>,
                    m![{ Symbol<T> }, Rows % 8, { Symbol<W> }],
                >,
            ),
            (
                &mut DmTensor<
                    bf16,
                    Chip,
                    Cluster<KvHeads>,
                    Slice<KvHeads, Groups>,
                    m![{ Symbol<T> }, Rows % 8, { Symbol<W> }],
                >,
                &mut state::Running<KvHeads, Groups, T>,
                DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, L>,
            ),
            usize,
        ),
    ) {
        let visible: VrfTensor<
            i4,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![Rows % 8, { Symbol<K> }],
        > = context
            .main
            .begin(Self::prefix(mask, t))
            .fetch::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .fetch_cast::<i32>()
            .collect::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Equal(0i32),
                Cmp::False,
                Cmp::False,
                Cmp::False,
            ]))
            .vector_log(&mut context.sub);
        // Both passes mask the scores as they read them: a masked key's zero less infinity is −∞.
        let top: Scalar<KvHeads, Groups> = context
            .main
            .begin(Self::prefix(scores.clone(), t))
            .fetch::<m![Rows % 8, { Symbol<K> } / 16], m![{ Symbol<K> } % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_init()
            .vector_tagged(&visible)
            .vector_logic(
                LogicBinaryOpF32::BitAnd,
                Branched::imm(state::PRESENT, 0.0f32),
            )
            .vector_narrow_split::<m![Rows % 8, { Symbol<K> } / 4], m![{ Symbol<K> } % 4]>()
            .vector_fp_binary(
                FpBinaryOp::AddF,
                Branched::imm(state::PRESENT, f32::NEG_INFINITY),
            )
            .vector_intra_slice_reduce::<K, m![Rows % 8], m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .to_vrf(&mut context.sub);
        context
            .main
            .begin(Self::prefix(scores, t))
            .fetch::<m![Rows % 8, { Symbol<K> } / 16], m![{ Symbol<K> } % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_init()
            .vector_tagged(&visible)
            .vector_logic(
                LogicBinaryOpF32::BitAnd,
                Branched::imm(state::PRESENT, 0.0f32),
            )
            .vector_narrow_split::<m![Rows % 8, { Symbol<K> } / 4], m![{ Symbol<K> } % 4]>()
            .vector_fp_binary(
                FpBinaryOp::SubF,
                Branched::imm(state::PRESENT, f32::INFINITY).rf(ADMITTED, &top),
            )
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_final()
            .cast::<bf16, m![{ Symbol<K> } % 8 # 16]>()
            .commit_trim::<m![{ Symbol<K> } % 8]>()
            .commit_view(unsafe {
                exponent.view_mut()
                .tile::<m![{ Symbol<T> }], 1, m![{ TileSlot<T> }, Rows % 8, { Symbol<W> }]>(t)
                .tile::<m![{ Symbol<W> }], KEYS,
                    m![{ TileSlot<T> }, Rows % 8, { Symbol<W> } = { KEYS } #{!} ({ Symbol<W> })]>(0)
                .reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                    m![{ TileSlot<T> }, Rows % 8, { Symbol<K> } #{!} ({ Symbol<W> })]>()
            });
        Op::forward(
            &Total::<K>(PhantomData),
            &mut *context,
            (Self::prefix(exponent.view(), t), &mut *total, t),
        );
        Op::forward(
            &Lse,
            context,
            (
                (
                    total
                        .view()
                        .tile::<m![{ Symbol<T> }], 1, TileStatistic<T>>(t),
                    &top,
                ),
                lse,
            ),
        );
    }
}

const ADMITTED: TagGuard =
    TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);

/// Unnormalized softmax of tile `t` against `top`, the row maximum or the running lse: `exp(score −
/// top)` into `exponent` and the row's total, which divides the weighted values instead of the row.
pub(crate) struct Exponentiate;

impl<KvHeads: AxisName, Groups: AxisName, K: AxisName, T: AxisName>
    Op<(
        (
            DmTensorView<
                '_,
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
            >,
            &Scalar<KvHeads, Groups>,
        ),
        (
            &mut DmTensor<
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
            >,
            &mut state::Running<KvHeads, Groups, T>,
        ),
        usize,
    )> for Exponentiate
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (input, output, t): (
            (
                DmTensorView<
                    '_,
                    bf16,
                    Chip,
                    Cluster<KvHeads>,
                    Slice<KvHeads, Groups>,
                    m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
                >,
                &Scalar<KvHeads, Groups>,
            ),
            (
                &mut DmTensor<
                    bf16,
                    Chip,
                    Cluster<KvHeads>,
                    Slice<KvHeads, Groups>,
                    m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
                >,
                &mut state::Running<KvHeads, Groups, T>,
            ),
            usize,
        ),
    ) {
        let (exponent, total) = output;
        Op::forward(&Exponents, &mut *context, (input, &mut *exponent, t));
        Op::forward(
            &Total::<K>(PhantomData),
            context,
            (
                exponent
                    .view()
                    .tile::<m![{ Symbol<T> }], 1, m![{ TileOf<T> }, Rows % 8, { Symbol<K> }]>(t),
                total,
                t,
            ),
        );
    }
}

/// Tile `t`'s `exp(score − top)` of a group's `K`-key score rows into `exponent`.
pub(crate) struct Exponents;

impl<KvHeads: AxisName, Groups: AxisName, K: AxisName, T: AxisName>
    Op<(
        (
            DmTensorView<
                '_,
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
            >,
            &Scalar<KvHeads, Groups>,
        ),
        &mut DmTensor<
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
        >,
        usize,
    )> for Exponents
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((scores, top), exponent, t): (
            (
                DmTensorView<
                    '_,
                    bf16,
                    Chip,
                    Cluster<KvHeads>,
                    Slice<KvHeads, Groups>,
                    m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
                >,
                &Scalar<KvHeads, Groups>,
            ),
            &mut DmTensor<
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Symbol<K> }],
            >,
            usize,
        ),
    ) {
        type Row<T, K> = m![{ TileOf<T> }, Rows % 8, { Symbol<K> }];
        type RowSlot<T, K> = m![{ TileSlot<T> }, Rows % 8, { Symbol<K> }];
        context
            .main
            .begin(scores.tile::<m![{ Symbol<T> }], 1, Row<T, K>>(t))
            .fetch::<m![Rows % 8, { Symbol<K> } / 16], m![{ Symbol<K> } % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([Cmp::Equal(f32::NEG_INFINITY); 4]))
            .vector_narrow_split::<m![Rows % 8, { Symbol<K> } / 4], m![{ Symbol<K> } % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, Branched::rf(state::PRESENT, top))
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Rows % 8, { Symbol<K> } / 8], m![{ Symbol<K> } % 8]>()
            .vector_final()
            .cast::<bf16, m![{ Symbol<K> } % 8 # 16]>()
            .commit_trim::<m![{ Symbol<K> } % 8]>()
            .commit_view(
                exponent
                    .view_mut()
                    .tile::<m![{ Symbol<T> }], 1, RowSlot<T, K>>(t),
            );
    }
}

/// Each row's lse, `log total + top`, into `lse`.
pub(crate) struct Lse;

impl<KvHeads: AxisName, Groups: AxisName, Q: M, L: M>
    Op<(
        (
            DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
            &Scalar<KvHeads, Groups>,
        ),
        DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, L>,
    )> for Lse
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((total, top), lse): (
            (
                DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
                &Scalar<KvHeads, Groups>,
            ),
            DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, L>,
        ),
    ) {
        context
            .main
            .begin(total)
            .fetch::<m![Rows % 8], m![1 # 8]>()
            .collect::<m![Rows % 8], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_unary(FpUnaryOp::Log)
            .vector_fp_binary(FpBinaryOp::AddF, top)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(lse);
    }
}

/// One 128-key panel's probabilities times one fill of its values, into `output`.
pub(crate) struct Weight;

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, P: M, E: M>
    Op<(
        DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, P>,
        &fresh::Values<KvHeads, Groups, Head>,
        DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
    )> for Weight
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (probabilities, value, output): (
            DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, P>,
            &fresh::Values<KvHeads, Groups, Head>,
            DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
        ),
    ) {
        context.main.begin(probabilities)
                .fetch::<
                    m![Rows % 2, { Channels<Head> } / 32, Cache / 8],
                    m![Rows % 8 / 2, Cache % 8],
                >()
                .collect::<
                    m![Rows % 2, { Channels<Head> } / 32, Cache / 8, Rows % 8 / 4],
                    m![Rows % 4 / 2, Cache % 8],
                >()
                .contract_outer::<
                    m![Rows % 2, { Channels<Head> } / 32, Cache / 8, Head % 32 / 8],
                    m![Rows % 8 / 2, Cache % 8],
                    _,
                    _,
                    _,
                >(value)
                .contract_packet::<m![Rows % 8 / 2]>()
                .contract_time::<m![Rows % 2, { Channels<Head> } / 32, Head % 32 / 8]>()
                .contract_lane::<
                    m![Rows % 2, { Channels<Head> } / 32, Head % 32 / 8, Head % 8],
                    m![Rows % 8 / 2 # 8],
                >(LaneMode::Sequential)
                .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_final()
                .transpose::<m![Rows % 2, { Channels<Head> } / 2, Rows % 8 / 2], m![Head % 2 # 8]>()
                .commit_trim::<m![Head % 2]>().commit_view(output);
    }
}

/// One 128-key panel's probabilities times every channel part of its values, a fill a part.
pub(crate) struct WeighParts<Head>(pub(crate) PhantomData<Head>);

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    P: M,
    S: ValueSource<KvHeads, Groups, Head>,
>
    Op<(
        DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, P>,
        &mut S,
    )> for WeighParts<Head>
{
    type Output = Partial<KvHeads, Groups, Head>;

    fn forward(
        &self,
        context: &mut Device,
        (probabilities, source): (
            DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, P>,
            &mut S,
        ),
    ) -> Self::Output {
        type Slot<Head> = m![Rows % 8, { PartSlot<Head> }, { Channels<Head> }];
        let mut value = source.value(context, 0);
        let mut output = Partial::<KvHeads, Groups, Head>::new();
        #[unroll]
        for j in 1..Part::<Head>::COUNT {
            let next = source.value_part(context, 0, j);
            Op::forward(
                &Weight,
                &mut *context,
                (
                    probabilities.clone(),
                    &value,
                    output.view_mut().tile::<Parts<Head>, 1, Slot<Head>>(j - 1),
                ),
            );
            value = next;
        }
        Op::forward(
            &Weight,
            context,
            (
                probabilities,
                &value,
                output
                    .view_mut()
                    .tile::<Parts<Head>, 1, Slot<Head>>(Part::<Head>::COUNT - 1),
            ),
        );
        output
    }
}

/// A tile row group's FP32 dot products with one panel's keys, in vector registers.
type DotRow<KvHeads, Groups> =
    VrfTensor<f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Rows % 8, Cache]>;

/// The sum of two FP32 dot products of one tile's 128 keys, rounded into `scores`.
pub(crate) struct Round;

impl<KvHeads: AxisName, Groups: AxisName, A: M, B: M, E: M>
    Op<(
        DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, A>,
        DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, B>,
        DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
    )> for Round
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (sum, last, scores): (
            DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, A>,
            DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, B>,
            DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
        ),
    ) {
        let sum: DotRow<KvHeads, Groups> = context
            .sub
            .begin(sum)
            .fetch::<m![Rows % 8, Cache / 8], m![Cache % 8]>()
            .collect::<m![Rows % 8, Cache / 8], m![Cache % 8]>()
            .to_vrf();
        context
            .main
            .begin(last)
            .fetch::<m![Rows % 8, Cache / 8], m![Cache % 8]>()
            .collect::<m![Rows % 8, Cache / 8], m![Cache % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &sum)
            .vector_final()
            .cast::<bf16, m![Cache % 8 # 16]>()
            .commit_trim::<m![Cache % 8]>()
            .commit_view(scores);
    }
}

/// Adds tile `t` of `last` into tile `t` of `sum`, both a part's FP32 dot products or their sum.
pub(crate) struct Accumulate;

impl<KvHeads: AxisName, Groups: AxisName, R: AxisName>
    Op<(
        &mut history::Dot<KvHeads, Groups, R>,
        &history::Dot<KvHeads, Groups, R>,
        usize,
    )> for Accumulate
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (sum, last, t): (
            &mut history::Dot<KvHeads, Groups, R>,
            &history::Dot<KvHeads, Groups, R>,
            usize,
        ),
    ) {
        type Row<R> = m![{ TileOf<R> }, Rows % 8, Cache];
        type RowSlot<R> = m![{ TileSlot<R> }, Rows % 8, Cache];
        let held: DotRow<KvHeads, Groups> = context
            .sub
            .begin(sum.view().tile::<m![{ Symbol<R> }], 1, Row<R>>(t))
            .fetch::<m![Rows % 8, Cache / 8], m![Cache % 8]>()
            .collect::<m![Rows % 8, Cache / 8], m![Cache % 8]>()
            .to_vrf();
        context
            .main
            .begin(last.view().tile::<m![{ Symbol<R> }], 1, Row<R>>(t))
            .fetch::<m![Rows % 8, Cache / 8], m![Cache % 8]>()
            .collect::<m![Rows % 8, Cache / 8], m![Cache % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &held)
            .vector_final()
            .commit_trim::<m![Cache % 8]>()
            .commit_view(sum.view_mut().tile::<m![{ Symbol<R> }], 1, RowSlot<R>>(t));
    }
}
