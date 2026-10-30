use super::*;
use crate::Op;

pub(crate) const PRESENT: TagGuard =
    TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);

/// The attended values and the log-sum-exp of `T` query tiles, one tile per step of `T`.
pub(crate) type Carry<KvHeads, Groups, Head, T> = DmTensor<
    f32,
    Chip,
    Cluster<KvHeads>,
    Slice<KvHeads, Groups>,
    m![{ Symbol<T> }, Rows % 8, Head],
>;
pub(crate) type Running<KvHeads, Groups, T> = DmTensor<
    f32,
    Chip,
    Cluster<KvHeads>,
    Slice<KvHeads, Groups>,
    m![{ Symbol<T> }, Rows % 8, 1 # 8],
>;

/// A group's tiles of a state owner: an owner of its own, or its group's half of a pair's.
pub(crate) trait Tiles<KvHeads: AxisName, Groups: AxisName, Head: AxisName> {
    type Group: M;
    type GroupSlot: M;
    type Tile: M;
    type TileSlot: M;
    fn group(
        &self,
    ) -> DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::Group>;
    fn group_mut(
        &mut self,
    ) -> DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::GroupSlot>;
    fn tile(
        &self,
        t: usize,
    ) -> DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::Tile>;
    fn tile_mut(
        &mut self,
        t: usize,
    ) -> DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::TileSlot>;
}

/// Either owner of a group of its own: its tiles' attended values or their running lse.
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, T: AxisName, E: M>
    Tiles<KvHeads, Groups, Head>
    for DmTensor<
        f32,
        Chip,
        Cluster<KvHeads>,
        Slice<KvHeads, Groups>,
        m![{ Symbol<T> }, Rows % 8, { E }],
    >
{
    type Group = m![{ Symbol<T> }, Rows % 8, { E }];
    type GroupSlot = m![{ Symbol<T> }, Rows % 8, { E }];
    type Tile = m![{ TileOf<T> }, Rows % 8, { E }];
    type TileSlot = m![{ TileSlot<T> }, Rows % 8, { E }];
    fn group(
        &self,
    ) -> DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::Group> {
        self.view()
    }
    fn group_mut(
        &mut self,
    ) -> DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::GroupSlot>
    {
        self.view_mut()
    }
    fn tile(
        &self,
        t: usize,
    ) -> DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::Tile> {
        self.view().tile::<m![{ Symbol<T> }], 1, Self::Tile>(t)
    }
    fn tile_mut(
        &mut self,
        t: usize,
    ) -> DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::TileSlot>
    {
        self.view_mut()
            .tile::<m![{ Symbol<T> }], 1, Self::TileSlot>(t)
    }
}

pub(crate) struct Operand;

impl<KvHeads: AxisName, Groups: AxisName, E: M>
    Op<DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>> for Operand
{
    type Output = Scalar<KvHeads, Groups>;

    fn forward(
        &self,
        context: &mut Device,
        value: DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
    ) -> Self::Output {
        context
            .sub
            .begin(value)
            .fetch::<m![Rows % 8], m![1 # 8]>()
            .collect::<m![Rows % 8], m![1 # 8]>()
            .to_vrf()
    }
}

struct Shift;

impl<KvHeads: AxisName, Groups: AxisName, E: M>
    Op<(
        DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
        &Scalar<KvHeads, Groups>,
    )> for Shift
{
    type Output = Local<KvHeads, Groups>;

    fn forward(
        &self,
        context: &mut Device,
        (value, top): (
            DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
            &Scalar<KvHeads, Groups>,
        ),
    ) -> Self::Output {
        context
            .main
            .begin(value)
            .fetch::<m![Rows % 8], m![1 # 8]>()
            .collect::<m![Rows % 8], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([Cmp::Equal(f32::NEG_INFINITY); 4]))
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, Branched::rf(PRESENT, top))
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit()
    }
}

struct Rescale<Head>(PhantomData<Head>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, E: M>
    Op<(
        DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
        &Scalar<KvHeads, Groups>,
    )> for Rescale<Head>
{
    type Output = Partial<KvHeads, Groups, Head>;

    fn forward(
        &self,
        context: &mut Device,
        (value, share): (
            DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
            &Scalar<KvHeads, Groups>,
        ),
    ) -> Self::Output {
        context
            .main
            .begin(value)
            .fetch::<m![Rows % 8, Head / 8], m![Head % 8]>()
            .collect::<m![Rows % 8, Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows % 8, Head / 4], m![Head % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), share)
            .vector_widen_concat::<m![Rows % 8, Head / 8], m![Head % 8]>()
            .vector_final()
            .commit_trim::<m![Head % 8]>()
            .commit()
    }
}

/// Folds a partial and its lse into tile `at` of the state, in place.
pub(crate) struct Merge;

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, T: AxisName, P: M, L: M>
    Op<(
        (
            &mut Carry<KvHeads, Groups, Head, T>,
            &mut Running<KvHeads, Groups, T>,
            usize,
        ),
        (
            DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, P>,
            DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, L>,
        ),
    )> for Merge
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (state, new): (
            (
                &mut Carry<KvHeads, Groups, Head, T>,
                &mut Running<KvHeads, Groups, T>,
                usize,
            ),
            (
                DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, P>,
                DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, L>,
            ),
        ),
    ) {
        let (carry, running, at) = state;
        let (partial, lse) = new;
        let held = Op::forward(
            &Operand,
            &mut *context,
            running
                .view()
                .tile::<m![{ Symbol<T> }], 1, TileStatistic<T>>(at),
        );
        let top: Scalar<KvHeads, Groups> = context
            .main
            .begin(lse.clone())
            .fetch::<m![Rows % 8], m![1 # 8]>()
            .collect::<m![Rows % 8], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Max, &held)
            .vector_final()
            .to_vrf(&mut context.sub);
        let old_mass = Op::forward(
            &Shift,
            &mut *context,
            (
                running
                    .view()
                    .tile::<m![{ Symbol<T> }], 1, TileStatistic<T>>(at),
                &top,
            ),
        );
        let new_mass = Op::forward(&Shift, &mut *context, (lse, &top));
        let mass = Op::forward(&Operand, &mut *context, new_mass.view());
        let sum: Local<KvHeads, Groups> = context
            .main
            .begin(old_mass.view())
            .fetch::<m![Rows % 8], m![1 # 8]>()
            .collect::<m![Rows % 8], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &mass)
            .vector_clip(ClipBinaryOpF32::Max, 1e-30f32)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        context
            .main
            .begin(sum.view())
            .fetch::<m![Rows % 8], m![1 # 8]>()
            .collect::<m![Rows % 8], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_unary(FpUnaryOp::Log)
            .vector_fp_binary(FpBinaryOp::AddF, &top)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(
                running
                    .view_mut()
                    .tile::<m![{ Symbol<T> }], 1, TileStatisticSlot<T>>(at),
            );
        let divisor = Op::forward(&Operand, &mut *context, sum.view());
        let share =
            |context: &mut Device, mass: &Local<KvHeads, Groups>| -> Scalar<KvHeads, Groups> {
                context
                    .main
                    .begin(mass.view())
                    .fetch::<m![Rows % 8], m![1 # 8]>()
                    .collect::<m![Rows % 8], m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_trim::<m![1 # 4]>()
                    .vector_fp_div(&divisor)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .to_vrf(&mut context.sub)
            };
        let old_share = share(context, &old_mass);
        let old_value = Op::forward(
            &Rescale::<Head>(PhantomData),
            &mut *context,
            (
                carry
                    .view()
                    .tile::<m![{ Symbol<T> }], 1, TileRows<Head, T>>(at),
                &old_share,
            ),
        );
        let new_share = share(context, &new_mass);
        let new_value = Op::forward(
            &Rescale::<Head>(PhantomData),
            &mut *context,
            (partial, &new_share),
        );
        let value: VrfTensor<
            f32,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![Rows % 8, Head],
        > = context
            .sub
            .begin(new_value.view())
            .fetch::<m![Rows % 8, Head / 8], m![Head % 8]>()
            .collect::<m![Rows % 8, Head / 8], m![Head % 8]>()
            .to_vrf();
        context
            .main
            .begin(old_value.view())
            .fetch::<m![Rows % 8, Head / 8], m![Head % 8]>()
            .collect::<m![Rows % 8, Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &value)
            .vector_final()
            .commit_trim::<m![Head % 8]>()
            .commit_view(
                carry
                    .view_mut()
                    .tile::<m![{ Symbol<T> }], 1, TileRowSlot<Head, T>>(at),
            );
    }
}

/// Starts every tile of the group from its first block: the summed per-panel weighted values over
/// the row total. The block's lse is already the tiles' running lse.
pub(crate) struct Seed;

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    S: AxisName,
    T: AxisName,
    C: Tiles<KvHeads, Groups, Head>,
>
    Op<(
        &mut C,
        &DmTensor<
            f32,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Symbol<T> }, { Symbol<S> }, Rows % 8, Head],
        >,
        &Running<KvHeads, Groups, T>,
    )> for Seed
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (carry, partials, total): (
            &mut C,
            &DmTensor<
                f32,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, { Symbol<S> }, Rows % 8, Head],
            >,
            &Running<KvHeads, Groups, T>,
        ),
    ) {
        type Panels<Head, T, S> = m![{ TileOf<T> }, { Symbol<S> }, Rows % 8, Head];
        #[unroll]
        for t in 0..T::SIZE {
            let total = Op::forward(
                &Operand,
                &mut *context,
                total
                    .view()
                    .tile::<m![{ Symbol<T> }], 1, TileStatistic<T>>(t),
            );
            context.main.begin(partials.view().tile::<m![{ Symbol<T> }], 1, Panels<Head, T, S>>(t))
                .fetch::<m![Rows % 8, Head / 8, { Symbol<S> }], m![Head % 8]>()
                .collect::<m![Rows % 8, Head / 8, { Symbol<S> }], m![Head % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Rows % 8, Head / 8, { Symbol<S> }, Head % 8 / 4],
                    m![Head % 4]>()
                .vector_intra_slice_reduce::<S, m![Rows % 8, Head / 8, Head % 8 / 4], m![Head % 4]>(
                    IntraSliceReduceOpF32::Add,
                )
                .vector_fp_div(&total)
                .vector_widen_concat::<m![Rows % 8, Head / 8], m![Head % 8]>()
                .vector_final()
                .commit_trim::<m![Head % 8]>()
                .commit_view(carry.tile_mut(t));
        }
    }
}

/// Starts tile `t` of the group from the first `S` of its `W` panels of a causal block: their
/// summed weighted values over the row total; `PANELS` is `S`'s size.
pub(crate) struct SeedPrefix<S, const PANELS: usize>(pub(crate) PhantomData<S>);

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    S: AxisName,
    W: AxisName,
    T: AxisName,
    C: Tiles<KvHeads, Groups, Head>,
    const PANELS: usize,
>
    Op<(
        &mut C,
        &DmTensor<
            f32,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Symbol<T> }, { Symbol<W> }, Rows % 8, Head],
        >,
        &Running<KvHeads, Groups, T>,
        usize,
    )> for SeedPrefix<S, PANELS>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (carry, partials, total, t): (
            &mut C,
            &DmTensor<
                f32,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, { Symbol<W> }, Rows % 8, Head],
            >,
            &Running<KvHeads, Groups, T>,
            usize,
        ),
    ) {
        let total = Op::forward(
            &Operand,
            &mut *context,
            total
                .view()
                .tile::<m![{ Symbol<T> }], 1, TileStatistic<T>>(t),
        );
        // SAFETY: the first S panels keep their positions in the tile's W-panel footprint.
        let panels = unsafe {
            partials.view()
            .tile::<m![{ Symbol<T> }], 1, m![{ TileOf<T> }, { Symbol<W> }, Rows % 8, Head]>(t)
            .tile::<m![{ Symbol<W> }], PANELS,
                m![{ TileOf<T> }, { Symbol<W> } = { PANELS } # ({ Symbol<W> }), Rows % 8, Head]>(0)
            .reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                m![{ TileOf<T> }, { Symbol<S> } # ({ Symbol<W> }), Rows % 8, Head]>()
        };
        context
            .main
            .begin(panels)
            .fetch::<m![Rows % 8, Head / 8, { Symbol<S> }], m![Head % 8]>()
            .collect::<m![Rows % 8, Head / 8, { Symbol<S> }], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![Rows % 8, Head / 8, { Symbol<S> }, Head % 8 / 4],
                m![Head % 4],
            >()
            .vector_intra_slice_reduce::<S, m![Rows % 8, Head / 8, Head % 8 / 4], m![Head % 4]>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_fp_div(&total)
            .vector_widen_concat::<m![Rows % 8, Head / 8], m![Head % 8]>()
            .vector_final()
            .commit_trim::<m![Head % 8]>()
            .commit_view(carry.tile_mut(t));
    }
}

/// Folds a block shifted by the running lse into every tile: as the old mass is 1 in that shift,
/// the state becomes `(carry + Σ partials) / (1 + total)` and the lse grows by `log(1 + total)`.
pub(crate) struct Absorb;

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    S: AxisName,
    T: AxisName,
    C: Tiles<KvHeads, Groups, Head>,
    R: Tiles<KvHeads, Groups, Head>,
>
    Op<(
        (&mut C, &mut R),
        &DmTensor<
            f32,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Symbol<T> }, { Symbol<S> }, Rows % 8, Head],
        >,
        &Running<KvHeads, Groups, T>,
    )> for Absorb
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((carry, running), partials, total): (
            (&mut C, &mut R),
            &DmTensor<
                f32,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, { Symbol<S> }, Rows % 8, Head],
            >,
            &Running<KvHeads, Groups, T>,
        ),
    ) {
        type Panels<Head, T, S> = m![{ TileOf<T> }, { Symbol<S> }, Rows % 8, Head];
        let held: VrfTensor<
            f32,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Symbol<T> }, Rows % 8, 1 # 8],
        > = context
            .sub
            .begin(running.group())
            .fetch::<m![{ Symbol<T> }, Rows % 8], m![1 # 8]>()
            .collect::<m![{ Symbol<T> }, Rows % 8], m![1 # 8]>()
            .to_vrf();
        let mass: Running<KvHeads, Groups, T> = context
            .main
            .begin(total.view())
            .fetch::<m![{ Symbol<T> }, Rows % 8], m![1 # 8]>()
            .collect::<m![{ Symbol<T> }, Rows % 8], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, 1.0f32)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        context
            .main
            .begin(mass.view())
            .fetch::<m![{ Symbol<T> }, Rows % 8], m![1 # 8]>()
            .collect::<m![{ Symbol<T> }, Rows % 8], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_unary(FpUnaryOp::Log)
            .vector_fp_binary(FpBinaryOp::AddF, &held)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(running.group_mut());
        let kept: Running<KvHeads, Groups, T> = context
            .main
            .begin(mass.view())
            .fetch::<m![{ Symbol<T> }, Rows % 8], m![1 # 8]>()
            .collect::<m![{ Symbol<T> }, Rows % 8], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_unary(FpUnaryOp::Log)
            .vector_fp_unary(FpUnaryOp::NegExp)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        #[unroll]
        for t in 0..T::SIZE {
            let kept = Op::forward(
                &Operand,
                &mut *context,
                kept.view()
                    .tile::<m![{ Symbol<T> }], 1, TileStatistic<T>>(t),
            );
            let old: VrfTensor<
                f32,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![Rows % 8, Head],
            > = context
                .main
                .begin(carry.tile(t))
                .fetch::<m![Rows % 8, Head / 8], m![Head % 8]>()
                .collect::<m![Rows % 8, Head / 8], m![Head % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Rows % 8, Head / 4], m![Head % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &kept)
                .vector_widen_concat::<m![Rows % 8, Head / 8], m![Head % 8]>()
                .vector_final()
                .to_vrf(&mut context.sub);
            let divisor = Op::forward(
                &Operand,
                &mut *context,
                mass.view()
                    .tile::<m![{ Symbol<T> }], 1, TileStatistic<T>>(t),
            );
            context.main.begin(partials.view().tile::<m![{ Symbol<T> }], 1, Panels<Head, T, S>>(t))
                .fetch::<m![Rows % 8, Head / 8, { Symbol<S> }], m![Head % 8]>()
                .collect::<m![Rows % 8, Head / 8, { Symbol<S> }], m![Head % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Rows % 8, Head / 8, { Symbol<S> }, Head % 8 / 4],
                    m![Head % 4]>()
                .vector_intra_slice_reduce::<S, m![Rows % 8, Head / 8, Head % 8 / 4], m![Head % 4]>(
                    IntraSliceReduceOpF32::Add,
                )
                .vector_fp_div(&divisor)
                .vector_widen_concat::<m![Rows % 8, Head / 8], m![Head % 8]>()
                .vector_clip(ClipBinaryOpF32::Add, &old)
                .vector_final()
                .commit_trim::<m![Head % 8]>()
                .commit_view(carry.tile_mut(t));
        }
    }
}
