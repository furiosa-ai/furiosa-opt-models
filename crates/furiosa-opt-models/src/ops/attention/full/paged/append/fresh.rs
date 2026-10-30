use super::*;
use crate::Op;

pub(crate) type ValueSlice<KvHeads, Head> = m![KvHeads % 4, Head / 16, Cache / 16];
/// One panel's values staged one 16 × 16 tile per slice.
pub(crate) type Staged<KvHeads, Head> =
    DmTensor<bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, m![Cache % 16, Head % 16]>;
/// Where a panel's keys wait in data memory before their switch into the register file.
pub(crate) type KeyStage<KvHeads> = m![KvHeads % 4, Cache / 8 % 8, Cache / 64, Cache / 2 % 4];
type ValueStage<KvHeads> = m![KvHeads % 4, Cache % 16 / 2, Cache / 16];
pub(crate) type Keys<KvHeads, Groups, Head> = TrfTensor<
    bf16,
    Chip,
    Cluster<KvHeads>,
    Slice<KvHeads, Groups>,
    m![Cache % 8],
    m![
        Head % 32 / 16,
        Cache / 8 % 8,
        Cache / 64,
        Head / 32,
        Head % 16
    ],
>;
pub(crate) type Values<KvHeads, Groups, Head> = TrfTensor<
    bf16,
    Chip,
    Cluster<KvHeads>,
    Slice<KvHeads, Groups>,
    m![Head % 8],
    m![Head % 16 / 8, Head / 16, Cache],
>;

/// Fills the register file with one panel's keys from HBM.
pub(crate) struct LoadKeys<KvHeads, Groups, Head>(pub(crate) PhantomData<(KvHeads, Groups, Head)>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, K: M> Op<HbmTensorView<'_, bf16, Chip, K>>
    for LoadKeys<KvHeads, Groups, Head>
{
    type Output = Keys<KvHeads, Groups, Head>;

    fn forward(
        &self,
        context: &mut Device,
        keys: HbmTensorView<'_, bf16, Chip, K>,
    ) -> Self::Output {
        let keys: DmTensor<bf16, Chip, Cluster<KvHeads>, KeyStage<KvHeads>, m![Cache % 2, Head]> =
            keys.to_dm(&mut context.tdma);
        Op::forward(&FillKeys::<Groups, Head>(PhantomData), context, keys.view())
    }
}

/// Fills the register file with one panel's keys already staged in data memory.
pub(crate) struct FillKeys<Groups, Head>(pub(crate) PhantomData<(Groups, Head)>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, E: M>
    Op<DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, KeyStage<KvHeads>, E>>
    for FillKeys<Groups, Head>
{
    type Output = Keys<KvHeads, Groups, Head>;

    fn forward(
        &self,
        context: &mut Device,
        keys: DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, KeyStage<KvHeads>, E>,
    ) -> Self::Output {
        let keys: DmTensor<
            bf16,
            Chip,
            Cluster<KvHeads>,
            Switched<KvHeads, Head>,
            m![Cache % 8, Head % 32],
        > = SwitchKeys::switched(&mut context.main, keys).commit();
        Op::forward(&RegisterKeys::<Groups>(PhantomData), context, keys.view())
    }
}

/// The slices a panel's switched keys occupy, the layout their register-file fill reads.
pub(crate) type Switched<KvHeads, Head> = m![KvHeads % 4, Cache / 8 % 8, Cache / 64, Head / 32];

/// Switches one panel's staged keys into `output`, where their fills read them.
pub(crate) struct SwitchKeys;

impl<KvHeads: AxisName, Head: AxisName, E: M, O: M>
    Op<(
        DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, KeyStage<KvHeads>, E>,
        DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, Switched<KvHeads, Head>, O>,
    )> for SwitchKeys
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (keys, output): (
            DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, KeyStage<KvHeads>, E>,
            DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, Switched<KvHeads, Head>, O>,
        ),
    ) {
        Self::switched::<KvHeads, Head, E>(&mut context.main, keys).commit_view(output);
    }
}

impl SwitchKeys {
    /// One panel's staged keys switched into the slices their fills read, ready to commit.
    fn switched<'a, KvHeads: AxisName, Head: AxisName, E: M>(
        main: &'a mut TuContext<{ Tu::Main }>,
        keys: DmTensorView<'a, bf16, Chip, Cluster<KvHeads>, KeyStage<KvHeads>, E>,
    ) -> CommitTrimTensor<
        'a,
        { Tu::Main },
        bf16,
        Chip,
        Cluster<KvHeads>,
        Switched<KvHeads, Head>,
        m![Cache % 2, Head % 32 / 16, Cache / 2 % 4],
        m![Head % 16],
    > {
        main.begin(keys)
            .fetch::<m![Cache % 2, Head / 16], m![Head % 16]>()
            .switch::<Switched<KvHeads, Head>, m![Cache % 2, Head % 32 / 16, Cache / 2 % 4]>(
                SwitchConfig::InterTranspose {
                    slice1: 4,
                    slice0: 1,
                    time0: 2,
                },
            )
            .collect::<m![Cache % 2, Head % 32 / 16, Cache / 2 % 4], m![Head % 16]>()
            .commit_trim::<m![Head % 16]>()
    }
}

/// Fills the register file with one panel's switched keys.
pub(crate) struct RegisterKeys<Groups>(pub(crate) PhantomData<Groups>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, E: M>
    Op<DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Switched<KvHeads, Head>, E>>
    for RegisterKeys<Groups>
{
    type Output = Keys<KvHeads, Groups, Head>;

    fn forward(
        &self,
        context: &mut Device,
        keys: DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Switched<KvHeads, Head>, E>,
    ) -> Self::Output {
        context
            .sub
            .begin(keys)
            .fetch::<m![Cache % 8, Head % 32 / 16], m![Head % 16]>()
            .switch::<Slice<KvHeads, Groups>, m![
                Cache % 8,
                Head % 32 / 16,
                Cache / 8 % 8,
                Cache / 64,
                Head / 32
            ]>(SwitchConfig::Broadcast01 {
                slice1: 8,
                slice0: 8,
                time0: 1,
            })
            .collect::<m![
                Cache % 8,
                Head % 32 / 16,
                Cache / 8 % 8,
                Cache / 64,
                Head / 32
            ], m![Head % 16]>()
            .to_trf()
    }
}

/// Scaled scores of one 128-key panel, written into `output`, a panel or a tile of a wider row.
pub(crate) struct FreshScores;

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, Q: M, E: M>
    Op<(
        &Attention<KvHeads, Groups, Head>,
        DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
        &Keys<KvHeads, Groups, Head>,
        DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
    )> for FreshScores
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (attention, projected, key, output): (
            &Attention<KvHeads, Groups, Head>,
            DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
            &Keys<KvHeads, Groups, Head>,
            DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
        ),
    ) {
        context.main.begin(projected)
        .fetch::<m![Rows % 8 / 2, Cache / 8 % 8, Head / 16], m![Rows % 2, Head % 16]>()
        .collect::<m![Rows % 8 / 2, Cache / 8 % 8, Head / 16, Rows % 2], m![Head % 16]>()
        .contract_outer::<
            m![Rows % 8 / 2, Cache / 8 % 8, Head / 16, Cache / 64],
            m![Rows % 2, Head % 16],
            _,
            _,
            _,
        >(&key)
        .contract_packet::<m![Rows % 2]>()
        .contract_time::<m![Rows % 8 / 2, Cache / 8 % 8, Cache / 64]>()
        .contract_lane::<
            m![Rows % 8 / 2, Cache / 8 % 8, Cache / 64, Rows % 2],
            m![Cache % 8],
        >(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<
            m![Rows % 8 / 2, Cache / 8 % 8, Cache / 64, Rows % 2, Cache % 8 / 4],
            m![Cache % 4],
        >()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), attention.scale)
        .vector_widen_concat::<
            m![Rows % 8 / 2, Cache / 8 % 8, Cache / 64, Rows % 2],
            m![Cache % 8],
        >()
        .vector_final()
        .commit_trim::<m![Cache % 8]>()
        .commit_cast::<bf16>()
        .commit_view(output);
    }
}

/// Fills the register file with one panel's values from HBM.
pub(crate) struct LoadValues<KvHeads, Groups, Head>(
    pub(crate) PhantomData<(KvHeads, Groups, Head)>,
);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, V: M> Op<HbmTensorView<'_, bf16, Chip, V>>
    for LoadValues<KvHeads, Groups, Head>
{
    type Output = Values<KvHeads, Groups, Head>;

    fn forward(
        &self,
        context: &mut Device,
        values: HbmTensorView<'_, bf16, Chip, V>,
    ) -> Self::Output {
        let mut staged = Staged::<KvHeads, Head>::new();
        Op::forward(&StageValues, &mut *context, (values, staged.view_mut()));
        Op::forward(&FillValues::<Groups>(PhantomData), context, staged.view())
    }
}

/// One panel's values from HBM, repartitioned one 16 × 16 tile per slice into `output`.
pub(crate) struct StageValues;

impl<KvHeads: AxisName, Head: AxisName, V: M, O: M>
    Op<(
        HbmTensorView<'_, bf16, Chip, V>,
        DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, O>,
    )> for StageValues
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (values, output): (
            HbmTensorView<'_, bf16, Chip, V>,
            DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, O>,
        ),
    ) {
        let values: DmTensor<
            bf16,
            Chip,
            Cluster<KvHeads>,
            ValueStage<KvHeads>,
            m![Cache % 2, Head],
        > = values.to_dm(&mut context.tdma);
        context
            .main
            .begin(values.view())
            .fetch::<m![Cache % 2, Head / 16], m![Head % 16]>()
            .switch::<ValueSlice<KvHeads, Head>, m![Cache % 2, Cache % 16 / 2]>(
                SwitchConfig::InterTranspose {
                    slice1: 8,
                    slice0: 8,
                    time0: 1,
                },
            )
            .collect::<m![Cache % 2, Cache % 16 / 2], m![Head % 16]>()
            .commit_trim::<m![Head % 16]>()
            .commit_view(output);
    }
}

/// Restages one panel's values from HBM into `output` with every row past `last` zero: row
/// `last` of the causal tile marks the keys past it.
pub(crate) struct Cut;

impl<KvHeads: AxisName, Head: AxisName, V: M, O: M>
    Op<(
        (HbmTensorView<'_, bf16, Chip, V>, (&Causal, usize)),
        DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, O>,
    )> for Cut
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((values, (causal, last)), output): (
            (HbmTensorView<'_, bf16, Chip, V>, (&Causal, usize)),
            DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, O>,
        ),
    ) {
        let tokens: history::Tokens<KvHeads, Head> = values.to_dm(&mut context.tdma);
        // SAFETY: the two row axes have the same contiguous 128-element order.
        let past: DmTensor<
            i8,
            Chip,
            Cluster<KvHeads>,
            history::DmSlices<KvHeads>,
            m![Cache % Block],
        > = unsafe { causal.view().reshape::<Chip, m![Rows, Cache]>() }
            .tile::<m![Rows], 1, m![Rows = 1 # Rows, Cache]>(last)
            .to_dm(&mut context.tdma);
        let distance: history::Distance<KvHeads> = context
            .main
            .begin(past.view())
            .fetch::<m![Cache % Block / 8], m![Cache % 8]>()
            .fetch_cast::<i32>()
            .collect::<m![Cache % Block / 8], m![Cache % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxpSat, 1i32)
            .vector_final()
            .commit_trim::<m![Cache % 8]>()
            .commit();
        let ended = Op::forward(&history::End::<Head>(PhantomData), &mut *context, &distance);
        let zeroed = Op::forward(&history::Zero, &mut *context, (&tokens, &ended));
        Op::forward(&StageValues, context, (zeroed.view(), output));
    }
}

/// Transposes each slice's 16 × 16 value tile into `output`, where the fills read it.
pub(crate) struct TransposeValues;

impl<KvHeads: AxisName, Head: AxisName, E: M, O: M>
    Op<(
        DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, E>,
        DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, O>,
    )> for TransposeValues
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (values, output): (
            DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, E>,
            DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, O>,
        ),
    ) {
        Self::transposed(&mut context.main, values).commit_view(output);
    }
}

impl TransposeValues {
    /// Each slice's 16 x 16 value tile transposed, ready to commit.
    fn transposed<'a, KvHeads: AxisName, Head: AxisName, E: M>(
        main: &'a mut TuContext<{ Tu::Main }>,
        values: DmTensorView<'a, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, E>,
    ) -> CommitTrimTensor<
        'a,
        { Tu::Main },
        bf16,
        Chip,
        Cluster<KvHeads>,
        ValueSlice<KvHeads, Head>,
        m![Cache % 16 / 4, Head % 16],
        m![Cache % 4],
    > {
        main.begin(values)
            .fetch::<m![Cache % 16], m![Head % 16]>()
            .collect::<m![Cache % 16], m![Head % 16]>()
            .transpose::<m![Cache % 16 / 4, Head % 16], m![Cache % 4 # 16]>()
            .commit_trim::<m![Cache % 4]>()
    }
}

/// Fills the register file with one panel's transposed values.
pub(crate) struct RegisterValues<Groups>(pub(crate) PhantomData<Groups>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, E: M>
    Op<DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, E>>
    for RegisterValues<Groups>
{
    type Output = Values<KvHeads, Groups, Head>;

    fn forward(
        &self,
        context: &mut Device,
        values: DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, E>,
    ) -> Self::Output {
        context
            .sub
            .begin(values)
            .fetch::<m![Head % 8, Head % 16 / 8], m![Cache % 16]>()
            .switch::<Slice<KvHeads, Groups>, m![Head % 8, Head % 16 / 8, Head / 16, Cache / 16]>(
                SwitchConfig::Broadcast01 {
                    slice1: 8,
                    slice0: 8,
                    time0: 1,
                },
            )
            .collect::<m![Head % 8, Head % 16 / 8, Head / 16, Cache / 16], m![Cache % 16]>()
            .to_trf()
    }
}

/// Fills the register file with one panel's values, repartitioned one 16 × 16 tile per slice.
pub(crate) struct FillValues<Groups>(pub(crate) PhantomData<Groups>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, E: M>
    Op<DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, E>>
    for FillValues<Groups>
{
    type Output = Values<KvHeads, Groups, Head>;

    fn forward(
        &self,
        context: &mut Device,
        values: DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, ValueSlice<KvHeads, Head>, E>,
    ) -> Self::Output {
        let values: DmTensor<
            bf16,
            Chip,
            Cluster<KvHeads>,
            ValueSlice<KvHeads, Head>,
            m![Head % 16, Cache % 16],
        > = TransposeValues::transposed(&mut context.main, values).commit();
        Op::forward(
            &RegisterValues::<Groups>(PhantomData),
            context,
            values.view(),
        )
    }
}
