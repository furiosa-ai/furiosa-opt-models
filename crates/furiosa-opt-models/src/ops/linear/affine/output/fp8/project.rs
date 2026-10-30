use super::*;
use crate::Op;
use crate::ops::linear::affine::output;

/// A launch's attended rows in HBM: `Groups` query heads of `Head` features for each of `KvHeads`.
type Attended<Reach, KvHeads, Groups, Head> =
    HbmTensor<bf16, Chip, m![Reach, KvHeads, Groups, Head]>;

/// What the first step leaves the second: the rows in DM, the resident weight, the quantized rows,
/// their scales, the partial sums, and the waves with their whole blocks, tail, first passes and
/// single pass.
type First<Hidden, QWidth, Rows> = (
    InDm<Hidden, QWidth, Rows>,
    Resident<Hidden, QWidth>,
    Source<Hidden, QWidth, Rows>,
    ReplicatedScales<Hidden, QWidth, Rows>,
    tiles::Partial<Hidden, QWidth, Rows>,
    [usize; 5],
);

/// What the second step leaves the third: the half of the sums a cluster keeps, the half it
/// received, and the waves.
type Second<Hidden, QWidth, Rows> = (
    tiles::Half<Hidden, QWidth, Rows>,
    tiles::Half<Hidden, QWidth, Rows>,
    usize,
);

/// The quantized projection landed: its weights, staged once, the step of the tile in progress
/// and what each step leaves the next. A tile takes three steps, one a poll: the tile's rows
/// landed and the first block's first pass, then the other passes and the exchange of one half of
/// the clusters' partial sums, then the sum, which it gives.
pub struct Landed<Hidden: AxisName, QWidth: AxisName, Rows: AxisName> {
    staged: Staged<Hidden, QWidth>,
    step: usize,
    first: Option<First<Hidden, QWidth, Rows>>,
    second: Option<Second<Hidden, QWidth, Rows>>,
}

/// The weights and their channel scales staged before any attended row arrives.
impl<Hidden: AxisName, QWidth: AxisName> Op<Begin<()>> for Stored<Hidden, QWidth> {
    type Output = Landed<Hidden, QWidth, Span>;

    fn forward(&self, context: &mut Device, _: Begin<()>) -> Self::Output {
        let mut staged = Self::empty();
        Op::forward(self, context, &mut staged);
        Landed {
            staged,
            step: 0,
            first: None,
            second: None,
        }
    }
}

/// The projection over weights staged beforehand, e.g. while an attention's last history is in
/// flight.
impl<Hidden: AxisName, QWidth: AxisName> Op<Begin<Staged<Hidden, QWidth>>>
    for Stored<Hidden, QWidth>
{
    type Output = Landed<Hidden, QWidth, Span>;

    fn forward(
        &self,
        _: &mut Device,
        Begin(staged): Begin<Staged<Hidden, QWidth>>,
    ) -> Self::Output {
        Landed {
            staged,
            step: 0,
            first: None,
            second: None,
        }
    }
}

/// One step of span tile `tile` of the attended rows; the sum once the third step gives it.
impl<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head>
    Op<(
        &'l mut Landed<Hidden, QWidth, Span>,
        &'a Attended<Reach, KvHeads, Groups, Head>,
        Live<(), Capacity>,
    )> for Stored<Hidden, QWidth>
where
    Hidden: AxisName,
    QWidth: AxisName,
    Capacity: AxisName,
    Reach: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
{
    type Output = Poll<Owned<Hidden, Span>>;

    fn forward(
        &self,
        context: &mut Device,
        (landed, attended, tile): (
            &'l mut Landed<Hidden, QWidth, Span>,
            &'a Attended<Reach, KvHeads, Groups, Head>,
            Live<(), Capacity>,
        ),
    ) -> Self::Output {
        type Rows = Span;
        let rows = tile.rows();
        let staged = &landed.staged;
        if landed.step < 1 {
            const { assert!(KvHeads::SIZE * Groups::SIZE * Head::SIZE == QWidth::SIZE) };
            // SAFETY: QWidth is the same contiguous KV-head, query-group and head product.
            let attended: HbmTensor<bf16, Chip, m![Reach, QWidth]> =
                unsafe { attended.view().alias() };
            let in_dm = Op::forward(
                &ToDm::<Hidden>(PhantomData),
                &mut *context,
                output::ToDm::<QWidth, Rows, Reach> {
                    attended: &attended,
                    start: tile.start(),
                    live: (tile.count::<crate::tensor::axis::Rows<128>>(), tile.rows()),
                    capacity: PhantomData,
                },
            );
            landed.step = 1;
            let (weight, _) = &staged.weight;
            let mut source = Source::new();
            let mut scales = ReplicatedScales::new();
            let resident = Op::forward(&contract::Stage, &mut *context, weight);
            let waves = (rows + 31) / 32;
            Op::forward(
                &quantize::First,
                &mut *context,
                (&in_dm, (&mut source, &mut scales)),
            );
            let (whole, tail) = (waves / 4, waves % 4);
            let first = (whole + 3) / 4;
            // One pass of eight rows when no whole block exists.
            let single = if whole < 1 { 1 } else { 0 };
            let mut partial = tiles::Partial::new();
            if first > 0 {
                Op::forward(
                    &contract::Pass::<32>,
                    &mut *context,
                    ((&resident, &source), (&mut partial, 0)),
                );
            } else {
                Op::forward(
                    &contract::Pass::<8>,
                    &mut *context,
                    ((&resident, &source), (&mut partial, 0)),
                );
            }
            landed.first = Some((
                in_dm,
                resident,
                source,
                scales,
                partial,
                [waves, whole, tail, first, single],
            ));
            return Poll::Pending;
        }
        if landed.step < 2 {
            landed.step = 2;
            if let Some((in_dm, resident, mut source, mut scales, mut partial, counts)) =
                landed.first.take()
            {
                let [waves, whole, tail, first, single] = counts;
                let channels = &staged.weight.1;
                if waves > 4 {
                    Op::forward(
                        &quantize::All,
                        &mut *context,
                        ((&in_dm, (&mut source, &mut scales)), waves),
                    );
                }
                for block in 0..3 * whole / 4 {
                    Op::forward(
                        &contract::Pass::<32>,
                        &mut *context,
                        ((&resident, &source), (&mut partial, block + 1)),
                    );
                }
                for wave in 0..(tail + 3 + first) % 4 {
                    Op::forward(
                        &contract::Pass::<8>,
                        &mut *context,
                        (
                            (&resident, &source),
                            (&mut partial, 4 * whole + single + wave),
                        ),
                    );
                }
                let scales = Op::forward(&scale::Gather, &mut *context, &scales);
                let partial = Op::forward(
                    &scale::Rescale,
                    &mut *context,
                    (&partial, (&scales, channels), waves),
                );
                let half = |context: &mut Device, indices| -> tiles::Half<Hidden, QWidth, Rows> {
                    partial
                        .view()
                        .asymmetric_cluster_slice::<m![Rows % (Rows / 4) / 64], _>(
                            &mut context.sub,
                            indices,
                        )
                };
                let (send, own) = (half(context, &[1, 0]), half(context, &[0, 1]));
                let few = if rows < 65 { 1 } else { 0 };
                let mut received = tiles::Half::new();
                for wave in 0..waves * few {
                    send.view()
                        .tile::<m![Rows % 64 / 8], 1, m![
                            Rows % 64 / 8 = 1 # 8,
                            Rows / (Rows / 4),
                            Rows % 8,
                            Hidden % 16
                        ]>(wave)
                        .cluster_swap()
                        .to_dm_view(
                            &mut context.tdma,
                            received.view_mut().tile::<m![Rows % 64 / 8], 1, m![
                                Rows % 64 / 8 = 1 #{!} 8,
                                Rows / (Rows / 4),
                                Rows % 8,
                                Hidden % 16
                            ]>(wave),
                        );
                }
                let whole = (rows / 129).min(1);
                for _ in 0..(rows / 65).min(1) * if whole < 1 { 1 } else { 0 } {
                    send.view()
                        .tile::<m![Rows % 64 / 32], 1, m![
                            Rows % 64 / 32 = 1 # 2,
                            Rows % 32 / 8,
                            Rows / (Rows / 4),
                            Rows % 8,
                            Hidden % 16
                        ]>(0)
                        .cluster_swap()
                        .to_dm_view(
                            &mut context.tdma,
                            received.view_mut().tile::<m![Rows % 64 / 32], 1, m![
                                Rows % 64 / 32 = 1 #{!} 2,
                                Rows % 32 / 8,
                                Rows / (Rows / 4),
                                Rows % 8,
                                Hidden % 16
                            ]>(0),
                        );
                }
                for _ in 0..whole {
                    send.view()
                        .cluster_swap()
                        .to_dm_view(&mut context.tdma, received.view_mut());
                }
                landed.second = Some((own, received, waves));
            }
            return Poll::Pending;
        }
        landed.step = 0;
        let mut summed = tiles::Half::<Hidden, QWidth, Rows>::new();
        if let Some((own, received, waves)) = landed.second.take() {
            let (small, big) = ((rows + 767) / 512 % 2, (rows + 255) / 512);
            Add::of::<
                m![Rows % 64 / 8, Rows / (Rows / 4)],
                m![Rows % 8, Hidden % 16 / 8],
                m![Hidden % 8],
                Term,
                _,
                _,
                _,
                _,
            >(&mut *context, (&own, &received))
            .commit_prefix::<m![Rows % 64 / 8], _, _>(summed.view_mut(), waves * small + 8 * big);
        }
        // SAFETY: each cluster owns its physical row half in order.
        Poll::Ready(unsafe { summed.reshape() })
    }
}
