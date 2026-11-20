use super::*;
use crate::Op;
use crate::ops::linear::affine::output;

/// A launch's attended rows in HBM: `Groups` query heads of `Head` features for each of `KvHeads`.
type Attended<Reach, KvHeads, Groups, Head> =
    HbmTensor<bf16, Chip, m![Reach, KvHeads, Groups, Head]>;

/// What the first step leaves the second: the rows in DM, the resident weight, the quantized rows,
/// their scales, the partial sums, and the waves with their whole blocks, tail, first passes and
/// single pass.
type First<Hidden, QWidth, Rows, D> = (
    InDm<Hidden, QWidth, Rows>,
    Resident<Hidden, QWidth>,
    Source<Hidden, QWidth, Rows>,
    ReplicatedScales<Hidden, QWidth, Rows, D>,
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
pub struct Landed<
    Hidden: AxisName,
    QWidth: AxisName,
    Rows: AxisName,
    D: Scalar = bf16,
    S: Scalar = f32,
> {
    staged: Staged<Hidden, QWidth, S>,
    step: usize,
    first: Option<First<Hidden, QWidth, Rows, D>>,
    second: Option<Second<Hidden, QWidth, Rows>>,
}

/// The weights and their channel scales staged before any attended row arrives.
impl<
    Hidden: AxisName,
    QWidth: AxisName,
    F: quantize::Tokens,
    S: MaterializableScalar + FetchCast<f32>,
> Op<Begin<()>> for Stored<Hidden, QWidth, F, S>
{
    type Output = Landed<Hidden, QWidth, Span, F::Scale, S>;

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
impl<
    Hidden: AxisName,
    QWidth: AxisName,
    F: quantize::Tokens,
    S: MaterializableScalar + FetchCast<f32>,
> Op<Begin<Staged<Hidden, QWidth, S>>> for Stored<Hidden, QWidth, F, S>
{
    type Output = Landed<Hidden, QWidth, Span, F::Scale, S>;

    fn forward(
        &self,
        _: &mut Device,
        Begin(staged): Begin<Staged<Hidden, QWidth, S>>,
    ) -> Self::Output {
        Landed {
            staged,
            step: 0,
            first: None,
            second: None,
        }
    }
}

/// Refill the same weight and channel-scale buffers after the preceding layer's final O pass.
impl<H: AxisName, Q: AxisName> Op<&mut Landed<H, Q, Span, f32, bf16>> for channel::Stored<H, Q> {
    type Output = ();
    fn forward(&self, device: &mut Device, landed: &mut Landed<H, Q, Span, f32, bf16>) {
        let (weight, scales) = &mut landed.staged.weight;
        self.weight
            .1
            .view()
            .to_dm_view(&mut device.tdma, scales.view_mut());
        self.weight
            .0
            .view()
            .to_dm_view(&mut device.tdma, weight.view_mut());
    }
}

impl<H: AxisName, Q: AxisName, F: quantize::Tokens, S: MaterializableScalar + FetchCast<f32>>
    Stored<H, Q, F, S>
{
    /// Land O's input first. The residual can then land while the first projection pass runs.
    pub(crate) fn begin_span<
        Capacity: AxisName,
        Reach: AxisName,
        KvHeads: AxisName,
        Groups: AxisName,
        Head: AxisName,
        T,
    >(
        &self,
        context: &mut Device,
        landed: &mut Landed<H, Q, Span, F::Scale, S>,
        attended: &Attended<Reach, KvHeads, Groups, Head>,
        tile: Live<(), Capacity>,
        after_rows: impl FnOnce(&mut Device) -> T,
    ) -> T {
        let rows = tile.rows();
        let in_dm =
            input_rows::<H, Q, Capacity, Reach, KvHeads, Groups, Head>(context, attended, tile);
        let ready = after_rows(context);
        landed.step = 1;
        let (weight, _) = &landed.staged.weight;
        let mut source = Source::new();
        let mut scales = ReplicatedScales::new();
        let resident = Op::forward(&contract::Stage, &mut *context, weight);
        let waves = (rows + 31) / 32;
        F::first(context, &in_dm, (&mut source, &mut scales), rows);
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
            F::pass(context, &resident, &source, &mut partial, 0, rows);
        }
        landed.first = Some((
            in_dm,
            resident,
            source,
            scales,
            partial,
            [waves, whole, tail, first, single],
        ));
        ready
    }
}

/// One step of span tile `tile` of the attended rows; the sum once the third step gives it.
impl<
    'l,
    'a,
    Hidden,
    QWidth,
    Capacity,
    Reach,
    KvHeads,
    Groups,
    Head,
    F: quantize::Tokens,
    S: MaterializableScalar + FetchCast<f32>,
>
    Op<(
        &'l mut Landed<Hidden, QWidth, Span, F::Scale, S>,
        &'a Attended<Reach, KvHeads, Groups, Head>,
        Live<(), Capacity>,
    )> for Stored<Hidden, QWidth, F, S>
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
            &'l mut Landed<Hidden, QWidth, Span, F::Scale, S>,
            &'a Attended<Reach, KvHeads, Groups, Head>,
            Live<(), Capacity>,
        ),
    ) -> Self::Output {
        let rows = tile.rows();
        let staged = &landed.staged;
        if landed.step < 1 {
            self.begin_span(context, landed, attended, tile, |_| ());
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
                    F::rest(context, &in_dm, (&mut source, &mut scales), rows);
                }
                for block in 0..3 * whole / 4 {
                    Op::forward(
                        &contract::Pass::<32>,
                        &mut *context,
                        ((&resident, &source), (&mut partial, block + 1)),
                    );
                }
                for wave in 0..(tail + 3 + first) % 4 {
                    F::pass(
                        context,
                        &resident,
                        &source,
                        &mut partial,
                        4 * whole + single + wave,
                        rows,
                    );
                }
                landed.second = Some(exchange::<Hidden, QWidth, F::Scale, S, F::Packet>(
                    context, &partial, &scales, channels, rows, waves,
                ));
            }
            return Poll::Pending;
        }
        landed.step = 0;
        Poll::Ready(sum_halves(context, landed.second.take(), rows))
    }
}

fn input_rows<
    H: AxisName,
    Q: AxisName,
    Capacity: AxisName,
    Reach: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
>(
    context: &mut Device,
    attended: &Attended<Reach, KvHeads, Groups, Head>,
    tile: Live<(), Capacity>,
) -> InDm<H, Q, Span> {
    const { assert!(KvHeads::SIZE * Groups::SIZE * Head::SIZE == Q::SIZE) };
    // SAFETY: Q is the same contiguous KV-head, query-group and head product.
    let attended: HbmTensor<bf16, Chip, m![Reach, Q]> = unsafe { attended.view().alias() };
    Op::forward(
        &ToDm::<H>(PhantomData),
        &mut *context,
        output::ToDm::<Q, Span, Reach> {
            attended: &attended,
            start: tile.start(),
            live: (tile.count::<crate::tensor::axis::Rows<128>>(), tile.rows()),
            capacity: PhantomData,
        },
    )
}

fn exchange<
    Hidden: AxisName,
    QWidth: AxisName,
    D: MaterializableScalar + FetchCast<f32>,
    S: MaterializableScalar + FetchCast<f32>,
    P: M + Count,
>(
    context: &mut Device,
    partial: &tiles::Partial<Hidden, QWidth, Span>,
    scales: &ReplicatedScales<Hidden, QWidth, Span, D>,
    channels: &Channels<Hidden, QWidth, S>,
    rows: usize,
    waves: usize,
) -> Second<Hidden, QWidth, Span> {
    let scales = Op::forward(&scale::Gather::<P>(PhantomData), &mut *context, scales);
    let partial = Op::forward(
        &scale::Rescale,
        &mut *context,
        (partial, (&scales, channels), waves),
    );
    exchange_halves(context, &partial, rows, waves)
}
fn exchange_halves<Hidden: AxisName, QWidth: AxisName>(
    context: &mut Device,
    partial: &tiles::Partial<Hidden, QWidth, Span>,
    rows: usize,
    waves: usize,
) -> Second<Hidden, QWidth, Span> {
    let half = |context: &mut Device, indices| -> tiles::Half<Hidden, QWidth, Span> {
        partial
            .view()
            .asymmetric_cluster_slice::<m![Span % (Span / 4) / 64], _>(&mut context.sub, indices)
    };
    let (send, own) = (half(context, &[1, 0]), half(context, &[0, 1]));
    let few = if rows < 65 { 1 } else { 0 };
    let mut received = tiles::Half::new();
    for wave in 0..waves * few {
        send.view()
            .tile::<m![Span % 64 / 8], 1, m![
                Span % 64 / 8 = 1 # 8,
                Span / (Span / 4),
                Span % 8,
                Hidden % 16
            ]>(wave)
            .cluster_swap()
            .to_dm_view(
                &mut context.tdma,
                received.view_mut().tile::<m![Span % 64 / 8], 1, m![
                    Span % 64 / 8 = 1 #{!} 8,
                    Span / (Span / 4),
                    Span % 8,
                    Hidden % 16
                ]>(wave),
            );
    }
    let whole = (rows / 129).min(1);
    for _ in 0..(rows / 65).min(1) * if whole < 1 { 1 } else { 0 } {
        send.view()
            .tile::<m![Span % 64 / 32], 1, m![
                Span % 64 / 32 = 1 # 2,
                Span % 32 / 8,
                Span / (Span / 4),
                Span % 8,
                Hidden % 16
            ]>(0)
            .cluster_swap()
            .to_dm_view(
                &mut context.tdma,
                received.view_mut().tile::<m![Span % 64 / 32], 1, m![
                    Span % 64 / 32 = 1 #{!} 2,
                    Span % 32 / 8,
                    Span / (Span / 4),
                    Span % 8,
                    Hidden % 16
                ]>(0),
            );
    }
    for _ in 0..whole {
        send.view()
            .cluster_swap()
            .to_dm_view(&mut context.tdma, received.view_mut());
    }
    (own, received, waves)
}

fn sum_halves<Hidden: AxisName, QWidth: AxisName>(
    context: &mut Device,
    halves: Option<Second<Hidden, QWidth, Span>>,
    rows: usize,
) -> Owned<Hidden, Span> {
    type Rows = Span;
    let mut summed = tiles::Half::<Hidden, QWidth, Rows>::new();
    if let Some((own, received, waves)) = halves {
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
    unsafe { summed.reshape() }
}

impl<H: AxisName, Q: AxisName> channel::Stored<H, Q> {
    /// Consume the same prefetched weights with token rows in TRF on an items-only launch.
    pub(crate) fn decode_span<
        Capacity: AxisName,
        Reach: AxisName,
        KvHeads: AxisName,
        Groups: AxisName,
        Head: AxisName,
        T,
        A,
    >(
        &self,
        device: &mut Device,
        landed: &Landed<H, Q, Span, f32, bf16>,
        attended: &Attended<Reach, KvHeads, Groups, Head>,
        tile: Live<(), Capacity>,
        after_rows: impl FnOnce(&mut Device) -> T,
        after_first: impl FnOnce(&mut Device) -> A,
    ) -> (T, A, Owned<H, Span>) {
        let input = input_rows::<H, Q, _, _, _, _, _>(device, attended, tile);
        let residual = after_rows(device);
        let rows = tile.rows();
        let waves = (rows + 31) / 32;
        let (weight, channels) = &landed.staged.weight;
        let (source, scales) = channel::quantize(device, &input, rows);
        let mut partial = channel::DecodePartial::new();
        channel::decode_wave(device, weight, &source, &mut partial, 0, rows.min(256));
        let ahead = after_first(device);
        if rows > 256 {
            channel::decode_wave(device, weight, &source, &mut partial, 8, rows - 256);
        }
        let scales = Op::forward(&scale::Gather::<m![8]>(PhantomData), &mut *device, &scales);
        let partial = scale::decode(device, &partial, &scales, channels, waves);
        let halves = exchange_halves(device, &partial, rows, waves);
        (residual, ahead, sum_halves(device, Some(halves), rows))
    }
}
