use crate::ops::attention::softmax::merge::merge;
use crate::ops::vector::Zip;
use crate::specialize::attention;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

type Chip<G> = <G as Partition>::Chip;

axes![Summand = 2];

type Slots<G, Count, D = bf16> = DmTensor<
    D,
    Chip<G>,
    <G as Partition>::Cluster,
    <G as Partition>::Slices,
    m![{ Count }, { <G as Partition>::Groups }, {
        <G as Partition>::Keys
    }],
>;

type Mask<G, Count> = VrfTensor<
    f32,
    Chip<G>,
    <G as Partition>::Cluster,
    <G as Partition>::Slices,
    m![{ Count }, { <G as Partition>::Groups }, {
        <G as Partition>::Keys
    }],
>;

pub(crate) type Local<G, Count = <G as Partition>::Rows> = DmTensor<
    f32,
    Chip<G>,
    <G as Partition>::Cluster,
    <G as Partition>::Slices,
    m![{Count}, {<G as Partition>::Groups}, 1 # 8],
>;

type Logits<'a, G, Count> = VectorFinalTensor<
    'a,
    { Tu::Main },
    f32,
    Chip<G>,
    <G as Partition>::Cluster,
    <G as Partition>::Slices,
    m![
        { Count },
        { <G as Partition>::Groups },
        { <G as Partition>::Keys } / 8
    ],
    m![{ <G as Partition>::Keys } % 8],
>;

type Statistics<G, Count = <G as Partition>::Rows> = VrfTensor<
    f32,
    Chip<G>,
    <G as Partition>::Cluster,
    <G as Partition>::Slices,
    m![{Count}, {<G as Partition>::Groups}, 1 # 8],
>;

/// A window's values transposed for the weighted sum's register fill, `Count` row groups of them.
pub(crate) type Transposed<G, Count = <G as Partition>::Rows> = DmTensor<
    bf16,
    Chip<G>,
    <G as Partition>::Cluster,
    <G as Partition>::Slices,
    m![{ Count }, { <G as Partition>::Head }, {
        <G as Partition>::Keys
    }],
>;

pub(crate) type Partial<G, Count = <G as Partition>::Rows> = DmTensor<
    f32,
    Chip<G>,
    <G as Partition>::Cluster,
    <G as Partition>::Slices,
    m![{ Count }, { <G as Partition>::Groups }, {
        <G as Partition>::Head
    }],
>;

type Summands<G> = DmTensor<
    f32,
    Chip<G>,
    <G as Partition>::Cluster,
    <G as Partition>::Slices,
    m![
        { <G as Partition>::Rows },
        { <G as Partition>::Groups },
        { <G as Partition>::Head } / 8,
        Summand,
        { <G as Partition>::Head } % 8
    ],
>;

pub(crate) type Attended<G, Merged = <G as Partition>::Merged> = DmTensor<
    bf16,
    Chip<G>,
    <G as Partition>::Cluster,
    Merged,
    m![{ <G as Partition>::Rows }, { <G as Partition>::Groups }, {
        <G as Partition>::Head
    }],
>;

/// A window's keys in the tensor registers and its score masks in the vector registers, for
/// `Count` row groups: all its scores read of where the window landed.
pub(crate) struct Registers<G: Partition, Count: M> {
    key: TrfTensor<
        bf16,
        Chip<G>,
        G::Cluster,
        G::Slices,
        m![{ G::Keys } % 8],
        m![{ Count }, { G::Keys } / 8, { G::Head }],
    >,
    masks: Masks<G, Count>,
}

/// A window's score mask and bias in the vector registers, for `Count` row groups: the slots its
/// scores keep.
pub(crate) struct Masks<G: Partition, Count: M> {
    mask: Mask<G, Count>,
    bias: Mask<G, Count>,
}

/// A pass's query rows in the tensor registers, a lane a query group, for `Count` row groups: each
/// window the pass scores streams its keys past them.
pub(crate) type Stationary<G, Count> = TrfTensor<
    bf16,
    Chip<G>,
    <G as Partition>::Cluster,
    <G as Partition>::Slices,
    m![{ <G as Partition>::Groups }],
    m![{ Count }, { <G as Partition>::Head }],
>;

/// An FP32 scale in the vector registers of every slice of partition `G`.
pub(crate) type Scale<G> =
    VrfTensor<f32, Chip<G>, <G as Partition>::Cluster, <G as Partition>::Slices, m![1 # 8]>;

/// A window's key and value scales where its cache holds codes: the factors of its scores and of
/// its weighted sum.
pub(crate) struct Scales<G: Partition>(pub Scale<G>, pub Scale<G>);

/// An element a window's values land in, transposed into BF16 for the weighted sum's registers.
pub(crate) trait Value: MaterializableScalar {
    fn transpose<G: Partition, Count: M, K: M, V: M>(
        device: &mut Device,
        value: DmTensorView<'_, Self, Chip<G>, G::Cluster, G::Slices, K>,
        values: DmTensorViewMut<'_, bf16, Chip<G>, G::Cluster, G::Slices, V>,
    );
}

/// BF16 values move as they are.
impl Value for bf16 {
    fn transpose<G: Partition, Count: M, K: M, V: M>(
        device: &mut Device,
        value: DmTensorView<'_, bf16, Chip<G>, G::Cluster, G::Slices, K>,
        values: DmTensorViewMut<'_, bf16, Chip<G>, G::Cluster, G::Slices, V>,
    ) {
        device
            .main
            .begin(value)
            .fetch::<m![
                { Count },
                { G::Keys } / 16,
                { G::Head } / 8,
                { G::Keys } % 16
            ], m![{ G::Head } % 8 # 16]>()
            .collect::<m![
                { Count },
                { G::Keys } / 16,
                { G::Head } / 8,
                { G::Keys } % 16
            ], m![{ G::Head } % 8 # 16]>()
            .transpose::<m![
                { Count },
                { G::Keys } / 16,
                { G::Head } / 8,
                { G::Keys } % 16 / 4,
                { G::Head } % 8
            ], m![{ G::Keys } % 4 # 16]>()
            .commit_trim::<m![{ G::Keys } % 4]>()
            .commit_view(values);
    }
}

/// E4M3 codes widen to BF16 on their way to the transpose, exactly: their scale is the weighted
/// sum's.
impl Value for f8e4m3 {
    fn transpose<G: Partition, Count: M, K: M, V: M>(
        device: &mut Device,
        value: DmTensorView<'_, f8e4m3, Chip<G>, G::Cluster, G::Slices, K>,
        values: DmTensorViewMut<'_, bf16, Chip<G>, G::Cluster, G::Slices, V>,
    ) {
        device
            .main
            .begin(value)
            .fetch::<m![
                { Count },
                { G::Keys } / 16,
                { G::Head } / 8,
                { G::Keys } % 16
            ], m![{ G::Head } % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![
                { Count },
                { G::Keys } / 16,
                { G::Head } / 8,
                { G::Keys } % 16
            ], m![{ G::Head } % 8]>()
            .cast::<bf16, m![{ G::Head } % 8 # 16]>()
            .transpose::<m![
                { Count },
                { G::Keys } / 16,
                { G::Head } / 8,
                { G::Keys } % 16 / 4,
                { G::Head } % 8
            ], m![{ G::Keys } % 4 # 16]>()
            .commit_trim::<m![{ G::Keys } % 4]>()
            .commit_view(values);
    }
}

/// One window of a history that spans at most `Rest` windows after the first.
#[derive(Clone, Debug)]
pub(crate) struct Window<Rest>(PhantomData<Rest>);

impl<Rest: AxisName> AxisName for Window<Rest> {
    const NAME: Ident = Ident::new("Window");
    const SIZE: usize = Rest::SIZE + 1;
}

/// How the windows of a history combine into its attention: each window's terms, its shard
/// partial and log-sum-exp, join the windows before it, and the joined windows merge over the
/// shards.
pub(crate) trait Terms<G: Partition>: Sized {
    /// The terms of a history's first window.
    fn first(device: &mut Device, terms: (Partial<G>, Local<G>)) -> Self;
    /// Window `at`'s terms, past the first, joined to the windows before it.
    fn join(&mut self, device: &mut Device, at: usize, terms: (Partial<G>, Local<G>));
    /// The attention over every window joined.
    fn merged(self, device: &mut Device) -> Attended<G>;
}

/// The windows folded into a running attention, each as it is attended.
pub(crate) struct Folded<G: Partition> {
    carry: Partial<G>,
    running: Local<G>,
}

impl<G: Partition> Terms<G> for Folded<G> {
    fn first(_: &mut Device, (carry, running): (Partial<G>, Local<G>)) -> Self {
        Self { carry, running }
    }

    fn join(&mut self, device: &mut Device, _: usize, (partial, lse): (Partial<G>, Local<G>)) {
        let (carry, running) = G::fold(device, &self.carry, &self.running, &partial, &lse);
        (self.carry, self.running) = (carry, running);
    }

    fn merged(self, device: &mut Device) -> Attended<G> {
        G::merged(device, (&self.carry, &self.running))
    }
}

/// Every window's terms, a slot a window, until one merge combines them over the windows and the
/// shards together.
pub(crate) struct Held<G: Partition, Rest: AxisName> {
    partials: DmTensor<
        f32,
        Chip<G>,
        G::Cluster,
        G::Slices,
        m![{ Symbol<Window<Rest>> }, { G::Rows }, { G::Groups }, { G::Head }],
    >,
    lses: DmTensor<
        f32,
        Chip<G>,
        G::Cluster,
        G::Slices,
        m![{ Symbol<Window<Rest>> }, { G::Rows }, { G::Groups }, 1 # 8],
    >,
}

impl<G: Partition, Rest: AxisName> Terms<G> for Held<G, Rest> {
    fn first(device: &mut Device, terms: (Partial<G>, Local<G>)) -> Self {
        // A slot past the history's windows keeps a log-sum-exp of negative infinity: no share.
        let mut lses = DmTensor::new();
        lses.view_mut().memset(f32::NEG_INFINITY, &mut device.sub);
        let mut held = Self {
            partials: DmTensor::new(),
            lses,
        };
        held.join(device, 0, terms);
        held
    }

    fn join(&mut self, device: &mut Device, at: usize, (partial, lse): (Partial<G>, Local<G>)) {
        device
            .main
            .begin(partial.view())
            .fetch::<m![{ G::Rows }, { G::Groups }, { G::Head } / 8], m![{ G::Head } % 8]>()
            .collect::<m![{ G::Rows }, { G::Groups }, { G::Head } / 8], m![{ G::Head } % 8]>()
            .commit_trim::<m![{ G::Head } % 8]>()
            .commit_view(self.partials.view_mut().tile::<m![{ Symbol<Window<Rest>> }], 1, m![
                { Symbol<Window<Rest>> } = 1 #{!} ({ Symbol<Window<Rest>> }), { G::Rows }, { G::Groups }, { G::Head }
            ]>(at));
        device
            .main
            .begin(lse.view())
            .fetch::<m![{ G::Rows }, { G::Groups }], m![1 # 8]>()
            .collect::<m![{ G::Rows }, { G::Groups }], m![1 # 8]>()
            .commit_trim::<m![1 # 8]>()
            .commit_view(self.lses.view_mut().tile::<m![{ Symbol<Window<Rest>> }], 1, m![
                { Symbol<Window<Rest>> } = 1 #{!} ({ Symbol<Window<Rest>> }), { G::Rows }, { G::Groups }, 1 # 8
            ]>(at));
    }

    fn merged(self, device: &mut Device) -> Attended<G> {
        // Each of a head's shard slices adds its part of the epsilon, so it is added once.
        merge::<
            Window<Rest>,
            m![{ Symbol<Window<Rest>> }],
            m![{ G::Rows }, { G::Groups }],
            G::Head,
            G::Agreed,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        >(
            device,
            (self.partials.view(), self.lses.view()),
            const { 1e-6f32 / G::SHARDS as f32 },
        )
    }
}

/// Geometry and score scaling shared by shard-local attention and its softmax reduction.
pub(crate) trait Partition: Sized {
    type Chip: M;
    type Cluster: M;
    type Slices: M;
    type Agreed: M;
    type Merged: M;
    /// The slices of a head's lanes and shards, which agree on a statistic where a pass spreads
    /// one row's windows over its lanes.
    type HeadSlices: M;
    /// A head's one slice over its lanes, its first lane's merged slot.
    type FirstLane: M;
    type Rows: M;
    type Groups: M;
    type Head: M;
    type Keys: M;
    type Key: AxisName;

    const SHARDS: usize;
    /// The immediate the logits are divided by; its reciprocal is exactly the score scale.
    const DIVISOR: f32;

    /// Every window of a history merged: `first` is the first window's history, which the caller
    /// issued, `land` issues window `at`'s, `attend` attends what they landed into the slot it is
    /// given, and `boundary` runs once the last window's history is in flight, so no window's
    /// landing queues behind its transfer. A history of at most `Rest` windows after the first
    /// holds every window's terms for one merge where they fit, else folds each into a running
    /// attention.
    fn windows<H, Rest: AxisName>(
        device: &mut Device,
        rest: usize,
        first: H,
        steps: (
            impl FnMut(&mut Device, usize) -> H,
            impl FnMut(&mut Device, H) -> (Partial<Self>, Local<Self>),
        ),
        boundary: impl FnMut(&mut Device),
    ) -> Attended<Self> {
        if const { holds::<Self, Rest>() } {
            Self::serial::<_, Held<Self, Rest>>(device, rest, first, steps, boundary)
        } else {
            Self::serial::<_, Folded<Self>>(device, rest, first, steps, boundary)
        }
    }

    /// [`Self::windows`] landing windows alternately in two places, `rest` (at most `Rest`) after
    /// the first, which is in `even` already; `prepare` runs before the next lands: its page mapping
    /// waits for every transfer.
    ///
    /// With `FOLLOW`, `follow` lands window `at` into `even`, or past the last window what follows
    /// it, once the last window is prepared: `attend` must then read only what `prepare` returned,
    /// or its window's place where a history spans two windows at most, whose first is attended
    /// before what follows lands into its place. A history of two windows or more lands what
    /// follows exactly once, one of a window never.
    ///
    /// `boundary` runs once, where the issue waits for a landing anyway, as a transfer issued while
    /// a window lands waits for it on both paths of a loop around it: before the last window is
    /// prepared, once the attention before it is issued; with `FOLLOW` before the first, as what
    /// follows may keep the transfers busy to the end.
    fn overlapped<P, V, F, Rest: AxisName, const FOLLOW: bool>(
        device: &mut Device,
        rest: usize,
        places: (&mut P, &mut P, &mut V),
        steps: (
            impl FnMut(&mut Device, usize, &mut P),
            impl FnMut(&mut Device, &P, &mut V) -> F,
            impl FnMut(&mut Device, (&P, F), &V) -> (Partial<Self>, Local<Self>),
        ),
        hooks: (
            impl FnMut(&mut Device),
            impl FnMut(&mut Device, usize, &mut P),
        ),
    ) -> Attended<Self> {
        if const { holds::<Self, Rest>() } {
            Self::paired::<_, _, _, Held<Self, Rest>, Rest, FOLLOW>(
                device, rest, places, steps, hooks,
            )
        } else {
            Self::paired::<_, _, _, Folded<Self>, Rest, FOLLOW>(device, rest, places, steps, hooks)
        }
    }

    /// [`Self::windows`] with one place, each window landed after the one before is attended;
    /// `boundary` runs behind the last window's landing, which the attention waits for anyway.
    fn serial<H, T: Terms<Self>>(
        device: &mut Device,
        rest: usize,
        first: H,
        (mut land, mut attend): (
            impl FnMut(&mut Device, usize) -> H,
            impl FnMut(&mut Device, H) -> (Partial<Self>, Local<Self>),
        ),
        mut boundary: impl FnMut(&mut Device),
    ) -> Attended<Self> {
        for _ in 0..1 - rest.min(1) {
            boundary(device);
        }
        let terms = attend(device, first);
        let mut terms = T::first(device, terms);
        for at in 0..rest {
            let window = land(device, at + 1);
            for _ in 0..1 - (rest - 1 - at).min(1) {
                boundary(device);
            }
            let window = attend(device, window);
            terms.join(device, at + 1, window);
        }
        terms.merged(device)
    }

    /// [`Self::overlapped`] with the windows' terms joined as `T` joins them.
    fn paired<P, V, F, T: Terms<Self>, Rest: AxisName, const FOLLOW: bool>(
        device: &mut Device,
        rest: usize,
        (even, odd, values): (&mut P, &mut P, &mut V),
        (mut land, mut prepare, mut attend): (
            impl FnMut(&mut Device, usize, &mut P),
            impl FnMut(&mut Device, &P, &mut V) -> F,
            impl FnMut(&mut Device, (&P, F), &V) -> (Partial<Self>, Local<Self>),
        ),
        (mut boundary, mut follow): (
            impl FnMut(&mut Device),
            impl FnMut(&mut Device, usize, &mut P),
        ),
    ) -> Attended<Self> {
        // A device loop whose bound folds to zero is refused, so fewer windows skip its loops.
        if const { !FOLLOW && Rest::SIZE > 0 } {
            for _ in 0..1 - rest.min(1) {
                boundary(device);
            }
        } else {
            boundary(device);
        }
        let window = prepare(device, even, values);
        if const { Rest::SIZE > 0 } {
            for _ in 0..rest.min(1) {
                land(device, 1, odd);
            }
        }
        let terms = attend(device, (even, window), values);
        let mut terms = T::first(device, terms);
        // Pair `p` attends windows `2p + 1` and `2p + 2`, each while the next one lands.
        if const { Rest::SIZE > 0 } {
            for pair in 0..(rest + 1) / 2 {
                let after = rest - 2 * pair;
                if const { !FOLLOW } {
                    for _ in 0..1 - (after - 1).min(1) {
                        boundary(device);
                    }
                }
                let window = prepare(device, odd, values);
                if const { FOLLOW } {
                    follow(device, 2 * pair + 2, even);
                } else if const { Rest::SIZE > 1 } {
                    for _ in 0..(after / 2).min(1) {
                        land(device, 2 * pair + 2, even);
                    }
                }
                let window = attend(device, (odd, window), values);
                terms.join(device, 2 * pair + 1, window);
                if const { Rest::SIZE > 1 } {
                    for _ in 0..(after / 2).min(1) {
                        if const { !FOLLOW } {
                            for _ in 0..1 - (after - 2).min(1) {
                                boundary(device);
                            }
                        }
                        let window = prepare(device, even, values);
                        if const { Rest::SIZE > 2 } {
                            for _ in 0..((after - 1) / 2).min(1) {
                                land(device, 2 * pair + 3, odd);
                            }
                        }
                        if const { FOLLOW } {
                            for _ in 0..1 - ((after - 1) / 2).min(1) {
                                follow(device, rest + 1, even);
                            }
                        }
                        let window = attend(device, (even, window), values);
                        terms.join(device, 2 * pair + 2, window);
                    }
                }
            }
        }
        terms.merged(device)
    }

    fn merged(
        device: &mut Device,
        (carry, running): (&Partial<Self>, &Local<Self>),
    ) -> Attended<Self> {
        // Each of a head's shard slices adds its part of the epsilon, so it is added once.
        merge::<
            Self::Key,
            m![1],
            m![{ Self::Rows }, { Self::Groups }],
            Self::Head,
            Self::Agreed,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        >(
            device,
            (carry.view(), running.view()),
            const { 1e-6f32 / Self::SHARDS as f32 },
        )
    }

    /// The attention of a lone row whose windows a pass's lanes hold, one a lane: each lane's
    /// window terms merged over its head's lanes and shards onto the head's first lane.
    fn spread(
        device: &mut Device,
        (partial, lse): (&Partial<Self>, &Local<Self>),
    ) -> Attended<Self, Self::FirstLane> {
        // Each of a head's lane and shard slices adds its part of the epsilon, so it is added once.
        merge::<
            Self::Key,
            m![1],
            m![{ Self::Rows }, { Self::Groups }],
            Self::Head,
            Self::HeadSlices,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        >(
            device,
            (partial.view(), lse.view()),
            const { 1e-6f32 / (attention::CLUSTER_LANES * Self::SHARDS) as f32 },
        )
    }

    /// A window's `value` rows, each a `D` its cache holds, transposed into `values` as BF16, as
    /// the shard's attention reads them, in runs of sixteen keys: a longer run holds one bank
    /// group past a DMA's NoC timeout.
    fn transpose<Count: M, K: M, V: M, D: Value>(
        device: &mut Device,
        value: DmTensorView<'_, D, Chip<Self>, Self::Cluster, Self::Slices, K>,
        values: DmTensorViewMut<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, V>,
    ) {
        D::transpose::<Self, Count, K, V>(device, value, values)
    }

    /// A window's `key` rows and its `bias` read into the registers its scores take them from.
    fn fill<Count: M, K: M, B: M>(
        device: &mut Device,
        key: DmTensorView<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, K>,
        bias: DmTensorView<'_, f32, Chip<Self>, Self::Cluster, Self::Slices, B>,
    ) -> Registers<Self, Count> {
        let masks = Self::mask::<Count, _>(device, bias);
        let key = device
            .sub
            .begin(key)
            .fetch::<m![
                { Self::Keys } % 8,
                { Count },
                { Self::Keys } / 8,
                { Self::Head } / 16
            ], m![{ Self::Head } % 16]>()
            .collect::<m![
                { Self::Keys } % 8,
                { Count },
                { Self::Keys } / 8,
                { Self::Head } / 16
            ], m![{ Self::Head } % 16]>()
            .to_trf();
        Registers { key, masks }
    }

    /// A window's `bias`, a row's, read once each query group into the mask and the bias its scores
    /// take.
    fn mask<Count: M, B: M>(
        device: &mut Device,
        bias: DmTensorView<'_, f32, Chip<Self>, Self::Cluster, Self::Slices, B>,
    ) -> Masks<Self, Count> {
        let mask = device
            .main
            .begin(bias.clone())
            .fetch::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>()
            .collect::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>(
            )
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Equal(0.0f32),
                Cmp::Equal(0.0f32),
                Cmp::Equal(0.0f32),
                Cmp::Equal(0.0f32),
            ]))
            .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
            .vector_logic(
                LogicBinaryOpF32::BitOr,
                Branched::imm(
                    TagGuard::matches([
                        BitReq::One,
                        BitReq::Ignore,
                        BitReq::Ignore,
                        BitReq::Ignore,
                    ]),
                    f32::from_bits(u32::MAX),
                ),
            )
            .vector_final()
            .to_vrf(&mut device.sub);
        let bias = device
            .main
            .begin(bias)
            .fetch::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>()
            .collect::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>(
            )
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([Cmp::Equal(0.0f32); 4]))
            .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
            .vector_logic(
                LogicBinaryOpF32::BitOr,
                Branched::imm(
                    TagGuard::matches([
                        BitReq::Zero,
                        BitReq::Ignore,
                        BitReq::Ignore,
                        BitReq::Ignore,
                    ]),
                    f32::NEG_INFINITY,
                ),
            )
            .vector_final()
            .to_vrf(&mut device.sub);
        Masks { mask, bias }
    }

    /// The shard's attention of `query` over one window's `registers` and its values transposed
    /// in `values`, their codes times the window's `scales` where it has them: `partial` and
    /// `lse`.
    fn attend<Count: M, Q: M, V: M, P: M, L: M>(
        device: &mut Device,
        query: DmTensorView<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, Q>,
        (registers, values, scales): (
            &Registers<Self, Count>,
            DmTensorView<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, V>,
            Option<&Scales<Self>>,
        ),
        (partial, lse): (
            DmTensorViewMut<'_, f32, Chip<Self>, Self::Cluster, Self::Slices, P>,
            DmTensorViewMut<'_, f32, Chip<Self>, Self::Cluster, Self::Slices, L>,
        ),
    ) {
        let key = scales.map(|Scales(key, _)| key);
        let logits: Slots<Self, Count> = qk_logits::<Self, Count, _>(device, query, registers, key)
            .cast::<bf16, m![{Self::Keys} % 8 # 16]>()
            .commit_trim::<m![{ Self::Keys } % 8]>()
            .commit();
        let (weight, total) = softmax::<Self, Count, _>(device, &logits, lse);
        let value = scales.map(|Scales(_, value)| value);
        weighted_sum::<Self, Count, _, _>(device, (&weight, &total, value), values, partial);
    }

    /// A pass's `query` rows read into the registers, a lane a query group, for every window the
    /// pass scores.
    fn station<Count: M, Q: M>(
        device: &mut Device,
        query: DmTensorView<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, Q>,
    ) -> Stationary<Self, Count> {
        device
            .sub
            .begin(query)
            .fetch::<m![{ Self::Groups }, { Count }, { Self::Head } / 16], m![{ Self::Head } % 16]>()
            .collect::<m![{ Self::Groups }, { Count }, { Self::Head } / 16], m![{ Self::Head } % 16]>(
            )
            .to_trf()
    }

    /// The softmax weights and their total of the scores of `query` over a window's `key` rows,
    /// which stream past it where they landed, each score kept where `masks` keep it and times the
    /// window's `scale` where its keys are codes; `lse` takes each row's log-sum-exp.
    fn score<Count: M, K: M, L: M>(
        device: &mut Device,
        query: &Stationary<Self, Count>,
        (key, masks, scale): (
            DmTensorView<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, K>,
            &Masks<Self, Count>,
            Option<&Scale<Self>>,
        ),
        lse: DmTensorViewMut<'_, f32, Chip<Self>, Self::Cluster, Self::Slices, L>,
    ) -> (Slots<Self, Count>, Statistics<Self, Count>) {
        // A key's scores leave the contraction a lane a group; the transpose lays them out a group
        // a row, as the softmax reads them.
        let scores: Slots<Self, Count, f32> = device
            .main
            .begin(key)
            .fetch::<m![{ Count }, { Self::Keys }, { Self::Head } / 16], m![{ Self::Head } % 16]>()
            .collect::<m![{ Count }, { Self::Keys }, { Self::Head } / 16], m![{ Self::Head } % 16]>(
            )
            .contract_outer::<
                m![{ Count }, { Self::Keys }, { Self::Head } / 32],
                m![{ Self::Head } % 32],
                _,
                _,
                _,
            >(query)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Count }, { Self::Keys }]>()
            .contract_lane::<m![{ Count }, { Self::Keys }], m![{ Self::Groups } # 8]>(
                LaneMode::Interleaved,
            )
            .transpose::<m![{ Count }, { Self::Keys } / 2, { Self::Groups }], m![{ Self::Keys } % 2 # 8]>(
            )
            .commit_trim::<m![{ Self::Keys } % 2]>()
            .commit();
        let logits: Slots<Self, Count> = logit::<Self, Count>(
            device
                .main
                .begin(scores.view())
                .fetch::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>()
                .collect::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>(
                )
                .vector_init(),
            masks,
            scale,
        )
        .cast::<bf16, m![{ Self::Keys } % 8 # 16]>()
        .commit_trim::<m![{ Self::Keys } % 8]>()
        .commit();
        softmax::<Self, Count, _>(device, &logits, lse)
    }

    /// The softmax `weight`s' sum of a window's `value` rows, which stream past them where they
    /// landed, over their `total`, each value times the window's `scale` where its values are codes.
    fn weigh<Count: M, V: M, P: M>(
        device: &mut Device,
        (weight, total, scale): (
            &Slots<Self, Count>,
            &Statistics<Self, Count>,
            Option<&Scale<Self>>,
        ),
        value: DmTensorView<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, V>,
        partial: DmTensorViewMut<'_, f32, Chip<Self>, Self::Cluster, Self::Slices, P>,
    ) {
        let weights: TrfTensor<
            bf16,
            Chip<Self>,
            Self::Cluster,
            Self::Slices,
            m![{ Self::Groups }],
            m![{ Count }, { Self::Keys }],
        > = device
            .sub
            .begin(weight.view())
            .fetch::<m![{ Self::Groups }, { Count }, { Self::Keys } / 16], m![{ Self::Keys } % 16]>()
            .collect::<m![{ Self::Groups }, { Count }, { Self::Keys } / 16], m![{ Self::Keys } % 16]>(
            )
            .to_trf();
        let sums = device
            .main
            .begin(value)
            .fetch::<m![{ Count }, { Self::Keys }, { Self::Head } / 16], m![{ Self::Head } % 16]>()
            .collect::<m![{ Count }, { Self::Keys }, { Self::Head } / 16], m![{ Self::Head } % 16]>(
            )
            .contract_outer::<
                m![{ Count }, { Self::Keys }, { Self::Head } / 32],
                m![{ Self::Head } % 32],
                _,
                _,
                _,
            >(&weights)
            .contract_packet::<m![{ Self::Head } % 32]>()
            .contract_time::<m![{ Count }, { Self::Head } / 32]>()
            .contract_lane::<
                m![{ Count }, { Self::Head } / 32, { Self::Groups }, { Self::Head } % 32 / 8],
                m![{ Self::Head } % 8],
            >(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![{ Count }, { Self::Head } / 32, { Self::Groups }, { Self::Head } % 32 / 4],
                m![{ Self::Head } % 4],
            >();
        type Wide<G, Count> = m![
            { Count },
            { <G as Partition>::Head } / 32,
            { <G as Partition>::Groups },
            { <G as Partition>::Head } % 32 / 8
        ];
        type Lanes<G> = m![{ <G as Partition>::Head } % 8];
        let summed = match scale {
            None => sums
                .vector_fp_div(total)
                .vector_widen_concat::<Wide<Self, Count>, Lanes<Self>>()
                .vector_final(),
            Some(scale) => sums
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
                .vector_fp_div(total)
                .vector_widen_concat::<Wide<Self, Count>, Lanes<Self>>()
                .vector_final(),
        };
        summed.commit_trim::<Lanes<Self>>().commit_view(partial);
    }

    fn attend_precise<Count: M, Q: M, K: M, B: M, P: M, L: M>(
        device: &mut Device,
        query: DmTensorView<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, Q>,
        key: DmTensorView<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, K>,
        value: DmTensorView<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, K>,
        bias: DmTensorView<'_, f32, Chip<Self>, Self::Cluster, Self::Slices, B>,
        partial: DmTensorViewMut<'_, f32, Chip<Self>, Self::Cluster, Self::Slices, P>,
        lse: DmTensorViewMut<'_, f32, Chip<Self>, Self::Cluster, Self::Slices, L>,
    ) {
        let registers = Self::fill::<Count, _, _>(device, key, bias);
        let logits: Slots<Self, Count, f32> =
            qk_logits::<Self, Count, _>(device, query, &registers, None)
                .commit_trim::<m![{ Self::Keys } % 8]>()
                .commit();
        let max: Statistics<Self, Count> = device
            .main
            .begin(logits.view())
            .fetch::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>()
            .fetch_cast::<f32>()
            .collect::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 8],
                m![{ Self::Keys } % 8],
            >()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 4],
                m![{ Self::Keys } % 4],
            >()
            .vector_intra_slice_reduce::<Self::Key, m![{ Count }, { Self::Groups }], m![1 # 4]>(
                IntraSliceReduceOpF32::Max,
            )
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .to_vrf(&mut device.sub);
        let weight: Slots<Self, Count, f32> = device
            .main
            .begin(logits.view())
            .fetch::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>()
            .fetch_cast::<f32>()
            .collect::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 8],
                m![{ Self::Keys } % 8],
            >()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([Cmp::Equal(f32::NEG_INFINITY); 4]))
            .vector_narrow_split::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 4],
                m![{ Self::Keys } % 4],
            >()
            .vector_fp_binary(FpBinaryOp::SubF, Branched::rf(PRESENT, &max))
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 8],
                m![{ Self::Keys } % 8],
            >()
            .vector_final()
            .commit_trim::<m![{ Self::Keys } % 8]>()
            .commit();
        // Empty shards need a finite divisor before their mass is marked absent.
        let sum: Local<Self, Count> = device
            .main
            .begin(weight.view())
            .fetch::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>()
            .fetch_cast::<f32>()
            .collect::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 8],
                m![{ Self::Keys } % 8],
            >()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 4],
                m![{ Self::Keys } % 4],
            >()
            .vector_intra_slice_reduce::<Self::Key, m![{ Count }, { Self::Groups }], m![1 # 4]>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Add, 1e-6f32)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let total = operand::<Self, Count>(device, &sum);
        device
            .main
            .begin(sum.view())
            .fetch::<m![{ Count }, { Self::Groups }], m![1 # 8]>()
            .collect::<m![{ Count }, { Self::Groups }], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_unary(FpUnaryOp::Log)
            .vector_fp_binary(FpBinaryOp::AddF, &max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(lse);
        // Retain the softmax rounding residual through a second BF16 contraction.
        let high: Slots<Self, Count> = device
            .main
            .begin(weight.view())
            .fetch::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>()
            .collect::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>(
            )
            .cast::<bf16, m![{ Self::Keys } % 8 # 16]>()
            .commit_trim::<m![{ Self::Keys } % 8]>()
            .commit();
        let rounded: Mask<Self, Count> = device
            .sub
            .begin(high.view())
            .fetch::<m![{ Count }, { Self::Groups }, { Self::Keys } / 16], m![{ Self::Keys } % 16]>(
            )
            .fetch_cast::<f32>()
            .collect::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>(
            )
            .to_vrf();
        let low: Slots<Self, Count> = device
            .main
            .begin(weight.view())
            .fetch::<m![{ Count }, { Self::Groups }, { Self::Keys } / 8], m![{ Self::Keys } % 8]>()
            .collect::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 8],
                m![{ Self::Keys } % 8],
            >()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 4],
                m![{ Self::Keys } % 4],
            >()
            .vector_fp_binary(FpBinaryOp::SubF, &rounded)
            .vector_widen_concat::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 8],
                m![{ Self::Keys } % 8],
            >()
            .vector_final()
            .cast::<bf16, m![{ Self::Keys } % 8 # 16]>()
            .commit_trim::<m![{ Self::Keys } % 8]>()
            .commit();
        let mut a = Partial::<Self, Count>::new();
        let mut b = Partial::<Self, Count>::new();
        let mut high_values = Transposed::<Self, Count>::new();
        Self::transpose::<Count, _, _, _>(device, value.clone(), high_values.view_mut());
        weighted_sum::<Self, Count, _, _>(
            device,
            (&high, &total, None),
            high_values.view(),
            a.view_mut(),
        );
        let mut low_values = Transposed::<Self, Count>::new();
        Self::transpose::<Count, _, _, _>(device, value, low_values.view_mut());
        let low_terms = (&low, &total, None);
        weighted_sum::<Self, Count, _, _>(device, low_terms, low_values.view(), b.view_mut());
        device
            .main
            .begin_interleaved::<Summand, _, _, _, _, _>(a.view(), b.view())
            .fetch::<
                m![{ Count }, { Self::Groups }, { Self::Head } / 8, Summand],
                m![{ Self::Head } % 8],
            >()
            .collect::<
                m![{ Count }, { Self::Groups }, { Self::Head } / 8, Summand],
                m![{ Self::Head } % 8],
            >()
            .vector_init()
            .vector_zip::<
                Summand,
                m![{ Count }, { Self::Groups }, { Self::Head } / 8],
                m![{ Count }, { Self::Groups }, { Self::Head } / 4],
                m![{ Self::Head } % 4],
            >(FpBinaryOp::AddF)
            .vector_final()
            .commit_trim::<m![{ Self::Head } % 8]>()
            .commit_view(partial);
    }

    fn fold(
        device: &mut Device,
        carry: &Partial<Self>,
        running: &Local<Self>,
        partial: &Partial<Self>,
        lse: &Local<Self>,
    ) -> (Partial<Self>, Local<Self>) {
        let shifted =
            |device: &mut Device, statistic: &Local<Self>, top: &Statistics<Self>| -> Local<Self> {
                device
                    .main
                    .begin(statistic.view())
                    .fetch::<m![{ Self::Rows }, { Self::Groups }], m![1 # 8]>()
                    .collect::<m![{ Self::Rows }, { Self::Groups }], m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Comparison([Cmp::Equal(f32::NEG_INFINITY); 4]))
                    .vector_narrow_trim::<m![1 # 4]>()
                    .vector_fp_binary(FpBinaryOp::SubF, Branched::rf(PRESENT, top))
                    .vector_fp_unary(FpUnaryOp::Exp)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit()
            };
        let share = |device: &mut Device,
                     shifted: &Local<Self>,
                     divisor: &Statistics<Self>|
         -> Statistics<Self> {
            device
                .main
                .begin(shifted.view())
                .fetch::<m![{ Self::Rows }, { Self::Groups }], m![1 # 8]>()
                .collect::<m![{ Self::Rows }, { Self::Groups }], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_div(divisor)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .to_vrf(&mut device.sub)
        };
        let scaled = |device: &mut Device,
                      term: &Partial<Self>,
                      share: &Statistics<Self>,
                      terms: &mut Summands<Self>,
                      at: usize| {
            device
                    .main
                    .begin(term.view())
                    .fetch::<
                        m![{ Self::Rows }, { Self::Groups }, { Self::Head } / 8],
                        m![{ Self::Head } % 8],
                    >()
                    .collect::<
                        m![{ Self::Rows }, { Self::Groups }, { Self::Head } / 8],
                        m![{ Self::Head } % 8],
                    >()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<
                        m![
                            { Self::Rows },
                            { Self::Groups },
                            { Self::Head } / 8,
                            { Self::Head } % 8 / 4
                        ],
                        m![{ Self::Head } % 4],
                    >()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), share)
                    .vector_widen_concat::<
                        m![{ Self::Rows }, { Self::Groups }, { Self::Head } / 8],
                        m![{ Self::Head } % 8],
                    >()
                    .vector_final()
                    .commit_trim::<m![{ Self::Head } % 8]>()
                    .commit_view(terms.view_mut().tile::<m![Summand], 1, m![
                        { Self::Rows },
                        { Self::Groups },
                        { Self::Head } / 8,
                        Summand = 1 #{!} 2,
                        { Self::Head } % 8
                    ]>(at));
        };
        let held: Statistics<Self> = operand::<Self, _>(device, running);
        let top: Statistics<Self> = device
            .main
            .begin(lse.view())
            .fetch::<m![{ Self::Rows }, { Self::Groups }], m![1 # 8]>()
            .collect::<m![{ Self::Rows }, { Self::Groups }], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Max, &held)
            .vector_final()
            .to_vrf(&mut device.sub);
        let old = shifted(device, running, &top);
        let new = shifted(device, lse, &top);
        let new_operand: Statistics<Self> = operand::<Self, _>(device, &new);
        let total: Local<Self> = device
            .main
            .begin(old.view())
            .fetch::<m![{ Self::Rows }, { Self::Groups }], m![1 # 8]>()
            .collect::<m![{ Self::Rows }, { Self::Groups }], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Add, &new_operand)
            .vector_clip(ClipBinaryOpF32::Max, 1e-6f32)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let divisor: Statistics<Self> = operand::<Self, _>(device, &total);
        let mut terms = Summands::<Self>::new();
        let old_share = share(device, &old, &divisor);
        scaled(device, carry, &old_share, &mut terms, 0);
        let new_share = share(device, &new, &divisor);
        scaled(device, partial, &new_share, &mut terms, 1);
        let folded: Partial<Self> = device
            .main
            .begin(terms.view())
            .fetch::<
                m![{ Self::Rows }, { Self::Groups }, { Self::Head } / 8, Summand],
                m![{ Self::Head } % 8],
            >()
            .collect::<
                m![{ Self::Rows }, { Self::Groups }, { Self::Head } / 8, Summand],
                m![{ Self::Head } % 8],
            >()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![
                    { Self::Rows },
                    { Self::Groups },
                    { Self::Head } / 8,
                    Summand,
                    { Self::Head } % 8 / 4
                ],
                m![{ Self::Head } % 4],
            >()
            .vector_intra_slice_reduce::<
                Summand,
                m![{ Self::Rows }, { Self::Groups }, { Self::Head } / 8, { Self::Head } % 8 / 4],
                m![{ Self::Head } % 4],
            >(
                IntraSliceReduceOpF32::Add,
            )
            .vector_widen_concat::<
                m![{ Self::Rows }, { Self::Groups }, { Self::Head } / 8],
                m![{ Self::Head } % 8],
            >()
            .vector_final()
            .commit_trim::<m![{ Self::Head } % 8]>()
            .commit();
        let statistic: Local<Self> = device
            .main
            .begin(total.view())
            .fetch::<m![{ Self::Rows }, { Self::Groups }], m![1 # 8]>()
            .collect::<m![{ Self::Rows }, { Self::Groups }], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_unary(FpUnaryOp::Log)
            .vector_fp_binary(FpBinaryOp::AddF, &top)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        (folded, statistic)
    }
}

/// Whether a history of at most `Rest` windows after the first holds every window's terms.
pub(crate) const fn holds<G: Partition, Rest: AxisName>() -> bool {
    attention::holds(
        <G::Head as M>::SIZE,
        <G::Groups as M>::SIZE,
        <G::Rows as M>::SIZE,
        Rest::SIZE + 1,
    )
}

const PRESENT: TagGuard =
    TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);

/// The scores of `query` over a window's keys in the registers, each times the window's `scale`
/// where its keys are codes, before the score divisor.
fn qk_logits<'a, G: Partition, Count: M, Q: M>(
    device: &'a mut Device,
    query: DmTensorView<'a, bf16, Chip<G>, G::Cluster, G::Slices, Q>,
    Registers { key, masks }: &Registers<G, Count>,
    scale: Option<&Scale<G>>,
) -> Logits<'a, G, Count> {
    let scores = device
        .main
        .begin(query)
        .fetch::<m![{ Count }, { G::Groups }, { G::Head } / 16], m![{ G::Head } % 16]>()
        .collect::<m![{ Count }, { G::Groups }, { G::Head } / 16], m![{ G::Head } % 16]>()
        .contract_outer::<
            m![{ Count }, { G::Groups }, { G::Head } / 32, { G::Keys } / 8],
            m![{ G::Head } % 32],
            _,
            _,
            _,
        >(
            key,
        )
        .contract_packet::<m![1]>()
        .contract_time::<m![{ Count }, { G::Groups }, { G::Keys } / 8]>()
        .contract_lane::<m![{ Count }, { G::Groups }, { G::Keys } / 8], m![{ G::Keys } % 8]>(
            LaneMode::Interleaved,
        )
        .vector_init();
    logit::<G, Count>(scores, masks, scale)
}

/// A window's `scores` kept where `masks` keep them, each times the window's `scale` where its keys
/// are codes, over the score divisor: its logits, a dropped slot's negative infinity.
fn logit<'a, G: Partition, Count: M>(
    scores: VectorInitTensor<
        'a,
        { Tu::Main },
        f32,
        Chip<G>,
        G::Cluster,
        G::Slices,
        m![{ Count }, { G::Groups }, { G::Keys } / 8],
        m![{ G::Keys } % 8],
    >,
    Masks { mask, bias }: &Masks<G, Count>,
    scale: Option<&Scale<G>>,
) -> Logits<'a, G, Count> {
    let scores = scores
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, mask)
        .vector_narrow_split::<m![
            { Count },
            { G::Groups },
            { G::Keys } / 8,
            { G::Keys } % 8 / 4
        ], m![{ G::Keys } % 4]>();
    type Wide<G, Count> = m![
        { Count },
        { <G as Partition>::Groups },
        { <G as Partition>::Keys } / 8
    ];
    type Lanes<G> = m![{ <G as Partition>::Keys } % 8];
    match scale {
        None => scores
            .vector_fp_div(G::DIVISOR)
            .vector_widen_concat::<Wide<G, Count>, Lanes<G>>()
            .vector_clip(ClipBinaryOpF32::Add, bias)
            .vector_final(),
        Some(scale) => scores
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
            .vector_fp_div(G::DIVISOR)
            .vector_widen_concat::<Wide<G, Count>, Lanes<G>>()
            .vector_clip(ClipBinaryOpF32::Add, bias)
            .vector_final(),
    }
}

/// The softmax weights of a window's `logits` and their total, each row's log-sum-exp in `lse`.
fn softmax<G: Partition, Count: M, L: M>(
    device: &mut Device,
    logits: &Slots<G, Count>,
    lse: DmTensorViewMut<'_, f32, Chip<G>, G::Cluster, G::Slices, L>,
) -> (Slots<G, Count>, Statistics<G, Count>) {
    // Shard-local softmax avoids broadcasting scores and three inter-slice reductions.
    // Its sum returns as a register operand for normalizing the shard's weighted values.
    let max: Statistics<G, Count> = device
        .main
        .begin(logits.view())
        .fetch::<m![{ Count }, { G::Groups }, { G::Keys } / 16], m![{ G::Keys } % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Count }, { G::Groups }, { G::Keys } / 8], m![{ G::Keys } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Count }, { G::Groups }, { G::Keys } / 4], m![{ G::Keys } % 4]>()
        .vector_intra_slice_reduce::<G::Key, m![{ Count }, { G::Groups }], m![1 # 4]>(
            IntraSliceReduceOpF32::Max,
        )
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    let weight: Slots<G, Count> = device
        .main
        .begin(logits.view())
        .fetch::<m![{ Count }, { G::Groups }, { G::Keys } / 16], m![{ G::Keys } % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Count }, { G::Groups }, { G::Keys } / 8], m![{ G::Keys } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([Cmp::Equal(f32::NEG_INFINITY); 4]))
        .vector_narrow_split::<m![{ Count }, { G::Groups }, { G::Keys } / 4], m![{ G::Keys } % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, Branched::rf(PRESENT, &max))
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![{ Count }, { G::Groups }, { G::Keys } / 8], m![{ G::Keys } % 8]>()
        .vector_final()
        .cast::<bf16, m![{ G::Keys } % 8 # 16]>()
        .commit_trim::<m![{ G::Keys } % 8]>()
        .commit();
    // Empty shards need a finite divisor before their mass is marked absent.
    let sum: Local<G, Count> = device
        .main
        .begin(weight.view())
        .fetch::<m![{ Count }, { G::Groups }, { G::Keys } / 16], m![{ G::Keys } % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Count }, { G::Groups }, { G::Keys } / 8], m![{ G::Keys } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Count }, { G::Groups }, { G::Keys } / 4], m![{ G::Keys } % 4]>()
        .vector_intra_slice_reduce::<G::Key, m![{ Count }, { G::Groups }], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, 1e-6f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let total = operand::<G, Count>(device, &sum);
    device
        .main
        .begin(sum.view())
        .fetch::<m![{ Count }, { G::Groups }], m![1 # 8]>()
        .collect::<m![{ Count }, { G::Groups }], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Log)
        .vector_fp_binary(FpBinaryOp::AddF, &max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(lse);
    (weight, total)
}

/// The softmax `weight`s' sum of `values` over their `total`, each value times the window's
/// `scale` where its values are codes.
fn weighted_sum<G: Partition, Count: M, V: M, P: M>(
    device: &mut Device,
    (weight, total, scale): (&Slots<G, Count>, &Statistics<G, Count>, Option<&Scale<G>>),
    values: DmTensorView<'_, bf16, Chip<G>, G::Cluster, G::Slices, V>,
    output: DmTensorViewMut<'_, f32, Chip<G>, G::Cluster, G::Slices, P>,
) {
    let value: TrfTensor<
        bf16,
        Chip<G>,
        G::Cluster,
        G::Slices,
        m![{ G::Head } % 8],
        m![{ Count }, { G::Head } / 8, { G::Keys }],
    > = device
        .sub
        .begin(values)
        .fetch::<m![
            { G::Head } % 8,
            { Count },
            { G::Head } / 8,
            { G::Keys } / 16
        ], m![{ G::Keys } % 16]>()
        .collect::<m![
            { G::Head } % 8,
            { Count },
            { G::Head } / 8,
            { G::Keys } / 16
        ], m![{ G::Keys } % 16]>()
        .to_trf();
    let sums = device
        .main
        .begin(weight.view())
        .fetch::<m![{ Count }, { G::Groups }, { G::Keys } / 16], m![{ G::Keys } % 16]>()
        .collect::<m![{ Count }, { G::Groups }, { G::Keys } / 16], m![{ G::Keys } % 16]>()
        .contract_outer::<
            m![{ Count }, { G::Groups }, { G::Keys } / 16, { G::Head } / 8],
            m![{ G::Keys } % 16],
            _,
            _,
            _,
        >(
            &value,
        )
        .contract_packet::<m![1]>()
        .contract_time::<m![{ Count }, { G::Groups }, { G::Head } / 8]>()
        .contract_lane::<m![{ Count }, { G::Groups }, { G::Head } / 8], m![{ G::Head } % 8]>(
            LaneMode::Interleaved,
        )
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<
            m![{ Count }, { G::Groups }, { G::Head } / 8, { G::Head } % 8 / 4],
            m![{ G::Head } % 4],
        >();
    type Wide<G, Count> = m![
        { Count },
        { <G as Partition>::Groups },
        { <G as Partition>::Head } / 8
    ];
    type Lanes<G> = m![{ <G as Partition>::Head } % 8];
    let summed = match scale {
        None => sums
            .vector_fp_div(total)
            .vector_widen_concat::<Wide<G, Count>, Lanes<G>>()
            .vector_final(),
        Some(scale) => sums
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
            .vector_fp_div(total)
            .vector_widen_concat::<Wide<G, Count>, Lanes<G>>()
            .vector_final(),
    };
    summed.commit_trim::<Lanes<G>>().commit_view(output);
}

fn operand<G: Partition, Count: M>(
    device: &mut Device,
    scalar: &Local<G, Count>,
) -> Statistics<G, Count> {
    device
        .sub
        .begin(scalar.view())
        .fetch::<m![{ Count }, { G::Groups }], m![1 # 8]>()
        .collect::<m![{ Count }, { G::Groups }], m![1 # 8]>()
        .to_vrf()
}
