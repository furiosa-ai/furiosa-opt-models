use crate::ops::attention::softmax::merge::merge;
use crate::ops::vector::Zip;
use furiosa_opt_std::prelude::*;

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

type Statistics<G, Count = <G as Partition>::Rows> = VrfTensor<
    f32,
    Chip<G>,
    <G as Partition>::Cluster,
    <G as Partition>::Slices,
    m![{Count}, {<G as Partition>::Groups}, 1 # 8],
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

type Terms<G> = DmTensor<
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

pub(crate) type Attended<G> = DmTensor<
    bf16,
    Chip<G>,
    <G as Partition>::Cluster,
    <G as Partition>::Merged,
    m![{ <G as Partition>::Rows }, { <G as Partition>::Groups }, {
        <G as Partition>::Head
    }],
>;

/// Geometry and score scaling shared by shard-local attention and its softmax reduction.
pub(crate) trait Partition: Sized {
    type Chip: M;
    type Cluster: M;
    type Slices: M;
    type Agreed: M;
    type Merged: M;
    type Rows: M;
    type Groups: M;
    type Head: M;
    type Keys: M;
    type Key: AxisName;

    const SHARDS: usize;
    /// The immediate the logits are divided by; its reciprocal is exactly the score scale.
    const DIVISOR: f32;

    fn windows(
        device: &mut Device,
        rest: usize,
        mut window: impl FnMut(&mut Device, usize) -> (Partial<Self>, Local<Self>),
    ) -> Attended<Self> {
        let (mut carry, mut running) = window(device, 0);
        for at in 0..rest {
            let (partial, lse) = window(device, at + 1);
            (carry, running) = Self::fold(device, &carry, &running, &partial, &lse);
        }
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

    fn attend<Count: M, Q: M, K: M, B: M, P: M, L: M>(
        device: &mut Device,
        query: DmTensorView<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, Q>,
        key: DmTensorView<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, K>,
        value: DmTensorView<'_, bf16, Chip<Self>, Self::Cluster, Self::Slices, K>,
        bias: DmTensorView<'_, f32, Chip<Self>, Self::Cluster, Self::Slices, B>,
        partial: DmTensorViewMut<'_, f32, Chip<Self>, Self::Cluster, Self::Slices, P>,
        lse: DmTensorViewMut<'_, f32, Chip<Self>, Self::Cluster, Self::Slices, L>,
    ) {
        let logits: Slots<Self, Count> =
            qk_logits::<Self, Count, _, _, _>(device, query, key, bias)
                .cast::<bf16, m![{Self::Keys} % 8 # 16]>()
                .commit_trim::<m![{ Self::Keys } % 8]>()
                .commit();
        // Shard-local softmax avoids broadcasting scores and three inter-slice reductions.
        // Its sum returns as a register operand for normalizing the shard's weighted values.
        let max: Statistics<Self, Count> = device
            .main
            .begin(logits.view())
            .fetch::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 16],
                m![{ Self::Keys } % 16],
            >()
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
        let weight: Slots<Self, Count> = device
            .main
            .begin(logits.view())
            .fetch::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 16],
                m![{ Self::Keys } % 16],
            >()
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
            .cast::<bf16, m![{ Self::Keys } % 8 # 16]>()
            .commit_trim::<m![{ Self::Keys } % 8]>()
            .commit();
        // Empty shards need a finite divisor before their mass is marked absent.
        let sum: Local<Self, Count> = device
            .main
            .begin(weight.view())
            .fetch::<
                m![{ Count }, { Self::Groups }, { Self::Keys } / 16],
                m![{ Self::Keys } % 16],
            >()
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
        weighted_sum::<Self, Count, _, _>(device, &weight, &total, value, partial);
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
        let logits: Slots<Self, Count, f32> =
            qk_logits::<Self, Count, _, _, _>(device, query, key, bias)
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
        weighted_sum::<Self, Count, _, _>(device, &high, &total, value.clone(), a.view_mut());
        weighted_sum::<Self, Count, _, _>(device, &low, &total, value, b.view_mut());
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
                      terms: &mut Terms<Self>,
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
        let mut terms = Terms::<Self>::new();
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

const PRESENT: TagGuard =
    TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);

fn qk_logits<'a, G: Partition, Count: M, Q: M, K: M, B: M>(
    device: &'a mut Device,
    query: DmTensorView<'a, bf16, Chip<G>, G::Cluster, G::Slices, Q>,
    chunk: DmTensorView<'_, bf16, Chip<G>, G::Cluster, G::Slices, K>,
    bias: DmTensorView<'_, f32, Chip<G>, G::Cluster, G::Slices, B>,
) -> VectorFinalTensor<
    'a,
    { Tu::Main },
    f32,
    Chip<G>,
    G::Cluster,
    G::Slices,
    m![{ Count }, { G::Groups }, { G::Keys } / 8],
    m![{ G::Keys } % 8],
> {
    let mask: Mask<G, Count> = device
        .main
        .begin(bias.clone())
        .fetch::<m![{ Count }, { G::Groups }, { G::Keys } / 8], m![{ G::Keys } % 8]>()
        .collect::<m![{ Count }, { G::Groups }, { G::Keys } / 8], m![{ G::Keys } % 8]>()
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
                TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]),
                f32::from_bits(u32::MAX),
            ),
        )
        .vector_final()
        .to_vrf(&mut device.sub);
    let bias: Mask<G, Count> = device
        .main
        .begin(bias.clone())
        .fetch::<m![{ Count }, { G::Groups }, { G::Keys } / 8], m![{ G::Keys } % 8]>()
        .collect::<m![{ Count }, { G::Groups }, { G::Keys } / 8], m![{ G::Keys } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([Cmp::Equal(0.0f32); 4]))
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_logic(
            LogicBinaryOpF32::BitOr,
            Branched::imm(
                TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]),
                f32::NEG_INFINITY,
            ),
        )
        .vector_final()
        .to_vrf(&mut device.sub);
    let key: TrfTensor<
        bf16,
        Chip<G>,
        G::Cluster,
        G::Slices,
        m![{ G::Keys } % 8],
        m![{ Count }, { G::Keys } / 8, { G::Head }],
    > = device
        .sub
        .begin(chunk)
        .fetch::<m![
            { G::Keys } % 8,
            { Count },
            { G::Keys } / 8,
            { G::Head } / 16
        ], m![{ G::Head } % 16]>()
        .collect::<m![
            { G::Keys } % 8,
            { Count },
            { G::Keys } / 8,
            { G::Head } / 16
        ], m![{ G::Head } % 16]>()
        .to_trf();
    device
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
            &key,
        )
        .contract_packet::<m![1]>()
        .contract_time::<m![{ Count }, { G::Groups }, { G::Keys } / 8]>()
        .contract_lane::<m![{ Count }, { G::Groups }, { G::Keys } / 8], m![{ G::Keys } % 8]>(
            LaneMode::Interleaved,
        )
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, &mask)
        .vector_narrow_split::<
            m![{ Count }, { G::Groups }, { G::Keys } / 8, { G::Keys } % 8 / 4],
            m![{ G::Keys } % 4],
        >(
        )
        .vector_fp_div(G::DIVISOR)
        .vector_widen_concat::<m![{ Count }, { G::Groups }, { G::Keys } / 8], m![{ G::Keys } % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &bias)
        .vector_final()
}

fn weighted_sum<G: Partition, Count: M, K: M, P: M>(
    device: &mut Device,
    weight: &Slots<G, Count>,
    total: &Statistics<G, Count>,
    chunk: DmTensorView<'_, bf16, Chip<G>, G::Cluster, G::Slices, K>,
    output: DmTensorViewMut<'_, f32, Chip<G>, G::Cluster, G::Slices, P>,
) {
    let value: DmTensor<
        bf16,
        Chip<G>,
        G::Cluster,
        G::Slices,
        m![{ Count }, { G::Head }, { G::Keys }],
    > =
        device
            .main
            .begin(chunk)
            .fetch::<m![{ Count }, { G::Head } / 8, { G::Keys }], m![{ G::Head } % 8 # 16]>()
            .collect::<m![{ Count }, { G::Head } / 8, { G::Keys }], m![{ G::Head } % 8 # 16]>()
            .transpose::<
                m![{ Count }, { G::Head } / 8, { G::Keys } / 4, { G::Head } % 8],
                m![{ G::Keys } % 4 # 16],
            >()
            .commit_trim::<m![{ G::Keys } % 4]>()
            .commit();
    let value: TrfTensor<
        bf16,
        Chip<G>,
        G::Cluster,
        G::Slices,
        m![{ G::Head } % 8],
        m![{ Count }, { G::Head } / 8, { G::Keys }],
    > = device
        .sub
        .begin(value.view())
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
    device
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
        >()
        .vector_fp_div(total)
        .vector_widen_concat::<m![{ Count }, { G::Groups }, { G::Head } / 8], m![{ G::Head } % 8]>()
        .vector_final()
        .commit_trim::<m![{ G::Head } % 8]>()
        .commit_view(output);
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
