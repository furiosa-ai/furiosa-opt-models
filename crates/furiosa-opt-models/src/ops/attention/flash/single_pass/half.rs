//! A unit's last span when its live keys fit the span's first half: those 1,024 keys over the same
//! 32 key slices, 32 a slice, so the span's scores, weights and values take half the work.
//!
//! - The visible keys are a prefix, so the keys past the half are absent and enter no sum; the
//!   running maximum then covers fewer absent keys, which may round the weights differently.
//! - The product of the half span is the full span's [`Product`], so the unit finishes as before.

use furiosa_opt_std::prelude::*;

use super::{
    Absorb, Accumulate, Agreed, AgreedTop, Attended, Calculate, Channel, Cluster, Decay, Duo, Head,
    Landed, LoadScore, Maximum, Multiply, OnlineSoftmax, Peak, Product, Query, QuerySlices, Raise,
    Ready, Register, Rows, Span, Sweep, Tail, Tokens, Trip, Unit, Visibility, Weigh, Whole,
};
use crate::Op;
use crate::tensor::parallel::Chip;

axes![HalfSpan = 1024];

/// Eight query groups of 512 rows by the half span's 32 key slices of 32 keys.
type Slices = m![Rows / 512, HalfSpan / 32];
/// The half span's keys or values, landed in every query group.
type Copies = m![8, HalfSpan / 32];

/// A unit's half span of keys or values in its query groups.
pub(super) type Key<Chips = Chip> = DmTensor<bf16, Chips, Cluster, Slices, m![HalfSpan % 32, Head]>;
type Transposed<Chips = Chip> = DmTensor<bf16, Chips, Cluster, Slices, m![Channel, HalfSpan % 32]>;
type KeyRegister<Chips = Chip> = TrfTensor<
    bf16,
    Chips,
    Cluster,
    Slices,
    m![HalfSpan % 8],
    m![HalfSpan % 32 / 16, Head / 16, HalfSpan % 16 / 8, Head % 16],
>;
type ValueRegister<Chips = Chip> = TrfTensor<
    bf16,
    Chips,
    Cluster,
    Slices,
    m![Channel % 8],
    m![HalfSpan % 32 / 16, Channel / 8, HalfSpan % 16],
>;
type Scores<Chips = Chip> = DmTensor<bf16, Chips, Cluster, Slices, m![Rows % 512, HalfSpan % 32]>;
type Seen<Chips = Chip> = DmTensor<bf16, Chips, Cluster, Slices, m![HalfSpan % 32]>;

/// What a unit lands of its half span.
pub(super) trait Halves<Pairs: AxisName>: Unit<Pairs> {
    /// The first half of span `span`'s keys or values of unit `unit`'s pairs, in their query
    /// groups.
    fn half<Spans: AxisName, Chips: M>(
        device: &mut Device,
        tokens: &Tokens<Spans, Pairs, Chips>,
        unit: usize,
        span: usize,
    ) -> Key<Chips>;
}

impl<Pairs: AxisName> Halves<Pairs> for Whole {
    fn half<Spans: AxisName, Chips: M>(
        device: &mut Device,
        tokens: &Tokens<Spans, Pairs, Chips>,
        pair: usize,
        span: usize,
    ) -> Key<Chips> {
        let keys = tokens
            .view()
            .tile::<m![Pairs], 1, m![Pairs = 1 # Pairs, Duo, Spans, Span, Head]>(pair)
            .tile::<m![Spans], 1, m![Pairs = 1 # Pairs, Duo, Spans = 1 # Spans, Span, Head]>(span)
            .tile::<m![Span], { HalfSpan::SIZE }, m![
                Pairs = 1 # Pairs,
                Duo,
                Spans = 1 # Spans,
                Span = 1024 # Span,
                Head
            ]>(0);
        // SAFETY: the tile's keys are the span's first half in order; only the axis is renamed.
        let keys = unsafe {
            keys.reshape::<Chips, m![Pairs = 1 # Pairs, Duo, Spans = 1 # Spans, HalfSpan # Span, Head]>()
        };
        let landed: DmTensor<bf16, Chips, Cluster, Copies, m![HalfSpan % 32, Head]> =
            keys.to_dm(&mut device.tdma);
        // SAFETY: every query group holds the same keys; only the broadcast axis is named.
        unsafe { landed.reshape() }
    }
}

impl<Pairs: AxisName> Halves<Pairs> for Tail {
    fn half<Spans: AxisName, Chips: M>(
        device: &mut Device,
        tokens: &Tokens<Spans, Pairs, Chips>,
        unit: usize,
        span: usize,
    ) -> Key<Chips> {
        let pairs = tokens
            .view()
            .tile::<m![Pairs], 4, m![Pairs = 4 # Pairs, Duo, Spans, Span, Head]>(
                (4 * unit).min(Pairs::SIZE - 4),
            );
        // SAFETY: pair p of the unit serves query rows p * 1024..; only the axis is renamed.
        let pairs =
            unsafe { pairs.reshape::<Chips, m![Rows / 1024 # Pairs, Duo, Spans, Span, Head]>() };
        let keys = pairs
            .tile::<m![Spans], 1, m![Rows / 1024 # Pairs, Duo, Spans = 1 # Spans, Span, Head]>(span)
            .tile::<m![Span], { HalfSpan::SIZE }, m![
                Rows / 1024 # Pairs,
                Duo,
                Spans = 1 # Spans,
                Span = 1024 # Span,
                Head
            ]>(0);
        // SAFETY: as for a whole unit's half.
        let keys = unsafe {
            keys.reshape::<Chips, m![Rows / 1024 # Pairs, Duo, Spans = 1 # Spans, HalfSpan # Span, Head]>()
        };
        let landed: DmTensor<
            bf16,
            Chips,
            Cluster,
            m![Rows / 1024, 2, HalfSpan / 32],
            m![HalfSpan % 32, Head],
        > = keys.to_dm(&mut device.tdma);
        // SAFETY: a pair's two query groups hold the same keys; only the broadcast axis is named.
        unsafe { landed.reshape() }
    }
}

struct Score<Chips: M>(KeyRegister<Chips>);

struct LoadHalf;

impl<Chips: M> Op<Key<Chips>> for LoadHalf {
    type Output = Score<Chips>;

    fn forward(&self, device: &mut Device, key: Key<Chips>) -> Self::Output {
        Score(
            device
                .sub
                .begin(key.view())
                .fetch::<m![
                    HalfSpan % 8,
                    HalfSpan % 32 / 16,
                    Head / 16,
                    HalfSpan % 16 / 8
                ], m![Head % 16]>()
                .collect::<m![
                    HalfSpan % 8,
                    HalfSpan % 32 / 16,
                    Head / 16,
                    HalfSpan % 16 / 8
                ], m![Head % 16]>()
                .to_trf(),
        )
    }
}

struct CalculateHalf;

impl<Chips: M> Op<(Score<Chips>, &Query<Chips>)> for CalculateHalf {
    type Output = Scores<Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (score, query): (Score<Chips>, &Query<Chips>),
    ) -> Self::Output {
        type Stream = m![
            Rows % 16 / 2,
            HalfSpan % 32 / 16,
            Rows % 512 / 16,
            Head / 16
        ];
        type Steps = m![{ Stream }, HalfSpan % 16 / 8];
        type Sums = m![
            Rows % 16 / 2,
            HalfSpan % 32 / 16,
            Rows % 512 / 16,
            HalfSpan % 16 / 8
        ];
        let mut scores = Scores::<Chips>::new();
        device
            .main
            .begin(query.view())
            .fetch::<m![Rows % 16 / 2, HalfSpan % 32 / 16, Head / 16], m![Rows % 2, Head % 16]>()
            .switch::<Slices, Stream>(SwitchConfig::Broadcast01 {
                slice1: 32,
                slice0: 1,
                time0: 8,
            })
            .collect::<m![{ Stream }, Rows % 2], m![Head % 16]>()
            .contract_outer::<Steps, m![Rows % 2, Head % 16], _, _, _>(&score.0)
            .contract_packet::<m![Rows % 2]>()
            .contract_time::<Sums>()
            .contract_lane::<m![{ Sums }, Rows % 2], m![HalfSpan % 8]>(LaneMode::Interleaved)
            .commit_trim::<m![HalfSpan % 8]>()
            .commit_cast::<bf16>()
            .commit_view(scores.view_mut());
        scores
    }
}

struct Weights<Chips: M> {
    values: Scores<Chips>,
}

struct RaiseHalf;

impl<Chips: M> Op<(&Scores<Chips>, AgreedTop<Chips>, f32)> for RaiseHalf {
    type Output = Peak<Chips>;

    /// As [`Raise`] over the half span's keys.
    fn forward(
        &self,
        device: &mut Device,
        (scores, held, scale): (&Scores<Chips>, AgreedTop<Chips>, f32),
    ) -> Self::Output {
        // SAFETY: a group's key slices hold the same running maximum; only the slice axis is renamed.
        let held: VrfTensor<f32, Chips, Cluster, Slices, m![Rows % 512]> =
            unsafe { held.reshape() };
        let raised: DmTensor<f32, Chips, Cluster, Agreed, m![Rows % 512]> = device
            .sub
            .begin(scores.view())
            .fetch::<m![Rows % 512, HalfSpan % 32 / 16], m![HalfSpan % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % 512, HalfSpan % 32 / 8], m![HalfSpan % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows % 512, HalfSpan % 32 / 4], m![HalfSpan % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
            .vector_intra_slice_reduce::<HalfSpan, m![Rows % 512], m![1 # 4]>(
                IntraSliceReduceOpF32::Max,
            )
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Max, &held)
            .vector_inter_slice_reduce::<Agreed, m![Rows % 512]>(InterSliceReduceOpF32::Max)
            .vector_final()
            .transpose::<m![Rows % 512 / 2], m![Rows % 2 # 8]>()
            .commit_trim::<m![Rows % 2]>()
            .commit();
        // SAFETY: every key slice of a group received the same maximum.
        unsafe { raised.reshape() }
    }
}

struct WeighHalf;

impl<Chips: M> Op<(OnlineSoftmax<Chips>, (Scores<Chips>, Peak<Chips>), f32)> for WeighHalf {
    type Output = (
        Weights<Chips>,
        OnlineSoftmax<Chips, (Peak<Chips>, Peak<Chips>)>,
    );

    fn forward(
        &self,
        device: &mut Device,
        (softmax, (scores, top), scale): (OnlineSoftmax<Chips>, (Scores<Chips>, Peak<Chips>), f32),
    ) -> Self::Output {
        let values = {
            // SAFETY: a group's key slices hold the same maximum; only the slice axis is renamed.
            let peak = unsafe {
                top.view()
                    .reshape::<Chips, Cluster, Slices, m![Rows % 512]>()
            };
            let operand: VrfTensor<f32, Chips, Cluster, Slices, m![Rows % 512]> = device
                .sub
                .begin(peak)
                .fetch::<m![Rows % 512 / 8], m![Rows % 8]>()
                .collect::<m![Rows % 512 / 8], m![Rows % 8]>()
                .to_vrf();
            device
                .sub
                .begin(scores.view())
                .fetch::<m![Rows % 512, HalfSpan % 32 / 16], m![HalfSpan % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![Rows % 512, HalfSpan % 32 / 8], m![HalfSpan % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Rows % 512, HalfSpan % 32 / 4], m![HalfSpan % 4]>()
                .vector_fp_ternary_with_mode(
                    FpTernaryOp::FmaF,
                    TernaryArgMode::Mode021,
                    (&operand, -scale),
                )
                .vector_fp_unary(FpUnaryOp::NegExp)
                .vector_widen_concat::<m![Rows % 512, HalfSpan % 32 / 8], m![HalfSpan % 8]>()
                .vector_final()
                .cast::<bf16, m![HalfSpan % 8 # 16]>()
                .commit_trim::<m![HalfSpan % 8]>()
                .commit()
        };
        (
            Weights { values },
            OnlineSoftmax {
                max: (softmax.max, top),
                sum: softmax.sum,
            },
        )
    }
}

struct RegisterHalf;

impl<Chips: M> Op<&Transposed<Chips>> for RegisterHalf {
    type Output = ValueRegister<Chips>;

    fn forward(&self, device: &mut Device, value: &Transposed<Chips>) -> Self::Output {
        device
            .main
            .begin(value.view())
            .fetch::<m![Channel % 8, HalfSpan % 32 / 16, Channel / 8], m![HalfSpan % 16]>()
            .collect::<m![Channel % 8, HalfSpan % 32 / 16, Channel / 8], m![HalfSpan % 16]>()
            .to_trf()
    }
}

struct VisibleHalf;

impl<Chips: M> Op<(&Key<Chips>, Seen<Chips>)> for VisibleHalf {
    type Output = Transposed<Chips>;

    /// `value`, a span's half, channel-major with every absent key's values cleared, its mass
    /// channels the keys' visibility `seen`.
    fn forward(
        &self,
        device: &mut Device,
        (value, seen): (&Key<Chips>, Seen<Chips>),
    ) -> Self::Output {
        let keep: VrfTensor<f32, Chips, Cluster, Slices, m![HalfSpan % 32]> = device
            .sub
            .begin(seen.view())
            .fetch::<m![HalfSpan % 32 / 16], m![HalfSpan % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![HalfSpan % 32 / 8], m![HalfSpan % 8]>()
            .to_vrf();
        let kept: Key<Chips> = device
            .sub
            .begin(value.view())
            .fetch::<m![HalfSpan % 32, Head / 16], m![Head % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![HalfSpan % 32, Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![HalfSpan % 32, Head / 4], m![Head % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &keep)
            .vector_widen_concat::<m![HalfSpan % 32, Head / 8], m![Head % 8]>()
            .vector_final()
            .cast::<bf16, m![Head % 8 # 16]>()
            .commit_trim::<m![Head % 8]>()
            .commit();
        let mut transposed = Transposed::<Chips>::new();
        device
            .sub
            .begin(kept.view())
            .fetch::<m![Head / 16, HalfSpan % 32], m![Head % 16]>()
            .collect::<m![Head / 16, HalfSpan % 32], m![Head % 16]>()
            .transpose::<m![Head / 16, HalfSpan % 32 / 4, Head % 16], m![HalfSpan % 4 # 16]>()
            .commit_trim::<m![HalfSpan % 4]>()
            .commit_view({
                let channels = transposed
                    .view_mut()
                    .tile::<m![Channel], { Head::SIZE }, m![Channel = 128 #{!} Channel, HalfSpan % 32]>(0);
                // SAFETY: the value channels are consecutive; only the axis is renamed.
                unsafe {
                    channels.reshape::<Chips, Cluster, Slices, m![Head #{!} Channel, HalfSpan % 32]>()
                }
            });
        let mass = transposed
            .view_mut()
            .tile::<m![Channel], 8, m![Channel = 8 #{!} Channel, HalfSpan % 32]>(Head::SIZE);
        // SAFETY: each mass channel holds the same visibility; only the broadcast axis is renamed.
        let mass = unsafe {
            mass.reshape::<Chips, Cluster, Slices, m![[8] #{!} Channel, HalfSpan % 32]>()
        };
        device
            .sub
            .begin(seen.view())
            .fetch::<m![8, HalfSpan % 32 / 16], m![HalfSpan % 16]>()
            .collect::<m![8, HalfSpan % 32 / 16], m![HalfSpan % 16]>()
            .commit_trim::<m![HalfSpan % 16]>()
            .commit_view(mass);
        transposed
    }
}

struct MultiplyHalf;

impl<Chips: M> Op<(Weights<Chips>, ValueRegister<Chips>)> for MultiplyHalf {
    type Output = Product<Chips>;

    /// As the full span's products: every row's weighted values and mass, summed over the group's
    /// key slices onto the row's own slice.
    fn forward(
        &self,
        device: &mut Device,
        (weights, value): (Weights<Chips>, ValueRegister<Chips>),
    ) -> Self::Output {
        let weights = weights.values;
        type Fetch = m![Rows % 16 / 2, Rows % 512 / 16, HalfSpan % 32 / 16];
        type Steps = m![{ Fetch }, Channel / 8];
        type Sums = m![Rows % 16 / 2, Rows % 512 / 16, Channel / 8];
        let mut product = Product::<Chips>::new();
        device
            .main
            .begin(weights.view())
            .fetch::<Fetch, m![Rows % 2, HalfSpan % 16]>()
            .collect::<m![{ Fetch }, Rows % 2], m![HalfSpan % 16]>()
            .contract_outer::<Steps, m![Rows % 2, HalfSpan % 16], _, _, _>(&value)
            .contract_packet::<m![Rows % 2]>()
            .contract_time::<Sums>()
            .contract_lane::<m![{ Sums }, Rows % 2], m![Channel % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<QuerySlices, m![Rows % 16 / 2, Channel / 8, Rows % 2]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .commit_trim::<m![Channel % 8]>()
            .commit_view(product.view_mut());
        product
    }
}

/// A unit's last trips when its last span holds no live key past its first half: the trip of the
/// span before it, landing the half; then the half while `following` lands the next unit.
pub(super) struct Finish<'a, U, S>(pub(super) core::marker::PhantomData<U>, pub(super) &'a S);

impl<
    Spans: AxisName,
    Pairs: AxisName,
    V: Visibility<Spans, Chips>,
    Chips: M,
    const SCALE: usize,
    const HALF: bool,
    U: Halves<Pairs>,
    F: FnOnce(&mut Device) -> Landed<Chips>,
    S: FnOnce(&mut Device, &Attended<Chips>),
>
    Op<(
        (Trip<Chips>, Query<Chips>, &mut super::Transposed<Chips>),
        (usize, F),
        S,
    )> for Finish<'_, U, Sweep<'_, Spans, Pairs, V, Chips, SCALE, HALF>>
{
    type Output = Ready<Chips>;

    fn forward(
        &self,
        device: &mut Device,
        ((trip, rows, transposed), (unit, following), store): (
            (Trip<Chips>, Query<Chips>, &mut super::Transposed<Chips>),
            (usize, F),
            S,
        ),
    ) -> Ready<Chips> {
        let (key, value, visible, live) = (self.1.key, self.1.value, self.1.visible, self.1.live);
        let scale = const { f32::from_bits(SCALE as u32) };
        // Trip live - 3: span live - 2's scores and span live - 3's values, landing the half.
        let half = (
            U::half(device, key, unit, live - 1),
            U::half(device, value, unit, live - 1),
        );
        let Trip {
            score,
            scored,
            previous,
            values,
            softmax,
        } = trip;
        let top = Raise.forward(device, (&scored, previous, scale));
        let next = Calculate.forward(device, (score, &rows));
        let (weights, softmax) = Weigh.forward(device, (softmax, (scored, top), scale));
        let vreg = Register.forward(device, &*transposed);
        let product = Multiply.forward(device, (weights, vreg));
        visible.stage(device, &values, live - 2, transposed);
        drop(values);
        let score = LoadHalf.forward(device, half.0);
        let (softmax, previous, _sum) = Absorb.forward(device, (softmax, product));
        let (scored, values) = (next, half.1);
        // The half's visibility lands ahead of the next unit, whose landings it would queue behind.
        let seen: Seen<Chips> = {
            let seen = visible
                .keys()
                .view()
                .tile::<m![Spans], 1, m![Spans = 1 # Spans, Span]>(live - 1)
                .tile::<m![Span], { HalfSpan::SIZE }, m![Spans = 1 # Spans, Span = 1024 # Span]>(0);
            // SAFETY: the tile's keys are the span's first half in order; only the axis is renamed.
            let seen = unsafe { seen.reshape::<Chips, m![Spans = 1 # Spans, HalfSpan # Span]>() };
            let seen: DmTensor<bf16, Chips, m![2], Copies, m![HalfSpan % 32]> =
                seen.to_dm(&mut device.tdma);
            // SAFETY: both clusters and every query group hold the same visibility; only broadcast
            // axes are named.
            unsafe { seen.reshape() }
        };
        // Span live - 2's values while Main scores the half.
        let Landed {
            rows: next_rows,
            keys,
            first,
            later,
            values: next_values,
        } = following(device);
        let top = Raise.forward(device, (&scored, previous, scale));
        let next = CalculateHalf.forward(device, (score, &rows));
        drop(rows);
        let (weights, softmax) = Weigh.forward(device, (softmax, (scored, top), scale));
        let vreg = Register.forward(device, &*transposed);
        let product = Multiply.forward(device, (weights, vreg));
        let score = LoadScore.forward(&mut device.sub, keys);
        let cleared = VisibleHalf.forward(device, (&values, seen));
        drop(values);
        let (softmax, previous, retired) = Absorb.forward(device, (softmax, product));
        let (scored, rows, values) = (next, next_rows, next_values);
        // The half, while Main scores the next unit's first span.
        let top = RaiseHalf.forward(device, (&scored, previous, scale));
        let next = Calculate.forward(device, (score, &rows));
        let (weights, softmax) = WeighHalf.forward(device, (softmax, (scored, top), scale));
        let vreg = RegisterHalf.forward(device, &cleared);
        drop(cleared);
        let product = MultiplyHalf.forward(device, (weights, vreg));
        {
            let first = first;
            visible.stage(device, &first, 0, transposed);
        }
        let (finished, sum) = {
            let OnlineSoftmax {
                max: (held, top),
                sum,
            } = softmax;
            let raised = Maximum.forward(device, &top);
            let factor = Decay.forward(device, (&held, &raised));
            drop((held, top, raised));
            let next = Accumulate.forward(&mut *device, (&sum, &product, &factor));
            drop((factor, product));
            let finished = {
                let softmax = OnlineSoftmax { max: (), sum: next };
                let sum = &softmax.sum;
                let ones = sum
                    .view()
                    .tile::<m![Channel], 8, m![Rows % 16, Channel = 8 # Channel]>(Head::SIZE);
                // SAFETY: every ones channel holds the row's mass.
                let mass = unsafe {
                    ones.reshape::<Chips, Cluster, QuerySlices, m![Rows % 16, [1 # 8] # Channel]>()
                };
                let total: VrfTensor<f32, Chips, Cluster, QuerySlices, m![Rows % 16, 1 # 8]> =
                    device
                        .sub
                        .begin(mass)
                        .fetch::<m![Rows % 16], m![1 # 8]>()
                        .collect::<m![Rows % 16], m![1 # 8]>()
                        .to_vrf();
                let values = sum
                    .view()
                    .tile::<m![Channel], { Head::SIZE }, m![Rows % 16, Channel = 128 # Channel]>(0);
                // SAFETY: the value channels are consecutive; only the axis is renamed.
                let values = unsafe {
                    values.reshape::<Chips, Cluster, QuerySlices, m![Rows % 16, Head # Channel]>()
                };
                device
                    .main
                    .begin(values)
                    .fetch::<m![Rows % 16, Head / 8], m![Head % 8]>()
                    .collect::<m![Rows % 16, Head / 8], m![Head % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Rows % 16, Head / 4], m![Head % 4]>()
                    .vector_fp_div(&total)
                    .vector_widen_concat::<m![Rows % 16, Head / 8], m![Head % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Head % 8 # 16]>()
                    .commit_trim::<m![Head % 8]>()
                    .commit()
            };
            (finished, sum)
        };
        // Retired sums keep their original allocation lifetimes through the unit's store.
        let _sums = (sum, retired);
        store(device, &finished);
        Ready {
            rows,
            scored: next,
            later,
            values,
        }
    }
}
