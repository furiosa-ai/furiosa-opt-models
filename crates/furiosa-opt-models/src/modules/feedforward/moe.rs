//! The routing of a mixture of experts: the experts selected for each row, their probabilities,
//! and the selected rows weighted by them.
//!
//! - A row selects its `K` most probable of `E` experts, best first; experts of an equal logit
//!   (`-0` and `+0` alike) go in ascending order.
//! - Probabilities and the weighted sums are FP32, a sum in choice order; a BF16 sum rounds once.
//! - Every op is generic over the chips its tensors lie on, each chip routing its own rows.

use crate::Module;
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::ops::collective::AllReduce;
use crate::ops::linear::experts::{Indices, Rows};
use crate::ops::normalization::{Chunk, FeatureSlices};
use crate::ops::reduction;
use crate::tensor::mapping::{BroadcastClusters as Cluster, Rows as Layout};
use crate::tensor::parallel::{Chip, Parallel, Parallelism};
use crate::tensor::scalar;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Each row's FP32 probabilities, paired with its selected experts, each in the first word of an
/// eight-byte pair.
pub type Probabilities<R, K, C = Chip> = HbmTensor<f32, C, m![R, K, 1 # 2]>;

/// One row a slice, a row's experts in its elements.
type PerExpert<D, R, E, C> = DmTensor<D, C, Cluster, m![R # 256], m![E]>;

/// One value a row.
type PerRow<D, R, C> = DmTensor<D, C, Cluster, m![R # 256], m![1 # 8]>;

/// One value a row and selection, each in an eight-byte pair.
type PerSelection<D, R, K, C> = DmTensor<D, C, Cluster, m![R # 256], m![K, 1 # 2]>;

/// Masks a selected key below every other; key differences stay within 2^25.
const MASKED: i32 = -(1 << 30);

/// The logit's tag of a negative value.
const NEGATIVE: TagGuard =
    TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);

/// Softmax of each row's BF16 logits over `Experts` experts in FP32, and its `Selected` largest
/// probabilities without renormalizing.
///
/// A key packs a logit's order-preserving bits above the expert's rank `Experts - 1 - expert`,
/// so one integer maximum finds the largest logit and the lower of equal ones.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct TopK<Experts, Selected, C = Chip, Ranks = HbmTensor<i32, C, m![Experts]>> {
    /// `Experts - 1 - expert` for every expert, on every chip.
    pub ranks: Ranks,
    experts: PhantomData<fn() -> (Experts, Selected, C)>,
}

impl<Experts, Selected, C, Ranks> TopK<Experts, Selected, C, Ranks> {
    pub const fn new(ranks: Ranks) -> Self {
        Self {
            ranks,
            experts: PhantomData,
        }
    }
}

/// The ranks computed on host. By hand, with the update below: a derived `with` field would
/// rebuild them in every update.
impl<'a, Experts: AxisName, Selected, C: M, Ranks> Load<&'a Checkpoint>
    for TopK<Experts, Selected, C, Ranks>
where
    Ranks: Load<HostTensor<i32, m![{ C }, { m![Experts] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("ranks", "ranks", |cx| {
                reduction::ranks::<Experts, C, _>(from, cx)
            })
            .map(Self::new)
        })
    }
}

/// The ranks are no update target.
impl<'t, 's, E, K, C, Ranks> Load<(&'t mut TopK<E, K, C, Ranks>, &'s Checkpoint)>
    for &'t mut TopK<E, K, C, Ranks>
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut TopK<E, K, C, Ranks>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |_| Ok(()))?;
        Ok(into)
    }
}

impl<Experts: AxisName, Selected: AxisName, R: AxisName, C: M, Env>
    Module<&HbmTensor<bf16, C, m![R, Experts]>, Env, &mut ()> for TopK<Experts, Selected, C>
{
    type Output = (Indices<R, Selected, C>, Probabilities<R, Selected, C>);

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &mut (),
        logits: &HbmTensor<bf16, C, m![R, Experts]>,
    ) -> Self::Output {
        use BitReq::{Ignore, One, Zero};
        const {
            assert!(R::SIZE <= 256 && Experts::SIZE <= 512 && Experts::SIZE.is_multiple_of(8));
            assert!(0 < Selected::SIZE && Selected::SIZE <= Experts::SIZE);
        };
        let logits: PerExpert<bf16, R, Experts, C> = logits.to_dm(&mut device.tdma);
        let ranks: DmTensor<i32, C, Cluster, m![256], m![Experts]> =
            self.ranks.to_dm(&mut device.tdma);
        let ranks: VrfTensor<i32, C, Cluster, m![256], m![Experts]> = device
            .sub
            .begin(ranks.view())
            .fetch::<m![Experts / 8], m![Experts % 8]>()
            .collect::<m![Experts / 8], m![Experts % 8]>()
            .to_vrf();
        // SAFETY: every slice holds all ranks, so each row's slice holds its own.
        let ranks: VrfTensor<i32, C, Cluster, m![R # 256], m![Experts]> =
            unsafe { ranks.reshape() };
        // Flipping a negative logit's magnitude bits makes the signed key follow the float.
        let zero = TagGuard::matches([Ignore, One, Ignore, Ignore]);
        let keys: PerExpert<i32, R, Experts, C> = device
            .main
            .begin(logits.view())
            .fetch::<m![Experts / 8], m![Experts % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Experts / 8], m![Experts % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Less(0.0),
                Cmp::Equal(0.0),
                Cmp::False,
                Cmp::False,
            ]))
            .vector_reinterpret::<i32>()
            .vector_logic(
                LogicBinaryOpI32::BitXor,
                Branched::imm(NEGATIVE, 0x7fff_0000),
            )
            .vector_logic(LogicBinaryOpI32::BitAnd, Branched::imm(zero, 0))
            .vector_fxp(FxpBinaryOp::ArithRightShift, 7)
            .vector_clip(ClipBinaryOpI32::AddFxp, &ranks)
            .vector_final()
            .commit_trim::<m![Experts % 8]>()
            .commit();
        let mut chosen: PerSelection<i32, R, Selected, C> = DmTensor::new();
        device
            .main
            .begin(keys.view())
            .fetch::<m![Experts / 8], m![Experts % 8]>()
            .collect::<m![Experts / 8], m![Experts % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Experts / 4], m![Experts % 4]>()
            .vector_intra_slice_reduce::<Experts, m![1], m![1 # 4]>(IntraSliceReduceOpI32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit_view(
                chosen
                    .view_mut()
                    .tile::<m![Selected], 1, m![Selected = 1 #{!} (Selected), 1 # 2]>(0),
            );
        // Each selection is the largest key below the one before: keys at or above it are masked.
        let selected = TagGuard::matches([Zero, Ignore, Ignore, Ignore]);
        #[unroll]
        for at in 0..Selected::SIZE - 1 {
            let previous: VrfTensor<i32, C, Cluster, m![R # 256], m![1 # 8]> = device
                .sub
                .begin(
                    chosen
                        .view()
                        .tile::<m![Selected], 1, m![Selected = 1 # (Selected), 1 # 2]>(at),
                )
                .fetch::<m![1], m![1 # 2]>()
                .collect::<m![1], m![1 # 8]>()
                .to_vrf();
            let distance: PerExpert<i32, R, Experts, C> = device
                .main
                .begin(keys.view())
                .fetch::<m![Experts / 8], m![Experts % 8]>()
                .collect::<m![Experts / 8], m![Experts % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp(FxpBinaryOp::SubFxp, &previous)
                .vector_final()
                .commit_trim::<m![Experts % 8]>()
                .commit();
            device
                .main
                .begin(distance.view())
                .fetch::<m![Experts / 8], m![Experts % 8]>()
                .collect::<m![Experts / 8], m![Experts % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Comparison([
                    Cmp::Less(0),
                    Cmp::False,
                    Cmp::False,
                    Cmp::False,
                ]))
                .vector_logic(LogicBinaryOpI32::BitOr, Branched::imm(selected, MASKED))
                .vector_fxp(FxpBinaryOp::AddFxp, &previous)
                .vector_narrow_split::<m![Experts / 4], m![Experts % 4]>()
                .vector_intra_slice_reduce::<Experts, m![1], m![1 # 4]>(IntraSliceReduceOpI32::Max)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 2]>()
                .commit_view(
                    chosen
                        .view_mut()
                        .tile::<m![Selected], 1, m![Selected = 1 #{!} (Selected), 1 # 2]>(at + 1),
                );
        }
        let maximum: PerRow<f32, R, C> = device
            .main
            .begin(logits.view())
            .fetch::<m![Experts / 8], m![Experts % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Experts / 8], m![Experts % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Experts / 4], m![Experts % 4]>()
            .vector_intra_slice_reduce::<Experts, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let maximum = scalar::register(device, &maximum);
        let total: PerRow<f32, R, C> = device
            .main
            .begin(logits.view())
            .fetch::<m![Experts / 8], m![Experts % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Experts / 8], m![Experts % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Experts / 4], m![Experts % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &maximum)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_intra_slice_reduce::<Experts, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let total = scalar::register(device, &total);
        let indices: PerSelection<i32, R, Selected, C> = device
            .main
            .begin(chosen.view())
            .fetch::<m![Selected], m![1 # 2]>()
            .collect::<m![Selected], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpI32::BitAnd, 511)
            .vector_fxp_with_mode(
                FxpBinaryOp::SubFxp,
                BinaryArgMode::Mode10,
                const { Experts::SIZE as i32 - 1 },
            )
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit();
        // A key without its rank, shifted back and unflipped, is the logit's FP32 bits.
        let probabilities: PerSelection<f32, R, Selected, C> = device
            .main
            .begin(chosen.view())
            .fetch::<m![Selected], m![1 # 2]>()
            .collect::<m![Selected], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Less(0),
                Cmp::False,
                Cmp::False,
                Cmp::False,
            ]))
            .vector_logic(LogicBinaryOpI32::BitAnd, -512)
            .vector_logic(
                LogicBinaryOpI32::BitXor,
                Branched::imm(NEGATIVE, 0x00ff_fe00),
            )
            .vector_fxp(FxpBinaryOp::LeftShift, 7)
            .vector_reinterpret::<f32>()
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &maximum)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_fp_div(&total)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit();
        (
            indices.to_hbm(&mut device.tdma),
            probabilities.to_hbm(&mut device.tdma),
        )
    }
}

/// Divide each row's selected probabilities by their FP32 sum.
#[derive(Default, DeviceSend, Load)]
pub struct Normalize;

impl<R: AxisName, K: AxisName, C: M, Env> Module<Probabilities<R, K, C>, Env, &mut ()>
    for Normalize
{
    type Output = Probabilities<R, K, C>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &mut (),
        probabilities: Probabilities<R, K, C>,
    ) -> Self::Output {
        const { assert!(R::SIZE <= 256) };
        let probabilities: PerSelection<f32, R, K, C> = probabilities.to_dm(&mut device.tdma);
        let sum: PerRow<f32, R, C> = device
            .main
            .begin(probabilities.view())
            .fetch::<m![K], m![1 # 2]>()
            .collect::<m![K], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_intra_slice_reduce::<K, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let sum = scalar::register(device, &sum);
        let normalized: PerSelection<f32, R, K, C> = device
            .main
            .begin(probabilities.view())
            .fetch::<m![K], m![1 # 2]>()
            .collect::<m![K], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_div(&sum)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit();
        normalized.to_hbm(&mut device.tdma)
    }
}

/// Sum each row's selected expert rows weighted by their FP32 probabilities, in FP32 in choice
/// order, rounded once to BF16. The selections spread over the chips of `P`, each chip's rows -0 for
/// the others' experts: the chips' rows add first, exactly, so every chip folds one chip's rows.
#[derive(DeviceSend, Load)]
pub struct Reduce<P = Parallel<1>>(PhantomData<P>);

impl<P> Default for Reduce<P> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Every row's selections, row after row.
#[derive(Clone, Debug)]
pub struct Selections<R, K>(PhantomData<(R, K)>);

impl<R: AxisName, K: AxisName> AxisName for Selections<R, K> {
    const NAME: Ident = Ident::new("Selections");
    const SIZE: usize = R::SIZE * K::SIZE;
}

impl<R: AxisName, K: AxisName, W: AxisName, P: Parallelism, Env: Copy>
    Module<(&Rows<R, K, W, P::Chip>, &Probabilities<R, K, P::Chip>), Env, &mut ()> for Reduce<P>
where
    AllReduce<P>: for<'s> Module<
            HbmTensor<bf16, P::Chip, m![{ Symbol<Selections<R, K>> }, W]>,
            Env,
            &'s mut (),
            Output = HbmTensor<bf16, P::Chip, m![{ Symbol<Selections<R, K>> }, W]>,
        >,
{
    type Output = HbmTensor<bf16, P::Chip, m![R, W]>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (rows, probabilities): (&Rows<R, K, W, P::Chip>, &Probabilities<R, K, P::Chip>),
    ) -> Self::Output {
        type C<P> = <P as Parallelism>::Chip;
        type Flat<R, K> = Symbol<Selections<R, K>>;
        if const { <P::Chip as M>::SIZE > 1 } {
            // A selection's row is nonzero on one chip, so the chips' rows add exactly.
            // SAFETY: the rows' selections in order; every cell keeps its address.
            let flat: HbmTensor<bf16, C<P>, m![{ Flat<R, K> }, W]> = unsafe { rows.view().alias() };
            let flat = AllReduce::<P>::default().forward(&mut *device, env, &mut (), flat);
            // SAFETY: as above, back to the rows' selections.
            let rows: Rows<R, K, W, C<P>> = unsafe { flat.view().alias() };
            return fold(device, &rows, probabilities);
        }
        fold(device, rows, probabilities)
    }
}

/// Each row's selected rows weighted by their probabilities, folded in FP32 in choice order and
/// rounded once, on every chip of `C` its own rows.
fn fold<R: AxisName, K: AxisName, W: AxisName, C: M>(
    device: &mut Device,
    rows: &Rows<R, K, W, C>,
    probabilities: &Probabilities<R, K, C>,
) -> HbmTensor<bf16, C, m![R, W]> {
    type Cs<W> = Symbol<Chunk<W>>;
    type Lanes<R> = m![{ Layout<R>::Lanes }];
    let rows: DmTensor<bf16, C, Cluster, FeatureSlices<W>, m![R, K, W % ({ Cs<W> })]> =
        rows.to_dm(&mut device.tdma);
    let probabilities: DmTensor<f32, C, Cluster, m![256], m![R, K, 1 # 2]> =
        probabilities.to_dm(&mut device.tdma);
    // SAFETY: every slice holds every probability; only the slice label changes.
    let probabilities: DmTensor<f32, C, Cluster, FeatureSlices<W>, m![R, K, 1 # 2]> =
        unsafe { probabilities.reshape() };
    let mut sums: DmTensor<f32, C, Cluster, FeatureSlices<W>, m![R, W % ({ Cs<W> })]> =
        DmTensor::new();
    // A group's probabilities take a 32-byte VRF packet each, so each group loads its own.
    for group in 0..<Layout<R>::Groups as M>::SIZE {
        let weight: VrfTensor<f32, C, Cluster, FeatureSlices<W>, m![{ Lanes<R> }, K, 1 # 8]> =
            device
                .sub
                .begin(
                    probabilities
                        .view()
                        .tile::<Layout<R>::Groups, 1, Layout<R>::Elements<m![K, 1 # 2]>>(group),
                )
                .fetch::<m![{ Lanes<R> }, K], m![1 # 2]>()
                .collect::<m![{ Lanes<R> }, K], m![1 # 8]>()
                .to_vrf();
        device
            .main
            .begin(
                rows.view()
                    .tile::<Layout<R>::Groups, 1, Layout<R>::Elements<m![K, W % ({ Cs<W> })]>>(
                        group,
                    ),
            )
            .fetch::<m![{ Lanes<R> }, K, W % ({ Cs<W> }) / 8], m![W % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Lanes<R> }, K, W % ({ Cs<W> }) / 8], m![W % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Lanes<R> }, K, W % ({ Cs<W> }) / 4], m![W % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight)
            .vector_intra_slice_reduce::<K, m![{ Lanes<R> }, W % ({ Cs<W> }) / 4], m![W % 4]>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_widen_concat::<m![{ Lanes<R> }, W % ({ Cs<W> }) / 8], m![W % 8]>()
            .vector_final()
            .commit_trim::<m![W % 8]>()
            .commit_view(
                sums.view_mut()
                    .tile::<Layout<R>::Groups, 1, Layout<R>::GroupMut<m![W % ({ Cs<W> })]>>(group),
            );
    }
    let rounded: DmTensor<bf16, C, Cluster, FeatureSlices<W>, m![R, W % ({ Cs<W> })]> = device
        .main
        .begin(sums.view())
        .fetch::<m![R, W % ({ Cs<W> }) / 8], m![W % 8]>()
        .collect::<m![R, W % ({ Cs<W> }) / 8], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .cast::<bf16, m![W % 8 # 16]>()
        .commit_trim::<m![W % 8]>()
        .commit();
    rounded.to_hbm(&mut device.tdma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::normalization::tests::random;
    use crate::tensor::axis::SixtyFour;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    use crate::modules::feedforward::ExpertMlp;
    use crate::ops::linear::experts::Experts as Bank;
    use crate::ops::pointwise::{activation::Silu, multiply::Multiply};

    axes![Experts = 128, Selected = 8, Hidden = 2048];

    type Rows = SixtyFour;

    axes![Model = 1024, Wide = 512, Total = 8, Picked = 4, Eight = 8];

    /// Experts no row selects: on two and four chips, the stand-ins a chip projects a selection of
    /// another chip's expert through. Their gate matrices overflow every row's FP32 product.
    const UNSELECTED: [usize; 2] = [3, 4];

    /// 2^127, the unselected gate matrices' first column: times a row's first feature, 4, it
    /// overflows FP32.
    const HUGE: f32 = 1.7014118e38;

    type Mlp<const L: usize, C> = ExpertMlp<
        Bank<Model, Wide, L, C>,
        Silu,
        Bank<Model, Wide, L, C>,
        Multiply,
        Bank<Wide, Model, L, C>,
    >;

    type Chips<const N: usize> = <Parallel<N> as Parallelism>::Chip;

    /// The block's host values: every expert's gate, up and down matrices, the logits and the rows.
    struct Values {
        gate: Vec<Vec<bf16>>,
        up: Vec<Vec<bf16>>,
        down: Vec<Vec<bf16>>,
        logits: Vec<bf16>,
        input: Vec<bf16>,
    }

    impl Values {
        fn new() -> Self {
            let scaled = |count, seed, scale: f32| -> Vec<bf16> {
                (random(count, seed).into_iter())
                    .map(|v| bf16::from_f32(v.to_f32() * scale))
                    .collect()
            };
            let experts = |seed: u64, scale| -> Vec<Vec<bf16>> {
                (0..Total::SIZE as u64)
                    .map(|e| scaled(Wide::SIZE * Model::SIZE, seed + e, scale))
                    .collect()
            };
            let mut gate = experts(0x100, 0.06);
            let mut logits = scaled(Eight::SIZE * Total::SIZE, 0x400, 4.0);
            for expert in UNSELECTED {
                (gate[expert].iter_mut().step_by(Model::SIZE))
                    .for_each(|v| *v = bf16::from_f32(HUGE));
                (logits.iter_mut().skip(expert).step_by(Total::SIZE))
                    .for_each(|v| *v = bf16::from_f32(-100.0));
            }
            let mut input = scaled(Eight::SIZE * Model::SIZE, 0x500, 1.0);
            (input.iter_mut().step_by(Model::SIZE)).for_each(|v| *v = bf16::from_f32(4.0));
            Self {
                gate,
                up: experts(0x200, 0.06),
                down: experts(0x300, 0.08),
                logits,
                input,
            }
        }
    }

    #[device(chip = 1)]
    fn select(
        device: &mut Device,
        router: &TopK<Experts, Selected>,
        logits: &HbmTensor<bf16, Chip, m![Rows, Experts]>,
    ) -> (Indices<Rows, Selected>, Probabilities<Rows, Selected>) {
        router.forward(device, (), &mut (), logits)
    }

    #[device(chip = 1)]
    fn normalize(
        device: &mut Device,
        probabilities: Probabilities<Rows, Selected>,
    ) -> Probabilities<Rows, Selected> {
        Normalize.forward(device, (), &mut (), probabilities)
    }

    #[device(chip = 1)]
    fn reduce(
        device: &mut Device,
        rows: &super::Rows<Rows, Selected, Hidden>,
        probabilities: &Probabilities<Rows, Selected>,
    ) -> HbmTensor<bf16, Chip, m![Rows, Hidden]> {
        Reduce::<Parallel<1>>::default().forward(device, (), &mut (), (rows, probabilities))
    }

    /// Row `row`'s logits: random rows, and rows whose largest logits tie, of either sign and
    /// zero of either sign, so that the lower expert must win each tie.
    fn logits(row: usize) -> Vec<bf16> {
        let values = random(Experts::SIZE, 0x2545_f491 + row as u64);
        let scaled = |value: bf16| bf16::from_f32(value.to_f32() * 8.0);
        let mut logits: Vec<bf16> = values.into_iter().map(scaled).collect();
        match row % 4 {
            1 => logits
                .iter_mut()
                .step_by(5)
                .for_each(|v| *v = bf16::from_f32(9.0)),
            2 => {
                logits
                    .iter_mut()
                    .for_each(|v| *v = bf16::from_f32(-v.to_f32().abs() - 1.0));
                (0..Experts::SIZE).step_by(7).for_each(|at| {
                    logits[at] = bf16::from_f32(if at % 2 == 0 { 0.0 } else { -0.0 });
                });
            }
            3 => logits.iter_mut().for_each(|v| *v = bf16::from_f32(-3.0)),
            _ => {}
        }
        logits
    }

    /// The `Selected` experts of `logits` by Transformers' rule (softmax in FP32, then top-k)
    /// with a tie to the lower expert, and their probabilities in f64.
    fn reference(logits: &[bf16]) -> Vec<(i32, f64)> {
        let values: Vec<f64> = logits.iter().map(|v| f64::from(v.to_f32())).collect();
        let top = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = values.iter().map(|v| (v - top).exp()).sum();
        let mut order: Vec<usize> = (0..values.len()).collect();
        // `partial_cmp` holds -0 and +0 equal, as the device's keys do.
        order.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap().then(a.cmp(&b)));
        order[..Selected::SIZE]
            .iter()
            .map(|&expert| (expert as i32, (values[expert] - top).exp() / total))
            .collect()
    }

    /// Each row selects its eight most probable of 128 experts, best first and the lower of
    /// equal logits first, with their FP32 softmax probabilities.
    #[tokio::test(flavor = "current_thread")]
    async fn selects_the_most_probable_experts() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(select.topology()).unwrap();
        let ranks: Vec<i32> = (0..Experts::SIZE as i32).rev().collect();
        let router = TopK::<Experts, Selected>::new(
            HostTensor::<i32, m![Experts]>::from_vec(ranks)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        );
        let logits: Vec<Vec<bf16>> = (0..Rows::SIZE).map(logits).collect();
        let input = HostTensor::<bf16, m![Rows, Experts]>::from_vec(logits.concat())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let (indices, probabilities) = launch(select, (&mut device, &router, &input))
            .await
            .unwrap();
        let indices = indices
            .to_host::<m![Rows, Selected]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let probabilities = probabilities
            .to_host::<m![Rows, Selected]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (row, logits) in logits.iter().enumerate() {
            for (at, (expert, probability)) in reference(logits).into_iter().enumerate() {
                let got = row * Selected::SIZE + at;
                assert_eq!(indices[got], expert, "row {row} selection {at}");
                let error = (f64::from(probabilities[got]) - probability).abs();
                assert!(
                    error <= 4e-7 * probability.max(1e-30),
                    "row {row} selection {at}: {} vs {probability}",
                    probabilities[got]
                );
            }
        }
    }

    /// Each row's selected probabilities divide by their FP32 sum.
    #[tokio::test(flavor = "current_thread")]
    async fn normalizes_each_row_to_one() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(normalize.topology()).unwrap();
        let values: Vec<f32> = random(Rows::SIZE * Selected::SIZE, 0x51ed)
            .into_iter()
            .map(|v| (v.to_f32() + 1.0) / 4.0 + 1e-3)
            .collect();
        let input = HostTensor::<f32, m![Rows, Selected]>::from_vec(values.clone())
            .to_hbm::<Chip, m![Rows, Selected, 1 # 2]>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(normalize, (&mut device, input))
            .await
            .unwrap()
            .to_host::<m![Rows, Selected]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (row, (got, values)) in output
            .chunks_exact(Selected::SIZE)
            .zip(values.chunks_exact(Selected::SIZE))
            .enumerate()
        {
            let sum = values.iter().fold(0.0_f32, |sum, v| sum + v);
            for (at, (got, value)) in got.iter().zip(values).enumerate() {
                let expected = value / sum;
                assert!(
                    (got - expected).abs() <= expected * f32::EPSILON,
                    "row {row} selection {at}: {got} vs {expected}"
                );
            }
        }
    }

    /// Each row's sum is the FP32 fold of `probability * row` in choice order rounded once to BF16,
    /// selections of zero probability included.
    #[tokio::test(flavor = "current_thread")]
    async fn sums_the_weighted_selections_in_fp32() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(reduce.topology()).unwrap();
        let rows = random(Rows::SIZE * Selected::SIZE * Hidden::SIZE, 0x9e37);
        let weights: Vec<f32> = (random(Rows::SIZE * Selected::SIZE, 0x7f4a).into_iter())
            .enumerate()
            .map(|(at, v)| {
                if at % 11 == 3 {
                    0.0
                } else {
                    v.to_f32().abs() + 0.25
                }
            })
            .collect();
        let expected: Vec<bf16> = (0..Rows::SIZE * Hidden::SIZE)
            .map(|at| {
                let (row, feature) = (at / Hidden::SIZE, at % Hidden::SIZE);
                let term = |k: usize| {
                    let selection = row * Selected::SIZE + k;
                    weights[selection] * rows[selection * Hidden::SIZE + feature].to_f32()
                };
                bf16::from_f32((0..Selected::SIZE).fold(0.0, |sum, k| sum + term(k)))
            })
            .collect();
        let input = HostTensor::<bf16, m![Rows, Selected, Hidden]>::from_vec(rows)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let weights = HostTensor::<f32, m![Rows, Selected]>::from_vec(weights)
            .to_hbm::<Chip, m![Rows, Selected, 1 # 2]>(&mut device.pdma)
            .await
            .unwrap();
        let sums = launch(reduce, (&mut device, &input, &weights))
            .await
            .unwrap()
            .to_host::<m![Rows, Hidden]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let differ = (sums.iter().zip(&expected))
            .filter(|(got, want)| got.to_bits() != want.to_bits())
            .count();
        assert_eq!(differ, 0, "{differ} of {} sums differ", sums.len());
    }

    // # Expert parallelism

    /// A MoE block of `L` experts a chip on the chips of `P`: route, project, weight and sum.
    fn block<const L: usize, P: Parallelism>(
        device: &mut Device,
        router: &TopK<Total, Picked, P::Chip>,
        mlp: &Mlp<L, P::Chip>,
        logits: &HbmTensor<bf16, P::Chip, m![Eight, Total]>,
        input: &HbmTensor<bf16, P::Chip, m![Eight, Model]>,
    ) -> HbmTensor<bf16, P::Chip, m![Eight, Model]>
    where
        AllReduce<P>: for<'s> Module<
                HbmTensor<bf16, P::Chip, m![{ Symbol<Selections<Eight, Picked>> }, Model]>,
                (),
                &'s mut (),
                Output = HbmTensor<bf16, P::Chip, m![{ Symbol<Selections<Eight, Picked>> }, Model]>,
            >,
    {
        let (indices, probabilities) = router.forward(&mut *device, (), &mut (), logits);
        let probabilities = Normalize.forward(&mut *device, (), &mut (), probabilities);
        let rows = mlp.forward(&mut *device, (), &mut (), (input, &indices));
        Reduce::<P>::default().forward(device, (), &mut (), (&rows, &probabilities))
    }

    #[device(chip = 1)]
    fn one(
        device: &mut Device,
        router: &TopK<Total, Picked>,
        mlp: &Mlp<8, Chip>,
        logits: &HbmTensor<bf16, Chip, m![Eight, Total]>,
        input: &HbmTensor<bf16, Chip, m![Eight, Model]>,
    ) -> HbmTensor<bf16, Chip, m![Eight, Model]> {
        block::<8, Parallel<1>>(device, router, mlp, logits, input)
    }

    #[device(chip = 2)]
    fn two(
        device: &mut Device,
        router: &TopK<Total, Picked, Chips<2>>,
        mlp: &Mlp<4, Chips<2>>,
        logits: &HbmTensor<bf16, Chips<2>, m![Eight, Total]>,
        input: &HbmTensor<bf16, Chips<2>, m![Eight, Model]>,
    ) -> HbmTensor<bf16, Chips<2>, m![Eight, Model]> {
        block::<4, Parallel<2>>(device, router, mlp, logits, input)
    }

    #[device(chip = 4)]
    fn four(
        device: &mut Device,
        router: &TopK<Total, Picked, Chips<4>>,
        mlp: &Mlp<2, Chips<4>>,
        logits: &HbmTensor<bf16, Chips<4>, m![Eight, Total]>,
        input: &HbmTensor<bf16, Chips<4>, m![Eight, Model]>,
    ) -> HbmTensor<bf16, Chips<4>, m![Eight, Model]> {
        block::<2, Parallel<4>>(device, router, mlp, logits, input)
    }

    /// `values` on every chip of `C`, a copy each.
    async fn replicated<D: MaterializableScalar, C: M, E: M>(
        device: &mut Device,
        values: &[D],
    ) -> HbmTensor<D, C, E> {
        HostTensor::<D, m![{ C }, { E }]>::from_vec(values.repeat(C::SIZE))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap()
    }

    /// `L` of `matrices` on each chip of `C`: chip `c` holds experts `c * L..(c + 1) * L`.
    async fn bank<const L: usize, C: M, O: AxisName, I: AxisName>(
        device: &mut Device,
        matrices: &[Vec<bf16>],
    ) -> Bank<I, O, L, C> {
        let mut weights = Vec::new();
        for local in 0..L {
            let shares: Vec<bf16> = (0..C::SIZE)
                .flat_map(|chip| matrices[chip * L + local].iter().copied())
                .collect();
            let share = HostTensor::<bf16, m![{ C }, O, I]>::from_vec(shares);
            weights.push(share.to_hbm(&mut device.pdma).await.unwrap());
        }
        let firsts: Vec<i32> = (0..C::SIZE).map(|chip| (chip * L) as i32).collect();
        Bank {
            weights: weights.try_into().unwrap(),
            first: HostTensor::<i32, m![{ C }, 1]>::from_vec(firsts)
                .to_hbm::<C, m![1 # 8]>(&mut device.pdma)
                .await
                .unwrap(),
        }
    }

    /// The block's router, its experts and its inputs on the chips of `C`, `L` experts a chip.
    async fn operands<const L: usize, C: M>(
        device: &mut Device,
        values: &Values,
    ) -> (
        TopK<Total, Picked, C>,
        Mlp<L, C>,
        HbmTensor<bf16, C, m![Eight, Total]>,
        HbmTensor<bf16, C, m![Eight, Model]>,
    ) {
        let ranks: Vec<i32> = (0..Total::SIZE as i32).rev().collect();
        let router = TopK::new(replicated::<i32, C, m![Total]>(device, &ranks).await);
        let mlp = ExpertMlp {
            gate: bank::<L, C, Wide, Model>(device, &values.gate).await,
            activation: Silu,
            up: bank::<L, C, Wide, Model>(device, &values.up).await,
            multiply: Multiply,
            down: bank::<L, C, Model, Wide>(device, &values.down).await,
        };
        let logits = replicated::<bf16, C, m![Eight, Total]>(device, &values.logits).await;
        let input = replicated::<bf16, C, m![Eight, Model]>(device, &values.input).await;
        (router, mlp, logits, input)
    }

    /// A block whose experts split whole over two or four chips gives, on every chip, the BF16 rows
    /// of the block of every expert on one chip, bit for bit: the chips' rows add exactly, then
    /// every chip folds them as one chip does. On four chips a chip holds fewer experts than a row
    /// selects, and on both a stand-in's projection overflows.
    #[tokio::test(flavor = "current_thread")]
    async fn splits_the_experts_over_chips_as_one_chip_sums_them() {
        let _lease = DEVICE.lock().await;
        let values = Values::new();
        let unsplit = {
            let mut device = Device::new(one.topology()).unwrap();
            let (router, mlp, logits, input) = operands::<8, Chip>(&mut device, &values).await;
            let rows = launch(one, (&mut device, &router, &mlp, &logits, &input)).await;
            (rows
                .unwrap()
                .to_host::<m![Eight, Model]>(&mut device.pdma)
                .await)
                .unwrap()
                .into_vec()
        };
        let bits = |values: &[bf16]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert!((unsplit.iter()).all(|v| v.to_f32().is_finite() && v.to_f32() != 0.0));
        let split = {
            let mut device = Device::new(two.topology()).unwrap();
            let (router, mlp, logits, input) = operands::<4, Chips<2>>(&mut device, &values).await;
            let rows = launch(two, (&mut device, &router, &mlp, &logits, &input)).await;
            (rows
                .unwrap()
                .to_host::<m![{ Chips<2> }, Eight, Model]>(&mut device.pdma)
                .await)
                .unwrap()
                .into_vec()
        };
        assert_eq!(bits(&split), bits(&unsplit.repeat(2)), "two chips");
        let split = {
            let mut device = Device::new(four.topology()).unwrap();
            let (router, mlp, logits, input) = operands::<2, Chips<4>>(&mut device, &values).await;
            let rows = launch(four, (&mut device, &router, &mlp, &logits, &input)).await;
            (rows
                .unwrap()
                .to_host::<m![{ Chips<4> }, Eight, Model]>(&mut device.pdma)
                .await)
                .unwrap()
                .into_vec()
        };
        assert_eq!(bits(&split), bits(&unsplit.repeat(4)), "four chips");
    }

    #[device(chip = 2)]
    fn reduce_two(
        device: &mut Device,
        rows: &super::Rows<Eight, Picked, Model, Chips<2>>,
        probabilities: &Probabilities<Eight, Picked, Chips<2>>,
    ) -> HbmTensor<bf16, Chips<2>, m![Eight, Model]> {
        Reduce::<Parallel<2>>::default().forward(device, (), &mut (), (rows, probabilities))
    }

    /// A split set's weighted rows fold on every chip in choice order, as one chip folds them, where
    /// the order decides the sum: selections alternate between two chips, and of the terms
    /// `[2^24 s, s, -2^24 s, s]` the choice-order fold is `s`, the sum of the chips' folds `2 s`.
    #[tokio::test(flavor = "current_thread")]
    async fn folds_split_selections_in_choice_order() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(reduce_two.topology()).unwrap();
        let negative = bf16::from_f32(-0.0);
        // Rows 0..4 cancel at every feature, at its own scale and sign; the others are random, a
        // row of -0 at their first feature.
        let random_rows = random(Eight::SIZE * Picked::SIZE * Model::SIZE, 0x3c6e);
        let rows: Vec<bf16> = (0..Eight::SIZE * Picked::SIZE * Model::SIZE)
            .map(|at| {
                let (row, k, feature) = (
                    at / (Picked::SIZE * Model::SIZE),
                    at / Model::SIZE % Picked::SIZE,
                    at % Model::SIZE,
                );
                let scale =
                    2f32.powi(feature as i32 % 48 - 24) * if feature % 2 == 0 { 1.0 } else { -1.0 };
                match (row, k) {
                    (0..4, _) => {
                        let [big, small] = [2f32.powi(24) * scale, scale];
                        let terms = match row % 2 {
                            0 => [big, small, -big, small],
                            _ => [small, big, small, -big],
                        };
                        bf16::from_f32(terms[k])
                    }
                    _ if feature == 0 => negative,
                    _ => random_rows[at],
                }
            })
            .collect();
        let probabilities: Vec<f32> = (random(Eight::SIZE * Picked::SIZE, 0x5a17).into_iter())
            .enumerate()
            .map(|(at, v)| {
                if at < 4 * Picked::SIZE {
                    1.0
                } else {
                    v.to_f32().abs() + 0.125
                }
            })
            .collect();
        let expected: Vec<u16> = (0..Eight::SIZE * Model::SIZE)
            .map(|at| {
                let (row, feature) = (at / Model::SIZE, at % Model::SIZE);
                let term = |k: usize| {
                    let selection = row * Picked::SIZE + k;
                    probabilities[selection] * rows[selection * Model::SIZE + feature].to_f32()
                };
                bf16::from_f32((0..Picked::SIZE).fold(0.0, |sum, k| sum + term(k))).to_bits()
            })
            .collect();
        // Chip `c` holds selection `k` when `k % 2 == c`, as an expert split makes it: -0 elsewhere.
        let shares: Vec<bf16> = (0..2)
            .flat_map(|chip| {
                let rows = &rows;
                (0..rows.len()).map(
                    move |at| match at / Model::SIZE % Picked::SIZE % 2 == chip {
                        true => rows[at],
                        false => negative,
                    },
                )
            })
            .collect();
        let input = HostTensor::<bf16, m![{ Chips<2> }, Eight, Picked, Model]>::from_vec(shares)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let weights =
            HostTensor::<f32, m![{ Chips<2> }, Eight, Picked]>::from_vec(probabilities.repeat(2))
                .to_hbm::<Chips<2>, m![Eight, Picked, 1 # 2]>(&mut device.pdma)
                .await
                .unwrap();
        let sums = launch(reduce_two, (&mut device, &input, &weights))
            .await
            .unwrap()
            .to_host::<m![{ Chips<2> }, Eight, Model]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let bits: Vec<u16> = sums.iter().map(|v| v.to_bits()).collect();
        for (chip, bits) in bits.chunks_exact(expected.len()).enumerate() {
            let differ: Vec<_> = (bits.iter().zip(&expected).enumerate())
                .filter(|(_, (got, want))| got != want)
                .collect();
            let first = differ.first();
            assert!(
                differ.is_empty(),
                "chip {chip}: {} sums differ, first {first:?}",
                differ.len()
            );
        }
    }
}
