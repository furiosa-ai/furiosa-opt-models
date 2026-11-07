use furiosa_opt_std::prelude::*;

use crate::ops::layout::Rows as Layout;
use crate::ops::layout::axes::Chip;
use crate::ops::vocabulary::VocabPad;
use crate::ops::vocabulary::grouped::{Arms, Pick};
use crate::ops::vocabulary::head as lm_head;
use crate::ops::vocabulary::projection::Cluster;
use crate::{ops::linear::Groups, ops::vocabulary::grouped::Grouped};

type P = m![VocabPad % ({ lm_head::PassRows })];
type Index<Hidden> = m![VocabPad % ({ lm_head::Block<Hidden> }) / ({ lm_head::PassRows })];
type Lanes<Rows> = m![{ Layout<Rows>::Lanes } # 8];
type Half<Rows> = m![{ Layout<Rows>::Lanes } # 8 % 4];
type Times<Rows> = m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 8 / 4];
type Batch<Rows> = m![{ Groups<Rows> } % 8];
/// Takes every logit difference of at least 2^-100 in magnitude past -2^26, where the exponential
/// is 0: a smaller nonzero difference needs logits below 2^-76.
const SCALE: f32 = 1.0e38;
type Batches<Rows> = m![{ Groups<Rows> } / 8];
type Stored<D, Hidden, Rows> = DmTensor<
    D,
    Chip,
    Cluster,
    lm_head::Reduced<Hidden>,
    m![{ Index<Hidden> }, { Groups<Rows> }, { Lanes<Rows> }],
>;
type Local<D, Hidden, Rows> =
    DmTensor<D, Chip, Cluster, lm_head::Reduced<Hidden>, m![{ Groups<Rows> }, { Lanes<Rows> }]>;
type Padded<Rows> = m![{ Groups<Rows> } # ({ lm_head::PaddedGroups })];
type Tile<Hidden, Rows> = DmTensor<
    f32,
    Chip,
    Cluster,
    lm_head::Reduced<Hidden>,
    m![{ P }, { Padded<Rows> }, { Lanes<Rows> }],
>;
/// One cluster's vocabulary groups.
type Blocks<Hidden> = m![VocabPad % (VocabPad / 2) / ({ lm_head::Block<Hidden> })];
/// Every group's row summaries, in one slice of each cluster.
type Gathered<D, Hidden, Rows> = DmTensor<
    D,
    Chip,
    Cluster,
    m![1 # 256],
    m![{ Blocks<Hidden> }, { Groups<Rows> }, { Lanes<Rows> }],
>;
type Global<D, Rows> =
    DmTensor<D, Chip, Cluster, m![1 # 256], m![{ Groups<Rows> }, { Lanes<Rows> }]>;

pub(crate) struct Candidates<Hidden: lm_head::Project, Rows: Grouped> {
    maximum: Stored<f32, Hidden, Rows>,
    rank: Stored<i32, Hidden, Rows>,
}

impl<Hidden: lm_head::Project, Rows: Grouped> Candidates<Hidden, Rows> {
    pub(crate) fn new() -> Self {
        Self {
            maximum: DmTensor::new(),
            rank: DmTensor::new(),
        }
    }

    pub(crate) fn push(
        &mut self,
        context: &mut Device,
        logits: &Tile<Hidden, Rows>,
        iota: DmTensorView<
            '_,
            i32,
            Chip,
            Cluster,
            lm_head::Reduced<Hidden>,
            lm_head::Column<Hidden>,
        >,
        at: usize,
    ) {
        // Only two lane halves remain inside the reduction; the accumulator has eight slots.
        // Both summaries land in their pass's slot of the stored tables through the tensor unit's
        // own commit, so the DMA queue carries nothing but the weight stream.
        context.main.begin(logits.view())
            .fetch::<m![{ Groups<Rows> }, { P }], Lanes<Rows>>()
            .collect::<m![{ Groups<Rows> }, { P }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Groups<Rows> }, { P }, { Layout<Rows>::Lanes } # 8 / 4], Half<Rows>>()
            .vector_intra_slice_reduce::<VocabPad, Times<Rows>, Half<Rows>>(IntraSliceReduceOpF32::Max)
            .vector_widen_concat::<Groups<Rows>, Lanes<Rows>>()
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit_view(self.maximum.view_mut().tile::<Index<Hidden>, 1,
                m![1 #{!} ({ Index<Hidden> }), { Groups<Rows> }, { Lanes<Rows> }],
            >(at));
        let maximum: VrfTensor<
            f32, Chip, Cluster, lm_head::Reduced<Hidden>,
            m![{ Groups<Rows> }, { Lanes<Rows> }],
        > = context.sub.begin(self.maximum.view().tile::<Index<Hidden>, 1,
            m![1 # ({ Index<Hidden> }), { Groups<Rows> }, { Lanes<Rows> }],
        >(at))
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .to_vrf();
        // The token numbers as floats, the factor of a row's maxima below.
        let numbers: DmTensor<f32, Chip, Cluster, lm_head::Reduced<Hidden>, P> = context
            .main
            .begin(iota)
            .fetch::<m![{ P } / 8], m![VocabPad % 8]>()
            .collect::<m![{ P } / 8], m![VocabPad % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp_to_fp(31)
            .vector_final()
            .commit_trim::<m![VocabPad % 8]>()
            .commit();
        let numbers: VrfTensor<f32, Chip, Cluster, lm_head::Reduced<Hidden>, P> = context
            .sub
            .begin(numbers.view())
            .fetch::<m![{ P } / 8], m![VocabPad % 8]>()
            .collect::<m![{ P } / 8], m![VocabPad % 8]>()
            .to_vrf();
        // Only the maximum's difference is exactly zero, every other one a negative float that the
        // scale takes past -2^26 (or to -inf), so the exponential is 1 at the maxima and 0
        // elsewhere, and the largest of those times the token numbers is the first maximum's.
        context.main.begin(logits.view())
            .fetch::<m![{ Groups<Rows> }, { P }], Lanes<Rows>>()
            .collect::<m![{ Groups<Rows> }, { P }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Groups<Rows> }, { P }, { Layout<Rows>::Lanes } # 8 / 4], Half<Rows>>()
            .vector_fp_binary(FpBinaryOp::SubF, &maximum)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), SCALE)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &numbers)
            .vector_intra_slice_reduce::<VocabPad, Times<Rows>, Half<Rows>>(IntraSliceReduceOpF32::Max)
            .vector_widen_concat::<Groups<Rows>, Lanes<Rows>>()
            .vector_fp_to_fxp(31)
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit_view(self.rank.view_mut().tile::<Index<Hidden>, 1,
                m![1 #{!} ({ Index<Hidden> }), { Groups<Rows> }, { Lanes<Rows> }],
            >(at));
    }

    pub(crate) fn select(&self, context: &mut Device) -> lm_head::Selected<Rows> {
        // Every tensor-unit operation runs on one live slice per vocabulary group, so each group
        // first settles its own maximum and rank there; tensor DMA then gathers the groups'
        // summaries into one slice per cluster, where the same maximum-then-rank fold finishes.
        let local: Local<f32, Hidden, Rows> = context.main.begin(self.maximum.view())
            .fetch::<m![{ Groups<Rows> }, { Index<Hidden> }], Lanes<Rows>>()
            .collect::<m![{ Groups<Rows> }, { Index<Hidden> }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Groups<Rows> }, { Index<Hidden> }, { Layout<Rows>::Lanes } # 8 / 4], Half<Rows>>()
            .vector_intra_slice_reduce::<VocabPad, Times<Rows>, Half<Rows>>(IntraSliceReduceOpF32::Max)
            .vector_widen_concat::<Groups<Rows>, Lanes<Rows>>()
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit();
        let maximum: VrfTensor<
            f32,
            Chip,
            Cluster,
            lm_head::Reduced<Hidden>,
            m![{ Groups<Rows> }, { Lanes<Rows> }],
        > = context
            .sub
            .begin(local.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .to_vrf();
        let shifted: Stored<f32, Hidden, Rows> = context
            .main
            .begin(self.maximum.view())
            .fetch::<m![{ Index<Hidden> }, { Groups<Rows> }], Lanes<Rows>>()
            .collect::<m![{ Index<Hidden> }, { Groups<Rows> }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Index<Hidden> }, { Times<Rows> }], Half<Rows>>()
            .vector_fp_binary(FpBinaryOp::SubF, &maximum)
            .vector_widen_concat::<m![{ Index<Hidden> }, { Groups<Rows> }], Lanes<Rows>>()
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit();
        let mut rank: Local<i32, Hidden, Rows> = DmTensor::new();
        <Rows::Short as Pick>::pick(
            context,
            LocalRanks::<Hidden, Rows> {
                rank: &self.rank,
                shifted: &shifted,
                out: &mut rank,
            },
        );
        // The groups' maxima and ranks move to one slice of each cluster.
        let mut maxima: Gathered<f32, Hidden, Rows> = DmTensor::new();
        local
            .view()
            .to_dm_view(&mut context.tdma, maxima.view_mut());
        let mut ranks: Gathered<i32, Hidden, Rows> = DmTensor::new();
        rank.view().to_dm_view(&mut context.tdma, ranks.view_mut());
        let maximum = Self::fold_max(context, &maxima);
        let other: Global<f32, Rows> = maximum.view().cluster_swap().to_dm(&mut context.tdma);
        let other: VrfTensor<
            f32,
            Chip,
            Cluster,
            m![1 # 256],
            m![{ Groups<Rows> }, { Lanes<Rows> }],
        > = context
            .sub
            .begin(other.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .to_vrf();
        let maximum: Global<f32, Rows> = context
            .main
            .begin(maximum.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Max, &other)
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit();
        let maximum: VrfTensor<
            f32,
            Chip,
            Cluster,
            m![1 # 256],
            m![{ Groups<Rows> }, { Lanes<Rows> }],
        > = context
            .sub
            .begin(maximum.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .to_vrf();
        let shifted: Gathered<f32, Hidden, Rows> = context
            .main
            .begin(maxima.view())
            .fetch::<m![{ Blocks<Hidden> }, { Groups<Rows> }], Lanes<Rows>>()
            .collect::<m![{ Blocks<Hidden> }, { Groups<Rows> }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Blocks<Hidden> }, { Times<Rows> }], Half<Rows>>()
            .vector_fp_binary(FpBinaryOp::SubF, &maximum)
            .vector_widen_concat::<m![{ Blocks<Hidden> }, { Groups<Rows> }], Lanes<Rows>>()
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit();
        let mut rank: Global<i32, Rows> = DmTensor::new();
        <Rows::Short as Pick>::pick(
            context,
            GlobalRanks::<Hidden, Rows> {
                ranks: &ranks,
                shifted: &shifted,
                out: &mut rank,
            },
        );
        let other: Global<i32, Rows> = rank.view().cluster_swap().to_dm(&mut context.tdma);
        let other: VrfTensor<
            i32,
            Chip,
            Cluster,
            m![1 # 256],
            m![{ Groups<Rows> }, { Lanes<Rows> }],
        > = context
            .sub
            .begin(other.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .to_vrf();
        let rank: Global<i32, Rows> = context
            .main
            .begin(rank.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpI32::Max, &other)
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit();
        let token: Global<i32, Rows> = context
            .main
            .begin(rank.view())
            .fetch::<Groups<Rows>, Lanes<Rows>>()
            .collect::<Groups<Rows>, Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp_with_mode(
                FxpBinaryOp::SubFxp,
                BinaryArgMode::Mode10,
                VocabPad::SIZE as i32 - 1,
            )
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit();
        <Rows::One as Pick>::pick(context, Chosen::<Rows> { token })
    }

    /// One cluster's row maxima over its vocabulary groups.
    fn fold_max(context: &mut Device, gathered: &Gathered<f32, Hidden, Rows>) -> Global<f32, Rows> {
        context.main.begin(gathered.view())
            .fetch::<m![{ Groups<Rows> }, { Blocks<Hidden> }], Lanes<Rows>>()
            .collect::<m![{ Groups<Rows> }, { Blocks<Hidden> }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Groups<Rows> }, { Blocks<Hidden> }, { Layout<Rows>::Lanes } # 8 / 4], Half<Rows>>()
            .vector_intra_slice_reduce::<VocabPad, Times<Rows>, Half<Rows>>(IntraSliceReduceOpF32::Max)
            .vector_widen_concat::<Groups<Rows>, Lanes<Rows>>()
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit()
    }
}

/// Each row's token as the head returns it, for whichever path the row count takes: one row
/// moves into its padded slot, more rows already sit in the cells the head returns.
struct Chosen<Rows: Grouped> {
    token: Global<i32, Rows>,
}

impl<Rows: Grouped> Arms for Chosen<Rows> {
    type Output = lm_head::Selected<Rows>;

    fn then(self, context: &mut Device) -> Self::Output {
        // SAFETY: both clusters hold the same tokens after the symmetric swap.
        let token: DmTensor<
            i32,
            Chip,
            m![1 # 2],
            m![1 # 256],
            m![{ Groups<Rows> }, { Lanes<Rows> }],
        > = unsafe { self.token.reshape() };
        let mut output = lm_head::Selected::<Rows>::new();
        token
            .view()
            .to_dm_view(&mut context.tdma, output.view_mut());
        output
    }

    fn otherwise(self, _context: &mut Device) -> Self::Output {
        // SAFETY: both clusters hold the same tokens after the symmetric swap.
        unsafe { self.token.reshape() }
    }
}

/// What a vocabulary group's rank fold reads and writes, for whichever path the row count takes.
struct LocalRanks<'a, Hidden: lm_head::Project, Rows: Grouped> {
    rank: &'a Stored<i32, Hidden, Rows>,
    shifted: &'a Stored<f32, Hidden, Rows>,
    out: &'a mut Local<i32, Hidden, Rows>,
}

impl<Hidden: lm_head::Project, Rows: Grouped> Arms for LocalRanks<'_, Hidden, Rows> {
    type Output = ();

    fn then(self, context: &mut Device) {
        let (shifted, rank) = (self.shifted, self.out);
        let ranks: VrfTensor<
            i32,
            Chip,
            Cluster,
            lm_head::Reduced<Hidden>,
            m![{ Groups<Rows> }, { Index<Hidden> }, { Lanes<Rows> }],
        > = context
            .sub
            .begin(self.rank.view())
            .fetch::<m![{ Groups<Rows> }, { Index<Hidden> }], Lanes<Rows>>()
            .collect::<m![{ Groups<Rows> }, { Index<Hidden> }], Lanes<Rows>>()
            .to_vrf();
        context.main.begin(shifted.view())
            .fetch::<m![{ Groups<Rows> }, { Index<Hidden> }], Lanes<Rows>>()
            .collect::<m![{ Groups<Rows> }, { Index<Hidden> }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>()
            .vector_fxp(FxpBinaryOp::AddFxpSat, &ranks)
            .vector_narrow_split::<m![{ Groups<Rows> }, { Index<Hidden> }, { Layout<Rows>::Lanes } # 8 / 4], Half<Rows>>()
            .vector_intra_slice_reduce::<VocabPad, Times<Rows>, Half<Rows>>(IntraSliceReduceOpI32::Max)
            .vector_widen_concat::<Groups<Rows>, Lanes<Rows>>()
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit_view(rank.view_mut());
    }

    fn otherwise(self, context: &mut Device) {
        let (shifted, rank) = (self.shifted, self.out);
        #[unroll]
        for batch in 0..<Batches<Rows> as M>::SIZE {
            let ranks: VrfTensor<i32, Chip, Cluster, lm_head::Reduced<Hidden>,
                m![{ Batch<Rows> }, { Index<Hidden> }, { Lanes<Rows> }]> =
            context.sub.begin(self.rank.view().tile::<Batches<Rows>, 1,
                m![{ Index<Hidden> }, 1 # ({ Batches<Rows> }), { Batch<Rows> }, { Lanes<Rows> }],
            >(batch))
                .fetch::<m![{ Batch<Rows> }, { Index<Hidden> }], Lanes<Rows>>()
                .collect::<m![{ Batch<Rows> }, { Index<Hidden> }], Lanes<Rows>>()
                .to_vrf();
            context.main.begin(shifted.view().tile::<Batches<Rows>, 1,
                m![{ Index<Hidden> }, 1 # ({ Batches<Rows> }), { Batch<Rows> }, { Lanes<Rows> }],
            >(batch))
            .fetch::<m![{ Batch<Rows> }, { Index<Hidden> }], Lanes<Rows>>()
            .collect::<m![{ Batch<Rows> }, { Index<Hidden> }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>()
            .vector_fxp(FxpBinaryOp::AddFxpSat, &ranks)
            .vector_narrow_split::<m![{ Batch<Rows> }, { Index<Hidden> }, { Layout<Rows>::Lanes } / 4], Half<Rows>>()
            .vector_intra_slice_reduce::<VocabPad, m![{ Batch<Rows> }, { Layout<Rows>::Lanes } / 4], Half<Rows>>(IntraSliceReduceOpI32::Max)
            .vector_widen_concat::<Batch<Rows>, Lanes<Rows>>()
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit_view(rank.view_mut().tile::<Batches<Rows>, 1,
                m![1 #{!} ({ Batches<Rows> }), { Batch<Rows> }, { Lanes<Rows> }],
            >(batch));
        }
    }
}

/// What the cluster-wide rank fold reads and writes, for whichever path the row count takes.
struct GlobalRanks<'a, Hidden: lm_head::Project, Rows: Grouped> {
    ranks: &'a Gathered<i32, Hidden, Rows>,
    shifted: &'a Gathered<f32, Hidden, Rows>,
    out: &'a mut Global<i32, Rows>,
}

impl<Hidden: lm_head::Project, Rows: Grouped> Arms for GlobalRanks<'_, Hidden, Rows> {
    type Output = ();

    fn then(self, context: &mut Device) {
        let (ranks, shifted, rank) = (self.ranks, self.shifted, self.out);
        let candidates: VrfTensor<
            i32,
            Chip,
            Cluster,
            m![1 # 256],
            m![{ Groups<Rows> }, { Blocks<Hidden> }, { Lanes<Rows> }],
        > = context
            .sub
            .begin(ranks.view())
            .fetch::<m![{ Groups<Rows> }, { Blocks<Hidden> }], Lanes<Rows>>()
            .collect::<m![{ Groups<Rows> }, { Blocks<Hidden> }], Lanes<Rows>>()
            .to_vrf();
        context.main.begin(shifted.view())
            .fetch::<m![{ Groups<Rows> }, { Blocks<Hidden> }], Lanes<Rows>>()
            .collect::<m![{ Groups<Rows> }, { Blocks<Hidden> }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>()
            .vector_fxp(FxpBinaryOp::AddFxpSat, &candidates)
            .vector_narrow_split::<m![{ Groups<Rows> }, { Blocks<Hidden> }, { Layout<Rows>::Lanes } # 8 / 4], Half<Rows>>()
            .vector_intra_slice_reduce::<VocabPad, Times<Rows>, Half<Rows>>(IntraSliceReduceOpI32::Max)
            .vector_widen_concat::<Groups<Rows>, Lanes<Rows>>()
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit_view(rank.view_mut());
    }

    fn otherwise(self, context: &mut Device) {
        let (ranks, shifted, rank) = (self.ranks, self.shifted, self.out);
        #[unroll]
        for batch in 0..<Batches<Rows> as M>::SIZE {
            let candidates: VrfTensor<i32, Chip, Cluster, m![1 # 256],
                m![{ Batch<Rows> }, { Blocks<Hidden> }, { Lanes<Rows> }]> =
            context.sub.begin(ranks.view().tile::<Batches<Rows>, 1,
                m![{ Blocks<Hidden> }, 1 # ({ Batches<Rows> }), { Batch<Rows> }, { Lanes<Rows> }],
            >(batch))
                .fetch::<m![{ Batch<Rows> }, { Blocks<Hidden> }], Lanes<Rows>>()
                .collect::<m![{ Batch<Rows> }, { Blocks<Hidden> }], Lanes<Rows>>()
                .to_vrf();
            context.main.begin(shifted.view().tile::<Batches<Rows>, 1,
                m![{ Blocks<Hidden> }, 1 # ({ Batches<Rows> }), { Batch<Rows> }, { Lanes<Rows> }],
            >(batch))
            .fetch::<m![{ Batch<Rows> }, { Blocks<Hidden> }], Lanes<Rows>>()
            .collect::<m![{ Batch<Rows> }, { Blocks<Hidden> }], Lanes<Rows>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>()
            .vector_fxp(FxpBinaryOp::AddFxpSat, &candidates)
            .vector_narrow_split::<m![{ Batch<Rows> }, { Blocks<Hidden> }, { Layout<Rows>::Lanes } / 4], Half<Rows>>()
            .vector_intra_slice_reduce::<VocabPad, m![{ Batch<Rows> }, { Layout<Rows>::Lanes } / 4], Half<Rows>>(IntraSliceReduceOpI32::Max)
            .vector_widen_concat::<Batch<Rows>, Lanes<Rows>>()
            .vector_final()
            .commit_trim::<Lanes<Rows>>()
            .commit_view(rank.view_mut().tile::<Batches<Rows>, 1,
                m![1 #{!} ({ Batches<Rows> }), { Batch<Rows> }, { Lanes<Rows> }],
            >(batch));
        }
    }
}
