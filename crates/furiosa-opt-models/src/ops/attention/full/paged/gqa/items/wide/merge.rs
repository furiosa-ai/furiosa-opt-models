//! The merge of a row's items: a job a row, eight jobs a tile on 32 slices each, two inputs a
//! slice, the inputs in ascending segment order.
//!
//! - Inputs are the items' f32 outputs `o` and log-sum-exps `lse`; a job takes
//!   `sum(e^(lse - top) o) / (sum(e^(lse - top)) + 1e-6)`, `top` its largest `lse`, reduced in
//!   pairs and then over its 32 slices, and rounds it into its row once.
//! - A row of one item has no inputs and writes the spare row: its wave wrote its row.

use super::*;
use crate::Op;
use crate::ops::attention::softmax::merge::merge;

/// A value a flit on slices `Slices`, `T` steps of them.
type Values<Slices, T> = DmTensor<i32, Chip, m![1 # 2], Slices, m![{ T }, 1 # 8]>;
/// A main-engine stream of [`Values`] at stage `S`.
type Flits<'l, Slices, T, S = stage::Tag, Stash = Fresh> = VectorTensor<
    'l,
    { Tu::Main },
    S,
    i32,
    Chip,
    m![1 # 2],
    Slices,
    T,
    m![1 # 8],
    Stash,
    { stage::VeOrder::IntraFirst },
>;

/// Applies stages to every value of an input, from the tag stage on, and commits them.
struct Map;

impl<Slices: M, T: M, S: stage::Stage, Stash: StashState, F> Op<(&Values<Slices, T>, F)> for Map
where
    F: for<'l> FnOnce(Flits<'l, Slices, T>) -> Flits<'l, Slices, T, S, Stash>,
{
    type Output = Values<Slices, T>;

    fn forward(
        &self,
        context: &mut Device,
        (input, stages): (&Values<Slices, T>, F),
    ) -> Values<Slices, T> {
        let stream = context.main.begin(input.view()).fetch::<T, m![1 # 8]>();
        let stream = stream.collect::<T, m![1 # 8]>().vector_init();
        stages(stream.vector_intra_slice_tag(TagMode::Zero))
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit()
    }
}

/// One value a job, or a job's input, every job on every slice.
type Jobs<T> = Values<m![1 # 256], T>;

/// Rows `8 tile .. 8 tile + 8` of `items`, each merged from its items' `partials` and `scales`
/// into its row of `attended`.
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, Waves: AxisName, Rows: AxisName>
    Op<(
        &Wide<KvHeads, Head, Waves>,
        (
            &HbmTensor<f32, Chip, m![Columns, KvHeads, Groups, Head]>,
            &HbmTensor<f32, Chip, m![Columns, KvHeads, Groups, Lane]>,
        ),
        usize,
        HbmTensorViewMut<'_, bf16, Chip, m![Rows, KvHeads, Groups, Head]>,
    )> for Attention<KvHeads, Groups, Head>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (items, (partials, scales), tile, attended): (
            &Wide<KvHeads, Head, Waves>,
            (
                &HbmTensor<f32, Chip, m![Columns, KvHeads, Groups, Head]>,
                &HbmTensor<f32, Chip, m![Columns, KvHeads, Groups, Lane]>,
            ),
            usize,
            HbmTensorViewMut<'_, bf16, Chip, m![Rows, KvHeads, Groups, Head]>,
        ),
    ) {
        let mut rows = |table: &HbmTensor<i32, Chip, m![Decodes, 1 # 8]>| -> Jobs<m![Eight]> {
            // SAFETY: a tile's eight rows of a table nothing writes once derived.
            let rows: HbmTensor<i32, Chip, m![Eight, 1 # 8]> = unsafe {
                let tile = table
                    .view()
                    .tile::<m![Decodes], 8, m![Decodes = 8 # Decodes, 1 # 8]>(8 * tile);
                tile.reshape::<Chip, m![Eight # Decodes, 1 # 8]>().alias()
            };
            rows.to_dm(&mut context.tdma)
        };
        let jobs = &items.jobs;
        let (firsts, counts, indices) = (rows(&jobs.firsts), rows(&jobs.counts), rows(&jobs.rows));
        // A job's inputs: its row's items when it has two or more; else it writes the spare row.
        let merging = Op::forward(
            &Map,
            &mut *context,
            (&counts, |stream| {
                let stream = stream.vector_fxp(FxpBinaryOp::AddFxp, -1);
                stream
                    .vector_clip(ClipBinaryOpI32::Max, 0)
                    .vector_clip(ClipBinaryOpI32::Min, 1)
            }),
        );
        let operand = |context: &mut Device,
                       input: &Jobs<m![Eight]>|
         -> VrfTensor<i32, Chip, m![1 # 2], m![1 # 256], m![Eight, 1 # 8]> {
            context
                .sub
                .begin(input.view())
                .fetch::<m![Eight], m![1 # 8]>()
                .collect::<m![Eight], m![1 # 8]>()
                .to_vrf()
        };
        let merging = operand(context, &merging);
        let inputs = Op::forward(
            &Map,
            &mut *context,
            (&counts, |stream| {
                stream.vector_fxp(FxpBinaryOp::MulInt, &merging)
            }),
        );
        let width = const { (<m![KvHeads, Groups, Head]>::SIZE * size_of::<bf16>()) as i32 };
        let spare = const { ((Rows::SIZE - 1) * <m![KvHeads, Groups, Head]>::SIZE * 2) as i32 };
        let targets = Op::forward(
            &Map,
            &mut *context,
            (&indices, |stream| {
                stream
                    .vector_fxp(FxpBinaryOp::MulInt, width)
                    .vector_fxp(FxpBinaryOp::AddFxp, -spare)
            }),
        );
        let targets = Op::forward(
            &Map,
            &mut *context,
            (&targets, |stream| {
                stream
                    .vector_fxp(FxpBinaryOp::MulInt, &merging)
                    .vector_fxp(FxpBinaryOp::AddFxp, spare)
            }),
        );
        let (firsts, limits) = (operand(context, &firsts), operand(context, &inputs));
        // Input `q` of job `j`: its row's item `first + q`, live while `q` is below its inputs.
        let ordinals: Jobs<m![Eight, Input]> = jobs.inputs.to_dm(&mut context.tdma);
        let slots = Op::forward(
            &Map,
            &mut *context,
            (&ordinals, |stream| {
                stream.vector_fxp(FxpBinaryOp::AddFxp, &firsts)
            }),
        );
        let live = Op::forward(
            &Map,
            &mut *context,
            (&ordinals, |stream| {
                stream
                    .vector_fxp(FxpBinaryOp::SubFxp, &limits)
                    .vector_fxp(FxpBinaryOp::ArithRightShift, 31)
            }),
        );
        // Each job input's slot as a byte offset, `size` bytes a slot.
        let bytes = |context: &mut Device, size: i32| -> HbmTensor<i32, Chip, m![Eight, Input]> {
            let bytes = Op::forward(
                &Map,
                &mut *context,
                (&slots, |stream| {
                    stream.vector_fxp(FxpBinaryOp::MulInt, size)
                }),
            );
            bytes.to_hbm(&mut context.tdma)
        };
        let pieces = bytes(context, Plan::partial_stride::<KvHeads, Groups, Head>());
        let parts = bytes(context, Plan::scale_stride::<KvHeads, Groups>());
        let inputs: HbmTensor<i32, Chip, m![Eight]> = inputs.to_hbm(&mut context.tdma);
        let targets: HbmTensor<i32, Chip, m![Eight]> = targets.to_hbm(&mut context.tdma);
        let live: HbmTensor<i32, Chip, m![Eight, Input]> = live.to_hbm(&mut context.tdma);
        let live: DmTensor<i32, Chip, Self::ItemCluster, m![Eight, Input / 2], m![Input % 2]> =
            live.to_dm(&mut context.tdma);
        let pieces: Self::MergeDm<
            Input,
            m![Input % 2, KvHeads % ({ Local<KvHeads> }), Groups, Head],
        > = partials
            .view()
            .gather::<m![Columns], m![KvHeads, Groups, Head]>()
            .by_byte_offsets(&pieces)
            .prefix::<m![Input], _>(&inputs, 0)
            .to_dm(&mut context.tdma);
        let terms: Self::MergeDm<
            Input,
            m![Input % 2, KvHeads % ({ Local<KvHeads> }), Groups, Lane],
        > = scales
            .view()
            .gather::<m![Columns], m![KvHeads, Groups, Lane]>()
            .by_byte_offsets(&parts)
            .prefix::<m![Input], _>(&inputs, 0)
            .to_dm(&mut context.tdma);
        // SAFETY: a stored register's first lane is its value, the rest padding.
        let terms: Self::MergeDm<
            Input,
            m![Input % 2, KvHeads % ({ Local<KvHeads> }), Groups, 1 # 8],
        > = unsafe { terms.reshape() };
        let tags: VrfTensor<
            i4,
            Chip,
            Self::ItemCluster,
            m![Eight, Input / 2],
            m![Input % 2, KvHeads % ({ Local<KvHeads> }), Groups, 1 # 8],
        > = context
            .main
            .begin(live.view())
            .fetch::<m![Input % 2, KvHeads % ({ Local<KvHeads> }), Groups], m![1 # 8]>()
            .collect::<m![Input % 2, KvHeads % ({ Local<KvHeads> }), Groups], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Equal(0),
                Cmp::False,
                Cmp::False,
                Cmp::False,
            ]))
            .vector_log(&mut context.sub);
        // A slot past a job's inputs is unspecified: its term is negative infinity.
        let terms: Self::MergeDm<
            Input,
            m![Input % 2, KvHeads % ({ Local<KvHeads> }), Groups, 1 # 8],
        > = context
            .main
            .begin(terms.view())
            .fetch::<m![Input % 2, KvHeads % ({ Local<KvHeads> }), Groups], m![1 # 8]>()
            .collect::<m![Input % 2, KvHeads % ({ Local<KvHeads> }), Groups], m![1 # 8]>()
            .vector_init()
            .vector_tagged(&tags)
            .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(ADMITTED, 0.0f32))
            .vector_clip(
                ClipBinaryOpF32::Add,
                Branched::imm(ADMITTED, f32::NEG_INFINITY),
            )
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        // Each of a job's 32 slices adds its part of the epsilon, so it is added once.
        let merged: DmTensor<
            bf16,
            Chip,
            Self::ItemCluster,
            m![Eight, 1 # 32],
            m![KvHeads % ({ Local<KvHeads> }), Groups, Head],
        > = merge::<
            Input,
            m![Input % 2],
            m![KvHeads % ({ Local<KvHeads> }), Groups],
            m![Head],
            m![Eight, 32],
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        >(context, (pieces.view(), terms.view()), 1e-6f32 / 32.0);
        // A tile's jobs write distinct rows, or the spare row with any bytes.
        let scatter = attended.scatter::<m![Rows], m![KvHeads, Groups, Head]>();
        scatter
            .by_byte_offsets(targets.view())
            .from_dm(&mut context.tdma, merged);
    }
}
