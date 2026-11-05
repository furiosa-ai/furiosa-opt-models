use furiosa_opt_std::prelude::*;

use crate::ops::layout::Rows as Layout;

use super::projection::Cluster;
use crate::ops::layout::axes::Chip;
use crate::ops::vocabulary::VocabPad;
use crate::{ops::linear::Groups, ops::vocabulary::grouped::Grouped, ops::weight::Weight};

pub(crate) mod tokens;

/// Two passes' weights fit beside the input rows in DM, allowing the next pass to stream
/// during the current contraction.
pub(crate) type PassRows = m![264];

/// The contraction chunk determines how many vocabulary passes fit in DM.
pub trait Project: AxisName {
    /// The contraction chunk one slice holds, its block of every hidden row.
    type Chunk: Count;
}

/// The slices sharing one row's contraction, `Hidden / chunk`.
pub(crate) type Parts<Hidden> = m![Hidden / ({ <Hidden as Project>::Chunk })];
/// The vocabulary rows one slice writes over every pass, `VocabPad * parts / 512`: its block of
/// the rows, `PassRows` a pass, since the blocks fill the `256 / parts` slots of both clusters.
pub(crate) type Block<Hidden> = m![(VocabPad, { Parts<Hidden> }) / 512];
/// The passes a slice's block takes.
pub(crate) type Passes<Hidden> = m![{ Block<Hidden> } / ({ PassRows })];

type Slice<Hidden> =
    m![VocabPad % (VocabPad / 2) / ({ Block<Hidden> }), 1 # ({ m![1] }), { Parts<Hidden> }];
pub(crate) type Reduced<Hidden> = m![
    VocabPad % (VocabPad / 2) / ({ Block<Hidden> }),
    1 # ({ m![1] }),
    { m![1 # ({ Parts<Hidden> })] }
];

pub(crate) type Selected<Rows> =
    DmTensor<i32, Chip, m![1 # 2], m![1 # 256], m![Rows # ({ Symbol<Selection<Rows>> })]>;

#[derive(Debug, Clone)]
pub(crate) struct Selection<Rows>(std::marker::PhantomData<Rows>);

impl<Rows: Grouped> AxisName for Selection<Rows> {
    const NAME: Ident = Rows::NAME;
    // Several rows keep each group's eight token lanes, the cells the head's tokens occupy, so
    // the head returns those cells in place.
    const SIZE: usize = if Rows::SIZE < 2 {
        2
    } else {
        <Groups<Rows> as M>::SIZE * 8
    };
}

pub(crate) type PassWeight<Hidden> = DmTensor<
    bf16,
    Chip,
    Cluster,
    Slice<Hidden>,
    m![
        VocabPad % ({ PassRows }),
        Hidden % ({ <Hidden as Project>::Chunk })
    ],
>;

/// Selects the first maximum of rows already in TRF from pass 0's weight in `first`. Each pass
/// issues the next pass's weight before its contraction, so the weight streams under the compute.
fn select_lanes<Hidden: Project, Rows: Grouped>(
    context: &mut Device,
    weight: &Weight<m![VocabPad, Hidden]>,
    lanes: &TrfTensor<
        bf16,
        Chip,
        Cluster,
        Slice<Hidden>,
        m![{ Layout<Rows>::Lanes }],
        m![{ Groups<Rows> }, Hidden % ({ Hidden::Chunk })],
    >,
    masks: &Columns<Hidden, f32>,
    iotas: &Columns<Hidden, i32>,
    first: PassWeight<Hidden>,
) -> Selected<Rows> {
    let mut candidates = crate::ops::vocabulary::select::Candidates::<Hidden, Rows>::new();
    let mut current = first;
    #[unroll]
    for tile in 0..<Passes<Hidden> as M>::SIZE {
        let mut next: DmTensor<
            bf16,
            Chip,
            Cluster,
            Slice<Hidden>,
            m![VocabPad % ({ PassRows }), Hidden % ({ Hidden::Chunk })],
        > = DmTensor::new();
        if tile + 1 < <Passes<Hidden> as M>::SIZE {
            weight
                .tensor()
                .view()
                .tile::<m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1, m![
                    VocabPad / ({ Block<Hidden> }),
                    VocabPad % ({ PassRows }) # ({ Block<Hidden> }), Hidden
                ]>(tile + 1)
                .to_dm_view(&mut context.tdma, next.view_mut());
        }
        let logits = crate::Module::forward(
            &crate::ops::vocabulary::LmHead(&current, core::marker::PhantomData::<Hidden>),
            (&mut *context, tokens::Projection::<Hidden, Rows> { lanes, mask: masks
                .view()
                .tile::<m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1, Column<Hidden>>(
                    tile,
                ) }),
        );
        candidates.push(
            context,
            &logits,
            iotas
                .view()
                .tile::<m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1, Column<Hidden>>(
                    tile,
                ),
            tile,
        );
        current = next;
    }
    candidates.select(context)
}

fn load_weight<Hidden: Project>(
    context: &mut Device,
    weight: &Weight<m![VocabPad, Hidden]>,
    tile: usize,
) -> PassWeight<Hidden> {
    weight
        .tensor()
        .view()
        .tile::<m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1, m![
            VocabPad / ({ Block<Hidden> }),
            VocabPad % ({ PassRows }) # ({ Block<Hidden> }), Hidden
        ]>(tile)
        .to_dm(&mut context.tdma)
}

/// The row groups of one vocabulary row, padded by one so that walking the vocabulary axis
/// for a fixed group steps 544 bytes, not 512: a 512-byte stride lands on the same four of a
/// slice's sixteen eight-byte banks every cycle, which the emitter treats as saturating the data
/// memory and serializes against the weight stream.
pub(crate) type PaddedGroups = m![17];

/// One per-vocabulary vector, every pass of each slice's block resident at once.
pub(crate) type Columns<Hidden, D> = DmTensor<
    D,
    Chip,
    Cluster,
    Reduced<Hidden>,
    m![VocabPad % ({ Block<Hidden> }) / ({ PassRows }), VocabPad % ({ PassRows })],
>;
/// One pass of a resident per-vocabulary vector.
pub(crate) type Column<Hidden> =
    m![1 # (VocabPad % ({ Block<Hidden> }) / ({ PassRows })), VocabPad % ({ PassRows })];

fn project<Hidden: Project, Rows: Grouped, Count: M, const PAD: usize>(
    context: &mut Device,
    weight: &DmTensor<
        bf16,
        Chip,
        Cluster,
        Slice<Hidden>,
        m![VocabPad % ({ PassRows }), Hidden % ({ Hidden::Chunk })],
    >,
    lanes: &TrfTensor<
        bf16,
        Chip,
        Cluster,
        Slice<Hidden>,
        m![{ Layout<Rows>::Lanes }],
        m![{ Count }, Hidden % ({ Hidden::Chunk })],
    >,
    mask: DmTensorView<'_, f32, Chip, Cluster, Reduced<Hidden>, Column<Hidden>>,
) -> DmTensor<
    f32,
    Chip,
    Cluster,
    Reduced<Hidden>,
    m![VocabPad % ({ PassRows }), { Count } # { PAD }, { Layout<Rows>::Lanes } # 8],
> {
    let mask: VrfTensor<f32, Chip, Cluster, Reduced<Hidden>, m![VocabPad % ({ PassRows })]> =
        context
            .sub
            .begin(mask)
            .fetch::<m![VocabPad % ({ PassRows }) / 8], m![VocabPad % 8]>()
            .collect::<m![VocabPad % ({ PassRows }) / 8], m![VocabPad % 8]>()
            .to_vrf();
    context.main.begin(weight.view())
            .fetch::<m![VocabPad % ({ PassRows }), Hidden % ({ Hidden::Chunk }) / 16], m![Hidden % 16]>()
            .collect::<m![VocabPad % ({ PassRows }), Hidden % ({ Hidden::Chunk }) / 16], m![Hidden % 16]>()
            .contract_outer::<m![VocabPad % ({ PassRows }), Hidden % ({ Hidden::Chunk }) / 32, { Count }], m![Hidden % 32], _, _, _>(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![VocabPad % ({ PassRows }), { Count }]>()
            .contract_lane::<m![VocabPad % ({ PassRows }), { Count }], m![{ Layout<Rows>::Lanes } # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Reduced<Hidden>, m![VocabPad % ({ PassRows }), { Count }]>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![VocabPad % ({ PassRows }), { Count }, { Layout<Rows>::Lanes } # 8 / 4],
                m![{ Layout<Rows>::Lanes } # 8 % 4],
            >()
            .vector_fp_binary(FpBinaryOp::AddF, &mask)
            .vector_widen_concat::<m![VocabPad % ({ PassRows }), { Count }], m![{ Layout<Rows>::Lanes } # 8]>()
            .vector_final()
            .commit_trim::<m![{ Layout<Rows>::Lanes } # 8]>()
            .commit()
}
