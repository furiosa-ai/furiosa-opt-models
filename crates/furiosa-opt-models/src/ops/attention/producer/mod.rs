use crate::ops::linear;
use crate::ops::normalization::{Lanes, RmsNorm, Root, Scale, Squares};
use crate::tensor::form::Weight;
use crate::tensor::mapping::Rows as Layout;
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;

use super::Chip;
use super::geometry;

type Groups = geometry::QueryGroups<4>;
type Hidden = geometry::Hidden<4096>;
type QkvRows = geometry::HeadRows<6>;

type Cluster<KvHeads> = m![KvHeads / 4];
type QuerySlice<KvHeads, Head> = m![KvHeads % 4, Groups, Head / 32, Hidden / 1024];
type QueryReduced<KvHeads, Head> = m![KvHeads % 4, Groups, Head / 8];

pub(crate) type Query<KvHeads, Head, Rows> =
    DmTensor<bf16, Chip, Cluster<KvHeads>, QueryReduced<KvHeads, Head>, m![Rows, Head % 8]>;

/// Each projection reads the same hidden owner and retains its head coordinates for consumers.
mod key_value;
mod leaves;

type Packed<'a, KvHeads, Head> = HbmTensorView<'a, bf16, Chip, m![KvHeads, QkvRows, Head, Hidden]>;
type QueryWeight<KvHeads, Head> =
    DmTensor<bf16, Chip, Cluster<KvHeads>, QuerySlice<KvHeads, Head>, m![Head % 32, Hidden % 1024]>;

type KeySlice<KvHeads, Head> = m![KvHeads, Head / 4];

pub(crate) type QueryRotated<KvHeads, Head, Rows> =
    DmTensor<bf16, Chip, Cluster<KvHeads>, m![KvHeads % 4, Groups, 1 # 16], m![Rows, Head]>;

type KeyRotated<KvHeads, Head, Rows> =
    DmTensor<bf16, Chip, m![2], m![KvHeads, 1 # 32], m![Rows, Head]>;

/// How a head's `Part` features a row move through its norm: one packet a row, padded to eight
/// lanes, split to four or, when it holds four, trimmed.
type Norm<Head, Rows, Part> = Lanes<
    m![Rows],
    m![Head % ({ Part }) # 8],
    m![Rows, Head % ({ Part }) / 4],
    m![Head % 4],
    m![Rows],
    m![Head % ({ Part }) # 8],
    m![Head % ({ Part })],
>;

fn load_query<KvHeads: AxisName, Head: AxisName>(
    device: &mut Device,
    weights: Packed<'_, KvHeads, Head>,
) -> QueryWeight<KvHeads, Head> {
    let query: DmTensor<
        bf16,
        Chip,
        Cluster<KvHeads>,
        m![KvHeads % 4, QkvRows = 4, Head / 32, Hidden / 1024],
        m![Head % 32, Hidden % 1024],
    > = weights
        .tile::<m![QkvRows], 4, m![KvHeads, QkvRows = 4 # 6, Head, Hidden]>(0)
        .to_dm(&mut device.tdma);
    // SAFETY: the four selected query rows keep the same cells in the same physical class.
    let query: DmTensor<
        bf16,
        Chip,
        Cluster<KvHeads>,
        QuerySlice<KvHeads, Head>,
        m![Head % 32, Hidden % 1024],
    > = unsafe { query.reshape() };
    query
}

fn contract_query<KvHeads: AxisName, Head: AxisName, Rows: AxisName>(
    device: &mut Device,
    query: &QueryWeight<KvHeads, Head>,
    input: DmTensorView<'_, bf16, Chip, m![2], m![Hidden / 16 # 256], m![Rows, Hidden % 16]>,
) -> Query<KvHeads, Head, Rows> {
    let rows: TrfTensor<
        bf16,
        Chip,
        Cluster<KvHeads>,
        QuerySlice<KvHeads, Head>,
        m![{ Layout<Rows>::Lanes }],
        m![{ linear::Groups<Rows> }, Hidden % 1024],
    > = device
        .sub
        // SAFETY: both clusters carry the same hidden values; each projection borrows that owner.
        .begin(unsafe { input.clone().reshape::<
            Chip,
            Cluster<KvHeads>,
            m![Hidden / 16],
            m![Rows, Hidden % 16],
        >() })
        .fetch::<m![{ Layout<Rows>::Lanes }, { linear::Groups<Rows> }], m![Hidden % 16]>()
        .switch::<
            QuerySlice<KvHeads, Head>,
            m![{ Layout<Rows>::Lanes }, { linear::Groups<Rows> }, Hidden % 1024 / 16],
        >(
            SwitchConfig::TransposedBroadcast1 { slice1: 4, slice0: 64 },
        )
        .collect::<
            m![{ Layout<Rows>::Lanes }, { linear::Groups<Rows> }, Hidden % 1024 / 16],
            m![Hidden % 16],
        >()
        .to_trf();
    device
        .main
        .begin(query.view())
        .fetch::<m![{ linear::Groups<Rows> }, Head % 32, Hidden % 1024 / 16], m![Hidden % 16]>()
        .collect::<m![{ linear::Groups<Rows> }, Head % 32, Hidden % 1024 / 16], m![Hidden % 16]>()
        .contract_outer::<
            m![{ linear::Groups<Rows> }, Head % 32, Hidden % 1024 / 32],
            m![Hidden % 32],
            _,
            _,
            _,
        >(
            &rows,
        )
        .contract_packet::<m![1]>()
        .contract_time::<m![{ linear::Groups<Rows> }, Head % 32]>()
        .contract_lane::<
            m![{ linear::Groups<Rows> }, Head % 32],
            m![{ Layout<Rows>::Lanes } # 8],
        >(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<
            QueryReduced<KvHeads, Head>,
            m![{ linear::Groups<Rows> }, Head % 8],
        >(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
        .transpose::<
            m![{ linear::Groups<Rows> }, Head % 8 / 4, { Layout<Rows>::Lanes }],
            m![Head % 4 # 16],
        >()
        .commit_trim::<m![Head % 4]>()
        .commit()
}

fn normalize<
    Head: AxisName,
    Rows: AxisName,
    Chips: M,
    S: M,
    Reduced: M,
    Part: Count,
    const EPSILON: usize,
    T: Borrow<HbmTensor<bf16, m![1], m![Head]>>,
>(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Chips, S, m![Rows, Head % ({ Part })]>,
    norm: &RmsNorm<Head, EPSILON, m![1], Weight<m![Head], m![1], T>>,
) -> DmTensor<bf16, Chip, Chips, S, m![Rows, Head % ({ Part })]> {
    let mean: DmTensor<f32, Chip, Chips, Reduced, m![Rows, 1 # 8]> =
        Squares::<Norm<Head, Rows, Part>, Head, m![Rows], m![Head / ({ Part })], EPSILON>::of(
            &mut device.main,
            input.view(),
        )
        .vector_inter_slice_reduce::<Reduced, m![Rows]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // SAFETY: the all-reduced statistic occupies every feature slice of its own head.
    let mean: DmTensor<f32, Chip, Chips, S, m![Rows, 1 # 8]> = unsafe { mean.reshape() };
    let inverse: VrfTensor<f32, Chip, Chips, S, m![Rows, 1 # 8]> =
        Root::<m![Rows], m![1 # 8]>::of(&mut device.sub, mean.view(), |square| square).to_vrf();
    let weight: DmTensor<bf16, Chip, Chips, S, m![Head % ({ Part })]> =
        norm.weight().tensor().to_dm(&mut device.tdma);
    let weight: VrfTensor<f32, Chip, Chips, S, m![Rows, Head % ({ Part }) # 8]> = device
        .sub
        .begin(weight.view())
        .fetch::<m![Rows], m![Head % ({ Part }) # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows], m![Head % ({ Part }) # 8]>()
        .to_vrf();
    Scale::<Norm<Head, Rows, Part>>::of(&mut device.main, input.view(), &inverse, &weight).commit()
}

fn rotate<Head: AxisName, Rows: AxisName, Chips: M, S: M, Table: M>(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Chips, S, m![Rows, Head]>,
    low: HbmTensorView<'_, bf16, Chip, Table>,
    high: HbmTensorView<'_, bf16, Chip, Table>,
) -> DmTensor<bf16, Chip, Chips, S, m![Rows, Head]> {
    use crate::ops::rotary::{Half, OutputHalf, Row};

    let low: DmTensor<bf16, Chip, Chips, S, m![Rows, Head]> = low.to_dm(&mut device.tdma);
    let high: DmTensor<bf16, Chip, Chips, S, m![Rows, Head]> = high.to_dm(&mut device.tdma);
    // SAFETY: each head is its output half followed by its feature within that half.
    let low =
        unsafe { low.reshape::<Chip, Chips, S, m![Rows, OutputHalf, { Symbol<Row<Head>> }]>() };
    let high =
        unsafe { high.reshape::<Chip, Chips, S, m![Rows, OutputHalf, { Symbol<Row<Head>> }]>() };
    let coefficient: TrfTensor<
        bf16,
        Chip,
        Chips,
        S,
        m![OutputHalf],
        m![Rows, { Symbol<Row<Head>> } / 32, Half, { Symbol<Row<Head>> } % 32],
    > = device
        .sub
        .begin_interleaved::<Half, _, _, _, _, _>(low.view(), high.view())
        .fetch::<m![
            OutputHalf,
            Rows,
            { Symbol<Row<Head>> } / 32,
            Half,
            { Symbol<Row<Head>> } % 32 / 16
        ], m![{ Symbol<Row<Head>> } % 16]>()
        .collect::<m![
            OutputHalf,
            Rows,
            { Symbol<Row<Head>> } / 32,
            Half,
            { Symbol<Row<Head>> } % 32 / 16
        ], m![{ Symbol<Row<Head>> } % 16]>()
        .to_trf();
    // SAFETY: the two input halves retain their original order and physical class.
    let input = unsafe {
        input
            .view()
            .reshape::<Chip, Chips, S, m![Rows, Half, { Symbol<Row<Head>> }]>()
    };
    let rotated: DmTensor<
        bf16,
        Chip,
        Chips,
        S,
        m![Rows, OutputHalf, { Symbol<Row<Head>> }],
    > = device
        .main
        .begin(input)
        .fetch::<
            m![Rows, { Symbol<Row<Head>> } / 32, Half, { Symbol<Row<Head>> } % 32 / 16],
            m![{ Symbol<Row<Head>> } % 16],
        >()
        .collect::<
            m![Rows, { Symbol<Row<Head>> } / 32, Half, { Symbol<Row<Head>> } % 32 / 16],
            m![{ Symbol<Row<Head>> } % 16],
        >()
        .contract_outer::<
            m![Rows, { Symbol<Row<Head>> } / 32, Half],
            m![{ Symbol<Row<Head>> } % 32],
            _,
            _,
            _,
        >(&coefficient)
        .contract_packet::<m![{ Symbol<Row<Head>> } % 32]>()
        .contract_time::<m![Rows, { Symbol<Row<Head>> } / 32]>()
        .contract_lane::<
            m![Rows, { Symbol<Row<Head>> } / 32, OutputHalf, { Symbol<Row<Head>> } % 32 / 8],
            m![{ Symbol<Row<Head>> } % 8],
        >(
            LaneMode::Sequential,
        )
        .cast::<bf16, m![{ Symbol<Row<Head>> } % 8 # 16]>()
        .commit_trim::<m![{ Symbol<Row<Head>> } % 8]>()
        .commit();
    // SAFETY: each output half and feature become the original contiguous head index.
    unsafe { rotated.reshape() }
}
