use super::*;
use crate::ops::linear::Groups;
use crate::tensor::mapping::Rows as Layout;
use std::marker::PhantomData;

type ProjectionSlices<KvHeads> = m![KvHeads % 4 / 2, Hidden / 32];
type Reduced<KvHeads, Head> = m![KvHeads % 4, Head / 2];
pub(super) type Weight<KvHeads, Head> = DmTensor<
    bf16,
    Chip,
    Cluster<KvHeads>,
    ProjectionSlices<KvHeads>,
    m![KvHeads % 2, Head, Hidden % 32],
>;
type Input<KvHeads, Rows> =
    DmTensor<bf16, Chip, Cluster<KvHeads>, ProjectionSlices<KvHeads>, m![Rows, Hidden % 32]>;

#[derive(Debug, Clone)]
struct Aligned<T, const N: usize>(PhantomData<T>);

impl<T: M + 'static, const N: usize> AxisName for Aligned<T, N> {
    const NAME: Ident = Ident::new("Aligned");
    const SIZE: usize = T::SIZE.div_ceil(N) * N;
}

type RowFrame<Rows> = m![Rows # ({ Symbol<Aligned<Symbol<Rows>, 16>> })];

type Lanes<KvHeads, Rows> = TrfTensor<
    bf16,
    Chip,
    Cluster<KvHeads>,
    ProjectionSlices<KvHeads>,
    Layout<Rows>::Lanes,
    m![{ Groups<Rows> }, Hidden % 32],
>;

type Projected<KvHeads, Head, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<KvHeads>,
    Reduced<KvHeads, Head>,
    m![Head % 2, { Groups<Rows> }, { Layout<Rows>::Lanes } # 8],
>;

pub(super) fn order<KvHeads: AxisName, Head: AxisName, Rows: AxisName>(
    device: &mut Device,
    input: &Projected<KvHeads, Head, Rows>,
) -> DmTensor<bf16, Chip, m![2], KeySlice<KvHeads, Head>, m![Rows, Head % 4]> {
    let mut remapped: DmTensor<
        bf16,
        Chip,
        m![2],
        KeySlice<KvHeads, Head>,
        m![Head % 4, { RowFrame<Rows> }],
    > = DmTensor::new();
    input
        .view()
        .to_dm_view(&mut device.tdma, remapped.view_mut());
    let mut ordered: DmTensor<bf16, Chip, m![2], KeySlice<KvHeads, Head>, m![Rows, Head % 4]> =
        DmTensor::new();
    device
        .main
        .begin(remapped.view())
        .fetch::<m![{ RowFrame<Rows> } / 16, Head % 4], m![{ RowFrame<Rows> } % 16]>()
        .collect::<m![{ RowFrame<Rows> } / 16, Head % 4], m![{ RowFrame<Rows> } % 16]>()
        .transpose::<m![Rows], m![Head % 4 # 16]>()
        .commit_trim::<m![Head % 4]>()
        .commit_view(ordered.view_mut());
    ordered
}

pub(super) fn project<KvHeads: AxisName, Head: AxisName, Rows: AxisName>(
    device: &mut Device,
    key: &Weight<KvHeads, Head>,
    value: &Weight<KvHeads, Head>,
    input: DmTensorView<'_, bf16, Chip, m![2], m![Hidden / 16 # 256], m![Rows, Hidden % 16]>,
) -> (
    Projected<KvHeads, Head, Rows>,
    Projected<KvHeads, Head, Rows>,
) {
    const { assert!(Rows::SIZE > 0 && Rows::SIZE <= 64) };
    let mut staged: Input<KvHeads, Rows> = DmTensor::new();
    device
        .main
        // Both clusters contain the complete hidden input before head ownership is selected.
        .begin(unsafe {
            input.reshape::<Chip, Cluster<KvHeads>, m![Hidden / 16], m![Rows, Hidden % 16]>()
        })
        .fetch::<m![Rows], m![Hidden % 16]>()
        .switch::<ProjectionSlices<KvHeads>, m![Rows, Hidden % 32 / 16]>(
            SwitchConfig::TransposedBroadcast1 {
                slice1: 128,
                slice0: 2,
            },
        )
        .collect::<m![Rows, Hidden % 32 / 16], m![Hidden % 16]>()
        .commit_trim::<m![Hidden % 16]>()
        .commit_view(staged.view_mut());
    let lanes: Lanes<KvHeads, Rows> = device
        .sub
        .begin(staged.view())
        .fetch::<m![{ Layout<Rows>::Lanes }, { Groups<Rows> }, Hidden % 32 / 16], m![Hidden % 16]>()
        .collect::<
            m![{ Layout<Rows>::Lanes }, { Groups<Rows> }, Hidden % 32 / 16],
            m![Hidden % 16],
        >()
        .to_trf();
    let mut result: Projected<KvHeads, Head, Rows> = DmTensor::new();
    device
        .main
        .begin(key.view())
        .fetch::<m![Head % 16, KvHeads % 2, Head / 16, Hidden % 32 / 16], m![Hidden % 16]>()
        .collect::<m![Head % 16, KvHeads % 2, Head / 16, Hidden % 32 / 16], m![Hidden % 16]>()
        .contract_outer::<
            m![Head % 16, KvHeads % 2, Head / 16, { Groups<Rows> }],
            m![Hidden % 32], _, _, _,
        >(&lanes)
        .contract_packet::<m![1]>()
        .contract_time::<m![Head % 16, KvHeads % 2, Head / 16, { Groups<Rows> }]>()
        .contract_lane::<
            m![Head % 16, KvHeads % 2, Head / 16, { Groups<Rows> }],
            m![{ Layout<Rows>::Lanes } # 8],
        >(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<
            Reduced<KvHeads, Head>,
            m![Head % 2, { Groups<Rows> }],
        >(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
        .commit_trim::<m![{ Layout<Rows>::Lanes } # 8]>()
        .commit_view(result.view_mut());
    let mut output: Projected<KvHeads, Head, Rows> = DmTensor::new();
    device
        .main
        .begin(value.view())
        .fetch::<m![Head % 16, KvHeads % 2, Head / 16, Hidden % 32 / 16], m![Hidden % 16]>()
        .collect::<m![Head % 16, KvHeads % 2, Head / 16, Hidden % 32 / 16], m![Hidden % 16]>()
        .contract_outer::<
            m![Head % 16, KvHeads % 2, Head / 16, { Groups<Rows> }],
            m![Hidden % 32], _, _, _,
        >(&lanes)
        .contract_packet::<m![1]>()
        .contract_time::<m![Head % 16, KvHeads % 2, Head / 16, { Groups<Rows> }]>()
        .contract_lane::<
            m![Head % 16, KvHeads % 2, Head / 16, { Groups<Rows> }],
            m![{ Layout<Rows>::Lanes } # 8],
        >(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<
            Reduced<KvHeads, Head>,
            m![Head % 2, { Groups<Rows> }],
        >(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
        .commit_trim::<m![{ Layout<Rows>::Lanes } # 8]>()
        .commit_view(output.view_mut());
    (result, output)
}

pub(super) fn load<KvHeads: AxisName, Head: AxisName>(
    device: &mut Device,
    weights: Packed<'_, KvHeads, Head>,
    part: usize,
) -> Weight<KvHeads, Head> {
    let in_dm: DmTensor<
        bf16,
        Chip,
        Cluster<KvHeads>,
        m![
            KvHeads % 4 / 2,
            Hidden / 128,
            KvHeads % 2,
            QkvRows = 1,
            Head / 64
        ],
        m![Head % 64, Hidden % 128],
    > = weights
        .tile::<m![QkvRows], 1, m![KvHeads, QkvRows = 1 # 6, Head, Hidden]>(part)
        .to_dm(&mut device.tdma);
    // Removing the singleton projection preserves each coefficient's physical address.
    let in_dm = unsafe {
        in_dm.reshape::<
            Chip, Cluster<KvHeads>, m![KvHeads % 4 / 2, Hidden / 128, KvHeads % 2, Head / 64],
            m![Head % 64, Hidden % 128],
        >()
    };
    let mut weight: Weight<KvHeads, Head> = DmTensor::new();
    device
        .main
        .begin(in_dm.view())
        .fetch::<m![Head % 64, Hidden % 128 / 32], m![Hidden % 32]>()
        .switch::<ProjectionSlices<KvHeads>, m![Head % 64, KvHeads % 2, Head / 64]>(
            SwitchConfig::InterTranspose {
                slice1: 4,
                slice0: 1,
                time0: 1,
            },
        )
        .collect::<m![Head % 64, KvHeads % 2, Head / 64, Hidden % 32 / 16], m![Hidden % 16]>()
        .commit_trim::<m![Hidden % 16]>()
        .commit_view(weight.view_mut());
    weight
}
