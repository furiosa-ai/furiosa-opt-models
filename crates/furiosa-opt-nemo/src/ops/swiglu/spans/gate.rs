use furiosa_opt_std::prelude::*;

pub(crate) type Normalized<Hidden, Rows, Chip> =
    HbmTensor<bf16, Chip, m![Rows / 8, Hidden / 32, Rows % 8, Hidden % 32]>;

pub(crate) fn pack<
    Rows: AxisName,
    Hidden: AxisName,
    Chip: M,
    Cluster: M,
    Slice: M,
    Element: M,
    Partition: M,
    Output: M,
>(
    context: &mut Device,
    input: DmTensorView<'_, bf16, Chip, Cluster, Slice, Element>,
) -> DmTensor<bf16, Chip, Cluster, Partition, Output> {
    context
        .main
        .begin(input)
        .fetch::<m![Rows % (Rows / 4), Rows % (Rows / 2) / (Rows / 4)], m![Hidden % 16]>()
        .switch::<Partition, m![Rows % (Rows / 4), Hidden % 32 / 16]>(
            SwitchConfig::InterTranspose {
                slice1: 2,
                slice0: 1,
                time0: 1,
            },
        )
        .collect::<m![Rows % (Rows / 4), Hidden % 32 / 16], m![Hidden % 16]>()
        .commit_trim::<m![Hidden % 16]>()
        .commit()
}

pub(crate) fn store<Hidden: AxisName, Rows: AxisName, Chip: M>(
    context: &mut Device,
    packed: &DmTensor<
        bf16,
        Chip,
        m![Rows / (Rows / 2)],
        m![Hidden / 32, Rows % (Rows / 2) / (Rows / 4)],
        m![Rows % (Rows / 4) / 8, Rows % 8, Hidden % 32],
    >,
    (live, rows): (usize, usize),
) -> Normalized<Hidden, Rows, Chip> {
    let mut stored = HbmTensor::new();
    if rows <= 8 {
        // SAFETY: row groups split into the cluster halves and their quarters in order.
        let quarters = unsafe {
            stored.view_mut().reshape::<Chip, m![
                Rows / (Rows / 2),
                Rows % (Rows / 2) / (Rows / 4),
                Rows % (Rows / 4) / 8,
                Hidden / 32,
                Rows % 8,
                Hidden % 32
            ]>()
        };
        packed.view().slice_tile::<m![Rows % (Rows / 2) / (Rows / 4)], 1,
            m![Hidden / 32, Rows % (Rows / 2) / (Rows / 4) = 1 # 2]>(0)
            .tile::<m![Rows % (Rows / 4) / 8], 1,
                m![Rows % (Rows / 4) / 8 = 1 # (Rows / 32), Rows % 8, Hidden % 32]>(0)
            .to_hbm_view(&mut context.tdma, quarters.tile::<m![Rows % (Rows / 2) / (Rows / 4)], 1,
                m![Rows / (Rows / 2), Rows % (Rows / 2) / (Rows / 4) = 1 #{!} 2,
                    Rows % (Rows / 4) / 8, Hidden / 32, Rows % 8, Hidden % 32]>(0)
                .tile::<m![Rows % (Rows / 4) / 8], 1,
                    m![Rows / (Rows / 2), Rows % (Rows / 2) / (Rows / 4) = 1 #{!} 2,
                        Rows % (Rows / 4) / 8 = 1 #{!} (Rows / 32), Hidden / 32, Rows % 8,
                        Hidden % 32]>(0));
    } else if live == 1 {
        // SAFETY: row groups split into the cluster halves and their quarters in order.
        let quarters = unsafe {
            stored.view_mut().reshape::<Chip, m![
                Rows / (Rows / 2),
                Rows % (Rows / 2) / (Rows / 4),
                Rows % (Rows / 4) / 8,
                Hidden / 32,
                Rows % 8,
                Hidden % 32
            ]>()
        };
        packed.view().slice_tile::<m![Rows % (Rows / 2) / (Rows / 4)], 1,
            m![Hidden / 32, Rows % (Rows / 2) / (Rows / 4) = 1 # 2]>(0)
            .to_hbm_view(&mut context.tdma, quarters.tile::<m![Rows % (Rows / 2) / (Rows / 4)], 1,
                m![Rows / (Rows / 2), Rows % (Rows / 2) / (Rows / 4) = 1 #{!} 2,
                    Rows % (Rows / 4) / 8, Hidden / 32, Rows % 8, Hidden % 32]>(0));
    } else {
        packed
            .view()
            .to_hbm_view(&mut context.tdma, stored.view_mut());
    }
    stored
}
