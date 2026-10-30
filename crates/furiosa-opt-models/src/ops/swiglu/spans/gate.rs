use furiosa_opt_std::prelude::*;

use crate::Op;
use core::marker::PhantomData;

pub(crate) type Normalized<Hidden, Rows, Chip> =
    HbmTensor<bf16, Chip, m![Rows / 8, Hidden / 32, Rows % 8, Hidden % 32]>;

/// Each row group's rows moved to the slices of its row quarter, sixteen features a packet.
pub(crate) struct Pack<Rows, Hidden, Partition, Output>(
    pub(crate) PhantomData<(Rows, Hidden, Partition, Output)>,
);

impl<
    'i,
    Rows: AxisName,
    Hidden: AxisName,
    Chip: M,
    Cluster: M,
    Slice: M,
    Element: M,
    Partition: M,
    Output: M,
> Op<DmTensorView<'i, bf16, Chip, Cluster, Slice, Element>>
    for Pack<Rows, Hidden, Partition, Output>
{
    type Output = DmTensor<bf16, Chip, Cluster, Partition, Output>;

    fn forward(
        &self,
        context: &mut Device,
        input: DmTensorView<'i, bf16, Chip, Cluster, Slice, Element>,
    ) -> Self::Output {
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
}

/// The packed rows stored in HBM by groups of eight rows, the live rows alone below a wave.
pub(crate) struct Store;

/// Packed rows: each cluster's row half, its quarters on the slices of each feature block.
type Packed<Hidden, Rows, Chip> = DmTensor<
    bf16,
    Chip,
    m![Rows / (Rows / 2)],
    m![Hidden / 32, Rows % (Rows / 2) / (Rows / 4)],
    m![Rows % (Rows / 4) / 8, Rows % 8, Hidden % 32],
>;

impl<'a, Hidden: AxisName, Rows: AxisName, Chip: M>
    Op<(&'a Packed<Hidden, Rows, Chip>, (usize, usize))> for Store
{
    type Output = Normalized<Hidden, Rows, Chip>;

    fn forward(
        &self,
        context: &mut Device,
        (packed, (live, rows)): (&'a Packed<Hidden, Rows, Chip>, (usize, usize)),
    ) -> Self::Output {
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
}
