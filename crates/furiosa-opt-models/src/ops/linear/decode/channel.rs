use super::*;
use crate::ops::vector::Restore;
use crate::tensor::form::BlockScaled;
use crate::tensor::parallel::Chip;
use core::borrow::Borrow;

impl<In: AxisName, Out: AxisName, K, V> Matrix
    for BlockScaled<In, Out, f8e4m3, f32, m![Out, 1 # 8], (), Chip, K, V>
where
    K: Borrow<HbmTensor<f8e4m3, Chip, m![Out, In]>>,
    V: Borrow<HbmTensor<f32, Chip, m![Out, 1 # 8]>>,
{
    type In = In;
    type Out = Out;

    fn tile<T: Tiling<In, Out>>(
        &self,
        device: &mut Device,
        at: usize,
    ) -> DmTensor<bf16, m![1], T::Cluster, T::Slices, m![{ T::Rows }, { T::Inputs }]> {
        type Selected<R> = m![{ R } = 1 # ({ R })];
        type SelectedMut<R> = m![{ R } = 1 #{!} ({ R })];
        let coded: DmTensor<f8e4m3, m![1], T::Cluster, T::Slices, m![{ T::Rows }, { T::Inputs }]> =
            T::to_dm(device, self.codes().view(), at);
        let scales: DmTensor<f32, m![1], T::Cluster, T::Slices, m![{ T::Rows }, 1 # 8]> =
            T::to_dm(device, self.scales().view(), at);
        let mut restored = DmTensor::new();
        for part in 0..<T::Restorations as M>::SIZE {
            let decoded: DmTensor<
                bf16,
                m![1],
                T::Cluster,
                T::Slices,
                m![{ T::Restored }, { T::Inputs }],
            > = device
                .main
                .begin(coded.view().tile::<T::Restorations, 1, m![
                    { Selected<T::Restorations> }, { T::Restored }, { T::Inputs }
                ]>(part))
                .fetch::<m![{ T::Restored }, { T::Inputs } / 16], m![In % 16]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![{ T::Restored }, { T::Inputs } / 16], m![In % 16]>()
                .commit_trim::<m![In % 16]>()
                .commit();
            let scale: VrfTensor<f32, m![1], T::Cluster, T::Slices, m![{ T::Restored }, 1 # 8]> =
                device
                    .sub
                    .begin(scales.view().tile::<T::Restorations, 1, m![
                        { Selected<T::Restorations> }, { T::Restored }, 1 # 8
                    ]>(part))
                    .fetch::<T::Restored, m![1 # 8]>()
                    .collect::<T::Restored, m![1 # 8]>()
                    .to_vrf();
            device
                .main
                .begin(decoded.view())
                .fetch::<m![{ T::Restored }, { T::Inputs } / 16], m![In % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![{ T::Restored }, { T::Inputs } / 8], m![In % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ T::Restored }, { T::Inputs } / 4], m![In % 4]>()
                .vector_restore::<m![{ T::Restored }, { T::Inputs } / 8], m![In % 8], _, _>(
                    &scale,
                    |values| values,
                )
                .commit_view(restored.view_mut().tile::<T::Restorations, 1, m![
                    { SelectedMut<T::Restorations> }, { T::Restored }, { T::Inputs }
                ]>(part));
        }
        restored
    }
}
