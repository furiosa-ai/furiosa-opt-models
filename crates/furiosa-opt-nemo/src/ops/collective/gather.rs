use super::{AllGather, ChipMap, Shard};
use crate::{
    Module,
    ops::layout::{Dm, axes::Parallel},
};
use furiosa_opt_std::prelude::*;

impl<D: Scalar, W: AxisName> Module<(&mut Device, HbmTensor<D, m![1], m![W]>)>
    for AllGather<Parallel<1>, W>
{
    type Output = HbmTensor<D, m![1], m![W]>;

    fn forward(&self, (_, input): (&mut Device, Self::Output)) -> Self::Output {
        input
    }
}

/// One chip's rows of a launch already hold every column.
impl<S, X, W: AxisName> Module<(&mut Device, Dm<S, X>)> for AllGather<Parallel<1>, W> {
    type Output = Dm<S, X>;

    fn forward(&self, (_, input): (&mut Device, Dm<S, X>)) -> Self::Output {
        input
    }
}

impl<D: MaterializableScalar, W: AxisName, V: AxisName, const N: usize>
    Module<(&mut Device, HbmTensor<D, ChipMap<N>, m![W]>)> for AllGather<Parallel<N>, V>
{
    type Output = HbmTensor<D, ChipMap<N>, m![V]>;

    fn forward(
        &self,
        (device, input): (&mut Device, HbmTensor<D, ChipMap<N>, m![W]>),
    ) -> Self::Output {
        const {
            assert!(V::SIZE == W::SIZE * N);
            assert!(W::SIZE > 0 && W::SIZE % 1024 == 0);
        }
        let input: DmTensor<D, ChipMap<N>, m![1 # 2], m![W % 1024 / 4], m![W / 1024, W % 4]> =
            input.to_dm(&mut device.tdma);
        let output = input
            .to_dm::<Broadcast<N>, m![1 # 2], m![W % 1024 / 4], m![{ Symbol<Shard<N>> }, W / 1024, W % 4]>(
                &mut device.tdma,
            )
            .to_hbm::<m![{ Symbol<Shard<N>> }, W]>(&mut device.tdma);
        // SAFETY: contiguous [Shard, W] has index shard * W::SIZE + local, exactly V order.
        // Both chip mappings have N slots; consuming the owner preserves exclusive ownership.
        unsafe { output.reshape::<ChipMap<N>, m![V]>() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::layout::axes::Part;
    use furiosa_opt_std::runtime::launch;

    axes![Vocabulary = 131072];

    #[device(chip = 2)]
    fn gather(
        device: &mut Device,
        input: HbmTensor<bf16, ChipMap<2>, m![{ Symbol<Part<Vocabulary, 2>> }]>,
    ) -> HbmTensor<bf16, ChipMap<2>, m![Vocabulary]> {
        AllGather::<Parallel<2>, Vocabulary>::default().forward((device, input))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn preserves_vocabulary_order() {
        type Local = Part<Vocabulary, 2>;
        let mut device = Device::new(gather.topology()).unwrap();
        let values: Vec<_> = (0..Vocabulary::SIZE)
            .map(|i| {
                bf16::from_f32(
                    f32::from(u16::try_from(i % 251).unwrap())
                        + 512.0 * f32::from(u16::try_from(i / Local::SIZE).unwrap()),
                )
            })
            .collect();
        let input: HbmTensor<bf16, ChipMap<2>, m![Local]> =
            HostTensor::<bf16, m![{ Symbol<Shard<2>> }, Local]>::from_vec(values.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let output = launch(gather, (&mut device, input)).await.unwrap();
        let output = output
            .to_host::<m![{ Symbol<Shard<2>> }, Vocabulary]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for chip in output.chunks_exact(Vocabulary::SIZE) {
            assert_eq!(chip, values);
        }
    }
}
