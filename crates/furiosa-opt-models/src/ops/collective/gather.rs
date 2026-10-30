//! Every chip's part on every chip, the parts in chip order: chip-local vectors joined into one
//! vector, or the rows of each chip stacked into the rows of all chips.

use super::{ChipMap, Rows, Shard};
use crate::{
    Op,
    tensor::{
        live::Live,
        parallel::{Parallel, Part},
    },
};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Every chip's part on every chip of placement `P`, the parts in chip order.
#[derive(DeviceSend)]
pub struct AllGather<P>(PhantomData<P>);

impl<P> Default for AllGather<P> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// The rows of `N` chips' parts of `S` rows each, the parts in chip order.
#[derive(Clone, Debug)]
pub struct Pieces<S, const N: usize>(PhantomData<S>);

impl<S: AxisName, const N: usize> AxisName for Pieces<S, N> {
    const NAME: Ident = Ident::new("Pieces");
    const SIZE: usize = N * S::SIZE;
}

/// One chip's tensor already holds every part.
impl<D: Scalar, E: M> Op<HbmTensor<D, m![1], E>> for AllGather<Parallel<1>> {
    type Output = HbmTensor<D, m![1], E>;

    fn forward(&self, _: &mut Device, input: Self::Output) -> Self::Output {
        input
    }
}

/// One chip's rows of a launch already hold every column.
impl<X, R> Op<Live<X, R>> for AllGather<Parallel<1>> {
    type Output = Live<X, R>;

    fn forward(&self, _: &mut Device, input: Live<X, R>) -> Self::Output {
        input
    }
}

/// Each chip's part of the vector `V`, joined into `V` on every chip.
impl<D: MaterializableScalar, V: AxisName, const N: usize>
    Op<HbmTensor<D, ChipMap<N>, m![{ Symbol<Part<V, N>> }]>> for AllGather<Parallel<N>>
{
    type Output = HbmTensor<D, ChipMap<N>, m![V]>;

    fn forward(
        &self,
        device: &mut Device,
        input: HbmTensor<D, ChipMap<N>, m![{ Symbol<Part<V, N>> }]>,
    ) -> Self::Output {
        const {
            assert!(V::SIZE == Part::<V, N>::SIZE * N);
            assert!(Part::<V, N>::SIZE > 0 && Part::<V, N>::SIZE % 1024 == 0);
        }
        let input: DmTensor<
            D,
            ChipMap<N>,
            m![1 # 2],
            m![{ Symbol<Part<V, N>> } % 1024 / 4],
            m![{ Symbol<Part<V, N>> } / 1024, { Symbol<Part<V, N>> } % 4],
        > = input.to_dm(&mut device.tdma);
        let output = input
            .to_dm::<Broadcast<N>, m![1 # 2], m![{ Symbol<Part<V, N>> } % 1024 / 4], m![
                { Symbol<Shard<N>> }, { Symbol<Part<V, N>> } / 1024, { Symbol<Part<V, N>> } % 4
            ]>(&mut device.tdma)
            .to_hbm::<m![{ Symbol<Shard<N>> }, { Symbol<Part<V, N>> }]>(&mut device.tdma);
        // SAFETY: contiguous [Shard, W] has index shard * W::SIZE + local, exactly V order.
        // Both chip mappings have N slots; consuming the owner preserves exclusive ownership.
        unsafe { output.reshape::<ChipMap<N>, m![V]>() }
    }
}

/// Each chip's `S` rows, stacked into the rows of all chips on every chip.
impl<'x, D: MaterializableScalar, S: AxisName, W: AxisName, const N: usize> Op<&'x Rows<D, N, S, W>>
    for AllGather<Parallel<N>>
{
    type Output = Rows<D, N, Pieces<S, N>, W>;

    fn forward(&self, device: &mut Device, input: &'x Rows<D, N, S, W>) -> Self::Output {
        let mut output: HbmTensor<D, Broadcast<N>, m![{ Symbol<Shard<N>> }, S, W]> =
            HbmTensor::new();
        input
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
        // SAFETY: every chip holds the parts in chip order, each its rows in order.
        unsafe { output.reshape() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::collective::tests::{Small, value};
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![Vocabulary = 131072, Width = 5376];

    #[device(chip = 2)]
    fn gather(
        device: &mut Device,
        input: HbmTensor<bf16, ChipMap<2>, m![{ Symbol<Part<Vocabulary, 2>> }]>,
    ) -> HbmTensor<bf16, ChipMap<2>, m![Vocabulary]> {
        AllGather::<Parallel<2>>::default().forward(device, input)
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

    #[device(chip = 4)]
    fn gather_small(
        device: &mut Device,
        input: &Rows<bf16, 4, Small, Width>,
    ) -> Rows<bf16, 4, Pieces<Small, 4>, Width> {
        AllGather::<Parallel<4>>::default().forward(device, input)
    }

    /// The gathered rows are every chip's part in chip order, on every chip.
    #[tokio::test(flavor = "current_thread")]
    async fn gathers_the_parts() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(gather_small.topology()).unwrap();
        let (rows, width) = (Small::SIZE, Width::SIZE);
        let parts: Vec<bf16> = (0..4 * rows)
            .flat_map(|row| (0..width).map(move |column| value(row, column)))
            .collect();
        let input = HostTensor::<bf16, m![{ Symbol<Shard<4>> }, Small, Width]>::from_vec(parts)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let gathered = launch(gather_small, (&mut device, &input))
            .await
            .unwrap()
            .to_host::<m![{ Symbol<Shard<4>> }, { Symbol<Pieces<Small, 4>> }, Width]>(
                &mut device.pdma,
            )
            .await
            .unwrap()
            .into_vec();
        for (chip, rows) in gathered.chunks_exact(4 * rows * width).enumerate() {
            for (at, got) in rows.iter().enumerate() {
                assert_eq!(
                    *got,
                    value(at / width, at % width),
                    "chip {chip} element {at}"
                );
            }
        }
    }
}
