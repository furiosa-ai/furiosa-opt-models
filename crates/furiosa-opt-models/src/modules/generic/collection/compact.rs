use super::View;
use crate::load::Context;
use furiosa_opt_std::prelude::*;

/// A borrowed compact backing gives each layer its existing tensor, with no full-depth alias.
impl<'a, D: Scalar, C: M + 'static, S: M + 'static, const N: usize> View<N>
    for &'a Slice<HbmTensor<D, C, S>>
{
    type Storage = Self;
    type Layer<'b> = &'b HbmTensor<D, C, S>;

    fn view<'b>(storage: &'b Self::Storage, layer: usize) -> Self::Layer<'b> {
        &(*storage)[layer]
    }

    fn stack(_: Vec<Self>, _: &mut Context<'_>) -> Self::Storage {
        unreachable!("a borrowed backing is not stacked from a checkpoint")
    }
}

#[cfg(test)]
mod tests {
    use super::super::stacked::tests::stacked;
    use super::*;
    use crate::modules::generic::collection::{Interval, Stacked};
    use crate::tensor::mapping::{BroadcastClusters, BroadcastSlices};
    use crate::tensor::{memory, parallel::Chip, scalar::Value};
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![Cells = 128];
    type Layer = HbmTensor<i32, Chip, m![Cells]>;

    /// An empty interval is invalid before any checkpoint weight is read.
    #[test]
    fn rejects_empty_interval() {
        refuse(2..2);
    }

    /// A reversed interval is invalid before any checkpoint weight is read.
    #[test]
    fn rejects_reversed_interval() {
        refuse(std::ops::Range { start: 3, end: 1 });
    }

    /// An interval beyond configured depth is invalid before any weight is read.
    #[test]
    fn rejects_excess_interval() {
        refuse(1..5);
    }

    fn refuse(range: std::ops::Range<usize>) {
        use crate::load::{Checkpoint, LoadError, Part};

        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            br#"{"num_hidden_layers":4}"#,
        )
        .unwrap();
        let header = br#"{"0":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}"#;
        let length = u64::try_from(header.len()).unwrap().to_le_bytes();
        std::fs::write(
            directory.path().join("model.safetensors"),
            [&length[..], header].concat(),
        )
        .unwrap();
        let source = Checkpoint::declared(directory.path()).unwrap();
        let refused = source.selected(Part::Layers(range)).err();
        assert!(matches!(refused, Some(LoadError::InvalidData { name, .. }) if name == "layers"));
    }

    #[device(chip = 1)]
    fn copy_layers(
        device: &mut Device,
        layers: &Stacked<&Slice<Layer>, 4, Interval<Value>>,
        output: &mut Slice<Layer>,
    ) {
        let count = layers.count();
        let mut rows: DmTensor<i32, Chip, BroadcastClusters, BroadcastSlices, m![Cells]> =
            layers.layer(0).to_dm(&mut device.tdma);
        for index in 0..count - 1 {
            rows.view()
                .to_hbm_view(&mut device.tdma, output[index].view_mut());
            layers
                .layer(index + 1)
                .view()
                .to_dm_view(&mut device.tdma, rows.view_mut());
        }
        rows.view()
            .to_hbm_view(&mut device.tdma, output[count - 1].view_mut());
    }

    /// A runtime-depth first/pair/last loop returns every compact layer's host values in order.
    #[tokio::test(flavor = "current_thread")]
    async fn copies_live_layers() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(copy_layers.topology()).unwrap();
        for count in [1, 3, 4] {
            let expected: Vec<i32> = (0..count * Cells::SIZE)
                .map(|cell| i32::try_from(1000 + cell).unwrap())
                .collect();
            let buffer = HbmBuffer::alloc(&device, expected.len() * size_of::<i32>()).unwrap();
            let mut storage: Slice<Layer> = memory::blocks(buffer, Layer::size()).unwrap().into();
            for (layer, values) in storage.iter_mut().zip(expected.chunks_exact(Cells::SIZE)) {
                HostTensor::<i32, m![Cells]>::from_vec(values.iter().copied())
                    .to_hbm(&mut device.pdma)
                    .output(layer)
                    .await
                    .unwrap();
            }
            let buffer = HbmBuffer::alloc(&device, expected.len() * size_of::<i32>()).unwrap();
            let mut output: Slice<Layer> = memory::blocks(buffer, Layer::size()).unwrap().into();
            let interval = Interval::new(0..count, 4).unwrap();
            let interval = crate::load::Load::load(&mut Context::new(&mut device), &interval);
            let layers = stacked::<&Slice<Layer>, 4, Interval<Value>>(&storage, interval.unwrap());
            launch(copy_layers, (&mut device, &layers, &mut output))
                .await
                .unwrap();
            for (layer, values) in output.iter().zip(expected.chunks_exact(Cells::SIZE)) {
                assert_eq!(
                    layer
                        .to_host::<m![Cells]>(&mut device.pdma)
                        .await
                        .unwrap()
                        .into_vec(),
                    values
                );
            }
        }
    }
}
