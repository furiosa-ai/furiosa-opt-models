use furiosa_opt_std::prelude::*;

use super::View;
use crate::load::Context;

/// A stacked leaf is one tensor per layer in one backing; a layer's view borrows its tensor.
impl<D: MaterializableScalar, C: M + 'static, S: M + 'static, const N: usize> View<N>
    for HbmTensor<D, C, S>
{
    type Storage = Slice<Self>;
    type Layer<'a> = &'a Self;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        &storage[layer]
    }

    fn stack(layers: Vec<Self>, _: &mut Context<'_>) -> Self::Storage {
        layers.into()
    }
}

/// A borrowed leaf borrows its backing, and each layer's view borrows that layer's tensor of it.
impl<'a, D: MaterializableScalar, C: M + 'static, S: M + 'static, const N: usize> View<N>
    for &'a HbmTensor<D, C, S>
{
    type Storage = &'a Slice<HbmTensor<D, C, S>>;
    type Layer<'b> = &'b HbmTensor<D, C, S>;

    fn view<'b>(storage: &'b Self::Storage, layer: usize) -> Self::Layer<'b> {
        <HbmTensor<D, C, S> as View<N>>::view(storage, layer)
    }

    fn stack(_: Vec<Self>, _: &mut Context<'_>) -> Self::Storage {
        unreachable!("a borrowed leaf borrows a loaded tree; no checkpoint load stacks it")
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{F, I, Layer, Model, checkpoint, tensors};
    use super::*;
    use crate::load::Load;
    use crate::ops::swiglu::GateUp;
    use crate::tensor::memory::Hbm;
    use crate::tests::DEVICE;

    /// The bytes of each HBM leaf of `layer`: its norm, packed gate/up and down weights.
    async fn read(layer: <Layer<Hbm> as View<2>>::Layer<'_>, device: &mut Device) -> [Vec<u8>; 3] {
        let (norm, mlp) = layer;
        let (gate_up, down) = mlp.0.into_parts();
        let norm = norm.0.to_host::<m![F]>(&mut device.pdma).await.unwrap();
        let gate_up = gate_up
            .tensor()
            .to_host::<m![{ Symbol<GateUp<I>> }, F]>(&mut device.pdma);
        let gate_up = gate_up.await.unwrap();
        let down = down
            .tensor()
            .to_host::<m![F, I]>(&mut device.pdma)
            .await
            .unwrap();
        [norm.into_vec(), gate_up.into_vec(), down.into_vec()].map(|v| bf16::to_buf(&v))
    }

    /// A stack in HBM holds every layer of a leaf in one backing, each layer's view its own
    /// values; a borrowed tree borrows the backings and views the same layers.
    #[tokio::test(flavor = "current_thread")]
    async fn loads_a_stack_into_one_backing_per_leaf() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let tensors = tensors(2);
        let model: Model<Hbm> = checkpoint(&tensors).load(&mut device).unwrap();
        let borrowed: Model<&Hbm> = Load::load(&mut Context::new(None), &model).unwrap();
        let stored = |name: &str| tensors.iter().find(|(n, ..)| n == name).unwrap().2.clone();
        for l in 0..2 {
            let at = |name: &str| stored(&format!("layers.{l}.{name}"));
            let gate_up = [at("mlp.gate_proj.weight"), at("mlp.up_proj.weight")].concat();
            let want = [at("input_layernorm"), gate_up, at("mlp.down_proj.weight")];
            assert_eq!(
                read(model.body.layers.0.layer(l), &mut device).await,
                want,
                "{l}"
            );
            let view = borrowed.body.layers.0.layer(l);
            assert_eq!(read(view, &mut device).await, want, "borrowed {l}");
        }
    }
}
