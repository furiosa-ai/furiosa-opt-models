use furiosa_opt_std::prelude::*;

use super::{Extent, Interval, Stacked, View};
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::modules::generic::{Collection, Full};

/// How a collection holds its `N` layers `L`: the collection type of the kind; a trait, as a kind
/// marker names its collection type (`Array` an array, `Stack` a [`Stacked`]).
pub trait Kind<L, const N: usize> {
    type Of;
}

/// Each layer holds its own parameters: `[L; N]`.
pub struct Array;

/// The layers are views of layer-major parameters: [`Stacked`].
pub struct Stack<I = Interval<usize>>(std::marker::PhantomData<I>);

impl<L, const N: usize> Kind<L, N> for Array {
    type Of = [L; N];
}

impl<L: View<N>, const N: usize, I> Kind<L, N> for Stack<I> {
    type Of = Stacked<L, N, I>;
}

/// `N` layers `L` in the collection their kind `K` holds them in. It names the layer and the depth
/// itself, so a model alias generic over its layers' parts keeps the kind a parameter.
#[derive(DeviceSend)]
pub struct Collected<K: Kind<L, N>, L, const N: usize>(pub K::Of);

/// The live layers of a stack are its interval's.
impl<L: View<N>, const N: usize, I: Extent<N>> Collected<Stack<I>, L, N> {
    pub fn origin(&self) -> usize {
        self.0.origin()
    }

    pub fn count(&self) -> usize {
        self.0.count()
    }
}

impl<K: Kind<L, N, Of: Full>, L, const N: usize> Full for Collected<K, L, N> {}

impl<K: Kind<L, N, Of: Clone>, L, const N: usize> Clone for Collected<K, L, N> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// The layers load as their kind's collection does, adding no segment.
impl<'a, K: Kind<L, N, Of: Load<&'a Checkpoint>>, L, const N: usize> Load<&'a Checkpoint>
    for Collected<K, L, N>
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        K::Of::load(context, from).map(Self)
    }
}

/// A borrowed collection borrows as its kind's collection does.
impl<'a, K: Kind<L, N>, J: Kind<B, N>, L, B, const N: usize> Load<&'a Collected<J, B, N>>
    for Collected<K, L, N>
where
    <K as Kind<L, N>>::Of: Load<&'a <J as Kind<B, N>>::Of>,
{
    fn load(context: &mut Context<'_>, from: &'a Collected<J, B, N>) -> Result<Self, LoadError> {
        <K as Kind<L, N>>::Of::load(context, &from.0).map(Self)
    }
}

/// The layers apply as their kind's collection applies them.
impl<X, Env, S, K, L, const N: usize> Collection<X, Env, S> for Collected<K, L, N>
where
    K: Kind<L, N, Of: Collection<X, Env, S>>,
{
    type Output = <K::Of as Collection<X, Env, S>>::Output;

    const COUNT: usize = <K::Of as Collection<X, Env, S>>::COUNT;

    fn count(&self) -> usize {
        self.0.count()
    }

    fn apply(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        hidden: X,
        index: usize,
    ) -> Self::Output {
        self.0.apply(device, env, state, hidden, index)
    }
}

#[cfg(test)]
mod tests {
    use super::super::stacked::tests::stacked;
    use super::super::tests::{Bytes, Model, parts, selected, tensors};
    use super::*;
    use crate::modules::generic::Layers;
    use crate::ops::attention::geometry::Hidden;
    use crate::ops::normalization::tests::random;
    use crate::ops::normalization::{RmsNorm, RowMajor, Spread};
    use crate::tensor::form::Weight;
    use crate::tensor::{axis::Sixteen, parallel::Chip, scalar::Value};
    use crate::tests::DEVICE;
    use crate::{Module, Rolled};
    use furiosa_opt_std::runtime::launch;

    type Features = Hidden<1024>;
    type Plane = Spread<Features, Sixteen>;
    type Launch = HbmTensor<bf16, Chip, RowMajor<Features, Sixteen>>;

    /// A layer: a norm, its gain viewed from the stacked gains in a stack.
    #[derive(DeviceSend)]
    struct Layer<T = RmsNorm<Features>>(T);

    impl<const N: usize> View<N> for Layer {
        type Storage = <RmsNorm<Features> as View<N>>::Storage;
        type Layer<'a> = Layer<<RmsNorm<Features> as View<N>>::Layer<'a>>;

        fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
            Layer(<RmsNorm<Features> as View<N>>::view(storage, layer))
        }

        fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
            <RmsNorm<Features> as View<N>>::stack(
                layers.into_iter().map(|Layer(norm)| norm).collect(),
                context,
            )
        }
    }

    impl<Env: Copy, T> Module<Plane, Env, &mut ()> for Layer<T>
    where
        T: for<'a, 's> Module<&'a Plane, Env, &'s mut (), Output = Plane>,
    {
        type Output = Plane;

        fn forward(&self, device: &mut Device, env: Env, state: &mut (), rows: Plane) -> Plane {
            self.0.forward(device, env, state, &rows)
        }
    }

    /// A collection of a stack's selected layers keeps the stack's interval and each selected
    /// layer's values, owned and borrowed: the second and third of four layers.
    #[test]
    fn selected_stack_keeps_its_interval_and_layers() {
        let tensors = tensors(4);
        let config = serde_json::json!({"text_config": {"num_hidden_layers": 4}});
        let source = selected(&tensors, config, 1..3);
        let model = Model::<Bytes, 4>::load(&mut Context::new(None), &source).unwrap();
        let borrowed: Model<&Bytes, 4> = Load::load(&mut Context::new(None), &model).unwrap();
        let (owned, borrowed) = (&model.body.layers, &borrowed.body.layers);
        assert_eq!((owned.origin(), owned.count()), (1, 2));
        assert_eq!((borrowed.origin(), borrowed.count()), (1, 2));
        let stored = |name: &str| tensors.iter().find(|(n, ..)| n == name).unwrap().2.clone();
        for (local, global) in [(0, 1), (1, 2)] {
            let at = |name: &str| stored(&format!("layers.{global}.{name}"));
            let gate_up = [at("mlp.gate_proj.weight"), at("mlp.up_proj.weight")].concat();
            let want = (at("input_layernorm"), gate_up);
            assert_eq!(parts(&owned.0.layer(local)), want, "owned {global}");
            assert_eq!(parts(&borrowed.0.layer(local)), want, "borrowed {global}");
        }
    }

    #[device(chip = 1)]
    fn rolled<C>(device: &mut Device, layers: &Layers<C, Rolled>, input: &Launch) -> Launch
    where
        C: DeviceSend + for<'s> Collection<Plane, (), &'s mut (), Output = Plane>,
    {
        let rows: Plane = input.to_dm(&mut device.tdma);
        let rows = layers.forward(&mut *device, (), &mut [(); 2], rows);
        rows.to_hbm(&mut device.tdma)
    }

    async fn tensor<S: M>(
        device: &mut Device,
        count: usize,
        seed: u64,
    ) -> HbmTensor<bf16, Chip, S> {
        let values = random(count, seed);
        HostTensor::<bf16, S>::from_vec(values)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap()
    }

    async fn array(device: &mut Device) -> [Layer; 2] {
        let first = tensor(device, Features::SIZE, 1).await;
        let second = tensor(device, Features::SIZE, 2).await;
        [first, second].map(|gain| Layer(RmsNorm::new(Weight::new(gain))))
    }

    async fn stack(device: &mut Device) -> Stacked<Layer, 2, Interval<Value>> {
        let storage = tensor(device, 2 * Features::SIZE, 3).await;
        let storage: HbmTensor<bf16, Chip, m![{ Symbol<super::super::Depth<2>> }, Features]> =
            storage;
        let storage = crate::tensor::memory::blocks(storage.into_buffer(), Features::SIZE * 2);
        let interval = Load::load(&mut Context::new(device), &Interval::new(0..2, 2).unwrap());
        stacked::<Layer, 2, Interval<Value>>(storage.unwrap().into(), interval.unwrap())
    }

    async fn rows(output: Launch, device: &mut Device) -> Vec<u16> {
        let rows = output.to_host::<RowMajor<Features, Sixteen>>(&mut device.pdma);
        rows.await
            .unwrap()
            .into_vec()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    /// A collection in its kind's newtype is launched and applied as the bare collection: a rolled
    /// loop over two layers computes the same rows from either.
    #[tokio::test(flavor = "current_thread")]
    async fn collected_layers_compute_what_bare_ones_do() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(rolled.topology()).unwrap();
        let input: Launch = tensor(&mut device, Sixteen::SIZE * Features::SIZE, 4).await;
        let layers = Layers {
            layers: array(&mut device).await,
            kind: Rolled,
        };
        let bare = launch(rolled, (&mut device, &layers, &input))
            .await
            .unwrap();
        let layers = Collected::<Array, _, 2>(array(&mut device).await);
        let layers = Layers {
            layers,
            kind: Rolled,
        };
        let collected = launch(rolled, (&mut device, &layers, &input))
            .await
            .unwrap();
        let want = rows(bare, &mut device).await;
        assert_eq!(want, rows(collected, &mut device).await, "rolled array");

        let layers = Layers {
            layers: stack(&mut device).await,
            kind: Rolled,
        };
        let bare = launch(rolled, (&mut device, &layers, &input))
            .await
            .unwrap();
        let layers = Collected::<Stack<Interval<Value>>, _, 2>(stack(&mut device).await);
        let layers = Layers {
            layers,
            kind: Rolled,
        };
        let collected = launch(rolled, (&mut device, &layers, &input))
            .await
            .unwrap();
        let want = rows(bare, &mut device).await;
        assert_eq!(want, rows(collected, &mut device).await, "rolled stack");

        assert!(want.iter().any(|&bits| bits != 0));
    }
}
