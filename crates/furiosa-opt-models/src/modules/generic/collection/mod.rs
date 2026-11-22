use furiosa_opt_std::prelude::*;

use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::tensor::memory::Memory;
use crate::tensor::memory::residency::{Leaves, Path, Residency, Visit};

/// A layer collection owning one backing per parameter kind.
#[derive(DeviceSend)]
pub struct Stacked<L: View<N>, const N: usize> {
    pub storage: L::Storage,
}

/// Construct a layer's parameter views without issuing device work; a trait, as every node of a
/// layer type rebuilds its part of a layer from the layer-major storage in its own shape.
///
/// - A layer of owned leaves is its own view; a layer of borrowed leaves borrows the storage and
///   owns its views, so a call's borrowed tree builds each layer without a temporary to point at.
/// - [`View::stack`] is the inverse of [`View::view`]: the layers a stacking load placed in one
///   backing per leaf, as that storage.
pub trait View<const N: usize>: Sized {
    type Storage;
    /// What [`View::view`] builds: `Self`, or `Self` with owned leaves when its leaves borrow.
    type Layer;
    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer;
    /// The storage whose layers `layers` are, as [`Context::layers`] loaded them.
    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage;
}

impl<L: View<N>, const N: usize> Stacked<L, N> {
    pub fn layer(&self, layer: usize) -> L::Layer {
        L::view(&self.storage, layer)
    }
}

/// The `N` layers under segments "0".."N-1", each read as one layer is, every leaf filling its
/// layer of one backing; the layers' views then stack into the storage.
impl<'a, L: View<N> + Load<&'a Checkpoint>, const N: usize> Load<&'a Checkpoint> for Stacked<L, N> {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        const { assert!(N > 0, "a stack holds a layer") };
        context.checkpoint(from, |cx| {
            cx.layers(
                |cx| L::load(cx, from),
                |layers, cx| Self {
                    storage: L::stack(layers, cx),
                },
            )
        })
    }
}

/// A borrowed stack borrows each backing, as the leaves of its storage borrow theirs.
impl<'a, L: View<N>, B: View<N>, const N: usize> Load<&'a Stacked<B, N>> for Stacked<L, N>
where
    L::Storage: Load<&'a B::Storage>,
{
    fn load(context: &mut Context<'_>, from: &'a Stacked<B, N>) -> Result<Self, LoadError> {
        L::Storage::load(context, &from.storage).map(|storage| Self { storage })
    }
}

/// Each layer is a view of the stacked parameters, built when it runs.
impl<X, Env, S, L: View<N, Layer: crate::Module<X, Env, S>>, const N: usize>
    super::Collection<X, Env, S> for Stacked<L, N>
{
    type Output = <L::Layer as crate::Module<X, Env, S>>::Output;

    const COUNT: usize = N;

    fn apply(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        hidden: X,
        index: usize,
    ) -> Self::Output {
        crate::Module::forward(&self.layer(index), device, env, state, hidden)
    }
}

/// How a collection holds its `N` layers `L`: the collection type of the kind; a trait, as a kind
/// marker names its collection type (`Array` an array, `Stack` a [`Stacked`]).
pub trait Kind<L, const N: usize> {
    type Of;
}

/// Each layer holds its own parameters: `[L; N]`.
pub struct Array;

/// The layers are views of layer-major parameters: [`Stacked`].
pub struct Stack;

impl<L, const N: usize> Kind<L, N> for Array {
    type Of = [L; N];
}

impl<L: View<N>, const N: usize> Kind<L, N> for Stack {
    type Of = Stacked<L, N>;
}

/// `N` layers `L` in the collection their kind `K` holds them in. It names the layer and the depth
/// itself, so a model alias generic over its layers' parts keeps the kind a parameter.
#[derive(DeviceSend)]
pub struct Collected<K: Kind<L, N>, L, const N: usize>(pub K::Of);

/// Array collections retain independent layer owners in every memory backend. Stacked HBM
/// backings have a different ownership contract and are not projected through this impl.
impl<L: Residency, const N: usize> Residency for Collected<Array, L, N> {
    type In<X: Memory> = Collected<Array, L::In<X>, N>;

    fn map<'t, From: Memory, To: Memory, P: Leaves<'t, From, To>>(
        tree: &'t Self::In<From>,
        leaves: &mut P,
        path: Path<'_>,
    ) -> Result<Self::In<To>, P::Error> {
        <[L; N] as Residency>::map(&tree.0, leaves, path).map(Collected::<Array, L::In<To>, N>)
    }

    fn visit<X: Memory, V: Visit<X>>(
        tree: &Self::In<X>,
        visitor: &mut V,
        path: Path<'_>,
    ) -> Result<(), V::Error> {
        <[L; N] as Residency>::visit(&tree.0, visitor, path)
    }
}

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
impl<'a, K: Kind<L, N> + Kind<B, N>, L, B, const N: usize> Load<&'a Collected<K, B, N>>
    for Collected<K, L, N>
where
    <K as Kind<L, N>>::Of: Load<&'a <K as Kind<B, N>>::Of>,
{
    fn load(context: &mut Context<'_>, from: &'a Collected<K, B, N>) -> Result<Self, LoadError> {
        <K as Kind<L, N>>::Of::load(context, &from.0).map(Self)
    }
}

/// The layers apply as their kind's collection applies them.
impl<X, Env, S, K, L, const N: usize> super::Collection<X, Env, S> for Collected<K, L, N>
where
    K: Kind<L, N, Of: super::Collection<X, Env, S>>,
{
    type Output = <K::Of as super::Collection<X, Env, S>>::Output;

    const COUNT: usize = <K::Of as super::Collection<X, Env, S>>::COUNT;

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

/// Outermost axis of a layer-major parameter tensor.
#[derive(Debug, Clone)]
pub struct Depth<const N: usize>;

impl<const N: usize> AxisName for Depth<N> {
    const NAME: Ident = Ident::new("Layers");
    const SIZE: usize = N;
}

pub type Tensor<D, C, S, const N: usize> = HbmTensor<D, C, m![{ Symbol<Depth<N>> }, { S }]>;

impl<D: MaterializableScalar, C: M, S: M, const N: usize> View<N> for HbmTensor<D, C, S> {
    type Storage = Tensor<D, C, S, N>;
    type Layer = Self;

    fn view(storage: &Self::Storage, layer: usize) -> Self {
        // SAFETY: parameters are read-only; the selected outer-axis tile retains one layer's order.
        unsafe {
            storage
                .view()
                .tile::<m![{ Symbol<Depth<N>> }], 1,
                    m![{ Symbol<Depth<N>> } = 1 # ({ Symbol<Depth<N>> }), { S }]>(layer)
                .alias::<C, S>()
        }
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        let first = layers.into_iter().next().expect("a stack holds a layer");
        Tensor::try_from(context.widen(first.into_buffer()))
            .expect("a stacking load places every layer of a leaf in one backing")
    }
}

/// A borrowed leaf borrows its backing and owns each layer's view of it.
impl<'a, D: MaterializableScalar, C: M, S: M, const N: usize> View<N> for &'a HbmTensor<D, C, S> {
    type Storage = &'a Tensor<D, C, S, N>;
    type Layer = HbmTensor<D, C, S>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        <HbmTensor<D, C, S> as View<N>>::view(storage, layer)
    }

    fn stack(_: [Self; N], _: &mut Context<'_>) -> Self::Storage {
        unreachable!("a borrowed leaf borrows a loaded tree; no checkpoint load stacks it")
    }
}

mod view;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::generic::{Collection, Layers};
    use crate::ops::attention::geometry::Hidden;
    use crate::ops::normalization::tests::random;
    use crate::ops::normalization::{RmsNorm, RowMajor, Spread};
    use crate::tensor::form::Weight;
    use crate::tensor::{axis::Sixteen, parallel::Chip};
    use crate::tests::DEVICE;
    use crate::{Module, Rolled};
    use furiosa_opt_std::runtime::launch;

    type Features = Hidden<1024>;
    type Plane = Spread<Features, Sixteen>;
    type Launch = HbmTensor<bf16, Chip, RowMajor<Features, Sixteen>>;

    /// A layer: a norm, its gain viewed from the stacked gains in a stack.
    #[derive(DeviceSend)]
    struct Layer(RmsNorm<Features>);

    impl<const N: usize> View<N> for Layer {
        type Storage = <RmsNorm<Features> as View<N>>::Storage;
        type Layer = Self;

        fn view(storage: &Self::Storage, layer: usize) -> Self {
            Self(<RmsNorm<Features> as View<N>>::view(storage, layer))
        }

        fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
            <RmsNorm<Features> as View<N>>::stack(layers.map(|Layer(norm)| norm), context)
        }
    }

    impl<Env: Copy> Module<Plane, Env, &mut ()> for Layer {
        type Output = Plane;

        fn forward(&self, device: &mut Device, env: Env, state: &mut (), rows: Plane) -> Plane {
            self.0.forward(device, env, state, &rows)
        }
    }

    // # Loading a stack

    mod load {
        use super::super::*;
        use crate::Rolled;
        use crate::load::Named;
        use crate::modules::generic::Layers;
        use crate::modules::language::Backbone;
        use crate::ops::linear::Linear;
        use crate::ops::swiglu::{GateUp, Intermediate, Mlp};
        use crate::tensor::memory::{Hbm, Memory};
        use crate::tests::DEVICE;
        use safetensors::Dtype;
        use std::marker::PhantomData;
        use std::ptr;

        axes![F = 32];
        type I = Intermediate<32>;

        /// A leaf on host holding the bytes it read; a stack's storage holds its layers' bytes in
        /// layer order.
        #[derive(Debug, PartialEq)]
        struct Leaf<D, C, S>(Vec<u8>, PhantomData<fn() -> (D, C, S)>);

        impl<D, C, S> Clone for Leaf<D, C, S> {
            fn clone(&self) -> Self {
                Leaf(self.0.clone(), PhantomData)
            }
        }

        /// The memory whose leaves are [`Leaf`]s.
        struct Bytes;

        impl Memory for Bytes {
            type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Leaf<D, C, S>;
        }

        impl<'a, D: 'static, C: 'static, S: 'static> Load<&'a Checkpoint> for Leaf<D, C, S> {
            fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
                let read = |cx: &mut Context<'_>| {
                    let name = cx.name();
                    cx.read(&name);
                    let view = from.get(&name).ok_or(LoadError::Missing(vec![name]))?;
                    Ok(Leaf(view.data().to_vec(), PhantomData))
                };
                context.checkpoint(from, |cx| {
                    cx.leaf(from, read, |_, leaf| Ok((leaf.clone(), leaf)))
                })
            }
        }

        impl<D: MaterializableScalar, C: M + 'static, S: M + 'static, L: M + 'static>
            Load<HostTensor<D, m![{ C }, { L }]>> for Leaf<D, C, S>
        {
            fn load(
                _: &mut Context<'_>,
                host: HostTensor<D, m![{ C }, { L }]>,
            ) -> Result<Self, LoadError> {
                Ok(Leaf(D::to_buf(&host.into_vec()), PhantomData))
            }
        }

        impl<'a, D, C, S> Load<&'a Leaf<D, C, S>> for &'a Leaf<D, C, S> {
            fn load(_: &mut Context<'_>, from: &'a Leaf<D, C, S>) -> Result<Self, LoadError> {
                Ok(from)
            }
        }

        impl<D, C, S, const N: usize> View<N> for Leaf<D, C, S> {
            type Storage = Self;
            type Layer = Self;

            fn view(storage: &Self, layer: usize) -> Self {
                let bytes = storage.0.len() / N;
                Leaf(storage.0[layer * bytes..][..bytes].to_vec(), PhantomData)
            }

            fn stack(layers: [Self; N], _: &mut Context<'_>) -> Self {
                Leaf(
                    layers.into_iter().flat_map(|layer| layer.0).collect(),
                    PhantomData,
                )
            }
        }

        impl<'a, D, C, S, const N: usize> View<N> for &'a Leaf<D, C, S> {
            type Storage = &'a Leaf<D, C, S>;
            type Layer = Leaf<D, C, S>;

            fn view(storage: &Self::Storage, layer: usize) -> Leaf<D, C, S> {
                <Leaf<D, C, S> as View<N>>::view(storage, layer)
            }

            fn stack(_: [Self; N], _: &mut Context<'_>) -> Self::Storage {
                unreachable!("a borrowed leaf is not loaded from a checkpoint")
            }
        }

        type Matrix<X, O, In> = <X as Memory>::Tensor<bf16, m![1], m![O, In]>;

        /// A layer: a norm's gain and a gated MLP, named as a checkpoint's decoder layer.
        type Layer<X> = (
            Named<"input_layernorm", <X as Memory>::Tensor<bf16, m![1], m![F]>>,
            Named<
                "mlp",
                Mlp<
                    F,
                    I,
                    Linear<F, GateUp<I>, m![1], Matrix<X, GateUp<I>, F>>,
                    Linear<I, F, m![1], Matrix<X, F, I>>,
                >,
            >,
        );

        type Model<X> = Backbone<
            Named<"embed_tokens", <X as Memory>::Tensor<bf16, m![1], m![F]>>,
            Layers<Collected<Stack, Layer<X>, 2>, Rolled>,
        >;

        /// BF16 bytes of `count` elements valued `value` plus a run of 0 to 6.
        fn values(count: usize, value: f32) -> Vec<u8> {
            (0..count)
                .flat_map(|index| half::bf16::from_f32(value + (index % 7) as f32).to_le_bytes())
                .collect()
        }

        /// The tensors of a two-layer model, layer `l`'s values starting at `10 * (l + 1)`.
        fn tensors() -> Vec<(String, Vec<usize>, Vec<u8>)> {
            let layer = |l: usize| {
                let at = 10.0 * (l + 1) as f32;
                let (f, i) = (F::SIZE, I::SIZE);
                [
                    ("input_layernorm", vec![f], values(f, at)),
                    ("mlp.gate_proj.weight", vec![i, f], values(i * f, at + 1.0)),
                    ("mlp.up_proj.weight", vec![i, f], values(i * f, at + 2.0)),
                    ("mlp.down_proj.weight", vec![f, i], values(f * i, at + 3.0)),
                ]
                .map(|(name, shape, data)| (format!("layers.{l}.{name}"), shape, data))
            };
            let embedding = (
                "embed_tokens".to_string(),
                vec![F::SIZE],
                values(F::SIZE, 0.0),
            );
            [vec![embedding], layer(0).to_vec(), layer(1).to_vec()].concat()
        }

        fn checkpoint(tensors: &[(String, Vec<usize>, Vec<u8>)]) -> Checkpoint {
            let mut checkpoint = Checkpoint::default();
            for (name, shape, data) in tensors {
                checkpoint
                    .insert(name, Dtype::BF16, shape, data.clone())
                    .unwrap();
            }
            checkpoint
        }

        fn load(source: &Checkpoint) -> Result<Model<Bytes>, LoadError> {
            Model::<Bytes>::load(&mut Context::new(None), source)
        }

        /// The norm and the packed gate/up of `layer`.
        fn parts(layer: &<Layer<Bytes> as View<2>>::Layer) -> (Vec<u8>, Vec<u8>) {
            let (norm, mlp) = layer;
            (norm.0.0.clone(), mlp.0.parts().0.affine.weight.0.clone())
        }

        /// A stack reads each layer under its global index in the body's `layers`, and its views
        /// give each layer its own values: the gate rows, then the up rows, of the layer's MLP.
        #[test]
        fn stacks_each_layer_under_its_index() {
            let tensors = tensors();
            let model = load(&checkpoint(&tensors)).unwrap();
            let stack = &model.body.layers.0;
            let stored = |name: &str| tensors.iter().find(|(n, ..)| n == name).unwrap().2.clone();
            for l in 0..2 {
                let (norm, gate_up) = parts(&stack.layer(l));
                assert_eq!(norm, stored(&format!("layers.{l}.input_layernorm")));
                let gate = stored(&format!("layers.{l}.mlp.gate_proj.weight"));
                let up = stored(&format!("layers.{l}.mlp.up_proj.weight"));
                assert_eq!(gate_up, [gate, up].concat(), "layer {l}");
            }
            assert_ne!(parts(&stack.layer(0)), parts(&stack.layer(1)));
        }

        /// A stack fails with every absent name of every layer, and with a name past its last layer.
        #[test]
        fn stack_reads_strictly() {
            let mut tensors = tensors();
            tensors.retain(|(name, ..)| {
                !["layers.0.mlp.up_proj.weight", "layers.1.input_layernorm"]
                    .contains(&name.as_str())
            });
            match load(&checkpoint(&tensors)).err() {
                Some(LoadError::Missing(names)) => assert_eq!(
                    names,
                    ["layers.0.mlp.up_proj.weight", "layers.1.input_layernorm"]
                ),
                other => panic!("{other:?} is not `Missing`"),
            }
            let mut tensors = self::tensors();
            let extra = (
                "layers.2.input_layernorm".into(),
                vec![F::SIZE],
                values(F::SIZE, 0.0),
            );
            tensors.push(extra);
            match load(&checkpoint(&tensors)).err() {
                Some(LoadError::Unexpected(names)) => {
                    assert_eq!(names, ["layers.2.input_layernorm"])
                }
                other => panic!("{other:?} is not `Unexpected`"),
            }
        }

        /// A borrowed stack borrows the loaded storage and builds each layer as the owned stack does.
        #[test]
        fn borrowed_stack_borrows_the_storage() {
            let model = load(&checkpoint(&tensors())).unwrap();
            let borrowed: Model<&Bytes> = Load::load(&mut Context::new(None), &model).unwrap();
            let (owned, stack) = (&model.body.layers.0, &borrowed.body.layers.0);
            assert!(ptr::eq(stack.storage.0, &owned.storage.0));
            for l in 0..2 {
                assert_eq!(parts(&stack.layer(l)), parts(&owned.layer(l)));
            }
        }

        /// The bytes of each HBM leaf of `layer`: its norm, packed gate/up and down weights.
        async fn read(layer: <Layer<Hbm> as View<2>>::Layer, device: &mut Device) -> [Vec<u8>; 3] {
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
            let tensors = tensors();
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

    async fn stack(device: &mut Device) -> Stacked<Layer, 2> {
        Stacked {
            storage: tensor(device, 2 * Features::SIZE, 3).await,
        }
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
        let layers = Collected::<Stack, _, 2>(stack(&mut device).await);
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

#[cfg(test)]
mod residency_tests {
    use super::{Array, Collected};
    use crate::load::{Checkpoint, Context, Load, LoadError, Named};
    use crate::modules::residual::Residual;
    use crate::ops::linear::grouped::{Grouped, Layout};
    use crate::ops::normalization::{EPSILON, RmsNorm};
    use crate::ops::pointwise::add::Add;
    use crate::tensor::form::Weight;
    use crate::tensor::memory::residency::{Leaves, Path, Residency, Visit};
    use crate::tensor::memory::{Hbm, Memory};
    use crate::tensor::parallel::Shard;
    use crate::{Layers, Sequential, Then};
    use furiosa_opt_std::prelude::*;
    use safetensors::Dtype;
    use std::{convert::Infallible, marker::PhantomData, sync::Arc};

    axes![Width = 128, Wide = 512];
    type Chips = m![{ Symbol<Shard<4>> }];
    type Gain<X> = Weight<m![Width], Chips, <X as Memory>::Tensor<bf16, Chips, m![Width]>>;
    type Norm<X> = RmsNorm<Width, EPSILON, Chips, Gain<X>>;
    type Projection<I, O, X> =
        Grouped<I, O, Chips, <X as Memory>::Tensor<bf16, Chips, Layout<I, O>>>;
    type Up<X> = Projection<Width, Wide, X>;
    type Down<X> = Projection<Wide, Width, X>;
    type Block<X> =
        Residual<Named<"norm", Norm<X>>, Then<Named<"up", Up<X>>, Named<"down", Down<X>>>, Add>;
    type Network<X> = Sequential<Named<"input", Norm<X>>, Layers<Collected<Array, Block<X>, 2>>>;

    #[derive(Load, Residency)]
    #[residency(memory(X))]
    struct Tree<X: Memory> {
        body: Network<X>,
        #[load(tied = "body.0.up")]
        tied: Up<X>,
    }

    /// Test storage retains the public loader's host result; it performs no layout conversion.
    struct Host;
    struct Leaf<D, C, S> {
        bytes: Arc<HostBuffer>,
        form: PhantomData<(D, C, S)>,
    }

    impl Memory for Host {
        type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Leaf<D, C, S>;
    }

    impl<D, C, S> Clone for Leaf<D, C, S> {
        fn clone(&self) -> Self {
            Self {
                bytes: self.bytes.clone(),
                form: PhantomData,
            }
        }
    }

    impl<D, C, S> Leaf<D, C, S> {
        fn split(_: &Context<'_>, leaf: Self) -> Result<(Self, Self), LoadError> {
            Ok((leaf.clone(), leaf))
        }
    }

    impl<'a, D: MaterializableScalar, C: M + 'static, S: M + 'static> Load<&'a Checkpoint>
        for Leaf<D, C, S>
    {
        fn load(cx: &mut Context<'_>, source: &'a Checkpoint) -> Result<Self, LoadError> {
            cx.checkpoint(source, |cx| {
                cx.leaf(
                    source,
                    |cx| {
                        let host = source.host::<D, S, m![{ C }, { S }]>(cx)?;
                        Ok(Self {
                            bytes: Arc::new(host.into_buffer()),
                            form: PhantomData,
                        })
                    },
                    Self::split,
                )
            })
        }
    }

    impl<D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M> Load<HostTensor<D, E>>
        for Leaf<D, C, S>
    {
        fn load(cx: &mut Context<'_>, host: HostTensor<D, E>) -> Result<Self, LoadError> {
            cx.share(
                |_| {
                    Ok(Self {
                        bytes: Arc::new(host.into_buffer()),
                        form: PhantomData,
                    })
                },
                Self::split,
            )
        }
    }

    struct Share;
    impl<'t> Leaves<'t, Host, Host> for Share {
        type Error = Infallible;

        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &'t Leaf<D, C, S>,
            _: Path<'_>,
        ) -> Result<Leaf<D, C, S>, Infallible> {
            Ok(leaf.clone())
        }
    }

    struct Borrow;
    impl<'t> Leaves<'t, Host, &'t Host> for Borrow {
        type Error = Infallible;

        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &'t Leaf<D, C, S>,
            _: Path<'_>,
        ) -> Result<&'t Leaf<D, C, S>, Infallible> {
            Ok(leaf)
        }
    }

    struct Seen {
        path: String,
        leaf: usize,
        bytes: Arc<HostBuffer>,
    }

    #[derive(Default)]
    struct Observe(Vec<Seen>);

    impl Observe {
        fn record<D, C, S>(&mut self, leaf: &Leaf<D, C, S>, path: Path<'_>) {
            self.0.push(Seen {
                path: path.to_string(),
                leaf: std::ptr::from_ref(leaf).addr(),
                bytes: leaf.bytes.clone(),
            });
        }
    }

    impl Visit<Host> for Observe {
        type Error = Infallible;

        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &Leaf<D, C, S>,
            path: Path<'_>,
        ) -> Result<(), Infallible> {
            self.record(leaf, path);
            Ok(())
        }
    }

    impl<'a> Visit<&'a Host> for Observe {
        type Error = Infallible;

        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &&'a Leaf<D, C, S>,
            path: Path<'_>,
        ) -> Result<(), Infallible> {
            self.record(*leaf, path);
            Ok(())
        }
    }

    /// Loads real numerical modules, traverses and shares their array-backed tree, and lends
    /// that tree without an HBM allocation.
    #[test]
    fn actual_array_layers_load_map_and_borrow_with_strict_ties() {
        let mut source = Checkpoint::default();
        let norm = bf16::from_f32(1.25)
            .to_bits()
            .to_le_bytes()
            .repeat(Width::SIZE);
        source
            .insert(
                "body.input.weight",
                Dtype::BF16,
                &[Width::SIZE],
                norm.clone(),
            )
            .unwrap();
        for layer in 0..2 {
            source
                .insert(
                    &format!("body.{layer}.norm.weight"),
                    Dtype::BF16,
                    &[Width::SIZE],
                    norm.clone(),
                )
                .unwrap();
            for (name, shape) in [
                ("up", [Wide::SIZE, Width::SIZE]),
                ("down", [Width::SIZE, Wide::SIZE]),
            ] {
                let bytes = (0..shape.iter().product())
                    .flat_map(|i: usize| {
                        bf16::from_f32((i % 47) as f32 / 32.0 - 0.5)
                            .to_bits()
                            .to_le_bytes()
                    })
                    .collect();
                source
                    .insert(
                        &format!("body.{layer}.{name}.weight"),
                        Dtype::BF16,
                        &shape,
                        bytes,
                    )
                    .unwrap();
            }
        }
        let load = |source: &Checkpoint| Tree::<Host>::load(&mut Context::new(None), source);
        let tree = load(&source).unwrap();
        let mut before = Observe::default();
        Tree::<Hbm>::visit(&tree, &mut before, Path::ROOT).unwrap();
        assert_eq!(
            before
                .0
                .iter()
                .map(|seen| seen.path.as_str())
                .collect::<Vec<_>>(),
            [
                "body.0.input.weight.tensor",
                "body.1.layers.0.0.norm.weight.tensor",
                "body.1.layers.0.1.0.up.weight",
                "body.1.layers.0.1.1.down.weight",
                "body.1.layers.1.0.norm.weight.tensor",
                "body.1.layers.1.1.0.up.weight",
                "body.1.layers.1.1.1.down.weight",
                "tied.weight",
            ]
        );
        assert!(Arc::ptr_eq(&before.0[2].bytes, &before.0[7].bytes));
        for (first, second) in before.0[1..4].iter().zip(&before.0[4..7]) {
            assert!(
                !Arc::ptr_eq(&first.bytes, &second.bytes),
                "layers own independent copies"
            );
            assert_eq!(&**first.bytes, &**second.bytes);
        }

        let shared: Tree<Host> = Tree::<Hbm>::map(&tree, &mut Share, Path::ROOT).unwrap();
        let mut owned = Observe::default();
        Tree::<Hbm>::visit(&shared, &mut owned, Path::ROOT).unwrap();
        for (first, second) in before.0.iter().zip(&owned.0) {
            assert_eq!(first.path, second.path);
            assert!(Arc::ptr_eq(&first.bytes, &second.bytes));
        }
        drop(tree);
        let borrowed: Tree<&Host> = Tree::<Hbm>::map(&shared, &mut Borrow, Path::ROOT).unwrap();
        let mut lent = Observe::default();
        Tree::<Hbm>::visit(&borrowed, &mut lent, Path::ROOT).unwrap();
        for (first, second) in owned.0.iter().zip(&lent.0) {
            assert_eq!(first.path, second.path);
            assert_eq!(
                first.leaf, second.leaf,
                "borrowed leaves point into the owning tree"
            );
            assert!(Arc::ptr_eq(&first.bytes, &second.bytes));
        }

        let mut invalid = source.read();
        invalid.remove("body.1.down.weight");
        assert!(
            matches!(load(&invalid).err(), Some(LoadError::Missing(names)) if names == ["body.1.down.weight"])
        );
        let mut invalid = source.read();
        invalid
            .insert("extra", Dtype::BF16, &[1], vec![0; 2])
            .unwrap();
        assert!(
            matches!(load(&invalid).err(), Some(LoadError::Unexpected(names)) if names == ["extra"])
        );

        let stored = source.get("body.0.up.weight").unwrap().data().to_vec();
        source
            .insert(
                "tied.weight",
                Dtype::BF16,
                &[Wide::SIZE, Width::SIZE],
                stored.clone(),
            )
            .unwrap();
        let matching = load(&source).expect("a matching stored copy is a valid tie");
        let mut seen = Observe::default();
        Tree::<Hbm>::visit(&matching, &mut seen, Path::ROOT).unwrap();
        assert!(Arc::ptr_eq(&seen.0[2].bytes, &seen.0[7].bytes));
        source.remove("tied.weight");
        let mut conflicting = stored;
        conflicting[0] ^= 1;
        source
            .insert(
                "tied.weight",
                Dtype::BF16,
                &[Wide::SIZE, Width::SIZE],
                conflicting,
            )
            .unwrap();
        assert!(
            matches!(load(&source).err(), Some(LoadError::InvalidData { name, reason })
            if name == "tied.weight" && reason == "tied copy differs from body.0.up.weight")
        );
    }
}
