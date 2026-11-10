//! The layers of a model in one collection, applied by index.
//!
//! - A [`Collected`] names its layer and depth and holds them as its [`Kind`] says: an [`Array`] of
//!   layers, or a [`Stack`], whose [`Stacked`] layers are views of layer-major storage.
//! - A [`View`] builds one layer of a stack from shared borrows of the storage; the stack's
//!   [`Interval`] is the live layers among its `N`.

use furiosa_opt_std::prelude::*;

use crate::load::Context;

pub use collected::{Array, Collected, Kind, Stack};
pub use interval::{Extent, Interval};
pub use stacked::Stacked;

mod collected;
mod compact;
mod interval;
mod leaf;
mod stacked;
mod view;

/// Construct a layer's parameter views without issuing device work; a trait, as every node of a
/// layer type rebuilds its part of a layer from the layer-major storage in its own shape.
///
/// - A layer's view borrows the storage: its leaves are shared borrows of that layer's tensors,
///   live while the storage is borrowed, whether the tree's leaves own or borrow their backings.
/// - [`View::stack`] is the inverse of [`View::view`]: the layers a stacking load placed in one
///   backing per leaf, as that storage.
pub trait View<const N: usize>: Sized {
    type Storage;
    /// What [`View::view`] builds: `Self` with each leaf borrowed from the storage for `'a`. No
    /// `Self: 'a` bound: a layer type is well formed at every `'a`, so a `for<'a>` bound on a
    /// borrowed tree's layers holds for leaves of any lifetime.
    type Layer<'a>;
    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a>;
    /// The storage whose layers `layers` are, as [`Context::layers`] loaded them.
    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage;
}

/// Outermost axis of a layer-major parameter tensor.
#[derive(Debug, Clone)]
pub struct Depth<const N: usize>;

impl<const N: usize> AxisName for Depth<N> {
    const NAME: Ident = Ident::new("Layers");
    const SIZE: usize = N;
}

/// A model of host leaves holding the bytes they read, which the collection's tests load.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rolled;
    use crate::load::{Checkpoint, Load, LoadError, Named, Part};
    use crate::modules::generic::Layers;
    use crate::modules::language::Backbone;
    use crate::ops::linear::Linear;
    use crate::ops::swiglu::{GateUp, Intermediate, Mlp};
    use crate::tensor::memory::Memory;
    use safetensors::Dtype;
    use std::marker::PhantomData;
    use std::ops::Range;

    axes![F = 32];
    pub(super) type I = Intermediate<32>;

    /// A leaf on host holding the bytes it read; a stack's storage holds its layers' leaves in
    /// layer order.
    #[derive(Debug, PartialEq)]
    pub(super) struct Leaf<D, C, S>(Vec<u8>, PhantomData<fn() -> (D, C, S)>);

    impl<D, C, S> Clone for Leaf<D, C, S> {
        fn clone(&self) -> Self {
            Leaf(self.0.clone(), PhantomData)
        }
    }

    /// The memory whose leaves are [`Leaf`]s.
    pub(super) struct Bytes;

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

    impl<D: 'static, C: 'static, S: 'static, const N: usize> View<N> for Leaf<D, C, S> {
        type Storage = Vec<Self>;
        type Layer<'a> = &'a Self;

        fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
            &storage[layer]
        }

        fn stack(layers: Vec<Self>, _: &mut Context<'_>) -> Self::Storage {
            layers
        }
    }

    impl<'a, D: 'static, C: 'static, S: 'static, const N: usize> View<N> for &'a Leaf<D, C, S> {
        type Storage = &'a Vec<Leaf<D, C, S>>;
        type Layer<'b> = &'b Leaf<D, C, S>;

        fn view<'b>(storage: &'b Self::Storage, layer: usize) -> Self::Layer<'b> {
            <Leaf<D, C, S> as View<N>>::view(storage, layer)
        }

        fn stack(_: Vec<Self>, _: &mut Context<'_>) -> Self::Storage {
            unreachable!("a borrowed leaf is not loaded from a checkpoint")
        }
    }

    impl<'a, D, C, S> Load<&'a Vec<Leaf<D, C, S>>> for &'a Vec<Leaf<D, C, S>> {
        fn load(_: &mut Context<'_>, from: &'a Vec<Leaf<D, C, S>>) -> Result<Self, LoadError> {
            Ok(from)
        }
    }

    type Matrix<X, O, In> = <X as Memory>::Tensor<bf16, m![1], m![O, In]>;

    /// A layer: a norm's gain and a gated MLP, named as a checkpoint's decoder layer.
    pub(super) type Layer<X> = (
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

    pub(super) type Model<X, const N: usize = 2> = Backbone<
        Named<"embed_tokens", <X as Memory>::Tensor<bf16, m![1], m![F]>>,
        Layers<Collected<Stack, Layer<X>, N>, Rolled>,
    >;

    /// BF16 bytes of `count` elements valued `value` plus a run of 0 to 6.
    pub(super) fn values(count: usize, value: f32) -> Vec<u8> {
        (0..count)
            .flat_map(|index| half::bf16::from_f32(value + (index % 7) as f32).to_le_bytes())
            .collect()
    }

    /// The tensors of a model of `depth` layers, at most four, layer `l`'s values starting at
    /// 11, 23, 37 or 53.
    pub(super) fn tensors(depth: usize) -> Vec<(String, Vec<usize>, Vec<u8>)> {
        let layer = |l: usize| {
            let at = [11.0, 23.0, 37.0, 53.0][l];
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
        std::iter::once(embedding)
            .chain((0..depth).flat_map(layer))
            .collect()
    }

    /// A whole checkpoint of `tensors`, without a config.
    pub(super) fn checkpoint(tensors: &[(String, Vec<usize>, Vec<u8>)]) -> Checkpoint {
        let mut checkpoint = Checkpoint::default();
        for (name, shape, data) in tensors {
            checkpoint
                .insert(name, Dtype::BF16, shape, data.clone())
                .unwrap();
        }
        checkpoint
    }

    /// A checkpoint of `tensors` under `config`, which a load reads for the layers `range` selects.
    pub(super) fn selected(
        tensors: &[(String, Vec<usize>, Vec<u8>)],
        config: serde_json::Value,
        range: Range<usize>,
    ) -> Checkpoint {
        let directory = tempfile::tempdir().unwrap();
        let mut source = Checkpoint::new(directory.path(), Some(config)).unwrap();
        for (name, shape, data) in tensors {
            source
                .insert(name, Dtype::BF16, shape, data.clone())
                .unwrap();
        }
        source.selected(Part::Layers(range)).unwrap()
    }

    pub(super) fn load(source: &Checkpoint) -> Result<Model<Bytes>, LoadError> {
        Model::<Bytes>::load(&mut Context::new(None), source)
    }

    /// The norm and the packed gate/up of `layer`.
    pub(super) fn parts(layer: &<Layer<Bytes> as View<2>>::Layer<'_>) -> (Vec<u8>, Vec<u8>) {
        let (norm, mlp) = layer;
        (norm.0.0.clone(), mlp.0.parts().0.affine.weight.0.clone())
    }
}
