use super::View;
use crate::load::Context;
use crate::modules::attention;
use crate::ops::attention::Attention;
use crate::ops::attention::full::{self, head::Replicated};
use crate::ops::index::{gather::Gather, scatter::Scatter};
use crate::ops::pointwise::add::Add;
use crate::ops::rotary::{self, SplitHalf, group::Table};
use crate::tensor::{form::Weight, parallel::Chip, scalar::F32};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

impl<P: View<N>, Q: View<N>, K: View<N>, const N: usize> View<N>
    for attention::NormalizedQkv<P, Q, K>
{
    type Storage = (P::Storage, Q::Storage, K::Storage);
    type Layer<'a> = attention::NormalizedQkv<P::Layer<'a>, Q::Layer<'a>, K::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        attention::NormalizedQkv {
            projection: P::view(&storage.0, layer),
            query_norm: Q::view(&storage.1, layer),
            key_norm: K::view(&storage.2, layer),
        }
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        let (p, (q, k)): (Vec<_>, (Vec<_>, Vec<_>)) = layers
            .into_iter()
            .map(|layer| (layer.projection, (layer.query_norm, layer.key_norm)))
            .unzip();
        (
            P::stack(p, context),
            Q::stack(q, context),
            K::stack(k, context),
        )
    }
}

impl<P: View<N>, R: View<N>, const N: usize> View<N> for attention::Positioned<P, R> {
    type Storage = (P::Storage, R::Storage);
    type Layer<'a> = attention::Positioned<P::Layer<'a>, R::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        attention::Positioned {
            projection: P::view(&storage.0, layer),
            positions: R::view(&storage.1, layer),
        }
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        let (p, r) = layers
            .into_iter()
            .map(|layer| (layer.projection, layer.positions))
            .unzip();
        (P::stack(p, context), R::stack(r, context))
    }
}

impl<W: View<N>, R: View<N>, const N: usize> View<N> for attention::KeyValue<W, R> {
    type Storage = (W::Storage, R::Storage);
    type Layer<'a> = attention::KeyValue<W::Layer<'a>, R::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        attention::KeyValue {
            write: W::view(&storage.0, layer),
            read: R::view(&storage.1, layer),
        }
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        let (w, r) = layers
            .into_iter()
            .map(|layer| (layer.write, layer.read))
            .unzip();
        (W::stack(w, context), R::stack(r, context))
    }
}

impl<Q: View<N>, C: View<N>, A: View<N>, O: View<N>, const N: usize> View<N>
    for attention::Attention<Q, C, A, O>
{
    type Storage = (Q::Storage, C::Storage, A::Storage, O::Storage);
    type Layer<'a> = attention::Attention<Q::Layer<'a>, C::Layer<'a>, A::Layer<'a>, O::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        attention::Attention {
            qkv: Q::view(&storage.0, layer),
            cache: C::view(&storage.1, layer),
            attend: A::view(&storage.2, layer),
            output: O::view(&storage.3, layer),
        }
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        let ((q, c), (a, o)): ((Vec<_>, Vec<_>), (Vec<_>, Vec<_>)) = layers
            .into_iter()
            .map(|layer| ((layer.qkv, layer.cache), (layer.attend, layer.output)))
            .unzip();
        (
            Q::stack(q, context),
            C::stack(c, context),
            A::stack(a, context),
            O::stack(o, context),
        )
    }
}

impl<Q: View<N>, K: View<N>, V: View<N>, const N: usize> View<N>
    for attention::Projection<Q, K, V>
{
    type Storage = (Q::Storage, K::Storage, V::Storage);
    type Layer<'a> = attention::Projection<Q::Layer<'a>, K::Layer<'a>, V::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        attention::Projection {
            query: Q::view(&storage.0, layer),
            key: K::view(&storage.1, layer),
            value: V::view(&storage.2, layer),
        }
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        let (q, (k, v)): (Vec<_>, (Vec<_>, Vec<_>)) = layers
            .into_iter()
            .map(|layer| (layer.query, (layer.key, layer.value)))
            .unzip();
        (
            Q::stack(q, context),
            K::stack(k, context),
            V::stack(v, context),
        )
    }
}

impl<H: AxisName, T: View<N>, const E: usize, const N: usize> View<N>
    for full::HeadNorm<H, Chip, Weight<Replicated<H>, Chip, T>, F32<E>>
{
    type Storage = T::Storage;
    type Layer<'a> = full::HeadNorm<H, Chip, Weight<Replicated<H>, Chip, T::Layer<'a>>, F32<E>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        full::HeadNorm {
            weight: Weight::new(T::view(storage, layer)),
            epsilon: F32,
            head: PhantomData,
        }
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        Weight::<Replicated<H>, Chip, T>::stack(
            layers.into_iter().map(|norm| norm.weight).collect(),
            context,
        )
    }
}

impl<'s, B: AxisName, H: AxisName, const N: usize> View<N>
    for SplitHalf<H, &'s Table<B, rotary::Row<H>>>
{
    type Storage = &'s Table<B, rotary::Row<H>>;
    type Layer<'a> = Self;

    fn view<'a>(storage: &'a Self::Storage, _: usize) -> Self::Layer<'a> {
        Self {
            frequency: *storage,
            head: PhantomData,
        }
    }

    fn stack(layers: Vec<Self>, _: &mut Context<'_>) -> Self::Storage {
        layers
            .into_iter()
            .next()
            .expect("a stack holds a layer")
            .frequency
    }
}

impl<const N: usize> View<N> for Add {
    type Storage = ();
    type Layer<'a> = Self;
    fn view<'a>(_: &'a (), _: usize) -> Self {
        Add
    }
    fn stack(_: Vec<Self>, _: &mut Context<'_>) {}
}

impl<const N: usize> View<N> for Scatter {
    type Storage = ();
    type Layer<'a> = Self;
    fn view<'a>(_: &'a (), _: usize) -> Self {
        Scatter
    }
    fn stack(_: Vec<Self>, _: &mut Context<'_>) {}
}

impl<const N: usize> View<N> for Gather {
    type Storage = ();
    type Layer<'a> = Self;
    fn view<'a>(_: &'a (), _: usize) -> Self {
        Gather
    }
    fn stack(_: Vec<Self>, _: &mut Context<'_>) {}
}

impl<H: AxisName, G: AxisName, W: AxisName, const S: usize, const N: usize> View<N>
    for Attention<PhantomData<(H, G, W)>, F32<S>>
{
    type Storage = ();
    type Layer<'a> = Self;
    fn view<'a>(_: &'a (), _: usize) -> Self {
        Self {
            heads: PhantomData,
            scale: F32,
        }
    }
    fn stack(_: Vec<Self>, _: &mut Context<'_>) {}
}
