use super::View;
use crate::load::Context;
use crate::ops::linear::{Affine, Linear};
use crate::ops::normalization::RmsNorm;
use crate::ops::swiglu::{Down, Mlp};
use crate::tensor::form::{BlockScaled, Weight, fp8};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

impl<In: M, Out: M, W: View<N>, S: View<N>, const N: usize> View<N> for fp8::Rows<In, Out, W, S> {
    type Storage = (W::Storage, S::Storage);
    type Layer<'a> = fp8::Rows<In, Out, W::Layer<'a>, S::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        fp8::Rows::new(W::view(&storage.0, layer), S::view(&storage.1, layer))
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        let (w, s) = layers.into_iter().map(|layer| (layer.0, layer.1)).unzip();
        (W::stack(w, context), S::stack(s, context))
    }
}

impl<
    In: AxisName,
    Out: AxisName,
    E: Scalar,
    S: Scalar,
    B: M,
    C: View<N>,
    Chips: M,
    W: View<N>,
    T: View<N>,
    const N: usize,
> View<N> for BlockScaled<In, Out, E, S, B, C, Chips, W, T>
{
    type Storage = (W::Storage, T::Storage, C::Storage);
    type Layer<'a> = BlockScaled<In, Out, E, S, B, C::Layer<'a>, Chips, W::Layer<'a>, T::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        BlockScaled {
            weight: W::view(&storage.0, layer),
            weight_scale: T::view(&storage.1, layer),
            contract: C::view(&storage.2, layer),
            shape: PhantomData,
        }
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        let (w, (t, c)): (Vec<_>, (Vec<_>, Vec<_>)) = layers
            .into_iter()
            .map(|layer| (layer.weight, (layer.weight_scale, layer.contract)))
            .unzip();
        (
            W::stack(w, context),
            T::stack(t, context),
            C::stack(c, context),
        )
    }
}

impl<T: View<N>, const N: usize> View<N> for fp8::Squares<T> {
    type Storage = T::Storage;
    type Layer<'a> = fp8::Squares<T::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        fp8::Squares(T::view(storage, layer))
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        T::stack(layers.into_iter().map(|layer| layer.0).collect(), context)
    }
}

impl<T: View<N>, const N: usize> View<N> for Affine<T, ()> {
    type Storage = T::Storage;
    type Layer<'a> = Affine<T::Layer<'a>, ()>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        Affine::new(T::view(storage, layer), ())
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        T::stack(
            layers.into_iter().map(|layer| layer.weight).collect(),
            context,
        )
    }
}

impl<W: M, C: M, T: View<N>, const N: usize> View<N> for Weight<W, C, T> {
    type Storage = T::Storage;
    type Layer<'a> = Weight<W, C, T::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        Weight::new(T::view(storage, layer))
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        T::stack(
            layers.into_iter().map(Weight::into_tensor).collect(),
            context,
        )
    }
}

impl<W: AxisName, C: M, const E: usize, T: View<N>, const N: usize> View<N>
    for RmsNorm<W, E, C, T>
{
    type Storage = T::Storage;
    type Layer<'a> = RmsNorm<W, E, C, T::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        RmsNorm::new(T::view(storage, layer))
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        T::stack(
            layers.into_iter().map(RmsNorm::into_weight).collect(),
            context,
        )
    }
}

impl<H: AxisName, I: AxisName, F: View<N>, D: View<N>, const N: usize> View<N> for Mlp<H, I, F, D>
where
    Down<I>: AxisName,
{
    type Storage = (F::Storage, D::Storage);
    type Layer<'a> = Mlp<H, I, F::Layer<'a>, D::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        Mlp::new(F::view(&storage.0, layer), D::view(&storage.1, layer))
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        let (f, d) = layers.into_iter().map(Mlp::into_parts).unzip();
        (F::stack(f, context), D::stack(d, context))
    }
}

impl<In: AxisName, Out: AxisName, C: M, T: View<N>, const N: usize> View<N>
    for Linear<In, Out, C, T>
{
    type Storage = T::Storage;
    type Layer<'a> = Linear<In, Out, C, T::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        Linear::new(Weight::new(T::view(storage, layer)))
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        T::stack(
            layers
                .into_iter()
                .map(|linear| linear.affine.weight)
                .collect(),
            context,
        )
    }
}
