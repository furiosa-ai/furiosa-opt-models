use super::View;
use crate::load::{Context, Named};
use crate::modules::{attention, residual::Residual};
use crate::ops::attention::Attention;
use crate::ops::attention::full::{self, head::Replicated};
use crate::ops::index::{gather::Gather, scatter::Scatter};
use crate::ops::linear::{Affine, Linear};
use crate::ops::normalization::RmsNorm;
use crate::ops::pointwise::add::Add;
use crate::ops::rotary::{self, SplitHalf, group::Table};
use crate::ops::swiglu::{Down, Mlp};
use crate::tensor::form::Weight;
use crate::tensor::form::fp8;
use crate::tensor::parallel::Chip;
use crate::tensor::scalar::F32;
use crate::{First, Sequential};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

impl<A: View<N>, B: View<N>, const N: usize> View<N> for (A, B) {
    type Storage = (A::Storage, B::Storage);
    type Layer = (A::Layer, B::Layer);

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        (A::view(&storage.0, layer), B::view(&storage.1, layer))
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        let (a, b) = unzip(layers);
        (A::stack(a, context), B::stack(b, context))
    }
}

impl<In, Out, W: View<N>, S: View<N>, const N: usize> View<N> for fp8::Rows<In, Out, W, S> {
    type Storage = (W::Storage, S::Storage);
    type Layer = fp8::Rows<In, Out, W::Layer, S::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        fp8::Rows::new(W::view(&storage.0, layer), S::view(&storage.1, layer))
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        let (w, s) = unzip(layers.map(|layer| (layer.0, layer.1)));
        (W::stack(w, context), S::stack(s, context))
    }
}

impl<T: View<N>, const N: usize> View<N> for fp8::Squares<T> {
    type Storage = T::Storage;
    type Layer = fp8::Squares<T::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        fp8::Squares(T::view(storage, layer))
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        T::stack(layers.map(|layer| layer.0), context)
    }
}

/// A part with no parameters, such as a key projection's absent head scales.
impl<const N: usize> View<N> for () {
    type Storage = ();
    type Layer = ();

    fn view(_: &(), _: usize) -> Self {}

    fn stack(_: [Self; N], _: &mut Context<'_>) {}
}

impl<const NAME: &'static str, A: View<N>, const N: usize> View<N> for Named<NAME, A> {
    type Storage = A::Storage;
    type Layer = Named<NAME, A::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        Named(A::view(storage, layer))
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        A::stack(layers.map(|Named(a)| a), context)
    }
}

impl<A: View<N>, B: View<N>, const N: usize> View<N> for Sequential<A, B> {
    type Storage = (A::Storage, B::Storage);
    type Layer = Sequential<A::Layer, B::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        Sequential(A::view(&storage.0, layer), B::view(&storage.1, layer))
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        let (a, b) = unzip(layers.map(|Sequential(a, b)| (a, b)));
        (A::stack(a, context), B::stack(b, context))
    }
}

impl<Norm: View<N>, Branch: View<N>, Add: View<N>, const N: usize> View<N>
    for Residual<Norm, Branch, Add>
{
    type Storage = (Norm::Storage, Branch::Storage, Add::Storage);
    type Layer = Residual<Norm::Layer, Branch::Layer, Add::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        Residual(
            Norm::view(&storage.0, layer),
            Branch::view(&storage.1, layer),
            Add::view(&storage.2, layer),
        )
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        let (a, b, c) = unzip3(layers.map(|Residual(a, b, c)| (a, b, c)));
        (
            Norm::stack(a, context),
            Branch::stack(b, context),
            Add::stack(c, context),
        )
    }
}

impl<const N: usize> View<N> for First {
    type Storage = ();
    type Layer = Self;

    fn view(_: &(), _: usize) -> Self {
        Self
    }

    fn stack(_: [Self; N], _: &mut Context<'_>) {}
}

impl<Projection: View<N>, QueryNorm: View<N>, KeyNorm: View<N>, const N: usize> View<N>
    for attention::NormalizedQkv<Projection, QueryNorm, KeyNorm>
{
    type Storage = (Projection::Storage, QueryNorm::Storage, KeyNorm::Storage);
    type Layer = attention::NormalizedQkv<Projection::Layer, QueryNorm::Layer, KeyNorm::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        attention::NormalizedQkv {
            projection: Projection::view(&storage.0, layer),
            query_norm: QueryNorm::view(&storage.1, layer),
            key_norm: KeyNorm::view(&storage.2, layer),
        }
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        let (p, q, k) = unzip3(layers.map(|layer| {
            let attention::NormalizedQkv {
                projection,
                query_norm,
                key_norm,
            } = layer;
            (projection, query_norm, key_norm)
        }));
        (
            Projection::stack(p, context),
            QueryNorm::stack(q, context),
            KeyNorm::stack(k, context),
        )
    }
}

impl<Projection: View<N>, Position: View<N>, const N: usize> View<N>
    for attention::Positioned<Projection, Position>
{
    type Storage = (Projection::Storage, Position::Storage);
    type Layer = attention::Positioned<Projection::Layer, Position::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        attention::Positioned {
            projection: Projection::view(&storage.0, layer),
            positions: Position::view(&storage.1, layer),
        }
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        let (p, r) = unzip(layers.map(|layer| (layer.projection, layer.positions)));
        (Projection::stack(p, context), Position::stack(r, context))
    }
}

impl<Write: View<N>, Read: View<N>, const N: usize> View<N> for attention::KeyValue<Write, Read> {
    type Storage = (Write::Storage, Read::Storage);
    type Layer = attention::KeyValue<Write::Layer, Read::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        attention::KeyValue {
            write: Write::view(&storage.0, layer),
            read: Read::view(&storage.1, layer),
        }
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        let (w, r) = unzip(layers.map(|layer| (layer.write, layer.read)));
        (Write::stack(w, context), Read::stack(r, context))
    }
}

impl<Q: View<N>, C: View<N>, A: View<N>, O: View<N>, const N: usize> View<N>
    for attention::Attention<Q, C, A, O>
{
    type Storage = (Q::Storage, C::Storage, A::Storage, O::Storage);
    type Layer = attention::Attention<Q::Layer, C::Layer, A::Layer, O::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        attention::Attention {
            qkv: Q::view(&storage.0, layer),
            cache: C::view(&storage.1, layer),
            attend: A::view(&storage.2, layer),
            output: O::view(&storage.3, layer),
        }
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        let (q, c, a, o) =
            unzip4(layers.map(|layer| (layer.qkv, layer.cache, layer.attend, layer.output)));
        (
            Q::stack(q, context),
            C::stack(c, context),
            A::stack(a, context),
            O::stack(o, context),
        )
    }
}

impl<Matrix: View<N>, const N: usize> View<N> for Affine<Matrix, ()> {
    type Storage = Matrix::Storage;
    type Layer = Affine<Matrix::Layer, ()>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        Affine::new(Matrix::view(storage, layer), ())
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        Matrix::stack(layers.map(|layer| layer.weight), context)
    }
}

impl<Q: View<N>, K: View<N>, V: View<N>, const N: usize> View<N>
    for attention::Projection<Q, K, V>
{
    type Storage = (Q::Storage, K::Storage, V::Storage);
    type Layer = attention::Projection<Q::Layer, K::Layer, V::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        attention::Projection {
            query: Q::view(&storage.0, layer),
            key: K::view(&storage.1, layer),
            value: V::view(&storage.2, layer),
        }
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        let (q, k, v) = unzip3(layers.map(|layer| (layer.query, layer.key, layer.value)));
        (
            Q::stack(q, context),
            K::stack(k, context),
            V::stack(v, context),
        )
    }
}

impl<W: M, C: M, T: View<N>, const N: usize> View<N> for Weight<W, C, T> {
    type Storage = T::Storage;
    type Layer = Weight<W, C, T::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        Weight::new(T::view(storage, layer))
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        T::stack(layers.map(Weight::into_tensor), context)
    }
}

impl<Width: AxisName, Chips: M, const E: usize, Parameter: View<N>, const N: usize> View<N>
    for RmsNorm<Width, E, Chips, Parameter>
{
    type Storage = Parameter::Storage;
    type Layer = RmsNorm<Width, E, Chips, Parameter::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        RmsNorm::new(Parameter::view(storage, layer))
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        Parameter::stack(layers.map(RmsNorm::into_weight), context)
    }
}

impl<H: AxisName, T: View<N>, const E: usize, const N: usize> View<N>
    for full::HeadNorm<H, Chip, Weight<Replicated<H>, Chip, T>, F32<E>>
{
    type Storage = T::Storage;
    type Layer = full::HeadNorm<H, Chip, Weight<Replicated<H>, Chip, T::Layer>, F32<E>>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        full::HeadNorm {
            weight: Weight::<Replicated<H>, Chip, T>::view(storage, layer),
            epsilon: F32,
            head: PhantomData,
        }
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        Weight::<Replicated<H>, Chip, T>::stack(layers.map(|norm| norm.weight), context)
    }
}

impl<H: AxisName, I: AxisName, F: View<N>, D: View<N>, const N: usize> View<N> for Mlp<H, I, F, D>
where
    Down<I>: AxisName,
{
    type Storage = (F::Storage, D::Storage);
    type Layer = Mlp<H, I, F::Layer, D::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        Mlp::new(F::view(&storage.0, layer), D::view(&storage.1, layer))
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        let (f, d) = unzip(layers.map(Mlp::into_parts));
        (F::stack(f, context), D::stack(d, context))
    }
}

impl<In: AxisName, Out: AxisName, C: M, T: View<N>, const N: usize> View<N>
    for Linear<In, Out, C, T>
{
    type Storage = T::Storage;
    type Layer = Linear<In, Out, C, T::Layer>;

    fn view(storage: &Self::Storage, layer: usize) -> Self::Layer {
        Linear::new(Weight::new(T::view(storage, layer)))
    }

    fn stack(layers: [Self; N], context: &mut Context<'_>) -> Self::Storage {
        T::stack(layers.map(|linear| linear.affine.weight), context)
    }
}

impl<'s, Buckets: AxisName, Head: AxisName, const N: usize> View<N>
    for SplitHalf<Head, &'s Table<Buckets, rotary::Row<Head>>>
{
    type Storage = &'s Table<Buckets, rotary::Row<Head>>;
    type Layer = Self;

    fn view(storage: &Self::Storage, _: usize) -> Self {
        Self {
            frequency: *storage,
            head: PhantomData,
        }
    }

    fn stack(layers: [Self; N], _: &mut Context<'_>) -> Self::Storage {
        let first = layers.into_iter().next().expect("a stack holds a layer");
        first.frequency
    }
}

impl<const N: usize> View<N> for Add {
    type Storage = ();
    type Layer = Self;

    fn view(_: &(), _: usize) -> Self {
        Add
    }

    fn stack(_: [Self; N], _: &mut Context<'_>) {}
}

impl<const N: usize> View<N> for Scatter {
    type Storage = ();
    type Layer = Self;

    fn view(_: &(), _: usize) -> Self {
        Scatter
    }

    fn stack(_: [Self; N], _: &mut Context<'_>) {}
}

impl<const N: usize> View<N> for Gather {
    type Storage = ();
    type Layer = Self;

    fn view(_: &(), _: usize) -> Self {
        Gather
    }

    fn stack(_: [Self; N], _: &mut Context<'_>) {}
}

impl<H: AxisName, G: AxisName, W: AxisName, const S: usize, const N: usize> View<N>
    for Attention<PhantomData<(H, G, W)>, F32<S>>
{
    type Storage = ();
    type Layer = Self;

    fn view(_: &(), _: usize) -> Self {
        Self {
            heads: PhantomData,
            scale: F32,
        }
    }

    fn stack(_: [Self; N], _: &mut Context<'_>) {}
}

/// Two parts of each of `N` layers, as two arrays of the layers.
fn unzip<A, B, const N: usize>(layers: [(A, B); N]) -> ([A; N], [B; N]) {
    let (a, b): (Vec<A>, Vec<B>) = Vec::from(layers).into_iter().unzip();
    let (Ok(a), Ok(b)) = (a.try_into(), b.try_into()) else {
        unreachable!("one part per layer")
    };
    (a, b)
}

/// Three parts of each of `N` layers, as three arrays of the layers.
fn unzip3<A, B, C, const N: usize>(layers: [(A, B, C); N]) -> ([A; N], [B; N], [C; N]) {
    let (ab, c) = unzip(layers.map(|(a, b, c)| ((a, b), c)));
    let (a, b) = unzip(ab);
    (a, b, c)
}

/// Four parts of each of `N` layers, as four arrays of the layers.
fn unzip4<A, B, C, D, const N: usize>(
    layers: [(A, B, C, D); N],
) -> ([A; N], [B; N], [C; N], [D; N]) {
    let (ab, cd) = unzip(layers.map(|(a, b, c, d)| ((a, b), (c, d))));
    let ((a, b), (c, d)) = (unzip(ab), unzip(cd));
    (a, b, c, d)
}
