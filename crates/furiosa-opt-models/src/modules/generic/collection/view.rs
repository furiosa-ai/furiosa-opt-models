use super::View;
use crate::load::{Context, Named};
use crate::modules::residual::Residual;
use crate::{First, Sequential};

mod attention;
mod parameter;

impl<A: View<N>, B: View<N>, const N: usize> View<N> for (A, B) {
    type Storage = (A::Storage, B::Storage);
    type Layer<'a> = (A::Layer<'a>, B::Layer<'a>);

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        (A::view(&storage.0, layer), B::view(&storage.1, layer))
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        let (a, b) = layers.into_iter().unzip();
        (A::stack(a, context), B::stack(b, context))
    }
}

impl<const N: usize> View<N> for () {
    type Storage = ();
    type Layer<'a> = ();

    fn view<'a>(_: &'a (), _: usize) {}
    fn stack(_: Vec<Self>, _: &mut Context<'_>) {}
}

impl<const NAME: &'static str, A: View<N>, const N: usize> View<N> for Named<NAME, A> {
    type Storage = A::Storage;
    type Layer<'a> = Named<NAME, A::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        Named(A::view(storage, layer))
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        A::stack(layers.into_iter().map(|Named(a)| a).collect(), context)
    }
}

impl<A: View<N>, B: View<N>, const N: usize> View<N> for Sequential<A, B> {
    type Storage = (A::Storage, B::Storage);
    type Layer<'a> = Sequential<A::Layer<'a>, B::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        Sequential(A::view(&storage.0, layer), B::view(&storage.1, layer))
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        let (a, b) = layers.into_iter().map(|Sequential(a, b)| (a, b)).unzip();
        (A::stack(a, context), B::stack(b, context))
    }
}

impl<A: View<N>, B: View<N>, C: View<N>, const N: usize> View<N> for Residual<A, B, C> {
    type Storage = (A::Storage, B::Storage, C::Storage);
    type Layer<'a> = Residual<A::Layer<'a>, B::Layer<'a>, C::Layer<'a>>;

    fn view<'a>(storage: &'a Self::Storage, layer: usize) -> Self::Layer<'a> {
        Residual(
            A::view(&storage.0, layer),
            B::view(&storage.1, layer),
            C::view(&storage.2, layer),
        )
    }

    fn stack(layers: Vec<Self>, context: &mut Context<'_>) -> Self::Storage {
        let (a, (b, c)): (Vec<_>, (Vec<_>, Vec<_>)) = layers
            .into_iter()
            .map(|Residual(a, b, c)| (a, (b, c)))
            .unzip();
        (
            A::stack(a, context),
            B::stack(b, context),
            C::stack(c, context),
        )
    }
}

impl<const N: usize> View<N> for First {
    type Storage = ();
    type Layer<'a> = Self;

    fn view<'a>(_: &'a (), _: usize) -> Self {
        Self
    }
    fn stack(_: Vec<Self>, _: &mut Context<'_>) {}
}
