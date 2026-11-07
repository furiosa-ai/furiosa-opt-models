use crate::Module;
use furiosa_opt_std::prelude::*;

/// A layer collection owning one backing per parameter kind.
#[derive(DeviceSend)]
pub struct Stacked<L: View<N>, const N: usize> {
    pub storage: L::Storage,
}

/// Construct a layer's parameter views without issuing device work.
pub trait View<const N: usize> {
    type Storage;
    fn view(storage: &Self::Storage, layer: usize) -> Self;
}

impl<L: View<N>, const N: usize> Stacked<L, N> {
    pub fn layer(&self, layer: usize) -> L {
        L::view(&self.storage, layer)
    }
}

impl<D, X, L: View<N>, P, const N: usize> Module<(&mut D, (X, P))> for super::Layers<Stacked<L, N>>
where
    P: super::Contexts<L, D, X>,
{
    type Output = X;

    fn forward(&self, (device, (mut hidden, mut contexts)): (&mut D, (X, P))) -> X {
        for index in 0..N {
            hidden = contexts.apply(&self.layers.layer(index), &mut *device, hidden, index);
        }
        hidden
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
}
