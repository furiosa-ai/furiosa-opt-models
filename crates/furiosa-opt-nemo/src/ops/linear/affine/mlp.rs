use furiosa_opt_std::prelude::*;

use crate::View;
use crate::ops::layout::axes::{Parallel, Parallelism};
use crate::ops::linear::Affine;

pub mod down;
pub mod gate;

pub type Chips = <Parallel<2> as Parallelism>::Chip;
axes![Factor = 8];
pub type Factors = m![Factor, 1 # 8];
pub type Scale = m![Factor = 2 # 8, 1 # 8];

/// A full checkpoint matrix and its TP2 storage coordinates.
pub trait Layout: Clone + std::fmt::Debug + 'static {
    type Source: M;
    type Stored: M;
    const SHAPE: [usize; 2];
    const AXIS: usize;

    fn source(weight: &[u8]) -> Vec<u8> {
        weight.to_vec()
    }
}

pub type Packed<L, const ROLE: usize> = Affine<Parameters<L, ROLE>, ()>;

/// Each projection owns its coefficients and exposes its two factors from one shared row.
#[derive(DeviceSend)]
pub struct Parameters<L: Layout, const ROLE: usize> {
    weight: HbmTensor<f8e4m3, Chips, L::Stored>,
    factors: HbmTensor<f32, Chips, Factors>,
}

impl<L: Layout, const ROLE: usize> Parameters<L, ROLE> {
    pub fn new(
        weight: HbmTensor<f8e4m3, Chips, L::Stored>,
        factors: HbmTensor<f32, Chips, Factors>,
    ) -> Self {
        const {
            assert!(ROLE < 3 && L::AXIS == if ROLE < 2 { 0 } else { 1 });
        }
        Self { weight, factors }
    }

    pub fn weight(&self) -> HbmTensorView<'_, f8e4m3, Chips, L::Stored> {
        self.weight.view()
    }

    /// The shared producer divides by the gate input factor; the up factor stays distinct.
    pub fn factors(&self) -> HbmTensorView<'_, f32, Chips, Scale> {
        self.factors.view().tile::<m![Factor], 2, Scale>(ROLE * 2)
    }

    pub fn pack(weight: &[u8]) -> Vec<u8> {
        let source = Tensor::<f8e4m3, L::Source>::from_buf(L::source(weight));
        let mut stored = Tensor::<f8e4m3, m![{ Chips }, { L::Stored }]>::zero();
        stored.view_mut().transpose(source.view(), true);
        stored.into_buf()
    }
}

impl<L: Layout, const ROLE: usize, const N: usize> View<N> for Parameters<L, ROLE> {
    type Storage = (
        <HbmTensor<f8e4m3, Chips, L::Stored> as View<N>>::Storage,
        <HbmTensor<f32, Chips, Factors> as View<N>>::Storage,
    );

    fn view(storage: &Self::Storage, layer: usize) -> Self {
        Self::new(
            View::<N>::view(&storage.0, layer),
            View::<N>::view(&storage.1, layer),
        )
    }
}
