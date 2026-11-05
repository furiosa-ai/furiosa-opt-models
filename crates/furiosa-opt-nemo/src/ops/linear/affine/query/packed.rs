use furiosa_opt_std::prelude::*;

use crate::View;
use crate::ops::layout::axes::{Parallel, Parallelism};
use crate::ops::linear::Affine;

/// TP2 head pairs, with 128 features per head and inputs grouped in blocks of 80.
pub trait Layout: Clone + std::fmt::Debug + 'static {
    type Input: AxisName;
    type Heads: AxisName;
    type Head: AxisName;
}

#[derive(Clone, Debug)]
pub struct Outputs<L>(std::marker::PhantomData<L>);

impl<L: Layout> AxisName for Outputs<L> {
    const NAME: Ident = Ident::new("Out");
    const SIZE: usize = L::Heads::SIZE * L::Head::SIZE / 4;
}

axes![Panel = 2, Factor = 8];
type Chips = <Parallel<2> as Parallelism>::Chip;
type I<L> = <L as Layout>::Input;
type O<L> = Symbol<Outputs<L>>;
pub type Stored<L> = m![
    { O<L> } / 512,
    { O<L> } % 512 / 128,
    { Symbol<I<L>> } / 80,
    { O<L> } % 8,
    { O<L> } % 128 / 32,
    { Symbol<I<L>> } % 80 / 16,
    { O<L> } % 32 / 8,
    { Symbol<I<L>> } % 16
];
pub type Weight<L> = m![Panel, { Stored<L> }];
pub type Factors = m![Factor, 1 # 8];
pub type Packed<L> = Affine<Parameters<L>, ()>;

/// Panel p contains local heads 2g+p; factors 0/1 restore the shared input and Q weight.
#[derive(DeviceSend)]
pub struct Parameters<L: Layout> {
    pub weight: HbmTensor<f8e4m3, Chips, Weight<L>>,
    pub factors: HbmTensor<f32, Chips, Factors>,
}

impl<L: Layout> Parameters<L> {
    pub fn pack(weight: &[u8]) -> Vec<u8> {
        const {
            assert!(L::Head::SIZE == 128);
            assert!(L::Heads::SIZE % 16 == 0);
            assert!(I::<L>::SIZE % 80 == 0);
        }
        (0..2)
            .flat_map(|rank| {
                (0..2).flat_map(move |panel| {
                    let source = Tensor::<f8e4m3, m![{ O<L> }, { Symbol<I<L>> }]>::from_buf(
                        (0..Outputs::<L>::SIZE)
                            .flat_map(|row| {
                                let head = rank * L::Heads::SIZE / 2 + 2 * (row / 128) + panel;
                                let at = (head * 128 + row % 128) * I::<L>::SIZE;
                                weight[at..at + I::<L>::SIZE].iter().copied()
                            })
                            .collect::<Vec<_>>(),
                    );
                    let mut stored = Tensor::<f8e4m3, Stored<L>>::zero();
                    stored.view_mut().transpose(source.view(), true);
                    stored.into_buf()
                })
            })
            .collect()
    }
}

impl<L: Layout, const N: usize> View<N> for Parameters<L> {
    type Storage = (
        <HbmTensor<f8e4m3, Chips, Weight<L>> as View<N>>::Storage,
        <HbmTensor<f32, Chips, Factors> as View<N>>::Storage,
    );

    fn view(storage: &Self::Storage, layer: usize) -> Self {
        Self {
            weight: View::<N>::view(&storage.0, layer),
            factors: View::<N>::view(&storage.1, layer),
        }
    }
}
