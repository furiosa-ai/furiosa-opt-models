use furiosa_opt_std::prelude::*;

use crate::ops::layout::axes::{Parallel, Parallelism};

/// TP2 output shards, interleaving each 64-output key block with its value block.
pub trait Layout: Clone + std::fmt::Debug + 'static {
    type Input: AxisName;
    type Output: AxisName;
}

type Chips = <Parallel<2> as Parallelism>::Chip;
type I<L> = <L as Layout>::Input;
type O<L> = Symbol<<L as Layout>::Output>;
pub type Stored<L> = m![
    { O<L> } / 512, { O<L> } % 512 / 128, { Symbol<I<L>> } / 80,
    { O<L> } % 8, { O<L> } % 128 / 32, { Symbol<I<L>> } % 80 / 16,
    { O<L> } % 32 / 8, { Symbol<I<L>> } % 16
];
pub type Factors<L> = m![
    { O<L> } / 512, { O<L> } % 512 / 128, { O<L> } % 128 / 64,
    { O<L> } % 64 / 32, 16, 1 # 8
];
pub type Region<L> = m![
    { O<L> } / 512, { O<L> } % 512 / 128, { Symbol<I<L>> } / 80,
    { O<L> } % 8, { O<L> } % 128 / 32 = 2 # 4, { Symbol<I<L>> } % 80 / 16,
    { O<L> } % 32 / 8, { Symbol<I<L>> } % 16
];
pub type Scale<L> = m![
    { O<L> } / 512, { O<L> } % 512 / 128, { O<L> } % 128 / 64 = 1 # 2,
    { O<L> } % 64 / 32, 16, 1 # 8
];

/// Each leaf exposes only its own weight and factor regions of the shared read-only pair.
#[derive(DeviceSend)]
pub struct Parameters<L: Layout, const ROLE: usize> {
    weight: HbmTensor<f8e4m3, Chips, Stored<L>>,
    factors: HbmTensor<f32, Chips, Factors<L>>,
}

impl<L: Layout, const ROLE: usize> Parameters<L, ROLE> {
    pub fn new(
        weight: HbmTensor<f8e4m3, Chips, Stored<L>>,
        factors: HbmTensor<f32, Chips, Factors<L>>,
    ) -> Self {
        const {
            assert!(ROLE < 2);
        }
        Self { weight, factors }
    }

    pub fn weight(&self) -> HbmTensorView<'_, f8e4m3, Chips, Region<L>> {
        self.weight
            .view()
            .tile::<m![{ O<L> } % 128 / 32], 2, Region<L>>(ROLE * 2)
    }

    pub fn factors(&self) -> HbmTensorView<'_, f32, Chips, Scale<L>> {
        self.factors
            .view()
            .tile::<m![{ O<L> } % 128 / 64], 1, Scale<L>>(ROLE)
    }

    pub fn pack(keys: &[u8], values: &[u8]) -> Vec<u8> {
        const {
            assert!(L::Output::SIZE % 512 == 0);
            assert!(I::<L>::SIZE % 80 == 0);
        }
        (0..2)
            .flat_map(|rank| {
                let source = Tensor::<f8e4m3, m![{ O<L> }, { Symbol<I<L>> }]>::from_buf(
                    (0..L::Output::SIZE)
                        .flat_map(|row| {
                            let at = (rank * L::Output::SIZE / 2 + row / 128 * 64 + row % 64)
                                * I::<L>::SIZE;
                            [keys, values][row % 128 / 64][at..at + I::<L>::SIZE]
                                .iter()
                                .copied()
                        })
                        .collect::<Vec<_>>(),
                );
                let mut stored = Tensor::<f8e4m3, Stored<L>>::zero();
                stored.view_mut().transpose(source.view(), true);
                stored.into_buf()
            })
            .collect()
    }
}
