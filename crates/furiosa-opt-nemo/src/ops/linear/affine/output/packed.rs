use furiosa_opt_std::prelude::*;

use crate::View;
use crate::ops::layout::axes::Parallel;
use crate::ops::layout::axes::Parallelism;
use crate::ops::linear::Affine;

/// A half-matrix's physical mapping, covering every local output and input exactly once.
pub trait Layout {
    type Input: AxisName;
    type Output: AxisName;
    type Half: M;
}

axes![Slot = 3, Factor = 8];
type Chips = <Parallel<2> as Parallelism>::Chip;
type Half<L> = <L as Layout>::Half;
pub type Stored<L> = m![Slot, { Half<L> }];
pub type Factors = m![Factor, 1 # 8];
pub type Packed<L> = Affine<Parameters<L>, ()>;

/// Slot `h-c+1` holds half `h` by owner `c` on chips `h` and `c`; factors 0/1 restore input/weight.
#[derive(DeviceSend)]
pub struct Parameters<L: Layout> {
    pub weight: HbmTensor<f8e4m3, Chips, Stored<L>>,
    pub factors: HbmTensor<f32, Chips, Factors>,
}

impl<L: Layout> Parameters<L> {
    pub fn pack(weight: &[u8]) -> Vec<u8> {
        type I<L> = <L as Layout>::Input;
        type O<L> = <L as Layout>::Output;
        const {
            assert!(I::<L>::SIZE % 2 == 0 && O::<L>::SIZE % 2 == 0);
            assert!(Half::<L>::SIZE == I::<L>::SIZE * O::<L>::SIZE / 4);
        }
        let (rows, columns) = (O::<L>::SIZE / 2, I::<L>::SIZE / 2);
        let mut packed = vec![0; 2 * Stored::<L>::SIZE];
        for h in 0..2 {
            for c in 0..2 {
                let source = Tensor::<
                    f8e4m3,
                    m![{ Symbol<O<L>> } % ({ Symbol<O<L>> } / 2),
                    { Symbol<I<L>> } % ({ Symbol<I<L>> } / 2)],
                >::from_buf(
                    (0..rows)
                        .flat_map(|row| {
                            let at = (h * rows + row) * 2 * columns + c * columns;
                            weight[at..at + columns].iter().copied()
                        })
                        .collect::<Vec<_>>(),
                );
                let mut half = Tensor::<f8e4m3, Half<L>>::zero();
                half.view_mut().transpose(source.view(), true);
                let bytes = half.into_buf();
                for rank in 0..2 {
                    if rank == h || rank == c {
                        let at = rank * Stored::<L>::SIZE + (h + 1 - c) * Half::<L>::SIZE;
                        packed[at..at + bytes.len()].copy_from_slice(&bytes);
                    }
                }
            }
        }
        packed
    }
}

impl<L: Layout, const N: usize> View<N> for Parameters<L> {
    type Storage = (
        <HbmTensor<f8e4m3, Chips, Stored<L>> as View<N>>::Storage,
        <HbmTensor<f32, Chips, Factors> as View<N>>::Storage,
    );

    fn view(storage: &Self::Storage, layer: usize) -> Self {
        Self {
            weight: View::<N>::view(&storage.0, layer),
            factors: View::<N>::view(&storage.1, layer),
        }
    }
}
