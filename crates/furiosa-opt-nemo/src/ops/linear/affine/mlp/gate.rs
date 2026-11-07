use super::Chips;
use furiosa_opt_std::prelude::*;

/// Hidden features grouped by 80 and local outputs grouped by 56.
#[derive(Clone, Debug)]
pub struct Layout<H, I>(std::marker::PhantomData<(H, I)>);

type Hidden<H> = Symbol<H>;
type Intermediate<I> = Symbol<I>;
pub type Stored<H, I> = m![
    { Intermediate<I> } % 3584 / 1792,
    { Intermediate<I> } / 3584,
    { Intermediate<I> } % 1792 / 448,
    { Intermediate<I> } % 448 / 56,
    { Hidden<H> } / 80,
    { Intermediate<I> } % 56 / 8,
    { Hidden<H> } % 80 / 8,
    { Intermediate<I> } % 8,
    { Hidden<H> } % 8
];

impl<H: AxisName, I: AxisName> super::Layout for Layout<H, I> {
    type Source = m![{ Chips }, { Intermediate<I> }, { Hidden<H> }];
    type Stored = Stored<H, I>;
    const SHAPE: [usize; 2] = [2 * I::SIZE, H::SIZE];
    const AXIS: usize = 0;

    fn source(weight: &[u8]) -> Vec<u8> {
        const {
            assert!(H::SIZE % 160 == 0 && I::SIZE % 3584 == 0);
        }
        weight.to_vec()
    }
}
