use super::Chips;
use furiosa_opt_std::prelude::*;

/// Row-parallel inputs zero-padded to 8192, stored in 80-output by 256-feature runs.
#[derive(Clone, Debug)]
pub struct Layout<H, I>(std::marker::PhantomData<(H, I)>);

type Hidden<H> = Symbol<H>;
type Intermediate<I> = Symbol<I>;
pub type Stored<H, I> = m![
    { Hidden<H> } / 1280,
    { Hidden<H> } % 1280 / 640,
    { Hidden<H> } % 640 / 320,
    { Intermediate<I> } # 8192 / 4096,
    { Intermediate<I> } # 8192 % 4096 / 2048,
    { Intermediate<I> } # 8192 % 2048 / 256,
    { Hidden<H> } % 320 / 80,
    { Hidden<H> } % 80,
    { Intermediate<I> } # 8192 % 256
];

impl<H: AxisName, I: AxisName> super::Layout for Layout<H, I> {
    type Source = m![{ Hidden<H> }, { Chips }, { Intermediate<I> } # 8192];
    type Stored = Stored<H, I>;
    const SHAPE: [usize; 2] = [H::SIZE, 2 * I::SIZE];
    const AXIS: usize = 1;

    fn source(weight: &[u8]) -> Vec<u8> {
        const {
            assert!(H::SIZE % 1280 == 0 && I::SIZE <= 8192);
        }
        let mut source = vec![0; H::SIZE * 2 * 8192];
        for (at, row) in weight.chunks_exact(I::SIZE).enumerate() {
            source[at * 8192..][..I::SIZE].copy_from_slice(row);
        }
        source
    }
}
