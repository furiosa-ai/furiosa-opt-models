//! An RMS norm's rows in tiles its vector registers hold: whole tiles, then a last one that may
//! be partial, each normalized from its rows' variances and the gain.

use crate::ops::normalization::{Lanes, Root, Scale};
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;

/// How a norm's `T` rows of `E` features each move through its passes.
pub(crate) type Rows<T, E> =
    Lanes<m![{ T }, { E } / 8], m![{ E } % 8], m![{ T }, { E } / 4], m![{ E } % 4]>;

/// A norm's rows, their variances and the gain.
pub(crate) type Inputs<'a, C, P, T, E> = (
    &'a DmTensor<bf16, Chip, C, P, m![{ T }, { E }]>,
    &'a DmTensor<f32, Chip, C, P, m![{ T }, 1 # 8]>,
    &'a VrfTensor<f32, Chip, C, P, E>,
);

/// The `LEN` rows of `T` from row `start` normalized, as the stream its caller commits into the
/// same rows of its output.
pub(crate) fn normalized<'l, const LEN: usize, C: M, P: M, T: M, E: M>(
    context: &'l mut Device,
    (rows, variance, gain): Inputs<'l, C, P, T, E>,
    start: usize,
) -> CommitTrimTensor<
    'l,
    { Tu::Main },
    bf16,
    Chip,
    C,
    P,
    m![{ T } = { LEN } # ({ T } / 1), { E } / 8],
    m![{ E } % 8],
> {
    type Window<T, const LEN: usize> = m![{ T } = { LEN } # ({ T } / 1)];
    let inverse: VrfTensor<f32, Chip, C, P, m![{ T } = { LEN }, 1 # 8]> =
        Root::<Window<T, LEN>, m![1 # 2]>::of(
            &mut context.main,
            variance
                .view()
                .tile::<T, LEN, m![{ Window<T, LEN> }, 1 # 8]>(start),
            |square| square,
        )
        .to_vrf(&mut context.sub);
    Scale::<Rows<Window<T, LEN>, E>>::of(
        &mut context.main,
        rows.view()
            .tile::<T, LEN, m![{ Window<T, LEN> }, { E }]>(start),
        &inverse,
        gain,
    )
}
