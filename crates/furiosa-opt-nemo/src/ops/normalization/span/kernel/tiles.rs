use furiosa_opt_std::prelude::*;

use super::{capacity, whole};
use crate::ops::layout::axes::Chip;

/// Whole tiles of `CAPACITY` rows followed by a last tile of `LAST` rows.
pub(crate) struct Tiles<
    const CAPACITY: usize,
    const LAST: usize,
    const EPSILON: usize = { crate::ops::normalization::EPSILON },
>;

/// Tile lengths and epsilon for a norm over rows `T` of elements `E`.
pub(crate) trait Tiling {
    const CAPACITY: usize;
    const LAST: usize;
    const EPSILON: usize;

    fn finish<C: M, P: M, T: M, E: M>(
        context: &mut Device,
        input: Inputs<'_, C, P, T, E>,
    ) -> DmTensor<bf16, Chip, C, P, m![{ T }, { E }]>;

    fn finish_live<C: M, P: M, T: M, E: M>(
        context: &mut Device,
        input: Inputs<'_, C, P, T, E>,
        rows: usize,
    ) -> DmTensor<bf16, Chip, C, P, m![{ T }, { E }]>;
}

/// A norm's rows, their variances and the gain.
type Inputs<'a, C, P, T, E> = (
    &'a DmTensor<bf16, Chip, C, P, m![{ T }, { E }]>,
    &'a DmTensor<f32, Chip, C, P, m![{ T }, 1 # 8]>,
    &'a VrfTensor<f32, Chip, C, P, E>,
);

impl<const CAPACITY: usize, const LAST: usize, const EPSILON: usize> Tiling
    for Tiles<CAPACITY, LAST, EPSILON>
{
    const CAPACITY: usize = CAPACITY;
    const LAST: usize = LAST;
    const EPSILON: usize = EPSILON;

    fn finish<C: M, P: M, T: M, E: M>(
        context: &mut Device,
        input: Inputs<'_, C, P, T, E>,
    ) -> DmTensor<bf16, Chip, C, P, m![{ T }, { E }]> {
        let mut output = DmTensor::new();
        if const { whole::<T, E>() } > 0 {
            #[unroll]
            for tile in 0..const { whole::<T, E>() } {
                apply::<C, P, T, E, CAPACITY>(
                    context,
                    input,
                    (&mut output, tile * const { capacity::<E>() }),
                );
            }
        }
        apply::<C, P, T, E, LAST>(
            context,
            input,
            (&mut output, const { whole::<T, E>() * capacity::<E>() }),
        );
        output
    }

    fn finish_live<C: M, P: M, T: M, E: M>(
        context: &mut Device,
        input: Inputs<'_, C, P, T, E>,
        rows: usize,
    ) -> DmTensor<bf16, Chip, C, P, m![{ T }, { E }]> {
        let mut output = DmTensor::new();
        if const { whole::<T, E>() } > 0 {
            #[unroll]
            for tile in 0..const { whole::<T, E>() } {
                apply_live::<C, P, T, E, CAPACITY>(
                    context,
                    input,
                    (&mut output, tile * const { capacity::<E>() }),
                    rows,
                );
            }
        }
        apply_live::<C, P, T, E, LAST>(
            context,
            input,
            (&mut output, const { whole::<T, E>() * capacity::<E>() }),
            rows,
        );
        output
    }
}

/// The `LEN` rows of a tile from `start` normalized, as the stream its caller commits.
fn tile<'l, C: M, P: M, T: M, E: M, const LEN: usize>(
    context: &'l mut Device,
    input: Inputs<'l, C, P, T, E>,
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
    type Tile<T, const LEN: usize> = m![{ T } = { LEN } # ({ T } / 1)];
    let inverse: VrfTensor<f32, Chip, C, P, m![{ T } = { LEN }, 1 # 8]> = context
        .main
        .begin(
            input
                .1
                .view()
                .tile::<T, LEN, m![{ Tile<T, LEN> }, 1 # 8]>(start),
        )
        .fetch::<Tile<T, LEN>, m![1 # 2]>()
        .collect::<Tile<T, LEN>, m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut context.sub);
    context
        .main
        .begin(
            input
                .0
                .view()
                .tile::<T, LEN, m![{ Tile<T, LEN> }, { E }]>(start),
        )
        .fetch::<m![{ Tile<T, LEN> }, { E } / 8], m![{ E } % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Tile<T, LEN> }, { E } / 8], m![{ E } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Tile<T, LEN> }, { E } / 4], m![{ E } % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inverse)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), input.2)
        .vector_widen_concat::<m![{ Tile<T, LEN> }, { E } / 8], m![{ E } % 8]>()
        .vector_final()
        .cast::<bf16, m![{ E } % 8 # 16]>()
        .commit_trim::<m![{ E } % 8]>()
}

fn apply<C: M, P: M, T: M, E: M, const LEN: usize>(
    context: &mut Device,
    input: Inputs<'_, C, P, T, E>,
    output: (&mut DmTensor<bf16, Chip, C, P, m![{ T }, { E }]>, usize),
) {
    tile::<C, P, T, E, LEN>(context, input, output.1).commit_view(
        output
            .0
            .view_mut()
            .tile::<T, LEN, m![{ T } = { LEN } #{!} ({ T } / 1), { E }]>(output.1),
    );
}

/// Normalize a counted tile without committing rows past `rows`.
fn apply_live<C: M, P: M, T: M, E: M, const LEN: usize>(
    context: &mut Device,
    input: Inputs<'_, C, P, T, E>,
    output: (&mut DmTensor<bf16, Chip, C, P, m![{ T }, { E }]>, usize),
    rows: usize,
) {
    tile::<C, P, T, E, LEN>(context, input, output.1).commit_prefix::<m![{ T } = { LEN }], _, _>(
        output
            .0
            .view_mut()
            .tile::<T, LEN, m![{ T } = { LEN } #{!} ({ T } / 1), { E }]>(output.1),
        rows - output.1,
    );
}
