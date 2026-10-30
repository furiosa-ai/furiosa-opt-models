//! The passes over tiles of packed rows in HBM, written once over the [`Tiling`] that lays a tile
//! over the two clusters and the slices of each.
//!
//! - A tiling only names its mappings and its row statistic; the passes read nothing else.
//! - A span's per-feature operands sit in the vector registers of every slice that holds their
//!   features; the next tile arrives while the current one computes, and a tile's result is
//!   stored while the next one computes.

use furiosa_opt_std::prelude::*;

use crate::ops::index::packed::{Group, Rows, Table};
use crate::ops::normalization::Widen;

/// How a tile of packed rows of `W` features lies over the two clusters and the slices of each.
pub(crate) trait Tiling<W: AxisName>: Sized {
    /// The tiles of `R` rows, in row order.
    type Tiles<R: AxisName>: Count;
    /// The rows of a tile.
    type Rows: M;
    /// A tile's rows over the clusters, then over a cluster's slices with their features.
    type Half: M;
    type Slices: M;
    /// A slice's part of a tile.
    type Element: M;
    /// The features of a row that one slice holds.
    type Run: Count;
    /// A row's statistic in each slice of the row.
    type Statistic: M;
    /// The order a modulating pass reads a part in, whole and split into four-lane halves.
    type Modulated: M;
    type Split: M;

    /// `1 / sqrt(mean(x²) + epsilon)` of each row of `part`; `EPSILON` holds epsilon's FP32 bits.
    fn inverse<const EPSILON: usize, X: M>(
        device: &mut Device,
        part: &Part<Self, W, X>,
    ) -> RowOperand<Self, W, X>;

    /// Tile `at` of `rows`.
    fn tile<R: AxisName, X: M>(rows: &Rows<R, W, X>, at: usize) -> View<'_, Self, R, W, X> {
        // SAFETY: the tiles split the row axis in row order; every cell keeps its address.
        unsafe {
            rows.view()
                .reshape::<X, m![{ Self::Tiles<R> }, { Self::Rows }, W]>()
        }
        .tile::<m![{ Self::Tiles<R> }], 1, m![
            { Self::Tiles<R> } = 1 # ({ Self::Tiles<R> }),
            { Self::Rows },
            W
        ]>(at)
    }

    /// Tile `at` of `rows` into `part`.
    fn fill<R: AxisName, X: M>(
        device: &mut Device,
        rows: &Rows<R, W, X>,
        at: usize,
        part: &mut Part<Self, W, X>,
    ) {
        Self::tile(rows, at).to_dm_view(&mut device.tdma, part.view_mut());
    }

    /// `part` into tile `at` of `rows`.
    fn spill<R: AxisName, X: M>(
        device: &mut Device,
        part: &Part<Self, W, X>,
        rows: &mut Rows<R, W, X>,
        at: usize,
    ) {
        part.view().to_hbm_view(
            &mut device.tdma,
            // SAFETY: the tiles split the row axis in row order; every cell keeps its address.
            unsafe {
                rows.view_mut()
                    .reshape::<X, m![{ Self::Tiles<R> }, { Self::Rows }, W]>()
            }
            .tile::<m![{ Self::Tiles<R> }], 1, m![
                { Self::Tiles<R> } = 1 #{!} ({ Self::Tiles<R> }),
                { Self::Rows },
                W
            ]>(at),
        );
    }

    /// A per-feature vector, in the vector registers of the slices that hold its features.
    fn operand<X: M>(
        device: &mut Device,
        vector: HbmTensorView<'_, bf16, X, m![W]>,
    ) -> Operand<Self, W, X> {
        let in_dm: Feature<Self, W, X> = vector.to_dm(&mut device.tdma);
        in_dm.widen::<W, 8>(&mut device.sub)
    }

    /// `bf16(x * inverse * gain + shift)` of a part, FP32 inside, into `output`: the inverse of the
    /// part's rows, the gain and shift of its span.
    fn modulate<X: M>(
        device: &mut Device,
        (part, inverse): (&Part<Self, W, X>, &RowOperand<Self, W, X>),
        (gain, shift): (&Operand<Self, W, X>, &Operand<Self, W, X>),
        output: &mut Part<Self, W, X>,
    ) {
        device
            .main
            .begin(part.view())
            .fetch::<Self::Modulated, m![W % 8]>()
            .fetch_cast::<f32>()
            .collect::<Self::Modulated, m![W % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Self::Split, m![W % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), inverse)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), gain)
            .vector_fp_binary(FpBinaryOp::AddF, shift)
            .vector_widen_concat::<Self::Modulated, m![W % 8]>()
            .vector_final()
            .cast::<bf16, m![W % 8 # 16]>()
            .commit_trim::<m![W % 8]>()
            .commit_view(output.view_mut());
    }
}

/// Tile `at` of the rows `R`, as the tiling cuts them.
pub(crate) type View<'r, T, R, W, X> = HbmTensorView<
    'r,
    bf16,
    X,
    m![
        { <T as Tiling<W>>::Tiles<R> } = 1 # ({ <T as Tiling<W>>::Tiles<R> }),
        { <T as Tiling<W>>::Rows },
        W
    ],
>;

/// A slice's part of a tile.
pub(crate) type Part<T, W, X, D = bf16> =
    DmTensor<D, X, <T as Tiling<W>>::Half, <T as Tiling<W>>::Slices, <T as Tiling<W>>::Element>;

/// A slice's part of a per-feature vector, in DM and in the vector registers.
pub(crate) type Feature<T, W, X> = DmTensor<
    bf16,
    X,
    <T as Tiling<W>>::Half,
    <T as Tiling<W>>::Slices,
    m![W % ({ <T as Tiling<W>>::Run })],
>;
pub(crate) type Operand<T, W, X> = VrfTensor<
    f32,
    X,
    <T as Tiling<W>>::Half,
    <T as Tiling<W>>::Slices,
    m![W % ({ <T as Tiling<W>>::Run })],
>;

/// Each row's statistic of a tile, in the vector registers.
pub(crate) type RowOperand<T, W, X> = VrfTensor<
    f32,
    X,
    <T as Tiling<W>>::Half,
    <T as Tiling<W>>::Slices,
    <T as Tiling<W>>::Statistic,
>;

/// Table row `index`.
pub(crate) fn row<W: AxisName, X: M>(
    table: &Table<W, X>,
    index: usize,
) -> HbmTensorView<'_, bf16, X, m![W]> {
    // SAFETY: a table row is its features, contiguous.
    unsafe {
        table
            .view()
            .tile::<m![Group], 1, m![Group = 1 # Group, W]>(index)
            .reshape::<X, m![W]>()
    }
}
