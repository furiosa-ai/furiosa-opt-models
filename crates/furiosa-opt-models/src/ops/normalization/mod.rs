//! Per-row normalization with learned feature scales and FP32 statistics: the rows' operand
//! types, beside one file per norm.

pub use adaptive::{AdaptiveRms, FinalNorm, Shift};
pub use gain::{Centered, Epsilon};
pub use group_norm::GroupNorm;
pub use layer_norm::LayerNorm;
pub use normalize::Normalize;
pub use rms_norm::{EPSILON, InDm, Learned, Rms, RmsNorm, Scaled, UnitRmsNorm};
pub(crate) use rms_norm::{Lanes, Root, Scale, Squares};

use crate::specialize;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

pub(crate) mod adaptive;
mod gain;
mod group_norm;
mod layer_norm;
mod normalize;
mod rms_norm;
pub(crate) mod span;

/// Feature slices retain every row in both clusters for the projections that consume them.
pub type Cluster = m![2];

/// Smallest multiple of 16 dividing Width into at most 256 hardware slices.
#[derive(Clone, Debug)]
pub struct Chunk<Width>(PhantomData<Width>);

impl<Width: AxisName> AxisName for Chunk<Width> {
    const NAME: Ident = Ident::new("Chunk");

    const SIZE: usize = {
        assert!(Width::SIZE > 0 && Width::SIZE.is_multiple_of(16));
        let mut chunk = 16;
        while Width::SIZE / chunk > 256 || Width::SIZE % chunk != 0 {
            chunk += 16;
        }
        chunk
    };
}

/// Feature chunks broadcast across the 256 hardware slices.
pub type FeatureSlices<Width> = m![Width / ({ Symbol<Chunk<Width>> }) # 256];

/// The chunk of a row's features one feature slice holds, and a feature's place in it.
pub(crate) type Part<Width> = m![Width / ({ Symbol<Chunk<Width>> })];
pub(crate) type Offset<Width> = m![Width % ({ Symbol<Chunk<Width>> })];

/// Each row's contiguous features within one chunk.
pub type Element<Width, Rows> = m![Rows, Width % ({ Symbol<Chunk<Width>> })];

/// Launch buffers keep each hidden row contiguous.
pub type RowMajor<Width, Rows> = m![Rows, Width];

/// Launch rows stored a feature chunk at a time: each block holds those features of every row.
pub(crate) type Blocked<Width, Rows> = m![{ Part<Width> }, Rows, { Offset<Width> }];

/// Hidden rows share the feature placement of the following projections.
pub type Spread<Width, Rows, Chips = m![1]> =
    DmTensor<bf16, Chips, Cluster, FeatureSlices<Width>, Element<Width, Rows>>;

/// Slices a row of `W` features spans when a norm over HBM rows keeps `V` FP32 vectors of the
/// features in its registers.
#[derive(Clone, Debug)]
pub(crate) struct TileParts<W, const V: usize>(PhantomData<W>);

impl<W: AxisName, const V: usize> AxisName for TileParts<W, V> {
    const NAME: Ident = Ident::new("TileParts");
    const SIZE: usize = specialize::normalization::parts(W::SIZE, V);
}

/// Rows of a tile a cluster holds: one a group of [`TileParts`] slices.
#[derive(Clone, Debug)]
pub(crate) struct Lines<W, const V: usize>(PhantomData<W>);

impl<W: AxisName, const V: usize> AxisName for Lines<W, V> {
    const NAME: Ident = Ident::new("Lines");
    const SIZE: usize = specialize::SLICES / TileParts::<W, V>::SIZE;
}

/// Rows of a tile: both clusters' [`Lines`].
#[derive(Clone, Debug)]
pub(crate) struct TileRows<W, const V: usize>(PhantomData<W>);

impl<W: AxisName, const V: usize> AxisName for TileRows<W, V> {
    const NAME: Ident = Ident::new("TileRows");
    const SIZE: usize = 2 * Lines::<W, V>::SIZE;
}

/// One slice's part of a row.
pub(crate) type TilePart<W, const V: usize> = m![W % (W / ({ Symbol<TileParts<W, V>> }))];

/// A tile of rows of `R`, every row a group of [`TileParts`] slices of a cluster, each slice one
/// contiguous part of the row's features, so every transfer moves kilobyte runs.
pub(crate) type Tile<D, R, W, const V: usize, Chips, E = TilePart<W, V>> = DmTensor<
    D,
    Chips,
    m![R % ({ Symbol<TileRows<W, V>> }) / ({ Symbol<Lines<W, V>> })],
    m![R % ({ Symbol<Lines<W, V>> }), W / (W / ({ Symbol<TileParts<W, V>> }))],
    E,
>;

/// A per-feature BF16 vector in the slices of the rows it acts on, read into their vector
/// registers in FP32.
pub(crate) trait Widen {
    type Output;

    /// Each slice's part of the `W` features, `READ` of them a fetch.
    fn widen<W: AxisName, const READ: usize>(
        &self,
        sub: &mut TuContext<{ Tu::Sub }>,
    ) -> Self::Output;
}

impl<X: M, C: M, S: M, E: M> Widen for DmTensor<bf16, X, C, S, E> {
    type Output = VrfTensor<f32, X, C, S, E>;

    fn widen<W: AxisName, const READ: usize>(
        &self,
        sub: &mut TuContext<{ Tu::Sub }>,
    ) -> Self::Output {
        sub.begin(self.view())
            .fetch::<m![{ E } / { READ }], m![W % { READ }]>()
            .fetch_cast::<f32>()
            .collect::<m![{ E } / 8], m![W % 8]>()
            .to_vrf()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use furiosa_opt_std::prelude::*;

    /// Pseudo-random values in `[-1, 1)`, rounded to what the device reads.
    pub(crate) fn random(count: usize, mut state: u64) -> Vec<bf16> {
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                bf16::from_f32((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
            })
            .collect()
    }

    /// FP32 rows of `width` features whose magnitudes differ by up to 2^6 from row to row, with
    /// values off the BF16 grid.
    pub(crate) fn residual(rows: usize, width: usize) -> Vec<f32> {
        let (high, low) = (
            random(rows * width, 0x2545_f491_4f6c_dd1d),
            random(rows * width, 0x0123_4567_89ab_cdef),
        );
        high.iter()
            .zip(&low)
            .enumerate()
            .map(|(index, (h, l))| {
                (h.to_f32() + l.to_f32() / 512.0) * (1 << (index / width % 7)) as f32
            })
            .collect()
    }

    /// Row-major `[rows, width]` values into `[width / block, rows, width % block]`, the layout
    /// the kernels move rows in, and back.
    pub(crate) fn blocked<T: Copy>(rows: &[T], width: usize, block: usize) -> Vec<T> {
        let count = rows.len() / width;
        let mut output = Vec::with_capacity(rows.len());
        for chunk in 0..width / block {
            for row in 0..count {
                output.extend_from_slice(&rows[row * width + chunk * block..][..block]);
            }
        }
        output
    }
}
