//! Scaled dot-product attention: each query's softmax-weighted values over the keys it sees.

use crate::Op;
use crate::load::Load;
use crate::ops::linear;
use crate::specialize::attention::Strategy;
use crate::tensor::cache::KeyValue;
use crate::tensor::scalar::F32;
use crate::tensor::{
    axis::{Single, Tile},
    mapping::Rows as Layout,
    parallel::Chip,
};
use furiosa_opt_std::prelude::*;
pub use split::{Packed, Split};
use std::marker::PhantomData;

pub mod flash;
pub mod full;
pub mod gqa;
pub mod latent;
mod output;
mod producer;
pub mod single;
/// The softmax pieces attention composes from its scores.
mod softmax {
    pub(super) mod merge;
}
mod split;

/// Scaled dot-product attention `softmax(scale * QK^T) V` over the keys each query sees, query
/// heads grouped over key/value heads, a decode's history attended as `STRATEGY` says.
#[derive(DeviceSend, Load)]
pub struct Attention<Heads = usize, Scale = f64, const STRATEGY: Strategy = { Strategy::Items }> {
    /// Query-head geometry.
    pub heads: Heads,
    /// Multiplier applied to query/key scores before additive bias.
    pub scale: Scale,
}

/// The attention a kernel compiles whole: its heads a type and its scale a compile-time FP32.
impl<Heads, const BITS: usize, const STRATEGY: Strategy> Default
    for Attention<PhantomData<Heads>, F32<BITS>, STRATEGY>
{
    fn default() -> Self {
        Self {
            heads: PhantomData,
            scale: F32,
        }
    }
}

type Rows<D, Count, Width, Chips = m![1]> = HbmTensor<D, Chips, m![Count, Width]>;
type Scale = HbmTensor<f32, m![1], m![1]>;

impl<
    D: Scalar,
    Width: AxisName,
    KvWidth: AxisName,
    Queries: AxisName,
    Keys: AxisName,
    Heads: AxisName,
    Bias,
>
    Op<(
        &Rows<D, Queries, Width>,
        &Rows<D, Keys, KvWidth>,
        &Rows<D, Keys, KvWidth>,
        Bias,
    )> for Attention<PhantomData<Heads>, Scale>
{
    type Output = Rows<D, Queries, Width>;

    fn forward(
        &self,
        _: &mut Device,
        _: (
            &Rows<D, Queries, Width>,
            &Rows<D, Keys, KvWidth>,
            &Rows<D, Keys, KvWidth>,
            Bias,
        ),
    ) -> Self::Output {
        const {
            assert!(Heads::SIZE > 0 && Width::SIZE > 0 && Width::SIZE % Heads::SIZE == 0);
            assert!(KvWidth::SIZE > 0 && KvWidth::SIZE % (Width::SIZE / Heads::SIZE) == 0);
            assert!(Width::SIZE % KvWidth::SIZE == 0);
        }
        todo!("NPU scaled dot-product attention with broadcast additive bias")
    }
}

impl<
    D: Scalar,
    Width: AxisName,
    KvWidth: AxisName,
    Queries: AxisName,
    Keys: AxisName,
    Heads: AxisName,
>
    Op<(
        &Rows<D, Queries, Width>,
        &Rows<D, Keys, KvWidth>,
        &Rows<D, Keys, KvWidth>,
    )> for Attention<PhantomData<Heads>, Scale>
{
    type Output = Rows<D, Queries, Width>;

    fn forward(
        &self,
        device: &mut Device,
        (query, key, value): (
            &Rows<D, Queries, Width>,
            &Rows<D, Keys, KvWidth>,
            &Rows<D, Keys, KvWidth>,
        ),
    ) -> Self::Output {
        self.forward(device, (query, key, value, ()))
    }
}

/// Width constructors a model module names its attention widths with; head counts, head width
/// and window envelope are the model's own.
pub mod geometry {

    use furiosa_opt_std::prelude::*;

    /// Hidden feature width of a decoder.
    #[derive(Clone, Debug)]
    pub struct Hidden<const N: usize>;

    impl<const N: usize> AxisName for Hidden<N> {
        const NAME: Ident = Ident::new("Hidden");
        const SIZE: usize = N;
    }

    /// Rows of a token table: `vocab_size` tokens.
    #[derive(Clone, Debug)]
    pub struct Vocabulary<const N: usize>;

    impl<const N: usize> AxisName for Vocabulary<N> {
        const NAME: Ident = Ident::new("Vocabulary");
        const SIZE: usize = N;
    }

    /// Attention heads of a layer, or its key and value heads.
    #[derive(Clone, Debug)]
    pub struct Heads<const N: usize>;

    impl<const N: usize> AxisName for Heads<N> {
        const NAME: Ident = Ident::new("Heads");
        const SIZE: usize = N;
    }

    /// Features of a head.
    #[derive(Clone, Debug)]
    pub struct Head<const N: usize>;

    impl<const N: usize> AxisName for Head<N> {
        const NAME: Ident = Ident::new("Head");
        const SIZE: usize = N;
    }

    /// Total feature width across all key heads, or all value heads.
    #[derive(Clone, Debug)]
    pub struct KvWidth<const N: usize>;

    impl<const N: usize> AxisName for KvWidth<N> {
        const NAME: Ident = Ident::new("KvWidth");
        const SIZE: usize = N;
    }

    /// Leading features of a head that rotate.
    #[derive(Clone, Debug)]
    pub struct Rotated<const N: usize>;

    impl<const N: usize> AxisName for Rotated<N> {
        const NAME: Ident = Ident::new("Rotated");
        const SIZE: usize = N;
    }

    /// Cache windows a layer's history spans.
    #[derive(Clone, Debug)]
    pub struct Windows<const N: usize>;

    impl<const N: usize> AxisName for Windows<N> {
        const NAME: Ident = Ident::new("Windows");
        const SIZE: usize = N;
    }

    /// Total feature width across all query heads.
    #[derive(Clone, Debug)]
    pub struct QueryWidth<const N: usize>;

    impl<const N: usize> AxisName for QueryWidth<N> {
        const NAME: Ident = Ident::new("QWidth");
        const SIZE: usize = N;
    }

    /// Total feature width of the packed query/key/value projection.
    #[derive(Clone, Debug)]
    pub struct ProjectionWidth<const N: usize>;

    impl<const N: usize> AxisName for ProjectionWidth<N> {
        const NAME: Ident = Ident::new("QkvWidth");
        const SIZE: usize = N;
    }

    /// Query heads sharing one key/value head.
    #[derive(Clone, Debug)]
    pub struct QueryGroups<const N: usize>;

    impl<const N: usize> AxisName for QueryGroups<N> {
        const NAME: Ident = Ident::new("Groups");
        const SIZE: usize = N;
    }

    /// Projection rows per key/value head, including the key and value rows.
    #[derive(Clone, Debug)]
    pub struct HeadRows<const N: usize>;

    impl<const N: usize> AxisName for HeadRows<N> {
        const NAME: Ident = Ident::new("QkvRows");
        const SIZE: usize = N;
    }

    /// The score multiplier `1 / sqrt(width)` of `width`-wide heads, as the bits of the FP32
    /// nearest to it.
    pub const fn scale(width: usize) -> usize {
        assert!(width > 0, "a head has channels");
        // The integer part of 2^60 / sqrt(width), 29 bits or more; its bits below the FP32
        // fraction are never exactly half, as only a power-of-four width makes it exact.
        let root = ((1u128 << 120) / width as u128).isqrt();
        let bits = u128::BITS - root.leading_zeros();
        let rounded = (root >> (bits - 24)) + ((root >> (bits - 25)) & 1);
        // A carry out of the 24 bits leaves 2^24: one more exponent step, a zero fraction.
        let exponent = bits + (rounded >> 24) as u32 + 127 - 61;
        ((exponent << 23) | (rounded as u32 & 0x7f_ffff)) as usize
    }
}

/// One slice per lane and KV head, holding the head's query rows or its key or value row of each
/// row group; a row's key and value scatter from here, so its group stays in the element. Fewer
/// than eight KV heads leave the slices of the absent ones empty.
pub(super) type HeadSlice<Rows, KvHeads> = m![{ Layout<Rows>::Lanes } # 8, KvHeads # 8, 1 # 4];

pub(super) type RowGroups<Rows, Group> = m![{ linear::Groups<Rows> }, Group];

/// Row groups of the `N`-row tile a leaf holds at once; a batch that fits the tile is one tile.
#[derive(Clone, Debug)]
pub struct TileGroups<Rows, const N: usize>(PhantomData<Rows>);

impl<Rows: AxisName, const N: usize> AxisName for TileGroups<Rows, N> {
    const NAME: Ident = Ident::new("TileGroups");
    const SIZE: usize = <linear::Groups<Tile<Rows, N>> as M>::SIZE;
}

pub(super) type Heads<Rows, KvHeads, Group, Head> =
    DmTensor<bf16, Chip, m![1 # 2], HeadSlice<Rows, KvHeads>, m![{ RowGroups<Rows, Group> }, Head]>;

#[cfg(test)]
pub(crate) mod tests {
    use super::geometry;
    use crate::tensor::cache;
    use furiosa_opt_std::prelude::*;

    // The geometry attention tests run at: eight KV heads of 128 channels and 80 windows.
    axes![KvHeads = 8, Head = 128, Chunks = 80];

    pub(crate) type Blocks = cache::Blocks<KvHeads, Head>;

    /// The score multiplier of a head width is the FP32 nearest to its inverse square root.
    #[test]
    fn scales_by_the_nearest_inverse_root() {
        for width in 1..=1024 {
            let scale = f32::from_bits(geometry::scale(width) as u32);
            let exact = 1.0 / (width as f64).sqrt();
            let error = |x: f32| (f64::from(x) - exact).abs();
            assert!(
                error(scale) <= error(scale.next_down()) && error(scale) <= error(scale.next_up()),
                "width {width}: {scale}"
            );
        }
    }
}
