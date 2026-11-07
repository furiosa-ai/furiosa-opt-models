pub mod cache;
pub mod flash;
pub mod full;
pub mod gqa;
pub mod rope;
mod split;

use self::geometry::Kv;
use crate::ops::layout::{
    Rows as Layout,
    axes::{Chip, Single, Tile},
};
use crate::ops::linear;
use furiosa_opt_std::prelude::*;
pub use split::Split;

/// Width constructors a model module names its attention widths with, and the key/value pair
/// each cached token stores; head counts, head width and window envelope are the model's own.
pub mod geometry {

    use furiosa_opt_std::prelude::*;

    axes![Kv = 2];

    /// Hidden feature width of a decoder.
    #[derive(Clone, Debug)]
    pub struct Hidden<const N: usize>;

    impl<const N: usize> AxisName for Hidden<N> {
        const NAME: Ident = Ident::new("Hidden");
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
}

/// One slice per lane and KV head, holding the head's query rows or its key or value row of each
/// row group; a row's key and value scatter from here, so its group stays in the element.
pub(super) type HeadSlice<Rows, KvHeads> = m![{ Layout<Rows>::Lanes } # 8, KvHeads, 1 # 4];

pub(super) type RowGroups<Rows, Group> = m![{ linear::Groups<Rows> }, Group];

/// Row groups of the `N`-row tile a leaf holds at once; a batch that fits the tile is one tile.
#[derive(Clone, Debug)]
pub struct TileGroups<Rows, const N: usize>(core::marker::PhantomData<Rows>);

impl<Rows: AxisName, const N: usize> AxisName for TileGroups<Rows, N> {
    const NAME: Ident = Ident::new("TileGroups");
    const SIZE: usize = <linear::Groups<Tile<Rows, N>> as M>::SIZE;
}

pub(super) type Heads<Rows, KvHeads, Group, Head> =
    DmTensor<bf16, Chip, m![1 # 2], HeadSlice<Rows, KvHeads>, m![{ RowGroups<Rows, Group> }, Head]>;

#[cfg(test)]
pub(crate) mod tests {
    use crate::ops::attention::cache;
    use furiosa_opt_std::prelude::*;

    // The geometry attention tests run at: eight KV heads of 128 channels and 80 windows.
    axes![KvHeads = 8, Head = 128, Chunks = 80];

    pub(crate) type Blocks = cache::Blocks<KvHeads, Head>;
}
