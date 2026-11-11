use super::shard::Partition;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

/// A partition's mappings: where a pass spreads one row's windows over its cells, the slices named
/// as `Cells`, its statistics agree over `HeadSlices` and its attention merges onto `FirstLane`, by
/// default its slices and its shards'.
pub(crate) struct Geometry<
    Chip,
    Cluster,
    Slices,
    Agreed,
    Merged,
    Rows,
    Groups,
    Head,
    Keys,
    Key,
    const SCALE: usize,
    HeadSlices = Agreed,
    FirstLane = Merged,
    Cells = Slices,
>(
    PhantomData<(
        Chip,
        Cluster,
        Slices,
        Agreed,
        Merged,
        Rows,
        Groups,
        Head,
        Keys,
        Key,
        HeadSlices,
        FirstLane,
        Cells,
    )>,
);

impl<
    Chip: M,
    Cluster: M,
    Slices: M,
    Agreed: M,
    Merged: M,
    Rows: M,
    Groups: M,
    Head: M,
    Keys: M,
    Key: AxisName,
    const SCALE: usize,
    HeadSlices: M,
    FirstLane: M,
    Cells: M,
> Partition
    for Geometry<
        Chip,
        Cluster,
        Slices,
        Agreed,
        Merged,
        Rows,
        Groups,
        Head,
        Keys,
        Key,
        SCALE,
        HeadSlices,
        FirstLane,
        Cells,
    >
{
    type Chip = Chip;
    type Cluster = Cluster;
    type Slices = Slices;
    type Agreed = Agreed;
    type Merged = Merged;
    type HeadSlices = HeadSlices;
    type FirstLane = FirstLane;
    type Cells = Cells;
    type Rows = Rows;
    type Groups = Groups;
    type Head = Head;
    type Keys = Keys;
    type Key = Key;

    const SHARDS: usize = Key::SIZE / Keys::SIZE;
    const DIVISOR: f32 = {
        let scale = f32::from_bits(SCALE as u32);
        let divisor = 1.0 / scale;
        assert!(
            scale.is_finite() && scale > 0.0 && (1.0 / divisor).to_bits() == scale.to_bits(),
            "the score scale must be a positive f32 whose reciprocal divisor inverts back to it"
        );
        divisor
    };
}
