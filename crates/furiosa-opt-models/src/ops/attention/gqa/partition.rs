use super::shard::Partition;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

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
    const SHARDS: usize,
    const SCALE: usize,
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
    const SHARDS: usize,
    const SCALE: usize,
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
        SHARDS,
        SCALE,
    >
{
    type Chip = Chip;
    type Cluster = Cluster;
    type Slices = Slices;
    type Agreed = Agreed;
    type Merged = Merged;
    type Rows = Rows;
    type Groups = Groups;
    type Head = Head;
    type Keys = Keys;
    type Key = Key;

    const SHARDS: usize = SHARDS;
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
