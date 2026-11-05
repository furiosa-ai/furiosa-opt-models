use furiosa_opt_std::prelude::*;

use crate::ops::layout::axes::Chip;

pub mod fp8;
pub(crate) mod group;
pub mod packed;

pub(crate) type Tile = crate::ops::layout::axes::Rows<128>;
pub(crate) type Cluster<Heads> = m![Heads / (Heads / 2)];
pub(crate) type Output<G, Heads, Head, Spread, Local> =
    DmTensor<bf16, Chip, Cluster<Heads>, Rows<G, Heads, Spread>, m![{ Local }, Tile % 8, Head]>;
pub(crate) type Rows<G, Heads, Spread> = m![
    Heads % (Heads / 2) / (Heads / 4),
    G,
    Tile / 8,
    Heads % (Heads / 4) / (Heads / 8),
    { Spread }
];
