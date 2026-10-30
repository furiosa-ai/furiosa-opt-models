//! The span MLP: one stream node over a span's normalized tiles, its stages run by the weight
//! forms, block-scaled E4M3 ([`fp8`]) or BF16 linears ([`linear`]).

use furiosa_opt_std::prelude::*;

use crate::ops::linear::Groups;
use crate::tensor::parallel::Chip;
use product::Halves;

pub mod fp8;
mod gate;
mod linear;
mod live;
mod product;

axes![Term = 2];
/// A chip's two clusters, each projecting half of the intermediate features.
type Cluster<Intermediate> = m![Intermediate / (Intermediate / 2)];
/// A slice's output features: its cluster's half over the cluster's 32 output blocks.
type Out<Intermediate> = m![Intermediate % (Intermediate / 64)];
/// A cluster's output blocks.
type Blocks<Intermediate> = m![Intermediate % (Intermediate / 2) / (Intermediate / 64)];
type K<Hidden> = m![Hidden % 128];
type Partition<Hidden, Intermediate> = m![{ Blocks<Intermediate> }, Hidden % 1024 / 128];
type Slice<Intermediate, Rows> = m![{ Blocks<Intermediate> }, Rows / (Rows / 8)];
type Projection<Intermediate, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<Intermediate>,
    Slice<Intermediate, Rows>,
    m![Rows % (Rows / 8), { Out<Intermediate> }],
>;
