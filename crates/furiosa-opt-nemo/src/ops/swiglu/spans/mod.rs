use furiosa_opt_std::prelude::*;

use crate::ops::layout::axes::Chip;
use crate::ops::linear::Groups;

pub mod fp8;
mod gate;
mod product;

pub(crate) use fp8::Span;

axes![Term = 2];
type Cluster<Intermediate> = m![Intermediate / 6144];
type Out<Intermediate> = m![Intermediate % 192];
type K<Hidden> = m![Hidden % 128];
type Partition<Hidden, Intermediate> = m![Intermediate % 6144 / 192, Hidden % 1024 / 128];
type Slice<Intermediate, Rows> = m![Intermediate % 6144 / 192, Rows / (Rows / 8)];
type Projection<Intermediate, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<Intermediate>,
    Slice<Intermediate, Rows>,
    m![Rows % (Rows / 8), { Out<Intermediate> }],
>;
