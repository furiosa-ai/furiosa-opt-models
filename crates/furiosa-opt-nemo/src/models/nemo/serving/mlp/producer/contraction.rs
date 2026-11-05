use furiosa_opt_std::prelude::*;

use super::{
    Branch, Cluster, Feed, Flit, Groups, HalfGroups, HalfSlice, HalfSteps, HalfSums, Hidden,
    Intermediate, Lanes, PairFeed, PairGroups, PairSteps, PairSums, Rows, Slice, Steps, Sums,
};

axes![Lane = 8, Both = 16, Prefix = 32];
pub(in crate::models::nemo::serving::mlp) type Single = crate::compose::Tile<Rows, Lane>;
pub(super) type Paired = crate::compose::Tile<Rows, Both>;
pub(super) type Partial = crate::compose::Tile<Rows, Prefix>;

macro_rules! mapping {
    ($space:ty, $weight:ty, $feed:ty, $steps:ty, $sums:ty, $groups:ty, $reduced:ty, $output:ty) => {
        impl crate::ops::linear::contraction::Mapping for $space {
            type Cluster = Cluster;
            type Slices = Slice;
            type Reduced = $reduced;
            type Input = m![
                Hidden % 160 / 32,
                Hidden % 640 / 160,
                Rows / 16,
                Hidden % 32
            ];
            type Lanes = Lanes;
            type Weight = $weight;
            type Feed = $feed;
            type Packet = Flit;
            type Flits = m![{ $feed }, Intermediate % 2];
            type Flit = m![Hidden % 32];
            type Steps = $steps;
            type Contracted = m![Intermediate % 2];
            type Sums = $sums;
            type Groups = $groups;
            type Output = $output;
        }
    };
}

mapping!(
    Single,
    m![Branch = 1 # 2, Intermediate % 56, Hidden % 640],
    Feed,
    Steps,
    Sums,
    Groups,
    m![Intermediate % 3584 / 56, Rows / 16],
    m![Intermediate % 56]
);
mapping!(
    Paired,
    m![Branch, Intermediate % 56, Hidden % 640],
    PairFeed,
    PairSteps,
    PairSums,
    PairGroups,
    m![Intermediate % 3584 / 56, Rows / 16],
    m![Branch, Intermediate % 56]
);
mapping!(
    Partial,
    m![Branch = 1 # 2, Intermediate % 56, Hidden % 640],
    Feed,
    HalfSteps,
    HalfSums,
    HalfGroups,
    HalfSlice,
    m![Intermediate % 56]
);
