use super::*;
use crate::compose::Space;
use crate::ops::linear::contraction::resident::Mapping;

impl AxisName for Whole {
    const NAME: Ident = Rows::NAME;
    const SIZE: usize = Rows::SIZE;
}

impl Space for Whole {
    type Envelope = Rows;
}

impl AxisName for Half {
    const NAME: Ident = Rows::NAME;
    const SIZE: usize = Rows::SIZE / 2;
}

impl Space for Half {
    type Envelope = Rows;
}

impl Mapping for Whole {
    type Cluster = Cluster;
    type Slices = Slices;
    type Lanes = m![Rows % 8];
    type Input = m![Hidden % 256 / 16, Rows / 8, Hidden % 16];
    type Weight = m![Vocabulary % 128, Hidden % 256];
    type Feed = Feed;
    type Packet = Packet;
    type Flits = m![{ Feed }, Vocabulary % 2];
    type Flit = m![Hidden % 16];
    type Steps = Steps;
    type Contracted = m![Vocabulary % 2];
    type Sums = Sums;
    type Groups = m![{ Sums }, Vocabulary % 2];
    type Transpose = m![
        Vocabulary % 128 / 4,
        Rows / 16,
        Vocabulary % 4 / 2,
        Rows % 16
    ];
    type Columns = m![Vocabulary % 2 # 8];
    type Output = m![Rows, Vocabulary % 128];
}

impl Mapping for Half {
    type Cluster = Cluster;
    type Slices = Slices;
    type Lanes = m![Rows % 8];
    type Input = m![Hidden % 256 / 16, Rows / 8, Hidden % 16];
    type Weight = m![Vocabulary % 128, Hidden % 256];
    type Feed = HalfFeed;
    type Packet = Packet;
    type Flits = m![{ HalfFeed }, Vocabulary % 2];
    type Flit = m![Hidden % 16];
    type Steps = HalfSteps;
    type Contracted = m![Vocabulary % 2];
    type Sums = HalfSums;
    type Groups = m![{ HalfSums }, Vocabulary % 2];
    type Transpose = m![
        Vocabulary % 128 / 4,
        Rows % 32 / 16,
        Vocabulary % 4 / 2,
        Rows % 16
    ];
    type Columns = m![Vocabulary % 2 # 8];
    type Output = m![Rows % 32, Vocabulary % 128];
}
