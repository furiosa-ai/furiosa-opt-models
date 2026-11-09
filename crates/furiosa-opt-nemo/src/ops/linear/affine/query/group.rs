use furiosa_opt_std::prelude::*;

pub(crate) trait Group: AxisName {
    type Copies: M;
    type Slot: M;
    type Rows<All: AxisName>: M;
}
