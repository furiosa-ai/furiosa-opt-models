use furiosa_opt_std::prelude::*;

use crate::Module;
use crate::compose::Points;
use crate::ops::add::clusters as phases;
use crate::ops::layout::axes::Chip;
use crate::ops::linear::Affine;
use crate::ops::normalization::span::Owned;

pub mod fp8;
pub mod packed;
mod waves;

type Cluster<QWidth> = m![QWidth / 2048];

pub(crate) fn wave_rows<QWidth: AxisName, Capacity: AxisName, Reach: AxisName>(
    attended: &HbmTensor<bf16, Chip, m![Reach, QWidth]>,
    start: usize,
) -> HbmTensorView<
    '_,
    bf16,
    Chip,
    m![Capacity % 32 / 8 # (Reach / 32), Capacity / (Capacity / 4), Capacity % 8, QWidth],
> {
    // SAFETY: the wave's memory rows are its eight-row groups of every quarter, in order.
    unsafe {
        attended
            .view()
            .tile::<m![Reach], 128, m![Reach = 128 # Reach, QWidth]>(start)
            .reshape::<Chip, m![Capacity % 32 / 8 # (Reach / 32), Capacity / (Capacity / 4),
                Capacity % 8, QWidth]>()
    }
}

pub(crate) struct Land<'a, QWidth: AxisName, Capacity: AxisName, Reach: AxisName> {
    pub attended: &'a HbmTensor<bf16, Chip, m![Reach, QWidth]>,
    pub start: usize,
    pub live: (usize, usize),
    pub capacity: core::marker::PhantomData<Capacity>,
}

pub(crate) struct Project<'a, Hidden: AxisName, QWidth: AxisName, Capacity: AxisName, P> {
    pub landing: &'a fp8::Landing8<Hidden, QWidth, Capacity>,
    pub points: P,
    pub live: (usize, usize),
}

pub(crate) struct Load<'a, Hidden: AxisName, QWidth: AxisName>(
    pub &'a fp8::Output8<Hidden, QWidth>,
    pub &'a mut fp8::Staged8<Hidden, QWidth>,
);

impl<Hidden: AxisName, QWidth: AxisName> Points<Device> for Load<'_, Hidden, QWidth> {
    type Next = Self;
    type Output = ();

    fn at(self, _: &mut Device) -> Self {
        self
    }

    fn between(&mut self, context: &mut Device) {
        self.0.load(context, self.1);
    }

    fn finish(self, _: &mut Device) {}
}
