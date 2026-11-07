use crate::Module;
use crate::ops::attention::rope::SplitHalf;
use crate::ops::layout::axes::Chip;
use crate::ops::linear::query::Tile;
use furiosa_opt_std::prelude::*;

mod gather;
pub(crate) use gather::Gather;

axes![InputHalf = 2, OutputHalf = 2];

pub type Table<Position, Half> = HbmTensor<bf16, Chip, m![Position, Half, OutputHalf, InputHalf]>;
pub(crate) type Rotation<S, Half> = HbmTensor<bf16, Chip, m![S, Tile, OutputHalf, InputHalf, Half]>;
pub(crate) type Coefficients<C, P, Half> =
    DmTensor<bf16, Chip, C, P, m![OutputHalf, Tile % 8, InputHalf, Half]>;
pub(crate) type Registers<C, P, Half> =
    TrfTensor<bf16, Chip, C, P, m![OutputHalf], m![Tile % 8, InputHalf, Half]>;

pub(crate) trait Coefficient<C: M, P: M, Half: AxisName> {
    fn view(&self)
    -> DmTensorView<'_, bf16, Chip, C, P, m![OutputHalf, Tile % 8, InputHalf, Half]>;
}

impl<C: M, P: M, Half: AxisName> Coefficient<C, P, Half> for Coefficients<C, P, Half> {
    fn view(
        &self,
    ) -> DmTensorView<'_, bf16, Chip, C, P, m![OutputHalf, Tile % 8, InputHalf, Half]> {
        self.view()
    }
}

impl<C: M, P: M, Half: AxisName> Coefficient<C, P, Half>
    for DmTensorView<'_, bf16, Chip, C, P, m![OutputHalf, Tile % 8, InputHalf, Half]>
{
    fn view(
        &self,
    ) -> DmTensorView<'_, bf16, Chip, C, P, m![OutputHalf, Tile % 8, InputHalf, Half]> {
        self.clone()
    }
}

impl<Head: AxisName, F> SplitHalf<Head, F> {
    pub(crate) fn stage<Half: AxisName, C: M, P: M>(
        &self,
        context: &mut Device,
    ) -> SplitHalf<Head, Registers<C, P, Half>>
    where
        F: Coefficient<C, P, Half>,
    {
        let coefficients: Registers<C, P, Half> = context
            .sub
            .begin(self.frequency.view())
            .fetch::<m![OutputHalf, Tile % 8, InputHalf, Half / 16], m![Half % 16]>()
            .collect::<m![OutputHalf, Tile % 8, InputHalf, Half / 16], m![Half % 16]>()
            .to_trf();
        SplitHalf {
            frequency: coefficients,
            head: core::marker::PhantomData,
        }
    }
}
