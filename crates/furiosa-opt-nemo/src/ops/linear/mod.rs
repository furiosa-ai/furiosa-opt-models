//! BF16 row projections and affine parameter forms.

mod affine;
pub mod contraction;
pub use affine::{mlp, output, pair, query};
pub mod fp8;
pub mod nvfp4;
mod projection;
pub(crate) mod rows;

pub use affine::Affine;

use furiosa_opt_std::prelude::*;

use crate::Module;

/// Head-major rows in HBM, each head's rows in order, project block by block into HBM.
impl<'d, 'x, In: AxisName, Out: AxisName, H: AxisName, Rows: AxisName, Chips: M>
    Module<(
        &'d mut Device,
        &'x HbmTensor<bf16, Chips, m![In / ({ Symbol<H> }), Rows, In % ({ Symbol<H> })]>,
    )> for Biased<In, Out, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Rows, Out]>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            &'x HbmTensor<bf16, Chips, m![In / ({ Symbol<H> }), Rows, In % ({ Symbol<H> })]>,
        ),
    ) -> HbmTensor<bf16, Chips, m![Rows, Out]> {
        let mut output = HbmTensor::new();
        rows::Projection::new(&self.weight, &self.bias).forward((device, (input, &mut output)));
        output
    }
}

/// BF16 coefficients with an FP32 bias.
pub type Biased<In, Out, Chips = m![1]> =
    Affine<HbmTensor<bf16, Chips, m![Out, In]>, HbmTensor<f32, Chips, m![Out]>>;

/// Rows in HBM project block by block into HBM, each output's bias added to its FP32 sum before
/// the one rounding.
impl<'d, 'x, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M>
    Module<(&'d mut Device, &'x HbmTensor<bf16, Chips, m![Rows, In]>)> for Biased<In, Out, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Rows, Out]>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x HbmTensor<bf16, Chips, m![Rows, In]>),
    ) -> HbmTensor<bf16, Chips, m![Rows, Out]> {
        let mut output = HbmTensor::new();
        rows::Projection::new(&self.weight, &self.bias).forward((device, (input, &mut output)));
        output
    }
}
