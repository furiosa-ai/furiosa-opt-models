//! Contiguous HBM rows normalized through a feature-sliced normalization kernel.

use super::Spread;
use crate::Module;
use crate::load::Load;
use crate::tensor::memory::residency::Residency;
use furiosa_opt_std::prelude::*;

/// Read BF16 rows into the normalizer's ordinary feature slices, then store its BF16 result.
/// This selects that kernel for short row sets, such as a decoder's query or key heads, without
/// requiring the whole row tiles of the HBM streaming implementation. The child owns the
/// normalizer's unchanged parameters; loading, borrowing and residency retain their names.
#[derive(DeviceSend, Load, Residency)]
#[residency(children(Norm))]
pub struct PerRow<Norm>(#[load(rename = "")] pub Norm);

impl<I, Norm> !crate::Op<I> for PerRow<Norm> {}

impl<R: AxisName, W: AxisName, C: M, Norm, E: Copy>
    Module<&HbmTensor<bf16, C, m![R, W]>, E, &mut ()> for PerRow<Norm>
where
    Norm: for<'x, 's> Module<&'x Spread<W, R, C>, E, &'s mut (), Output = Spread<W, R, C>>,
{
    type Output = HbmTensor<bf16, C, m![R, W]>;

    fn forward(
        &self,
        device: &mut Device,
        env: E,
        _: &mut (),
        input: &HbmTensor<bf16, C, m![R, W]>,
    ) -> Self::Output {
        let input: Spread<W, R, C> = input.to_dm(&mut device.tdma);
        let output = self.0.forward(&mut *device, env, &mut (), &input);
        output.to_hbm(&mut device.tdma)
    }
}
