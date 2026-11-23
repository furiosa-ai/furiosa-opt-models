//! One-dimensional text positions through the shared rotary coefficient calculation.

use super::{Chips, Head};
use crate::Module;
use crate::load::{Context, Load, LoadError};
use crate::ops::rotary::{Coefficients, PairFrequencies, Rotation};
use crate::tensor::{
    axis::Single,
    memory::{Hbm, Memory, residency::Residency},
    parallel::Parallel,
};
use core::borrow::Borrow;
use furiosa_opt_std::prelude::*;

/// Prepared constants of the encoder's full-head rotary position calculation.
pub type Positions<X = Hbm> = Positioning<
    Chips<Parallel<4>>,
    <X as Memory>::Tensor<f32, Chips<Parallel<4>>, m![Head]>,
    <X as Memory>::Tensor<f32, Chips<Parallel<4>>, m![Single, Head]>,
>;

/// The same coefficient generator used by the shared decoder, with its frequency row.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T, F))]
pub struct Positioning<C: M, T, F> {
    pub coefficients: Coefficients<Head, C, T>,
    pub frequency: F,
}

impl<I, C: M, T, F> !crate::Op<I> for Positioning<C, T, F> {}

impl<'a, C: M + 'static, T, F> Load<&'a PairFrequencies<Head>> for Positioning<C, T, F>
where
    T: Load<HostTensor<f32, m![{ C }, { m![Head] }]>>,
    F: Load<HostTensor<f32, m![{ C }, { m![Single, Head] }]>>,
{
    fn load(
        context: &mut Context<'_>,
        frequencies: &'a PairFrequencies<Head>,
    ) -> Result<Self, LoadError> {
        let coefficients =
            context.field("coefficients", "", |cx| Coefficients::load(cx, frequencies))?;
        let frequency = context.field("frequency", "", |cx| {
            F::load(
                cx,
                HostTensor::from_vec(frequencies.coordinates().repeat(C::SIZE)),
            )
        })?;
        Ok(Self {
            coefficients,
            frequency,
        })
    }
}

impl<C: M, T, F, E> Module<&HbmTensor<f32, C, m![Single, Head]>, E, &mut ()>
    for Positioning<C, T, F>
where
    T: Borrow<HbmTensor<f32, C, m![Head]>>,
    F: Borrow<HbmTensor<f32, C, m![Single, Head]>>,
{
    type Output = Rotation<Head, C>;

    fn forward(
        &self,
        device: &mut Device,
        _: E,
        _: &mut (),
        position: &HbmTensor<f32, C, m![Single, Head]>,
    ) -> Self::Output {
        crate::Op::forward(
            &self.coefficients,
            device,
            (self.frequency.borrow(), position),
        )
    }
}
