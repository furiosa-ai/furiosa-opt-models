//! Calculate one shared attention context before traversing its independent layer pools.

use crate::ops::attention::cache::Pool;
use crate::{DeviceModule, Module};
use furiosa_opt_std::prelude::{AxisName, Device, DeviceSend, M, Scalar};

/// Computes shared attention contexts once, then traverses the corresponding layer pools.
#[derive(DeviceSend)]
pub struct Indexed<Index, Body> {
    /// Computes each pool's context from the mutable invocation state.
    pub index: Index,
    /// Consumes input, layer pools and the computed contexts.
    pub body: Body,
}

impl<T, C, P, O, H: AxisName, W: AxisName, S: Scalar, Chip: M, Index, Body, const N: usize>
    Module<(&mut Device, &mut (T, C, &mut Pool<H, W, S, Chip, N>))> for Indexed<Index, Body>
where
    Index: for<'c> DeviceModule<&'c mut C, Output = P>,
    Body:
        for<'i, 'a, 'b> DeviceModule<(&'i T, (&'a mut Pool<H, W, S, Chip, N>, &'b P)), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        (device, (input, context, pools)): (&mut Device, &mut (T, C, &mut Pool<H, W, S, Chip, N>)),
    ) -> O {
        let context = self.index.forward((&mut *device, context));
        self.body
            .forward((device, (&*input, (&mut **pools, &context))))
    }
}

impl<
    T,
    C,
    P,
    Q,
    O,
    H: AxisName,
    W: AxisName,
    J: AxisName,
    E: AxisName,
    S: Scalar,
    Chip: M,
    A,
    B,
    Body,
    const N: usize,
    const SECOND: usize,
>
    Module<(
        &mut Device,
        &mut (
            T,
            C,
            (
                &mut Pool<H, W, S, Chip, N>,
                &mut Pool<J, E, S, Chip, SECOND>,
            ),
        ),
    )> for Indexed<(A, B), Body>
where
    A: for<'c> DeviceModule<&'c mut C, Output = P>,
    B: for<'c> DeviceModule<&'c mut C, Output = Q>,
    Body: for<'i, 'c, 'b> DeviceModule<
            (
                &'i T,
                (
                    (&'b mut Pool<H, W, S, Chip, N>, &'c P),
                    (&'b mut Pool<J, E, S, Chip, SECOND>, &'c Q),
                ),
            ),
            Output = O,
        >,
{
    type Output = O;

    fn forward(
        &self,
        (device, (input, context, (first, second))): (
            &mut Device,
            &mut (
                T,
                C,
                (
                    &mut Pool<H, W, S, Chip, N>,
                    &mut Pool<J, E, S, Chip, SECOND>,
                ),
            ),
        ),
    ) -> O {
        let a = self.index.0.forward((&mut *device, &mut *context));
        let b = self.index.1.forward((&mut *device, context));
        self.body
            .forward((device, (&*input, ((&mut **first, &a), (&mut **second, &b)))))
    }
}
