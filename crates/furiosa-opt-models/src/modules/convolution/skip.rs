//! U-Net skip blocks: down stages keep their features and up stages join them back, newest first.

use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Retains the input and first output while two blocks produce the down-stage output.
#[derive(DeviceSend)]
pub struct Down<First, Second>(pub First, pub Second);

/// The state holds each block's slot.
impl<'s, 'x, X, Y, Z, FirstState, SecondState, First, Second, Env: Copy>
    Module<&'x X, Env, &'s mut (FirstState, SecondState)> for Down<First, Second>
where
    First: Module<&'x X, Env, &'s mut FirstState, Output = Y>,
    Second: for<'y> Module<&'y Y, Env, &'s mut SecondState, Output = Z>,
{
    type Output = (Z, (Y, &'x X));

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (first, second): &'s mut (FirstState, SecondState),
        input: &'x X,
    ) -> Self::Output {
        let hidden = self.0.forward(&mut *device, env, first, input);
        let output = self.1.forward(device, env, second, &hidden);
        (output, (hidden, input))
    }
}

/// Joins three retained features to successive blocks, newest feature first.
#[derive(DeviceSend)]
pub struct Up<First, Second, Third, Join> {
    pub layers: (First, Second, Third),
    pub join: Join,
}

/// The input is the hidden value beside the down stage's output and retained features; the state
/// holds each block's slot.
impl<
    's,
    'x,
    'y,
    X,
    Y,
    Z,
    I,
    A,
    B,
    C,
    U,
    V,
    O,
    FirstState,
    SecondState,
    ThirdState,
    First,
    Second,
    Third,
    Join,
    Env: Copy,
> Module<(I, (&'y Z, (Y, &'x X))), Env, &'s mut (FirstState, SecondState, ThirdState)>
    for Up<First, Second, Third, Join>
where
    Join: for<'d, 'a, 'b> Module<(&'a I, &'b Z), Env, &'d mut (), Output = A>
        + for<'d, 'a, 'b> Module<(&'a U, &'b Y), Env, &'d mut (), Output = B>
        + for<'d, 'a, 'b> Module<(&'a V, &'b X), Env, &'d mut (), Output = C>,
    First: for<'a> Module<&'a A, Env, &'s mut FirstState, Output = U>,
    Second: for<'a> Module<&'a B, Env, &'s mut SecondState, Output = V>,
    Third: for<'a> Module<&'a C, Env, &'s mut ThirdState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (first, second, third): &'s mut (FirstState, SecondState, ThirdState),
        (hidden, (last, (previous, input))): (I, (&'y Z, (Y, &'x X))),
    ) -> O {
        let hidden = self
            .join
            .forward(&mut *device, env, &mut (), (&hidden, last));
        let hidden = self.layers.0.forward(&mut *device, env, first, &hidden);
        let hidden = self
            .join
            .forward(&mut *device, env, &mut (), (&hidden, &previous));
        let hidden = self.layers.1.forward(&mut *device, env, second, &hidden);
        let hidden = self
            .join
            .forward(&mut *device, env, &mut (), (&hidden, input));
        self.layers.2.forward(device, env, third, &hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::super::UNet;
    use super::*;
    use crate::Module;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;

    /// A block that scales its input and adds the condition and its slot's offset.
    struct Affine(i32);

    impl<E> Module<&i32, (&i32, E), &mut i32> for Affine {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            (condition, _): (&i32, E),
            offset: &mut i32,
            input: &i32,
        ) -> i32 {
            input * self.0 + condition + *offset
        }
    }

    struct Join;

    impl<Env> Module<(&i32, &i32), Env, &mut ()> for Join {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), (hidden, skip): (&i32, &i32)) -> i32 {
            hidden * 10 + skip
        }
    }

    /// U-Net stages consume retained features in reverse order with their own weights and slots.
    #[test]
    fn consumes_reverse_skips() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let model = UNet {
            down: Down(Affine(2), Affine(3)),
            middle: Affine(4),
            up: Up {
                layers: (Affine(5), Affine(6), Affine(7)),
                join: Join,
            },
        };
        let output = model.forward(device, (&2, ()), &mut ((3, 4), 5, (6, 7, 8)), &1);
        assert_eq!(output, 24_754_187);
    }
}
