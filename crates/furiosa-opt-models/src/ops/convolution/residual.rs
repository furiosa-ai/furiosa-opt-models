//! Elementwise sums of time-major frames: residual updates and the average of parallel blocks.
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use super::frames::{self, Blocked, Edge, Frames, LEAD, Pads, Row, Window, Zero};
use crate::Op;
use crate::load::Load;

axes![Term = 2];

/// `x + y` on every row of each block.
impl<C: AxisName, S: Window, X: M> Op<(&Frames<f32, C, S, X>, &Frames<f32, C, S, X>, ())> for Sum {
    type Output = Blocked<f32, C, S, X>;

    fn forward(
        &self,
        device: &mut Device,
        (x, y, ()): (&Frames<f32, C, S, X>, &Frames<f32, C, S, X>, ()),
    ) -> Blocked<f32, C, S, X> {
        let x = frames::FrameAxis::<S>::new().forward(&mut *device, (x, LEAD));
        let y = frames::FrameAxis::<S>::new().forward(&mut *device, (y, LEAD));
        self.forward(device, ((&x, &y), 1.0))
    }
}

/// `(x + y) * scale` on every row of each block.
impl<C: AxisName, S: Window, X: M> Op<((&Blocked<f32, C, S, X>, &Blocked<f32, C, S, X>), f32)>
    for Sum
{
    type Output = Blocked<f32, C, S, X>;

    fn forward(
        &self,
        device: &mut Device,
        ((x, y), scale): ((&Blocked<f32, C, S, X>, &Blocked<f32, C, S, X>), f32),
    ) -> Blocked<f32, C, S, X> {
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(x.view(), y.view())
            .fetch::<m![{ Row<S> }, C / 8, Term], m![C % 8]>()
            .collect::<m![{ Row<S> }, C / 8, Term], m![C % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, m![{ Row<S> }, C / 8]>()
            .vector_narrow_split::<m![{ Row<S> }, C / 4], m![C % 4]>()
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
            .vector_widen_concat::<m![{ Row<S> }, C / 8], m![C % 8]>()
            .vector_final()
            .commit_trim::<m![C % 8]>()
            .commit()
    }
}

/// `x + y`: a residual update; the result's padding rows repeat its edge samples, for the
/// activation that follows.
#[derive(Clone, Copy, Debug, Default, DeviceSend, Load)]
pub struct Sum;

impl<'x, 'y, C: AxisName, S: Window, X: M> Op<(&'x Frames<f32, C, S, X>, &'y Frames<f32, C, S, X>)>
    for Sum
{
    type Output = Frames<f32, C, S, X>;

    fn forward(
        &self,
        device: &mut Device,
        (x, y): (&'x Frames<f32, C, S, X>, &'y Frames<f32, C, S, X>),
    ) -> Self::Output {
        let rows = self.forward(&mut *device, (x, y, ()));
        let mut output = Frames::<f32, C, S, X>::new();
        frames::FrameAxis::<S>::new().forward(&mut *device, (&rows, &mut output));
        Edge::fill(device, &mut output);
        output
    }
}

/// The average of three parallel blocks' outputs, its padding rows prepared for `P`'s kind of
/// consumer.
#[derive(Clone, Copy, Debug, Default, DeviceSend, Load)]
pub struct Mean<P = Zero>(PhantomData<P>);

impl<'x, C: AxisName, S: Window, P: Pads, X: M>
    Op<&'x (
        Frames<f32, C, S, X>,
        (Frames<f32, C, S, X>, Frames<f32, C, S, X>),
    )> for Mean<P>
{
    type Output = Frames<f32, C, S, X>;

    fn forward(
        &self,
        device: &mut Device,
        (a, (b, c)): &'x (
            Frames<f32, C, S, X>,
            (Frames<f32, C, S, X>, Frames<f32, C, S, X>),
        ),
    ) -> Self::Output {
        let rows = {
            let partial = Sum.forward(&mut *device, (a, b, ()));
            let c = frames::FrameAxis::<S>::new().forward(&mut *device, (c, LEAD));
            Sum.forward(&mut *device, ((&partial, &c), 1.0 / 3.0))
        };
        let mut output = Frames::<f32, C, S, X>::new();
        frames::FrameAxis::<S>::new().forward(&mut *device, (&rows, &mut output));
        P::fill(device, &mut output);
        output
    }
}
