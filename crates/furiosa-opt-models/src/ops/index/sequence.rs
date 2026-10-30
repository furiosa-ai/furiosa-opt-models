//! Concatenation and selection of contiguous token rows and feature columns.
use crate::Op;
use crate::tensor::image::Rows;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct Sum<A, B>(PhantomData<(A, B)>);

impl<A: AxisName, B: AxisName> AxisName for Sum<A, B> {
    const NAME: Ident = Ident::new("Sum");
    const SIZE: usize = A::SIZE + B::SIZE;
}

#[derive(DeviceSend)]
pub struct Concat;

impl<W: AxisName, A: AxisName, B: AxisName> Op<(&Rows<W, A>, &Rows<W, B>)> for Concat {
    type Output = Rows<W, Sum<A, B>>;

    fn forward(&self, _: &mut Device, _: (&Rows<W, A>, &Rows<W, B>)) -> Self::Output {
        todo!("NPU concatenation of contiguous token rows, first input before second")
    }
}

#[derive(DeviceSend)]
pub struct Split<A, B>(pub PhantomData<(A, B)>);

impl<A, B> Default for Split<A, B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<W: AxisName, A: AxisName, B: AxisName> Op<&Rows<W, Sum<A, B>>> for Split<A, B> {
    type Output = (Rows<W, A>, Rows<W, B>);

    fn forward(&self, _: &mut Device, _: &Rows<W, Sum<A, B>>) -> Self::Output {
        todo!("NPU split of contiguous token rows at the first sequence length")
    }
}

#[derive(DeviceSend)]
pub struct Suffix<A, B>(pub PhantomData<(A, B)>);

impl<A, B> Default for Suffix<A, B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<W: AxisName, A: AxisName, B: AxisName> Op<(&Rows<W, Sum<A, B>>, ())> for Suffix<A, B> {
    type Output = Rows<W, B>;

    fn forward(&self, _: &mut Device, _: (&Rows<W, Sum<A, B>>, ())) -> Self::Output {
        todo!("NPU selection of the rows following a statically sized prefix")
    }
}

#[derive(DeviceSend)]
pub struct Columns<Output>(pub PhantomData<Output>);

impl<Output> Default for Columns<Output> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<A: AxisName, B: AxisName, R: AxisName, Output: AxisName> Op<(&Rows<A, R>, &Rows<B, R>)>
    for Columns<Output>
{
    type Output = Rows<Output, R>;

    fn forward(&self, _: &mut Device, _: (&Rows<A, R>, &Rows<B, R>)) -> Self::Output {
        const { assert!(Output::SIZE == A::SIZE + B::SIZE) };
        todo!("NPU feature concatenation, first input columns before second")
    }
}
