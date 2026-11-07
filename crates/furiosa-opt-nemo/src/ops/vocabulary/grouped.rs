use furiosa_opt_std::prelude::*;

pub(crate) trait Grouped: AxisName {
    type Short: Pick;
    type One: Pick;
}

impl<R: AxisName> Grouped for R {
    type Short = Short<R>;
    type One = One<R>;
}

pub(crate) trait Pick {
    fn pick<A: Arms>(context: &mut Device, arms: A) -> A::Output;
}

pub(crate) trait Arms {
    type Output;
    fn then(self, context: &mut Device) -> Self::Output;
    fn otherwise(self, context: &mut Device) -> Self::Output;
}

pub(crate) struct Short<R>(core::marker::PhantomData<R>);
pub(crate) struct One<R>(core::marker::PhantomData<R>);

impl<R: AxisName> Pick for Short<R> {
    fn pick<A: Arms>(context: &mut Device, arms: A) -> A::Output {
        if const { R::SIZE <= 64 } {
            arms.then(context)
        } else {
            arms.otherwise(context)
        }
    }
}

impl<R: AxisName> Pick for One<R> {
    fn pick<A: Arms>(context: &mut Device, arms: A) -> A::Output {
        if const { R::SIZE == 1 } {
            arms.then(context)
        } else {
            arms.otherwise(context)
        }
    }
}
