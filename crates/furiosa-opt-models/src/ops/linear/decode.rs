use super::kernel::Tiling;
use furiosa_opt_std::prelude::*;

pub(super) use early::Coded;
mod channel;
mod early;

pub(crate) trait Matrix<X: M = m![1]> {
    type In: AxisName;
    type Out: AxisName;

    fn tile<T: Tiling<Self::In, Self::Out>>(
        &self,
        device: &mut Device,
        at: usize,
    ) -> DmTensor<bf16, X, T::Cluster, T::Slices, m![{ T::Rows }, { T::Inputs }]>;
}

impl<X: M, T: Matrix<X>> Matrix<X> for &T {
    type In = T::In;
    type Out = T::Out;

    fn tile<S: Tiling<Self::In, Self::Out>>(
        &self,
        device: &mut Device,
        at: usize,
    ) -> DmTensor<bf16, X, S::Cluster, S::Slices, m![{ S::Rows }, { S::Inputs }]> {
        (**self).tile::<S>(device, at)
    }
}

impl<In: AxisName, Out: AxisName, X: M> Matrix<X> for HbmTensor<bf16, X, m![Out, In]> {
    type In = In;
    type Out = Out;

    fn tile<T: Tiling<In, Out>>(
        &self,
        device: &mut Device,
        at: usize,
    ) -> DmTensor<bf16, X, T::Cluster, T::Slices, m![{ T::Rows }, { T::Inputs }]> {
        T::to_dm(device, self.view(), at)
    }
}

impl<In: AxisName, Out: AxisName, X: M> Matrix<X> for HbmTensorView<'_, bf16, X, m![Out, In]> {
    type In = In;
    type Out = Out;

    fn tile<T: Tiling<In, Out>>(
        &self,
        device: &mut Device,
        at: usize,
    ) -> DmTensor<bf16, X, T::Cluster, T::Slices, m![{ T::Rows }, { T::Inputs }]> {
        T::to_dm(device, self.clone(), at)
    }
}
