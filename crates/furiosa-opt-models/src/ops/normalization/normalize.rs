//! `x / max(norm(x), epsilon)` per row: the Euclidean norm, as `torch.nn.functional.normalize`.

use crate::Op;
use furiosa_opt_std::prelude::*;

/// Divide each row by max(its Euclidean norm, epsilon); zero preserves unclamped division.
/// The caller supplies a finite, nonnegative epsilon scalar.
#[derive(DeviceSend)]
pub struct Normalize {
    /// Default lower bound for the row norm; an explicit input epsilon overrides it.
    pub epsilon: HbmTensor<f32, m![1], m![1]>,
}

impl<W: AxisName, N: AxisName>
    Op<(
        &HbmTensor<f32, m![1], m![N, W]>,
        &HbmTensor<f32, m![1], m![1]>,
    )> for Normalize
{
    type Output = HbmTensor<f32, m![1], m![N, W]>;

    fn forward(
        &self,
        _: &mut Device,
        _: (
            &HbmTensor<f32, m![1], m![N, W]>,
            &HbmTensor<f32, m![1], m![1]>,
        ),
    ) -> Self::Output {
        todo!("NPU row-wise L2 normalization with a caller-provided nonnegative epsilon")
    }
}

impl<W: AxisName, N: AxisName> Op<&HbmTensor<f32, m![1], m![N, W]>> for Normalize {
    type Output = HbmTensor<f32, m![1], m![N, W]>;

    fn forward(&self, device: &mut Device, rows: &HbmTensor<f32, m![1], m![N, W]>) -> Self::Output {
        self.forward(device, (rows, &self.epsilon))
    }
}

impl<W: AxisName, N: AxisName> Op<&mut HbmTensor<f32, m![1], m![N, W]>> for Normalize {
    type Output = HbmTensor<f32, m![1], m![N, W]>;

    fn forward(
        &self,
        device: &mut Device,
        rows: &mut HbmTensor<f32, m![1], m![N, W]>,
    ) -> Self::Output {
        self.forward(device, &*rows)
    }
}

impl<W: AxisName, N: AxisName>
    Op<&mut (
        HbmTensor<f32, m![1], m![N, W]>,
        HbmTensor<f32, m![1], m![1]>,
    )> for Normalize
{
    type Output = HbmTensor<f32, m![1], m![N, W]>;

    fn forward(
        &self,
        device: &mut Device,
        (rows, epsilon): &mut (
            HbmTensor<f32, m![1], m![N, W]>,
            HbmTensor<f32, m![1], m![1]>,
        ),
    ) -> Self::Output {
        self.forward(device, (&*rows, &*epsilon))
    }
}
