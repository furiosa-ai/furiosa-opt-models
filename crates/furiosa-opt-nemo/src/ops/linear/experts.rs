//! Per-expert matrices selected by device-resident routing indices.

use crate::Module;
use crate::ops::layout::axes::Chip;
use furiosa_opt_std::prelude::*;

/// Device-resident expert index for each selected row.
pub type Indices<Selected> = HbmTensor<i32, Chip, m![Selected]>;

/// One contiguous BF16 input row shared by selected experts.
pub type Vector<Width> = HbmTensor<bf16, Chip, m![Width]>;

/// Contiguous BF16 rows, one per selected expert.
pub type Rows<Width, Selected> = HbmTensor<bf16, Chip, m![Selected, Width]>;

/// `output[k] = weights[indices[k]] @ input[k]`, or one shared input row; 0 < K <= COUNT.
#[derive(DeviceSend)]
pub struct Experts<In: AxisName, Out: AxisName, const COUNT: usize> {
    /// Expert matrices in [Out, In] order.
    pub weights: [HbmTensor<bf16, Chip, m![Out, In]>; COUNT],
}

impl<I: AxisName, O: AxisName, K: AxisName, const N: usize>
    Module<(&mut Device, (&Vector<I>, &Indices<K>))> for Experts<I, O, N>
{
    type Output = Rows<O, K>;

    fn forward(&self, _: (&mut Device, (&Vector<I>, &Indices<K>))) -> Self::Output {
        const { assert!(K::SIZE > 0 && K::SIZE <= N) };
        todo!("NPU selected expert matrices: output[k] = weights[indices[k]] @ input")
    }
}

impl<I: AxisName, O: AxisName, K: AxisName, const N: usize>
    Module<(&mut Device, (&Rows<I, K>, &Indices<K>))> for Experts<I, O, N>
{
    type Output = Rows<O, K>;

    fn forward(&self, _: (&mut Device, (&Rows<I, K>, &Indices<K>))) -> Self::Output {
        const { assert!(K::SIZE > 0 && K::SIZE <= N) };
        todo!("NPU selected expert matrices: output[k] = weights[indices[k]] @ input[k]")
    }
}
