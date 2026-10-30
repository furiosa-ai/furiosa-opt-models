//! The selective scan: a state-space recurrence over live rows with caller-owned state.

use crate::Op;
use furiosa_opt_std::prelude::*;

/// Rows of features, a row a token.
pub(crate) type Rows<Width, Count, D = bf16> = HbmTensor<D, m![1], m![Count, Width]>;

/// Only the leading live rows advance either recurrent history.
pub type Length = HbmTensor<i32, m![1], m![1]>;

/// HF's selective scan over `(x, dt, B, C)`: `h = exp(dt A) h + dt B x` and `y = C h + D x`, with
/// `A = -exp(decay)` and the time steps `dt` given. History and output remain FP32 until gating.
#[derive(DeviceSend)]
pub struct SelectiveScan<Decay, Skip> {
    /// `A_log`.
    pub decay: Decay,
    /// `D`, the direct contribution of `x`.
    pub skip: Skip,
}

/// Mamba's scan: a time step and an `A_log` per feature, `B` and `C` per state.
impl<Width: AxisName, State: AxisName, Count: AxisName>
    Op<(
        (
            &Rows<Width, Count>,
            &Rows<Width, Count>,
            &Rows<State, Count>,
            &Rows<State, Count>,
        ),
        &Length,
        &mut HbmTensor<f32, m![1], m![Width, State]>,
    )>
    for SelectiveScan<HbmTensor<f32, m![1], m![Width, State]>, HbmTensor<f32, m![1], m![Width]>>
{
    type Output = Rows<Width, Count, f32>;

    fn forward(
        &self,
        _: &mut Device,
        _: (
            (
                &Rows<Width, Count>,
                &Rows<Width, Count>,
                &Rows<State, Count>,
                &Rows<State, Count>,
            ),
            &Length,
            &mut HbmTensor<f32, m![1], m![Width, State]>,
        ),
    ) -> Self::Output {
        todo!("NPU selective recurrence: FP32 history and output until gating")
    }
}

/// Mamba2's scan: a time step, an `A_log` and a `D` per head, `B` and `C` per group of heads, the
/// state per head, head feature and state.
impl<
    Width: AxisName,
    Groups: AxisName,
    State: AxisName,
    Heads: AxisName,
    Head: AxisName,
    Count: AxisName,
>
    Op<(
        (
            &Rows<Width, Count>,
            &Rows<Heads, Count>,
            &HbmTensor<bf16, m![1], m![Count, Groups, State]>,
            &HbmTensor<bf16, m![1], m![Count, Groups, State]>,
        ),
        &Length,
        &mut HbmTensor<f32, m![1], m![Heads, Head, State]>,
    )> for SelectiveScan<HbmTensor<f32, m![1], m![Heads]>, HbmTensor<f32, m![1], m![Heads]>>
{
    type Output = Rows<Width, Count, f32>;

    fn forward(
        &self,
        _: &mut Device,
        _: (
            (
                &Rows<Width, Count>,
                &Rows<Heads, Count>,
                &HbmTensor<bf16, m![1], m![Count, Groups, State]>,
                &HbmTensor<bf16, m![1], m![Count, Groups, State]>,
            ),
            &Length,
            &mut HbmTensor<f32, m![1], m![Heads, Head, State]>,
        ),
    ) -> Self::Output {
        const {
            assert!(Width::SIZE == Heads::SIZE * Head::SIZE);
            assert!(Heads::SIZE % Groups::SIZE == 0);
        }
        todo!()
    }
}
