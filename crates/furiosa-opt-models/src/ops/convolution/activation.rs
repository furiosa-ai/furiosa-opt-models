//! SnakeBeta over the fixed-point copy of an up-sampled signal: `u + sin^2(alpha u) / beta`, with
//! the sine argument reduced to one period exactly in fixed point.
use core::borrow::Borrow;
use core::f32::consts::TAU;
use core::marker::PhantomData;
use std::f64::consts::PI;

use furiosa_opt_std::prelude::*;

use super::Term;
use super::frames::{Blocks, Phase, Row, Stereo, Upsampled, Window};
use super::transpose::INTEGER;
use crate::Op;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::tensor::parallel::Chip;

/// Per-channel SnakeBeta parameters in the fixed-point forms its activation reads, from the
/// checkpoint's log-scale `alpha` and `beta`, each held as `I` or `F`, owned or borrowed.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Snake<
    C: AxisName,
    X: M = Chip,
    I = HbmTensor<i32, X, m![C]>,
    F = HbmTensor<f32, X, m![C]>,
> {
    /// `alpha / pi` in Q1.30.
    pub periods: I,
    /// `1 / (2 (beta + 1e-9))` in Q10.21.
    pub offset: I,
    /// `-1 / (2 (beta + 1e-9))`.
    pub weight: F,
    pub shape: PhantomData<fn() -> (C, X)>,
}

/// The fixed-point forms of `alpha` and `beta` on every chip of `X`.
type Forms<C, X> = (
    HostTensor<i32, m![{ X }, { m![C] }]>,
    HostTensor<i32, m![{ X }, { m![C] }]>,
    HostTensor<f32, m![{ X }, { m![C] }]>,
);

impl<C: AxisName, X: M + 'static, I, F> Snake<C, X, I, F> {
    /// The forms of the log-scale `alpha` and `beta` at the context's position, F32 or BF16.
    // TODO: read the values only for a leaf that holds bytes, as a converted leaf's build does.
    fn forms(source: &Checkpoint, context: &mut Context<'_>) -> Result<Forms<C, X>, LoadError> {
        let mut children = Children::default();
        let alpha = context.field("alpha", "alpha", |cx| cx.floats(source, &[C::SIZE]));
        let alpha = children.ask(alpha)?;
        let beta = context.field("beta", "beta", |cx| cx.floats(source, &[C::SIZE]));
        let beta = children.ask(beta)?;
        let (alpha, beta) = children.node(alpha.zip(beta))?;
        let fixed = |value: f64, bits: i32| (value * 2f64.powi(bits)).round_ties_even() as i32;
        let period = |a: &f32| fixed(f64::from(*a).exp() / PI, 30);
        let periods: Vec<i32> = alpha.iter().map(period).collect();
        let scale = |b: &f32| 0.5 / (f64::from(*b).exp() + 1e-9);
        let offset: Vec<i32> = beta.iter().map(|b| fixed(scale(b), 21)).collect();
        let weight: Vec<f32> = beta.iter().map(|b| -scale(b) as f32).collect();
        Ok((
            HostTensor::from_vec(periods.repeat(X::SIZE)),
            HostTensor::from_vec(offset.repeat(X::SIZE)),
            HostTensor::from_vec(weight.repeat(X::SIZE)),
        ))
    }
}

/// The three forms of `alpha` and `beta`.
/// By hand, with the update below: one read fills three leaves, which no field attribute does.
impl<'a, C: AxisName, X: M + 'static, I, F> Load<&'a Checkpoint> for Snake<C, X, I, F>
where
    I: Load<HostTensor<i32, m![{ X }, { m![C] }]>>,
    F: Load<HostTensor<f32, m![{ X }, { m![C] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let (periods, offset, weight) = Self::forms(from, cx)?;
            Ok(Self {
                periods: cx.field("periods", "", |cx| I::load(cx, periods))?,
                offset: cx.field("offset", "", |cx| I::load(cx, offset))?,
                weight: cx.field("weight", "", |cx| F::load(cx, weight))?,
                shape: PhantomData,
            })
        })
    }
}

/// The three forms written in place, one rule of an update.
impl<'t, 's, C: AxisName, X: M + 'static, I, F> Load<(&'t mut Snake<C, X, I, F>, &'s Checkpoint)>
    for &'t mut Snake<C, X, I, F>
where
    for<'u> &'u mut I: Load<(&'u mut I, HostTensor<i32, m![{ X }, { m![C] }]>)>,
    for<'u> &'u mut F: Load<(&'u mut F, HostTensor<f32, m![{ X }, { m![C] }]>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Snake<C, X, I, F>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.rule(
                |cx| Snake::<C, X, I, F>::forms(from, cx),
                |cx, (periods, offset, weight)| {
                    cx.field("periods", "", |cx| {
                        <&mut I as Load<_>>::load(cx, (&mut into.periods, periods))
                    })?;
                    cx.field("offset", "", |cx| {
                        <&mut I as Load<_>>::load(cx, (&mut into.offset, offset))
                    })?;
                    cx.field("weight", "", |cx| {
                        <&mut F as Load<_>>::load(cx, (&mut into.weight, weight))
                    })
                    .map(drop)
                },
            )
            .map(drop)
        })?;
        Ok(into)
    }
}

/// `u + sin^2(alpha u) / (beta + 1e-9)` as `u + (1 - cos(2 pi frac(alpha u / pi))) / (2 beta)`,
/// a shift pair dropping the fixed-point period count's integer bits.
impl<'x, C: AxisName, S: Window, X: M, I, F> Op<&'x Upsampled<i32, C, S, X>> for Snake<C, X, I, F>
where
    I: Borrow<HbmTensor<i32, X, m![C]>>,
    F: Borrow<HbmTensor<f32, X, m![C]>>,
{
    type Output = Upsampled<bf16, C, S, X>;

    fn forward(&self, device: &mut Device, fixed: &'x Upsampled<i32, C, S, X>) -> Self::Output {
        type Parameter<D, S, C, X> = DmTensor<D, X, m![Stereo], Blocks<S>, m![C]>;
        type Operand<D, S, C, X> = VrfTensor<D, X, m![Stereo], Blocks<S>, m![C]>;
        let periods: &HbmTensor<i32, X, m![C]> = self.periods.borrow();
        let offset: &HbmTensor<i32, X, m![C]> = self.offset.borrow();
        let weight: &HbmTensor<f32, X, m![C]> = self.weight.borrow();
        let periods: Parameter<i32, S, C, X> = periods.to_dm(&mut device.tdma);
        let periods: Operand<i32, S, C, X> = device
            .sub
            .begin(periods.view())
            .fetch::<m![C / 8], m![C % 8]>()
            .collect::<m![C / 8], m![C % 8]>()
            .to_vrf();
        let offset: Parameter<i32, S, C, X> = offset.to_dm(&mut device.tdma);
        let offset: Operand<i32, S, C, X> = device
            .sub
            .begin(offset.view())
            .fetch::<m![C / 8], m![C % 8]>()
            .collect::<m![C / 8], m![C % 8]>()
            .to_vrf();
        let weight: Parameter<f32, S, C, X> = weight.to_dm(&mut device.tdma);
        let weight: Operand<f32, S, C, X> = device
            .sub
            .begin(weight.view())
            .fetch::<m![C / 8], m![C % 8]>()
            .collect::<m![C / 8], m![C % 8]>()
            .to_vrf();
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(fixed.view(), fixed.view())
            .fetch::<m![{ Row<S> }, Phase, C / 8, Term], m![C % 8]>()
            .collect::<m![{ Row<S> }, Phase, C / 8, Term], m![C % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, m![{ Row<S> }, Phase, C / 8]>()
            .vector_fxp(FxpBinaryOp::MulFxp, &periods, ())
            .vector_fxp(FxpBinaryOp::LeftShift, 12, ())
            .vector_fxp(FxpBinaryOp::LogicRightShift, 11, ())
            .vector_fxp(FxpBinaryOp::AddFxp, (), &offset)
            .vector_fxp_to_fp(INTEGER)
            .vector_narrow_split::<m![{ Row<S> }, Phase, C / 4], m![C % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), TAU, ())
            .vector_fp_unary(FpUnaryOp::Cos, true, false)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight, ())
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_widen_concat::<m![{ Row<S> }, Phase, C / 8], m![C % 8]>()
            .vector_final()
            .cast::<bf16, m![C % 8 # 16]>()
            .commit_trim::<m![C % 8]>()
            .commit()
    }
}
