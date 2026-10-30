//! Transposed convolution, torch's `conv_transpose1d`: one leaf, [`ConvolutionTranspose`], dense
//! over phases or depthwise as a band.
//!
//! - Dense: `y[s m + p]` for each phase `p` reads `x[m - 1 ..= m + 1]` only, so one three-tap
//!   convolution with `s` times the output channels computes every phase of a row, and the phases
//!   of a row are its consecutive samples.
//! - Depthwise by two: an output block of four input rows times two phases takes eight lanes, and a
//!   window pair of sixteen input rows covers its reach (`x[t + 2 ..= t + 8]`). Its output is the
//!   signal's fixed-point copy, which SnakeBeta reads.
use core::borrow::Borrow;
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use super::frames::{self, Blocks, Frames, Groups, LEAD, Pads, Phase, Stereo, Window};
use super::{Chunks, Flit, Lag, Packet, Split, Term};
use crate::Op;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::specialize::convolution::{Narrow, Strategy};
use crate::tensor::form::Weight;
use crate::tensor::parallel::Chip;

axes![Three = 3, Quad = 4, Half = 2];

type Group<S> = Symbol<Groups<S>>;

/// Every phase of every output channel, phase outermost.
#[derive(Clone, Debug)]
pub struct Phased<Out, const STRIDE: usize>(PhantomData<Out>);

impl<Out: AxisName, const STRIDE: usize> AxisName for Phased<Out, STRIDE> {
    const NAME: Ident = Ident::new("Phased");
    const SIZE: usize = STRIDE * Out::SIZE;
}

/// Dense coefficients `[phase * out, tap, in]`: phase `p`, tap `k` holds the transposed kernel's
/// coefficient `s (1 - k) + p + padding`, zero outside it; held as `T`, owned or borrowed.
pub type Coefficients<
    In,
    Out,
    const STRIDE: usize,
    X = Chip,
    T = HbmTensor<bf16, X, m![{ Symbol<Phased<Out, STRIDE>> }, Three, In]>,
> = Weight<m![{ Symbol<Phased<Out, STRIDE>> }, Three, In], X, T>;

/// An FP32 bias per output channel of dense coefficients, repeated for every phase.
pub type Bias<Out, const STRIDE: usize, X = Chip> =
    HbmTensor<f32, X, m![{ Symbol<Phased<Out, STRIDE>> }]>;

impl<Out: AxisName, const STRIDE: usize> Phased<Out, STRIDE> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Out: AxisName, const STRIDE: usize, S: Window, U: Window, X: M>
    Op<(
        &Frames<f32, Phased<Out, STRIDE>, S, X>,
        &mut Frames<f32, Out, U, X>,
    )> for Phased<Out, STRIDE>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (phased, output): (
            &Frames<f32, Phased<Out, STRIDE>, S, X>,
            &mut Frames<f32, Out, U, X>,
        ),
    ) {
        // SAFETY: the phased rows are consecutive output samples; only the row axis is renamed.
        let rows: HbmTensorView<
            '_,
            f32,
            X,
            m![Stereo, { Symbol<U> } # ({ Scaled<S, STRIDE> }), Out],
        > = unsafe { S::window(phased, LEAD).reshape() };
        rows.to_hbm_view(&mut device.tdma, U::interior(output));
    }
}

/// A frame's rows counted in up-sampled rows.
#[derive(Clone, Debug)]
pub struct ScaledAxis<S, const STRIDE: usize>(PhantomData<S>);

impl<S: AxisName, const STRIDE: usize> AxisName for ScaledAxis<S, STRIDE> {
    const NAME: Ident = Ident::new("Scaled");
    const SIZE: usize = STRIDE * (LEAD + S::SIZE + LEAD);
}

type Scaled<S, const STRIDE: usize> = Symbol<ScaledAxis<S, STRIDE>>;

/// A span whose `STRIDE`-fold up-sampling a model computes, and the span the model names to hold
/// it.
pub trait Upsampled<const STRIDE: usize>: Window {
    type Output: Window;
}

/// torch's transposed convolution `y[o, t] = b[o] + sum_c sum_k w[c, o, k] x[c, (t + pad - k) /
/// STRIDE]` over `STRIDE`-divisible `t + pad - k`; padding rows hold what `P` reads.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct ConvolutionTranspose<W, B, const STRIDE: usize, P = ()> {
    pub weight: W,
    pub bias: B,
    pub pads: PhantomData<P>,
}

/// The phased coefficients and the bias of every phase on every chip of `X`.
type Phases<In, Out, const STRIDE: usize, X> = (
    HostTensor<bf16, m![{ X }, { m![{ Symbol<Phased<Out, STRIDE>> }, Three, In] }]>,
    HostTensor<f32, m![{ X }, { m![{ Symbol<Phased<Out, STRIDE>> }] }]>,
);

/// The coefficients and bias of the stored weight-normalized kernel: `weight_g`, `weight_v`
/// `[input, output, at]` (`at` the stride plus twice the padding) and `bias`, every phase's.
impl<In: AxisName, Out: AxisName, const STRIDE: usize, X: M + 'static, T, B, P>
    ConvolutionTranspose<Coefficients<In, Out, STRIDE, X, T>, B, STRIDE, P>
{
    fn phases(
        source: &Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<Phases<In, Out, STRIDE, X>, LoadError> {
        let (inputs, outputs) = (In::SIZE, Out::SIZE);
        let direction = context.field("weight_v", "weight_v", |cx| cx.name());
        let stored = source.get(&direction).map(|view| view.shape().to_vec());
        let kernel = match stored.as_deref() {
            Some(&[i, o, kernel]) if i == inputs && o == outputs && kernel >= STRIDE => kernel,
            Some(shape) => {
                context.read(&direction);
                return Err(LoadError::InvalidData {
                    name: direction,
                    reason: format!(
                        "shape {shape:?}, the leaf holds [{inputs}, {outputs}, at least {STRIDE}]"
                    ),
                });
            }
            None => 0,
        };
        let mut children = Children::default();
        let weight = children.ask(super::normed(source, context, &[inputs, outputs, kernel]))?;
        let bias =
            children.ask(context.field("bias", "bias", |cx| cx.floats(source, &[outputs])))?;
        let (weight, bias) = children.node(weight.zip(bias))?;
        let padding = (kernel - STRIDE) / 2;
        let mut phased = vec![0.0_f64; STRIDE * outputs * Three::SIZE * inputs];
        for phase in 0..STRIDE {
            for tap in 0..Three::SIZE {
                let at = (STRIDE * (2 - tap) + phase + padding).checked_sub(STRIDE);
                let Some(at) = at.filter(|&at| at < kernel) else {
                    continue;
                };
                for out in 0..outputs {
                    for input in 0..inputs {
                        phased[((phase * outputs + out) * Three::SIZE + tap) * inputs + input] =
                            weight[(input * outputs + out) * kernel + at];
                    }
                }
            }
        }
        let phased: Vec<bf16> = phased.iter().map(|&v| bf16::from_f32(v as f32)).collect();
        let bias = bias.repeat(STRIDE);
        Ok((
            HostTensor::from_vec(phased.repeat(X::SIZE)),
            HostTensor::from_vec(bias.repeat(X::SIZE)),
        ))
    }
}

/// The phases of the weight-normalized kernel and the bias repeated for every phase.
/// By hand, with the update below: one read fills both leaves, which no field attribute does.
impl<'a, In: AxisName, Out: AxisName, const STRIDE: usize, X, T, B, P> Load<&'a Checkpoint>
    for ConvolutionTranspose<Coefficients<In, Out, STRIDE, X, T>, B, STRIDE, P>
where
    X: M + 'static,
    Coefficients<In, Out, STRIDE, X, T>:
        Load<HostTensor<bf16, m![{ X }, { m![{ Symbol<Phased<Out, STRIDE>> }, Three, In] }]>>,
    B: Load<HostTensor<f32, m![{ X }, { m![{ Symbol<Phased<Out, STRIDE>> }] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let (weight, bias) = Self::phases(from, cx)?;
            Ok(Self {
                weight: cx.field("weight", "", |cx| Load::load(cx, weight))?,
                bias: cx.field("bias", "bias", |cx| B::load(cx, bias))?,
                pads: PhantomData,
            })
        })
    }
}

/// The phases and the bias written in place, one rule of an update.
impl<'t, 's, In: AxisName, Out: AxisName, const STRIDE: usize, X, T, B, P>
    Load<(
        &'t mut ConvolutionTranspose<Coefficients<In, Out, STRIDE, X, T>, B, STRIDE, P>,
        &'s Checkpoint,
    )> for &'t mut ConvolutionTranspose<Coefficients<In, Out, STRIDE, X, T>, B, STRIDE, P>
where
    X: M + 'static,
    for<'u> &'u mut Coefficients<In, Out, STRIDE, X, T>: Load<(
        &'u mut Coefficients<In, Out, STRIDE, X, T>,
        HostTensor<bf16, m![{ X }, { m![{ Symbol<Phased<Out, STRIDE>> }, Three, In] }]>,
    )>,
    for<'u> &'u mut B: Load<(
        &'u mut B,
        HostTensor<f32, m![{ X }, { m![{ Symbol<Phased<Out, STRIDE>> }] }]>,
    )>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut ConvolutionTranspose<Coefficients<In, Out, STRIDE, X, T>, B, STRIDE, P>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        type Transposed<In, Out, const STRIDE: usize, X, T, B, P> =
            ConvolutionTranspose<Coefficients<In, Out, STRIDE, X, T>, B, STRIDE, P>;
        context.checkpoint(from, |cx| {
            cx.rule(
                |cx| Transposed::<In, Out, STRIDE, X, T, B, P>::phases(from, cx),
                |cx, (weight, bias)| {
                    cx.field("weight", "", |cx| {
                        <&mut Coefficients<In, Out, STRIDE, X, T> as Load<_>>::load(
                            cx,
                            (&mut into.weight, weight),
                        )
                    })?;
                    cx.field("bias", "bias", |cx| {
                        <&mut B as Load<_>>::load(cx, (&mut into.bias, bias))
                    })
                    .map(drop)
                },
            )
            .map(drop)
        })?;
        Ok(into)
    }
}

/// Three taps into every phase of every output channel in FP32, the phases then unfolded into
/// consecutive rows.
impl<'x, In, Out, const STRIDE: usize, S, P, X, T, B> Op<&'x Frames<f32, In, S, X>>
    for ConvolutionTranspose<Coefficients<In, Out, STRIDE, X, T>, B, STRIDE, P>
where
    In: AxisName,
    Out: AxisName,
    S: Upsampled<STRIDE>,
    P: Pads,
    X: M,
    T: Borrow<HbmTensor<bf16, X, m![{ Symbol<Phased<Out, STRIDE>> }, Three, In]>>,
    B: Borrow<Bias<Out, STRIDE, X>>,
{
    type Output = Frames<f32, Out, S::Output, X>;

    fn forward(&self, device: &mut Device, input: &'x Frames<f32, In, S, X>) -> Self::Output {
        const {
            assert!(matches!(
                Strategy::of(1, STRIDE * Out::SIZE, Three::SIZE, 1),
                Strategy::Dense
            ));
            assert!(S::Output::SIZE == STRIDE * S::SIZE);
        };
        let phased = super::Dense {
            weight: self.weight.tensor(),
            bias: self.bias.borrow(),
        }
        .forward(&mut *device, (input, 1));
        let mut output = Frames::<f32, Out, S::Output, X>::new();
        Phased::<Out, STRIDE>::new().forward(&mut *device, (&phased, &mut output));
        P::fill(device, &mut output);
        output
    }
}

/// Integer bits of the fixed-point copy of the up-sampled signal: `|u| < 2^9`.
pub(crate) const INTEGER: u32 = 10;

/// Depthwise coefficients by two: lane `(row % 4, phase)` and group `row / 4 % 2` weigh the 16
/// window-pair rows from 5 before the block's first input, a BF16 sum of two (`Split`).
pub type UpBand<X = Chip> = HbmTensor<bf16, X, m![Quad, Phase, Half, Split, Term, Lag]>;

/// `u[c, 2t + p] = sum_m w_p[m] x[c, clamp(t + m - 5)]`, then its Q10.21 fixed-point copy, for
/// every row of each block; `input` repeats its edge samples over its padding rows.
impl<'x, D: MaterializableScalar + FetchCast<bf16>, C: AxisName, S: Window, X: M, U>
    Op<&'x Frames<D, C, S, X>> for ConvolutionTranspose<U, (), 2>
where
    U: Borrow<UpBand<X>>,
{
    type Output = frames::Upsampled<i32, C, S, X>;

    fn forward(&self, device: &mut Device, input: &'x Frames<D, C, S, X>) -> Self::Output {
        const {
            assert!(matches!(
                Strategy::of(C::SIZE, C::SIZE, Strategy::reach(2), 2),
                Strategy::Depthwise
            ))
        };
        let band: &UpBand<X> = self.weight.borrow();
        type Rows<D, C, S, X> = DmTensor<D, X, m![Stereo], Blocks<S>, m![{ Group<S> }, Lag, C]>;
        // SAFETY: each window's rows are relabeled as eight-row groups and their offset in the
        // group.
        let near: Rows<D, C, S, X> = unsafe {
            frames::FrameAxis::<S>::new()
                .forward(&mut *device, (input, LEAD - 5))
                .reshape()
        };
        let far: Rows<D, C, S, X> = unsafe {
            frames::FrameAxis::<S>::new()
                .forward(&mut *device, (input, LEAD - 5 + 8))
                .reshape()
        };
        let band: DmTensor<
            bf16,
            X,
            m![Stereo],
            Blocks<S>,
            m![Quad, Phase, Half, Split, Term, Lag],
        > = band.to_dm(&mut device.tdma);
        let register: TrfTensor<
            bf16,
            X,
            m![Stereo],
            Blocks<S>,
            m![Quad, Phase],
            m![Half, Split, Term, Lag],
        > = device
            .sub
            .begin(band.view())
            .fetch::<m![Quad, Phase, Half, Split], m![Term, Lag]>()
            .collect::<m![Quad, Phase, Half, Split], m![Term, Lag]>()
            .to_trf();
        // Every window row streams once for each of the coefficients' two BF16 parts.
        type Time<C, S> = m![{ Group<S> }, { Chunks<C> }, Half, Split, Term, Lag];
        let fixed: DmTensor<i32, X, m![Stereo], Blocks<S>, m![{ Group<S> }, Half, Quad, Phase, C]> =
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(near.view(), far.view())
            .fetch::<Time<C, S>, Packet<C>>()
            .fetch_cast::<bf16>()
            .collect::<Time<C, S>, Flit<C>>()
            .contract_outer::<Time<C, S>, Flit<C>, _, _, _>(&register)
            .contract_packet::<Packet<C>>()
            .contract_time::<m![{ Group<S> }, { Chunks<C> }, Half]>()
            .contract_lane::<
                m![{ Group<S> }, { Chunks<C> }, Half, Quad, Phase, C % ({ Symbol<Narrow<C>> }) / 8],
                m![C % 8],
            >(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fp_to_fxp(INTEGER)
            .vector_final()
            .commit_trim::<m![C % 8]>()
            .commit();
        // SAFETY: a group's two halves of four rows are the block's rows in order.
        unsafe { fixed.reshape() }
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use super::super::frames::{FrameRows, Frames, LEAD, Span, Stereo, Window};
    use super::Phased;
    use crate::tensor::parallel::Chip;
    use crate::tests::DEVICE;

    struct Fixture;

    axes![Wide = 512, Narrow = 8];

    impl Fixture {
        fn unfold<Out: AxisName, const STRIDE: usize, S: Window, U: Window>(
            device: &mut Device,
            phased: &Frames<f32, Phased<Out, STRIDE>, S>,
        ) -> Frames<f32, Out, U> {
            let mut output = Frames::<f32, Out, U>::new();
            crate::Op::forward(&Phased::<Out, STRIDE>::new(), device, (phased, &mut output));
            output
        }

        /// Each phase of a row is written as the row's consecutive samples, for a stride of two and
        /// five.
        async fn unfolds<Out: AxisName, const STRIDE: usize, S: Window, U: Window>(
            run: impl AsyncFnOnce(
                &mut Device,
                &Frames<f32, Phased<Out, STRIDE>, S>,
            ) -> Frames<f32, Out, U>,
            device: &mut Device,
        ) {
            let (channels, rows) = (Out::SIZE, LEAD + S::SIZE + LEAD);
            let values: Vec<f32> = (0..2 * rows * STRIDE * channels)
                .map(|i| i as f32)
                .collect();
            let phased = HostTensor::<
                f32,
                m![Stereo, { FrameRows<S> }, { Symbol<Phased<Out, STRIDE>> }],
            >::from_vec(values.clone())
            .to_hbm::<Chip, m![Stereo, { FrameRows<S> }, { Symbol<Phased<Out, STRIDE>> }]>(
                &mut device.pdma,
            )
            .await
            .unwrap();
            let output = run(device, &phased).await;
            let actual = output
                .to_host::<m![Stereo, { FrameRows<U> }, Out]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let out_rows = LEAD + U::SIZE + LEAD;
            let mut bad = 0;
            for b in 0..2 {
                for s in 0..S::SIZE {
                    for p in 0..STRIDE {
                        for c in 0..channels {
                            let want = values[((b * rows + LEAD + s) * STRIDE + p) * channels + c];
                            let got = actual[(b * out_rows + LEAD + s * STRIDE + p) * channels + c];
                            if got != want {
                                bad += 1;
                            }
                        }
                    }
                }
            }
            assert_eq!(bad, 0, "{bad} misplaced samples for stride {STRIDE}");
        }
    }

    #[device(chip = 1)]
    fn wide(
        device: &mut Device,
        phased: &Frames<f32, Phased<Wide, 5>, Span<208, 8, 207>>,
    ) -> Frames<f32, Wide, Span<1040, 8, 1035>> {
        Fixture::unfold::<Wide, 5, Span<208, 8, 207>, Span<1040, 8, 1035>>(device, phased)
    }
    #[device(chip = 1)]
    fn narrow(
        device: &mut Device,
        phased: &Frames<f32, Phased<Narrow, 2>, Span<208, 8, 207>>,
    ) -> Frames<f32, Narrow, Span<416, 8, 414>> {
        Fixture::unfold::<Narrow, 2, Span<208, 8, 207>, Span<416, 8, 414>>(device, phased)
    }
    #[tokio::test(flavor = "current_thread")]
    async fn unfold_places_phases() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(wide.topology()).unwrap();
        Fixture::unfolds::<Narrow, 2, Span<208, 8, 207>, Span<416, 8, 414>>(
            async |device, phased| launch(narrow, (device, phased)).await.unwrap(),
            &mut device,
        )
        .await;
        Fixture::unfolds::<Wide, 5, Span<208, 8, 207>, Span<1040, 8, 1035>>(
            async |device, phased| launch(wide, (device, phased)).await.unwrap(),
            &mut device,
        )
        .await;
    }
}
