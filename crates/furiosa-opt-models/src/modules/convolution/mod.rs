//! Convolutional residual and attention blocks, conditioned resolution stages and the anti-aliased
//! activation.

use super::residual::Residual;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::convolution::activation::Snake;
use crate::ops::convolution::frames::{self, Blocked, Doubled, Frames, Phase, Window};
use crate::ops::convolution::transpose::{ConvolutionTranspose, Half, Quad, UpBand};
use crate::ops::convolution::{Convolution, DownBand, Lag, Lane, Reach, Split, Term};
use crate::tensor::parallel::Chip;
use crate::{Module, Sequential, Then};
pub use attention::Projection;
use core::borrow::Borrow;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;
pub use skip::{Down, Up};

mod attention;

mod skip;

/// Normalizes and activates hidden features before a convolution.
pub type Normalized<Norm, Activation, Convolution> = Then<Then<Norm, Activation>, Convolution>;

/// A residual block's branch: first layer, norm, activation, then second layer.
pub type Branch<First, Norm, Activation, Second> =
    Then<First, Then<Norm, Then<Activation, Second>>>;

/// Attends to projected spatial features, restores their original shape and adds a residual.
pub type Attention<Norm, Project, Body, Restore, Add> =
    Residual<Norm, Projection<Project, Body, Restore>, Add>;

/// Injects projected time features between two blocks and adds a projected shortcut.
#[derive(DeviceSend)]
pub struct Conditioned<First, Time, Merge, Second, Shortcut, Add> {
    pub first: First,
    pub time: Time,
    pub merge: Merge,
    pub second: Second,
    pub shortcut: Shortcut,
    pub add: Add,
}

/// The env holds the condition, which the time projection reads, beside the reader every block
/// reads; the state holds the first block's, the time projection's and the second block's slots.
impl<
    's,
    'e,
    X,
    E,
    Y,
    T,
    Z,
    W,
    O,
    FirstState,
    TimeState,
    SecondState,
    First,
    Time,
    Merge,
    Second,
    Shortcut,
    Add,
    Env: Copy,
> Module<&X, (&'e E, Env), &'s mut (FirstState, TimeState, SecondState)>
    for Conditioned<First, Time, Merge, Second, Shortcut, Add>
where
    First: for<'x> Module<&'x X, (&'e E, Env), &'s mut FirstState, Output = Y>,
    Time: Module<&'e E, (&'e E, Env), &'s mut TimeState, Output = T>,
    Merge: for<'a, 'b, 'd> Module<(&'a Y, &'b T), (&'e E, Env), &'d mut (), Output = Z>,
    Second: for<'x> Module<&'x Z, (&'e E, Env), &'s mut SecondState, Output = W>,
    Shortcut: for<'x, 'd> Module<&'x X, (&'e E, Env), &'d mut (), Output = W>,
    Add: for<'a, 'b, 'd> Module<(&'a W, &'b W), (&'e E, Env), &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: (&'e E, Env),
        (first, time, second): &'s mut (FirstState, TimeState, SecondState),
        input: &X,
    ) -> O {
        let hidden = self.first.forward(&mut *device, env, first, input);
        let time = self.time.forward(&mut *device, env, time, env.0);
        let hidden = self
            .merge
            .forward(&mut *device, env, &mut (), (&hidden, &time));
        let hidden = self.second.forward(&mut *device, env, second, &hidden);
        let shortcut = self.shortcut.forward(&mut *device, env, &mut (), input);
        self.add.forward(device, env, &mut (), (&shortcut, &hidden))
    }
}

/// Runs down and middle stages, then consumes the down-stage skips during the up stage.
#[derive(DeviceSend)]
pub struct UNet<Down, Middle, Up> {
    pub down: Down,
    pub middle: Middle,
    pub up: Up,
}

/// The state holds the down stage's, the middle stage's and the up stage's slots.
impl<'s, 'x, X, Y, Z, O, S, DownState, MiddleState, UpState, Down, Middle, Up, Env: Copy>
    Module<&'x X, Env, &'s mut (DownState, MiddleState, UpState)> for UNet<Down, Middle, Up>
where
    Down: Module<&'x X, Env, &'s mut DownState, Output = (Y, S)>,
    Middle: for<'y> Module<&'y Y, Env, &'s mut MiddleState, Output = Z>,
    Up: for<'y> Module<(Z, (&'y Y, S)), Env, &'s mut UpState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (down, middle, up): &'s mut (DownState, MiddleState, UpState),
        hidden: &'x X,
    ) -> Self::Output {
        let (hidden, skips) = self.down.forward(&mut *device, env, down, hidden);
        let output = self.middle.forward(&mut *device, env, middle, &hidden);
        self.up.forward(device, env, up, (output, (&hidden, skips)))
    }
}

/// Downsamples, runs an inner stage and upsamples.
pub type Scale<Downsample, Inner, Upsample> = Then<Downsample, Then<Inner, Upsample>>;

/// A resolution stage may change both the tensor type and its spatial or temporal extent.
pub type Stage<Layers, Resample> = Then<Layers, Resample>;

/// Adds normalized convolution and feed-forward output to the original hidden input.
pub type ConvNext<Convolution, Norm, FeedForward, Add> =
    Residual<(), Then<Convolution, Then<Norm, FeedForward>>, Add>;

/// Applies a channel-changing first layer, then `REPEATS` layers at the new width.
pub type Stack<First, Layer, const REPEATS: usize> = Sequential<First, [Layer; REPEATS]>;

/// `down(snake(up(x)))` by two from checkpoint `upsample.filter`, `downsample.lowpass.filter` and
/// `act`: `input` repeats its edge samples over its padding rows, the result's are zero.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct AntiAliased<
    C: AxisName,
    X: M = Chip,
    U = UpBand<X>,
    D = DownBand<X>,
    I = HbmTensor<i32, X, m![C]>,
    F = HbmTensor<f32, X, m![C]>,
> {
    pub up: ConvolutionTranspose<U, (), 2>,
    pub snake: Snake<C, X, I, F>,
    pub down: Convolution<D, (), 2>,
}

/// The two bands of the filters at the context's position, on every chip of `X`.
type Bands<X> = (
    HostTensor<bf16, m![{ X }, { m![Quad, Phase, Half, Split, Term, Lag] }]>,
    HostTensor<bf16, m![{ X }, { m![Lane, Split, Term, Reach] }]>,
);

impl<C: AxisName, X: M + 'static, U, D, I, F> AntiAliased<C, X, U, D, I, F> {
    /// The up-sampling band of `upsample.filter` and the low-pass band of
    /// `downsample.lowpass.filter`, each coefficient a BF16 sum of two parts.
    // TODO: read the taps only for a leaf that holds bytes, as a converted leaf's build does.
    fn bands(source: &Checkpoint, context: &mut Context<'_>) -> Result<Bands<X>, LoadError> {
        let taps = |cx: &mut Context<'_>| -> Result<[f32; 12], LoadError> {
            let taps = cx.floats(source, &[1, 1, 12])?;
            Ok(taps.try_into().expect("twelve taps, the shape checked"))
        };
        let mut children = Children::default();
        let up = children.ask(context.field("up", "upsample.filter", taps))?;
        let down = children.ask(context.field("down", "downsample.lowpass.filter", taps))?;
        let (up, down) = children.node(up.zip(down))?;
        let part = |weight: f32, part: usize| {
            let high = bf16::from_f32(weight);
            [high, bf16::from_f32(weight - high.to_f32())][part]
        };
        let mut upsampling = Vec::with_capacity(512);
        for quad in 0..4 {
            for phase in 0..2 {
                for pair in 0..2 {
                    for split in 0..2 {
                        for term in 0..2 {
                            for lag in 0..8 {
                                let m = (8 * term + lag) as isize - (4 * pair + quad) as isize;
                                let tap = 15 + phase as isize - 2 * m;
                                let weight = match (2..=8).contains(&m) && (0..12).contains(&tap) {
                                    true => 2.0 * up[tap as usize],
                                    false => 0.0,
                                };
                                upsampling.push(part(weight, split));
                            }
                        }
                    }
                }
            }
        }
        let mut lowpass = Vec::with_capacity(512);
        for lane in 0..8 {
            for split in 0..2 {
                for term in 0..2 {
                    for reach in 0..16 {
                        let tap = (16 * term + reach) as isize - 2 * lane as isize;
                        let weight = match (0..12).contains(&tap) {
                            true => down[tap as usize],
                            false => 0.0,
                        };
                        lowpass.push(part(weight, split));
                    }
                }
            }
        }
        Ok((
            HostTensor::from_vec(upsampling.repeat(X::SIZE)),
            HostTensor::from_vec(lowpass.repeat(X::SIZE)),
        ))
    }
}

/// The bands of the stored filters, then SnakeBeta under `act`.
/// By hand, with the update below: one read fills both bands, which no field attribute does.
impl<'a, C: AxisName, X: M + 'static, U, D, I, F> Load<&'a Checkpoint>
    for AntiAliased<C, X, U, D, I, F>
where
    U: Load<HostTensor<bf16, m![{ X }, { m![Quad, Phase, Half, Split, Term, Lag] }]>>,
    D: Load<HostTensor<bf16, m![{ X }, { m![Lane, Split, Term, Reach] }]>>,
    Snake<C, X, I, F>: Load<&'a Checkpoint>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let bands = children.ask(Self::bands(from, cx))?;
            let snake = cx.field("snake", "act", |cx| Snake::<C, X, I, F>::load(cx, from));
            let snake = children.ask(snake)?;
            let ((up, down), snake) = children.node(bands.zip(snake))?;
            Ok(Self {
                up: ConvolutionTranspose {
                    weight: cx.field("up", "upsample.filter", |cx| U::load(cx, up))?,
                    bias: (),
                    pads: PhantomData,
                },
                snake,
                down: Convolution {
                    weight: cx
                        .field("down", "downsample.lowpass.filter", |cx| D::load(cx, down))?,
                    bias: (),
                    pads: PhantomData,
                },
            })
        })
    }
}

/// The bands and SnakeBeta written in place, the bands one rule of an update.
impl<'t, 's, C: AxisName, X: M + 'static, U, D, I, F>
    Load<(&'t mut AntiAliased<C, X, U, D, I, F>, &'s Checkpoint)>
    for &'t mut AntiAliased<C, X, U, D, I, F>
where
    for<'u> &'u mut U: Load<(
        &'u mut U,
        HostTensor<bf16, m![{ X }, { m![Quad, Phase, Half, Split, Term, Lag] }]>,
    )>,
    for<'u> &'u mut D: Load<(
        &'u mut D,
        HostTensor<bf16, m![{ X }, { m![Lane, Split, Term, Reach] }]>,
    )>,
    for<'u> &'u mut Snake<C, X, I, F>: Load<(&'u mut Snake<C, X, I, F>, &'s Checkpoint)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut AntiAliased<C, X, U, D, I, F>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.rule(
                |cx| AntiAliased::<C, X, U, D, I, F>::bands(from, cx),
                |cx, (up, down)| {
                    cx.field("up", "upsample.filter", |cx| {
                        <&mut U as Load<_>>::load(cx, (&mut into.up.weight, up))
                    })?;
                    cx.field("down", "downsample.lowpass.filter", |cx| {
                        <&mut D as Load<_>>::load(cx, (&mut into.down.weight, down))
                    })
                    .map(drop)
                },
            )?;
            cx.field("snake", "act", |cx| {
                <&mut Snake<C, X, I, F> as Load<_>>::load(cx, (&mut into.snake, from))
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

impl<'x, D: MaterializableScalar + FetchCast<bf16>, C: AxisName, S: Doubled, X: M, U, B, I, F>
    Module<&'x Frames<D, C, S, X>, (), &mut ()> for AntiAliased<C, X, U, B, I, F>
where
    U: Borrow<UpBand<X>>,
    B: Borrow<DownBand<X>>,
    I: Borrow<HbmTensor<i32, X, m![C]>>,
    F: Borrow<HbmTensor<f32, X, m![C]>>,
{
    type Output = Frames<bf16, C, S, X>;

    fn forward(
        &self,
        device: &mut Device,
        env: (),
        _: &mut (),
        input: &'x Frames<D, C, S, X>,
    ) -> Self::Output {
        const {
            assert!(
                S::Twice::SIZE == 2 * S::SIZE
                    && S::Twice::BLOCK == 2 * S::BLOCK
                    && S::Twice::SAMPLES == 2 * S::SAMPLES
            )
        };
        let fixed = self.up.forward(&mut *device, env, &mut (), input);
        let activated = self.snake.forward(&mut *device, env, &mut (), &fixed);
        // SAFETY: two phases of a block's rows are the doubled span's rows of that block.
        let activated: Blocked<bf16, C, S::Twice, X> = unsafe { activated.reshape() };
        let mut upsampled = Frames::<bf16, C, S::Twice, X>::new();
        frames::FrameAxis::<S::Twice>::new().forward(
            &mut *device,
            env,
            &mut (),
            (&activated, &mut upsampled),
        );
        frames::Edge.forward(&mut *device, env, &mut (), (&mut upsampled, 2 * S::SAMPLES));
        let filtered =
            self.down
                .forward(&mut *device, env, &mut (), (&upsampled, PhantomData::<S>));
        let mut output = Frames::<bf16, C, S, X>::new();
        frames::FrameAxis::<S>::new().forward(&mut *device, env, &mut (), (&filtered, &mut output));
        frames::Zero.forward(&mut *device, env, &mut (), (&mut output, S::SAMPLES));
        output
    }
}

#[cfg(test)]
mod tests {
    use crate::Module;
    use crate::load::Checkpoint;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;
    use safetensors::Dtype;

    use super::AntiAliased;
    use crate::ops::convolution;
    use crate::ops::convolution::frames::{self, Frames, LEAD, Span, Stereo, Window};
    use crate::tensor::parallel::Chip;

    type Rows = Span<4000, 16, 3990>;

    impl frames::Doubled for Rows {
        type Twice = Span<8000, 32, 7980>;
    }
    type Wide = Span<1040, 8, 1035>;

    impl frames::Doubled for Wide {
        type Twice = Span<2080, 16, 2070>;
    }
    axes![C = 16, W = 512];

    /// The anti-aliasing filter every activation of the checkpoint stores.
    pub(crate) const KAISER: [f32; 12] = [
        0.0020289646927267313,
        0.009389465674757957,
        -0.0255434587597847,
        -0.057657383382320404,
        0.12857258319854736,
        0.44320979714393616,
        0.44320979714393616,
        0.12857258319854736,
        -0.057657383382320404,
        -0.0255434587597847,
        0.009389465674757957,
        0.0020289646927267313,
    ];

    /// The Kaiser-windowed sinc of cutoff 0.15, where [`KAISER`]'s is 0.25: another filter a
    /// checkpoint could store, so a load that reads one stored filter for the other fails.
    const NARROW: [f32; 12] = [
        -0.0025476434,
        -0.011789749,
        -0.0056311246,
        0.057454333,
        0.17895807,
        0.2835561,
        0.2835561,
        0.17895807,
        0.057454333,
        -0.0056311246,
        -0.011789749,
        -0.0025476434,
    ];

    /// A stereo signal and the checkpoint of an activation: its up-sampling and low-pass filters,
    /// and SnakeBeta's log-scale `alpha` and `beta` of each channel.
    struct Case {
        signal: Vec<f32>,
        filters: ([f32; 12], [f32; 12]),
        alpha: Vec<f32>,
        beta: Vec<f32>,
    }

    impl Case {
        fn new(samples: usize, channels: usize, filters: ([f32; 12], [f32; 12])) -> Self {
            let random = convolution::tests::Case::random;
            Self {
                signal: random(2 * samples * channels, 21)
                    .into_iter()
                    .map(|v| v * 3.0)
                    .collect(),
                filters,
                alpha: random(channels, 5).into_iter().map(|v| v * 1.4).collect(),
                beta: random(channels, 9).into_iter().map(|v| v * 1.5).collect(),
            }
        }

        /// The activation's checkpoint as the audio decoder stores one: both filters `[1, 1, 12]`
        /// and SnakeBeta's parameters `[C]`, in F32.
        fn checkpoint(&self) -> Checkpoint {
            let (up, down) = &self.filters;
            let (filter, channel) = ([1, 1, 12], [self.alpha.len()]);
            let mut checkpoint = Checkpoint::default();
            for (name, shape, values) in [
                ("upsample.filter", &filter[..], &up[..]),
                ("downsample.lowpass.filter", &filter, down),
                ("act.alpha", &channel, &self.alpha),
                ("act.beta", &channel, &self.beta),
            ] {
                let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                checkpoint.insert(name, Dtype::F32, shape, bytes).unwrap();
            }
            checkpoint
        }

        fn frame<S: Window>(&self, channels: usize) -> Vec<f32> {
            let (rows, samples) = (LEAD + S::SIZE + LEAD, S::SAMPLES);
            let mut frame = vec![0.0f32; 2 * rows * channels];
            for b in 0..2 {
                for r in 0..rows {
                    let t = (r as isize - LEAD as isize).clamp(0, samples as isize - 1) as usize;
                    for c in 0..channels {
                        frame[(b * rows + r) * channels + c] =
                            self.signal[(b * samples + t) * channels + c];
                    }
                }
            }
            frame
        }

        fn bf(v: f64) -> f64 {
            f64::from(bf16::from_f32(v as f32).to_f32())
        }

        /// `Activation1d(SnakeBeta)` of channel `c`'s samples `x` with the stored filters and
        /// log-scale parameters, rounding where the device rounds.
        fn reference(&self, x: &[f64], c: usize) -> Vec<f64> {
            let (up, down) = &self.filters;
            let [alpha, beta] = [&self.alpha, &self.beta].map(|stored| f64::from(stored[c]).exp());
            let t = x.len();
            let at = |i: isize| x[i.clamp(0, t as isize - 1) as usize];
            let upsampled: Vec<f64> = (0..2 * t)
                .map(|j| {
                    let (base, phase) = ((j / 2) as isize, j % 2);
                    (2..=8)
                        .filter_map(|m: isize| {
                            let tap = 15 + phase as isize - 2 * m;
                            (0..12).contains(&tap).then(|| {
                                2.0 * f64::from(up[tap as usize]) * Self::bf(at(base + m - 5))
                            })
                        })
                        .sum()
                })
                .collect();
            let snake: Vec<f64> = upsampled
                .iter()
                .map(|u| Self::bf(u + (alpha * u).sin().powi(2) / (beta + 1e-9)))
                .collect();
            let v = |i: isize| snake[i.clamp(0, 2 * t as isize - 1) as usize];
            (0..t)
                .map(|y| {
                    (0..12)
                        .map(|k| f64::from(down[k]) * v(2 * y as isize + k as isize - 5))
                        .sum()
                })
                .collect()
        }

        /// The live samples of every `stride`-th channel against the reference, and the padding
        /// rows the next convolution reads, zero on every channel.
        fn check<S: Window>(&self, actual: &[bf16], channels: usize, stride: usize) {
            let (rows, samples) = (LEAD + S::SIZE + LEAD, S::SAMPLES);
            let mut worst = 0.0f64;
            for b in 0..2 {
                for c in (0..channels).step_by(stride) {
                    let x: Vec<f64> = (0..samples)
                        .map(|t| f64::from(self.signal[(b * samples + t) * channels + c]))
                        .collect();
                    for (t, want) in self.reference(&x, c).iter().enumerate() {
                        let got = f64::from(actual[(b * rows + LEAD + t) * channels + c].to_f32());
                        let error = (got - want).abs() / want.abs().max(1.0);
                        worst = worst.max(error);
                        assert!(
                            error <= 2e-2,
                            "stereo {b} channel {c} sample {t}: {got} vs {want}"
                        );
                    }
                }
                for r in (0..LEAD).chain(LEAD + samples..LEAD + samples + LEAD) {
                    for c in 0..channels {
                        let got = actual[(b * rows + r) * channels + c].to_f32();
                        assert!(got == 0.0, "stereo {b} channel {c} padding row {r}: {got}");
                    }
                }
            }
            eprintln!("activation over {channels} channels worst relative error {worst:e}");
        }
    }

    #[device(chip = 1)]
    fn activation(
        device: &mut Device,
        input: &Frames<f32, C, Rows>,
        module: &AntiAliased<C>,
    ) -> Frames<bf16, C, Rows> {
        module.forward(device, (), &mut (), input)
    }

    #[device(chip = 1)]
    fn wide(
        device: &mut Device,
        input: &Frames<f32, W, Wide>,
        module: &AntiAliased<W>,
    ) -> Frames<bf16, W, Wide> {
        module.forward(device, (), &mut (), input)
    }

    /// Up-sampling, SnakeBeta and low-pass of a 16-channel stereo signal whose length is not a
    /// whole block, loaded from a checkpoint whose two filters differ: the definition with edge
    /// replication on the live samples, zero on the padding rows the next convolution reads.
    #[tokio::test(flavor = "current_thread")]
    async fn activation_matches_definition() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(activation.topology()).unwrap();
        let case = Case::new(Rows::SAMPLES, C::SIZE, (KAISER, NARROW));
        let input = HostTensor::<f32, m![Stereo, { frames::FrameRows<Rows> }, C]>::from_vec(
            case.frame::<Rows>(C::SIZE),
        )
        .to_hbm::<Chip, m![Stereo, { frames::FrameRows<Rows> }, C]>(&mut device.pdma)
        .await
        .unwrap();
        let module: AntiAliased<C> = case.checkpoint().load(&mut device).unwrap();
        let output = launch(activation, (&mut device, &input, &module))
            .await
            .unwrap();
        let actual = output
            .to_host::<m![Stereo, { frames::FrameRows<Rows> }, C]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        case.check::<Rows>(&actual, C::SIZE, 1);
    }

    /// The same over 512 channels in 32 packets of sixteen, both filters the checkpoint's.
    #[tokio::test(flavor = "current_thread")]
    async fn wide_activation_matches_definition() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(wide.topology()).unwrap();
        let case = Case::new(Wide::SAMPLES, W::SIZE, (KAISER, KAISER));
        let input = HostTensor::<f32, m![Stereo, { frames::FrameRows<Wide> }, W]>::from_vec(
            case.frame::<Wide>(W::SIZE),
        )
        .to_hbm::<Chip, m![Stereo, { frames::FrameRows<Wide> }, W]>(&mut device.pdma)
        .await
        .unwrap();
        let module: AntiAliased<W> = case.checkpoint().load(&mut device).unwrap();
        let output = launch(wide, (&mut device, &input, &module)).await.unwrap();
        let actual = output
            .to_host::<m![Stereo, { frames::FrameRows<Wide> }, W]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        case.check::<Wide>(&actual, W::SIZE, 37);
    }
}
