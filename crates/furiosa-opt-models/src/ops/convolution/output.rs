//! Seven-tap, one-output convolution and a fused waveform clamp.
//!
//! Coefficients and inputs are BF16; products accumulate in FP32 before clipping to `[-1, 1]`.
//! Each stereo channel retains its own contiguous span, including computed rows past the live
//! samples. The caller supplies zeros in the input frame's padding rows.
use core::borrow::Borrow;
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use super::Convolution;
use super::frames::{Frames, Stereo, Window};
use crate::Op;
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::tensor::memory::residency::Residency;
use crate::tensor::parallel::Chip;

pub use super::{Band, Lag, Lane, Term};

/// Seven taps into one channel, without bias, followed by a clamp to `[-1, 1]`.
///
/// `T` retains the coefficients as an owned tensor, borrowed tensor, or another weight-tree
/// storage leaf. A model supplies the checkpoint prefix with `Named`; this leaf reads only
/// `weight_g: [1, 1, 1]` and `weight_v: [1, C, 7]` below it. Weight normalization and conversion
/// to the lane band use the same loader as the unclamped banded [`Convolution`].
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T))]
pub struct Waveform<C: AxisName, X: M = Chip, T = HbmTensor<bf16, X, m![Lane, Term, Lag, C]>> {
    /// Lane `l` weighs row `q` of the window pair by tap `q - l`.
    pub band: T,
    shape: PhantomData<fn() -> (C, X)>,
}

impl<C: AxisName, X: M, T> Waveform<C, X, T> {
    /// Retain a prepared band, owned or borrowed, without copying its storage.
    pub fn new(band: Band<C, X, T>) -> Self {
        Self {
            band: band.into_tensor(),
            shape: PhantomData,
        }
    }
}

/// Read the normalized coefficients once and retain their packed BF16 image in `T`.
impl<'a, C: AxisName, X: M + 'static, T> Load<&'a Checkpoint> for Waveform<C, X, T>
where
    T: Load<HostTensor<bf16, m![{ X }, { m![Lane, Term, Lag, C] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let band = cx.field("band", "", |cx| {
                let host = Convolution::<Band<C, X, T>>::band(from, cx)?;
                Load::load(cx, host)
            })?;
            Ok(Self {
                band,
                shape: PhantomData,
            })
        })
    }
}

/// Rebuild the band as one update rule and write the retained storage in place.
impl<'t, 's, C: AxisName, X: M + 'static, T> Load<(&'t mut Waveform<C, X, T>, &'s Checkpoint)>
    for &'t mut Waveform<C, X, T>
where
    for<'u> &'u mut T: Load<(
        &'u mut T,
        HostTensor<bf16, m![{ X }, { m![Lane, Term, Lag, C] }]>,
    )>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Waveform<C, X, T>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("band", "", |cx| {
                cx.rule(
                    |cx| Convolution::<Band<C, X, T>>::band(from, cx),
                    |cx, host| <&mut T as Load<_>>::load(cx, (&mut into.band, host)).map(drop),
                )
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

/// `clamp(sum_k sum_c w[k, c] x[t + k - 3, c], -1, 1)` for seven taps.
impl<'x, C: AxisName, S: Window, X: M, T> Op<&'x Frames<bf16, C, S, X>> for Waveform<C, X, T>
where
    T: Borrow<HbmTensor<bf16, X, m![Lane, Term, Lag, C]>>,
{
    type Output = HbmTensor<f32, X, m![Stereo, { Symbol<S> }]>;

    fn forward(&self, device: &mut Device, input: &'x Frames<bf16, C, S, X>) -> Self::Output {
        const { assert!(C::SIZE > 0 && C::SIZE <= 16) };
        super::banded::<C, S, X, true>(device, self.band.borrow(), input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::Named;
    use crate::ops::convolution::frames::{self, FrameRows, LEAD, Span};
    use crate::tensor::memory::Dtype;
    use furiosa_opt_std::runtime::launch;

    axes![Channels = 8];
    type Chips = m![2];
    type Signal = Span<32, 8, 29>;

    fn checkpoint(magnitude: f32, direction: &[f32]) -> Checkpoint {
        let mut cp = Checkpoint::default();
        cp.insert(
            "conv_post.weight_g",
            Dtype::F32,
            &[1, 1, 1],
            f32::to_buf(&[magnitude]),
        )
        .unwrap();
        cp.insert(
            "conv_post.weight_v",
            Dtype::F32,
            &[1, Channels::SIZE, 7],
            f32::to_buf(direction),
        )
        .unwrap();
        cp
    }

    fn direction() -> Vec<f32> {
        (0..Channels::SIZE * 7)
            .map(|i| (i as f32 - 23.0) / 32.0)
            .collect()
    }

    /// Independent checkpoint formula, before the Toeplitz-band layout is chosen.
    fn coefficients(magnitude: f32, direction: &[f32]) -> Vec<f64> {
        let norm = direction
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        direction
            .iter()
            .map(|&v| {
                let normalized = f64::from(magnitude) * f64::from(v) / norm;
                f64::from(bf16::from_f32(normalized as f32).to_f32())
            })
            .collect()
    }

    #[device(chip = 1)]
    fn waveform(
        device: &mut Device,
        layer: &Waveform<Channels>,
        input: &Frames<bf16, Channels, Signal>,
    ) -> HbmTensor<f32, Chip, m![Stereo, { Symbol<Signal> }]> {
        layer.forward(device, input)
    }

    /// Every tap in isolation checks orientation, stereo order, slice seams, and both padded
    /// edges. A signed multichannel case then checks accumulation and both clamp boundaries.
    #[tokio::test(flavor = "current_thread")]
    async fn waveform_matches_independent_convolution_and_clamp() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(waveform.topology()).unwrap();
        let samples = Signal::SAMPLES;
        let frame_rows = FrameRows::<Signal>::SIZE;
        let mut signal = vec![0.0f32; 2 * samples * Channels::SIZE];
        for stereo in 0..2 {
            for row in 0..samples {
                for channel in 0..Channels::SIZE {
                    let at = (stereo * samples + row) * Channels::SIZE + channel;
                    signal[at] = bf16::from_f32(
                        ((row * 17 + channel * 11 + stereo * 7) % 61) as f32 / 8.0 - 3.75,
                    )
                    .to_f32();
                }
            }
        }
        let mut frame = vec![bf16::from_f32(0.0); 2 * frame_rows * Channels::SIZE];
        for stereo in 0..2 {
            for row in 0..samples {
                for channel in 0..Channels::SIZE {
                    frame[((stereo * frame_rows + LEAD + row) * Channels::SIZE) + channel] =
                        bf16::from_f32(signal[(stereo * samples + row) * Channels::SIZE + channel]);
                }
            }
        }
        let input =
            HostTensor::<bf16, m![Stereo, { FrameRows<Signal> }, Channels]>::from_vec(frame)
                .to_hbm::<Chip, m![Stereo, { FrameRows<Signal> }, Channels]>(&mut device.pdma)
                .await
                .unwrap();
        for case in 0..8 {
            let (g, v) = if case < 7 {
                let mut v = vec![0.0; Channels::SIZE * 7];
                v[(case % Channels::SIZE) * 7 + case] = 1.0;
                (1.0, v)
            } else {
                (1.75, direction())
            };
            let cp = checkpoint(g, &v);
            let layer = cp
                .load::<Named<"conv_post", Waveform<Channels>>>(&mut device)
                .unwrap();
            let output = launch(waveform, (&mut device, &layer.0, &input))
                .await
                .unwrap();
            let actual = output
                .to_host::<m![Stereo, { Symbol<Signal> }]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let weights = coefficients(g, &v);
            let mut clamped = [false; 2];
            for stereo in 0..2 {
                for row in 0..Signal::SIZE {
                    let mut sum = 0.0f64;
                    let mut absolute_products = 0.0f64;
                    for tap in 0..7 {
                        let source = row as isize + tap as isize - 3;
                        if (0..samples as isize).contains(&source) {
                            for channel in 0..Channels::SIZE {
                                let product = weights[channel * 7 + tap]
                                    * f64::from(
                                        signal[(stereo * samples + source as usize)
                                            * Channels::SIZE
                                            + channel],
                                    );
                                sum += product;
                                absolute_products += product.abs();
                            }
                        }
                    }
                    clamped[0] |= sum < -1.0;
                    clamped[1] |= sum > 1.0;
                    let want = sum.clamp(-1.0, 1.0);
                    let got = f64::from(actual[stereo * Signal::SIZE + row]);
                    // BF16 products are exact in FP32. Use a conservative reduction bound for
                    // 56 terms, allowing any sum order; clipping cannot amplify this error.
                    let bound = 56.0 * f64::from(f32::EPSILON) * absolute_products + 1e-7;
                    assert!(
                        (got - want).abs() <= bound,
                        "case {case}, stereo {stereo}, row {row}: {got} vs {want}"
                    );
                    assert!((-1.0..=1.0).contains(&got));
                }
            }
            if case < 7 {
                assert_eq!(clamped, [true, true], "impulse case {case}");
            }
        }
    }
}
