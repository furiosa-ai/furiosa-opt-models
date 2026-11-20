//! H3's stereo audio decoder, composed from the public BigVGAN modules and convolution leaves.
//!
//! Load relative to the audio component's physical checkpoint prefix. The folded input reads
//! `dec_in_proj.*` and `decoder.conv_pre.*`; stages and waveform read the named children of
//! `decoder`. No checkpoint prefix is ignored, and latent statistics do not transform decoder
//! inputs or parameters.
//!
//! Each activation retains both of its stored filters through the ordinary `AntiAliased` loader.
//! In checkpoint revision `42ed227ee7df40d41602854ae760620d6eb651fe`, all 254 stored activation
//! filters were checked and are bit-identical to `decoder.activation_post.upsample.filter`.
//! Each activation loads its own filters, retaining stored differences in other checkpoints.

use furiosa_opt_std::prelude::*;

use super::embeddings::Chips;
use crate::Then;
use crate::load::Named;
use crate::models::diffusion::{AmpBlock, AmpUnit, BigVGAN, BigVganStage};
use crate::modules::convolution::AntiAliased;
use crate::ops::convolution::frames::{Doubled, Edge, Phase, Span, Stereo, Zero};
use crate::ops::convolution::input::Lift;
use crate::ops::convolution::output::{Lag, Lane, Term, Waveform};
use crate::ops::convolution::residual::{Mean, Sum};
use crate::ops::convolution::transpose::{self, Half, Phased, Quad, Three, Upsampled};
use crate::ops::convolution::{self, Convolution, Produced, Reach, Split};
use crate::tensor::memory::{Hbm, Memory};

// The folded input's output axis must also name the first stage's input axis.
pub use input::{C1024, K7};

pub mod input;

/// Source: https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/diffusion/models/minimax_h3/vae.py
/// The upstream audio-decoder component contract, bound to the stereo latent geometry below.
/// Lifted stereo latents, the folded input convolution, seven stages, then clamped FP32 samples.
///
/// `X` selects the storage of every retained coefficient, bias, filter and SnakeBeta parameter;
/// `&X` borrows those same leaves. Execution uses the ordinary nested `Module` state of this
/// composition. The default mapping replicates the decoder over the deployment's four chips.
pub type Decoder<C = Chips, X = Hbm> =
    BigVGAN<Lift, input::Input<C, X>, Named<"decoder", Stages<C, X>>, Named<"decoder", Head<C, X>>>;

/// The seven stages at widths 1024 → 512 → 256 → 128 → 64 → 32 → 16 → 8.
///
/// Each stage reads sibling checkpoint paths: `ups.I.0` and `resblocks.(3*I+K)`.
/// Naming each child directly keeps ordinary tree loading, borrowing and in-place updates;
/// the nested `Then` and `Fork` nodes add no checkpoint segment. Only the final average repeats
/// its edge samples, preparing the post activation; earlier averages zero their padding rows.
pub type Stages<C = Chips, X = Hbm> = Then<
    Stage<
        Named<"ups.0.0", Upsample<C1024, C512, 5, C, X>>,
        Named<"resblocks.0", Block<C512, K3, C, X>>,
        Named<"resblocks.1", Block<C512, K7, C, X>>,
        Named<"resblocks.2", Block<C512, K11, C, X>>,
    >,
    Then<
        Stage<
            Named<"ups.1.0", Upsample<C512, C256, 5, C, X>>,
            Named<"resblocks.3", Block<C256, K3, C, X>>,
            Named<"resblocks.4", Block<C256, K7, C, X>>,
            Named<"resblocks.5", Block<C256, K11, C, X>>,
        >,
        Then<
            Stage<
                Named<"ups.2.0", Upsample<C256, C128, 2, C, X>>,
                Named<"resblocks.6", Block<C128, K3, C, X>>,
                Named<"resblocks.7", Block<C128, K7, C, X>>,
                Named<"resblocks.8", Block<C128, K11, C, X>>,
            >,
            Then<
                Stage<
                    Named<"ups.3.0", Upsample<C128, C64, 2, C, X>>,
                    Named<"resblocks.9", Block<C64, K3, C, X>>,
                    Named<"resblocks.10", Block<C64, K7, C, X>>,
                    Named<"resblocks.11", Block<C64, K11, C, X>>,
                >,
                Then<
                    Stage<
                        Named<"ups.4.0", Upsample<C64, C32, 2, C, X>>,
                        Named<"resblocks.12", Block<C32, K3, C, X>>,
                        Named<"resblocks.13", Block<C32, K7, C, X>>,
                        Named<"resblocks.14", Block<C32, K11, C, X>>,
                    >,
                    Then<
                        Stage<
                            Named<"ups.5.0", Upsample<C32, C16, 2, C, X>>,
                            Named<"resblocks.15", Block<C16, K3, C, X>>,
                            Named<"resblocks.16", Block<C16, K7, C, X>>,
                            Named<"resblocks.17", Block<C16, K11, C, X>>,
                        >,
                        Stage<
                            Named<"ups.6.0", Upsample<C16, C8, 2, C, X>>,
                            Named<"resblocks.18", Block<C8, K3, C, X>>,
                            Named<"resblocks.19", Block<C8, K7, C, X>>,
                            Named<"resblocks.20", Block<C8, K11, C, X>>,
                            Edge,
                        >,
                    >,
                >,
            >,
        >,
    >,
>;

/// An upsampling convolution, parallel three-block residual stacks, then their average.
pub type Stage<Upsample, Narrow, Medium, Wide, P = Zero> =
    BigVganStage<Upsample, Narrow, Medium, Wide, Mean<P>>;

/// Normalized transposed coefficients split into three-tap phases, with a bias per phase.
/// Input padding is zero; the upsampled frame repeats edge samples for the first activation.
pub type Upsample<In, Out, const STRIDE: usize, C = Chips, X = Hbm> =
    transpose::ConvolutionTranspose<
        transpose::Coefficients<
            In,
            Out,
            STRIDE,
            C,
            <X as Memory>::Tensor<bf16, C, m![{ Symbol<Phased<Out, STRIDE>> }, Three, In]>,
        >,
        <X as Memory>::Tensor<f32, C, m![{ Symbol<Phased<Out, STRIDE>> }]>,
        STRIDE,
        Edge,
    >;

/// Three residual units of one kernel width, at dilations 1, 3 and 5.
/// The public block already binds `activations.0..5`, `convs1.0..2` and `convs2.0..2`.
pub type Block<Channels, Tap, C = Chips, X = Hbm> = AmpBlock<
    Activation<Channels, C, X>,
    Dilated<Channels, Tap, 1, C, X>,
    Dilated<Channels, Tap, 3, C, X>,
    Dilated<Channels, Tap, 5, C, X>,
    Dilated<Channels, Tap, 1, C, X>,
    Sum,
>;

/// An unnamed residual unit; checkpoint names for its two activations differ in `AmpBlock`.
pub type Unit<Channels, Tap, const DILATION: usize, C = Chips, X = Hbm> = AmpUnit<
    Activation<Channels, C, X>,
    Dilated<Channels, Tap, DILATION, C, X>,
    Activation<Channels, C, X>,
    Dilated<Channels, Tap, 1, C, X>,
    Sum,
>;

/// Dense normalized BF16 coefficients and FP32 bias; edge padding prepares the next activation.
pub type Dilated<Channels, Tap, const DILATION: usize, C = Chips, X = Hbm> = Convolution<
    convolution::Coefficients<
        Channels,
        Channels,
        Tap,
        C,
        <X as Memory>::Tensor<bf16, C, m![{ Symbol<Produced<Channels>> }, Tap, Channels]>,
    >,
    <X as Memory>::Tensor<f32, C, m![{ Symbol<Produced<Channels>> }]>,
    1,
    DILATION,
    Edge,
>;

/// The stored up/down filters around SnakeBeta, with all four storage kinds selected by `X`.
/// The existing loader owns filter packing and the log-scale parameter conversion under `act`.
pub type Activation<Channels, C = Chips, X = Hbm> = AntiAliased<
    Channels,
    C,
    <X as Memory>::Tensor<bf16, C, m![Quad, Phase, Half, Split, Term, Lag]>,
    <X as Memory>::Tensor<bf16, C, m![Lane, Split, Term, Reach]>,
    <X as Memory>::Tensor<i32, C, m![Channels]>,
    <X as Memory>::Tensor<f32, C, m![Channels]>,
>;

/// Post activation, then the bias-free seven-tap output convolution and waveform clamp.
pub type Head<C = Chips, X = Hbm> = Then<
    Named<"activation_post", Activation<C8, C, X>>,
    Named<"conv_post", Waveform<C8, C, <X as Memory>::Tensor<bf16, C, m![Lane, Term, Lag, C8]>>>,
>;

/// Stereo latent rows, 207 live out of 208 allocated rows per channel.
pub type Input<C = Chips> = crate::ops::convolution::input::Rows<Latent, C>;

/// Stereo waveform, 165,600 live samples per channel in a 166,400-sample envelope.
/// Consumers trim each channel to `Samples::SAMPLES`, rather than concatenate padding as audio.
pub type Output<C = Chips> = HbmTensor<f32, C, m![Stereo, { Symbol<Samples> }]>;

/// Allocated rows, slice block sizes and live samples at each successive upsampling stage.
pub type Latent = Span<208, 8, 207>;
pub type Fifth = Span<1040, 8, 1035>;
pub type TwentyFifth = Span<5200, 40, 5175>;
pub type Fiftieth = Span<10400, 80, 10350>;
pub type Hundredth = Span<20800, 104, 20700>;
pub type TwoHundredth = Span<41600, 200, 41400>;
pub type FourHundredth = Span<83200, 400, 82800>;
pub type Samples = Span<166400, 800, 165600>;

impl Upsampled<5> for Latent {
    type Output = Fifth;
}
impl Upsampled<5> for Fifth {
    type Output = TwentyFifth;
}
impl Upsampled<2> for TwentyFifth {
    type Output = Fiftieth;
}
impl Upsampled<2> for Fiftieth {
    type Output = Hundredth;
}
impl Upsampled<2> for Hundredth {
    type Output = TwoHundredth;
}
impl Upsampled<2> for TwoHundredth {
    type Output = FourHundredth;
}
impl Upsampled<2> for FourHundredth {
    type Output = Samples;
}

impl Doubled for Fifth {
    type Twice = Span<2080, 16, 2070>;
}
impl Doubled for TwentyFifth {
    type Twice = Span<10400, 80, 10350>;
}
impl Doubled for Fiftieth {
    type Twice = Span<20800, 160, 20700>;
}
impl Doubled for Hundredth {
    type Twice = Span<41600, 208, 41400>;
}
impl Doubled for TwoHundredth {
    type Twice = Span<83200, 400, 82800>;
}
impl Doubled for FourHundredth {
    type Twice = Span<166400, 800, 165600>;
}
impl Doubled for Samples {
    type Twice = Span<332800, 1600, 331200>;
}

axes![
    C512 = 512,
    C256 = 256,
    C128 = 128,
    C64 = 64,
    C32 = 32,
    C16 = 16,
    C8 = 8,
    K3 = 3,
    K11 = 11
];

#[cfg(test)]
mod tests {
    use super::Fifth as Wide;
    use crate::Module;
    use crate::modules::convolution::{
        AntiAliased,
        tests::{Case, KAISER},
    };
    use crate::ops::convolution::frames::{self, Frames, Stereo, Window};
    use crate::tensor::parallel::Chip;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    axes![W = 512];

    #[device(chip = 1)]
    fn wide(
        device: &mut Device,
        input: &Frames<f32, W, Wide>,
        module: &AntiAliased<W>,
    ) -> Frames<bf16, W, Wide> {
        module.forward(device, (), &mut (), input)
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
