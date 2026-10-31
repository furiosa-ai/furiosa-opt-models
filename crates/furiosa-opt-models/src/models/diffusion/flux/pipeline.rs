//! Text-conditioned flow matching with caller-owned noise and sigma schedules.
use super::{AXES, Head, Width};
use crate::models::Scalar;
use crate::models::diffusion::flux::FluxPipeline;
use crate::ops::index::sequence::Sum;
use crate::ops::index::table::Offset;
use crate::ops::rotary::Axial;
use crate::{First, Identity, Module, Sequential, Then};
use crate::{
    models::{diffusion::autoencoder, encoder::clip, encoder_decoder::t5},
    modules::diffusion::{self, flow_match},
    ops::{
        convolution::Double,
        pointwise::{
            activation::NewGelu,
            guidance,
            multiply::Scale,
            rescale::{Denormalize, Rescale},
        },
    },
    tensor::image::{ChannelFirst, Grid, Image, Pack, Rows, Unpatch, Vector},
};
use encode::rotary;
use furiosa_opt_std::prelude::*;

/// Text encoding, guided flow matching, and latent decoding composition.
pub type Model<G> = FluxPipeline<
    encode::Clip,
    encode::T5,
    rotary::Model,
    Prepare,
    diffusion::Guider<predict::Model<G>, guidance::Combine<SampleShape, SampleShape>>,
    Update,
    Decode,
>;

/// Pooled text, token text, and coordinate-dependent rotary conditioning.
pub mod encode {
    use super::super::Encoded;
    use super::super::Phases;
    use super::*;

    /// Computes both text representations and their joint rotary phases.
    #[derive(DeviceSend)]
    pub struct Model<Clip = self::Clip, T5 = self::T5, Rotary = rotary::Model> {
        /// Pooled text tower.
        pub clip: Clip,
        /// Token-level text tower.
        pub t5: T5,
        /// Position-phase construction from three-axis coordinates.
        pub rotary: Rotary,
    }

    /// The state holds each text tower's slot.
    impl<
        's,
        Clip,
        T5,
        Rotary,
        C,
        T,
        R,
        ClipState,
        T5State,
        Text: AxisName,
        Patches: AxisName,
        Env: Copy,
    > Module<&mut Context<C, T, R>, Env, &'s mut (ClipState, T5State)> for Model<Clip, T5, Rotary>
    where
        Clip: for<'i> Module<&'i mut C, Env, &'s mut ClipState, Output = Vector<Width<768>>>,
        T5: for<'i> Module<&'i mut T, Env, &'s mut T5State, Output = Rows<Width<4096>, Text>>,
        Rotary: for<'x, 'd> crate::Module<&'x R, Env, &'d mut (), Output = Phases<Patches, Text>>,
    {
        type Output = Condition<Text, Patches>;

        fn forward(
            &self,
            device: &mut Device,
            env: Env,
            (clip, t5): &'s mut (ClipState, T5State),
            context: &mut Context<C, T, R>,
        ) -> Self::Output {
            let pooled = self
                .clip
                .forward(&mut *device, env, clip, &mut context.clip);
            let tokens = self.t5.forward(&mut *device, env, t5, &mut context.t5);
            let phases = self
                .rotary
                .forward(device, env, &mut (), &context.coordinates);
            (Encoded { tokens, pooled }, phases)
        }
    }

    impl<'s, Clip, T5, Rotary, C, T, R, S, Text: AxisName, Patches: AxisName, Env: Copy>
        Module<Context<C, T, R>, Env, &'s mut S> for Model<Clip, T5, Rotary>
    where
        Self: for<'i> Module<
                &'i mut Context<C, T, R>,
                Env,
                &'s mut S,
                Output = Condition<Text, Patches>,
            >,
    {
        type Output = Condition<Text, Patches>;

        fn forward(
            &self,
            device: &mut Device,
            env: Env,
            state: &'s mut S,
            mut context: Context<C, T, R>,
        ) -> Self::Output {
            self.forward(device, env, state, &mut context)
        }
    }

    /// Unprojected pooled text features from the smaller text tower.
    pub type Clip = clip::Encoder<
        clip::Text<Width<49408>, Tokens<77>, Width<768>, Width<3072>, Width<12>, 12>,
        Identity,
    >;

    /// Token-level text features from the gated feed-forward tower.
    pub type T5 = t5::Encoder<
        Width<32128>,
        Width<4096>,
        Width<4096>,
        Width<64>,
        Width<32>,
        t5::Gated<Width<4096>, Width<10240>, NewGelu>,
        24,
    >;

    /// Joint-stream rotary phases derived from three-axis coordinates.
    pub mod rotary {
        use super::*;

        pub type Model = Axial<Head, Width<{ AXES[0] }>, Width<{ AXES[1] }>, Width<{ AXES[2] }>>;
        pub type Output = super::super::super::Phases<Patches, Text>;
    }

    /// Three coordinates per joint row: stream, spatial row, and spatial column.
    pub type Coordinates = HbmTensor<f32, m![1], m![{Symbol<Sum<Text,Patches>>},3]>;

    /// Text operands and byte offset of the selected pooling row.
    pub type ClipInput = (clip::Operands<Width<768>, Tokens<77>, 12>, Offset);

    /// Text operands and coordinates in the same joint row order as attention.
    #[derive(DeviceSend)]
    pub struct Context<C = ClipInput, T = t5::Context<Width<4096>, Text, 24>, R = Coordinates> {
        /// Pooled text inputs and row selection.
        pub clip: C,
        /// Token text inputs and writable layer captures.
        pub t5: T,
        /// Text rows precede image rows; coordinates are (stream, row, column).
        pub coordinates: R,
    }

    /// Token features, pooled vector, and text-then-image rotary phases.
    pub type Output = (
        (Rows<Width<4096>, Text>, Vector<Width<768>>),
        Phases<Patches, Text>,
    );

    /// Text-only conditioning stage, independent of image coordinates.
    pub mod text {
        use super::*;

        pub type Model = (T5, Clip);
        pub type Output = (Rows<Width<4096>, Text>, Vector<Width<768>>);
    }
}

/// Packs each 2×2 latent patch into one feature row.
pub type Prepare = Then<First, Pack<Width<64>, Width<2>, Width<2>, ChannelFirst>>;

/// Latent velocity prediction with learned guidance scaled to training units.
pub mod predict {
    use super::super::Guidance;
    use super::*;
    use crate::modules::attention;
    use std::borrow::Borrow;

    /// Transformer plus learned-guidance conversion to training-timestep units.
    #[derive(DeviceSend)]
    pub struct Model<G> {
        /// Joint text/image latent predictor.
        pub transformer: super::super::Model<Patches, Text, G>,
        /// Multiplier converting learned guidance into embedding units.
        pub guidance: Scale,
    }

    impl<G: Guidance, P: Borrow<Context<G>>, Env: Copy>
        Module<(&Sample, (&Condition, P)), Env, &mut ()> for Model<G>
    {
        type Output = Sample;

        fn forward(
            &self,
            device: &mut Device,
            _: Env,
            _: &mut (),
            (sample, ((encoded, phases), context)): (&Sample, (&Condition, P)),
        ) -> Sample {
            let context = context.borrow();
            let guidance = G::map(&self.guidance, &mut *device, &context.guidance);
            let attention = attention::Context {
                positions: phases,
                write: (),
                read: (),
            };
            self.transformer.forward(
                device,
                (((&attention, ()), &attention), ()),
                &mut (
                    Default::default(),
                    Default::default(),
                    (Default::default(), [Default::default(); _]),
                    Default::default(),
                ),
                (sample, encoded, (&context.time, &guidance)),
            )
        }
    }

    /// Per-step training time and the learned-guidance value `G` reads.
    #[derive(DeviceSend)]
    pub struct Context<G: Guidance> {
        /// Training timestep (1000 * sigma); the schedule chooses sigma.
        pub time: Scalar,
        /// Learned guidance, distinct from classifier-free guidance.
        pub guidance: G::Input,
    }

    /// Packed latent patches with reusable conditioning and per-step values.
    #[derive(DeviceSend)]
    pub struct Input<G: Guidance> {
        /// Current packed latent patches.
        pub sample: Sample,
        /// Text features and joint rotary phases.
        pub condition: Condition,
        /// Training time and learned guidance for this prediction.
        pub context: Context<G>,
    }

    /// The sample, condition and per-step values a launch's input holds.
    impl<G: Guidance, Env: Copy> Module<&mut Input<G>, Env, &mut ()> for Model<G> {
        type Output = Sample;

        fn forward(
            &self,
            device: &mut Device,
            env: Env,
            _: &mut (),
            input: &mut Input<G>,
        ) -> Sample {
            self.forward(
                device,
                env,
                &mut (),
                (&input.sample, (&input.condition, &input.context)),
            )
        }
    }
}

/// Deterministic Euler update between caller-supplied sigmas.
pub type Update = flow_match::Update<SampleShape>;

/// Unpacks patches, reverses latent scaling/shift, and decodes clipped RGB pixels.
pub type Decode = Then<
    Unpatch<autoencoder::Channels<16>, Height, Columns, Width<2>, Width<2>, ChannelFirst>,
    Then<Rescale, Then<Decoder, Denormalize>>,
>;

/// Convolutional decoder of an unprojected sixteen-channel latent.
pub type Decoder = Sequential<
    Identity,
    autoencoder::DecoderBody<
        autoencoder::Channels<3>,
        autoencoder::Channels<16>,
        autoencoder::Channels<128>,
        autoencoder::Channels<256>,
        autoencoder::Channels<512>,
        autoencoder::Channels<32>,
        2,
    >,
>;

/// Row-major grid of 2×2 latent patches.
pub type Patches = Grid<Height, Columns, Width<2>, Width<2>>;

/// Padded token count of the T5 conditioning sequence.
pub type Text = Tokens<512>;

/// Packed patch rows and their 64 features.
pub type SampleShape = m![{Symbol<Patches>}, {Symbol<Width<64>>}];

/// Packed latent patches used by prediction and integration.
pub type Sample = super::Sample<Patches>;

/// Caller-supplied starting noise in channel, row, column order.
pub type Noise = Image<autoencoder::Channels<16>, Height, Columns>;

/// Decoded RGB image clipped to [0, 1].
pub type Pixels = Image<
    autoencoder::Channels<3>,
    Double<Double<Double<Height>>>,
    Double<Double<Double<Columns>>>,
>;

/// Text features paired with rotary phases in text-then-image order.
pub type Condition<T = Text, P = Patches> = (super::Encoded<T>, super::Phases<P, T>);

/// Prediction conditions and classifier-free scale paired with a sigma update.
pub type Step<G, const N: usize> = (
    ([predict::Context<G>; N], guidance::Factor),
    flow_match::Timestep,
);

/// Rotary base for the three coordinate axes.
pub const THETA: f32 = 10000.0;

/// Training timesteps per scheduler sigma or learned-guidance unit.
pub const TIMESTEPS: f32 = 1000.0;

/// Number of token or patch rows in a compiled stage.
#[derive(Clone, Debug)]
pub struct Tokens<const N: usize>;

impl<const N: usize> AxisName for Tokens<N> {
    const NAME: Ident = Ident::new("Tokens");
    const SIZE: usize = N;
}

/// Latent image row count before packing.
pub type Height = autoencoder::Height<64>;

/// Latent image column count before packing.
pub type Columns = autoencoder::Width<64>;
