//! Class-conditioned DDIM generation; noise and the complete timestep schedule are caller inputs.
use super::{Channels, Columns, Height, IN_CHANNELS, OUT_CHANNELS, SAMPLE_SIZE, Time};
use crate::{First, Identity, Then};
use crate::{
    models::diffusion::autoencoder,
    modules::diffusion::{self, ddim},
    ops::{
        convolution::Double,
        pointwise::{guidance, multiply::Scale, rescale::Denormalize},
    },
};
use furiosa_opt_std::prelude::*;

/// Class-conditioned DDIM integration followed by latent decoding.
pub type Model = diffusion::Pipeline<
    Identity,
    First,
    diffusion::Sampler<diffusion::Guider<super::Model, Combination>, Update>,
    Decode,
>;

/// Classifier-free combination of conditioned and unconditioned predictions.
pub type Combination = guidance::Combine<PredictionShape, SampleShape>;

/// DDIM latent update using the caller's training-timestep schedule.
pub type Update = ddim::Update<SampleShape, Training>;

/// Reverses latent scaling, decodes RGB, and clips pixels to [0, 1].
pub type Decode = Then<Scale<guidance::Factor>, Then<Decoder, Denormalize>>;

/// Three-stage convolutional decoder with a projected four-channel latent.
pub type Decoder = autoencoder::Decoder<
    Channels<3>,
    Channels<IN_CHANNELS>,
    autoencoder::Channels<128>,
    autoencoder::Channels<256>,
    autoencoder::Channels<512>,
    autoencoder::Channels<32>,
    2,
>;

/// Four-channel latent image in channel, row, column order.
pub type SampleShape = m![
    {Symbol<Channels<IN_CHANNELS>>},
    {Symbol<Height<SAMPLE_SIZE>>},
    {Symbol<Columns<SAMPLE_SIZE>>}
];

/// Eight channels containing noise and variance predictions.
pub type PredictionShape = m![
    {Symbol<Channels<OUT_CHANNELS>>},
    {Symbol<Height<SAMPLE_SIZE>>},
    {Symbol<Columns<SAMPLE_SIZE>>}
];

type Expanded<A> = Double<Double<Double<A>>>;

/// RGB image after three spatial upsampling stages.
pub type PixelShape = m![
    {Symbol<Channels<3>>},
    {Symbol<Expanded<Height<SAMPLE_SIZE>>>},
    {Symbol<Expanded<Columns<SAMPLE_SIZE>>>}
];

/// RGB samples rescaled and clipped to [0, 1].
pub type Pixels = HbmTensor<f32, m![1], PixelShape>;

/// Paired prediction times, classifier-free scale, and DDIM update timestep.
pub type Step = (([Time; 2], guidance::Factor), ddim::Timestep);

axes![Training = 1000];
