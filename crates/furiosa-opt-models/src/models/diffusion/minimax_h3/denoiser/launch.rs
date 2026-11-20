//! Borrowed numerical compositions for the separately launched H3 denoising stages.

use super::super::request;
use crate::Module;
use crate::models::diffusion::minimax_h3::{
    denoiser::{self, BlockState},
    embeddings::{self, Chips, Condition as Encoded, Latent, Patch, Width},
    joint, transformer,
};
use crate::modules::attention;
use crate::ops::{
    attention::flash,
    index::{
        gather::{Gather, RowStorage},
        packed::{Join, Split, Timestep, Trim},
    },
};
use crate::tensor::{axis::Rows, memory::Hbm, scalar::Value};
use furiosa_opt_std::prelude::*;

/// Joined text/audio/video hidden rows on the pipeline's four chips.
pub type Hidden<R> = HbmTensor<bf16, Chips, m![R, Width]>;
/// FP32 video patches or velocities.
pub type Video<R> = HbmTensor<f32, Chips, m![R, Patch]>;
/// FP32 audio latents or velocities.
pub type Audio<R> = HbmTensor<f32, Chips, m![R, Latent]>;
/// FP32 `[64, 8]` timesteps, with video and audio in column zero of rows zero and one.
pub type Times = HbmTensor<f32, Chips, m![{ Symbol<Rows<64>> }, 1 # 8]>;

/// Source offsets select actual encoder rows and a separately initialized zero row for padding.
/// Every gathered row is initialized before the existing condition projection reads it.
#[derive(DeviceSend)]
pub struct Condition<'a, 'w: 'a> {
    /// Borrowed text projection weights.
    pub model: &'a embeddings::Text<Encoded, Width, Chips, &'w Hbm>,
}

/// Encoder row storage and checked gather offsets, including a zero-padding row.
pub type EncoderRows<'a, R> = (
    &'a RowStorage<Encoded, bf16, Chips>,
    &'a HbmTensor<i32, Chips, m![R]>,
);

impl<'i, R: AxisName> Module<EncoderRows<'i, R>, (), &mut ()> for Condition<'_, '_> {
    type Output = Hidden<R>;

    fn forward(
        &self,
        device: &mut Device,
        _: (),
        _: &mut (),
        source: EncoderRows<'i, R>,
    ) -> Self::Output {
        let encoded = crate::Op::forward(&Gather, &mut *device, source);
        self.model.forward(device, (), &mut (), &encoded)
    }
}

/// An independently launched text-refiner layer with all attention heads on each chip.
#[derive(DeviceSend)]
pub struct RefinerLayer<'a, 'w: 'a> {
    /// Borrowed weights for one text-refiner layer.
    pub model:
        &'a transformer::RefinerLayer<Width, transformer::Attended, transformer::Expanded, &'w Hbm>,
}

type RefinerQkvState = ((), ((), (), ()));
type RefinerLayerState = ((RefinerQkvState, ()), ((), ((), ())));

impl<'e, const N: usize, K: AxisName>
    Module<
        Hidden<crate::ops::index::packed::Span<N, N>>,
        (&'e flash::Visible<K, Chips>, &'e Value<Chips>),
        &mut (),
    > for RefinerLayer<'_, '_>
{
    type Output = Hidden<crate::ops::index::packed::Span<N, N>>;
    fn forward(
        &self,
        device: &mut Device,
        (mask, bound): (&'e flash::Visible<K, Chips>, &'e Value<Chips>),
        _: &mut (),
        input: Hidden<crate::ops::index::packed::Span<N, N>>,
    ) -> Self::Output {
        let context = attention::Context {
            positions: (),
            write: (),
            read: (mask, bound),
        };
        let mut state: RefinerLayerState = Default::default();
        self.model.forward(device, &context, &mut state, input)
    }
}

/// Final normalization of the separately launched text refiner.
#[derive(DeviceSend)]
pub struct RefinerNorm<'a, 'w: 'a> {
    /// Borrowed final normalization weights.
    pub model: &'a transformer::Norm<Width, &'w Hbm>,
}

impl<'i, const N: usize> Module<&'i Hidden<crate::ops::index::packed::Span<N, N>>, (), &mut ()>
    for RefinerNorm<'_, '_>
{
    type Output = Hidden<crate::ops::index::packed::Span<N, N>>;
    fn forward(
        &self,
        device: &mut Device,
        _: (),
        _: &mut (),
        input: &'i Hidden<crate::ops::index::packed::Span<N, N>>,
    ) -> Self::Output {
        self.model.forward(device, (), &mut (), input)
    }
}

/// The two embedded timesteps retained beside the block loop.
pub type EmbeddedTime = HbmTensor<f32, Chips, m![Timestep, { Symbol<embeddings::Time> }]>;
/// Timesteps, input media and the request's conditioned text rows.
pub type StepInput<'a, T, A, V> = (&'a Times, (&'a Video<V>, &'a Audio<A>), &'a Hidden<T>);

/// Project media and timesteps, then join media with the request's conditioned text.
#[derive(DeviceSend)]
pub struct Embed<'a, 'w: 'a> {
    /// Borrowed media and timestep projections.
    pub model: &'a denoiser::Embeddings<&'w Hbm>,
}

impl<'i, 'e, R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName>
    Module<StepInput<'i, T, A, V>, &'e request::Inputs<R, K, T, A, V>, &mut ()> for Embed<'_, '_>
{
    type Output = (Hidden<R>, EmbeddedTime);
    fn forward(
        &self,
        device: &mut Device,
        inputs: &'e request::Inputs<R, K, T, A, V>,
        _: &mut (),
        (times, media, text): StepInput<'i, T, A, V>,
    ) -> Self::Output {
        let (time, media) = self
            .model
            .forward(&mut *device, (), &mut (), (times, media));
        let time = crate::Op::forward(&Trim, &mut *device, &time);
        let rows = inputs.rows.read();
        let hidden = crate::Op::forward(&Join, device, (&media, text, &rows));
        (hidden.into_value(), time)
    }
}

/// One launch per transformer block. The request owns transport; the model owns only this layer.
#[derive(DeviceSend)]
pub struct Block<'a, 'w: 'a> {
    /// Borrowed transformer block weights.
    pub model: &'a denoiser::Block<&'w Hbm>,
}

impl<'i, 'e, R: AxisName, K: AxisName, T, A, V>
    Module<(Hidden<R>, &'i EmbeddedTime), &'e request::Inputs<R, K, T, A, V>, &mut ()>
    for Block<'_, '_>
{
    type Output = Hidden<R>;
    fn forward(
        &self,
        device: &mut Device,
        inputs: &'e request::Inputs<R, K, T, A, V>,
        _: &mut (),
        (hidden, time): (Hidden<R>, &'i EmbeddedTime),
    ) -> Self::Output {
        let call = inputs.blocks.read();
        let context = joint::Context::new(&call, &inputs.rotation, &inputs.visible);
        let mut state: BlockState = Default::default();
        self.model
            .forward(
                device,
                (time, (call.spans(), context.environment())),
                &mut state,
                call.prefix().interval().with(hidden),
            )
            .into_value()
    }
}

/// Split the final joint rows and project video and audio velocities.
#[derive(DeviceSend)]
pub struct Heads<'a, 'w: 'a> {
    /// Borrowed video and audio readout weights.
    pub model: &'a denoiser::Heads<&'w Hbm>,
}

impl<'i, 'e, R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName>
    Module<(&'i Hidden<R>, &'i EmbeddedTime), &'e request::Inputs<R, K, T, A, V>, &mut ()>
    for Heads<'_, '_>
{
    type Output = (Video<V>, Audio<A>);
    fn forward(
        &self,
        device: &mut Device,
        inputs: &'e request::Inputs<R, K, T, A, V>,
        _: &mut (),
        (hidden, time): (&'i Hidden<R>, &'i EmbeddedTime),
    ) -> Self::Output {
        let rows = inputs.rows.read();
        let media = crate::Op::forward(&Split, &mut *device, (hidden, &rows));
        self.model
            .forward(device, (), &mut ((), ()), (&media, time))
    }
}
