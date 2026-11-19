//! Borrowed component entries for request preparation and one uncached denoising step.

use super::request;
use furiosa_opt_models::Module;
use furiosa_opt_models::models::diffusion::minimax_h3::{
    denoiser::{self, BlockState},
    embeddings::{self, Chips, Condition as Encoded, Latent, Patch, Width},
    joint, transformer,
};
use furiosa_opt_models::modules::attention;
use furiosa_opt_models::ops::{
    attention::flash,
    index::{
        gather::{Gather, RowStorage},
        packed::{Join, Split, Timestep, Trim},
    },
};
use furiosa_opt_models::tensor::{axis::Rows, memory::Hbm, scalar::Value};
use furiosa_opt_std::prelude::*;

type Hidden<R> = HbmTensor<bf16, Chips, m![R, Width]>;
type Video<R> = HbmTensor<f32, Chips, m![R, Patch]>;
type Audio<R> = HbmTensor<f32, Chips, m![R, Latent]>;
type Times = HbmTensor<f32, Chips, m![{ Symbol<Rows<64>> }, 1 # 8]>;

/// Source offsets select actual encoder rows and a separately initialized zero row for padding.
/// Every gathered row is initialized before the existing condition projection reads it.
#[derive(DeviceSend)]
pub(super) struct Condition<'a, 'w: 'a> {
    pub(super) model: &'a embeddings::Text<Encoded, Width, Chips, &'w Hbm>,
}

type EncoderRows<'a, R> = (
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
        let encoded = furiosa_opt_models::Op::forward(&Gather, &mut *device, source);
        self.model.forward(device, (), &mut (), &encoded)
    }
}

#[device(chip = 4)]
pub(super) fn condition<R: AxisName>(
    device: &mut Device,
    module: &Condition<'_, '_>,
    env: (),
    state: &mut (),
    source: EncoderRows<'_, R>,
) -> Hidden<R> {
    module.forward(device, env, state, source)
}

/// The original refiner's independently launched layer, with all heads on every chip.
#[derive(DeviceSend)]
pub(super) struct RefinerLayer<'a, 'w: 'a> {
    pub(super) model:
        &'a transformer::RefinerLayer<Width, transformer::Attended, transformer::Expanded, &'w Hbm>,
}

type RefinerQkvState = ((), ((), (), ()));
type RefinerLayerState = ((RefinerQkvState, ()), ((), ((), ())));

impl<'e, const N: usize, K: AxisName>
    Module<
        Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>>,
        (&'e flash::Visible<K, Chips>, &'e Value<Chips>),
        &mut (),
    > for RefinerLayer<'_, '_>
{
    type Output = Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>>;
    fn forward(
        &self,
        device: &mut Device,
        (mask, bound): (&'e flash::Visible<K, Chips>, &'e Value<Chips>),
        _: &mut (),
        input: Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>>,
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

#[device(chip = 4)]
pub(super) fn refine<const N: usize, K: AxisName>(
    device: &mut Device,
    module: &RefinerLayer<'_, '_>,
    context: (&flash::Visible<K, Chips>, &Value<Chips>),
    state: &mut (),
    input: Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>>,
) -> Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>> {
    module.forward(device, context, state, input)
}

#[derive(DeviceSend)]
pub(super) struct RefinerNorm<'a, 'w: 'a> {
    pub(super) model: &'a transformer::Norm<Width, &'w Hbm>,
}

impl<'i, const N: usize>
    Module<&'i Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>>, (), &mut ()>
    for RefinerNorm<'_, '_>
{
    type Output = Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>>;
    fn forward(
        &self,
        device: &mut Device,
        _: (),
        _: &mut (),
        input: &'i Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>>,
    ) -> Self::Output {
        self.model.forward(device, (), &mut (), input)
    }
}

#[device(chip = 4)]
pub(super) fn refine_norm<const N: usize>(
    device: &mut Device,
    module: &RefinerNorm<'_, '_>,
    env: (),
    state: &mut (),
    input: &Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>>,
) -> Hidden<furiosa_opt_models::ops::index::packed::Span<N, N>> {
    module.forward(device, env, state, input)
}

/// The two embedded timesteps retained beside the block loop.
pub(super) type EmbeddedTime = HbmTensor<f32, Chips, m![Timestep, { Symbol<embeddings::Time> }]>;
type StepInput<'a, T, A, V> = (&'a Times, (&'a Video<V>, &'a Audio<A>), &'a Hidden<T>);

#[derive(DeviceSend)]
pub(super) struct Embed<'a, 'w: 'a> {
    pub(super) model: &'a denoiser::Embeddings<&'w Hbm>,
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
        let time = furiosa_opt_models::Op::forward(&Trim, &mut *device, &time);
        let rows = inputs.rows.read();
        let hidden = furiosa_opt_models::Op::forward(&Join, device, (&media, text, &rows));
        (hidden.into_value(), time)
    }
}

#[device(chip = 4)]
pub(super) fn embed<R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName>(
    device: &mut Device,
    module: &Embed<'_, '_>,
    inputs: &request::Inputs<R, K, T, A, V>,
    state: &mut (),
    source: StepInput<'_, T, A, V>,
) -> (Hidden<R>, EmbeddedTime) {
    module.forward(device, inputs, state, source)
}

/// One launch per original block. The request owns transport; the model owns only this layer.
#[derive(DeviceSend)]
pub(super) struct Block<'a, 'w: 'a> {
    pub(super) model: &'a denoiser::Block<&'w Hbm>,
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

#[device(chip = 4)]
pub(super) fn block<R: AxisName, K: AxisName, T, A, V>(
    device: &mut Device,
    module: &Block<'_, '_>,
    inputs: &request::Inputs<R, K, T, A, V>,
    state: &mut (),
    source: (Hidden<R>, &EmbeddedTime),
) -> Hidden<R> {
    module.forward(device, inputs, state, source)
}

#[derive(DeviceSend)]
pub(super) struct Heads<'a, 'w: 'a> {
    pub(super) model: &'a denoiser::Heads<&'w Hbm>,
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
        let media = furiosa_opt_models::Op::forward(&Split, &mut *device, (hidden, &rows));
        self.model
            .forward(device, (), &mut ((), ()), (&media, time))
    }
}

#[device(chip = 4)]
pub(super) fn heads<R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName>(
    device: &mut Device,
    module: &Heads<'_, '_>,
    inputs: &request::Inputs<R, K, T, A, V>,
    state: &mut (),
    source: (&Hidden<R>, &EmbeddedTime),
) -> (Video<V>, Audio<A>) {
    module.forward(device, inputs, state, source)
}
