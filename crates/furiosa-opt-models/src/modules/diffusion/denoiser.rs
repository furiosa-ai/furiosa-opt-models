//! The denoising network: input embeddings (one for a sequence that packs several streams), layer
//! bodies sharing one condition, control branches and condition-bearing heads.

use crate::load::Load;
use crate::tensor::image::Vector;
use crate::{First, Layers, Module, Unrolled};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Embeds the latent, conditions a layer stack and applies its head without geometry restoration.
pub type Denoiser<Embedding, Conditioning, Collection, Head> =
    Transformer<Embedding, Conditioning, Layers<Collection, Unrolled>, Unpatch<Head, First>>;

/// Embeds the input and the condition, runs the body and applies the head.
#[derive(DeviceSend, Load)]
pub struct Transformer<Embedding, Conditioning, Body, Head> {
    #[load(rename = "")]
    pub embedding: Embedding,
    #[load(rename = "")]
    pub conditioning: Conditioning,
    #[load(rename = "")]
    pub body: Body,
    #[load(rename = "")]
    pub head: Head,
}

/// The input is the sample, the encoded text and the time; the env holds the body's reader and the
/// head's. The body reads the condition in its env, the head beside the hidden values.
impl<
    's,
    'x,
    I,
    C,
    T,
    X,
    K,
    Y,
    O,
    BodyEnv: Copy,
    HeadEnv: Copy,
    EmbeddingState,
    ConditioningState,
    BodyState,
    HeadState,
    Embedding,
    Conditioning,
    Body,
    Head,
>
    Module<
        (I, &'x C, T),
        (BodyEnv, HeadEnv),
        &'s mut (EmbeddingState, ConditioningState, BodyState, HeadState),
    > for Transformer<Embedding, Conditioning, Body, Head>
where
    Embedding: Module<(I, &'x C), (BodyEnv, HeadEnv), &'s mut EmbeddingState, Output = X>,
    Conditioning: Module<(T, &'x C), (BodyEnv, HeadEnv), &'s mut ConditioningState, Output = K>,
    Body: for<'k> Module<X, (&'k K, BodyEnv), &'s mut BodyState, Output = Y>,
    Head: for<'y, 'k> Module<(&'y Y, &'k K), HeadEnv, &'s mut HeadState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: (BodyEnv, HeadEnv),
        (embedding, conditioning, body, head): &'s mut (
            EmbeddingState,
            ConditioningState,
            BodyState,
            HeadState,
        ),
        (input, encoded, time): (I, &'x C, T),
    ) -> Self::Output {
        let condition = self
            .conditioning
            .forward(&mut *device, env, conditioning, (time, encoded));
        let hidden = self
            .embedding
            .forward(&mut *device, env, embedding, (input, encoded));
        let hidden = self
            .body
            .forward(&mut *device, (&condition, env.0), body, hidden);
        self.head
            .forward(device, env.1, head, (&hidden, &condition))
    }
}

/// Timestep cosine/sine embedding with period 10000 and a half-width frequency divisor.
#[derive(DeviceSend)]
pub struct Time<Width>(pub PhantomData<Width>);

impl<Width> Default for Time<Width> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Width: AxisName, Env> Module<&HbmTensor<f32, m![1], m![1]>, Env, &mut ()> for Time<Width> {
    type Output = Vector<Width>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: &HbmTensor<f32, m![1], m![1]>,
    ) -> Self::Output {
        const { assert!(Width::SIZE % 2 == 0) };
        todo!(
            "NPU timestep cosine/sine embedding with period 10000 and half-width frequency divisor"
        )
    }
}

/// A step's packed rows and its timesteps' embedding: the timesteps embedded, then their rows
/// trimmed; each media embedded, then its span taken; then the spans packed with the text rows.
#[derive(DeviceSend, Load)]
pub struct PackedEmbedding<Time, Trim, Media, Spans, Join> {
    #[load(rename = "")]
    pub time: Time,
    #[load(rename = "")]
    pub trim: Trim,
    #[load(rename = "")]
    pub media: Media,
    #[load(rename = "")]
    pub spans: Spans,
    #[load(rename = "")]
    pub join: Join,
}

/// The state holds the media embeddings' and the spans' slots.
impl<
    's,
    T,
    M,
    X,
    E,
    U,
    V,
    W,
    S,
    O,
    MediaState,
    SpansState,
    Time,
    Trim,
    Media,
    Spans,
    Join,
    Env: Copy,
> Module<(T, M, X), Env, &'s mut (MediaState, SpansState)>
    for PackedEmbedding<Time, Trim, Media, Spans, Join>
where
    Time: for<'d> Module<T, Env, &'d mut (), Output = E>,
    Trim: for<'d, 'e> Module<&'e E, Env, &'d mut (), Output = U>,
    Media: Module<M, Env, &'s mut MediaState, Output = (V, W)>,
    Spans: for<'v> Module<(&'v V, &'v W), Env, &'s mut SpansState, Output = S>,
    Join: for<'d, 'p> Module<(&'p S, X), Env, &'d mut (), Output = O>,
{
    type Output = (O, U);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (media_state, spans): &'s mut (MediaState, SpansState),
        (times, media, text): (T, M, X),
    ) -> Self::Output {
        let embedded = self.time.forward(&mut *device, env, &mut (), times);
        let time = self.trim.forward(&mut *device, env, &mut (), &embedded);
        let media = self.media.forward(&mut *device, env, media_state, media);
        let spans = self
            .spans
            .forward(&mut *device, env, spans, (&media.0, &media.1));
        let hidden = self.join.forward(device, env, &mut (), (&spans, text));
        (hidden, time)
    }
}

/// Embeds a control input and emits a residual from each condition-bearing layer.
#[derive(DeviceSend)]
pub struct Control<Embedding, Projection, Add, Modules> {
    pub embedding: Embedding,
    pub projection: Projection,
    pub add: Add,
    pub layers: Modules,
}

/// The env holds the condition and the reader every control layer reads; the state holds each
/// control layer's slot.
impl<
    's,
    'e,
    X,
    I,
    A,
    Y,
    E,
    R,
    LayerState,
    Embedding,
    Projection,
    Add,
    Layer,
    Head,
    const N: usize,
    Env: Copy,
> Module<(&X, I), (&'e E, Env), &'s mut [LayerState; N]>
    for Control<Embedding, Projection, Add, [(Layer, Head); N]>
where
    Embedding: for<'d> Module<I, Env, &'d mut (), Output = A>,
    Projection: for<'d, 'x> Module<&'x A, Env, &'d mut (), Output = Y>,
    Add: for<'d, 'x, 'y> Module<(&'x X, &'y Y), Env, &'d mut (), Output = Y>,
    Layer: Module<Y, (&'e E, Env), &'s mut LayerState, Output = Y>,
    Head: for<'d, 'x> Module<&'x Y, Env, &'d mut (), Output = R>,
{
    type Output = [R; N];

    fn forward(
        &self,
        device: &mut Device,
        (condition, env): (&'e E, Env),
        state: &'s mut [LayerState; N],
        (input, control): (&X, I),
    ) -> Self::Output {
        let control = self.embedding.forward(&mut *device, env, &mut (), control);
        let control = self
            .projection
            .forward(&mut *device, env, &mut (), &control);
        let mut hidden = Some(
            self.add
                .forward(&mut *device, env, &mut (), (input, &control)),
        );
        let mut layers = state.iter_mut().zip(&self.layers);
        [(); N].map(|()| {
            let (slot, (layer, head)) = layers.next().expect("one slot for each control layer");
            let input = hidden
                .take()
                .expect("every layer hands its output to the next");
            let output = layer.forward(&mut *device, (condition, env), slot, input);
            let residual = head.forward(&mut *device, env, &mut (), &output);
            hidden = Some(output);
            residual
        })
    }
}

/// Computes control residuals once and injects them after each conditioned layer.
#[derive(DeviceSend)]
pub struct Controlled<Control, Modules, Inject> {
    pub control: Control,
    pub layers: Modules,
    pub inject: Inject,
}

/// The env holds the condition, then the control's reader and the one every layer reads; the state
/// holds the control's slot and, for each layer, its slot and its injection's.
impl<
    's,
    'e,
    X,
    I,
    E,
    R,
    ControlEnv: Copy,
    LayerEnv: Copy,
    ControlState,
    LayerState,
    InjectState,
    Control,
    Layer,
    Inject,
    const N: usize,
>
    Module<
        (X, I),
        (&'e E, (ControlEnv, LayerEnv)),
        &'s mut (ControlState, [(LayerState, InjectState); N]),
    > for Controlled<Control, [Layer; N], Inject>
where
    Control: for<'x> Module<(&'x X, I), (&'e E, ControlEnv), &'s mut ControlState, Output = R>,
    Layer: Module<X, (&'e E, LayerEnv), &'s mut LayerState, Output = X>,
    Inject: for<'r> Module<(X, &'r R), (&'e E, LayerEnv), &'s mut InjectState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        (condition, (control_env, env)): (&'e E, (ControlEnv, LayerEnv)),
        (control_state, layers): &'s mut (ControlState, [(LayerState, InjectState); N]),
        (hidden, control): (X, I),
    ) -> X {
        let residuals = self.control.forward(
            &mut *device,
            (condition, control_env),
            control_state,
            (&hidden, control),
        );
        layers.iter_mut().zip(&self.layers).fold(
            hidden,
            |hidden, ((layer_state, inject_state), layer)| {
                let next = layer.forward(&mut *device, (condition, env), layer_state, hidden);
                self.inject.forward(
                    &mut *device,
                    (condition, env),
                    inject_state,
                    (next, &residuals),
                )
            },
        )
    }
}

/// Runs two streams, joins them for a shared body and selects output using the retained spans.
#[derive(DeviceSend)]
pub struct DualStream<First, Join, Second, Select> {
    pub first: First,
    pub join: Join,
    pub second: Second,
    pub select: Select,
}

/// The env holds the condition, then the first stage's reader and the second's; the state holds
/// both stages' slots.
impl<
    's,
    'e,
    X,
    Y,
    Z,
    W,
    O,
    S,
    E,
    FirstEnv: Copy,
    SecondEnv: Copy,
    FirstState,
    SecondState,
    First,
    Join,
    Second,
    Select,
> Module<(X, Y), (&'e E, (FirstEnv, SecondEnv)), &'s mut (FirstState, SecondState)>
    for DualStream<First, Join, Second, Select>
where
    First: Module<(X, Y), (&'e E, FirstEnv), &'s mut FirstState, Output = (X, Y)>,
    Join:
        for<'d, 'x> Module<&'x (X, Y), (&'e E, (FirstEnv, SecondEnv)), &'d mut (), Output = (Z, S)>,
    Second: Module<Z, (&'e E, SecondEnv), &'s mut SecondState, Output = W>,
    Select: for<'d, 'x> Module<(&'x W, S), (&'e E, (FirstEnv, SecondEnv)), &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: (&'e E, (FirstEnv, SecondEnv)),
        (first, second): &'s mut (FirstState, SecondState),
        hidden: (X, Y),
    ) -> Self::Output {
        let (condition, (first_env, second_env)) = env;
        let hidden = self
            .first
            .forward(&mut *device, (condition, first_env), first, hidden);
        let (hidden, spans) = self.join.forward(&mut *device, env, &mut (), &hidden);
        let hidden = self
            .second
            .forward(&mut *device, (condition, second_env), second, hidden);
        self.select.forward(device, env, &mut (), (&hidden, spans))
    }
}

/// Refines image and text streams separately, then joins them for a conditioned body.
#[derive(DeviceSend, Load)]
pub struct Refined<Image, Text, Join, Body> {
    #[load(rename = "")]
    pub image: Image,
    #[load(rename = "")]
    pub text: Text,
    #[load(rename = "")]
    pub join: Join,
    #[load(rename = "")]
    pub body: Body,
}

/// The env holds the condition, then the image refiner's, the text refiner's and the body's
/// readers; the text refiner reads no condition. The state holds the three stages' slots.
impl<
    's,
    'e,
    X,
    Y,
    A,
    B,
    Z,
    O,
    E,
    ImageEnv: Copy,
    TextEnv: Copy,
    BodyEnv: Copy,
    ImageState,
    TextState,
    BodyState,
    Image,
    Text,
    Join,
    Body,
> Module<(X, Y), (&'e E, (ImageEnv, TextEnv, BodyEnv)), &'s mut (ImageState, TextState, BodyState)>
    for Refined<Image, Text, Join, Body>
where
    Image: Module<X, (&'e E, ImageEnv), &'s mut ImageState, Output = A>,
    Text: Module<Y, TextEnv, &'s mut TextState, Output = B>,
    Join: for<'d, 'a, 'b> Module<
            (&'a A, &'b B),
            (&'e E, (ImageEnv, TextEnv, BodyEnv)),
            &'d mut (),
            Output = Z,
        >,
    Body: Module<Z, (&'e E, BodyEnv), &'s mut BodyState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: (&'e E, (ImageEnv, TextEnv, BodyEnv)),
        (image_state, text_state, body): &'s mut (ImageState, TextState, BodyState),
        (image, text): (X, Y),
    ) -> Self::Output {
        let (condition, (image_env, text_env, body_env)) = env;
        let image = self
            .image
            .forward(&mut *device, (condition, image_env), image_state, image);
        let text = self.text.forward(&mut *device, text_env, text_state, text);
        let hidden = self
            .join
            .forward(&mut *device, env, &mut (), (&image, &text));
        self.body
            .forward(device, (condition, body_env), body, hidden)
    }
}

/// Runs a conditioned body, then a final stage that borrows its output and the same condition.
#[derive(DeviceSend, Load)]
pub struct Terminal<Body, Last> {
    #[load(rename = "")]
    pub body: Body,
    #[load(rename = "")]
    pub last: Last,
}

/// The env holds the condition, then the body's reader and the last stage's; the state holds both
/// stages' slots.
impl<'s, 'e, X, Y, O, E, BodyEnv: Copy, LastEnv: Copy, BodyState, LastState, Body, Last>
    Module<X, (&'e E, (BodyEnv, LastEnv)), &'s mut (BodyState, LastState)> for Terminal<Body, Last>
where
    Body: Module<X, (&'e E, BodyEnv), &'s mut BodyState, Output = Y>,
    Last: for<'y> Module<&'y Y, (&'e E, LastEnv), &'s mut LastState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        (condition, (body_env, last_env)): (&'e E, (BodyEnv, LastEnv)),
        (body, last): &'s mut (BodyState, LastState),
        hidden: X,
    ) -> Self::Output {
        let hidden = self
            .body
            .forward(&mut *device, (condition, body_env), body, hidden);
        self.last
            .forward(device, (condition, last_env), last, &hidden)
    }
}

/// Applies a condition-bearing head and restores its output using caller-supplied geometry.
#[derive(DeviceSend, Load)]
pub struct Unpatch<Head, Restore> {
    #[load(rename = "")]
    pub head: Head,
    #[load(rename = "")]
    pub restore: Restore,
}

/// The env is the geometry the restore reads beside the head's output; the state is the head's
/// slot.
impl<X, E, G: Copy, S, Y, O, Head, Restore> Module<(&X, &E), G, S> for Unpatch<Head, Restore>
where
    Head: for<'x, 'e> Module<(&'x X, &'e E), G, S, Output = Y>,
    Restore: for<'d> Module<(Y, G), G, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, shape: G, state: S, input: (&X, &E)) -> O {
        let hidden = self.head.forward(&mut *device, shape, state, input);
        self.restore
            .forward(device, shape, &mut (), (hidden, shape))
    }
}

/// Modulates the final norm from the condition and projects normalized hidden values to output.
#[derive(DeviceSend, Load)]
pub struct Head<Modulation, Norm, Projection> {
    #[load(rename = "")]
    pub modulation: Modulation,
    #[load(rename = "")]
    pub norm: Norm,
    #[load(rename = "")]
    pub projection: Projection,
}

/// The state is the modulation's slot.
impl<X, E, M, Y, O, S, Modulation, Norm, Projection, Env: Copy> Module<(&X, &E), Env, S>
    for Head<Modulation, Norm, Projection>
where
    Modulation: for<'x> Module<&'x E, Env, S, Output = M>,
    Norm: for<'a, 'b, 's> Module<(&'a X, &'b M), Env, &'s mut (), Output = Y>,
    Projection: for<'x, 's> Module<&'x Y, Env, &'s mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, state: S, (hidden, condition): (&X, &E)) -> O {
        let modulation = self.modulation.forward(&mut *device, env, state, condition);
        let hidden = self
            .norm
            .forward(&mut *device, env, &mut (), (hidden, &modulation));
        self.projection.forward(device, env, &mut (), &hidden)
    }
}

/// Applies independent video and audio heads using the same condition.
#[derive(DeviceSend, Load)]
pub struct PairedHead<Video, Audio> {
    #[load(rename = "")]
    pub video: Video,
    #[load(rename = "")]
    pub audio: Audio,
}

/// The state holds the video head's slot, then the audio head's.
impl<'s, X, Y, E, V, A, VideoState, AudioState, Video, Audio, Env: Copy>
    Module<(&(X, Y), &E), Env, &'s mut (VideoState, AudioState)> for PairedHead<Video, Audio>
where
    Video: for<'a, 'b> Module<(&'a X, &'b E), Env, &'s mut VideoState, Output = V>,
    Audio: for<'a, 'b> Module<(&'a Y, &'b E), Env, &'s mut AudioState, Output = A>,
{
    type Output = (V, A);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (video_state, audio_state): &'s mut (VideoState, AudioState),
        ((video, audio), condition): (&(X, Y), &E),
    ) -> Self::Output {
        let video = self
            .video
            .forward(&mut *device, env, video_state, (video, condition));
        let audio = self
            .audio
            .forward(device, env, audio_state, (audio, condition));
        (video, audio)
    }
}
