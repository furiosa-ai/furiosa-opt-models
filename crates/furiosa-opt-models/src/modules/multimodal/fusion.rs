//! Feature fusion: deep media features injected after prefix layers.

use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend, unroll};

/// Merges media into token embeddings and adds deep visual features after each prefix layer.
#[derive(DeviceSend)]
pub struct DeepFusion<
    Embedding,
    Encoder,
    Project,
    Merge,
    Layer,
    Add,
    const K: usize,
    const TAIL: usize,
> {
    pub embedding: Embedding,
    pub encoder: Encoder,
    pub projector: Project,
    pub merge: Merge,
    /// Each prefix layer, the add of the features injected after it and their projection.
    pub prefix: [(Layer, Add, Project); K],
    pub layers: [Layer; TAIL],
}

/// The slots of a deep fusion: the embedding's, the encoder's, the projector's, each prefix layer's
/// beside its feature projection's, then each tail layer's.
type Slots<
    EmbeddingState,
    EncoderState,
    ProjectState,
    LayerState,
    const K: usize,
    const TAIL: usize,
> = (
    EmbeddingState,
    EncoderState,
    ProjectState,
    [(LayerState, ProjectState); K],
    [LayerState; TAIL],
);

impl<Embedding, Encoder, Project, Merge, Layer, Add, const K: usize, const TAIL: usize>
    DeepFusion<Embedding, Encoder, Project, Merge, Layer, Add, K, TAIL>
{
    /// The prefix layers, each followed by every media's injected features, then the tail layers.
    fn finish<'s, X, F, Y, S, LayerState, ProjectState, Env: Copy, const MEDIA: usize>(
        &self,
        device: &mut Device,
        env: Env,
        (prefix, tail): (
            &'s mut [(LayerState, ProjectState); K],
            &'s mut [LayerState; TAIL],
        ),
        mut hidden: X,
        features: &[([F; K], &S); MEDIA],
    ) -> X
    where
        Layer: Module<X, Env, &'s mut LayerState, Output = X>,
        Project: for<'x, 't> Module<&'x F, Env, &'t mut ProjectState, Output = Y>,
        Add: for<'x, 'y, 'b, 't> Module<(&'x X, &'y Y, &'b S), Env, &'t mut (), Output = X>,
    {
        if K > 0 {
            #[unroll]
            for (index, (slot, projection)) in prefix.iter_mut().enumerate() {
                let (layer, add, project) = &self.prefix[index];
                hidden = layer.forward(&mut *device, env, slot, hidden);
                if MEDIA > 0 {
                    #[unroll]
                    for media in 0..MEDIA {
                        let (features, spans) = &features[media];
                        let feature =
                            project.forward(&mut *device, env, &mut *projection, &features[index]);
                        hidden =
                            add.forward(&mut *device, env, &mut (), (&hidden, &feature, *spans));
                    }
                }
            }
        }
        self.layers.forward(device, env, tail, hidden)
    }
}

/// The encoder is a tower of its own, with an empty env, and every media's pass takes its and the
/// projector's slots again; the layers read the env.
impl<
    's,
    Tokens,
    I,
    X,
    F,
    Y,
    S,
    EmbeddingState,
    EncoderState,
    ProjectState,
    LayerState,
    Embedding,
    Encoder,
    Project,
    Merge,
    Layer,
    Add,
    const K: usize,
    const TAIL: usize,
    const MEDIA: usize,
    Env: Copy,
>
    Module<
        (Tokens, [(I, &S); MEDIA]),
        Env,
        &'s mut Slots<EmbeddingState, EncoderState, ProjectState, LayerState, K, TAIL>,
    > for DeepFusion<Embedding, Encoder, Project, Merge, Layer, Add, K, TAIL>
where
    Embedding: Module<Tokens, Env, &'s mut EmbeddingState, Output = X>,
    Encoder: for<'t> Module<I, (), &'t mut EncoderState, Output = (F, [F; K])>,
    Project: for<'x, 't> Module<&'x F, Env, &'t mut ProjectState, Output = Y>,
    Merge: for<'x, 'y, 'b, 't> Module<(&'x X, &'y Y, &'b S), Env, &'t mut (), Output = X>,
    Layer: Module<X, Env, &'s mut LayerState, Output = X>,
    Add: for<'x, 'y, 'b, 't> Module<(&'x X, &'y Y, &'b S), Env, &'t mut (), Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (embedding, encoder, projector, prefix, tail): &'s mut Slots<
            EmbeddingState,
            EncoderState,
            ProjectState,
            LayerState,
            K,
            TAIL,
        >,
        (tokens, media): (Tokens, [(I, &S); MEDIA]),
    ) -> X {
        let mut hidden = self.embedding.forward(&mut *device, env, embedding, tokens);
        let features = media.map(|(input, spans)| {
            let (features, intermediate) =
                self.encoder.forward(&mut *device, (), &mut *encoder, input);
            let features = self
                .projector
                .forward(&mut *device, env, &mut *projector, &features);
            hidden = self
                .merge
                .forward(&mut *device, env, &mut (), (&hidden, &features, spans));
            (intermediate, spans)
        });
        self.finish(device, env, (prefix, tail), hidden, &features)
    }
}

// Text-only decode does not run an encoder or manufacture empty visual tensors; the encoder's and
// the projections' slots are unused.
impl<
    's,
    Tokens,
    X,
    EmbeddingState,
    EncoderState,
    ProjectState,
    LayerState,
    Embedding,
    Encoder,
    Project,
    Merge,
    Layer,
    Add,
    const K: usize,
    const TAIL: usize,
    Env: Copy,
>
    Module<
        (Tokens, ()),
        Env,
        &'s mut Slots<EmbeddingState, EncoderState, ProjectState, LayerState, K, TAIL>,
    > for DeepFusion<Embedding, Encoder, Project, Merge, Layer, Add, K, TAIL>
where
    Embedding: Module<Tokens, Env, &'s mut EmbeddingState, Output = X>,
    Layer: Module<X, Env, &'s mut LayerState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (embedding, _, _, prefix, tail): &'s mut Slots<
            EmbeddingState,
            EncoderState,
            ProjectState,
            LayerState,
            K,
            TAIL,
        >,
        (tokens, ()): (Tokens, ()),
    ) -> X {
        let mut hidden = self.embedding.forward(&mut *device, env, embedding, tokens);
        if K > 0 {
            #[unroll]
            for (index, (slot, _)) in prefix.iter_mut().enumerate() {
                hidden = self.prefix[index]
                    .0
                    .forward(&mut *device, env, slot, hidden);
            }
        }
        self.layers.forward(device, env, tail, hidden)
    }
}

// Audio contributes initial embeddings; only visual sources provide deepstack features.
// Their token spans must be disjoint, as in the framework's multimodal token placement.
impl<
    's,
    Tokens,
    I,
    J,
    X,
    F,
    Y,
    S,
    EmbeddingState,
    AudioState,
    VisualState,
    ProjectState,
    LayerState,
    Embedding,
    Audio,
    Visual,
    Project,
    Merge,
    Layer,
    Add,
    const K: usize,
    const TAIL: usize,
    const AUDIO: usize,
    const VISUAL: usize,
    Env: Copy,
>
    Module<
        (Tokens, ([(I, &S); AUDIO], [(J, &S); VISUAL])),
        Env,
        &'s mut Slots<EmbeddingState, (AudioState, VisualState), ProjectState, LayerState, K, TAIL>,
    > for DeepFusion<Embedding, (Audio, Visual), Project, Merge, Layer, Add, K, TAIL>
where
    Embedding: Module<Tokens, Env, &'s mut EmbeddingState, Output = X>,
    Audio: for<'t> Module<I, (), &'t mut AudioState, Output = F>,
    Visual: for<'t> Module<J, (), &'t mut VisualState, Output = (F, [F; K])>,
    Project: for<'x, 't> Module<&'x F, Env, &'t mut ProjectState, Output = Y>,
    Merge: for<'x, 'y, 'b, 't> Module<(&'x X, &'y Y, &'b S), Env, &'t mut (), Output = X>,
    Layer: Module<X, Env, &'s mut LayerState, Output = X>,
    Add: for<'x, 'y, 'b, 't> Module<(&'x X, &'y Y, &'b S), Env, &'t mut (), Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (embedding, (audio_state, visual_state), projector, prefix, tail): &'s mut Slots<
            EmbeddingState,
            (AudioState, VisualState),
            ProjectState,
            LayerState,
            K,
            TAIL,
        >,
        (tokens, (audio, visual)): (Tokens, ([(I, &S); AUDIO], [(J, &S); VISUAL])),
    ) -> X {
        let mut hidden = self.embedding.forward(&mut *device, env, embedding, tokens);
        let _ = audio.map(|(input, spans)| {
            let features = self
                .encoder
                .0
                .forward(&mut *device, (), &mut *audio_state, input);
            let features = self
                .projector
                .forward(&mut *device, env, &mut *projector, &features);
            hidden = self
                .merge
                .forward(&mut *device, env, &mut (), (&hidden, &features, spans));
        });
        let features = visual.map(|(input, spans)| {
            let (features, intermediate) =
                self.encoder
                    .1
                    .forward(&mut *device, (), &mut *visual_state, input);
            let features = self
                .projector
                .forward(&mut *device, env, &mut *projector, &features);
            hidden = self
                .merge
                .forward(&mut *device, env, &mut (), (&hidden, &features, spans));
            (intermediate, spans)
        });
        self.finish(device, env, (prefix, tail), hidden, &features)
    }
}
