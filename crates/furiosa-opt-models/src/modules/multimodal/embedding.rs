//! Media embeddings merged into token embeddings at the spans the caller supplies.

use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend, unroll};

/// Encodes and projects media, then merges its selected features into token embeddings.
#[derive(Clone, DeviceSend)]
pub struct InputEmbedding<Tokens, Encoder, Project, Merge> {
    pub tokens: Tokens,
    pub encoder: Encoder,
    pub projector: Project,
    pub merge: Merge,
}

/// The encoder is a tower of its own, with an empty env; the state holds the tokens', the
/// encoder's, the projector's and the merge's slots, which every image's pass reuses.
impl<
    's,
    K,
    I,
    S,
    X,
    F,
    Y,
    TokenState,
    EncoderState,
    ProjectorState,
    MergeState,
    Tokens,
    Encoder,
    Project,
    Merge,
    const N: usize,
    Env: Copy,
> Module<(K, &mut [(I, S); N]), Env, &'s mut (TokenState, EncoderState, ProjectorState, MergeState)>
    for InputEmbedding<Tokens, Encoder, Project, Merge>
where
    Tokens: Module<K, Env, &'s mut TokenState, Output = X>,
    Encoder: for<'i, 't> Module<&'i mut I, (), &'t mut EncoderState, Output = F>,
    Project: for<'x, 't> Module<&'x F, Env, &'t mut ProjectorState, Output = Y>,
    Merge: for<'x, 'y, 'b, 't> Module<(&'x X, &'y Y, &'b S), Env, &'t mut MergeState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (tokens_state, encoder, projector, merge): &'s mut (
            TokenState,
            EncoderState,
            ProjectorState,
            MergeState,
        ),
        (tokens, images): (K, &mut [(I, S); N]),
    ) -> X {
        let mut hidden = self.tokens.forward(&mut *device, env, tokens_state, tokens);
        if N > 0 {
            #[expect(
                clippy::needless_range_loop,
                reason = "device unrolling requires indexed loops"
            )]
            #[unroll]
            for index in 0..N {
                let (image, selection) = &mut images[index];
                let features = self.encoder.forward(&mut *device, (), &mut *encoder, image);
                let features =
                    self.projector
                        .forward(&mut *device, env, &mut *projector, &features);
                hidden = self.merge.forward(
                    &mut *device,
                    env,
                    &mut *merge,
                    (&hidden, &features, &*selection),
                );
            }
        }
        hidden
    }
}

/// Encoded features skip the encoder and the projector; their slots are unused.
impl<
    's,
    K,
    Y,
    S,
    X,
    TokenState,
    EncoderState,
    ProjectorState,
    MergeState,
    Tokens,
    Encoder,
    Project,
    Merge,
    const N: usize,
    Env: Copy,
> Module<(K, &[(Y, S); N]), Env, &'s mut (TokenState, EncoderState, ProjectorState, MergeState)>
    for InputEmbedding<Tokens, Encoder, Project, Merge>
where
    Tokens: Module<K, Env, &'s mut TokenState, Output = X>,
    Merge: for<'x, 'y, 'b, 't> Module<(&'x X, &'y Y, &'b S), Env, &'t mut MergeState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (tokens_state, _, _, merge): &'s mut (TokenState, EncoderState, ProjectorState, MergeState),
        (tokens, features): (K, &[(Y, S); N]),
    ) -> X {
        let mut hidden = self.tokens.forward(&mut *device, env, tokens_state, tokens);
        if N > 0 {
            #[expect(
                clippy::needless_range_loop,
                reason = "device unrolling requires indexed loops"
            )]
            #[unroll]
            for index in 0..N {
                let (features, selection) = &features[index];
                hidden = self.merge.forward(
                    &mut *device,
                    env,
                    &mut *merge,
                    (&hidden, features, selection),
                );
            }
        }
        hidden
    }
}

/// Merges independently embedded tokens and media using caller-supplied spans.
#[derive(DeviceSend)]
pub struct TokenEmbedding<Tokens, Media, Merge> {
    pub tokens: Tokens,
    pub media: Media,
    pub merge: Merge,
}

/// The state holds the token embedding's slot, then the media encoder's.
impl<'s, T, I, S, X, Y, O, TokenState, MediaState, Tokens, Media, Merge, Env: Copy>
    Module<(T, I, S), Env, &'s mut (TokenState, MediaState)>
    for TokenEmbedding<Tokens, Media, Merge>
where
    Tokens: Module<T, Env, &'s mut TokenState, Output = X>,
    Media: Module<I, Env, &'s mut MediaState, Output = Y>,
    Merge: for<'x, 'y, 'd> Module<(&'x X, &'y Y, S), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (token_state, media_state): &'s mut (TokenState, MediaState),
        (tokens, media, spans): (T, I, S),
    ) -> O {
        let tokens = self.tokens.forward(&mut *device, env, token_state, tokens);
        let media = self.media.forward(&mut *device, env, media_state, media);
        self.merge
            .forward(device, env, &mut (), (&tokens, &media, spans))
    }
}

/// Replaces the input with the item a selection picks from another input.
#[derive(DeviceSend)]
pub struct Replace<Selection>(pub Selection);

/// The state is the selection's slot.
impl<X, Y, S, T, Selection, Env: Copy> Module<(&X, &Y, &S), Env, T> for Replace<Selection>
where
    Selection: for<'y, 's> Module<(&'y Y, &'s S), Env, T, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: T,
        (_, source, selection): (&X, &Y, &S),
    ) -> X {
        self.0.forward(device, env, state, (source, selection))
    }
}
