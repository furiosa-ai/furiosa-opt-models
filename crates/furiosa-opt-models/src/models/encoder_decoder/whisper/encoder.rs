//! Spectrogram encoding uses two temporal convolutions and unmasked self attention.

use super::{Ids, Rows};
use crate::Module;
use crate::ops::convolution::Half;
use furiosa_opt_std::prelude::*;

/// Spectrogram embedding, unmasked transformer layers, and final normalization.
#[derive(DeviceSend)]
pub struct Encoder<Embedding, Layers, Norm> {
    /// Convolutional frontend and absolute-position embedding.
    pub embedding: Embedding,
    /// Unmasked self-attention and feed-forward layers.
    pub layers: Layers,
    /// Final per-frame layer normalization.
    pub norm: Norm,
}

/// Unmasked layers read an empty env, whatever the caller's; the state holds the embedding's slot,
/// one slot a layer and the norm's.
impl<
    's,
    Embedding,
    Layer,
    Norm,
    EmbeddingState,
    LayerState,
    NormState,
    Mel: AxisName,
    Frames: AxisName,
    Width: AxisName,
    const DEPTH: usize,
    Env,
> Module<&mut Input<Mel, Frames>, Env, &'s mut (EmbeddingState, [LayerState; DEPTH], NormState)>
    for Encoder<Embedding, [Layer; DEPTH], Norm>
where
    Embedding: for<'x> Module<
            (&'x Features<Mel, Frames>, &'x Ids<Half<Frames>>),
            (),
            &'s mut EmbeddingState,
            Output = Rows<Width, Half<Frames>>,
        >,
    Layer: Module<
            Rows<Width, Half<Frames>>,
            (),
            &'s mut LayerState,
            Output = Rows<Width, Half<Frames>>,
        >,
    Norm: for<'x> Module<
            &'x Rows<Width, Half<Frames>>,
            (),
            &'s mut NormState,
            Output = Rows<Width, Half<Frames>>,
        >,
{
    type Output = Rows<Width, Half<Frames>>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        (embedding, layers, norm): &'s mut (EmbeddingState, [LayerState; DEPTH], NormState),
        input: &mut Input<Mel, Frames>,
    ) -> Self::Output {
        let hidden = self.embedding.forward(
            &mut *device,
            (),
            embedding,
            (&input.features, &input.positions),
        );
        let hidden = self.layers.forward(&mut *device, (), layers, hidden);
        self.norm.forward(device, (), norm, &hidden)
    }
}

/// Preprocessed mel features in channel, frame order.
pub type Features<Mel, Frames> = HbmTensor<f32, m![1], m![Mel, Frames]>;

/// Mel features and absolute positions for the temporally downsampled rows.
#[derive(DeviceSend)]
pub struct Input<Mel: AxisName, Frames: AxisName> {
    /// Caller-preprocessed log-mel spectrogram.
    pub features: Features<Mel, Frames>,
    /// Position indices after the stride-two convolution.
    pub positions: Ids<Half<Frames>>,
}
