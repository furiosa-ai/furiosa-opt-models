//! Tiled image encoding with intermediate features projected into language width.
use super::*;
use crate::models::Scalar;
use crate::modules::vision;
use crate::modules::vision::Class;
use crate::modules::vision::Patch;
use crate::modules::vision::{Captured, GatedPosition, Mask, Merge, Pad, Positions, Rows};
use crate::modules::{attention, feedforward, residual::Scaled, transformer};
use crate::ops::{
    index::sequence::Sum,
    linear::Affine,
    normalization::LayerNorm,
    pointwise::{
        activation::{Gelu, Tanh},
        add::Add,
        gated,
    },
};
use crate::tensor::axis::Single;
use crate::tensor::image::Grid;
use crate::{First, Identity, Module, Then};
use std::marker::PhantomData;

/// Local and global image features merged with captures and projected into text width.
#[derive(DeviceSend)]
pub struct Model {
    /// Local image transformer and intermediate feature capture.
    pub local: Local,
    /// Tile positions added between local and global encoding.
    pub position: Positions<bf16, AspectRatios, Tiles, VisionWidth>,
    /// Gated global transformer layers.
    pub global: [Global; 8],
    /// Removes padding and concatenates captured feature channels.
    pub merge: Merge<Tiles, Tokens, Padded>,
    /// Projects merged features into language hidden width.
    pub projector: Linear<Captured<VisionWidth, 5>, Hidden>,
    /// Additive attention mask derived from tile validity.
    pub mask: Mask<Tiles, Tokens, Padded>,
}

impl<Env: Copy> Module<&mut Input, Env, &mut ()> for Model {
    type Output = Output;

    fn forward(&self, device: &mut Device, env: Env, _: &mut (), input: &mut Input) -> Output {
        let mask: Bias = self.mask.forward(&mut *device, env, &mut (), &input.valid);
        let attention = attention::Context {
            positions: &(),
            write: (),
            read: &mask,
        };
        let hidden = self.local.forward(
            &mut *device,
            &attention,
            &mut Default::default(),
            (
                (&input.pixels, &input.aspect),
                (&CAPTURES, &mut input.captures),
            ),
        );
        let hidden: vision::Features<bf16, Tiles, Padded, VisionWidth> =
            unsafe { hidden.reshape() };
        let hidden = self
            .position
            .forward(&mut *device, env, &mut (), (&hidden, &input.aspect));
        let hidden: Features = unsafe { hidden.reshape() };
        let hidden = self
            .global
            .forward(&mut *device, &attention, &mut Default::default(), hidden);
        let hidden = self
            .merge
            .forward(&mut *device, env, &mut (), (&hidden, &input.captures));
        self.projector.forward(device, env, &mut (), &hidden)
    }
}

type Local = transformer::Encoder<Embedding, [Layer; 32], Then<Identity, Norm>>;

/// Patch, tile, class, and spatial embedding with normalized padded rows.
#[derive(DeviceSend)]
pub struct Embedding {
    /// Per-tile patch projection.
    pub patch: Patch<Channels, VisionWidth, PatchHeight, PatchWidth, bf16, ()>,
    /// Aspect-dependent tile positions before local encoding.
    pub tiles: Positions<bf16, AspectRatios, Tiles, VisionWidth>,
    /// Class token inserted before each tile's patch rows.
    pub class: Class<bf16, VisionWidth>,
    /// Gated spatial and tile position embeddings.
    pub positions: GatedPosition<bf16, AspectRatios, Tiles, Tokens, VisionWidth>,
    /// Per-row normalization before the local transformer.
    pub norm: Norm,
    /// Aligns each tile's token count for subsequent attention.
    pub pad: Pad<Padded>,
}

impl<Env: Copy> Module<(&Pixels, &Aspect), Env, &mut ()> for Embedding {
    type Output = Features;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (pixels, aspect): (&Pixels, &Aspect),
    ) -> Features {
        let hidden = self.patch.forward(&mut *device, env, &mut (), pixels);
        let hidden = self
            .tiles
            .forward(&mut *device, env, &mut (), (&hidden, aspect));
        let hidden = self.class.forward(&mut *device, env, &mut (), &hidden);
        let hidden = self
            .positions
            .forward(&mut *device, env, &mut (), (&hidden, aspect));
        let hidden: HbmTensor<bf16, m![1], m![Source, VisionWidth]> = unsafe { hidden.reshape() };
        let hidden = self.norm.forward(&mut *device, env, &mut (), &hidden);
        let hidden: vision::Features<bf16, Tiles, Tokens, VisionWidth> =
            unsafe { hidden.reshape() };
        let hidden = self.pad.forward(&mut *device, env, &mut (), &hidden);
        unsafe { hidden.reshape() }
    }
}

pub type Norm = LayerNorm<HbmTensor<bf16, m![1], m![VisionWidth]>, Scalar>;

pub type Layer = transformer::PreNorm<Attention, Mlp, Norm, Add>;

type Attention = attention::Full<
    attention::Positioned<attention::Projection<Linear<VisionWidth, VisionWidth, 0>>, First>,
    crate::ops::attention::Attention<PhantomData<VisionHeads>, Scalar>,
    Linear<VisionWidth, VisionWidth, 0>,
>;

type Linear<I, O, const B: usize = 1> =
    Affine<HbmTensor<bf16, m![1], m![O, I]>, [HbmTensor<bf16, m![1], m![O]>; B]>;

type Mlp = feedforward::Mlp<
    Linear<VisionWidth, VisionIntermediate>,
    Gelu,
    Linear<VisionIntermediate, VisionWidth>,
>;

type Global = transformer::PreNorm<Attention, Mlp, Norm, SumGate>;

type SumGate = Scaled<Scalar, Tanh, gated::Add>;

/// Tiled RGB inputs in tile, channel, row, column order.
pub type Pixels = HbmTensor<bf16, m![1], m![Tiles, Channels, Height, Width]>;

/// Aspect-ratio embedding index for the image's tile arrangement.
pub type Aspect = HbmTensor<i32, m![1], m![1]>;

/// Per-tile validity used to mask padded image rows.
pub type Valid = HbmTensor<i32, m![1], m![Tiles]>;

/// Spatial patch count of one tile, excluding its class token.
pub type Patches = Grid<Height, Width, PatchHeight, PatchWidth>;

/// Class token followed by spatial patches of one tile.
pub type Tokens = Sum<Single, Patches>;

/// Flattened tile/token rows with no alignment padding.
pub type Source = Rows<Tiles, Tokens>;

/// Flattened tile/token rows including per-tile alignment padding.
pub type Count = Rows<Tiles, Padded>;

/// Padded local/global image features in flattened tile order.
pub type Features = HbmTensor<bf16, m![1], m![Count, VisionWidth]>;

/// Unpadded image rows projected into language hidden width.
pub type Output = HbmTensor<bf16, m![1], m![Source, Hidden]>;

type Bias = HbmTensor<f32, m![1], m![Count, Count]>;

/// Image tiles, aspect selection, validity, and writable local feature captures.
#[derive(DeviceSend)]
pub struct Input {
    /// Tiled RGB samples.
    pub pixels: Pixels,
    /// Aspect-ratio embedding index.
    pub aspect: Aspect,
    /// Tile validity flags for the attention mask.
    pub valid: Valid,
    /// Writable feature rows for the local layers named by CAPTURES.
    pub captures: [Features; 5],
}

/// Local layer indices whose intermediate features join the final image features.
pub const CAPTURES: [usize; 5] = [3, 7, 15, 23, 30];
