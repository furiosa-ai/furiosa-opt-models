//! Self-attention decoding interleaved with gated reads of cached image features.
use super::encoder;
use super::*;
use crate::Module;
use crate::models::vision_language::mllama::CrossLanguage;
use crate::models::vision_language::mllama::GatedCross;
pub use crate::models::{Ids, Scalar};
use crate::modules::attention;
use crate::modules::{feedforward::glu::Glu, language, transformer};
use crate::ops::attention::Attention;
use crate::ops::index::gather::RowStorage;
use crate::ops::pointwise::cast::Convert;
use crate::ops::{
    attention::full,
    embedding::Lookup,
    index::{
        scatter::Scatter,
        table::{KeyValue, Read, Selection},
    },
    linear::Affine,
    normalization::Rms,
    pointwise::{
        activation::Silu,
        add::Add,
        gated,
        multiply::{Multiply, RowScale},
    },
    rotary,
};
use crate::{Branch, Fork, Interleave, Then};
use std::marker::PhantomData;

/// Catalog composition specialized to the checkpoint's text and vision sizes.
pub type Model = CrossLanguage<
    vision::Model,
    Embedding,
    Layer,
    CrossAttention,
    Mlp,
    Norm,
    Scalar,
    RowScale,
    gated::Add,
    Logits,
    32,
    8,
    { &ORDER },
>;

/// Normalized text body followed by its vocabulary head.
pub type Decoder = language::Readout<Body, Logits>;

/// Token decoder with interleaved gated image attention and final RMS normalization.
pub type Body = language::Normalized<
    language::Backbone<Embedding, Interleave<([Layer; 32], [CrossLayer; 8]), { &ORDER }>>,
    Norm,
>;

/// Self-attention layers read the step's positions, cross layers the rows their gates keep; a
/// layer's slot holds its cache, writes and history read, or for a cross layer its memory read.
impl<Q: AxisName> Module<&mut Input<Q>, (), &mut ()> for Body {
    type Output = Rows<Hidden, Q>;

    fn forward(
        &self,
        device: &mut Device,
        _: (),
        _: &mut (),
        input: &mut Input<Q>,
    ) -> Self::Output {
        let Input {
            tokens,
            step,
            states,
            memory,
        } = input;
        // Indexed borrows, since `each_ref` builds raw pointers that device code cannot lower.
        let cross = [0, 1, 2, 3, 4, 5, 6, 7].map(|index| {
            let (offsets, count) = &memory.read[index];
            let read = Read {
                count,
                offsets,
                bias: &memory.bias,
            };
            (
                (
                    Default::default(),
                    (read, &memory.caches[index]),
                    Default::default(),
                ),
                Default::default(),
            )
        });
        let mut histories = states
            .each_mut()
            .map(|(cache, write, offsets)| (cache, &*write, step.read(offsets)));
        let text = histories.each_mut().map(|(cache, write, read)| {
            let cache = (&mut **cache, *write, &*read);
            let attention = (Default::default(), cache, Default::default());
            (attention, Default::default())
        });
        self.forward(
            device,
            ((&step.positions, &()), &memory.rows),
            &mut (((), (text, cross)), ()),
            &*tokens,
        )
    }
}

/// Without image memory only the text layers run, so no cross layer's slot is lent.
impl<Q: AxisName> Module<&mut Input<Q, ()>, (), &mut ()> for Body {
    type Output = Rows<Hidden, Q>;

    fn forward(
        &self,
        device: &mut Device,
        env: (),
        _: &mut (),
        input: &mut Input<Q, ()>,
    ) -> Self::Output {
        let Input {
            tokens,
            step,
            states,
            ..
        } = input;
        let mut histories = states
            .each_mut()
            .map(|(cache, write, offsets)| (cache, &*write, step.read(offsets)));
        let text = histories.each_mut().map(|(cache, write, read)| {
            let cache = (&mut **cache, *write, &*read);
            let attention = (Default::default(), cache, Default::default());
            (attention, Default::default())
        });
        let hidden = self.first().text().forward(
            &mut *device,
            (&step.positions, &()),
            &mut ((), text),
            &*tokens,
        );
        self.second().forward(device, env, &mut (), &hidden)
    }
}

impl<Q: AxisName> Module<&mut Input<Q>, (), &mut ()> for encoder::Model {
    type Output = Rows<Hidden, Q>;

    fn forward(
        &self,
        device: &mut Device,
        env: (),
        _: &mut (),
        input: &mut Input<Q>,
    ) -> Self::Output {
        Module::forward(&self.decoder, device, env, &mut (), input)
    }
}

type Embedding = Lookup<HbmTensor<bf16, m![1], m![Embeddings, Hidden]>>;

pub type Layer = transformer::PreNorm<SelfAttention, Mlp, Norm, Add>;

type SelfAttention = attention::Cached<
    Projection,
    rotary::SplitHalf<Head>,
    Scatter,
    (),
    Attend,
    Linear<Hidden, Hidden>,
>;

type Projection =
    attention::Projection<Linear<Hidden, Hidden>, Linear<Hidden, KvWidth>, Linear<Hidden, KvWidth>>;

type Linear<I, O> = Affine<HbmTensor<bf16, m![1], m![O, I]>, [HbmTensor<bf16, m![1], m![O]>; 0]>;

pub type Attend = Attention<PhantomData<Heads>, Scalar>;

pub type Mlp = Glu<
    Linear<Hidden, Intermediate>,
    Silu,
    Linear<Hidden, Intermediate>,
    Multiply,
    Linear<Intermediate, Hidden>,
>;

pub type Norm = Rms<HbmTensor<bf16, m![1], m![Hidden]>, Scalar>;

pub type CrossLayer = GatedCross<CrossAttention, Mlp, Norm, Scalar, RowScale, gated::Add>;

/// Normalized text queries attending to cached projected image keys and values.
pub type CrossAttention = attention::Cross<
    Query,
    Fork<(Key, Linear<Hidden, KvWidth>)>,
    Scatter,
    (),
    Attend,
    Linear<Hidden, Hidden>,
>;

// A decoded row and the image rows come in no whole norm tile.
type Query = Then<Linear<Hidden, Hidden>, full::Untiled<Head>>;
type Key = Then<Linear<Hidden, KvWidth>, full::Untiled<Head>>;

/// Bias-free vocabulary projection returned as F32 scores.
pub type Logits = Then<Linear<Hidden, Vocabulary>, Convert<f32>>;

/// Query-major BF16 feature rows.
pub type Rows<W, Q> = HbmTensor<bf16, m![1], m![Q, W]>;

/// Opaque key/value row storage shared between image encoding and text decoding.
pub type Storage = KeyValue<RowStorage<KvWidth, bf16>>;

/// Key/value byte offsets for each query row.
pub type Offsets<Q> = KeyValue<Ids<Q>>;

/// Per-row key/value offsets and number of rows to write.
pub type Write<Q> = (Offsets<Q>, HbmTensor<i32, m![1], m![1]>);

/// Writable cache with its write destinations and the history rows it reads.
pub type State<Q> = (Storage, Write<Q>, Selection);

/// Rotary positions, attended length and query-major bias every self-attention layer reads.
#[derive(DeviceSend)]
pub struct Step<Q: AxisName> {
    /// Rotary position of each query row.
    pub positions: Ids<Q>,
    /// Number of history rows each layer reads.
    pub length: HbmTensor<i32, m![1], m![1]>,
    /// Query-major additive mask over the history rows.
    pub bias: Slice<Scalar>,
}

impl<Q: AxisName> Step<Q> {
    /// A layer's read of its history rows at `offsets`.
    fn read<'a>(&'a self, offsets: &'a Selection) -> Read<'a> {
        Read {
            count: &self.length,
            offsets,
            bias: &self.bias,
        }
    }
}

/// Query-major vocabulary scores, excluding the eight image-only embedding entries.
pub type Scores<Q> = HbmTensor<f32, m![1], m![Q, Vocabulary]>;

/// Image cross-attention caches, row selection, and per-query residual mask.
#[derive(DeviceSend)]
pub struct Memory<Q: AxisName> {
    /// One key/value store per cross-attention layer.
    pub caches: [Storage; 8],
    /// Per-layer image-memory row selection and the number of rows read.
    pub read: [(Selection, HbmTensor<i32, m![1], m![1]>); 8],
    /// Query-major bias over the image-memory rows every cross-attention layer reads.
    pub bias: Slice<Scalar>,
    /// Per-query multiplier masking gated image-attention residuals.
    pub rows: HbmTensor<bf16, m![1], m![Q]>,
}

/// Token rows, writable self-attention state, and optional cached image memory.
#[derive(DeviceSend)]
pub struct Input<Q: AxisName, M = Memory<Q>> {
    /// Token indices, including image-only entries in the embedding table.
    pub tokens: Ids<Q>,
    /// What every self-attention layer reads.
    pub step: Step<Q>,
    /// Writable state of each self-attention layer.
    pub states: [State<Q>; 32],
    /// Image caches and read mask, or unit for text-only execution.
    pub memory: M,
}

/// Decoder positions occupied by cross-attention rather than self-attention layers.
pub const CROSS: [usize; 8] = [3, 8, 13, 18, 23, 28, 33, 38];

/// Layer execution order, with image-memory layers marked as the second branch.
pub const ORDER: [Branch; 40] = {
    let mut order = [Branch::First; 40];
    let mut i = 0;
    while i < CROSS.len() {
        order[CROSS[i]] = Branch::Second;
        i += 1;
    }
    order
};
