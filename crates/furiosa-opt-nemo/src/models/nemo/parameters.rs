//! The weights one serving step reads, each in the layout its piece reads, packed on the host: TP2
//! head pairs of the query, K and V pairs, output halves, SwiGLU panels and a permuted readout.

use super::serving::{
    Chips, Hidden, head::Vocabulary, mlp, output, qkv, scale, stack::Depth, step,
};
use crate::ops::linear::output::packed::Layout;
use furiosa_opt_std::prelude::*;
mod binding;

pub struct Output;
#[derive(Clone, Debug)]
pub struct Inputs;

impl AxisName for Inputs {
    const NAME: Ident = super::serving::Width::NAME;
    const SIZE: usize = 2 * super::serving::Width::SIZE;
}

impl Layout for Output {
    type Input = Inputs;
    type Output = super::serving::Hidden;
    type Half = m![
        { Symbol<Self::Output> } % 1280 / 640,
        { Symbol<Self::Output> } % 640 / 320,
        { Symbol<Self::Output> } % 320 / 160,
        { Symbol<Self::Input> } % 2048 / 64,
        { Symbol<Self::Output> } % 2560 / 1280,
        { Symbol<Self::Output> } % 160 / 80,
        { Symbol<Self::Output> } % 80 / 8,
        { Symbol<Self::Input> } % 64 / 8,
        { Symbol<Self::Output> } % 8,
        { Symbol<Self::Input> } % 8
    ];
}

#[derive(Clone, Debug)]
pub struct Query;
axes![Heads = 32, Head = 128];

impl crate::ops::linear::query::packed::Layout for Query {
    type Input = super::serving::Hidden;
    type Heads = Heads;
    type Head = Head;
}

#[derive(Clone, Debug)]
pub struct KeyValue;
axes![PairOutputs = 1024];

impl crate::ops::linear::pair::Layout for KeyValue {
    type Input = super::serving::Hidden;
    type Output = PairOutputs;
}

pub type GateUp = crate::ops::linear::mlp::gate::Layout<
    super::serving::Hidden,
    super::serving::mlp::Intermediate,
>;

pub type Down = crate::ops::linear::mlp::down::Layout<
    super::serving::Hidden,
    super::serving::mlp::Intermediate,
>;

#[derive(Clone, Debug)]
pub struct Readout;

impl crate::ops::vocabulary::readout::Layout for Readout {
    type Hidden = super::serving::Hidden;
    type Vocabulary = super::serving::step::Vocabulary;

    fn token(rank: usize, entry: usize) -> usize {
        1024 * (entry % 1024 / 8) + 512 * rank + 8 * (entry / 1024) + entry % 8
    }
}

pub type Greedy = crate::ops::vocabulary::Argmax<
    crate::ops::vocabulary::readout::Local<Readout>,
    super::serving::Chips,
>;
pub type Sampling = crate::ops::vocabulary::readout::Sampling<Readout, super::serving::Chips>;

pub type Embedding = crate::ops::embedding::Embedding<
    super::serving::step::Vocabulary,
    super::serving::Hidden,
    super::serving::Chips,
    HbmTensor<
        bf16,
        super::serving::Chips,
        m![{ Symbol<super::serving::step::Vocabulary> }, { Symbol<super::serving::Hidden> }],
    >,
>;

/// A weight of every layer on each chip, layer after layer, each in the layout `S` its piece reads.
pub type Stacked<D, S> = HbmTensor<D, Chips, m![Depth, { S }]>;

/// Every layer's weights and scales, the storage `stack::Layers` borrows.
#[derive(DeviceSend)]
pub struct Layers {
    pub norm: Stacked<bf16, m![Hidden]>,
    pub query: Stacked<f8e4m3, qkv::query::Stored>,
    pub pairs: Stacked<f8e4m3, qkv::contraction::Stored>,
    /// Factor 0 holds the QKV input scale and factor 1 the Q weight scale.
    pub qkv_scales: Stacked<f32, scale::Packed>,
    pub pair_scale: Stacked<f32, qkv::PairScale>,
    pub output: Stacked<f8e4m3, output::Stored>,
    /// Factor 0 holds the O input scale and factor 1 its weight scale.
    pub output_scales: Stacked<f32, scale::Packed>,
    pub post_norm: Stacked<bf16, m![Hidden]>,
    pub gate: Stacked<f8e4m3, mlp::weight::Stored>,
    pub up: Stacked<f8e4m3, mlp::weight::Stored>,
    pub down: Stacked<f8e4m3, mlp::down::Stored>,
    /// Factors 0-5: gate input and weight, up input and weight, down input and weight scales.
    pub mlp_scales: Stacked<f32, scale::Packed>,
}

/// The weights a step reads: the embedding, every layer's, the final norm, the readout's head and
/// the tables its greedy and sampled draws read.
#[derive(DeviceSend)]
pub struct Model {
    pub embedding: Embedding,
    pub layers: Layers,
    pub norm: step::Norm,
    pub head: HbmTensor<bf16, Chips, m![Hidden / 256, Vocabulary, Hidden % 256]>,
    pub greedy: Greedy,
    pub sampler: Sampling,
}
