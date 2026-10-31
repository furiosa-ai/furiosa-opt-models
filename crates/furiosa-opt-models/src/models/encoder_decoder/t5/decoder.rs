//! Causal decoding appends self history and either creates or reads immutable encoder memory.

use super::{Embedding, Ids, Linear, Mask, Norm, Positions, Rows, Scalar};
use crate::modules::attention::relative;
use crate::modules::{attention, language, transformer};
use crate::ops;
use crate::ops::{
    index::{gather::Gather, scatter::Scatter},
    pointwise::{add::Add, multiply::Scale},
};
use crate::{First, Fork, Module, Sequential};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Relative-position causal decoder with final normalization and vocabulary scores.
pub type Model<Vocabulary, Width, KvWidth, Heads, Buckets, F, const LAYERS: usize> =
    language::Readout<
        Body<
            Decoder<Vocabulary, Width, KvWidth, Heads, F, LAYERS>,
            Norm<Width>,
            relative::Bias<relative::Table<Buckets, Heads>, false>,
        >,
        Head<Vocabulary, Width>,
    >;

/// Decoder body sharing one relative-position bias across its layers.
#[derive(DeviceSend)]
pub struct Body<Function, Norm, Bias> {
    /// Token embedding and self/cross-attention decoder layers.
    pub function: Function,
    /// Final RMS normalization.
    pub norm: Norm,
    /// Causal relative-position table and bucket distance.
    pub bias: Bias,
}

/// Self attention takes the bias the step's positions and mask give, cross attention the source
/// mask beside the call's memory access; the state holds the function's slot, then the norm's.
impl<
    's,
    'e,
    't,
    Function,
    Norm,
    Bias,
    A: Copy,
    FunctionState,
    NormState,
    W: AxisName,
    Q: AxisName,
    H: AxisName,
    S: AxisName,
    Heads: AxisName,
>
    Module<
        &'t Ids<Q>,
        ((&'e Positions<Q, H>, &'e Mask<Q, H>, &'e Mask<Q, S>), A),
        &'s mut (FunctionState, NormState),
    > for Body<Function, Norm, Bias>
where
    Bias: for<'d> Module<
            (&'e Positions<Q, H>, &'e Mask<Q, H>),
            (),
            &'d mut (),
            Output = super::Bias<Heads, Q, H>,
        >,
    Function: for<'b> Module<
            &'t Ids<Q>,
            ((&'b (), &'b super::Bias<Heads, Q, H>), (&'e Mask<Q, S>, A)),
            &'s mut FunctionState,
            Output = Rows<W, Q>,
        >,
    Norm: for<'x> Module<&'x Rows<W, Q>, (), &'s mut NormState, Output = Rows<W, Q>>,
{
    type Output = Rows<W, Q>;

    fn forward(
        &self,
        device: &mut Device,
        ((positions, mask, cross_mask), access): (
            (&'e Positions<Q, H>, &'e Mask<Q, H>, &'e Mask<Q, S>),
            A,
        ),
        (function, norm): &'s mut (FunctionState, NormState),
        tokens: &'t Ids<Q>,
    ) -> Self::Output {
        let bias = self
            .bias
            .forward(&mut *device, (), &mut (), (positions, mask));
        let hidden = self.function.forward(
            &mut *device,
            ((&(), &bias), (cross_mask, access)),
            function,
            tokens,
        );
        self.norm.forward(device, (), norm, &hidden)
    }
}

type Decoder<Vocabulary, Width, KvWidth, Heads, F, const LAYERS: usize> =
    language::Backbone<Embedding<Vocabulary, Width>, [Layer<Width, KvWidth, Heads, F>; LAYERS]>;

/// Causal self attention, encoded-source attention, and feed-forward residuals.
pub type Layer<Width, KvWidth, Heads, F> = transformer::CrossLayer<
    Attention<Width, KvWidth, Heads>,
    CrossAttention<Width, KvWidth, Heads>,
    F,
    Norm<Width>,
    Add,
>;

type Attention<Width, KvWidth, Heads> = attention::Cached<
    attention::Projection<Linear<Width, KvWidth>>,
    First,
    Scatter,
    Gather,
    ops::attention::Attention<PhantomData<Heads>, Scalar>,
    Linear<KvWidth, Width>,
>;

type CrossAttention<Width, KvWidth, Heads> = attention::Cross<
    Linear<Width, KvWidth>,
    Fork<(Linear<Width, KvWidth>, Linear<Width, KvWidth>)>,
    Scatter,
    Gather,
    ops::attention::Attention<PhantomData<Heads>, Scalar>,
    Linear<KvWidth, Width>,
>;

/// Optional tied-embedding scaling followed by the vocabulary projection.
pub type Head<Vocabulary, Width> = Sequential<Scale, Linear<Width, Vocabulary>>;
