//! Mamba2, by published size: a size's [`Geometry`] is what its checkpoints declare, and its
//! [`Config`] refuses a value the size does not hold or a feature Mamba2 does not serve.
//!
//! - Every size's mixer is twice as wide as its rows, in heads of 64 features scanned over one
//!   group's 128-feature state, after a 4-tap causal convolution; no time step is clamped.
//! - The published size's head reads its embedding's table, and its residual rows stay in FP32.
//! - [`Model`] is the catalog's composition, over the [`Mamba2ForCausalLM`] template.

use super::mamba::{Body, Head};
use crate::Then;
use crate::models::decoder::recurrent::Geometry;
use crate::models::{Family, Scalar, vocabulary};
use crate::modules::recurrent::{Channels, Convolution, History, Projected, Projection};
use crate::modules::recurrent::{GateFirst, RmsNormGated};
use crate::modules::{language, recurrent, residual};
use crate::ops::attention::geometry;
use crate::ops::normalization::RmsNorm;
use crate::ops::{
    embedding::Lookup,
    linear::Affine,
    normalization::Rms,
    pointwise::{add::Add, cast::Convert as Cast, clamp::Clamp, softplus::Softplus},
    recurrent::SelectiveScan,
};
use crate::tensor::form::Weight;
use crate::tensor::memory::Memory;
use crate::tensor::scalar::F32;
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::io;

pub type Model<
    Vocabulary,
    Width,
    Intermediate,
    Kernel,
    Recurrence,
    Groups,
    Heads,
    const LAYERS: usize,
    const BIAS: usize = 0,
    const CONVOLUTION_BIAS: usize = 1,
    Order = GateFirst,
    const LOWER: usize = { TIME_STEP_LIMIT[0].to_bits() as usize },
    const UPPER: usize = { TIME_STEP_LIMIT[1].to_bits() as usize },
> = language::Readout<
    Decoder<
        Vocabulary,
        Width,
        Intermediate,
        Kernel,
        Recurrence,
        Groups,
        Heads,
        LAYERS,
        BIAS,
        CONVOLUTION_BIAS,
        Order,
        LOWER,
        UPPER,
    >,
    Head<Width, Vocabulary>,
>;

/// HF's default `time_step_limit`, which clamps no time step.
pub const TIME_STEP_LIMIT: [f32; 2] = [0.0, f32::INFINITY];

pub type Decoder<
    Vocabulary,
    Width,
    Intermediate,
    Kernel,
    Recurrence,
    Groups,
    Heads,
    const LAYERS: usize,
    const BIAS: usize = 0,
    const CONVOLUTION_BIAS: usize = 1,
    Order = GateFirst,
    const LOWER: usize = { TIME_STEP_LIMIT[0].to_bits() as usize },
    const UPPER: usize = { TIME_STEP_LIMIT[1].to_bits() as usize },
> = Body<
    Then<Lookup<HbmTensor<bf16, m![1], m![Vocabulary, Width]>>, Cast<f32>>,
    [Layer<
        Width,
        Intermediate,
        Kernel,
        Recurrence,
        Groups,
        Heads,
        BIAS,
        CONVOLUTION_BIAS,
        Order,
        LOWER,
        UPPER,
    >; LAYERS],
    Norm<Width>,
>;

/// The cast and norm of the residual rows lead the branch, as a Mamba layer's do.
type Layer<
    Width,
    Intermediate,
    Kernel,
    State,
    Groups,
    Heads,
    const BIAS: usize,
    const CONVOLUTION_BIAS: usize,
    Order,
    const LOWER: usize,
    const UPPER: usize,
> = residual::Residual<
    (),
    Then<
        Then<Cast<bf16>, Norm<Width>>,
        Then<
            Mixer<
                Width,
                Intermediate,
                Kernel,
                State,
                Groups,
                Heads,
                BIAS,
                CONVOLUTION_BIAS,
                Order,
                LOWER,
                UPPER,
            >,
            Cast<f32>,
        >,
    >,
    Add,
>;

type Norm<Width> = Rms<HbmTensor<bf16, m![1], m![Width]>, Scalar>;

/// The time steps are `clamp(softplus(dt + dt_bias), LOWER, UPPER)` of the `dt` rows of `in_proj`,
/// the bounds [`F32`] bits.
pub type Mixer<
    Width,
    Intermediate,
    Kernel,
    State,
    Groups,
    Heads,
    const BIAS: usize,
    const CONVOLUTION_BIAS: usize,
    Order,
    const LOWER: usize,
    const UPPER: usize,
> = recurrent::Mamba2<
    Projection<
        HbmTensor<bf16, m![1], m![{ Symbol<Projected<Intermediate, Groups, State>> }, Width]>,
        [HbmTensor<bf16, m![1], m![{ Symbol<Projected<Intermediate, Groups, State>> }]>; BIAS],
    >,
    Then<Linear<Width, Heads, 1>, Then<Softplus, Clamp<F32<LOWER>, F32<UPPER>>>>,
    Convolution<
        HbmTensor<bf16, m![1], m![{ Symbol<Channels<Intermediate, Groups, State>> }, 1, Kernel]>,
        [HbmTensor<bf16, m![1], m![{ Symbol<Channels<Intermediate, Groups, State>> }]>;
            CONVOLUTION_BIAS],
    >,
    SelectiveScan<HbmTensor<f32, m![1], m![Heads]>, HbmTensor<f32, m![1], m![Heads]>>,
    RmsNormGated<HbmTensor<bf16, m![1], m![Intermediate]>, Scalar, Order>,
    Linear<Intermediate, Width, BIAS>,
>;

pub type Linear<In, Out, const BIAS: usize = 0> =
    Affine<HbmTensor<bf16, m![1], m![Out, In]>, [HbmTensor<bf16, m![1], m![Out]>; BIAS]>;

pub type State<Intermediate, Kernel, Recurrence, Groups, Heads, Head> = recurrent::Ssm<
    HbmTensor<
        f32,
        m![1],
        m![{ Symbol<History<Kernel>> }, { Symbol<Channels<Intermediate, Groups, Recurrence>> }],
    >,
    HbmTensor<f32, m![1], m![Heads, Head, Recurrence]>,
>;

/// A published Mamba2 checkpoint size, by its hub spelling.
pub struct Size<const N: &'static str>;

/// The mixer axes of the published size.
pub mod axis {
    use furiosa_opt_std::prelude::*;

    axes![
        Intermediate = 1536,
        Heads = 24,
        Head = 64,
        Groups = 1,
        State = 128,
        Kernel = 4,
        Vocabulary = 50288
    ];
}

/// `layer_norm_epsilon` 1e-5, as the bits of the FP32 nearest to it.
const EPSILON: usize = 1e-5_f32.to_bits() as usize;

impl Geometry for Size<"130m"> {
    const SIZE: &'static str = "130m";
    const FAMILY: &'static str = "mamba2";
    type Hidden = geometry::Hidden<768>;
    type Vocabulary = axis::Vocabulary;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> =
        RmsNorm<W, EPSILON, m![1], Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>>;

    const LAYERS: usize = 24;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = true;
}

/// What a Mamba2 checkpoint's config declares that its size fixes, and the features Mamba2 does not
/// serve; an absent key reads as Transformers reads it.
#[derive(Deserialize)]
pub struct Config {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_heads: usize,
    vocab_size: usize,
    #[serde(default)]
    expand: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    n_groups: Option<usize>,
    #[serde(default)]
    state_size: Option<usize>,
    #[serde(default)]
    conv_kernel: Option<usize>,
    #[serde(default)]
    layer_norm_epsilon: Option<f64>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    use_bias: bool,
    #[serde(default)]
    use_conv_bias: Option<bool>,
    #[serde(default)]
    residual_in_fp32: Option<bool>,
    #[serde(default)]
    rms_norm: Option<bool>,
    /// Read through each number's text, which also spells an infinite bound.
    #[serde(default)]
    time_step_limit: Option<[serde_json::Number; 2]>,
}

/// A config is of a size when it declares the size's values, at most its compiled vocabulary rows,
/// and none of the features Mamba2 does not serve.
impl<const N: &'static str> TryFrom<Config> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let hidden = <Self as Geometry>::Hidden::SIZE;
        let fixed = [
            ("hidden_size", config.hidden_size, hidden),
            ("num_hidden_layers", config.num_hidden_layers, Self::LAYERS),
            (
                "expand",
                config.expand.unwrap_or(2),
                axis::Intermediate::SIZE / hidden,
            ),
            ("num_heads", config.num_heads, axis::Heads::SIZE),
            ("head_dim", config.head_dim.unwrap_or(64), axis::Head::SIZE),
            ("n_groups", config.n_groups.unwrap_or(8), axis::Groups::SIZE),
            (
                "state_size",
                config.state_size.unwrap_or(128),
                axis::State::SIZE,
            ),
            (
                "conv_kernel",
                config.conv_kernel.unwrap_or(4),
                axis::Kernel::SIZE,
            ),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(invalid(format!(
                "{key}: {declared}, the size {N} holds {held}"
            )));
        }
        vocabulary::<<Self as Geometry>::Vocabulary>(config.vocab_size)?;
        let epsilon = config.layer_norm_epsilon.unwrap_or(1e-5);
        if (epsilon as f32).to_bits() as usize != Self::EPSILON {
            return Err(invalid(format!(
                "layer_norm_epsilon: {epsilon}, another than the size's"
            )));
        }
        if config.tie_word_embeddings != Self::TIED_EMBEDDINGS {
            return Err(invalid(format!(
                "tie_word_embeddings: {}, the size ties: {}",
                config.tie_word_embeddings,
                Self::TIED_EMBEDDINGS
            )));
        }
        let limit = config.time_step_limit.as_ref().map(|bounds| {
            bounds
                .each_ref()
                .map(|bound| bound.to_string().parse::<f32>().ok())
        });
        let features = [
            (
                config
                    .hidden_act
                    .as_deref()
                    .is_some_and(|act| act != "silu"),
                "hidden_act: Mamba2 gates its mixer with SiLU",
            ),
            (config.use_bias, "use_bias: Mamba2 projects without biases"),
            (
                config.use_conv_bias == Some(false),
                "use_conv_bias: Mamba2's convolution adds its bias",
            ),
            (
                config.residual_in_fp32 == Some(false),
                "residual_in_fp32: Mamba2 keeps its residual rows in FP32",
            ),
            (
                config.rms_norm == Some(false),
                "rms_norm: Mamba2 normalizes its gated mixer output",
            ),
            (
                limit.is_some_and(|limit| limit != TIME_STEP_LIMIT.map(Some)),
                "time_step_limit: Mamba2 clamps no time step",
            ),
        ];
        match features.into_iter().find(|(declared, _)| *declared) {
            Some((_, reason)) => Err(unsupported(reason)),
            None => Ok(Size),
        }
    }
}

impl<const N: &'static str> Family for Size<N>
where
    Self: Geometry,
{
    type Hf = Config;
}

/// Source: vLLM `vllm/model_executor/models/mamba2.py`.
pub type Mamba2ForCausalLM<
    Embedding,
    Norm,
    InputProjection,
    Time,
    Convolution,
    Recurrence,
    OutputGate,
    OutputProjection,
    Add,
    Head,
    const DEPTH: usize,
> = language::Decoder<
    Embedding,
    [residual::Residual<
        Norm,
        recurrent::Mamba2<
            InputProjection,
            Time,
            Convolution,
            Recurrence,
            OutputGate,
            OutputProjection,
        >,
        Add,
    >; DEPTH],
    Norm,
    Head,
>;
