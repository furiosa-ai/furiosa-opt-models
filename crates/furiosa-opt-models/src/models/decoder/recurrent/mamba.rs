//! Mamba, by published size: a size's [`Geometry`] is what its checkpoints declare, and its
//! [`Config`] refuses a value the size does not hold or a feature Mamba does not serve.
//!
//! - Every size's mixer is twice as wide as its rows, with a 16-feature state, a 4-tap causal
//!   convolution, and time steps projected through `ceil(hidden_size / 16)` features.
//! - Every size's head reads its embedding's table, and its residual rows stay in FP32.
//! - [`Model`] is the catalog's composition, over the [`MambaForCausalLM`] template.

pub use crate::models::Ids;
use crate::models::decoder::recurrent::Geometry;
use crate::models::{Family, Scalar, vocabulary};
use crate::modules::recurrent::Gate;
use crate::modules::recurrent::{Convolution, Projection};
use crate::modules::{language, recurrent, residual};
use crate::ops::attention::geometry;
use crate::ops::normalization::RmsNorm;
use crate::ops::{
    embedding::Lookup,
    linear::Affine,
    normalization::Rms,
    pointwise::{add::Add, cast::Convert as Cast, softplus::Softplus},
    recurrent::SelectiveScan,
};
use crate::tensor::form::Weight;
use crate::tensor::memory::Memory;
use crate::{Fork, Layers, Then, Unrolled};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use serde_json::Value;
use std::io;

pub type Model<
    Vocabulary,
    Width,
    Intermediate,
    Kernel,
    State,
    Rank,
    const LAYERS: usize,
    const BIAS: usize = 0,
    const CONVOLUTION_BIAS: usize = 1,
> = language::Readout<
    Decoder<Vocabulary, Width, Intermediate, Kernel, State, Rank, LAYERS, BIAS, CONVOLUTION_BIAS>,
    Head<Width, Vocabulary>,
>;

pub type Decoder<
    Vocabulary,
    Width,
    Intermediate,
    Kernel,
    State,
    Rank,
    const LAYERS: usize,
    const BIAS: usize = 0,
    const CONVOLUTION_BIAS: usize = 1,
> = Body<
    Then<Lookup<HbmTensor<bf16, m![1], m![Vocabulary, Width]>>, Cast<f32>>,
    [Layer<Width, Intermediate, Kernel, State, Rank, BIAS, CONVOLUTION_BIAS>; LAYERS],
    Norm<Width>,
>;

/// Embeds the tokens, runs the layers unrolled, each with its own state, and normalizes.
pub type Body<Embedding, Collection, Norm> =
    language::Normalized<language::Backbone<Embedding, Layers<Collection, Unrolled>>, Norm>;

/// The cast and norm of the residual rows are composite, so they lead the branch: a residual's norm
/// place holds one norm.
type Layer<
    Width,
    Intermediate,
    Kernel,
    State,
    Rank,
    const BIAS: usize,
    const CONVOLUTION_BIAS: usize,
> = residual::Residual<
    (),
    Then<
        Then<Cast<bf16>, Norm<Width>>,
        Then<Mixer<Width, Intermediate, Kernel, State, Rank, BIAS, CONVOLUTION_BIAS>, Cast<f32>>,
    >,
    Add,
>;

pub type Norm<Width> = Rms<HbmTensor<bf16, m![1], m![Width]>, Scalar>;

/// The time steps are `softplus(dt_proj(x_proj(x)))`, as HF's slow path computes them.
pub type Mixer<
    Width,
    Intermediate,
    Kernel,
    State,
    Rank,
    const BIAS: usize,
    const CONVOLUTION_BIAS: usize,
> = recurrent::Mamba<
    Projection<
        HbmTensor<bf16, m![1], m![2, Intermediate, Width]>,
        [HbmTensor<bf16, m![1], m![2, Intermediate]>; BIAS],
    >,
    Convolution<
        HbmTensor<bf16, m![1], m![Intermediate, 1, Kernel]>,
        [HbmTensor<bf16, m![1], m![Intermediate]>; CONVOLUTION_BIAS],
    >,
    Fork<(
        Linear<Intermediate, Rank>,
        Linear<Intermediate, State>,
        Linear<Intermediate, State>,
    )>,
    Then<Linear<Rank, Intermediate, 1>, Softplus>,
    SelectiveScan<
        HbmTensor<f32, m![1], m![Intermediate, State]>,
        HbmTensor<f32, m![1], m![Intermediate]>,
    >,
    Gate,
    Linear<Intermediate, Width, BIAS>,
>;

pub type Linear<In, Out, const BIAS: usize = 0, D = bf16> =
    Affine<HbmTensor<D, m![1], m![Out, In]>, [HbmTensor<D, m![1], m![Out]>; BIAS]>;

pub type Head<Width, Vocabulary> = Then<Cast<bf16>, Then<Linear<Width, Vocabulary>, Cast<f32>>>;

pub type Rows<Width, Count, D = bf16> = HbmTensor<D, m![1], m![Count, Width]>;

pub type State<Width, Kernel, Recurrence> = recurrent::Ssm<
    HbmTensor<f32, m![1], m![Width, Kernel]>,
    HbmTensor<f32, m![1], m![Width, Recurrence]>,
>;

/// A published Mamba checkpoint size, by its hub spelling.
pub struct Size<const N: &'static str>;

/// The axes every Mamba size shares, and those its rows' width fixes.
pub mod axis {
    use furiosa_opt_std::prelude::*;
    use std::marker::PhantomData;

    axes![State = 16, Kernel = 4, Vocabulary = 50280];

    /// Features of the mixer of rows of `H` features: `expand * hidden_size`, `expand` 2.
    #[derive(Clone, Debug)]
    pub struct Intermediate<H>(PhantomData<H>);

    impl<H: AxisName> AxisName for Intermediate<H> {
        const NAME: Ident = Ident::new("Intermediate");
        const SIZE: usize = 2 * H::SIZE;
    }

    /// Features the time steps of rows of `H` features are projected through: `time_step_rank`,
    /// Transformers' `auto`.
    #[derive(Clone, Debug)]
    pub struct Rank<H>(PhantomData<H>);

    impl<H: AxisName> AxisName for Rank<H> {
        const NAME: Ident = Ident::new("Rank");
        const SIZE: usize = H::SIZE.div_ceil(16);
    }
}

/// The norm of every Mamba size: `layer_norm_epsilon` 1e-5.
type MambaNorm<W, X> =
    RmsNorm<W, EPSILON, m![1], Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>>;

const EPSILON: usize = 1e-5_f32.to_bits() as usize;

macro_rules! sizes {
    ($($size:literal: $hidden:literal x $layers:literal;)+) => {$(
        impl Geometry for Size<$size> {
            const SIZE: &'static str = $size;
            const FAMILY: &'static str = "mamba";
            type Hidden = geometry::Hidden<$hidden>;
            type Vocabulary = axis::Vocabulary;
            type Layers<L> = [L; $layers];
            type Norm<W: AxisName, X: Memory> = MambaNorm<W, X>;

            const LAYERS: usize = $layers;
            const EPSILON: usize = EPSILON;
            const TIED_EMBEDDINGS: bool = true;
        }
    )+};
}

sizes! {
    "130m": 768 x 24;
    "370m": 1024 x 48;
    "790m": 1536 x 48;
    "1.4b": 2048 x 48;
    "2.8b": 2560 x 64;
}

/// What a Mamba checkpoint's config declares that its size fixes, and the features Mamba does not
/// serve; an absent key reads as Transformers reads it.
#[derive(Deserialize)]
pub struct Config {
    hidden_size: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    #[serde(default)]
    expand: Option<usize>,
    #[serde(default)]
    state_size: Option<usize>,
    #[serde(default)]
    conv_kernel: Option<usize>,
    /// A count, or `"auto"`.
    #[serde(default)]
    time_step_rank: Option<Value>,
    #[serde(default)]
    layer_norm_epsilon: Option<f64>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    use_bias: bool,
    #[serde(default)]
    use_conv_bias: Option<bool>,
    #[serde(default)]
    residual_in_fp32: Option<bool>,
}

/// A config is of a size when it declares the size's values, at most its compiled vocabulary rows,
/// and none of the features Mamba does not serve.
impl<const N: &'static str> TryFrom<Config> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let hidden = <Self as Geometry>::Hidden::SIZE;
        let rank = axis::Rank::<<Self as Geometry>::Hidden>::SIZE;
        let declared_rank = match &config.time_step_rank {
            Some(Value::String(auto)) if auto == "auto" => Some(rank),
            Some(declared) => declared
                .as_u64()
                .and_then(|count| usize::try_from(count).ok()),
            None => Some(rank),
        };
        let fixed = [
            ("hidden_size", Some(config.hidden_size), hidden),
            (
                "num_hidden_layers",
                Some(config.num_hidden_layers),
                Self::LAYERS,
            ),
            (
                "expand",
                Some(config.expand.unwrap_or(2)),
                axis::Intermediate::<<Self as Geometry>::Hidden>::SIZE / hidden,
            ),
            (
                "state_size",
                Some(config.state_size.unwrap_or(16)),
                axis::State::SIZE,
            ),
            (
                "conv_kernel",
                Some(config.conv_kernel.unwrap_or(4)),
                axis::Kernel::SIZE,
            ),
            ("time_step_rank", declared_rank, rank),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| *declared != Some(*held))
        {
            let (kind, declared) = match declared {
                Some(count) => (io::ErrorKind::Unsupported, count.to_string()),
                None => (io::ErrorKind::InvalidData, "not a count".into()),
            };
            let reason = format!("{key}: {declared}, the size {N} holds {held}");
            return Err(io::Error::new(kind, reason));
        }
        vocabulary::<<Self as Geometry>::Vocabulary>(config.vocab_size)?;
        let epsilon = config.layer_norm_epsilon.unwrap_or(1e-5);
        if (epsilon as f32).to_bits() as usize != Self::EPSILON {
            return Err(unsupported(&format!(
                "layer_norm_epsilon: {epsilon}, another than the size's"
            )));
        }
        let tied = config.tie_word_embeddings.unwrap_or(true);
        if tied != Self::TIED_EMBEDDINGS {
            return Err(unsupported(&format!(
                "tie_word_embeddings: {tied}, the size ties: {}",
                Self::TIED_EMBEDDINGS
            )));
        }
        let features = [
            (
                config
                    .hidden_act
                    .as_deref()
                    .is_some_and(|act| act != "silu"),
                "hidden_act: Mamba gates its mixer with SiLU",
            ),
            (config.use_bias, "use_bias: Mamba projects without biases"),
            (
                config.use_conv_bias == Some(false),
                "use_conv_bias: Mamba's convolution adds its bias",
            ),
            (
                config.residual_in_fp32 == Some(false),
                "residual_in_fp32: Mamba keeps its residual rows in FP32",
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

/// Source: vLLM `vllm/model_executor/models/mamba.py`.
pub type MambaForCausalLM<
    Embedding,
    Norm,
    InputProjection,
    Convolution,
    Selection,
    Time,
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
        recurrent::Mamba<
            InputProjection,
            Convolution,
            Selection,
            Time,
            Recurrence,
            OutputGate,
            OutputProjection,
        >,
        Add,
    >; DEPTH],
    Norm,
    Head,
>;
