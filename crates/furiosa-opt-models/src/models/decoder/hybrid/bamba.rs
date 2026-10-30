//! Bamba, by published size: a size's [`Geometry`] is what its checkpoints declare, and its
//! [`Config`] refuses a value the size does not hold or a feature Bamba does not serve.
//!
//! - Every size interleaves attention layers at [`ATTENTION_LAYERS`] with Mamba2 mixer layers in
//!   checkpoint order ([`ORDER`]); its attention rotates the first half of each head's features.
//! - [`Model`] is the catalog's composition, a call's attention and recurrent state as its input.

use crate::models::config::{
    self, Positive,
    decoder::Prefix,
    scaling::{Defaults, Spectrum},
};
use crate::models::decoder::attention::{Chunks, Geometry};
use crate::models::decoder::recurrent::mamba::{self, Ids, Rows};
use crate::models::decoder::recurrent::mamba2;
use crate::models::{Family, Scalar};
use crate::modules::attention;
use crate::modules::recurrent::GateFirst;
use crate::modules::{feedforward::glu::Glu, language, transformer};
use crate::ops::{
    self,
    attention::{
        geometry::{QueryGroups, QueryWidth},
        gqa,
    },
    embedding::Lookup,
    index::{
        gather::{Gather, RowStorage},
        scatter::Scatter,
        table::KeyValue,
    },
    linear::Affine,
    normalization::{Rms, RmsNorm},
    pointwise::{activation::Silu, add::Add, multiply::Multiply},
    recurrent::Length,
    rotary::prefix,
};
use crate::tensor::form::Weight;
use crate::tensor::memory::Memory;
use crate::{Branch, Interleave, Module};
use furiosa_opt_std::prelude::*;
use nine_billion::Kernel;
use nine_billion::*;
use serde::Deserialize;
use serde_json::Value;
use std::{borrow::BorrowMut, io, marker::PhantomData};

pub type Model = language::Readout<Body<Function>, mamba::Head<Width, Vocabulary>>;

#[derive(DeviceSend)]
pub struct Body<Function>(pub Function);

/// Attention layers read the step's positions and mask, recurrent layers its attended length; a
/// layer's slot holds its history or recurrent state beside its feed-forward's.
impl<Queries: AxisName, Keys: AxisName, Storage: BorrowMut<Cache>>
    Module<&mut Input<Queries, Keys, Storage>, (), &mut ()> for Body<Function>
{
    type Output = Rows<Width, Queries>;

    fn forward(
        &self,
        device: &mut Device,
        _: (),
        _: &mut (),
        input: &mut Input<Queries, Keys, Storage>,
    ) -> Self::Output {
        let Input {
            tokens,
            step,
            attention,
            recurrent,
        } = input;
        let mut histories = attention.each_mut().map(|history| {
            let History { cache, write, read } = history;
            (cache.borrow_mut(), (&*write, &step.length), &*read)
        });
        let attention = histories.each_mut().map(|(cache, write, read)| {
            let cache = (&mut **cache, &*write, *read);
            let attention = (Default::default(), cache, Default::default());
            (attention, Default::default())
        });
        let recurrent = recurrent
            .each_mut()
            .map(|state| ((Default::default(), state), Default::default()));
        self.0.forward(
            device,
            ((&step.positions, &step.mask), &step.length),
            &mut (((), (attention, recurrent)), ()),
            &*tokens,
        )
    }
}

type Function = language::Normalized<
    language::Backbone<Lookup<HbmTensor<bf16, m![1], m![Vocabulary, Width]>>, Layers>,
    Norm,
>;

type Layers = Interleave<([Block<Attention>; 3], [Block<Mixer>; 29]), ORDER>;

pub type Block<M> = transformer::PreNorm<M, Mlp, Norm, Add>;

type Mlp = Glu<
    Linear<Width, FeedForward>,
    Silu,
    Linear<Width, FeedForward>,
    Multiply,
    Linear<FeedForward, Width>,
>;

type Linear<In, Out> =
    Affine<HbmTensor<bf16, m![1], m![Out, In]>, [HbmTensor<bf16, m![1], m![Out]>; 0]>;

type Norm = Rms<HbmTensor<bf16, m![1], m![Width]>, Scalar>;

pub type Attention = attention::Cached<
    attention::Projection<Linear<Width, Width>, Linear<Width, KvWidth>>,
    prefix::Rotary<AttentionHead, Rotated>,
    Scatter,
    Gather,
    ops::attention::Attention<PhantomData<AttentionHeads>, Scalar>,
    Linear<Width, Width>,
>;

pub type Mixer = mamba2::Mixer<
    Width,
    Intermediate,
    Kernel,
    State,
    Groups,
    Heads,
    0,
    1,
    GateFirst,
    { mamba2::TIME_STEP_LIMIT[0].to_bits() as usize },
    { mamba2::TIME_STEP_LIMIT[1].to_bits() as usize },
>;

pub type Ssm = mamba2::State<Intermediate, Kernel, State, Groups, Heads, Head>;
pub type Cache = KeyValue<RowStorage<KvWidth, bf16>>;
pub type Offsets<Count> = KeyValue<Ids<Count>>;
pub type Mask<Queries, Keys> = HbmTensor<f32, m![1], m![Queries, Keys]>;

#[derive(DeviceSend)]
pub struct History<Queries: AxisName, Keys: AxisName, Storage = Cache> {
    pub cache: Storage,
    pub write: Offsets<Queries>,
    pub read: Offsets<Keys>,
}

/// The positions, attended length and mask every layer of one call reads.
#[derive(DeviceSend)]
pub struct Step<Queries: AxisName, Keys: AxisName> {
    pub positions: Ids<Queries>,
    pub length: Length,
    pub mask: Mask<Queries, Keys>,
}

/// Both cache arrays follow their compact layer order; their positions correspond to ORDER.
#[derive(DeviceSend)]
pub struct Input<Queries: AxisName, Keys: AxisName, Storage = Cache> {
    pub tokens: Ids<Queries>,
    pub step: Step<Queries, Keys>,
    pub attention: [History<Queries, Keys, Storage>; 3],
    pub recurrent: [Ssm; 29],
}

pub mod nine_billion {
    use furiosa_opt_std::prelude::*;

    axes![
        Vocabulary = 128256,
        Width = 4096,
        KvWidth = 1024,
        FeedForward = 14336,
        Intermediate = 8192,
        Kernel = 4,
        State = 128,
        Groups = 1,
        Heads = 128,
        Head = 64,
        AttentionHeads = 32,
        AttentionHead = 128,
        KvHeads = 8,
        Rotated = 64
    ];
}

pub const ATTENTION_LAYERS: [usize; 3] = [9, 18, 27];

pub const ORDER: &[Branch] = &{
    let mut order = [Branch::Second; 32];
    let mut at = 0;
    while at < ATTENTION_LAYERS.len() {
        order[ATTENTION_LAYERS[at]] = Branch::First;
        at += 1;
    }
    order
};

/// A published Bamba checkpoint size, by its hub spelling.
pub struct Size<const N: &'static str>;

/// `rms_norm_eps` 1e-5, as the bits of the FP32 nearest to it.
const EPSILON: usize = 1e-5_f32.to_bits() as usize;

/// The attention layers' geometry, which the size's attention decoder template holds.
impl Geometry for Size<"9B"> {
    const SIZE: &'static str = "9B";
    const FAMILY: &'static str = "bamba";
    type Queries = QueryWidth<{ AttentionHeads::SIZE * AttentionHead::SIZE }>;
    type Hidden = Width;
    type Intermediate = FeedForward;
    type Vocabulary = Vocabulary;
    type KvHeads = KvHeads;
    type Groups = QueryGroups<{ AttentionHeads::SIZE / KvHeads::SIZE }>;
    type Head = AttentionHead;
    type Layers<L> = [L; ORDER.len()];
    type Norm<W: AxisName, X: Memory> =
        RmsNorm<W, EPSILON, m![1], Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>>;

    const LAYERS: usize = ORDER.len();
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

/// What a Bamba checkpoint's config declares that its size fixes, and the features Bamba does not
/// serve; an absent key reads as Transformers reads it.
#[derive(Deserialize)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    attn_layer_indices: Vec<usize>,
    mamba_n_heads: usize,
    /// A count, or `"auto"`.
    #[serde(default)]
    mamba_d_head: Option<Value>,
    #[serde(default)]
    mamba_n_groups: Option<usize>,
    #[serde(default)]
    mamba_d_state: Option<usize>,
    #[serde(default)]
    mamba_d_conv: Option<usize>,
    #[serde(default)]
    mamba_expand: Option<usize>,
    /// The rotated features of a head, which Transformers fixes at half of them.
    #[serde(default)]
    attn_rotary_emb: Option<usize>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    mamba_proj_bias: bool,
    #[serde(default)]
    mamba_conv_bias: Option<bool>,
    #[serde(flatten, deserialize_with = "Config::spectrum")]
    rope: Spectrum,
}

/// `BambaConfig`'s: Transformers' default base, and half of each head, which the class fixes.
impl Defaults for Config {
    const THETA: Positive = Positive::new(10000.0);
    const FRACTION: f64 = 0.5;
}

/// A config is of a size when it declares the size's values and layer order, at most its compiled
/// vocabulary rows and context, and none of the features Bamba does not serve.
impl<const N: &'static str> TryFrom<Config> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let heads = <Self as Geometry>::KvHeads::SIZE;
        let hidden = <Self as Geometry>::Hidden::SIZE;
        let head = match &config.mamba_d_head {
            Some(Value::String(auto)) if auto == "auto" => {
                Some(Intermediate::SIZE / config.mamba_n_heads.max(1))
            }
            Some(declared) => declared
                .as_u64()
                .and_then(|count| usize::try_from(count).ok()),
            None => Some(Intermediate::SIZE / config.mamba_n_heads.max(1)),
        };
        let fixed = [
            ("hidden_size", Some(config.hidden_size), hidden),
            (
                "intermediate_size",
                Some(config.intermediate_size),
                <Self as Geometry>::Intermediate::SIZE,
            ),
            (
                "num_hidden_layers",
                Some(config.num_hidden_layers),
                Self::LAYERS,
            ),
            (
                "num_key_value_heads",
                Some(config.num_key_value_heads),
                heads,
            ),
            (
                "num_attention_heads",
                Some(config.num_attention_heads),
                heads * <Self as Geometry>::Groups::SIZE,
            ),
            (
                "attn_rotary_emb",
                Some(config.attn_rotary_emb.unwrap_or(Rotated::SIZE)),
                Rotated::SIZE,
            ),
            (
                "mamba_expand",
                Some(config.mamba_expand.unwrap_or(2)),
                Intermediate::SIZE / hidden,
            ),
            ("mamba_n_heads", Some(config.mamba_n_heads), Heads::SIZE),
            ("mamba_d_head", head, Head::SIZE),
            (
                "mamba_n_groups",
                Some(config.mamba_n_groups.unwrap_or(1)),
                Groups::SIZE,
            ),
            (
                "mamba_d_state",
                Some(config.mamba_d_state.unwrap_or(256)),
                State::SIZE,
            ),
            (
                "mamba_d_conv",
                Some(config.mamba_d_conv.unwrap_or(4)),
                Kernel::SIZE,
            ),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| *declared != Some(*held))
        {
            let declared = declared.map_or("not a count".into(), |count| count.to_string());
            return Err(invalid(format!(
                "{key}: {declared}, the size {N} holds {held}"
            )));
        }
        if config.attn_layer_indices != ATTENTION_LAYERS {
            return Err(invalid(format!(
                "attn_layer_indices: {:?}, the size {N} holds {ATTENTION_LAYERS:?}",
                config.attn_layer_indices
            )));
        }
        let rows = <Self as Geometry>::Vocabulary::SIZE;
        if !(1..=rows).contains(&config.vocab_size) {
            return Err(invalid(format!(
                "vocab_size: {}, the size compiles {rows} rows",
                config.vocab_size
            )));
        }
        let epsilon = config.rms_norm_eps.unwrap_or(1e-5);
        if (epsilon as f32).to_bits() as usize != Self::EPSILON {
            return Err(invalid(format!(
                "rms_norm_eps: {epsilon}, another than the size's"
            )));
        }
        if config.tie_word_embeddings != Self::TIED_EMBEDDINGS {
            return Err(invalid(format!(
                "tie_word_embeddings: {}, the size ties: {}",
                config.tie_word_embeddings,
                Self::TIED_EMBEDDINGS
            )));
        }
        Prefix::<AttentionHead, Rotated>::try_from(config.rope).map_err(config::refusal)?;
        let context = Chunks::SIZE * gqa::Cache::SIZE;
        if config.max_position_embeddings.unwrap_or(262144) > context {
            return Err(unsupported(&format!(
                "max_position_embeddings: past the compiled context of {context}"
            )));
        }
        let features = [
            (
                config
                    .hidden_act
                    .as_deref()
                    .is_some_and(|act| act != "silu"),
                "hidden_act: Bamba gates its feed-forward and mixer with SiLU",
            ),
            (
                config.mamba_proj_bias,
                "mamba_proj_bias: Bamba's mixer projects without biases",
            ),
            (
                config.mamba_conv_bias == Some(false),
                "mamba_conv_bias: Bamba's convolution adds its bias",
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
