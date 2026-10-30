//! A compiled decoder's geometry and cache support.
//!
//! - A decoder's sizes are type data: a [`Geometry`] holds the model axes and the [`Group`] of
//!   each attention group, and [`Projections`] holds each projection's parameter [`Form`].
//! - [`Model`] declares a decoder of one attention group, [`Interleaved`] one whose layers
//!   interleave two; a family's configuration names the declaration its checkpoint compiles to.
//! - A group's [`Rotation`] is the coordinates its positions turn: the leading `R` of each head
//!   for an axis `R`.
//! - [`attention`] holds the attention decoder families declared by their Hugging Face classes.

use crate::load::{Checkpoint, LoadError};
use crate::models::Family;
use crate::models::config::{self, scaling::Spectrum};
use crate::modules::feedforward::glu::Glu;
use crate::modules::language::Greedy;
use crate::modules::parallel::RowParallel;
use crate::ops;
use crate::ops::collective::AllGather;
use crate::ops::linear::{self, fp8};
use crate::ops::pointwise::multiply::Multiply;
use crate::ops::pointwise::multiply::{QueryScale, Temperature};
use crate::ops::rotary;
use crate::specialize::attention::Strategy;
use crate::tensor::axis;
use crate::tensor::mapping::Reshape;
use crate::tensor::parallel::Chip;
use crate::tensor::{
    axis::Single,
    memory::{Hbm, Memory},
    parallel::{Parallel, Parallelism as Placement},
    scalar::F32,
};
use crate::{Sequential, Then};
use furiosa_opt_std::prelude::*;
use std::fmt::Debug;
use std::{io, marker::PhantomData};

pub mod attention;
pub mod hybrid;
pub mod recurrent;

/// A decoder's declaration, holding no parameters: its geometry, cache scalar, placement `P`, the
/// [`Query`] pass, rotated coordinates, call [`Envelope`] and loaded `Composition`.
pub struct Model<
    Heads,
    Head,
    Windows,
    const LAYERS: usize,
    CacheScalar,
    Hidden,
    Vocabulary,
    Composition,
    P = Parallel<1>,
    Query = (),
    Rotated = Head,
    Rows = Single,
>(
    PhantomData<(
        Heads,
        Head,
        Windows,
        CacheScalar,
        Hidden,
        Vocabulary,
        Composition,
        P,
        Query,
        Rotated,
        Rows,
    )>,
);

/// The rows one call runs, one or an envelope whose live rows the call counts; it selects the views
/// a composition's leaves take them in and the vocabulary head that reads them.
pub trait Envelope: AxisName {
    /// `Queries` query heads of `Head` coordinates, `Groups` of them for each of `Heads` key/value
    /// heads, as one projection's features are read.
    type Queries<Queries: AxisName, Heads: AxisName, Groups: AxisName, Head: AxisName>: M;

    /// The vocabulary head of a decoder of geometry `G` on the chips of `P`.
    type Head<G: Shape, P: Placement>;
}

/// One row reads its query heads in projection order, and its head scores the checkpoint's tokens
/// on every chip of the placement.
impl Envelope for Single {
    type Queries<Queries: AxisName, Heads: AxisName, Groups: AxisName, Head: AxisName> =
        m![Queries, Head];

    type Head<G: Shape, P: Placement> = Logits<<G as Shape>::Hidden, <G as Shape>::Vocabulary, P>;
}

/// Several rows hold each key/value head's query heads together, as their attention reads them;
/// their head projects onto the compiled vocabulary and selects among the checkpoint's tokens.
impl<const N: usize> Envelope for axis::Rows<N> {
    type Queries<Queries: AxisName, Heads: AxisName, Groups: AxisName, Head: AxisName> =
        m![Heads, Groups, Head];

    type Head<G: Shape, P: Placement> = Greedy<<G as Shape>::Hidden, <G as Shape>::Compiled>;
}

/// Each row's scores of the `Vocabulary` tokens: each chip of `P` projects its part, and the parts
/// gather on every chip.
pub type Logits<Hidden, Vocabulary, P = Parallel<1>> = Sequential<
    linear::Affine<
        HbmTensor<
            bf16,
            <P as Placement>::Chip,
            m![{ Symbol<<P as Placement>::Local<Vocabulary>> }, Hidden],
        >,
        (),
    >,
    AllGather<P>,
>;

/// A homogeneous decoder group loaded with one shared rotary coefficient table.
pub trait Config: Family + Send + 'static {
    /// Geometry and composition for the requested cache scalar.
    type Model<CacheScalar: Scalar>;

    /// Chips of each tensor-parallel placement the composition compiles for, one chip first.
    const TENSORS: &'static [usize] = &[1];

    /// The leading coordinates of each head that the group's positions rotate.
    type Rotated: AxisName;

    /// Maximum token positions and the rotary base, in that order.
    fn rotary(&self) -> (usize, f32);

    /// The frequency of each rotated pair: the unscaled spectrum of the rotary base over heads
    /// whose every coordinate rotates, unless the configuration binds others.
    fn frequencies(&self) -> io::Result<rotary::PairFrequencies<Self::Rotated>> {
        rotary::PairFrequencies::unscaled::<Self::Rotated>(self.rotary().1)
    }
}

/// The tensor-parallel placements a composition compiles, one chip first: `(Parallel<1>,)`, or
/// one, two and four chips.
pub(crate) trait Placements: 'static {
    /// Chips of each placement.
    const TENSORS: &'static [usize];
}

impl Placements for (Parallel<1>,) {
    const TENSORS: &'static [usize] = &[1];
}

impl Placements for (Parallel<1>, Parallel<2>, Parallel<4>) {
    const TENSORS: &'static [usize] = &[1, 2, 4];
}

/// What each row's queries pass after their rotation: nothing, or [`QueryScale`], which
/// multiplies them by the temperature of the row's position.
pub trait Query: 'static {
    /// The position op of a group whose [`Rotation`] positions its queries and keys with `Op`.
    type Position<Op>;

    /// What the op reads from the checkpoint: nothing, or the temperature of every position.
    type Parameter: Copy + Send + Debug;

    /// The parameter of a checkpoint that declares `temperature`; a temperature the op does not
    /// read, or a missing one it does, is refused.
    fn bind(temperature: Option<Temperature>) -> io::Result<Self::Parameter>;
}

impl Query for () {
    type Position<Op> = Op;

    type Parameter = ();

    fn bind(temperature: Option<Temperature>) -> io::Result<()> {
        match temperature {
            None => Ok(()),
            Some(temperature) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("found query temperature {temperature:?}, compiled unscaled queries"),
            )),
        }
    }
}

impl Query for QueryScale {
    type Position<Op> = Scaled<Op>;

    type Parameter = Temperature;

    fn bind(temperature: Option<Temperature>) -> io::Result<Temperature> {
        temperature.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "found no query temperature, compiled queries that scale by one",
            )
        })
    }
}

/// Positions queries and keys with `Op`, then scales the queries by their rows' temperatures: the
/// positions hold the rotation's operand, then the temperatures.
#[derive(Clone, Copy, Default, DeviceSend)]
pub struct Scaled<Op>(pub Op);

impl<'x, Q, K, R, T, Y, Z, Op, Env: Copy> crate::Module<((Q, K), &'x (R, T)), Env, &mut ()>
    for Scaled<Op>
where
    Op: for<'d> crate::Module<((Q, K), &'x R), Env, &'d mut (), Output = Y>,
    QueryScale: for<'d> crate::Module<(Y, &'x T), Env, &'d mut (), Output = Z>,
{
    type Output = Z;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (heads, (rotation, temperatures)): ((Q, K), &'x (R, T)),
    ) -> Z {
        let rotated = self
            .0
            .forward(&mut *device, env, &mut (), (heads, rotation));
        QueryScale.forward(device, env, &mut (), (rotated, temperatures))
    }
}

/// A configuration whose declaration's rotated queries pass `Q`, with the parameter `Q` reads.
pub trait Queries<Q: Query> {
    fn query(&self) -> Q::Parameter;
}

/// Unscaled queries read nothing.
impl<S: Config> Queries<()> for S {
    fn query(&self) {}
}

/// Declares a decoder whose layers interleave `FIRST` layers of `G`'s first attention group and
/// `SECOND` of its second, with cache scalar `Cache`; `Composition` is the loaded body/readout.
pub struct Interleaved<G, const FIRST: usize, const SECOND: usize, Cache, Composition>(
    PhantomData<(G, Cache, Composition)>,
);

/// A decoder of two interleaved attention groups, each with its own cache and rotary table.
pub trait Paired: Family + Send + 'static {
    /// Geometry and composition for the requested cache scalar.
    type Model<CacheScalar: Scalar>;

    /// The rotation of the first group's layers, then the second's.
    type First: Rotation;
    type Second: Rotation;

    /// Maximum token positions.
    fn positions(&self) -> usize;

    /// The frequencies the first group's rotation binds and the window it attends in tokens,
    /// then the second group's.
    fn groups(
        &self,
    ) -> (
        (<Self::First as Rotation>::Frequencies, usize),
        (<Self::Second as Rotation>::Frequencies, usize),
    );

    /// The factors of the composition's constant scales.
    fn constants(&self) -> Constants;
}

/// The factors of a composition's constant scale leaves and the epsilons of its norms that hold
/// theirs as data, which its configuration fixes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Constants {
    /// Multiplier of every embedded token row.
    pub embedding: Option<f32>,
    /// Multiplier of every normalized query head.
    pub query: Option<f32>,
    /// Multiplier of every logit, before a softcap bounds it.
    pub logits: Option<f32>,
    /// The bound `c` of a head whose logits `z` become `c * tanh(z / c)`.
    pub softcap: Option<f32>,
    /// The epsilon of a norm whose type fixes none, by its side of a branch.
    pub epsilon: Epsilons,
}

/// The epsilons of the norms before a branch, or outside every branch, and after a branch.
#[derive(Clone, Copy, Debug, Default)]
pub struct Epsilons {
    pub before: Option<f32>,
    pub after: Option<f32>,
}

/// One attention group: `Heads` key/value heads of `Head`, `Groups` query heads each, both
/// projection widths, the heads' [`Rotation`] and the `Windows` of cache blocks a row reads.
pub struct Group<Heads, Head, Groups, QueryWidth, KvWidth, Rotated, Windows>(
    PhantomData<(Heads, Head, Groups, QueryWidth, KvWidth, Rotated, Windows)>,
);

/// A [`Group`]'s axes by role.
pub trait Attention: 'static {
    type Heads: AxisName;
    type Head: AxisName;
    type Groups: AxisName;
    type QueryWidth: AxisName;
    type KvWidth: AxisName;
    type Rotated: Rotation;
    type Windows: AxisName;

    /// Whether the widths are the head products the attention kernels assume.
    const CONSISTENT: bool = Self::Groups::SIZE * Self::Heads::SIZE * Self::Head::SIZE
        == Self::QueryWidth::SIZE
        && Self::Heads::SIZE * Self::Head::SIZE == Self::KvWidth::SIZE
        && Self::Rotated::COORDINATES <= Self::Head::SIZE
        && Self::Windows::SIZE > 0;
}

impl<Heads, Head, Groups, QueryWidth, KvWidth, Rotated, Windows> Attention
    for Group<Heads, Head, Groups, QueryWidth, KvWidth, Rotated, Windows>
where
    Heads: AxisName,
    Head: AxisName,
    Groups: AxisName,
    QueryWidth: AxisName,
    KvWidth: AxisName,
    Rotated: Rotation,
    Windows: AxisName,
{
    type Heads = Heads;
    type Head = Head;
    type Groups = Groups;
    type QueryWidth = QueryWidth;
    type KvWidth = KvWidth;
    type Rotated = Rotated;
    type Windows = Windows;
}

/// The coordinates of each head that a group's positions turn, which decide how its layers
/// position queries and keys and what a call prepares for them.
pub trait Rotation: 'static {
    /// Rotated coordinates of each head.
    const COORDINATES: usize;

    /// The op that positions a group's queries and keys of `Head` coordinates.
    type Position<Head: AxisName, D: Scalar, Chips: M>;

    /// What a group binds at load to position its rows: the frequency of each rotated pair.
    type Frequencies: Clone + Send;

    /// The frequencies a `Head`-wide group binds where a checkpoint's `spectrum`, `None` for
    /// layers that rotate nothing, turns them; a spectrum this rotation does not hold is refused.
    fn spectrum<Head: AxisName>(spectrum: Option<Spectrum>) -> io::Result<Self::Frequencies>;
}

/// Rotary positions turn the leading `R` coordinates of each head.
impl<R: AxisName> Rotation for R {
    const COORDINATES: usize = R::SIZE;

    type Position<Head: AxisName, D: Scalar, Chips: M> = rotary::Rotary<Head, R, D, Chips>;

    type Frequencies = rotary::PairFrequencies<R>;

    fn spectrum<Head: AxisName>(
        spectrum: Option<Spectrum>,
    ) -> io::Result<rotary::PairFrequencies<R>> {
        let spectrum = spectrum.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "found layers that rotate nothing, compiled {} rotated",
                    R::SIZE
                ),
            )
        })?;
        spectrum.bind::<Head, R>()
    }
}

/// A decoder's hidden, feed-forward and vocabulary widths, its query head count, its attention
/// groups and the vocabulary rows its head computes; a decoder of one group uses `First` only.
pub struct Geometry<
    Hidden,
    Intermediate,
    Vocabulary,
    Queries,
    First,
    Second = First,
    Compiled = Vocabulary,
>(
    PhantomData<(
        Hidden,
        Intermediate,
        Vocabulary,
        Queries,
        First,
        Second,
        Compiled,
    )>,
);

/// A [`Geometry`]'s axes by role.
pub trait Shape: 'static {
    type Hidden: AxisName;
    type Intermediate: AxisName;
    type Vocabulary: AxisName;
    /// The checkpoint's tokens, then the zero rows that let the head tile.
    type Compiled: AxisName;
    type Queries: AxisName;
    type First: Attention;
    type Second: Attention;

    /// Whether each group's widths are the products the attention kernels assume and each
    /// group's key/value heads, times their query groups, are the decoder's query heads.
    const CONSISTENT: bool = Self::First::CONSISTENT
        && Self::Second::CONSISTENT
        && <Self::First as Attention>::Heads::SIZE * <Self::First as Attention>::Groups::SIZE
            == Self::Queries::SIZE
        && <Self::Second as Attention>::Heads::SIZE * <Self::Second as Attention>::Groups::SIZE
            == Self::Queries::SIZE;
}

impl<Hidden, Intermediate, Vocabulary, Queries, First, Second, Compiled> Shape
    for Geometry<Hidden, Intermediate, Vocabulary, Queries, First, Second, Compiled>
where
    Hidden: AxisName,
    Intermediate: AxisName,
    Vocabulary: AxisName,
    Queries: AxisName,
    First: Attention,
    Second: Attention,
    Compiled: AxisName,
{
    type Hidden = Hidden;
    type Intermediate = Intermediate;
    type Vocabulary = Vocabulary;
    type Compiled = Compiled;
    type Queries = Queries;
    type First = First;
    type Second = Second;
}

/// A projection's parameter form: the leaf that projects `In`-wide rows to `Out` features of
/// scalar `D` on chips `Chips`, its weight's stored values held as `V` and its other leaves in
/// memory `X`. A form without a scalar or chip choice ignores them.
///
/// A head-major decoder layer binds its six projections through the form's own items, each over
/// the axes it projects and the memory `X` of its leaves; a form without them leaves [`Absent`].
pub trait Form: 'static {
    /// The scalar a weight stores its values in: BF16 values, or the codes its scales restore.
    type Stored: MaterializableScalar;

    type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V>;

    /// The projection of `Hidden` rows to `Heads` query heads of `Head` features.
    type Query<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> = Absent;

    /// The projection of `Hidden` rows to `Heads` key heads of `Head` features.
    type Key<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> = Absent;

    /// The projection of `Hidden` rows to `Heads` value heads of `Head` features.
    type Value<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> = Absent;

    /// The projection of the attended `Queries` features back to `Hidden` rows.
    type Output<Queries: AxisName, Hidden: AxisName, X: Memory> = Absent;

    /// The packed gate and up projections of `Hidden` rows to `Intermediate` features each.
    type GateUp<Hidden: AxisName, Intermediate: AxisName, X: Memory> = Absent;

    /// The projection of `Intermediate` features back to `Hidden` rows.
    type Down<Hidden: AxisName, Intermediate: AxisName, X: Memory> = Absent;

    /// What a checkpoint storing this form's weights declares about them.
    const REQUIREMENT: Requirement;

    /// The KV-cache scalars a value projection of this form writes: `(bf16,)`, or
    /// `(bf16, CacheScalar)` for a form that also writes `CacheScalar`.
    type Caches;
}

/// The projection a [`Form`] does not provide: no value has this type, so no tree holds one.
pub enum Absent {}

/// The projection in form `F` whose weight a checkpoint stores as one `[Out, In]` tensor, every
/// leaf in memory `X`.
pub type Projection<F, In, Out, D = bf16, Chips = Chip, X = Hbm> = <F as Form>::Of<
    In,
    Out,
    D,
    Chips,
    X,
    <X as Memory>::Tensor<<F as Form>::Stored, Chips, m![Out, In]>,
>;

/// What a checkpoint declares about the weights its projections store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Requirement {
    /// BF16 weights, which no quantization declares.
    Unquantized,
    /// Quantized weights whose form reads its scales from their tensors alone.
    Undeclared,
    /// Quantized weights that the checkpoint's `quantization_config` declares this way.
    Declared(Declaration),
}

impl Requirement {
    /// What a checkpoint declares for projections of requirements `self` and `other`: a
    /// declaration covers the projections that read none, and two different ones cannot hold.
    pub const fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Declared(first), Self::Declared(second)) => {
                assert!(
                    first.same(second),
                    "projections declare different quantizations"
                );
                self
            }
            (Self::Declared(_), _) | (Self::Undeclared, Self::Unquantized) => self,
            (Self::Unquantized, _) | (Self::Undeclared, _) => other,
        }
    }

    /// Whether the weights are quantized.
    pub const fn quantized(self) -> bool {
        !matches!(self, Self::Unquantized)
    }

    /// Refuses `source` unless it holds the weights these projections read: BF16 ones if it declares
    /// no quantization, quantized ones only if a load quantizes them, else the ones it declares.
    pub fn admits(self, source: &Checkpoint) -> Result<(), LoadError> {
        const DECLARED: &str = "config.json:quantization_config";
        let declared = (source.declaration())
            .map(config::parse::<config::Quantization>)
            .transpose()
            .map_err(|error| {
                let (name, reason) = (DECLARED.into(), error.to_string());
                match error.kind() {
                    io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
                    _ => LoadError::InvalidData { name, reason },
                }
            })?;
        let refused = |name: &str, reason: String| {
            Err(LoadError::Unsupported {
                name: name.into(),
                reason,
            })
        };
        let held = match self {
            Self::Unquantized => "BF16 projections".to_owned(),
            Self::Undeclared => "projections a load quantizes from BF16".to_owned(),
            Self::Declared(declaration) => format!("{declaration:?} projections"),
        };
        match (
            self,
            declared.map(|declared| declared.scheme),
            source.quantization(),
        ) {
            (Self::Unquantized, None, None) | (Self::Undeclared, None, Some(_)) => Ok(()),
            (Self::Declared(declaration), Some(scheme), None) if declaration.reads(scheme) => {
                Ok(())
            }
            (Self::Declared(_), None, None) => {
                refused(DECLARED, format!("missing, the line holds {held}"))
            }
            (_, Some(scheme), _) => refused(DECLARED, format!("{scheme:?}, the line holds {held}")),
            (_, None, Some(_)) => refused(
                "quantization",
                format!("a load quantizes the checkpoint, the line holds {held}"),
            ),
            (Self::Undeclared, None, None) => refused(
                "quantization",
                format!("the checkpoint is read as stored, the line holds {held}"),
            ),
        }
    }
}

/// A checkpoint's `quantization_config`, by the projection forms that load it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Declaration {
    /// FP8 weights and activations scaled as the scheme says.
    Fp8(config::Scheme),
    /// NVFP4 weights and inputs under static tensor scales, whose inputs the projections read in
    /// the given form.
    Nvfp4(Inputs),
}

/// The form a [`Late`](crate::tensor::form::nvfp4::Late) projection's inputs enter its contraction
/// in, which decides the factors its tensor scales become.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inputs {
    /// Quantized at run time as the weights are, under the checkpoint's input scale.
    Quantized,
    /// BF16 as they are, so the checkpoint's input scale goes unread.
    Bf16,
}

impl Declaration {
    const fn same(self, other: Self) -> bool {
        match (self, other) {
            (Self::Fp8(first), Self::Fp8(second)) => first as u8 == second as u8,
            (Self::Nvfp4(first), Self::Nvfp4(second)) => first as u8 == second as u8,
            _ => false,
        }
    }

    /// Whether a checkpoint declaring `scheme` stores the weights this declaration reads.
    fn reads(self, scheme: config::Scheme) -> bool {
        match self {
            Self::Fp8(own) => own == scheme,
            Self::Nvfp4(_) => scheme == config::Scheme::Nvfp4,
        }
    }
}

/// The projection parameter forms a checkpoint can store.
pub mod form {
    use super::{Declaration, Form, Inputs, Memory, PhantomData, Requirement, fp8, linear};
    use crate::models::config::Scheme;
    use crate::ops::linear::{output, query};
    use crate::ops::swiglu;
    use crate::tensor;
    use crate::tensor::form::fp8::Block;
    use crate::tensor::form::fp8::Scales;
    use crate::tensor::form::nvfp4::{self, Early, Late, W4A4, W4A16};
    use crate::tensor::parallel::Chip;
    use furiosa_opt_std::prelude::*;

    /// BF16 weights.
    pub struct Affine;

    /// FP8 weights with one FP32 scale per output channel.
    pub struct ChannelScaled;

    /// FP8 weights and activations with static per-tensor scales.
    pub struct TensorScaled;

    /// FP8 weights with one scale per 128 x 128 block, and activations that quantize per 128
    /// inputs at run time.
    pub struct BlockScaled;

    /// FP4 weights with per-16 FP8 block scales and one global scale.
    pub struct Nvfp4;

    /// FP4 weights with per-16 FP8 block scales under a late FP32 tensor scale, whose inputs enter
    /// in form `A`: quantized at run time as the weights are, or as the BF16 they are.
    pub struct Nvfp4BlockScaled<A = W4A4>(PhantomData<A>);

    impl Form for Affine {
        type Stored = bf16;

        type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
            linear::Affine<V, ()>;

        type Query<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> =
            linear::Affine<X::Tensor<bf16, Chip, m![Heads, Head, Hidden]>, ()>;
        type Key<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> =
            linear::Affine<(X::Tensor<bf16, Chip, m![Heads, Head, Hidden]>, ()), ()>;
        type Value<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> =
            Self::Key<Hidden, Heads, Head, X>;
        type Output<Queries: AxisName, Hidden: AxisName, X: Memory> =
            linear::Affine<X::Tensor<bf16, Chip, m![Hidden, Queries]>, ()>;
        type GateUp<Hidden: AxisName, Intermediate: AxisName, X: Memory> =
            linear::Linear<Hidden, swiglu::GateUp<Intermediate>, Chip>;
        type Down<Hidden: AxisName, Intermediate: AxisName, X: Memory> =
            linear::Linear<Intermediate, Hidden, Chip>;

        const REQUIREMENT: Requirement = Requirement::Unquantized;

        type Caches = (bf16,);
    }

    impl Form for ChannelScaled {
        type Stored = f8e4m3;

        type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
            fp8::ChannelScaled<In, Out, V, X::Tensor<f32, Chip, m![Out, 1 # 8]>>;

        const REQUIREMENT: Requirement = Requirement::Undeclared;

        type Caches = (bf16,);
    }

    impl Form for TensorScaled {
        type Stored = f8e4m3;

        type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
            fp8::TensorScaled<In, Out, D, Chips, V, X::Tensor<f32, Chips, m![1 # 8]>>;

        const REQUIREMENT: Requirement = Requirement::Declared(Declaration::Fp8(Scheme::Tensor));

        type Caches = (bf16, f8e4m3);
    }

    impl Form for BlockScaled {
        type Stored = f8e4m3;

        type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
            tensor::form::BlockScaled<
                In,
                Out,
                f8e4m3,
                f32,
                Scales<In, Out>,
                Block,
                Chips,
                V,
                X::Tensor<f32, Chips, Scales<In, Out>>,
            >;

        /// Each run of heads spread over the slices, the queries' and keys' with their heads'
        /// squared scales, the values' with their heads' scales; every leaf in HBM, whatever `X`.
        type Query<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> = linear::Affine<
            query::fp8::Weights<Hidden, Heads, Head, query::fp8::HeadSquares<Heads>>,
            (),
        >;
        type Key<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> = linear::Affine<
            (
                query::fp8::Weights<Hidden, Heads, Head, query::fp8::HeadSquares<Heads>>,
                (),
            ),
            (),
        >;
        type Value<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> = linear::Affine<
            query::fp8::Weights<Hidden, Heads, Head, query::fp8::HeadScales<Heads, Head>>,
            (),
        >;
        type Output<Queries: AxisName, Hidden: AxisName, X: Memory> =
            output::fp8::Stored<Hidden, Queries>;
        type GateUp<Hidden: AxisName, Intermediate: AxisName, X: Memory> =
            swiglu::spans::fp8::GateUpWeights<Hidden, Intermediate>;
        type Down<Hidden: AxisName, Intermediate: AxisName, X: Memory> =
            swiglu::spans::fp8::DownWeights<Hidden, Intermediate>;

        const REQUIREMENT: Requirement = Requirement::Declared(Declaration::Fp8(Scheme::Block));

        type Caches = (bf16,);
    }

    impl Form for Nvfp4 {
        type Stored = f4e2m1;

        type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
            linear::Affine<
                tensor::form::BlockScaled<
                    In,
                    Out,
                    f4e2m1,
                    f8e4m3,
                    nvfp4::Scales<In, Out>,
                    Early<W4A16, X>,
                    Chip,
                    V,
                    X::Tensor<f8e4m3, Chip, nvfp4::Scales<In, Out>>,
                >,
                (),
            >;

        const REQUIREMENT: Requirement = Requirement::Undeclared;

        type Caches = (bf16,);
    }

    impl Form for Nvfp4BlockScaled {
        type Stored = f4e2m1;

        type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
            tensor::form::BlockScaled<
                In,
                Out,
                f4e2m1,
                f8e4m3,
                nvfp4::Scales<In, Out>,
                Late<W4A4<X>, X>,
                Chip,
                V,
                X::Tensor<f8e4m3, Chip, nvfp4::Scales<In, Out>>,
            >;

        const REQUIREMENT: Requirement =
            Requirement::Declared(Declaration::Nvfp4(Inputs::Quantized));

        type Caches = (bf16,);
    }

    impl Form for Nvfp4BlockScaled<W4A16> {
        type Stored = f4e2m1;

        type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
            tensor::form::BlockScaled<
                In,
                Out,
                f4e2m1,
                f8e4m3,
                nvfp4::Scales<In, Out>,
                Late<W4A16, X>,
                Chip,
                V,
                X::Tensor<f8e4m3, Chip, nvfp4::Scales<In, Out>>,
            >;

        const REQUIREMENT: Requirement = Requirement::Declared(Declaration::Nvfp4(Inputs::Bf16));

        type Caches = (bf16,);
    }
}

/// The parameter form of each projection a decoder layer holds; each role defaults to the one
/// before it, so `Projections<F>` stores every projection in form `F`.
pub struct Projections<
    Query,
    Key = Query,
    Value = Key,
    Output = Value,
    Gate = Output,
    Up = Gate,
    Down = Up,
>(PhantomData<(Query, Key, Value, Output, Gate, Up, Down)>);

/// A [`Projections`]' forms by role.
pub trait Forms: 'static {
    type Query: Form;
    type Key: Form;
    type Value: Form;
    type Output: Form;
    type Gate: Form;
    type Up: Form;
    type Down: Form;

    /// What the checkpoint declares about every projection's weights; forms whose declarations
    /// differ do not compile, since a checkpoint declares one quantization.
    const REQUIREMENT: Requirement = Self::Query::REQUIREMENT
        .and(Self::Key::REQUIREMENT)
        .and(Self::Value::REQUIREMENT)
        .and(Self::Output::REQUIREMENT)
        .and(Self::Gate::REQUIREMENT)
        .and(Self::Up::REQUIREMENT)
        .and(Self::Down::REQUIREMENT);
}

impl<Query: Form, Key: Form, Value: Form, Output: Form, Gate: Form, U: Form, Down: Form> Forms
    for Projections<Query, Key, Value, Output, Gate, U, Down>
{
    type Query = Query;
    type Key = Key;
    type Value = Value;
    type Output = Output;
    type Gate = Gate;
    type Up = U;
    type Down = Down;
}

/// `Hidden` rows projected in form `F` to `Width` features of scalar `D`, each chip of placement
/// `P` holding its part of them, read as `View`.
pub type Heads<F, Hidden, Width, View, D = bf16, P = Parallel<1>> = Sequential<
    Projection<F, Hidden, <P as Placement>::Local<Width>, D, <P as Placement>::Chip>,
    Reshape<View>,
>;

/// `Count` heads of `Head` coordinates, as the projection of [`Heads`] reads its features.
pub type HeadView<Count, Head> = m![Count, Head];

/// [`Heads`] of `Count` heads of `Head` coordinates followed by their norm.
pub type NormalizedHeads<F, Hidden, Width, Count, Head, Norm> =
    Then<Heads<F, Hidden, Width, HeadView<Count, Head>>, Norm>;

/// The op that positions group `A`'s queries and keys of scalar `D` on chips `Chips`.
pub type Position<A, D = bf16, Chips = Chip> =
    <<A as Attention>::Rotated as Rotation>::Position<<A as Attention>::Head, D, Chips>;

/// Grouped-query attention of group `A` over each chip's heads of placement `P`, its scores
/// multiplied by `SCALE` (FP32 bits), over the windows of history a call pages.
pub type Attend<A, const SCALE: usize, P = Parallel<1>> = ops::attention::Attention<
    PhantomData<(
        <P as Placement>::Local<<A as Attention>::Heads>,
        <A as Attention>::Groups,
        <A as Attention>::Head,
    )>,
    F32<SCALE>,
    { Strategy::Windows },
>;

/// Group `A`'s attended heads projected back to `Hidden` rows in `F`'s output form, the chips of
/// placement `P` summing their parts.
pub type Output<F, Hidden, A, P = Parallel<1>> = RowParallel<
    Projection<
        <F as Forms>::Output,
        <P as Placement>::Local<<A as Attention>::QueryWidth>,
        Hidden,
        bf16,
        <P as Placement>::Chip,
    >,
    P,
>;

/// The `Activation` of the gate projection times the up projection, projected back to `Hidden`;
/// each chip of placement `P` holds its part of the intermediate features.
pub type FeedForward<F, Hidden, Intermediate, Activation, P = Parallel<1>> = Glu<
    Projection<
        <F as Forms>::Gate,
        Hidden,
        <P as Placement>::Local<Intermediate>,
        bf16,
        <P as Placement>::Chip,
    >,
    Activation,
    Projection<
        <F as Forms>::Up,
        Hidden,
        <P as Placement>::Local<Intermediate>,
        bf16,
        <P as Placement>::Chip,
    >,
    Multiply,
    RowParallel<
        Projection<
            <F as Forms>::Down,
            <P as Placement>::Local<Intermediate>,
            Hidden,
            bf16,
            <P as Placement>::Chip,
        >,
        P,
    >,
>;

#[cfg(test)]
mod tests {
    use super::*;

    /// A group binds the frequencies of the spectrum its rotation holds, and refuses a checkpoint
    /// whose layers rotate otherwise.
    #[test]
    fn binds_its_own_rotation() {
        axes![Rotated = 64];
        let spectrum = || config::parse::<Spectrum>(&serde_json::json!({"rope_theta": 1e4})).ok();
        let frequencies = Rotated::spectrum::<Rotated>(spectrum()).unwrap();
        assert_eq!(
            frequencies,
            rotary::PairFrequencies::unscaled::<Rotated>(1e4).unwrap()
        );
        let refusal = Rotated::spectrum::<Rotated>(None).unwrap_err();
        assert_eq!(
            (refusal.kind(), refusal.to_string()),
            (
                io::ErrorKind::Unsupported,
                "found layers that rotate nothing, compiled 64 rotated".into()
            )
        );
    }

    /// A layer requires the declaration its quantized projections read, which covers those that
    /// read none, and no declaration where none reads one.
    const _: () = {
        use crate::tensor::form::nvfp4::W4A16;
        use form::{Affine, BlockScaled, ChannelScaled, Nvfp4, Nvfp4BlockScaled};
        assert!(matches!(
            <Projections<Affine, Affine, Affine, Affine, Affine> as Forms>::REQUIREMENT,
            Requirement::Unquantized
        ));
        assert!(matches!(
            <Projections<ChannelScaled, ChannelScaled, ChannelScaled, ChannelScaled, Nvfp4> as Forms>::REQUIREMENT,
            Requirement::Undeclared
        ));
        assert!(matches!(
            <Projections<Affine, Affine, Affine, Affine, Nvfp4BlockScaled> as Forms>::REQUIREMENT,
            Requirement::Declared(Declaration::Nvfp4(Inputs::Quantized))
        ));
        assert!(matches!(
            <Projections<Affine, Affine, Affine, Affine, Nvfp4BlockScaled<W4A16>> as Forms>::REQUIREMENT,
            Requirement::Declared(Declaration::Nvfp4(Inputs::Bf16))
        ));
        assert!(matches!(
            <Projections<ChannelScaled, BlockScaled, BlockScaled, Affine, Affine> as Forms>::REQUIREMENT,
            Requirement::Declared(Declaration::Fp8(config::Scheme::Block))
        ));
    };

    /// An absent or null `quantization_config` declares BF16 weights, which BF16 projections read as
    /// stored and quantized ones only quantized at load; a declared one admits only its own forms.
    #[test]
    fn admits_absent_and_null_declarations_alike() {
        use crate::tensor::form::fp8::Granularity;
        use serde_json::json;
        let directory = tempfile::tempdir().unwrap();
        let source = |declared: Option<serde_json::Value>| {
            let mut config = json!({});
            if let Some(declared) = declared {
                config["quantization_config"] = declared;
            }
            Checkpoint::new(directory.path(), Some(config)).unwrap()
        };
        let refused = |result: Result<(), LoadError>| match result {
            Err(LoadError::Unsupported { name, .. }) => name,
            other => panic!("{other:?} is not `Unsupported`"),
        };
        for stored in [source(None), source(Some(serde_json::Value::Null))] {
            let quantized = stored.quantized(Granularity::Channel).unwrap();
            assert!(Requirement::Unquantized.admits(&stored).is_ok());
            assert!(Requirement::Undeclared.admits(&quantized).is_ok());
            assert_eq!(
                refused(Requirement::Unquantized.admits(&quantized)),
                "quantization"
            );
            assert_eq!(
                refused(Requirement::Undeclared.admits(&stored)),
                "quantization"
            );
        }
        let blocks = source(Some(json!({
            "quant_method": "fp8", "activation_scheme": "dynamic",
            "weight_block_size": [128, 128], "ignored_layers": ["lm_head"],
        })));
        let declared = "config.json:quantization_config";
        for requirement in [Requirement::Unquantized, Requirement::Undeclared] {
            assert_eq!(refused(requirement.admits(&blocks)), declared);
        }
        let block = Requirement::Declared(Declaration::Fp8(config::Scheme::Block));
        assert!(block.admits(&blocks).is_ok());
        match blocks.quantized(Granularity::Channel).err() {
            Some(LoadError::InvalidData { name, .. }) => assert_eq!(name, declared),
            other => panic!("{other:?} is not `InvalidData`"),
        }
        match Requirement::Unquantized.admits(&source(Some(json!(5)))) {
            Err(LoadError::InvalidData { name, .. }) => assert_eq!(name, declared),
            other => panic!("{other:?} is not `InvalidData`"),
        }
    }
}
