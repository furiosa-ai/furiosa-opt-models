//! A compiled decoder's geometry and cache support.
//!
//! - A decoder's sizes are type data: a [`Geometry`] holds the model axes and the [`Group`] of
//!   each attention group, and [`Projections`] holds each projection's parameter [`Form`].
//! - [`Model`] declares a decoder of one attention group; a family's configuration names the
//!   declaration its checkpoint compiles to.
//! - A group's [`Rotation`] is the coordinates its positions turn: the leading `R` of each head
//!   for an axis `R`.
//! - [`attention`] holds the attention decoder families declared by their Hugging Face classes.

use crate::Sequential;
use crate::load::{Checkpoint, LoadError};
use crate::models::Family;
use crate::models::config::{self, scaling::Spectrum};
use crate::modules::attention::key_value;
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
use furiosa_opt_std::prelude::*;
use std::fmt::Debug;
use std::{io, marker::PhantomData};

pub mod attention;
pub mod hybrid;
pub mod launch;
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

/// One attention group: `Heads` key/value heads of `Head`, `Groups` query heads each, both
/// projection widths, the heads' [`Rotation`], the `Windows` of cache blocks a row reads and the
/// element its cache stores each key and value as.
pub struct Group<Heads, Head, Groups, QueryWidth, KvWidth, Rotated, Windows, Cache = bf16>(
    PhantomData<(
        Heads,
        Head,
        Groups,
        QueryWidth,
        KvWidth,
        Rotated,
        Windows,
        Cache,
    )>,
);

/// A [`Group`]'s axes by role, and its cache's element.
pub trait Attention: 'static {
    type Heads: AxisName;
    type Head: AxisName;
    type Groups: AxisName;
    type QueryWidth: AxisName;
    type KvWidth: AxisName;
    type Rotated: Rotation;
    type Windows: AxisName;
    type Cache: Cache = bf16;

    /// Whether the widths are the head products the attention kernels assume.
    const CONSISTENT: bool = Self::Groups::SIZE * Self::Heads::SIZE * Self::Head::SIZE
        == Self::QueryWidth::SIZE
        && Self::Heads::SIZE * Self::Head::SIZE == Self::KvWidth::SIZE
        && Self::Rotated::COORDINATES <= Self::Head::SIZE
        && Self::Windows::SIZE > 0;
}

impl<Heads, Head, Groups, QueryWidth, KvWidth, Rotated, Windows, C> Attention
    for Group<Heads, Head, Groups, QueryWidth, KvWidth, Rotated, Windows, C>
where
    Heads: AxisName,
    Head: AxisName,
    Groups: AxisName,
    QueryWidth: AxisName,
    KvWidth: AxisName,
    Rotated: Rotation,
    Windows: AxisName,
    C: Cache,
{
    type Heads = Heads;
    type Head = Head;
    type Groups = Groups;
    type QueryWidth = QueryWidth;
    type KvWidth = KvWidth;
    type Rotated = Rotated;
    type Windows = Windows;
    type Cache = C;
}

/// How a cache stores each key and value: as BF16, or as an E4M3 code under its layer's static
/// scales; the op that writes a call's rows into its blocks, holding its leaves in memory `X`, and
/// what that op lands once a layer for itself and the attention that reads the blocks.
pub trait Cache: MaterializableScalar + FetchCast<f32> + 'static {
    /// The element as a checkpoint's tensors name it.
    const DTYPE: safetensors::Dtype;
    type Write<X: Memory>;
    type Landed: Default;
}

/// The rows as they are, read as stored.
impl Cache for bf16 {
    const DTYPE: safetensors::Dtype = safetensors::Dtype::BF16;
    type Write<X: Memory> = ops::index::scatter::Scatter;
    type Landed = ();
}

/// The rows divided by the layer's scales and rounded, read back times them.
impl Cache for f8e4m3 {
    const DTYPE: safetensors::Dtype = safetensors::Dtype::F8_E4M3;
    type Write<X: Memory> = ops::index::scatter::Scaled<X>;
    type Landed = crate::tensor::cache::Landed;
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

/// What a checkpoint declares about the weights a decoder layer's projections store: the
/// declaration each role's projection reads, and whether a load may quantize BF16 weights for
/// them instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requirement {
    /// The declaration the projection of each role reads, in [`ROLES`] order; none for BF16.
    declarations: [Option<Declaration>; ROLES.len()],
    /// Whether every quantized projection also takes the E4M3 weights a load quantizes from BF16.
    online: bool,
}

/// Where a decoder layer keeps the projection of each role, in [`Forms`] order, as Hugging Face
/// names it.
const ROLES: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

impl Requirement {
    /// BF16 weights, which no quantization declares.
    pub const UNQUANTIZED: Self = Self {
        declarations: [None; ROLES.len()],
        online: false,
    };

    /// Quantized weights that the checkpoint's `quantization_config` declares as `declaration`.
    pub const fn declared(declaration: Declaration) -> Self {
        Self {
            declarations: [Some(declaration); ROLES.len()],
            online: false,
        }
    }

    /// Quantized weights that the checkpoint declares as `declaration`, or that a load quantizes
    /// from BF16 ones.
    pub const fn quantizable(declaration: Declaration) -> Self {
        Self {
            online: true,
            ..Self::declared(declaration)
        }
    }

    /// What a checkpoint declares for a layer whose projection of each role is in the form whose
    /// requirement `forms` holds at the role's place: that form's declaration, and the weights a
    /// load quantizes when every quantized role's form takes them.
    pub const fn roles(forms: [Self; ROLES.len()]) -> Self {
        let mut declarations = [None; ROLES.len()];
        let (mut quantized, mut online) = (false, true);
        let mut role = 0;
        while role < ROLES.len() {
            declarations[role] = forms[role].declarations[role];
            if declarations[role].is_some() {
                quantized = true;
                online &= forms[role].online;
            }
            role += 1;
        }
        Self {
            declarations,
            online: quantized && online,
        }
    }

    /// Whether the weights are quantized.
    pub fn quantized(self) -> bool {
        self.declarations.iter().any(Option::is_some)
    }

    /// Whether a load may quantize the BF16 weights of every quantized projection.
    pub const fn online(self) -> bool {
        self.online
    }

    /// The declaration the projections read, when they read one alone.
    pub fn declaration(self) -> Option<Declaration> {
        let mut read = self.declarations.into_iter().flatten();
        let first = read.next()?;
        read.all(|other| other == first).then_some(first)
    }

    /// Refuses `source` unless it holds the weights these projections read in the decoder layers
    /// it keeps at `layers`: BF16 ones if it declares no quantization, quantized ones only if a
    /// load quantizes them, else each projection in the scheme of its form by the targets naming
    /// it, every target naming one.
    pub fn admits(self, source: &Checkpoint, layers: &[String]) -> Result<(), LoadError> {
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
        let own = (self.declarations.iter().flatten()).fold(Vec::new(), |mut read, declaration| {
            let name = format!("{declaration:?}");
            if !read.contains(&name) {
                read.push(name);
            }
            read
        });
        let own = own.join(" and ");
        let held = match (self.quantized(), self.online) {
            (false, _) => "BF16 projections".to_owned(),
            (true, false) => format!("{own} projections"),
            (true, true) => format!("{own} projections, or ones a load quantizes from BF16"),
        };
        match (declared, source.quantization()) {
            (None, None) if !self.quantized() => Ok(()),
            (None, Some(_)) if self.online => Ok(()),
            (None, None) if self.online => refused(
                "quantization",
                format!("the checkpoint is read as stored, the line holds {held}"),
            ),
            (None, None) => refused(DECLARED, format!("missing, the line holds {held}")),
            (None, Some(_)) => refused(
                "quantization",
                format!("a load quantizes the checkpoint, the line holds {held}"),
            ),
            // A declared checkpoint loads its own weights: no load quantizes it.
            (Some(declared), _) => match self.mismatch(&declared, layers) {
                None => Ok(()),
                Some(reason) => refused(DECLARED, reason),
            },
        }
    }

    /// How `declared` departs from these projections in the decoder layers at `layers`: the first
    /// projection it quantizes otherwise than its form reads, else a target quantizing none.
    fn mismatch(self, declared: &config::Quantization, layers: &[String]) -> Option<String> {
        let modules: Vec<_> = (layers.iter())
            .flat_map(|layer| {
                let modules = ROLES.map(|role| format!("{layer}.{role}"));
                modules.into_iter().zip(self.declarations)
            })
            .collect();
        let named = |scheme: &config::Scheme| format!("{scheme:?}");
        let projection = modules.iter().find_map(|(module, declaration)| {
            let found = declared.schemes(module);
            let own = declaration.map(Declaration::scheme);
            (found != own.as_slice()).then(|| {
                let found = match found.as_slice() {
                    [] => "unquantized".to_owned(),
                    found => found.iter().map(named).collect::<Vec<_>>().join(" and "),
                };
                let own = own.as_ref().map_or("BF16".to_owned(), named);
                format!("{module}: {found}, the line reads {own}")
            })
        });
        let idle = || {
            let quantizes = |target| modules.iter().any(|(at, _)| declared.quantizes(target, at));
            let (target, _) = (declared.targets.iter()).find(|(target, _)| !quantizes(target))?;
            Some(format!(
                "target {target} quantizes no projection the line holds"
            ))
        };
        projection.or_else(idle)
    }
}

/// A checkpoint's `quantization_config`, by the projection forms that load it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Declaration {
    /// FP8 weights and activations scaled as the scheme says.
    Fp8(config::Scheme),
    /// NVFP4 weights and inputs under static tensor scales; a projection's form type says whether
    /// it reads its inputs quantized or as BF16.
    Nvfp4,
}

impl Declaration {
    /// The scheme a checkpoint declares for the weights this declaration reads.
    const fn scheme(self) -> config::Scheme {
        match self {
            Self::Fp8(scheme) => scheme,
            Self::Nvfp4 => config::Scheme::Nvfp4,
        }
    }
}

/// The projection parameter forms a checkpoint can store.
pub mod form {
    use super::{Declaration, Form, Memory, PhantomData, Projection, Requirement, fp8, linear};
    use crate::models::config::Scheme;
    use crate::ops::linear::{output, query};
    use crate::ops::swiglu;
    use crate::tensor;
    use crate::tensor::form::fp8::Block;
    use crate::tensor::form::fp8::Scales;
    use crate::tensor::form::nvfp4::{self, Early, Features, Late, Summed, W4A4, W4A16};
    use crate::tensor::{parallel::Chip, scalar};
    use furiosa_opt_std::prelude::*;

    /// BF16 weights.
    pub struct Affine;

    /// FP8 weights with one FP32 scale per output channel, as a load quantizes BF16 ones or a
    /// checkpoint declares them; inputs stay BF16. HBM holds the codes as the contraction's slices
    /// hold them, a load lays them out so.
    pub struct ChannelScaled;

    /// How HBM holds [`ChannelScaled`]'s codes.
    pub mod channel {
        pub use crate::ops::linear::fp8::Codes;
    }

    /// [`ChannelScaled`]'s FP8 weights, whose inputs quantize to E4M3 at run time under one scale
    /// a row, as a checkpoint declaring per-token dynamic activations states them.
    pub struct TokenScaled;

    /// FP8 weights and activations with static per-tensor scales.
    pub struct TensorScaled;

    /// FP8 weights with one scale per 128 x 128 block, and activations that quantize per 128
    /// inputs at run time.
    pub struct BlockScaled;

    /// FP4 weights with per-16 FP8 block scales and one global scale, as a compressed-tensors
    /// checkpoint stores them; inputs stay BF16.
    pub struct Nvfp4;

    /// [`Nvfp4`]'s weights under the experimental summed contract: a row's FP32 sum divides by the
    /// global scale, so no coefficient rounds to BF16. Opt in by naming it in place of [`Nvfp4`].
    /// A launch's projected rows end where `P` places them, as [`Summed`] reads it. HBM holds the
    /// codes and block scales as the slices that sum them hold them, each slice's block scales
    /// followed by the global scale's packet ([`summed::lead`]), so one transfer lands both, and
    /// the contract's global scale as [`Copies`](crate::tensor::scalar::Copies); a load lays them
    /// out so.
    pub struct Nvfp4Summed<P = Features>(PhantomData<P>);

    /// How HBM holds [`Nvfp4Summed`]'s codes and block scales.
    pub mod summed {
        pub use crate::ops::linear::nvfp4::summed::rows::{Blocks, Codes, Ownership, lead};
    }

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

        const REQUIREMENT: Requirement = Requirement::UNQUANTIZED;

        type Caches = (bf16,);
    }

    impl Form for ChannelScaled {
        type Stored = f8e4m3;

        type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
            fp8::ChannelScaled<
                In,
                Out,
                X::Tensor<f8e4m3, Chip, channel::Codes<In, Out>>,
                X::Tensor<f32, Chip, m![Out, 1 # 8]>,
            >;

        type Query<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> =
            query::matrix::ChannelScaled<Hidden, Heads, Head, X>;
        type Key<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> =
            (Self::Query<Hidden, Heads, Head, X>, ());
        type Value<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> =
            Self::Key<Hidden, Heads, Head, X>;
        type Output<Queries: AxisName, Hidden: AxisName, X: Memory> =
            output::matrix::ChannelScaled<Queries, Hidden, X>;
        type GateUp<Hidden: AxisName, Intermediate: AxisName, X: Memory> = linear::Linear<
            Hidden,
            swiglu::GateUp<Intermediate>,
            Chip,
            Projection<Self, Hidden, swiglu::GateUp<Intermediate>, bf16, Chip, X>,
        >;
        type Down<Hidden: AxisName, Intermediate: AxisName, X: Memory> = linear::Linear<
            Intermediate,
            Hidden,
            Chip,
            Projection<Self, Intermediate, Hidden, bf16, Chip, X>,
        >;

        const REQUIREMENT: Requirement =
            Requirement::quantizable(Declaration::Fp8(Scheme::Channel));

        type Caches = (bf16,);
    }

    impl Form for TokenScaled {
        type Stored = f8e4m3;

        type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
            fp8::TokenScaled<
                In,
                Out,
                X::Tensor<f8e4m3, Chip, channel::Codes<In, Out>>,
                X::Tensor<f32, Chip, m![Out, 1 # 8]>,
            >;

        const REQUIREMENT: Requirement = Requirement::declared(Declaration::Fp8(Scheme::Channel));

        type Caches = (bf16,);
    }

    impl Form for TensorScaled {
        type Stored = f8e4m3;

        type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
            fp8::TensorScaled<In, Out, D, Chips, V, X::Tensor<f32, Chips, m![1 # 8]>>;

        const REQUIREMENT: Requirement = Requirement::declared(Declaration::Fp8(Scheme::Tensor));

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

        const REQUIREMENT: Requirement = Requirement::declared(Declaration::Fp8(Scheme::Block));

        type Caches = (bf16,);
    }

    impl Form for Nvfp4 {
        type Stored = f4e2m1;

        type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
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
            >;

        const REQUIREMENT: Requirement = Requirement::declared(Declaration::Nvfp4);

        type Caches = (bf16,);
    }

    impl<P: summed::Ownership + 'static> Form for Nvfp4Summed<P> {
        type Stored = f4e2m1;

        type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
            tensor::form::BlockScaled<
                In,
                Out,
                f4e2m1,
                f8e4m3,
                nvfp4::Scales<In, Out>,
                Summed<W4A16, X, P, scalar::Copies>,
                Chip,
                X::Tensor<f4e2m1, Chip, summed::Codes<In, Out, P>>,
                X::Tensor<f8e4m3, Chip, summed::Blocks<In, Out, P>>,
            >;

        const REQUIREMENT: Requirement = Requirement::declared(Declaration::Nvfp4);

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

        const REQUIREMENT: Requirement = Requirement::declared(Declaration::Nvfp4);

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

        const REQUIREMENT: Requirement = Requirement::declared(Declaration::Nvfp4);

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

/// The forms of a decoder layer's projections by role: a [`Projections`], or one [`Form`] for
/// every role.
pub trait Forms: 'static {
    type Query: Form;
    type Key: Form;
    type Value: Form;
    type Output: Form;
    type Gate: Form;
    type Up: Form;
    type Down: Form;

    /// What the checkpoint declares about every projection's weights: the declaration of each
    /// role's form, as a mixed-precision checkpoint declares one for each module group.
    const REQUIREMENT: Requirement = Requirement::roles([
        <Self::Query as Form>::REQUIREMENT,
        <Self::Key as Form>::REQUIREMENT,
        <Self::Value as Form>::REQUIREMENT,
        <Self::Output as Form>::REQUIREMENT,
        <Self::Gate as Form>::REQUIREMENT,
        <Self::Up as Form>::REQUIREMENT,
        <Self::Down as Form>::REQUIREMENT,
    ]);
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

/// One form stores every projection.
impl<F: Form> Forms for F {
    type Query = F;
    type Key = F;
    type Value = F;
    type Output = F;
    type Gate = F;
    type Up = F;
    type Down = F;
}

/// `Hidden` rows projected in form `F` to `Width` features of scalar `D`, each chip of placement
/// `P` holding its part of them, read as `View`; its leaves are in memory `X`.
pub type Heads<F, Hidden, Width, View, D = bf16, P = Parallel<1>, X = Hbm> = Sequential<
    Projection<F, Hidden, <P as Placement>::Local<Width>, D, <P as Placement>::Chip, X>,
    Reshape<View>,
>;

/// `Count` heads of `Head` coordinates, as the projection of [`Heads`] reads its features.
pub type HeadView<Count, Head> = m![Count, Head];

/// The op that positions group `A`'s queries and keys of scalar `D` on chips `Chips`.
pub type Position<A, D = bf16, Chips = Chip> =
    <<A as Attention>::Rotated as Rotation>::Position<<A as Attention>::Head, D, Chips>;

/// Group `A`'s constants that prepare a call of `R` rows over its cache: the rows' rotations,
/// write slots and paged history reads, bound to the group's rotary spectrum and window; in memory
/// `X`.
pub type Paging<A, R, X = Hbm> = key_value::Preparation<
    R,
    key_value::Paging<
        <A as Attention>::Heads,
        <A as Attention>::Head,
        R,
        <A as Attention>::Windows,
        <A as Attention>::Cache,
        X,
        Chip,
        <A as Attention>::Rotated,
        A,
    >,
    Chip,
    X,
>;

/// The op that writes a call's keys and values into group `A`'s cache, its leaves in memory `X`.
pub type Write<A, X = Hbm> = <<A as Attention>::Cache as Cache>::Write<X>;

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
/// placement `P` summing their parts; its leaves are in memory `X`.
pub type Output<F, Hidden, A, P = Parallel<1>, X = Hbm> = RowParallel<
    Projection<
        <F as Forms>::Output,
        <P as Placement>::Local<<A as Attention>::QueryWidth>,
        Hidden,
        bf16,
        <P as Placement>::Chip,
        X,
    >,
    P,
>;

/// The `Activation` of the gate projection times the up projection, projected back to `Hidden`;
/// each chip of placement `P` holds its part of the intermediate features, in memory `X`.
pub type FeedForward<F, Hidden, Intermediate, Activation, P = Parallel<1>, X = Hbm> = Glu<
    Projection<
        <F as Forms>::Gate,
        Hidden,
        <P as Placement>::Local<Intermediate>,
        bf16,
        <P as Placement>::Chip,
        X,
    >,
    Activation,
    Projection<
        <F as Forms>::Up,
        Hidden,
        <P as Placement>::Local<Intermediate>,
        bf16,
        <P as Placement>::Chip,
        X,
    >,
    Multiply,
    RowParallel<
        Projection<
            <F as Forms>::Down,
            <P as Placement>::Local<Intermediate>,
            Hidden,
            bf16,
            <P as Placement>::Chip,
            X,
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

    /// A layer reads the declaration of each role's form, one alone only when every quantized role
    /// reads it, and takes the weights a load quantizes only when each quantized projection does;
    /// BF16 projections require none, and one form of every role requires what the form does.
    #[test]
    fn requires_each_declaration_its_forms_read() {
        use crate::tensor::form::nvfp4::W4A16;
        use form::{Affine, BlockScaled, ChannelScaled, Nvfp4, Nvfp4BlockScaled, Nvfp4Summed};
        let channel = Declaration::Fp8(config::Scheme::Channel);
        let nvfp4 = Declaration::Nvfp4;
        type Mixed<F> = Projections<ChannelScaled, ChannelScaled, ChannelScaled, ChannelScaled, F>;
        type Mlp<F> = Projections<Affine, Affine, Affine, Affine, F>;
        type Blocks = Projections<ChannelScaled, BlockScaled, BlockScaled, Affine, Affine>;
        let read = |found: Requirement| (found.quantized(), found.declaration(), found.online());
        for (found, expected) in [
            (<Affine as Forms>::REQUIREMENT, (false, None, false)),
            (
                <ChannelScaled as Forms>::REQUIREMENT,
                (true, Some(channel), true),
            ),
            (
                <Mixed<Affine> as Forms>::REQUIREMENT,
                (true, Some(channel), true),
            ),
            (<Mixed<Nvfp4> as Forms>::REQUIREMENT, (true, None, false)),
            (
                <Mixed<Nvfp4Summed> as Forms>::REQUIREMENT,
                (true, None, false),
            ),
            (
                <Mlp<Nvfp4BlockScaled> as Forms>::REQUIREMENT,
                (true, Some(nvfp4), false),
            ),
            (
                <Mlp<Nvfp4BlockScaled<W4A16>> as Forms>::REQUIREMENT,
                (true, Some(nvfp4), false),
            ),
            (
                <Projections<Nvfp4BlockScaled, Nvfp4BlockScaled<W4A16>> as Forms>::REQUIREMENT,
                (true, Some(nvfp4), false),
            ),
            (<Blocks as Forms>::REQUIREMENT, (true, None, false)),
        ] {
            assert_eq!(read(found), expected);
        }
        assert_eq!(
            <Projections<Affine> as Forms>::REQUIREMENT,
            Requirement::UNQUANTIZED
        );
        assert_eq!(
            <ChannelScaled as Forms>::REQUIREMENT,
            Requirement::quantizable(channel)
        );
    }

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
        let (bf16, channels) = (
            Requirement::UNQUANTIZED,
            Requirement::quantizable(Declaration::Fp8(config::Scheme::Channel)),
        );
        let layers = ["model.layers.0".to_owned()];
        for stored in [source(None), source(Some(serde_json::Value::Null))] {
            let quantized = stored.quantized(Granularity::Channel).unwrap();
            assert!(bf16.admits(&stored, &layers).is_ok());
            assert!(channels.admits(&quantized, &layers).is_ok());
            assert_eq!(refused(bf16.admits(&quantized, &layers)), "quantization");
            assert_eq!(refused(channels.admits(&stored, &layers)), "quantization");
        }
        let blocks = source(Some(json!({
            "quant_method": "fp8", "activation_scheme": "dynamic",
            "weight_block_size": [128, 128], "ignored_layers": ["lm_head"],
        })));
        let declared = "config.json:quantization_config";
        for requirement in [bf16, channels] {
            assert_eq!(refused(requirement.admits(&blocks, &layers)), declared);
        }
        let block = Requirement::declared(Declaration::Fp8(config::Scheme::Block));
        assert!(block.admits(&blocks, &layers).is_ok());
        match blocks.quantized(Granularity::Channel).err() {
            Some(LoadError::InvalidData { name, .. }) => assert_eq!(name, declared),
            other => panic!("{other:?} is not `InvalidData`"),
        }
        match bf16.admits(&source(Some(json!(5))), &layers) {
            Err(LoadError::InvalidData { name, .. }) => assert_eq!(name, declared),
            other => panic!("{other:?} is not `InvalidData`"),
        }
    }

    /// An NVFP4 line refuses a checkpoint that declares no quantization, naming its scheme alone.
    #[test]
    fn refuses_an_undeclared_checkpoint_by_its_scheme() {
        let directory = tempfile::tempdir().unwrap();
        let source = Checkpoint::new(directory.path(), Some(serde_json::json!({}))).unwrap();
        let layers = ["model.layers.0".to_owned()];
        match Requirement::declared(Declaration::Nvfp4).admits(&source, &layers) {
            Err(LoadError::Unsupported { name, reason }) => assert_eq!(
                (name.as_str(), reason.as_str()),
                (
                    "config.json:quantization_config",
                    "missing, the line holds Nvfp4 projections"
                )
            ),
            other => panic!("{other:?} is not `Unsupported`"),
        }
    }
}
