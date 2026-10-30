//! Loading weight trees: one trait for every direction a tree is loaded in.
//!
//! - A tree is built from a [`Checkpoint`] into HBM, updated in place from one (an [`Update`]
//!   gathers the chunks), borrowed for a call, or written in place from a value of its own type.
//!   Each direction is an impl of [`Load`]; `#[derive(Load)]` writes them field-wise, and every
//!   one fails with [`LoadError`].
//! - Names: a field's checkpoint segment is its name or `rename`; array elements and the layers of
//!   a stack are "0".."N-1", a stack's leaves each filling one backing of every layer; a
//!   tuple field or element adds no checkpoint segment; [`Named`] names a generic child; a child
//!   without weights (`()`, `PhantomData`) reads nothing. Tree paths use field names and tuple
//!   indices.
//! - Strict: the outermost checkpoint traversal fails with every name no rule read and no `ignore`
//!   prefix covers ([`LoadError::Unexpected`]); absent names are reported together
//!   ([`LoadError::Missing`]).
//! - Leaves are HBM tensors the checkpoint converts and uploads one at a time; a leaf computed or
//!   joined on host converts its host value with `Load<HostTensor<..>>`. Dropping a tree frees its
//!   storage. A tree's [`Footprint`] is the bytes its leaves take there.
//!
//! Attributes: `rename = "s"`; `config = "key"` (the field reads its config under `key`, a leaf
//! the value there, a subtree every config value it reads, as `text_config.vocab_size`);
//! `tied = "a.b"` (the field reads the subtree at `a.b` of the struct, compares a stored copy
//! under its own name, and a leaf of the target's type is the target's storage); `tied` with
//! `config = "key"` (the tie holds when `G::TIED_EMBEDDINGS` of the struct's `G: Geometry` says
//! so, and the config must agree); `from = T` (read `T`, convert with `Load<T>`); `with = f`
//! (`f(checkpoint, context)` reads the field); on the struct, `from = T`, `ignore = "p"`,
//! `config = "key"` (every field reads its config under `key`) and `tree` (only the tree
//! directions, borrow and copy, are derived: the struct's own impls read the checkpoint, as a
//! fused or computed read needs).

#![deny(clippy::arbitrary_source_item_ordering)]

pub use checkpoint::{Checkpoint, Part};
pub use context::Context;
pub use furiosa_opt_models_macros::Load;
pub use named::Named;
#[cfg(any(test, feature = "test-support"))]
pub use shapes::{Shape, Shapes};
pub use update::{Failed, Update};

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

use furiosa_opt_std::prelude::{HbmTensor, M, Scalar};

mod checkpoint;
mod context;
mod leaf;
mod named;
pub(crate) mod quantization;
mod tensor;
mod tuple;
mod update;

/// `Self` built from `from`, or, when `Self` is `&mut T`, that `T` changed in place.
pub trait Load<From>: Sized {
    fn load(context: &mut Context<'_>, from: From) -> Result<Self, LoadError>;
}

/// Why a tree does not load. The checkpoint kinds name checkpoint names (`config.json:key` for a
/// config value); the device kinds name the tree path of the leaf that failed.
#[derive(Debug)]
pub enum LoadError {
    /// Every name the tree reads that the checkpoint lacks.
    Missing(Vec<String>),
    /// Every name the checkpoint holds that no rule read and no `ignore` prefix covers.
    Unexpected(Vec<String>),
    InvalidData {
        name: String,
        reason: String,
    },
    Unsupported {
        name: String,
        reason: String,
    },
    /// The device refused or failed the transfer of the leaf at `path`.
    Device {
        path: String,
        source: furiosa_opt_std::Error,
    },
}

impl LoadError {
    /// `Missing` of `names`, each once, in the order they were first missed.
    pub fn missing(names: impl IntoIterator<Item = String>) -> Self {
        let mut seen = BTreeSet::new();
        Self::Missing(
            names
                .into_iter()
                .filter(|name| seen.insert(name.clone()))
                .collect(),
        )
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(names) => write!(
                formatter,
                "missing from the checkpoint: {}",
                names.join(", ")
            ),
            Self::Unexpected(names) => write!(formatter, "read by no rule: {}", names.join(", ")),
            Self::InvalidData { name, reason } => write!(formatter, "{name}: {reason}"),
            Self::Unsupported { name, reason } => {
                write!(formatter, "{name}: unsupported, {reason}")
            }
            Self::Device { path, source } => write!(formatter, "{path}: {source}"),
        }
    }
}

impl Error for LoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Device { source, .. } => Some(source),
            Self::Missing(_)
            | Self::Unexpected(_)
            | Self::InvalidData { .. }
            | Self::Unsupported { .. } => None,
        }
    }
}

/// The reads of one node's children, in visit order. A read that fails with an error other than
/// `Missing` ends them with it; otherwise the node reports every name its children lack together.
#[derive(Default)]
pub struct Children {
    missing: Vec<String>,
}

impl Children {
    /// The value a child's `read` holds, or `None` once the names it lacks are kept; another error
    /// is returned as it is.
    pub fn ask<T>(&mut self, read: Result<T, LoadError>) -> Result<Option<T>, LoadError> {
        match read {
            Ok(value) => Ok(Some(value)),
            Err(LoadError::Missing(names)) => {
                self.missing.extend(names);
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    /// The node built of every child's value, or `Missing` of every name the children lack.
    pub fn node<T>(self, node: Option<T>) -> Result<T, LoadError> {
        node.ok_or_else(|| LoadError::missing(self.missing))
    }
}

/// The bytes a tree's leaves take in HBM, as Hugging Face's memory footprint counts parameters and
/// buffers; `#[derive(Load)]` sums its fields', and a leaf sharing another's storage counts again.
pub trait Footprint {
    const SIZE: usize;
}

impl<D: Scalar, C: M, S: M> Footprint for HbmTensor<D, C, S> {
    const SIZE: usize = HbmTensor::<D, C, S>::size();
}

impl<T: Footprint, const N: usize> Footprint for [T; N] {
    const SIZE: usize = N * T::SIZE;
}

impl Footprint for () {
    const SIZE: usize = 0;
}

impl<T> Footprint for PhantomData<T> {
    const SIZE: usize = 0;
}

/// What one rule of an update pass read and lacked, at its tree path. A rule writes when it lacks
/// nothing; one that lacks every name it read was not fed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    pub tree: String,
    pub required: Vec<String>,
    pub absent: Vec<String>,
}

/// Array elements: element `i` under segment `"i"` in both paths.
impl<'a, T: Load<&'a Checkpoint>, const N: usize> Load<&'a Checkpoint> for [T; N] {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let built = (0..N)
                .map(|i| {
                    let element = cx.field(&i.to_string(), &i.to_string(), |cx| T::load(cx, from));
                    children.ask(element)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let built = children.node(built.into_iter().collect::<Option<Vec<T>>>())?;
            Ok(built.try_into().ok().expect("one element per index"))
        })
    }
}

impl<'t, 's, T, const N: usize> Load<(&'t mut [T; N], &'s Checkpoint)> for &'t mut [T; N]
where
    for<'u> &'u mut T: Load<(&'u mut T, &'s Checkpoint)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut [T; N], &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            into.iter_mut().enumerate().try_for_each(|(i, t)| {
                cx.field(&i.to_string(), &i.to_string(), |cx| {
                    <&mut T as Load<_>>::load(cx, (t, from))
                })
                .map(|_| ())
            })
        })?;
        Ok(into)
    }
}

impl<'a, T: Load<&'a U>, U, const N: usize> Load<&'a [U; N]> for [T; N] {
    fn load(context: &mut Context<'_>, from: &'a [U; N]) -> Result<Self, LoadError> {
        let built = from
            .iter()
            .enumerate()
            .map(|(i, u)| context.field(&i.to_string(), &i.to_string(), |cx| T::load(cx, u)))
            .collect::<Result<Vec<T>, LoadError>>()?;
        Ok(built.try_into().ok().expect("one element per element"))
    }
}

impl<'t, T, const N: usize> Load<(&'t mut [T; N], [T; N])> for &'t mut [T; N]
where
    for<'u> &'u mut T: Load<(&'u mut T, T)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut [T; N], [T; N]),
    ) -> Result<Self, LoadError> {
        into.iter_mut()
            .zip(from)
            .enumerate()
            .try_for_each(|(i, (t, value))| {
                context
                    .field(&i.to_string(), &i.to_string(), |cx| {
                        <&mut T as Load<_>>::load(cx, (t, value))
                    })
                    .map(|_| ())
            })?;
        Ok(into)
    }
}

/// A child without weights: it reads nothing and is no update target.
impl<'a> Load<&'a Checkpoint> for () {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |_| Ok(()))
    }
}

impl<'t, 's> Load<(&'t mut (), &'s Checkpoint)> for &'t mut () {
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut (), &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |_| Ok(()))?;
        Ok(into)
    }
}

impl<'a> Load<&'a ()> for () {
    fn load(_: &mut Context<'_>, _: &'a ()) -> Result<Self, LoadError> {
        Ok(())
    }
}

impl<'t> Load<(&'t mut (), ())> for &'t mut () {
    fn load(_: &mut Context<'_>, (into, ()): (&'t mut (), ())) -> Result<Self, LoadError> {
        Ok(into)
    }
}

/// A type marker, of any memory: it reads nothing and is no update target.
impl<'a, T> Load<&'a Checkpoint> for PhantomData<T> {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |_| Ok(PhantomData))
    }
}

impl<'t, 's, T> Load<(&'t mut PhantomData<T>, &'s Checkpoint)> for &'t mut PhantomData<T> {
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut PhantomData<T>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |_| Ok(()))?;
        Ok(into)
    }
}

impl<'a, T, U> Load<&'a PhantomData<U>> for PhantomData<T> {
    fn load(_: &mut Context<'_>, _: &'a PhantomData<U>) -> Result<Self, LoadError> {
        Ok(PhantomData)
    }
}

impl<'t, T> Load<(&'t mut PhantomData<T>, PhantomData<T>)> for &'t mut PhantomData<T> {
    fn load(
        _: &mut Context<'_>,
        (into, _): (&'t mut PhantomData<T>, PhantomData<T>),
    ) -> Result<Self, LoadError> {
        Ok(into)
    }
}

/// Trees checked against a checkpoint on host, for the tests of crates that read published
/// checkpoints: a leaf holds nothing, and its load checks its name, dtype and shape as an HBM
/// leaf's load does, without reading the values.
#[cfg(any(test, feature = "test-support"))]
mod shapes {
    use std::marker::PhantomData;

    use furiosa_opt_std::prelude::{HostTensor, M, MaterializableScalar, Pair, m};

    use super::{Checkpoint, Context, Load, LoadError};
    use crate::modules::generic::collection::View;
    use crate::tensor::memory::Memory;

    /// The memory whose leaves are [`Shape`]s.
    pub struct Shapes;

    impl Memory for Shapes {
        type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Shape<D, C, S>;
    }

    /// A leaf found under its name with a dtype and shape its type holds.
    pub struct Shape<D, C, S>(PhantomData<(D, C, S)>);

    /// The leaf at the current name, laid out as an HBM leaf of its chips reads it, through the
    /// tie protocol every leaf follows, so a stored tied copy is compared and consumed; strict
    /// when loaded alone.
    impl<'a, D: MaterializableScalar, C: M + 'static, S: M + 'static> Load<&'a Checkpoint>
        for Shape<D, C, S>
    {
        fn load(context: &mut Context<'_>, source: &'a Checkpoint) -> Result<Self, LoadError> {
            let shape = || Self(PhantomData);
            let check = |cx: &mut Context<'_>| {
                let checked = match Checkpoint::splits::<C, S>() {
                    true => source.check::<D, S, m![{ C }, { S }]>(cx),
                    false => source.check::<D, S, S>(cx),
                };
                checked.map(|()| shape())
            };
            context.checkpoint(source, |cx| {
                cx.leaf(source, check, |_, _| Ok((shape(), shape())))
            })
        }
    }

    /// A stack of shapes holds no value: one shape stands for every layer's.
    impl<D, C, S, const N: usize> View<N> for Shape<D, C, S> {
        type Storage = Self;
        type Layer = Self;

        fn view(_: &Self, _: usize) -> Self {
            Self(PhantomData)
        }

        fn stack(_: [Self; N], _: &mut Context<'_>) -> Self {
            Self(PhantomData)
        }
    }

    /// A leaf built on host hands its value over; this memory keeps nothing of it.
    impl<D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M> Load<HostTensor<D, E>>
        for Shape<D, C, S>
    {
        fn load(_: &mut Context<'_>, _: HostTensor<D, E>) -> Result<Self, LoadError> {
            Ok(Self(PhantomData))
        }
    }

    /// A leaf from a function of its host value, which this memory never calls: a load reads no
    /// value.
    impl<'f, D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M>
        Load<&'f (dyn Fn() -> Result<HostTensor<D, E>, LoadError> + 'f)> for Shape<D, C, S>
    {
        fn load(
            _: &mut Context<'_>,
            _: &'f (dyn Fn() -> Result<HostTensor<D, E>, LoadError> + 'f),
        ) -> Result<Self, LoadError> {
            Ok(Self(PhantomData))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::memory::{Hbm, Memory};
    use furiosa_opt_std::prelude::{Broadcast, HbmTensor, M, MaterializableScalar, bf16, m};
    use safetensors::Dtype;
    use std::marker::PhantomData;

    /// The traversals of a checkpoint the tests run without a device.
    trait Traversals {
        /// A tree loaded strictly from the whole checkpoint.
        fn build<T: for<'a> Load<&'a Checkpoint>>(&self) -> Result<T, LoadError>;

        /// `tree` updated in place from this checkpoint as one pass.
        fn update<T>(&self, tree: &mut T) -> Result<(), LoadError>
        where
            for<'t, 's> &'t mut T: Load<(&'t mut T, &'s Checkpoint)>;
    }

    impl Traversals for Checkpoint {
        fn build<T: for<'a> Load<&'a Checkpoint>>(&self) -> Result<T, LoadError> {
            T::load(&mut Context::new(None), self)
        }

        fn update<T>(&self, tree: &mut T) -> Result<(), LoadError>
        where
            for<'t, 's> &'t mut T: Load<(&'t mut T, &'s Checkpoint)>,
        {
            <&mut T as Load<_>>::load(&mut Context::new(None), (tree, self)).map(drop)
        }
    }

    /// A leaf of scalar `D` holding the bytes it read; a second owner of its storage ends in `'`.
    #[derive(Debug, Clone, PartialEq)]
    struct Leaf<D>(Vec<u8>, PhantomData<D>);

    /// The memory whose leaves are [`Leaf`]s.
    struct Bytes;

    impl Memory for Bytes {
        type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Leaf<D>;
    }

    type Tensor<X> = <X as Memory>::Tensor<bf16, m![1], m![1]>;

    impl<'a, D: 'static> Load<&'a Checkpoint> for Leaf<D> {
        fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
            let split = |_: &Context<'_>, leaf: Self| {
                let owner = Leaf([leaf.0.as_slice(), b"'"].concat(), PhantomData);
                Ok((leaf, owner))
            };
            context.checkpoint(from, |cx| cx.leaf(from, |cx| read(from, cx), split))
        }
    }

    /// An update writes the bytes read; bytes `[0xff]` are refused.
    impl<'t, 's, D: 'static> Load<(&'t mut Leaf<D>, &'s Checkpoint)> for &'t mut Leaf<D> {
        fn load(
            context: &mut Context<'_>,
            (into, from): (&'t mut Leaf<D>, &'s Checkpoint),
        ) -> Result<Self, LoadError> {
            let write = |cx: &mut Context<'_>| {
                let new = read(from, cx)?;
                match new.0.as_slice() {
                    [0xff] => Err(LoadError::InvalidData {
                        name: cx.path(),
                        reason: "refused".into(),
                    }),
                    _ => Ok(*into = new),
                }
            };
            context.checkpoint(from, |cx| {
                cx.rule(|cx| cx.overwrite::<Leaf<D>>(from, write), |_, ()| Ok(()))
            })?;
            Ok(into)
        }
    }

    impl<'t, D> Load<(&'t mut Leaf<D>, Leaf<D>)> for &'t mut Leaf<D> {
        fn load(
            _: &mut Context<'_>,
            (into, from): (&'t mut Leaf<D>, Leaf<D>),
        ) -> Result<Self, LoadError> {
            *into = from;
            Ok(into)
        }
    }

    impl<'a, D> Load<&'a Leaf<D>> for &'a Leaf<D> {
        fn load(_: &mut Context<'_>, from: &'a Leaf<D>) -> Result<Self, LoadError> {
            Ok(from)
        }
    }

    #[derive(Load)]
    struct Linear<X: Memory> {
        weight: Tensor<X>,
    }

    #[derive(Load)]
    struct Block<X: Memory> {
        attention: Linear<X>,
        bias: Tensor<X>,
    }

    #[derive(Load)]
    struct Stack<X: Memory> {
        layers: [Linear<X>; 3],
    }

    #[derive(Load)]
    struct Two<A, B>(A, B);

    #[derive(Load)]
    struct Holder<X: Memory> {
        pair: Two<Named<"a", Linear<X>>, Named<"b", Linear<X>>>,
    }

    #[derive(Load)]
    struct Renamed<X: Memory> {
        #[load(rename = "q_norm")]
        query_norm: Linear<X>,
        #[load(rename = "")]
        projection: Linear<X>,
    }

    #[derive(Load)]
    #[load(ignore = "mtp")]
    struct Ignoring<X: Memory> {
        weight: Tensor<X>,
    }

    #[derive(Load)]
    struct Norm<X: Memory> {
        #[load(config = "rms_norm_eps")]
        epsilon: Tensor<X>,
        weight: Tensor<X>,
    }

    #[derive(Load)]
    #[load(config = "text_config")]
    struct Text<X: Memory> {
        norm: Norm<X>,
        #[load(config = "vision_config")]
        vision: Norm<X>,
    }

    #[derive(Load)]
    struct Tied<X: Memory> {
        a: Linear<X>,
        b: Linear<X>,
        #[load(tied = "a")]
        c: Linear<X>,
    }

    type Wide<X> = <X as Memory>::Tensor<f32, m![1], m![1]>;

    #[derive(Load)]
    struct Widened<X: Memory> {
        weight: Wide<X>,
    }

    #[derive(Load)]
    struct Mixed<X: Memory> {
        a: Linear<X>,
        #[load(tied = "a")]
        c: Linear<X>,
        #[load(tied = "a")]
        d: Linear<X>,
        #[load(tied = "a")]
        w: Widened<X>,
    }

    /// Decides the tie of a struct generic over it, as a size's geometry does.
    trait Geometry {
        const TIED_EMBEDDINGS: bool;
    }

    #[derive(Load)]
    struct Tying;

    #[derive(Load)]
    struct Untying;

    impl Geometry for Tying {
        const TIED_EMBEDDINGS: bool = true;
    }

    impl Geometry for Untying {
        const TIED_EMBEDDINGS: bool = false;
    }

    #[derive(Load)]
    struct Head<G: Geometry, X: Memory> {
        geometry: G,
        embed: Linear<X>,
        #[load(tied = "embed", config = "tie_word_embeddings")]
        head: Linear<X>,
    }

    #[derive(Load)]
    struct Children<P, Q> {
        projection: P,
        norms: [Q; 2],
    }

    #[derive(Load)]
    struct Weightless<X: Memory> {
        weight: Tensor<X>,
        unit: (),
        marker: PhantomData<X>,
        pair: (Named<"a", Linear<X>>, ()),
    }

    /// A tree's footprint is the bytes of its HBM leaves: arrays, tuples and `Named` add theirs,
    /// weightless children none.
    const _: () = {
        let leaf = HbmTensor::<bf16, m![1], m![1]>::size();
        assert!(<Stack<Hbm> as Footprint>::SIZE == 3 * leaf);
        assert!(<Weightless<Hbm> as Footprint>::SIZE == 2 * leaf);
    };

    #[derive(Load)]
    struct Parts<X: Memory> {
        q_proj: Linear<X>,
        k_proj: Linear<X>,
    }

    #[derive(Load)]
    #[load(from = Parts<X>)]
    struct Packed<X: Memory> {
        weight: Tensor<X>,
    }

    impl Load<Parts<Bytes>> for Packed<Bytes> {
        fn load(_: &mut Context<'_>, parts: Parts<Bytes>) -> Result<Self, LoadError> {
            let weight = [parts.q_proj.weight.0, parts.k_proj.weight.0].concat();
            Ok(Packed {
                weight: Leaf(weight, PhantomData),
            })
        }
    }

    #[derive(Load)]
    struct Converted {
        #[load(with = doubled)]
        scale: Leaf<bf16>,
    }

    /// A checkpoint in memory of byte tensors, without a config.
    fn checkpoint(tensors: &[(&str, &[u8])]) -> Checkpoint {
        let mut checkpoint = Checkpoint::default();
        for (name, bytes) in tensors {
            let shape = [bytes.len()];
            checkpoint
                .insert(name, Dtype::U8, &shape, bytes.to_vec())
                .unwrap();
        }
        checkpoint
    }

    fn leaf(bytes: &[u8]) -> Leaf<bf16> {
        Leaf(bytes.to_vec(), PhantomData)
    }

    /// The names `error` lists as missing.
    fn missing(error: Option<LoadError>) -> Vec<String> {
        match error {
            Some(LoadError::Missing(names)) => names,
            other => panic!("{other:?} is not `Missing`"),
        }
    }

    /// The names `error` lists as unexpected.
    fn unexpected(error: Option<LoadError>) -> Vec<String> {
        match error {
            Some(LoadError::Unexpected(names)) => names,
            other => panic!("{other:?} is not `Unexpected`"),
        }
    }

    /// The name `error` refuses as invalid data, and why.
    fn invalid(error: Option<LoadError>) -> (String, String) {
        match error {
            Some(LoadError::InvalidData { name, reason }) => (name, reason),
            other => panic!("{other:?} is not `InvalidData`"),
        }
    }

    /// The bytes at the current name, or the text of the current config value.
    fn read<D>(from: &Checkpoint, context: &mut Context<'_>) -> Result<Leaf<D>, LoadError> {
        if let Some(key) = context.config_key().map(str::to_owned) {
            let (_, value) = context.config_value(from, &key)?;
            return Ok(Leaf(value.to_string().into_bytes(), PhantomData));
        }
        let name = context.name();
        context.read(&name);
        let view = from
            .get(&name)
            .ok_or_else(|| LoadError::Missing(vec![name.clone()]))?;
        Ok(Leaf(view.data().to_vec(), PhantomData))
    }

    /// A struct reads each field under its name, nested fields under their parents' names.
    #[test]
    fn field_names_are_checkpoint_segments() {
        let source = checkpoint(&[("attention.weight", b"w"), ("bias", b"b")]);
        let block: Block<Bytes> = source.build().unwrap();
        assert_eq!(
            (block.attention.weight, block.bias),
            (leaf(b"w"), leaf(b"b"))
        );
    }

    /// Array elements read under "0".."N-1".
    #[test]
    fn arrays_name_elements_from_zero() {
        let source = checkpoint(&[
            ("layers.0.weight", b"0"),
            ("layers.1.weight", b"1"),
            ("layers.2.weight", b"2"),
        ]);
        let stack: Stack<Bytes> = source.build().unwrap();
        assert_eq!(
            stack.layers.map(|layer| layer.weight),
            [leaf(b"0"), leaf(b"1"), leaf(b"2")]
        );
    }

    /// A tuple field adds no checkpoint segment; `Named` adds its name.
    #[test]
    fn tuples_add_no_segment_and_named_does() {
        let source = checkpoint(&[("pair.a.weight", b"a"), ("pair.b.weight", b"b")]);
        let holder: Holder<Bytes> = source.build().unwrap();
        assert_eq!(
            (holder.pair.0.0.weight, holder.pair.1.0.weight),
            (leaf(b"a"), leaf(b"b"))
        );
    }

    /// `rename` changes the checkpoint segment a field reads and updates under; an empty rename
    /// adds none.
    #[test]
    fn rename_changes_the_checkpoint_name_only() {
        let source = checkpoint(&[("q_norm.weight", b"q"), ("weight", b"p")]);
        let mut tree: Renamed<Bytes> = source.build().unwrap();
        assert_eq!(
            (
                tree.query_norm.weight.clone(),
                tree.projection.weight.clone()
            ),
            (leaf(b"q"), leaf(b"p"))
        );
        let chunk = checkpoint(&[("q_norm.weight", b"r"), ("weight", b"s")]);
        chunk.update(&mut tree).unwrap();
        assert_eq!(
            (&tree.query_norm.weight, &tree.projection.weight),
            (&leaf(b"r"), &leaf(b"s"))
        );
        let error = checkpoint(&[("query_norm.weight", b"t")])
            .update(&mut tree)
            .err();
        assert_eq!(unexpected(error), ["query_norm.weight"]);
    }

    /// Every absent name is reported together, in visit order.
    #[test]
    fn missing_lists_every_absent_name() {
        let error = checkpoint(&[]).build::<Block<Bytes>>().err();
        assert_eq!(missing(error), ["attention.weight", "bias"]);
    }

    /// A name no rule read fails the load, under any context the load is given.
    #[test]
    fn unexpected_lists_unread_names_under_any_context() {
        let source = checkpoint(&[
            ("attention.weight", b"w"),
            ("bias", b"b"),
            ("attention.extra", b"x"),
        ]);
        assert_eq!(
            unexpected(source.build::<Block<Bytes>>().err()),
            ["attention.extra"]
        );
        let bypass = Block::<Bytes>::load(&mut Context::new(None), &source).err();
        assert_eq!(unexpected(bypass), ["attention.extra"]);
        let source = checkpoint(&[("weight", b"w"), ("weight.scale", b"s")]);
        let mut context = Context::new(None);
        let root = context
            .field("weight", "weight", |cx| Leaf::<bf16>::load(cx, &source))
            .err();
        assert_eq!(
            unexpected(root),
            ["weight.scale"],
            "a leaf as the root is strict"
        );
    }

    /// `ignore` covers a prefix and the names below it, not a name it is a stem of.
    #[test]
    fn ignore_covers_a_prefix_not_a_stem() {
        let source = checkpoint(&[
            ("weight", b"w"),
            ("mtp", b"m"),
            ("mtp.0.w", b"m"),
            ("mtps.w", b"s"),
        ]);
        let error = source.build::<Ignoring<Bytes>>().err();
        assert_eq!(unexpected(error), ["mtps.w"]);
        let mut tree: Ignoring<Bytes> = checkpoint(&[("weight", b"w")]).build().unwrap();
        assert!(checkpoint(&[("mtp.0.w", b"m")]).update(&mut tree).is_ok());
    }

    fn configured(tensors: &[(&str, &[u8])], config: serde_json::Value) -> Checkpoint {
        Checkpoint {
            config,
            ..checkpoint(tensors)
        }
    }

    /// A config leaf reads its key; an absent key is missing under its config name.
    #[test]
    fn config_leaf_reads_the_key() {
        let source = configured(
            &[("weight", b"w")],
            serde_json::json!({"rms_norm_eps": 1e-6}),
        );
        let norm: Norm<Bytes> = source.build().unwrap();
        assert_eq!(norm.epsilon, leaf(b"1e-6"));
        let error = checkpoint(&[("weight", b"w")]).build::<Norm<Bytes>>().err();
        assert_eq!(missing(error), ["config.json:rms_norm_eps"]);
    }

    /// A struct's config key scopes its fields' config reads; a subtree field's key nests in it.
    #[test]
    fn config_keys_scope_the_reads_below_them() {
        let tensors = [("norm.weight", &b"w"[..]), ("vision.weight", b"v")];
        let nested = serde_json::json!({"rms_norm_eps": 0.5});
        let config =
            serde_json::json!({"text_config": {"rms_norm_eps": 1e-6, "vision_config": nested}});
        let text: Text<Bytes> = configured(&tensors, config).build().unwrap();
        assert_eq!(
            (text.norm.epsilon, text.vision.epsilon),
            (leaf(b"1e-6"), leaf(b"0.5"))
        );
        let error = checkpoint(&tensors).build::<Text<Bytes>>().err();
        assert_eq!(
            missing(error),
            [
                "config.json:text_config.rms_norm_eps",
                "config.json:text_config.vision_config.rms_norm_eps"
            ]
        );
    }

    /// A tied field reads its target and takes its storage; a stored copy under its own name is
    /// compared and consumed.
    #[test]
    fn tied_reads_the_target_and_compares_a_stored_copy() {
        let tree: Tied<Bytes> = checkpoint(&[("a.weight", b"1"), ("b.weight", b"2")])
            .build()
            .unwrap();
        assert_eq!((tree.a.weight, tree.c.weight), (leaf(b"1"), leaf(b"1'")));
        let equal = checkpoint(&[("a.weight", b"1"), ("b.weight", b"2"), ("c.weight", b"1")]);
        assert!(
            equal.build::<Tied<Bytes>>().is_ok(),
            "an equal stored copy is consumed"
        );
        let differs = checkpoint(&[("a.weight", b"1"), ("b.weight", b"2"), ("c.weight", b"5")]);
        let refused = differs.build::<Tied<Bytes>>().err();
        assert_eq!(
            invalid(refused),
            ("c.weight".into(), "tied copy differs from a.weight".into())
        );
    }

    /// A name two rules miss is listed once.
    #[test]
    fn missing_names_are_listed_once() {
        let error = checkpoint(&[]).build::<Tied<Bytes>>().err();
        assert_eq!(missing(error), ["a.weight", "b.weight"]);
    }

    /// Every tied leaf of the target's type takes the target's storage; a tied leaf of another
    /// type converts the target's tensor on its own.
    #[test]
    fn tied_leaves_of_the_target_type_share_its_storage() {
        let tree: Mixed<Bytes> = checkpoint(&[("a.weight", b"1")]).build().unwrap();
        assert_eq!(
            (tree.c.weight, tree.d.weight, tree.w.weight.0),
            (leaf(b"1'"), leaf(b"1''"), b"1".to_vec())
        );
    }

    /// An update writes a tied target once, which reaches the tied leaves of its type; a tied
    /// leaf of another type is written on its own.
    #[test]
    fn update_writes_a_tied_target_once() {
        let mut tree: Mixed<Bytes> = checkpoint(&[("a.weight", b"1")]).build().unwrap();
        checkpoint(&[("a.weight", b"7")]).update(&mut tree).unwrap();
        assert_eq!(
            (
                &tree.a.weight,
                &tree.c.weight,
                &tree.d.weight,
                &tree.w.weight.0
            ),
            (&leaf(b"7"), &leaf(b"1'"), &leaf(b"1''"), &b"7".to_vec()),
            "only the target and the leaf of another type are written"
        );
    }

    /// The geometry decides the tie; the config must agree with a boolean, and an absent one is
    /// missing.
    #[test]
    fn tying_follows_the_type_and_the_config_agrees() {
        let tied = serde_json::json!({"tie_word_embeddings": true});
        let untied = serde_json::json!({"tie_word_embeddings": false});
        let one = [("embed.weight", &b"1"[..])];
        let both = [("embed.weight", &b"1"[..]), ("head.weight", b"2")];
        let tree: Head<Tying, Bytes> = configured(&one, tied.clone()).build().unwrap();
        assert_eq!(tree.head.weight, leaf(b"1'"));
        let mut tree: Head<Untying, Bytes> = configured(&both, untied.clone()).build().unwrap();
        assert_eq!(tree.head.weight, leaf(b"2"));
        checkpoint(&[("head.weight", b"9")])
            .update(&mut tree)
            .unwrap();
        assert_eq!(
            tree.head.weight,
            leaf(b"9"),
            "an untied head updates from its own name"
        );
        let key = || "config.json:tie_word_embeddings".to_string();
        let error = configured(&one, untied).build::<Head<Tying, Bytes>>().err();
        assert_eq!(
            invalid(error),
            (key(), "false, the model type ties: true".into())
        );
        let error = configured(&both, tied)
            .build::<Head<Untying, Bytes>>()
            .err();
        assert_eq!(
            invalid(error),
            (key(), "true, the model type ties: false".into())
        );
        let spelled = serde_json::json!({"tie_word_embeddings": "yes"});
        let error = configured(&one, spelled)
            .build::<Head<Tying, Bytes>>()
            .err();
        assert_eq!(invalid(error), (key(), "\"yes\" is not a boolean".into()));
        let absent = checkpoint(&[]).build::<Head<Tying, Bytes>>().err();
        assert_eq!(
            missing(absent),
            ["embed.weight", "config.json:tie_word_embeddings"]
        );
    }

    /// A child without weights reads no name; a tuple element adds nothing to the checkpoint name
    /// it reads and updates under.
    #[test]
    fn weightless_children_read_nothing_and_tuples_name_no_segment() {
        let source = checkpoint(&[("weight", b"w"), ("pair.a.weight", b"a")]);
        let mut tree: Weightless<Bytes> = source.build().unwrap();
        let borrowed: Weightless<&Bytes> = Load::load(&mut Context::new(None), &tree).unwrap();
        assert_eq!(borrowed.pair.0.0.weight, &leaf(b"a"));
        let extra = checkpoint(&[("weight", b"w"), ("pair.a.weight", b"a"), ("unit", b"u")]);
        assert_eq!(
            unexpected(extra.build::<Weightless<Bytes>>().err()),
            ["unit"]
        );
        checkpoint(&[("pair.a.weight", b"b")])
            .update(&mut tree)
            .unwrap();
        assert_eq!(
            (tree.weight, tree.pair.0.0.weight),
            (leaf(b"w"), leaf(b"b"))
        );
    }

    /// Generic children and arrays of them borrow field-wise.
    #[test]
    fn nested_generic_children_convert() {
        type Tree<X> = Children<Linear<X>, Named<"n", Linear<X>>>;

        let source = checkpoint(&[
            ("projection.weight", b"p"),
            ("norms.0.n.weight", b"0"),
            ("norms.1.n.weight", b"1"),
        ]);
        let tree: Tree<Bytes> = source.build().unwrap();
        let borrowed: Tree<&Bytes> = Load::load(&mut Context::new(None), &tree).unwrap();
        assert_eq!(borrowed.projection.weight, &leaf(b"p"));
        assert_eq!(borrowed.norms[1].0.weight, &leaf(b"1"));
    }

    /// A write refused after an earlier one of the same chunk leaves the earlier write in place.
    #[test]
    fn failing_write_keeps_earlier_writes() {
        let mut block: Block<Bytes> = checkpoint(&[("attention.weight", b"w"), ("bias", b"b")])
            .build()
            .unwrap();
        let chunk = checkpoint(&[("attention.weight", b"v"), ("bias", &[0xff])]);
        let mut context = Context::new(None);
        let result = <&mut Block<Bytes> as Load<_>>::load(&mut context, (&mut block, &chunk));
        assert_eq!(invalid(result.err()), ("bias".into(), "refused".into()));
        assert_eq!(block.attention.weight, leaf(b"v"));
    }

    /// A `from` struct reads the parts at its own path and converts them; an update of it writes
    /// once all its parts arrived.
    #[test]
    fn from_reads_parts_then_converts() {
        let source = checkpoint(&[("q_proj.weight", b"q"), ("k_proj.weight", b"k")]);
        let mut packed: Packed<Bytes> = source.build().unwrap();
        assert_eq!(packed.weight, leaf(b"qk"));
        checkpoint(&[("q_proj.weight", b"Q")])
            .update(&mut packed)
            .unwrap();
        assert_eq!(packed.weight, leaf(b"qk"), "a waiting rule writes nothing");
        let chunk = checkpoint(&[("q_proj.weight", b"Q"), ("k_proj.weight", b"K")]);
        chunk.update(&mut packed).unwrap();
        assert_eq!(packed.weight, leaf(b"QK"));
    }

    fn doubled(from: &Checkpoint, context: &mut Context<'_>) -> Result<Leaf<bf16>, LoadError> {
        let name = context.name();
        context.read(&name);
        let view = from
            .get(&name)
            .ok_or_else(|| LoadError::Missing(vec![name.clone()]))?;
        Ok(Leaf(view.data().repeat(2), PhantomData))
    }

    /// A `with` field reads through its function, in a load and as one rule of an update.
    #[test]
    fn with_reads_through_its_function() {
        let mut tree: Converted = checkpoint(&[("scale", b"s")]).build().unwrap();
        assert_eq!(tree.scale, leaf(b"ss"));
        checkpoint(&[("scale", b"t")]).update(&mut tree).unwrap();
        assert_eq!(tree.scale, leaf(b"tt"));
    }
}
