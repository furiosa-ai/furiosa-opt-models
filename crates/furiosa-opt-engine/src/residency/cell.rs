//! Typed handles to retained host parameters, loaded through the public named-tree rules.
//!
//! Cloning a handle shares its parameter. Device copies are only lent under residency admission;
//! loading a cell never produces an independently owned HBM tensor.

use super::parameter::{Destination, Parameter};
use super::source::Source;
use furiosa_opt_models::load::{Checkpoint, Context, Load, LoadError};
use furiosa_opt_models::tensor::memory::Memory;
use furiosa_opt_std::prelude::{HostTensor, M, MaterializableScalar, Pair, m};
use std::{io, marker::PhantomData, sync::Arc};

/// The storage of a tree whose typed leaves share movable parameters.
pub(crate) struct Cells;

impl Memory for Cells {
    type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Cell<D, C, S>;
}

/// One typed occurrence of a parameter; every clone refers to the same retained source and copy.
pub(crate) struct Cell<D, C, S> {
    parameter: Arc<Parameter>,
    form: PhantomData<(D, C, S)>,
}

impl<D, C, S> Cell<D, C, S> {
    pub(super) fn parameter(&self) -> &Arc<Parameter> {
        &self.parameter
    }

    /// Two occurrences of one parameter, for the named loader's tie protocol.
    fn split(_: &Context<'_>, cell: Self) -> Result<(Self, Self), LoadError> {
        Ok((cell.clone(), cell))
    }
}

impl<D: MaterializableScalar, C: M + 'static, S: M + 'static> Cell<D, C, S> {
    /// Retain only a checked, already prepared host layout; this does not pack or upload it.
    fn prepared<E: M>(host: HostTensor<D, E>, name: &str) -> Result<Self, LoadError> {
        let source = Source::from_host::<D, C, S, E>(host, name)?;
        Ok(Self {
            parameter: Arc::new(Parameter::new(source)),
            form: PhantomData,
        })
    }

    pub(super) fn upload(&self, destination: &Destination<'_>, path: &str) -> io::Result<()> {
        self.parameter.upload::<D, C, S>(destination, path)
    }
}

impl<D, C, S> Clone for Cell<D, C, S> {
    fn clone(&self) -> Self {
        Self {
            parameter: self.parameter.clone(),
            form: PhantomData,
        }
    }
}

/// Ordinary leaves retain exactly the public loader's converted host value. The outer traversal
/// stays strict, and tied stored copies are compared before sharing the target's parameter.
impl<'a, D: MaterializableScalar, C: M + 'static, S: M + 'static> Load<&'a Checkpoint>
    for Cell<D, C, S>
{
    fn load(context: &mut Context<'_>, source: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(source, |cx| {
            cx.leaf(
                source,
                |cx| {
                    if Checkpoint::splits::<C, S>() {
                        Self::prepared(source.host::<D, S, m![{ C }, { S }]>(cx)?, &cx.name())
                    } else {
                        Self::prepared(source.host::<D, S, S>(cx)?, &cx.name())
                    }
                },
                Self::split,
            )
        })
    }
}

/// Fused or packed leaves hand over their host result without a device round trip. A checkpoint
/// rule calling this must first compare and account for its source tensors, including tied copies.
impl<D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M> Load<HostTensor<D, E>>
    for Cell<D, C, S>
{
    fn load(context: &mut Context<'_>, host: HostTensor<D, E>) -> Result<Self, LoadError> {
        context.share(|cx| Self::prepared(host, &cx.name()), Self::split)
    }
}

/// A tied cache hit does not call the host builder or repeat its conversion/packing. The caller
/// is responsible for source-name accounting and tied-copy comparison, as for a prepared value.
impl<'f, D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M>
    Load<&'f dyn Fn() -> Result<HostTensor<D, E>, LoadError>> for Cell<D, C, S>
{
    fn load(
        context: &mut Context<'_>,
        build: &'f dyn Fn() -> Result<HostTensor<D, E>, LoadError>,
    ) -> Result<Self, LoadError> {
        context.share(|cx| Self::prepared(build()?, &cx.name()), Self::split)
    }
}

/// Component/tree copies share the parameter; no HBM ownership escapes the residency boundary.
impl<D, C, S> Load<&Cell<D, C, S>> for Cell<D, C, S> {
    fn load(_: &mut Context<'_>, cell: &Self) -> Result<Self, LoadError> {
        Ok(cell.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::{Cell, Cells};
    use furiosa_opt_models::load::{Checkpoint, Context, Load, LoadError};
    use furiosa_opt_models::ops::linear::Linear;
    use furiosa_opt_models::tensor::form::grouped::{Grouped, Layout};
    use furiosa_opt_models::tensor::memory::Memory;
    use furiosa_opt_models::tensor::parallel::Shard;
    use furiosa_opt_std::prelude::*;
    use safetensors::Dtype;
    use std::sync::Arc;

    axes![Width = 128, Out = 512, Rows = 4, Columns = 4];
    type Chips = m![{ Symbol<Shard<4>> }];
    type Projection<X> = Linear<
        Width,
        Out,
        Chips,
        Grouped<Width, Out, Chips, <X as Memory>::Tensor<bf16, Chips, Layout<Width, Out>>>,
    >;

    #[derive(Load)]
    struct Norm<X: Memory> {
        weight: X::Tensor<bf16, Chips, m![Width]>,
    }

    /// The projection uses the real grouped loader, which produces a packed host value.
    #[derive(Load)]
    struct Tree<X: Memory> {
        norm: Norm<X>,
        #[load(tied = "norm")]
        tied: Norm<X>,
        separate: Norm<X>,
        projection: Projection<X>,
    }

    fn checkpoint() -> Checkpoint {
        let mut source = Checkpoint::default();
        for name in ["norm.weight", "separate.weight"] {
            source
                .insert(name, Dtype::BF16, &[Width::SIZE], vec![0; Width::SIZE * 2])
                .unwrap();
        }
        let weights = (0..Out::SIZE * Width::SIZE)
            .flat_map(|i| {
                bf16::from_f32((i % 61) as f32 / 32.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        source
            .insert(
                "projection.weight",
                Dtype::BF16,
                &[Out::SIZE, Width::SIZE],
                weights,
            )
            .unwrap();
        source
    }

    fn load(source: &Checkpoint) -> Result<Tree<Cells>, LoadError> {
        Tree::load(&mut Context::new(None), source)
    }

    #[test]
    fn loads_and_clones_a_strict_host_tree_with_packed_weights() {
        let source = checkpoint();
        let tree = load(&source).unwrap();
        drop(source);
        let parameter = tree.norm.weight.parameter();
        assert!(parameter.owner().is_none());
        assert_eq!(parameter.source().len(), Width::SIZE * size_of::<bf16>());
        assert!(Arc::ptr_eq(parameter, tree.tied.weight.parameter()));
        assert!(!Arc::ptr_eq(parameter, tree.separate.weight.parameter()));
        assert_eq!(
            Arc::strong_count(parameter),
            2,
            "the tie cache released its owner"
        );

        let copy: Tree<Cells> = Load::load(&mut Context::new(None), &tree).unwrap();
        assert!(Arc::ptr_eq(parameter, copy.norm.weight.parameter()));
        assert!(Arc::ptr_eq(parameter, copy.tied.weight.parameter()));
        assert!(Arc::ptr_eq(
            tree.separate.weight.parameter(),
            copy.separate.weight.parameter()
        ));
        assert_eq!(Arc::strong_count(parameter), 4);
        drop(tree);
        assert!(copy.norm.weight.parameter().owner().is_none());
    }

    #[test]
    fn missing_names_are_aggregated_and_unexpected_names_are_refused() {
        let mut source = checkpoint();
        source.remove("norm.weight");
        source.remove("projection.weight");
        let error = load(&source).err().expect("two leaves are missing");
        assert!(matches!(error, LoadError::Missing(names)
            if names == ["norm.weight", "projection.weight"]));

        let mut source = checkpoint();
        source
            .insert("extra", Dtype::BF16, &[1], vec![0; 2])
            .unwrap();
        let error = load(&source).err().expect("extra is unused");
        assert!(matches!(error, LoadError::Unexpected(names) if names == ["extra"]));
    }

    #[test]
    fn stored_tied_copies_are_compared_before_sharing() {
        let mut source = checkpoint();
        source
            .insert(
                "tied.weight",
                Dtype::BF16,
                &[Width::SIZE],
                vec![0; Width::SIZE * 2],
            )
            .unwrap();
        let tree = load(&source).unwrap();
        assert!(Arc::ptr_eq(
            tree.norm.weight.parameter(),
            tree.tied.weight.parameter()
        ));

        source.remove("tied.weight");
        let mut conflicting = vec![0; Width::SIZE * 2];
        conflicting[..2].copy_from_slice(&bf16::from_f32(1.0).to_bits().to_le_bytes());
        source
            .insert("tied.weight", Dtype::BF16, &[Width::SIZE], conflicting)
            .unwrap();
        let error = load(&source).err().expect("the stored tie differs");
        assert!(matches!(error, LoadError::InvalidData { name, reason }
            if name == "tied.weight" && reason == "tied copy differs from norm.weight"));
    }

    #[test]
    fn malformed_leaf_shapes_and_dtypes_are_refused() {
        for (dtype, shape, reason) in [
            (Dtype::BF16, 127, "shape [127], the leaf holds [128]"),
            (Dtype::I32, 128, "stored as I32, the leaf holds BF16"),
        ] {
            let mut source = checkpoint();
            source.remove("norm.weight");
            source
                .insert(
                    "norm.weight",
                    dtype,
                    &[shape],
                    vec![0; shape * dtype.size()],
                )
                .unwrap();
            let error = load(&source)
                .err()
                .expect("the norm cannot hold this tensor");
            assert!(
                matches!(error, LoadError::InvalidData { name, reason: actual }
                if name == "norm.weight" && actual == reason)
            );
        }
    }

    #[test]
    fn ordinary_sharded_leaves_retain_one_image_per_chip() {
        type Shape = m![Rows, Columns % 2];
        type Leaf = Cell<f32, m![Columns / 2], Shape>;
        let mut source = Checkpoint::default();
        source
            .insert(
                "weight",
                Dtype::F32,
                &[Rows::SIZE, Columns::SIZE],
                (0..16).flat_map(|i| (i as f32).to_le_bytes()).collect(),
            )
            .unwrap();
        let cell = Context::new(None)
            .field("weight", "weight", |cx| Leaf::load(cx, &source))
            .unwrap();
        let source = cell.parameter().source();
        assert_eq!(source.len(), 8 * size_of::<f32>());
        let pieces = source.pieces(2).unwrap();
        assert_eq!(pieces.len(), 1, "one chip-major image contains both shares");
        assert_eq!(pieces[0].chips, 0..2);
        assert_eq!(pieces[0].bytes, 0..32);
        assert!(source.pieces(4).is_err());
    }

    #[test]
    fn deferred_tied_leaves_build_once_and_share_the_parameter() {
        type Shape = Layout<Width, Out>;
        type Leaf = Cell<bf16, Chips, Shape>;
        type Host = HostTensor<bf16, m![{ Chips }, { Shape }]>;
        let calls = std::cell::Cell::new(0);
        let build = || {
            calls.set(calls.get() + 1);
            Ok(Host::from_vec(vec![
                bf16::from_f32(0.5);
                Chips::SIZE * Shape::SIZE
            ]))
        };
        let build: &dyn Fn() -> Result<Host, LoadError> = &build;
        let (first, second) = Context::new(None)
            .ties(&["weight"], |cx| {
                let first = cx.field("weight", "weight", |cx| Leaf::load(cx, build))?;
                let second = cx.tied("tied", "tied", "weight", |cx| Leaf::load(cx, build))?;
                Ok::<_, LoadError>((first, second))
            })
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert!(Arc::ptr_eq(first.parameter(), second.parameter()));
        assert_eq!(Arc::strong_count(first.parameter()), 2);
        assert_eq!(
            first.parameter().source().len(),
            Shape::SIZE * size_of::<bf16>()
        );
        assert!(first.parameter().owner().is_none());
    }
}
