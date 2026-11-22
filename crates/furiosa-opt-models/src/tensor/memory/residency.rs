//! Structural traversal of weight trees over [`Memory`].
//!
//! A canonical HBM tree describes its projection into any memory backend. [`Residency::map`]
//! converts leaves and [`Residency::visit`] reads them, in declaration order. Paths describe the
//! tree, not checkpoint tensor names; checkpoint naming remains in `Load`.
//! Arrays contain independent children: traversal never stacks or aliases layers' HBM backings.

pub use furiosa_opt_models_macros::Residency;
pub use path::Path;

use super::Memory;
use furiosa_opt_std::prelude::{HbmTensor, M, MaterializableScalar};

mod path;
mod sequence;

/// A canonical weight tree and its projection into memory `X`.
///
/// Declare the memory position explicitly. Other type parameters retain their meaning:
///
/// ```
/// use furiosa_opt_models::tensor::memory::{Hbm, Memory};
/// use furiosa_opt_models::tensor::memory::residency::Residency;
/// use furiosa_opt_std::prelude::*;
///
/// #[derive(Residency)]
/// #[residency(memory(X))]
/// struct Norm<X: Memory> { weight: X::Tensor<f32, m![1], m![64]> }
///
/// struct Shapes;
/// impl Memory for Shapes {
///     type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = ();
/// }
/// let _: <Norm<Hbm> as Residency>::In<Shapes> = Norm::<Shapes> { weight: () };
/// ```
///
/// Compositions declare their child positions; every field is traversed, including new fields:
///
/// ```
/// use furiosa_opt_models::tensor::memory::residency::Residency;
/// #[derive(Residency)]
/// #[residency(children(A, B))]
/// struct Layer<A, B> { norm: A, branch: B, final_norm: A }
/// ```
///
/// There is no implicit metadata skip. A field without a projection fails compilation:
///
/// ```compile_fail,E0277
/// use furiosa_opt_models::tensor::memory::residency::Residency;
/// #[derive(Residency)]
/// struct Unsupported { label: String }
/// ```
///
/// A fixed HBM leaf in a memory-generic tree also fails: its declared field cannot change backend.
///
/// ```compile_fail,E0308
/// use furiosa_opt_models::tensor::memory::residency::Residency;
/// use furiosa_opt_std::prelude::*;
/// #[derive(Residency)]
/// struct Fixed { weight: HbmTensor<f32, m![1], m![64]> }
/// ```
///
/// `PhantomData<T>` stays unchanged, so its marker must not name a projected memory or child
/// position. Such a declaration is refused rather than silently changing its marker:
///
/// ```compile_fail,E0308
/// use furiosa_opt_models::tensor::memory::{Memory, Hbm};
/// use furiosa_opt_models::tensor::memory::residency::Residency;
/// use std::marker::PhantomData;
/// #[derive(Residency)]
/// #[residency(memory(X))]
/// struct Marker<X: Memory> { marker: PhantomData<X> }
/// ```
///
/// On fields, `#[residency(transparent)]` omits that field's segment, and `role = Role` uses the
/// role type's snake-case name. Enums add the variant's snake-case name. These attributes affect
/// tree paths only. There is no `keep` or `skip` attribute.
pub trait Residency {
    /// The canonical tree with each leaf in `X`.
    type In<X: Memory>;

    /// Maps every leaf in tree order, stopping at the first error.
    fn map<'t, From: Memory, To: Memory, L: Leaves<'t, From, To>>(
        tree: &'t Self::In<From>,
        leaves: &mut L,
        path: Path<'_>,
    ) -> Result<Self::In<To>, L::Error>;

    /// Visits every leaf in tree order, stopping at the first error.
    fn visit<X: Memory, V: Visit<X>>(
        tree: &Self::In<X>,
        visitor: &mut V,
        path: Path<'_>,
    ) -> Result<(), V::Error>;
}

/// Converts the leaf storage lent by a tree for `'t`; the mapper owns any sharing policy.
/// An error drops the partially built destination tree without changing the source.
pub trait Leaves<'t, From: Memory, To: Memory> {
    type Error;

    fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
        &mut self,
        leaf: &'t From::Tensor<D, C, S>,
        path: Path<'_>,
    ) -> Result<To::Tensor<D, C, S>, Self::Error>;
}

/// Reads one backend's leaves without changing the tree.
pub trait Visit<X: Memory> {
    type Error;

    fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
        &mut self,
        leaf: &X::Tensor<D, C, S>,
        path: Path<'_>,
    ) -> Result<(), Self::Error>;
}

impl<D: MaterializableScalar, C: M + 'static, S: M + 'static> Residency for HbmTensor<D, C, S> {
    type In<X: Memory> = X::Tensor<D, C, S>;

    fn map<'t, From: Memory, To: Memory, L: Leaves<'t, From, To>>(
        tree: &'t Self::In<From>,
        leaves: &mut L,
        path: Path<'_>,
    ) -> Result<Self::In<To>, L::Error> {
        leaves.leaf::<D, C, S>(tree, path)
    }

    fn visit<X: Memory, V: Visit<X>>(
        tree: &Self::In<X>,
        visitor: &mut V,
        path: Path<'_>,
    ) -> Result<(), V::Error> {
        visitor.leaf::<D, C, S>(tree, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::Named;
    use crate::tensor::memory::Hbm;
    use furiosa_opt_std::prelude::{Broadcast, bf16, m};
    use std::convert::Infallible;
    use std::marker::PhantomData;
    use std::sync::{Arc, Weak};

    struct Sources;
    struct Encoded;

    impl Memory for Sources {
        type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Arc<usize>;
    }

    impl Memory for Encoded {
        type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Arc<String>;
    }

    #[derive(Residency)]
    #[residency(memory(X))]
    struct Norm<X: Memory> {
        weight: X::Tensor<f32, m![1], m![64]>,
        bias: Option<X::Tensor<bf16, m![1], m![128]>>,
    }

    #[derive(Residency)]
    #[residency(children(N, B))]
    struct Layer<N, B> {
        norm: N,
        branch: B,
        final_norm: N,
    }

    #[derive(Residency)]
    #[residency(memory(X), children(B))]
    struct Body<X: Memory, B> {
        norm: Norm<X>,
        child: B,
    }

    #[derive(Residency)]
    #[residency(memory(X))]
    struct Tree<'a, X, G, const N: usize>
    where
        X: Memory,
    {
        layers: [Norm<X>; N],
        pair: (Named<"left", Norm<X>>, ()),
        marker: PhantomData<(&'a (), G)>,
    }

    #[derive(Residency)]
    #[residency(children(T))]
    enum Branch<T> {
        FeedForward { r#type: T },
        Pair(T, T),
        Empty,
    }

    struct AttentionNorm;

    #[derive(Residency)]
    #[residency(children(T))]
    struct Wrapped<T>(
        #[residency(transparent)] T,
        #[residency(role = AttentionNorm)] T,
    );

    type Gain = HbmTensor<f32, m![1], m![64]>;

    #[derive(Default)]
    struct Paths {
        seen: Vec<String>,
        refuse: Option<&'static str>,
    }

    impl Visit<Sources> for Paths {
        type Error = String;

        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            _: &Arc<usize>,
            path: Path<'_>,
        ) -> Result<(), Self::Error> {
            let path = path.to_string();
            self.seen.push(path.clone());
            if self.refuse == Some(path.as_str()) {
                Err(path)
            } else {
                Ok(())
            }
        }
    }

    struct Encode {
        seen: Vec<String>,
        built: Vec<Weak<String>>,
        refuse: Option<&'static str>,
    }

    impl<'t> Leaves<'t, Sources, Encoded> for Encode {
        type Error = String;

        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &'t Arc<usize>,
            path: Path<'_>,
        ) -> Result<Arc<String>, Self::Error> {
            let path = path.to_string();
            self.seen.push(path.clone());
            if self.refuse == Some(path.as_str()) {
                return Err(path);
            }
            let leaf = Arc::new(format!("weight:{}", **leaf));
            self.built.push(Arc::downgrade(&leaf));
            Ok(leaf)
        }
    }

    struct Borrow;

    impl<'t> Leaves<'t, Sources, &'t Sources> for Borrow {
        type Error = Infallible;

        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &'t Arc<usize>,
            _: Path<'_>,
        ) -> Result<&'t Arc<usize>, Self::Error> {
            Ok(leaf)
        }
    }

    struct Share;

    impl<'t> Leaves<'t, Sources, Sources> for Share {
        type Error = Infallible;

        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &'t Arc<usize>,
            _: Path<'_>,
        ) -> Result<Arc<usize>, Self::Error> {
            Ok(leaf.clone())
        }
    }

    fn norm(weight: usize, bias: Option<usize>) -> Norm<Sources> {
        Norm {
            weight: Arc::new(weight),
            bias: bias.map(Arc::new),
        }
    }

    fn tree() -> Tree<'static, Sources, (), 2> {
        Tree {
            layers: [norm(10, Some(11)), norm(20, None)],
            pair: (Named(norm(30, Some(31))), ()),
            marker: PhantomData,
        }
    }

    #[test]
    fn visits_declared_fields_in_tree_order() {
        let mut paths = Paths::default();
        Tree::<'static, Hbm, (), 2>::visit(&tree(), &mut paths, Path::ROOT).unwrap();
        assert_eq!(
            paths.seen,
            [
                "layers.0.weight",
                "layers.0.bias",
                "layers.1.weight",
                "pair.0.left.weight",
                "pair.0.left.bias",
            ]
        );

        let mut paths = Paths::default();
        let layer = Layer {
            norm: Arc::new(1),
            branch: (Arc::new(2),),
            final_norm: Arc::new(3),
        };
        Layer::<Gain, (Gain,)>::visit(&layer, &mut paths, Path::ROOT).unwrap();
        assert_eq!(paths.seen, ["norm", "branch.0", "final_norm"]);
    }

    #[test]
    fn maps_memory_backends_and_borrows_the_source_tree() {
        let source = tree();
        let mut encode = Encode {
            seen: Vec::new(),
            built: Vec::new(),
            refuse: None,
        };
        let mapped: Tree<'static, Encoded, (), 2> =
            Tree::<'static, Hbm, (), 2>::map(&source, &mut encode, Path::ROOT).unwrap();
        assert_eq!(*mapped.layers[0].weight, "weight:10");
        assert_eq!(
            mapped.layers[0].bias.as_deref().map(String::as_str),
            Some("weight:11")
        );
        assert_eq!(*mapped.layers[1].weight, "weight:20");
        assert!(mapped.layers[1].bias.is_none());
        assert_eq!(*mapped.pair.0.0.weight, "weight:30");
        let mut paths = Paths::default();
        Tree::<'static, Hbm, (), 2>::visit(&source, &mut paths, Path::ROOT).unwrap();
        assert_eq!(encode.seen, paths.seen);

        let borrowed: Tree<'static, &Sources, (), 2> =
            Tree::<'static, Hbm, (), 2>::map(&source, &mut Borrow, Path::ROOT).unwrap();
        assert!(std::ptr::eq(
            borrowed.layers[0].weight,
            &source.layers[0].weight
        ));
        assert!(std::ptr::eq(
            borrowed.pair.0.0.weight,
            &source.pair.0.0.weight
        ));
    }

    #[test]
    fn mapper_controls_sharing_and_layers_remain_independent() {
        let tied = Arc::new(7);
        let source = [norm(1, None), norm(2, None)];
        let mapped = <[Norm<Hbm>; 2] as Residency>::map(&source, &mut Share, Path::ROOT).unwrap();
        assert!(!Arc::ptr_eq(&mapped[0].weight, &mapped[1].weight));
        assert!(Arc::ptr_eq(&mapped[0].weight, &source[0].weight));
        let source = (tied.clone(), tied);
        let mapped = <(Gain, Gain)>::map(&source, &mut Share, Path::ROOT).unwrap();
        assert!(
            Arc::ptr_eq(&mapped.0, &mapped.1),
            "a shared leaf remains shared by mapper policy"
        );
    }

    #[test]
    fn projects_memory_and_child_positions_together() {
        let source = Body {
            norm: norm(1, None),
            child: (Arc::new(2),),
        };
        let mapped: Body<Sources, (Arc<usize>,)> =
            Body::<Hbm, (Gain,)>::map(&source, &mut Share, Path::ROOT).unwrap();
        assert!(Arc::ptr_eq(&mapped.norm.weight, &source.norm.weight));
        assert!(Arc::ptr_eq(&mapped.child.0, &source.child.0));
        let mut paths = Paths::default();
        Body::<Hbm, (Gain,)>::visit(&mapped, &mut paths, Path::ROOT).unwrap();
        assert_eq!(paths.seen, ["norm.weight", "child.0"]);
    }

    #[test]
    fn first_mapping_error_drops_partial_results_and_preserves_source() {
        let source = tree();
        let mut encode = Encode {
            seen: Vec::new(),
            built: Vec::new(),
            refuse: Some("layers.1.weight"),
        };
        let result = Tree::<'static, Hbm, (), 2>::map(&source, &mut encode, Path::ROOT);
        assert_eq!(result.err(), Some("layers.1.weight".to_owned()));
        assert_eq!(
            encode.seen,
            ["layers.0.weight", "layers.0.bias", "layers.1.weight"]
        );
        assert!(encode.built.iter().all(|leaf| leaf.upgrade().is_none()));
        assert_eq!(*source.layers[0].weight, 10);
        assert_eq!(*source.layers[1].weight, 20);
    }

    #[test]
    fn visit_stops_at_the_first_refused_leaf() {
        let mut paths = Paths {
            seen: Vec::new(),
            refuse: Some("layers.0.bias"),
        };
        let error =
            Tree::<'static, Hbm, (), 2>::visit(&tree(), &mut paths, Path::ROOT).unwrap_err();
        assert_eq!(error, "layers.0.bias");
        assert_eq!(paths.seen, ["layers.0.weight", "layers.0.bias"]);
    }

    #[test]
    fn variants_and_explicit_path_segments_are_structural() {
        let variants = [
            Branch::FeedForward {
                r#type: Arc::new(1),
            },
            Branch::Pair(Arc::new(2), Arc::new(3)),
            Branch::Empty,
        ];
        let mut paths = Paths::default();
        <[Branch<Gain>; 3]>::visit(&variants, &mut paths, Path::ROOT).unwrap();
        assert_eq!(paths.seen, ["0.feed_forward.type", "1.pair.0", "1.pair.1"]);
        let mapped =
            <[Branch<Gain>; 3] as Residency>::map(&variants, &mut Share, Path::ROOT).unwrap();
        assert!(matches!(&mapped[2], Branch::Empty));
        let mut paths = Paths::default();
        Wrapped::<Gain>::visit(&Wrapped(Arc::new(4), Arc::new(5)), &mut paths, Path::ROOT).unwrap();
        assert_eq!(paths.seen, ["", "attention_norm"]);
    }

    #[test]
    fn empty_collections_and_weightless_markers_have_no_leaves() {
        let mut paths = Paths::default();
        <([Gain; 0], Option<Gain>, (), PhantomData<str>)>::visit(
            &([], None, (), PhantomData),
            &mut paths,
            Path::ROOT,
        )
        .unwrap();
        assert!(paths.seen.is_empty());
        let mapped = <([Gain; 0], Option<Gain>, (), PhantomData<str>)>::map(
            &([], None, (), PhantomData),
            &mut Share,
            Path::ROOT,
        )
        .unwrap();
        assert!(mapped.1.is_none());
    }
}
