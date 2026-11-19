//! Bind a request's checked block prefix to a leaf while preserving owned live results.

use crate::Module;
use crate::load::Load;
use crate::tensor::live::{BlockPrefix, Live};
use crate::tensor::memory::residency::Residency;
use furiosa_opt_std::prelude::{AxisName, Device, DeviceSend};

/// Project `&Live<T, R>` to a borrowed checked-prefix operand for `M`.
///
/// The environment is `(&prefix, env)`. The child receives `&prefix.with(input.value())`,
/// the remaining `env` and the state unchanged. Its owned [`Live`] result is returned directly;
/// no tensor allocation, copy, repacking or metadata read is added.
///
/// **Same-interval precondition:** request preparation must derive `prefix` and the input's
/// initialized live interval from the same checked work geometry. The input must start at zero
/// and have exactly `prefix.rows()` initialized rows; the child must preserve that interval.
/// This adapter does not round, truncate or extend arbitrary live ranges. It does not check or
/// assert the precondition on the device. Derive both readers once at the checked request
/// boundary and preserve the interval through the composition.
///
/// Loading and residency preserve the wrapped child's existing names and leaf identities.
#[derive(Clone, Copy, DeviceSend, Load, Residency)]
#[residency(children(M))]
pub struct Prefix<M>(
    #[load(rename = "")]
    #[residency(transparent)]
    pub M,
);

impl<I, M> !crate::Op<I> for Prefix<M> {}

impl<'i, 'p, T, U, R: AxisName, B: AxisName, M, E, S>
    Module<&'i Live<T, R>, (&'p BlockPrefix<(), R, B>, E), S> for Prefix<M>
where
    M: for<'v> Module<&'v BlockPrefix<&'i T, R, B>, E, S, Output = Live<U, R>>,
{
    type Output = Live<U, R>;

    fn forward(
        &self,
        device: &mut Device,
        (prefix, env): (&'p BlockPrefix<(), R, B>, E),
        state: S,
        input: &'i Live<T, R>,
    ) -> Self::Output {
        let rows = prefix.with(input.value());
        self.0.forward(device, env, state, &rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::{Checkpoint, Context, LoadError, Named};
    use crate::tensor::memory::Memory;
    use crate::tensor::memory::residency::{Leaves, Path, Visit};
    use furiosa_opt_std::prelude::*;
    use std::convert::Infallible;
    use std::sync::Arc;

    struct HostWeight(HostTensor<bf16, m![1]>);

    impl<'a> Load<&'a Checkpoint> for HostWeight {
        fn load(cx: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
            from.host::<bf16, m![1], m![1]>(cx).map(Self)
        }
    }

    impl<'a> Load<&'a HostWeight> for &'a HostWeight {
        fn load(_: &mut Context<'_>, from: &'a HostWeight) -> Result<Self, LoadError> {
            Ok(from)
        }
    }

    struct Sources;

    impl Memory for Sources {
        type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Arc<usize>;
    }

    #[derive(Default)]
    struct Paths(Vec<String>);

    impl Visit<Sources> for Paths {
        type Error = Infallible;

        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            _: &Arc<usize>,
            path: Path<'_>,
        ) -> Result<(), Self::Error> {
            self.0.push(path.to_string());
            Ok(())
        }
    }

    impl<'t> Leaves<'t, Sources, Sources> for Paths {
        type Error = Infallible;

        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &'t Arc<usize>,
            path: Path<'_>,
        ) -> Result<Arc<usize>, Self::Error> {
            self.0.push(path.to_string());
            Ok(Arc::clone(leaf))
        }
    }

    #[test]
    fn residency_adds_no_path_segment_or_new_leaf_identity() {
        type Tree = Prefix<Named<"weight", HbmTensor<bf16, m![1], m![1]>>>;
        let tree: <Tree as Residency>::In<Sources> = Prefix(Named(Arc::new(7)));
        let mut mapped = Paths::default();
        let copy = Tree::map::<Sources, Sources, _>(&tree, &mut mapped, Path::ROOT).unwrap();
        assert_eq!(mapped.0, ["weight"]);
        assert!(Arc::ptr_eq(&tree.0.0, &copy.0.0));
        let mut visited = Paths::default();
        Tree::visit::<Sources, _>(&copy, &mut visited, Path::ROOT).unwrap();
        assert_eq!(visited.0, ["weight"]);
    }

    #[test]
    fn named_host_loading_remains_strict_and_borrowing_keeps_the_child() {
        type Tree = Prefix<Named<"weight", HostWeight>>;
        let mut source = Checkpoint::default();
        source
            .insert(
                "weight",
                safetensors::Dtype::BF16,
                &[1],
                bf16::from_f32(2.0).to_bits().to_le_bytes().to_vec(),
            )
            .unwrap();
        let tree = Tree::load(&mut Context::new(None), &source).unwrap();
        let borrowed =
            Prefix::<Named<"weight", &HostWeight>>::load(&mut Context::new(None), &tree).unwrap();
        assert!(std::ptr::eq(borrowed.0.0, &tree.0.0));
        assert_eq!(tree.0.0.0.into_vec(), [bf16::from_f32(2.0)]);

        source
            .insert("unused", safetensors::Dtype::BF16, &[1], vec![0; 2])
            .unwrap();
        assert!(matches!(
            Tree::load(&mut Context::new(None), &source),
            Err(LoadError::Unexpected(names)) if names == ["unused"]
        ));
        assert!(matches!(
            Tree::load(&mut Context::new(None), &Checkpoint::default()),
            Err(LoadError::Missing(names)) if names == ["weight"]
        ));
    }
}
