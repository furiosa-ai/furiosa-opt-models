//! The standard collections a catalog tree nests: arrays and tuples name each child by its
//! index, an option is its child when it holds one, and `()` and `PhantomData` hold no weight.

use super::super::Memory;
use super::{Leaves, Path, Residency, Visit};
use crate::load::Named;
use std::marker::PhantomData;

impl<T: Residency, const N: usize> Residency for [T; N] {
    type In<R: Memory> = [T::In<R>; N];

    fn map<'t, From: Memory, To: Memory, L: Leaves<'t, From, To>>(
        tree: &'t [T::In<From>; N],
        leaves: &mut L,
        path: Path<'_>,
    ) -> Result<[T::In<To>; N], L::Error> {
        let items = (tree.iter().enumerate())
            .map(|(index, item)| T::map(item, leaves, path.index(index)))
            .collect::<Result<Vec<_>, _>>()?;
        let Ok(items) = items.try_into() else {
            unreachable!("one item maps from each of the N items")
        };
        Ok(items)
    }

    fn visit<R: Memory, V: Visit<R>>(
        tree: &[T::In<R>; N],
        visitor: &mut V,
        path: Path<'_>,
    ) -> Result<(), V::Error> {
        (tree.iter().enumerate())
            .try_for_each(|(index, item)| T::visit(item, visitor, path.index(index)))
    }
}

/// A tuple of children, each named by its index.
macro_rules! tuple {
    ($($child:ident $index:tt),*) => {
        impl<$($child: Residency),*> Residency for ($($child,)*) {
            type In<R: Memory> = ($($child::In<R>,)*);

            fn map<'t, From: Memory, To: Memory, L: Leaves<'t, From, To>>(
                tree: &'t Self::In<From>,
                leaves: &mut L,
                path: Path<'_>,
            ) -> Result<Self::In<To>, L::Error> {
                Ok(($($child::map(&tree.$index, leaves, path.index($index))?,)*))
            }

            fn visit<R: Memory, V: Visit<R>>(
                tree: &Self::In<R>,
                visitor: &mut V,
                path: Path<'_>,
            ) -> Result<(), V::Error> {
                $($child::visit(&tree.$index, visitor, path.index($index))?;)*
                Ok(())
            }

        }
    };
}

tuple!(A 0);
tuple!(A 0, B 1);
tuple!(A 0, B 1, C 2);
tuple!(A 0, B 1, C 2, D 3);

/// A child a tree may omit stands for the tree where it is present.
impl<T: Residency> Residency for Option<T> {
    type In<R: Memory> = Option<T::In<R>>;

    fn map<'t, From: Memory, To: Memory, L: Leaves<'t, From, To>>(
        tree: &'t Option<T::In<From>>,
        leaves: &mut L,
        path: Path<'_>,
    ) -> Result<Option<T::In<To>>, L::Error> {
        (tree.as_ref())
            .map(|child| T::map(child, leaves, path))
            .transpose()
    }

    fn visit<R: Memory, V: Visit<R>>(
        tree: &Option<T::In<R>>,
        visitor: &mut V,
        path: Path<'_>,
    ) -> Result<(), V::Error> {
        (tree.as_ref()).map_or(Ok(()), |child| T::visit(child, visitor, path))
    }
}

/// Markers that hold no weight are the same in every residence.
macro_rules! weightless {
    ($([$($generics:tt)*] $ty:ty => $value:expr;)*) => {$(
        impl<$($generics)*> Residency for $ty {
            type In<R: Memory> = Self;

            fn map<'t, From: Memory, To: Memory, L: Leaves<'t, From, To>>(
                _: &'t Self,
                _: &mut L,
                _: Path<'_>,
            ) -> Result<Self, L::Error> {
                Ok($value)
            }

            fn visit<R: Memory, V: Visit<R>>(
                _: &Self,
                _: &mut V,
                _: Path<'_>,
            ) -> Result<(), V::Error> {
                Ok(())
            }

        }
    )*};
}

weightless! {
    [] () => ();
    [T: ?Sized] PhantomData<T> => PhantomData;
}

/// A named generic child uses its existing tree name, independently of checkpoint loading.
impl<const NAME: &'static str, T: Residency> Residency for Named<NAME, T> {
    type In<X: Memory> = Named<NAME, T::In<X>>;

    fn map<'t, From: Memory, To: Memory, L: Leaves<'t, From, To>>(
        tree: &'t Self::In<From>,
        leaves: &mut L,
        path: Path<'_>,
    ) -> Result<Self::In<To>, L::Error> {
        T::map(&tree.0, leaves, path.field(NAME)).map(Named)
    }

    fn visit<X: Memory, V: Visit<X>>(
        tree: &Self::In<X>,
        visitor: &mut V,
        path: Path<'_>,
    ) -> Result<(), V::Error> {
        T::visit(&tree.0, visitor, path.field(NAME))
    }
}
