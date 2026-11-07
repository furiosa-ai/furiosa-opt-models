//! Two layer collections applied in one fixed interleaved order.

use super::Contexts;
use crate::{Module, Schedule};
use furiosa_opt_std::prelude::{DeviceSend, unroll};
use std::marker::ConstParamTy;

/// Which collection supplies the next layer in a compiled composition.
#[derive(Clone, Copy, PartialEq, Eq, ConstParamTy)]
pub enum Branch {
    First,
    Second,
}

impl Branch {
    /// The order of `N` layers whose every `period`-th layer, counting from one, is the second
    /// collection's.
    pub const fn periodic<const N: usize>(period: usize) -> [Self; N] {
        assert!(period > 0 && N.is_multiple_of(period));
        let mut order = [Self::First; N];
        let mut index = period - 1;
        while index < N {
            order[index] = Self::Second;
            index += period;
        }
        order
    }
}

impl const super::Entry for Branch {
    fn branch(self) -> usize {
        match self {
            Self::First => 0,
            Self::Second => 1,
        }
    }
}

/// Fixed heterogeneous ordering, with homogeneous parameters and state stored separately.
#[derive(DeviceSend)]
pub struct Interleave<A, B, const ORDER: &'static [Branch]> {
    pub first: A,
    pub second: B,
}

impl<A, B, const FIRST: usize, const SECOND: usize, const ORDER: &'static [Branch]>
    Interleave<[A; FIRST], [B; SECOND], ORDER>
{
    /// Positions in the complete composition, in each collection's storage order.
    pub const POSITIONS: ([usize; FIRST], [usize; SECOND]) =
        (Self::SCHEDULE.positions.0, Self::SCHEDULE.positions.1);

    pub(crate) const SCHEDULE: Schedule<FIRST, SECOND> = Schedule::new(ORDER, 0);

    // Repeated periods stay rolled; each period is unrolled for its two layer types.
    const PERIOD: usize = Self::SCHEDULE.period;

    const GROUPS: usize = ORDER.len() / Self::PERIOD;

    const COUNTS: (usize, usize) = (Self::SCHEDULE.counts[0], Self::SCHEDULE.counts[1]);

    pub fn try_from_fn<E>(
        first: impl FnMut(usize) -> Result<A, E>,
        second: impl FnMut(usize) -> Result<B, E>,
    ) -> Result<Self, E> {
        let (a, b) = Self::POSITIONS;
        let first = a.into_iter().map(first).collect::<Result<Vec<_>, E>>()?;
        let second = b.into_iter().map(second).collect::<Result<Vec<_>, E>>()?;
        Ok(Self {
            first: first.try_into().unwrap_or_else(|_| unreachable!()),
            second: second.try_into().unwrap_or_else(|_| unreachable!()),
        })
    }
}

impl<D, X, P, Q, A, B, const FIRST: usize, const SECOND: usize, const ORDER: &'static [Branch]>
    Module<(&mut D, (X, (P, Q)))> for Interleave<[A; FIRST], [B; SECOND], ORDER>
where
    P: Contexts<A, D, X>,
    Q: Contexts<B, D, X>,
{
    type Output = X;

    fn forward(&self, (device, (mut hidden, (mut first, mut second))): (&mut D, (X, (P, Q)))) -> X {
        // Device code reads the order through constant positions; a slice constant does not lower.
        let order = Self::POSITIONS.0;
        for group in 0..Self::GROUPS {
            let (mut a, mut b) = (0, 0);
            #[unroll]
            for offset in 0..Self::PERIOD {
                match a < Self::COUNTS.0 && order[a] == offset {
                    true => {
                        let index = group * Self::COUNTS.0 + a;
                        hidden = first.apply(&self.first[index], &mut *device, hidden, index);
                        a += 1;
                    }
                    false => {
                        let index = group * Self::COUNTS.1 + b;
                        hidden = second.apply(&self.second[index], &mut *device, hidden, index);
                        b += 1;
                    }
                }
            }
        }
        hidden
    }
}

impl<'c, D, X, C, K, A, B, const ORDER: &'static [Branch]> Module<(&mut D, (X, &'c mut (C, K)))>
    for Interleave<A, B, ORDER>
where
    Self: for<'d> Module<(&'d mut D, (X, (&'c mut C, &'c mut K))), Output = X>,
{
    type Output = X;

    fn forward(&self, (device, (value, (first, second))): (&mut D, (X, &'c mut (C, K)))) -> X {
        self.forward((device, (value, (first, second))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Interleaving preserves each branch's state and skips an absent second context.
    #[test]
    fn preserves_interleaved_states() {
        struct First;
        struct Second;
        impl Module<(&mut (), (&i32, &mut i32))> for First {
            type Output = i32;

            fn forward(&self, (_, (hidden, state)): (&mut (), (&i32, &mut i32))) -> i32 {
                *state += hidden;
                *state
            }
        }
        impl Module<(&mut (), (&i32, &mut i32))> for Second {
            type Output = i32;

            fn forward(&self, (_, (hidden, state)): (&mut (), (&i32, &mut i32))) -> i32 {
                *state *= hidden;
                *state
            }
        }
        const ORDER: &[Branch] = &[Branch::First, Branch::Second, Branch::First];
        type Model = Interleave<[First; 2], [Second; 1], ORDER>;
        let (mut first_indices, mut second_indices) = (Vec::new(), Vec::new());
        let model = Model::try_from_fn::<()>(
            |index| {
                first_indices.push(index);
                Ok(First)
            },
            |index| {
                second_indices.push(index);
                Ok(Second)
            },
        )
        .unwrap();
        assert_eq!(Model::POSITIONS, ([0, 2], [1]));
        assert_eq!((first_indices, second_indices), (vec![0, 2], vec![1]));
        assert_eq!(
            Model::try_from_fn(
                |index| Err(index),
                |_| panic!("construction already failed")
            )
            .err(),
            Some(0)
        );
        let (mut first, mut second) = ([1, 10], [2]);
        assert_eq!(model.forward((&mut (), (3, (&mut first, &mut second)))), 18);
        assert_eq!((first, second), ([4, 18], [8]));
        let mut contexts = ([1, 10], [2]);
        assert_eq!(model.forward((&mut (), (3, &mut contexts))), 18);
        assert_eq!(contexts, ([4, 18], [8]));
        let mut first = [1, 10];
        assert_eq!(model.forward((&mut (), (3, (&mut first, ())))), 14);
        assert_eq!(first, [4, 14]);
        let empty = Interleave::<[First; 0], [Second; 0], { &[] }>::try_from_fn::<()>(
            |_| panic!("empty first collection"),
            |_| panic!("empty second collection"),
        )
        .unwrap();
        assert_eq!(empty.forward((&mut (), (3, (&mut [], &mut [])))), 3);
        let second_only = Interleave::<[First; 0], _, { &[Branch::Second] }> {
            first: [],
            second: [Second],
        };
        let mut second = [2];
        assert_eq!(
            second_only.forward((&mut (), (3, (&mut [], &mut second)))),
            6
        );
        assert_eq!(second, [6]);
    }
}
