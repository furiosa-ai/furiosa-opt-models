//! The chips a composition runs on, and the share of a tensor each one holds.

use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// One device chip.
pub type Chip = m![1];

/// Chip coordinates shared by the tensors of a composition.
pub trait Parallelism {
    type Chip: M + Send;
    type Local<W: AxisName>: AxisName;
}

/// Two- or four-chip shard coordinates supported by the collective layouts.
#[derive(Clone, Debug)]
pub struct Shard<const N: usize>;

impl<const N: usize> AxisName for Shard<N> {
    const NAME: Ident = Ident::new("Chips");

    const SIZE: usize = {
        assert!(N == 2 || N == 4);
        N
    };
}

/// Tensor parallelism over `N` chips; one chip holds every tensor whole.
#[derive(Clone, Debug)]
pub struct Parallel<const N: usize>;

impl Parallelism for Parallel<1> {
    type Chip = m![1];
    type Local<W: AxisName> = W;
}

impl Parallelism for Parallel<2> {
    type Chip = m![{ Symbol<Shard<2>> }];
    type Local<W: AxisName> = Part<W, 2>;
}

impl Parallelism for Parallel<4> {
    type Chip = m![{ Symbol<Shard<4>> }];
    type Local<W: AxisName> = Part<W, 4>;
}

/// `N` chips, each holding a copy of every tensor and running the whole module.
#[derive(Clone, Debug)]
pub struct Replicated<const N: usize>;

impl<const N: usize> Parallelism for Replicated<N> {
    type Chip = m![{ Symbol<Shard<N>> }];
    type Local<W: AxisName> = W;
}

/// One shard's equal share of a feature axis, retaining its axis name.
#[derive(Clone, Debug)]
pub struct Part<W, const N: usize>(PhantomData<W>);

impl<W: AxisName, const N: usize> AxisName for Part<W, N> {
    const NAME: Ident = W::NAME;

    const SIZE: usize = {
        assert!(W::SIZE.is_multiple_of(Shard::<N>::SIZE));
        W::SIZE / Shard::<N>::SIZE
    };
}
