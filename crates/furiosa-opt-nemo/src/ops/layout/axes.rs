//! Logical row and chip coordinates shared by device layouts.

use furiosa_opt_std::prelude::*;

axes![Single = 1, Eight = 8, Sixteen = 16];

/// One device chip.
pub type Chip = m![1];

/// Chip coordinates shared by the tensors of a composition.
pub trait Parallelism {
    type Chip: M + Send;
    type Local<W: AxisName>: AxisName;

    const CHIPS: usize = Self::Chip::SIZE;
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

/// One shard's equal share of a feature axis, retaining its axis name.
#[derive(Clone, Debug)]
pub struct Part<W, const N: usize>(std::marker::PhantomData<W>);

impl<W: AxisName, const N: usize> AxisName for Part<W, N> {
    const NAME: Ident = W::NAME;

    const SIZE: usize = {
        assert!(W::SIZE.is_multiple_of(Shard::<N>::SIZE));
        W::SIZE / Shard::<N>::SIZE
    };
}

/// Compiled row count.
#[derive(Debug, Clone)]
pub struct Rows<const N: usize>;

impl<const N: usize> AxisName for Rows<N> {
    const NAME: Ident = Ident::new("Rows");
    const SIZE: usize = N;
}

/// Row lanes in a vector packet: the greatest divisor of the row count within the hardware's
/// eight lanes, so every group of rows fills the same lanes.
#[derive(Debug, Clone)]
pub struct Lanes<R>(core::marker::PhantomData<R>);

impl<R: AxisName> AxisName for Lanes<R> {
    const NAME: Ident = Ident::new("Lanes");

    const SIZE: usize = {
        let mut lanes = if R::SIZE < 8 { R::SIZE } else { 8 };
        while R::SIZE % lanes != 0 {
            lanes -= 1;
        }
        lanes
    };
}

/// The row packets of an `R`-row envelope: a loop over its live rows, one packet of lanes a trip.
impl<R: AxisName> crate::compose::Space for Lanes<R> {
    type Envelope = R;
}
