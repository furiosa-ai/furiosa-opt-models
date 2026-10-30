//! Logical row coordinates shared by device layouts.

use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

axes![Single = 1, Eight = 8, Sixteen = 16];

/// Compiled row count.
#[derive(Debug, Clone)]
pub struct Rows<const N: usize>;

pub type SixtyFour = Rows<64>;

impl<const N: usize> AxisName for Rows<N> {
    const NAME: Ident = Ident::new("Rows");
    const SIZE: usize = N;
}

/// A batch of at most `CAP` rows from a capacity of `MAX`.
#[derive(Clone, Debug)]
pub struct Limit<const MAX: usize, const CAP: usize>;

impl<const MAX: usize, const CAP: usize> AxisName for Limit<MAX, CAP> {
    const NAME: Ident = Ident::new("Rows");

    const SIZE: usize = {
        assert!(MAX == 16 || MAX == 64);
        if MAX < CAP { MAX } else { CAP }
    };
}

/// Row tile bounded by N; a non-divisible capacity falls back to one lane group.
#[derive(Debug, Clone)]
pub struct Tile<R, const N: usize>(PhantomData<R>);

impl<R: AxisName, const N: usize> AxisName for Tile<R, N> {
    const NAME: Ident = Ident::new("Tile");

    const SIZE: usize = {
        assert!(N >= 8 && N % 8 == 0);
        if R::SIZE <= N {
            R::SIZE
        } else if R::SIZE % N == 0 {
            N
        } else {
            Lanes::<R>::SIZE
        }
    };
}

/// Row lanes in a vector packet: the greatest divisor of the row count within the hardware's
/// eight lanes, so every group of rows fills the same lanes.
#[derive(Debug, Clone)]
pub struct Lanes<R>(PhantomData<R>);

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
