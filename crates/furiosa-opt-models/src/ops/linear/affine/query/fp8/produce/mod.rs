//! The quantized form of a launch's periods: its rows quantized once into periods, pairs of tiles
//! then a last single tile, and each period's queries, keys and values, each by its projection.

use super::*;
use crate::Begin;
use crate::ops::linear::query::group::{TilePair, Tiles};
use crate::ops::linear::query::launch::{Launch, Next};
use crate::tensor::live::Live;
use core::mem;
pub(crate) use period::Period;
use std::marker::PhantomData;

mod lanes;
mod period;

pub(crate) type Single = Tiles<1>;

/// What the quantized form keeps between a launch's periods: the pairs' quantized rows, staged
/// once in HBM when the pairs begin, and the next pair's rows, copied while a pair projects.
pub struct Quantized<Hidden: AxisName, KvHeads: AxisName, All: AxisName> {
    lanes: lanes::Lanes<Hidden, All>,
    next: Period<Hidden, TilePair, KvHeads>,
}

/// The value projection begins the input its siblings read, as the periods' row scales lie in the
/// slices of its heads: the launch's live rows `live`, before any span arrives.
impl<Hidden: AxisName, KvHeads: AxisName, Head: AxisName, All: AxisName>
    crate::Module<Begin<()>, Live<(), All>, &mut ()>
    for Affine<
        (
            Weight<Hidden, KvHeads, Head>,
            ValueScale<Hidden, KvHeads, Head>,
        ),
        (),
    >
{
    type Output = Launch<Hidden, Quantized<Hidden, KvHeads, All>, All>;

    fn forward(
        &self,
        _: &mut Device,
        live: Live<(), All>,
        _: &mut (),
        _: Begin<()>,
    ) -> Self::Output {
        Launch::new(
            live,
            Quantized {
                lanes: lanes::Lanes::new(),
                next: Period::new(DmTensor::new(), DmTensor::new(), Tile::SIZE / 8),
            },
        )
    }
}

/// The pairs begun: every pair's rows quantized and staged, and the first pair's rows copied.
impl<Hidden: AxisName, KvHeads: AxisName, Head: AxisName, All: AxisName>
    Op<Begin<Next<'_, TilePair, Hidden, Quantized<Hidden, KvHeads, All>, All>>>
    for Affine<
        (
            Weight<Hidden, KvHeads, Head>,
            ValueScale<Hidden, KvHeads, Head>,
        ),
        (),
    >
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        Begin(next): Begin<Next<'_, TilePair, Hidden, Quantized<Hidden, KvHeads, All>, All>>,
    ) {
        let Next {
            rows,
            staging,
            live,
            ..
        } = next;
        let begin = Begin((live.start() / Tile::SIZE, live.rows()));
        let spread = staging.forward(&mut *device, (rows, begin));
        let (lanes, first) = staging.forward(device, spread);
        staging.lanes = lanes;
        staging.next = first;
    }
}

/// A pair of tiles, copied while the pair before it projected; it copies the next pair's rows, the
/// last pair rows past the launch's pairs, which no period reads, so every pair does the same work.
impl<Hidden: AxisName, KvHeads: AxisName, Head: AxisName, All: AxisName>
    Op<Next<'_, TilePair, Hidden, Quantized<Hidden, KvHeads, All>, All>>
    for Affine<
        (
            Weight<Hidden, KvHeads, Head>,
            ValueScale<Hidden, KvHeads, Head>,
        ),
        (),
    >
{
    type Output = Period<Hidden, TilePair, KvHeads>;

    fn forward(
        &self,
        device: &mut Device,
        next: Next<'_, TilePair, Hidden, Quantized<Hidden, KvHeads, All>, All>,
    ) -> Self::Output {
        let Next {
            staging, live, at, ..
        } = next;
        let ahead = (live.start() / Tile::SIZE, at + 1);
        let ahead = staging.forward(device, (&staging.lanes, ahead));
        mem::replace(&mut staging.next, ahead)
    }
}

/// The last single tile, copied and quantized alone.
impl<Hidden: AxisName, KvHeads: AxisName, Head: AxisName, All: AxisName>
    Op<Next<'_, Single, Hidden, Quantized<Hidden, KvHeads, All>, All>>
    for Affine<
        (
            Weight<Hidden, KvHeads, Head>,
            ValueScale<Hidden, KvHeads, Head>,
        ),
        (),
    >
{
    type Output = Period<Hidden, Single, KvHeads>;

    fn forward(
        &self,
        device: &mut Device,
        next: Next<'_, Single, Hidden, Quantized<Hidden, KvHeads, All>, All>,
    ) -> Self::Output {
        let Next {
            rows,
            staging,
            live,
            ..
        } = next;
        let (in_dm, _) = staging.forward(&mut *device, (rows, Begin(live)));
        staging.forward(device, in_dm)
    }
}
