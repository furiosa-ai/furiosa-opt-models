//! Row grouping and explicit copies between device tensor mappings.

pub mod axes;

use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use axes::Lanes;

pub(crate) type BroadcastClusters = m![1 # 2];
pub(crate) type BroadcastSlices = m![1 # 256];

/// The live rows of a launch over an `R`-row envelope, in packets of lanes.
pub type Live<R> = crate::compose::Tiles<Lanes<R>>;

/// A launch's rows on chip and the live ones among them: a leaf computes the live rows of `value`
/// and hands the same `tiles` on with its result; a row past them is padding nothing reads.
pub struct Dm<S, X> {
    pub value: X,
    pub tiles: crate::compose::Tiles<S>,
}

/// Elements in a 256-bit DMA packet.
#[derive(Clone, Debug)]
pub(crate) struct Packet<D>(PhantomData<D>);

impl<D: Scalar> AxisName for Packet<D> {
    const NAME: Ident = Ident::new("Packet");
    const SIZE: usize = 256 / D::BITS;
}
