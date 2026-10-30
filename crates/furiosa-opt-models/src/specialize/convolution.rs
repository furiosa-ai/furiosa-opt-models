//! How a convolution maps onto a cluster: the strategy its contraction takes over a slice's lanes,
//! and how a dense one's work splits over the slices (input channels a packet, output channels a
//! slice, rows a slice and output slabs in turn), each the choice the slice's register file and
//! device memory allow.

#![deny(clippy::arbitrary_source_item_ordering)]

use core::marker::PhantomData;
use core::mem;

use furiosa_opt_std::prelude::*;

use super::{SLICES, TRF};

/// How a convolution's contraction maps onto a slice's lanes.
pub(crate) enum Strategy {
    /// Eight output channels a lane: the input channels reduce in the packet, the taps in time.
    Dense,
    /// One output sample a lane, weighing its band of an input window pair: the input channels
    /// reduce in the packet into one output channel.
    Banded,
    /// As banded, every channel filtered alone by one filter: the channels pass in the packet.
    Depthwise,
}

/// Lanes of a slice's contraction: a dense convolution's output channels, a band's output samples.
const LANES: usize = 8;

impl Strategy {
    /// The strategy of a convolution of `groups` groups into `outputs` channels over `taps` taps at
    /// `stride`: a band when each channel is filtered alone or all reduce into one, else dense.
    pub(crate) const fn of(groups: usize, outputs: usize, taps: usize, stride: usize) -> Self {
        if groups == 1 && outputs != 1 {
            assert!(
                outputs.is_multiple_of(LANES),
                "a dense convolution's lanes hold eight output channels each"
            );
            return Strategy::Dense;
        }
        assert!(
            taps <= Strategy::reach(stride),
            "a band holds the taps of its window pair"
        );
        if groups == 1 {
            Strategy::Banded
        } else {
            assert!(
                groups == outputs,
                "a grouped convolution filters each channel alone"
            );
            Strategy::Depthwise
        }
    }

    /// Taps a band holds at `stride`: lane `l` reads from `l` strides into a window pair twice
    /// `LANES` strides long, which leaves the last lane `LANES` strides and one more.
    pub(crate) const fn reach(stride: usize) -> usize {
        (LANES + 1) * stride
    }
}

/// BF16 weights a slice's tensor register file holds.
const REGISTER: usize = TRF / mem::size_of::<bf16>();

/// Bytes of device memory a slice's columns, weights and results may take.
const MEMORY: usize = 384 * 1024;

/// Input channels a slice reduces: all of them up to sixteen.
#[derive(Clone, Debug)]
pub(crate) struct Narrow<In>(PhantomData<In>);

impl<In: AxisName> AxisName for Narrow<In> {
    const NAME: Ident = Ident::new("Narrow");
    const SIZE: usize = {
        assert!(In::SIZE == 8 || In::SIZE.is_multiple_of(16));
        if In::SIZE < 16 { In::SIZE } else { 16 }
    };
}

/// Output channels a slice computes: the most, in eights dividing `Out`, whose weights for one
/// part of the input channels fit the tensor register file.
#[derive(Clone, Debug)]
pub(crate) struct Outputs<Out, Tap>(PhantomData<(Out, Tap)>);

impl<Out: AxisName, Tap: AxisName> AxisName for Outputs<Out, Tap> {
    const NAME: Ident = Ident::new("Outputs");
    const SIZE: usize = {
        assert!(Out::SIZE.is_multiple_of(8));
        let mut block = Out::SIZE;
        while block * Tap::SIZE * 16 > REGISTER || Out::SIZE % block != 0 {
            block -= 8;
        }
        block
    };
}

/// Rows a slice computes: the fewest dividing the span that the slices left by the output blocks
/// and input parts cover.
#[derive(Clone, Debug)]
pub(crate) struct Group<In, Out, Tap, S>(PhantomData<(In, Out, Tap, S)>);

impl<In: AxisName, Out: AxisName, Tap: AxisName, S: AxisName> AxisName for Group<In, Out, Tap, S> {
    const NAME: Ident = Ident::new("Group");
    const SIZE: usize = {
        let others = Out::SIZE / Outputs::<Out, Tap>::SIZE * (In::SIZE / Narrow::<In>::SIZE);
        assert!(
            others <= SLICES,
            "the output blocks and input parts take more than a cluster"
        );
        let mut groups = SLICES / others;
        while S::SIZE % groups != 0 {
            groups -= 1;
        }
        S::SIZE / groups
    };
}

/// Output slabs a slice computes one after another: the fewest whose results and weights fit the
/// slice's device memory beside the columns, which every slab reads.
#[derive(Clone, Debug)]
pub(crate) struct Slabs<D, In, Out, Tap, S>(PhantomData<(D, In, Out, Tap, S)>);

impl<D: Scalar, In: AxisName, Out: AxisName, Tap: AxisName, S: AxisName> AxisName
    for Slabs<D, In, Out, Tap, S>
{
    const NAME: Ident = Ident::new("Slabs");
    const SIZE: usize = {
        let (part, block, rows) = (
            Narrow::<In>::SIZE,
            Outputs::<Out, Tap>::SIZE,
            Group::<In, Out, Tap, S>::SIZE,
        );
        let columns = rows * Tap::SIZE * part * mem::size_of::<D>();
        let per = rows * 4 + Tap::SIZE * part * 2;
        let mut slabs = 1;
        while columns + per * (block / slabs) > MEMORY {
            slabs *= 2;
            assert!(
                block % (8 * slabs) == 0,
                "a slice's rows do not fit its device memory"
            );
        }
        slabs
    };
}
