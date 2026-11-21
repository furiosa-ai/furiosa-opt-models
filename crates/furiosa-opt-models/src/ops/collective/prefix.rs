//! Whole compact-prefix exchanges inside one capacity and one runtime operation.

use super::{ChipMap, Exchange, O, Pieces, Rows};
use crate::tensor::{
    live::{BlockPrefix, Live},
    parallel::{Parallel, Part, Shard},
    scalar::Value,
};
use furiosa_opt_std::prelude::*;
use std::{io, marker::PhantomData};

/// A nonempty, increasing list of fixed DMA extents. `(S,)` is the final extent;
/// `(S, Tail)` prepends one extent. The model supplies this type once for its envelope.
pub trait Extents {
    /// The first extent in this list.
    type First: AxisName;
    /// The remaining extents, or this same terminal list.
    type Rest: Extents;
    /// Whether this list contains only its final extent.
    const LAST: bool;
}

impl<S: AxisName> Extents for (S,) {
    type First = S;
    type Rest = Self;
    const LAST: bool = true;
}

impl<S: AxisName, Tail: Extents> Extents for (S, Tail) {
    type First = S;
    type Rest = Tail;
    const LAST: bool = false;
}

/// Zero-based work whose row count belongs to `G`, in whole `B`-row blocks.
/// Construct on the host before upload; device readers retain the same checked extent set.
pub struct Prefix<X, R, G, B> {
    prefix: BlockPrefix<X, R, B>,
    extents: PhantomData<G>,
}

impl<R: AxisName, G: Extents, B: AxisName> Prefix<(), R, G, B> {
    /// Refuse work outside the fixed extent set before allocating device metadata.
    pub fn new(prefix: BlockPrefix<(), R, B>) -> io::Result<Self> {
        const { assert!(valid::<R, G>()) };
        if !contains::<G>(prefix.rows()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "exchange work is not one of the fixed transfer extents",
            ));
        }
        i32::try_from(prefix.blocks()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "exchange blocks exceed I32")
        })?;
        Ok(Self {
            prefix,
            extents: PhantomData,
        })
    }

    /// Upload the checked block count once; its immutable owner must outlive every reader.
    pub async fn upload<Chips: M>(
        &self,
        device: &mut Device,
    ) -> io::Result<Inputs<R, G, B, Chips>> {
        let count = HostTensor::<i32, m![1]>::from_vec([self.prefix.blocks() as i32])
            .to_hbm::<Chips, m![1]>(&mut device.pdma)
            .await
            .map_err(io::Error::other)?;
        Ok(Inputs {
            count,
            shape: PhantomData,
        })
    }
}

impl<X, R: AxisName, G: Extents, B: AxisName> Prefix<X, R, G, B> {
    /// Borrow the same zero-based work for consumers that do not select a DMA extent.
    pub fn prefix(&self) -> &BlockPrefix<X, R, B> {
        &self.prefix
    }

    /// Bind another value without changing the checked work or its extent set.
    pub fn with<Y>(&self, value: Y) -> Prefix<Y, R, G, B> {
        Prefix {
            prefix: self.prefix.with(value),
            extents: PhantomData,
        }
    }
}

/// Immutable uploaded metadata; only a checked [`Prefix`] can construct it.
#[derive(DeviceSend)]
pub struct Inputs<R, G, B, Chips: M> {
    count: Value<Chips>,
    shape: PhantomData<(R, G, B)>,
}

impl<R: AxisName, G: Extents, B: AxisName, Chips: M> Inputs<R, G, B, Chips> {
    /// Decode the privately owned count once, retaining the host-checked extent set.
    pub fn read(&self) -> Prefix<(), R, G, B> {
        Prefix {
            prefix: BlockPrefix::read(&self.count),
            extents: PhantomData,
        }
    }
}

impl<const N: usize> Exchange<Parallel<N>> {
    /// Exchange one initialized prefix with one whole TDMA copy, preserving destination tails.
    /// The checked prefix guarantees a zero start and membership in `G`.
    /// Source parts use that compact row count as their stride, within allocation capacity `C`.
    /// Only the DMA mapping varies internally; callers use the same operation for every request.
    pub fn write_prefix<
        D: MaterializableScalar,
        C: AxisName,
        W: AxisName,
        G: Extents,
        B: AxisName,
    >(
        &self,
        device: &mut Device,
        input: &Prefix<&Split<D, N, C, W>, C, G, B>,
        output: &mut Rows<D, N, C, W>,
    ) {
        let prefix = input.prefix();
        self.write_selected::<D, C, W, G>(device, &prefix.interval().with(prefix.value()), output);
    }

    fn write_selected<D: MaterializableScalar, C: AxisName, W: AxisName, G: Extents>(
        &self,
        device: &mut Device,
        input: &Live<&Split<D, N, C, W>, C>,
        output: &mut Rows<D, N, C, W>,
    ) {
        const {
            assert!(G::First::SIZE > 0 && G::First::SIZE <= C::SIZE);
            assert!(G::LAST || G::First::SIZE < <G::Rest as Extents>::First::SIZE);
        }
        if G::LAST || input.rows() == G::First::SIZE {
            self.write_extent::<D, G::First, C, W>(device, input.value(), output);
        } else {
            self.write_selected::<D, C, W, G::Rest>(device, input, output);
        }
    }

    fn write_extent<D: MaterializableScalar, S: AxisName, C: AxisName, W: AxisName>(
        &self,
        device: &mut Device,
        input: &Split<D, N, C, W>,
        output: &mut Rows<D, N, C, W>,
    ) {
        const { assert!(N > 0 && W::SIZE > 0 && S::SIZE > 0 && S::SIZE <= C::SIZE) };
        type L<W, const N: usize> = Symbol<Part<W, N>>;
        // SAFETY: the first alias keeps the full physical extent and declares explicit outer
        // padding. The second names only that initialized contiguous prefix, preserving byte
        // order. Neither alias writes, and the original input owner stays borrowed throughout.
        let padded: PaddedRows<D, N, S, C, W> = unsafe { input.view().alias() };
        let compact: Rows<D, N, Pieces<S, N>, Part<W, N>> = unsafe { padded.view().alias() };
        // SAFETY: the compact parts are in chip order, each its rows in order.
        let source = unsafe {
            compact
                .view()
                .reshape::<ChipMap<N>, m![{ O<N> }, S, { L<W, N> }]>()
        };
        // SAFETY: this declares only outer padding; the physical extent and row-major byte
        // order remain C * W. The selected prefix keeps the original mutable borrow.
        let destination = unsafe { output.view_mut().reshape::<ChipMap<N>, m![S # C, W]>() };
        // SAFETY: a row's columns are the source chips' contiguous parts. Retain the outer
        // allocation padding so the view keeps the original destination's writable provenance.
        let destination = unsafe {
            destination.reshape::<m![{ O<N> }], m![S # C, { Symbol<Shard<N>> }, { L<W, N> }]>()
        };
        source.to_hbm_view(&mut device.tdma, destination);
    }
}

type Split<D, const N: usize, C, W> = Rows<D, N, Pieces<C, N>, Part<W, N>>;

type PaddedRows<D, const N: usize, S, C, W> = HbmTensor<
    D,
    ChipMap<N>,
    m![{ Symbol<Pieces<S, N>> } # ({ Symbol<Pieces<C, N>> }), { Symbol<Part<W, N>> }],
>;

/// Whether every DMA extent preserves a consumer's block alignment.
pub(crate) const fn aligned<G: Extents>(block: usize) -> bool {
    G::First::SIZE.is_multiple_of(block) && (G::LAST || aligned::<G::Rest>(block))
}

const fn valid<C: AxisName, G: Extents>() -> bool {
    G::First::SIZE > 0
        && G::First::SIZE <= C::SIZE
        && (G::LAST
            || (G::First::SIZE < <G::Rest as Extents>::First::SIZE && valid::<C, G::Rest>()))
}

const fn contains<G: Extents>(rows: usize) -> bool {
    rows == G::First::SIZE || (!G::LAST && contains::<G::Rest>(rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::axis::Rows;

    #[test]
    fn refuses_work_outside_the_transfer_extents_before_upload() {
        type C = Rows<1280>;
        type G = (Rows<256>, (Rows<768>, (Rows<1280>,)));
        for rows in [0, 512, 1024] {
            let prefix = BlockPrefix::<(), C, Rows<256>>::try_over(rows).unwrap();
            let error = Prefix::<(), C, G, Rows<256>>::new(prefix).err().unwrap();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(
                error.to_string(),
                "exchange work is not one of the fixed transfer extents"
            );
        }
    }
}
