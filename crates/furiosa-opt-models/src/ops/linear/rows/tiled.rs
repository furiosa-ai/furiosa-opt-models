//! Row contractions over independently arriving collective tiles.
//!
//! Adapted from the repro's tiled grouped projection. The contraction is shared
//! with ordinary row projections; tile boundaries retain the pending result.

use core::marker::PhantomData;

use super::{
    B, Blocked, Bs, Cluster, Hr, InDm, Ks, Lines, Np, Offset, Parts, Passes, Projection, Register,
    Result, Rounding, RowBlock, Rows, Sink, Slab, Source,
};
use crate::ops::collective::{Owners, Pieces, TileRows, Tiled};
use furiosa_opt_std::prelude::*;

/// A whole HBM write unit, in bytes.
const UNIT: usize = 256;

/// A cluster's block rows, one row per slice.
type RowSlices<R> = m![{ B<R> } % 4, { B<R> } % ({ Hr<R> }) / 4 # 64];
/// Whole pass rows staged before splitting query, key and value columns.
type Staged<D, I, O, R, Chips> =
    DmTensor<D, Chips, Cluster<R>, RowSlices<R>, m![O % ({ Np<I, O, R> })]>;

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M, Bias: Offset<I, O, R, Chips>, Out, W>
    Projection<'_, I, O, R, Chips, Bias, Out, W>
where
    W: Slab<I, O, R, Chips>,
{
    /// Every pass over rows that arrive [`InTiles`](crate::ops::collective::InTiles), the
    /// pass's slab landed once: the first tile's `HEAD` rows of each part, one block a part's slot,
    /// then each whole tile's blocks, each block into its own rows of `output`. A tile's first
    /// landing waits for its transfer alone, under the last contraction of the tile before it.
    pub(crate) fn tiled<
        'r,
        D,
        RE,
        X,
        Y,
        const P: usize,
        const T: usize,
        const K: usize,
        const HEAD: usize,
    >(
        &self,
        device: &mut Device,
        (rows, output): (&'r Tiled<X, R, P, T, K, HEAD>, &mut Y),
    ) where
        D: Rounding,
        RE: M,
        Strided<'r, X>: Source<I, O, R, Chips, Order = Lines<I, O, R>>,
        Y: Sink<D, I, O, R, Chips, Order = RE>,
    {
        const {
            let block = <RowBlock<R> as AxisName>::SIZE;
            assert!(K >= 2, "a whole tile follows the first");
            assert!(HEAD == block, "the first tile is a block a part");
            assert!(
                T.is_multiple_of(block) && (R::SIZE / P).is_multiple_of(block),
                "a tile holds whole blocks of each part"
            );
        };
        let (tile_blocks, part_blocks) = (
            T / <RowBlock<R> as AxisName>::SIZE,
            R::SIZE / P / <RowBlock<R> as AxisName>::SIZE,
        );
        // Whole tile `t` holds blocks `1 + t * tile_blocks..` of every part, part after part.
        let at = |t: usize, b: usize| {
            b / tile_blocks * part_blocks + 1 + t * tile_blocks + b % tile_blocks
        };
        // Unrolled: a device loop's captured tensors are published at its entry, so every tile
        // would wait ahead of the first.
        #[unroll]
        for pass in 0..<Passes<I, O, R> as AxisName>::SIZE {
            let slab = self.weight.slab(device, pass);
            let offsets = self.bias.pass(device, pass);
            let source = Strided {
                rows: &rows.0[0].0,
                slot: tile_blocks,
            };
            let mut first: InDm<I, O, R, Chips, Lines<I, O, R>> = DmTensor::new();
            let mut second: InDm<I, O, R, Chips, Lines<I, O, R>> = DmTensor::new();
            source.to_dm(device, 0, &mut first);
            if const { P > 1 } {
                source.to_dm(device, 1, &mut second);
            }
            let mut store = |device: &mut Device, kept: &Result<D, I, O, R, Chips, RE>, block| {
                output.store(device, kept, (block, pass))
            };
            let following = Strided {
                rows: &rows.0[1].0,
                slot: 1,
            };
            let (mut landed, mut pending) = trips(
                device,
                (
                    &source,
                    |b| b * part_blocks,
                    PhantomData::<m![{ Symbol<Owners<P>> }]>,
                ),
                (first, second),
                (),
                |device: &mut Device, b, landed: &mut InDm<I, O, R, Chips, Lines<I, O, R>>| {
                    following.to_dm(device, b, landed)
                },
                |device, register, kept| Bias::contract(device, (&slab, &offsets), register, kept),
                &mut store,
            );
            #[unroll]
            for t in 0..K - 2 {
                let (source, following) = (
                    Strided {
                        rows: &rows.0[t + 1].0,
                        slot: 1,
                    },
                    Strided {
                        rows: &rows.0[t + 2].0,
                        slot: 1,
                    },
                );
                (landed, pending) = trips(
                    device,
                    (
                        &source,
                        |b| at(t, b),
                        PhantomData::<Bs<Pieces<TileRows<T>, P>>>,
                    ),
                    landed,
                    pending,
                    |device: &mut Device, b, landed: &mut InDm<I, O, R, Chips, Lines<I, O, R>>| {
                        following.to_dm(device, b, landed)
                    },
                    |device, register, kept| {
                        Bias::contract(device, (&slab, &offsets), register, kept)
                    },
                    &mut store,
                );
            }
            let source = Strided {
                rows: &rows.0[K - 1].0,
                slot: 1,
            };
            let (_, (last, block)) = trips(
                device,
                (
                    &source,
                    |b| at(K - 2, b),
                    PhantomData::<Bs<Pieces<TileRows<T>, P>>>,
                ),
                landed,
                pending,
                |_: &mut Device, _, _: &mut InDm<I, O, R, Chips, Lines<I, O, R>>| {},
                |device, register, kept| Bias::contract(device, (&slab, &offsets), register, kept),
                &mut store,
            );
            store(device, &last, block);
        }
    }
}

/// A tile's rows as blocks of the rows `R` they are part of, one block in every `slot` holding:
/// block `b` lands from the tile's block `slot * b`.
pub(crate) struct Strided<'a, X> {
    rows: &'a X,
    slot: usize,
}

impl<'a, X> Strided<'a, X> {
    pub(super) fn contiguous(rows: &'a X) -> Self {
        Self { rows, slot: 1 }
    }

    /// Logical block `b` is physical block `slot * b`, as in the first gathered tile.
    pub(super) fn with_stride(rows: &'a X, slot: usize) -> Self {
        Self { rows, slot }
    }
}

impl<I: AxisName, O: AxisName, R: AxisName, L: AxisName, Chips: M> Source<I, O, R, Chips>
    for Strided<'_, HbmTensor<bf16, Chips, m![L, I]>>
{
    type Order = Lines<I, O, R>;

    fn to_dm(
        &self,
        device: &mut Device,
        block: usize,
        landed: &mut InDm<I, O, R, Chips, Self::Order>,
    ) {
        const {
            assert!(
                <RowBlock<L> as AxisName>::SIZE == <RowBlock<R> as AxisName>::SIZE,
                "a tile's blocks are not the rows' blocks"
            )
        };
        // SAFETY: the blocks split the tile's rows in order; every cell keeps its address.
        unsafe {
            self.rows
                .view()
                .reshape::<Chips, m![{ Bs<L> }, { B<R> }, I]>()
        }
        .tile::<m![{ Bs<L> }], 1, m![{ Bs<L> } = 1 # ({ Bs<L> }), { B<R> }, I]>(self.slot * block)
        .to_dm_view(&mut device.tdma, landed.view_mut());
    }
}

/// Rows whose outputs are three parts of `W` columns in order, each part rows of its own: a block's
/// rows gather into a pass row a slice, and each part of it leaves for its own rows.
impl<'a, D, I, O, R, Chips, W> Sink<D, I, O, R, Chips>
    for (
        &'a mut HbmTensor<D, Chips, m![R, W]>,
        &'a mut HbmTensor<D, Chips, m![R, W]>,
        &'a mut HbmTensor<D, Chips, m![R, W]>,
    )
where
    D: MaterializableScalar,
    I: AxisName,
    O: AxisName,
    R: AxisName,
    Chips: M,
    W: AxisName,
{
    type Order = Rows<I, O, R>;

    fn store(
        &mut self,
        device: &mut Device,
        kept: &Result<D, I, O, R, Chips, Self::Order>,
        (block, _): (usize, usize),
    ) {
        const {
            assert!(
                O::SIZE == 3 * W::SIZE && <Passes<I, O, R> as AxisName>::SIZE == 1,
                "one pass holds the three parts"
            );
            assert!(
                (W::SIZE * D::BITS / 8).is_multiple_of(UNIT),
                "a part's row ends inside an HBM unit"
            );
        };
        let mut rows: Staged<D, I, O, R, Chips> = DmTensor::new();
        kept.view().to_dm_view(&mut device.tdma, rows.view_mut());
        let leave = |device: &mut Device,
                     part: usize,
                     output: &mut HbmTensor<D, Chips, m![R, W]>| {
            // SAFETY: a row's outputs are its three parts in order.
            let parts = unsafe {
                rows.view()
                    .reshape::<Chips, Cluster<R>, RowSlices<R>, m![Parts, W]>()
            };
            // SAFETY: as for the input.
            let tile = unsafe { output.view_mut().reshape::<Chips, Blocked<R, W>>() }
                .tile::<m![{ Bs<R> }], 1, m![{ Bs<R> } = 1 #{!} ({ Bs<R> }), { B<R> }, W]>(block);
            parts
                .tile::<m![Parts], 1, m![Parts = 1 # 3, W]>(part)
                .to_hbm_view(&mut device.tdma, tile);
        };
        leave(device, 0, &mut *self.0);
        leave(device, 1, &mut *self.1);
        leave(device, 2, &mut *self.2);
    }
}

/// A result a trips stores once its first contraction is issued: nothing, or the last result of
/// the trips before it and its block of the rows, so the tensor unit runs on from one to the next.
pub(super) trait Pending<D: Scalar, I: AxisName, O: AxisName, R: AxisName, Chips: M, E: M> {
    fn store<Store>(self, device: &mut Device, store: &mut Store)
    where
        Store: FnMut(&mut Device, &Result<D, I, O, R, Chips, E>, usize);
}

impl<D: Scalar, I: AxisName, O: AxisName, R: AxisName, Chips: M, E: M> Pending<D, I, O, R, Chips, E>
    for ()
{
    fn store<Store>(self, _: &mut Device, _: &mut Store)
    where
        Store: FnMut(&mut Device, &Result<D, I, O, R, Chips, E>, usize),
    {
    }
}

impl<D: Scalar, I: AxisName, O: AxisName, R: AxisName, Chips: M, E: M> Pending<D, I, O, R, Chips, E>
    for (Result<D, I, O, R, Chips, E>, usize)
{
    fn store<Store>(self, device: &mut Device, store: &mut Store)
    where
        Store: FnMut(&mut Device, &Result<D, I, O, R, Chips, E>, usize),
    {
        let (result, block) = self;
        store(device, &result, block);
    }
}

/// Blocks `0..Blocks` of `rows`, blocks 0 and 1 landed in `first` and `second`: each block fills
/// while the one before it contracts, and is stored at its block of the rows (`place`) while the
/// one after it contracts. `pending` is stored once the first contraction is issued, and once the
/// last is issued `next` lands blocks 0 and 1 of the rows that follow. Returns those two landings
/// and the last result, not yet stored, with its block of the rows.
pub(super) fn trips<Blocks, I, O, R, Chips, D, LE, RE, S, Place, P, Next, Contract, Store>(
    device: &mut Device,
    (rows, place, _): (&S, Place, PhantomData<Blocks>),
    (mut first, mut second): (InDm<I, O, R, Chips, LE>, InDm<I, O, R, Chips, LE>),
    pending: P,
    mut next: Next,
    mut contract: Contract,
    store: &mut Store,
) -> (
    (InDm<I, O, R, Chips, LE>, InDm<I, O, R, Chips, LE>),
    (Result<D, I, O, R, Chips, RE>, usize),
)
where
    Blocks: M,
    I: AxisName,
    O: AxisName,
    R: AxisName,
    Chips: M,
    D: Scalar,
    LE: M,
    RE: M,
    S: Source<I, O, R, Chips, Order = LE>,
    Place: Fn(usize) -> usize,
    P: Pending<D, I, O, R, Chips, RE>,
    Next: FnMut(&mut Device, usize, &mut InDm<I, O, R, Chips, LE>),
    Contract: FnMut(&mut Device, &Register<I, O, R, Chips>, &mut Result<D, I, O, R, Chips, RE>),
    Store: FnMut(&mut Device, &Result<D, I, O, R, Chips, RE>, usize),
{
    let fill = |device: &mut Device, landed: &InDm<I, O, R, Chips, LE>| {
        device.sub.begin(landed.view())
            .fetch::<m![{ B<R> } % 8, I % ({ Ks<I, O, R> }) / 4, { B<R> } % ({ Hr<R> }) / 32], m![{ B<R> } % 32 / 8, I % 4]>()
            .collect::<m![{ B<R> } % 8, I % ({ Ks<I, O, R> }) / 4, { B<R> } % ({ Hr<R> }) / 32], m![{ B<R> } % 32 / 8, I % 4]>()
            .to_trf()
    };
    let last = Blocks::SIZE - 1;
    // Two results, so that a block contracts while the block before it is stored.
    let mut kept: Result<D, I, O, R, Chips, RE> = DmTensor::new();
    let mut held: Result<D, I, O, R, Chips, RE> = DmTensor::new();
    let register = fill(device, &first);
    contract(device, &register, &mut kept);
    pending.store(device, store);
    rows.to_dm(device, 2.min(last), &mut first);
    for pair in 0..last / 2 {
        let block = 2 * pair + 1;
        let next = fill(device, &second);
        contract(device, &next, &mut held);
        rows.to_dm(device, (block + 2).min(last), &mut second);
        store(device, &kept, place(block - 1));
        let register = fill(device, &first);
        contract(device, &register, &mut kept);
        rows.to_dm(device, (block + 3).min(last), &mut first);
        store(device, &held, place(block));
    }
    if const { Blocks::SIZE % 2 == 0 } {
        let register = fill(device, &second);
        contract(device, &register, &mut held);
        store(device, &kept, place(last - 1));
        next(device, 0, &mut first);
        next(device, 1, &mut second);
        ((first, second), (held, place(last)))
    } else {
        next(device, 0, &mut first);
        next(device, 1, &mut second);
        ((first, second), (kept, place(last)))
    }
}
