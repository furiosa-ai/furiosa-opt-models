//! Row contractions over checked contiguous prefixes or runtime intervals of a tiled gather.

use super::tiled::{Pending, Strided};
use super::{
    B, Bs, Hr, InDm, Ks, Lines, Offset, Passes, Projection, Register, Result, Rounding, RowBlock,
    Sink, Slab, Source,
};
use crate::ops::collective::{Owners, Pieces, TileRows, Tiled};
use crate::tensor::live::{BlockPrefix, Live};
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M, Bias: Offset<I, O, R, Chips>, Out, W>
    Projection<'_, I, O, R, Chips, Bias, Out, W>
where
    W: Slab<I, O, R, Chips>,
{
    /// Contract only a checked prefix, keeping the allocation's row geometry and coefficient
    /// layout. Fixed alternating buffers preserve the static contraction pipeline with a runtime
    /// block count; an empty prefix touches neither source nor destination.
    pub(crate) fn prefix<D, LE, RE, X, Y, U>(
        &self,
        device: &mut Device,
        rows: &X,
        work: BlockPrefix<(), R, U>,
        output: &mut Y,
    ) where
        D: Rounding,
        LE: M,
        RE: M,
        X: Source<I, O, R, Chips, Order = LE>,
        Y: Sink<D, I, O, R, Chips, Order = RE>,
        U: AxisName,
    {
        const {
            assert!(U::SIZE == RowBlock::<R>::SIZE);
        }
        let blocks = work.blocks();
        if blocks == 0 {
            return;
        }
        #[unroll]
        for pass in 0..<Passes<I, O, R> as AxisName>::SIZE {
            let mut first: InDm<I, O, R, Chips, LE> = DmTensor::new();
            let mut second: InDm<I, O, R, Chips, LE> = DmTensor::new();
            rows.to_dm(device, 0, &mut first);
            let slab = self.weight.slab(device, pass);
            if blocks > 1 {
                rows.to_dm(device, 1, &mut second);
            }
            let offsets = self.bias.pass(device, pass);
            let mut kept: Result<D, I, O, R, Chips, RE> = DmTensor::new();
            let mut held: Result<D, I, O, R, Chips, RE> = DmTensor::new();
            let fill = |device: &mut Device, landed: &InDm<I, O, R, Chips, LE>| {
                device.sub.begin(landed.view())
                    .fetch::<m![{ B<R> } % 8, I % ({ Ks<I, O, R> }) / 4, { B<R> } % ({ Hr<R> }) / 32], m![{ B<R> } % 32 / 8, I % 4]>()
                    .collect::<m![{ B<R> } % 8, I % ({ Ks<I, O, R> }) / 4, { B<R> } % ({ Hr<R> }) / 32], m![{ B<R> } % 32 / 8, I % 4]>()
                    .to_trf()
            };
            let last = blocks - 1;
            let register = fill(device, &first);
            Bias::contract(device, (&slab, &offsets), &register, &mut kept);
            rows.to_dm(device, 2.min(last), &mut first);
            // Keep each buffer's role fixed through the loop, as in the static pipelined
            // contraction. Runtime parity chooses only the final store, not a destination
            // inside every iteration. The next contraction overlaps the previous store.
            for pair in 0..last / 2 {
                let block = 2 * pair + 1;
                let register = fill(device, &second);
                Bias::contract(device, (&slab, &offsets), &register, &mut held);
                rows.to_dm(device, (block + 2).min(last), &mut second);
                output.store(device, &kept, (block - 1, pass));
                let register = fill(device, &first);
                Bias::contract(device, (&slab, &offsets), &register, &mut kept);
                rows.to_dm(device, (block + 3).min(last), &mut first);
                output.store(device, &held, (block, pass));
            }
            if blocks % 2 == 0 {
                let register = fill(device, &second);
                Bias::contract(device, (&slab, &offsets), &register, &mut held);
                output.store(device, &kept, (last - 1, pass));
                output.store(device, &held, (last, pass));
            } else {
                output.store(device, &kept, (last, pass));
            }
        }
    }

    /// Only the blocks touching each part's interval are contracted. Parts retain their capacity
    /// stride, including padding at either boundary of a partial block. A tile's final contraction
    /// prefetches the next active part/tile before its result is stored.
    pub(crate) fn tiled_live<
        'r,
        D,
        RE,
        X,
        Y,
        S,
        const P: usize,
        const T: usize,
        const K: usize,
        const HEAD: usize,
    >(
        &self,
        device: &mut Device,
        rows: &'r Tiled<X, R, P, T, K, HEAD>,
        interval: Live<(), S>,
        output: &mut Y,
    ) where
        D: Rounding,
        RE: M,
        S: AxisName,
        Strided<'r, X>: Source<I, O, R, Chips, Order = Lines<I, O, R>>,
        Y: Sink<D, I, O, R, Chips, Order = RE>,
    {
        self.tiled_live_with::<D, RE, X, Y, S, _, P, T, K, HEAD>(
            device,
            rows,
            interval,
            output,
            |part, block| part * (S::SIZE / RowBlock::<R>::SIZE) + block,
        );
    }

    /// Store initialized, block-aligned prefixes consecutively, without copying the results
    /// through an intermediate HBM tensor. The input parts keep their original capacity stride.
    pub(crate) fn tiled_prefix<
        'r,
        D,
        RE,
        X,
        Y,
        S,
        U,
        const P: usize,
        const T: usize,
        const K: usize,
        const HEAD: usize,
    >(
        &self,
        device: &mut Device,
        rows: &'r Tiled<X, R, P, T, K, HEAD>,
        work: BlockPrefix<(), S, U>,
        output: &mut Y,
    ) where
        D: Rounding,
        RE: M,
        S: AxisName,
        U: AxisName,
        Strided<'r, X>: Source<I, O, R, Chips, Order = Lines<I, O, R>>,
        Y: Sink<D, I, O, R, Chips, Order = RE>,
    {
        const {
            assert!(U::SIZE == RowBlock::<R>::SIZE);
        }
        let blocks = work.blocks();
        self.tiled_live_with::<D, RE, X, Y, S, _, P, T, K, HEAD>(
            device,
            rows,
            work.interval(),
            output,
            |part, block| part * blocks + block,
        );
    }

    fn tiled_live_with<
        'r,
        D,
        RE,
        X,
        Y,
        S,
        Place,
        const P: usize,
        const T: usize,
        const K: usize,
        const HEAD: usize,
    >(
        &self,
        device: &mut Device,
        rows: &'r Tiled<X, R, P, T, K, HEAD>,
        interval: Live<(), S>,
        output: &mut Y,
        place: Place,
    ) where
        D: Rounding,
        RE: M,
        S: AxisName,
        Place: Fn(usize, usize) -> usize,
        Strided<'r, X>: Source<I, O, R, Chips, Order = Lines<I, O, R>>,
        Y: Sink<D, I, O, R, Chips, Order = RE>,
    {
        const {
            assert!(P > 0 && K > 0 && R::SIZE == P * S::SIZE);
            assert!(HEAD == RowBlock::<R>::SIZE);
            assert!(T.is_multiple_of(RowBlock::<R>::SIZE));
            assert!(S::SIZE.is_multiple_of(RowBlock::<R>::SIZE));
            assert!(RowBlock::<S>::SIZE == RowBlock::<R>::SIZE);
            assert!(HEAD + (K - 1) * T == S::SIZE);
        };
        if interval.rows() == 0 {
            return;
        }
        let block = RowBlock::<R>::SIZE;
        let tile_blocks = T / block;
        let (start, count) = interval.groups::<RowBlock<R>>();
        let end = start + count;
        #[unroll]
        for pass in 0..<Passes<I, O, R> as AxisName>::SIZE {
            let slab = self.weight.slab(device, pass);
            let offsets = self.bias.pass(device, pass);
            let mut first: InDm<I, O, R, Chips, Lines<I, O, R>> = DmTensor::new();
            let mut second: InDm<I, O, R, Chips, Lines<I, O, R>> = DmTensor::new();
            // Fixed alternating destinations keep tensor ownership out of conditional merges.
            // The next unit stores the previous destination under its first contraction.
            let mut even: Result<D, I, O, R, Chips, RE> = DmTensor::new();
            let mut odd: Result<D, I, O, R, Chips, RE> = DmTensor::new();
            let mut store = |device: &mut Device, kept: &Result<D, I, O, R, Chips, RE>, at| {
                output.store(device, kept, (at, pass))
            };
            // Unrolled so the initial consumer depends on its own allocation, not later tiles.
            #[unroll]
            for tile in 0..K {
                let begin = if tile == 0 {
                    0
                } else {
                    1 + (tile - 1) * tile_blocks
                };
                let limit = begin + if tile == 0 { 1 } else { tile_blocks };
                let first_block = start.max(begin);
                let stop = end.min(limit);
                if stop > first_block {
                    let blocks = stop - first_block;
                    let source = Strided::contiguous(&rows.0[tile].0);
                    let previous_exists = first_block > start;
                    let previous_block = place(P - 1, begin.max(1) - 1);
                    let next_tile = (tile + 1).min(K - 1);
                    let next_begin = if next_tile == 0 {
                        0
                    } else {
                        1 + (next_tile - 1) * tile_blocks
                    };
                    let next_limit = next_begin + if next_tile == 0 { 1 } else { tile_blocks };
                    let next_first = start.max(next_begin);
                    let next_stop = end.min(next_limit);
                    let following_blocks = if next_stop > next_first {
                        next_stop - next_first
                    } else {
                        0
                    };
                    let following_source = Strided::contiguous(&rows.0[next_tile].0);
                    let following = Window {
                        source: &following_source,
                        first: next_first - next_begin,
                    };
                    let following_count = if following_blocks == tile_blocks {
                        P * following_blocks
                    } else {
                        following_blocks
                    };
                    let following_exists = tile + 1 < K && following_blocks > 0;
                    let mut run =
                        |device: &mut Device,
                         result: &mut Result<D, I, O, R, Chips, RE>,
                         previous: &Result<D, I, O, R, Chips, RE>| {
                            let mut next = |device: &mut Device,
                                            b,
                                            landed: &mut InDm<I, O, R, Chips, Lines<I, O, R>>| {
                                if following_exists && b < following_count {
                                    following.to_dm(device, b, landed);
                                }
                            };
                            macro_rules! project {
                                ($device:expr, $rows:expr, $place:expr, $bound:ty) => {{
                                    let rows = &$rows;
                                    let count = <$bound>::SIZE;
                                    if !previous_exists {
                                        rows.to_dm($device, 0, &mut first);
                                        if count > 1 {
                                            rows.to_dm($device, 1, &mut second);
                                        }
                                    }
                                    trips_into::<_, _, _, _, _, _, _, _, _, _, _, _, _, _>(
                                        $device,
                                        (rows, $place, PhantomData::<$bound>),
                                        (&mut first, &mut second),
                                        result,
                                        Deferred {
                                            result: previous,
                                            block: previous_block,
                                            ready: previous_exists,
                                        },
                                        &mut next,
                                        |device, register, kept| {
                                            Bias::contract(
                                                device,
                                                (&slab, &offsets),
                                                register,
                                                kept,
                                            )
                                        },
                                        &mut store,
                                    );
                                }};
                            }
                            if tile == 0 {
                                let head = Strided::with_stride(&rows.0[tile].0, tile_blocks);
                                project!(
                                    device,
                                    head,
                                    |part| place(part, 0),
                                    m![{ Symbol<Owners<P>> }]
                                );
                            } else if blocks == tile_blocks {
                                // A complete tile is the reference's contiguous P-part run.
                                project!(
                                    device,
                                    source,
                                    |b| place(b / tile_blocks, begin + b % tile_blocks),
                                    Bs<Pieces<TileRows<T>, P>>
                                );
                            } else {
                                // Partial parts retain the established Live address and contraction
                                // path; the paired schedule above needs only static full runs.
                                let mut part_even: Result<D, I, O, R, Chips, RE> = DmTensor::new();
                                let mut part_odd: Result<D, I, O, R, Chips, RE> = DmTensor::new();
                                #[unroll]
                                for part in 0..P {
                                    let window = Window {
                                        source: &source,
                                        first: part * tile_blocks + first_block - begin,
                                    };
                                    let next_source = Strided::contiguous(
                                        &rows.0[if part + 1 < P { tile } else { next_tile }].0,
                                    );
                                    let next_window = Window {
                                        source: &next_source,
                                        first: if part + 1 < P {
                                            (part + 1) * tile_blocks + first_block - begin
                                        } else {
                                            next_first - next_begin
                                        },
                                    };
                                    let next_count = if part + 1 < P {
                                        blocks
                                    } else {
                                        following_count
                                    };
                                    let has_next = part + 1 < P || following_exists;
                                    let mut next_part = |device: &mut Device,
                                                         b,
                                                         landed: &mut InDm<
                                        I,
                                        O,
                                        R,
                                        Chips,
                                        Lines<I, O, R>,
                                    >| {
                                        if has_next && b < next_count {
                                            next_window.to_dm(device, b, landed);
                                        }
                                    };
                                    let mut part_run = |device: &mut Device,
                                                        part_result: &mut Result<
                                        D,
                                        I,
                                        O,
                                        R,
                                        Chips,
                                        RE,
                                    >,
                                                        part_previous: &Result<
                                        D,
                                        I,
                                        O,
                                        R,
                                        Chips,
                                        RE,
                                    >| {
                                        let deferred = Deferred {
                                            result: part_previous,
                                            block: if part > 0 {
                                                place(part - 1, stop - 1)
                                            } else {
                                                previous_block
                                            },
                                            ready: part > 0 || previous_exists,
                                        };
                                        if part == 0 && !previous_exists {
                                            window.to_dm(device, 0, &mut first);
                                            if blocks > 1 {
                                                window.to_dm(device, 1, &mut second);
                                            }
                                        }
                                        trips_partial(
                                            device,
                                            (
                                                &window,
                                                |b| place(part, first_block + b),
                                                PhantomData::<Bs<crate::tensor::axis::Rows<T>>>,
                                                blocks,
                                            ),
                                            (&mut first, &mut second),
                                            part_result,
                                            deferred,
                                            &mut next_part,
                                            |device, register, kept| {
                                                Bias::contract(
                                                    device,
                                                    (&slab, &offsets),
                                                    register,
                                                    kept,
                                                )
                                            },
                                            &mut store,
                                        );
                                    };
                                    if part + 1 == P {
                                        if part == 0 {
                                            part_run(device, result, previous);
                                        } else if part % 2 == 1 {
                                            part_run(device, result, &part_even);
                                        } else {
                                            part_run(device, result, &part_odd);
                                        }
                                    } else if part % 2 == 0 {
                                        if part == 0 {
                                            part_run(device, &mut part_even, previous);
                                        } else {
                                            part_run(device, &mut part_even, &part_odd);
                                        }
                                    } else {
                                        part_run(device, &mut part_odd, &part_even);
                                    }
                                }
                            }
                            if !following_exists {
                                store(device, result, place(P - 1, stop - 1));
                            }
                        };
                    // Fixed destinations preserve the pending result across active tiles.
                    if tile % 2 == 0 {
                        run(device, &mut even, &odd);
                    } else {
                        run(device, &mut odd, &even);
                    }
                }
            }
        }
    }
}

/// A live block window inside one part of an independently arriving tile.
struct Window<'a, X> {
    source: &'a X,
    first: usize,
}

impl<I: AxisName, O: AxisName, R: AxisName, Chips: M, X: Source<I, O, R, Chips>>
    Source<I, O, R, Chips> for Window<'_, X>
{
    type Order = X::Order;

    fn to_dm(
        &self,
        device: &mut Device,
        block: usize,
        landed: &mut InDm<I, O, R, Chips, Self::Order>,
    ) {
        self.source.to_dm(device, self.first + block, landed);
    }
}

/// A prior result exists only after the first active part. Its storage survives until the next
/// contraction has accepted all its inputs, and is never observed before it is initialized.
struct Deferred<'a, D: Scalar, I: AxisName, O: AxisName, R: AxisName, Chips: M, E: M> {
    result: &'a Result<D, I, O, R, Chips, E>,
    block: usize,
    ready: bool,
}

impl<D: Scalar, I: AxisName, O: AxisName, R: AxisName, Chips: M, E: M> Pending<D, I, O, R, Chips, E>
    for Deferred<'_, D, I, O, R, Chips, E>
{
    fn store<Store>(self, device: &mut Device, store: &mut Store)
    where
        Store: FnMut(&mut Device, &Result<D, I, O, R, Chips, E>, usize),
    {
        if self.ready {
            store(device, self.result, self.block);
        }
    }
}

/// The reference's fixed paired schedule for a full or head tile. Its final result remains
/// in caller-owned storage so runtime tile selection does not merge tensor ownership.
fn trips_into<Blocks, I, O, R, Chips, D, LE, RE, S, Place, P, Next, Contract, Store>(
    device: &mut Device,
    (rows, place, _): (&S, Place, PhantomData<Blocks>),
    (first, second): (&mut InDm<I, O, R, Chips, LE>, &mut InDm<I, O, R, Chips, LE>),
    last: &mut Result<D, I, O, R, Chips, RE>,
    pending: P,
    mut next: Next,
    mut contract: Contract,
    store: &mut Store,
) where
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
    let blocks = Blocks::SIZE;
    let mut other: Result<D, I, O, R, Chips, RE> = DmTensor::new();
    let (kept, held) = if const { Blocks::SIZE % 2 == 1 } {
        (last, &mut other)
    } else {
        (&mut other, last)
    };
    let end = blocks - 1;
    let register = fill(device, first);
    contract(device, &register, kept);
    pending.store(device, store);
    rows.to_dm(device, 2.min(end), first);
    for pair in 0..end / 2 {
        let block = 2 * pair + 1;
        let register = fill(device, second);
        contract(device, &register, held);
        rows.to_dm(device, (block + 2).min(end), second);
        store(device, kept, place(block - 1));
        let register = fill(device, first);
        contract(device, &register, kept);
        rows.to_dm(device, (block + 3).min(end), first);
        store(device, held, place(block));
    }
    if const { Blocks::SIZE % 2 == 0 } {
        let register = fill(device, second);
        contract(device, &register, held);
        store(device, kept, place(end - 1));
    }
    next(device, 0, first);
    next(device, 1, second);
}

/// Contract a nonempty bounded run into fixed destinations. The final result stays in `last`
/// for the following run to store; intermediate results alternate while stores overlap compute.
fn trips_partial<Blocks, I, O, R, Chips, D, LE, RE, S, Place, P, Next, Contract, Store>(
    device: &mut Device,
    (rows, place, _, blocks): (&S, Place, PhantomData<Blocks>, usize),
    (first, second): (&mut InDm<I, O, R, Chips, LE>, &mut InDm<I, O, R, Chips, LE>),
    last: &mut Result<D, I, O, R, Chips, RE>,
    pending: P,
    mut next: Next,
    mut contract: Contract,
    store: &mut Store,
) where
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
    let blocks = blocks.min(Blocks::SIZE);
    let mut kept: Result<D, I, O, R, Chips, RE> = DmTensor::new();
    let mut held: Result<D, I, O, R, Chips, RE> = DmTensor::new();
    let register = fill(device, first);
    if blocks == 1 {
        contract(device, &register, last);
    } else {
        contract(device, &register, &mut kept);
    }
    pending.store(device, store);
    if blocks > 2 {
        rows.to_dm(device, 2, first);
    }
    for block in 1..blocks {
        if block % 2 == 1 {
            let register = fill(device, second);
            if block + 1 == blocks {
                contract(device, &register, last);
            } else {
                contract(device, &register, &mut held);
            }
            if block + 2 < blocks {
                rows.to_dm(device, block + 2, second);
            }
            store(device, &kept, place(block - 1));
        } else {
            let register = fill(device, first);
            if block + 1 == blocks {
                contract(device, &register, last);
            } else {
                contract(device, &register, &mut kept);
            }
            if block + 2 < blocks {
                rows.to_dm(device, block + 2, first);
            }
            store(device, &held, place(block - 1));
        }
    }
    next(device, 0, first);
    next(device, 1, second);
}
