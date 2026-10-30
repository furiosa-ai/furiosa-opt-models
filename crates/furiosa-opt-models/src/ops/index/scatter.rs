//! Rows stored into HBM: whole from their projection feature slices, or by index into slots.

use crate::Op;
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::modules::generic::schedule::{IntoStream, RowWise, Stream};
use crate::ops::attention;
use crate::ops::index::gather::RowStorage;
use crate::ops::index::table::{self, Access, Length, Offsets};
use crate::ops::normalization::Chunk;
use crate::ops::normalization::{RowMajor, Spread};
use crate::ops::reduction::Selected;
use crate::specialize::index::store;
use crate::tensor::cache::Blocks;
use crate::tensor::cache::{Block, KeyValue};
use crate::tensor::{
    axis::{self, Lanes, Single, Tile},
    live::Live,
    mapping::Rows as Layout,
    parallel::Chip,
};
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;
use std::marker::PhantomData;

type HiddenRows<Chips, Width, Rows> = HbmTensor<bf16, Chips, RowMajor<Width, Rows>>;

/// Store rows in HBM: projection feature slices as contiguous BF16 rows, a selection's tokens, or
/// a tensor or a tile of rows into storage its caller owns.
#[derive(Default, DeviceSend, Load)]
pub struct Store;

/// The features of every row one stored slice holds: those of sixteen feature slices.
#[derive(Clone, Debug)]
struct Span<Width>(PhantomData<Width>);

impl<Width: AxisName> AxisName for Span<Width> {
    const NAME: Ident = Ident::new("Span");
    const SIZE: usize = 16 * <Chunk<Width> as AxisName>::SIZE;
}

type Spanned<Width> = Symbol<Span<Width>>;

/// Rows in the slices a store transfers whole: each slice holds a span of every row.
type Stored<Chips, Width, Rows> = DmTensor<
    bf16,
    Chips,
    m![2],
    m![Width / ({ Spanned<Width> }) # 16, 1 # 16],
    m![Rows, Width % ({ Spanned<Width> })],
>;

impl Store {
    fn rows<Width: AxisName, Rows: AxisName, Chips: M + 'static>(
        device: &mut Device,
        input: &Spread<Width, Rows, Chips>,
    ) -> Stored<Chips, Width, Rows> {
        type Slice<Width> = Symbol<Chunk<Width>>;
        if const { <Chunk<Width> as AxisName>::SIZE == 16 } {
            return device
                .main
                .begin(input.view())
                .fetch::<m![Rows], m![Width % 16]>()
                .switch::<m![Width / ({ Spanned<Width> }) # 16, 1 # 16], m![
                    Rows,
                    Width % ({ Spanned<Width> }) / 16
                ]>(SwitchConfig::Broadcast1 {
                    slice1: 16,
                    slice0: 1,
                })
                .collect::<m![Rows, Width % ({ Spanned<Width> }) / 16], m![Width % 16]>()
                .commit_trim::<m![Width % 16]>()
                .commit();
        }
        // A slice holds several switch packets, and the switch delivers the other slices'
        // packets innermost, so the rows are written in feature order through a view.
        let mut stored: Stored<Chips, Width, Rows> = DmTensor::new();
        device
            .main
            .begin(input.view())
            .fetch::<m![Rows, Width % ({ Slice<Width> }) / 16], m![Width % 16]>()
            .switch::<m![Width / ({ Spanned<Width> }) # 16, 1 # 16], m![
                Rows,
                Width % ({ Slice<Width> }) / 16,
                Width % ({ Spanned<Width> }) / ({ Slice<Width> })
            ]>(SwitchConfig::Broadcast1 {
                slice1: 16,
                slice0: 1,
            })
            .collect::<m![
                Rows,
                Width % ({ Slice<Width> }) / 16,
                Width % ({ Spanned<Width> }) / ({ Slice<Width> })
            ], m![Width % 16]>()
            .commit_trim::<m![Width % 16]>()
            .commit_view(stored.view_mut());
        stored
    }
}

impl<Width: AxisName, Rows: AxisName, Chips: M + 'static> Op<&Spread<Width, Rows, Chips>>
    for Store
{
    type Output = HiddenRows<Chips, Width, Rows>;

    fn forward(&self, device: &mut Device, input: &Spread<Width, Rows, Chips>) -> Self::Output {
        let mut output = HbmTensor::new();
        Self::rows(device, input)
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
        output
    }
}

/// Only the live rows: every row in one transfer when all are live, else each whole packet in
/// one, then each row of a partial last packet alone; a later row of the output is left unwritten.
impl<Width: AxisName, Rows: AxisName, Chips: M + 'static>
    Op<(
        (&Spread<Width, Rows, Chips>, Live<(), Rows>),
        &mut HiddenRows<Chips, Width, Rows>,
    )> for Store
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        ((input, live), output): (
            (&Spread<Width, Rows, Chips>, Live<(), Rows>),
            &mut HiddenRows<Chips, Width, Rows>,
        ),
    ) {
        let rows = Self::rows(device, input);
        live.cover::<Lanes<Rows>, _>(
            &mut (device, output),
            (
                |(device, output)| {
                    rows.view().to_hbm_view(&mut device.tdma, output.view_mut());
                },
                |(device, output), packet| {
                    rows.view()
                        .tile::<
                            Layout<Rows>::Groups,
                            1,
                            Layout<Rows>::Elements<m![Width % ({ Spanned<Width> })]>,
                        >(
                            packet,
                        )
                        .to_hbm_view(
                            &mut device.tdma,
                            output
                                .view_mut()
                                .tile::<Layout<Rows>::Groups, 1, Layout<Rows>::GroupMut<m![Width]>>(
                                    packet,
                                ),
                        );
                },
                |(device, output), row| {
                    rows.view()
                        .tile::<m![Rows], 1, m![Rows = 1 # Rows, Width % ({ Spanned<Width> })]>(row)
                        .to_hbm_view(
                            &mut device.tdma,
                            output
                                .view_mut()
                                .tile::<m![Rows], 1, m![Rows = 1 #{!} Rows, Width]>(row),
                        );
                },
            ),
        );
    }
}

/// Only the live rows, into a new place.
impl<Width: AxisName, Rows: AxisName, Chips: M + 'static>
    Op<(&Spread<Width, Rows, Chips>, Live<(), Rows>)> for Store
{
    type Output = HiddenRows<Chips, Width, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        input: (&Spread<Width, Rows, Chips>, Live<(), Rows>),
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        self.forward(device, (input, &mut output));
        output
    }
}

impl<Width: AxisName, Rows: AxisName, Chips: M + 'static>
    Op<&Live<Spread<Width, Rows, Chips>, Rows>> for Store
{
    type Output = HiddenRows<Chips, Width, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        input: &Live<Spread<Width, Rows, Chips>, Rows>,
    ) -> Self::Output {
        self.forward(device, (input.value(), input.with(())))
    }
}

/// An HBM tensor into storage its caller owns, element for element.
impl<D: MaterializableScalar, Chips: M, E: M>
    Op<(&HbmTensor<D, Chips, E>, &mut HbmTensor<D, Chips, E>)> for Store
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, output): (&HbmTensor<D, Chips, E>, &mut HbmTensor<D, Chips, E>),
    ) {
        input
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }
}

/// Tile `at` of an envelope's rows, `N` rows a tile, into storage its caller owns.
impl<D: MaterializableScalar, R: AxisName, S: AxisName, const N: usize, Chips: M + 'static>
    Op<(
        &HbmTensor<D, Chips, m![{ Symbol<Tile<R, N>> }, S]>,
        (&mut HbmTensor<D, Chips, m![R, S]>, usize),
    )> for Store
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (tile, (rows, at)): (
            &HbmTensor<D, Chips, m![{ Symbol<Tile<R, N>> }, S]>,
            (&mut HbmTensor<D, Chips, m![R, S]>, usize),
        ),
    ) {
        type T<R, const N: usize> = Symbol<Tile<R, N>>;
        // SAFETY: the destination's row tile supplies the tile's outer offset.
        let tile = unsafe { tile.view().reshape::<Chips, m![R % ({ T<R, N> }), S]>() };
        tile.to_hbm_view(
            &mut device.tdma,
            rows.view_mut().tile::<m![R / ({ T<R, N> })], 1, m![
                R / ({ T<R, N> }) = 1 #{!} (R / ({ T<R, N> })),
                R % ({ T<R, N> }),
                S
            ]>(at),
        );
    }
}

/// Each row's token.
impl<R: AxisName, Chips: M + 'static> Op<&Selected<R, Chips>> for Store {
    type Output = HbmTensor<i32, Chips, m![R]>;

    fn forward(&self, device: &mut Device, tokens: &Selected<R, Chips>) -> Self::Output {
        tokens.to_hbm(&mut device.tdma)
    }
}

/// Tile `at` of a launch's tokens, `R` rows a tile, where its rows sit among the launch's.
impl<R: AxisName, Capacity: AxisName, Chips: M + 'static>
    Op<(
        &Selected<R, Chips>,
        (&mut HbmTensor<i32, Chips, m![Capacity]>, usize),
    )> for Store
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (tokens, (launch, at)): (
            &Selected<R, Chips>,
            (&mut HbmTensor<i32, Chips, m![Capacity]>, usize),
        ),
    ) {
        tokens.view().to_hbm_view(
            &mut device.tdma,
            launch.view_mut().tile::<m![Capacity / R], 1, m![
                Capacity / R = 1 #{!} (Capacity / R), Capacity % R
            ]>(at),
        );
    }
}

/// A tile's live rows into the place the node's state holds, so a stream stores each tile in turn.
impl<Width: AxisName, Rows: AxisName, Env, Chips: M + 'static>
    crate::Module<&Live<Spread<Width, Rows, Chips>, Rows>, Env, &mut HiddenRows<Chips, Width, Rows>>
    for Store
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        output: &mut HiddenRows<Chips, Width, Rows>,
        input: &Live<Spread<Width, Rows, Chips>, Rows>,
    ) {
        Op::forward(self, device, ((input.value(), input.with(())), output));
    }
}

/// A store streams one `forward` a tile into the place its slot holds.
impl<'m, 's, U: Stream, E, S> IntoStream<U, E, &'s mut S> for &'m Store
where
    RowWise<'s, &'m Store, E, S, U>: Stream,
{
    type Stream = RowWise<'s, &'m Store, E, S, U>;

    fn into_stream(self, env: E, state: &'s mut S, upstream: U) -> Self::Stream {
        RowWise::new(self, env, state, upstream)
    }
}

/// Write key and value rows to distinct physical byte offsets, bounded by the supplied count.
#[derive(Default, DeviceSend, Load)]
pub struct Scatter;

/// The offsets and count are owned or borrowed, so a count every layer shares stays one operand.
impl<
    Width: AxisName,
    Rows: AxisName,
    D: MaterializableScalar,
    O: Borrow<Offsets<Rows, Chips>>,
    L: Borrow<Length<Chips>>,
    Chips: M + 'static,
>
    Op<(
        &mut table::KeyValue<RowStorage<Width, D, Chips>>,
        &(O, L),
        (
            &HbmTensor<D, Chips, m![Rows, Width]>,
            &HbmTensor<D, Chips, m![Rows, Width]>,
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (cache, (offsets, count), (keys, values)): (
            &mut table::KeyValue<RowStorage<Width, D, Chips>>,
            &(O, L),
            (
                &HbmTensor<D, Chips, m![Rows, Width]>,
                &HbmTensor<D, Chips, m![Rows, Width]>,
            ),
        ),
    ) {
        let (offsets, count): (&Offsets<Rows, Chips>, &Length<Chips>) =
            (offsets.borrow(), count.borrow());
        self.forward(&mut *device, (&mut cache.0, &offsets.0, count, keys));
        self.forward(device, (&mut cache.1, &offsets.1, count, values));
    }
}

impl<Width: AxisName, Rows: AxisName, D: MaterializableScalar, Chips: M + 'static>
    Op<(
        &mut Access<Width, Rows, D, Chips>,
        &HbmTensor<D, Chips, m![Rows, Width]>,
        &HbmTensor<D, Chips, m![Rows, Width]>,
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (cache, keys, values): (
            &mut Access<Width, Rows, D, Chips>,
            &HbmTensor<D, Chips, m![Rows, Width]>,
            &HbmTensor<D, Chips, m![Rows, Width]>,
        ),
    ) {
        Scatter.forward(device, (&mut cache.0, &cache.1, (keys, values)))
    }
}

impl<W: AxisName, N: AxisName, D: MaterializableScalar, Chips: M + 'static>
    Op<(
        &mut (
            &mut table::KeyValue<RowStorage<W, D, Chips>>,
            &(Offsets<N, Chips>, Length<Chips>),
        ),
        &HbmTensor<D, Chips, m![N, W]>,
        &HbmTensor<D, Chips, m![N, W]>,
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        ((cache, write), keys, values): (
            &mut (
                &mut table::KeyValue<RowStorage<W, D, Chips>>,
                &(Offsets<N, Chips>, Length<Chips>),
            ),
            &HbmTensor<D, Chips, m![N, W]>,
            &HbmTensor<D, Chips, m![N, W]>,
        ),
    ) {
        Scatter.forward(device, (&mut **cache, *write, (keys, values)))
    }
}

impl<W: AxisName, Q: AxisName, D: MaterializableScalar, Chips: M + 'static>
    Op<(
        &mut RowStorage<W, D, Chips>,
        &(Offsets<Q, Chips>, Length<Chips>),
        &HbmTensor<D, Chips, m![Q, W]>,
        &HbmTensor<D, Chips, m![Q, W]>,
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (storage, write, keys, values): (
            &mut RowStorage<W, D, Chips>,
            &(Offsets<Q, Chips>, Length<Chips>),
            &HbmTensor<D, Chips, m![Q, W]>,
            &HbmTensor<D, Chips, m![Q, W]>,
        ),
    ) {
        Scatter.forward(device, (storage, write, (keys, values)))
    }
}

/// Rows whose tokens each head's slices hold whole: each row's interleaved key and value is written
/// in the slot its byte offset names, so one call writes any number of rows as one-row calls would.
impl<
    R: M,
    Heads: AxisName,
    Head: AxisName,
    D: MaterializableScalar,
    Chips: M,
    Cluster: M,
    Spread: M,
>
    Op<(
        &mut Blocks<Heads, Head, D, Chips>,
        &HbmTensor<i32, Chips, R>,
        &DmTensor<D, Chips, Cluster, m![Heads, { Spread }], m![{ R }, KeyValue, Head]>,
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (layer, slot, token): (
            &mut Blocks<Heads, Head, D, Chips>,
            &HbmTensor<i32, Chips, R>,
            &DmTensor<D, Chips, Cluster, m![Heads, { Spread }], m![{ R }, KeyValue, Head]>,
        ),
    ) {
        layer
            .tokens
            .scatter::<m![Block], m![Heads, { Spread }, KeyValue, Head]>()
            .by_byte_offsets(slot.view())
            .from_dm_view(&mut device.tdma, token.view());
    }
}

impl<Heads: AxisName, Head: AxisName, D: MaterializableScalar, Chips: M>
    Op<(
        &mut Blocks<Heads, Head, D, Chips>,
        &HbmTensor<i32, Chips, m![1]>,
        (
            &HbmTensor<D, Chips, m![Heads, Head]>,
            &HbmTensor<D, Chips, m![Heads, Head]>,
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (layer, slot, (key, value)): (
            &mut Blocks<Heads, Head, D, Chips>,
            &HbmTensor<i32, Chips, m![1]>,
            (
                &HbmTensor<D, Chips, m![Heads, Head]>,
                &HbmTensor<D, Chips, m![Heads, Head]>,
            ),
        ),
    ) {
        type Slices<Heads> = m![Heads, 1 # (256 / Heads)];
        let mut token: DmTensor<D, Chips, m![1 # 2], Slices<Heads>, m![1, KeyValue, Head]> =
            DmTensor::new();
        let key: DmTensor<D, Chips, m![1 # 2], Slices<Heads>, m![Head]> =
            key.to_dm(&mut device.tdma);
        key.view().to_dm_view(
            &mut device.tdma,
            token
                .view_mut()
                .tile::<m![KeyValue], 1, m![1, KeyValue = 1 #{!} 2, Head]>(0),
        );
        let value: DmTensor<D, Chips, m![1 # 2], Slices<Heads>, m![Head]> =
            value.to_dm(&mut device.tdma);
        value.view().to_dm_view(
            &mut device.tdma,
            token
                .view_mut()
                .tile::<m![KeyValue], 1, m![1, KeyValue = 1 #{!} 2, Head]>(1),
        );
        self.forward(device, (layer, slot, &token))
    }
}

/// The slots of an envelope's rows and its live rows: rows `0..rows.rows()` write their slots,
/// and a later row's entry is never read.
pub type Slots<'s, R, Chips = Chip> = (&'s HbmTensor<i32, Chips, m![R]>, Live<(), R>);

/// The live rows' interleaved keys and values written in the slots their byte offsets name, all in
/// one transfer when every row is live, else a packet at a time; no other slot changes.
impl<
    R: AxisName,
    Heads: AxisName,
    Head: AxisName,
    D: MaterializableScalar,
    Chips: M,
    Cluster: M,
    Spread: M,
>
    Op<(
        &mut Blocks<Heads, Head, D, Chips>,
        Slots<'_, R, Chips>,
        &DmTensor<D, Chips, Cluster, m![Heads, { Spread }], m![R, KeyValue, Head]>,
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (layer, (slot, rows), token): (
            &mut Blocks<Heads, Head, D, Chips>,
            Slots<'_, R, Chips>,
            &DmTensor<D, Chips, Cluster, m![Heads, { Spread }], m![R, KeyValue, Head]>,
        ),
    ) {
        type Payload<Heads, Spread, Head> = m![Heads, { Spread }, KeyValue, Head];
        rows.cover::<Lanes<R>, _>(
            &mut (device, layer),
            (
                |(device, layer)| {
                    layer
                        .tokens
                        .scatter::<m![Block], Payload<Heads, Spread, Head>>()
                        .by_byte_offsets(slot.view())
                        .from_dm_view(&mut device.tdma, token.view());
                },
                |(device, layer), packet| {
                    // A scatter takes its offsets as a whole tensor.
                    let mut index: HbmTensor<i32, Chips, m![{ Layout<R>::Lanes }]> =
                        HbmTensor::new();
                    slot.view()
                        .tile::<
                            Layout<R>::Groups,
                            1,
                            m![{ Layout<R>::Selected }, { Layout<R>::Lanes }],
                        >(
                            packet,
                        )
                        .to_hbm_view(&mut device.tdma, index.view_mut());
                    layer
                        .tokens
                        .scatter::<m![Block], Payload<Heads, Spread, Head>>()
                        .by_byte_offsets(index.view())
                        .from_dm_view(
                            &mut device.tdma,
                            token
                                .view()
                                .tile::<
                                    Layout<R>::Groups,
                                    1,
                                    Layout<R>::Elements<m![KeyValue, Head]>,
                                >(
                                    packet,
                                ),
                        );
                },
                |(device, layer), row| {
                    let mut index: HbmTensor<i32, Chips, m![1]> = HbmTensor::new();
                    slot.view()
                        .tile::<m![R], 1, m![R = 1 # R]>(row)
                        .to_hbm_view(&mut device.tdma, index.view_mut());
                    layer
                        .tokens
                        .scatter::<m![Block], Payload<Heads, Spread, Head>>()
                        .by_byte_offsets(index.view())
                        .from_dm_view(
                            &mut device.tdma,
                            token
                                .view()
                                .tile::<m![R], 1, m![R = 1 # R, KeyValue, Head]>(row),
                        );
                },
            ),
        );
    }
}

/// Rows spread over the lanes of each head's slices, as the dense projection leaves them: each
/// row's key and value move beside each other into the row's token, then only the live rows write.
impl<R: AxisName, Heads: AxisName, Head: AxisName, Chips: M + 'static>
    Op<(
        &mut Blocks<Heads, Head, bf16, Chips>,
        Slots<'_, R, Chips>,
        (
            &attention::Heads<R, Heads, Single, Head>,
            &attention::Heads<R, Heads, Single, Head>,
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (layer, slots, (key, value)): (
            &mut Blocks<Heads, Head, bf16, Chips>,
            Slots<'_, R, Chips>,
            (
                &attention::Heads<R, Heads, Single, Head>,
                &attention::Heads<R, Heads, Single, Head>,
            ),
        ),
    ) {
        let mut token: DmTensor<
            bf16,
            Chips,
            m![1 # 2],
            m![Heads, 1 # (256 / Heads)],
            m![R, KeyValue, Head],
        > = DmTensor::new();
        key.view().to_dm_view(
            &mut device.tdma,
            token
                .view_mut()
                .tile::<m![KeyValue], 1, m![R, KeyValue = 1 #{!} 2, Head]>(0),
        );
        value.view().to_dm_view(
            &mut device.tdma,
            token
                .view_mut()
                .tile::<m![KeyValue], 1, m![R, KeyValue = 1 #{!} 2, Head]>(1),
        );
        self.forward(device, (layer, slots, &token))
    }
}

/// A launch's keys and values as attention projects them: only the live rows write, each into the
/// slot the write operand names for it, in the layer's blocks.
impl<R: AxisName, Heads: AxisName, Head: AxisName, Chips: M + 'static>
    Op<(
        &mut Blocks<Heads, Head, bf16, Chips>,
        &HbmTensor<i32, Chips, m![R]>,
        (
            &Live<attention::Heads<R, Heads, Single, Head>, R>,
            &Live<attention::Heads<R, Heads, Single, Head>, R>,
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (layer, slot, (key, value)): (
            &mut Blocks<Heads, Head, bf16, Chips>,
            &HbmTensor<i32, Chips, m![R]>,
            (
                &Live<attention::Heads<R, Heads, Single, Head>, R>,
                &Live<attention::Heads<R, Heads, Single, Head>, R>,
            ),
        ),
    ) {
        self.forward(
            device,
            (layer, (slot, key.with(())), (key.value(), value.value())),
        )
    }
}

impl<W: AxisName, Q: AxisName, D: MaterializableScalar, Chips: M + 'static>
    Op<(
        &mut RowStorage<W, D, Chips>,
        &HbmTensor<i32, Chips, m![Q]>,
        &Length<Chips>,
        &HbmTensor<D, Chips, m![Q, W]>,
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        _: &mut Device,
        _: (
            &mut RowStorage<W, D, Chips>,
            &HbmTensor<i32, Chips, m![Q]>,
            &Length<Chips>,
            &HbmTensor<D, Chips, m![Q, W]>,
        ),
    ) {
        todo!("NPU scatter of only the first count rows to distinct physical byte offsets")
    }
}

impl<W: AxisName, Q: AxisName, D: MaterializableScalar, Chips: M + 'static>
    Op<(
        &mut RowStorage<W, D, Chips>,
        &(Offsets<Q, Chips>, Length<Chips>),
        (
            &HbmTensor<D, Chips, m![Q, W]>,
            &HbmTensor<D, Chips, m![Q, W]>,
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (storage, (offsets, count), (keys, values)): (
            &mut RowStorage<W, D, Chips>,
            &(Offsets<Q, Chips>, Length<Chips>),
            (
                &HbmTensor<D, Chips, m![Q, W]>,
                &HbmTensor<D, Chips, m![Q, W]>,
            ),
        ),
    ) {
        self.forward(&mut *device, (&mut *storage, &offsets.0, count, keys));
        self.forward(device, (storage, &offsets.1, count, values));
    }
}

/// The keys and values of a launch's rows in HBM, the slot each row names, its live rows and the
/// layer they write.
pub(crate) struct Rows<
    'a,
    Reach: AxisName,
    Heads: AxisName,
    Head: AxisName,
    R: AxisName,
    Chips: M + 'static = Chip,
> {
    pub pair: (
        &'a HbmTensor<bf16, Chips, m![Reach, Heads, Head]>,
        &'a HbmTensor<bf16, Chips, m![Reach, Heads, Head]>,
    ),
    pub slot: &'a HbmTensor<i32, Chips, m![Reach]>,
    pub rows: Live<(), R>,
    pub pool: &'a mut Blocks<Heads, Head, bf16, Chips>,
}

/// A store's `T` rows in DM, `P` a slice: each cluster copies and scatters its own four heads, so
/// no scatter reads the other cluster's copy.
type Token<Chips, T, const P: usize, Heads, Head> = DmTensor<
    bf16,
    Chips,
    m![Heads / 4],
    m![Heads % 4, T / { P } # 64],
    m![T % { P }, KeyValue, Head],
>;

impl<'a, Reach: AxisName, Heads: AxisName, Head: AxisName, R: AxisName, Chips: M + 'static>
    Rows<'a, Reach, Heads, Head, R, Chips>
{
    /// Copies the `T` rows from `row` and writes each to its slot; `N` is `T`'s size.
    fn store<T: AxisName, const N: usize, const P: usize>(
        &mut self,
        device: &mut Device,
        row: usize,
    ) {
        const { assert!(T::SIZE == N && P == per_slice(N)) };
        let mut token = Token::<Chips, T, P, Heads, Head>::new();
        let (key, value) = self.pair;
        let window = |tensor: &'a HbmTensor<bf16, Chips, m![Reach, Heads, Head]>| {
            // SAFETY: the row relabel keeps the envelope padding and every live row coordinate.
            unsafe {
                tensor
                    .view()
                    .tile::<m![Reach], N, m![Reach = { N } # Reach, Heads, Head]>(row)
                    .reshape::<Chips, m![T # Reach, Heads, Head]>()
            }
        };
        window(key).to_dm_view(
            &mut device.tdma,
            token
                .view_mut()
                .tile::<m![KeyValue], 1, m![T % { P }, KeyValue = 1 #{!} 2, Head]>(0),
        );
        window(value).to_dm_view(
            &mut device.tdma,
            token
                .view_mut()
                .tile::<m![KeyValue], 1, m![T % { P }, KeyValue = 1 #{!} 2, Head]>(1),
        );
        // SAFETY: the immutable window contains the store's slots in row order.
        let destination: HbmTensor<i32, Chips, m![T]> = unsafe {
            self.slot
                .view()
                .tile::<m![Reach], N, m![Reach = { N } # Reach]>(row)
                .reshape::<Chips, m![T # Reach]>()
                .alias()
        };
        // SAFETY: the host reserves a distinct slot for every live row; all fit the selected
        // layer and exclude committed history.
        unsafe {
            self.pool
                .tokens
                .scatter::<m![Block], m![KeyValue, Heads, Head]>()
                .by_byte_offsets(destination.view())
                .unchecked()
                .from_dm_view(&mut device.tdma, token.view())
        };
    }
}

/// Only the live rows write, each into its slot: stores of each of [`store::LENGTHS`], longest
/// first, while the live rows fill one, then the rest a packet, then a row, at a time.
impl<Reach: AxisName, Heads: AxisName, Head: AxisName, R: AxisName, Chips: M + 'static>
    Op<Rows<'_, Reach, Heads, Head, R, Chips>> for Scatter
{
    type Output = ();

    fn forward(&self, device: &mut Device, mut rows: Rows<'_, Reach, Heads, Head, R, Chips>) {
        const LONG: usize = store::LENGTHS[0];
        const MEDIUM: usize = store::LENGTHS[1];
        const SHORT: usize = store::LENGTHS[2];
        const PACKET: usize = store::PACKET;
        const {
            assert!(LONG % MEDIUM == 0 && MEDIUM % SHORT == 0 && SHORT % PACKET == 0);
        };
        let (start, live) = (rows.rows.start(), rows.rows.rows());
        for tile in 0..live / LONG {
            rows.store::<axis::Rows<LONG>, LONG, { per_slice(LONG) }>(device, start + tile * LONG);
        }
        let first = start + live / LONG * LONG;
        for tile in 0..live % LONG / MEDIUM {
            rows.store::<axis::Rows<MEDIUM>, MEDIUM, { per_slice(MEDIUM) }>(
                device,
                first + tile * MEDIUM,
            );
        }
        let first = start + live / MEDIUM * MEDIUM;
        for tile in 0..live % MEDIUM / SHORT {
            rows.store::<axis::Rows<SHORT>, SHORT, { per_slice(SHORT) }>(
                device,
                first + tile * SHORT,
            );
        }
        let first = start + live / SHORT * SHORT;
        for packet in 0..live % SHORT / PACKET {
            rows.store::<axis::Rows<PACKET>, PACKET, { per_slice(PACKET) }>(
                device,
                first + packet * PACKET,
            );
        }
        let first = start + live / PACKET * PACKET;
        for row in 0..live % PACKET {
            rows.store::<Single, 1, { per_slice(1) }>(device, first + row);
        }
    }
}

/// `N` learned rows written over rows `AT..AT + N` of a stream, after its `AT` tokens: the rows of
/// its checkpoint tensor, then zero rows, held as `T`, the HBM tensor or a borrow of it.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Registers<
    W: AxisName,
    const AT: usize,
    const N: usize,
    Chips: M = m![1],
    T = HbmTensor<f32, Chips, m![{ Symbol<Run<N>> }, W]>,
> {
    rows: T,
    shape: PhantomData<fn() -> (W, Chips)>,
}

impl<W: AxisName, const AT: usize, const N: usize, Chips: M + 'static, T>
    Registers<W, AT, N, Chips, T>
{
    /// The rows of the tensor at the context's name, F32 or BF16 rows of `W` (leading axes of one
    /// row count as rows), at most `N` of them, then zero rows up to `N`, on every chip.
    // TODO: read the rows only for a leaf that holds bytes, as a converted leaf's build does.
    fn learned(
        source: &Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<HostTensor<f32, m![{ Chips }, { m![{ Symbol<Run<N>> }, W] }]>, LoadError> {
        let (name, view) = context.tensor(source)?;
        let shape = view.shape().to_vec();
        let rows: usize = shape
            .split_last()
            .map_or(0, |(_, rows)| rows.iter().product());
        if shape.last() != Some(&W::SIZE) || rows > N {
            return Err(LoadError::InvalidData {
                name,
                reason: format!(
                    "shape {shape:?}, the leaf holds at most {N} rows of {}",
                    W::SIZE
                ),
            });
        }
        let mut values = context.floats(source, &shape)?;
        values.resize(N * W::SIZE, 0.0);
        Ok(HostTensor::from_vec(values.repeat(Chips::SIZE)))
    }
}

/// The rows at the registers' own name.
/// By hand, with the update below: a derived `with` update copies a new leaf through host.
impl<'a, W: AxisName, const AT: usize, const N: usize, Chips, T> Load<&'a Checkpoint>
    for Registers<W, AT, N, Chips, T>
where
    Chips: M + 'static,
    T: Load<HostTensor<f32, m![{ Chips }, { m![{ Symbol<Run<N>> }, W] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("rows", "", |cx| {
                let learned = Self::learned(from, cx)?;
                let rows = T::load(cx, learned)?;
                Ok(Self {
                    rows,
                    shape: PhantomData,
                })
            })
        })
    }
}

/// The rows written in place, one rule of an update.
impl<'t, 's, W: AxisName, const AT: usize, const N: usize, Chips, T>
    Load<(&'t mut Registers<W, AT, N, Chips, T>, &'s Checkpoint)>
    for &'t mut Registers<W, AT, N, Chips, T>
where
    Chips: M + 'static,
    for<'u> &'u mut T: Load<(
        &'u mut T,
        HostTensor<f32, m![{ Chips }, { m![{ Symbol<Run<N>> }, W] }]>,
    )>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Registers<W, AT, N, Chips, T>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("rows", "", |cx| {
                cx.rule(
                    |cx| Registers::<W, AT, N, Chips, T>::learned(from, cx),
                    |cx, rows| <&mut T as Load<_>>::load(cx, (&mut into.rows, rows)).map(drop),
                )
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

/// `N` consecutive rows, as one copy moves them.
#[derive(Clone, Debug)]
pub struct Run<const N: usize>;

impl<const N: usize> AxisName for Run<N> {
    const NAME: Ident = Ident::new("Run");
    const SIZE: usize = N;
}

impl<W: AxisName, R: AxisName, const AT: usize, const N: usize, Chips: M, T>
    Op<HbmTensor<f32, Chips, m![R, W]>> for Registers<W, AT, N, Chips, T>
where
    T: Borrow<HbmTensor<f32, Chips, m![{ Symbol<Run<N>> }, W]>>,
{
    type Output = HbmTensor<f32, Chips, m![R, W]>;

    fn forward(
        &self,
        device: &mut Device,
        mut stream: HbmTensor<f32, Chips, m![R, W]>,
    ) -> Self::Output {
        type L<const N: usize> = Symbol<Run<N>>;
        let target = stream
            .view_mut()
            .tile::<m![R], N, m![R = ({ L<N> }) #{!} R, W]>(AT);
        // SAFETY: the tile's rows are the run's in order; only the row axis is renamed.
        let target = unsafe { target.reshape::<Chips, m![{ L<N> } #{!} R, W]>() };
        let rows: &HbmTensor<f32, Chips, m![{ L<N> }, W]> = self.rows.borrow();
        rows.view().to_hbm_view(&mut device.tdma, target);
        stream
    }
}

/// Rows a slice holds of a store of `rows` rows over a head's 64 slices.
const fn per_slice(rows: usize) -> usize {
    rows.div_ceil(64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::attention::Heads;
    use crate::ops::attention::geometry::Hidden;
    use crate::ops::attention::tests::{Blocks, Head, KvHeads};
    use crate::ops::normalization::tests::random;
    use crate::tensor::axis::Single;
    use crate::tensor::axis::{Eight, Sixteen, SixtyFour};
    use crate::tensor::{live::tests::count, scalar::Value};
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    type Width = Hidden<1024>;

    /// Token slots of the pools the write tests fill.
    const TOKENS: usize = 4096;

    type Capacity = axis::Rows<1024>;
    type Reach = axis::Rows<2048>;
    type Projected = HbmTensor<bf16, Chip, m![Reach, KvHeads, Head]>;

    #[device(chip = 1)]
    fn store_live<R: AxisName>(
        device: &mut Device,
        rows: &Value,
        input: &HbmTensor<bf16, Chip, m![R, Width]>,
    ) -> HiddenRows<Chip, Width, R> {
        let input: Spread<Width, R> = input.to_dm(&mut device.tdma);
        Store.forward(device, (&input, Live::<(), R>::read(rows)))
    }

    /// A store of `live` of `R` rows writes each live row as it is and leaves every later row of
    /// its output as it was.
    async fn stores_only_live_rows<R: AxisName>(device: &mut Device, live: usize) {
        let values = random(R::SIZE * Width::SIZE, 0x5851_f42d_4c95_7f2d);
        let input = HostTensor::<bf16, m![R, Width]>::from_vec(values.clone())
            .to_hbm::<Chip, m![R, Width]>(&mut device.pdma)
            .await
            .unwrap();
        let canary = bf16::from_f32(3.0);
        let mut output =
            HostTensor::<bf16, m![R, Width]>::from_vec(vec![canary; R::SIZE * Width::SIZE])
                .to_hbm::<Chip, m![R, Width]>(&mut device.pdma)
                .await
                .unwrap();
        let rows = count(live).to_hbm(&mut device.pdma).await.unwrap();
        launch(store_live, (&mut *device, &rows, &input))
            .output(&mut output)
            .await
            .unwrap();
        let output = output
            .to_host::<m![R, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, got) in output.iter().enumerate() {
            let row = at / Width::SIZE;
            let want = if row < live { values[at] } else { canary };
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{live} of {} rows live: row {row} column {}",
                R::SIZE,
                at % Width::SIZE
            );
        }
    }

    /// Only live rows are stored, at no row, one, a partial and a whole first packet, a partial
    /// second packet, and one short of and at a whole 64-row envelope.
    #[tokio::test(flavor = "current_thread")]
    async fn stores_only_live_rows_at_every_count() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        for live in [0, 1, 7, 8] {
            stores_only_live_rows::<Eight>(&mut device, live).await;
        }
        stores_only_live_rows::<Sixteen>(&mut device, 9).await;
        for live in [63, 64] {
            stores_only_live_rows::<SixtyFour>(&mut device, live).await;
        }
    }

    #[device(chip = 1)]
    fn write_rows<R: AxisName>(
        device: &mut Device,
        blocks: &mut Blocks,
        slot: &HbmTensor<i32, Chip, m![R]>,
        rows: &Value,
        token: &HbmTensor<bf16, Chip, m![KvHeads, R, KeyValue, Head]>,
    ) {
        let token: DmTensor<bf16, Chip, m![1 # 2], m![KvHeads, 1 # 32], m![R, KeyValue, Head]> =
            token.to_dm(&mut device.tdma);
        Scatter.forward(device, (blocks, (slot, Live::<(), R>::read(rows)), &token));
    }

    #[device(chip = 1)]
    fn write_lanes<R: AxisName>(
        device: &mut Device,
        blocks: &mut Blocks,
        slot: &HbmTensor<i32, Chip, m![R]>,
        rows: &Value,
        key: &HbmTensor<bf16, Chip, m![R, KvHeads, Head]>,
        value: &HbmTensor<bf16, Chip, m![R, KvHeads, Head]>,
    ) {
        let key: Heads<R, KvHeads, Single, Head> = key.to_dm(&mut device.tdma);
        let value: Heads<R, KvHeads, Single, Head> = value.to_dm(&mut device.tdma);
        Scatter.forward(
            device,
            (blocks, (slot, Live::<(), R>::read(rows)), (&key, &value)),
        );
    }

    #[device(chip = 1)]
    fn write_row(
        device: &mut Device,
        blocks: &mut Blocks,
        slot: &HbmTensor<i32, Chip, m![1]>,
        key: &HbmTensor<bf16, Chip, m![KvHeads, Head]>,
        value: &HbmTensor<bf16, Chip, m![KvHeads, Head]>,
    ) {
        Scatter.forward(device, (blocks, slot, (key, value)));
    }

    /// One call over an envelope of `R` rows, `live` of them live, leaves the pool one-row calls
    /// leave for its live rows, with its rows held whole by each head or spread over lanes. Each
    /// padding row's entry names a writable slot of its own, and that slot keeps its value.
    async fn writes_only_live_rows<R: AxisName>(device: &mut Device, live: usize) {
        let width = KeyValue::SIZE * KvHeads::SIZE * Head::SIZE;
        let values = random(R::SIZE * width, 0x3c6e_f372_fe94_f82b);
        let token = |row: usize, kv: usize, head: usize, channel: usize| {
            values[((row * KeyValue::SIZE + kv) * KvHeads::SIZE + head) * Head::SIZE + channel]
        };
        let slots: Vec<usize> = (0..R::SIZE)
            .map(|row| {
                if row < live {
                    (3 * row + 1) * Block::SIZE + row % Block::SIZE
                } else {
                    TOKENS - R::SIZE + row
                }
            })
            .collect();
        let offset = |slot: usize| slot as i32 * <Blocks>::ROW_STRIDE;
        let canary = || vec![bf16::from_f32(3.0); TOKENS * width];

        let mut sequential = Blocks::from_rows(device, canary()).await;
        for row in 0..live {
            let half = |kv: usize| {
                HostTensor::<bf16, m![KvHeads, Head]>::from_vec(
                    (0..KvHeads::SIZE * Head::SIZE)
                        .map(|at| token(row, kv, at / Head::SIZE, at % Head::SIZE)),
                )
            };
            let key = half(0).to_hbm(&mut device.pdma).await.unwrap();
            let value = half(1).to_hbm(&mut device.pdma).await.unwrap();
            let slot = HostTensor::<i32, m![1]>::from_vec([offset(slots[row])])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            launch(
                write_row,
                (&mut *device, &mut sequential, &slot, &key, &value),
            )
            .await
            .unwrap();
        }

        let slot = HostTensor::<i32, m![R]>::from_vec(slots.iter().map(|&slot| offset(slot)))
            .to_hbm::<Chip, m![R]>(&mut device.pdma)
            .await
            .unwrap();
        let rows = count(live).to_hbm(&mut device.pdma).await.unwrap();
        let mut whole = Blocks::from_rows(device, canary()).await;
        let tokens = HostTensor::<bf16, m![KvHeads, R, KeyValue, Head]>::from_vec(
            (0..width * R::SIZE).map(|at| {
                let channel = at % Head::SIZE;
                let kv = at / Head::SIZE % KeyValue::SIZE;
                let row = at / (Head::SIZE * KeyValue::SIZE) % R::SIZE;
                token(
                    row,
                    kv,
                    at / (Head::SIZE * KeyValue::SIZE * R::SIZE),
                    channel,
                )
            }),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        launch(
            write_rows,
            (&mut *device, &mut whole, &slot, &rows, &tokens),
        )
        .await
        .unwrap();
        let mut lanes = Blocks::from_rows(device, canary()).await;
        let half = |kv: usize| {
            HostTensor::<bf16, m![R, KvHeads, Head]>::from_vec(
                (0..R::SIZE * KvHeads::SIZE * Head::SIZE).map(|at| {
                    let head = at / Head::SIZE % KvHeads::SIZE;
                    token(at / (Head::SIZE * KvHeads::SIZE), kv, head, at % Head::SIZE)
                }),
            )
        };
        let key = half(0).to_hbm(&mut device.pdma).await.unwrap();
        let value = half(1).to_hbm(&mut device.pdma).await.unwrap();
        launch(
            write_lanes,
            (&mut *device, &mut lanes, &slot, &rows, &key, &value),
        )
        .await
        .unwrap();

        let blocks = 0..TOKENS / Block::SIZE;
        let sequential = sequential.read(device, blocks.clone()).await.unwrap();
        for (form, pool) in [("whole", whole), ("lanes", lanes)] {
            let pool = pool.read(device, blocks.clone()).await.unwrap();
            for slot in 0..TOKENS {
                let range = slot * width..(slot + 1) * width;
                assert!(
                    pool[range.clone()]
                        .iter()
                        .zip(&sequential[range])
                        .all(|(got, want)| got.to_bits() == want.to_bits()),
                    "{form}, {live} of {} rows live: slot {slot} differs",
                    R::SIZE
                );
            }
        }
    }

    /// Only live rows write, at no row, one, a partial and a whole first packet, a partial second
    /// packet, and one short of and at a whole 64-row envelope.
    #[tokio::test(flavor = "current_thread")]
    async fn writes_only_live_rows_at_every_count() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        for live in [0, 1, 7, 8] {
            writes_only_live_rows::<Eight>(&mut device, live).await;
        }
        writes_only_live_rows::<Sixteen>(&mut device, 9).await;
        for live in [63, 64] {
            writes_only_live_rows::<SixtyFour>(&mut device, live).await;
        }
    }

    #[device(chip = 1)]
    fn write_projected(
        device: &mut Device,
        blocks: &mut Blocks,
        slot: &HbmTensor<i32, Chip, m![Reach]>,
        (live, first): (&Value, &Value),
        (key, value): (&Projected, &Projected),
    ) {
        let live = Live::<(), Capacity>::read(live);
        let rows = live.subrange(first.extent::<m![Capacity]>(), live.rows());
        Scatter.forward(
            device,
            Rows {
                pair: (key, value),
                slot,
                rows,
                pool: blocks,
            },
        );
    }

    /// Each live row of HBM keys and values from the first live row writes its slot, and no other
    /// slot changes, though every row names a slot of its own. The live rows start at no row and
    /// mid-store, end mid-store and on a 512-row edge, and leave after their longest stores every
    /// kind of remainder: none, rows, a packet and rows, and 32-row and 128-row stores.
    #[tokio::test(flavor = "current_thread")]
    async fn writes_each_live_projected_row_to_its_slot() {
        const TOKENS: usize = 2 * Reach::SIZE;
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(write_projected.topology()).unwrap();
        let width = KeyValue::SIZE * KvHeads::SIZE * Head::SIZE;
        let half = KvHeads::SIZE * Head::SIZE;
        let keys = random(Reach::SIZE * half, 0x9e37_79b9_7f4a_7c15);
        let values = random(Reach::SIZE * half, 0x3c6e_f372_fe94_f82b);
        let upload = async |device: &mut Device, rows: &[bf16]| -> Projected {
            HostTensor::<bf16, m![Reach, KvHeads, Head]>::from_vec(rows.iter().copied())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap()
        };
        let key = upload(&mut device, &keys).await;
        let value = upload(&mut device, &values).await;
        let own = |row: usize| 2 * row + 1;
        let slots =
            (0..Reach::SIZE).map(|row| i32::try_from(own(row)).unwrap() * Blocks::ROW_STRIDE);
        let slot = HostTensor::<i32, m![Reach]>::from_vec(slots)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        for (first, live) in [
            (0, 0),
            (0, 1),
            (0, 7),
            (0, 8),
            (0, 31),
            (0, 32),
            (0, 45),
            (30, 58),
            (0, 128),
            (0, 300),
            (0, 512),
            (100, 613),
            (0, 1000),
            (384, 1024),
            (0, 1024),
        ] {
            let canary = bf16::from_f32(3.0);
            let mut pool = Blocks::from_rows(&mut device, vec![canary; TOKENS * width]).await;
            let counts = (
                count(live).to_hbm(&mut device.pdma).await.unwrap(),
                count(first).to_hbm(&mut device.pdma).await.unwrap(),
            );
            let arguments = (
                &mut device,
                &mut pool,
                &slot,
                (&counts.0, &counts.1),
                (&key, &value),
            );
            launch(write_projected, arguments).await.unwrap();
            let pool = pool
                .read(&mut device, 0..TOKENS / Block::SIZE)
                .await
                .unwrap();
            for token in 0..TOKENS {
                let got = &pool[token * width..(token + 1) * width];
                let row = (token % 2 == 1).then_some(token / 2);
                let want: Vec<bf16> = match row {
                    Some(row) if (first..live).contains(&row) => {
                        let row = |source: &[bf16]| source[row * half..(row + 1) * half].to_vec();
                        [row(&keys), row(&values)].concat()
                    }
                    _ => vec![canary; width],
                };
                assert!(
                    got.iter()
                        .zip(&want)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "live rows {first}..{live}: slot {token} of row {row:?}"
                );
            }
        }
    }
}
