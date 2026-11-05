//! Indexed KV row transfers preserve caller-owned allocations and independent key/value offsets.

use super::Block;
use crate::{
    Module,
    ops::{
        attention::geometry::Kv,
        indexing::{RowStorage, Select},
        layout::{
            Dm, Live, Rows as Layout,
            axes::{Chip, Lanes, Single},
        },
    },
};
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;

pub type KeyValue<T> = (T, T);
type Rows<Width, Count, D = f32> = HbmTensor<D, m![1], m![Count, Width]>;
type Offsets<Count> = KeyValue<HbmTensor<i32, m![1], m![Count]>>;
type Length = HbmTensor<i32, m![1], m![1]>;
type Access<Width, Rows, D = f32> = (KeyValue<RowStorage<Width, D>>, (Offsets<Rows>, Length));

/// Gather key and value rows from caller-owned storage using independent byte offsets.
#[derive(Default, DeviceSend)]
pub struct Gather;

impl<Width: AxisName, Count: AxisName, D: MaterializableScalar>
    Module<(
        &mut Device,
        (&KeyValue<RowStorage<Width, D>>, &Offsets<Count>),
    )> for Gather
{
    type Output = KeyValue<Rows<Width, Count, D>>;

    fn forward(
        &self,
        (device, (cache, offsets)): (
            &mut Device,
            (&KeyValue<RowStorage<Width, D>>, &Offsets<Count>),
        ),
    ) -> Self::Output {
        (
            Select::<Width, D>::default().forward((&mut *device, (&cache.0, &offsets.0))),
            Select::<Width, D>::default().forward((device, (&cache.1, &offsets.1))),
        )
    }
}

impl<Width: AxisName, Count: AxisName, D: MaterializableScalar>
    Module<(&mut Device, &Access<Width, Count, D>)> for Gather
{
    type Output = KeyValue<Rows<Width, Count, D>>;

    fn forward(&self, (device, cache): (&mut Device, &Access<Width, Count, D>)) -> Self::Output {
        self.forward((device, (&cache.0, &cache.1.0)))
    }
}

/// Write key and value rows to distinct physical byte offsets, bounded by the supplied count.
#[derive(Default, DeviceSend)]
pub struct Scatter;

/// The offsets and count are owned or borrowed, so a count every layer shares stays one operand.
impl<
    Width: AxisName,
    Rows: AxisName,
    D: MaterializableScalar,
    O: Borrow<Offsets<Rows>>,
    L: Borrow<Length>,
>
    Module<(
        &mut Device,
        (
            &mut KeyValue<RowStorage<Width, D>>,
            &(O, L),
            (&self::Rows<Width, Rows, D>, &self::Rows<Width, Rows, D>),
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        (device, (cache, (offsets, count), (keys, values))): (
            &mut Device,
            (
                &mut KeyValue<RowStorage<Width, D>>,
                &(O, L),
                (&self::Rows<Width, Rows, D>, &self::Rows<Width, Rows, D>),
            ),
        ),
    ) {
        let (offsets, count): (&Offsets<Rows>, &Length) = (offsets.borrow(), count.borrow());
        self.forward((&mut *device, (&mut cache.0, &offsets.0, count, keys)));
        self.forward((device, (&mut cache.1, &offsets.1, count, values)));
    }
}

/// Write key and value rows through a composition's storage, offsets and count context.
#[derive(DeviceSend)]
pub struct Store;

impl<Width: AxisName, Rows: AxisName, D: MaterializableScalar>
    Module<(
        &mut Device,
        (
            &mut Access<Width, Rows, D>,
            &self::Rows<Width, Rows, D>,
            &self::Rows<Width, Rows, D>,
        ),
    )> for Store
{
    type Output = ();

    fn forward(
        &self,
        (device, (cache, keys, values)): (
            &mut Device,
            (
                &mut Access<Width, Rows, D>,
                &self::Rows<Width, Rows, D>,
                &self::Rows<Width, Rows, D>,
            ),
        ),
    ) {
        Scatter.forward((device, (&mut cache.0, &cache.1, (keys, values))))
    }
}

/// Rows whose tokens each head's slices hold whole: every row's interleaved key and value lands in
/// the slot its byte offset names, so one call writes any number of rows as one-row calls would.
impl<R: M, Heads: AxisName, Head: AxisName, D: MaterializableScalar, Chips: M, Cl: M, Spread: M>
    Module<(
        &mut Device,
        (
            &mut super::Layer<'_, Heads, Head, D, Chips>,
            &HbmTensor<i32, Chips, R>,
            &DmTensor<D, Chips, Cl, m![Heads, { Spread }], m![{ R }, Kv, Head]>,
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        (device, (layer, slot, token)): (
            &mut Device,
            (
                &mut super::Layer<'_, Heads, Head, D, Chips>,
                &HbmTensor<i32, Chips, R>,
                &DmTensor<D, Chips, Cl, m![Heads, { Spread }], m![{ R }, Kv, Head]>,
            ),
        ),
    ) {
        layer
            .blocks
            .tokens
            .scatter::<m![Block], m![Heads, { Spread }, Kv, Head]>()
            .entry(layer.entry)
            .by_byte_offsets(slot.view())
            .from_dm_view(&mut device.tdma, token.view());
    }
}

impl<Heads: AxisName, Head: AxisName, D: MaterializableScalar, Chips: M>
    Module<(
        &mut Device,
        (
            &mut super::Layer<'_, Heads, Head, D, Chips>,
            &HbmTensor<i32, Chips, m![1]>,
            (
                &HbmTensor<D, Chips, m![Heads, Head]>,
                &HbmTensor<D, Chips, m![Heads, Head]>,
            ),
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        (device, (layer, slot, (key, value))): (
            &mut Device,
            (
                &mut super::Layer<'_, Heads, Head, D, Chips>,
                &HbmTensor<i32, Chips, m![1]>,
                (
                    &HbmTensor<D, Chips, m![Heads, Head]>,
                    &HbmTensor<D, Chips, m![Heads, Head]>,
                ),
            ),
        ),
    ) {
        type Slices<Heads> = m![Heads, 1 # (256 / Heads)];
        let mut token: DmTensor<D, Chips, m![1 # 2], Slices<Heads>, m![1, Kv, Head]> =
            DmTensor::new();
        let key: DmTensor<D, Chips, m![1 # 2], Slices<Heads>, m![Head]> =
            key.to_dm(&mut device.tdma);
        key.view().to_dm_view(
            &mut device.tdma,
            token
                .view_mut()
                .tile::<m![Kv], 1, m![1, Kv = 1 #{!} 2, Head]>(0),
        );
        let value: DmTensor<D, Chips, m![1 # 2], Slices<Heads>, m![Head]> =
            value.to_dm(&mut device.tdma);
        value.view().to_dm_view(
            &mut device.tdma,
            token
                .view_mut()
                .tile::<m![Kv], 1, m![1, Kv = 1 #{!} 2, Head]>(1),
        );
        self.forward((device, (layer, slot, &token)))
    }
}

/// The slots of an envelope's rows and its live rows: rows `0..rows.rows()` write their slots,
/// and a later row's entry is never read.
pub type Slots<'s, R, Chips = Chip> = (&'s HbmTensor<i32, Chips, m![R]>, Live<R>);

/// Rows whose tokens each head's slices hold whole, of which only the live ones write: each live
/// row's interleaved key and value lands in the slot its byte offset names, every row in one
/// transfer when all are live, else a whole packet at a time, and no other slot changes.
impl<
    R: AxisName,
    Heads: AxisName,
    Head: AxisName,
    D: MaterializableScalar,
    Chips: M,
    Cl: M,
    Spread: M,
>
    Module<(
        &mut Device,
        (
            &mut super::Layer<'_, Heads, Head, D, Chips>,
            Slots<'_, R, Chips>,
            &DmTensor<D, Chips, Cl, m![Heads, { Spread }], m![R, Kv, Head]>,
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        (device, (layer, (slot, rows), token)): (
            &mut Device,
            (
                &mut super::Layer<'_, Heads, Head, D, Chips>,
                Slots<'_, R, Chips>,
                &DmTensor<D, Chips, Cl, m![Heads, { Spread }], m![R, Kv, Head]>,
            ),
        ),
    ) {
        type Payload<Heads, Spread, Head> = m![Heads, { Spread }, Kv, Head];
        rows.cover(
            &mut (device, layer),
            (
                |(device, layer)| {
                    layer
                        .blocks
                        .tokens
                        .scatter::<m![Block], Payload<Heads, Spread, Head>>()
                        .entry(layer.entry)
                        .by_byte_offsets(slot.view())
                        .from_dm_view(&mut device.tdma, token.view());
                },
                |(device, layer), packet| {
                    // A scatter takes its offsets as a whole tensor.
                    let mut index: HbmTensor<i32, Chips, m![{ Layout<R>::Lanes }]> = HbmTensor::new();
                    slot.view()
                        .tile::<Layout<R>::Groups, 1, m![{ Layout<R>::Selected }, { Layout<R>::Lanes }]>(
                            packet,
                        )
                        .to_hbm_view(&mut device.tdma, index.view_mut());
                    layer
                        .blocks
                        .tokens
                        .scatter::<m![Block], Payload<Heads, Spread, Head>>()
                        .entry(layer.entry)
                        .by_byte_offsets(index.view())
                        .from_dm_view(
                            &mut device.tdma,
                            token
                                .view()
                                .tile::<Layout<R>::Groups, 1, Layout<R>::Elements<m![Kv, Head]>>(
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
                        .blocks
                        .tokens
                        .scatter::<m![Block], Payload<Heads, Spread, Head>>()
                        .entry(layer.entry)
                        .by_byte_offsets(index.view())
                        .from_dm_view(
                            &mut device.tdma,
                            token.view().tile::<m![R], 1, m![R = 1 # R, Kv, Head]>(row),
                        );
                },
            ),
        );
    }
}

/// Rows spread over the lanes of each head's slices, as the dense projection leaves them: each
/// row's key and value move beside each other into the row's token, then only the live rows write.
impl<R: AxisName, Heads: AxisName, Head: AxisName>
    Module<(
        &mut Device,
        (
            &mut super::Layer<'_, Heads, Head>,
            Slots<'_, R>,
            (
                &super::super::Heads<R, Heads, Single, Head>,
                &super::super::Heads<R, Heads, Single, Head>,
            ),
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        (device, (layer, slots, (key, value))): (
            &mut Device,
            (
                &mut super::Layer<'_, Heads, Head>,
                Slots<'_, R>,
                (
                    &super::super::Heads<R, Heads, Single, Head>,
                    &super::super::Heads<R, Heads, Single, Head>,
                ),
            ),
        ),
    ) {
        let mut token: DmTensor<
            bf16,
            Chip,
            m![1 # 2],
            m![Heads, 1 # (256 / Heads)],
            m![R, Kv, Head],
        > = DmTensor::new();
        key.view().to_dm_view(
            &mut device.tdma,
            token
                .view_mut()
                .tile::<m![Kv], 1, m![R, Kv = 1 #{!} 2, Head]>(0),
        );
        value.view().to_dm_view(
            &mut device.tdma,
            token
                .view_mut()
                .tile::<m![Kv], 1, m![R, Kv = 1 #{!} 2, Head]>(1),
        );
        self.forward((device, (layer, slots, &token)))
    }
}

/// A launch's keys and values as attention projects them: only the live rows write, each into the
/// slot the write operand names for it; the cache is the layer range a pool lends.
impl<R: AxisName, Heads: AxisName, Head: AxisName>
    Module<(
        &mut Device,
        (
            &mut &mut super::Layer<'_, Heads, Head>,
            &HbmTensor<i32, Chip, m![R]>,
            (
                &Dm<Lanes<R>, super::super::Heads<R, Heads, Single, Head>>,
                &Dm<Lanes<R>, super::super::Heads<R, Heads, Single, Head>>,
            ),
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        (device, (layer, slot, (key, value))): (
            &mut Device,
            (
                &mut &mut super::Layer<'_, Heads, Head>,
                &HbmTensor<i32, Chip, m![R]>,
                (
                    &Dm<Lanes<R>, super::super::Heads<R, Heads, Single, Head>>,
                    &Dm<Lanes<R>, super::super::Heads<R, Heads, Single, Head>>,
                ),
            ),
        ),
    ) {
        self.forward((
            device,
            (&mut **layer, (slot, key.tiles), (&key.value, &value.value)),
        ))
    }
}

impl<Heads: AxisName, Head: AxisName, D: MaterializableScalar, Chips: M>
    Module<(
        &mut Device,
        (
            &mut &mut super::Layer<'_, Heads, Head, D, Chips>,
            &HbmTensor<i32, Chips, m![1]>,
            (
                &HbmTensor<D, Chips, m![Heads, Head]>,
                &HbmTensor<D, Chips, m![Heads, Head]>,
            ),
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        (device, (layer, slot, pair)): (
            &mut Device,
            (
                &mut &mut super::Layer<'_, Heads, Head, D, Chips>,
                &HbmTensor<i32, Chips, m![1]>,
                (
                    &HbmTensor<D, Chips, m![Heads, Head]>,
                    &HbmTensor<D, Chips, m![Heads, Head]>,
                ),
            ),
        ),
    ) {
        self.forward((device, (&mut **layer, slot, pair)))
    }
}

/// Physical key/value byte offsets and additive visibility for an indexed attention read,
/// borrowed from wherever the call keeps them, so layers can read one count or bias.
pub struct Read<'a> {
    /// Number of logical rows to read from each offset list.
    pub count: &'a Length,
    /// Key and value byte offsets in logical row order.
    pub offsets: &'a Selection,
    /// Query-major additive mask over the selected rows.
    pub bias: &'a Slice<HbmTensor<f32, m![1], m![1]>>,
}

/// Key and value byte offsets of selected rows, one scalar per row.
pub type Selection = KeyValue<Slice<HbmTensor<i32, m![1], m![1]>>>;

/// Borrow key and value storage without copying or changing its physical rows.
#[derive(DeviceSend)]
pub struct Inspect;

impl<'s, W: AxisName, D: Scalar> Module<(&mut Device, &'s KeyValue<RowStorage<W, D>>)> for Inspect {
    type Output = (&'s RowStorage<W, D>, &'s RowStorage<W, D>);

    fn forward(&self, (_, cache): (&mut Device, &'s KeyValue<RowStorage<W, D>>)) -> Self::Output {
        (&cache.0, &cache.1)
    }
}

impl<'s, W: AxisName, N: AxisName, D: Scalar> Module<(&mut Device, &'s Access<W, N, D>)>
    for Inspect
{
    type Output = (&'s RowStorage<W, D>, &'s RowStorage<W, D>);

    fn forward(&self, (device, cache): (&mut Device, &'s Access<W, N, D>)) -> Self::Output {
        self.forward((device, &cache.0))
    }
}

impl<W: AxisName, N: AxisName, D: MaterializableScalar>
    Module<(
        &mut Device,
        (
            &mut (&mut KeyValue<RowStorage<W, D>>, &(Offsets<N>, Length)),
            &Rows<W, N, D>,
            &Rows<W, N, D>,
        ),
    )> for Store
{
    type Output = ();

    fn forward(
        &self,
        (device, ((cache, write), keys, values)): (
            &mut Device,
            (
                &mut (&mut KeyValue<RowStorage<W, D>>, &(Offsets<N>, Length)),
                &Rows<W, N, D>,
                &Rows<W, N, D>,
            ),
        ),
    ) {
        Scatter.forward((device, (&mut **cache, *write, (keys, values))))
    }
}

impl<W: AxisName, Q: AxisName, D: MaterializableScalar>
    Module<(&mut Device, (&RowStorage<W, D>, &Offsets<Q>))> for Gather
{
    type Output = KeyValue<Rows<W, Q, D>>;

    fn forward(
        &self,
        (device, (storage, offsets)): (&mut Device, (&RowStorage<W, D>, &Offsets<Q>)),
    ) -> Self::Output {
        (
            Select::<W, D>::default().forward((&mut *device, (storage, &offsets.0))),
            Select::<W, D>::default().forward((device, (storage, &offsets.1))),
        )
    }
}

impl<W: AxisName, Q: AxisName, D: MaterializableScalar>
    Module<(
        &mut Device,
        (
            &mut RowStorage<W, D>,
            &HbmTensor<i32, m![1], m![Q]>,
            &Length,
            &Rows<W, Q, D>,
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        _: (
            &mut Device,
            (
                &mut RowStorage<W, D>,
                &HbmTensor<i32, m![1], m![Q]>,
                &Length,
                &Rows<W, Q, D>,
            ),
        ),
    ) {
        todo!("NPU scatter of only the first count rows to distinct physical byte offsets")
    }
}

impl<W: AxisName, Q: AxisName, D: MaterializableScalar>
    Module<(
        &mut Device,
        (
            &mut RowStorage<W, D>,
            &(Offsets<Q>, Length),
            (&Rows<W, Q, D>, &Rows<W, Q, D>),
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        (device, (storage, (offsets, count), (keys, values))): (
            &mut Device,
            (
                &mut RowStorage<W, D>,
                &(Offsets<Q>, Length),
                (&Rows<W, Q, D>, &Rows<W, Q, D>),
            ),
        ),
    ) {
        self.forward((&mut *device, (&mut *storage, &offsets.0, count, keys)));
        self.forward((device, (storage, &offsets.1, count, values)));
    }
}

impl<W: AxisName, Q: AxisName, D: MaterializableScalar>
    Module<(
        &mut Device,
        (
            &mut RowStorage<W, D>,
            &(Offsets<Q>, Length),
            &Rows<W, Q, D>,
            &Rows<W, Q, D>,
        ),
    )> for Store
{
    type Output = ();

    fn forward(
        &self,
        (device, (storage, write, keys, values)): (
            &mut Device,
            (
                &mut RowStorage<W, D>,
                &(Offsets<Q>, Length),
                &Rows<W, Q, D>,
                &Rows<W, Q, D>,
            ),
        ),
    ) {
        Scatter.forward((device, (storage, write, (keys, values))))
    }
}
