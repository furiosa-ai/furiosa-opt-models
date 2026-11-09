//! Indexed KV row transfers preserve caller-owned allocations and independent key/value offsets.

use super::Block;
use crate::{
    Module,
    ops::{attention::geometry::Kv, indexing::RowStorage},
};
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;

pub type KeyValue<T> = (T, T);
type Rows<Width, Count, D = f32> = HbmTensor<D, m![1], m![Count, Width]>;
type Offsets<Count> = KeyValue<HbmTensor<i32, m![1], m![Count]>>;
type Length = HbmTensor<i32, m![1], m![1]>;

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
