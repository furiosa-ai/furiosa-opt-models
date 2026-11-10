//! Resident Q/K/V outputs stay in DM through rotation, then land in retained scratch.
//! The current layer writes its live K/V rows through the native cache stage.
use super::qkv::{self as qkv_resident, QkvProjection, Weight};
use crate::Op;
#[cfg(any(test, feature = "kernel-diagnostics"))]
use crate::models::decoder::attention::{Head, KvHeads, llama::nvfp4::QueryHeads};
use crate::ops::normalization::span::{Owned, Span};
use crate::ops::rotary::{
    self as rope, SplitHalf,
    group::{self, Coefficients, InputHalf, OutputHalf},
};
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;
type Wave = crate::ops::linear::query::Tile;
type Rotations<Head, Slots> = SplitHalf<Head, group::Rotation<Slots, rope::Row<Head>>>;
type Cl<H> = m![H / (H / 2)];
type Sl<H> = m![H % (H / 2) # 16, Wave / 8];
type Heads<H, Head> = DmTensor<bf16, Chip, Cl<H>, Sl<H>, m![Wave % 8, Head]>;

fn landed<H: AxisName, W: AxisName, R: AxisName, Head: AxisName>(
    _device: &mut Device,
    rows: qkv_resident::WaveOutput<W, R>,
) -> Heads<H, Head> {
    const {
        assert!(W::SIZE == H::SIZE * Head::SIZE && Head::SIZE == 128);
    }
    // SAFETY: QKV already landed in the head owners; only axis names change.
    unsafe { rows.reshape() }
}

fn rotate<H: AxisName, Head: AxisName, Slots: AxisName>(
    device: &mut Device,
    rows: Heads<H, Head>,
    rotations: &Rotations<Head, Slots>,
    wave: usize,
) -> Heads<H, Head> {
    let frequency: Coefficients<Cl<H>, Sl<H>, rope::Row<Head>> = rotations.frequency.view()
        .tile::<m![Slots], 1, m![Slots = 1 # Slots, Wave, OutputHalf, InputHalf, {Symbol<rope::Row<Head>>}]>(wave)
        .to_dm(&mut device.tdma);
    let rotation = group::Stage.forward(
        &mut *device,
        &SplitHalf::<Head, _> {
            frequency,
            head: PhantomData,
        },
    );
    // SAFETY: insert the rotation module's unit axis without moving any element.
    let rows = unsafe { rows.reshape::<Chip, Cl<H>, Sl<H>, m![1, Wave % 8, Head]>() };
    let output = rotation.forward(device, &rows);
    // SAFETY: remove the same unit axis, retaining cluster/slice ownership.
    unsafe { output.reshape() }
}

fn place<H: AxisName, Head: AxisName, Reach: AxisName>(
    device: &mut Device,
    rows: &Heads<H, Head>,
    output: &mut HbmTensor<bf16, Chip, m![Reach, H, Head]>,
    start: usize,
) {
    let target = output
        .view_mut()
        .tile::<m![Reach], 128, m![Reach = 128 #{!} Reach, H, Head]>(start);
    // SAFETY: only rename the selected rows; preserve the destination's padded reach stride.
    rows.view().to_hbm_view(&mut device.tdma, unsafe {
        target.reshape::<Chip, m![Wave #{!} Reach, H, Head]>()
    });
}

/// A span's projections and rotation, retained until the current layer attends.
/// Start is 128-row aligned; full tiles are stored but only [start, start + rows)
/// is valid. Attention masks padded keys and zeros padded values.
pub(crate) fn project<
    H: AxisName,
    Q: AxisName,
    K: AxisName,
    Head: AxisName,
    KvHeads: AxisName,
    QueryHeads: AxisName,
    Reach: AxisName,
    Slots: AxisName,
>(
    device: &mut Device,
    qkv: (
        &QkvProjection<H, Q>,
        &QkvProjection<H, K>,
        &QkvProjection<H, K>,
    ),
    prepared: (&Weight<H, Q>, &Weight<H, K>, &Weight<H, K>),
    input: &Owned<H, Span>,
    rotations: &Rotations<Head, Slots>,
    query: &mut HbmTensor<bf16, Chip, m![Reach, Q]>,
    key: &mut HbmTensor<bf16, Chip, m![Reach, KvHeads, Head]>,
    value: &mut HbmTensor<bf16, Chip, m![Reach, KvHeads, Head]>,
    start: usize,
    rows: usize,
) {
    let query_target = query;
    let key_target = key;
    let value_target = value;
    qkv_resident::consume(
        device,
        qkv,
        prepared,
        input,
        rows,
        |device, query, key, value, wave| {
            let at = start + wave * 128;
            let query = landed::<QueryHeads, Q, Span, Head>(device, query);
            let query = rotate(device, query, rotations, at / 128);
            // SAFETY: QWidth is exactly QueryHeads × Head, contiguous in each scratch row.
            let query_target = unsafe {
                query_target
                    .view_mut()
                    .reshape::<Chip, m![Reach, QueryHeads, Head]>()
            };
            // SAFETY: rename the selected wave, preserving the full scratch-row allocation stride.
            query.view().to_hbm_view(&mut device.tdma, unsafe {
                query_target
                    .tile::<m![Reach], 128, m![Reach = 128 #{!} Reach, QueryHeads, Head]>(at)
                    .reshape::<Chip, m![Wave #{!} Reach, QueryHeads, Head]>()
            });
            let key = landed::<KvHeads, K, Span, Head>(device, key);
            let key = rotate(device, key, rotations, at / 128);
            let value = landed::<KvHeads, K, Span, Head>(device, value);
            place(device, &key, key_target, at);
            place(device, &value, value_target, at);
        },
    );
}

/// Exercise the production raw-QKV DM redistribution without RoPE or cache casts.
#[cfg(any(test, feature = "kernel-diagnostics"))]
pub fn inspect_qkv_heads<H: AxisName, Q: AxisName, K: AxisName, R: AxisName>(
    device: &mut Device,
    projections: (
        &QkvProjection<H, Q>,
        &QkvProjection<H, K>,
        &QkvProjection<H, K>,
    ),
    input: &Owned<H, R>,
    rows: usize,
) -> (
    HbmTensor<bf16, Chip, m![R, Q]>,
    HbmTensor<bf16, Chip, m![R, K]>,
    HbmTensor<bf16, Chip, m![R, K]>,
) {
    let (mut qo, mut ko, mut vo) = (HbmTensor::new(), HbmTensor::new(), HbmTensor::new());
    let mut store = |device: &mut Device,
                     q: Heads<QueryHeads, Head>,
                     k: Heads<KvHeads, Head>,
                     v: Heads<KvHeads, Head>,
                     wave: usize| {
        // SAFETY: head and channel axes concatenate to each projection width.
        unsafe {
            q.view()
                .reshape::<Chip, Cl<QueryHeads>, Sl<QueryHeads>, m![Wave % 8, Head]>()
                .to_hbm_view(
                    &mut device.tdma,
                    qo.view_mut()
                        .tile::<m![R], 128, m![R = 128 #{!} R, Q]>(wave * 128)
                        .reshape::<Chip, m![Wave #{!} R, QueryHeads, Head]>(),
                );
            k.view().to_hbm_view(
                &mut device.tdma,
                ko.view_mut()
                    .tile::<m![R], 128, m![R = 128 #{!} R, K]>(wave * 128)
                    .reshape::<Chip, m![Wave #{!} R, KvHeads, Head]>(),
            );
            v.view().to_hbm_view(
                &mut device.tdma,
                vo.view_mut()
                    .tile::<m![R], 128, m![R = 128 #{!} R, K]>(wave * 128)
                    .reshape::<Chip, m![Wave #{!} R, KvHeads, Head]>(),
            );
        }
    };
    {
        let (q, k, v) = (
            qkv_resident::prepare(device, projections.0, rows),
            qkv_resident::prepare(device, projections.1, rows),
            qkv_resident::prepare(device, projections.2, rows),
        );
        qkv_resident::consume(
            device,
            projections,
            (&q, &k, &v),
            input,
            rows,
            |device, q, k, v, wave| {
                let q = landed::<QueryHeads, Q, R, Head>(device, q);
                let k = landed::<KvHeads, K, R, Head>(device, k);
                let v = landed::<KvHeads, K, R, Head>(device, v);
                store(device, q, k, v, wave);
            },
        );
    }
    (qo, ko, vo)
}
