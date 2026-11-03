//! Resident Q/K/V outputs stay in DM through rotation, then land in retained scratch.
//! The current layer writes its live K/V rows through the native cache stage.
use super::panels::{Panels, Prepared, resident as qkv_resident};
use crate::Op;
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
    device: &mut Device,
    rows: qkv_resident::WaveOutput<W, R>,
) -> Heads<H, Head> {
    const {
        assert!(W::SIZE == H::SIZE * Head::SIZE);
    }
    // SAFETY: the 64 projection owners split into heads and head fragments;
    // each cluster retains the same half of the heads. Only axis names change.
    let rows = unsafe {
        rows.reshape::<Chip, Cl<H>,
        m![H % (H / 2), Head / (W / 64), Wave % 64 / 8],
        m![Wave / 64, Wave % 8, Head % (W / 64)]>()
    };
    // Redistribute head fragments and row ownership directly between DM slices.
    // K/V use four of the sixteen padded head slots per cluster, as RoPE expects.
    let mut output = Heads::new();
    rows.view().to_dm_view(&mut device.tdma, output.view_mut());
    output
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
    qkv: (&Panels<H, Q>, &Panels<H, K>, &Panels<H, K>),
    prepared: (&Prepared<H, Q>, &Prepared<H, K>, &Prepared<H, K>),
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
    qkv_resident::consume_normalized(
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
