use super::*;
use crate::ops::{attention::cache::Scatter, layout::Dm};

type WriteSpace = crate::compose::Tile<Rows, Rows>;

type Slices = m![Tile % 64 / 16, Hidden / 80];
/// A pair's two consecutive waves of the slice-blocked state, named by the pair's rows.
type PairWaves = m![Tile / 64 #{!} Waves, Hidden / 80, Tile % 64, Hidden % 80];
const _: () = assert!(Slices::SIZE == 256);
const _: () = assert!(KvHeads::SIZE == AttentionHeads::SIZE && Head::SIZE == AttentionHead::SIZE);

/// A launch's token offsets, embedded as `pairs` pairs of 64-row waves.
pub(super) struct Embedded<'a> {
    pub(super) ids: &'a Offsets,
    pub(super) pairs: usize,
}

/// Embeds every live row: each 128-row gather lands one wave a cluster and stores both waves into
/// the slice-blocked state, so a slice writes its 16 rows of a hidden group as one run.
impl Module<(&mut Device, Embedded<'_>, &mut HiddenRows)>
    for crate::models::nemo::parameters::Embedding
{
    type Output = ();

    fn forward(
        &self,
        (device, Embedded { ids, pairs }, output): (&mut Device, Embedded<'_>, &mut HiddenRows),
    ) {
        type Ids = DmTensor<i32, Chips, m![1 # 2], m![Capacity / 16], m![Capacity % 16]>;
        let ids: Ids = ids.to_dm(&mut device.tdma);
        let offsets: Ids = device
            .main
            .begin(ids.view())
            .fetch::<m![Capacity % 16 / 8], m![Capacity % 8]>()
            .collect::<m![Capacity % 16 / 8], m![Capacity % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::MulInt, 10240_i32)
            .vector_final()
            .commit_trim::<m![Capacity % 8]>()
            .commit();
        let offsets: Offsets = offsets.to_hbm(&mut device.tdma);
        for pair in 0..pairs {
            let offsets = offsets
                .view()
                .tile::<m![Capacity], 128, m![Capacity = 128 # Capacity]>(pair * 128);
            // SAFETY: the offset tile keeps its capacity stride and consecutive token order.
            let offsets = unsafe { offsets.reshape::<Chips, m![Tile # Capacity]>() };
            let mut indices: HbmTensor<i32, Chips, m![Tile]> = HbmTensor::new();
            offsets.to_hbm_view(&mut device.tdma, indices.view_mut());
            let rows: DmTensor<bf16, Chips, m![Tile / 64], Slices, m![Tile % 16, Hidden % 80]> =
                self.weight()
                    .gather::<m![Vocabulary], m![Hidden]>()
                    .by_byte_offsets(&indices)
                    .to_dm(&mut device.tdma);
            let waves = output
                .view_mut()
                .tile::<m![Waves], 2, m![Waves = 2 #{!} Waves, { Blocked }]>(2 * pair);
            // SAFETY: the two waves are consecutive, so their rows are the pair's 128 rows in
            // order; only the wave and row labels are renamed.
            let waves = unsafe { waves.reshape::<Chips, PairWaves>() };
            rows.view().to_hbm_view(&mut device.tdma, waves);
        }
    }
}

/// A wave's K and V rows as records, one row a slice holding both of its cluster's heads.
type WaveRecords = DmTensor<
    f8e4m3,
    Chips,
    m![AttentionHeads / 2],
    m![Rows # 256],
    m![AttentionHeads % 2, Kv, AttentionHead],
>;

/// Decode wave 0's pool offsets, the same at every layer, so a step copies them once: its
/// leading 8, 16, 32 and 64 rows'.
pub(super) struct Wave {
    one: HbmTensor<i32, Chips, m![Rows % 8]>,
    two: HbmTensor<i32, Chips, m![Rows % 16]>,
    four: HbmTensor<i32, Chips, m![Rows % 32]>,
    all: HbmTensor<i32, Chips, m![Rows]>,
}

impl Wave {
    pub(super) fn copy(device: &mut Device, input: &Input) -> Self {
        Self {
            one: leading::<8, m![Capacity = 8 # Capacity], _, m![Rows % 8 # Capacity]>(
                device,
                (&input.keys, 0),
            ),
            two: leading::<16, m![Capacity = 16 # Capacity], _, m![Rows % 16 # Capacity]>(
                device,
                (&input.keys, 0),
            ),
            four: leading::<32, m![Capacity = 32 # Capacity], _, m![Rows % 32 # Capacity]>(
                device,
                (&input.keys, 0),
            ),
            all: entries(device, input, 0),
        }
    }

    /// Wave 0's `live` row groups of `quantized` into the pool at layer `entry`.
    pub(super) fn write(
        &self,
        device: &mut Device,
        (pool, entry): (&mut attention::Pool, usize),
        quantized: Quantized,
        live: Live,
    ) {
        let records = records(device, quantized);
        for _ in 0..live.one {
            Scatter.forward((
                &mut *device,
                (
                    &mut *pool,
                    (entry, &self.one),
                    Dm::<WriteSpace, _> {
                        tiles: crate::compose::Tiles::over(8),
                        value: records
                            .view()
                            .slice_tile::<m![Rows / 8], 1, m![Rows / 8 = 1 # 32, Rows % 8]>(0),
                    },
                ),
            ));
        }
        for _ in 0..live.two {
            Scatter.forward((
                &mut *device,
                (
                    &mut *pool,
                    (entry, &self.two),
                    Dm::<WriteSpace, _> {
                        tiles: crate::compose::Tiles::over(16),
                        value: records
                            .view()
                            .slice_tile::<m![Rows / 16], 1, m![Rows / 16 = 1 # 16, Rows % 16]>(0),
                    },
                ),
            ));
        }
        for _ in 0..live.four {
            Scatter.forward((
                &mut *device,
                (
                    &mut *pool,
                    (entry, &self.four),
                    Dm::<WriteSpace, _> {
                        tiles: crate::compose::Tiles::over(32),
                        value: records
                            .view()
                            .slice_tile::<m![Rows / 32], 1, m![Rows / 32 = 1 # 8, Rows % 32]>(0),
                    },
                ),
            ));
        }
        for _ in 0..live.all {
            Scatter.forward((
                &mut *device,
                (
                    &mut *pool,
                    (entry, &self.all),
                    Dm::<WriteSpace, _> {
                        tiles: crate::compose::Tiles::over(Rows::SIZE),
                        value: records.view(),
                    },
                ),
            ));
        }
    }
}

/// Decode wave `wave`'s pool offsets, as its records' scatter index tensor.
pub(super) fn entries(
    device: &mut Device,
    input: &Input,
    wave: usize,
) -> HbmTensor<i32, Chips, m![Rows]> {
    leading::<64, m![Capacity = 64 # Capacity], _, m![Rows # Capacity]>(
        device,
        (&input.keys, 64 * wave),
    )
}

/// A decode wave's `quantized` K and V into the pool at layer `entry` by its `entries`.
pub(super) fn write_wave(
    device: &mut Device,
    (pool, entry): (&mut attention::Pool, usize),
    quantized: Quantized,
    entries: &HbmTensor<i32, Chips, m![Rows]>,
) {
    let records = records(device, quantized);
    Scatter.forward((
        device,
        (
            pool,
            (entry, entries),
            Dm::<WriteSpace, _> {
                tiles: crate::compose::Tiles::over(Rows::SIZE),
                value: records.view(),
            },
        ),
    ));
}

/// A wave's K or V rows in E4M3, still one head of one row a slice.
type Rotated = DmTensor<f8e4m3, Chips, qkv::PairCluster, qkv::PairSlice, m![1, Head]>;

/// A wave's K and V rows in E4M3, as `quantize` leaves them for `records`.
pub(super) struct Quantized(Rotated, Rotated);

/// A wave's K and V rows in E4M3: Main work only, so the records' copies can queue later.
pub(super) fn quantize(
    device: &mut Device,
    (key, value): (&qkv::WaveRows, &qkv::WaveRows),
) -> Quantized {
    Quantized(e4m3(device, key), e4m3(device, value))
}

fn e4m3(device: &mut Device, rows: &qkv::WaveRows) -> Rotated {
    device
        .main
        .begin(rows.view())
        .fetch::<m![1, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1, Head / 8], m![Head % 8]>()
        .cast::<f8e4m3, m![Head % 8 # 32]>()
        .commit_trim::<m![Head % 8]>()
        .commit()
}

/// The quantized rows copied into the records.
fn records(device: &mut Device, Quantized(key, value): Quantized) -> WaveRecords {
    let mut records = WaveRecords::new();
    copy(device, key, (&mut records, 0));
    copy(device, value, (&mut records, 1));
    records
}

/// `rows` copied into half `kv` of every row's record: 0 the keys, 1 the values.
fn copy(device: &mut Device, rows: Rotated, (records, kv): (&mut WaveRecords, usize)) {
    // SAFETY: the K/V heads, rows and features keep their order under the cache's axis names.
    let rows = unsafe {
        rows.reshape::<Chips, m![AttentionHeads / 2], m![AttentionHeads % 2, Rows # 128], m![
            AttentionHead
        ]>()
    };
    rows.view().to_dm_view(
        &mut device.tdma,
        records
            .view_mut()
            .tile::<m![Kv], 1, m![AttentionHeads % 2, Kv = 1 #{!} 2, AttentionHead]>(kv),
    );
}

impl crate::ops::attention::cache::write::Mapping for crate::compose::Tile<Rows, Rows> {
    type Page = m![Page];
    type Payload = m![AttentionHeads, Kv, AttentionHead];
}

/// `LEN` rows' pool byte offsets from capacity row `at` (tile `T`, renamed `P`), as a scatter's
/// index tensor.
fn leading<const LEN: usize, T: M, R: M, P: M>(
    device: &mut Device,
    (offsets, at): (&Offsets, usize),
) -> HbmTensor<i32, Chips, R> {
    let offsets = offsets.view().tile::<m![Capacity], LEN, T>(at);
    // SAFETY: the offset tile names exactly its source rows in the same order.
    let offsets = unsafe { offsets.reshape::<Chips, P>() };
    let mut indices: HbmTensor<i32, Chips, R> = HbmTensor::new();
    offsets.to_hbm_view(&mut device.tdma, indices.view_mut());
    indices
}
