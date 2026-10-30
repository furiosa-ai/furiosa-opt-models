use super::partition::Geometry;
use super::{Attended, Cache, Paged, Windowed, shard};
use crate::ops::index::table::{BlockCounts, BlockIndex};
use crate::tensor::cache::{Block, Blocks, KeyValue};
use crate::tensor::live::Live;
use crate::tensor::scalar::Value;
use crate::{Op, ops::pointwise::cast::Convert};
use furiosa_opt_std::prelude::*;

use crate::tensor::{mapping::BroadcastClusters as Cluster, parallel::Chip};

type Slices<Heads> = m![Heads, 1 # (16 / Heads), Cache / 32];
type Merged<Heads> = m![Heads, 1 # (16 / Heads), 1 # 16];
type Query<Queries, Head, Chips = Chip> = HbmTensor<bf16, Chips, m![Queries, Head]>;

/// Cache blocks as the sparse DMA writes them, before element conversion.
type CacheTile<Heads, Head, D = bf16, Chips = Chip> = DmTensor<
    D,
    Chips,
    Cluster,
    m![Heads, 1 # (16 / Heads), BlockIndex / 2],
    m![1, KeyValue, BlockIndex % 2, Block, Head],
>;

/// Block-table byte offsets, per-window visibility and the count of windows after the first.
#[derive(DeviceSend)]
pub struct Read<Windows: AxisName, Chips: M = Chip> {
    /// Physical block byte offsets in logical window order.
    pub blocks: HbmTensor<i32, Chips, m![Windows, BlockIndex]>,
    /// Number of blocks read by each attention window.
    pub lengths: HbmTensor<i32, Chips, m![Windows, 1]>,
    /// Additive attention mask for each token in each window.
    pub bias: HbmTensor<f32, Chips, m![Windows, Cache]>,
    /// Number of windows after the first; zero still attends to window zero.
    pub rest: HbmTensor<i32, Chips, m![1]>,
}

/// The pool blocks each row of an envelope attends over, as the block-indexed history of a step.
#[derive(DeviceSend)]
pub struct Pages<Rows: AxisName, Windows: AxisName, Chips: M = Chip> {
    /// Byte offsets of pool blocks, per row and window. Every entry must be readable, including
    /// rounded overfetch.
    pub table: HbmTensor<i32, Chips, m![Rows, Windows, BlockIndex]>,
    /// Blocks read per row and window, including the block of the row's own token.
    pub lengths: BlockCounts<Rows, Windows, Chips>,
    /// Additive mask of each token: 0 for the visible prefix, padding past it.
    pub bias: HbmTensor<f32, Chips, m![Rows, Windows, Cache]>,
    /// Windows after the first, which every row attends over at least.
    pub rest: Value<Chips>,
}

type Partition<Heads, Groups, Head, const SCALE: usize, Chips> = Geometry<
    Chips,
    Cluster,
    Slices<Heads>,
    m![Heads, 1 # (16 / Heads), 16],
    Merged<Heads>,
    m![1],
    m![Groups],
    m![Head],
    m![Cache % 32],
    Cache,
    16,
    SCALE,
>;

impl<
    Heads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Windows: AxisName,
    Queries: AxisName,
    const SCALE: usize,
    D: MaterializableScalar + FetchCast<f32>,
    Chip: M,
>
    Op<(
        &Query<Queries, Head, Chip>,
        &Blocks<Heads, Head, D, Chip>,
        &Read<Windows, Chip>,
    )> for Windowed<Heads, Groups, Head, SCALE>
where
    Convert<bf16>: Op<CacheTile<Heads, Head, D, Chip>, Output = CacheTile<Heads, Head, bf16, Chip>>,
{
    type Output = Query<Queries, Head, Chip>;

    fn forward(
        &self,
        device: &mut Device,
        (query, cache, read): (
            &Query<Queries, Head, Chip>,
            &Blocks<Heads, Head, D, Chip>,
            &Read<Windows, Chip>,
        ),
    ) -> Self::Output {
        const {
            assert!(Queries::SIZE == Heads::SIZE * Groups::SIZE);
            assert!(Heads::SIZE > 0 && 16 % Heads::SIZE == 0);
            assert!(Windows::SIZE > 0);
        }
        let query: DmTensor<
            bf16,
            Chip,
            Cluster,
            m![Heads, 1 # (16 / Heads), 16],
            m![1, Groups, Head],
        > = unsafe { query.view().reshape::<Chip, m![Heads, Groups, Head]>() }
            .to_dm(&mut device.tdma);
        // The broadcast copies become the shards that consume the same query.
        let query: DmTensor<bf16, Chip, Cluster, Slices<Heads>, m![1, Groups, Head]> =
            unsafe { query.reshape() };
        let attended = <Partition<Heads, Groups, Head, SCALE, Chip> as shard::Partition>::windows(
            device,
            read.rest.extent::<m![Windows]>(),
            |device, at| {
                let mut blocks: HbmTensor<i32, Chip, m![BlockIndex]> = HbmTensor::new();
                read.blocks
                    .view()
                    .tile::<m![Windows], 1, m![Windows = 1 # Windows, BlockIndex]>(at)
                    .to_hbm_view(&mut device.tdma, blocks.view_mut());
                let mut lengths: HbmTensor<i32, Chip, m![1]> = HbmTensor::new();
                read.lengths
                    .view()
                    .tile::<m![Windows], 1, m![Windows = 1 # Windows, 1]>(at)
                    .to_hbm_view(&mut device.tdma, lengths.view_mut());
                let in_dm: CacheTile<Heads, Head, D, Chip> = cache
                    .tokens
                    .gather::<m![1], m![Block, KeyValue, Heads, Head]>()
                    .by_byte_offsets(&blocks)
                    .prefix::<m![BlockIndex], _>(&lengths, 0)
                    .to_dm(&mut device.tdma);
                let in_dm = Convert::<bf16>::default().forward(device, in_dm);
                // Two consecutive blocks fill each shard's 32 token slots.
                let in_dm: DmTensor<
                    bf16,
                    Chip,
                    Cluster,
                    Slices<Heads>,
                    m![1, KeyValue, Cache % 32, Head],
                > = unsafe { in_dm.reshape() };
                let mut bias: DmTensor<
                    f32,
                    Chip,
                    Cluster,
                    Slices<Heads>,
                    m![1, Groups, Cache % 32],
                > = DmTensor::new();
                read.bias
                    .view()
                    .tile::<m![Windows], 1, m![Windows = 1 # Windows, Cache]>(at)
                    .to_dm_view(&mut device.tdma, bias.view_mut());
                let mut partial =
                    shard::Partial::<Partition<Heads, Groups, Head, SCALE, Chip>>::new();
                let mut lse = shard::Local::<Partition<Heads, Groups, Head, SCALE, Chip>>::new();
                <Partition<Heads, Groups, Head, SCALE, Chip> as shard::Partition>::attend_precise::<
                    m![1],
                    _,
                    _,
                    _,
                    _,
                    _,
                >(
                    device,
                    query.view(),
                    in_dm
                        .view()
                        .tile::<m![KeyValue], 1, m![1, KeyValue = 1 # 2, Cache % 32, Head]>(0),
                    in_dm
                        .view()
                        .tile::<m![KeyValue], 1, m![1, KeyValue = 1 # 2, Cache % 32, Head]>(1),
                    bias.view(),
                    partial.view_mut(),
                    lse.view_mut(),
                );
                (partial, lse)
            },
        );
        let output: HbmTensor<bf16, Chip, m![Heads, Groups, Head]> =
            attended.to_hbm(&mut device.tdma);
        unsafe { output.reshape() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::memory;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![
        Heads = 8,
        Groups = 2,
        Head = 256,
        FullHeads = 1,
        FullGroups = 16,
        FullHead = 512,
        Queries = 16,
        Windows = 3,
        Tokens = 1536
    ];

    const UNIT: usize = 1.0_f32.to_bits() as usize;

    #[device(chip = 1)]
    fn forward<H: AxisName, G: AxisName, D: AxisName, const SCALE: usize>(
        device: &mut Device,
        attention: &Windowed<H, G, D, SCALE>,
        query: &Query<Queries, D>,
        cache: &Blocks<H, D>,
        read: &Read<Windows>,
    ) -> Query<Queries, D> {
        attention.forward(device, (query, cache, read))
    }

    /// Each length's attention, checked against an oracle that applies `SCALE` to the scores.
    async fn check<H: AxisName, G: AxisName, D: AxisName, const SCALE: usize>(
        device: &mut Device,
    ) -> Vec<Vec<bf16>> {
        let scale = f32::from_bits(SCALE as u32) as f64;
        let mut attended = Vec::new();
        let values = (0..Tokens::SIZE * 2 * H::SIZE * D::SIZE)
            .map(|at| {
                let token = at / (2 * H::SIZE * D::SIZE);
                let channel = at % D::SIZE;
                bf16::from_f32(if token >= 1100 {
                    f32::NAN
                } else if at / (H::SIZE * D::SIZE) % 2 == 1 {
                    ((token * 7 + channel * 3) % 19) as f32 / 16.0 - 0.5
                } else {
                    match channel {
                        0 => 32.0 + (token / 512) as f32 / 2.0,
                        1 => (token % 17) as f32 / 4.0 - 2.0,
                        _ => 0.0,
                    }
                })
            })
            .collect::<Vec<_>>();
        let storage = HostBufferSlice::from(&bf16::to_buf(&values))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let stride = Block::SIZE * 2 * H::SIZE * D::SIZE * 2;
        let cache = Blocks {
            tokens: memory::blocks(storage, stride).unwrap().into(),
        };
        let query =
            HostTensor::<bf16, m![Queries, D]>::from_vec((0..Queries::SIZE * D::SIZE).map(|at| {
                bf16::from_f32(match at % D::SIZE {
                    0 => 1.0,
                    1 => (at / D::SIZE % 4 + 1) as f32 / 32.0,
                    _ => 0.0,
                })
            }))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        for length in [31usize, 513, 1100] {
            let read = Read {
                blocks: HostTensor::<i32, m![Windows, BlockIndex]>::from_vec(
                    (0..Tokens::SIZE / Block::SIZE).map(|i| (i * stride) as i32),
                )
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),

                lengths: HostTensor::<i32, m![Windows, 1]>::from_vec(
                    (0..Windows::SIZE)
                        .map(|i| length.saturating_sub(i * 512).min(512).div_ceil(16) as i32),
                )
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
                bias: HostTensor::<f32, m![Windows, Cache]>::from_vec(
                    (0..Tokens::SIZE).map(|i| if i < length { 0.0 } else { f32::NEG_INFINITY }),
                )
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
                rest: HostTensor::<i32, m![1]>::from_vec([((length - 1) / 512) as i32])
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap()
                    .into(),
            };
            let actual = launch(
                forward,
                (
                    &mut *device,
                    &Windowed::<H, G, D, SCALE>::default(),
                    &query,
                    &cache,
                    &read,
                ),
            )
            .await
            .unwrap()
            .to_host::<m![Queries, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
            for head in 0..Queries::SIZE {
                let weights = (0..length)
                    .map(|token| {
                        let base = token * 2 * H::SIZE * D::SIZE + head / G::SIZE * D::SIZE;
                        (scale
                            * ((values[base].to_f32() as f64 - 33.0)
                                + (head % 4 + 1) as f64 / 32.0 * values[base + 1].to_f32() as f64))
                            .exp()
                    })
                    .collect::<Vec<_>>();
                let sum = weights.iter().sum::<f64>();
                for channel in 0..D::SIZE {
                    let expected = weights
                        .iter()
                        .enumerate()
                        .map(|(token, weight)| {
                            let at = (token * 2 + 1) * H::SIZE * D::SIZE
                                + head / G::SIZE * D::SIZE
                                + channel;
                            weight * values[at].to_f32() as f64 / sum
                        })
                        .sum::<f64>();
                    let expected = bf16::from_f32(expected as f32).to_f32();
                    let actual = actual[head * D::SIZE + channel].to_f32();
                    assert!(
                        (actual - expected).abs() < 0.001,
                        "length {length}, head {head}, channel {channel}: {actual} vs {expected}"
                    );
                }
            }
            attended.push(actual);
        }
        attended
    }

    #[tokio::test(flavor = "current_thread")]
    async fn preserves_score_differences_across_blocks() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(forward.topology()).unwrap();
        check::<Heads, Groups, Head, UNIT>(&mut device).await;
        check::<FullHeads, FullGroups, FullHead, UNIT>(&mut device).await;
    }

    /// Two leaves that differ only in their scale attend differently, each as an oracle that
    /// applies its own scale says.
    #[tokio::test(flavor = "current_thread")]
    async fn scores_scale_by_the_leaf_scale() {
        const HALF: usize = 0.5_f32.to_bits() as usize;
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(forward.topology()).unwrap();
        let unit = check::<Heads, Groups, Head, UNIT>(&mut device).await;
        let half = check::<Heads, Groups, Head, HALF>(&mut device).await;
        assert_ne!(unit, half, "the scale did not change the attention");
    }
}

/// Each live row's queries, as attention projects them, attend over the pages its row of the
/// read operand names in the layer's blocks; a row group past the live rows reads nothing.
impl<
    Heads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Windows: AxisName,
    Rows: AxisName,
    const SCALE: usize,
>
    Op<(
        &Live<super::super::Heads<Rows, Heads, Groups, Head>, Rows>,
        &Blocks<Heads, Head>,
        &Pages<Rows, Windows>,
    )> for Windowed<Heads, Groups, Head, SCALE>
{
    type Output = Live<Attended<Rows, Heads, Groups, Head>, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        (query, blocks, read): (
            &Live<super::super::Heads<Rows, Heads, Groups, Head>, Rows>,
            &Blocks<Heads, Head>,
            &Pages<Rows, Windows>,
        ),
    ) -> Self::Output {
        let mut heads = DmTensor::new();
        query
            .value()
            .view()
            .to_dm_view(&mut device.tdma, heads.view_mut());
        let history = Paged {
            blocks,
            table: &read.table,
            lengths: &read.lengths,
        };
        query.with(self.forward(
            device,
            (&heads, &history, &read.bias, &read.rest, query.with(())),
        ))
    }
}
