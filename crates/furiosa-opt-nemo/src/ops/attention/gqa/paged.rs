use super::{Cache, shard};
use crate::ops::attention::{
    cache::{Block, BlockIndex, Blocks},
    geometry::Kv,
};
use crate::{DeviceModule, Module, ops::cast::Convert};
use furiosa_opt_std::prelude::*;

use crate::ops::layout::{BroadcastClusters as Cluster, axes::Chip};

type Slices<Heads> = m![Heads, 1 # (16 / Heads), Cache / 32];
type Merged<Heads> = m![Heads, 1 # (16 / Heads), 1 # 16];
type Query<Queries, Head, Chips = Chip> = HbmTensor<bf16, Chips, m![Queries, Head]>;

/// Cache blocks as the sparse DMA lands them, before element conversion.
type CacheTile<Heads, Head, D = bf16, Chips = Chip> = DmTensor<
    D,
    Chips,
    Cluster,
    m![Heads, 1 # (16 / Heads), BlockIndex / 2],
    m![1, Kv, BlockIndex % 2, Block, Head],
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

/// Grouped-query attention over block-indexed history; `SCALE` holds the FP32 bits of the
/// multiplier applied to every score.
#[derive(DeviceSend)]
pub struct PagedAttention<Heads, Groups, Head, const SCALE: usize, Chips: M = Chip>(
    std::marker::PhantomData<(Heads, Groups, Head, Chips)>,
);

impl<Heads, Groups, Head, const SCALE: usize, Chips: M> Default
    for PagedAttention<Heads, Groups, Head, SCALE, Chips>
{
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

type Partition<Heads, Groups, Head, const SCALE: usize, Chips> = super::partition::Geometry<
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
    Module<(
        &mut Device,
        (
            &Query<Queries, Head, Chip>,
            (&Blocks<Heads, Head, D, Chip>, usize),
            &Read<Windows, Chip>,
        ),
    )> for PagedAttention<Heads, Groups, Head, SCALE, Chip>
where
    Convert<bf16>:
        DeviceModule<CacheTile<Heads, Head, D, Chip>, Output = CacheTile<Heads, Head, bf16, Chip>>,
{
    type Output = Query<Queries, Head, Chip>;

    fn forward(
        &self,
        (device, (query, (cache, entry), read)): (
            &mut Device,
            (
                &Query<Queries, Head, Chip>,
                (&Blocks<Heads, Head, D, Chip>, usize),
                &Read<Windows, Chip>,
            ),
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
                let landed: CacheTile<Heads, Head, D, Chip> = cache
                    .tokens
                    .gather::<m![1], m![Block, Kv, Heads, Head]>()
                    .entry(entry)
                    .by_byte_offsets(&blocks)
                    .prefix::<m![BlockIndex], _>(&lengths, 0)
                    .to_dm(&mut device.tdma);
                let landed = Convert::<bf16>::default().forward((device, landed));
                // Two consecutive blocks fill each shard's 32 token slots.
                let landed: DmTensor<
                    bf16,
                    Chip,
                    Cluster,
                    Slices<Heads>,
                    m![1, Kv, Cache % 32, Head],
                > = unsafe { landed.reshape() };
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
                    landed
                        .view()
                        .tile::<m![Kv], 1, m![1, Kv = 1 # 2, Cache % 32, Head]>(0),
                    landed
                        .view()
                        .tile::<m![Kv], 1, m![1, Kv = 1 # 2, Cache % 32, Head]>(1),
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
    Module<(
        &mut Device,
        (
            &Query<Queries, Head, Chip>,
            &&mut super::super::cache::Layer<'_, Heads, Head, D, Chip>,
            &Read<Windows, Chip>,
        ),
    )> for PagedAttention<Heads, Groups, Head, SCALE, Chip>
where
    Convert<bf16>:
        DeviceModule<CacheTile<Heads, Head, D, Chip>, Output = CacheTile<Heads, Head, bf16, Chip>>,
{
    type Output = Query<Queries, Head, Chip>;

    fn forward(
        &self,
        (device, (query, layer, read)): (
            &mut Device,
            (
                &Query<Queries, Head, Chip>,
                &&mut super::super::cache::Layer<'_, Heads, Head, D, Chip>,
                &Read<Windows, Chip>,
            ),
        ),
    ) -> Self::Output {
        self.forward((device, (query, (&*layer.blocks, layer.entry), read)))
    }
}
