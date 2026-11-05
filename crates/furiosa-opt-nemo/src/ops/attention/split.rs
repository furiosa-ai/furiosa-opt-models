use super::{Chip, HeadSlice, Heads, Single};
use crate::Module;
use crate::ops::layout::{Dm, Reshape, Rows as Layout, axes::Lanes};
use crate::ops::normalization::Spread;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Unpacks head-major QKV rows from the feature slices a projection leaves them in.
#[derive(DeviceSend)]
pub struct Split<Width, KvHeads, Groups, Head>(PhantomData<(Width, KvHeads, Groups, Head)>);

impl<Width, KvHeads, Groups, Head> Split<Width, KvHeads, Groups, Head> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Width, KvHeads, Groups, Head> Default for Split<Width, KvHeads, Groups, Head> {
    fn default() -> Self {
        Self::new()
    }
}

/// A launch's query, key or value rows as the split leaves them, its live tiles kept.
type Part<Rows, KvHeads, Group, Head> = Dm<Lanes<Rows>, Heads<Rows, KvHeads, Group, Head>>;

type Unpacked<Rows, KvHeads, Groups, Head> = (
    Part<Rows, KvHeads, Groups, Head>,
    Part<Rows, KvHeads, Single, Head>,
    Part<Rows, KvHeads, Single, Head>,
);

#[derive(Clone, Debug)]
struct HeadCount<Groups>(PhantomData<Groups>);

impl<Groups: AxisName> AxisName for HeadCount<Groups> {
    const NAME: Ident = Ident::new("QkvRows");
    const SIZE: usize = Groups::SIZE + 2;
}

type Held<Groups> = Symbol<HeadCount<Groups>>;
type Lane<Rows> = m![{ Layout<Rows>::Lanes } # 256];

type Staged<Width, Rows> =
    DmTensor<bf16, Chip, m![1 # 2], Lane<Rows>, m![{ crate::ops::linear::Groups<Rows> }, Width]>;

type HeadMajor<Rows, KvHeads, HeadRows, Head> =
    m![{ crate::ops::linear::Groups<Rows> }, KvHeads, { HeadRows }, Head];

type Tiled<Rows, KvHeads, Groups, Head, Count> =
    HeadMajor<Rows, KvHeads, m![{ Held<Groups> } = ({ Count }) # ({ Held<Groups> })], Head>;

type Selected<Rows, KvHeads, Groups, Head, Count> = DmTensor<
    bf16,
    Chip,
    m![1 # 2],
    HeadSlice<Rows, KvHeads>,
    m![{ crate::ops::linear::Groups<Rows> }, { Held<Groups> } = ({ Count }), Head],
>;

/// The query, key and value heads of every row, in the feature slices a projection leaves them
/// in.
impl<'d, Width: AxisName, KvHeads: AxisName, Groups: AxisName, Head: AxisName, Rows: AxisName>
    Module<(&'d mut Device, Dm<Lanes<Rows>, Spread<Width, Rows>>)>
    for Split<Width, KvHeads, Groups, Head>
{
    type Output = Unpacked<Rows, KvHeads, Groups, Head>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, Dm<Lanes<Rows>, Spread<Width, Rows>>),
    ) -> Self::Output {
        let mut rows: Staged<Width, Rows> = DmTensor::new();
        input
            .value
            .view()
            .to_dm_view(&mut device.tdma, rows.view_mut());
        const {
            assert!(Width::SIZE == KvHeads::SIZE * (Groups::SIZE + 2) * Head::SIZE);
        }
        // SAFETY: the fused width is head-major within every row. Relabeling preserves cells.
        let rows = unsafe {
            rows.reshape::<Chip, m![1 # 2], Lane<Rows>, HeadMajor<Rows, KvHeads, Held<Groups>, Head>>()
        };
        let mut query: Selected<Rows, KvHeads, Groups, Head, Symbol<Groups>> = DmTensor::new();
        // SAFETY: query starts at offset zero. The omitted K/V rows remain padding, so each
        // head and row retains its original stride and the view stays within the allocation.
        unsafe {
            rows.view().reshape::<
                Chip,
                m![1 # 2],
                Lane<Rows>,
                Tiled<Rows, KvHeads, Groups, Head, Symbol<Groups>>,
            >()
        }
        .to_dm_view(&mut device.tdma, query.view_mut());
        let mut key: Selected<Rows, KvHeads, Groups, Head, m![1]> = DmTensor::new();
        rows.view()
            .tile::<Held<Groups>, 1, Tiled<Rows, KvHeads, Groups, Head, m![1]>>(Groups::SIZE)
            .to_dm_view(&mut device.tdma, key.view_mut());
        let mut value: Selected<Rows, KvHeads, Groups, Head, m![1]> = DmTensor::new();
        rows.view()
            .tile::<Held<Groups>, 1, Tiled<Rows, KvHeads, Groups, Head, m![1]>>(Groups::SIZE + 1)
            .to_dm_view(&mut device.tdma, value.view_mut());
        // SAFETY: each result only renames the selected head rows, preserving their order.
        let (query, key, value) = unsafe { (query.reshape(), key.reshape(), value.reshape()) };
        (
            Dm {
                value: query,
                tiles: input.tiles,
            },
            Dm {
                value: key,
                tiles: input.tiles,
            },
            Dm {
                value,
                tiles: input.tiles,
            },
        )
    }
}

/// The rows a projection of one head kind leaves in feature slices, as `Heads` heads of
/// `Groups` query heads each; each row keeps its lane and its heads their order.
impl<'d, Width: AxisName, Heads: AxisName, Groups: AxisName, Head: AxisName, Rows: AxisName>
    Module<(&'d mut Device, Dm<Lanes<Rows>, Spread<Width, Rows>>)>
    for Reshape<m![Heads, Groups, Head]>
{
    type Output = Dm<Lanes<Rows>, super::Heads<Rows, Heads, Groups, Head>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, Dm<Lanes<Rows>, Spread<Width, Rows>>),
    ) -> Self::Output {
        const { assert!(Width::SIZE == Heads::SIZE * Groups::SIZE * Head::SIZE) };
        let mut rows: Staged<Width, Rows> = DmTensor::new();
        input
            .value
            .view()
            .to_dm_view(&mut device.tdma, rows.view_mut());
        let mut heads: super::Heads<Rows, Heads, Groups, Head> = DmTensor::new();
        // SAFETY: the features are head-major within every row. Relabeling preserves cells.
        unsafe {
            rows.view().reshape::<
                Chip,
                m![1 # 2],
                Lane<Rows>,
                m![{ crate::ops::linear::Groups<Rows> }, Heads, Groups, Head],
            >()
        }
        .to_dm_view(&mut device.tdma, heads.view_mut());
        Dm {
            value: heads,
            tiles: input.tiles,
        }
    }
}

/// The rows a key or value projection leaves in feature slices, as `Heads` heads: the view above
/// with one head in each group.
impl<'d, Width: AxisName, Heads: AxisName, Head: AxisName, Rows: AxisName>
    Module<(&'d mut Device, Dm<Lanes<Rows>, Spread<Width, Rows>>)> for Reshape<m![Heads, Head]>
{
    type Output = Dm<Lanes<Rows>, super::Heads<Rows, Heads, Single, Head>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, Dm<Lanes<Rows>, Spread<Width, Rows>>),
    ) -> Self::Output {
        Reshape::<m![Heads, Single, Head]>::default().forward((device, input))
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;

    use crate::ops::attention::geometry::{ProjectionWidth, QueryGroups};
    use crate::ops::attention::tests::{Head, KvHeads};
    use crate::ops::layout::axes::{Chip, Eight, Single, Sixteen};

    use super::Split;
    use crate::Module;

    #[device(chip = 1)]
    fn split_rows<Width: AxisName, Groups: AxisName, Rows: AxisName>(
        device: &mut Device,
        input: &HbmTensor<bf16, Chip, m![Rows, Width]>,
        live: &crate::compose::Count,
        split: &Split<Width, KvHeads, Groups, Head>,
    ) -> (
        HbmTensor<bf16, Chip, m![Rows, KvHeads, Groups, Head]>,
        HbmTensor<bf16, Chip, m![Rows, KvHeads, Head]>,
        HbmTensor<bf16, Chip, m![Rows, KvHeads, Head]>,
    ) {
        let input: crate::ops::normalization::Spread<Width, Rows> = input.to_dm(&mut device.tdma);
        let input = crate::ops::layout::Dm {
            value: input,
            tiles: crate::ops::layout::Live::<Rows>::read(live),
        };
        let (query, key, value) = split.forward((&mut *device, input));
        (
            query.value.to_hbm(&mut device.tdma),
            key.value.to_hbm(&mut device.tdma),
            value.value.to_hbm(&mut device.tdma),
        )
    }

    async fn separates<Width: AxisName, Groups: AxisName, Rows: AxisName>(live: usize) {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(split_rows.topology()).unwrap();
        let values = (0..Rows::SIZE * Width::SIZE)
            .map(|index| bf16::from_f32(((index * 37) % 251) as f32))
            .collect::<Vec<_>>();
        let input = HostTensor::<bf16, m![Rows, Width]>::from_vec(values.clone())
            .to_hbm::<Chip, m![Rows, Width]>(&mut device.pdma)
            .await
            .unwrap();
        let count = crate::compose::tests::count(live)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let (query, key, value): (
            HbmTensor<bf16, Chip, m![Rows, KvHeads, Groups, Head]>,
            HbmTensor<bf16, Chip, m![Rows, KvHeads, Head]>,
            HbmTensor<bf16, Chip, m![Rows, KvHeads, Head]>,
        ) = launch(
            split_rows,
            (
                &mut device,
                &input,
                &count,
                &Split::<Width, KvHeads, Groups, Head>::new(),
            ),
        )
        .await
        .unwrap();
        let actual = [
            query
                .to_host::<m![Rows, KvHeads, Groups, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec(),
            key.to_host::<m![Rows, KvHeads, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec(),
            value
                .to_host::<m![Rows, KvHeads, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec(),
        ];
        let groups = Groups::SIZE;
        for (part, (begin, len)) in [(0, groups), (groups, 1), (groups + 1, 1)]
            .into_iter()
            .enumerate()
        {
            let width = KvHeads::SIZE * len * Head::SIZE;
            let expected = values
                .chunks_exact((groups + 2) * Head::SIZE)
                .flat_map(|head| {
                    head[begin * Head::SIZE..(begin + len) * Head::SIZE]
                        .iter()
                        .copied()
                })
                .take(live * width)
                .collect::<Vec<_>>();
            assert_eq!(actual[part][..live * width], expected[..], "part {part}");
        }
    }

    type Envelope = crate::ops::layout::axes::Rows<8>;
    type Factors = HbmTensor<f32, Chip, m![Envelope, 1 # 8]>;

    /// Envelope rows' query and key projections as heads, the queries times each row's temperature.
    #[device(chip = 1)]
    fn scaled_heads(
        device: &mut Device,
        (query, key): (
            &HbmTensor<bf16, Chip, m![Envelope, QueryWidth]>,
            &HbmTensor<bf16, Chip, m![Envelope, KeyWidth]>,
        ),
        (live, factors): (&crate::compose::Count, &Factors),
    ) -> (
        HbmTensor<bf16, Chip, m![Envelope, KvHeads, Four, Head]>,
        HbmTensor<bf16, Chip, m![Envelope, KvHeads, Head]>,
    ) {
        use crate::ops::layout::{Dm, Live, Reshape};
        let tiles = Live::<Envelope>::read(live);
        let query: crate::ops::normalization::Spread<QueryWidth, Envelope> =
            query.to_dm(&mut device.tdma);
        let key: crate::ops::normalization::Spread<KeyWidth, Envelope> =
            key.to_dm(&mut device.tdma);
        let query = Reshape::<m![KvHeads, Four, Head]>::default().forward((
            &mut *device,
            Dm {
                value: query,
                tiles,
            },
        ));
        let key = Reshape::<m![KvHeads, Head]>::default()
            .forward((&mut *device, Dm { value: key, tiles }));
        let (query, key) =
            crate::ops::multiply::QueryScale.forward((&mut *device, ((query, key), factors)));
        (
            query.value.to_hbm(&mut device.tdma),
            key.value.to_hbm(&mut device.tdma),
        )
    }

    axes![QueryWidth = 4096, KeyWidth = 1024];
    type Four = QueryGroups<4>;

    /// Each live row's projected features read as heads in projection order, and its queries times
    /// its own position's temperature, rounded to BF16 once; the keys keep their values.
    #[tokio::test(flavor = "current_thread")]
    async fn scales_queries_per_row() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(scaled_heads.topology()).unwrap();
        let values = |count: usize, seed: usize| {
            (0..count)
                .map(|at| bf16::from_f32(((at * 37 + seed) % 251) as f32 / 31.0 - 4.0))
                .collect::<Vec<_>>()
        };
        let (queries, keys) = (values(8 * 4096, 3), values(8 * 1024, 7));
        let temperatures = [
            1.0, 1.0703125, 1.109375, 1.0, 1.1328125, 1.0703125, 1.0, 1.15625,
        ];
        let query = HostTensor::<bf16, m![Envelope, QueryWidth]>::from_vec(queries.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let key = HostTensor::<bf16, m![Envelope, KeyWidth]>::from_vec(keys.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let factors: Factors = HostTensor::<f32, m![Envelope, 1 # 8]>::from_vec(
            temperatures.iter().flat_map(|&t: &f32| [t; 8]),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        for live in [8, 3] {
            let count = crate::compose::tests::count(live)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let (scaled, viewed) = launch(
                scaled_heads,
                (&mut device, (&query, &key), (&count, &factors)),
            )
            .await
            .unwrap();
            let scaled = scaled
                .to_host::<m![Envelope, KvHeads, Four, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let viewed = viewed
                .to_host::<m![Envelope, KvHeads, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let bits = |values: &[bf16]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            for row in 0..live {
                let expected: Vec<bf16> = queries[row * 4096..][..4096]
                    .iter()
                    .map(|q| bf16::from_f32(q.to_f32() * temperatures[row]))
                    .collect();
                assert_eq!(
                    bits(&scaled[row * 4096..][..4096]),
                    bits(&expected),
                    "query row {row} of {live}"
                );
                assert_eq!(
                    bits(&viewed[row * 1024..][..1024]),
                    bits(&keys[row * 1024..][..1024]),
                    "key row {row} of {live}"
                );
            }
        }
    }

    /// Two and four query heads per KV head unpack at one, eight and sixteen rows, and the live
    /// rows unpack when a launch's second row group is idle.
    #[tokio::test]
    async fn splits_each_head() {
        separates::<ProjectionWidth<4096>, QueryGroups<2>, Single>(1).await;
        separates::<ProjectionWidth<4096>, QueryGroups<2>, Eight>(8).await;
        separates::<ProjectionWidth<4096>, QueryGroups<2>, Sixteen>(16).await;
        separates::<ProjectionWidth<4096>, QueryGroups<2>, Sixteen>(3).await;
        separates::<ProjectionWidth<6144>, QueryGroups<4>, Single>(1).await;
        separates::<ProjectionWidth<6144>, QueryGroups<4>, Eight>(8).await;
        separates::<ProjectionWidth<6144>, QueryGroups<4>, Sixteen>(16).await;
        separates::<ProjectionWidth<6144>, QueryGroups<4>, Sixteen>(3).await;
    }
}
