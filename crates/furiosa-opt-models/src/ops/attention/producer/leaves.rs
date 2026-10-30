//! The catalog's attention leaves at the producer's geometry: the head-major projection, its
//! split, the query and key head norms and the rotation.
//! - Each leaf runs the producer's contraction, reduction or rotation on the producer's slices;
//!   those slices are the types they hand on.
//! - A head norm sums a head's squares over the slices the head spans: eight features a slice
//!   for a query head, four for a key head, then one inter-slice sum.

use super::key_value::{Weight, load, order, project};
use super::*;
use crate::Op;
use crate::ops::attention::{Split, split};
use crate::ops::linear;
use crate::ops::linear::Affine;
use crate::ops::normalization::Spread;
use crate::ops::rotary::{Rotary, Rotation};
use crate::tensor::form;
use crate::tensor::{
    axis::{self, Lanes, Single},
    live::Live,
    memory::Hbm,
};
use std::borrow::Borrow;

type Heads<R, KvHeads, Group, Head> = super::super::Heads<R, KvHeads, Group, Head>;

type Rows<'a, R> = DmTensorView<'a, bf16, Chip, m![2], m![Hidden / 16 # 256], m![R, Hidden % 16]>;

/// A row's key heads, or its value heads, four features of a head a slice as the key norm reads
/// them.
pub(crate) type Key<KvHeads, Head, R> =
    DmTensor<bf16, Chip, m![2], KeySlice<KvHeads, Head>, m![R, Head % 4]>;

/// Every row's query heads in the slices their norm reduces over, its key and value heads in
/// theirs.
pub struct Projection<R: AxisName, KvHeads: AxisName, Head: AxisName> {
    query: Query<KvHeads, Head, R>,
    key: Key<KvHeads, Head, R>,
    value: Key<KvHeads, Head, R>,
}

impl<R: AxisName, KvHeads: AxisName, Head: AxisName> Projection<R, KvHeads, Head> {
    /// Projects the query heads of row tile `at`, `Tile` rows.
    fn queries<Tile: AxisName>(
        &mut self,
        device: &mut Device,
        weight: &QueryWeight<KvHeads, Head>,
        rows: Rows<'_, R>,
        at: usize,
    ) {
        let rows = Self::tile::<Tile>(rows, at);
        let query = contract_query::<KvHeads, Head, Tile>(device, weight, rows);
        // SAFETY: the destination supplies the outer row offset of this unchanged local row tile.
        let query = unsafe {
            query.view().reshape::<
                Chip, Cluster<KvHeads>, QueryReduced<KvHeads, Head>, m![R % Tile, Head % 8]
            >()
        };
        query.to_dm_view(
            &mut device.tdma,
            self.query.view_mut().tile::<m![R / Tile], 1, m![
                R / Tile = 1 #{!} (R / Tile), R % Tile, Head % 8
            ]>(at),
        );
    }

    /// Projects the key and value heads of row tile `at`, `Tile` rows.
    fn pairs<Tile: AxisName>(
        &mut self,
        device: &mut Device,
        weights: (&Weight<KvHeads, Head>, &Weight<KvHeads, Head>),
        rows: Rows<'_, R>,
        at: usize,
    ) {
        let (key, value) = project::<KvHeads, Head, Tile>(
            device,
            weights.0,
            weights.1,
            Self::tile::<Tile>(rows, at),
        );
        let to_dm = |device: &mut Device,
                     ordered: &Key<KvHeads, Head, Tile>,
                     heads: &mut Key<KvHeads, Head, R>| {
            // SAFETY: the destination supplies this unchanged local row tile's outer row offset.
            let ordered = unsafe {
                ordered
                    .view()
                    .reshape::<Chip, m![2], KeySlice<KvHeads, Head>, m![R % Tile, Head % 4]>()
            };
            ordered.to_dm_view(
                &mut device.tdma,
                heads.view_mut().tile::<m![R / Tile], 1, m![
                    R / Tile = 1 #{!} (R / Tile), R % Tile, Head % 4
                ]>(at),
            );
        };
        let key = order::<KvHeads, Head, Tile>(device, &key);
        to_dm(device, &key, &mut self.key);
        let value = order::<KvHeads, Head, Tile>(device, &value);
        to_dm(device, &value, &mut self.value);
    }

    /// Row tile `at` of `Tile` rows, its hidden cells and physical strides unchanged.
    fn tile<'a, Tile: AxisName>(rows: Rows<'a, R>, at: usize) -> Rows<'a, Tile> {
        let selected = rows.tile::<m![R / Tile], 1, m![
            R / Tile = 1 # (R / Tile), R % Tile, Hidden % 16
        ]>(at);
        // SAFETY: the row tile retains its hidden cells and physical strides.
        unsafe {
            selected.reshape::<Chip, m![2], m![Hidden / 16 # 256], m![Tile # R, Hidden % 16]>()
        }
        .unpad::<m![Tile, Hidden % 16]>()
    }
}

/// Every live row group's query, key and value heads, in the producer's row tiles: the query
/// weight first, then the key and value weights.
impl<'x, 'w, KvHeads: AxisName, Head: AxisName, R: AxisName> Op<&'x Live<Spread<Hidden, R>, R>>
    for Affine<split::Packed<KvHeads, m![KvHeads, QkvRows, Head, Hidden], &'w Hbm>, ()>
{
    type Output = Live<Projection<R, KvHeads, Head>, R>;

    fn forward(&self, device: &mut Device, input: &'x Live<Spread<Hidden, R>, R>) -> Self::Output {
        // SAFETY: the width's chunk is sixteen features, so the slices and cells are unchanged.
        let rows = unsafe {
            input
                .value()
                .view()
                .reshape::<Chip, m![2], m![Hidden / 16 # 256], m![R, Hidden % 16]>()
        };
        let live = input.with(());
        let weights = self.weight.0.view();
        let query = load_query(device, weights.clone());
        let mut projection = Projection {
            query: DmTensor::new(),
            key: DmTensor::new(),
            value: DmTensor::new(),
        };
        if live.count::<Lanes<R>>() == <linear::Groups<R> as M>::SIZE {
            #[unroll]
            for at in 0..<m![R / ({ Symbol<axis::Tile<R, 32>> })] as M>::SIZE {
                projection.queries::<axis::Tile<R, 32>>(device, &query, rows.clone(), at);
            }
            let (key, value) = (load(device, weights.clone(), 4), load(device, weights, 5));
            #[unroll]
            for at in 0..<m![R / ({ Symbol<axis::Tile<R, 64>> })] as M>::SIZE {
                projection.pairs::<axis::Tile<R, 64>>(device, (&key, &value), rows.clone(), at);
            }
        } else {
            for group in 0..live.count::<Lanes<R>>() {
                projection.queries::<axis::Tile<R, 8>>(device, &query, rows.clone(), group);
            }
            let (key, value) = (load(device, weights.clone(), 4), load(device, weights, 5));
            for group in 0..live.count::<Lanes<R>>() {
                projection.pairs::<axis::Tile<R, 8>>(device, (&key, &value), rows.clone(), group);
            }
        }
        live.with(projection)
    }
}

/// The query and key heads stay in the slices their norms reduce over; each value head gathers
/// into one slice, as the cache write reads it.
impl<Width: AxisName, KvHeads: AxisName, Head: AxisName, R: AxisName>
    Op<Live<Projection<R, KvHeads, Head>, R>> for Split<Width, KvHeads, Groups, Head>
{
    type Output = (
        Live<Query<KvHeads, Head, R>, R>,
        Live<Key<KvHeads, Head, R>, R>,
        Live<Heads<R, KvHeads, Single, Head>, R>,
    );

    fn forward(
        &self,
        device: &mut Device,
        input: Live<Projection<R, KvHeads, Head>, R>,
    ) -> Self::Output {
        let live = input.with(());
        let Projection { query, key, value } = input.into_value();
        let mut values = DmTensor::new();
        gathered(device, &value)
            .view()
            .to_dm_view(&mut device.tdma, values.view_mut());
        (live.with(query), live.with(key), live.with(values))
    }
}

/// Each query head over the slices holding eight of its features.
impl<'x, KvHeads: AxisName, Head: AxisName, R: AxisName, const EPSILON: usize, T>
    Op<&'x Live<Query<KvHeads, Head, R>, R>>
    for RmsNorm<Head, EPSILON, m![1], form::Weight<m![Head], m![1], T>>
where
    T: Borrow<HbmTensor<bf16, m![1], m![Head]>>,
{
    type Output = Live<Query<KvHeads, Head, R>, R>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<Query<KvHeads, Head, R>, R>,
    ) -> Self::Output {
        input.with(normalize::<
            Head,
            R,
            Cluster<KvHeads>,
            QueryReduced<KvHeads, Head>,
            m![KvHeads % 4, Groups, 16],
            m![8],
            EPSILON,
            T,
        >(device, input.value(), self))
    }
}

/// Each key head over the slices holding four of its features.
impl<'x, KvHeads: AxisName, Head: AxisName, R: AxisName, const EPSILON: usize, T>
    Op<&'x Live<Key<KvHeads, Head, R>, R>>
    for RmsNorm<Head, EPSILON, m![1], form::Weight<m![Head], m![1], T>>
where
    T: Borrow<HbmTensor<bf16, m![1], m![Head]>>,
{
    type Output = Live<Key<KvHeads, Head, R>, R>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<Key<KvHeads, Head, R>, R>,
    ) -> Self::Output {
        input.with(normalize::<
            Head,
            R,
            m![2],
            KeySlice<KvHeads, Head>,
            m![KvHeads, 32],
            m![4],
            EPSILON,
            T,
        >(device, input.value(), self))
    }
}

/// Each row's query and key heads gathered into whole heads and rotated by the producer's
/// contraction against the row's rotation, written in the heads attention reads.
impl<KvHeads: AxisName, Head: AxisName, R: AxisName>
    Op<(
        (
            Live<Query<KvHeads, Head, R>, R>,
            Live<Key<KvHeads, Head, R>, R>,
        ),
        &Rotation<Head, Chip, R>,
    )> for Rotary<Head, Head, bf16, Chip>
{
    type Output = (
        Live<Heads<R, KvHeads, Groups, Head>, R>,
        Live<Heads<R, KvHeads, Single, Head>, R>,
    );

    fn forward(
        &self,
        device: &mut Device,
        ((query, key), rotation): (
            (
                Live<Query<KvHeads, Head, R>, R>,
                Live<Key<KvHeads, Head, R>, R>,
            ),
            &Rotation<Head, Chip, R>,
        ),
    ) -> Self::Output {
        let whole: QueryRotated<KvHeads, Head, R> = device
            .main
            .begin(query.value().view())
            .fetch::<m![R], m![Head % 8]>()
            .switch::<m![KvHeads % 4, Groups, 1 # 16], m![R, Head / 8]>(
                SwitchConfig::CustomBroadcast { ring_size: 16 },
            )
            .collect::<m![R, Head / 8], m![Head % 8 # 16]>()
            .commit_trim::<m![Head % 8]>()
            .commit();
        let mut queries = DmTensor::new();
        rotate::<Head, R, _, _, _>(device, &whole, rotation.low.view(), rotation.high.view())
            .view()
            .to_dm_view(&mut device.tdma, queries.view_mut());
        let whole = gathered(device, key.value());
        let mut keys = DmTensor::new();
        rotate::<Head, R, _, _, _>(device, &whole, rotation.low.view(), rotation.high.view())
            .view()
            .to_dm_view(&mut device.tdma, keys.view_mut());
        (query.with(queries), key.with(keys))
    }
}

/// Each head's four-feature slices gathered into one slice of whole heads.
fn gathered<KvHeads: AxisName, Head: AxisName, R: AxisName>(
    device: &mut Device,
    input: &Key<KvHeads, Head, R>,
) -> KeyRotated<KvHeads, Head, R> {
    device
        .main
        .begin(input.view())
        .fetch::<m![R], m![Head % 4]>()
        .switch::<m![KvHeads, 1 # 32], m![R, Head / 4]>(SwitchConfig::CustomBroadcast {
            ring_size: 32,
        })
        .collect::<m![R, Head / 4], m![Head % 4 # 16]>()
        .commit_trim::<m![Head % 4]>()
        .commit()
}
