use super::*;
use crate::modules::generic::Either;
use crate::ops::rotary::group::{OutputHalf, Rotated};
use crate::tensor::live::Spill;

/// A launch's query rows in HBM: `Groups` query heads of `Head` features for each of `KvHeads`.
type QueryRows<All, KvHeads, Groups, Head> = HbmTensor<bf16, Chip, m![All, KvHeads, Groups, Head]>;

/// A launch's key or value rows in HBM: `KvHeads` heads of `Head` features a row.
type HeadRows<All, KvHeads, Head> = HbmTensor<bf16, Chip, m![All, KvHeads, Head]>;

/// The rotated queries of a group's tiles from row `start`.
impl<
    'r,
    G: Group,
    Heads: AxisName,
    Half: AxisName,
    All: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
>
    Spill<
        Device,
        &'r Rotated<Cluster<Heads>, QuerySlices<G, Heads>, QueryLocal<G, Heads>, Half>,
        usize,
    > for QueryRows<All, KvHeads, Groups, Head>
{
    fn spill(
        &mut self,
        context: &mut Device,
        rows: &'r Rotated<Cluster<Heads>, QuerySlices<G, Heads>, QueryLocal<G, Heads>, Half>,
        start: usize,
    ) {
        const { assert!(KvHeads::SIZE * Groups::SIZE == Heads::SIZE) };
        // SAFETY: the KV heads' query groups are the query heads, the head the output half and
        // feature coordinates, in HBM order.
        let query = unsafe {
            self.view_mut()
                .reshape::<Chip, m![All, Heads, OutputHalf, Half]>()
        };
        // SAFETY: BF16 names the same reduced slices as the FP8 contraction.
        let rows = unsafe {
            rows.view().reshape::<Chip, Cluster<Heads>, Reduced<G, Heads, QuerySpread<G, Heads>>,
                m![{ QueryLocal<G, Heads> }, Tile % 8, OutputHalf, Half]>()
        };
        rows.to_hbm_view(&mut context.tdma, G::window(query, start));
    }
}

/// The rotated queries of a period of a pair of tiles or of a single tile.
impl<'r, P, S, All: AxisName, KvHeads: AxisName, Groups: AxisName, Head: AxisName>
    Spill<Device, &'r Either<P, S>, usize> for QueryRows<All, KvHeads, Groups, Head>
where
    Self: Spill<Device, &'r P, usize> + Spill<Device, &'r S, usize>,
{
    fn spill(&mut self, context: &mut Device, rows: &'r Either<P, S>, start: usize) {
        match rows {
            Either::Left(rows) => self.spill(context, rows, start),
            Either::Right(rows) => self.spill(context, rows, start),
        }
    }
}

/// The values, then the rotated keys, of a group's tiles from row `start`.
impl<'r, G: Group, Half: AxisName, All: AxisName, KvHeads: AxisName, Head: AxisName>
    Spill<
        Device,
        (
            &'r Rotated<Cluster<KvHeads>, PairSlices<G, KvHeads>, PairLocal<KvHeads>, Half>,
            &'r Pairs<G, KvHeads, Head>,
        ),
        usize,
    > for (HeadRows<All, KvHeads, Head>, HeadRows<All, KvHeads, Head>)
{
    fn spill(
        &mut self,
        context: &mut Device,
        (keys, values): (
            &'r Rotated<Cluster<KvHeads>, PairSlices<G, KvHeads>, PairLocal<KvHeads>, Half>,
            &'r Pairs<G, KvHeads, Head>,
        ),
        start: usize,
    ) {
        let (key, value) = self;
        // SAFETY: BF16 names the same reduced slices as the FP8 contraction.
        let values = unsafe {
            values.view().reshape::<Chip, Cluster<KvHeads>, Reduced<G, KvHeads, G::Copies>,
                m![KvHeads % (KvHeads / 8), Tile % 8, Head]>()
        };
        values.to_hbm_view(&mut context.tdma, G::window(value.view_mut(), start));
        // SAFETY: Head concatenates the output half and feature coordinates in HBM order.
        let key = unsafe {
            key.view_mut()
                .reshape::<Chip, m![All, KvHeads, OutputHalf, Half]>()
        };
        // SAFETY: BF16 names the same reduced slices as the FP8 contraction.
        let keys = unsafe {
            keys.view().reshape::<Chip, Cluster<KvHeads>, Reduced<G, KvHeads, G::Copies>,
                m![KvHeads % (KvHeads / 8), Tile % 8, OutputHalf, Half]>()
        };
        keys.to_hbm_view(&mut context.tdma, G::window(key, start));
    }
}

/// The values and rotated keys of a period of a pair of tiles or of a single tile; a period's
/// keys and values share its geometry, so no other pairing occurs.
impl<'r, KP, KS, VP, VS, All: AxisName, KvHeads: AxisName, Head: AxisName>
    Spill<Device, (&'r Either<KP, KS>, &'r Either<VP, VS>), usize>
    for (HeadRows<All, KvHeads, Head>, HeadRows<All, KvHeads, Head>)
where
    Self: Spill<Device, (&'r KP, &'r VP), usize> + Spill<Device, (&'r KS, &'r VS), usize>,
{
    fn spill(
        &mut self,
        context: &mut Device,
        rows: (&'r Either<KP, KS>, &'r Either<VP, VS>),
        start: usize,
    ) {
        match rows {
            (Either::Left(keys), Either::Left(values)) => {
                self.spill(context, (keys, values), start)
            }
            (Either::Right(keys), Either::Right(values)) => {
                self.spill(context, (keys, values), start)
            }
            _ => {}
        }
    }
}
