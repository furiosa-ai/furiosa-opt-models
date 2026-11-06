use super::*;
use crate::Op;

/// A period's quantized rows and their row scales in the slices of `KvHeads`, `groups` groups of
/// eight rows: what each of its projections reads.
#[must_use]
pub struct Period<Hidden: AxisName, G: Group, KvHeads: AxisName> {
    own: Input<Hidden, G>,
    scales: RowGroups<G, KvHeads>,
    groups: usize,
}

impl<Hidden: AxisName, G: Group, KvHeads: AxisName> Period<Hidden, G, KvHeads> {
    pub(super) const fn new(
        own: Input<Hidden, G>,
        scales: RowGroups<G, KvHeads>,
        groups: usize,
    ) -> Self {
        Self {
            own,
            scales,
            groups,
        }
    }
}

/// A period's queries, their heads spread over the slices, known up to the factors their head
/// norm cancels.
impl<Hidden: AxisName, Heads: AxisName, Head: AxisName, G: Group, KvHeads: AxisName>
    Op<&Period<Hidden, G, KvHeads>> for Affine<Weight<Hidden, Heads, Head>, ()>
{
    type Output = Scaled<Queries<G, Heads, Head>>;

    fn forward(&self, context: &mut Device, rows: &Period<Hidden, G, KvHeads>) -> Self::Output {
        let lanes = Fill(PhantomData).forward(&mut *context, (&rows.own, rows.groups));
        Affine::new(&self.weight, ()).forward(
            context,
            Counted {
                lanes: &lanes,
                groups: rows.groups,
                output: PhantomData,
            },
        )
    }
}

/// A period's keys, each run of heads copied over the slices as values are, known up to the
/// factors their head norm cancels.
impl<Hidden: AxisName, Heads: AxisName, Head: AxisName, G: Group, KvHeads: AxisName>
    Op<&Period<Hidden, G, KvHeads>> for Affine<(Weight<Hidden, Heads, Head>, ()), ()>
{
    type Output = Scaled<Pairs<G, Heads, Head>>;

    fn forward(&self, context: &mut Device, rows: &Period<Hidden, G, KvHeads>) -> Self::Output {
        let lanes = Fill(PhantomData).forward(&mut *context, (&rows.own, rows.groups));
        Affine::new(&self.weight.0, ()).forward(
            context,
            Counted {
                lanes: &lanes,
                groups: rows.groups,
                output: PhantomData,
            },
        )
    }
}

/// A period's values, by the value projection with its head scales fetched for the period's
/// geometry; the rows' scales are the value rows'.
impl<Hidden: AxisName, Heads: AxisName, Head: AxisName, G: Group> Op<&Period<Hidden, G, Heads>>
    for Affine<(Weight<Hidden, Heads, Head>, ValueScale<Hidden, Heads, Head>), ()>
{
    type Output = Pairs<G, Heads, Head>;

    fn forward(&self, context: &mut Device, rows: &Period<Hidden, G, Heads>) -> Self::Output {
        let lanes = Fill(PhantomData).forward(&mut *context, (&rows.own, rows.groups));
        let value = value::Stage::<G>(PhantomData).forward(&mut *context, self);
        value.forward(
            context,
            value::Counted {
                lanes: &lanes,
                rows: &rows.scales,
                groups: rows.groups,
            },
        )
    }
}
