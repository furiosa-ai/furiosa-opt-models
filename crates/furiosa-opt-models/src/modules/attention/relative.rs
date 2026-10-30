//! Learned attention bias between explicit query and key positions.

use crate::Module;
use furiosa_opt_std::prelude::*;
use std::{borrow::Borrow, marker::PhantomData};

#[derive(Clone, Debug)]
pub struct Bucket<A>(PhantomData<A>);

impl<A: AxisName> AxisName for Bucket<A> {
    const NAME: Ident = Ident::new("Bucket");
    const SIZE: usize = A::SIZE;
}

pub type Table<Buckets, Heads> = HbmTensor<f32, m![1], m![{ Symbol<Bucket<Buckets>> }, Heads]>;

/// Explicit logical positions of query and key rows.
#[derive(Clone, DeviceSend)]
pub struct Positions<Query = Vec<usize>, Key = Query> {
    /// Query token positions.
    pub query: Query,
    /// Key token positions.
    pub key: Key,
}

/// Look up learned logarithmic relative-position buckets and add the visibility mask.
/// `BIDIRECTIONAL` splits buckets by direction; false uses causal distances only.
#[derive(DeviceSend)]
pub struct Bias<Weight, const BIDIRECTIONAL: bool = true> {
    /// Learned additive bias per bucket and head.
    pub weight: Weight,
    /// Distance in tokens at which the logarithmic bucket range saturates.
    pub distance: HbmTensor<i32, m![1], m![1]>,
}

type Ids<Count> = HbmTensor<i32, m![1], m![Count]>;

/// Positions are owned or borrowed, so one tensor can be both the query and the key positions.
impl<
    Buckets: AxisName,
    Heads: AxisName,
    Queries: AxisName,
    Keys: AxisName,
    Query: Borrow<Ids<Queries>>,
    Key: Borrow<Ids<Keys>>,
    const BIDIRECTIONAL: bool,
    Env,
>
    Module<
        (
            &Positions<Query, Key>,
            &HbmTensor<f32, m![1], m![Queries, Keys]>,
        ),
        Env,
        &mut (),
    > for Bias<Table<Buckets, Heads>, BIDIRECTIONAL>
{
    type Output = HbmTensor<f32, m![1], m![Heads, Queries, Keys]>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: (
            &Positions<Query, Key>,
            &HbmTensor<f32, m![1], m![Queries, Keys]>,
        ),
    ) -> Self::Output {
        const { assert!(Buckets::SIZE >= 4 && Buckets::SIZE % 2 == 0) };
        todo!(
            "NPU logarithmic relative-position buckets, learned lookup, and broadcast mask addition"
        )
    }
}
