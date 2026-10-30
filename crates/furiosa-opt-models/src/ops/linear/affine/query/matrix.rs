//! The BF16 forms of the query, key and value projections: each weight a dense matrix the
//! contraction reads where it is stored, each period's live rows read where the launch stored
//! them, and each output moved through HBM into the slices its head norm and rotation read.

use super::group::{Group, TilePair};
use super::launch::{Launch, Next, Normalized};
use super::{Cluster, Output, PairLocal, Pairs, Queries, QueryLocal, QuerySpread, Reduced, Tile};
use crate::Module;
use crate::modules::generic::schedule::Begin;
use crate::ops::linear::Affine;
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

/// The output features of `Heads` heads of `Head` features, as the contraction's one output axis.
#[derive(Clone, Debug)]
pub struct Width<Heads, Head>(PhantomData<(Heads, Head)>);

impl<Heads: AxisName, Head: AxisName> AxisName for Width<Heads, Head> {
    const NAME: Ident = Ident::new("HeadWidth");
    const SIZE: usize = Heads::SIZE * Head::SIZE;
}

/// A projection's dense weight: `Heads` heads of `Head` rows over `Hidden` inputs.
pub type Dense<Hidden, Heads, Head> = HbmTensor<bf16, Chip, m![Heads, Head, Hidden]>;

/// A dense weight as the contraction reads it, its heads' rows one output axis.
type Flat<Hidden, Heads, Head> = HbmTensor<bf16, Chip, m![{ Symbol<Width<Heads, Head>> }, Hidden]>;

/// A period's live rows where the launch stored them, as the dense contraction reads them.
#[must_use]
pub struct Period<Hidden: AxisName, G: Group, All: AxisName> {
    rows: Normalized<Hidden, All>,
    live: Live<(), All>,
    group: PhantomData<G>,
}

impl<Hidden: AxisName, P: AxisName, G: Group<Period = P>, All: AxisName> Period<Hidden, G, All> {
    /// The period's rows projected by `weight` into the output of `Heads`, `Local` of a slice's
    /// run in each slice and the rest spread over `Spread`. The contraction projects the period's
    /// live rows, and its output lands through HBM, as its rows are not the output's slices.
    fn project<Heads: AxisName, Head: AxisName, Spread: M, Local: M, Env: Copy>(
        &self,
        device: &mut Device,
        env: Env,
        weight: &Flat<Hidden, Heads, Head>,
    ) -> Output<G, Heads, Head, Spread, Local> {
        const { assert!(G::SIZE * Tile::SIZE == P::SIZE) };
        let mut lines = HbmTensor::<bf16, Chip, m![P, { Symbol<Width<Heads, Head>> }]>::new();
        Affine::new(weight, ()).forward(
            &mut *device,
            env,
            &mut (),
            (self.live, &self.rows, &mut lines),
        );
        // SAFETY: the period's rows are its group's tiles in order, and a row's features its heads'
        // features in order.
        let lines: HbmTensor<bf16, Chip, m![G, Tile, Heads, Head]> =
            unsafe { lines.view().alias() };
        let output: DmTensor<
            bf16,
            Chip,
            Cluster<Heads>,
            Reduced<G, Heads, Spread>,
            m![{ Local }, Tile % 8, Head],
        > = lines.to_dm(&mut device.tdma);
        // SAFETY: the output's slice names for Reduced's slices, as the other forms give them.
        unsafe { output.reshape() }
    }
}

/// The dense query projection lands nothing before its rows arrive: the contraction reads each
/// period's weight slabs from HBM, so no live layer holds BF16 weights in DM.
impl<Hidden: AxisName, Heads: AxisName, Head: AxisName, Env> Module<Begin<()>, Env, &mut ()>
    for Affine<Dense<Hidden, Heads, Head>, ()>
{
    type Output = Affine<Flat<Hidden, Heads, Head>, ()>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) -> Self::Output {
        // SAFETY: the node that holds the alias keeps the weight it reads, heads in HBM order.
        Affine::new(unsafe { self.weight.view().alias() }, ())
    }
}

/// The dense key or value projection lands nothing before its rows arrive.
impl<Hidden: AxisName, Heads: AxisName, Head: AxisName, Env> Module<Begin<()>, Env, &mut ()>
    for Affine<(Dense<Hidden, Heads, Head>, ()), ()>
{
    type Output = Affine<(Flat<Hidden, Heads, Head>, ()), ()>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) -> Self::Output {
        // SAFETY: as for the queries.
        Affine::new((unsafe { self.weight.0.view().alias() }, ()), ())
    }
}

/// The value projection begins the input its siblings read: the launch's live rows `live`,
/// before any span arrives; dense periods keep nothing between them.
impl<Hidden: AxisName, Heads: AxisName, Head: AxisName, All: AxisName>
    Module<Begin<()>, Live<(), All>, &mut ()> for Affine<(Flat<Hidden, Heads, Head>, ()), ()>
{
    type Output = Launch<Hidden, (), All>;

    fn forward(
        &self,
        _: &mut Device,
        live: Live<(), All>,
        _: &mut (),
        _: Begin<()>,
    ) -> Self::Output {
        Launch::new(live, ())
    }
}

/// A period's live rows, read where the launch stored them.
impl<G, P, Hidden, Heads, Head, All, Env> Module<Next<'_, G, Hidden, (), All>, Env, &mut ()>
    for Affine<(Flat<Hidden, Heads, Head>, ()), ()>
where
    G: Group<Period = P>,
    P: AxisName,
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
    All: AxisName,
{
    type Output = Period<Hidden, G, All>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        next: Next<'_, G, Hidden, (), All>,
    ) -> Self::Output {
        let Next { rows, live, at, .. } = next;
        Period {
            // SAFETY: the launch writes its rows only before its first period, and keeps them.
            rows: unsafe { rows.view().alias() },
            live: live.subrange(at * <TilePair as Group>::Period::SIZE, P::SIZE),
            group: PhantomData,
        }
    }
}

/// A period's queries, their heads spread over the slices.
impl<
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
    G: Group<Period = P>,
    P: AxisName,
    All: AxisName,
    Env,
> Module<&Period<Hidden, G, All>, Env, &mut ()> for Affine<Flat<Hidden, Heads, Head>, ()>
where
    Env: Copy,
{
    type Output = Queries<G, Heads, Head>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        rows: &Period<Hidden, G, All>,
    ) -> Self::Output {
        rows.project::<Heads, Head, QuerySpread<G, Heads>, QueryLocal<G, Heads>, _>(
            device,
            env,
            &self.weight,
        )
    }
}

/// A period's keys or values, each run of heads copied over the slices.
impl<
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
    G: Group<Period = P>,
    P: AxisName,
    All: AxisName,
    Env,
> Module<&Period<Hidden, G, All>, Env, &mut ()> for Affine<(Flat<Hidden, Heads, Head>, ()), ()>
where
    Env: Copy,
{
    type Output = Pairs<G, Heads, Head>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        rows: &Period<Hidden, G, All>,
    ) -> Self::Output {
        rows.project::<Heads, Head, G::Copies, PairLocal<Heads>, _>(device, env, &self.weight.0)
    }
}
