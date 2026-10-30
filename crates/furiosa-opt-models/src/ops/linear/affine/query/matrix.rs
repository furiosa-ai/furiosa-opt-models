//! The dense forms of the query, key and value projections: each weight a matrix the contraction
//! reads as BF16 tiles where it is stored, BF16 coefficients or E4M3 codes it restores by their
//! rows' scales; each period's live rows read where the launch stored them, and each output moved
//! through HBM into the slices its head norm and rotation read.

use super::group::{Group, TilePair};
use super::launch::{Launch, Next, Normalized};
use super::{Cluster, Output, PairLocal, Pairs, Queries, QueryLocal, QuerySpread, Reduced, Tile};
use crate::Begin;
use crate::Module;
use crate::modules::lora::{self, Term};
use crate::ops::linear::{Affine, Matrix, fp8};
use crate::tensor::form::Weight;
use crate::tensor::live::Live;
use crate::tensor::memory::{Hbm, Memory};
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

/// A projection's channel-scaled weight: `Heads` heads of `Head` rows over `Hidden` inputs, the
/// E4M3 codes and the FP32 scale of each row in memory `X`.
pub type ChannelScaled<Hidden, Heads, Head, X = Hbm> = Weight<
    m![Heads, Head, Hidden],
    Chip,
    Coded<
        Hidden,
        Heads,
        Head,
        <X as Memory>::Tensor<f8e4m3, Chip, m![{ Symbol<Width<Heads, Head>> }, Hidden]>,
        <X as Memory>::Tensor<f32, Chip, m![{ Symbol<Width<Heads, Head>> }, 1 # 8]>,
    >,
>;

/// A channel-scaled weight as the contraction reads it, its heads' rows one output axis, the codes
/// held as `W` and the scales as `T`.
type Coded<
    Hidden,
    Heads,
    Head,
    W = HbmTensor<f8e4m3, Chip, m![{ Symbol<Width<Heads, Head>> }, Hidden]>,
    T = HbmTensor<f32, Chip, m![{ Symbol<Width<Heads, Head>> }, 1 # 8]>,
> = fp8::ChannelScaled<Hidden, Width<Heads, Head>, W, T>;

/// A period's live rows where the launch stored them, as the dense contraction reads them.
#[must_use]
pub struct Period<Hidden: AxisName, G: Group, All: AxisName> {
    rows: Normalized<Hidden, All>,
    live: Live<(), All>,
    group: PhantomData<G>,
}

impl<Hidden: AxisName, P: AxisName, G: Group<Period = P>, All: AxisName> Period<Hidden, G, All> {
    /// The period's live rows projected by `weight` into rows of `Heads` heads in HBM, a row a
    /// period row; the contraction projects only the live rows.
    fn lines<Heads: AxisName, Head: AxisName, Env: Copy, W>(
        &self,
        device: &mut Device,
        env: Env,
        weight: &W,
    ) -> Lines<P, Heads, Head>
    where
        W: Matrix<In = Hidden, Out = Width<Heads, Head>>,
    {
        const { assert!(G::SIZE * Tile::SIZE == P::SIZE) };
        let mut lines = Lines::new();
        Affine::new(weight, ()).forward(
            &mut *device,
            env,
            &mut (),
            (self.live, &self.rows, &mut lines),
        );
        lines
    }

    /// The period's projected `lines` landed as the output of `Heads`, `Local` of a slice's run in
    /// each slice and the rest spread over `Spread`: through HBM, as its rows are not the output's
    /// slices.
    fn land<Heads: AxisName, Head: AxisName, Spread: M, Local: M>(
        device: &mut Device,
        lines: Lines<P, Heads, Head>,
    ) -> Output<G, Heads, Head, Spread, Local> {
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

    /// The period's projected `lines` plus each of `adapters`' update of its live rows.
    fn adapt<'a, Heads: AxisName, Head: AxisName, A, B, K: AxisName, const N: usize>(
        &self,
        device: &mut Device,
        adapters: &lora::Adapters<'a, Term<A, B>, Selection<All, K>, N>,
        lines: Lines<P, Heads, Head>,
    ) -> Lines<P, Heads, Head>
    where
        Affine<A, ()>: for<'i, 'o> crate::Op<
                (
                    Live<(), All>,
                    &'i Normalized<Hidden, All>,
                    &'o mut HbmTensor<bf16, Chip, m![P, K]>,
                ),
                Output = (),
            >,
        Affine<B, ()>: for<'s, 'o> crate::Op<
                (
                    Live<(), P>,
                    &'s HbmTensor<bf16, Chip, m![P, K]>,
                    &'o mut Lines<P, Heads, Head>,
                ),
                Output = (),
            >,
    {
        adapters.add(device, self.live, &self.rows, self.live.start(), lines)
    }
}

/// A period's projected rows in HBM, a row of `Heads` heads' features a period row.
type Lines<P, Heads, Head> = HbmTensor<bf16, Chip, m![P, { Symbol<Width<Heads, Head>> }]>;

/// An adapter's selection of a launch's `All` rows over `K` rank columns.
type Selection<All, K> = HbmTensor<bf16, Chip, m![All, K]>;

/// The dense query projection lands nothing before its rows arrive: the contraction reads each
/// period's weight slabs from HBM, so no live layer holds BF16 weights in DM.
impl<Hidden: AxisName, Heads: AxisName, Head: AxisName, Env> Module<Begin<()>, Env, &mut ()>
    for Affine<Dense<Hidden, Heads, Head>, ()>
{
    type Output = Affine<Flat<Hidden, Heads, Head>, ()>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) -> Self::Output {
        // SAFETY: the landed form that holds the alias keeps the weight it reads, heads in HBM order.
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

/// Nor does the channel-scaled query projection: the contraction restores each period's weight
/// slabs from their codes where they are stored.
impl<Hidden: AxisName, Heads: AxisName, Head: AxisName, Env> Module<Begin<()>, Env, &mut ()>
    for ChannelScaled<Hidden, Heads, Head>
{
    type Output = Affine<Coded<Hidden, Heads, Head>, ()>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) -> Self::Output {
        Affine::new(self.held().clone(), ())
    }
}

/// Nor the channel-scaled key or value projection.
impl<Hidden: AxisName, Heads: AxisName, Head: AxisName, Env> Module<Begin<()>, Env, &mut ()>
    for (ChannelScaled<Hidden, Heads, Head>, ())
{
    type Output = Affine<(Coded<Hidden, Heads, Head>, ()), ()>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) -> Self::Output {
        Affine::new((self.0.held().clone(), ()), ())
    }
}

/// The value projection begins the input its siblings read: the launch's live rows `live`,
/// before any span arrives; dense periods keep nothing between them.
impl<Hidden: AxisName, All: AxisName, W: Matrix<In = Hidden>>
    Module<Begin<()>, Live<(), All>, &mut ()> for Affine<(W, ()), ()>
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
impl<G, P, Hidden, All, W, Env> Module<Next<'_, G, Hidden, (), All>, Env, &mut ()>
    for Affine<(W, ()), ()>
where
    G: Group<Period = P>,
    P: AxisName,
    Hidden: AxisName,
    All: AxisName,
    W: Matrix<In = Hidden>,
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
impl<Hidden, Heads, Head, G, P, All, W, Env> Module<&Period<Hidden, G, All>, Env, &mut ()>
    for Affine<W, ()>
where
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
    G: Group<Period = P>,
    P: AxisName,
    All: AxisName,
    W: Matrix<In = Hidden, Out = Width<Heads, Head>>,
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
        let lines = rows.lines(&mut *device, env, &self.weight);
        Period::<Hidden, G, All>::land::<Heads, Head, QuerySpread<G, Heads>, QueryLocal<G, Heads>>(
            device, lines,
        )
    }
}

/// A period's keys or values, each run of heads copied over the slices.
impl<Hidden, Heads, Head, G, P, All, W, Env> Module<&Period<Hidden, G, All>, Env, &mut ()>
    for Affine<(W, ()), ()>
where
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
    G: Group<Period = P>,
    P: AxisName,
    All: AxisName,
    W: Matrix<In = Hidden, Out = Width<Heads, Head>>,
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
        let lines = rows.lines(&mut *device, env, &self.weight.0);
        Period::<Hidden, G, All>::land::<Heads, Head, G::Copies, PairLocal<Heads>>(device, lines)
    }
}

/// A period's queries plus each adapter's update of its rows, their heads spread over the slices.
impl<'s, 'a, Hidden, Heads, Head, G, P, All, Env, A, B, K, const N: usize>
    Module<
        &Period<Hidden, G, All>,
        Env,
        &'s mut lora::Adapters<'a, Term<A, B>, Selection<All, K>, N>,
    > for Affine<Flat<Hidden, Heads, Head>, ()>
where
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
    G: Group<Period = P>,
    P: AxisName,
    All: AxisName,
    Env: Copy,
    K: AxisName,
    Affine<A, ()>: for<'i, 'o> crate::Op<
            (
                Live<(), All>,
                &'i Normalized<Hidden, All>,
                &'o mut HbmTensor<bf16, Chip, m![P, K]>,
            ),
            Output = (),
        >,
    Affine<B, ()>: for<'r, 'o> crate::Op<
            (
                Live<(), P>,
                &'r HbmTensor<bf16, Chip, m![P, K]>,
                &'o mut Lines<P, Heads, Head>,
            ),
            Output = (),
        >,
{
    type Output = Queries<G, Heads, Head>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        adapters: &'s mut lora::Adapters<'a, Term<A, B>, Selection<All, K>, N>,
        rows: &Period<Hidden, G, All>,
    ) -> Self::Output {
        let lines = rows.lines(&mut *device, env, &self.weight);
        let lines = rows.adapt(&mut *device, adapters, lines);
        Period::<Hidden, G, All>::land::<Heads, Head, QuerySpread<G, Heads>, QueryLocal<G, Heads>>(
            device, lines,
        )
    }
}

/// A period's keys or values plus each adapter's update of its rows, each run of heads copied
/// over the slices.
impl<'s, 'a, Hidden, Heads, Head, G, P, All, Env, A, B, K, const N: usize>
    Module<
        &Period<Hidden, G, All>,
        Env,
        &'s mut lora::Adapters<'a, Term<A, B>, Selection<All, K>, N>,
    > for Affine<(Flat<Hidden, Heads, Head>, ()), ()>
where
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
    G: Group<Period = P>,
    P: AxisName,
    All: AxisName,
    Env: Copy,
    K: AxisName,
    Affine<A, ()>: for<'i, 'o> crate::Op<
            (
                Live<(), All>,
                &'i Normalized<Hidden, All>,
                &'o mut HbmTensor<bf16, Chip, m![P, K]>,
            ),
            Output = (),
        >,
    Affine<B, ()>: for<'r, 'o> crate::Op<
            (
                Live<(), P>,
                &'r HbmTensor<bf16, Chip, m![P, K]>,
                &'o mut Lines<P, Heads, Head>,
            ),
            Output = (),
        >,
{
    type Output = Pairs<G, Heads, Head>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        adapters: &'s mut lora::Adapters<'a, Term<A, B>, Selection<All, K>, N>,
        rows: &Period<Hidden, G, All>,
    ) -> Self::Output {
        let lines = rows.lines(&mut *device, env, &self.weight.0);
        let lines = rows.adapt(&mut *device, adapters, lines);
        Period::<Hidden, G, All>::land::<Heads, Head, G::Copies, PairLocal<Heads>>(device, lines)
    }
}
