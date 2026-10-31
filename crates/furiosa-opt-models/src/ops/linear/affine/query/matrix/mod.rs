//! Dense projection periods read their live rows and restore stored weight tiles.

use super::group::{Group, TilePair};
use super::launch::{Launch, Next, Normalized};
use super::{Cluster, Output, PairLocal, Pairs, Queries, QueryLocal, QuerySpread, Reduced, Tile};
use crate::modules::lora::{self, Term};
use crate::ops::linear::{Affine, Matrix, fp8};
use crate::tensor::form::Weight;
use crate::tensor::live::Live;
use crate::tensor::memory::{Hbm, Memory};
use crate::tensor::parallel::Chip;
use crate::{Begin, Module};
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

mod adapters;
mod phase;

#[derive(Clone, Debug)]
pub struct Width<Heads, Head>(PhantomData<(Heads, Head)>);

impl<Heads: AxisName, Head: AxisName> AxisName for Width<Heads, Head> {
    const NAME: Ident = Ident::new("HeadWidth");
    const SIZE: usize = Heads::SIZE * Head::SIZE;
}

pub type Dense<Hidden, Heads, Head> = HbmTensor<bf16, Chip, m![Heads, Head, Hidden]>;

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

type Coded<
    Hidden,
    Heads,
    Head,
    W = HbmTensor<f8e4m3, Chip, m![{ Symbol<Width<Heads, Head>> }, Hidden]>,
    T = HbmTensor<f32, Chip, m![{ Symbol<Width<Heads, Head>> }, 1 # 8]>,
> = fp8::ChannelScaled<Hidden, Width<Heads, Head>, W, T>;

#[must_use]
pub struct Period<Hidden: AxisName, G: Group, All: AxisName> {
    rows: Normalized<Hidden, All>,
    live: Live<(), All>,
    group: PhantomData<G>,
}

impl<Hidden: AxisName, P: AxisName, G: Group<Period = P>, All: AxisName> Period<Hidden, G, All> {
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

    fn land<Heads: AxisName, Head: AxisName, Spread: M, Local: M>(
        device: &mut Device,
        lines: Lines<P, Heads, Head>,
    ) -> Output<G, Heads, Head, Spread, Local> {
        // SAFETY: period rows are group tiles in order; each row keeps its head features in order.
        let lines = unsafe { lines.view().reshape::<Chip, m![G, Tile, Heads, Head]>() };
        let output: DmTensor<
            bf16,
            Chip,
            Cluster<Heads>,
            Reduced<G, Heads, Spread>,
            m![{ Local }, Tile % 8, Head],
        > = lines.to_dm(&mut device.tdma);
        // SAFETY: the output's slice names name Reduced's slices in their existing order.
        unsafe { output.reshape() }
    }

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

type Lines<P, Heads, Head> = HbmTensor<bf16, Chip, m![P, { Symbol<Width<Heads, Head>> }]>;
type Selection<All, K> = HbmTensor<bf16, Chip, m![All, K]>;
