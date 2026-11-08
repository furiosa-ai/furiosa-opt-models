//! Interleaved layers run by the decoder schedule: the order's periods rolled, each period
//! unrolled for its layer types, every edge between two attentions the one [`Tokenwise`] work.
//!
//! - The first layer's start runs over every tile of the rows the home holds; then each layer
//!   attends and, tile by tile, its end and the next layer's start run, or the final norm after the
//!   last layer, each tile's rows back in the home.
//! - A period's last edge leads to the next period's first layer; the last period, peeled, ends in
//!   the final norm instead.
//! - A run of a period's edges that each lead to the next layer of the same collection rolls too:
//!   one device loop over the run's layers, as those edges differ only in their layers' indices.
//! - Each group's layers share its places: a layer stores into them once the group's previous
//!   attention read them.
//! - A causal model over such collections launches once a call: each group's context prepared from
//!   its history, the embedded tokens in the home, the schedule, then the head's tokens of the
//!   rows the launch returns.

use super::{Cut, Decoder, Places, Pool, Tokenwise};
use crate::load::Named;
use crate::models::decoder::attention::{ForCausalLM, Geometry, Model};
use crate::models::decoder::{self, launch::interleaved};
use crate::modules::attention;
use crate::modules::language::{Backbone, Cap, Greedy, Normalized};
use crate::ops::attention::gqa::{Counted, Pages};
use crate::ops::embedding::Ids;
use crate::ops::index::gather::Gather;
use crate::ops::index::scatter::Store;
use crate::ops::linear::{self, Affine};
use crate::ops::normalization::{RowMajor, Spread};
use crate::ops::reduction::Argmax;
use crate::ops::rotary::Rotation;
use crate::tensor::live::{Live, Spill};
use crate::tensor::parallel::Chip;
use crate::{Branch, Interleave, Module, Order, Then};
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;

/// A launch's rows through two collections of layers in their interleaved order and then the
/// final norm, in place in `home`: the layers of `X` attend in group `A` over `CX`, run with `T`
/// and share `Places<A, RA>`, those of `Y` likewise; each layer holds its cache in its slot.
impl<
    'm,
    'i,
    's,
    'b,
    X,
    Y,
    F,
    A,
    B,
    CX,
    CY,
    T,
    U,
    W,
    E,
    Tl,
    RA,
    RB,
    H,
    LX,
    LY,
    K,
    const FIRST: usize,
    const SECOND: usize,
    const ORDER: &'static [Branch],
>
    Module<
        (&'i mut H, Live<(), E>),
        (CX, CY),
        &'s mut (
            (Places<A, RA>, [(&'b mut Pool<A>, T); FIRST]),
            (Places<B, RB>, [(&'b mut Pool<B>, U); SECOND]),
        ),
    >
    for Decoder<(
        &'m Interleave<([X; FIRST], [Y; SECOND]), ORDER>,
        &'m Named<"norm", F>,
    )>
where
    A: decoder::Attention,
    B: decoder::Attention,
    E: AxisName,
    Tl: AxisName,
    RA: AxisName,
    RB: AxisName,
    CX: Copy,
    CY: Copy,
    K: Default,
    T: Copy,
    U: Copy,
    X: Cut<
            A,
            CX,
            T,
            Rows = W,
            Envelope = E,
            Tile = Tl,
            Reach = RA,
            Home = H,
            Landed = LX,
            Ahead = K,
        >,
    Y: Cut<
            B,
            CY,
            U,
            Rows = W,
            Envelope = E,
            Tile = Tl,
            Reach = RB,
            Home = H,
            Landed = LY,
            Ahead = K,
        >,
    Named<"norm", F>: for<'r, 'x> Module<&'r Live<W, E>, (), &'x mut (), Output = Live<W, E>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (cx, cy): (CX, CY),
        ((pa, sa), (pb, sb)): &'s mut (
            (Places<A, RA>, [(&'b mut Pool<A>, T); FIRST]),
            (Places<B, RB>, [(&'b mut Pool<B>, U); SECOND]),
        ),
        (home, live): (&'i mut H, Live<(), E>),
    ) {
        let (interleave, norm) = self.0;
        let (first, second) = &interleave.layers;
        let (positions, period, groups, counts, runs) = const {
            let order = Order::<FIRST, SECOND>::new(ORDER);
            (
                order.positions.0,
                order.period,
                ORDER.len() / order.period,
                order.counts,
                runs::<FIRST, SECOND>(ORDER, order.period),
            )
        };
        // Whether a period starts with a layer of the first collection.
        let leads = counts[0] > 0 && positions[0] == 0;
        let mut ahead = K::default();
        // The first layer's start over every tile.
        macro_rules! starts {
            ($layer:expr, $call:expr, $places:expr, $slot:expr) => {{
                let mut slot = $slot;
                for at in 0..live.count::<Tl>() {
                    let tile = live.tile::<Tl>(at);
                    let state = (&mut $places.stored, ahead, &mut slot);
                    let work = Tokenwise(&*home, $layer);
                    let (rows, landed) = work.forward(&mut *device, $call, state, tile);
                    ahead = landed;
                    $layer.spill(&mut *device, home, rows.with(rows.value()));
                }
            }};
        }
        // Tile by tile, a layer's end from its attention and the next part's start; `$ahead` takes
        // what the next part landed ahead, carried by value so its value before is dead once read.
        macro_rules! edge {
            (
                $layer:expr, $places:expr, $landed:expr, $work:expr, $call:expr, $state:expr
                $(, $ahead:ident)?
            ) => {{
                for at in 0..live.count::<Tl>() {
                    let tile = live.tile::<Tl>(at);
                    let input = tile.with($layer.fill(&mut *device, home, tile));
                    let output = (tile.with(&$places.attended), &mut $landed);
                    let rows = $work.forward(&mut *device, $call, $state, (output, input));
                    $(let (rows, landed) = rows; $ahead = landed;)?
                    $layer.spill(&mut *device, home, rows.with(rows.value()));
                }
            }};
        }
        // `$body` for each edge of a run of `$run`, `$step` its place in the run: a device loop
        // over a run of two or more.
        macro_rules! roll {
            ($run:expr, $step:ident, $body:block) => {
                match $run > 1 {
                    true => {
                        for $step in 0..$run $body
                    }
                    false => {
                        let $step = 0;
                        $body
                    }
                }
            };
        }
        match leads {
            true => starts!(&first[0], cx, pa, sa[0].1),
            false => starts!(&second[0], cy, pb, sb[0].1),
        }
        let mut period_of = |group: usize, last: bool, mut ahead: K| {
            // The end of the last run, whose device loop ran the edges before it.
            let (mut a, mut b, mut end) = (0, 0, 0);
            #[unroll]
            for offset in 0..period {
                let current = a < counts[0] && positions[a] == offset;
                let (na, nb) = match current {
                    true => (a + 1, b),
                    false => (a, b + 1),
                };
                let wraps = offset + 1 == period;
                let next = match wraps {
                    true => leads,
                    false => na < counts[0] && positions[na] == offset + 1,
                };
                let (ja, jb) = match wraps {
                    true => ((group + 1) * counts[0], (group + 1) * counts[1]),
                    false => (group * counts[0] + na, group * counts[1] + nb),
                };
                let run = match current {
                    true => runs.0[a],
                    false => runs.1[b],
                };
                // The last period's last edge leads to the final norm, outside any run.
                let run = match last && offset + run == period {
                    true => run - 1,
                    false => run,
                };
                if offset >= end {
                    match current {
                        true => roll!(run, step, {
                            let index = group * counts[0] + a + step;
                            let layer = &first[index];
                            let attend = (&mut sa[index].0, live);
                            let mut landed = layer.attend(&mut *device, cx, attend, pa);
                            let mut slot = sa[index].1;
                            if wraps && last {
                                let work = Tokenwise(layer, norm);
                                edge!(layer, pa, landed, work, cx, (&ahead, &mut slot));
                            } else if next {
                                let mut following = sa[index + 1].1;
                                let work = Tokenwise(layer, &first[index + 1]);
                                edge!(
                                    layer,
                                    pa,
                                    landed,
                                    work,
                                    (cx, cx),
                                    (&mut pa.stored, ahead, (&mut slot, &mut following)),
                                    ahead
                                );
                            } else {
                                let mut following = sb[jb].1;
                                let work = Tokenwise(layer, &second[jb]);
                                edge!(
                                    layer,
                                    pa,
                                    landed,
                                    work,
                                    (cx, cy),
                                    (&mut pb.stored, ahead, (&mut slot, &mut following)),
                                    ahead
                                );
                            }
                        }),
                        false => roll!(run, step, {
                            let index = group * counts[1] + b + step;
                            let layer = &second[index];
                            let attend = (&mut sb[index].0, live);
                            let mut landed = layer.attend(&mut *device, cy, attend, pb);
                            let mut slot = sb[index].1;
                            if wraps && last {
                                let work = Tokenwise(layer, norm);
                                edge!(layer, pb, landed, work, cy, (&ahead, &mut slot));
                            } else if next {
                                let mut following = sa[ja].1;
                                let work = Tokenwise(layer, &first[ja]);
                                edge!(
                                    layer,
                                    pb,
                                    landed,
                                    work,
                                    (cy, cx),
                                    (&mut pa.stored, ahead, (&mut slot, &mut following)),
                                    ahead
                                );
                            } else {
                                let mut following = sb[index + 1].1;
                                let work = Tokenwise(layer, &second[index + 1]);
                                edge!(
                                    layer,
                                    pb,
                                    landed,
                                    work,
                                    (cy, cy),
                                    (&mut pb.stored, ahead, (&mut slot, &mut following)),
                                    ahead
                                );
                            }
                        }),
                    }
                    end = offset + run;
                }
                (a, b) = (na, nb);
            }
            ahead
        };
        for group in 0..groups - 1 {
            ahead = period_of(group, false, ahead);
        }
        period_of(groups - 1, true, ahead);
    }
}

/// Each layer's run in a period of `order`, the first collection's layers and then the second's,
/// each in storage order: how many edges from the layer on each lead to the next layer of the same
/// collection, the period's last edge to the next period's first layer.
const fn runs<const FIRST: usize, const SECOND: usize>(
    order: &[Branch],
    period: usize,
) -> ([usize; FIRST], [usize; SECOND]) {
    let mut runs = ([0; FIRST], [0; SECOND]);
    let (mut offset, mut a, mut b) = (0, 0, 0);
    while offset < period {
        let mut run = 0;
        while offset + run < period
            && order[offset + run] as usize == order[(offset + run + 1) % period] as usize
        {
            run += 1;
        }
        match order[offset] {
            Branch::First => {
                runs.0[a] = run;
                a += 1;
            }
            _ => {
                runs.1[b] = run;
                b += 1;
            }
        }
        offset += 1;
    }
    runs
}

/// A call's context of `R` rows in group `A`: their rotation, write slots and pages.
type Context<A, R> = attention::Context<
    Rotation<<A as decoder::Attention>::Rotated, Chip, R>,
    HbmTensor<i32, Chip, m![R]>,
    Pages<R, <A as decoder::Attention>::Windows, Chip, Counted<R>>,
>;

/// A launch's `R` rows of `Hidden` features in HBM, where a tile of them waits between two
/// attentions.
type Home<Hidden, R> = HbmTensor<bf16, Chip, RowMajor<Hidden, R>>;

/// A launch of a causal model whose layers interleave in two collections, those of `X` attending
/// in group `A` and those of `Y` in group `B`, each over pages of its group's cache: both groups'
/// contexts prepared from their histories, the tokens embedded into the home, the layers and the
/// final norm by the schedule over each group's places, then the greedy token of each row the
/// launch returns, in its order.
impl<
    'i,
    's,
    'b,
    G,
    Hidden,
    E,
    X,
    Y,
    F,
    A,
    B,
    R,
    P,
    V,
    W,
    Bias,
    I,
    C,
    Env,
    const FIRST: usize,
    const SECOND: usize,
    const ORDER: &'static [Branch],
>
    Module<
        &'i mut interleaved::Input<R, A, B>,
        Env,
        interleaved::State<'s, 'b, R, A, B, FIRST, SECOND>,
    >
    for ForCausalLM<
        G,
        Model<
            G,
            Normalized<Backbone<E, Interleave<([X; FIRST], [Y; SECOND]), ORDER>>, F>,
            (decoder::Paging<A, R>, decoder::Paging<B, R>),
            P,
        >,
        Greedy<Hidden, V, Affine<W, Bias>, Argmax<V, Chip, I>, C>,
        P,
    >
where
    G: Geometry<Hidden = Hidden>,
    Hidden: AxisName,
    A: decoder::Attention<Rotated: AxisName>,
    B: decoder::Attention<Rotated: AxisName>,
    R: AxisName,
    V: AxisName,
    W: linear::Table<Chip, Hidden = Hidden, Vocabulary = V>,
    Bias: Borrow<HbmTensor<f32, Chip, m![V]>>,
    I: Borrow<HbmTensor<i32, Chip, m![V]>>,
    C: Cap<Hidden, V>,
    decoder::Paging<A, R>: for<'x> Module<
            (Live<(), R>, &'x mut interleaved::History<R, A>),
            (),
            &'x mut (),
            Output = Context<A, R>,
        >,
    decoder::Paging<B, R>: for<'x> Module<
            (Live<(), R>, &'x mut interleaved::History<R, B>),
            (),
            &'x mut (),
            Output = Context<B, R>,
        >,
    E: for<'x, 'y> Module<
            (&'x Ids<R>, Live<(), R>),
            (),
            &'y mut ((), ()),
            Output = Live<Spread<Hidden, R, Chip>, R>,
        >,
    for<'m, 'c, 'h, 'x> Decoder<(
        &'m Interleave<([X; FIRST], [Y; SECOND]), ORDER>,
        &'m Named<"norm", F>,
    )>: Module<
            (&'h mut Home<Hidden, R>, Live<(), R>),
            (&'c Context<A, R>, &'c Context<B, R>),
            &'x mut (
                (Places<A, R>, [interleaved::Layer<'b, A>; FIRST]),
                (Places<B, R>, [interleaved::Layer<'b, B>; SECOND]),
            ),
            Output = (),
        >,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        ((first, second), tokens): interleaved::State<'s, 'b, R, A, B, FIRST, SECOND>,
        input: &'i mut interleaved::Input<R, A, B>,
    ) {
        let live = Live::<(), R>::read(&input.count);
        let (a, b, returned) = &mut input.cache;
        let rotary = &self.model.rotary_emb;
        let contexts = (
            rotary.0.forward(&mut *device, (), &mut (), (live, a)),
            rotary.1.forward(&mut *device, (), &mut (), (live, b)),
        );
        let Then(Backbone { embedding, body }, norm) = &self.model.body;
        let embedded = embedding.forward(&mut *device, (), &mut ((), ()), (&input.tokens, live));
        let mut home: Home<Hidden, R> = crate::Op::forward(&Store, &mut *device, &embedded);
        let mut places = (
            (Places::<A, R>::new(), first),
            (Places::<B, R>::new(), second),
        );
        let call = (&contexts.0, &contexts.1);
        Decoder((body, norm)).forward(&mut *device, call, &mut places, (&mut home, live));
        let rows: HbmTensor<bf16, Chip, m![R, Hidden]> =
            crate::Op::forward(&Gather, &mut *device, (&home, &*returned, live));
        let rows: Spread<Hidden, R, Chip> = rows.to_dm(&mut device.tdma);
        let head = self.lm_head.to_dm(&mut *device);
        let chosen = head.forward(&mut *device, (), &mut (), &rows);
        tokens.spill(device, &chosen, (live, 0));
    }
}
