//! Interleaved layers run by the decoder schedule: the order's periods rolled, each period
//! unrolled for its layer types, every edge between two attentions the one [`Tokenwise`] work.
//!
//! - The first layer's start runs over every tile of the rows the home holds; then each layer
//!   attends and, tile by tile, its end and the next layer's start run, or the final norm after the
//!   last layer, each tile's rows back in the home.
//! - A period's last edge leads to the next period's first layer; the last period, peeled, ends in
//!   the final norm instead.
//! - Each group's layers share its places: a layer stores into them once the group's previous
//!   attention read them.

use super::{Cut, Decoder, Places, Pool, Tokenwise};
use crate::load::Named;
use crate::models::decoder;
use crate::tensor::live::Live;
use crate::{Branch, Interleave, Module, Order};
use furiosa_opt_std::prelude::*;

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
    T: Copy,
    U: Copy,
    X: Cut<A, CX, T, Rows = W, Envelope = E, Tile = Tl, Reach = RA, Home = H, Landed = LX>,
    Y: Cut<B, CY, U, Rows = W, Envelope = E, Tile = Tl, Reach = RB, Home = H, Landed = LY>,
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
        let (positions, period, groups, counts) = const {
            let order = Order::<FIRST, SECOND>::new(ORDER);
            (
                order.positions.0,
                order.period,
                ORDER.len() / order.period,
                order.counts,
            )
        };
        // Whether a period starts with a layer of the first collection.
        let leads = counts[0] > 0 && positions[0] == 0;
        // The first layer's start over every tile.
        macro_rules! starts {
            ($layer:expr, $call:expr, $places:expr, $slot:expr) => {{
                let mut slot = $slot;
                for at in 0..live.count::<Tl>() {
                    let tile = live.tile::<Tl>(at);
                    let state = (&mut $places.stored, &mut slot);
                    let rows = Tokenwise(&*home, $layer).forward(&mut *device, $call, state, tile);
                    $layer.spill(&mut *device, home, rows.with(rows.value()));
                }
            }};
        }
        // Tile by tile, a layer's end from its attention and the next part's start.
        macro_rules! edge {
            ($layer:expr, $places:expr, $landed:expr, $work:expr, $call:expr, $state:expr) => {{
                for at in 0..live.count::<Tl>() {
                    let tile = live.tile::<Tl>(at);
                    let input = tile.with($layer.fill(&mut *device, home, tile));
                    let output = (tile.with(&$places.attended), &mut $landed);
                    let rows = $work.forward(&mut *device, $call, $state, (output, input));
                    $layer.spill(&mut *device, home, rows.with(rows.value()));
                }
            }};
        }
        match leads {
            true => starts!(&first[0], cx, pa, sa[0].1),
            false => starts!(&second[0], cy, pb, sb[0].1),
        }
        let mut period_of = |group: usize, last: bool| {
            let (mut a, mut b) = (0, 0);
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
                match current {
                    true => {
                        let index = group * counts[0] + a;
                        let layer = &first[index];
                        let attend = (&mut sa[index].0, live);
                        let mut landed = layer.attend(&mut *device, cx, attend, pa);
                        let mut slot = sa[index].1;
                        if wraps && last {
                            edge!(layer, pa, landed, Tokenwise(layer, norm), cx, &mut slot);
                        } else if next {
                            let mut following = sa[ja].1;
                            let work = Tokenwise(layer, &first[ja]);
                            edge!(
                                layer,
                                pa,
                                landed,
                                work,
                                (cx, cx),
                                (&mut pa.stored, (&mut slot, &mut following))
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
                                (&mut pb.stored, (&mut slot, &mut following))
                            );
                        }
                    }
                    false => {
                        let index = group * counts[1] + b;
                        let layer = &second[index];
                        let attend = (&mut sb[index].0, live);
                        let mut landed = layer.attend(&mut *device, cy, attend, pb);
                        let mut slot = sb[index].1;
                        if wraps && last {
                            edge!(layer, pb, landed, Tokenwise(layer, norm), cy, &mut slot);
                        } else if next {
                            let mut following = sa[ja].1;
                            let work = Tokenwise(layer, &first[ja]);
                            edge!(
                                layer,
                                pb,
                                landed,
                                work,
                                (cy, cx),
                                (&mut pa.stored, (&mut slot, &mut following))
                            );
                        } else {
                            let mut following = sb[jb].1;
                            let work = Tokenwise(layer, &second[jb]);
                            edge!(
                                layer,
                                pb,
                                landed,
                                work,
                                (cy, cy),
                                (&mut pb.stored, (&mut slot, &mut following))
                            );
                        }
                    }
                }
                (a, b) = (na, nb);
            }
        };
        for group in 0..groups - 1 {
            period_of(group, false);
        }
        period_of(groups - 1, true);
    }
}
