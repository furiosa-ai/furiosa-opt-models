//! A greedy head over a body's normalized rows: it reads the tokens of the rows a launch selects.

use super::{InDm, Tree};
use crate::Module;
use crate::modules::generic::schedule::{Fill, Spill};
use crate::modules::generic::schedule::{IntoStream, Stream};
use crate::ops::index::scatter::Store;
use crate::ops::normalization::span::{Selection, Span, Tile};
use crate::ops::normalization::{Blocked, Spread};
use crate::ops::reduction;
use crate::tensor::live::{Live, OnTile};
use crate::tensor::scalar::Value;
use crate::tensor::{axis::Rows, parallel::Chip};
use core::task::Poll;
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;

/// The rows a launch returns tokens of: their count and each one's blocked hidden-row offset.
#[derive(DeviceSend)]
pub struct Selected<Capacity: AxisName> {
    pub live: Value,
    pub rows: HbmTensor<i32, Chip, m![Capacity]>,
}

/// The selected rows' tokens into the head's slot. The node takes every tile of its upstream into
/// a place of its own; once the upstream ends, it reads out the selected rows' tokens, a span a trip.
impl<'a, 's, 'i, U, E, X, Width: AxisName, V: AxisName, W, B, I, Capacity: AxisName>
    IntoStream<U, E, &'s mut HbmTensor<i32, Chip, m![Capacity]>> for &'a Tree<Width, V, W, B, I>
where
    U: Stream<Item = Live<X, Capacity>>,
    E: Copy + Into<&'i Selected<Capacity>>,
    W: Borrow<HbmTensor<bf16, Chip, m![V, Width]>>,
    B: Borrow<HbmTensor<f32, Chip, m![V]>>,
    I: Borrow<HbmTensor<i32, Chip, m![V]>>,
    HbmTensor<bf16, Chip, Blocked<Width, Capacity>>:
        for<'x> Spill<Device, &'x X, Live<(), Capacity>>,
{
    type Stream = Node<'a, 's, U, E, Width, V, W, B, I, Capacity>;

    fn into_stream(
        self,
        env: E,
        tokens: &'s mut HbmTensor<i32, Chip, m![Capacity]>,
        upstream: U,
    ) -> Self::Stream {
        Node {
            head: self,
            upstream,
            env,
            hidden: HbmTensor::new(),
            tokens,
            done: false,
        }
    }
}

/// A greedy head's node: the head, its upstream, the env, the place the upstream's rows gather in,
/// its tokens' place, and whether it read them out.
pub(crate) struct Node<'a, 's, U, E, Width: AxisName, V: AxisName, W, B, I, Capacity: AxisName> {
    head: &'a Tree<Width, V, W, B, I>,
    upstream: U,
    env: E,
    hidden: HbmTensor<bf16, Chip, Blocked<Width, Capacity>>,
    tokens: &'s mut HbmTensor<i32, Chip, m![Capacity]>,
    done: bool,
}

/// Once the upstream ends, so after its last effect, a trip a span of the selected rows gathers
/// the span, writes the head's masks in data memory, then reads and stores each of its tiles.
impl<'i, U, E, X, Width: AxisName, V: AxisName, W, B, I, Capacity: AxisName> Stream
    for Node<'_, '_, U, E, Width, V, W, B, I, Capacity>
where
    U: Stream<Item = Live<X, Capacity>>,
    E: Copy + Into<&'i Selected<Capacity>>,
    W: Borrow<HbmTensor<bf16, Chip, m![V, Width]>>,
    B: Borrow<HbmTensor<f32, Chip, m![V]>>,
    I: Borrow<HbmTensor<i32, Chip, m![V]>>,
    HbmTensor<bf16, Chip, Blocked<Width, Capacity>>:
        for<'x> Spill<Device, &'x X, Live<(), Capacity>>,
{
    type Item = ();

    fn poll_next(&mut self, context: &mut Device) -> Poll<Option<()>> {
        if self.done {
            return Poll::Ready(None);
        }
        match self.upstream.poll_next(&mut *context) {
            Poll::Ready(Some(tile)) => {
                let at = tile.with(());
                self.hidden.spill(&mut *context, tile.value(), at);
                return Poll::Pending;
            }
            Poll::Pending => return Poll::Pending,
            Poll::Ready(None) => {}
        }
        self.done = true;
        self.head
            .read(context, &self.hidden, self.env.into(), &mut *self.tokens);
        Poll::Ready(None)
    }
}

impl<Width: AxisName, V: AxisName, W, B, I> Tree<Width, V, W, B, I>
where
    W: Borrow<HbmTensor<bf16, Chip, m![V, Width]>>,
    B: Borrow<HbmTensor<f32, Chip, m![V]>>,
    I: Borrow<HbmTensor<i32, Chip, m![V]>>,
{
    /// The tokens of the rows `selected` names among a launch's normalized rows `hidden`, a span
    /// a trip, into `tokens`; called once the rows' last effect is issued.
    pub(crate) fn read<Capacity: AxisName>(
        &self,
        context: &mut Device,
        hidden: &HbmTensor<bf16, Chip, Blocked<Width, Capacity>>,
        selected: &Selected<Capacity>,
        tokens: &mut HbmTensor<i32, Chip, m![Capacity]>,
    ) {
        // Read after the rows: waiting for the read drains the DMA queue, retiring at once every
        // transfer the rows left pending, whose completion ids and memory the head reuses.
        let chosen = Live::<(), Capacity>::read(&selected.live);
        let selection = Selection {
            hidden,
            rows: &selected.rows,
        };
        for index in 0..chosen.count::<Span>() {
            let tile = chosen.tile::<Span>(index);
            let rows = selection.fill(&mut *context, tile);
            let head = self.to_dm(&mut *context);
            tile.tiles::<_, Rows<128>, Rows<64>, Rows<8>>(
                &mut *context,
                &mut Tiles {
                    rows: &rows,
                    head: &head,
                    tokens: &mut *tokens,
                    tile,
                },
            );
        }
    }
}

/// One span's tiles: each read by the head, and its tokens stored where its rows sit among the
/// launch's.
struct Tiles<'a, Width: AxisName, V: AxisName, Capacity: AxisName> {
    rows: &'a Spread<Width, Span>,
    head: &'a InDm<'a, Width, V>,
    tokens: &'a mut HbmTensor<i32, Chip, m![Capacity]>,
    tile: Live<(), Capacity>,
}

impl<R: AxisName, Width: AxisName, V: AxisName, Capacity: AxisName> OnTile<Device, R>
    for Tiles<'_, Width, V, Capacity>
{
    fn run(&mut self, device: &mut Device, index: usize) {
        let mut rows: Spread<Width, R> = DmTensor::new();
        let wave = Tile::<Width, m![Span % R]>::of(self.rows, index, &mut rows);
        wave.input.to_dm_view(&mut device.tdma, wave.output);
        let selected = self.head.forward(&mut *device, (), &mut (), &rows);
        self.tokens.spill(device, &selected, (self.tile, index));
    }
}

/// Tile `index` of a span's tokens, stored where its rows sit among the launch's selected rows.
impl<'a, Capacity: AxisName, R: AxisName>
    Spill<Device, &'a reduction::Selected<R>, (Live<(), Capacity>, usize)>
    for HbmTensor<i32, Chip, m![Capacity]>
{
    fn spill(
        &mut self,
        context: &mut Device,
        tokens: &'a reduction::Selected<R>,
        (tile, index): (Live<(), Capacity>, usize),
    ) {
        Store.forward(
            context,
            (),
            &mut (),
            (tokens, (self, tile.start() / R::SIZE + index)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::Greedy;
    use super::*;
    use crate::load::{Checkpoint, Context, Load};
    use crate::modules::generic::schedule::Scheduled;
    use crate::ops::linear::Affine;
    use crate::ops::normalization::span::{Owned, Tiled};
    use crate::ops::normalization::{RmsNorm, tests::blocked};
    use crate::ops::reduction::Argmax;
    use crate::tensor::form::Weight;
    use crate::{Then, tests::DEVICE};

    axes![Hidden = 4096, Padded = 152064];

    impl Tiled for Hidden {
        type Capacity = Rows<126>;
        type Last = Rows<4>;
    }

    type Capacity = Rows<1024>;
    type Place = HbmTensor<bf16, Chip, Blocked<Hidden, Capacity>>;
    type Tokens = HbmTensor<i32, Chip, m![Capacity]>;
    /// A greedy head over borrowed forms of its projection and its ranks.
    type Head<'w> = Greedy<
        Hidden,
        Padded,
        Affine<&'w HbmTensor<bf16, Chip, m![Padded, Hidden]>, &'w HbmTensor<f32, Chip, m![Padded]>>,
        Argmax<Padded, Chip, &'w HbmTensor<i32, Chip, m![Padded]>>,
    >;
    type Env<'i> = (Live<(), Capacity>, &'i Selected<Capacity>);

    /// The checkpoint's tokens; the compiled vocabulary pads them.
    const LIVE: usize = 151_936 - 256;
    /// Tokens of the rows' cases: the first and second features' own, two tied on the third.
    const FIRST: usize = 5;
    const SECOND: usize = 9;
    const TIED: usize = 77;
    const LATER: usize = 100;

    impl<'i> From<Env<'i>> for &'i Selected<Capacity> {
        fn from((_, selected): Env<'i>) -> Self {
            selected
        }
    }

    /// A body ending in no norm: it gives each span of the launch's live rows, one a poll; at its
    /// end, after its last tile, it stores the first span again at the trace span its ends count.
    #[derive(DeviceSend)]
    struct Spans;

    struct Node<'s, 'h, 'i> {
        env: Env<'i>,
        places: &'s mut (&'h Place, &'h mut Place),
        given: usize,
        ends: usize,
    }

    impl<'m, 's, 'h, 'i, U: Stream> IntoStream<U, Env<'i>, &'s mut (&'h Place, &'h mut Place)>
        for &'m Spans
    {
        type Stream = Node<'s, 'h, 'i>;

        fn into_stream(
            self,
            env: Env<'i>,
            places: &'s mut (&'h Place, &'h mut Place),
            _: U,
        ) -> Self::Stream {
            Node {
                env,
                places,
                given: 0,
                ends: 0,
            }
        }
    }

    impl Stream for Node<'_, '_, '_> {
        type Item = Live<Owned<Hidden, Span>, Capacity>;

        fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Self::Item>> {
            let live = self.env.0;
            if self.given < live.count::<Span>() {
                let tile = live.tile::<Span>(self.given);
                self.given += 1;
                return Poll::Ready(Some(tile.with(self.places.0.fill(device, tile))));
            }
            let rows = self.places.0.fill(device, live.tile::<Span>(0));
            let at = Live::<(), Capacity>::all().tile::<Span>(self.ends);
            self.places.1.spill(device, &rows, at);
            self.ends += 1;
            Poll::Ready(None)
        }
    }

    macro_rules! read_out {
        ($name:ident, $body:ty, $places:pat => $state:expr) => {
            #[device(chip = 1)]
            fn $name(
                device: &mut Device,
                model: &Then<$body, Head<'_>>,
                count: &Value,
                selected: &Selected<Capacity>,
                input: &Place,
                trace: &mut Place,
            ) -> Tokens {
                let env = (Live::<(), Capacity>::read(count), selected);
                let $places = (input, trace);
                let mut state = ($state, Tokens::new());
                Scheduled(model).forward(device, env, &mut state, ());
                state.1
            }
        };
    }

    read_out!(plain, Spans, places => places);
    read_out!(normalized, Then<Spans, RmsNorm<Hidden>>, places => (places, ()));

    /// Row `row`'s feature values: a feature of its own, the first two, the third, or the second.
    fn features(row: usize) -> &'static [usize] {
        const CASES: [&[usize]; 4] = [&[], &[0, 1], &[2], &[1]];
        CASES[row % 4]
    }

    /// Row `row`'s token, its first two features weighted by `gain`.
    fn token(row: usize, gain: (f32, f32)) -> usize {
        match row % 4 {
            0 => 1000 + row,
            1 if gain.1 > gain.0 => SECOND,
            1 => FIRST,
            2 => TIED,
            _ => SECOND,
        }
    }

    /// Each selected row's first maximum among the checkpoint's tokens, in selection order, read
    /// from the normalized rows the body gives, after the body's last effect, which runs once.
    #[tokio::test(flavor = "current_thread")]
    async fn reads_the_body_once_it_ends() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let rows = 520;
        let mut input = vec![bf16::from_f32(0.0); Capacity::SIZE * Hidden::SIZE];
        for row in 0..rows {
            let own = [8 + row];
            let set = if row % 4 == 0 {
                &own[..]
            } else {
                features(row)
            };
            for &feature in set {
                input[row * Hidden::SIZE + feature] = bf16::from_f32(1.0);
            }
        }
        let mut weight = vec![bf16::from_f32(0.0); Padded::SIZE * Hidden::SIZE];
        let mut set = |token: usize, feature: usize, value: f32| {
            weight[token * Hidden::SIZE + feature] = bf16::from_f32(value);
        };
        set(FIRST, 0, 1.0);
        set(SECOND, 1, 1.0);
        set(TIED, 2, 1.0);
        set(LATER, 2, 1.0);
        // The first padded token and the last outscore every checkpoint token.
        for feature in 0..3 {
            set(LIVE, feature, 4.0);
            set(Padded::SIZE - 1, feature, 4.0);
        }
        for row in (0..rows).step_by(4) {
            set(1000 + row, 8 + row, 1.0);
        }
        let place = |values: Vec<bf16>| {
            HostTensor::<bf16, Blocked<Hidden, Capacity>>::from_vec(blocked(
                &values,
                Hidden::SIZE,
                16,
            ))
        };
        let input_rows = place(input.clone())
            .to_hbm::<Chip, Blocked<Hidden, Capacity>>(&mut device.pdma)
            .await
            .unwrap();
        let count =
            |value: usize| HostTensor::<i32, m![1]>::from_vec([i32::try_from(value).unwrap()]);
        let launch_rows = count(rows).to_hbm(&mut device.pdma).await.unwrap();
        let weight = HostTensor::<bf16, m![Padded, Hidden]>::from_vec(weight)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mask = (0..Padded::SIZE).map(|token| if token < LIVE { 0.0 } else { f32::MIN });
        let mask = HostTensor::<f32, m![Padded]>::from_vec(mask)
            .to_hbm::<Chip, m![Padded]>(&mut device.pdma)
            .await
            .unwrap();
        let ranks = Argmax::<Padded>::load(&mut Context::new(&mut device), &Checkpoint::default());
        let ranks = ranks.unwrap();
        // Out of order across both spans, the last span partial; none, and a tile of eight.
        let chosen: Vec<usize> = (0..140).map(|index| (index * 37 + 11) % rows).collect();
        let selections = [&chosen[..], &[], &[517, 2, 513, 1, 0]];
        let gains = [None, Some((2.0, 1.0)), Some((1.0, 2.0))];
        for (selection, gain) in selections.iter().flat_map(|s| gains.map(|g| (*s, g))) {
            let mut offsets = vec![0; Capacity::SIZE];
            for (slot, &row) in selection.iter().enumerate() {
                offsets[slot] = i32::try_from(row * 32).unwrap();
            }
            let selected = Selected {
                live: count(selection.len())
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                rows: HostTensor::<i32, m![Capacity]>::from_vec(offsets)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            };
            let mut trace = place(vec![bf16::from_f32(0.0); Capacity::SIZE * Hidden::SIZE])
                .to_hbm::<Chip, Blocked<Hidden, Capacity>>(&mut device.pdma)
                .await
                .unwrap();
            let argmax = Load::load(&mut Context::new(None), &ranks).unwrap();
            let head: Head<'_> = Greedy::new(Affine::new(&weight, &mask), argmax);
            let tokens = match gain {
                None => launch(
                    plain,
                    (
                        &mut device,
                        &Then(Spans, head),
                        &launch_rows,
                        &selected,
                        &input_rows,
                        &mut trace,
                    ),
                )
                .await
                .unwrap(),
                Some((first, second)) => {
                    let mut gains = vec![bf16::from_f32(1.0); Hidden::SIZE];
                    gains[0] = bf16::from_f32(first);
                    gains[1] = bf16::from_f32(second);
                    let norm = RmsNorm::new(Weight::new(
                        HostTensor::<bf16, m![Hidden]>::from_vec(gains)
                            .to_hbm(&mut device.pdma)
                            .await
                            .unwrap(),
                    ));
                    let model = Then(Then(Spans, norm), head);
                    launch(
                        normalized,
                        (
                            &mut device,
                            &model,
                            &launch_rows,
                            &selected,
                            &input_rows,
                            &mut trace,
                        ),
                    )
                    .await
                    .unwrap()
                }
            };
            let tokens = tokens
                .to_host::<m![Capacity]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (slot, &row) in selection.iter().enumerate() {
                let want = token(row, gain.unwrap_or((1.0, 1.0)));
                assert_eq!(
                    tokens[slot] as usize, want,
                    "{gain:?} slot {slot} row {row}"
                );
            }
            let trace = trace
                .to_host::<Blocked<Hidden, Capacity>>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let span = Span::SIZE;
            for chunk in 0..Hidden::SIZE / 16 {
                for row in 0..Capacity::SIZE {
                    for lane in 0..16 {
                        let at = (chunk * Capacity::SIZE + row) * 16 + lane;
                        let want = if row < span {
                            input[row * Hidden::SIZE + chunk * 16 + lane]
                        } else {
                            bf16::from_f32(0.0)
                        };
                        assert_eq!(
                            trace[at].to_f32(),
                            want.to_f32(),
                            "{gain:?} body end: row {row} feature {}",
                            chunk * 16 + lane
                        );
                    }
                }
            }
        }
    }
}
