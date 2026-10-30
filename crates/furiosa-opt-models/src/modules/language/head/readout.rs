//! A greedy head over a body's normalized rows: it reads the tokens of the rows a launch selects.

use super::{InDm, Tree};
use crate::Module;
use crate::modules::generic::schedule::{Fill, Spill};
use crate::ops::index::scatter::Store;
use crate::ops::normalization::span::{Selection, Span, Tile};
use crate::ops::normalization::{Blocked, Spread};
use crate::ops::reduction;
use crate::tensor::live::{Live, OnTile};
use crate::tensor::scalar::Value;
use crate::tensor::{axis::Rows, parallel::Chip};
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;

/// The rows a launch returns tokens of: their count and each one's blocked hidden-row offset.
#[derive(DeviceSend)]
pub struct Selected<Capacity: AxisName> {
    pub live: Value,
    pub rows: HbmTensor<i32, Chip, m![Capacity]>,
}

impl<Capacity: AxisName> Selected<Capacity> {
    /// The rows of span `tile` of the selected ones among a launch's normalized rows `hidden`, in
    /// order, as rows in HBM; rows past the selected ones are unspecified.
    pub(super) fn span<Width: AxisName>(
        &self,
        context: &mut Device,
        hidden: &HbmTensor<bf16, Chip, Blocked<Width, Capacity>>,
        tile: Live<(), Capacity>,
    ) -> HbmTensor<bf16, Chip, m![Span, Width]> {
        let selection = Selection {
            hidden,
            rows: &self.rows,
        };
        let rows = selection.fill(&mut *context, tile);
        crate::Op::forward(&Store, &mut *context, &rows)
    }

    /// The hidden values of the selected rows among a launch's normalized rows `hidden`, in
    /// order, into the first rows of `rows`; called once the rows' last effect is issued.
    pub(crate) fn hidden<Width: AxisName>(
        &self,
        context: &mut Device,
        hidden: &HbmTensor<bf16, Chip, Blocked<Width, Capacity>>,
        rows: &mut HbmTensor<bf16, Chip, m![Capacity, Width]>,
    ) {
        let chosen = Live::<(), Capacity>::read(&self.live);
        for index in 0..chosen.count::<Span>() {
            let tile = chosen.tile::<Span>(index);
            let span = self.span(&mut *context, hidden, tile);
            span.view().to_hbm_view(
                &mut context.tdma,
                rows.view_mut().tile::<m![Capacity], { Span::SIZE }, m![
                    Capacity = { Span::SIZE } #{!} Capacity, Width
                ]>(tile.start()),
            );
        }
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
    use crate::ops::linear::Affine;
    use crate::ops::normalization::tests::blocked;
    use crate::ops::reduction::Argmax;
    use crate::tests::DEVICE;

    axes![Hidden = 4096, Padded = 152064];

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

    /// The checkpoint's tokens; the compiled vocabulary pads them.
    const LIVE: usize = 151_936 - 256;
    /// Tokens of the rows' cases: the first and second features' own, two tied on the third.
    const FIRST: usize = 5;
    const SECOND: usize = 9;
    const TIED: usize = 77;
    const LATER: usize = 100;

    /// The tokens the head reads of the selected rows among a launch's rows.
    #[device(chip = 1)]
    fn read_out(
        device: &mut Device,
        head: &Head<'_>,
        selected: &Selected<Capacity>,
        input: &Place,
    ) -> Tokens {
        let mut tokens = Tokens::new();
        head.read(device, input, selected, &mut tokens);
        tokens
    }

    /// Row `row`'s feature values: a feature of its own, the first two, the third, or the second.
    fn features(row: usize) -> &'static [usize] {
        const CASES: [&[usize]; 4] = [&[], &[0, 1], &[2], &[1]];
        CASES[row % 4]
    }

    /// Row `row`'s token: its own, the first of two tied, or the second feature's.
    fn token(row: usize) -> usize {
        match row % 4 {
            0 => 1000 + row,
            1 => FIRST,
            2 => TIED,
            _ => SECOND,
        }
    }

    /// Each selected row's first maximum among the checkpoint's tokens, in selection order, read
    /// from a launch's rows.
    #[tokio::test(flavor = "current_thread")]
    async fn reads_each_selected_rows_first_maximum() {
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
        for selection in [&chosen[..], &[], &[517, 2, 513, 1, 0]] {
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
            let argmax = Load::load(&mut Context::new(None), &ranks).unwrap();
            let head: Head<'_> = Greedy::new(Affine::new(&weight, &mask), argmax);
            let tokens = launch(read_out, (&mut device, &head, &selected, &input_rows))
                .await
                .unwrap();
            let tokens = tokens
                .to_host::<m![Capacity]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (slot, &row) in selection.iter().enumerate() {
                assert_eq!(tokens[slot] as usize, token(row), "slot {slot} row {row}");
            }
        }
    }
}
