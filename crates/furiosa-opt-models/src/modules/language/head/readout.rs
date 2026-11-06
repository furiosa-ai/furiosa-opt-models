//! A greedy head over a body's normalized rows: it reads the tokens of the rows a launch selects.

use super::{InDm, Tree};
use crate::Module;
use crate::ops::index::scatter::Store;
use crate::ops::normalization::span::{Selection, Span};
use crate::ops::normalization::{Blocked, RmsNorm, Spread};
use crate::ops::reduction;
use crate::tensor::live::{Fill, Live, OnTile, Spill};
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
        self.read_with(context, hidden, selected, tokens, ());
    }

    /// Gather only the requested rows, normalizing each 8/64/128-row head tile. Limiting
    /// the norm to a tile also keeps its row inverses within the vector register file.
    pub(crate) fn read_normalized<Capacity: AxisName, const EPSILON: usize>(
        &self,
        context: &mut Device,
        hidden: &HbmTensor<bf16, Chip, Blocked<Width, Capacity>>,
        selected: &Selected<Capacity>,
        tokens: &mut HbmTensor<i32, Chip, m![Capacity]>,
        norm: &RmsNorm<Width, EPSILON, Chip>,
    ) {
        self.read_with(context, hidden, selected, tokens, norm);
    }

    fn read_with<Capacity: AxisName, P: Prepare<Width>>(
        &self,
        context: &mut Device,
        hidden: &HbmTensor<bf16, Chip, Blocked<Width, Capacity>>,
        selected: &Selected<Capacity>,
        tokens: &mut HbmTensor<i32, Chip, m![Capacity]>,
        prepare: P,
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
            let head = self.to_dm(&mut *context);
            tile.tiles::<_, Rows<128>, Rows<64>, Rows<8>>(
                &mut *context,
                &mut Tiles {
                    selection: &selection,
                    prepare: &prepare,
                    head: &head,
                    tokens: &mut *tokens,
                    tile,
                },
            );
        }
    }
}

/// A tile's optional final normalization, specialized at compile time.
trait Prepare<Width: AxisName> {
    fn apply<R: AxisName>(&self, device: &mut Device, rows: Spread<Width, R>) -> Spread<Width, R>;
}
impl<Width: AxisName> Prepare<Width> for () {
    fn apply<R: AxisName>(&self, _: &mut Device, rows: Spread<Width, R>) -> Spread<Width, R> {
        rows
    }
}
impl<Width: AxisName, const EPSILON: usize> Prepare<Width> for &RmsNorm<Width, EPSILON, Chip> {
    fn apply<R: AxisName>(&self, device: &mut Device, rows: Spread<Width, R>) -> Spread<Width, R> {
        crate::Op::forward(*self, device, &rows)
    }
}

/// One span's tiles of the selected rows: each gathered and read by the head, and its tokens
/// stored where its rows sit among the launch's.
struct Tiles<'a, Width: AxisName, V: AxisName, Capacity: AxisName, P> {
    selection: &'a Selection<'a, Width, Capacity>,
    prepare: &'a P,
    head: &'a InDm<'a, Width, V>,
    tokens: &'a mut HbmTensor<i32, Chip, m![Capacity]>,
    tile: Live<(), Capacity>,
}

impl<R: AxisName, Width: AxisName, V: AxisName, Capacity: AxisName, P: Prepare<Width>>
    OnTile<Device, R> for Tiles<'_, Width, V, Capacity, P>
{
    fn run(&mut self, device: &mut Device, index: usize) {
        let rows: Spread<Width, R> = self.selection.gather(&mut *device, self.tile, index);
        let rows = self.prepare.apply(device, rows);
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
    use crate::tensor::form::Weight;
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
    #[device(chip = 1)]
    fn selected_normalized(
        device: &mut Device,
        head: &Head<'_>,
        norm: &RmsNorm<Hidden>,
        selected: &Selected<Capacity>,
        input: &Place,
    ) -> Tokens {
        let mut tokens = Tokens::new();
        head.read_normalized(device, input, selected, &mut tokens, norm);
        tokens
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore = "opens a device"]
    async fn normalizes_large_selected_tile_and_tail() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        // Nonuniform gain changes the winner once, and again if applied twice.
        let gains = [2.0_f32, 1.0];
        let pair = |row: usize| [1.0_f32, [1.5, 3.0, 0.5, 4.0][row % 4]];
        let mut input = vec![bf16::from_f32(0.0); Capacity::SIZE * Hidden::SIZE];
        for row in 0..Capacity::SIZE {
            for (feature, value) in pair(row).into_iter().enumerate() {
                input[row * Hidden::SIZE + feature] = bf16::from_f32(value);
            }
        }
        let input = HostTensor::<bf16, Blocked<Hidden, Capacity>>::from_vec(blocked(
            &input,
            Hidden::SIZE,
            16,
        ))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let mut weights = vec![bf16::from_f32(0.0); Padded::SIZE * Hidden::SIZE];
        weights[FIRST * Hidden::SIZE] = bf16::from_f32(1.0);
        weights[SECOND * Hidden::SIZE + 1] = bf16::from_f32(1.0);
        let weights = HostTensor::<bf16, m![Padded, Hidden]>::from_vec(weights)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        // A constant competing logit makes the RMS divisor observable to argmax.
        let mut bias = vec![f32::MIN; Padded::SIZE];
        bias[FIRST] = 0.0;
        bias[SECOND] = 0.0;
        bias[TIED] = 55.0;
        let bias = HostTensor::<f32, m![Padded]>::from_vec(bias)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let ranks =
            Argmax::<Padded>::load(&mut Context::new(&mut device), &Checkpoint::default()).unwrap();
        let argmax = Load::load(&mut Context::new(None), &ranks).unwrap();
        let head: Head<'_> = Greedy::new(Affine::new(&weights, &bias), argmax);
        let mut gain = vec![bf16::from_f32(1.0); Hidden::SIZE];
        gain[0] = bf16::from_f32(gains[0]);
        let norm = RmsNorm::new(Weight::new(
            HostTensor::<bf16, m![Hidden]>::from_vec(gain)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        ));
        let oracle = |row: usize, times: usize| {
            let mut values = pair(row);
            for _ in 0..times {
                let inv = (values.iter().map(|v| v * v).sum::<f32>() / Hidden::SIZE as f32 + 1e-6)
                    .sqrt()
                    .recip();
                for i in 0..2 {
                    values[i] = bf16::from_f32(values[i] * inv * gains[i]).to_f32();
                }
            }
            [(FIRST, values[0]), (SECOND, values[1]), (TIED, 55.0)]
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0 as i32
        };
        assert!((0..4).any(|row| oracle(row, 1) != oracle(row, 0)));
        assert!((0..4).any(|row| oracle(row, 1) != oracle(row, 2)));
        // Shuffled rows span both hidden-storage halves, including their exact boundary.
        let chosen: Vec<_> = [511, 512]
            .into_iter()
            .chain((0..127).map(|i| (i * 37 + 11) % 1024))
            .collect();
        for count in [129] {
            let mut offsets = vec![0_i32; Capacity::SIZE];
            for (at, &row) in chosen[..count].iter().enumerate() {
                offsets[at] = row as i32 * 32;
            }
            let selected = Selected {
                live: HostTensor::<i32, m![1]>::from_vec([count as i32])
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                rows: HostTensor::<i32, m![Capacity]>::from_vec(offsets)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            };
            let normalized_tokens = launch(
                selected_normalized,
                (&mut device, &head, &norm, &selected, &input),
            )
            .await
            .unwrap()
            .to_host::<m![Capacity]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
            for (at, &row) in chosen[..count].iter().enumerate() {
                assert_eq!(
                    normalized_tokens[at],
                    oracle(row, 1),
                    "normalized count={count} slot={at} row={row}"
                );
            }
        }
    }
}
