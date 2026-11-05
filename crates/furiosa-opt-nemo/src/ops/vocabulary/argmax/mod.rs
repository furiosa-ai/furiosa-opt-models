mod candidates;
pub mod sharded;
pub(crate) mod vector;

pub(super) use candidates::Candidates;
pub use vector::Elements;

/// Device-resident first-maximum token per row in the fused candidate reduction mapping.
pub type Selected<Rows> = candidates::Selected<Rows>;

/// Finite 2^30 mask penalty; padded logits must remain below valid logits after subtraction.
pub const PENALTY: f32 = 1_073_741_824.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::{Count, tests::count};
    use crate::ops::vocabulary::{Greedy, Tokens, Vocab, VocabPad};
    use crate::ops::weight::Weight;
    use crate::{
        Module,
        ops::layout::Live,
        ops::layout::axes::{Chip, Eight, Rows},
        ops::normalization::Spread,
    };
    use furiosa_opt_std::prelude::*;

    axes![Hidden = 1024];

    /// A checkpoint vocabulary below the compiled one, so the mask, not the padding, bounds the
    /// selection.
    const LIVE: usize = Vocab::SIZE - 256;

    #[device(chip = 1)]
    fn selected<R: AxisName>(
        device: &mut Device,
        head: &Greedy<Hidden>,
        rows: &Count,
        input: &HbmTensor<bf16, Chip, m![R, Hidden]>,
    ) -> Tokens<R> {
        let input: Spread<Hidden, R> = input.to_dm(&mut device.tdma);
        head.forward((&mut *device, (&input, Live::read(rows))))
            .to_hbm(&mut device.tdma)
    }

    /// Fused Greedy selects each live row's first maximum among the checkpoint's tokens across ties
    /// and tile boundaries, never a padded column, with every row live and with only the first
    /// rows of one or two row tiles.
    #[tokio::test(flavor = "current_thread")]
    async fn selects_first_maximum() {
        selects_first_maximum_of::<Eight>(&[8, 1]).await;
        selects_first_maximum_of::<Rows<32>>(&[32]).await;
        selects_first_maximum_of::<Rows<64>>(&[64, 33, 1]).await;
    }

    /// Row `r` runs case `r % 8` on its own pair of input columns, for each count of live rows.
    async fn selects_first_maximum_of<R: AxisName>(lives: &[usize]) {
        let cases: [(&[(usize, f32)], usize); Eight::SIZE] = [
            (&[(42, 5.0)], 42),
            (&[(7, 5.0), (VocabPad::SIZE / 3 + 19, 5.0)], 7),
            (
                &[(LIVE - 1, 5.0), (LIVE, 10.0), (VocabPad::SIZE - 1, 10.0)],
                LIVE - 1,
            ),
            (&[(0, 5.0f32.next_down()), (LIVE - 1, 5.0)], LIVE - 1),
            (
                &[(2 * VocabPad::SIZE / 3 + 5, 3.0)],
                2 * VocabPad::SIZE / 3 + 5,
            ),
            (&[(1234, -0.5), (1233, -0.5)], 1233),
            (
                &[(VocabPad::SIZE / 3 - 1, 2.0), (VocabPad::SIZE / 3, 2.0)],
                VocabPad::SIZE / 3 - 1,
            ),
            (&[(99_999, 7.0)], 99_999),
        ];
        let case = |row: usize| cases[row % Eight::SIZE];
        let mut weight = vec![bf16::from_f32(0.0); VocabPad::SIZE * Hidden::SIZE];
        for token in 0..VocabPad::SIZE {
            for row in 0..R::SIZE {
                weight[token * Hidden::SIZE + 2 * row] = bf16::from_f32(-1.0);
            }
        }
        let mut input = vec![bf16::from_f32(0.0); R::SIZE * Hidden::SIZE];
        for row in 0..R::SIZE {
            input[row * Hidden::SIZE + 2 * row] = bf16::from_f32(1.0);
            input[row * Hidden::SIZE + 2 * row + 1] = bf16::from_f32(1.0);
            for &(token, value) in case(row).0 {
                // Two BF16 coefficients preserve the FP32 next_down case through the contraction.
                let high = bf16::from_f32(value);
                let low = bf16::from_f32(value - high.to_f32());
                assert_eq!(high.to_f32() + low.to_f32(), value);
                weight[token * Hidden::SIZE + 2 * row] = high;
                weight[token * Hidden::SIZE + 2 * row + 1] = low;
            }
        }
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let weight = Weight::new(
            HostTensor::<bf16, m![VocabPad, Hidden]>::from_vec(weight)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        );
        let input = HostTensor::<bf16, m![R, Hidden]>::from_vec(input)
            .to_hbm::<Chip, m![R, Hidden]>(&mut device.pdma)
            .await
            .unwrap();
        let prepared = Greedy::<Hidden>::prepare(LIVE).unwrap().tensors;
        let words = |name: &str| {
            prepared[name]
                .data
                .chunks_exact(4)
                .map(|word| [word[0], word[1], word[2], word[3]])
                .collect::<Vec<_>>()
        };
        let mask = HostTensor::<f32, m![VocabPad]>::from_vec(
            words("mask").into_iter().map(f32::from_le_bytes),
        )
        .to_hbm::<Chip, m![VocabPad]>(&mut device.pdma)
        .await
        .unwrap();
        let iota = HostTensor::<i32, m![VocabPad]>::from_vec(
            words("iota").into_iter().map(i32::from_le_bytes),
        )
        .to_hbm::<Chip, m![VocabPad]>(&mut device.pdma)
        .await
        .unwrap();

        let head = Greedy::<Hidden>::new(weight, mask, iota);
        for &live in lives {
            let rows = count(live).to_hbm(&mut device.pdma).await.unwrap();
            let tokens = launch(selected, (&mut device, &head, &rows, &input))
                .await
                .unwrap()
                .to_host::<m![R, 1 # 4]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for row in 0..live {
                assert_eq!(
                    tokens[row * 4] as usize,
                    case(row).1,
                    "row {row} of {live} live"
                );
            }
        }
    }
}
