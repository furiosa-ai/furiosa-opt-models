//! Projection onto a vocabulary compiled with more rows than the checkpoint has tokens.

use crate::Module;
use crate::ops::{add::Add, layout::axes::Chip, linear::Affine};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Scores of `P` compiled vocabulary rows whose first `V` are the checkpoint's tokens; padded rows
/// score negative infinity, so through order-preserving stages no padded row exceeds a token, and
/// the first-maximum selection keeps the token a tie puts first.
#[derive(DeviceSend)]
pub struct Padded<V: AxisName, P: AxisName, H: AxisName, Chips: M = Chip> {
    /// BF16 [compiled rows, hidden width] weights; the rows from `V` on are zero.
    projection: Affine<HbmTensor<bf16, Chips, m![P, H]>, ()>,
    /// Zero for the checkpoint's tokens and negative infinity for the padded rows.
    mask: HbmTensor<bf16, Chips, m![P]>,
    vocabulary: PhantomData<V>,
}

impl<V: AxisName, P: AxisName, H: AxisName, Chips: M> Padded<V, P, H, Chips> {
    pub fn new(
        weight: HbmTensor<bf16, Chips, m![P, H]>,
        mask: HbmTensor<bf16, Chips, m![P]>,
    ) -> Self {
        Self {
            projection: Affine::new(weight, ()),
            mask,
            vocabulary: PhantomData,
        }
    }

    /// The mask's values: zero for the checkpoint's tokens and negative infinity for the padded
    /// rows.
    pub fn mask() -> Vec<bf16> {
        (0..P::SIZE)
            .map(|row| match row < V::SIZE {
                true => bf16::from_f32(0.0),
                false => bf16::from_f32(f32::NEG_INFINITY),
            })
            .collect()
    }
}

impl<V: AxisName, P: AxisName, H: AxisName, Chips: M>
    Module<(&mut Device, &HbmTensor<bf16, Chips, m![H]>)> for Padded<V, P, H, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![P]>;

    fn forward(
        &self,
        (device, input): (&mut Device, &HbmTensor<bf16, Chips, m![H]>),
    ) -> Self::Output {
        const { assert!(V::SIZE < P::SIZE) };
        let scores = self.projection.forward((&mut *device, input));
        Add.forward((device, (&scores, &self.mask)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Then;
    use crate::ops::vocabulary::Argmax;

    // A checkpoint vocabulary that is no multiple of 256, padded to the next first-maximum tiling.
    axes![Hidden = 256, Tokens = 202_048, Rows = 202_752];

    type Head = Padded<Tokens, Rows, Hidden>;

    #[device(chip = 1)]
    fn project(
        device: &mut Device,
        head: &Head,
        input: &HbmTensor<bf16, Chip, m![Hidden]>,
    ) -> HbmTensor<bf16, Chip, m![Rows]> {
        head.forward((device, input))
    }

    #[device(chip = 1)]
    fn select(
        device: &mut Device,
        head: &Head,
        selector: &Argmax<Rows>,
        input: &HbmTensor<bf16, Chip, m![Hidden]>,
    ) -> HbmTensor<i32, Chip, m![1 # 4]> {
        Then(head, selector).forward((device, input))
    }

    /// The first maximum of the checkpoint's tokens wins though every token scores below zero, even
    /// far below any finite penalty, and padded rows hold larger products; token scores pass
    /// unchanged, padded rows score negative infinity.
    #[tokio::test(flavor = "current_thread")]
    async fn selects_first_token_maximum() {
        let (live, rows) = (Tokens::SIZE, Rows::SIZE);
        // Case `k` reads input columns 2k and 2k + 1; tokens without an update score -1 there and
        // padded rows without one score zero, as the loader's zero rows do. The first case reads
        // its columns at 2^31, so every token scores -2^31.
        let cases: [(&[(usize, f32)], usize); 9] = [
            (&[], 0),
            (&[(42, -0.5)], 42),
            (&[(live - 1, -0.25)], live - 1),
            (&[(7, -0.5), (live / 3 + 19, -0.5)], 7),
            (
                &[
                    (live, 10.0),
                    (live + 100, 10.0),
                    (rows - 1, 10.0),
                    (1234, -0.5),
                ],
                1234,
            ),
            (&[(99_999, 7.0), (rows - 1, 100.0)], 99_999),
            (&[], 0),
            (&[(0, -0.125), (live, -0.125)], 0),
            (&[(live - 1, 3.0), (live, 3.0)], live - 1),
        ];
        let scale = |case: usize| if case == 0 { 2f32.powi(31) } else { 1.0 };
        let mut weight = vec![bf16::from_f32(0.0); rows * Hidden::SIZE];
        for row in 0..live {
            for case in 0..cases.len() {
                weight[row * Hidden::SIZE + 2 * case] = bf16::from_f32(-1.0);
            }
        }
        for (case, &(updates, _)) in cases.iter().enumerate() {
            for &(token, value) in updates {
                // Two BF16 coefficients hold each score exactly through the FP32 contraction.
                let high = bf16::from_f32(value);
                let low = bf16::from_f32(value - high.to_f32());
                weight[token * Hidden::SIZE + 2 * case] = high;
                weight[token * Hidden::SIZE + 2 * case + 1] = low;
            }
        }
        let expected = |case: usize, row: usize| {
            (weight[row * Hidden::SIZE + 2 * case].to_f32()
                + weight[row * Hidden::SIZE + 2 * case + 1].to_f32())
                * scale(case)
        };
        let mut device = Device::new(select.topology()).unwrap();
        let head = Head::new(
            HostTensor::<bf16, m![Rows, Hidden]>::from_vec(weight.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            HostTensor::<bf16, m![Rows]>::from_vec(Head::mask())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        );
        let selector = Argmax::<Rows> {
            indices: HostTensor::<i32, m![Rows]>::from_vec(
                (0..rows).map(|i| (rows - 1 - i) as i32),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        };
        for (case, &(_, selected)) in cases.iter().enumerate() {
            let mut values = vec![bf16::from_f32(0.0); Hidden::SIZE];
            values[2 * case] = bf16::from_f32(scale(case));
            values[2 * case + 1] = bf16::from_f32(scale(case));
            let input = HostTensor::<bf16, m![Hidden]>::from_vec(values)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let chosen = launch(select, (&mut device, &head, &selector, &input))
                .await
                .unwrap()
                .to_host::<m![1 # 4]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec()[0];
            assert_eq!(chosen as usize, selected, "case {case}");
            let scores = launch(project, (&mut device, &head, &input))
                .await
                .unwrap()
                .to_host::<m![Rows]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (row, score) in scores.iter().enumerate() {
                if row < live {
                    assert_eq!(
                        score.to_f32(),
                        expected(case, row),
                        "case {case} token {row}"
                    );
                } else {
                    assert_eq!(
                        score.to_f32(),
                        f32::NEG_INFINITY,
                        "case {case} padded row {row}"
                    );
                }
            }
        }
    }
}
