//! Bidirectional attention of a 2,048-row tile whose keys fit one 2,048-key span, eight heads at
//! a time: each trip is a unit of one span, which the long-sequence flash kernel's last-span step
//! attends; [`super::flash`] runs it on these types.
//!
//! - A cluster attends four heads of a trip's eight; its 256 slices are the four heads by two
//!   query groups of 1,024 rows by 32 key slices of 64 keys. A slice keeps its keys and its values
//!   in its tensor register file; its group's queries stream past the keys through the switch,
//!   and the weights it computed stream past the values.
//! - The values carry 16 channels of the keys' visibility after their 64, so one product holds a
//!   row's weighted values and its mass; the products sum over the key slices onto the slice that
//!   holds the row.
//! - Scores round to BF16 before they are scaled; the maximum and the products are FP32.

use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use super::flash::{Duo, Span, SpanStep};
use super::full::heads::tiles;
use crate::ops::rotary;

axes![Rows = 2048, Head = 64, Channel = 80];

/// The heads of a trip one cluster attends at once, as many as a span step spreads over a cluster
/// for the tile's rows and head width: four.
#[derive(Debug, Clone)]
pub struct Quad;

impl AxisName for Quad {
    const NAME: Ident = Ident::new("Quad");
    const SIZE: usize = SpanStep::<Head>::heads(Rows::SIZE);
}

/// Every head of the tile's rows, head-major, eight heads a trip: queries, keys, values or the
/// attended values.
pub type Tokens<Trips, Chips = m![1]> = HbmTensor<bf16, Chips, m![Trips, Duo, Quad, Rows, Head]>;
/// A view of [`Tokens`].
pub type TokensView<'a, Trips, Chips = m![1]> =
    HbmTensorView<'a, bf16, Chips, m![Trips, Duo, Quad, Rows, Head]>;
/// Each key's visibility: 1 where the key is visible, 0 where it is absent; an absent key's row
/// may hold any finite values.
pub type Visible<Chips = m![1]> = HbmTensor<bf16, Chips, m![Span]>;

/// The trips over the heads of rows `W` features wide, eight heads a trip.
#[derive(Clone, Debug)]
pub struct Trips<W>(PhantomData<W>);

impl<W: AxisName> AxisName for Trips<W> {
    const NAME: Ident = Ident::new("Trips");

    const SIZE: usize = {
        assert!(W::SIZE.is_multiple_of(Duo::SIZE * Quad::SIZE * Head::SIZE));
        W::SIZE / (Duo::SIZE * Quad::SIZE * Head::SIZE)
    };
}

/// Head-major rows of `W` features.
pub type Major<W, Chips = m![1]> = tiles::Major<Rows, W, Head, Chips>;

/// Each row's rotation of a 64-wide head.
pub type Rotation<Chips = m![1]> = rotary::Rotation<Head, Chips, Rows>;

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use super::{Duo, Head, Quad, Rows, Span, Tokens, Visible};
    use crate::ops::attention::Attention;
    use crate::ops::normalization::tests::random;
    use crate::tensor::scalar::F32;
    use crate::tests::DEVICE;
    use std::marker::PhantomData;

    axes![One = 1];

    /// The score multiplier of 64-wide heads, `1 / 8`.
    const EIGHTH: usize = 0.125_f32.to_bits() as usize;

    #[device(chip = 1)]
    fn attend_heads(
        device: &mut Device,
        query: &Tokens<One>,
        key: &Tokens<One>,
        value: &Tokens<One>,
        visible: &Visible,
    ) -> Tokens<One> {
        use crate::Op;
        Attention {
            heads: PhantomData::<One>,
            scale: F32::<EIGHTH>,
        }
        .forward(device, ((query.view(), key.view(), value.view()), visible))
    }

    /// Eight heads over the first 1,797 keys, the absent ones four times larger so a leak shows,
    /// against an FP64 softmax with BF16-rounded scores on every 7th row.
    #[tokio::test(flavor = "current_thread")]
    async fn attends_the_visible_keys() {
        const LIVE: usize = 1797;
        let _lease = DEVICE.lock().await;
        let heads = One::SIZE * Duo::SIZE * Quad::SIZE;
        let width = heads * Head::SIZE;
        let tokens = |seed: u64, (gain, absent): (f32, f32)| -> Vec<bf16> {
            random(Rows::SIZE * width, seed)
                .into_iter()
                .enumerate()
                .map(|(at, x)| {
                    let scale = if at / width >= LIVE { absent } else { 1.0 };
                    bf16::from_f32(x.to_f32() * gain * scale)
                })
                .collect()
        };
        // Scaled queries make the softmax peaked.
        let (q, k, v) = (
            tokens(0x9e37_79b9_7f4a_7c15, (8.0, 1.0)),
            tokens(0x2545_f491_4f6c_dd1d, (1.0, 4.0)),
            tokens(0x1405_7b7e_f767_814f, (1.0, 4.0)),
        );
        let mut device = Device::new(attend_heads.topology()).unwrap();
        // Token-major values, head-major on chip.
        let upload = async |device: &mut Device, values: &[bf16]| -> Tokens<One> {
            let major: Vec<bf16> = (0..values.len())
                .map(|at| {
                    let (head, row, d) = (
                        at / (Rows::SIZE * Head::SIZE),
                        at / Head::SIZE % Rows::SIZE,
                        at % Head::SIZE,
                    );
                    values[(row * heads + head) * Head::SIZE + d]
                })
                .collect();
            HostTensor::<bf16, m![One, Duo, Quad, Rows, Head]>::from_vec(major)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap()
        };
        let (query, key, value) = (
            upload(&mut device, &q).await,
            upload(&mut device, &k).await,
            upload(&mut device, &v).await,
        );
        let visible: Visible = HostTensor::<bf16, m![Span]>::from_vec(
            (0..Span::SIZE)
                .map(|at| bf16::from_f32(if at < LIVE { 1.0 } else { 0.0 }))
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let major = launch(attend_heads, (&mut device, &query, &key, &value, &visible))
            .await
            .unwrap()
            .to_host::<m![One, Duo, Quad, Rows, Head]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let got: Vec<bf16> = (0..major.len())
            .map(|at| {
                let (row, head, d) = (
                    at / (heads * Head::SIZE),
                    at / Head::SIZE % heads,
                    at % Head::SIZE,
                );
                major[(head * Rows::SIZE + row) * Head::SIZE + d]
            })
            .collect();
        let (mut error, mut norm) = (0.0f64, 0.0f64);
        for row in (0..Rows::SIZE).step_by(7) {
            for head in 0..heads {
                let at = |t: &[bf16], r: usize, d: usize| {
                    f64::from(t[(r * heads + head) * Head::SIZE + d].to_f32())
                };
                // The kernel rounds each score to BF16 before scaling it.
                let scores: Vec<f64> = (0..LIVE)
                    .map(|key| {
                        let dot: f64 = (0..Head::SIZE)
                            .map(|d| at(&q, row, d) * at(&k, key, d))
                            .sum();
                        0.125 * f64::from(bf16::from_f32(dot as f32).to_f32())
                    })
                    .collect();
                let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                let mass: f64 = weights.iter().sum();
                for d in 0..Head::SIZE {
                    let want = weights
                        .iter()
                        .enumerate()
                        .map(|(key, w)| w * at(&v, key, d))
                        .sum::<f64>()
                        / mass;
                    let found = at(&got, row, d);
                    assert!(
                        found.is_finite(),
                        "row {row} head {head} channel {d}: {found}"
                    );
                    error += (found - want).powi(2);
                    norm += want.powi(2);
                }
            }
        }
        let relative = (error / norm).sqrt();
        eprintln!("one-span attention {heads} heads x {LIVE} keys: relative L2 {relative:.3e}");
        assert!(relative < 1e-2, "relative L2 {relative}");
    }
}
