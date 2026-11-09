//! The attention's output projection over a launch's attended rows: its weights staged once, then
//! their projection a span at a time in three steps, of which the dense forms work in the last.

use furiosa_opt_std::prelude::*;

use crate::Begin;
use crate::Module;
use crate::ops::linear::Affine;
use crate::ops::normalization::span::{Owned, Span};
use crate::ops::pointwise::add::{Add, Term};
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use core::marker::PhantomData;
use core::task::Poll;

pub mod fp8;
pub mod matrix;
mod tiles;

type Cluster<QWidth> = m![QWidth / 2048];

pub(crate) struct ToDm<'a, QWidth: AxisName, Capacity: AxisName, Reach: AxisName> {
    pub attended: &'a HbmTensor<bf16, Chip, m![Reach, QWidth]>,
    pub start: usize,
    pub live: (usize, usize),
    pub capacity: PhantomData<Capacity>,
}

#[cfg(test)]
mod tests {
    use super::fp8::{Blocked, Stored};
    use super::*;
    use crate::ops::attention::geometry::{Hidden, QueryGroups, QueryWidth};
    use crate::ops::normalization::Blocked as Rows;
    use crate::ops::normalization::tests::random;
    use crate::tensor::axis;
    use crate::tensor::live::{Spill, tests::count};
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![KvHeads = 8, Head = 128];

    type H = Hidden<4096>;
    type QWidth = QueryWidth<4096>;
    type Groups = QueryGroups<4>;
    type All = axis::Rows<1024>;
    type Attended = HbmTensor<bf16, Chip, m![All, KvHeads, Groups, Head]>;
    type Projected<H = Hidden<4096>> = HbmTensor<bf16, Chip, Rows<H, All>>;
    type Dense<H = Hidden<4096>> = Affine<HbmTensor<bf16, Chip, m![H, QWidth]>, ()>;

    /// The projection of the live rows `count` names of a launch's attended rows as a schedule
    /// takes it: begun, its weights landed, then each span in its three steps, stored at its rows.
    macro_rules! project {
        ($name:ident, $projection:ty) => {
            #[device(chip = 1)]
            fn $name(
                device: &mut Device,
                projection: &$projection,
                attended: &Attended,
                count: &Value,
            ) -> Projected {
                let live = Live::<(), All>::read(count);
                let mut stored = Projected::new();
                let mut landed = projection.forward(&mut *device, (), &mut (), Begin(()));
                projection.forward(&mut *device, (), &mut (), &mut landed);
                for span in 0..live.count::<Span>() {
                    let tile = live.tile::<Span>(span);
                    let begun = Begin(Begin((&mut landed, tile.with(attended))));
                    let begun = projection.forward(&mut *device, (), &mut (), begun);
                    let begun = projection.forward(&mut *device, (), &mut (), begun);
                    let rows = projection.forward(&mut *device, (), &mut (), begun);
                    stored.spill(&mut *device, &rows, tile);
                }
                stored
            }
        };
    }

    project!(dense, Dense);
    project!(quantized, Stored<H, QWidth>);

    /// The dense projection into `W` hidden features of the live rows `count` names of a launch's
    /// attended rows, each span in its three steps, stored at its rows.
    #[device(chip = 1)]
    fn dense_into<W: AxisName>(
        device: &mut Device,
        projection: &Dense<W>,
        attended: &Attended,
        count: &Value,
    ) -> Projected<W> {
        let live = Live::<(), All>::read(count);
        let mut stored = Projected::<W>::new();
        // The dense weights land nothing: each span's last step reads them from HBM.
        projection.forward(&mut *device, (), &mut (), Begin(()));
        projection.forward(&mut *device, (), &mut (), &mut ());
        for span in 0..live.count::<Span>() {
            let tile = live.tile::<Span>(span);
            let begun = Begin(Begin((&mut (), tile.with(attended))));
            let begun = projection.forward(&mut *device, (), &mut (), begun);
            let begun = projection.forward(&mut *device, (), &mut (), begun);
            let rows = projection.forward(&mut *device, (), &mut (), begun);
            stored.spill(&mut *device, &rows, tile);
        }
        stored
    }

    /// The RMS of `got - expected` relative to that of `expected`.
    fn error(got: &[f64], expected: &[f64]) -> f64 {
        let (difference, norm) = got.iter().zip(expected).fold((0.0, 0.0), |(d, n), (g, e)| {
            (d + (g - e) * (g - e), n + e * e)
        });
        (difference / norm).sqrt()
    }

    /// One output projection fixture over both weight forms of the same logical `[H, QWidth]`
    /// weights, E4M3-representable coefficients times a power-of-two scale of each output: at one
    /// row and at two spans, the second partial, each live row matches the dense reference within the
    /// form's `tolerance` of its RMS, and rows past the live ones never reach them.
    async fn check<H: AxisName, P: DeviceSend>(
        device: &mut Device,
        projection: &P,
        run: impl AsyncFn(&mut Device, &P, &Value, &Attended) -> Projected<H>,
        weights: &[f32],
        tolerance: f64,
    ) {
        let width = QWidth::SIZE;
        let values = random(All::SIZE * width, 0x2545_f491_4f6c_dd1d);
        let rows: Vec<f32> = values.iter().map(|v| v.to_f32()).collect();
        let outputs = async |device: &mut Device, live: usize, padding: f32| {
            let mut padded = values.clone();
            padded[live * width..].fill(bf16::from_f32(padding));
            let attended = HostTensor::<bf16, m![All, KvHeads, Groups, Head]>::from_vec(padded)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let count = count(live).to_hbm(&mut device.pdma).await.unwrap();
            run(&mut *device, projection, &count, &attended)
                .await
                .to_host::<m![All, H]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec()
        };
        for live in [1, 600] {
            let got = outputs(&mut *device, live, 0.0).await;
            let padded = outputs(&mut *device, live, 1000.0).await;
            assert_eq!(
                got[..live * H::SIZE],
                padded[..live * H::SIZE],
                "{live} live rows: padding reached live rows"
            );
            for row in [0, live / 2, live - 1, 511, 512] {
                if row >= live {
                    continue;
                }
                let x = &rows[row * width..][..width];
                let expected: Vec<f64> = weights
                    .chunks_exact(width)
                    .map(|w| {
                        w.iter()
                            .zip(x)
                            .map(|(w, x)| f64::from(*w) * f64::from(*x))
                            .sum()
                    })
                    .collect();
                let got: Vec<f64> = got[row * H::SIZE..][..H::SIZE]
                    .iter()
                    .map(|v| f64::from(v.to_f32()))
                    .collect();
                let error = error(&got, &expected);
                assert!(
                    error <= tolerance,
                    "{live} live rows, row {row}: relative error {error}"
                );
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn projects_each_row_in_both_forms() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let codes: Vec<f8e4m3> = random(H::SIZE * QWidth::SIZE, 0x9e37)
            .iter()
            .map(|v| f8e4m3::from_f32(v.to_f32()))
            .collect();
        let channels: Vec<f32> = (0..H::SIZE)
            .map(|o| 1.0 / f32::from(1u16 << (4 + o % 3)))
            .collect();
        let weights: Vec<f32> = codes
            .iter()
            .zip(channels.iter().flat_map(|&s| [s; QWidth::SIZE]))
            .map(|(c, s)| c.to_f32() * s)
            .collect();
        let matrix = Affine::new(
            HostTensor::<bf16, m![H, QWidth]>::from_vec(
                weights
                    .iter()
                    .map(|w| bf16::from_f32(*w))
                    .collect::<Vec<_>>(),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
            (),
        );
        let run = async |device: &mut Device, p: &Dense, live: &Value, attended: &Attended| {
            launch(dense, (device, p, attended, live)).await.unwrap()
        };
        check(&mut device, &matrix, run, &weights, 0.004).await;
        let quantized_weights = Affine::new(
            crate::tensor::form::fp8::Rows::new(
                HostTensor::<f8e4m3, m![H, QWidth]>::from_vec(codes)
                    .to_hbm::<Chip, Blocked<H, QWidth>>(&mut device.pdma)
                    .await
                    .unwrap(),
                HostTensor::<f32, m![H]>::from_vec(channels)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ),
            (),
        );
        let run = async |device: &mut Device, p: &Stored<H, QWidth>, live: &Value, a: &Attended| {
            launch(quantized, (device, p, a, live)).await.unwrap()
        };
        check(&mut device, &quantized_weights, run, &weights, 0.04).await;
    }

    /// The dense projection into each listed hidden width, padded feature slices included,
    /// matches the reference on the live rows of two spans.
    #[tokio::test(flavor = "current_thread")]
    async fn projects_into_every_width() {
        async fn projects<W: AxisName>(device: &mut Device) {
            let weights: Vec<f32> = random(W::SIZE * QWidth::SIZE, 0x9e37)
                .iter()
                .map(|v| v.to_f32() / 16.0)
                .collect();
            let matrix = Affine::new(
                HostTensor::<bf16, m![W, QWidth]>::from_vec(
                    weights
                        .iter()
                        .map(|w| bf16::from_f32(*w))
                        .collect::<Vec<_>>(),
                )
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
                (),
            );
            let run = async |device: &mut Device, p: &Dense<W>, live: &Value, a: &Attended| {
                launch(dense_into, (device, p, a, live)).await.unwrap()
            };
            check(&mut *device, &matrix, run, &weights, 0.004).await;
        }
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        projects::<Hidden<2560>>(&mut device).await;
        projects::<Hidden<5120>>(&mut device).await;
        projects::<Hidden<6656>>(&mut device).await;
    }
}
