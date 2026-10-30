use crate::tensor::axis;
use furiosa_opt_std::prelude::*;

use crate::tensor::parallel::Chip;
use group::Group;

pub mod fp8;
pub(crate) mod group;
pub(crate) mod launch;
pub mod matrix;

pub(crate) type Tile = axis::Rows<128>;
pub(crate) type Cluster<Heads> = m![Heads / (Heads / 2)];
pub(crate) type Output<G, Heads, Head, Spread, Local> =
    DmTensor<bf16, Chip, Cluster<Heads>, Rows<G, Heads, Spread>, m![{ Local }, Tile % 8, Head]>;
pub(crate) type Rows<G, Heads, Spread> = m![
    Heads % (Heads / 2) / (Heads / 4),
    G,
    Tile / 8,
    Heads % (Heads / 4) / (Heads / 8),
    { Spread }
];
/// A projection's output slices in the order its reduction lands them; [`Output`] names the same
/// slices in the order its consumers read them.
pub(crate) type Reduced<G, Heads, Spread> = m![
    Heads % (Heads / 2) / (Heads / 4),
    Heads % (Heads / 4) / (Heads / 8),
    G,
    Tile / 8,
    { Spread }
];
/// The heads of a slice's run that each slice of [`Queries`] keeps.
pub(crate) type QueryLocal<G, Heads> = m![Heads % G];
/// The rest of a slice's run, which [`Queries`] spreads over the slices.
pub(crate) type QuerySpread<G, Heads> = m![Heads % (Heads / 8) / G];
/// The slices of [`Queries`].
pub(crate) type QuerySlices<G, Heads> = Rows<G, Heads, QuerySpread<G, Heads>>;
/// The projected queries of `Heads`, as their head norm and rotation read them.
pub(crate) type Queries<G, Heads, Head> =
    Output<G, Heads, Head, QuerySpread<G, Heads>, QueryLocal<G, Heads>>;
/// The heads of a slice's run that each slice of [`Pairs`] keeps: all of them.
pub(crate) type PairLocal<Heads> = m![Heads % (Heads / 8)];
/// The slices of [`Pairs`], copying each run `G::Copies` ways.
pub(crate) type PairSlices<G, Heads> = Rows<G, Heads, <G as Group>::Copies>;
/// The projected keys or values of `Heads`.
pub(crate) type Pairs<G, Heads, Head> =
    Output<G, Heads, Head, <G as Group>::Copies, PairLocal<Heads>>;

#[cfg(test)]
mod tests {
    use super::fp8::{HeadScales, HeadSquares, Layout, Stored};
    use super::group::{TilePair, Tiles};
    use super::matrix::Dense;
    use super::*;
    use crate::modules::attention::Projection;
    use crate::modules::generic::Either;
    use crate::modules::generic::schedule::{Fill, IntoStream, Stream};
    use crate::ops::attention::geometry::Hidden;
    use crate::ops::linear::Affine;
    use crate::ops::normalization::tests::random;
    use crate::ops::normalization::{
        Blocked, Scaled,
        span::{Owned, Span},
    };
    use crate::tensor::live::{self, Live};
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use core::task::Poll;
    use furiosa_opt_std::runtime::launch;

    axes![QueryHeads = 32, KvHeads = 8, Head = 128];

    type H = Hidden<4096>;
    type All = axis::Rows<1024>;
    type Input = HbmTensor<bf16, Chip, Blocked<H, All>>;
    type QueryRows = HbmTensor<bf16, Chip, m![All, QueryHeads, Head]>;
    type KeyRows = HbmTensor<bf16, Chip, m![All, KvHeads, Head]>;
    type Single = Tiles<1>;

    type Matrices = Projection<
        Affine<Dense<H, QueryHeads, Head>, ()>,
        Affine<(Dense<H, KvHeads, Head>, ()), ()>,
        Affine<(Dense<H, KvHeads, Head>, ()), ()>,
    >;
    type Quantized = Projection<
        Affine<super::fp8::Weights<H, QueryHeads, Head, HeadSquares<QueryHeads>>, ()>,
        Affine<
            (
                super::fp8::Weights<H, KvHeads, Head, HeadSquares<KvHeads>>,
                (),
            ),
            (),
        >,
        Affine<super::fp8::Weights<H, KvHeads, Head, HeadScales<KvHeads, Head>>, ()>,
    >;

    /// A period's projected rows without the factor a scaled form carries, which a norm cancels.
    trait Rows {
        type Rows;

        fn rows(self) -> Self::Rows;
    }

    impl<X, S> Rows for Scaled<X, S> {
        type Rows = X;

        fn rows(self) -> X {
            self.cancel().0
        }
    }

    impl<C: M, S: M, E: M> Rows for DmTensor<bf16, Chip, C, S, E> {
        type Rows = Self;

        fn rows(self) -> Self {
            self
        }
    }

    /// A launch's spans from its rows in HBM, one a poll, then its end.
    struct Spans<'i> {
        input: &'i Input,
        live: Live<(), All>,
        given: usize,
    }

    impl Stream for Spans<'_> {
        type Item = Live<Owned<H, Span>, All>;

        fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Self::Item>> {
            if self.given < self.live.count::<Span>() {
                let tile = self.live.tile::<Span>(self.given);
                self.given += 1;
                return Poll::Ready(Some(tile.with(self.input.fill(device, tile))));
            }
            Poll::Ready(None)
        }
    }

    /// Each period its upstream gives, its queries, keys and values stored at its rows.
    struct Store<'o, U> {
        upstream: U,
        outputs: &'o mut (QueryRows, KeyRows, KeyRows),
    }

    impl<U, QP, QS, KP, KS, VP, VS> Stream for Store<'_, U>
    where
        U: Stream<Item = Live<(Either<QP, QS>, Either<KP, KS>, Either<VP, VS>), All>>,
        QP: Rows<Rows = Queries<TilePair, QueryHeads, Head>>,
        QS: Rows<Rows = Queries<Single, QueryHeads, Head>>,
        KP: Rows<Rows = Pairs<TilePair, KvHeads, Head>>,
        KS: Rows<Rows = Pairs<Single, KvHeads, Head>>,
        VP: Rows<Rows = Pairs<TilePair, KvHeads, Head>>,
        VS: Rows<Rows = Pairs<Single, KvHeads, Head>>,
    {
        type Item = ();

        fn poll_next(&mut self, device: &mut Device) -> Poll<Option<()>> {
            let period = match self.upstream.poll_next(&mut *device) {
                Poll::Ready(Some(period)) => period,
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            };
            let (queries, keys, values) = &mut *self.outputs;
            let start = period.start();
            let (query, key, value) = period.into_value();
            match query {
                Either::Left(rows) => store(&mut *device, &rows.rows(), queries, start),
                Either::Right(rows) => store(&mut *device, &rows.rows(), queries, start),
            }
            match key {
                Either::Left(rows) => store(&mut *device, &rows.rows(), keys, start),
                Either::Right(rows) => store(&mut *device, &rows.rows(), keys, start),
            }
            match value {
                Either::Left(rows) => store(&mut *device, &rows.rows(), values, start),
                Either::Right(rows) => store(device, &rows.rows(), values, start),
            }
            Poll::Ready(Some(()))
        }
    }

    /// The projections streamed over a launch's spans, each period's queries, keys and values
    /// stored at its rows; `$live` binds the launch's live rows, a constant or the count `live`.
    macro_rules! project {
        ($name:ident, $projection:ty, |$count:ident| $live:expr) => {
            #[device(chip = 1)]
            fn $name(
                device: &mut Device,
                projection: &$projection,
                input: &Input,
                $count: &Value,
            ) -> (QueryRows, KeyRows, KeyRows) {
                let live: Live<(), All> = $live;
                let projection = Projection {
                    query: &projection.query,
                    key: &projection.key,
                    value: &projection.value,
                };
                let mut state = ((), (), ());
                let mut outputs = (QueryRows::new(), KeyRows::new(), KeyRows::new());
                {
                    let spans = Spans {
                        input,
                        live,
                        given: 0,
                    };
                    let mut root = Store {
                        upstream: projection.into_stream(live, &mut state, spans),
                        outputs: &mut outputs,
                    };
                    let mut done = false;
                    while !done {
                        done = matches!(root.poll_next(&mut *device), Poll::Ready(None));
                    }
                }
                outputs
            }
        };
    }

    project!(dense, Matrices, |live| Live::read(live));
    project!(quantized, Quantized, |live| Live::read(live));
    project!(dense_span, Matrices, |live| {
        Live::read(live).tile::<Span>(1)
    });
    project!(quantized_span, Quantized, |live| {
        Live::read(live).tile::<Span>(1)
    });

    /// Logical weights as both forms compute with them, `[out, in]` with `out` heads then their
    /// features: E4M3-representable query and key coefficients, and value coefficients times a
    /// power-of-two scale of each value head feature.
    struct Weights {
        codes: [Vec<f8e4m3>; 3],
        scales: Vec<f32>,
    }

    impl Weights {
        fn new(seed: u64) -> Self {
            let codes = |count, seed| -> Vec<f8e4m3> {
                random(count, seed)
                    .iter()
                    .map(|v| f8e4m3::from_f32(v.to_f32()))
                    .collect()
            };
            let (query, pair) = (QueryHeads::SIZE * Head::SIZE, KvHeads::SIZE * Head::SIZE);
            Self {
                codes: [
                    codes(query * H::SIZE, seed),
                    codes(pair * H::SIZE, seed ^ 0x9e37_79b9),
                    codes(pair * H::SIZE, seed ^ 0x7f4a_7c15),
                ],
                scales: (0..pair)
                    .map(|f| 1.0 / f32::from(1u16 << (2 + f % 5)))
                    .collect(),
            }
        }

        /// Projection `which`'s logical coefficients.
        fn logical(&self, which: usize) -> Vec<f32> {
            let codes = self.codes[which].iter().map(|c| c.to_f32());
            if which < 2 {
                return codes.collect();
            }
            codes
                .zip(self.scales.iter().flat_map(|&s| [s; H::SIZE]))
                .map(|(c, s)| c * s)
                .collect()
        }

        async fn dense(&self, device: &mut Device) -> Matrices {
            async fn upload<Heads: AxisName>(
                device: &mut Device,
                values: Vec<f32>,
            ) -> Dense<H, Heads, Head> {
                let values = values.into_iter().map(bf16::from_f32).collect::<Vec<_>>();
                HostTensor::<bf16, m![Heads, Head, H]>::from_vec(values)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap()
            }
            Projection {
                query: Affine::new(upload::<QueryHeads>(device, self.logical(0)).await, ()),
                key: Affine::new((upload::<KvHeads>(device, self.logical(1)).await, ()), ()),
                value: Affine::new((upload::<KvHeads>(device, self.logical(2)).await, ()), ()),
            }
        }

        async fn quantized(&self, device: &mut Device) -> Quantized {
            async fn upload<Heads: AxisName>(
                device: &mut Device,
                codes: &[f8e4m3],
            ) -> Stored<H, Heads, Head> {
                HostTensor::<f8e4m3, m![Heads, Head, H]>::from_vec(codes.to_vec())
                    .to_hbm::<Chip, Layout<H, Heads, Head>>(&mut device.pdma)
                    .await
                    .unwrap()
            }
            let scales = HostTensor::<f32, m![KvHeads, Head]>::from_vec(self.scales.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            // Codes that hold their scales: each head's squared scale is one.
            async fn ones<Heads: AxisName>(device: &mut Device) -> HeadSquares<Heads> {
                let ones = vec![1.0; Heads::SIZE * Tile::SIZE];
                crate::tensor::form::fp8::Squares(
                    HostTensor::<f32, m![Heads, Tile]>::from_vec(ones)
                        .to_hbm(&mut device.pdma)
                        .await
                        .unwrap(),
                )
            }
            Projection {
                query: Affine::new(
                    crate::tensor::form::fp8::Rows::new(
                        upload::<QueryHeads>(device, &self.codes[0]).await,
                        ones::<QueryHeads>(device).await,
                    ),
                    (),
                ),
                key: Affine::new(
                    (
                        crate::tensor::form::fp8::Rows::new(
                            upload::<KvHeads>(device, &self.codes[1]).await,
                            ones::<KvHeads>(device).await,
                        ),
                        (),
                    ),
                    (),
                ),
                value: Affine::new(
                    crate::tensor::form::fp8::Rows::new(
                        upload::<KvHeads>(device, &self.codes[2]).await,
                        scales,
                    ),
                    (),
                ),
            }
        }
    }

    /// A period's output rows from row `start` into the launch's rows in HBM, `Local` of a
    /// slice's run of heads in each slice and the rest spread over `Spread`.
    fn store<G: Group, Heads: AxisName, Spread: M, Local: M, All: AxisName>(
        device: &mut Device,
        rows: &Output<G, Heads, Head, Spread, Local>,
        into: &mut HbmTensor<bf16, Chip, m![All, Heads, Head]>,
        start: usize,
    ) {
        // SAFETY: every form names Reduced's slices with the output's slice names.
        let rows = unsafe {
            rows.view().reshape::<Chip, Cluster<Heads>, Reduced<G, Heads, Spread>,
                m![{ Local }, Tile % 8, Head]>()
        };
        rows.to_hbm_view(&mut device.tdma, G::window(into.view_mut(), start));
    }

    /// `out` of each row `x` over `weights`, in FP64.
    fn reference(x: &[f32], weights: &[f32]) -> Vec<f64> {
        weights
            .chunks_exact(x.len())
            .map(|w| {
                w.iter()
                    .zip(x)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum()
            })
            .collect()
    }

    /// The RMS of `got - expected` relative to that of `expected`; each head's rows first scaled
    /// to unit RMS when `per_head`, so a row's positive factor cancels.
    fn error(got: &[f64], expected: &[f64], per_head: bool) -> f64 {
        let unit = |rows: &[f64]| -> Vec<f64> {
            rows.chunks_exact(Head::SIZE)
                .flat_map(|head| {
                    let rms = (head.iter().map(|v| v * v).sum::<f64>() / Head::SIZE as f64).sqrt();
                    head.iter()
                        .map(move |v| if per_head { v / rms } else { *v })
                })
                .collect()
        };
        let (got, expected) = (unit(got), unit(expected));
        let (difference, norm) = got
            .iter()
            .zip(&expected)
            .fold((0.0, 0.0), |(d, n), (g, e)| {
                (d + (g - e) * (g - e), n + e * e)
            });
        (difference / norm).sqrt()
    }

    /// One projection fixture over both weight forms of the same logical weights, one kernel each
    /// for every count of live rows it reads at run time: at one row, at a pair of tiles and a
    /// partial last tile, and at two pairs over two spans, each live row's queries and keys match
    /// the dense reference up to each row's factor and its values exactly as stored, within the
    /// form's `tolerance` of the rows' RMS, and rows outside the live ones never reach them. The
    /// live rows start at row `first`: 0, or a later span's first, that span launched alone.
    async fn check<P: DeviceSend>(
        device: &mut Device,
        projection: &P,
        run: impl AsyncFn(&mut Device, &P, &Input, &Value) -> (QueryRows, KeyRows, KeyRows),
        weights: &Weights,
        (tolerance, first): (f64, usize),
    ) {
        let hidden = H::SIZE;
        let values = random(All::SIZE * hidden, 0x2545_f491_4f6c_dd1d);
        let rows: Vec<f32> = values.iter().map(|v| v.to_f32()).collect();
        let logical = [0, 1, 2].map(|which| weights.logical(which));
        let outputs = async |device: &mut Device, live: usize, padding: f32| {
            let mut padded = values.clone();
            padded[..first * hidden].fill(bf16::from_f32(padding));
            padded[live * hidden..].fill(bf16::from_f32(padding));
            let input = HostTensor::<bf16, m![All, H]>::from_vec(padded)
                .to_hbm::<Chip, Blocked<H, All>>(&mut device.pdma)
                .await
                .unwrap();
            let count = live::tests::count(live)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let (queries, keys, values) = run(&mut *device, projection, &input, &count).await;
            let host = async |device: &mut Device, rows: KeyRows| {
                rows.to_host::<m![All, KvHeads, Head]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec()
            };
            let queries = queries
                .to_host::<m![All, QueryHeads, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            [
                queries,
                host(&mut *device, keys).await,
                host(&mut *device, values).await,
            ]
        };
        for count in [1, 300, 600] {
            let live = (first + count).min(All::SIZE);
            let got = outputs(&mut *device, live, 0.0).await;
            let padded = outputs(&mut *device, live, 1000.0).await;
            for (which, (got, padded)) in got.iter().zip(&padded).enumerate() {
                let width = got.len() / All::SIZE;
                assert_eq!(
                    got[first * width..live * width],
                    padded[first * width..live * width],
                    "rows {first}..{live}, projection {which}: padding reached live rows"
                );
            }
            for row in [0, count / 2, count - 1, 127, 128, 255, 256, 511, 512, 599] {
                let row = first + row;
                if row >= live {
                    continue;
                }
                let x = &rows[row * hidden..][..hidden];
                for (which, got) in got.iter().enumerate() {
                    let width = got.len() / All::SIZE;
                    let got: Vec<f64> = got[row * width..][..width]
                        .iter()
                        .map(|v| f64::from(v.to_f32()))
                        .collect();
                    let error = error(&got, &reference(x, &logical[which]), which < 2);
                    assert!(
                        error <= tolerance,
                        "rows {first}..{live}, row {row}, projection {which}: relative error {error}"
                    );
                }
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn projects_each_row_in_both_forms() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let weights = Weights::new(0x3c6e_f372_fe94_f82b);
        let matrices = weights.dense(&mut device).await;
        let run = async |device: &mut Device, p: &Matrices, input: &Input, live: &Value| {
            launch(dense, (device, p, input, live)).await.unwrap()
        };
        check(&mut device, &matrices, run, &weights, (0.004, 0)).await;
        let run = async |device: &mut Device, p: &Matrices, input: &Input, live: &Value| {
            launch(dense_span, (device, p, input, live)).await.unwrap()
        };
        check(&mut device, &matrices, run, &weights, (0.004, Span::SIZE)).await;
        let quantized_weights = weights.quantized(&mut device).await;
        let run = async |device: &mut Device, p: &Quantized, input: &Input, live: &Value| {
            launch(quantized, (device, p, input, live)).await.unwrap()
        };
        check(&mut device, &quantized_weights, run, &weights, (0.04, 0)).await;
        let run = async |device: &mut Device, p: &Quantized, input: &Input, live: &Value| {
            launch(quantized_span, (device, p, input, live))
                .await
                .unwrap()
        };
        check(
            &mut device,
            &quantized_weights,
            run,
            &weights,
            (0.04, Span::SIZE),
        )
        .await;
    }
}
