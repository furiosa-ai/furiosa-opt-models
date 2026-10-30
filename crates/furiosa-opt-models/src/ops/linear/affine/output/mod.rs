//! The attention's output projection over a launch's attended rows: one node stages its weights,
//! takes the attended rows once and gives their projection a span at a time; the weights' form
//! projects each span, in as many polls as it takes.

use furiosa_opt_std::prelude::*;

use crate::Module;
use crate::modules::generic::schedule::{Begin, IntoStream, Slot, Stream};
use crate::ops::linear::Affine;
use crate::ops::normalization::span::{Owned, Span};
use crate::ops::pointwise::add::{Add, Term};
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use core::borrow::Borrow;
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

/// The projection of a launch's attended rows, which its upstream gives once, a span at a time.
impl<U, E, S, W, L, A, O, Capacity> IntoStream<U, E, S> for Affine<W, ()>
where
    Capacity: AxisName,
    E: Into<Live<(), Capacity>>,
    U: Stream<Item = Live<A, Capacity>>,
    Self: for<'s> Module<Begin<()>, (), &'s mut (), Output = L>
        + for<'s, 'l> Module<&'l mut L, (), &'s mut (), Output = ()>
        + for<'s, 'l, 'a> Module<
            (&'l mut L, &'a A, Live<(), Capacity>),
            (),
            &'s mut (),
            Output = Poll<O>,
        >,
{
    type Stream = Node<Self, Self, L, A, Capacity, U>;

    fn into_stream(self, env: E, _: S, upstream: U) -> Self::Stream {
        Node::new(self, env.into(), upstream)
    }
}

/// A borrowed projection streams as its reference.
impl<'m, U, E, S, W, L, A, O, Capacity> IntoStream<U, E, S> for &'m Affine<W, ()>
where
    Capacity: AxisName,
    E: Into<Live<(), Capacity>>,
    U: Stream<Item = Live<A, Capacity>>,
    Affine<W, ()>: for<'s> Module<Begin<()>, (), &'s mut (), Output = L>
        + for<'s, 'l> Module<&'l mut L, (), &'s mut (), Output = ()>
        + for<'s, 'l, 'a> Module<
            (&'l mut L, &'a A, Live<(), Capacity>),
            (),
            &'s mut (),
            Output = Poll<O>,
        >,
{
    type Stream = Node<Self, Affine<W, ()>, L, A, Capacity, U>;

    fn into_stream(self, env: E, _: S, upstream: U) -> Self::Stream {
        Node::new(self, env.into(), upstream)
    }
}

/// An output projection's node: its projection `T`, held as `M`, what its weights landed at its
/// first poll, the launch's live rows, its attended rows, the spans it gave and its upstream. It
/// polls its upstream for the attended rows, then for its end once it gave every span.
pub(crate) struct Node<M, T, L, A, Capacity, U> {
    module: M,
    landed: Option<L>,
    live: Live<(), Capacity>,
    attended: Option<A>,
    given: usize,
    upstream: U,
    projection: PhantomData<T>,
}

impl<M, T, L, A, Capacity, U> Node<M, T, L, A, Capacity, U> {
    const fn new(module: M, live: Live<(), Capacity>, upstream: U) -> Self {
        Self {
            module,
            landed: None,
            live,
            attended: None,
            given: 0,
            upstream,
            projection: PhantomData,
        }
    }
}

impl<X, M, T, L, A, Capacity, U: AsMut<Slot<X>>> AsMut<Slot<X>> for Node<M, T, L, A, Capacity, U> {
    fn as_mut(&mut self) -> &mut Slot<X> {
        self.upstream.as_mut()
    }
}

impl<M, W, L, A, O, Capacity, U> Stream for Node<M, Affine<W, ()>, L, A, Capacity, U>
where
    Capacity: AxisName,
    M: Borrow<Affine<W, ()>>,
    U: Stream<Item = Live<A, Capacity>>,
    Affine<W, ()>: for<'s> Module<Begin<()>, (), &'s mut (), Output = L>
        + for<'s, 'l> Module<&'l mut L, (), &'s mut (), Output = ()>
        + for<'s, 'l, 'a> Module<
            (&'l mut L, &'a A, Live<(), Capacity>),
            (),
            &'s mut (),
            Output = Poll<O>,
        >,
{
    type Item = Live<O, Capacity>;

    fn poll_next(&mut self, context: &mut Device) -> Poll<Option<Self::Item>> {
        let projection = self.module.borrow();
        if matches!(self.attended, None) {
            let polled = self.upstream.poll_next(&mut *context);
            if matches!(self.landed, None) {
                let mut landed = projection.forward(&mut *context, (), &mut (), Begin(()));
                projection.forward(&mut *context, (), &mut (), &mut landed);
                self.landed = Some(landed);
            }
            match polled {
                Poll::Ready(Some(attended)) => self.attended = Some(attended.into_value()),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
        if self.given >= self.live.count::<Span>() {
            return match self.upstream.poll_next(context) {
                Poll::Ready(None) => Poll::Ready(None),
                _ => Poll::Pending,
            };
        }
        let (Some(landed), Some(attended)) = (&mut self.landed, &self.attended) else {
            return Poll::Pending;
        };
        let tile = self.live.tile::<Span>(self.given);
        match projection.forward(&mut *context, (), &mut (), (landed, attended, tile)) {
            Poll::Ready(rows) => {
                self.given += 1;
                Poll::Ready(Some(tile.with(rows)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fp8::{Blocked, Stored};
    use super::*;
    use crate::modules::generic::schedule::Spill;
    use crate::ops::attention::geometry::{Hidden, QueryGroups, QueryWidth};
    use crate::ops::normalization::Blocked as Rows;
    use crate::ops::normalization::tests::random;
    use crate::tensor::axis;
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

    /// The launch's attended rows, given once, then its end.
    struct Given(Option<Live<Attended, All>>);

    impl Stream for Given {
        type Item = Live<Attended, All>;

        fn poll_next(&mut self, _: &mut Device) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.0.take())
        }
    }

    /// Each span its upstream gives, stored at its rows.
    struct Store<'o, H: AxisName, U> {
        upstream: U,
        stored: &'o mut Projected<H>,
    }

    impl<H: AxisName, U: Stream<Item = Live<Owned<H, Span>, All>>> Stream for Store<'_, H, U> {
        type Item = ();

        fn poll_next(&mut self, device: &mut Device) -> Poll<Option<()>> {
            match self.upstream.poll_next(&mut *device) {
                Poll::Ready(Some(rows)) => {
                    self.stored.spill(device, rows.value(), rows.with(()));
                    Poll::Ready(Some(()))
                }
                Poll::Ready(None) => Poll::Ready(None),
                Poll::Pending => Poll::Pending,
            }
        }
    }

    /// The projection streamed over a launch's attended rows, given once, each span stored at its
    /// rows.
    macro_rules! project {
        ($name:ident, $projection:ty, $live:literal) => {
            #[device(chip = 1)]
            fn $name(
                device: &mut Device,
                projection: &$projection,
                attended: &Attended,
            ) -> Projected {
                let live = Live::<(), All>::all().subrange(0, $live);
                let mut stored = Projected::new();
                {
                    // SAFETY: the stream reads the attended rows only while they exist.
                    let given = Given(Some(live.with(unsafe { attended.view().alias() })));
                    let mut root = Store {
                        upstream: projection.into_stream(live, (), given),
                        stored: &mut stored,
                    };
                    let mut done = false;
                    while !done {
                        done = matches!(root.poll_next(&mut *device), Poll::Ready(None));
                    }
                }
                stored
            }
        };
    }

    project!(dense_one, Dense, 1);
    project!(dense_many, Dense, 600);
    project!(quantized_one, Stored<H, QWidth>, 1);
    project!(quantized_many, Stored<H, QWidth>, 600);

    /// The dense projection into `W` hidden features streamed over a launch's attended rows,
    /// given once, each span stored at its rows.
    macro_rules! project_into {
        ($name:ident, $live:literal) => {
            #[device(chip = 1)]
            fn $name<W: AxisName>(
                device: &mut Device,
                projection: &Dense<W>,
                attended: &Attended,
            ) -> Projected<W>
            where
                for<'p> &'p Dense<W>: IntoStream<
                        Given,
                        Live<(), All>,
                        (),
                        Stream: Stream<Item = Live<Owned<W, Span>, All>>,
                    >,
            {
                let live = Live::<(), All>::all().subrange(0, $live);
                let mut stored = Projected::<W>::new();
                {
                    // SAFETY: the stream reads the attended rows only while they exist.
                    let given = Given(Some(live.with(unsafe { attended.view().alias() })));
                    let mut root = Store {
                        upstream: projection.into_stream(live, (), given),
                        stored: &mut stored,
                    };
                    let mut done = false;
                    while !done {
                        done = matches!(root.poll_next(&mut *device), Poll::Ready(None));
                    }
                }
                stored
            }
        };
    }

    project_into!(dense_into_one, 1);
    project_into!(dense_into_many, 600);

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
        run: impl AsyncFn(&mut Device, &P, usize, &Attended) -> Projected<H>,
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
            run(&mut *device, projection, live, &attended)
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
        let run =
            async |device: &mut Device, p: &Dense, live: usize, attended: &Attended| match live {
                1 => launch(dense_one, (device, p, attended)).await.unwrap(),
                _ => launch(dense_many, (device, p, attended)).await.unwrap(),
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
        let run = async |device: &mut Device, p: &Stored<H, QWidth>, live: usize, a: &Attended| {
            match live {
                1 => launch(quantized_one, (device, p, a)).await.unwrap(),
                _ => launch(quantized_many, (device, p, a)).await.unwrap(),
            }
        };
        check(&mut device, &quantized_weights, run, &weights, 0.04).await;
    }

    /// The dense projection into each listed hidden width, padded feature slices included,
    /// matches the reference on the live rows of two spans.
    #[tokio::test(flavor = "current_thread")]
    async fn projects_into_every_width() {
        async fn projects<W: AxisName>(device: &mut Device)
        where
            for<'p> &'p Dense<W>: IntoStream<
                    Given,
                    Live<(), All>,
                    (),
                    Stream: Stream<Item = Live<Owned<W, Span>, All>>,
                >,
        {
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
            let run =
                async |device: &mut Device, p: &Dense<W>, live: usize, a: &Attended| match live {
                    1 => launch(dense_into_one, (device, p, a)).await.unwrap(),
                    _ => launch(dense_into_many, (device, p, a)).await.unwrap(),
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
