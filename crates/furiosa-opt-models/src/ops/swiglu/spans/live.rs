use crate::Module;
use crate::modules::generic::schedule::{Begin, IntoStream, Slot, Stream};
use crate::ops::normalization::span::Owned;
use crate::ops::pointwise::SwiGlu;
use crate::ops::swiglu::{Down, Mlp};
use crate::tensor::live::Live;
use core::borrow::Borrow;
use core::marker::PhantomData;
use core::task::Poll;
use furiosa_opt_std::prelude::*;

/// A span's MLP on each normalized tile its upstream gives, whatever form its weights take.
impl<U, E, S, H: AxisName, I: AxisName, F, D, Rows: AxisName, Capacity: AxisName, A, G, P, C>
    IntoStream<U, E, S> for Mlp<H, I, F, D>
where
    Down<I>: AxisName,
    E: Into<Live<(), Capacity>>,
    U: Stream<Item = Live<Owned<H, Rows>, Capacity>>,
    F: for<'s> Module<Begin<()>, (), &'s mut (), Output = A>,
    F: for<'s, 't> Module<(A, Live<&'t Owned<H, Rows>, Capacity>), (), &'s mut (), Output = G>,
    SwiGlu: for<'s> Module<Live<G, Capacity>, (), &'s mut (), Output = P>,
    D: for<'s> Module<Begin<Live<P, Capacity>>, (), &'s mut (), Output = C>,
    Node<Self, Self, A, C, Capacity, U>: Stream,
{
    type Stream = Node<Self, Self, A, C, Capacity, U>;

    fn into_stream(self, env: E, _: S, upstream: U) -> Self::Stream {
        Node::new(self, env.into(), upstream)
    }
}

/// A borrowed MLP streams as its reference.
impl<'m, U, E, S, H, I, F, D, Rows, Capacity, A, G, P, C> IntoStream<U, E, S>
    for &'m Mlp<H, I, F, D>
where
    H: AxisName,
    I: AxisName,
    Rows: AxisName,
    Capacity: AxisName,
    Down<I>: AxisName,
    E: Into<Live<(), Capacity>>,
    U: Stream<Item = Live<Owned<H, Rows>, Capacity>>,
    F: for<'s> Module<Begin<()>, (), &'s mut (), Output = A>,
    F: for<'s, 't> Module<(A, Live<&'t Owned<H, Rows>, Capacity>), (), &'s mut (), Output = G>,
    SwiGlu: for<'s> Module<Live<G, Capacity>, (), &'s mut (), Output = P>,
    D: for<'s> Module<Begin<Live<P, Capacity>>, (), &'s mut (), Output = C>,
    Node<Self, Mlp<H, I, F, D>, A, C, Capacity, U>: Stream,
{
    type Stream = Node<Self, Mlp<H, I, F, D>, A, C, Capacity, U>;

    fn into_stream(self, env: E, _: S, upstream: U) -> Self::Stream {
        Node::new(self, env.into(), upstream)
    }
}

/// A span MLP's node: its MLP `T`, held as `M`, the launch's live rows, the tiles it gave, the
/// step of the tile in progress, what the gate/up weights land before a tile (`A`), what the down
/// projection keeps of a tile's product between its two steps (`C`), and its upstream. A tile takes
/// two steps, one a poll: the gate/up projections, their SwiGLU and the product's staging, then the
/// down projection, whose output it gives. It polls its upstream only between tiles; its first poll
/// after a tile lands the next tile's first gate/up weights unless the upstream ended, so they land
/// while its upstream is busy.
pub(crate) struct Node<M, T, A, C, Capacity, U> {
    module: M,
    live: Live<(), Capacity>,
    given: usize,
    step: usize,
    ahead: Option<A>,
    product: Option<C>,
    upstream: U,
    mlp: PhantomData<T>,
}

impl<M, T, A, C, Capacity, U> Node<M, T, A, C, Capacity, U> {
    const fn new(module: M, live: Live<(), Capacity>, upstream: U) -> Self {
        Self {
            module,
            live,
            given: 0,
            step: 0,
            ahead: None,
            product: None,
            upstream,
            mlp: PhantomData,
        }
    }
}

impl<X, M, T, A, C, Capacity, U: AsMut<Slot<X>>> AsMut<Slot<X>> for Node<M, T, A, C, Capacity, U> {
    fn as_mut(&mut self) -> &mut Slot<X> {
        self.upstream.as_mut()
    }
}

impl<M, H, I, F, D, Rows, Capacity, A, G, P, C, U> Stream
    for Node<M, Mlp<H, I, F, D>, A, C, Capacity, U>
where
    H: AxisName,
    I: AxisName,
    Rows: AxisName,
    Capacity: AxisName,
    Down<I>: AxisName,
    M: Borrow<Mlp<H, I, F, D>>,
    U: Stream<Item = Live<Owned<H, Rows>, Capacity>>,
    F: for<'s> Module<Begin<()>, (), &'s mut (), Output = A>,
    F: for<'s, 't> Module<(A, Live<&'t Owned<H, Rows>, Capacity>), (), &'s mut (), Output = G>,
    SwiGlu: for<'s> Module<Live<G, Capacity>, (), &'s mut (), Output = P>,
    D: for<'s> Module<Begin<Live<P, Capacity>>, (), &'s mut (), Output = C>,
    D: for<'s> Module<Live<C, Capacity>, (), &'s mut (), Output = Owned<H, Rows>>,
{
    type Item = Live<Owned<H, Rows>, Capacity>;

    fn poll_next(&mut self, context: &mut Device) -> Poll<Option<Self::Item>> {
        let mlp = self.module.borrow();
        let tile = self.live.tile::<Rows>(self.given);
        if self.step < 1 {
            let polled = self.upstream.poll_next(&mut *context);
            // Not by the tiles still to come: a value set at a poll the device picks at run time is
            // a selection it cannot make.
            let lands = matches!(
                (&self.ahead, &polled),
                (None, Poll::Pending | Poll::Ready(Some(_)))
            );
            if lands {
                self.ahead = Some(mlp.gate_up.forward(&mut *context, (), &mut (), Begin(())));
            }
            let normalized = match polled {
                Poll::Ready(Some(tile)) => tile.into_value(),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            };
            self.step = 1;
            if let Some(ahead) = self.ahead.take() {
                let projected = mlp.gate_up.forward(
                    &mut *context,
                    (),
                    &mut (),
                    (ahead, tile.with(&normalized)),
                );
                let product = SwiGlu.forward(&mut *context, (), &mut (), tile.with(projected));
                self.product =
                    Some(
                        mlp.down
                            .forward(&mut *context, (), &mut (), Begin(tile.with(product))),
                    );
            }
            return Poll::Pending;
        }
        self.step = 0;
        self.given += 1;
        let mut output = Owned::<H, Rows>::new();
        if let Some(product) = self.product.take() {
            output = mlp
                .down
                .forward(&mut *context, (), &mut (), tile.with(product));
        }
        Poll::Ready(Some(tile.with(output)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::generic::schedule::{Fill, Spill};
    use crate::ops::attention::geometry::Hidden;
    use crate::ops::linear::Linear;
    use crate::ops::normalization::tests::random;
    use crate::ops::normalization::{Blocked, span::Span};
    use crate::ops::swiglu::spans::Term;
    use crate::ops::swiglu::spans::fp8::{self, QuantizedMlp};
    use crate::ops::swiglu::{GateUp, Intermediate};
    use crate::tensor::axis;
    use crate::tensor::form::Weight;
    use crate::tensor::live::tests::count;
    use crate::tensor::parallel::Chip;
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    type Rows<H> = HbmTensor<bf16, Chip, Blocked<H, Span>>;
    type Tile<H> = Live<Owned<H, Span>, Span>;
    type Linears<H, I> = Mlp<H, I, Linear<H, GateUp<I>, Chip>, Linear<I, H, Chip>>;

    const SEEDS: (u64, u64, u64) = (
        0x9e37_79b9_7f4a_7c15,
        0x0123_4567_89ab_cdef,
        0xd1b5_4a32_d192_ed03,
    );

    /// The span's rows through the MLP's stages one after another.
    #[device(chip = 1)]
    fn sequential<H: AxisName, I: AxisName, F: DeviceSend, D: DeviceSend>(
        device: &mut Device,
        mlp: &Mlp<H, I, F, D>,
        rows: &Value,
        input: &Rows<H>,
    ) -> Rows<H>
    where
        Down<I>: AxisName,
        Mlp<H, I, F, D>: for<'x, 's> Module<&'x Tile<H>, (), &'s mut (), Output = Tile<H>>,
    {
        let live = Live::<(), Span>::read(rows);
        let tile = live.with(input.fill(&mut *device, live));
        let output = mlp.forward(&mut *device, (), &mut (), &tile);
        let mut stored = Rows::<H>::new();
        stored.spill(&mut *device, output.value(), live);
        stored
    }

    /// The span's rows through the MLP's stream: its first poll lands the first gate/up weights and
    /// runs the gate/up half, its second the down half.
    #[device(chip = 1)]
    fn streamed<H: AxisName, I: AxisName, F: DeviceSend, D: DeviceSend>(
        device: &mut Device,
        mlp: &Mlp<H, I, F, D>,
        rows: &Value,
        input: &Rows<H>,
    ) -> Rows<H>
    where
        Down<I>: AxisName,
        for<'m> &'m Mlp<H, I, F, D>:
            IntoStream<Slot<Tile<H>>, Live<(), Span>, (), Stream: Stream<Item = Tile<H>>>,
    {
        let live = Live::<(), Span>::read(rows);
        let mut slot = Slot::new();
        slot.put(Poll::Ready(Some(live.with(input.fill(&mut *device, live)))));
        let mut node = mlp.into_stream(live, (), slot);
        let _ = node.poll_next(&mut *device);
        let mut stored = Rows::<H>::new();
        if let Poll::Ready(Some(output)) = node.poll_next(&mut *device) {
            stored.spill(&mut *device, output.value(), live);
        }
        stored
    }

    /// Two spans of rows: a window of two live layers polls them over each other.
    type Spans = axis::Rows<1024>;
    type Both<H> = HbmTensor<bf16, Chip, Blocked<H, Spans>>;
    type Window<H> = Live<Owned<H, Span>, Spans>;

    /// Two spans through two layers polled as a window of two live layers: each layer's steps run
    /// while the other holds what it keeps between polls, its staged product or its next tile's
    /// first gate/up weights, so both layers' data memory must fit together.
    #[device(chip = 1)]
    fn window<H: AxisName, I: AxisName, F: DeviceSend, D: DeviceSend>(
        device: &mut Device,
        (first, second): (&Mlp<H, I, F, D>, &Mlp<H, I, F, D>),
        rows: &Value,
        input: &Both<H>,
    ) -> Both<H>
    where
        Down<I>: AxisName,
        for<'m> &'m Mlp<H, I, F, D>: IntoStream<
                Slot<Window<H>>,
                Live<(), Spans>,
                (),
                Stream: Stream<Item = Window<H>> + AsMut<Slot<Window<H>>>,
            >,
    {
        let live = Live::<(), Spans>::read(rows);
        let tiles = [live.tile::<Span>(0), live.tile::<Span>(1)];
        let mut stored = Both::<H>::new();
        let (mut lower, mut upper) = (
            first.into_stream(live, (), Slot::new()),
            second.into_stream(live, (), Slot::new()),
        );
        let given = Poll::Ready(Some(tiles[0].with(input.fill(&mut *device, tiles[0]))));
        lower.as_mut().put(given);
        let _ = lower.poll_next(&mut *device);
        let output = lower.poll_next(&mut *device);
        upper.as_mut().put(output);
        let given = Poll::Ready(Some(tiles[1].with(input.fill(&mut *device, tiles[1]))));
        lower.as_mut().put(given);
        let _ = lower.poll_next(&mut *device);
        let _ = upper.poll_next(&mut *device);
        let output = lower.poll_next(&mut *device);
        if let Poll::Ready(Some(rows)) = upper.poll_next(&mut *device) {
            stored.spill(&mut *device, rows.value(), tiles[0]);
        }
        upper.as_mut().put(output);
        let _ = lower.poll_next(&mut *device);
        let _ = upper.poll_next(&mut *device);
        if let Poll::Ready(Some(rows)) = upper.poll_next(&mut *device) {
            stored.spill(&mut *device, rows.value(), tiles[1]);
        }
        stored
    }

    /// Logical weights as the MLP computes with them: gate and up `[I, H]`, down `[H, I]`, each
    /// coefficient its stored value times its channel scale.
    struct Weights {
        hidden: usize,
        width: usize,
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
    }

    impl Weights {
        /// Pseudo-random E4M3-representable coefficients of `hidden` inputs and `width`
        /// intermediate features for seeds `(gate, up, down)`.
        fn new(
            (hidden, width): (usize, usize),
            (gate, up, down): (u64, u64, u64),
        ) -> (Self, [Vec<f8e4m3>; 3]) {
            let coefficients = |count, seed| -> Vec<f8e4m3> {
                random(count, seed)
                    .iter()
                    .map(|v| f8e4m3::from_f32(v.to_f32()))
                    .collect()
            };
            let stored = [
                coefficients(width * hidden, gate),
                coefficients(width * hidden, up),
                coefficients(hidden * width, down),
            ];
            let scaled = |values: &[f8e4m3], scale: f32| {
                values
                    .iter()
                    .map(|v| v.to_f32() * scale)
                    .collect::<Vec<f32>>()
            };
            let weights = Self {
                hidden,
                width,
                gate: scaled(&stored[0], Self::GATE),
                up: scaled(&stored[1], Self::GATE),
                down: scaled(&stored[2], Self::DOWN),
            };
            (weights, stored)
        }

        const GATE: f32 = 1.0 / 32.0;
        const DOWN: f32 = 1.0 / 64.0;

        /// `down(bf16(bf16(silu(gate x)) * up x))` of one row, gate and up rounded to BF16.
        fn reference(&self, row: &[f32]) -> Vec<f64> {
            let rounded = |x: f64| f64::from(bf16::from_f32(x as f32).to_f32());
            let dot = |a: &[f32], b: &[f32]| -> f64 {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum()
            };
            let (hidden, width) = (self.hidden, self.width);
            let product: Vec<f32> = (0..width)
                .map(|j| {
                    let gate = rounded(dot(row, &self.gate[j * hidden..][..hidden]));
                    let up = rounded(dot(row, &self.up[j * hidden..][..hidden]));
                    rounded(rounded(gate / (1.0 + (-gate).exp())) * up) as f32
                })
                .collect();
            (0..hidden)
                .map(|o| dot(&product, &self.down[o * width..][..width]))
                .collect()
        }
    }

    /// The E4M3 MLP of `stored` coefficients, every channel scale the same.
    async fn quantized<H: AxisName, I: AxisName>(
        device: &mut Device,
        stored: &[Vec<f8e4m3>; 3],
    ) -> QuantizedMlp<H, I>
    where
        Down<I>: AxisName,
    {
        let packed = [stored[0].clone(), stored[1].clone()].concat();
        let gate_up = HostTensor::<f8e4m3, m![Term, I, H]>::from_vec(packed)
            .to_hbm::<Chip, fp8::Blocked<H, I>>(&mut device.pdma)
            .await
            .unwrap();
        let channels = HostTensor::<f32, m![Term, I]>::from_vec(vec![Weights::GATE; 2 * I::SIZE])
            .to_hbm::<Chip, m![Term, I]>(&mut device.pdma)
            .await
            .unwrap();
        let down = HostTensor::<f8e4m3, m![H, I]>::from_vec(stored[2].clone())
            .to_hbm::<Chip, fp8::Stored<H, I>>(&mut device.pdma)
            .await
            .unwrap();
        let scales = HostTensor::<f32, m![H]>::from_vec(vec![Weights::DOWN; H::SIZE])
            .to_hbm::<Chip, m![H]>(&mut device.pdma)
            .await
            .unwrap();
        Mlp::new((gate_up, channels), (down, scales))
    }

    /// The BF16 MLP of the same logical weights.
    async fn linears<H: AxisName, I: AxisName>(
        device: &mut Device,
        weights: &Weights,
    ) -> Linears<H, I>
    where
        Down<I>: AxisName,
    {
        let bf16s = |values: &[f32]| {
            values
                .iter()
                .map(|v| bf16::from_f32(*v))
                .collect::<Vec<_>>()
        };
        let packed = [bf16s(&weights.gate), bf16s(&weights.up)].concat();
        let gate_up = HostTensor::<bf16, m![{ Symbol<GateUp<I>> }, H]>::from_vec(packed)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let down = HostTensor::<bf16, m![H, I]>::from_vec(bf16s(&weights.down))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        Mlp::new(
            Linear::new(Weight::new(gate_up)),
            Linear::new(Weight::new(down)),
        )
    }

    /// One MLP over the same span in both executions, at each live row count: the stream gives
    /// the sequential rows exactly, the live rows match the reference within `tolerance` of each
    /// row's RMS, an all-zero row stays zero at the floor of its row scales, two layers polled as
    /// a window give the layers one after another, and changing the gate, up or down weights
    /// alone changes the rows.
    async fn check<H: AxisName, I: AxisName, F: DeviceSend, D: DeviceSend>(
        device: &mut Device,
        mlp: &Mlp<H, I, F, D>,
        weights: &Weights,
        changed: [&Mlp<H, I, F, D>; 3],
        tolerance: f64,
    ) where
        Down<I>: AxisName,
        Mlp<H, I, F, D>: for<'x, 's> Module<&'x Tile<H>, (), &'s mut (), Output = Tile<H>>,
        for<'m> &'m Mlp<H, I, F, D>:
            IntoStream<Slot<Tile<H>>, Live<(), Span>, (), Stream: Stream<Item = Tile<H>>>,
        for<'m> &'m Mlp<H, I, F, D>: IntoStream<
                Slot<Window<H>>,
                Live<(), Spans>,
                (),
                Stream: Stream<Item = Window<H>> + AsMut<Slot<Window<H>>>,
            >,
    {
        let hidden = H::SIZE;
        let mut values = random(Span::SIZE * hidden, 0x2545_f491_4f6c_dd1d);
        let zero = 100;
        values[zero * hidden..][..hidden].fill(bf16::from_f32(0.0));
        let rows: Vec<f32> = values.iter().map(|v| v.to_f32()).collect();
        let input = HostTensor::<bf16, m![Span, H]>::from_vec(values)
            .to_hbm::<Chip, Blocked<H, Span>>(&mut device.pdma)
            .await
            .unwrap();
        let run = async |device: &mut Device, mlp: &Mlp<H, I, F, D>, live: usize, stream: bool| {
            let count = count(live).to_hbm(&mut device.pdma).await.unwrap();
            let output = if stream {
                launch(streamed, (&mut *device, mlp, &count, &input)).await
            } else {
                launch(sequential, (&mut *device, mlp, &count, &input)).await
            };
            output
                .unwrap()
                .to_host::<m![Span, H]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec()
                .iter()
                .map(|v| f64::from(v.to_f32()))
                .collect::<Vec<f64>>()
        };
        let error = |got: &[f64], expected: &[f64]| {
            let (difference, norm) = got.iter().zip(expected).fold((0.0, 0.0), |(d, n), (g, e)| {
                (d + (g - e) * (g - e), n + e * e)
            });
            (difference / norm).sqrt()
        };
        for live in [1, 200, 300, 512] {
            let one = run(&mut *device, mlp, live, false).await;
            let polled = run(&mut *device, mlp, live, true).await;
            for row in 0..live {
                let at = row * hidden..(row + 1) * hidden;
                assert_eq!(polled[at.clone()], one[at], "{live} live rows, row {row}");
            }
            let zeros = &polled[zero * hidden..][..hidden];
            assert!(
                live <= zero || zeros.iter().all(|&v| v == 0.0),
                "{live} live rows, zero row"
            );
            for row in [0, live / 2, live - 1]
                .into_iter()
                .filter(|&row| row != zero)
            {
                let expected = weights.reference(&rows[row * hidden..][..hidden]);
                let got = &polled[row * hidden..][..hidden];
                let error = error(got, &expected);
                assert!(
                    error <= tolerance,
                    "{live} live rows, row {row}: relative error {error}"
                );
            }
        }
        window_matches(&mut *device, (mlp, changed[2])).await;
        let base = run(&mut *device, mlp, 300, true).await;
        for (part, changed) in ["gate", "up", "down"].into_iter().zip(changed) {
            let other = run(&mut *device, changed, 300, true).await;
            let at = 299 * hidden..300 * hidden;

            assert!(
                error(&other[at.clone()], &base[at]) > 0.1,
                "changed {part} weights"
            );
        }
    }

    /// Two layers polled as a window of two live layers over two spans, the second partial, give
    /// each span's rows through the two layers one after another.
    async fn window_matches<H: AxisName, I: AxisName, F: DeviceSend, D: DeviceSend>(
        device: &mut Device,
        layers: (&Mlp<H, I, F, D>, &Mlp<H, I, F, D>),
    ) where
        Down<I>: AxisName,
        Mlp<H, I, F, D>: for<'x, 's> Module<&'x Tile<H>, (), &'s mut (), Output = Tile<H>>,
        for<'m> &'m Mlp<H, I, F, D>: IntoStream<
                Slot<Window<H>>,
                Live<(), Spans>,
                (),
                Stream: Stream<Item = Window<H>> + AsMut<Slot<Window<H>>>,
            >,
    {
        let (hidden, live) = (H::SIZE, 700);
        let values = random(Spans::SIZE * hidden, 0x3c6e_f372_fe94_f82b);
        let input = HostTensor::<bf16, m![Spans, H]>::from_vec(values.clone())
            .to_hbm::<Chip, Blocked<H, Spans>>(&mut device.pdma)
            .await
            .unwrap();
        let rows = count(live).to_hbm(&mut device.pdma).await.unwrap();
        let polled = launch(window, (&mut *device, layers, &rows, &input))
            .await
            .unwrap()
            .to_host::<m![Spans, H]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (span, rows) in values.chunks_exact(Span::SIZE * hidden).enumerate() {
            let tile = (live - span * Span::SIZE).min(Span::SIZE);
            let count = count(tile).to_hbm(&mut device.pdma).await.unwrap();
            let rows = HostTensor::<bf16, m![Span, H]>::from_vec(rows.to_vec())
                .to_hbm::<Chip, Blocked<H, Span>>(&mut device.pdma)
                .await
                .unwrap();
            let lower = launch(sequential, (&mut *device, layers.0, &count, &rows))
                .await
                .unwrap();
            let upper = launch(sequential, (&mut *device, layers.1, &count, &lower))
                .await
                .unwrap()
                .to_host::<m![Span, H]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let at = span * Span::SIZE * hidden;
            assert_eq!(
                polled[at..at + tile * hidden],
                upper[..tile * hidden],
                "span {span} through two layers"
            );
        }
    }

    /// The logical weights of `H` inputs and `I` features, and the three that each change one of
    /// the gate, up or down weights.
    fn weights<H: AxisName, I: AxisName>() -> [(Weights, [Vec<f8e4m3>; 3]); 4]
    where
        Down<I>: AxisName,
    {
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)].map(|(g, u, d)| {
            Weights::new((H::SIZE, I::SIZE), (SEEDS.0 + g, SEEDS.1 + u, SEEDS.2 + d))
        })
    }

    /// The BF16 MLP of `H` inputs and `I` features within BF16 rounding of the reference.
    async fn linears_match<H: AxisName, I: AxisName>(device: &mut Device)
    where
        Down<I>: AxisName,
    {
        let [base, changes @ ..] = weights::<H, I>();
        let mlp = linears::<H, I>(&mut *device, &base.0).await;
        let mut changed = Vec::new();
        for (weights, _) in &changes {
            changed.push(linears::<H, I>(&mut *device, weights).await);
        }
        check(
            &mut *device,
            &mlp,
            &base.0,
            [&changed[0], &changed[1], &changed[2]],
            0.02,
        )
        .await;
    }

    /// Both weight forms of the same MLP: the E4M3 one within its activation quantization of the
    /// reference, the BF16 one within BF16 rounding.
    #[tokio::test(flavor = "current_thread")]
    async fn streams_the_sequential_mlp_in_both_forms() {
        type H = Hidden<4096>;
        type I = Intermediate<12288>;
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let [base, changes @ ..] = weights::<H, I>();
        let mlp = quantized::<H, I>(&mut device, &base.1).await;
        let mut changed = Vec::new();
        for (_, stored) in &changes {
            changed.push(quantized::<H, I>(&mut device, stored).await);
        }
        check(
            &mut device,
            &mlp,
            &base.0,
            [&changed[0], &changed[1], &changed[2]],
            0.06,
        )
        .await;
        linears_match::<H, I>(&mut device).await;
    }

    /// The BF16 MLP at each listed hidden width and intermediate width, padded feature slices
    /// included, gives the sequential rows when streamed and when two layers are windowed.
    #[tokio::test(flavor = "current_thread")]
    async fn streams_the_bf16_mlp_at_every_width() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        linears_match::<Hidden<1024>, Intermediate<3072>>(&mut device).await;
        linears_match::<Hidden<2048>, Intermediate<6144>>(&mut device).await;
        linears_match::<Hidden<2560>, Intermediate<9728>>(&mut device).await;
        linears_match::<Hidden<5120>, Intermediate<14336>>(&mut device).await;
        linears_match::<Hidden<6656>, Intermediate<19968>>(&mut device).await;
    }
}
