//! Dense attention over caller-supplied visible keys.

use crate::Module;
use furiosa_opt_std::prelude::*;

/// Project the same input independently into query, key and value rows.
#[derive(DeviceSend)]
pub struct Projection<Query, Key = Query, Value = Key> {
    /// Query projection.
    pub query: Query,
    /// Key projection.
    pub key: Key,
    /// Value projection.
    pub value: Value,
}

impl<X, Q, K, V, Query, Key, Value> Module<(&mut Device, &X)> for Projection<Query, Key, Value>
where
    Query: crate::Unary<X, Output = Q>,
    Key: crate::Unary<X, Output = K>,
    Value: crate::Unary<X, Output = V>,
{
    type Output = (Q, K, V);

    fn forward(&self, (device, input): (&mut Device, &X)) -> Self::Output {
        let query = self.query.forward((&mut *device, input));
        let key = self.key.forward((&mut *device, input));
        let value = self.value.forward((device, input));
        (query, key, value)
    }
}

/// A row's query, key and value columns, `W` each, in that order.
#[derive(Clone, Debug)]
pub struct Joint<W>(std::marker::PhantomData<W>);

impl<W: AxisName> AxisName for Joint<W> {
    const NAME: Ident = Ident::new("Joint");
    const SIZE: usize = 3 * W::SIZE;
}

/// Project rows once into their query, key and value columns side by side, the weight's outputs
/// the three in that order, and return each as rows of its own.
#[derive(DeviceSend)]
pub struct Fused<L>(pub L);

impl<'x, In: AxisName, W: AxisName, R: AxisName, Chips: M>
    Module<(&mut Device, &'x HbmTensor<bf16, Chips, m![R, In]>)>
    for Fused<crate::ops::linear::Linear<In, Joint<W>, Chips>>
{
    type Output = (
        HbmTensor<bf16, Chips, m![R, W]>,
        HbmTensor<bf16, Chips, m![R, W]>,
        HbmTensor<bf16, Chips, m![R, W]>,
    );

    fn forward(
        &self,
        (device, input): (&mut Device, &'x HbmTensor<bf16, Chips, m![R, In]>),
    ) -> Self::Output {
        let (mut query, mut key, mut value) =
            (HbmTensor::new(), HbmTensor::new(), HbmTensor::new());
        crate::ops::linear::Projection::<In, Joint<W>, R, Chips, ()>::new(self.0.tensor(), &())
            .forward((&mut *device, (input, (&mut query, &mut key, &mut value))));
        (query, key, value)
    }
}

/// Attend over caller-supplied visible keys with independent query counts and grouped KV heads.
#[derive(DeviceSend)]
pub struct Attention<Heads = usize, Scale = f64> {
    /// Query-head geometry.
    pub heads: Heads,
    /// Multiplier applied to query/key scores before additive bias.
    pub scale: Scale,
}

type Rows<D, Count, Width, Chips = m![1]> = HbmTensor<D, Chips, m![Count, Width]>;
type Scale = HbmTensor<f32, m![1], m![1]>;

impl<
    D: Scalar,
    Width: AxisName,
    KvWidth: AxisName,
    Queries: AxisName,
    Keys: AxisName,
    Heads: AxisName,
    Bias,
>
    Module<(
        &mut Device,
        (
            &Rows<D, Queries, Width>,
            &Rows<D, Keys, KvWidth>,
            &Rows<D, Keys, KvWidth>,
            Bias,
        ),
    )> for Attention<std::marker::PhantomData<Heads>, Scale>
{
    type Output = Rows<D, Queries, Width>;

    fn forward(
        &self,
        _: (
            &mut Device,
            (
                &Rows<D, Queries, Width>,
                &Rows<D, Keys, KvWidth>,
                &Rows<D, Keys, KvWidth>,
                Bias,
            ),
        ),
    ) -> Self::Output {
        const {
            assert!(Heads::SIZE > 0 && Width::SIZE > 0 && Width::SIZE % Heads::SIZE == 0);
            assert!(KvWidth::SIZE > 0 && KvWidth::SIZE % (Width::SIZE / Heads::SIZE) == 0);
            assert!(Width::SIZE % KvWidth::SIZE == 0);
        }
        todo!("NPU scaled dot-product attention with broadcast additive bias")
    }
}

impl<
    D: Scalar,
    Width: AxisName,
    KvWidth: AxisName,
    Queries: AxisName,
    Keys: AxisName,
    Heads: AxisName,
>
    Module<(
        &mut Device,
        (
            &Rows<D, Queries, Width>,
            &Rows<D, Keys, KvWidth>,
            &Rows<D, Keys, KvWidth>,
        ),
    )> for Attention<std::marker::PhantomData<Heads>, Scale>
{
    type Output = Rows<D, Queries, Width>;

    fn forward(
        &self,
        (device, (query, key, value)): (
            &mut Device,
            (
                &Rows<D, Queries, Width>,
                &Rows<D, Keys, KvWidth>,
                &Rows<D, Keys, KvWidth>,
            ),
        ),
    ) -> Self::Output {
        self.forward((device, (query, key, value, ())))
    }
}

/// A score multiplier fixed when the kernel compiles, the bits of an FP32.
#[derive(Clone, Copy, Debug, DeviceSend)]
pub struct Scaled<const BITS: usize>;

/// A packed sequence's text, audio and video spans, each with its live rows.
pub type Packed<
    const TL: usize,
    const TR: usize,
    const AL: usize,
    const AR: usize,
    const VL: usize,
    const VR: usize,
> = crate::ops::diffusion::packed::Packed<
    crate::ops::diffusion::packed::Span<TL, TR>,
    crate::ops::diffusion::packed::Span<AL, AR>,
    crate::ops::diffusion::packed::Span<VL, VR>,
>;

/// A packed sequence's parts on `N` chips, each its share of every span's rows: every part but the
/// last holds live rows only, and the last `TL`, `AL` and `VL` live rows.
pub type Split<
    const TL: usize,
    const TR: usize,
    const AL: usize,
    const AR: usize,
    const VL: usize,
    const VR: usize,
    const N: usize,
> = crate::ops::collective::rows::Pieces<Packed<TL, TR, AL, AR, VL, VR>, N>;

/// Each row's rotation of a 128-wide head.
type Rotation<R, C = crate::ops::layout::axes::Chip> =
    super::rope::Rotation<super::flash::Head, C, R>;

/// A context: each row's rotation, the keys' visibility and the attended spans.
type Context<'a, R, K, Chips> = (
    &'a Rotation<R, Chips>,
    &'a super::flash::Visible<K, Chips>,
    &'a crate::compose::Count<Chips>,
);

/// Bidirectional attention of a packed text, audio and video sequence over its visible keys, each
/// row rotated by its own rotation, 4,096 query rows at a time; a pad row's output is unspecified.
impl<
    'a,
    const TL: usize,
    const TR: usize,
    const AL: usize,
    const AR: usize,
    const VL: usize,
    const VR: usize,
    W: AxisName,
    Heads: AxisName,
    K: AxisName,
    Chips: M,
    const BITS: usize,
>
    Module<(
        &mut Device,
        (
            &Rows<bf16, Packed<TL, TR, AL, AR, VL, VR>, W, Chips>,
            &Rows<bf16, Packed<TL, TR, AL, AR, VL, VR>, W, Chips>,
            &Rows<bf16, Packed<TL, TR, AL, AR, VL, VR>, W, Chips>,
            Context<'a, Packed<TL, TR, AL, AR, VL, VR>, K, Chips>,
        ),
    )> for Attention<std::marker::PhantomData<Heads>, Scaled<BITS>>
{
    type Output = Rows<bf16, Packed<TL, TR, AL, AR, VL, VR>, W, Chips>;

    fn forward(
        &self,
        (device, (query, key, value, (rotation, visible, spans))): (
            &mut Device,
            (
                &Rows<bf16, Packed<TL, TR, AL, AR, VL, VR>, W, Chips>,
                &Rows<bf16, Packed<TL, TR, AL, AR, VL, VR>, W, Chips>,
                &Rows<bf16, Packed<TL, TR, AL, AR, VL, VR>, W, Chips>,
                Context<'a, Packed<TL, TR, AL, AR, VL, VR>, K, Chips>,
            ),
        ),
    ) -> Self::Output {
        const {
            assert!(
                Heads::SIZE == 2 * super::flash::HeadPairs::<W>::SIZE,
                "a chip attends its heads in pairs"
            )
        };
        let (query, rotated) =
            super::rope::Rotary::<super::flash::Head, super::flash::Head, bf16, Chips>::new()
                .forward((&mut *device, (&(query, key), rotation)));
        let keys = Keys::<K>::default().forward((&mut *device, &rotated));
        drop(rotated);
        let values = Keys::<K>::default().forward((&mut *device, value));
        // SAFETY: a row is its heads in order, two a pair.
        let query: super::flash::Queries<
            Packed<TL, TR, AL, AR, VL, VR>,
            super::flash::HeadPairs<W>,
            Chips,
        > = unsafe { query.reshape() };
        let output = self.forward((device, (&query, &keys, &values, (visible, spans))));
        // SAFETY: as for the queries.
        unsafe { output.reshape() }
    }
}

/// Normalize each head independently while retaining flattened feature columns.
#[derive(DeviceSend)]
pub struct HeadNorm<
    Head: AxisName,
    Parameter = HbmTensor<f32, m![1], m![Head]>,
    Epsilon = HbmTensor<f32, m![1], m![1]>,
> {
    pub weight: Parameter,
    pub epsilon: Epsilon,
    pub head: std::marker::PhantomData<Head>,
}

pub type Distributed<Head, Chips> =
    HeadNorm<Head, HbmTensor<f32, Chips, m![Head]>, HbmTensor<f32, Chips, m![1]>>;

#[derive(Clone, Copy, DeviceSend)]
pub struct Compiled<const EPSILON: usize>;

impl<const BITS: usize> From<Compiled<BITS>> for f32 {
    fn from(_: Compiled<BITS>) -> Self {
        f32::from_bits(
            const {
                assert!(BITS <= u32::MAX as usize);
                BITS as u32
            },
        )
    }
}

/// Bidirectional attention of a packed sequence split over the chips, as for one sequence, with
/// every part's video, audio and text rows landing as keys in that order.
impl<
    'a,
    const TL: usize,
    const TR: usize,
    const AL: usize,
    const AR: usize,
    const VL: usize,
    const VR: usize,
    const N: usize,
    W: AxisName,
    Heads: AxisName,
    K: AxisName,
    Chips: M,
    const BITS: usize,
>
    Module<(
        &mut Device,
        (
            &Rows<bf16, Split<TL, TR, AL, AR, VL, VR, N>, W, Chips>,
            &Rows<bf16, Split<TL, TR, AL, AR, VL, VR, N>, W, Chips>,
            &Rows<bf16, Split<TL, TR, AL, AR, VL, VR, N>, W, Chips>,
            Context<'a, Split<TL, TR, AL, AR, VL, VR, N>, K, Chips>,
        ),
    )> for Attention<std::marker::PhantomData<Heads>, Scaled<BITS>>
{
    type Output = Rows<bf16, Split<TL, TR, AL, AR, VL, VR, N>, W, Chips>;

    fn forward(
        &self,
        (device, (query, key, value, (rotation, visible, spans))): (
            &mut Device,
            (
                &Rows<bf16, Split<TL, TR, AL, AR, VL, VR, N>, W, Chips>,
                &Rows<bf16, Split<TL, TR, AL, AR, VL, VR, N>, W, Chips>,
                &Rows<bf16, Split<TL, TR, AL, AR, VL, VR, N>, W, Chips>,
                Context<'a, Split<TL, TR, AL, AR, VL, VR, N>, K, Chips>,
            ),
        ),
    ) -> Self::Output {
        const {
            assert!(
                Heads::SIZE == 2 * super::flash::HeadPairs::<W>::SIZE,
                "a chip attends its heads in pairs"
            )
        };
        let (query, rotated) =
            super::rope::Rotary::<super::flash::Head, super::flash::Head, bf16, Chips>::new()
                .forward((&mut *device, (&(query, key), rotation)));
        let keys = Keys::<K>::default().forward((&mut *device, &rotated));
        drop(rotated);
        let values = Keys::<K>::default().forward((&mut *device, value));
        // SAFETY: a row is its heads in order, two a pair.
        let query: super::flash::Queries<
            Split<TL, TR, AL, AR, VL, VR, N>,
            super::flash::HeadPairs<W>,
            Chips,
        > = unsafe { query.reshape() };
        let output = self.forward((device, (&query, &keys, &values, (visible, spans))));
        // SAFETY: as for the queries.
        unsafe { output.reshape() }
    }
}

/// A packed sequence's live rows as keys or values, video, audio, then the text envelope; copies
/// of the envelope's rows fill the last span, so every absent key is finite.
struct Keys<K>(std::marker::PhantomData<K>);

impl<K> Default for Keys<K> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

#[derive(Clone, Debug)]
struct KeyRows<Spans>(std::marker::PhantomData<Spans>);

impl<Spans: AxisName> AxisName for KeyRows<Spans> {
    const NAME: Ident = Ident::new("Keys");
    const SIZE: usize = Spans::SIZE * super::flash::Span::SIZE;
}

#[derive(Clone, Debug)]
struct Run<const N: usize>;

impl<const N: usize> AxisName for Run<N> {
    const NAME: Ident = Ident::new("Run");
    const SIZE: usize = N;
}

struct Copy<Spans, const N: usize>(std::marker::PhantomData<Spans>);

impl<Spans: AxisName, const N: usize, R: AxisName, W: AxisName, Chips: M>
    Module<(
        &mut Device,
        (
            (&HbmTensor<bf16, Chips, m![R, W]>, usize),
            (
                &mut super::flash::Tokens<Spans, super::flash::HeadPairs<W>, Chips>,
                usize,
            ),
        ),
    )> for Copy<Spans, N>
{
    type Output = ();

    fn forward(
        &self,
        (device, ((rows, from), (tokens, to))): (
            &mut Device,
            (
                (&HbmTensor<bf16, Chips, m![R, W]>, usize),
                (
                    &mut super::flash::Tokens<Spans, super::flash::HeadPairs<W>, Chips>,
                    usize,
                ),
            ),
        ),
    ) {
        use super::flash::{Duo, Head, HeadPairs};
        type P<W> = Symbol<HeadPairs<W>>;
        type K<Spans> = Symbol<KeyRows<Spans>>;
        type L<const N: usize> = Symbol<Run<N>>;
        // SAFETY: a row is its heads in order, two a pair.
        let source = unsafe { rows.view().reshape::<Chips, m![R, { P<W> }, Duo, Head]>() }
            .tile::<m![R], N, m![R = ({ L<N> }) # R, { P<W> }, Duo, Head]>(from);
        // SAFETY: the tile's rows are the run's in order; only the row axis is renamed.
        let source = unsafe { source.reshape::<Chips, m![{ L<N> } # R, { P<W> }, Duo, Head]>() };
        // SAFETY: a head's keys of every span are its keys in order.
        let target = unsafe {
            tokens
                .view_mut()
                .reshape::<Chips, m![{ P<W> }, Duo, { K<Spans> }, Head]>()
        }
        .tile::<m![{ K<Spans> }], N, m![
            { P<W> },
            Duo,
            { K<Spans> } = ({ L<N> }) #{!} ({ K<Spans> }),
            Head
        ]>(to);
        // SAFETY: as for the source.
        let target = unsafe {
            target.reshape::<Chips, m![{ P<W> }, Duo, { L<N> } #{!} ({ K<Spans> }), Head]>()
        };
        source.to_hbm_view(&mut device.tdma, target);
    }
}

impl<
    const TL: usize,
    const TR: usize,
    const AL: usize,
    const AR: usize,
    const VL: usize,
    const VR: usize,
    W: AxisName,
    K: AxisName,
    Chips: M,
>
    Module<(
        &mut Device,
        &Rows<bf16, Packed<TL, TR, AL, AR, VL, VR>, W, Chips>,
    )> for Keys<K>
{
    type Output = super::flash::Tokens<K, super::flash::HeadPairs<W>, Chips>;

    fn forward(
        &self,
        (device, rows): (
            &mut Device,
            &Rows<bf16, Packed<TL, TR, AL, AR, VL, VR>, W, Chips>,
        ),
    ) -> Self::Output {
        use super::flash;
        let (text, end) = (VL + AL, K::SIZE * flash::Span::SIZE);
        const {
            let end = K::SIZE * super::flash::Span::SIZE;
            assert!(
                end >= VL + AL + TR && end <= VL + AL + TR + AR + VR,
                "the keys past the text envelope continue the packed rows"
            )
        };
        let mut tokens = HbmTensor::new();
        Copy::<K, VL>(std::marker::PhantomData)
            .forward((device, ((rows, TR + AR), (&mut tokens, 0))));
        Copy::<K, AL>(std::marker::PhantomData).forward((device, ((rows, TR), (&mut tokens, VL))));
        for run in 0..(end - text).div_ceil(TR) {
            let to = (text + run * TR).min(end - TR);
            Copy::<K, TR>(std::marker::PhantomData)
                .forward((device, ((rows, to - text), (&mut tokens, to))));
        }
        tokens
    }
}

/// The parts' live rows as keys or values: every part's video rows, then audio, then text; copies
/// of audio rows fill the last span, so every absent key is finite.
impl<
    const TL: usize,
    const TR: usize,
    const AL: usize,
    const AR: usize,
    const VL: usize,
    const VR: usize,
    const N: usize,
    W: AxisName,
    K: AxisName,
    Chips: M,
>
    Module<(
        &mut Device,
        &Rows<bf16, Split<TL, TR, AL, AR, VL, VR, N>, W, Chips>,
    )> for Keys<K>
{
    type Output = super::flash::Tokens<K, super::flash::HeadPairs<W>, Chips>;

    fn forward(
        &self,
        (device, rows): (
            &mut Device,
            &Rows<bf16, Split<TL, TR, AL, AR, VL, VR, N>, W, Chips>,
        ),
    ) -> Self::Output {
        use super::flash;
        let (part, end) = (TR + AR + VR, K::SIZE * flash::Span::SIZE);
        let (video, audio) = ((N - 1) * VR + VL, (N - 1) * AR + AL);
        const {
            let end = K::SIZE * super::flash::Span::SIZE;
            let keys = (N - 1) * (VR + AR) + VL + AL + N * TR;
            assert!(
                end >= keys && end - keys <= N * (TR + AR + VR),
                "the keys past the text envelope are rows"
            );
        };
        let mut tokens = HbmTensor::new();
        #[unroll]
        for at in 0..N - 1 {
            Copy::<K, VR>(std::marker::PhantomData).forward((
                device,
                ((rows, at * part + TR + AR), (&mut tokens, at * VR)),
            ));
            Copy::<K, AR>(std::marker::PhantomData).forward((
                device,
                ((rows, at * part + TR), (&mut tokens, video + at * AR)),
            ));
        }
        let last = (N - 1) * part;
        Copy::<K, VL>(std::marker::PhantomData)
            .forward((device, ((rows, last + TR + AR), (&mut tokens, video - VL))));
        Copy::<K, AL>(std::marker::PhantomData).forward((
            device,
            ((rows, last + TR), (&mut tokens, video + audio - AL)),
        ));
        #[unroll]
        for at in 0..N {
            Copy::<K, TR>(std::marker::PhantomData).forward((
                device,
                ((rows, at * part), (&mut tokens, video + audio + at * TR)),
            ));
        }
        let text = video + audio + N * TR;
        for run in 0..(end - text).div_ceil(AR) {
            let to = (text + run * AR).min(end - AR);
            Copy::<K, AR>(std::marker::PhantomData)
                .forward((device, ((rows, to - text), (&mut tokens, to))));
        }
        tokens
    }
}

/// One span's rows, as a text envelope holds them.
type Text<const LIVE: usize, const ROWS: usize> = crate::ops::diffusion::packed::Span<LIVE, ROWS>;

/// Bidirectional attention of one span's rows over any visible keys: copies of the rows fill a
/// query tile and `K` key spans, so every row read is finite; the tile's first rows are the output.
impl<
    const LIVE: usize,
    const ROWS: usize,
    W: AxisName,
    Heads: AxisName,
    K: AxisName,
    Chips: M,
    const BITS: usize,
>
    Module<(
        &mut Device,
        (
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            (
                &super::flash::Visible<K, Chips>,
                &crate::compose::Count<Chips>,
            ),
        ),
    )> for Attention<std::marker::PhantomData<Heads>, Scaled<BITS>>
{
    type Output = Rows<bf16, Text<LIVE, ROWS>, W, Chips>;

    fn forward(
        &self,
        (device, (query, key, value, (visible, spans))): (
            &mut Device,
            (
                &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
                &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
                &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
                (
                    &super::flash::Visible<K, Chips>,
                    &crate::compose::Count<Chips>,
                ),
            ),
        ),
    ) -> Self::Output {
        use super::flash::{self, HeadPairs, Rows as TileRows};
        use crate::ops::diffusion::packed::Span;
        const {
            assert!(
                Heads::SIZE == 2 * HeadPairs::<W>::SIZE,
                "a chip attends its heads in pairs"
            );
            assert!(TileRows::SIZE.is_multiple_of(ROWS));
            assert!((K::SIZE * flash::Span::SIZE).is_multiple_of(ROWS));
        };
        let mut tile: Rows<bf16, TileRows, W, Chips> = HbmTensor::new();
        for copy in 0..TileRows::SIZE / ROWS {
            Span::<LIVE, ROWS>::new().forward((&mut *device, (query, (&mut tile, copy * ROWS))));
        }
        // SAFETY: a row is its heads in order, two a pair.
        let tile: flash::Queries<TileRows, HeadPairs<W>, Chips> = unsafe { tile.reshape() };
        let tokens = |device: &mut Device, rows: &Rows<bf16, Text<LIVE, ROWS>, W, Chips>| {
            let mut tokens = HbmTensor::new();
            for copy in 0..K::SIZE * flash::Span::SIZE / ROWS {
                Copy::<K, ROWS>(std::marker::PhantomData)
                    .forward((device, ((rows, 0), (&mut tokens, copy * ROWS))));
            }
            tokens
        };
        let (keys, values) = (tokens(device, key), tokens(device, value));
        let output = self.forward((
            &mut *device,
            (&tile, &keys, &values, (&flash::Scattered(visible), spans)),
        ));
        // SAFETY: as for the query tile.
        let output: Rows<bf16, TileRows, W, Chips> = unsafe { output.reshape() };
        Span::<LIVE, ROWS>::new().forward((device, (&output, 0)))
    }
}

/// The same attention with the context a layer of a collection lends it.
impl<
    'c,
    'v,
    'n,
    const LIVE: usize,
    const ROWS: usize,
    W: AxisName,
    Heads: AxisName,
    K: AxisName,
    Chips: M,
    const BITS: usize,
>
    Module<(
        &mut Device,
        (
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            &'c mut (
                &'v super::flash::Visible<K, Chips>,
                &'n crate::compose::Count<Chips>,
            ),
        ),
    )> for Attention<std::marker::PhantomData<Heads>, Scaled<BITS>>
{
    type Output = Rows<bf16, Text<LIVE, ROWS>, W, Chips>;

    fn forward(
        &self,
        (device, (query, key, value, context)): (
            &mut Device,
            (
                &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
                &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
                &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
                &'c mut (
                    &'v super::flash::Visible<K, Chips>,
                    &'n crate::compose::Count<Chips>,
                ),
            ),
        ),
    ) -> Self::Output {
        self.forward((device, (query, key, value, *context)))
    }
}

impl<Head: AxisName, Width: AxisName, Count: AxisName>
    Module<(&mut Device, &Rows<f32, Count, Width>)> for HeadNorm<Head>
{
    type Output = Rows<f32, Count, Width>;

    fn forward(&self, _: (&mut Device, &Rows<f32, Count, Width>)) -> Self::Output {
        const { assert!(Head::SIZE > 0 && Width::SIZE % Head::SIZE == 0) };
        todo!("NPU per-head RMS normalization with shared head-width weight and FP32 statistics")
    }
}

/// [`HeadNorm`] over rows that come in no whole norm tile, such as one decoded row: a typed
/// contract without a kernel.
#[derive(DeviceSend)]
pub struct Untiled<Head: AxisName>(pub HeadNorm<Head>);

impl<D: Scalar, Head: AxisName, Width: AxisName, Count: AxisName>
    Module<(&mut Device, &Rows<D, Count, Width>)> for Untiled<Head>
{
    type Output = Rows<D, Count, Width>;

    fn forward(&self, _: (&mut Device, &Rows<D, Count, Width>)) -> Self::Output {
        const { assert!(Head::SIZE > 0 && Width::SIZE % Head::SIZE == 0) };
        todo!("NPU per-head RMS normalization of rows that come in no whole norm tile")
    }
}

/// Head rows of a row sequence in HBM.
pub(crate) mod heads {
    use crate::ops::vector::MeanSquare;
    use furiosa_opt_std::prelude::*;
    use std::marker::PhantomData;

    /// The heads of a row of `Width` features, `Head` features each.
    #[derive(Clone, Debug)]
    pub struct Heads<Width, Head>(PhantomData<(Width, Head)>);

    impl<Width: AxisName, Head: AxisName> AxisName for Heads<Width, Head> {
        const NAME: Ident = Ident::new("Heads");

        const SIZE: usize = {
            assert!(Head::SIZE > 0 && Width::SIZE % Head::SIZE == 0);
            assert!(Width::SIZE / Head::SIZE <= 64);
            Width::SIZE / Head::SIZE
        };
    }

    pub(crate) type H<Width, Head> = Symbol<Heads<Width, Head>>;

    /// Heads of a row padded to the power of two whose slices a row spans.
    #[derive(Clone, Debug)]
    pub struct Padded<Width, Head>(PhantomData<(Width, Head)>);

    impl<Width: AxisName, Head: AxisName> AxisName for Padded<Width, Head> {
        const NAME: Ident = Ident::new("Padded");
        const SIZE: usize = Heads::<Width, Head>::SIZE.next_power_of_two();
    }

    /// Rows one slice holds of a norm tile.
    #[derive(Clone, Debug)]
    pub struct Lanes<Count, Width, Head>(PhantomData<(Count, Width, Head)>);

    impl<Count: AxisName, Width: AxisName, Head: AxisName> AxisName for Lanes<Count, Width, Head> {
        const NAME: Ident = Ident::new("NormLanes");
        const SIZE: usize = {
            let groups = 2 * 256 / Padded::<Width, Head>::SIZE;
            let mut lanes = 32;
            while lanes > 1 && !Count::SIZE.is_multiple_of(groups * lanes) {
                lanes /= 2;
            }
            assert!(
                Count::SIZE.is_multiple_of(groups * lanes),
                "rows come in whole norm tiles"
            );
            lanes
        };
    }

    /// Rows a cluster holds of a norm tile.
    #[derive(Clone, Debug)]
    pub struct Halves<Count, Width, Head>(PhantomData<(Count, Width, Head)>);

    impl<Count: AxisName, Width: AxisName, Head: AxisName> AxisName for Halves<Count, Width, Head> {
        const NAME: Ident = Ident::new("NormHalves");
        const SIZE: usize = 256 / Padded::<Width, Head>::SIZE * Lanes::<Count, Width, Head>::SIZE;
    }

    /// Rows of a norm tile.
    #[derive(Clone, Debug)]
    pub struct Block<Count, Width, Head>(PhantomData<(Count, Width, Head)>);

    impl<Count: AxisName, Width: AxisName, Head: AxisName> AxisName for Block<Count, Width, Head> {
        const NAME: Ident = Ident::new("NormBlock");
        const SIZE: usize = 2 * Halves::<Count, Width, Head>::SIZE;
    }

    /// The norm tiles of the rows, in row order.
    #[derive(Clone, Debug)]
    pub struct Blocks<Count, Width, Head>(PhantomData<(Count, Width, Head)>);

    impl<Count: AxisName, Width: AxisName, Head: AxisName> AxisName for Blocks<Count, Width, Head> {
        const NAME: Ident = Ident::new("NormBlocks");
        const SIZE: usize = Count::SIZE / Block::<Count, Width, Head>::SIZE;
    }

    type P<W, Hd> = Symbol<Padded<W, Hd>>;
    type L<C, W, Hd> = Symbol<Lanes<C, W, Hd>>;
    type Hb<C, W, Hd> = Symbol<Halves<C, W, Hd>>;
    type B<C, W, Hd> = Symbol<Block<C, W, Hd>>;
    type Bs<C, W, Hd> = Symbol<Blocks<C, W, Hd>>;
    type Blocked<C, W, Hd> = m![{ Bs<C, W, Hd> }, { B<C, W, Hd> }, { H<W, Hd> }, Hd];
    type Cluster<C, W, Hd> = m![{ B<C, W, Hd> } / ({ Hb<C, W, Hd> })];
    /// Heads by row groups: a row's heads over the slices of its group.
    type Groups<C, W, Hd> = m![
        { H<W, Hd> } # ({ P<W, Hd> }),
        { B<C, W, Hd> } % ({ Hb<C, W, Hd> }) / ({ L<C, W, Hd> })
    ];
    type Part<C, W, Hd, X = m![1]> = DmTensor<
        bf16,
        X,
        Cluster<C, W, Hd>,
        Groups<C, W, Hd>,
        m![{ B<C, W, Hd> } % ({ L<C, W, Hd> }), Hd],
    >;
    type Statistic<C, W, Hd, X = m![1]> = DmTensor<
        f32,
        X,
        Cluster<C, W, Hd>,
        Groups<C, W, Hd>,
        m![{ B<C, W, Hd> } % ({ L<C, W, Hd> }), 1 # 8],
    >;
    type Scalar<C, W, Hd, X = m![1]> =
        VrfTensor<f32, X, Cluster<C, W, Hd>, Groups<C, W, Hd>, m![1 # 8]>;
    type RowOperand<C, W, Hd, X = m![1]> = VrfTensor<
        f32,
        X,
        Cluster<C, W, Hd>,
        Groups<C, W, Hd>,
        m![{ B<C, W, Hd> } % ({ L<C, W, Hd> }), 1 # 8],
    >;

    impl<Head: AxisName, Width: AxisName, Count: AxisName, X: M>
        crate::Module<(&mut Device, &HbmTensor<bf16, X, m![Count, Width]>)>
        for super::HeadNorm<Head, HbmTensor<f32, X, m![Head]>, HbmTensor<f32, X, m![1]>>
    {
        type Output = HbmTensor<bf16, X, m![Count, Width]>;

        fn forward(
            &self,
            (device, input): (&mut Device, &HbmTensor<bf16, X, m![Count, Width]>),
        ) -> Self::Output {
            let fill = |device: &mut Device,
                        rows: &HbmTensor<bf16, X, m![Count, Width]>,
                        at: usize,
                        part: &mut Part<Count, Width, Head, X>| {
                // SAFETY: the tiles split the rows in order and a row is its heads in order.
                unsafe { rows.view().reshape::<X, Blocked<Count, Width, Head>>() }
                    .tile::<m![{ Bs<Count, Width, Head> }], 1, m![
                        { Bs<Count, Width, Head> } = 1 # ({ Bs<Count, Width, Head> }),
                        { B<Count, Width, Head> },
                        { H<Width, Head> },
                        Head
                    ]>(at)
                    .to_dm_view(&mut device.tdma, part.view_mut());
            };

            let drain = |device: &mut Device,
                         part: &Part<Count, Width, Head, X>,
                         rows: &mut HbmTensor<bf16, X, m![Count, Width]>,
                         at: usize| {
                part.view().to_hbm_view(
                    &mut device.tdma,
                    // SAFETY: as for `fill`.
                    unsafe { rows.view_mut().reshape::<X, Blocked<Count, Width, Head>>() }
                        .tile::<m![{ Bs<Count, Width, Head> }], 1, m![
                            { Bs<Count, Width, Head> } = 1 #{!} ({ Bs<Count, Width, Head> }),
                            { B<Count, Width, Head> },
                            { H<Width, Head> },
                            Head
                        ]>(at),
                );
            };

            let rms = |device: &mut Device,
                       part: &Part<Count, Width, Head, X>,
                       epsilon: &Scalar<Count, Width, Head, X>|
             -> RowOperand<Count, Width, Head, X> {
                let mean: Statistic<Count, Width, Head, X> = device
            .main
            .begin(part.view())
            .fetch::<m![{ B<Count, Width, Head> } % ({ L<Count, Width, Head> }), Head / 8], m![Head % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ B<Count, Width, Head> } % ({ L<Count, Width, Head> }), Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![{ B<Count, Width, Head> } % ({ L<Count, Width, Head> }), Head / 8, Head % 8 / 4 % 2],
                m![Head % 4],
            >()
            .vector_mean_square::<Head, m![{ B<Count, Width, Head> } % ({ L<Count, Width, Head> })], m![1], 0>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
                device
                .sub
                .begin(mean.view())
                .fetch::<m![{ B<Count, Width, Head> } % ({ L<Count, Width, Head> })], m![1 # 8]>()
                .collect::<m![{ B<Count, Width, Head> } % ({ L<Count, Width, Head> })], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_binary(FpBinaryOp::AddF, epsilon)
                .vector_fp_unary(FpUnaryOp::Sqrt)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .to_vrf()
            };

            let scale = |device: &mut Device,
                         part: &Part<Count, Width, Head, X>,
                         (weight, rms): (
                &VrfTensor<
                    f32,
                    X,
                    Cluster<Count, Width, Head>,
                    Groups<Count, Width, Head>,
                    m![Head],
                >,
                &RowOperand<Count, Width, Head, X>,
            ),
                         output: &mut Part<Count, Width, Head, X>| {
                device
            .main
            .begin(part.view())
            .fetch::<m![{ B<Count, Width, Head> } % ({ L<Count, Width, Head> }), Head / 8], m![Head % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ B<Count, Width, Head> } % ({ L<Count, Width, Head> }), Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![{ B<Count, Width, Head> } % ({ L<Count, Width, Head> }), Head / 8, Head % 8 / 4 % 2],
                m![Head % 4],
            >()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight)
            .vector_fp_div(rms)
            .vector_widen_concat::<
                m![{ B<Count, Width, Head> } % ({ L<Count, Width, Head> }), Head / 8],
                m![Head % 8],
            >()
            .vector_final()
            .cast::<bf16, m![Head % 8 # 16]>()
            .commit_trim::<m![Head % 8]>()
            .commit_view(output.view_mut());
            };
            let (weight, epsilon) = (&self.weight, &self.epsilon);
            let weight: DmTensor<
                f32,
                X,
                Cluster<Count, Width, Head>,
                Groups<Count, Width, Head>,
                m![Head],
            > = weight.to_dm(&mut device.tdma);
            let weight: VrfTensor<
                f32,
                X,
                Cluster<Count, Width, Head>,
                Groups<Count, Width, Head>,
                m![Head],
            > = device
                .sub
                .begin(weight.view())
                .fetch::<m![Head / 8], m![Head % 8]>()
                .collect::<m![Head / 8], m![Head % 8]>()
                .to_vrf();
            // One copy a slice, shared by its rows: a copy a row lands as one slow 32-byte write each.
            let epsilon: DmTensor<
                f32,
                X,
                Cluster<Count, Width, Head>,
                Groups<Count, Width, Head>,
                m![1 # 8],
            > = epsilon.view().pad::<m![1 # 8]>().to_dm(&mut device.tdma);
            let epsilon: Scalar<Count, Width, Head, X> = device
                .sub
                .begin(epsilon.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .to_vrf();
            let tiles = <Bs<Count, Width, Head> as M>::SIZE;
            let last = tiles - 1;
            let mut output = HbmTensor::new();
            let (mut current, mut next): (
                Part<Count, Width, Head, X>,
                Part<Count, Width, Head, X>,
            ) = (DmTensor::new(), DmTensor::new());
            let (mut kept, mut held): (Part<Count, Width, Head, X>, Part<Count, Width, Head, X>) =
                (DmTensor::new(), DmTensor::new());
            fill(device, input, 0, &mut current);
            fill(device, input, 1.min(last), &mut next);
            // An odd count's last pair normalizes the last tile twice.
            for pair in 0..(tiles + 1) / 2 {
                let at = 2 * pair;
                let first = rms(device, &current, &epsilon);
                let second = rms(device, &next, &epsilon);
                scale(device, &current, (&weight, &first), &mut kept);
                fill(device, input, (at + 2).min(last), &mut current);
                drain(device, &kept, &mut output, at);
                scale(device, &next, (&weight, &second), &mut held);
                fill(device, input, (at + 3).min(last), &mut next);
                drain(device, &held, &mut output, (at + 1).min(last));
            }
            output
        }
    }

    /// Head rows a tile at a time, row-major or head-major, each slice 512 bytes of a row; the
    /// next tile lands, and the last one stores, while the current one computes.
    pub(crate) mod tiles {
        use crate::ops::vector::MeanSquare;
        use furiosa_opt_std::prelude::*;
        use std::marker::PhantomData;

        /// The heads a slice holds of a row.
        #[derive(Clone, Debug)]
        pub struct Kept<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Kept<Width, Head> {
            const NAME: Ident = Ident::new("HeadsKept");
            const SIZE: usize = {
                let kept = if Head::SIZE >= 256 {
                    1
                } else {
                    256 / Head::SIZE
                };
                assert!((Width::SIZE / Head::SIZE) % kept == 0);
                kept
            };
        }

        /// The slices of a row group: its head groups, padded.
        #[derive(Clone, Debug)]
        pub struct Slots<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Slots<Width, Head> {
            const NAME: Ident = Ident::new("HeadSlots");
            const SIZE: usize = {
                let groups = Width::SIZE / Head::SIZE / Kept::<Width, Head>::SIZE;
                assert!(groups > 0 && groups <= 64);
                groups.next_power_of_two()
            };
        }

        /// The rows a slice holds.
        #[derive(Clone, Debug)]
        pub struct Depth<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Depth<Width, Head> {
            const NAME: Ident = Ident::new("HeadDepth");
            const SIZE: usize = if Head::SIZE <= 64 { 8 } else { 4 };
        }

        /// The rows of a tile.
        #[derive(Clone, Debug)]
        pub struct Strip<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Strip<Width, Head> {
            const NAME: Ident = Ident::new("HeadStrip");
            const SIZE: usize = 2 * (256 / Slots::<Width, Head>::SIZE) * Depth::<Width, Head>::SIZE;
        }

        /// The rows of a tile one cluster holds.
        #[derive(Clone, Debug)]
        pub struct Group<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Group<Width, Head> {
            const NAME: Ident = Ident::new("HeadGroup");
            const SIZE: usize = Strip::<Width, Head>::SIZE / 2;
        }

        /// The tiles of a row sequence.
        #[derive(Clone, Debug)]
        pub struct Strips<Count, Width, Head, P = (), Chips = m![1], Q = ()>(
            pub PhantomData<(Count, Width, Head, Chips, Q)>,
            pub P,
        );

        impl<Count: AxisName, Width: AxisName, Head: AxisName> AxisName for Strips<Count, Width, Head> {
            const NAME: Ident = Ident::new("HeadStrips");
            const SIZE: usize = {
                let strip = Strip::<Width, Head>::SIZE;
                assert!(Count::SIZE % strip == 0);
                Count::SIZE / strip
            };
        }

        use super::H;
        pub(crate) type K<Width, Head> = Symbol<Kept<Width, Head>>;
        pub(crate) type P<Width, Head> = Symbol<Slots<Width, Head>>;
        pub(crate) type D<Width, Head> = Symbol<Depth<Width, Head>>;
        pub(crate) type T<Width, Head> = Symbol<Strip<Width, Head>>;
        pub(crate) type G<Width, Head> = Symbol<Group<Width, Head>>;
        pub(crate) type S<Count, Width, Head> = Symbol<Strips<Count, Width, Head>>;
        pub(crate) type Banded<Count, Width, Head> =
            m![{ S<Count, Width, Head> }, { T<Width, Head> }, { H<Width, Head> }, Head];
        pub(crate) type Half<Width, Head> = m![{ T<Width, Head> } / ({ G<Width, Head> })];
        pub(crate) type Slices<Width, Head> = m![
            { H<Width, Head> } / ({ K<Width, Head> }) # ({ P<Width, Head> }),
            { T<Width, Head> } % ({ G<Width, Head> }) / ({ D<Width, Head> })
        ];
        /// A slice's rows, then the heads it holds of each: the order row-major rows move in.
        pub(crate) type Rows<Width, Head> = m![
            { T<Width, Head> } % ({ D<Width, Head> }),
            { H<Width, Head> } % ({ K<Width, Head> })
        ];
        /// A slice's heads, then each head's rows: the order head-major rows move in.
        pub(crate) type Columns<Width, Head> = m![
            { H<Width, Head> } % ({ K<Width, Head> }),
            { T<Width, Head> } % ({ D<Width, Head> })
        ];
        /// A tile, its rows' heads in a slice in order `Q`.
        pub(crate) type Tile<E, Width, Head, Chips, Q> =
            DmTensor<E, Chips, Half<Width, Head>, Slices<Width, Head>, m![{ Q }, Head]>;
        type Statistic<Width, Head, Chips, Q> =
            DmTensor<f32, Chips, Half<Width, Head>, Slices<Width, Head>, m![{ Q }, 1 # 8]>;
        type RowOperand<Width, Head, Chips, Q> =
            VrfTensor<f32, Chips, Half<Width, Head>, Slices<Width, Head>, m![{ Q }, 1 # 8]>;

        /// Head rows in HBM: row-major, every row its heads in order, or head-major, every head
        /// its rows in order.
        pub(crate) trait Layout<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M> {
            /// The order a slice holds its rows' heads in: the order they move in.
            type Order: M;

            fn new() -> Self;

            /// Tile `at`, landed into `tile`.
            fn land(
                &self,
                device: &mut Device,
                at: usize,
                tile: &mut Tile<bf16, Width, Head, Chips, Self::Order>,
            );

            /// `tile` stored as tile `at`.
            fn store(
                &mut self,
                device: &mut Device,
                tile: &Tile<bf16, Width, Head, Chips, Self::Order>,
                at: usize,
            );
        }

        /// Head-major rows of `Width` features, `Head` features a head.
        pub(crate) type Major<Count, Width, Head, Chips = m![1]> = HbmTensor<
            bf16,
            Chips,
            m![Width / ({ Symbol<Head> }), Count, Width % ({ Symbol<Head> })],
        >;
        type Headed<Count, Width, Head> =
            m![{ H<Width, Head> }, { S<Count, Width, Head> }, { T<Width, Head> }, Head];

        impl<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M>
            Layout<Count, Width, Head, Chips> for HbmTensor<bf16, Chips, m![Count, Width]>
        {
            type Order = Rows<Width, Head>;

            fn new() -> Self {
                HbmTensor::new()
            }

            fn land(
                &self,
                device: &mut Device,
                at: usize,
                tile: &mut Tile<bf16, Width, Head, Chips, Self::Order>,
            ) {
                // SAFETY: the tiles split the rows in order and a row is its heads in order.
                unsafe { self.view().reshape::<Chips, Banded<Count, Width, Head>>() }
                    .tile::<m![{ S<Count, Width, Head> }], 1, m![
                        { S<Count, Width, Head> } = 1 # ({ S<Count, Width, Head> }),
                        { T<Width, Head> },
                        { H<Width, Head> },
                        Head
                    ]>(at)
                    .to_dm_view(&mut device.tdma, tile.view_mut());
            }

            fn store(
                &mut self,
                device: &mut Device,
                tile: &Tile<bf16, Width, Head, Chips, Self::Order>,
                at: usize,
            ) {
                tile.view().to_hbm_view(
                    &mut device.tdma,
                    // SAFETY: as for `land`.
                    unsafe {
                        self.view_mut()
                            .reshape::<Chips, Banded<Count, Width, Head>>()
                    }
                    .tile::<m![{ S<Count, Width, Head> }], 1, m![
                        { S<Count, Width, Head> } = 1 #{!} ({ S<Count, Width, Head> }),
                        { T<Width, Head> },
                        { H<Width, Head> },
                        Head
                    ]>(at),
                );
            }
        }

        impl<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M>
            Layout<Count, Width, Head, Chips> for Major<Count, Width, Head, Chips>
        {
            type Order = Columns<Width, Head>;

            fn new() -> Self {
                HbmTensor::new()
            }

            fn land(
                &self,
                device: &mut Device,
                at: usize,
                tile: &mut Tile<bf16, Width, Head, Chips, Self::Order>,
            ) {
                // SAFETY: the tiles split each head's rows in order; a head is its features in
                // order.
                unsafe { self.view().reshape::<Chips, Headed<Count, Width, Head>>() }
                    .tile::<m![{ S<Count, Width, Head> }], 1, m![
                        { H<Width, Head> },
                        { S<Count, Width, Head> } = 1 # ({ S<Count, Width, Head> }),
                        { T<Width, Head> },
                        Head
                    ]>(at)
                    .to_dm_view(&mut device.tdma, tile.view_mut());
            }

            fn store(
                &mut self,
                device: &mut Device,
                tile: &Tile<bf16, Width, Head, Chips, Self::Order>,
                at: usize,
            ) {
                tile.view().to_hbm_view(
                    &mut device.tdma,
                    // SAFETY: as for `land`.
                    unsafe {
                        self.view_mut()
                            .reshape::<Chips, Headed<Count, Width, Head>>()
                    }
                    .tile::<m![{ S<Count, Width, Head> }], 1, m![
                        { H<Width, Head> },
                        { S<Count, Width, Head> } = 1 #{!} ({ S<Count, Width, Head> }),
                        { T<Width, Head> },
                        Head
                    ]>(at),
                );
            }
        }

        /// One pass over a tile of head rows.
        pub(crate) trait Pass<Width: AxisName, Head: AxisName, Chips: M, Q: M> {
            /// Tile `at`, landed, through the pass into `output`.
            fn apply(
                &self,
                device: &mut Device,
                tile: &Tile<bf16, Width, Head, Chips, Q>,
                at: usize,
                output: &mut Tile<bf16, Width, Head, Chips, Q>,
            );
        }

        /// Every tile of `input` through `pass` into `output`, the next tile landing while one
        /// computes and the tile before it storing.
        impl<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M, Q: M, L, P>
            crate::Module<(&mut Device, (&L, &mut L))> for Strips<Count, Width, Head, P, Chips, Q>
        where
            L: Layout<Count, Width, Head, Chips, Order = Q>,
            P: Pass<Width, Head, Chips, Q>,
        {
            type Output = ();

            fn forward(&self, (device, (input, output)): (&mut Device, (&L, &mut L))) {
                let pass = &self.1;
                let tiles = Strips::<Count, Width, Head>::SIZE;
                let (mut first, mut second): (
                    Tile<bf16, Width, Head, Chips, Q>,
                    Tile<bf16, Width, Head, Chips, Q>,
                ) = (DmTensor::new(), DmTensor::new());
                let (mut kept, mut held): (
                    Tile<bf16, Width, Head, Chips, Q>,
                    Tile<bf16, Width, Head, Chips, Q>,
                ) = (DmTensor::new(), DmTensor::new());
                input.land(device, 0, &mut first);
                if const { Strips::<Count, Width, Head>::SIZE > 1 } {
                    input.land(device, 1, &mut second);
                }
                for pair in 0..tiles / 2 {
                    let at = 2 * pair;
                    pass.apply(device, &first, at, &mut kept);
                    input.land(device, (at + 2).min(tiles - 1), &mut first);
                    output.store(device, &kept, at);
                    pass.apply(device, &second, at + 1, &mut held);
                    input.land(device, (at + 3).min(tiles - 1), &mut second);
                    output.store(device, &held, at + 1);
                }
                if const { Strips::<Count, Width, Head>::SIZE % 2 == 1 } {
                    pass.apply(device, &first, tiles - 1, &mut kept);
                    output.store(device, &kept, tiles - 1);
                }
            }
        }

        /// One pass over the same tile of two row sequences, so what the tile's rows share lands
        /// once for both.
        pub(crate) trait Both<Width: AxisName, Head: AxisName, Chips: M, Q: M> {
            /// Tile `at` of both sequences, landed, through the pass into the outputs.
            fn apply(
                &self,
                device: &mut Device,
                tiles: (
                    &Tile<bf16, Width, Head, Chips, Q>,
                    &Tile<bf16, Width, Head, Chips, Q>,
                ),
                at: usize,
                outputs: (
                    &mut Tile<bf16, Width, Head, Chips, Q>,
                    &mut Tile<bf16, Width, Head, Chips, Q>,
                ),
            );
        }

        impl<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M, Q: M, L, P>
            crate::Module<(&mut Device, ((&L, &L), (&mut L, &mut L)))>
            for Strips<Count, Width, Head, P, Chips, Q>
        where
            L: Layout<Count, Width, Head, Chips, Order = Q>,
            P: Both<Width, Head, Chips, Q>,
        {
            type Output = ();

            fn forward(
                &self,
                (device, (inputs, outputs)): (&mut Device, ((&L, &L), (&mut L, &mut L))),
            ) {
                let pass = &self.1;
                type Two<Width, Head, Chips, Q> = (
                    Tile<bf16, Width, Head, Chips, Q>,
                    Tile<bf16, Width, Head, Chips, Q>,
                );
                let tiles = Strips::<Count, Width, Head>::SIZE;
                let land_both =
                    |device: &mut Device, at: usize, both: &mut Two<Width, Head, Chips, Q>| {
                        inputs.0.land(device, at, &mut both.0);
                        inputs.1.land(device, at, &mut both.1);
                    };
                let (first_out, second_out) = outputs;
                let mut store_both =
                    |device: &mut Device, both: &Two<Width, Head, Chips, Q>, at: usize| {
                        first_out.store(device, &both.0, at);
                        second_out.store(device, &both.1, at);
                    };
                let mut first: Two<Width, Head, Chips, Q> = (DmTensor::new(), DmTensor::new());
                let mut second: Two<Width, Head, Chips, Q> = (DmTensor::new(), DmTensor::new());
                let mut kept: Two<Width, Head, Chips, Q> = (DmTensor::new(), DmTensor::new());
                let mut held: Two<Width, Head, Chips, Q> = (DmTensor::new(), DmTensor::new());
                land_both(device, 0, &mut first);
                if const { Strips::<Count, Width, Head>::SIZE > 1 } {
                    land_both(device, 1, &mut second);
                }
                for pair in 0..tiles / 2 {
                    let at = 2 * pair;
                    pass.apply(device, (&first.0, &first.1), at, (&mut kept.0, &mut kept.1));
                    land_both(device, (at + 2).min(tiles - 1), &mut first);
                    store_both(device, &kept, at);
                    pass.apply(
                        device,
                        (&second.0, &second.1),
                        at + 1,
                        (&mut held.0, &mut held.1),
                    );
                    land_both(device, (at + 3).min(tiles - 1), &mut second);
                    store_both(device, &held, at + 1);
                }
                if const { Strips::<Count, Width, Head>::SIZE % 2 == 1 } {
                    pass.apply(
                        device,
                        (&first.0, &first.1),
                        tiles - 1,
                        (&mut kept.0, &mut kept.1),
                    );
                    store_both(device, &kept, tiles - 1);
                }
            }
        }

        /// The head norm's operands.
        struct Normalize<Width: AxisName, Head: AxisName, Chips: M> {
            weight: VrfTensor<f32, Chips, Half<Width, Head>, Slices<Width, Head>, m![Head]>,
            epsilon: VrfTensor<f32, Chips, Half<Width, Head>, Slices<Width, Head>, m![1 # 8]>,
        }

        impl<Width: AxisName, Head: AxisName, Chips: M, Q: M> Pass<Width, Head, Chips, Q>
            for Normalize<Width, Head, Chips>
        {
            fn apply(
                &self,
                device: &mut Device,
                rows: &Tile<bf16, Width, Head, Chips, Q>,
                _: usize,
                normalized: &mut Tile<bf16, Width, Head, Chips, Q>,
            ) {
                let mean: Statistic<Width, Head, Chips, Q> = device
                    .main
                    .begin(rows.view())
                    .fetch::<m![{ Q }, Head / 8], m![Head % 8]>()
                    .fetch_cast::<f32>()
                    .collect::<m![{ Q }, Head / 8], m![Head % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![{ Q }, Head / 8, Head % 8 / 4 % 2], m![Head % 4]>()
                    .vector_mean_square::<Head, m![{ Q }], m![1], 0>()
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                let rms: RowOperand<Width, Head, Chips, Q> = device
                    .sub
                    .begin(mean.view())
                    .fetch::<m![{ Q }], m![1 # 8]>()
                    .collect::<m![{ Q }], m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_trim::<m![1 # 4]>()
                    .vector_fp_binary(FpBinaryOp::AddF, &self.epsilon)
                    .vector_fp_unary(FpUnaryOp::Sqrt)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .to_vrf();
                device
                    .main
                    .begin(rows.view())
                    .fetch::<m![{ Q }, Head / 8], m![Head % 8]>()
                    .fetch_cast::<f32>()
                    .collect::<m![{ Q }, Head / 8], m![Head % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![{ Q }, Head / 8, Head % 8 / 4 % 2], m![Head % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &self.weight)
                    .vector_fp_div(&rms)
                    .vector_widen_concat::<m![{ Q }, Head / 8], m![Head % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Head % 8 # 16]>()
                    .commit_trim::<m![Head % 8]>()
                    .commit_view(normalized.view_mut());
            }
        }

        /// Every head row of `input` normalized by its RMS and the shared head weight.
        impl<Head: AxisName, Width: AxisName, Count: AxisName, Chips: M>
            crate::Module<(&mut Device, &Major<Count, Width, Head, Chips>)>
            for super::super::HeadNorm<
                Head,
                HbmTensor<f32, Chips, m![Head]>,
                HbmTensor<f32, Chips, m![1]>,
            >
        {
            type Output = Major<Count, Width, Head, Chips>;

            fn forward(
                &self,
                (device, input): (&mut Device, &Major<Count, Width, Head, Chips>),
            ) -> Self::Output {
                let (weight, epsilon) = (&self.weight, &self.epsilon);
                let weight: DmTensor<f32, Chips, Half<Width, Head>, Slices<Width, Head>, m![Head]> =
                    weight.to_dm(&mut device.tdma);
                let weight = device
                    .sub
                    .begin(weight.view())
                    .fetch::<m![Head / 8], m![Head % 8]>()
                    .collect::<m![Head / 8], m![Head % 8]>()
                    .to_vrf();
                let epsilon: DmTensor<
                    f32,
                    Chips,
                    Half<Width, Head>,
                    Slices<Width, Head>,
                    m![1 # 8],
                > = epsilon.view().pad::<m![1 # 8]>().to_dm(&mut device.tdma);
                let epsilon = device
                    .sub
                    .begin(epsilon.view())
                    .fetch::<m![1], m![1 # 8]>()
                    .collect::<m![1], m![1 # 8]>()
                    .to_vrf();
                let mut output = HbmTensor::new();
                Strips::<Count, Width, Head, _, Chips, Columns<Width, Head>>(
                    PhantomData,
                    Normalize { weight, epsilon },
                )
                .forward((device, (input, &mut output)));
                output
            }
        }
    }
}

impl<D: Scalar, W: AxisName, R: AxisName, H: AxisName>
    Module<(&mut Device, &(Rows<D, R, W>, Rows<D, R, W>, Rows<D, R, W>))>
    for Attention<std::marker::PhantomData<H>, Scale>
{
    type Output = Rows<D, R, W>;

    fn forward(
        &self,
        (device, (query, key, value)): (
            &mut Device,
            &(Rows<D, R, W>, Rows<D, R, W>, Rows<D, R, W>),
        ),
    ) -> Self::Output {
        self.forward((device, (query, key, value)))
    }
}

impl<'k, 'v, D: Scalar, W: AxisName, K: AxisName, Q: AxisName, H: AxisName>
    Module<(
        &mut Device,
        (
            &Rows<D, Q, W>,
            &&'k crate::ops::indexing::RowStorage<K, D>,
            &&'v crate::ops::indexing::RowStorage<K, D>,
            &super::cache::Read<'_>,
        ),
    )> for Attention<std::marker::PhantomData<H>, Scale>
{
    type Output = Rows<D, Q, W>;

    fn forward(
        &self,
        _: (
            &mut Device,
            (
                &Rows<D, Q, W>,
                &&'k crate::ops::indexing::RowStorage<K, D>,
                &&'v crate::ops::indexing::RowStorage<K, D>,
                &super::cache::Read<'_>,
            ),
        ),
    ) -> Self::Output {
        const {
            assert!(
                H::SIZE > 0
                    && W::SIZE > 0
                    && W::SIZE % H::SIZE == 0
                    && K::SIZE > 0
                    && K::SIZE % (W::SIZE / H::SIZE) == 0
                    && W::SIZE % K::SIZE == 0
            );
        }
        todo!(
            "NPU indexed attention over the first count physical offsets; bias is query-major, empty/all-masked rows return zero, chunks share one softmax denominator"
        )
    }
}

impl<D: Scalar, W: AxisName, K: AxisName, Q: AxisName, H: AxisName>
    Module<(
        &mut Device,
        (
            &Rows<D, Q, W>,
            &super::cache::KeyValue<crate::ops::indexing::RowStorage<K, D>>,
            &super::cache::Read<'_>,
        ),
    )> for Attention<std::marker::PhantomData<H>, Scale>
{
    type Output = Rows<D, Q, W>;

    fn forward(
        &self,
        (device, (query, cache, read)): (
            &mut Device,
            (
                &Rows<D, Q, W>,
                &super::cache::KeyValue<crate::ops::indexing::RowStorage<K, D>>,
                &super::cache::Read<'_>,
            ),
        ),
    ) -> Self::Output {
        self.forward((device, (query, &&cache.0, &&cache.1, read)))
    }
}

#[cfg(test)]
mod tests {
    struct Fixture;
    use super::*;
    use crate::ops::normalization::tests::random;
    use furiosa_opt_std::runtime::launch;

    axes![
        Head = 128,
        Width = 7168,
        Count = 512,
        Shard = 1792,
        Sharded = 2048
    ];

    axes![
        Input = 5376,
        Bundle = 128,
        Double = 256,
        Query = 7168,
        Local = 1792
    ];

    #[device(chip = 1)]
    fn project_joint<R: AxisName, W: AxisName>(
        device: &mut Device,
        projection: &Fused<crate::ops::linear::Linear<Input, Joint<W>>>,
        rows: &Rows<bf16, R, Input>,
    ) -> Rows<bf16, R, Joint<W>> {
        projection.0.forward((device, rows))
    }

    #[device(chip = 1)]
    fn project_parts<R: AxisName, W: AxisName>(
        device: &mut Device,
        projection: &Fused<crate::ops::linear::Linear<Input, Joint<W>>>,
        rows: &Rows<bf16, R, Input>,
    ) -> (Rows<bf16, R, W>, Rows<bf16, R, W>, Rows<bf16, R, W>) {
        projection.forward((device, rows))
    }

    /// Each part a fused projection returns is its columns of the whole projection, bit for bit,
    /// whether a pass computes one part or all three.
    #[tokio::test(flavor = "current_thread")]
    async fn fused_parts_are_the_joint_columns() {
        Fixture::fused_parts::<Bundle, Query>().await;
        Fixture::fused_parts::<Double, Local>().await;
    }

    impl Fixture {
        async fn fused_parts<R: AxisName, W: AxisName>() {
            let _lease = crate::tests::DEVICE.lock().await;
            let mut device = Device::new(project_joint.topology()).unwrap();
            let weight = HostTensor::<bf16, m![{ Symbol<Joint<W>> }, Input]>::from_vec(random(
                Joint::<W>::SIZE * Input::SIZE,
                7,
            ))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            let rows: Rows<bf16, R, Input> =
                HostTensor::<bf16, m![R, Input]>::from_vec(random(R::SIZE * Input::SIZE, 11))
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap();
            let projection = Fused(crate::ops::linear::Linear::new(
                crate::ops::weight::Weight::new(weight),
            ));
            let whole: Vec<bf16> = launch(project_joint, (&mut device, &projection, &rows))
                .await
                .unwrap()
                .to_host::<m![R, { Symbol<Joint<W>> }]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let (query, key, value): (Rows<bf16, R, W>, Rows<bf16, R, W>, Rows<bf16, R, W>) =
                launch(project_parts, (&mut device, &projection, &rows))
                    .await
                    .unwrap();
            for (part, rows) in [query, key, value].into_iter().enumerate() {
                let got = rows
                    .to_host::<m![R, W]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec();
                for (at, value) in got.iter().enumerate() {
                    let (row, column) = (at / W::SIZE, at % W::SIZE);
                    let want = whole[row * Joint::<W>::SIZE + part * W::SIZE + column];
                    assert_eq!(
                        value.to_bits(),
                        want.to_bits(),
                        "part {part} row {row} column {column}"
                    );
                }
            }
        }
    }
    #[device(chip = 1)]
    fn normalized<W: AxisName, C: AxisName>(
        device: &mut Device,
        norm: &HeadNorm<Head>,
        input: &Rows<bf16, C, W>,
    ) -> Rows<bf16, C, W> {
        norm.forward((device, input))
    }

    /// Each head of each row is RMS-normalized alone, as the FP64 definition rounded once to BF16;
    /// a row's heads differ in magnitude by up to 2^9, so a statistic shared across heads fails.
    #[tokio::test(flavor = "current_thread")]
    async fn normalizes_each_head_alone() {
        Fixture::heads_normalize_alone::<Width, Count>().await;
        Fixture::heads_normalize_alone::<Shard, Sharded>().await;
    }

    impl Fixture {
        async fn heads_normalize_alone<Width: AxisName, Count: AxisName>() {
            let _lease = crate::tests::DEVICE.lock().await;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let inputs: Vec<bf16> = random(Count::SIZE * Width::SIZE, 0x2545_f491_4f6c_dd1d)
                .iter()
                .enumerate()
                .map(|(index, v)| {
                    bf16::from_f32(v.to_f32() * (1 << (index / Head::SIZE % 10)) as f32)
                })
                .collect();
            let weight: Vec<f32> = random(Head::SIZE, 0x9e37_79b9_7f4a_7c15)
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let norm = HeadNorm {
                head: std::marker::PhantomData,
                weight: HostTensor::<f32, m![Head]>::from_vec(weight.clone())
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                epsilon: HostTensor::<f32, m![1]>::from_vec(vec![1e-5])
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            };
            let input = HostTensor::<bf16, m![Count, Width]>::from_vec(inputs.clone())
                .to_hbm::<m![1], m![Count, Width]>(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(normalized, (&mut device, &norm, &input))
                .await
                .unwrap()
                .to_host::<m![Count, Width]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (head, values) in inputs.chunks_exact(Head::SIZE).enumerate() {
                let mean = values
                    .iter()
                    .map(|v| f64::from(v.to_f32()).powi(2))
                    .sum::<f64>()
                    / Head::SIZE as f64;
                let inverse = 1.0 / (mean + 1e-5).sqrt();
                for (dim, value) in values.iter().enumerate() {
                    let expected = f64::from(value.to_f32()) * inverse * f64::from(weight[dim]);
                    let got = f64::from(output[head * Head::SIZE + dim].to_f32());
                    assert!(
                        (got - expected).abs() <= expected.abs() / 128.0 + 1e-3,
                        "row {} head {} dim {dim}: {got} vs {expected}",
                        head / (Width::SIZE / Head::SIZE),
                        head % (Width::SIZE / Head::SIZE),
                    );
                }
            }
        }
    }
    // Live rows of 128: text 77 this call, audio 100; video 5,800 of 5,888, or 4,800 of 4,864 so
    // the last query tile is a 1,024-row tail.
    type Sequence = Packed<128, 128, 100, 128, 5800, 5888>;
    type Tailed = Packed<128, 128, 100, 128, 4800, 4864>;
    use crate::ops::attention::flash::{Head as Dims, Span as KeySpan, Visible};
    axes![PairWidth = 256, QuadWidth = 1024, Two = 2, Eight = 8];

    /// The score multiplier of 128-wide heads, `1 / sqrt(128)`.
    const INVERSE_ROOT: usize = (std::f32::consts::FRAC_1_SQRT_2 / 8.0).to_bits() as usize;

    type Scaled128<H> = Attention<std::marker::PhantomData<H>, Scaled<INVERSE_ROOT>>;

    #[device(chip = 1)]
    fn attended<S: AxisName, W: AxisName, H: AxisName, K: AxisName>(
        device: &mut Device,
        attention: &Scaled128<H>,
        (query, key, value): (&Rows<bf16, S, W>, &Rows<bf16, S, W>, &Rows<bf16, S, W>),
        rotation: &Rotation<S>,
        visible: &Visible<K>,
        spans: &crate::compose::Count,
    ) -> Rows<bf16, S, W>
    where
        Attention<std::marker::PhantomData<H>, Scaled<INVERSE_ROOT>>: for<'a> Module<
                (
                    &'a mut Device,
                    (
                        &'a Rows<bf16, S, W>,
                        &'a Rows<bf16, S, W>,
                        &'a Rows<bf16, S, W>,
                        (&'a Rotation<S>, &'a Visible<K>, &'a crate::compose::Count),
                    ),
                ),
                Output = Rows<bf16, S, W>,
            >,
    {
        attention.forward((device, (query, key, value, (rotation, visible, spans))))
    }

    /// Every live row attends the live keys of every span and no pad row, against an FP64 softmax
    /// with BF16-rounded rotations and scores; pad rows are four times larger, so a leak shows.
    #[tokio::test(flavor = "current_thread")]
    async fn packed_rows_attend_the_live_keys() {
        Fixture::packed_rows_attend::<Sequence, PairWidth, Two, Three>(5800, |_| 23).await;
        Fixture::packed_rows_attend::<Tailed, QuadWidth, Eight, Three>(4800, |row| {
            if row < 4096 { 211 } else { 11 }
        })
        .await;
    }

    /// `stride(row)` is the row step of the check from `row`.
    impl Fixture {
        async fn packed_rows_attend<S: AxisName, W: AxisName, H: AxisName, K: AxisName>(
            video: usize,
            stride: impl Fn(usize) -> usize,
        ) where
            Attention<std::marker::PhantomData<H>, Scaled<INVERSE_ROOT>>: for<'a> Module<
                    (
                        &'a mut Device,
                        (
                            &'a Rows<bf16, S, W>,
                            &'a Rows<bf16, S, W>,
                            &'a Rows<bf16, S, W>,
                            (&'a Rotation<S>, &'a Visible<K>, &'a crate::compose::Count),
                        ),
                    ),
                    Output = Rows<bf16, S, W>,
                >,
        {
            const LIVE_TEXT: usize = 77;
            let _lease = crate::tests::DEVICE.lock().await;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let (rows, width, head) = (S::SIZE, W::SIZE, 128);
            let heads = width / head;
            let live = |row: usize| match row {
                r if r < 128 => r < LIVE_TEXT,
                r if r < 256 => r < 228,
                r => r < 256 + video,
            };
            let operand = |seed: u64, gain: f32| -> Vec<bf16> {
                random(rows * width, seed)
                    .into_iter()
                    .enumerate()
                    .map(|(at, x)| {
                        let pad = if live(at / width) { 1.0 } else { 4.0 };
                        bf16::from_f32(x.to_f32() * gain * pad)
                    })
                    .collect()
            };
            let (q, k, v) = (
                operand(0x9e37_79b9_7f4a_7c15, 8.0),
                operand(0x2545_f491_4f6c_dd1d, 1.0),
                operand(0x1405_7b7e_f767_814f, 1.0),
            );
            let angle = |row: usize, i: usize| (row * 7 + i * 13) as f64 * 0.011;
            let half = head / 2;
            let table = |low: bool| -> Vec<bf16> {
                (0..rows * head)
                    .map(|at| {
                        let (row, out, i) = (at / head, at % head / half, at % half);
                        let (c, s) = (angle(row, i).cos() as f32, angle(row, i).sin() as f32);
                        bf16::from_f32(match (low, out) {
                            (true, 0) => c,
                            (true, _) => s,
                            (false, 0) => -s,
                            (false, _) => c,
                        })
                    })
                    .collect()
            };
            let upload = async |device: &mut Device, values: &[bf16]| {
                HostTensor::<bf16, m![S, W]>::from_vec(values.to_vec())
                    .to_hbm::<m![1], m![S, W]>(&mut device.pdma)
                    .await
                    .unwrap()
            };
            let coefficients = async |device: &mut Device, values: Vec<bf16>| {
                HostTensor::<bf16, m![S, Dims]>::from_vec(values)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap()
            };
            let (low, high) = (table(true), table(false));
            let rotation = Rotation {
                low: coefficients(&mut device, low.clone()).await,
                high: coefficients(&mut device, high.clone()).await,
            };
            let visible_keys = video + 100 + LIVE_TEXT;
            let visible: Visible<K> = HostTensor::<bf16, m![K, KeySpan]>::from_vec(
                (0..K::SIZE * 2048)
                    .map(|at| bf16::from_f32(if at < visible_keys { 1.0 } else { 0.0 }))
                    .collect::<Vec<_>>(),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            let spans = crate::compose::tests::count(K::SIZE)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let operands = (
                upload(&mut device, &q).await,
                upload(&mut device, &k).await,
                upload(&mut device, &v).await,
            );
            let attention: Scaled128<H> = Attention {
                heads: std::marker::PhantomData,
                scale: Scaled,
            };
            let got = launch(
                attended,
                (
                    &mut device,
                    &attention,
                    (&operands.0, &operands.1, &operands.2),
                    &rotation,
                    &visible,
                    &spans,
                ),
            )
            .await
            .unwrap()
            .to_host::<m![S, W]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
            let at =
                |t: &[bf16], row: usize, h: usize, d: usize| t[row * width + h * head + d].to_f32();
            // Rotated as the kernel rounds it: once to BF16 from the BF16 coefficients.
            let rotate = |t: &[bf16], row: usize, h: usize| -> Vec<f64> {
                let c = |table: &[bf16], i: usize| f64::from(table[row * head + i].to_f32());
                (0..head)
                    .map(|d| {
                        let (x0, x1) = (
                            f64::from(at(t, row, h, d % half)),
                            f64::from(at(t, row, h, d % half + half)),
                        );
                        let (i, j) = (d % half, d % half + half);
                        let y = if d < half {
                            x0 * c(&low, i) + x1 * c(&high, i)
                        } else {
                            x0 * c(&low, j) + x1 * c(&high, j)
                        };
                        f64::from(bf16::from_f32(y as f32).to_f32())
                    })
                    .collect()
            };
            let scale = f64::from(f32::from_bits(INVERSE_ROOT as u32));
            let keys: Vec<usize> = (0..rows).filter(|&row| live(row)).collect();
            let checked: Vec<usize> = {
                let mut checked = Vec::new();
                let mut row = 0;
                while row < rows {
                    if live(row) {
                        checked.push(row);
                    }
                    row += stride(row);
                }
                checked
            };
            let (mut error, mut norm) = (0.0f64, 0.0f64);
            for h in 0..heads {
                let rotated: Vec<Vec<f64>> = keys.iter().map(|&key| rotate(&k, key, h)).collect();
                for &row in &checked {
                    let query = rotate(&q, row, h);
                    let scores: Vec<f64> = rotated
                        .iter()
                        .map(|key| {
                            let dot: f64 = query.iter().zip(key).map(|(a, b)| a * b).sum();
                            scale * f64::from(bf16::from_f32(dot as f32).to_f32())
                        })
                        .collect();
                    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                    let mass: f64 = weights.iter().sum();
                    for d in 0..head {
                        let want = keys
                            .iter()
                            .zip(&weights)
                            .map(|(&key, w)| w * f64::from(at(&v, key, h, d)))
                            .sum::<f64>()
                            / mass;
                        let found = f64::from(at(&got, row, h, d));
                        assert!(found.is_finite(), "row {row} head {h} channel {d}: {found}");
                        error += (found - want).powi(2);
                        norm += want.powi(2);
                    }
                }
            }
            let relative = (error / norm).sqrt();
            eprintln!("packed attention, {rows} rows of {heads} heads: relative L2 {relative:.3e}");
            assert!(relative < 1e-2, "relative L2 {relative}");
        }
    }
    type Envelope = Text<128, 128>;
    axes![Three = 3];

    #[device(chip = 1)]
    fn text_attended(
        device: &mut Device,
        (query, key, value): (
            &Rows<bf16, Envelope, PairWidth>,
            &Rows<bf16, Envelope, PairWidth>,
            &Rows<bf16, Envelope, PairWidth>,
        ),
        visible: &Visible<Three>,
        spans: &crate::compose::Count,
    ) -> Rows<bf16, Envelope, PairWidth> {
        Attention {
            heads: std::marker::PhantomData::<Two>,
            scale: Scaled::<INVERSE_ROOT>,
        }
        .forward((device, (query, key, value, (visible, spans))))
    }

    /// Every live row of a text envelope attends its live keys alone, against an FP64 softmax with
    /// BF16-rounded scores; pad rows are four times larger, so a leak shows.
    #[tokio::test(flavor = "current_thread")]
    async fn text_rows_attend_the_live_keys() {
        const LIVE: usize = 77;
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(text_attended.topology()).unwrap();
        let (rows, width, head) = (Envelope::SIZE, PairWidth::SIZE, 128);
        let operand = |seed: u64, gain: f32| -> Vec<bf16> {
            random(rows * width, seed)
                .into_iter()
                .enumerate()
                .map(|(at, x)| {
                    let pad = if at / width < LIVE { 1.0 } else { 4.0 };
                    bf16::from_f32(x.to_f32() * gain * pad)
                })
                .collect()
        };
        let (q, k, v) = (
            operand(0x9e37_79b9_7f4a_7c15, 8.0),
            operand(0x2545_f491_4f6c_dd1d, 1.0),
            operand(0x1405_7b7e_f767_814f, 1.0),
        );
        let upload = async |device: &mut Device, values: &[bf16]| {
            HostTensor::<bf16, m![Envelope, PairWidth]>::from_vec(values.to_vec())
                .to_hbm::<m![1], m![Envelope, PairWidth]>(&mut device.pdma)
                .await
                .unwrap()
        };
        let visible: Visible<Three> = HostTensor::<bf16, m![Three, KeySpan]>::from_vec(
            (0..3 * 2048)
                .map(|at| bf16::from_f32(if at < LIVE { 1.0 } else { 0.0 }))
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let spans = crate::compose::tests::count(3)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let operands = (
            upload(&mut device, &q).await,
            upload(&mut device, &k).await,
            upload(&mut device, &v).await,
        );
        let got = launch(
            text_attended,
            (
                &mut device,
                (&operands.0, &operands.1, &operands.2),
                &visible,
                &spans,
            ),
        )
        .await
        .unwrap()
        .to_host::<m![Envelope, PairWidth]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
        let at = |t: &[bf16], row: usize, h: usize, d: usize| {
            f64::from(t[row * width + h * head + d].to_f32())
        };
        let scale = f64::from(f32::from_bits(INVERSE_ROOT as u32));
        let (mut error, mut norm) = (0.0f64, 0.0f64);
        for h in 0..2 {
            for row in 0..LIVE {
                let scores: Vec<f64> = (0..LIVE)
                    .map(|key| {
                        let dot: f64 = (0..head)
                            .map(|d| at(&q, row, h, d) * at(&k, key, h, d))
                            .sum();
                        scale * f64::from(bf16::from_f32(dot as f32).to_f32())
                    })
                    .collect();
                let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                let mass: f64 = weights.iter().sum();
                for d in 0..head {
                    let want = weights
                        .iter()
                        .enumerate()
                        .map(|(key, w)| w * at(&v, key, h, d))
                        .sum::<f64>()
                        / mass;
                    let found = at(&got, row, h, d);
                    assert!(found.is_finite(), "row {row} head {h} channel {d}: {found}");
                    error += (found - want).powi(2);
                    norm += want.powi(2);
                }
            }
        }
        let relative = (error / norm).sqrt();
        eprintln!("text attention: relative L2 {relative:.3e}");
        assert!(relative < 1e-2, "relative L2 {relative}");
    }
}
