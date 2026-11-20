//! Dense attention over caller-supplied visible keys.

use super::flash;
use super::{Rows, Scale};
use crate::Op;
use crate::load::{self, Checkpoint, Load, LoadError};
use crate::ops::collective::Pieces;
use crate::ops::index::gather::RowStorage;
use crate::ops::index::packed;
use crate::ops::index::table::KeyValue;
use crate::ops::index::table::Read;
use crate::ops::linear;
use crate::ops::linear::Linear;
use crate::ops::normalization;
use crate::ops::rotary;
use crate::ops::rotary::Rotary;
use crate::tensor::memory::residency::Residency;
use crate::tensor::parallel::Chip;
use crate::tensor::scalar;
use crate::tensor::scalar::F32;
use core::borrow::Borrow;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::HostBuffer;

/// The attention op, as the FP8 execution form's files implement it from here.
pub(crate) use super::Attention;
pub use grouped::Joined;
pub use pack::{PackInputs, PackKeys, PackPlan};

mod grouped;
pub mod head;
pub mod pack;
pub mod paged;
mod text;

/// A row's query, key and value columns, `W` each, in that order.
#[derive(Clone, Debug)]
pub struct Joint<W>(PhantomData<W>);

impl<W: AxisName> AxisName for Joint<W> {
    const NAME: Ident = Ident::new("Joint");
    const SIZE: usize = 3 * W::SIZE;
}

/// Rows projected once into query, key and value columns, each returned as rows; a load orders the
/// head-major `weight` by part, pairing query and key features by `Rotated`, each chip its heads.
#[derive(DeviceSend, Residency)]
#[residency(children(L))]
pub struct Projection<L, Head, Rotated = Head>(pub L, PhantomData<fn() -> (Head, Rotated)>);

impl<L, Head, Rotated> Projection<L, Head, Rotated> {
    pub const fn new(projection: L) -> Self {
        Self(projection, PhantomData)
    }
}

impl<L, Head: AxisName, Rotated: AxisName> Projection<L, Head, Rotated> {
    /// The weight at the context's position in the leaf's order on every chip of `Chips`: the
    /// chip's heads' query rows, then their key and value rows. A stored tensor of the leaf's heads
    /// is every chip's; one of `Chips` times as many gives chip `c` its `c`-th part of them.
    // TODO: read the rows only for a leaf that holds bytes, as a converted leaf's build does.
    fn joint<In: AxisName, W: AxisName, Chips: M>(
        source: &Checkpoint,
        context: &mut load::Context<'_>,
    ) -> Result<HostTensor<bf16, m![{ Chips }, { m![{ Symbol<Joint<W>> }, In] }]>, LoadError> {
        const { assert!(W::SIZE.is_multiple_of(Head::SIZE)) };
        let (name, view) = context.tensor(source)?;
        let invalid = |reason: String| LoadError::InvalidData {
            name: name.clone(),
            reason,
        };
        let rows = Joint::<W>::SIZE;
        let shared = match view.shape() {
            [stored, columns] if *columns == In::SIZE && *stored == rows => true,
            [stored, columns] if *columns == In::SIZE && *stored == Chips::SIZE * rows => false,
            stored => {
                return Err(invalid(format!(
                    "shape {stored:?}, the leaf holds {rows} rows of {} on each of {} chips",
                    In::SIZE,
                    Chips::SIZE
                )));
            }
        };
        let data = Checkpoint::bf16(&view, &name)?;
        let (heads, row) = (W::SIZE / Head::SIZE, In::SIZE * size_of::<bf16>());
        let mut joint = vec![0; Chips::SIZE * rows * row];
        for (chip, image) in joint.chunks_exact_mut(rows * row).enumerate() {
            let first = if shared { 0 } else { chip * heads };
            for head in 0..heads {
                for part in 0..3 {
                    for d in 0..Head::SIZE {
                        let to = match part {
                            2 => d,
                            _ => Rotary::<Head, Rotated>::paired(d),
                        };
                        let from = (((first + head) * 3 + part) * Head::SIZE + d) * row;
                        let at = ((part * heads + head) * Head::SIZE + to) * row;
                        image[at..][..row].copy_from_slice(&data[from..][..row]);
                    }
                }
            }
        }
        let image = furiosa_opt_std::runtime::HostBuffer::from_vec(joint);
        HostTensor::try_from(image).map_err(|error| invalid(error.to_string()))
    }
}

/// The weight under `weight`, in the leaf's order.
impl<'a, In: AxisName, W: AxisName, Chips, T, Head: AxisName, Rotated: AxisName>
    Load<&'a Checkpoint> for Projection<Linear<In, Joint<W>, Chips, T>, Head, Rotated>
where
    Chips: M + 'static,
    Linear<In, Joint<W>, Chips, T>:
        Load<HostTensor<bf16, m![{ Chips }, { m![{ Symbol<Joint<W>> }, In] }]>>,
{
    fn load(context: &mut load::Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("0", "weight", |cx| {
                let joint = Self::joint::<In, W, Chips>(from, cx)?;
                Ok(Self::new(Load::load(cx, joint)?))
            })
        })
    }
}

/// The weight written in place, one rule of an update.
impl<'t, 's, In: AxisName, W: AxisName, Chips, T, Head: AxisName, Rotated: AxisName>
    Load<(
        &'t mut Projection<Linear<In, Joint<W>, Chips, T>, Head, Rotated>,
        &'s Checkpoint,
    )> for &'t mut Projection<Linear<In, Joint<W>, Chips, T>, Head, Rotated>
where
    Chips: M + 'static,
    for<'u> &'u mut Linear<In, Joint<W>, Chips, T>: Load<(
        &'u mut Linear<In, Joint<W>, Chips, T>,
        HostTensor<bf16, m![{ Chips }, { m![{ Symbol<Joint<W>> }, In] }]>,
    )>,
{
    fn load(
        context: &mut load::Context<'_>,
        (into, from): (
            &'t mut Projection<Linear<In, Joint<W>, Chips, T>, Head, Rotated>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("0", "weight", |cx| {
                cx.rule(
                    |cx| {
                        Projection::<Linear<In, Joint<W>, Chips, T>, Head, Rotated>::joint(from, cx)
                    },
                    |cx, joint| {
                        <&mut Linear<In, Joint<W>, Chips, T> as Load<_>>::load(
                            cx,
                            (&mut into.0, joint),
                        )
                        .map(drop)
                    },
                )
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

impl<'a, L, K, Head, Rotated> Load<&'a Projection<K, Head, Rotated>>
    for Projection<L, Head, Rotated>
where
    L: Load<&'a K>,
{
    fn load(
        context: &mut load::Context<'_>,
        from: &'a Projection<K, Head, Rotated>,
    ) -> Result<Self, LoadError> {
        context
            .field("0", "", |cx| L::load(cx, &from.0))
            .map(Self::new)
    }
}

impl<'t, L, Head, Rotated>
    Load<(
        &'t mut Projection<L, Head, Rotated>,
        Projection<L, Head, Rotated>,
    )> for &'t mut Projection<L, Head, Rotated>
where
    for<'u> &'u mut L: Load<(&'u mut L, L)>,
{
    fn load(
        context: &mut load::Context<'_>,
        (into, from): (
            &'t mut Projection<L, Head, Rotated>,
            Projection<L, Head, Rotated>,
        ),
    ) -> Result<Self, LoadError> {
        context.field("0", "", |cx| {
            <&mut L as Load<_>>::load(cx, (&mut into.0, from.0))
        })?;
        Ok(into)
    }
}

impl<'x, In: AxisName, W: AxisName, R: AxisName, Chips: M, T, Head, Rotated>
    Op<&'x HbmTensor<bf16, Chips, m![R, In]>>
    for Projection<Linear<In, Joint<W>, Chips, T>, Head, Rotated>
where
    T: Borrow<HbmTensor<bf16, Chips, m![{ Symbol<Joint<W>> }, In]>>,
{
    type Output = (
        HbmTensor<bf16, Chips, m![R, W]>,
        HbmTensor<bf16, Chips, m![R, W]>,
        HbmTensor<bf16, Chips, m![R, W]>,
    );

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, Chips, m![R, In]>,
    ) -> Self::Output {
        let (mut query, mut key, mut value) =
            (HbmTensor::new(), HbmTensor::new(), HbmTensor::new());
        linear::Projection::<In, Joint<W>, R, Chips, ()>::new(self.0.tensor(), &())
            .forward(&mut *device, (input, (&mut query, &mut key, &mut value)));
        (query, key, value)
    }
}

/// A packed sequence's text, audio and video spans, each with its live rows.
pub type Packed<
    const TEXT_LIVE: usize,
    const TEXT_ROWS: usize,
    const AUDIO_LIVE: usize,
    const AUDIO_ROWS: usize,
    const VIDEO_LIVE: usize,
    const VIDEO_ROWS: usize,
> = packed::Packed<
    packed::Span<TEXT_LIVE, TEXT_ROWS>,
    packed::Span<AUDIO_LIVE, AUDIO_ROWS>,
    packed::Span<VIDEO_LIVE, VIDEO_ROWS>,
>;

/// A packed sequence's parts on `N` chips, each its share of every span's rows: every part but the
/// last holds live rows only, and the last `TEXT_LIVE`, `AUDIO_LIVE` and `VIDEO_LIVE` live rows.
pub type Split<
    const TEXT_LIVE: usize,
    const TEXT_ROWS: usize,
    const AUDIO_LIVE: usize,
    const AUDIO_ROWS: usize,
    const VIDEO_LIVE: usize,
    const VIDEO_ROWS: usize,
    const N: usize,
> = Pieces<Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>, N>;

/// Each row's rotation of a 128-wide head.
type Rotation<R, C = Chip> = rotary::Rotation<flash::Head, C, R>;

/// A context: each row's rotation, the keys' visibility and the live keys.
type Context<'a, R, K, Chips> = (
    &'a Rotation<R, Chips>,
    &'a flash::Visible<K, Chips>,
    &'a scalar::Value<Chips>,
);

/// Bidirectional attention of a packed text, audio and video sequence over its visible keys, each
/// row rotated by its own rotation, 4,096 query rows at a time; a pad row's output is unspecified.
impl<
    'a,
    const TEXT_LIVE: usize,
    const TEXT_ROWS: usize,
    const AUDIO_LIVE: usize,
    const AUDIO_ROWS: usize,
    const VIDEO_LIVE: usize,
    const VIDEO_ROWS: usize,
    W: AxisName,
    Heads: AxisName,
    K: AxisName,
    Chips: M,
    const BITS: usize,
>
    Op<(
        &Rows<
            bf16,
            Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>,
            W,
            Chips,
        >,
        &Rows<
            bf16,
            Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>,
            W,
            Chips,
        >,
        &Rows<
            bf16,
            Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>,
            W,
            Chips,
        >,
        Context<
            'a,
            Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>,
            K,
            Chips,
        >,
    )> for Attention<PhantomData<Heads>, F32<BITS>>
{
    type Output = Rows<
        bf16,
        Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>,
        W,
        Chips,
    >;

    fn forward(
        &self,
        device: &mut Device,
        (query, key, value, (rotation, visible, live)): (
            &Rows<
                bf16,
                Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>,
                W,
                Chips,
            >,
            &Rows<
                bf16,
                Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>,
                W,
                Chips,
            >,
            &Rows<
                bf16,
                Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>,
                W,
                Chips,
            >,
            Context<
                'a,
                Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>,
                K,
                Chips,
            >,
        ),
    ) -> Self::Output {
        const {
            assert!(
                Heads::SIZE == 2 * flash::HeadPairs::<W>::SIZE,
                "a chip attends its heads in pairs"
            )
        };
        let (query, rotated) = Rotary::<super::flash::Head, super::flash::Head, bf16, Chips>::new()
            .forward(&mut *device, (&(query, key), rotation));
        let keys = Keys::<K>::default().forward(&mut *device, &rotated);
        drop(rotated);
        let values = Keys::<K>::default().forward(&mut *device, value);
        // SAFETY: a row is its heads in order, two a pair.
        let query: flash::Queries<
            Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>,
            flash::HeadPairs<W>,
            Chips,
        > = unsafe { query.reshape() };
        let output = self.forward(device, (&query, &keys, &values, (visible, live)));
        // SAFETY: a row is its heads in order, two a pair.
        unsafe { output.reshape() }
    }
}

/// Bidirectional attention of a packed sequence split over the chips, as for one sequence, with
/// every part's video, audio and text rows written as keys in that order.
impl<
    'a,
    const TEXT_LIVE: usize,
    const TEXT_ROWS: usize,
    const AUDIO_LIVE: usize,
    const AUDIO_ROWS: usize,
    const VIDEO_LIVE: usize,
    const VIDEO_ROWS: usize,
    const N: usize,
    W: AxisName,
    Heads: AxisName,
    K: AxisName,
    Chips: M,
    const BITS: usize,
>
    Op<(
        &Rows<
            bf16,
            Split<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS, N>,
            W,
            Chips,
        >,
        &Rows<
            bf16,
            Split<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS, N>,
            W,
            Chips,
        >,
        &Rows<
            bf16,
            Split<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS, N>,
            W,
            Chips,
        >,
        Context<
            'a,
            Split<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS, N>,
            K,
            Chips,
        >,
    )> for Attention<PhantomData<Heads>, F32<BITS>>
{
    type Output = Rows<
        bf16,
        Split<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS, N>,
        W,
        Chips,
    >;

    fn forward(
        &self,
        device: &mut Device,
        (query, key, value, (rotation, visible, live)): (
            &Rows<
                bf16,
                Split<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS, N>,
                W,
                Chips,
            >,
            &Rows<
                bf16,
                Split<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS, N>,
                W,
                Chips,
            >,
            &Rows<
                bf16,
                Split<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS, N>,
                W,
                Chips,
            >,
            Context<
                'a,
                Split<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS, N>,
                K,
                Chips,
            >,
        ),
    ) -> Self::Output {
        const {
            assert!(
                Heads::SIZE == 2 * flash::HeadPairs::<W>::SIZE,
                "a chip attends its heads in pairs"
            )
        };
        let (query, rotated) = Rotary::<super::flash::Head, super::flash::Head, bf16, Chips>::new()
            .forward(&mut *device, (&(query, key), rotation));
        let keys = Keys::<K>::default().forward(&mut *device, &rotated);
        drop(rotated);
        let values = Keys::<K>::default().forward(&mut *device, value);
        // SAFETY: a row is its heads in order, two a pair.
        let query: flash::Queries<
            Split<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS, N>,
            flash::HeadPairs<W>,
            Chips,
        > = unsafe { query.reshape() };
        let output = self.forward(device, (&query, &keys, &values, (visible, live)));
        // SAFETY: a row is its heads in order, two a pair.
        unsafe { output.reshape() }
    }
}

/// A packed sequence's live rows as keys or values, video, audio, then the text envelope; copies
/// of the envelope's rows fill the last span, so every absent key is finite.
struct Keys<K>(PhantomData<K>);

impl<K> Default for Keys<K> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

#[derive(Clone, Debug)]
struct KeyRows<Spans>(PhantomData<Spans>);

impl<Spans: AxisName> AxisName for KeyRows<Spans> {
    const NAME: Ident = Ident::new("Keys");
    const SIZE: usize = Spans::SIZE * flash::Span::SIZE;
}

#[derive(Clone, Debug)]
struct Run<const N: usize>;

impl<const N: usize> AxisName for Run<N> {
    const NAME: Ident = Ident::new("Run");
    const SIZE: usize = N;
}

struct Copy<Spans, const N: usize>(PhantomData<Spans>);

impl<Spans: AxisName, const N: usize, R: AxisName, W: AxisName, Chips: M>
    Op<(
        (&HbmTensor<bf16, Chips, m![R, W]>, usize),
        (&mut flash::Tokens<Spans, flash::HeadPairs<W>, Chips>, usize),
    )> for Copy<Spans, N>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        ((rows, from), (tokens, to)): (
            (&HbmTensor<bf16, Chips, m![R, W]>, usize),
            (&mut flash::Tokens<Spans, flash::HeadPairs<W>, Chips>, usize),
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
        // SAFETY: the tile's keys are the run's in order; only the key axis is renamed.
        let target = unsafe {
            target.reshape::<Chips, m![{ P<W> }, Duo, { L<N> } #{!} ({ K<Spans> }), Head]>()
        };
        source.to_hbm_view(&mut device.tdma, target);
    }
}

impl<
    const TEXT_LIVE: usize,
    const TEXT_ROWS: usize,
    const AUDIO_LIVE: usize,
    const AUDIO_ROWS: usize,
    const VIDEO_LIVE: usize,
    const VIDEO_ROWS: usize,
    W: AxisName,
    K: AxisName,
    Chips: M,
>
    Op<
        &Rows<
            bf16,
            Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>,
            W,
            Chips,
        >,
    > for Keys<K>
{
    type Output = flash::Tokens<K, flash::HeadPairs<W>, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        rows: &Rows<
            bf16,
            Packed<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS>,
            W,
            Chips,
        >,
    ) -> Self::Output {
        use super::flash;
        let (text, end) = (VIDEO_LIVE + AUDIO_LIVE, K::SIZE * flash::Span::SIZE);
        const {
            let end = K::SIZE * flash::Span::SIZE;
            assert!(
                end >= VIDEO_LIVE + AUDIO_LIVE + TEXT_ROWS
                    && end <= VIDEO_LIVE + AUDIO_LIVE + TEXT_ROWS + AUDIO_ROWS + VIDEO_ROWS,
                "the keys past the text envelope continue the packed rows"
            )
        };
        let mut tokens = HbmTensor::new();
        Copy::<K, VIDEO_LIVE>(PhantomData)
            .forward(device, ((rows, TEXT_ROWS + AUDIO_ROWS), (&mut tokens, 0)));
        Copy::<K, AUDIO_LIVE>(PhantomData)
            .forward(device, ((rows, TEXT_ROWS), (&mut tokens, VIDEO_LIVE)));
        for run in 0..(end - text).div_ceil(TEXT_ROWS) {
            let to = (text + run * TEXT_ROWS).min(end - TEXT_ROWS);
            Copy::<K, TEXT_ROWS>(PhantomData)
                .forward(device, ((rows, to - text), (&mut tokens, to)));
        }
        tokens
    }
}

/// The parts' live rows as keys or values: every part's video rows, then audio, then text; copies
/// of audio rows fill the last span, so every absent key is finite.
impl<
    const TEXT_LIVE: usize,
    const TEXT_ROWS: usize,
    const AUDIO_LIVE: usize,
    const AUDIO_ROWS: usize,
    const VIDEO_LIVE: usize,
    const VIDEO_ROWS: usize,
    const N: usize,
    W: AxisName,
    K: AxisName,
    Chips: M,
>
    Op<
        &Rows<
            bf16,
            Split<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS, N>,
            W,
            Chips,
        >,
    > for Keys<K>
{
    type Output = flash::Tokens<K, flash::HeadPairs<W>, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        rows: &Rows<
            bf16,
            Split<TEXT_LIVE, TEXT_ROWS, AUDIO_LIVE, AUDIO_ROWS, VIDEO_LIVE, VIDEO_ROWS, N>,
            W,
            Chips,
        >,
    ) -> Self::Output {
        use super::flash;
        let (part, end) = (
            TEXT_ROWS + AUDIO_ROWS + VIDEO_ROWS,
            K::SIZE * flash::Span::SIZE,
        );
        let (video, audio) = (
            (N - 1) * VIDEO_ROWS + VIDEO_LIVE,
            (N - 1) * AUDIO_ROWS + AUDIO_LIVE,
        );
        const {
            let end = K::SIZE * flash::Span::SIZE;
            let keys =
                (N - 1) * (VIDEO_ROWS + AUDIO_ROWS) + VIDEO_LIVE + AUDIO_LIVE + N * TEXT_ROWS;
            assert!(
                end >= keys && end - keys <= N * (TEXT_ROWS + AUDIO_ROWS + VIDEO_ROWS),
                "the keys past the text envelope are rows"
            );
        };
        let mut tokens = HbmTensor::new();
        #[unroll]
        for at in 0..N - 1 {
            Copy::<K, VIDEO_ROWS>(PhantomData).forward(
                device,
                (
                    (rows, at * part + TEXT_ROWS + AUDIO_ROWS),
                    (&mut tokens, at * VIDEO_ROWS),
                ),
            );
            Copy::<K, AUDIO_ROWS>(PhantomData).forward(
                device,
                (
                    (rows, at * part + TEXT_ROWS),
                    (&mut tokens, video + at * AUDIO_ROWS),
                ),
            );
        }
        let last = (N - 1) * part;
        Copy::<K, VIDEO_LIVE>(PhantomData).forward(
            device,
            (
                (rows, last + TEXT_ROWS + AUDIO_ROWS),
                (&mut tokens, video - VIDEO_LIVE),
            ),
        );
        Copy::<K, AUDIO_LIVE>(PhantomData).forward(
            device,
            (
                (rows, last + TEXT_ROWS),
                (&mut tokens, video + audio - AUDIO_LIVE),
            ),
        );
        #[unroll]
        for at in 0..N {
            Copy::<K, TEXT_ROWS>(PhantomData).forward(
                device,
                (
                    (rows, at * part),
                    (&mut tokens, video + audio + at * TEXT_ROWS),
                ),
            );
        }
        let text = video + audio + N * TEXT_ROWS;
        for run in 0..(end - text).div_ceil(AUDIO_ROWS) {
            let to = (text + run * AUDIO_ROWS).min(end - AUDIO_ROWS);
            Copy::<K, AUDIO_ROWS>(PhantomData)
                .forward(device, ((rows, to - text), (&mut tokens, to)));
        }
        tokens
    }
}

/// One span's rows, as a text envelope holds them.
type Text<const LIVE: usize, const ROWS: usize> = packed::Span<LIVE, ROWS>;

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
    Op<(
        &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
        &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
        &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
        (&flash::Visible<K, Chips>, &scalar::Value<Chips>),
    )> for Attention<PhantomData<Heads>, F32<BITS>>
{
    type Output = Rows<bf16, Text<LIVE, ROWS>, W, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (query, key, value, (visible, live)): (
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            (&flash::Visible<K, Chips>, &scalar::Value<Chips>),
        ),
    ) -> Self::Output {
        text::attend::<LIVE, ROWS, W, Heads, K, Chips, _>(
            device,
            self,
            (query, key, value, (visible, live)),
        )
    }
}

/// The same finite text-envelope preparation, with Compact selected explicitly.
impl<
    const LIVE: usize,
    const ROWS: usize,
    W: AxisName,
    Heads: AxisName,
    K: AxisName,
    Chips: M,
    const BITS: usize,
>
    Op<(
        &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
        &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
        &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
        (&flash::Visible<K, Chips>, &scalar::Value<Chips>),
    )> for flash::Compact<Heads, BITS>
{
    type Output = Rows<bf16, Text<LIVE, ROWS>, W, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (query, key, value, (visible, live)): (
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            (&flash::Visible<K, Chips>, &scalar::Value<Chips>),
        ),
    ) -> Self::Output {
        text::attend::<LIVE, ROWS, W, Heads, K, Chips, _>(
            device,
            self,
            (query, key, value, (visible, live)),
        )
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
    Op<(
        &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
        &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
        &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
        &'c mut (&'v flash::Visible<K, Chips>, &'n scalar::Value<Chips>),
    )> for Attention<PhantomData<Heads>, F32<BITS>>
{
    type Output = Rows<bf16, Text<LIVE, ROWS>, W, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (query, key, value, context): (
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            &Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
            &'c mut (&'v flash::Visible<K, Chips>, &'n scalar::Value<Chips>),
        ),
    ) -> Self::Output {
        self.forward(device, (query, key, value, *context))
    }
}

/// Each head normalized alone over flattened feature columns, its `weight` gain in the order the
/// `Rotated` rotation pairs the features; `EPSILON` holds epsilon's FP32 bits, which a load places.
#[derive(DeviceSend, Residency)]
#[residency(children(Parameter, Epsilon))]
pub struct HeadNorm<
    Head: AxisName,
    Chips: M = m![1],
    Parameter = HbmTensor<f32, Chips, m![Head]>,
    Epsilon = HbmTensor<f32, Chips, m![1]>,
    const EPSILON: usize = { normalization::EPSILON },
    Rotated = Head,
> {
    pub weight: Parameter,
    pub epsilon: Epsilon,
    pub head: PhantomData<(Head, Chips, Rotated)>,
}

/// Head-major BF16 rows scaled as `(x * gain) / sqrt(mean(x²) + epsilon)`.
/// The wrapped head norm supplies its existing gain and epsilon; division stays in the final
/// scaling pass, retaining that arithmetic's BF16 rounding rather than multiplying a reciprocal.
#[derive(DeviceSend)]
pub struct Division<Norm>(pub Norm);

/// The rotation-ordered gain of a head norm, on every chip of `C`.
type Gain<Head, C> = HostTensor<f32, m![{ C }, { m![Head] }]>;

impl<
    Head: AxisName,
    Chips: M + 'static,
    Parameter,
    Epsilon,
    const EPSILON: usize,
    Rotated: AxisName,
> HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>
{
    /// The stored gain at the context's name, feature `d` at `Rotary::paired(d)`.
    fn gain(
        from: &Checkpoint,
        context: &mut load::Context<'_>,
    ) -> Result<Gain<Head, Chips>, LoadError> {
        let stored = context.floats(from, &[Head::SIZE])?;
        let mut ordered = vec![0.0; Head::SIZE];
        for (d, value) in stored.into_iter().enumerate() {
            ordered[Rotary::<Head, Rotated>::paired(d)] = value;
        }
        Ok(HostTensor::from_vec(ordered.repeat(Chips::SIZE)))
    }
}

/// The gain under `weight` in the rotation's order; the epsilon placed from `EPSILON` on every
/// chip.
impl<'a, Head: AxisName, Chips, Parameter, Epsilon, const EPSILON: usize, Rotated: AxisName>
    Load<&'a Checkpoint> for HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>
where
    Chips: M + 'static,
    Parameter: Load<Gain<Head, Chips>>,
    Epsilon: Load<HostTensor<f32, m![{ Chips }, { m![1] }]>>,
{
    fn load(context: &mut load::Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let weight = cx.field("weight", "weight", |cx| {
                let gain = Self::gain(from, cx)?;
                Parameter::load(cx, gain)
            })?;
            let epsilon = cx.field("epsilon", "epsilon", |cx| {
                Epsilon::load(cx, F32::<EPSILON>::host::<Chips, m![1]>())
            })?;
            Ok(Self {
                weight,
                epsilon,
                head: PhantomData,
            })
        })
    }
}

/// The gain ordered again once it arrives, one rule of an update; the epsilon is no update target.
impl<'t, 's, Head: AxisName, Chips, Parameter, Epsilon, const EPSILON: usize, Rotated: AxisName>
    Load<(
        &'t mut HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>,
        &'s Checkpoint,
    )> for &'t mut HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>
where
    Chips: M + 'static,
    for<'u> &'u mut Parameter: Load<(&'u mut Parameter, Gain<Head, Chips>)>,
{
    fn load(
        context: &mut load::Context<'_>,
        (into, from): (
            &'t mut HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("weight", "weight", |cx| {
                cx.rule(
                    |cx| {
                        HeadNorm::<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>::gain(
                            from, cx,
                        )
                    },
                    |cx, host| {
                        <&mut Parameter as Load<_>>::load(cx, (&mut into.weight, host)).map(drop)
                    },
                )
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

impl<'a, Head: AxisName, Chips: M, Parameter, Epsilon, Q, F, const EPSILON: usize, Rotated>
    Load<&'a HeadNorm<Head, Chips, Q, F, EPSILON, Rotated>>
    for HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>
where
    Parameter: Load<&'a Q>,
    Epsilon: Load<&'a F>,
{
    fn load(
        context: &mut load::Context<'_>,
        from: &'a HeadNorm<Head, Chips, Q, F, EPSILON, Rotated>,
    ) -> Result<Self, LoadError> {
        Ok(Self {
            weight: context.field("weight", "weight", |cx| Parameter::load(cx, &from.weight))?,
            epsilon: context.field("epsilon", "epsilon", |cx| Epsilon::load(cx, &from.epsilon))?,
            head: PhantomData,
        })
    }
}

impl<'t, Head: AxisName, Chips: M, Parameter, Epsilon, const EPSILON: usize, Rotated>
    Load<(
        &'t mut HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>,
        HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>,
    )> for &'t mut HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>
where
    for<'u> &'u mut Parameter: Load<(&'u mut Parameter, Parameter)>,
    for<'u> &'u mut Epsilon: Load<(&'u mut Epsilon, Epsilon)>,
{
    fn load(
        context: &mut load::Context<'_>,
        (into, from): (
            &'t mut HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>,
            HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>,
        ),
    ) -> Result<Self, LoadError> {
        context.field("weight", "weight", |cx| {
            <&mut Parameter as Load<_>>::load(cx, (&mut into.weight, from.weight))
        })?;
        context.field("epsilon", "epsilon", |cx| {
            <&mut Epsilon as Load<_>>::load(cx, (&mut into.epsilon, from.epsilon))
        })?;
        Ok(into)
    }
}

impl<Head: AxisName, Width: AxisName, Count: AxisName> Op<&Rows<f32, Count, Width>>
    for HeadNorm<Head>
{
    type Output = Rows<f32, Count, Width>;

    fn forward(&self, _: &mut Device, _: &Rows<f32, Count, Width>) -> Self::Output {
        const { assert!(Head::SIZE > 0 && Width::SIZE % Head::SIZE == 0) };
        todo!("NPU per-head RMS normalization with shared head-width weight and FP32 statistics")
    }
}

/// [`HeadNorm`] over rows that come in no whole norm tile, such as one decoded row: a typed
/// contract without a kernel.
#[derive(DeviceSend)]
pub struct Untiled<Head: AxisName>(pub HeadNorm<Head>);

impl<D: Scalar, Head: AxisName, Width: AxisName, Count: AxisName> Op<&Rows<D, Count, Width>>
    for Untiled<Head>
{
    type Output = Rows<D, Count, Width>;

    fn forward(&self, _: &mut Device, _: &Rows<D, Count, Width>) -> Self::Output {
        const { assert!(Head::SIZE > 0 && Width::SIZE % Head::SIZE == 0) };
        todo!("NPU per-head RMS normalization of rows that come in no whole norm tile")
    }
}

/// Head rows of a row sequence in HBM.
pub(crate) mod heads {
    use crate::ops::normalization::{self, Root, Scale, Squares};
    use crate::tensor::live::Live;
    use core::borrow::Borrow;
    use furiosa_opt_std::prelude::*;
    use std::marker::PhantomData;

    /// The heads of a row of `Width` features, `Head` features each.
    #[derive(Clone, Debug)]
    pub(crate) struct Heads<Width, Head>(PhantomData<(Width, Head)>);

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
    pub(crate) struct Padded<Width, Head>(PhantomData<(Width, Head)>);

    impl<Width: AxisName, Head: AxisName> AxisName for Padded<Width, Head> {
        const NAME: Ident = Ident::new("Padded");
        const SIZE: usize = Heads::<Width, Head>::SIZE.next_power_of_two();
    }

    /// Rows one slice holds of a norm tile.
    #[derive(Clone, Debug)]
    pub(crate) struct Lanes<Count, Width, Head>(PhantomData<(Count, Width, Head)>);

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
    pub(crate) struct Halves<Count, Width, Head>(PhantomData<(Count, Width, Head)>);

    impl<Count: AxisName, Width: AxisName, Head: AxisName> AxisName for Halves<Count, Width, Head> {
        const NAME: Ident = Ident::new("NormHalves");
        const SIZE: usize = 256 / Padded::<Width, Head>::SIZE * Lanes::<Count, Width, Head>::SIZE;
    }

    /// Rows of a norm tile.
    #[derive(Clone, Debug)]
    pub(crate) struct Block<Count, Width, Head>(PhantomData<(Count, Width, Head)>);

    impl<Count: AxisName, Width: AxisName, Head: AxisName> AxisName for Block<Count, Width, Head> {
        const NAME: Ident = Ident::new("NormBlock");
        const SIZE: usize = 2 * Halves::<Count, Width, Head>::SIZE;
    }

    /// The norm tiles of the rows, in row order.
    #[derive(Clone, Debug)]
    pub(crate) struct Blocks<Count, Width, Head>(PhantomData<(Count, Width, Head)>);

    impl<Count: AxisName, Width: AxisName, Head: AxisName> AxisName for Blocks<Count, Width, Head> {
        const NAME: Ident = Ident::new("NormBlocks");
        const SIZE: usize = Count::SIZE / Block::<Count, Width, Head>::SIZE;
    }

    type P<W, Head> = Symbol<Padded<W, Head>>;
    type L<C, W, Head> = Symbol<Lanes<C, W, Head>>;
    type Hb<C, W, Head> = Symbol<Halves<C, W, Head>>;
    type B<C, W, Head> = Symbol<Block<C, W, Head>>;
    type Bs<C, W, Head> = Symbol<Blocks<C, W, Head>>;
    type Blocked<C, W, Head> = m![{ Bs<C, W, Head> }, { B<C, W, Head> }, { H<W, Head> }, Head];
    /// A slice's rows of a norm tile.
    type Rows<C, W, Head> = m![{ B<C, W, Head> } % ({ L<C, W, Head> })];
    /// How `Q` rows of heads move through a norm's passes.
    type Norm<Q, Head> = normalization::Lanes<
        m![{ Q }, Head / 8],
        m![Head % 8],
        m![{ Q }, Head / 8, Head % 8 / 4 % 2],
        m![Head % 4],
    >;
    type Cluster<C, W, Head> = m![{ B<C, W, Head> } / ({ Hb<C, W, Head> })];
    /// Heads by row groups: a row's heads over the slices of its group.
    type Groups<C, W, Head> = m![
        { H<W, Head> } # ({ P<W, Head> }),
        { B<C, W, Head> } % ({ Hb<C, W, Head> }) / ({ L<C, W, Head> })
    ];
    type Part<C, W, Head, X = m![1]> = DmTensor<
        bf16,
        X,
        Cluster<C, W, Head>,
        Groups<C, W, Head>,
        m![{ B<C, W, Head> } % ({ L<C, W, Head> }), Head],
    >;
    type Statistic<C, W, Head, X = m![1]> = DmTensor<
        f32,
        X,
        Cluster<C, W, Head>,
        Groups<C, W, Head>,
        m![{ B<C, W, Head> } % ({ L<C, W, Head> }), 1 # 8],
    >;
    type Scalar<C, W, Head, X = m![1]> =
        VrfTensor<f32, X, Cluster<C, W, Head>, Groups<C, W, Head>, m![1 # 8]>;
    type RowOperand<C, W, Head, X = m![1]> = VrfTensor<
        f32,
        X,
        Cluster<C, W, Head>,
        Groups<C, W, Head>,
        m![{ B<C, W, Head> } % ({ L<C, W, Head> }), 1 # 8],
    >;

    impl<
        Head: AxisName,
        Width: AxisName,
        Count: AxisName,
        Chips: M,
        Parameter,
        Epsilon,
        const EPSILON: usize,
        Rotated,
    > crate::Op<&HbmTensor<bf16, Chips, m![Count, Width]>>
        for super::HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>
    where
        Parameter: Borrow<HbmTensor<f32, Chips, m![Head]>>,
        Epsilon: Borrow<HbmTensor<f32, Chips, m![1]>>,
    {
        type Output = HbmTensor<bf16, Chips, m![Count, Width]>;

        fn forward(
            &self,
            device: &mut Device,
            input: &HbmTensor<bf16, Chips, m![Count, Width]>,
        ) -> Self::Output {
            crate::Op::forward(self, device, Live::<(), Count>::all().with(input)).into_value()
        }
    }

    impl<
        Head: AxisName,
        Width: AxisName,
        Count: AxisName,
        Chips: M,
        Parameter,
        Epsilon,
        const EPSILON: usize,
        Rotated,
    > crate::Op<Live<&HbmTensor<bf16, Chips, m![Count, Width]>, Count>>
        for super::HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>
    where
        Parameter: Borrow<HbmTensor<f32, Chips, m![Head]>>,
        Epsilon: Borrow<HbmTensor<f32, Chips, m![1]>>,
    {
        type Output = Live<HbmTensor<bf16, Chips, m![Count, Width]>, Count>;

        fn forward(
            &self,
            device: &mut Device,
            input: Live<&HbmTensor<bf16, Chips, m![Count, Width]>, Count>,
        ) -> Self::Output {
            let mut output = HbmTensor::new();
            self.write_live(device, input.with(*input.value()), &mut output);
            input.with(output)
        }
    }

    impl<Head: AxisName, Chips: M, Parameter, Epsilon, const EPSILON: usize, Rotated>
        super::HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>
    where
        Parameter: Borrow<HbmTensor<f32, Chips, m![Head]>>,
        Epsilon: Borrow<HbmTensor<f32, Chips, m![1]>>,
    {
        /// Normalize the whole norm tiles intersecting `input`, preserving all other output rows.
        ///
        /// Each tile contains `2 * (256 / padded_heads) * lanes` rows, where `padded_heads` is
        /// the next power of two of `Width / Head`, and `lanes` starts at 32 and halves until
        /// the capacity is divisible by the tile size. Partial edge tiles must be initialized
        /// in the input too; the returned interval does not grant access to that padding.
        /// Aligned intervals read and write exactly their rows. Gains and epsilon stage once even
        /// for empty work, but empty work reads no input rows and writes no output rows.
        /// Allocation capacity never becomes initialized work.
        pub fn write_live<Count: AxisName, Width: AxisName>(
            &self,
            device: &mut Device,
            input: Live<&HbmTensor<bf16, Chips, m![Count, Width]>, Count>,
            output: &mut HbmTensor<bf16, Chips, m![Count, Width]>,
        ) -> Live<(), Count> {
            let (first_tile, tiles) = input.groups::<Block<Count, Width, Head>>();
            let input = input.with(*input.value());
            let fill = |device: &mut Device,
                        rows: &HbmTensor<bf16, Chips, m![Count, Width]>,
                        at: usize,
                        part: &mut Part<Count, Width, Head, Chips>| {
                // SAFETY: the tiles split the rows in order and a row is its heads in order.
                unsafe { rows.view().reshape::<Chips, Blocked<Count, Width, Head>>() }
                    .tile::<m![{ Bs<Count, Width, Head> }], 1, m![
                        { Bs<Count, Width, Head> } = 1 # ({ Bs<Count, Width, Head> }),
                        { B<Count, Width, Head> },
                        { H<Width, Head> },
                        Head
                    ]>(at)
                    .to_dm_view(&mut device.tdma, part.view_mut());
            };

            let drain = |device: &mut Device,
                         part: &Part<Count, Width, Head, Chips>,
                         rows: &mut HbmTensor<bf16, Chips, m![Count, Width]>,
                         at: usize| {
                part.view().to_hbm_view(
                    &mut device.tdma,
                    // SAFETY: the tiles split the rows in order and a row is its heads in order.
                    unsafe {
                        rows.view_mut()
                            .reshape::<Chips, Blocked<Count, Width, Head>>()
                    }
                    .tile::<m![{ Bs<Count, Width, Head> }], 1, m![
                        { Bs<Count, Width, Head> } = 1 #{!} ({ Bs<Count, Width, Head> }),
                        { B<Count, Width, Head> },
                        { H<Width, Head> },
                        Head
                    ]>(at),
                );
            };

            let inverse = |device: &mut Device,
                           part: &Part<Count, Width, Head, Chips>,
                           epsilon: &Scalar<Count, Width, Head, Chips>|
             -> RowOperand<Count, Width, Head, Chips> {
                let mean: Statistic<Count, Width, Head, Chips> =
                    Squares::<
                        Norm<Rows<Count, Width, Head>, Head>,
                        Head,
                        Rows<Count, Width, Head>,
                        m![1],
                        0,
                    >::of(&mut device.main, part.view())
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                Root::<Rows<Count, Width, Head>, m![1 # 8]>::of(
                    &mut device.sub,
                    mean.view(),
                    |square| square.vector_fp_binary(FpBinaryOp::AddF, epsilon),
                )
                .to_vrf()
            };

            let scale = |device: &mut Device,
                         part: &Part<Count, Width, Head, Chips>,
                         (weight, inverse): (
                &VrfTensor<
                    f32,
                    Chips,
                    Cluster<Count, Width, Head>,
                    Groups<Count, Width, Head>,
                    m![Head],
                >,
                &RowOperand<Count, Width, Head, Chips>,
            ),
                         output: &mut Part<Count, Width, Head, Chips>| {
                Scale::<Norm<Rows<Count, Width, Head>, Head>>::of(
                    &mut device.main,
                    part.view(),
                    inverse,
                    weight,
                )
                .commit_view(output.view_mut());
            };
            let (weight, epsilon): (
                &HbmTensor<f32, Chips, m![Head]>,
                &HbmTensor<f32, Chips, m![1]>,
            ) = (self.weight.borrow(), self.epsilon.borrow());
            let weight: DmTensor<
                f32,
                Chips,
                Cluster<Count, Width, Head>,
                Groups<Count, Width, Head>,
                m![Head],
            > = weight.to_dm(&mut device.tdma);
            let weight: VrfTensor<
                f32,
                Chips,
                Cluster<Count, Width, Head>,
                Groups<Count, Width, Head>,
                m![Head],
            > = device
                .sub
                .begin(weight.view())
                .fetch::<m![Head / 8], m![Head % 8]>()
                .collect::<m![Head / 8], m![Head % 8]>()
                .to_vrf();
            // One copy a slice, shared by its rows: a copy a row is written as one slow 32-byte
            // write each.
            let epsilon: DmTensor<
                f32,
                Chips,
                Cluster<Count, Width, Head>,
                Groups<Count, Width, Head>,
                m![1 # 8],
            > = epsilon.view().pad::<m![1 # 8]>().to_dm(&mut device.tdma);
            let epsilon: Scalar<Count, Width, Head, Chips> = device
                .sub
                .begin(epsilon.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .to_vrf();
            let last = tiles.max(1) - 1;
            let (mut current, mut next): (
                Part<Count, Width, Head, Chips>,
                Part<Count, Width, Head, Chips>,
            ) = (DmTensor::new(), DmTensor::new());
            let (mut kept, mut held): (
                Part<Count, Width, Head, Chips>,
                Part<Count, Width, Head, Chips>,
            ) = (DmTensor::new(), DmTensor::new());
            // Keep only row DMA conditional: staged VRF parameters remain outside runtime
            // control flow, including for empty work, as in the ordinary full-capacity path.
            for _ in 0..tiles.min(1) {
                fill(device, input.value(), first_tile, &mut current);
                fill(device, input.value(), first_tile + 1.min(last), &mut next);
            }
            // An odd count's last pair normalizes the last tile twice.
            for pair in 0..(tiles + 1) / 2 {
                let at = 2 * pair;
                let first = inverse(device, &current, &epsilon);
                let second = inverse(device, &next, &epsilon);
                scale(device, &current, (&weight, &first), &mut kept);
                fill(
                    device,
                    input.value(),
                    first_tile + (at + 2).min(last),
                    &mut current,
                );
                drain(device, &kept, output, first_tile + at);
                scale(device, &next, (&weight, &second), &mut held);
                fill(
                    device,
                    input.value(),
                    first_tile + (at + 3).min(last),
                    &mut next,
                );
                drain(device, &held, output, first_tile + (at + 1).min(last));
            }
            input.with(())
        }
    }

    /// Head rows a tile at a time, row-major or head-major, each slice 512 bytes of a row; the
    /// next tile arrives, and the last one stores, while the current one computes.
    pub(crate) mod tiles {
        use crate::Op;
        use crate::ops::normalization::{Root, Scale, Squares};
        use core::borrow::Borrow;
        use furiosa_opt_std::prelude::*;
        use std::marker::PhantomData;

        use super::super::{Division, HeadNorm};
        use super::H;
        use crate::tensor::live::Live;

        /// The heads a slice holds of a row.
        #[derive(Clone, Debug)]
        pub(crate) struct Kept<Width, Head>(PhantomData<(Width, Head)>);

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
        pub(crate) struct Slots<Width, Head>(PhantomData<(Width, Head)>);

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
        pub(crate) struct Depth<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Depth<Width, Head> {
            const NAME: Ident = Ident::new("HeadDepth");
            const SIZE: usize = if Head::SIZE <= 64 { 8 } else { 4 };
        }

        /// The rows of a tile.
        #[derive(Clone, Debug)]
        pub(crate) struct Strip<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Strip<Width, Head> {
            const NAME: Ident = Ident::new("HeadStrip");
            const SIZE: usize = 2 * (256 / Slots::<Width, Head>::SIZE) * Depth::<Width, Head>::SIZE;
        }

        /// The rows of a tile one cluster holds.
        #[derive(Clone, Debug)]
        pub(crate) struct Group<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Group<Width, Head> {
            const NAME: Ident = Ident::new("HeadGroup");
            const SIZE: usize = Strip::<Width, Head>::SIZE / 2;
        }

        /// The tiles of a row sequence.
        #[derive(Clone, Debug)]
        pub(crate) struct Strips<Count, Width, Head, P = (), Chips = m![1], Q = ()>(
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

            /// Tile `at`, written into `tile`.
            fn to_dm(
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

            fn to_dm(
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
                    // SAFETY: the tiles split the rows in order and a row is its heads in order.
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

            fn to_dm(
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
                    // SAFETY: the tiles split each head's rows in order; a head is its features in
                    // order.
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
            /// Tile `at`, written, through the pass into `output`.
            fn apply(
                &self,
                device: &mut Device,
                tile: &Tile<bf16, Width, Head, Chips, Q>,
                at: usize,
                output: &mut Tile<bf16, Width, Head, Chips, Q>,
            );
        }

        /// Every tile of `input` through `pass` into `output`, the next tile arriving while one
        /// computes and the tile before it storing.
        impl<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M, Q: M, L, P>
            crate::Op<(&L, &mut L)> for Strips<Count, Width, Head, P, Chips, Q>
        where
            L: Layout<Count, Width, Head, Chips, Order = Q>,
            P: Pass<Width, Head, Chips, Q>,
        {
            type Output = ();

            fn forward(&self, device: &mut Device, (input, output): (&L, &mut L)) {
                crate::Op::forward(self, device, (Live::<(), Count>::all().with(input), output));
            }
        }

        impl<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M, Q: M, L, P>
            crate::Op<(Live<&L, Count>, &mut L)> for Strips<Count, Width, Head, P, Chips, Q>
        where
            L: Layout<Count, Width, Head, Chips, Order = Q>,
            P: Pass<Width, Head, Chips, Q>,
        {
            type Output = ();

            fn forward(&self, device: &mut Device, (input, output): (Live<&L, Count>, &mut L)) {
                let pass = &self.1;
                let (first_tile, tiles) = input.groups::<Strip<Width, Head>>();
                if tiles == 0 {
                    return;
                }
                let (mut first, mut second): (
                    Tile<bf16, Width, Head, Chips, Q>,
                    Tile<bf16, Width, Head, Chips, Q>,
                ) = (DmTensor::new(), DmTensor::new());
                let (mut kept, mut held): (
                    Tile<bf16, Width, Head, Chips, Q>,
                    Tile<bf16, Width, Head, Chips, Q>,
                ) = (DmTensor::new(), DmTensor::new());
                input.value().to_dm(device, first_tile, &mut first);
                if tiles > 1 {
                    input.value().to_dm(device, first_tile + 1, &mut second);
                }
                for pair in 0..tiles / 2 {
                    let at = 2 * pair;
                    pass.apply(device, &first, first_tile + at, &mut kept);
                    input
                        .value()
                        .to_dm(device, first_tile + (at + 2).min(tiles - 1), &mut first);
                    output.store(device, &kept, first_tile + at);
                    pass.apply(device, &second, first_tile + at + 1, &mut held);
                    input
                        .value()
                        .to_dm(device, first_tile + (at + 3).min(tiles - 1), &mut second);
                    output.store(device, &held, first_tile + at + 1);
                }
                if tiles % 2 == 1 {
                    pass.apply(device, &first, first_tile + tiles - 1, &mut kept);
                    output.store(device, &kept, first_tile + tiles - 1);
                }
            }
        }

        /// One pass over the same tile of two row sequences, so what the tile's rows share writes
        /// once for both.
        pub(crate) trait Both<Width: AxisName, Head: AxisName, Chips: M, Q: M> {
            /// Tile `at` of both sequences, written, through the pass into the outputs.
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
            crate::Op<((&L, &L), (&mut L, &mut L))> for Strips<Count, Width, Head, P, Chips, Q>
        where
            L: Layout<Count, Width, Head, Chips, Order = Q>,
            P: Both<Width, Head, Chips, Q>,
        {
            type Output = ();

            fn forward(
                &self,
                device: &mut Device,
                (inputs, outputs): ((&L, &L), (&mut L, &mut L)),
            ) {
                crate::Op::forward(
                    self,
                    device,
                    (Live::<(), Count>::all().with(inputs), outputs),
                );
            }
        }

        impl<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M, Q: M, L, P>
            crate::Op<(Live<(&L, &L), Count>, (&mut L, &mut L))>
            for Strips<Count, Width, Head, P, Chips, Q>
        where
            L: Layout<Count, Width, Head, Chips, Order = Q>,
            P: Both<Width, Head, Chips, Q>,
        {
            type Output = ();

            fn forward(
                &self,
                device: &mut Device,
                (inputs, outputs): (Live<(&L, &L), Count>, (&mut L, &mut L)),
            ) {
                let pass = &self.1;
                type Two<Width, Head, Chips, Q> = (
                    Tile<bf16, Width, Head, Chips, Q>,
                    Tile<bf16, Width, Head, Chips, Q>,
                );
                let (first_tile, tiles) = inputs.groups::<Strip<Width, Head>>();
                if tiles == 0 {
                    return;
                }
                let both_to_dm =
                    |device: &mut Device, at: usize, both: &mut Two<Width, Head, Chips, Q>| {
                        inputs.value().0.to_dm(device, first_tile + at, &mut both.0);
                        inputs.value().1.to_dm(device, first_tile + at, &mut both.1);
                    };
                let (first_out, second_out) = outputs;
                let mut store_both =
                    |device: &mut Device, both: &Two<Width, Head, Chips, Q>, at: usize| {
                        first_out.store(device, &both.0, first_tile + at);
                        second_out.store(device, &both.1, first_tile + at);
                    };
                let mut first: Two<Width, Head, Chips, Q> = (DmTensor::new(), DmTensor::new());
                let mut second: Two<Width, Head, Chips, Q> = (DmTensor::new(), DmTensor::new());
                let mut kept: Two<Width, Head, Chips, Q> = (DmTensor::new(), DmTensor::new());
                let mut held: Two<Width, Head, Chips, Q> = (DmTensor::new(), DmTensor::new());
                both_to_dm(device, 0, &mut first);
                if tiles > 1 {
                    both_to_dm(device, 1, &mut second);
                }
                for pair in 0..tiles / 2 {
                    let at = 2 * pair;
                    pass.apply(
                        device,
                        (&first.0, &first.1),
                        first_tile + at,
                        (&mut kept.0, &mut kept.1),
                    );
                    both_to_dm(device, (at + 2).min(tiles - 1), &mut first);
                    store_both(device, &kept, at);
                    pass.apply(
                        device,
                        (&second.0, &second.1),
                        first_tile + at + 1,
                        (&mut held.0, &mut held.1),
                    );
                    both_to_dm(device, (at + 3).min(tiles - 1), &mut second);
                    store_both(device, &held, at + 1);
                }
                if tiles % 2 == 1 {
                    pass.apply(
                        device,
                        (&first.0, &first.1),
                        first_tile + tiles - 1,
                        (&mut kept.0, &mut kept.1),
                    );
                    store_both(device, &kept, tiles - 1);
                }
            }
        }

        /// The head norm's operands.
        struct Normalize<Width: AxisName, Head: AxisName, Chips: M, const DIVIDE: bool> {
            weight: VrfTensor<f32, Chips, Half<Width, Head>, Slices<Width, Head>, m![Head]>,
            epsilon: VrfTensor<f32, Chips, Half<Width, Head>, Slices<Width, Head>, m![1 # 8]>,
        }

        impl<Width: AxisName, Head: AxisName, Chips: M, Q: M, const DIVIDE: bool>
            Pass<Width, Head, Chips, Q> for Normalize<Width, Head, Chips, DIVIDE>
        {
            fn apply(
                &self,
                device: &mut Device,
                rows: &Tile<bf16, Width, Head, Chips, Q>,
                _: usize,
                normalized: &mut Tile<bf16, Width, Head, Chips, Q>,
            ) {
                let mean: Statistic<Width, Head, Chips, Q> =
                    Squares::<super::Norm<Q, Head>, Head, m![{ Q }], m![1], 0>::of(
                        &mut device.main,
                        rows.view(),
                    )
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                if const { DIVIDE } {
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
                        .vector_narrow_split::<m![{ Q }, Head / 8, Head % 8 / 4 % 2], m![Head % 4]>(
                        )
                        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &self.weight)
                        .vector_fp_div(&rms)
                        .vector_widen_concat::<m![{ Q }, Head / 8], m![Head % 8]>()
                        .vector_final()
                        .cast::<bf16, m![Head % 8 # 16]>()
                        .commit_trim::<m![Head % 8]>()
                        .commit_view(normalized.view_mut());
                } else {
                    let inverse: RowOperand<Width, Head, Chips, Q> =
                        Root::<m![{ Q }], m![1 # 8]>::of(&mut device.sub, mean.view(), |square| {
                            square.vector_fp_binary(FpBinaryOp::AddF, &self.epsilon)
                        })
                        .to_vrf();
                    Scale::<super::Norm<Q, Head>>::of(
                        &mut device.main,
                        rows.view(),
                        &inverse,
                        &self.weight,
                    )
                    .commit_view(normalized.view_mut());
                }
            }
        }

        /// Every head row of `input` normalized by its RMS and the shared head weight.
        impl<
            Head: AxisName,
            Width: AxisName,
            Count: AxisName,
            Chips: M,
            Parameter,
            Epsilon,
            const EPSILON: usize,
            Rotated,
        > crate::Op<&Major<Count, Width, Head, Chips>>
            for HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>
        where
            Parameter: Borrow<HbmTensor<f32, Chips, m![Head]>>,
            Epsilon: Borrow<HbmTensor<f32, Chips, m![1]>>,
        {
            type Output = Major<Count, Width, Head, Chips>;

            fn forward(
                &self,
                device: &mut Device,
                input: &Major<Count, Width, Head, Chips>,
            ) -> Self::Output {
                self.normalized_major::<Count, Width, false>(device, input)
            }
        }

        /// Every head row of `input` normalized by its RMS and the shared head weight.
        impl<
            Head: AxisName,
            Width: AxisName,
            Count: AxisName,
            Chips: M,
            Parameter,
            Epsilon,
            const EPSILON: usize,
            Rotated,
        > crate::Op<&Major<Count, Width, Head, Chips>>
            for Division<&HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>>
        where
            Parameter: Borrow<HbmTensor<f32, Chips, m![Head]>>,
            Epsilon: Borrow<HbmTensor<f32, Chips, m![1]>>,
        {
            type Output = Major<Count, Width, Head, Chips>;

            fn forward(
                &self,
                device: &mut Device,
                input: &Major<Count, Width, Head, Chips>,
            ) -> Self::Output {
                self.0.normalized_major::<Count, Width, true>(device, input)
            }
        }

        impl<Head: AxisName, Chips: M, Parameter, Epsilon, const EPSILON: usize, Rotated>
            HeadNorm<Head, Chips, Parameter, Epsilon, EPSILON, Rotated>
        where
            Parameter: Borrow<HbmTensor<f32, Chips, m![Head]>>,
            Epsilon: Borrow<HbmTensor<f32, Chips, m![1]>>,
        {
            fn normalized_major<Count: AxisName, Width: AxisName, const DIVIDE: bool>(
                &self,
                device: &mut Device,
                input: &Major<Count, Width, Head, Chips>,
            ) -> Major<Count, Width, Head, Chips> {
                let (weight, epsilon): (
                    &HbmTensor<f32, Chips, m![Head]>,
                    &HbmTensor<f32, Chips, m![1]>,
                ) = (self.weight.borrow(), self.epsilon.borrow());
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
                    Normalize::<Width, Head, Chips, DIVIDE> { weight, epsilon },
                )
                .forward(device, (input, &mut output));
                output
            }
        }
    }
}

impl<D: Scalar, W: AxisName, R: AxisName, H: AxisName>
    Op<&(Rows<D, R, W>, Rows<D, R, W>, Rows<D, R, W>)> for Attention<PhantomData<H>, Scale>
{
    type Output = Rows<D, R, W>;

    fn forward(
        &self,
        device: &mut Device,
        (query, key, value): &(Rows<D, R, W>, Rows<D, R, W>, Rows<D, R, W>),
    ) -> Self::Output {
        self.forward(device, (query, key, value))
    }
}

impl<'k, 'v, D: Scalar, W: AxisName, K: AxisName, Q: AxisName, H: AxisName>
    Op<(
        &Rows<D, Q, W>,
        &&'k RowStorage<K, D>,
        &&'v RowStorage<K, D>,
        &Read<'_>,
    )> for Attention<PhantomData<H>, Scale>
{
    type Output = Rows<D, Q, W>;

    fn forward(
        &self,
        _: &mut Device,
        _: (
            &Rows<D, Q, W>,
            &&'k RowStorage<K, D>,
            &&'v RowStorage<K, D>,
            &Read<'_>,
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
    Op<(&Rows<D, Q, W>, &KeyValue<RowStorage<K, D>>, &Read<'_>)>
    for Attention<PhantomData<H>, Scale>
{
    type Output = Rows<D, Q, W>;

    fn forward(
        &self,
        device: &mut Device,
        (query, cache, read): (&Rows<D, Q, W>, &KeyValue<RowStorage<K, D>>, &Read<'_>),
    ) -> Self::Output {
        self.forward(device, (query, &&cache.0, &&cache.1, read))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Module;
    use crate::ops::attention::flash::{Head as Dims, Span as KeySpan, Visible};
    use crate::ops::linear::Linear;
    use crate::ops::normalization::tests::random;
    use crate::tensor::form::Weight;
    use crate::tensor::live::tests::count;
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;
    use std::f32::consts::FRAC_1_SQRT_2;

    struct Fixture;

    axes![
        Head = 128,
        Width = 7168,
        Count = 512,
        Shard = 1792,
        Sharded = 2048,
        TileWidth = 2048,
        ShortHead = 64
    ];

    axes![
        Input = 5376,
        Bundle = 128,
        Double = 256,
        Query = 7168,
        Local = 1792
    ];

    impl Fixture {
        async fn fused_parts<R: AxisName, W: AxisName>() {
            let _lease = DEVICE.lock().await;
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
            let projection = Projection::new(Linear::new(Weight::new(weight)));
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

    impl Fixture {
        async fn heads_normalize_alone<Width: AxisName, Count: AxisName, Head: AxisName>() {
            let _lease = DEVICE.lock().await;
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
                weight: HostTensor::<f32, m![Head]>::from_vec(weight.clone())
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                epsilon: HostTensor::<f32, m![1]>::from_vec(vec![1e-5])
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                head: PhantomData,
            };
            let input = HostTensor::<bf16, m![Count, Width]>::from_vec(inputs.clone())
                .to_hbm::<m![1], m![Count, Width]>(&mut device.pdma)
                .await
                .unwrap();
            let row_major = launch(normalized, (&mut device, &norm, &input))
                .await
                .unwrap()
                .to_host::<m![Count, Width]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let major = HostTensor::<bf16, m![Count, Width]>::from_vec(inputs.clone())
                .to_hbm::<m![1], m![Width / ({ Symbol<Head> }), Count, Width % ({ Symbol<Head> })]>(
                    &mut device.pdma,
                )
                .await
                .unwrap();
            let ordinary = launch(normalized_major, (&mut device, &norm, &major))
                .await
                .unwrap();
            let divided = launch(normalized_division, (&mut device, &norm, &major))
                .await
                .unwrap();
            let mut outputs = vec![row_major];
            for output in [ordinary, divided] {
                outputs.push(
                    output
                        .to_host::<m![Count, Width]>(&mut device.pdma)
                        .await
                        .unwrap()
                        .into_vec(),
                );
            }
            for (variant, output) in outputs.iter().enumerate() {
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
                            "variant {variant} row {} head {} dim {dim}: {got} vs {expected}",
                            head / (Width::SIZE / Head::SIZE),
                            head % (Width::SIZE / Head::SIZE),
                        );
                    }
                }
            }
        }
    }
    // Live rows of 128: text 77 this call, audio 100; video 5,800 of 5,888, or 4,800 of 4,864 so
    // the last query tile is a 1,024-row tail.
    type Sequence = Packed<128, 128, 100, 128, 5800, 5888>;
    type Tailed = Packed<128, 128, 100, 128, 4800, 4864>;
    axes![PairWidth = 256, QuadWidth = 1024, Two = 2, Eight = 8];

    /// The score multiplier of 128-wide heads, `1 / sqrt(128)`.
    const INVERSE_ROOT: usize = (FRAC_1_SQRT_2 / 8.0).to_bits() as usize;

    type Scaled128<H> = Attention<PhantomData<H>, F32<INVERSE_ROOT>>;

    /// `stride(row)` is the row step of the check from `row`.
    impl Fixture {
        async fn packed_rows_attend<S: AxisName, W: AxisName, H: AxisName, K: AxisName>(
            video: usize,
            stride: impl Fn(usize) -> usize,
        ) where
            Attention<PhantomData<H>, F32<INVERSE_ROOT>>: for<'a> Module<
                    (
                        &'a Rows<bf16, S, W>,
                        &'a Rows<bf16, S, W>,
                        &'a Rows<bf16, S, W>,
                        (&'a Rotation<S>, &'a Visible<K>, &'a Value),
                    ),
                    (),
                    &'a mut (),
                    Output = Rows<bf16, S, W>,
                >,
        {
            const LIVE_TEXT: usize = 77;
            let _lease = DEVICE.lock().await;
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
            let limit = count(K::SIZE * 2048)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let operands = (
                upload(&mut device, &q).await,
                upload(&mut device, &k).await,
                upload(&mut device, &v).await,
            );
            let attention: Scaled128<H> = Attention {
                heads: PhantomData,
                scale: F32,
            };
            let got = launch(
                attended,
                (
                    &mut device,
                    &attention,
                    (&operands.0, &operands.1, &operands.2),
                    &rotation,
                    &visible,
                    &limit,
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
    fn project_joint<R: AxisName, W: AxisName>(
        device: &mut Device,
        projection: &Projection<Linear<Input, Joint<W>>, Head>,
        rows: &Rows<bf16, R, Input>,
    ) -> Rows<bf16, R, Joint<W>> {
        Op::forward(&projection.0, device, rows)
    }

    #[device(chip = 1)]
    fn project_parts<R: AxisName, W: AxisName>(
        device: &mut Device,
        projection: &Projection<Linear<Input, Joint<W>>, Head>,
        rows: &Rows<bf16, R, Input>,
    ) -> (Rows<bf16, R, W>, Rows<bf16, R, W>, Rows<bf16, R, W>) {
        Op::forward(projection, device, rows)
    }

    /// Each part a fused projection returns is its columns of the whole projection, bit for bit,
    /// whether a pass computes one part or all three.
    #[tokio::test(flavor = "current_thread")]
    async fn fused_parts_are_the_joint_columns() {
        Fixture::fused_parts::<Bundle, Query>().await;
        Fixture::fused_parts::<Double, Local>().await;
    }
    #[device(chip = 1)]
    fn normalized<W: AxisName, C: AxisName, H: AxisName>(
        device: &mut Device,
        norm: &HeadNorm<H>,
        input: &Rows<bf16, C, W>,
    ) -> Rows<bf16, C, W> {
        Op::forward(norm, device, input)
    }

    #[device(chip = 1)]
    fn normalized_major<W: AxisName, C: AxisName, H: AxisName>(
        device: &mut Device,
        norm: &HeadNorm<H>,
        input: &heads::tiles::Major<C, W, H, m![1]>,
    ) -> heads::tiles::Major<C, W, H, m![1]> {
        Op::forward(norm, device, input)
    }

    #[device(chip = 1)]
    fn normalized_division<W: AxisName, C: AxisName, H: AxisName>(
        device: &mut Device,
        norm: &HeadNorm<H>,
        input: &heads::tiles::Major<C, W, H, m![1]>,
    ) -> heads::tiles::Major<C, W, H, m![1]> {
        Op::forward(&Division(norm), device, input)
    }

    /// Each head of each row is RMS-normalized alone, as the FP64 definition rounded once to BF16;
    /// a row's heads differ in magnitude by up to 2^9, so a statistic shared across heads fails.
    #[tokio::test(flavor = "current_thread")]
    async fn normalizes_each_head_alone() {
        Fixture::heads_normalize_alone::<Width, Count, Head>().await;
        Fixture::heads_normalize_alone::<Shard, Sharded, Head>().await;
        Fixture::heads_normalize_alone::<TileWidth, Sharded, ShortHead>().await;
    }

    #[device(chip = 1)]
    fn attended<S: AxisName, W: AxisName, H: AxisName, K: AxisName>(
        device: &mut Device,
        attention: &Scaled128<H>,
        (query, key, value): (&Rows<bf16, S, W>, &Rows<bf16, S, W>, &Rows<bf16, S, W>),
        rotation: &Rotation<S>,
        visible: &Visible<K>,
        live: &Value,
    ) -> Rows<bf16, S, W>
    where
        Attention<PhantomData<H>, F32<INVERSE_ROOT>>: for<'a> Module<
                (
                    &'a Rows<bf16, S, W>,
                    &'a Rows<bf16, S, W>,
                    &'a Rows<bf16, S, W>,
                    (&'a Rotation<S>, &'a Visible<K>, &'a Value),
                ),
                (),
                &'a mut (),
                Output = Rows<bf16, S, W>,
            >,
    {
        Module::forward(
            attention,
            device,
            (),
            &mut (),
            (query, key, value, (rotation, visible, live)),
        )
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

    #[device(chip = 1)]
    fn text_attended(
        device: &mut Device,
        (query, key, value): (
            &Rows<bf16, Envelope, PairWidth>,
            &Rows<bf16, Envelope, PairWidth>,
            &Rows<bf16, Envelope, PairWidth>,
        ),
        visible: &Visible<Three>,
        live: &Value,
    ) -> Rows<bf16, Envelope, PairWidth> {
        Op::forward(
            &Attention {
                heads: PhantomData::<Two>,
                scale: F32::<INVERSE_ROOT>,
            },
            device,
            (query, key, value, (visible, live)),
        )
    }
    /// Every live row of a text envelope attends its live keys alone, against an FP64 softmax with
    /// BF16-rounded scores; pad rows are four times larger, so a leak shows.
    #[tokio::test(flavor = "current_thread")]
    async fn text_rows_attend_the_live_keys() {
        const LIVE: usize = 77;
        let _lease = DEVICE.lock().await;
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
        let live = count(3 * 2048).to_hbm(&mut device.pdma).await.unwrap();
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
                &live,
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

#[cfg(test)]
mod envelope_tests {
    //! Replicated refiner attention with one 2048-row envelope and explicit scattered key visibility.
    //!
    //! Synthetic finite Q/K/V exercise all 56 heads on every chip. Both Compact and ordinary
    //! attention use the shared envelope preparation and are checked against independent FP64
    //! softmax values. This is an attention boundary contract, not an owning-refiner execution test.

    use crate::Op;
    use crate::ops::attention::Attention;
    use crate::ops::attention::flash::{Compact, Head, Span, Visible};
    use crate::ops::index::packed;
    use crate::tensor::parallel::Shard;
    use crate::tensor::scalar::{F32, Value};
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;
    use std::marker::PhantomData;

    axes![Width = 7168, Heads = 56, Spans = 3];

    type Envelope = packed::Span<2048, 2048>;
    type Chips = m![{ Symbol<Shard<4>> }];
    type Rows = packed::Rows<Envelope, Width, Chips>;
    type Mask = Visible<Spans, Chips>;
    type Operands<'a> = (&'a Rows, &'a Rows, &'a Rows, &'a Mask, &'a Value<Chips>);
    type Ordinary = Attention<PhantomData<Heads>, F32<SCALE>>;

    const SCALE: usize = (std::f32::consts::FRAC_1_SQRT_2 / 8.).to_bits() as usize;
    const KEY_BOUND: usize = 6144;
    const CLASSES: usize = 7;
    // Existing permanent Compact attention gates use a 1e-2 relative RMS bound.
    const LIMIT: f64 = 1e-2;

    #[derive(Clone, Copy, Debug)]
    struct Case {
        valid: usize,
        scattered: bool,
        uniform: bool,
    }

    #[device(chip = 4)]
    fn attend<A>(device: &mut Device, attention: &A, (q, k, v, mask, bound): Operands<'_>) -> Rows
    where
        A: DeviceSend,
        A: for<'a> Op<(&'a Rows, &'a Rows, &'a Rows, (&'a Mask, &'a Value<Chips>)), Output = Rows>,
    {
        attention.forward(device, (q, k, v, (mask, bound)))
    }

    fn amplitude(chip: usize, row: usize, head: usize, uniform: bool) -> bf16 {
        let integer = (row % CLASSES) as i32 - 3 + (head % 3) as i32 - (chip % 2) as i32;
        bf16::from_f32(if uniform { 0. } else { integer as f32 * 4. })
    }

    fn key(row: usize, valid: usize, channel: usize) -> bf16 {
        let value = if channel != 0 {
            0.
        } else if row < valid {
            [-1., -0.5, 0.5, 1.][row % 4]
        } else {
            // Distracting padding is finite, and would dominate an accidentally unmasked key.
            6. + (row % 3) as f32
        };
        bf16::from_f32(value)
    }

    fn value(chip: usize, row: usize, head: usize, channel: usize, valid: usize) -> bf16 {
        let base = (32 + (chip * 5 + head * 3 + channel * 7) % 29) as f32 / 64.;
        let part = [-3., -1., 1., 3.][row % 4] / 16.;
        bf16::from_f32(base + part + if row < valid { 0. } else { 16. })
    }

    fn visible(case: Case, position: usize) -> bool {
        if case.scattered {
            // First/last keys of all three spans. A token-count bound would miss later spans.
            [0, 2047, 2048, 4095, 4096, 6143].contains(&position)
        } else {
            position < case.valid
        }
    }

    async fn upload(device: &mut Device, target: &mut Rows, case: Case, term: usize) {
        let values = (0..4).flat_map(|chip| {
            (0..Envelope::SIZE).flat_map(move |row| {
                (0..Width::SIZE).map(move |column| {
                    let (head, channel) = (column / Head::SIZE, column % Head::SIZE);
                    match term {
                        0 if channel == 0 => amplitude(chip, row, head, case.uniform),
                        0 => bf16::from_f32(0.),
                        1 => key(row, case.valid, channel),
                        _ => value(chip, row, head, channel, case.valid),
                    }
                })
            })
        });
        HostTensor::<bf16, m![{ Chips }, Envelope, Width]>::from_vec(values)
            .to_hbm(&mut device.pdma)
            .output(target)
            .await
            .unwrap();
    }

    // The source formula has four key/value classes. Count actual visible key positions, including
    // repeated source rows in later spans; no production packing or attention helper is used here.
    fn oracle(case: Case) -> Vec<f64> {
        let mut counts = [0usize; 4];
        for position in 0..KEY_BOUND {
            if visible(case, position) {
                counts[position % Envelope::SIZE % 4] += 1;
            }
        }
        assert!(counts.iter().sum::<usize>() > 0);
        let mut result = Vec::with_capacity(4 * Heads::SIZE * CLASSES * Head::SIZE);
        for chip in 0..4 {
            for head in 0..Heads::SIZE {
                for class in 0..CLASSES {
                    let q = f64::from(amplitude(chip, class, head, case.uniform).to_f32());
                    let masses: [f64; 4] = std::array::from_fn(|kind| {
                        let k = f64::from(key(kind, 2048, 0).to_f32());
                        counts[kind] as f64
                            * (q * k * f64::from(f32::from_bits(SCALE as u32))).exp()
                    });
                    let denominator = masses.iter().sum::<f64>();
                    for channel in 0..Head::SIZE {
                        let numerator = (0..4)
                            .map(|kind| {
                                masses[kind]
                                    * f64::from(value(chip, kind, head, channel, 2048).to_f32())
                            })
                            .sum::<f64>();
                        result.push(numerator / denominator);
                    }
                }
            }
        }
        result
    }

    fn check(actual: &[bf16], expected: &[f64], case: Case, variant: &str) {
        assert_eq!(actual.len(), 4 * Envelope::SIZE * Width::SIZE);
        for chip in 0..4 {
            for row in 0..Envelope::SIZE {
                for head in 0..Heads::SIZE {
                    let mut error = 0.;
                    let mut square = 0.;
                    for channel in 0..Head::SIZE {
                        let at = ((chip * Heads::SIZE + head) * CLASSES + row % CLASSES)
                            * Head::SIZE
                            + channel;
                        let want = expected[at];
                        let column = head * Head::SIZE + channel;
                        let got = f64::from(
                            actual[(chip * Envelope::SIZE + row) * Width::SIZE + column].to_f32(),
                        );
                        assert!(
                            got.is_finite(),
                            "{variant} {case:?} chip={chip} row={row} head={head} channel={channel}"
                        );
                        error += (got - want).powi(2);
                        square += want.powi(2);
                    }
                    let relative = (error / square).sqrt();
                    assert!(
                        relative <= LIMIT,
                        "{variant} {case:?} chip={chip} row={row} head={head} relative_RMS={relative}, limit={LIMIT}"
                    );
                }
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore = "requires four explicitly selected RNGD chips"]
    async fn compact_and_ordinary_refiner_envelopes_preserve_masks_and_values() {
        let _lease = crate::tests::DEVICE.lock().await;
        const {
            assert!(
                Heads::SIZE * Head::SIZE == Width::SIZE && Spans::SIZE * Span::SIZE == KEY_BOUND
            )
        };
        let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
        let mut q = Rows::alloc(&device).unwrap();
        let mut k = Rows::alloc(&device).unwrap();
        let mut v = Rows::alloc(&device).unwrap();
        let mut mask = Mask::alloc(&device).unwrap();
        let bound: Value<Chips> = HostTensor::<i32, m![1]>::from_vec([KEY_BOUND as i32])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let compact = Compact::<Heads, SCALE>::default();
        let ordinary = Ordinary::default();
        let mut cases: Vec<_> = [128, 129, 511, 1025, 2047, 2048, 1, 128, 2048, 129, 128]
            .into_iter()
            .enumerate()
            .map(|(index, valid)| Case {
                valid,
                scattered: false,
                uniform: index == 0,
            })
            .collect();
        cases.insert(
            6,
            Case {
                valid: 2048,
                scattered: true,
                uniform: false,
            },
        );
        for (index, case) in cases.into_iter().enumerate() {
            // Reuse allocations and rewrite every cell: shrink after growth cannot retain old keys.
            upload(&mut device, &mut q, case, 0).await;
            upload(&mut device, &mut k, case, 1).await;
            upload(&mut device, &mut v, case, 2).await;
            HostTensor::<bf16, m![{ Chips }, Spans, Span]>::from_vec((0..4).flat_map(|_| {
                (0..KEY_BOUND).map(|position| bf16::from_f32(f32::from(visible(case, position))))
            }))
            .to_hbm(&mut device.pdma)
            .output(&mut mask)
            .await
            .unwrap();
            let expected = oracle(case);
            // Alternate variant order while reusing the rewritten operands and mask.
            for use_compact in [index % 2 == 0, index % 2 != 0] {
                let variant = if use_compact { "Compact" } else { "ordinary" };
                let output = if use_compact {
                    launch(attend, (&mut device, &compact, (&q, &k, &v, &mask, &bound)))
                        .await
                        .unwrap()
                } else {
                    launch(
                        attend,
                        (&mut device, &ordinary, (&q, &k, &v, &mask, &bound)),
                    )
                    .await
                    .unwrap()
                };
                output.ready().await.unwrap();
                let values = output
                    .to_host::<m![{ Chips }, Envelope, Width]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec();
                check(&values, &expected, case, variant);
                drop(output);
            }
        }
    }
}
