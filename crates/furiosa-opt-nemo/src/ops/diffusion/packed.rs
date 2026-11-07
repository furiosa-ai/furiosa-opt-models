//! Text, audio and video rows packed into one sequence, and the row operations whose modulation
//! each row's modality and timestep select.
//!
//! - The spans are text, audio, video, in that order, each a whole number of 64-row tiles (16-row
//!   ones in a chip's part of a sequence split over chips), so a tile never mixes modalities.
//! - A modulation table holds one row per timestep and modality, row `timestep * 3 + modality`
//!   (video 0, text 1, audio 2). Text and video rows take the video timestep 0, audio rows the
//!   audio timestep 1, so a span's table row is a constant.
//! - A tile spreads its features over the slices and its two halves over the clusters; its row
//!   statistics reduce across slices, and per-feature operands stay in the slices that use them.

use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

use super::sequence::Sum;
use crate::Module;
use crate::ops::layout::axes::{self, Shard};
use crate::ops::normalization::{Chunk, FeatureSlices};
use crate::ops::vector::MeanSquare;
use crate::ops::weight::Weight;

mod blocks;
mod rows;

axes![
    Strip = 64,
    Group = 6,
    Timestep = 2,
    Modality = 3,
    Chunks = 6
];

/// Text, audio and video spans, in that order.
pub type Packed<Text, Audio, Video> = Sum<Text, Sum<Audio, Video>>;

/// A modality's span: `LIVE` rows of tokens, then pad rows up to `ROWS`, a whole number of 16-row
/// tiles.
#[derive(Clone, Debug, DeviceSend)]
pub struct Span<const LIVE: usize, const ROWS: usize>;

impl<const LIVE: usize, const ROWS: usize> AxisName for Span<LIVE, ROWS> {
    const NAME: Ident = Ident::new("Span");

    const SIZE: usize = {
        assert!(LIVE <= ROWS && ROWS.is_multiple_of(16));
        ROWS
    };
}

/// BF16 rows in HBM, a copy or a part on each chip of `Chips`.
pub type Rows<R, W, Chips = m![1]> = HbmTensor<bf16, Chips, m![R, W]>;

/// One modulation row per timestep and modality.
pub type Table<W, Chips = m![1]> = HbmTensor<bf16, Chips, m![Group, W]>;

/// The rows of `span` as rows `at..` of `packed`.
impl<const LIVE: usize, const ROWS: usize> Span<LIVE, ROWS> {
    pub const fn new() -> Self {
        Self
    }
}

impl<const LIVE: usize, const ROWS: usize, R: AxisName, W: AxisName, Chips: M>
    Module<(
        &mut Device,
        (
            &Rows<Span<LIVE, ROWS>, W, Chips>,
            (&mut Rows<R, W, Chips>, usize),
        ),
    )> for Span<LIVE, ROWS>
{
    type Output = ();

    fn forward(
        &self,
        (device, (span, (packed, at))): (
            &mut Device,
            (
                &Rows<Span<LIVE, ROWS>, W, Chips>,
                (&mut Rows<R, W, Chips>, usize),
            ),
        ),
    ) {
        type S<const LIVE: usize, const ROWS: usize> = Symbol<Span<LIVE, ROWS>>;
        let rows = packed
            .view_mut()
            .tile::<m![R], ROWS, m![R = ({ S<LIVE, ROWS> }) #{!} R, W]>(at);
        // SAFETY: the tile's rows are the span's in order; only the row axis is renamed.
        let rows = unsafe { rows.reshape::<Chips, m![{ S<LIVE, ROWS> } #{!} R, W]>() };
        span.view().to_hbm_view(&mut device.tdma, rows);
    }
}

impl<const LIVE: usize, const ROWS: usize, R: AxisName, W: AxisName, Chips: M>
    Module<(&mut Device, (&Rows<R, W, Chips>, usize))> for Span<LIVE, ROWS>
{
    type Output = Rows<Span<LIVE, ROWS>, W, Chips>;

    fn forward(
        &self,
        (device, (packed, at)): (&mut Device, (&Rows<R, W, Chips>, usize)),
    ) -> Self::Output {
        type S<const LIVE: usize, const ROWS: usize> = Symbol<Span<LIVE, ROWS>>;
        let rows = packed
            .view()
            .tile::<m![R], ROWS, m![R = ({ S<LIVE, ROWS> }) # R, W]>(at);
        // SAFETY: as for `place`.
        let rows = unsafe { rows.reshape::<Chips, m![{ S<LIVE, ROWS> } # R, W]>() };
        let mut span = HbmTensor::new();
        rows.to_hbm_view(&mut device.tdma, span.view_mut());
        span
    }
}

/// The span's rows at the start of `packed`.
impl<const LIVE: usize, const ROWS: usize, R: AxisName, W: AxisName, Chips: M>
    Module<(&mut Device, &Rows<R, W, Chips>)> for Span<LIVE, ROWS>
{
    type Output = Rows<Span<LIVE, ROWS>, W, Chips>;

    fn forward(&self, (device, packed): (&mut Device, &Rows<R, W, Chips>)) -> Self::Output {
        self.forward((device, (packed, 0)))
    }
}

/// Video and audio rows with the text rows packed into one sequence: text, audio, then video.
#[derive(DeviceSend)]
pub struct Join;

impl<
    'd,
    'm,
    't,
    const TL: usize,
    const TR: usize,
    const AL: usize,
    const AR: usize,
    const VL: usize,
    const VR: usize,
    W: AxisName,
    Chips: M,
>
    Module<(
        &'d mut Device,
        (
            &'m (Rows<Span<VL, VR>, W, Chips>, Rows<Span<AL, AR>, W, Chips>),
            &'t Rows<Span<TL, TR>, W, Chips>,
        ),
    )> for Join
{
    type Output = Rows<Packed<Span<TL, TR>, Span<AL, AR>, Span<VL, VR>>, W, Chips>;

    fn forward(
        &self,
        (device, ((video, audio), text)): (
            &'d mut Device,
            (
                &'m (Rows<Span<VL, VR>, W, Chips>, Rows<Span<AL, AR>, W, Chips>),
                &'t Rows<Span<TL, TR>, W, Chips>,
            ),
        ),
    ) -> Self::Output {
        let mut packed = HbmTensor::new();
        Span::<TL, TR>::new().forward((&mut *device, (text, (&mut packed, 0))));
        Span::<AL, AR>::new().forward((&mut *device, (audio, (&mut packed, TR))));
        Span::<VL, VR>::new().forward((&mut *device, (video, (&mut packed, TR + AR))));
        packed
    }
}

/// The video and audio rows of a packed sequence, for the heads.
#[derive(DeviceSend)]
pub struct Split;

impl<
    'd,
    'x,
    K,
    const TL: usize,
    const TR: usize,
    const AL: usize,
    const AR: usize,
    const VL: usize,
    const VR: usize,
    W: AxisName,
    Chips: M,
>
    Module<(
        &'d mut Device,
        (
            &'x Rows<Packed<Span<TL, TR>, Span<AL, AR>, Span<VL, VR>>, W, Chips>,
            K,
        ),
    )> for Split
{
    type Output = (Rows<Span<VL, VR>, W, Chips>, Rows<Span<AL, AR>, W, Chips>);

    fn forward(
        &self,
        (device, (packed, _)): (
            &'d mut Device,
            (
                &'x Rows<Packed<Span<TL, TR>, Span<AL, AR>, Span<VL, VR>>, W, Chips>,
                K,
            ),
        ),
    ) -> Self::Output {
        (
            Span::<VL, VR>::new().forward((&mut *device, (packed, TR + AR))),
            Span::<AL, AR>::new().forward((&mut *device, (packed, TR))),
        )
    }
}

/// A video and an audio span each in rows of its own, `Video` and `Audio` rows long: both are
/// allocated, then the video span lands in the leading rows of the first, then the audio span.
#[derive(DeviceSend)]
pub struct Materialize<Video, Audio>(PhantomData<(Video, Audio)>);

impl<V, A> Materialize<V, A> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<
    'd,
    'm,
    const VL: usize,
    const VR: usize,
    const AL: usize,
    const AR: usize,
    V: AxisName,
    A: AxisName,
    W: AxisName,
    Chips: M,
>
    Module<(
        &'d mut Device,
        &'m (Rows<Span<VL, VR>, W, Chips>, Rows<Span<AL, AR>, W, Chips>),
    )> for Materialize<V, A>
{
    type Output = (Rows<V, W, Chips>, Rows<A, W, Chips>);

    fn forward(
        &self,
        (device, (video, audio)): (
            &'d mut Device,
            &'m (Rows<Span<VL, VR>, W, Chips>, Rows<Span<AL, AR>, W, Chips>),
        ),
    ) -> Self::Output {
        let mut media: Self::Output = (HbmTensor::new(), HbmTensor::new());
        Span::<VL, VR>::new().forward((&mut *device, (video, (&mut media.0, 0))));
        Span::<AL, AR>::new().forward((&mut *device, (audio, (&mut media.1, 0))));
        media
    }
}

/// Rows of the strips a packed sequence's spans come in: 64 when every span holds whole 64-row
/// tiles, else 16.
impl Join {
    const fn edge<T: AxisName, A: AxisName, V: AxisName>() -> usize {
        let wide = Strip::SIZE;
        if T::SIZE % wide == 0 && A::SIZE % wide == 0 && V::SIZE % wide == 0 {
            wide
        } else {
            assert!(T::SIZE % 16 == 0 && A::SIZE % 16 == 0 && V::SIZE % 16 == 0);
            16
        }
    }

    /// Each span's first strip, its strips and its table row, in packing order, in strips of
    /// the sequence's strip rows.
    const fn spans<T: AxisName, A: AxisName, V: AxisName>() -> [(usize, usize, usize); 3] {
        let unit = Self::edge::<T, A, V>();
        let (text, audio) = (T::SIZE / unit, A::SIZE / unit);
        [
            (0, text, 1),
            (text, audio, 5),
            (text + audio, V::SIZE / unit, 0),
        ]
    }
}

/// The tiles of a packed sequence.
#[derive(Clone, Debug)]
pub struct Strips<R>(PhantomData<R>);

impl<R: AxisName> AxisName for Strips<R> {
    const NAME: Ident = Ident::new("Strips");
    const SIZE: usize = R::SIZE / Strip::SIZE;
}

type Banded<R, W> = m![{ Symbol<Strips<R>> }, Strip, W];
type Half = m![Strip / 32];
/// Four groups of eight rows, each over the slices of its features 128 apiece: every slice moves
/// 256-byte runs of a row and the row statistics reduce over the inner digit.
type Slices<W> = m![Strip % 32 / 8, W / 128 # 64];
type Tile<D, W, X = m![1]> = DmTensor<D, X, Half, Slices<W>, m![Strip % 8, W % 128]>;
type Feature<D, W, X = m![1]> = DmTensor<D, X, Half, Slices<W>, m![W % 128]>;
type Operand<W, X = m![1]> = VrfTensor<f32, X, Half, Slices<W>, m![W % 128]>;
type RowOperand<W, X = m![1]> = VrfTensor<f32, X, Half, Slices<W>, m![Strip % 8, 1 # 8]>;

impl<R: AxisName> Strips<R> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<R: AxisName, W: AxisName, X: M>
    Module<(&mut Device, (&Tile<bf16, W, X>, &mut Rows<R, W, X>, usize))> for Strips<R>
{
    type Output = ();

    fn forward(
        &self,
        (device, (tile, rows, at)): (&mut Device, (&Tile<bf16, W, X>, &mut Rows<R, W, X>, usize)),
    ) {
        tile.view().to_hbm_view(
            &mut device.tdma,
            // SAFETY: the tiles split the row axis in row order; every cell keeps its address.
            unsafe { rows.view_mut().reshape::<X, Banded<R, W>>() }.tile::<m![
                { Symbol<Strips<R>> }
            ], 1, m![
                { Symbol<Strips<R>> } = 1 #{!} ({ Symbol<Strips<R>> }),
                Strip,
                W
            ]>(at),
        );
    }
}

/// A landed per-feature vector, widened.
impl<W: AxisName, const EPSILON: usize, X: M> Module<(&mut Device, &Feature<bf16, W, X>)>
    for FinalNorm<W, EPSILON, X>
{
    type Output = Operand<W, X>;

    fn forward(&self, (device, landed): (&mut Device, &Feature<bf16, W, X>)) -> Self::Output {
        device
            .sub
            .begin(landed.view())
            .fetch::<m![W % 128 / 8], m![W % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![W % 128 / 8], m![W % 8]>()
            .to_vrf()
    }
}

/// `bf16(x * rsqrt(mean(x²) + eps) * (w * (1 + scale)) + shift)` with each span's table row, FP32
/// inside; `EPSILON` holds epsilon's FP32 bits.
#[derive(DeviceSend)]
pub struct AdaptiveRms<Width: AxisName, const EPSILON: usize, Chips: M = m![1]> {
    weight: Weight<m![Width], Chips>,
}

impl<Width: AxisName, const EPSILON: usize, Chips: M> AdaptiveRms<Width, EPSILON, Chips> {
    pub const fn new(weight: Weight<m![Width], Chips>) -> Self {
        Self { weight }
    }
}

impl<'d, 'x, 'm, T: AxisName, A: AxisName, V: AxisName, W: AxisName, Chips: M, const EPSILON: usize>
    Module<(
        &'d mut Device,
        (
            &'x Rows<Packed<T, A, V>, W, Chips>,
            &'m (Table<W, Chips>, Table<W, Chips>),
        ),
    )> for AdaptiveRms<W, EPSILON, Chips>
{
    type Output = Rows<Packed<T, A, V>, W, Chips>;

    fn forward(
        &self,
        (device, (input, modulation)): (
            &'d mut Device,
            (
                &'x Rows<Packed<T, A, V>, W, Chips>,
                &'m (Table<W, Chips>, Table<W, Chips>),
            ),
        ),
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        self.forward((device, (input, &mut output, modulation)));
        output
    }
}

/// `bf16(x + gate * branch)` per row, FP32 inside, the gate the table row of the row's span.
impl<'d, 'x, 'b, 'g, T: AxisName, A: AxisName, V: AxisName, W: AxisName, Chips: M>
    Module<(
        &'d mut Device,
        (
            &'x Rows<Packed<T, A, V>, W, Chips>,
            &'b Rows<Packed<T, A, V>, W, Chips>,
            &'g Table<W, Chips>,
        ),
    )> for super::GatedAdd
{
    type Output = Rows<Packed<T, A, V>, W, Chips>;

    fn forward(
        &self,
        (device, (input, branch, gate)): (
            &'d mut Device,
            (
                &'x Rows<Packed<T, A, V>, W, Chips>,
                &'b Rows<Packed<T, A, V>, W, Chips>,
                &'g Table<W, Chips>,
            ),
        ),
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        self.forward((device, (input, branch, &mut output, gate)));
        output
    }
}

/// `bf16(silu(t))` of each timestep's time embedding, FP32 inside and rounded once.
impl<In: AxisName, Chips: M> Module<(&mut Device, HbmTensorView<'_, f32, Chips, m![Timestep, In]>)>
    for crate::ops::activation::Silu
{
    type Output = DmTensor<bf16, Chips, m![2], m![1 # 256], m![Timestep, In]>;

    fn forward(
        &self,
        (device, time): (&mut Device, HbmTensorView<'_, f32, Chips, m![Timestep, In]>),
    ) -> Self::Output {
        let landed: DmTensor<f32, Chips, m![2], m![1 # 256], m![Timestep, In]> =
            time.to_dm(&mut device.tdma);
        device
            .main
            .begin(landed.view())
            .fetch::<m![Timestep, In / 8], m![In % 8]>()
            .collect::<m![Timestep, In / 8], m![In % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Timestep, In / 8, In % 8 / 4 % 2], m![In % 4]>()
            .vector_stash()
            .vector_fp_unary(FpUnaryOp::Sigmoid)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
            .vector_widen_concat::<m![Timestep, In / 8], m![In % 8]>()
            .vector_final()
            .cast::<bf16, m![In % 8 # 16]>()
            .commit_trim::<m![In % 8]>()
            .commit()
    }
}

/// A shift, then a scale, of each feature.
#[derive(Clone, Debug)]
pub struct ShiftScale<W>(PhantomData<W>);

impl<W: AxisName> AxisName for ShiftScale<W> {
    const NAME: Ident = Ident::new("ShiftScale");
    const SIZE: usize = 2 * W::SIZE;
}

/// The final norm's `(shift, scale)` of timestep `TIMESTEP` (video 0, audio 1): `bf16(silu(t))`
/// projected with an FP32 bias, rounded once, the shift features first.
#[derive(DeviceSend)]
pub struct HeadModulation<In: AxisName, W: AxisName, const TIMESTEP: usize, Chips: M = m![1]> {
    weight: HbmTensor<bf16, Chips, m![{ Symbol<ShiftScale<W>> }, In]>,
    bias: HbmTensor<f32, Chips, m![{ Symbol<ShiftScale<W>> }, 1 # 8]>,
}

impl<In: AxisName, W: AxisName, const TIMESTEP: usize, Chips: M>
    HeadModulation<In, W, TIMESTEP, Chips>
{
    pub const fn new(
        weight: HbmTensor<bf16, Chips, m![{ Symbol<ShiftScale<W>> }, In]>,
        bias: HbmTensor<f32, Chips, m![{ Symbol<ShiftScale<W>> }, 1 # 8]>,
    ) -> Self {
        Self { weight, bias }
    }
}

impl<'d, 'x, In: AxisName, W: AxisName, const TIMESTEP: usize, Chips: M>
    Module<(&'d mut Device, &'x HbmTensor<f32, Chips, m![Timestep, In]>)>
    for HeadModulation<In, W, TIMESTEP, Chips>
{
    type Output = Shift<W, Chips>;

    fn forward(
        &self,
        (device, time): (&'d mut Device, &'x HbmTensor<f32, Chips, m![Timestep, In]>),
    ) -> Self::Output {
        let activated = crate::ops::activation::Silu.forward((&mut *device, time.view()));
        let projected = crate::ops::linear::kernel::Biased::<Timestep>(PhantomData)
            .forward((&mut *device, ((&self.weight, &self.bias), activated.view())));
        let mut packed: HbmTensor<bf16, Chips, m![Timestep, { Symbol<ShiftScale<W>> }]> =
            HbmTensor::new();
        projected
            .view()
            .to_hbm_view(&mut device.tdma, packed.view_mut());
        // SAFETY: a timestep's outputs are its shift, then its scale.
        let packed = unsafe { packed.view().reshape::<Chips, m![Timestep, Part, W]>() };
        let part = |device: &mut Device, part: usize| -> HbmTensor<bf16, Chips, m![W]> {
            let mut vector = HbmTensor::new();
            packed
                .tile::<m![Timestep], 1, m![Timestep = 1 # 2, Part, W]>(TIMESTEP)
                .tile::<m![Part], 1, m![Timestep = 1 # 2, Part = 1 # 2, W]>(part)
                .to_hbm_view(
                    &mut device.tdma,
                    // SAFETY: a part is its features in order.
                    unsafe {
                        vector
                            .view_mut()
                            .reshape::<Chips, m![Timestep = 1 # 2, Part = 1 # 2, W]>()
                    },
                );
            vector
        };
        (part(device, 0), part(device, 1))
    }
}

/// Six modulation chunks of each modality: one projection output per timestep.
#[derive(Clone, Debug)]
pub struct Modulations<W>(PhantomData<W>);

impl<W: AxisName> AxisName for Modulations<W> {
    const NAME: Ident = Ident::new("Modulations");
    const SIZE: usize = Modality::SIZE * Chunks::SIZE * W::SIZE;
}

type Outputs<W> = Symbol<Modulations<W>>;

/// The chips of an `N`-way split.
type Chips<const N: usize> = m![{ Symbol<Shard<N>> }];

/// A layer's attention and feed-forward `((shift, scale), gate)` tables.
pub type Gates<W, Chips = m![1]> = ((Table<W, Chips>, Table<W, Chips>), Table<W, Chips>);

/// A layer's modulation tables, `bf16(silu(t))` projected with an FP32 bias: feature
/// `modality * 6W + chunk * W + j` lands in table `chunk`, row `timestep * 3 + modality`.
#[derive(DeviceSend)]
pub struct Projection<In: AxisName, W: AxisName, Chips: M = m![1], L: AxisName = W> {
    weight: HbmTensor<bf16, Chips, m![{ Outputs<L> }, In]>,
    bias: HbmTensor<f32, Chips, m![{ Outputs<L> }, 1 # 8]>,
    width: PhantomData<W>,
}

impl<In: AxisName, W: AxisName, Chips: M, L: AxisName> Projection<In, W, Chips, L> {
    pub const fn new(
        weight: HbmTensor<bf16, Chips, m![{ Outputs<L> }, In]>,
        bias: HbmTensor<f32, Chips, m![{ Outputs<L> }, 1 # 8]>,
    ) -> Self {
        Self {
            weight,
            bias,
            width: PhantomData,
        }
    }
}

impl<In: AxisName, W: AxisName, Chips: M, L: AxisName>
    Module<(
        &mut Device,
        (HbmTensorView<'_, f32, Chips, m![Timestep, In]>,),
    )> for Projection<In, W, Chips, L>
{
    type Output = HbmTensor<bf16, Chips, m![Timestep, { Outputs<L> }]>;

    fn forward(
        &self,
        (device, (time,)): (
            &mut Device,
            (HbmTensorView<'_, f32, Chips, m![Timestep, In]>,),
        ),
    ) -> Self::Output {
        let activated = crate::ops::activation::Silu.forward((&mut *device, time));
        let projected = crate::ops::linear::kernel::Biased::<Timestep>(PhantomData)
            .forward((&mut *device, ((&self.weight, &self.bias), activated.view())));
        let mut packed = HbmTensor::new();
        projected
            .view()
            .to_hbm_view(&mut device.tdma, packed.view_mut());
        packed
    }
}

impl<'d, 'x, In: AxisName, W: AxisName, Chips: M>
    Module<(&'d mut Device, &'x HbmTensor<f32, Chips, m![Timestep, In]>)>
    for Projection<In, W, Chips>
{
    type Output = (Gates<W, Chips>, Gates<W, Chips>);

    fn forward(
        &self,
        (device, time): (&'d mut Device, &'x HbmTensor<f32, Chips, m![Timestep, In]>),
    ) -> Self::Output {
        let packed = self.forward((&mut *device, (time.view(),)));
        // SAFETY: an output feature is its modality, its chunk and its feature, in that order.
        let packed = unsafe {
            packed
                .view()
                .reshape::<Chips, m![Timestep, Modality, Chunks, W]>()
        };
        let table = |device: &mut Device, chunk: usize| -> Table<W, Chips> {
            let mut table: Table<W, Chips> = HbmTensor::new();
            packed
                .tile::<m![Chunks], 1, m![Timestep, Modality, Chunks = 1 # 6, W]>(chunk)
                .to_hbm_view(
                    &mut device.tdma,
                    // SAFETY: a table row is `timestep * 3 + modality`.
                    unsafe {
                        table
                            .view_mut()
                            .reshape::<Chips, m![Timestep, Modality, W]>()
                    },
                );
            table
        };
        let attention = ((table(device, 0), table(device, 1)), table(device, 2));
        let feedforward = ((table(device, 3), table(device, 4)), table(device, 5));
        (attention, feedforward)
    }
}

/// Features split over `N` chips: each chip projects its part, and every table lands on every
/// chip with the parts in chip order, one transfer from each chip to each chip.
impl<'d, 'x, In: AxisName, W: AxisName, const N: usize>
    Module<(
        &'d mut Device,
        &'x HbmTensor<f32, Chips<N>, m![Timestep, In]>,
    )> for Projection<In, W, Chips<N>, axes::Part<W, N>>
{
    type Output = (Gates<W, Chips<N>>, Gates<W, Chips<N>>);

    fn forward(
        &self,
        (device, time): (
            &'d mut Device,
            &'x HbmTensor<f32, Chips<N>, m![Timestep, In]>,
        ),
    ) -> Self::Output {
        type L<W, const N: usize> = Symbol<axes::Part<W, N>>;
        let packed = self.forward((&mut *device, (time.view(),)));
        // SAFETY: an output feature is its modality, its chunk and its part's feature, in order.
        let packed = unsafe {
            packed
                .view()
                .reshape::<Chips<N>, m![Timestep, Modality, Chunks, { L<W, N> }]>()
        };
        let table = |device: &mut Device, chunk: usize| -> Table<W, Chips<N>> {
            let mut table: HbmTensor<
                bf16,
                Broadcast<N>,
                m![Timestep, Modality, { Symbol<Shard<N>> }, { L<W, N> }],
            > = HbmTensor::new();
            packed
                .tile::<m![Chunks], 1, m![Timestep, Modality, Chunks = 1 # 6, { L<W, N> }]>(chunk)
                .to_hbm_view(&mut device.tdma, table.view_mut());
            // SAFETY: a table row is `timestep * 3 + modality`, its features the parts in chip
            // order, on every chip.
            unsafe { table.reshape() }
        };
        let attention = ((table(device, 0), table(device, 1)), table(device, 2));
        let feedforward = ((table(device, 3), table(device, 4)), table(device, 5));
        (attention, feedforward)
    }
}

/// BF16 rows projected with a per-output FP32 offset: each FP32 sum plus its offset, rounded once
/// to BF16.
#[derive(DeviceSend)]
pub struct Biased<In: AxisName, Out: AxisName, X: M = m![1]> {
    weight: crate::ops::linear::Linear<In, Out, X>,
    bias: HbmTensor<f32, X, m![Out]>,
}

impl<In: AxisName, Out: AxisName, X: M> Biased<In, Out, X> {
    pub fn new(weight: Weight<m![Out, In], X>, bias: HbmTensor<f32, X, m![Out]>) -> Self {
        Self {
            weight: crate::ops::linear::Linear::new(weight),
            bias,
        }
    }
}

impl<'d, 'x, In: AxisName, Out: AxisName, R: AxisName, X: M>
    Module<(&'d mut Device, &'x Rows<R, In, X>)> for Biased<In, Out, X>
{
    type Output = Rows<R, Out, X>;

    fn forward(&self, (device, input): (&'d mut Device, &'x Rows<R, In, X>)) -> Self::Output {
        type C<W> = Symbol<Chunk<W>>;
        type Sums<D, W, X> = DmTensor<D, X, Half, FeatureSlices<W>, m![Strip % 32, W % ({ C<W> })]>;
        let mut sums: HbmTensor<f32, X, m![R, Out]> = HbmTensor::new();
        self.weight.forward((&mut *device, (input, &mut sums)));
        let bias: DmTensor<f32, X, Half, FeatureSlices<Out>, m![Out % ({ C<Out> })]> =
            self.bias.view().to_dm(&mut device.tdma);
        let bias: VrfTensor<f32, X, Half, FeatureSlices<Out>, m![Out % ({ C<Out> })]> = device
            .sub
            .begin(bias.view())
            .fetch::<m![Out % ({ C<Out> }) / 8], m![Out % 8]>()
            .collect::<m![Out % ({ C<Out> }) / 8], m![Out % 8]>()
            .to_vrf();
        // SAFETY: the tiles split the rows in order.
        let sums = unsafe { sums.view().reshape::<X, Banded<R, Out>>() };
        let mut output = HbmTensor::new();
        for tile in 0..Strips::<R>::SIZE {
            let landed: Sums<f32, Out, X> = sums
                .tile::<m![{ Symbol<Strips<R>> }], 1, m![
                    { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }),
                    Strip,
                    Out
                ]>(tile)
                .to_dm(&mut device.tdma);
            let rounded: Sums<bf16, Out, X> = device
                .main
                .begin(landed.view())
                .fetch::<m![Strip % 32, Out % ({ C<Out> }) / 8], m![Out % 8]>()
                .collect::<m![Strip % 32, Out % ({ C<Out> }) / 8], m![Out % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, &bias)
                .vector_final()
                .cast::<bf16, m![Out % 8 # 16]>()
                .commit_trim::<m![Out % 8]>()
                .commit();
            rounded.view().to_hbm_view(
                &mut device.tdma,
                // SAFETY: as for the sums.
                unsafe { output.view_mut().reshape::<X, Banded<R, Out>>() }.tile::<m![
                    { Symbol<Strips<R>> }
                ], 1, m![
                    { Symbol<Strips<R>> } = 1 #{!} ({ Symbol<Strips<R>> }),
                    Strip,
                    Out
                ]>(tile),
            );
        }
        output
    }
}

/// Two features per output feature: its BF16 high part, then its BF16 low part.
#[derive(Clone, Debug)]
pub struct Stacked<Out>(PhantomData<Out>);

impl<Out: AxisName> AxisName for Stacked<Out> {
    const NAME: Ident = Ident::new("Stacked");
    const SIZE: usize = 2 * Out::SIZE;
}

/// Two features per input feature: its BF16 high part's, then its low part's.
#[derive(Clone, Debug)]
pub struct Sides<In>(PhantomData<In>);

impl<In: AxisName> AxisName for Sides<In> {
    const NAME: Ident = Ident::new("Sides");
    const SIZE: usize = 2 * In::SIZE;
}

axes![Part = 2, Frequency = 128];

/// FP32 coefficients as stacked BF16 high and low parts with FP32 offsets: one contraction lands
/// both sums, which with the offset make the FP32 product to within 2^-17 of each term.
#[derive(DeviceSend)]
pub struct Precise<In: AxisName, Out: AxisName, X: M = m![1]> {
    weight: crate::ops::linear::Linear<In, Stacked<Out>, X>,
    bias: HbmTensor<f32, X, m![Out]>,
}

impl<In: AxisName, Out: AxisName, X: M> Precise<In, Out, X> {
    pub fn new(
        parts: Weight<m![{ Symbol<Stacked<Out>> }, In], X>,
        bias: HbmTensor<f32, X, m![Out]>,
    ) -> Self {
        Self {
            weight: crate::ops::linear::Linear::new(parts),
            bias,
        }
    }
}

impl<'d, 'x, In: AxisName, Out: AxisName, R: AxisName, X: M>
    Module<(&'d mut Device, &'x Rows<R, In, X>)> for Precise<In, Out, X>
{
    type Output = HbmTensor<f32, X, m![R, Out]>;

    fn forward(&self, (device, input): (&'d mut Device, &'x Rows<R, In, X>)) -> Self::Output {
        // A row a slice: an output this narrow spreads over too few slices to share them.
        type Each = m![Strip % 32 # 256];
        type Sums<Out, X> = DmTensor<f32, X, Half, Each, m![Out]>;
        let mut sums: HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]> = HbmTensor::new();
        self.weight.forward((&mut *device, (input, &mut sums)));
        let bias: DmTensor<f32, X, Half, Each, m![Out]> = self.bias.view().to_dm(&mut device.tdma);
        let bias: VrfTensor<f32, X, Half, Each, m![Out]> = device
            .sub
            .begin(bias.view())
            .fetch::<m![Out / 8], m![Out % 8]>()
            .collect::<m![Out / 8], m![Out % 8]>()
            .to_vrf();
        // SAFETY: a row's stacked sums are its high parts, then its low parts, and the tiles split
        // the rows in order.
        let sums = unsafe {
            sums.view()
                .reshape::<X, m![{ Symbol<Strips<R>> }, Strip, Part, Out]>()
        };
        let mut output = HbmTensor::new();
        for tile in 0..Strips::<R>::SIZE {
            let part = |device: &mut Device, part: usize| -> Sums<Out, X> {
                sums.tile::<m![{ Symbol<Strips<R>> }], 1, m![
                    { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }),
                    Strip,
                    Part,
                    Out
                ]>(tile)
                    .tile::<m![Part], 1, m![
                        { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }),
                        Strip,
                        Part = 1 # 2,
                        Out
                    ]>(part)
                    .to_dm(&mut device.tdma)
            };
            let (high, low) = (part(device, 0), part(device, 1));
            let low: VrfTensor<f32, X, Half, Each, m![Out]> = device
                .sub
                .begin(low.view())
                .fetch::<m![Out / 8], m![Out % 8]>()
                .collect::<m![Out / 8], m![Out % 8]>()
                .to_vrf();
            let sum: Sums<Out, X> = device
                .main
                .begin(high.view())
                .fetch::<m![Out / 8], m![Out % 8]>()
                .collect::<m![Out / 8], m![Out % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Out / 8, Out % 8 / 4 % 2], m![Out % 4]>()
                .vector_fp_binary(FpBinaryOp::AddF, &low)
                .vector_widen_concat::<m![Out / 8], m![Out % 8]>()
                .vector_clip(ClipBinaryOpF32::Add, &bias)
                .vector_final()
                .commit_trim::<m![Out % 8]>()
                .commit();
            sum.view().to_hbm_view(
                &mut device.tdma,
                // SAFETY: the tiles split the rows in order.
                unsafe { output.view_mut().reshape::<X, Banded<R, Out>>() }.tile::<m![
                    { Symbol<Strips<R>> }
                ], 1, m![
                    { Symbol<Strips<R>> } = 1 #{!} ({ Symbol<Strips<R>> }),
                    Strip,
                    Out
                ]>(tile),
            );
        }
        output
    }
}

/// One modality's final norm: RMS normalization rounded to BF16, then `bf16(n * (1 + scale) +
/// shift)` of its timestep row; `EPSILON` holds epsilon's FP32 bits.
#[derive(DeviceSend)]
pub struct FinalNorm<Width: AxisName, const EPSILON: usize, X: M = m![1]> {
    weight: Weight<m![Width], X>,
}

impl<Width: AxisName, const EPSILON: usize, X: M> FinalNorm<Width, EPSILON, X> {
    pub const fn new(weight: Weight<m![Width], X>) -> Self {
        Self { weight }
    }
}

/// One timestep's `(shift, scale)`.
pub type Shift<W, Chips = m![1]> = (HbmTensor<bf16, Chips, m![W]>, HbmTensor<bf16, Chips, m![W]>);

impl<'d, 'x, 's, R: AxisName, W: AxisName, const EPSILON: usize, X: M>
    Module<(&'d mut Device, (&'x Rows<R, W, X>, &'s Shift<W, X>))> for FinalNorm<W, EPSILON, X>
{
    type Output = Rows<R, W, X>;

    fn forward(
        &self,
        (device, (input, (shift, scale))): (&'d mut Device, (&'x Rows<R, W, X>, &'s Shift<W, X>)),
    ) -> Self::Output {
        let weight: Feature<bf16, W, X> = self.weight.tensor().view().to_dm(&mut device.tdma);
        let weight = self.forward((&mut *device, &weight));
        let scale: Feature<bf16, W, X> = scale.view().to_dm(&mut device.tdma);
        let gain: Operand<W, X> = device
            .main
            .begin(scale.view())
            .fetch::<m![W % 128 / 8], m![W % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![W % 128 / 8], m![W % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![W % 128 / 8, W % 8 / 4 % 2], m![W % 4]>()
            .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)
            .vector_widen_concat::<m![W % 128 / 8], m![W % 8]>()
            .vector_final()
            .to_vrf(&mut device.sub);
        let shift: Feature<bf16, W, X> = shift.view().to_dm(&mut device.tdma);
        let shift = self.forward((&mut *device, &shift));
        let mut output = HbmTensor::new();
        for tile in 0..Strips::<R>::SIZE {
            let rows: Tile<bf16, W, X> = {
                // SAFETY: the tiles split the row axis in row order; every cell keeps its address.
                unsafe { input.view().reshape::<X, Banded<R, W>>() }
                    .tile::<m![{ Symbol<Strips<R>> }], 1, m![
                        { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }), Strip, W
                    ]>(tile)
                    .to_dm(&mut device.tdma)
            };
            let inverse: RowOperand<W, X> = {
                let variance: DmTensor<f32, X, Half, m![Strip % 32 / 8, 64], m![Strip % 8, 1 # 8]> = device
                    .main
                    .begin(rows.view())
                    .fetch::<m![Strip % 8, W % 128 / 8], m![W % 8]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Strip % 8, W % 128 / 8], m![W % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Strip % 8, W % 128 / 8, W % 8 / 4 % 2], m![W % 4]>()
                    .vector_mean_square::<W, m![Strip % 8], m![W / 128], EPSILON>()
                    .vector_inter_slice_reduce::<m![Strip % 32 / 8, 64], m![Strip % 8]>(InterSliceReduceOpF32::Add)
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                // SAFETY: every slice holds the same row statistic; each row keeps its address.
                let variance: DmTensor<f32, X, Half, Slices<W>, m![Strip % 8, 1 # 8]> =
                    unsafe { variance.reshape() };
                device
                    .sub
                    .begin(variance.view())
                    .fetch::<m![Strip % 8], m![1 # 8]>()
                    .collect::<m![Strip % 8], m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_trim::<m![1 # 4]>()
                    .vector_fp_unary(FpUnaryOp::Sqrt)
                    .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .to_vrf()
            };
            let normalized: Tile<bf16, W, X> = device
                .main
                .begin(rows.view())
                .fetch::<m![Strip % 8, W % 128 / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Strip % 8, W % 128 / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Strip % 8, W % 128 / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inverse)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight)
                .vector_widen_concat::<m![Strip % 8, W % 128 / 8], m![W % 8]>()
                .vector_final()
                .cast::<bf16, m![W % 8 # 16]>()
                .commit_trim::<m![W % 8]>()
                .commit();
            let modulated: Tile<bf16, W, X> = device
                .main
                .begin(normalized.view())
                .fetch::<m![Strip % 8, W % 128 / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Strip % 8, W % 128 / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Strip % 8, W % 128 / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gain)
                .vector_fp_binary(FpBinaryOp::AddF, &shift)
                .vector_widen_concat::<m![Strip % 8, W % 128 / 8], m![W % 8]>()
                .vector_final()
                .cast::<bf16, m![W % 8 # 16]>()
                .commit_trim::<m![W % 8]>()
                .commit();
            Strips::<R>::new().forward((&mut *device, (&modulated, &mut output, tile)));
        }
        output
    }
}

/// FP32 rows times FP32 coefficients plus offsets, rounded once: rows as BF16 high and low parts,
/// coefficients as `[high | high]` and `[low | 0]`, within 2^-17 of each term.
#[derive(DeviceSend)]
pub struct Embedding<In: AxisName, Out: AxisName, X: M = m![1]> {
    weight: crate::ops::linear::Linear<Sides<In>, Stacked<Out>, X>,
    bias: HbmTensor<f32, X, m![Out]>,
}

/// One part of tile `at`'s rows, stored beside the other.
impl<In: AxisName> Sides<In> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<In: AxisName, R: AxisName, S: M, E: M, X: M>
    Module<(
        &mut Device,
        (
            &DmTensor<bf16, X, Half, S, E>,
            &mut Rows<R, Sides<In>, X>,
            (usize, usize),
        ),
    )> for Sides<In>
{
    type Output = ();

    fn forward(
        &self,
        (device, (values, parts, (at, part))): (
            &mut Device,
            (
                &DmTensor<bf16, X, Half, S, E>,
                &mut Rows<R, Sides<In>, X>,
                (usize, usize),
            ),
        ),
    ) {
        values.view().to_hbm_view(
            &mut device.tdma,
            // SAFETY: a row's parts are its high part, then its low part, and the tiles split the rows
            // in order.
            unsafe {
                parts
                    .view_mut()
                    .reshape::<X, m![{ Symbol<Strips<R>> }, Strip, Part, In]>()
            }
            .tile::<m![{ Symbol<Strips<R>> }], 1, m![
                { Symbol<Strips<R>> } = 1 #{!} ({ Symbol<Strips<R>> }),
                Strip,
                Part,
                In
            ]>(at)
            .tile::<m![Part], 1, m![
                { Symbol<Strips<R>> } = 1 #{!} ({ Symbol<Strips<R>> }),
                Strip,
                Part = 1 #{!} 2,
                In
            ]>(part),
        );
    }
}

impl<In: AxisName, Out: AxisName, X: M> Embedding<In, Out, X> {
    pub fn new(
        parts: Weight<m![{ Symbol<Stacked<Out>> }, { Symbol<Sides<In>> }], X>,
        bias: HbmTensor<f32, X, m![Out]>,
    ) -> Self {
        Self {
            weight: crate::ops::linear::Linear::new(parts),
            bias,
        }
    }
}

impl<'d, 'x, In: AxisName, Out: AxisName, R: AxisName, X: M>
    Module<(&'d mut Device, &'x HbmTensor<f32, X, m![R, In]>)> for Embedding<In, Out, X>
{
    type Output = Rows<R, Out, X>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x HbmTensor<f32, X, m![R, In]>),
    ) -> Self::Output {
        type Each = m![Strip % 32 # 256];
        type Wide<W, X> = DmTensor<f32, X, Half, Slices<W>, m![Strip % 8, W % 128]>;
        // Each row's high and low parts, side by side.
        let mut parts: Rows<R, Sides<In>, X> = HbmTensor::new();
        for tile in 0..Strips::<R>::SIZE {
            let rows: DmTensor<f32, X, Half, Each, m![In]> =
                // SAFETY: the tiles split the rows in order.
                unsafe { input.view().reshape::<X, Banded<R, In>>() }
                    .tile::<m![{ Symbol<Strips<R>> }], 1, m![
                        { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }),
                        Strip,
                        In
                    ]>(tile)
                    .to_dm(&mut device.tdma);
            let high: DmTensor<bf16, X, Half, Each, m![In]> = device
                .main
                .begin(rows.view())
                .fetch::<m![In / 8], m![In % 8]>()
                .collect::<m![In / 8], m![In % 8]>()
                .cast::<bf16, m![In % 8 # 16]>()
                .commit_trim::<m![In % 8]>()
                .commit();
            let widened: VrfTensor<f32, X, Half, Each, m![In]> = device
                .sub
                .begin(high.view())
                .fetch::<m![In / 8], m![In % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![In / 8], m![In % 8]>()
                .to_vrf();
            let low: DmTensor<bf16, X, Half, Each, m![In]> = device
                .main
                .begin(rows.view())
                .fetch::<m![In / 8], m![In % 8]>()
                .collect::<m![In / 8], m![In % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![In / 8, In % 8 / 4 % 2], m![In % 4]>()
                .vector_fp_binary(FpBinaryOp::SubF, &widened)
                .vector_widen_concat::<m![In / 8], m![In % 8]>()
                .vector_final()
                .cast::<bf16, m![In % 8 # 16]>()
                .commit_trim::<m![In % 8]>()
                .commit();
            Sides::<In>::new().forward((&mut *device, (&high, &mut parts, (tile, 0))));
            Sides::<In>::new().forward((&mut *device, (&low, &mut parts, (tile, 1))));
        }
        let mut sums: HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]> = HbmTensor::new();
        self.weight.forward((&mut *device, (&parts, &mut sums)));
        let bias: Feature<f32, Out, X> = self.bias.view().to_dm(&mut device.tdma);
        let bias: Operand<Out, X> = device
            .sub
            .begin(bias.view())
            .fetch::<m![Out % 128 / 8], m![Out % 8]>()
            .collect::<m![Out % 128 / 8], m![Out % 8]>()
            .to_vrf();
        // SAFETY: a row's stacked sums are its first sums, then its second, and the tiles split the
        // rows in order.
        let sums = unsafe {
            sums.view()
                .reshape::<X, m![{ Symbol<Strips<R>> }, Strip, Part, Out]>()
        };
        let mut output = HbmTensor::new();
        for tile in 0..Strips::<R>::SIZE {
            let part = |device: &mut Device, part: usize| -> Wide<Out, X> {
                sums.tile::<m![{ Symbol<Strips<R>> }], 1, m![
                    { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }),
                    Strip,
                    Part,
                    Out
                ]>(tile)
                    .tile::<m![Part], 1, m![
                        { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }),
                        Strip,
                        Part = 1 # 2,
                        Out
                    ]>(part)
                    .to_dm(&mut device.tdma)
            };
            let (first, second) = (part(device, 0), part(device, 1));
            let second: VrfTensor<f32, X, Half, Slices<Out>, m![Strip % 8, Out % 128]> = device
                .sub
                .begin(second.view())
                .fetch::<m![Strip % 8, Out % 128 / 8], m![Out % 8]>()
                .collect::<m![Strip % 8, Out % 128 / 8], m![Out % 8]>()
                .to_vrf();
            let sum: Tile<bf16, Out, X> = device
                .main
                .begin(first.view())
                .fetch::<m![Strip % 8, Out % 128 / 8], m![Out % 8]>()
                .collect::<m![Strip % 8, Out % 128 / 8], m![Out % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Strip % 8, Out % 128 / 8, Out % 8 / 4 % 2], m![Out % 4]>()
                .vector_fp_binary(FpBinaryOp::AddF, &second)
                .vector_widen_concat::<m![Strip % 8, Out % 128 / 8], m![Out % 8]>()
                .vector_clip(ClipBinaryOpF32::Add, &bias)
                .vector_final()
                .cast::<bf16, m![Out % 8 # 16]>()
                .commit_trim::<m![Out % 8]>()
                .commit();
            Strips::<R>::new().forward((&mut *device, (&sum, &mut output, tile)));
        }
        output
    }
}

/// A timestep's sinusoid: the cosines of its frequencies, then their sines.
#[derive(Clone, Debug)]
pub struct Sinusoid;

impl AxisName for Sinusoid {
    const NAME: Ident = Ident::new("Sinusoid");
    const SIZE: usize = 2 * Frequency::SIZE;
}

/// FP32 sums of a stacked projection, their two halves added with the FP32 offset.
impl<In: AxisName, Out: AxisName, X: M, R: AxisName>
    Module<(
        &mut Device,
        (&HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]>,),
    )> for Embedding<In, Out, X>
{
    type Output = HbmTensor<f32, X, m![R, Out]>;

    fn forward(
        &self,
        (device, (sums,)): (
            &mut Device,
            (&HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]>,),
        ),
    ) -> Self::Output {
        type C<W> = Symbol<Chunk<W>>;
        type Wide<W, X> = DmTensor<f32, X, Half, FeatureSlices<W>, m![Strip % 32, W % ({ C<W> })]>;
        let bias: DmTensor<f32, X, Half, FeatureSlices<Out>, m![Out % ({ C<Out> })]> =
            self.bias.view().to_dm(&mut device.tdma);
        let bias: VrfTensor<f32, X, Half, FeatureSlices<Out>, m![Out % ({ C<Out> })]> = device
            .sub
            .begin(bias.view())
            .fetch::<m![Out % ({ C<Out> }) / 8], m![Out % 8]>()
            .collect::<m![Out % ({ C<Out> }) / 8], m![Out % 8]>()
            .to_vrf();
        // SAFETY: a row's stacked sums are its first sums, then its second, and the tiles split the
        // rows in order.
        let sums = unsafe {
            sums.view()
                .reshape::<X, m![{ Symbol<Strips<R>> }, Strip, Part, Out]>()
        };
        let mut output = HbmTensor::new();
        for tile in 0..Strips::<R>::SIZE {
            let part = |device: &mut Device, part: usize| -> Wide<Out, X> {
                sums.tile::<m![{ Symbol<Strips<R>> }], 1, m![
                    { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }),
                    Strip,
                    Part,
                    Out
                ]>(tile)
                    .tile::<m![Part], 1, m![
                        { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }),
                        Strip,
                        Part = 1 # 2,
                        Out
                    ]>(part)
                    .to_dm(&mut device.tdma)
            };
            let (first, second) = (part(device, 0), part(device, 1));
            let second: VrfTensor<
                f32,
                X,
                Half,
                FeatureSlices<Out>,
                m![Strip % 32, Out % ({ C<Out> })],
            > = device
                .sub
                .begin(second.view())
                .fetch::<m![Strip % 32, Out % ({ C<Out> }) / 8], m![Out % 8]>()
                .collect::<m![Strip % 32, Out % ({ C<Out> }) / 8], m![Out % 8]>()
                .to_vrf();
            let sum: Wide<Out, X> = device
            .main
            .begin(first.view())
            .fetch::<m![Strip % 32, Out % ({ C<Out> }) / 8], m![Out % 8]>()
            .collect::<m![Strip % 32, Out % ({ C<Out> }) / 8], m![Out % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Strip % 32, Out % ({ C<Out> }) / 8, Out % 8 / 4 % 2], m![Out % 4]>()
            .vector_fp_binary(FpBinaryOp::AddF, &second)
            .vector_widen_concat::<m![Strip % 32, Out % ({ C<Out> }) / 8], m![Out % 8]>()
            .vector_clip(ClipBinaryOpF32::Add, &bias)
            .vector_final()
            .commit_trim::<m![Out % 8]>()
            .commit();
            sum.view().to_hbm_view(
                &mut device.tdma,
                // SAFETY: the tiles split the rows in order.
                unsafe { output.view_mut().reshape::<X, Banded<R, Out>>() }.tile::<m![
                    { Symbol<Strips<R>> }
                ], 1, m![
                    { Symbol<Strips<R>> } = 1 #{!} ({ Symbol<Strips<R>> }),
                    Strip,
                    Out
                ]>(tile),
            );
        }
        output
    }
}

/// FP32 rows as their BF16 high and low parts, side by side.
impl<In: AxisName, R: AxisName, X: M> Module<(&mut Device, &HbmTensor<f32, X, m![R, In]>)>
    for Sides<In>
{
    type Output = Rows<R, Sides<In>, X>;

    fn forward(
        &self,
        (device, input): (&mut Device, &HbmTensor<f32, X, m![R, In]>),
    ) -> Self::Output {
        type C<W> = Symbol<Chunk<W>>;
        type Values<D, W, X> =
            DmTensor<D, X, Half, FeatureSlices<W>, m![Strip % 32, W % ({ C<W> })]>;
        let mut parts: Rows<R, Sides<In>, X> = HbmTensor::new();
        for tile in 0..Strips::<R>::SIZE {
            let rows: Values<f32, In, X> =
            // SAFETY: the tiles split the rows in order.
            unsafe { input.view().reshape::<X, Banded<R, In>>() }
                .tile::<m![{ Symbol<Strips<R>> }], 1, m![
                    { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }),
                    Strip,
                    In
                ]>(tile)
                .to_dm(&mut device.tdma);
            let high: Values<bf16, In, X> = device
                .main
                .begin(rows.view())
                .fetch::<m![Strip % 32, In % ({ C<In> }) / 8], m![In % 8]>()
                .collect::<m![Strip % 32, In % ({ C<In> }) / 8], m![In % 8]>()
                .cast::<bf16, m![In % 8 # 16]>()
                .commit_trim::<m![In % 8]>()
                .commit();
            let widened: VrfTensor<
                f32,
                X,
                Half,
                FeatureSlices<In>,
                m![Strip % 32, In % ({ C<In> })],
            > = device
                .sub
                .begin(high.view())
                .fetch::<m![Strip % 32, In % ({ C<In> }) / 8], m![In % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Strip % 32, In % ({ C<In> }) / 8], m![In % 8]>()
                .to_vrf();
            let low: Values<bf16, In, X> = device
            .main
            .begin(rows.view())
            .fetch::<m![Strip % 32, In % ({ C<In> }) / 8], m![In % 8]>()
            .collect::<m![Strip % 32, In % ({ C<In> }) / 8], m![In % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Strip % 32, In % ({ C<In> }) / 8, In % 8 / 4 % 2], m![In % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &widened)
            .vector_widen_concat::<m![Strip % 32, In % ({ C<In> }) / 8], m![In % 8]>()
            .vector_final()
            .cast::<bf16, m![In % 8 # 16]>()
            .commit_trim::<m![In % 8]>()
            .commit();
            self.forward((&mut *device, (&high, &mut parts, (tile, 0))));
            self.forward((&mut *device, (&low, &mut parts, (tile, 1))));
        }
        parts
    }
}

/// The time embedding of `t` in `[0, 1]`: `[cos | sin](t f)` with `f_j = 10000^(-j / 128)`, then
/// FP32 projection, SiLU and FP32 projection, each within 2^-17 of each term.
#[derive(DeviceSend)]
pub struct TimeEmbedding<Hidden: AxisName, Out: AxisName, X: M = m![1]> {
    frequencies: HbmTensor<f32, X, m![Frequency]>,
    first: Embedding<Sinusoid, Hidden, X>,
    second: Embedding<Hidden, Out, X>,
}

impl<Hidden: AxisName, Out: AxisName, X: M> TimeEmbedding<Hidden, Out, X> {
    pub const fn new(
        frequencies: HbmTensor<f32, X, m![Frequency]>,
        first: Embedding<Sinusoid, Hidden, X>,
        second: Embedding<Hidden, Out, X>,
    ) -> Self {
        Self {
            frequencies,
            first,
            second,
        }
    }
}

impl<'d, 'x, Hidden: AxisName, Out: AxisName, R: AxisName, X: M>
    Module<(&'d mut Device, &'x HbmTensor<f32, X, m![R, 1 # 8]>)>
    for TimeEmbedding<Hidden, Out, X>
{
    type Output = HbmTensor<f32, X, m![R, Out]>;

    fn forward(
        &self,
        (device, time): (&'d mut Device, &'x HbmTensor<f32, X, m![R, 1 # 8]>),
    ) -> Self::Output {
        use crate::ops::activation::Silu;
        type Each = m![Strip % 32 # 256];
        type Waves<X> = DmTensor<f32, X, Half, Each, m![Part, Frequency]>;
        let frequencies: DmTensor<f32, X, Half, Each, m![Frequency]> =
            self.frequencies.view().to_dm(&mut device.tdma);
        let mut sinusoids: HbmTensor<f32, X, m![R, Sinusoid]> = HbmTensor::new();
        for tile in 0..Strips::<R>::SIZE {
            let time: DmTensor<f32, X, Half, Each, m![1 # 8]> =
                // SAFETY: the tiles split the rows in order.
                unsafe { time.view().reshape::<X, m![{ Symbol<Strips<R>> }, Strip, 1 # 8]>() }
                    .tile::<m![{ Symbol<Strips<R>> }], 1, m![
                        { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }),
                        Strip,
                        1 # 8
                    ]>(tile)
                    .to_dm(&mut device.tdma);
            let time: VrfTensor<f32, X, Half, Each, m![1 # 8]> = device
                .sub
                .begin(time.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .to_vrf();
            let mut waves = Waves::<X>::new();
            let wave = |device: &mut Device, op: FpUnaryOp, waves: &mut Waves<X>, part: usize| {
                device
                    .main
                    .begin(frequencies.view())
                    .fetch::<m![Frequency / 8], m![Frequency % 8]>()
                    .collect::<m![Frequency / 8], m![Frequency % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Frequency / 8, Frequency % 8 / 4 % 2], m![Frequency % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &time)
                    .vector_fp_unary(op)
                    .vector_widen_concat::<m![Frequency / 8], m![Frequency % 8]>()
                    .vector_final()
                    .commit_trim::<m![Frequency % 8]>()
                    .commit_view(
                        waves
                            .view_mut()
                            .tile::<m![Part], 1, m![Part = 1 #{!} 2, Frequency]>(part),
                    );
            };
            wave(device, FpUnaryOp::Cos, &mut waves, 0);
            wave(device, FpUnaryOp::Sin, &mut waves, 1);
            waves.view().to_hbm_view(
                &mut device.tdma,
                // SAFETY: a sinusoid is its cosines, then its sines, and the tiles split the rows in
                // order.
                unsafe {
                    sinusoids
                        .view_mut()
                        .reshape::<X, m![{ Symbol<Strips<R>> }, Strip, Part, Frequency]>()
                }
                .tile::<m![{ Symbol<Strips<R>> }], 1, m![
                    { Symbol<Strips<R>> } = 1 #{!} ({ Symbol<Strips<R>> }),
                    Strip,
                    Part,
                    Frequency
                ]>(tile),
            );
        }
        let first = Sides::<Sinusoid>::new().forward((&mut *device, &sinusoids));
        let mut sums: HbmTensor<f32, X, m![R, { Symbol<Stacked<Hidden>> }]> = HbmTensor::new();
        self.first
            .weight
            .forward((&mut *device, (&first, &mut sums)));
        let hidden = self.first.forward((&mut *device, (&sums,)));
        let hidden = Silu.forward((&mut *device, &hidden));
        let second = Sides::<Hidden>::new().forward((&mut *device, &hidden));
        let mut sums: HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]> = HbmTensor::new();
        self.second
            .weight
            .forward((&mut *device, (&second, &mut sums)));
        self.second.forward((device, (&sums,)))
    }
}

/// The embeddings of a step's timesteps out of a time embedding over a whole strip of rows: the
/// strip's leading `Timestep` rows.
#[derive(DeviceSend)]
pub struct Trim;

impl<R: AxisName, W: AxisName, Chips: M> Module<(&mut Device, &HbmTensor<f32, Chips, m![R, W]>)>
    for Trim
{
    type Output = HbmTensor<f32, Chips, m![Timestep, W]>;

    fn forward(
        &self,
        (device, embedded): (&mut Device, &HbmTensor<f32, Chips, m![R, W]>),
    ) -> Self::Output {
        let mut time = HbmTensor::new();
        let rows = embedded
            .view()
            .tile::<m![R], { Timestep::SIZE }, m![R = (Timestep) # R, W]>(0);
        // SAFETY: the tile is the leading rows in order; only the row axis is renamed.
        let rows = unsafe { rows.reshape::<Chips, m![Timestep # R, W]>() };
        rows.to_hbm_view(&mut device.tdma, time.view_mut());
        time
    }
}

#[cfg(test)]
mod tests {
    struct Fixture;
    use super::*;
    use crate::ops::normalization::tests::random;
    use furiosa_opt_std::runtime::launch;

    axes![Text = 64, Audio = 64, Video = 128, Hidden = 5376];

    type Sequence = Packed<Text, Audio, Video>;

    const EPSILON: usize = 1e-5_f32.to_bits() as usize;

    #[device(chip = 1)]
    fn modulated<T: AxisName, A: AxisName, V: AxisName>(
        device: &mut Device,
        norm: &AdaptiveRms<Hidden, EPSILON>,
        input: &Rows<Packed<T, A, V>, Hidden>,
        modulation: &(Table<Hidden>, Table<Hidden>),
    ) -> Rows<Packed<T, A, V>, Hidden> {
        norm.forward((device, (input, modulation)))
    }

    /// Every row is normalized and modulated by its span's table row, as the FP64 definition
    /// rounded once; rows span four magnitudes, over spans of strips and of whole blocks.
    #[tokio::test(flavor = "current_thread")]
    async fn spans_take_their_modality_row() {
        Fixture::span_rows_modulate::<Text, Audio, ShortVideo>().await;
        Fixture::span_rows_modulate::<TextEnvelope, AudioEnvelope, LongVideo>().await;
        Fixture::span_rows_modulate::<TextEnvelope, AudioEnvelope, MidVideo>().await;
    }

    impl Fixture {
        async fn span_rows_modulate<T: AxisName, A: AxisName, V: AxisName>() {
            let _lease = crate::tests::DEVICE.lock().await;
            type Sequence<T, A, V> = Packed<T, A, V>;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let width = Hidden::SIZE;
            let inputs: Vec<bf16> =
                random(Sequence::<T, A, V>::SIZE * width, 0x2545_f491_4f6c_dd1d)
                    .iter()
                    .enumerate()
                    .map(|(index, v)| {
                        bf16::from_f32(v.to_f32() * (1 << (index / width % 4 * 3)) as f32)
                    })
                    .collect();
            let weight = random(width, 0x9e37_79b9_7f4a_7c15);
            let shift = random(Group::SIZE * width, 0x0123_4567_89ab_cdef);
            let scale = random(Group::SIZE * width, 0x7f4a_7c15_9e37_79b9);
            let upload = async |device: &mut Device, values: &[bf16]| {
                HostTensor::<bf16, m![Group, Hidden]>::from_vec(values.to_vec())
                    .to_hbm::<m![1], m![Group, Hidden]>(&mut device.pdma)
                    .await
                    .unwrap()
            };
            let modulation = (
                upload(&mut device, &shift).await,
                upload(&mut device, &scale).await,
            );
            let norm = AdaptiveRms::new(Weight::new(
                HostTensor::<bf16, m![Hidden]>::from_vec(weight.clone())
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ));
            let input = HostTensor::<bf16, m![{ Symbol<Sequence<T, A, V>> }, Hidden]>::from_vec(
                inputs.clone(),
            )
            .to_hbm::<m![1], m![{ Symbol<Sequence<T, A, V>> }, Hidden]>(&mut device.pdma)
            .await
            .unwrap();
            let output = launch(modulated, (&mut device, &norm, &input, &modulation))
                .await
                .unwrap()
                .to_host::<m![{ Symbol<Sequence<T, A, V>> }, Hidden]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (row, values) in inputs.chunks_exact(width).enumerate() {
                let mean = values
                    .iter()
                    .map(|v| f64::from(v.to_f32()).powi(2))
                    .sum::<f64>()
                    / width as f64;
                let inverse = 1.0 / (mean + 1e-5).sqrt();
                let table = match row {
                    row if row < T::SIZE => 1,
                    row if row < T::SIZE + A::SIZE => 5,
                    _ => 0,
                } * width;
                for (column, value) in values.iter().enumerate() {
                    let normalized = f64::from(value.to_f32()) * inverse;
                    let gain = f64::from(weight[column].to_f32())
                        * (1.0 + f64::from(scale[table + column].to_f32()));
                    let expected = normalized * gain + f64::from(shift[table + column].to_f32());
                    let got = f64::from(output[row * width + column].to_f32());
                    assert!(
                        (got - expected).abs() <= expected.abs() / 128.0 + 1e-3,
                        "row {row} column {column}: {got} vs {expected}"
                    );
                }
            }
        }
    }
    #[device(chip = 1)]
    fn gated_rows<T: AxisName, A: AxisName, V: AxisName>(
        device: &mut Device,
        input: &Rows<Packed<T, A, V>, Hidden>,
        branch: &Rows<Packed<T, A, V>, Hidden>,
        gate: &Table<Hidden>,
    ) -> Rows<Packed<T, A, V>, Hidden> {
        super::super::GatedAdd.forward((device, (input, branch, gate)))
    }

    /// Every row adds its branch times its span's gate row, as the FP64 sum rounded once, over
    /// spans of strips only and of strips and whole blocks.
    #[tokio::test(flavor = "current_thread")]
    async fn spans_gate_their_branch() {
        Fixture::span_rows_gate::<Text, Audio, ShortVideo>().await;
        Fixture::span_rows_gate::<TextEnvelope, AudioEnvelope, LongVideo>().await;
        Fixture::span_rows_gate::<TextEnvelope, AudioEnvelope, MidVideo>().await;
    }

    impl Fixture {
        async fn span_rows_gate<T: AxisName, A: AxisName, V: AxisName>() {
            let _lease = crate::tests::DEVICE.lock().await;
            type Sequence<T, A, V> = Packed<T, A, V>;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let width = Hidden::SIZE;
            let inputs = random(Sequence::<T, A, V>::SIZE * width, 0x2545_f491_4f6c_dd1d);
            let branches = random(Sequence::<T, A, V>::SIZE * width, 0x0123_4567_89ab_cdef);
            let gates = random(Group::SIZE * width, 0x7f4a_7c15_9e37_79b9);
            let rows = async |device: &mut Device, values: &[bf16]| {
                HostTensor::<bf16, m![{ Symbol<Sequence<T, A, V>> }, Hidden]>::from_vec(
                    values.to_vec(),
                )
                .to_hbm::<m![1], m![{ Symbol<Sequence<T, A, V>> }, Hidden]>(&mut device.pdma)
                .await
                .unwrap()
            };
            let (input, branch) = (
                rows(&mut device, &inputs).await,
                rows(&mut device, &branches).await,
            );
            let gate = HostTensor::<bf16, m![Group, Hidden]>::from_vec(gates.clone())
                .to_hbm::<m![1], m![Group, Hidden]>(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(gated_rows, (&mut device, &input, &branch, &gate))
                .await
                .unwrap()
                .to_host::<m![{ Symbol<Sequence<T, A, V>> }, Hidden]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (index, got) in output.iter().enumerate() {
                let (row, column) = (index / width, index % width);
                let table = match row {
                    row if row < T::SIZE => 1,
                    row if row < T::SIZE + A::SIZE => 5,
                    _ => 0,
                };
                let expected = f64::from(inputs[index].to_f32())
                    + f64::from(gates[table * width + column].to_f32())
                        * f64::from(branches[index].to_f32());
                let got = f64::from(got.to_f32());
                assert!(
                    (got - expected).abs() <= expected.abs() / 256.0 + 1e-4,
                    "row {row} column {column}: {got} vs {expected}"
                );
            }
        }
    }
    type TextRows = Span<50, 64>;
    type AudioRows = Span<40, 64>;
    type VideoRows = Span<100, 128>;
    type Joined = Packed<TextRows, AudioRows, VideoRows>;
    axes![Narrow = 256];

    #[device(chip = 1)]
    fn joined(
        device: &mut Device,
        media: &(Rows<VideoRows, Narrow>, Rows<AudioRows, Narrow>),
        text: &Rows<TextRows, Narrow>,
    ) -> (
        Rows<Joined, Narrow>,
        (Rows<VideoRows, Narrow>, Rows<AudioRows, Narrow>),
    ) {
        let packed = Join.forward((&mut *device, (media, text)));
        let split = Split.forward((device, (&packed, ())));
        (packed, split)
    }

    /// Joined rows hold the text, audio and video spans whole, in that order, and the split gives
    /// the video and audio spans back unchanged.
    #[tokio::test(flavor = "current_thread")]
    async fn joins_and_splits_the_spans() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let width = Narrow::SIZE;
        let text = random(64 * width, 0x2545_f491_4f6c_dd1d);
        let audio = random(64 * width, 0x9e37_79b9_7f4a_7c15);
        let video = random(128 * width, 0x0123_4567_89ab_cdef);
        let media = (
            HostTensor::<bf16, m![VideoRows, Narrow]>::from_vec(video.clone())
                .to_hbm::<m![1], m![VideoRows, Narrow]>(&mut device.pdma)
                .await
                .unwrap(),
            HostTensor::<bf16, m![AudioRows, Narrow]>::from_vec(audio.clone())
                .to_hbm::<m![1], m![AudioRows, Narrow]>(&mut device.pdma)
                .await
                .unwrap(),
        );
        let words = HostTensor::<bf16, m![TextRows, Narrow]>::from_vec(text.clone())
            .to_hbm::<m![1], m![TextRows, Narrow]>(&mut device.pdma)
            .await
            .unwrap();
        let (packed, (video_rows, audio_rows)) =
            launch(joined, (&mut device, &media, &words)).await.unwrap();
        let packed = packed
            .to_host::<m![Joined, Narrow]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        assert_eq!(packed, [text, audio.clone(), video.clone()].concat());
        let video_rows = video_rows
            .to_host::<m![VideoRows, Narrow]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let audio_rows = audio_rows
            .to_host::<m![AudioRows, Narrow]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        assert_eq!((video_rows, audio_rows), (video, audio));
    }

    axes![Time = 2688];

    #[device(chip = 1)]
    fn projected(
        device: &mut Device,
        projection: &Projection<Time, Hidden>,
        time: &HbmTensor<f32, m![1], m![Timestep, Time]>,
    ) -> (Gates<Hidden>, Gates<Hidden>) {
        projection.forward((device, time))
    }

    /// Each table row is `bf16(bf16(silu(t)) . w + b)` of its timestep and of its modality's
    /// chunk, the sum FP64 and rounded once, within FP32 accumulation error.
    #[tokio::test(flavor = "current_thread")]
    async fn tables_take_their_modality_chunk() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let (width, inputs) = (Hidden::SIZE, Time::SIZE);
        let outputs = Modulations::<Hidden>::SIZE;
        let time: Vec<f32> = random(Timestep::SIZE * inputs, 0x2545_f491_4f6c_dd1d)
            .iter()
            .map(|v| v.to_f32() * 4.0)
            .collect();
        let weight = random(outputs * inputs, 0x9e37_79b9_7f4a_7c15);
        let bias = random(outputs, 0x0123_4567_89ab_cdef);
        let projection = Projection::new(
            HostTensor::<bf16, m![{ Outputs<Hidden> }, Time]>::from_vec(weight.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            HostTensor::<f32, m![{ Outputs<Hidden> }, 1 # 8]>::from_vec(
                bias.iter()
                    .flat_map(|b| [b.to_f32(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    .collect::<Vec<f32>>(),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        );
        let input = HostTensor::<f32, m![Timestep, Time]>::from_vec(time.clone())
            .to_hbm::<m![1], m![Timestep, Time]>(&mut device.pdma)
            .await
            .unwrap();
        let (attention, feedforward) = launch(projected, (&mut device, &projection, &input))
            .await
            .unwrap();
        let ((shift, scale), gate) = attention;
        let ((shift2, scale2), gate2) = feedforward;
        let mut tables = Vec::new();
        for table in [shift, scale, gate, shift2, scale2, gate2] {
            tables.push(
                table
                    .to_host::<m![Group, Hidden]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec(),
            );
        }
        let activated: Vec<f64> = time
            .iter()
            .map(|t| {
                let t = f64::from(*t);
                f64::from(bf16::from_f32((t / (1.0 + (-t).exp())) as f32).to_f32())
            })
            .collect();
        for (chunk, table) in tables.iter().enumerate() {
            for group in 0..Group::SIZE {
                let (timestep, modality) = (group / 3, group % 3);
                let a = &activated[timestep * inputs..][..inputs];
                for j in 0..width {
                    let output = modality * 6 * width + chunk * width + j;
                    let expected: f64 = a
                        .iter()
                        .zip(&weight[output * inputs..][..inputs])
                        .map(|(a, w)| a * f64::from(w.to_f32()))
                        .sum::<f64>()
                        + f64::from(bias[output].to_f32());
                    let got = f64::from(table[group * width + j].to_f32());
                    assert!(
                        (got - expected).abs() <= expected.abs() / 256.0 + 0.01,
                        "chunk {chunk} row {group} feature {j}: {got} vs {expected}"
                    );
                }
            }
        }
    }

    #[device(chip = 1)]
    fn shifted(
        device: &mut Device,
        modulation: &HeadModulation<Time, Hidden, 1>,
        time: &HbmTensor<f32, m![1], m![Timestep, Time]>,
    ) -> Shift<Hidden> {
        modulation.forward((device, time))
    }

    /// The head modulation is `bf16(bf16(silu(t)) . w + b)` of the audio timestep, shift then
    /// scale; the video timestep's row differs, so taking it fails.
    #[tokio::test(flavor = "current_thread")]
    async fn head_takes_its_timestep_row() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let (width, inputs) = (Hidden::SIZE, Time::SIZE);
        let time: Vec<f32> = random(Timestep::SIZE * inputs, 0x2545_f491_4f6c_dd1d)
            .iter()
            .map(|v| v.to_f32() * 4.0)
            .collect();
        let weight = random(2 * width * inputs, 0x9e37_79b9_7f4a_7c15);
        let bias = random(2 * width, 0x0123_4567_89ab_cdef);
        let modulation = HeadModulation::new(
            HostTensor::<bf16, m![{ Symbol<ShiftScale<Hidden>> }, Time]>::from_vec(weight.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            HostTensor::<f32, m![{ Symbol<ShiftScale<Hidden>> }, 1 # 8]>::from_vec(
                bias.iter()
                    .flat_map(|b| [b.to_f32(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    .collect::<Vec<f32>>(),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        );
        let input = HostTensor::<f32, m![Timestep, Time]>::from_vec(time.clone())
            .to_hbm::<m![1], m![Timestep, Time]>(&mut device.pdma)
            .await
            .unwrap();
        let (shift, scale) = launch(shifted, (&mut device, &modulation, &input))
            .await
            .unwrap();
        let activated: Vec<f64> = time[inputs..]
            .iter()
            .map(|t| {
                let t = f64::from(*t);
                f64::from(bf16::from_f32((t / (1.0 + (-t).exp())) as f32).to_f32())
            })
            .collect();
        for (part, vector) in [shift, scale].iter().enumerate() {
            let got = vector
                .to_host::<m![Hidden]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for j in 0..width {
                let output = part * width + j;
                let expected: f64 = activated
                    .iter()
                    .zip(&weight[output * inputs..][..inputs])
                    .map(|(a, w)| a * f64::from(w.to_f32()))
                    .sum::<f64>()
                    + f64::from(bias[output].to_f32());
                let got = f64::from(got[j].to_f32());
                assert!(
                    (got - expected).abs() <= expected.abs() / 256.0 + 0.01,
                    "part {part} feature {j}: {got} vs {expected}"
                );
            }
        }
    }

    axes![Encoded = 5120, Envelope = 128];

    #[device(chip = 1)]
    fn biased(
        device: &mut Device,
        projection: &Biased<Encoded, Hidden>,
        input: &Rows<Envelope, Encoded>,
    ) -> Rows<Envelope, Hidden> {
        projection.forward((device, input))
    }

    /// Rows projected with an offset land `bf16(x . w + b)`, the FP64 sum rounded once, within FP32
    /// accumulation error; the rows at both ends of both tiles are checked whole.
    #[tokio::test(flavor = "current_thread")]
    async fn offsets_join_the_sum_before_rounding() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let (inputs, outputs) = (Encoded::SIZE, Hidden::SIZE);
        let rows = random(Envelope::SIZE * inputs, 0x2545_f491_4f6c_dd1d);
        let weight = random(outputs * inputs, 0x9e37_79b9_7f4a_7c15);
        let bias: Vec<f32> = (0..outputs)
            .map(|o| (o % 97) as f32 * 0.25 - 12.0)
            .collect();
        let projection = Biased::new(
            Weight::new(
                HostTensor::<bf16, m![Hidden, Encoded]>::from_vec(weight.clone())
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ),
            HostTensor::<f32, m![Hidden]>::from_vec(bias.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        );
        let input = HostTensor::<bf16, m![Envelope, Encoded]>::from_vec(rows.clone())
            .to_hbm::<m![1], m![Envelope, Encoded]>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(biased, (&mut device, &projection, &input))
            .await
            .unwrap()
            .to_host::<m![Envelope, Hidden]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for row in [0, 63, 64, 127] {
            for column in 0..outputs {
                let expected: f64 = rows[row * inputs..][..inputs]
                    .iter()
                    .zip(&weight[column * inputs..][..inputs])
                    .map(|(x, w)| f64::from(x.to_f32()) * f64::from(w.to_f32()))
                    .sum::<f64>()
                    + f64::from(bias[column]);
                let got = f64::from(output[row * outputs + column].to_f32());
                assert!(
                    (got - expected).abs() <= expected.abs() / 256.0 + 0.01,
                    "row {row} column {column}: {got} vs {expected}"
                );
            }
        }
    }

    axes![Velocity = 96];

    #[device(chip = 1)]
    fn precise(
        device: &mut Device,
        projection: &Precise<Hidden, Velocity>,
        input: &Rows<Sequence, Hidden>,
    ) -> HbmTensor<f32, m![1], m![Sequence, Velocity]> {
        projection.forward((device, input))
    }

    /// BF16 rows times FP32 coefficients plus FP32 offsets land the FP64 product within 2^-17 of
    /// its summed term magnitudes, though the contraction itself multiplies BF16.
    #[tokio::test(flavor = "current_thread")]
    async fn projects_with_fp32_coefficients() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let (inputs, outputs) = (Hidden::SIZE, Velocity::SIZE);
        let rows = random(Sequence::SIZE * inputs, 0x2545_f491_4f6c_dd1d);
        let weight: Vec<f32> = (0..outputs * inputs)
            .map(|index| ((index * 2_654_435_761) % 1_000_003) as f32 / 1_000_003.0 - 0.5)
            .collect();
        let bias: Vec<f32> = (0..outputs).map(|o| o as f32 * 0.137 - 3.0).collect();
        let high: Vec<bf16> = weight.iter().map(|w| bf16::from_f32(*w)).collect();
        let low: Vec<bf16> = weight
            .iter()
            .zip(&high)
            .map(|(w, h)| bf16::from_f32(w - h.to_f32()))
            .collect();
        let projection = Precise::new(
            Weight::new(
                HostTensor::<bf16, m![{ Symbol<Stacked<Velocity>> }, Hidden]>::from_vec(
                    high.into_iter().chain(low).collect::<Vec<bf16>>(),
                )
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            ),
            HostTensor::<f32, m![Velocity]>::from_vec(bias.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        );
        let input = HostTensor::<bf16, m![Sequence, Hidden]>::from_vec(rows.clone())
            .to_hbm::<m![1], m![Sequence, Hidden]>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(precise, (&mut device, &projection, &input))
            .await
            .unwrap()
            .to_host::<m![Sequence, Velocity]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (index, got) in output.iter().enumerate() {
            let (row, column) = (index / outputs, index % outputs);
            let terms = rows[row * inputs..][..inputs]
                .iter()
                .zip(&weight[column * inputs..][..inputs])
                .map(|(x, w)| f64::from(x.to_f32()) * f64::from(*w));
            let (sum, magnitude) = terms.fold((0.0, 0.0), |(s, m), t| (s + t, m + t.abs()));
            let expected = sum + f64::from(bias[column]);
            assert!(
                (f64::from(*got) - expected).abs() <= magnitude / f64::from(1 << 17) + 1e-6,
                "row {row} column {column}: {got} vs {expected} (terms {magnitude})"
            );
        }
    }

    #[device(chip = 1)]
    fn final_norm(
        device: &mut Device,
        norm: &FinalNorm<Hidden, EPSILON>,
        input: &Rows<Sequence, Hidden>,
        modulation: &Shift<Hidden>,
    ) -> Rows<Sequence, Hidden> {
        norm.forward((device, (input, modulation)))
    }

    /// Each row's RMS normalization rounds to BF16 before its timestep's modulation, which rounds
    /// again: both roundings as the host computes them, within one BF16 step of a near tie.
    #[tokio::test(flavor = "current_thread")]
    async fn rounds_the_norm_before_the_modulation() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let width = Hidden::SIZE;
        let inputs: Vec<bf16> = random(Sequence::SIZE * width, 0x2545_f491_4f6c_dd1d)
            .iter()
            .enumerate()
            .map(|(index, v)| bf16::from_f32(v.to_f32() * (1 << (index / width % 4 * 3)) as f32))
            .collect();
        let weight = random(width, 0x9e37_79b9_7f4a_7c15);
        let shift = random(width, 0x0123_4567_89ab_cdef);
        let scale = random(width, 0x7f4a_7c15_9e37_79b9);
        let vector = async |device: &mut Device, values: &[bf16]| {
            HostTensor::<bf16, m![Hidden]>::from_vec(values.to_vec())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap()
        };
        let modulation = (
            vector(&mut device, &shift).await,
            vector(&mut device, &scale).await,
        );
        let norm = FinalNorm::new(Weight::new(vector(&mut device, &weight).await));
        let input = HostTensor::<bf16, m![Sequence, Hidden]>::from_vec(inputs.clone())
            .to_hbm::<m![1], m![Sequence, Hidden]>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(final_norm, (&mut device, &norm, &input, &modulation))
            .await
            .unwrap()
            .to_host::<m![Sequence, Hidden]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let rounded = |x: f64| f64::from(bf16::from_f32(x as f32).to_f32());
        for (row, values) in inputs.chunks_exact(width).enumerate() {
            let mean = values
                .iter()
                .map(|v| f64::from(v.to_f32()).powi(2))
                .sum::<f64>()
                / width as f64;
            let inverse = 1.0 / (mean + 1e-5).sqrt();
            for (column, value) in values.iter().enumerate() {
                let normalized = rounded(
                    f64::from(value.to_f32()) * inverse * f64::from(weight[column].to_f32()),
                );
                let gain = 1.0 + f64::from(scale[column].to_f32());
                let expected = normalized * gain + f64::from(shift[column].to_f32());
                let got = f64::from(output[row * width + column].to_f32());
                // One BF16 step of the rounded norm, scaled by the gain, and the last rounding.
                let bound = (normalized * gain).abs() / 128.0 + expected.abs() / 256.0 + 1e-6;
                assert!(
                    (got - expected).abs() <= bound,
                    "row {row} column {column}: {got} vs {expected}"
                );
            }
        }
    }

    axes![Patch = 96];

    #[device(chip = 1)]
    fn embedded(
        device: &mut Device,
        embedding: &Embedding<Patch, Hidden>,
        input: &HbmTensor<f32, m![1], m![Sequence, Patch]>,
    ) -> Rows<Sequence, Hidden> {
        embedding.forward((device, input))
    }

    /// FP32 latent rows times FP32 coefficients plus FP32 offsets land the FP64 product rounded
    /// once to BF16, within 2^-17 of its summed term magnitudes before the rounding.
    #[tokio::test(flavor = "current_thread")]
    async fn embeds_fp32_rows() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let (inputs, outputs) = (Patch::SIZE, Hidden::SIZE);
        let rows: Vec<f32> = (0..Sequence::SIZE * inputs)
            .map(|index| ((index * 2_246_822_519) % 999_983) as f32 / 333_327.0 - 1.5)
            .collect();
        let weight: Vec<f32> = (0..outputs * inputs)
            .map(|index| ((index * 2_654_435_761) % 1_000_003) as f32 / 1_000_003.0 - 0.5)
            .collect();
        let bias: Vec<f32> = (0..outputs)
            .map(|o| (o % 97) as f32 * 0.041 - 2.0)
            .collect();
        let high: Vec<bf16> = weight.iter().map(|w| bf16::from_f32(*w)).collect();
        let low: Vec<bf16> = weight
            .iter()
            .zip(&high)
            .map(|(w, h)| bf16::from_f32(w - h.to_f32()))
            .collect();
        let zero = bf16::from_f32(0.0);
        let mut stacked = Vec::with_capacity(4 * outputs * inputs);
        for o in 0..outputs {
            stacked.extend_from_slice(&high[o * inputs..][..inputs]);
            stacked.extend_from_slice(&high[o * inputs..][..inputs]);
        }
        for o in 0..outputs {
            stacked.extend_from_slice(&low[o * inputs..][..inputs]);
            stacked.extend(std::iter::repeat_n(zero, inputs));
        }
        let embedding = Embedding::new(
            Weight::new(
                HostTensor::<bf16, m![{ Symbol<Stacked<Hidden>> }, { Symbol<Sides<Patch>> }]>::from_vec(
                    stacked,
                )
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            ),
            HostTensor::<f32, m![Hidden]>::from_vec(bias.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        );
        let input = HostTensor::<f32, m![Sequence, Patch]>::from_vec(rows.clone())
            .to_hbm::<m![1], m![Sequence, Patch]>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(embedded, (&mut device, &embedding, &input))
            .await
            .unwrap()
            .to_host::<m![Sequence, Hidden]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (index, got) in output.iter().enumerate() {
            let (row, column) = (index / outputs, index % outputs);
            let terms = rows[row * inputs..][..inputs]
                .iter()
                .zip(&weight[column * inputs..][..inputs])
                .map(|(x, w)| f64::from(*x) * f64::from(*w));
            let (sum, magnitude) = terms.fold((0.0, 0.0), |(s, m), t| (s + t, m + t.abs()));
            let expected = sum + f64::from(bias[column]);
            let bound = magnitude / f64::from(1 << 17) + expected.abs() / 256.0 + 1e-6;
            assert!(
                (f64::from(got.to_f32()) - expected).abs() <= bound,
                "row {row} column {column}: {} vs {expected}",
                got.to_f32()
            );
        }
    }

    axes![Embedded = 2688, Steps = 64];

    #[device(chip = 1)]
    fn time_embedded(
        device: &mut Device,
        embedding: &TimeEmbedding<Hidden, Embedded>,
        time: &HbmTensor<f32, m![1], m![Steps, 1 # 8]>,
    ) -> HbmTensor<f32, m![1], m![Steps, Embedded]> {
        embedding.forward((device, time))
    }

    /// `[high | high]` then `[low | 0]` of FP32 coefficients by output, for an [`Embedding`].
    impl Fixture {
        fn stacked(weight: &[f32], inputs: usize) -> Vec<bf16> {
            let high: Vec<bf16> = weight.iter().map(|w| bf16::from_f32(*w)).collect();
            let low: Vec<bf16> = weight
                .iter()
                .zip(&high)
                .map(|(w, h)| bf16::from_f32(w - h.to_f32()))
                .collect();
            let outputs = weight.len() / inputs;
            let zero = bf16::from_f32(0.0);
            let mut stacked = Vec::with_capacity(4 * weight.len());
            for o in 0..outputs {
                stacked.extend_from_slice(&high[o * inputs..][..inputs]);
                stacked.extend_from_slice(&high[o * inputs..][..inputs]);
            }
            for o in 0..outputs {
                stacked.extend_from_slice(&low[o * inputs..][..inputs]);
                stacked.extend(std::iter::repeat_n(zero, inputs));
            }
            stacked
        }
    }
    /// The video and audio timesteps' embeddings match the FP64 definition, cosines first, to
    /// within 1e-4 of the embedding's RMS.
    #[tokio::test(flavor = "current_thread")]
    async fn embeds_two_timesteps() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let (sinusoid, hidden, out) = (Sinusoid::SIZE, Hidden::SIZE, Embedded::SIZE);
        let coefficients = |count: usize, seed: usize| -> Vec<f32> {
            (0..count)
                .map(|index| {
                    ((index * 2_654_435_761 + seed) % 1_000_003) as f32 / 4_000_012.0 - 0.125
                })
                .collect()
        };
        let (first, second) = (
            coefficients(hidden * sinusoid, 7),
            coefficients(out * hidden, 11),
        );
        let first_bias: Vec<f32> = (0..hidden).map(|o| (o % 89) as f32 * 0.011 - 0.5).collect();
        let second_bias: Vec<f32> = (0..out).map(|o| (o % 83) as f32 * 0.013 - 0.5).collect();
        let frequencies: Vec<f32> = (0..Frequency::SIZE)
            .map(|j| (-(10000f32.ln()) * j as f32 / Frequency::SIZE as f32).exp())
            .collect();
        let times = [0.0f32, 0.683_421];
        let embedding =
            TimeEmbedding::new(
                HostTensor::<f32, m![Frequency]>::from_vec(frequencies.clone())
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                Embedding::new(
                    Weight::new(
                        HostTensor::<
                            bf16,
                            m![{ Symbol<Stacked<Hidden>> }, { Symbol<Sides<Sinusoid>> }],
                        >::from_vec(Fixture::stacked(&first, sinusoid))
                        .to_hbm(&mut device.pdma)
                        .await
                        .unwrap(),
                    ),
                    HostTensor::<f32, m![Hidden]>::from_vec(first_bias.clone())
                        .to_hbm(&mut device.pdma)
                        .await
                        .unwrap(),
                ),
                Embedding::new(
                    Weight::new(
                        HostTensor::<
                            bf16,
                            m![{ Symbol<Stacked<Embedded>> }, { Symbol<Sides<Hidden>> }],
                        >::from_vec(Fixture::stacked(&second, hidden))
                        .to_hbm(&mut device.pdma)
                        .await
                        .unwrap(),
                    ),
                    HostTensor::<f32, m![Embedded]>::from_vec(second_bias.clone())
                        .to_hbm(&mut device.pdma)
                        .await
                        .unwrap(),
                ),
            );
        let mut steps = vec![0.0f32; Steps::SIZE * 8];
        steps[0] = times[0];
        steps[8] = times[1];
        let time = HostTensor::<f32, m![Steps, 1 # 8]>::from_vec(steps)
            .to_hbm::<m![1], m![Steps, 1 # 8]>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(time_embedded, (&mut device, &embedding, &time))
            .await
            .unwrap()
            .to_host::<m![Steps, Embedded]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (row, t) in times.iter().enumerate() {
            let waves: Vec<f64> = (0..sinusoid)
                .map(|j| {
                    let angle = f64::from(*t) * f64::from(frequencies[j % Frequency::SIZE]);
                    if j < Frequency::SIZE {
                        angle.cos()
                    } else {
                        angle.sin()
                    }
                })
                .collect();
            let hidden_values: Vec<f64> = (0..hidden)
                .map(|o| {
                    let x = waves
                        .iter()
                        .zip(&first[o * sinusoid..][..sinusoid])
                        .map(|(e, w)| e * f64::from(*w))
                        .sum::<f64>()
                        + f64::from(first_bias[o]);
                    x / (1.0 + (-x).exp())
                })
                .collect();
            let expected: Vec<f64> = (0..out)
                .map(|o| {
                    hidden_values
                        .iter()
                        .zip(&second[o * hidden..][..hidden])
                        .map(|(h, w)| h * f64::from(*w))
                        .sum::<f64>()
                        + f64::from(second_bias[o])
                })
                .collect();
            let rms = (expected.iter().map(|e| e * e).sum::<f64>() / out as f64).sqrt();
            let worst = expected
                .iter()
                .enumerate()
                .map(|(o, e)| (f64::from(output[row * out + o]) - e).abs())
                .fold(0.0, f64::max);
            assert!(
                worst <= 1e-4 * rms,
                "timestep {t}: worst {worst} of rms {rms}"
            );
        }
    }

    axes![
        TextEnvelope = 128,
        AudioEnvelope = 448,
        ShortVideo = 384,
        LongVideo = 960,
        MidVideo = 704
    ];
}
