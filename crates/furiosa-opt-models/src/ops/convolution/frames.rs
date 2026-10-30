//! Time-major stereo signals: HBM frames with padding rows and the row blocks slices compute.
//!
//! - A frame holds `LEAD` padding rows, a span of rows and `LEAD` more. The span holds the live
//!   samples, rounded up to whole blocks. A row is one sample's channels, so a shift by whole rows
//!   keeps every transfer 8-byte aligned, which device reads require.
//! - A span splits into equal blocks of a multiple of eight rows, one block a slice; each stereo
//!   channel takes one cluster.
//! - A window is the span of rows that starts at an offset into the frame: the input of one tap.
//! - Only the `LEAD` rows on each side of the live samples are padding; a consumer reads no further
//!   for a live output, and rows of the span past the live samples hold whatever their producer
//!   computed.
//! - A span carries its rows, block and live samples as constants, so its window length is a
//!   constant parameter of the device program rather than an expression over one; a model names
//!   each span it computes, and how its spans relate ([`Doubled`]).
use furiosa_opt_std::prelude::*;

use crate::load::Load;
use crate::tensor::parallel::Chip;
use core::marker::PhantomData;

axes![Stereo = 2, Phase = 2];

/// Padding rows before the live samples, and after them.
pub const LEAD: usize = 32;

/// `ROWS` rows in blocks of `BLOCK`, one block a slice, the first `LIVE` of them samples.
#[derive(Clone, Debug)]
pub struct Span<const ROWS: usize, const BLOCK: usize, const LIVE: usize>;

impl<const ROWS: usize, const BLOCK: usize, const LIVE: usize> AxisName
    for Span<ROWS, BLOCK, LIVE>
{
    const NAME: Ident = Ident::new("Span");
    const SIZE: usize = {
        assert!(BLOCK % 8 == 0 && ROWS % BLOCK == 0 && ROWS / BLOCK <= 256 && LIVE <= ROWS);
        ROWS
    };
}

/// A span the device programs window.
pub trait Window: AxisName {
    /// Rows a slice holds.
    const BLOCK: usize;
    /// Live samples of the span; the rows after them are padding.
    const SAMPLES: usize;

    /// The span starting `offset` rows into each stereo channel's frame.
    fn window<D: MaterializableScalar, C: AxisName, X: M>(
        frames: &Frames<D, C, Self, X>,
        offset: usize,
    ) -> HbmTensorView<'_, D, X, Windowed<C, Self>>;

    /// The span after each stereo channel's leading padding, to write.
    fn interior<D: MaterializableScalar, C: AxisName, X: M>(
        frames: &mut Frames<D, C, Self, X>,
    ) -> HbmTensorViewMut<'_, D, X, Written<C, Self>>;
}

/// A span with a counterpart of twice the rows, block and samples: the up-sampled rows of an
/// activation. The model that computes the span names the counterpart.
pub trait Doubled: Window {
    type Twice: Window;
}

/// Every row of a frame.
#[derive(Clone, Debug)]
pub struct FrameAxis<S>(PhantomData<S>);

impl<S: AxisName> AxisName for FrameAxis<S> {
    const NAME: Ident = Ident::new("Frame");
    const SIZE: usize = LEAD + S::SIZE + LEAD;
}

/// The rows of one slice's block.
#[derive(Clone, Debug)]
pub struct BlockAxis<S>(PhantomData<S>);

impl<S: Window> AxisName for BlockAxis<S> {
    const NAME: Ident = Ident::new("Block");
    const SIZE: usize = S::BLOCK;
}

/// The rows of a padding.
#[derive(Clone, Debug)]
pub struct PadAxis;

impl AxisName for PadAxis {
    const NAME: Ident = Ident::new("Pad");
    const SIZE: usize = LEAD;
}

pub type FrameRows<S> = Symbol<FrameAxis<S>>;
pub type Block<S> = Symbol<BlockAxis<S>>;

/// A stereo signal in HBM, `LEAD` padding rows on each side of its span.
pub type Frames<D, C, S, X = Chip> = HbmTensor<D, X, m![Stereo, { FrameRows<S> }, C]>;

/// A frame's window: the span's rows inside the frame's extent.
pub type Windowed<C, S> = m![Stereo, { Symbol<S> } # ({ FrameRows<S> }), C];

/// The span of a frame to write.
pub type Written<C, S> = m![Stereo, { Symbol<S> } #{!} ({ FrameRows<S> }), C];

/// Slices of a cluster: one block of a span each.
pub type Blocks<S> = m![{ Symbol<S> } / ({ Block<S> }) # 256];

/// A row of a slice's block.
pub type Row<S> = m![{ Symbol<S> } % ({ Block<S> })];

/// Eight-row groups of a slice's block, named apart from the span's own digits so a banded
/// convolution's contraction reduces over the window rows alone.
#[derive(Clone, Debug)]
pub struct Groups<S>(PhantomData<S>);

impl<S: Window> AxisName for Groups<S> {
    const NAME: Ident = Ident::new("Groups");
    const SIZE: usize = S::BLOCK / 8;
}

/// A span of rows on the slices, one block a slice and one stereo channel a cluster.
pub type Blocked<D, C, S, X = Chip> = DmTensor<D, X, m![Stereo], Blocks<S>, m![{ Row<S> }, C]>;

/// Up-sampled rows of a block: two phases for each input row.
pub type Upsampled<D, C, S, X = Chip> =
    DmTensor<D, X, m![Stereo], Blocks<S>, m![{ Row<S> }, Phase, C]>;

/// The span of rows starting `offset` rows into the frame, written one block a slice.
impl<S: Window> FrameAxis<S> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<S: Window, D: MaterializableScalar, C: AxisName, X: M> crate::Op<(&Frames<D, C, S, X>, usize)>
    for FrameAxis<S>
{
    type Output = Blocked<D, C, S, X>;

    fn forward(
        &self,
        device: &mut Device,
        (frames, offset): (&Frames<D, C, S, X>, usize),
    ) -> Self::Output {
        S::window(frames, offset).to_dm(&mut device.tdma)
    }
}

impl<S: Window, D: MaterializableScalar, C: AxisName, X: M>
    crate::Op<(&Blocked<D, C, S, X>, &mut Frames<D, C, S, X>)> for FrameAxis<S>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (rows, frames): (&Blocked<D, C, S, X>, &mut Frames<D, C, S, X>),
    ) {
        rows.view()
            .to_hbm_view(&mut device.tdma, S::interior(frames));
    }
}

/// `LEAD` padding rows of each stereo channel, as one value for every row.
type Fill<D, C, X> = DmTensor<D, X, m![1 # 2], m![1 # 256], m![Stereo, PadAxis, C]>;

/// A sample type with a zero to pad with.
pub(crate) trait Sample: MaterializableScalar {
    const ZERO: Self;
}

impl Sample for f32 {
    const ZERO: Self = 0.0;
}

impl Sample for bf16 {
    const ZERO: Self = bf16::from_f32(0.0);
}

/// Zeroes the `LEAD` rows on each side of the `samples` live samples.
impl<S: Window, D: Sample, C: AxisName, X: M> crate::Op<(&mut Frames<D, C, S, X>, usize)> for Zero {
    type Output = ();

    fn forward(&self, device: &mut Device, (frames, samples): (&mut Frames<D, C, S, X>, usize)) {
        let mut zeros: Fill<D, C, X> = DmTensor::new();
        zeros.view_mut().memset(D::ZERO, &mut device.sub);
        zeros
            .view()
            .to_hbm_view(&mut device.tdma, FrameAxis::<S>::padding(frames, 0));
        zeros.view().to_hbm_view(
            &mut device.tdma,
            FrameAxis::<S>::padding(frames, LEAD + samples),
        );
    }
}

impl<S: Window, D: MaterializableScalar, C: AxisName, X: M>
    crate::Op<(&mut Frames<D, C, S, X>, usize)> for Edge
{
    type Output = ();

    fn forward(&self, device: &mut Device, (frames, samples): (&mut Frames<D, C, S, X>, usize)) {
        let first: DmTensor<D, X, m![1 # 2], m![1 # 256], m![Stereo, C]> =
            FrameAxis::<S>::edge(frames, LEAD).to_dm(&mut device.tdma);
        let last: DmTensor<D, X, m![1 # 2], m![1 # 256], m![Stereo, C]> =
            FrameAxis::<S>::edge(frames, LEAD + samples - 1).to_dm(&mut device.tdma);
        let mut head: Fill<D, C, X> = DmTensor::new();
        first.view().to_dm_view(&mut device.tdma, head.view_mut());
        let mut tail: Fill<D, C, X> = DmTensor::new();
        last.view().to_dm_view(&mut device.tdma, tail.view_mut());
        head.view()
            .to_hbm_view(&mut device.tdma, FrameAxis::<S>::padding(frames, 0));
        tail.view().to_hbm_view(
            &mut device.tdma,
            FrameAxis::<S>::padding(frames, LEAD + samples),
        );
    }
}

/// What the padding rows of a frame hold, for the consumer of the frame.
pub(crate) trait Pads {
    fn fill<D: Sample, C: AxisName, S: Window, X: M>(
        device: &mut Device,
        frames: &mut Frames<D, C, S, X>,
    );
}

/// Zeros, for a convolution.
#[derive(Clone, Copy, Debug, Default, DeviceSend, Load)]
pub struct Zero;

impl Pads for Zero {
    fn fill<D: Sample, C: AxisName, S: Window, X: M>(
        device: &mut Device,
        frames: &mut Frames<D, C, S, X>,
    ) {
        crate::Op::forward(&Zero, device, (frames, S::SAMPLES));
    }
}

/// The edge sample repeated, for a resampling filter.
#[derive(Clone, Copy, Debug, Default, DeviceSend, Load)]
pub struct Edge;

impl Pads for Edge {
    fn fill<D: Sample, C: AxisName, S: Window, X: M>(
        device: &mut Device,
        frames: &mut Frames<D, C, S, X>,
    ) {
        crate::Op::forward(&Edge, device, (frames, S::SAMPLES));
    }
}

/// Row `row` of each stereo channel.
impl<S: Window> FrameAxis<S> {
    fn edge<D: MaterializableScalar, C: AxisName, X: M>(
        frames: &Frames<D, C, S, X>,
        row: usize,
    ) -> HbmTensorView<'_, D, X, m![Stereo, 1 # ({ FrameRows<S> }), C]> {
        frames
            .view()
            .tile::<m![{ FrameRows<S> }], 1, m![Stereo, 1 # ({ FrameRows<S> }), C]>(row)
    }

    /// The `LEAD` rows of each stereo channel from row `start` on.
    fn padding<D: MaterializableScalar, C: AxisName, X: M>(
        frames: &mut Frames<D, C, S, X>,
        start: usize,
    ) -> HbmTensorViewMut<'_, D, X, m![Stereo, PadAxis #{!} ({ FrameRows<S> }), C]> {
        let rows = frames.view_mut().tile::<m![{ FrameRows<S> }], LEAD, m![
            Stereo,
            { FrameRows<S> } = { LEAD } #{!} ({ FrameRows<S> }),
            C
        ]>(start);
        // SAFETY: the window's live rows are the `LEAD` rows from `start`; only the axis is
        // renamed.
        unsafe { rows.reshape() }
    }
}

impl<const ROWS: usize, const BLOCK: usize, const LIVE: usize> Window for Span<ROWS, BLOCK, LIVE> {
    const BLOCK: usize = BLOCK;
    const SAMPLES: usize = LIVE;

    fn window<D: MaterializableScalar, C: AxisName, X: M>(
        frames: &Frames<D, C, Self, X>,
        offset: usize,
    ) -> HbmTensorView<'_, D, X, Windowed<C, Self>> {
        let window = frames.view().tile::<m![{ FrameRows<Self> }], ROWS, m![
            Stereo,
            { FrameRows<Self> } = { ROWS } # ({ FrameRows<Self> }),
            C
        ]>(offset);
        // SAFETY: the window's live rows are the span from `offset`; only the axis is renamed.
        unsafe { window.reshape() }
    }

    fn interior<D: MaterializableScalar, C: AxisName, X: M>(
        frames: &mut Frames<D, C, Self, X>,
    ) -> HbmTensorViewMut<'_, D, X, Written<C, Self>> {
        let window = frames.view_mut().tile::<m![{ FrameRows<Self> }], ROWS, m![
            Stereo,
            { FrameRows<Self> } = { ROWS } #{!} ({ FrameRows<Self> }),
            C
        ]>(LEAD);
        // SAFETY: the window's live rows are the span after the leading padding.
        unsafe { window.reshape() }
    }
}
