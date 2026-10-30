//! Convolution, torch's `conv1d` and `conv2d`: one leaf, [`Convolution`], whose coefficients come
//! in the mapping its strategy contracts with, over time-major frames, or over channel-first
//! signals and images, which no kernel implements yet.
//!
//! - The strategy is how the contraction maps onto a slice's lanes, chosen from the groups, the
//!   output channels and the taps (`specialize::convolution::Strategy::of`): dense, banded or
//!   depthwise. The loader builds the coefficients in the strategy's mapping.
//! - Dense: a cluster's slices split the work three ways, groups of rows, blocks of output channels
//!   and parts of the input channels. A slice keeps its block's weights for its part, every tap,
//!   in its tensor register file, eight output channels a lane.
//! - Each tap's input is one window of the frame, written as that tap's rows of the column tensor;
//!   the write repeats a part's rows on every output block's slices.
//! - Input channels stream up to sixteen a packet and reduce in the packet and over taps in time;
//!   the parts of the input channels then reduce across slices. Products accumulate in FP32 and
//!   the bias is added on the way out.
//! - Banded and depthwise: each lane computes one output sample. The tensor register file holds
//!   each lane's band of the coefficients (the Toeplitz band: lane `l` weighs window row `q` by tap
//!   `q - STRIDE l`), and two windows of the input one block apart cover an output block's reach.
//!   Banded reduces the input channels into one output channel; depthwise filters each channel
//!   alone with one filter, the channels passing through the packet.
use crate::tensor::image;
use core::borrow::Borrow;
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use crate::Op;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::linear;
use crate::specialize::convolution::{Group, Narrow, Outputs, Slabs, Strategy};
use crate::tensor::form::Weight;
use crate::tensor::image::{Grid, Image};
use crate::tensor::{axis::Single, parallel::Chip};
use frames::{Blocked, Doubled, FrameRows, Frames, Groups, LEAD, Pads, Stereo, Window};

pub mod activation;
pub mod frames;
pub mod input;
pub mod residual;
pub mod transpose;

axes![Term = 2, Lag = 8, Lane = 8, Reach = 16, Split = 2];

/// torch's convolution `y[o, t] = b[o] + sum_k sum_c w[o, c, k] x[c, STRIDE t + DILATION k - pad]`,
/// `W` in its strategy's mapping; returned padding rows hold what `P` reads.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Convolution<W, B = (), const STRIDE: usize = 1, const DILATION: usize = 1, P = ()> {
    pub weight: W,
    pub bias: B,
    pub pads: PhantomData<P>,
}

/// Dense coefficients over frames, BF16 in output, tap, input order, held as `T`, the HBM tensor
/// or a borrow of it.
pub type Coefficients<
    In,
    Out,
    Tap,
    X = Chip,
    T = HbmTensor<bf16, X, m![{ Symbol<Produced<Out>> }, Tap, In]>,
> = Weight<m![{ Symbol<Produced<Out>> }, Tap, In], X, T>;

/// An FP32 bias per output channel of dense coefficients over frames.
pub type Bias<Out, X = Chip> = HbmTensor<f32, X, m![{ Symbol<Produced<Out>> }]>;

/// The same length, taps `DILATION` samples apart and symmetric padding, in FP32; `input` holds
/// zeros in its padding rows.
impl<'x, D, In, Out, Tap, const DILATION: usize, P, S, X, T, B> Op<&'x Frames<D, In, S, X>>
    for Convolution<Coefficients<In, Out, Tap, X, T>, B, 1, DILATION, P>
where
    D: MaterializableScalar + FetchCast<bf16>,
    In: AxisName,
    Out: AxisName,
    Tap: AxisName,
    P: Pads,
    S: Window,
    X: M,
    T: Borrow<HbmTensor<bf16, X, m![{ Symbol<Produced<Out>> }, Tap, In]>>,
    B: Borrow<Bias<Out, X>>,
{
    type Output = Frames<f32, Out, S, X>;

    fn forward(&self, device: &mut Device, input: &'x Frames<D, In, S, X>) -> Self::Output {
        const {
            assert!(matches!(
                Strategy::of(1, Out::SIZE, Tap::SIZE, 1),
                Strategy::Dense
            ));
            assert!(Tap::SIZE % 2 == 1 && (Tap::SIZE - 1) * DILATION / 2 <= LEAD);
        };
        let output = Dense {
            weight: self.weight.tensor(),
            bias: self.bias.borrow(),
        }
        .forward(&mut *device, (input, DILATION));
        // SAFETY: the produced channels are the output channels.
        let mut output: Frames<f32, Out, S, X> = unsafe { output.reshape() };
        P::fill(device, &mut output);
        output
    }
}

/// The weight-normalized coefficients of `weight_g` and `weight_v`, `[output, input, tap]` as
/// stored, in output, tap, input order on every chip.
impl<In: AxisName, Out: AxisName, Tap: AxisName, X: M + 'static, T, B, const D: usize, P>
    Convolution<Coefficients<In, Out, Tap, X, T>, B, 1, D, P>
{
    fn coefficients(
        source: &Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<HostTensor<bf16, m![{ X }, { m![{ Symbol<Produced<Out>> }, Tap, In] }]>, LoadError>
    {
        let (inputs, taps) = (In::SIZE, Tap::SIZE);
        let normed = normed(source, context, &[Out::SIZE, inputs, taps])?;
        let at = |index: usize| {
            let (out, tap, input) = (
                index / (taps * inputs),
                index / inputs % taps,
                index % inputs,
            );
            bf16::from_f32(normed[(out * inputs + input) * taps + tap] as f32)
        };
        let coefficients: Vec<bf16> = (0..normed.len()).map(at).collect();
        Ok(HostTensor::from_vec(coefficients.repeat(X::SIZE)))
    }
}

/// The weight-normalized coefficients and the bias as stored.
/// By hand, with the update below: one impl per weight form, where a derived impl is generic.
impl<'a, In: AxisName, Out: AxisName, Tap: AxisName, X, T, B, const D: usize, P>
    Load<&'a Checkpoint> for Convolution<Coefficients<In, Out, Tap, X, T>, B, 1, D, P>
where
    X: M + 'static,
    Coefficients<In, Out, Tap, X, T>:
        Load<HostTensor<bf16, m![{ X }, { m![{ Symbol<Produced<Out>> }, Tap, In] }]>>,
    B: Load<&'a Checkpoint>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let weight = children.ask(cx.field("weight", "", |cx| {
                let coefficients = Self::coefficients(from, cx)?;
                Load::load(cx, coefficients)
            }))?;
            let bias = children.ask(cx.field("bias", "bias", |cx| B::load(cx, from)))?;
            let (weight, bias) = children.node(weight.zip(bias))?;
            Ok(Self {
                weight,
                bias,
                pads: PhantomData,
            })
        })
    }
}

/// The coefficients, one rule of an update, and the bias written in place.
impl<'t, 's, In: AxisName, Out: AxisName, Tap: AxisName, X, T, B, const D: usize, P>
    Load<(
        &'t mut Convolution<Coefficients<In, Out, Tap, X, T>, B, 1, D, P>,
        &'s Checkpoint,
    )> for &'t mut Convolution<Coefficients<In, Out, Tap, X, T>, B, 1, D, P>
where
    X: M + 'static,
    for<'u> &'u mut Coefficients<In, Out, Tap, X, T>: Load<(
        &'u mut Coefficients<In, Out, Tap, X, T>,
        HostTensor<bf16, m![{ X }, { m![{ Symbol<Produced<Out>> }, Tap, In] }]>,
    )>,
    for<'u> &'u mut B: Load<(&'u mut B, &'s Checkpoint)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut Convolution<Coefficients<In, Out, Tap, X, T>, B, 1, D, P>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("weight", "", |cx| {
                cx.rule(
                    |cx| {
                        Convolution::<Coefficients<In, Out, Tap, X, T>, B, 1, D, P>::coefficients(
                            from, cx,
                        )
                    },
                    |cx, host| {
                        <&mut Coefficients<In, Out, Tap, X, T> as Load<_>>::load(
                            cx,
                            (&mut into.weight, host),
                        )
                        .map(drop)
                    },
                )
            })?;
            cx.field("bias", "bias", |cx| {
                <&mut B as Load<_>>::load(cx, (&mut into.bias, from))
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

/// The output channels of a convolution, named apart from its input channels.
#[derive(Clone, Debug)]
pub struct Produced<Out>(PhantomData<Out>);

impl<Out: AxisName> AxisName for Produced<Out> {
    const NAME: Ident = Ident::new("Produced");
    const SIZE: usize = Out::SIZE;
}

/// Depthwise coefficients by two: each of eight output lanes' weights of the 32 window-pair rows,
/// from 5 rows before the block's first input row, as a BF16 sum of two (`Split`).
pub type DownBand<X = Chip> = HbmTensor<bf16, X, m![Lane, Split, Term, Reach]>;

/// `y[c, t] = sum_k f[k] x[c, clamp(2t + k - 5)]` for every row of each block, the result in the
/// output span `S`; `input` repeats its edge samples over its padding rows.
impl<'x, C: AxisName, S: Doubled, X: M, W> Op<(&'x Frames<bf16, C, S::Twice, X>, PhantomData<S>)>
    for Convolution<W, (), 2>
where
    W: Borrow<DownBand<X>>,
{
    type Output = Blocked<bf16, C, S, X>;

    fn forward(
        &self,
        device: &mut Device,
        (input, _): (&'x Frames<bf16, C, S::Twice, X>, PhantomData<S>),
    ) -> Self::Output {
        const {
            assert!(matches!(
                Strategy::of(C::SIZE, C::SIZE, Strategy::reach(2), 2),
                Strategy::Depthwise
            ))
        };
        let band: &DownBand<X> = self.weight.borrow();
        type Rows<C, S, X> =
            DmTensor<bf16, X, m![Stereo], frames::Blocks<S>, m![{ Symbol<Groups<S>> }, Reach, C]>;
        // SAFETY: each window's sixteen-row groups are relabeled as the output's eight-row groups.
        let near: Rows<C, S, X> = unsafe {
            frames::FrameAxis::<S::Twice>::new()
                .forward(&mut *device, (input, LEAD - 5))
                .reshape()
        };
        let far: Rows<C, S, X> = unsafe {
            frames::FrameAxis::<S::Twice>::new()
                .forward(&mut *device, (input, LEAD - 5 + 16))
                .reshape()
        };
        let band: DmTensor<bf16, X, m![Stereo], frames::Blocks<S>, m![Lane, Split, Term, Reach]> =
            band.to_dm(&mut device.tdma);
        let register: TrfTensor<
            bf16,
            X,
            m![Stereo],
            frames::Blocks<S>,
            m![Lane],
            m![Split, Term, Reach],
        > = device
            .sub
            .begin(band.view())
            .fetch::<m![Lane, Split, Term], m![Reach]>()
            .collect::<m![Lane, Split, Term], m![Reach]>()
            .to_trf();
        type Time<C, S> = m![{ Symbol<Groups<S>> }, { Chunks<C> }, Split, Term, Reach];
        let filtered: DmTensor<
            bf16,
            X,
            m![Stereo],
            frames::Blocks<S>,
            m![{ Symbol<Groups<S>> }, Lane, C],
        > = device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(near.view(), far.view())
        .fetch::<Time<C, S>, Packet<C>>()
        .collect::<Time<C, S>, Flit<C>>()
        .contract_outer::<Time<C, S>, Flit<C>, _, _, _>(&register)
        .contract_packet::<Packet<C>>()
        .contract_time::<m![{ Symbol<Groups<S>> }, { Chunks<C> }]>()
        .contract_lane::<
            m![{ Symbol<Groups<S>> }, { Chunks<C> }, Lane, C % ({ Symbol<Narrow<C>> }) / 8],
            m![C % 8],
        >(LaneMode::Sequential)
        .cast::<bf16, m![C % 8 # 16]>()
        .commit_trim::<m![C % 8]>()
        .commit();
        // SAFETY: a group's eight lanes are the block's rows in order.
        unsafe { filtered.reshape() }
    }
}

/// Banded coefficients into one channel: lane `l` weighs row `q` of the window pair by tap `q - l`;
/// held as `T`, the HBM tensor or a borrow of it.
pub type Band<C, X = Chip, T = HbmTensor<bf16, X, m![Lane, Term, Lag, C]>> =
    Weight<m![Lane, Term, Lag, C], X, T>;

/// The band of the weight-normalized seven taps of `weight_g` and `weight_v`, `[1, channel, tap]`
/// as stored: lane `l`, row `q` holds tap `q - l` of every channel, on every chip.
impl<C: AxisName, X: M + 'static, T> Convolution<Band<C, X, T>> {
    fn band(
        source: &Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<HostTensor<bf16, m![{ X }, { m![Lane, Term, Lag, C] }]>, LoadError> {
        const TAPS: usize = 7;
        let channels = C::SIZE;
        let normed = normed(source, context, &[1, channels, TAPS])?;
        let band = (0..Lane::SIZE).flat_map(|lane| {
            let normed = &normed;
            (0..Term::SIZE * Lag::SIZE).flat_map(move |row| {
                (0..channels).map(move |channel| {
                    let value = match row.checked_sub(lane).filter(|&tap| tap < TAPS) {
                        Some(tap) => normed[channel * TAPS + tap] as f32,
                        None => 0.0,
                    };
                    bf16::from_f32(value)
                })
            })
        });
        let band: Vec<bf16> = band.collect();
        Ok(HostTensor::from_vec(band.repeat(X::SIZE)))
    }
}

/// The band of the weight-normalized taps; the convolution has no bias.
/// By hand, with the update below: one impl per weight form, where a derived impl is generic.
impl<'a, C: AxisName, X: M + 'static, T> Load<&'a Checkpoint> for Convolution<Band<C, X, T>>
where
    Band<C, X, T>: Load<HostTensor<bf16, m![{ X }, { m![Lane, Term, Lag, C] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let weight = cx.field("weight", "", |cx| {
                let band = Self::band(from, cx)?;
                Load::load(cx, band)
            })?;
            Ok(Self {
                weight,
                bias: (),
                pads: PhantomData,
            })
        })
    }
}

/// The band written in place, one rule of an update.
impl<'t, 's, C: AxisName, X: M + 'static, T>
    Load<(&'t mut Convolution<Band<C, X, T>>, &'s Checkpoint)>
    for &'t mut Convolution<Band<C, X, T>>
where
    for<'u> &'u mut Band<C, X, T>: Load<(
        &'u mut Band<C, X, T>,
        HostTensor<bf16, m![{ X }, { m![Lane, Term, Lag, C] }]>,
    )>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Convolution<Band<C, X, T>>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("weight", "", |cx| {
                cx.rule(
                    |cx| Convolution::<Band<C, X, T>>::band(from, cx),
                    |cx, host| {
                        <&mut Band<C, X, T> as Load<_>>::load(cx, (&mut into.weight, host))
                            .map(drop)
                    },
                )
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

/// `y[t] = sum_k sum_c w[k, c] x[c, t + k - 3]` for seven taps, the input channels into one;
/// `input` holds zeros in its padding rows.
impl<'x, C: AxisName, S: Window, X: M, T> Op<&'x Frames<bf16, C, S, X>>
    for Convolution<Band<C, X, T>>
where
    T: Borrow<HbmTensor<bf16, X, m![Lane, Term, Lag, C]>>,
{
    type Output = HbmTensor<f32, X, m![Stereo, { Symbol<S> }]>;

    fn forward(&self, device: &mut Device, input: &'x Frames<bf16, C, S, X>) -> Self::Output {
        const {
            assert!(C::SIZE <= 16);
            assert!(matches!(Strategy::of(1, 1, 7, 1), Strategy::Banded));
        };
        type Rows<C, S, X> =
            DmTensor<bf16, X, m![Stereo], frames::Blocks<S>, m![{ Symbol<Groups<S>> }, Lag, C]>;
        // SAFETY: each window's rows are relabeled as eight-row groups and their offset in the
        // group.
        let near: Rows<C, S, X> = unsafe {
            frames::FrameAxis::<S>::new()
                .forward(&mut *device, (input, LEAD - 3))
                .reshape()
        };
        let far: Rows<C, S, X> = unsafe {
            frames::FrameAxis::<S>::new()
                .forward(&mut *device, (input, LEAD - 3 + 8))
                .reshape()
        };
        let band: DmTensor<bf16, X, m![Stereo], frames::Blocks<S>, m![Lane, Term, Lag, C]> =
            self.weight.tensor().to_dm(&mut device.tdma);
        let register: TrfTensor<
            bf16,
            X,
            m![Stereo],
            frames::Blocks<S>,
            m![Lane],
            m![Term, Lag, C # 16],
        > = device
            .sub
            .begin(band.view())
            .fetch::<m![Lane, Term, Lag], m![C]>()
            .collect::<m![Lane, Term, Lag], m![C # 16]>()
            .to_trf();
        let samples: DmTensor<
            f32,
            X,
            m![Stereo],
            frames::Blocks<S>,
            m![{ Symbol<Groups<S>> }, Lane],
        > = device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(near.view(), far.view())
            .fetch::<m![{ Symbol<Groups<S>> }, Term, Lag], m![C]>()
            .collect::<m![{ Symbol<Groups<S>> }, Term, Lag], m![C # 16]>()
            .contract_outer::<m![{ Symbol<Groups<S>> }, Term, Lag], m![C # 16], _, _, _>(&register)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Symbol<Groups<S>> }]>()
            .contract_lane::<m![{ Symbol<Groups<S>> }], m![Lane]>(LaneMode::Interleaved)
            .commit_trim::<m![Lane]>()
            .commit();
        // SAFETY: a group's eight lanes are the block's rows in order.
        let samples: DmTensor<f32, X, m![Stereo], frames::Blocks<S>, m![{ frames::Row<S> }]> =
            unsafe { samples.reshape() };
        samples.to_hbm(&mut device.tdma)
    }
}

/// Scale an axis extent by an exact positive rational factor.
#[derive(Clone, Debug)]
pub struct Scale<Extent, const NUMERATOR: usize, const DENOMINATOR: usize>(PhantomData<Extent>);

impl<Extent: AxisName, const N: usize, const D: usize> AxisName for Scale<Extent, N, D> {
    const NAME: Ident = Extent::NAME;

    const SIZE: usize = {
        assert!(N > 0 && D > 0 && Extent::SIZE % D == 0);
        Extent::SIZE / D * N
    };
}

/// Half an even axis extent.
pub type Half<E> = Scale<E, 1, 2>;

/// Twice an axis extent.
pub type Double<E> = Scale<E, 2, 1>;

/// Row-major spatial positions of an image.
pub type Pixels<H, W> = Grid<H, W, Single, Single>;

/// Channel-first FP32 coefficients over one spatial axis: output channel, input channel, tap.
pub type Temporal<In, Out, Tap> = HbmTensor<f32, m![1], m![Out, In, Tap]>;

/// Preserve temporal length with symmetric `Kernel::SIZE / 2` zero padding.
impl<I: AxisName, Out: AxisName, Tap: AxisName, T: AxisName> Op<&HbmTensor<f32, m![1], m![I, T]>>
    for Convolution<Temporal<I, Out, Tap>, Bias<Out, m![1]>>
{
    type Output = HbmTensor<f32, m![1], m![Out, T]>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<f32, m![1], m![I, T]>) -> Self::Output {
        const { assert!(Tap::SIZE > 0 && Tap::SIZE % 2 == 1) };
        todo!("NPU stride-one temporal convolution with symmetric kernel/2 zero padding")
    }
}

/// Halve temporal length with a three-element kernel and one zero on each side.
impl<I: AxisName, Out: AxisName, Tap: AxisName, T: AxisName> Op<&HbmTensor<f32, m![1], m![I, T]>>
    for Convolution<Temporal<I, Out, Tap>, Bias<Out, m![1]>, 2>
{
    type Output = HbmTensor<f32, m![1], m![Out, { Symbol<Half<T>> }]>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<f32, m![1], m![I, T]>) -> Self::Output {
        const { assert!(Tap::SIZE == 3 && T::SIZE % 2 == 0) };
        todo!("NPU stride-two temporal convolution with one zero on each side")
    }
}

/// Convert channel-first temporal features into contiguous feature rows.
#[derive(Default, DeviceSend, Load)]
pub struct Transpose;

impl<C: AxisName, T: AxisName> Op<&HbmTensor<f32, m![1], m![C, T]>> for Transpose {
    type Output = HbmTensor<f32, m![1], m![T, C]>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<f32, m![1], m![C, T]>) -> Self::Output {
        todo!("NPU transpose from channel-first temporal features to contiguous feature rows")
    }
}

/// Channel-first FP32 coefficients over two spatial axes: output channel, input channel, kernel
/// row and column.
pub type Spatial<In, Out, Kernel> = HbmTensor<
    f32,
    m![1],
    m![
        { Symbol<linear::Outputs<Out>> },
        { Symbol<linear::Inputs<In>> },
        { Symbol<KernelRows<Kernel>> },
        { Symbol<KernelColumns<Kernel>> }
    ],
>;

/// Vertical position in a square convolution kernel.
#[derive(Clone, Debug)]
pub struct KernelRows<K>(PhantomData<K>);

impl<K: AxisName> AxisName for KernelRows<K> {
    const NAME: Ident = Ident::new("KernelRows");
    const SIZE: usize = K::SIZE;
}

/// Horizontal position in a square convolution kernel.
#[derive(Clone, Debug)]
pub struct KernelColumns<K>(PhantomData<K>);

impl<K: AxisName> AxisName for KernelColumns<K> {
    const NAME: Ident = Ident::new("KernelColumns");
    const SIZE: usize = K::SIZE;
}

/// Preserve image dimensions with symmetric `Kernel::SIZE / 2` zero padding.
impl<I: AxisName, Out: AxisName, Kernel: AxisName, H: AxisName, W: AxisName> Op<&Image<I, H, W>>
    for Convolution<Spatial<I, Out, Kernel>, Bias<Out, m![1]>>
{
    type Output = Image<Out, H, W>;

    fn forward(&self, _: &mut Device, _: &Image<I, H, W>) -> Self::Output {
        const { assert!(Kernel::SIZE > 0 && Kernel::SIZE % 2 == 1) };
        todo!("NPU stride-one convolution with symmetric kernel/2 zero padding")
    }
}

/// Halve both image dimensions with a three-by-three kernel and bottom/right zero padding.
impl<I: AxisName, Out: AxisName, Kernel: AxisName, H: AxisName, W: AxisName> Op<&Image<I, H, W>>
    for Convolution<Spatial<I, Out, Kernel>, Bias<Out, m![1]>, 2>
{
    type Output = Image<Out, Half<H>, Half<W>>;

    fn forward(&self, _: &mut Device, _: &Image<I, H, W>) -> Self::Output {
        const { assert!(Kernel::SIZE == 3 && H::SIZE % 2 == 0 && W::SIZE % 2 == 0) };
        todo!("NPU stride-two convolution with zero padding only at bottom and right")
    }
}

/// Double both image dimensions by nearest-neighbor resampling.
#[derive(Default, DeviceSend, Load)]
pub struct Upsample;

impl<C: AxisName, H: AxisName, W: AxisName> Op<&Image<C, H, W>> for Upsample {
    type Output = Image<C, Double<H>, Double<W>>;

    fn forward(&self, _: &mut Device, _: &Image<C, H, W>) -> Self::Output {
        todo!("NPU nearest-neighbor spatial resampling by two")
    }
}

/// Convert CHW storage into spatial rows of channels, retaining the image-shape witness.
#[derive(Default, DeviceSend, Load)]
pub struct Flatten;

impl<C: AxisName, H: AxisName, W: AxisName> Op<&Image<C, H, W>> for Flatten {
    type Output = (image::Rows<C, Pixels<H, W>>, PhantomData<(H, W)>);

    fn forward(&self, _: &mut Device, _: &Image<C, H, W>) -> Self::Output {
        todo!("NPU transpose from CHW to row-major spatial positions and channels")
    }
}

/// Convert spatial rows of channels back into CHW storage using the image-shape witness.
#[derive(Default, DeviceSend, Load)]
pub struct Restore;

impl<C: AxisName, H: AxisName, W: AxisName> Op<(&image::Rows<C, Pixels<H, W>>, PhantomData<(H, W)>)>
    for Restore
{
    type Output = Image<C, H, W>;

    fn forward(
        &self,
        _: &mut Device,
        _: (&image::Rows<C, Pixels<H, W>>, PhantomData<(H, W)>),
    ) -> Self::Output {
        todo!("NPU transpose from row-major spatial positions and channels to CHW")
    }
}

/// `y[t] = b + sum_k W_k x[t + k * dilation - pad]` in FP32, `pad = (taps - 1) * dilation / 2`,
/// over zero-padded `input`; the result's padding rows are left unwritten.
struct Dense<'a, In: AxisName, Out: AxisName, Tap: AxisName, X: M> {
    pub weight: &'a HbmTensor<bf16, X, m![Out, Tap, In]>,
    pub bias: &'a HbmTensor<f32, X, m![Out]>,
}

impl<
    'x,
    D: MaterializableScalar + FetchCast<bf16>,
    In: AxisName,
    Out: AxisName,
    Tap: AxisName,
    S: Window,
    X: M,
> Op<(&'x Frames<D, In, S, X>, usize)> for Dense<'_, In, Out, Tap, X>
{
    type Output = Frames<f32, Out, S, X>;

    fn forward(
        &self,
        device: &mut Device,
        (input, dilation): (&'x Frames<D, In, S, X>, usize),
    ) -> Self::Output {
        let (weight, bias) = (self.weight, self.bias);
        const { assert!(Tap::SIZE % 2 == 1) };
        let pad = (Tap::SIZE - 1) * dilation / 2;
        let mut columns: DmTensor<
            D,
            X,
            m![Stereo],
            Slices<D, In, Out, Tap, S>,
            m![Tap, { Row<In, Out, Tap, S> }, { Packet<In> }],
        > = DmTensor::new();
        #[unroll]
        for tap in 0..Tap::SIZE {
            S::window(input, LEAD - pad + tap * dilation).to_dm_view(
                &mut device.tdma,
                columns.view_mut().tile::<m![Tap], 1, m![
                    Tap = 1 #{!} Tap,
                    { Row<In, Out, Tap, S> },
                    { Packet<In> }
                ]>(tap),
            );
        }
        let mut output = Frames::<f32, Out, S, X>::new();
        type Selected<D, In, Out, Tap, S> =
            m![{ Slab<D, In, Out, Tap, S> } = 1 # ({ Slab<D, In, Out, Tap, S> })];
        type Time<D, In, Out, Tap, S> =
            m![{ Row<In, Out, Tap, S> }, { Lanes<D, In, Out, Tap, S> }, Tap];
        for slab in 0..<Slab<D, In, Out, Tap, S> as M>::SIZE {
            let weight: DmTensor<
                bf16,
                X,
                m![Stereo],
                Slices<D, In, Out, Tap, S>,
                m![{ Within<D, In, Out, Tap, S> }, Tap, { Packet<In> }],
            > = weight
                .view()
                .tile::<Slab<D, In, Out, Tap, S>, 1, m![
                    { Selected<D, In, Out, Tap, S> },
                    Out % (Out / ({ Count<D, In, Out, Tap, S> })),
                    Tap,
                    In
                ]>(slab)
                .to_dm(&mut device.tdma);
            let register: TrfTensor<
                bf16,
                X,
                m![Stereo],
                Slices<D, In, Out, Tap, S>,
                m![Out % 8],
                m![{ Lanes<D, In, Out, Tap, S> }, Tap, { Flit<In> }],
            > = device
                .sub
                .begin(weight.view())
                .fetch::<m![Out % 8, { Lanes<D, In, Out, Tap, S> }, Tap], Packet<In>>()
                .collect::<m![Out % 8, { Lanes<D, In, Out, Tap, S> }, Tap], Flit<In>>()
                .to_trf();
            let bias: DmTensor<
                f32,
                X,
                m![Stereo],
                Reduced<D, In, Out, Tap, S>,
                m![{ Within<D, In, Out, Tap, S> }],
            > = bias
                .view()
                .tile::<Slab<D, In, Out, Tap, S>, 1, m![
                    { Selected<D, In, Out, Tap, S> },
                    Out % (Out / ({ Count<D, In, Out, Tap, S> }))
                ]>(slab)
                .to_dm(&mut device.tdma);
            let bias: VrfTensor<
                f32,
                X,
                m![Stereo],
                Reduced<D, In, Out, Tap, S>,
                m![{ Within<D, In, Out, Tap, S> }],
            > = device
                .sub
                .begin(bias.view())
                .fetch::<m![{ Lanes<D, In, Out, Tap, S> }], m![Out % 8]>()
                .collect::<m![{ Lanes<D, In, Out, Tap, S> }], m![Out % 8]>()
                .to_vrf();
            let result: DmTensor<
            f32,
            X,
            m![Stereo],
            Reduced<D, In, Out, Tap, S>,
            m![{ Row<In, Out, Tap, S> }, { Within<D, In, Out, Tap, S> }],
        > = device
            .main
            .begin(columns.view())
            .fetch::<Time<D, In, Out, Tap, S>, Packet<In>>()
            .fetch_cast::<bf16>()
            .collect::<Time<D, In, Out, Tap, S>, Flit<In>>()
            .contract_outer::<Time<D, In, Out, Tap, S>, Flit<In>, _, _, _>(&register)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Row<In, Out, Tap, S> }, { Lanes<D, In, Out, Tap, S> }]>()
            .contract_lane::<
                m![{ Row<In, Out, Tap, S> }, { Lanes<D, In, Out, Tap, S> }],
                m![Out % 8],
            >(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<
                Reduced<D, In, Out, Tap, S>,
                m![{ Row<In, Out, Tap, S> }, { Lanes<D, In, Out, Tap, S> }],
            >(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![{ Row<In, Out, Tap, S> }, { Lanes<D, In, Out, Tap, S> }, Out % 8 / 4],
                m![Out % 4],
            >()
            .vector_fp_binary(FpBinaryOp::AddF, &bias)
            .vector_widen_concat::<
                m![{ Row<In, Out, Tap, S> }, { Lanes<D, In, Out, Tap, S> }],
                m![Out % 8],
            >()
            .vector_final()
            .commit_trim::<m![Out % 8]>()
            .commit();
            // SAFETY: the interior's channels are regrouped in place as slab and channel of the
            // slab.
            let interior: HbmTensorViewMut<'_, f32, X, _> = unsafe {
                S::interior(&mut output).reshape::<X, m![
                    Stereo,
                    { Symbol<S> } #{!} ({ FrameRows<S> }),
                    { Slab<D, In, Out, Tap, S> },
                    Out % (Out / ({ Count<D, In, Out, Tap, S> }))
                ]>()
            };
            result.view().to_hbm_view(
                &mut device.tdma,
                interior.tile::<Slab<D, In, Out, Tap, S>, 1, m![
                    Stereo,
                    { Symbol<S> } #{!} ({ FrameRows<S> }),
                    { Slab<D, In, Out, Tap, S> } = 1 #{!} ({ Slab<D, In, Out, Tap, S> }),
                    Out % (Out / ({ Count<D, In, Out, Tap, S> }))
                ]>(slab),
            );
        }
        output
    }
}

type Part<In> = Symbol<Narrow<In>>;
type Block<Out, Tap> = Symbol<Outputs<Out, Tap>>;
type Rows<In, Out, Tap, S> = Symbol<Group<In, Out, Tap, S>>;
type Count<D, In, Out, Tap, S> = Symbol<Slabs<D, In, Out, Tap, S>>;

/// A slab's output channels: the outer digit of `Out`.
type Slab<D, In, Out, Tap, S> = m![Out / (Out / ({ Count<D, In, Out, Tap, S> }))];
/// A slice's output channels of one slab.
type Within<D, In, Out, Tap, S> = m![Out % ({ Block<Out, Tap> } / ({ Count<D, In, Out, Tap, S> }))];
/// The output block of a slab a slice computes.
type Blocks<D, In, Out, Tap, S> = m![
    Out % (Out / ({ Count<D, In, Out, Tap, S> })) / ({ Block<
        Out,
        Tap,
    > } / ({ Count<D, In, Out, Tap, S> }))
];

/// A cluster's slices: row group, output block, input part.
type Slices<D, In, Out, Tap, S> = m![[
    { Symbol<S> } / ({ Rows<In, Out, Tap, S> }),
    { Blocks<D, In, Out, Tap, S> },
    In / ({ Part<In> })
] # 256];

/// The slices holding results: the first input part of each row group and output block.
type Reduced<D, In, Out, Tap, S> = m![[
    { Symbol<S> } / ({ Rows<In, Out, Tap, S> }),
    { Blocks<D, In, Out, Tap, S> },
    1 # (In / ({ Part<In> }))
] # 256];

type Row<In, Out, Tap, S> = m![{ Symbol<S> } % ({ Rows<In, Out, Tap, S> })];
type Packet<In> = m![In % ({ Part<In> })];
type Chunks<In> = m![In / ({ Part<In> })];
type Flit<In> = m![In % ({ Part<In> }) # 16];
type Lanes<D, In, Out, Tap, S> = m![{ Within<D, In, Out, Tap, S> } / 8];

/// The weight-normalized coefficients at the context's position in FP64: `g v / |v|` for each row
/// of `weight_v` of `shape`, with its `weight_g`, the norm over all but the first axis.
// TODO: read the values only for a leaf that holds bytes, as a converted leaf's build does.
pub(crate) fn normed(
    source: &Checkpoint,
    context: &mut Context<'_>,
    shape: &[usize],
) -> Result<Vec<f64>, LoadError> {
    let rows = shape[0];
    let mut children = Children::default();
    let magnitude = context.field("weight_g", "weight_g", |cx| {
        cx.floats(source, &[rows, 1, 1])
    });
    let magnitude = children.ask(magnitude)?;
    let direction = context.field("weight_v", "weight_v", |cx| cx.floats(source, shape));
    let direction = children.ask(direction)?;
    let (magnitude, direction) = children.node(magnitude.zip(direction))?;
    let row = direction.len() / rows;
    let normed = direction
        .chunks_exact(row)
        .zip(magnitude)
        .flat_map(|(row, g)| {
            let norm = row
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt();
            row.iter().map(move |&v| f64::from(g) * f64::from(v) / norm)
        });
    Ok(normed.collect())
}

#[cfg(test)]
pub(crate) mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use super::frames::{self, FrameRows, Frames, LEAD, Span, Stereo, Window};
    use crate::tensor::parallel::Chip;

    /// Pseudo-random values in `[-1, 1)`.
    impl Case {
        pub(crate) fn random(count: usize, mut state: u64) -> Vec<f32> {
            (0..count)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
                })
                .collect()
        }

        fn bf(values: Vec<f32>) -> Vec<f32> {
            values
                .into_iter()
                .map(|v| bf16::from_f32(v).to_f32())
                .collect()
        }

        /// A stereo signal as a frame of `rows` rows, every padding row zero.
        pub(crate) fn frame(
            samples: usize,
            channels: usize,
            rows: usize,
            values: &[f32],
        ) -> Vec<f32> {
            let mut frame = vec![0.0; 2 * rows * channels];
            for b in 0..2 {
                for t in 0..samples {
                    let (to, from) = (
                        (b * rows + LEAD + t) * channels,
                        (b * samples + t) * channels,
                    );
                    frame[to..to + channels].copy_from_slice(&values[from..from + channels]);
                }
            }
            frame
        }
    }
    /// Random BF16 signal, weights and FP32 bias of a convolution.
    pub(crate) struct Case {
        pub(crate) signal: Vec<f32>,
        pub(crate) weights: Vec<f32>,
        pub(crate) bias: Vec<f32>,
    }

    impl Case {
        pub(crate) fn new(samples: usize, cin: usize, cout: usize, taps: usize) -> Self {
            let scale = 1.0 / ((cin * taps) as f32).sqrt();
            Self {
                signal: Self::bf(Self::random(2 * samples * cin, 7)),
                weights: Self::bf(
                    Self::random(cout * taps * cin, 11)
                        .into_iter()
                        .map(|v| v * scale)
                        .collect(),
                ),
                bias: Self::random(cout, 13),
            }
        }
    }

    /// The case uploaded as the device convolution reads it.
    impl Case {
        async fn upload<In: AxisName, Out: AxisName, Tap: AxisName, S: Window>(
            device: &mut Device,
            case: &Case,
            samples: usize,
        ) -> (
            Frames<bf16, In, S>,
            HbmTensor<bf16, Chip, m![Out, Tap, In]>,
            HbmTensor<f32, Chip, m![Out]>,
        ) {
            let rows = LEAD + S::SIZE + LEAD;
            let input = HostTensor::<bf16, m![Stereo, { FrameRows<S> }, In]>::from_vec(
                Self::frame(samples, In::SIZE, rows, &case.signal)
                    .into_iter()
                    .map(bf16::from_f32)
                    .collect::<Vec<_>>(),
            )
            .to_hbm::<Chip, m![Stereo, { FrameRows<S> }, In]>(&mut device.pdma)
            .await
            .unwrap();
            let weight = HostTensor::<bf16, m![Out, Tap, In]>::from_vec(
                case.weights
                    .iter()
                    .copied()
                    .map(bf16::from_f32)
                    .collect::<Vec<_>>(),
            )
            .to_hbm::<Chip, m![Out, Tap, In]>(&mut device.pdma)
            .await
            .unwrap();
            let bias = HostTensor::<f32, m![Out]>::from_vec(case.bias.clone())
                .to_hbm::<Chip, m![Out]>(&mut device.pdma)
                .await
                .unwrap();
            (input, weight, bias)
        }

        /// The device result against the definition with zeros outside the signal, on every
        /// `stride`-th output row and the last; padding rows must be zero.
        fn check<In: AxisName, Out: AxisName, Tap: AxisName, const DILATION: usize, S: Window>(
            case: &Case,
            actual: &[f32],
            samples: usize,
            stride: usize,
        ) {
            Self::check_rows::<In, Out, Tap, DILATION, S>(case, actual, samples, stride, true);
        }

        fn check_rows<
            In: AxisName,
            Out: AxisName,
            Tap: AxisName,
            const DILATION: usize,
            S: Window,
        >(
            case: &Case,
            actual: &[f32],
            samples: usize,
            stride: usize,
            pads: bool,
        ) {
            let (cin, cout, taps) = (In::SIZE, Out::SIZE, Tap::SIZE);
            let pad = (taps - 1) * DILATION / 2;
            let rows = LEAD + S::SIZE + LEAD;
            let mut worst = 0.0f64;
            for b in 0..2 {
                for t in (0..samples).step_by(stride).chain([samples - 1]) {
                    for o in 0..cout {
                        let mut want = f64::from(case.bias[o]);
                        for k in 0..taps {
                            let s = (t + k * DILATION) as isize - pad as isize;
                            if (0..samples as isize).contains(&s) {
                                let x = &case.signal[(b * samples + s as usize) * cin..][..cin];
                                let w = &case.weights[(o * taps + k) * cin..][..cin];
                                want += x
                                    .iter()
                                    .zip(w)
                                    .map(|(x, w)| f64::from(*x) * f64::from(*w))
                                    .sum::<f64>();
                            }
                        }
                        let got = f64::from(actual[(b * rows + LEAD + t) * cout + o]);
                        let error = (got - want).abs() / want.abs().max(1.0);
                        worst = worst.max(error);
                        assert!(
                            error <= 1e-4,
                            "stereo {b} row {t} channel {o}: {got} vs {want}"
                        );
                    }
                }
                for t in (0..LEAD)
                    .chain(LEAD + samples..LEAD + samples + LEAD)
                    .filter(|_| pads)
                {
                    for o in 0..cout {
                        assert_eq!(actual[(b * rows + t) * cout + o], 0.0, "padding row {t}");
                    }
                }
            }
            eprintln!("conv {cin}->{cout} k{taps} d{DILATION} worst relative error {worst:e}");
        }
    }
    mod narrow {
        use super::super::Dense;
        use super::*;
        use crate::Op;
        use crate::tests::DEVICE;
        type Rows = Span<4000, 16, 3990>;
        axes![In = 8, Out = 8, Tap = 11];

        #[device(chip = 1)]
        fn convolution(
            device: &mut Device,
            input: &Frames<bf16, In, Rows>,
            weight: &HbmTensor<bf16, Chip, m![Out, Tap, In]>,
            bias: &HbmTensor<f32, Chip, m![Out]>,
        ) -> Frames<f32, Out, Rows> {
            let mut output = Dense { weight, bias }.forward(&mut *device, (input, 5));
            frames::Zero.forward(&mut *device, (&mut output, 4000));
            output
        }

        /// Eleven taps at dilation five over eight channels: one slice a block of rows.
        #[tokio::test(flavor = "current_thread")]
        async fn dilated_convolution_matches_definition() {
            let _lease = DEVICE.lock().await;
            let mut device = Device::new(convolution.topology()).unwrap();
            let case = Case::new(4000, In::SIZE, Out::SIZE, Tap::SIZE);
            let (input, weight, bias) =
                Case::upload::<In, Out, Tap, Rows>(&mut device, &case, 4000).await;
            let output = launch(convolution, (&mut device, &input, &weight, &bias))
                .await
                .unwrap();
            let actual = output
                .to_host::<m![Stereo, { FrameRows<Rows> }, Out]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            Case::check::<In, Out, Tap, 5, Rows>(&case, &actual, 4000, 1);
        }
    }

    mod split {
        use super::super::Dense;
        use super::*;
        use crate::Op;
        use crate::tests::DEVICE;
        type Rows = Span<1040, 8, 1035>;
        axes![In = 512, Out = 512, Tap = 11];

        #[device(chip = 1)]
        fn convolution(
            device: &mut Device,
            input: &Frames<bf16, In, Rows>,
            weight: &HbmTensor<bf16, Chip, m![Out, Tap, In]>,
            bias: &HbmTensor<f32, Chip, m![Out]>,
        ) -> Frames<f32, Out, Rows> {
            let mut output = Dense { weight, bias }.forward(&mut *device, (input, 5));
            frames::Zero.forward(&mut *device, (&mut output, 1035));
            output
        }

        /// Eleven taps at dilation five over 512 channels: input parts reduce across slices and
        /// the rows are written on every output block's slices.
        #[tokio::test(flavor = "current_thread")]
        async fn split_convolution_matches_definition() {
            let _lease = DEVICE.lock().await;
            let mut device = Device::new(convolution.topology()).unwrap();
            let case = Case::new(1035, In::SIZE, Out::SIZE, Tap::SIZE);
            let (input, weight, bias) =
                Case::upload::<In, Out, Tap, Rows>(&mut device, &case, 1035).await;
            let output = launch(convolution, (&mut device, &input, &weight, &bias))
                .await
                .unwrap();
            let actual = output
                .to_host::<m![Stereo, { FrameRows<Rows> }, Out]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            Case::check::<In, Out, Tap, 5, Rows>(&case, &actual, 1035, 37);
        }
    }

    mod wide {
        use super::super::Dense;
        use super::*;
        use crate::Op;
        use crate::tests::DEVICE;
        type Rows = Span<208, 8, 207>;
        axes![In = 1024, Out = 2560, Tap = 3];

        #[device(chip = 1)]
        fn single(
            device: &mut Device,
            input: &Frames<f32, In, Rows>,
            weight: &HbmTensor<bf16, Chip, m![Out, Tap, In]>,
            bias: &HbmTensor<f32, Chip, m![Out]>,
        ) -> Frames<f32, Out, Rows> {
            Dense { weight, bias }.forward(device, (input, 1))
        }

        #[device(chip = 1)]
        fn half(
            device: &mut Device,
            input: &Frames<bf16, In, Rows>,
            weight: &HbmTensor<bf16, Chip, m![Out, Tap, In]>,
            bias: &HbmTensor<f32, Chip, m![Out]>,
        ) -> Frames<f32, Out, Rows> {
            Dense { weight, bias }.forward(device, (input, 1))
        }

        /// 1024 input channels in 64 parts and 2560 outputs in four blocks of two slabs, every row
        /// on one slice group: FP32 input.
        #[tokio::test(flavor = "current_thread")]
        async fn wide_single_convolution_matches_definition() {
            let _lease = DEVICE.lock().await;
            let mut device = Device::new(single.topology()).unwrap();
            let case = Case::new(207, In::SIZE, Out::SIZE, Tap::SIZE);
            let (_, weight, bias) =
                Case::upload::<In, Out, Tap, Rows>(&mut device, &case, 207).await;
            let rows = LEAD + Rows::SIZE + LEAD;
            let input = HostTensor::<f32, m![Stereo, { FrameRows<Rows> }, In]>::from_vec(
                Case::frame(207, In::SIZE, rows, &case.signal),
            )
            .to_hbm::<Chip, m![Stereo, { FrameRows<Rows> }, In]>(&mut device.pdma)
            .await
            .unwrap();
            let output = launch(single, (&mut device, &input, &weight, &bias))
                .await
                .unwrap();
            let actual = output
                .to_host::<m![Stereo, { FrameRows<Rows> }, Out]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            Case::check_rows::<In, Out, Tap, 1, Rows>(&case, &actual, 207, 23, false);
        }

        /// The same with BF16 input.
        #[tokio::test(flavor = "current_thread")]
        async fn wide_half_convolution_matches_definition() {
            let _lease = DEVICE.lock().await;
            let mut device = Device::new(half.topology()).unwrap();
            let case = Case::new(207, In::SIZE, Out::SIZE, Tap::SIZE);
            let (input, weight, bias) =
                Case::upload::<In, Out, Tap, Rows>(&mut device, &case, 207).await;
            let output = launch(half, (&mut device, &input, &weight, &bias))
                .await
                .unwrap();
            let actual = output
                .to_host::<m![Stereo, { FrameRows<Rows> }, Out]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            Case::check_rows::<In, Out, Tap, 1, Rows>(&case, &actual, 207, 23, false);
        }
    }
}
