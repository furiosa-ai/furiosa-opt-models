//! Projections of packed rows whose FP32 offsets join their sums before one rounding, and the
//! projections of FP32 rows or coefficients through their BF16 high and low parts.

use crate::Op;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::index::packed::Rows;
use crate::ops::linear::Linear;
use crate::ops::normalization::adaptive::{
    Banded, Feature, Half, Operand, Slices, Strip, Strips, Tile,
};
use crate::ops::normalization::{Chunk, FeatureSlices};
use crate::tensor::form::Weight;
use crate::tensor::form::split::{Sides, Stacked};
use core::borrow::Borrow;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

/// BF16 rows projected by `weight` with a per-output FP32 `bias`, each FP32 sum plus its offset
/// rounded once to BF16; both are held owned or borrowed.
#[derive(DeviceSend, Load)]
pub struct Biased<
    In: AxisName,
    Out: AxisName,
    X: M = m![1],
    Matrix = Linear<In, Out, X>,
    Offset = HbmTensor<f32, X, m![Out]>,
> {
    #[load(rename = "")]
    weight: Matrix,
    bias: Offset,
    shape: PhantomData<fn() -> (In, Out, X)>,
}

impl<In: AxisName, Out: AxisName, X: M> Biased<In, Out, X> {
    pub fn new(weight: Weight<m![Out, In], X>, bias: HbmTensor<f32, X, m![Out]>) -> Self {
        Self {
            weight: Linear::new(weight),
            bias,
            shape: PhantomData,
        }
    }
}

impl<'x, In: AxisName, Out: AxisName, R: AxisName, X: M, Matrix, Offset> Op<&'x Rows<R, In, X>>
    for Biased<In, Out, X, Matrix, Offset>
where
    Matrix: for<'o> Op<(&'x Rows<R, In, X>, &'o mut HbmTensor<f32, X, m![R, Out]>), Output = ()>,
    Offset: Borrow<HbmTensor<f32, X, m![Out]>>,
{
    type Output = Rows<R, Out, X>;

    fn forward(&self, device: &mut Device, input: &'x Rows<R, In, X>) -> Self::Output {
        type C<Matrix> = Symbol<Chunk<Matrix>>;
        type Sums<D, Matrix, X> =
            DmTensor<D, X, Half, FeatureSlices<Matrix>, m![Strip % 32, Matrix % ({ C<Matrix> })]>;
        let mut sums: HbmTensor<f32, X, m![R, Out]> = HbmTensor::new();
        self.weight.forward(&mut *device, (input, &mut sums));
        let bias: &HbmTensor<f32, X, m![Out]> = self.bias.borrow();
        let bias: DmTensor<f32, X, Half, FeatureSlices<Out>, m![Out % ({ C<Out> })]> =
            bias.view().to_dm(&mut device.tdma);
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
            let in_dm: Sums<f32, Out, X> = sums
                .tile::<m![{ Symbol<Strips<R>> }], 1, m![
                    { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }),
                    Strip,
                    Out
                ]>(tile)
                .to_dm(&mut device.tdma);
            let rounded: Sums<bf16, Out, X> = device
                .main
                .begin(in_dm.view())
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

axes![Part = 2, Frequency = 128];

/// An FP32 projection as its checkpoint stores it: `weight`, `Out` rows of `In` values, and `bias`,
/// `Out` values, each stored as F32 or BF16.
pub struct Stored<Out, In> {
    weight: Vec<f32>,
    bias: Vec<f32>,
    shape: PhantomData<fn() -> (Out, In)>,
}

/// By hand: the weight and bias are read as host floats, a value no tree leaf holds.
impl<'a, Out: AxisName, In: AxisName> Load<&'a Checkpoint> for Stored<Out, In> {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        // TODO: read the values only for a leaf that holds bytes, as a converted leaf's build does.
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let weight = cx.field("weight", "weight", |cx| {
                cx.floats(from, &[Out::SIZE, In::SIZE])
            });
            let weight = children.ask(weight)?;
            let bias =
                children.ask(cx.field("bias", "bias", |cx| cx.floats(from, &[Out::SIZE])))?;
            children.node(weight.zip(bias)).map(|(weight, bias)| Self {
                weight,
                bias,
                shape: PhantomData,
            })
        })
    }
}

impl<Out, In> Stored<Out, In> {
    /// Every weight value's BF16 high part, its rounding, and its low part, the rounding of what
    /// remains.
    fn parts(&self) -> (Vec<bf16>, Vec<bf16>) {
        self.weight
            .iter()
            .map(|&value| {
                let high = bf16::from_f32(value);
                (high, bf16::from_f32(value - high.to_f32()))
            })
            .unzip()
    }
}

/// FP32 coefficients as stacked BF16 high and low parts, made from the checkpoint's FP32 `weight`,
/// with FP32 offsets: one contraction makes the FP32 product within 2^-17 of each term.
#[derive(DeviceSend, Load)]
#[load(from = Stored<Out, In>)]
pub struct Precise<
    In: AxisName,
    Out: AxisName,
    X: M = m![1],
    Matrix = Linear<In, Stacked<Out>, X>,
    Offset = HbmTensor<f32, X, m![Out]>,
> {
    weight: Matrix,
    bias: Offset,
    shape: PhantomData<fn() -> (In, Out, X)>,
}

impl<In: AxisName, Out: AxisName, X: M> Precise<In, Out, X> {
    pub fn new(
        parts: Weight<m![{ Symbol<Stacked<Out>> }, In], X>,
        bias: HbmTensor<f32, X, m![Out]>,
    ) -> Self {
        Self {
            weight: Linear::new(parts),
            bias,
            shape: PhantomData,
        }
    }
}

/// The stacked rows of `weight`: every output's high parts, then every output's low parts.
impl<In: AxisName, Out: AxisName, X: M + 'static, Matrix, Offset> Load<Stored<Out, In>>
    for Precise<In, Out, X, Matrix, Offset>
where
    Matrix: Load<HostTensor<bf16, m![{ X }, { m![{ Symbol<Stacked<Out>> }, In] }]>>,
    Offset: Load<HostTensor<f32, m![{ X }, { m![Out] }]>>,
{
    fn load(context: &mut Context<'_>, stored: Stored<Out, In>) -> Result<Self, LoadError> {
        let (high, low) = stored.parts();
        let parts = [high, low].concat();
        Ok(Self {
            weight: context.field("weight", "weight", |cx| {
                Matrix::load(cx, HostTensor::from_vec(parts.repeat(X::SIZE)))
            })?,
            bias: context.field("bias", "bias", |cx| {
                Offset::load(cx, HostTensor::from_vec(stored.bias.repeat(X::SIZE)))
            })?,
            shape: PhantomData,
        })
    }
}

impl<'x, In: AxisName, Out: AxisName, R: AxisName, X: M, Matrix, Offset> Op<&'x Rows<R, In, X>>
    for Precise<In, Out, X, Matrix, Offset>
where
    Matrix: for<'o> Op<
            (
                &'x Rows<R, In, X>,
                &'o mut HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]>,
            ),
            Output = (),
        >,
    Offset: Borrow<HbmTensor<f32, X, m![Out]>>,
{
    type Output = HbmTensor<f32, X, m![R, Out]>;

    fn forward(&self, device: &mut Device, input: &'x Rows<R, In, X>) -> Self::Output {
        // A row a slice: an output this narrow spreads over too few slices to share them.
        type Each = m![Strip % 32 # 256];
        type Sums<Out, X> = DmTensor<f32, X, Half, Each, m![Out]>;
        let mut sums: HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]> = HbmTensor::new();
        self.weight.forward(&mut *device, (input, &mut sums));
        let bias: &HbmTensor<f32, X, m![Out]> = self.bias.borrow();
        let bias: DmTensor<f32, X, Half, Each, m![Out]> = bias.view().to_dm(&mut device.tdma);
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

/// FP32 rows times FP32 coefficients plus offsets, rounded once, within 2^-17 of each term: rows as
/// BF16 high and low parts, coefficients from FP32 `weight` as `[high | high]` and `[low | 0]`.
#[derive(DeviceSend, Load)]
#[load(from = Stored<Out, In>)]
pub struct Embedding<
    In: AxisName,
    Out: AxisName,
    X: M = m![1],
    Matrix = Linear<Sides<In>, Stacked<Out>, X>,
    Offset = HbmTensor<f32, X, m![Out]>,
> {
    weight: Matrix,
    bias: Offset,
    shape: PhantomData<fn() -> (In, Out, X)>,
}

/// One part of tile `at`'s rows, stored beside the other.
impl<In: AxisName, R: AxisName, S: M, E: M, X: M>
    Op<(
        &DmTensor<bf16, X, Half, S, E>,
        &mut Rows<R, Sides<In>, X>,
        (usize, usize),
    )> for Sides<In>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (values, parts, (at, part)): (
            &DmTensor<bf16, X, Half, S, E>,
            &mut Rows<R, Sides<In>, X>,
            (usize, usize),
        ),
    ) {
        values.view().to_hbm_view(
            &mut device.tdma,
            // SAFETY: a row's parts are its high part, then its low part, and the tiles split the
            // rows in order.
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
            weight: Linear::new(parts),
            bias,
            shape: PhantomData,
        }
    }
}

/// The coefficients of `weight`: each output's high parts twice, then each output's low parts and
/// zeros.
impl<In: AxisName, Out: AxisName, X: M + 'static, Matrix, Offset> Load<Stored<Out, In>>
    for Embedding<In, Out, X, Matrix, Offset>
where
    Matrix: Load<
        HostTensor<
            bf16,
            m![{ X }, {
                m![{ Symbol<Stacked<Out>> }, { Symbol<Sides<In>> }]
            }],
        >,
    >,
    Offset: Load<HostTensor<f32, m![{ X }, { m![Out] }]>>,
{
    fn load(context: &mut Context<'_>, stored: Stored<Out, In>) -> Result<Self, LoadError> {
        let (high, low) = stored.parts();
        let zeros = vec![bf16::from_f32(0.0); In::SIZE];
        let high = high
            .chunks_exact(In::SIZE)
            .flat_map(|row| [row, row].concat());
        let low = low
            .chunks_exact(In::SIZE)
            .flat_map(|row| [row, &zeros].concat());
        let coefficients: Vec<bf16> = high.chain(low).collect();
        Ok(Self {
            weight: context.field("weight", "weight", |cx| {
                Matrix::load(cx, HostTensor::from_vec(coefficients.repeat(X::SIZE)))
            })?,
            bias: context.field("bias", "bias", |cx| {
                Offset::load(cx, HostTensor::from_vec(stored.bias.repeat(X::SIZE)))
            })?,
            shape: PhantomData,
        })
    }
}

/// Rows' high and low parts, side by side, project into the stacked sums.
impl<'x, 'o, In: AxisName, Out: AxisName, R: AxisName, X: M, Matrix, Offset>
    Op<(
        &'x Rows<R, Sides<In>, X>,
        &'o mut HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]>,
    )> for Embedding<In, Out, X, Matrix, Offset>
where
    Matrix: Op<
            (
                &'x Rows<R, Sides<In>, X>,
                &'o mut HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]>,
            ),
            Output = (),
        >,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        input: (
            &'x Rows<R, Sides<In>, X>,
            &'o mut HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]>,
        ),
    ) {
        self.weight.forward(device, input)
    }
}

impl<'x, In: AxisName, Out: AxisName, R: AxisName, X: M, Matrix, Offset>
    Op<&'x HbmTensor<f32, X, m![R, In]>> for Embedding<In, Out, X, Matrix, Offset>
where
    Matrix: for<'p, 'o> Op<
            (
                &'p Rows<R, Sides<In>, X>,
                &'o mut HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]>,
            ),
            Output = (),
        >,
    Offset: Borrow<HbmTensor<f32, X, m![Out]>>,
{
    type Output = Rows<R, Out, X>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<f32, X, m![R, In]>,
    ) -> Self::Output {
        type Each = m![Strip % 32 # 256];
        type Wide<Matrix, X> = DmTensor<f32, X, Half, Slices<Matrix>, m![Strip % 8, Matrix % 128]>;
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
            Sides::<In>::new().forward(&mut *device, (&high, &mut parts, (tile, 0)));
            Sides::<In>::new().forward(&mut *device, (&low, &mut parts, (tile, 1)));
        }
        let mut sums: HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]> = HbmTensor::new();
        self.weight.forward(&mut *device, (&parts, &mut sums));
        let bias: &HbmTensor<f32, X, m![Out]> = self.bias.borrow();
        let bias: Feature<f32, Out, X> = bias.view().to_dm(&mut device.tdma);
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
            Strips::<R>::new().forward(&mut *device, (&sum, &mut output, tile));
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
impl<In: AxisName, Out: AxisName, X: M, R: AxisName, Matrix, Offset>
    Op<(&HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]>,)>
    for Embedding<In, Out, X, Matrix, Offset>
where
    Offset: Borrow<HbmTensor<f32, X, m![Out]>>,
{
    type Output = HbmTensor<f32, X, m![R, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        (sums,): (&HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]>,),
    ) -> Self::Output {
        type C<Matrix> = Symbol<Chunk<Matrix>>;
        type Wide<Matrix, X> =
            DmTensor<f32, X, Half, FeatureSlices<Matrix>, m![Strip % 32, Matrix % ({ C<Matrix> })]>;
        let bias: &HbmTensor<f32, X, m![Out]> = self.bias.borrow();
        let bias: DmTensor<f32, X, Half, FeatureSlices<Out>, m![Out % ({ C<Out> })]> =
            bias.view().to_dm(&mut device.tdma);
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
            .vector_narrow_split::<
                m![Strip % 32, Out % ({ C<Out> }) / 8, Out % 8 / 4 % 2],
                m![Out % 4],
            >()
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
impl<In: AxisName, R: AxisName, X: M> Op<&HbmTensor<f32, X, m![R, In]>> for Sides<In> {
    type Output = Rows<R, Sides<In>, X>;

    fn forward(&self, device: &mut Device, input: &HbmTensor<f32, X, m![R, In]>) -> Self::Output {
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
            .vector_narrow_split::<
                m![Strip % 32, In % ({ C<In> }) / 8, In % 8 / 4 % 2],
                m![In % 4],
            >()
            .vector_fp_binary(FpBinaryOp::SubF, &widened)
            .vector_widen_concat::<m![Strip % 32, In % ({ C<In> }) / 8], m![In % 8]>()
            .vector_final()
            .cast::<bf16, m![In % 8 # 16]>()
            .commit_trim::<m![In % 8]>()
            .commit();
            self.forward(&mut *device, (&high, &mut parts, (tile, 0)));
            self.forward(&mut *device, (&low, &mut parts, (tile, 1)));
        }
        parts
    }
}

/// The time embedding of `t` in `[0, 1]`: `[cos | sin](t f)`, `f_j = 10000^(-j / 128)`, then FP32
/// projection, SiLU and FP32 projection, each within 2^-17 of each term; a load places `f`.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct TimeEmbedding<
    Hidden: AxisName,
    Out: AxisName,
    X: M = m![1],
    First = Embedding<Sinusoid, Hidden, X>,
    Second = Embedding<Hidden, Out, X>,
    Frequencies = HbmTensor<f32, X, m![Frequency]>,
> {
    frequencies: Frequencies,
    first: First,
    second: Second,
    shape: PhantomData<fn() -> (Hidden, Out, X)>,
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
            shape: PhantomData,
        }
    }
}

/// The two projections in their own scopes, the frequencies placed on every chip.
/// By hand, with the update below: a derived `with` field would rebuild the frequencies in every
/// update.
impl<'a, Hidden: AxisName, Out: AxisName, X: M + 'static, Frequencies, First, Second>
    Load<&'a Checkpoint> for TimeEmbedding<Hidden, Out, X, First, Second, Frequencies>
where
    First: Load<&'a Checkpoint>,
    Second: Load<&'a Checkpoint>,
    Frequencies: Load<HostTensor<f32, m![{ X }, { m![Frequency] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let first = children.ask(cx.field("first", "", |cx| First::load(cx, from)))?;
            let second = children.ask(cx.field("second", "", |cx| Second::load(cx, from)))?;
            let (first, second) = children.node(first.zip(second))?;
            let frequencies = (0..Frequency::SIZE)
                .map(|j| (-(10000f32.ln()) * j as f32 / Frequency::SIZE as f32).exp());
            let frequencies: Vec<f32> = frequencies.collect();
            let frequencies = cx.field("frequencies", "frequencies", |cx| {
                Frequencies::load(cx, HostTensor::from_vec(frequencies.repeat(X::SIZE)))
            })?;
            Ok(Self {
                frequencies,
                first,
                second,
                shape: PhantomData,
            })
        })
    }
}

/// The two projections written in place; the frequencies are no update target.
impl<'t, 's, Hidden: AxisName, Out: AxisName, X: M, Frequencies, First, Second>
    Load<(
        &'t mut TimeEmbedding<Hidden, Out, X, First, Second, Frequencies>,
        &'s Checkpoint,
    )> for &'t mut TimeEmbedding<Hidden, Out, X, First, Second, Frequencies>
where
    for<'u> &'u mut First: Load<(&'u mut First, &'s Checkpoint)>,
    for<'u> &'u mut Second: Load<(&'u mut Second, &'s Checkpoint)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut TimeEmbedding<Hidden, Out, X, First, Second, Frequencies>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("first", "", |cx| {
                <&mut First as Load<_>>::load(cx, (&mut into.first, from))
            })?;
            cx.field("second", "", |cx| {
                <&mut Second as Load<_>>::load(cx, (&mut into.second, from))
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

/// The two projections are called as modules, so a child its family names runs as it does.
impl<'x, Hidden: AxisName, Out: AxisName, R: AxisName, X: M, First, Second, Frequencies>
    Op<&'x HbmTensor<f32, X, m![R, 1 # 8]>>
    for TimeEmbedding<Hidden, Out, X, First, Second, Frequencies>
where
    First: for<'p, 'o, 'd> crate::Module<
            (
                &'p Rows<R, Sides<Sinusoid>, X>,
                &'o mut HbmTensor<f32, X, m![R, { Symbol<Stacked<Hidden>> }]>,
            ),
            (),
            &'d mut (),
            Output = (),
        > + for<'p, 'd> crate::Module<
            (&'p HbmTensor<f32, X, m![R, { Symbol<Stacked<Hidden>> }]>,),
            (),
            &'d mut (),
            Output = HbmTensor<f32, X, m![R, Hidden]>,
        >,
    Second: for<'p, 'o, 'd> crate::Module<
            (
                &'p Rows<R, Sides<Hidden>, X>,
                &'o mut HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]>,
            ),
            (),
            &'d mut (),
            Output = (),
        > + for<'p, 'd> crate::Module<
            (&'p HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]>,),
            (),
            &'d mut (),
            Output = HbmTensor<f32, X, m![R, Out]>,
        >,
    Frequencies: Borrow<HbmTensor<f32, X, m![Frequency]>>,
{
    type Output = HbmTensor<f32, X, m![R, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        time: &'x HbmTensor<f32, X, m![R, 1 # 8]>,
    ) -> Self::Output {
        use crate::ops::pointwise::activation::Silu;
        type Each = m![Strip % 32 # 256];
        type Tiles<X> = DmTensor<f32, X, Half, Each, m![Part, Frequency]>;
        let frequencies: &HbmTensor<f32, X, m![Frequency]> = self.frequencies.borrow();
        let frequencies: DmTensor<f32, X, Half, Each, m![Frequency]> =
            frequencies.view().to_dm(&mut device.tdma);
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
            let mut waves = Tiles::<X>::new();
            let wave = |device: &mut Device, op: FpUnaryOp, waves: &mut Tiles<X>, part: usize| {
                device
                    .main
                    .begin(frequencies.view())
                    .fetch::<m![Frequency / 8], m![Frequency % 8]>()
                    .collect::<m![Frequency / 8], m![Frequency % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<
                        m![Frequency / 8, Frequency % 8 / 4 % 2],
                        m![Frequency % 4],
                    >()
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
                // SAFETY: a sinusoid is its cosines, then its sines, and the tiles split the rows
                // in order.
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
        let first = Sides::<Sinusoid>::new().forward(&mut *device, &sinusoids);
        let mut sums: HbmTensor<f32, X, m![R, { Symbol<Stacked<Hidden>> }]> = HbmTensor::new();
        crate::Module::forward(&self.first, &mut *device, (), &mut (), (&first, &mut sums));
        let hidden = crate::Module::forward(&self.first, &mut *device, (), &mut (), (&sums,));
        let hidden = Silu.forward(&mut *device, &hidden);
        let second = Sides::<Hidden>::new().forward(&mut *device, &hidden);
        let mut sums: HbmTensor<f32, X, m![R, { Symbol<Stacked<Out>> }]> = HbmTensor::new();
        crate::Module::forward(
            &self.second,
            &mut *device,
            (),
            &mut (),
            (&second, &mut sums),
        );
        crate::Module::forward(&self.second, device, (), &mut (), (&sums,))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::index::packed::Packed;
    use crate::ops::normalization::tests::random;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;
    use std::iter;

    struct Fixture;

    axes![Text = 64, Audio = 64, Video = 128, Hidden = 5376];

    type Sequence = Packed<Text, Audio, Video>;

    axes![Encoded = 5120, Envelope = 128];

    axes![Velocity = 96];

    axes![Patch = 96];

    axes![Embedded = 2688, Steps = 64];

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
                stacked.extend(iter::repeat_n(zero, inputs));
            }
            stacked
        }
    }

    #[device(chip = 1)]
    fn biased(
        device: &mut Device,
        projection: &Biased<Encoded, Hidden>,
        input: &Rows<Envelope, Encoded>,
    ) -> Rows<Envelope, Hidden> {
        projection.forward(device, input)
    }

    /// Rows projected with an offset give `bf16(x . w + b)`, the FP64 sum rounded once, within FP32
    /// accumulation error; the rows at both ends of both tiles are checked whole.
    #[tokio::test(flavor = "current_thread")]
    async fn offsets_join_the_sum_before_rounding() {
        let _lease = DEVICE.lock().await;
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

    #[device(chip = 1)]
    fn precise(
        device: &mut Device,
        projection: &Precise<Hidden, Velocity>,
        input: &Rows<Sequence, Hidden>,
    ) -> HbmTensor<f32, m![1], m![Sequence, Velocity]> {
        projection.forward(device, input)
    }

    /// BF16 rows times FP32 coefficients plus FP32 offsets give the FP64 product within 2^-17 of
    /// its summed term magnitudes, though the contraction itself multiplies BF16.
    #[tokio::test(flavor = "current_thread")]
    async fn projects_with_fp32_coefficients() {
        let _lease = DEVICE.lock().await;
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
    fn embedded(
        device: &mut Device,
        embedding: &Embedding<Patch, Hidden>,
        input: &HbmTensor<f32, m![1], m![Sequence, Patch]>,
    ) -> Rows<Sequence, Hidden> {
        embedding.forward(device, input)
    }

    /// FP32 latent rows times FP32 coefficients plus FP32 offsets give the FP64 product rounded
    /// once to BF16, within 2^-17 of its summed term magnitudes before the rounding.
    #[tokio::test(flavor = "current_thread")]
    async fn embeds_fp32_rows() {
        let _lease = DEVICE.lock().await;
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
            stacked.extend(iter::repeat_n(zero, inputs));
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
    #[device(chip = 1)]
    fn time_embedded(
        device: &mut Device,
        embedding: &TimeEmbedding<Hidden, Embedded>,
        time: &HbmTensor<f32, m![1], m![Steps, 1 # 8]>,
    ) -> HbmTensor<f32, m![1], m![Steps, Embedded]> {
        embedding.forward(device, time)
    }
    /// The video and audio timesteps' embeddings match the FP64 definition, cosines first, to
    /// within 1e-4 of the embedding's RMS.
    #[tokio::test(flavor = "current_thread")]
    async fn embeds_two_timesteps() {
        let _lease = DEVICE.lock().await;
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
}
