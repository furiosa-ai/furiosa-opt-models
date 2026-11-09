//! Rotary position embedding: each head's channel pairs rotated by the angles of its row's
//! position.

use crate::Op;
use crate::load::Context;
use crate::load::Load;
use crate::load::LoadError;
use crate::modules::generic::collection::View;
use crate::ops::pointwise::cast::Convert;
use crate::tensor::mapping::Packet;
use core::borrow::Borrow;
use core::f32::consts::FRAC_PI_2;
use core::f32::consts::TAU;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;
use std::io;

use crate::ops::attention::full::heads::tiles;
use crate::ops::linear::Groups;

use crate::ops::attention::{RowGroups, TileGroups};
use crate::ops::vector::Zip;
use crate::tensor::{axis::Lanes, parallel::Chip};
use crate::tensor::{live::Live, mapping::Rows as Layout};

use crate::ops::pointwise::multiply::QueryScale;
use crate::tensor::axis;
use crate::tensor::mapping::BroadcastClusters as Cluster;

// The rotation pairs each position with the one a half-head away: `Half` is the half a value is
// read from, `OutputHalf` the half it is rotated into, `Row` its position within either.
axes![Half = 2, OutputHalf = 2];

#[derive(Clone, Debug)]
pub struct Row<Head>(PhantomData<Head>);

impl<Head: AxisName> AxisName for Row<Head> {
    const NAME: Ident = Ident::new("Row");
    const SIZE: usize = Head::SIZE / 2;
}

type Table<Slice, Count, Row, Chips = Chip> =
    DmTensor<bf16, Chips, Cluster, Slice, m![{ Count }, OutputHalf, Row]>;

type Coefficient<Slice, Count, Row> =
    VrfTensor<f32, Chip, Cluster, Slice, m![{ Count }, OutputHalf, Row]>;

/// Row groups one pass rotates; each group's two coefficient tables take 1 KiB of the shared
/// 8 KiB VRF.
type TileRows<Rows> = m![{ Groups<Rows> } % ({ Symbol<TileGroups<Rows, 32>> })];

/// The pass a row group belongs to.
type Tiles<Rows> = m![{ Groups<Rows> } / ({ Symbol<TileGroups<Rows, 32>> })];

/// Rotates every live row's query and key by its position's coefficients.
#[derive(DeviceSend)]
pub struct RotaryEmbedding<Query, Key, Rows>(PhantomData<(Query, Key, Rows)>);

impl<Query, Key, Rows> RotaryEmbedding<Query, Key, Rows> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// The query and key heads, the coefficient rows of their positions and the live rows.
pub type Input<'a, Slice, Query, Key, Rows, Head> = (
    &'a DmTensor<bf16, Chip, Cluster, Slice, m![{ RowGroups<Rows, Query> }, Head]>,
    &'a DmTensor<bf16, Chip, Cluster, Slice, m![{ RowGroups<Rows, Key> }, Head]>,
    &'a HbmTensor<bf16, Chip, m![Rows, Head]>,
    &'a HbmTensor<bf16, Chip, m![Rows, Head]>,
    Live<(), Rows>,
);

impl<'a, Slice: M, Query: AxisName, Key: AxisName, Rows: AxisName, Head: AxisName>
    crate::Op<Input<'a, Slice, Query, Key, Rows, Head>> for RotaryEmbedding<Query, Key, Rows>
{
    type Output = (
        DmTensor<bf16, Chip, Cluster, Slice, m![{ RowGroups<Rows, Query> }, Head]>,
        DmTensor<bf16, Chip, Cluster, Slice, m![{ RowGroups<Rows, Key> }, Head]>,
    );

    /// Every row group in shared passes when every group holds a live row; otherwise each live
    /// group alone, and a group past the live rows is not rotated.
    fn forward(
        &self,
        device: &mut Device,
        (query, key, low, high, live): Input<'a, Slice, Query, Key, Rows, Head>,
    ) -> Self::Output {
        let table = |device: &mut Device,
                     table: &HbmTensor<bf16, Chip, m![Rows, Head]>|
         -> Table<Slice, Groups<Rows>, Row<Head>> {
            // SAFETY: a coefficient row consists of its two consecutive output halves.
            unsafe {
                table
                    .view()
                    .reshape::<Chip, m![Rows, OutputHalf, { Symbol<Row<Head>> }]>()
            }
            .to_dm(&mut device.tdma)
        };
        let low = table(device, low);
        let high = table(device, high);
        let mut rotated_query: DmTensor<
            bf16,
            Chip,
            Cluster,
            Slice,
            m![{ Groups<Rows> }, Query, OutputHalf, { Symbol<Row<Head>> }],
        > = DmTensor::new();
        let mut rotated_key: DmTensor<
            bf16,
            Chip,
            Cluster,
            Slice,
            m![{ Groups<Rows> }, Key, OutputHalf, { Symbol<Row<Head>> }],
        > = DmTensor::new();
        // SAFETY: each head's two halves remain consecutive within the same row and slice.
        let query = unsafe {
            query
                .view()
                .reshape::<
                    Chip,
                    Cluster,
                    Slice,
                    m![{ Groups<Rows> }, Query, Half, { Symbol<Row<Head>> }],
                >()
        };
        let key = unsafe {
            key.view()
                .reshape::<
                    Chip,
                    Cluster,
                    Slice,
                    m![{ Groups<Rows> }, Key, Half, { Symbol<Row<Head>> }],
                >()
        };
        if live.count::<Lanes<Rows>>() == <Groups<Rows> as M>::SIZE {
            #[unroll]
            for tile in 0..<Tiles<Rows> as M>::SIZE {
                let low = Fetch::<TileRows<Rows>, Row<Head>>(PhantomData).forward(
                    device,
                    low.view().tile::<Tiles<Rows>, 1, m![
                        { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, OutputHalf,
                        { Symbol<Row<Head>> }
                    ]>(tile),
                );
                let high = Fetch::<TileRows<Rows>, Row<Head>>(PhantomData).forward(
                    device,
                    high.view().tile::<Tiles<Rows>, 1, m![
                        { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, OutputHalf,
                        { Symbol<Row<Head>> }
                    ]>(tile),
                );
                RotateRows::<
                    TileRows<Rows>,
                    m![Query],
                    Row<Head>,
                >(PhantomData).forward(
                    device, (
                    query.clone().tile::<Tiles<Rows>, 1, m![
                        { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, Query, Half,
                        { Symbol<Row<Head>> }
                    ]>(tile),
                    rotated_query.view_mut().tile::<Tiles<Rows>, 1, m![
                        { Tiles<Rows> } = 1 #{!} ({ Tiles<Rows> }), { TileRows<Rows> }, Query,
                        OutputHalf, { Symbol<Row<Head>> }
                    ]>(tile),
                    &low,
                    &high,
                ));
                RotateRows::<TileRows<Rows>, m![Key], Row<Head>>(PhantomData).forward(
                    device,
                    (
                        key.clone().tile::<Tiles<Rows>, 1, m![
                            { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, Key, Half,
                            { Symbol<Row<Head>> }
                        ]>(tile),
                        rotated_key.view_mut().tile::<Tiles<Rows>, 1, m![
                            { Tiles<Rows> } = 1 #{!} ({ Tiles<Rows> }), { TileRows<Rows> }, Key,
                            OutputHalf, { Symbol<Row<Head>> }
                        ]>(tile),
                        &low,
                        &high,
                    ),
                );
            }
        } else {
            for group in 0..live.count::<Lanes<Rows>>() {
                let low = Fetch::<m![1], Row<Head>>(PhantomData).forward(
                    device,
                    low.view().tile::<Groups<Rows>, 1, m![
                        { Layout<Rows>::Selected }, OutputHalf, { Symbol<Row<Head>> }
                    ]>(group),
                );
                let high = Fetch::<m![1], Row<Head>>(PhantomData).forward(
                    device,
                    high.view().tile::<Groups<Rows>, 1, m![
                        { Layout<Rows>::Selected }, OutputHalf, { Symbol<Row<Head>> }
                    ]>(group),
                );
                RotateRows::<m![1], m![Query], Row<Head>>(PhantomData).forward(
                    device,
                    (
                        query.clone().tile::<Groups<Rows>, 1, m![
                            { Layout<Rows>::Selected }, Query, Half, { Symbol<Row<Head>> }
                        ]>(group),
                        rotated_query.view_mut().tile::<Groups<Rows>, 1, m![
                            { Layout<Rows>::SelectedMut }, Query, OutputHalf, { Symbol<Row<Head>> }
                        ]>(group),
                        &low,
                        &high,
                    ),
                );
                RotateRows::<m![1], m![Key], Row<Head>>(PhantomData).forward(
                    device,
                    (
                        key.clone().tile::<Groups<Rows>, 1, m![
                            { Layout<Rows>::Selected }, Key, Half, { Symbol<Row<Head>> }
                        ]>(group),
                        rotated_key.view_mut().tile::<Groups<Rows>, 1, m![
                            { Layout<Rows>::SelectedMut }, Key, OutputHalf, { Symbol<Row<Head>> }
                        ]>(group),
                        &low,
                        &high,
                    ),
                );
            }
        }
        // SAFETY: output halves concatenate to Head within each unchanged row.
        unsafe { (rotated_query.reshape(), rotated_key.reshape()) }
    }
}

struct Fetch<Count, Row>(PhantomData<(Count, Row)>);

impl<Slice: M, Count: M, Element: M, Row: AxisName>
    crate::Op<DmTensorView<'_, bf16, Chip, Cluster, Slice, Element>> for Fetch<Count, Row>
{
    type Output = Coefficient<Slice, Count, Row>;

    fn forward(
        &self,
        device: &mut Device,
        table: DmTensorView<'_, bf16, Chip, Cluster, Slice, Element>,
    ) -> Coefficient<Slice, Count, Row> {
        device
            .sub
            .begin(table)
            .fetch::<m![{ Count }, OutputHalf, Row / 16], m![Row % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Count }, OutputHalf, Row / 8], m![Row % 8]>()
            .to_vrf()
    }
}

struct RotateRows<Count, Group, Row>(PhantomData<(Count, Group, Row)>);

impl<Slice: M, Count: M, Group: M, Row: AxisName, Input: M, Output: M>
    crate::Op<(
        DmTensorView<'_, bf16, Chip, Cluster, Slice, Input>,
        DmTensorViewMut<'_, bf16, Chip, Cluster, Slice, Output>,
        &Coefficient<Slice, Count, Row>,
        &Coefficient<Slice, Count, Row>,
    )> for RotateRows<Count, Group, Row>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, output, low, high): (
            DmTensorView<'_, bf16, Chip, Cluster, Slice, Input>,
            DmTensorViewMut<'_, bf16, Chip, Cluster, Slice, Output>,
            &Coefficient<Slice, Count, Row>,
            &Coefficient<Slice, Count, Row>,
        ),
    ) {
        device
            .main
            .begin_interleaved::<OutputHalf, _, _, _, _, _>(input.clone(), input)
            .fetch::<m![{ Count }, { Group }, OutputHalf, Half, Row / 16], m![Row % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Count }, { Group }, OutputHalf, Half, Row / 8], m![Row % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Half, m![{ Count }, { Group }, OutputHalf, Row / 8]>()
            .vector_narrow_split::<m![{ Count }, { Group }, OutputHalf, Row / 4], m![Row % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), low, ())
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), high)
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_widen_concat::<m![{ Count }, { Group }, OutputHalf, Row / 8], m![Row % 8]>()
            .vector_final()
            .cast::<bf16, m![Row % 8 # 16]>()
            .commit_trim::<m![Row % 8]>()
            .commit_view(output);
    }
}

impl<Head: AxisName> Rotation<Head> {
    /// Host coefficients with FP32 angles rounded to BF16.
    pub fn coefficients(position: usize, theta: f32) -> (Vec<bf16>, Vec<bf16>) {
        let half = Head::SIZE / 2;
        let angle = |index: usize| {
            let inverse = 1.0f32 / theta.powf(2.0 * (index % half) as f32 / Head::SIZE as f32);
            position as f32 * inverse
        };
        let row = |low: fn(f32) -> f32, high: fn(f32) -> f32| {
            (0..Head::SIZE)
                .map(|index| {
                    let table = if index < half { low } else { high };
                    bf16::from_f32(table(angle(index)))
                })
                .collect()
        };
        (row(f32::cos, f32::sin), row(|angle| -angle.sin(), f32::cos))
    }
}

type Scalars<Rows, Rotated, Chips = Chip> = HbmTensor<f32, Chips, m![Rows, Rotated]>;
type Parameters<Rotated, Chips = Chip> = HbmTensor<f32, Chips, m![Rotated]>;

type Angles<Rows, Rotated, Chips = Chip> =
    DmTensor<f32, Chips, Cluster, m![Rotated / 8 # 256], m![Rows, Rotated % 8]>;

type Constants<Rotated, Chips = Chip> =
    VrfTensor<f32, Chips, Cluster, m![Rotated / 8 # 256], m![Rotated % 8]>;

type CoefficientRows<Rows, Rotated, Chips = Chip> = HbmTensor<bf16, Chips, m![Rows, Rotated]>;

type CoefficientSlices<Rows, Rotated, Chips = Chip> =
    DmTensor<bf16, Chips, Cluster, m![Rotated / 8 # 256], m![Rows, Rotated % 8]>;

/// Produce each row's rotation from the frequency of each rotated coordinate and the row's token
/// position; each constant is held as `T`: the HBM tensor, or a borrow of it in a borrowed tree.
#[derive(DeviceSend)]
pub struct Coefficients<Rotated: AxisName, Chips: M = Chip, T = Parameters<Rotated, Chips>> {
    /// The first input half's coefficients.
    low: Component<T>,
    /// The second input half's coefficients.
    high: Component<T>,
    /// The amplitude of every coefficient, on each coordinate.
    amplitude: T,
    shape: core::marker::PhantomData<fn() -> (Rotated, Chips)>,
}

/// The constants of one half's coefficients: the sine of each angle times `sign` plus `phase`.
#[derive(DeviceSend)]
pub struct Component<T> {
    /// Phase shifts in radians.
    phase: T,
    /// Angle multipliers.
    sign: T,
}

impl<Rotated: AxisName, Chips: M, T> Coefficients<Rotated, Chips, T> {
    /// The constants the rotation reads: each half's phase and sign, low first, then the amplitude.
    pub fn constants(&self) -> [&T; 5] {
        let (low, high) = (&self.low, &self.high);
        [
            &low.phase,
            &low.sign,
            &high.phase,
            &high.sign,
            &self.amplitude,
        ]
    }
}

/// The constants of a rotation at the amplitude of `frequencies`, as no checkpoint stores them.
impl<'f, Rotated: AxisName, Chips: M, T> Load<&'f PairFrequencies<Rotated>>
    for Coefficients<Rotated, Chips, T>
where
    T: Load<HostTensor<f32, m![{ Chips }, { m![Rotated] }]>>,
{
    fn load(
        context: &mut Context<'_>,
        frequencies: &'f PairFrequencies<Rotated>,
    ) -> Result<Self, LoadError> {
        const { assert!(Rotated::SIZE > 0 && Rotated::SIZE.is_multiple_of(16)) };
        let halves = |first: f32, second: f32| -> Vec<f32> {
            let half = |i| if i < Rotated::SIZE / 2 { first } else { second };
            (0..Rotated::SIZE)
                .map(half)
                .collect::<Vec<_>>()
                .repeat(Chips::SIZE)
        };
        let constant = |cx: &mut Context<'_>, name: &str, values: Vec<f32>| {
            cx.field(name, name, |cx| T::load(cx, HostTensor::from_vec(values)))
        };
        let component = |cx: &mut Context<'_>, (phase, sign): (Vec<f32>, Vec<f32>)| {
            Ok(Component {
                phase: constant(cx, "phase", phase)?,
                sign: constant(cx, "sign", sign)?,
            })
        };
        let amplitude = frequencies.amplitude();
        let low = (halves(FRAC_PI_2, 0.0), halves(1.0, 1.0));
        let high = (halves(0.0, FRAC_PI_2), halves(-1.0, 1.0));
        Ok(Self {
            low: context.field("low", "low", |cx| component(cx, low))?,
            high: context.field("high", "high", |cx| component(cx, high))?,
            amplitude: constant(context, "amplitude", halves(amplitude, amplitude))?,
            shape: core::marker::PhantomData,
        })
    }
}

/// A tree's constants borrowed, or held as another tree holds them.
impl<'a, Rotated: AxisName, Chips: M, T: Load<&'a U>, U> Load<&'a Coefficients<Rotated, Chips, U>>
    for Coefficients<Rotated, Chips, T>
{
    fn load(
        context: &mut Context<'_>,
        from: &'a Coefficients<Rotated, Chips, U>,
    ) -> Result<Self, LoadError> {
        let component = |cx: &mut Context<'_>, from: &'a Component<U>| {
            Ok(Component {
                phase: cx.field("phase", "phase", |cx| T::load(cx, &from.phase))?,
                sign: cx.field("sign", "sign", |cx| T::load(cx, &from.sign))?,
            })
        };
        Ok(Self {
            low: context.field("low", "low", |cx| component(cx, &from.low))?,
            high: context.field("high", "high", |cx| component(cx, &from.high))?,
            amplitude: context
                .field("amplitude", "amplitude", |cx| T::load(cx, &from.amplitude))?,
            shape: core::marker::PhantomData,
        })
    }
}

impl<Rotated: AxisName, Chip: M, T> Coefficients<Rotated, Chip, T> {
    fn angle<Rows: AxisName>(
        &self,
        device: &mut Device,
        (frequency, position): (&Scalars<Rows, Rotated, Chip>, &Scalars<Rows, Rotated, Chip>),
    ) -> Angles<Rows, Rotated, Chip> {
        let frequency: Angles<Rows, Rotated, Chip> = frequency.to_dm(&mut device.tdma);
        let position: Angles<Rows, Rotated, Chip> = position.to_dm(&mut device.tdma);
        let position: VrfTensor<f32, Chip, Cluster, m![Rotated / 8 # 256], m![Rows, Rotated % 8]> =
            device
                .sub
                .begin(position.view())
                .fetch::<m![Rows], m![Rotated % 8]>()
                .collect::<m![Rows], m![Rotated % 8]>()
                .to_vrf();
        device
            .main
            .begin(frequency.view())
            .fetch::<m![Rows], m![Rotated % 8]>()
            .collect::<m![Rows], m![Rotated % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows, Rotated % 8 / 4], m![Rotated % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &position)
            .vector_widen_concat::<m![Rows], m![Rotated % 8]>()
            .vector_final()
            .commit_trim::<m![Rotated % 8]>()
            .commit()
    }

    /// One coefficient component of each row: `amplitude` times the sine of its angle times
    /// `sign` plus `phase`.
    fn component<Rows: AxisName>(
        device: &mut Device,
        angle: &Angles<Rows, Rotated, Chip>,
        Component { phase, sign }: &Component<T>,
        amplitude: &Parameters<Rotated, Chip>,
    ) -> CoefficientSlices<Rows, Rotated, Chip>
    where
        T: Borrow<Parameters<Rotated, Chip>>,
    {
        let (phase, sign) = (phase.borrow(), sign.borrow());
        let constants =
            |device: &mut Device, input: &Parameters<Rotated, Chip>| -> Constants<Rotated, Chip> {
                let input: DmTensor<f32, Chip, Cluster, m![Rotated / 8 # 256], m![Rotated % 8]> =
                    input.to_dm(&mut device.tdma);
                device
                    .sub
                    .begin(input.view())
                    .fetch::<m![1], m![Rotated % 8]>()
                    .collect::<m![1], m![Rotated % 8]>()
                    .to_vrf()
            };
        let phase = constants(device, phase);
        let sign = constants(device, sign);
        let amplitude = constants(device, amplitude);
        let shifted: Angles<Rows, Rotated, Chip> = device
            .main
            .begin(angle.view())
            .fetch::<m![Rows], m![Rotated % 8]>()
            .collect::<m![Rows], m![Rotated % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows, Rotated % 8 / 4], m![Rotated % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sign)
            .vector_widen_concat::<m![Rows], m![Rotated % 8]>()
            .vector_clip(ClipBinaryOpF32::Add, &phase)
            .vector_final()
            .commit_trim::<m![Rotated % 8]>()
            .commit();
        // Hardware trigonometric instructions require a reduced angle within one turn.
        let turns: DmTensor<i32, Chip, Cluster, m![Rotated / 8 # 256], m![Rows, Rotated % 8]> =
            device
                .main
                .begin(shifted.view())
                .fetch::<m![Rows], m![Rotated % 8]>()
                .collect::<m![Rows], m![Rotated % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Rows, Rotated % 8 / 4], m![Rotated % 4]>()
                .vector_fp_div(TAU)
                .vector_widen_concat::<m![Rows], m![Rotated % 8]>()
                .vector_fp_to_fxp(31)
                .vector_final()
                .commit_trim::<m![Rotated % 8]>()
                .commit();
        let turns: VrfTensor<f32, Chip, Cluster, m![Rotated / 8 # 256], m![Rows, Rotated % 8]> =
            device
                .sub
                .begin(turns.view())
                .fetch::<m![Rows], m![Rotated % 8]>()
                .collect::<m![Rows], m![Rotated % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp_to_fp(31)
                .vector_narrow_split::<m![Rows, Rotated % 8 / 4], m![Rotated % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), TAU)
                .vector_widen_concat::<m![Rows], m![Rotated % 8]>()
                .vector_final()
                .to_vrf();
        device
            .main
            .begin(shifted.view())
            .fetch::<m![Rows], m![Rotated % 8]>()
            .collect::<m![Rows], m![Rotated % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows, Rotated % 8 / 4], m![Rotated % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &turns)
            .vector_fp_unary(FpUnaryOp::Sin)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &amplitude)
            .vector_widen_concat::<m![Rows], m![Rotated % 8]>()
            .vector_final()
            .cast::<bf16, m![Rotated % 8 # 16]>()
            .commit_trim::<m![Rotated % 8]>()
            .commit()
    }
}

impl<Rows: AxisName, Rotated: AxisName, Chip: M, T>
    crate::Op<(&Scalars<Rows, Rotated, Chip>, &Scalars<Rows, Rotated, Chip>)>
    for Coefficients<Rotated, Chip, T>
where
    T: Borrow<Parameters<Rotated, Chip>>,
{
    type Output = Rotation<Rotated, Chip, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        scalars: (&Scalars<Rows, Rotated, Chip>, &Scalars<Rows, Rotated, Chip>),
    ) -> Self::Output {
        let angle = self.angle(device, scalars);
        let amplitude = self.amplitude.borrow();
        let low = Self::component(device, &angle, &self.low, amplitude).to_hbm(&mut device.tdma);
        let high = Self::component(device, &angle, &self.high, amplitude).to_hbm(&mut device.tdma);
        Rotation { low, high }
    }
}

/// The rotation of the live rows only; a later row's rotation is unspecified.
impl<Rows: AxisName, Rotated: AxisName, Chip: M, T>
    crate::Op<(
        &Scalars<Rows, Rotated, Chip>,
        &Scalars<Rows, Rotated, Chip>,
        Live<(), Rows>,
    )> for Coefficients<Rotated, Chip, T>
where
    T: Borrow<Parameters<Rotated, Chip>>,
{
    type Output = Rotation<Rotated, Chip, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        (frequency, position, live): (
            &Scalars<Rows, Rotated, Chip>,
            &Scalars<Rows, Rotated, Chip>,
            Live<(), Rows>,
        ),
    ) -> Self::Output {
        // Stores only the live rows; a later row is left unwritten.
        let store = |device: &mut Device,
                     table: &CoefficientSlices<Rows, Rotated, Chip>|
         -> CoefficientRows<Rows, Rotated, Chip> {
            let mut output: CoefficientRows<Rows, Rotated, Chip> = HbmTensor::new();
            live.cover::<Lanes<Rows>, _>(
                &mut (device, &mut output),
                (
                    |(device, output)| {
                        table
                            .view()
                            .to_hbm_view(&mut device.tdma, output.view_mut());
                    },
                    |(device, output), packet| {
                        table
                        .view()
                        .tile::<Layout<Rows>::Groups, 1, Layout<Rows>::Elements<m![Rotated % 8]>>(
                            packet,
                        )
                        .to_hbm_view(
                            &mut device.tdma,
                            output
                                .view_mut()
                                .tile::<
                                    Layout<Rows>::Groups,
                                    1,
                                    Layout<Rows>::GroupMut<m![Rotated]>,
                                >(
                                    packet,
                                ),
                        );
                    },
                    |(device, output), row| {
                        table
                            .view()
                            .tile::<m![Rows], 1, m![Rows = 1 # Rows, Rotated % 8]>(row)
                            .to_hbm_view(
                                &mut device.tdma,
                                output
                                    .view_mut()
                                    .tile::<m![Rows], 1, m![Rows = 1 #{!} Rows, Rotated]>(row),
                            );
                    },
                ),
            );
            output
        };
        let angle = self.angle(device, (frequency, position));
        let amplitude = self.amplitude.borrow();
        let low = Self::component(device, &angle, &self.low, amplitude);
        let low = store(device, &low);
        let high = Self::component(device, &angle, &self.high, amplitude);
        let high = store(device, &high);
        Rotation { low, high }
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use super::{Chip, RotaryEmbedding};
    use crate::Op;
    use crate::ops::attention::{
        Heads,
        tests::{Head, KvHeads},
    };
    use crate::tensor::axis::{Rows, Single};
    use crate::tensor::live::Live;
    use crate::tests::DEVICE;
    use std::iter;

    axes![Groups = 2];

    type QueryRows<Count> = HbmTensor<bf16, Chip, m![Count, KvHeads, Groups, Head]>;
    type KeyRows<Count> = HbmTensor<bf16, Chip, m![Count, KvHeads, Head]>;

    #[device(chip = 1)]
    pub(super) fn positioned<Rows: AxisName, Rotated: AxisName>(
        device: &mut Device,
        coefficients: &super::Coefficients<Rotated>,
        frequency: &super::Scalars<Rows, Rotated>,
        position: &super::Scalars<Rows, Rotated>,
    ) -> (
        super::CoefficientRows<Rows, Rotated>,
        super::CoefficientRows<Rows, Rotated>,
    ) {
        let rotation = coefficients.forward(device, (frequency, position));
        (rotation.low, rotation.high)
    }

    pub(super) async fn coefficients<Rotated: AxisName>(
        device: &mut Device,
        amplitude: f32,
    ) -> super::Coefficients<Rotated> {
        let pairs = vec![1.0; Rotated::SIZE / 2];
        let frequencies = super::PairFrequencies::<Rotated>::new(&pairs, amplitude.into()).unwrap();
        let mut context = crate::load::Context::new(device);
        crate::load::Load::load(&mut context, &frequencies).unwrap()
    }

    async fn check_coefficients<Head: AxisName, Rotated: AxisName>(
        device: &mut Device,
        amplitude: f32,
    ) {
        use crate::tensor::axis::Eight;
        let coefficients = coefficients::<Rotated>(device, amplitude).await;
        let positions = [
            (0, 10_000.0),
            (1, 10_000.0),
            (7, 1_000_000.0),
            (255, 1_000_000.0),
            (4095, 10_000.0),
            (8191, 1_000_000.0),
            (40_959, 10_000.0),
            (40_959, 1_000_000.0),
        ];
        let frequency: super::Scalars<Eight, Rotated> =
            HostTensor::<f32, m![Eight, Rotated]>::from_vec(positions.iter().flat_map(
                |&(_, theta)| {
                    super::PairFrequencies::<Rotated>::unscaled::<Head>(theta)
                        .unwrap()
                        .coordinates()
                },
            ))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let position: super::Scalars<Eight, Rotated> =
            HostTensor::<f32, m![Eight, Rotated]>::from_vec(
                positions
                    .iter()
                    .flat_map(|&(position, _)| iter::repeat_n(position as f32, Rotated::SIZE)),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let (low, high) = launch(
            positioned,
            (&mut *device, &coefficients, &frequency, &position),
        )
        .await
        .unwrap();
        let low = low
            .to_host::<m![Eight, Rotated]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let high = high
            .to_host::<m![Eight, Rotated]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (row, (position, theta)) in positions.into_iter().enumerate() {
            let angle = |index: usize| {
                position as f32
                    / theta.powf(2.0 * (index % (Rotated::SIZE / 2)) as f32 / Head::SIZE as f32)
            };
            let expected_low = (0..Rotated::SIZE)
                .map(|index| {
                    bf16::from_f32(if index < Rotated::SIZE / 2 {
                        angle(index).cos() * amplitude
                    } else {
                        angle(index).sin() * amplitude
                    })
                })
                .collect::<Vec<_>>();
            let expected_high = (0..Rotated::SIZE)
                .map(|index| {
                    bf16::from_f32(if index < Rotated::SIZE / 2 {
                        -angle(index).sin() * amplitude
                    } else {
                        angle(index).cos() * amplitude
                    })
                })
                .collect::<Vec<_>>();
            for (actual, expected) in [(&low, expected_low), (&high, expected_high)] {
                for (lane, expected) in expected.into_iter().enumerate() {
                    let actual = actual[row * Rotated::SIZE + lane].to_f32();
                    let expected = expected.to_f32();
                    assert!(
                        (actual - expected).abs() <= 0.01 * amplitude,
                        "position {position}, theta {theta}, lane {lane}: {actual} vs {expected}"
                    );
                }
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn matches_position_coefficients() {
        axes![Full = 256, Partial = 512];
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(positioned.topology()).unwrap();
        check_coefficients::<Head, Head>(&mut device, 1.0).await;
        check_coefficients::<Full, Full>(&mut device, 1.0).await;
        check_coefficients::<Partial, Head>(&mut device, 1.0).await;
        // YaRN's amplitude at factor 4.
        check_coefficients::<Head, Head>(&mut device, 0.1 * 4f32.ln() + 1.0).await;
    }

    #[device(chip = 1)]
    fn forward<Count: AxisName>(
        device: &mut Device,
        query: &QueryRows<Count>,
        key: &KeyRows<Count>,
        low: &HbmTensor<bf16, Chip, m![Count, Head]>,
        high: &HbmTensor<bf16, Chip, m![Count, Head]>,
    ) -> (QueryRows<Count>, KeyRows<Count>) {
        let query: Heads<Count, KvHeads, Groups, Head> = query.to_dm(&mut device.tdma);
        let key: Heads<Count, KvHeads, Single, Head> = key.to_dm(&mut device.tdma);
        let (query, key) = RotaryEmbedding::<Groups, Single, Count>::new()
            .forward(&mut *device, (&query, &key, low, high, Live::all()));
        (query.to_hbm(&mut device.tdma), key.to_hbm(&mut device.tdma))
    }

    async fn check_rotation<Count: AxisName>(device: &mut Device) {
        let low = (0..Count::SIZE * Head::SIZE)
            .map(|at| bf16::from_f32(((at * 3 + at / Head::SIZE) % 17) as f32 / 16.0))
            .collect::<Vec<_>>();
        let high = (0..Count::SIZE * Head::SIZE)
            .map(|at| bf16::from_f32(((at * 5 + at / Head::SIZE) % 19) as f32 / 16.0 - 0.5))
            .collect::<Vec<_>>();
        let values = |groups: usize| {
            (0..Count::SIZE * KvHeads::SIZE * groups * Head::SIZE)
                .map(|at| bf16::from_f32(((at * 7 + at / Head::SIZE * 5) % 37) as f32 / 16.0 - 1.0))
                .collect::<Vec<_>>()
        };
        let query = values(Groups::SIZE);
        let key = values(1);
        let query_hbm: QueryRows<Count> =
            HostTensor::<bf16, m![Count, KvHeads, Groups, Head]>::from_vec(query.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let key_hbm: KeyRows<Count> =
            HostTensor::<bf16, m![Count, KvHeads, Head]>::from_vec(key.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let low_hbm: super::CoefficientRows<Count, Head> =
            HostTensor::<bf16, m![Count, Head]>::from_vec(low.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let high_hbm: super::CoefficientRows<Count, Head> =
            HostTensor::<bf16, m![Count, Head]>::from_vec(high.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let (query_output, key_output) = launch(
            forward,
            (&mut *device, &query_hbm, &key_hbm, &low_hbm, &high_hbm),
        )
        .await
        .unwrap();
        let query_output = query_output
            .to_host::<m![Count, KvHeads, Groups, Head]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let key_output = key_output
            .to_host::<m![Count, KvHeads, Head]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (groups, input, output) in [
            (Groups::SIZE, &query, &query_output),
            (1, &key, &key_output),
        ] {
            for (at, found) in output.iter().enumerate() {
                let row = at / (KvHeads::SIZE * groups * Head::SIZE);
                let channel = at % Head::SIZE;
                let start = at / Head::SIZE * Head::SIZE;
                let half = Head::SIZE / 2;
                let expected = input[start + channel % half].to_f32()
                    * low[row * Head::SIZE + channel].to_f32()
                    + input[start + half + channel % half].to_f32()
                        * high[row * Head::SIZE + channel].to_f32();
                assert_eq!(
                    found.to_f32(),
                    bf16::from_f32(expected).to_f32(),
                    "groups {groups}, row {row}, channel {channel}",
                );
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn matches_rotation() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(forward.topology()).unwrap();
        check_rotation::<Single>(&mut device).await;
        check_rotation::<Rows<32>>(&mut device).await;
        check_rotation::<Rows<64>>(&mut device).await;
    }

    #[device(chip = 1)]
    fn rotate<D: AxisName, R: AxisName>(
        device: &mut Device,
        rotary: &super::Rotary<D, R>,
        input: &HbmTensor<bf16, Chip, m![Groups, D]>,
        rotation: &super::Rotation<R>,
    ) -> HbmTensor<bf16, Chip, m![Groups, D]> {
        let mut copied = HbmTensor::new();
        input
            .view()
            .to_hbm_view(&mut device.tdma, copied.view_mut());
        rotary.rotate(device, copied, rotation)
    }

    async fn check_products<D: AxisName, R: AxisName>(device: &mut Device) {
        let rotary = super::Rotary::<D, R>::new();
        let input = (0..Groups::SIZE * D::SIZE)
            .map(|at| bf16::from_f32((at * 7 % 137) as f32 / 64.0 - 1.0))
            .collect::<Vec<_>>();
        let low = (0..R::SIZE)
            .map(|i| bf16::from_f32(if i < R::SIZE / 2 { 0.8 } else { 0.6 }))
            .collect::<Vec<_>>();
        let high = (0..R::SIZE)
            .map(|i| bf16::from_f32(if i < R::SIZE / 2 { -0.6 } else { 0.8 }))
            .collect::<Vec<_>>();
        let values = HostTensor::<bf16, m![Groups, D]>::from_vec(input.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let rotation = super::Rotation {
            low: HostTensor::<bf16, m![Single, R]>::from_vec(low.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            high: HostTensor::<bf16, m![Single, R]>::from_vec(high.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        };
        let actual = launch(rotate, (&mut *device, &rotary, &values, &rotation))
            .await
            .unwrap()
            .to_host::<m![Groups, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, actual) in actual.into_iter().enumerate() {
            let channel = at % D::SIZE;
            let pair = channel % (D::SIZE / 2);
            let expected = if pair < R::SIZE / 2 {
                let start = at / D::SIZE * D::SIZE;
                let coefficient = pair + channel / (D::SIZE / 2) * (R::SIZE / 2);
                let a = bf16::from_f32(input[start + pair].to_f32() * low[coefficient].to_f32());
                let b = bf16::from_f32(
                    input[start + D::SIZE / 2 + pair].to_f32() * high[coefficient].to_f32(),
                );
                bf16::from_f32(a.to_f32() + b.to_f32())
            } else {
                input[at]
            };
            assert_eq!(actual, expected, "channel {at}");
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rounds_products_before_adding() {
        axes![Full = 256, Partial = 512];
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(rotate.topology()).unwrap();
        check_products::<Full, Full>(&mut device).await;
        check_products::<Partial, Head>(&mut device).await;
    }
}

#[derive(Clone, Debug)]
struct PairWidth<Rotated>(PhantomData<Rotated>);

impl<Rotated: AxisName> AxisName for PairWidth<Rotated> {
    const NAME: Ident = Ident::new("PairWidth");
    const SIZE: usize = Rotated::SIZE / 2;
}

/// Whole heads within each rotary DM slice.
type HeadTile<Heads, Head, D = bf16, Chips = Chip> =
    DmTensor<D, Chips, Cluster, m![{Heads} # 256], m![Head]>;

/// Rotate the first Rotated/2 pairs across the two Head/2 halves by each row's rotation; `D` is
/// the element type of the rotated key output.
#[derive(DeviceSend, Load)]
pub struct Rotary<Head: AxisName, Rotated: AxisName = Head, D: Scalar = bf16, Chips: M = Chip>(
    PhantomData<(Head, Rotated, D, Chips)>,
);

impl<Head: AxisName, Rotated: AxisName, D: Scalar, Chips: M> Rotary<Head, Rotated, D, Chips> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    /// Dimension permutation that places each rotated pair across the two head halves.
    pub const fn paired(dim: usize) -> usize {
        let (head, rotated) = (Head::SIZE, Rotated::SIZE);
        let (half, pair, rest) = (head / 2, rotated / 2, (head - rotated) / 2);
        if dim < pair {
            dim
        } else if dim < rotated {
            dim - pair + half
        } else if dim < rotated + rest {
            dim - rotated + pair
        } else {
            dim - rotated - rest + half + pair
        }
    }
}

impl<'q, 'k, Head: AxisName, Width: AxisName, R: AxisName, Chips: M>
    crate::Op<(
        (
            &'q HbmTensor<bf16, Chips, m![R, Width]>,
            &'k HbmTensor<bf16, Chips, m![R, Width]>,
        ),
        &Rotation<Head, Chips, R>,
    )> for Rotary<Head, Head, bf16, Chips>
{
    type Output = (
        HbmTensor<bf16, Chips, m![R, Width]>,
        HbmTensor<bf16, Chips, m![R, Width]>,
    );

    fn forward(
        &self,
        device: &mut Device,
        ((query, key), rotation): (
            (
                &'q HbmTensor<bf16, Chips, m![R, Width]>,
                &'k HbmTensor<bf16, Chips, m![R, Width]>,
            ),
            &Rotation<Head, Chips, R>,
        ),
    ) -> Self::Output {
        (
            Heads::<Width, R, Head, Chips, _>(PhantomData).forward(&mut *device, (query, rotation)),
            Heads::<Width, R, Head, Chips, _>(PhantomData).forward(device, (key, rotation)),
        )
    }
}

impl<'q, 'k, Head: AxisName, Width: AxisName, R: AxisName, Chips: M>
    crate::Op<(
        (
            &'q tiles::Major<R, Width, Head, Chips>,
            &'k tiles::Major<R, Width, Head, Chips>,
        ),
        &Rotation<Head, Chips, R>,
    )> for Rotary<Head, Head, bf16, Chips>
{
    type Output = (
        tiles::Major<R, Width, Head, Chips>,
        tiles::Major<R, Width, Head, Chips>,
    );

    fn forward(
        &self,
        device: &mut Device,
        ((query, key), rotation): (
            (
                &'q tiles::Major<R, Width, Head, Chips>,
                &'k tiles::Major<R, Width, Head, Chips>,
            ),
            &Rotation<Head, Chips, R>,
        ),
    ) -> Self::Output {
        (
            Heads::<Width, R, Head, Chips, _>(PhantomData).forward(&mut *device, (query, rotation)),
            Heads::<Width, R, Head, Chips, _>(PhantomData).forward(device, (key, rotation)),
        )
    }
}

struct Heads<Width, R, Head, Chips, Q>(PhantomData<(Width, R, Head, Chips, Q)>);

impl<Head: AxisName, Chips: M, Width: AxisName, R: AxisName, Q: M, L>
    crate::Op<(&L, &Rotation<Head, Chips, R>)> for Heads<Width, R, Head, Chips, Q>
where
    L: tiles::Layout<R, Width, Head, Chips, Order = Q>,
{
    type Output = L;

    fn forward(
        &self,
        device: &mut Device,
        (input, rotation): (&L, &Rotation<Head, Chips, R>),
    ) -> L {
        let mut output = L::new();
        tiles::Strips::<R, Width, Head, _, Chips, Q>(PhantomData, Rotate { rotation })
            .forward(device, (input, &mut output));
        output
    }
}

impl<Head: AxisName, Chips: M, Width: AxisName, R: AxisName, Q: M, L>
    crate::Op<((&L, &L), &Rotation<Head, Chips, R>)> for Heads<Width, R, Head, Chips, Q>
where
    L: tiles::Layout<R, Width, Head, Chips, Order = Q>,
{
    type Output = (L, L);

    fn forward(
        &self,
        device: &mut Device,
        ((query, key), rotation): ((&L, &L), &Rotation<Head, Chips, R>),
    ) -> (L, L) {
        let (mut rotated_query, mut rotated_key) = (L::new(), L::new());
        tiles::Strips::<R, Width, Head, _, Chips, Q>(PhantomData, Rotate { rotation }).forward(
            device,
            ((query, key), (&mut rotated_query, &mut rotated_key)),
        );
        (rotated_query, rotated_key)
    }
}

impl<Head: AxisName, Width: AxisName, R: AxisName, Chips: M>
    crate::Op<(
        &(
            &HbmTensor<bf16, Chips, m![R, Width]>,
            &HbmTensor<bf16, Chips, m![R, Width]>,
        ),
        &Rotation<Head, Chips, R>,
    )> for Rotary<Head, Head, bf16, Chips>
{
    type Output = (
        HbmTensor<bf16, Chips, m![R, Width]>,
        HbmTensor<bf16, Chips, m![R, Width]>,
    );

    fn forward(
        &self,
        device: &mut Device,
        (rows, rotation): (
            &(
                &HbmTensor<bf16, Chips, m![R, Width]>,
                &HbmTensor<bf16, Chips, m![R, Width]>,
            ),
            &Rotation<Head, Chips, R>,
        ),
    ) -> Self::Output {
        Heads::<Width, R, Head, Chips, _>(PhantomData).forward(device, (*rows, rotation))
    }
}

impl<Head: AxisName, Width: AxisName, R: AxisName, Chips: M>
    crate::Op<(
        &(
            &tiles::Major<R, Width, Head, Chips>,
            &tiles::Major<R, Width, Head, Chips>,
        ),
        &Rotation<Head, Chips, R>,
    )> for Rotary<Head, Head, bf16, Chips>
{
    type Output = (
        tiles::Major<R, Width, Head, Chips>,
        tiles::Major<R, Width, Head, Chips>,
    );

    fn forward(
        &self,
        device: &mut Device,
        (rows, rotation): (
            &(
                &tiles::Major<R, Width, Head, Chips>,
                &tiles::Major<R, Width, Head, Chips>,
            ),
            &Rotation<Head, Chips, R>,
        ),
    ) -> Self::Output {
        Heads::<Width, R, Head, Chips, _>(PhantomData).forward(device, (*rows, rotation))
    }
}

struct Rotate<'r, Head: AxisName, R: AxisName, Chips: M> {
    rotation: &'r Rotation<Head, Chips, R>,
}

type Lines<Width, Head> = m![{ tiles::T<Width, Head> } % ({ tiles::D<Width, Head> })];
type TileCoefficients<Width, Head, Chips> = VrfTensor<
    f32,
    Chips,
    tiles::Half<Width, Head>,
    tiles::Slices<Width, Head>,
    m![{ Lines<Width, Head> }, OutputHalf, { Symbol<Row<Head>> }],
>;

struct FetchTiles<Width, Head, R, Chips>(PhantomData<(Width, Head, R, Chips)>);

impl<Head: AxisName, R: AxisName, Chips: M, Width: AxisName>
    crate::Op<(&Rotation<Head, Chips, R>, usize)> for FetchTiles<Width, Head, R, Chips>
{
    type Output = (
        TileCoefficients<Width, Head, Chips>,
        TileCoefficients<Width, Head, Chips>,
    );

    fn forward(
        &self,
        device: &mut Device,
        (rotation, at): (&Rotation<Head, Chips, R>, usize),
    ) -> (
        TileCoefficients<Width, Head, Chips>,
        TileCoefficients<Width, Head, Chips>,
    ) {
        let rows = |device: &mut Device, table: &CoefficientRows<R, Head, Chips>| {
            // SAFETY: the tiles split rows in order and each coefficient row is its two halves.
            let rows = unsafe {
                table.view().reshape::<Chips, m![
                    { tiles::S<R, Width, Head> },
                    { tiles::T<Width, Head> },
                    OutputHalf,
                    { Symbol<Row<Head>> }
                ]>()
            };
            let in_dm: DmTensor<
                bf16,
                Chips,
                tiles::Half<Width, Head>,
                tiles::Slices<Width, Head>,
                m![{ Lines<Width, Head> }, OutputHalf, { Symbol<Row<Head>> }],
            > = rows
                .tile::<m![{ tiles::S<R, Width, Head> }], 1, m![
                    { tiles::S<R, Width, Head> } = 1 # ({ tiles::S<R, Width, Head> }),
                    { tiles::T<Width, Head> },
                    OutputHalf,
                    { Symbol<Row<Head>> }
                ]>(at)
                .to_dm(&mut device.tdma);
            let operand: TileCoefficients<Width, Head, Chips> = device.sub.begin(in_dm.view())
                .fetch::<
                    m![{ Lines<Width, Head> }, OutputHalf, { Symbol<Row<Head>> } / 16],
                    m![{ Symbol<Row<Head>> } % 16],
                >()
                .fetch_cast::<f32>()
                .collect::<
                    m![{ Lines<Width, Head> }, OutputHalf, { Symbol<Row<Head>> } / 8],
                    m![{ Symbol<Row<Head>> } % 8],
                >()
                .to_vrf();
            operand
        };
        (rows(device, &rotation.low), rows(device, &rotation.high))
    }
}

struct RotateTile<Head>(PhantomData<Head>);

impl<Head: AxisName, Chips: M, Width: AxisName, Q: M>
    crate::Op<(
        &tiles::Tile<bf16, Width, Head, Chips, Q>,
        &(
            TileCoefficients<Width, Head, Chips>,
            TileCoefficients<Width, Head, Chips>,
        ),
        &mut tiles::Tile<bf16, Width, Head, Chips, Q>,
    )> for RotateTile<Head>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (rows, (low, high), output): (
            &tiles::Tile<bf16, Width, Head, Chips, Q>,
            &(
                TileCoefficients<Width, Head, Chips>,
                TileCoefficients<Width, Head, Chips>,
            ),
            &mut tiles::Tile<bf16, Width, Head, Chips, Q>,
        ),
    ) {
        // SAFETY: a head is its two halves in order.
        let halves = unsafe {
            rows.view().reshape::<
                Chips,
                tiles::Half<Width, Head>,
                tiles::Slices<Width, Head>,
                m![{ Q }, Half, { Symbol<Row<Head>> }],
            >()
        };
        // SAFETY: the two output halves are the head in order.
        let rotated = unsafe {
            output.view_mut().reshape::<
                Chips,
                tiles::Half<Width, Head>,
                tiles::Slices<Width, Head>,
                m![{ Q }, OutputHalf, { Symbol<Row<Head>> }],
            >()
        };
        device.main.begin_interleaved::<OutputHalf, _, _, _, _, _>(halves.clone(), halves)
            .fetch::<
                m![{ Q }, OutputHalf, Half, { Symbol<Row<Head>> } / 16],
                m![{ Symbol<Row<Head>> } % 16],
            >()
            .fetch_cast::<f32>()
            .collect::<
                m![{ Q }, OutputHalf, Half, { Symbol<Row<Head>> } / 8],
                m![{ Symbol<Row<Head>> } % 8],
            >()
            .vector_init()
            .vector_intra_slice_unzip::<Half, m![{ Q }, OutputHalf, { Symbol<Row<Head>> } / 8]>()
            .vector_narrow_split::<
                m![{ Q }, OutputHalf, { Symbol<Row<Head>> } / 4],
                m![{ Symbol<Row<Head>> } % 4],
            >()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), low, ())
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), high)
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_widen_concat::<
                m![{ Q }, OutputHalf, { Symbol<Row<Head>> } / 8],
                m![{ Symbol<Row<Head>> } % 8],
            >()
            .vector_final()
            .cast::<bf16, m![{ Symbol<Row<Head>> } % 8 # 16]>()
            .commit_trim::<m![{ Symbol<Row<Head>> } % 8]>()
            .commit_view(rotated);
    }
}

impl<Head: AxisName, Width: AxisName, R: AxisName, Chips: M, Q: M>
    tiles::Pass<Width, Head, Chips, Q> for Rotate<'_, Head, R, Chips>
{
    fn apply(
        &self,
        device: &mut Device,
        rows: &tiles::Tile<bf16, Width, Head, Chips, Q>,
        at: usize,
        output: &mut tiles::Tile<bf16, Width, Head, Chips, Q>,
    ) {
        let coefficients =
            FetchTiles::<Width, Head, R, Chips>(PhantomData).forward(device, (self.rotation, at));
        RotateTile::<Head>(PhantomData).forward(device, (rows, &coefficients, output));
    }
}

impl<Head: AxisName, Width: AxisName, R: AxisName, Chips: M, Q: M>
    tiles::Both<Width, Head, Chips, Q> for Rotate<'_, Head, R, Chips>
{
    fn apply(
        &self,
        device: &mut Device,
        (first, second): (
            &tiles::Tile<bf16, Width, Head, Chips, Q>,
            &tiles::Tile<bf16, Width, Head, Chips, Q>,
        ),
        at: usize,
        (first_out, second_out): (
            &mut tiles::Tile<bf16, Width, Head, Chips, Q>,
            &mut tiles::Tile<bf16, Width, Head, Chips, Q>,
        ),
    ) {
        let coefficients =
            FetchTiles::<Width, Head, R, Chips>(PhantomData).forward(device, (self.rotation, at));
        RotateTile::<Head>(PhantomData).forward(device, (first, &coefficients, first_out));
        RotateTile::<Head>(PhantomData).forward(device, (second, &coefficients, second_out));
    }
}

impl<Head: AxisName, Rotated: AxisName, D: Scalar, Chips: M> Default
    for Rotary<Head, Rotated, D, Chips>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Head: AxisName, Rotated: AxisName, D: Scalar, Chips: M, const N: usize> View<N>
    for Rotary<Head, Rotated, D, Chips>
{
    type Storage = ();
    type Layer = Self;

    fn view(_: &(), _: usize) -> Self {
        Self::new()
    }

    fn stack(_: [Self; N], _: &mut Context<'_>) {}
}

/// Each of `Rows` rows' rotation over the rotated dimensions: `[cos | sin]` for the first input
/// half and `[-sin | cos]` for the second.
#[derive(DeviceSend)]
pub struct Rotation<Rotated: AxisName, Chips: M = Chip, Rows: AxisName = axis::Single> {
    pub low: CoefficientRows<Rows, Rotated, Chips>,
    pub high: CoefficientRows<Rows, Rotated, Chips>,
}

impl<Head: AxisName, Rotated: AxisName, Q: M, K: M, D: MaterializableScalar, Chip: M>
    crate::Op<(
        (
            HbmTensor<bf16, Chip, m![{ Q }, Head]>,
            HbmTensor<bf16, Chip, m![{ K }, Head]>,
        ),
        &Rotation<Rotated, Chip>,
    )> for Rotary<Head, Rotated, D, Chip>
where
    f32: CastEngineCast<D>,
    Convert<D>: Op<HeadTile<K, Head, bf16, Chip>, Output = HeadTile<K, Head, D, Chip>>,
{
    type Output = (
        HbmTensor<bf16, Chip, m![{ Q }, Head]>,
        HbmTensor<D, Chip, m![{ K }, Head]>,
    );

    fn forward(
        &self,
        device: &mut Device,
        ((query, key), rotation): (
            (
                HbmTensor<bf16, Chip, m![{ Q }, Head]>,
                HbmTensor<bf16, Chip, m![{ K }, Head]>,
            ),
            &Rotation<Rotated, Chip>,
        ),
    ) -> Self::Output {
        (
            self.rotate::<Q, bf16>(device, query, rotation),
            self.rotate::<K, D>(device, key, rotation),
        )
    }
}

/// Query and key head rows as attention projects them, for `Group` heads per key/value head.
type HeadRows<Rows, Slice, Group, Head> =
    Live<DmTensor<bf16, Chip, Cluster, Slice, m![{ RowGroups<Rows, Group> }, Head]>, Rows>;

/// Each live row's query and key rotated by its own row of the rotation.
impl<Head: AxisName, Rows: AxisName, Slice: M, Query: AxisName, Key: AxisName>
    crate::Op<(
        (
            HeadRows<Rows, Slice, Query, Head>,
            HeadRows<Rows, Slice, Key, Head>,
        ),
        &Rotation<Head, Chip, Rows>,
    )> for Rotary<Head, Head, bf16, Chip>
{
    type Output = (
        HeadRows<Rows, Slice, Query, Head>,
        HeadRows<Rows, Slice, Key, Head>,
    );

    fn forward(
        &self,
        device: &mut Device,
        ((query, key), rotation): (
            (
                HeadRows<Rows, Slice, Query, Head>,
                HeadRows<Rows, Slice, Key, Head>,
            ),
            &Rotation<Head, Chip, Rows>,
        ),
    ) -> Self::Output {
        let (rotated_query, rotated_key) = RotaryEmbedding::<Query, Key, Rows>::new().forward(
            device,
            (
                query.value(),
                key.value(),
                &rotation.low,
                &rotation.high,
                query.with(()),
            ),
        );
        (query.with(rotated_query), key.with(rotated_key))
    }
}

impl<Head: AxisName, Rotated: AxisName, D: Scalar, Chip: M> Rotary<Head, Rotated, D, Chip> {
    fn rotate<Heads: M, Out: MaterializableScalar>(
        &self,
        device: &mut Device,
        input: HbmTensor<bf16, Chip, m![{ Heads }, Head]>,
        rotation: &Rotation<Rotated, Chip>,
    ) -> HbmTensor<Out, Chip, m![{ Heads }, Head]>
    where
        f32: CastEngineCast<Out>,
        Convert<Out>:
            Op<HeadTile<Heads, Head, bf16, Chip>, Output = HeadTile<Heads, Head, Out, Chip>>,
    {
        const {
            assert!(
                Rotated::SIZE > 0
                    && Rotated::SIZE <= Head::SIZE
                    && Rotated::SIZE % 32 == 0
                    && Head::SIZE % Rotated::SIZE == 0
            );
        }
        type HeadSlices<Heads> = m![{Heads} # 256];
        type Row<R> = PairWidth<R>;
        let heads: DmTensor<bf16, Chip, Cluster, HeadSlices<Heads>, m![Head]> =
            input.to_dm(&mut device.tdma);
        let heads: DmTensor<bf16, Chip, Cluster, HeadSlices<Heads>, m![Half, {Symbol<Row<Head>>}]> =
            unsafe { heads.reshape() };
        let mut prefix: DmTensor<
            bf16,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![Half, {Symbol<Row<Head>>} = (Rotated / 2)],
        > = DmTensor::new();
        heads
            .view()
            .tile::<m![{Symbol<Row<Head>>} / (Rotated / 2)], 1, m![
                Half,
                {Symbol<Row<Head>>} / (Rotated / 2) = 1 # (Head / Rotated),
                {Symbol<Row<Head>>} % (Rotated / 2)
            ]>(0)
            .to_dm_view(&mut device.tdma, prefix.view_mut());
        let prefix: DmTensor<
            bf16,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![1, 1, Half, {Symbol<Row<Rotated>>}],
        > = unsafe { prefix.reshape() };
        let mut output: DmTensor<
            Out,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![1, 1, OutputHalf, {Symbol<Row<Rotated>>}],
        > = DmTensor::new();
        // SAFETY: a one-row rotation has the same contiguous order as the rotation vector.
        let (low, high) = unsafe {
            (
                rotation.low.view().reshape::<Chip, m![Rotated]>(),
                rotation.high.view().reshape::<Chip, m![Rotated]>(),
            )
        };
        let low: DmTensor<bf16, Chip, Cluster, m![256], m![Rotated]> = low.to_dm(&mut device.tdma);
        let high: DmTensor<bf16, Chip, Cluster, m![256], m![Rotated]> =
            high.to_dm(&mut device.tdma);
        let low: Table<HeadSlices<Heads>, m![1], Row<Rotated>, Chip> = unsafe { low.reshape() };
        let high: Table<HeadSlices<Heads>, m![1], Row<Rotated>, Chip> = unsafe { high.reshape() };
        let mut factors: DmTensor<
            bf16,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![1, 1, OutputHalf, Half, {Symbol<Row<Rotated>>}],
        > = DmTensor::new();
        low.view().to_dm_view(
            &mut device.tdma,
            factors
                .view_mut()
                .tile::<m![Half], 1, m![1, 1, OutputHalf, Half = 1 #{!} 2, {Symbol<Row<Rotated>>}]>(
                    0,
                ),
        );
        high.view().to_dm_view(
            &mut device.tdma,
            factors
                .view_mut()
                .tile::<m![Half], 1, m![1, 1, OutputHalf, Half = 1 #{!} 2, {Symbol<Row<Rotated>>}]>(
                    1,
                ),
        );
        let factors: VrfTensor<
            f32,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![1, 1, OutputHalf, Half, {Symbol<Row<Rotated>>}],
        > = device.sub.begin(factors.view())
            .fetch::<
                m![OutputHalf, Half, {Symbol<Row<Rotated>>} / 16],
                m![{Symbol<Row<Rotated>>} % 16],
            >()
            .fetch_cast::<f32>()
            .collect::<
                m![OutputHalf, Half, {Symbol<Row<Rotated>>} / 8],
                m![{Symbol<Row<Rotated>>} % 8],
            >()
            .to_vrf();
        let mut values: DmTensor<
            bf16,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![1, 1, OutputHalf, Half, {Symbol<Row<Rotated>>}],
        > = DmTensor::new();
        prefix
            .view()
            .to_dm_view(&mut device.tdma, values.view_mut());
        let products: DmTensor<
            bf16,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![1, 1, OutputHalf, Half, {Symbol<Row<Rotated>>}],
        > = device
            .main
            .begin(values.view())
            .fetch::<
                m![OutputHalf, Half, {Symbol<Row<Rotated>>} / 16],
                m![{Symbol<Row<Rotated>>} % 16],
            >()
            .fetch_cast::<f32>()
            .collect::<
                m![OutputHalf, Half, {Symbol<Row<Rotated>>} / 8],
                m![{Symbol<Row<Rotated>>} % 8],
            >()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![OutputHalf, Half, {Symbol<Row<Rotated>>} / 4],
                m![{Symbol<Row<Rotated>>} % 4],
            >()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factors)
            .vector_widen_concat::<
                m![OutputHalf, Half, {Symbol<Row<Rotated>>} / 8],
                m![{Symbol<Row<Rotated>>} % 8],
            >()
            .vector_final()
            .cast::<bf16, m![{Symbol<Row<Rotated>>} % 8 # 16]>()
            .commit_trim::<m![{Symbol<Row<Rotated>>} % 8]>()
            .commit();
        device.main.begin(products.view())
            .fetch::<
                m![OutputHalf, Half, {Symbol<Row<Rotated>>} / 16],
                m![{Symbol<Row<Rotated>>} % 16],
            >()
            .fetch_cast::<f32>()
            .collect::<
                m![OutputHalf, Half, {Symbol<Row<Rotated>>} / 8],
                m![{Symbol<Row<Rotated>>} % 8],
            >()
            .vector_init()
            .vector_zip::<
                Half,
                m![OutputHalf, {Symbol<Row<Rotated>>} / 8],
                m![OutputHalf, {Symbol<Row<Rotated>>} / 4],
                m![{Symbol<Row<Rotated>>} % 4],
            >(FpBinaryOp::AddF)
            .vector_final()
            .cast::<Out, m![{Symbol<Row<Rotated>>} % 8 # ({Symbol<Packet<Out>>})]>()
            .commit_trim::<m![{Symbol<Row<Rotated>>} % 8]>()
            .commit_view(output.view_mut());
        let output: DmTensor<
            Out,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![Half, {Symbol<Row<Head>>} = (Rotated / 2)],
        > = unsafe { output.reshape() };
        // SAFETY: join the existing consecutive halves without changing their scalar type.
        let heads: HeadTile<Heads, Head, bf16, Chip> = unsafe { heads.reshape() };
        let heads = Convert::<Out>::default().forward(device, heads);
        // SAFETY: recover the same consecutive halves after element conversion.
        let mut heads: DmTensor<
            Out,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![Half, {Symbol<Row<Head>>}],
        > = unsafe { heads.reshape() };
        output.view().to_dm_view(
            &mut device.tdma,
            heads
                .view_mut()
                .tile::<m![{Symbol<Row<Head>>} / (Rotated / 2)], 1, m![
                    Half,
                    {Symbol<Row<Head>>} / (Rotated / 2) = 1 #{!} (Head / Rotated),
                    {Symbol<Row<Head>>} % (Rotated / 2)
                ]>(0),
        );
        let heads: DmTensor<Out, Chip, Cluster, HeadSlices<Heads>, m![Head]> =
            unsafe { heads.reshape() };
        heads.to_hbm(&mut device.tdma)
    }
}

/// Cosine and sine for adjacent feature pairs, in token order.
pub type Phases<Rows, Head> = (
    HbmTensor<f32, m![1], m![Rows, Head / 2]>,
    HbmTensor<f32, m![1], m![Rows, Head / 2]>,
);

type Dense<Width, Rows> = HbmTensor<f32, m![1], m![Rows, Width]>;
type Qkv<Width, Rows> = (Dense<Width, Rows>, Dense<Width, Rows>, Dense<Width, Rows>);

#[derive(DeviceSend)]
pub struct Adjacent<Head>(pub PhantomData<Head>);

impl<Head> Default for Adjacent<Head> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Head: AxisName, W: AxisName, R: AxisName> crate::Op<(Qkv<W, R>, &Phases<R, Head>)>
    for Adjacent<Head>
{
    type Output = Qkv<W, R>;

    fn forward(&self, _: &mut Device, _: (Qkv<W, R>, &Phases<R, Head>)) -> Self::Output {
        const { assert!(Head::SIZE > 0 && Head::SIZE % 2 == 0 && W::SIZE % Head::SIZE == 0) };
        todo!(
            "NPU adjacent-pair query/key rotation using caller-supplied cosine and sine per token"
        )
    }
}

/// Independent coordinate axes occupy consecutive pairs of rotary channels.
#[derive(DeviceSend)]
pub struct Axial<Head, First, Second, Third> {
    pub theta: HbmTensor<f32, m![1], m![1]>,
    pub dimensions: PhantomData<(Head, First, Second, Third)>,
}

impl<Head: AxisName, First: AxisName, Second: AxisName, Third: AxisName, R: AxisName>
    crate::Op<&HbmTensor<f32, m![1], m![R, 3]>> for Axial<Head, First, Second, Third>
{
    type Output = Phases<R, Head>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<f32, m![1], m![R, 3]>) -> Self::Output {
        const {
            assert!(
                Head::SIZE == First::SIZE + Second::SIZE + Third::SIZE
                    && First::SIZE % 2 == 0
                    && Second::SIZE % 2 == 0
                    && Third::SIZE % 2 == 0
            );
        }
        todo!(
            "NPU cos/sin of coordinate * theta^(-2*pair/axis_width), concatenated in coordinate order"
        )
    }
}

/// Inverse frequencies and amplitude are parameters; positions produce device cosine/sine.
#[derive(DeviceSend)]
pub struct Frequencies<Head: AxisName> {
    pub inverse: HbmTensor<f32, m![1], m![Head / 2]>,
    pub amplitude: HbmTensor<f32, m![1], m![1]>,
}

impl<Head: AxisName, R: AxisName> crate::Op<&HbmTensor<i32, m![1], m![R]>> for Frequencies<Head> {
    type Output = Phases<R, Head>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<i32, m![1], m![R]>) -> Self::Output {
        todo!("NPU amplitude * (cos, sin)(position * inverse frequency) for every position")
    }
}

impl<Head: AxisName, Q: AxisName, K: AxisName>
    crate::Op<(
        &HbmTensor<bf16, m![1], m![Q, Head]>,
        &HbmTensor<bf16, m![1], m![K, Head]>,
        &Phases<axis::Single, Head>,
    )> for Adjacent<Head>
{
    type Output = (
        HbmTensor<bf16, m![1], m![Q, Head]>,
        HbmTensor<bf16, m![1], m![K, Head]>,
    );

    fn forward(
        &self,
        _: &mut Device,
        _: (
            &HbmTensor<bf16, m![1], m![Q, Head]>,
            &HbmTensor<bf16, m![1], m![K, Head]>,
            &Phases<axis::Single, Head>,
        ),
    ) -> Self::Output {
        const { assert!(Head::SIZE > 0 && Head::SIZE.is_multiple_of(2)) };
        todo!("NPU adjacent-pair rotary transform, sharing one position across query and key heads")
    }
}

/// Half-split rotary pairs with a load-time frequency spectrum.
#[derive(DeviceSend)]
pub struct SplitHalf<Head: AxisName, Frequency = HbmTensor<f32, m![1], m![Head / 2]>> {
    pub frequency: Frequency,
    pub head: PhantomData<Head>,
}

impl<D: Scalar, Head: AxisName, Q: AxisName, K: AxisName, R: AxisName>
    crate::Op<(
        (HbmTensor<D, m![1], m![R, Q]>, HbmTensor<D, m![1], m![R, K]>),
        &HbmTensor<i32, m![1], m![R]>,
    )> for SplitHalf<Head>
{
    type Output = (HbmTensor<D, m![1], m![R, Q]>, HbmTensor<D, m![1], m![R, K]>);

    fn forward(
        &self,
        _: &mut Device,
        _: (Self::Output, &HbmTensor<i32, m![1], m![R]>),
    ) -> Self::Output {
        const {
            assert!(
                Head::SIZE > 0
                    && Head::SIZE % 2 == 0
                    && Q::SIZE % Head::SIZE == 0
                    && K::SIZE % Head::SIZE == 0
            );
        }
        todo!(
            "NPU half-split per-head rotary: angle=position*frequency, FP32 sine/cosine and products, source-dtype output"
        )
    }
}

/// The `R / 2` rotated pairs' frequencies in radians per position and every rotation's amplitude,
/// positive and finite in FP32; coordinates `i` and `i + R / 2` turn at pair `i`'s.
#[derive(Debug)]
pub struct PairFrequencies<R> {
    pairs: Vec<f32>,
    amplitude: f32,
    rotated: core::marker::PhantomData<fn() -> R>,
}

impl<R> Clone for PairFrequencies<R> {
    fn clone(&self) -> Self {
        Self {
            pairs: self.pairs.clone(),
            amplitude: self.amplitude,
            rotated: core::marker::PhantomData,
        }
    }
}

impl<R> PartialEq for PairFrequencies<R> {
    fn eq(&self, other: &Self) -> bool {
        (&self.pairs, self.amplitude) == (&other.pairs, other.amplitude)
    }
}

impl<R: AxisName> PairFrequencies<R> {
    /// The frequencies of `pairs` at `amplitude`, rounded to FP32; another count than `R / 2`, or
    /// a frequency or amplitude FP32 does not hold as a positive finite value, is refused.
    pub fn new(pairs: &[f64], amplitude: f64) -> std::io::Result<Self> {
        let held = |value: f32| value.is_normal() && value > 0.0;
        let rounded: Vec<f32> = pairs.iter().map(|&frequency| frequency as f32).collect();
        match pairs.len() * 2 == R::SIZE && rounded.iter().all(|&f| held(f)) {
            true if held(amplitude as f32) => Ok(Self {
                pairs: rounded,
                amplitude: amplitude as f32,
                rotated: core::marker::PhantomData,
            }),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!(
                    "rotary frequencies: found {pairs:?} at amplitude {amplitude}, compiled {} \
                     positive finite pairs at a positive finite amplitude",
                    R::SIZE / 2
                ),
            )),
        }
    }

    /// The unscaled spectrum of base `theta` over a `Head`-wide head: pair `i` turns at
    /// `theta^(-2i / Head)`, at amplitude one.
    pub fn unscaled<Head: AxisName>(theta: f32) -> std::io::Result<Self> {
        let theta = f64::from(theta);
        let pairs: Vec<f64> = (0..R::SIZE / 2)
            .map(|pair| theta.powf(-2.0 * pair as f64 / Head::SIZE as f64))
            .collect();
        Self::new(&pairs, 1.0)
    }

    /// The frequency of each pair.
    pub fn pairs(&self) -> &[f32] {
        &self.pairs
    }

    /// The amplitude every pair's rotation scales by.
    pub fn amplitude(&self) -> f32 {
        self.amplitude
    }

    /// The frequency of each of the `R` coordinates.
    pub fn coordinates(&self) -> Vec<f32> {
        [self.pairs.as_slice(), self.pairs.as_slice()].concat()
    }
}

/// The rotation as a projection applies it a wave group at a time: coefficient tables gathered
/// by position and written beside the projected heads.
pub mod group {
    use super::SplitHalf;
    use crate::Op;
    use crate::load::{Context, Load, LoadError};
    use crate::modules::generic::Either;
    use crate::ops::linear::query::group::Group;
    use crate::ops::linear::query::{Cluster, SpreadCopies, Tile, fp8};
    use crate::tensor::live::Live;
    use crate::tensor::parallel::Chip;
    use furiosa_opt_std::prelude::*;
    use std::marker::PhantomData;

    axes![InputHalf = 2, OutputHalf = 2];

    pub type Table<Position, Half> =
        HbmTensor<bf16, Chip, m![Position, Half, OutputHalf, InputHalf]>;

    /// Each position's rotation, a [`Table`] owned or borrowed: what a gather reads.
    pub trait Rotations {
        type Position: AxisName;
        type Half: AxisName;
        fn table(&self) -> &Table<Self::Position, Self::Half>;
    }

    impl<Position: AxisName, Half: AxisName> Rotations for Table<Position, Half> {
        type Position = Position;
        type Half = Half;
        fn table(&self) -> &Self {
            self
        }
    }

    impl<Position: AxisName, Half: AxisName> Rotations for &Table<Position, Half> {
        type Position = Position;
        type Half = Half;
        fn table(&self) -> &Table<Position, Half> {
            self
        }
    }

    /// Each position's rotation at `frequencies`: pair `j` of position `p` turns by `p f_j`, an
    /// output half the sum over the input halves of `[[cos, -sin], [sin, cos]]` times them, scaled
    /// by the frequencies' amplitude, for every position the compiled `Position` holds.
    impl<'f, Head: AxisName, Position: AxisName> Load<&'f super::PairFrequencies<Head>>
        for SplitHalf<Head, Table<Position, super::Row<Head>>>
    {
        fn load(
            context: &mut Context<'_>,
            frequencies: &'f super::PairFrequencies<Head>,
        ) -> Result<Self, LoadError> {
            let amplitude = f64::from(frequencies.amplitude());
            let pairs = frequencies.pairs();
            let table = (0..Position::SIZE).flat_map(|position| {
                pairs.iter().flat_map(move |&frequency| {
                    let (sin, cos) = f64::from(position as f32 * frequency).sin_cos();
                    [cos, -sin, sin, cos].map(|value| bf16::from_f32((amplitude * value) as f32))
                })
            });
            let table = HostTensor::<
                bf16,
                m![{ Chip }, {
                    m![Position, { Symbol<super::Row<Head>> }, OutputHalf, InputHalf]
                }],
            >::from_vec(table.collect::<Vec<_>>().repeat(Chip::SIZE));
            Ok(Self {
                frequency: context.field("frequency", "", |cx| Load::load(cx, table))?,
                head: PhantomData,
            })
        }
    }
    pub(crate) type Rotation<S, Half> =
        HbmTensor<bf16, Chip, m![S, Tile, OutputHalf, InputHalf, Half]>;
    pub(crate) type Coefficients<C, P, Half> =
        DmTensor<bf16, Chip, C, P, m![OutputHalf, Tile % 8, InputHalf, Half]>;
    pub(crate) type Registers<C, P, Half> =
        TrfTensor<bf16, Chip, C, P, m![OutputHalf], m![Tile % 8, InputHalf, Half]>;
    /// Rows of `Local` heads rotated in the slices `P` of clusters `C`, each head's halves apart.
    pub(crate) type Rotated<C, P, Local, Half> =
        DmTensor<bf16, Chip, C, P, m![{ Local }, Tile % 8, OutputHalf, Half]>;

    /// A rotation's coefficients in data memory, owned or borrowed: the view `Stage` reads.
    pub(crate) trait Coefficient {
        type Cluster: M;
        type Slice: M;
        type Half: AxisName;
        fn view(
            &self,
        ) -> DmTensorView<
            '_,
            bf16,
            Chip,
            Self::Cluster,
            Self::Slice,
            m![OutputHalf, Tile % 8, InputHalf, { Symbol<Self::Half> }],
        >;
    }

    impl<Rotated: M, Chips: M, Half: AxisName> Coefficient for Coefficients<Rotated, Chips, Half> {
        type Cluster = Rotated;
        type Slice = Chips;
        type Half = Half;
        fn view(
            &self,
        ) -> DmTensorView<'_, bf16, Chip, Rotated, Chips, m![OutputHalf, Tile % 8, InputHalf, Half]>
        {
            self.view()
        }
    }

    impl<C: M, P: M, Half: AxisName> Coefficient
        for DmTensorView<'_, bf16, Chip, C, P, m![OutputHalf, Tile % 8, InputHalf, Half]>
    {
        type Cluster = C;
        type Slice = P;
        type Half = Half;
        fn view(
            &self,
        ) -> DmTensorView<'_, bf16, Chip, C, P, m![OutputHalf, Tile % 8, InputHalf, Half]> {
            self.clone()
        }
    }

    /// The rotation with its coefficients filled into the register file.
    pub(crate) struct Stage;

    impl<Head: AxisName, F: Coefficient> Op<&SplitHalf<Head, F>> for Stage {
        type Output = SplitHalf<Head, Registers<F::Cluster, F::Slice, F::Half>>;

        fn forward(&self, context: &mut Device, rotation: &SplitHalf<Head, F>) -> Self::Output {
            let coefficients: Registers<F::Cluster, F::Slice, F::Half> = context
                .sub
                .begin(rotation.frequency.view())
                .fetch::<m![OutputHalf, Tile % 8, InputHalf, { Symbol<F::Half> } / 16],
                    m![{ Symbol<F::Half> } % 16]>()
                .collect::<m![OutputHalf, Tile % 8, InputHalf, { Symbol<F::Half> } / 16],
                    m![{ Symbol<F::Half> } % 16]>()
                .to_trf();
            SplitHalf {
                frequency: coefficients,
                head: PhantomData,
            }
        }
    }

    /// Writes wave group `group`'s coefficients of a rotation in the slices of the projected
    /// queries of `Heads`, the ones their heads leave empty included, `G` waves at a time.
    pub(crate) struct ToDm<G, Heads>(pub(crate) PhantomData<(G, Heads)>);

    impl<G: Group, Heads: AxisName, Head: AxisName, S: AxisName, Half: AxisName>
        Op<(&SplitHalf<Head, Rotation<S, Half>>, usize)> for ToDm<G, Heads>
    {
        type Output =
            SplitHalf<Head, Coefficients<Cluster<Heads>, fp8::QuerySlices<G, Heads>, Half>>;

        fn forward(
            &self,
            context: &mut Device,
            (rotation, group): (&SplitHalf<Head, Rotation<S, Half>>, usize),
        ) -> Self::Output {
            let copies: DmTensor<
                bf16,
                Chip,
                Cluster<Heads>,
                fp8::Reduced<G, Heads, SpreadCopies<G, Heads>>,
                m![OutputHalf, Tile % 8, InputHalf, Half],
            > = G::stacked(&rotation.frequency, group).to_dm(&mut context.tdma);
            SplitHalf {
                // SAFETY: head digits only rename replicated coefficient coordinates, which every
                // copy holds.
                frequency: unsafe { copies.reshape() },
                head: PhantomData,
            }
        }
    }

    /// The rotation of the keys of `KvHeads`: coefficients written in every slice of the queries,
    /// filled into the register file in the slices of the keys.
    impl<Head: AxisName, G: Group, QueryHeads: AxisName, KvHeads: AxisName, Half: AxisName>
        Op<(
            &SplitHalf<
                Head,
                Coefficients<Cluster<QueryHeads>, fp8::QuerySlices<G, QueryHeads>, Half>,
            >,
            PhantomData<KvHeads>,
        )> for Stage
    {
        type Output =
            SplitHalf<Head, Registers<Cluster<KvHeads>, fp8::PairSlices<G, KvHeads>, Half>>;

        #[allow(clippy::type_complexity)]
        fn forward(
            &self,
            context: &mut Device,
            (coefficients, _): (
                &SplitHalf<
                    Head,
                    Coefficients<Cluster<QueryHeads>, fp8::QuerySlices<G, QueryHeads>, Half>,
                >,
                PhantomData<KvHeads>,
            ),
        ) -> Self::Output {
            // SAFETY: query head digits replicate coefficients, which every query slice holds, the
            // ones the queries leave empty included; key row digits occupy the same slices.
            let frequency = unsafe {
                coefficients.frequency.view().reshape::<Chip, Cluster<KvHeads>,
                    fp8::PairSlices<G, KvHeads>, m![OutputHalf, Tile % 8, InputHalf, Half]>()
            };
            self.forward(
                context,
                &SplitHalf::<Head, _> {
                    frequency,
                    head: PhantomData,
                },
            )
        }
    }

    impl<Head: AxisName, Half: AxisName, Local: M, C: M, P: M>
        Op<&DmTensor<bf16, Chip, C, P, m![{ Local }, Tile % 8, Head]>>
        for SplitHalf<Head, Registers<C, P, Half>>
    {
        type Output = Rotated<C, P, Local, Half>;

        fn forward(
            &self,
            context: &mut Device,
            normalized: &DmTensor<bf16, Chip, C, P, m![{ Local }, Tile % 8, Head]>,
        ) -> Self::Output {
            const { assert!(Head::SIZE == 2 * Half::SIZE) };
            // SAFETY: Head preserves contiguous InputHalf and Half coordinates.
            let normalized = unsafe {
                normalized
                    .view()
                    .reshape::<Chip, C, P, m![{ Local }, Tile % 8, InputHalf, Half]>()
            };
            context.main.begin(normalized)
                .fetch::<m![{ Local }, Tile % 8, Half / 32, InputHalf], m![Half % 32]>()
                .collect::<m![{ Local }, Tile % 8, Half / 32, InputHalf, Half % 32 / 16],
                    m![Half % 16]>()
                .contract_outer::<m![{ Local }, Tile % 8, Half / 32, InputHalf],
                    m![Half % 32], _, _, _>(&self.frequency)
                .contract_packet::<m![Half % 32]>()
                .contract_time::<m![{ Local }, Tile % 8, Half / 32]>()
                .contract_lane::<m![{ Local }, Tile % 8, Half / 32, OutputHalf, Half % 32 / 8],
                    m![Half % 8]>(LaneMode::Sequential)
                .commit_trim::<m![Half % 8]>().commit_cast::<bf16>().commit()
        }
    }

    impl<Head: AxisName, Half: AxisName, Local: M, C: M, P: M>
        Op<(
            &DmTensor<bf16, Chip, C, P, m![{ Local }, Tile % 8, Head]>,
            &SplitHalf<Head, Registers<C, P, Half>>,
        )> for super::Rotary<Head>
    {
        type Output = Rotated<C, P, Local, Half>;

        fn forward(
            &self,
            device: &mut Device,
            (normalized, rotation): (
                &DmTensor<bf16, Chip, C, P, m![{ Local }, Tile % 8, Head]>,
                &SplitHalf<Head, Registers<C, P, Half>>,
            ),
        ) -> Self::Output {
            rotation.forward(device, normalized)
        }
    }

    /// The queries and keys of a period of a pair of tiles or of a single tile, rotated by the
    /// launch's rotation at the period's wave group, which its live rows start: its coefficients
    /// are written once in the queries' slices, and the queries and the keys read them in their
    /// own slices.
    impl<
        'r,
        Head: AxisName,
        QueryHeads: AxisName,
        KvHeads: AxisName,
        P: Group,
        G: Group,
        S: AxisName,
        Half: AxisName,
        R: AxisName,
    >
        Op<(
            (
                Either<fp8::Queries<P, QueryHeads, Head>, fp8::Queries<G, QueryHeads, Head>>,
                Either<fp8::Pairs<P, KvHeads, Head>, fp8::Pairs<G, KvHeads, Head>>,
            ),
            &(&'r SplitHalf<Head, Rotation<S, Half>>, Live<(), R>),
        )> for super::Rotary<Head>
    {
        type Output = (
            Either<
                Rotated<
                    Cluster<QueryHeads>,
                    fp8::QuerySlices<P, QueryHeads>,
                    fp8::QueryLocal<P, QueryHeads>,
                    Half,
                >,
                Rotated<
                    Cluster<QueryHeads>,
                    fp8::QuerySlices<G, QueryHeads>,
                    fp8::QueryLocal<G, QueryHeads>,
                    Half,
                >,
            >,
            Either<
                Rotated<
                    Cluster<KvHeads>,
                    fp8::PairSlices<P, KvHeads>,
                    fp8::PairLocal<KvHeads>,
                    Half,
                >,
                Rotated<
                    Cluster<KvHeads>,
                    fp8::PairSlices<G, KvHeads>,
                    fp8::PairLocal<KvHeads>,
                    Half,
                >,
            >,
        );

        #[allow(clippy::type_complexity)]
        fn forward(
            &self,
            device: &mut Device,
            ((queries, keys), &(rotation, rows)): (
                (
                    Either<fp8::Queries<P, QueryHeads, Head>, fp8::Queries<G, QueryHeads, Head>>,
                    Either<fp8::Pairs<P, KvHeads, Head>, fp8::Pairs<G, KvHeads, Head>>,
                ),
                &(&'r SplitHalf<Head, Rotation<S, Half>>, Live<(), R>),
            ),
        ) -> Self::Output {
            let paired = |device: &mut Device| {
                ToDm::<P, QueryHeads>(PhantomData)
                    .forward(device, (rotation, rows.start() / (P::SIZE * Tile::SIZE)))
            };
            let grouped = |device: &mut Device| {
                ToDm::<G, QueryHeads>(PhantomData)
                    .forward(device, (rotation, rows.start() / (G::SIZE * Tile::SIZE)))
            };
            match (queries, keys) {
                (Either::Left(queries), Either::Left(keys)) => {
                    let coefficients = paired(&mut *device);
                    let (q, k) = (
                        Stage.forward(&mut *device, &coefficients),
                        Stage.forward(&mut *device, (&coefficients, PhantomData::<KvHeads>)),
                    );
                    let queries = self.forward(&mut *device, (&queries, &q));
                    (
                        Either::Left(queries),
                        Either::Left(self.forward(device, (&keys, &k))),
                    )
                }
                (Either::Right(queries), Either::Right(keys)) => {
                    let coefficients = grouped(&mut *device);
                    let (q, k) = (
                        Stage.forward(&mut *device, &coefficients),
                        Stage.forward(&mut *device, (&coefficients, PhantomData::<KvHeads>)),
                    );
                    let queries = self.forward(&mut *device, (&queries, &q));
                    (
                        Either::Right(queries),
                        Either::Right(self.forward(device, (&keys, &k))),
                    )
                }
                (Either::Left(queries), Either::Right(keys)) => {
                    let (q, k) = (paired(&mut *device), grouped(&mut *device));
                    let (q, k) = (
                        Stage.forward(&mut *device, &q),
                        Stage.forward(&mut *device, (&k, PhantomData::<KvHeads>)),
                    );
                    let queries = self.forward(&mut *device, (&queries, &q));
                    (
                        Either::Left(queries),
                        Either::Right(self.forward(device, (&keys, &k))),
                    )
                }
                (Either::Right(queries), Either::Left(keys)) => {
                    let (q, k) = (grouped(&mut *device), paired(&mut *device));
                    let (q, k) = (
                        Stage.forward(&mut *device, &q),
                        Stage.forward(&mut *device, (&k, PhantomData::<KvHeads>)),
                    );
                    let queries = self.forward(&mut *device, (&queries, &q));
                    (
                        Either::Right(queries),
                        Either::Left(self.forward(device, (&keys, &k))),
                    )
                }
            }
        }
    }

    pub(crate) struct Gather<'a, All: AxisName, S: AxisName, Half: AxisName> {
        pub(crate) positions: &'a HbmTensor<i32, Chip, m![All]>,
        pub(crate) coefficients: &'a mut Rotation<S, Half>,
        pub(crate) live: usize,
    }

    impl<Head: AxisName, Half: AxisName, All: AxisName, S: AxisName, F: Rotations<Half = Half>>
        Op<Gather<'_, All, S, Half>> for SplitHalf<Head, F>
    {
        type Output = ();

        fn forward(
            &self,
            context: &mut Device,
            Gather {
                positions,
                coefficients,
                live,
            }: Gather<'_, All, S, Half>,
        ) {
            const {
                assert!(All::SIZE >= Tile::SIZE);
                assert!(S::SIZE == All::SIZE.div_ceil(Tile::SIZE));
                assert!(Head::SIZE == 2 * Half::SIZE);
            }
            for wave in 0..live {
                type Halves = m![Tile / (Tile / 2)];
                type Copies = m![Tile % (Tile / 2), 512 / Tile];
                let (rotation, start) = (&mut *coefficients, wave * Tile::SIZE);
                let positions = positions
                    .view()
                    .tile::<m![All], { Tile::SIZE }, m![All = { Tile::SIZE } # All]>(start);
                // SAFETY: the window's rows are the wave's, in order.
                let positions: DmTensor<i32, Chip, Halves, Copies, m![1 # 2]> =
                    unsafe { positions.reshape::<Chip, m![Tile # All]>() }.to_dm(&mut context.tdma);
                let coefficients: DmTensor<
                    bf16,
                    Chip,
                    Halves,
                    Copies,
                    m![Half, OutputHalf, InputHalf],
                > = self.frequency.table().dma_gather_unscaled(&positions);
                let coefficients: DmTensor<
                    bf16,
                    Chip,
                    Halves,
                    Copies,
                    m![OutputHalf, InputHalf, Half],
                > = context
                    .main
                    .begin(coefficients.view())
                    .fetch::<m![Half], m![OutputHalf, InputHalf]>()
                    .collect::<m![Half], m![(OutputHalf, InputHalf) # 16]>()
                    .transpose::<m![Half / 4, OutputHalf, InputHalf], m![Half % 4 # 16]>()
                    .commit_trim::<m![Half % 4]>()
                    .commit();
                coefficients.view().to_hbm_view(
                    &mut context.tdma,
                    rotation
                        .view_mut()
                        .tile::<m![S], 1, m![S = 1 #{!} S, Tile, OutputHalf, InputHalf, Half]>(
                            wave,
                        ),
                );
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        axes![Position = 4, Head = 8];

        /// Every position's rotation loads as `[[cos, -sin], [sin, cos]]` of each pair's angle,
        /// scaled by the frequencies' amplitude.
        #[tokio::test(flavor = "current_thread")]
        async fn loads_every_positions_rotation() {
            type Rotation = SplitHalf<Head, Table<Position, super::super::Row<Head>>>;
            let _lease = crate::tests::DEVICE.lock().await;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let pairs: Vec<f64> = (0..Head::SIZE / 2)
                .map(|pair| 1e4f64.powf(-2.0 * pair as f64 / Head::SIZE as f64))
                .collect();
            let amplitude = 1.25;
            let frequencies =
                super::super::PairFrequencies::<Head>::new(&pairs, amplitude).unwrap();
            let rotation: Rotation =
                Load::load(&mut Context::new(&mut device), &frequencies).unwrap();
            let table = rotation
                .frequency
                .to_host::<m![Position, { Symbol<super::super::Row<Head>> }, OutputHalf, InputHalf]>(
                    &mut device.pdma,
                )
                .await
                .unwrap()
                .into_vec();
            for (index, found) in table.iter().enumerate() {
                let (position, pair, cell) = (
                    index / (pairs.len() * 4),
                    index / 4 % pairs.len(),
                    index % 4,
                );
                let angle = position as f64 * pairs[pair];
                let expected =
                    amplitude * [angle.cos(), -angle.sin(), angle.sin(), angle.cos()][cell];
                let error = (f64::from(found.to_f32()) - expected).abs();
                assert!(error <= 2f64.powi(-7), "{index}: {found:?} vs {expected}");
            }
        }
    }
}

/// NeoX rotation of a contiguous head prefix, preserving the remaining dimensions.
pub mod prefix {

    use crate::Op;
    use furiosa_opt_std::prelude::*;
    use std::marker::PhantomData;

    type Rows<Width, Count> = HbmTensor<bf16, m![1], m![Count, Width]>;

    /// Rotate pairs `j` and `j + Rotated/2` in each head prefix, preserving the suffix.
    /// Each pair's angle is `position * theta^(-2j/Rotated)`.
    #[derive(DeviceSend)]
    pub struct Rotary<Head, Rotated> {
        /// Base of the rotary frequency progression.
        pub theta: HbmTensor<f32, m![1], m![1]>,
        /// Full head width and the even-width prefix to rotate.
        pub dimensions: PhantomData<(Head, Rotated)>,
    }

    impl<Head: AxisName, Rotated: AxisName, Width: AxisName, KvWidth: AxisName, Count: AxisName>
        Op<(
            (Rows<Width, Count>, Rows<KvWidth, Count>),
            &HbmTensor<i32, m![1], m![Count]>,
        )> for Rotary<Head, Rotated>
    {
        type Output = (Rows<Width, Count>, Rows<KvWidth, Count>);

        fn forward(
            &self,
            _: &mut Device,
            _: (
                (Rows<Width, Count>, Rows<KvWidth, Count>),
                &HbmTensor<i32, m![1], m![Count]>,
            ),
        ) -> Self::Output {
            const {
                assert!(Head::SIZE > 0 && Rotated::SIZE > 0);
                assert!(Rotated::SIZE <= Head::SIZE && Rotated::SIZE % 2 == 0);
                assert!(Width::SIZE % Head::SIZE == 0 && KvWidth::SIZE % Head::SIZE == 0);
            }
            todo!(
                "NPU per-row NeoX RoPE: pair j with j+Rotated/2 within each head prefix, angle=position*theta^(-2j/Rotated), preserve head suffix"
            )
        }
    }
}

#[cfg(test)]
mod frequencies {
    use super::PairFrequencies;
    use super::tests::positioned;
    use crate::ops::attention::tests::Head;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;
    use std::io::ErrorKind;
    use std::iter;

    /// A spectrum binds the FP32 frequency of each of its pairs, the first pair's too, from the
    /// largest to the smallest one FP32 holds, and its amplitude; another pair count, or a
    /// frequency or amplitude FP32 rounds to zero or infinity, is refused.
    #[test]
    fn binds_every_pair_frequency() {
        let pairs = |first: f64, last: f64| {
            let mut pairs = vec![1.0; Head::SIZE / 2];
            (pairs[0], pairs[Head::SIZE / 2 - 1]) = (first, last);
            pairs
        };
        let held = PairFrequencies::<Head>::new(&pairs(0.5, 1e-30), 1.25).unwrap();
        assert_eq!(
            (held.pairs()[0], held.pairs()[Head::SIZE / 2 - 1]),
            (0.5, 1e-30)
        );
        assert_eq!(held.amplitude(), 1.25);
        assert_eq!(held.coordinates().len(), Head::SIZE);
        for (refused, amplitude) in [
            (pairs(1.0, 1e-50), 1.0),
            (pairs(1e40, 1.0), 1.0),
            (pairs(0.0, 1.0), 1.0),
            (vec![1.0; 3], 1.0),
            (pairs(1.0, 1.0), 0.0),
            (pairs(1.0, 1.0), 1e40),
        ] {
            let refusal = PairFrequencies::<Head>::new(&refused, amplitude).unwrap_err();
            assert_eq!(refusal.kind(), ErrorKind::Unsupported, "{refusal}");
        }
    }

    /// A rescaled spectrum's bound frequencies give its coefficients on device at the first
    /// position, on either side of its trained context and at the last admitted one, a spectrum
    /// whose first pair turns at half speed included; its unscaled spectrum misses them past the
    /// trained context.
    #[tokio::test(flavor = "current_thread")]
    async fn writes_rescaled_coefficients() {
        use crate::models::config::{parse, scaling::Spectrum};
        use crate::tensor::axis::Eight;
        use std::f64::consts::TAU;
        let yarn = serde_json::json!({"rope_parameters": {
            "rope_type": "yarn", "rope_theta": 1e9, "factor": 16.0, "mscale": 1.0,
            "mscale_all_dim": 1.0, "original_max_position_embeddings": 16384,
        }});
        let llama3 = serde_json::json!({"rope_theta": 1e6, "rope_scaling": {
            "rope_type": "llama3", "factor": 16.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
        }});
        let halved = serde_json::json!({"rope_parameters": {
            "rope_type": "proportional", "rope_theta": 1e4, "factor": 2.0,
        }});
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(super::tests::positioned.topology()).unwrap();
        let coefficients = super::tests::coefficients::<Head>(&mut device, 1.0).await;
        for (document, original, admitted) in [
            (yarn, 16384, 262144),
            (llama3, 8192, 131072),
            (halved, 1, 4096),
        ] {
            let spectrum: Spectrum = parse(&document).unwrap();
            let frequencies = spectrum.frequencies(Head::SIZE, Head::SIZE).unwrap();
            let positions = [
                0,
                1,
                original - 1,
                original,
                original + 1,
                2 * original,
                admitted / 2,
                admitted - 1,
            ];
            let unscaled =
                PairFrequencies::<Head>::unscaled::<Head>(spectrum.theta().get()).unwrap();
            for (bound, matches) in [
                (spectrum.bind::<Head, Head>().unwrap(), true),
                (unscaled, false),
            ] {
                let frequency: super::Scalars<Eight, Head> =
                    HostTensor::<f32, m![Eight, Head]>::from_vec(
                        bound.coordinates().repeat(Eight::SIZE),
                    )
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap();
                let position: super::Scalars<Eight, Head> =
                    HostTensor::<f32, m![Eight, Head]>::from_vec(
                        positions
                            .iter()
                            .flat_map(|&at| iter::repeat_n(at as f32, Head::SIZE)),
                    )
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap();
                let (low, high) = launch(
                    positioned,
                    (&mut device, &coefficients, &frequency, &position),
                )
                .await
                .unwrap();
                let mut tables = Vec::new();
                for table in [low, high] {
                    let values = table.to_host::<m![Eight, Head]>(&mut device.pdma).await;
                    tables.push(values.unwrap().into_vec());
                }
                let half = Head::SIZE / 2;
                let matched = positions.iter().enumerate().all(|(row, &at)| {
                    (0..Head::SIZE).all(|lane| {
                        let angle = at as f64 * frequencies[lane % half];
                        let exact = match lane < half {
                            true => [angle.cos(), -angle.sin()],
                            false => [angle.sin(), angle.cos()],
                        };
                        // BF16 rounding, then F32 rounding of the frequency and of the reduction
                        // to one turn, both growing with the angle.
                        let reduction = match angle >= TAU {
                            true => 4.0 * f64::from(f32::EPSILON) + 1e-6 / TAU,
                            false => 0.0,
                        };
                        exact.iter().zip(&tables).all(|(&value, found)| {
                            let found = f64::from(found[row * Head::SIZE + lane].to_f32());
                            let tolerance = value.abs() * 2f64.powi(-8)
                                + 2f64.powi(-12)
                                + angle * (4e-7 + reduction);
                            (found - value).abs() <= tolerance
                        })
                    })
                });
                assert_eq!(matched, matches, "{document}: {bound:?}");
            }
        }
    }

    /// Qwen3's coefficients under YaRN and linear scaling are Transformers 5.14.1's cos and sin,
    /// attention scaling included, on both sides of the 32768 trained and 40960 declared positions.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_transformers_rotary_embedding() {
        use crate::models::config::{parse, scaling::Spectrum};
        use crate::tensor::axis::Eight;
        use serde_json::json;
        use std::f64::consts::TAU;
        const POSITIONS: [usize; 8] = [0, 1, 32767, 32768, 40959, 40960, 65536, 131071];
        const PAIRS: [usize; 4] = [0, 1, 31, 63];
        // `Qwen3RotaryEmbedding(config)(x, positions)` in F32, Qwen3-0.6B's config with the
        // `rope_scaling` beside it: (cos, sin) of each of `PAIRS` at each of `POSITIONS`.
        const YARN: [[(f32, f32); 4]; 8] = [
            [(1.1386294, 0.0); 4],
            [
                (0.61520416, 0.9581236),
                (0.7885053, 0.8214234),
                (1.1386291, 9.142735e-4),
                (1.1386294, 3.5324206e-7),
            ],
            [
                (1.118434, 0.21350048),
                (-1.1369075, 0.06259694),
                (0.43600324, 1.0518452),
                (1.1385705, 0.011574483),
            ],
            [
                (0.424638, 1.0564846),
                (-0.8318507, -0.7774969),
                (0.4351585, 1.0521948),
                (1.1385705, 0.011574836),
            ],
            [
                (0.53162295, -1.0069033),
                (0.690014, 0.90573597),
                (0.11173727, 1.1331335),
                (1.1385375, 0.014468052),
            ],
            [
                (1.134517, -0.09668686),
                (-0.17427416, 1.1252136),
                (0.11082949, 1.1332228),
                (1.1385375, 0.014468406),
            ],
            [
                (-0.82190233, 0.7880061),
                (0.07682404, 1.1360348),
                (-0.806014, 0.8042503),
                (1.1383941, 0.023148477),
            ],
            [
                (-0.9313801, -0.6549871),
                (-0.67179704, 0.919329),
                (0.0015820703, -1.1386284),
                (1.1376883, 0.046287034),
            ],
        ];
        const LINEAR: [[(f32, f32); 4]; 8] = [
            [(1.0, 0.0); 4],
            [
                (0.95156795, 0.30743852),
                (0.9684591, 0.24917248),
                (0.99999994, 3.877931e-4),
                (1.0, 3.8779305e-7),
            ],
            [
                (-0.32759252, -0.9448191),
                (-0.17820404, 0.98399353),
                (0.9901538, 0.13998385),
                (0.9999193, 0.012706473),
            ],
            [
                (-0.021252776, -0.99977416),
                (-0.41788322, 0.90850073),
                (0.9900994, 0.14036816),
                (0.9999193, 0.012706861),
            ],
            [
                (0.66818535, 0.74399483),
                (-0.7777999, -0.62851197),
                (-0.98461246, -0.17475225),
                (0.9998739, 0.015882947),
            ],
            [
                (0.4070911, 0.91338754),
                (-0.59655726, -0.8025705),
                (-0.9845447, -0.17513347),
                (0.9998739, 0.015883334),
            ],
            [
                (-0.99909663, 0.04249595),
                (-0.65074724, -0.75929445),
                (0.9605936, 0.27795687),
                (0.99967706, 0.02541167),
            ],
            [
                (0.9220249, -0.38713062),
                (0.09619098, 0.9953629),
                (0.8456877, 0.5336781),
                (0.9987085, 0.05080654),
            ],
        ];
        let yarn = json!({
            "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768,
        });
        let linear = json!({"rope_type": "linear", "factor": 3.2});
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(super::tests::positioned.topology()).unwrap();
        for (rope, golden) in [(yarn, YARN), (linear, LINEAR)] {
            let document = json!({"rope_theta": 1e6, "rope_scaling": rope});
            let spectrum: Spectrum = parse(&document).unwrap();
            let bound = spectrum.bind::<Head, Head>().unwrap();
            let amplitude = bound.amplitude();
            let coefficients = super::tests::coefficients::<Head>(&mut device, amplitude).await;
            let frequency: super::Scalars<Eight, Head> =
                HostTensor::<f32, m![Eight, Head]>::from_vec(
                    bound.coordinates().repeat(Eight::SIZE),
                )
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let position: super::Scalars<Eight, Head> =
                HostTensor::<f32, m![Eight, Head]>::from_vec(
                    POSITIONS
                        .iter()
                        .flat_map(|&at| std::iter::repeat_n(at as f32, Head::SIZE)),
                )
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let (low, high) = launch(
                super::tests::positioned,
                (&mut device, &coefficients, &frequency, &position),
            )
            .await
            .unwrap();
            let mut tables = Vec::new();
            for table in [low, high] {
                let values = table.to_host::<m![Eight, Head]>(&mut device.pdma).await;
                tables.push(values.unwrap().into_vec());
            }
            let half = Head::SIZE / 2;
            for (row, &at) in POSITIONS.iter().enumerate() {
                for (&pair, &(cos, sin)) in PAIRS.iter().zip(&golden[row]) {
                    let angle = at as f64 * f64::from(bound.pairs()[pair]);
                    // BF16 rounding, then F32 rounding of the angle and of its reduction to one
                    // turn on either side, both growing with the angle.
                    let reduction = match angle >= TAU {
                        true => 4.0 * f64::from(f32::EPSILON) + 1e-6 / TAU,
                        false => 0.0,
                    };
                    let lanes = [
                        (0, pair, cos),
                        (0, half + pair, sin),
                        (1, pair, -sin),
                        (1, half + pair, cos),
                    ];
                    for (table, lane, expected) in lanes {
                        let expected = f64::from(expected);
                        let found = f64::from(tables[table][row * Head::SIZE + lane].to_f32());
                        let tolerance = expected.abs() * 2f64.powi(-8)
                            + f64::from(amplitude) * (2f64.powi(-12) + angle * (4e-7 + reduction));
                        assert!(
                            (found - expected).abs() <= tolerance,
                            "{document}: position {at}, lane {lane}: {found} vs {expected}"
                        );
                    }
                }
            }
        }
    }
}

/// Each live row's queries times the FP32 temperature of its own position, rounded to BF16 once;
/// the keys, and so the cache, keep their values.
impl<'s, Head: AxisName, Rows: AxisName, Slice: M, Query: AxisName, K>
    crate::Op<(
        (HeadRows<Rows, Slice, Query, Head>, K),
        &'s HbmTensor<f32, Chip, m![Rows, 1 # 8]>,
    )> for QueryScale
{
    type Output = (HeadRows<Rows, Slice, Query, Head>, K);

    fn forward(
        &self,
        device: &mut Device,
        ((query, key), factors): (
            (HeadRows<Rows, Slice, Query, Head>, K),
            &'s HbmTensor<f32, Chip, m![Rows, 1 # 8]>,
        ),
    ) -> Self::Output {
        type Grouped<Rows, Query, Head> = m![{ Groups<Rows> }, Query, Head];
        let factors: DmTensor<f32, Chip, Cluster, Slice, m![{ Groups<Rows> }, 1 # 8]> =
            factors.to_dm(&mut device.tdma);
        // SAFETY: a head row's group and query coordinates only flatten into one element axis.
        let rows = unsafe {
            query
                .value()
                .view()
                .reshape::<Chip, Cluster, Slice, Grouped<Rows, Query, Head>>()
        };
        let mut scaled: DmTensor<bf16, Chip, Cluster, Slice, Grouped<Rows, Query, Head>> =
            DmTensor::new();
        for group in 0..query.count::<Lanes<Rows>>() {
            let factor: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = device
                .sub
                .begin(factors.view().tile::<Groups<Rows>, 1, m![
                    { Layout<Rows>::Selected }, 1 # 8
                ]>(group))
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .to_vrf();
            device
                .main
                .begin(rows.clone().tile::<Groups<Rows>, 1, m![
                    { Layout<Rows>::Selected }, Query, Head
                ]>(group))
                .fetch::<m![Query, Head / 8], m![Head % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Query, Head / 8], m![Head % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Query, Head / 4], m![Head % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factor)
                .vector_widen_concat::<m![Query, Head / 8], m![Head % 8]>()
                .vector_final()
                .cast::<bf16, m![Head % 8 # 16]>()
                .commit_trim::<m![Head % 8]>()
                .commit_view(scaled.view_mut().tile::<Groups<Rows>, 1, m![
                    { Layout<Rows>::SelectedMut }, Query, Head
                ]>(group));
        }
        // SAFETY: the scaled rows keep the input's cells; only the element axes regroup.
        let value = unsafe { scaled.reshape() };
        (query.with(value), key)
    }
}
