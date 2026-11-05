use crate::{
    DeviceModule,
    ops::{cast::Convert, layout::Packet},
};
use furiosa_opt_std::prelude::*;

use super::full::heads::tiles;
use crate::ops::linear::Groups;

pub mod group;
pub mod resident;

use super::{RowGroups, TileGroups};
use crate::ops::layout::axes::{Chip, Lanes};
use crate::ops::layout::{Dm, Live, Rows as Layout};
use crate::ops::vector::Zip;

use crate::ops::layout::BroadcastClusters as Cluster;

// The rotation pairs each position with the one a half-head away: `Half` is the half a value is
// read from, `OutHalf` the half it is rotated into, `Row` its position within either.
axes![Half = 2, OutHalf = 2];

#[derive(Clone, Debug)]
pub struct Row<Head>(core::marker::PhantomData<Head>);

impl<Head: AxisName> AxisName for Row<Head> {
    const NAME: Ident = Ident::new("Row");
    const SIZE: usize = Head::SIZE / 2;
}

type Table<Slice, Count, Row, Chips = Chip> =
    DmTensor<bf16, Chips, Cluster, Slice, m![{ Count }, OutHalf, Row]>;

type Coefficient<Slice, Count, Row> =
    VrfTensor<f32, Chip, Cluster, Slice, m![{ Count }, OutHalf, Row]>;

/// Row groups one pass rotates; each group's two coefficient tables take 1 KiB of the shared
/// 8 KiB VRF.
type TileRows<Rows> = m![{ Groups<Rows> } % ({ Symbol<TileGroups<Rows, 32>> })];

/// The pass a row group belongs to.
type Tiles<Rows> = m![{ Groups<Rows> } / ({ Symbol<TileGroups<Rows, 32>> })];

/// Rotates every live row's query and key by its position's coefficients.
#[derive(DeviceSend)]
pub struct RotaryEmbedding<Query, Key, Rows>(core::marker::PhantomData<(Query, Key, Rows)>);

impl<Query, Key, Rows> RotaryEmbedding<Query, Key, Rows> {
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

/// The query and key heads, the coefficient rows of their positions and the live rows.
pub type Input<'a, Slice, Query, Key, Rows, Head> = (
    &'a DmTensor<bf16, Chip, Cluster, Slice, m![{ RowGroups<Rows, Query> }, Head]>,
    &'a DmTensor<bf16, Chip, Cluster, Slice, m![{ RowGroups<Rows, Key> }, Head]>,
    &'a HbmTensor<bf16, Chip, m![Rows, Head]>,
    &'a HbmTensor<bf16, Chip, m![Rows, Head]>,
    Live<Rows>,
);

impl<'d, 'a, Slice: M, Query: AxisName, Key: AxisName, Rows: AxisName, Head: AxisName>
    crate::Module<(&'d mut Device, Input<'a, Slice, Query, Key, Rows, Head>)>
    for RotaryEmbedding<Query, Key, Rows>
{
    type Output = (
        DmTensor<bf16, Chip, Cluster, Slice, m![{ RowGroups<Rows, Query> }, Head]>,
        DmTensor<bf16, Chip, Cluster, Slice, m![{ RowGroups<Rows, Key> }, Head]>,
    );

    /// Every row group in shared passes when every group holds a live row; otherwise each live
    /// group alone, and a group past the live rows is not rotated.
    fn forward(
        &self,
        (device, (query, key, low, high, live)): (
            &'d mut Device,
            Input<'a, Slice, Query, Key, Rows, Head>,
        ),
    ) -> Self::Output {
        let table = |device: &mut Device,
                     table: &HbmTensor<bf16, Chip, m![Rows, Head]>|
         -> Table<Slice, Groups<Rows>, Row<Head>> {
            // SAFETY: a coefficient row consists of its two consecutive output halves.
            unsafe {
                table
                    .view()
                    .reshape::<Chip, m![Rows, OutHalf, { Symbol<Row<Head>> }]>()
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
            m![{ Groups<Rows> }, Query, OutHalf, { Symbol<Row<Head>> }],
        > = DmTensor::new();
        let mut rotated_key: DmTensor<
            bf16,
            Chip,
            Cluster,
            Slice,
            m![{ Groups<Rows> }, Key, OutHalf, { Symbol<Row<Head>> }],
        > = DmTensor::new();
        // SAFETY: each head's two halves remain consecutive within the same row and slice.
        let query = unsafe {
            query
                .view()
                .reshape::<Chip, Cluster, Slice, m![{ Groups<Rows> }, Query, Half, { Symbol<Row<Head>> }]>()
        };
        let key = unsafe {
            key.view()
                .reshape::<Chip, Cluster, Slice, m![{ Groups<Rows> }, Key, Half, { Symbol<Row<Head>> }]>()
        };
        if live.count() == <Groups<Rows> as M>::SIZE {
            #[unroll]
            for tile in 0..<Tiles<Rows> as M>::SIZE {
                let low = Fetch::<TileRows<Rows>, Row<Head>>(core::marker::PhantomData).forward((
                    device,
                    low.view().tile::<Tiles<Rows>, 1, m![
                        { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, OutHalf, { Symbol<Row<Head>> }
                    ]>(tile),
                ));
                let high = Fetch::<TileRows<Rows>, Row<Head>>(core::marker::PhantomData).forward((
                    device,
                    high.view().tile::<Tiles<Rows>, 1, m![
                        { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, OutHalf, { Symbol<Row<Head>> }
                    ]>(tile),
                ));
                RotateRows::<TileRows<Rows>, m![Query], Row<Head>>(core::marker::PhantomData).forward((
                    device, (
                    query.clone().tile::<Tiles<Rows>, 1, m![
                        { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, Query, Half, { Symbol<Row<Head>> }
                    ]>(tile),
                    rotated_query.view_mut().tile::<Tiles<Rows>, 1, m![
                        { Tiles<Rows> } = 1 #{!} ({ Tiles<Rows> }), { TileRows<Rows> }, Query, OutHalf, { Symbol<Row<Head>> }
                    ]>(tile),
                    &low,
                    &high,
                )));
                RotateRows::<TileRows<Rows>, m![Key], Row<Head>>(core::marker::PhantomData).forward((
                    device, (
                    key.clone().tile::<Tiles<Rows>, 1, m![
                        { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, Key, Half, { Symbol<Row<Head>> }
                    ]>(tile),
                    rotated_key.view_mut().tile::<Tiles<Rows>, 1, m![
                        { Tiles<Rows> } = 1 #{!} ({ Tiles<Rows> }), { TileRows<Rows> }, Key, OutHalf, { Symbol<Row<Head>> }
                    ]>(tile),
                    &low,
                    &high,
                )));
            }
        } else {
            for group in 0..live.count() {
                let low = Fetch::<m![1], Row<Head>>(core::marker::PhantomData).forward((
                    device,
                    low.view().tile::<Groups<Rows>, 1, m![
                        { Layout<Rows>::Selected }, OutHalf, { Symbol<Row<Head>> }
                    ]>(group),
                ));
                let high = Fetch::<m![1], Row<Head>>(core::marker::PhantomData).forward((
                    device,
                    high.view().tile::<Groups<Rows>, 1, m![
                        { Layout<Rows>::Selected }, OutHalf, { Symbol<Row<Head>> }
                    ]>(group),
                ));
                RotateRows::<m![1], m![Query], Row<Head>>(core::marker::PhantomData).forward((
                    device,
                    (
                        query.clone().tile::<Groups<Rows>, 1, m![
                            { Layout<Rows>::Selected }, Query, Half, { Symbol<Row<Head>> }
                        ]>(group),
                        rotated_query.view_mut().tile::<Groups<Rows>, 1, m![
                            { Layout<Rows>::SelectedMut }, Query, OutHalf, { Symbol<Row<Head>> }
                        ]>(group),
                        &low,
                        &high,
                    ),
                ));
                RotateRows::<m![1], m![Key], Row<Head>>(core::marker::PhantomData).forward((
                    device,
                    (
                        key.clone().tile::<Groups<Rows>, 1, m![
                            { Layout<Rows>::Selected }, Key, Half, { Symbol<Row<Head>> }
                        ]>(group),
                        rotated_key.view_mut().tile::<Groups<Rows>, 1, m![
                            { Layout<Rows>::SelectedMut }, Key, OutHalf, { Symbol<Row<Head>> }
                        ]>(group),
                        &low,
                        &high,
                    ),
                ));
            }
        }
        // SAFETY: output halves concatenate to Head within each unchanged row.
        unsafe { (rotated_query.reshape(), rotated_key.reshape()) }
    }
}

struct Fetch<Count, Row>(core::marker::PhantomData<(Count, Row)>);

impl<Slice: M, Count: M, Element: M, Row: AxisName>
    crate::Module<(
        &mut Device,
        DmTensorView<'_, bf16, Chip, Cluster, Slice, Element>,
    )> for Fetch<Count, Row>
{
    type Output = Coefficient<Slice, Count, Row>;

    fn forward(
        &self,
        (device, table): (
            &mut Device,
            DmTensorView<'_, bf16, Chip, Cluster, Slice, Element>,
        ),
    ) -> Coefficient<Slice, Count, Row> {
        device
            .sub
            .begin(table)
            .fetch::<m![{ Count }, OutHalf, Row / 16], m![Row % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Count }, OutHalf, Row / 8], m![Row % 8]>()
            .to_vrf()
    }
}

struct RotateRows<Count, Group, Row>(core::marker::PhantomData<(Count, Group, Row)>);

impl<Slice: M, Count: M, Group: M, Row: AxisName, Input: M, Output: M>
    crate::Module<(
        &mut Device,
        (
            DmTensorView<'_, bf16, Chip, Cluster, Slice, Input>,
            DmTensorViewMut<'_, bf16, Chip, Cluster, Slice, Output>,
            &Coefficient<Slice, Count, Row>,
            &Coefficient<Slice, Count, Row>,
        ),
    )> for RotateRows<Count, Group, Row>
{
    type Output = ();

    fn forward(
        &self,
        (device, (input, output, low, high)): (
            &mut Device,
            (
                DmTensorView<'_, bf16, Chip, Cluster, Slice, Input>,
                DmTensorViewMut<'_, bf16, Chip, Cluster, Slice, Output>,
                &Coefficient<Slice, Count, Row>,
                &Coefficient<Slice, Count, Row>,
            ),
        ),
    ) {
        device
            .main
            .begin_interleaved::<OutHalf, _, _, _, _, _>(input.clone(), input)
            .fetch::<m![{ Count }, { Group }, OutHalf, Half, Row / 16], m![Row % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Count }, { Group }, OutHalf, Half, Row / 8], m![Row % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Half, m![{ Count }, { Group }, OutHalf, Row / 8]>()
            .vector_narrow_split::<m![{ Count }, { Group }, OutHalf, Row / 4], m![Row % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), low, ())
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), high)
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_widen_concat::<m![{ Count }, { Group }, OutHalf, Row / 8], m![Row % 8]>()
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
/// position.
#[derive(DeviceSend)]
pub struct Coefficients<Rotated: AxisName, Chips: M = Chip> {
    /// Phase shifts in radians for the first input half's coefficients.
    pub low_phase: Parameters<Rotated, Chips>,
    /// Angle multipliers for the first input half's coefficients.
    pub low_sign: Parameters<Rotated, Chips>,
    /// Phase shifts in radians for the second input half's coefficients.
    pub high_phase: Parameters<Rotated, Chips>,
    /// Angle multipliers for the second input half's coefficients.
    pub high_sign: Parameters<Rotated, Chips>,
}

impl<Rotated: AxisName, Chips: M> Coefficients<Rotated, Chips> {
    pub fn new(
        low_phase: Parameters<Rotated, Chips>,
        low_sign: Parameters<Rotated, Chips>,
        high_phase: Parameters<Rotated, Chips>,
        high_sign: Parameters<Rotated, Chips>,
    ) -> Self {
        Self {
            low_phase,
            low_sign,
            high_phase,
            high_sign,
        }
    }

    /// Host values of each field, named and ordered as `new` takes them.
    pub fn rows() -> [(&'static str, Vec<f32>); 4] {
        const { assert!(Rotated::SIZE > 0 && Rotated::SIZE.is_multiple_of(16)) };
        [
            (
                "low_phase",
                (0..Rotated::SIZE)
                    .map(|i| {
                        if i < Rotated::SIZE / 2 {
                            core::f32::consts::FRAC_PI_2
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            ),
            ("low_sign", vec![1.0; Rotated::SIZE]),
            (
                "high_phase",
                (0..Rotated::SIZE)
                    .map(|i| {
                        if i < Rotated::SIZE / 2 {
                            0.0
                        } else {
                            core::f32::consts::FRAC_PI_2
                        }
                    })
                    .collect(),
            ),
            (
                "high_sign",
                (0..Rotated::SIZE)
                    .map(|i| if i < Rotated::SIZE / 2 { -1.0 } else { 1.0 })
                    .collect(),
            ),
        ]
    }
}

impl<Rotated: AxisName, Chip: M> Coefficients<Rotated, Chip> {
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

    /// One coefficient component of each row: the sine of its angle times `sign` plus `phase`.
    fn component<Rows: AxisName>(
        device: &mut Device,
        angle: &Angles<Rows, Rotated, Chip>,
        phase: &Parameters<Rotated, Chip>,
        sign: &Parameters<Rotated, Chip>,
    ) -> CoefficientSlices<Rows, Rotated, Chip> {
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
                .vector_fp_div(core::f32::consts::TAU)
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
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), core::f32::consts::TAU)
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
            .vector_widen_concat::<m![Rows], m![Rotated % 8]>()
            .vector_final()
            .cast::<bf16, m![Rotated % 8 # 16]>()
            .commit_trim::<m![Rotated % 8]>()
            .commit()
    }
}

impl<Rows: AxisName, Rotated: AxisName, Chip: M>
    crate::Module<(
        &mut Device,
        (&Scalars<Rows, Rotated, Chip>, &Scalars<Rows, Rotated, Chip>),
    )> for Coefficients<Rotated, Chip>
{
    type Output = Rotation<Rotated, Chip, Rows>;

    fn forward(
        &self,
        (device, scalars): (
            &mut Device,
            (&Scalars<Rows, Rotated, Chip>, &Scalars<Rows, Rotated, Chip>),
        ),
    ) -> Self::Output {
        let angle = self.angle(device, scalars);
        let low = Self::component(device, &angle, &self.low_phase, &self.low_sign)
            .to_hbm(&mut device.tdma);
        let high = Self::component(device, &angle, &self.high_phase, &self.high_sign)
            .to_hbm(&mut device.tdma);
        Rotation { low, high }
    }
}

/// The rotation of the live rows only; a later row's rotation is unspecified.
impl<Rows: AxisName, Rotated: AxisName, Chip: M>
    crate::Module<(
        &mut Device,
        (
            &Scalars<Rows, Rotated, Chip>,
            &Scalars<Rows, Rotated, Chip>,
            Live<Rows>,
        ),
    )> for Coefficients<Rotated, Chip>
{
    type Output = Rotation<Rotated, Chip, Rows>;

    fn forward(
        &self,
        (device, (frequency, position, live)): (
            &mut Device,
            (
                &Scalars<Rows, Rotated, Chip>,
                &Scalars<Rows, Rotated, Chip>,
                Live<Rows>,
            ),
        ),
    ) -> Self::Output {
        // Stores only the live rows; a later row is left unwritten.
        let store = |device: &mut Device,
                     table: &CoefficientSlices<Rows, Rotated, Chip>|
         -> CoefficientRows<Rows, Rotated, Chip> {
            let mut output: CoefficientRows<Rows, Rotated, Chip> = HbmTensor::new();
            live.cover(
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
                                .tile::<Layout<Rows>::Groups, 1, Layout<Rows>::GroupMut<m![Rotated]>>(
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
        let low = Self::component(device, &angle, &self.low_phase, &self.low_sign);
        let low = store(device, &low);
        let high = Self::component(device, &angle, &self.high_phase, &self.high_sign);
        let high = store(device, &high);
        Rotation { low, high }
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use super::super::{
        Heads,
        tests::{Head, KvHeads},
    };
    use super::{Chip, RotaryEmbedding};
    use crate::Module;
    use crate::ops::layout::axes::{Rows, Single};

    axes![Groups = 2];

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
        let rotation = coefficients.forward((device, (frequency, position)));
        (rotation.low, rotation.high)
    }

    pub(super) async fn coefficients<Rotated: AxisName>(
        device: &mut Device,
    ) -> super::Coefficients<Rotated> {
        let [low_phase, low_sign, high_phase, high_sign] = super::Coefficients::<Rotated>::rows()
            .map(|(_, values)| HostTensor::<f32, m![Rotated]>::from_vec(values));
        let mut upload =
            async |row: HostTensor<f32, m![Rotated]>| row.to_hbm(&mut device.pdma).await.unwrap();
        super::Coefficients::new(
            upload(low_phase).await,
            upload(low_sign).await,
            upload(high_phase).await,
            upload(high_sign).await,
        )
    }

    async fn check_coefficients<Head: AxisName, Rotated: AxisName>(device: &mut Device) {
        use crate::ops::layout::axes::Eight;
        let coefficients = coefficients::<Rotated>(device).await;
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
                    .flat_map(|&(position, _)| std::iter::repeat_n(position as f32, Rotated::SIZE)),
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
                        angle(index).cos()
                    } else {
                        angle(index).sin()
                    })
                })
                .collect::<Vec<_>>();
            let expected_high = (0..Rotated::SIZE)
                .map(|index| {
                    bf16::from_f32(if index < Rotated::SIZE / 2 {
                        -angle(index).sin()
                    } else {
                        angle(index).cos()
                    })
                })
                .collect::<Vec<_>>();
            for (actual, expected) in [(&low, expected_low), (&high, expected_high)] {
                for (lane, expected) in expected.into_iter().enumerate() {
                    let actual = actual[row * Rotated::SIZE + lane].to_f32();
                    let expected = expected.to_f32();
                    assert!(
                        (actual - expected).abs() <= 0.01,
                        "position {position}, theta {theta}, lane {lane}: {actual} vs {expected}"
                    );
                }
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn matches_position_coefficients() {
        axes![Full = 256, Partial = 512];
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(positioned.topology()).unwrap();
        check_coefficients::<Head, Head>(&mut device).await;
        check_coefficients::<Full, Full>(&mut device).await;
        check_coefficients::<Partial, Head>(&mut device).await;
    }

    type QueryRows<Count> = HbmTensor<bf16, Chip, m![Count, KvHeads, Groups, Head]>;
    type KeyRows<Count> = HbmTensor<bf16, Chip, m![Count, KvHeads, Head]>;

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
        let (query, key) = RotaryEmbedding::<Groups, Single, Count>::new().forward((
            &mut *device,
            (
                &query,
                &key,
                low,
                high,
                crate::ops::layout::Live::over(Count::SIZE),
            ),
        ));
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
        let _lease = crate::tests::DEVICE.lock().await;
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
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(rotate.topology()).unwrap();
        check_products::<Full, Full>(&mut device).await;
        check_products::<Partial, Head>(&mut device).await;
    }
}

#[derive(Clone, Debug)]
struct PairWidth<Rotated>(core::marker::PhantomData<Rotated>);

impl<Rotated: AxisName> AxisName for PairWidth<Rotated> {
    const NAME: Ident = Ident::new("PairWidth");
    const SIZE: usize = Rotated::SIZE / 2;
}

/// Whole heads within each rotary DM slice.
type HeadTile<Heads, Head, D = bf16, Chips = Chip> =
    DmTensor<D, Chips, Cluster, m![{Heads} # 256], m![Head]>;

/// Rotate the first Rotated/2 pairs across the two Head/2 halves by each row's rotation; `D` is
/// the element type of the rotated key output.
#[derive(DeviceSend)]
pub struct Rotary<Head: AxisName, Rotated: AxisName = Head, D: Scalar = bf16, Chips: M = Chip>(
    core::marker::PhantomData<(Head, Rotated, D, Chips)>,
);

impl<Head: AxisName, Rotated: AxisName, D: Scalar, Chips: M> Rotary<Head, Rotated, D, Chips> {
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
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
    crate::Module<(
        &mut Device,
        (
            (
                &'q HbmTensor<bf16, Chips, m![R, Width]>,
                &'k HbmTensor<bf16, Chips, m![R, Width]>,
            ),
            &Rotation<Head, Chips, R>,
        ),
    )> for Rotary<Head, Head, bf16, Chips>
{
    type Output = (
        HbmTensor<bf16, Chips, m![R, Width]>,
        HbmTensor<bf16, Chips, m![R, Width]>,
    );

    fn forward(
        &self,
        (device, ((query, key), rotation)): (
            &mut Device,
            (
                (
                    &'q HbmTensor<bf16, Chips, m![R, Width]>,
                    &'k HbmTensor<bf16, Chips, m![R, Width]>,
                ),
                &Rotation<Head, Chips, R>,
            ),
        ),
    ) -> Self::Output {
        (
            Heads::<Width, R, Head, Chips, _>(core::marker::PhantomData)
                .forward((&mut *device, (query, rotation))),
            Heads::<Width, R, Head, Chips, _>(core::marker::PhantomData)
                .forward((device, (key, rotation))),
        )
    }
}

impl<'q, 'k, Head: AxisName, Width: AxisName, R: AxisName, Chips: M>
    crate::Module<(
        &mut Device,
        (
            (
                &'q tiles::Major<R, Width, Head, Chips>,
                &'k tiles::Major<R, Width, Head, Chips>,
            ),
            &Rotation<Head, Chips, R>,
        ),
    )> for Rotary<Head, Head, bf16, Chips>
{
    type Output = (
        tiles::Major<R, Width, Head, Chips>,
        tiles::Major<R, Width, Head, Chips>,
    );

    fn forward(
        &self,
        (device, ((query, key), rotation)): (
            &mut Device,
            (
                (
                    &'q tiles::Major<R, Width, Head, Chips>,
                    &'k tiles::Major<R, Width, Head, Chips>,
                ),
                &Rotation<Head, Chips, R>,
            ),
        ),
    ) -> Self::Output {
        (
            Heads::<Width, R, Head, Chips, _>(core::marker::PhantomData)
                .forward((&mut *device, (query, rotation))),
            Heads::<Width, R, Head, Chips, _>(core::marker::PhantomData)
                .forward((device, (key, rotation))),
        )
    }
}

struct Heads<Width, R, Head, Chips, Q>(core::marker::PhantomData<(Width, R, Head, Chips, Q)>);

impl<Head: AxisName, Chips: M, Width: AxisName, R: AxisName, Q: M, L>
    crate::Module<(&mut Device, (&L, &Rotation<Head, Chips, R>))>
    for Heads<Width, R, Head, Chips, Q>
where
    L: tiles::Layout<R, Width, Head, Chips, Order = Q>,
{
    type Output = L;

    fn forward(
        &self,
        (device, (input, rotation)): (&mut Device, (&L, &Rotation<Head, Chips, R>)),
    ) -> L {
        let mut output = L::new();
        tiles::Strips::<R, Width, Head, _, Chips, Q>(
            core::marker::PhantomData,
            Rotate { rotation },
        )
        .forward((device, (input, &mut output)));
        output
    }
}

impl<Head: AxisName, Chips: M, Width: AxisName, R: AxisName, Q: M, L>
    crate::Module<(&mut Device, ((&L, &L), &Rotation<Head, Chips, R>))>
    for Heads<Width, R, Head, Chips, Q>
where
    L: tiles::Layout<R, Width, Head, Chips, Order = Q>,
{
    type Output = (L, L);

    fn forward(
        &self,
        (device, ((query, key), rotation)): (&mut Device, ((&L, &L), &Rotation<Head, Chips, R>)),
    ) -> (L, L) {
        let (mut rotated_query, mut rotated_key) = (L::new(), L::new());
        tiles::Strips::<R, Width, Head, _, Chips, Q>(
            core::marker::PhantomData,
            Rotate { rotation },
        )
        .forward((
            device,
            ((query, key), (&mut rotated_query, &mut rotated_key)),
        ));
        (rotated_query, rotated_key)
    }
}

impl<Head: AxisName, Width: AxisName, R: AxisName, Chips: M>
    crate::Module<(
        &mut Device,
        (
            &(
                &HbmTensor<bf16, Chips, m![R, Width]>,
                &HbmTensor<bf16, Chips, m![R, Width]>,
            ),
            &Rotation<Head, Chips, R>,
        ),
    )> for Rotary<Head, Head, bf16, Chips>
{
    type Output = (
        HbmTensor<bf16, Chips, m![R, Width]>,
        HbmTensor<bf16, Chips, m![R, Width]>,
    );

    fn forward(
        &self,
        (device, (rows, rotation)): (
            &mut Device,
            (
                &(
                    &HbmTensor<bf16, Chips, m![R, Width]>,
                    &HbmTensor<bf16, Chips, m![R, Width]>,
                ),
                &Rotation<Head, Chips, R>,
            ),
        ),
    ) -> Self::Output {
        Heads::<Width, R, Head, Chips, _>(core::marker::PhantomData)
            .forward((device, (*rows, rotation)))
    }
}

impl<Head: AxisName, Width: AxisName, R: AxisName, Chips: M>
    crate::Module<(
        &mut Device,
        (
            &(
                &tiles::Major<R, Width, Head, Chips>,
                &tiles::Major<R, Width, Head, Chips>,
            ),
            &Rotation<Head, Chips, R>,
        ),
    )> for Rotary<Head, Head, bf16, Chips>
{
    type Output = (
        tiles::Major<R, Width, Head, Chips>,
        tiles::Major<R, Width, Head, Chips>,
    );

    fn forward(
        &self,
        (device, (rows, rotation)): (
            &mut Device,
            (
                &(
                    &tiles::Major<R, Width, Head, Chips>,
                    &tiles::Major<R, Width, Head, Chips>,
                ),
                &Rotation<Head, Chips, R>,
            ),
        ),
    ) -> Self::Output {
        Heads::<Width, R, Head, Chips, _>(core::marker::PhantomData)
            .forward((device, (*rows, rotation)))
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
    m![{ Lines<Width, Head> }, OutHalf, { Symbol<Row<Head>> }],
>;

struct FetchTiles<Width, Head, R, Chips>(core::marker::PhantomData<(Width, Head, R, Chips)>);

impl<Head: AxisName, R: AxisName, Chips: M, Width: AxisName>
    crate::Module<(&mut Device, (&Rotation<Head, Chips, R>, usize))>
    for FetchTiles<Width, Head, R, Chips>
{
    type Output = (
        TileCoefficients<Width, Head, Chips>,
        TileCoefficients<Width, Head, Chips>,
    );

    fn forward(
        &self,
        (device, (rotation, at)): (&mut Device, (&Rotation<Head, Chips, R>, usize)),
    ) -> (
        TileCoefficients<Width, Head, Chips>,
        TileCoefficients<Width, Head, Chips>,
    ) {
        let rows = |device: &mut Device, table: &CoefficientRows<R, Head, Chips>| {
            // SAFETY: the tiles split rows in order and each coefficient row is its two halves.
            let rows = unsafe {
                table.view().reshape::<Chips, m![{ tiles::S<R, Width, Head> }, { tiles::T<Width, Head> }, OutHalf, { Symbol<Row<Head>> }]>()
            };
            let landed: DmTensor<bf16, Chips, tiles::Half<Width, Head>, tiles::Slices<Width, Head>, m![{ Lines<Width, Head> }, OutHalf, { Symbol<Row<Head>> }]> = rows
                .tile::<m![{ tiles::S<R, Width, Head> }], 1, m![{ tiles::S<R, Width, Head> } = 1 # ({ tiles::S<R, Width, Head> }), { tiles::T<Width, Head> }, OutHalf, { Symbol<Row<Head>> }]>(at)
                .to_dm(&mut device.tdma);
            let operand: TileCoefficients<Width, Head, Chips> = device.sub.begin(landed.view())
                .fetch::<m![{ Lines<Width, Head> }, OutHalf, { Symbol<Row<Head>> } / 16], m![{ Symbol<Row<Head>> } % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Lines<Width, Head> }, OutHalf, { Symbol<Row<Head>> } / 8], m![{ Symbol<Row<Head>> } % 8]>()
                .to_vrf();
            operand
        };
        (rows(device, &rotation.low), rows(device, &rotation.high))
    }
}

struct RotateTile<Head>(core::marker::PhantomData<Head>);

impl<Head: AxisName, Chips: M, Width: AxisName, Q: M>
    crate::Module<(
        &mut Device,
        (
            &tiles::Tile<bf16, Width, Head, Chips, Q>,
            &(
                TileCoefficients<Width, Head, Chips>,
                TileCoefficients<Width, Head, Chips>,
            ),
            &mut tiles::Tile<bf16, Width, Head, Chips, Q>,
        ),
    )> for RotateTile<Head>
{
    type Output = ();

    fn forward(
        &self,
        (device, (rows, (low, high), output)): (
            &mut Device,
            (
                &tiles::Tile<bf16, Width, Head, Chips, Q>,
                &(
                    TileCoefficients<Width, Head, Chips>,
                    TileCoefficients<Width, Head, Chips>,
                ),
                &mut tiles::Tile<bf16, Width, Head, Chips, Q>,
            ),
        ),
    ) {
        // SAFETY: a head is its two halves in order.
        let halves = unsafe {
            rows.view().reshape::<Chips, tiles::Half<Width, Head>, tiles::Slices<Width, Head>, m![{ Q }, Half, { Symbol<Row<Head>> }]>()
        };
        // SAFETY: the two output halves are the head in order.
        let rotated = unsafe {
            output.view_mut().reshape::<Chips, tiles::Half<Width, Head>, tiles::Slices<Width, Head>, m![{ Q }, OutHalf, { Symbol<Row<Head>> }]>()
        };
        device.main.begin_interleaved::<OutHalf, _, _, _, _, _>(halves.clone(), halves)
            .fetch::<m![{ Q }, OutHalf, Half, { Symbol<Row<Head>> } / 16], m![{ Symbol<Row<Head>> } % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Q }, OutHalf, Half, { Symbol<Row<Head>> } / 8], m![{ Symbol<Row<Head>> } % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Half, m![{ Q }, OutHalf, { Symbol<Row<Head>> } / 8]>()
            .vector_narrow_split::<m![{ Q }, OutHalf, { Symbol<Row<Head>> } / 4], m![{ Symbol<Row<Head>> } % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), low, ())
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), high)
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_widen_concat::<m![{ Q }, OutHalf, { Symbol<Row<Head>> } / 8], m![{ Symbol<Row<Head>> } % 8]>()
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
        let coefficients = FetchTiles::<Width, Head, R, Chips>(core::marker::PhantomData)
            .forward((device, (self.rotation, at)));
        RotateTile::<Head>(core::marker::PhantomData)
            .forward((device, (rows, &coefficients, output)));
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
        let coefficients = FetchTiles::<Width, Head, R, Chips>(core::marker::PhantomData)
            .forward((device, (self.rotation, at)));
        RotateTile::<Head>(core::marker::PhantomData)
            .forward((device, (first, &coefficients, first_out)));
        RotateTile::<Head>(core::marker::PhantomData)
            .forward((device, (second, &coefficients, second_out)));
    }
}

impl<Head: AxisName, Rotated: AxisName, D: Scalar, Chips: M> Default
    for Rotary<Head, Rotated, D, Chips>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Each of `Rows` rows' rotation over the rotated dimensions: `[cos | sin]` for the first input
/// half and `[-sin | cos]` for the second.
#[derive(DeviceSend)]
pub struct Rotation<
    Rotated: AxisName,
    Chips: M = Chip,
    Rows: AxisName = crate::ops::layout::axes::Single,
> {
    pub low: CoefficientRows<Rows, Rotated, Chips>,
    pub high: CoefficientRows<Rows, Rotated, Chips>,
}

impl<Head: AxisName, Rotated: AxisName, Q: M, K: M, D: MaterializableScalar, Chip: M>
    crate::Module<(
        &mut Device,
        (
            (
                HbmTensor<bf16, Chip, m![{ Q }, Head]>,
                HbmTensor<bf16, Chip, m![{ K }, Head]>,
            ),
            &Rotation<Rotated, Chip>,
        ),
    )> for Rotary<Head, Rotated, D, Chip>
where
    f32: CastEngineCast<D>,
    Convert<D>: DeviceModule<HeadTile<K, Head, bf16, Chip>, Output = HeadTile<K, Head, D, Chip>>,
{
    type Output = (
        HbmTensor<bf16, Chip, m![{ Q }, Head]>,
        HbmTensor<D, Chip, m![{ K }, Head]>,
    );

    fn forward(
        &self,
        (device, ((query, key), rotation)): (
            &mut Device,
            (
                (
                    HbmTensor<bf16, Chip, m![{ Q }, Head]>,
                    HbmTensor<bf16, Chip, m![{ K }, Head]>,
                ),
                &Rotation<Rotated, Chip>,
            ),
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
    Dm<Lanes<Rows>, DmTensor<bf16, Chip, Cluster, Slice, m![{ RowGroups<Rows, Group> }, Head]>>;

/// Each live row's query and key rotated by its own row of the rotation.
impl<Head: AxisName, Rows: AxisName, Slice: M, Query: AxisName, Key: AxisName>
    crate::Module<(
        &mut Device,
        (
            (
                HeadRows<Rows, Slice, Query, Head>,
                HeadRows<Rows, Slice, Key, Head>,
            ),
            &Rotation<Head, Chip, Rows>,
        ),
    )> for Rotary<Head, Head, bf16, Chip>
{
    type Output = (
        HeadRows<Rows, Slice, Query, Head>,
        HeadRows<Rows, Slice, Key, Head>,
    );

    fn forward(
        &self,
        (device, ((query, key), rotation)): (
            &mut Device,
            (
                (
                    HeadRows<Rows, Slice, Query, Head>,
                    HeadRows<Rows, Slice, Key, Head>,
                ),
                &Rotation<Head, Chip, Rows>,
            ),
        ),
    ) -> Self::Output {
        let (rotated_query, rotated_key) = crate::Module::forward(
            &RotaryEmbedding::<Query, Key, Rows>::new(),
            (
                device,
                (
                    &query.value,
                    &key.value,
                    &rotation.low,
                    &rotation.high,
                    query.tiles,
                ),
            ),
        );
        (
            Dm {
                value: rotated_query,
                tiles: query.tiles,
            },
            Dm {
                value: rotated_key,
                tiles: key.tiles,
            },
        )
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
        Convert<Out>: DeviceModule<
                HeadTile<Heads, Head, bf16, Chip>,
                Output = HeadTile<Heads, Head, Out, Chip>,
            >,
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
        heads.view().tile::<
            m![{Symbol<Row<Head>>} / (Rotated / 2)], 1,
            m![Half, {Symbol<Row<Head>>} / (Rotated / 2) = 1 # (Head / Rotated), {Symbol<Row<Head>>} % (Rotated / 2)]
        >(0).to_dm_view(&mut device.tdma, prefix.view_mut());
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
            m![1, 1, OutHalf, {Symbol<Row<Rotated>>}],
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
            m![1, 1, OutHalf, Half, {Symbol<Row<Rotated>>}],
        > = DmTensor::new();
        low.view().to_dm_view(
            &mut device.tdma,
            factors
                .view_mut()
                .tile::<m![Half], 1, m![1, 1, OutHalf, Half = 1 #{!} 2, {Symbol<Row<Rotated>>}]>(0),
        );
        high.view().to_dm_view(
            &mut device.tdma,
            factors
                .view_mut()
                .tile::<m![Half], 1, m![1, 1, OutHalf, Half = 1 #{!} 2, {Symbol<Row<Rotated>>}]>(1),
        );
        let factors: VrfTensor<f32, Chip, Cluster, HeadSlices<Heads>, m![1, 1, OutHalf, Half, {Symbol<Row<Rotated>>}]> = device.sub.begin(factors.view())
            .fetch::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 16], m![{Symbol<Row<Rotated>>} % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 8], m![{Symbol<Row<Rotated>>} % 8]>()
            .to_vrf();
        let mut values: DmTensor<
            bf16,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![1, 1, OutHalf, Half, {Symbol<Row<Rotated>>}],
        > = DmTensor::new();
        prefix
            .view()
            .to_dm_view(&mut device.tdma, values.view_mut());
        let products: DmTensor<bf16, Chip, Cluster, HeadSlices<Heads>, m![1, 1, OutHalf, Half, {Symbol<Row<Rotated>>}]> = device
            .main
            .begin(values.view())
            .fetch::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 16], m![{Symbol<Row<Rotated>>} % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 8], m![{Symbol<Row<Rotated>>} % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 4], m![{Symbol<Row<Rotated>>} % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factors)
            .vector_widen_concat::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 8], m![{Symbol<Row<Rotated>>} % 8]>()
            .vector_final()
            .cast::<bf16, m![{Symbol<Row<Rotated>>} % 8 # 16]>()
            .commit_trim::<m![{Symbol<Row<Rotated>>} % 8]>()
            .commit();
        device.main.begin(products.view())
            .fetch::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 16], m![{Symbol<Row<Rotated>>} % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 8], m![{Symbol<Row<Rotated>>} % 8]>()
            .vector_init()
            .vector_zip::<
                Half,
                m![OutHalf, {Symbol<Row<Rotated>>} / 8],
                m![OutHalf, {Symbol<Row<Rotated>>} / 4],
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
        let heads = crate::Module::forward(&Convert::<Out>::default(), (device, heads));
        // SAFETY: recover the same consecutive halves after element conversion.
        let mut heads: DmTensor<
            Out,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![Half, {Symbol<Row<Head>>}],
        > = unsafe { heads.reshape() };
        output.view().to_dm_view(&mut device.tdma, heads.view_mut().tile::<
            m![{Symbol<Row<Head>>} / (Rotated / 2)], 1,
            m![Half, {Symbol<Row<Head>>} / (Rotated / 2) = 1 #{!} (Head / Rotated), {Symbol<Row<Head>>} % (Rotated / 2)]
        >(0));
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
pub struct Adjacent<Head>(pub core::marker::PhantomData<Head>);

impl<Head> Default for Adjacent<Head> {
    fn default() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<H: AxisName, W: AxisName, R: AxisName> crate::Module<(&mut Device, (Qkv<W, R>, &Phases<R, H>))>
    for Adjacent<H>
{
    type Output = Qkv<W, R>;

    fn forward(&self, _: (&mut Device, (Qkv<W, R>, &Phases<R, H>))) -> Self::Output {
        const { assert!(H::SIZE > 0 && H::SIZE % 2 == 0 && W::SIZE % H::SIZE == 0) };
        todo!(
            "NPU adjacent-pair query/key rotation using caller-supplied cosine and sine per token"
        )
    }
}

/// Independent coordinate axes occupy consecutive pairs of rotary channels.
#[derive(DeviceSend)]
pub struct Axial<Head, First, Second, Third> {
    pub theta: HbmTensor<f32, m![1], m![1]>,
    pub dimensions: std::marker::PhantomData<(Head, First, Second, Third)>,
}

impl<H: AxisName, A: AxisName, B: AxisName, C: AxisName, R: AxisName>
    crate::Module<(&mut Device, &HbmTensor<f32, m![1], m![R, 3]>)> for Axial<H, A, B, C>
{
    type Output = Phases<R, H>;

    fn forward(&self, _: (&mut Device, &HbmTensor<f32, m![1], m![R, 3]>)) -> Self::Output {
        const {
            assert!(
                H::SIZE == A::SIZE + B::SIZE + C::SIZE
                    && A::SIZE % 2 == 0
                    && B::SIZE % 2 == 0
                    && C::SIZE % 2 == 0
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

impl<H: AxisName, R: AxisName> crate::Module<(&mut Device, &HbmTensor<i32, m![1], m![R]>)>
    for Frequencies<H>
{
    type Output = Phases<R, H>;

    fn forward(&self, _: (&mut Device, &HbmTensor<i32, m![1], m![R]>)) -> Self::Output {
        todo!("NPU amplitude * (cos, sin)(position * inverse frequency) for every position")
    }
}

impl<H: AxisName, Q: AxisName, K: AxisName>
    crate::Module<(
        &mut Device,
        (
            &HbmTensor<bf16, m![1], m![Q, H]>,
            &HbmTensor<bf16, m![1], m![K, H]>,
            &Phases<crate::ops::layout::axes::Single, H>,
        ),
    )> for Adjacent<H>
{
    type Output = (
        HbmTensor<bf16, m![1], m![Q, H]>,
        HbmTensor<bf16, m![1], m![K, H]>,
    );

    fn forward(
        &self,
        _: (
            &mut Device,
            (
                &HbmTensor<bf16, m![1], m![Q, H]>,
                &HbmTensor<bf16, m![1], m![K, H]>,
                &Phases<crate::ops::layout::axes::Single, H>,
            ),
        ),
    ) -> Self::Output {
        const { assert!(H::SIZE > 0 && H::SIZE.is_multiple_of(2)) };
        todo!("NPU adjacent-pair rotary transform, sharing one position across query and key heads")
    }
}

/// Half-split rotary pairs with a load-time frequency spectrum.
#[derive(DeviceSend)]
pub struct SplitHalf<Head: AxisName, Frequency = HbmTensor<f32, m![1], m![Head / 2]>> {
    pub frequency: Frequency,
    pub head: core::marker::PhantomData<Head>,
}

impl<D: Scalar, H: AxisName, Q: AxisName, K: AxisName, R: AxisName>
    crate::Module<(
        &mut Device,
        (
            (HbmTensor<D, m![1], m![R, Q]>, HbmTensor<D, m![1], m![R, K]>),
            &HbmTensor<i32, m![1], m![R]>,
        ),
    )> for SplitHalf<H>
{
    type Output = (HbmTensor<D, m![1], m![R, Q]>, HbmTensor<D, m![1], m![R, K]>);

    fn forward(
        &self,
        _: (&mut Device, (Self::Output, &HbmTensor<i32, m![1], m![R]>)),
    ) -> Self::Output {
        const {
            assert!(
                H::SIZE > 0 && H::SIZE % 2 == 0 && Q::SIZE % H::SIZE == 0 && K::SIZE % H::SIZE == 0
            );
        }
        todo!(
            "NPU half-split per-head rotary: angle=position*frequency, FP32 sine/cosine and products, source-dtype output"
        )
    }
}

/// The frequency of each of the `R / 2` rotated pairs, in radians per position, positive and
/// finite in FP32; coordinates `i` and `i + R / 2` turn at pair `i`'s.
#[derive(Debug)]
pub struct PairFrequencies<R>(Vec<f32>, core::marker::PhantomData<fn() -> R>);

impl<R> Clone for PairFrequencies<R> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), core::marker::PhantomData)
    }
}

impl<R> PartialEq for PairFrequencies<R> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<R: AxisName> PairFrequencies<R> {
    /// The frequencies of `pairs`, rounded to FP32; another count than `R / 2`, or a frequency
    /// FP32 does not hold as a positive finite value, is refused.
    pub fn new(pairs: &[f64]) -> std::io::Result<Self> {
        let rounded: Vec<f32> = pairs.iter().map(|&frequency| frequency as f32).collect();
        match pairs.len() * 2 == R::SIZE && rounded.iter().all(|f| f.is_normal() && *f > 0.0) {
            true => Ok(Self(rounded, core::marker::PhantomData)),
            false => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!(
                    "rotary frequencies: found {pairs:?}, compiled {} positive finite pairs",
                    R::SIZE / 2
                ),
            )),
        }
    }

    /// The unscaled spectrum of base `theta` over a `Head`-wide head: pair `i` turns at
    /// `theta^(-2i / Head)`.
    pub fn unscaled<Head: AxisName>(theta: f32) -> std::io::Result<Self> {
        let theta = f64::from(theta);
        let pairs: Vec<f64> = (0..R::SIZE / 2)
            .map(|pair| theta.powf(-2.0 * pair as f64 / Head::SIZE as f64))
            .collect();
        Self::new(&pairs)
    }

    /// The frequency of each pair.
    pub fn pairs(&self) -> &[f32] {
        &self.0
    }

    /// The frequency of each of the `R` coordinates.
    pub fn coordinates(&self) -> Vec<f32> {
        [self.0.as_slice(), self.0.as_slice()].concat()
    }
}

#[cfg(test)]
mod frequencies {
    use super::super::tests::Head;
    use super::PairFrequencies;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;
    use std::io::ErrorKind;

    /// A spectrum binds the FP32 frequency of each of its pairs, the first pair's too, from the
    /// largest to the smallest one FP32 holds; another pair count, or a frequency FP32 rounds to
    /// zero or infinity, is refused.
    #[test]
    fn binds_every_pair_frequency() {
        let pairs = |first: f64, last: f64| {
            let mut pairs = vec![1.0; Head::SIZE / 2];
            (pairs[0], pairs[Head::SIZE / 2 - 1]) = (first, last);
            pairs
        };
        let held = PairFrequencies::<Head>::new(&pairs(0.5, 1e-30)).unwrap();
        assert_eq!(
            (held.pairs()[0], held.pairs()[Head::SIZE / 2 - 1]),
            (0.5, 1e-30)
        );
        assert_eq!(held.coordinates().len(), Head::SIZE);
        for refused in [
            pairs(1.0, 1e-50),
            pairs(1e40, 1.0),
            pairs(0.0, 1.0),
            vec![1.0; 3],
        ] {
            let refusal = PairFrequencies::<Head>::new(&refused).unwrap_err();
            assert_eq!(refusal.kind(), ErrorKind::Unsupported, "{refusal}");
        }
    }

    /// A rescaled spectrum's bound frequencies land its coefficients on device at the first
    /// position, on either side of its trained context and at the last admitted one, a spectrum
    /// whose first pair turns at half speed included; its unscaled spectrum misses them past the
    /// trained context.
    #[tokio::test(flavor = "current_thread")]
    async fn lands_rescaled_coefficients() {
        use crate::models::config::{parse, scaling::Spectrum};
        use crate::ops::layout::axes::Eight;
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
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(super::tests::positioned.topology()).unwrap();
        let coefficients = super::tests::coefficients::<Head>(&mut device).await;
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
            let unscaled = PairFrequencies::<Head>::unscaled::<Head>(spectrum.theta.get()).unwrap();
            for (bound, lands) in [
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
                let landed = positions.iter().enumerate().all(|(row, &at)| {
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
                assert_eq!(landed, lands, "{document}: {bound:?}");
            }
        }
    }
}

/// Each live row's queries times the FP32 temperature of its own position, rounded to BF16 once;
/// the keys, and so the cache, keep their values.
impl<'s, Head: AxisName, Rows: AxisName, Slice: M, Query: AxisName, K>
    crate::Module<(
        &mut Device,
        (
            (HeadRows<Rows, Slice, Query, Head>, K),
            &'s HbmTensor<f32, Chip, m![Rows, 1 # 8]>,
        ),
    )> for crate::ops::multiply::QueryScale
{
    type Output = (HeadRows<Rows, Slice, Query, Head>, K);

    fn forward(
        &self,
        (device, ((query, key), factors)): (
            &mut Device,
            (
                (HeadRows<Rows, Slice, Query, Head>, K),
                &'s HbmTensor<f32, Chip, m![Rows, 1 # 8]>,
            ),
        ),
    ) -> Self::Output {
        type Grouped<Rows, Query, Head> = m![{ Groups<Rows> }, Query, Head];
        let factors: DmTensor<f32, Chip, Cluster, Slice, m![{ Groups<Rows> }, 1 # 8]> =
            factors.to_dm(&mut device.tdma);
        // SAFETY: a head row's group and query coordinates only flatten into one element axis.
        let rows = unsafe {
            query
                .value
                .view()
                .reshape::<Chip, Cluster, Slice, Grouped<Rows, Query, Head>>()
        };
        let mut scaled: DmTensor<bf16, Chip, Cluster, Slice, Grouped<Rows, Query, Head>> =
            DmTensor::new();
        for group in 0..query.tiles.count() {
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
        (
            Dm {
                value,
                tiles: query.tiles,
            },
            key,
        )
    }
}
