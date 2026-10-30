//! RMS norms of packed rows whose gain and shift each row's modality and timestep select: the
//! tiles they move rows in, beside one file per tiling.
//!
//! - A tile spreads its features over the slices and its two halves over the clusters; its row
//!   statistics reduce across slices, and per-feature operands stay in the slices that use them.

use crate::Op;
use crate::load::Load;
use crate::ops::index::packed::{Packed, Rows, Table};
use crate::ops::normalization::{Lanes, Root, Scale, Squares};
use crate::tensor::form::Weight;
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;
use std::marker::PhantomData;

pub(crate) mod blocks;
pub(crate) mod rows;

axes![Strip = 64];

/// The tiles of a packed sequence.
#[derive(Clone, Debug)]
pub(crate) struct Strips<R>(PhantomData<R>);

impl<R: AxisName> AxisName for Strips<R> {
    const NAME: Ident = Ident::new("Strips");
    const SIZE: usize = R::SIZE / Strip::SIZE;
}

pub(crate) type Banded<R, W> = m![{ Symbol<Strips<R>> }, Strip, W];
pub(crate) type Half = m![Strip / 32];
/// Four groups of eight rows, each over the slices of its features 128 apiece: every slice moves
/// 256-byte runs of a row and the row statistics reduce over the inner digit.
pub(crate) type Slices<W> = m![Strip % 32 / 8, W / 128 # 64];
pub(crate) type Tile<D, W, X = m![1]> = DmTensor<D, X, Half, Slices<W>, m![Strip % 8, W % 128]>;
pub(crate) type Feature<D, W, X = m![1]> = DmTensor<D, X, Half, Slices<W>, m![W % 128]>;
pub(crate) type Operand<W, X = m![1]> = VrfTensor<f32, X, Half, Slices<W>, m![W % 128]>;
type RowOperand<W, X = m![1]> = VrfTensor<f32, X, Half, Slices<W>, m![Strip % 8, 1 # 8]>;
/// How a tile's rows move through a norm's passes.
type StripLanes<W> = Lanes<
    m![Strip % 8, W % 128 / 8],
    m![W % 8],
    m![Strip % 8, W % 128 / 8, W % 8 / 4 % 2],
    m![W % 4],
>;

impl<R: AxisName> Strips<R> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<R: AxisName, W: AxisName, X: M> Op<(&Tile<bf16, W, X>, &mut Rows<R, W, X>, usize)>
    for Strips<R>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (tile, rows, at): (&Tile<bf16, W, X>, &mut Rows<R, W, X>, usize),
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

/// A written per-feature vector, widened.
impl<Width: AxisName, const EPSILON: usize, X: M, T> Op<&Feature<bf16, Width, X>>
    for FinalNorm<Width, EPSILON, X, T>
{
    type Output = Operand<Width, X>;

    fn forward(&self, device: &mut Device, in_dm: &Feature<bf16, Width, X>) -> Self::Output {
        device
            .sub
            .begin(in_dm.view())
            .fetch::<m![Width % 128 / 8], m![Width % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Width % 128 / 8], m![Width % 8]>()
            .to_vrf()
    }
}

/// `bf16(x * rsqrt(mean(x²) + eps) * (w * (1 + scale)) + shift)` with each span's table row, FP32
/// inside, `EPSILON` epsilon's FP32 bits and the gain `w` the checkpoint's `weight`, held as `T`.
#[derive(DeviceSend, Load)]
pub struct AdaptiveRms<
    Width: AxisName,
    const EPSILON: usize,
    Chips: M = m![1],
    T = HbmTensor<bf16, Chips, m![Width]>,
> {
    weight: Weight<m![Width], Chips, T>,
}

impl<Width: AxisName, const EPSILON: usize, Chips: M> AdaptiveRms<Width, EPSILON, Chips> {
    pub const fn new(weight: Weight<m![Width], Chips>) -> Self {
        Self { weight }
    }
}

impl<
    'x,
    'm,
    T: AxisName,
    A: AxisName,
    V: AxisName,
    Width: AxisName,
    Chips: M,
    const EPSILON: usize,
    G,
>
    Op<(
        &'x Rows<Packed<T, A, V>, Width, Chips>,
        &'m (Table<Width, Chips>, Table<Width, Chips>),
    )> for AdaptiveRms<Width, EPSILON, Chips, G>
where
    G: Borrow<HbmTensor<bf16, Chips, m![Width]>>,
{
    type Output = Rows<Packed<T, A, V>, Width, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (input, modulation): (
            &'x Rows<Packed<T, A, V>, Width, Chips>,
            &'m (Table<Width, Chips>, Table<Width, Chips>),
        ),
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        self.forward(device, (input, &mut output, modulation));
        output
    }
}

/// One modality's final norm: RMS normalization rounded to BF16, then `bf16(n * (1 + scale) +
/// shift)` of its timestep row, `EPSILON` epsilon's FP32 bits and the gain `weight` held as `T`.
#[derive(DeviceSend, Load)]
pub struct FinalNorm<
    Width: AxisName,
    const EPSILON: usize,
    X: M = m![1],
    T = HbmTensor<bf16, X, m![Width]>,
> {
    weight: Weight<m![Width], X, T>,
}

impl<Width: AxisName, const EPSILON: usize, X: M> FinalNorm<Width, EPSILON, X> {
    pub const fn new(weight: Weight<m![Width], X>) -> Self {
        Self { weight }
    }
}

/// One timestep's `(shift, scale)`.
pub type Shift<W, Chips = m![1]> = (HbmTensor<bf16, Chips, m![W]>, HbmTensor<bf16, Chips, m![W]>);

impl<'x, 's, R: AxisName, Width: AxisName, const EPSILON: usize, X: M, T>
    Op<(&'x Rows<R, Width, X>, &'s Shift<Width, X>)> for FinalNorm<Width, EPSILON, X, T>
where
    T: Borrow<HbmTensor<bf16, X, m![Width]>>,
{
    type Output = Rows<R, Width, X>;

    fn forward(
        &self,
        device: &mut Device,
        (input, (shift, scale)): (&'x Rows<R, Width, X>, &'s Shift<Width, X>),
    ) -> Self::Output {
        let weight: Feature<bf16, Width, X> = self.weight.tensor().view().to_dm(&mut device.tdma);
        let weight = self.forward(&mut *device, &weight);
        let scale: Feature<bf16, Width, X> = scale.view().to_dm(&mut device.tdma);
        let gain: Operand<Width, X> = device
            .main
            .begin(scale.view())
            .fetch::<m![Width % 128 / 8], m![Width % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Width % 128 / 8], m![Width % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Width % 128 / 8, Width % 8 / 4 % 2], m![Width % 4]>()
            .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)
            .vector_widen_concat::<m![Width % 128 / 8], m![Width % 8]>()
            .vector_final()
            .to_vrf(&mut device.sub);
        let shift: Feature<bf16, Width, X> = shift.view().to_dm(&mut device.tdma);
        let shift = self.forward(&mut *device, &shift);
        let mut output = HbmTensor::new();
        for tile in 0..Strips::<R>::SIZE {
            let rows: Tile<bf16, Width, X> = {
                // SAFETY: the tiles split the row axis in row order; every cell keeps its address.
                unsafe { input.view().reshape::<X, Banded<R, Width>>() }
                    .tile::<m![{ Symbol<Strips<R>> }], 1, m![
                        { Symbol<Strips<R>> } = 1 # ({ Symbol<Strips<R>> }), Strip, Width
                    ]>(tile)
                    .to_dm(&mut device.tdma)
            };
            let inverse: RowOperand<Width, X> = {
                let variance: DmTensor<f32, X, Half, m![Strip % 32 / 8, 64], m![Strip % 8, 1 # 8]> =
                    Squares::<
                        StripLanes<Width>,
                        Width,
                        m![Strip % 8],
                        m![Width / 128],
                        EPSILON,
                    >::of(
                        &mut device.main,
                        rows.view(),
                    )
                    .vector_inter_slice_reduce::<m![Strip % 32 / 8, 64], m![Strip % 8]>(
                        InterSliceReduceOpF32::Add,
                    )
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                // SAFETY: every slice holds the same row statistic; each row keeps its address.
                let variance: DmTensor<f32, X, Half, Slices<Width>, m![Strip % 8, 1 # 8]> =
                    unsafe { variance.reshape() };
                Root::<m![Strip % 8], m![1 # 8]>::of(&mut device.sub, variance.view(), |square| {
                    square
                })
                .to_vrf()
            };
            let normalized: Tile<bf16, Width, X> =
                Scale::<StripLanes<Width>>::of(&mut device.main, rows.view(), &inverse, &weight)
                    .commit();
            let modulated: Tile<bf16, Width, X> = device
                .main
                .begin(normalized.view())
                .fetch::<m![Strip % 8, Width % 128 / 8], m![Width % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Strip % 8, Width % 128 / 8], m![Width % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<
                    m![Strip % 8, Width % 128 / 8, Width % 8 / 4 % 2],
                    m![Width % 4],
                >()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gain)
                .vector_fp_binary(FpBinaryOp::AddF, &shift)
                .vector_widen_concat::<m![Strip % 8, Width % 128 / 8], m![Width % 8]>()
                .vector_final()
                .cast::<bf16, m![Width % 8 # 16]>()
                .commit_trim::<m![Width % 8]>()
                .commit();
            Strips::<R>::new().forward(&mut *device, (&modulated, &mut output, tile));
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::index::packed::Group;
    use crate::ops::normalization::tests::random;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    struct Fixture;

    axes![Text = 64, Audio = 64, Video = 128, Hidden = 5376];

    type Sequence = Packed<Text, Audio, Video>;

    const EPSILON: usize = 1e-5_f32.to_bits() as usize;

    impl Fixture {
        async fn span_rows_modulate<T: AxisName, A: AxisName, V: AxisName>() {
            let _lease = DEVICE.lock().await;
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

    axes![
        TextEnvelope = 128,
        AudioEnvelope = 448,
        ShortVideo = 384,
        LongVideo = 960,
        MidVideo = 704
    ];

    #[device(chip = 1)]
    fn modulated<T: AxisName, A: AxisName, V: AxisName>(
        device: &mut Device,
        norm: &AdaptiveRms<Hidden, EPSILON>,
        input: &Rows<Packed<T, A, V>, Hidden>,
        modulation: &(Table<Hidden>, Table<Hidden>),
    ) -> Rows<Packed<T, A, V>, Hidden> {
        norm.forward(device, (input, modulation))
    }

    /// Every row is normalized and modulated by its span's table row, as the FP64 definition
    /// rounded once; rows span four magnitudes, over spans of strips and of whole blocks.
    #[tokio::test(flavor = "current_thread")]
    async fn spans_take_their_modality_row() {
        Fixture::span_rows_modulate::<Text, Audio, ShortVideo>().await;
        Fixture::span_rows_modulate::<TextEnvelope, AudioEnvelope, LongVideo>().await;
        Fixture::span_rows_modulate::<TextEnvelope, AudioEnvelope, MidVideo>().await;
    }

    #[device(chip = 1)]
    fn final_norm(
        device: &mut Device,
        norm: &FinalNorm<Hidden, EPSILON>,
        input: &Rows<Sequence, Hidden>,
        modulation: &Shift<Hidden>,
    ) -> Rows<Sequence, Hidden> {
        norm.forward(device, (input, modulation))
    }
    /// Each row's RMS normalization rounds to BF16 before its timestep's modulation, which rounds
    /// again: both roundings as the host computes them, within one BF16 step of a near tie.
    #[tokio::test(flavor = "current_thread")]
    async fn rounds_the_norm_before_the_modulation() {
        let _lease = DEVICE.lock().await;
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
}
