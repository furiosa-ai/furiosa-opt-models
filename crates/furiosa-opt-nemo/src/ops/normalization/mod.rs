//! Per-row normalization with learned feature scales and FP32 statistics.

mod gain;
mod kernel;
pub(crate) mod span;
pub mod staging;

pub use gain::{Centered, Epsilon, Gain};

use furiosa_opt_std::prelude::*;

use crate::compose::Space;
use crate::ops::layout::{Dm, axes::Single};
use crate::{Module, ops::weight::Weight};

/// Feature slices retain every row in both clusters for the projections that consume them.
pub type Cluster = m![2];

/// Smallest multiple of 16 dividing Width into at most 256 hardware slices.
#[derive(Clone, Debug)]
pub struct Chunk<Width>(core::marker::PhantomData<Width>);

impl<Width: AxisName> AxisName for Chunk<Width> {
    const NAME: Ident = Ident::new("Chunk");

    const SIZE: usize = {
        assert!(Width::SIZE > 0 && Width::SIZE.is_multiple_of(16));
        let mut chunk = 16;
        while Width::SIZE / chunk > 256 || Width::SIZE % chunk != 0 {
            chunk += 16;
        }
        chunk
    };
}

/// Feature chunks broadcast across the 256 hardware slices.
pub type FeatureSlices<Width> = m![Width / ({ Symbol<Chunk<Width>> }) # 256];

/// Each row's contiguous features within one chunk.
pub type Element<Width, Rows> = m![Rows, Width % ({ Symbol<Chunk<Width>> })];

/// Launch buffers keep each hidden row contiguous.
pub type RowMajor<Width, Rows> = m![Rows, Width];

/// Launch rows stored sixteen features at a time: each block holds those features of every row.
pub(crate) type Blocked<Width, Rows> = m![Width / 16, Rows, Width % 16];

/// Hidden rows share the feature placement of the following projections.
pub type Spread<Width, Rows, Chips = m![1]> =
    DmTensor<bf16, Chips, Cluster, FeatureSlices<Width>, Element<Width, Rows>>;

/// FP32 bits of the epsilon an [`RmsNorm`] adds when its type names none.
pub const EPSILON: usize = 1e-6_f32.to_bits() as usize;

/// BF16 rows times learned weights, divided by sqrt(mean(x²) + epsilon).
/// EPSILON stores the FP32 bit pattern in a usize const parameter.
#[derive(DeviceSend)]
pub struct RmsNorm<
    Width: AxisName,
    const EPSILON: usize = { self::EPSILON },
    Chips: M = m![1],
    Parameter = Weight<m![Width], Chips>,
> {
    weight: Parameter,
    features: core::marker::PhantomData<(Width, Chips)>,
}

impl<Width: AxisName, const EPSILON: usize, Chips: M, Parameter>
    RmsNorm<Width, EPSILON, Chips, Parameter>
{
    /// Retain one learned BF16 scale per feature on the same chip mapping as the input.
    pub const fn new(weight: Parameter) -> Self {
        Self {
            weight,
            features: core::marker::PhantomData,
        }
    }

    /// Borrow the retained per-feature scale.
    pub fn weight(&self) -> &Parameter {
        &self.weight
    }

    pub fn weight_mut(&mut self) -> &mut Parameter {
        &mut self.weight
    }
}

/// A norm's learned per-feature scale, in a form the norm reads into each feature slice.
pub trait Learned<Width: AxisName, Chips: M> {
    fn lanes(&self, device: &mut Device) -> kernel::Features<Width, Chips>;
}

/// A norm gain landed in the feature slices of the rows it scales.
pub type Landed<Width, Chips = m![1]> =
    DmTensor<bf16, Chips, Cluster, FeatureSlices<Width>, m![Width % ({ Symbol<Chunk<Width>> })]>;

impl<Width: AxisName, Chips: M> Learned<Width, Chips> for Landed<Width, Chips> {
    fn lanes(&self, device: &mut Device) -> kernel::Features<Width, Chips> {
        device
            .sub
            .begin(self.view())
            .fetch::<m![Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
            .to_vrf()
    }
}

/// The retained HBM gain lands in the slices each time a norm reads it.
impl<Width: AxisName, Chips: M> Learned<Width, Chips> for Weight<m![Width], Chips> {
    fn lanes(&self, device: &mut Device) -> kernel::Features<Width, Chips> {
        let landed: Landed<Width, Chips> = self.tensor().to_dm(&mut device.tdma);
        landed.lanes(device)
    }
}

impl<Width: AxisName, const EPSILON: usize, Shape: M, Chips: M>
    RmsNorm<Width, EPSILON, Chips, Weight<Shape, Chips>>
{
    /// The same norm with its retained gain landed once in the DM mapping its forward reads.
    pub fn stage<Cl: M, S: M, E: M>(
        &self,
        device: &mut Device,
    ) -> RmsNorm<Width, EPSILON, Chips, DmTensor<bf16, Chips, Cl, S, E>> {
        RmsNorm::new(self.weight.tensor().to_dm(&mut device.tdma))
    }
}

/// An [`RmsNorm`] of unit gain, whose checkpoint stores no weight: it multiplies by a placed vector
/// of ones, so it runs the learned norm's kernel.
#[derive(DeviceSend)]
pub struct UnitRmsNorm<Width: AxisName, const EPSILON: usize = { self::EPSILON }, Chips: M = m![1]>(
    RmsNorm<Width, EPSILON, Chips>,
);

impl<Width: AxisName, const EPSILON: usize, Chips: M> UnitRmsNorm<Width, EPSILON, Chips> {
    /// The norm multiplying by `ones`, which must hold one in every feature.
    pub const fn new(ones: Weight<m![Width], Chips>) -> Self {
        Self(RmsNorm::new(ones))
    }
}

impl<I, Width: AxisName, const EPSILON: usize, Chips: M> Module<I>
    for UnitRmsNorm<Width, EPSILON, Chips>
where
    RmsNorm<Width, EPSILON, Chips>: Module<I>,
{
    type Output = <RmsNorm<Width, EPSILON, Chips> as Module<I>>::Output;

    fn forward(&self, input: I) -> Self::Output {
        self.0.forward(input)
    }
}

impl<'d, 'x, Width: AxisName, Rows: AxisName, const EPSILON: usize, Chips: M, G>
    Module<(&'d mut Device, &'x Spread<Width, Rows, Chips>)> for RmsNorm<Width, EPSILON, Chips, G>
where
    G: Learned<Width, Chips>,
{
    type Output = Spread<Width, Rows, Chips>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Spread<Width, Rows, Chips>),
    ) -> Self::Output {
        kernel::rms_norm::<Width, m![Rows], EPSILON, Chips, _>(device, input.view(), &self.weight)
            .commit()
    }
}

/// Every row of the envelope is normalized alone, so the live rows land what they land in a
/// launch of their own.
impl<'d, 'x, Width: AxisName, S: Space, const EPSILON: usize, Chips: M, G>
    Module<(&'d mut Device, &'x Dm<S, Spread<Width, S::Envelope, Chips>>)>
    for RmsNorm<Width, EPSILON, Chips, G>
where
    G: Learned<Width, Chips>,
{
    type Output = Dm<S, Spread<Width, S::Envelope, Chips>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Dm<S, Spread<Width, S::Envelope, Chips>>),
    ) -> Self::Output {
        Dm {
            value: self.forward((device, &input.value)),
            tiles: input.tiles,
        }
    }
}

impl<Width: AxisName, const EPSILON: usize, Chips: M>
    Module<(&mut Device, &HbmTensor<bf16, Chips, m![Width]>)> for RmsNorm<Width, EPSILON, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Width]>;

    fn forward(
        &self,
        (device, input): (&mut Device, &HbmTensor<bf16, Chips, m![Width]>),
    ) -> Self::Output {
        let input: Spread<Width, Single, Chips> = input.to_dm(&mut device.tdma);
        self.forward((&mut *device, &input))
            .to_hbm(&mut device.tdma)
    }
}

impl<Width: AxisName, Rows: AxisName, const EPSILON: usize, Chips: M>
    Module<(&mut Device, &HbmTensor<bf16, Chips, m![Rows, Width]>)>
    for RmsNorm<Width, EPSILON, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Rows, Width]>;

    fn forward(
        &self,
        (device, input): (&mut Device, &HbmTensor<bf16, Chips, m![Rows, Width]>),
    ) -> Self::Output {
        let input: Spread<Width, Rows, Chips> = input.to_dm(&mut device.tdma);
        self.forward((&mut *device, &input))
            .to_hbm(&mut device.tdma)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::Module;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use crate::compose::{Count, tests::count};
    use crate::ops::attention::geometry::{Hidden, QueryGroups};
    use crate::ops::attention::{
        Heads,
        tests::{Head, KvHeads},
    };
    use crate::ops::layout::{
        Dm, Live,
        axes::{Eight, Single, Sixteen},
    };
    use crate::ops::normalization::RowMajor;
    use crate::{ops::normalization::RmsNorm, ops::weight::Weight};

    type Chip = m![1];
    type Rows<Width, Count> = HbmTensor<bf16, Chip, RowMajor<Width, Count>>;

    /// Row `row` scaled by `scale * 2^row`, so a variance shared across rows misnormalizes every
    /// row but one.
    fn values(rows: usize, width: usize, scale: f32) -> Vec<bf16> {
        (0..rows * width)
            .map(|index| {
                let (row, column) = (index / width, index % width);
                bf16::from_f32(
                    (((column * 37) % 101) as f32 / 25.0 - 2.0) * scale * (1 << row) as f32,
                )
            })
            .collect()
    }

    fn scales(width: usize) -> Vec<bf16> {
        (0..width)
            .map(|index| bf16::from_f32(((index * 13) % 29) as f32 / 24.0 - 0.6))
            .collect()
    }

    /// Each row of `weight.len()` values times its weights over the root of its FP32 mean square
    /// plus epsilon.
    fn definition(rows: &[bf16], weight: &[bf16]) -> Vec<f32> {
        rows.chunks_exact(weight.len())
            .flat_map(|row| {
                let mean =
                    row.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>() / weight.len() as f32;
                let inverse = 1.0 / (mean + 1e-6).sqrt();
                row.iter()
                    .zip(weight)
                    .map(move |(v, w)| v.to_f32() * inverse * w.to_f32())
            })
            .collect()
    }

    /// Every value within 2% of the definition's, or of one where that is smaller.
    fn assert_lands(actual: &[bf16], expected: &[f32], width: usize, scale: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (got, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (got.to_f32() - expected).abs() <= 0.02 * expected.abs().max(1.0),
                "scale {scale} row {} column {}: {} vs {expected}",
                index / width,
                index % width,
                got.to_f32(),
            );
        }
    }

    /// Pseudo-random values in `[-1, 1)`, rounded to what the device reads.
    pub(crate) fn random(count: usize, mut state: u64) -> Vec<bf16> {
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                bf16::from_f32((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
            })
            .collect()
    }

    /// Row-major `[rows, width]` values into `[width / block, rows, width % block]`, the layout
    /// the kernels move rows in, and back.
    pub(crate) fn blocked<T: Copy>(rows: &[T], width: usize, block: usize) -> Vec<T> {
        let count = rows.len() / width;
        let mut output = Vec::with_capacity(rows.len());
        for chunk in 0..width / block {
            for row in 0..count {
                output.extend_from_slice(&rows[row * width + chunk * block..][..block]);
            }
        }
        output
    }

    #[device(chip = 1)]
    fn normalized<Width: AxisName, Count: AxisName>(
        device: &mut Device,
        norm: &RmsNorm<Width>,
        input: &Rows<Width, Count>,
        output: &mut Rows<Width, Count>,
    ) {
        let input: super::Spread<Width, Count> = input.to_dm(&mut device.tdma);
        norm.forward((&mut *device, &input))
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    /// Against the definition on the host, per row. The small scale shrinks the mean square to
    /// epsilon's order, so an epsilon added once per slice instead of once overall fails loudly.
    async fn matches_the_definition_over<Width: AxisName, Count: AxisName>() {
        let mut device = Device::new(normalized.topology()).unwrap();
        for scale in [1.0f32, 1e-3] {
            let values = values(Count::SIZE, Width::SIZE, scale);
            let scales = scales(Width::SIZE);
            let expected = definition(&values, &scales);
            let input = HostTensor::<bf16, RowMajor<Width, Count>>::from_vec(values)
                .to_hbm::<Chip, RowMajor<Width, Count>>(&mut device.pdma)
                .await
                .unwrap();
            let norm = RmsNorm::new(Weight::new(
                HostTensor::<bf16, m![Width]>::from_vec(scales)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ));
            let mut actual = HostTensor::<bf16, RowMajor<Width, Count>>::zero()
                .to_hbm::<Chip, RowMajor<Width, Count>>(&mut device.pdma)
                .await
                .unwrap();
            launch(normalized, (&mut device, &norm, &input, &mut actual))
                .await
                .unwrap();
            let actual = actual
                .to_host::<RowMajor<Width, Count>>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            assert_lands(&actual, &expected, Width::SIZE, scale);
        }
    }

    /// Per-row RMS normalization matches FP32 statistics when epsilon dominates the variance.
    #[tokio::test]
    async fn matches_the_definition() {
        matches_the_definition_over::<Hidden<1024>, Eight>().await;
    }

    /// Sixteen rows preserve independent statistics across all 256 feature slices.
    #[tokio::test]
    async fn sixteen_rows_match_the_definition() {
        matches_the_definition_over::<Hidden<4096>, Sixteen>().await;
    }

    #[device(chip = 1)]
    fn heads<Group: AxisName>(
        device: &mut Device,
        norm: &RmsNorm<Head>,
        live: &Count,
        input: &HbmTensor<bf16, Chip, m![Sixteen, KvHeads, Group, Head]>,
    ) -> HbmTensor<bf16, Chip, m![Sixteen, KvHeads, Group, Head]> {
        let input: Dm<_, Heads<Sixteen, KvHeads, Group, Head>> = Dm {
            value: input.to_dm(&mut device.tdma),
            tiles: Live::<Sixteen>::read(live),
        };
        norm.forward((&mut *device, &input))
            .value
            .to_hbm(&mut device.tdma)
    }

    /// Every head row of `Group` heads per key/value head, in the slices a projection lands them
    /// in, against the definition on the host. Mean squares differ from row to row, and the
    /// small scale shrinks them to epsilon's order.
    async fn normalizes_each_head_of<Group: AxisName>() {
        let mut device = Device::new(heads.topology()).unwrap();
        let live = count(Sixteen::SIZE).to_hbm(&mut device.pdma).await.unwrap();
        for scale in [1.0f32, 1e-3] {
            let values: Vec<bf16> = random(
                Sixteen::SIZE * KvHeads::SIZE * Group::SIZE * Head::SIZE,
                0xaaaa_bbbb,
            )
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let row = index / Head::SIZE;
                bf16::from_f32(value.to_f32() * scale * (1 + row % 5) as f32)
            })
            .collect();
            let scales = scales(Head::SIZE);
            let expected = definition(&values, &scales);
            let input = HostTensor::<bf16, m![Sixteen, KvHeads, Group, Head]>::from_vec(values)
                .to_hbm::<Chip, m![Sixteen, KvHeads, Group, Head]>(&mut device.pdma)
                .await
                .unwrap();
            let norm = RmsNorm::new(Weight::new(
                HostTensor::<bf16, m![Head]>::from_vec(scales)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ));
            let actual = launch(heads, (&mut device, &norm, &live, &input))
                .await
                .unwrap()
                .to_host::<m![Sixteen, KvHeads, Group, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            assert_lands(&actual, &expected, Head::SIZE, scale);
        }
    }

    /// Query and key head rows each normalize by their own statistics.
    #[tokio::test]
    async fn normalizes_each_head() {
        normalizes_each_head_of::<QueryGroups<2>>().await;
        normalizes_each_head_of::<Single>().await;
    }
}

/// `input * weight / sqrt(mean(input^2) + epsilon)` over each of the `P` head rows of every
/// slice, in two passes: one sums the squares, the other scales, with the square root taken on
/// the way into the divisor register. `EPSILON` holds epsilon's FP32 bits.
impl<'d, 'x, T: Space, S: M, P: M, Head: AxisName, const EPSILON: usize>
    Module<(
        &'d mut Device,
        &'x Dm<T, DmTensor<bf16, m![1], m![1 # 2], S, m![{ P }, Head]>>,
    )> for RmsNorm<Head, EPSILON>
{
    type Output = Dm<T, DmTensor<bf16, m![1], m![1 # 2], S, m![{ P }, Head]>>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            &'x Dm<T, DmTensor<bf16, m![1], m![1 # 2], S, m![{ P }, Head]>>,
        ),
    ) -> Self::Output {
        Dm {
            value: kernel::heads::<S, P, Head, EPSILON>(device, &input.value, self.weight.tensor()),
            tiles: input.tiles,
        }
    }
}

/// Per-row mean/variance normalization followed by a learned scale and bias.
#[derive(DeviceSend)]
pub struct LayerNorm<Weight, Epsilon> {
    /// Learned scale per feature.
    pub weight: Weight,
    /// Learned offset per feature.
    pub bias: Weight,
    /// Scalar added to the variance before taking its square root.
    pub epsilon: Epsilon,
}

impl<Weight, Epsilon> LayerNorm<Weight, Epsilon> {
    pub fn new(weight: Weight, bias: Weight, epsilon: Epsilon) -> Self {
        Self {
            weight,
            bias,
            epsilon,
        }
    }
}

impl<D: Scalar, Width: AxisName, Rows: AxisName>
    Module<(&mut Device, &HbmTensor<D, m![1], m![Rows, Width]>)>
    for LayerNorm<HbmTensor<D, m![1], m![Width]>, HbmTensor<f32, m![1], m![1]>>
{
    type Output = HbmTensor<D, m![1], m![Rows, Width]>;

    fn forward(&self, _: (&mut Device, &HbmTensor<D, m![1], m![Rows, Width]>)) -> Self::Output {
        todo!("NPU layer normalization over contiguous rows")
    }
}

/// Typed RMS contract with runtime epsilon, FP32 statistics and source-dtype output.
#[derive(DeviceSend)]
pub struct Rms<Weight, Epsilon> {
    /// Per-feature gain: a learned scale, a [`Centered`] offset from one, or `()` for unit gain.
    pub weight: Weight,
    /// Device-resident scalar added to the mean square.
    pub epsilon: Epsilon,
}

impl<Weight, Epsilon> Rms<Weight, Epsilon> {
    pub fn new(weight: Weight, epsilon: Epsilon) -> Self {
        Self { weight, epsilon }
    }
}

impl<D: Scalar, Source: Scalar, Width: AxisName, Count: AxisName>
    Module<(&mut Device, &HbmTensor<Source, m![1], m![Count, Width]>)>
    for Rms<HbmTensor<D, m![1], m![Width]>, HbmTensor<f32, m![1], m![1]>>
{
    type Output = HbmTensor<Source, m![1], m![Count, Width]>;

    fn forward(
        &self,
        _: (&mut Device, &HbmTensor<Source, m![1], m![Count, Width]>),
    ) -> Self::Output {
        todo!("NPU RMS normalization with FP32 statistics and source-dtype output")
    }
}

/// Normalize each channel group over its channels and spatial extent with FP32 statistics.
#[derive(DeviceSend)]
pub struct GroupNorm<Channels: AxisName, Groups: AxisName> {
    /// Learned scale per channel.
    pub weight: HbmTensor<f32, m![1], m![Channels]>,
    /// Learned offset per channel.
    pub bias: HbmTensor<f32, m![1], m![Channels]>,
    /// Device-resident scalar added to each group's variance.
    pub epsilon: HbmTensor<f32, m![1], m![1]>,
    /// Compile-time channel group count; Channels must divide evenly into Groups.
    pub groups: std::marker::PhantomData<Groups>,
}

impl<C: AxisName, G: AxisName, H: AxisName, W: AxisName>
    Module<(&mut Device, &crate::ops::diffusion::Image<C, H, W>)> for GroupNorm<C, G>
{
    type Output = crate::ops::diffusion::Image<C, H, W>;

    fn forward(&self, _: (&mut Device, &crate::ops::diffusion::Image<C, H, W>)) -> Self::Output {
        const { assert!(G::SIZE > 0 && C::SIZE % G::SIZE == 0) };
        todo!(
            "NPU group normalization with FP32 statistics over each channel group and its spatial extent"
        )
    }
}
