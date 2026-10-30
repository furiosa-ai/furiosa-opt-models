//! `(x - mean(x)) / sqrt(var(x) + epsilon) * w + b` per row.

use super::{Lines, Tile, TilePart, TileParts, TileRows, Widen};
use crate::Op;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::tensor::scalar::F32;
use furiosa_opt_std::prelude::*;

/// FP32 bits of the epsilon a [`LayerNorm`] adds when its type names none, `torch.nn.LayerNorm`'s.
pub(crate) const EPSILON: usize = 1e-5_f32.to_bits() as usize;

/// Per-row mean/variance normalization, then the checkpoint's `weight` scale and `bias`; `EPSILON`
/// holds epsilon's FP32 bits, which a load places in the `Epsilon` leaf.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct LayerNorm<Weight, Epsilon, const EPSILON: usize = { self::EPSILON }> {
    /// Learned scale per feature.
    pub weight: Weight,
    /// Learned offset per feature.
    pub bias: Weight,
    /// Scalar added to the variance before taking its square root.
    pub epsilon: Epsilon,
}

impl<Weight, Epsilon, const EPSILON: usize> LayerNorm<Weight, Epsilon, EPSILON> {
    pub fn new(weight: Weight, bias: Weight, epsilon: Epsilon) -> Self {
        Self {
            weight,
            bias,
            epsilon,
        }
    }
}

/// The scale and offset under `weight` and `bias`, the epsilon placed from `EPSILON`. By hand, with
/// the update below: a derived `with` field would rebuild the epsilon in every update.
impl<'a, Weight, Epsilon, const EPSILON: usize> Load<&'a Checkpoint>
    for LayerNorm<Weight, Epsilon, EPSILON>
where
    Weight: Load<&'a Checkpoint>,
    // TODO: place the epsilon on several chips from one host copy.
    Epsilon: Load<HostTensor<f32, m![{ m![1] }, { m![1] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let weight = children.ask(cx.field("weight", "weight", |cx| Weight::load(cx, from)))?;
            let bias = children.ask(cx.field("bias", "bias", |cx| Weight::load(cx, from)))?;
            let (weight, bias) = children.node(weight.zip(bias))?;
            let epsilon = cx.field("epsilon", "epsilon", |cx| {
                Epsilon::load(cx, F32::<EPSILON>::host::<m![1], m![1]>())
            })?;
            Ok(Self::new(weight, bias, epsilon))
        })
    }
}

/// The scale and offset written in place, two rules of an update; the epsilon is no update
/// target.
impl<'t, 's, Weight, Epsilon, const EPSILON: usize>
    Load<(&'t mut LayerNorm<Weight, Epsilon, EPSILON>, &'s Checkpoint)>
    for &'t mut LayerNorm<Weight, Epsilon, EPSILON>
where
    for<'u> &'u mut Weight: Load<(&'u mut Weight, &'s Checkpoint)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut LayerNorm<Weight, Epsilon, EPSILON>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("weight", "weight", |cx| {
                <&mut Weight as Load<_>>::load(cx, (&mut into.weight, from))
            })?;
            cx.field("bias", "bias", |cx| {
                <&mut Weight as Load<_>>::load(cx, (&mut into.bias, from))
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

impl<Width: AxisName, Rows: AxisName, const EPSILON: usize>
    Op<&HbmTensor<f32, m![1], m![Rows, Width]>>
    for LayerNorm<HbmTensor<f32, m![1], m![Width]>, HbmTensor<f32, m![1], m![1]>, EPSILON>
{
    type Output = HbmTensor<f32, m![1], m![Rows, Width]>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<f32, m![1], m![Rows, Width]>) -> Self::Output {
        todo!("NPU layer normalization of FP32 rows with FP32 parameters")
    }
}

/// The FP32 vectors of the features the normalizing pass keeps in its registers: weight and bias.
const VECTORS: usize = 2;

/// The owned norm runs as its borrowed view.
impl<D, W: AxisName, R: AxisName, Chips: M, const EPSILON: usize> Op<&HbmTensor<D, Chips, m![R, W]>>
    for LayerNorm<HbmTensor<bf16, Chips, m![W]>, HbmTensor<f32, Chips, m![1]>, EPSILON>
where
    D: MaterializableScalar + FetchCast<f32>,
{
    type Output = HbmTensor<bf16, Chips, m![R, W]>;

    fn forward(&self, device: &mut Device, input: &HbmTensor<D, Chips, m![R, W]>) -> Self::Output {
        let view: LayerNorm<_, _, EPSILON> =
            LayerNorm::new(&self.weight, &self.bias, &self.epsilon);
        view.forward(device, input)
    }
}

/// Rows of `D` as `bf16((x - mean) / sqrt(var + eps) * w + b)`, the variance of the centered rows,
/// FP32 inside, tile `t` computing while `t + 1` arrives and `t - 1` is stored.
impl<'w, D, W: AxisName, R: AxisName, Chips: M, const EPSILON: usize>
    Op<&HbmTensor<D, Chips, m![R, W]>>
    for LayerNorm<&'w HbmTensor<bf16, Chips, m![W]>, &'w HbmTensor<f32, Chips, m![1]>, EPSILON>
where
    D: MaterializableScalar + FetchCast<f32>,
{
    type Output = HbmTensor<bf16, Chips, m![R, W]>;

    fn forward(&self, device: &mut Device, input: &HbmTensor<D, Chips, m![R, W]>) -> Self::Output {
        type T<W> = Symbol<TileRows<W, VECTORS>>;
        let tiles = const {
            assert!(
                R::SIZE.is_multiple_of(TileRows::<W, VECTORS>::SIZE),
                "rows come in whole tiles"
            );
            R::SIZE / TileRows::<W, VECTORS>::SIZE
        };
        let feature = |device: &mut Device, vector: &HbmTensor<bf16, Chips, m![W]>| {
            let in_dm: Tile<bf16, R, W, VECTORS, Chips> = vector.view().to_dm(&mut device.tdma);
            in_dm.widen::<W, 16>(&mut device.sub)
        };
        let (weight, bias) = (feature(device, self.weight), feature(device, self.bias));
        let epsilon: Tile<f32, R, W, VECTORS, Chips, m![1 # 8]> = self
            .epsilon
            .view()
            .pad::<m![1 # 8]>()
            .to_dm(&mut device.tdma);
        let epsilon = device
            .sub
            .begin(epsilon.view())
            .fetch::<m![1], m![1 # 8]>()
            .collect::<m![1], m![1 # 8]>()
            .to_vrf::<m![1 # 8]>();
        let to_dm = |device: &mut Device, at: usize, tile: &mut Tile<D, R, W, VECTORS, Chips>| {
            input
                .view()
                .tile::<m![R / ({ T<W> })], 1, m![R / ({ T<W> }) = 1 # (R / ({ T<W> })), R % ({ T<W> }), W]>(at)
                .to_dm_view(&mut device.tdma, tile.view_mut());
        };
        let mut output = HbmTensor::new();
        let mut store = |device: &mut Device,
                         tile: &Tile<bf16, R, W, VECTORS, Chips>,
                         at: usize| {
            tile.view().to_hbm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![R / ({ T<W> })], 1, m![R / ({ T<W> }) = 1 #{!} (R / ({ T<W> })), R % ({ T<W> }), W]>(at),
            );
        };
        type Statistic<R, W, Chips> = DmTensor<
            f32,
            Chips,
            m![R % ({ Symbol<TileRows<W, VECTORS>> }) / ({ Symbol<Lines<W, VECTORS>> })],
            m![R % ({ Symbol<Lines<W, VECTORS>> }), { Symbol<TileParts<W, VECTORS>> }],
            m![1 # 8],
        >;
        let standardize =
            |device: &mut Device,
             rows: &Tile<D, R, W, VECTORS, Chips>,
             normalized: &mut Tile<bf16, R, W, VECTORS, Chips>| {
                // SAFETY: every slice of a row holds the same statistic.
                let agree =
                    |sums: Statistic<R, W, Chips>| -> Tile<f32, R, W, VECTORS, Chips, m![1 # 8]> {
                        unsafe { sums.reshape() }
                    };
                let total: Statistic<R, W, Chips> = device
                .main
                .begin(rows.view())
                .fetch::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ TilePart<W, VECTORS> } / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_intra_slice_reduce::<W, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div(const { W::SIZE as f32 })
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<
                    m![R % ({ Symbol<Lines<W, VECTORS>> }), { Symbol<TileParts<W, VECTORS>> }],
                    m![1],
                >(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
                let total = agree(total);
                let mean = device
                    .sub
                    .begin(total.view())
                    .fetch::<m![1], m![1 # 8]>()
                    .collect::<m![1], m![1 # 8]>()
                    .to_vrf::<m![1 # 8]>();
                let spread: Statistic<R, W, Chips> = device
                .main
                .begin(rows.view())
                .fetch::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ TilePart<W, VECTORS> } / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_fp_binary(FpBinaryOp::SubF, &mean)
                .vector_stash()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                .vector_intra_slice_reduce::<W, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div(const { W::SIZE as f32 })
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<
                    m![R % ({ Symbol<Lines<W, VECTORS>> }), { Symbol<TileParts<W, VECTORS>> }],
                    m![1],
                >(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
                let spread = agree(spread);
                let inverse = device
                    .sub
                    .begin(spread.view())
                    .fetch::<m![1], m![1 # 8]>()
                    .collect::<m![1], m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_trim::<m![1 # 4]>()
                    .vector_fp_binary(FpBinaryOp::AddF, &epsilon)
                    .vector_fp_unary(FpUnaryOp::Sqrt)
                    .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .to_vrf::<m![1 # 8]>();
                device
                    .main
                    .begin(rows.view())
                    .fetch::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
                    .fetch_cast::<f32>()
                    .collect::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![{ TilePart<W, VECTORS> } / 8, W % 8 / 4 % 2], m![W % 4]>()
                    .vector_fp_binary(FpBinaryOp::SubF, &mean)
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inverse)
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight)
                    .vector_widen_concat::<m![{ TilePart<W, VECTORS> } / 8], m![W % 8]>()
                    .vector_clip(ClipBinaryOpF32::Add, &bias)
                    .vector_final()
                    .cast::<bf16, m![W % 8 # 16]>()
                    .commit_trim::<m![W % 8]>()
                    .commit_view(normalized.view_mut());
            };
        let (mut first, mut second): (
            Tile<D, R, W, VECTORS, Chips>,
            Tile<D, R, W, VECTORS, Chips>,
        ) = (DmTensor::new(), DmTensor::new());
        let (mut kept, mut held): (
            Tile<bf16, R, W, VECTORS, Chips>,
            Tile<bf16, R, W, VECTORS, Chips>,
        ) = (DmTensor::new(), DmTensor::new());
        to_dm(&mut *device, 0, &mut first);
        if const { R::SIZE / TileRows::<W, VECTORS>::SIZE > 1 } {
            to_dm(&mut *device, 1, &mut second);
        }
        for pair in 0..tiles / 2 {
            let at = 2 * pair;
            standardize(&mut *device, &first, &mut kept);
            to_dm(&mut *device, (at + 2).min(tiles - 1), &mut first);
            store(&mut *device, &kept, at);
            standardize(&mut *device, &second, &mut held);
            to_dm(&mut *device, (at + 3).min(tiles - 1), &mut second);
            store(&mut *device, &held, at + 1);
        }
        if const { R::SIZE / TileRows::<W, VECTORS>::SIZE % 2 == 1 } {
            standardize(&mut *device, &first, &mut kept);
            store(&mut *device, &kept, tiles - 1);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use super::LayerNorm;
    use crate::Op;
    use crate::ops::normalization::tests::{random, residual};
    use crate::tensor::parallel::Chip;
    use crate::tests::DEVICE;

    axes![Width = 2048, Wide = 6912, Count = 2048, Rows = 256];

    type Standard<W> = LayerNorm<HbmTensor<bf16, Chip, m![W]>, HbmTensor<f32, Chip, m![1]>>;

    #[device(chip = 1)]
    fn standardized<D: MaterializableScalar + FetchCast<f32>, R: AxisName, W: AxisName>(
        device: &mut Device,
        norm: &Standard<W>,
        input: &HbmTensor<D, Chip, m![R, W]>,
    ) -> HbmTensor<bf16, Chip, m![R, W]> {
        norm.forward(device, input)
    }

    /// Each row of `D`, its mean far from zero, is standardized, scaled and shifted, as the FP64
    /// definition rounded once to BF16.
    async fn standardizes<D: MaterializableScalar + FetchCast<f32>, R: AxisName, W: AxisName>(
        cast: fn(f32) -> D,
        round: fn(f32) -> f32,
    ) {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let values: Vec<f32> = residual(R::SIZE, W::SIZE)
            .iter()
            .enumerate()
            .map(|(index, v)| round(v + (index / W::SIZE % 13) as f32 - 6.0))
            .collect();
        let feature = |seed: u64, offset: f32| -> Vec<bf16> {
            random(W::SIZE, seed)
                .iter()
                .map(|v| bf16::from_f32(offset + v.to_f32() / 2.0))
                .collect()
        };
        let (weight, bias) = (
            feature(0x9e37_79b9_7f4a_7c15, 1.0),
            feature(0x1405_7b7e_f767_814f, 0.0),
        );
        let upload = async |device: &mut Device, values: &[bf16]| {
            HostTensor::<bf16, m![W]>::from_vec(values.to_vec())
                .to_hbm::<Chip, m![W]>(&mut device.pdma)
                .await
                .unwrap()
        };
        let norm = LayerNorm::new(
            upload(&mut device, &weight).await,
            upload(&mut device, &bias).await,
            HostTensor::<f32, m![1]>::from_vec(vec![1e-5])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        );
        let input = HostTensor::<D, m![R, W]>::from_vec(
            values.iter().map(|v| cast(*v)).collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![R, W]>(&mut device.pdma)
        .await
        .unwrap();
        let output = launch(standardized, (&mut device, &norm, &input))
            .await
            .unwrap()
            .to_host::<m![R, W]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (row, values) in values.chunks_exact(W::SIZE).enumerate() {
            let values: Vec<f64> = values.iter().map(|v| f64::from(*v)).collect();
            let mean = values.iter().sum::<f64>() / W::SIZE as f64;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / W::SIZE as f64;
            let inverse = 1.0 / (variance + 1e-5).sqrt();
            for (column, value) in values.iter().enumerate() {
                let expected = (value - mean) * inverse * f64::from(weight[column].to_f32())
                    + f64::from(bias[column].to_f32());
                let got = f64::from(output[row * W::SIZE + column].to_f32());
                assert!(
                    (got - expected).abs() <= expected.abs() / 256.0 + 1e-3,
                    "row {row} column {column}: {got} vs {expected}"
                );
            }
        }
    }

    /// FP32 rows, a quarter of a row a slice.
    #[tokio::test(flavor = "current_thread")]
    async fn standardizes_fp32_rows_into_bf16() {
        standardizes::<f32, Count, Width>(|v| v, |v| v).await;
    }

    /// BF16 rows too wide for a quarter of a row's weight and bias in the registers.
    #[tokio::test(flavor = "current_thread")]
    async fn standardizes_wide_bf16_rows() {
        standardizes::<bf16, Rows, Wide>(bf16::from_f32, |v| bf16::from_f32(v).to_f32()).await;
    }
}
