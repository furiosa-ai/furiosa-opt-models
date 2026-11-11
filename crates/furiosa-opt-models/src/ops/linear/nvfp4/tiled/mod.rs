//! W4A16 tiled projections: packed weights, BF16 inputs and a late FP32 tensor scale.
//! The panel layouts and transfer overlap follow internal PR #457 (d2186073c).

use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::modules::generic::collection::View;
use crate::tensor::{
    form::{
        BlockScaled,
        nvfp4::{self, Scale},
    },
    parallel::Chip,
};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

pub(crate) mod block;
pub(crate) mod compact;
pub(crate) mod heads;
pub mod mlp;
pub mod output;
pub mod panels;
pub(crate) mod prefetch;
pub mod qkv;

/// The existing block-scaled form with each kernel's packed storage mapping.
/// Canonical checkpoint tensors are transposed by the host upload, never at runtime.
pub type Weights<I, O, W, B> = BlockScaled<
    I,
    O,
    f4e2m1,
    f8e4m3,
    nvfp4::Scales<I, O>,
    Scales,
    Chip,
    HbmTensor<f4e2m1, Chip, W>,
    Blocks<I, O, B>,
>;

/// A scale per 16 canonical input features, uploaded into a kernel's storage mapping.
/// The explicit source shape avoids inferring checkpoint axes from a partitioned mapping.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Blocks<I: AxisName, O: AxisName, B: M> {
    pub tensor: HbmTensor<f8e4m3, Chip, B>,
    shape: PhantomData<fn() -> (I, O)>,
}

impl<I: AxisName, O: AxisName, B: M> Blocks<I, O, B> {
    pub(crate) fn new(tensor: HbmTensor<f8e4m3, Chip, B>) -> Self {
        Self {
            tensor,
            shape: PhantomData,
        }
    }
}

impl<'a, I: AxisName, O: AxisName, W: M + 'static, B: M + 'static> Load<&'a Checkpoint>
    for Weights<I, O, W, B>
{
    fn load(cx: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        cx.checkpoint(from, |cx| Self::read(from, cx, "weight"))
    }
}

impl<'t, 's, I: AxisName, O: AxisName, W: M + 'static, B: M + 'static>
    Load<(&'t mut Weights<I, O, W, B>, &'s Checkpoint)> for &'t mut Weights<I, O, W, B>
{
    fn load(
        cx: &mut Context<'_>,
        (into, from): (&'t mut Weights<I, O, W, B>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        cx.checkpoint(from, |cx| into.update(from, cx, "weight"))?;
        Ok(into)
    }
}

impl<'a, I: AxisName, O: AxisName, B: M + 'static> Load<&'a Checkpoint> for Blocks<I, O, B> {
    fn load(cx: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let host = block_scales::<I, O>(from, cx)?;
        Load::load(cx, host).map(Self::new)
    }
}

impl<'t, 's, I: AxisName, O: AxisName, B: M + 'static>
    Load<(&'t mut Blocks<I, O, B>, &'s Checkpoint)> for &'t mut Blocks<I, O, B>
{
    fn load(
        cx: &mut Context<'_>,
        (into, from): (&'t mut Blocks<I, O, B>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        cx.rule(
            |cx| block_scales::<I, O>(from, cx),
            |cx, host| {
                <&mut HbmTensor<f8e4m3, Chip, B> as Load<_>>::load(cx, (&mut into.tensor, host))
                    .map(drop)
            },
        )?;
        Ok(into)
    }
}

impl<I: AxisName, O: AxisName, B: M, const N: usize> View<N> for Blocks<I, O, B> {
    type Storage = <HbmTensor<f8e4m3, Chip, B> as View<N>>::Storage;
    type Layer = Self;
    fn view(storage: &Self::Storage, layer: usize) -> Self {
        Self::new(<HbmTensor<f8e4m3, Chip, B> as View<N>>::view(
            storage, layer,
        ))
    }
    fn stack(layers: [Self; N], cx: &mut Context<'_>) -> Self::Storage {
        <HbmTensor<f8e4m3, Chip, B> as View<N>>::stack(layers.map(|layer| layer.tensor), cx)
    }
}

/// Only the weight tensor scale participates in W4A16 arithmetic. Input calibration is
/// validated when loading and discarded; activations and cached keys/values remain BF16.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Scales {
    pub output: Scale,
}

impl<'a> Load<&'a Checkpoint> for Scales {
    fn load(cx: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        cx.checkpoint(from, |cx| {
            let host = tensor_scale(from, cx)?;
            Ok(Self {
                output: cx.field("output", "", |cx| Load::load(cx, host))?,
            })
        })
    }
}
impl<'t, 's> Load<(&'t mut Scales, &'s Checkpoint)> for &'t mut Scales {
    fn load(
        cx: &mut Context<'_>,
        (into, from): (&'t mut Scales, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        cx.checkpoint(from, |cx| {
            cx.rule(
                |cx| tensor_scale(from, cx),
                |cx, host| {
                    cx.field("output", "", |cx| {
                        <&mut Scale as Load<_>>::load(cx, (&mut into.output, host)).map(drop)
                    })
                },
            )
        })?;
        Ok(into)
    }
}

impl<const N: usize> View<N> for Scales {
    type Storage = <Scale as View<N>>::Storage;
    type Layer = Self;
    fn view(storage: &Self::Storage, layer: usize) -> Self {
        Self {
            output: <Scale as View<N>>::view(storage, layer),
        }
    }
    fn stack(layers: [Self; N], cx: &mut Context<'_>) -> Self::Storage {
        <Scale as View<N>>::stack(layers.map(|layer| layer.output), cx)
    }
}

fn block_scales<I: AxisName, O: AxisName>(
    from: &Checkpoint,
    cx: &mut Context<'_>,
) -> Result<HostTensor<f8e4m3, m![O, I / 16]>, LoadError> {
    let (name, tensor) = cx.tensor(from)?;
    let invalid = |reason| LoadError::InvalidData {
        name: name.clone(),
        reason,
    };
    if tensor.dtype() != safetensors::Dtype::F8_E4M3 || tensor.shape() != [O::SIZE, I::SIZE / 16] {
        return Err(invalid(format!(
            "expected F8_E4M3 [{}, {}]",
            O::SIZE,
            I::SIZE / 16
        )));
    }
    if tensor.data().iter().any(|&v| v & 0x7f == 0x7f || v > 0x80) {
        return Err(invalid(
            "block scales must be finite and nonnegative".into(),
        ));
    }
    HostTensor::try_from(furiosa_opt_std::runtime::HostBuffer::from_vec(
        tensor.data().to_vec(),
    ))
    .map_err(|error| invalid(error.to_string()))
}

fn tensor_scale(
    from: &Checkpoint,
    cx: &mut Context<'_>,
) -> Result<HostTensor<f32, m![1 # 8]>, LoadError> {
    let [_, output] =
        crate::tensor::form::nvfp4::scales(from, cx, ["input_scale", "weight_scale_2"])?;
    Ok(HostTensor::from_vec([
        output, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]))
}

#[cfg(test)]
mod tests {
    // Scalar oracle and transfers shared by the child projection tests.
    use crate::{
        ops::linear::nvfp4::tiled::Scales,
        tensor::{form::nvfp4::Scale, scalar::Value as Count},
    };
    use furiosa_opt_std::prelude::*;

    axes![
        Hidden = 4096,
        OutputWidth = 4096,
        Intermediate = 14336,
        Rows = 512
    ];

    /// Checkpoint scales retain the canonical matrix shape and finite, nonnegative E4M3 values.
    #[test]
    fn reads_canonical_block_scales() {
        use crate::load::{Checkpoint, Context, LoadError};
        use safetensors::Dtype;
        axes![I = 32, O = 2];
        let read = |dtype, shape: &[usize], bytes: Vec<u8>| {
            let mut source = Checkpoint::default();
            source.insert("weight_scale", dtype, shape, bytes).unwrap();
            Context::new(None).field("weight_scale", "weight_scale", |cx| {
                super::block_scales::<I, O>(&source, cx)
            })
        };
        let bytes = vec![0, 0x80, 0x38, 0x7e];
        let loaded = read(Dtype::F8_E4M3, &[2, 2], bytes.clone()).unwrap();
        assert_eq!(Vec::<u8>::from(loaded.into_buffer()), bytes);
        for (dtype, shape, bytes, reason) in [
            (
                Dtype::U8,
                &[2, 2][..],
                vec![0; 4],
                "expected F8_E4M3 [2, 2]",
            ),
            (
                Dtype::F8_E4M3,
                &[1, 4][..],
                vec![0; 4],
                "expected F8_E4M3 [2, 2]",
            ),
            (
                Dtype::F8_E4M3,
                &[2, 2][..],
                vec![0x81; 4],
                "block scales must be finite and nonnegative",
            ),
            (
                Dtype::F8_E4M3,
                &[2, 2][..],
                vec![0x7f; 4],
                "block scales must be finite and nonnegative",
            ),
        ] {
            match read(dtype, shape, bytes).err() {
                Some(LoadError::InvalidData {
                    name,
                    reason: found,
                }) => {
                    assert_eq!((name.as_str(), found.as_str()), ("weight_scale", reason));
                }
                other => panic!("{other:?}"),
            }
        }
    }

    /// Keep test setup explicit while sharing the repeated transfer/error handling.
    pub(super) async fn upload<D: MaterializableScalar, Shape: M>(
        pdma: &mut DmaContext<{ Dma::Pcie }>,
        values: impl IntoIterator<Item = D>,
    ) -> HbmTensor<D, m![1], Shape> {
        HostTensor::<D, Shape>::from_vec(values)
            .to_hbm(pdma)
            .await
            .unwrap()
    }

    pub(super) async fn read<D: MaterializableScalar, Shape: M>(
        pdma: &mut DmaContext<{ Dma::Pcie }>,
        tensor: &HbmTensor<D, m![1], Shape>,
    ) -> Vec<D> {
        tensor.to_host::<Shape>(pdma).await.unwrap().into_vec()
    }

    async fn scalar(device: &mut Device, value: f32) -> Scale {
        upload::<f32, m![1 # 8]>(&mut device.pdma, vec![value; 8]).await
    }

    fn feature<I: AxisName>(output: usize, term: usize) -> usize {
        if I::SIZE == 14336 {
            (output % 16) * 896 + term * 128 + output % 128
        } else {
            output * 97 % I::SIZE
        }
    }

    fn terms<I: AxisName>() -> usize {
        if I::SIZE == 14336 { 7 } else { 1 }
    }
    fn block_scale(output: usize, input: usize) -> f32 {
        [0.5, 1.0, 1.5, 2.0][(output + input / 16) % 4]
    }
    fn weight_value(output: usize) -> f32 {
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ][output % 16]
    }
    fn base_value(row: usize, feature: usize) -> f32 {
        [
            6.0, 0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5,
            -3.5, -5.0,
        ][(row + feature) % 16]
    }
    fn amplitude(row: usize, feature: usize) -> f32 {
        [0.0, 0.25, 0.5, 1.0, 2.0][(row + feature / 16) % 5]
    }
    fn value(row: usize, feature: usize) -> f32 {
        base_value(row, feature) * amplitude(row, feature)
    }
    pub(super) async fn weights<I: AxisName, O: AxisName, W: M, B: M>(
        device: &mut Device,
        pack_weight: impl FnOnce(&[f4e2m1]) -> Vec<f4e2m1>,
        pack_scale: impl FnOnce(&[f8e4m3]) -> Vec<f8e4m3>,
    ) -> (
        HbmTensor<f4e2m1, m![1], W>,
        HbmTensor<f8e4m3, m![1], B>,
        Scales,
    ) {
        weights_with::<I, O, W, B>(device, pack_weight, pack_scale, feature::<I>).await
    }

    pub(super) async fn weights_with<I: AxisName, O: AxisName, W: M, B: M>(
        device: &mut Device,
        pack_weight: impl FnOnce(&[f4e2m1]) -> Vec<f4e2m1>,
        pack_scale: impl FnOnce(&[f8e4m3]) -> Vec<f8e4m3>,
        feature: impl Fn(usize, usize) -> usize,
    ) -> (
        HbmTensor<f4e2m1, m![1], W>,
        HbmTensor<f8e4m3, m![1], B>,
        Scales,
    ) {
        let mut codes = vec![f4e2m1::from_bits(0); O::SIZE * I::SIZE];
        for output in 0..O::SIZE {
            for term in 0..terms::<I>() {
                codes[output * I::SIZE + feature(output, term)] =
                    f4e2m1::from_bits((output % 16) as u8);
            }
        }
        (
            upload::<f4e2m1, W>(&mut device.pdma, pack_weight(&codes)).await,
            upload::<f8e4m3, B>(
                &mut device.pdma,
                pack_scale(
                    &(0..O::SIZE * I::SIZE / 16)
                        .map(|at| {
                            f8e4m3::from_f32(block_scale(
                                at / (I::SIZE / 16),
                                at % (I::SIZE / 16) * 16,
                            ))
                        })
                        .collect::<Vec<_>>(),
                ),
            )
            .await,
            // Only weight_scale_2=.5 applies; fractional inputs detect unwanted A4 quantization.
            Scales {
                output: scalar(device, 0.5).await,
            },
        )
    }

    pub(super) async fn input<I: AxisName>(
        device: &mut Device,
    ) -> HbmTensor<bf16, m![1], m![Rows, I]> {
        upload::<bf16, m![Rows, I]>(
            &mut device.pdma,
            (0..Rows::SIZE * I::SIZE).map(|at| bf16::from_f32(value(at / I::SIZE, at % I::SIZE))),
        )
        .await
    }

    pub(super) async fn count(device: &mut Device, rows: usize) -> Count {
        upload::<i32, m![1]>(&mut device.pdma, vec![rows as i32]).await
    }

    pub(super) async fn check<I: AxisName, O: AxisName>(
        device: &mut Device,
        output: HbmTensor<bf16, m![1], m![Rows, O]>,
        rows: usize,
    ) {
        let values = read::<_, m![Rows, O]>(&mut device.pdma, &output).await;
        for (index, found) in values.into_iter().take(rows * O::SIZE).enumerate() {
            let (row, output) = (index / O::SIZE, index % O::SIZE);
            let expected = bf16::from_f32(
                (0..terms::<I>())
                    .map(|term| {
                        let feature = feature::<I>(output, term);
                        base_value(row, feature)
                            * amplitude(row, feature)
                            * weight_value(output)
                            * block_scale(output, feature)
                            * 0.5
                    })
                    .sum::<f32>(),
            )
            .to_f32();
            assert_eq!(
                found.to_f32(),
                expected,
                "live {rows} row {row} output {output}"
            );
        }
    }
}
