//! W4A16 tiled projections: packed weights, BF16 inputs and a late FP32 tensor scale.
//! The panel layouts and transfer overlap follow internal PR #457 (d2186073c).

use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::tensor::form::nvfp4::Scale;
use furiosa_opt_std::prelude::*;

pub(crate) mod block;
pub(crate) mod compact;
pub(crate) mod heads;
pub mod mlp;
pub mod output;
pub mod panels;
pub(crate) mod prefetch;

/// Only the weight tensor scale participates in W4A16 arithmetic. Input calibration is
/// validated when loading and discarded; activations and cached keys/values remain BF16.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Scales {
    pub output: Scale,
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

// Each projection reads canonical checkpoint rows and lays out its packed codes and block
// scales once, on host. The derive supplies tree borrowing; these implementations keep
// checkpoint validation, in-place updates and stacked layer backing in the native loader.
macro_rules! loading {
    ($projection:ident) => {
        fn codes<I: AxisName, O: AxisName>(
            from: &crate::load::Checkpoint,
            cx: &mut crate::load::Context<'_>,
        ) -> Result<HostTensor<f4e2m1, Stored<I, O>>, crate::load::LoadError> {
            super_load::<I, O, f4e2m1, Stored<I, O>, 2>(from, cx, safetensors::Dtype::U8)
        }
        fn scales<I: AxisName, O: AxisName>(
            from: &crate::load::Checkpoint,
            cx: &mut crate::load::Context<'_>,
        ) -> Result<HostTensor<f8e4m3, StoredScales<I, O>>, crate::load::LoadError> {
            super_load::<I, O, f8e4m3, StoredScales<I, O>, 16>(
                from,
                cx,
                safetensors::Dtype::F8_E4M3,
            )
        }
        fn super_load<
            I: AxisName,
            O: AxisName,
            D: MaterializableScalar,
            E: M,
            const PACK: usize,
        >(
            from: &crate::load::Checkpoint,
            cx: &mut crate::load::Context<'_>,
            dtype: safetensors::Dtype,
        ) -> Result<HostTensor<D, E>, crate::load::LoadError> {
            let (name, tensor) = cx.tensor(from)?;
            let invalid = |reason| crate::load::LoadError::InvalidData {
                name: name.clone(),
                reason,
            };
            if tensor.dtype() != dtype || tensor.shape() != [O::SIZE, I::SIZE / PACK] {
                return Err(invalid(format!(
                    "expected {dtype:?} [{}, {}]",
                    O::SIZE,
                    I::SIZE / PACK
                )));
            }
            if PACK == 16 && tensor.data().iter().any(|&v| v & 0x7f == 0x7f || v > 0x80) {
                return Err(invalid(
                    "block scales must be finite and nonnegative".into(),
                ));
            }
            let bytes = $projection::<I, O>::reorder::<PACK, _>(tensor.data());
            HostTensor::try_from(furiosa_opt_std::runtime::HostBuffer::from_vec(bytes))
                .map_err(|error| invalid(error.to_string()))
        }
        impl<'a, I: AxisName, O: AxisName> crate::load::Load<&'a crate::load::Checkpoint>
            for $projection<I, O>
        {
            fn load(
                cx: &mut crate::load::Context<'_>,
                from: &'a crate::load::Checkpoint,
            ) -> Result<Self, crate::load::LoadError> {
                use crate::load::{Children, Load};
                cx.checkpoint(from, |cx| {
                    let mut children = Children::default();
                    let weight = children.ask(cx.field("weight", "weight", |cx| {
                        let host = codes::<I, O>(from, cx)?;
                        Load::load(cx, host)
                    }))?;
                    let weight_scale =
                        children.ask(cx.field("weight_scale", "weight_scale", |cx| {
                            let host = scales::<I, O>(from, cx)?;
                            Load::load(cx, host)
                        }))?;
                    let factors =
                        children.ask(cx.field("scales", "", |cx| Load::load(cx, from)))?;
                    children.node(weight.zip(weight_scale).zip(factors)).map(
                        |((weight, weight_scale), scales)| Self {
                            weight,
                            weight_scale,
                            scales,
                        },
                    )
                })
            }
        }
        impl<'t, 's, I: AxisName, O: AxisName>
            crate::load::Load<(&'t mut $projection<I, O>, &'s crate::load::Checkpoint)>
            for &'t mut $projection<I, O>
        {
            fn load(
                cx: &mut crate::load::Context<'_>,
                (into, from): (&'t mut $projection<I, O>, &'s crate::load::Checkpoint),
            ) -> Result<Self, crate::load::LoadError> {
                use crate::load::Load;
                cx.checkpoint(from, |cx| {
                    cx.field("weight", "weight", |cx| {
                        cx.rule(
                            |cx| codes::<I, O>(from, cx),
                            |cx, host| {
                                <&mut HbmTensor<f4e2m1, Chip, Stored<I, O>> as Load<_>>::load(
                                    cx,
                                    (&mut into.weight, host),
                                )
                                .map(drop)
                            },
                        )
                    })?;
                    cx.field("weight_scale", "weight_scale", |cx| {
                        cx.rule(
                            |cx| scales::<I, O>(from, cx),
                            |cx, host| {
                                <&mut HbmTensor<f8e4m3, Chip, StoredScales<I, O>> as Load<_>>::load(
                                    cx,
                                    (&mut into.weight_scale, host),
                                )
                                .map(drop)
                            },
                        )
                    })?;
                    cx.field("scales", "", |cx| {
                        <&mut crate::ops::linear::nvfp4::tiled::Scales as Load<_>>::load(
                            cx,
                            (&mut into.scales, from),
                        )
                    })
                    .map(drop)
                })?;
                Ok(into)
            }
        }
        impl<I: AxisName, O: AxisName, const N: usize> crate::modules::generic::collection::View<N>
            for $projection<I, O>
        {
            type Storage = (
                crate::modules::generic::collection::Tensor<f4e2m1, Chip, Stored<I, O>, N>,
                crate::modules::generic::collection::Tensor<f8e4m3, Chip, StoredScales<I, O>, N>,
                crate::modules::generic::collection::Tensor<f32, Chip, m![1 # 8], N>,
            );
            type Layer = Self;
            fn view(storage: &Self::Storage, layer: usize) -> Self {
                use crate::modules::generic::collection::View;
                Self {
                    weight: <HbmTensor<f4e2m1, Chip, Stored<I, O>> as View<N>>::view(
                        &storage.0, layer,
                    ),
                    weight_scale: <HbmTensor<f8e4m3, Chip, StoredScales<I, O>> as View<N>>::view(
                        &storage.1, layer,
                    ),
                    scales: crate::ops::linear::nvfp4::tiled::Scales {
                        output: <crate::tensor::form::nvfp4::Scale as View<N>>::view(
                            &storage.2, layer,
                        ),
                    },
                }
            }
            fn stack(layers: [Self; N], cx: &mut crate::load::Context<'_>) -> Self::Storage {
                use crate::modules::generic::collection::View;
                let mut layers = layers.map(Some);
                let mut weights = Vec::new();
                let mut scales = Vec::new();
                let mut factors = Vec::new();
                for layer in &mut layers {
                    let layer = layer.take().unwrap();
                    weights.push(layer.weight);
                    scales.push(layer.weight_scale);
                    factors.push(layer.scales.output);
                }
                (
                    <HbmTensor<f4e2m1, Chip, Stored<I, O>> as View<N>>::stack(
                        weights.try_into().ok().unwrap(),
                        cx,
                    ),
                    <HbmTensor<f8e4m3, Chip, StoredScales<I, O>> as View<N>>::stack(
                        scales.try_into().ok().unwrap(),
                        cx,
                    ),
                    <crate::tensor::form::nvfp4::Scale as View<N>>::stack(
                        factors.try_into().ok().unwrap(),
                        cx,
                    ),
                )
            }
        }
    };
}
pub(crate) use loading;
