//! The NVFP4 contracts of a [`BlockScaled`](super::BlockScaled) weight: E2M1 codes under one E4M3
//! scale per 16 inputs and an FP32 tensor scale, restored before the contraction.
//!
//! - Code times block scale is exact in BF16. The tensor scale divides each restored coefficient
//!   before its one BF16 rounding ([`Early`]), or multiplies the reduced FP32 sum once ([`Late`]),
//!   as a block-scaled FP4 product does.
//! - [`W4A4`] inputs quantize as the weights do: a 16-input block's E4M3 scale comes from its
//!   largest magnitude under the input's FP32 tensor scale, and its codes round to nearest even,
//!   bit for bit as the reference quantizer `ref_nvfp4_quant` does. [`W4A16`] inputs stay BF16.
//!
//! Source: <https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py>.

use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::HostBuffer;
use safetensors::Dtype;

use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::tensor::memory::{Hbm, Memory};
use crate::tensor::parallel::Chip;

/// FP32 scalar in the first lane of an eight-element padded vector.
pub type Scale = HbmTensor<f32, Chip, m![1 # 8]>;

/// A [`Scale`] held in memory `X`.
pub type Held<X> = <X as Memory>::Tensor<f32, Chip, m![1 # 8]>;

/// One E4M3 scale for each 16 consecutive inputs of an output row.
pub type Scales<In, Out> = m![Out, In / 16];

/// Each restored coefficient divided by the tensor scale before its one BF16 rounding, inputs in
/// form `A`: a compressed-tensors checkpoint's `weight_global_scale`.
#[derive(DeviceSend, Load)]
pub struct Early<A, X: Memory = Hbm> {
    /// The form inputs enter the contraction in.
    #[load(rename = "")]
    pub input: A,
    /// The coefficients' tensor scale: a coefficient restores to `code * block scale / global`.
    #[load(rename = "weight_global_scale")]
    pub global: Held<X>,
}

/// The reduced FP32 sum times the tensor scales once, inputs in form `A`.
#[derive(DeviceSend)]
pub struct Late<A, X: Memory = Hbm> {
    /// The form inputs enter the contraction in.
    pub input: A,
    /// Returns the contracted sum to output units: `input_scale * weight_scale_2` for [`W4A4`]
    /// inputs, `weight_scale_2` for [`W4A16`] ones.
    pub output: Held<X>,
}

/// BF16 inputs, which enter the contraction as they are; a checkpoint's input scales go unused.
#[derive(Clone, Copy, Default, DeviceSend, Load)]
#[load(ignore = "input_scale", ignore = "input_global_scale")]
pub struct W4A16;

/// Inputs quantized at run time as the weights are, under FP32 scalars derived from a
/// checkpoint's input scale: its calibrated input maximum over 6 x 448.
#[derive(DeviceSend)]
pub struct W4A4<X: Memory = Hbm> {
    /// `1 / input_scale`: block scales are E4M3 in these units.
    pub global: Held<X>,
    /// `1 / global`, which returns a block scale to input units.
    pub restore: Held<X>,
}

/// A ModelOpt checkpoint's weight tensor scale `weight_scale_2` beside the projection's leaves.
impl<'a, X: Memory> Load<&'a Checkpoint> for Late<W4A16, X>
where
    Held<X>: Load<&'a Checkpoint>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let input = cx.field("input", "", |cx| W4A16::load(cx, from))?;
            let output = cx.field("output", "weight_scale_2", |cx| Held::<X>::load(cx, from))?;
            Ok(Self { input, output })
        })
    }
}

impl<'t, 's, X: Memory> Load<(&'t mut Late<W4A16, X>, &'s Checkpoint)> for &'t mut Late<W4A16, X>
where
    for<'u> &'u mut Held<X>: Load<(&'u mut Held<X>, &'s Checkpoint)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Late<W4A16, X>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("input", "", |cx| {
                <&mut W4A16 as Load<_>>::load(cx, (&mut into.input, from))
            })?;
            cx.field("output", "weight_scale_2", |cx| {
                <&mut Held<X> as Load<_>>::load(cx, (&mut into.output, from))
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

/// A ModelOpt checkpoint's input scale and weight tensor scale: the sum's factor is
/// `input_scale * weight_scale_2`, the inputs quantize under `input_scale`.
impl<'a, X: Memory> Load<&'a Checkpoint> for Late<W4A4<X>, X>
where
    Held<X>: Load<HostTensor<f32, m![{ Chip }, { m![1 # 8] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let [input, weight] = scales(from, cx, ["input_scale", "weight_scale_2"])?;
            let output = cx.field("output", "output", |cx| {
                Held::<X>::load(cx, factor(input * weight))
            })?;
            let input = cx.field("input", "", |cx| W4A4::placed(input, cx))?;
            Ok(Self { input, output })
        })
    }
}

/// The factors written in place, one rule of an update once both scales arrived.
impl<'t, 's, X: Memory> Load<(&'t mut Late<W4A4<X>, X>, &'s Checkpoint)>
    for &'t mut Late<W4A4<X>, X>
where
    for<'u> &'u mut Held<X>: Load<(
        &'u mut Held<X>,
        HostTensor<f32, m![{ Chip }, { m![1 # 8] }]>,
    )>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Late<W4A4<X>, X>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.rule(
                |cx| scales(from, cx, ["input_scale", "weight_scale_2"]),
                |cx, [input, weight]| {
                    let factors = [
                        (&mut into.output, input * weight),
                        (&mut into.input.global, 1.0 / input),
                        (&mut into.input.restore, 1.0 / (1.0 / input)),
                    ];
                    factors.into_iter().try_for_each(|(leaf, value)| {
                        <&mut Held<X> as Load<_>>::load(cx, (leaf, factor(value))).map(drop)
                    })
                },
            )
        })?;
        Ok(into)
    }
}

impl<X: Memory> W4A4<X> {
    /// The input quantizer of `input_scale`: `global` its inverse, `restore` the inverse of that,
    /// as the reference rounds it.
    fn placed(input: f32, context: &mut Context<'_>) -> Result<Self, LoadError>
    where
        Held<X>: Load<HostTensor<f32, m![{ Chip }, { m![1 # 8] }]>>,
    {
        let global = context.field("global", "global", |cx| {
            Held::<X>::load(cx, factor(1.0 / input))
        })?;
        let restore = context.field("restore", "restore", |cx| {
            Held::<X>::load(cx, factor(1.0 / (1.0 / input)))
        })?;
        Ok(Self { global, restore })
    }
}

impl<'a, A: Load<&'a B>, B, X: Memory, Y: Memory> Load<&'a Late<B, Y>> for Late<A, X>
where
    Held<X>: Load<&'a Held<Y>>,
{
    fn load(context: &mut Context<'_>, from: &'a Late<B, Y>) -> Result<Self, LoadError> {
        Ok(Self {
            input: context.field("input", "", |cx| A::load(cx, &from.input))?,
            output: context.field("output", "output", |cx| Held::<X>::load(cx, &from.output))?,
        })
    }
}

impl<'t, A, X: Memory> Load<(&'t mut Late<A, X>, Late<A, X>)> for &'t mut Late<A, X>
where
    for<'u> &'u mut A: Load<(&'u mut A, A)>,
    for<'u> &'u mut Held<X>: Load<(&'u mut Held<X>, Held<X>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Late<A, X>, Late<A, X>),
    ) -> Result<Self, LoadError> {
        context.field("input", "", |cx| {
            <&mut A as Load<_>>::load(cx, (&mut into.input, from.input))
        })?;
        context.field("output", "output", |cx| {
            <&mut Held<X> as Load<_>>::load(cx, (&mut into.output, from.output))
        })?;
        Ok(into)
    }
}

impl<'a, X: Memory, Y: Memory> Load<&'a W4A4<Y>> for W4A4<X>
where
    Held<X>: Load<&'a Held<Y>>,
{
    fn load(context: &mut Context<'_>, from: &'a W4A4<Y>) -> Result<Self, LoadError> {
        Ok(Self {
            global: context.field("global", "global", |cx| Held::<X>::load(cx, &from.global))?,
            restore: context.field("restore", "restore", |cx| {
                Held::<X>::load(cx, &from.restore)
            })?,
        })
    }
}

impl<'t, X: Memory> Load<(&'t mut W4A4<X>, W4A4<X>)> for &'t mut W4A4<X>
where
    for<'u> &'u mut Held<X>: Load<(&'u mut Held<X>, Held<X>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut W4A4<X>, W4A4<X>),
    ) -> Result<Self, LoadError> {
        context.field("global", "global", |cx| {
            <&mut Held<X> as Load<_>>::load(cx, (&mut into.global, from.global))
        })?;
        context.field("restore", "restore", |cx| {
            <&mut Held<X> as Load<_>>::load(cx, (&mut into.restore, from.restore))
        })?;
        Ok(into)
    }
}

/// The packed E2M1 codes at the context's name, two a byte as stored (`[out, in / 2]` U8), on
/// every chip of `C`.
pub(crate) fn codes<C: M + 'static, S: M + 'static>(
    from: &Checkpoint,
    context: &mut Context<'_>,
) -> Result<HostTensor<f4e2m1, m![{ C }, { S }]>, LoadError> {
    let (name, view) = context.tensor(from)?;
    let elements = <S as M>::SIZE;
    let rows = view.shape().first().copied().unwrap_or(0);
    if view.dtype() != Dtype::U8 || view.shape().len() != 2 || 2 * view.data().len() != elements {
        return Err(LoadError::InvalidData {
            name,
            reason: format!(
                "expected {elements} packed E2M1 codes in {rows} U8 rows, two codes a byte"
            ),
        });
    }
    HostTensor::try_from(HostBuffer::from_vec(view.data().repeat(C::SIZE))).map_err(|error| {
        LoadError::InvalidData {
            name,
            reason: error.to_string(),
        }
    })
}

/// The tensor scales `leaves` beside the context's name, absent ones reported together.
fn scales<const N: usize>(
    from: &Checkpoint,
    context: &mut Context<'_>,
    leaves: [&str; N],
) -> Result<[f32; N], LoadError> {
    // Each is one finite, positive FP32 value.
    let mut children = Children::default();
    let read = leaves.into_iter().map(|leaf| {
        let value = context.field(leaf, leaf, |cx| {
            let (name, view) = cx.tensor(from)?;
            let value = match (view.dtype(), view.shape(), view.data()) {
                (Dtype::F32, [] | [1], &[a, b, c, d]) => f32::from_le_bytes([a, b, c, d]),
                _ => {
                    return Err(LoadError::InvalidData {
                        name,
                        reason: "expected one FP32 scale".into(),
                    });
                }
            };
            match value.is_finite() && value > 0.0 {
                true => Ok(value),
                false => Err(LoadError::InvalidData {
                    name,
                    reason: "the scale must be finite and positive".into(),
                }),
            }
        });
        children.ask(value)
    });
    let read = read.collect::<Result<Vec<_>, _>>()?;
    let values = children.node(read.into_iter().collect::<Option<Vec<f32>>>())?;
    Ok(values.try_into().expect("one value a leaf"))
}

/// `value` as a [`Scale`]'s host value: the first lane, then zero padding.
fn factor(value: f32) -> HostTensor<f32, m![{ Chip }, { m![1 # 8] }]> {
    HostTensor::from_vec([value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![Outputs = 2, Inputs = 32];

    /// The checkpoint at `name` of `tensors`, read by `rule` without a device.
    fn read<T>(
        tensors: &[(&str, Dtype, &[usize], Vec<u8>)],
        name: &str,
        rule: impl FnOnce(&Checkpoint, &mut Context<'_>) -> Result<T, LoadError>,
    ) -> Result<T, LoadError> {
        let mut source = Checkpoint::default();
        for (name, dtype, shape, bytes) in tensors {
            source.insert(name, *dtype, shape, bytes.clone()).unwrap();
        }
        Context::new(None).field(name, name, |cx| rule(&source, cx))
    }

    /// `(name, reason)` of an `InvalidData` refusal.
    fn invalid<T>(read: Result<T, LoadError>) -> (String, String) {
        match read.err() {
            Some(LoadError::InvalidData { name, reason }) => (name, reason),
            other => panic!("{other:?}"),
        }
    }

    /// The names a `Missing` refusal lists.
    fn missing<T>(read: Result<T, LoadError>) -> Vec<String> {
        match read.err() {
            Some(LoadError::Missing(names)) => names,
            other => panic!("{other:?}"),
        }
    }

    /// The codes are a U8 matrix holding two codes a byte, kept as stored; another dtype, rank or
    /// count is refused, and absent codes are missing.
    #[test]
    fn reads_codes_two_a_byte() {
        let stored_as = |dtype, shape: &[usize], bytes: usize| {
            let bytes = (0..bytes).map(|byte| byte as u8).collect();
            let tensors = [("weight", dtype, shape, bytes)];
            read(&tensors, "weight", codes::<m![1], m![Outputs, Inputs]>)
        };
        let stored = stored_as(Dtype::U8, &[2, 16], 32).unwrap();
        let expected: Vec<u8> = (0..32).collect();
        assert_eq!(Vec::<u8>::from(stored.into_buffer()), expected);
        for (dtype, shape, bytes, rows) in [
            (Dtype::I8, &[2, 16][..], 32, 2),
            (Dtype::U8, &[1, 2, 16][..], 32, 1),
            (Dtype::U8, &[2, 8][..], 16, 2),
        ] {
            let reason =
                format!("expected 64 packed E2M1 codes in {rows} U8 rows, two codes a byte");
            let refused = ("weight".to_owned(), reason);
            assert_eq!(
                invalid(stored_as(dtype, shape, bytes)),
                refused,
                "{dtype:?} {shape:?}"
            );
        }
        assert_eq!(
            missing(read(&[], "weight", codes::<m![1], m![Outputs, Inputs]>)),
            ["weight"]
        );
    }

    /// A tensor scale is one finite positive FP32 value, stored as a scalar or one element; another
    /// dtype, shape or value is refused, and an absent scale is missing.
    #[test]
    fn reads_one_finite_positive_fp32_scale() {
        let f32s = |value: f32, count: usize| value.to_le_bytes().repeat(count);
        let scale = |from: &Checkpoint, cx: &mut Context<'_>| scales(from, cx, ["s"]).map(|[s]| s);
        let at = |dtype, shape: &[usize], bytes| read(&[("s", dtype, shape, bytes)], "", scale);
        assert_eq!(at(Dtype::F32, &[], f32s(0.5, 1)).unwrap(), 0.5);
        assert_eq!(at(Dtype::F32, &[1], f32s(0.5, 1)).unwrap(), 0.5);
        let (one, positive) = (
            "expected one FP32 scale",
            "the scale must be finite and positive",
        );
        for (dtype, shape, bytes, reason) in [
            (
                Dtype::BF16,
                &[][..],
                half::bf16::ONE.to_le_bytes().to_vec(),
                one,
            ),
            (Dtype::F32, &[2][..], f32s(0.5, 2), one),
            (Dtype::F32, &[][..], f32s(0.0, 1), positive),
            (Dtype::F32, &[][..], f32s(-0.5, 1), positive),
            (Dtype::F32, &[][..], f32s(f32::NAN, 1), positive),
            (Dtype::F32, &[][..], f32s(f32::INFINITY, 1), positive),
        ] {
            let refused = ("s".to_owned(), reason.to_owned());
            assert_eq!(
                invalid(at(dtype, shape, bytes)),
                refused,
                "{dtype:?} {shape:?}"
            );
        }
        assert_eq!(missing(read(&[], "", scale)), ["s"]);
    }
}
