//! FP8 projections a load quantizes from BF16: the codes and channel scales of a stored weight.
//!
//! - In a checkpoint read [quantized](Checkpoint::quantized), each channel-scaled projection holds
//!   the codes and scales of the BF16 weight stored at its name, under one [`Granularity`].
//! - Frameworks fuse the projections that read one input into one matrix (a layer's query, key and
//!   value projections; its gate and up projections) and scale it as one tensor.

use furiosa_opt_std::prelude::*;
use safetensors::Dtype;
use safetensors::tensor::TensorView;

use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::tensor::form::fp8::{Finite, Granularity};

/// The projections that read one input, by their checkpoint names.
const FUSED: [&[&str]; 2] = [&["q_proj", "k_proj", "v_proj"], &["gate_proj", "up_proj"]];

/// The host value of a channel-scaled `[Out, In]` weight's E4M3 codes on every chip of `C`.
pub(crate) type Codes<In, Out, C> = HostTensor<f8e4m3, m![{ C }, { m![Out, In] }]>;

/// The host value of a channel-scaled weight's row scales on every chip of `C`, each in the first
/// of eight lanes.
pub(crate) type Scales<Out, C> = HostTensor<f32, m![{ C }, { m![Out, 1 # 8] }]>;

/// The host values a channel-scaled weight loads quantized: its codes, then its row scales.
pub(crate) type Quantized<In, Out, C> = (Codes<In, Out, C>, Scales<Out, C>);

/// A leaf of E4M3 codes in `[Out, In]` order on chips `C`: where a checkpoint storing its
/// projection in BF16 keeps the values the codes quantize, and the host value its codes load from.
pub(crate) trait Unquantized<In: AxisName, Out: AxisName, C: M> {
    /// The host value the leaf loads its codes from.
    type Host;

    /// The values at the context's position, finite.
    fn read(from: &Checkpoint, context: &mut Context<'_>) -> Result<Finite, LoadError>;

    /// `codes` as the host value the leaf loads.
    fn host(codes: Codes<In, Out, C>) -> Self::Host;
}

/// A leaf of any memory that loads its codes as they are quantizes the weight stored at its name.
impl<In: AxisName, Out: AxisName, C: M, T: Load<Codes<In, Out, C>>> Unquantized<In, Out, C> for T {
    type Host = Codes<In, Out, C>;

    fn read(from: &Checkpoint, context: &mut Context<'_>) -> Result<Finite, LoadError> {
        from.unquantized::<In, Out>(context)
    }

    fn host(codes: Codes<In, Out, C>) -> Codes<In, Out, C> {
        codes
    }
}

impl Checkpoint {
    /// The finite values of the `[Out, In]` weight at the context's name, stored in BF16, F16 or
    /// FP32; recorded as read.
    pub(crate) fn unquantized<In: AxisName, Out: AxisName>(
        &self,
        context: &mut Context<'_>,
    ) -> Result<Finite, LoadError> {
        let (name, view) = context.tensor(self)?;
        if Self::floats(&view).is_none() {
            return Err(LoadError::InvalidData {
                name,
                reason: format!("stored as {:?}, a load quantizes BF16", view.dtype()),
            });
        }
        let values = Self::value::<f32, m![Out, In], m![Out, In]>(&view, &name)?.into_vec();
        Self::finite(values, In::SIZE, &name)
    }

    /// `values` in rows of `inputs`, unless one is not finite, which leaves the weight at `name`
    /// without a scale.
    pub(crate) fn finite(values: Vec<f32>, inputs: usize, name: &str) -> Result<Finite, LoadError> {
        Finite::new(values, inputs).ok_or_else(|| LoadError::InvalidData {
            name: name.into(),
            reason: "holds a value that is not finite, so it has no scale".into(),
        })
    }

    /// The codes and row scales of the projection at the context's position: `weight` under this
    /// checkpoint's granularity, a tensor scale covering the weights fused with it.
    pub(crate) fn channel_scaled<In: AxisName, Out: AxisName, C: M>(
        &self,
        context: &mut Context<'_>,
        weight: Finite,
    ) -> Result<Quantized<In, Out, C>, LoadError> {
        let granularity = self.quantization.expect("a load quantizes this checkpoint");
        let fused = match granularity {
            Granularity::Tensor => self.fused(context)?,
            Granularity::Channel => 0.0,
        };
        let (codes, scales) = granularity.quantize(&weight, fused);
        let name = context.name();
        let codes = f8e4m3::to_buf(&codes);
        let codes = TensorView::new(Dtype::F8_E4M3, vec![Out::SIZE, In::SIZE], &codes);
        let scales: Vec<u8> = scales
            .iter()
            .flat_map(|scale| scale.to_le_bytes())
            .collect();
        let scales = TensorView::new(Dtype::F32, vec![Out::SIZE], &scales);
        Ok((
            Self::value::<_, m![Out, In], _>(&codes.expect("a code an element"), &name)?,
            Self::value::<_, m![Out, 1 # 8], _>(&scales.expect("a scale a row"), &name)?,
        ))
    }

    /// The largest magnitude of the weights fused with the projection at the context's position,
    /// each recorded as read; zero for a projection fused with none.
    fn fused(&self, context: &mut Context<'_>) -> Result<f32, LoadError> {
        let name = context.name();
        let own = name.rsplit('.').next().unwrap_or_default();
        let group = FUSED.iter().find(|group| group.contains(&own));
        let mut siblings = group
            .into_iter()
            .flat_map(|group| group.iter())
            .filter(|&&part| part != own);
        siblings.try_fold(0.0f32, |largest, part| {
            let name = format!("{}.weight", context.sibling(part));
            context.read(&name);
            let view = (self.get(&name)).ok_or_else(|| LoadError::Missing(vec![name.clone()]))?;
            let invalid = || LoadError::InvalidData {
                name: name.clone(),
                reason: format!("stored as {:?}, a load quantizes BF16", view.dtype()),
            };
            let values = Self::floats(&view).ok_or_else(invalid)?;
            let inputs = view.shape().last().copied().unwrap_or(1).max(1);
            Ok(largest.max(Self::finite(values, inputs, &name)?.largest()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::checkpoint::tests::{bf16s, checkpoint, invalid, missing};

    axes![Rows = 2, Inputs = 4];

    /// `rule` at `name` of `source` read quantized under `granularity`, without a device.
    fn at<T>(
        source: &Checkpoint,
        granularity: Granularity,
        name: &str,
        rule: impl FnOnce(&Checkpoint, &mut Context<'_>) -> Result<T, LoadError>,
    ) -> Result<T, LoadError> {
        let quantized = source.quantized(granularity)?;
        Context::new(None).field(name, name, |cx| rule(&quantized, cx))
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let bytes = values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_bits().to_le_bytes());
        bytes.collect()
    }

    /// Gate and up weights; the up weight's largest magnitude is 8, the gate weight's 1.
    fn mlp() -> Checkpoint {
        let gate = bf16_bytes(&[1.0, -0.5, 0.25, 0.0, 0.125, -1.0, 0.5, 0.75]);
        let up = bf16_bytes(&[2.0, -8.0, 4.0, 1.0, 0.5, 0.25, -2.0, 3.0]);
        checkpoint(&[
            ("mlp.gate_proj.weight", Dtype::BF16, &[2, 4], gate),
            ("mlp.up_proj.weight", Dtype::BF16, &[2, 4], up),
            ("mlp.down_proj.weight", Dtype::BF16, &[4, 2], bf16s(8, 1.0)),
        ])
    }

    fn quantized(
        source: &Checkpoint,
        granularity: Granularity,
        name: &str,
    ) -> Result<Quantized<Inputs, Rows, m![2]>, LoadError> {
        at(source, granularity, name, |from, cx| {
            let weight = cx.field("weight", "weight", |cx| {
                from.unquantized::<Inputs, Rows>(cx)
            })?;
            from.channel_scaled::<Inputs, Rows, m![2]>(cx, weight)
        })
    }

    /// A tensor scale covers the projection fused with the one loading, a channel scale its own
    /// row; each scale is written to the first of its row's eight lanes, on every chip.
    #[test]
    fn scales_a_fused_projection_as_one_tensor() {
        let (codes, scales) = quantized(&mlp(), Granularity::Tensor, "mlp.gate_proj").unwrap();
        let lanes =
            |scales: [f32; 2]| scales.map(|scale| [scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            scales.into_vec(),
            lanes([8.0 / 448.0; 2]).as_flattened().repeat(2)
        );
        let codes = codes.into_vec();
        assert_eq!(f32::from(codes[0]), 1.0 * 56.0);
        assert_eq!(codes.len(), 16);
        let (_, scales) = quantized(&mlp(), Granularity::Channel, "mlp.gate_proj").unwrap();
        let rows = lanes([1.0 / 448.0, 1.0 / 448.0]);
        assert_eq!(scales.into_vec(), rows.as_flattened().repeat(2));
    }

    /// A weight of another dtype or shape, holding a value that is not finite, or whose fused
    /// projection is absent is refused, naming it, each with its reason.
    #[test]
    fn refuses_weights_it_cannot_quantize() {
        let source = |gate: (Dtype, &[usize], Vec<u8>)| {
            checkpoint(&[
                ("mlp.gate_proj.weight", gate.0, gate.1, gate.2),
                ("mlp.up_proj.weight", Dtype::BF16, &[2, 4], bf16s(8, 1.0)),
            ])
        };
        let mut infinite = bf16s(8, 1.0);
        infinite[..2].copy_from_slice(&bf16::from_f32(f32::INFINITY).to_bits().to_le_bytes());
        for (gate, reason) in [
            (
                (Dtype::F8_E4M3, &[2, 4][..], vec![0x38; 8]),
                "stored as F8_E4M3, a load quantizes BF16",
            ),
            (
                (Dtype::BF16, &[4, 2][..], bf16s(8, 1.0)),
                "shape [4, 2], the leaf holds [2, 4]",
            ),
            (
                (Dtype::BF16, &[2, 4][..], infinite),
                "holds a value that is not finite, so it has no scale",
            ),
        ] {
            let error = quantized(&source(gate), Granularity::Channel, "mlp.gate_proj").err();
            let refused = ("mlp.gate_proj.weight".to_owned(), reason.to_owned());
            assert_eq!(invalid(error), refused);
        }
        let alone = checkpoint(&[("mlp.gate_proj.weight", Dtype::BF16, &[2, 4], bf16s(8, 1.0))]);
        let error = quantized(&alone, Granularity::Tensor, "mlp.gate_proj").err();
        assert_eq!(missing(error), ["mlp.up_proj.weight"]);
    }
}
