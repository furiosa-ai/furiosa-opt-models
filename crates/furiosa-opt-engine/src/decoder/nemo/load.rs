//! The checkpoint read strictly into the step's weights, each packed on the host as its piece reads
//! it, and the step's state allocated beside them.
//!
//! - A projection's FP8 weight packs into its TP2 layout, each chip's part layer after layer; its
//!   input and weight scales fill the factor row its piece lands, a copy on each chip.
//! - Q, K and V share one input scale, which the step codes the shared input with; a checkpoint
//!   whose three differ is refused.
//! - The readout's head and its greedy and stop tables take the vocabulary in the readout's order;
//!   the stops are the config's end-of-sequence tokens.

use super::{Nemo, Resident};
use crate::{Checkpoint, Load, LoadError};
use furiosa_opt_models::load::Context;
use furiosa_opt_nemo::models::nemo::{LAYERS, parameters, plan, serving};
use furiosa_opt_nemo::ops::linear::{mlp, output, pair, query};
use furiosa_opt_nemo::ops::normalization::RmsNorm;
use furiosa_opt_nemo::ops::vocabulary::readout::{Local, Order};
use furiosa_opt_nemo::ops::weight::Weight;
use furiosa_opt_std::prelude::*;
use safetensors::Dtype;
use serving::{Chips, attention, step};
use std::sync::Mutex;

type Hidden = serving::Hidden;
type Vocabulary = serving::step::Vocabulary;
/// A chip's share of the vocabulary, in the readout's order.
type Shard = serving::head::Vocabulary;
type Selected = step::Selected;

/// Floats of one factor row: a kernel's scales, eight lanes each.
const FACTORS: usize = 64;

impl<'a> Load<&'a Checkpoint> for Nemo {
    fn load(context: &mut Context<'_>, source: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(source, |cx| {
            let model = Reader { cx, source }.model()?;
            let resident = Resident::alloc(cx.device()).map_err(device("state"))?;
            Ok(Self {
                model,
                resident: Mutex::new(resident),
            })
        })
    }
}

/// A traversal of the checkpoint's tensors onto the device, each name marked read.
struct Reader<'c, 'd, 's> {
    cx: &'c mut Context<'d>,
    source: &'s Checkpoint,
}

impl<'s> Reader<'_, '_, 's> {
    fn model(mut self) -> Result<parameters::Model, LoadError> {
        let stops = self.stops()?;
        let order =
            Order::<parameters::Readout>::new(&stops).map_err(|error| LoadError::InvalidData {
                name: "config.json:eos_token_id".into(),
                reason: error.to_string(),
            })?;
        let embedding = {
            let shape = [Vocabulary::SIZE, Hidden::SIZE];
            let table = self.bytes("model.embed_tokens.weight", Dtype::BF16, &shape)?;
            parameters::Embedding::new(self.replicated("model.embed_tokens.weight", table)?)
        };
        let layers = self.layers()?;
        let norm = self.bytes("model.norm.weight", Dtype::BF16, &[Hidden::SIZE])?;
        let norm = RmsNorm::new(Weight::new(self.replicated("model.norm.weight", norm)?));
        let head = self.head(&order)?;
        let tokens = |chip: usize| {
            &order.tokens()[chip * Local::<parameters::Readout>::SIZE..]
                [..Local::<parameters::Readout>::SIZE]
        };
        let indices: Vec<u8> = (0..2)
            .flat_map(|chip| tokens(chip).iter())
            .flat_map(|&token| ((Vocabulary::SIZE - 1 - token) as i32).to_le_bytes())
            .collect();
        let stops: Vec<u8> = (0..2)
            .flat_map(|chip| tokens(chip).iter())
            .flat_map(|token| match order.stops().binary_search(token) {
                Ok(_) => f32::MIN.to_le_bytes(),
                Err(_) => 0_f32.to_le_bytes(),
            })
            .collect();
        Ok(parameters::Model {
            embedding,
            layers,
            norm,
            head,
            greedy: parameters::Greedy {
                indices: self.split("greedy.indices", &indices)?,
            },
            sampler: parameters::Sampling {
                stops: self.split("sampler.stops", &stops)?,
            },
        })
    }

    /// Every layer's weights and scales, stacked layer after layer on each chip.
    fn layers(&mut self) -> Result<parameters::Layers, LoadError> {
        let at = |layer: usize, name: &str| format!("model.layers.{layer}.{name}");
        let projection = |reader: &mut Self, layer: usize, name: &str, shape: [usize; 2]| {
            reader.bytes(
                &at(layer, &format!("{name}.weight")),
                Dtype::F8_E4M3,
                &shape,
            )
        };
        let scales = |reader: &mut Self, layer: usize, name: &str| -> Result<[f32; 2], LoadError> {
            Ok([
                reader.scale(&at(layer, &format!("{name}.input_scale")))?,
                reader.scale(&at(layer, &format!("{name}.weight_scale")))?,
            ])
        };
        // Each factor row's lanes `8 f .. 8 f + 8` hold factor `f`.
        let row = |factors: &[f32]| -> Vec<u8> {
            let mut row = vec![0_f32; FACTORS];
            for (factor, &value) in factors.iter().enumerate() {
                row[factor * 8..(factor + 1) * 8].fill(value);
            }
            let bytes: Vec<u8> = row.iter().flat_map(|value| value.to_le_bytes()).collect();
            bytes.repeat(2)
        };
        let norm = |reader: &mut Self, layer: usize, name: &str| -> Result<Vec<u8>, LoadError> {
            let shape = [Hidden::SIZE];
            Ok(reader
                .bytes(&at(layer, name), Dtype::BF16, &shape)?
                .repeat(2))
        };
        let (hidden, queries, pairs) = (Hidden::SIZE, 4096, 1024);
        let intermediate = 2 * serving::mlp::Intermediate::SIZE;
        Ok(parameters::Layers {
            norm: self.stacked("input_layernorm", |reader, layer| {
                norm(reader, layer, "input_layernorm.weight")
            })?,
            query: self.stacked("self_attn.q_proj", |reader, layer| {
                let weight = projection(reader, layer, "self_attn.q_proj", [queries, hidden])?;
                Ok(query::packed::Parameters::<parameters::Query>::pack(weight))
            })?,
            pairs: self.stacked("self_attn.k_proj", |reader, layer| {
                let keys = projection(reader, layer, "self_attn.k_proj", [pairs, hidden])?;
                let values = projection(reader, layer, "self_attn.v_proj", [pairs, hidden])?;
                Ok(pair::Parameters::<parameters::KeyValue, 0>::pack(
                    keys, values,
                ))
            })?,
            qkv_scales: self.stacked("self_attn.q_proj.scales", |reader, layer| {
                let [input, weight] = scales(reader, layer, "self_attn.q_proj")?;
                for name in ["self_attn.k_proj", "self_attn.v_proj"] {
                    let name = at(layer, &format!("{name}.input_scale"));
                    if reader.scale(&name)?.to_bits() != input.to_bits() {
                        return Err(LoadError::InvalidData {
                            name,
                            reason: "Q, K and V must share the input scale".into(),
                        });
                    }
                }
                Ok(row(&[input, weight]))
            })?,
            pair_scale: self.stacked("self_attn.k_proj.scales", |reader, layer| {
                let keys = reader.scale(&at(layer, "self_attn.k_proj.weight_scale"))?;
                let values = reader.scale(&at(layer, "self_attn.v_proj.weight_scale"))?;
                // Each chip's row repeats a 256-float key run and a value run.
                let factors = pair::Factors::<parameters::KeyValue>::SIZE;
                Ok((0..2 * factors)
                    .flat_map(|at| [keys, values][at / 256 % 2].to_le_bytes())
                    .collect())
            })?,
            output: self.stacked("self_attn.o_proj", |reader, layer| {
                let weight = projection(reader, layer, "self_attn.o_proj", [hidden, queries])?;
                Ok(output::packed::Parameters::<parameters::Output>::pack(
                    weight,
                ))
            })?,
            output_scales: self.stacked("self_attn.o_proj.scales", |reader, layer| {
                Ok(row(&scales(reader, layer, "self_attn.o_proj")?))
            })?,
            post_norm: self.stacked("post_attention_layernorm", |reader, layer| {
                norm(reader, layer, "post_attention_layernorm.weight")
            })?,
            gate: self.stacked("mlp.gate_proj", |reader, layer| {
                let weight = projection(reader, layer, "mlp.gate_proj", [intermediate, hidden])?;
                Ok(mlp::Parameters::<parameters::GateUp, 0>::pack(weight))
            })?,
            up: self.stacked("mlp.up_proj", |reader, layer| {
                let weight = projection(reader, layer, "mlp.up_proj", [intermediate, hidden])?;
                Ok(mlp::Parameters::<parameters::GateUp, 1>::pack(weight))
            })?,
            down: self.stacked("mlp.down_proj", |reader, layer| {
                let weight = projection(reader, layer, "mlp.down_proj", [hidden, intermediate])?;
                Ok(mlp::Parameters::<parameters::Down, 2>::pack(weight))
            })?,
            mlp_scales: self.stacked("mlp.scales", |reader, layer| {
                let [gate, up, down] = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
                    .map(|name| scales(reader, layer, name));
                Ok(row(&[gate?, up?, down?].concat()))
            })?,
        })
    }

    /// The readout's head: each chip's vocabulary rows in the readout's order, one 256-feature
    /// tile after another.
    fn head(
        &mut self,
        order: &Order<parameters::Readout>,
    ) -> Result<HbmTensor<bf16, Chips, m![Hidden / 256, Shard, Hidden % 256]>, LoadError> {
        let shape = [Vocabulary::SIZE, Hidden::SIZE];
        let source = self.bytes("lm_head.weight", Dtype::BF16, &shape)?;
        let mut bytes = Vec::with_capacity(source.len());
        for tokens in order
            .tokens()
            .chunks_exact(Local::<parameters::Readout>::SIZE)
        {
            for tile in 0..Hidden::SIZE / 256 {
                for &token in tokens {
                    let at = (token * Hidden::SIZE + tile * 256) * 2;
                    bytes.extend_from_slice(&source[at..at + 512]);
                }
            }
        }
        self.split("lm_head.weight", &bytes)
    }

    /// The config's end-of-sequence tokens: one ID, a list of them, or none.
    fn stops(&mut self) -> Result<Vec<usize>, LoadError> {
        let invalid = |reason: &str| LoadError::InvalidData {
            name: "config.json:eos_token_id".into(),
            reason: reason.into(),
        };
        let id = |value: &serde_json::Value| {
            (value.as_u64().and_then(|id| usize::try_from(id).ok()))
                .ok_or_else(|| invalid("not a token ID"))
        };
        match self.source.config().get("eos_token_id") {
            None | Some(serde_json::Value::Null) => Ok(Vec::new()),
            Some(serde_json::Value::Array(ids)) => ids.iter().map(id).collect(),
            Some(value) => Ok(vec![id(value)?]),
        }
    }

    /// The bytes of tensor `name`, which must hold `dtype` elements of `shape`; marked read.
    fn bytes(&mut self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&'s [u8], LoadError> {
        self.cx.read(name);
        let tensor =
            (self.source.get(name)).ok_or_else(|| LoadError::Missing(vec![name.into()]))?;
        let scalar = shape.is_empty() && matches!(tensor.shape(), [] | [1]);
        if tensor.dtype() != dtype || !(scalar || tensor.shape() == shape) {
            return Err(LoadError::InvalidData {
                name: name.into(),
                reason: format!(
                    "{:?} {:?}, the step reads {dtype:?} {shape:?}",
                    tensor.dtype(),
                    tensor.shape()
                ),
            });
        }
        Ok(tensor.data())
    }

    /// The FP32 scale `name`, finite and positive.
    fn scale(&mut self, name: &str) -> Result<f32, LoadError> {
        let bytes = self.bytes(name, Dtype::F32, &[])?;
        let value = f32::from_le_bytes(bytes.try_into().map_err(|_| LoadError::InvalidData {
            name: name.into(),
            reason: "not one FP32 value".into(),
        })?);
        match value.is_finite() && value > 0.0 {
            true => Ok(value),
            false => Err(LoadError::InvalidData {
                name: name.into(),
                reason: format!("scale {value} is not finite and positive"),
            }),
        }
    }

    /// Every layer's `pack`, each chip-major, layer after layer on each chip; `name` names the
    /// leaf a device failure reports.
    fn stacked<D: Scalar, E: M>(
        &mut self,
        name: &str,
        mut pack: impl FnMut(&mut Self, usize) -> Result<Vec<u8>, LoadError>,
    ) -> Result<parameters::Stacked<D, E>, LoadError> {
        let size = parameters::Stacked::<D, E>::size() / Chips::SIZE;
        let per = size / LAYERS;
        let failed = device(name);
        let mut buffer = HbmBuffer::alloc(self.cx.device(), size).map_err(&failed)?;
        for layer in 0..LAYERS {
            let packed = pack(self, layer)?;
            if packed.len() != 2 * per {
                return Err(LoadError::InvalidData {
                    name: name.into(),
                    reason: format!(
                        "a layer packs {} bytes, the step reads {}",
                        packed.len(),
                        2 * per
                    ),
                });
            }
            for chip in 0..2 {
                let target = buffer.slice_mut(chip, layer * per..(layer + 1) * per);
                let write = (self.cx.device().pdma).to_hbm([(
                    &packed[chip * per..(chip + 1) * per],
                    target.map_err(&failed)?,
                )]);
                futures_executor::block_on(write.into_future()).map_err(&failed)?;
            }
        }
        HbmTensor::try_from(buffer).map_err(failed)
    }

    /// One copy of `bytes` on each of the two chips `C` names.
    fn replicated<D: Scalar, C: M, E: M>(
        &mut self,
        name: &str,
        bytes: &[u8],
    ) -> Result<HbmTensor<D, C, E>, LoadError> {
        let failed = device(name);
        let mut buffer = HbmBuffer::alloc(self.cx.device(), bytes.len()).map_err(&failed)?;
        for chip in 0..2 {
            let target = buffer.slice_mut(chip, ..).map_err(&failed)?;
            let write = self.cx.device().pdma.to_hbm([(bytes, target)]);
            futures_executor::block_on(write.into_future()).map_err(&failed)?;
        }
        HbmTensor::try_from(buffer).map_err(failed)
    }

    /// `bytes` split over the chips, each chip's part after the one before.
    fn split<D: Scalar, E: M>(
        &mut self,
        name: &str,
        bytes: &[u8],
    ) -> Result<HbmTensor<D, Chips, E>, LoadError> {
        let failed = device(name);
        let write = HostBufferSlice::from(bytes).to_hbm(&mut self.cx.device().pdma);
        let buffer = futures_executor::block_on(write.into_future()).map_err(&failed)?;
        HbmTensor::try_from(buffer).map_err(failed)
    }
}

impl Resident {
    /// The step's state on `device`: its pool comes from each call's cache.
    fn alloc(device: &mut Device) -> Result<Self, furiosa_opt_std::Error> {
        let bytes = HbmTensor::<i32, Chips, m![Selected, 1 # 4]>::size() / Chips::SIZE;
        let tokens = HbmBuffer::alloc(device, bytes)?;
        let state = step::State {
            hidden: HbmTensor::alloc(device)?,
            queries: HbmTensor::alloc(device)?,
            keys: HbmTensor::alloc(device)?,
            values: HbmTensor::alloc(device)?,
            pool: Vec::new().into(),
            partials: attention::Partials {
                stats: blocks(device)?,
                output: blocks(device)?,
            },
            logits: HbmTensor::alloc(device)?,
            tokens: HbmTensor::try_from(tokens.alias(..)?)?,
            prompt: HbmTensor::alloc(device)?,
        };
        Ok(Self { state, tokens })
    }
}

/// The attention partials' blocks, one strided run of them, so their gathers bind one table.
fn blocks<D: Scalar, E: M>(
    device: &mut Device,
) -> Result<Slice<HbmTensor<D, Chips, E>>, furiosa_opt_std::Error> {
    let bytes = HbmTensor::<D, Chips, E>::size() / Chips::SIZE;
    let buffer = HbmBuffer::alloc(device, plan::PARTIALS * bytes)?;
    (0..plan::PARTIALS)
        .map(|block| HbmTensor::try_from(buffer.alias(block * bytes..(block + 1) * bytes)?))
        .collect()
}

/// A device failure at the leaf `path`.
fn device(path: &str) -> impl Fn(furiosa_opt_std::Error) -> LoadError + '_ {
    move |source| LoadError::Device {
        path: path.into(),
        source,
    }
}
