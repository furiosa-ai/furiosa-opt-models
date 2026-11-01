//! PEFT's LoRA checkpoint format: the configuration of its updates and the factors it stores.

use std::collections::BTreeMap;

use safetensors::Dtype;
use serde_json::Value;

use super::RANK;
use crate::load::{Checkpoint, LoadError};

/// What a PEFT LoRA checkpoint's `adapter_config.json` declares of its updates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// The rank of every update: `r`.
    pub rank: usize,
    /// The factor of every update: `lora_alpha / r`, or `lora_alpha / sqrt(r)` under rsLoRA.
    pub scale: f32,
}

/// A configuration field that changes the update unless `unset` holds of its value, and what it
/// adds.
struct Form {
    field: &'static str,
    unset: fn(&Value) -> bool,
    adds: &'static str,
}

impl Form {
    /// A field that adds nothing when absent, null, false or empty.
    const fn new(field: &'static str, adds: &'static str) -> Self {
        Self {
            field,
            unset: |value| match value {
                Value::Null | Value::Bool(false) => true,
                Value::Array(values) => values.is_empty(),
                Value::Object(entries) => entries.is_empty(),
                _ => false,
            },
            adds,
        }
    }
}

const FORMS: [Form; 15] = [
    Form::new("use_dora", "DoRA magnitudes"),
    Form {
        field: "bias",
        unset: |value| value.is_null() || value == "none",
        adds: "bias terms",
    },
    Form::new("lora_bias", "bias terms"),
    Form::new("modules_to_save", "replaced modules"),
    Form::new("rank_pattern", "per-module ranks"),
    Form::new("alpha_pattern", "per-module scales"),
    Form::new("fan_in_fan_out", "transposed factors"),
    Form::new("trainable_token_indices", "trained token embeddings"),
    Form::new("target_parameters", "adapted parameters"),
    Form::new("alora_invocation_tokens", "activated LoRA"),
    Form::new("use_qalora", "QA-LoRA pooling"),
    Form::new("layer_replication", "replicated layers"),
    Form::new("use_bdlora", "block-diagonal factors"),
    Form::new("arrow_config", "Arrow routing"),
    Form::new("kasa_config", "KaSA singular values"),
];

/// A configuration is a LoRA one of a rank the terms hold and a scale FP32 holds, none of whose
/// fields changes the update a term computes.
impl TryFrom<&Value> for Config {
    type Error = LoadError;

    fn try_from(config: &Value) -> Result<Self, LoadError> {
        let name = |field: &str| format!("adapter_config.json:{field}");
        let invalid = |field: &str, reason: String| LoadError::InvalidData {
            name: name(field),
            reason,
        };
        if let Some(kind) = config.get("peft_type").filter(|kind| *kind != "LORA") {
            return Err(invalid("peft_type", format!("{kind}, not LORA")));
        }
        if let Some(Form { field, adds, .. }) = FORMS
            .iter()
            .find(|form| (config.get(form.field)).is_some_and(|value| !(form.unset)(value)))
        {
            return Err(LoadError::Unsupported {
                name: name(field),
                reason: format!("it adds {adds}, which no term holds"),
            });
        }
        let rank = (config.get("r").and_then(Value::as_u64))
            .filter(|&rank| rank > 0)
            .ok_or_else(|| invalid("r", "not a positive integer".into()))?
            as usize;
        if rank > RANK {
            return Err(LoadError::Unsupported {
                name: name("r"),
                reason: format!("{rank}, past the rank {RANK} a term holds"),
            });
        }
        // PEFT's default numerator of the scale.
        let alpha = config.get("lora_alpha").map_or(Some(8.0), Value::as_f64);
        let alpha = alpha.ok_or_else(|| invalid("lora_alpha", "not a number".into()))?;
        let rslora = match config.get("use_rslora") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(rslora)) => *rslora,
            Some(other) => return Err(invalid("use_rslora", format!("{other}, not a boolean"))),
        };
        let denominator = match rslora {
            true => (rank as f64).sqrt(),
            false => rank as f64,
        };
        let scale = (alpha / denominator) as f32;
        if !scale.is_finite() {
            return Err(invalid(
                "lora_alpha",
                format!("{alpha} scales the update by {scale} in FP32"),
            ));
        }
        Ok(Self { rank, scale })
    }
}

/// The factors a PEFT checkpoint stores of each module it adapts, below the module's name.
pub(super) const FACTORS: [&str; 2] = ["lora_A.weight", "lora_B.weight"];

/// `checkpoint`, a PEFT LoRA checkpoint, read once with its configuration applied: each module's
/// `lora_A` and its `lora_B` times the scale, in FP32 and of the shapes stored, and every other
/// tensor as stored, which no term reads. A lone factor, one that is no matrix of floats of the
/// configured rank, and a configuration no term follows are refused.
pub(super) fn scaled(checkpoint: &Checkpoint) -> Result<Checkpoint, LoadError> {
    let Config { rank, scale } = Config::try_from(checkpoint.config())?;
    // The module of a factor's name, and which of the two factors it names.
    let factor = |name: &str| {
        FACTORS.iter().enumerate().find_map(|(at, factor)| {
            let module = name.strip_suffix(factor)?.strip_suffix('.')?;
            Some((module.to_string(), at))
        })
    };
    let mut pairs: BTreeMap<String, [bool; 2]> = BTreeMap::new();
    for (module, at) in checkpoint.names().iter().filter_map(|name| factor(name)) {
        pairs.entry(module).or_default()[at] = true;
    }
    let missing: Vec<String> = (pairs.iter())
        .flat_map(|(module, held)| {
            (FACTORS.iter().zip(held))
                .filter(|(_, held)| !**held)
                .map(move |(factor, _)| format!("{module}.{factor}"))
        })
        .collect();
    if !missing.is_empty() {
        return Err(LoadError::missing(missing));
    }
    let mut scaled = Checkpoint::default();
    for name in checkpoint.names() {
        let view = checkpoint.get(&name).expect("a name the checkpoint lists");
        let (dtype, data) = match factor(&name) {
            None => (view.dtype(), view.data().to_vec()),
            Some((_, at)) => {
                let invalid = |reason: String| LoadError::InvalidData {
                    name: name.clone(),
                    reason,
                };
                // `lora_A` is `[rank, inputs]` and `lora_B` `[outputs, rank]`.
                let shape = view.shape();
                if shape.len() != 2 {
                    return Err(invalid(format!("shape {shape:?}, not a matrix")));
                }
                if shape[at] != rank {
                    let reason = format!(
                        "shape {shape:?}, of rank {}, not the configured {rank}",
                        shape[at]
                    );
                    return Err(invalid(reason));
                }
                let dtype = view.dtype();
                let values = Checkpoint::floats(&view)
                    .ok_or_else(|| invalid(format!("stored as {dtype:?}, not a float")))?;
                // `lora_B` carries the scale.
                let times = if at == 1 { scale } else { 1.0 };
                let bytes = values
                    .into_iter()
                    .flat_map(|value| (times * value).to_le_bytes());
                (Dtype::F32, bytes.collect())
            }
        };
        scaled
            .insert(&name, dtype, view.shape(), data)
            .expect("each name once, its bytes filling its shape");
    }
    Ok(scaled)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A configuration as PEFT writes it reads as its rank and its scale under each rule; each
    /// field that changes the update, a rank past the terms', a scale past FP32 and a malformed
    /// value, such as an rsLoRA flag that is no boolean, are refused, naming the field.
    #[test]
    fn reads_peft_configurations() {
        let config = |rslora: bool| {
            serde_json::json!({"peft_type": "LORA", "r": 16, "lora_alpha": 32, "use_rslora": rslora,
                "bias": "none", "use_dora": false, "rank_pattern": {}, "modules_to_save": null})
        };
        let read = |value: &Value| Config::try_from(value);
        assert_eq!(
            read(&config(false)).unwrap(),
            Config {
                rank: 16,
                scale: 2.0
            }
        );
        assert_eq!(
            read(&config(true)).unwrap(),
            Config {
                rank: 16,
                scale: 8.0
            }
        );
        let with = |field: &str, value: Value| {
            let mut config = config(false);
            config[field] = value;
            config
        };
        let refused = [
            (with("use_dora", true.into()), "use_dora", true),
            (with("bias", "all".into()), "bias", true),
            (
                with("modules_to_save", serde_json::json!(["lm_head"])),
                "modules_to_save",
                true,
            ),
            (
                with("rank_pattern", serde_json::json!({"q_proj": 8})),
                "rank_pattern",
                true,
            ),
            (with("r", 64.into()), "r", true),
            (with("r", 0.into()), "r", false),
            (with("peft_type", "IA3".into()), "peft_type", false),
            (with("lora_alpha", "x".into()), "lora_alpha", false),
            (
                with("lora_alpha", serde_json::json!(1e40)),
                "lora_alpha",
                false,
            ),
            (with("use_rslora", "true".into()), "use_rslora", false),
            (with("use_rslora", 1.into()), "use_rslora", false),
        ];
        for (config, field, unsupported) in refused {
            let field = format!("adapter_config.json:{field}");
            match (read(&config), unsupported) {
                (Err(LoadError::Unsupported { name, .. }), true)
                | (Err(LoadError::InvalidData { name, .. }), false) => assert_eq!(name, field),
                (other, _) => panic!("{field}: {other:?}"),
            }
        }
    }
}
