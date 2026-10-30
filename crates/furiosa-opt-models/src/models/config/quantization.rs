//! Checkpoint quantization declarations: E4M3 weights under static per-tensor scales, 128 x 128
//! block scales or per-channel scales, or E2M1 weights and inputs with E4M3 scales per 16 inputs
//! under static FP32 tensor scales, and the modules a checkpoint leaves unquantized.
//!
//! - A mixed-precision checkpoint declares one scheme for each module group.

use super::{
    Absent, Constant, Disabled, Enabled, Exclusion, Exclusions, Literal, UNSUPPORTED, flattened,
    located, refuse,
};
use crate::tensor::form::fp8::GROUP;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

/// The quantization a checkpoint's `quantization_config` declares: how its projections scale, the
/// modules it leaves unquantized and whether its KV cache is quantized too.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(try_from = "Value")]
pub struct Quantization {
    /// The scheme of each module group, once, in `Scheme` order.
    pub schemes: Vec<Scheme>,
    /// The modules left unquantized: by name, by a name prefix, or by a part of their name.
    pub excluded: Exclusions,
    /// Whether the checkpoint declares an E4M3 KV cache. An NVFP4 cache is at unit scale, since
    /// the NVFP4 parameter check refuses cache scales; a BF16 cache keeps its values unrounded.
    pub kv_cache: bool,
}

/// How a checkpoint quantizes its projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Scheme {
    /// E4M3 weights and input activations, each under one static scale.
    Tensor,
    /// E4M3 weights with one scale per 128 x 128 block; the activations quantize at run time per
    /// group of 128 inputs.
    Block,
    /// E4M3 weights with one static scale per output channel; the activations quantize at run time
    /// per token, if at all.
    Channel,
    /// E2M1 weights and inputs with an E4M3 scale per 16 inputs under a static FP32 tensor scale.
    Nvfp4,
}

/// A checkpoint's `quantization_config`, by its quantization method.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "quant_method", rename_all = "lowercase")]
enum Hf {
    Fp8 {
        activation_scheme: Activations,
        #[serde(default)]
        weight_block_size: Option<[Constant<GROUP>; 2]>,
        ignored_layers: Option<Vec<String>>,
        modules_to_not_convert: Option<Vec<String>>,
        kv_cache_scheme: Option<Tensor>,
        #[serde(default)]
        dequantize: Disabled,
        scale_fmt: Option<Literal<"float">>,
    },
    /// Read through a JSON map, so that its algorithm is a string as the document writes it.
    #[serde(deserialize_with = "flattened")]
    Modelopt(Modelopt),
    /// Its module groups are read once their format is known.
    #[serde(rename = "compressed-tensors")]
    CompressedTensors {
        format: Format,
        quantization_status: Literal<"compressed">,
        config_groups: BTreeMap<String, Value>,
        ignore: Option<Vec<String>>,
        kv_cache_scheme: Option<Tensor>,
        sparsity_config: Option<BTreeMap<String, Value>>,
        transform_config: Option<BTreeMap<String, Value>>,
    },
}

/// How a compressed-tensors checkpoint stores its quantized weights: E4M3 values, packed E2M1
/// values, or each module group in the format it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
enum Format {
    Float,
    Packed,
    Mixed,
}

impl TryFrom<String> for Format {
    type Error = String;

    fn try_from(found: String) -> Result<Self, Self::Error> {
        match found.as_str() {
            "float-quantized" => Ok(Self::Float),
            "nvfp4-pack-quantized" => Ok(Self::Packed),
            "mixed-precision" => Ok(Self::Mixed),
            _ => Err(refuse(
                format_args!("{found:?}"),
                r#""float-quantized", "nvfp4-pack-quantized" or "mixed-precision""#,
            )),
        }
    }
}

/// A ModelOpt `quantization_config`, by its quantization algorithm.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "quant_algo")]
enum Modelopt {
    #[serde(rename = "FP8")]
    Fp8 {
        config_groups: BTreeMap<String, Group<Tensor>>,
        ignore: Option<Vec<String>>,
        kv_cache_scheme: Option<Tensor>,
    },
    #[serde(rename = "NVFP4")]
    Nvfp4 {
        config_groups: BTreeMap<String, Group<Values>>,
        ignore: Option<Vec<String>>,
        kv_cache_scheme: Option<Cache>,
    },
    /// An algorithm the projections do not implement.
    #[serde(other)]
    Other,
}

/// When an FP8 checkpoint scales its activations.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Activations {
    Static,
    Dynamic,
}

/// One quantized module group of a ModelOpt checkpoint, its values of kind `V`.
#[derive(Clone, Debug, Deserialize)]
struct Group<V> {
    weights: V,
    input_activations: V,
    targets: Vec<String>,
}

/// One quantized tensor kind: static, symmetric, per-tensor 8-bit floats.
#[derive(Clone, Debug, Deserialize)]
struct Tensor {
    num_bits: Constant<8>,
    #[serde(rename = "type")]
    kind: Literal<"float">,
    dynamic: Disabled,
    strategy: Option<Literal<"tensor">>,
    symmetric: Option<bool>,
    #[serde(default)]
    group_size: Absent,
    #[serde(default)]
    block_structure: Absent,
    #[serde(default)]
    block_size: Absent,
}

impl Tensor {
    fn symmetric(&self) -> bool {
        let Self {
            num_bits: Constant,
            kind: Literal,
            dynamic: Disabled,
            strategy: None | Some(Literal),
            symmetric,
            group_size: Absent,
            block_structure: Absent,
            block_size: Absent,
        } = self;
        symmetric.is_none_or(|symmetric| symmetric)
    }
}

/// Symmetric four-bit floats in blocks of 16 under a calibrated, static tensor scale.
#[derive(Clone, Debug, Deserialize)]
struct Values {
    num_bits: Constant<4>,
    #[serde(rename = "type")]
    kind: Literal<"float">,
    dynamic: Disabled,
    group_size: Constant<16>,
    symmetric: Option<Enabled>,
    strategy: Option<Literal<"tensor_group">>,
}

/// Static eight-bit float cache values of an NVFP4 checkpoint.
#[derive(Clone, Debug, Deserialize)]
struct Cache {
    num_bits: Constant<8>,
    #[serde(rename = "type")]
    kind: Literal<"float">,
    dynamic: Disabled,
}

impl Group<Tensor> {
    /// Symmetric weights and activations on every linear module.
    fn linear(&self) -> bool {
        self.weights.symmetric() && self.input_activations.symmetric() && self.targets == ["Linear"]
    }
}

impl Group<Values> {
    /// Whether the group quantizes linear layers; deserialization already refined its values.
    fn linear(&self) -> bool {
        let refined = |values: &Values| {
            let Values {
                num_bits: Constant,
                kind: Literal,
                dynamic: Disabled,
                group_size: Constant,
                symmetric: None | Some(Enabled),
                strategy: None | Some(Literal),
            } = values;
        };
        refined(&self.weights);
        refined(&self.input_activations);
        self.targets == ["Linear"]
    }
}

/// One quantized module group of a compressed-tensors checkpoint: symmetric 8-bit float weights
/// with one scale per 128 x 128 block, and activations scaled at run time per 128 inputs.
#[derive(Clone, Debug, Deserialize)]
struct BlockGroup {
    weights: BlockWeights,
    input_activations: GroupActivations,
    #[serde(default)]
    output_activations: Absent,
    targets: Vec<String>,
}
#[derive(Clone, Debug, Deserialize)]
struct BlockWeights {
    num_bits: Constant<8>,
    #[serde(rename = "type")]
    kind: Literal<"float">,
    symmetric: bool,
    dynamic: Disabled,
    strategy: Literal<"block">,
    block_structure: [Constant<GROUP>; 2],
    #[serde(default)]
    group_size: Absent,
    #[serde(default)]
    actorder: Absent,
    #[serde(default)]
    scale_dtype: Absent,
    #[serde(default)]
    zp_dtype: Absent,
}

#[derive(Clone, Debug, Deserialize)]
struct GroupActivations {
    num_bits: Constant<8>,
    #[serde(rename = "type")]
    kind: Literal<"float">,
    symmetric: bool,
    dynamic: bool,
    strategy: Literal<"group">,
    group_size: Constant<GROUP>,
    #[serde(default)]
    block_structure: Absent,
    #[serde(default)]
    actorder: Absent,
    #[serde(default)]
    scale_dtype: Absent,
    #[serde(default)]
    zp_dtype: Absent,
}

impl BlockGroup {
    /// Symmetric weights and activations, the activations scaled at run time, on every linear
    /// module.
    fn linear(&self) -> bool {
        let Self {
            weights:
                BlockWeights {
                    num_bits: Constant,
                    kind: Literal,
                    symmetric: weights,
                    dynamic: Disabled,
                    strategy: Literal,
                    block_structure: [Constant, Constant],
                    group_size: Absent,
                    actorder: Absent,
                    scale_dtype: Absent,
                    zp_dtype: Absent,
                },
            input_activations:
                GroupActivations {
                    num_bits: Constant,
                    kind: Literal,
                    symmetric: activations,
                    dynamic,
                    strategy: Literal,
                    group_size: Constant,
                    block_structure: Absent,
                    actorder: Absent,
                    scale_dtype: Absent,
                    zp_dtype: Absent,
                },
            output_activations: Absent,
            targets,
        } = self;
        *weights && *activations && *dynamic && targets == &["Linear"]
    }
}

/// One quantized module group of a compressed-tensors checkpoint: symmetric 8-bit float weights
/// with one static scale per output channel, and activations quantized at run time per token, or
/// left as they are.
#[derive(Clone, Debug, Deserialize)]
struct ChannelGroup {
    weights: ChannelWeights,
    input_activations: Option<TokenActivations>,
    #[serde(default)]
    output_activations: Absent,
    targets: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct ChannelWeights {
    num_bits: Constant<8>,
    #[serde(rename = "type")]
    kind: Literal<"float">,
    symmetric: Enabled,
    dynamic: Disabled,
    strategy: Literal<"channel">,
    #[serde(default)]
    group_size: Absent,
    #[serde(default)]
    block_structure: Absent,
    #[serde(default)]
    actorder: Absent,
    #[serde(default)]
    scale_dtype: Absent,
    #[serde(default)]
    zp_dtype: Absent,
}

#[derive(Clone, Debug, Deserialize)]
struct TokenActivations {
    num_bits: Constant<8>,
    #[serde(rename = "type")]
    kind: Literal<"float">,
    symmetric: Enabled,
    dynamic: Enabled,
    strategy: Literal<"token">,
    #[serde(default)]
    group_size: Absent,
    #[serde(default)]
    block_structure: Absent,
    #[serde(default)]
    actorder: Absent,
    #[serde(default)]
    scale_dtype: Absent,
    #[serde(default)]
    zp_dtype: Absent,
}

impl ChannelGroup {
    /// Whether the group quantizes some modules; deserialization already refined its values.
    fn quantizes(&self) -> bool {
        let token = |activations: &TokenActivations| {
            let TokenActivations {
                num_bits: Constant,
                kind: Literal,
                symmetric: Enabled,
                dynamic: Enabled,
                strategy: Literal,
                group_size: Absent,
                block_structure: Absent,
                actorder: Absent,
                scale_dtype: Absent,
                zp_dtype: Absent,
            } = activations;
        };
        let Self {
            weights:
                ChannelWeights {
                    num_bits: Constant,
                    kind: Literal,
                    symmetric: Enabled,
                    dynamic: Disabled,
                    strategy: Literal,
                    group_size: Absent,
                    block_structure: Absent,
                    actorder: Absent,
                    scale_dtype: Absent,
                    zp_dtype: Absent,
                },
            input_activations,
            output_activations: Absent,
            targets,
        } = self;
        input_activations.iter().for_each(token);
        !targets.is_empty()
    }
}

/// One NVFP4 module group of a compressed-tensors checkpoint: symmetric four-bit float weights in
/// blocks of 16 under E4M3 block scales and a static tensor scale, and inputs that quantize so at
/// run time under their static tensor scale, or left as they are.
#[derive(Clone, Debug, Deserialize)]
struct PackedGroup {
    weights: PackedWeights,
    input_activations: Option<PackedInputs>,
    #[serde(default)]
    output_activations: Absent,
    targets: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct PackedWeights {
    num_bits: Constant<4>,
    #[serde(rename = "type")]
    kind: Literal<"float">,
    symmetric: Enabled,
    dynamic: Disabled,
    strategy: Literal<"tensor_group">,
    group_size: Constant<16>,
    #[serde(default)]
    block_structure: Absent,
    /// The order calibration visited the columns in, which leaves them stored in place.
    actorder: Option<Literal<"static">>,
    scale_dtype: Option<Literal<"torch.float8_e4m3fn">>,
    #[serde(default)]
    zp_dtype: Absent,
}

/// Inputs whose E4M3 block scales come at run time under a static tensor scale.
#[derive(Clone, Debug, Deserialize)]
struct PackedInputs {
    num_bits: Constant<4>,
    #[serde(rename = "type")]
    kind: Literal<"float">,
    symmetric: Enabled,
    dynamic: Literal<"local">,
    strategy: Literal<"tensor_group">,
    group_size: Constant<16>,
    #[serde(default)]
    block_structure: Absent,
    #[serde(default)]
    actorder: Absent,
    scale_dtype: Option<Literal<"torch.float8_e4m3fn">>,
    #[serde(default)]
    zp_dtype: Absent,
}

impl PackedGroup {
    /// Whether the group quantizes some modules; deserialization already refined its values.
    fn quantizes(&self) -> bool {
        let inputs = |inputs: &PackedInputs| {
            let PackedInputs {
                num_bits: Constant,
                kind: Literal,
                symmetric: Enabled,
                dynamic: Literal,
                strategy: Literal,
                group_size: Constant,
                block_structure: Absent,
                actorder: Absent,
                scale_dtype: None | Some(Literal),
                zp_dtype: Absent,
            } = inputs;
        };
        let Self {
            weights:
                PackedWeights {
                    num_bits: Constant,
                    kind: Literal,
                    symmetric: Enabled,
                    dynamic: Disabled,
                    strategy: Literal,
                    group_size: Constant,
                    block_structure: Absent,
                    actorder: None | Some(Literal),
                    scale_dtype: None | Some(Literal),
                    zp_dtype: Absent,
                },
            input_activations,
            output_activations: Absent,
            targets,
        } = self;
        input_activations.iter().for_each(inputs);
        !targets.is_empty()
    }
}

impl Format {
    /// The scheme of a module group of a checkpoint stored in this format; only a mixed-precision
    /// checkpoint's groups name formats of their own.
    fn scheme(self, group: &Value) -> Result<Scheme, String> {
        let read = |error: serde_json::Error| error.to_string();
        let unsupported = |message: &str| format!("{UNSUPPORTED}{message}");
        let own = (group.get("format").filter(|format| !format.is_null()))
            .map(|format| Self::deserialize(format).map_err(read))
            .transpose()?;
        let packed = match (self, own) {
            (Self::Float | Self::Mixed, Some(Self::Float)) | (Self::Float, None) => false,
            (Self::Packed | Self::Mixed, Some(Self::Packed)) | (Self::Packed, None) => true,
            _ => {
                return Err(unsupported(
                    "a module group's format: expected the checkpoint's, or its own in a \
                     mixed-precision checkpoint",
                ));
            }
        };
        let strategy = &group["weights"]["strategy"];
        let (quantizes, scheme, refusal) = match (packed, strategy.as_str()) {
            (true, _) => {
                let group = PackedGroup::deserialize(group).map_err(read)?;
                (group.quantizes(), Scheme::Nvfp4, NVFP4)
            }
            (false, Some("channel")) => {
                let group = ChannelGroup::deserialize(group).map_err(read)?;
                (group.quantizes(), Scheme::Channel, FP8)
            }
            (false, Some("block")) => {
                let group = BlockGroup::deserialize(group).map_err(read)?;
                (group.linear(), Scheme::Block, FP8)
            }
            (false, _) => return Err(refuse(strategy, r#""block" or "channel""#)),
        };
        quantizes
            .then_some(scheme)
            .ok_or_else(|| unsupported(refusal))
    }
}

/// The FP8 scales a declaration's FP8 groups may hold, which a refusal of others names.
const FP8: &str = "expected static per-tensor E4M3 scales, or 128 x 128 block weight scales with \
                   activations scaled at run time per 128 inputs";

/// The NVFP4 groups a declaration may hold, which a refusal of others names.
const NVFP4: &str = "expected NVFP4 linear weights and inputs";

impl TryFrom<Value> for Quantization {
    type Error = String;

    fn try_from(declared: Value) -> Result<Self, Self::Error> {
        let unsupported = |message: &str| format!("{UNSUPPORTED}{message}");
        let fp8 = || unsupported(FP8);
        // A declared cache, or none when its values are asymmetric.
        let symmetric = |cache: Option<Tensor>| {
            cache.map_or(Some(false), |cache| cache.symmetric().then_some(true))
        };
        // ModelOpt's NVFP4 declarations name prefixes; the others `re:.*part.*` patterns.
        let prefixes = |entries: Vec<String>| {
            let prefix = |entry: String| match entry.strip_suffix('*') {
                Some(prefix) => Exclusion::Prefix(prefix.to_owned()),
                None => Exclusion::Module(entry),
            };
            (entries.into_iter())
                .map(|entry| {
                    prefix(entry).named().ok_or_else(|| {
                        unsupported("exclusions must be module names or prefixes ending in `*`")
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let patterns = |entries: Vec<String>| {
            let pattern = |entry: String| {
                let part = (entry.strip_prefix("re:.*")).and_then(|part| part.strip_suffix(".*"));
                match part {
                    Some(part) => Exclusion::Part(part.to_owned()),
                    None => Exclusion::Module(entry),
                }
            };
            (entries.into_iter())
                .map(|entry| {
                    pattern(entry).named().ok_or_else(|| {
                        unsupported("exclusions must be module names or `re:.*part.*` patterns")
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let (mut schemes, excluded, kv_cache) = match located::<_, Hf>(declared)? {
            Hf::Fp8 {
                activation_scheme,
                weight_block_size,
                ignored_layers,
                modules_to_not_convert,
                kv_cache_scheme,
                dequantize: Disabled,
                scale_fmt: None | Some(Literal),
            } => {
                let scheme = match (activation_scheme, weight_block_size) {
                    (Activations::Static, None) => Scheme::Tensor,
                    (Activations::Dynamic, Some([Constant, Constant])) => Scheme::Block,
                    _ => return Err(fp8()),
                };
                let ignored = ignored_layers.into_iter().chain(modules_to_not_convert);
                (
                    vec![scheme],
                    patterns(ignored.flatten().collect())?,
                    symmetric(kv_cache_scheme),
                )
            }
            Hf::Modelopt(Modelopt::Fp8 {
                config_groups,
                ignore,
                kv_cache_scheme,
            }) => {
                if config_groups.is_empty() || !config_groups.values().all(Group::<Tensor>::linear)
                {
                    return Err(fp8());
                }
                (
                    vec![Scheme::Tensor],
                    patterns(ignore.unwrap_or_default())?,
                    symmetric(kv_cache_scheme),
                )
            }
            Hf::Modelopt(Modelopt::Nvfp4 {
                config_groups,
                ignore,
                kv_cache_scheme,
            }) => {
                if config_groups.is_empty() || !config_groups.values().all(Group::<Values>::linear)
                {
                    return Err(unsupported(NVFP4));
                }
                let cache = kv_cache_scheme.map(
                    |Cache {
                         num_bits: Constant,
                         kind: Literal,
                         dynamic: Disabled,
                     }| (),
                );
                (
                    vec![Scheme::Nvfp4],
                    prefixes(ignore.unwrap_or_default())?,
                    Some(cache.is_some()),
                )
            }
            Hf::Modelopt(Modelopt::Other) => {
                return Err(unsupported("quant_algo: expected \"FP8\" or \"NVFP4\""));
            }
            Hf::CompressedTensors {
                format,
                quantization_status: Literal,
                config_groups,
                ignore,
                kv_cache_scheme,
                sparsity_config,
                transform_config,
            } => {
                let empty =
                    |part: Option<BTreeMap<String, Value>>| part.is_none_or(|part| part.is_empty());
                if config_groups.is_empty() || !empty(transform_config) {
                    return Err(fp8());
                }
                if !empty(sparsity_config) {
                    return Err(refuse("a value", "none"));
                }
                let schemes = (config_groups.values())
                    .map(|group| format.scheme(group))
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    schemes,
                    patterns(ignore.unwrap_or_default())?,
                    symmetric(kv_cache_scheme),
                )
            }
        };
        schemes.sort_unstable();
        schemes.dedup();
        Ok(Self {
            schemes,
            excluded: Exclusions(excluded),
            kv_cache: kv_cache.ok_or_else(fp8)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse;
    use super::*;
    use serde_json::{Value, json};
    use std::io;

    fn quantization(document: &Value) -> io::Result<Quantization> {
        parse(document)
    }

    /// The quantization objects of published FP8 checkpoints, by dialect.
    fn published() -> [(Value, Scheme); 5] {
        let blocks = |dynamic: bool| {
            json!({
                "weights": {
                    "num_bits": 8, "type": "float", "symmetric": true, "dynamic": false,
                    "strategy": "block", "block_structure": [128, 128], "group_size": null,
                    "actorder": null, "observer": "memoryless_minmax", "scale_dtype": null,
                    "zp_dtype": null
                },
                "input_activations": {
                    "num_bits": 8, "type": "float", "symmetric": true, "dynamic": dynamic,
                    "strategy": "group", "group_size": 128, "block_structure": null,
                    "actorder": null, "observer": null, "scale_dtype": null, "zp_dtype": null
                },
                "output_activations": null,
                "targets": ["Linear"]
            })
        };
        [
            (
                json!({"quant_method": "fp8", "activation_scheme": "static",
                       "ignored_layers": ["lm_head"]}),
                Scheme::Tensor,
            ),
            (
                json!({"quant_method": "modelopt", "quant_algo": "FP8", "ignore": ["lm_head"],
                       "config_groups": {"group_0": {
                           "weights": {"num_bits": 8, "type": "float", "dynamic": false},
                           "input_activations": {"num_bits": 8, "type": "float", "dynamic": false},
                           "targets": ["Linear"]}},
                       "kv_cache_scheme": {"num_bits": 8, "type": "float", "dynamic": false}}),
                Scheme::Tensor,
            ),
            // mistralai/Ministral-3-14B-Instruct-2512
            (
                json!({"quant_method": "fp8", "activation_scheme": "static", "dequantize": false,
                       "weight_block_size": null,
                       "modules_to_not_convert": ["model.vision_tower", "lm_head",
                                                  "model.vision_tower"]}),
                Scheme::Tensor,
            ),
            // furiosa-ai/EXAONE-4.0-32B-FP8
            (
                json!({"quant_method": "fp8", "activation_scheme": "dynamic",
                       "weight_block_size": [128, 128], "modules_to_not_convert": null}),
                Scheme::Block,
            ),
            // RedHatAI/Muse-Glimmer-30B-FP8-block
            (
                json!({"quant_method": "compressed-tensors", "format": "float-quantized",
                       "quantization_status": "compressed", "kv_cache_scheme": null,
                       "global_compression_ratio": null, "transform_config": {},
                       "ignore": ["re:.*vision.*", "lm_head", "re:.*embed_tokens.*"],
                       "config_groups": {"FP8_BLOCK": blocks(true)}}),
                Scheme::Block,
            ),
        ]
    }

    /// Each published dialect parses to its scale scheme, exclusions and cache quantization.
    #[test]
    fn parses_each_published_scheme() {
        let parsed: Vec<_> = published()
            .into_iter()
            .map(|(document, scheme)| (quantization(&document).unwrap(), scheme))
            .collect();
        for (config, scheme) in &parsed {
            assert_eq!(config.schemes, [*scheme]);
        }
        assert!(parsed[1].0.kv_cache && !parsed[0].0.kv_cache);
        let excludes = |at: usize, module| parsed[at].0.excluded.excludes(module);
        assert!(excludes(2, "model.vision_tower.layers.0.proj"));
        assert!(excludes(4, "model.vision_tower.layers.0.attn.q_proj"));
        assert!(excludes(4, "model.language_model.embed_tokens"));
        assert!(excludes(4, "lm_head"));
        assert!(!excludes(
            4,
            "model.language_model.layers.0.self_attn.q_proj"
        ));
    }

    /// A scheme the projections do not implement is unsupported, never mistaken for a malformed
    /// declaration.
    #[test]
    fn refuses_other_schemes() {
        let [static_, modelopt, _, dynamic, compressed] = published().map(|(document, _)| document);
        let change = |document: &Value, path: &[&str], value: Value| {
            let mut document = document.clone();
            let (last, path) = path.split_last().unwrap();
            path.iter()
                .fold(&mut document, |document, key| &mut document[*key])
                .as_object_mut()
                .unwrap()
                .insert((*last).to_owned(), value);
            document
        };
        let group = ["config_groups", "FP8_BLOCK"];
        let modelopt_group = ["config_groups", "group_0"];
        // Every FP8 dialect's expectation, which a scheme refusal names when no one field refuses.
        let scales = concat!(
            "expected static per-tensor E4M3 scales, or 128 x 128 block weight scales with ",
            "activations scaled at run time per 128 inputs",
        );
        let patterns = "exclusions must be module names or `re:.*part.*` patterns";
        let asymmetric =
            json!({"num_bits": 8, "type": "float", "dynamic": false, "symmetric": false});
        let cases = [
            (
                &static_,
                &["kv_cache_scheme"][..],
                asymmetric.clone(),
                scales,
            ),
            (
                &static_,
                &["ignored_layers"][..],
                json!(["re:.*.*"]),
                patterns,
            ),
            (
                &static_,
                &["ignored_layers"][..],
                json!(["re:.*vision.tower.*"]),
                patterns,
            ),
            (&modelopt, &["config_groups"][..], json!({}), scales),
            (
                &modelopt,
                &[modelopt_group[0], modelopt_group[1], "weights", "symmetric"][..],
                json!(false),
                scales,
            ),
            (
                &modelopt,
                &[
                    modelopt_group[0],
                    modelopt_group[1],
                    "input_activations",
                    "symmetric",
                ][..],
                json!(false),
                scales,
            ),
            (
                &modelopt,
                &[modelopt_group[0], modelopt_group[1], "targets"][..],
                json!(["Embedding"]),
                scales,
            ),
            (
                &modelopt,
                &["kv_cache_scheme"][..],
                asymmetric.clone(),
                scales,
            ),
            (&compressed, &["config_groups"][..], json!({}), scales),
            (
                &compressed,
                &[group[0], group[1], "input_activations", "symmetric"][..],
                json!(false),
                scales,
            ),
            (
                &static_,
                &["activation_scheme"][..],
                json!("dynamic"),
                scales,
            ),
            (
                &static_,
                &["weight_block_size"][..],
                json!([128, 128]),
                scales,
            ),
            (
                &static_,
                &["dequantize"][..],
                json!(true),
                "found true, compiled false",
            ),
            (
                &static_,
                &["ignored_layers"][..],
                json!(["re:.*"]),
                patterns,
            ),
            (
                &dynamic,
                &["weight_block_size"][..],
                json!([64, 64]),
                "found 64, compiled 128",
            ),
            (&dynamic, &["weight_block_size"][..], json!(null), scales),
            (
                &dynamic,
                &["scale_fmt"][..],
                json!("ue8m0"),
                "found \"ue8m0\", compiled \"float\"",
            ),
            (
                &modelopt,
                &["config_groups", "group_0", "weights", "strategy"][..],
                json!("channel"),
                "found \"channel\", compiled \"tensor\"",
            ),
            (
                &compressed,
                &[group[0], group[1], "weights", "strategy"][..],
                json!("tensor"),
                r#"found "tensor", compiled "block" or "channel""#,
            ),
            (
                &compressed,
                &[group[0], group[1], "weights", "dynamic"][..],
                json!(true),
                "found true, compiled false",
            ),
            (
                &compressed,
                &[group[0], group[1], "weights", "symmetric"][..],
                json!(false),
                scales,
            ),
            (
                &compressed,
                &[group[0], group[1], "weights", "block_structure"][..],
                json!([128, 64]),
                "found 64, compiled 128",
            ),
            (
                &compressed,
                &[group[0], group[1], "weights", "scale_dtype"][..],
                json!("float8"),
                "found a value, compiled none",
            ),
            (
                &compressed,
                &[group[0], group[1], "input_activations", "dynamic"][..],
                json!(false),
                scales,
            ),
            (
                &compressed,
                &[group[0], group[1], "input_activations", "group_size"][..],
                json!(64),
                "found 64, compiled 128",
            ),
            (
                &compressed,
                &[group[0], group[1], "input_activations", "strategy"][..],
                json!("token"),
                "found \"token\", compiled \"group\"",
            ),
            (
                &compressed,
                &[group[0], group[1], "targets"][..],
                json!(["Embedding"]),
                scales,
            ),
            (
                &compressed,
                &["transform_config"][..],
                json!({"hadamard": {}}),
                scales,
            ),
            (
                &compressed,
                &["kv_cache_scheme"][..],
                asymmetric.clone(),
                scales,
            ),
            (
                &compressed,
                &["sparsity_config"][..],
                json!({"format": "dense"}),
                "found a value, compiled none",
            ),
            (
                &compressed,
                &["ignore"][..],
                json!(["re:model\\..*"]),
                patterns,
            ),
            (
                &compressed,
                &["format"][..],
                json!("pack-quantized"),
                concat!(
                    r#"found "pack-quantized", compiled "float-quantized", "#,
                    r#""nvfp4-pack-quantized" or "mixed-precision""#,
                ),
            ),
        ];
        let found: Vec<_> = cases
            .iter()
            .map(|(document, path, value, _)| {
                let error = quantization(&change(document, path, value.clone())).unwrap_err();
                (*path, value.to_string(), error.kind(), error.to_string())
            })
            .collect();
        let expected: Vec<_> = (cases.iter())
            .map(|(_, path, value, reason)| {
                let reason = reason.to_string();
                (*path, value.to_string(), io::ErrorKind::Unsupported, reason)
            })
            .collect();
        assert_eq!(found, expected);
    }

    /// ModelOpt reads its algorithm as the string the document writes: another algorithm is
    /// unsupported, and one that is no string or is missing invalid.
    #[test]
    fn reads_the_modelopt_algorithm_as_written() {
        let [_, modelopt, ..] = published().map(|(document, _)| document);
        let refusal = |algorithm: Option<Value>| {
            let mut document = modelopt.clone();
            match algorithm {
                Some(algorithm) => document["quant_algo"] = algorithm,
                None => drop(document.as_object_mut().unwrap().remove("quant_algo")),
            }
            let error = quantization(&document).err()?;
            Some((error.kind(), error.to_string()))
        };
        assert_eq!(refusal(Some(json!("FP8"))), None);
        let number = "quant_algo: invalid type: number, expected variant identifier";
        for (algorithm, kind, reason) in [
            (
                Some(json!("W4A8_AWQ")),
                io::ErrorKind::Unsupported,
                r#"quant_algo: expected "FP8" or "NVFP4""#,
            ),
            (Some(json!(0)), io::ErrorKind::InvalidData, number),
            (Some(json!(1)), io::ErrorKind::InvalidData, number),
            (
                None,
                io::ErrorKind::InvalidData,
                "missing field `quant_algo`",
            ),
        ] {
            let refused = refusal(algorithm.clone());
            assert_eq!(refused, Some((kind, reason.into())), "{algorithm:?}");
        }
    }

    /// A published ModelOpt NVFP4 declaration.
    fn nvfp4() -> Value {
        let values = json!({"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16});
        let mut weights = values.clone();
        weights["symmetric"] = json!(true);
        weights["strategy"] = json!("tensor_group");
        json!({
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "producer": {"name": "modelopt", "version": "0.37.0"},
            "config_groups": {
                "group_0": {"input_activations": values, "weights": weights, "targets": ["Linear"]}
            },
            "ignore": ["lm_head", "model.layers.0.self_attn*"],
            "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"}
        })
    }

    /// The NVFP4 declaration's linear groups, prefix exclusions and E4M3 cache parse; another
    /// quantization, other values, no group or a malformed exclusion are refused.
    #[test]
    fn parses_nvfp4_declarations() {
        let config = quantization(&nvfp4()).unwrap();
        assert_eq!(config.schemes, [Scheme::Nvfp4]);
        assert_eq!(
            config.excluded.0,
            [
                Exclusion::Module("lm_head".into()),
                Exclusion::Prefix("model.layers.0.self_attn".into()),
            ]
        );
        assert!(config.kv_cache);
        let exclusions = "exclusions must be module names or prefixes ending in `*`";
        let linear = "expected NVFP4 linear weights and inputs";
        for (path, value, reason) in [
            ("/quant_algo", json!("FP8"), "found a value, compiled none"),
            (
                "/config_groups/group_0/weights/num_bits",
                json!(8),
                "found 8, compiled 4",
            ),
            (
                "/config_groups/group_0/input_activations/group_size",
                json!(32),
                "found 32, compiled 16",
            ),
            (
                "/config_groups/group_0/input_activations/dynamic",
                json!(true),
                "found true, compiled false",
            ),
            ("/config_groups/group_0/targets", json!(["Conv2d"]), linear),
            ("/config_groups", json!({}), linear),
            (
                "/config_groups/group_0/weights/symmetric",
                json!(false),
                "found false, compiled true",
            ),
            (
                "/config_groups/group_0/weights/strategy",
                json!("channel"),
                r#"found "channel", compiled "tensor_group""#,
            ),
            ("/ignore", json!(["model.layers.*.self_attn"]), exclusions),
            ("/ignore", json!(["*"]), exclusions),
            ("/kv_cache_scheme/num_bits", json!(4), "found 4, compiled 8"),
        ] {
            let mut document = nvfp4();
            *document.pointer_mut(path).unwrap() = value;
            let error = quantization(&document).unwrap_err();
            let refusal = (error.kind(), error.to_string());
            assert_eq!(
                refusal,
                (io::ErrorKind::Unsupported, reason.into()),
                "{path}"
            );
        }
        let mut document = nvfp4();
        document.as_object_mut().unwrap().remove("kv_cache_scheme");
        assert!(!quantization(&document).unwrap().kv_cache);
        document["quant_method"] = json!("compressed-tensors");
        let error = quantization(&document).unwrap_err();
        assert_eq!(
            (error.kind(), error.to_string()),
            (io::ErrorKind::InvalidData, "missing field `format`".into())
        );
    }

    /// A mixed-precision declaration parses to the scheme of each module group, its exclusions and
    /// its E4M3 cache; a group's other values or format, or a sparse checkpoint, are refused.
    #[test]
    fn parses_mixed_precision_declarations() {
        use crate::models::decoder::attention::gemma4;
        let config = quantization(&gemma4::tests::quantization()).unwrap();
        assert_eq!(config.schemes, [Scheme::Channel, Scheme::Nvfp4]);
        assert!(config.kv_cache);
        for module in ["lm_head", "model.embed_audio.embedding_projection"] {
            assert!(config.excluded.excludes(module), "{module}");
        }
        assert!(!(config.excluded).excludes("model.language_model.layers.0.self_attn.q_proj"));
        let fp8 = concat!(
            "expected static per-tensor E4M3 scales, or 128 x 128 block weight scales with ",
            "activations scaled at run time per 128 inputs",
        );
        let format = "a module group's format: expected the checkpoint's, or its own in a \
                      mixed-precision checkpoint";
        let (channel, packed) = ("/config_groups/group_0", "/config_groups/group_1");
        for (path, value, reason) in [
            (
                format!("{channel}/weights/strategy"),
                json!("group"),
                r#"found "group", compiled "block" or "channel""#,
            ),
            (
                format!("{channel}/weights/dynamic"),
                json!(true),
                "found true, compiled false",
            ),
            (
                format!("{channel}/input_activations/strategy"),
                json!("group"),
                r#"found "group", compiled "token""#,
            ),
            (format!("{channel}/targets"), json!([]), fp8),
            (
                format!("{packed}/weights/num_bits"),
                json!(8),
                "found 8, compiled 4",
            ),
            (
                format!("{packed}/weights/actorder"),
                json!("group"),
                r#"found "group", compiled "static""#,
            ),
            (
                format!("{packed}/input_activations/dynamic"),
                json!("dynamic"),
                r#"found "dynamic", compiled "local""#,
            ),
            (format!("{packed}/format"), json!(null), format),
            (format!("{packed}/format"), json!("mixed-precision"), format),
            (
                "/sparsity_config".to_owned(),
                json!({"format": "dense"}),
                "found a value, compiled none",
            ),
            ("/kv_cache_scheme/symmetric".to_owned(), json!(false), fp8),
        ] {
            let mut document = gemma4::tests::quantization();
            *document.pointer_mut(&path).unwrap() = value;
            let error = quantization(&document).unwrap_err();
            let refusal = (error.kind(), error.to_string());
            assert_eq!(
                refusal,
                (io::ErrorKind::Unsupported, reason.into()),
                "{path}"
            );
        }
        let mut nvfp4 = gemma4::tests::quantization();
        nvfp4["format"] = json!("nvfp4-pack-quantized");
        let error = quantization(&nvfp4).unwrap_err();
        assert_eq!(error.to_string(), format);
        (nvfp4["config_groups"].as_object_mut().unwrap()).remove("group_0");
        assert_eq!(quantization(&nvfp4).unwrap().schemes, [Scheme::Nvfp4]);
    }
}
