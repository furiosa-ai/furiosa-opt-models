//! Parts that family configurations parse from checkpoint documents, and the refinement types
//! that hold a checkpoint value to what the compiled composition accepts.
//!
//! - A family reads a document in two steps: serde deserializes the family's `Hf` schema, then
//!   `TryFrom` narrows the schema to the family's configuration.
//! - A refinement type (`Constant`, `Literal`, `Disabled`, `Absent`) refuses a well-formed value
//!   that the compiled composition does not hold. [`parse`] reports that refusal as
//!   `Unsupported` and any other malformed document as `InvalidData`.
//! - Every flattened part reads the document through a JSON map, so two parts may read one field.

pub use quantization::{Quantization, Scheme};

use furiosa_opt_std::prelude::AxisName;
use serde::{Deserialize, Deserializer, de};
use serde_json::{Map, Value};
use std::marker::PhantomData;
use std::{
    borrow::Cow,
    collections::BTreeMap,
    fmt::{Debug, Display},
    io,
    num::NonZeroUsize,
};

pub mod ddim;
pub mod decoder;
mod quantization;
pub mod scaling;

/// Opens every refinement's refusal, which [`parse`] reports as unsupported.
const UNSUPPORTED: &str = "unsupported: ";

/// An integer the compiled composition fixes to `N`.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(try_from = "usize")]
pub struct Constant<const N: usize>;

impl<const N: usize> TryFrom<usize> for Constant<N> {
    type Error = String;

    fn try_from(found: usize) -> Result<Self, Self::Error> {
        (found == N).then_some(Self).ok_or_else(|| refuse(found, N))
    }
}

/// A size the compiled composition fixes to axis `A`'s, for a schema generic over its axes.
#[derive(Deserialize)]
#[serde(try_from = "usize", bound = "A: AxisName")]
pub struct Size<A>(PhantomData<fn() -> A>);

impl<A: AxisName> TryFrom<usize> for Size<A> {
    type Error = String;

    fn try_from(found: usize) -> Result<Self, Self::Error> {
        (found == A::SIZE)
            .then_some(Self(PhantomData))
            .ok_or_else(|| refuse(found, A::SIZE))
    }
}

/// A string the compiled composition fixes to `S`.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(try_from = "String")]
pub struct Literal<const S: &'static str>;

impl<const S: &'static str> TryFrom<String> for Literal<S> {
    type Error = String;

    fn try_from(found: String) -> Result<Self, Self::Error> {
        (found == S)
            .then_some(Self)
            .ok_or_else(|| refuse(format_args!("{found:?}"), format_args!("{S:?}")))
    }
}

/// A flag the compiled composition holds off; with `#[serde(default)]`, absent means off.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(try_from = "bool")]
pub struct Disabled;

impl TryFrom<bool> for Disabled {
    type Error = String;

    fn try_from(found: bool) -> Result<Self, Self::Error> {
        (!found).then_some(Self).ok_or_else(|| refuse(true, false))
    }
}

/// The modules a quantization declaration leaves unquantized, each named one supported way.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Exclusions(pub Vec<Exclusion>);

/// One way a quantization declaration names modules it leaves unquantized.
#[derive(Clone, Debug, PartialEq)]
pub enum Exclusion {
    /// A module by name, any module of that name within a parent, and every module below them.
    Module(String),
    /// Every module whose name starts with a prefix.
    Prefix(String),
    /// Every module whose name holds a part.
    Part(String),
}

impl Exclusion {
    /// The exclusion, when it names modules: dot-separated identifiers, a part one identifier.
    fn named(self) -> Option<Self> {
        let identifiers = |name: &str| {
            !name.is_empty()
                && name
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || "._".contains(character))
        };
        let named = match &self {
            Self::Module(name) | Self::Prefix(name) => identifiers(name),
            Self::Part(part) => identifiers(part) && !part.contains('.'),
        };
        named.then_some(self)
    }
}

impl Exclusions {
    /// Whether the declaration leaves `module` unquantized.
    pub fn excludes(&self, module: &str) -> bool {
        self.0.iter().any(|exclusion| match exclusion {
            Exclusion::Module(name) => {
                module == name
                    || module.ends_with(&format!(".{name}"))
                    || module.starts_with(&format!("{name}."))
            }
            Exclusion::Prefix(prefix) => module.starts_with(prefix.as_str()),
            Exclusion::Part(part) => module.contains(part.as_str()),
        })
    }
}

/// A flag the compiled composition holds on.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(try_from = "bool")]
pub struct Enabled;

impl TryFrom<bool> for Enabled {
    type Error = String;

    fn try_from(found: bool) -> Result<Self, Self::Error> {
        found.then_some(Self).ok_or_else(|| refuse(false, true))
    }
}

/// A part the compiled composition has no place for: null, or absent with `#[serde(default)]`.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(try_from = "Option<de::IgnoredAny>")]
pub struct Absent;

impl TryFrom<Option<de::IgnoredAny>> for Absent {
    type Error = String;

    fn try_from(found: Option<de::IgnoredAny>) -> Result<Self, Self::Error> {
        found.map_or(Ok(Self), |_| Err(refuse("a value", "none")))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(try_from = "f32")]
pub struct Positive(f32);

impl Positive {
    /// A value the code fixes, such as a compiled base; a constant that is not positive and
    /// finite fails the build.
    pub const fn new(value: f32) -> Self {
        assert!(value.is_finite() && value > 0.0, "a positive finite value");
        Self(value)
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for Positive {
    type Error = String;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("found {value}, expected a positive finite value"));
        }
        Ok(Self(value))
    }
}

/// Transformers' `per_layer_config`, by layer index: the fields each layer sets for itself.
/// Transformers 5.15 and later write a field that differs between layers this way.
#[derive(Default, Deserialize)]
#[serde(try_from = "BTreeMap<String, Map<String, Value>>")]
pub struct PerLayer(BTreeMap<usize, Map<String, Value>>);

impl TryFrom<BTreeMap<String, Map<String, Value>>> for PerLayer {
    type Error = String;

    /// Reads each key as the layer index, which Transformers writes zero-padded.
    fn try_from(layers: BTreeMap<String, Map<String, Value>>) -> Result<Self, Self::Error> {
        layers
            .into_iter()
            .map(|(layer, fields)| match layer.parse() {
                Ok(index) => Ok((index, fields)),
                Err(_) => Err(format!("per_layer_config: {layer:?} is not a layer index")),
            })
            .collect::<Result<_, _>>()
            .map(Self)
    }
}

impl PerLayer {
    /// Whether no layer sets a field for itself.
    pub fn is_empty(&self) -> bool {
        self.0.values().all(Map::is_empty)
    }

    /// Where `document`, or a configuration nested in it, sets fields per layer: the path of the
    /// first non-empty `per_layer_config`, empty for the document's own.
    pub fn find(document: &Value) -> Option<String> {
        let fields = document.as_object()?;
        if fields
            .get("per_layer_config")
            .and_then(Value::as_object)
            .is_some_and(|layers| !layers.is_empty())
        {
            return Some(String::new());
        }
        fields.iter().find_map(|(name, nested)| {
            Self::find(nested).map(|path| match path.is_empty() {
                true => name.clone(),
                false => format!("{name}.{path}"),
            })
        })
    }

    /// The fields, as `T`, that every layer of type `kind` of `layer_types` `types` sets, or none;
    /// refused unless the layers of one type set the same, as one compiled group holds them.
    pub fn of<T: de::DeserializeOwned>(
        &self,
        types: &[String],
        kind: &str,
    ) -> io::Result<Option<T>> {
        if let Some(layer) = self.0.keys().find(|&&layer| layer >= types.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "per_layer_config: found layer {layer}, the document has {} layers",
                    types.len()
                ),
            ));
        }
        let mut layers = types
            .iter()
            .enumerate()
            .filter(|(_, layer)| *layer == kind)
            .map(|(index, _)| self.0.get(&index).cloned().unwrap_or_default());
        let first = layers.next().unwrap_or_default();
        if !layers.all(|fields| fields == first) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "per_layer_config: the {kind} layers set different fields, compiled one group"
                ),
            ));
        }
        if first.is_empty() {
            return Ok(None);
        }
        serde_json::from_value(Value::Object(first))
            .map(Some)
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("per_layer_config: the {kind} layers set {error}"),
                )
            })
    }
}

/// Deserializes the schema `T` from a checkpoint document.
pub fn parse<T: de::DeserializeOwned>(document: &Value) -> io::Result<T> {
    located(document).map_err(refusal)
}

/// The error of a part's refusal `message`: unsupported where it carries the unsupported mark,
/// which it loses, and else invalid.
pub(crate) fn refusal(message: String) -> io::Error {
    match message.strip_prefix(UNSUPPORTED) {
        Some(refusal) => io::Error::new(io::ErrorKind::Unsupported, refusal.to_owned()),
        None => io::Error::new(io::ErrorKind::InvalidData, message),
    }
}

/// Deserializes `T`, naming the field an error comes from as `field: reason`; a refusal keeps its
/// unsupported mark first, so a nested part's refusal stays one.
fn located<'de, D: Deserializer<'de>, T: Deserialize<'de>>(deserializer: D) -> Result<T, String> {
    serde_path_to_error::deserialize(deserializer).map_err(|error| {
        let path = error.path().to_string();
        let reason = error.into_inner().to_string();
        let (mark, reason) = match reason.strip_prefix(UNSUPPORTED) {
            Some(reason) => (UNSUPPORTED, reason.to_owned()),
            None => ("", reason),
        };
        match path.as_str() {
            "." | "?" => format!("{mark}{reason}"),
            path => format!("{mark}{path}: {reason}"),
        }
    })
}

/// The refusal of a value the compiled composition does not hold, which [`parse`] reports as
/// unsupported.
fn refuse(found: impl Display, compiled: impl Display) -> String {
    format!("{UNSUPPORTED}found {found}, compiled {compiled}")
}

/// Deserializes a `#[serde(flatten)]` part through a JSON map: serde buffers a flattened part's
/// numbers in a form that only `serde_json::Value` decodes once `arbitrary_precision` is on.
pub fn flattened<'de, D: Deserializer<'de>, T: de::DeserializeOwned>(
    deserializer: D,
) -> Result<T, D::Error> {
    located(Value::Object(Map::deserialize(deserializer)?)).map_err(de::Error::custom)
}

/// Deserializes `max_position_embeddings` as a flattened part, which leaves it to the document's
/// spectrum: a rescaled spectrum is trained on the declared context where it names none.
pub fn declared<'de, D: Deserializer<'de>>(deserializer: D) -> Result<NonZeroUsize, D::Error> {
    #[derive(Deserialize)]
    struct Declared {
        max_position_embeddings: NonZeroUsize,
    }
    flattened(deserializer).map(|declared: Declared| declared.max_position_embeddings)
}

/// Deserializes a flattened language part: the `text_config` a multimodal checkpoint nests, else
/// the document itself.
pub fn text<'de, D: Deserializer<'de>, T: de::DeserializeOwned>(
    deserializer: D,
) -> Result<T, D::Error> {
    nested(deserializer, "text_config")
}

/// Deserializes a flattened image part: the `vision_config` a multimodal checkpoint nests, else
/// the document itself.
pub fn vision<'de, D: Deserializer<'de>, T: de::DeserializeOwned>(
    deserializer: D,
) -> Result<T, D::Error> {
    nested(deserializer, "vision_config")
}

fn nested<'de, D: Deserializer<'de>, T: de::DeserializeOwned>(
    deserializer: D,
    key: &str,
) -> Result<T, D::Error> {
    let mut map = Map::deserialize(deserializer)?;
    let part = match map.remove(key) {
        None | Some(Value::Null) => Value::Object(map),
        Some(part) => part,
    };
    located(part).map_err(de::Error::custom)
}

/// The language model's fields: a nested `text_config`, taking the whole model's quantization and
/// tying where it omits them and refused if it contradicts that quantization, else the document.
pub fn language(document: &Value) -> io::Result<Cow<'_, Value>> {
    let mut text = match document.get("text_config") {
        None | Some(Value::Null) => return Ok(Cow::Borrowed(document)),
        Some(Value::Object(text)) => text.clone(),
        Some(other) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("text_config: expected an object, found {other}"),
            ));
        }
    };
    for field in ["quantization_config", "tie_word_embeddings"] {
        let Some(outer) = document.get(field).filter(|value| !value.is_null()) else {
            continue;
        };
        match text.get(field).filter(|value| !value.is_null()) {
            None => {
                text.insert(field.to_owned(), outer.clone());
            }
            // A serialized composition writes its class's default tying beside the language
            // model's own, which the checkpoint's head follows.
            Some(_) if field == "tie_word_embeddings" => {}
            Some(inner) if inner == outer => {}
            Some(inner) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "text_config.{field}: found {inner}, the configuration declares {outer}"
                    ),
                ));
            }
        }
    }
    Ok(Cow::Owned(Value::Object(text)))
}

/// Refuses a checkpoint value the compiled kernel does not hold, for a value a refinement type
/// cannot name: one derived from a generic parameter needs `generic_const_exprs` to be a type.
pub fn compiled<T: Debug + PartialEq>(field: &str, found: T, expected: T) -> io::Result<T> {
    if found != expected {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("{field}: found {found:?}, compiled {expected:?}"),
        ));
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed schema fields accept exactly the compiled integer; another integer is unsupported by
    /// its value, null or a negative number is invalid.
    #[test]
    fn parses_compiled_constants() {
        let refusal = |value| {
            let error = parse::<Constant<7>>(&serde_json::json!(value)).err()?;
            Some((error.kind(), error.to_string()))
        };
        assert_eq!(refusal(Some(7)), None);
        for (value, kind, reason) in [
            (Some(8), io::ErrorKind::Unsupported, "found 8, compiled 7"),
            (
                None,
                io::ErrorKind::InvalidData,
                "invalid type: null, expected usize",
            ),
            (Some(-1), io::ErrorKind::InvalidData, "invalid number"),
        ] {
            assert_eq!(refusal(value), Some((kind, reason.into())), "{value:?}");
        }
    }

    /// HF disabled flags accept false; true is unsupported, any other JSON value invalid.
    #[test]
    fn refuses_enabled_or_malformed_bias() {
        let refusal = |value| {
            let error = parse::<Disabled>(&value).err()?;
            Some((error.kind(), error.to_string()))
        };
        assert_eq!(refusal(serde_json::json!(false)), None);
        for (value, kind, reason) in [
            (
                serde_json::json!(true),
                io::ErrorKind::Unsupported,
                "found true, compiled false",
            ),
            (
                serde_json::json!(null),
                io::ErrorKind::InvalidData,
                "invalid type: null, expected a boolean",
            ),
            (
                serde_json::json!("false"),
                io::ErrorKind::InvalidData,
                "invalid type: string \"false\", expected a boolean",
            ),
        ] {
            let refused = refusal(value.clone());
            assert_eq!(refused, Some((kind, reason.into())), "{value}");
        }
    }

    /// A well-formed value outside the compiled composition is unsupported, through flattened
    /// parts too; a malformed one is invalid.
    #[test]
    fn refuses_compiled_values_as_unsupported() {
        #[derive(Deserialize)]
        struct Part {
            hidden_act: Literal<"silu">,
        }
        #[derive(Deserialize)]
        struct Schema {
            #[serde(flatten, deserialize_with = "flattened")]
            part: Part,
        }
        let refusal = |document| {
            let parsed = parse(&document).map(
                |Schema {
                     part:
                         Part {
                             hidden_act: Literal,
                         },
                 }| (),
            );
            parsed.err().map(|error| (error.kind(), error.to_string()))
        };
        assert_eq!(refusal(serde_json::json!({"hidden_act": "silu"})), None);
        assert_eq!(
            refusal(serde_json::json!({"hidden_act": "gelu"})),
            Some((
                io::ErrorKind::Unsupported,
                r#"hidden_act: found "gelu", compiled "silu""#.into()
            ))
        );
        assert_eq!(
            refusal(serde_json::json!({"hidden_act": 1})),
            Some((
                io::ErrorKind::InvalidData,
                "hidden_act: invalid type: number, expected a string".into()
            ))
        );
    }

    /// Per-layer fields are read by layer type from Transformers' zero-padded indices into the
    /// fields a group may set; a key that is no index, a layer past the document's, layers of one
    /// type that differ and a field the group does not name are refused.
    #[test]
    fn reads_per_layer_fields_by_layer_type() {
        use serde_json::json;
        #[derive(Debug, PartialEq, Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Head {
            head_dim: usize,
        }
        let types = ["sliding", "full", "sliding", "full"].map(str::to_owned);
        let per_layer = |document| parse::<PerLayer>(&document);
        let full = json!({"head_dim": 512});
        let parsed = per_layer(json!({"01": full, "03": full})).unwrap();
        assert_eq!(
            parsed.of(&types, "full").unwrap(),
            Some(Head { head_dim: 512 })
        );
        assert_eq!(parsed.of::<Head>(&types, "sliding").unwrap(), None);
        let error = per_layer(json!({"first": full})).err().unwrap();
        assert_eq!(
            (error.kind(), error.to_string()),
            (
                io::ErrorKind::InvalidData,
                r#"per_layer_config: "first" is not a layer index"#.into()
            )
        );
        for (document, kind, reason) in [
            (
                json!({"4": full}),
                io::ErrorKind::InvalidData,
                "per_layer_config: found layer 4, the document has 4 layers",
            ),
            (
                json!({"1": full}),
                io::ErrorKind::Unsupported,
                "per_layer_config: the full layers set different fields, compiled one group",
            ),
            (
                json!({"1": {"head_dim": 512, "intermediate_size": 1}, "3": {"head_dim": 512, "intermediate_size": 1}}),
                io::ErrorKind::Unsupported,
                "per_layer_config: the full layers set unknown field `intermediate_size`, \
                 expected `head_dim`",
            ),
        ] {
            let error = per_layer(document)
                .unwrap()
                .of::<Head>(&types, "full")
                .unwrap_err();
            assert_eq!((error.kind(), error.to_string()), (kind, reason.into()));
        }
    }

    /// A refusal names the field it comes from, through flattened parts too, and stays a refusal.
    #[test]
    fn names_the_refused_field() {
        use serde_json::json;
        #[derive(Debug, Deserialize)]
        struct Part {
            #[allow(dead_code)]
            num_hidden_layers: Constant<2>,
        }
        #[derive(Debug, Deserialize)]
        struct Whole {
            #[allow(dead_code)]
            #[serde(flatten, deserialize_with = "flattened")]
            part: Part,
        }
        let error = parse::<Whole>(&json!({"num_hidden_layers": 3})).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Unsupported, "{error}");
        assert_eq!(error.to_string(), "num_hidden_layers: found 3, compiled 2");
    }

    /// Rotary bases are positive and finite in their runtime f32 representation.
    #[test]
    fn refuses_invalid_theta() {
        for value in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            let reason = Positive::try_from(value).unwrap_err();
            assert_eq!(
                reason,
                format!("found {value}, expected a positive finite value")
            );
        }
        assert_eq!(Positive::try_from(7.0).unwrap().get(), 7.0);
    }

    /// A nested `text_config` is the language model's fields, with the whole model's quantization
    /// and tying where it omits them and its own tying over the whole model's; its own fields,
    /// rotary objects included, are untouched.
    #[test]
    fn reads_nested_language_fields() {
        use serde_json::json;
        let yarn = json!({
            "beta_fast": 32.0, "beta_slow": 1.0, "factor": 16.0, "llama_4_scaling_beta": 0.1,
            "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 16384,
            "rope_theta": 1000000000.0, "rope_type": "yarn", "type": "yarn",
        });
        let ministral = json!({
            "hidden_size": 5120, "hidden_act": "silu", "rope_parameters": yarn,
            "sliding_window": null, "vocab_size": 131072,
        });
        let vision = json!({"rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}});
        let fp8 = json!({
            "activation_scheme": "static", "quant_method": "fp8", "weight_block_size": null,
            "modules_to_not_convert": ["model.vision_tower", "model.multi_modal_projector", "lm_head"],
        });
        let nvfp4 = json!({
            "quant_method": "modelopt", "quant_algo": "NVFP4",
            "ignore": ["lm_head", "model.embed_vision*"],
            "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
        });
        let gemma = json!({
            "hidden_size": 5376, "hidden_activation": "gelu_pytorch_tanh",
            "tie_word_embeddings": true,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            },
        });
        let block = json!({
            "quant_method": "compressed-tensors", "format": "float-quantized",
            "ignore": ["re:.*vision.*", "lm_head", "re:.*embed_tokens.*"],
        });
        let muse = json!({
            "hidden_size": 6656, "hidden_activation": "silu", "tie_word_embeddings": false,
            "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0],
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        });
        let with = |mut text: Value, fields: Value| {
            text.as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            text
        };
        for (checkpoint, document, expected) in [
            (
                "Ministral-3-14B BF16",
                json!({"text_config": ministral, "vision_config": vision}),
                ministral.clone(),
            ),
            (
                "Ministral-3-14B FP8",
                json!({
                    "text_config": ministral, "vision_config": vision,
                    "quantization_config": fp8, "tie_word_embeddings": false,
                }),
                with(
                    ministral.clone(),
                    json!({"quantization_config": fp8, "tie_word_embeddings": false}),
                ),
            ),
            (
                "Gemma-4-31B NVFP4",
                json!({
                    "text_config": gemma, "vision_config": vision,
                    "quantization_config": nvfp4, "tie_word_embeddings": true,
                }),
                with(gemma.clone(), json!({"quantization_config": nvfp4})),
            ),
            (
                "Muse-Glimmer-30B FP8 block",
                json!({"text_config": muse, "vision_config": vision, "quantization_config": block}),
                with(muse.clone(), json!({"quantization_config": block})),
            ),
        ] {
            assert_eq!(*language(&document).unwrap(), expected, "{checkpoint}");
        }
        let root = json!({
            "hidden_size": 5120, "rope_theta": 1000000, "tie_word_embeddings": false,
            "rope_scaling": {
                "factor": 16.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192, "rope_type": "llama3",
            },
            "quantization_config": {"activation_scheme": "dynamic", "quant_method": "fp8"},
        });
        assert!(matches!(language(&root).unwrap(), Cow::Borrowed(fields) if *fields == root));
        for (outer, inner, kept) in [
            (true, Some(false), false),
            (false, Some(true), true),
            (true, None, true),
        ] {
            let text = inner.map_or(json!({}), |tied| json!({"tie_word_embeddings": tied}));
            let document = json!({"tie_word_embeddings": outer, "text_config": text});
            assert_eq!(
                language(&document).unwrap()["tie_word_embeddings"],
                json!(kept),
                "{document}"
            );
        }
    }

    /// A nested `text_config` that contradicts the whole model's quantization, or that is no
    /// object, is invalid, naming both declarations or what it found.
    #[test]
    fn refuses_contradicting_language_fields() {
        use serde_json::json;
        for (document, reason) in [
            (
                json!({
                    "quantization_config": {"quant_method": "fp8"},
                    "text_config": {"quantization_config": {"quant_method": "modelopt"}},
                }),
                r#"text_config.quantization_config: found {"quant_method":"modelopt"}, the configuration declares {"quant_method":"fp8"}"#,
            ),
            (
                json!({"text_config": "nested"}),
                r#"text_config: expected an object, found "nested""#,
            ),
        ] {
            let error = language(&document).unwrap_err();
            let refusal = (error.kind(), error.to_string());
            assert_eq!(
                refusal,
                (io::ErrorKind::InvalidData, reason.into()),
                "{document}"
            );
        }
    }
}
