//! The decoder schema that the transformer decoder families share: geometry, rotary part and the
//! flags their compositions hold fixed.

use super::scaling::{Kind, Multimodal, Sections, Spectrum};
use super::{Disabled, Literal, PerLayer, Positive, flattened, refuse};
use furiosa_opt_std::prelude::AxisName;
use serde::{Deserialize, de};
use std::{collections::BTreeMap, io, marker::PhantomData, num::NonZeroUsize};

#[derive(Clone, Debug, Deserialize)]
pub struct Geometry {
    pub hidden_size: NonZeroUsize,
    #[serde(default)]
    pub intermediate_size: usize,
    pub num_hidden_layers: NonZeroUsize,
    pub num_attention_heads: NonZeroUsize,
    pub num_key_value_heads: NonZeroUsize,
    pub vocab_size: NonZeroUsize,
    pub head_dim: Option<NonZeroUsize>,
    pub rms_norm_eps: Option<f32>,
}

impl Geometry {
    pub fn head_dim(&self) -> usize {
        self.head_dim.map_or_else(
            || self.hidden_size.get() / self.num_attention_heads.get(),
            NonZeroUsize::get,
        )
    }
}

/// A decoder's configuration as Transformers' decoder configurations share it, with the fields
/// only its architecture has in `X`: the geometry every layer has, and each layer type's own.
#[derive(Debug, Deserialize)]
#[serde(
    try_from = "Written<X>",
    bound(deserialize = "X: de::DeserializeOwned")
)]
pub struct Language<X> {
    pub geometry: Geometry,
    /// Each layer's type, in layer order; empty for a decoder of one layer type.
    layer_types: Vec<String>,
    groups: BTreeMap<String, Group>,
    pub extra: X,
}

/// The attention geometry of one type of layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group {
    pub num_key_value_heads: usize,
    pub head_dim: usize,
}

impl<X> Language<X> {
    /// Each layer's type, in layer order; empty for a decoder of one layer type.
    pub fn layer_types(&self) -> &[String] {
        &self.layer_types
    }

    /// The geometry of the layers of type `kind`: their own fields over the document's.
    pub fn group(&self, kind: &str) -> Group {
        self.groups.get(kind).copied().unwrap_or(Group {
            num_key_value_heads: self.geometry.num_key_value_heads.get(),
            head_dim: self.geometry.head_dim(),
        })
    }
}

/// A decoder configuration as a checkpoint writes it, in either spelling of a layer type's own
/// geometry.
#[derive(Deserialize)]
#[serde(bound(deserialize = "X: de::DeserializeOwned"))]
struct Written<X> {
    #[serde(flatten, deserialize_with = "flattened")]
    geometry: Geometry,
    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default)]
    per_layer_config: PerLayer,
    /// The full attention layers' own geometry as Gemma configurations wrote it before
    /// Transformers 5.15, which writes it per layer instead.
    num_global_key_value_heads: Option<usize>,
    global_head_dim: Option<usize>,
    #[serde(flatten, deserialize_with = "flattened")]
    extra: X,
}

/// The fields a type of layer may set for itself.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Own {
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
}

/// The layer type the `global_*` fields describe.
const GLOBAL: &str = "full_attention";

impl<X> TryFrom<Written<X>> for Language<X> {
    type Error = String;

    fn try_from(written: Written<X>) -> Result<Self, Self::Error> {
        if written.layer_types.is_empty() && !written.per_layer_config.is_empty() {
            return Err(refuse(
                "per_layer_config without layer_types",
                "one geometry for every layer",
            ));
        }
        let mut kinds = written.layer_types.clone();
        kinds.sort();
        kinds.dedup();
        let base = (
            written.geometry.num_key_value_heads.get(),
            written.geometry.head_dim(),
        );
        let mut groups = BTreeMap::new();
        for kind in kinds {
            let own = written
                .per_layer_config
                .of::<Own>(&written.layer_types, &kind)
                .map_err(|error| match error.kind() {
                    io::ErrorKind::Unsupported => format!("{}{error}", super::UNSUPPORTED),
                    _ => error.to_string(),
                })?;
            let global = (kind == GLOBAL)
                .then_some((written.num_global_key_value_heads, written.global_head_dim));
            let pick =
                |(name, field): (&str, &str), global: Option<usize>, own, base| match (global, own)
                {
                    (Some(global), Some(own)) if global != own => Err(format!(
                        "{name}: found {global}, but the {kind} layers' {field} is {own}"
                    )),
                    (global, own) => Ok(global.or(own).unwrap_or(base)),
                };
            let group = Group {
                num_key_value_heads: pick(
                    ("num_global_key_value_heads", "num_key_value_heads"),
                    global.and_then(|global| global.0),
                    own.as_ref().and_then(|own| own.num_key_value_heads),
                    base.0,
                )?,
                head_dim: pick(
                    ("global_head_dim", "head_dim"),
                    global.and_then(|global| global.1),
                    own.as_ref().and_then(|own| own.head_dim),
                    base.1,
                )?,
            };
            groups.insert(kind, group);
        }
        Ok(Self {
            geometry: written.geometry,
            layer_types: written.layer_types,
            groups,
            extra: written.extra,
        })
    }
}

/// A checkpoint's unscaled rotary base, for a family whose kernels hold no rescaling of its
/// frequencies: any other spectrum is refused as unsupported.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(try_from = "Spectrum")]
pub struct Rotary(pub Positive);

impl TryFrom<Spectrum> for Rotary {
    type Error = String;

    fn try_from(spectrum: Spectrum) -> Result<Self, Self::Error> {
        match spectrum.kind() == Kind::Default && spectrum.fraction() == 1.0 {
            true => Ok(Self(spectrum.theta())),
            false => Err(refuse(
                format_args!(
                    "{:?} rotating {} of each head",
                    spectrum.kind(),
                    spectrum.fraction()
                ),
                "the unscaled spectrum over the whole head",
            )),
        }
    }
}

/// A checkpoint's unscaled rotary base, for a family whose kernels rotate the leading `Rotated`
/// coordinates of each `Head`-wide head as one rotation at that width's frequencies, pair `i` with
/// pair `i + Rotated / 2`: any other spectrum is refused as unsupported.
#[derive(Clone, Copy, Debug)]
pub struct Prefix<Head, Rotated>(pub Positive, PhantomData<fn() -> (Head, Rotated)>);

impl<Head: AxisName, Rotated: AxisName> TryFrom<Spectrum> for Prefix<Head, Rotated> {
    type Error = String;

    fn try_from(spectrum: Spectrum) -> Result<Self, Self::Error> {
        match spectrum.kind() == Kind::Default && spectrum.rotated(Head::SIZE) == Rotated::SIZE {
            true => Ok(Self(spectrum.theta(), PhantomData)),
            false => Err(refuse(
                format_args!(
                    "{:?} rotating {} of each head",
                    spectrum.kind(),
                    spectrum.fraction()
                ),
                format_args!(
                    "the unscaled spectrum over the first {} of {} coordinates",
                    Rotated::SIZE,
                    Head::SIZE
                ),
            )),
        }
    }
}

/// A checkpoint's unscaled rotary base over the whole head, for a family whose kernels hold a
/// multimodal rotation's sections interleaved across each `Head`-wide head: sections in one block
/// each are refused as unsupported, and sections that do not cover a head's pairs as malformed.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(try_from = "Multimodal", bound = "Head: AxisName")]
pub struct Interleaved<Head>(pub Positive, PhantomData<fn() -> Head>);

impl<Head: AxisName> TryFrom<Multimodal> for Interleaved<Head> {
    type Error = String;

    fn try_from(Multimodal { spectrum, sections }: Multimodal) -> Result<Self, Self::Error> {
        let Rotary(theta) = Rotary::try_from(spectrum)?;
        match sections {
            Some(Sections {
                pairs,
                interleaved: false,
            }) => Err(refuse(
                format_args!("sections {pairs:?} in one block each"),
                "interleaved sections",
            )),
            Some(Sections { pairs, .. }) if pairs.iter().sum::<usize>() != Head::SIZE / 2 => {
                Err(format!(
                    "mrope_section: found {pairs:?}, expected sections of the {} pairs of a \
                     {}-wide head",
                    Head::SIZE / 2,
                    Head::SIZE
                ))
            }
            _ => Ok(Self(theta, PhantomData)),
        }
    }
}

/// The feed-forward activation, which the compiled composition fixes to `S`, under either name
/// Transformers configurations give it.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(from = "Names<S>")]
pub struct Activation<const S: &'static str>;

/// The two names of a configuration's activation, each fixed to `S` when given.
#[derive(Deserialize)]
struct Names<const S: &'static str> {
    #[serde(rename = "hidden_act")]
    _act: Option<Literal<S>>,
    #[serde(rename = "hidden_activation")]
    _activation: Option<Literal<S>>,
}

impl<const S: &'static str> From<Names<S>> for Activation<S> {
    fn from(_: Names<S>) -> Self {
        Self
    }
}

/// A decoder's schema, with `R` the rotary part its kernels hold: [`Rotary`] for an unscaled
/// base, or a [`Spectrum`].
#[derive(Clone, Debug, Deserialize)]
#[serde(bound(deserialize = "R: de::DeserializeOwned"))]
pub struct Config<R = Rotary> {
    #[serde(flatten, deserialize_with = "flattened")]
    pub geometry: Geometry,
    #[serde(flatten, deserialize_with = "flattened")]
    pub rotary: R,
    #[serde(flatten, deserialize_with = "super::declared")]
    pub max_position_embeddings: NonZeroUsize,
    #[serde(flatten, deserialize_with = "flattened")]
    pub activation: Activation<"silu">,
    #[serde(default)]
    pub attention_bias: Disabled,
    #[serde(default)]
    pub mlp_bias: Disabled,
    pub tie_word_embeddings: Option<bool>,
    pub sliding_window: Option<NonZeroUsize>,
    pub quantization_config: Option<super::Quantization>,
}

impl<R> Config<R> {
    /// The context this function serves: the checkpoint's trained length, bounded by the
    /// compiled cache. A checkpoint trained longer still runs, with a shorter context.
    pub fn capacity(&self, maximum: usize) -> usize {
        self.max_position_embeddings.get().min(maximum)
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse;
    use super::*;
    use serde_json::json;

    /// Each layer type's geometry is its own per-layer fields over the document's, with the full
    /// layers' `global_*` spelling read as theirs; per-layer fields without layer types are refused.
    #[test]
    fn reads_each_layer_type_geometry() {
        #[derive(Debug, Deserialize)]
        struct Nothing {}
        let document = |extra: serde_json::Value| {
            let mut document = json!({
                "hidden_size": 1024, "num_hidden_layers": 2, "num_attention_heads": 8,
                "num_key_value_heads": 4, "head_dim": 128, "vocab_size": 8,
                "layer_types": ["sliding_attention", "full_attention"],
            });
            document
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            document
        };
        let group = |heads, head_dim| Group {
            num_key_value_heads: heads,
            head_dim,
        };
        for extra in [
            json!({"per_layer_config": {"1": {"num_key_value_heads": 2, "head_dim": 256}}}),
            json!({"num_global_key_value_heads": 2, "global_head_dim": 256}),
        ] {
            let language: Language<Nothing> = parse(&document(extra)).unwrap();
            assert_eq!(language.group("sliding_attention"), group(4, 128));
            assert_eq!(language.group("full_attention"), group(2, 256));
        }
        let mut lone = document(json!({"per_layer_config": {"1": {"head_dim": 256}}}));
        lone.as_object_mut().unwrap().remove("layer_types");
        let error = parse::<Language<Nothing>>(&lone).unwrap_err();
        assert_eq!(
            (error.kind(), error.to_string()),
            (
                io::ErrorKind::Unsupported,
                "found per_layer_config without layer_types, compiled one geometry for every layer"
                    .into()
            )
        );
    }

    /// A decoder config with floating-point fields, as published checkpoints ship it, parses: the
    /// workspace's `arbitrary_precision` JSON numbers must survive the flattened parts.
    #[test]
    fn parses_published_float_fields() {
        let document = serde_json::from_str::<serde_json::Value>(
            r#"{"hidden_size": 5120, "intermediate_size": 14336, "num_hidden_layers": 40,
                "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                "vocab_size": 131072, "rms_norm_eps": 1e-05, "rope_theta": 1000000.0,
                "max_position_embeddings": 1024000, "hidden_act": "silu"}"#,
        )
        .unwrap();
        let decoder: Config = parse(&document).unwrap();
        assert_eq!(decoder.geometry.rms_norm_eps, Some(1e-5));
        assert_eq!(decoder.rotary.0.get(), 1e6);
        assert_eq!(decoder.capacity(40960), 40960);
        assert_eq!(decoder.capacity(2048000), 1024000);
    }

    /// The activation reads under either name; another activation is unsupported under either,
    /// named by its key with the compiled one.
    #[test]
    fn reads_either_activation_name() {
        for names in [
            json!({}),
            json!({"hidden_act": "silu"}),
            json!({"hidden_activation": "silu"}),
            json!({"hidden_act": "silu", "hidden_activation": "silu"}),
        ] {
            super::super::parse::<Activation<"silu">>(&names).unwrap();
        }
        for (names, reason) in [
            (
                json!({"hidden_activation": "gelu_pytorch_tanh"}),
                r#"hidden_activation: found "gelu_pytorch_tanh", compiled "silu""#,
            ),
            (
                json!({"hidden_act": "gelu", "hidden_activation": "silu"}),
                r#"hidden_act: found "gelu", compiled "silu""#,
            ),
            (
                json!({"hidden_act": "silu", "hidden_activation": "gelu"}),
                r#"hidden_activation: found "gelu", compiled "silu""#,
            ),
        ] {
            let error = parse::<Activation<"silu">>(&names).unwrap_err();
            let refusal = (error.kind(), error.to_string());
            assert_eq!(
                refusal,
                (io::ErrorKind::Unsupported, reason.into()),
                "{names}"
            );
        }
    }

    /// A family that holds an unscaled base refuses a rescaled or partial spectrum as unsupported,
    /// in either spelling, naming the spectrum it found.
    #[test]
    fn refuses_rescaled_spectra() {
        let llama3 = json!({
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
        });
        let mut nested = llama3.clone();
        nested["rope_theta"] = json!(5e5);
        let rescaled = "Llama3(Config { factor: Positive(8.0), low: Positive(1.0), \
                        high: Positive(4.0), original: 8192 }) rotating 1";
        for (document, found) in [
            (json!({"rope_theta": 5e5, "rope_scaling": llama3}), rescaled),
            (json!({"rope_parameters": nested}), rescaled),
            (
                json!({"rope_theta": 5e5, "partial_rotary_factor": 0.5}),
                "Default rotating 0.5",
            ),
        ] {
            let error = parse::<Rotary>(&document).unwrap_err();
            let reason = format!(
                "found {found} of each head, compiled the unscaled spectrum over the whole head"
            );
            assert_eq!(
                (error.kind(), error.to_string()),
                (io::ErrorKind::Unsupported, reason),
                "{document}"
            );
        }
    }

    /// A family that rotates a head's prefix holds the unscaled spectrum whose rotated width is
    /// the prefix; a spectrum over another width, or a rescaled one, is refused as unsupported,
    /// naming the spectrum it found.
    #[test]
    fn refuses_spectra_off_the_rotated_prefix() {
        use furiosa_opt_std::prelude::*;
        axes![Whole = 128, Half = 64];
        let prefix = |document: serde_json::Value| {
            let spectrum: Spectrum = parse(&document).unwrap();
            Prefix::<Whole, Half>::try_from(spectrum).map_err(super::super::refusal)
        };
        let theta = prefix(json!({"rope_theta": 1e4, "partial_rotary_factor": 0.5})).unwrap();
        assert_eq!(theta.0.get(), 1e4);
        for (document, found) in [
            (json!({"rope_theta": 1e4}), "Default rotating 1"),
            (
                json!({"rope_theta": 1e4, "partial_rotary_factor": 0.25}),
                "Default rotating 0.25",
            ),
            (
                json!({"rope_parameters": {
                    "rope_type": "linear", "factor": 2.0, "rope_theta": 1e4,
                    "partial_rotary_factor": 0.5,
                }}),
                "Linear { factor: 2.0 } rotating 0.5",
            ),
            (
                json!({"rope_parameters": {
                    "rope_type": "proportional", "rope_theta": 1e4, "partial_rotary_factor": 0.5,
                }}),
                "Proportional { factor: Positive(1.0) } rotating 0.5",
            ),
        ] {
            let error = prefix(document.clone()).unwrap_err();
            let reason = format!(
                "found {found} of each head, compiled the unscaled spectrum over the first 64 of \
                 128 coordinates"
            );
            assert_eq!(
                (error.kind(), error.to_string()),
                (io::ErrorKind::Unsupported, reason),
                "{document}"
            );
        }
    }

    /// A family that interleaves a multimodal rotation's sections holds the unscaled spectrum
    /// with sections, or none, interleaved over a head's pairs; sections in one block each and a
    /// rescaled spectrum are unsupported, sections over another count of pairs malformed.
    #[test]
    fn holds_interleaved_sections_of_the_head() {
        use furiosa_opt_std::prelude::*;
        axes![Whole = 128];
        let sections = |section: serde_json::Value, interleaved: bool| {
            json!({"rope_theta": 5e6, "rope_scaling": {
                "rope_type": "default", "mrope_section": section, "mrope_interleaved": interleaved,
            }})
        };
        for document in [
            sections(json!([24, 20, 20]), true),
            json!({"rope_theta": 5e6}),
        ] {
            let Interleaved(theta, _) = parse::<Interleaved<Whole>>(&document).unwrap();
            assert_eq!(theta.get(), 5e6, "{document}");
        }
        let linear = json!({"rope_theta": 5e6, "rope_scaling": {
            "rope_type": "linear", "factor": 2.0, "mrope_section": [24, 20, 20],
            "mrope_interleaved": true,
        }});
        for (document, kind, reason) in [
            (
                sections(json!([24, 20, 20]), false),
                io::ErrorKind::Unsupported,
                "found sections [24, 20, 20] in one block each, compiled interleaved sections",
            ),
            (
                sections(json!([24, 20, 16]), true),
                io::ErrorKind::InvalidData,
                "mrope_section: found [24, 20, 16], expected sections of the 64 pairs of a \
                 128-wide head",
            ),
            (
                linear,
                io::ErrorKind::Unsupported,
                "found Linear { factor: 2.0 } rotating 1 of each head, compiled the unscaled \
                 spectrum over the whole head",
            ),
        ] {
            let error = parse::<Interleaved<Whole>>(&document).unwrap_err();
            assert_eq!(
                (error.kind(), error.to_string()),
                (kind, reason.into()),
                "{document}"
            );
        }
    }
}
