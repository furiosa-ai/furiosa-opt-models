//! A checkpoint's rotary spectrum, bound before execution: no scaling branch remains in model
//! forward.
//!
//! - A spectrum is a base `theta`, the fraction of a head it rotates and the kind of rescaling of
//!   its frequencies that `rope_type` names. [`Spectrum::frequencies`] computes Transformers'
//!   frequency of each rotated pair, for the head layout the rotary op holds.
//! - A checkpoint spells one spectrum as `rope_parameters`, or as `rope_theta` beside
//!   `rope_scaling`; a description nested by layer type is [`Spectra`]. Two spellings of one value
//!   must agree.
//! - A field that a configuration leaves out reads as its class's [`Defaults`]; read without a
//!   class, a spectrum refuses a missing base.

use super::{Positive, refuse};
use crate::ops::pointwise::multiply::Temperature;
use crate::ops::rotary::PairFrequencies;
use furiosa_opt_std::prelude::AxisName;
use serde::{Deserialize, Deserializer, de};
use serde_json::{Map, Value};
use std::fmt::Debug;
use std::mem;
use std::{f64::consts::TAU, io, num::NonZeroUsize};

/// One rotary spectrum: every pair `i` of the rotated coordinates turns at a frequency derived
/// from `theta^(-2i / head)` by `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(try_from = "Spectra")]
pub struct Spectrum {
    theta: Positive,
    /// `partial_rotary_factor`: the fraction of a head's coordinates the spectrum rotates, in
    /// (0, 1] as the parse establishes.
    fraction: f64,
    kind: Kind,
}

/// The rescaling of a spectrum's frequencies, as `rope_type` names it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// `default`: the unscaled frequencies.
    Default,
    /// `proportional`: the unscaled frequencies over the whole head width, divided by `factor`, in
    /// the first `fraction` of each head half; the other coordinates are not rotated.
    Proportional { factor: Positive },
    /// `linear`: the unscaled frequencies divided by `factor`, at least one.
    Linear { factor: f64 },
    /// `llama3`: Llama 3's rescaling of the low frequencies.
    Llama3(Config),
    /// `yarn`: YaRN's ramp from extrapolated to interpolated frequencies, and its amplitude.
    Yarn(Yarn),
}

impl Spectrum {
    pub fn theta(&self) -> Positive {
        self.theta
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub(super) fn fraction(&self) -> f64 {
        self.fraction
    }

    /// The frequency of each of the `rotated / 2` pairs of a `head`-wide head, as Transformers
    /// computes it; refused when it rotates part of a head, since `i` pairs with `i + head / 2`.
    pub fn frequencies(&self, head: usize, rotated: usize) -> io::Result<Vec<f64>> {
        let theta = f64::from(self.theta.get());
        let unscaled = |pair: usize| theta.powf(-2.0 * pair as f64 / head as f64);
        // Transformers truncates the rotated width, `int(head * fraction)`.
        let width = match self.kind {
            Kind::Proportional { .. } => 2 * (self.fraction * head as f64 / 2.0) as usize,
            _ => (self.fraction * head as f64) as usize,
        };
        let holds = match self.kind {
            Kind::Proportional { .. } => width == rotated,
            _ => width == head && rotated == head,
        };
        if !holds {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "partial_rotary_factor: found {} of a {head}-wide head, compiled {rotated} \
                     rotated coordinates",
                    self.fraction
                ),
            ));
        }
        let pairs = 0..rotated / 2;
        Ok(match self.kind {
            Kind::Default => pairs.map(unscaled).collect(),
            Kind::Proportional { factor } => pairs
                .map(|pair| unscaled(pair) / f64::from(factor.get()))
                .collect(),
            Kind::Linear { factor } => pairs.map(|pair| unscaled(pair) / factor).collect(),
            Kind::Llama3(config) => pairs.map(|pair| config.rescale(unscaled(pair))).collect(),
            Kind::Yarn(yarn) => yarn.frequencies(theta, head),
        })
    }

    /// The positions a checkpoint declaring `trained` serves, as vLLM and SGLang derive them:
    /// `factor` times more under linear scaling; a YaRN or Llama 3 one declares its extension.
    pub fn context(&self, trained: NonZeroUsize) -> usize {
        match self.kind {
            Kind::Linear { factor } => (trained.get() as f64 * factor) as usize,
            _ => trained.get(),
        }
    }

    /// The frequencies of a `Head`-wide head's `Rotated` coordinates, at the spectrum's
    /// amplitude.
    pub fn bind<Head: AxisName, Rotated: AxisName>(&self) -> io::Result<PairFrequencies<Rotated>> {
        let amplitude = match self.kind {
            Kind::Yarn(yarn) => yarn.amplitude,
            _ => 1.0,
        };
        PairFrequencies::new(&self.frequencies(Head::SIZE, Rotated::SIZE)?, amplitude)
    }
}

/// The values a configuration class gives the rotary fields that a configuration leaves out, as
/// Transformers' class attributes give them: its `default_theta`, and the part of each head it
/// rotates.
pub trait Defaults {
    /// The base of a configuration that names none.
    const THETA: Positive;

    /// The fraction of each head that a configuration naming none rotates.
    const FRACTION: f64 = 1.0;

    /// Deserializes a flattened spectrum as the class reads it.
    fn spectrum<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Spectrum, D::Error> {
        let document: Document = super::flattened(deserializer)?;
        Spectra::read(document, Some(Self::THETA), Self::FRACTION)
            .and_then(Spectrum::try_from)
            .map_err(de::Error::custom)
    }
}

/// The spectrum of every layer type: one for every layer, or one per layer type that a
/// description nested by layer type names, `None` for a type that rotates nothing.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(try_from = "Document")]
pub struct Spectra(Layers);

#[derive(Clone, Debug, PartialEq)]
enum Layers {
    Shared(Spectrum),
    Typed(Vec<(String, Option<Spectrum>)>),
}

impl Spectra {
    /// The spectrum of the layers of type `layer`; `None` where they rotate nothing.
    pub fn of(&self, layer: &str) -> io::Result<Option<Spectrum>> {
        match &self.0 {
            Layers::Shared(spectrum) => Ok(Some(*spectrum)),
            Layers::Typed(types) => types
                .iter()
                .find_map(|(name, spectrum)| (name == layer).then_some(*spectrum))
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("rope_parameters: no entry for layer type {layer:?}"),
                    )
                }),
        }
    }
}

/// The fields that spell a spectrum, flat or nested, with the defaults around them.
#[derive(Deserialize)]
struct Document {
    rope_theta: Option<Positive>,
    rope_parameters: Option<Map<String, Value>>,
    rope_scaling: Option<Map<String, Value>>,
    partial_rotary_factor: Option<f64>,
    original_max_position_embeddings: Option<NonZeroUsize>,
    max_position_embeddings: Option<NonZeroUsize>,
}

impl TryFrom<Document> for Spectra {
    type Error = Refusal;

    fn try_from(document: Document) -> Result<Self, Self::Error> {
        Self::read(document, None, 1.0)
    }
}

impl Spectra {
    /// The spectra `document` spells, a description that names no base taking `theta` and one
    /// that names no fraction rotating `fraction` of each head.
    fn read(document: Document, theta: Option<Positive>, fraction: f64) -> Result<Self, Refusal> {
        let outer = Outer {
            theta: document.rope_theta,
            fraction: document.partial_rotary_factor,
            original: document.original_max_position_embeddings,
            declared: document.max_position_embeddings,
            default_theta: theta,
            default_fraction: fraction,
        };
        // Transformers 5 nests the base among the rescaling's parameters; earlier releases keep
        // the base at the top and the rescaling under its own name.
        let layers = match (document.rope_parameters, document.rope_scaling) {
            (Some(parameters), Some(scaling)) => {
                let layers = outer.layers(parameters)?;
                if outer.layers(scaling)? != layers {
                    return Err("rope_scaling: differs from rope_parameters beside it".to_owned());
                }
                layers
            }
            (parameters, scaling) => outer.layers(parameters.or(scaling).unwrap_or_default())?,
        };
        Ok(Self(layers))
    }
}

/// One spectrum for every layer; a nested description must give every layer type the same.
impl TryFrom<Spectra> for Spectrum {
    type Error = Refusal;

    fn try_from(Spectra(layers): Spectra) -> Result<Self, Self::Error> {
        let types = match layers {
            Layers::Shared(spectrum) => return Ok(spectrum),
            Layers::Typed(types) => types,
        };
        let mut spectra = types.into_iter().map(|(_, spectrum)| spectrum);
        match spectra.next().flatten() {
            Some(first) if spectra.all(|other| other == Some(first)) => Ok(first),
            _ => Err(refuse("a spectrum per layer type", "one for every layer")),
        }
    }
}

/// Transformers' name for the field of a spectrum's parameters that scales each row's rotated
/// queries by `1 + beta * ln(1 + floor(position / original))`; a [`Spectrum`] alone refuses it.
pub const TEMPERATURE: &str = "llama_4_scaling_beta";

/// A spectrum, and the temperature by which each row's queries scale after their rotation where
/// the spectrum's parameters declare one.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(try_from = "Map<String, Value>")]
pub struct Rope {
    pub spectrum: Spectrum,
    pub temperature: Option<Temperature>,
}

impl TryFrom<Map<String, Value>> for Rope {
    type Error = Refusal;

    fn try_from(mut document: Map<String, Value>) -> Result<Self, Self::Error> {
        let outer = document.get("original_max_position_embeddings").cloned();
        // Both spellings of the parameters must agree, so each loses its temperature, which must
        // agree with the other's.
        let declared = ["rope_parameters", "rope_scaling"]
            .into_iter()
            .filter_map(|spelling| match document.get_mut(spelling) {
                Some(Value::Object(parameters)) => parameters.remove(TEMPERATURE).map(|beta| {
                    let original = parameters.get("original_max_position_embeddings").cloned();
                    (beta, original)
                }),
                _ => None,
            })
            .collect::<Vec<_>>()
            .into_iter()
            .try_fold(None, |held, declaration| {
                agree(TEMPERATURE, held, Some(declaration))
            })?;
        let malformed = |error: serde_json::Error| error.to_string();
        let spectrum = Spectrum::deserialize(Value::Object(document)).map_err(malformed)?;
        let temperature = declared
            .map(|(beta, original)| {
                let number = |value: Option<Value>| value.map(serde_json::from_value).transpose();
                let beta: f32 = serde_json::from_value(beta).map_err(malformed)?;
                let original = agree(
                    "original_max_position_embeddings",
                    number(original).map_err(malformed)?,
                    number(outer).map_err(malformed)?,
                )?
                .ok_or_else(|| {
                    format!("{TEMPERATURE}: missing field `original_max_position_embeddings`")
                })?;
                match beta.is_finite() {
                    true => Ok(Temperature { beta, original }),
                    false => Err(format!(
                        "{TEMPERATURE}: found {beta}, expected a finite value"
                    )),
                }
            })
            .transpose()?;
        Ok(Self {
            spectrum,
            temperature,
        })
    }
}

/// What one spectrum's description defaults from the document around it, then from the
/// configuration's class.
struct Outer {
    theta: Option<Positive>,
    fraction: Option<f64>,
    original: Option<NonZeroUsize>,
    /// `max_position_embeddings`: the context the checkpoint declares.
    declared: Option<NonZeroUsize>,
    /// The class's base, which a description naming none takes; without one it is refused.
    default_theta: Option<Positive>,
    /// The class's fraction of each head, which a description naming none rotates.
    default_fraction: f64,
}

/// A refusal raised outside serde, which keeps the unsupported prefix of [`refuse`].
type Refusal = String;

impl Outer {
    /// The spectra a description gives: one of its rotary fields, or one per layer type that
    /// keys an object, or null for no rotation, by the type's name.
    fn layers(&self, parameters: Map<String, Value>) -> Result<Layers, Refusal> {
        // A null among objects is a layer type without rotation; among fields, an absent field.
        let objects = parameters
            .values()
            .filter(|value| value.is_object())
            .count();
        let nulls = parameters.values().filter(|value| value.is_null()).count();
        if objects == 0 {
            return self.spectrum(parameters).map(Layers::Shared);
        }
        if objects + nulls != parameters.len() {
            return Err("rope_parameters: mixes layer types with rotary fields".to_owned());
        }
        parameters
            .into_iter()
            .map(|(layer, entry)| {
                let spectrum = match entry {
                    Value::Object(parameters) => Some(self.spectrum(parameters)?),
                    _ => None,
                };
                Ok((layer, spectrum))
            })
            .collect::<Result<_, Refusal>>()
            .map(Layers::Typed)
    }

    fn spectrum(&self, mut parameters: Map<String, Value>) -> Result<Spectrum, Refusal> {
        if parameters.contains_key(TEMPERATURE) {
            return Err(refuse(
                format_args!("a query temperature ({TEMPERATURE})"),
                "a rotation alone",
            ));
        }
        #[derive(Deserialize)]
        struct Common {
            rope_type: Option<String>,
            #[serde(rename = "type")]
            alias: Option<String>,
            rope_theta: Option<Positive>,
            partial_rotary_factor: Option<f64>,
            original_max_position_embeddings: Option<NonZeroUsize>,
            factor: Option<Positive>,
        }
        let common = Common::deserialize(Value::Object(parameters.clone()))
            .map_err(|error| error.to_string())?;
        let kind = match agree("rope_type", common.rope_type, common.alias)? {
            Some(kind) => kind,
            // Only a base and its rotated fraction stand without a kind.
            None if parameters
                .keys()
                .all(|key| ["rope_theta", "partial_rotary_factor"].contains(&key.as_str())) =>
            {
                "default".to_owned()
            }
            None => {
                return Err(refuse(
                    "a rescaling without rope_type",
                    "a rope_type for every rescaling",
                ));
            }
        };
        let theta = agree("rope_theta", common.rope_theta, self.theta)?
            .or(self.default_theta)
            .ok_or("missing field `rope_theta`")?;
        let fraction = agree(
            "partial_rotary_factor",
            common.partial_rotary_factor,
            self.fraction,
        )?
        .unwrap_or(self.default_fraction);
        if !(fraction > 0.0 && fraction <= 1.0) {
            return Err(format!(
                "partial_rotary_factor: found {fraction}, expected a fraction in (0, 1]"
            ));
        }
        // A rescaled spectrum's trained context is its own value; Transformers takes the declared
        // context where a checkpoint names none.
        let original = agree(
            "original_max_position_embeddings",
            common.original_max_position_embeddings,
            self.original,
        )?
        .or(self.declared);
        let mut typed = |kind: &str| {
            parameters.insert("rope_type".to_owned(), kind.into());
            if let Some(original) = original {
                parameters.insert(
                    "original_max_position_embeddings".to_owned(),
                    original.get().into(),
                );
            }
            Value::Object(mem::take(&mut parameters))
        };
        let kind = match kind.as_str() {
            "default" => Kind::Default,
            "proportional" => Kind::Proportional {
                factor: common.factor.unwrap_or(Positive(1.0)),
            },
            // Transformers requires the factor of a linear rescaling.
            "linear" => Kind::Linear {
                factor: match parameters.get("factor").and_then(Value::as_f64) {
                    Some(factor) if factor >= 1.0 && factor.is_finite() => factor,
                    Some(factor) => return Err(format!("factor: found {factor}, expected >= 1")),
                    None => return Err("missing field `factor`".to_owned()),
                },
            },
            "llama3" => Kind::Llama3(
                Config::deserialize(typed("llama3")).map_err(|error| error.to_string())?,
            ),
            "yarn" => {
                let mut yarn = typed("yarn");
                // Transformers reads a null factor as the declared context over the trained one.
                let null = yarn.get("factor").is_some_and(Value::is_null);
                if let (true, Some(declared), Some(original)) = (null, self.declared, original) {
                    yarn["factor"] = (declared.get() as f64 / original.get() as f64).into();
                }
                Kind::Yarn(Yarn::deserialize(yarn).map_err(|error| error.to_string())?)
            }
            other => {
                return Err(refuse(
                    format_args!("rope_type {other:?}"),
                    "default, proportional, linear, llama3 or yarn",
                ));
            }
        };
        Ok(Spectrum {
            theta,
            fraction,
            kind,
        })
    }
}

/// Llama 3's rescaling: a frequency whose wavelength exceeds `original / low` is divided by
/// `factor`, one below `original / high` is kept, and one between blends the two.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(try_from = "Raw")]
pub struct Config {
    pub(crate) factor: Positive,
    pub(crate) low: Positive,
    pub(crate) high: Positive,
    pub(crate) original: NonZeroUsize,
}

impl Config {
    /// Llama 3 rescaled rotary frequency of each dimension pair of a `head`-wide attention head.
    pub fn frequencies(&self, theta: Positive, head: usize) -> Vec<f32> {
        let theta = f64::from(theta.get());
        (0..head / 2)
            .map(|pair| self.rescale(theta.powf(-2.0 * pair as f64 / head as f64)) as f32)
            .collect()
    }

    /// The rescaled value of an unscaled frequency.
    fn rescale(&self, frequency: f64) -> f64 {
        let (factor, low, high, original) = (
            f64::from(self.factor.get()),
            f64::from(self.low.get()),
            f64::from(self.high.get()),
            self.original.get() as f64,
        );
        let wavelength = TAU / frequency;
        if wavelength > original / low {
            frequency / factor
        } else if wavelength < original / high {
            frequency
        } else {
            let blend = (original / wavelength - low) / (high - low);
            (1.0 - blend) * frequency / factor + blend * frequency
        }
    }
}

#[derive(serde::Deserialize)]
struct Raw {
    rope_type: super::Literal<"llama3">,
    factor: Positive,
    low_freq_factor: Positive,
    high_freq_factor: Positive,
    original_max_position_embeddings: NonZeroUsize,
}

impl TryFrom<Raw> for Config {
    type Error = String;

    fn try_from(raw: Raw) -> Result<Self, Self::Error> {
        let super::Literal = raw.rope_type;
        if raw.factor.get() < 1.0 {
            return Err(format!("factor: found {}, expected >= 1", raw.factor.get()));
        }
        if raw.high_freq_factor.get() <= raw.low_freq_factor.get() {
            return Err(format!(
                "high_freq_factor: found {}, expected > low_freq_factor {}",
                raw.high_freq_factor.get(),
                raw.low_freq_factor.get()
            ));
        }
        Ok(Self {
            factor: raw.factor,
            low: raw.low_freq_factor,
            high: raw.high_freq_factor,
            original: raw.original_max_position_embeddings,
        })
    }
}

/// YaRN's rescaling: pairs turning over `fast` times in the `original` context keep their
/// frequency, under `slow` divide it by `factor`, a ramp blends; `amplitude` scales each rotation.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(try_from = "YarnRaw")]
pub struct Yarn {
    factor: f64,
    fast: f64,
    slow: f64,
    original: NonZeroUsize,
    truncate: bool,
    amplitude: f64,
}

impl Yarn {
    /// The frequency of each pair of a `head`-wide head, as Transformers' YaRN computes it.
    fn frequencies(&self, theta: f64, head: usize) -> Vec<f64> {
        let width = head as f64;
        // The pair index at which a frequency turns `rotations` times over the original context.
        let correction = |rotations: f64| {
            width * (self.original.get() as f64 / (rotations * TAU)).ln() / (2.0 * theta.ln())
        };
        let (low, high) = match self.truncate {
            true => (correction(self.fast).floor(), correction(self.slow).ceil()),
            false => (correction(self.fast), correction(self.slow)),
        };
        let (low, high) = (low.max(0.0), high.min(width - 1.0));
        // Transformers widens an empty ramp by a thousandth of a pair.
        let high = if low == high { high + 0.001 } else { high };
        (0..head / 2)
            .map(|pair| {
                let extrapolated = theta.powf(-2.0 * pair as f64 / width);
                let kept = 1.0 - ((pair as f64 - low) / (high - low)).clamp(0.0, 1.0);
                extrapolated / self.factor * (1.0 - kept) + extrapolated * kept
            })
            .collect()
    }
}

#[derive(serde::Deserialize)]
struct YarnRaw {
    rope_type: super::Literal<"yarn">,
    factor: Positive,
    original_max_position_embeddings: NonZeroUsize,
    beta_fast: Option<f64>,
    beta_slow: Option<f64>,
    attention_factor: Option<f64>,
    mscale: Option<f64>,
    mscale_all_dim: Option<f64>,
    truncate: Option<bool>,
}

impl TryFrom<YarnRaw> for Yarn {
    type Error = String;

    fn try_from(raw: YarnRaw) -> Result<Self, Self::Error> {
        let super::Literal = raw.rope_type;
        let factor = f64::from(raw.factor.get());
        if factor < 1.0 {
            return Err(format!("factor: found {factor}, expected >= 1"));
        }
        // Transformers reads an absent or zero ramp bound as its default.
        let nonzero = |value: Option<f64>| value.filter(|value| *value != 0.0);
        let (fast, slow) = (
            nonzero(raw.beta_fast).unwrap_or(32.0),
            nonzero(raw.beta_slow).unwrap_or(1.0),
        );
        if !(fast > slow && slow > 0.0) {
            return Err(format!(
                "beta_fast: found {fast}, expected > beta_slow {slow} > 0"
            ));
        }
        let scale = |mscale: f64| match factor <= 1.0 {
            true => 1.0,
            false => 0.1 * mscale * factor.ln() + 1.0,
        };
        let amplitude = match (
            raw.attention_factor,
            nonzero(raw.mscale),
            nonzero(raw.mscale_all_dim),
        ) {
            (Some(amplitude), _, _) => amplitude,
            (None, Some(mscale), Some(all)) => scale(mscale) / scale(all),
            (None, _, _) => scale(1.0),
        };
        if !(amplitude > 0.0 && amplitude.is_finite()) {
            return Err(format!(
                "attention_factor: found {amplitude}, expected a positive finite value"
            ));
        }
        Ok(Self {
            factor,
            fast,
            slow,
            original: raw.original_max_position_embeddings,
            truncate: raw.truncate.unwrap_or(true),
            amplitude,
        })
    }
}

/// The value two spellings of one field give, refusing two that differ.
fn agree<T: PartialEq + Debug>(
    field: &str,
    nested: Option<T>,
    outer: Option<T>,
) -> Result<Option<T>, Refusal> {
    match (nested, outer) {
        (Some(nested), Some(outer)) if nested != outer => {
            Err(format!("{field}: found {nested:?} and {outer:?}"))
        }
        (nested, outer) => Ok(nested.or(outer)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_std::prelude::*;
    use serde::de;
    use serde_json::json;

    axes![Head = 128, Wide = 512, Quarter = 128];

    fn parse<T: de::DeserializeOwned>(document: Value) -> io::Result<T> {
        super::super::parse(&document)
    }

    /// YaRN parameters, with Transformers' `type` alias.
    fn yarn() -> Value {
        json!({
            "rope_type": "yarn", "type": "yarn", "factor": 16.0, "beta_fast": 32.0,
            "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 16384,
        })
    }

    /// Llama 3 rescaling at factor 16 over an 8192-token trained context.
    fn llama3() -> Value {
        json!({
            "rope_type": "llama3", "factor": 16.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
        })
    }

    /// `parameters` with `theta` among them, as Transformers 5 writes them.
    fn nested(mut parameters: Value, theta: f64) -> Value {
        parameters["rope_theta"] = json!(theta);
        json!({ "rope_parameters": parameters })
    }

    /// One spectrum reads alike from `rope_parameters` and from `rope_theta` beside
    /// `rope_scaling`, however each spells its base, a null field reading as absent; two
    /// spellings that differ, or layer types mixed with rotary fields, are malformed.
    #[test]
    fn reads_one_spectrum_from_either_spelling() {
        for (parameters, theta) in [(yarn(), 1e9), (llama3(), 1e6)] {
            let flat = json!({"rope_theta": theta as u64, "rope_scaling": parameters.clone()});
            let spectrum: Spectrum = parse(nested(parameters.clone(), theta)).unwrap();
            assert_eq!(parse::<Spectrum>(flat).unwrap(), spectrum);
            assert_eq!(f64::from(spectrum.theta.get()), theta);
            let mut both = nested(parameters, theta);
            both["rope_scaling"] = both["rope_parameters"].clone();
            assert_eq!(parse::<Spectrum>(both).unwrap(), spectrum);
        }
        assert_eq!(
            parse::<Spectrum>(json!({"rope_theta": 1e4})).unwrap(),
            Spectrum {
                theta: Positive(1e4),
                fraction: 1.0,
                kind: Kind::Default
            }
        );
        let absent =
            json!({"rope_parameters": {"rope_type": "default", "rope_theta": 1e4, "factor": null}});
        assert_eq!(
            parse::<Spectrum>(absent).unwrap(),
            parse::<Spectrum>(json!({"rope_theta": 1e4})).unwrap()
        );
        let mut inherited = json!({"rope_theta": 1e9, "rope_scaling": yarn()});
        inherited["rope_parameters"] = nested(yarn(), 1e9)["rope_parameters"].clone();
        assert_eq!(
            parse::<Spectrum>(inherited).unwrap(),
            parse::<Spectrum>(nested(yarn(), 1e9)).unwrap()
        );
        let mut aliased = yarn();
        aliased["type"] = json!("llama3");
        let mut scaled = nested(llama3(), 1e6);
        scaled["rope_scaling"] = nested(llama3(), 1e6)["rope_parameters"].clone();
        scaled["rope_scaling"]["factor"] = json!(8.0);
        for (document, reason) in [
            (
                json!({"rope_theta": 1e6, "rope_parameters": {"rope_theta": 1e4}}),
                "rope_theta: found Positive(10000.0) and Positive(1000000.0)",
            ),
            (
                nested(aliased, 1e9),
                r#"rope_type: found "yarn" and "llama3""#,
            ),
            (
                scaled,
                "rope_scaling: differs from rope_parameters beside it",
            ),
            (
                json!({
                    "partial_rotary_factor": 0.5,
                    "rope_parameters": {"rope_theta": 1e4, "partial_rotary_factor": 0.25},
                }),
                "partial_rotary_factor: found 0.25 and 0.5",
            ),
            (
                json!({
                    "original_max_position_embeddings": 4096,
                    "rope_theta": 1e6, "rope_scaling": llama3(),
                }),
                "original_max_position_embeddings: found 8192 and 4096",
            ),
            (
                json!({"rope_parameters": {"sliding_attention": {"rope_theta": 1e4}, "rope_theta": 1e6}}),
                "rope_parameters: mixes layer types with rotary fields",
            ),
            (
                nested(json!({"rope_type": "linear"}), 1e4),
                "missing field `factor`",
            ),
            (
                nested(json!({"rope_type": "linear", "factor": 0.5}), 1e4),
                "factor: found 0.5, expected >= 1",
            ),
        ] {
            let refusal = parse::<Spectra>(document.clone()).unwrap_err();
            assert_eq!(
                (refusal.kind(), refusal.to_string()),
                (io::ErrorKind::InvalidData, reason.into()),
                "{document}"
            );
        }
    }

    /// Each rescaling's parameters are refused outside the ranges its formula takes, at the
    /// boundary each range admits: Llama 3 and YaRN factors below one, a Llama 3 band whose high
    /// factor is not above its low one, a YaRN ramp whose fast bound is not above its slow one, and
    /// a rotated fraction outside (0, 1].
    #[test]
    fn refuses_parameters_outside_their_ranges() {
        let with = |mut parameters: Value, field: &str, value: f64| {
            parameters[field] = json!(value);
            nested(parameters, 1e6)
        };
        for document in [
            with(llama3(), "factor", 1.0),
            with(llama3(), "high_freq_factor", 1.5),
            with(yarn(), "factor", 1.0),
            with(yarn(), "beta_fast", 1.5),
            with(
                json!({"rope_type": "default"}),
                "partial_rotary_factor",
                1.0,
            ),
            with(
                json!({"rope_type": "default"}),
                "partial_rotary_factor",
                1e-3,
            ),
        ] {
            parse::<Spectrum>(document.clone())
                .unwrap_or_else(|error| panic!("{document}: {error}"));
        }
        for (document, reason) in [
            (
                with(llama3(), "factor", 0.5),
                "factor: found 0.5, expected >= 1",
            ),
            (
                with(llama3(), "high_freq_factor", 1.0),
                "high_freq_factor: found 1, expected > low_freq_factor 1",
            ),
            (
                with(yarn(), "factor", 0.5),
                "factor: found 0.5, expected >= 1",
            ),
            (
                with(yarn(), "beta_fast", 1.0),
                "beta_fast: found 1, expected > beta_slow 1 > 0",
            ),
            (
                with(
                    json!({"rope_type": "default"}),
                    "partial_rotary_factor",
                    0.0,
                ),
                "partial_rotary_factor: found 0, expected a fraction in (0, 1]",
            ),
            (
                with(
                    json!({"rope_type": "default"}),
                    "partial_rotary_factor",
                    1.5,
                ),
                "partial_rotary_factor: found 1.5, expected a fraction in (0, 1]",
            ),
        ] {
            let refusal = parse::<Spectrum>(document.clone()).unwrap_err();
            assert_eq!(
                (refusal.kind(), refusal.to_string()),
                (io::ErrorKind::InvalidData, reason.into()),
                "{document}"
            );
        }
    }

    /// A query temperature among a spectrum's parameters reads beside the spectrum, in either
    /// spelling, over the trained context of its own parameters; a spectrum alone refuses it, a
    /// temperature without a trained context or outside FP32 is refused, and two spellings must
    /// declare the same one.
    #[test]
    fn reads_query_temperatures_beside_spectra() {
        let mut tempered = yarn();
        tempered["llama_4_scaling_beta"] = json!(0.1);
        let flat = json!({"rope_theta": 1e9, "rope_scaling": tempered.clone()});
        for document in [nested(tempered.clone(), 1e9), flat.clone()] {
            let rope: Rope = parse(document.clone()).unwrap();
            assert_eq!(
                rope.spectrum,
                parse::<Spectrum>(nested(yarn(), 1e9)).unwrap()
            );
            let temperature = rope.temperature.unwrap();
            assert_eq!((temperature.beta, temperature.original.get()), (0.1, 16384));
            let refusal = parse::<Spectrum>(document).unwrap_err();
            assert_eq!(
                (refusal.kind(), refusal.to_string()),
                (
                    io::ErrorKind::Unsupported,
                    "found a query temperature (llama_4_scaling_beta), compiled a rotation alone"
                        .into()
                )
            );
        }
        let unscaled: Rope = parse(nested(yarn(), 1e9)).unwrap();
        assert!(unscaled.temperature.is_none());
        let mut untrained =
            json!({"rope_parameters": {"rope_theta": 1e4, "llama_4_scaling_beta": 0.1}});
        let refusal = parse::<Rope>(untrained.clone()).unwrap_err();
        assert_eq!(
            (refusal.kind(), refusal.to_string()),
            (
                io::ErrorKind::InvalidData,
                "llama_4_scaling_beta: missing field `original_max_position_embeddings`".into()
            )
        );
        untrained["original_max_position_embeddings"] = json!(4096);
        let rope: Rope = parse(untrained.clone()).unwrap();
        assert_eq!(rope.temperature.unwrap().original.get(), 4096);
        untrained["rope_parameters"]["llama_4_scaling_beta"] = json!(3e38);
        let rope: Rope = parse(untrained.clone()).unwrap();
        assert_eq!(rope.temperature.unwrap().beta, 3e38);
        untrained["rope_parameters"]["llama_4_scaling_beta"] = json!(1e39);
        let refusal = parse::<Rope>(untrained).unwrap_err();
        assert_eq!(
            (refusal.kind(), refusal.to_string()),
            (
                io::ErrorKind::InvalidData,
                "llama_4_scaling_beta: found inf, expected a finite value".into()
            )
        );
        let tempered = |beta: f64, original: usize| {
            let mut parameters = yarn();
            parameters["llama_4_scaling_beta"] = json!(beta);
            parameters["original_max_position_embeddings"] = json!(original);
            parameters
        };
        let both = |first: Value, second: Value| {
            let mut document = nested(first, 1e9);
            document["rope_scaling"] = second;
            document["rope_scaling"]["rope_theta"] = json!(1e9);
            document
        };
        let agreeing: Rope = parse(both(tempered(0.1, 16384), tempered(0.1, 16384))).unwrap();
        assert_eq!(agreeing.temperature.unwrap().beta, 0.1);
        for (document, (beta, original)) in [
            (
                both(tempered(0.1, 16384), tempered(0.2, 16384)),
                ("0.2", 16384),
            ),
            (
                both(tempered(0.1, 16384), tempered(0.1, 8192)),
                ("0.1", 8192),
            ),
        ] {
            let refusal = parse::<Rope>(document.clone()).unwrap_err();
            let reason = format!(
                "llama_4_scaling_beta: found (Number(0.1), Some(Number(16384))) and \
                 (Number({beta}), Some(Number({original})))"
            );
            assert_eq!(
                (refusal.kind(), refusal.to_string()),
                (io::ErrorKind::InvalidData, reason),
                "{document}"
            );
        }
    }

    /// A description nested by layer type gives each type its own spectrum, none for a null
    /// entry; one spectrum for every layer refuses types that differ.
    #[test]
    fn reads_spectra_by_layer_type() {
        let typed = json!({
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {
                    "rope_type": "proportional", "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        });
        let spectra: Spectra = parse(typed.clone()).unwrap();
        let spectrum = |theta, fraction, kind| Spectrum {
            theta: Positive(theta),
            fraction,
            kind,
        };
        assert_eq!(
            spectra.of("sliding_attention").unwrap(),
            Some(spectrum(1e4, 1.0, Kind::Default))
        );
        let proportional = Kind::Proportional {
            factor: Positive(1.0),
        };
        assert_eq!(
            spectra.of("full_attention").unwrap(),
            Some(spectrum(1e6, 0.25, proportional))
        );
        let missing = spectra.of("chunked_attention").unwrap_err();
        assert_eq!(
            (missing.kind(), missing.to_string()),
            (
                io::ErrorKind::InvalidData,
                r#"rope_parameters: no entry for layer type "chunked_attention""#.into()
            )
        );
        let shared = parse::<Spectrum>(typed).unwrap_err();
        assert_eq!(
            (shared.kind(), shared.to_string()),
            (
                io::ErrorKind::Unsupported,
                "found a spectrum per layer type, compiled one for every layer".into()
            )
        );
        let unrotated: Spectra = parse(json!({
            "rope_theta": 5e5,
            "rope_parameters": {"sliding_attention": {"rope_type": "default"}, "full_attention": null},
        }))
        .unwrap();
        assert_eq!(unrotated.of("full_attention").unwrap(), None);
        assert_eq!(
            unrotated.of("sliding_attention").unwrap(),
            Some(spectrum(5e5, 1.0, Kind::Default))
        );
        let shared: Spectra = parse(json!({"rope_theta": 5e5})).unwrap();
        assert_eq!(
            shared.of("full_attention").unwrap(),
            Some(spectrum(5e5, 1.0, Kind::Default))
        );
    }

    /// Each pair's frequency matches Transformers 5.14's at the edges of every rescaled band, for
    /// YaRN, for Llama 3 rescaling at two factors and for linear scaling.
    #[test]
    fn matches_transformers_frequencies() {
        let mut llama31 = llama3();
        llama31["factor"] = json!(8.0);
        let cases: [(Value, f64, &[(usize, f64)]); 4] = [
            (
                json!({"rope_type": "linear", "factor": 4.0}),
                5e5,
                &[
                    (0, 2.5e-1),
                    (1, 2.036543041e-1),
                    (31, 4.340116866e-4),
                    (63, 6.137851756e-7),
                ],
            ),
            (
                yarn(),
                1e9,
                &[
                    (0, 1.0),
                    (1, 7.233941555e-1),
                    (13, 1.485508028e-2),
                    (14, 9.906540625e-3),
                    (19, 1.130899298e-3),
                    (24, 5.930106272e-5),
                    (25, 1.906579928e-5),
                    (63, 8.639826354e-11),
                ],
            ),
            (
                llama3(),
                1e6,
                &[
                    (0, 1.0),
                    (1, 8.058422208e-1),
                    (26, 3.651741426e-3),
                    (27, 2.792574232e-3),
                    (30, 5.812029121e-4),
                    (33, 6.312134065e-5),
                    (34, 4.058635386e-5),
                    (63, 7.755861020e-8),
                ],
            ),
            (
                llama31,
                5e5,
                &[
                    (0, 1.0),
                    (1, 8.146172166e-1),
                    (30, 1.371893683e-3),
                    (31, 8.567514597e-4),
                    (35, 9.556212171e-5),
                    (63, 3.068925878e-7),
                ],
            ),
        ];
        for (parameters, theta, golden) in cases {
            let spectrum: Spectrum = parse(nested(parameters, theta)).unwrap();
            let frequencies = spectrum.frequencies(Head::SIZE, Head::SIZE).unwrap();
            for &(pair, expected) in golden {
                // Transformers computes the frequencies in F32.
                let error = (frequencies[pair] - expected).abs() / expected;
                assert!(error <= 1e-6, "{theta} pair {pair}: {}", frequencies[pair]);
            }
        }
    }

    /// A YaRN spectrum binds Transformers' attention factor as its amplitude: given, implied by its
    /// two mscales or by its factor alone; a factor that is not positive is malformed.
    #[test]
    fn binds_yarn_attention_factors() {
        let with = |fields: Value, removed: &[&str]| {
            let mut parameters = yarn();
            let object = parameters.as_object_mut().unwrap();
            removed.iter().for_each(|field| _ = object.remove(*field));
            object.extend(fields.as_object().unwrap().clone());
            nested(parameters, 1e9)
        };
        let mscale = ["mscale", "mscale_all_dim"];
        for (document, amplitude) in [
            (with(json!({}), &[]), 1.0),
            (with(json!({"attention_factor": 0.9}), &[]), 0.9),
            (with(json!({}), &mscale), 1.277_258_872),
            (with(json!({"mscale": 0.707}), &[]), 0.936_397_506),
        ] {
            let spectrum: Spectrum = parse(document.clone()).unwrap();
            let bound = spectrum.bind::<Head, Head>().unwrap().amplitude();
            assert!(
                (f64::from(bound) - amplitude).abs() <= 1e-7,
                "{document}: {bound}"
            );
        }
        let refusal = parse::<Spectrum>(with(json!({"attention_factor": 0.0}), &[])).unwrap_err();
        assert_eq!(refusal.kind(), io::ErrorKind::InvalidData, "{refusal}");
    }

    /// The trained context defaults to the declared one and a null YaRN factor to their ratio
    /// (Transformers 5.14); a YaRN factor absent, or null with no declared context, is malformed.
    #[test]
    fn defaults_as_transformers_reads_the_config() {
        let declared = |parameters: Value, theta: f64| {
            let mut document = nested(parameters, theta);
            document["max_position_embeddings"] = json!(131072);
            document
        };
        let yarn = |fields: Value| {
            let mut parameters = json!({"rope_type": "yarn"});
            let object = parameters.as_object_mut().unwrap();
            object.extend(fields.as_object().unwrap().clone());
            parameters
        };
        let recipe = json!({"factor": 4.0, "original_max_position_embeddings": 32768});
        let explicit: Spectrum = parse(declared(yarn(recipe), 1e6)).unwrap();
        let cases: [(Value, f32, &[(usize, f64)]); 2] = [
            (
                yarn(json!({"factor": 4.0})),
                1.138_629_4,
                &[
                    (13, 6.04296438395977e-2),
                    (24, 5.623413249850273e-3),
                    (25, 4.531583748757839e-3),
                    (31, 1.186190522275865e-3),
                    (63, 3.102344408034696e-7),
                ],
            ),
            (
                yarn(json!({"factor": null})),
                1.0,
                &[(31, 1.2409378541633487e-3), (63, 1.2409377632138785e-6)],
            ),
        ];
        for (parameters, amplitude, golden) in cases {
            let spectrum: Spectrum = parse(declared(parameters.clone(), 1e6)).unwrap();
            let bound = spectrum.bind::<Head, Head>().unwrap();
            assert!(
                (bound.amplitude() - amplitude).abs() <= 1e-6,
                "{parameters}"
            );
            for &(pair, expected) in golden {
                let error = (f64::from(bound.pairs()[pair]) - expected).abs() / expected;
                assert!(
                    error <= 1e-6,
                    "{parameters} pair {pair}: {}",
                    bound.pairs()[pair]
                );
            }
        }
        let ratio = yarn(json!({"factor": null, "original_max_position_embeddings": 32768}));
        assert_eq!(parse::<Spectrum>(declared(ratio, 1e6)).unwrap(), explicit);
        let mut untrained = llama3();
        untrained
            .as_object_mut()
            .unwrap()
            .remove("original_max_position_embeddings");
        let mut llama31 = llama3();
        llama31["original_max_position_embeddings"] = json!(131072);
        assert_eq!(
            parse::<Spectrum>(declared(untrained, 5e5)).unwrap(),
            parse::<Spectrum>(declared(llama31, 5e5)).unwrap()
        );
        let undeclared = nested(yarn(json!({"factor": null})), 1e6);
        for document in [declared(yarn(json!({})), 1e6), undeclared] {
            let refusal = parse::<Spectrum>(document.clone()).unwrap_err();
            assert_eq!(
                refusal.kind(),
                io::ErrorKind::InvalidData,
                "{document}: {refusal}"
            );
        }
    }

    /// Another rope type, a rescaling without one and a partial layout that the rotary op does not
    /// pair are unsupported, and no base is invalid.
    #[test]
    fn refuses_unheld_spectra() {
        let kinds = "default, proportional, linear, llama3 or yarn";
        for (document, found, compiled) in [
            (
                nested(json!({"rope_type": "longrope", "factor": 2.0}), 1e4),
                r#"rope_type "longrope""#,
                kinds,
            ),
            (
                nested(json!({"rope_type": "dynamic", "factor": 2.0}), 1e4),
                r#"rope_type "dynamic""#,
                kinds,
            ),
            (
                json!({"rope_theta": 1e4, "rope_scaling": {"factor": 2.0}}),
                "a rescaling without rope_type",
                "a rope_type for every rescaling",
            ),
        ] {
            let refusal = parse::<Spectrum>(document.clone()).unwrap_err();
            assert_eq!(
                (refusal.kind(), refusal.to_string()),
                (
                    io::ErrorKind::Unsupported,
                    format!("found {found}, compiled {compiled}")
                ),
                "{document}"
            );
        }
        let baseless = json!({"rope_parameters": {"rope_type": "default"}});
        let refusal = parse::<Spectrum>(baseless).unwrap_err();
        assert_eq!(
            (refusal.kind(), refusal.to_string()),
            (
                io::ErrorKind::InvalidData,
                "missing field `rope_theta`".into()
            )
        );
        let layout = |spectrum: &Spectrum, (head, rotated): (usize, usize)| {
            let refusal = spectrum.frequencies(head, rotated).unwrap_err();
            (refusal.kind(), refusal.to_string())
        };
        let partial: Spectrum =
            parse(json!({"rope_theta": 1e4, "partial_rotary_factor": 0.5})).unwrap();
        assert_eq!(
            layout(&partial, (Head::SIZE, Head::SIZE / 2)),
            (
                io::ErrorKind::Unsupported,
                "partial_rotary_factor: found 0.5 of a 128-wide head, compiled 64 rotated \
                 coordinates"
                    .into()
            )
        );
        let proportional: Spectrum = parse(nested(
            json!({"rope_type": "proportional", "partial_rotary_factor": 0.25}),
            1e6,
        ))
        .unwrap();
        assert_eq!(proportional.frequencies(512, 128).unwrap().len(), 64);
        assert_eq!(
            layout(&proportional, (512, 256)),
            (
                io::ErrorKind::Unsupported,
                "partial_rotary_factor: found 0.25 of a 512-wide head, compiled 256 rotated \
                 coordinates"
                    .into()
            )
        );
    }

    /// A checkpoint serves `factor` times its trained context under linear scaling, as vLLM and
    /// SGLang derive their context length, and its trained context under any other spectrum.
    #[test]
    fn serves_the_context_frameworks_derive() {
        let trained = NonZeroUsize::new(8192).unwrap();
        for (parameters, context) in [
            (json!({"rope_type": "linear", "factor": 4.0}), 32768),
            (json!({"rope_type": "linear", "factor": 2.5}), 20480),
            (json!({"rope_type": "default"}), 8192),
            (llama3(), 8192),
            (yarn(), 8192),
        ] {
            let spectrum: Spectrum = parse(nested(parameters.clone(), 5e5)).unwrap();
            assert_eq!(spectrum.context(trained), context, "{parameters}");
        }
    }

    /// A spectrum binds the frequency of every pair, an unscaled one the progression of its base
    /// and a proportional one whose first pair turns at half speed its halved frequencies.
    #[test]
    fn binds_every_pair_frequency() {
        let unscaled: Spectrum = parse(json!({"rope_theta": 1e4})).unwrap();
        assert_eq!(
            unscaled.bind::<Head, Head>().unwrap(),
            PairFrequencies::unscaled::<Head>(1e4).unwrap()
        );
        let halved = json!({
            "rope_type": "proportional", "partial_rotary_factor": 0.25, "factor": 2.0,
        });
        let halved: Spectrum = parse(nested(halved, 1e6)).unwrap();
        let bound = halved.bind::<Wide, Quarter>().unwrap();
        let unscaled = PairFrequencies::<Quarter>::unscaled::<Wide>(1e6).unwrap();
        let ratios = bound.pairs().iter().zip(unscaled.pairs());
        assert!(
            ratios
                .map(|(halved, unscaled)| halved / unscaled)
                .all(|ratio| ratio == 0.5)
        );
    }

    /// A configuration class reads a spectrum's absent base and fraction as its own, under every
    /// spelling the document gives none in; a base or fraction the document names wins over the
    /// class's, at the top or nested.
    #[test]
    fn reads_absent_fields_as_the_class_defaults_them() {
        struct Class;
        impl Defaults for Class {
            const THETA: Positive = Positive(5e5);
            const FRACTION: f64 = 0.5;
        }
        #[derive(Debug, Deserialize)]
        struct Part {
            #[serde(flatten, deserialize_with = "Class::spectrum")]
            rope: Spectrum,
        }
        let read = |document: Value| parse::<Part>(document).unwrap().rope;
        let spectrum = |theta, fraction, kind| Spectrum {
            theta: Positive(theta),
            fraction,
            kind,
        };
        let linear = Kind::Linear { factor: 2.0 };
        for (document, expected) in [
            (json!({}), spectrum(5e5, 0.5, Kind::Default)),
            (
                json!({"rope_scaling": {"rope_type": "linear", "factor": 2.0}}),
                spectrum(5e5, 0.5, linear),
            ),
            (
                json!({"rope_theta": 1e4}),
                spectrum(1e4, 0.5, Kind::Default),
            ),
            (
                nested(json!({"rope_type": "linear", "factor": 2.0}), 1e6),
                spectrum(1e6, 0.5, linear),
            ),
            (
                json!({"partial_rotary_factor": 1.0}),
                spectrum(5e5, 1.0, Kind::Default),
            ),
        ] {
            assert_eq!(read(document.clone()), expected, "{document}");
        }
        let refusal = parse::<Part>(json!({"partial_rotary_factor": 0.0})).unwrap_err();
        assert_eq!(
            (refusal.kind(), refusal.to_string()),
            (
                io::ErrorKind::InvalidData,
                "partial_rotary_factor: found 0, expected a fraction in (0, 1]".into()
            )
        );
    }
}
