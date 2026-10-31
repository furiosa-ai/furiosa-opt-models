//! The choices one load makes: the registry implementation a checkpoint loads as, its chips, its KV
//! cache and the quantization it gives a BF16 checkpoint, each checked against the checkpoint's
//! config and names before any weight is read.
//!
//! - The architecture is the one the options name, else the first of the config's `architectures`,
//!   then its `_class_name`, that the registry lists, since a framework's fallback names may precede
//!   the native one; the registry then chooses the one implementation of it whose scope serves the
//!   checkpoint.
//! - A load reads the whole model, or its media encoder alone, which the registry lists as an
//!   implementation of its own.
//! - Every implementation runs on one chip, caches keys and values in BF16 and loads whole; any
//!   other choice is unsupported until the registry holds an implementation that makes it.
//! - A batch-invariant load asks every implementation's calls to tile their attention fixed, which
//!   they serve as an `Instance` of their decoder.
//! - A quantization reads the checkpoint quantized, so only an implementation whose projections a
//!   load quantizes serves it; every other one holds BF16 projections.
//! - A load for calls that run LoRA adapters selects only an implementation whose projections take
//!   them.

use std::fmt;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::str::FromStr;

use crate::registry::{Implementation, Model};
use furiosa_opt_models::load::{Checkpoint, Context, LoadError, Part};
use furiosa_opt_models::ops::attention::full::paged::Tiling;
use furiosa_opt_models::tensor::form::fp8::Granularity;
use furiosa_opt_std::prelude::Device;
use safetensors::Dtype;
use serde_json::Value;

/// The choices of one load; a choice the registry refuses is reported before any weight is read.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    chips: Option<NonZeroUsize>,
    kv_cache: Option<Dtype>,
    architecture: Option<&'static str>,
    layers: Option<Range<usize>>,
    tiling: Tiling,
    quantization: Option<Quantization>,
    part: Part,
    adapters: bool,
}

/// The quantization a load gives the projections of a checkpoint that stores them in BF16: E4M3
/// codes, restored to BF16 before each contraction, under FP32 scales of the largest magnitude
/// over 448, each value times its scale's reciprocal rounded to nearest even.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantization {
    /// One scale per projection, shared by the projections of one input that vLLM fuses (a layer's
    /// query, key and value; its gate and up): vLLM's `fp8_per_tensor`.
    Fp8PerTensor,
    /// One scale per output channel.
    Fp8PerChannel,
}

/// The implementation a checkpoint loads as, under the architecture it implements, the part of
/// the model it reads, the form of its weights and of its KV cache, the chips it loads onto, and
/// the tiling its calls attend with.
#[derive(Clone, Debug)]
pub struct Config {
    architecture: &'static str,
    implementation: &'static Implementation,
    part: Part,
    precision: Dtype,
    kv_cache: Dtype,
    chips: usize,
    tiling: Tiling,
}

impl LoadOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load onto `chips` chips instead of the one count the selected model allows; refuses zero.
    pub fn chips(self, chips: usize) -> Result<Self, LoadError> {
        let chips = NonZeroUsize::new(chips).ok_or_else(|| LoadError::InvalidData {
            name: "chips".into(),
            reason: "a model loads onto at least one chip".into(),
        })?;
        Ok(Self {
            chips: Some(chips),
            ..self
        })
    }

    /// Cache keys and values as `dtype` instead of BF16.
    pub fn kv_cache(self, dtype: Dtype) -> Self {
        Self {
            kv_cache: Some(dtype),
            ..self
        }
    }

    /// Load the architecture `name` instead of the one the checkpoint's config names; refuses a
    /// name the registry does not hold as invalid, and one it lists no implementation of as
    /// unsupported.
    pub fn architecture(self, name: &str) -> Result<Self, LoadError> {
        let (name, _) = Implementation::listed(&[name])?;
        Ok(Self {
            architecture: Some(name),
            ..self
        })
    }

    /// Load only the layers `layers`, at their positions in the whole model.
    pub fn layers(self, layers: Range<usize>) -> Self {
        Self {
            layers: Some(layers),
            ..self
        }
    }

    /// Serve calls whose rows keep their values' bits whatever other rows share the calls: their
    /// attention tiles each row's history by itself alone, `Tiling::Fixed`.
    pub fn batch_invariant(self) -> Self {
        Self {
            tiling: Tiling::Fixed,
            ..self
        }
    }

    /// Quantize a BF16 checkpoint's projections at load as `quantization` says.
    pub fn quantization(self, quantization: Quantization) -> Self {
        Self {
            quantization: Some(quantization),
            ..self
        }
    }

    /// Load the model's media encoder alone, whose rows its language model reads.
    pub fn encoder(self) -> Self {
        Self {
            part: Part::Encoder,
            ..self
        }
    }

    /// Load for calls that run LoRA adapters: an implementation whose projections take none is
    /// unsupported.
    pub fn adapters(self) -> Self {
        Self {
            adapters: true,
            ..self
        }
    }

    /// The implementation `checkpoint` loads as with these choices, from its config and its names
    /// alone: no weight is read and no device is needed.
    pub fn parse(&self, checkpoint: &Checkpoint) -> Result<Config, LoadError> {
        self.read(checkpoint, |checkpoint| self.select(checkpoint))
    }

    /// The implementation `checkpoint` loads as with these choices, loaded strictly into HBM of
    /// `device` as the model a framework serves: a decoder, or the media encoder of a load of the
    /// encoder alone; a refused choice is reported before any weight is read. A decoder serves its
    /// calls with the config's tiling. Dropping the model frees its device memory, and a new load
    /// of the same implementation serves the caches the dropped one did.
    pub fn load(&self, device: &mut Device, checkpoint: &Checkpoint) -> Result<Model, LoadError> {
        self.read(checkpoint, |checkpoint| {
            let Config {
                implementation,
                tiling,
                ..
            } = self.select(checkpoint)?;
            implementation.load(&mut Context::new(device), checkpoint, tiling)
        })
    }

    /// `visit` of `checkpoint` as the load reads it: for the part the options choose, quantized
    /// when they quantize it.
    fn read<R>(
        &self,
        checkpoint: &Checkpoint,
        visit: impl FnOnce(&Checkpoint) -> Result<R, LoadError>,
    ) -> Result<R, LoadError> {
        let encoder = (self.part == Part::Encoder).then(|| checkpoint.encoder());
        let checkpoint = encoder.as_ref().unwrap_or(checkpoint);
        let granularity = match self.quantization {
            Some(Quantization::Fp8PerTensor) => Granularity::Tensor,
            Some(Quantization::Fp8PerChannel) => Granularity::Channel,
            None => return visit(checkpoint),
        };
        visit(&checkpoint.quantized(granularity)?)
    }

    /// The implementation the load of `checkpoint`, as it reads it, selects.
    fn select(&self, checkpoint: &Checkpoint) -> Result<Config, LoadError> {
        let config = checkpoint.config();
        let names: Vec<&str> = match self.architecture {
            Some(name) => vec![name],
            None => (config["architectures"].as_array().into_iter().flatten())
                .chain([&config["_class_name"]])
                .filter_map(Value::as_str)
                .collect(),
        };
        if names.is_empty() {
            return Err(LoadError::InvalidData {
                name: "config.json:architectures".into(),
                reason: "the config names no architecture".into(),
            });
        }
        let (architecture, implementations) = Implementation::listed(&names)?;
        let implementation = Implementation::choose(architecture, implementations, checkpoint)?;
        let unsupported = |name: &str, reason: &str| LoadError::Unsupported {
            name: name.into(),
            reason: reason.into(),
        };
        if self.chips.is_some_and(|chips| chips.get() != 1) {
            return Err(unsupported(
                "chips",
                "every implementation runs on one chip",
            ));
        }
        if self.kv_cache.is_some_and(|dtype| dtype != Dtype::BF16) {
            return Err(unsupported(
                "kv_cache",
                "every implementation caches keys and values in BF16",
            ));
        }
        if self.layers.is_some() {
            return Err(unsupported("layers", "every implementation loads whole"));
        }
        if self.adapters && implementation.adapters() == 0 {
            return Err(LoadError::Unsupported {
                name: "adapters".into(),
                reason: format!(
                    "the {architecture} {} implementation takes no adapter",
                    implementation.size()
                ),
            });
        }
        Ok(Config {
            architecture,
            implementation,
            part: checkpoint.part(),
            // An implementation serving a quantization holds E4M3 projections: the block-FP8 one
            // serves a checkpoint declaring one, the channel-scaled ones a checkpoint a load
            // quantizes. A media encoder holds BF16 ones whatever its language model's.
            precision: match (
                checkpoint.part(),
                checkpoint.declaration(),
                checkpoint.quantization(),
            ) {
                (Part::Encoder, ..) | (Part::Whole, None, None) => Dtype::BF16,
                _ => Dtype::F8_E4M3,
            },
            kv_cache: Dtype::BF16,
            chips: 1,
            tiling: self.tiling,
        })
    }
}

impl Config {
    /// The architecture as the registry spells it.
    pub fn architecture(&self) -> &'static str {
        self.architecture
    }

    /// The implementation's size, as the hub spells it.
    pub fn size(&self) -> &'static str {
        self.implementation.size()
    }

    /// The class of the implementation's type.
    pub fn class(&self) -> furiosa_opt_models::models::Class {
        self.implementation.class()
    }

    /// The part of the model the implementation reads.
    pub fn part(&self) -> Part {
        self.part
    }

    /// The form of the implementation's projection weights.
    pub fn precision(&self) -> Dtype {
        self.precision
    }

    /// The form its KV cache stores keys and values in.
    pub fn kv_cache(&self) -> Dtype {
        self.kv_cache
    }

    pub fn chips(&self) -> usize {
        self.chips
    }

    /// How the implementation's calls tile each row's attention.
    pub fn tiling(&self) -> Tiling {
        self.tiling
    }
}

impl fmt::Display for Quantization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Fp8PerTensor => "fp8_per_tensor",
            Self::Fp8PerChannel => "fp8_per_channel",
        })
    }
}

impl FromStr for Quantization {
    type Err = LoadError;

    fn from_str(name: &str) -> Result<Self, LoadError> {
        [Self::Fp8PerTensor, Self::Fp8PerChannel]
            .into_iter()
            .find(|quantization| quantization.to_string() == name)
            .ok_or_else(|| LoadError::InvalidData {
                name: name.into(),
                reason: "a quantization is fp8_per_tensor or fp8_per_channel".into(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::LoadOptions;
    use furiosa_opt_models::load::{Checkpoint, LoadError};
    use serde_json::json;

    /// A config's architecture is the first of its `architectures`, then its `_class_name`, that the
    /// registry lists, refused or not, past any unknown name; names it lists none of are invalid.
    #[test]
    fn selects_the_first_listed_architecture() {
        let empty = tempfile::tempdir().unwrap();
        let refusal = |config| {
            let checkpoint = Checkpoint::new(empty.path(), Some(config)).unwrap();
            match LoadOptions::new().parse(&checkpoint) {
                Err(LoadError::InvalidData { name, .. }) => ("invalid", name),
                Err(LoadError::Unsupported { name, .. }) => ("unsupported", name),
                other => panic!("{:?}", other.map(|chosen| chosen.architecture())),
            }
        };
        let unknown = "UnknownForCausalLM";
        for (config, (kind, name)) in [
            // Every Qwen3 implementation refuses a config without dimensions, under the chosen name.
            (
                json!({"architectures": [unknown, "Qwen3ForCausalLM"]}),
                ("invalid", "Qwen3ForCausalLM"),
            ),
            (
                json!({"architectures": [unknown, "ApertusForCausalLM", "Qwen3ForCausalLM"]}),
                ("unsupported", "ApertusForCausalLM"),
            ),
            (
                json!({"architectures": [unknown], "_class_name": "FluxPipeline"}),
                ("unsupported", "FluxPipeline"),
            ),
            (
                json!({"architectures": [unknown, "UnknownModel"]}),
                ("invalid", "UnknownForCausalLM, UnknownModel"),
            ),
        ] {
            assert_eq!(refusal(config), (kind, name.to_owned()));
        }
    }
}
