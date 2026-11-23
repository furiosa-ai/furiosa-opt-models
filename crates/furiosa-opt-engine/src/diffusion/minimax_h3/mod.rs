//! H3 execution on the pipeline's original four-chip device context.

use furiosa_opt_models::load::{Checkpoint, LoadError, Part};
use furiosa_opt_models::models::diffusion::minimax_h3::Inputs;
use furiosa_opt_models::models::{Class, Implements};
use furiosa_opt_models::ops::attention::full::paged::Tiling;
use safetensors::Dtype;

mod audio;
mod cache;
mod condition;
mod denoiser;
mod flight;
mod kernel;
mod layout;
mod memory;
mod pipeline;
mod request;
mod runtime;
pub(crate) mod text;
mod video;
mod weights;

pub use runtime::{
    AudioOutput, Decoded, Encoded, HostOutput, Input, Output, Pipeline, Prepared, PreparedAudio,
    PreparedStep, PreparedText, PreparedVideo, VideoDecoding,
};

impl Implements for Pipeline {
    const SIZE: &'static str = "FL2VA";
    const CLASS: Class = Pipeline::CLASS;

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        if source.config()["_class_name"].as_str() != Some("MiniMaxH3Pipeline") {
            return Err(LoadError::InvalidData {
                name: "config.json:_class_name".into(),
                reason: "expected MiniMaxH3Pipeline".into(),
            });
        }
        Ok("")
    }
}

impl crate::registry::Pipeline for Pipeline {
    const CHIPS: usize = Pipeline::CHIPS;
    type Inputs = Inputs;
    type Prepared = Prepared;

    fn reads(part: Part) -> Result<(), LoadError> {
        match part {
            Part::Whole => Ok(()),
            Part::Encoder => Err(LoadError::Unsupported {
                name: "encoder".into(),
                reason: "H3 loads as one complete pipeline".into(),
            }),
        }
    }

    fn options(
        chips: Option<usize>,
        kv_cache: Option<Dtype>,
        quantization: Option<crate::Quantization>,
        tiling: Tiling,
    ) -> Result<(), LoadError> {
        let unsupported = |name: &str, reason: &str| LoadError::Unsupported {
            name: name.into(),
            reason: reason.into(),
        };
        if chips.is_some_and(|chosen| chosen != Pipeline::CHIPS) {
            return Err(unsupported("chips", "H3 runs on four chips"));
        }
        if kv_cache.is_some() {
            return Err(unsupported("kv_cache", "H3 has no KV cache"));
        }
        if quantization.is_some() {
            return Err(unsupported(
                "quantization",
                "H3 loads its checkpoint precision",
            ));
        }
        if tiling == Tiling::Fixed {
            return Err(unsupported(
                "batch_invariant",
                "H3 uses its original attention tiling",
            ));
        }
        Ok(())
    }

    fn inputs(source: &Checkpoint) -> Result<Inputs, LoadError> {
        Inputs::read(source)
    }

    fn prepare(source: &Checkpoint, inputs: Inputs) -> Result<Prepared, LoadError> {
        Prepared::from_inputs(source, inputs)
    }

    fn check_placement(placement: &furiosa_opt_std::Placement) -> Result<(), LoadError> {
        if placement.chips.len() != Pipeline::CHIPS || placement.pes != (0..8) {
            return Err(LoadError::Unsupported {
                name: "placement".into(),
                reason: "H3 requires four chips with all eight PEs per chip".into(),
            });
        }
        Ok(())
    }
}
