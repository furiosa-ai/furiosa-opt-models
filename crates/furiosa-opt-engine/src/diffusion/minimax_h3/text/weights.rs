//! Bind the checked pipeline documents to the public encoder tree without a second tensor owner.

use super::{Cells, DEFAULT_MAX_TEXT, P, text};
use furiosa_opt_models::load::{Checkpoint, Context, Load, LoadError};
use furiosa_opt_models::models::decoder::attention::Head;
use furiosa_opt_models::models::diffusion::minimax_h3::Inputs;
use furiosa_opt_models::models::diffusion::minimax_h3::config::TextEncoder;
use furiosa_opt_models::ops::rotary::PairFrequencies;

pub(crate) struct Weights {
    pub(super) body: text::Body<P, Cells>,
    pub(super) positions: text::position::Positions<Cells>,
    pub(super) capacity: usize,
}

impl Weights {
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Called at the pipeline root. The component scope remains strict even when used alone;
    /// in a full traversal its exact exclusions participate in the parent's strict accounting.
    pub(crate) fn load(
        context: &mut Context<'_>,
        source: &Checkpoint,
        inputs: &Inputs,
        capacity: usize,
    ) -> Result<Self, LoadError> {
        let config = inputs.text_encoder();
        if !(1..=DEFAULT_MAX_TEXT).contains(&capacity)
            || config
                .positions()
                .is_some_and(|positions| positions.get() < capacity)
        {
            return Err(LoadError::Unsupported {
                name: "text_encoder.capacity".into(),
                reason: "text capacity must be in 1..=2048 and fit the checkpoint's position limit"
                    .into(),
            });
        }
        let component = inputs.layout().text_encoder();
        let directory = component
            .directory()
            .to_str()
            .expect("checked JSON component name");
        // The checked text schema accepts the public merged config's nested and flat forms.
        // Tensor names continue to use the physical prefix, including an optional source_path.
        let key = if source.config()[directory].get("text_config").is_some() {
            format!("{directory}.text_config")
        } else {
            directory.to_owned()
        };
        let body = context.config(&key, |cx| {
            cx.field("text_encoder", component.tensor_prefix(), |cx| {
                cx.checkpoint(source, |cx| {
                    for prefix in [
                        "model.visual",
                        "visual",
                        "lm_head",
                        "model.language_model.norm",
                    ] {
                        cx.ignore(prefix);
                    }
                    for at in text::LAYERS..config.checkpoint_layers() {
                        cx.ignore(&format!("model.language_model.layers.{at}"));
                    }
                    cx.field("body", "model.language_model", |cx| Load::load(cx, source))
                })
            })
        })?;
        // The checked schema fixes the rotary base; this uses the ordinary shared generator.
        let frequencies = PairFrequencies::<Head>::unscaled::<Head>(TextEncoder::ROPE_THETA)
            .map_err(|error| LoadError::InvalidData {
                name: "text_encoder.rope_theta".into(),
                reason: error.to_string(),
            })?;
        let positions = context.field("text_positions", "", |cx| Load::load(cx, &frequencies))?;
        Ok(Self {
            body,
            positions,
            capacity,
        })
    }
}
