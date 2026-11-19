//! Retained weights for H3's resident denoiser and media decoders.
//!
//! Loading prepares host cells only. The pipeline places this tree on its existing device;
//! request metadata, decoder call contexts and streamed text weights have separate owners.

use crate::residency::{Cells, Resident};
use furiosa_opt_models::load::{Checkpoint, Children, Context, Load, LoadError};
use furiosa_opt_models::models::diffusion::minimax_h3::denoiser::Denoiser;
use furiosa_opt_models::models::diffusion::minimax_h3::embeddings::Chips;
use furiosa_opt_models::models::diffusion::minimax_h3::{Inputs, audio, video};
use furiosa_opt_models::tensor::memory::residency::Residency;
use std::sync::Arc;

/// One strict traversal prepares all pipeline weights without opening or placing a device.
/// The text controller consumes its prepared tree once; callable resident views share the owner.
pub(super) struct Prepared {
    pub(super) weights: Arc<Weights>,
    pub(super) text: super::text::Weights,
}

impl Prepared {
    pub(super) fn load(source: &Checkpoint, capacity: usize) -> Result<Self, LoadError> {
        let inputs = Inputs::read(source)?;
        Self::load_with_inputs(source, capacity, inputs)
    }

    /// Reuse the documents already checked by the load boundary; no second sidecar read.
    pub(super) fn load_with_inputs(
        source: &Checkpoint,
        capacity: usize,
        inputs: Inputs,
    ) -> Result<Self, LoadError> {
        Context::new(None).checkpoint(source, |cx| {
            // The text loader checks the capacity before preparing any weights. Keep this
            // ahead of the much larger resident tree so invalid capacities fail immediately.
            let text = super::text::Weights::load(cx, source, &inputs, capacity)?;
            let weights = Arc::new(Weights::load(cx, source, &inputs)?);
            Ok(Self { weights, text })
        })
    }
}

/// One retained tree and the checked documents used to resolve it. Lengths share these weights.
pub(super) struct Weights {
    inputs: Inputs,
    resident: Resident<Components>,
}

impl Weights {
    /// Load at the checkpoint root using documents already checked by the caller.
    ///
    /// Each physical component is strict when this helper is called alone. For a full pipeline,
    /// call this helper and the text loader inside one outer `Context::checkpoint` traversal:
    /// that traversal also rejects unclaimed tensors outside these three component prefixes.
    /// Text tensors are neither read nor excluded here. No placement or upload occurs here.
    pub(super) fn load(
        context: &mut Context<'_>,
        source: &Checkpoint,
        inputs: &Inputs,
    ) -> Result<Self, LoadError> {
        let layout = inputs.layout();
        let mut children = Children::default();
        let denoiser = children.ask(context.field(
            "denoiser",
            layout.transformer().tensor_prefix(),
            |cx| {
                // Request rotations are generated from the checked rope_theta configuration.
                // This stored inverse-frequency buffer is not a callable weight of the tree.
                // Account for this exact buffer without hiding any transformer parameters.
                cx.ignore("rope.inv_freq");
                Denoiser::<Cells>::load(cx, source)
            },
        ))?;
        let video =
            children.ask(
                context.field("video", layout.video_vae().tensor_prefix(), |cx| {
                    cx.checkpoint(source, |cx| {
                        // The preserved pipeline decodes only. Keep the public physical prefix,
                        // including source_path, and account for just the unused encoder weights.
                        cx.ignore("encoder");
                        cx.ignore("quant_conv");
                        // The video owner accounts for its own unused decoder.mask_token.
                        video::Decoder::<Chips, Cells>::load(cx, source)
                    })
                }),
            )?;
        let audio =
            children.ask(
                context.field("audio", layout.audio_vae().tensor_prefix(), |cx| {
                    cx.checkpoint(source, |cx| {
                        for prefix in ["encoder", "pre_block", "logs_proj", "mean_proj"] {
                            cx.ignore(prefix);
                        }
                        audio::Decoder::<Chips, Cells>::load(cx, source)
                    })
                }),
            )?;
        let ((denoiser, video), audio) = children.node(denoiser.zip(video).zip(audio))?;
        Ok(Self {
            // Retain sidecar statistics and schedule shifts without introducing a statistics
            // fold: the existing decoder loaders preserve the reference arithmetic boundary.
            inputs: inputs.clone(),
            resident: Resident::<Components>::new(Components {
                denoiser,
                video,
                audio,
            }),
        })
    }

    pub(super) fn inputs(&self) -> &Inputs {
        &self.inputs
    }

    /// The whole resident group for placement, admission or retirement through shared residency.
    /// Upload waits and request-buffer cleanup remain the pipeline caller's responsibility.
    pub(super) fn resident(&self) -> &Resident<Components> {
        &self.resident
    }

    pub(super) fn denoiser(self: &Arc<Self>) -> Component<Denoiser> {
        Component {
            owner: Arc::clone(self),
            resident: self
                .resident
                .part::<Denoiser>("denoiser", |tree| &tree.denoiser),
        }
    }

    pub(super) fn video(self: &Arc<Self>) -> Component<video::Decoder> {
        Component {
            owner: Arc::clone(self),
            resident: self
                .resident
                .part::<video::Decoder>("video", |tree| &tree.video),
        }
    }

    pub(super) fn audio(self: &Arc<Self>) -> Component<audio::Decoder> {
        Component {
            owner: Arc::clone(self),
            resident: self
                .resident
                .part::<audio::Decoder>("audio", |tree| &tree.audio),
        }
    }
}

/// Named children of the resident group; only their existing model trees define weight formats.
/// This tree's paths identify residency views, independently of physical checkpoint prefixes.
#[derive(Residency)]
#[residency(children(D, V, A))]
pub(super) struct Components<D = Denoiser, V = video::Decoder, A = audio::Decoder> {
    pub(super) denoiser: D,
    pub(super) video: V,
    pub(super) audio: A,
}

/// A callable subtree retaining all resident siblings and the checked inputs, without a reload.
/// A future externally exposed component must additionally retain the complete pipeline owner,
/// including its text controller and original device context.
pub(super) struct Component<T: Residency> {
    owner: Arc<Weights>,
    resident: Resident<T>,
}

impl<T: Residency> Component<T> {
    pub(super) fn resident(&self) -> &Resident<T> {
        &self.resident
    }
}

impl<T: Residency> Clone for Component<T> {
    fn clone(&self) -> Self {
        Self {
            owner: Arc::clone(&self.owner),
            resident: self.resident.clone(),
        }
    }
}
