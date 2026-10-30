//! Diffusion pipelines, numerical trajectories and condition-modulated transformer blocks.

pub use block::{Adaptive, Conditioned, Cross, Single, Spatial};
pub use coupled::Coupled;
pub use denoiser::{
    Control, Controlled, Denoiser, DualStream, Head, PackedEmbedding, PairedHead, Refined,
    Terminal, Time, Transformer,
};
pub use modulation::{
    AdaptiveNorm, Gates, Modulated, Modulation, ScaleFirst, ShiftFirst, SplitGate, SplitGates,
    SplitModulation,
};
pub use pipeline::{Chunked, Pipeline};
pub use self_conditioning::{ConditionedEmbedding, SelfConditioning};
pub use stream::{Dual, JointFinal, Stream};
pub use trajectory::{Cascade, Experts, Guider, Sampler, Step, Values, ddim, flow_match};

mod block;
pub mod coupled;
mod denoiser;
mod modulation;
mod pipeline;
mod self_conditioning;
mod stream;
mod trajectory;
