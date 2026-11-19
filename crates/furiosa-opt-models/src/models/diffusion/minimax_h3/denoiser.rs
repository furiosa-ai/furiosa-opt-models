//! The denoiser's single checkpoint-named owner, with separately callable execution stages.
//!
//! Condition projection and the two-layer refiner run once per request. Media/timestep
//! projections, the 50 joint blocks and final heads run each denoising step. Mutable request
//! metadata, refined text and scratch remain outside this tree. The parent pipeline supplies
//! the transformer's physical checkpoint prefix; this owner consumes its whole weight subtree.

use super::{embeddings, transformer};
use crate::load::Load;
use crate::modules::diffusion::WithSpans;
use crate::tensor::memory::{Hbm, residency::Residency};
use crate::{Layers, Rolled};
use furiosa_opt_std::prelude::DeviceSend;

pub mod stages;

/// The default four-chip weight tree, retaining each joint block as its own parameter owner.
pub type Denoiser<X = Hbm> = Model<
    Embeddings<X>,
    transformer::Refiner<embeddings::Width, transformer::Attended, transformer::Expanded, X>,
    Layers<[Block<X>; DEPTH], Rolled>,
    Heads<X>,
>;

/// The same checkpoint-owned projections, also lent to a separately launched embedding stage.
pub type Embeddings<X = Hbm> = embeddings::Embeddings<
    embeddings::Text<embeddings::Condition, embeddings::Width, embeddings::Chips, X>,
    embeddings::Input<embeddings::Patch, embeddings::Width, embeddings::Chips, X>,
    embeddings::Input<embeddings::Latent, embeddings::Width, embeddings::Chips, X>,
    embeddings::Timesteps<embeddings::Width, embeddings::Time, embeddings::Chips, X>,
>;

/// One ordinary adaptive block; host sequencing borrows each layer from the retained stack.
pub type Block<X = Hbm> = WithSpans<
    transformer::Block<
        embeddings::Width,
        transformer::Attended,
        transformer::Expanded,
        embeddings::Time,
        X,
    >,
>;

/// The same final projections, also lent to a separately launched readout stage.
pub type Heads<X = Hbm> = embeddings::FinalLayer<
    embeddings::VideoHead<
        embeddings::Width,
        embeddings::Time,
        embeddings::Patch,
        embeddings::Chips,
        X,
    >,
    embeddings::AudioHead<
        embeddings::Width,
        embeddings::Time,
        embeddings::Latent,
        embeddings::Chips,
        X,
    >,
>;

/// Owning composition of the stages; all weight names stay relative to the transformer root.
/// Callers borrow the children for separately launched embeddings, blocks and heads,
/// including TeaCache's first-block probe, from this one weight tree.
#[derive(DeviceSend, Load, Residency)]
#[residency(children(Embeddings, Refiner, Blocks, Heads))]
pub struct Model<Embeddings, Refiner, Blocks, Heads> {
    #[load(rename = "")]
    pub embeddings: Embeddings,
    #[load(rename = "token_refiner")]
    pub refiner: Refiner,
    #[load(rename = "blocks")]
    pub blocks: Blocks,
    #[load(rename = "")]
    pub heads: Heads,
}

/// Unit child slots of one Live adaptive block; request tensors and metadata live separately.
pub type BlockState = ((), (((), (), ()), ()), ((), ((), ())));

pub const DEPTH: usize = 50;
