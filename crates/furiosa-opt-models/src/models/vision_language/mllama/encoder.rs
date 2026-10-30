//! Populate each cross-attention cache from one independent image chunk.
use super::{text, vision};
use crate::Module;
use crate::modules::multimodal::EncoderDecoder;
use furiosa_opt_std::prelude::*;

/// Image encoder paired with the text body whose cross-attention projections it uses.
pub type Model = EncoderDecoder<vision::Model, text::Body>;

/// Borrowed image encoder and text projections sharing the caller's model weights.
#[derive(DeviceSend)]
pub struct Encoder<'a> {
    /// Image feature encoder.
    pub vision: &'a vision::Model,
    /// Text body supplying each cross-attention key/value projection.
    pub text: &'a text::Body,
}

impl<Env: Copy> Module<(&mut Input, &mut [text::Storage; 8]), Env, &mut ()> for Encoder<'_> {
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (input, memory): (&mut Input, &mut [text::Storage; 8]),
    ) {
        let hidden = self
            .vision
            .forward(&mut *device, env, &mut (), &mut input.vision);
        #[unroll]
        #[expect(
            clippy::needless_range_loop,
            reason = "device unrolling requires index ranges"
        )]
        for index in 0..8 {
            let cross = &self.text.first().body.layers.1[index].0.1.0;
            let mut view = (&mut memory[index], &input.writes[index]);
            cross.cache.forward(
                &mut *device,
                env,
                &mut (&mut view, Default::default()),
                &hidden,
            );
        }
    }
}

/// One image chunk and destinations for its projected cross-attention keys/values.
#[derive(DeviceSend)]
pub struct Input {
    /// Tiled image samples and writable vision captures.
    pub vision: vision::Input,
    /// Per-layer key/value byte offsets and write counts.
    pub writes: [text::Write<vision::Source>; 8],
}
