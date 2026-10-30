//! Encoder-decoder memory: cross layers write the encoded source once and read it on later calls.

use crate::modules::language::Backbone;
use crate::{Branch, Interleave, Layers, Module, Rolled, Sequential};
use furiosa_opt_std::prelude::{Device, DeviceSend};

impl<Embedding, Text, Cross, const TEXT: usize, const CROSS: usize, const ORDER: &'static [Branch]>
    Backbone<Embedding, Interleave<([Text; TEXT], [Cross; CROSS]), ORDER>>
{
    /// The backbone without its cross layers, for a call without encoder memory: the text layers
    /// run in their order and no cross layer's state is read or written.
    pub fn text(&self) -> Sequential<&Embedding, Layers<&[Text; TEXT], Rolled>> {
        Sequential(
            &self.embedding,
            Layers {
                layers: &self.body.layers.0,
                kind: Rolled,
            },
        )
    }
}

/// Encodes a source and runs the decoder while populating caller-owned encoder memory.
#[derive(Clone, DeviceSend)]
pub struct EncoderDecoder<Encoder, Decoder> {
    pub encoder: Encoder,
    pub decoder: Decoder,
}

/// The decoder reads the env beside the encoded source, which its cross layers store in their
/// memory slots; later calls of the decoder alone pass `()` for the source and read that memory.
impl<'s, I, J, M, EncoderState, DecoderState, O, Encoder, Decoder, Env: Copy>
    Module<(I, J), Env, &'s mut (EncoderState, DecoderState)> for EncoderDecoder<Encoder, Decoder>
where
    Encoder: Module<I, Env, &'s mut EncoderState, Output = M>,
    Decoder: for<'m> Module<J, (Env, &'m M), &'s mut DecoderState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (encoder, decoder): &'s mut (EncoderState, DecoderState),
        (input, tokens): (I, J),
    ) -> Self::Output {
        let hidden = self.encoder.forward(&mut *device, env, encoder, input);
        self.decoder
            .forward(device, (env, &hidden), decoder, tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use crate::{Identity, Then};
    use furiosa_opt_std::prelude::Topology;

    struct CrossLayer;

    struct Token;

    impl<Env> Module<&i32, Env, &mut ()> for Token {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), token: &i32) -> i32 {
            *token
        }
    }

    struct Score;

    impl<Env> Module<&i32, Env, &mut ()> for Score {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), hidden: &i32) -> i32 {
            hidden * 2
        }
    }

    impl<'e, Env> Module<i32, (Env, &'e i32), &mut (i32, i32)> for CrossLayer {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            (_, source): (Env, &'e i32),
            (state, cache): &mut (i32, i32),
            input: i32,
        ) -> i32 {
            *cache = *source;
            *state += input + *cache;
            *state
        }
    }

    impl<Env> Module<i32, (Env, ()), &mut (i32, i32)> for CrossLayer {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            _: (Env, ()),
            (state, cache): &mut (i32, i32),
            input: i32,
        ) -> i32 {
            *state += input + *cache;
            *state
        }
    }

    /// Encode writes memory once; decode reuses it without modifying the stored encoder values.
    #[test]
    fn reuses_encoder_memory() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let model = EncoderDecoder {
            encoder: Identity,
            decoder: Then(
                Backbone {
                    embedding: Token,
                    body: [CrossLayer, CrossLayer],
                },
                Score,
            ),
        };
        let mut state = ((), (((), [(0, 0), (0, 0)]), ()));
        let expected = model.forward(device, (), &mut state, (7, &3));
        let ((), ((_, [(_, first), (_, second)]), ())) = state;
        assert_eq!((first, second), (7, 7));
        let mut body = ((), [(0, first), (0, second)]);
        let read = ((), ());
        let hidden = model.decoder.body().forward(device, read, &mut body, &3);
        for _ in 0..2 {
            assert_eq!(
                model.decoder.head().forward(device, read, &mut (), &hidden),
                expected
            );
        }
        assert_eq!(body.1, [(10, 7), (17, 7)]);
    }
}
