//! The language backbone: embed the input, then run the body over the embedded rows.

use crate::Module;
use crate::load::Load;
use crate::modules::generic::schedule::{IntoStream, Stream};
use furiosa_opt_std::prelude::*;

/// Embeds the input, then passes the hidden values to the body; the body is its checkpoint's
/// `layers`, and the embedding names itself.
#[derive(Clone, DeviceSend, Load)]
pub struct Backbone<Embedding, Body> {
    #[load(rename = "")]
    pub embedding: Embedding,
    #[load(rename = "layers")]
    pub body: Body,
}

/// Supplied hidden rows are the complete embedding subtree's output, including its scale.
#[derive(DeviceSend)]
pub(crate) struct Embedded<X> {
    pub(crate) hidden: X,
}

/// The state holds the embedding's slot, then the body's; a borrowed input embeds by reference.
impl<'s, 'x, I, X, Y, EmbeddingState, BodyState, Embedding, Body, Env: Copy>
    Module<&'x I, Env, &'s mut (EmbeddingState, BodyState)> for Backbone<Embedding, Body>
where
    Embedding: Module<&'x I, Env, &'s mut EmbeddingState, Output = X>,
    Body: Module<X, Env, &'s mut BodyState, Output = Y>,
{
    type Output = Y;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (embedding, body): &'s mut (EmbeddingState, BodyState),
        input: &'x I,
    ) -> Y {
        let hidden = self.embedding.forward(&mut *device, env, embedding, input);
        self.body.forward(device, env, body, hidden)
    }
}

/// An input pair, such as tokens with their live rows or images, embeds as one input.
impl<'s, K, I, X, Y, EmbeddingState, BodyState, Embedding, Body, Env: Copy>
    Module<(K, I), Env, &'s mut (EmbeddingState, BodyState)> for Backbone<Embedding, Body>
where
    Embedding: Module<(K, I), Env, &'s mut EmbeddingState, Output = X>,
    Body: Module<X, Env, &'s mut BodyState, Output = Y>,
{
    type Output = Y;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (embedding, body): &'s mut (EmbeddingState, BodyState),
        input: (K, I),
    ) -> Y {
        let hidden = self.embedding.forward(&mut *device, env, embedding, input);
        self.body.forward(device, env, body, hidden)
    }
}

/// The body's stream over the embedding's, which streams over the backbone's upstream.
impl<'s, U: Stream, Env: Copy, EmbeddingState, BodyState, Embedding, Body>
    IntoStream<U, Env, &'s mut (EmbeddingState, BodyState)> for Backbone<Embedding, Body>
where
    Embedding: IntoStream<U, Env, &'s mut EmbeddingState>,
    Body: IntoStream<Embedding::Stream, Env, &'s mut BodyState>,
{
    type Stream = Body::Stream;

    fn into_stream(
        self,
        env: Env,
        (embedding, body): &'s mut (EmbeddingState, BodyState),
        upstream: U,
    ) -> Self::Stream {
        let embedded = self.embedding.into_stream(env, embedding, upstream);
        self.body.into_stream(env, body, embedded)
    }
}

/// A borrowed backbone streams as the backbone of its borrowed parts.
impl<'m, 's, U: Stream, Env, EmbeddingState, BodyState, Embedding, Body>
    IntoStream<U, Env, &'s mut (EmbeddingState, BodyState)> for &'m Backbone<Embedding, Body>
where
    Backbone<&'m Embedding, &'m Body>: IntoStream<U, Env, &'s mut (EmbeddingState, BodyState)>,
{
    type Stream = <Backbone<&'m Embedding, &'m Body> as IntoStream<
        U,
        Env,
        &'s mut (EmbeddingState, BodyState),
    >>::Stream;

    fn into_stream(
        self,
        env: Env,
        state: &'s mut (EmbeddingState, BodyState),
        upstream: U,
    ) -> Self::Stream {
        let Backbone { embedding, body } = self;
        Backbone { embedding, body }.into_stream(env, state, upstream)
    }
}

/// Supplied rows skip the embedding; its slot is unused.
impl<'s, X, Y, EmbeddingState, BodyState, Embedding, Body, Env: Copy>
    Module<Embedded<X>, Env, &'s mut (EmbeddingState, BodyState)> for Backbone<Embedding, Body>
where
    Body: Module<X, Env, &'s mut BodyState, Output = Y>,
{
    type Output = Y;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (_, body): &'s mut (EmbeddingState, BodyState),
        Embedded { hidden }: Embedded<X>,
    ) -> Y {
        self.body.forward(device, env, body, hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use crate::{Layers, Rolled, Then, modules::language::Normalized};

    struct Id(usize);
    struct Hidden(i32);
    struct Embedding([i32; 2]);

    struct Scale(i32);
    struct Layer(i32);
    struct Norm(i32);

    impl<Env> Module<&Id, Env, &mut ()> for Embedding {
        type Output = Hidden;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), id: &Id) -> Hidden {
            Hidden(self.0[id.0])
        }
    }

    impl<Env> Module<&Hidden, Env, &mut ()> for Scale {
        type Output = Hidden;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), hidden: &Hidden) -> Hidden {
            Hidden(hidden.0 * self.0)
        }
    }

    impl<Env> Module<Hidden, Env, &mut i32> for Layer {
        type Output = Hidden;

        fn forward(&self, _: &mut Device, _: Env, effects: &mut i32, hidden: Hidden) -> Hidden {
            *effects += 1;
            Hidden(hidden.0 + self.0)
        }
    }

    impl<Env> Module<&Hidden, Env, &mut ()> for Norm {
        type Output = Hidden;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), hidden: &Hidden) -> Hidden {
            Hidden(hidden.0 * self.0)
        }
    }

    /// An ID and the post-scale row its embedding makes enter one normalized catalog and leave it
    /// as the same row, each pass through the body's layers recording its effect.
    #[test]
    fn accepts_ids_and_embedded_rows() {
        let _lease = DEVICE.blocking_lock();
        let device =
            &mut Device::new(furiosa_opt_std::prelude::Topology { chips: 1, pes: 8 }).unwrap();
        let model = Normalized::new(
            Backbone {
                embedding: Then(Embedding([7, 13]), Scale(3)),
                body: Layers {
                    layers: [Layer(5), Layer(11)],
                    kind: Rolled,
                },
            },
            Norm(2),
        );
        let mut state = ((((), ()), [0, 0]), ());
        let Hidden(ids) = model.forward(&mut *device, (), &mut state, &Id(0));
        let Hidden(embedded) =
            model.forward(device, (), &mut state, Embedded { hidden: Hidden(21) });
        assert_eq!(
            (ids, embedded),
            ((7 * 3 + 5 + 11) * 2, (7 * 3 + 5 + 11) * 2)
        );
        assert_eq!(state.0.1, [2, 2]);
    }
}
