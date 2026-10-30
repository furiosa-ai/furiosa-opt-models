//! Modality-routed generation: text or image input runs one shared body and that modality's head.

use crate::{Either, Module};
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Embeds text or image input, runs the shared body and applies that modality's head.
#[derive(DeviceSend)]
pub struct Generative<Text, Image, Body, TextHead, ImageHead> {
    pub text: Text,
    pub image: Image,
    pub body: Body,
    pub heads: (TextHead, ImageHead),
}

/// Either modality's input beside its head's operand, the env telling the body to write or read
/// its shared memory; the state holds each embedding's slot, the shared body's and each head's.
impl<
    's,
    T,
    I,
    X,
    Y,
    U,
    V,
    H,
    L,
    O,
    P,
    TextState,
    ImageState,
    BodyState,
    TextHeadState,
    ImageHeadState,
    Text,
    Image,
    Body,
    TextHead,
    ImageHead,
    Env: Copy,
>
    Module<
        Either<(T, H), (I, L)>,
        Env,
        &'s mut (
            TextState,
            ImageState,
            BodyState,
            (TextHeadState, ImageHeadState),
        ),
    > for Generative<Text, Image, Body, TextHead, ImageHead>
where
    Text: Module<T, Env, &'s mut TextState, Output = X>,
    Image: Module<I, Env, &'s mut ImageState, Output = Y>,
    Body: Module<X, Env, &'s mut BodyState, Output = U>
        + Module<Y, Env, &'s mut BodyState, Output = V>,
    TextHead: for<'x> Module<(&'x U, H), Env, &'s mut TextHeadState, Output = O>,
    ImageHead: for<'x> Module<(&'x V, L), Env, &'s mut ImageHeadState, Output = P>,
{
    type Output = Either<O, P>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (text, image, body, (text_head, image_head)): &'s mut (
            TextState,
            ImageState,
            BodyState,
            (TextHeadState, ImageHeadState),
        ),
        input: Either<(T, H), (I, L)>,
    ) -> Self::Output {
        match input {
            Either::Left((input, head)) => {
                let hidden = self.text.forward(&mut *device, env, text, input);
                let hidden = self.body.forward(&mut *device, env, body, hidden);
                Either::Left(
                    self.heads
                        .0
                        .forward(device, env, text_head, (&hidden, head)),
                )
            }
            Either::Right((input, head)) => {
                let hidden = self.image.forward(&mut *device, env, image, input);
                let hidden = self.body.forward(&mut *device, env, body, hidden);
                Either::Right(
                    self.heads
                        .1
                        .forward(device, env, image_head, (&hidden, head)),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use crate::{Either, Identity};
    use furiosa_opt_std::prelude::Topology;
    use std::iter;

    /// A body whose shared prefix a call that writes (env `&()`) updates and a call that reads
    /// (env `()`) leaves unchanged.
    struct Shared(f64);

    impl Module<f64, &(), &mut f64> for Shared {
        type Output = f64;

        fn forward(&self, _: &mut Device, _: &(), cache: &mut f64, input: f64) -> Self::Output {
            *cache += input * self.0;
            *cache
        }
    }

    impl Module<f64, (), &mut f64> for Shared {
        type Output = f64;

        fn forward(&self, _: &mut Device, (): (), cache: &mut f64, input: f64) -> Self::Output {
            *cache + input * self.0
        }
    }

    struct Text;

    impl<Env> Module<(&f64, ()), Env, &mut ()> for Text {
        type Output = f64;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), (hidden, _): (&f64, ())) -> f64 {
            *hidden
        }
    }

    struct Image;

    impl<Env> Module<(&f64, usize), Env, &mut ()> for Image {
        type Output = Vec<f64>;

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            (hidden, size): (&f64, usize),
        ) -> Self::Output {
            iter::repeat_n(*hidden, size).collect()
        }
    }

    /// Both modalities reach the body with its one shared slot: text writes the shared prefix and
    /// repeated image readouts read it unchanged.
    #[test]
    fn shares_the_body_slot() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let model = Generative {
            text: Identity,
            image: Identity,
            body: Shared(2.0),
            heads: (Text, Image),
        };
        let mut state = ((), (), 10.0, ((), ()));
        let text: Either<_, (f64, usize)> = Either::Left((3.0, ()));
        assert!(matches!(
            model.forward(device, &(), &mut state, text),
            Either::Left(16.0)
        ));
        for _ in 0..2 {
            let image: Either<(f64, ()), _> = Either::Right((3.0, 2));
            assert!(
                matches!(model.forward(device, (), &mut state, image), Either::Right(values) if values == [22.0, 22.0])
            );
        }
        assert_eq!(state.2, 16.0);
    }
}
