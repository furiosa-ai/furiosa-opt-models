//! Diffusion pipelines: encode a condition, denoise prepared noise along a trajectory, decode.

use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Encodes the condition, denoises prepared noise along a trajectory and decodes the result.
#[derive(Clone, DeviceSend)]
pub struct Pipeline<Encode, Prepare, Denoise, Decode> {
    pub encode: Encode,
    pub prepare: Prepare,
    pub denoise: Denoise,
    pub decode: Decode,
}

/// The state holds the encoder's, the preparation's, the trajectory's and the decoder's slots; the
/// decoder reads the denoised latent.
impl<
    's,
    I,
    N,
    C,
    X,
    Y,
    O,
    J,
    EncodeState,
    PrepareState,
    DenoiseState,
    DecodeState,
    Encode,
    Prepare,
    Denoise,
    Decode,
    Env: Copy,
> Module<(I, (N, J)), Env, &'s mut (EncodeState, PrepareState, DenoiseState, DecodeState)>
    for Pipeline<Encode, Prepare, Denoise, Decode>
where
    Encode: Module<I, Env, &'s mut EncodeState, Output = C>,
    Prepare: for<'c> Module<(N, &'c C), Env, &'s mut PrepareState, Output = X>,
    Denoise: for<'c> Module<(X, (&'c C, J)), Env, &'s mut DenoiseState, Output = Y>,
    Decode: for<'y> Module<&'y Y, Env, &'s mut DecodeState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (encode, prepare, denoise, decode): &'s mut (
            EncodeState,
            PrepareState,
            DenoiseState,
            DecodeState,
        ),
        (input, (noise, steps)): (I, (N, J)),
    ) -> Self::Output {
        let condition = self.encode.forward(&mut *device, env, encode, input);
        let latent = self
            .prepare
            .forward(&mut *device, env, prepare, (noise, &condition));
        let latent =
            self.denoise
                .forward(&mut *device, env, denoise, (latent, (&condition, steps)));
        self.decode.forward(device, env, decode, &latent)
    }
}

/// Denoises chunks in turn, each conditioned on the history the earlier chunks committed.
#[derive(DeviceSend)]
pub struct Chunked<Condition, Prepare, Denoise, Commit> {
    pub condition: Condition,
    pub prepare: Prepare,
    pub denoise: Denoise,
    pub commit: Commit,
}

/// The state holds the committed history, then the condition's, the preparation's and the
/// trajectory's slots, which every chunk takes again; each chunk's operands are the input's.
impl<
    H,
    C,
    N,
    J,
    K,
    E,
    X,
    Y,
    ConditionState,
    PrepareState,
    DenoiseState,
    Condition,
    Prepare,
    Denoise,
    Commit,
    const COUNT: usize,
    Env: Copy,
> Module<[(C, N, J, K); COUNT], Env, &mut (H, ConditionState, PrepareState, DenoiseState)>
    for Chunked<Condition, Prepare, Denoise, Commit>
where
    Condition: for<'h, 's> Module<(&'h H, C), Env, &'s mut ConditionState, Output = E>,
    Prepare: for<'e, 's> Module<(N, &'e E), Env, &'s mut PrepareState, Output = X>,
    Denoise: for<'e, 's> Module<(X, (&'e E, J)), Env, &'s mut DenoiseState, Output = Y>,
    Commit: for<'d, 'h, 'y> Module<(&'h mut H, &'y Y, K), Env, &'d mut (), Output = ()>,
{
    type Output = [Y; COUNT];

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (history, condition, prepare, denoise): &mut (
            H,
            ConditionState,
            PrepareState,
            DenoiseState,
        ),
        chunks: [(C, N, J, K); COUNT],
    ) -> Self::Output {
        chunks.map(|(context, noise, steps, commit)| {
            let encoded =
                self.condition
                    .forward(&mut *device, env, &mut *condition, (&*history, context));
            let latent = self
                .prepare
                .forward(&mut *device, env, &mut *prepare, (noise, &encoded));
            let output = self.denoise.forward(
                &mut *device,
                env,
                &mut *denoise,
                (latent, (&encoded, steps)),
            );
            self.commit
                .forward(&mut *device, env, &mut (), (&mut *history, &output, commit));
            output
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::diffusion::{Denoiser, Guider, Sampler};
    use crate::tests::DEVICE;
    use crate::{First, Identity, Layers, Unrolled};
    use furiosa_opt_std::prelude::Topology;

    struct Latent {
        value: f64,
        previous: Option<f64>,
    }

    struct Embed;

    impl<Env> Module<(&Latent, &f64), Env, &mut ()> for Embed {
        type Output = [f64; 2];

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            (latent, condition): (&Latent, &f64),
        ) -> Self::Output {
            [latent.value, *condition]
        }
    }

    struct Time;

    impl<Env> Module<(f64, &f64), Env, &mut ()> for Time {
        type Output = f64;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), (time, _): (f64, &f64)) -> f64 {
            time
        }
    }

    struct Layer;

    /// Reads its condition from the env and its prefix from its slot.
    impl Module<[f64; 2], (&f64, ()), &mut f64> for Layer {
        type Output = [f64; 2];

        fn forward(
            &self,
            _: &mut Device,
            (time, ()): (&f64, ()),
            prefix: &mut f64,
            hidden: [f64; 2],
        ) -> Self::Output {
            [hidden[0] + hidden[1] + time + *prefix, hidden[1]]
        }
    }

    struct Head;

    impl<Env> Module<(&[f64; 2], &f64), Env, &mut ()> for Head {
        type Output = f64;

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            (hidden, _): (&[f64; 2], &f64),
        ) -> f64 {
            hidden[0]
        }
    }

    /// The denoiser under a trajectory's prediction contract: a branch's time conditions it, and
    /// the branch's slot is the denoiser's, each layer's prefix in its own slot.
    struct Prediction(Denoiser<Embed, Time, [Layer; 1], Head>);

    type Slot = ((), (), [f64; 1], ());

    impl<Env> Module<(&Latent, (&f64, (f64, ()))), Env, &mut Slot> for Prediction {
        type Output = f64;

        fn forward(
            &self,
            device: &mut Device,
            _: Env,
            state: &mut Slot,
            (latent, (condition, (time, ()))): (&Latent, (&f64, (f64, ()))),
        ) -> f64 {
            self.0
                .forward(device, ((), ()), state, (latent, condition, time))
        }
    }

    struct Combine;

    impl<Env> Module<([f64; 2], f64), Env, &mut ()> for Combine {
        type Output = f64;

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            (predictions, scale): ([f64; 2], f64),
        ) -> f64 {
            predictions[0] + scale * (predictions[1] - predictions[0])
        }
    }

    struct Update;

    impl<Env> Module<(Latent, f64, f64), Env, &mut ()> for Update {
        type Output = Latent;

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            (latent, prediction, delta): (Latent, f64, f64),
        ) -> Latent {
            let derivative = latent
                .previous
                .map_or(prediction, |previous| 1.5 * prediction - 0.5 * previous);
            Latent {
                value: latent.value - delta * derivative,
                previous: Some(prediction),
            }
        }
    }

    struct Decode;

    /// The decoder's history is its slot.
    impl<Env> Module<&Latent, Env, &mut f64> for Decode {
        type Output = ([f64; 2], Option<f64>);

        fn forward(&self, _: &mut Device, _: Env, tail: &mut f64, latent: &Latent) -> Self::Output {
            let waveform = [*tail, latent.value];
            *tail = latent.value;
            (waveform, latent.previous)
        }
    }

    struct Condition;

    impl<Env> Module<(&i32, ()), Env, &mut ()> for Condition {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            (prefix, _): (&i32, ()),
        ) -> Self::Output {
            *prefix
        }
    }

    struct Predict;

    impl<Env> Module<(&i32, (&i32, ())), Env, &mut ()> for Predict {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            (latent, (prefix, _)): (&i32, (&i32, ())),
        ) -> Self::Output {
            latent + prefix
        }
    }

    struct Advance;

    impl<Env> Module<(i32, i32, ()), Env, &mut ()> for Advance {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            (_, prediction, _): (i32, i32, ()),
        ) -> i32 {
            prediction
        }
    }

    struct Append;

    impl<Env> Module<(&mut i32, &i32, ()), Env, &mut ()> for Append {
        type Output = ();

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            (prefix, output, _): (&mut i32, &i32, ()),
        ) {
            *prefix += output;
        }
    }

    /// Guidance reads independent prefixes while solver and decoder histories evolve separately.
    #[test]
    fn preserves_independent_histories() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let model = Pipeline {
            encode: Identity,
            prepare: First,
            denoise: Sampler {
                predict: Guider {
                    predict: Prediction(Denoiser {
                        embedding: Embed,
                        conditioning: Time,
                        body: Layers {
                            layers: [Layer],
                            kind: Unrolled,
                        },
                        head: Head,
                    }),
                    combine: Combine,
                },
                update: Update,
            },
            decode: Decode,
        };
        let branches = [((), (), [2.0], ()), ((), (), [5.0], ())];
        for tail in [0.0, 100.0] {
            let mut state = ((), (), (branches, ()), tail);
            let steps = (0..2).map(|time| {
                let time = f64::from(time);
                (([(time, ()), (time, ())], 2.0), 0.1)
            });
            let (waveform, history) = model.forward(
                device,
                (),
                &mut state,
                (
                    [1.0, 3.0],
                    (
                        Latent {
                            value: 10.0,
                            previous: None,
                        },
                        steps,
                    ),
                ),
            );
            let ((), (), (slots, ()), last) = state;
            assert_eq!(slots, branches);
            assert_eq!(waveform[0], tail);
            assert!((waveform[1] - 5.595_f64).abs() <= 1e-9);
            assert!((history.unwrap() - 21.7_f64).abs() <= 1e-9);
            assert!((last - 5.595_f64).abs() <= 1e-9);
        }
    }

    /// Prefix history receives completed chunks, never intermediate numerical predictions.
    #[test]
    fn commits_completed_chunks() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let model = Chunked {
            condition: Condition,
            prepare: First,
            denoise: Sampler {
                predict: Predict,
                update: Advance,
            },
            commit: Append,
        };
        let chunks = [1, 1].map(|noise| ((), noise, [((), ()), ((), ())], ()));
        let mut state = (2, (), (), ((), ()));
        let outputs = model.forward(device, (), &mut state, chunks);
        assert_eq!((state.0, outputs), (22, [5, 15]));
    }
}
