//! Numerical trajectories: solver steps over predictions, guidance, noise-regime experts and
//! cascades of trajectories.

use crate::{Either, Module};
use furiosa_opt_std::prelude::*;

/// FP32 values of a sample, a prediction or a schedule coefficient.
pub type Values<S> = HbmTensor<f32, m![1], S>;

/// Applies prediction and solver updates along a trajectory, with one shared condition.
#[derive(Clone, DeviceSend)]
pub struct Sampler<Predict, Update> {
    pub predict: Predict,
    pub update: Update,
}

/// The state holds the prediction's slot and the update's; every step takes them again, so they
/// hold no borrowed state. Each step's operands are the input's.
impl<'c, X, Y, C, P, U, J, PredictState, UpdateState, Predict, Update, Env: Copy>
    Module<(X, (&'c C, J)), Env, &mut (PredictState, UpdateState)> for Sampler<Predict, Update>
where
    J: IntoIterator<Item = (P, U)>,
    Predict: for<'x, 's> Module<(&'x X, (&'c C, P)), Env, &'s mut PredictState, Output = Y>,
    Update: for<'s> Module<(X, Y, U), Env, &'s mut UpdateState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &mut (PredictState, UpdateState),
        (latent, (condition, steps)): (X, (&'c C, J)),
    ) -> Self::Output {
        let step = Step {
            predict: &self.predict,
            update: &self.update,
        };
        steps.into_iter().fold(latent, |latent, operands| {
            step.forward(
                &mut *device,
                env,
                &mut *state,
                (latent, (condition, operands)),
            )
        })
    }
}

/// One numerical update; the caller supplies its time, conditioning and solver operands.
#[derive(DeviceSend)]
pub struct Step<Predict, Update> {
    pub predict: Predict,
    pub update: Update,
}

/// The state holds the prediction's slot and the update's.
impl<'s, 'c, X, Y, C, P, U, PredictState, UpdateState, Predict, Update, Env: Copy>
    Module<(X, (&'c C, (P, U))), Env, &'s mut (PredictState, UpdateState)> for Step<Predict, Update>
where
    Predict: for<'x> Module<(&'x X, (&'c C, P)), Env, &'s mut PredictState, Output = Y>,
    Update: Module<(X, Y, U), Env, &'s mut UpdateState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (prediction, update): &'s mut (PredictState, UpdateState),
        (latent, (condition, (predict, operands))): (X, (&'c C, (P, U))),
    ) -> Self::Output {
        let predicted = self.predict.forward(
            &mut *device,
            env,
            prediction,
            (&latent, (condition, predict)),
        );
        self.update
            .forward(device, env, update, (latent, predicted, operands))
    }
}

/// Combines the predictions of several independently conditioned branches.
#[derive(Clone, DeviceSend)]
pub struct Guider<Predict, Combine> {
    pub predict: Predict,
    pub combine: Combine,
}

/// The state is one prediction slot a branch.
impl<'c, X, Y, O, C, B, G, PredictState, Predict, Combine, const N: usize, Env: Copy>
    Module<(&X, (&'c [C; N], ([B; N], G))), Env, &mut [PredictState; N]>
    for Guider<Predict, Combine>
where
    Predict: for<'x, 's> Module<(&'x X, (&'c C, B)), Env, &'s mut PredictState, Output = Y>,
    Combine: for<'d> Module<([Y; N], G), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &mut [PredictState; N],
        (latent, (conditions, (branches, guidance))): (&X, (&'c [C; N], ([B; N], G))),
    ) -> O {
        const {
            assert!(N > 0, "guidance requires a prediction");
        }
        let mut index = 0;
        let predictions = branches.map(|branch| {
            let (condition, slot) = (&conditions[index], &mut state[index]);
            index += 1;
            self.predict
                .forward(&mut *device, env, slot, (latent, (condition, branch)))
        });
        self.combine
            .forward(device, env, &mut (), (predictions, guidance))
    }
}

/// Applies the high- or low-noise expert that the input's regime selects.
#[derive(DeviceSend)]
pub struct Experts<High, Low> {
    pub high: High,
    pub low: Low,
}

/// The state holds the high expert's slot and the low expert's; the selected one is used.
impl<'s, 'c, X, C, H, L, O, HighState, LowState, High, Low, Env: Copy>
    Module<(&X, (&'c C, Either<H, L>)), Env, &'s mut (HighState, LowState)> for Experts<High, Low>
where
    High: for<'x> Module<(&'x X, (&'c C, H)), Env, &'s mut HighState, Output = O>,
    Low: for<'x> Module<(&'x X, (&'c C, L)), Env, &'s mut LowState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (high, low): &'s mut (HighState, LowState),
        (latent, (condition, regime)): (&X, (&'c C, Either<H, L>)),
    ) -> Self::Output {
        match regime {
            Either::Left(operands) => {
                self.high
                    .forward(device, env, high, (latent, (condition, operands)))
            }
            Either::Right(operands) => {
                self.low
                    .forward(device, env, low, (latent, (condition, operands)))
            }
        }
    }
}

/// Runs two conditioned trajectories separated by resampling, sharing the encoded condition.
#[derive(DeviceSend)]
pub struct Cascade<First, Resample, Second> {
    pub first: First,
    pub resample: Resample,
    pub second: Second,
}

/// The state holds the first trajectory's slot, the resampling's and the second trajectory's.
impl<
    's,
    'c,
    X,
    Y,
    Z,
    O,
    C,
    J,
    U,
    K,
    FirstState,
    ResampleState,
    SecondState,
    First,
    Resample,
    Second,
    Env: Copy,
> Module<(X, (&'c C, (J, U, K))), Env, &'s mut (FirstState, ResampleState, SecondState)>
    for Cascade<First, Resample, Second>
where
    First: Module<(X, (&'c C, J)), Env, &'s mut FirstState, Output = Y>,
    Resample: for<'x> Module<(&'x Y, U), Env, &'s mut ResampleState, Output = Z>,
    Second: Module<(Z, (&'c C, K)), Env, &'s mut SecondState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (first_state, resample_state, second_state): &'s mut (
            FirstState,
            ResampleState,
            SecondState,
        ),
        (latent, (condition, (first, resample, second))): (X, (&'c C, (J, U, K))),
    ) -> Self::Output {
        let latent =
            self.first
                .forward(&mut *device, env, first_state, (latent, (condition, first)));
        let latent = self
            .resample
            .forward(&mut *device, env, resample_state, (&latent, resample));
        self.second
            .forward(device, env, second_state, (latent, (condition, second)))
    }
}

/// Deterministic DDIM updates from an epsilon prediction and explicit timestep indices.
pub mod ddim {
    use super::Values;
    use crate::Module;
    use furiosa_opt_std::prelude::*;
    use std::{borrow::Borrow, marker::PhantomData};

    type Coefficient = Values<m![1 # 8]>;

    /// Current training index and the previous index; a negative previous index ends the
    /// trajectory.
    #[derive(DeviceSend)]
    pub struct Timestep {
        pub current: HbmTensor<i32, m![1], m![1]>,
        pub previous: HbmTensor<i32, m![1], m![1]>,
    }

    #[derive(DeviceSend)]
    pub struct Update<S, Training: AxisName> {
        pub(crate) alpha: Values<m![Training]>,
        pub(crate) terminal: Coefficient,
        pub(crate) limit: Coefficient,
        pub(crate) shape: PhantomData<S>,
    }

    impl<S, Training: AxisName> Update<S, Training> {
        pub fn new(alpha: Values<m![Training]>, terminal: Coefficient, limit: Coefficient) -> Self {
            Self {
                alpha,
                terminal,
                limit,
                shape: PhantomData,
            }
        }
    }

    /// One DDIM step: `sqrt(p) c + sqrt(1 - p) e` with `c = clamp((x - sqrt(1 - a) e) / sqrt(a),
    /// ±limit)`, `a`/`p` the current/previous alphas (`terminal` past the end), `e` the noise.
    impl<
        S: M,
        Training: AxisName,
        X: Borrow<Values<S>>,
        Y: Borrow<Values<S>>,
        J: Borrow<Timestep>,
        Env,
    > Module<(X, Y, J), Env, &mut ()> for Update<S, Training>
    {
        type Output = Values<S>;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: (X, Y, J)) -> Self::Output {
            todo!("NPU DDIM step")
        }
    }

    #[derive(DeviceSend)]
    pub struct Context<S: M> {
        pub sample: Values<S>,
        pub prediction: Values<S>,
        pub timestep: Timestep,
    }

    /// The sample, the prediction and the timestep a launch's context holds.
    impl<S: M, Training: AxisName, Env: Copy> Module<&mut Context<S>, Env, &mut ()>
        for Update<S, Training>
    {
        type Output = Values<S>;

        fn forward(
            &self,
            device: &mut Device,
            env: Env,
            _: &mut (),
            context: &mut Context<S>,
        ) -> Values<S> {
            self.forward(
                device,
                env,
                &mut (),
                (&context.sample, &context.prediction, &context.timestep),
            )
        }
    }
}

/// Deterministic Euler integration of a velocity field along an explicit sigma schedule.
pub mod flow_match {
    use super::Values;
    use crate::Module;
    use furiosa_opt_std::prelude::*;
    use std::{borrow::Borrow, marker::PhantomData};

    #[derive(DeviceSend)]
    pub struct Timestep {
        pub current: Values<m![1]>,
        pub next: Values<m![1]>,
    }

    #[derive(DeviceSend)]
    pub struct Update<S>(pub PhantomData<S>);

    impl<S: M, X: Borrow<Values<S>>, Y: Borrow<Values<S>>, T: Borrow<Timestep>, Env>
        Module<(X, Y, T), Env, &mut ()> for Update<S>
    {
        type Output = Values<S>;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: (X, Y, T)) -> Self::Output {
            todo!("NPU Euler sample + (next_sigma - current_sigma) * velocity")
        }
    }

    #[derive(DeviceSend)]
    pub struct Context<S: M> {
        pub sample: Values<S>,
        pub prediction: Values<S>,
        pub timestep: Timestep,
    }

    /// The sample, the prediction and the timestep a launch's context holds.
    impl<S: M, Env: Copy> Module<&mut Context<S>, Env, &mut ()> for Update<S> {
        type Output = Values<S>;

        fn forward(
            &self,
            device: &mut Device,
            env: Env,
            _: &mut (),
            context: &mut Context<S>,
        ) -> Values<S> {
            self.forward(
                device,
                env,
                &mut (),
                (&context.sample, &context.prediction, &context.timestep),
            )
        }
    }
}
