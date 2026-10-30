//! Dense, gated and routed feed-forward networks.

use super::residual::Residual;
use super::residual::Sum;
use crate::load::Load;
use crate::{Fork, Module, Then};
use furiosa_opt_std::prelude::{Device, DeviceSend};

pub mod glu;
pub mod moe;

/// Projects hidden values up, applies an activation and projects back down.
pub type Mlp<Up, Activation, Down> = Then<Up, Then<Activation, Down>>;

/// Selects experts from the input and combines their outputs with routing probabilities.
#[derive(DeviceSend, Load)]
pub struct Routed<Router, Experts, Weighting> {
    /// Produces expert indices and their routing probabilities.
    #[load(rename = "")]
    pub router: Router,
    /// Evaluates the experts selected by the router.
    #[load(rename = "")]
    pub experts: Experts,
    /// Combines expert outputs using the routing probabilities.
    #[load(rename = "")]
    pub weight: Weighting,
}

/// A gated MLP whose projections use caller-supplied expert indices.
#[derive(DeviceSend, Load)]
pub struct ExpertMlp<Gate, Activation, Up, Multiply, Down> {
    /// Indexed projection supplying the gate values.
    #[load(rename = "")]
    pub gate: Gate,
    /// Nonlinearity on the gate branch.
    #[load(rename = "")]
    pub activation: Activation,
    /// Indexed projection of the same hidden input.
    #[load(rename = "")]
    pub up: Up,
    /// Elementwise product of the gate and up branches.
    #[load(rename = "")]
    pub multiply: Multiply,
    /// Indexed projection back into the output feature space.
    #[load(rename = "")]
    pub down: Down,
}

impl<X, I, Y, Z, V, W, O, Gate, Activation, Up, Multiply, Down, Env: Copy>
    Module<(&X, &I), Env, &mut ()> for ExpertMlp<Gate, Activation, Up, Multiply, Down>
where
    Gate: for<'a, 'b, 's> Module<(&'a X, &'b I), Env, &'s mut (), Output = Y>,
    Activation: for<'x, 's> Module<&'x Y, Env, &'s mut (), Output = Z>,
    Up: for<'a, 'b, 's> Module<(&'a X, &'b I), Env, &'s mut (), Output = V>,
    Multiply: for<'a, 'b, 's> Module<(&'a Z, &'b V), Env, &'s mut (), Output = W>,
    Down: for<'a, 'b, 's> Module<(&'a W, &'b I), Env, &'s mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, _: &mut (), (hidden, indices): (&X, &I)) -> O {
        let gate = self
            .gate
            .forward(&mut *device, env, &mut (), (hidden, indices));
        let gate = self.activation.forward(&mut *device, env, &mut (), &gate);
        let up = self
            .up
            .forward(&mut *device, env, &mut (), (hidden, indices));
        let hidden = self
            .multiply
            .forward(&mut *device, env, &mut (), (&gate, &up));
        self.down.forward(device, env, &mut (), (&hidden, indices))
    }
}

/// The state holds the router's slot, then the experts'.
impl<'s, X, I, P, Y, O, RouterState, ExpertsState, Router, Experts, Weighting, Env: Copy>
    Module<&X, Env, &'s mut (RouterState, ExpertsState)> for Routed<Router, Experts, Weighting>
where
    Router: for<'x> Module<&'x X, Env, &'s mut RouterState, Output = (I, P)>,
    Experts: for<'a, 'b> Module<(&'a X, &'b I), Env, &'s mut ExpertsState, Output = Y>,
    Weighting: for<'a, 'b, 'd> Module<(&'a Y, &'b P), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (router, experts): &'s mut (RouterState, ExpertsState),
        input: &X,
    ) -> Self::Output {
        let (indices, probabilities) = self.router.forward(&mut *device, env, router, input);
        let experts = self
            .experts
            .forward(&mut *device, env, experts, (input, &indices));
        self.weight
            .forward(device, env, &mut (), (&experts, &probabilities))
    }
}

/// Adds shared expert output to probability-weighted routed expert output; the shared experts read
/// under the name their family gives them.
pub type Moe<Routed, Shared, Add> = Sum<Fork<(Routed, Shared)>, Add>;

/// Computes `x + feedforward(norm(x))`.
pub type Layer<Norm, FeedForward, Add> = Residual<Norm, FeedForward, Add>;
