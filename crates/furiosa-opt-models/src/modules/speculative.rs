//! Draft structures for speculative decoding, which predict further tokens from the target
//! model's hidden value.

use crate::Module;
use furiosa_opt_std::prelude::Device;

/// Applies every head to the same hidden value.
pub struct Heads<Head> {
    pub heads: Vec<Head>,
}

/// The state is one slot a head.
impl<'x, 's, X, Head, HeadState, O, Env: Copy> Module<&'x X, Env, &'s mut [HeadState]>
    for Heads<Head>
where
    Head: Module<&'x X, Env, &'s mut HeadState, Output = O>,
{
    type Output = Vec<O>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &'s mut [HeadState],
        input: &'x X,
    ) -> Self::Output {
        assert_eq!(state.len(), self.heads.len(), "one slot for each head");
        self.heads
            .iter()
            .zip(state)
            .map(|(head, slot)| head.forward(&mut *device, env, slot, input))
            .collect()
    }
}

/// Applies each layer as an activated residual update, then the head.
pub struct Prediction<Layer, Activation, Add, Head> {
    pub layers: Vec<Layer>,
    pub activation: Activation,
    pub add: Add,
    pub head: Head,
}

/// A composition, not an op.
impl<X, Layer, Activation, Add, Head> !crate::Op<&X> for Prediction<Layer, Activation, Add, Head> {}

/// The layers are projections; the state is the head's slot.
impl<X, Y, Z, O, S, Layer, Activation, Add, Head, Env: Copy> Module<&X, Env, S>
    for Prediction<Layer, Activation, Add, Head>
where
    Layer: for<'x, 's> Module<&'x X, Env, &'s mut (), Output = Y>,
    Activation: for<'x, 's> Module<&'x Y, Env, &'s mut (), Output = Z>,
    Add: for<'a, 'b, 's> Module<(&'a X, &'b Z), Env, &'s mut (), Output = X>,
    Head: for<'x> Module<&'x X, Env, S, Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, state: S, input: &X) -> Self::Output {
        let mut value = None;
        for layer in &self.layers {
            let input = value.as_ref().unwrap_or(input);
            let hidden = layer.forward(&mut *device, env, &mut (), input);
            let hidden = self.activation.forward(&mut *device, env, &mut (), &hidden);
            value = Some(
                self.add
                    .forward(&mut *device, env, &mut (), (input, &hidden)),
            );
        }
        self.head
            .forward(device, env, state, value.as_ref().unwrap_or(input))
    }
}

/// Applies each layer to the hidden value before it and returns every hidden value, the initial
/// one first.
pub struct Collect<Layer> {
    pub layers: Vec<Layer>,
}

/// The state is one slot a layer.
impl<'s, X, LayerState, Layer, Env: Copy> Module<X, Env, &'s mut [LayerState]> for Collect<Layer>
where
    Layer: for<'b> Module<&'b X, Env, &'s mut LayerState, Output = X>,
{
    type Output = Vec<X>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &'s mut [LayerState],
        hidden: X,
    ) -> Vec<X> {
        assert_eq!(state.len(), self.layers.len(), "one slot for each layer");
        let mut outputs = Vec::with_capacity(self.layers.len() + 1);
        outputs.push(hidden);
        for (layer, slot) in self.layers.iter().zip(state) {
            let hidden = layer.forward(
                &mut *device,
                env,
                slot,
                outputs.last().expect("initial value present"),
            );
            outputs.push(hidden);
        }
        outputs
    }
}

/// Embeds the tokens, joins them with the target model's hidden value and projects the result.
pub struct Embedding<Tokens, Join, Projection> {
    pub embedding: Tokens,
    pub join: Join,
    pub projection: Projection,
}

/// The state is the token embedding's slot.
impl<'t, I, X, Y, Z, S, Tokens, Join, Projection, Env: Copy> Module<(I, &'t X), Env, S>
    for Embedding<Tokens, Join, Projection>
where
    Tokens: Module<I, Env, S, Output = X>,
    Join: for<'a, 'b, 'd> Module<(&'a X, &'b X), Env, &'d mut (), Output = Y>,
    Projection: for<'x, 'd> Module<&'x Y, Env, &'d mut (), Output = Z>,
{
    type Output = Z;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        (input, target): (I, &'t X),
    ) -> Self::Output {
        let hidden = self.embedding.forward(&mut *device, env, state, input);
        let hidden = self
            .join
            .forward(&mut *device, env, &mut (), (&hidden, target));
        self.projection.forward(device, env, &mut (), &hidden)
    }
}

/// Applies the module the index selects, or nothing when the index is out of range.
pub struct Indexed<Compute> {
    pub modules: Vec<Compute>,
}

/// The state is one slot a module; the selected module takes its own.
impl<'s, I, S, Compute, O, Env: Copy> Module<(usize, I), Env, &'s mut [S]> for Indexed<Compute>
where
    Compute: Module<I, Env, &'s mut S, Output = O>,
{
    type Output = Option<O>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &'s mut [S],
        (index, input): (usize, I),
    ) -> Self::Output {
        assert_eq!(state.len(), self.modules.len(), "one slot for each module");
        let module = self.modules.get(index)?;
        Some(module.forward(device, env, &mut state[index], input))
    }
}
