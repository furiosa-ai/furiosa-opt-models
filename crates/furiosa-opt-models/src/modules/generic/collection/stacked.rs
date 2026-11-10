use furiosa_opt_std::prelude::*;

use super::{Extent, Interval, View};
use crate::load::{Checkpoint, Context, Load, LoadError, Part};
use crate::modules::generic::Collection;

/// A layer collection owning one backing per parameter kind, and the interval of its live layers.
#[derive(DeviceSend)]
pub struct Stacked<L: View<N>, const N: usize, I = Interval<usize>> {
    storage: L::Storage,
    interval: I,
}

impl<L: View<N>, const N: usize, I: Extent<N>> Stacked<L, N, I> {
    pub fn origin(&self) -> usize {
        self.interval.origin()
    }

    pub fn count(&self) -> usize {
        self.interval.count()
    }

    pub fn layer(&self, layer: usize) -> L::Layer<'_> {
        L::view(&self.storage, layer)
    }
}

/// The layers the checkpoint's part selects, under segments of their global indices, each read as
/// one layer is, every leaf filling its layer of one backing; the configured layers left out are
/// known unused, and the layers' views then stack into the storage.
impl<'a, L: View<N> + Load<&'a Checkpoint>, const N: usize> Load<&'a Checkpoint> for Stacked<L, N> {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        const { assert!(N > 0, "a stack holds a layer") };
        let (interval, depth) = match from.part() {
            Part::Layers(interval) if interval.range().end > N => {
                return Err(LoadError::InvalidData {
                    name: "layers".into(),
                    reason: format!("{:?} ends past the stack's {N} layers", interval.range()),
                });
            }
            Part::Layers(interval) => (*interval, from.depth()?),
            Part::Whole | Part::Encoder => (Interval::new(0..N, N)?, N),
        };
        let storage = context.checkpoint(from, |cx| {
            for layer in (0..depth).filter(|layer| !interval.range().contains(layer)) {
                cx.ignore(&layer.to_string());
            }
            cx.layers(interval, |cx| L::load(cx, from), L::stack)
        })?;
        Ok(Self { storage, interval })
    }
}

/// A borrowed stack borrows each backing, as the leaves of its storage borrow theirs.
impl<'a, L: View<N>, B: View<N>, const N: usize, I, J> Load<&'a Stacked<B, N, J>>
    for Stacked<L, N, I>
where
    L::Storage: Load<&'a B::Storage>,
    I: Load<&'a J>,
{
    fn load(context: &mut Context<'_>, from: &'a Stacked<B, N, J>) -> Result<Self, LoadError> {
        Ok(Self {
            storage: L::Storage::load(context, &from.storage)?,
            interval: I::load(context, &from.interval)?,
        })
    }
}

/// Each layer is a view of the stacked parameters, built when it runs.
impl<X, Env, S, L: View<N>, const N: usize, I: Extent<N>, O> Collection<X, Env, S>
    for Stacked<L, N, I>
where
    for<'a> L::Layer<'a>: crate::Module<X, Env, S, Output = O>,
{
    type Output = O;

    const COUNT: usize = N;

    fn count(&self) -> usize {
        self.count()
    }

    fn apply(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        hidden: X,
        index: usize,
    ) -> Self::Output {
        crate::Module::forward(&self.layer(index), device, env, state, hidden)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::tests::{
        Bytes, F, Model, checkpoint, load, parts, selected, tensors, values,
    };
    use super::*;
    use std::ptr;

    /// A stack of `storage` whose live layers are `interval`'s.
    pub(in super::super) fn stacked<L: View<N>, const N: usize, I>(
        storage: L::Storage,
        interval: I,
    ) -> Stacked<L, N, I> {
        Stacked { storage, interval }
    }

    /// A stack reads each layer under its global index in the body's `layers`, and its views
    /// give each layer its own values: the gate rows, then the up rows, of the layer's MLP.
    #[test]
    fn stacks_each_layer_under_its_index() {
        let tensors = tensors(2);
        let model = load(&checkpoint(&tensors)).unwrap();
        let stack = &model.body.layers.0;
        let stored = |name: &str| tensors.iter().find(|(n, ..)| n == name).unwrap().2.clone();
        for l in 0..2 {
            let (norm, gate_up) = parts(&stack.layer(l));
            assert_eq!(norm, stored(&format!("layers.{l}.input_layernorm")));
            let gate = stored(&format!("layers.{l}.mlp.gate_proj.weight"));
            let up = stored(&format!("layers.{l}.mlp.up_proj.weight"));
            assert_eq!(gate_up, [gate, up].concat(), "layer {l}");
        }
        assert_ne!(parts(&stack.layer(0)), parts(&stack.layer(1)));
    }

    /// A stack fails with every absent name of every layer, and with a name past its last layer.
    #[test]
    fn stack_reads_strictly() {
        let mut tensors = tensors(2);
        tensors.retain(|(name, ..)| {
            !["layers.0.mlp.up_proj.weight", "layers.1.input_layernorm"].contains(&name.as_str())
        });
        match load(&checkpoint(&tensors)).err() {
            Some(LoadError::Missing(names)) => assert_eq!(
                names,
                ["layers.0.mlp.up_proj.weight", "layers.1.input_layernorm"]
            ),
            other => panic!("{other:?} is not `Missing`"),
        }
        let mut tensors = super::super::tests::tensors(2);
        let extra = (
            "layers.2.input_layernorm".into(),
            vec![F::SIZE],
            values(F::SIZE, 0.0),
        );
        tensors.push(extra);
        match load(&checkpoint(&tensors)).err() {
            Some(LoadError::Unexpected(names)) => {
                assert_eq!(names, ["layers.2.input_layernorm"])
            }
            other => panic!("{other:?} is not `Unexpected`"),
        }
    }

    /// A selection reads strictly to the depth the nested text config declares, below the
    /// stack's: an unselected configured layer is unused, a layer past that depth unexpected,
    /// and an absent selected one missing.
    #[test]
    fn selection_reads_to_the_configured_depth() {
        let config = serde_json::json!({"text_config": {"num_hidden_layers": 3}});
        let read = |tensors: &[(String, Vec<usize>, Vec<u8>)]| {
            let source = selected(tensors, config.clone(), 1..3);
            Model::<Bytes, 4>::load(&mut Context::new(None), &source).err()
        };
        assert!(read(&tensors(3)).is_none());
        let mut past: Vec<String> = (tensors(4).into_iter().map(|(name, ..)| name))
            .filter(|name| name.starts_with("layers.3."))
            .collect();
        past.sort();
        match read(&tensors(4)) {
            Some(LoadError::Unexpected(names)) => assert_eq!(names, past),
            other => panic!("{other:?} is not `Unexpected`"),
        }
        let mut absent = tensors(3);
        absent.retain(|(name, ..)| name != "layers.2.input_layernorm");
        match read(&absent) {
            Some(LoadError::Missing(names)) => assert_eq!(names, ["layers.2.input_layernorm"]),
            other => panic!("{other:?} is not `Missing`"),
        }
    }

    /// A selection ending past the stack's layers is refused as invalid, not loaded.
    #[test]
    fn refuses_a_selection_past_its_layers() {
        let source = selected(
            &tensors(4),
            serde_json::json!({"num_hidden_layers": 4}),
            1..4,
        );
        match Model::<Bytes>::load(&mut Context::new(None), &source).err() {
            Some(LoadError::InvalidData { name, .. }) => assert_eq!(name, "layers"),
            other => panic!("{other:?} is not `InvalidData`"),
        }
    }

    /// A borrowed stack borrows the loaded storage and builds each layer as the owned stack does.
    #[test]
    fn borrowed_stack_borrows_the_storage() {
        let model = load(&checkpoint(&tensors(2))).unwrap();
        let borrowed: Model<&Bytes> = Load::load(&mut Context::new(None), &model).unwrap();
        let (owned, stack) = (&model.body.layers.0, &borrowed.body.layers.0);
        assert!(ptr::eq(stack.storage.0, &owned.storage.0));
        for l in 0..2 {
            assert_eq!(parts(&stack.layer(l)), parts(&owned.layer(l)));
        }
    }
}
