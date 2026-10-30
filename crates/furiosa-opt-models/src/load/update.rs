//! An update of a loaded tree's weights in place, from the chunks of a checkpoint.

use std::collections::{BTreeMap, BTreeSet};

use furiosa_opt_std::prelude::Device;

use super::{Checkpoint, Context, Load, LoadError, Rule};

/// An update of `T`'s weights in place: each rule writes once every name it reads has arrived,
/// and `finish` reports what never completed. Structure and tying stay as loaded; config leaves
/// are no update targets.
pub struct Update<'m, T> {
    tree: &'m mut T,
    device: Option<&'m mut Device>,
    /// The tensors arrived that a rule still waits on, with those of the chunk being loaded.
    received: Checkpoint,
    /// How far each rule got, by tree path.
    rules: BTreeMap<String, Rule>,
    unexpected: BTreeSet<String>,
}

/// Why an update did not load a chunk.
pub enum Failed<'m, T> {
    /// A tensor was refused before the device took a write of it: the update goes on, its leaves
    /// as they were; an `Unexpected` chunk wrote its other tensors.
    Refused(Update<'m, T>, LoadError),
    /// A write failed after the device took it: the tree's values are unknown until it is loaded
    /// again, so the update ends.
    Poisoned(LoadError),
}

impl<'m, T> Update<'m, T>
where
    for<'t, 's> &'t mut T: Load<(&'t mut T, &'s Checkpoint)>,
{
    /// An update of `tree`, whose HBM leaves are written on `device`; a tree on host needs none.
    pub fn new(tree: &'m mut T, device: impl Into<Option<&'m mut Device>>) -> Self {
        Self {
            tree,
            device: device.into(),
            received: Checkpoint::default(),
            rules: BTreeMap::new(),
            unexpected: BTreeSet::new(),
        }
    }

    /// Writes every rule `chunk` completes with what earlier chunks brought. A tensor no rule
    /// reads is refused as `Unexpected` at once and again by `finish`.
    pub fn load(mut self, chunk: &Checkpoint) -> Result<Self, Failed<'m, T>> {
        let arrived = chunk
            .tensors
            .iter()
            .map(|(name, stored)| (name.clone(), stored.clone()));
        self.received.tensors.extend(arrived);
        let mut context = Context::new(self.device.as_deref_mut());
        // `T` is named: an inferred tree type makes the solver try every `&mut _` impl of `Load`.
        let pass = <&mut T as Load<_>>::load(&mut context, (&mut *self.tree, &self.received));
        for rule in context.into_rules() {
            if !(rule.untouched() && self.rules.contains_key(&rule.tree)) {
                self.rules.insert(rule.tree.clone(), rule);
            }
        }
        let pass = pass.map(|_| ());
        match &pass {
            Ok(()) => {}
            Err(LoadError::Unexpected(names)) => self.unexpected.extend(names.iter().cloned()),
            Err(LoadError::InvalidData { name, .. } | LoadError::Unsupported { name, .. }) => {
                self.received.tensors.remove(name);
                return Err(Failed::Refused(self, pass.unwrap_err()));
            }
            Err(
                LoadError::Device {
                    source: furiosa_opt_std::Error::Invalid(_),
                    ..
                }
                | LoadError::Missing(_),
            ) => return Err(Failed::Refused(self, pass.unwrap_err())),
            Err(LoadError::Device { .. }) => return Err(Failed::Poisoned(pass.unwrap_err())),
        }
        let waiting: BTreeSet<&str> = self
            .rules
            .values()
            .filter(|rule| !rule.absent.is_empty())
            .flat_map(|rule| rule.required.iter().map(String::as_str))
            .collect();
        (self.received.tensors).retain(|name, _| waiting.contains(name.as_str()));
        match pass {
            Ok(()) => Ok(self),
            Err(error) => Err(Failed::Refused(self, error)),
        }
    }

    /// `Missing` for every rule fed only part of its names, then `Unexpected` for every tensor no
    /// rule read.
    pub fn finish(self) -> Result<(), LoadError> {
        let missing: Vec<String> = self
            .rules
            .values()
            .filter(|rule| !rule.absent.is_empty() && !rule.untouched())
            .flat_map(|rule| rule.absent.iter().cloned())
            .collect();
        if !missing.is_empty() {
            return Err(LoadError::missing(missing));
        }
        match self.unexpected.is_empty() {
            true => Ok(()),
            false => Err(LoadError::Unexpected(self.unexpected.into_iter().collect())),
        }
    }
}

impl Rule {
    /// Whether the rule lacks every name it reads: no chunk fed it.
    fn untouched(&self) -> bool {
        self.absent.iter().collect::<BTreeSet<_>>() == self.required.iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::checkpoint::tests::Value;
    use crate::load::checkpoint::tests::{
        Causal, EMBED, Embedding, H, V, Values, bf16s, causal, checkpoint, device, host, invalid,
        load, missing, unexpected,
    };
    use crate::tensor::memory::{Hbm, Memory};
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::*;
    use safetensors::Dtype;

    axes![R = 2];

    #[derive(crate::load::Load)]
    struct Projection<X: Memory> {
        weight: X::Tensor<bf16, m![1], m![R, H]>,
    }

    #[derive(crate::load::Load)]
    struct Parts<X: Memory> {
        q_proj: Projection<X>,
        k_proj: Projection<X>,
    }

    /// The query and key rows, stacked.
    #[derive(crate::load::Load)]
    #[load(from = Parts<X>)]
    struct Packed<X: Memory> {
        weight: X::Tensor<bf16, m![1], m![V / 2, H]>,
    }

    impl Load<Parts<Values>> for Packed<Values> {
        fn load(_: &mut Context<'_>, parts: Parts<Values>) -> Result<Self, LoadError> {
            let values = [parts.q_proj.weight.values(), parts.k_proj.weight.values()].concat();
            let weight = Value(HostTensor::from_vec(values));
            Ok(Packed { weight })
        }
    }

    const PART: &[usize] = &[2, 4];

    #[derive(crate::load::Load)]
    struct Twins<X: Memory>(
        #[load(rename = "a")] Packed<X>,
        #[load(rename = "b")] Packed<X>,
    );

    /// A chunk of BF16 `tensors`, each valued alike.
    fn chunk(tensors: &[(&str, &[usize], f32)]) -> Checkpoint {
        let tensors: Vec<_> = tensors
            .iter()
            .map(|(name, shape, value)| {
                let bytes = bf16s(shape.iter().product(), *value);
                (*name, Dtype::BF16, *shape, bytes)
            })
            .collect();
        checkpoint(&tensors)
    }

    /// A chunk of one BF16 tensor stored from `byte`.
    fn bytes(name: &str, shape: &[usize], byte: u8) -> Checkpoint {
        let data = vec![byte; shape.iter().product::<usize>() * 2];
        checkpoint(&[(name, Dtype::BF16, shape, data)])
    }

    /// The update after a chunk it loaded.
    fn loaded<'m, T>(result: Result<Update<'m, T>, Failed<'m, T>>) -> Update<'m, T> {
        match result {
            Ok(update) => update,
            Err(Failed::Refused(_, error) | Failed::Poisoned(error)) => panic!("{error}"),
        }
    }

    /// The update a refused chunk left, and why it refused it.
    fn refused<'m, T>(result: Result<Update<'m, T>, Failed<'m, T>>) -> (Update<'m, T>, LoadError) {
        match result {
            Err(Failed::Refused(update, error)) => (update, error),
            Err(Failed::Poisoned(error)) => panic!("poisoned: {error}"),
            Ok(_) => panic!("loaded"),
        }
    }

    fn embedding(value: f32) -> Embedding<Values> {
        load(&chunk(&[("weight", EMBED, value)])).unwrap()
    }

    fn tied(value: f32) -> Causal<Values> {
        load(&Checkpoint::new(causal(value).path(), None).unwrap()).unwrap()
    }

    /// Each pass builds its values from what arrived: the second value of a name wins.
    #[test]
    fn second_value_wins() {
        let mut tree = embedding(0.0);
        let update = loaded(Update::new(&mut tree, None).load(&chunk(&[("weight", EMBED, 1.0)])));
        let update = loaded(update.load(&chunk(&[("weight", EMBED, 2.0)])));
        update.finish().unwrap();
        assert_eq!(tree.weight.values(), vec![bf16::from_f32(2.0); 32]);
    }

    fn packed() -> Packed<Values> {
        load(&chunk(&[
            ("q_proj.weight", PART, 0.0),
            ("k_proj.weight", PART, 0.0),
        ]))
        .unwrap()
    }

    /// A rule reading two names writes once both arrived, one per chunk.
    #[test]
    fn two_name_rule_writes_when_both_arrived() {
        let mut tree = packed();
        let update = Update::new(&mut tree, None);
        let update = loaded(update.load(&chunk(&[("q_proj.weight", PART, 1.0)])));
        let update = loaded(update.load(&chunk(&[("k_proj.weight", PART, 2.0)])));
        update.finish().unwrap();
        let expected: Vec<_> = [1.0, 2.0]
            .into_iter()
            .flat_map(|v| vec![bf16::from_f32(v); 8])
            .collect();
        assert_eq!(tree.weight.values(), expected);
    }

    /// Tuple fields renamed apart update as rules of their own: each fed half of its names is
    /// missing its other half, and each fed whole holds its own values.
    #[test]
    fn renamed_tuple_fields_update_apart() {
        let parts = |a, b| chunk(&[("a.q_proj.weight", PART, a), ("b.q_proj.weight", PART, b)]);
        let keys = |a, b| chunk(&[("a.k_proj.weight", PART, a), ("b.k_proj.weight", PART, b)]);
        let mut tree: Twins<Values> = load(&chunk(&[
            ("a.q_proj.weight", PART, 0.0),
            ("a.k_proj.weight", PART, 0.0),
            ("b.q_proj.weight", PART, 0.0),
            ("b.k_proj.weight", PART, 0.0),
        ]))
        .unwrap();
        let update = loaded(Update::new(&mut tree, None).load(&parts(1.0, 3.0)));
        assert_eq!(
            missing(update.finish().err()),
            ["a.k_proj.weight", "b.k_proj.weight"]
        );
        let update = loaded(Update::new(&mut tree, None).load(&parts(1.0, 3.0)));
        let update = loaded(update.load(&keys(2.0, 4.0)));
        update.finish().unwrap();
        let rows = |q: f32, k: f32| -> Vec<_> {
            [q, k]
                .into_iter()
                .flat_map(|value| vec![bf16::from_f32(value); 8])
                .collect()
        };
        assert_eq!(tree.0.weight.values(), rows(1.0, 2.0));
        assert_eq!(tree.1.weight.values(), rows(3.0, 4.0));
    }

    /// A rule fed only some of its names is missing the rest at the end and writes nothing.
    #[test]
    fn partially_fed_rule_is_missing() {
        let mut tree = packed();
        let update = Update::new(&mut tree, None);
        let update = loaded(update.load(&chunk(&[("q_proj.weight", PART, 1.0)])));
        assert_eq!(missing(update.finish().err()), ["k_proj.weight"]);
        assert_eq!(tree.weight.values(), vec![bf16::from_f32(0.0); 16]);
    }

    /// A tied rule nothing fed is no update at the end, and both leaves keep their values.
    #[test]
    fn untouched_tied_rule_is_a_no_op() {
        let mut tree = tied(1.0);
        loaded(Update::new(&mut tree, None).load(&chunk(&[])))
            .finish()
            .unwrap();
        assert_eq!(tree.lm_head.weight.values(), vec![bf16::from_f32(1.0); 32]);
    }

    /// An update writes a tied target once: the tied leaf of its type is the target's storage in
    /// HBM, so it is not written again; an equal stored copy is read.
    #[test]
    fn tied_target_is_written_once() {
        let mut tree = tied(1.0);
        let both = chunk(&[
            ("model.embed_tokens.weight", EMBED, 2.0),
            ("lm_head.weight", EMBED, 2.0),
        ]);
        loaded(Update::new(&mut tree, None).load(&both))
            .finish()
            .unwrap();
        let embed = tree.model.embed_tokens.weight.values();
        assert_eq!(embed, vec![bf16::from_f32(2.0); 32]);
        let lm_head = tree.lm_head.weight.values();
        assert_eq!(
            lm_head,
            vec![bf16::from_f32(1.0); 32],
            "a host leaf shares no storage"
        );
    }

    /// A stored copy arriving before its target is read and waits for it: no name is unexpected,
    /// and the tied rule is missing the target until it arrives.
    #[test]
    fn stored_copy_before_its_target_waits() {
        let mut tree = tied(1.0);
        let update = Update::new(&mut tree, None);
        let update = loaded(update.load(&chunk(&[("lm_head.weight", EMBED, 5.0)])));
        let update = loaded(update.load(&chunk(&[("model.embed_tokens.weight", EMBED, 5.0)])));
        update.finish().unwrap();
        assert_eq!(
            tree.model.embed_tokens.weight.values(),
            vec![bf16::from_f32(5.0); 32]
        );
        let update = Update::new(&mut tree, None);
        let update = loaded(update.load(&chunk(&[("lm_head.weight", EMBED, 6.0)])));
        assert_eq!(
            missing(update.finish().err()),
            ["model.embed_tokens.weight"]
        );
    }

    /// A config leaf is no update target: a chunk naming it is unexpected at once and at the end.
    #[test]
    fn chunk_naming_a_config_leaf_is_unexpected() {
        let mut tree = tied(1.0);
        let update = Update::new(&mut tree, None);
        let (update, error) = refused(update.load(&chunk(&[("model.epsilon", &[1], 2.0)])));
        assert_eq!(unexpected(Some(error)), ["model.epsilon"]);
        assert_eq!(unexpected(update.finish().err()), ["model.epsilon"]);
        assert_eq!(tree.model.epsilon.values()[0], 0.5);
    }

    /// A tensor its rule refuses, or the device refuses before the copy, leaves the leaf as it was
    /// and the update going on: a later chunk writes the leaf.
    #[test]
    fn refused_tensor_leaves_the_leaf_and_the_update() {
        let wide = chunk(&[("weight", &[8, 5], 1.0)]);
        let unwritable = bytes("weight", EMBED, 0xfe);
        let mut tree = embedding(0.0);
        let (update, error) = refused(Update::new(&mut tree, None).load(&wide));
        let reason = "shape [8, 5], the leaf holds [8, 4]";
        assert_eq!(invalid(Some(error)), ("weight".into(), reason.into()));
        drop(update);
        assert_eq!(tree.weight.values(), vec![bf16::from_f32(0.0); 32]);
        let (update, error) = refused(Update::new(&mut tree, None).load(&unwritable));
        assert!(
            matches!(&error, LoadError::Device {
                path,
                source: furiosa_opt_std::Error::Invalid("refused"),
            } if path == "weight"),
            "{error}"
        );
        drop(update);
        assert_eq!(tree.weight.values(), vec![bf16::from_f32(0.0); 32]);
        let (update, _) = refused(Update::new(&mut tree, None).load(&wide));
        let (update, _) = refused(update.load(&unwritable));
        let update = loaded(update.load(&chunk(&[("weight", EMBED, 3.0)])));
        update.finish().unwrap();
        assert_eq!(tree.weight.values(), vec![bf16::from_f32(3.0); 32]);
    }

    /// A write that fails after the device took it ends the update with the leaf's path and the
    /// device's failure.
    #[test]
    fn failure_after_submission_ends_the_update() {
        let mut tree = embedding(0.0);
        let update = Update::new(&mut tree, None);
        match update.load(&bytes("weight", EMBED, 0xff)) {
            Err(Failed::Poisoned(LoadError::Device {
                path,
                source: furiosa_opt_std::Error::Poisoned,
            })) => assert_eq!(path, "weight"),
            Err(Failed::Poisoned(error) | Failed::Refused(_, error)) => panic!("{error}"),
            Ok(_) => panic!("loaded"),
        }
    }

    /// An update writes an HBM leaf in its allocation, and the tied leaf, the same storage, reads
    /// the new value.
    #[test]
    fn update_writes_in_place_and_the_tied_leaf_sees_it() {
        let _lease = DEVICE.blocking_lock();
        let mut device = device();
        let files = causal(1.0);
        let mut tree: Causal<Hbm> = Checkpoint::new(files.path(), None)
            .unwrap()
            .load(&mut device)
            .unwrap();
        let update = Update::new(&mut tree, &mut device);
        loaded(update.load(&chunk(&[("model.embed_tokens.weight", EMBED, 4.0)])))
            .finish()
            .unwrap();
        let lm_head: HostTensor<bf16, m![V, H]> = host(&tree.lm_head.weight, &mut device);
        assert_eq!(lm_head.into_vec(), vec![bf16::from_f32(4.0); 32]);
    }
}
