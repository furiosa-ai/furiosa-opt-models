//! The position of a traversal, the scopes it is in, and what it read.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::mem;
use std::pin::Pin;

use furiosa_opt_std::prelude::{Device, HbmBuffer, HbmTensor, HostTensor, M, MaterializableScalar};
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use tokio::task;

use super::{Checkpoint, Children, LoadError, Rule};
use ties::Ties;

mod ties;

/// One traversal of a tree: the device its leaves load into, where it is, and what it read.
pub struct Context<'d> {
    device: Option<&'d mut Device>,
    cursor: Cursor,
    ties: Ties,
    /// Every name read, in order: a rule read the names read since it started.
    read: Vec<String>,
    strict: Option<Strict>,
    rules: Vec<Rule>,
    stack: Option<Stack>,
}

/// The layers a traversal reads at once: each HBM leaf fills its layer of one backing, by tree path.
struct Stack {
    count: usize,
    layer: usize,
    backings: BTreeMap<String, HbmBuffer>,
}

/// Tree path, checkpoint name, and the scopes a traversal is in.
#[derive(Default)]
struct Cursor {
    tree: Vec<String>,
    path: Vec<String>,
    /// The config key entered at the current position: the value a leaf here reads, and the scope
    /// of the fields below it.
    config: Option<String>,
    /// The config keys entered above the current position, outermost first.
    scope: Vec<String>,
    /// The target and own checkpoint prefixes of the tied field being read.
    tied: Option<(String, String)>,
}

/// The outermost checkpoint traversal: it fails with every name below `root` that it neither read
/// (from `start` of the names read) nor ignored, and waits for the copies its leaves submitted.
struct Strict {
    root: String,
    start: usize,
    ignore: Vec<String>,
    /// Each copy into a leaf's storage, in the order the leaves submitted them.
    copies: Vec<Pending>,
}

/// A copy into a leaf's storage, submitted and not yet seen to end; it fails as its leaf.
type Pending = Pin<Box<dyn Future<Output = Result<(), LoadError>>>>;

impl<'d> Context<'d> {
    /// A traversal at the root; leaves load into HBM of `device`, which a borrow does not need.
    pub fn new(device: impl Into<Option<&'d mut Device>>) -> Self {
        Self {
            device: device.into(),
            cursor: Cursor::default(),
            ties: Ties::default(),
            read: Vec::new(),
            strict: None,
            rules: Vec::new(),
            stack: None,
        }
    }

    /// The checkpoint name of the current position.
    pub fn name(&self) -> String {
        Self::join(&self.cursor.path)
    }

    /// The tree path of the current position.
    pub fn path(&self) -> String {
        Self::join(&self.cursor.tree)
    }

    /// The checkpoint name `name` beside the current name, under the same parent.
    pub fn sibling(&self, name: &str) -> String {
        match self.name().rsplit_once('.') {
            Some((parent, _)) => format!("{parent}.{name}"),
            None => name.to_string(),
        }
    }

    /// The config key the current leaf reads instead of a tensor.
    pub fn config_key(&self) -> Option<&str> {
        self.cursor.config.as_deref()
    }

    /// The device HBM leaves load into; a traversal that only borrows or builds host leaves has
    /// none.
    pub fn device(&mut self) -> &mut Device {
        (self.device.as_deref_mut()).expect("an HBM leaf loads through a context with a device")
    }

    /// Marks `name` read: a strict traversal does not report it, and the current rule requires it.
    pub fn read(&mut self, name: &str) {
        self.read.push(name.into());
    }

    /// Runs `visit` one field down: `tree` and `checkpoint` segments, both restored after it. A
    /// config key entered above the field scopes the config reads below it.
    pub fn field<R>(
        &mut self,
        tree: &str,
        checkpoint: &str,
        visit: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.cursor.tree.push(tree.into());
        self.cursor.path.push(checkpoint.into());
        let result = self.below(visit);
        self.cursor.path.pop();
        self.cursor.tree.pop();
        result
    }

    /// Runs `visit` reading its config under `key`, relative to the current scope: a leaf reads
    /// the value at `key`, a subtree resolves its config reads there.
    pub fn config<R>(&mut self, key: &str, visit: impl FnOnce(&mut Self) -> R) -> R {
        let entered = match &self.cursor.config {
            Some(outer) => Self::join(&[outer.clone(), key.into()]),
            None => key.into(),
        };
        let previous = self.cursor.config.replace(entered);
        let result = visit(self);
        self.cursor.config = previous;
        result
    }

    /// The config value at `key` in the current scope and its config name
    /// (`config.json:text_config.vocab_size`); recorded as read, `Missing` under that name.
    pub fn config_value<'s>(
        &mut self,
        source: &'s Checkpoint,
        key: &str,
    ) -> Result<(String, &'s serde_json::Value), LoadError> {
        let path = Self::join(&[Self::join(&self.cursor.scope), key.into()]);
        let name = format!("config.json:{path}");
        self.read(&name);
        let value = (path.split('.')).try_fold(source.config(), |value, key| value.get(key));
        match value {
            Some(value) => Ok((name, value)),
            None => Err(LoadError::Missing(vec![name])),
        }
    }

    /// The number at `key` in the current scope and its config name, read as `config_value` reads.
    pub fn config_number(
        &mut self,
        source: &Checkpoint,
        key: &str,
    ) -> Result<(String, f64), LoadError> {
        let (name, value) = self.config_value(source, key)?;
        match value.as_f64() {
            Some(number) => Ok((name, number)),
            None => Err(LoadError::InvalidData {
                reason: format!("{value} is not a number"),
                name,
            }),
        }
    }

    /// The tensor at the current name and that name; recorded as read, `Missing` under it.
    pub fn tensor<'s>(
        &mut self,
        source: &'s Checkpoint,
    ) -> Result<(String, TensorView<'s>), LoadError> {
        self.find(|name| source.get(name))
    }

    /// The dtype and shape of the tensor at the current name, read without its values, and that
    /// name; recorded as read, `Missing` under it.
    pub fn header<'s>(
        &mut self,
        source: &'s Checkpoint,
    ) -> Result<(String, (Dtype, &'s [usize])), LoadError> {
        self.find(|name| source.header(name))
    }

    /// The values of the float tensor of `shape` at the current name, as FP32.
    pub fn floats(&mut self, source: &Checkpoint, shape: &[usize]) -> Result<Vec<f32>, LoadError> {
        let (name, view) = self.tensor(source)?;
        let invalid = |reason: String| LoadError::InvalidData {
            name: name.clone(),
            reason,
        };
        if view.shape() != shape {
            let stored = view.shape();
            return Err(invalid(format!(
                "shape {stored:?}, the leaf holds {shape:?}"
            )));
        }
        Checkpoint::floats(&view).ok_or_else(|| {
            let stored = view.dtype();
            invalid(format!("stored as {stored:?}, the leaf holds FP32 values"))
        })
    }

    /// Runs `visit` for the field `tree` (checkpoint segment `own`) reading the checkpoint at
    /// `target`, a path relative to the declaring struct.
    pub fn tied<R>(
        &mut self,
        tree: &str,
        own: &str,
        target: &str,
        visit: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let base = self.name();
        let at = |relative: &str| Self::join(&[base.clone(), relative.into()]);
        let previous = self.cursor.tied.replace((at(target), at(own)));
        let path = mem::replace(&mut self.cursor.path, vec![at(target)]);
        self.cursor.tree.push(tree.into());
        let result = self.below(visit);
        self.cursor.tree.pop();
        self.cursor.path = path;
        self.cursor.tied = previous;
        result
    }

    /// Declares `prefix` (relative to the current name) known-unused.
    pub fn ignore(&mut self, prefix: &str) {
        let name = Self::join(&[self.name(), prefix.into()]);
        if let Some(strict) = &mut self.strict {
            strict.ignore.push(name);
        }
    }

    /// Every traversal of `source`: the outermost one runs `visit` and waits for every copy its
    /// leaves submitted, then fails with the first copy that failed, else as `visit` failed, else
    /// with each name below its root that nothing read or ignored; nested ones only run `visit`.
    pub fn checkpoint<R>(
        &mut self,
        source: &Checkpoint,
        visit: impl FnOnce(&mut Self) -> Result<R, LoadError>,
    ) -> Result<R, LoadError> {
        if self.strict.is_some() {
            return visit(self);
        }
        self.strict = Some(Strict {
            root: self.name(),
            start: self.read.len(),
            ignore: Vec::new(),
            copies: Vec::new(),
        });
        let result = visit(self);
        let Strict {
            root,
            start,
            ignore,
            copies,
        } = self.strict.take().expect("set above");
        Self::wait(copies)?;
        let value = result?;
        let read: BTreeSet<&String> = self.read[start..].iter().collect();
        let unexpected: Vec<String> = source
            .names()
            .into_iter()
            .filter(|name| {
                Self::under(name, &root)
                    && !read.contains(name)
                    && !ignore.iter().any(|prefix| Self::under(name, prefix))
            })
            .collect();
        match unexpected.is_empty() {
            true => Ok(value),
            false => Err(LoadError::Unexpected(unexpected)),
        }
    }

    /// One rule of an update at the current tree path: `build` its new value, then `write` it.
    /// `Missing` reports the rule as waiting and writes nothing; a report is kept once it wrote.
    pub fn rule<V, R>(
        &mut self,
        build: impl FnOnce(&mut Self) -> Result<V, LoadError>,
        write: impl FnOnce(&mut Self, V) -> Result<R, LoadError>,
    ) -> Result<Option<R>, LoadError> {
        let (tree, start) = (self.path(), self.read.len());
        let built = build(self);
        let required = self.read[start..]
            .iter()
            .fold(Vec::new(), |mut names, name| {
                if !names.contains(name) {
                    names.push(name.clone());
                }
                names
            });
        match built {
            Ok(value) => {
                let written = write(self, value)?;
                self.rules.push(Rule {
                    tree,
                    required,
                    absent: Vec::new(),
                });
                Ok(Some(written))
            }
            Err(LoadError::Missing(absent)) => {
                self.rules.push(Rule {
                    tree,
                    required,
                    absent,
                });
                Ok(None)
            }
            Err(other) => Err(other),
        }
    }

    /// Runs `read` for each of `N` layers under its index, every HBM leaf it uploads filling its
    /// layer of one backing; then `stack`s the layers, whose leaves [`Context::widen`] takes back
    /// to their backings. Absent names of every layer are reported together.
    pub fn layers<L, R, const N: usize>(
        &mut self,
        mut read: impl FnMut(&mut Self) -> Result<L, LoadError>,
        stack: impl FnOnce([L; N], &mut Self) -> R,
    ) -> Result<R, LoadError> {
        assert!(self.stack.is_none(), "a stack's layer holds no stack");
        self.stack = Some(Stack {
            count: N,
            layer: 0,
            backings: BTreeMap::new(),
        });
        let mut children = Children::default();
        let built = (0..N)
            .map(|layer| {
                self.stack.as_mut().expect("set above").layer = layer;
                self.cursor.path.push(layer.to_string());
                let loaded = read(self);
                self.cursor.path.pop();
                children.ask(loaded)
            })
            .collect::<Result<Vec<_>, _>>();
        let stacked = built
            .and_then(|built| children.node(built.into_iter().collect::<Option<Vec<L>>>()))
            .map(|built| stack(built.try_into().ok().expect("one layer per index"), self));
        self.stack = None;
        stacked
    }

    /// The backing whose first layer `buffer` holds, as a stacking traversal filled it.
    pub(crate) fn widen(&mut self, buffer: HbmBuffer) -> HbmBuffer {
        let backings = &mut self
            .stack
            .as_mut()
            .expect("a stack widens its layers")
            .backings;
        let path = backings
            .iter()
            .find(|(_, backing)| backing.addr() == buffer.addr())
            .map(|(path, _)| path.clone())
            .expect("a stacked leaf is the first layer of its backing");
        backings.remove(&path).expect("found above")
    }

    /// `host` copied into HBM as the leaf at the current position: a new allocation, or in a stack
    /// the current layer of the leaf's backing, which the leaf's first layer allocates.
    pub(crate) fn upload<D: MaterializableScalar, C: M, S: M, E: M>(
        &mut self,
        host: &HostTensor<D, E>,
    ) -> Result<HbmTensor<D, C, S>, LoadError> {
        let path = self.path();
        let failed = |source| LoadError::Device {
            path: path.clone(),
            source,
        };
        let device = self.device.as_deref_mut();
        let device = device.expect("an HBM leaf loads through a context with a device");
        let mut leaf = match &mut self.stack {
            None => HbmTensor::alloc(device).map_err(failed)?,
            Some(stack) => {
                // Each chip holds its share of every layer, layer after layer.
                let bytes = HbmTensor::<D, C, S>::size() / C::SIZE;
                let backing = match stack.backings.get(&path) {
                    Some(backing) => backing,
                    None => {
                        let backing =
                            HbmBuffer::alloc(device, bytes * stack.count).map_err(failed)?;
                        stack.backings.entry(path.clone()).or_insert(backing)
                    }
                };
                let at = stack.layer * bytes;
                (backing.alias(at..at + bytes))
                    .and_then(HbmTensor::try_from)
                    .map_err(failed)?
            }
        };
        let copy = host
            .to_hbm::<C, S>(&mut device.pdma)
            .output(&mut leaf)
            .submit();
        self.submitted(copy)?;
        Ok(leaf)
    }

    /// Takes `copy`, submitted into the storage of the leaf at the current position: the outermost
    /// checkpoint traversal waits for it, and fails with it as the leaf's failure; a leaf loaded
    /// outside one waits for it here. The submitted copy holds the host bytes it reads.
    pub(crate) fn submitted(
        &mut self,
        copy: impl Future<Output = Result<(), furiosa_opt_std::Error>> + 'static,
    ) -> Result<(), LoadError> {
        let path = self.path();
        let copy = async move {
            copy.await
                .map_err(|source| LoadError::Device { path, source })
        };
        let Some(strict) = &mut self.strict else {
            return Self::wait([copy]);
        };
        strict.copies.push(Box::pin(copy));
        Ok(())
    }

    /// The reports of the rules this traversal ran.
    pub fn into_rules(self) -> Vec<Rule> {
        self.rules
    }

    /// `visit` one level down, its config reads scoped by the key entered at the current position.
    fn below<R>(&mut self, visit: impl FnOnce(&mut Self) -> R) -> R {
        let entered = self.cursor.config.take();
        let scoped = entered.is_some();
        self.cursor.scope.extend(entered);
        let result = visit(self);
        if scoped {
            self.cursor.config = self.cursor.scope.pop();
        }
        result
    }

    /// Waits for each of `copies` in turn: the first that failed, as the failure of its leaf.
    fn wait(
        copies: impl IntoIterator<Item: Future<Output = Result<(), LoadError>>>,
    ) -> Result<(), LoadError> {
        futures_executor::block_on(task::unconstrained(async move {
            let mut first = Ok(());
            for copy in copies {
                let ended = copy.await;
                first = first.and(ended);
            }
            first
        }))
    }

    /// What `lookup` finds at the current name, and that name; recorded as read, `Missing` under
    /// it.
    fn find<T>(
        &mut self,
        lookup: impl FnOnce(&str) -> Option<T>,
    ) -> Result<(String, T), LoadError> {
        let name = self.name();
        self.read(&name);
        match lookup(&name) {
            Some(found) => Ok((name, found)),
            None => Err(LoadError::Missing(vec![name])),
        }
    }

    fn join(segments: &[String]) -> String {
        segments
            .iter()
            .filter(|segment| !segment.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(".")
    }

    /// Whether `name` is `prefix` or below it.
    fn under(name: &str, prefix: &str) -> bool {
        prefix.is_empty()
            || name
                .strip_prefix(prefix)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
    }
}
