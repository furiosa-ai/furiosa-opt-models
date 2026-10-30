//! A checkpoint: tensors by name, mapped from a model's or a pipeline's files or given in memory,
//! and its config.
//!
//! - A model's folder holds the files its `model.safetensors.index.json` names, else its
//!   `.safetensors` files, and `config.json`.
//! - A pipeline's folder holds `model_index.json` and a folder per component, nested or not: a
//!   tensor's name is its folder's path, dotted, then its own (`video_vae.source.decoder..`), and
//!   the config is `model_index.json` with each folder's `config.json` at the folder's path.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use furiosa_opt_std::prelude::Device;
use memmap2::Mmap;
use safetensors::Dtype;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

use super::{Context, Load, LoadError};
use crate::tensor::form::fp8::Granularity;

/// Tensors by name and the config of one model or pipeline. A tree loads from it into HBM tensor
/// by tensor; an empty one gathers the chunks of an update.
#[derive(Default)]
pub struct Checkpoint {
    pub(super) config: serde_json::Value,
    pub(super) tensors: BTreeMap<String, Stored>,
    /// The folder the checkpoint was opened from, which holds its tokenizer; empty in memory.
    pub(super) directory: PathBuf,
    /// How a load quantizes the projections the checkpoint stores in BF16, if it does.
    pub(super) quantization: Option<Granularity>,
    /// The part of the model a load from the checkpoint reads.
    pub(super) part: Part,
}

/// The part of a model a load reads from a checkpoint; a registry implementation reads one kind
/// of part, and refuses a checkpoint read for another.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Part {
    #[default]
    Whole,
    /// The media encoder alone: the towers or embedders whose rows the language model reads.
    Encoder,
}

/// A tensor's dtype, shape, and where its little-endian bytes are: a range of a mapped file or of a
/// buffer, or nowhere for a tensor declared without values.
#[derive(Clone)]
pub(super) struct Stored {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Option<(Bytes, Range<usize>)>,
}

/// The bytes a stored tensor's range indexes: a mapped file or a buffer.
type Bytes = Arc<dyn AsRef<[u8]> + Send + Sync>;

impl Checkpoint {
    /// The checkpoint of a model's or a pipeline's `directory`, its tensors mapped, and `config`,
    /// or the directory's own when it is `None`. A name in two files is refused.
    pub fn new(directory: impl AsRef<Path>, config: Option<serde_json::Value>) -> io::Result<Self> {
        Self::open(directory.as_ref(), config, |path| {
            // SAFETY: nothing writes the checkpoint's files while it is open.
            let mapping = Arc::new(unsafe { Mmap::map(&File::open(path)?)? });
            let file = SafeTensors::deserialize(&mapping[..])
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
            let mapped = file.tensors().into_iter().map(|(name, view)| {
                let start = view.data().as_ptr().addr() - mapping.as_ptr().addr();
                let stored = Stored {
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    bytes: Some((mapping.clone(), start..start + view.data().len())),
                };
                (name, stored)
            });
            Ok(mapped.collect())
        })
    }

    /// This checkpoint with every tensor's bytes copied into host memory, where neither the page
    /// cache's eviction nor a later change to its files reaches them.
    pub fn read(&self) -> Self {
        let read = |(name, stored): (&String, &Stored)| {
            let bytes = stored.view(name).data().to_vec();
            let range = 0..bytes.len();
            let stored = Stored {
                dtype: stored.dtype,
                shape: stored.shape.clone(),
                bytes: Some((Arc::new(bytes), range)),
            };
            (name.clone(), stored)
        };
        Self {
            config: self.config.clone(),
            tensors: self.tensors.iter().map(read).collect(),
            directory: self.directory.clone(),
            quantization: self.quantization,
            part: self.part,
        }
    }

    /// The checkpoint of a model's or a pipeline's `directory` without values: its config, and each
    /// tensor's dtype and shape from the header of each of its files, as `new` finds them; a file
    /// may hold its header alone, and a read of values panics.
    #[cfg(any(test, feature = "test-support"))]
    pub fn declared(directory: impl AsRef<Path>) -> io::Result<Self> {
        use std::io::Read;
        Self::open(directory.as_ref(), None, |path| {
            let mut file = File::open(path)?;
            let mut length = [0; 8];
            file.read_exact(&mut length)?;
            let mut header = Vec::new();
            file.take(u64::from_le_bytes(length))
                .read_to_end(&mut header)?;
            let header: safetensors::tensor::Metadata = serde_json::from_slice(&header)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            let declared = header.tensors().into_iter().map(|(name, info)| {
                let stored = Stored {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    bytes: None,
                };
                (name, stored)
            });
            Ok(declared.collect())
        })
    }

    /// The same tensors and config, read as a framework quantizes a BF16 checkpoint online: each
    /// projection that holds E4M3 codes quantizes its BF16 weight under `granularity`. A config
    /// that declares a quantization is refused: such a checkpoint loads its own weights.
    pub fn quantized(&self, granularity: Granularity) -> Result<Self, LoadError> {
        if self.declaration().is_some() {
            return Err(LoadError::InvalidData {
                name: "config.json:quantization_config".into(),
                reason: "the checkpoint is quantized already, so it loads its own weights".into(),
            });
        }
        Ok(Self {
            config: self.config.clone(),
            tensors: self.tensors.clone(),
            directory: self.directory.clone(),
            quantization: Some(granularity),
            part: self.part,
        })
    }

    /// How a load quantizes the projections the checkpoint stores in BF16, if it does.
    pub fn quantization(&self) -> Option<Granularity> {
        self.quantization
    }

    /// The same tensors and config, which a load reads for the model's media encoder alone.
    pub fn encoder(&self) -> Self {
        Self {
            config: self.config.clone(),
            tensors: self.tensors.clone(),
            directory: self.directory.clone(),
            quantization: self.quantization,
            part: Part::Encoder,
        }
    }

    /// The part of the model a load from the checkpoint reads.
    pub fn part(&self) -> Part {
        self.part
    }

    /// The quantization the config declares for the stored weights: none when its
    /// `quantization_config` is absent or null, as Transformers reads it.
    pub fn declaration(&self) -> Option<&serde_json::Value> {
        (self.config.get("quantization_config")).filter(|declared| !declared.is_null())
    }

    /// `T` loaded strictly from the whole checkpoint into HBM of `device`.
    pub fn load<T>(&self, device: &mut Device) -> Result<T, LoadError>
    where
        T: for<'a> Load<&'a Self>,
    {
        T::load(&mut Context::new(device), self)
    }

    /// The tensor `name`.
    pub fn get(&self, name: &str) -> Option<TensorView<'_>> {
        self.tensors.get(name).map(|stored| stored.view(name))
    }

    /// Tensor `name`'s header entry, its dtype and shape, read without its values.
    pub(crate) fn header(&self, name: &str) -> Option<(Dtype, &[usize])> {
        (self.tensors.get(name)).map(|stored| (stored.dtype, stored.shape.as_slice()))
    }

    /// The names of every tensor, in order.
    pub fn names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    pub fn config(&self) -> &serde_json::Value {
        &self.config
    }

    /// The id of the vocabulary entry `token` in the checkpoint's `tokenizer.json`, an added token
    /// or one of its model's.
    pub fn token(&self, token: &str) -> Result<u32, LoadError> {
        let name = format!("tokenizer.json:{token}");
        let path = self.directory.join("tokenizer.json");
        let invalid = |reason: String| LoadError::InvalidData {
            name: name.clone(),
            reason,
        };
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(LoadError::Missing(vec![name]));
            }
            Err(error) => return Err(invalid(error.to_string())),
        };
        let tokenizer: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|error| invalid(error.to_string()))?;
        let added = tokenizer["added_tokens"].as_array().into_iter().flatten();
        let added = added
            .filter(|entry| entry["content"] == token)
            .find_map(|entry| entry["id"].as_u64());
        let id = added.or_else(|| tokenizer["model"]["vocab"][token].as_u64());
        let id = id.ok_or_else(|| invalid("the vocabulary holds no such token".into()))?;
        u32::try_from(id).map_err(|error| invalid(error.to_string()))
    }

    /// `bytes` parsed as a config, the JSON Python's `json` writes, which spells an infinite float
    /// as a bare `Infinity` or `-Infinity` outside strings; a bare `NaN` stays refused.
    pub fn parse_config(bytes: &[u8]) -> serde_json::Result<serde_json::Value> {
        let mut normalized = Vec::with_capacity(bytes.len());
        let (mut quoted, mut escaped, mut boundary, mut index) = (false, false, true, 0);
        while index < bytes.len() {
            let byte = bytes[index];
            if quoted {
                normalized.push(byte);
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    quoted = false;
                    boundary = false;
                }
                index += 1;
                continue;
            }
            // `1e9999` overflows to an infinite float, and arbitrary-precision numbers keep it as is.
            let infinity = [
                (b"-Infinity".as_slice(), b"-1e9999".as_slice()),
                (b"Infinity".as_slice(), b"1e9999".as_slice()),
            ]
            .into_iter()
            .find(|(spelling, _)| {
                bytes[index..].starts_with(spelling)
                    && boundary
                    && bytes[index + spelling.len()..]
                        .iter()
                        .copied()
                        .find(|v| !v.is_ascii_whitespace())
                        .is_none_or(|v| matches!(v, b',' | b']' | b'}'))
            });
            if let Some((spelling, number)) = infinity {
                normalized.extend_from_slice(number);
                boundary = false;
                index += spelling.len();
            } else {
                normalized.push(byte);
                quoted = byte == b'"';
                if !byte.is_ascii_whitespace() {
                    boundary = matches!(byte, b':' | b'[' | b',');
                }
                index += 1;
            }
        }
        serde_json::from_slice(&normalized)
    }

    /// Adds the tensor `name`; a second tensor of one name, or bytes that do not fill `shape`, are
    /// refused.
    pub fn insert(
        &mut self,
        name: &str,
        dtype: Dtype,
        shape: &[usize],
        data: Vec<u8>,
    ) -> Result<(), LoadError> {
        let invalid = |reason: String| LoadError::InvalidData {
            name: name.into(),
            reason,
        };
        TensorView::new(dtype, shape.to_vec(), &data)
            .map_err(|error| invalid(error.to_string()))?;
        if self.tensors.contains_key(name) {
            return Err(invalid("already in the checkpoint".into()));
        }
        let range = 0..data.len();
        let stored = Stored {
            dtype,
            shape: shape.to_vec(),
            bytes: Some((Arc::new(data), range)),
        };
        self.tensors.insert(name.into(), stored);
        Ok(())
    }

    /// Removes the tensor `name`; whether the checkpoint held it.
    #[cfg(any(test, feature = "test-support"))]
    pub fn remove(&mut self, name: &str) -> bool {
        self.tensors.remove(name).is_some()
    }

    /// The checkpoint of a model's or a pipeline's `directory`, the tensors `read` finds in each of
    /// its files, and `config`, or the directory's own when it is `None`. A name in two files, and
    /// a file `read` refuses as invalid data, are refused naming the file.
    fn open(
        directory: &Path,
        config: Option<serde_json::Value>,
        read: impl Fn(&Path) -> io::Result<Vec<(String, Stored)>>,
    ) -> io::Result<Self> {
        /// Every folder below `root/relative` but hidden ones, each before the folders in it.
        fn folders(root: &Path, relative: &Path) -> io::Result<Vec<PathBuf>> {
            let mut entries = fs::read_dir(root.join(relative))?
                .map(|entry| Ok(relative.join(entry?.file_name())))
                .collect::<io::Result<Vec<PathBuf>>>()?;
            entries.sort();
            let hidden = |path: &PathBuf| {
                path.file_name()
                    .map(|name| name.as_encoded_bytes().starts_with(b"."))
            };
            entries.retain(|path| root.join(path).is_dir() && hidden(path) == Some(false));
            let nested = entries.into_iter().map(|folder| {
                let below = folders(root, &folder)?;
                Ok([vec![folder], below].concat())
            });
            Ok(nested.collect::<io::Result<Vec<_>>>()?.concat())
        }
        let invalid = |path: &Path, error: String| {
            let error = format!("{}: {error}", path.display());
            io::Error::new(io::ErrorKind::InvalidData, error)
        };
        let parse = |path: &Path| -> io::Result<serde_json::Value> {
            Self::parse_config(&fs::read(path)?).map_err(|error| invalid(path, error.to_string()))
        };
        let model_index = directory.join("model_index.json");
        let pipeline = model_index.try_exists()?;
        let folders = match pipeline {
            true => folders(directory, Path::new(""))?,
            false => vec![PathBuf::new()],
        };
        let segments = |folder: &Path| {
            let segments: Option<Vec<String>> =
                folder.iter().map(|s| Some(s.to_str()?.into())).collect();
            segments.ok_or_else(|| invalid(&directory.join(folder), "a name not in UTF-8".into()))
        };
        let mut tensors = BTreeMap::new();
        for folder in &folders {
            let place = directory.join(folder);
            let prefix: String = segments(folder)?.into_iter().map(|s| s + ".").collect();
            // A hub snapshot may also hold the indexed weights under another layout, which
            // Transformers does not read (`consolidated.safetensors`).
            let index = place.join("model.safetensors.index.json");
            let mut files: Vec<_> = match fs::read(&index) {
                Ok(bytes) => {
                    let document = Self::parse_config(&bytes)
                        .map_err(|error| invalid(&index, error.to_string()))?;
                    let shards = document["weight_map"].as_object().ok_or_else(|| {
                        invalid(&index, "holds no weight_map of names to files".into())
                    })?;
                    let named = shards.values().map(|file| {
                        file.as_str()
                            .map(|file| place.join(file))
                            .ok_or_else(|| invalid(&index, format!("{file} is no file name")))
                    });
                    let named: io::Result<BTreeSet<_>> = named.collect();
                    named?.into_iter().collect()
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => fs::read_dir(&place)?
                    .map(|entry| Ok(entry?.path()))
                    .filter(|path| {
                        path.as_ref()
                            .is_ok_and(|path| path.extension().is_some_and(|e| e == "safetensors"))
                    })
                    .collect::<io::Result<_>>()?,
                Err(error) => return Err(error),
            };
            files.sort();
            for path in files {
                let named = |error: io::Error| match error.kind() {
                    io::ErrorKind::InvalidData => invalid(&path, error.to_string()),
                    _ => error,
                };
                for (name, stored) in read(&path).map_err(named)? {
                    let name = format!("{prefix}{name}");
                    if tensors.insert(name.clone(), stored).is_some() {
                        return Err(invalid(&path, format!("{name} is in another file too")));
                    }
                }
            }
        }
        let config = match (config, pipeline) {
            (Some(config), _) => config,
            (None, false) => parse(&directory.join("config.json"))?,
            (None, true) => {
                let mut document = parse(&model_index)?;
                for folder in &folders {
                    let path = directory.join(folder).join("config.json");
                    if !path.try_exists()? {
                        continue;
                    }
                    let mut place = &mut document;
                    for segment in segments(folder)? {
                        if !(place.is_object() || place.is_null()) {
                            let nested = "nests in a component configured by no file";
                            return Err(invalid(&path, nested.into()));
                        }
                        place = &mut place[segment.as_str()];
                    }
                    *place = parse(&path)?;
                }
                document
            }
        };
        Ok(Self {
            config,
            tensors,
            directory: directory.into(),
            quantization: None,
            part: Part::Whole,
        })
    }
}

impl Stored {
    /// The tensor stored as `name`, with its values; a tensor declared without values has none to
    /// read.
    pub(super) fn view(&self, name: &str) -> TensorView<'_> {
        let Some((bytes, range)) = &self.bytes else {
            panic!(
                "{name} is declared without values: its checkpoint holds names, dtypes and shapes"
            );
        };
        let view = TensorView::new(
            self.dtype,
            self.shape.clone(),
            &(**bytes).as_ref()[range.clone()],
        );
        view.expect("checked when stored")
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::future::IntoFuture;

    use super::*;
    use crate::load::Load;
    use crate::tensor::memory::Memory;
    use furiosa_opt_std::prelude::*;

    axes![V = 8, H = 4];

    /// A leaf on host: the value a checkpoint tensor converts to, which an HBM leaf uploads. The
    /// write of a tensor stored from byte `0xff` submits a copy that fails as a device failing
    /// after it took the copy, one from `0xfe` a copy that fails as a device refusing it before.
    pub(in crate::load) struct Value<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
        pub(in crate::load) HostTensor<D, m![{ C }, { S }]>,
    );

    impl<D: MaterializableScalar, C: M + 'static, S: M + 'static> Value<D, C, S> {
        pub(in crate::load) fn values(&self) -> Vec<D> {
            self.0.clone().into_vec()
        }
    }

    /// The memory of [`Value`] leaves: trees the tests load without a device.
    pub(in crate::load) struct Values;

    impl Memory for Values {
        type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Value<D, C, S>;
    }

    impl<'a, D: MaterializableScalar, C: M + 'static, S: M + 'static> Load<&'a Checkpoint>
        for Value<D, C, S>
    {
        fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
            let split = |_: &Context<'_>, leaf: Self| Ok((Value(leaf.0.clone()), leaf));
            context.checkpoint(from, |cx| {
                cx.leaf(from, |cx| from.host::<D, S, _>(cx).map(Value), split)
            })
        }
    }

    impl<'t, 's, D: MaterializableScalar, C: M + 'static, S: M + 'static>
        Load<(&'t mut Value<D, C, S>, &'s Checkpoint)> for &'t mut Value<D, C, S>
    {
        fn load(
            context: &mut Context<'_>,
            (into, from): (&'t mut Value<D, C, S>, &'s Checkpoint),
        ) -> Result<Self, LoadError> {
            let write = |cx: &mut Context<'_>| {
                let new = Value(from.host::<D, S, _>(cx)?);
                let first = from
                    .get(&cx.name())
                    .and_then(|view| view.data().first().copied());
                let mut failed = |error| cx.submitted(std::future::ready(Err(error)));
                match first {
                    Some(0xff) => failed(furiosa_opt_std::Error::Poisoned),
                    Some(0xfe) => failed(furiosa_opt_std::Error::Invalid("refused")),
                    _ => Ok(*into = new),
                }
            };
            context.checkpoint(from, |cx| {
                cx.rule(
                    |cx| cx.overwrite::<Value<D, C, S>>(from, write),
                    |_, ()| Ok(()),
                )
            })?;
            Ok(into)
        }
    }

    impl<'t, D: MaterializableScalar, C: M + 'static, S: M + 'static>
        Load<(&'t mut Value<D, C, S>, Value<D, C, S>)> for &'t mut Value<D, C, S>
    {
        fn load(
            _: &mut Context<'_>,
            (into, from): (&'t mut Value<D, C, S>, Value<D, C, S>),
        ) -> Result<Self, LoadError> {
            *into = from;
            Ok(into)
        }
    }

    #[derive(crate::load::Load)]
    pub(in crate::load) struct Embedding<X: Memory> {
        pub(in crate::load) weight: X::Tensor<bf16, m![1], m![V, H]>,
    }

    #[derive(crate::load::Load)]
    pub(in crate::load) struct Body<X: Memory> {
        pub(in crate::load) embed_tokens: Embedding<X>,
        #[load(config = "rms_norm_eps")]
        pub(in crate::load) epsilon: X::Tensor<f32, m![1], m![1 # 8]>,
    }

    #[derive(crate::load::Load)]
    pub(in crate::load) struct Causal<X: Memory> {
        pub(in crate::load) model: Body<X>,
        #[load(tied = "model.embed_tokens")]
        pub(in crate::load) lm_head: Embedding<X>,
    }

    pub(in crate::load) const EMBED: &[usize] = &[8, 4];

    /// BF16 bytes of `count` elements valued `value`.
    pub(in crate::load) fn bf16s(count: usize, value: f32) -> Vec<u8> {
        half::bf16::from_f32(value).to_le_bytes().repeat(count)
    }

    /// A checkpoint in memory of `tensors`, without a config: a chunk of an update.
    pub(in crate::load) fn checkpoint(tensors: &[(&str, Dtype, &[usize], Vec<u8>)]) -> Checkpoint {
        let mut checkpoint = Checkpoint::default();
        for (name, dtype, shape, data) in tensors {
            checkpoint
                .insert(name, *dtype, shape, data.clone())
                .unwrap();
        }
        checkpoint
    }

    /// A checkpoint directory of `config` and one `.safetensors` file per group of `files`.
    pub(in crate::load) fn directory(
        config: serde_json::Value,
        files: &[&[(&str, Dtype, &[usize], Vec<u8>)]],
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("config.json"), config.to_string()).unwrap();
        for (index, tensors) in files.iter().enumerate() {
            let views: Vec<_> = tensors
                .iter()
                .map(|(name, dtype, shape, data)| {
                    let view = TensorView::new(*dtype, shape.to_vec(), data);
                    (name.to_string(), view.unwrap())
                })
                .collect();
            let file = directory.path().join(format!("model-{index}.safetensors"));
            safetensors::serialize_to_file(views, &None, &file).unwrap();
        }
        directory
    }

    /// The names `error` lists as missing.
    pub(in crate::load) fn missing(error: Option<LoadError>) -> Vec<String> {
        match error {
            Some(LoadError::Missing(names)) => names,
            other => panic!("{other:?} is not `Missing`"),
        }
    }

    /// The names `error` lists as unexpected.
    pub(in crate::load) fn unexpected(error: Option<LoadError>) -> Vec<String> {
        match error {
            Some(LoadError::Unexpected(names)) => names,
            other => panic!("{other:?} is not `Unexpected`"),
        }
    }

    /// The name `error` refuses as invalid data, and why.
    pub(in crate::load) fn invalid(error: Option<LoadError>) -> (String, String) {
        match error {
            Some(LoadError::InvalidData { name, reason }) => (name, reason),
            other => panic!("{other:?} is not `InvalidData`"),
        }
    }

    pub(in crate::load) fn device() -> Device {
        Device::new(Topology { chips: 1, pes: 8 }).unwrap()
    }

    /// The values of an HBM leaf, read back laid out as `E`.
    pub(in crate::load) fn host<D: MaterializableScalar, C: M, S: M, E: M>(
        leaf: &HbmTensor<D, C, S>,
        device: &mut Device,
    ) -> HostTensor<D, E> {
        futures_executor::block_on(leaf.to_host::<E>(&mut device.pdma).into_future()).unwrap()
    }

    /// The directory of a tied model's checkpoint: the embedding valued `value`, and the norm's
    /// epsilon in the config.
    pub(in crate::load) fn causal(value: f32) -> tempfile::TempDir {
        let embedding = [(
            "model.embed_tokens.weight",
            Dtype::BF16,
            EMBED,
            bf16s(32, value),
        )];
        directory(serde_json::json!({"rms_norm_eps": 0.5}), &[&embedding])
    }

    /// A tree of [`Value`] leaves loaded strictly from `source`.
    pub(in crate::load) fn load<T: for<'a> Load<&'a Checkpoint>>(
        source: &Checkpoint,
    ) -> Result<T, LoadError> {
        T::load(&mut Context::new(None), source)
    }

    /// A load reads strictly across every file: an absent tensor is missing once, though two
    /// rules read it, and a tensor no rule reads in another file is unexpected.
    #[test]
    fn loads_strictly_across_files() {
        let files = causal(1.0);
        let extra = [("model.extra", Dtype::BF16, &[1][..], bf16s(1, 0.0))];
        let embedding = [(
            "model.embed_tokens.weight",
            Dtype::BF16,
            EMBED,
            bf16s(32, 1.0),
        )];
        let config = serde_json::json!({"rms_norm_eps": 0.5});
        let two = directory(config.clone(), &[&embedding, &extra]);
        let error = load::<Causal<Values>>(&Checkpoint::new(two.path(), None).unwrap()).err();
        assert_eq!(unexpected(error), ["model.extra"]);
        let empty = directory(config, &[]);
        let error = load::<Causal<Values>>(&Checkpoint::new(empty.path(), None).unwrap()).err();
        assert_eq!(missing(error), ["model.embed_tokens.weight"]);
        assert!(load::<Causal<Values>>(&Checkpoint::new(files.path(), None).unwrap()).is_ok());
    }

    /// Leaves convert float tensors to their scalar and pad config values out to their mapping;
    /// other dtypes and shapes, and a config value that is no number, are refused.
    #[test]
    fn leaves_convert_and_pad() {
        let f32s = 2.0f32.to_le_bytes().repeat(32);
        let embedding = [("embed_tokens.weight", Dtype::F32, EMBED, f32s)];
        let files = directory(serde_json::json!({"rms_norm_eps": 0.5}), &[&embedding]);
        let body: Body<Values> = load(&Checkpoint::new(files.path(), None).unwrap()).unwrap();
        assert_eq!(
            body.embed_tokens.weight.values(),
            vec![bf16::from_f32(2.0); 32]
        );
        assert_eq!(
            body.epsilon.values(),
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        let ints = checkpoint(&[("weight", Dtype::I32, EMBED, vec![0; 128])]);
        let error = load::<Embedding<Values>>(&ints).err();
        assert_eq!(
            invalid(error),
            ("weight".into(), "stored as I32, the leaf holds BF16".into())
        );
        let worded = [("embed_tokens.weight", Dtype::BF16, EMBED, bf16s(32, 0.0))];
        let worded = directory(serde_json::json!({"rms_norm_eps": "small"}), &[&worded]);
        let error = load::<Body<Values>>(&Checkpoint::new(worded.path(), None).unwrap()).err();
        assert_eq!(
            invalid(error),
            (
                "config.json:rms_norm_eps".into(),
                r#""small" is not a number"#.into()
            )
        );
        let wide = checkpoint(&[("weight", Dtype::BF16, &[8, 5], bf16s(40, 0.0))]);
        let error = load::<Embedding<Values>>(&wide).err();
        assert_eq!(
            invalid(error),
            (
                "weight".into(),
                "shape [8, 5], the leaf holds [8, 4]".into()
            )
        );
    }

    /// A directory opens as every tensor of each of its files, mapped, and its config unless one is
    /// given; a file that is no safetensors file and a name two files hold are refused.
    #[test]
    fn opens_every_file_of_a_directory() {
        let first = [("a", Dtype::U8, &[2][..], vec![1, 2])];
        let second = [("b", Dtype::BF16, &[1][..], bf16s(1, 3.0))];
        let files = directory(serde_json::json!({"k": 1}), &[&first, &second]);
        let opened = Checkpoint::new(files.path(), None).unwrap();
        assert_eq!(opened.names(), ["a", "b"]);
        assert_eq!(opened.get("a").unwrap().data(), [1, 2]);
        assert_eq!(opened.get("b").unwrap().dtype(), Dtype::BF16);
        assert_eq!(opened.config(), &serde_json::json!({"k": 1}));
        let overridden = Checkpoint::new(files.path(), Some(serde_json::json!({"k": 2}))).unwrap();
        assert_eq!(overridden.config(), &serde_json::json!({"k": 2}));
        let truncated = tempfile::tempdir().unwrap();
        std::fs::write(truncated.path().join("config.json"), "{}").unwrap();
        std::fs::write(truncated.path().join("model.safetensors"), [0; 3]).unwrap();
        let error = Checkpoint::new(truncated.path(), None).err().unwrap();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            error
                .to_string()
                .ends_with("model.safetensors: HeaderTooSmall"),
            "{error}"
        );
        let twice = directory(serde_json::json!({}), &[&first, &first]);
        let error = Checkpoint::new(twice.path(), None).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(
            error.to_string().ends_with("a is in another file too"),
            "{error}"
        );
    }

    /// A read checkpoint keeps every tensor and its config once its files are rewritten.
    #[test]
    fn read_outlives_its_files() {
        let tensors = [("a", Dtype::U8, &[2][..], vec![1, 2])];
        let files = directory(serde_json::json!({"k": 1}), &[&tensors]);
        let read = Checkpoint::new(files.path(), None).unwrap().read();
        fs::write(files.path().join("model-0.safetensors"), b"").unwrap();
        assert_eq!(read.names(), ["a"]);
        assert_eq!(read.get("a").unwrap().data(), [1, 2]);
        assert_eq!(read.config(), &serde_json::json!({"k": 1}));
    }

    /// A directory with an index opens the files the index names, not another layout of them; an
    /// index without a map of names to file names is refused.
    #[test]
    fn opens_the_files_its_index_names() {
        let first = [("a", Dtype::U8, &[2][..], vec![1, 2])];
        let files = directory(serde_json::json!({}), &[&first]);
        let index = files.path().join("model.safetensors.index.json");
        let named = serde_json::json!({"weight_map": {"a": "model-0.safetensors"}});
        fs::write(&index, named.to_string()).unwrap();
        let other = TensorView::new(Dtype::U8, vec![2], &[1, 2]).unwrap();
        let consolidated = files.path().join("consolidated.safetensors");
        safetensors::serialize_to_file([("a", other)], &None, &consolidated).unwrap();
        let opened = Checkpoint::new(files.path(), None).unwrap();
        assert_eq!(opened.names(), ["a"]);
        let unmapped = "holds no weight_map of names to files";
        for (document, reason) in [
            (serde_json::json!({}), unmapped),
            (
                serde_json::json!({"weight_map": ["model-0.safetensors"]}),
                unmapped,
            ),
            (
                serde_json::json!({"weight_map": {"a": 0}}),
                "0 is no file name",
            ),
        ] {
            std::fs::write(&index, document.to_string()).unwrap();
            let error = Checkpoint::new(files.path(), None).err().unwrap();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
            let expected = format!("model.safetensors.index.json: {reason}");
            assert!(error.to_string().ends_with(&expected), "{error}");
        }
    }

    /// A pipeline opens as its folders' tensors, named after their folders, and its index with
    /// each folder's configuration at the folder's path; a configuration below a component the
    /// index declares as no object, with no file of its own, and a folder not named in UTF-8 are
    /// refused.
    #[test]
    fn opens_a_pipeline_as_its_components() {
        let files = tempfile::tempdir().unwrap();
        let index = serde_json::json!({"_class_name": "P", "a": ["lib", "A"], "b": ["lib", "B"]});
        fs::write(files.path().join("model_index.json"), index.to_string()).unwrap();
        let components = [
            ("a", Some(serde_json::json!({"x": 1})), Some("w")),
            ("b", Some(serde_json::json!({"z": 3})), None),
            ("b/c", Some(serde_json::json!({"y": 2})), Some("v")),
            (".cache", None, Some("hidden")),
        ];
        for (folder, config, tensor) in components {
            let place = files.path().join(folder);
            fs::create_dir_all(&place).unwrap();
            if let Some(config) = config {
                fs::write(place.join("config.json"), config.to_string()).unwrap();
            }
            if let Some(name) = tensor {
                let view = TensorView::new(Dtype::U8, vec![1], &[7]).unwrap();
                let file = place.join("model.safetensors");
                safetensors::serialize_to_file([(name, view)], &None, &file).unwrap();
            }
        }
        let opened = Checkpoint::new(files.path(), None).unwrap();
        assert_eq!(opened.names(), ["a.w", "b.c.v"]);
        let config = serde_json::json!({
            "_class_name": "P",
            "a": {"x": 1},
            "b": {"z": 3, "c": {"y": 2}},
        });
        assert_eq!(opened.config(), &config);
        let nested = tempfile::tempdir().unwrap();
        let index = serde_json::json!({"_class_name": "P", "a": ["lib", "A"]});
        std::fs::write(nested.path().join("model_index.json"), index.to_string()).unwrap();
        let below = nested.path().join("a").join("c");
        std::fs::create_dir_all(&below).unwrap();
        std::fs::write(below.join("config.json"), "{}").unwrap();
        let error = Checkpoint::new(nested.path(), None).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
        let reason = "a/c/config.json: nests in a component configured by no file";
        assert!(error.to_string().ends_with(reason), "{error}");
        let unnamed = tempfile::tempdir().unwrap();
        std::fs::write(unnamed.path().join("model_index.json"), "{}").unwrap();
        let folder = <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(b"\xff");
        std::fs::create_dir(unnamed.path().join(folder)).unwrap();
        let error = Checkpoint::new(unnamed.path(), None).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
        let reason = "\u{fffd}: a name not in UTF-8";
        assert!(error.to_string().ends_with(reason), "{error}");
    }

    /// The float a JSON number's text reads as; `as_f64` has no infinite value.
    fn number(value: &serde_json::Value) -> Option<f32> {
        value.as_number()?.to_string().parse().ok()
    }

    /// A declared checkpoint's dtypes and shapes, read from a safetensors file holding its header
    /// alone, load strictly through `Shapes`, and a load of its values panics, naming the tensor.
    #[test]
    #[should_panic(expected = "embed_tokens.weight is declared without values")]
    fn declared_tensors_hold_no_values() {
        let files = tempfile::tempdir().unwrap();
        let write =
            |file: &str, bytes: &[u8]| std::fs::write(files.path().join(file), bytes).unwrap();
        write("config.json", br#"{"rms_norm_eps": 0.5}"#);
        let header =
            br#"{"embed_tokens.weight":{"dtype":"BF16","shape":[8,4],"data_offsets":[0,64]}}"#;
        let length = u64::try_from(header.len()).unwrap().to_le_bytes();
        write("model.safetensors", &[&length[..], header].concat());
        let declared = Checkpoint::declared(files.path()).unwrap();
        assert!(load::<Body<crate::load::Shapes>>(&declared).is_ok());
        let _ = load::<Body<Values>>(&declared);
    }

    /// A config file reads as Python's `json` writes it: a bare `Infinity` is an infinite number,
    /// and a bare `NaN` is invalid data.
    #[test]
    fn reads_the_config_as_python_writes_it() {
        let files = tempfile::tempdir().unwrap();
        let config = files.path().join("config.json");
        fs::write(
            &config,
            r#"{"time_step_limit": [0.0, Infinity], "low": -Infinity}"#,
        )
        .unwrap();
        let opened = Checkpoint::new(files.path(), None).unwrap();
        assert_eq!(
            number(&opened.config()["time_step_limit"][1]),
            Some(f32::INFINITY)
        );
        assert_eq!(number(&opened.config()["low"]), Some(f32::NEG_INFINITY));
        fs::write(&config, r#"{"time_step_limit": [0.0, NaN]}"#).unwrap();
        let error = Checkpoint::new(files.path(), None).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
        let reason = "config.json: expected value at line 1 column 27";
        assert!(error.to_string().ends_with(reason), "{error}");
    }

    /// Only bare infinities change: other values keep their meaning, and `Infinity` inside a
    /// string stays text.
    #[test]
    fn keeps_strings_and_finite_values() {
        let source = concat!(
            r#"{"size":768,"bounds":[0.0,Infinity,-Infinity],"#,
            r#""text":"a\"Infinity b\\NaN","unicode":"\u0049nfinity","null":null}"#,
        );
        let value = Checkpoint::parse_config(source.as_bytes()).unwrap();
        assert_eq!(value["size"].as_u64(), Some(768));
        assert_eq!(number(&value["bounds"][0]), Some(0.0));
        assert_eq!(number(&value["bounds"][1]), Some(f32::INFINITY));
        assert_eq!(number(&value["bounds"][2]), Some(f32::NEG_INFINITY));
        assert_eq!(value["text"], "a\"Infinity b\\NaN");
        assert_eq!(value["unicode"], "Infinity");
        assert_eq!(number(&value["unicode"]), None);
        assert_eq!(number(&value["null"]), None);
    }

    /// An infinity reads only bare, where a value starts and up to a delimiter; `NaN`, other
    /// spellings and malformed documents are refused, each for what JSON finds there.
    #[test]
    fn refuses_nan_and_other_spellings() {
        use serde_json::error::Category::{Eof, Syntax};
        let value = (Syntax, "expected value");
        let number = (Syntax, "invalid number");
        for (source, (category, reason)) in [
            ("[NaN]", value),
            ("[-NaN]", number),
            ("[+NaN]", value),
            ("[Inf]", value),
            ("[+Infinity]", value),
            ("[-Inf]", number),
            ("[Infinityx]", value),
            ("[1Infinity]", (Syntax, "expected `,` or `]`")),
            ("{Infinity:0}", (Syntax, "key must be a string")),
            ("[Infinity,]", (Syntax, "trailing comma")),
            ("[Infinity /* comment */]", value),
            ("[Infinity NaN]", value),
            ("[Infinity]false", (Syntax, "trailing characters")),
            ("{\"a\":Infinity:0}", value),
            (r#"["Infinity]"#, (Eof, "EOF while parsing a string")),
            (r#"["bad\q",Infinity]"#, (Syntax, "invalid escape")),
        ] {
            let error = Checkpoint::parse_config(source.as_bytes()).unwrap_err();
            // The position counts the normalized text, where an infinity is spelled otherwise.
            let message = error.to_string();
            let (found, _) = message.split_once(" at line ").unwrap();
            assert_eq!((error.classify(), found), (category, reason), "{source}");
        }
    }
}
