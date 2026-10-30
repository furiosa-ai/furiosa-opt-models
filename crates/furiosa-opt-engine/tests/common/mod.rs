//! The published checkpoints the engine's tests read: a folder of a Hugging Face hub repository at a
//! pinned commit, its config and its safetensors headers, fetched once into the hub's cache.
//!
//! - The cache is `$HF_HOME`, else `~/.cache/huggingface`, as the hub's own tools find it; a folder's
//!   files sit under `assets/furiosa-opt-engine/<org>--<name>/<revision>`, each safetensors file cut
//!   to its header, so no tensor value is fetched or kept.
//! - A later run reads the cache and opens no connection; a gated repository is read with the token
//!   the hub's login stores in the cache.

#![allow(
    dead_code,
    reason = "each test file compiles this module and calls what it reads"
)]

use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};

use furiosa_opt_engine::Checkpoint;

/// The weights of a model folder stored whole, and the index of weights stored in shards, as
/// Transformers names them.
const WEIGHTS: &str = "model.safetensors";
const INDEX: &str = "model.safetensors.index.json";

/// A folder of a hub repository at a commit.
#[derive(Debug)]
pub struct Published {
    repo: &'static str,
    revision: &'static str,
    folder: &'static str,
}

impl Published {
    /// The root folder of `repo` at the commit `revision`.
    pub const fn new(repo: &'static str, revision: &'static str) -> Self {
        Self {
            repo,
            revision,
            folder: "",
        }
    }

    /// The folder `folder` of this repository, as a pipeline holds a component.
    pub const fn folder(self, folder: &'static str) -> Self {
        Self { folder, ..self }
    }

    /// The folder's config.
    pub fn config(&self) -> serde_json::Value {
        let path = self.cached("config.json", || self.get("config.json", None));
        Checkpoint::parse_config(&std::fs::read(path).unwrap()).unwrap()
    }

    /// The folder's checkpoint of its config alone, holding no tensor: what a choice of model reads.
    pub fn checkpoint(&self) -> Checkpoint {
        let empty = tempfile::tempdir().unwrap();
        Checkpoint::new(empty.path(), Some(self.config())).unwrap()
    }

    /// The folder's checkpoint without values: its config, and the name, dtype and shape of every
    /// tensor of the files its index names, else of its `model.safetensors`, read from the headers.
    pub fn declared(&self) -> Checkpoint {
        self.cached("config.json", || self.get("config.json", None));
        let cached = |file| self.directory().join(file).exists();
        if !cached(INDEX) && !cached(WEIGHTS) {
            match self.get(INDEX, None) {
                // The index is stored last: a cached one means every file it names is cached.
                Some(index) => {
                    let document = Checkpoint::parse_config(&index).unwrap();
                    let files = document["weight_map"].as_object().unwrap().values();
                    let mut files: Vec<&str> = files.map(|file| file.as_str().unwrap()).collect();
                    files.sort_unstable();
                    files.dedup();
                    for file in files {
                        self.cached(file, || self.header(file));
                    }
                    self.store(INDEX, &index);
                }
                None => {
                    self.cached(WEIGHTS, || self.header(WEIGHTS));
                }
            }
        }
        Checkpoint::declared(self.directory()).unwrap()
    }

    /// The hub's cache.
    fn home() -> PathBuf {
        let home = std::env::var_os("HF_HOME").map(PathBuf::from);
        home.unwrap_or_else(|| std::env::home_dir().unwrap().join(".cache/huggingface"))
    }

    /// The folder's directory in the cache.
    fn directory(&self) -> PathBuf {
        let repo = self.repo.replace('/', "--");
        Self::home()
            .join("assets/furiosa-opt-engine")
            .join(repo)
            .join(self.revision)
            .join(self.folder)
    }

    /// The cached `file` of the folder, stored from `fetch` when the cache lacks it.
    fn cached(&self, file: &str, fetch: impl FnOnce() -> Option<Vec<u8>>) -> PathBuf {
        let path = self.directory().join(file);
        if !path.exists() {
            let bytes = fetch().unwrap_or_else(|| panic!("{} holds no {file}", self.repo));
            self.store(file, &bytes);
        }
        path
    }

    /// `bytes` stored as the folder's `file`, whole or not at all, so a reader never sees a part.
    fn store(&self, file: &str, bytes: &[u8]) {
        let path = self.directory().join(file);
        let folder = path.parent().unwrap();
        std::fs::create_dir_all(folder).unwrap();
        let mut written = tempfile::NamedTempFile::new_in(folder).unwrap();
        written.write_all(bytes).unwrap();
        written.persist(&path).unwrap();
    }

    /// The safetensors header of the folder's `file`: its length, then the JSON it measures.
    fn header(&self, file: &str) -> Option<Vec<u8>> {
        let length = self.get(file, Some(0..8))?;
        let size = u64::from_le_bytes(length.as_slice().try_into().unwrap());
        let json = self.get(file, Some(8..8 + size))?;
        Some([length, json].concat())
    }

    /// The bytes `range` of the folder's `file` on the hub, or all of them; `None` when the
    /// commit holds no such file.
    fn get(&self, file: &str, range: Option<Range<u64>>) -> Option<Vec<u8>> {
        let path = Path::new(self.folder).join(file);
        let path = path.to_str().unwrap();
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{path}",
            self.repo, self.revision
        );
        let config = ureq::Agent::config_builder()
            .redirect_auth_headers(ureq::config::RedirectAuthHeaders::SameHost)
            .build();
        let mut request = ureq::Agent::new_with_config(config).get(&url);
        if let Ok(token) = std::fs::read_to_string(Self::home().join("token")) {
            request = request.header("Authorization", format!("Bearer {}", token.trim()));
        }
        let limit = match &range {
            Some(range) => {
                let bytes = format!("bytes={}-{}", range.start, range.end - 1);
                request = request.header("Range", bytes);
                // The reader refuses to read on once it took the limit, even at the body's end.
                range.end - range.start + 1
            }
            None => u64::MAX,
        };
        match request.call() {
            Ok(mut response) => {
                let body = response.body_mut().with_config().limit(limit);
                Some(
                    body.read_to_vec()
                        .unwrap_or_else(|error| panic!("{url}: {error}")),
                )
            }
            Err(ureq::Error::StatusCode(404)) => None,
            Err(error) => panic!("{url}: {error}"),
        }
    }
}
