//! H3's host-side document resolution. Numerical leaves receive no filesystem paths or JSON.

use super::config::{self, Audio, AudioMetadata, TextEncoder, Transformer, Video, VideoSource};
use crate::load::{Checkpoint, LoadError};
use crate::models::Family;
use crate::models::config::{self as schema, Literal};
use serde::Deserialize;
use serde_json::Value;
use std::path::{Component as PathComponent, Path, PathBuf};

/// A component's physical folder and weight prefix in the public checkpoint.
/// Both are relative to the checkpoint root, including a wrapper's optional `source_path`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    directory: PathBuf,
    weights: PathBuf,
    prefix: String,
    source: Option<String>,
}

impl Component {
    pub fn directory(&self) -> &Path {
        &self.directory
    }
    pub fn weights_directory(&self) -> &Path {
        &self.weights
    }
    pub fn tensor_prefix(&self) -> &str {
        &self.prefix
    }

    fn wrapper<'a>(&self, merged: &'a Value) -> &'a Value {
        // Construction checked that this single component folder has an object configuration.
        &merged[self
            .directory
            .to_str()
            .expect("constructed from a JSON string")]
    }

    fn config_name(&self) -> String {
        format!("config.json:{}", self.directory.display())
    }

    fn source_config<'a>(&self, merged: &'a Value) -> Result<&'a Value, LoadError> {
        let wrapper = self.wrapper(merged);
        match &self.source {
            None => Ok(wrapper),
            Some(name) => object_at(wrapper, name, &format!("{}.{}", self.config_name(), name)),
        }
    }
}

/// Required component paths resolved from original index declarations and merged wrapper configs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    transformer: Component,
    text_encoder: Component,
    video_vae: Component,
    audio_vae: Component,
}

impl Layout {
    pub fn transformer(&self) -> &Component {
        &self.transformer
    }
    pub fn text_encoder(&self) -> &Component {
        &self.text_encoder
    }
    pub fn video_vae(&self) -> &Component {
        &self.video_vae
    }
    pub fn audio_vae(&self) -> &Component {
        &self.audio_vae
    }

    fn resolve(merged: &Value, index: &Value) -> Result<Self, LoadError> {
        let DeclaredIndex { class: Literal } = parse::<DeclaredIndex>(index, "model_index.json")?;
        Ok(Self {
            transformer: component(merged, index, "transformer", "MiniMaxH3DiTModel")?,
            text_encoder: component(merged, index, "text_encoder", "MiniMaxH3Qwen3VLHFEncoder")?,
            video_vae: component(merged, index, "video_vae", "MiniMaxH3VideoVAE")?,
            audio_vae: component(merged, index, "audio_vae", "MiniMaxH3AudioVAE")?,
        })
    }
}

#[derive(Deserialize)]
struct DeclaredIndex {
    #[serde(rename = "_class_name")]
    class: Literal<"MiniMaxH3Pipeline">,
}

/// Checked inputs for one full four-chip pipeline, retaining no checkpoint tensors or JSON borrows.
/// Text is required; separate stages receiving precomputed text are not full pipeline inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct Inputs {
    config: config::Config,
    transformer: Transformer,
    text_encoder: TextEncoder,
    video: Video,
    video_source: VideoSource,
    audio: Audio,
    audio_metadata: AudioMetadata,
    layout: Layout,
}

impl Inputs {
    /// Read the original index and audio metadata once, leaving the checkpoint's tensor/config
    /// ownership and strict loading rules intact. This method runs only on the host, outside kernels.
    ///
    /// ```no_run
    /// use furiosa_opt_models::load::Checkpoint;
    /// use furiosa_opt_models::models::diffusion::minimax_h3::Inputs;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let checkpoint = Checkpoint::new("/path/to/MiniMax-H3", None)?;
    /// let inputs = Inputs::read(&checkpoint)?;
    /// let video_prefix = inputs.layout().video_vae().tensor_prefix();
    /// # let _ = video_prefix;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read(source: &Checkpoint) -> Result<Self, LoadError> {
        let index = source.document(Path::new("model_index.json"))?;
        let prepared = prepare(source.config(), &index)?;
        let path = prepared.layout.audio_vae.weights.join("metadata.json");
        let metadata = source.document(&path)?;
        prepared.finish(&metadata)
    }

    /// Check explicit borrowed documents without reading files or inventing merged sidecar keys.
    /// `merged` is the public checkpoint's config, `index` the original model_index.json, and
    /// `audio_metadata` the document beside the resolved audio weights. Values are owned on return.
    pub fn from_documents(
        merged: &Value,
        index: &Value,
        audio_metadata: &Value,
    ) -> Result<Self, LoadError> {
        prepare(merged, index)?.finish(audio_metadata)
    }

    pub fn config(&self) -> &config::Config {
        &self.config
    }
    pub fn transformer(&self) -> &Transformer {
        &self.transformer
    }
    pub fn text_encoder(&self) -> &TextEncoder {
        &self.text_encoder
    }
    pub fn video(&self) -> &Video {
        &self.video
    }
    pub fn video_source(&self) -> &VideoSource {
        &self.video_source
    }
    pub fn audio(&self) -> &Audio {
        &self.audio
    }
    pub fn audio_metadata(&self) -> &AudioMetadata {
        &self.audio_metadata
    }
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

struct Prepared {
    config: config::Config,
    transformer: Transformer,
    text_encoder: TextEncoder,
    video: Video,
    video_source: VideoSource,
    audio: Audio,
    layout: Layout,
}

impl Prepared {
    fn finish(self, metadata: &Value) -> Result<Inputs, LoadError> {
        let document = self.layout.audio_vae.weights.join("metadata.json");
        let audio_metadata = family(metadata, &document.to_string_lossy())?;
        Ok(Inputs {
            config: self.config,
            transformer: self.transformer,
            text_encoder: self.text_encoder,
            video: self.video,
            video_source: self.video_source,
            audio: self.audio,
            audio_metadata,
            layout: self.layout,
        })
    }
}

fn prepare(merged: &Value, index: &Value) -> Result<Prepared, LoadError> {
    let layout = Layout::resolve(merged, index)?;
    let config = family(merged, "config.json")?;
    let transformer = family(
        layout.transformer.wrapper(merged),
        &layout.transformer.config_name(),
    )?;
    let text_encoder = family(
        layout.text_encoder.wrapper(merged),
        &layout.text_encoder.config_name(),
    )?;
    let video = family(
        layout.video_vae.wrapper(merged),
        &layout.video_vae.config_name(),
    )?;
    let source_name = match &layout.video_vae.source {
        Some(source) => format!("{}.{}", layout.video_vae.config_name(), source),
        None => layout.video_vae.config_name(),
    };
    let video_source = family(layout.video_vae.source_config(merged)?, &source_name)?;
    let audio = family(
        layout.audio_vae.wrapper(merged),
        &layout.audio_vae.config_name(),
    )?;
    Ok(Prepared {
        config,
        transformer,
        text_encoder,
        video,
        video_source,
        audio,
        layout,
    })
}

fn component(
    merged: &Value,
    index: &Value,
    name: &str,
    class: &str,
) -> Result<Component, LoadError> {
    let key = format!("model_index.json:{name}");
    let value = index
        .get(name)
        .filter(|v| !v.is_null())
        .ok_or_else(|| LoadError::Missing(vec![key.clone()]))?;
    let array = value
        .as_array()
        .ok_or_else(|| invalid(&key, "expected a component declaration array"))?;
    if array.iter().all(Value::is_null) {
        return Err(LoadError::Missing(vec![key]));
    }
    let folder = match array.as_slice() {
        [Value::String(_), Value::String(found)] if found == class => name,
        [Value::String(_), Value::String(found), Value::Object(spec)] if found == class => spec
            .get("subfolder")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid(&key, "modular component requires a string subfolder"))?,
        [Value::String(_), Value::String(found), ..] if found != class => {
            return Err(invalid(&key, format!("expected {class}, found {found}")));
        }
        _ => {
            return Err(invalid(
                &key,
                "expected [library, class] or [library, class, {subfolder}]",
            ));
        }
    };
    single_entry(folder, &key)?;
    let config_name = format!("config.json:{folder}");
    let wrapper = object_at(merged, folder, &config_name)?;
    let source = wrapper
        .get("source_path")
        .map(|value| {
            let name = format!("{config_name}.source_path");
            let entry = value
                .as_str()
                .ok_or_else(|| invalid(&name, "expected one filename"))?;
            single_entry(entry, &name)?;
            Ok(entry.to_owned())
        })
        .transpose()?;
    let directory = PathBuf::from(folder);
    let weights = source
        .as_ref()
        .map_or_else(|| directory.clone(), |source| directory.join(source));
    let prefix = source
        .as_ref()
        .map_or_else(|| folder.to_owned(), |source| format!("{folder}.{source}"));
    Ok(Component {
        directory,
        weights,
        prefix,
        source,
    })
}

fn single_entry(entry: &str, name: &str) -> Result<(), LoadError> {
    let mut components = Path::new(entry).components();
    if !matches!(components.next(), Some(PathComponent::Normal(_)))
        || components.next().is_some()
        || Path::new(entry)
            .file_name()
            .is_none_or(|file| file != entry)
    {
        return Err(invalid(
            name,
            "expected one normal filename, without parent, absolute or nested paths",
        ));
    }
    Ok(())
}

fn object_at<'a>(parent: &'a Value, key: &str, name: &str) -> Result<&'a Value, LoadError> {
    let value = parent
        .get(key)
        .ok_or_else(|| LoadError::Missing(vec![name.to_owned()]))?;
    if !value.is_object() {
        return Err(invalid(name, "expected a configuration object"));
    }
    Ok(value)
}

fn family<T: Family>(document: &Value, name: &str) -> Result<T, LoadError> {
    T::parse(document).map_err(|error| located(name, error))
}

fn parse<T: serde::de::DeserializeOwned>(document: &Value, name: &str) -> Result<T, LoadError> {
    schema::parse(document).map_err(|error| located(name, error))
}

fn located(name: &str, error: std::io::Error) -> LoadError {
    match error.kind() {
        std::io::ErrorKind::Unsupported => LoadError::Unsupported {
            name: name.into(),
            reason: error.to_string(),
        },
        _ => invalid(name, error.to_string()),
    }
}

fn invalid(name: &str, reason: impl Into<String>) -> LoadError {
    LoadError::InvalidData {
        name: name.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::documents;
    use super::*;
    use serde_json::json;

    fn error_name(error: LoadError) -> String {
        match error {
            LoadError::Missing(names) => {
                assert_eq!(names.len(), 1);
                names[0].clone()
            }
            LoadError::InvalidData { name, .. } | LoadError::Unsupported { name, .. } => name,
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn owns_checked_values_and_preserves_physical_prefixes() {
        let (mut merged, index, metadata) = documents();
        let inputs = Inputs::from_documents(&merged, &index, &metadata).unwrap();
        let layout = inputs.layout();
        assert_eq!(layout.transformer().tensor_prefix(), "transformer");
        assert_eq!(layout.text_encoder().tensor_prefix(), "text_encoder");
        assert_eq!(layout.video_vae().directory(), Path::new("video_vae"));
        assert_eq!(
            layout.video_vae().weights_directory(),
            Path::new("video_vae/source")
        );
        assert_eq!(layout.video_vae().tensor_prefix(), "video_vae.source");
        assert_eq!(layout.audio_vae().tensor_prefix(), "audio_vae");
        merged["video_vae"]["latents_mean"][0] = json!(123.);
        drop((merged, index, metadata));
        assert_eq!(inputs.video().statistics().mean()[0], 0.);
        assert_eq!(inputs.audio().statistics().std(), &[1.; 32]);
        assert_eq!(inputs.text_encoder().checkpoint_layers(), 64);
        assert_eq!(inputs.config().shifts().video(), 2.);
    }

    #[test]
    fn requires_the_full_pipeline_text_declaration() {
        let (merged, index, metadata) = documents();
        for value in [None, Some(Value::Null), Some(json!([null, null]))] {
            let mut document = index.clone();
            match value {
                None => {
                    document.as_object_mut().unwrap().remove("text_encoder");
                }
                Some(value) => document["text_encoder"] = value,
            }
            let error = Inputs::from_documents(&merged, &document, &metadata).unwrap_err();
            assert!(matches!(error, LoadError::Missing(_)));
            assert_eq!(error_name(error), "model_index.json:text_encoder");
        }
        let mut document = merged;
        document.as_object_mut().unwrap().remove("text_encoder");
        assert_eq!(
            error_name(Inputs::from_documents(&document, &index, &metadata).unwrap_err()),
            "config.json:text_encoder"
        );
    }

    #[test]
    fn validates_classes_in_the_original_index_and_locates_errors() {
        let (merged, index, metadata) = documents();
        for component in ["transformer", "text_encoder", "video_vae", "audio_vae"] {
            let mut document = index.clone();
            document[component][1] = json!("OtherClass");
            let error = Inputs::from_documents(&merged, &document, &metadata).unwrap_err();
            assert!(matches!(error, LoadError::InvalidData { .. }));
            assert_eq!(error_name(error), format!("model_index.json:{component}"));
        }
        let mut document = merged.clone();
        document["transformer"]["num_layers"] = json!(1);
        let error = Inputs::from_documents(&document, &index, &metadata).unwrap_err();
        assert!(matches!(error, LoadError::Unsupported { .. }));
        assert_eq!(error_name(error), "config.json:transformer");
        let mut document = merged;
        document["video_vae"]
            .as_object_mut()
            .unwrap()
            .remove("source");
        let error = Inputs::from_documents(&document, &index, &metadata).unwrap_err();
        assert!(matches!(error, LoadError::Missing(_)));
        assert_eq!(error_name(error), "config.json:video_vae.source");
    }

    #[test]
    fn resolves_modular_folders_and_sources_without_renaming_tensors() {
        let (mut merged, mut index, metadata) = documents();
        index["video_vae"] = json!(["diffusers", "MiniMaxH3VideoVAE", {"subfolder":"video"}]);
        let mut video = merged.as_object_mut().unwrap().remove("video_vae").unwrap();
        let source = video.as_object_mut().unwrap().remove("source").unwrap();
        video["source_path"] = json!("weights");
        video["weights"] = source;
        merged["video"] = video;
        merged["audio_vae"]["source_path"] = json!("weights");
        let inputs = Inputs::from_documents(&merged, &index, &metadata).unwrap();
        assert_eq!(
            inputs.layout().video_vae().weights_directory(),
            Path::new("video/weights")
        );
        assert_eq!(inputs.layout().video_vae().tensor_prefix(), "video.weights");
        assert_eq!(
            inputs.layout().audio_vae().tensor_prefix(),
            "audio_vae.weights"
        );
        // A synthetic key in merged config cannot replace the separately supplied metadata.
        merged["audio_vae"]["metadata"] = metadata;
        let error = Inputs::from_documents(&merged, &index, &json!({})).unwrap_err();
        assert_eq!(error_name(error), "audio_vae/weights/metadata.json");
    }

    #[test]
    fn refuses_expanded_paths_and_malformed_component_declarations() {
        let (merged, index, metadata) = documents();
        for path in [
            "",
            ".",
            "..",
            "../up",
            "a/b",
            "/absolute",
            "a/",
            "a/.",
            "./a",
        ] {
            for component in ["video_vae", "audio_vae"] {
                let mut document = merged.clone();
                document[component]["source_path"] = json!(path);
                let error = Inputs::from_documents(&document, &index, &metadata).unwrap_err();
                assert!(
                    matches!(error, LoadError::InvalidData { .. }),
                    "{path}: {error}"
                );
                assert_eq!(
                    error_name(error),
                    format!("config.json:{component}.source_path")
                );
            }
            let mut document = index.clone();
            document["video_vae"] = json!(["lib", "MiniMaxH3VideoVAE", {"subfolder":path}]);
            assert_eq!(
                error_name(Inputs::from_documents(&merged, &document, &metadata).unwrap_err()),
                "model_index.json:video_vae"
            );
        }
        for declaration in [
            json!({}),
            json!(["lib"]),
            json!([1, "MiniMaxH3VideoVAE"]),
            json!(["lib", "MiniMaxH3VideoVAE", {}]),
            json!(["lib", "MiniMaxH3VideoVAE", {"subfolder":"video_vae"}, "extra"]),
        ] {
            let mut document = index.clone();
            document["video_vae"] = declaration;
            assert_eq!(
                error_name(Inputs::from_documents(&merged, &document, &metadata).unwrap_err()),
                "model_index.json:video_vae"
            );
        }
        for value in [Value::Null, json!(1)] {
            let mut document = merged.clone();
            document["audio_vae"]["source_path"] = value;
            assert_eq!(
                error_name(Inputs::from_documents(&document, &index, &metadata).unwrap_err()),
                "config.json:audio_vae.source_path"
            );
        }
    }

    #[test]
    fn reads_original_index_and_metadata_beside_weights_through_checkpoint() {
        fn write(root: &Path, name: &str, value: &Value) {
            let path = root.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
        }
        let root = tempfile::tempdir().unwrap();
        let (mut merged, index, metadata) = documents();
        merged["audio_vae"]["source_path"] = json!("weights");
        write(root.path(), "model_index.json", &index);
        for component in ["transformer", "text_encoder", "audio_vae"] {
            write(
                root.path(),
                &format!("{component}/config.json"),
                &merged[component],
            );
        }
        let mut video = merged["video_vae"].clone();
        let source = video.as_object_mut().unwrap().remove("source").unwrap();
        write(root.path(), "video_vae/config.json", &video);
        write(root.path(), "video_vae/source/config.json", &source);
        write(
            root.path(),
            "audio_vae/metadata.json",
            &json!({"deliberately":"wrong location"}),
        );
        write(root.path(), "audio_vae/weights/metadata.json", &metadata);
        let checkpoint = Checkpoint::new(root.path(), None).unwrap();
        let expected = Inputs::from_documents(checkpoint.config(), &index, &metadata).unwrap();
        let inputs = Inputs::read(&checkpoint).unwrap();
        assert_eq!(inputs, expected);
        std::fs::remove_file(root.path().join("audio_vae/weights/metadata.json")).unwrap();
        assert_eq!(inputs.audio().statistics().std(), &[1.; 32]);
        let error = Inputs::read(&checkpoint).unwrap_err();
        assert!(matches!(error, LoadError::Missing(_)));
        assert_eq!(error_name(error), "audio_vae/weights/metadata.json");
        // An override still belongs to Checkpoint; declarations are checked against disk's index.
        let mut override_config = checkpoint.config().clone();
        override_config["_minimax_h3"]["sigma_shift_scales"]["audio"] = json!(3.);
        write(root.path(), "audio_vae/weights/metadata.json", &metadata);
        let checkpoint = Checkpoint::new(root.path(), Some(override_config)).unwrap();
        assert_eq!(
            Inputs::read(&checkpoint).unwrap().config().shifts().audio(),
            3.
        );
    }
}
