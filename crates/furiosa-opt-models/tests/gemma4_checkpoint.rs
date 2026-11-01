//! The published Gemma 4 checkpoint read strictly into the causal model's tree, and the
//! quantization declarations its lines admit by the modules each group targets.

use furiosa_opt_models::load::{Checkpoint, LoadError};
use furiosa_opt_models::models::Implements;
use furiosa_opt_models::models::decoder::attention::gemma4::{self, ForCausalLM};
use furiosa_opt_models::models::decoder::{Projections, form};
use furiosa_opt_std::prelude::*;
use serde_json::{Value, json};
use std::ops::Range;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// The published NVFP4 Gemma 4 12B checkpoint: its hub repository and revision.
const HUB: (&str, &str) = (
    "unsloth/gemma-4-12b-it-NVFP4",
    "b1f649734b34aa5575b03d186abd1b9be3d0d5c4",
);

/// What a line's scope refuses a checkpoint's `quantization_config` by.
const DECLARED: &str = "config.json:quantization_config";

/// The scope a line reads a checkpoint at, or why it does not serve it.
type Scope = fn(&Checkpoint) -> Result<&'static str, LoadError>;

/// The hub cache's directory of the files of checkpoint `repo` at `revision` that `include`
/// matches, fetched there once.
fn snapshot((repo, revision): (&str, &str), include: &[&str]) -> PathBuf {
    let mut command = Command::new("uvx");
    command.args(["--from", "huggingface-hub", "hf", "download"]);
    command.args([repo, "--revision", revision, "--quiet"]);
    for pattern in include {
        command.args(["--include", pattern]);
    }
    let output = command
        .stderr(Stdio::inherit())
        .output()
        .expect("uvx fetches the test checkpoint");
    assert!(output.status.success(), "the {repo} download failed");
    String::from_utf8(output.stdout).unwrap().trim().into()
}

/// The published checkpoint's config.
fn config() -> Value {
    let path = snapshot(HUB, &["config.json"]).join("config.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// The name and reason of `line`'s refusal of a checkpoint of `config` that declares `declared`,
/// none if the line reads it at its root.
fn refusal(line: Scope, config: &Value, declared: Value) -> Option<(String, String)> {
    let mut config = config.clone();
    config["quantization_config"] = declared;
    let empty = tempfile::tempdir().unwrap();
    match line(&Checkpoint::new(empty.path(), Some(config)).unwrap()) {
        Ok("") => None,
        Ok(scope) => panic!("read at {scope:?}"),
        Err(LoadError::Unsupported { name, reason }) => Some((name, reason)),
        Err(other) => panic!("{other:?} is not `Unsupported`"),
    }
}

/// The published NVFP4 Gemma 4 12B checkpoint is the causal model's, and loads strictly into
/// it: every text tensor read by name, the media embedders and the cache scales unread.
#[tokio::test(flavor = "current_thread")]
#[ignore = "reads the published checkpoint from the local hub cache"]
async fn loads_the_published_checkpoint() {
    let snapshot = snapshot(HUB, &["*.json", "*.safetensors"]);
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let source = Checkpoint::new(snapshot, None).unwrap();
    assert_eq!(<ForCausalLM>::scope(&source).unwrap(), "");
    source.load::<ForCausalLM>(&mut device).unwrap();
}

/// A line admits a declaration that quantizes each projection of every layer in the scheme its
/// form reads, as the targets naming it and the exclusions give it: the published groups, one
/// scheme of every linear module, or an NVFP4 feed-forward beside excluded attention. A projection
/// declared in another scheme is refused by its name.
#[test]
fn admits_each_projection_in_the_scheme_its_targets_give() {
    type Mlp =
        Projections<form::Affine, form::Affine, form::Affine, form::Affine, form::Nvfp4BlockScaled>;
    let config = config();
    let published = config["quantization_config"].clone();
    let targets = |group: &str| published["config_groups"][group]["targets"].clone();
    let mut swapped = published.clone();
    swapped["config_groups"]["group_0"]["targets"] = targets("group_1");
    swapped["config_groups"]["group_1"]["targets"] = targets("group_0");
    let mut channels = published.clone();
    channels["format"] = json!("float-quantized");
    let groups = channels["config_groups"].as_object_mut().unwrap();
    groups.remove("group_1");
    groups["group_0"]["targets"] = json!(["Linear"]);
    let layer = |at: usize| format!("model.language_model.layers.{at}");
    // A ModelOpt declaration of every linear module, the attention of `layers` excluded.
    let modelopt = |layers: Range<usize>| {
        let values = json!({"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16});
        let attention = layers.map(|at| format!("{}.self_attn*", layer(at)));
        let ignore: Vec<_> = ["lm_head", "model.embed_vision*"]
            .map(String::from)
            .into_iter()
            .chain(attention)
            .collect();
        json!({
            "quant_method": "modelopt", "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {"weights": values, "input_activations": values, "targets": ["Linear"]},
            },
            "ignore": ignore,
            "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
        })
    };
    let (mixed, channel, mlp): (Scope, Scope, Scope) = (
        <ForCausalLM>::scope,
        <ForCausalLM<form::ChannelScaled>>::scope,
        <ForCausalLM<Mlp>>::scope,
    );
    let refused = |at: usize, projection: &str, own: &str| {
        let reason = format!("{}.{projection}: Nvfp4, the line reads {own}", layer(at));
        Some((DECLARED.to_owned(), reason))
    };
    let cases = [
        (mixed, published.clone(), None),
        (channel, channels, None),
        (mlp, modelopt(0..gemma4::SLIDING + gemma4::FULL), None),
        (mixed, swapped, refused(0, "self_attn.q_proj", "Channel")),
        (channel, published, refused(0, "mlp.gate_proj", "Channel")),
        (mlp, modelopt(0..1), refused(1, "self_attn.q_proj", "BF16")),
    ];
    let found: Vec<_> = (cases.iter())
        .map(|(line, declaration, _)| refusal(*line, &config, declaration.clone()))
        .collect();
    let expected: Vec<_> = cases.into_iter().map(|(.., expected)| expected).collect();
    assert_eq!(found, expected);
}

/// A target that quantizes no projection, or that no supported selector spells, is refused: a
/// target is `Linear`, or a pattern of literal names joined by `.*`, ending in `$` and holding one
/// choice at most.
#[test]
fn refuses_a_target_that_quantizes_no_projection() {
    let config = config();
    let targeting = |targets: Value| {
        let mut declared = config["quantization_config"].clone();
        declared["config_groups"]["group_1"]["targets"] = targets;
        declared
    };
    let unsupported = |target: &str| {
        let selectors = r#""Linear" or "re:.*name.*name(a|b)name$" patterns of literal names"#;
        format!("found {target:?}, compiled {selectors}")
    };
    let feedforward = r"re:.*language_model.*\.mlp\.(gate|up|down)_proj$";
    let experts = r"re:.*mlp\.experts\.down_proj$";
    let regex = r"re:.*mlp\.[a-z]+_proj$";
    let cases = [
        (
            json!([feedforward, experts]),
            format!("target {experts} quantizes no projection the line holds"),
        ),
        (json!(["Embedding"]), unsupported("Embedding")),
        (json!([regex]), unsupported(regex)),
    ];
    let found: Vec<_> = (cases.iter())
        .map(|(targets, _)| refusal(<ForCausalLM>::scope, &config, targeting(targets.clone())))
        .collect();
    let expected: Vec<_> = (cases.into_iter())
        .map(|(_, reason)| Some((DECLARED.to_owned(), reason)))
        .collect();
    assert_eq!(found, expected);
}
