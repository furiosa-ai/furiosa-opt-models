//! The published Gemma 4 checkpoint read strictly into the causal model's tree.

use furiosa_opt_models::load::Checkpoint;
use furiosa_opt_models::models::Implements;
use furiosa_opt_models::models::decoder::attention::gemma4::ForCausalLM;
use furiosa_opt_std::prelude::*;
use std::path::PathBuf;
use std::process::{Command, Stdio};

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

/// The published NVFP4 Gemma 4 12B checkpoint is the causal model's, and loads strictly into
/// it: every text tensor read by name, the media embedders and the cache scales unread.
#[tokio::test(flavor = "current_thread")]
#[ignore = "reads the published checkpoint from the local hub cache"]
async fn loads_the_published_checkpoint() {
    let hub = (
        "unsloth/gemma-4-12b-it-NVFP4",
        "b1f649734b34aa5575b03d186abd1b9be3d0d5c4",
    );
    let snapshot = snapshot(hub, &["*.json", "*.safetensors"]);
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let source = Checkpoint::new(snapshot, None).unwrap();
    assert_eq!(<ForCausalLM>::scope(&source).unwrap(), "");
    source.load::<ForCausalLM>(&mut device).unwrap();
}
