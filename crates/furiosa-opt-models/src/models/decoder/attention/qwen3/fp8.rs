//! Qwen3-8B with block-scaled E4M3 projections: the checkpoint contract and the registry line of
//! the catalog model a launch runs in that form.

use super::{QueryHeads, Size};
use crate::load::{Checkpoint, Load, LoadError};
use crate::models::config::decoder::Rotary;
use crate::models::config::{
    self, Exclusions, Quantization, Scheme, compiled, decoder, scaling::Spectrum,
};
use crate::models::decoder::attention::extra_large::QWidth;
use crate::models::decoder::attention::{Geometry, Head};
use crate::models::decoder::form::BlockScaled;
use crate::models::decoder::launch;
use crate::models::{Class, Implements};
use furiosa_opt_std::prelude::*;
use std::io;

const _: () = assert!(QueryHeads::<Size<"8B">>::SIZE * Head::SIZE == QWidth::SIZE);

/// The checkpoint contract of the compiled decoder.
#[derive(Clone, Copy)]
pub struct Config {
    pub theta: f32,
    pub tied: bool,
    pub window: usize,
}

impl TryFrom<&decoder::Config<Spectrum>> for Config {
    type Error = io::Error;

    fn try_from(config: &decoder::Config<Spectrum>) -> io::Result<Self> {
        if !matches!(
            &config.quantization_config,
            Some(Quantization { scheme: Scheme::Block, excluded: Exclusions(excluded), kv_cache: false })
                if excluded.is_empty()
        ) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "quantization_config: dense serving supports only block-scaled E4M3 FP8 Qwen3-8B",
            ));
        }
        let Rotary(theta) = Rotary::try_from(config.rotary).map_err(config::refusal)?;
        let geometry = &config.geometry;
        for (field, found, expected) in [
            ("hidden_size", geometry.hidden_size.get(), 4096),
            ("intermediate_size", geometry.intermediate_size, 12288),
            (
                "num_hidden_layers",
                geometry.num_hidden_layers.get(),
                <Size<"8B"> as Geometry>::LAYERS,
            ),
            (
                "num_attention_heads",
                geometry.num_attention_heads.get(),
                32,
            ),
            ("num_key_value_heads", geometry.num_key_value_heads.get(), 8),
            ("head_dim", geometry.head_dim(), 128),
            ("vocab_size", geometry.vocab_size.get(), 151936),
        ] {
            compiled(field, found, expected)?;
        }
        compiled(
            "rms_norm_eps",
            config.geometry.rms_norm_eps.unwrap_or(1e-6),
            1e-6,
        )?;
        compiled("sliding_window", config.sliding_window, None)?;
        Ok(Self {
            theta: theta.get(),
            tied: config.tie_word_embeddings.unwrap_or(false),
            window: config.capacity(launch::Positions::<Size<"8B">>::SIZE),
        })
    }
}

/// HF `Qwen3ForCausalLM` of a block-scaled E4M3 Qwen3-8B checkpoint: the model its launches run.
#[derive(DeviceSend, Load)]
pub struct ForCausalLM {
    #[load(rename = "")]
    pub model: launch::Model<Size<"8B">, BlockScaled>,
}

/// Block-scaled E4M3 Qwen3-8B reads a checkpoint at its root, and refuses one whose config is not of
/// it.
impl Implements for ForCausalLM {
    const SIZE: &'static str = "8B";
    const CLASS: Class = Class {
        family: <Size<"8B"> as Geometry>::FAMILY,
        name: "ForCausalLM",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        let refused = |error: io::Error| {
            let (name, reason) = ("config.json".into(), error.to_string());
            match error.kind() {
                io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
                _ => LoadError::InvalidData { name, reason },
            }
        };
        let config = crate::models::config::parse::<decoder::Config<Spectrum>>(source.config())
            .map_err(refused)?;
        Config::try_from(&config).map(|_| "").map_err(refused)
    }
}

#[cfg(test)]
mod tests {
    use super::super::Bound;
    use super::*;
    use crate::load::Checkpoint;
    use crate::models::decoder::attention::extra_large::Hidden;
    use crate::tensor::memory::Hbm;
    use crate::tests::DEVICE;
    use serde_json::{Value, json};

    /// The catalog binding at `N` stacked layers.
    type Layered<const N: usize> = super::super::Bound<Size<"8B">, BlockScaled, bf16, Hbm, N>;

    /// The checkpoint of a two-layer Qwen3-8B with block-scaled E4M3 projections, block scales
    /// stored as BF16 inverses, and its config; `heads` gives the query and key projections'
    /// block scales.
    fn checkpoint(heads: fn(usize) -> f32) -> crate::load::Checkpoint {
        use safetensors::Dtype;
        let (hidden, vocab) = (Hidden::SIZE, 151936);
        let directory = tempfile::tempdir().unwrap();
        let config = serde_json::json!({ "vocab_size": vocab });
        let mut checkpoint = crate::load::Checkpoint::new(directory.path(), Some(config)).unwrap();
        let floats = |checkpoint: &mut crate::load::Checkpoint,
                      name: &str,
                      shape: &[usize],
                      value: &dyn Fn(usize) -> f32| {
            let data = (0..shape.iter().product())
                .flat_map(|index| half::bf16::from_f32(value(index)).to_le_bytes())
                .collect();
            checkpoint.insert(name, Dtype::BF16, shape, data).unwrap();
        };
        let gain = |index: usize| 0.5 + (index % 7) as f32 / 8.0;
        floats(
            &mut checkpoint,
            "embed_tokens.weight",
            &[vocab, hidden],
            &|i| (i % 13) as f32 / 64.0,
        );
        // The readout's projection, which the tree reads at its root.
        floats(&mut checkpoint, "weight", &[vocab, hidden], &|i| {
            (i % 11) as f32 / 64.0
        });
        floats(&mut checkpoint, "norm.weight", &[hidden], &gain);
        let mut projections = Vec::new();
        for layer in 0..2 {
            let at = format!("layers.{layer}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                floats(
                    &mut checkpoint,
                    &format!("{at}.{name}.weight"),
                    &[hidden],
                    &gain,
                );
            }
            for name in ["q_norm", "k_norm"] {
                floats(
                    &mut checkpoint,
                    &format!("{at}.self_attn.{name}.weight"),
                    &[Head::SIZE],
                    &gain,
                );
            }
            for (name, rows, columns, head) in [
                ("self_attn.q_proj", 4096, hidden, true),
                ("self_attn.k_proj", 1024, hidden, true),
                ("self_attn.v_proj", 1024, hidden, false),
                ("self_attn.o_proj", hidden, 4096, false),
                ("mlp.gate_proj", 12288, hidden, false),
                ("mlp.up_proj", 12288, hidden, false),
                ("mlp.down_proj", hidden, 12288, false),
            ] {
                projections.push((format!("{at}.{name}"), rows, columns, head, layer));
            }
        }
        for (name, rows, columns, head, layer) in projections {
            let scale = |block: usize| match head {
                true => heads(block),
                false => 0.01 + ((block * 13 + layer * 5) % 17) as f32 * 0.37,
            };
            let codes: Vec<f8e4m3> = (0..rows * columns)
                .map(|i| f8e4m3::from_f32(0.125 + ((i * 37) % 61) as f32 / 8.0))
                .collect();
            let shape = [rows, columns];
            let codes = f8e4m3::to_buf(&codes);
            checkpoint
                .insert(&format!("{name}.weight"), Dtype::F8_E4M3, &shape, codes)
                .unwrap();
            let grid = [rows / 128, columns / 128];
            floats(
                &mut checkpoint,
                &format!("{name}.weight_scale_inv"),
                &grid,
                &|i| scale(i % grid[1]),
            );
        }
        checkpoint
    }

    /// Query and key block scales a power of two apart along each row.
    fn exact(block: usize) -> f32 {
        [2.0, 0.5, 1.0, 0.25][block % 4]
    }

    /// A block-FP8 Qwen3-8B checkpoint loads into the decoder with every name read, whatever its
    /// query and key block scales; one more name or one fewer is refused.
    #[tokio::test(flavor = "current_thread")]
    async fn loads_a_block_scaled_checkpoint() {
        use crate::load::LoadError;
        use furiosa_opt_std::prelude::Topology;
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mut source = checkpoint(exact);
        let canonical = |name: String| name.replace("weight_scale_inv", "weight_scale");
        let names: Vec<String> = source.names().into_iter().map(canonical).collect();
        let empty = crate::load::Checkpoint::default().load::<Layered<2>>(&mut device);
        match empty.err() {
            Some(LoadError::Missing(missing)) => {
                assert!(
                    missing.contains(&"config.json:vocab_size".to_string()),
                    "{missing:?}"
                )
            }
            other => panic!("{other:?} is not `Missing`"),
        }
        let directory = tempfile::tempdir().unwrap();
        let config = serde_json::json!({ "vocab_size": 151936 });
        let bare = crate::load::Checkpoint::new(directory.path(), Some(config)).unwrap();
        match bare.load::<Layered<2>>(&mut device).err() {
            Some(LoadError::Missing(mut missing)) => {
                missing.sort();
                assert_eq!(missing, names);
            }
            other => panic!("{other:?} is not `Missing`"),
        }
        source.load::<Layered<2>>(&mut device).unwrap();
        let extra = [0u8; 4];
        source
            .insert("extra", safetensors::Dtype::F32, &[1], extra.to_vec())
            .unwrap();
        match source.load::<Layered<2>>(&mut device).err() {
            Some(LoadError::Unexpected(names)) => assert_eq!(names, ["extra"]),
            other => panic!("{other:?} is not `Unexpected`"),
        }
        source.remove("extra");
        source.remove("layers.1.self_attn.k_norm.weight");
        match source.load::<Layered<2>>(&mut device).err() {
            Some(LoadError::Missing(names)) => {
                assert_eq!(names, ["layers.1.self_attn.k_norm.weight"])
            }
            other => panic!("{other:?} is not `Missing`"),
        }
        checkpoint(|block| [0.37, 0.3][block % 2])
            .load::<Layered<2>>(&mut device)
            .unwrap();
    }

    /// The published block-FP8 Qwen3-8B checkpoint is served by the FP8 line and loads strictly
    /// into it, its rotary table built from the config.
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "reads the published checkpoint from the local hub cache"]
    async fn loads_the_published_checkpoint() {
        use furiosa_opt_std::prelude::Topology;
        let hub = (
            "furiosa-ai/Qwen3-8B-FP8",
            "187ef89f2eecd7685df013bebd2fe46108aecd12",
        );
        let snapshot = crate::tests::snapshot(hub, &["*.json", "*.safetensors"]);
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let source = Checkpoint::new(snapshot, None).unwrap();
        assert_eq!(ForCausalLM::scope(&source).unwrap(), "");
        source.load::<ForCausalLM>(&mut device).unwrap();
    }

    /// `furiosa-ai/Qwen3-8B-FP8`'s config.json at revision 187ef89f2eecd7685df013bebd2fe46108aecd12.
    fn published() -> Value {
        json!({
            "architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
            "bos_token_id": 151643, "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu",
            "hidden_size": 4096, "initializer_range": 0.02, "intermediate_size": 12288,
            "max_position_embeddings": 40960, "max_window_layers": 36, "model_type": "qwen3",
            "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 1000000,
            "sliding_window": null, "tie_word_embeddings": false, "torch_dtype": "bfloat16",
            "transformers_version": "4.51.0", "use_cache": true, "use_sliding_window": false,
            "vocab_size": 151936,
            "quantization_config": {
                "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                "weight_block_size": [128, 128]
            }
        })
    }

    /// The compiled FP8 decoder reads its published config; another quantization, geometry,
    /// epsilon, sliding window or rotation is refused as unsupported by its field.
    #[test]
    fn refuses_what_the_dense_decoder_does_not_serve() {
        let parse = |edit: &dyn Fn(&mut Value)| {
            let mut document = published();
            edit(&mut document);
            let declared: decoder::Config<Spectrum> = crate::models::config::parse(&document)?;
            Config::try_from(&declared).map(drop)
        };
        parse(&|_| ()).unwrap();
        let quantized =
            "quantization_config: dense serving supports only block-scaled E4M3 FP8 Qwen3-8B";
        type Edit = Box<dyn Fn(&mut Value)>;
        let mut cases: Vec<(Edit, String)> = vec![
            (Box::new(|document| document["quantization_config"] = Value::Null), quantized.into()),
            (
                Box::new(|document| {
                    document["quantization_config"] =
                        json!({"quant_method": "fp8", "activation_scheme": "static"})
                }),
                quantized.into(),
            ),
            (
                Box::new(|document| document["quantization_config"]["ignored_layers"] = json!(["lm_head"])),
                quantized.into(),
            ),
            (
                Box::new(|document| {
                    document["quantization_config"]["kv_cache_scheme"] =
                        json!({"num_bits": 8, "type": "float", "dynamic": false})
                }),
                quantized.into(),
            ),
            (
                Box::new(|document| document["partial_rotary_factor"] = json!(0.5)),
                "found Default rotating 0.5 of each head, compiled the unscaled spectrum over the whole head".into(),
            ),
            (
                Box::new(|document| document["rms_norm_eps"] = json!(1e-5)),
                "rms_norm_eps: found 1e-5, compiled 1e-6".into(),
            ),
            (
                Box::new(|document| document["sliding_window"] = json!(4096)),
                "sliding_window: found Some(4096), compiled None".into(),
            ),
        ];
        for (field, found, held) in [
            ("hidden_size", 2048, 4096),
            ("intermediate_size", 6144, 12288),
            ("num_hidden_layers", 28, 36),
            ("num_attention_heads", 16, 32),
            ("num_key_value_heads", 4, 8),
            ("head_dim", 64, 128),
            ("vocab_size", 152064, 151936),
        ] {
            let edit = move |document: &mut Value| document[field] = json!(found);
            cases.push((
                Box::new(edit),
                format!("{field}: found {found}, compiled {held}"),
            ));
        }
        for (edit, reason) in cases {
            let error = parse(&*edit).unwrap_err();
            let refusal = (error.kind(), error.to_string());
            assert_eq!(
                refusal,
                (io::ErrorKind::Unsupported, reason.clone()),
                "{reason}"
            );
        }
    }
}
