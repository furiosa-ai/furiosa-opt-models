//! Qwen3-VL's size reads its published config, whose text model rotates by one multimodal rotation
//! of interleaved sections, and refuses a rotation the text model does not hold.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;

use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::vision_language::qwen3_vl::Size;
use serde_json::{Value, json};

use common::Published;

const QWEN3_VL_32B: Published = Published::new(
    "Qwen/Qwen3-VL-32B-Instruct",
    "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
);

/// The size parses its published checkpoint's config, and the same rotation in Transformers 5's
/// spelling.
#[test]
fn size_reads_its_published_config() {
    let published = QWEN3_VL_32B.config();
    Size::<"32B">::parse(&published).unwrap();
    let mut respelled = published;
    let text = &mut respelled["text_config"];
    let mut rope = text["rope_scaling"].take();
    rope["rope_theta"] = text["rope_theta"].take();
    text["rope_parameters"] = rope;
    Size::<"32B">::parse(&respelled).unwrap();
}

/// A rotation the text model does not hold is refused, each case by its kind and reason.
#[test]
fn size_refuses_another_rotation() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = QWEN3_VL_32B.config();
    let cases: [(&[&str], Value, io::ErrorKind, &str); 5] = [
        (
            &["rope_scaling", "mrope_interleaved"],
            json!(false),
            Unsupported,
            "text_config: found sections [24, 20, 20] in one block each, compiled interleaved \
             sections",
        ),
        (
            &["rope_scaling", "mrope_section"],
            json!([24, 20, 16]),
            InvalidData,
            "text_config: mrope_section: found [24, 20, 16], expected sections of the 64 pairs \
             of a 128-wide head",
        ),
        (
            &["rope_scaling"],
            json!({
                "rope_type": "linear", "factor": 2.0, "mrope_section": [24, 20, 20],
                "mrope_interleaved": true,
            }),
            Unsupported,
            "text_config: found Linear { factor: 2.0 } rotating 1 of each head, compiled the \
             unscaled spectrum over the whole head",
        ),
        (
            &["partial_rotary_factor"],
            json!(0.5),
            Unsupported,
            "text_config: found Default rotating 0.5 of each head, compiled the unscaled \
             spectrum over the whole head",
        ),
        (
            &["rope_theta"],
            json!(null),
            InvalidData,
            "text_config: missing field `rope_theta`",
        ),
    ];
    for (path, value, kind, reason) in cases {
        let mut document = published.clone();
        let (key, parents) = path.split_last().unwrap();
        let parent = parents
            .iter()
            .fold(&mut document["text_config"], |part, key| &mut part[*key]);
        parent[*key] = value;
        let error = Size::<"32B">::parse(&document).map(drop).unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{}", path.join("."));
    }
}
