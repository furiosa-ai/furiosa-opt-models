//! Native framework transformer arguments, adapted from Furiosa's Rust component parser.
//!
//! Framework-only metadata is checked here. The engine's checked layout owns canonical
//! text/audio/video geometry, selected-row bounds and the final validation before any DMA.

use crate::tensor::HostTensor;
use furiosa_opt_engine::diffusion::minimax_h3::Input;
use furiosa_opt_models::models::diffusion::minimax_h3::{
    embeddings::{Condition, Latent, Patch},
    packing::{AUDIO_TOKENS, Capacity},
};
use furiosa_opt_models::tensor::memory::Dtype;
use furiosa_opt_std::prelude::{AxisName, bf16};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Parse the keyword-only forward call into owned host values for the shared H3 facade.
/// The containing callable rejects positional arguments before entering this parser.
pub(super) fn parse(kwargs: &Bound<'_, PyDict>) -> PyResult<Input> {
    let names = [
        "x",
        "audio_x",
        "img_position_ids",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "skip_mask_out_condition",
        "prompt_embeds",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "packed_seq_params",
        "refiner_packed_seq_params",
        "video_token_layout",
        "rope_table",
    ];
    for key in kwargs.keys() {
        let key: String = key.extract()?;
        if !names.contains(&key.as_str()) {
            return Err(PyTypeError::new_err(format!(
                "unexpected transformer kwarg {key}"
            )));
        }
    }
    let get = |name: &str| {
        kwargs
            .get_item(name)?
            .ok_or_else(|| PyTypeError::new_err(format!("missing transformer kwarg {name}")))
    };

    let video = read(&get("x")?, Dtype::F32)?;
    let audio = read(&get("audio_x")?, Dtype::F32)?;
    let sequence = *video
        .shape
        .get(1)
        .ok_or_else(|| PyValueError::new_err("x must be [1, S, 96]"))?;
    if video.shape != [1, sequence, Patch::SIZE] || audio.shape != [1, sequence, Latent::SIZE] {
        return Err(PyValueError::new_err(
            "x and audio_x must be [1, S, 96] and [1, S, 32] FP32",
        ));
    }

    // Index and mask tensors retain the repro's logical flattening semantics; exporters with
    // strides are copied in logical order by HostTensor, without changing dtype or values.
    let rows = |name| indices(&field(&get(name)?, "position_ids")?, Dtype::I64);
    let text_rows = rows("text_pos_info")?;
    let audio_rows = rows("audio_pos_info")?;
    let video_rows = rows("img_pos_info")?;
    let selected_video = rows("img_pos_for_infer_output_info")?;
    let text = text_rows.len();
    // One maximum-2048 Live capacity, including intermediate valid lengths; no bucket routing.
    let plan = Capacity::default()
        .plan(text)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let packed = get("packed_seq_params")?;
    if let Ok(requests) = field(&packed, "num_requests") {
        if requests.extract::<usize>()? != 1 {
            return Err(PyValueError::new_err(
                "transformer accepts one request per call",
            ));
        }
    }
    let bounds = indices(&field(&packed, "cu_seqlens_q")?, Dtype::I32)?;
    if field(&packed, "max_seqlen_q")?.extract::<usize>()? != plan.key_count() {
        return Err(PyValueError::new_err(
            "packed max_seqlen_q must equal the live row count",
        ));
    }
    let refiner = get("refiner_packed_seq_params")?;
    if indices(&field(&refiner, "cu_seqlens_q")?, Dtype::I32)? != [0, text, text]
        || field(&refiner, "max_seqlen_q")?.extract::<usize>()? != text
    {
        return Err(PyValueError::new_err(
            "refiner bounds must describe the single prompt",
        ));
    }

    let prompt = read(&get("prompt_embeds")?, Dtype::BF16)?;
    if prompt.shape != [text, Condition::SIZE] {
        return Err(PyValueError::new_err(
            "prompt_embeds must be [1 to 2048, 5120] BF16, with one row per valid text token",
        ));
    }
    let positions = read(&get("img_position_ids")?, Dtype::F64)?;
    if positions.shape != [1, sequence, 3] {
        return Err(PyValueError::new_err(
            "img_position_ids must be [1, S, 3] FP64",
        ));
    }
    let positions = positions
        .buffer
        .chunks_exact(24)
        .map(|bytes| {
            std::array::from_fn(|axis| {
                let start = axis * 8;
                f64::from_le_bytes([
                    bytes[start],
                    bytes[start + 1],
                    bytes[start + 2],
                    bytes[start + 3],
                    bytes[start + 4],
                    bytes[start + 5],
                    bytes[start + 6],
                    bytes[start + 7],
                ])
            })
        })
        .collect();

    let inverse = indices(&get("inverse_indices")?, Dtype::I64)?;
    let tags = read(&get("token_tags")?, Dtype::I64)?;
    let tags = tags
        .buffer
        .chunks_exact(8)
        .map(integer64)
        .collect::<Vec<_>>();
    if inverse.len() != sequence || tags.len() != sequence {
        return Err(PyValueError::new_err(
            "inverse_indices and token_tags must cover S rows",
        ));
    }
    let timesteps = super::floats(&read(&get("unique_timesteps")?, Dtype::F32)?)?;
    if inverse.iter().any(|&index| index >= timesteps.len())
        || timesteps.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "inverse_indices must select finite unique_timesteps",
        ));
    }
    // Layout is checked by the engine, so use checked indexing while reading its unrefined
    // row lists here. Empty modalities and out-of-range indices must never panic the binding.
    let first = |rows: &[usize]| {
        rows.first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("video and audio row lists must be nonempty"))
    };
    let time_at = |row: usize| -> PyResult<f32> {
        let index = inverse.get(row).ok_or_else(|| {
            PyValueError::new_err("modality position_ids must index the upstream sequence")
        })?;
        // All inverse values, including upstream padding, were checked above.
        Ok(timesteps[*index])
    };
    let video_start = first(&video_rows)?;
    let times = [time_at(video_start)?, time_at(first(&audio_rows)?)?];
    for (rows, time, tag) in [
        (&video_rows, times[0], 0_i64),
        (&audio_rows, times[1], 2_i64),
        (&text_rows, times[0], 1_i64),
    ] {
        for &row in rows {
            let row_time = time_at(row)?;
            let actual = tags[row];
            if (tag == 1 && ![0, 1].contains(&actual))
                || (tag != 1 && (actual != tag || row_time != time))
            {
                return Err(PyValueError::new_err(
                    "compiled transformer requires video/audio tags 0/2, encoder text tags 0 or 1, and one timestep per media modality",
                ));
            }
        }
    }
    if kwargs
        .get_item("rope_table")?
        .is_some_and(|value| !value.is_none())
    {
        return Err(PyValueError::new_err(
            "prepared rope_table is unsupported; supply img_position_ids",
        ));
    }
    let skip = kwargs
        .get_item("skip_mask_out_condition")?
        .map(|value| value.extract::<bool>())
        .transpose()?
        .unwrap_or(false);
    let mask = |value: &Bound<'_, PyAny>, count| -> PyResult<Vec<bool>> {
        let tensor = read(value, Dtype::BOOL)?;
        if tensor.buffer.len() != count {
            return Err(PyValueError::new_err(
                "update mask must cover selected rows",
            ));
        }
        Ok(tensor
            .buffer
            .iter()
            .map(|&value| skip || value != 0)
            .collect())
    };
    let video_mask = mask(&get("update_mask")?, selected_video.len())?;
    let audio_mask = match kwargs.get_item("update_audio_mask")? {
        Some(value) if !value.is_none() => mask(&value, AUDIO_TOKENS)?,
        _ => vec![true; AUDIO_TOKENS],
    };

    if let Some(value) = kwargs.get_item("video_token_layout")? {
        if !value.is_none() {
            let spans = field(&value, "video_spans")?;
            let spans = spans.try_iter()?.collect::<PyResult<Vec<_>>>()?;
            let (grid, start) = match spans.as_slice() {
                [] => (field(&value, "latent_grid")?, field(&value, "prefix_len")?),
                [span] => (field(span, "latent_grid")?, field(span, "start")?),
                _ => {
                    return Err(PyValueError::new_err(
                        "reference video spans are unsupported",
                    ));
                }
            };
            if grid.extract::<(usize, usize, usize)>()? != (37, 24, 42)
                || start.extract::<usize>()? != video_start
            {
                return Err(PyValueError::new_err(
                    "video layout must describe 1344x768, 124 frames with no reference spans",
                ));
            }
        }
    }

    Ok(Input {
        text_rows,
        audio_rows,
        video_rows,
        selected_video,
        bounds,
        sequence,
        prompt: prompt
            .buffer
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect(),
        prompt_shape: [text, Condition::SIZE],
        positions,
        times,
        video: super::floats(&video)?,
        video_shape: [sequence, Patch::SIZE],
        audio: super::floats(&audio)?,
        audio_shape: [sequence, Latent::SIZE],
        video_mask,
        audio_mask,
    })
}

/// Reuse the binding's DLPack importer; an existing HostTensor is already host-owned.
fn read(value: &Bound<'_, PyAny>, dtype: Dtype) -> PyResult<HostTensor> {
    let tensor = super::host(value)?;
    if tensor.dtype != dtype {
        return Err(PyValueError::new_err(format!(
            "expected host {dtype:?}, got {:?}",
            tensor.dtype
        )));
    }
    Ok(tensor)
}

fn indices(value: &Bound<'_, PyAny>, dtype: Dtype) -> PyResult<Vec<usize>> {
    let host = read(value, dtype)?;
    host.buffer
        .chunks_exact(dtype.size())
        .map(|bytes| {
            let index = match dtype {
                Dtype::I64 => integer64(bytes),
                Dtype::I32 => {
                    i64::from(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                }
                _ => unreachable!("indices use signed integers"),
            };
            usize::try_from(index)
                .map_err(|_| PyValueError::new_err("row indices must be nonnegative"))
        })
        .collect()
}

fn integer64(bytes: &[u8]) -> i64 {
    i64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

fn field<'py>(value: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(mapping) = value.cast::<PyDict>() {
        mapping
            .get_item(name)?
            .ok_or_else(|| PyTypeError::new_err(format!("missing {name}")))
    } else {
        value.getattr(name)
    }
}
