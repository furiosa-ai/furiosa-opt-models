//! Synthetic four-chip contracts for H3's loaded input projections and final heads.
//!
//! Actual family widths, sparse analytic coefficients and chip-distinct operands; no MiniMax
//! weights or captured outputs. Run only with four explicitly selected RNGD chips.

#![feature(register_tool, adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Module;
use furiosa_opt_models::load::{Checkpoint, Context, Load, Named};
use furiosa_opt_models::models::diffusion::minimax_h3::embeddings::{
    self as h3, Chips, Condition, Latent, Patch, Time, Width,
};
use furiosa_opt_models::ops::index::packed::Timestep;
use furiosa_opt_models::tensor::memory::Hbm;
use furiosa_opt_models::tensor::memory::residency::Residency;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::{DeviceOutput, launch};
use safetensors::Dtype;
use tokio::sync::Mutex;

axes![Rows = 128, TimeRows = 64];

static DEVICE: Mutex<()> = Mutex::const_new(());

type Inputs = Named<"transformer", h3::Embeddings>;
type InputView<'a> = <Inputs as Residency>::In<&'a Hbm>;
type Heads = Named<"transformer", h3::FinalLayer>;
type HeadView<'a> = <Heads as Residency>::In<&'a Hbm>;
type Hidden = HbmTensor<bf16, Chips, m![Rows, Width]>;

#[device(chip = 4)]
fn apply<M, I, S, O>(device: &mut Device, model: &M, state: S, input: I) -> O
where
    M: DeviceSend + Module<I, (), S, Output = O>,
    I: DeviceSend,
    S: DeviceSend,
    O: DeviceOutput,
{
    model.forward(device, (), state, input)
}

// Two nonzeros per output row. These describe the synthetic matrix mathematically, rather than
// reproducing any production layout, BF16 decomposition, projection or device implementation.
fn columns(out: usize, input: usize) -> [usize; 2] {
    let first = (61 * out + 17) % input;
    [first, (first + input / 2 + 1) % input]
}

fn coefficient(out: usize, term: usize) -> f32 {
    match term {
        0 => 0.812_731 + (out % 5) as f32 / 32.,
        _ => -0.437_611 + (out % 3) as f32 / 64.,
    }
}

fn offset(out: usize) -> f32 {
    (out % 11) as f32 / 256. - 5. / 256. + 0.000_731
}

fn text_offset(out: usize) -> f32 {
    // Exactly representable FP32, with offsets that lose information if prematurely cast to BF16.
    ((out % 17) as f32 - 8.) / 256. + 3. / 16_384.
}

fn put_f32(source: &mut Checkpoint, name: &str, shape: &[usize], values: &[f32]) {
    source
        .insert(name, Dtype::F32, shape, f32::to_buf(values))
        .unwrap();
}

fn put_bf16(source: &mut Checkpoint, name: &str, shape: &[usize], values: &[bf16]) {
    source
        .insert(name, Dtype::BF16, shape, bf16::to_buf(values))
        .unwrap();
}

fn precise_projection(source: &mut Checkpoint, name: &str, output: usize, input: usize) {
    let mut weight = vec![0.; output * input];
    for out in 0..output {
        for (term, column) in columns(out, input).into_iter().enumerate() {
            let value = coefficient(out, term);
            assert_ne!(
                value,
                bf16::from_f32(value).to_f32(),
                "exercise a nonzero low part"
            );
            weight[out * input + column] = value;
        }
    }
    put_f32(
        source,
        &format!("transformer.{name}.weight"),
        &[output, input],
        &weight,
    );
    put_f32(
        source,
        &format!("transformer.{name}.bias"),
        &[output],
        &(0..output).map(offset).collect::<Vec<_>>(),
    );
}

fn input_checkpoint() -> Checkpoint {
    let mut source = Checkpoint::default();
    let mut text = vec![bf16::from_f32(0.); Width::SIZE * Condition::SIZE];
    for out in 0..Width::SIZE {
        let [first, second] = columns(out, Condition::SIZE);
        text[out * Condition::SIZE + first] = bf16::from_f32(0.5);
        text[out * Condition::SIZE + second] = bf16::from_f32(-0.25);
    }
    put_bf16(
        &mut source,
        "transformer.condition_proj.weight",
        &[Width::SIZE, Condition::SIZE],
        &text,
    );
    put_f32(
        &mut source,
        "transformer.condition_proj.bias",
        &[Width::SIZE],
        &(0..Width::SIZE).map(text_offset).collect::<Vec<_>>(),
    );
    precise_projection(&mut source, "video_patch_proj", Width::SIZE, Patch::SIZE);
    precise_projection(&mut source, "audio_patch_proj", Width::SIZE, Latent::SIZE);
    precise_projection(&mut source, "time_embedder.proj_in", Width::SIZE, 256);
    precise_projection(
        &mut source,
        "time_embedder.proj_out",
        Time::SIZE,
        Width::SIZE,
    );
    source
}

fn text_input(chip: usize, row: usize, col: usize) -> bf16 {
    bf16::from_f32(((5 * chip + 7 * row + 11 * col) % 31) as f32 / 32. - 15. / 32.)
}

fn media_input(chip: usize, row: usize, col: usize) -> f32 {
    ((17 * chip + 3 * row + 5 * col) % 43) as f32 / 32. - 21. / 32.
        + (col % 3 + 1) as f32 * 0.000_173
}

fn scalar_time(chip: usize, row: usize) -> f32 {
    match row {
        0 => 0.137_31 + chip as f32 / 32.,
        1 => 0.683_421 - 3. * chip as f32 / 64.,
        _ => 0., // Only the first two rows are meaningful; Trim belongs outside this component.
    }
}

fn linear(out: usize, input: usize, value: impl Fn(usize) -> f64) -> (f64, f64) {
    let mut sum = f64::from(offset(out));
    let mut magnitude = 0.;
    for (term, column) in columns(out, input).into_iter().enumerate() {
        let product = value(column) * f64::from(coefficient(out, term));
        sum += product;
        magnitude += product.abs();
    }
    (sum, magnitude)
}

fn silu(value: f64) -> f64 {
    value / (1. + (-value).exp())
}
fn rounded(value: f64) -> f64 {
    f64::from(bf16::from_f32(value as f32).to_f32())
}

fn expected_time(time: f32) -> Vec<f64> {
    (0..Time::SIZE)
        .map(|out| {
            linear(out, Width::SIZE, |hidden| {
                let projected = linear(hidden, 256, |position| {
                    let frequency = 10_000_f64.powf(-((position % 128) as f64) / 128.);
                    let angle = f64::from(time) * frequency;
                    if position < 128 {
                        angle.cos()
                    } else {
                        angle.sin()
                    }
                })
                .0;
                silu(projected)
            })
            .0
        })
        .collect()
}

fn check_text(got: &[bf16]) {
    assert_eq!(got.len(), 4 * Rows::SIZE * Width::SIZE);
    for chip in 0..4 {
        for row in 0..Rows::SIZE {
            for out in 0..Width::SIZE {
                let [first, second] = columns(out, Condition::SIZE);
                let expected = f64::from(text_input(chip, row, first).to_f32()) * 0.5
                    - f64::from(text_input(chip, row, second).to_f32()) * 0.25
                    + f64::from(text_offset(out));
                // Dyadic products, sums and biases fit exactly in FP32, so this is an exact BF16 oracle.
                assert_eq!(
                    got[(chip * Rows::SIZE + row) * Width::SIZE + out].to_bits(),
                    bf16::from_f32(expected as f32).to_bits(),
                    "text chip {chip} row {row} output {out}"
                );
            }
        }
    }
}

fn check_media(got: &[bf16], input: usize, name: &str) {
    assert_eq!(got.len(), 4 * Rows::SIZE * Width::SIZE);
    for chip in 0..4 {
        for row in 0..Rows::SIZE {
            for out in 0..Width::SIZE {
                let (expected, magnitude) =
                    linear(out, input, |col| f64::from(media_input(chip, row, col)));
                let actual = f64::from(got[(chip * Rows::SIZE + row) * Width::SIZE + out].to_f32());
                // Exactly the existing split::Embedding FP64 contract, including its last BF16 rounding.
                let bound = magnitude / 131_072. + expected.abs() / 256. + 1e-6;
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= bound,
                    "{name} chip {chip} row {row} output {out}: {actual} versus {expected}, bound {bound}"
                );
            }
        }
    }
}

fn check_time(got: &[f32]) {
    assert_eq!(got.len(), 4 * TimeRows::SIZE * Time::SIZE);
    let padding = expected_time(0.);
    for chip in 0..4 {
        let meaningful = [
            expected_time(scalar_time(chip, 0)),
            expected_time(scalar_time(chip, 1)),
        ];
        for row in 0..TimeRows::SIZE {
            let expected = if row < 2 { &meaningful[row] } else { &padding };
            let rms = (expected.iter().map(|value| value * value).sum::<f64>() / Time::SIZE as f64)
                .sqrt();
            let worst = (0..Time::SIZE)
                .map(|col| {
                    let actual = got[(chip * TimeRows::SIZE + row) * Time::SIZE + col];
                    assert!(actual.is_finite());
                    (f64::from(actual) - expected[col]).abs()
                })
                .fold(0., f64::max);
            // Existing TimeEmbedding contract; do not replace it with a wider composed tolerance.
            assert!(
                worst <= 1e-4 * rms,
                "time chip {chip} row {row}: worst {worst}, RMS {rms}"
            );
        }
    }
}

/// Real named-tree loading and borrowed Module execution, including the complete step composition.
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn loaded_input_components_match_independent_sparse_references() {
    let _lease = DEVICE.lock().await;
    let source = input_checkpoint();
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let owned: Inputs = source.load(&mut device).unwrap();
    drop(source);
    let view: InputView<'_> = Load::load(&mut Context::new(None), &owned).unwrap();
    let text: HbmTensor<bf16, Chips, m![Rows, Condition]> =
        HostTensor::<bf16, m![{ Chips }, Rows, Condition]>::from_vec((0..4).flat_map(|c| {
            (0..Rows::SIZE)
                .flat_map(move |r| (0..Condition::SIZE).map(move |i| text_input(c, r, i)))
        }))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let video: HbmTensor<f32, Chips, m![Rows, Patch]> =
        HostTensor::<f32, m![{ Chips }, Rows, Patch]>::from_vec((0..4).flat_map(|c| {
            (0..Rows::SIZE).flat_map(move |r| (0..Patch::SIZE).map(move |i| media_input(c, r, i)))
        }))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let audio: HbmTensor<f32, Chips, m![Rows, Latent]> =
        HostTensor::<f32, m![{ Chips }, Rows, Latent]>::from_vec((0..4).flat_map(|c| {
            (0..Rows::SIZE).flat_map(move |r| (0..Latent::SIZE).map(move |i| media_input(c, r, i)))
        }))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let times: HbmTensor<f32, Chips, m![TimeRows,1 # 8]> =
        HostTensor::<f32, m![{ Chips },TimeRows,1 # 8]>::from_vec((0..4).flat_map(|c| {
            (0..TimeRows::SIZE).flat_map(move |r| [scalar_time(c, r), 0., 0., 0., 0., 0., 0., 0.])
        }))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    // Check repeated calls through the same loaded, borrowed tree.
    for _ in 0..4 {
        let output = launch(apply, (&mut device, &view.0.text, &mut (), &text))
            .await
            .unwrap();
        output.ready().await.unwrap();
        let got = output
            .to_host::<m![{ Chips }, Rows, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        check_text(&got);
        let output = launch(
            apply,
            (&mut device, &view, &mut (), (&times, (&video, &audio))),
        )
        .await
        .unwrap();
        output.0.ready().await.unwrap();
        output.1.0.ready().await.unwrap();
        output.1.1.ready().await.unwrap();
        let got_time = output
            .0
            .to_host::<m![{ Chips }, TimeRows, Time]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let got_video = output
            .1
            .0
            .to_host::<m![{ Chips }, Rows, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let got_audio = output
            .1
            .1
            .to_host::<m![{ Chips }, Rows, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        check_time(&got_time);
        check_media(&got_video, Patch::SIZE, "video input");
        check_media(&got_audio, Latent::SIZE, "audio input");
    }
}

fn norm_weight() -> f32 {
    1. + 1. / 128.
}
fn amplitude(chip: usize, row: usize, audio: bool) -> f32 {
    0.5 + ((chip + row + usize::from(audio)) % 4) as f32 / 4.
}
fn hidden(chip: usize, row: usize, col: usize, audio: bool) -> bf16 {
    let sign = if (13 * chip + 7 * row + col + usize::from(audio)).is_multiple_of(3) {
        -1.
    } else {
        1.
    };
    bf16::from_f32(sign * amplitude(chip, row, audio))
}
fn condition(chip: usize, step: usize, col: usize) -> f32 {
    if step == 0 {
        0.
    } else if (chip + col).is_multiple_of(3) {
        -1.
    } else {
        1.
    }
}
fn modulation_column(output: usize) -> usize {
    (17 * output + 3) % Time::SIZE
}
fn modulation_weight(output: usize) -> f32 {
    if output < Width::SIZE {
        1. / 16.
    } else {
        1. / 8.
    }
}

fn head_checkpoint() -> Checkpoint {
    let mut source = Checkpoint::default();
    let mut modulation = vec![bf16::from_f32(0.); 2 * Width::SIZE * Time::SIZE];
    for out in 0..2 * Width::SIZE {
        modulation[out * Time::SIZE + modulation_column(out)] =
            bf16::from_f32(modulation_weight(out));
    }
    put_bf16(
        &mut source,
        "transformer.final_layer.adaln_proj.linear.weight",
        &[2 * Width::SIZE, Time::SIZE],
        &modulation,
    );
    let bias: Vec<f32> = (0..2 * Width::SIZE)
        .map(|out| if out < Width::SIZE { 1. / 256. } else { 0. })
        .collect();
    put_f32(
        &mut source,
        "transformer.final_layer.adaln_proj.linear.bias",
        &[2 * Width::SIZE],
        &bias,
    );
    put_bf16(
        &mut source,
        "transformer.final_layer.norm.weight",
        &[Width::SIZE],
        &vec![bf16::from_f32(norm_weight()); Width::SIZE],
    );
    precise_projection(
        &mut source,
        "final_layer.video_out",
        Patch::SIZE,
        Width::SIZE,
    );
    precise_projection(
        &mut source,
        "final_layer.audio_out",
        Latent::SIZE,
        Width::SIZE,
    );
    source
}

fn normalized(chip: usize, row: usize, col: usize, audio: bool, step: usize, fused: bool) -> f64 {
    // Every hidden feature has the same magnitude, so mean(x²) is exactly amplitude².
    let square = f64::from(amplitude(chip, row, audio)).powi(2);
    let epsilon = f64::from(f32::from_bits(h3::EPSILON as u32));
    let norm = f64::from(hidden(chip, row, col, audio).to_f32()) * f64::from(norm_weight())
        / (square + epsilon).sqrt();
    let projected = |output: usize| {
        let activated = rounded(silu(f64::from(condition(
            chip,
            step,
            modulation_column(output),
        ))));
        rounded(
            activated * f64::from(modulation_weight(output))
                + if output < Width::SIZE { 1. / 256. } else { 0. },
        )
    };
    let (shift, scale) = (projected(col), projected(Width::SIZE + col));
    rounded((if fused { norm } else { rounded(norm) }) * (1. + scale) + shift)
}

fn expected_head(
    chip: usize,
    row: usize,
    out: usize,
    audio: bool,
    step: usize,
    fused: bool,
) -> (f64, f64) {
    linear(out, Width::SIZE, |col| {
        normalized(chip, row, col, audio, step, fused)
    })
}

fn check_heads(got: &[f32], audio: bool) {
    let outputs = if audio { Latent::SIZE } else { Patch::SIZE };
    assert_eq!(got.len(), 4 * Rows::SIZE * outputs);
    let step = usize::from(audio);
    for chip in 0..4 {
        for row in 0..Rows::SIZE {
            for out in 0..outputs {
                let (expected, magnitude) = expected_head(chip, row, out, audio, step, false);
                let actual = f64::from(got[(chip * Rows::SIZE + row) * outputs + out]);
                // Existing split::Precise bound. The fixture puts the first RMS rounding safely away
                // from a tie; final modulation is dyadic after BF16 SiLU, preserving both roundings.
                let bound = magnitude / 131_072. + 1e-6;
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= bound,
                    "head audio={audio} chip {chip} row {row} output {out}: {actual} versus {expected}, bound {bound}"
                );
            }
        }
    }
}

/// Actual final-head Module, sensitive to both RMS roundings and the two timestep selectors.
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn loaded_final_heads_preserve_rounding_and_timestep_selection() {
    let _lease = DEVICE.lock().await;
    // Establish that this analytic fixture would fail a fused RMS+modulation or the wrong STEP.
    assert!((0..Patch::SIZE).any(|out| {
        (expected_head(0, 0, out, false, 0, false).0 - expected_head(0, 0, out, false, 0, true).0)
            .abs()
            > 0.001
    }));
    assert!((0..Latent::SIZE).any(|out| {
        (expected_head(0, 0, out, true, 1, false).0 - expected_head(0, 0, out, true, 0, false).0)
            .abs()
            > 0.01
    }));
    let source = head_checkpoint();
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let owned: Heads = source.load(&mut device).unwrap();
    drop(source);
    let view: HeadView<'_> = Load::load(&mut Context::new(None), &owned).unwrap();
    let mut values = Vec::new();
    for audio in [false, true] {
        let tensor: Hidden =
            HostTensor::<bf16, m![{ Chips }, Rows, Width]>::from_vec((0..4).flat_map(|c| {
                (0..Rows::SIZE)
                    .flat_map(move |r| (0..Width::SIZE).map(move |i| hidden(c, r, i, audio)))
            }))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        values.push(tensor);
    }
    let mut tensors = values.into_iter();
    let hidden = (tensors.next().unwrap(), tensors.next().unwrap());
    let times: HbmTensor<f32, Chips, m![Timestep, Time]> =
        HostTensor::<f32, m![{ Chips }, Timestep, Time]>::from_vec((0..4).flat_map(|c| {
            (0..2).flat_map(move |step| (0..Time::SIZE).map(move |i| condition(c, step, i)))
        }))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    // Both timestep selectors and their rounding contracts hold on repeated calls.
    for _ in 0..4 {
        let output = launch(
            apply,
            (&mut device, &view, &mut ((), ()), (&hidden, &times)),
        )
        .await
        .unwrap();
        output.0.ready().await.unwrap();
        output.1.ready().await.unwrap();
        let video = output
            .0
            .to_host::<m![{ Chips }, Rows, Patch]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let audio = output
            .1
            .to_host::<m![{ Chips }, Rows, Latent]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        check_heads(&video, false);
        check_heads(&audio, true);
    }
}
