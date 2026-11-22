//! Strictly loaded two-layer H3 refiner, with synthetic sparse weights and an analytic oracle.
//!
//! Reduced feature widths keep this a composition contract. The text envelope is the production
//! 2048 rows; every chip executes the same replicated tree with distinct inputs. No model assets
//! or captured outputs are used. Full-width performance and upstream agreement are separate gates.

#![feature(register_tool, adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Module;
use furiosa_opt_models::load::{Checkpoint, Context, Load, Named};
use furiosa_opt_models::models::diffusion::minimax_h3::{embeddings::Chips, transformer};
use furiosa_opt_models::modules::attention;
use furiosa_opt_models::ops::attention::flash::{Span as KeySpan, Visible};
use furiosa_opt_models::ops::index::packed::Span;
use furiosa_opt_models::tensor::{memory::Hbm, scalar::Value};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;
use safetensors::Dtype;

axes![Width = 256, Attended = 512, Expanded = 512, Spans = 3];
type Envelope = Span<2048, 2048>;
type Hidden = HbmTensor<bf16, Chips, m![Envelope, Width]>;
type Mask = Visible<Spans, Chips>;
type Model<X = Hbm> =
    Named<"transformer.token_refiner", transformer::Refiner<Width, Attended, Expanded, X>>;
type QkvState = ((), ((), (), ()));
type LayerState = ((QkvState, ()), ((), ((), ())));
type State = ([LayerState; 2], ());
const LIMIT: f64 = 1e-2;

#[device(chip = 4)]
fn refine(
    device: &mut Device,
    model: &Model<&Hbm>,
    input: Hidden,
    mask: &Mask,
    bound: &Value<Chips>,
) -> Hidden {
    let mut state: State = Default::default();
    model.forward(
        device,
        &attention::Context {
            positions: (),
            write: (),
            read: (mask, bound),
        },
        &mut state,
        input,
    )
}

fn rounded(value: f64) -> f64 {
    f64::from(bf16::from_f32(value as f32).to_f32())
}

fn gain(column: usize, stage: usize) -> f64 {
    0.75 + ((column + 3 * stage) % 7) as f64 / 32.
}

fn source(channel: usize, stage: usize) -> usize {
    (channel * 7 + stage * 19 + 3) % Width::SIZE
}

fn input(chip: usize, row: usize, column: usize, valid: usize) -> f64 {
    let base = ((chip * 5 + row * 3 + column * 7) % 29) as f64 / 32. - 7. / 16.;
    rounded(base + if row < valid { 0. } else { 8. })
}

fn put(cp: &mut Checkpoint, name: &str, shape: &[usize], values: Vec<bf16>) {
    cp.insert(name, Dtype::BF16, shape, bf16::to_buf(&values))
        .unwrap();
}

fn checkpoint() -> Checkpoint {
    let mut cp = Checkpoint::default();
    let zero = bf16::from_f32(0.);
    for layer in 0..2 {
        let prefix = format!("transformer.token_refiner.blocks.{layer}");
        // Checkpoint QKV is [head, term, channel, input]. Q/K are exactly zero: softmax is
        // uniform over visible rows. Each V channel selects an independently specified input.
        let mut qkv = vec![zero; 3 * Attended::SIZE * Width::SIZE];
        for channel in 0..Attended::SIZE {
            let out = (channel / 128 * 3 + 2) * 128 + channel % 128;
            qkv[out * Width::SIZE + source(channel, layer)] = bf16::from_f32(0.5);
        }
        put(
            &mut cp,
            &format!("{prefix}.attn.qkv_proj.weight"),
            &[3 * Attended::SIZE, Width::SIZE],
            qkv,
        );
        let mut output = vec![zero; Width::SIZE * Attended::SIZE];
        for column in 0..Width::SIZE {
            output[column * Attended::SIZE + column] = bf16::from_f32(0.25);
            output[column * Attended::SIZE + column + Width::SIZE] = bf16::from_f32(-0.125);
        }
        put(
            &mut cp,
            &format!("{prefix}.attn.out_proj.weight"),
            &[Width::SIZE, Attended::SIZE],
            output,
        );
        for name in ["q_norm", "k_norm"] {
            cp.insert(
                &format!("{prefix}.attn.{name}.weight"),
                Dtype::F32,
                &[128],
                f32::to_buf(&vec![1.; 128]),
            )
            .unwrap();
        }
        for (name, stage) in [("norm1", 2 * layer), ("norm2", 2 * layer + 1)] {
            put(
                &mut cp,
                &format!("{prefix}.{name}.weight"),
                &[Width::SIZE],
                (0..Width::SIZE)
                    .map(|column| bf16::from_f32(gain(column, stage) as f32))
                    .collect(),
            );
        }
        let mut up = vec![zero; 2 * Expanded::SIZE * Width::SIZE];
        for channel in 0..Expanded::SIZE {
            up[channel * Width::SIZE + source(channel, layer + 2)] = bf16::from_f32(0.5);
            up[(Expanded::SIZE + channel) * Width::SIZE + source(channel, layer + 4)] =
                bf16::from_f32(-0.25);
        }
        put(
            &mut cp,
            &format!("{prefix}.mlp.fc1.weight"),
            &[2 * Expanded::SIZE, Width::SIZE],
            up,
        );
        let mut down = vec![zero; Width::SIZE * Expanded::SIZE];
        for column in 0..Width::SIZE {
            down[column * Expanded::SIZE + column] = bf16::from_f32(0.5);
            down[column * Expanded::SIZE + column + Width::SIZE] = bf16::from_f32(-0.25);
        }
        put(
            &mut cp,
            &format!("{prefix}.mlp.fc2.weight"),
            &[Width::SIZE, Expanded::SIZE],
            down,
        );
    }
    put(
        &mut cp,
        "transformer.token_refiner.final_norm.weight",
        &[Width::SIZE],
        (0..Width::SIZE)
            .map(|column| bf16::from_f32(gain(column, 4) as f32))
            .collect(),
    );
    cp
}

fn normalize(values: &[f64], stage: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    for row in values.chunks_exact(Width::SIZE) {
        let inverse = (row.iter().map(|x| x * x).sum::<f64>() / Width::SIZE as f64
            + f64::from(1e-5f32))
        .sqrt()
        .recip();
        result.extend(
            row.iter()
                .enumerate()
                .map(|(column, value)| rounded(value * inverse * gain(column, stage))),
        );
    }
    result
}

// Analytic sparse matrices, independent row-wise RMS and uniform visible-key softmax. All
// specified BF16 boundaries are explicit; no production loader, layout or numerical op is used.
fn oracle(chip: usize, valid: usize) -> Vec<f64> {
    let mut hidden: Vec<_> = (0..Envelope::SIZE)
        .flat_map(|row| (0..Width::SIZE).map(move |column| input(chip, row, column, valid)))
        .collect();
    for layer in 0..2 {
        let normalized = normalize(&hidden, 2 * layer);
        let mean: Vec<_> = (0..Attended::SIZE)
            .map(|channel| {
                rounded(
                    (0..valid)
                        .map(|row| {
                            rounded(0.5 * normalized[row * Width::SIZE + source(channel, layer)])
                        })
                        .sum::<f64>()
                        / valid as f64,
                )
            })
            .collect();
        for row in hidden.chunks_exact_mut(Width::SIZE) {
            for (column, value) in row.iter_mut().enumerate() {
                *value = rounded(
                    *value + rounded(0.25 * mean[column] - 0.125 * mean[column + Width::SIZE]),
                );
            }
        }
        let normalized = normalize(&hidden, 2 * layer + 1);
        for (row, n) in hidden
            .chunks_exact_mut(Width::SIZE)
            .zip(normalized.chunks_exact(Width::SIZE))
        {
            let products: Vec<_> = (0..Expanded::SIZE)
                .map(|channel| {
                    let gate = rounded(0.5 * n[source(channel, layer + 2)]);
                    let up = rounded(-0.25 * n[source(channel, layer + 4)]);
                    rounded(rounded(gate / (1. + (-gate).exp())) * up)
                })
                .collect();
            for (column, value) in row.iter_mut().enumerate() {
                *value = rounded(
                    *value
                        + rounded(0.5 * products[column] - 0.25 * products[column + Width::SIZE]),
                );
            }
        }
    }
    normalize(&hidden, 4)
}

#[tokio::test(flavor = "current_thread")]
async fn loaded_two_layer_refiner_matches_analytic_values() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let model: Model = checkpoint().load(&mut device).unwrap();
    let view = Model::<&Hbm>::load(&mut Context::new(None), &model).unwrap();
    let mut mask = Mask::alloc(&device).unwrap();
    let bound: Value<Chips> = HostTensor::<i32, m![1]>::from_vec([6144])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    for valid in [128, 2048, 129, 1, 128] {
        let hidden: Hidden =
            HostTensor::<bf16, m![{ Chips }, Envelope, Width]>::from_vec((0..4).flat_map(|chip| {
                (0..Envelope::SIZE).flat_map(move |row| {
                    (0..Width::SIZE)
                        .map(move |column| bf16::from_f32(input(chip, row, column, valid) as f32))
                })
            }))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        HostTensor::<bf16, m![{ Chips }, Spans, KeySpan]>::from_vec(
            (0..4 * 6144).map(|at| bf16::from_f32(if at % 6144 < valid { 1. } else { 0. })),
        )
        .to_hbm(&mut device.pdma)
        .output(&mut mask)
        .await
        .unwrap();
        let uploaded_mask = mask
            .to_host::<m![{ Chips }, Spans, KeySpan]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        assert!(
            uploaded_mask
                .iter()
                .enumerate()
                .all(|(at, value)| value.to_f32() == if at % 6144 < valid { 1. } else { 0. })
        );
        let result = launch(refine, (&mut device, &view, hidden, &mask, &bound))
            .await
            .unwrap()
            .to_host::<m![{ Chips }, Envelope, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let mut worst = 0_f64;
        for chip in 0..4 {
            let expected = oracle(chip, valid);
            for (row, reference) in result
                [chip * Envelope::SIZE * Width::SIZE..(chip + 1) * Envelope::SIZE * Width::SIZE]
                .chunks_exact(Width::SIZE)
                .zip(expected.chunks_exact(Width::SIZE))
            {
                let error = row
                    .iter()
                    .zip(reference)
                    .map(|(got, want)| (f64::from(got.to_f32()) - want).powi(2))
                    .sum::<f64>();
                let norm = reference.iter().map(|x| x * x).sum::<f64>();
                assert!(row.iter().all(|value| value.to_f32().is_finite()));
                let relative = (error / norm).sqrt();
                worst = worst.max(relative);
                assert!(
                    relative < LIMIT,
                    "valid={valid} chip={chip} row-relative RMS={relative}; first values {:?}; expected {:?}",
                    &row[..16],
                    &reference[..16]
                );
            }
        }
        eprintln!(
            "two_layer_refiner valid={valid} worst_row_relative_rms={worst:.9} limit={LIMIT}"
        );
    }
}
