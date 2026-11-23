//! A strictly loaded joint block: modulation, Live QKV/attention/exchange, FFN and residuals.
//!
//! Synthetic sparse matrices give an independent analytic oracle with explicit BF16 boundaries.
//! Feature widths are reduced; row geometry and maximum text capacity are the production ones.
//! This checks composition and alternating requests, not full-width performance or GPU parity.

#![feature(register_tool, adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Module;
use furiosa_opt_models::load::{Checkpoint, Context, Load, Named};
use furiosa_opt_models::models::diffusion::minimax_h3::{
    embeddings::Chips,
    joint,
    packing::{Capacity, Plan},
    positions::Positions,
    request::{BlockInputs, BlockPlan},
    transformer,
};
use furiosa_opt_models::modules::diffusion::WithSpans;
use furiosa_opt_models::ops::attention::flash::{Head, Span, Visible};
use furiosa_opt_models::ops::collective::Pieces;
use furiosa_opt_models::ops::index::packed::Timestep;
use furiosa_opt_models::ops::rotary::Rotation;
use furiosa_opt_models::tensor::{axis::Rows, memory::Hbm};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;
use safetensors::Dtype;
use std::collections::BTreeMap;

axes![
    Width = 256,
    Attended = 2048,
    Expanded = 512,
    Time = 64,
    Spans = 20
];
type Local = Rows<10496>;
type Joined = Pieces<Local, 4>;
type Hidden = HbmTensor<bf16, Chips, m![Local, Width]>;
type Condition = HbmTensor<f32, Chips, m![Timestep, Time]>;
type Model<X = Hbm> = Named<
    "transformer.blocks.0",
    WithSpans<transformer::Block<Width, Attended, Expanded, Time, X>>,
>;
type State = ((), (((), (), ()), ()), ((), ((), ())));
type Class = (usize, usize, usize, bool);
const LIMIT: f64 = 1e-2;
const POISON: u16 = 0x7fc1;

#[device(chip = 4)]
fn block(
    device: &mut Device,
    model: &Model<&Hbm>,
    input: Hidden,
    condition: &Condition,
    metadata: &BlockInputs<Local, Spans>,
    rotation: &Rotation<Head, Chips, Joined>,
    visible: &Visible<Spans, Chips>,
) -> Hidden {
    let call = metadata.read();
    let context = joint::Context::new(&call, rotation, visible);
    let hidden = call.prefix().interval().with(input);
    let mut state: State = Default::default();
    model
        .forward(
            device,
            (condition, (call.spans(), context.environment())),
            &mut state,
            hidden,
        )
        .into_value()
}

fn rounded(value: f64) -> f64 {
    f64::from(bf16::from_f32(value as f32).to_f32())
}

// Geometry below describes the semantic problem directly, rather than using production
// addressing or key-run helpers. The four parts divide each modality's padded storage.
fn class(part: usize, row: usize, tokens: usize) -> Option<Class> {
    let text = tokens.div_ceil(64) * 16;
    let (modality, valid) = if row < text {
        (1, part * text + row < tokens)
    } else if row < text + 112 {
        (2, part * 112 + row - text < 414)
    } else if row < text + 112 + 9328 {
        (0, part * 9328 + row - text - 112 < 37296)
    } else {
        return None;
    };
    Some((part, row % 29, modality, valid))
}

fn input((part, row, _, valid): Class, column: usize) -> f64 {
    1. + ((part * 13 + row * 7 + column * 3) % 29) as f64 / 32. + if valid { 0. } else { 8. }
}

fn gain(column: usize, stage: usize) -> f64 {
    0.75 + ((column * 3 + stage * 5) % 7) as f64 / 32.
}

fn modulation(modality: usize, chunk: usize) -> f64 {
    match chunk {
        0 => modality as f64 / 64.,
        1 => modality as f64 / 32.,
        2 => 0.5 + modality as f64 / 8.,
        3 => -(modality as f64) / 64.,
        4 => modality as f64 / 64.,
        5 => -0.25 - modality as f64 / 16.,
        _ => unreachable!(),
    }
}

fn v_source(channel: usize) -> usize {
    (channel * 7 + channel / 128 * 19 + 3) % Width::SIZE
}

fn v_scale(channel: usize) -> f64 {
    0.25 + (channel / 128) as f64 / 16.
}

fn out_source(column: usize) -> [usize; 2] {
    [
        column * 7 % Attended::SIZE,
        (column * 11 + 257) % Attended::SIZE,
    ]
}

fn put(cp: &mut Checkpoint, name: &str, shape: &[usize], values: Vec<bf16>) {
    cp.insert(name, Dtype::BF16, shape, bf16::to_buf(&values))
        .unwrap();
}

fn checkpoint() -> Checkpoint {
    let mut cp = Checkpoint::default();
    let prefix = "transformer.blocks.0";
    let zero = bf16::from_f32(0.);
    let mut qkv = vec![zero; 3 * Attended::SIZE * Width::SIZE];
    // Original checkpoint order is [head, Q/K/V, channel, input]. Zero Q/K make attention
    // uniform over exactly the semantic keys. Each global head has a distinct V projection.
    for channel in 0..Attended::SIZE {
        let out = (channel / 128 * 3 + 2) * 128 + channel % 128;
        qkv[out * Width::SIZE + v_source(channel)] = bf16::from_f32(v_scale(channel) as f32);
    }
    put(
        &mut cp,
        &format!("{prefix}.attn.qkv_proj.weight"),
        &[3 * Attended::SIZE, Width::SIZE],
        qkv,
    );
    let mut output = vec![zero; Width::SIZE * Attended::SIZE];
    for column in 0..Width::SIZE {
        let [first, second] = out_source(column);
        output[column * Attended::SIZE + first] = bf16::from_f32(0.25);
        output[column * Attended::SIZE + second] = bf16::from_f32(-0.125);
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
    for (name, stage) in [("norm1", 0), ("norm2", 1)] {
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
        up[channel * Width::SIZE + (channel * 3 + 7) % Width::SIZE] = bf16::from_f32(0.5);
        up[(Expanded::SIZE + channel) * Width::SIZE + (channel * 5 + 13) % Width::SIZE] =
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
    put(
        &mut cp,
        &format!("{prefix}.adaln_proj.linear.weight"),
        &[18 * Width::SIZE, Time::SIZE],
        vec![zero; 18 * Width::SIZE * Time::SIZE],
    );
    let bias: Vec<f32> = (0..18 * Width::SIZE)
        .map(|row| modulation(row / (6 * Width::SIZE), row / Width::SIZE % 6) as f32)
        .collect();
    cp.insert(
        &format!("{prefix}.adaln_proj.linear.bias"),
        Dtype::F32,
        &[18 * Width::SIZE],
        f32::to_buf(&bias),
    )
    .unwrap();
    cp
}

fn normalize(row: &[f64], modality: usize, stage: usize) -> Vec<f64> {
    let inverse = (row.iter().map(|x| x * x).sum::<f64>() / Width::SIZE as f64
        + f64::from(1e-5f32))
    .sqrt()
    .recip();
    row.iter()
        .enumerate()
        .map(|(column, value)| {
            rounded(
                value * inverse * gain(column, stage) * (1. + modulation(modality, 3 * stage + 1))
                    + modulation(modality, 3 * stage),
            )
        })
        .collect()
}

// Enumerate actual semantic keys, then solve the sparse matrices and uniform softmax on the
// host. Class counts only avoid repeating identical arithmetic for periodic synthetic rows.
fn oracle(tokens: usize) -> BTreeMap<Class, Vec<f64>> {
    let text = tokens.div_ceil(64) * 16;
    let mut counts = BTreeMap::<Class, usize>::new();
    for part in 0..4 {
        for row in 0..text + 112 + 9328 {
            *counts.entry(class(part, row, tokens).unwrap()).or_default() += 1;
        }
    }
    let normalized: BTreeMap<_, _> = counts
        .keys()
        .map(|&key| {
            let row: Vec<_> = (0..Width::SIZE).map(|column| input(key, column)).collect();
            (key, normalize(&row, key.2, 0))
        })
        .collect();
    let key_count: usize = counts.iter().filter(|(key, _)| key.3).map(|(_, n)| n).sum();
    assert_eq!(key_count, 37296 + 414 + tokens);
    let mean: Vec<_> = (0..Attended::SIZE)
        .map(|channel| {
            rounded(
                counts
                    .iter()
                    .filter(|(key, _)| key.3)
                    .map(|(key, count)| {
                        *count as f64
                            * rounded(v_scale(channel) * normalized[key][v_source(channel)])
                    })
                    .sum::<f64>()
                    / key_count as f64,
            )
        })
        .collect();
    let attention: Vec<_> = (0..Width::SIZE)
        .map(|column| {
            let [a, b] = out_source(column);
            rounded(0.25 * mean[a] - 0.125 * mean[b])
        })
        .collect();
    counts
        .keys()
        .map(|&key| {
            let mut row: Vec<_> = (0..Width::SIZE)
                .map(|column| {
                    rounded(input(key, column) + modulation(key.2, 2) * attention[column])
                })
                .collect();
            let norm = normalize(&row, key.2, 1);
            let product: Vec<_> = (0..Expanded::SIZE)
                .map(|channel| {
                    let gate = rounded(0.5 * norm[(channel * 3 + 7) % Width::SIZE]);
                    let up = rounded(-0.25 * norm[(channel * 5 + 13) % Width::SIZE]);
                    rounded(rounded(gate / (1. + (-gate).exp())) * up)
                })
                .collect();
            for (column, value) in row.iter_mut().enumerate() {
                let branch = rounded(0.5 * product[column] - 0.25 * product[column + Width::SIZE]);
                *value = rounded(*value + modulation(key.2, 5) * branch);
            }
            (key, row)
        })
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn loaded_live_joint_block_matches_analytic_values_across_alternating_requests() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let model: Model = checkpoint().load(&mut device).unwrap();
    let view = Model::<&Hbm>::load(&mut Context::new(None), &model).unwrap();
    let condition: Condition =
        HostTensor::<f32, m![Timestep, Time]>::from_vec(vec![0.; 2 * Time::SIZE])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
    for tokens in [77, 2048, 129, 77] {
        let plan = Capacity::default().plan(tokens).unwrap();
        let metadata = BlockPlan::<Local, Spans>::new(plan)
            .unwrap()
            .upload(&mut device)
            .await
            .unwrap();
        let source: Hidden =
            HostTensor::<bf16, m![{ Chips }, Local, Width]>::from_vec((0..4).flat_map(|part| {
                (0..Local::SIZE).flat_map(move |row| {
                    (0..Width::SIZE).map(move |column| {
                        class(part, row, tokens)
                            .map(|key| bf16::from_f32(input(key, column) as f32))
                            .unwrap_or_else(|| bf16::from_bits(POISON))
                    })
                })
            }))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let positions = Positions::new(plan, &vec![[0.; 3]; plan.key_count()], 10000.).unwrap();
        let (_, [low, high], visible) = positions.into_parts();
        let rotation = Rotation {
            low: HostTensor::<bf16, m![{ Symbol<Joined> }, Head]>::from_vec(
                low.into_iter().map(|x| bf16::from_bits(x.to_bits())),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
            high: HostTensor::<bf16, m![{ Symbol<Joined> }, Head]>::from_vec(
                high.into_iter().map(|x| bf16::from_bits(x.to_bits())),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        };
        let visible: Visible<Spans, Chips> = HostTensor::<bf16, m![Spans, Span]>::from_vec(
            visible.into_iter().map(|x| bf16::from_bits(x.to_bits())),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let result = launch(
            block,
            (
                &mut device,
                &view,
                source,
                &condition,
                &metadata,
                &rotation,
                &visible,
            ),
        )
        .await
        .unwrap();
        // The module owns its output; only the initialized work is downloaded.
        let bytes = result
            .into_buffer()
            .slice(.., ..plan.work_rows_per_part() * Width::SIZE * 2)
            .unwrap()
            .to_host(&mut device.pdma)
            .await
            .unwrap();
        check(plan, &bytes);
    }
}

fn check(plan: Plan, bytes: &[u8]) {
    let expected = oracle(plan.text_tokens());
    let work = plan.work_rows_per_part();
    assert_eq!(bytes.len(), 4 * work * Width::SIZE * 2);
    let mut worst = 0_f64;
    for (at, row) in bytes.chunks_exact(Width::SIZE * 2).enumerate() {
        let part = at / work;
        let local = at % work;
        if let Some(key) = class(part, local, plan.text_tokens()) {
            let mut error = 0.;
            let mut magnitude = 0.;
            for (column, pair) in row.chunks_exact(2).enumerate() {
                let got =
                    f64::from(bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32());
                let want = expected[&key][column];
                assert!(
                    got.is_finite(),
                    "tokens {}, part {part}, row {local}, column {column}",
                    plan.text_tokens()
                );
                error += (got - want).powi(2);
                magnitude += want.powi(2);
            }
            let relative = (error / magnitude.max(1e-30)).sqrt();
            assert!(
                relative < LIMIT,
                "tokens {}, part {part}, row {local}: RMS {relative}",
                plan.text_tokens()
            );
            worst = worst.max(relative);
        } else {
            assert!(
                row.iter().all(|&byte| byte == 0),
                "work guards are positive zero"
            );
        }
    }
    println!(
        "tokens={}, work={work}, worst_row_relative_rms={worst}",
        plan.text_tokens()
    );
}
