//! Four-chip gather/QKV numerical comparison inside one 10496-row allocation envelope.
//!
//! Synthetic source-head-major weights exercise the actual 128/96 partial rotary permutation.
//! Runtime Joined prefixes are compared with compile-time matched work and fixed g=9984 work,
//! not with all 10496 capacity rows. Full bit comparisons and independent FP64 dots check
//! numerical agreement, including repeated calls with changing work and schedule order.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Module;
use furiosa_opt_models::load::{Checkpoint, Context, Load};
use furiosa_opt_models::modules::parallel::Gathered;
use furiosa_opt_models::ops::attention::full::{Joined, Joint, Projection};
use furiosa_opt_models::ops::collective::{InTiles, Pieces};
use furiosa_opt_models::ops::linear::Grouped;
use furiosa_opt_models::ops::linear::grouped::Layout;
use furiosa_opt_models::tensor::axis::Rows;
use furiosa_opt_models::tensor::live::BlockPrefix;
use furiosa_opt_models::tensor::parallel::{Parallel, Shard};
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::{HbmBuffer, HostBuffer, launch};
use std::marker::PhantomData;

axes![InputWidth = 5376, Width = 1792, Head = 128, Rotated = 96];

type Capacity = Rows<10496>;
type Block = Rows<256>;
type Chips = m![{ Symbol<Shard<4>> }];
type Input = HbmTensor<bf16, Chips, m![Capacity, InputWidth]>;
type Output = HbmTensor<bf16, Chips, m![{ Symbol<Pieces<Capacity, 4>> }, Width]>;
type Qkv = (Output, Output, Output);
type Buffers = [HbmBuffer; 3];
type Owned = Projection<Grouped<InputWidth, Joint<Width>, Chips>, Head, Rotated>;
type Borrowed<'a> = Projection<
    Grouped<
        InputWidth,
        Joint<Width>,
        Chips,
        &'a HbmTensor<bf16, Chips, Layout<InputWidth, Joint<Width>>>,
    >,
    Head,
    Rotated,
>;
type Weights<'a> = Joined<Borrowed<'a>>;
type Tiles = InTiles<1024, 11, 256>;

const WORK: [usize; 3] = [9472, 9728, 9984];
const HEADS: usize = 14;

#[derive(Clone, Copy, Debug)]
enum Schedule {
    Runtime,
    Matched,
    Fixed2048,
}

impl Schedule {
    const ALL: [Self; 3] = [Self::Runtime, Self::Matched, Self::Fixed2048];

    fn rows(self, work: usize) -> usize {
        match self {
            Self::Fixed2048 => 9984,
            _ => work,
        }
    }
}

#[device(chip = 4)]
fn runtime_prefix(
    device: &mut Device,
    weights: &Weights<'_>,
    input: &Input,
    blocks: &Value<Chips>,
) -> Qkv {
    let prefix = BlockPrefix::<(), Capacity, Block>::read(blocks);
    Gathered::<_, Parallel<4>, Tiles>::new(weights)
        .forward(device, (), &mut (), &prefix.with(input))
        .into_value()
}

#[device(chip = 4)]
fn static_prefix<Work: AxisName>(
    device: &mut Device,
    weights: &Weights<'_>,
    input: &Input,
    _work: &PhantomData<Work>,
) -> Qkv {
    let prefix = BlockPrefix::<(), Capacity, Block>::fixed::<Work>();
    Gathered::<_, Parallel<4>, Tiles>::new(weights)
        .forward(device, (), &mut (), &prefix.with(input))
        .into_value()
}

async fn issue(
    device: &mut Device,
    weights: &Weights<'_>,
    input: &Input,
    blocks: &Value<Chips>,
    schedule: Schedule,
    work: usize,
) -> Qkv {
    let result = match schedule {
        Schedule::Runtime => launch(runtime_prefix, (device, weights, input, blocks)).await,
        _ => match schedule.rows(work) {
            9472 => {
                launch(
                    static_prefix,
                    (device, weights, input, &PhantomData::<Rows<9472>>),
                )
                .await
            }
            9728 => {
                launch(
                    static_prefix,
                    (device, weights, input, &PhantomData::<Rows<9728>>),
                )
                .await
            }
            9984 => {
                launch(
                    static_prefix,
                    (device, weights, input, &PhantomData::<Rows<9984>>),
                )
                .await
            }
            _ => unreachable!("host work is one of the three validated extents"),
        },
    };
    result.unwrap_or_else(|error| panic!("{schedule:?} g={work} launch: {error}"))
}

async fn ready(output: &Qkv, schedule: Schedule, work: usize) {
    for (term, tensor) in [&output.0, &output.1, &output.2].into_iter().enumerate() {
        tensor
            .ready()
            .await
            .unwrap_or_else(|error| panic!("{schedule:?} g={work} term={term} readiness: {error}"));
    }
}

// Both operands are small dyadic BF16 values. Products are integer multiples of 2^-10;
// even the sum of absolute numerators is below 2^23, so FP32 sums and the FP64 dot are exact.
fn coefficient(source_row: usize, column: usize) -> bf16 {
    let integer = ((source_row * 17
        + column * 13
        + source_row / 128 * 7
        + source_row / (3 * Width::SIZE) * 11)
        % 63) as i32
        - 31;
    bf16::from_f32(integer as f32 / 32.)
}

fn operand(chip: usize, row: usize, column: usize) -> bf16 {
    let integer = ((chip * 19 + row * 23 + column * 11 + row / 256 * 5) % 61) as i32 - 30;
    bf16::from_f32(integer as f32 / 32.)
}

// Independent source-dimension -> projected-dimension permutation, not the production helper.
fn paired96(dimension: usize) -> usize {
    match dimension {
        0..48 => dimension,
        48..96 => dimension + 16,
        96..112 => dimension - 48,
        112..128 => dimension,
        _ => unreachable!(),
    }
}

fn expected(
    chip: usize,
    input_part: usize,
    row: usize,
    term: usize,
    head: usize,
    d: usize,
) -> bf16 {
    let source_row = ((chip * HEADS + head) * 3 + term) * 128 + d;
    let dot: f64 = (0..InputWidth::SIZE)
        .map(|k| {
            f64::from(operand(input_part, row, k).to_f32())
                * f64::from(coefficient(source_row, k).to_f32())
        })
        .sum();
    bf16::from_f32(dot as f32)
}

fn checkpoint() -> (tempfile::TempDir, Checkpoint) {
    let weights: Vec<_> = (0..4 * 3 * Width::SIZE)
        .flat_map(|row| (0..InputWidth::SIZE).map(move |column| coefficient(row, column)))
        .collect();
    let bytes = bf16::to_buf(&weights);
    let view = safetensors::tensor::TensorView::new(
        safetensors::Dtype::BF16,
        vec![4 * 3 * Width::SIZE, InputWidth::SIZE],
        &bytes,
    )
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    safetensors::serialize_to_file(
        [("weight", view)],
        &None,
        &directory.path().join("weights.safetensors"),
    )
    .unwrap();
    let source = Checkpoint::new(directory.path(), Some(serde_json::json!({}))).unwrap();
    (directory, source)
}

// Download only initialized rows, with each schedule's compact part stride, on all four chips.
async fn rows(device: &mut Device, buffer: &HbmBuffer, first: usize, count: usize) -> HostBuffer {
    buffer
        .slice(
            ..,
            first * Width::SIZE * 2..(first + count) * Width::SIZE * 2,
        )
        .unwrap()
        .to_host(&mut device.pdma)
        .await
        .unwrap()
}

fn bits(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[2 * at], bytes[2 * at + 1]])
}

async fn validate(device: &mut Device, outputs: &[Buffers; 3], work: usize) {
    let mut sampled_rows = vec![0, 255, 256, 1279, 1280, work - 1];
    // Include the last full body and the final partial body, when distinct.
    sampled_rows.extend([8447, 8448]);
    sampled_rows.sort_unstable();
    sampled_rows.dedup();
    for (term, ((runtime, matched), fixed)) in outputs[0]
        .iter()
        .zip(&outputs[1])
        .zip(&outputs[2])
        .enumerate()
    {
        for part in 0..4 {
            let a = rows(device, runtime, part * work, work).await;
            let b = rows(device, matched, part * work, work).await;
            let c = rows(device, fixed, part * 9984, work).await;
            assert_eq!(a.len(), 4 * work * Width::SIZE * 2);
            assert_eq!(a.len(), b.len());
            assert_eq!(a.len(), c.len());
            for at in 0..a.len() / 2 {
                let actual = bits(&a, at);
                assert_eq!(
                    actual,
                    bits(&b, at),
                    "matched g={work} term={term} part={part} at={at}"
                );
                assert_eq!(
                    actual,
                    bits(&c, at),
                    "fixed2048 g={work} term={term} part={part} at={at}"
                );
            }
            for chip in 0..4 {
                for &row in &sampled_rows {
                    for head in [0, HEADS - 1] {
                        for d in [0, 47, 48, 95, 96, 111, 112, 127] {
                            let column = head * 128 + if term == 2 { d } else { paired96(d) };
                            let at = (chip * work + row) * Width::SIZE + column;
                            let reference = expected(chip, part, row, term, head, d);
                            assert_eq!(
                                bits(&a, at),
                                reference.to_bits(),
                                "FP64 g={work} chip={chip} part={part} row={row} term={term} head={head} source_dimension={d}"
                            );
                        }
                    }
                }
            }
        }
    }
}

// Repeated calls rotate the checked row, head and dimension as well as request/schedule order.
async fn spot_check(
    device: &mut Device,
    output: &Buffers,
    schedule: Schedule,
    work: usize,
    sample: usize,
) {
    let stride = schedule.rows(work);
    for (term, buffer) in output.iter().enumerate() {
        let row = [0, 255, 256, 1279, 1280, work - 1][sample % 6];
        for part in 0..4 {
            let bytes = rows(device, buffer, part * stride + row, 1).await;
            for chip in 0..4 {
                let head = (sample + part + chip) % HEADS;
                let d = [47, 48, 95, 96, 111, 112][sample % 6];
                let column = head * 128 + if term == 2 { d } else { paired96(d) };
                assert_eq!(
                    bits(&bytes, chip * Width::SIZE + column),
                    expected(chip, part, row, term, head, d).to_bits(),
                    "sample {sample} {schedule:?} g={work} term={term} part={part} chip={chip}"
                );
            }
        }
    }
}

fn buffers((query, key, value): Qkv) -> Buffers {
    [query.into_buffer(), key.into_buffer(), value.into_buffer()]
}

#[tokio::test(flavor = "current_thread")]
async fn runtime_gather_matches_static_work_and_partial_rotary() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let (directory, source) = checkpoint();
    let owned: Owned = source.load(&mut device).unwrap();
    let weights = Joined(Load::load(&mut Context::new(None), &owned).unwrap());
    drop((source, directory));
    let input: Input = HostTensor::<bf16, m![{ Chips }, Capacity, InputWidth]>::from_vec(
        (0..4).flat_map(|chip| {
            (0..Capacity::SIZE).flat_map(move |row| {
                (0..InputWidth::SIZE).map(move |column| {
                    if row < 9984 {
                        operand(chip, row, column)
                    } else {
                        bf16::from_bits(0x7fc1)
                    }
                })
            })
        }),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let mut counts = Vec::new();
    for work in WORK {
        let blocks = BlockPrefix::<(), Capacity, Block>::try_over(work)
            .unwrap()
            .blocks();
        let value: Value<Chips> = HostTensor::<i32, m![1]>::from_vec([blocks as i32])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        counts.push(value);
    }
    // Compare all three output sets together before exercising repeated individual calls.
    for (index, work) in WORK.into_iter().enumerate() {
        let mut outputs = Vec::new();
        for schedule in Schedule::ALL {
            let output = issue(
                &mut device,
                &weights,
                &input,
                &counts[index],
                schedule,
                work,
            )
            .await;
            ready(&output, schedule, work).await;
            outputs.push(buffers(output));
        }
        let outputs: [Buffers; 3] = outputs.try_into().ok().unwrap();
        validate(&mut device, &outputs, work).await;
        drop(outputs);
    }
    // Keep all nine rounds: each round selects a distinct oracle head/row/dimension combination.
    for round in 0..9 {
        // Rotate both request geometry and schedule across repeated calls.
        for offset in 0..3 {
            let index = (round + offset) % 3;
            let work = WORK[index];
            for position in 0..3 {
                let schedule = Schedule::ALL[(round + index + position) % 3];
                let output = issue(
                    &mut device,
                    &weights,
                    &input,
                    &counts[index],
                    schedule,
                    work,
                )
                .await;
                ready(&output, schedule, work).await;
                let output = buffers(output);
                spot_check(&mut device, &output, schedule, work, round).await;
                drop(output);
            }
        }
    }
    drop((owned, input, counts));
}
