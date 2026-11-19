//! Checked valid-run plans and transport through the production K/V transpose and finite padding.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::models::diffusion::minimax_h3::packing::{Capacity, Modality, Plan};
use furiosa_opt_models::ops::attention::flash::{Duo, Head, HeadPairs, Span, Tokens};
use furiosa_opt_models::ops::attention::full::{PackInputs, PackKeys, PackPlan};
use furiosa_opt_models::tensor::axis::Rows;
use furiosa_opt_models::tensor::parallel::Shard;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::{HbmBuffer, launch};

type Chips = m![{ Symbol<Shard<4>> }];
type Source<R, W> = HbmTensor<bf16, Chips, m![R, W]>;
type Output<S, W> = Tokens<S, HeadPairs<W>, Chips>;
type Outputs<S, W> = (Output<S, W>, Output<S, W>);
type Request<R, S> = PackInputs<R, S, Chips, 12>;
type Checked<R, S> = PackPlan<R, S, 12>;
axes![
    Width = 1792,
    ToyWidth = 256,
    Three = 3,
    Nineteen = 19,
    Twenty = 20
];
type H3Rows = Rows<41984>;
const POISON: u16 = 0x7fc1;

fn checked<R: AxisName, S: AxisName>(plan: Plan) -> Checked<R, S> {
    assert_eq!(plan.capacity().query_capacity(), R::SIZE);
    assert_eq!(plan.capacity().key_rows(), S::SIZE * Span::SIZE);
    let pack = PackPlan::try_from_runs(
        0..plan.query_rows(),
        plan.key_runs().map(|run| (run.source(), run.destination())),
    )
    .unwrap();
    assert_eq!(pack.key_count(), plan.key_count());
    pack
}

#[test]
fn host_plan_rejects_invalid_prefixes_ranges_counts_and_overflow() {
    type P = PackPlan<Rows<4096>, Three, 2>;
    for runs in [
        vec![(12..13, 1..2)],                                 // destination gap
        vec![(12..14, 0..2), (20..21, 1..2)],                 // overlap
        vec![(12..14, 0..3)],                                 // unequal lengths
        vec![(11..13, 0..2)],                                 // before source initialization
        vec![(100..102, 0..2)],                               // after source initialization
        vec![(12..12, 0..0)],                                 // individual empty run
        vec![(12..13, 0..1), (13..14, 1..2), (14..15, 2..3)], // slots exhausted
    ] {
        assert_eq!(
            P::try_from_runs(12..101, runs).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
    }
    assert!(P::try_from_runs(0..4097, []).is_err());
    assert!(P::try_from_runs(std::ops::Range { start: 40, end: 30 }, []).is_err());
    type Small = PackPlan<Rows<8192>, Rows<1>, 1>;
    assert!(Small::try_from_runs(0..8192, [(0..2049, 0..2049)]).is_err());
    type Huge = Rows<{ usize::MAX }>;
    assert!(PackPlan::<Huge, Three, 1>::try_from_runs(0..i32::MAX as usize + 1, []).is_err());
    assert!(PackPlan::<Rows<1>, Huge, 1>::try_from_runs(0..1, []).is_err());
}

#[test]
fn transport_uses_rows_preserves_repeated_sources_and_zeroes_unused_slots() {
    let plan = PackPlan::<Rows<8192>, Three, 4>::try_from_runs(
        37..8001,
        [(53..2123, 0..2070), (64..67, 2070..2073)],
    )
    .unwrap();
    let (header, runs) = plan.row_counts();
    assert_eq!(header, [37, 7964, 2, 2073]);
    assert_eq!(runs, [[53, 2070], [64, 3], [0, 0], [0, 0]]);
    assert_eq!(
        plan.runs().collect::<Vec<_>>(),
        [(53..2123, 0..2070), (64..67, 2070..2073)]
    );
    let empty = PackPlan::<Rows<8192>, Three, 4>::try_from_runs(8192..8192, []).unwrap();
    assert_eq!(empty.row_counts(), ([8192, 0, 0, 0], [[0, 0]; 4]));
    assert_eq!(empty.key_count(), 0);
}

#[test]
fn h3_valid_runs_match_semantic_addresses_and_exact_capacity_boundary() {
    for (maximum, text) in [(2048, 77), (2048, 129), (2048, 2048), (1202, 1202)] {
        let model = Capacity::new(maximum).unwrap().plan(text).unwrap();
        let runs: Vec<_> = model
            .key_runs()
            .map(|run| (run.source(), run.destination()))
            .collect();
        let (end, count) = if maximum == 1202 {
            let pack = checked::<H3Rows, Nineteen>(model);
            (pack.key_count(), pack.copy_count())
        } else {
            let pack = checked::<H3Rows, Twenty>(model);
            (pack.key_count(), pack.copy_count())
        };
        assert_eq!(end, 37710 + text);
        if text == 77 {
            assert_eq!(count, 49);
        }
        let mut seen = 0;
        for (source, destination) in runs {
            for (source, key) in source.zip(destination) {
                assert_eq!(key, seen);
                assert_eq!(semantic_source(model, key), source);
                seen += 1;
            }
        }
        assert_eq!(seen, end);
        if maximum == 1202 {
            assert_eq!(end, 38912);
            assert_eq!(model.text_rows_per_part() * 4 + 37710, 38926);
        }
    }
}

// A different lookup from the transfer-run iterator: keys are semantic video/audio/text rows.
fn semantic_source(plan: Plan, key: usize) -> usize {
    let (modality, row) = if key < 37296 {
        (Modality::Video, key)
    } else if key < 37710 {
        (Modality::Audio, key - 37296)
    } else {
        (Modality::Text, key - 37710)
    };
    plan.locate(modality, row).unwrap().query_row()
}

// Exact finite BF16 payload, sensitive to chip, source row, head, feature and K/V plane.
fn value(plane: usize, chip: usize, row: usize, column: usize) -> bf16 {
    let mut hash = (row as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (column as u64 + 1).wrapping_mul(0xbf58476d1ce4e5b9)
        ^ (chip as u64 + 1).wrapping_mul(0x94d049bb133111eb)
        ^ (plane as u64 + 1).wrapping_mul(0x2545f4914f6cdd1d);
    hash ^= hash >> 31;
    bf16::from_bits((0x2000 + (hash % 0x3f00) as u16) | ((hash >> 32) as u16 & 0x8000))
}

#[device(chip = 4)]
fn pack_kv<R: AxisName, S: AxisName, W: AxisName>(
    device: &mut Device,
    request: &Request<R, S>,
    input: (&Source<R, W>, &Source<R, W>),
    output: &mut Outputs<S, W>,
) {
    let plan = request.read();
    let interval = plan.source_work();
    let op = PackKeys::<S>::default();
    op.write(device, interval.with(input.0), &plan, &mut output.0);
    op.write(device, interval.with(input.1), &plan, &mut output.1);
}

async fn input<R: AxisName, W: AxisName>(
    device: &mut Device,
    plane: usize,
    runs: &[(std::ops::Range<usize>, std::ops::Range<usize>)],
) -> Source<R, W> {
    let mut data = vec![bf16::from_bits(POISON); 4 * R::SIZE * W::SIZE];
    for chip in 0..4 {
        for (source, _) in runs {
            for row in source.clone() {
                for column in 0..W::SIZE {
                    data[(chip * R::SIZE + row) * W::SIZE + column] =
                        value(plane, chip, row, column);
                }
            }
        }
    }
    HostTensor::<bf16, m![{ Chips }, R, W]>::from_vec(data)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

async fn output<S: AxisName, W: AxisName>(device: &mut Device) -> Output<S, W> {
    let buffer = HbmBuffer::alloc_shared(device, S::SIZE * Span::SIZE * W::SIZE * 2).unwrap();
    let mut output = buffer.try_into().unwrap();
    // A nonzero finite pattern makes incomplete clearing distinguishable from NaN propagation.
    HostTensor::<bf16, m![{ Chips },{ Symbol<HeadPairs<W>> },Duo,S,Span,Head]>::from_vec(
        vec![bf16::from_bits(0x42a5); 4 * S::SIZE * Span::SIZE * W::SIZE],
    )
    .to_hbm(&mut device.pdma)
    .output(&mut output)
    .await
    .unwrap();
    output
}

async fn check<S: AxisName, W: AxisName>(
    device: &mut Device,
    buffer: HbmBuffer,
    plane: usize,
    sources: &[usize],
) {
    let capacity = S::SIZE * Span::SIZE;
    let row_bytes = Head::SIZE * 2;
    let mut checked = 0;
    // Read one head at a time in bounded chunks, independently of the device's run tiling.
    for head in 0..W::SIZE / Head::SIZE {
        for first in (0..capacity).step_by(512) {
            let rows = (capacity - first).min(512);
            let at = (head * capacity + first) * row_bytes;
            let data = buffer
                .slice(.., at..at + rows * row_bytes)
                .unwrap()
                .to_host(&mut device.pdma)
                .await
                .unwrap();
            for chip in 0..4 {
                for row in 0..rows {
                    for dim in 0..Head::SIZE {
                        let key = first + row;
                        let at = ((chip * rows + row) * Head::SIZE + dim) * 2;
                        let got = u16::from_le_bytes([data[at], data[at + 1]]);
                        let want = if key < sources.len() {
                            value(plane, chip, sources[key], head * Head::SIZE + dim).to_bits()
                        } else {
                            0
                        };
                        assert_eq!(
                            got, want,
                            "plane {plane}, chip {chip}, head {head}, key {key}, dim {dim}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "K/V plane {plane}: {checked} exact BF16 values, valid {} and zero suffix {}",
        sources.len(),
        capacity - sources.len()
    );
}

async fn run_h3<S: AxisName>(device: &mut Device, texts: &[usize], maximum: usize) {
    let mut outputs: Outputs<S, Width> = (output(device).await, output(device).await);
    for &text in texts {
        let model = Capacity::new(maximum).unwrap().plan(text).unwrap();
        let plan = checked::<H3Rows, S>(model);
        let runs: Vec<_> = plan.runs().collect();
        let query = input::<H3Rows, Width>(device, 0, &runs).await;
        let value = input::<H3Rows, Width>(device, 1, &runs).await;
        let request = Request::upload(device, &plan).await.unwrap();
        launch(
            pack_kv,
            (&mut *device, &request, (&query, &value), &mut outputs),
        )
        .await
        .unwrap();
        let sources: Vec<_> = (0..model.key_count())
            .map(|key| semantic_source(model, key))
            .collect();
        // Retain the allocations across changing requests, so stale valid keys must be cleared.
        let (key, value) = outputs;
        let key = key.into_buffer();
        let value = value.into_buffer();
        check::<S, Width>(device, key.alias(..).unwrap(), 0, &sources).await;
        check::<S, Width>(device, value.alias(..).unwrap(), 1, &sources).await;
        outputs = (key.try_into().unwrap(), value.try_into().unwrap());
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn h3_valid_runs_pack_exactly_and_clear_stale_keys() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    run_h3::<Twenty>(&mut device, &[77, 2048, 129, 77], 2048).await;
    run_h3::<Nineteen>(&mut device, &[1202], 1202).await;
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn unaligned_runs_cross_spans_and_empty_plan_clears_every_key() {
    type R = Rows<8192>;
    let lengths = [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 2051];
    let mut next = 0;
    let runs: Vec<_> = lengths
        .into_iter()
        .enumerate()
        .map(|(i, n)| {
            let destination = next..next + n;
            next += n;
            (13 + 17 * i..13 + 17 * i + n, destination)
        })
        .collect();
    let plan = Checked::<R, Three>::try_from_runs(13..8180, runs).unwrap();
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let mut outputs: Outputs<Three, ToyWidth> =
        (output(&mut device).await, output(&mut device).await);
    for plan in [plan, Checked::try_from_runs(8192..8192, []).unwrap()] {
        let runs: Vec<_> = plan.runs().collect();
        let query = input::<R, ToyWidth>(&mut device, 0, &runs).await;
        let value = input::<R, ToyWidth>(&mut device, 1, &runs).await;
        let request = Request::upload(&mut device, &plan).await.unwrap();
        launch(
            pack_kv,
            (&mut device, &request, (&query, &value), &mut outputs),
        )
        .await
        .unwrap();
        let sources: Vec<_> = runs.iter().flat_map(|(source, _)| source.clone()).collect();
        let (key, value) = outputs;
        let key = key.into_buffer();
        let value = value.into_buffer();
        check::<Three, ToyWidth>(&mut device, key.alias(..).unwrap(), 0, &sources).await;
        check::<Three, ToyWidth>(&mut device, value.alias(..).unwrap(), 1, &sources).await;
        outputs = (key.try_into().unwrap(), value.try_into().unwrap());
    }
}
