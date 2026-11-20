//! Checked row transfers against host addresses, independently for Join and Split.
//!
//! No checkpoint data is used. Sources outside the declared initialized intervals are NaN;
//! copies preserve all BF16 bits, gaps become positive zero, and retained allocation tails keep
//! their previous values. Head outputs initialize the same aligned extensions used by the repro.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Op;
use furiosa_opt_models::models::diffusion::minimax_h3::packing::Capacity;
use furiosa_opt_models::ops::index::packed::{
    Join, Materialize, RowInputs, RowPlan, Rows, Span, Split,
};
use furiosa_opt_models::tensor::{axis, parallel::Shard};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;
use std::ops::Range;

type Chips = m![{ Symbol<Shard<4>> }];
axes![Width = 128, HeadWidth = 5376, NarrowHead = 64];
type Media<A, V, W = Width> = (Rows<V, W, Chips>, Rows<A, W, Chips>);
type Header<R, T, A, V> = RowInputs<R, T, A, V, Chips>;

#[derive(Clone, Debug)]
struct Case {
    initialized: [Range<usize>; 3],
    copies: [(Range<usize>, usize); 3],
    work: usize,
}

#[device(chip = 4)]
fn join<R: AxisName, T: AxisName, A: AxisName, V: AxisName>(
    device: &mut Device,
    media: &Media<A, V>,
    text: &Rows<T, Width, Chips>,
    header: &Header<R, T, A, V>,
    output: &mut Rows<R, Width, Chips>,
) {
    let plan = header.read();
    Join.forward(device, (media, text, output, &plan));
}

#[device(chip = 4)]
fn join_owned<R: AxisName, T: AxisName, A: AxisName, V: AxisName>(
    device: &mut Device,
    media: &Media<A, V>,
    text: &Rows<T, Width, Chips>,
    header: &Header<R, T, A, V>,
) -> Rows<R, Width, Chips> {
    let plan = header.read();
    Join.forward(device, (media, (text, &plan))).into_value()
}

#[device(chip = 4)]
fn split<R: AxisName, T: AxisName, A: AxisName, V: AxisName>(
    device: &mut Device,
    input: &Rows<R, Width, Chips>,
    header: &Header<R, T, A, V>,
) -> Media<A, V> {
    let plan = header.read();
    Split.forward(device, (input, &plan))
}

fn source(chip: usize, modality: usize, row: usize, column: usize) -> bf16 {
    // Some copied values are signed zero or NaN payloads: this boundary copies, not computes.
    match (row * 17 + column * 7 + chip * 3 + modality) % 257 {
        0 => bf16::from_bits(0x8000),
        1 => bf16::from_bits(0x7fc5),
        _ => bf16::from_f32(
            ((chip * 73 + modality * 53 + row * 11 + column * 7) % 251) as f32 / 16. - 8.,
        ),
    }
}

fn independent_packed(chip: usize, row: usize, column: usize) -> bf16 {
    bf16::from_f32(((chip * 101 + row * 23 + column * 13) % 509) as f32 / 32. - 7.)
}

fn untouched(chip: usize, row: usize, column: usize) -> bf16 {
    if (row + column + chip).is_multiple_of(7) {
        bf16::from_bits(0x7fc3)
    } else {
        bf16::from_f32(-32. - ((row + column + chip) % 32) as f32)
    }
}

async fn upload<R: AxisName, W: AxisName>(
    device: &mut Device,
    values: impl IntoIterator<Item = bf16>,
) -> Rows<R, W, Chips> {
    HostTensor::<bf16, m![{ Chips }, R, W]>::from_vec(values)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

async fn download<R: AxisName, W: AxisName>(
    device: &mut Device,
    values: &Rows<R, W, Chips>,
) -> Vec<bf16> {
    values
        .to_host::<m![{ Chips }, R, W]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec()
}

fn assert_bits(actual: &[bf16], expected: &[bf16], label: &str, case: usize) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}, case {case}, cell {index}"
        );
    }
}

async fn cases<R: AxisName, T: AxisName, A: AxisName, V: AxisName>(
    device: &mut Device,
    cases: &[(Case, RowPlan<R, T, A, V>)],
) {
    let mut expected: Vec<_> = (0..4)
        .flat_map(|chip| {
            (0..R::SIZE).flat_map(move |row| {
                (0..Width::SIZE).map(move |column| untouched(chip, row, column))
            })
        })
        .collect();
    // Reuse the same allocation while work grows and shrinks, without resetting its tail.
    let mut output = upload::<R, Width>(device, expected.clone()).await;
    for (case_index, (case, plan)) in cases.iter().enumerate() {
        let header = Header::upload(device, plan).await.unwrap();
        let values = |modality: usize, capacity: usize| {
            (0..4).flat_map(move |chip| {
                (0..capacity).flat_map(move |row| {
                    (0..Width::SIZE).map(move |column| {
                        if case.initialized[modality].contains(&row) {
                            source(chip, modality, row, column)
                        } else {
                            bf16::from_bits(0x7fc1)
                        }
                    })
                })
            })
        };
        let text = upload::<T, Width>(device, values(0, T::SIZE)).await;
        let media = (
            upload::<V, Width>(device, values(2, V::SIZE)).await,
            upload::<A, Width>(device, values(1, A::SIZE)).await,
        );
        launch(join, (&mut *device, &media, &text, &header, &mut output))
            .await
            .unwrap();
        // Host address oracle: initialize the work prefix, then place each source cell at its
        // destination. This does not use Join, Split, plan accessors or production strip logic.
        for chip in 0..4 {
            let start = chip * R::SIZE * Width::SIZE;
            expected[start..start + case.work * Width::SIZE].fill(bf16::from_bits(0));
            for (modality, (range, at)) in case.copies.iter().enumerate() {
                for (offset, row) in range.clone().enumerate() {
                    for column in 0..Width::SIZE {
                        expected[start + (at + offset) * Width::SIZE + column] =
                            source(chip, modality, row, column);
                    }
                }
            }
        }
        assert_bits(
            &download(device, &output).await,
            &expected,
            "Join",
            case_index,
        );
        if case_index == 1 {
            let owned = launch(join_owned, (&mut *device, &media, &text, &header))
                .await
                .unwrap();
            // An owned Join deliberately leaves its tail uninitialized: read only work on each
            // chip, keeping the allocation-capacity stride rather than globally truncating.
            let owned = owned.into_buffer();
            for chip in 0..4 {
                let buffer = owned
                    .slice(chip..chip + 1, ..case.work * Width::SIZE * 2)
                    .unwrap();
                let bytes = buffer.to_host(&mut device.pdma).await.unwrap();
                let actual: Vec<_> = bytes
                    .chunks_exact(2)
                    .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                    .collect();
                let at = chip * R::SIZE * Width::SIZE;
                assert_bits(
                    &actual,
                    &expected[at..at + case.work * Width::SIZE],
                    "owned Join",
                    case_index,
                );
            }
        }
        // Split receives an independent host tensor, so an inverse Join bug cannot cancel a
        // Split bug. Gaps and the uncomputed tail are NaN and must never appear in its outputs.
        let packed = upload::<R, Width>(
            device,
            (0..4).flat_map(|chip| {
                (0..R::SIZE).flat_map(move |row| {
                    (0..Width::SIZE).map(move |column| {
                        if case
                            .copies
                            .iter()
                            .any(|(range, at)| (*at..at + range.len()).contains(&row))
                        {
                            independent_packed(chip, row, column)
                        } else {
                            bf16::from_bits(0x7fc7)
                        }
                    })
                })
            }),
        )
        .await;
        let extracted = launch(split, (&mut *device, &packed, &header))
            .await
            .unwrap();
        for (modality, capacity, actual) in [
            (2, V::SIZE, download(device, &extracted.0).await),
            (1, A::SIZE, download(device, &extracted.1).await),
        ] {
            let (range, at) = &case.copies[modality];
            let mut expected = Vec::with_capacity(4 * capacity * Width::SIZE);
            for chip in 0..4 {
                for row in 0..capacity {
                    for column in 0..Width::SIZE {
                        expected.push(if range.contains(&row) {
                            independent_packed(chip, at + row - range.start, column)
                        } else {
                            bf16::from_bits(0)
                        });
                    }
                }
            }
            assert_bits(&actual, &expected, "independent Split", case_index);
        }
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn checked_join_split_preserve_addresses_and_work_guards() {
    type Transfers = RowPlan<axis::Rows<10496>, axis::Rows<512>, axis::Rows<112>, axis::Rows<9328>>;
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let requests: Vec<_> = [77usize, 129, 2048, 1, 77]
        .into_iter()
        .map(|tokens| {
            let text = tokens.div_ceil(64) * 16;
            let work = (text + 112 + 9328).div_ceil(256) * 256;
            let expected = Case {
                initialized: [0..text, 0..112, 0..9328],
                copies: [(0..text, 0), (0..112, text), (0..9328, text + 112)],
                work,
            };
            let plan = Capacity::default()
                .plan(tokens)
                .unwrap()
                .row_plan()
                .unwrap();
            (expected, plan)
        })
        .chain([(
            Case {
                initialized: [0..0, 0..0, 0..0],
                copies: [(0..0, 0), (0..0, 0), (0..0, 0)],
                work: 0,
            },
            Transfers::try_new([0..0, 0..0, 0..0], [(0..0, 0), (0..0, 0), (0..0, 0)], 0).unwrap(),
        )])
        .collect();
    cases::<axis::Rows<10496>, axis::Rows<512>, axis::Rows<112>, axis::Rows<9328>>(
        &mut device,
        &requests,
    )
    .await;
}

type Video = Span<50, 64>;
type Audio = Span<10, 16>;
type VideoHead = axis::Rows<80>;
type AudioHead = axis::Rows<32>;

#[device(chip = 4)]
fn materialize<W: AxisName>(
    device: &mut Device,
    media: &Media<Audio, Video, W>,
) -> (Rows<VideoHead, W, Chips>, Rows<AudioHead, W, Chips>) {
    Materialize::<VideoHead, AudioHead>::new().forward(device, media)
}

async fn materialize_case<W: AxisName>(device: &mut Device) {
    let values = |modality, capacity| {
        (0..4).flat_map(move |chip| {
            (0..capacity).flat_map(move |row| {
                (0..W::SIZE).map(move |column| source(chip, modality, row, column))
            })
        })
    };
    let media = (
        upload::<Video, W>(device, values(2, 64)).await,
        upload::<Audio, W>(device, values(1, 16)).await,
    );
    let output = launch(materialize, (&mut *device, &media)).await.unwrap();
    for (modality, copied, capacity, actual) in [
        (2, 64, 80, download(device, &output.0).await),
        (1, 16, 32, download(device, &output.1).await),
    ] {
        let expected: Vec<_> = (0..4)
            .flat_map(|chip| {
                (0..capacity).flat_map(move |row| {
                    (0..W::SIZE).map(move |column| {
                        if row < copied {
                            source(chip, modality, row, column)
                        } else {
                            bf16::from_bits(0)
                        }
                    })
                })
            })
            .collect();
        assert_bits(&actual, &expected, "Materialize", 0);
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn materialize_preserves_source_padding_and_initializes_head_extensions() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    materialize_case::<HeadWidth>(&mut device).await;
    materialize_case::<NarrowHead>(&mut device).await;
}
