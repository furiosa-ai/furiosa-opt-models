//! Replicated refiner attention with one 2048-row envelope and explicit scattered key visibility.
//!
//! Synthetic finite Q/K/V exercise all 56 heads on every chip. Both Compact and ordinary
//! attention use the shared envelope preparation and are checked against independent FP64
//! softmax values. This is an attention boundary contract, not an owning-refiner execution test.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Op;
use furiosa_opt_models::ops::attention::Attention;
use furiosa_opt_models::ops::attention::flash::{Compact, Head, Span, Visible};
use furiosa_opt_models::ops::index::packed;
use furiosa_opt_models::tensor::parallel::Shard;
use furiosa_opt_models::tensor::scalar::{F32, Value};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;
use std::marker::PhantomData;

axes![Width = 7168, Heads = 56, Spans = 3];

type Envelope = packed::Span<2048, 2048>;
type Chips = m![{ Symbol<Shard<4>> }];
type Rows = packed::Rows<Envelope, Width, Chips>;
type Mask = Visible<Spans, Chips>;
type Operands<'a> = (&'a Rows, &'a Rows, &'a Rows, &'a Mask, &'a Value<Chips>);
type Ordinary = Attention<PhantomData<Heads>, F32<SCALE>>;

const SCALE: usize = (std::f32::consts::FRAC_1_SQRT_2 / 8.).to_bits() as usize;
const KEY_BOUND: usize = 6144;
const CLASSES: usize = 7;
// Existing permanent Compact attention gates use a 1e-2 relative RMS bound.
const LIMIT: f64 = 1e-2;

#[derive(Clone, Copy, Debug)]
struct Case {
    valid: usize,
    scattered: bool,
    uniform: bool,
}

#[device(chip = 4)]
fn attend<A>(device: &mut Device, attention: &A, (q, k, v, mask, bound): Operands<'_>) -> Rows
where
    A: DeviceSend,
    A: for<'a> Op<(&'a Rows, &'a Rows, &'a Rows, (&'a Mask, &'a Value<Chips>)), Output = Rows>,
{
    attention.forward(device, (q, k, v, (mask, bound)))
}

fn amplitude(chip: usize, row: usize, head: usize, uniform: bool) -> bf16 {
    let integer = (row % CLASSES) as i32 - 3 + (head % 3) as i32 - (chip % 2) as i32;
    bf16::from_f32(if uniform { 0. } else { integer as f32 * 4. })
}

fn key(row: usize, valid: usize, channel: usize) -> bf16 {
    let value = if channel != 0 {
        0.
    } else if row < valid {
        [-1., -0.5, 0.5, 1.][row % 4]
    } else {
        // Distracting padding is finite, and would dominate an accidentally unmasked key.
        6. + (row % 3) as f32
    };
    bf16::from_f32(value)
}

fn value(chip: usize, row: usize, head: usize, channel: usize, valid: usize) -> bf16 {
    let base = (32 + (chip * 5 + head * 3 + channel * 7) % 29) as f32 / 64.;
    let part = [-3., -1., 1., 3.][row % 4] / 16.;
    bf16::from_f32(base + part + if row < valid { 0. } else { 16. })
}

fn visible(case: Case, position: usize) -> bool {
    if case.scattered {
        // First/last keys of all three spans. A token-count bound would miss later spans.
        [0, 2047, 2048, 4095, 4096, 6143].contains(&position)
    } else {
        position < case.valid
    }
}

async fn upload(device: &mut Device, target: &mut Rows, case: Case, term: usize) {
    let values = (0..4).flat_map(|chip| {
        (0..Envelope::SIZE).flat_map(move |row| {
            (0..Width::SIZE).map(move |column| {
                let (head, channel) = (column / Head::SIZE, column % Head::SIZE);
                match term {
                    0 if channel == 0 => amplitude(chip, row, head, case.uniform),
                    0 => bf16::from_f32(0.),
                    1 => key(row, case.valid, channel),
                    _ => value(chip, row, head, channel, case.valid),
                }
            })
        })
    });
    HostTensor::<bf16, m![{ Chips }, Envelope, Width]>::from_vec(values)
        .to_hbm(&mut device.pdma)
        .output(target)
        .await
        .unwrap();
}

// The source formula has four key/value classes. Count actual visible key positions, including
// repeated source rows in later spans; no production packing or attention helper is used here.
fn oracle(case: Case) -> Vec<f64> {
    let mut counts = [0usize; 4];
    for position in 0..KEY_BOUND {
        if visible(case, position) {
            counts[position % Envelope::SIZE % 4] += 1;
        }
    }
    assert!(counts.iter().sum::<usize>() > 0);
    let mut result = Vec::with_capacity(4 * Heads::SIZE * CLASSES * Head::SIZE);
    for chip in 0..4 {
        for head in 0..Heads::SIZE {
            for class in 0..CLASSES {
                let q = f64::from(amplitude(chip, class, head, case.uniform).to_f32());
                let masses: [f64; 4] = std::array::from_fn(|kind| {
                    let k = f64::from(key(kind, 2048, 0).to_f32());
                    counts[kind] as f64 * (q * k * f64::from(f32::from_bits(SCALE as u32))).exp()
                });
                let denominator = masses.iter().sum::<f64>();
                for channel in 0..Head::SIZE {
                    let numerator = (0..4)
                        .map(|kind| {
                            masses[kind]
                                * f64::from(value(chip, kind, head, channel, 2048).to_f32())
                        })
                        .sum::<f64>();
                    result.push(numerator / denominator);
                }
            }
        }
    }
    result
}

fn check(actual: &[bf16], expected: &[f64], case: Case, variant: &str) {
    assert_eq!(actual.len(), 4 * Envelope::SIZE * Width::SIZE);
    for chip in 0..4 {
        for row in 0..Envelope::SIZE {
            for head in 0..Heads::SIZE {
                let mut error = 0.;
                let mut square = 0.;
                for channel in 0..Head::SIZE {
                    let at = ((chip * Heads::SIZE + head) * CLASSES + row % CLASSES) * Head::SIZE
                        + channel;
                    let want = expected[at];
                    let column = head * Head::SIZE + channel;
                    let got = f64::from(
                        actual[(chip * Envelope::SIZE + row) * Width::SIZE + column].to_f32(),
                    );
                    assert!(
                        got.is_finite(),
                        "{variant} {case:?} chip={chip} row={row} head={head} channel={channel}"
                    );
                    error += (got - want).powi(2);
                    square += want.powi(2);
                }
                let relative = (error / square).sqrt();
                assert!(
                    relative <= LIMIT,
                    "{variant} {case:?} chip={chip} row={row} head={head} relative_RMS={relative}, limit={LIMIT}"
                );
            }
        }
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn compact_and_ordinary_refiner_envelopes_preserve_masks_and_values() {
    const { assert!(Heads::SIZE * Head::SIZE == Width::SIZE && Spans::SIZE * Span::SIZE == KEY_BOUND) };
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let mut q = Rows::alloc(&device).unwrap();
    let mut k = Rows::alloc(&device).unwrap();
    let mut v = Rows::alloc(&device).unwrap();
    let mut mask = Mask::alloc(&device).unwrap();
    let bound: Value<Chips> = HostTensor::<i32, m![1]>::from_vec([KEY_BOUND as i32])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let compact = Compact::<Heads, SCALE>::default();
    let ordinary = Ordinary::default();
    let mut cases: Vec<_> = [128, 129, 511, 1025, 2047, 2048, 1, 128, 2048, 129, 128]
        .into_iter()
        .enumerate()
        .map(|(index, valid)| Case {
            valid,
            scattered: false,
            uniform: index == 0,
        })
        .collect();
    cases.insert(
        6,
        Case {
            valid: 2048,
            scattered: true,
            uniform: false,
        },
    );
    for (index, case) in cases.into_iter().enumerate() {
        // Reuse allocations and rewrite every cell: shrink after growth cannot retain old keys.
        upload(&mut device, &mut q, case, 0).await;
        upload(&mut device, &mut k, case, 1).await;
        upload(&mut device, &mut v, case, 2).await;
        HostTensor::<bf16, m![{ Chips }, Spans, Span]>::from_vec((0..4).flat_map(|_| {
            (0..KEY_BOUND).map(|position| bf16::from_f32(f32::from(visible(case, position))))
        }))
        .to_hbm(&mut device.pdma)
        .output(&mut mask)
        .await
        .unwrap();
        let expected = oracle(case);
        // Alternate variant order while reusing the rewritten operands and mask.
        for use_compact in [index % 2 == 0, index % 2 != 0] {
            let variant = if use_compact { "Compact" } else { "ordinary" };
            let output = if use_compact {
                launch(attend, (&mut device, &compact, (&q, &k, &v, &mask, &bound)))
                    .await
                    .unwrap()
            } else {
                launch(
                    attend,
                    (&mut device, &ordinary, (&q, &k, &v, &mask, &bound)),
                )
                .await
                .unwrap()
            };
            output.ready().await.unwrap();
            let values = output
                .to_host::<m![{ Chips }, Envelope, Width]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            check(&values, &expected, case, variant);
            drop(output);
        }
    }
}
