//! Bounded HBM head normalization and rotary, checked with synthetic inputs and an FP64 oracle.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Module;
use furiosa_opt_models::ops::attention::full::HeadNorm;
use furiosa_opt_models::ops::rotary::{Rotary, Rotation};
use furiosa_opt_models::tensor::axis::Rows;
use furiosa_opt_models::tensor::live::Live;
use furiosa_opt_models::tensor::parallel::Shard;
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::{HbmBuffer, launch};
use std::marker::PhantomData;

axes![Head = 128, Rotated = 96, Width = 1792];
type Count = Rows<41984>;
type Chips = m![{ Symbol<Shard<4>> }];
type Data = HbmTensor<bf16, Chips, m![Count, Width]>;
type Gain = HbmTensor<f32, Chips, m![Head]>;
type Epsilon = HbmTensor<f32, Chips, m![1]>;
type Norm = HeadNorm<Head, Chips, Gain, Epsilon, EPSILON, Rotated>;
type Positions = Rotation<Head, Chips, Count>;
type Outputs = (Data, Data, Data, Data);
type Inputs<'a> = (
    &'a Data,
    &'a Data,
    &'a Positions,
    &'a Value<Chips>,
    &'a Value<Chips>,
);
const EPSILON: usize = 1e-5_f32.to_bits() as usize;
const POISON: u16 = 0x7fc1;
const LIMIT: f64 = 1e-2;
const HEADS: usize = Width::SIZE / Head::SIZE;
const PATTERNS: usize = 17;
const PHASES: usize = 13;

#[derive(DeviceSend)]
struct Qk {
    query: Norm,
    key: Norm,
}

impl Qk {
    fn write(
        &self,
        device: &mut Device,
        interval: Live<(), Count>,
        (query, key, positions): (&Data, &Data, &Positions),
        (nq, nk, rq, rk): &mut Outputs,
    ) {
        self.query.write_live(device, interval.with(query), nq);
        self.key.write_live(device, interval.with(key), nk);
        Rotary::<Head, Head, bf16, Chips>::new().write_live(
            device,
            (interval.with((&*nq, &*nk)), positions),
            (rq, rk),
        );
    }
}

impl<'a> Module<Inputs<'a>, (), &'a mut Outputs> for Qk {
    type Output = ();

    fn forward(&self, device: &mut Device, _: (), output: &'a mut Outputs, input: Inputs<'a>) {
        let (query, key, positions, start, count) = input;
        let interval = Live::<(), Count>::all()
            .subrange(start.extent::<m![Count]>(), count.extent::<m![Count]>());
        self.write(device, interval, (query, key, positions), output);
    }
}

#[device(chip = 4)]
fn live_qk(device: &mut Device, module: &Qk, input: Inputs<'_>, output: &mut Outputs) {
    module.forward(device, (), output, input);
}

// Independent piecewise permutation, rather than calling the production pairing helper.
fn paired(d: usize) -> usize {
    match d {
        0..48 => d,
        48..96 => d + 16,
        96..112 => d - 48,
        _ => d,
    }
}

fn rounded(value: f64) -> bf16 {
    bf16::from_f32(value as f32)
}

fn sample(kind: usize, pattern: usize, original: usize) -> bf16 {
    bf16::from_f32(
        (((pattern * 5 + original * 3 + kind * 11) % 31) as f32 / 16.0 - 15.0 / 16.0)
            * 2.0_f32.powi((pattern % 5) as i32 - 2),
    )
}

fn gain(kind: usize, chip: usize, original: usize) -> f32 {
    1.0 + ((original * 7 + chip * 3 + kind * 5) % 19) as f32 / 32.0
}

fn coordinates(class: usize) -> [f64; 3] {
    [
        0.25 + class as f64 / 8.0,
        -1.0 - class as f64 / 16.0,
        1.0 + class as f64 / 4.0,
    ]
}

struct Oracle {
    normalized: Vec<bf16>,
    rotated: Vec<bf16>,
}

impl Oracle {
    fn new() -> Self {
        let mut normalized = vec![bf16::from_f32(0.0); 2 * 4 * PATTERNS * Head::SIZE];
        for kind in 0..2 {
            for chip in 0..4 {
                for pattern in 0..PATTERNS {
                    let square = (0..128)
                        .map(|d| f64::from(sample(kind, pattern, d).to_f32()).powi(2))
                        .sum::<f64>()
                        / 128.0;
                    let inverse = 1.0 / (square + f64::from(f32::from_bits(EPSILON as u32))).sqrt();
                    for d in 0..128 {
                        let value = f64::from(sample(kind, pattern, d).to_f32())
                            * inverse
                            * f64::from(gain(kind, chip, d));
                        normalized[Self::norm_at(kind, chip, pattern, paired(d))] = rounded(value);
                    }
                }
            }
        }
        let mut rotated = vec![bf16::from_f32(0.0); 2 * 4 * PATTERNS * PHASES * Head::SIZE];
        for kind in 0..2 {
            for chip in 0..4 {
                for pattern in 0..PATTERNS {
                    for class in 0..PHASES {
                        let position = coordinates(class);
                        for angle in 0..64 {
                            let low = f64::from(
                                normalized[Self::norm_at(kind, chip, pattern, angle)].to_f32(),
                            );
                            let high = f64::from(
                                normalized[Self::norm_at(kind, chip, pattern, 64 + angle)].to_f32(),
                            );
                            let (cosine, sine) = if angle < 48 {
                                let phase = position[angle / 16]
                                    / 10000_f64.powf((angle % 16) as f64 / 16.0);
                                (
                                    f64::from(rounded(phase.cos()).to_f32()),
                                    f64::from(rounded(phase.sin()).to_f32()),
                                )
                            } else {
                                (1.0, 0.0)
                            };
                            rotated[Self::rope_at(kind, chip, pattern, class, angle)] =
                                rounded(low * cosine - high * sine);
                            rotated[Self::rope_at(kind, chip, pattern, class, 64 + angle)] =
                                rounded(low * sine + high * cosine);
                        }
                    }
                }
            }
        }
        Self {
            normalized,
            rotated,
        }
    }

    fn norm_at(kind: usize, chip: usize, pattern: usize, d: usize) -> usize {
        ((kind * 4 + chip) * PATTERNS + pattern) * 128 + d
    }

    fn rope_at(kind: usize, chip: usize, pattern: usize, class: usize, d: usize) -> usize {
        (((kind * 4 + chip) * PATTERNS + pattern) * PHASES + class) * 128 + d
    }

    fn row(&self, plane: usize, chip: usize, row: usize, head: usize) -> &[bf16] {
        let pattern = (row * 3 + head * 5 + chip * 7) % PATTERNS;
        let at = if plane < 2 {
            Self::norm_at(plane, chip, pattern, 0)
        } else {
            Self::rope_at(plane - 2, chip, pattern, row % PHASES, 0)
        };
        let table = if plane < 2 {
            &self.normalized
        } else {
            &self.rotated
        };
        &table[at..at + 128]
    }
}

fn touched(start: usize, count: usize, tile: usize) -> std::ops::Range<usize> {
    if count == 0 {
        return start..start;
    }
    start / tile * tile..(start + count).div_ceil(tile) * tile
}

async fn scalar(device: &mut Device, n: usize) -> Value<Chips> {
    HostTensor::<i32, m![{ Chips }, 1]>::from_vec([n as i32; 4])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

async fn module(device: &mut Device) -> Qk {
    let mut norms = Vec::new();
    for kind in 0..2 {
        let mut weights = vec![0.0; 4 * 128];
        for chip in 0..4 {
            for d in 0..128 {
                weights[chip * 128 + paired(d)] = gain(kind, chip, d);
            }
        }
        norms.push(HeadNorm {
            weight: HostTensor::<f32, m![{ Chips }, Head]>::from_vec(weights)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            epsilon: HostTensor::<f32, m![{ Chips }, 1]>::from_vec(
                [f32::from_bits(EPSILON as u32); 4],
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
            head: PhantomData,
        });
    }
    Qk {
        key: norms.pop().unwrap(),
        query: norms.pop().unwrap(),
    }
}

async fn positions(device: &mut Device) -> Positions {
    // Production table arithmetic is FP32 before rounding. The oracle above uses independent
    // FP64 trigonometry and pair equations; it does not read these tables or call rotary code.
    let mut low = vec![bf16::from_f32(0.0); Count::SIZE * 128];
    let mut high = low.clone();
    for row in 0..Count::SIZE {
        let position = coordinates(row % PHASES);
        for angle in 0..64 {
            let (cosine, sine) = if angle < 48 {
                let phase = position[angle / 16] as f32
                    * (1.0 / 10000_f32.powf((angle % 16) as f32 / 16.0));
                (bf16::from_f32(phase.cos()), bf16::from_f32(phase.sin()))
            } else {
                (bf16::from_f32(1.0), bf16::from_f32(0.0))
            };
            low[row * 128 + angle] = cosine;
            low[row * 128 + 64 + angle] = sine;
            high[row * 128 + angle] = bf16::from_bits(sine.to_bits() ^ 0x8000);
            high[row * 128 + 64 + angle] = cosine;
        }
    }
    let low = HostTensor::<bf16, m![{ Chips }, Count, Head]>::from_vec(low.repeat(4))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let high = HostTensor::<bf16, m![{ Chips }, Count, Head]>::from_vec(high.repeat(4))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    // A shared host image is replicated to the selected chips.
    Positions { low, high }
}

async fn input(device: &mut Device, kind: usize, initialized: std::ops::Range<usize>) -> Data {
    let mut image = vec![bf16::from_bits(POISON); 4 * Count::SIZE * Width::SIZE];
    for chip in 0..4 {
        for row in initialized.clone() {
            for head in 0..HEADS {
                let at = (chip * Count::SIZE + row) * Width::SIZE + head * 128;
                let pattern = (row * 3 + head * 5 + chip * 7) % PATTERNS;
                for d in 0..128 {
                    image[at + paired(d)] = sample(kind, pattern, d);
                }
            }
        }
    }
    HostTensor::<bf16, m![{ Chips }, Count, Width]>::from_vec(image)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

async fn output(device: &mut Device) -> Data {
    let buffer = HbmBuffer::alloc_shared(device, Count::SIZE * Width::SIZE * 2).unwrap();
    let mut output: Data = buffer.try_into().unwrap();
    HostTensor::<bf16, m![{ Chips }, Count, Width]>::from_vec(vec![
        bf16::from_bits(POISON);
        4 * Count::SIZE * Width::SIZE
    ])
    .to_hbm(&mut device.pdma)
    .output(&mut output)
    .await
    .unwrap();
    output
}

async fn check(device: &mut Device, buffers: Outputs, start: usize, count: usize, oracle: &Oracle) {
    for (plane, tensor) in [buffers.0, buffers.1, buffers.2, buffers.3]
        .into_iter()
        .enumerate()
    {
        let buffer = tensor.into_buffer();
        let written = touched(start, count, if plane < 2 { 1024 } else { 256 });
        let (mut error, mut norm) = ([0.0_f64; 4 * HEADS], [0.0_f64; 4 * HEADS]);
        let mut worst = 0.0_f64;
        for first in (0..Count::SIZE).step_by(256) {
            let rows = (Count::SIZE - first).min(256);
            let bytes = buffer
                .slice(
                    ..,
                    first * Width::SIZE * 2..(first + rows) * Width::SIZE * 2,
                )
                .unwrap()
                .to_host(&mut device.pdma)
                .await
                .unwrap();
            for chip in 0..4 {
                for local in 0..rows {
                    let row = first + local;
                    let at = (chip * rows + local) * Width::SIZE * 2;
                    for head in 0..HEADS {
                        let expected = oracle.row(plane, chip, row, head);
                        for (d, want) in expected.iter().enumerate() {
                            let at = at + (head * 128 + d) * 2;
                            let bits = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
                            if !written.contains(&row) {
                                assert_eq!(
                                    bits, POISON,
                                    "plane {plane}, chip {chip}, untouched row {row}, head {head}, dim {d}"
                                );
                                continue;
                            }
                            let got = f64::from(bf16::from_bits(bits).to_f32());
                            assert!(
                                got.is_finite(),
                                "plane {plane}, chip {chip}, row {row}, head {head}, dim {d}"
                            );
                            let want = f64::from(want.to_f32());
                            let diff = got - want;
                            let group = chip * HEADS + head;
                            error[group] += diff * diff;
                            norm[group] += want * want;
                            worst = worst.max(diff.abs());
                        }
                    }
                }
            }
        }
        let mut maximum = 0.0_f64;
        for group in 0..4 * HEADS {
            if count == 0 {
                continue;
            }
            let relative = (error[group] / norm[group]).sqrt();
            assert!(
                relative < LIMIT,
                "plane {plane}, group {group}: relative L2 {relative}, start {start}, count {count}"
            );
            maximum = maximum.max(relative);
        }
        eprintln!(
            "Live QK start {start}, count {count}, plane {plane}: rL2 {maximum:.6e}, worst abs {worst:.6e}; all {} values/tails checked",
            4 * Count::SIZE * Width::SIZE
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn live_hbm_norm_and_rope_match_fp64_and_preserve_unwritten_rows() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let module = module(&mut device).await;
    let positions = positions(&mut device).await;
    let oracle = Oracle::new();
    // One compiled entry: empty, grow/shrink primary work, nonzero origin, and documented
    // partial-edge behavior. Input NaNs start immediately outside initialized norm tiles.
    for (start, count) in [
        (41984, 0),
        (0, 37888),
        (0, 39936),
        (0, 38912),
        (1024, 1024),
        (1025, 1),
    ] {
        let initialized = touched(start, count, 1024);
        let query = input(&mut device, 0, initialized.clone()).await;
        let key = input(&mut device, 1, initialized).await;
        let origin = scalar(&mut device, start).await;
        let length = scalar(&mut device, count).await;
        let mut outputs = (
            output(&mut device).await,
            output(&mut device).await,
            output(&mut device).await,
            output(&mut device).await,
        );
        launch(
            live_qk,
            (
                &mut device,
                &module,
                (&query, &key, &positions, &origin, &length),
                &mut outputs,
            ),
        )
        .await
        .unwrap();
        check(&mut device, outputs, start, count, &oracle).await;
    }
}
