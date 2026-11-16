//! A token-scaled FP8 projection's live rows at each width Gemma 4 projects a row to: every live
//! row quantizes to E4M3 under its own scale, its largest magnitude over the largest E4M3 value
//! rounded once to BF16, and each of its outputs holds within one BF16 rounding of that arithmetic
//! in FP64, wherever the launch's live rows start and end: inside one lane group, across several,
//! and over half the envelope or more, whose rows then contract in halves. A row of zeros projects
//! to zeros, and the rows outside the live ones, poison, reach no live output.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use core::marker::PhantomData;
use core::ops::Range;

use furiosa_opt_models::Op;
use furiosa_opt_models::models::decoder::Projection;
use furiosa_opt_models::models::decoder::attention::gemma4::{self, full, sliding};
use furiosa_opt_models::models::decoder::form::{TokenScaled, channel};
use furiosa_opt_models::ops::linear::Native;
use furiosa_opt_models::ops::normalization::Spread;
use furiosa_opt_models::tensor::axis::Single;
use furiosa_opt_models::tensor::form::BlockScaled;
use furiosa_opt_models::tensor::form::fp8::Token;
use furiosa_opt_models::tensor::live::Live;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_std::{prelude::*, runtime::launch};
use tokio::sync::Mutex;

/// The device tests share one chip, so they run one at a time.
static DEVICE: Mutex<()> = Mutex::const_new(());

type Rows = gemma4::Rows;
type Hidden = gemma4::Hidden;

/// The token-scaled projection to `O` outputs as the form holds it, spelled out so that a kernel's
/// `O` names its outputs, read as `Heads` heads of `Groups` heads of `Head` features each.
type Weight<O, Heads, Groups, Head> = Native<
    BlockScaled<
        Hidden,
        O,
        f8e4m3,
        f32,
        m![O, 1 # 8],
        Token,
        Chip,
        HbmTensor<f8e4m3, Chip, channel::Codes<Hidden, O>>,
        HbmTensor<f32, Chip, m![O, 1 # 8]>,
    >,
    m![Heads, Groups, Head],
>;

/// The form's projection is the one spelled out above.
const _: fn(
    Projection<TokenScaled, Hidden, sliding::QueryWidth>,
) -> Weight<sliding::QueryWidth, sliding::Heads, sliding::Groups, sliding::Head> =
    |weight| Native(weight, PhantomData);

/// The smallest row scale, as the form floors it: a row of zeros quantizes to zeros.
const FLOOR: f32 = 1e-12;

/// The row of zeros: in the second lane group, so that several of the cases hold it live.
const ZEROS: usize = 13;

/// The live rows these cases name: none, one, the first group and one row past it, two and three
/// whole groups, rows that start inside a group or cross from one into the next, half the
/// envelope, rows from inside the first group into the sixth, and the envelope.
const CASES: [Range<usize>; 10] = [
    0..0,
    0..1,
    0..9,
    0..16,
    0..24,
    8..10,
    5..11,
    0..32,
    3..41,
    0..64,
];

/// The live rows `start` and `rows` name, through `weight`, from HBM rows to heads and back.
#[device(chip = 1)]
fn projected<O: AxisName, Heads: AxisName, Groups: AxisName, Head: AxisName>(
    device: &mut Device,
    weight: &Weight<O, Heads, Groups, Head>,
    (input, (start, rows)): (&HbmTensor<bf16, Chip, m![Rows, Hidden]>, (&Value, &Value)),
) -> HbmTensor<bf16, Chip, m![Rows, Heads, Groups, Head]> {
    let live =
        Live::<(), Rows>::all().subrange(start.extent::<m![Rows]>(), rows.extent::<m![Rows]>());
    let rows: Live<Spread<Hidden, Rows, Chip>, Rows> = live.with(input.to_dm(&mut device.tdma));
    weight
        .forward(&mut *device, &rows)
        .value()
        .to_hbm(&mut device.tdma)
}

/// Pseudo-random 64-bit words from `state`.
fn words(count: usize, mut state: u64) -> impl Iterator<Item = u64> {
    (0..count).map(move |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    })
}

/// A token-scaled projection to `O` outputs on the device, with its codes and output scales on the
/// host: codes over E4M3's eighths in [-1.875, 1.875], each output's FP32 scale in [0.01, 0.058].
async fn weight<O: AxisName, Heads: AxisName, Groups: AxisName, Head: AxisName>(
    device: &mut Device,
    seed: u64,
) -> (Weight<O, Heads, Groups, Head>, Vec<f8e4m3>, Vec<f32>) {
    let codes: Vec<f8e4m3> = (words(O::SIZE * Hidden::SIZE, seed))
        .map(|word| f8e4m3::from_f32(((word >> 59) % 31) as f32 / 8.0 - 1.875))
        .collect();
    let scales: Vec<f32> = (0..O::SIZE)
        .map(|output| 0.01 + (output % 17) as f32 * 0.003)
        .collect();
    let pdma = &mut device.pdma;
    let weight = BlockScaled {
        weight: (HostTensor::<f8e4m3, m![O, Hidden]>::from_vec(codes.clone())
            .to_hbm(pdma)
            .await)
            .unwrap(),
        weight_scale: (HostTensor::<f32, m![O, 1 # 8]>::from_vec(
            (scales.iter()).flat_map(|&scale| core::iter::once(scale).chain([0.0; 7])),
        )
        .to_hbm(pdma)
        .await)
            .unwrap(),
        contract: Token,
        shape: PhantomData,
    };
    (Native(weight, PhantomData), codes, scales)
}

/// Every row of the envelope, each of one magnitude up to 2^4 apart so that rows quantize under
/// different scales, row `ZEROS` all zeros; the rows outside `live` poison.
fn rows(state: u64, live: Range<usize>) -> Vec<bf16> {
    (words(Rows::SIZE * Hidden::SIZE, state).enumerate())
        .map(|(at, word)| match at / Hidden::SIZE {
            row if !live.contains(&row) => bf16::from_bits(0x7fc1),
            ZEROS => bf16::from_f32(0.0),
            row => {
                let unit = (word >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
                bf16::from_f32(unit * (1 << (row % 5)) as f32)
            }
        })
        .collect()
}

/// A count as the device reads it.
fn count(value: usize) -> HostTensor<i32, m![1]> {
    HostTensor::from_vec([i32::try_from(value).expect("counts fit I32")])
}

/// One row's FP64 outputs under the token contract: its inputs quantized to E4M3 under its scale,
/// contracted against the codes, times that scale and each output's.
fn exact(row: &[bf16], codes: &[f8e4m3], scales: &[f32]) -> Vec<f64> {
    let largest = row.iter().map(|x| x.to_f32().abs()).fold(0.0f32, f32::max);
    let scale = bf16::from_f32((largest / 448.0).max(FLOOR)).to_f32();
    let quantized: Vec<f64> = (row.iter())
        .map(|x| f64::from(f8e4m3::from_f32((x.to_f32() / scale).clamp(-448.0, 448.0)).to_f32()))
        .collect();
    (codes.chunks_exact(Hidden::SIZE).zip(scales))
        .map(|(weights, &output)| {
            let sum: f64 = (weights.iter().zip(&quantized))
                .map(|(w, x)| f64::from(w.to_f32()) * x)
                .sum();
            sum * f64::from(scale) * f64::from(output)
        })
        .collect()
}

/// Every live row of every case through the projection to `O` outputs, each output within one
/// BF16 rounding of its FP64 value plus its row's RMS over 512, what reordering the sums leaves.
async fn projects<O: AxisName, Heads: AxisName, Groups: AxisName, Head: AxisName>(
    device: &mut Device,
    seed: u64,
) {
    let (weight, codes, scales) = weight::<O, Heads, Groups, Head>(device, seed).await;
    let exact: Vec<Vec<f64>> = (rows(seed + 1, 0..Rows::SIZE).chunks_exact(Hidden::SIZE))
        .map(|row| exact(row, &codes, &scales))
        .collect();
    for live in CASES {
        let input = (HostTensor::<bf16, m![Rows, Hidden]>::from_vec(rows(seed + 1, live.clone())))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let start = count(live.start).to_hbm(&mut device.pdma).await.unwrap();
        let counted = count(live.len()).to_hbm(&mut device.pdma).await.unwrap();
        let found = launch(
            projected,
            (&mut *device, &weight, (&input, (&start, &counted))),
        )
        .await
        .unwrap()
        .to_host::<m![Rows, Heads, Groups, Head]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
        for row in live.clone() {
            let exact = &exact[row];
            let rms = (exact.iter().map(|e| e * e).sum::<f64>() / O::SIZE as f64).sqrt();
            for (at, &want) in exact.iter().enumerate() {
                let got = f64::from(found[row * O::SIZE + at].to_f32());
                assert!(
                    (got - want).abs() <= want.abs() / 256.0 + rms / 512.0,
                    "{} outputs, rows {live:?}: row {row} output {at}: {got} for {want}",
                    O::SIZE,
                );
            }
        }
    }
}

/// Each live row of every case within its rounding at Gemma 4's widths: a sliding layer's queries
/// and its keys or values, and a full layer's queries and keys.
#[tokio::test(flavor = "current_thread")]
async fn projects_each_live_row_within_its_rounding() {
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    projects::<sliding::QueryWidth, sliding::Heads, sliding::Groups, sliding::Head>(&mut device, 1)
        .await;
    projects::<sliding::KvWidth, sliding::Heads, Single, sliding::Head>(&mut device, 3).await;
    projects::<full::QueryWidth, full::Heads, full::Groups, full::Head>(&mut device, 5).await;
    projects::<full::KvWidth, full::Heads, Single, full::Head>(&mut device, 7).await;
}
