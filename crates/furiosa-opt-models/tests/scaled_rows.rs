//! A launch's live rows in feature slices: scaled, each live row's features times an FP32 factor
//! rounded once to BF16, and stored a tile at a time, each tile through its own live rows.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Op;
use furiosa_opt_models::ops::index::scatter::Store;
use furiosa_opt_models::ops::normalization::Spread;
use furiosa_opt_models::ops::pointwise::multiply::Scale;
use furiosa_opt_models::tensor::axis::{Eight, Sixteen, SixtyFour};
use furiosa_opt_models::tensor::live::Live;
use furiosa_opt_models::tensor::{parallel::Chip, scalar::Value};
use furiosa_opt_std::{prelude::*, runtime::launch};
use tokio::sync::Mutex;

/// The device tests share one chip, so they run one at a time.
static DEVICE: Mutex<()> = Mutex::const_new(());

axes![Features = 3840];

type Rows = m![Sixteen, Features];

/// A launch of 64 rows of the features.
type Launch = m![SixtyFour, Features];

/// What an output holds before a launch: a row the launch does not write keeps it.
const KEPT: f32 = 30720.0;

/// Pseudo-random values in `[-1, 1)`, rounded to what the device reads.
fn random(count: usize, mut state: u64) -> Vec<bf16> {
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bf16::from_f32((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
        })
        .collect()
}

/// A step's count, to upload.
fn count(value: usize) -> HostTensor<i32, m![1]> {
    HostTensor::from_vec([i32::try_from(value).expect("counts fit I32")])
}

#[device(chip = 1)]
fn scaled_rows(
    device: &mut Device,
    scale: &Scale<HbmTensor<f32, Chip, m![1 # 8]>>,
    input: &HbmTensor<bf16, Chip, Rows>,
    (start, rows): (&Value, &Value),
    output: &mut HbmTensor<bf16, Chip, Rows>,
) {
    let live = Live::<(), Sixteen>::all()
        .subrange(start.extent::<m![Sixteen]>(), rows.extent::<m![Sixteen]>());
    let input: Live<Spread<Features, Sixteen>, Sixteen> = live.with(input.to_dm(&mut device.tdma));
    let scaled = scale.forward(&mut *device, &input);
    Store.forward(device, ((scaled.value(), scaled.with(())), output));
}

/// The live rows a step's count names, each tile of eight stored alone through the tile's own live
/// rows.
#[device(chip = 1)]
fn stored_tiles(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, Launch>,
    rows: &Value,
    output: &mut HbmTensor<bf16, Chip, Launch>,
) {
    let live = Live::<(), SixtyFour>::read(rows);
    let input: Spread<Features, SixtyFour> = input.to_dm(&mut device.tdma);
    for at in 0..live.count::<Eight>() {
        Store.forward(
            &mut *device,
            ((&input, live.tile::<Eight>(at)), &mut *output),
        );
    }
}

/// Each live row's features times the FP32 factor round once to BF16, wherever the live rows
/// start; the rows outside them keep what the output held.
#[tokio::test(flavor = "current_thread")]
async fn scales_live_rows_once_rounded() {
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(scaled_rows.topology()).unwrap();
    // Magnitudes up to 2^4 apart from row to row, so products cross binades.
    let inputs: Vec<f32> = random(Sixteen::SIZE * Features::SIZE, 0x5ca1_e0ff)
        .iter()
        .enumerate()
        .map(|(at, value)| value.to_f32() * (1 << (at / Features::SIZE % 5)) as f32)
        .collect();
    for factor in [62.0, core::f32::consts::FRAC_1_SQRT_2] {
        let scale = Scale(
            HostTensor::<f32, m![1 # 8]>::from_vec([factor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        );
        for (first, rows) in [(0, 11), (8, 2), (7, 2), (0, 16)] {
            let live = |at: usize| (first..first + rows).contains(&(at / Features::SIZE));
            let input = inputs
                .iter()
                .enumerate()
                .map(|(at, &value)| bf16::from_f32(if live(at) { value } else { f32::NAN }));
            let input = HostTensor::<bf16, Rows>::from_vec(input.collect::<Vec<_>>())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let (start, count) = (
                count(first).to_hbm(&mut device.pdma).await.unwrap(),
                count(rows).to_hbm(&mut device.pdma).await.unwrap(),
            );
            let mut output = HostTensor::<bf16, Rows>::from_vec(vec![
                bf16::from_f32(KEPT);
                Sixteen::SIZE * Features::SIZE
            ])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            launch(
                scaled_rows,
                (&mut device, &scale, &input, (&start, &count), &mut output),
            )
            .await
            .unwrap();
            let output = output
                .to_host::<Rows>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (at, (got, &value)) in output.iter().zip(&inputs).enumerate() {
                let want = match live(at) {
                    true => bf16::from_f32(bf16::from_f32(value).to_f32() * factor),
                    false => bf16::from_f32(KEPT),
                };
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{factor} rows {first}+{rows}: element {at}"
                );
            }
        }
    }
}

/// A store of each eight-row tile of a launch's live rows writes exactly those rows, whatever tile
/// the last of them ends in; every other row keeps what the output held.
#[tokio::test(flavor = "current_thread")]
async fn stores_each_tile_of_live_rows() {
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(stored_tiles.topology()).unwrap();
    let inputs = random(SixtyFour::SIZE * Features::SIZE, 0x7a11_0f5e);
    let input = HostTensor::<bf16, Launch>::from_vec(inputs.clone())
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    for rows in [1, 8, 9, 64] {
        let live = count(rows).to_hbm(&mut device.pdma).await.unwrap();
        let mut output = HostTensor::<bf16, Launch>::from_vec(vec![
            bf16::from_f32(KEPT);
            SixtyFour::SIZE * Features::SIZE
        ])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        launch(stored_tiles, (&mut device, &input, &live, &mut output))
            .await
            .unwrap();
        let output = output
            .to_host::<Launch>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, (got, value)) in output.iter().zip(&inputs).enumerate() {
            let want = match at / Features::SIZE < rows {
                true => *value,
                false => bf16::from_f32(KEPT),
            };
            assert_eq!(got.to_bits(), want.to_bits(), "{rows} rows: element {at}");
        }
    }
}
