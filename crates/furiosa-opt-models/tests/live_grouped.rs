//! Checked runtime prefixes across strict loading, borrowed storage and public grouped Ops.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Op;
use furiosa_opt_models::load::{Checkpoint, Context, Load};
use furiosa_opt_models::ops::attention::geometry::{Hidden, ProjectionWidth};
use furiosa_opt_models::ops::linear::{Grouped, grouped::Layout};
use furiosa_opt_models::tensor::axis::Rows;
use furiosa_opt_models::tensor::form::Weight;
use furiosa_opt_models::tensor::live::{BlockPrefix, Live};
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;

type R = Rows<2048>;
type B = Rows<256>;
type In = Hidden<128>;
type Out = ProjectionWidth<2048>;
type Input = HbmTensor<bf16, m![1], m![R, In]>;
type Output = HbmTensor<bf16, m![1], m![R, Out]>;
type Owned = Grouped<In, Out>;
type Borrowed<'a> = Grouped<In, Out, m![1], &'a HbmTensor<bf16, m![1], Layout<In, Out>>>;

type FullIn = Hidden<5376>;
type FullOut = ProjectionWidth<28672>;
type FullRows = Rows<10496>;
type FullInput = HbmTensor<bf16, m![1], m![FullRows, FullIn]>;
type FullOutput = HbmTensor<bf16, m![1], m![FullRows, FullOut]>;
type FullProjection = Grouped<FullIn, FullOut>;

const SENTINEL: f32 = 13.0;
const POISON: u16 = 0x7fc1;
const INPUT_PERIOD: usize = 29;
const OUTPUT_PERIOD: usize = 31;

#[device(chip = 1)]
fn into(
    device: &mut Device,
    module: &Borrowed<'_>,
    input: &Input,
    blocks: &Value<m![1]>,
    output: &mut Output,
) {
    let prefix = BlockPrefix::<(), R, B>::read(blocks).with(input);
    let _: Live<(), R> = module.forward(device, (&prefix, output));
}

#[device(chip = 1)]
fn owned(device: &mut Device, module: &Owned, input: &Input, blocks: &Value<m![1]>) -> Output {
    let prefix = BlockPrefix::<(), R, B>::read(blocks).with(input);
    let result: Live<Output, R> = module.forward(device, &prefix);
    result.into_value()
}

#[device(chip = 1)]
fn full_prefix(
    device: &mut Device,
    module: &FullProjection,
    input: &FullInput,
    blocks: &Value<m![1]>,
    output: &mut FullOutput,
) {
    let prefix = BlockPrefix::<(), FullRows, B>::read(blocks).with(input);
    module.forward(device, (&prefix, output));
}

#[device(chip = 1)]
fn full_fixed(device: &mut Device, module: &FullProjection, input: &FullInput) -> FullOutput {
    module.forward(device, input)
}

fn coefficient(column: usize, k: usize) -> bf16 {
    bf16::from_f32(((column * 17 + k * 3) % OUTPUT_PERIOD) as f32 / 16.0 - 15.0 / 16.0)
}

fn input(row: usize, k: usize) -> bf16 {
    bf16::from_f32(((row * 11 + k * 5) % INPUT_PERIOD) as f32 / 16.0 - 14.0 / 16.0)
}

/// Compute independent FP64 dot products for every distinct row/column class. Periods coprime
/// to the tile and pass widths keep permutation mistakes observable without repeating the same
/// FP64 dot millions of times. Dyadic products and these small sums are exactly representable
/// in FP32, so the final BF16 bits are the complete numerical oracle.
fn reference() -> Vec<bf16> {
    (0..INPUT_PERIOD * OUTPUT_PERIOD)
        .map(|index| {
            let (row, column) = (index / OUTPUT_PERIOD, index % OUTPUT_PERIOD);
            let dot = (0..In::SIZE)
                .map(|k| {
                    f64::from(input(row, k).to_f32()) * f64::from(coefficient(column, k).to_f32())
                })
                .sum::<f64>();
            bf16::from_f32(dot as f32)
        })
        .collect()
}

fn check(values: &[bf16], rows: usize, tail: bool, reference: &[bf16]) {
    let allocated = if tail { R::SIZE } else { rows };
    assert_eq!(values.len(), allocated * Out::SIZE);
    for row in 0..allocated {
        for column in 0..Out::SIZE {
            let expected = if row < rows {
                reference[(row % INPUT_PERIOD) * OUTPUT_PERIOD + column % OUTPUT_PERIOD]
            } else {
                bf16::from_f32(SENTINEL)
            };
            assert_eq!(
                values[row * Out::SIZE + column].to_bits(),
                expected.to_bits(),
                "row {row}, column {column}, work {rows}"
            );
        }
    }
}

fn checkpoint() -> (tempfile::TempDir, Checkpoint) {
    let directory = tempfile::tempdir().unwrap();
    let values: Vec<_> = (0..Out::SIZE * In::SIZE)
        .map(|at| coefficient(at / In::SIZE, at % In::SIZE))
        .collect();
    let bytes = bf16::to_buf(&values);
    let view = safetensors::tensor::TensorView::new(
        safetensors::Dtype::BF16,
        vec![Out::SIZE, In::SIZE],
        &bytes,
    )
    .unwrap();
    safetensors::serialize_to_file(
        [("weight", view)],
        &None,
        &directory.path().join("weights.safetensors"),
    )
    .unwrap();
    let source = Checkpoint::new(directory.path(), Some(serde_json::json!({}))).unwrap();
    (directory, source)
}

/// Refuse invalid work before preparing transport or launching a projection.
#[test]
fn grouped_work_refuses_unaligned_or_oversized_prefixes() {
    for rows in [1, 255, 257, R::SIZE + B::SIZE] {
        assert!(
            BlockPrefix::<(), R, B>::try_over(rows).is_none(),
            "rows {rows}"
        );
    }
    for rows in [0, B::SIZE, R::SIZE] {
        let prefix = BlockPrefix::<(), R, B>::try_over(rows).unwrap();
        assert_eq!(prefix.blocks() * B::SIZE, rows);
    }
}

/// Extend the repro's full-width grouped/static equality check to the runtime prefix that
/// failed compilation at 5376 -> 28672. Empty, odd and even work also preserve destination tails.
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires an RNGD device"]
async fn full_width_prefix_matches_the_static_grouped_projection() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let canonical = Tensor::<bf16, m![FullOut, FullIn]>::from_vec(
        (0..FullOut::SIZE * FullIn::SIZE)
            .map(|at| coefficient(at / FullIn::SIZE, at % FullIn::SIZE)),
    );
    let packed = canonical
        .transpose::<Layout<FullIn, FullOut>>(false)
        .into_vec();
    let model = FullProjection::new(Weight::new(
        HostTensor::<bf16, Layout<FullIn, FullOut>>::from_vec(packed)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
    ));
    for rows in [0, 256, 512, 9472, 9728, 9984] {
        let source: FullInput = HostTensor::<bf16, m![FullRows, FullIn]>::from_vec(
            (0..FullRows::SIZE * FullIn::SIZE).map(|at| {
                if at / FullIn::SIZE < rows {
                    input(at / FullIn::SIZE, at % FullIn::SIZE)
                } else {
                    bf16::from_bits(POISON)
                }
            }),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let count: Value<m![1]> = HostTensor::<i32, m![1]>::from_vec([rows as i32 / 256])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut destination: FullOutput = HostTensor::<bf16, m![FullRows, FullOut]>::from_vec(
            std::iter::repeat_n(bf16::from_f32(SENTINEL), FullRows::SIZE * FullOut::SIZE),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        launch(
            full_prefix,
            (&mut device, &model, &source, &count, &mut destination),
        )
        .await
        .unwrap();
        let actual = destination
            .to_host::<m![FullRows, FullOut]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        if rows > 0 {
            let expected = launch(full_fixed, (&mut device, &model, &source))
                .await
                .unwrap()
                .to_host::<m![FullRows, FullOut]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for at in 0..rows * FullOut::SIZE {
                assert_eq!(
                    actual[at].to_bits(),
                    expected[at].to_bits(),
                    "work {rows}, element {at}"
                );
            }
        }
        assert!(
            actual[rows * FullOut::SIZE..]
                .iter()
                .all(|value| value.to_bits() == bf16::from_f32(SENTINEL).to_bits()),
            "work {rows}: allocation tail changed"
        );
    }
}

/// One compiled entry handles empty, single, paired, odd/even and full work. Source NaNs past
/// each prefix expose padded computation, and reused destination sentinels detect stray stores.
/// The empty and short cases also exercise prefetch bounds. The borrowed
/// load shares the exact owned weight; both public output forms match independent FP64 dots.
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires an explicitly selected RNGD chip"]
async fn loaded_grouped_prefixes_match_fp64_and_preserve_allocation_tails() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let (_directory, source) = checkpoint();
    let module: Owned = source.load(&mut device).unwrap();
    let borrowed = Borrowed::load(&mut Context::new(None), &module).unwrap();
    assert!(std::ptr::eq(module.tensor(), borrowed.tensor()));
    let reference = reference();
    let sentinel = || {
        HostTensor::<bf16, m![R, Out]>::from_vec(std::iter::repeat_n(
            bf16::from_f32(SENTINEL),
            R::SIZE * Out::SIZE,
        ))
    };
    let mut destination: Output = sentinel().to_hbm(&mut device.pdma).await.unwrap();
    for blocks in [0, 1, 2, 3, 4, 7, 8, 1, 0] {
        let rows = blocks * B::SIZE;
        let prefix = BlockPrefix::<(), R, B>::try_over(rows).unwrap();
        let count = HostTensor::<i32, m![1]>::from_vec([prefix.blocks() as i32])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let values: Input =
            HostTensor::<bf16, m![R, In]>::from_vec((0..R::SIZE * In::SIZE).map(|at| {
                if at / In::SIZE < rows {
                    input(at / In::SIZE, at % In::SIZE)
                } else {
                    bf16::from_bits(POISON)
                }
            }))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        sentinel()
            .to_hbm(&mut device.pdma)
            .output(&mut destination)
            .await
            .unwrap();
        launch(
            into,
            (&mut device, &borrowed, &values, &count, &mut destination),
        )
        .await
        .unwrap();
        let got = destination
            .to_host::<m![R, Out]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        check(&got, rows, true, &reference);
        let result = launch(owned, (&mut device, &module, &values, &count))
            .await
            .unwrap();
        if rows > 0 {
            let bytes = result
                .into_buffer()
                .slice(.., ..rows * Out::SIZE * 2)
                .unwrap()
                .to_host(&mut device.pdma)
                .await
                .unwrap();
            let got: Vec<_> = bytes
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            check(&got, rows, false, &reference);
        }
    }
}
