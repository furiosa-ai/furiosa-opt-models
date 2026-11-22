//! Checked prefix composition of grouped gate/up, HBM SwiGLU and grouped down projections.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::load::{Checkpoint, Context, Load, Named};
use furiosa_opt_models::modules::generic::Prefix;
use furiosa_opt_models::ops::attention::geometry::Hidden;
use furiosa_opt_models::ops::linear::{Grouped, grouped::Layout};
use furiosa_opt_models::ops::pointwise::SwiGlu;
use furiosa_opt_models::ops::swiglu::{GateUp, Intermediate};
use furiosa_opt_models::tensor::{axis::Rows, live::BlockPrefix, scalar::Value};
use furiosa_opt_models::{Op, Then};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;

type R = Rows<10496>;
type B = Rows<256>;
type H = Hidden<128>;
type I = Intermediate<2048>;
type Input<N = R> = HbmTensor<bf16, m![1], m![{ Symbol<N> }, H]>;
type Packed<N = R> = HbmTensor<bf16, m![1], m![{ Symbol<N> }, { Symbol<GateUp<I>> }]>;
type Product<N = R> = HbmTensor<bf16, m![1], m![{ Symbol<N> }, I]>;
type Scratch<N = R> = (Packed<N>, Product<N>, Input<N>);

#[derive(DeviceSend, Load)]
struct Weights<U, D> {
    fc1: U,
    fc2: D,
}

type Owned = Weights<Grouped<H, GateUp<I>>, Grouped<I, H>>;
type Borrowed<'a> = Weights<
    Grouped<H, GateUp<I>, m![1], &'a HbmTensor<bf16, m![1], Layout<H, GateUp<I>>>>,
    Grouped<I, H, m![1], &'a HbmTensor<bf16, m![1], Layout<I, H>>>,
>;

const PERIOD: usize = 29;
const POISON: u16 = 0x7fc1;
const SENTINELS: [f32; 3] = [13.0, 17.0, 19.0];

#[device(chip = 1)]
fn into(
    device: &mut Device,
    weights: &Borrowed<'_>,
    input: &Input,
    count: &Value<m![1]>,
    scratch: &mut Scratch,
) {
    let prefix = BlockPrefix::<(), R, B>::read(count);
    weights
        .fc1
        .forward(&mut *device, (&prefix.with(input), &mut scratch.0));
    SwiGlu.forward(&mut *device, (&prefix.with(&scratch.0), &mut scratch.1));
    weights
        .fc2
        .forward(device, (&prefix.with(&scratch.1), &mut scratch.2));
}

#[device(chip = 1)]
fn owned(device: &mut Device, weights: &Owned, input: &Input, count: &Value<m![1]>) -> Input {
    let prefix = BlockPrefix::<(), R, B>::read(count);
    let projected = weights.fc1.forward(&mut *device, &prefix.with(input));
    let product = SwiGlu.forward(&mut *device, &prefix.with(projected.value()));
    weights
        .fc2
        .forward(device, &prefix.with(product.value()))
        .into_value()
}

/// The ordinary Mlp Module receives the same checked request interval through every child.
/// Ownership is passed into Live once; Then lends each intermediate to the next Prefix.
#[device(chip = 1)]
fn module_ffn(
    device: &mut Device,
    weights: &Borrowed<'_>,
    input: Input,
    count: &Value<m![1]>,
) -> Input {
    let prefix = BlockPrefix::<(), R, B>::read(count);
    let model = Then(
        Prefix(Named::<"fc1", _>(&weights.fc1)),
        Then(Prefix(SwiGlu), Prefix(Named::<"fc2", _>(&weights.fc2))),
    );
    let mut state = ((), ((), ()));
    furiosa_opt_models::Module::forward(
        &model,
        device,
        (&prefix, ()),
        &mut state,
        &prefix.interval().with(input),
    )
    .into_value()
}

#[device(chip = 1)]
fn fixed<N: AxisName>(
    device: &mut Device,
    weights: &Owned,
    input: &Input<N>,
    scratch: &mut Scratch<N>,
) {
    weights.fc1.forward(&mut *device, (input, &mut scratch.0));
    SwiGlu.forward(&mut *device, (&scratch.0, &mut scratch.1));
    weights.fc2.forward(device, (&scratch.1, &mut scratch.2));
}

#[device(chip = 1)]
fn legacy_128(device: &mut Device, input: &Packed<Rows<128>>) -> Product<Rows<128>> {
    SwiGlu.forward(device, input)
}

fn value(row: usize, column: usize) -> bf16 {
    if row.is_multiple_of(PERIOD) {
        bf16::from_f32(0.0)
    } else {
        bf16::from_f32(((row * 11 + column * 5) % PERIOD) as f32 / 4.0 - 14.0 / 4.0)
    }
}

fn up(column: usize, k: usize) -> bf16 {
    let feature = column % I::SIZE;
    let channel = if column < I::SIZE {
        feature % H::SIZE
    } else {
        (feature * 7 + 1) % H::SIZE
    };
    bf16::from_f32(if k == channel { 1.0 } else { 0.0 })
}

fn down(column: usize, k: usize) -> bf16 {
    bf16::from_f32(((column * 11 + k * 7) % 31) as f32 / 64.0 - 15.0 / 64.0)
}

fn rounded(x: f64) -> bf16 {
    bf16::from_f32(x as f32)
}

struct Reference {
    packed: Vec<bf16>,
    product: Vec<bf16>,
    output: Vec<bf16>,
}

impl Reference {
    fn new() -> Self {
        let mut result = Self {
            packed: Vec::new(),
            product: Vec::new(),
            output: Vec::new(),
        };
        let mut differing_rounds = 0;
        for row in 0..PERIOD {
            let projected: Vec<_> = (0..2 * I::SIZE)
                .map(|column| {
                    rounded(
                        (0..H::SIZE)
                            .map(|k| {
                                f64::from(value(row, k).to_f32())
                                    * f64::from(up(column, k).to_f32())
                            })
                            .sum(),
                    )
                })
                .collect();
            let products: Vec<_> = (0..I::SIZE)
                .map(|k| {
                    let gate = f64::from(projected[k].to_f32());
                    let up = f64::from(projected[I::SIZE + k].to_f32());
                    let silu = gate / (1.0 + (-gate).exp());
                    let twice = rounded(f64::from(rounded(silu).to_f32()) * up);
                    differing_rounds +=
                        usize::from(twice.to_bits() != rounded(silu * up).to_bits());
                    twice
                })
                .collect();
            for column in 0..H::SIZE {
                result.output.push(rounded(
                    (0..I::SIZE)
                        .map(|k| {
                            f64::from(products[k].to_f32()) * f64::from(down(column, k).to_f32())
                        })
                        .sum(),
                ));
            }
            result.packed.extend(projected);
            result.product.extend(products);
        }
        assert!(
            differing_rounds > 100,
            "fixture must distinguish the two rounding boundaries"
        );
        result
    }
}

fn checkpoint() -> (tempfile::TempDir, Checkpoint) {
    let directory = tempfile::tempdir().unwrap();
    let up: Vec<_> = (0..2 * I::SIZE * H::SIZE)
        .map(|at| up(at / H::SIZE, at % H::SIZE))
        .collect();
    let down: Vec<_> = (0..H::SIZE * I::SIZE)
        .map(|at| down(at / I::SIZE, at % I::SIZE))
        .collect();
    let up = bf16::to_buf(&up);
    let down = bf16::to_buf(&down);
    let views = [
        (
            "fc1.weight",
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::BF16,
                vec![2 * I::SIZE, H::SIZE],
                &up,
            )
            .unwrap(),
        ),
        (
            "fc2.weight",
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::BF16,
                vec![H::SIZE, I::SIZE],
                &down,
            )
            .unwrap(),
        ),
    ];
    safetensors::serialize_to_file(views, &None, &directory.path().join("weights.safetensors"))
        .unwrap();
    let source = Checkpoint::new(directory.path(), Some(serde_json::json!({}))).unwrap();
    (directory, source)
}

async fn scratch<N: AxisName>(device: &mut Device) -> Scratch<N> {
    (
        HostTensor::<bf16, m![{ Symbol<N> }, { Symbol<GateUp<I>> }]>::from_vec(
            std::iter::repeat_n(bf16::from_f32(SENTINELS[0]), N::SIZE * 2 * I::SIZE),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap(),
        HostTensor::<bf16, m![{ Symbol<N> },I]>::from_vec(std::iter::repeat_n(
            bf16::from_f32(SENTINELS[1]),
            N::SIZE * I::SIZE,
        ))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap(),
        HostTensor::<bf16, m![{ Symbol<N> },H]>::from_vec(std::iter::repeat_n(
            bf16::from_f32(SENTINELS[2]),
            N::SIZE * H::SIZE,
        ))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap(),
    )
}

fn check(values: &[bf16], rows: usize, width: usize, oracle: &[bf16], sentinel: f32, exact: bool) {
    assert_eq!(values.len(), R::SIZE * width);
    for row in 0..R::SIZE {
        for column in 0..width {
            let got = values[row * width + column];
            if row >= rows {
                assert_eq!(
                    got.to_bits(),
                    bf16::from_f32(sentinel).to_bits(),
                    "tail row {row} column {column}"
                );
            } else {
                let want = oracle[(row % PERIOD) * width + column];
                if exact || row.is_multiple_of(PERIOD) {
                    assert_eq!(got.to_bits(), want.to_bits(), "row {row} column {column}");
                } else {
                    let (got, want) = (f64::from(got.to_f32()), f64::from(want.to_f32()));
                    assert!(
                        got.is_finite() && (got - want).abs() <= want.abs() / 128.0 + 1e-3,
                        "row {row} column {column}: {got} vs {want}"
                    );
                }
            }
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn checked_ffn_prefixes_preserve_rounding_values_zero_rows_and_tails() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let (_directory, source) = checkpoint();
    let weights: Owned = source.load(&mut device).unwrap();
    let borrowed = Borrowed::load(&mut Context::new(None), &weights).unwrap();
    assert!(std::ptr::eq(weights.fc1.tensor(), borrowed.fc1.tensor()));
    assert!(std::ptr::eq(weights.fc2.tensor(), borrowed.fc2.tensor()));
    let reference = Reference::new();
    for blocks in [0, 1, 2, 3, 37, 38, 39, 41, 1, 0] {
        let rows = blocks * B::SIZE;
        let prefix = BlockPrefix::<(), R, B>::try_over(rows).unwrap();
        let count = HostTensor::<i32, m![1]>::from_vec([prefix.blocks() as i32])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let input: Input =
            HostTensor::<bf16, m![R, H]>::from_vec((0..R::SIZE * H::SIZE).map(|at| {
                if at / H::SIZE < rows {
                    value(at / H::SIZE, at % H::SIZE)
                } else {
                    bf16::from_bits(POISON)
                }
            }))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut active = scratch::<R>(&mut device).await;
        launch(into, (&mut device, &borrowed, &input, &count, &mut active))
            .await
            .unwrap();
        let packed = active
            .0
            .to_host::<m![R,{ Symbol<GateUp<I>> }]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let product = active
            .1
            .to_host::<m![R, I]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let output = active
            .2
            .to_host::<m![R, H]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        check(
            &packed,
            rows,
            2 * I::SIZE,
            &reference.packed,
            SENTINELS[0],
            true,
        );
        check(
            &product,
            rows,
            I::SIZE,
            &reference.product,
            SENTINELS[1],
            false,
        );
        // The final accumulation can amplify near-tie SiLU rounding, so use the existing
        // public MLP's 1% relative-L2 criterion, retaining exact zeros and strict tail bits.
        let (mut error, mut norm) = (0.0f64, 0.0f64);
        for row in 0..R::SIZE {
            for column in 0..H::SIZE {
                let got = output[row * H::SIZE + column];
                if row >= rows {
                    assert_eq!(got.to_bits(), bf16::from_f32(SENTINELS[2]).to_bits());
                } else {
                    let want = reference.output[(row % PERIOD) * H::SIZE + column];
                    assert!(got.to_f32().is_finite());
                    if row.is_multiple_of(PERIOD) {
                        assert_eq!(got.to_bits(), 0);
                    }
                    error += (f64::from(got.to_f32()) - f64::from(want.to_f32())).powi(2);
                    norm += f64::from(want.to_f32()).powi(2);
                }
            }
        }
        if norm > 0.0 {
            assert!(
                (error / norm).sqrt() < 1e-2,
                "work {rows}, relative L2 {}",
                (error / norm).sqrt()
            );
        }
        let mut baseline = scratch::<R>(&mut device).await;
        launch(fixed, (&mut device, &weights, &input, &mut baseline))
            .await
            .unwrap();
        let static_packed = baseline
            .0
            .to_host::<m![R,{Symbol<GateUp<I>>}]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let static_product = baseline
            .1
            .to_host::<m![R, I]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let static_output = baseline
            .2
            .to_host::<m![R, H]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (got, want, width) in [
            (&packed, &static_packed, 2 * I::SIZE),
            (&product, &static_product, I::SIZE),
            (&output, &static_output, H::SIZE),
        ] {
            assert!(
                got[..rows * width]
                    .iter()
                    .zip(&want[..rows * width])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "static parity, work {rows}, width {width}"
            );
        }
        let owned_result = launch(owned, (&mut device, &weights, &input, &count))
            .await
            .unwrap();
        if rows > 0 {
            let bytes = owned_result
                .into_buffer()
                .slice(.., ..rows * H::SIZE * 2)
                .unwrap()
                .to_host(&mut device.pdma)
                .await
                .unwrap();
            for (at, b) in bytes.chunks_exact(2).enumerate() {
                assert_eq!(
                    u16::from_le_bytes([b[0], b[1]]),
                    output[at].to_bits(),
                    "owned parity, work {rows}, element {at}"
                );
            }
        }
        // The generic Module composition must match the checked leaf composition bit for bit.
        // Consume the already uploaded input only after all borrowed leaf comparisons finish.
        let module_result = launch(module_ffn, (&mut device, &borrowed, input, &count))
            .await
            .unwrap();
        if rows > 0 {
            let bytes = module_result
                .into_buffer()
                .slice(.., ..rows * H::SIZE * 2)
                .unwrap()
                .to_host(&mut device.pdma)
                .await
                .unwrap();
            for (at, b) in bytes.chunks_exact(2).enumerate() {
                assert_eq!(
                    u16::from_le_bytes([b[0], b[1]]),
                    output[at].to_bits(),
                    "Module parity, work {rows}, element {at}"
                );
            }
        }
    }
    let packed: Packed<Rows<128>> = HostTensor::<
        bf16,
        m![{ Symbol<Rows<128>> }, { Symbol<GateUp<I>> }],
    >::from_vec((0..128 * 2 * I::SIZE).map(|at| {
        reference.packed[((at / (2 * I::SIZE)) % PERIOD) * 2 * I::SIZE + at % (2 * I::SIZE)]
    }))
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let legacy = launch(legacy_128, (&mut device, &packed))
        .await
        .unwrap()
        .to_host::<m![{ Symbol<Rows<128>> }, I]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    for (at, got) in legacy.iter().enumerate() {
        let want = reference.product[((at / I::SIZE) % PERIOD) * I::SIZE + at % I::SIZE];
        let (got, want) = (f64::from(got.to_f32()), f64::from(want.to_f32()));
        assert!(
            got.is_finite() && (got - want).abs() <= want.abs() / 128.0 + 1e-3,
            "legacy 128 element {at}"
        );
    }
}
