//! Strict checkpoint loading and public compositions around K-grouped projections.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::load::{Checkpoint, Load};
use furiosa_opt_models::modules::parallel::Exchanged;
use furiosa_opt_models::ops::attention::geometry::{Hidden, ProjectionWidth};
use furiosa_opt_models::ops::collective::Pieces;
use furiosa_opt_models::ops::linear::Linear;
use furiosa_opt_models::ops::pointwise::SwiGlu;
use furiosa_opt_models::ops::swiglu::{GateUp, Intermediate};
use furiosa_opt_models::tensor::axis::Rows;
use furiosa_opt_models::tensor::form::grouped::Grouped;
use furiosa_opt_models::tensor::parallel::{Parallel, Part, Shard};
use furiosa_opt_models::{Module, Then};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;

type Chips = m![{ Symbol<Shard<4>> }];
type H = Hidden<128>;
type O = ProjectionWidth<2048>;
type I = Intermediate<2048>;
type R = Rows<768>;
type Input = HbmTensor<bf16, Chips, m![R, H]>;
type Split = HbmTensor<bf16, Chips, m![{ Symbol<Pieces<R, 4>> }, { Symbol<Part<H, 4>> }]>;
type Output = HbmTensor<bf16, Chips, m![R, O]>;

#[derive(DeviceSend, Load)]
struct ExchangeWeights<P> {
    output: Exchanged<P, Parallel<4>>,
}

type PackedExchange = ExchangeWeights<Linear<H, O, Chips, Grouped<H, O, Chips>>>;
type OrdinaryExchange = ExchangeWeights<Linear<H, O, Chips>>;

#[derive(DeviceSend, Load)]
struct Mlp<U, D> {
    fc1: U,
    fc2: D,
}

type PackedMlp = Mlp<
    Linear<H, GateUp<I>, Chips, Grouped<H, GateUp<I>, Chips>>,
    Linear<I, H, Chips, Grouped<I, H, Chips>>,
>;
type OrdinaryMlp = Mlp<Linear<H, GateUp<I>, Chips>, Linear<I, H, Chips>>;

#[device(chip = 4)]
fn exchange(device: &mut Device, weights: &PackedExchange, input: &Split) -> Output {
    weights.output.forward(device, (), &mut (), input)
}

#[device(chip = 4)]
fn ordinary_exchange(device: &mut Device, weights: &OrdinaryExchange, input: &Split) -> Output {
    weights.output.forward(device, (), &mut (), input)
}

#[device(chip = 4)]
fn mlp(device: &mut Device, weights: &PackedMlp, input: &Input) -> Input {
    Then(&weights.fc1, Then(SwiGlu, &weights.fc2)).forward(device, (), &mut ((), ((), ())), input)
}

#[device(chip = 4)]
fn ordinary_mlp(device: &mut Device, weights: &OrdinaryMlp, input: &Input) -> Input {
    Then(&weights.fc1, Then(SwiGlu, &weights.fc2)).forward(device, (), &mut ((), ((), ())), input)
}

fn checkpoint(tensors: Vec<(&str, Vec<usize>, Vec<bf16>)>) -> (tempfile::TempDir, Checkpoint) {
    let directory = tempfile::tempdir().unwrap();
    let bytes: Vec<_> = tensors
        .iter()
        .map(|(_, _, data)| bf16::to_buf(data))
        .collect();
    let views = tensors.iter().zip(&bytes).map(|((name, shape, _), bytes)| {
        (
            *name,
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, shape.clone(), bytes)
                .unwrap(),
        )
    });
    safetensors::serialize_to_file(views, &None, &directory.path().join("weights.safetensors"))
        .unwrap();
    let source = Checkpoint::new(directory.path(), Some(serde_json::json!({}))).unwrap();
    (directory, source)
}

fn coefficient(index: usize) -> bf16 {
    bf16::from_f32(((index * 17 % 63) as f32 - 31.0) / 32.0)
}

fn input(index: usize) -> bf16 {
    bf16::from_f32(((index * 23 % 61) as f32 - 30.0) / 32.0)
}

/// Replicated checkpoint coefficients read all four source chips after the whole TDMA exchange.
#[tokio::test(flavor = "current_thread")]
async fn exchanged_grouped_projection_preserves_rows_and_columns() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let weights: Vec<_> = (0..O::SIZE * H::SIZE).map(coefficient).collect();
    let (_directory, source) = checkpoint(vec![(
        "output.weight",
        vec![O::SIZE, H::SIZE],
        weights.clone(),
    )]);
    let packed: PackedExchange = source.load(&mut device).unwrap();
    let ordinary: OrdinaryExchange = source.load(&mut device).unwrap();
    let values: Vec<_> = (0..4 * R::SIZE * H::SIZE).map(input).collect();
    let split: Split = HostTensor::<
        bf16,
        m![{ Chips }, { Symbol<Pieces<R, 4>> }, { Symbol<Part<H, 4>> }],
    >::from_vec(values.clone())
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let got = launch(exchange, (&mut device, &packed, &split))
        .await
        .unwrap()
        .to_host::<m![{ Chips }, R, O]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    let plain = launch(ordinary_exchange, (&mut device, &ordinary, &split))
        .await
        .unwrap()
        .to_host::<m![{ Chips }, R, O]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    for (index, (got, plain)) in got.iter().zip(plain).enumerate() {
        assert_eq!(got.to_bits(), plain.to_bits(), "element {index}");
    }
    for chip in 0..4 {
        for row in [0, 127, 128, 255, 256, 511, 512, R::SIZE - 1] {
            for column in 0..O::SIZE {
                let expected: f64 = (0..H::SIZE)
                    .map(|k| {
                        let part = H::SIZE / 4;
                        let index = ((k / part * 4 + chip) * R::SIZE + row) * part + k % part;
                        f64::from(values[index].to_f32())
                            * f64::from(weights[column * H::SIZE + k].to_f32())
                    })
                    .sum();
                assert_eq!(
                    got[(chip * R::SIZE + row) * O::SIZE + column].to_bits(),
                    bf16::from_f32(expected as f32).to_bits(),
                    "chip {chip}, row {row}, column {column}"
                );
            }
        }
    }
}

/// Loaded packed gate/up and down projections preserve the composition and its BF16 boundaries.
#[tokio::test(flavor = "current_thread")]
async fn grouped_mlp_matches_an_independent_reference() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let gate_up: Vec<_> = (0..2 * I::SIZE * H::SIZE).map(coefficient).collect();
    let down: Vec<_> = (0..H::SIZE * I::SIZE).map(coefficient).collect();
    let (_directory, source) = checkpoint(vec![
        ("fc1.weight", vec![2 * I::SIZE, H::SIZE], gate_up.clone()),
        ("fc2.weight", vec![H::SIZE, I::SIZE], down.clone()),
    ]);
    let packed: PackedMlp = source.load(&mut device).unwrap();
    let ordinary: OrdinaryMlp = source.load(&mut device).unwrap();
    let values: Vec<_> = (0..4 * R::SIZE * H::SIZE).map(input).collect();
    let tensor: Input = HostTensor::<bf16, m![{ Chips }, R, H]>::from_vec(values.clone())
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let got = launch(mlp, (&mut device, &packed, &tensor))
        .await
        .unwrap()
        .to_host::<m![{ Chips }, R, H]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    let plain = launch(ordinary_mlp, (&mut device, &ordinary, &tensor))
        .await
        .unwrap()
        .to_host::<m![{ Chips }, R, H]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    for (index, (got, plain)) in got.iter().zip(plain).enumerate() {
        assert_eq!(got.to_bits(), plain.to_bits(), "element {index}");
    }
    let (mut error, mut norm) = (0.0f64, 0.0f64);
    for chip in 0..4 {
        for row in [0, 127, 128, 255, 256, 511, 512, R::SIZE - 1] {
            let projected = |column: usize| {
                let dot: f64 = (0..H::SIZE)
                    .map(|k| {
                        f64::from(values[(chip * R::SIZE + row) * H::SIZE + k].to_f32())
                            * f64::from(gate_up[column * H::SIZE + k].to_f32())
                    })
                    .sum();
                f64::from(bf16::from_f32(dot as f32).to_f32())
            };
            let products: Vec<_> = (0..I::SIZE)
                .map(|k| {
                    let gate = projected(k);
                    let silu = bf16::from_f32((gate / (1.0 + (-gate).exp())) as f32);
                    bf16::from_f32(silu.to_f32() * projected(I::SIZE + k) as f32)
                })
                .collect();
            for column in 0..H::SIZE {
                let sum: f64 = products
                    .iter()
                    .enumerate()
                    .map(|(k, x)| {
                        f64::from(x.to_f32()) * f64::from(down[column * I::SIZE + k].to_f32())
                    })
                    .sum();
                let expected = f64::from(bf16::from_f32(sum as f32).to_f32());
                let actual = f64::from(got[(chip * R::SIZE + row) * H::SIZE + column].to_f32());
                assert!(actual.is_finite());
                error += (actual - expected).powi(2);
                norm += expected.powi(2);
            }
        }
    }
    let relative = (error / norm).sqrt();
    assert!(relative < 1e-2, "independent MLP relative L2 {relative}");
}
