#![allow(dead_code)] // Shared by projection and MLP tests, which use different helpers.
//! Sparse independent numerical oracle shared by the projection device tests.
use furiosa_opt_models::{
    ops::linear::nvfp4::tiled::Scales,
    tensor::{form::nvfp4::Scale, scalar::Value as Count},
};
use furiosa_opt_std::prelude::*;

axes![
    Hidden = 4096,
    OutputWidth = 4096,
    Intermediate = 14336,
    Rows = 512
];

async fn scalar(device: &mut Device, value: f32) -> Scale {
    HostTensor::<f32, m![1 # 8]>::from_vec(vec![value; 8])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

fn feature<I: AxisName>(output: usize, term: usize) -> usize {
    if I::SIZE == 14336 {
        (output % 16) * 896 + term * 128 + output % 128
    } else {
        output * 97 % I::SIZE
    }
}

fn terms<I: AxisName>() -> usize {
    if I::SIZE == 14336 { 7 } else { 1 }
}
fn block_scale(output: usize, input: usize) -> f32 {
    [0.5, 1.0, 1.5, 2.0][(output + input / 16) % 4]
}
fn weight_value(output: usize) -> f32 {
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ][output % 16]
}
fn base_value(row: usize, feature: usize) -> f32 {
    [
        6.0, 0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5,
        -5.0,
    ][(row + feature) % 16]
}
fn amplitude(row: usize, feature: usize) -> f32 {
    [0.0, 0.25, 0.5, 1.0, 2.0][(row + feature / 16) % 5]
}
fn value(row: usize, feature: usize) -> f32 {
    base_value(row, feature) * amplitude(row, feature)
}
pub async fn weights<I: AxisName, O: AxisName, W: M, B: M>(
    device: &mut Device,
    pack_weight: impl FnOnce(&[f4e2m1]) -> Vec<f4e2m1>,
    pack_scale: impl FnOnce(&[f8e4m3]) -> Vec<f8e4m3>,
) -> (
    HbmTensor<f4e2m1, m![1], W>,
    HbmTensor<f8e4m3, m![1], B>,
    Scales,
) {
    weights_with::<I, O, W, B>(device, pack_weight, pack_scale, feature::<I>).await
}

pub async fn weights_with<I: AxisName, O: AxisName, W: M, B: M>(
    device: &mut Device,
    pack_weight: impl FnOnce(&[f4e2m1]) -> Vec<f4e2m1>,
    pack_scale: impl FnOnce(&[f8e4m3]) -> Vec<f8e4m3>,
    feature: impl Fn(usize, usize) -> usize,
) -> (
    HbmTensor<f4e2m1, m![1], W>,
    HbmTensor<f8e4m3, m![1], B>,
    Scales,
) {
    let mut codes = vec![f4e2m1::from_bits(0); O::SIZE * I::SIZE];
    for output in 0..O::SIZE {
        for term in 0..terms::<I>() {
            codes[output * I::SIZE + feature(output, term)] =
                f4e2m1::from_bits((output % 16) as u8);
        }
    }
    (
        HostTensor::<f4e2m1, W>::from_vec(pack_weight(&codes))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        HostTensor::<f8e4m3, B>::from_vec(pack_scale(
            &(0..O::SIZE * I::SIZE / 16)
                .map(|at| {
                    f8e4m3::from_f32(block_scale(at / (I::SIZE / 16), at % (I::SIZE / 16) * 16))
                })
                .collect::<Vec<_>>(),
        ))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap(),
        // Only weight_scale_2=.5 applies; fractional inputs detect unwanted A4 quantization.
        Scales {
            output: scalar(device, 0.5).await,
        },
    )
}

pub async fn input<I: AxisName>(device: &mut Device) -> HbmTensor<bf16, m![1], m![Rows, I]> {
    HostTensor::<bf16, m![Rows, I]>::from_vec(
        (0..Rows::SIZE * I::SIZE).map(|at| bf16::from_f32(value(at / I::SIZE, at % I::SIZE))),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap()
}

pub async fn count(device: &mut Device, rows: usize) -> Count {
    HostTensor::<i32, m![1]>::from_vec(vec![rows as i32])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

pub async fn check<I: AxisName, O: AxisName>(
    device: &mut Device,
    output: HbmTensor<bf16, m![1], m![Rows, O]>,
    rows: usize,
) {
    let values = output
        .to_host::<m![Rows, O]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    for (index, found) in values.into_iter().take(rows * O::SIZE).enumerate() {
        let (row, output) = (index / O::SIZE, index % O::SIZE);
        let expected = bf16::from_f32(
            (0..terms::<I>())
                .map(|term| {
                    let feature = feature::<I>(output, term);
                    base_value(row, feature)
                        * amplitude(row, feature)
                        * weight_value(output)
                        * block_scale(output, feature)
                        * 0.5
                })
                .sum::<f32>(),
        )
        .to_f32();
        assert_eq!(
            found.to_f32(),
            expected,
            "live {rows} row {row} output {output}"
        );
    }
}
