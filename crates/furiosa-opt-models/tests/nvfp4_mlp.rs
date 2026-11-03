#![feature(register_tool)]
#![register_tool(furiosa_opt)]

#[path = "support/nvfp4.rs"]
mod fixtures;

type Rows = furiosa_opt_models::tensor::axis::Rows<512>;
use furiosa_opt_models::{
    models::decoder::attention::llama::nvfp4::{Hidden, Intermediate, QWidth},
    ops::linear::nvfp4::tiled::{mlp, panels::Panels, qkv::QkvProjection},
    tensor::scalar::Value as Count,
};
use furiosa_opt_std::{prelude::*, runtime::launch};

#[device(chip = 1)]
fn fused(
    device: &mut Device,
    mlp: &mlp::Mlp<Hidden, Intermediate>,
    count: &Count,
    input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
) -> HbmTensor<bf16, m![1], m![Rows, Hidden]> {
    let rows = furiosa_opt_models::tensor::live::Live::<(), Rows>::read(count).rows();
    let input = input.to_dm(&mut device.tdma);
    let output = mlp::forward::<Hidden, Intermediate, Rows>(
        device,
        &mlp.gate_proj,
        &mlp.up_proj,
        &mlp.down_proj,
        &input,
        rows,
    );
    output.to_hbm(&mut device.tdma)
}

fn input(row: usize, feature: usize) -> f32 {
    let values = [
        6.0, 0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5,
        -5.0,
    ];
    values[(row + feature) % 16] * [0.0, 0.25, 0.5, 1.0, 2.0][(row + feature / 16) % 5]
}

fn scalar_weight(output: usize, feature: usize) -> f32 {
    let codes = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    codes[output % 16] * [0.5, 1.0, 1.5, 2.0][(output + feature / 16) % 4]
}

/// Independent scalar W4A16 oracle, including both BF16 SwiGLU roundings.
fn reference(row: usize) -> Vec<f32> {
    let hidden = (0..4096).map(|i| input(row, i)).collect::<Vec<_>>();
    let product = (0..14336)
        .map(|out| {
            let feature = out * 97 % 4096;
            let dot = hidden[feature] * scalar_weight(out, feature);
            let gate = bf16::from_f32(dot * 0.5).to_f32();
            let up = bf16::from_f32(dot * 0.25).to_f32();
            let silu = bf16::from_f32(gate * (1.0 / (1.0 + (-gate).exp()))).to_f32();
            bf16::from_f32(silu * up).to_f32()
        })
        .collect::<Vec<_>>();
    let operand = product;
    (0..4096)
        .map(|out| {
            let sum = (0..7)
                .map(|term| {
                    let feature = ((out % 16 + 8 * (term % 2)) % 16) * 896 + term * 128 + out % 128;
                    operand[feature] * scalar_weight(out, feature)
                })
                .sum::<f32>();
            bf16::from_f32(sum * 0.5).to_f32()
        })
        .collect()
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "opens a device"]
async fn fused_mlp_matches_scalar_w4a16_with_poisoned_padding() {
    let mut device = Device::new(fused.topology()).unwrap();
    let (weight, weight_scale, scales) = fixtures::weights::<Hidden, Intermediate, _, _>(
        &mut device,
        Panels::<Hidden, Intermediate>::reorder::<1, _>,
        Panels::<Hidden, Intermediate>::reorder::<16, _>,
    )
    .await;
    let gate = Panels {
        weight,
        weight_scale,
        scales,
    };
    let (weight, weight_scale, mut scales) = fixtures::weights::<Hidden, Intermediate, _, _>(
        &mut device,
        Panels::<Hidden, Intermediate>::reorder::<1, _>,
        Panels::<Hidden, Intermediate>::reorder::<16, _>,
    )
    .await;
    // Distinct weight tensor factors detect swapped gate/up paths.
    scales.output = HostTensor::<f32, m![1 # 8]>::from_vec(vec![0.25; 8])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let up = Panels {
        weight,
        weight_scale,
        scales,
    };
    let (weight, weight_scale, scales) = fixtures::weights_with::<Intermediate, Hidden, _, _>(
        &mut device,
        mlp::Down::<Intermediate, Hidden>::reorder::<1, _>,
        mlp::Down::<Intermediate, Hidden>::reorder::<16, _>,
        // Select terms from both intermediate-feature clusters for each output.
        |out, term| ((out % 16 + 8 * (term % 2)) % 16) * 896 + term * 128 + out % 128,
    )
    .await;
    let mlp = mlp::Mlp {
        gate_proj: gate,
        up_proj: up,
        down_proj: mlp::Down {
            weight,
            weight_scale,
            scales,
        },
    };
    let expected = (0..512).map(reference).collect::<Vec<_>>();
    for rows in [
        1, 4, 8, 9, 16, 17, 32, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512,
    ] {
        let input = HostTensor::<bf16, m![Rows, Hidden]>::from_vec((0..512 * 4096).map(|at| {
            bf16::from_f32(if at / 4096 < rows {
                input(at / 4096, at % 4096)
            } else {
                f32::NAN
            })
        }))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let count = fixtures::count(&mut device, rows).await;
        let output = launch(fused, (&mut device, &mlp, &count, &input))
            .await
            .unwrap()
            .to_host::<m![Rows, Hidden]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let mut max_error = 0.0_f32;
        for at in 0..rows * 4096 {
            let found = output[at].to_f32();
            let want = expected[at / 4096][at % 4096];
            let error = (found - want).abs();
            max_error = max_error.max(error);
            assert!(
                found.is_finite() && found == want,
                "rows={rows} row={} feature={} found={found} expected={want}",
                at / 4096,
                at % 4096
            );
        }
        eprintln!("fused MLP oracle rows={rows}: max absolute error {max_error}");
    }
    #[cfg(feature = "kernel-diagnostics")]
    {
        use furiosa_opt_models::{
            models::decoder::attention::llama::nvfp4::{Norm, QWidth, Qkv},
            tensor::form::Weight,
        };
        let qkv = Qkv {
            query: furiosa_opt_models::load::Named(projection(&mut device).await),
            key: furiosa_opt_models::load::Named(projection(&mut device).await),
            value: furiosa_opt_models::load::Named(projection(&mut device).await),
        };
        let norm = Norm::new(Weight::new(
            HostTensor::<bf16, m![Hidden]>::from_vec(vec![bf16::from_f32(1.0); 4096])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        ));
        for rows in [
            1, 4, 8, 9, 16, 17, 32, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512,
        ] {
            let input = HostTensor::<bf16, m![Rows, Hidden]>::from_vec((0..512 * 4096).map(|at| {
                bf16::from_f32(if at / 4096 < rows {
                    input(at / 4096, at % 4096)
                } else {
                    f32::NAN
                })
            }))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            let count = fixtures::count(&mut device, rows).await;
            let output = launch(transition, (&mut device, &mlp, &norm, &qkv, &count, &input))
                .await
                .unwrap()
                .to_host::<m![Rows, QWidth]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for row in 0..rows {
                let scale = (expected[row].iter().map(|x| x * x).sum::<f32>() / 4096.0 + 1e-5)
                    .sqrt()
                    .recip();
                for out in 0..4096 {
                    let feature = out * 97 % 4096;
                    let normalized = bf16::from_f32(expected[row][feature] * scale).to_f32();
                    let want =
                        bf16::from_f32(normalized * scalar_weight(out, feature) * 0.5).to_f32();
                    let got = output[row * 4096 + out].to_f32();
                    assert!(
                        got.is_finite() && (got - want).abs() <= 0.02_f32.max(want.abs() * 0.015),
                        "prefetched MLP rows={rows} row={row} feature={out} got={got} want={want}"
                    );
                }
            }
            eprintln!("MLP -> prefetched norm/QKV oracle rows={rows}: passed");
        }
    }
}
/// Exercise the real next-layer prefetch while the down projection is still live.
#[cfg(feature = "kernel-diagnostics")]
#[device(chip = 1)]
fn transition(
    device: &mut Device,
    mlp: &mlp::Mlp<Hidden, Intermediate>,
    norm: &furiosa_opt_models::models::decoder::attention::llama::nvfp4::Norm,
    qkv: &furiosa_opt_models::models::decoder::attention::llama::nvfp4::Qkv,
    count: &Count,
    input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
) -> HbmTensor<bf16, m![1], m![Rows, QWidth]> {
    use furiosa_opt_models::models::decoder::attention::llama::nvfp4::{
        QWidth, inspect_transition,
    };
    let rows = furiosa_opt_models::tensor::live::Live::<(), Rows>::read(count).rows();
    let input = input.to_dm(&mut device.tdma);
    let (q, _, _) = inspect_transition(device, mlp, norm, qkv, &input, rows);
    let mut output = HbmTensor::<bf16, m![1], m![Rows, QWidth]>::new();
    q.view().to_hbm_view(&mut device.tdma, output.view_mut());
    output
}

#[cfg(feature = "kernel-diagnostics")]
async fn projection<O: AxisName>(device: &mut Device) -> QkvProjection<Hidden, O> {
    let (weight, weight_scale, scales) = fixtures::weights::<Hidden, O, _, _>(
        device,
        QkvProjection::<Hidden, O>::reorder::<1, _>,
        QkvProjection::<Hidden, O>::reorder::<16, _>,
    )
    .await;
    QkvProjection {
        weight,
        weight_scale,
        scales,
    }
}
