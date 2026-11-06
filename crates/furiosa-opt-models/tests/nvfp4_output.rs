#![feature(register_tool)]
#![register_tool(furiosa_opt)]

#[path = "support/nvfp4.rs"]
mod fixtures;
use fixtures::*;
use furiosa_opt_models::ops::linear::nvfp4::tiled::output::Output as OutputProjection;
use furiosa_opt_models::{
    Op,
    tensor::{live::Live, scalar::Value as Count},
};
use furiosa_opt_std::{prelude::*, runtime::launch};

#[device(chip = 1)]
fn project_output(
    device: &mut Device,
    projection: &OutputProjection<Hidden, OutputWidth>,
    count: &Count,
    input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
) -> HbmTensor<bf16, m![1], m![Rows, OutputWidth]> {
    projection.forward(device, (Live::<(), Rows>::read(count), input))
}

axes![Reach = 1024];
#[device(chip = 1)]
fn project_window(
    device: &mut Device,
    projection: &OutputProjection<Hidden, OutputWidth>,
    count: &Count,
    input: &HbmTensor<bf16, m![1], m![Reach, Hidden]>,
) -> HbmTensor<bf16, m![1], m![Rows, OutputWidth]> {
    projection.forward(device, (Live::<(), Rows>::read(count), input, 512))
}
#[tokio::test(flavor = "current_thread")]
#[ignore = "opens a device"]
async fn streams_output_in_eight_and_32_row_passes() {
    let mut device = Device::new(project_output.topology()).unwrap();
    let (weight, weight_scale, scales) = weights::<Hidden, OutputWidth, _, _>(
        &mut device,
        OutputProjection::<Hidden, OutputWidth>::reorder::<1, _>,
        OutputProjection::<Hidden, OutputWidth>::reorder::<16, _>,
    )
    .await;
    let mut projection = OutputProjection {
        weight,
        weight_scale,
        scales,
    };
    let input = input::<Hidden>(&mut device).await;
    let original = input
        .to_host::<m![Rows, Hidden]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    for rows in [
        1, 4, 8, 9, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512,
    ] {
        let mut padded = original.clone();
        padded[rows * Hidden::SIZE..].fill(bf16::from_f32(f32::NAN));
        let input = HostTensor::<bf16, m![Rows, Hidden]>::from_vec(padded.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut window = vec![bf16::from_f32(f32::NAN); Rows::SIZE * Hidden::SIZE];
        window.extend(padded);
        let window = HostTensor::<bf16, m![Reach, Hidden]>::from_vec(window)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let count = count(&mut device, rows).await;
        let started = std::time::Instant::now();
        let result = launch(project_output, (&mut device, &projection, &count, &input))
            .await
            .unwrap();
        eprintln!("projection rows={rows}: {:?}", started.elapsed());
        check::<Hidden, OutputWidth>(&mut device, result, rows).await;
        let result = launch(project_window, (&mut device, &projection, &count, &window))
            .await
            .unwrap();
        check::<Hidden, OutputWidth>(&mut device, result, rows).await;
    }
    // Dense contributions exercise every input partition and every packet sum.
    // BF16 activation 6 × restored weight 1/8 × 4096 × weight tensor factor.
    projection.weight =
        HostTensor::<
            f4e2m1,
            furiosa_opt_models::ops::linear::nvfp4::tiled::output::Stored<Hidden, OutputWidth>,
        >::from_vec(vec![f4e2m1::from_bits(2); Hidden::SIZE * OutputWidth::SIZE])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    projection.weight_scale = HostTensor::<
        f8e4m3,
        furiosa_opt_models::ops::linear::nvfp4::tiled::output::StoredScales<Hidden, OutputWidth>,
    >::from_vec(vec![
        f8e4m3::from_f32(0.125);
        Hidden::SIZE * OutputWidth::SIZE / 16
    ])
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let input = HostTensor::<bf16, m![Rows, Hidden]>::from_vec(vec![
        bf16::from_f32(6.0);
        Rows::SIZE * Hidden::SIZE
    ])
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    // Non-power-of-two and negative factors exercise scaling after the complete
    // FP32 inter-slice sum, including all four compact activation waves.
    for (rows, factor) in [(1, 0.125_f32), (512, 0.125), (1, 0.1337), (512, -0.1337)] {
        projection.scales.output = HostTensor::<f32, m![1 # 8]>::from_vec(vec![factor; 8])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let count = count(&mut device, rows).await;
        let result = launch(project_output, (&mut device, &projection, &count, &input))
            .await
            .unwrap()
            .to_host::<m![Rows, OutputWidth]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        assert!(
            result[..rows * OutputWidth::SIZE]
                .iter()
                .all(|v| *v == bf16::from_f32(3072.0 * factor))
        );
    }

    // The two input halves nearly cancel: +/-1536 plus a 0.125..0.5 residual.
    // Rounding cluster partials to BF16 before exchange would erase that residual.
    let factor = 0.1337_f32;
    projection.scales.output = HostTensor::<f32, m![1 # 8]>::from_vec(vec![factor; 8])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let mut cancellation = Vec::with_capacity(Rows::SIZE * Hidden::SIZE);
    for row in 0..Rows::SIZE {
        let sign = if row % 2 == 0 { 1.0 } else { -1.0 };
        let mut values = vec![6.0 * sign; Hidden::SIZE];
        values[Hidden::SIZE / 2..].fill(-6.0 * sign);
        values[0] += sign * (row % 4 + 1) as f32;
        cancellation.extend(values.into_iter().map(bf16::from_f32));
    }
    let input = HostTensor::<bf16, m![Rows, Hidden]>::from_vec(cancellation)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    for rows in [1, 16, 129, 257, 512] {
        let count = count(&mut device, rows).await;
        let result = launch(project_output, (&mut device, &projection, &count, &input))
            .await
            .unwrap()
            .to_host::<m![Rows, OutputWidth]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for row in 0..rows {
            let sign = if row % 2 == 0 { 1.0 } else { -1.0 };
            let expected = bf16::from_f32(sign * (row % 4 + 1) as f32 * 0.125 * factor);
            assert!(
                result[row * OutputWidth::SIZE..(row + 1) * OutputWidth::SIZE]
                    .iter()
                    .all(|value| *value == expected),
                "cancellation rows={rows}, row={row}"
            );
        }
    }

    // Independent FP64 oracle for dense, signed, varied-exponent dot products.
    // This checks the new FP32 reduction order without requiring bitwise equality
    // to a different FP32 summation tree. Test every output in nine boundary rows.
    let mut state = 42_u32;
    let mut random = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        state
    };
    let codes: Vec<_> = (0..Hidden::SIZE * OutputWidth::SIZE)
        .map(|_| f4e2m1::from_bits((random() >> 28) as u8))
        .collect();
    let scales: Vec<_> = (0..Hidden::SIZE * OutputWidth::SIZE / 16)
        .map(|_| {
            let value = random();
            f8e4m3::from_f32(
                (1.0 + (value & 7) as f32 / 8.0) * 2.0_f32.powi(((value >> 8) % 10) as i32 - 6),
            )
        })
        .collect();
    let inputs: Vec<_> = (0..Rows::SIZE * Hidden::SIZE)
        .map(|_| {
            let value = random();
            bf16::from_bits(
                ((value >> 16) as u16 & 0x8000)
                    | (((120 + ((value >> 7) & 15)) as u16) << 7)
                    | (value as u16 & 127),
            )
        })
        .collect();
    projection.weight = HostTensor::<
        f4e2m1,
        furiosa_opt_models::ops::linear::nvfp4::tiled::output::Stored<Hidden, OutputWidth>,
    >::from_vec(OutputProjection::<Hidden, OutputWidth>::reorder::<1, _>(
        &codes,
    ))
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    projection.weight_scale = HostTensor::<
        f8e4m3,
        furiosa_opt_models::ops::linear::nvfp4::tiled::output::StoredScales<Hidden, OutputWidth>,
    >::from_vec(OutputProjection::<Hidden, OutputWidth>::reorder::<
        16,
        _,
    >(&scales))
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let input = HostTensor::<bf16, m![Rows, Hidden]>::from_vec(inputs.clone())
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    // Check both sides of the direct/restored boundary and retain the full launch.
    for rows in [32, 33, Rows::SIZE] {
        let count = count(&mut device, rows).await;
        let result = launch(project_output, (&mut device, &projection, &count, &input))
            .await
            .unwrap()
            .to_host::<m![Rows, OutputWidth]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let (mut checked, mut exact, mut max_rounding_units) = (0, 0, 0.0_f64);
        for row in [0, 7, 31, 32, 127, 128, 255, 256, 511]
            .into_iter()
            .filter(|row| *row < rows)
        {
            for output in 0..OutputWidth::SIZE {
                let (mut sum, mut absolute_sum) = (0.0_f64, 0.0_f64);
                for input in 0..Hidden::SIZE {
                    let term = inputs[row * Hidden::SIZE + input].to_f32() as f64
                        * codes[output * Hidden::SIZE + input].to_f32() as f64
                        * scales[output * Hidden::SIZE / 16 + input / 16].to_f32() as f64;
                    sum += term;
                    absolute_sum += term.abs();
                }
                let expected = sum * factor as f64;
                let rounded = bf16::from_f32(expected as f32);
                let ulp = 2.0_f64.powi(((rounded.to_bits() >> 7) & 255) as i32 - 127 - 7);
                let actual = result[row * OutputWidth::SIZE + output];
                let error = (actual.to_f32() as f64 - expected).abs();
                // BF16 rounding plus a small FP32 accumulation allowance. The dense
                // oracle is independent of the cluster, slice, and contraction order.
                let tolerance = 0.5 * ulp + 1e-6 * absolute_sum * (factor as f64).abs();
                assert!(
                    actual.to_f32().is_finite() && error <= tolerance,
                    "dense oracle rows={rows} row={row} output={output}: actual={actual:?} expected={expected} error={error} tolerance={tolerance}"
                );
                checked += 1;
                exact += usize::from(actual == rounded);
                max_rounding_units = max_rounding_units.max(error / ulp);
            }
        }
        eprintln!(
            "Dense FP64 oracle rows={rows}: {exact}/{checked} match rounded reference; max error {max_rounding_units:.6} BF16 ULP"
        );
    }
}
