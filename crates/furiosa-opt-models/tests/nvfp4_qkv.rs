#![feature(register_tool)]
#![register_tool(furiosa_opt)]
#[path = "support/nvfp4.rs"]
mod fixtures;
use fixtures::*;
use furiosa_opt_models::{
    ops::linear::nvfp4::tiled::{panels::Panels, qkv::QkvProjection},
    tensor::scalar::Value as Count,
};
use furiosa_opt_std::{prelude::*, runtime::launch};
axes![KvWidth = 1024];
#[device(chip = 1)]
fn query(
    device: &mut Device,
    projection: &Panels<Hidden, OutputWidth>,
    count: &Count,
    input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
) -> HbmTensor<bf16, m![1], m![Rows, OutputWidth]> {
    furiosa_opt_models::Op::forward(
        projection,
        device,
        (
            furiosa_opt_models::tensor::live::Live::<(), Rows>::read(count),
            input,
        ),
    )
}
#[device(chip = 1)]
fn kv(
    device: &mut Device,
    projection: &Panels<Hidden, KvWidth>,
    count: &Count,
    input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
) -> HbmTensor<bf16, m![1], m![Rows, KvWidth]> {
    furiosa_opt_models::Op::forward(
        projection,
        device,
        (
            furiosa_opt_models::tensor::live::Live::<(), Rows>::read(count),
            input,
        ),
    )
}
#[device(chip = 1)]
fn shared(
    device: &mut Device,
    q: &QkvProjection<Hidden, OutputWidth>,
    k: &QkvProjection<Hidden, KvWidth>,
    v: &QkvProjection<Hidden, KvWidth>,
    count: &Count,
    input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
) -> (
    HbmTensor<bf16, m![1], m![Rows, OutputWidth]>,
    HbmTensor<bf16, m![1], m![Rows, KvWidth]>,
    HbmTensor<bf16, m![1], m![Rows, KvWidth]>,
) {
    let own = input.to_dm(&mut device.tdma);
    let (q, k, v) = furiosa_opt_models::models::decoder::attention::llama::nvfp4::inspect_qkv_heads::<
        Hidden,
        OutputWidth,
        KvWidth,
        Rows,
    >(
        device,
        (q, k, v),
        &own,
        furiosa_opt_models::tensor::live::Live::<(), Rows>::read(count).rows(),
    );
    let (mut qo, mut ko, mut vo) = (HbmTensor::new(), HbmTensor::new(), HbmTensor::new());
    q.view().to_hbm_view(&mut device.tdma, qo.view_mut());
    k.view().to_hbm_view(&mut device.tdma, ko.view_mut());
    v.view().to_hbm_view(&mut device.tdma, vo.view_mut());
    (qo, ko, vo)
}
#[tokio::test(flavor = "current_thread")]
#[ignore = "opens a device"]
async fn resident_q_and_kv_match_scalar_w4a16() {
    let mut device = Device::new(query.topology()).unwrap();
    let (weight, weight_scale, scales) = weights::<Hidden, OutputWidth, _, _>(
        &mut device,
        Panels::<Hidden, OutputWidth>::reorder::<1, _>,
        Panels::<Hidden, OutputWidth>::reorder::<16, _>,
    )
    .await;
    let mut q = Panels {
        weight,
        weight_scale,
        scales,
    };
    let (weight, weight_scale, scales) = weights::<Hidden, KvWidth, _, _>(
        &mut device,
        Panels::<Hidden, KvWidth>::reorder::<1, _>,
        Panels::<Hidden, KvWidth>::reorder::<16, _>,
    )
    .await;
    let k = Panels {
        weight,
        weight_scale,
        scales,
    };
    let mut qn = projection::<OutputWidth>(&mut device).await;
    let kn = projection::<KvWidth>(&mut device).await;
    let mut vn = projection::<KvWidth>(&mut device).await;
    vn.scales.output = HostTensor::<f32, m![1 # 8]>::from_vec(vec![-0.5; 8])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let input = input::<Hidden>(&mut device).await;
    let original = input
        .to_host::<m![Rows, Hidden]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    for rows in [
        1, 4, 8, 9, 16, 17, 23, 24, 25, 32, 63, 64, 65, 71, 72, 73, 127, 128, 129, 247, 248, 249,
        255, 256, 257, 383, 384, 385, 511, 512,
    ] {
        let mut padded = original.clone();
        padded[rows * Hidden::SIZE..].fill(bf16::from_f32(f32::NAN));
        let input = HostTensor::<bf16, m![Rows, Hidden]>::from_vec(padded)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let count = count(&mut device, rows).await;
        let out = launch(query, (&mut device, &q, &count, &input))
            .await
            .unwrap();
        check::<Hidden, OutputWidth>(&mut device, out, rows).await;
        let out = launch(kv, (&mut device, &k, &count, &input)).await.unwrap();
        check::<Hidden, KvWidth>(&mut device, out, rows).await;
        let (qo, ko, vo) = launch(shared, (&mut device, &qn, &kn, &vn, &count, &input))
            .await
            .unwrap();
        let kh = ko
            .to_host::<m![Rows, KvWidth]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let vh = vo
            .to_host::<m![Rows, KvWidth]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        assert!(
            kh.iter()
                .zip(&vh)
                .take(rows * KvWidth::SIZE)
                .all(|(k, v)| k.to_f32() == -v.to_f32()),
            "V must use its own output scale"
        );
        check::<Hidden, OutputWidth>(&mut device, qo, rows).await;
        check::<Hidden, KvWidth>(&mut device, ko, rows).await;
    }
    // Dense nonzero operands exercise accumulation through every hidden slab.
    // Codes=1, block scale=1/8, BF16 input=6.
    // Each of 4096 products is 0.75; weight tensor factor=1/2 => 1536.
    q.weight = HostTensor::<
        f4e2m1,
        furiosa_opt_models::ops::linear::nvfp4::tiled::panels::Stored<Hidden, OutputWidth>,
    >::from_vec(vec![f4e2m1::from_bits(2); Hidden::SIZE * OutputWidth::SIZE])
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    q.weight_scale = HostTensor::<
        f8e4m3,
        furiosa_opt_models::ops::linear::nvfp4::tiled::panels::StoredScales<Hidden, OutputWidth>,
    >::from_vec(vec![
        f8e4m3::from_f32(0.125);
        Hidden::SIZE * OutputWidth::SIZE / 16
    ])
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    qn.weight = HostTensor::<
        f4e2m1,
        furiosa_opt_models::ops::linear::nvfp4::tiled::qkv::Stored<Hidden, OutputWidth>,
    >::from_vec(vec![f4e2m1::from_bits(2); Hidden::SIZE * OutputWidth::SIZE])
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    qn.weight_scale = HostTensor::<
        f8e4m3,
        furiosa_opt_models::ops::linear::nvfp4::tiled::qkv::StoredScales<Hidden, OutputWidth>,
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
    let count = count(&mut device, 1).await;
    let out = launch(query, (&mut device, &q, &count, &input))
        .await
        .unwrap()
        .to_host::<m![Rows, OutputWidth]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    assert!(
        out[..OutputWidth::SIZE]
            .iter()
            .all(|v| *v == bf16::from_f32(1536.0))
    );
    let (shared_q, _, _) = launch(shared, (&mut device, &qn, &kn, &vn, &count, &input))
        .await
        .unwrap();
    let shared_q = shared_q
        .to_host::<m![Rows, OutputWidth]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    assert!(
        shared_q[..OutputWidth::SIZE]
            .iter()
            .all(|v| *v == bf16::from_f32(1536.0))
    );
}

async fn projection<O: AxisName>(device: &mut Device) -> QkvProjection<Hidden, O> {
    let (weight, weight_scale, scales) = weights::<Hidden, O, _, _>(
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

#[test]
fn checkpoint_reorder_preserves_packed_weight_and_scale_coordinates() {
    fn check<const PACK: usize>() {
        let original: Vec<_> = (0..Hidden::SIZE * KvWidth::SIZE / PACK).collect();
        let reordered = QkvProjection::<Hidden, KvWidth>::reorder::<PACK, _>(&original);
        // Literal landmarks independently pin output-row, input-slice, and owner transitions.
        if PACK == 2 {
            assert_eq!(
                [
                    reordered[31],
                    reordered[32],
                    reordered[4095],
                    reordered[4096],
                    reordered[262144]
                ],
                [31, 2048, 260127, 32, 262144]
            );
        } else {
            assert_eq!(
                [
                    reordered[3],
                    reordered[4],
                    reordered[511],
                    reordered[512],
                    reordered[32768]
                ],
                [3, 256, 32515, 4, 32768]
            );
        }

        for owner in 0..8 {
            for slice in 0..64 {
                for out in 0..128 {
                    for column in 0..64 / PACK {
                        let target = ((owner * 64 + slice) * 128 + out) * (64 / PACK) + column;
                        let source = ((owner * 128 + out) * 4096 + slice * 64) / PACK + column;
                        assert_eq!(reordered[target], original[source]);
                    }
                }
            }
        }
    }
    check::<2>();
    check::<16>();
}
