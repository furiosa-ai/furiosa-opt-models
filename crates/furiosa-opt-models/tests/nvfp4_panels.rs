#![feature(register_tool)]
#![register_tool(furiosa_opt)]

#[path = "support/nvfp4.rs"]
mod fixtures;
use fixtures::*;
use furiosa_opt_models::ops::linear::nvfp4::tiled::panels::Panels;
use furiosa_opt_models::{
    Op,
    tensor::{live::Live, scalar::Value as Count},
};
use furiosa_opt_std::{prelude::*, runtime::launch};

#[device(chip = 1)]
fn gate(
    device: &mut Device,
    projection: &Panels<Hidden, Intermediate>,
    count: &Count,
    input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
) -> HbmTensor<bf16, m![1], m![Rows, Intermediate]> {
    projection.forward(device, (Live::<(), Rows>::read(count), input))
}

/// Exercise all four hidden panels, 224-wide feature groups, tie rounding, tensor/block
/// scales, both clusters and partial token waves. The sparse oracle is independent of tiling.
#[tokio::test(flavor = "current_thread")]
#[ignore = "opens a device"]
async fn projects_llama_gate_geometry() {
    let mut device = Device::new(gate.topology()).unwrap();
    let (weight, weight_scale, scales) = weights::<Hidden, Intermediate, _, _>(
        &mut device,
        Panels::<Hidden, Intermediate>::reorder::<1, _>,
        Panels::<Hidden, Intermediate>::reorder::<16, _>,
    )
    .await;
    let projection = Panels {
        weight,
        weight_scale,
        scales,
    };
    let input = input::<Hidden>(&mut device).await;
    for rows in [1, 16, 129, 512] {
        let count = count(&mut device, rows).await;
        let started = std::time::Instant::now();
        let output = launch(gate, (&mut device, &projection, &count, &input))
            .await
            .unwrap();
        eprintln!("projection rows={rows}: {:?}", started.elapsed());
        check::<Hidden, Intermediate>(&mut device, output, rows).await;
    }
}
