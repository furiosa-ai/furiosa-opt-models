//! Check the table that the composed launch actually loads, independently of Spectrum::bind.
use furiosa_opt_models::{
    load::Checkpoint,
    ops::rotary::{
        Row, SplitHalf,
        group::{InputHalf, OutputHalf, Table},
    },
};
use furiosa_opt_std::prelude::*;

axes![Head = 128, Position = 8193];

#[tokio::test(flavor = "current_thread")]
#[ignore = "opens a device"]
async fn loads_llama3_scaled_rotation_coefficients() {
    let config = serde_json::from_str(include_str!("fixtures/llama_nvfp4_config.json")).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = Checkpoint::new(directory.path(), Some(config)).unwrap();
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let rotation: SplitHalf<Head, Table<Position, Row<Head>>> =
        checkpoint.load(&mut device).unwrap();
    let table = rotation
        .frequency
        .to_host::<m![Position, { Symbol<Row<Head>> }, OutputHalf, InputHalf]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    // Published Llama 3.1 rule: preserve short wavelengths, divide long ones by eight,
    // and interpolate between the original 8192-token context's two thresholds.
    for position in [0, 1, 1024, 8192] {
        for pair in 0..64 {
            let base = 500_000_f64.powf(-2.0 * pair as f64 / 128.0);
            let wavelength = std::f64::consts::TAU / base;
            let frequency = if wavelength < 8192.0 / 4.0 {
                base
            } else if wavelength > 8192.0 {
                base / 8.0
            } else {
                let smooth = (8192.0 / wavelength - 1.0) / 3.0;
                (1.0 - smooth) * base / 8.0 + smooth * base
            };
            let angle = f64::from(position as f32 * frequency as f32);
            for (cell, expected) in [angle.cos(), -angle.sin(), angle.sin(), angle.cos()]
                .into_iter()
                .enumerate()
            {
                assert_eq!(
                    table[(position * 64 + pair) * 4 + cell],
                    bf16::from_f32(expected as f32),
                    "position={position} pair={pair} cell={cell}",
                );
            }
        }
    }
}
