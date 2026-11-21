//! Expand H3's geometry into fixed-size whole-transfer choices.

use std::{env, fmt::Write, fs, path::PathBuf};

#[path = "src/models/diffusion/minimax_h3/packing/geometry.rs"]
mod geometry;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/models/diffusion/minimax_h3/packing/geometry.rs");
    let mut extents = String::from("pub(crate) type Transfers = ");
    for step in 0..geometry::TRANSFER_ROWS.len() {
        write!(extents, "(Transfer<{step}>, ").unwrap();
    }
    extents.push_str(&")".repeat(geometry::TRANSFER_ROWS.len()));
    extents.push_str(";\n");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    fs::write(output.join("h3_transfers.rs"), extents).unwrap();
}
