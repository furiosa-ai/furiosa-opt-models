//! Engine-owned joint request with model-checked numerical metadata.

use furiosa_opt_models::models::diffusion::minimax_h3::packing::Plan;
use furiosa_opt_std::prelude::*;
use std::io;

pub(super) use furiosa_opt_models::models::diffusion::minimax_h3::request::{Inputs, Prepared};

/// Host geometry stays outside device arguments; numerical entries borrow only `inputs`.
pub(super) struct Request<R: AxisName, K: AxisName, T, A, V> {
    pub(super) plan: Plan,
    pub(super) inputs: Inputs<R, K, T, A, V>,
}

impl<R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName> Request<R, K, T, A, V> {
    /// Upload one checked metadata set before the engine admits any numerical reader.
    pub(super) async fn upload(
        prepared: Prepared<R, K, T, A, V>,
        device: &mut Device,
    ) -> io::Result<Self> {
        let plan = prepared.plan();
        let inputs = prepared.upload(device).await?;
        Ok(Self { plan, inputs })
    }
}
