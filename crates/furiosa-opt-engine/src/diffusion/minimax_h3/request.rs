//! Request-owned readers derived together from the checked packing geometry.

use furiosa_opt_models::models::diffusion::minimax_h3::{
    embeddings::Chips,
    packing::{Capacity, PARTS, Plan},
    positions::Positions,
    request::{BlockInputs, BlockPlan},
};
use furiosa_opt_models::ops::{
    attention::flash,
    collective::Pieces,
    index::packed::{RowInputs, RowPlan},
    rotary::Rotation,
};
use furiosa_opt_std::prelude::*;
use std::io;

/// Host checks finish before the first transfer. Every numerical reader derives from `plan`;
/// callers cannot mix another request's row transfers, modulation spans or packed-key order.
pub(super) struct Prepared<R, K, T, A, V> {
    blocks: BlockPlan<R, K>,
    rows: RowPlan<R, T, A, V>,
    positions: Positions,
}

impl<R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName> Prepared<R, K, T, A, V> {
    pub(super) fn plan(&self) -> Plan {
        self.blocks.packing()
    }

    pub(super) fn new(
        capacity: Capacity,
        tokens: usize,
        positions: &[[f64; 3]],
        theta: f32,
    ) -> io::Result<Self> {
        let plan = capacity.plan(tokens)?;
        let blocks = BlockPlan::new(plan)?;
        let rows = plan.row_plan()?;
        let positions = Positions::new(plan, positions, theta)?;
        Ok(Self {
            blocks,
            rows,
            positions,
        })
    }

    /// Initial metadata upload has no kernel readers. The engine completes it before admitting
    /// a launch and retains the resulting owner through all accepted request work.
    pub(super) async fn upload(self, device: &mut Device) -> io::Result<Request<R, K, T, A, V>> {
        let (plan, [low, high], visible) = self.positions.into_parts();
        let blocks = self.blocks.upload(device).await?;
        let rows = RowInputs::upload(device, &self.rows).await?;
        let low = HostTensor::<bf16, m![{ Symbol<Pieces<R, PARTS>> }, { Symbol<flash::Head> }]>::from_vec(low.into_iter().map(|value| bf16::from_bits(value.to_bits())))
            .to_hbm(&mut device.pdma)
            .await
            .map_err(io::Error::other)?;
        let high = HostTensor::<bf16, m![{ Symbol<Pieces<R, PARTS>> }, { Symbol<flash::Head> }]>::from_vec(high.into_iter().map(|value| bf16::from_bits(value.to_bits())))
            .to_hbm(&mut device.pdma)
            .await
            .map_err(io::Error::other)?;
        let visible = HostTensor::<bf16, m![K, { Symbol<flash::Span> }]>::from_vec(
            visible
                .into_iter()
                .map(|value| bf16::from_bits(value.to_bits())),
        )
        .to_hbm(&mut device.pdma)
        .await
        .map_err(io::Error::other)?;
        Ok(Request {
            plan,
            inputs: Inputs {
                blocks,
                rows,
                rotation: Rotation { low, high },
                visible,
            },
        })
    }
}

/// Host geometry stays outside device arguments; numerical entries borrow only `inputs`.
pub(super) struct Request<R: AxisName, K: AxisName, T, A, V> {
    pub(super) plan: Plan,
    pub(super) inputs: Inputs<R, K, T, A, V>,
}

/// Immutable numerical transport. A request's flight guard retains this complete owner rather
/// than only aliases of individual readers; the entry decodes block and row plans once.
#[derive(DeviceSend)]
pub(super) struct Inputs<R: AxisName, K: AxisName, T, A, V> {
    pub(super) blocks: BlockInputs<R, K>,
    pub(super) rows: RowInputs<R, T, A, V, Chips>,
    pub(super) rotation: Rotation<flash::Head, Chips, Pieces<R, PARTS>>,
    pub(super) visible: flash::Visible<K, Chips>,
}
