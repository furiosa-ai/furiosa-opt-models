//! Bind the numerical consumers' work to one checked H3 request before uploading metadata.

use super::packing::{
    AUDIO_ROWS, Capacity, DEFAULT_MAX_TEXT, PARTS, Plan, ROW_BLOCK, TEXT_STRIP, VIDEO_ROWS,
};
use super::{embeddings::Chips, positions::Positions};
use crate::ops::attention::{
    flash,
    full::{PackInputs, PackPlan},
};
use crate::ops::collective::Pieces;
use crate::ops::index::packed::{RowInputs, RowPlan, SpanPlan, live::SpanInputs};
use crate::ops::rotary::Rotation;
use crate::tensor::{axis::Rows, live::BlockPrefix, scalar::Value};
use furiosa_opt_std::prelude::*;
use std::io;

/// At most one valid run per modality and part. Empty part spans contribute no run.
pub const KEY_RUNS: usize = 3 * PARTS;

/// Whole-transfer work extents inside the Live envelope, derived from the modality geometry.
/// These are not token buckets: every valid text length still supplies its own keys and spans.
#[derive(Clone, Debug)]
pub(crate) struct Transfer<const STEP: usize>;

impl<const STEP: usize> AxisName for Transfer<STEP> {
    const NAME: Ident = Ident::new("JointTransfer");
    const SIZE: usize = {
        let first = (AUDIO_ROWS + VIDEO_ROWS + TEXT_STRIP).div_ceil(ROW_BLOCK) * ROW_BLOCK;
        let text = DEFAULT_MAX_TEXT.div_ceil(PARTS * TEXT_STRIP) * TEXT_STRIP;
        let last = (AUDIO_ROWS + VIDEO_ROWS + text).div_ceil(ROW_BLOCK) * ROW_BLOCK;
        // The joint execution adapter dispatches these three whole-transfer geometries.
        assert!(last - first == 2 * ROW_BLOCK && STEP <= 2);
        first + STEP * ROW_BLOCK
    };
}

/// A block's local initialized work, within the compiled gather allocation.
pub type Prefix<R> = BlockPrefix<(), R, Rows<ROW_BLOCK>>;
/// Valid semantic K/V transfers from the compact joined projection.
pub type Keys<R, K> = PackPlan<Pieces<R, PARTS>, K, KEY_RUNS>;

/// Host preparation for a single compiled local-row and key-span allocation.
///
/// Modulation, projection and K/V plans all derive from the same [`Plan`]. Construction checks
/// the compiled capacities and every transport field before the first upload. This describes
/// addresses and initialized work; positions, visibility, hidden tensors and weights are separate.
pub struct BlockPlan<R, K> {
    plan: Plan,
    spans: SpanPlan<R, TEXT_STRIP>,
    prefix: Prefix<R>,
    keys: Keys<R, K>,
    blocks: i32,
}

impl<R: AxisName, K: AxisName> BlockPlan<R, K> {
    pub fn new(plan: Plan) -> io::Result<Self> {
        if plan.capacity().max_text() > DEFAULT_MAX_TEXT {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("joint execution supports maximum text capacity {DEFAULT_MAX_TEXT}"),
            ));
        }
        if K::SIZE != plan.capacity().key_spans() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "compiled key spans differ from the request capacity",
            ));
        }
        let spans = plan.span_plan::<R>()?;
        let prefix = plan.block_prefix::<R>()?;
        let keys = Keys::try_from_runs(
            0..plan.query_rows(),
            plan.key_runs().map(|run| (run.source(), run.destination())),
        )?;
        spans.strip_counts()?;
        let blocks = i32::try_from(prefix.blocks()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "projection blocks exceed I32")
        })?;
        Ok(Self {
            plan,
            spans,
            prefix,
            keys,
            blocks,
        })
    }

    pub fn packing(&self) -> Plan {
        self.plan
    }

    pub fn spans(&self) -> &SpanPlan<R, TEXT_STRIP> {
        &self.spans
    }

    pub fn prefix(&self) -> &Prefix<R> {
        &self.prefix
    }

    pub fn keys(&self) -> &Keys<R, K> {
        &self.keys
    }

    /// Upload outside the repeated block loop. Keep the returned owner through every launch
    /// borrowing it. A failed upload drops only this preparation's completed partial inputs.
    pub async fn upload(&self, device: &mut Device) -> io::Result<BlockInputs<R, K>> {
        let spans = SpanInputs::upload(device, &self.spans).await?;
        let keys = PackInputs::upload(device, &self.keys).await?;
        let blocks = HostTensor::<i32, m![1]>::from_vec([self.blocks])
            .to_hbm::<Chips, m![1]>(&mut device.pdma)
            .await
            .map_err(io::Error::other)?;
        Ok(BlockInputs {
            spans,
            blocks,
            keys,
        })
    }
}

/// Immutable block metadata owned by request state. Fields are private: construction is only
/// through [`BlockPlan::upload`], and consumers receive shared readers.
#[derive(DeviceSend)]
pub struct BlockInputs<R, K> {
    spans: SpanInputs<R, Chips, TEXT_STRIP>,
    blocks: Value<Chips>,
    keys: PackInputs<Pieces<R, PARTS>, K, Chips, KEY_RUNS>,
}

impl<R: AxisName, K: AxisName> BlockInputs<R, K> {
    /// Decode once at numerical entry and reuse the resulting call through the layer stack.
    /// Metadata is prepared as one unit; do not bind another request's tensor work to this call.
    pub fn read(&self) -> BlockCall<'_, R, K> {
        BlockCall {
            spans: self.spans.read(),
            prefix: BlockPrefix::read(&self.blocks),
            keys: self.keys.read(),
            key_count: self.keys.key_count(),
        }
    }
}

/// Local decoded readers, shared by the block's norms, projections and attention. The key-count
/// reference retains its request owner; this value does not own or initialize any hidden rows.
pub struct BlockCall<'a, R, K> {
    spans: SpanPlan<R, TEXT_STRIP>,
    prefix: Prefix<R>,
    keys: Keys<R, K>,
    key_count: &'a Value<Chips>,
}

impl<R: AxisName, K: AxisName> BlockCall<'_, R, K> {
    pub fn spans(&self) -> &SpanPlan<R, TEXT_STRIP> {
        &self.spans
    }

    pub fn prefix(&self) -> &Prefix<R> {
        &self.prefix
    }

    pub fn keys(&self) -> &Keys<R, K> {
        &self.keys
    }

    pub fn key_count(&self) -> &Value<Chips> {
        self.key_count
    }
}

/// Host checks finish before the first transfer. Every numerical reader derives from `plan`;
/// callers cannot mix another request's row transfers, modulation spans or packed-key order.
pub struct Prepared<R, K, T, A, V> {
    blocks: BlockPlan<R, K>,
    rows: RowPlan<R, T, A, V>,
    positions: Positions,
}

impl<R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName> Prepared<R, K, T, A, V> {
    /// Return the one packing plan shared by all prepared numerical inputs.
    pub fn plan(&self) -> Plan {
        self.blocks.packing()
    }

    /// Check joint geometry and positions before allocating any device storage.
    pub fn new(
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
    pub async fn upload(self, device: &mut Device) -> io::Result<Inputs<R, K, T, A, V>> {
        let (_, [low, high], visible) = self.positions.into_parts();
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
        Ok(Inputs {
            blocks,
            rows,
            rotation: Rotation { low, high },
            visible,
        })
    }
}

/// Immutable numerical transport. A request's flight guard retains this complete owner rather
/// than only aliases of individual readers; the entry decodes block and row plans once.
/// Constructed only through [`Prepared::upload`] from one checked request.
#[derive(DeviceSend)]
pub struct Inputs<R: AxisName, K: AxisName, T, A, V> {
    /// Checked spans, initialized work and packed-key metadata.
    pub(crate) blocks: BlockInputs<R, K>,
    /// Checked text/audio/video row transfers.
    pub(crate) rows: RowInputs<R, T, A, V, Chips>,
    /// Paired rotary coefficients for the joined rows.
    pub(crate) rotation: Rotation<flash::Head, Chips, Pieces<R, PARTS>>,
    /// Attention visibility for the packed key spans.
    pub(crate) visible: flash::Visible<K, Chips>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::diffusion::minimax_h3::packing::Capacity;

    axes![Twenty = 20, Nineteen = 19];
    type Local = Rows<10496>;

    #[test]
    fn all_consumer_intervals_derive_from_the_same_validated_request() {
        for tokens in 1..=2048 {
            let source = Capacity::default().plan(tokens).unwrap();
            let prepared = BlockPlan::<Local, Twenty>::new(source).unwrap();
            assert_eq!(prepared.packing(), source);
            assert_eq!(prepared.spans().work_rows(), prepared.prefix().rows());
            assert_eq!(
                prepared.keys().source_work().rows(),
                PARTS * prepared.prefix().rows()
            );
            assert_eq!(prepared.keys().key_count(), source.key_count());
            assert_eq!(
                prepared.keys().runs().collect::<Vec<_>>(),
                source
                    .key_runs()
                    .map(|run| (run.source(), run.destination()))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn preparation_refuses_capacity_mismatches_before_any_device_is_opened() {
        let source = Capacity::default().plan(77).unwrap();
        assert_eq!(
            BlockPlan::<Rows<9984>, Twenty>::new(source)
                .err()
                .unwrap()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            BlockPlan::<Local, Nineteen>::new(source)
                .err()
                .unwrap()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        let exact = Capacity::new(1202).unwrap().plan(1202).unwrap();
        let prepared = BlockPlan::<Local, Nineteen>::new(exact).unwrap();
        assert_eq!(prepared.keys().key_count(), 38912);
        assert_eq!(prepared.keys().runs().last().unwrap().1.end, 38912);
    }

    #[test]
    fn joint_preparation_refuses_uncompiled_text_capacities() {
        let source = Capacity::new(DEFAULT_MAX_TEXT + 1)
            .unwrap()
            .plan(1)
            .unwrap();
        let error = BlockPlan::<Local, Twenty>::new(source).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
        assert!(error.to_string().contains("maximum text capacity 2048"));
    }
}
