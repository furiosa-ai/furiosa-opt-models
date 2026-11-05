//! The MLP's leaves at the 6144-wide intermediate, in a paired tiling: the gate/up projection
//! pair lands each half's FP32 sums, [`SwiGlu`] takes SiLU of the FP32 gate and multiplies the
//! rounded halves, and the down projection adds the two clusters' BF16 partials.
//! - The down weight loads when its projection runs, after the product, so its transfer does not
//!   overlap the gate and up projections.

use super::*;
use crate::ops::layout::Rows as Layout;
use crate::ops::linear::Affine;

mod down;

type Intermediate = super::Intermediate<6144>;

type Cluster = m![Intermediate / 3072];
type PairedGate<Hidden> = m![Intermediate % 3072 / 192, Hidden / 128];
type Down<Hidden> = m![Hidden / 64, Intermediate % 3072 / 384];
type Slices<Rows> = m![Intermediate % 3072 / 24, { Groups<Rows> } # 2];
type Packet = m![Intermediate % 24 # 32];

/// Each half's rows of FP32 sums, gate then up, as the paired contraction reduces them.
pub type Sums<Rows> = DmTensor<
    f32,
    Chip,
    Cluster,
    Slices<Rows>,
    m![Term, Intermediate % 24, { Layout<Rows>::Lanes } # 8],
>;

/// The BF16 product `silu(gate) * up` in the slices of the paired contraction.
pub type Product<Rows> =
    DmTensor<bf16, Chip, Cluster, Slices<Rows>, m![1, { Layout<Rows>::Lanes }, Intermediate % 24]>;

type Weight<Hidden> =
    DmTensor<bf16, Chip, Cluster, PairedGate<Hidden>, m![Intermediate % 192, Hidden % 128]>;

/// The rows each slice of a projection pair reads, held in the TRF.
type Staged<Hidden, Rows> = TrfTensor<
    bf16,
    Chip,
    Cluster,
    PairedGate<Hidden>,
    m![{ Layout<Rows>::Lanes }],
    m![{ Groups<Rows> }, Hidden % 128],
>;

/// Every row group's gate and up projections, the weights loaded first.
impl<'d, 'x, Hidden: AxisName, Rows: AxisName>
    Module<(&'d mut Device, &'x Dm<Lanes<Rows>, Spread<Hidden, Rows>>)>
    for Affine<HbmTensor<bf16, Chip, m![Term, Intermediate, Hidden]>, ()>
{
    type Output = Dm<Lanes<Rows>, Sums<Rows>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Dm<Lanes<Rows>, Spread<Hidden, Rows>>),
    ) -> Self::Output {
        let half = |device: &mut Device, term: usize| -> Weight<Hidden> {
            self.weight
                .view()
                .tile::<m![Term], 1, m![Term = 1 # 2, Intermediate, Hidden]>(term)
                .to_dm(&mut device.tdma)
        };
        let gate = half(device, 0);
        let up = half(device, 1);
        // SAFETY: both clusters hold a complete row; the cluster label names the weight partition.
        let rows = unsafe {
            input
                .value
                .view()
                .reshape::<Chip, Cluster, m![Hidden / 16 # 256], m![Rows, Hidden % 16]>()
        };
        let lanes: Staged<Hidden, Rows> = device
            .sub
            .begin(rows)
            .fetch::<m![{ Layout<Rows>::Lanes }, { Groups<Rows> }], m![Hidden % 16]>()
            .switch::<PairedGate<Hidden>, m![
                { Layout<Rows>::Lanes }, { Groups<Rows> }, Hidden % 128 / 16,
            ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
            .collect::<m![{ Layout<Rows>::Lanes }, { Groups<Rows> }, Hidden % 128 / 16], m![Hidden % 16]>()
            .to_trf();
        let mut sums: Sums<Rows> = DmTensor::new();
        let mut project = |device: &mut Device, term: usize, weight: &Weight<Hidden>| {
            device.main.begin(weight.view())
                .fetch::<m![Intermediate % 24, Intermediate % 192 / 48, Hidden % 128 / 16],
                    m![Intermediate % 48 / 24, Hidden % 16]>()
                .collect::<m![Intermediate % 24, Intermediate % 192 / 48, Hidden % 128 / 16,
                    Intermediate % 48 / 24], m![Hidden % 16]>()
                .contract_outer::<m![Intermediate % 24, Intermediate % 192 / 48,
                    Hidden % 128 / 16, { Groups<Rows> }],
                    m![Intermediate % 48 / 24, Hidden % 16], _, _, _,>(&lanes)
                .contract_packet::<m![Intermediate % 48 / 24]>()
                .contract_time::<m![Intermediate % 24, Intermediate % 192 / 48, { Groups<Rows> }]>()
                .contract_lane::<m![Intermediate % 24, Intermediate % 192 / 48, { Groups<Rows> },
                    Intermediate % 48 / 24], m![{ Layout<Rows>::Lanes } # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<Slices<Rows>, m![Intermediate % 24]>(
                    InterSliceReduceOpF32::Add,
                )
                .vector_final()
                .commit_trim::<m![{ Layout<Rows>::Lanes } # 8]>()
                .commit_view(sums.view_mut().tile::<m![Term], 1, m![
                    Term = 1 #{!} 2, Intermediate % 24, { Layout<Rows>::Lanes } # 8
                ]>(term));
        };
        project(device, 0, &gate);
        project(device, 1, &up);
        Dm {
            value: sums,
            tiles: input.tiles,
        }
    }
}

/// SiLU of each FP32 gate sum, and each up sum, round to BF16 before they multiply.
impl<'d, 'x, Rows: AxisName> Module<(&'d mut Device, &'x Dm<Lanes<Rows>, Sums<Rows>>)> for SwiGlu {
    type Output = Dm<Lanes<Rows>, Product<Rows>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Dm<Lanes<Rows>, Sums<Rows>>),
    ) -> Self::Output {
        let half = |term: usize| {
            input.value.view().tile::<m![Term], 1, m![
                Term = 1 # 2, Intermediate % 24, { Layout<Rows>::Lanes } # 8
            ]>(term)
        };
        let gate: Product<Rows> = device.main.begin(half(0))
            .fetch::<m![Intermediate % 24], m![{ Layout<Rows>::Lanes } # 8]>()
            .collect::<m![Intermediate % 24], m![{ Layout<Rows>::Lanes } # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Intermediate % 24, { Layout<Rows>::Lanes } / 4],
                m![{ Layout<Rows>::Lanes } % 4 # 4]>()
            .vector_stash()
            .vector_fp_unary(FpUnaryOp::Sigmoid)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
            .vector_widen_concat::<m![Intermediate % 24], m![{ Layout<Rows>::Lanes } # 8]>()
            .vector_final()
            .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
            .transpose::<m![Intermediate % 24 / 4, { Layout<Rows>::Lanes }], m![Intermediate % 4 # 16]>()
            .commit_trim::<m![Intermediate % 4]>()
            .commit();
        let up: Product<Rows> = device.main.begin(half(1))
            .fetch::<m![Intermediate % 24], m![{ Layout<Rows>::Lanes } # 8]>()
            .collect::<m![Intermediate % 24], m![{ Layout<Rows>::Lanes } # 8]>()
            .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
            .transpose::<m![Intermediate % 24 / 4, { Layout<Rows>::Lanes }], m![Intermediate % 4 # 16]>()
            .commit_trim::<m![Intermediate % 4]>()
            .commit();
        let product = device.main
            .begin_interleaved::<Term, _, _, _, _, _>(gate.view(), up.view())
            .fetch::<m![Term, { Layout<Rows>::Lanes }, Intermediate % 24 / 8], m![Intermediate % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Term, { Layout<Rows>::Lanes }, Intermediate % 24 / 8], m![Intermediate % 8]>()
            .vector_init()
            .vector_zip::<
                Term,
                m![{ Layout<Rows>::Lanes }, Intermediate % 24 / 8],
                m![{ Layout<Rows>::Lanes }, Intermediate % 24 / 4],
                m![Intermediate % 4],
            >(FpBinaryOp::MulF(FpMulAlu::Mul0))
            .vector_final()
            .cast::<bf16, m![Intermediate % 8 # 16]>()
            .commit_trim::<m![Intermediate % 8]>()
            .commit();
        Dm {
            value: product,
            tiles: input.tiles,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::attention::geometry::Hidden;

    /// The leaves at the 2048-wide hidden rows they serve match the oracle.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_the_oracle_on_live_rows() {
        super::super::tests::leaves_match_the_oracle::<Hidden<2048>, super::Intermediate>().await;
    }
}
