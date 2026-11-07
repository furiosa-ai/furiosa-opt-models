//! The MLP's leaves at the 3072-wide intermediate, in a register-resident tiling: each
//! projection's weight loads into the TRF and the rows stream past it. The gate/up pair lands
//! each half's FP32 sums, [`SwiGlu`] takes SiLU of the FP32 gate and multiplies the rounded
//! halves, and the down projection adds the two clusters' BF16 partials.

use super::*;
use crate::ops::layout::Rows as Layout;
use crate::ops::linear::Affine;

mod down;

type Intermediate = super::Intermediate<3072>;

type Cluster = m![Intermediate / 1536];
type Gate<Hidden> = m![Intermediate % 1536 / 96, Hidden / 64];
type Down<Hidden> = m![Hidden / 64, Intermediate % 1536 / 96];
type Slices<Rows> = m![Intermediate % 1536 / 96, { Layout<Rows>::Lanes } # 16];

/// Each half's rows of FP32 sums, gate then up, as the projection reduces them.
pub type Sums<Rows> =
    DmTensor<f32, Chip, Cluster, Slices<Rows>, m![Term, { Groups<Rows> }, Intermediate % 96]>;

/// The BF16 product `silu(gate) * up` in the slices of the projection.
pub type Product<Rows> =
    DmTensor<bf16, Chip, Cluster, Slices<Rows>, m![1, { Groups<Rows> }, Intermediate % 96]>;

type GateWeight<Hidden> = TrfTensor<
    bf16,
    Chip,
    Cluster,
    Gate<Hidden>,
    m![Intermediate % 8],
    m![Intermediate % 96 / 8, Hidden % 64],
>;

/// Every row group's gate projection, then its up projection, each weight loaded before it.
impl<'d, 'x, Hidden: AxisName, Rows: AxisName>
    Module<(&'d mut Device, &'x Dm<Lanes<Rows>, Spread<Hidden, Rows>>)>
    for Affine<HbmTensor<bf16, Chip, m![Term, Intermediate, Hidden]>, ()>
{
    type Output = Dm<Lanes<Rows>, Sums<Rows>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Dm<Lanes<Rows>, Spread<Hidden, Rows>>),
    ) -> Self::Output {
        let weight = |device: &mut Device, term: usize| -> GateWeight<Hidden> {
            let weight: DmTensor<
                bf16,
                Chip,
                Cluster,
                Gate<Hidden>,
                m![Intermediate % 8, Intermediate % 96 / 8, Hidden % 64],
            > = self
                .weight
                .view()
                .tile::<m![Term], 1, m![Term = 1 # 2, Intermediate, Hidden]>(term)
                .to_dm(&mut device.tdma);
            device.sub.begin(weight.view())
                .fetch::<m![Intermediate % 8, Intermediate % 96 / 8, Hidden % 64 / 16], m![Hidden % 16]>()
                .collect::<m![Intermediate % 8, Intermediate % 96 / 8, Hidden % 64 / 16], m![Hidden % 16]>()
                .to_trf()
        };
        let mut sums: Sums<Rows> = DmTensor::new();
        let mut project = |device: &mut Device, term: usize, weight: &GateWeight<Hidden>| {
            // Both clusters retain a complete row; lanes occupy independent slice positions.
            let mut rows: DmTensor<
                bf16,
                Chip,
                Cluster,
                m![{ Layout<Rows>::Lanes } # 16, Hidden / 64],
                m![{ Groups<Rows> }, Hidden % 64],
            > = DmTensor::new();
            input
                .value
                .view()
                .to_dm_view(&mut device.tdma, rows.view_mut());
            device.main.begin(rows.view())
                .fetch::<m![{ Groups<Rows> }], m![Hidden % 64]>()
                .switch::<Gate<Hidden>, m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 16]>(
                    SwitchConfig::Broadcast1 { slice1: 16, slice0: 16 },
                )
                .collect::<m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 16, Hidden % 64 / 16], m![Hidden % 16]>()
                .contract_outer::<
                    m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 16, Hidden % 64 / 16, Intermediate % 96 / 8],
                    m![Hidden % 16], _, _, _,
                >(weight)
                .contract_packet::<m![1]>()
                .contract_time::<m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 16, Intermediate % 96 / 8]>()
                .contract_lane::<m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 16, Intermediate % 96 / 8],
                    m![Intermediate % 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<Slices<Rows>, m![{ Groups<Rows> }, Intermediate % 96 / 8]>(
                    InterSliceReduceOpF32::Add,
                )
                .vector_final()
                .commit_trim::<m![Intermediate % 8]>()
                .commit_view(sums.view_mut().tile::<m![Term], 1, m![
                    Term = 1 #{!} 2, { Groups<Rows> }, Intermediate % 96
                ]>(term));
        };
        let gate = weight(device, 0);
        project(device, 0, &gate);
        let up = weight(device, 1);
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
                Term = 1 # 2, { Groups<Rows> }, Intermediate % 96
            ]>(term)
        };
        let gate: Product<Rows> = device.main.begin(half(0))
            .fetch::<m![{ Groups<Rows> }, Intermediate % 96 / 8], m![Intermediate % 8]>()
            .collect::<m![{ Groups<Rows> }, Intermediate % 96 / 8], m![Intermediate % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Groups<Rows> }, Intermediate % 96 / 4], m![Intermediate % 4]>()
            .vector_stash()
            .vector_fp_unary(FpUnaryOp::Sigmoid)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
            .vector_widen_concat::<m![{ Groups<Rows> }, Intermediate % 96 / 8], m![Intermediate % 8]>()
            .vector_final()
            .cast::<bf16, m![Intermediate % 8 # 16]>()
            .commit_trim::<m![Intermediate % 8]>()
            .commit();
        let up: Product<Rows> = device
            .main
            .begin(half(1))
            .fetch::<m![{ Groups<Rows> }, Intermediate % 96 / 8], m![Intermediate % 8]>()
            .collect::<m![{ Groups<Rows> }, Intermediate % 96 / 8], m![Intermediate % 8]>()
            .cast::<bf16, m![Intermediate % 8 # 16]>()
            .commit_trim::<m![Intermediate % 8]>()
            .commit();
        let product = device.main.begin_interleaved::<Term, _, _, _, _, _>(gate.view(), up.view())
            .fetch::<m![{ Groups<Rows> }, Term, Intermediate % 96 / 8], m![Intermediate % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Groups<Rows> }, Term, Intermediate % 96 / 8], m![Intermediate % 8]>()
            .vector_init()
            .vector_zip::<
                Term,
                m![{ Groups<Rows> }, Intermediate % 96 / 8],
                m![{ Groups<Rows> }, Intermediate % 96 / 4],
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
