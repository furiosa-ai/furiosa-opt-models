//! The down projection of the resident product: the product transposed across the slices and
//! contracted against the TRF weight, the two clusters' BF16 partials added in FP32.

use super::*;

type DownWeight<Hidden> = TrfTensor<
    bf16,
    Chip,
    Cluster,
    Down<Hidden>,
    m![Hidden % 8],
    m![Hidden % 64 / 8, Intermediate % 96],
>;

type Partial<Hidden, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster,
    m![Hidden / 64, 1 # 16],
    m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Hidden % 64],
>;

/// Each cluster's half of the intermediate contracted to BF16 partials, then their FP32 sum.
impl<'d, 'x, Hidden: AxisName, Rows: AxisName>
    Module<(&'d mut Device, &'x Dm<Lanes<Rows>, Product<Rows>>)>
    for Affine<HbmTensor<bf16, Chip, m![Hidden, Intermediate]>, ()>
{
    type Output = Dm<Lanes<Rows>, Spread<Hidden, Rows>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Dm<Lanes<Rows>, Product<Rows>>),
    ) -> Self::Output {
        let weight: DmTensor<
            bf16,
            Chip,
            Cluster,
            Down<Hidden>,
            m![Hidden % 8, Hidden % 64 / 8, Intermediate % 96],
        > = self.weight.to_dm(&mut device.tdma);
        let weight: DownWeight<Hidden> = device.sub.begin(weight.view())
            .fetch::<m![Hidden % 8, Hidden % 64 / 8, Intermediate % 96 / 16], m![Intermediate % 16]>()
            .collect::<m![Hidden % 8, Hidden % 64 / 8, Intermediate % 96 / 16], m![Intermediate % 16]>()
            .to_trf();
        let mut sums: Partial<Hidden, Rows> = DmTensor::new();
        let transposed: DmTensor<
            bf16,
            Chip,
            Cluster,
            m![{ Layout<Rows>::Lanes } # 16, Intermediate % 1536 / 96],
            m![{ Groups<Rows> }, Intermediate % 96],
        > = device
            .main
            .begin(input.value.view())
            .fetch::<m![{ Groups<Rows> }], m![Intermediate % 96]>()
            .switch::<m![{ Layout<Rows>::Lanes } # 16, Intermediate % 1536 / 96], m![{ Groups<Rows> }]>(
                SwitchConfig::Transpose {
                    slice1: 16,
                    slice0: 16,
                },
            )
            .collect::<m![{ Groups<Rows> }, Intermediate % 96 / 16], m![Intermediate % 16]>()
            .commit_trim::<m![Intermediate % 16]>()
            .commit();
        let projected: DmTensor<
            bf16,
            Chip,
            Cluster,
            m![Hidden / 64, { Layout<Rows>::Lanes } # 16],
            m![{ Groups<Rows> }, Hidden % 64],
        > = device.main.begin(transposed.view())
            .fetch::<m![{ Groups<Rows> }], m![Intermediate % 96]>()
            .switch::<Down<Hidden>, m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 16]>(
                SwitchConfig::Broadcast1 { slice1: 16, slice0: 16 },
            )
            .collect::<m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 16, Intermediate % 96 / 16],
                m![Intermediate % 16]>()
            .contract_outer::<
                m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 16, Intermediate % 96 / 16, Hidden % 64 / 8],
                m![Intermediate % 16], _, _, _,
            >(&weight)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 16, Hidden % 64 / 8]>()
            .contract_lane::<m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 16, Hidden % 64 / 8], m![Hidden % 8]>(
                LaneMode::Interleaved,
            )
            .vector_init()
            .vector_inter_slice_reduce::<m![Hidden / 64, { Layout<Rows>::Lanes } # 16], m![{ Groups<Rows> }, Hidden % 64 / 8]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .cast::<bf16, m![Hidden % 8 # 16]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit();
        projected
            .view()
            .to_dm_view(&mut device.tdma, sums.view_mut());
        let exchanged = sums.view().cluster_swap().to_dm(&mut device.tdma);
        let result: Partial<Hidden, Rows> = device.main.begin_interleaved::<Term, _, _, _, _, _>(
                sums.view(), exchanged.view(),
            )
            .fetch::<m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Term, Hidden % 64 / 8], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Term, Hidden % 64 / 8], m![Hidden % 8]>()
            .vector_init()
            .vector_zip::<
                Term,
                m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Hidden % 64 / 8],
                m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Hidden % 64 / 4],
                m![Hidden % 4],
            >(FpBinaryOp::AddF)
            .vector_final()
            .cast::<bf16, m![Hidden % 8 # 16]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit();
        // SAFETY: exchange-add leaves identical sums in both clusters; only the cluster label changes.
        let result = unsafe {
            result.reshape::<
                Chip, crate::ops::normalization::Cluster, m![Hidden / 64, 1 # 16],
                m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Hidden % 64],
            >()
        };
        let mut output: Spread<Hidden, Rows> = DmTensor::new();
        result
            .view()
            .to_dm_view(&mut device.tdma, output.view_mut());
        Dm {
            value: output,
            tiles: input.tiles,
        }
    }
}
