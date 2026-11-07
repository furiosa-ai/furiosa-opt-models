//! The down projection of the paired product: each cluster's 384-channel slices contracted to BF16
//! partials, the two clusters' partials added in FP32.

use super::*;

type DownWeight<Hidden> =
    DmTensor<bf16, Chip, Cluster, Down<Hidden>, m![Hidden % 64, Intermediate % 384]>;

type Activation<Hidden, Rows, Count> = TrfTensor<
    bf16,
    Chip,
    Cluster,
    Down<Hidden>,
    m![{ Layout<Rows>::Lanes }],
    m![Intermediate % 384 / 24, { Count }, { Packet }],
>;

type Partial<Hidden, Rows, Count> = DmTensor<
    bf16,
    Chip,
    Cluster,
    m![Hidden / 8],
    m![{ Count }, { Layout<Rows>::Lanes }, Hidden % 8],
>;

/// Each cluster's half of the intermediate contracted to BF16 partials, then their FP32 sum; every
/// row group when each holds a live row, otherwise the first group only.
impl<'d, 'x, Hidden: AxisName, Rows: AxisName>
    Module<(&'d mut Device, &'x Dm<Lanes<Rows>, Product<Rows>>)>
    for Affine<HbmTensor<bf16, Chip, m![Hidden, Intermediate]>, ()>
{
    type Output = Dm<Lanes<Rows>, Spread<Hidden, Rows>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Dm<Lanes<Rows>, Product<Rows>>),
    ) -> Self::Output {
        let weight: DownWeight<Hidden> = self.weight.to_dm(&mut device.tdma);
        let product: DmTensor<
            bf16,
            Chip,
            Cluster,
            Slices<Rows>,
            m![{ Layout<Rows>::Lanes }, { Packet }],
        > = device
            .sub
            .begin(input.value.view())
            .fetch::<m![{ Layout<Rows>::Lanes }], m![Intermediate % 24]>()
            .collect::<m![{ Layout<Rows>::Lanes }, { Packet } / 16], m![{ Packet } % 16]>()
            .commit_trim::<m![{ Packet } % 16]>()
            .commit();
        let mut sums: Partial<Hidden, Rows, Groups<Rows>> = DmTensor::new();
        if const { <Groups<Rows> as M>::SIZE == 2 } && input.tiles.count() == 2 {
            let activation = InputLanes::<Hidden, Rows, Groups<Rows>>(core::marker::PhantomData)
                .forward((&mut *device, product.view()));
            Contract::<Hidden, Rows, Groups<Rows>>(core::marker::PhantomData)
                .forward((&mut *device, (&weight, &activation)))
                .view()
                .to_dm_view(&mut device.tdma, sums.view_mut());
        } else {
            sums.view_mut()
                .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
            if 0 < input.tiles.count() {
                let activation = InputLanes::<Hidden, Rows, m![1 # 2]>(core::marker::PhantomData).forward((
                    &mut *device,
                    product.view().slice_tile::<
                        m![{ Groups<Rows> }], 1,
                        m![Intermediate % 3072 / 24, { Groups<Rows> } = 1 # 2],
                    >(0),
                ));
                Contract::<Hidden, Rows, m![1 # 2]>(core::marker::PhantomData)
                    .forward((&mut *device, (&weight, &activation)))
                    .view()
                    .to_dm_view(
                        &mut device.tdma,
                        Layout::<Rows>::selected_mut::<
                            _,
                            _,
                            _,
                            m![{ Layout<Rows>::Lanes }, Hidden % 8],
                        >(sums.view_mut(), 0),
                    );
            }
        }
        let exchanged = sums.view().cluster_swap().to_dm(&mut device.tdma);
        let mut result: Partial<Hidden, Rows, Groups<Rows>> = DmTensor::new();
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(sums.view(), exchanged.view())
            .fetch::<m![Term, { Groups<Rows> }, { Layout<Rows>::Lanes }, Hidden % 8 / 8], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Term, { Groups<Rows> }, { Layout<Rows>::Lanes }, Hidden % 8 / 8], m![Hidden % 8]>()
            .vector_init()
            .vector_zip::<
                Term,
                m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Hidden % 8 / 8],
                m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Hidden % 8 / 4],
                m![Hidden % 4],
            >(FpBinaryOp::AddF)
            .vector_final()
            .cast::<bf16, m![Hidden % 8 # 16]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit_view(result.view_mut());
        // SAFETY: exchange-add leaves identical sums in both clusters; only the cluster label changes.
        let result = unsafe {
            result.reshape::<
                Chip, crate::ops::normalization::Cluster, m![Hidden / 8],
                m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Hidden % 8],
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

struct InputLanes<Hidden, Rows, Count>(core::marker::PhantomData<(Hidden, Rows, Count)>);

impl<'i, Hidden: AxisName, Rows: AxisName, Count: M, Slice: M, Element: M>
    Module<(
        &mut Device,
        DmTensorView<'i, bf16, Chip, Cluster, Slice, Element>,
    )> for InputLanes<Hidden, Rows, Count>
{
    type Output = Activation<Hidden, Rows, Count>;

    /// `Count` row groups of the packed product, each slice's 384 channels lane by lane.
    fn forward(
        &self,
        (device, activated): (
            &mut Device,
            DmTensorView<'i, bf16, Chip, Cluster, Slice, Element>,
        ),
    ) -> Self::Output {
        device
        .sub
        .begin(activated)
        .fetch::<m![{ Layout<Rows>::Lanes }], m![Intermediate % 24]>()
        .switch::<Down<Hidden>, m![{ Layout<Rows>::Lanes }, Intermediate % 384 / 24, { Count }]>(
            SwitchConfig::TransposedBroadcast1 {
                slice1: 8,
                slice0: 32,
            },
        )
        .collect::<m![
            { Layout<Rows>::Lanes },
            Intermediate % 384 / 24,
            { Count },
            { Packet } / 16
        ], m![{ Packet } % 16]>()
        .to_trf()
    }
}

struct Contract<Hidden, Rows, Count>(core::marker::PhantomData<(Hidden, Rows, Count)>);

impl<Hidden: AxisName, Rows: AxisName, Count: M>
    Module<(
        &mut Device,
        (&DownWeight<Hidden>, &Activation<Hidden, Rows, Count>),
    )> for Contract<Hidden, Rows, Count>
{
    type Output = Partial<Hidden, Rows, Count>;

    /// One cluster's BF16 partial of the down projection of `Count` row groups.
    fn forward(
        &self,
        (device, (weight, activation)): (
            &mut Device,
            (&DownWeight<Hidden>, &Activation<Hidden, Rows, Count>),
        ),
    ) -> Self::Output {
        device.main.begin(weight.view())
        .fetch::<m![Hidden % 8 / 4, Hidden % 64 / 8, Intermediate % 384 / 8],
            m![Hidden % 4, Intermediate % 8]>()
        .collect::<m![Hidden % 8 / 4, Hidden % 64 / 8, Intermediate % 384 / 8,
            Hidden % 4 / 2], m![Hidden % 2, Intermediate % 8]>()
        .contract_outer::<m![Hidden % 8 / 4, Hidden % 64 / 8, Intermediate % 384 / 8, { Count }],
            m![Hidden % 4, Intermediate % 8], _, _, _,>(activation)
        .contract_packet::<m![Hidden % 4]>()
        .contract_time::<m![Hidden % 8 / 4, Hidden % 64 / 8, { Count }]>()
        .contract_lane::<m![Hidden % 8 / 4, Hidden % 64 / 8, { Count }, Hidden % 4],
            m![{ Layout<Rows>::Lanes } # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![Hidden / 8], m![Hidden % 8 / 4, { Count }, Hidden % 4]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
        .transpose::<m![Hidden % 8 / 4, { Count }, { Layout<Rows>::Lanes }], m![Hidden % 4 # 16]>()
        .commit_trim::<m![Hidden % 4]>()
        .commit()
    }
}
