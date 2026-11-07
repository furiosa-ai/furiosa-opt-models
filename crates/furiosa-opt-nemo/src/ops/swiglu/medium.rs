use super::tiles::{Operand, Tile};
use super::*;
use crate::ops::layout::Rows as Layout;

type Intermediate = super::Intermediate<9728, 10240>;

type Padded = super::Down<Intermediate>;

type Cluster = m![Intermediate / 4864];
type Gate<Hidden> = m![Intermediate % 4864 / 38, Hidden % 1280 / 640];
type ProductSlices = m![Intermediate % 4864 / 19];
type Down<Hidden> = m![Hidden % 1280 / 40, Padded % 5120 / 640];

type GateWeight<Hidden> = Tile<Cluster, Gate<Hidden>, m![Intermediate % 38, Hidden % 640]>;

type DownWeight<Hidden> = Tile<m![Hidden / 1280], Down<Hidden>, m![Hidden % 40, Padded % 640]>;

type Values<Rows, Channels, Cluster, Slice, Count> =
    Tile<Cluster, Slice, m![{ Channels }, { Count }, { Layout<Rows>::Lanes } # 8]>;

type Packet<Rows> = m![
    { Groups<Rows> } % ({ m![
        { Symbol<crate::ops::layout::axes::Tile<Rows, 16>> } / ({ Symbol<crate::ops::layout::axes::Lanes<Rows>> })
    ] }),
    { Layout<Rows>::Lanes } # 8
];

type Waves<Rows> = m![{ Groups<Rows> } / ({ m![
    { Symbol<crate::ops::layout::axes::Tile<Rows, 16>> } / ({ Symbol<crate::ops::layout::axes::Lanes<Rows>> })
] })];

type Product<Rows> = Values<Rows, m![Intermediate % 19], Cluster, ProductSlices, Groups<Rows>>;
type Reduced<Hidden> = m![Hidden % 1280 / 40, Hidden % 8];
type Shared<Hidden> = m![Hidden % 1280 / 40, 8];

type Partial<Hidden, Rows, Count> =
    Values<Rows, m![Hidden % 40 / 8], m![Hidden / 1280], Reduced<Hidden>, Count>;

type Output<Hidden, Rows, Count> =
    Tile<m![Hidden / 1280], Shared<Hidden>, m![{ Count }, { Layout<Rows>::Lanes }, Hidden % 40]>;

type Input<Hidden, Rows> = Tile<
    Cluster,
    Gate<Hidden>,
    m![Hidden / 1280, { Layout<Rows>::Lanes }, { Groups<Rows> }, Hidden % 640],
>;

type Lanes<Hidden, Rows> =
    Operand<Cluster, Gate<Hidden>, m![{ Layout<Rows>::Lanes }], m![{ Groups<Rows> }, Hidden % 640]>;

type Gathered<Hidden, Rows> = Tile<
    m![Hidden / 1280],
    m![Padded / 40],
    m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Padded % 40],
>;

type Outer<Rows> = m![Padded / 5120, { Layout<Rows>::Lanes }];
type Frame = m![Padded % 640 / 40, Padded % 40 # 48];

type Staging<Hidden, Rows> =
    Tile<m![Hidden / 1280], Down<Hidden>, m![{ Outer<Rows> }, { Groups<Rows> }, { Frame }]>;

type Activation<Hidden, Rows, Count> = Operand<
    m![Hidden / 1280],
    Down<Hidden>,
    m![{ Layout<Rows>::Lanes }],
    m![{ Count }, Padded % 640 / 40, Padded % 40 # 48],
>;

// The padded down projection uses 38-channel gate tiles and 640-channel staging packets.
impl<'d, 'x, 'c, Hidden: AxisName, Rows: AxisName>
    Module<(&'d mut Device, (&'x Spread<Hidden, Rows>, Live<Rows>))> for Mlp<Hidden, Intermediate>
{
    type Output = Spread<Hidden, Rows>;

    /// Every row group when each holds a live row; otherwise each live group alone.
    fn forward(
        &self,
        (device, (hidden, live)): (&'d mut Device, (&'x Spread<Hidden, Rows>, Live<Rows>)),
    ) -> Self::Output {
        if live.count() == <Groups<Rows> as M>::SIZE {
            return self.forward((device, hidden));
        }
        self.each_group::<Rows>(device, hidden, live, |mlp, device, rows| {
            mlp.forward((device, rows))
        })
    }
}

/// Every row group of the envelope.
impl<'d, 'x, Hidden: AxisName, Rows: AxisName> Module<(&'d mut Device, &'x Spread<Hidden, Rows>)>
    for Mlp<Hidden, Intermediate>
{
    type Output = Spread<Hidden, Rows>;

    fn forward(
        &self,
        (device, hidden): (&'d mut Device, &'x Spread<Hidden, Rows>),
    ) -> Self::Output {
        let weight = |device: &mut Device, term: usize, tile: usize| -> GateWeight<Hidden> {
            // SAFETY: loading places the whole gate matrix before the whole up matrix.
            let weight = unsafe {
                self.gate_up
                    .tensor()
                    .view()
                    .reshape::<Chip, m![Term, Intermediate, Hidden]>()
            };
            weight.tile::<m![Term], 1, m![Term = 1 # 2, Intermediate, Hidden]>(term)
                .tile::<m![Hidden / 1280], 1,
                    m![Term = 1 # 2, Intermediate, Hidden / 1280 = 1 # 2, Hidden % 1280]>(tile)
                .to_dm(&mut device.tdma)
        };
        let down_weight = |device: &mut Device, tile: usize| -> DownWeight<Hidden> {
            self.down
                .tensor()
                .view()
                .tile::<m![Padded / 5120], 1, m![Hidden, Padded / 5120 = 1 # 2, Padded % 5120]>(
                    tile,
                )
                .to_dm(&mut device.tdma)
        };
        let lanes = |device: &mut Device,
                     input: &Input<Hidden, Rows>,
                     tile: usize|
         -> Lanes<Hidden, Rows> {
            device.sub.begin(input.view().tile::<m![Hidden / 1280], 1,
                m![Hidden / 1280 = 1 # 2, { Layout<Rows>::Lanes }, { Groups<Rows> }, Hidden % 640]>(tile))
                .fetch::<m![{ Layout<Rows>::Lanes }, { Groups<Rows> }, Hidden % 640 / 16], m![Hidden % 16]>()
                .collect::<m![{ Layout<Rows>::Lanes }, { Groups<Rows> }, Hidden % 640 / 16], m![Hidden % 16]>()
                .to_trf()
        };
        let project = |device: &mut Device,
                       weight: &GateWeight<Hidden>,
                       lanes: &Lanes<Hidden, Rows>|
         -> Product<Rows> {
            let mut output = DmTensor::new();
            device.main.begin(weight.view())
                .fetch::<m![Intermediate % 19, Hidden % 640 / 16], m![Intermediate % 38 / 19, Hidden % 16]>()
                .collect::<m![Intermediate % 19, Hidden % 640 / 16, Intermediate % 38 / 19], m![Hidden % 16]>()
                .contract_outer::<
                    m![Intermediate % 19, Hidden % 640 / 16, { Groups<Rows> }],
                    m![Intermediate % 38 / 19, Hidden % 16], _, _, _,
                >(lanes)
                .contract_packet::<m![Intermediate % 38 / 19]>()
                .contract_time::<m![Intermediate % 19, { Groups<Rows> }]>()
                .contract_lane::<
                    m![Intermediate % 19, { Groups<Rows> }, Intermediate % 38 / 19],
                    m![{ Layout<Rows>::Lanes } # 8],
                >(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<ProductSlices, m![Intermediate % 19, { Groups<Rows> }]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
                .commit_trim::<m![{ Layout<Rows>::Lanes } # 8]>()
                .commit_view(output.view_mut());
            output
        };
        let activation = |device: &mut Device,
                          input: &Staging<Hidden, Rows>,
                          tile: usize|
         -> Activation<Hidden, Rows, Groups<Rows>> {
            device.sub.begin(input.view().tile::<m![Padded / 5120], 1,
                m![Padded / 5120 = 1 # 2, { Layout<Rows>::Lanes },
                   { Groups<Rows> },
                   Padded % 640 / 40, Padded % 40 # 48]>(tile))
                .fetch::<m![{ Layout<Rows>::Lanes }, { Groups<Rows> }, Padded % 640 / 40, Padded % 40 # 48 / 16],
                    m![Padded % 40 # 48 % 16]>()
                .collect::<m![{ Layout<Rows>::Lanes }, { Groups<Rows> }, Padded % 640 / 40, Padded % 40 # 48 / 16],
                    m![Padded % 40 # 48 % 16]>()
                .to_trf()
        };
        let contract = |device: &mut Device,
                        weight: &DownWeight<Hidden>,
                        activation: &Activation<Hidden, Rows, Groups<Rows>>|
         -> Partial<Hidden, Rows, Groups<Rows>> {
            let mut output = DmTensor::new();
            device.main.begin(weight.view())
                .fetch::<m![Hidden % 40 / 4, Padded % 640 / 8], m![Hidden % 4, Padded % 8]>()
                .collect::<m![Hidden % 40 / 4, Padded % 640 / 8, Hidden % 4 / 2], m![Hidden % 2, Padded % 8]>()
                .contract_outer::<m![Hidden % 40 / 4, Padded % 640 / 8, { Groups<Rows> }],
                    m![Hidden % 4, Padded % 8], _, _, _,
                >(activation)
                .contract_packet::<m![Hidden % 4]>()
                .contract_time::<m![Hidden % 40 / 4, { Groups<Rows> }]>()
                .contract_lane::<m![Hidden % 40 / 4, { Groups<Rows> }, Hidden % 4],
                    m![{ Layout<Rows>::Lanes } # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<Reduced<Hidden>, m![Hidden % 40 / 8, { Groups<Rows> }]>(
                    InterSliceReduceOpF32::Add,
                )
                .vector_final()
                .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
                .commit_trim::<m![{ Layout<Rows>::Lanes } # 8]>()
                .commit_view(output.view_mut());
            output
        };
        let gate0 = weight(device, 0, 0);
        let gate1 = weight(device, 0, 1);
        let input: Input<Hidden, Rows> = {
            // SAFETY: both clusters contain each complete row; only the cluster label changes.
            let hidden = unsafe {
                hidden
                    .view()
                    .reshape::<Chip, Cluster, m![Hidden / 16 # 256], m![Rows, Hidden % 16]>()
            };
            let mut output = DmTensor::new();
            device.sub.begin(hidden)
                .fetch::<m![{ Layout<Rows>::Lanes }, { Groups<Rows> }], m![Hidden % 16]>()
                .switch::<Gate<Hidden>,
                    m![{ Layout<Rows>::Lanes }, { Groups<Rows> }, Hidden / 1280, Hidden % 640 / 16]>(
                        SwitchConfig::CustomBroadcast { ring_size: 256 },
                    )
                .collect::<
                    m![{ Layout<Rows>::Lanes }, { Groups<Rows> }, Hidden / 1280, Hidden % 640 / 16],
                    m![Hidden % 16],
                >().commit_trim::<m![Hidden % 16]>().commit_view(output.view_mut());
            output
        };
        let lanes0 = lanes(device, &input, 0);
        let lanes1 = lanes(device, &input, 1);
        let gate0 = project(device, &gate0, &lanes0);
        let gate1 = project(device, &gate1, &lanes1);
        let gate = self.combine::<Rows, _, _, _, _>(device, &gate0, &gate1, FpBinaryOp::AddF);
        let gate: Product<Rows> = {
            let mut output = DmTensor::new();
            device.main.begin(gate.view())
                .fetch::<m![{ Groups<Rows> }, Intermediate % 19], m![{ Layout<Rows>::Lanes } # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Groups<Rows> }, Intermediate % 19], m![{ Layout<Rows>::Lanes } # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Groups<Rows> }, Intermediate % 19, { Layout<Rows>::Lanes } # 8 / 4],
                    m![{ Layout<Rows>::Lanes } # 8 % 4]>()
                .vector_stash()
                .vector_fp_unary(FpUnaryOp::Sigmoid)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                .vector_widen_concat::<m![{ Groups<Rows> }, Intermediate % 19], m![{ Layout<Rows>::Lanes } # 8]>()
                .vector_final()
                .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
                .commit_trim::<m![{ Layout<Rows>::Lanes } # 8]>()
                .commit_view(output.view_mut());
            output
        };
        let up1 = weight(device, 1, 1);
        let up0 = weight(device, 1, 0);
        let up1 = project(device, &up1, &lanes1);
        let up0 = project(device, &up0, &lanes0);
        let up = self.combine::<Rows, _, _, _, _>(device, &up0, &up1, FpBinaryOp::AddF);
        let product =
            self.combine::<Rows, _, _, _, _>(device, &gate, &up, FpBinaryOp::MulF(FpMulAlu::Mul0));
        let mut gathered: DmTensor<
            bf16,
            Chip,
            Cluster,
            m![Intermediate % 4864 / 152, 8],
            m![Intermediate % 152, { Groups<Rows> }, { Layout<Rows>::Lanes } # 8],
        > = DmTensor::new();
        // Each wave packs one 16-row tile's lane groups into a 16-lane packet.
        device.main.begin(product.view())
            .fetch::<m![{ Waves<Rows> }, Intermediate % 19], Packet<Rows>>()
            .switch::<m![Intermediate % 4864 / 152, 8],
                m![{ Waves<Rows> }, Intermediate % 19, Intermediate % 152 / 19]>(
                SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 },
            )
            .collect::<m![{ Waves<Rows> }, Intermediate % 19, Intermediate % 152 / 19], m![{ Packet<Rows> } # 16]>()
            .commit_trim::<Packet<Rows>>()
            .commit_view(gathered.view_mut());
        // Contiguous channels keep single-row DMA packets clear of the unused lane slots.
        let gathered: DmTensor<
            bf16,
            Chip,
            Cluster,
            m![Intermediate % 4864 / 152, 8],
            m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Intermediate % 152],
        > = device.sub.begin(gathered.view())
            .fetch::<m![{ Groups<Rows> }, Intermediate % 152], m![{ Layout<Rows>::Lanes } # 8]>()
            .collect::<m![{ Groups<Rows> }, Intermediate % 152], m![{ Layout<Rows>::Lanes } # 16]>()
            .transpose::<m![{ Groups<Rows> }, Intermediate % 152 / 4, { Layout<Rows>::Lanes }],
                m![Intermediate % 4 # 16]>()
            .commit_trim::<m![Intermediate % 4]>()
            .commit();
        let mut padded: Gathered<Hidden, Rows> = DmTensor::new();
        padded
            .view_mut()
            .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
        // SAFETY: the prefix changes only live coordinates in the same allocated slice/row cells.
        let prefix = unsafe {
            padded.view_mut().reshape::<
                Chip, m![Hidden / 1280], m![Intermediate #{!} 10240 / 40],
                m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Intermediate #{!} 10240 % 40],
            >()
        };
        gathered.view().to_dm_view(&mut device.tdma, prefix);
        let broadcast: Staging<Hidden, Rows> = {
            let input = &padded;
            let mut packed: DmTensor<
                bf16,
                Chip,
                m![Hidden / 1280],
                m![Padded / 40],
                m![{ Groups<Rows> }, { Layout<Rows>::Lanes } # 8, Padded % 40 # 48],
            > = DmTensor::new();
            packed
                .view_mut()
                .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
            device.sub.begin(input.view())
                .fetch::<m![{ Groups<Rows> }, { Layout<Rows>::Lanes }], m![Padded % 40]>()
                .collect::<m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Padded % 40 # 48 / 16],
                    m![Padded % 40 # 48 % 16]>()
                .commit_trim::<m![Padded % 40 # 48 % 16]>()
                .commit_view(packed.view_mut());
            let mut output = DmTensor::new();
            device.sub.begin(packed.view())
                .fetch::<m![{ Layout<Rows>::Lanes }, { Groups<Rows> }, Padded % 40 # 48 / 16],
                    m![Padded % 40 # 48 % 16]>()
                .switch::<Down<Hidden>, m![{ Layout<Rows>::Lanes }, { Groups<Rows> }, Padded % 40 # 48 / 16,
                    Padded / 5120, Padded % 640 / 40]>(
                        SwitchConfig::CustomBroadcast { ring_size: 256 },
                    )
                .collect::<m![{ Layout<Rows>::Lanes }, { Groups<Rows> }, Padded % 40 # 48 / 16,
                    Padded / 5120, Padded % 640 / 40], m![Padded % 40 # 48 % 16]>()
                .commit_trim::<m![Padded % 40 # 48 % 16]>()
                .commit_view(output.view_mut());
            output
        };
        let down0 = down_weight(device, 0);
        let down1 = down_weight(device, 1);
        let mut sums: Output<Hidden, Rows, Groups<Rows>> = DmTensor::new();
        sums.view_mut()
            .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
        let activation0 = activation(device, &broadcast, 0);
        let activation1 = activation(device, &broadcast, 1);
        let down0 = contract(device, &down0, &activation0);
        let down1 = contract(device, &down1, &activation1);
        let accumulated: Output<Hidden, Rows, Groups<Rows>> = {
            let combined = self.combine::<Rows, _, _, _, Groups<Rows>>(
                device,
                &down0,
                &down1,
                FpBinaryOp::AddF,
            );
            let mut gathered: Values<
                Rows,
                m![Hidden % 40],
                m![Hidden / 1280],
                Shared<Hidden>,
                Groups<Rows>,
            > = DmTensor::new();
            device
                .main
                .begin(combined.view())
                .fetch::<m![{ Groups<Rows> }, Hidden % 40 / 8], m![{ Layout<Rows>::Lanes } # 8]>()
                .switch::<Shared<Hidden>, m![{ Groups<Rows> }, Hidden % 40 / 8, Hidden % 8]>(
                    SwitchConfig::Broadcast1 {
                        slice1: 8,
                        slice0: 1,
                    },
                )
                .collect::<m![{ Groups<Rows> }, Hidden % 40 / 8, Hidden % 8], m![{ Layout<Rows>::Lanes } # 16]>()
                .commit_trim::<m![{ Layout<Rows>::Lanes } # 8]>()
                .commit_view(gathered.view_mut());
            let mut output = DmTensor::new();
            device
                .sub
                .begin(gathered.view())
                .fetch::<m![{ Groups<Rows> }, Hidden % 40], m![{ Layout<Rows>::Lanes } # 8]>()
                .collect::<m![{ Groups<Rows> }, Hidden % 40], m![{ Layout<Rows>::Lanes } # 16]>()
                .transpose::<m![{ Groups<Rows> }, Hidden % 40 / 4, { Layout<Rows>::Lanes }], m![Hidden % 4 # 16]>()
                .commit_trim::<m![Hidden % 4]>()
                .commit_view(output.view_mut());
            output
        };
        accumulated
            .view()
            .to_dm_view(&mut device.tdma, sums.view_mut());
        let mut output: Spread<Hidden, Rows> = DmTensor::new();
        sums.view().to_dm_view(&mut device.tdma, output.view_mut());
        output
    }
}

impl<Hidden: AxisName> Mlp<Hidden, Intermediate> {
    fn combine<Rows: AxisName, Channels: M, Chips: M, S: M, Count: M>(
        &self,
        device: &mut Device,
        lhs: &Values<Rows, Channels, Chips, S, Count>,
        rhs: &Values<Rows, Channels, Chips, S, Count>,
        operation: FpBinaryOp,
    ) -> Values<Rows, Channels, Chips, S, Count> {
        let mut output = DmTensor::new();
        device.main.begin_interleaved::<Term, _, _, _, _, _>(lhs.view(), rhs.view())
            .fetch::<m![{ Count }, Term, { Channels }], m![{ Layout<Rows>::Lanes } # 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Count }, Term, { Channels }], m![{ Layout<Rows>::Lanes } # 8]>()
            .vector_init()
            .vector_zip::<
                Term,
                m![{ Count }, { Channels }],
                m![{ Count }, { Channels }, { Layout<Rows>::Lanes } # 8 / 4],
                m![{ Layout<Rows>::Lanes } # 8 % 4],
            >(operation)
            .vector_final()
            .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
            .commit_trim::<m![{ Layout<Rows>::Lanes } # 8]>()
            .commit_view(output.view_mut());
        output
    }
}
