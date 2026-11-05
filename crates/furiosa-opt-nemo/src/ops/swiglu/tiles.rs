use super::*;
use crate::ops::layout::Rows as Layout;

/// An intermediate width whose MLP streams its weights in 2048-wide intermediate tiles.
pub trait Tiled: AxisName {}

impl Tiled for super::Intermediate<12288> {}

impl Tiled for super::Intermediate<16384> {}

pub(super) type Tile<Cluster, Slices, Elements> = DmTensor<bf16, Chip, Cluster, Slices, Elements>;
pub(super) type Operand<Cluster, Slices, Lanes, Elements> =
    TrfTensor<bf16, Chip, Cluster, Slices, Lanes, Elements>;

type Cluster<I> = m![I / (I / 2)];
type Gate<I, Hidden> = m![I % 2048 / 32, Hidden / 1024];
type Down<I, Hidden> = m![Hidden / 32, I % 2048 / 1024];
type ProductSlices<I, Count> = m![I % 2048 / 16, { Count } # 2];
type Tiles<I> = m![I % (I / 2) / 2048];

type GateWeight<I, Hidden> = Tile<Cluster<I>, Gate<I, Hidden>, m![I % 32, Hidden % 1024]>;

type DownWeight<I, Hidden> = Tile<Cluster<I>, Down<I, Hidden>, m![Hidden % 32, I % 1024]>;

type Partial<I, Hidden, Rows, Count> =
    Tile<Cluster<I>, m![Hidden / 16], m![{ Count }, { Layout<Rows>::Lanes }, Hidden % 16]>;

type Product<I, Rows, Count = Groups<Rows>> =
    Tile<Cluster<I>, ProductSlices<I, Count>, m![{ Layout<Rows>::Lanes }, I % 16]>;

type Lanes<I, Hidden, Rows, Count = Groups<Rows>> =
    Operand<Cluster<I>, Gate<I, Hidden>, m![{ Layout<Rows>::Lanes }], m![{ Count }, Hidden % 1024]>;

type Activation<I, Hidden, Rows, Count> = Operand<
    Cluster<I>,
    Down<I, Hidden>,
    m![{ Layout<Rows>::Lanes }],
    m![I % 1024 / 16, { Count } # 2, I % 16],
>;

/// The rows one contraction carries, at most two lane groups.
type RowTile<Rows> = crate::ops::layout::axes::Tile<Rows, 16>;

/// One intermediate tile's up, gate and down weights.
type Weights<I, Hidden> = (
    GateWeight<I, Hidden>,
    GateWeight<I, Hidden>,
    DownWeight<I, Hidden>,
);

/// A launch's live rows through the MLP.
impl<'d, 'x, Hidden: AxisName, I: Tiled, Rows: AxisName>
    Module<(&'d mut Device, (&'x Spread<Hidden, Rows>, Live<Rows>))> for Mlp<Hidden, I>
where
    super::Down<I>: AxisName,
{
    type Output = Spread<Hidden, Rows>;

    fn forward(
        &self,
        (device, (hidden, live)): (&'d mut Device, (&'x Spread<Hidden, Rows>, Live<Rows>)),
    ) -> Self::Output {
        // One row tile streams every intermediate tile behind its own contraction; several row
        // tiles take each intermediate tile's weights once for all of them.
        if const { <m![Rows / ({ Symbol<RowTile<Rows>> })] as M>::SIZE == 1 } {
            if live.count() == <Groups<Rows> as M>::SIZE {
                return self.forward((device, hidden));
            }
            // Each live group alone, so a group past the live rows projects nothing.
            return self.each_group::<Rows>(device, hidden, live, |mlp, device, rows| {
                mlp.forward((device, rows))
            });
        }
        let mut sums: Partial<I, Hidden, Rows, Groups<Rows>> = DmTensor::new();
        sums.view_mut()
            .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
        // One row tile's rows through one intermediate tile's weights.
        let fill = |mlp: &Self,
                    device: &mut Device,
                    (up, gate, down): &Weights<I, Hidden>|
         -> Partial<I, Hidden, Rows, Groups<Rows>> {
            const { Self::check_rows::<RowTile<Rows>>() };
            let mut output: Partial<I, Hidden, Rows, Groups<Rows>> = DmTensor::new();
            #[unroll]
            for batch in 0..<m![Rows / ({ Symbol<RowTile<Rows>> })] as M>::SIZE {
                let lanes = mlp.lanes::<Rows, m![Rows % ({ Symbol<RowTile<Rows>> }) / 8], _>(
                    device,
                    hidden.view().tile::<
                        m![Rows / ({ Symbol<RowTile<Rows>> })], 1,
                        m![Rows / ({ Symbol<RowTile<Rows>> }) = 1 # (Rows / ({ Symbol<RowTile<Rows>> })), Rows % ({ Symbol<RowTile<Rows>> }), Hidden % 16],
                    >(batch),
                );
                let up = mlp.project::<Rows, m![Rows % ({ Symbol<RowTile<Rows>> }) / 8], false>(
                    device, &lanes, up,
                );
                let gate = mlp.project::<Rows, m![Rows % ({ Symbol<RowTile<Rows>> }) / 8], true>(
                    device, &lanes, gate,
                );
                let product = mlp.multiply::<Rows, m![Rows % ({ Symbol<RowTile<Rows>> }) / 8]>(
                    device, &gate, &up,
                );
                let activation = mlp
                    .broadcast::<Rows, m![Rows % ({ Symbol<RowTile<Rows>> }) / 8], _>(
                        device,
                        product.view(),
                    );
                mlp.contract::<Rows, m![Rows % ({ Symbol<RowTile<Rows>> }) / 8]>(device, down, &activation)
                    .view().to_dm_view(
                        &mut device.tdma,
                        output.view_mut().tile::<
                            m![Rows / ({ Symbol<RowTile<Rows>> })], 1,
                            m![Rows / ({ Symbol<RowTile<Rows>> }) = 1 #{!} (Rows / ({ Symbol<RowTile<Rows>> })), Rows % ({ Symbol<RowTile<Rows>> }), Hidden % 16],
                        >(batch),
                    );
            }
            output
        };
        // As `fill`, for the live row groups only.
        let tile = |mlp: &Self,
                    device: &mut Device,
                    (up, gate, down): &Weights<I, Hidden>|
         -> Partial<I, Hidden, Rows, Groups<Rows>> {
            const { Self::check_rows::<RowTile<Rows>>() };
            // Only the live groups' sums are read, so the groups past them stay unwritten.
            let mut output: Partial<I, Hidden, Rows, Groups<Rows>> = DmTensor::new();
            #[unroll]
            for batch in 0..<m![Rows / ({ Symbol<RowTile<Rows>> })] as M>::SIZE {
                if batch * <Groups<RowTile<Rows>> as M>::SIZE < live.count() {
                    let lanes = mlp.lanes::<Rows, m![Rows % ({ Symbol<RowTile<Rows>> }) / 8], _>(
                        device,
                        hidden.view().tile::<
                            m![Rows / ({ Symbol<RowTile<Rows>> })], 1,
                            m![Rows / ({ Symbol<RowTile<Rows>> }) = 1 # (Rows / ({ Symbol<RowTile<Rows>> })), Rows % ({ Symbol<RowTile<Rows>> }), Hidden % 16],
                        >(batch),
                    );
                    let up = mlp
                        .project::<Rows, m![Rows % ({ Symbol<RowTile<Rows>> }) / 8], false>(
                            device, &lanes, up,
                        );
                    let gate = mlp
                        .project::<Rows, m![Rows % ({ Symbol<RowTile<Rows>> }) / 8], true>(
                            device, &lanes, gate,
                        );
                    let product = mlp.multiply::<Rows, m![Rows % ({ Symbol<RowTile<Rows>> }) / 8]>(
                        device, &gate, &up,
                    );
                    if (batch + 1) * <Groups<RowTile<Rows>> as M>::SIZE <= live.count() {
                        let activation = mlp
                            .broadcast::<Rows, m![Rows % ({ Symbol<RowTile<Rows>> }) / 8], _>(
                                device,
                                product.view(),
                            );
                        mlp.contract::<Rows, m![Rows % ({ Symbol<RowTile<Rows>> }) / 8]>(device, down, &activation)
                            .view().to_dm_view(
                                &mut device.tdma,
                                output.view_mut().tile::<
                                    m![Rows / ({ Symbol<RowTile<Rows>> })], 1,
                                    m![Rows / ({ Symbol<RowTile<Rows>> }) = 1 #{!} (Rows / ({ Symbol<RowTile<Rows>> })), Rows % ({ Symbol<RowTile<Rows>> }), Hidden % 16],
                                >(batch),
                            );
                    } else {
                        let activation = mlp.broadcast::<Rows, m![1], _>(
                            device,
                            product.view().slice_tile::<
                                m![Rows % ({ Symbol<RowTile<Rows>> }) / 8], 1,
                                m![I % 2048 / 16, Rows % ({ Symbol<RowTile<Rows>> }) / 8 = 1 # 2],
                            >(0),
                        );
                        mlp.contract::<Rows, m![1]>(device, down, &activation)
                            .view()
                            .to_dm_view(
                                &mut device.tdma,
                                Layout::<Rows>::selected_mut::<
                                    _,
                                    _,
                                    _,
                                    m![{ Layout<Rows>::Lanes }, Hidden % 16],
                                >(
                                    output.view_mut(),
                                    batch * <Groups<RowTile<Rows>> as M>::SIZE,
                                ),
                            );
                    }
                }
            }
            output
        };
        // The sum of the first `live.count()` row groups into `output`; its later groups keep their
        // values, and the inputs' later groups are not read.
        let add_live =
            |_: &Self,
             device: &mut Device,
             (lhs, rhs): (
                &Partial<I, Hidden, Rows, Groups<Rows>>,
                &Partial<I, Hidden, Rows, Groups<Rows>>,
            ),
             output: &mut Partial<I, Hidden, Rows, Groups<Rows>>| {
                device
                .main
                .begin_interleaved::<Term, _, _, _, _, _>(lhs.view(), rhs.view())
                .fetch::<m![{ Groups<Rows> }, Term, { Layout<Rows>::Lanes }, Hidden % 16 / 8], m![Hidden % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Groups<Rows> }, Term, { Layout<Rows>::Lanes }, Hidden % 16 / 8], m![Hidden % 8]>()
                .vector_init()
                .vector_zip::<
                    Term,
                    m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Hidden % 16 / 8],
                    m![{ Groups<Rows> }, { Layout<Rows>::Lanes }, Hidden % 16 / 4],
                    m![Hidden % 4],
                >(FpBinaryOp::AddF)
                .vector_final()
                .cast::<bf16, m![Hidden % 8 # 16]>()
                .commit_trim::<m![Hidden % 8]>()
                .commit_prefix::<Groups<Rows>, _, _>(output.view_mut(), live.count());
            };
        let first = self.weights(device, 0);
        if <Groups<Rows> as M>::SIZE <= live.count() {
            self.reduce::<Rows>(
                device,
                first,
                &mut sums,
                fill,
                |mlp, device, (lhs, rhs), output| {
                    mlp.add::<Rows, Groups<Rows>, _>(device, lhs, rhs, output.view_mut())
                },
            );
        } else {
            // A group past the live rows is neither projected nor summed; its sum keeps `output`'s.
            self.reduce::<Rows>(device, first, &mut sums, tile, add_live);
        }

        self.finish::<Rows>(device, &sums)
    }
}

/// One row tile, every row group of it, through every intermediate tile.
impl<'d, 'x, Hidden: AxisName, I: Tiled, Rows: AxisName>
    Module<(&'d mut Device, &'x Spread<Hidden, Rows>)> for Mlp<Hidden, I>
where
    super::Down<I>: AxisName,
{
    type Output = Spread<Hidden, Rows>;

    fn forward(
        &self,
        (device, hidden): (&'d mut Device, &'x Spread<Hidden, Rows>),
    ) -> Self::Output {
        let mut sums: Partial<I, Hidden, Rows, Groups<Rows>> = DmTensor::new();
        let up = self.weight(device, 1, 0);
        let gate = self.weight(device, 0, 0);
        // One row tile through every intermediate tile, each tile's weights loaded ahead of the
        // contraction that needs them.
        {
            let (up, gate, output) = (&up, &gate, sums.view_mut());
            let lanes = self.lanes::<Rows, Groups<Rows>, _>(device, hidden.view());
            let up0 = self.project::<Rows, Groups<Rows>, false>(device, &lanes, up);
            let gate0 = self.project::<Rows, Groups<Rows>, true>(device, &lanes, gate);
            let gate1 = self.weight(device, 0, 1);
            let product0 = self.multiply::<Rows, Groups<Rows>>(device, &gate0, &up0);
            let activation0 = self.broadcast::<Rows, Groups<Rows>, _>(device, product0.view());
            let gate1 = self.project::<Rows, Groups<Rows>, true>(device, &lanes, &gate1);
            let up1 = self.weight(device, 1, 1);
            let up1 = self.project::<Rows, Groups<Rows>, false>(device, &lanes, &up1);
            let down0 = self.down_weight(device, 0);
            let product1 = self.multiply::<Rows, Groups<Rows>>(device, &gate1, &up1);
            let activation1 = self.broadcast::<Rows, Groups<Rows>, _>(device, product1.view());
            let down0 = self.contract::<Rows, Groups<Rows>>(device, &down0, &activation0);
            let gate2 = self.weight(device, 0, 2);
            let lanes = self.lanes::<Rows, Groups<Rows>, _>(device, hidden.view());
            let gate2 = self.project::<Rows, Groups<Rows>, true>(device, &lanes, &gate2);
            let up2 = self.weight(device, 1, 2);
            let up2 = self.project::<Rows, Groups<Rows>, false>(device, &lanes, &up2);
            let down1 = self.down_weight(device, 1);
            let product2 = self.multiply::<Rows, Groups<Rows>>(device, &gate2, &up2);
            let down1 = self.contract::<Rows, Groups<Rows>>(device, &down1, &activation1);
            let activation2 = self.broadcast::<Rows, Groups<Rows>, _>(device, product2.view());
            let down2 = self.down_weight(device, 2);
            let down2 = self.contract::<Rows, Groups<Rows>>(device, &down2, &activation2);
            let mut sums: Partial<I, Hidden, Rows, Groups<Rows>> = DmTensor::new();
            self.add::<Rows, Groups<Rows>, _>(device, &down0, &down1, sums.view_mut());
            if const { <Tiles<I> as M>::SIZE == 3 } {
                self.add::<Rows, Groups<Rows>, _>(device, &sums, &down2, output);
            } else {
                let gate = self.weight(device, 0, 3);
                let up = self.weight(device, 1, 3);
                let gate = self.project::<Rows, Groups<Rows>, true>(device, &lanes, &gate);
                let up = self.project::<Rows, Groups<Rows>, false>(device, &lanes, &up);
                let product = self.multiply::<Rows, Groups<Rows>>(device, &gate, &up);
                let activation = self.broadcast::<Rows, Groups<Rows>, _>(device, product.view());
                let down = self.down_weight(device, 3);
                let down = self.contract::<Rows, Groups<Rows>>(device, &down, &activation);
                let mut partial: Partial<I, Hidden, Rows, Groups<Rows>> = DmTensor::new();
                self.add::<Rows, Groups<Rows>, _>(device, &sums, &down2, partial.view_mut());
                self.add::<Rows, Groups<Rows>, _>(device, &partial, &down, output);
            }
        }
        self.finish::<Rows>(device, &sums)
    }
}

impl<Hidden: AxisName, I: Tiled> Mlp<Hidden, I>
where
    super::Down<I>: AxisName,
{
    /// Both clusters' partial sums added into each row's hidden output.
    fn finish<Rows: AxisName>(
        &self,
        device: &mut Device,
        sums: &Partial<I, Hidden, Rows, Groups<Rows>>,
    ) -> Spread<Hidden, Rows> {
        let exchanged = sums.view().cluster_swap().to_dm(&mut device.tdma);
        let mut result: Partial<I, Hidden, Rows, Groups<Rows>> = DmTensor::new();
        self.add::<Rows, Groups<Rows>, _>(device, sums, &exchanged, result.view_mut());
        // SAFETY: the cluster sum is complete in both clusters, with identical row placement.
        unsafe {
            result.reshape::<
            Chip, crate::ops::normalization::Cluster, crate::ops::normalization::FeatureSlices<Hidden>,
                crate::ops::normalization::Element<Hidden, Rows>,
            >()
        }
    }

    const fn check_rows<Rows: AxisName>() {
        assert!(<Groups<Rows> as M>::SIZE <= 2);
    }

    fn weights(&self, device: &mut Device, tile: usize) -> Weights<I, Hidden> {
        (
            self.weight(device, 1, tile),
            self.weight(device, 0, tile),
            self.down_weight(device, tile),
        )
    }

    /// Sums every intermediate tile's down projection in tile order with `add`. A tile's sum
    /// waits for the next tile's projection, so no add stands between two tiles' contractions.
    fn reduce<Rows: AxisName>(
        &self,
        device: &mut Device,
        first: Weights<I, Hidden>,
        output: &mut Partial<I, Hidden, Rows, Groups<Rows>>,
        rows: impl Fn(&Self, &mut Device, &Weights<I, Hidden>) -> Partial<I, Hidden, Rows, Groups<Rows>>,
        add: impl Fn(
            &Self,
            &mut Device,
            (
                &Partial<I, Hidden, Rows, Groups<Rows>>,
                &Partial<I, Hidden, Rows, Groups<Rows>>,
            ),
            &mut Partial<I, Hidden, Rows, Groups<Rows>>,
        ),
    ) {
        let mut sum = rows(self, device, &first);
        let second = self.weights(device, 1);
        let mut next = rows(self, device, &second);
        #[unroll]
        for tile in 0..<Tiles<I> as M>::SIZE - 2 {
            let weights = self.weights(device, tile + 2);
            let partial = rows(self, device, &weights);
            let mut total: Partial<I, Hidden, Rows, Groups<Rows>> = DmTensor::new();
            add(self, device, (&sum, &next), &mut total);
            sum = total;
            next = partial;
        }
        add(self, device, (&sum, &next), output);
    }

    fn lanes<Rows: AxisName, Count: M, Element: M>(
        &self,
        device: &mut Device,
        hidden: DmTensorView<
            '_,
            bf16,
            Chip,
            crate::ops::normalization::Cluster,
            crate::ops::normalization::FeatureSlices<Hidden>,
            Element,
        >,
    ) -> Lanes<I, Hidden, Rows, Count> {
        // SAFETY: both clusters hold the same hidden rows; only the cluster axis is renamed.
        let rows = unsafe { hidden.reshape::<Chip, Cluster<I>, m![Hidden / 16], Element>() };
        device
            .sub
            .begin(rows)
            .fetch::<m![{ Layout<Rows>::Lanes }, { Count }], m![Hidden % 16]>()
            .switch::<Gate<I, Hidden>, m![{ Layout<Rows>::Lanes }, { Count }, Hidden % 1024 / 16]>(
                SwitchConfig::TransposedBroadcast1 {
                    slice1: 4,
                    slice0: 64,
                },
            )
            .collect::<m![{ Layout<Rows>::Lanes }, { Count }, Hidden % 1024 / 16], m![Hidden % 16]>(
            )
            .to_trf()
    }

    fn weight(&self, device: &mut Device, term: usize, tile: usize) -> GateWeight<I, Hidden> {
        // SAFETY: loading places the whole gate matrix before the whole up matrix.
        let weight = unsafe {
            self.gate_up
                .tensor()
                .view()
                .reshape::<Chip, m![Term, I, Hidden]>()
        };
        weight
            .tile::<m![Term], 1, m![Term = 1 # 2, I, Hidden]>(term)
            .tile::<Tiles<I>, 1, m![
                Term = 1 # 2,
                I / (I / 2),
                I % (I / 2) / 2048 = 1 # (I / 4096),
                I % 2048,
                Hidden
            ]>(tile)
            .to_dm(&mut device.tdma)
    }

    fn down_weight(&self, device: &mut Device, tile: usize) -> DownWeight<I, Hidden> {
        self.down.tensor().view().tile::<
            Tiles<I>, 1,
            m![Hidden, I / (I / 2), I % (I / 2) / 2048 = 1 # (I / 4096), I % 2048],
        >(tile).to_dm(&mut device.tdma)
    }

    /// Up, or with `ACTIVATE` the SiLU-activated gate, for `Count` row groups.
    fn project<Rows: AxisName, Count: M, const ACTIVATE: bool>(
        &self,
        device: &mut Device,
        lanes: &Lanes<I, Hidden, Rows, Count>,
        weight: &GateWeight<I, Hidden>,
    ) -> Product<I, Rows, Count> {
        let values = device
            .main
            .begin(weight.view())
            .fetch::<
                m![I % 16, Hidden % 1024 / 16],
                m![I % 32 / 16, Hidden % 16],
            >()
            .collect::<
                m![I % 16, Hidden % 1024 / 16, I % 32 / 16],
                m![Hidden % 16],
            >()
            .contract_outer::<
                m![I % 16, Hidden % 1024 / 16, { Count }],
                m![I % 32 / 16, Hidden % 16], _, _, _,
            >(lanes)
            .contract_packet::<m![I % 32 / 16]>()
            .contract_time::<m![I % 16, { Count }]>()
            .contract_lane::<
                m![I % 16, { Count }, I % 32 / 16],
                m![{ Layout<Rows>::Lanes } # 8],
            >(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<ProductSlices<I, Count>, m![I % 16]>(
                InterSliceReduceOpF32::Add,
            );
        if ACTIVATE {
            values
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<
                    m![I % 16, { Layout<Rows>::Lanes } / 4],
                    m![{ Layout<Rows>::Lanes } % 4 # 4],
                >()
                .vector_stash()
                .vector_fp_unary(FpUnaryOp::Sigmoid)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                .vector_widen_concat::<m![I % 16], m![{ Layout<Rows>::Lanes } # 8]>()
                .vector_final()
                .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
                .transpose::<
                    m![I % 16 / 4, { Layout<Rows>::Lanes }],
                    m![I % 4 # 16],
                >()
                .commit_trim::<m![I % 4]>()
                .commit()
        } else {
            values
                .vector_final()
                .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
                .transpose::<m![I % 16 / 4, { Layout<Rows>::Lanes }], m![I % 4 # 16]>()
                .commit_trim::<m![I % 4]>()
                .commit()
        }
    }

    fn multiply<Rows: AxisName, Count: M>(
        &self,
        device: &mut Device,
        gate: &Product<I, Rows, Count>,
        up: &Product<I, Rows, Count>,
    ) -> Product<I, Rows, Count> {
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(gate.view(), up.view())
            .fetch::<m![Term, { Layout<Rows>::Lanes }, I % 16 / 8], m![I % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Term, { Layout<Rows>::Lanes }, I % 16 / 8], m![I % 8]>()
            .vector_init()
            .vector_zip::<
                Term,
                m![{ Layout<Rows>::Lanes }, I % 16 / 8],
                m![{ Layout<Rows>::Lanes }, I % 16 / 4],
                m![I % 4],
            >(FpBinaryOp::MulF(FpMulAlu::Mul0))
            .vector_final()
            .cast::<bf16, m![I % 8 # 16]>()
            .commit_trim::<m![I % 8]>()
            .commit()
    }

    fn broadcast<Rows: AxisName, Count: M, Slice: M>(
        &self,
        device: &mut Device,
        activated: DmTensorView<
            '_,
            bf16,
            Chip,
            Cluster<I>,
            Slice,
            m![{ Layout<Rows>::Lanes }, I % 16],
        >,
    ) -> Activation<I, Hidden, Rows, Count> {
        device
            .sub
            .begin(activated)
            .fetch::<m![{ Layout<Rows>::Lanes }], m![I % 16]>()
            .switch::<Down<I, Hidden>, m![{ Layout<Rows>::Lanes }, I % 1024 / 16, { Count } # 2]>(
                SwitchConfig::TransposedBroadcast1 {
                    slice1: 2,
                    slice0: 128,
                },
            )
            .collect::<m![{ Layout<Rows>::Lanes }, I % 1024 / 16, { Count } # 2], m![I % 16]>()
            .to_trf()
    }

    fn contract<Rows: AxisName, Count: M>(
        &self,
        device: &mut Device,
        weight: &DownWeight<I, Hidden>,
        activation: &Activation<I, Hidden, Rows, Count>,
    ) -> Partial<I, Hidden, Rows, Count> {
        device
            .main
            .begin(weight.view())
            .fetch::<
                m![Hidden % 16, I % 1024 / 16],
                m![Hidden % 32 / 16, I % 16],
            >()
            .collect::<
                m![Hidden % 16, I % 1024 / 16, Hidden % 32 / 16],
                m![I % 16],
            >()
            .contract_outer::<
                m![Hidden % 16, I % 1024 / 16, { Count }],
                m![Hidden % 32 / 16, I % 16], _, _, _,
            >(activation)
            .contract_packet::<m![Hidden % 32 / 16]>()
            .contract_time::<m![Hidden % 16, { Count }]>()
            .contract_lane::<m![Hidden % 16, { Count }, Hidden % 32 / 16], m![{ Layout<Rows>::Lanes } # 8]>(
                LaneMode::Interleaved,
            )
            .vector_init()
            .vector_inter_slice_reduce::<m![Hidden / 16], m![Hidden % 16, { Count }]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .cast::<bf16, m![{ Layout<Rows>::Lanes } # 16]>()
            .transpose::<m![Hidden % 16 / 4, { Count }, { Layout<Rows>::Lanes }], m![Hidden % 4 # 16]>()
            .commit_trim::<m![Hidden % 4]>()
            .commit()
    }

    fn add<Rows: AxisName, Count: M, Element: M>(
        &self,
        device: &mut Device,
        lhs: &Partial<I, Hidden, Rows, Count>,
        rhs: &Partial<I, Hidden, Rows, Count>,
        output: DmTensorViewMut<'_, bf16, Chip, Cluster<I>, m![Hidden / 16], Element>,
    ) {
        if const { <Count as M>::SIZE <= 2 } {
            device
                .main
                .begin_interleaved::<Term, _, _, _, _, _>(lhs.view(), rhs.view())
                .fetch::<m![Term, { Count }, { Layout<Rows>::Lanes }, Hidden % 16 / 8], m![Hidden % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Term, { Count }, { Layout<Rows>::Lanes }, Hidden % 16 / 8], m![Hidden % 8]>()
                .vector_init()
                .vector_zip::<
                    Term,
                    m![{ Count }, { Layout<Rows>::Lanes }, Hidden % 16 / 8],
                    m![{ Count }, { Layout<Rows>::Lanes }, Hidden % 16 / 4],
                    m![Hidden % 4],
                >(FpBinaryOp::AddF)
                .vector_final()
                .cast::<bf16, m![Hidden % 8 # 16]>()
                .commit_trim::<m![Hidden % 8]>()
                .commit_view(output);
        } else {
            device
                .main
                .begin_interleaved::<Term, _, _, _, _, _>(lhs.view(), rhs.view())
                .fetch::<m![{ Count }, Term, { Layout<Rows>::Lanes }, Hidden % 16 / 8], m![Hidden % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Count }, Term, { Layout<Rows>::Lanes }, Hidden % 16 / 8], m![Hidden % 8]>()
                .vector_init()
                .vector_zip::<
                    Term,
                    m![{ Count }, { Layout<Rows>::Lanes }, Hidden % 16 / 8],
                    m![{ Count }, { Layout<Rows>::Lanes }, Hidden % 16 / 4],
                    m![Hidden % 4],
                >(FpBinaryOp::AddF)
                .vector_final()
                .cast::<bf16, m![Hidden % 8 # 16]>()
                .commit_trim::<m![Hidden % 8]>()
                .commit_view(output);
        }
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_std::runtime::launch;

    use super::*;

    type Intermediate = super::super::Intermediate<12288>;
    type GateUp = crate::ops::swiglu::GateUp<Intermediate>;
    type Hidden = crate::ops::attention::geometry::Hidden<4096>;

    use crate::ops::layout::axes::{Eight, Sixteen};
    use crate::ops::weight::Weight;

    const COLUMNS: [usize; 6] = [0, 129, 1023, 2048, 3073, 4095];

    fn coefficient(term: usize, row: usize, column: usize) -> bf16 {
        bf16::from_f32(((row * 31 + column * 13 + term * 47) % 37) as f32 / 16.0 - 1.125)
    }

    fn channel(row: usize, part: usize) -> usize {
        part * 2048 + (row * 17 + part * 131) % 2048
    }

    #[device(chip = 1)]
    fn projected<Rows: AxisName>(
        device: &mut Device,
        mlp: &Mlp<Hidden, Intermediate>,
        rows: &crate::compose::Count,
        input: &HbmTensor<bf16, Chip, m![Rows, Hidden]>,
        output: &mut HbmTensor<bf16, Chip, m![Rows, Hidden]>,
    ) {
        let input: Spread<Hidden, Rows> = input.to_dm(&mut device.tdma);
        mlp.forward((&mut *device, (&input, Live::read(rows))))
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    async fn project<Rows: AxisName>(
        device: &mut Device,
        mlp: &Mlp<Hidden, Intermediate>,
        input: &[bf16],
    ) -> Vec<bf16> {
        project_live::<Rows>(device, mlp, input, Rows::SIZE).await
    }

    async fn project_live<Rows: AxisName>(
        device: &mut Device,
        mlp: &Mlp<Hidden, Intermediate>,
        input: &[bf16],
        rows: usize,
    ) -> Vec<bf16> {
        let live = crate::compose::tests::count(rows)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let input = HostTensor::<bf16, m![Rows, Hidden]>::from_vec(
            input[..Rows::SIZE * Hidden::SIZE].to_vec(),
        )
        .to_hbm::<Chip, m![Rows, Hidden]>(&mut device.pdma)
        .await
        .unwrap();
        let mut output = HostTensor::<bf16, m![Rows, Hidden]>::zero()
            .to_hbm::<Chip, m![Rows, Hidden]>(&mut device.pdma)
            .await
            .unwrap();
        launch(projected, (&mut *device, mlp, &live, &input, &mut output))
            .await
            .unwrap();
        output
            .to_host::<m![Rows, Hidden]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }

    /// Live row bounds zero skipped groups while computed rows match the staged BF16 reference.
    #[tokio::test(flavor = "current_thread")]
    async fn respects_live_groups() {
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let inputs = (0..Sixteen::SIZE)
            .flat_map(|row| {
                (0..Hidden::SIZE).map(move |column| {
                    bf16::from_f32(
                        ((row * 19 + column * 29 + row * column % 17) % 113) as f32 / 64.0 - 0.875,
                    )
                })
            })
            .collect::<Vec<_>>();
        let expected = inputs
            .chunks_exact(Hidden::SIZE)
            .flat_map(|row| {
                let activated = (0..Intermediate::SIZE)
                    .map(|channel| {
                        let term = |term| {
                            COLUMNS
                                .iter()
                                .enumerate()
                                .map(|(index, column)| {
                                    row[*column].to_f32()
                                        * coefficient(term, channel, index).to_f32()
                                })
                                .sum::<f32>()
                        };
                        let gate = term(0);
                        let gate = bf16::from_f32(gate / (1.0 + (-gate).exp()));
                        let up = bf16::from_f32(term(1));
                        bf16::from_f32(gate.to_f32() * up.to_f32())
                    })
                    .collect::<Vec<_>>();
                (0..Hidden::SIZE).map(move |row| {
                    let sum = |cluster: usize| {
                        let partial = |tile: usize| {
                            let part = cluster * 3 + tile;
                            bf16::from_f32(
                                activated[channel(row, part)].to_f32()
                                    * coefficient(2, row, part).to_f32(),
                            )
                            .to_f32()
                        };
                        let sum = bf16::from_f32(partial(0) + partial(1));
                        bf16::from_f32(sum.to_f32() + partial(2)).to_f32()
                    };
                    bf16::from_f32(sum(0) + sum(1)).to_f32()
                })
            })
            .collect::<Vec<_>>();
        let mut fused = vec![bf16::from_f32(0.0); GateUp::SIZE * Hidden::SIZE];
        for channel in 0..Intermediate::SIZE {
            for term in 0..2 {
                for (index, column) in COLUMNS.iter().enumerate() {
                    let row = term * Intermediate::SIZE + channel;
                    fused[row * Hidden::SIZE + column] = coefficient(term, channel, index);
                }
            }
        }
        let gate_up = Linear::new(Weight::new(
            HostTensor::<bf16, m![GateUp, Hidden]>::from_vec(fused)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        ));
        let mut down = vec![bf16::from_f32(0.0); Hidden::SIZE * Intermediate::SIZE];
        for row in 0..Hidden::SIZE {
            for part in 0..6 {
                down[row * Intermediate::SIZE + channel(row, part)] = coefficient(2, row, part);
            }
        }
        let mlp = Mlp::new(
            gate_up,
            Linear::new(Weight::new(
                HostTensor::<bf16, m![Hidden, Intermediate]>::from_vec(down)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            )),
        );
        let sixteen = project::<Sixteen>(&mut device, &mlp, &inputs).await;
        let partial = project_live::<Sixteen>(&mut device, &mlp, &inputs, Eight::SIZE).await;
        assert_eq!(
            &partial[..Eight::SIZE * Hidden::SIZE],
            &sixteen[..Eight::SIZE * Hidden::SIZE]
        );
        assert!(
            partial[Eight::SIZE * Hidden::SIZE..]
                .iter()
                .all(|value| value.to_f32() == 0.0)
        );
        let empty = project_live::<Sixteen>(&mut device, &mlp, &inputs, 0).await;
        assert!(empty.iter().all(|value| value.to_f32() == 0.0));
        for (index, (got, expected)) in sixteen.iter().zip(expected).enumerate() {
            assert!(
                (got.to_f32() - expected).abs() <= 0.03 * expected.abs().max(1.0),
                "row {} channel {}: {} vs {expected}",
                index / Hidden::SIZE,
                index % Hidden::SIZE,
                got.to_f32(),
            );
        }
    }
}
