use super::{geometry, gqa};
use crate::Op;
use crate::ops::linear::{self, Groups, Linear};
use crate::ops::normalization::{self, Spread};
use crate::ops::vector::Zip;
use crate::tensor::axis;
use crate::tensor::form::BlockScaled;
use crate::tensor::form::nvfp4::{Held, Late, Scale, Scales};
use crate::tensor::memory::Memory;
use crate::tensor::{axis::Lanes, live::Live, mapping::Rows as Layout, parallel::Chip};
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;

type Hidden = geometry::Hidden<4096>;
type QWidth = geometry::QueryWidth<4096>;

axes![Term = 2];

/// Rows one contraction pass holds: the switched input pads its row groups to two.
type Tile<Rows> = axis::Tile<Rows, 16>;

type Cluster = m![QWidth / 2048];
type ProjectionSlices = m![Hidden / 32, QWidth % 2048 / 1024];
type Reduced = m![Hidden / 16];
type Weight = DmTensor<bf16, Chip, Cluster, ProjectionSlices, m![Hidden % 32, QWidth % 1024]>;

/// Rows of the staged input that share one slice: one lane group of every tile.
#[derive(Debug, Clone)]
struct Part<Rows>(PhantomData<Rows>);

impl<Rows: AxisName> AxisName for Part<Rows> {
    const NAME: Ident = Ident::new("Part");
    const SIZE: usize = Rows::SIZE / <Groups<Tile<Rows>> as M>::SIZE;
}

#[derive(Debug, Clone)]
struct Lane<Rows>(PhantomData<Rows>);

impl<Rows: AxisName> AxisName for Lane<Rows> {
    const NAME: Ident = Ident::new("Lane");
    const SIZE: usize = <Layout<Tile<Rows>>::Lanes as M>::SIZE;
}

type Staging<Rows> = m![Hidden / 2048, QWidth % 2048 / 32, Rows / ({ Symbol<Part<Rows>> }) # 2];

type Input<Rows> =
    DmTensor<bf16, Chip, Cluster, Staging<Rows>, m![Rows % ({ Symbol<Part<Rows>> }), QWidth % 32]>;

type Output<Rows> = DmTensor<bf16, Chip, Cluster, Reduced, m![Rows, Hidden % 16]>;

/// Which tile a lane group of the staged input belongs to: a slice holds one of every tile.
type Tiles<Rows> = m![Rows % ({ Symbol<Part<Rows>> }) / ({ Symbol<Lane<Rows>> })];

type Read<'x, Width, Rows, KvHeads> = DmTensorView<
    'x,
    bf16,
    Chip,
    gqa::Cluster<Rows>,
    m![Width / (Width / KvHeads), { Layout<Rows>::Lanes } # 8 % 4, 1 # 8],
    m![{ linear::Groups<Rows> }, Width % (Width / KvHeads)],
>;

/// The output projection of every live row's attention, each row read as one row of `QWidth`:
/// its heads' values in head order.
impl<
    'x,
    Hidden: AxisName,
    QWidth: AxisName,
    Rows: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Matrix,
> Op<&'x Live<gqa::Attended<Rows, KvHeads, Groups, Head>, Rows>> for linear::Affine<Matrix, ()>
where
    Matrix: linear::Matrix<Chip, In = QWidth, Out = Hidden>
        + Borrow<HbmTensor<bf16, Chip, m![Hidden, QWidth]>>,
{
    type Output = Live<Spread<Hidden, Rows>, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<gqa::Attended<Rows, KvHeads, Groups, Head>, Rows>,
    ) -> Self::Output {
        let (live, rows) = read::<QWidth, _, _, _, _>(input);
        // This output kernel uses a shape-specific contraction and cluster reduction.
        if const { Hidden::SIZE == 4096 && QWidth::SIZE == 4096 } {
            type Width = geometry::Hidden<4096>;
            type InputWidth = geometry::QueryWidth<4096>;

            // SAFETY: the selected dimensions keep the original row-major weight coordinates.
            let weight = unsafe {
                self.weight
                    .borrow()
                    .view()
                    .reshape::<Chip, m![Width, InputWidth]>()
            };
            let rows = unsafe {
                rows.reshape::<
                    Chip, gqa::Cluster<Rows>,
                    m![InputWidth / (QWidth / KvHeads), { Layout<Rows>::Lanes } # 8 % 4, 1 # 8],
                m![{ linear::Groups<Rows> }, InputWidth % (QWidth / KvHeads)],
                >()
            };
            // Every live row through the projection; a tile holding no live row is not contracted.
            let result: Spread<Width, Rows> = {
                let input = rows;
                // Contracts tile `tile` of the staged rows with the weight into its cells of
                // `partial`.
                let contract = |device: &mut Device,
                                weight: &Weight,
                                rows: &Input<Rows>,
                                partial: &mut Output<Rows>,
                                tile: usize| {
                    let input = rows.view().tile::<Tiles<Rows>, 1, m![
                        { Tiles<Rows> } = 1 # ({ Tiles<Rows> }),
                        Rows % ({ Symbol<Lane<Rows>> }), InputWidth % 32,
                    ]>(tile);
                    // SAFETY: spatial row groups and the selected lane prefix keep their physical
                    // cells.
                    let input = unsafe {
                        input.reshape::<Chip, Cluster, m![
                            Width / 2048,
                            InputWidth % 2048 / 32,
                            { linear::Groups<Tile<Rows>> } # 2
                        ], m![
                            { Layout<Tile<Rows>>::Lanes } # ({ Symbol<Part<Rows>> }),
                            InputWidth % 32
                        ]>()
                    }
                    .unpad::<m![{ Layout<Tile<Rows>>::Lanes }, InputWidth % 32]>();
                    let lanes: TrfTensor<
                        bf16,
                        Chip,
                        Cluster,
                        ProjectionSlices,
                        m![{ Layout<Tile<Rows>>::Lanes }],
                        m![
                            InputWidth % 1024 / 32,
                            { linear::Groups<Tile<Rows>> } # 2,
                            InputWidth % 32
                        ],
                    > = device
                        .sub
                        .begin(input)
                        .fetch::<m![{ Layout<Tile<Rows>>::Lanes }], m![InputWidth % 32]>()
                        .switch::<ProjectionSlices, m![
                            { Layout<Tile<Rows>>::Lanes },
                            InputWidth % 1024 / 32,
                            { linear::Groups<Tile<Rows>> } # 2
                        ]>(SwitchConfig::TransposedBroadcast1 {
                            slice1: 2,
                            slice0: 64,
                        })
                        .collect::<m![
                            { Layout<Tile<Rows>>::Lanes }, InputWidth % 1024 / 32,
                            { linear::Groups<Tile<Rows>> } # 2,
                            InputWidth % 32 / 16,
                        ], m![InputWidth % 16]>()
                        .to_trf();
                    let output = partial.view_mut().tile::<Tiles<Rows>, 1, m![
                        Rows / ({ Symbol<Part<Rows>> }),
                        { Tiles<Rows> } = 1 #{!} ({ Tiles<Rows> }),
                        Rows % ({ Symbol<Lane<Rows>> }), Width % 16,
                    ]>(tile);
                    // SAFETY: each group writes the selected lanes while preserving the owner's row
                    // stride.
                    let output = unsafe {
                        output.reshape::<Chip, Cluster, Reduced, m![
                            { linear::Groups<Tile<Rows>> },
                            { Layout<Tile<Rows>>::Lanes } #{!} ({ Symbol<Part<Rows>> }),
                            Width % 16,
                        ]>()
                    };
                    device.main.begin(weight.view())
                        .fetch::<
                            m![Width % 16, InputWidth % 1024 / 16],
                            m![Width % 32 / 16, InputWidth % 16],
                        >()
                        .collect::<
                            m![Width % 16, InputWidth % 1024 / 16, Width % 32 / 16],
                            m![InputWidth % 16],
                        >()
                        .contract_outer::<
                            m![Width % 16, InputWidth % 1024 / 16, { linear::Groups<Tile<Rows>> }],
                            m![Width % 32 / 16, InputWidth % 16], _, _, _,
                        >(&lanes)
                        .contract_packet::<m![Width % 32 / 16]>()
                        .contract_time::<m![Width % 16, { linear::Groups<Tile<Rows>> }]>()
                        .contract_lane::<
                            m![Width % 16, { linear::Groups<Tile<Rows>> }, Width % 32 / 16],
                            m![{ Layout<Tile<Rows>>::Lanes } # 8],
                        >(LaneMode::Interleaved)
                        .vector_init()
                        .vector_inter_slice_reduce::<
                            Reduced,
                            m![Width % 16, { linear::Groups<Tile<Rows>> }],
                        >(
                            InterSliceReduceOpF32::Add,
                        )
                        .vector_final()
                        .cast::<bf16, m![{ Layout<Tile<Rows>>::Lanes } # 16]>()
                        .transpose::<
                            m![
                                Width % 16 / 4,
                                { linear::Groups<Tile<Rows>> },
                                { Layout<Tile<Rows>>::Lanes }
                            ],
                            m![Width % 4 # 16],
                        >()
                        .commit_trim::<m![Width % 4]>()
                        .commit_view(output);
                };
                let weight: Weight = weight.to_dm(&mut device.tdma);
                let mut rows: Input<Rows> = DmTensor::new();
                input.to_dm_view(&mut device.tdma, rows.view_mut());
                let mut partial: Output<Rows> = DmTensor::new();
                // Live row group `k` is written in tile `k`: fewer live groups than tiles leave the
                // later tiles out.
                if <Tiles<Rows> as M>::SIZE <= live.count::<Lanes<Rows>>() {
                    #[unroll]
                    for tile in 0..<Tiles<Rows> as M>::SIZE {
                        contract(device, &weight, &rows, &mut partial, tile);
                    }
                } else {
                    for tile in 0..live.count::<Lanes<Rows>>() % <Tiles<Rows> as M>::SIZE {
                        contract(device, &weight, &rows, &mut partial, tile);
                    }
                }
                let exchanged = partial.view().cluster_swap().to_dm(&mut device.tdma);
                let result: Output<Rows> = device
                    .main
                    .begin_interleaved::<Term, _, _, _, _, _>(partial.view(), exchanged.view())
                    .fetch::<
                        m![{ linear::Groups<Rows> }, Term, { Layout<Rows>::Lanes }, Width % 16 / 8],
                        m![Width % 8],
                    >()
                    .fetch_cast::<f32>()
                    .collect::<
                        m![{ linear::Groups<Rows> }, Term, { Layout<Rows>::Lanes }, Width % 16 / 8],
                        m![Width % 8],
                    >()
                    .vector_init()
                    .vector_zip::<
                        Term,
                        m![{ linear::Groups<Rows> }, { Layout<Rows>::Lanes }, Width % 16 / 8],
                        m![{ linear::Groups<Rows> }, { Layout<Rows>::Lanes }, Width % 16 / 4],
                        m![Width % 4],
                    >(FpBinaryOp::AddF)
                    .vector_final()
                    .cast::<bf16, m![Width % 8 # 16]>()
                    .commit_trim::<m![Width % 8]>()
                    .commit();
                // SAFETY: both clusters now contain the complete sum in the same row-major cells.
                unsafe { result.reshape() }
            };
            // SAFETY: Width and Hidden name the same coordinates and cells in this branch.
            return live.with(unsafe { result.reshape() });
        }
        project(self, device, (live, rows))
    }
}

/// The channel-scaled output projection of every live row's attention, its codes restored to the
/// BF16 tiles the general BF16 projection contracts.
impl<
    'x,
    Hidden: AxisName,
    QWidth: AxisName,
    Rows: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    K,
    V,
> Op<&'x Live<gqa::Attended<Rows, KvHeads, Groups, Head>, Rows>>
    for BlockScaled<QWidth, Hidden, f8e4m3, f32, m![Hidden, 1 # 8], (), Chip, K, V>
where
    K: Borrow<HbmTensor<f8e4m3, Chip, m![Hidden, QWidth]>>,
    V: Borrow<HbmTensor<f32, Chip, m![Hidden, 1 # 8]>>,
{
    type Output = Live<Spread<Hidden, Rows>, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<gqa::Attended<Rows, KvHeads, Groups, Head>, Rows>,
    ) -> Self::Output {
        let rows = read::<QWidth, _, _, _, _>(input);
        project(&linear::Affine::new(self, ()), device, rows)
    }
}

/// The block-scaled output projection of every live row's attention, each row read as one row of
/// `QWidth` as the BF16 projection reads it.
impl<
    'x,
    Hidden: AxisName,
    QWidth: AxisName,
    Rows: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    A,
    K,
    V,
    Y: Memory,
> Op<&'x Live<gqa::Attended<Rows, KvHeads, Groups, Head>, Rows>>
    for BlockScaled<QWidth, Hidden, f4e2m1, f8e4m3, Scales<QWidth, Hidden>, Late<A, Y>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, m![Hidden, QWidth]>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Scales<QWidth, Hidden>>>,
    Held<Y>: Borrow<Scale>,
    Self: Op<
            (
                Live<PhantomData<Rows>, Rows>,
                Read<'x, QWidth, Rows, KvHeads>,
            ),
            Output = Live<Spread<Hidden, Rows>, Rows>,
        >,
{
    type Output = Live<Spread<Hidden, Rows>, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<gqa::Attended<Rows, KvHeads, Groups, Head>, Rows>,
    ) -> Self::Output {
        self.forward(device, read::<QWidth, _, _, _, _>(input))
    }
}

/// The live rows and each row's attention as one row of `Width`: its heads' values in head order.
fn read<Width: AxisName, Rows: AxisName, KvHeads: AxisName, Groups: AxisName, Head: AxisName>(
    input: &Live<gqa::Attended<Rows, KvHeads, Groups, Head>, Rows>,
) -> (
    Live<PhantomData<Rows>, Rows>,
    Read<'_, Width, Rows, KvHeads>,
) {
    const { assert!(Width::SIZE == KvHeads::SIZE * Groups::SIZE * Head::SIZE) };
    // SAFETY: each physical class keeps its cells; the head index and each head's values
    // become the high and low digits of `Width`, while row coordinates stay in place.
    (input.with(PhantomData), unsafe {
        input.value().view().reshape()
    })
}

/// Every live row's attention, read as `rows`, through `projection`'s contraction of BF16 tiles,
/// written to the feature slices the next projection reads.
fn project<'x, Hidden, QWidth, Rows, KvHeads, Matrix>(
    projection: &linear::Affine<Matrix, ()>,
    device: &mut Device,
    (live, rows): (
        Live<PhantomData<Rows>, Rows>,
        Read<'x, QWidth, Rows, KvHeads>,
    ),
) -> Live<Spread<Hidden, Rows>, Rows>
where
    Hidden: AxisName,
    QWidth: AxisName,
    Rows: AxisName,
    KvHeads: AxisName,
    Matrix: linear::Matrix<Chip, In = QWidth, Out = Hidden>,
{
    let group = |at| {
        rows.clone().tile::<Layout<Rows>::Groups, 1, m![
            { Layout<Rows>::Selected },
            QWidth % (QWidth / KvHeads)
        ]>(at)
    };
    let projected = projection.forward(&mut *device, (live, rows.clone(), group));
    let mut output: Spread<Hidden, Rows> = DmTensor::new();
    live.cover::<Lanes<Rows>, _>(
        &mut (device, &mut output),
        (
            |(device, output)| {
                projected
                    .view()
                    .to_dm_view(&mut device.tdma, output.view_mut())
            },
            |(device, output), packet| {
                Linear::<QWidth, Hidden>::group(&projected, packet).to_dm_view(
                    &mut device.tdma,
                    output
                        .view_mut()
                        .tile::<Layout<Rows>::Groups, 1, Layout<Rows>::GroupMut<
                            m![Hidden % ({ Symbol<normalization::Chunk<Hidden>> })],
                        >>(packet),
                );
            },
            |(device, output), row| {
                projected.view().tile::<m![Rows], 1, m![
                Hidden / (Hidden / ({ linear::kernel::Passes<QWidth, Hidden> })),
                Rows = 1 # Rows,
                Hidden % ({ linear::kernel::Block<QWidth, Hidden> }) / 4,
                Hidden % 4
            ]>(row).to_dm_view(&mut device.tdma,
                output.view_mut().tile::<
                    m![Rows],
                    1,
                    m![Rows = 1 #{!} Rows, Hidden % ({ Symbol<normalization::Chunk<Hidden>> })],
                >(row));
            },
        ),
    );
    live.with(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Module;
    use crate::ops::attention::tests::{Head, KvHeads};
    use crate::ops::linear::nvfp4::tests::projection;
    use crate::ops::normalization::{RowMajor, tests::random};
    use crate::tensor::axis::Sixteen;
    use crate::tensor::form::nvfp4::W4A16;
    use crate::tensor::{live::tests::count, scalar::Value};
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;
    use std::ops::Range;

    /// Live rows of a partial launch: one whole row group and part of the next.
    const LIVE: usize = 11;

    #[device(chip = 1)]
    fn attended_rows<H: AxisName, G: AxisName, P>(
        device: &mut Device,
        output: &P,
        live: &Value,
        attended: &HbmTensor<bf16, Chip, m![Sixteen, KvHeads, G, Head]>,
    ) -> HbmTensor<bf16, Chip, RowMajor<H, Sixteen>>
    where
        P: DeviceSend
            + for<'d, 'x> Module<
                &'x Live<gqa::Attended<Sixteen, KvHeads, G, Head>, Sixteen>,
                (),
                &'d mut (),
                Output = Live<Spread<H, Sixteen>, Sixteen>,
            >,
    {
        let value = attended.to_dm(&mut device.tdma);
        let attended: Live<gqa::Attended<Sixteen, KvHeads, G, Head>, Sixteen> =
            Live::<(), Sixteen>::read(live).with(value);
        output
            .forward(&mut *device, (), &mut (), &attended)
            .value()
            .to_hbm(&mut device.tdma)
    }

    /// Each live row's attended heads, read in head order as one input row, project to the FP32
    /// product with the output weight; the 4096-wide contraction rounds each cluster's half-row
    /// partial to BF16 before adding the two.
    async fn projects_attended_rows<H: AxisName, G: AxisName, Q: AxisName>(device: &mut Device) {
        let live = count(LIVE).to_hbm(&mut device.pdma).await.unwrap();
        let attended = random(Sixteen::SIZE * Q::SIZE, 0x3f84_d5b5);
        let weight = random(H::SIZE * Q::SIZE, 0xb547_0917);
        let output = linear::Affine::new(
            HostTensor::<bf16, m![H, Q]>::from_vec(weight.clone())
                .to_hbm::<Chip, m![H, Q]>(&mut device.pdma)
                .await
                .unwrap(),
            (),
        );
        let input = HostTensor::<bf16, m![Sixteen, KvHeads, G, Head]>::from_vec(attended.clone())
            .to_hbm::<Chip, m![Sixteen, KvHeads, G, Head]>(&mut device.pdma)
            .await
            .unwrap();
        let projected: HbmTensor<bf16, Chip, RowMajor<H, Sixteen>> =
            launch(attended_rows, (&mut *device, &output, &live, &input))
                .await
                .unwrap();
        let projected = projected
            .to_host::<RowMajor<H, Sixteen>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for row in 0..LIVE {
            for column in 0..H::SIZE {
                let partial = |inputs: Range<usize>| -> f32 {
                    inputs
                        .map(|at| {
                            attended[row * Q::SIZE + at].to_f32()
                                * weight[column * Q::SIZE + at].to_f32()
                        })
                        .sum()
                };
                let (low, high) = (partial(0..Q::SIZE / 2), partial(Q::SIZE / 2..Q::SIZE));
                let expected = low + high;
                let tolerance = (low.abs() + high.abs()) / 128.0 + expected.abs() / 256.0 + 1e-3;
                let got = projected[row * H::SIZE + column].to_f32();
                assert!(
                    (got - expected).abs() <= tolerance,
                    "{} wide, row {row} column {column}: {got} vs {expected}",
                    H::SIZE,
                );
            }
        }
    }

    /// The general and the 4096-wide contractions project each live row's heads in head order.
    #[tokio::test(flavor = "current_thread")]
    async fn projects_attended_rows_in_head_order() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attended_rows.topology()).unwrap();
        projects_attended_rows::<
            geometry::Hidden<2048>,
            geometry::QueryGroups<2>,
            geometry::QueryWidth<2048>,
        >(&mut device)
        .await;
        projects_attended_rows::<
            geometry::Hidden<4096>,
            geometry::QueryGroups<4>,
            geometry::QueryWidth<4096>,
        >(&mut device)
        .await;
    }

    /// Block-scaled weights project each live row's heads in head order as BF16 inputs: the exact
    /// sum times the weight tensor scale, within one BF16 rounding and FP32 accumulation error.
    #[tokio::test(flavor = "current_thread")]
    async fn block_scaled_projects_bf16_attended_rows() {
        type H = geometry::Hidden<2048>;
        type G = geometry::QueryGroups<2>;
        type Q = geometry::QueryWidth<2048>;
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attended_rows.topology()).unwrap();
        let live = count(LIVE).to_hbm(&mut device.pdma).await.unwrap();
        let attended = random(Sixteen::SIZE * Q::SIZE, 0x3f84_d5b5);
        let codes: Vec<f4e2m1> = (0..H::SIZE * Q::SIZE)
            .map(|at| f4e2m1::from_bits((at * 7 + at / 13) as u8 & 15))
            .collect();
        let scales: Vec<f8e4m3> = random(H::SIZE * Q::SIZE / 16, 0xb547_0917)
            .iter()
            .map(|seed| f8e4m3::from_f32(2f32.powf(6.0 * seed.to_f32() + 2.0)))
            .collect();
        let weight_scale_2 = 0.00017293295240961015;
        let output =
            projection::<Q, H, _>(&mut device, &codes, &scales, W4A16, weight_scale_2).await;
        let input = HostTensor::<bf16, m![Sixteen, KvHeads, G, Head]>::from_vec(attended.clone())
            .to_hbm::<Chip, m![Sixteen, KvHeads, G, Head]>(&mut device.pdma)
            .await
            .unwrap();
        let projected: HbmTensor<bf16, Chip, RowMajor<H, Sixteen>> =
            launch(attended_rows, (&mut device, &output, &live, &input))
                .await
                .unwrap();
        let projected = projected
            .to_host::<RowMajor<H, Sixteen>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let scale = f64::from(weight_scale_2);
        for row in 0..LIVE {
            for column in 0..H::SIZE {
                let (sum, magnitude) = (0..Q::SIZE)
                    .map(|at| {
                        let weight = codes[column * Q::SIZE + at].to_f32()
                            * scales[(column * Q::SIZE + at) / 16].to_f32();
                        f64::from(weight) * f64::from(attended[row * Q::SIZE + at].to_f32())
                    })
                    .fold((0.0, 0.0), |(sum, magnitude), term: f64| {
                        (sum + term, magnitude + term.abs())
                    });
                let exact = sum * scale;
                let bound = exact.abs() / 128.0 + magnitude * scale / f64::from(1 << 14);
                let got = f64::from(projected[row * H::SIZE + column].to_f32());
                assert!(
                    (got - exact).abs() <= bound,
                    "row {row} column {column}: {got} against {exact}"
                );
            }
        }
    }
}
