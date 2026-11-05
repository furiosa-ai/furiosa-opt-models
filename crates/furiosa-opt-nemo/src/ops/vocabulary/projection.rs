//! Vocabulary tiles shared by full-logit projection and fused greedy selection.

use crate::ops::layout::{Dm, Live, Rows as Layout, axes::Lanes};

use furiosa_opt_std::prelude::*;

use super::argmax::{Candidates, Selected};
use super::{Greedy, LmHead};
use crate::Module;
use crate::ops::layout::axes::{Chip, Tile};
use crate::ops::normalization::Spread;
use crate::ops::vocabulary::VocabPad;
use crate::{ops::linear::Groups, ops::weight::Weight};

/// Two passes' weights fit beside the input rows in DM, allowing the next pass to stream
/// during the current contraction.
pub(crate) type PassRows = m![198];

pub type Scores<Rows> = m![Rows, VocabPad];

#[derive(Clone, Debug)]
pub struct Contraction<Hidden>(core::marker::PhantomData<Hidden>);

impl<Hidden: AxisName> AxisName for Contraction<Hidden> {
    const NAME: Ident = Ident::new("Contraction");

    const SIZE: usize = match Hidden::SIZE {
        4096 => 512,
        5120 => 640,
        _ => Hidden::SIZE / 4,
    };
}

pub type Chunk<Hidden> = Symbol<Contraction<Hidden>>;

/// The slices sharing one row's contraction, `Hidden / chunk`.
pub(super) type Parts<Hidden> = m![Hidden / ({ Chunk<Hidden> })];

/// The vocabulary rows one slice writes over every pass, `VocabPad * parts / 512`: its block of
/// the rows, `PassRows` a pass, since the blocks fill the `256 / parts` slots of both clusters.
pub(super) type Block<Hidden> = m![(VocabPad, { Parts<Hidden> }) / 512];

/// The passes a slice's block takes.
type Passes<Hidden> = m![{ Block<Hidden> } / ({ PassRows })];

pub(super) type Cluster = m![VocabPad / (VocabPad / 2)];
type ProjectionSlices<Hidden> =
    m![VocabPad % (VocabPad / 2) / ({ Block<Hidden> }), { Parts<Hidden> }];

pub(super) type Reduced<Hidden> =
    m![VocabPad % (VocabPad / 2) / ({ Block<Hidden> }), 1 # ({ Parts<Hidden> })];

/// A row tile of at most 32 rows, whose input lanes are restaged for each weight pass.
type RowTile<Rows> = Symbol<Tile<Rows, 32>>;

/// Scores of the live rows; each weight pass projects only the live row tiles (every row of a
/// one-tile envelope), so the weight streams once a call. Rows past the live count are unspecified.
impl<'d, 'x, Hidden: AxisName, Rows: AxisName>
    Module<(&'d mut Device, (&'x Spread<Hidden, Rows>, Live<Rows>))> for LmHead<Hidden>
{
    type Output = HbmTensor<f32, Chip, Scores<Rows>>;

    fn forward(
        &self,
        (device, (input, live)): (&'d mut Device, (&'x Spread<Hidden, Rows>, Live<Rows>)),
    ) -> Self::Output {
        let weight = &self.0;
        // Up to sixteen rows keep their lanes across every pass; more rows restage each row tile's
        // lanes per pass.
        if const { Rows::SIZE > 16 } {
            let packet_logits = |device: &mut Device,
                                 weight: &DmTensor<
                bf16,
                Chip,
                Cluster,
                ProjectionSlices<Hidden>,
                m![VocabPad % ({ PassRows }), Hidden % ({ Chunk<Hidden> })],
            >,
                                 packet: usize| {
                let lanes = Input::<Hidden, Rows, m![1]>(core::marker::PhantomData)
                    .forward((&mut *device, Layout::<Rows>::group(input.view(), packet)));
                Product::<Layout<Rows>::Lanes>(core::marker::PhantomData)
                    .forward((device, (weight, &lanes)))
            };
            // A store pays per row (792-byte runs of each) and a projection per eight-row packet,
            // so the live rows go as whole tiles, then whole packets, then the last packet's rows.
            let tiles = live.rows() / <RowTile<Rows> as M>::SIZE;
            let packets = live.rows() / Lanes::<Rows>::SIZE;
            let first = tiles * <RowTile<Rows> as M>::SIZE / Lanes::<Rows>::SIZE;
            let mut output: HbmTensor<f32, Chip, Scores<Rows>> = HbmTensor::new();
            for tile in 0..<Passes<Hidden> as M>::SIZE {
                let weight = Slab.forward((&mut *device, (weight, tile)));
                for row in 0..tiles {
                    let input = input.view().tile::<m![Rows / ({ RowTile<Rows> })], 1,
                        m![Rows / ({ RowTile<Rows> }) = 1 # (Rows / ({ RowTile<Rows> })), Rows % ({ RowTile<Rows> }), Hidden % 16]>(row);
                    let lanes = Input::<Hidden, Rows, m![Rows % ({ RowTile<Rows> }) / 8]>(
                        core::marker::PhantomData,
                    )
                    .forward((&mut *device, input));
                    let logits =
                        Product::<m![Rows % ({ RowTile<Rows> })]>(core::marker::PhantomData)
                            .forward((&mut *device, (&weight, &lanes)));
                    logits.view().to_hbm_view(&mut device.tdma,
                        output.view_mut().tile::<m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1,
                            m![Rows, VocabPad / ({ Block<Hidden> }), 1 #{!} ({ Passes<Hidden> }), VocabPad % ({ PassRows })]>(tile)
                        .tile::<m![Rows / ({ RowTile<Rows> })], 1,
                            m![1 #{!} (Rows / ({ RowTile<Rows> })), Rows % ({ RowTile<Rows> }), VocabPad / ({ Block<Hidden> }), 1 #{!} ({ Passes<Hidden> }), VocabPad % ({ PassRows })]>(row));
                }
                for index in 0..packets - first {
                    let packet = first + index;
                    let logits = packet_logits(device, &weight, packet);
                    logits.view().to_hbm_view(&mut device.tdma,
                        output.view_mut().tile::<m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1,
                            m![Rows, VocabPad / ({ Block<Hidden> }), 1 #{!} ({ Passes<Hidden> }), VocabPad % ({ PassRows })]>(tile)
                        .tile::<Layout<Rows>::Groups, 1,
                            Layout<Rows>::GroupMut<m![VocabPad / ({ Block<Hidden> }), 1 #{!} ({ Passes<Hidden> }), VocabPad % ({ PassRows })]>>(packet));
                }
                for _ in 0..live.count() - packets {
                    let logits = packet_logits(device, &weight, packets);
                    for lane in 0..live.rows() % Lanes::<Rows>::SIZE {
                        logits.view()
                            .tile::<Layout<Rows>::Lanes, 1,
                                m![{ Layout<Rows>::Lanes } = 1 # ({ Symbol<Lanes<Rows>> }), VocabPad % ({ PassRows })]>(lane)
                            .to_hbm_view(&mut device.tdma,
                                output.view_mut().tile::<m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1,
                                    m![Rows, VocabPad / ({ Block<Hidden> }), 1 #{!} ({ Passes<Hidden> }), VocabPad % ({ PassRows })]>(tile)
                                .tile::<Layout<Rows>::Groups, 1,
                                    Layout<Rows>::GroupMut<m![VocabPad / ({ Block<Hidden> }), 1 #{!} ({ Passes<Hidden> }), VocabPad % ({ PassRows })]>>(packets)
                                .tile::<Layout<Rows>::Lanes, 1,
                                    m![{ Layout<Rows>::SelectedMut }, { Layout<Rows>::Lanes } = 1 #{!} ({ Symbol<Lanes<Rows>> }), VocabPad / ({ Block<Hidden> }), 1 #{!} ({ Passes<Hidden> }), VocabPad % ({ PassRows })]>(lane));
                    }
                }
            }
            return output;
        }
        let lanes = Input::<Hidden, Rows, Groups<Rows>>(core::marker::PhantomData)
            .forward((&mut *device, input.view()));
        let mut output: HbmTensor<f32, Chip, Scores<Rows>> = HbmTensor::new();
        #[unroll]
        for tile in 0..<Passes<Hidden> as M>::SIZE {
            let weight = Slab.forward((&mut *device, (weight, tile)));
            let logits = Product::<m![Rows]>(core::marker::PhantomData)
                .forward((&mut *device, (&weight, &lanes)));
            logits.view().to_hbm_view(&mut device.tdma, output.view_mut().tile::<
                m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1,
                m![Rows, VocabPad / ({ Block<Hidden> }), 1 #{!} ({ Passes<Hidden> }), VocabPad % ({ PassRows })]>(tile));
        }
        output
    }
}

/// Select each live row's first maximum while each vocabulary tile is still live; each weight
/// pass projects only the live row tiles. Tokens of rows past the live count are unspecified.
impl<'d, 'x, Hidden: AxisName, Rows: AxisName>
    Module<(&'d mut Device, (&'x Spread<Hidden, Rows>, Live<Rows>))> for Greedy<Hidden>
{
    type Output = Selected<Rows>;

    fn forward(
        &self,
        (device, (input, live)): (&'d mut Device, (&'x Spread<Hidden, Rows>, Live<Rows>)),
    ) -> Self::Output {
        let (weight, mask, iota) = (&self.head.0, &self.mask, &self.iota);
        // One row tile keeps its lanes across every pass; several restage each tile's lanes per
        // pass.
        if const { <m![Rows / ({ RowTile<Rows> })] as M>::SIZE > 1 } {
            let tiles = live.retile::<Tile<Rows, 32>>().count();
            // Every vocabulary weight stays live while bounded row tiles consume it.
            let mut candidates = Candidates::<Hidden, Rows>::new();
            #[unroll]
            for tile in 0..<Passes<Hidden> as M>::SIZE {
                let weight = Slab.forward((&mut *device, (weight, tile)));
                for row_tile in 0..tiles {
                    let input = input.view().tile::<m![Rows / ({ RowTile<Rows> })], 1,
                        m![Rows / ({ RowTile<Rows> }) = 1 # (Rows / ({ RowTile<Rows> })), Rows % ({ RowTile<Rows> }), Hidden % 16]>(row_tile);
                    let lanes = Input::<Hidden, Rows, m![Rows % ({ RowTile<Rows> }) / 8]>(
                        core::marker::PhantomData,
                    )
                    .forward((&mut *device, input));
                    let logits =
                        Product::<m![Rows % ({ RowTile<Rows> })]>(core::marker::PhantomData)
                            .forward((&mut *device, (&weight, &lanes)));
                    candidates.push::<Tile<Rows, 32>, _>(device, &logits,
                        mask.view().tile::<m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1,
                            m![VocabPad / ({ Block<Hidden> }), VocabPad % ({ PassRows }) # ({ Block<Hidden> })]>(tile),
                        iota.view().tile::<m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1,
                            m![VocabPad / ({ Block<Hidden> }), VocabPad % ({ PassRows }) # ({ Block<Hidden> })]>(tile),
                        tile, row_tile);
                }
            }
            return candidates.select(device);
        }
        let lanes = Input::<Hidden, Rows, Groups<Rows>>(core::marker::PhantomData)
            .forward((&mut *device, input.view()));
        let mut candidates = Candidates::<Hidden, Rows>::new();
        #[unroll]
        for tile in 0..<Passes<Hidden> as M>::SIZE {
            let weight = Slab.forward((&mut *device, (weight, tile)));
            let logits = Product::<m![Rows % Rows]>(core::marker::PhantomData)
                .forward((&mut *device, (&weight, &lanes)));
            candidates.push::<Rows, _>(device, &logits,
                mask.view().tile::<m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1,
                    m![VocabPad / ({ Block<Hidden> }), VocabPad % ({ PassRows }) # ({ Block<Hidden> })]>(tile),
                iota.view().tile::<m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1,
                    m![VocabPad / ({ Block<Hidden> }), VocabPad % ({ PassRows }) # ({ Block<Hidden> })]>(tile),
                tile, 0);
        }
        candidates.select(device)
    }
}

impl<'d, 'x, Hidden: AxisName, Rows: AxisName>
    Module<(&'d mut Device, &'x Dm<Lanes<Rows>, Spread<Hidden, Rows>>)> for Greedy<Hidden>
{
    type Output = Selected<Rows>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Dm<Lanes<Rows>, Spread<Hidden, Rows>>),
    ) -> Self::Output {
        self.forward((device, (&input.value, input.tiles)))
    }
}

pub(super) struct Input<Hidden, Rows, Count, C = Chunk<Hidden>, Slices = ProjectionSlices<Hidden>>(
    pub core::marker::PhantomData<(Hidden, Rows, Count, C, Slices)>,
);

impl<
    'i,
    Hidden: AxisName,
    Rows: AxisName,
    Count: M,
    Element: M,
    C: furiosa_opt_std::prelude::Count,
    Slices: M,
>
    Module<(
        &mut Device,
        DmTensorView<
            'i,
            bf16,
            Chip,
            crate::ops::normalization::Cluster,
            crate::ops::normalization::FeatureSlices<Hidden>,
            Element,
        >,
    )> for Input<Hidden, Rows, Count, C, Slices>
{
    type Output = TrfTensor<
        bf16,
        Chip,
        Cluster,
        Slices,
        m![{ Layout<Rows>::Lanes }],
        m![{ Count }, Hidden % ({ C })],
    >;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            DmTensorView<
                'i,
                bf16,
                Chip,
                crate::ops::normalization::Cluster,
                crate::ops::normalization::FeatureSlices<Hidden>,
                Element,
            >,
        ),
    ) -> Self::Output {
        if const { Hidden::SIZE == 16 * 256 } {
            // Every slice holds sixteen features in both clusters, preserving the source addresses.
            let input = unsafe { input.reshape::<Chip, Cluster, m![Hidden / 16], Element>() };
            device
                .sub
                .begin(input)
                .fetch::<m![{ Layout<Rows>::Lanes }, { Count }], m![Hidden % 16]>()
                .switch::<Slices, m![
                    { Layout<Rows>::Lanes },
                    { Count },
                    Hidden % ({ C }) / 16
                ]>(SwitchConfig::TransposedBroadcast1 {
                    slice1: Hidden::SIZE / C::SIZE,
                    slice0: C::SIZE / 16,
                })
                .collect::<m![
                    { Layout<Rows>::Lanes },
                    { Count },
                    Hidden % ({ C }) / 16
                ], m![Hidden % 16]>()
                .to_trf()
        } else if const { Hidden::SIZE > 16 * 256 } {
            // A slice holds several switch packets, and the switch delivers a part's packets from other
            // slices innermost, so the parts land in DM before their lanes fill in feature order.
            type Slice<Hidden> = Symbol<crate::ops::normalization::Chunk<Hidden>>;
            let input = unsafe {
                input.reshape::<Chip, Cluster, crate::ops::normalization::FeatureSlices<Hidden>, Element>()
            };
            let mut staged: DmTensor<
                bf16,
                Chip,
                Cluster,
                Slices,
                m![{ Count }, { Layout<Rows>::Lanes }, Hidden % ({ C })],
            > = DmTensor::new();
            device
                .sub
                .begin(input)
                .fetch::<m![
                    { Layout<Rows>::Lanes },
                    { Count },
                    Hidden % ({ Slice<Hidden> }) / 16
                ], m![Hidden % 16]>()
                .switch::<Slices, m![
                    { Layout<Rows>::Lanes },
                    { Count },
                    Hidden % ({ Slice<Hidden> }) / 16,
                    Hidden % ({ C }) / ({ Slice<Hidden> })
                ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
                .collect::<m![
                    { Layout<Rows>::Lanes },
                    { Count },
                    Hidden % ({ Slice<Hidden> }) / 16,
                    Hidden % ({ C }) / ({ Slice<Hidden> })
                ], m![Hidden % 16]>()
                .commit_trim::<m![Hidden % 16]>()
                .commit_view(staged.view_mut());
            device
                .sub
                .begin(staged.view())
                .fetch::<m![
                    { Layout<Rows>::Lanes },
                    { Count },
                    Hidden % ({ C }) / 16
                ], m![Hidden % 16]>()
                .collect::<m![
                    { Layout<Rows>::Lanes },
                    { Count },
                    Hidden % ({ C }) / 16
                ], m![Hidden % 16]>()
                .to_trf()
        } else {
            // Both physical clusters already hold every row; only the logical cluster axis changes.
            let input = unsafe { input.reshape::<Chip, Cluster, m![Hidden / 16 # 256], Element>() };
            device
                .sub
                .begin(input)
                .fetch::<m![{ Layout<Rows>::Lanes }, { Count }], m![Hidden % 16]>()
                .switch::<Slices, m![
                    { Layout<Rows>::Lanes },
                    { Count },
                    Hidden % ({ C }) / 16
                ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
                .collect::<m![
                    { Layout<Rows>::Lanes },
                    { Count },
                    Hidden % ({ C }) / 16
                ], m![Hidden % 16]>()
                .to_trf()
        }
    }
}

struct Slab;

impl<Hidden: AxisName> Module<(&mut Device, (&Weight<m![VocabPad, Hidden]>, usize))> for Slab {
    type Output = DmTensor<
        bf16,
        Chip,
        Cluster,
        ProjectionSlices<Hidden>,
        m![VocabPad % ({ PassRows }), Hidden % ({ Chunk<Hidden> })],
    >;

    fn forward(
        &self,
        (device, (weight, tile)): (&mut Device, (&Weight<m![VocabPad, Hidden]>, usize)),
    ) -> Self::Output {
        weight
            .tensor()
            .view()
            .tile::<m![VocabPad % ({ Block<Hidden> }) / ({ PassRows })], 1, m![
                VocabPad / ({ Block<Hidden> }),
                VocabPad % ({ PassRows }) # ({ Block<Hidden> }), Hidden
            ]>(tile)
            .to_dm(&mut device.tdma)
    }
}

struct Product<Part>(core::marker::PhantomData<Part>);

impl<Hidden: AxisName, Lanes: M, Count: M, Part: M>
    Module<(
        &mut Device,
        (
            &DmTensor<
                bf16,
                Chip,
                Cluster,
                ProjectionSlices<Hidden>,
                m![VocabPad % ({ PassRows }), Hidden % ({ Chunk<Hidden> })],
            >,
            &TrfTensor<
                bf16,
                Chip,
                Cluster,
                ProjectionSlices<Hidden>,
                m![{ Lanes }],
                m![{ Count }, Hidden % ({ Chunk<Hidden> })],
            >,
        ),
    )> for Product<Part>
{
    type Output =
        DmTensor<f32, Chip, Cluster, Reduced<Hidden>, m![{ Part }, VocabPad % ({ PassRows })]>;

    fn forward(
        &self,
        (device, (weight, lanes)): (
            &mut Device,
            (
                &DmTensor<
                    bf16,
                    Chip,
                    Cluster,
                    ProjectionSlices<Hidden>,
                    m![VocabPad % ({ PassRows }), Hidden % ({ Chunk<Hidden> })],
                >,
                &TrfTensor<
                    bf16,
                    Chip,
                    Cluster,
                    ProjectionSlices<Hidden>,
                    m![{ Lanes }],
                    m![{ Count }, Hidden % ({ Chunk<Hidden> })],
                >,
            ),
        ),
    ) -> Self::Output {
        device.main.begin(weight.view())
            .fetch::<m![VocabPad % ({ PassRows }), Hidden % ({ Chunk<Hidden> }) / 16], m![Hidden % 16]>()
            .collect::<m![VocabPad % ({ PassRows }), Hidden % ({ Chunk<Hidden> }) / 16], m![Hidden % 16]>()
            .contract_outer::<m![VocabPad % ({ PassRows }), Hidden % ({ Chunk<Hidden> }) / 32, { Count }], m![Hidden % 32], _, _, _>(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![VocabPad % ({ PassRows }), { Count }]>()
            .contract_lane::<m![VocabPad % ({ PassRows }), { Count }], m![{ Lanes } # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Reduced<Hidden>, m![VocabPad % ({ PassRows }), { Count }]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .transpose::<m![VocabPad % ({ PassRows }) / 2, { Part }], m![VocabPad % 2 # 8]>()
            .commit_trim::<m![VocabPad % 2]>()
            .commit()
    }
}
