use super::*;
use crate::Op;
use crate::ops::vector::RowQuantize;
use core::marker::PhantomData;

/// A gate or up panel's channel scales, fetched into the VRF of each row slice.
pub(super) struct FetchChannels;

impl<'a, Intermediate: AxisName, Rows: AxisName> Op<&'a Channels<Intermediate, Rows>>
    for FetchChannels
{
    type Output = ChannelScale<Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        scale: &'a Channels<Intermediate, Rows>,
    ) -> Self::Output {
        context
            .sub
            .begin(scale.view())
            .fetch::<m![{ Out<Intermediate> } / 8], m![{ Out<Intermediate> } % 8]>()
            .collect::<m![{ Out<Intermediate> } / 8], m![{ Out<Intermediate> } % 8]>()
            .to_vrf()
    }
}

/// The span rows' BF16 row scales, stored in HBM in the row order of each gate/up wave.
pub(super) struct InputScales;

impl<'a, Hidden: AxisName, Rows: AxisName> Op<&'a Owned<Hidden, Rows>> for InputScales {
    type Output = Scales<Rows>;

    fn forward(&self, context: &mut Device, rows: &'a Owned<Hidden, Rows>) -> Self::Output {
        let mut scales: DmTensor<bf16, Chip, m![Rows / (Rows / 2)], m![1 # 256], Labels<Rows>> =
            DmTensor::new();
        context
            .main
            .begin(rows.view())
            .fetch::<m![Rows % (Rows / 2), Hidden % 16 / 8], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % (Rows / 2), Hidden % 16 / 8], m![Hidden % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows % (Rows / 2), Hidden % 16 / 4], m![Hidden % 4]>()
            .vector_row_scale::<Hidden, m![Rows % (Rows / 2)]>()
            .vector_inter_slice_reduce::<m![1 # 256], m![Rows % (Rows / 2)]>(
                InterSliceReduceOpF32::Max,
            )
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![Rows % (Rows / 2) / 4], m![Rows % 4 # 16]>()
            .commit_trim::<m![Rows % 4]>()
            .commit_view(scales.view_mut());
        let mut stored: Scales<Rows> = HbmTensor::new();
        // SAFETY: the stored labels preserve the row order consumed by each gate/up wave.
        let labelled = unsafe {
            stored.view_mut().reshape::<Chip, m![
                Rows % (Rows / 8) / 8,
                Rows / (Rows / 4),
                Rows % (Rows / 4) / (Rows / 8),
                Rows % 8
            ]>()
        };
        scales.view().to_hbm_view(&mut context.tdma, labelled);
        stored
    }
}

/// The copied row scales, fetched into the VRF of each slice.
pub(super) struct FetchRows;

impl<'i, Intermediate: AxisName, Rows: AxisName, S: M>
    Op<DmTensorView<'i, bf16, Chip, Cluster<Intermediate>, S, m![Rows % (Rows / 8)]>>
    for FetchRows
{
    type Output = VrfTensor<f32, Chip, Cluster<Intermediate>, S, m![Rows % (Rows / 8)]>;

    fn forward(
        &self,
        context: &mut Device,
        scales: DmTensorView<'i, bf16, Chip, Cluster<Intermediate>, S, m![Rows % (Rows / 8)]>,
    ) -> Self::Output {
        context
            .sub
            .begin(scales)
            .fetch::<m![Rows % (Rows / 8) / 8], m![Rows % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % (Rows / 8) / 8], m![Rows % 8]>()
            .to_vrf()
    }
}

type Tile<Hidden, Intermediate, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<Intermediate>,
    m![Hidden % 1024 / 32, Rows / (Rows / 8)],
    m![
        Hidden / 1024,
        Rows % (Rows / 8) / 16 = 1,
        Rows % 16,
        Hidden % 32
    ],
>;

/// The stored span rows written as up to four waves of the cluster's rows, the first wave's live
/// rows alone below a row group.
pub(super) struct Tiles<Intermediate>(pub(super) PhantomData<Intermediate>);

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(&'a gate::Normalized<Hidden, Rows, Chip>, (usize, usize))> for Tiles<Intermediate>
{
    type Output = [Tile<Hidden, Intermediate, Rows>; 4];

    fn forward(
        &self,
        context: &mut Device,
        (input, (live, rows)): (&'a gate::Normalized<Hidden, Rows, Chip>, (usize, usize)),
    ) -> Self::Output {
        let to_dm =
            |context: &mut Device,
             stored: &HbmTensorView<
                '_,
                bf16,
                Chip,
                m![
                    Rows % (Rows / 8) / 16,
                    Rows % 16 / 8,
                    Rows / (Rows / 8),
                    Hidden / 32,
                    Rows % 8,
                    Hidden % 32
                ],
            >,
             (output, wave): (&mut Tile<Hidden, Intermediate, Rows>, usize)| {
                stored.tile::<m![Rows % (Rows / 8) / 16], 1,
                m![Rows % (Rows / 8) / 16 = 1 # (Rows / 128), Rows % 16 / 8,
                    Rows / (Rows / 8), Hidden / 32, Rows % 8, Hidden % 32]>(wave)
                .to_dm_view(&mut context.tdma, output.view_mut());
            };
        // SAFETY: wave, half, group and lane digits preserve the stored rows' coordinates.
        let view = unsafe {
            input.view().reshape::<Chip, m![
                Rows % (Rows / 8) / 16,
                Rows % 16 / 8,
                Rows / (Rows / 8),
                Hidden / 32,
                Rows % 8,
                Hidden % 32
            ]>()
        };
        let mut waves = [Tile::new(), Tile::new(), Tile::new(), Tile::new()];
        if rows <= 8 {
            view.tile::<m![Rows % (Rows / 8) / 16], 1, m![Rows % (Rows / 8) / 16 = 1 # (Rows / 128),
                Rows % 16 / 8, Rows / (Rows / 8), Hidden / 32, Rows % 8, Hidden % 32]>(0)
                .tile::<m![Rows % 16 / 8], 1, m![Rows % (Rows / 8) / 16 = 1 # (Rows / 128),
                    Rows % 16 / 8 = 1 # 2, Rows / (Rows / 8), Hidden / 32, Rows % 8,
                    Hidden % 32]>(0)
                .tile::<m![Rows / (Rows / 8)], 1, m![Rows % (Rows / 8) / 16 = 1 # (Rows / 128),
                    Rows % 16 / 8 = 1 # 2, Rows / (Rows / 8) = 1 # 8, Hidden / 32, Rows % 8,
                    Hidden % 32]>(0)
                .to_dm_view(&mut context.tdma, waves[0].view_mut()
                    .slice_tile::<m![Rows / (Rows / 8)], 1,
                        m![Hidden % 1024 / 32, Rows / (Rows / 8) = 1 #{!} 8]>(0)
                    .tile::<m![Rows % 16 / 8], 1, m![Hidden / 1024, Rows % (Rows / 8) / 16 = 1,
                        Rows % 16 / 8 = 1 #{!} 2, Rows % 8, Hidden % 32]>(0));
        } else {
            to_dm(context, &view, (&mut waves[0], 0));
            if live > 1 {
                to_dm(context, &view, (&mut waves[1], 1));
                if live > 2 {
                    to_dm(context, &view, (&mut waves[2], 2));
                    to_dm(context, &view, (&mut waves[3], 3));
                }
            }
        }
        waves
    }
}

type TileScale<Hidden, Intermediate, Rows> = VrfTensor<
    f32,
    Chip,
    Cluster<Intermediate>,
    m![Hidden % 1024 / 32, Rows / (Rows / 8)],
    m![Rows % 16],
>;

/// Wave `wave`'s row scales, fetched into the VRF of each slice.
pub(super) struct FetchTile;

/// The row scales of a span, written in each slice.
type InDm<Hidden, Intermediate, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<Intermediate>,
    m![Hidden % 1024 / 32, Rows / (Rows / 8)],
    m![Rows % (Rows / 8)],
>;

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(&'a InDm<Hidden, Intermediate, Rows>, usize)> for FetchTile
{
    type Output = TileScale<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        (scales, wave): (&'a InDm<Hidden, Intermediate, Rows>, usize),
    ) -> Self::Output {
        context.sub.begin(scales.view().tile::<m![Rows % (Rows / 8) / 16], 1,
            m![Rows % (Rows / 8) / 16 = 1 # (Rows / 128), Rows % 16]>(wave))
            .fetch::<m![Rows % 16 / 8], m![Rows % 8]>().fetch_cast::<f32>()
            .collect::<m![Rows % 16 / 8], m![Rows % 8]>().to_vrf()
    }
}

/// A wave's rows as E4M3, divided by their row scales, in the wave's rows of the quantized input.
pub(super) struct Quantize;

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(
        &'a Tile<Hidden, Intermediate, Rows>,
        (&'a TileScale<Hidden, Intermediate, Rows>, usize),
        &'a mut Input<Hidden, Intermediate, Rows>,
    )> for Quantize
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (input, (scale, wave), output): (
            &'a Tile<Hidden, Intermediate, Rows>,
            (&'a TileScale<Hidden, Intermediate, Rows>, usize),
            &'a mut Input<Hidden, Intermediate, Rows>,
        ),
    ) {
        context.main.begin(input.view())
            .fetch::<m![Hidden / 1024, Rows % 16, Hidden % 32 / 8], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Hidden / 1024, Rows % 16, Hidden % 32 / 8], m![Hidden % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Hidden / 1024, Rows % 16, Hidden % 32 / 4], m![Hidden % 4]>()
            .vector_quantize::<
                m![Hidden / 1024, Rows % 16, Hidden % 32 / 8],
                m![Hidden % 8],
                _,
            >(scale)
            .commit_view(output.view_mut().tile::<m![Rows % (Rows / 8) / 16], 1,
                m![Hidden / 1024, Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128),
                    Rows % 16, Hidden % 32]>(wave));
    }
}

/// Panel `panel`'s inputs of wave `wave`, in the TRF lanes of every partition.
pub(super) struct Fill;

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(&'a Input<Hidden, Intermediate, Rows>, (usize, usize))> for Fill
{
    type Output = Lanes<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        (input, (panel, wave)): (&'a Input<Hidden, Intermediate, Rows>, (usize, usize)),
    ) -> Self::Output {
        context.sub.begin(input.view()
            .tile::<
                m![Hidden / 1024],
                1,
                m![Hidden / 1024 = 1 # (Hidden / 1024), Rows % (Rows / 8), Hidden % 32],
            >(panel)
            .tile::<m![Rows % (Rows / 8) / 16], 1, m![Hidden / 1024 = 1 # (Hidden / 1024),
                Rows % (Rows / 8) / 16 = 1 # (Rows / 128), Rows % 16, Hidden % 32]>(wave))
            .fetch::<m![Rows % 8, Rows % 16 / 8], m![Hidden % 32]>()
            .switch::<Partition<Hidden, Intermediate>,
                m![Rows % 8, Rows % 16 / 8, Hidden % 128 / 32, Rows / (Rows / 8)]>(
                SwitchConfig::TransposedBroadcast1 { slice1: 8, slice0: 32 })
            .collect::<m![Rows % 8, Rows % 16 / 8, Hidden % 128 / 32, Rows / (Rows / 8)],
                m![Hidden % 32]>().to_trf()
    }
}
