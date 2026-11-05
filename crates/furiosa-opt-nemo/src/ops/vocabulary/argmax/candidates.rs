use crate::ops::layout::Rows as Layout;
use furiosa_opt_std::prelude::*;

use crate::ops::linear::Groups;

use super::super::{VocabPad, projection};
use crate::ops::layout::axes::Chip;

axes![Position = 99, Member = 2, Entry = 1536];

type CandidateGroups<Hidden> = m![VocabPad % ({ projection::Block<Hidden> }) / 198];

type Stored<D, Hidden, Rows> = DmTensor<
    D,
    Chip,
    projection::Cluster,
    projection::Reduced<Hidden>,
    m![{ CandidateGroups<Hidden> }, Rows, Member],
>;

type Local<D, Hidden, Rows> =
    DmTensor<D, Chip, projection::Cluster, projection::Reduced<Hidden>, m![{ Rows }, Member]>;

type Tile<Hidden, Rows> = DmTensor<
    f32,
    Chip,
    projection::Cluster,
    projection::Reduced<Hidden>,
    m![{ Rows }, Position, Member],
>;

type SourceSlice<Hidden> = m![
        Entry % (Entry / 2) / ({ m![({ CandidateGroups<Hidden> }, Member) / 1] }),
    1 # ({ projection::Parts<Hidden> })
];

type SourceElement<Hidden, Rows> = m![
    Entry % ({ m![({ CandidateGroups<Hidden> }, Member) / 1] }) / 2,
    Rows,
    Entry % 2
];

type Cluster<Rows> = m![{ Layout<Rows>::Lanes } # 8 / 4];
type CandidateSlices<Rows> = m![{ Layout<Rows>::Lanes } # 8 % 4, Entry / 24];
type Reduced<Rows> = m![{ Layout<Rows>::Lanes } # 8 % 4, 1 # 64];
type Element<Rows> = m![{ Groups<Rows> }, Entry % 24];
type Gathered<D, Rows> = DmTensor<D, Chip, Cluster<Rows>, CandidateSlices<Rows>, Element<Rows>>;

pub(super) type Selected<Rows> =
    DmTensor<i32, Chip, Cluster<Rows>, Reduced<Rows>, m![{ Groups<Rows> }, 1 # 2]>;

/// Per-pass pairwise maxima and reversed token ranks reduced to each row's first maximum.
pub(in crate::ops::vocabulary) struct Candidates<Hidden: AxisName, Rows: AxisName> {
    maximum: Stored<f32, Hidden, Rows>,
    rank: Stored<i32, Hidden, Rows>,
}

impl<Hidden: AxisName, Rows: AxisName> Candidates<Hidden, Rows> {
    pub(in crate::ops::vocabulary) fn new() -> Self {
        Self {
            maximum: DmTensor::new(),
            rank: DmTensor::new(),
        }
    }

    /// Stores pass `at`'s candidates for row tile `tile` of `Tile` rows.
    pub(in crate::ops::vocabulary) fn push<Tile: AxisName, Columns: M>(
        &mut self,
        device: &mut Device,
        logits: &DmTensor<
            f32,
            Chip,
            projection::Cluster,
            projection::Reduced<Hidden>,
            m![Rows % Tile, VocabPad % 198],
        >,
        mask: HbmTensorView<'_, f32, Chip, Columns>,
        iota: HbmTensorView<'_, i32, Chip, Columns>,
        at: usize,
        tile: usize,
    ) {
        let maximum = self
            .maximum
            .view_mut()
                .tile::<CandidateGroups<Hidden>, 1, m![1 #{!} ({ CandidateGroups<Hidden> }), Rows, Member]>(at)
                .tile::<m![Rows / Tile], 1, m![1 #{!} ({ CandidateGroups<Hidden> }),
                Rows / Tile = 1 #{!} (Rows / Tile), Rows % Tile, Member]>(tile);
        let rank = self
            .rank
            .view_mut()
                .tile::<CandidateGroups<Hidden>, 1, m![1 #{!} ({ CandidateGroups<Hidden> }), Rows, Member]>(at)
                .tile::<m![Rows / Tile], 1, m![1 #{!} ({ CandidateGroups<Hidden> }),
                Rows / Tile = 1 #{!} (Rows / Tile), Rows % Tile, Member]>(tile);
        Self::store::<m![Rows % Tile], _, _>(device, logits, mask, iota, maximum, rank);
    }

    fn store<Part: M, Columns: M, Destination: M>(
        device: &mut Device,
        logits: &DmTensor<
            f32,
            Chip,
            projection::Cluster,
            projection::Reduced<Hidden>,
            m![{ Part }, VocabPad % 198],
        >,
        mask: HbmTensorView<'_, f32, Chip, Columns>,
        iota: HbmTensorView<'_, i32, Chip, Columns>,
        maximum_destination: DmTensorViewMut<
            '_,
            f32,
            Chip,
            projection::Cluster,
            projection::Reduced<Hidden>,
            Destination,
        >,
        rank_destination: DmTensorViewMut<
            '_,
            i32,
            Chip,
            projection::Cluster,
            projection::Reduced<Hidden>,
            Destination,
        >,
    ) {
        // Each adjacent pair keeps two independent maxima, so only the 99-pair axis is reduced.
        let logits: DmTensorView<
            '_,
            f32,
            Chip,
            projection::Cluster,
            projection::Reduced<Hidden>,
            m![{ Part }, Position, Member],
        > = unsafe { logits.view().reshape() };
        let mask: DmTensor<
            f32,
            Chip,
            projection::Cluster,
            projection::Reduced<Hidden>,
            m![VocabPad % 198],
        > = mask.to_dm(&mut device.tdma);
        let mask: DmTensor<
            f32,
            Chip,
            projection::Cluster,
            projection::Reduced<Hidden>,
            m![Position, Member],
        > = unsafe { mask.reshape() };
        let mask: VrfTensor<
            f32,
            Chip,
            projection::Cluster,
            projection::Reduced<Hidden>,
            m![Position, Member # 8],
        > = device
            .sub
            .begin(mask.view())
            .fetch::<m![Position], m![Member]>()
            .collect::<m![Position], m![Member # 8]>()
            .to_vrf();
        let masked: Tile<Hidden, Part> = device
            .main
            .begin(logits)
            .fetch::<m![{ Part }, Position], m![Member]>()
            .collect::<m![{ Part }, Position], m![Member # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![Member # 4]>()
            .vector_fp_binary(FpBinaryOp::AddF, &mask)
            .vector_widen_pad::<m![Member # 8]>()
            .vector_final()
            .commit_trim::<m![Member]>()
            .commit();
        let maximum: Local<f32, Hidden, Part> = device
            .main
            .begin(masked.view())
            .fetch::<m![{ Part }, Position], m![Member]>()
            .collect::<m![{ Part }, Position], m![Member # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![Member # 4]>()
            .vector_intra_slice_reduce::<Position, m![{ Part }], m![Member # 4]>(
                IntraSliceReduceOpF32::Max,
            )
            .vector_widen_pad::<m![Member # 8]>()
            .vector_final()
            .commit_trim::<m![Member]>()
            .commit();
        maximum
            .view()
            .to_dm_view(&mut device.tdma, maximum_destination);
        let maximum: VrfTensor<
            f32,
            Chip,
            projection::Cluster,
            projection::Reduced<Hidden>,
            m![{ Part }, Member # 8],
        > = device
            .sub
            .begin(maximum.view())
            .fetch::<m![{ Part }], m![Member]>()
            .collect::<m![{ Part }], m![Member # 8]>()
            .to_vrf();
        let shifted: Tile<Hidden, Part> = device
            .main
            .begin(masked.view())
            .fetch::<m![{ Part }, Position], m![Member]>()
            .collect::<m![{ Part }, Position], m![Member # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![Member # 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &maximum)
            .vector_widen_pad::<m![Member # 8]>()
            .vector_final()
            .commit_trim::<m![Member]>()
            .commit();
        let iota: DmTensor<
            i32,
            Chip,
            projection::Cluster,
            projection::Reduced<Hidden>,
            m![VocabPad % 198],
        > = iota.to_dm(&mut device.tdma);
        let iota: DmTensor<
            i32,
            Chip,
            projection::Cluster,
            projection::Reduced<Hidden>,
            m![Position, Member],
        > = unsafe { iota.reshape() };
        let iota: VrfTensor<
            i32,
            Chip,
            projection::Cluster,
            projection::Reduced<Hidden>,
            m![Position, Member # 8],
        > = device
            .sub
            .begin(iota.view())
            .fetch::<m![Position], m![Member]>()
            .collect::<m![Position], m![Member # 8]>()
            .to_vrf();
        let rank: Local<i32, Hidden, Part> = device
            .main
            .begin(shifted.view())
            .fetch::<m![{ Part }, Position], m![Member]>()
            .collect::<m![{ Part }, Position], m![Member # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>()
            .vector_fxp(FxpBinaryOp::AddFxpSat, &iota)
            .vector_narrow_trim::<m![Member # 4]>()
            .vector_intra_slice_reduce::<Position, m![{ Part }], m![Member # 4]>(
                IntraSliceReduceOpI32::Max,
            )
            .vector_widen_pad::<m![Member # 8]>()
            .vector_final()
            .commit_trim::<m![Member]>()
            .commit();
        rank.view().to_dm_view(&mut device.tdma, rank_destination);
    }

    pub(in crate::ops::vocabulary) fn select(&self, device: &mut Device) -> Selected<Rows> {
        let mut values: Gathered<f32, Rows> = DmTensor::new();
        // Entry numbers enumerate (vocabulary partition, pass, pair member) in address order.
        // Their ranks retain the original token indices through this change of coordinates.
        let maximum: DmTensorView<
            '_,
            f32,
            Chip,
            m![Entry / (Entry / 2)],
            SourceSlice<Hidden>,
            SourceElement<Hidden, Rows>,
        > = unsafe { self.maximum.view().reshape() };
        maximum.to_dm_view(&mut device.tdma, values.view_mut());
        let mut ranks: Gathered<i32, Rows> = DmTensor::new();
        let rank: DmTensorView<
            '_,
            i32,
            Chip,
            m![Entry / (Entry / 2)],
            SourceSlice<Hidden>,
            SourceElement<Hidden, Rows>,
        > = unsafe { self.rank.view().reshape() };
        rank.to_dm_view(&mut device.tdma, ranks.view_mut());
        let maximum: DmTensor<
            f32,
            Chip,
            Cluster<Rows>,
            m![{ Layout<Rows>::Lanes } # 8 % 4, 64],
            m![{ Groups<Rows> }, 1 # 8],
        > = device
            .main
            .begin(values.view())
            .fetch::<m![{ Groups<Rows> }, Entry % 24 / 8], m![Entry % 8]>()
            .collect::<m![{ Groups<Rows> }, Entry % 24 / 8], m![Entry % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Groups<Rows> }, Entry % 24 / 4], m![Entry % 4]>()
            .vector_intra_slice_reduce::<Entry, m![{ Groups<Rows> }], m![1 # 4]>(
                IntraSliceReduceOpF32::Max,
            )
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<m![{ Layout<Rows>::Lanes } # 8 % 4, 64], m![{ Groups<Rows> }]>(
                InterSliceReduceOpF32::Max,
            )
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        // Every vocabulary slice received the same row maximum; its physical address is unchanged.
        let maximum: DmTensor<
            f32,
            Chip,
            Cluster<Rows>,
            CandidateSlices<Rows>,
            m![{ Groups<Rows> }, 1 # 8],
        > = unsafe { maximum.reshape() };
        let maximum: VrfTensor<
            f32,
            Chip,
            Cluster<Rows>,
            CandidateSlices<Rows>,
            m![{ Groups<Rows> }, 1 # 8],
        > = device
            .sub
            .begin(maximum.view())
            .fetch::<m![{ Groups<Rows> }], m![1 # 8]>()
            .collect::<m![{ Groups<Rows> }], m![1 # 8]>()
            .to_vrf();
        let shifted: Gathered<f32, Rows> = device
            .main
            .begin(values.view())
            .fetch::<m![{ Groups<Rows> }, Entry % 24 / 8], m![Entry % 8]>()
            .collect::<m![{ Groups<Rows> }, Entry % 24 / 8], m![Entry % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Groups<Rows> }, Entry % 24 / 4], m![Entry % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &maximum)
            .vector_widen_concat::<m![{ Groups<Rows> }, Entry % 24 / 8], m![Entry % 8]>()
            .vector_final()
            .commit_trim::<m![Entry % 8]>()
            .commit();
        let ranks: VrfTensor<
            i32,
            Chip,
            Cluster<Rows>,
            CandidateSlices<Rows>,
            m![{ Groups<Rows> }, Entry % 24],
        > = device
            .sub
            .begin(ranks.view())
            .fetch::<m![{ Groups<Rows> }, Entry % 24 / 8], m![Entry % 8]>()
            .collect::<m![{ Groups<Rows> }, Entry % 24 / 8], m![Entry % 8]>()
            .to_vrf();
        let rank: DmTensor<i32, Chip, Cluster<Rows>, Reduced<Rows>, m![{ Groups<Rows> }, 1 # 8]> =
            device
                .main
                .begin(shifted.view())
                .fetch::<m![{ Groups<Rows> }, Entry % 24 / 8], m![Entry % 8]>()
                .collect::<m![{ Groups<Rows> }, Entry % 24 / 8], m![Entry % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_reinterpret::<i32>()
                .vector_fxp(FxpBinaryOp::AddFxpSat, &ranks)
                .vector_narrow_split::<m![{ Groups<Rows> }, Entry % 24 / 4], m![Entry % 4]>()
                .vector_intra_slice_reduce::<Entry, m![{ Groups<Rows> }], m![1 # 4]>(
                    IntraSliceReduceOpI32::Max,
                )
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<Reduced<Rows>, m![{ Groups<Rows> }]>(
                    InterSliceReduceOpI32::Max,
                )
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
        device
            .main
            .begin(rank.view())
            .fetch::<m![{ Groups<Rows> }], m![1 # 8]>()
            .fetch_cast::<i32>()
            .collect::<m![{ Groups<Rows> }], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp_with_mode(
                FxpBinaryOp::SubFxp,
                BinaryArgMode::Mode10,
                const { VocabPad::SIZE as i32 - 1 },
            )
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit()
    }
}
