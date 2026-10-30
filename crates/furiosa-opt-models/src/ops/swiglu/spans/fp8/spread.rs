use super::*;
use crate::Op;
use crate::ops::vector::RowQuantize;

pub(super) type SpreadScale<Intermediate, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<Intermediate>,
    m![Rows / (Rows / 8), Copies],
    m![Panels, Rows % (Rows / 8)],
>;
pub(super) type ChunkMax<Intermediate, Rows, P> =
    DmTensor<bf16, Chip, Cluster<Intermediate>, P, m![Panels, Rows % (Rows / 8)]>;

/// Each row's scale over panel `P` of the products, in its panel of the chunk maxima.
pub(super) struct PanelMaxima<const P: usize>;

impl<'a, Intermediate: AxisName, Rows: AxisName, const P: usize>
    Op<(
        &'a Halves<Intermediate, Rows>,
        &'a mut ChunkMax<Intermediate, Rows, Slice<Intermediate, Rows>>,
    )> for PanelMaxima<P>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (product, chunks): (
            &'a Halves<Intermediate, Rows>,
            &'a mut ChunkMax<Intermediate, Rows, Slice<Intermediate, Rows>>,
        ),
    ) {
        // SAFETY: a chunk's panel digit and a slice's row digits name the same coordinates.
        let spread = unsafe {
            product.view().reshape::<Chip, Cluster<Intermediate>, Slice<Intermediate, Rows>,
                m![Panels, Rows % (Rows / 8), Intermediate % 64]>()
        };
        context
            .main
            .begin(
                spread
                    .tile::<
                        m![Panels],
                        1,
                        m![Panels = 1 # Panels, Rows % (Rows / 8), Intermediate % 64],
                    >(P),
            )
            .fetch::<m![Rows % (Rows / 8), Intermediate % 64 / 8], m![Intermediate % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % (Rows / 8), Intermediate % 64 / 8], m![Intermediate % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![Rows % (Rows / 8), Intermediate % 64 / 4],
                m![Intermediate % 4],
            >()
            .vector_row_scale::<Intermediate, m![Rows % (Rows / 8)]>()
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![Rows % (Rows / 8) / 4], m![Rows % 4 # 16]>()
            .commit_trim::<m![Rows % 4]>()
            .commit_view(
                chunks
                    .view_mut()
                    .tile::<m![Panels], 1, m![Panels = 1 #{!} Panels, Rows % (Rows / 8)]>(P),
            );
    }
}

/// Each row group's scales, the maximum over its slices, in `groups` and again in every slice of
/// the group.
pub(super) struct GroupScales;

impl<'a, Intermediate: AxisName, Rows: AxisName>
    Op<(
        &'a ChunkMax<Intermediate, Rows, Slice<Intermediate, Rows>>,
        &'a mut SpreadScale<Intermediate, Rows>,
    )> for GroupScales
{
    type Output = ChunkMax<Intermediate, Rows, Slice<Intermediate, Rows>>;

    fn forward(
        &self,
        context: &mut Device,
        (chunks, groups): (
            &'a ChunkMax<Intermediate, Rows, Slice<Intermediate, Rows>>,
            &'a mut SpreadScale<Intermediate, Rows>,
        ),
    ) -> Self::Output {
        let mut inner =
            ChunkMax::<Intermediate, Rows, m![Rows / (Rows / 8), { Blocks<Intermediate> }]>::new();
        chunks
            .view()
            .to_dm_view(&mut context.tdma, inner.view_mut());
        context.main.begin(inner.view())
            .fetch::<m![Panels, Rows % (Rows / 8) / 8], m![Rows % 8]>().fetch_cast::<f32>()
            .collect::<m![Panels, Rows % (Rows / 8) / 8], m![Rows % 8]>()
            .vector_init()
            .vector_inter_slice_reduce::<m![Rows / (Rows / 8), Copies],
                m![Panels, Rows % (Rows / 8) / 8]>(InterSliceReduceOpF32::Max)
            .vector_final().cast::<bf16, m![Rows % 8 # 16]>()
            .commit_trim::<m![Rows % 8]>().commit_view(groups.view_mut());
        // SAFETY: the reduction copies each row group's scales to every slice of the group.
        let copies = unsafe {
            groups.view().reshape::<Chip, Cluster<Intermediate>,
                m![Rows / (Rows / 8), { Blocks<Intermediate> }], m![Panels, Rows % (Rows / 8)]>()
        };
        let mut broadcast = ChunkMax::<Intermediate, Rows, Slice<Intermediate, Rows>>::new();
        copies.to_dm_view(&mut context.tdma, broadcast.view_mut());
        broadcast
    }
}

/// Panel `P` of the products as E4M3, divided by the row scales, in its panel of the quantized
/// rows.
pub(super) struct QuantizeSpread<const P: usize>;

/// The products, the broadcast row scales and the quantized rows a [`QuantizeSpread`] writes.
type Spreads<'a, Intermediate, Rows> = (
    &'a Halves<Intermediate, Rows>,
    &'a ChunkMax<Intermediate, Rows, Slice<Intermediate, Rows>>,
    &'a mut Spread<Intermediate, Rows, f8e4m3>,
);

impl<'a, Intermediate: AxisName, Rows: AxisName, const P: usize> Op<Spreads<'a, Intermediate, Rows>>
    for QuantizeSpread<P>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (product, broadcast, quantized): Spreads<'a, Intermediate, Rows>,
    ) {
        let scale: VrfTensor<
            f32,
            Chip,
            Cluster<Intermediate>,
            Slice<Intermediate, Rows>,
            m![Rows % (Rows / 8)],
        > = context
            .sub
            .begin(
                broadcast
                    .view()
                    .tile::<m![Panels], 1, m![Panels = 1 # Panels, Rows % (Rows / 8)]>(P),
            )
            .fetch::<m![Rows % (Rows / 8) / 8], m![Rows % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % (Rows / 8) / 8], m![Rows % 8]>()
            .to_vrf();
        // SAFETY: a chunk's panel digit and a slice's row digits name the same coordinates.
        let spread = unsafe {
            product.view().reshape::<Chip, Cluster<Intermediate>, Slice<Intermediate, Rows>,
                m![Panels, Rows % (Rows / 8), Intermediate % 64]>()
        };
        context
            .main
            .begin(
                spread
                    .tile::<
                        m![Panels],
                        1,
                        m![Panels = 1 # Panels, Rows % (Rows / 8), Intermediate % 64],
                    >(P),
            )
            .fetch::<m![Rows % (Rows / 8), Intermediate % 64 / 8], m![Intermediate % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % (Rows / 8), Intermediate % 64 / 8], m![Intermediate % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![Rows % (Rows / 8), Intermediate % 64 / 4],
                m![Intermediate % 4],
            >()
            .vector_quantize::<
                m![Rows % (Rows / 8), Intermediate % 64 / 8],
                m![Intermediate % 8],
                _,
            >(&scale)
            .commit_view(
                quantized
                    .view_mut()
                    .tile::<
                        m![Panels],
                        1,
                        m![Panels = 1 #{!} Panels, Rows % (Rows / 8), Intermediate % 64],
                    >(
                        P,
                    ),
            );
    }
}
