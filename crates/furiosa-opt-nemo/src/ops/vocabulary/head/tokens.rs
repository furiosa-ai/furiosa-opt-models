use super::*;
use crate::Module;
use crate::ops::normalization::Spread;
use crate::ops::vocabulary::{Greedy, LmHead};
use std::borrow::Borrow;

pub(crate) struct Wave<'a, Hidden: AxisName, Rows: Grouped> {
    pub input: &'a Spread<Hidden, Rows>,
}

impl<
    Hidden: Project,
    Rows: Grouped,
    Parameter: Borrow<Weight<m![VocabPad, Hidden]>>,
    Mask: Borrow<Columns<Hidden, f32>>,
    Iota: Borrow<Columns<Hidden, i32>>,
> Module<(&mut Device, Wave<'_, Hidden, Rows>)> for Greedy<Hidden, Parameter, Mask, Iota>
{
    type Output = Selected<Rows>;

    fn forward(
        &self,
        (context, Wave { input }): (&mut Device, Wave<'_, Hidden, Rows>),
    ) -> Self::Output {
        let lanes = super::super::projection::Input::<
            Hidden,
            Rows,
            Groups<Rows>,
            Hidden::Chunk,
            Slice<Hidden>,
        >(core::marker::PhantomData)
        .forward((context, input.view()));
        let first = load_weight(context, self.head.0.borrow(), 0);
        select_lanes::<Hidden, Rows>(
            context,
            self.head.0.borrow(),
            &lanes,
            self.mask.borrow(),
            self.iota.borrow(),
            first,
        )
    }
}

pub(crate) struct Projection<'a, Hidden: Project, Rows: Grouped> {
    pub(crate) lanes: &'a TrfTensor<
        bf16,
        Chip,
        Cluster,
        Slice<Hidden>,
        m![{ Layout<Rows>::Lanes }],
        m![{ Groups<Rows> }, Hidden % ({ Hidden::Chunk })],
    >,
    pub(crate) mask: DmTensorView<'a, f32, Chip, Cluster, Reduced<Hidden>, Column<Hidden>>,
}

impl<Hidden: Project, Rows: Grouped, Parameter: Borrow<PassWeight<Hidden>>>
    Module<(&mut Device, Projection<'_, Hidden, Rows>)> for LmHead<Hidden, Parameter>
{
    type Output = DmTensor<
        f32,
        Chip,
        Cluster,
        Reduced<Hidden>,
        m![VocabPad % ({ PassRows }), { Groups<Rows> } # 17, { Layout<Rows>::Lanes } # 8],
    >;

    fn forward(
        &self,
        (context, input): (&mut Device, Projection<'_, Hidden, Rows>),
    ) -> Self::Output {
        project::<Hidden, Rows, Groups<Rows>, 17>(context, self.0.borrow(), input.lanes, input.mask)
    }
}

impl<Hidden: Project> Greedy<Hidden> {
    pub(crate) fn stage(
        &self,
        context: &mut Device,
    ) -> Greedy<Hidden, &Weight<m![VocabPad, Hidden]>, Columns<Hidden, f32>, Columns<Hidden, i32>>
    {
        Greedy {
            head: LmHead(&self.head.0, core::marker::PhantomData),
            mask: self.mask.view().to_dm(&mut context.tdma),
            iota: self.iota.view().to_dm(&mut context.tdma),
        }
    }
}

/// Wave `index` of a span's tokens, stored where its rows sit among the launch's selected rows.
impl<'a, Capacity: AxisName, R: Grouped>
    crate::compose::Spill<
        Device,
        &'a Selected<R>,
        (
            crate::compose::Tiles<
                crate::compose::Tile<Capacity, crate::ops::normalization::span::Span>,
            >,
            usize,
        ),
    > for HbmTensor<i32, Chip, m![Capacity]>
{
    fn spill(
        &mut self,
        context: &mut Device,
        tokens: &'a Selected<R>,
        (tile, index): (
            crate::compose::Tiles<
                crate::compose::Tile<Capacity, crate::ops::normalization::span::Span>,
            >,
            usize,
        ),
    ) {
        tokens.view().to_hbm_view(
            &mut context.tdma,
            self.view_mut().tile::<m![Capacity / R], 1, m![
                Capacity / R = 1 #{!} (Capacity / R), Capacity % R
            ]>(tile.start() / R::SIZE + index),
        );
    }
}
