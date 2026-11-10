use super::*;
use std::borrow::Borrow;

impl<Hidden: AxisName, QWidth: AxisName, W, S> Op<Begin<()>>
    for Affine<Rows<m![QWidth], m![Hidden], W, S>, ()>
{
    type Output = Landed<Hidden, QWidth, Span>;

    fn forward(&self, _: &mut Device, _: Begin<()>) -> Self::Output {
        Landed {
            staged: Affine::new((Weight::new(), Channels::new()), ()),
            step: 0,
            first: None,
            second: None,
        }
    }
}

impl<Hidden: AxisName, QWidth: AxisName, W, S> Op<&mut Landed<Hidden, QWidth, Span>>
    for Affine<Rows<m![QWidth], m![Hidden], W, S>, ()>
where
    W: Borrow<HbmTensor<f8e4m3, Chip, Blocked<Hidden, QWidth>>>,
    S: Borrow<HbmTensor<f32, Chip, m![Hidden]>>,
{
    type Output = ();

    fn forward(&self, context: &mut Device, landed: &mut Landed<Hidden, QWidth, Span>) {
        let (weight, channels) = &mut landed.staged.weight;
        self.weight
            .0
            .borrow()
            .view()
            .to_dm_view(&mut context.tdma, weight.view_mut());
        self.weight
            .1
            .borrow()
            .view()
            .to_dm_view(&mut context.tdma, channels.view_mut());
    }
}

impl<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head, W, S>
    Op<Begin<Begin<Tile<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head>>>>
    for Affine<Rows<m![QWidth], m![Hidden], W, S>, ()>
where
    Hidden: AxisName,
    QWidth: AxisName,
    Capacity: AxisName,
    Reach: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
{
    type Output = Begin<Tile<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head>>;

    fn forward(
        &self,
        context: &mut Device,
        Begin(Begin((landed, attended))): Begin<
            Begin<Tile<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head>>,
        >,
    ) -> Self::Output {
        let _ = Op::forward(
            self,
            context,
            (&mut *landed, attended.into_value(), attended.with(())),
        );
        Begin((landed, attended))
    }
}

impl<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head, W, S>
    Op<Begin<Tile<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head>>>
    for Affine<Rows<m![QWidth], m![Hidden], W, S>, ()>
where
    Hidden: AxisName,
    QWidth: AxisName,
    Capacity: AxisName,
    Reach: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
{
    type Output = Tile<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head>;

    fn forward(
        &self,
        context: &mut Device,
        Begin((landed, attended)): Begin<
            Tile<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head>,
        >,
    ) -> Self::Output {
        let _ = Op::forward(
            self,
            context,
            (&mut *landed, attended.into_value(), attended.with(())),
        );
        (landed, attended)
    }
}

impl<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head, W, S>
    Op<Tile<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head>>
    for Affine<Rows<m![QWidth], m![Hidden], W, S>, ()>
where
    Hidden: AxisName,
    QWidth: AxisName,
    Capacity: AxisName,
    Reach: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
{
    type Output = Owned<Hidden, Span>;

    fn forward(
        &self,
        context: &mut Device,
        (landed, attended): Tile<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head>,
    ) -> Self::Output {
        match Op::forward(
            self,
            context,
            (landed, attended.into_value(), attended.with(())),
        ) {
            Poll::Ready(rows) => rows,
            Poll::Pending => Owned::new(),
        }
    }
}
