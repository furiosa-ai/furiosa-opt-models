//! The BF16 form of the output projection: a dense matrix the contraction reads where it is
//! stored, over each span's live attended rows where they are stored, whose projection lands as
//! the span's rows.

use super::*;
use crate::modules::lora;

/// A launch's attended rows in HBM: `Groups` query heads of `Head` features for each of `KvHeads`.
type Attended<Reach, KvHeads, Groups, Head> =
    HbmTensor<bf16, Chip, m![Reach, KvHeads, Groups, Head]>;

/// The dense projection lands nothing before the attended rows arrive: the contraction reads each
/// span's weight slabs from HBM, so no live layer holds BF16 weights in DM.
impl<Hidden: AxisName, QWidth: AxisName, Env> Module<Begin<()>, Env, &mut ()>
    for Affine<HbmTensor<bf16, Chip, m![Hidden, QWidth]>, ()>
{
    type Output = ();

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) {}
}

/// Nor does it stage anything while an attention's last history is in flight.
impl<Hidden: AxisName, QWidth: AxisName, Env> Module<&mut (), Env, &mut ()>
    for Affine<HbmTensor<bf16, Chip, m![Hidden, QWidth]>, ()>
{
    type Output = ();

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: &mut ()) {}
}

/// Span tile `attended` of the attended rows as the steps take it, with what the projection landed.
type Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head> = (
    &'l mut (),
    Live<&'a Attended<Reach, KvHeads, Groups, Head>, Capacity>,
);

/// The dense projection takes the block-scaled form's steps, its first two doing nothing: begun,
/// a tile is given back for its second step...
impl<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head, Env>
    Module<Begin<Begin<Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>>>, Env, &mut ()>
    for Affine<HbmTensor<bf16, Chip, m![Hidden, QWidth]>, ()>
where
    Hidden: AxisName,
    QWidth: AxisName,
    Capacity: AxisName,
    Reach: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
{
    type Output = Begin<Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        Begin(tile): Begin<Begin<Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>>>,
    ) -> Self::Output {
        tile
    }
}

/// ...and after it for its last.
impl<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head, Env>
    Module<Begin<Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>>, Env, &mut ()>
    for Affine<HbmTensor<bf16, Chip, m![Hidden, QWidth]>, ()>
where
    Hidden: AxisName,
    QWidth: AxisName,
    Capacity: AxisName,
    Reach: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
{
    type Output = Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        Begin(tile): Begin<Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>>,
    ) -> Self::Output {
        tile
    }
}

/// A tile's last step: the tile's live rows projected, through HBM into the span's rows.
impl<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head, Env: Copy>
    Module<Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>, Env, &mut ()>
    for Affine<HbmTensor<bf16, Chip, m![Hidden, QWidth]>, ()>
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
        device: &mut Device,
        env: Env,
        _: &mut (),
        ((), attended): Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>,
    ) -> Self::Output {
        const { assert!(KvHeads::SIZE * Groups::SIZE * Head::SIZE == QWidth::SIZE) };
        let tile = attended.with(());
        // SAFETY: QWidth is the same contiguous KV-head, query-group and head product.
        let attended: HbmTensor<bf16, Chip, m![Reach, QWidth]> =
            unsafe { attended.into_value().view().alias() };
        let live = Live::<(), Reach>::all().subrange(tile.start(), tile.rows());
        let mut lines = HbmTensor::<bf16, Chip, m![Span, Hidden]>::new();
        Affine::new(&self.weight, ()).forward(
            &mut *device,
            env,
            &mut (),
            (live, &attended, &mut lines),
        );
        lines.to_dm(&mut device.tdma)
    }
}

/// A tile's last step in a call that runs adapters: the tile's live rows projected, plus each
/// adapter's update of them, through HBM into the span's rows.
impl<
    's,
    'l,
    'a,
    't,
    Hidden,
    QWidth,
    Capacity,
    Reach,
    KvHeads,
    Groups,
    Head,
    Env,
    A,
    B,
    K,
    const N: usize,
>
    Module<
        Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>,
        Env,
        &'s mut lora::Adapters<'t, lora::Term<A, B>, HbmTensor<bf16, Chip, m![Capacity, K]>, N>,
    > for Affine<HbmTensor<bf16, Chip, m![Hidden, QWidth]>, ()>
where
    Hidden: AxisName,
    QWidth: AxisName,
    Capacity: AxisName,
    Reach: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Env: Copy,
    K: AxisName,
    Affine<A, ()>: for<'i, 'o> crate::Op<
            (
                Live<(), Reach>,
                &'i HbmTensor<bf16, Chip, m![Reach, QWidth]>,
                &'o mut HbmTensor<bf16, Chip, m![Span, K]>,
            ),
            Output = (),
        >,
    Affine<B, ()>: for<'r, 'o> crate::Op<
            (
                Live<(), Span>,
                &'r HbmTensor<bf16, Chip, m![Span, K]>,
                &'o mut HbmTensor<bf16, Chip, m![Span, Hidden]>,
            ),
            Output = (),
        >,
{
    type Output = Owned<Hidden, Span>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        adapters: &'s mut lora::Adapters<
            't,
            lora::Term<A, B>,
            HbmTensor<bf16, Chip, m![Capacity, K]>,
            N,
        >,
        ((), attended): Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>,
    ) -> Self::Output {
        const { assert!(KvHeads::SIZE * Groups::SIZE * Head::SIZE == QWidth::SIZE) };
        let tile = attended.with(());
        // SAFETY: as for the tile's projection without adapters.
        let attended: HbmTensor<bf16, Chip, m![Reach, QWidth]> =
            unsafe { attended.into_value().view().alias() };
        let live = Live::<(), Reach>::all().subrange(tile.start(), tile.rows());
        let mut lines = HbmTensor::<bf16, Chip, m![Span, Hidden]>::new();
        Affine::new(&self.weight, ()).forward(
            &mut *device,
            env,
            &mut (),
            (live, &attended, &mut lines),
        );
        let lines = adapters.add(device, live, &attended, tile.start(), lines);
        lines.to_dm(&mut device.tdma)
    }
}

/// Span tile `tile` of the attended rows projected in one poll.
impl<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head, Env: Copy>
    Module<
        (
            &'l mut (),
            &'a Attended<Reach, KvHeads, Groups, Head>,
            Live<(), Capacity>,
        ),
        Env,
        &mut (),
    > for Affine<HbmTensor<bf16, Chip, m![Hidden, QWidth]>, ()>
where
    Hidden: AxisName,
    QWidth: AxisName,
    Capacity: AxisName,
    Reach: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
{
    type Output = Poll<Owned<Hidden, Span>>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (landed, attended, tile): (
            &'l mut (),
            &'a Attended<Reach, KvHeads, Groups, Head>,
            Live<(), Capacity>,
        ),
    ) -> Self::Output {
        Poll::Ready(self.forward(device, env, &mut (), (landed, tile.with(attended))))
    }
}
