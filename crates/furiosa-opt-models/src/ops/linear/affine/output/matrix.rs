//! The BF16 form of the output projection: a dense matrix the contraction reads where it is
//! stored, over each span's live attended rows where they are stored, whose projection lands as
//! the span's rows.

use super::*;

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

/// Span tile `tile` of the attended rows projected in one poll, through HBM into the span's rows.
/// It projects the tile's live rows only.
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
        ((), attended, tile): (
            &'l mut (),
            &'a Attended<Reach, KvHeads, Groups, Head>,
            Live<(), Capacity>,
        ),
    ) -> Self::Output {
        const { assert!(KvHeads::SIZE * Groups::SIZE * Head::SIZE == QWidth::SIZE) };
        // SAFETY: QWidth is the same contiguous KV-head, query-group and head product.
        let attended: HbmTensor<bf16, Chip, m![Reach, QWidth]> = unsafe { attended.view().alias() };
        let live = Live::<(), Reach>::all().subrange(tile.start(), tile.rows());
        let mut lines = HbmTensor::<bf16, Chip, m![Span, Hidden]>::new();
        Affine::new(&self.weight, ()).forward(
            &mut *device,
            env,
            &mut (),
            (live, &attended, &mut lines),
        );
        Poll::Ready(lines.to_dm(&mut device.tdma))
    }
}
