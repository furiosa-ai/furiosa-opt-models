//! The dense forms of the output projection: a matrix the contraction reads as BF16 tiles where it
//! is stored, BF16 coefficients or E4M3 codes it restores by their rows' scales, over each span's
//! live attended rows where they are stored, whose projection lands as the span's rows.

use super::*;
use crate::ops::linear::{Matrix, fp8};
use crate::tensor::form::Weight;
use crate::tensor::memory::{Hbm, Memory};

/// A launch's attended rows in HBM: `Groups` query heads of `Head` features for each of `KvHeads`.
type Attended<Reach, KvHeads, Groups, Head> =
    HbmTensor<bf16, Chip, m![Reach, KvHeads, Groups, Head]>;

/// The output projection's channel-scaled weight: `Hidden` rows over `QWidth` inputs, the E4M3
/// codes and the FP32 scale of each row in memory `X`.
pub type ChannelScaled<QWidth, Hidden, X = Hbm> = Weight<
    m![Hidden, QWidth],
    Chip,
    fp8::ChannelScaled<
        QWidth,
        Hidden,
        <X as Memory>::Tensor<f8e4m3, Chip, m![Hidden, QWidth]>,
        <X as Memory>::Tensor<f32, Chip, m![Hidden, 1 # 8]>,
    >,
>;

impl<W> Affine<W, ()> {
    /// Span tile `tile` of the `attended` rows projected through HBM into the span's rows: the
    /// tile's live rows only.
    fn span<QWidth, Hidden, Capacity, Reach, KvHeads, Groups, Head, Env: Copy>(
        &self,
        device: &mut Device,
        env: Env,
        attended: &Attended<Reach, KvHeads, Groups, Head>,
        tile: Live<(), Capacity>,
    ) -> Owned<Hidden, Span>
    where
        W: Matrix<In = QWidth, Out = Hidden>,
        QWidth: AxisName,
        Hidden: AxisName,
        Capacity: AxisName,
        Reach: AxisName,
        KvHeads: AxisName,
        Groups: AxisName,
        Head: AxisName,
    {
        const { assert!(KvHeads::SIZE * Groups::SIZE * Head::SIZE == QWidth::SIZE) };
        // SAFETY: QWidth is the same contiguous KV-head, query-group and head product.
        let attended: HbmTensor<bf16, Chip, m![Reach, QWidth]> = unsafe { attended.view().alias() };
        let live = Live::<(), Reach>::all().subrange(tile.start(), tile.rows());
        let mut lines = HbmTensor::<bf16, Chip, m![Span, Hidden]>::new();
        self.forward(&mut *device, env, &mut (), (live, &attended, &mut lines));
        lines.to_dm(&mut device.tdma)
    }
}

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
        Poll::Ready(Affine::new(&self.weight, ()).span(device, env, attended, tile))
    }
}

/// The channel-scaled projection takes the dense projection's steps over its codes: it lands
/// nothing before the attended rows arrive...
impl<Hidden: AxisName, QWidth: AxisName, Env> Module<Begin<()>, Env, &mut ()>
    for ChannelScaled<QWidth, Hidden>
{
    type Output = ();

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) {}
}

/// ...nor while an attention's last history is in flight...
impl<Hidden: AxisName, QWidth: AxisName, Env> Module<&mut (), Env, &mut ()>
    for ChannelScaled<QWidth, Hidden>
{
    type Output = ();

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: &mut ()) {}
}

/// ...and projects a span tile's live attended rows in one poll, restoring each weight slab from
/// its codes.
impl<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head, Env: Copy>
    Module<
        (
            &'l mut (),
            &'a Attended<Reach, KvHeads, Groups, Head>,
            Live<(), Capacity>,
        ),
        Env,
        &mut (),
    > for ChannelScaled<QWidth, Hidden>
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
        Poll::Ready(Affine::new(self.held(), ()).span(device, env, attended, tile))
    }
}
