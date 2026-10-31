//! The dense forms of the output projection: a matrix the contraction reads as BF16 tiles where it
//! is stored, BF16 coefficients or E4M3 codes it restores by their rows' scales, over each span's
//! live attended rows where they are stored, whose projection lands as the span's rows.

use super::*;
use crate::modules::lora;
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
        let tile = attended.with(());
        Affine::new(&self.weight, ()).span(device, env, attended.into_value(), tile)
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

/// ...gives a begun tile back for its second step and after it for its last...
impl<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head, Env>
    Module<Begin<Begin<Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>>>, Env, &mut ()>
    for ChannelScaled<QWidth, Hidden>
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

impl<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head, Env>
    Module<Begin<Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>>, Env, &mut ()>
    for ChannelScaled<QWidth, Hidden>
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

/// ...and projects the tile's live attended rows in the last, restoring each weight slab from its
/// codes.
impl<'l, 'a, Hidden, QWidth, Capacity, Reach, KvHeads, Groups, Head, Env: Copy>
    Module<Tile<'l, 'a, Capacity, Reach, KvHeads, Groups, Head>, Env, &mut ()>
    for ChannelScaled<QWidth, Hidden>
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
        let tile = attended.with(());
        Affine::new(self.held(), ()).span(device, env, attended.into_value(), tile)
    }
}
