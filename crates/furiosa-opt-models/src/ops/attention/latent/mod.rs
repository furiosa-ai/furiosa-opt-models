//! Compressed attention retains one latent vector and a rotary key per cached token.

use crate::Op;
use crate::ops::index::table::Logical;
use crate::tensor::axis::Single;
use crate::tensor::cache::Block;
use crate::tensor::{live::Live, parallel::Chip};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Concatenated latent and rotary dimensions.
#[derive(Clone, Debug)]
pub struct Combined<L, R>(PhantomData<(L, R)>);

impl<L: AxisName, R: AxisName> AxisName for Combined<L, R> {
    const NAME: Ident = Ident::new("Combined");
    const SIZE: usize = L::SIZE + R::SIZE;
}

/// Packed per-head nonrotary key and value projection width.
#[derive(Clone, Debug)]
pub struct ProjectionWidth<H, N, V>(PhantomData<(H, N, V)>);

impl<H: AxisName, N: AxisName, V: AxisName> AxisName for ProjectionWidth<H, N, V> {
    const NAME: Ident = Ident::new("ProjectionWidth");
    const SIZE: usize = H::SIZE * (N::SIZE + V::SIZE);
}

type Rows<H, D> = HbmTensor<bf16, m![1], m![H, D]>;

/// Latent and rotary query channels, with one row per query head.
pub type Query<H, L, R> = (Rows<H, L>, Rows<H, R>);

/// Logical token slot within blocked latent storage.
pub type Slot = HbmTensor<i32, m![1], m![Single]>;

/// The first L channels are both the latent key and value; the last R contain the rotary key.
#[derive(DeviceSend)]
pub struct LatentKeyValue<L: AxisName, R: AxisName> {
    pub blocks: Slice<HbmTensor<bf16, m![1], m![Block, { Symbol<Combined<L, R>> }]>>,
}

/// Additive attention bias per logical history token.
pub type Bias = Slice<HbmTensor<f32, m![1], m![Single]>>;

/// Block IDs of a history of at most `T` tokens.
#[derive(Clone, Debug)]
pub struct Table<T>(PhantomData<fn() -> T>);

impl<T: AxisName> AxisName for Table<T> {
    const NAME: Ident = Ident::new("Table");
    const SIZE: usize = T::SIZE.div_ceil(Block::SIZE);
}

/// Block IDs, token count and optional additive mask for a latent attention history.
#[derive(DeviceSend)]
pub struct History<Blocks: AxisName, B = ()> {
    /// Physical block IDs in logical history order; those past the visible tokens are unread.
    pub blocks: HbmTensor<i32, m![1], m![Blocks]>,
    /// Number of logical tokens visible to attention.
    pub length: Slot,
    /// Per-token additive mask, or `()` for an unmasked history.
    pub bias: B,
}

/// The latent geometry of a step's preparation: latent attention reads logical slots and block
/// IDs, so it has no constants.
#[derive(Default, DeviceSend)]
pub struct Prepare;

/// The latent read form of a one-row step from its logical rows and its history's flat table of
/// block IDs: the row's logical position and slot, its visible length and its block IDs.
impl<R: AxisName, B: AxisName> Op<(Live<(), R>, (&Logical<R>, &HbmTensor<i32, Chip, m![B]>))>
    for Prepare
{
    type Output = (Slot, Slot, History<B>);

    fn forward(
        &self,
        device: &mut Device,
        (tiles, (logical, table)): (Live<(), R>, (&Logical<R>, &HbmTensor<i32, Chip, m![B]>)),
    ) -> Self::Output {
        let mut positions: HbmTensor<i32, m![1], m![{ Symbol<R> }]> = HbmTensor::new();
        let mut writes: HbmTensor<i32, m![1], m![{ Symbol<R> }]> = HbmTensor::new();
        let mut lengths: HbmTensor<i32, m![1], m![{ Symbol<R> }]> = HbmTensor::new();
        let mut blocks: HbmTensor<i32, m![1], m![{ Symbol<R> }, B]> = HbmTensor::new();
        for row in 0..tiles.rows() {
            let input = logical.row(device, row);
            input.position.view().to_hbm_view(
                &mut device.tdma,
                positions
                    .view_mut()
                    .tile::<m![{ Symbol<R> }], 1, m![{ Symbol<R> } = 1 #{!} ({ Symbol<R> })]>(row),
            );
            input.write.view().to_hbm_view(
                &mut device.tdma,
                writes
                    .view_mut()
                    .tile::<m![{ Symbol<R> }], 1, m![{ Symbol<R> } = 1 #{!} ({ Symbol<R> })]>(row),
            );
            input.length.view().to_hbm_view(
                &mut device.tdma,
                lengths
                    .view_mut()
                    .tile::<m![{ Symbol<R> }], 1, m![{ Symbol<R> } = 1 #{!} ({ Symbol<R> })]>(row),
            );
            let start = input.table.extent::<m![B]>();
            let visible = (input.length.extent::<m![B, Block]>() + Block::SIZE - 1) / Block::SIZE;
            for block in 0..visible {
                table.view().tile::<m![B], 1, m![B = 1 # B]>(start + block)
                    .to_hbm_view(&mut device.tdma, blocks.view_mut()
                        .tile::<m![{ Symbol<R> }], 1,
                            m![{ Symbol<R> } = 1 #{!} ({ Symbol<R> }), B]>(row)
                        .tile::<m![B], 1,
                            m![{ Symbol<R> } = 1 #{!} ({ Symbol<R> }),
                                B = 1 #{!} B]>(block));
            }
        }
        // SAFETY: each reshape only drops a one-row axis, which keeps the wire order; another
        // envelope fails to compile, since each reshape asserts its sizes.
        unsafe {
            (
                positions.reshape(),
                writes.reshape(),
                History {
                    blocks: blocks.reshape(),
                    length: lengths.reshape(),
                    bias: (),
                },
            )
        }
    }
}

/// Multiply each query's nonrotary channels by the corresponding key up-projection.
#[derive(DeviceSend)]
pub struct Key<H: AxisName, N: AxisName, V: AxisName, L: AxisName> {
    pub weight: HbmTensor<bf16, m![1], m![{ Symbol<ProjectionWidth<H, N, V>> }, L]>,
}

impl<H: AxisName, N: AxisName, V: AxisName, L: AxisName> Op<&Rows<H, N>> for Key<H, N, V, L> {
    type Output = Rows<H, L>;

    fn forward(&self, _: &mut Device, _: &Rows<H, N>) -> Self::Output {
        todo!("NPU output[h,l] = sum_n input[h,n] * weight[h*(N+V)+n,l], BF16 output")
    }
}

/// Use the value part of the same packed key/value up-projection after latent attention.
#[derive(DeviceSend)]
pub struct Value<H: AxisName, N: AxisName, V: AxisName, L: AxisName> {
    pub weight: HbmTensor<bf16, m![1], m![{ Symbol<ProjectionWidth<H, N, V>> }, L]>,
}

impl<H: AxisName, N: AxisName, V: AxisName, L: AxisName> Op<&Rows<H, L>> for Value<H, N, V, L> {
    type Output = Rows<H, V>;

    fn forward(&self, _: &mut Device, _: &Rows<H, L>) -> Self::Output {
        todo!("NPU output[h,v] = sum_l input[h,l] * weight[h*(N+V)+N+v,l], BF16 output")
    }
}

/// Store one latent vector followed by its rotary key, leaving other token slots unchanged.
#[derive(DeviceSend)]
pub struct Scatter;

impl<L: AxisName, R: AxisName>
    Op<(
        &mut LatentKeyValue<L, R>,
        &Slot,
        (&Rows<Single, L>, &Rows<Single, R>),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        _: &mut Device,
        _: (
            &mut LatentKeyValue<L, R>,
            &Slot,
            (&Rows<Single, L>, &Rows<Single, R>),
        ),
    ) {
        todo!(
            "NPU write latent followed by rotary key into slot/Block, slot%Block; other rows unchanged"
        )
    }
}

/// Softmax of `scale * (q_latent · latent + q_rotary · rotary) + bias` reduces latent values.
/// Only tokens below the history length participate; all-masked queries produce zero.
#[derive(DeviceSend)]
pub struct Attention<H, L, R, T> {
    /// Multiplier applied to both query/key dot products before additive bias.
    pub scale: HbmTensor<f32, m![1], m![1]>,
    /// Query heads, latent width, rotary width and maximum history length.
    pub shape: PhantomData<(H, L, R, T)>,
}

impl<H: AxisName, L: AxisName, R: AxisName, T: AxisName>
    Op<(
        &Query<H, L, R>,
        &LatentKeyValue<L, R>,
        &History<Table<T>, ()>,
    )> for Attention<H, L, R, T>
{
    type Output = Rows<H, L>;

    fn forward(
        &self,
        _: &mut Device,
        _: (
            &Query<H, L, R>,
            &LatentKeyValue<L, R>,
            &History<Table<T>, ()>,
        ),
    ) -> Self::Output {
        todo!(
            "NPU latent attention: score = scale*(q_latent dot latent + q_rotary dot rotary). \
             No additive bias, softmax over logical rows below length, then reduce latent values. \
             All-masked rows produce zero; validate length <= T and block/slot bounds."
        )
    }
}

impl<H: AxisName, L: AxisName, R: AxisName, T: AxisName>
    Op<(
        &Query<H, L, R>,
        &LatentKeyValue<L, R>,
        &History<Table<T>, &Bias>,
    )> for Attention<H, L, R, T>
{
    type Output = Rows<H, L>;

    fn forward(
        &self,
        _: &mut Device,
        _: (
            &Query<H, L, R>,
            &LatentKeyValue<L, R>,
            &History<Table<T>, &Bias>,
        ),
    ) -> Self::Output {
        todo!(
            "NPU latent attention: score = scale*(q_latent dot latent + q_rotary dot rotary). \
             Add the supplied per-token bias, softmax over logical rows below length, then reduce latent values. \
             All-masked rows produce zero; validate length <= T and block/slot bounds."
        )
    }
}
