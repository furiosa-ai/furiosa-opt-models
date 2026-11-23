//! Fused query/key/value projection with grouped weights and tiled gather input.

use super::{Joint, Projection};
use crate::Op;
use crate::load::Load;
use crate::ops::collective::{GatherTile as Tile, Pieces, Tiled};
use crate::ops::linear::{self, Linear};
use crate::tensor::form::grouped::{Grouped, Layout};
use crate::tensor::live::{BlockPrefix, Live, Partitioned};
use crate::tensor::memory::residency::Residency;
use crate::tensor::parallel::Shard;
use core::borrow::Borrow;
use furiosa_opt_std::prelude::*;

/// Place each gathered part's computed prefix immediately after the preceding part's prefix.
/// The projection stores establish a contiguous work interval without a separate HBM copy.
/// Input intervals must start at zero and contain whole contraction blocks; every input row
/// within that work, including semantic padding, must be initialized.
#[derive(DeviceSend, Load, Residency)]
#[residency(children(P))]
pub struct Joined<P>(
    #[load(rename = "")]
    #[residency(transparent)]
    pub P,
);

impl<
    'x,
    In: AxisName,
    W: AxisName,
    S: AxisName,
    U: AxisName,
    Storage,
    Head,
    Rotated,
    const N: usize,
    const T: usize,
    const K: usize,
    const HEAD: usize,
> Op<&'x BlockPrefix<Tiled<Tile<bf16, N, In, T>, Pieces<S, N>, N, T, K, HEAD>, S, U>>
    for Joined<
        Projection<
            Linear<
                In,
                Joint<W>,
                m![{ Symbol<Shard<N>> }],
                Grouped<In, Joint<W>, m![{ Symbol<Shard<N>> }], Storage>,
            >,
            Head,
            Rotated,
        >,
    >
where
    Storage: Borrow<HbmTensor<bf16, m![{ Symbol<Shard<N>> }], Layout<In, Joint<W>>>>,
{
    type Output = Live<
        (
            HbmTensor<bf16, m![{ Symbol<Shard<N>> }], m![{ Symbol<Pieces<S, N>> }, W]>,
            HbmTensor<bf16, m![{ Symbol<Shard<N>> }], m![{ Symbol<Pieces<S, N>> }, W]>,
            HbmTensor<bf16, m![{ Symbol<Shard<N>> }], m![{ Symbol<Pieces<S, N>> }, W]>,
        ),
        Pieces<S, N>,
    >;

    fn forward(
        &self,
        device: &mut Device,
        tiles: &'x BlockPrefix<Tiled<Tile<bf16, N, In, T>, Pieces<S, N>, N, T, K, HEAD>, S, U>,
    ) -> Self::Output {
        let (mut query, mut key, mut value) =
            (HbmTensor::new(), HbmTensor::new(), HbmTensor::new());
        let weight = linear::rows::grouped::Packed(self.0.0.affine.weight.tensor());
        linear::Projection::<In, Joint<W>, Pieces<S, N>, m![{ Symbol<Shard<N>> }], (), bf16, _>::new(
            &weight,
            &(),
        )
        .tiled_prefix::<bf16, _, _, _, _, U, N, T, K, HEAD>(
            device,
            tiles.value(),
            tiles.with(()),
            &mut (&mut query, &mut key, &mut value),
        );
        Live::<(), Pieces<S, N>>::all()
            .subrange(0, N * tiles.rows())
            .with((query, key, value))
    }
}

/// Project the same live interval of each gathered part, retaining the parts' capacity stride.
/// The result's intervals exclude padding computed at either edge of a row block.
impl<
    'x,
    In: AxisName,
    W: AxisName,
    S: AxisName,
    Storage,
    Head,
    Rotated,
    const N: usize,
    const T: usize,
    const K: usize,
    const HEAD: usize,
> Op<&'x Partitioned<Tiled<Tile<bf16, N, In, T>, Pieces<S, N>, N, T, K, HEAD>, S, N>>
    for Projection<
        Linear<
            In,
            Joint<W>,
            m![{ Symbol<Shard<N>> }],
            Grouped<In, Joint<W>, m![{ Symbol<Shard<N>> }], Storage>,
        >,
        Head,
        Rotated,
    >
where
    Storage: Borrow<HbmTensor<bf16, m![{ Symbol<Shard<N>> }], Layout<In, Joint<W>>>>,
{
    type Output = Partitioned<
        (
            HbmTensor<bf16, m![{ Symbol<Shard<N>> }], m![{ Symbol<Pieces<S, N>> }, W]>,
            HbmTensor<bf16, m![{ Symbol<Shard<N>> }], m![{ Symbol<Pieces<S, N>> }, W]>,
            HbmTensor<bf16, m![{ Symbol<Shard<N>> }], m![{ Symbol<Pieces<S, N>> }, W]>,
        ),
        S,
        N,
    >;

    fn forward(
        &self,
        device: &mut Device,
        tiles: &'x Partitioned<Tiled<Tile<bf16, N, In, T>, Pieces<S, N>, N, T, K, HEAD>, S, N>,
    ) -> Self::Output {
        let (mut query, mut key, mut value) =
            (HbmTensor::new(), HbmTensor::new(), HbmTensor::new());
        let weight = linear::rows::grouped::Packed(self.0.affine.weight.tensor());
        linear::Projection::<In, Joint<W>, Pieces<S, N>, m![{ Symbol<Shard<N>> }], (), bf16, _>::new(
            &weight,
            &(),
        )
        .tiled_live::<bf16, _, _, _, _, N, T, K, HEAD>(
            device,
            tiles.value(),
            tiles.interval(),
            &mut (&mut query, &mut key, &mut value),
        );
        tiles.with((query, key, value))
    }
}

impl<'x, In: AxisName, W: AxisName, R: AxisName, Chips: M, T, Head, Rotated>
    Op<&'x HbmTensor<bf16, Chips, m![R, In]>>
    for Projection<Linear<In, Joint<W>, Chips, Grouped<In, Joint<W>, Chips, T>>, Head, Rotated>
where
    T: Borrow<HbmTensor<bf16, Chips, Layout<In, Joint<W>>>>,
{
    type Output = (
        HbmTensor<bf16, Chips, m![R, W]>,
        HbmTensor<bf16, Chips, m![R, W]>,
        HbmTensor<bf16, Chips, m![R, W]>,
    );

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, Chips, m![R, In]>,
    ) -> Self::Output {
        let (mut query, mut key, mut value) =
            (HbmTensor::new(), HbmTensor::new(), HbmTensor::new());
        let weight = linear::rows::grouped::Packed(self.0.affine.weight.tensor());
        let projection =
            linear::Projection::<In, Joint<W>, R, Chips, (), bf16, _>::new(&weight, &());
        if const { <linear::rows::Passes<In, Joint<W>, R> as AxisName>::SIZE == 1 } {
            projection.forward(device, (input, &mut (&mut query, &mut key, &mut value)));
        } else {
            projection.forward(device, (input, (&mut query, &mut key, &mut value)));
        }
        (query, key, value)
    }
}

impl<
    'x,
    In: AxisName,
    W: AxisName,
    S: AxisName,
    Storage,
    Head,
    Rotated,
    const N: usize,
    const T: usize,
    const K: usize,
    const HEAD: usize,
> Op<&'x Tiled<Tile<bf16, N, In, T>, Pieces<S, N>, N, T, K, HEAD>>
    for Projection<
        Linear<
            In,
            Joint<W>,
            m![{ Symbol<Shard<N>> }],
            Grouped<In, Joint<W>, m![{ Symbol<Shard<N>> }], Storage>,
        >,
        Head,
        Rotated,
    >
where
    Storage: Borrow<HbmTensor<bf16, m![{ Symbol<Shard<N>> }], Layout<In, Joint<W>>>>,
{
    type Output = (
        HbmTensor<bf16, m![{ Symbol<Shard<N>> }], m![{ Symbol<Pieces<S, N>> }, W]>,
        HbmTensor<bf16, m![{ Symbol<Shard<N>> }], m![{ Symbol<Pieces<S, N>> }, W]>,
        HbmTensor<bf16, m![{ Symbol<Shard<N>> }], m![{ Symbol<Pieces<S, N>> }, W]>,
    );

    fn forward(
        &self,
        device: &mut Device,
        tiles: &'x Tiled<Tile<bf16, N, In, T>, Pieces<S, N>, N, T, K, HEAD>,
    ) -> Self::Output {
        let (mut query, mut key, mut value) =
            (HbmTensor::new(), HbmTensor::new(), HbmTensor::new());
        let weight = linear::rows::grouped::Packed(self.0.affine.weight.tensor());
        linear::Projection::<In, Joint<W>, Pieces<S, N>, m![{ Symbol<Shard<N>> }], (), bf16, _>::new(
            &weight, &(),
        ).tiled::<bf16, _, _, _, N, T, K, HEAD>(
            device, (tiles, &mut (&mut query, &mut key, &mut value)),
        );
        (query, key, value)
    }
}
