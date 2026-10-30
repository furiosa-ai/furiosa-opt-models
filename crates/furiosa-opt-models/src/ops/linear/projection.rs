//! Single-row BF16 projections that read the weight four output rows of each block at a time.

use super::kernel::{Contraction, Rounded, Rounding, Tiling};
use super::{Affine, Matrix};
use crate::ops::normalization::Spread;
use crate::tensor::form::BlockScaled;
use crate::tensor::live::Live;
use crate::{Op, tensor::parallel::Chip};
use furiosa_opt_std::prelude::*;

pub(super) use crate::tensor::mapping::BroadcastClusters as Cluster;
use std::marker::PhantomData;

type Vector<W, Chips = Chip> = HbmTensor<bf16, Chips, m![W]>;

#[derive(Clone, Debug)]
pub(super) struct Rows<W>(PhantomData<W>);

impl<W: AxisName> AxisName for Rows<W> {
    const NAME: Ident = Ident::new("Rows");
    const SIZE: usize = W::SIZE / 64;
}

pub(super) type Block<W> = Symbol<Rows<W>>;
pub(super) type Slices<I, O> = m![O / ({ Block<O> }), 1 # 2, I / (I / 2)];
pub(super) type Reduced<O> = m![O / ({ Block<O> }), 1 # 2, 1 # 2];
type Elements<O> = m![O % ({ Block<O> }) = 4];

/// One-row sums at stage `Stage` of their epilogue, after the reduction over the parts: `Rows`
/// outputs of a slice, four lanes each.
pub(super) type Summed<'l, Stage, X, O, Rows> = VectorTensor<
    'l,
    { Tu::Main },
    Stage,
    f32,
    X,
    Cluster,
    Reduced<O>,
    Rows,
    m![1 # 4],
    Fresh,
    { stage::VeOrder::InterFirst },
    stage::Standalone,
    { stage::Way::Way4 },
>;

/// A one-row projection's FP32 sums of `Rows` outputs a slice, added over the slices that hold
/// the row's parts, scaled by their caller and rounded once.
pub(super) struct Partials<O, Rows>(PhantomData<(O, Rows)>);

impl<O: AxisName, Rows: M> Partials<O, Rows> {
    /// Each slice's `partial` sums added over the row's parts, in four lanes for the scales.
    pub(super) fn reduced<'l, X: M, S: M, E: M>(
        main: &'l mut TuContext<{ Tu::Main }>,
        partial: DmTensorView<'l, f32, X, Cluster, S, E>,
    ) -> Summed<'l, stage::Narrow, X, O, Rows> {
        main.begin(partial)
            .fetch::<Rows, m![1 # 8]>()
            .collect::<Rows, m![1 # 8]>()
            .vector_init()
            .vector_inter_slice_reduce::<Reduced<O>, Rows>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
    }

    /// The `scaled` sums rounded once to `T`, a slice's outputs in one packet, as the stream a
    /// caller commits.
    pub(super) fn rounded<'l, T: MaterializableScalar, X: M>(
        scaled: Summed<'l, stage::Fp, X, O, Rows>,
    ) -> CommitTrimTensor<'l, { Tu::Main }, T, X, Cluster, Reduced<O>, m![1], Rows>
    where
        f32: CastEngineCast<T>,
    {
        Rounded::rounded::<T, _, _, _, _, m![1], m![1], Rows, _>(
            scaled.vector_widen_pad::<m![1 # 8]>().vector_final(),
        )
    }
}

/// The tiling of a one-row projection: each slice keeps half a row's inputs for four output rows
/// of its block; tile `at` is the four rows from row `at` of every block.
pub(super) struct OneRow<I, O>(PhantomData<(I, O)>);

impl<I: AxisName, O: AxisName> Tiling<I, O> for OneRow<I, O> {
    type Cluster = Cluster;
    type Slices = Slices<I, O>;
    type Rows = Elements<O>;
    type Inputs = m![I % (I / 2)];
    type Restorations = m![{ Elements<O> } / 4];
    type Restored = Elements<O>;

    fn to_dm<D: MaterializableScalar, X: M, K: M, E: M>(
        device: &mut Device,
        tensor: &HbmTensor<D, X, m![O, { K }]>,
        at: usize,
    ) -> DmTensor<D, X, Cluster, Slices<I, O>, E> {
        tensor
            .view()
            .tile::<
                m![O % ({ Block<O> })],
                4,
                m![O / ({ Block<O> }), { Elements<O> } # ({ Block<O> }), { K }],
            >(at)
            .to_dm(&mut device.tdma)
    }
}

/// Project one contiguous BF16 input row; both widths must be positive multiples of 256.
impl<I: AxisName, O: AxisName, Chip: M, W: Matrix<Chip, In = I, Out = O>>
    Op<HbmTensorView<'_, bf16, Chip, m![I]>> for Affine<W, ()>
{
    type Output = Vector<O, Chip>;

    fn forward(
        &self,
        device: &mut Device,
        input: HbmTensorView<'_, bf16, Chip, m![I]>,
    ) -> Self::Output {
        const {
            assert!(
                I::SIZE > 0
                    && O::SIZE > 0
                    && I::SIZE.is_multiple_of(256)
                    && O::SIZE.is_multiple_of(256)
            )
        };
        let input: DmTensor<bf16, Chip, Cluster, m![64, 1 # 2, I / (I / 2)], m![I % (I / 2)]> =
            input.to_dm(&mut device.tdma);
        let input: DmTensor<bf16, Chip, Cluster, Slices<I, O>, m![I % (I / 2)]> =
            unsafe { input.reshape() };
        let input: TrfTensor<bf16, Chip, Cluster, Slices<I, O>, m![1], m![1, I % (I / 2)]> = device
            .sub
            .begin(input.view())
            .fetch::<m![I % (I / 2) / 16], m![I % 16]>()
            .collect::<m![I % (I / 2) / 16], m![I % 16]>()
            .to_trf();
        let mut output: DmTensor<bf16, Chip, Cluster, Reduced<O>, m![O % ({ Block<O> })]> =
            DmTensor::new();
        for block in 0..(O::SIZE / 256) {
            let index = 4 * block;
            let tile = self.weight.tile::<OneRow<I, O>>(device, index);
            bf16::finish::<
                Reduced<O>,
                m![{ Elements<O> }, 1],
                m![1],
                m![1],
                Elements<O>,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            >(
                &(),
                Contraction::<I, O>::sums(device, &tile, &input),
                output
                    .view_mut()
                    .tile::<m![O % ({ Block<O> })], 4, m![{ Elements<O> } #{!} ({ Block<O> })]>(
                        index,
                    ),
            );
        }
        output.to_hbm(&mut device.tdma)
    }
}

/// Every weight form projects a row through its view.
impl<I: AxisName, O: AxisName, Chip: M, W: Matrix<Chip, In = I, Out = O>> Op<&Vector<I, Chip>>
    for Affine<W, ()>
{
    type Output = Vector<O, Chip>;

    fn forward(&self, device: &mut Device, input: &Vector<I, Chip>) -> Self::Output {
        self.forward(device, input.view())
    }
}

/// A block-scaled weight a contraction reads as BF16 tiles projects as the affine projection of
/// itself without a bias: the live rows of a launch.
impl<'x, I, O, E: Scalar, S: Scalar, B: M, C, K, V, Rows, Chips: M, P: M, F: M>
    Op<&'x Live<DmTensor<bf16, m![1], Chips, P, m![Rows, { F }]>, Rows>>
    for BlockScaled<I, O, E, S, B, C, Chip, K, V>
where
    I: AxisName,
    O: AxisName,
    Rows: AxisName,
    Self: Matrix<In = I, Out = O>,
{
    type Output = Live<Spread<O, Rows>, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<DmTensor<bf16, m![1], Chips, P, m![Rows, { F }]>, Rows>,
    ) -> Self::Output {
        Affine::new(self, ()).forward(device, input)
    }
}

/// One contiguous row through a block-scaled weight, as through its affine projection.
impl<I: AxisName, O: AxisName, E: Scalar, S: Scalar, B: M, C, K, V>
    Op<HbmTensorView<'_, bf16, Chip, m![I]>> for BlockScaled<I, O, E, S, B, C, Chip, K, V>
where
    Self: Matrix<Chip, In = I, Out = O>,
{
    type Output = Vector<O, Chip>;

    fn forward(
        &self,
        device: &mut Device,
        input: HbmTensorView<'_, bf16, Chip, m![I]>,
    ) -> Self::Output {
        Affine::new(self, ()).forward(device, input)
    }
}

/// A row in HBM through a block-scaled weight, as through its affine projection.
impl<I: AxisName, O: AxisName, E: Scalar, S: Scalar, B: M, C, K, V> Op<&Vector<I, Chip>>
    for BlockScaled<I, O, E, S, B, C, Chip, K, V>
where
    Self: Matrix<Chip, In = I, Out = O>,
{
    type Output = Vector<O, Chip>;

    fn forward(&self, device: &mut Device, input: &Vector<I, Chip>) -> Self::Output {
        Affine::new(self, ()).forward(device, input)
    }
}

/// A row read as its heads in order through a block-scaled weight.
impl<I, O, E: Scalar, S: Scalar, B: M, C, K, V, Heads: AxisName, Head: AxisName>
    Op<&HbmTensor<bf16, Chip, m![Heads, Head]>> for BlockScaled<I, O, E, S, B, C, Chip, K, V>
where
    I: AxisName,
    O: AxisName,
    Self: Matrix<Chip, In = I, Out = O>,
{
    type Output = Vector<O, Chip>;

    fn forward(
        &self,
        device: &mut Device,
        input: &HbmTensor<bf16, Chip, m![Heads, Head]>,
    ) -> Self::Output {
        Affine::new(self, ()).forward(device, input)
    }
}

/// A row read as its heads in order.
impl<I: AxisName, O: AxisName, Heads: AxisName, Head: AxisName, Chip: M, W>
    Op<&HbmTensor<bf16, Chip, m![Heads, Head]>> for Affine<W, ()>
where
    W: Matrix<Chip, In = I, Out = O>,
{
    type Output = Vector<O, Chip>;

    fn forward(
        &self,
        device: &mut Device,
        input: &HbmTensor<bf16, Chip, m![Heads, Head]>,
    ) -> Self::Output {
        const { assert!(I::SIZE == Heads::SIZE * Head::SIZE) };
        // SAFETY: the heads of a row are its `I` inputs in order.
        let input: HbmTensorView<'_, bf16, Chip, m![I]> = unsafe { input.view().reshape() };
        self.forward(device, input)
    }
}
