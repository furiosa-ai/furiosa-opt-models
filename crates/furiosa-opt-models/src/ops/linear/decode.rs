//! The weight forms a contraction reads as BF16 tiles: each decodes its stored coefficients and
//! scales into the BF16 values the contraction multiplies.

use crate::ops::vector::{Restore, Scaled};
use crate::specialize::contraction;
use crate::tensor::form::BlockScaled;
use crate::tensor::form::nvfp4::{Early, Held, Scale, Scales, W4A16};
use crate::tensor::memory::Memory;
use crate::tensor::{parallel::Chip, scalar};
use core::borrow::Borrow;
use core::{fmt, marker::PhantomData};
use furiosa_opt_std::prelude::*;

use super::kernel::Tiling;

/// A weight form a contraction reads as BF16 tiles: BF16 as stored, E4M3 restored by its row's FP32
/// scale, or E2M1 restored by its E4M3 block scales under an early tensor scale.
pub(crate) trait Matrix<X: M = m![1]> {
    type In: AxisName;
    type Out: AxisName;

    /// Tile `at` of tiling `T`, in BF16.
    fn tile<T: Tiling<Self::In, Self::Out>>(
        &self,
        device: &mut Device,
        at: usize,
    ) -> DmTensor<bf16, X, T::Cluster, T::Slices, m![{ T::Rows }, { T::Inputs }]>;
}

/// A borrowed weight form reads as the form it borrows.
impl<X: M, T: Matrix<X>> Matrix<X> for &T {
    type In = T::In;
    type Out = T::Out;

    fn tile<S: Tiling<Self::In, Self::Out>>(
        &self,
        device: &mut Device,
        at: usize,
    ) -> DmTensor<bf16, X, S::Cluster, S::Slices, m![{ S::Rows }, { S::Inputs }]> {
        (**self).tile::<S>(device, at)
    }
}

impl<In: AxisName, Out: AxisName, X: M> Matrix<X> for HbmTensor<bf16, X, m![Out, In]> {
    type In = In;
    type Out = Out;

    fn tile<T: Tiling<In, Out>>(
        &self,
        device: &mut Device,
        at: usize,
    ) -> DmTensor<bf16, X, T::Cluster, T::Slices, m![{ T::Rows }, { T::Inputs }]> {
        T::to_dm(device, self, at)
    }
}

/// E4M3 coefficients in [output, input] order with one FP32 scale per output row.
impl<In: AxisName, Out: AxisName, K, V> Matrix
    for BlockScaled<In, Out, f8e4m3, f32, m![Out, 1 # 8], (), Chip, K, V>
where
    K: Borrow<HbmTensor<f8e4m3, Chip, m![Out, In]>>,
    V: Borrow<HbmTensor<f32, Chip, m![Out, 1 # 8]>>,
{
    type In = In;
    type Out = Out;

    /// `bf16(coefficient * scale)` with the FP32 scale of the coefficient's output row.
    fn tile<T: Tiling<In, Out>>(
        &self,
        device: &mut Device,
        at: usize,
    ) -> DmTensor<bf16, m![1], T::Cluster, T::Slices, m![{ T::Rows }, { T::Inputs }]> {
        type Selected<R> = m![{ R } = 1 # ({ R })];
        type SelectedMut<R> = m![{ R } = 1 #{!} ({ R })];
        let coded: DmTensor<f8e4m3, m![1], T::Cluster, T::Slices, m![{ T::Rows }, { T::Inputs }]> =
            T::to_dm(device, self.codes(), at);
        let scales: DmTensor<f32, m![1], T::Cluster, T::Slices, m![{ T::Rows }, 1 # 8]> =
            T::to_dm(device, self.scales(), at);
        let mut restored = DmTensor::new();
        for part in 0..<T::Restorations as M>::SIZE {
            let decoded: DmTensor<
                bf16,
                m![1],
                T::Cluster,
                T::Slices,
                m![{ T::Restored }, { T::Inputs }],
            > = device
                .main
                .begin(coded.view().tile::<T::Restorations, 1, m![
                    { Selected<T::Restorations> }, { T::Restored }, { T::Inputs }
                ]>(part))
                .fetch::<m![{ T::Restored }, { T::Inputs } / 16], m![In % 16]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![{ T::Restored }, { T::Inputs } / 16], m![In % 16]>()
                .commit_trim::<m![In % 16]>()
                .commit();
            let scale: VrfTensor<f32, m![1], T::Cluster, T::Slices, m![{ T::Restored }, 1 # 8]> =
                device
                    .sub
                    .begin(scales.view().tile::<T::Restorations, 1, m![
                        { Selected<T::Restorations> }, { T::Restored }, 1 # 8
                    ]>(part))
                    .fetch::<T::Restored, m![1 # 8]>()
                    .collect::<T::Restored, m![1 # 8]>()
                    .to_vrf();
            device
                .main
                .begin(decoded.view())
                .fetch::<m![{ T::Restored }, { T::Inputs } / 16], m![In % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![{ T::Restored }, { T::Inputs } / 8], m![In % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ T::Restored }, { T::Inputs } / 4], m![In % 4]>()
                .vector_restore::<m![{ T::Restored }, { T::Inputs } / 8], m![In % 8], _, _>(
                    &scale,
                    |values| values,
                )
                .commit_view(restored.view_mut().tile::<T::Restorations, 1, m![
                    { SelectedMut<T::Restorations> }, { T::Restored }, { T::Inputs }
                ]>(part));
        }
        restored
    }
}

/// E2M1 coefficients whose E4M3 block scales and early tensor scale restore them for BF16 inputs.
impl<In: AxisName, Out: AxisName, K, V, Y: Memory> Matrix
    for BlockScaled<In, Out, f4e2m1, f8e4m3, Scales<In, Out>, Early<W4A16, Y>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, m![Out, In]>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Scales<In, Out>>>,
    Held<Y>: Borrow<Scale>,
{
    type In = In;
    type Out = Out;

    /// `bf16(coefficient * block scale / global)`, a slice's rows in one restoration.
    fn tile<T: Tiling<In, Out>>(
        &self,
        device: &mut Device,
        at: usize,
    ) -> DmTensor<bf16, m![1], T::Cluster, T::Slices, m![{ T::Rows }, { T::Inputs }]> {
        let coded = (
            T::to_dm(device, self.codes(), at),
            T::to_dm(device, self.scales(), at),
        );
        let global: &Scale = self.contract.global.borrow();
        let global: DmTensor<f32, m![1], T::Cluster, T::Slices, m![1 # 8]> =
            global.to_dm(&mut device.tdma);
        let global = scalar::register(device, &global);
        Self::restore::<
            _,
            _,
            _,
            _,
            m![{ T::Rows } / ({ Symbol<Pass<T, In, Out>> })],
            m![{ T::Rows } % ({ Symbol<Pass<T, In, Out>> })],
            _,
            _,
        >(device, coded, |values| values.vector_fp_div(&global))
    }
}

/// Rows of a tile one restoration scales: the most, up to twelve, dividing the tile's rows whose
/// block scales fit the vector registers.
struct Pass<T, In, Out>(PhantomData<fn() -> (T, In, Out)>);

impl<T, In, Out> Clone for Pass<T, In, Out> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T, In, Out> fmt::Debug for Pass<T, In, Out> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Pass")
    }
}

impl<In: AxisName, Out: AxisName, T: Tiling<In, Out>> AxisName for Pass<T, In, Out> {
    const NAME: Ident = Ident::new("Pass");
    const SIZE: usize = contraction::height(<T::Rows as M>::SIZE, <T::Inputs as M>::SIZE);
}

/// A tile of E2M1 codes, `Rows` rows of `Inputs` inputs in each slice of partition `(C, S)`, and
/// the E4M3 scale of each 16 inputs of a row.
pub(super) type Coded<C, S, Rows, Inputs> = (
    DmTensor<f4e2m1, Chip, C, S, m![{ Rows }, { Inputs }]>,
    DmTensor<f8e4m3, Chip, C, S, m![{ Rows }, { Inputs } / 16]>,
);

/// A [`Coded`] tile's `Rows` rows decoded to FP32 four lanes wide, times their block scales, at
/// stage `Stage`.
type Placed<'l, I, C, S, Rows, Inputs, Stage> =
    Scaled<'l, { Tu::Main }, Stage, Chip, C, S, m![{ Rows }, { Inputs } / 4], m![I % 4]>;

impl<In: AxisName, Out: AxisName, Contract, K, V>
    BlockScaled<In, Out, f4e2m1, f8e4m3, Scales<In, Out>, Contract, Chip, K, V>
{
    /// A coded tile as BF16 code times block scale, through the tensor scale's `placement`: one
    /// pass over each of its `Parts`, `Part` rows whose block scales fit the vector registers.
    pub(super) fn restore<C: M, S: M, Rows: M, Inputs: M, Parts: Count, Part: M, Stage, P>(
        device: &mut Device,
        (codes, scale): Coded<C, S, Rows, Inputs>,
        placement: P,
    ) -> DmTensor<bf16, Chip, C, S, m![{ Rows }, { Inputs }]>
    where
        Stage: stage::Stage + stage::CanTransitionTo<stage::Widen>,
        P: Copy
            + for<'l> Fn(
                Placed<'l, In, C, S, Part, Inputs, stage::Fp>,
            ) -> Placed<'l, In, C, S, Part, Inputs, Stage>,
    {
        let mut restored = DmTensor::new();
        #[unroll]
        for part in 0..<Parts as M>::SIZE {
            let scale: VrfTensor<f32, Chip, C, S, m![{ Part }, { Inputs } / 16]> = device
                .sub
                .begin(scale.view().tile::<Parts, 1, m![
                    { Parts } = 1 # ({ Parts }), { Part }, { Inputs } / 16
                ]>(part))
                .fetch::<Part, m![{ Inputs } / 16]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Part }, { Inputs } / 128], m![In / 16 % 8]>()
                .to_vrf();
            device
                .main
                .begin(codes.view().tile::<Parts, 1, m![
                    { Parts } = 1 # ({ Parts }), { Part }, { Inputs }
                ]>(part))
                .fetch::<m![{ Part }, { Inputs } / 32], m![In % 32]>()
                .fetch_table_lookup::<f8e4m3>()
                .fetch_cast::<f32>()
                .collect::<m![{ Part }, { Inputs } / 8], m![In % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Part }, { Inputs } / 4], m![In % 4]>()
                .vector_restore::<m![{ Part }, { Inputs } / 8], m![In % 8], _, _>(&scale, placement)
                .commit_view(restored.view_mut().tile::<Parts, 1, m![
                    { Parts } = 1 #{!} ({ Parts }), { Part }, { Inputs }
                ]>(part));
        }
        restored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Op;
    use crate::ops::attention::geometry;
    use crate::ops::normalization::{RowMajor, Spread, tests::random};
    use crate::ops::swiglu;
    use crate::tensor::{axis::Sixteen, live::Live};
    use crate::tensor::{live::tests::count, scalar::Value};
    use furiosa_opt_std::runtime::launch;
    use std::iter;

    /// Live rows of a partial launch: one whole row group and part of the next.
    const LIVE: usize = 11;

    type Coded<I, O> = BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Early<W4A16>>;

    #[device(chip = 1)]
    fn early_rows<I: AxisName, O: AxisName>(
        device: &mut Device,
        affine: &Coded<I, O>,
        live: &Value,
        input: &HbmTensor<bf16, Chip, RowMajor<I, Sixteen>>,
    ) -> HbmTensor<bf16, Chip, RowMajor<O, Sixteen>> {
        let value = input.to_dm(&mut device.tdma);
        let input: Live<Spread<I, Sixteen>, Sixteen> = Live::<(), Sixteen>::read(live).with(value);
        affine
            .forward(&mut *device, &input)
            .value()
            .to_hbm(&mut device.tdma)
    }

    /// Every live row is the FP32 sum of its products with the BF16 coefficients
    /// `bf16(code * block scale / global)`, within one BF16 rounding of the sum and the FP32 error
    /// of summing in another order.
    async fn writes_the_early_rows<I: AxisName, O: AxisName>(device: &mut Device) {
        let mut state = 0x4e5f_0e2d_u64;
        let codes: Vec<f4e2m1> = (0..O::SIZE * I::SIZE)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                f4e2m1::from_bits((state >> 40) as u8 & 0xf)
            })
            .collect();
        let scales: Vec<f8e4m3> = (0..O::SIZE * I::SIZE / 16)
            .map(|at| f8e4m3::from_f32([0.5, 0.75, 1.25, 2.0, 1.5][at * 7 % 5]))
            .collect();
        let global = 37.5f32;
        let restored: Vec<bf16> = codes
            .iter()
            .zip(scales.iter().flat_map(|&scale| iter::repeat_n(scale, 16)))
            .map(|(code, scale)| bf16::from_f32(code.to_f32() * scale.to_f32() / global))
            .collect();
        let affine: Coded<I, O> = BlockScaled {
            weight: HostTensor::<f4e2m1, m![O, I]>::from_vec(codes)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            weight_scale: HostTensor::<f8e4m3, Scales<I, O>>::from_vec(scales)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            contract: Early {
                input: W4A16,
                global: HostTensor::<f32, m![1 # 8]>::from_vec(iter::once(global).chain([0.0; 7]))
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            },
            shape: PhantomData,
        };
        let live = count(LIVE).to_hbm(&mut device.pdma).await.unwrap();
        let inputs = random(Sixteen::SIZE * I::SIZE, 0x0b5e_ed11);
        let input = HostTensor::<bf16, RowMajor<I, Sixteen>>::from_vec(inputs.clone())
            .to_hbm::<Chip, RowMajor<I, Sixteen>>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(early_rows, (&mut *device, &affine, &live, &input))
            .await
            .unwrap()
            .to_host::<RowMajor<O, Sixteen>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for row in 0..LIVE {
            let input = &inputs[row * I::SIZE..][..I::SIZE];
            for (column, weights) in restored.chunks_exact(I::SIZE).enumerate() {
                let (sum, magnitude) =
                    input
                        .iter()
                        .zip(weights)
                        .fold((0.0f32, 0.0f32), |(sum, magnitude), (x, w)| {
                            let term = x.to_f32() * w.to_f32();
                            (sum + term, magnitude + term.abs())
                        });
                let tolerance = sum.abs() / 128.0 + magnitude * I::SIZE as f32 * f32::EPSILON;
                let got = output[row * O::SIZE + column].to_f32();
                assert!(
                    (got - sum).abs() <= tolerance,
                    "{} -> {}, row {row} column {column}: {got} vs {sum}",
                    I::SIZE,
                    O::SIZE,
                );
            }
        }
    }

    /// The live rows of an early-scaled NVFP4 projection give the host's sums over the restored
    /// rows, into and out of a gated MLP's intermediate width.
    #[tokio::test(flavor = "current_thread")]
    async fn writes_the_early_rows_at_each_tiling() {
        let mut device = Device::new(early_rows.topology()).unwrap();
        type Hidden = geometry::Hidden<3840>;
        type Intermediate = swiglu::Intermediate<15360>;
        writes_the_early_rows::<Hidden, Intermediate>(&mut device).await;
        writes_the_early_rows::<Intermediate, Hidden>(&mut device).await;
    }
}
