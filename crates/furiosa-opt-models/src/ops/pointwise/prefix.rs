//! Per-chip sums of a checked, initialized whole-block prefix of a row allocation.
//!
//! A block defines the numerical window geometry independently of allocation capacity.
//! Only those windows are fetched; allocation rows after the prefix may be uninitialized.
//! The child's narrowing and the intra-slice, inter-slice and final-window reductions retain
//! the ordinary [`Sum`] order. Unused window-sum slots contribute initialized FP32 zeros.

use super::sum::{Sum, Total, Windowed};
use super::{Float, Rows, Window};
use crate::Module;
use crate::tensor::live::BlockPrefix;
use furiosa_opt_std::prelude::*;

type Shape<R, W> = m![R, W];
/// Reduce only the initialized prefix, passing each original DM window to the child module.
impl<D: Float, R: AxisName, W: AxisName, B: AxisName, Chips: M, F, Env: Copy, State>
    Module<&BlockPrefix<&HbmTensor<D, Chips, Shape<R, W>>, R, B>, Env, &mut State> for Sum<F>
where
    F: for<'t, 's> Module<
            &'t Rows<D, Shape<B, W>, Chips>,
            Env,
            &'s mut State,
            Output = Rows<D, Shape<B, W>, Chips>,
        >,
{
    type Output = Total<Chips>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &mut State,
        input: &BlockPrefix<&HbmTensor<D, Chips, Shape<R, W>>, R, B>,
    ) -> Self::Output {
        sum::<D, R, W, B, Chips, _>(device, input, |device, tile| {
            self.0.forward(device, env, &mut *state, tile)
        })
    }
}

/// Both operands share one checked prefix and the same row, feature and chip layout.
impl<D: Float, R: AxisName, W: AxisName, B: AxisName, Chips: M, F, Env: Copy, State>
    Module<
        &BlockPrefix<
            (
                &HbmTensor<D, Chips, Shape<R, W>>,
                &HbmTensor<D, Chips, Shape<R, W>>,
            ),
            R,
            B,
        >,
        Env,
        &mut State,
    > for Sum<F>
where
    F: for<'t, 's> Module<
            &'t (Rows<D, Shape<B, W>, Chips>, Rows<D, Shape<B, W>, Chips>),
            Env,
            &'s mut State,
            Output = Rows<D, Shape<B, W>, Chips>,
        >,
{
    type Output = Total<Chips>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &mut State,
        input: &BlockPrefix<
            (
                &HbmTensor<D, Chips, Shape<R, W>>,
                &HbmTensor<D, Chips, Shape<R, W>>,
            ),
            R,
            B,
        >,
    ) -> Self::Output {
        sum::<D, R, W, B, Chips, _>(device, input, |device, tiles| {
            self.0.forward(device, env, &mut *state, tiles)
        })
    }
}

/// Use whole-block numerical geometry independently of the allocation capacity. The checked
/// block count and a compile-time windows-per-block ratio bound every source transfer.
fn sum<
    D: Float,
    R: AxisName,
    W: AxisName,
    B: AxisName,
    Chips: M,
    I: Windowed<Shape<R, W>, Shape<B, W>> + Copy,
>(
    device: &mut Device,
    input: &BlockPrefix<I, R, B>,
    mut formula: impl FnMut(&mut Device, &I::Tile) -> Rows<D, Shape<B, W>, Chips>,
) -> Total<Chips> {
    input
        .value()
        .sum_windows(device, windows::<R, W, B, _>(input), |device, tiles| {
            formula(device, tiles)
        })
}

/// Bound work in whole numerical windows using the same checked block geometry for every op.
pub(super) fn windows<R: AxisName, W: AxisName, B: AxisName, I>(
    input: &BlockPrefix<I, R, B>,
) -> usize {
    const {
        assert!(R::SIZE > 0 && W::SIZE > 0 && B::SIZE > 0);
        assert!(R::SIZE.is_multiple_of(B::SIZE));
        assert!(
            Shape::<R, W>::SIZE.is_multiple_of(8),
            "pointwise sums require complete eight-element vectors"
        );
        assert!(
            (B::SIZE * W::SIZE).is_multiple_of(Window::<Shape<B, W>>::SIZE),
            "pointwise prefix blocks must contain complete block windows"
        );
    }
    input.blocks() * const { B::SIZE * W::SIZE / Window::<Shape<B, W>>::SIZE }
}

#[cfg(test)]
mod tests {
    use super::{Sum, Total};
    use crate::ops::pointwise::abs::Abs;
    use crate::ops::pointwise::subtract::Subtract;
    use crate::ops::pointwise::subtract::tests::{
        Chips, Host, Part, ReferenceFeatures, rounding_terms,
    };
    use crate::tensor::live::BlockPrefix;
    use crate::tensor::parallel::Shard;
    use crate::tensor::scalar::Value;
    use crate::{Module, Then};
    use furiosa_opt_std::{prelude::*, runtime::launch};

    // Four blocks, each containing three complete original 224-slice x 512-element windows.
    axes![Capacity = 256, Block = 64];
    type Input = Part<Capacity, ReferenceFeatures>;
    type Distance = (Sum<Then<Subtract, Abs>>, Sum<Abs>);

    #[device(chip = 4)]
    fn magnitudes(
        device: &mut Device,
        distance: &Distance,
        (left, right): (&Input, &Input),
        blocks: &Value<Chips>,
    ) -> (Total<Chips>, Total<Chips>) {
        let prefix = BlockPrefix::<(), Capacity, Block>::read(blocks);
        let operands = prefix.with((left, right));
        let norm = prefix.with(right);
        distance.forward(device, (), &mut (((), ()), ()), (&operands, &norm))
    }

    /// Each chip's prefix L1 norm and rounded-difference distance agree with independent FP64
    /// sums at the ordinary 1e-5 tolerance. NaN suffix rows must never affect either sum, including
    /// an empty prefix; each intermediate prefix and the full allocation use the same geometry.
    #[tokio::test(flavor = "current_thread")]
    async fn sums_only_initialized_prefixes() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(magnitudes.topology()).unwrap();
        let elements = Capacity::SIZE * ReferenceFeatures::SIZE;
        let (left, right) = rounding_terms(elements);
        let distance = (Sum(Then(Subtract, Abs)), Sum(Abs));
        for rows in [0, 64, 128, 192, Capacity::SIZE] {
            let prefix = BlockPrefix::<(), Capacity, Block>::try_over(rows).unwrap();
            let active = prefix.rows() * ReferenceFeatures::SIZE;
            let expected: Vec<(f64, f64)> = (0..4)
                .map(|chip| {
                    left.iter()
                        .zip(&right)
                        .skip(chip * elements)
                        .take(active)
                        .fold((0.0_f64, 0.0_f64), |(distance, norm), (left, right)| {
                            let difference = bf16::from_f32(left.to_f32() - right.to_f32());
                            (
                                distance + f64::from(difference.to_f32().abs()),
                                norm + f64::from(right.to_f32().abs()),
                            )
                        })
                })
                .collect();
            let mut left = left.clone();
            let mut right = right.clone();
            for chip in 0..4 {
                left[chip * elements + active..(chip + 1) * elements]
                    .fill(bf16::from_f32(f32::NAN));
                right[chip * elements + active..(chip + 1) * elements]
                    .fill(bf16::from_f32(f32::NAN));
            }
            let left: Input = Host::<Capacity, ReferenceFeatures>::from_vec(left)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let right: Input = Host::<Capacity, ReferenceFeatures>::from_vec(right)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let blocks: Value<Chips> =
                HostTensor::<i32, m![1]>::from_vec([i32::try_from(prefix.blocks()).unwrap()])
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap();
            let (distances, norms) = launch(
                magnitudes,
                (&mut device, &distance, (&left, &right), &blocks),
            )
            .await
            .unwrap();
            let read = async |device: &mut Device, total: Total<Chips>| {
                total
                    .to_host::<m![{ Symbol<Shard<4>> }, 1 # 8]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec()
                    .chunks_exact(8)
                    .map(|chip| f64::from(chip[0]))
                    .collect::<Vec<_>>()
            };
            let (distances, norms) = (
                read(&mut device, distances).await,
                read(&mut device, norms).await,
            );
            for chip in 0..4 {
                let (distance, norm) = expected[chip];
                let close = |got: f64, want: f64| (got - want).abs() <= want * 1e-5;
                assert!(
                    close(distances[chip], distance) && close(norms[chip], norm),
                    "{rows} prefix rows, chip {chip}: ({}, {}) against ({distance}, {norm})",
                    distances[chip],
                    norms[chip]
                );
            }
        }
    }
}
