//! Strict checkpoint partitioning for the text encoder's BF16 projections.
//!
//! Public projection leaves consume chip-local matrices. The checkpoint stores the whole
//! matrix: column-parallel projections take consecutive output rows, row-parallel projections
//! take consecutive input columns. Conversion and updates share the same checked partition.

use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::ops::linear::Affine;
use crate::tensor::memory::residency::Residency;
use crate::{Module, Op};
use core::{borrow::Borrow, marker::PhantomData};
use furiosa_opt_std::prelude::*;

/// A chip-local numerical projection loaded from a whole `[WholeOut, WholeIn]` matrix.
/// Exactly one axis may be partitioned, into one equal consecutive share per chip.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T))]
pub struct Projection<In: AxisName, Out: AxisName, WholeIn: AxisName, WholeOut: AxisName, C: M, T> {
    pub weight: T,
    shape: PhantomData<fn() -> (In, Out, WholeIn, WholeOut, C)>,
}

impl<I, In: AxisName, Out: AxisName, WI: AxisName, WO: AxisName, C: M, T> !Op<I>
    for Projection<In, Out, WI, WO, C, T>
{
}

impl<I, O, E, In: AxisName, Out: AxisName, WI: AxisName, WO: AxisName, C: M, T>
    Module<I, E, &mut ()> for Projection<In, Out, WI, WO, C, T>
where
    T: Borrow<HbmTensor<bf16, C, m![Out, In]>>,
    for<'a> Affine<&'a HbmTensor<bf16, C, m![Out, In]>, ()>: Op<I, Output = O>,
{
    type Output = O;
    fn forward(&self, device: &mut Device, _: E, _: &mut (), input: I) -> O {
        Op::forward(&Affine::new(self.weight.borrow(), ()), device, input)
    }
}

impl<'a, In: AxisName, Out: AxisName, WI: AxisName, WO: AxisName, C: M + 'static, T>
    Load<&'a Checkpoint> for Projection<In, Out, WI, WO, C, T>
where
    T: Load<HostTensor<bf16, m![{ C }, { m![Out, In] }]>>,
{
    fn load(context: &mut Context<'_>, source: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(source, |cx| {
            cx.field("weight", "weight", |cx| {
                let host = image::<In, Out, WI, WO, C>(source, cx)?;
                Ok(Self {
                    weight: T::load(cx, host)?,
                    shape: PhantomData,
                })
            })
        })
    }
}

impl<'t, 's, In: AxisName, Out: AxisName, WI: AxisName, WO: AxisName, C: M + 'static, T>
    Load<(&'t mut Projection<In, Out, WI, WO, C, T>, &'s Checkpoint)>
    for &'t mut Projection<In, Out, WI, WO, C, T>
where
    for<'u> &'u mut T: Load<(&'u mut T, HostTensor<bf16, m![{ C }, { m![Out, In] }]>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, source): (&'t mut Projection<In, Out, WI, WO, C, T>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(source, |cx| {
            cx.field("weight", "weight", |cx| {
                cx.rule(
                    |cx| image::<In, Out, WI, WO, C>(source, cx),
                    |cx, host| <&mut T as Load<_>>::load(cx, (&mut into.weight, host)).map(drop),
                )
            })
        })?;
        Ok(into)
    }
}

fn image<In: AxisName, Out: AxisName, WI: AxisName, WO: AxisName, C: M>(
    source: &Checkpoint,
    context: &mut Context<'_>,
) -> Result<HostTensor<bf16, m![{ C }, { m![Out, In] }]>, LoadError> {
    const {
        assert!(C::SIZE > 0 && In::SIZE > 0 && Out::SIZE > 0);
        assert!(
            (WI::SIZE == In::SIZE && WO::SIZE == Out::SIZE * C::SIZE)
                || (WO::SIZE == Out::SIZE && WI::SIZE == In::SIZE * C::SIZE)
        );
    }
    let whole = source
        .host::<bf16, m![WO, WI], m![WO, WI]>(context)?
        .into_vec();
    let mut local = Vec::with_capacity(C::SIZE * Out::SIZE * In::SIZE);
    for chip in 0..C::SIZE {
        for row in 0..Out::SIZE {
            let whole_row = if WO::SIZE == Out::SIZE {
                row
            } else {
                chip * Out::SIZE + row
            };
            let column = if WI::SIZE == In::SIZE {
                0
            } else {
                chip * In::SIZE
            };
            let at = whole_row * WI::SIZE + column;
            local.extend_from_slice(&whole[at..at + In::SIZE]);
        }
    }
    Ok(HostTensor::from_vec(local))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::Dtype;

    axes![Two = 2, Four = 4, Eight = 8];

    struct Image(Vec<bf16>);

    impl<E: M> Load<HostTensor<bf16, E>> for Image {
        fn load(_: &mut Context<'_>, host: HostTensor<bf16, E>) -> Result<Self, LoadError> {
            Ok(Self(host.into_vec()))
        }
    }

    /// Loading either partition must give the same matrix-vector product as the whole source.
    /// The oracle reads the unpartitioned checkpoint values with the specified BF16 rounding.
    #[test]
    fn partitions_whole_checkpoint_projections() {
        fn check<In: AxisName, Out: AxisName, WI: AxisName, WO: AxisName>() {
            let source: Vec<f32> = (0..WI::SIZE * WO::SIZE)
                .map(|at| ((at * 17 + 11) % 43) as f32 * 0.03127 - 0.5)
                .collect();
            let input: Vec<_> = (0..WI::SIZE)
                .map(|at| bf16::from_f32(at as f32 * 0.125 - 0.375).to_f32() as f64)
                .collect();
            let mut checkpoint = Checkpoint::default();
            checkpoint
                .insert(
                    "weight",
                    Dtype::F32,
                    &[WO::SIZE, WI::SIZE],
                    f32::to_buf(&source),
                )
                .unwrap();
            let projection = Projection::<In, Out, WI, WO, m![4], Image>::load(
                &mut Context::new(None),
                &checkpoint,
            )
            .unwrap();
            let mut product = vec![0.0_f64; WO::SIZE];
            for (chip, matrix) in projection
                .weight
                .0
                .chunks_exact(Out::SIZE * In::SIZE)
                .enumerate()
            {
                for (row, weights) in matrix.chunks_exact(In::SIZE).enumerate() {
                    // The mathematical placement of the result and input, independently of packing.
                    let (output, first) = if WI::SIZE == In::SIZE {
                        (chip * Out::SIZE + row, 0)
                    } else {
                        (row, chip * In::SIZE)
                    };
                    product[output] += weights
                        .iter()
                        .zip(&input[first..])
                        .map(|(weight, input)| weight.to_f32() as f64 * input)
                        .sum::<f64>();
                }
            }
            let expected: Vec<f64> = source
                .chunks_exact(WI::SIZE)
                .map(|row| {
                    row.iter()
                        .zip(&input)
                        .map(|(&weight, input)| bf16::from_f32(weight).to_f32() as f64 * input)
                        .sum()
                })
                .collect();
            assert_eq!(product, expected);
        }
        check::<Four, Two, Four, Eight>();
        check::<Two, Four, Eight, Four>();
    }
}
