//! K-grouped BF16 coefficients for the register-resident row projection.

use core::borrow::Borrow;
use core::marker::PhantomData;

use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::tensor::form::Weight;
use crate::tensor::memory::residency::Residency;
use furiosa_opt_std::prelude::*;

/// Coefficients grouped as 32 outputs by one K group of the 256-row tile.
pub type Layout<In, Out> = m![
    Out / 32,
    In / ({ Symbol<Chunk<In, Out>> }),
    Out % 32,
    In % ({ Symbol<Chunk<In, Out>> })
];

/// Features of one K group in the packed 256-row projection's weight.
#[derive(Clone, Debug)]
pub struct Chunk<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for Chunk<In, Out> {
    const NAME: Ident = Ident::new("RowChunk");
    const SIZE: usize =
        In::SIZE / crate::specialize::contraction::RowConfig::new(In::SIZE, Out::SIZE, 256).parts;
}

/// BF16 coefficients packed by output block and K group, held as `T`, owned or borrowed.
/// Checkpoint matrices retain their logical [Out, In] shape; packing preserves every BF16 bit.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T))]
pub struct Grouped<
    In: AxisName,
    Out: AxisName,
    Chips: M = m![1],
    T = HbmTensor<bf16, Chips, Layout<In, Out>>,
> {
    #[load(rename = "")]
    #[residency(transparent)]
    weight: T,
    shape: PhantomData<fn() -> (In, Out, Chips)>,
}

impl<In: AxisName, Out: AxisName, Chips: M, T> Grouped<In, Out, Chips, T> {
    /// Retain an already packed matrix without copying it.
    pub fn new(weight: Weight<Layout<In, Out>, Chips, T>) -> Self {
        Self {
            weight: weight.into_tensor(),
            shape: PhantomData,
        }
    }
}

impl<In: AxisName, Out: AxisName, Chips: M, T> Grouped<In, Out, Chips, T>
where
    T: Borrow<HbmTensor<bf16, Chips, Layout<In, Out>>>,
{
    /// The packed coefficients of this projection.
    pub fn tensor(&self) -> &HbmTensor<bf16, Chips, Layout<In, Out>> {
        self.weight.borrow()
    }
}

/// Strictly read an ordinary checkpoint matrix and pack it before upload.
impl<'a, In: AxisName, Out: AxisName, Chips: M + 'static, T> Load<&'a Checkpoint>
    for Grouped<In, Out, Chips, T>
where
    T: Load<HostTensor<bf16, m![{ Chips }, { Layout<In, Out> }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.compare(from)?;
            let host = from.host::<bf16, m![Out, In], m![{ Chips }, { m![Out, In] }]>(cx)?;
            T::load(cx, pack::<In, Out, Chips>(host)).map(|weight| Self {
                weight,
                shape: PhantomData,
            })
        })
    }
}

impl<In: AxisName, Out: AxisName, Chips: M + 'static, T>
    Load<HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>> for Grouped<In, Out, Chips, T>
where
    T: Load<HostTensor<bf16, m![{ Chips }, { Layout<In, Out> }]>>,
{
    fn load(
        context: &mut Context<'_>,
        host: HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>,
    ) -> Result<Self, LoadError> {
        T::load(context, pack::<In, Out, Chips>(host)).map(|weight| Self {
            weight,
            shape: PhantomData,
        })
    }
}

/// Pack a converted matrix and write the held tensor's existing storage.
impl<'t, In: AxisName, Out: AxisName, Chips: M + 'static, T>
    Load<(
        &'t mut Grouped<In, Out, Chips, T>,
        HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>,
    )> for &'t mut Grouped<In, Out, Chips, T>
where
    for<'u> &'u mut T: Load<(
        &'u mut T,
        HostTensor<bf16, m![{ Chips }, { Layout<In, Out> }]>,
    )>,
{
    fn load(
        context: &mut Context<'_>,
        (into, host): (
            &'t mut Grouped<In, Out, Chips, T>,
            HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>,
        ),
    ) -> Result<Self, LoadError> {
        <&mut T as Load<_>>::load(context, (&mut into.weight, pack::<In, Out, Chips>(host)))?;
        Ok(into)
    }
}

/// Pack a checkpoint update and write the existing allocation, preserving ties.
impl<'t, 's, In: AxisName, Out: AxisName, Chips: M + 'static, T>
    Load<(&'t mut Grouped<In, Out, Chips, T>, &'s Checkpoint)>
    for &'t mut Grouped<In, Out, Chips, T>
where
    for<'u> &'u mut T: Load<(
        &'u mut T,
        HostTensor<bf16, m![{ Chips }, { Layout<In, Out> }]>,
    )>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Grouped<In, Out, Chips, T>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.rule(
                |cx| {
                    cx.compare(from)?;
                    let host =
                        from.host::<bf16, m![Out, In], m![{ Chips }, { m![Out, In] }]>(cx)?;
                    Ok(pack::<In, Out, Chips>(host))
                },
                |cx, host| <&mut T as Load<_>>::load(cx, (&mut into.weight, host)).map(drop),
            )
        })?;
        Ok(into)
    }
}

/// Only the storage layout changes: all BF16 bit patterns survive the transpose.
pub(crate) fn pack<In: AxisName, Out: AxisName, Chips: M>(
    host: HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>,
) -> HostTensor<bf16, m![{ Chips }, { Layout<In, Out> }]> {
    let matrix = Tensor::<bf16, m![{ Chips }, { m![Out, In] }]>::from_vec(host.into_vec());
    let packed = matrix.transpose::<m![{ Chips }, { Layout<In, Out> }]>(false);
    HostTensor::from_vec(packed.into_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_preserves_coefficients_on_each_chip() {
        axes![In = 128, Out = 2048];
        type Chips = m![2];
        let original: Vec<bf16> = (0..2 * In::SIZE * Out::SIZE)
            .map(|i| bf16::from_bits(i.wrapping_mul(40503) as u16))
            .collect();
        let packed = pack::<In, Out, Chips>(HostTensor::from_vec(original.clone()));
        let packed =
            Tensor::<bf16, m![{ Chips }, { Layout<In, Out> }]>::from_vec(packed.into_vec());
        let restored = packed
            .transpose::<m![{ Chips }, { m![Out, In] }]>(false)
            .into_vec();
        assert!(
            original
                .iter()
                .zip(restored)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
}
