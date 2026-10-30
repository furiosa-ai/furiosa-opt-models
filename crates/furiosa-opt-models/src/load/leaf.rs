//! The HBM leaf: built from a checkpoint or from a value built on host (or a function making it),
//! written in place, borrowed for a call.

use furiosa_opt_std::prelude::{HbmTensor, HostTensor, M, MaterializableScalar, Pair, m};
use furiosa_opt_std::runtime::HostBuffer;

use super::{Checkpoint, Context, Load, LoadError};

/// The leaf at the current name, in HBM; a leaf of a tie target is split into two owners of its
/// storage, one for each tied field of its type. Chips that each hold every value read one copy.
impl<'a, D: MaterializableScalar, C: M + 'static, S: M + 'static> Load<&'a Checkpoint>
    for HbmTensor<D, C, S>
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let upload = |cx: &mut Context<'_>| match Checkpoint::splits::<C, S>() {
            true => {
                let host = from.host::<D, S, m![{ C }, { S }]>(cx)?;
                cx.upload::<D, C, S, _>(&host)
            }
            false => {
                let host = from.host::<D, S, S>(cx)?;
                cx.upload::<D, C, S, _>(&host)
            }
        };
        context.checkpoint(from, |cx| cx.leaf(from, upload, Context::owners))
    }
}

/// The leaf at the current name written in place: one rule of an update.
impl<'t, 's, D: MaterializableScalar, C: M + 'static, S: M + 'static>
    Load<(&'t mut HbmTensor<D, C, S>, &'s Checkpoint)> for &'t mut HbmTensor<D, C, S>
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut HbmTensor<D, C, S>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.rule(
                |cx| {
                    cx.overwrite::<HbmTensor<D, C, S>>(from, |cx| {
                        let written = match Checkpoint::splits::<C, S>() {
                            true => {
                                let host = from.host::<D, S, m![{ C }, { S }]>(cx)?;
                                <&mut HbmTensor<D, C, S> as Load<_>>::load(cx, (&mut *into, host))
                            }
                            false => {
                                let host = from.host::<D, S, S>(cx)?;
                                <&mut HbmTensor<D, C, S> as Load<_>>::load(cx, (&mut *into, host))
                            }
                        };
                        written.map(drop)
                    })
                },
                |_, ()| Ok(()),
            )
        })?;
        Ok(into)
    }
}

/// `from`, a leaf built in this traversal, copied into `into`'s storage. The device copies only
/// between HBM and host, so the copy goes through host memory.
impl<'t, D: MaterializableScalar, C: M + 'static, S: M + 'static>
    Load<(&'t mut HbmTensor<D, C, S>, HbmTensor<D, C, S>)> for &'t mut HbmTensor<D, C, S>
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut HbmTensor<D, C, S>, HbmTensor<D, C, S>),
    ) -> Result<Self, LoadError> {
        // A read into ordinary memory ends inside `submit`, so the write copies what it read.
        let mut host = HostBuffer::from_vec(vec![0; HbmTensor::<D, C, S>::size()]);
        let read = context.device().pdma.to_host([(&from, &mut host)]).submit();
        context.submitted(read)?;
        let write = context.device().pdma.to_hbm([(&host, &mut *into)]).submit();
        context.submitted(write)?;
        Ok(into)
    }
}

/// A `T<Hbm>` leaf borrowed as a `T<&Hbm>` leaf.
impl<'a, D: MaterializableScalar, C: M + 'static, S: M + 'static> Load<&'a HbmTensor<D, C, S>>
    for &'a HbmTensor<D, C, S>
{
    fn load(_: &mut Context<'_>, from: &'a HbmTensor<D, C, S>) -> Result<Self, LoadError> {
        Ok(from)
    }
}

/// A leaf whose value is built on host, as a computed or joined leaf's is, laid out as `E` (one
/// copy, the whole tensor, or every chip's share) and copied into a new allocation in HBM of the
/// context's device; it takes part in a tie as a checkpoint leaf does.
impl<D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M> Load<HostTensor<D, E>>
    for HbmTensor<D, C, S>
{
    fn load(context: &mut Context<'_>, host: HostTensor<D, E>) -> Result<Self, LoadError> {
        context.share(|cx| cx.upload::<D, C, S, E>(&host), Context::owners)
    }
}

/// A leaf whose value is built on host, laid out as `E`, written into `into`'s storage.
impl<'t, D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M>
    Load<(&'t mut HbmTensor<D, C, S>, HostTensor<D, E>)> for &'t mut HbmTensor<D, C, S>
{
    fn load(
        context: &mut Context<'_>,
        (into, host): (&'t mut HbmTensor<D, C, S>, HostTensor<D, E>),
    ) -> Result<Self, LoadError> {
        let write = host.to_hbm::<C, S>(&mut context.device().pdma);
        let write = write.output(&mut *into).submit();
        context.submitted(write)?;
        Ok(into)
    }
}

/// A leaf from a function of its host value: a converted leaf checks its parts' headers first,
/// and only a leaf that holds bytes calls the function.
impl<'f, D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M>
    Load<&'f dyn Fn() -> Result<HostTensor<D, E>, LoadError>> for HbmTensor<D, C, S>
{
    fn load(
        context: &mut Context<'_>,
        build: &'f dyn Fn() -> Result<HostTensor<D, E>, LoadError>,
    ) -> Result<Self, LoadError> {
        Self::load(context, build()?)
    }
}

/// A leaf whose host value `build` makes, written into `into`'s storage.
impl<'t, 'f, D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M>
    Load<(
        &'t mut HbmTensor<D, C, S>,
        &'f dyn Fn() -> Result<HostTensor<D, E>, LoadError>,
    )> for &'t mut HbmTensor<D, C, S>
{
    fn load(
        context: &mut Context<'_>,
        (into, build): (
            &'t mut HbmTensor<D, C, S>,
            &'f dyn Fn() -> Result<HostTensor<D, E>, LoadError>,
        ),
    ) -> Result<Self, LoadError> {
        <&mut HbmTensor<D, C, S> as Load<_>>::load(context, (into, build()?))
    }
}

#[cfg(test)]
mod tests {
    use super::super::Update;
    use super::super::checkpoint::tests::{checkpoint, host};
    use super::*;
    use crate::load::Failed;
    use crate::tensor::memory::{Hbm, Memory};
    use crate::tensor::parallel::Shard;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::*;
    use safetensors::Dtype;

    axes![O = 2, I = 4];

    /// One stored `[O, I]` tensor under two names: the chips of `split` split its axis `I`, a
    /// share a chip, and the chips of `copies` name no stored axis.
    #[derive(crate::load::Load)]
    struct Leaves<X: Memory> {
        split: X::Tensor<f32, m![I / 2], m![O, I % 2]>,
        copies: X::Tensor<f32, m![{ Symbol<Shard<2>> }], m![O, I]>,
    }

    /// An HBM leaf of the stored `[O, I]` tensor whose chips split its axis `I`, a share a chip.
    type Split = HbmTensor<f32, m![I / 2], m![O, I % 2]>;

    /// A leaf a function builds: the stored `[O, I]` tensor doubled.
    #[derive(crate::load::Load)]
    struct Built {
        #[load(with = doubled)]
        split: Split,
    }

    /// A checkpoint of `values`, the `[O, I]` tensor in row-major order, under both names.
    fn stored(values: &[f32; 8]) -> Checkpoint {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let tensor = |name| (name, Dtype::F32, &[2, 4][..], bytes.clone());
        checkpoint(&[tensor("split"), tensor("copies")])
    }

    /// Each chip's share of `values`, the `[O, I]` tensor `w` in row-major order, chip after chip:
    /// chip `c` holds `w[o][2c + j]`.
    fn shares(values: &[f32; 8]) -> Vec<f32> {
        (0..2)
            .flat_map(|chip| (0..2).flat_map(move |o| (0..2).map(move |j| (chip, o, j))))
            .map(|(chip, o, j)| values[o * I::SIZE + 2 * chip + j])
            .collect()
    }

    /// Every chip's values of each leaf, read back chip after chip, against the stored values
    /// indexed by chip: each chip's share of the split leaf, and the whole tensor of the copies.
    fn assert_held(leaves: &Leaves<Hbm>, values: &[f32; 8], device: &mut Device) {
        let copies: Vec<f32> = values.repeat(2);
        let held = host::<f32, m![I / 2], m![O, I % 2], m![I / 2, O, I % 2]>(&leaves.split, device);
        assert_eq!(held.into_vec(), shares(values));
        let held = host::<f32, m![{ Symbol<Shard<2>> }], m![O, I], m![{ Symbol<Shard<2>> }, O, I]>(
            &leaves.copies,
            device,
        );
        assert_eq!(held.into_vec(), copies);
    }

    /// A leaf whose chips split a stored axis holds each chip's share of the stored tensor on
    /// that chip, one whose chips name no stored axis the whole tensor on every chip, both when
    /// loaded and when an update writes them in place.
    #[test]
    fn each_chip_holds_its_share_or_a_copy() {
        let _lease = DEVICE.blocking_lock();
        let mut device = Device::new(Topology { chips: 2, pes: 8 }).unwrap();
        let loaded = [0.5, -1.25, 3.0, 7.75, -2.5, 4.125, 9.0, -6.5];
        let mut leaves: Leaves<Hbm> = stored(&loaded).load(&mut device).unwrap();
        assert_held(&leaves, &loaded, &mut device);
        let written = [11.0, 12.5, -13.0, 14.25, 15.0, -16.5, 17.0, 18.75];
        let update = Update::new(&mut leaves, &mut device);
        match update.load(&stored(&written)) {
            Ok(update) => update.finish().unwrap(),
            Err(Failed::Refused(_, error) | Failed::Poisoned(error)) => panic!("{error}"),
        }
        assert_held(&leaves, &written, &mut device);
    }

    fn doubled(from: &Checkpoint, context: &mut Context<'_>) -> Result<Split, LoadError> {
        let values = from.host::<f32, m![O, I % 2], m![I / 2, O, I % 2]>(context)?;
        let doubled = values.into_vec().into_iter().map(|value| 2.0 * value);
        HbmTensor::load(
            context,
            HostTensor::<f32, m![I / 2, O, I % 2]>::from_vec(doubled),
        )
    }

    /// An update writes a built leaf's new value into the storage its load allocated, each chip
    /// its share.
    #[test]
    fn update_copies_a_built_leaf_into_its_storage() {
        let _lease = DEVICE.blocking_lock();
        let mut device = Device::new(Topology { chips: 2, pes: 8 }).unwrap();
        let split = |values: &[f32; 8]| {
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            checkpoint(&[("split", Dtype::F32, &[2, 4][..], bytes)])
        };
        let mut built: Built = split(&[1.0; 8]).load(&mut device).unwrap();
        let written = [11.0, 12.5, -13.0, 14.25, 15.0, -16.5, 17.0, 18.75];
        match Update::new(&mut built, &mut device).load(&split(&written)) {
            Ok(update) => update.finish().unwrap(),
            Err(Failed::Refused(_, error) | Failed::Poisoned(error)) => panic!("{error}"),
        }
        let held =
            host::<f32, m![I / 2], m![O, I % 2], m![I / 2, O, I % 2]>(&built.split, &mut device);
        let doubled: Vec<f32> = shares(&written)
            .into_iter()
            .map(|value| 2.0 * value)
            .collect();
        assert_eq!(held.into_vec(), doubled);
    }
}
