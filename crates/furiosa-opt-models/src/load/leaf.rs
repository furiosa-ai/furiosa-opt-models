//! The HBM leaf: built from a checkpoint or from a value built on host (or a function making it),
//! written in place, borrowed for a call.

use std::future::IntoFuture;

use furiosa_opt_std::prelude::Slice;
use furiosa_opt_std::prelude::{HbmTensor, HostTensor, M, MaterializableScalar, Pair, m};
use tokio::task;

use super::{Checkpoint, Context, Load, LoadError};

impl<'a, T> Load<&'a Slice<T>> for &'a Slice<T> {
    fn load(_: &mut Context<'_>, from: &'a Slice<T>) -> Result<Self, LoadError> {
        Ok(from)
    }
}

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
                                let write = host.to_hbm::<C, S>(&mut cx.device().pdma);
                                let write = write.output(&mut *into).into_future();
                                futures_executor::block_on(task::unconstrained(write))
                            }
                            false => {
                                let host = from.host::<D, S, S>(cx)?;
                                let write = host.to_hbm::<C, S>(&mut cx.device().pdma);
                                let write = write.output(&mut *into).into_future();
                                futures_executor::block_on(task::unconstrained(write))
                            }
                        };
                        written.map_err(|source| LoadError::Device {
                            path: cx.path(),
                            source,
                        })
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
        let path = context.path();
        let device = context.device();
        let copy = async {
            let host = from.to_host::<m![{ C }, { S }]>(&mut device.pdma).await?;
            host.to_hbm::<C, S>(&mut device.pdma)
                .output(&mut *into)
                .await
        };
        futures_executor::block_on(task::unconstrained(copy.into_future()))
            .map_err(|source| LoadError::Device { path, source })?;
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

/// A leaf whose value is built on host, laid out as `E`, written into `into`'s storage; the host
/// value stays its owner's.
impl<'t, 'h, D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M>
    Load<(&'t mut HbmTensor<D, C, S>, &'h HostTensor<D, E>)> for &'t mut HbmTensor<D, C, S>
{
    fn load(
        context: &mut Context<'_>,
        (into, host): (&'t mut HbmTensor<D, C, S>, &'h HostTensor<D, E>),
    ) -> Result<Self, LoadError> {
        let path = context.path();
        let device = context.device();
        let write = host.to_hbm::<C, S>(&mut device.pdma).output(&mut *into);
        futures_executor::block_on(task::unconstrained(write.into_future()))
            .map_err(|source| LoadError::Device { path, source })?;
        Ok(into)
    }
}

/// The same write from a host value the caller gives up.
impl<'t, D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M>
    Load<(&'t mut HbmTensor<D, C, S>, HostTensor<D, E>)> for &'t mut HbmTensor<D, C, S>
{
    fn load(
        context: &mut Context<'_>,
        (into, host): (&'t mut HbmTensor<D, C, S>, HostTensor<D, E>),
    ) -> Result<Self, LoadError> {
        <&mut HbmTensor<D, C, S> as Load<_>>::load(context, (into, &host))
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
    use super::super::checkpoint::tests::{checkpoint, device, host};
    use super::*;
    use crate::load::{Failed, Named};
    use crate::tensor::memory::{Hbm, Memory};
    use crate::tensor::parallel::{Chip, Shard};
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

    /// A checkpoint of `values`, the `[O, I]` tensor in row-major order, under both names.
    fn stored(values: &[f32; 8]) -> Checkpoint {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let tensor = |name| (name, Dtype::F32, &[2, 4][..], bytes.clone());
        checkpoint(&[tensor("split"), tensor("copies")])
    }

    /// Every chip's values of each leaf, read back chip after chip, against the stored values
    /// indexed by chip: chip `c` holds `w[o][2c + j]` of the split leaf and all of `w` of the
    /// copies.
    fn assert_held(leaves: &Leaves<Hbm>, values: &[f32; 8], device: &mut Device) {
        let w = |o: usize, i: usize| values[o * I::SIZE + i];
        let shares: Vec<f32> = (0..2)
            .flat_map(|chip| (0..2).flat_map(move |o| (0..2).map(move |j| (chip, o, j))))
            .map(|(chip, o, j)| w(o, 2 * chip + j))
            .collect();
        let copies: Vec<f32> = (0..2)
            .flat_map(|_| (0..2).flat_map(|o| (0..4).map(move |i| w(o, i))))
            .collect();
        let held = host::<f32, m![I / 2], m![O, I % 2], m![I / 2, O, I % 2]>(&leaves.split, device);
        assert_eq!(held.into_vec(), shares);
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

    /// A tree written in place from a borrowed tree of host values holds those values in the
    /// storage its load allocated, write after write, through every structural node.
    #[test]
    fn writes_a_host_tree_into_a_loaded_one() {
        type Tree<A, B> = Named<"tree", ([A; 2], (Named<"a", A>, Named<"b", B>))>;
        type Leaf<S> = HbmTensor<f32, Chip, S>;
        type Staged<S> = HostTensor<f32, m![{ Chip }, { S }]>;
        type Matrix = m![O, I];
        type Vector = m![I];

        let values = |tree: f32| {
            let leaf = |index: f32, len: u8| -> Vec<f32> {
                (0..len)
                    .map(|i| 100.0 * tree + 10.0 * index + f32::from(i))
                    .collect()
            };
            [leaf(0.0, 8), leaf(1.0, 8), leaf(2.0, 8), leaf(3.0, 4)]
        };
        let staged = |[l0, l1, a, b]: &[Vec<f32>; 4]| -> Tree<Staged<Matrix>, Staged<Vector>> {
            Named((
                [
                    HostTensor::from_vec(l0.iter().copied()),
                    HostTensor::from_vec(l1.iter().copied()),
                ],
                (
                    Named(HostTensor::from_vec(a.iter().copied())),
                    Named(HostTensor::from_vec(b.iter().copied())),
                ),
            ))
        };
        let matrix = |leaf: &Leaf<Matrix>, device: &mut Device| {
            host::<f32, Chip, Matrix, m![{ Chip }, { Matrix }]>(leaf, device).into_vec()
        };
        let vector = |leaf: &Leaf<Vector>, device: &mut Device| {
            host::<f32, Chip, Vector, m![{ Chip }, { Vector }]>(leaf, device).into_vec()
        };
        let bytes = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        let (matrix_shape, vector_shape): (&[usize], &[usize]) = (&[2, 4], &[4]);
        let initial = values(1.0);
        let stored = checkpoint(&[
            ("tree.0", Dtype::F32, matrix_shape, bytes(&initial[0])),
            ("tree.1", Dtype::F32, matrix_shape, bytes(&initial[1])),
            ("tree.a", Dtype::F32, matrix_shape, bytes(&initial[2])),
            ("tree.b", Dtype::F32, vector_shape, bytes(&initial[3])),
        ]);

        let _lease = DEVICE.blocking_lock();
        let mut device = device();
        let mut tree: Tree<Leaf<Matrix>, Leaf<Vector>> = stored.load(&mut device).unwrap();
        let free = device.memory().usage().available;
        for written in [values(2.0), values(3.0)] {
            let from = staged(&written);
            <&mut Tree<Leaf<Matrix>, Leaf<Vector>> as Load<_>>::load(
                &mut Context::new(&mut device),
                (&mut tree, &from),
            )
            .unwrap();
            let Named(([l0, l1], (Named(a), Named(b)))) = &tree;
            let held = [
                matrix(l0, &mut device),
                matrix(l1, &mut device),
                matrix(a, &mut device),
                vector(b, &mut device),
            ];
            assert_eq!(held, written);
            assert_eq!(
                device.memory().usage().available,
                free,
                "a write allocates no HBM"
            );
        }
    }
}
