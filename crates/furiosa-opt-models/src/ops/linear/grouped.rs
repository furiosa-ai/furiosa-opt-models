//! BF16 projections with K-grouped weights for the register-resident row tile.

use core::borrow::Borrow;
use core::marker::PhantomData;

use crate::Op;
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::tensor::form::Weight;
use crate::tensor::live::{BlockPrefix, Live};
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
        <super::rows::Chunked<In, Out, crate::tensor::axis::Rows<256>> as AxisName>::SIZE;
}

/// Multiply BF16 HBM rows by K-grouped coefficients, rounding each output once.
/// Owned and borrowed trees hold the same weight layout and run the same kernel.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T))]
pub struct Grouped<
    In: AxisName,
    Out: AxisName,
    Chips: M = m![1],
    T = HbmTensor<bf16, Chips, Layout<In, Out>>,
> {
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
            cx.field("weight", "weight", |cx| {
                cx.compare(from)?;
                let host = from.host::<bf16, m![Out, In], m![{ Chips }, { m![Out, In] }]>(cx)?;
                T::load(cx, pack::<In, Out, Chips>(host))
            })
            .map(|weight| Self {
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
            cx.field("weight", "weight", |cx| {
                cx.rule(
                    |cx| {
                        cx.compare(from)?;
                        let host =
                            from.host::<bf16, m![Out, In], m![{ Chips }, { m![Out, In] }]>(cx)?;
                        Ok(pack::<In, Out, Chips>(host))
                    },
                    |cx, host| <&mut T as Load<_>>::load(cx, (&mut into.weight, host)).map(drop),
                )
            })
        })?;
        Ok(into)
    }
}

impl<'x, 'o, In: AxisName, Out: AxisName, R: AxisName, Chips: M, T>
    Op<(
        &'x HbmTensor<bf16, Chips, m![R, In]>,
        &'o mut HbmTensor<bf16, Chips, m![R, Out]>,
    )> for Grouped<In, Out, Chips, T>
where
    T: Borrow<HbmTensor<bf16, Chips, Layout<In, Out>>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, output): (
            &'x HbmTensor<bf16, Chips, m![R, In]>,
            &'o mut HbmTensor<bf16, Chips, m![R, Out]>,
        ),
    ) {
        let weight = super::rows::grouped::Packed(self.tensor());
        super::rows::Projection::<In, Out, R, Chips, (), bf16, _>::new(&weight, &())
            .forward(device, (input, &mut super::rows::grouped::Output(output)));
    }
}

impl<'x, In: AxisName, Out: AxisName, R: AxisName, Chips: M, T>
    Op<&'x HbmTensor<bf16, Chips, m![R, In]>> for Grouped<In, Out, Chips, T>
where
    T: Borrow<HbmTensor<bf16, Chips, Layout<In, Out>>>,
{
    type Output = HbmTensor<bf16, Chips, m![R, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, Chips, m![R, In]>,
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        self.forward(device, (input, &mut output));
        output
    }
}

/// Project a checked prefix into caller-owned storage. The grouped coefficient layout always
/// uses 256-row blocks; a different block size fails at compilation. Rows outside the returned
/// interval are untouched, and an empty prefix issues no row or coefficient transfer.
impl<'x, 'r, 'o, In: AxisName, Out: AxisName, R: AxisName, B: AxisName, Chips: M, T>
    Op<(
        &'r BlockPrefix<&'x HbmTensor<bf16, Chips, m![R, In]>, R, B>,
        &'o mut HbmTensor<bf16, Chips, m![R, Out]>,
    )> for Grouped<In, Out, Chips, T>
where
    T: Borrow<HbmTensor<bf16, Chips, Layout<In, Out>>>,
{
    type Output = Live<(), R>;

    fn forward(
        &self,
        device: &mut Device,
        (input, output): (
            &'r BlockPrefix<&'x HbmTensor<bf16, Chips, m![R, In]>, R, B>,
            &'o mut HbmTensor<bf16, Chips, m![R, Out]>,
        ),
    ) -> Self::Output {
        const {
            assert!(B::SIZE == 256, "grouped prefixes use 256-row blocks");
            assert!(R::SIZE.is_multiple_of(256));
        }
        let weight = super::rows::grouped::Packed(self.tensor());
        super::rows::Projection::<In, Out, R, Chips, (), bf16, _>::new(&weight, &())
            .prefix::<bf16, _, _, _, _, B>(
                device,
                *input.value(),
                input.with(()),
                &mut super::rows::grouped::Output(output),
            );
        input.interval()
    }
}

/// Allocate the usual row envelope and initialize only its checked work prefix. The owned result
/// carries that interval; allocation rows outside it are not initialized or exposed as work.
impl<'x, 'r, In: AxisName, Out: AxisName, R: AxisName, B: AxisName, Chips: M, T>
    Op<&'r BlockPrefix<&'x HbmTensor<bf16, Chips, m![R, In]>, R, B>> for Grouped<In, Out, Chips, T>
where
    T: Borrow<HbmTensor<bf16, Chips, Layout<In, Out>>>,
{
    type Output = Live<HbmTensor<bf16, Chips, m![R, Out]>, R>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'r BlockPrefix<&'x HbmTensor<bf16, Chips, m![R, In]>, R, B>,
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        self.forward(device, (input, &mut output)).with(output)
    }
}

/// Only the storage layout changes: all BF16 bit patterns survive the transpose.
fn pack<In: AxisName, Out: AxisName, Chips: M>(
    host: HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>,
) -> HostTensor<bf16, m![{ Chips }, { Layout<In, Out> }]> {
    let matrix = Tensor::<bf16, m![{ Chips }, { m![Out, In] }]>::from_vec(host.into_vec());
    let packed = matrix.transpose::<m![{ Chips }, { Layout<In, Out> }]>(false);
    HostTensor::from_vec(packed.into_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::attention::geometry::{Hidden, ProjectionWidth};
    use crate::tensor::axis::Rows;

    type In = Hidden<128>;
    type Out = ProjectionWidth<2048>;
    type R = Rows<768>;

    #[test]
    fn packing_preserves_coefficients_on_each_chip() {
        type In = Hidden<128>;
        type Out = ProjectionWidth<2048>;
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

    #[device(chip = 1)]
    fn project(
        device: &mut Device,
        grouped: &Grouped<In, Out>,
        input: &HbmTensor<bf16, m![1], m![R, In]>,
    ) -> HbmTensor<bf16, m![1], m![R, Out]> {
        grouped.forward(device, input)
    }

    /// Exact dyadic inputs make every FP32 sum exact, leaving only BF16 rounding.
    #[tokio::test(flavor = "current_thread")]
    async fn projects_against_fp64_dot_products() {
        use furiosa_opt_std::runtime::launch;

        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(project.topology()).unwrap();
        type In = Hidden<128>;
        type Out = ProjectionWidth<2048>;
        type R = Rows<768>;
        let weight: Vec<bf16> = (0..In::SIZE * Out::SIZE)
            .map(|i| bf16::from_f32(((i * 17 % 63) as f32 - 31.0) / 32.0))
            .collect();
        let rows: Vec<bf16> = (0..R::SIZE * In::SIZE)
            .map(|i| bf16::from_f32(((i * 23 % 61) as f32 - 30.0) / 32.0))
            .collect();
        let packed = pack::<In, Out, m![1]>(HostTensor::from_vec(weight.clone()));
        let grouped = Grouped::new(Weight::new(
            packed
                .to_hbm::<m![1], Layout<In, Out>>(&mut device.pdma)
                .await
                .unwrap(),
        ));
        let input = HostTensor::<bf16, m![R, In]>::from_vec(rows.clone())
            .to_hbm::<m![1], m![R, In]>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(project, (&mut device, &grouped, &input))
            .await
            .unwrap()
            .to_host::<m![R, Out]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for row in [0, 255, 256, 511, 512, 767] {
            for column in 0..Out::SIZE {
                let expected: f64 = rows[row * In::SIZE..][..In::SIZE]
                    .iter()
                    .zip(&weight[column * In::SIZE..][..In::SIZE])
                    .map(|(x, w)| f64::from(x.to_f32()) * f64::from(w.to_f32()))
                    .sum();
                assert_eq!(
                    output[row * Out::SIZE + column].to_bits(),
                    bf16::from_f32(expected as f32).to_bits(),
                    "row {row}, column {column}"
                );
            }
        }
    }
}

#[cfg(test)]
mod update_tests {
    use super::{Grouped, Layout};
    use crate::load::{Checkpoint, Context, Failed, Load, LoadError, Update};
    use crate::tensor::memory::residency::{Path, Residency, Visit};
    use crate::tensor::memory::{Hbm, Memory};
    use furiosa_opt_std::prelude::*;
    use safetensors::Dtype;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    axes![In = 128, Out = 512];
    type Chips = m![1];

    struct Storage;
    struct Values {
        data: Mutex<Vec<bf16>>,
        writes: AtomicUsize,
    }
    #[derive(Clone)]
    struct Leaf(Arc<Values>);

    impl Memory for Storage {
        type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Leaf;
    }

    impl<E: M> Load<HostTensor<bf16, E>> for Leaf {
        fn load(cx: &mut Context<'_>, host: HostTensor<bf16, E>) -> Result<Self, LoadError> {
            cx.share(
                |_| {
                    Ok(Self(Arc::new(Values {
                        data: Mutex::new(host.into_vec()),
                        writes: AtomicUsize::new(0),
                    })))
                },
                |_, leaf| Ok((leaf.clone(), leaf)),
            )
        }
    }

    /// The existing allocation is observed through every alias, just as a device leaf's is.
    impl<'t, E: M> Load<(&'t mut Leaf, HostTensor<bf16, E>)> for &'t mut Leaf {
        fn load(
            _: &mut Context<'_>,
            (into, host): (&'t mut Leaf, HostTensor<bf16, E>),
        ) -> Result<Self, LoadError> {
            let values = host.into_vec();
            into.0.data.lock().unwrap().copy_from_slice(&values);
            into.0.writes.fetch_add(1, Ordering::Relaxed);
            Ok(into)
        }
    }

    type Projection<X> =
        Grouped<In, Out, Chips, <X as Memory>::Tensor<bf16, Chips, Layout<In, Out>>>;

    #[derive(Load, Residency)]
    #[residency(memory(X))]
    struct Tied<X: Memory> {
        target: Projection<X>,
        #[load(tied = "target")]
        tied: Projection<X>,
    }

    #[derive(Default)]
    struct Observe(Vec<Arc<Values>>);
    impl Visit<Storage> for Observe {
        type Error = std::convert::Infallible;

        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &Leaf,
            _: Path<'_>,
        ) -> Result<(), Self::Error> {
            self.0.push(leaf.0.clone());
            Ok(())
        }
    }

    fn chunk(tensors: &[(&str, f32)]) -> Checkpoint {
        let mut source = Checkpoint::default();
        for &(name, value) in tensors {
            source
                .insert(
                    name,
                    Dtype::BF16,
                    &[Out::SIZE, In::SIZE],
                    bf16::from_f32(value)
                        .to_bits()
                        .to_le_bytes()
                        .repeat(Out::SIZE * In::SIZE),
                )
                .unwrap();
        }
        source
    }

    fn loaded<'a>(
        result: Result<Update<'a, Tied<Storage>>, Failed<'a, Tied<Storage>>>,
    ) -> Update<'a, Tied<Storage>> {
        match result {
            Ok(update) => update,
            Err(Failed::Refused(_, error) | Failed::Poisoned(error)) => panic!("{error}"),
        }
    }

    /// Checkpoint updates retain the existing shared allocation, consume matching tied copies,
    /// wait for a missing target, and refuse malformed data before the affected write.
    #[test]
    fn strict_updates_compare_tied_copies_and_keep_existing_storage() {
        let mut tree =
            Tied::<Storage>::load(&mut Context::new(None), &chunk(&[("target.weight", 1.0)]))
                .unwrap();
        let mut seen = Observe::default();
        Tied::<Hbm>::visit(&tree, &mut seen, Path::ROOT).unwrap();
        assert!(Arc::ptr_eq(&seen.0[0], &seen.0[1]));
        let existing = seen.0[0].clone();
        let assert_value = |expected: f32| {
            assert!(
                existing
                    .data
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|&value| value == bf16::from_f32(expected))
            );
        };
        loaded(
            Update::<Tied<Storage>>::new(&mut tree, None)
                .load(&chunk(&[("target.weight", 2.0), ("tied.weight", 2.0)])),
        )
        .finish()
        .unwrap();
        assert_value(2.0);
        let mut after = Observe::default();
        Tied::<Hbm>::visit(&tree, &mut after, Path::ROOT).unwrap();
        assert!(after.0.iter().all(|owner| Arc::ptr_eq(owner, &existing)));

        let before = existing.writes.load(Ordering::Relaxed);
        let update = loaded(
            Update::<Tied<Storage>>::new(&mut tree, None).load(&chunk(&[("tied.weight", 3.0)])),
        );
        assert_eq!(existing.writes.load(Ordering::Relaxed), before);
        assert_value(2.0);
        loaded(update.load(&chunk(&[("target.weight", 3.0)])))
            .finish()
            .unwrap();
        assert_value(3.0);

        for (dtype, shape, reason) in [
            (
                Dtype::BF16,
                [Out::SIZE / 2, In::SIZE * 2],
                "shape [256, 256], the leaf holds [512, 128]",
            ),
            (
                Dtype::I32,
                [Out::SIZE, In::SIZE],
                "stored as I32, the leaf holds BF16",
            ),
        ] {
            let mut invalid = Checkpoint::default();
            invalid
                .insert(
                    "target.weight",
                    dtype,
                    &shape,
                    vec![0; shape.iter().product::<usize>() * dtype.size()],
                )
                .unwrap();
            let before = existing.writes.load(Ordering::Relaxed);
            match Update::<Tied<Storage>>::new(&mut tree, None).load(&invalid) {
                Err(Failed::Refused(
                    _,
                    LoadError::InvalidData {
                        name,
                        reason: actual,
                    },
                )) => {
                    assert_eq!(name, "target.weight");
                    assert_eq!(actual, reason);
                }
                Err(Failed::Refused(_, error) | Failed::Poisoned(error)) => panic!("{error}"),
                Ok(_) => panic!("malformed data must not update the existing projection"),
            }
            assert_eq!(existing.writes.load(Ordering::Relaxed), before);
            assert_value(3.0);
        }

        let before = existing.writes.load(Ordering::Relaxed);
        match Update::<Tied<Storage>>::new(&mut tree, None)
            .load(&chunk(&[("target.weight", 3.0), ("tied.weight", 4.0)]))
        {
            Err(Failed::Refused(_, LoadError::InvalidData { name, reason })) => {
                assert_eq!(name, "tied.weight");
                assert_eq!(reason, "tied copy differs from target.weight");
            }
            Err(Failed::Refused(_, error) | Failed::Poisoned(error)) => panic!("{error}"),
            Ok(_) => panic!("a conflicting stored tie must be refused"),
        }
        assert_eq!(
            existing.writes.load(Ordering::Relaxed),
            before + 1,
            "the valid target may write before the conflicting tied rule refuses"
        );
        assert_value(3.0);
    }
}
