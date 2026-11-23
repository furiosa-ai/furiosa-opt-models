//! BF16 row projections whose Linear retains K-grouped coefficients.

use core::borrow::Borrow;

use super::Linear;
use crate::Op;
use crate::tensor::form::grouped::{Grouped, Layout};
use crate::tensor::live::{BlockPrefix, Live};
use furiosa_opt_std::prelude::*;

impl<'x, 'o, In: AxisName, Out: AxisName, R: AxisName, Chips: M, T>
    Op<(
        &'x HbmTensor<bf16, Chips, m![R, In]>,
        &'o mut HbmTensor<bf16, Chips, m![R, Out]>,
    )> for Linear<In, Out, Chips, Grouped<In, Out, Chips, T>>
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
        let weight = super::rows::grouped::Packed(self.affine.weight.tensor());
        super::rows::Projection::<In, Out, R, Chips, (), bf16, _>::new(&weight, &())
            .forward(device, (input, &mut super::rows::grouped::Output(output)));
    }
}

impl<'x, In: AxisName, Out: AxisName, R: AxisName, Chips: M, T>
    Op<&'x HbmTensor<bf16, Chips, m![R, In]>> for Linear<In, Out, Chips, Grouped<In, Out, Chips, T>>
where
    T: Borrow<HbmTensor<bf16, Chips, Layout<In, Out>>>,
{
    type Output = HbmTensor<bf16, Chips, m![R, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, Chips, m![R, In]>,
    ) -> HbmTensor<bf16, Chips, m![R, Out]> {
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
    )> for Linear<In, Out, Chips, Grouped<In, Out, Chips, T>>
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
    ) -> Live<(), R> {
        const {
            assert!(B::SIZE == 256, "grouped prefixes use 256-row blocks");
            assert!(R::SIZE.is_multiple_of(256));
        }
        let weight = super::rows::grouped::Packed(self.affine.weight.tensor());
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
    Op<&'r BlockPrefix<&'x HbmTensor<bf16, Chips, m![R, In]>, R, B>>
    for Linear<In, Out, Chips, Grouped<In, Out, Chips, T>>
where
    T: Borrow<HbmTensor<bf16, Chips, Layout<In, Out>>>,
{
    type Output = Live<HbmTensor<bf16, Chips, m![R, Out]>, R>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'r BlockPrefix<&'x HbmTensor<bf16, Chips, m![R, In]>, R, B>,
    ) -> Live<HbmTensor<bf16, Chips, m![R, Out]>, R> {
        let mut output = HbmTensor::new();
        self.forward(device, (input, &mut output)).with(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::attention::geometry::{Hidden, ProjectionWidth};
    use crate::tensor::axis::Rows;
    use crate::tensor::form::Weight;
    use crate::tensor::form::grouped::pack;

    type In = Hidden<128>;
    type Out = ProjectionWidth<2048>;
    type R = Rows<768>;

    #[device(chip = 1)]
    fn project(
        device: &mut Device,
        grouped: &Linear<In, Out, m![1], Grouped<In, Out>>,
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
        let grouped = Linear::new(Weight::new(Grouped::new(Weight::new(
            packed
                .to_hbm::<m![1], Layout<In, Out>>(&mut device.pdma)
                .await
                .unwrap(),
        ))));
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
    use super::{Grouped, Layout, Linear};
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

    type Projection<X> = Linear<
        In,
        Out,
        Chips,
        Grouped<In, Out, Chips, <X as Memory>::Tensor<bf16, Chips, Layout<In, Out>>>,
    >;

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

#[cfg(test)]
mod prefix_tests {
    //! Checked runtime prefixes across strict loading, borrowed storage and public grouped Ops.

    use crate::Op;
    use crate::load::{Checkpoint, Context, Load};
    use crate::ops::attention::geometry::{Hidden, ProjectionWidth};
    use crate::ops::linear::Linear;
    use crate::tensor::axis::Rows;
    use crate::tensor::form::Weight;
    use crate::tensor::form::grouped::{Grouped, Layout};
    use crate::tensor::live::{BlockPrefix, Live};
    use crate::tensor::scalar::Value;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    type R = Rows<2048>;
    type B = Rows<256>;
    type In = Hidden<128>;
    type Out = ProjectionWidth<2048>;
    type Input = HbmTensor<bf16, m![1], m![R, In]>;
    type Output = HbmTensor<bf16, m![1], m![R, Out]>;
    type Owned = Linear<In, Out, m![1], Grouped<In, Out>>;
    type Borrowed<'a> = Linear<
        In,
        Out,
        m![1],
        Grouped<In, Out, m![1], &'a HbmTensor<bf16, m![1], Layout<In, Out>>>,
    >;

    type FullIn = Hidden<5376>;
    type FullOut = ProjectionWidth<28672>;
    type FullRows = Rows<10496>;
    type FullInput = HbmTensor<bf16, m![1], m![FullRows, FullIn]>;
    type FullOutput = HbmTensor<bf16, m![1], m![FullRows, FullOut]>;
    type FullProjection = Linear<FullIn, FullOut, m![1], Grouped<FullIn, FullOut>>;

    const SENTINEL: f32 = 13.0;
    const POISON: u16 = 0x7fc1;
    const INPUT_PERIOD: usize = 29;
    const OUTPUT_PERIOD: usize = 31;

    #[device(chip = 1)]
    fn into(
        device: &mut Device,
        module: &Borrowed<'_>,
        input: &Input,
        blocks: &Value<m![1]>,
        output: &mut Output,
    ) {
        let prefix = BlockPrefix::<(), R, B>::read(blocks).with(input);
        let _: Live<(), R> = module.forward(device, (&prefix, output));
    }

    #[device(chip = 1)]
    fn owned(device: &mut Device, module: &Owned, input: &Input, blocks: &Value<m![1]>) -> Output {
        let prefix = BlockPrefix::<(), R, B>::read(blocks).with(input);
        let result: Live<Output, R> = module.forward(device, &prefix);
        result.into_value()
    }

    #[device(chip = 1)]
    fn full_prefix(
        device: &mut Device,
        module: &FullProjection,
        input: &FullInput,
        blocks: &Value<m![1]>,
        output: &mut FullOutput,
    ) {
        let prefix = BlockPrefix::<(), FullRows, B>::read(blocks).with(input);
        module.forward(device, (&prefix, output));
    }

    #[device(chip = 1)]
    fn full_fixed(device: &mut Device, module: &FullProjection, input: &FullInput) -> FullOutput {
        module.forward(device, input)
    }

    fn coefficient(column: usize, k: usize) -> bf16 {
        bf16::from_f32(((column * 17 + k * 3) % OUTPUT_PERIOD) as f32 / 16.0 - 15.0 / 16.0)
    }

    fn input(row: usize, k: usize) -> bf16 {
        bf16::from_f32(((row * 11 + k * 5) % INPUT_PERIOD) as f32 / 16.0 - 14.0 / 16.0)
    }

    /// Compute independent FP64 dot products for every distinct row/column class. Periods coprime
    /// to the tile and pass widths keep permutation mistakes observable without repeating the same
    /// FP64 dot millions of times. Dyadic products and these small sums are exactly representable
    /// in FP32, so the final BF16 bits are the complete numerical oracle.
    fn reference() -> Vec<bf16> {
        (0..INPUT_PERIOD * OUTPUT_PERIOD)
            .map(|index| {
                let (row, column) = (index / OUTPUT_PERIOD, index % OUTPUT_PERIOD);
                let dot = (0..In::SIZE)
                    .map(|k| {
                        f64::from(input(row, k).to_f32())
                            * f64::from(coefficient(column, k).to_f32())
                    })
                    .sum::<f64>();
                bf16::from_f32(dot as f32)
            })
            .collect()
    }

    fn check(values: &[bf16], rows: usize, tail: bool, reference: &[bf16]) {
        let allocated = if tail { R::SIZE } else { rows };
        assert_eq!(values.len(), allocated * Out::SIZE);
        for row in 0..allocated {
            for column in 0..Out::SIZE {
                let expected = if row < rows {
                    reference[(row % INPUT_PERIOD) * OUTPUT_PERIOD + column % OUTPUT_PERIOD]
                } else {
                    bf16::from_f32(SENTINEL)
                };
                assert_eq!(
                    values[row * Out::SIZE + column].to_bits(),
                    expected.to_bits(),
                    "row {row}, column {column}, work {rows}"
                );
            }
        }
    }

    fn checkpoint() -> (tempfile::TempDir, Checkpoint) {
        let directory = tempfile::tempdir().unwrap();
        let values: Vec<_> = (0..Out::SIZE * In::SIZE)
            .map(|at| coefficient(at / In::SIZE, at % In::SIZE))
            .collect();
        let bytes = bf16::to_buf(&values);
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![Out::SIZE, In::SIZE],
            &bytes,
        )
        .unwrap();
        safetensors::serialize_to_file(
            [("weight", view)],
            &None,
            &directory.path().join("weights.safetensors"),
        )
        .unwrap();
        let source = Checkpoint::new(directory.path(), Some(serde_json::json!({}))).unwrap();
        (directory, source)
    }

    /// Refuse invalid work before preparing transport or launching a projection.
    #[test]
    fn grouped_work_refuses_unaligned_or_oversized_prefixes() {
        for rows in [1, 255, 257, R::SIZE + B::SIZE] {
            assert!(
                BlockPrefix::<(), R, B>::try_over(rows).is_none(),
                "rows {rows}"
            );
        }
        for rows in [0, B::SIZE, R::SIZE] {
            let prefix = BlockPrefix::<(), R, B>::try_over(rows).unwrap();
            assert_eq!(prefix.blocks() * B::SIZE, rows);
        }
    }

    /// Check a 5376 -> 28672 runtime prefix against the static grouped projection.
    /// Empty, odd and even work also preserve destination tails.
    #[tokio::test(flavor = "current_thread")]
    async fn full_width_prefix_matches_the_static_grouped_projection() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let canonical = Tensor::<bf16, m![FullOut, FullIn]>::from_vec(
            (0..FullOut::SIZE * FullIn::SIZE)
                .map(|at| coefficient(at / FullIn::SIZE, at % FullIn::SIZE)),
        );
        let packed = canonical
            .transpose::<Layout<FullIn, FullOut>>(false)
            .into_vec();
        let weight = Grouped::new(Weight::new(
            HostTensor::<bf16, Layout<FullIn, FullOut>>::from_vec(packed)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        ));
        let model = FullProjection::new(Weight::new(weight));
        for rows in [0, 256, 512, 9472, 9728, 9984] {
            let source: FullInput = HostTensor::<bf16, m![FullRows, FullIn]>::from_vec(
                (0..FullRows::SIZE * FullIn::SIZE).map(|at| {
                    if at / FullIn::SIZE < rows {
                        input(at / FullIn::SIZE, at % FullIn::SIZE)
                    } else {
                        bf16::from_bits(POISON)
                    }
                }),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            let count: Value<m![1]> = HostTensor::<i32, m![1]>::from_vec([rows as i32 / 256])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let mut destination: FullOutput = HostTensor::<bf16, m![FullRows, FullOut]>::from_vec(
                std::iter::repeat_n(bf16::from_f32(SENTINEL), FullRows::SIZE * FullOut::SIZE),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            launch(
                full_prefix,
                (&mut device, &model, &source, &count, &mut destination),
            )
            .await
            .unwrap();
            let actual = destination
                .to_host::<m![FullRows, FullOut]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            if rows > 0 {
                let expected = launch(full_fixed, (&mut device, &model, &source))
                    .await
                    .unwrap()
                    .to_host::<m![FullRows, FullOut]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec();
                for at in 0..rows * FullOut::SIZE {
                    assert_eq!(
                        actual[at].to_bits(),
                        expected[at].to_bits(),
                        "work {rows}, element {at}"
                    );
                }
            }
            assert!(
                actual[rows * FullOut::SIZE..]
                    .iter()
                    .all(|value| value.to_bits() == bf16::from_f32(SENTINEL).to_bits()),
                "work {rows}: allocation tail changed"
            );
        }
    }

    /// One compiled entry handles empty, single, paired, odd/even and full work. Source NaNs past
    /// each prefix expose padded computation, and reused destination sentinels detect stray stores.
    /// The empty and short cases also exercise prefetch bounds. Owned and borrowed
    /// projections, in both public output forms, match independent FP64 dots.
    #[tokio::test(flavor = "current_thread")]
    async fn loaded_grouped_prefixes_match_fp64_and_preserve_allocation_tails() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let (_directory, source) = checkpoint();
        let module: Owned = source.load(&mut device).unwrap();
        let borrowed = Borrowed::load(&mut Context::new(None), &module).unwrap();
        let reference = reference();
        let sentinel = || {
            HostTensor::<bf16, m![R, Out]>::from_vec(std::iter::repeat_n(
                bf16::from_f32(SENTINEL),
                R::SIZE * Out::SIZE,
            ))
        };
        let mut destination: Output = sentinel().to_hbm(&mut device.pdma).await.unwrap();
        for blocks in [0, 1, 2, 3, 4, 7, 8, 1, 0] {
            let rows = blocks * B::SIZE;
            let prefix = BlockPrefix::<(), R, B>::try_over(rows).unwrap();
            let count = HostTensor::<i32, m![1]>::from_vec([prefix.blocks() as i32])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let values: Input =
                HostTensor::<bf16, m![R, In]>::from_vec((0..R::SIZE * In::SIZE).map(|at| {
                    if at / In::SIZE < rows {
                        input(at / In::SIZE, at % In::SIZE)
                    } else {
                        bf16::from_bits(POISON)
                    }
                }))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            sentinel()
                .to_hbm(&mut device.pdma)
                .output(&mut destination)
                .await
                .unwrap();
            launch(
                into,
                (&mut device, &borrowed, &values, &count, &mut destination),
            )
            .await
            .unwrap();
            let got = destination
                .to_host::<m![R, Out]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            check(&got, rows, true, &reference);
            let result = launch(owned, (&mut device, &module, &values, &count))
                .await
                .unwrap();
            if rows > 0 {
                let bytes = result
                    .into_buffer()
                    .slice(.., ..rows * Out::SIZE * 2)
                    .unwrap()
                    .to_host(&mut device.pdma)
                    .await
                    .unwrap();
                let got: Vec<_> = bytes
                    .chunks_exact(2)
                    .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                    .collect();
                check(&got, rows, false, &reference);
            }
        }
    }
}
