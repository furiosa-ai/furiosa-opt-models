//! Elementwise arithmetic over a checked, initialized prefix of a row allocation.

use super::add::Add;
use super::prefix::windows;
use super::subtract::Subtract;
use super::sum::Windowed;
use crate::Op;
use crate::tensor::live::{BlockPrefix, Live};
use furiosa_opt_std::prelude::*;

type Shape<R, W> = m![R, W];
type Operands<'a, D, R, W, C, B> = BlockPrefix<
    (
        &'a HbmTensor<D, C, Shape<R, W>>,
        &'a HbmTensor<D, C, Shape<R, W>>,
    ),
    R,
    B,
>;

impl<D: super::Float, R: AxisName, W: AxisName, C: M, B: AxisName> Op<&Operands<'_, D, R, W, C, B>>
    for Subtract
{
    type Output = Live<HbmTensor<D, C, Shape<R, W>>, R>;

    fn forward(&self, device: &mut Device, input: &Operands<'_, D, R, W, C, B>) -> Self::Output {
        let result = Windowed::<Shape<R, W>, Shape<B, W>>::map_windows(
            *input.value(),
            device,
            windows::<R, W, B, _>(input),
            self,
        );
        input.interval().with(result)
    }
}

impl<D: super::Float, R: AxisName, W: AxisName, C: M, B: AxisName> Op<&Operands<'_, D, R, W, C, B>>
    for Add
{
    type Output = Live<HbmTensor<D, C, Shape<R, W>>, R>;

    fn forward(&self, device: &mut Device, input: &Operands<'_, D, R, W, C, B>) -> Self::Output {
        let result = Windowed::<Shape<R, W>, Shape<B, W>>::map_windows(
            *input.value(),
            device,
            windows::<R, W, B, _>(input),
            self,
        );
        input.interval().with(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{Add, Subtract};
    use crate::Op;
    use crate::ops::index::scatter::Store;
    use crate::ops::pointwise::subtract::tests::{
        Chips, Host, Part, ReferenceFeatures, rounding_terms,
    };
    use crate::tensor::{live::BlockPrefix, scalar::Value};
    use furiosa_opt_std::{prelude::*, runtime::launch};

    axes![Capacity = 256, Block = 64];
    type Input = Part<Capacity, ReferenceFeatures>;

    #[device(chip = 4)]
    fn arithmetic(
        device: &mut Device,
        (left, right): (&Input, &Input),
        blocks: &Value<Chips>,
        (sum, difference): (&mut Input, &mut Input),
    ) {
        let prefix = BlockPrefix::<(), Capacity, Block>::read(blocks);
        let operands = prefix.with((left, right));
        let added = Add.forward(&mut *device, &operands);
        let subtracted = Subtract.forward(&mut *device, &operands);
        Store.forward(&mut *device, (&prefix.with(added.value()), sum));
        Store.forward(device, (&prefix.with(subtracted.value()), difference));
    }

    /// Every initialized result bit has the ordinary BF16 rounding. NaN source suffixes are
    /// unread, and initialized destination suffixes are unchanged, including an empty prefix.
    #[tokio::test(flavor = "current_thread")]
    async fn computes_and_stores_only_initialized_rows() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(arithmetic.topology()).unwrap();
        let elements = Capacity::SIZE * ReferenceFeatures::SIZE;
        let (left, right) = rounding_terms(elements);
        let sentinel = bf16::from_f32(-9.0);
        for rows in [0, 64, 128, 192, Capacity::SIZE] {
            let active = rows * ReferenceFeatures::SIZE;
            let prefix = BlockPrefix::<(), Capacity, Block>::try_over(rows).unwrap();
            let mut source_left = left.clone();
            let mut source_right = right.clone();
            for chip in 0..4 {
                source_left[chip * elements + active..(chip + 1) * elements]
                    .fill(bf16::from_f32(f32::NAN));
                source_right[chip * elements + active..(chip + 1) * elements]
                    .fill(bf16::from_f32(f32::NAN));
            }
            let source_left: Input = Host::<Capacity, ReferenceFeatures>::from_vec(source_left)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let source_right: Input = Host::<Capacity, ReferenceFeatures>::from_vec(source_right)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let mut added: Input =
                Host::<Capacity, ReferenceFeatures>::from_vec(vec![sentinel; 4 * elements])
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap();
            let mut subtracted: Input =
                Host::<Capacity, ReferenceFeatures>::from_vec(vec![sentinel; 4 * elements])
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap();
            let blocks: Value<Chips> =
                HostTensor::<i32, m![1]>::from_vec([i32::try_from(prefix.blocks()).unwrap()])
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap();
            launch(
                arithmetic,
                (
                    &mut device,
                    (&source_left, &source_right),
                    &blocks,
                    (&mut added, &mut subtracted),
                ),
            )
            .await
            .unwrap();
            let added = added
                .to_host::<m![{ Chips }, Capacity, ReferenceFeatures]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let subtracted = subtracted
                .to_host::<m![{ Chips }, Capacity, ReferenceFeatures]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for index in 0..4 * elements {
                let initialized = index % elements < active;
                let want_sum = if initialized {
                    bf16::from_f32(left[index].to_f32() + right[index].to_f32())
                } else {
                    sentinel
                };
                let want_difference = if initialized {
                    bf16::from_f32(left[index].to_f32() - right[index].to_f32())
                } else {
                    sentinel
                };
                assert_eq!(
                    added[index].to_bits(),
                    want_sum.to_bits(),
                    "sum at {rows} rows, element {index}"
                );
                assert_eq!(
                    subtracted[index].to_bits(),
                    want_difference.to_bits(),
                    "difference at {rows} rows, element {index}"
                );
            }
        }
    }
}
