//! The gated residual: a branch, times a gate, added to the residual stream.

use crate::Op;
use crate::load::Load;
use crate::ops::index::packed::{self, Packed, Table};
use crate::ops::normalization::adaptive::blocks::{Block, Part};
use crate::ops::normalization::adaptive::rows;
use crate::tensor::image::{Rows, Vector};
use furiosa_opt_std::prelude::*;

/// Add a residual multiplied by a broadcast scalar or per-channel gate.
#[derive(Default, DeviceSend, Load)]
pub struct Add;

impl<W: AxisName, R: AxisName> Op<(&Rows<W, R>, &Rows<W, R>, &Vector<W>)> for Add {
    type Output = Rows<W, R>;

    fn forward(&self, _: &mut Device, _: (&Rows<W, R>, &Rows<W, R>, &Vector<W>)) -> Self::Output {
        todo!("NPU residual addition with a broadcast per-channel gate")
    }
}

impl<D: Scalar, W: AxisName, R: AxisName>
    Op<(
        &HbmTensor<D, m![1], m![R, W]>,
        &HbmTensor<D, m![1], m![R, W]>,
        &HbmTensor<f32, m![1], m![1]>,
    )> for Add
{
    type Output = HbmTensor<D, m![1], m![R, W]>;

    fn forward(
        &self,
        _: &mut Device,
        _: (
            &HbmTensor<D, m![1], m![R, W]>,
            &HbmTensor<D, m![1], m![R, W]>,
            &HbmTensor<f32, m![1], m![1]>,
        ),
    ) -> Self::Output {
        todo!("NPU residual addition with a broadcast scalar gate")
    }
}

/// `bf16(x + gate * branch)` per row, FP32 inside, the gate the table row of the row's span.
impl<'x, 'b, 'g, T: AxisName, A: AxisName, V: AxisName, W: AxisName, Chips: M>
    Op<(
        &'x packed::Rows<Packed<T, A, V>, W, Chips>,
        &'b packed::Rows<Packed<T, A, V>, W, Chips>,
        &'g Table<W, Chips>,
    )> for Add
{
    type Output = packed::Rows<Packed<T, A, V>, W, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (input, branch, gate): (
            &'x packed::Rows<Packed<T, A, V>, W, Chips>,
            &'b packed::Rows<Packed<T, A, V>, W, Chips>,
            &'g Table<W, Chips>,
        ),
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        self.forward(device, (input, branch, &mut output, gate));
        output
    }
}

/// Every span of a packed sequence with its branch added under the span's gate row: its edge
/// strips as strips, its whole blocks as blocks.
impl<T: AxisName, A: AxisName, V: AxisName, W: AxisName, X: M>
    Op<(
        &packed::Rows<Packed<T, A, V>, W, X>,
        &packed::Rows<Packed<T, A, V>, W, X>,
        &mut packed::Rows<Packed<T, A, V>, W, X>,
        &Table<W, X>,
    )> for Add
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, branch, output, gate): (
            &packed::Rows<Packed<T, A, V>, W, X>,
            &packed::Rows<Packed<T, A, V>, W, X>,
            &mut packed::Rows<Packed<T, A, V>, W, X>,
            &Table<W, X>,
        ),
    ) {
        type R<T, A, V> = Packed<T, A, V>;
        type B<T, A, V> = Block<R<T, A, V>>;
        B::<T, A, V>::spans(
            device,
            output,
            |device, output, (first, count, index)| {
                let gate =
                    B::<T, A, V>::new().forward(&mut *device, B::<T, A, V>::row(gate, index));
                let last = first + count - 1;
                let mut current: (Part<bf16, R<T, A, V>, W, X>, Part<bf16, R<T, A, V>, W, X>) =
                    (DmTensor::new(), DmTensor::new());
                let mut next: (Part<bf16, R<T, A, V>, W, X>, Part<bf16, R<T, A, V>, W, X>) =
                    (DmTensor::new(), DmTensor::new());
                let (mut kept, mut held): (
                    Part<bf16, R<T, A, V>, W, X>,
                    Part<bf16, R<T, A, V>, W, X>,
                ) = (DmTensor::new(), DmTensor::new());
                B::<T, A, V>::new().forward(&mut *device, ((input, branch), first, &mut current));
                B::<T, A, V>::new().forward(
                    &mut *device,
                    ((input, branch), (first + 1).min(last), &mut next),
                );
                for pair in 0..(count + 1) / 2 {
                    let at = first + 2 * pair;
                    B::<T, A, V>::new()
                        .forward(&mut *device, ((&current.0, &current.1), &gate, &mut kept));
                    B::<T, A, V>::new().forward(
                        &mut *device,
                        ((input, branch), (at + 2).min(last), &mut current),
                    );
                    B::<T, A, V>::new().forward(&mut *device, (&kept, &mut *output, at));
                    B::<T, A, V>::new()
                        .forward(&mut *device, ((&next.0, &next.1), &gate, &mut held));
                    B::<T, A, V>::new().forward(
                        &mut *device,
                        ((input, branch), (at + 3).min(last), &mut next),
                    );
                    B::<T, A, V>::new()
                        .forward(&mut *device, (&held, &mut *output, (at + 1).min(last)));
                }
            },
            |device, output, strips| {
                if const { Packed::<T, A, V>::STRIP == 64 } {
                    rows::Wide::<W>::new().forward(device, (input, branch, output, gate, strips));
                } else {
                    rows::Narrow::<W>::new().forward(device, (input, branch, output, gate, strips));
                }
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::index::packed::Group;
    use crate::ops::normalization::tests::random;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    struct Fixture;

    axes![Text = 64, Audio = 64, Video = 128, Hidden = 5376];

    impl Fixture {
        async fn span_rows_gate<T: AxisName, A: AxisName, V: AxisName>() {
            let _lease = DEVICE.lock().await;
            type Sequence<T, A, V> = Packed<T, A, V>;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let width = Hidden::SIZE;
            let inputs = random(Sequence::<T, A, V>::SIZE * width, 0x2545_f491_4f6c_dd1d);
            let branches = random(Sequence::<T, A, V>::SIZE * width, 0x0123_4567_89ab_cdef);
            let gates = random(Group::SIZE * width, 0x7f4a_7c15_9e37_79b9);
            let rows = async |device: &mut Device, values: &[bf16]| {
                HostTensor::<bf16, m![{ Symbol<Sequence<T, A, V>> }, Hidden]>::from_vec(
                    values.to_vec(),
                )
                .to_hbm::<m![1], m![{ Symbol<Sequence<T, A, V>> }, Hidden]>(&mut device.pdma)
                .await
                .unwrap()
            };
            let (input, branch) = (
                rows(&mut device, &inputs).await,
                rows(&mut device, &branches).await,
            );
            let gate = HostTensor::<bf16, m![Group, Hidden]>::from_vec(gates.clone())
                .to_hbm::<m![1], m![Group, Hidden]>(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(gated_rows, (&mut device, &input, &branch, &gate))
                .await
                .unwrap()
                .to_host::<m![{ Symbol<Sequence<T, A, V>> }, Hidden]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (index, got) in output.iter().enumerate() {
                let (row, column) = (index / width, index % width);
                let table = match row {
                    row if row < T::SIZE => 1,
                    row if row < T::SIZE + A::SIZE => 5,
                    _ => 0,
                };
                let expected = f64::from(inputs[index].to_f32())
                    + f64::from(gates[table * width + column].to_f32())
                        * f64::from(branches[index].to_f32());
                let got = f64::from(got.to_f32());
                assert!(
                    (got - expected).abs() <= expected.abs() / 256.0 + 1e-4,
                    "row {row} column {column}: {got} vs {expected}"
                );
            }
        }
    }

    axes![
        TextEnvelope = 128,
        AudioEnvelope = 448,
        ShortVideo = 384,
        LongVideo = 960,
        MidVideo = 704
    ];

    #[device(chip = 1)]
    fn gated_rows<T: AxisName, A: AxisName, V: AxisName>(
        device: &mut Device,
        input: &packed::Rows<Packed<T, A, V>, Hidden>,
        branch: &packed::Rows<Packed<T, A, V>, Hidden>,
        gate: &Table<Hidden>,
    ) -> packed::Rows<Packed<T, A, V>, Hidden> {
        Add.forward(device, (input, branch, gate))
    }
    /// Every row adds its branch times its span's gate row, as the FP64 sum rounded once, over
    /// spans of strips only and of strips and whole blocks.
    #[tokio::test(flavor = "current_thread")]
    async fn spans_gate_their_branch() {
        Fixture::span_rows_gate::<Text, Audio, ShortVideo>().await;
        Fixture::span_rows_gate::<TextEnvelope, AudioEnvelope, LongVideo>().await;
        Fixture::span_rows_gate::<TextEnvelope, AudioEnvelope, MidVideo>().await;
    }
}
