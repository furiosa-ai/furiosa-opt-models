//! Runtime modulation spans, sharing the static block/strip arithmetic.

use super::{AdaptiveRms, blocks::Block, rows};
use crate::Op;
use crate::ops::index::packed::{Rows, SpanPlan, Table};
use crate::ops::pointwise::gated;
use crate::tensor::live::Live;
use core::borrow::Borrow;
use furiosa_opt_std::prelude::*;

/// Normalize only modulated spans and overwrite guards with exact zero. The allocation tail
/// outside `plan.work()` is untouched; every row of returned work is initialized.
impl<R: AxisName, W: AxisName, C: M, G, const EPSILON: usize, const EDGE: usize>
    Op<(
        &Rows<R, W, C>,
        &mut Rows<R, W, C>,
        &(Table<W, C>, Table<W, C>),
        &SpanPlan<R, EDGE>,
    )> for AdaptiveRms<W, EPSILON, C, G>
where
    G: Borrow<HbmTensor<bf16, C, m![W]>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, output, modulation, plan): (
            &Rows<R, W, C>,
            &mut Rows<R, W, C>,
            &(Table<W, C>, Table<W, C>),
            &SpanPlan<R, EDGE>,
        ),
    ) {
        self.apply_spans::<R, EDGE>(
            device,
            input,
            output,
            modulation,
            Block::<R>::plan(plan.spans(), EDGE),
        );
        clear_guards(device, output, plan);
    }
}

/// An owned result retains the complete work prefix, including initialized unmodulated guards.
impl<R: AxisName, W: AxisName, C: M, G, const EPSILON: usize, const EDGE: usize>
    Op<(
        &Rows<R, W, C>,
        &(Table<W, C>, Table<W, C>),
        &SpanPlan<R, EDGE>,
    )> for AdaptiveRms<W, EPSILON, C, G>
where
    G: Borrow<HbmTensor<bf16, C, m![W]>>,
{
    type Output = Live<Rows<R, W, C>, R>;

    fn forward(
        &self,
        device: &mut Device,
        (input, modulation, plan): (
            &Rows<R, W, C>,
            &(Table<W, C>, Table<W, C>),
            &SpanPlan<R, EDGE>,
        ),
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        self.forward(device, (input, &mut output, modulation, plan));
        plan.work().with(output)
    }
}

/// Add the gated branch only within modulated spans. Guards are overwritten directly, including
/// when their residual or branch contains NaN, and the allocation tail is untouched.
impl<R: AxisName, W: AxisName, C: M, const EDGE: usize>
    Op<(
        &Rows<R, W, C>,
        &Rows<R, W, C>,
        &mut Rows<R, W, C>,
        &Table<W, C>,
        &SpanPlan<R, EDGE>,
    )> for gated::Add
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, branch, output, gate, plan): (
            &Rows<R, W, C>,
            &Rows<R, W, C>,
            &mut Rows<R, W, C>,
            &Table<W, C>,
            &SpanPlan<R, EDGE>,
        ),
    ) {
        self.apply_spans::<R, W, C, EDGE>(
            device,
            input,
            branch,
            output,
            gate,
            Block::<R>::plan(plan.spans(), EDGE),
        );
        clear_guards(device, output, plan);
    }
}

impl<R: AxisName, W: AxisName, C: M, const EDGE: usize>
    Op<(
        &Rows<R, W, C>,
        &Rows<R, W, C>,
        &Table<W, C>,
        &SpanPlan<R, EDGE>,
    )> for gated::Add
{
    type Output = Live<Rows<R, W, C>, R>;

    fn forward(
        &self,
        device: &mut Device,
        (input, branch, gate, plan): (
            &Rows<R, W, C>,
            &Rows<R, W, C>,
            &Table<W, C>,
            &SpanPlan<R, EDGE>,
        ),
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        self.forward(device, (input, branch, &mut output, gate, plan));
        plan.work().with(output)
    }
}

fn clear_guards<R: AxisName, W: AxisName, C: M, const EDGE: usize>(
    device: &mut Device,
    output: &mut Rows<R, W, C>,
    plan: &SpanPlan<R, EDGE>,
) {
    let guards = plan.guards();
    let (first, count) = (guards.start() / EDGE, guards.rows() / EDGE);
    if const { EDGE == 64 } {
        rows::Wide::<W>::new().clear(device, output, first, count);
    } else {
        rows::Narrow::<W>::new().clear(device, output, first, count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::index::packed::{Group, live::SpanInputs};
    use crate::tensor::axis;
    use crate::tensor::form::Weight;
    use crate::tensor::parallel::Chip;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    const EPSILON: usize = (1e-5f32).to_bits() as usize;

    #[device(chip = 1)]
    fn into<R: AxisName, W: AxisName, const EDGE: usize>(
        device: &mut Device,
        norm: &AdaptiveRms<W, EPSILON>,
        input: &Rows<R, W>,
        branch: &Rows<R, W>,
        modulation: &(Table<W>, Table<W>),
        gate: &Table<W>,
        header: &SpanInputs<R, Chip, EDGE>,
        output: &mut (Rows<R, W>, Rows<R, W>),
    ) {
        let plan = header.read();
        norm.forward(&mut *device, (input, &mut output.0, modulation, &plan));
        gated::Add.forward(device, (input, branch, &mut output.1, gate, &plan));
    }

    #[device(chip = 1)]
    fn owned<R: AxisName, W: AxisName, const EDGE: usize>(
        device: &mut Device,
        norm: &AdaptiveRms<W, EPSILON>,
        input: &Rows<R, W>,
        branch: &Rows<R, W>,
        modulation: &(Table<W>, Table<W>),
        gate: &Table<W>,
        header: &SpanInputs<R, Chip, EDGE>,
    ) -> (Rows<R, W>, Rows<R, W>) {
        let plan = header.read();
        let norm = norm.forward(&mut *device, (input, modulation, &plan));
        let gated = gated::Add.forward(device, (input, branch, gate, &plan));
        (norm.into_value(), gated.into_value())
    }

    fn input(row: usize, column: usize) -> bf16 {
        let magnitude = [1.0 / 256.0, 1.0 / 8.0, 8.0, 512.0][row % 4];
        bf16::from_f32((((row * 13 + column * 7) % 251) as f32 - 125.0) / 64.0 * magnitude)
    }

    fn branch(row: usize, column: usize) -> bf16 {
        bf16::from_f32((((row * 17 + column * 11) % 127) as f32 - 63.0) / 32.0)
    }

    fn weight(column: usize) -> bf16 {
        bf16::from_f32((column % 17 + 1) as f32 / 16.0)
    }

    fn shift(table: usize, column: usize) -> bf16 {
        bf16::from_f32((table as f32 - 2.0) / 4.0 + (column % 7) as f32 / 32.0)
    }

    fn scale(table: usize, column: usize) -> bf16 {
        bf16::from_f32((table as f32 - 3.0) / 8.0 + (column % 5) as f32 / 64.0)
    }

    fn gate(table: usize, column: usize) -> bf16 {
        bf16::from_f32((table as f32 - 2.0) / 8.0 - (column % 3) as f32 / 16.0)
    }

    fn check<R: AxisName, W: AxisName, const EDGE: usize>(
        values: &[bf16],
        plan: &SpanPlan<R, EDGE>,
        normalized: bool,
        include_tail: bool,
    ) {
        let [text, audio, _] = plan.modality_rows();
        let rows = if include_tail {
            R::SIZE
        } else {
            plan.work_rows()
        };
        assert_eq!(values.len(), rows * W::SIZE);
        for row in 0..rows {
            let table = if row < text {
                1
            } else if row < text + audio {
                5
            } else {
                0
            };
            let inverse = if normalized && row < plan.packed_rows() {
                let square = (0..W::SIZE)
                    .map(|column| f64::from(input(row, column).to_f32()).powi(2))
                    .sum::<f64>()
                    / W::SIZE as f64;
                (square + f64::from(f32::from_bits(EPSILON as u32)))
                    .sqrt()
                    .recip()
            } else {
                0.0
            };
            for column in 0..W::SIZE {
                let actual = values[row * W::SIZE + column];
                if row >= plan.work_rows() {
                    assert_eq!(
                        actual.to_bits(),
                        bf16::from_f32(11.0).to_bits(),
                        "tail row {row} column {column}"
                    );
                } else if row >= plan.packed_rows() {
                    assert_eq!(actual.to_bits(), 0, "guard row {row} column {column}");
                } else {
                    let expected = if normalized {
                        f64::from(input(row, column).to_f32())
                            * inverse
                            * f64::from(weight(column).to_f32())
                            * (1.0 + f64::from(scale(table, column).to_f32()))
                            + f64::from(shift(table, column).to_f32())
                    } else {
                        f64::from(input(row, column).to_f32())
                            + f64::from(gate(table, column).to_f32())
                                * f64::from(branch(row, column).to_f32())
                    };
                    let rounded = bf16::from_f32(expected as f32).to_f32();
                    let got = actual.to_f32();
                    let tolerance = if normalized {
                        expected.abs() / 128.0 + 1e-3
                    } else {
                        expected.abs() / 256.0 + 1e-4
                    };
                    assert!(
                        got.is_finite() && (f64::from(got) - expected).abs() <= tolerance,
                        "norm {normalized}, row {row}, column {column}: {got} vs FP64 {expected}, BF16 {rounded}"
                    );
                }
            }
        }
    }

    async fn cases<R: AxisName, W: AxisName, const EDGE: usize>(
        device: &mut Device,
        cases: &[([usize; 3], usize)],
        owned_case: bool,
    ) {
        let norm = AdaptiveRms::<W, EPSILON>::new(Weight::new(
            HostTensor::<bf16, m![W]>::from_vec((0..W::SIZE).map(weight))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        ));
        let mut table = async |value: fn(usize, usize) -> bf16| {
            HostTensor::<bf16, m![Group, W]>::from_vec(
                (0..Group::SIZE * W::SIZE).map(|at| value(at / W::SIZE, at % W::SIZE)),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap()
        };
        let modulation = (table(shift).await, table(scale).await);
        let gate = table(gate).await;
        let mut output: (Rows<R, W>, Rows<R, W>) = (
            HostTensor::<bf16, m![R, W]>::from_vec(std::iter::repeat_n(
                bf16::from_f32(11.0),
                R::SIZE * W::SIZE,
            ))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
            HostTensor::<bf16, m![R, W]>::from_vec(std::iter::repeat_n(
                bf16::from_f32(11.0),
                R::SIZE * W::SIZE,
            ))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        );
        for (case, &(lengths, work)) in cases.iter().enumerate() {
            let plan = SpanPlan::<R, EDGE>::try_packed(lengths, work).unwrap();
            let mut upload = async |value: fn(usize, usize) -> bf16| {
                HostTensor::<bf16, m![R, W]>::from_vec((0..R::SIZE * W::SIZE).map(|at| {
                    if at / W::SIZE < plan.packed_rows() {
                        value(at / W::SIZE, at % W::SIZE)
                    } else {
                        bf16::from_bits(0x7fc1)
                    }
                }))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap()
            };
            let input = upload(input).await;
            let branch = upload(branch).await;
            let header = SpanInputs::<R, Chip, EDGE>::upload(device, &plan)
                .await
                .unwrap();
            // Reinitialize destinations, including the tail, while retaining the same allocations.
            let mut reset = async |target: &mut Rows<R, W>| {
                HostTensor::<bf16, m![R, W]>::from_vec(std::iter::repeat_n(
                    bf16::from_f32(11.0),
                    R::SIZE * W::SIZE,
                ))
                .to_hbm(&mut device.pdma)
                .output(target)
                .await
                .unwrap();
            };
            reset(&mut output.0).await;
            reset(&mut output.1).await;
            launch(
                into,
                (
                    &mut *device,
                    &norm,
                    &input,
                    &branch,
                    &modulation,
                    &gate,
                    &header,
                    &mut output,
                ),
            )
            .await
            .unwrap();
            for (normalized, tensor) in [(true, &output.0), (false, &output.1)] {
                let values = tensor
                    .to_host::<m![R, W]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec();
                check::<R, W, EDGE>(&values, &plan, normalized, true);
            }
            if owned_case && case == 1 {
                let owned_result = launch(
                    owned,
                    (
                        &mut *device,
                        &norm,
                        &input,
                        &branch,
                        &modulation,
                        &gate,
                        &header,
                    ),
                )
                .await
                .unwrap();
                for (normalized, tensor) in [(true, owned_result.0), (false, owned_result.1)] {
                    let bytes = tensor
                        .into_buffer()
                        .slice(.., ..plan.work_rows() * W::SIZE * 2)
                        .unwrap()
                        .to_host(&mut device.pdma)
                        .await
                        .unwrap();
                    let values: Vec<_> = bytes
                        .chunks_exact(2)
                        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                        .collect();
                    check::<R, W, EDGE>(&values, &plan, normalized, false);
                }
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore = "requires an explicitly selected RNGD chip"]
    async fn runtime_spans_preserve_values_guards_and_allocation_tails() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mut requests: Vec<_> = [32usize, 48, 512, 16, 32]
            .into_iter()
            .map(|text| ([text, 112, 9328], (text + 112 + 9328).div_ceil(256) * 256))
            .collect();
        requests.extend([([0, 0, 0], 256), ([0, 0, 0], 0)]);
        cases::<axis::Rows<10496>, axis::Rows<5376>, 16>(&mut device, &requests, true).await;
        cases::<axis::Rows<1536>, axis::Rows<256>, 64>(
            &mut device,
            &[([128, 128, 768], 1280)],
            false,
        )
        .await;
        cases::<axis::Rows<1280>, axis::Rows<256>, 64>(
            &mut device,
            &[([64, 64, 704], 1024)],
            false,
        )
        .await;
    }
}
