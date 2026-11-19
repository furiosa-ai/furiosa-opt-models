//! Whole TDMA exchange of compact parts inside capacity-backed allocations.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::ops::collective::{Exchange, Pieces};
use furiosa_opt_models::tensor::axis::Rows;
use furiosa_opt_models::tensor::parallel::{Parallel, Part, Shard};
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::{HbmBuffer, launch};

axes![Width = 7168, Narrow = 512];

type Chips = m![{ Symbol<Shard<4>> }];
type Input<C, W> = HbmTensor<bf16, Chips, m![{ Symbol<Pieces<C, 4>> }, { Symbol<Part<W, 4>> }]>;
type Output<C, W> = HbmTensor<bf16, Chips, m![C, W]>;
type Inputs<A, B, E, W> = (Input<A, W>, Input<B, W>, Input<E, W>);
type Outputs<A, B, E, W> = (Output<A, W>, Output<B, W>, Output<E, W>);

#[device(chip = 4)]
fn runtime_prefix<C: AxisName, W: AxisName, A: AxisName, B: AxisName, E: AxisName>(
    device: &mut Device,
    input: &Inputs<A, B, E, W>,
    output: &mut Outputs<A, B, E, W>,
    rows: &Value<Chips>,
    _capacity: &std::marker::PhantomData<C>,
) {
    let rows = rows.extent::<m![C]>();
    let exchange = Exchange::<Parallel<4>>::default();
    if rows == A::SIZE {
        exchange.write(device, input.0.view(), output.0.view_mut());
    } else if rows == B::SIZE {
        exchange.write(device, input.1.view(), output.1.view_mut());
    } else {
        exchange.write(device, input.2.view(), output.2.view_mut());
    }
}

#[device(chip = 4)]
fn device_prefix<C: AxisName, W: AxisName, A: AxisName, B: AxisName, E: AxisName>(
    device: &mut Device,
    input: &Input<C, W>,
    output: &mut Output<C, W>,
    rows: &Value<Chips>,
    _geometries: &std::marker::PhantomData<(A, B, E)>,
) {
    let rows = rows.extent::<m![C]>();
    let exchange = Exchange::<Parallel<4>>::default();
    if rows == A::SIZE {
        exchange.write_prefix::<bf16, A, C, W>(device, input, output);
    } else if rows == B::SIZE {
        exchange.write_prefix::<bf16, B, C, W>(device, input, output);
    } else {
        exchange.write_prefix::<bf16, E, C, W>(device, input, output);
    }
}

fn coefficient(index: usize) -> bf16 {
    bf16::from_bits(0x3f00 | ((index * 131 + index / 17) % 1024) as u16)
}

async fn check<C: AxisName, W: AxisName, A: AxisName, B: AxisName, E: AxisName>(internal: bool) {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let geometry = std::marker::PhantomData::<C>;
    let geometries = std::marker::PhantomData::<(A, B, E)>;
    let local_width = W::SIZE / 4;
    let input_per_chip = 4 * C::SIZE * local_width;
    let output_per_chip = C::SIZE * W::SIZE;
    let poison = bf16::from_bits(0x7fc1);
    // Reuse one compiled entry and alternate work sizes; no per-length model or kernel reload.
    for work in [A::SIZE, E::SIZE, B::SIZE, A::SIZE] {
        let input: Input<C, W> = HostTensor::<
            bf16,
            m![{ Chips }, { Symbol<Pieces<C, 4>> }, { Symbol<Part<W, 4>> }],
        >::from_vec((0..4 * input_per_chip).map(|index| {
            if index % input_per_chip < 4 * work * local_width {
                coefficient(index)
            } else {
                poison
            }
        }))
        .to_hbm::<Chips, m![{ Symbol<Pieces<C, 4>> }, { Symbol<Part<W, 4>> }]>(&mut device.pdma)
        .await
        .unwrap();
        let mut output: Output<C, W> = HbmBuffer::alloc_shared(&device, output_per_chip * 2)
            .unwrap()
            .try_into()
            .unwrap();
        HostTensor::<bf16, m![{ Chips }, C, W]>::from_vec(std::iter::repeat_n(
            poison,
            4 * output_per_chip,
        ))
        .to_hbm(&mut device.pdma)
        .output(&mut output)
        .await
        .unwrap();
        let rows: Value<Chips> = HostTensor::<i32, m![1]>::from_vec([work as i32])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let input = input.into_buffer();
        let output = output.into_buffer();
        let capacity_input: Input<C, W> = input.alias(..).unwrap().try_into().unwrap();
        let mut capacity_output: Output<C, W> = output.alias(..).unwrap().try_into().unwrap();
        // SAFETY: runtime_prefix reaches the destination through exactly one geometry in
        // each launch. No full-capacity destination argument or concurrent writer is bound.
        unsafe { output.partition() };
        // Prepare alternative root windows on the host. Aliases share their backing allocation;
        // only the selected geometry writes during a launch. Retain every owner through completion.
        let inputs: Inputs<A, B, E, W> = (
            input
                .alias(..4 * A::SIZE * local_width * 2)
                .unwrap()
                .try_into()
                .unwrap(),
            input
                .alias(..4 * B::SIZE * local_width * 2)
                .unwrap()
                .try_into()
                .unwrap(),
            input
                .alias(..4 * E::SIZE * local_width * 2)
                .unwrap()
                .try_into()
                .unwrap(),
        );
        let mut outputs: Outputs<A, B, E, W> = (
            output
                .alias(..A::SIZE * W::SIZE * 2)
                .unwrap()
                .try_into()
                .unwrap(),
            output
                .alias(..B::SIZE * W::SIZE * 2)
                .unwrap()
                .try_into()
                .unwrap(),
            output
                .alias(..E::SIZE * W::SIZE * 2)
                .unwrap()
                .try_into()
                .unwrap(),
        );
        // Reuse the same destination across completed writes before checking its rows and tail.
        for _ in 0..2 {
            if internal {
                launch(
                    device_prefix,
                    (
                        &mut device,
                        &capacity_input,
                        &mut capacity_output,
                        &rows,
                        &geometries,
                    ),
                )
                .await
                .unwrap();
            } else {
                launch(
                    runtime_prefix,
                    (&mut device, &inputs, &mut outputs, &rows, &geometry),
                )
                .await
                .unwrap();
            }
        }
        let output: Output<C, W> = output.try_into().unwrap();
        let result = output
            .to_host::<m![{ Chips }, C, W]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (index, actual) in result.into_iter().enumerate() {
            let chip = index / output_per_chip;
            let row = index / W::SIZE % C::SIZE;
            let column = index % W::SIZE;
            let expected = if row < work {
                let source_chip = column / local_width;
                let source_row = chip * work + row;
                coefficient(
                    source_chip * input_per_chip + source_row * local_width + column % local_width,
                )
            } else {
                poison
            };
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "work {work}, chip {chip}, row {row}, column {column}"
            );
        }
    }
}

/// Root windows must describe compact parts rather than retain capacity-strided coordinates.
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn compact_whole_exchange_preserves_prefix_and_capacity_tail() {
    check::<Rows<1280>, Narrow, Rows<256>, Rows<768>, Rows<1280>>(false).await;
    check::<Rows<10496>, Width, Rows<9472>, Rows<9728>, Rows<9984>>(false).await;
}

/// The same prefix can come from a preceding operation's capacity-shaped output inside a launch.
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn device_prefix_exchange_preserves_rows_and_capacity_tail() {
    check::<Rows<1280>, Narrow, Rows<256>, Rows<768>, Rows<1280>>(true).await;
    check::<Rows<10496>, Width, Rows<9472>, Rows<9728>, Rows<9984>>(true).await;
}
