//! Whole compact-prefix exchanges inside one capacity and one runtime operation.

use super::{ChipMap, Exchange, O, Pieces, Rows};
use crate::tensor::{
    live::{Extents, Live, Next, PrefixPlan},
    parallel::{Parallel, Part, Shard},
};
use furiosa_opt_std::prelude::*;

impl<const N: usize> Exchange<Parallel<N>> {
    /// Exchange one initialized prefix with one whole TDMA copy, preserving destination tails.
    /// The checked prefix guarantees a zero start and membership in `G`.
    /// Source parts use that compact row count as their stride, within allocation capacity `C`.
    /// Only the DMA mapping varies internally; callers use the same operation for every request.
    pub fn write_prefix<
        D: MaterializableScalar,
        C: AxisName,
        W: AxisName,
        G: Extents,
        B: AxisName,
    >(
        &self,
        device: &mut Device,
        input: &PrefixPlan<&Split<D, N, C, W>, C, G, B>,
        output: &mut Rows<D, N, C, W>,
    ) {
        let prefix = input.prefix();
        self.write_selected::<D, C, W, G::First, G::Last, G::Step>(
            device,
            &prefix.interval().with(prefix.value()),
            output,
        );
    }

    fn write_selected<
        D: MaterializableScalar,
        C: AxisName,
        W: AxisName,
        Current: AxisName,
        Last: AxisName,
        Step: AxisName,
    >(
        &self,
        device: &mut Device,
        input: &Live<&Split<D, N, C, W>, C>,
        output: &mut Rows<D, N, C, W>,
    ) {
        // Keep `if const`: the pinned compiler prunes the recursive arm at `Last` before
        // instantiating `Next`. A plain `if` keeps growing the type and hits the recursion limit.
        if const { Current::SIZE < Last::SIZE } {
            if input.rows() == Current::SIZE {
                self.write_extent::<D, Current, C, W>(device, input.value(), output);
            } else {
                self.write_selected::<D, C, W, Next<Current, Step>, Last, Step>(
                    device, input, output,
                );
            }
        } else {
            self.write_extent::<D, Current, C, W>(device, input.value(), output);
        }
    }

    fn write_extent<D: MaterializableScalar, S: AxisName, C: AxisName, W: AxisName>(
        &self,
        device: &mut Device,
        input: &Split<D, N, C, W>,
        output: &mut Rows<D, N, C, W>,
    ) {
        const { assert!(N > 0 && W::SIZE > 0 && S::SIZE > 0 && S::SIZE <= C::SIZE) };
        type L<W, const N: usize> = Symbol<Part<W, N>>;
        // SAFETY: the first alias keeps the full physical extent and declares explicit outer
        // padding. The second names only that initialized contiguous prefix, preserving byte
        // order. Neither alias writes, and the original input owner stays borrowed throughout.
        let padded: PaddedRows<D, N, S, C, W> = unsafe { input.view().alias() };
        let compact: Rows<D, N, Pieces<S, N>, Part<W, N>> = unsafe { padded.view().alias() };
        // SAFETY: the compact parts are in chip order, each its rows in order.
        let source = unsafe {
            compact
                .view()
                .reshape::<ChipMap<N>, m![{ O<N> }, S, { L<W, N> }]>()
        };
        // SAFETY: this declares only outer padding; the physical extent and row-major byte
        // order remain C * W. The selected prefix keeps the original mutable borrow.
        let destination = unsafe { output.view_mut().reshape::<ChipMap<N>, m![S # C, W]>() };
        // SAFETY: a row's columns are the source chips' contiguous parts. Retain the outer
        // allocation padding so the view keeps the original destination's writable provenance.
        let destination = unsafe {
            destination.reshape::<m![{ O<N> }], m![S # C, { Symbol<Shard<N>> }, { L<W, N> }]>()
        };
        source.to_hbm_view(&mut device.tdma, destination);
    }
}

type Split<D, const N: usize, C, W> = Rows<D, N, Pieces<C, N>, Part<W, N>>;

type PaddedRows<D, const N: usize, S, C, W> = HbmTensor<
    D,
    ChipMap<N>,
    m![{ Symbol<Pieces<S, N>> } # ({ Symbol<Pieces<C, N>> }), { Symbol<Part<W, N>> }],
>;

#[cfg(test)]
mod tests {
    //! Whole TDMA exchange of compact parts inside capacity-backed allocations.

    use super::super::tests::{Chips, Input, Narrow, Output, Width, assert_output, buffers};
    use crate::ops::collective::Exchange;
    use crate::tensor::axis::Rows;
    use crate::tensor::live::{BlockPrefix, ExtentRange, PrefixInputs, PrefixPlan};
    use crate::tensor::parallel::Parallel;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    type Work<C, First, Last, Step> =
        PrefixInputs<C, ExtentRange<First, Last, Step>, Rows<1>, Chips>;

    #[device(chip = 4)]
    fn device_prefix<C: AxisName, W: AxisName, First: AxisName, Last: AxisName, Step: AxisName>(
        device: &mut Device,
        input: &Input<C, W>,
        output: &mut Output<C, W>,
        rows: &Work<C, First, Last, Step>,
        _geometries: &std::marker::PhantomData<(First, Last, Step)>,
    ) {
        let rows = rows.read();
        Exchange::<Parallel<4>>::default()
            .write_prefix::<bf16, C, W, ExtentRange<First, Last, Step>, Rows<1>>(
                device,
                &rows.with(input),
                output,
            );
    }

    async fn check_device_prefix<
        C: AxisName,
        W: AxisName,
        First: AxisName,
        Last: AxisName,
        Step: AxisName,
    >() {
        let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
        let geometries = std::marker::PhantomData::<(First, Last, Step)>;
        // One compiled entry serves alternating sizes, including a return to the first size.
        for work in [
            First::SIZE,
            Last::SIZE,
            First::SIZE + Step::SIZE,
            First::SIZE,
        ] {
            let (input, mut output) = buffers::<C, W>(&mut device, work).await;
            let plan = PrefixPlan::<(), C, ExtentRange<First, Last, Step>, Rows<1>>::new(
                BlockPrefix::try_over(work).unwrap(),
            )
            .unwrap();
            let rows = Work::<C, First, Last, Step>::upload(&mut device, &plan)
                .await
                .unwrap();
            for _ in 0..2 {
                launch(
                    device_prefix,
                    (&mut device, &input, &mut output, &rows, &geometries),
                )
                .await
                .unwrap();
            }
            assert_output(&mut device, &output, work).await;
        }
    }

    /// The same prefix can come from a preceding operation's capacity-shaped output inside a launch.
    #[tokio::test(flavor = "current_thread")]
    async fn device_prefix_exchange_preserves_rows_and_capacity_tail() {
        let _lease = crate::tests::DEVICE.lock().await;
        check_device_prefix::<Rows<1280>, Narrow, Rows<256>, Rows<1280>, Rows<512>>().await;
        check_device_prefix::<Rows<10496>, Width, Rows<9472>, Rows<9984>, Rows<256>>().await;
    }
}
