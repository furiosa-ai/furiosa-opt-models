//! Rows split over the chips of a placement: every chip's part gathered onto every chip, or every
//! row's column parts exchanged for every column of a part of the rows.
//!
//! - The rows of all chips are the parts in chip order, each `S` rows: `Pieces<S, N>`.
//! - A collective is one HBM-to-HBM transfer from each chip to each chip, its own included.

use std::marker::PhantomData;

use crate::Module;
use crate::ops::layout::axes::{Parallel, Part, Shard};
use furiosa_opt_std::prelude::*;

use super::ChipMap;

/// The rows of `N` chips' parts of `S` rows each, the parts in chip order.
#[derive(Clone, Debug)]
pub struct Pieces<S, const N: usize>(PhantomData<S>);

impl<S: AxisName, const N: usize> AxisName for Pieces<S, N> {
    const NAME: Ident = Ident::new("Pieces");
    const SIZE: usize = N * S::SIZE;
}

/// The chips whose parts make up the rows, as an axis of the rows.
#[derive(Clone, Debug)]
pub struct Owners<const N: usize>;

impl<const N: usize> AxisName for Owners<N> {
    const NAME: Ident = Ident::new("Owners");
    const SIZE: usize = N;
}

type O<const N: usize> = Symbol<Owners<N>>;
type Rows<D, const N: usize, R, W> = HbmTensor<D, ChipMap<N>, m![R, W]>;

/// Every chip's rows on every chip, the parts in chip order.
#[derive(DeviceSend)]
pub struct Gather<P>(PhantomData<P>);

impl<P> Default for Gather<P> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<'x, D: MaterializableScalar, S: AxisName, W: AxisName, const N: usize>
    Module<(&mut Device, &'x Rows<D, N, S, W>)> for Gather<Parallel<N>>
{
    type Output = Rows<D, N, Pieces<S, N>, W>;

    fn forward(&self, (device, input): (&mut Device, &'x Rows<D, N, S, W>)) -> Self::Output {
        let mut output: HbmTensor<D, Broadcast<N>, m![{ Symbol<Shard<N>> }, S, W]> =
            HbmTensor::new();
        input
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
        // SAFETY: every chip holds the parts in chip order, each its rows in order.
        unsafe { output.reshape() }
    }
}

/// Every row's part of the columns on each chip exchanged for every column of the chip's part of
/// the rows: chip `t` receives rows `t * S..` of each chip's columns, the columns in chip order.
#[derive(DeviceSend)]
pub struct Exchange<P>(PhantomData<P>);

impl<P> Default for Exchange<P> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<'x, D: MaterializableScalar, S: AxisName, W: AxisName, const N: usize>
    Module<(&mut Device, &'x Rows<D, N, Pieces<S, N>, Part<W, N>>)> for Exchange<Parallel<N>>
{
    type Output = Rows<D, N, S, W>;

    fn forward(
        &self,
        (device, input): (&mut Device, &'x Rows<D, N, Pieces<S, N>, Part<W, N>>),
    ) -> Self::Output {
        type L<W, const N: usize> = Symbol<Part<W, N>>;
        // SAFETY: the rows are the parts in chip order, each its rows in order.
        let rows = unsafe {
            input
                .view()
                .reshape::<ChipMap<N>, m![{ O<N> }, S, { L<W, N> }]>()
        };
        let mut output: HbmTensor<D, m![{ O<N> }], m![S, { Symbol<Shard<N>> }, { L<W, N> }]> =
            HbmTensor::new();
        rows.to_hbm_view(&mut device.tdma, output.view_mut());
        // SAFETY: chip `t` holds part `t` of the rows, its columns the parts in chip order.
        unsafe { output.reshape() }
    }
}

#[cfg(test)]
mod tests {
    struct Fixture;
    use super::*;
    use furiosa_opt_std::runtime::launch;

    axes![Width = 5376, Attended = 7168];

    #[derive(Clone, Debug)]
    struct Part4<const ROWS: usize>;

    impl<const ROWS: usize> AxisName for Part4<ROWS> {
        const NAME: Ident = Ident::new("Shard");
        const SIZE: usize = ROWS;
    }

    type Small = Part4<1024>;
    type Local = Part<Attended, 4>;

    #[device(chip = 4)]
    fn gather_small(
        device: &mut Device,
        input: &Rows<bf16, 4, Small, Width>,
    ) -> Rows<bf16, 4, Pieces<Small, 4>, Width> {
        Gather::<Parallel<4>>::default().forward((device, input))
    }

    #[device(chip = 4)]
    fn exchange_small(
        device: &mut Device,
        input: &Rows<bf16, 4, Pieces<Small, 4>, Local>,
    ) -> Rows<bf16, 4, Small, Attended> {
        Exchange::<Parallel<4>>::default().forward((device, input))
    }

    /// Element `(row, column)` of the rows of every chip: a value each neighbor differs in.
    impl Fixture {
        fn value(row: usize, column: usize) -> bf16 {
            bf16::from_f32(f32::from(
                u16::try_from((row * 131 + column * 17) % 251).unwrap(),
            ))
        }
    }
    /// The gathered rows are every chip's part in chip order, on every chip; the exchanged rows are
    /// the chip's part of the rows with every chip's columns in chip order.
    #[tokio::test(flavor = "current_thread")]
    async fn gathers_and_exchanges_the_parts() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(gather_small.topology()).unwrap();
        let (rows, width) = (Small::SIZE, Width::SIZE);
        let parts: Vec<bf16> = (0..4 * rows)
            .flat_map(|row| (0..width).map(move |column| Fixture::value(row, column)))
            .collect();
        let input = HostTensor::<bf16, m![{ Symbol<Shard<4>> }, Small, Width]>::from_vec(parts)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let gathered = launch(gather_small, (&mut device, &input))
            .await
            .unwrap()
            .to_host::<m![{ Symbol<Shard<4>> }, { Symbol<Pieces<Small, 4>> }, Width]>(
                &mut device.pdma,
            )
            .await
            .unwrap()
            .into_vec();
        for (chip, rows) in gathered.chunks_exact(4 * rows * width).enumerate() {
            for (at, got) in rows.iter().enumerate() {
                assert_eq!(
                    *got,
                    Fixture::value(at / width, at % width),
                    "chip {chip} element {at}"
                );
            }
        }

        let (local, attended) = (Attended::SIZE / 4, Attended::SIZE);
        let columns: Vec<bf16> = (0..4)
            .flat_map(|chip| {
                (0..4 * rows).flat_map(move |row| {
                    (0..local).map(move |column| Fixture::value(row, chip * local + column))
                })
            })
            .collect();
        let input = HostTensor::<
            bf16,
            m![
                { Symbol<Shard<4>> },
                { Symbol<Pieces<Small, 4>> },
                Local
            ],
        >::from_vec(columns)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let exchanged = launch(exchange_small, (&mut device, &input))
            .await
            .unwrap()
            .to_host::<m![{ Symbol<Shard<4>> }, Small, Attended]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (chip, part) in exchanged.chunks_exact(rows * attended).enumerate() {
            for (at, got) in part.iter().enumerate() {
                let (row, column) = (chip * rows + at / attended, at % attended);
                assert_eq!(
                    *got,
                    Fixture::value(row, column),
                    "chip {chip} element {at}"
                );
            }
        }
    }
}
