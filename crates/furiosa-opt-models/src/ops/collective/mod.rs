//! Collectives over the chips of a placement: chip-local vectors summed with a canonical FP32
//! reduction tree, every chip's part gathered onto every chip, or each chip's columns exchanged
//! for another chip's rows.

use std::marker::PhantomData;

use crate::specialize::collective;
use crate::{
    Op,
    tensor::{
        axis::Single,
        live::Live,
        parallel::{Parallel, Part, Shard},
    },
};
use furiosa_opt_std::prelude::*;
pub use gather::{AllGather, Pieces};

mod gather;

/// Sum chip-local vectors in a fixed pairwise f32 order and replicate the result on every chip.
#[derive(DeviceSend)]
pub struct AllReduce<P>(PhantomData<P>);

impl<P> Default for AllReduce<P> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<D: Scalar, W: AxisName> Op<HbmTensor<D, m![1], m![W]>> for AllReduce<Parallel<1>> {
    type Output = HbmTensor<D, m![1], m![W]>;

    fn forward(&self, _: &mut Device, input: Self::Output) -> Self::Output {
        input
    }
}

/// One chip's rows are already the sum.
impl<D: Scalar, R: AxisName, W: AxisName> Op<HbmTensor<D, m![1], m![R, W]>>
    for AllReduce<Parallel<1>>
{
    type Output = HbmTensor<D, m![1], m![R, W]>;

    fn forward(&self, _: &mut Device, input: Self::Output) -> Self::Output {
        input
    }
}

/// One chip's rows of a launch are already the sum.
impl<X, R> Op<Live<X, R>> for AllReduce<Parallel<1>> {
    type Output = Live<X, R>;

    fn forward(&self, _: &mut Device, input: Live<X, R>) -> Self::Output {
        input
    }
}

/// A vector is the one row of a sum of rows.
impl<D: OutputScalar, W: AxisName, const N: usize> Op<HbmTensor<D, ChipMap<N>, m![W]>>
    for AllReduce<Parallel<N>>
{
    type Output = HbmTensor<D, ChipMap<N>, m![W]>;

    fn forward(&self, device: &mut Device, input: Self::Output) -> Self::Output {
        // SAFETY: a vector is its one row, every cell at its address.
        let row = unsafe { input.reshape::<ChipMap<N>, m![Single, W]>() };
        // SAFETY: as above, back to the vector.
        unsafe { self.forward(device, row).reshape() }
    }
}

/// Every row summed over the chips: FP32 pairwise rounds, then one narrowing.
impl<D: OutputScalar, R: AxisName, W: AxisName, const N: usize>
    Op<HbmTensor<D, ChipMap<N>, m![R, W]>> for AllReduce<Parallel<N>>
{
    type Output = HbmTensor<D, ChipMap<N>, m![R, W]>;

    fn forward(&self, device: &mut Device, input: Self::Output) -> Self::Output {
        type T<R, W> = Tile<R, W>;
        type Passes<R, W> = m![{ Symbol<Tiles<R, W>> }];
        const { assert!(W::SIZE > 0 && W::SIZE % 1024 == 0 && W::SIZE <= 32768) };
        // SAFETY: the passes split the rows in order; every cell keeps its address.
        let rows = unsafe {
            input
                .view()
                .reshape::<ChipMap<N>, m![{ Symbol<Tiles<R, W>> }, { Symbol<T<R, W>> }, W]>()
        };
        let mut output: HbmTensor<D, ChipMap<N>, m![R, W]> = HbmTensor::new();
        for pass in 0..<Passes<R, W> as M>::SIZE {
            let tile: Local<D, ChipMap<N>, T<R, W>, W> = rows
                .tile::<Passes<R, W>, 1, m![
                    { Symbol<Tiles<R, W>> } = 1 # ({ Symbol<Tiles<R, W>> }),
                    { Symbol<T<R, W>> },
                    W
                ]>(pass)
                .to_dm(&mut device.tdma);
            let mut sum: Local<f32, ChipMap<N>, T<R, W>, W> = device
                .main
                .begin(tile.view())
                .fetch::<m![{ Symbol<T<R, W>> }, W / 1024], m![W % 4 # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Symbol<T<R, W>> }, W / 1024], m![W % 4 # 8]>()
                .commit_trim::<m![W % 4]>()
                .commit();
            #[unroll]
            for round in 0..const { collective::rounds(N) } {
                sum = self.reduce::<T<R, W>, W>(device, sum, round);
            }
            // SAFETY: as for the input.
            let target = unsafe {
                output
                    .view_mut()
                    .reshape::<ChipMap<N>, m![{ Symbol<Tiles<R, W>> }, { Symbol<T<R, W>> }, W]>()
            };
            D::narrow::<ChipMap<N>, T<R, W>, W>(device, sum)
                .view()
                .to_hbm_view(
                    &mut device.tdma,
                    target.tile::<Passes<R, W>, 1, m![
                        { Symbol<Tiles<R, W>> } = 1 #{!} ({ Symbol<Tiles<R, W>> }),
                        { Symbol<T<R, W>> },
                        W
                    ]>(pass),
                );
        }
        output
    }
}

impl<const N: usize> AllReduce<Parallel<N>> {
    /// The pairwise sums of round `round`, every row at once.
    fn reduce<R: AxisName, W: AxisName>(
        &self,
        device: &mut Device,
        input: Local<f32, ChipMap<N>, R, W>,
        round: usize,
    ) -> Local<f32, ChipMap<N>, R, W> {
        let exchanged = input
            .chip_shuffle(const { collective::partners::<N>() }[round])
            .to_dm::<m![R, W / 1024, W % 4]>(&mut device.tdma);
        let mut pair: DmTensor<
            f32,
            ChipMap<N>,
            m![1 # 2],
            ReductionSlices<W>,
            m![Two, R, W / 1024, W % 4],
        > = DmTensor::new();
        input.view().to_dm_view(
            &mut device.tdma,
            pair.view_mut()
                .tile::<m![Two], 1, m![Two = 1 #{!} 2, R, W / 1024, W % 4]>(0),
        );
        exchanged.view().to_dm_view(
            &mut device.tdma,
            pair.view_mut()
                .tile::<m![Two], 1, m![Two = 1 #{!} 2, R, W / 1024, W % 4]>(1),
        );
        let low = pair
            .chip_slice::<m![Two], m![R, W / 1024, W % 4]>(
                const { collective::selections::<N>(false) }[round],
            )
            .to_dm::<m![R, W / 1024, W % 4]>(&mut device.tdma);
        let high = pair
            .chip_slice::<m![Two], m![R, W / 1024, W % 4]>(
                const { collective::selections::<N>(true) }[round],
            )
            .to_dm::<m![R, W / 1024, W % 4]>(&mut device.tdma);
        let high: VrfTensor<
            f32,
            ChipMap<N>,
            m![1 # 2],
            ReductionSlices<W>,
            m![R, W / 1024, W % 4 # 8],
        > = device
            .sub
            .begin(high.view())
            .fetch::<m![R, W / 1024], m![W % 4 # 8]>()
            .collect::<m![R, W / 1024], m![W % 4 # 8]>()
            .to_vrf();
        // Both partners use low + high, including when NaN operand order matters.
        device
            .main
            .begin(low.view())
            .fetch::<m![R, W / 1024], m![W % 4 # 8]>()
            .collect::<m![R, W / 1024], m![W % 4 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R, W / 1024, W % 4 / 4 # 2], m![W % 4]>()
            .vector_fp_binary(FpBinaryOp::AddF, &high)
            .vector_widen_concat::<m![R, W / 1024], m![W % 4 # 8]>()
            .vector_final()
            .commit_trim::<m![W % 4]>()
            .commit()
    }
}

/// The chips whose parts make up the rows, as an axis of the rows.
#[derive(Clone, Debug)]
pub struct Owners<const N: usize>;

impl<const N: usize> AxisName for Owners<N> {
    const NAME: Ident = Ident::new("Owners");
    const SIZE: usize = N;
}

type O<const N: usize> = Symbol<Owners<N>>;

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
    Op<&'x Rows<D, N, Pieces<S, N>, Part<W, N>>> for Exchange<Parallel<N>>
{
    type Output = Rows<D, N, S, W>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Rows<D, N, Pieces<S, N>, Part<W, N>>,
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

axes![Two = 2];

type ChipMap<const N: usize> = m![{ Symbol<Shard<N>> }];
type Rows<D, const N: usize, R, W> = HbmTensor<D, ChipMap<N>, m![R, W]>;
type ReductionSlices<W> = m![W % 1024 / 128, W % 128 / 4];
/// A chip's rows in the sum's layout: each slice one part of every row.
type Local<D, Chips, R, W> =
    DmTensor<D, Chips, m![1 # 2], ReductionSlices<W>, m![R, W / 1024, W % 4]>;

/// The rows one pass of a sum of rows takes: the partner's rows of a pass take one 32-byte VRF
/// packet for each row and kilofeature, at most 4 KiB.
#[derive(Clone, Debug)]
struct Tile<R, W>(PhantomData<(R, W)>);

impl<R: AxisName, W: AxisName> AxisName for Tile<R, W> {
    const NAME: Ident = Ident::new("SumRows");
    const SIZE: usize = {
        let mut rows = 4096 / (32 * (W::SIZE / 1024));
        while !R::SIZE.is_multiple_of(rows) {
            rows -= 1;
        }
        rows
    };
}

/// The passes of a sum of rows.
#[derive(Clone, Debug)]
struct Tiles<R, W>(PhantomData<(R, W)>);

impl<R: AxisName, W: AxisName> AxisName for Tiles<R, W> {
    const NAME: Ident = Ident::new("SumPasses");
    const SIZE: usize = R::SIZE / Tile::<R, W>::SIZE;
}

/// Floating-point storage supported by the chip sum.
pub(crate) trait OutputScalar: MaterializableScalar + FetchCast<f32> {
    /// Convert the completed f32 reduction of every row to its output element type.
    fn narrow<Chips: M, R: AxisName, W: AxisName>(
        device: &mut Device,
        input: Local<f32, Chips, R, W>,
    ) -> Local<Self, Chips, R, W>;
}

impl OutputScalar for f32 {
    fn narrow<Chips: M, R: AxisName, W: AxisName>(
        _: &mut Device,
        input: Local<f32, Chips, R, W>,
    ) -> Local<Self, Chips, R, W> {
        input
    }
}

impl OutputScalar for bf16 {
    fn narrow<Chips: M, R: AxisName, W: AxisName>(
        device: &mut Device,
        input: Local<f32, Chips, R, W>,
    ) -> Local<Self, Chips, R, W> {
        device
            .main
            .begin(input.view())
            .fetch::<m![R, W / 1024], m![W % 4 # 8]>()
            .collect::<m![R, W / 1024], m![W % 4 # 8]>()
            .cast::<bf16, m![W % 4 # 16]>()
            .commit_trim::<m![W % 4]>()
            .commit()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;
    use std::iter;

    axes![Hidden = 5120, Attended = 7168];

    #[derive(Clone, Debug)]
    pub(crate) struct Part4<const ROWS: usize>;

    impl<const ROWS: usize> AxisName for Part4<ROWS> {
        const NAME: Ident = Ident::new("Shard");
        const SIZE: usize = ROWS;
    }

    /// A chip's part of the rows.
    pub(crate) type Small = Part4<1024>;
    type Local = Part<Attended, 4>;

    /// Element `(row, column)` of the rows of every chip: a value each neighbor differs in.
    pub(crate) fn value(row: usize, column: usize) -> bf16 {
        bf16::from_f32(f32::from(
            u16::try_from((row * 131 + column * 17) % 251).unwrap(),
        ))
    }

    #[device(chip = 2)]
    fn sum2(
        device: &mut Device,
        input: HbmTensor<f32, ChipMap<2>, m![Hidden]>,
    ) -> HbmTensor<f32, ChipMap<2>, m![Hidden]> {
        AllReduce::<Parallel<2>>::default().forward(device, input)
    }

    #[device(chip = 4)]
    fn sum4(
        device: &mut Device,
        input: HbmTensor<bf16, ChipMap<4>, m![Hidden]>,
    ) -> HbmTensor<bf16, ChipMap<4>, m![Hidden]> {
        AllReduce::<Parallel<4>>::default().forward(device, input)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sums_two_chips() {
        let mut device = Device::new(sum2.topology()).unwrap();
        let values: Vec<_> = (0_u16..2)
            .flat_map(|chip| {
                (0..Hidden::SIZE).map(move |i| {
                    f32::from(u16::try_from(i % 101).unwrap()) / 8.0 + f32::from(chip)
                })
            })
            .collect();
        let expected: Vec<_> = values[..Hidden::SIZE]
            .iter()
            .zip(&values[Hidden::SIZE..])
            .map(|(low, high)| (low + high).to_bits())
            .collect();
        let input = HostTensor::<f32, m![{ Symbol<Shard<2>> }, Hidden]>::from_vec(values)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(sum2, (&mut device, input))
            .await
            .unwrap()
            .to_host::<m![{ Symbol<Shard<2>> }, Hidden]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for chip in output.chunks_exact(Hidden::SIZE) {
            assert_eq!(
                chip.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rounds_after_sum() {
        let mut device = Device::new(sum4.topology()).unwrap();
        let input = HostTensor::<bf16, m![{ Symbol<Shard<4>> }, Hidden]>::from_vec(
            [256.0_f32, 1.0, -256.0, 1.0]
                .into_iter()
                .flat_map(|value| iter::repeat_n(bf16::from_f32(value), Hidden::SIZE)),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let output = launch(sum4, (&mut device, input))
            .await
            .unwrap()
            .to_host::<m![{ Symbol<Shard<4>> }, Hidden]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        assert!(output.iter().all(|value| *value == bf16::from_f32(2.0)));
    }

    #[device(chip = 4)]
    fn exchange_small(
        device: &mut Device,
        input: &Rows<bf16, 4, Pieces<Small, 4>, Local>,
    ) -> Rows<bf16, 4, Small, Attended> {
        Exchange::<Parallel<4>>::default().forward(device, input)
    }

    /// The exchanged rows are the chip's part of the rows with every chip's columns in chip
    /// order.
    #[tokio::test(flavor = "current_thread")]
    async fn exchanges_the_parts() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(exchange_small.topology()).unwrap();
        let rows = Small::SIZE;
        let (local, attended) = (Attended::SIZE / 4, Attended::SIZE);
        let columns: Vec<bf16> = (0..4)
            .flat_map(|chip| {
                (0..4 * rows).flat_map(move |row| {
                    (0..local).map(move |column| value(row, chip * local + column))
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
                assert_eq!(*got, value(row, column), "chip {chip} element {at}");
            }
        }
    }
}
