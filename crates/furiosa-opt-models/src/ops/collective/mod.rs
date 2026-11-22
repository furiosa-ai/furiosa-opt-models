//! Collectives over the chips of a placement: chip-local vectors summed with a canonical FP32
//! reduction tree, every chip's part gathered onto every chip, or each chip's columns exchanged
//! for another chip's rows.

use std::marker::PhantomData;

use crate::specialize::collective;
use crate::{
    Op,
    tensor::{
        live::Live,
        parallel::{Parallel, Part, Shard},
    },
};
use furiosa_opt_std::prelude::*;
pub use gather::{AllGather, InTiles, Pieces, Tile as GatherTile, TileRows, Tiled, Whole};

mod gather;
pub mod prefix;

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

impl<D: OutputScalar, W: AxisName, const N: usize> Op<HbmTensor<D, ChipMap<N>, m![W]>>
    for AllReduce<Parallel<N>>
{
    type Output = HbmTensor<D, ChipMap<N>, m![W]>;

    fn forward(&self, device: &mut Device, input: Self::Output) -> Self::Output {
        const {
            assert!(W::SIZE > 0 && W::SIZE % 1024 == 0 && W::SIZE <= 32768);
        }
        let input: Local<D, ChipMap<N>, W> = input.to_dm(&mut device.tdma);
        let input: Local<f32, ChipMap<N>, W> = device
            .main
            .begin(input.view())
            .fetch::<m![W / 1024], m![W % 4 # 8]>()
            .fetch_cast::<f32>()
            .collect::<m![W / 1024], m![W % 4 # 8]>()
            .commit_trim::<m![W % 4]>()
            .commit();
        let mut sum = input;
        #[unroll]
        for round in 0..const { collective::rounds(N) } {
            sum = self.reduce::<W>(device, sum, round);
        }
        D::narrow::<ChipMap<N>, W>(device, sum).to_hbm(&mut device.tdma)
    }
}

/// Every row summed over the chips as the vector sum is: FP32 pairwise rounds, then one narrowing.
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
            let tile: RowsLocal<D, ChipMap<N>, T<R, W>, W> = rows
                .tile::<Passes<R, W>, 1, m![
                    { Symbol<Tiles<R, W>> } = 1 # ({ Symbol<Tiles<R, W>> }),
                    { Symbol<T<R, W>> },
                    W
                ]>(pass)
                .to_dm(&mut device.tdma);
            let mut sum: RowsLocal<f32, ChipMap<N>, T<R, W>, W> = device
                .main
                .begin(tile.view())
                .fetch::<m![{ Symbol<T<R, W>> }, W / 1024], m![W % 4 # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Symbol<T<R, W>> }, W / 1024], m![W % 4 # 8]>()
                .commit_trim::<m![W % 4]>()
                .commit();
            #[unroll]
            for round in 0..const { collective::rounds(N) } {
                sum = self.reduce_rows::<T<R, W>, W>(device, sum, round);
            }
            // SAFETY: as for the input.
            let target = unsafe {
                output
                    .view_mut()
                    .reshape::<ChipMap<N>, m![{ Symbol<Tiles<R, W>> }, { Symbol<T<R, W>> }, W]>()
            };
            D::narrow_rows::<ChipMap<N>, T<R, W>, W>(device, sum)
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
    /// Each chip's partner in each round: round `r` pairs the chips whose indices differ in bit
    /// `r`. The first `rounds(N)` of the `N` rows are the rounds.
    const fn partners() -> [[usize; N]; N] {
        let mut partners = [[0; N]; N];
        let (mut round, mut mask) = (0, 1);
        while mask < N {
            let mut chip = 0;
            while chip < N {
                partners[round][chip] = chip ^ mask;
                chip += 1;
            }
            (round, mask) = (round + 1, mask * 2);
        }
        partners
    }

    /// Where each chip finds, in each round, the partial of the pair's chip whose round bit is
    /// `high`: its own (0) or its partner's (1).
    const fn selections(high: bool) -> [[usize; N]; N] {
        let mut selections = [[0; N]; N];
        let (mut round, mut mask) = (0, 1);
        while mask < N {
            let mut chip = 0;
            while chip < N {
                selections[round][chip] = if (chip & mask != 0) == high { 0 } else { 1 };
                chip += 1;
            }
            (round, mask) = (round + 1, mask * 2);
        }
        selections
    }

    /// The pairwise sums of round `round`.
    fn reduce<W: AxisName>(
        &self,
        device: &mut Device,
        input: Local<f32, ChipMap<N>, W>,
        round: usize,
    ) -> Local<f32, ChipMap<N>, W> {
        let exchanged = input
            .chip_shuffle(const { Self::partners() }[round])
            .to_dm::<m![W / 1024, W % 4]>(&mut device.tdma);
        let mut pair: DmTensor<
            f32,
            ChipMap<N>,
            m![1 # 2],
            ReductionSlices<W>,
            m![Two, W / 1024, W % 4],
        > = DmTensor::new();
        input.view().to_dm_view(
            &mut device.tdma,
            pair.view_mut()
                .tile::<m![Two], 1, m![Two = 1 #{!} 2, W / 1024, W % 4]>(0),
        );
        exchanged.view().to_dm_view(
            &mut device.tdma,
            pair.view_mut()
                .tile::<m![Two], 1, m![Two = 1 #{!} 2, W / 1024, W % 4]>(1),
        );
        let low = pair
            .chip_slice::<m![Two], m![W / 1024, W % 4]>(const { Self::selections(false) }[round])
            .to_dm::<m![W / 1024, W % 4]>(&mut device.tdma);
        let high = pair
            .chip_slice::<m![Two], m![W / 1024, W % 4]>(const { Self::selections(true) }[round])
            .to_dm::<m![W / 1024, W % 4]>(&mut device.tdma);
        let high: VrfTensor<
            f32,
            ChipMap<N>,
            m![1 # 2],
            ReductionSlices<W>,
            m![W / 1024, W % 4 # 8],
        > = device
            .sub
            .begin(high.view())
            .fetch::<m![W / 1024], m![W % 4 # 8]>()
            .collect::<m![W / 1024], m![W % 4 # 8]>()
            .to_vrf();
        // Both partners use low + high, including when NaN operand order matters.
        device
            .main
            .begin(low.view())
            .fetch::<m![W / 1024], m![W % 4 # 8]>()
            .collect::<m![W / 1024], m![W % 4 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![W / 1024, W % 4 / 4 # 2], m![W % 4]>()
            .vector_fp_binary(FpBinaryOp::AddF, &high)
            .vector_widen_concat::<m![W / 1024], m![W % 4 # 8]>()
            .vector_final()
            .commit_trim::<m![W % 4]>()
            .commit()
    }

    /// The pairwise sums of round `round`, every row at once.
    fn reduce_rows<R: AxisName, W: AxisName>(
        &self,
        device: &mut Device,
        input: RowsLocal<f32, ChipMap<N>, R, W>,
        round: usize,
    ) -> RowsLocal<f32, ChipMap<N>, R, W> {
        let exchanged = input
            .chip_shuffle(const { Self::partners() }[round])
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
            .chip_slice::<m![Two], m![R, W / 1024, W % 4]>(const { Self::selections(false) }[round])
            .to_dm::<m![R, W / 1024, W % 4]>(&mut device.tdma);
        let high = pair
            .chip_slice::<m![Two], m![R, W / 1024, W % 4]>(const { Self::selections(true) }[round])
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
        let mut output = HbmTensor::new();
        self.write(device, input.view(), output.view_mut());
        output
    }
}

impl<const N: usize> Exchange<Parallel<N>> {
    /// Exchange one whole, statically described set of parts into caller-owned storage.
    /// Views can name compact root windows inside larger allocations. Their row stride must
    /// describe the actual parts; cropping a capacity-strided tensor does not compact its rows.
    pub fn write<D: MaterializableScalar, S: AxisName, W: AxisName>(
        &self,
        device: &mut Device,
        input: RowsView<'_, D, N, Pieces<S, N>, Part<W, N>>,
        output: HbmTensorViewMut<'_, D, ChipMap<N>, m![S, W]>,
    ) {
        type L<W, const N: usize> = Symbol<Part<W, N>>;
        // SAFETY: the rows are the parts in chip order, each its rows in order.
        let rows = unsafe { input.reshape::<ChipMap<N>, m![{ O<N> }, S, { L<W, N> }]>() };
        // SAFETY: chip `t` holds part `t` of the rows, its columns the parts in chip order.
        let target =
            unsafe { output.reshape::<m![{ O<N> }], m![S, { Symbol<Shard<N>> }, { L<W, N> }]>() };
        rows.to_hbm_view(&mut device.tdma, target);
    }
}

axes![Two = 2];

type ChipMap<const N: usize> = m![{ Symbol<Shard<N>> }];
type Rows<D, const N: usize, R, W> = HbmTensor<D, ChipMap<N>, m![R, W]>;
type RowsView<'a, D, const N: usize, R, W> = HbmTensorView<'a, D, ChipMap<N>, m![R, W]>;
type ReductionSlices<W> = m![W % 1024 / 128, W % 128 / 4];
type Local<D, Chips, W> = DmTensor<D, Chips, m![1 # 2], ReductionSlices<W>, m![W / 1024, W % 4]>;
/// A chip's rows in the sum's layout: each slice one part of every row.
type RowsLocal<D, Chips, R, W> =
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
    /// Convert the completed f32 reduction to its output element type.
    fn narrow<Chips: M, W: AxisName>(
        device: &mut Device,
        input: Local<f32, Chips, W>,
    ) -> Local<Self, Chips, W>;

    /// [`OutputScalar::narrow`] for every row of a sum of rows.
    fn narrow_rows<Chips: M, R: AxisName, W: AxisName>(
        device: &mut Device,
        input: RowsLocal<f32, Chips, R, W>,
    ) -> RowsLocal<Self, Chips, R, W>;
}

impl OutputScalar for f32 {
    fn narrow<Chips: M, W: AxisName>(
        _: &mut Device,
        input: Local<f32, Chips, W>,
    ) -> Local<Self, Chips, W> {
        input
    }

    fn narrow_rows<Chips: M, R: AxisName, W: AxisName>(
        _: &mut Device,
        input: RowsLocal<f32, Chips, R, W>,
    ) -> RowsLocal<Self, Chips, R, W> {
        input
    }
}

impl OutputScalar for bf16 {
    fn narrow<Chips: M, W: AxisName>(
        device: &mut Device,
        input: Local<f32, Chips, W>,
    ) -> Local<Self, Chips, W> {
        device
            .main
            .begin(input.view())
            .fetch::<m![W / 1024], m![W % 4 # 8]>()
            .collect::<m![W / 1024], m![W % 4 # 8]>()
            .cast::<bf16, m![W % 4 # 16]>()
            .commit_trim::<m![W % 4]>()
            .commit()
    }

    fn narrow_rows<Chips: M, R: AxisName, W: AxisName>(
        device: &mut Device,
        input: RowsLocal<f32, Chips, R, W>,
    ) -> RowsLocal<Self, Chips, R, W> {
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
    use crate::tensor::{axis, scalar::Value};
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::{HbmBuffer, launch};
    use std::iter;

    axes![Hidden = 5120, Attended = 7168, Width = 7168, Narrow = 512];

    #[derive(Clone, Debug)]
    pub(crate) struct Part4<const ROWS: usize>;

    impl<const ROWS: usize> AxisName for Part4<ROWS> {
        const NAME: Ident = Ident::new("Shard");
        const SIZE: usize = ROWS;
    }

    /// A chip's part of the rows.
    pub(crate) type Small = Part4<1024>;
    type Local = Part<Attended, 4>;

    pub(super) type Chips = m![{ Symbol<Shard<4>> }];
    pub(super) type Input<C, W> =
        HbmTensor<bf16, Chips, m![{ Symbol<Pieces<C, 4>> }, { Symbol<Part<W, 4>> }]>;
    pub(super) type Output<C, W> = HbmTensor<bf16, Chips, m![C, W]>;
    type SourceRoots<A, B, E, W> = (Input<A, W>, Input<B, W>, Input<E, W>);
    type DestinationRoots<A, B, E, W> = (Output<A, W>, Output<B, W>, Output<E, W>);

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

    #[device(chip = 4)]
    fn root_windows<C: AxisName, W: AxisName, A: AxisName, B: AxisName, E: AxisName>(
        device: &mut Device,
        input: &SourceRoots<A, B, E, W>,
        output: &mut DestinationRoots<A, B, E, W>,
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

    fn coefficient(index: usize) -> bf16 {
        bf16::from_bits(0x3f00 | ((index * 131 + index / 17) % 1024) as u16)
    }

    pub(super) async fn buffers<C: AxisName, W: AxisName>(
        device: &mut Device,
        work: usize,
    ) -> (Input<C, W>, Output<C, W>) {
        let local_width = W::SIZE / 4;
        let input_per_chip = 4 * C::SIZE * local_width;
        let output_per_chip = C::SIZE * W::SIZE;
        let poison = bf16::from_bits(0x7fc1);
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
        let mut output: Output<C, W> = HbmBuffer::alloc_shared(device, output_per_chip * 2)
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
        (input, output)
    }

    pub(super) async fn assert_output<C: AxisName, W: AxisName>(
        device: &mut Device,
        output: &Output<C, W>,
        work: usize,
    ) {
        let local_width = W::SIZE / 4;
        let input_per_chip = 4 * C::SIZE * local_width;
        let output_per_chip = C::SIZE * W::SIZE;
        let poison = bf16::from_bits(0x7fc1);
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

    async fn check_root_windows<C: AxisName, W: AxisName, A: AxisName, B: AxisName, E: AxisName>() {
        let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
        let geometry = std::marker::PhantomData::<C>;
        let local_width = W::SIZE / 4;
        for work in [A::SIZE, E::SIZE, B::SIZE, A::SIZE] {
            let (input, output) = buffers::<C, W>(&mut device, work).await;
            let rows: Value<Chips> = HostTensor::<i32, m![1]>::from_vec([work as i32])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let input = input.into_buffer();
            let output = output.into_buffer();
            // SAFETY: root_windows reaches the destination through exactly one geometry in
            // each launch. No full-capacity destination argument or concurrent writer is bound.
            unsafe { output.partition() };
            // Prepare alternative root windows on the host. Aliases share their backing allocation;
            // only the selected geometry writes during a launch. Retain every owner through completion.
            let inputs: SourceRoots<A, B, E, W> = (
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
            let mut outputs: DestinationRoots<A, B, E, W> = (
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

            // Reuse the same destination across completed writes, then check rows and tail.
            for _ in 0..2 {
                launch(
                    root_windows,
                    (&mut device, &inputs, &mut outputs, &rows, &geometry),
                )
                .await
                .unwrap();
            }
            let output: Output<C, W> = output.try_into().unwrap();
            assert_output(&mut device, &output, work).await;
        }
    }

    /// Root windows must describe compact parts rather than retain capacity-strided coordinates.
    #[tokio::test(flavor = "current_thread")]
    async fn compact_whole_exchange_preserves_prefix_and_capacity_tail() {
        let _lease = crate::tests::DEVICE.lock().await;
        check_root_windows::<
            axis::Rows<1280>,
            Narrow,
            axis::Rows<256>,
            axis::Rows<768>,
            axis::Rows<1280>,
        >()
        .await;
        check_root_windows::<
            axis::Rows<10496>,
            Width,
            axis::Rows<9472>,
            axis::Rows<9728>,
            axis::Rows<9984>,
        >()
        .await;
    }
}
