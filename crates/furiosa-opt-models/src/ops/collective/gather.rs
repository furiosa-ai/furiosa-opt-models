//! Every chip's part on every chip, the parts in chip order: chip-local vectors joined into one
//! vector, or the rows of each chip stacked into the rows of all chips.

use super::{ChipMap, Rows, Shard};
use crate::{
    Op,
    tensor::{
        live::{BlockPrefix, Live, Partitioned},
        parallel::{Parallel, Part},
    },
};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Every chip's part on every chip of placement `P`, the parts in chip order.
#[derive(DeviceSend)]
pub struct AllGather<P, Strategy = Whole>(PhantomData<(P, Strategy)>);

impl<P, Strategy> Default for AllGather<P, Strategy> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// The rows of `N` chips' parts of `S` rows each, the parts in chip order.
#[derive(Clone, Debug)]
pub struct Pieces<S, const N: usize>(PhantomData<S>);

impl<S: AxisName, const N: usize> AxisName for Pieces<S, N> {
    const NAME: Ident = Ident::new("Pieces");
    const SIZE: usize = N * S::SIZE;
}

/// All gathered rows arrive in one allocation.
pub struct Whole;

/// The first HEAD rows of every part, then K-1 tiles of T rows per part.
pub struct InTiles<const T: usize, const K: usize, const HEAD: usize>;

/// Rows of one part in a tile.
#[derive(Clone, Debug)]
pub struct TileRows<const T: usize>;

impl<const T: usize> AxisName for TileRows<T> {
    const NAME: Ident = Ident::new("TileRows");
    const SIZE: usize = T;
}

/// One independently allocated gather tile, with every chip's rows in chip order.
pub type Tile<D, const N: usize, W, const T: usize> = Rows<D, N, Pieces<TileRows<T>, N>, W>;

/// Allocates each tensor where the device creates its array.
pub(crate) struct Fresh<X>(pub(crate) X);

impl<D: Scalar, C: M, E: M> Default for Fresh<HbmTensor<D, C, E>> {
    fn default() -> Self {
        Self(HbmTensor::new())
    }
}

/// Independently arriving tiles of R rows from P parts. The first tile uses only
/// HEAD rows at the start of each part's T-row slot.
pub struct Tiled<X, R, const P: usize, const T: usize, const K: usize, const HEAD: usize>(
    pub(crate) [Fresh<X>; K],
    PhantomData<R>,
);

impl<D: Scalar, W: AxisName> Op<HbmTensor<D, m![1], m![W]>> for AllGather<Parallel<1>> {
    type Output = HbmTensor<D, m![1], m![W]>;

    fn forward(&self, _: &mut Device, input: Self::Output) -> Self::Output {
        input
    }
}

/// One chip's rows of a launch already hold every column.
impl<X, R> Op<Live<X, R>> for AllGather<Parallel<1>> {
    type Output = Live<X, R>;

    fn forward(&self, _: &mut Device, input: Live<X, R>) -> Self::Output {
        input
    }
}

/// Each chip's part of the vector `V`, joined into `V` on every chip.
impl<D: MaterializableScalar, V: AxisName, const N: usize>
    Op<HbmTensor<D, ChipMap<N>, m![{ Symbol<Part<V, N>> }]>> for AllGather<Parallel<N>>
{
    type Output = HbmTensor<D, ChipMap<N>, m![V]>;

    fn forward(
        &self,
        device: &mut Device,
        input: HbmTensor<D, ChipMap<N>, m![{ Symbol<Part<V, N>> }]>,
    ) -> Self::Output {
        const {
            assert!(V::SIZE == Part::<V, N>::SIZE * N);
            assert!(Part::<V, N>::SIZE > 0 && Part::<V, N>::SIZE % 1024 == 0);
        }
        let input: DmTensor<
            D,
            ChipMap<N>,
            m![1 # 2],
            m![{ Symbol<Part<V, N>> } % 1024 / 4],
            m![{ Symbol<Part<V, N>> } / 1024, { Symbol<Part<V, N>> } % 4],
        > = input.to_dm(&mut device.tdma);
        let output = input
            .to_dm::<Broadcast<N>, m![1 # 2], m![{ Symbol<Part<V, N>> } % 1024 / 4], m![
                { Symbol<Shard<N>> }, { Symbol<Part<V, N>> } / 1024, { Symbol<Part<V, N>> } % 4
            ]>(&mut device.tdma)
            .to_hbm::<m![{ Symbol<Shard<N>> }, { Symbol<Part<V, N>> }]>(&mut device.tdma);
        // SAFETY: contiguous [Shard, W] has index shard * W::SIZE + local, exactly V order.
        // Both chip mappings have N slots; consuming the owner preserves exclusive ownership.
        unsafe { output.reshape::<ChipMap<N>, m![V]>() }
    }
}

/// Each chip's `S` rows, stacked into the rows of all chips on every chip.
impl<'x, D: MaterializableScalar, S: AxisName, W: AxisName, const N: usize> Op<&'x Rows<D, N, S, W>>
    for AllGather<Parallel<N>>
{
    type Output = Rows<D, N, Pieces<S, N>, W>;

    fn forward(&self, device: &mut Device, input: &'x Rows<D, N, S, W>) -> Self::Output {
        let mut output: HbmTensor<D, Broadcast<N>, m![{ Symbol<Shard<N>> }, S, W]> =
            HbmTensor::new();
        input
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
        // SAFETY: every chip holds the parts in chip order, each its rows in order.
        unsafe { output.reshape() }
    }
}

impl<'x, D, S, W, const N: usize, const T: usize, const K: usize, const HEAD: usize>
    Op<&'x Rows<D, N, S, W>> for AllGather<Parallel<N>, InTiles<T, K, HEAD>>
where
    D: MaterializableScalar,
    S: AxisName,
    W: AxisName,
    [Fresh<Tile<D, N, W, T>>; K]: Default,
{
    type Output = Tiled<Tile<D, N, W, T>, Pieces<S, N>, N, T, K, HEAD>;

    fn forward(&self, device: &mut Device, input: &'x Rows<D, N, S, W>) -> Self::Output {
        self.tiles(device, input, Live::all())
    }
}

/// Gather only tiles intersecting each chip's live interval, retaining the capacity stride
/// between the gathered parts. A partial tile may also transfer padding; only the returned
/// intervals are live. Empty intervals transfer no rows.
impl<'x, D, S, W, const N: usize, const T: usize, const K: usize, const HEAD: usize>
    Op<&'x Live<&Rows<D, N, S, W>, S>> for AllGather<Parallel<N>, InTiles<T, K, HEAD>>
where
    D: MaterializableScalar,
    S: AxisName,
    W: AxisName,
    [Fresh<Tile<D, N, W, T>>; K]: Default,
{
    type Output = Partitioned<Tiled<Tile<D, N, W, T>, Pieces<S, N>, N, T, K, HEAD>, S, N>;

    fn forward(&self, device: &mut Device, input: &'x Live<&Rows<D, N, S, W>, S>) -> Self::Output {
        let interval = input.with(());
        Partitioned::new(self.tiles(device, input.value(), interval), interval)
    }
}

/// Preserve the checked block-prefix contract while gathering its independently arriving tiles.
impl<'x, D, S, W, B, const N: usize, const T: usize, const K: usize, const HEAD: usize>
    Op<&'x BlockPrefix<&Rows<D, N, S, W>, S, B>> for AllGather<Parallel<N>, InTiles<T, K, HEAD>>
where
    D: MaterializableScalar,
    S: AxisName,
    W: AxisName,
    B: AxisName,
    [Fresh<Tile<D, N, W, T>>; K]: Default,
{
    type Output = BlockPrefix<Tiled<Tile<D, N, W, T>, Pieces<S, N>, N, T, K, HEAD>, S, B>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x BlockPrefix<&Rows<D, N, S, W>, S, B>,
    ) -> Self::Output {
        input.with(self.tiles(device, input.value(), input.interval()))
    }
}

impl<const N: usize, const T: usize, const K: usize, const HEAD: usize>
    AllGather<Parallel<N>, InTiles<T, K, HEAD>>
{
    fn tiles<D: MaterializableScalar, S: AxisName, W: AxisName>(
        &self,
        device: &mut Device,
        input: &Rows<D, N, S, W>,
        interval: Live<(), S>,
    ) -> Tiled<Tile<D, N, W, T>, Pieces<S, N>, N, T, K, HEAD>
    where
        [Fresh<Tile<D, N, W, T>>; K]: Default,
    {
        const { assert!(K > 0 && HEAD > 0 && HEAD <= T && HEAD + (K - 1) * T == S::SIZE) };
        type L<const T: usize> = Symbol<TileRows<T>>;
        type Slots<const N: usize, W, const T: usize> = m![{ Symbol<Shard<N>> }, { L<T> }, W];
        let mut tiles: [Fresh<Tile<D, N, W, T>>; K] = Default::default();
        if interval.rows() > 0 && interval.start() < HEAD {
            // SAFETY: the part's first rows in order into the start of each slot; only the row axis
            // is renamed.
            let source = unsafe {
                input
                    .view()
                    .tile::<m![S], HEAD, m![S = ({ L<HEAD> }) # S, W]>(0)
                    .reshape::<ChipMap<N>, m![{ L<HEAD> } # S, W]>()
            };
            // SAFETY: every chip holds the first rows of the parts in chip order, each in order.
            let target = unsafe {
                tiles[0]
                .0
                .view_mut()
                .reshape::<Broadcast<N>, Slots<N, W, T>>()
                .tile::<m![{ L<T> }], HEAD, m![
                    { Symbol<Shard<N>> },
                    { L<T> } = ({ L<HEAD> }) #{!} ({ L<T> }),
                    W
                ]>(0)
                .reshape::<Broadcast<N>, m![{ Symbol<Shard<N>> }, { L<HEAD> } #{!} ({ L<T> }), W]>()
            };
            source.to_hbm_view(&mut device.tdma, target);
        }
        #[unroll]
        for t in 0..K - 1 {
            let begin = HEAD + T * t;
            if interval.rows() > 0
                && interval.start() < begin + T
                && interval.start() + interval.rows() > begin
            {
                // SAFETY: as for the first tile, the part's rows from `HEAD + T * t` in order.
                let source = unsafe {
                    input
                        .view()
                        .tile::<m![S], T, m![S = ({ L<T> }) # S, W]>(HEAD + T * t)
                        .reshape::<ChipMap<N>, m![{ L<T> } # S, W]>()
                };
                // SAFETY: every chip holds the tile's rows of the parts in chip order, each in order.
                let target = unsafe {
                    tiles[t + 1]
                        .0
                        .view_mut()
                        .reshape::<Broadcast<N>, Slots<N, W, T>>()
                };
                source.to_hbm_view(&mut device.tdma, target);
            }
        }
        Tiled(tiles, PhantomData)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::collective::tests::{Small, value};
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![Vocabulary = 131072, Width = 5376];

    #[device(chip = 2)]
    fn gather(
        device: &mut Device,
        input: HbmTensor<bf16, ChipMap<2>, m![{ Symbol<Part<Vocabulary, 2>> }]>,
    ) -> HbmTensor<bf16, ChipMap<2>, m![Vocabulary]> {
        AllGather::<Parallel<2>>::default().forward(device, input)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn preserves_vocabulary_order() {
        type Local = Part<Vocabulary, 2>;
        let mut device = Device::new(gather.topology()).unwrap();
        let values: Vec<_> = (0..Vocabulary::SIZE)
            .map(|i| {
                bf16::from_f32(
                    f32::from(u16::try_from(i % 251).unwrap())
                        + 512.0 * f32::from(u16::try_from(i / Local::SIZE).unwrap()),
                )
            })
            .collect();
        let input: HbmTensor<bf16, ChipMap<2>, m![Local]> =
            HostTensor::<bf16, m![{ Symbol<Shard<2>> }, Local]>::from_vec(values.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let output = launch(gather, (&mut device, input)).await.unwrap();
        let output = output
            .to_host::<m![{ Symbol<Shard<2>> }, Vocabulary]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for chip in output.chunks_exact(Vocabulary::SIZE) {
            assert_eq!(chip, values);
        }
    }

    #[device(chip = 4)]
    fn gather_small(
        device: &mut Device,
        input: &Rows<bf16, 4, Small, Width>,
    ) -> Rows<bf16, 4, Pieces<Small, 4>, Width> {
        AllGather::<Parallel<4>>::default().forward(device, input)
    }

    /// The gathered rows are every chip's part in chip order, on every chip.
    #[tokio::test(flavor = "current_thread")]
    async fn gathers_the_parts() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(gather_small.topology()).unwrap();
        let (rows, width) = (Small::SIZE, Width::SIZE);
        let parts: Vec<bf16> = (0..4 * rows)
            .flat_map(|row| (0..width).map(move |column| value(row, column)))
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
                    value(at / width, at % width),
                    "chip {chip} element {at}"
                );
            }
        }
    }
}
