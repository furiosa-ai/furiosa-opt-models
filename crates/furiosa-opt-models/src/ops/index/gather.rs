//! Rows gathered by byte offset or by position.

use crate::Op;
use crate::load::Load;
use crate::ops::index::table::{self, Access, Offsets};
use crate::tensor::cache::Blocks;
use crate::tensor::cache::{Block, KeyValue};
use crate::tensor::{
    axis::Tile, live::Live, mapping::Rows as Layout, parallel::Chip, scalar::Value,
};
use furiosa_opt_std::prelude::*;
use std::{io, marker::PhantomData};

/// Runtime list of individually retained contiguous rows.
pub type RowStorage<Width, D = f32, Chips = Chip> = Slice<HbmTensor<D, Chips, m![Width]>>;

/// Rows selected by index: rows of a row list, a dense tensor or key and value storage at aligned
/// byte offsets, one lane group of at most eight rows a transfer, and the rows of a paged history.
#[derive(Default, DeviceSend, Load)]
pub struct Gather;

type Selected<Chips, W, R, D> =
    DmTensor<D, Chips, m![1 # 2], m![{ Layout<R>::Lanes } # 256], m![W]>;
type Indices<Chips, R> = HbmTensor<i32, Chips, m![{ Layout<R>::Lanes }]>;
type Groups<R> = Layout<R>::Groups;
type Write<W, R> = Layout<R>::GroupMut<m![W]>;

/// Rows a gather reads whole at byte offsets: a row list, each row a tensor of its own, or a dense
/// tensor, its rows any mapping before the features.
pub trait Source<Chips: M> {
    type Scalar: MaterializableScalar;
    /// The features of a row.
    type Width: AxisName;

    /// The rows at the byte `offsets`, in data memory.
    fn rows<C: M, S: M, E: M, I: M>(
        &self,
        device: &mut Device,
        offsets: &HbmTensor<i32, Chips, I>,
    ) -> DmTensor<Self::Scalar, Chips, C, S, E>;
}

impl<W: AxisName, D: MaterializableScalar, Chips: M> Source<Chips> for RowStorage<W, D, Chips> {
    type Scalar = D;
    type Width = W;

    fn rows<C: M, S: M, E: M, I: M>(
        &self,
        device: &mut Device,
        offsets: &HbmTensor<i32, Chips, I>,
    ) -> DmTensor<D, Chips, C, S, E> {
        self.gather::<m![1], m![W]>()
            .by_byte_offsets(offsets)
            .to_dm(&mut device.tdma)
    }
}

impl<R: M, W: AxisName, D: MaterializableScalar, Chips: M> Source<Chips>
    for HbmTensor<D, Chips, m![{ R }, W]>
{
    type Scalar = D;
    type Width = W;

    fn rows<C: M, S: M, E: M, I: M>(
        &self,
        device: &mut Device,
        offsets: &HbmTensor<i32, Chips, I>,
    ) -> DmTensor<D, Chips, C, S, E> {
        self.gather::<R, m![W]>()
            .by_byte_offsets(offsets)
            .to_dm(&mut device.tdma)
    }
}

impl<S: Source<Chips>, C: AxisName, Chips: M + 'static> Op<(&S, &HbmTensor<i32, Chips, m![C]>)>
    for Gather
{
    type Output = HbmTensor<S::Scalar, Chips, m![C, { Symbol<S::Width> }]>;

    fn forward(
        &self,
        device: &mut Device,
        (input, offsets): (&S, &HbmTensor<i32, Chips, m![C]>),
    ) -> Self::Output {
        if const { <Groups<C> as M>::SIZE == 1 } {
            let selected: Selected<Chips, S::Width, C, S::Scalar> = input.rows(device, offsets);
            selected.to_hbm(&mut device.tdma)
        } else {
            let mut output = HbmTensor::new();
            for group in 0..<Groups<C> as M>::SIZE {
                let mut index: Indices<Chips, C> = HbmTensor::new();
                offsets
                    .view()
                    .tile::<Groups<C>, 1, m![{ Layout<C>::Selected }, { Layout<C>::Lanes }]>(group)
                    .to_hbm_view(&mut device.tdma, index.view_mut());
                let selected: Selected<Chips, S::Width, C, S::Scalar> = input.rows(device, &index);
                selected.view().to_hbm_view(
                    &mut device.tdma,
                    output
                        .view_mut()
                        .tile::<Groups<C>, 1, Write<S::Width, C>>(group),
                );
            }
            output
        }
    }
}

impl<S: Source<Chips>, C: AxisName, Chips: M + 'static> Op<&(S, HbmTensor<i32, Chips, m![C]>)>
    for Gather
{
    type Output = HbmTensor<S::Scalar, Chips, m![C, { Symbol<S::Width> }]>;

    fn forward(
        &self,
        device: &mut Device,
        (input, offsets): &(S, HbmTensor<i32, Chips, m![C]>),
    ) -> Self::Output {
        self.forward(device, (input, offsets))
    }
}

/// The rows of a slice, the live rows `0..len` of an `R`-row envelope.
#[derive(DeviceSend)]
pub struct Rows<'a, D: Scalar, W: AxisName, R: AxisName, Chips: M + 'static = Chip> {
    source: &'a RowStorage<W, D, Chips>,
    /// The source's row count, which the device reads at the launch's start.
    count: Value<Chips>,
    shape: PhantomData<R>,
}

impl<'a, D: Scalar, W: AxisName, R: AxisName, Chips: M + 'static> Rows<'a, D, W, R, Chips> {
    /// Borrow `source` and upload its row count; refuse more rows than the envelope holds.
    pub async fn new(device: &mut Device, source: &'a RowStorage<W, D, Chips>) -> io::Result<Self> {
        if source.len() > R::SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "rows exceed the compiled envelope",
            ));
        }
        let rows = i32::try_from(source.len()).expect("an envelope's rows fit I32");
        let count = HostTensor::<i32, m![1]>::from_vec([rows])
            .to_hbm(&mut device.pdma)
            .await
            .map_err(io::Error::other)?;
        Ok(Self {
            source,
            count,
            shape: PhantomData,
        })
    }

    /// The rows the source holds, read by the device from the uploaded count.
    pub(crate) fn live(&self) -> Live<(), R> {
        Live::<(), R>::read(&self.count)
    }

    pub(crate) fn source(&self) -> &'a RowStorage<W, D, Chips> {
        self.source
    }
}

impl<D: MaterializableScalar, W: AxisName, R: AxisName, Chips: M + 'static>
    Rows<'_, D, W, R, Chips>
{
    /// The live rows staged as the first rows of an envelope tensor, whose later rows are
    /// unspecified.
    pub(crate) fn stage(
        &self,
        device: &mut Device,
    ) -> (HbmTensor<D, Chips, m![R, W]>, Live<(), R>) {
        let live = self.live();
        let mut output: HbmTensor<D, Chips, m![R, W]> = HbmTensor::new();
        for row in 0..live.rows() {
            self.source[row].view().to_hbm_view(
                &mut device.tdma,
                output.view_mut().tile::<m![R], 1, m![R = 1 #{!} R, W]>(row),
            );
        }
        (output, live)
    }
}

/// Copy a slice's rows into the first rows of a compiled envelope; later rows are unspecified.
#[derive(DeviceSend)]
pub struct Copy<Width, Rows, D = bf16>(PhantomData<(Width, Rows, D)>);

impl<Width, Rows, D> Default for Copy<Width, Rows, D> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Width: AxisName, Rows: AxisName, D: MaterializableScalar, Chips: M + 'static>
    Op<&RowStorage<Width, D, Chips>> for Copy<Width, Rows, D>
{
    type Output = HbmTensor<D, Chips, m![Rows, Width]>;

    fn forward(&self, device: &mut Device, source: &RowStorage<Width, D, Chips>) -> Self::Output {
        let mut output: HbmTensor<D, Chips, m![Rows, Width]> = HbmTensor::new();
        #[unroll]
        for row in 0..Rows::SIZE {
            if row < source.len() {
                source[row].view().to_hbm_view(
                    &mut device.tdma,
                    output
                        .view_mut()
                        .tile::<m![Rows], 1, m![Rows = 1 #{!} Rows, Width]>(row),
                );
            }
        }
        output
    }
}

impl<Width: AxisName, Rows: AxisName, D: MaterializableScalar, Chips: M + 'static>
    Op<&mut &RowStorage<Width, D, Chips>> for Copy<Width, Rows, D>
{
    type Output = HbmTensor<D, Chips, m![Rows, Width]>;

    fn forward(
        &self,
        device: &mut Device,
        source: &mut &RowStorage<Width, D, Chips>,
    ) -> Self::Output {
        self.forward(device, *source)
    }
}

/// Tile `at` of a slice's `live` rows, `N` rows a tile, into the rows of one tile; rows past the
/// live ones are unspecified.
impl<Width: AxisName, R: AxisName, D: MaterializableScalar, const N: usize, Chips: M + 'static>
    Op<((&RowStorage<Width, D, Chips>, Live<(), R>), usize)> for Copy<Width, Tile<R, N>, D>
{
    type Output = HbmTensor<D, Chips, m![{ Symbol<Tile<R, N>> }, Width]>;

    fn forward(
        &self,
        device: &mut Device,
        ((source, live), at): ((&RowStorage<Width, D, Chips>, Live<(), R>), usize),
    ) -> Self::Output {
        type T<R, const N: usize> = Symbol<Tile<R, N>>;
        let mut rows: HbmTensor<D, Chips, m![{ T<R, N> }, Width]> = HbmTensor::new();
        #[unroll]
        for lane in 0..<T<R, N> as M>::SIZE {
            let row = at * <T<R, N> as M>::SIZE + lane;
            if row < live.rows() {
                source[row].view().to_hbm_view(
                    &mut device.tdma,
                    rows.view_mut().tile::<T<R, N>, 1, m![
                        { T<R, N> } = 1 #{!} ({ T<R, N> }), Width
                    ]>(lane),
                );
            }
        }
        rows
    }
}

/// Key and value rows from caller-owned storage at independent byte offsets.
impl<Width: AxisName, Count: AxisName, D: MaterializableScalar, Chips: M + 'static>
    Op<(
        &table::KeyValue<RowStorage<Width, D, Chips>>,
        &Offsets<Count, Chips>,
    )> for Gather
{
    type Output = table::KeyValue<HbmTensor<D, Chips, m![Count, Width]>>;

    fn forward(
        &self,
        device: &mut Device,
        (cache, offsets): (
            &table::KeyValue<RowStorage<Width, D, Chips>>,
            &Offsets<Count, Chips>,
        ),
    ) -> Self::Output {
        (
            self.forward(&mut *device, (&cache.0, &offsets.0)),
            self.forward(device, (&cache.1, &offsets.1)),
        )
    }
}

impl<Width: AxisName, Count: AxisName, D: MaterializableScalar, Chips: M + 'static>
    Op<&Access<Width, Count, D, Chips>> for Gather
{
    type Output = table::KeyValue<HbmTensor<D, Chips, m![Count, Width]>>;

    fn forward(&self, device: &mut Device, cache: &Access<Width, Count, D, Chips>) -> Self::Output {
        self.forward(device, (&cache.0, &cache.1.0))
    }
}

impl<W: AxisName, Q: AxisName, D: MaterializableScalar, Chips: M + 'static>
    Op<(&RowStorage<W, D, Chips>, &Offsets<Q, Chips>)> for Gather
{
    type Output = table::KeyValue<HbmTensor<D, Chips, m![Q, W]>>;

    fn forward(
        &self,
        device: &mut Device,
        (storage, offsets): (&RowStorage<W, D, Chips>, &Offsets<Q, Chips>),
    ) -> Self::Output {
        (
            self.forward(&mut *device, (storage, &offsets.0)),
            self.forward(device, (storage, &offsets.1)),
        )
    }
}

pub(crate) struct Read<
    'a,
    Heads: AxisName,
    Head: AxisName,
    C: M,
    S: M,
    E: M,
    I: M,
    Chips: M + 'static = Chip,
> {
    pub pool: &'a Blocks<Heads, Head, bf16, Chips>,
    pub indices: &'a HbmTensor<i32, Chips, I>,
    pub output: PhantomData<(C, S, E)>,
}

/// `read` limited to each batch's first `lengths` sparse positions past `offset`; a batch may read
/// up to the longest prefix, so its cells past its own are unspecified.
pub(crate) struct Prefix<Read, Sparse: M, Lengths> {
    pub read: Read,
    pub lengths: Lengths,
    pub offset: i32,
    pub sparse: PhantomData<Sparse>,
}

impl<Heads: AxisName, Head: AxisName, C: M, S: M, E: M, I: M, Chips: M + 'static>
    Op<Read<'_, Heads, Head, C, S, E, I, Chips>> for Gather
{
    type Output = DmTensor<bf16, Chips, C, S, E>;

    fn forward(
        &self,
        context: &mut Device,
        Read { pool, indices, .. }: Read<'_, Heads, Head, C, S, E, I, Chips>,
    ) -> Self::Output {
        // SAFETY: the caller supplies offsets in range of this layer's blocks.
        unsafe {
            pool.tokens
                .gather::<m![1], m![Block, KeyValue, Heads, Head]>()
                .by_byte_offsets(indices)
                .unchecked()
                .to_dm(&mut context.tdma)
        }
    }
}

impl<Heads: AxisName, Head: AxisName, C: M, S: M, E: M, I: M, Sparse: M, L: M, Chips: M + 'static>
    Op<Prefix<Read<'_, Heads, Head, C, S, E, I, Chips>, Sparse, &HbmTensor<i32, Chips, L>>>
    for Gather
{
    type Output = DmTensor<bf16, Chips, C, S, E>;

    fn forward(
        &self,
        context: &mut Device,
        Prefix {
            read: Read { pool, indices, .. },
            lengths,
            offset,
            ..
        }: Prefix<
            Read<'_, Heads, Head, C, S, E, I, Chips>,
            Sparse,
            &HbmTensor<i32, Chips, L>,
        >,
    ) -> Self::Output {
        // SAFETY: the caller bounds every index up to the longest prefix by this layer's blocks.
        unsafe {
            pool.tokens
                .gather::<m![1], m![Block, KeyValue, Heads, Head]>()
                .by_byte_offsets(indices)
                .prefix::<Sparse, _>(lengths, offset)
                .overfetch()
                .unchecked()
                .to_dm(&mut context.tdma)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::attention::geometry::Hidden;
    use crate::tensor::axis::{Eight, Single, Sixteen};
    use crate::tensor::memory::alias;
    use furiosa_opt_std::runtime::launch;

    #[device(chip = 1)]
    fn selected<Width: AxisName, D: MaterializableScalar, Rows: AxisName>(
        device: &mut Device,
        select: &Gather,
        input: &(RowStorage<Width, D>, HbmTensor<i32, m![1], m![Rows]>),
    ) -> HbmTensor<D, m![1], m![Rows, Width]> {
        select.forward(device, input)
    }

    async fn select<D: MaterializableScalar, Width: AxisName, Rows: AxisName>(
        values: Vec<D>,
        rows: Vec<usize>,
    ) {
        assert_eq!(rows.len(), Rows::SIZE);
        let mut device = Device::new(selected.topology()).unwrap();
        let mut bytes = vec![0; D::size_in_bytes_from_length(values.len())];
        for (index, &value) in values.iter().enumerate() {
            D::store(&mut bytes, index, value);
        }
        let source = HostBufferSlice::from(&bytes)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let source: RowStorage<Width, D> = (0..values.len() / Width::SIZE)
            .map(|row| {
                alias(
                    &source,
                    row * Width::SIZE * D::BITS / 8..(row + 1) * Width::SIZE * D::BITS / 8,
                )
                .unwrap()
            })
            .collect();
        let bytes = rows
            .iter()
            .map(|&row| i32::try_from(row * Width::SIZE * D::BITS / 8).unwrap())
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let indices = HostBufferSlice::from(&bytes)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let offsets: HbmTensor<i32, m![1], m![Rows]> = indices.try_into().unwrap();
        let buffer = HbmBuffer::alloc(&device, Rows::SIZE * Width::SIZE * D::BITS / 8).unwrap();
        let mut output: HbmTensor<D, m![1], m![Rows, Width]> = alias(&buffer, ..).unwrap();
        let model = Gather;
        let input = (source, offsets);
        launch(selected, (&mut device, &model, &input))
            .output(&mut output)
            .await
            .unwrap();
        let actual = buffer.to_host(&mut device.pdma).await.unwrap();
        for (index, row) in rows.into_iter().enumerate() {
            for column in 0..Width::SIZE {
                assert_eq!(
                    D::load(&actual, index * Width::SIZE + column),
                    values[row * Width::SIZE + column],
                    "row {index}, column {column}",
                );
            }
        }
    }

    #[device(chip = 1)]
    fn gathered<Width: AxisName, D: MaterializableScalar, Source: AxisName, Rows: AxisName>(
        device: &mut Device,
        select: &Gather,
        input: &(
            HbmTensor<D, m![1], m![Source, Width]>,
            HbmTensor<i32, m![1], m![Rows]>,
        ),
    ) -> HbmTensor<D, m![1], m![Rows, Width]> {
        select.forward(device, input)
    }

    axes![Source = 5, Many = 512];

    async fn gather<D: MaterializableScalar, Width: AxisName, Rows: AxisName>(
        values: Vec<D>,
        rows: Vec<usize>,
    ) {
        assert_eq!(rows.len(), Rows::SIZE);
        let mut device = Device::new(gathered.topology()).unwrap();
        let source = HostTensor::<D, m![Source, Width]>::from_vec(values.clone())
            .to_hbm::<m![1], m![Source, Width]>(&mut device.pdma)
            .await
            .unwrap();
        let offsets = HostTensor::<i32, m![Rows]>::from_vec(
            rows.iter()
                .map(|&row| i32::try_from(row * Width::SIZE * D::BITS / 8).unwrap()),
        )
        .to_hbm::<m![1], m![Rows]>(&mut device.pdma)
        .await
        .unwrap();
        let model = Gather;
        let actual = launch(gathered, (&mut device, &model, &(source, offsets)))
            .await
            .unwrap()
            .to_host::<m![Rows, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (index, row) in rows.into_iter().enumerate() {
            for column in 0..Width::SIZE {
                assert_eq!(
                    actual[index * Width::SIZE + column],
                    values[row * Width::SIZE + column],
                    "row {index}, column {column}",
                );
            }
        }
    }

    /// Byte offsets select complete rows of a dense tensor in one, two or sixty-four gather groups.
    #[tokio::test]
    async fn selects_requested_rows_of_a_tensor() {
        let values = |width: usize| {
            (0..Source::SIZE * width).map(move |index| {
                (((index / width) * 37 + (index % width) * 13) % 251) as f32 - 125.0
            })
        };
        gather::<f32, Hidden<512>, Single>(values(512).collect(), vec![3]).await;
        gather::<f32, Hidden<1024>, Sixteen>(
            values(1024).collect(),
            (0..16).map(|row| (row * 3) % 5).collect(),
        )
        .await;
        gather::<f32, Hidden<512>, Many>(
            values(512).collect(),
            (0..Many::SIZE).map(|row| (row * 3) % 5).collect(),
        )
        .await;
    }

    /// Byte offsets select complete rows for both dtypes and single or multiple gather groups.
    #[tokio::test]
    async fn selects_requested_rows() {
        let values = |width| {
            (0..5 * width).map(move |index| {
                (((index / width) * 37 + (index % width) * 13) % 251) as f32 - 125.0
            })
        };
        select::<f32, Hidden<512>, Single>(values(512).collect(), vec![2]).await;
        select::<bf16, Hidden<768>, Eight>(
            values(768).map(bf16::from_f32).collect(),
            vec![2, 0, 2, 4, 1, 3, 0, 4],
        )
        .await;
        select::<f32, Hidden<1024>, Sixteen>(
            values(1024).collect(),
            (0..16).map(|row| (row * 3) % 5).collect(),
        )
        .await;
    }
}
