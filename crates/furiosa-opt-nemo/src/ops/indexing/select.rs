use super::RowStorage;
use crate::{Module, ops::layout::Rows as Layout};
use furiosa_opt_std::prelude::*;

/// Select complete rows from a row list or dense tensor using aligned byte offsets, not row IDs.
/// Each gather transfers one lane group of at most eight rows.
#[derive(DeviceSend)]
pub struct Select<Width, D = f32>(std::marker::PhantomData<(Width, D)>);

impl<Width, D> Default for Select<Width, D> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

type Selected<W, R, D> = DmTensor<D, m![1], m![1 # 2], m![{ Layout<R>::Lanes } # 256], m![W]>;
type Indices<R> = HbmTensor<i32, m![1], m![{ Layout<R>::Lanes }]>;
type Groups<R> = Layout<R>::Groups;
type Read<R> = m![{ Layout<R>::Selected }, { Layout<R>::Lanes }];
type Write<W, R> = Layout<R>::GroupMut<m![W]>;

impl<W: AxisName, R: AxisName, D: MaterializableScalar>
    Module<(
        &mut Device,
        (&RowStorage<W, D>, &HbmTensor<i32, m![1], m![R]>),
    )> for Select<W, D>
{
    type Output = HbmTensor<D, m![1], m![R, W]>;

    fn forward(
        &self,
        (device, (input, offsets)): (
            &mut Device,
            (&RowStorage<W, D>, &HbmTensor<i32, m![1], m![R]>),
        ),
    ) -> Self::Output {
        if const { <Groups<R> as M>::SIZE == 1 } {
            let selected: Selected<W, R, D> = input
                .gather::<m![1], m![W]>()
                .by_byte_offsets(offsets)
                .to_dm(&mut device.tdma);
            selected.to_hbm(&mut device.tdma)
        } else {
            let mut output = HbmTensor::new();
            for group in 0..<Groups<R> as M>::SIZE {
                let mut index: Indices<R> = HbmTensor::new();
                offsets
                    .view()
                    .tile::<Groups<R>, 1, Read<R>>(group)
                    .to_hbm_view(&mut device.tdma, index.view_mut());
                let selected: Selected<W, R, D> = input
                    .gather::<m![1], m![W]>()
                    .by_byte_offsets(&index)
                    .to_dm(&mut device.tdma);
                selected.view().to_hbm_view(
                    &mut device.tdma,
                    output.view_mut().tile::<Groups<R>, 1, Write<W, R>>(group),
                );
            }
            output
        }
    }
}

impl<W: AxisName, R: AxisName, D: MaterializableScalar>
    Module<(
        &mut Device,
        &(RowStorage<W, D>, HbmTensor<i32, m![1], m![R]>),
    )> for Select<W, D>
{
    type Output = HbmTensor<D, m![1], m![R, W]>;

    fn forward(
        &self,
        (device, (input, offsets)): (
            &mut Device,
            &(RowStorage<W, D>, HbmTensor<i32, m![1], m![R]>),
        ),
    ) -> Self::Output {
        self.forward((device, (input, offsets)))
    }
}

impl<W: AxisName, R: AxisName, D: MaterializableScalar, C: AxisName>
    Module<(
        &mut Device,
        (
            &HbmTensor<D, m![1], m![R, W]>,
            &HbmTensor<i32, m![1], m![C]>,
        ),
    )> for Select<W, D>
{
    type Output = HbmTensor<D, m![1], m![C, W]>;

    fn forward(
        &self,
        (device, (input, offsets)): (
            &mut Device,
            (
                &HbmTensor<D, m![1], m![R, W]>,
                &HbmTensor<i32, m![1], m![C]>,
            ),
        ),
    ) -> Self::Output {
        if const { <Groups<C> as M>::SIZE == 1 } {
            let selected: Selected<W, C, D> = input
                .gather::<m![R], m![W]>()
                .by_byte_offsets(offsets)
                .to_dm(&mut device.tdma);
            selected.to_hbm(&mut device.tdma)
        } else {
            let mut output = HbmTensor::new();
            for group in 0..<Groups<C> as M>::SIZE {
                let mut index: Indices<C> = HbmTensor::new();
                offsets
                    .view()
                    .tile::<Groups<C>, 1, Read<C>>(group)
                    .to_hbm_view(&mut device.tdma, index.view_mut());
                let selected: Selected<W, C, D> = input
                    .gather::<m![R], m![W]>()
                    .by_byte_offsets(&index)
                    .to_dm(&mut device.tdma);
                selected.view().to_hbm_view(
                    &mut device.tdma,
                    output.view_mut().tile::<Groups<C>, 1, Write<W, C>>(group),
                );
            }
            output
        }
    }
}

impl<W: AxisName, N: AxisName, C: AxisName, D: MaterializableScalar>
    Module<(
        &mut Device,
        &(HbmTensor<D, m![1], m![N, W]>, HbmTensor<i32, m![1], m![C]>),
    )> for Select<W, D>
{
    type Output = HbmTensor<D, m![1], m![C, W]>;

    fn forward(
        &self,
        (device, (rows, offsets)): (
            &mut Device,
            &(HbmTensor<D, m![1], m![N, W]>, HbmTensor<i32, m![1], m![C]>),
        ),
    ) -> Self::Output {
        self.forward((device, (rows, offsets)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::attention::geometry::Hidden;
    use crate::ops::layout::axes::{Eight, Single, Sixteen};
    use furiosa_opt_std::runtime::launch;

    #[device(chip = 1)]
    fn selected<Width: AxisName, D: MaterializableScalar, Rows: AxisName>(
        device: &mut Device,
        select: &Select<Width, D>,
        input: &(RowStorage<Width, D>, HbmTensor<i32, m![1], m![Rows]>),
    ) -> HbmTensor<D, m![1], m![Rows, Width]> {
        select.forward((device, input))
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
                crate::ops::tensor::alias(
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
        let mut output: HbmTensor<D, m![1], m![Rows, Width]> =
            crate::ops::tensor::alias(&buffer, ..).unwrap();
        let model = Select::<Width, D>::default();
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
        select: &Select<Width, D>,
        input: &(
            HbmTensor<D, m![1], m![Source, Width]>,
            HbmTensor<i32, m![1], m![Rows]>,
        ),
    ) -> HbmTensor<D, m![1], m![Rows, Width]> {
        select.forward((device, input))
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
        let model = Select::<Width, D>::default();
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
