use crate::Module;
use crate::compose::Tiles;
use crate::ops::indexing::Offset;
pub use crate::ops::layout::Packets;
use crate::ops::layout::{
    Clusters, Dm, Live, Rows as Layout,
    axes::{Chip, Lanes, Single},
};
use crate::ops::linear::Groups;
use crate::ops::normalization::Spread;
use crate::ops::vector::scalar;
use crate::ops::weight::Weight;
use furiosa_opt_std::prelude::*;

/// The one tile width, from the gather's 16-element slices: 16 divides it and it divides
/// 4096 = 256 * 16, so a tile fills 32 slices and `Hidden / 512` gathers cover any such width.
pub const TILE: usize = 512;

/// The ids of `Rows` tokens, a packet of eight per row group as the DMA moves them; the lanes past
/// `Rows` are ignored.
pub type Ids<Rows> = HbmTensor<i32, crate::ops::layout::axes::Chip, Packets<Rows>>;

/// Look up token features by a row byte offset or by packet-padded token IDs.
#[derive(DeviceSend)]
pub struct Embedding<
    Vocab: AxisName,
    Hidden: AxisName,
    Chips: M = m![1],
    Parameter = Weight<m![Vocab, Hidden], Chips>,
> {
    weight: Parameter,
    features: core::marker::PhantomData<(Vocab, Hidden, Chips)>,
}

impl<Vocab: AxisName, Hidden: AxisName, Chips: M, Parameter>
    Embedding<Vocab, Hidden, Chips, Parameter>
{
    pub const fn new(weight: Parameter) -> Self {
        Self {
            weight,
            features: core::marker::PhantomData,
        }
    }

    pub const fn weight(&self) -> &Parameter {
        &self.weight
    }
}

/// Select one embedding row by its byte offset.
impl<Vocab: AxisName, Hidden: AxisName, Chips: M> Module<(&mut Device, &Offset<Chips>)>
    for Embedding<Vocab, Hidden, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Hidden]>;

    fn forward(&self, (device, offset): (&mut Device, &Offset<Chips>)) -> Self::Output {
        let row: DmTensor<bf16, Chips, m![1 # 2], m![1 # 256], m![Hidden]> = self
            .weight
            .tensor()
            .gather::<m![Vocab], m![Hidden]>()
            .by_byte_offsets(offset)
            .to_dm(&mut device.tdma);
        row.to_hbm(&mut device.tdma)
    }
}

/// Select the row of one token ID, at the byte offset of its row.
impl<Vocab: AxisName, Hidden: AxisName, Chips: M>
    Module<(&mut Device, &HbmTensor<i32, Chips, m![1]>)> for Embedding<Vocab, Hidden, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Hidden]>;

    fn forward(&self, (device, id): (&mut Device, &HbmTensor<i32, Chips, m![1]>)) -> Self::Output {
        let offset = scalar::offset::<bf16, m![Hidden], m![Single], _>(device, id);
        self.forward((device, &offset))
    }
}

// Keep row axes visible: associated mappings hide Rows from inference between composed modules.
pub type Vectors<Hidden, Rows> = DmTensor<
    bf16,
    m![1],
    m![Rows / ({ Symbol<Lanes<Rows>> }) % ({ Symbol<Clusters<Rows>> }) # 2],
    m![Rows % ({ Symbol<Lanes<Rows>> }) # 8, Hidden % 512 / 16],
    m![Hidden / 512, Rows / ({ Symbol<Lanes<Rows>> }) / ({ Symbol<Clusters<Rows>> }), Hidden % 16],
>;

impl<'d, 'x, Vocab: AxisName, Hidden: AxisName, Rows: AxisName>
    Module<(&'d mut Device, &'x HbmTensor<i32, m![1], Packets<Rows>>)>
    for Embedding<Vocab, Hidden>
{
    type Output = Vectors<Hidden, Rows>;

    fn forward(
        &self,
        (device, ids): (&'d mut Device, &'x HbmTensor<i32, m![1], Packets<Rows>>),
    ) -> Self::Output {
        use crate::ops::layout::{BroadcastClusters as Cluster, BroadcastSlices};

        // The byte offset of each token's embedding row.
        let ids: DmTensor<i32, Chip, Cluster, BroadcastSlices, Packets<Rows>> =
            ids.to_dm(&mut device.tdma);
        let offsets: DmTensor<i32, Chip, Cluster, BroadcastSlices, Packets<Rows>> = device
            .main
            .begin(ids.view())
            .fetch::<m![{ Groups<Rows> }], m![{ Layout<Rows>::Lanes } # 8]>()
            .fetch_cast::<i32>()
            .collect::<m![{ Groups<Rows> }], m![{ Layout<Rows>::Lanes } # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(
                FxpBinaryOp::MulInt,
                const { (Hidden::SIZE * core::mem::size_of::<bf16>()) as i32 },
            )
            .vector_final()
            .commit_trim::<m![{ Layout<Rows>::Lanes } # 8]>()
            .commit();
        let offsets: HbmTensor<i32, Chip, m![Rows]> = offsets.to_hbm(&mut device.tdma);
        let weight = self.weight.tensor();
        let mut output = DmTensor::new();
        for tile in 0..(Hidden::SIZE / TILE) {
            let gathered: DmTensor<
                bf16, m![1], m![{ Layout<Rows>::Spatial } # 2],
                m![{ Layout<Rows>::Lanes } # 8, Hidden % 512 / 16],
                m![{ Layout<Rows>::Serial }, Hidden % 16],
            > = weight.view()
                .tile::<m![Hidden / 512], 1, m![Vocab, Hidden / 512 = 1 # (Hidden / 512), Hidden % 512]>(tile)
                .gather::<m![Vocab], m![Hidden / 512 = 1 # (Hidden / 512), Hidden % 512]>()
                .by_byte_offsets(&offsets)
                .to_dm(&mut device.tdma);
            gathered.view().to_dm_view(
                &mut device.tdma,
                output.view_mut()
                    .tile::<m![Hidden / 512], 1, m![1 #{!} (Hidden / 512), { Layout<Rows>::Serial }, Hidden % 16]>(tile),
            );
        }
        output
    }
}

/// The rows of a launch's tokens in the decoder's feature mapping, with the launch's live tiles.
impl<'x, Vocab: AxisName, Hidden: AxisName, Rows: AxisName>
    Module<(&mut Device, (&'x Ids<Rows>, Live<Rows>))> for Embedding<Vocab, Hidden>
{
    type Output = Dm<Lanes<Rows>, Spread<Hidden, Rows>>;

    fn forward(
        &self,
        (device, (ids, tiles)): (&mut Device, (&'x Ids<Rows>, Live<Rows>)),
    ) -> Self::Output {
        let rows = Dm {
            value: self.forward((&mut *device, ids)),
            tiles,
        };
        Reorder::<Hidden>::default().forward((device, rows))
    }
}

/// Places embedding rows in the feature mapping required by a decoder.
#[derive(DeviceSend)]
pub struct Reorder<Hidden>(core::marker::PhantomData<Hidden>);

impl<Hidden> Default for Reorder<Hidden> {
    fn default() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<Hidden: AxisName, Rows: AxisName> Module<(&mut Device, Vectors<Hidden, Rows>)>
    for Reorder<Hidden>
{
    type Output = Spread<Hidden, Rows>;

    fn forward(&self, (device, input): (&mut Device, Vectors<Hidden, Rows>)) -> Self::Output {
        crate::ops::layout::Reorder::default().forward((device, input))
    }
}

/// Only the live rows: every row in one copy when every packet is live, else each serial step of
/// the gathered rows that holds a live row; a later step's rows are left unwritten.
impl<Hidden: AxisName, Rows: AxisName> Module<(&mut Device, Dm<Lanes<Rows>, Vectors<Hidden, Rows>>)>
    for Reorder<Hidden>
{
    type Output = Dm<Lanes<Rows>, Spread<Hidden, Rows>>;

    fn forward(
        &self,
        (device, input): (&mut Device, Dm<Lanes<Rows>, Vectors<Hidden, Rows>>),
    ) -> Self::Output {
        let Dm { value, tiles } = input;
        if tiles.count() == <Groups<Rows> as M>::SIZE {
            return Dm {
                value: self.forward((device, value)),
                tiles,
            };
        }
        // Gathered rows split each step of `Lanes * Clusters` rows over clusters and slices, so a
        // step is the smallest copy.
        let step = Lanes::<Rows>::SIZE * Clusters::<Rows>::SIZE;
        let mut output: Spread<Hidden, Rows> = DmTensor::new();
        for serial in 0..(tiles.rows() + step - 1) / step {
            value
                .view()
                .tile::<Layout<Rows>::Serial, 1, m![
                    Hidden / 512,
                    { Layout<Rows>::Serial } = 1 # ({ Layout<Rows>::Serial }),
                    Hidden % 16
                ]>(serial)
                .to_dm_view(
                    &mut device.tdma,
                    output.view_mut().tile::<Layout<Rows>::Serial, 1, m![
                        { Layout<Rows>::Serial } = 1 #{!} ({ Layout<Rows>::Serial }),
                        { Layout<Rows>::Spatial },
                        { Layout<Rows>::Lanes },
                        Hidden % ({ Symbol<crate::ops::normalization::Chunk<Hidden>> })
                    ]>(serial),
                );
        }
        Dm {
            value: output,
            tiles,
        }
    }
}

/// A launch's supplied rows in the decoder's feature mapping, with the launch's live tiles.
impl<'x, Hidden: AxisName, Rows: AxisName, S>
    Module<(
        &mut Device,
        (&'x HbmTensor<bf16, m![1], m![Rows, Hidden]>, Tiles<S>),
    )> for Reorder<Hidden>
{
    type Output = Dm<S, Spread<Hidden, Rows>>;

    fn forward(
        &self,
        (device, (rows, tiles)): (
            &mut Device,
            (&'x HbmTensor<bf16, m![1], m![Rows, Hidden]>, Tiles<S>),
        ),
    ) -> Self::Output {
        Dm {
            value: rows.to_dm(&mut device.tdma),
            tiles,
        }
    }
}

/// Embedding parameters whose row and column counts come from the checkpoint.
#[derive(DeviceSend)]
pub struct Lookup<Weight> {
    pub weight: Weight,
}

impl<Weight> Lookup<Weight> {
    pub fn new(weight: Weight) -> Self {
        Self { weight }
    }
}

impl<D: Scalar, Vocab: AxisName, Width: AxisName, Rows: AxisName>
    Module<(&mut Device, &HbmTensor<i32, m![1], m![Rows]>)>
    for Lookup<HbmTensor<D, m![1], m![Vocab, Width]>>
{
    type Output = HbmTensor<D, m![1], m![Rows, Width]>;

    fn forward(&self, _: (&mut Device, &HbmTensor<i32, m![1], m![Rows]>)) -> Self::Output {
        todo!("NPU embedding lookup into contiguous rows")
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use crate::Module;
    use crate::compose::{Count, tests::count};
    use crate::ops::layout::Live;
    use crate::ops::layout::axes::{Eight, Single, Sixteen};
    use crate::ops::normalization::RowMajor;
    use crate::ops::weight::Weight;

    use super::{Embedding, Ids, Packets};

    #[device(chip = 1)]
    fn forward<Vocab: AxisName, Hidden: AxisName, Rows: AxisName>(
        device: &mut Device,
        embedding: &Embedding<Vocab, Hidden>,
        ids: &Ids<Rows>,
        output: &mut HbmTensor<bf16, m![1], RowMajor<Hidden, Rows>>,
    ) {
        embedding
            .forward((&mut *device, ids))
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    async fn embedding<Vocab: AxisName, Hidden: AxisName>(
        device: &mut Device,
        weight: HostTensor<bf16, m![Vocab, Hidden]>,
    ) -> std::io::Result<Embedding<Vocab, Hidden>> {
        Ok(Embedding {
            features: core::marker::PhantomData,
            weight: Weight::new(
                weight
                    .to_hbm(&mut device.pdma)
                    .await
                    .map_err(std::io::Error::other)?,
            ),
        })
    }

    axes![Table = 64, Served = 1024, Wide = 5120];

    type Chip = m![1];

    fn table<Hidden: AxisName>() -> Vec<bf16> {
        (0..Table::SIZE * Hidden::SIZE)
            .map(|index| bf16::from_f32(((index * 37) % 101) as f32 / 25.0 - 2.0))
            .collect()
    }

    /// The token ids, eight to the packet, the lanes past them repeating
    /// the first.
    fn ids(rows: &[usize]) -> Vec<i32> {
        (0..8)
            .map(|lane| rows[lane.min(rows.len() - 1)] as i32)
            .collect()
    }

    /// Eight tokens' rows land each in its own hidden row, bit for bit.
    #[tokio::test]
    async fn gathers_a_row_per_token() {
        let mut device = Device::new(forward.topology()).unwrap();
        let values = table::<Served>();
        let rows = [23, 5, 61, 0, 40, 7, 63, 12];
        let embedding =
            embedding::<Table, Served>(&mut device, HostTensor::from_vec(values.clone()))
                .await
                .unwrap();
        let offsets = HostTensor::<i32, Packets<Eight>>::from_vec(ids(&rows))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut output: HbmTensor<bf16, Chip, RowMajor<Served, Eight>> =
            HostTensor::<bf16, RowMajor<Served, Eight>>::zero()
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();

        launch(forward, (&mut device, &embedding, &offsets, &mut output))
            .await
            .unwrap();

        let gathered = output
            .to_host::<RowMajor<Served, Eight>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, row) in rows.into_iter().enumerate() {
            assert_eq!(
                gathered[at * Served::SIZE..][..Served::SIZE]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                values[row * Served::SIZE..][..Served::SIZE]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "token {at} (row {row})"
            );
        }
    }

    /// A 5120-wide row spans 320 slices, exceeding one gather instruction's slice count.
    #[tokio::test]
    async fn gathers_a_width_the_monolithic_geometry_cannot_lay_out() {
        let mut device = Device::new(forward.topology()).unwrap();
        let values = table::<Wide>();
        let row = 41;
        let embedding = embedding::<Table, Wide>(&mut device, HostTensor::from_vec(values.clone()))
            .await
            .unwrap();
        let offsets = HostTensor::<i32, Packets<Single>>::from_vec(ids(&[row]))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut output: HbmTensor<bf16, Chip, RowMajor<Wide, Single>> =
            HostTensor::<bf16, RowMajor<Wide, Single>>::zero()
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();

        launch(forward, (&mut device, &embedding, &offsets, &mut output))
            .await
            .unwrap();

        let gathered = output
            .to_host::<RowMajor<Wide, Single>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (column, observed) in gathered.iter().enumerate() {
            let expected = values[row * Wide::SIZE + column];
            assert_eq!(
                observed.to_bits(),
                expected.to_bits(),
                "column {column}: expected {}, observed {}",
                expected.to_f32(),
                observed.to_f32(),
            );
        }
    }

    #[device(chip = 1)]
    fn live<Vocab: AxisName, Hidden: AxisName>(
        device: &mut Device,
        embedding: &Embedding<Vocab, Hidden>,
        rows: &Count,
        ids: &Ids<Sixteen>,
    ) -> HbmTensor<bf16, Chip, RowMajor<Hidden, Sixteen>> {
        embedding
            .forward((&mut *device, (ids, Live::<Sixteen>::read(rows))))
            .value
            .to_hbm(&mut device.tdma)
    }

    /// The live tokens of a partial launch, one whole row group and part of the next, each land
    /// their table row bit for bit in the decoder's feature mapping.
    #[tokio::test]
    async fn lands_the_rows_of_live_tokens() {
        let mut device = Device::new(live.topology()).unwrap();
        let values = table::<Served>();
        let tokens: Vec<usize> = (0..Sixteen::SIZE)
            .map(|row| (row * 37 + 11) % Table::SIZE)
            .collect();
        let embedding =
            embedding::<Table, Served>(&mut device, HostTensor::from_vec(values.clone()))
                .await
                .unwrap();
        let ids =
            HostTensor::<i32, Packets<Sixteen>>::from_vec(tokens.iter().map(|&row| row as i32))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let live_rows = 11;
        let rows = count(live_rows).to_hbm(&mut device.pdma).await.unwrap();
        let landed = launch(live, (&mut device, &embedding, &rows, &ids))
            .await
            .unwrap()
            .to_host::<RowMajor<Served, Sixteen>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, row) in tokens.into_iter().take(live_rows).enumerate() {
            assert_eq!(
                landed[at * Served::SIZE..][..Served::SIZE]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                values[row * Served::SIZE..][..Served::SIZE]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "token {at} (row {row})"
            );
        }
    }
}
