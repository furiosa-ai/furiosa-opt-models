//! The block-scaled E4M3 forms of the span MLP's weights: projections of E4M3 rows quantized
//! by row scales, with the weights' channel scales folded into the results.

use super::*;
use crate::Op;
use crate::load::{Context, Load, LoadError};
use crate::modules::generic::schedule::Begin;
use crate::ops::normalization::span::Owned;
use crate::ops::pointwise::add::Add;
use crate::ops::swiglu::Mlp;
use crate::tensor::form::{Projections, fp8};
use crate::tensor::mapping;
use activation::*;
use contract::*;
use input::*;
use spread::*;
use transfer::*;

mod activation;
mod compute;
mod contract;
mod down;
mod input;
mod spread;
mod transfer;

const WAVE: usize = 256;
axes![Copies = 32, Side = 2, Panels = 3];

pub(crate) type Panel<H, I> =
    DmTensor<f8e4m3, Chip, Cluster<I>, Partition<H, I>, m![{ Out<I> }, { K<H> }]>;
pub(crate) type GateUp<H, I> =
    DmTensor<f8e4m3, Chip, Cluster<I>, Partition<H, I>, m![Term, { Out<I> }, { K<H> }]>;
type Both<I, Rows> =
    DmTensor<bf16, Chip, Cluster<I>, Slice<I, Rows>, m![Rows % (Rows / 8), Term, { Out<I> }]>;
type Input<H, I, Rows> = DmTensor<
    f8e4m3,
    Chip,
    Cluster<I>,
    m![H % 1024 / 32, Rows / (Rows / 8)],
    m![H / 1024, Rows % (Rows / 8), H % 32],
>;
type Lanes<H, I, Rows> = TrfTensor<
    f8e4m3,
    Chip,
    Cluster<I>,
    Partition<H, I>,
    m![Rows % 8],
    m![Rows % 16 / 8, H % 128 / 32, Rows / (Rows / 8), H % 32],
>;
/// The down projection's eight blocks of hidden outputs, one a slice of each output block.
type DownSlice<H, I> = m![H / (H / 8), { Blocks<I> }];
pub(crate) type Down<H, I> =
    DmTensor<f8e4m3, Chip, Cluster<I>, DownSlice<H, I>, m![H % (H / 8), I % 64]>;
type Product<H, I, Rows> = DmTensor<bf16, Chip, Cluster<I>, DownSlice<H, I>, m![Rows, I % 64]>;
type Activation<H, I, Rows> = DmTensor<f8e4m3, Chip, Cluster<I>, DownSlice<H, I>, m![Rows, I % 64]>;
type Spread<I, Rows, E> =
    DmTensor<E, Chip, Cluster<I>, Slice<I, Rows>, m![Panels, Rows % (Rows / 8), I % 64]>;
type Scale<H, I, Rows> = DmTensor<bf16, Chip, Cluster<I>, m![H / 16], m![Rows]>;
type RowLanes<Rows> = mapping::Rows<Rows>::Lanes;
type Operand<H, I, Rows, Fill, const W: usize> = TrfTensor<
    f8e4m3,
    Chip,
    Cluster<I>,
    DownSlice<H, I>,
    RowLanes<Rows>,
    m![
        { Fill } % { W } / 32,
        { Fill } / { W },
        Rows % 32 / 8,
        I % 64
    ],
>;
pub(crate) type Chunks<I, Rows> = m![{ Cluster<I> }, I % (I / 2) / 64, Rows, I % 64];
type ChunkSlices<I, Rows> = m![{ Cluster<I> }, { Blocks<I> }, { Out<I> } / 64, Rows, I % 64];
type ChunkPanel<I, Rows> = m![
    { Cluster<I> },
    { Blocks<I> },
    { Out<I> } / 64 = 1 # ({ Out<I> } / 64),
    Rows,
    I % 64
];
type Partial<H, I, Rows> =
    DmTensor<bf16, Chip, Cluster<I>, m![H / 16], m![{ Groups<Rows> }, { RowLanes<Rows> }, H % 16]>;
type Scales<Rows> = HbmTensor<bf16, Chip, m![Rows]>;
type Labels<Rows> = m![
    Rows % (Rows / 8) / 8,
    Rows % (Rows / 2) / (Rows / 4),
    Rows % (Rows / 4) / (Rows / 8),
    Rows % 8
];
type RowScale<I, Rows> = VrfTensor<f32, Chip, Cluster<I>, Slice<I, Rows>, m![Rows % (Rows / 8)]>;
type ChannelScale<I, Rows> = VrfTensor<f32, Chip, Cluster<I>, Slice<I, Rows>, m![{ Out<I> }]>;
type PanelScale<H, I, Rows> =
    VrfTensor<f32, Chip, Cluster<I>, m![H / 16], m![{ Groups<Rows> }, { RowLanes<Rows> }]>;
pub type Stored<H, I> = m![
    { Out<I> } / 64,
    { Cluster<I> },
    H / (H / 8),
    { Blocks<I> },
    H % (H / 8),
    I % 64
];
type StoredPanel<H, I> = m![{ Out<I> } / 64 = 1 # ({ Out<I> } / 64), { Cluster<I> },
    H / (H / 8), { Blocks<I> }, H % (H / 8), I % 64];
pub type Blocked<H, I> = m![Term, H / 1024, I / (I / 64), H % 1024 / 128, { Out<I> }, H % 128];
/// The packed gate/up weights, E4M3 blocks of 128 inputs, and their FP32 channel scales.
pub type GateUpWeights<H, I> = (
    HbmTensor<f8e4m3, Chip, Blocked<H, I>>,
    HbmTensor<f32, Chip, m![Term, I]>,
);
/// The down weights, E4M3 panels of 64 inputs, and their FP32 channel scales.
pub type DownWeights<H, I> = (
    HbmTensor<f8e4m3, Chip, Stored<H, I>>,
    HbmTensor<f32, Chip, m![H]>,
);
pub type QuantizedMlp<H, I> = Mlp<H, I, GateUpWeights<H, I>, DownWeights<H, I>>;

/// The gate and up projections of a block-FP8 checkpoint, each row's block scales folded under its
/// largest ([`fp8::folded`]): their codes packed in [`Blocked`] order, their row scales after them.
impl<'a, H: AxisName, I: AxisName> Load<Projections<'a, 2>> for GateUpWeights<H, I> {
    fn load(context: &mut Context<'_>, projections: Projections<'a, 2>) -> Result<Self, LoadError> {
        let [(gate, gates), (up, ups)] = projections.read(context, |from, cx| {
            cx.field("weight", "weight", |cx| {
                fp8::folded::<m![H], m![I]>(from, cx)
            })
        })?;
        let codes = HostTensor::<f8e4m3, m![{ Chip }, { m![Term, I, H] }]>::from_vec(
            [gate, up].concat().repeat(Chip::SIZE),
        );
        let scales = HostTensor::<f32, m![{ Chip }, { m![Term, I] }]>::from_vec(
            [gates, ups].concat().repeat(Chip::SIZE),
        );
        Ok((
            context.field("0", "", |cx| Load::load(cx, codes))?,
            context.field("1", "", |cx| Load::load(cx, scales))?,
        ))
    }
}

/// The down projection of a block-FP8 checkpoint, each row's block scales folded under its largest
/// ([`fp8::folded`]): its codes in [`Stored`] order, its row scales after them.
impl<'a, H: AxisName, I: AxisName> Load<Projections<'a, 1>> for DownWeights<H, I> {
    fn load(context: &mut Context<'_>, projections: Projections<'a, 1>) -> Result<Self, LoadError> {
        let [(codes, scales)] = projections.read(context, |from, cx| {
            cx.field("weight", "weight", |cx| {
                fp8::folded::<m![I], m![H]>(from, cx)
            })
        })?;
        let codes =
            HostTensor::<f8e4m3, m![{ Chip }, { m![H, I] }]>::from_vec(codes.repeat(Chip::SIZE));
        let scales =
            HostTensor::<f32, m![{ Chip }, { m![H] }]>::from_vec(scales.repeat(Chip::SIZE));
        Ok((
            context.field("0", "", |cx| Load::load(cx, codes))?,
            context.field("1", "", |cx| Load::load(cx, scales))?,
        ))
    }
}
type Channels<I, Rows> = DmTensor<f32, Chip, Cluster<I>, Slice<I, Rows>, m![{ Out<I> }]>;

/// Gate/up weight pair `panel`, the weights of 1024 inputs, written on chip.
impl<Hidden: AxisName, Intermediate: AxisName> Op<Begin<usize>>
    for GateUpWeights<Hidden, Intermediate>
{
    type Output = GateUp<Hidden, Intermediate>;

    fn forward(&self, context: &mut Device, Begin(panel): Begin<usize>) -> Self::Output {
        self.0
            .view()
            .tile::<m![Hidden / 1024], 1, m![Term, Hidden / 1024 = 1 # (Hidden / 1024),
                Intermediate / (Intermediate / 64), Hidden % 1024 / 128, { Out<Intermediate> },
                Hidden % 128]>(panel)
            .to_dm(&mut context.tdma)
    }
}

/// The first gate/up weight pair, which needs no rows: it is copied while the rows travel.
impl<Hidden: AxisName, Intermediate: AxisName> Op<Begin<()>>
    for GateUpWeights<Hidden, Intermediate>
{
    type Output = GateUp<Hidden, Intermediate>;

    fn forward(&self, context: &mut Device, _: Begin<()>) -> Self::Output {
        self.forward(context, Begin(0))
    }
}

/// Down weight panel `panel`, the weights of 64 inputs of each slice's output block, written on
/// chip.
impl<Hidden: AxisName, Intermediate: AxisName> Op<Begin<usize>>
    for DownWeights<Hidden, Intermediate>
{
    type Output = Down<Hidden, Intermediate>;

    fn forward(&self, context: &mut Device, Begin(panel): Begin<usize>) -> Self::Output {
        let mut down = Down::new();
        self.0
            .view()
            .tile::<m![{ Out<Intermediate> } / 64], 1, StoredPanel<Hidden, Intermediate>>(panel)
            .to_dm_view(&mut context.tdma, down.view_mut());
        down
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::Checkpoint;
    use crate::modules::generic::collection::Stacked;
    use crate::ops::swiglu::Intermediate;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;
    use safetensors::Dtype;

    axes![H = 1024];
    type I = Intermediate<4096>;

    /// The E4M3 codes of an `rows` x `columns` projection of layer `layer`, normal values of both
    /// signs.
    fn codes(rows: usize, columns: usize, layer: usize) -> Vec<f8e4m3> {
        (0..rows * columns)
            .map(|index| {
                let magnitude = 0.125 + ((index * 37 + layer * 11) % 61) as f32 / 8.0;
                f8e4m3::from_f32(if index % 3 == 0 {
                    -magnitude
                } else {
                    magnitude
                })
            })
            .collect()
    }

    /// Block scales of a `rows` x `columns` projection that differ along each row by factors no
    /// power of two gives.
    fn blocks(rows: usize, columns: usize, layer: usize) -> Vec<f32> {
        (0..rows.div_ceil(128) * columns.div_ceil(128))
            .map(|block| 0.01 + ((block * 13 + layer * 5) % 17) as f32 * 0.37)
            .collect()
    }

    /// The projections of two layers' MLPs, `layer.{gate,up,down}_proj`, with BF16 block scales.
    fn checkpoint() -> Checkpoint {
        let mut checkpoint = Checkpoint::default();
        for layer in 0..2 {
            for (part, (name, rows, columns)) in [
                ("gate_proj", I::SIZE, H::SIZE),
                ("up_proj", I::SIZE, H::SIZE),
                ("down_proj", H::SIZE, I::SIZE),
            ]
            .into_iter()
            .enumerate()
            {
                let (name, layer) = (format!("{layer}.{name}"), 3 * layer + part);
                let codes = f8e4m3::to_buf(&codes(rows, columns, layer));
                let shape = [rows, columns];
                checkpoint
                    .insert(&format!("{name}.weight"), Dtype::F8_E4M3, &shape, codes)
                    .unwrap();
                let scales: Vec<u8> = blocks(rows, columns, layer)
                    .into_iter()
                    .flat_map(|scale| half::bf16::from_f32(scale).to_le_bytes())
                    .collect();
                let grid = [rows.div_ceil(128), columns.div_ceil(128)];
                let name = format!("{name}.weight_scale_inv");
                checkpoint
                    .insert(&name, Dtype::BF16, &grid, scales)
                    .unwrap();
            }
        }
        checkpoint
    }

    /// The largest error of `folded` codes times their row `scales` against the projection
    /// `name` of `source`, in units of one E4M3 rounding of the stored value.
    fn error(source: &Checkpoint, name: &str, folded: &[f8e4m3], scales: &[f32]) -> f32 {
        let codes = source.get(&format!("{name}.weight")).unwrap();
        let blocks = source.get(&format!("{name}.weight_scale_inv")).unwrap();
        let columns = codes.shape()[1];
        let width = blocks.shape()[1];
        let blocks = Checkpoint::floats(&blocks).unwrap();
        codes
            .data()
            .iter()
            .zip(folded)
            .enumerate()
            .fold(0.0, |worst, (index, (&code, folded))| {
                let (row, column) = (index / columns, index % columns);
                let code = <f8e4m3 as ScalarBytes>::from_le_bytes(&[code]).to_f32();
                let want = code * blocks[row / 128 * width + column / 128];
                let bound = (want.abs() / 16.0).max(scales[row] / 1024.0);
                worst.max((folded.to_f32() * scales[row] - want).abs() / bound)
            })
    }

    /// A stack of block-FP8 MLPs loads each layer's gate and up projections into one packed weight
    /// and its down projection into its own, each row's block scales folded within one E4M3
    /// rounding of every stored value, though they differ along the row.
    #[tokio::test(flavor = "current_thread")]
    async fn loads_block_scaled_projections_folded_by_row() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let source = checkpoint();
        let stack: Stacked<QuantizedMlp<H, I>, 2> = source.load(&mut device).unwrap();
        for layer in 0..2 {
            let (gate_up, down) = stack.layer(layer).into_parts();
            let pdma = &mut device.pdma;
            let codes = gate_up
                .0
                .to_host::<m![Term, I, H]>(pdma)
                .await
                .unwrap()
                .into_vec();
            let scales = gate_up
                .1
                .to_host::<m![Term, I]>(pdma)
                .await
                .unwrap()
                .into_vec();
            let (gate, up) = codes.split_at(I::SIZE * H::SIZE);
            let (gates, ups) = scales.split_at(I::SIZE);
            for (name, codes, scales) in [("gate_proj", gate, gates), ("up_proj", up, ups)] {
                let name = format!("{layer}.{name}");
                assert!(error(&source, &name, codes, scales) <= 1.0, "{name}");
            }
            let codes = down.0.to_host::<m![H, I]>(pdma).await.unwrap().into_vec();
            let scales = down.1.to_host::<m![H]>(pdma).await.unwrap().into_vec();
            let name = format!("{layer}.down_proj");
            assert!(error(&source, &name, &codes, &scales) <= 1.0, "{name}");
        }
    }
}
