//! Block-scaled E4M3 projections of resident rows: a launch's rows share each weight byte it loads.
//!
//! - A panel is a run of whole 128-input blocks; the weights are stored panel by panel. Each slice
//!   holds one panel of one owner's output rows, so it streams only its panel's weights.
//! - A cluster owns half the output rows; an owner, a quarter of a cluster's rows, takes one PE's
//!   slices (two PEs at long inputs). For example 5120 inputs make 40 panels of one block each.
//! - Each slice quantizes its panel of every row by the row's own 128-input group scale and holds
//!   the codes in its eight tensor lanes, one row a lane, as the weights stream past once.
//! - Each block's FP32 sum takes the row's activation scale, then the block's weight scale; the
//!   blocks then add up within the slice and the panels across the owner's slices.

mod layout;

use super::GROUP;
use super::block::{FLOOR, Group, MAGNITUDE, RECIPROCAL};
use crate::Module;
use crate::ops::layout::{Dm, axes::Lanes};
use crate::ops::normalization::Spread;
use furiosa_opt_std::prelude::*;
use layout::*;

pub use layout::Panel;

// The tensor lanes the rows of a launch take, one row a lane.
axes![Lane = 8];

/// E4M3 coefficients by input panel: `[panel, output, input of the panel]`.
pub type Weights<In, Out> =
    m![In / ({ Symbol<Panel<In, Out>> }), Out, In % ({ Symbol<Panel<In, Out>> })];

/// One FP32 weight scale per four output rows and 128 inputs, by input panel: a checkpoint's
/// 128 x 128 block scale repeats over the 32 four-row tiles of its block.
pub type Scales<In, Out> =
    m![In / ({ Symbol<Panel<In, Out>> }), Out / 4, In % ({ Symbol<Panel<In, Out>> }) / 128];

/// E4M3 weights in 128 x 128 blocks of one scale each, stored by input panel, contracted with
/// resident rows quantized per row and 128-input group at run time.
#[derive(DeviceSend)]
pub struct BlockPanels<In: AxisName, Out: AxisName> {
    /// E4M3 coefficients by input panel.
    pub weight: HbmTensor<f8e4m3, m![1], Weights<In, Out>>,
    /// Multiply a block's contracted sum by its scale to restore weight units.
    pub weight_scale: HbmTensor<f32, m![1], Scales<In, Out>>,
}

impl<In: AxisName, Out: AxisName> BlockPanels<In, Out> {
    /// Canonical `[Out, In]` coefficients in the stored order: each input panel's
    /// `[Out, inputs of the panel]` in turn.
    pub fn weights<T: Copy>(canonical: &[T]) -> Vec<T> {
        let (inputs, panel) = (In::SIZE, Panel::<In, Out>::SIZE);
        assert_eq!(canonical.len(), Out::SIZE * inputs, "an [Out, In] matrix");
        (0..inputs / panel)
            .flat_map(|part| {
                canonical
                    .chunks_exact(inputs)
                    .flat_map(move |row| &row[part * panel..][..panel])
            })
            .copied()
            .collect()
    }

    /// A canonical `[Out / 128, In / 128]` block-scale grid in the stored order: each input
    /// panel's `[Out / 4, blocks of the panel]` in turn, a block row once a four-row tile.
    pub fn scales(grid: &[f32]) -> Vec<f32> {
        let (groups, blocks) = (In::SIZE / GROUP, Panel::<In, Out>::SIZE / GROUP);
        assert_eq!(
            grid.len(),
            Out::SIZE / GROUP * groups,
            "an [Out / 128, In / 128] grid"
        );
        (0..groups / blocks)
            .flat_map(|part| {
                (0..Out::SIZE / 4).flat_map(move |tile| {
                    let row = tile * 4 / GROUP * groups + part * blocks;
                    (row..row + blocks).map(|at| grid[at])
                })
            })
            .collect()
    }
}

/// The rows of a launch, each projected as it would be alone; the live tiles pass on unchanged,
/// since every row of an envelope of at most eight shares one pass.
impl<'d, 'x, I: AxisName, O: AxisName, R: AxisName, C: M, S: M, F: M>
    Module<(
        &'d mut Device,
        &'x Dm<Lanes<R>, DmTensor<bf16, m![1], C, S, m![R, { F }]>>,
    )> for BlockPanels<I, O>
{
    type Output = Dm<Lanes<R>, Spread<O, R>>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            &'x Dm<Lanes<R>, DmTensor<bf16, m![1], C, S, m![R, { F }]>>,
        ),
    ) -> Self::Output {
        const { assert!(R::SIZE <= Lane::SIZE) };
        let mut staged: Sliced<
            bf16,
            I,
            O,
            m![
                { Inputs<I, O> } / ({ Symbol<Granule<I, O>> }),
                R # 8,
                { Inputs<I, O> } % ({ Symbol<Granule<I, O>> })
            ],
        > = DmTensor::new();
        input
            .value
            .view()
            .to_dm_view(&mut device.tdma, staged.view_mut());
        let staged: Sliced<
            bf16,
            I,
            O,
            m![
                [{ Blocks<I, O> }, Group] / ({ Symbol<Granule<I, O>> }),
                Lane,
                [{ Blocks<I, O> }, Group] % ({ Symbol<Granule<I, O>> })
            ],
        > = unsafe { staged.reshape() };
        let scales: Sliced<f32, I, O, m![{ Outputs<I, O> } / 4, { Blocks<I, O> }]> =
            self.weight_scale.to_dm(&mut device.tdma);
        let activation: Sliced<f32, I, O, m![{ Blocks<I, O> }, Lane]> = device
            .main
            .begin(staged.view())
            .fetch::<m![{ Blocks<I, O> }, Lane, Group / 16], m![Group % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Blocks<I, O> }, Lane, Group / 8], m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpF32::BitAnd, MAGNITUDE)
            .vector_narrow_split::<m![{ Blocks<I, O> }, Lane, Group / 4], m![Group % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), RECIPROCAL)
            .vector_intra_slice_reduce::<Group, m![{ Blocks<I, O> }, Lane], m![1 # 4]>(
                IntraSliceReduceOpF32::Max,
            )
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Max, FLOOR)
            .vector_final()
            .transpose::<m![{ Blocks<I, O> }, Lane / 2], m![Lane % 2 # 8]>()
            .commit_trim::<m![Lane % 2]>()
            .commit();
        let activation: VrfTensor<
            f32,
            m![1],
            Cluster<O>,
            Slices<I, O>,
            m![{ Blocks<I, O> }, Lane],
        > = device
            .sub
            .begin(activation.view())
            .fetch::<Blocks<I, O>, m![Lane]>()
            .collect::<Blocks<I, O>, m![Lane]>()
            .to_vrf();
        let codes: Sliced<f8e4m3, I, O, m![Lane, { Blocks<I, O> }, Group]> = device
            .main
            .begin(staged.view())
            .fetch::<m![Lane, { Blocks<I, O> }, Group / 16], m![Group % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Lane, { Blocks<I, O> }, Group / 8], m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Lane, { Blocks<I, O> }, Group / 4], m![Group % 4]>()
            .vector_fp_div(&activation)
            .vector_widen_concat::<m![Lane, { Blocks<I, O> }, Group / 8], m![Group % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Group % 8 # 32]>()
            .commit_trim::<m![Group % 8]>()
            .commit();
        let codes: Sliced<f8e4m3, I, O, m![Lane, { Inputs<I, O> }]> = unsafe { codes.reshape() };
        let lanes: TrfTensor<f8e4m3, m![1], Cluster<O>, Slices<I, O>, m![Lane], Inputs<I, O>> =
            device
                .sub
                .begin(codes.view())
                .fetch::<m![Lane, { Inputs<I, O> } / 32], m![I % 32]>()
                .collect::<m![Lane, { Inputs<I, O> } / 32], m![I % 32]>()
                .to_trf();
        let mut partials: Sliced<f32, I, O, m![{ Outputs<I, O> }, Lane]> = DmTensor::new();
        #[unroll]
        for slab in 0..<Slabs<I, O> as M>::SIZE {
            let weight: Sliced<f8e4m3, I, O, m![{ Carried<I, O> }, { Inputs<I, O> }]> = self
                .weight
                .view()
                .tile::<Slabs<I, O>, 1, m![
                    { Parts<I, O> },
                    O / ({ Symbol<Owned<I, O>> }),
                    { Slabs<I, O> } = 1 # ({ Slabs<I, O> }),
                    { Carried<I, O> },
                    { Inputs<I, O> }
                ]>(slab)
                .to_dm(&mut device.tdma);
            let scale: VrfTensor<
                f32,
                m![1],
                Cluster<O>,
                Slices<I, O>,
                m![{ Carried<I, O> } / 4, { Blocks<I, O> }, 1 # 8],
            > = device
                .sub
                .begin(scales.view().tile::<Slabs<I, O>, 1, m![
                    { Slabs<I, O> } = 1 # ({ Slabs<I, O> }),
                    { Carried<I, O> } / 4,
                    { Blocks<I, O> }
                ]>(slab))
                .fetch::<m![{ Carried<I, O> } / 4, { Blocks<I, O> }], m![1 # 8]>()
                .collect::<m![{ Carried<I, O> } / 4, { Blocks<I, O> }], m![1 # 8]>()
                .to_vrf();
            device
                .main
                .begin(weight.view())
                .fetch::<m![{ Carried<I, O> }, { Inputs<I, O> } / 32], m![I % 32]>()
                .collect::<m![{ Carried<I, O> }, { Inputs<I, O> } / 32], m![I % 32]>()
                .contract_outer::<
                    m![{ Carried<I, O> }, { Blocks<I, O> }, I % 128 / 64],
                    m![I % 64],
                    _,
                    _,
                    _,
                >(&lanes)
                .contract_packet::<m![1]>()
                .contract_time::<m![{ Carried<I, O> }, { Blocks<I, O> }]>()
                .contract_lane::<m![{ Carried<I, O> }, { Blocks<I, O> }], m![Lane]>(
                    LaneMode::Interleaved,
                )
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Carried<I, O> }, { Blocks<I, O> }, Lane / 4], m![Lane % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &activation)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
                .vector_intra_slice_reduce::<I, m![{ Carried<I, O> }, Lane / 4], m![Lane % 4]>(
                    IntraSliceReduceOpF32::Add,
                )
                .vector_widen_concat::<Carried<I, O>, m![Lane]>()
                .vector_final()
                .commit_trim::<m![Lane]>()
                .commit_view(partials.view_mut().tile::<Slabs<I, O>, 1, m![
                    { Slabs<I, O> } = 1 #{!} ({ Slabs<I, O> }),
                    { Carried<I, O> },
                    Lane
                ]>(slab));
        }
        let result: DmTensor<
            bf16,
            m![1],
            Cluster<O>,
            Reduced<I, O>,
            m![
                { Outputs<I, O> } / ({ Symbol<Piece<I, O>> }),
                Lane,
                { Outputs<I, O> } % ({ Symbol<Piece<I, O>> })
            ],
        > = device
            .main
            .begin(partials.view())
            .fetch::<Outputs<I, O>, m![Lane]>()
            .collect::<Outputs<I, O>, m![Lane]>()
            .vector_init()
            .vector_inter_slice_reduce::<Reduced<I, O>, Outputs<I, O>>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![Lane # 16]>()
            .transpose::<m![{ Outputs<I, O> } / 4, Lane], m![O % 4 # 16]>()
            .commit_trim::<m![O % 4]>()
            .commit();
        let result: DmTensor<
            bf16,
            m![1],
            Cluster<O>,
            Reduced<I, O>,
            m![
                { Outputs<I, O> } / ({ Symbol<Piece<I, O>> }),
                R # 8,
                { Outputs<I, O> } % ({ Symbol<Piece<I, O>> })
            ],
        > = unsafe { result.reshape() };
        let mut output: Spread<O, R> = DmTensor::new();
        result
            .view()
            .to_dm_view(&mut device.tdma, output.view_mut());
        Dm {
            value: output,
            tiles: input.tiles,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::block::tests::{EXPONENTS, hash, integers, powers, reference};
    use super::*;
    use crate::compose::{Count, Tiles, tests::count};
    use crate::ops::layout::axes::Eight;

    axes![
        Narrow = 768,
        Middle = 512,
        Wide = 1024,
        Long = 1536,
        Three = 3
    ];

    #[device(chip = 1)]
    fn projected<R: AxisName, I: AxisName, O: AxisName>(
        device: &mut Device,
        linear: &BlockPanels<I, O>,
        rows: &HbmTensor<bf16, m![1], m![R, I]>,
        live: &Count,
    ) -> HbmTensor<bf16, m![1], m![R, O]> {
        let rows: Spread<I, R> = rows.to_dm(&mut device.tdma);
        let rows: Dm<Lanes<R>, _> = Dm {
            value: rows,
            tiles: Tiles::read(live),
        };
        linear
            .forward((&mut *device, &rows))
            .value
            .to_hbm(&mut device.tdma)
    }

    async fn panels<I: AxisName, O: AxisName>(
        device: &mut Device,
        weight: &[f8e4m3],
        grid: &[f32],
    ) -> BlockPanels<I, O> {
        BlockPanels {
            weight: HostTensor::<f8e4m3, Weights<I, O>>::from_vec(BlockPanels::<I, O>::weights(
                weight,
            ))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
            weight_scale: HostTensor::<f32, Scales<I, O>>::from_vec(BlockPanels::<I, O>::scales(
                grid,
            ))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        }
    }

    /// The `[R, O]` outputs of `R` rows of which the first `live` are live.
    async fn land<R: AxisName, I: AxisName, O: AxisName>(
        device: &mut Device,
        linear: &BlockPanels<I, O>,
        rows: &[bf16],
        live: usize,
    ) -> Vec<f32> {
        let rows: HbmTensor<bf16, m![1], m![R, I]> =
            HostTensor::<bf16, m![R, I]>::from_vec(rows.to_vec())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let live: Count = count(live).to_hbm(&mut device.pdma).await.unwrap();
        launch(projected, (&mut *device, linear, &rows, &live))
            .await
            .unwrap()
            .to_host::<m![R, O]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
            .into_iter()
            .map(|value| value.to_f32())
            .collect()
    }

    /// Each live row quantizes by its own group scales as the reference does, whatever rows pad
    /// the launch.
    #[tokio::test(flavor = "current_thread")]
    async fn quantizes_rows_alone() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        let (inputs, outputs) = (Narrow::SIZE, Wide::SIZE);
        // Ties to even, E4M3 subnormals and flushes, saturation, an all-zero and a floored group;
        // each output reads one input through its own block scale.
        let edges = [
            448.0,
            -448.0,
            9.5,
            -9.5,
            17.0,
            1.0625,
            1.1875,
            -1.1875,
            0.00390625,
            0.0029296875,
            0.00048828125,
            -0.0029296875,
            0.0,
            300.0,
            3.3,
            -0.7,
        ];
        let rows: Vec<bf16> = (0..Eight::SIZE * inputs)
            .map(|index| {
                let (row, at) = (index / inputs, index % inputs);
                let unit = 2f32.powi(row as i32 - 3);
                bf16::from_f32(match (at / GROUP + row) % 6 {
                    0 => edges[at % edges.len()] / 4.0 * unit,
                    1 if at % GROUP == 77 => 3.7 * unit,
                    1 => ((hash(index, 7) % 1945) as f32 - 972.0) / 270.0 * unit,
                    2 => 0.0,
                    3 => ((hash(index, 9) % 201) as f32 - 100.0) * 1e-14,
                    _ => ((hash(index, 11) % 897) as f32 - 448.0) / 64.0 * unit,
                })
            })
            .collect();
        let source = |output: usize| (output * 5 + 131) % inputs;
        let weight: Vec<f8e4m3> = (0..outputs * inputs)
            .map(|index| f8e4m3::from_f32(f32::from(index % inputs == source(index / inputs))))
            .collect();
        let grid: Vec<f32> = (0..(outputs / GROUP) * (inputs / GROUP))
            .map(|block| 2f32.powi((block * 7 % 9) as i32 - 4))
            .collect();
        let linear = panels::<Narrow, Wide>(&mut device, &weight, &grid).await;
        for live in [8, 3, 1] {
            // Rows from `live` on hold values no live row's result may depend on.
            let mut rows = rows.clone();
            for (index, value) in rows.iter_mut().enumerate().skip(live * inputs) {
                *value =
                    bf16::from_f32([f32::INFINITY, f32::NAN, 3.0e38, -1.0e-30, 0.0][index % 5]);
            }
            let landed = land::<Eight, _, _>(&mut device, &linear, &rows, live).await;
            for (row, input) in rows.chunks_exact(inputs).take(live).enumerate() {
                let expected = reference(input, &weight, &grid);
                for (output, expected) in expected.iter().enumerate() {
                    assert_eq!(
                        landed[row * outputs + output],
                        bf16::from_f32(*expected).to_f32(),
                        "{live} live: row {row} output {output} reads {}",
                        input[source(output)].to_f32()
                    );
                }
            }
        }
    }

    /// Each block's sum takes its row's activation scale and its weight scale before the blocks
    /// add up, as the reference does.
    #[tokio::test(flavor = "current_thread")]
    async fn scales_blocks_before_summing() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        let (inputs, outputs) = (Narrow::SIZE, Middle::SIZE);
        let rows: Vec<bf16> = (0..Eight::SIZE)
            .flat_map(|row| {
                let unit = 2f32.powi(row as i32 % 3 - 1);
                powers(inputs)
                    .into_iter()
                    .map(move |value| bf16::from_f32(value.to_f32() * unit))
            })
            .collect();
        let (weight, grid) = integers(outputs, inputs, 0);
        let linear = panels::<Narrow, Middle>(&mut device, &weight, &grid).await;
        let landed = land::<Eight, _, _>(&mut device, &linear, &rows, Eight::SIZE).await;
        // One scale after the whole sum misses most outputs, so the fixture tells the orders apart.
        let mut differs = 0;
        for (row, input) in rows.chunks_exact(inputs).enumerate() {
            let expected = reference(input, &weight, &grid);
            let unit = 2f32.powi(row as i32 % 3 - 1);
            for (output, expected) in expected.iter().enumerate() {
                let landed = landed[row * outputs + output];
                assert_eq!(
                    landed,
                    bf16::from_f32(*expected).to_f32(),
                    "row {row} output {output}"
                );
                let whole: f32 = (0..inputs)
                    .map(|at| {
                        let code = input[at].to_f32() / (2f32.powi(EXPONENTS[at / GROUP]) * unit);
                        weight[output * inputs + at].to_f32() * code
                    })
                    .sum();
                let once = whole * 2f32.powi(EXPONENTS[0]) * unit * grid[output / GROUP * 6];
                differs += usize::from(bf16::from_f32(once).to_f32() != landed);
            }
        }
        let total = Eight::SIZE * outputs;
        assert!(
            differs > total / 2,
            "{differs} of {total} outputs tell the schemes apart"
        );
    }

    /// Random rows with outliers, E4M3 weights over their whole range and BF16 block scales land
    /// within one BF16 step of the reference's FP32 outputs.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_the_reference() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        let (inputs, outputs) = (Wide::SIZE, Long::SIZE);
        let rows: Vec<bf16> = (0..Eight::SIZE * inputs)
            .map(|index| {
                let value = ((hash(index, 11) % 20001) as f32 - 10000.0) / 4000.0;
                let value = value * (1 + index / inputs) as f32;
                bf16::from_f32(if index % 97 == 0 { value * 40.0 } else { value })
            })
            .collect();
        let weight: Vec<f8e4m3> = (0..outputs * inputs)
            .map(|index| {
                let value = (hash(index, 13) % 897) as f32 - 448.0;
                f8e4m3::from_f32(value / (1 + index % 7) as f32)
            })
            .collect();
        let grid: Vec<f32> = (0..(outputs / GROUP) * (inputs / GROUP))
            .map(|block| {
                bf16::from_f32(1e-4 * (1.0 + (hash(block, 17) % 1000) as f32 / 100.0)).to_f32()
            })
            .collect();
        let linear = panels::<Wide, Long>(&mut device, &weight, &grid).await;
        let landed = land::<Eight, _, _>(&mut device, &linear, &rows, Eight::SIZE).await;
        for (row, input) in rows.chunks_exact(inputs).enumerate() {
            let expected = reference(input, &weight, &grid);
            for (output, expected) in expected.iter().enumerate() {
                let landed = landed[row * outputs + output];
                let step =
                    2f32.powi(expected.abs().max(f32::MIN_POSITIVE).log2().floor() as i32 - 7);
                assert!(
                    (landed - expected).abs() <= step,
                    "row {row} output {output}: {landed} vs {expected}"
                );
            }
        }
    }

    /// An envelope of three rows lands what the envelope of eight lands for the same rows.
    #[tokio::test(flavor = "current_thread")]
    async fn projects_alike_in_any_envelope() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        let (inputs, outputs) = (Narrow::SIZE, Middle::SIZE);
        let rows: Vec<bf16> = (0..Eight::SIZE)
            .flat_map(|_| powers(inputs))
            .enumerate()
            .map(|(index, value)| bf16::from_f32(value.to_f32() * (1 + index / inputs) as f32))
            .collect();
        let (weight, grid) = integers(outputs, inputs, 1);
        let linear = panels::<Narrow, Middle>(&mut device, &weight, &grid).await;
        let eight = land::<Eight, _, _>(&mut device, &linear, &rows, Eight::SIZE).await;
        let three =
            land::<Three, _, _>(&mut device, &linear, &rows[..Three::SIZE * inputs], 3).await;
        assert_eq!(three, eight[..Three::SIZE * outputs]);
    }
}
