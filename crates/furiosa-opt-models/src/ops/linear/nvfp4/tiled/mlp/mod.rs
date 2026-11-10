//! Span MLP: paired gate/up sweeps, FP32 partials, and a blocked product/down handoff.
//! W4A16 arithmetic retains per-16 weight block scales and FP32 sums.

use super::{block, panels::Panels};
use crate::modules::generic::collection::View;
use crate::ops::linear::nvfp4::tiled::prefetch::Points;
use crate::ops::normalization::span::Owned;
use crate::ops::pointwise::add::Add;
use crate::tensor::parallel::Chip;
pub use down::Down;
use furiosa_opt_std::prelude::*;

mod compact;
pub mod down;
mod gate;
mod product;
mod restored_compact;

/// The paired gate/up projections and partitioned down projection of a gated MLP.
#[derive(DeviceSend, crate::load::Load)]
pub struct Mlp<H: AxisName, I: AxisName> {
    pub gate_proj: Panels<H, I>,
    pub up_proj: Panels<H, I>,
    pub down_proj: Down<I, H>,
}
impl<H: AxisName, I: AxisName, const N: usize> View<N> for Mlp<H, I> {
    type Storage = <(Panels<H, I>, (Panels<H, I>, Down<I, H>)) as View<N>>::Storage;
    type Layer = Self;
    fn view(storage: &Self::Storage, layer: usize) -> Self {
        let (gate_proj, (up_proj, down_proj)) =
            <(Panels<H, I>, (Panels<H, I>, Down<I, H>)) as View<N>>::view(storage, layer);
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }
    fn stack(layers: [Self; N], cx: &mut crate::load::Context<'_>) -> Self::Storage {
        <(Panels<H, I>, (Panels<H, I>, Down<I, H>)) as View<N>>::stack(
            layers.map(|layer| (layer.gate_proj, (layer.up_proj, layer.down_proj))),
            cx,
        )
    }
}

type Cluster<I> = m![I / (I / 2)];
type Owners<I> = m![I % (I / 2) / 224];
// The low six row bits are eight slice owners of eight lanes. Keeping bit six in
// local time lets a contraction trim a 128-row wave to its first 64 rows.
type Slices<I, R> = m![{ Owners<I> }, R % 64 / 8];
type LocalRows<R> = m![R % 128 / 64, R % 8];

type Ready<I, R> =
    DmTensor<bf16, Chip, Cluster<I>, Slices<I, R>, m![R / 128, { LocalRows<R> }, I % 224]>;
type Products<I, R> = HbmTensor<bf16, Chip, m![I / 224, I % 224 / 32, R, I % 32]>;
pub(super) trait Product<I: AxisName, R: AxisName> {
    fn stripe<O: AxisName, const N: usize, E: M>(
        &self,
        device: &mut Device,
        stripe: usize,
        start: usize,
        target: DmTensorViewMut<'_, bf16, Chip, Cluster<I>, down::Partition<I, O>, E>,
    );
}

impl<I: AxisName, R: AxisName> Product<I, R> for Products<I, R> {
    fn stripe<O: AxisName, const N: usize, E: M>(
        &self,
        device: &mut Device,
        stripe: usize,
        start: usize,
        target: DmTensorViewMut<'_, bf16, Chip, Cluster<I>, down::Partition<I, O>, E>,
    ) {
        self.view()
            .tile::<m![I % 224 / 32], 1, m![I / 224, I % 224 / 32 = 1 # 7, R, I % 32]>(stripe)
            .tile::<m![R], N, m![I / 224, I % 224 / 32 = 1 # 7, R = {N} # R, I % 32]>(start)
            .to_dm_view(&mut device.tdma, target);
    }
}

type Input<H, R> = HbmTensor<bf16, Chip, m![H / 128, R, H % 128]>;

pub(crate) type Entry<H, I> = (
    super::panels::Packed<H, I>,
    super::panels::Packed<H, I>,
    down::Packed<I, H, down::Wide>,
);

/// Stage the first gate/up slab before the preceding O projection. The returned
/// point loads down panel 0 while O's first contraction is in flight.
pub(crate) fn stage<'a, H: AxisName, I: AxisName>(
    device: &mut Device,
    gate: &Panels<H, I>,
    up: &Panels<H, I>,
    down: &'a Down<I, H>,
) -> impl Points<Device, Output = Entry<H, I>> + use<'a, H, I> {
    let gate = gate.load(device, 0);
    let up = up.load(device, 0);
    crate::ops::linear::nvfp4::tiled::prefetch::Stage(move |device: &mut Device| {
        (gate, up, down.stage(device, 0))
    })
}

/// Gate and up share unquantized BF16 input rows; their weight scales remain independent.
pub fn forward<H: AxisName, I: AxisName, R: AxisName>(
    device: &mut Device,
    gate: &Panels<H, I>,
    up: &Panels<H, I>,
    down: &Down<I, H>,
    input: &Owned<H, R>,
    rows: usize,
) -> Owned<H, R> {
    forward_with_next(device, gate, up, down, input, rows, ()).0
}

/// Host the caller's following operand loads at the first two down-panel boundaries.
pub(crate) fn forward_with_next<H: AxisName, I: AxisName, R: AxisName, N: Points<Device>>(
    device: &mut Device,
    gate: &Panels<H, I>,
    up: &Panels<H, I>,
    down: &Down<I, H>,
    input: &Owned<H, R>,
    rows: usize,
    next: N,
) -> (Owned<H, R>, N::Output) {
    let entry = stage(device, gate, up, down).finish(device);
    forward_prepared(device, gate, up, down, input, rows, entry, next)
}

pub(crate) fn forward_prepared<H: AxisName, I: AxisName, R: AxisName, N: Points<Device>>(
    device: &mut Device,
    gate: &Panels<H, I>,
    up: &Panels<H, I>,
    down: &Down<I, H>,
    input: &Owned<H, R>,
    rows: usize,
    entry: Entry<H, I>,
    next: N,
) -> (Owned<H, R>, N::Output) {
    const {
        assert!(H::SIZE == 4096 && I::SIZE == 14336 && R::SIZE == 512);
    }
    let mut product = Products::new();
    let first = if rows > 16 {
        let (g, u, first) = gate::sweep::<H, I, R>(device, gate, up, down, input, rows, entry);
        product::stage(device, &g, &u, rows, &mut product);
        first
    } else if rows > 8 {
        restored_compact::prepare::<H, I, R, 16>(device, gate, up, down, input, entry, &mut product)
    } else {
        compact::prepare::<H, I, R, 8>(device, gate, up, down, input, entry, &mut product)
    };
    down.forward(device, &product, first, rows, next)
}

/// FP32 panel sums remain in DM; no BF16 rounding occurs between panels.
fn sum_scaled<C: M, S: M, Time: M, Feature: M>(
    device: &mut Device,
    scales: &super::Scales,
    left: &DmTensor<f32, Chip, C, S, m![{ Time }, { Feature }]>,
    right: &DmTensor<f32, Chip, C, S, m![{ Time }, { Feature }]>,
) -> DmTensor<bf16, Chip, C, S, m![{ Time }, { Feature }]> {
    const {
        assert!(Feature::SIZE <= 256 && Feature::SIZE.is_multiple_of(8));
    }
    axes![Term = 2];
    let factor = block::scalar::<C, S>(device, &scales.output);
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(left.view(), right.view())
        .fetch::<m![{ Time }, Term, { Feature } / 8], m![{ Feature } % 8]>()
        .collect::<m![{ Time }, Term, { Feature } / 8], m![{ Feature } % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![{ Time }, { Feature } / 8]>()
        .vector_narrow_split::<m![{ Time }, { Feature } / 4], m![{ Feature } % 4]>()
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factor)
        .vector_widen_concat::<m![{ Time }, { Feature } / 8], m![{ Feature } % 8]>()
        .vector_final()
        .cast::<bf16, m![{ Feature } % 8 # 16]>()
        .commit_trim::<m![{ Feature } % 8]>()
        .commit()
}

/// Add views directly into a caller-owned buffer. No temporary old-sum tensor or
/// copy back into the span is needed; all partials remain FP32.
fn sum_into<C: M, S: M, Time: M, Feature: M, Prefix: M, Input: M, E: M>(
    device: &mut Device,
    left: DmTensorView<'_, f32, Chip, C, S, Input>,
    right: DmTensorView<'_, f32, Chip, C, S, Input>,
    output: DmTensorViewMut<'_, f32, Chip, C, S, E>,
    live: usize,
) {
    axes![Side = 2];
    Add::stream::<
        f32,
        m![{ Time }, Side, { Feature } / 8],
        m![{ Time }, { Feature } / 8],
        m![{ Feature } % 8],
        Side,
        _,
        _,
        _,
        _,
    >(device, (left, right))
    .commit_trim::<m![{ Feature } % 8]>()
    .commit_prefix::<Prefix, _, _>(output, live);
}

#[cfg(test)]
mod tests {
    use crate::ops::linear::nvfp4::tiled::Blocks;
    use crate::ops::linear::nvfp4::tiled::tests as fixtures;
    use crate::tensor::form::BlockScaled;

    use crate::{
        models::decoder::attention::llama::nvfp4::{Hidden, Intermediate, QWidth},
        ops::linear::nvfp4::tiled::{mlp, panels::Panels, qkv::QkvProjection},
        tensor::scalar::Value as Count,
    };
    use furiosa_opt_std::{prelude::*, runtime::launch};

    type Rows = crate::tensor::axis::Rows<512>;
    const DENSE_FACTORS: [f32; 3] = [0.0019, 0.0013, 0.0037];

    #[device(chip = 1)]
    fn fused(
        device: &mut Device,
        mlp: &mlp::Mlp<Hidden, Intermediate>,
        count: &Count,
        input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
    ) -> HbmTensor<bf16, m![1], m![Rows, Hidden]> {
        let rows = crate::tensor::live::Live::<(), Rows>::read(count).rows();
        let input = input.to_dm(&mut device.tdma);
        let output = mlp::forward::<Hidden, Intermediate, Rows>(
            device,
            &mlp.gate_proj,
            &mlp.up_proj,
            &mlp.down_proj,
            &input,
            rows,
        );
        output.to_hbm(&mut device.tdma)
    }

    /// Exercise the real next-layer prefetch while the down projection is still live.
    #[device(chip = 1)]
    fn transition(
        device: &mut Device,
        mlp: &mlp::Mlp<Hidden, Intermediate>,
        norm: &crate::models::decoder::attention::llama::nvfp4::Norm,
        qkv: &crate::models::decoder::attention::llama::nvfp4::Qkv,
        count: &Count,
        input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
    ) -> HbmTensor<bf16, m![1], m![Rows, QWidth]> {
        use crate::models::decoder::attention::llama::nvfp4::QWidth;
        use crate::schedule::decoder::nvfp4::inspect_transition;
        let rows = crate::tensor::live::Live::<(), Rows>::read(count).rows();
        let input = input.to_dm(&mut device.tdma);
        let (q, _, _) = inspect_transition(device, mlp, norm, qkv, &input, rows);
        let mut output = HbmTensor::<bf16, m![1], m![Rows, QWidth]>::new();
        q.view().to_hbm_view(&mut device.tdma, output.view_mut());
        output
    }

    fn input(row: usize, feature: usize) -> f32 {
        let values = [
            6.0, 0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5,
            -3.5, -5.0,
        ];
        values[(row + feature) % 16] * [0.0, 0.25, 0.5, 1.0, 2.0][(row + feature / 16) % 5]
    }

    fn scalar_code(output: usize) -> f32 {
        let codes = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        codes[output % 16]
    }

    fn scalar_weight(output: usize, feature: usize) -> f32 {
        scalar_code(output) * [0.5, 1.0, 1.5, 2.0][(output + feature / 16) % 4]
    }

    fn up_weight(output: usize, feature: usize) -> f32 {
        scalar_code(output) * [2.0, 1.5, 1.0, 0.5][(output + feature / 16) % 4]
    }

    /// Independent scalar W4A16 oracle, including both BF16 SwiGLU roundings.
    fn reference(row: usize) -> Vec<f32> {
        let hidden = (0..4096).map(|i| input(row, i)).collect::<Vec<_>>();
        let product = (0..14336)
            .map(|out| {
                let feature = out * 97 % 4096;
                let dot = hidden[feature] * scalar_weight(out, feature);
                let gate = bf16::from_f32(dot * 0.5).to_f32();
                let up = bf16::from_f32(hidden[feature] * up_weight(out, feature) * 0.25).to_f32();
                let silu = bf16::from_f32(gate * (1.0 / (1.0 + (-gate).exp()))).to_f32();
                bf16::from_f32(silu * up).to_f32()
            })
            .collect::<Vec<_>>();
        let operand = product;
        (0..4096)
            .map(|out| {
                let sum = (0..7)
                    .map(|term| {
                        let feature =
                            ((out % 16 + 8 * (term % 2)) % 16) * 896 + term * 128 + out % 128;
                        operand[feature] * scalar_weight(out, feature)
                    })
                    .sum::<f32>();
                bf16::from_f32(sum * 0.5).to_f32()
            })
            .collect()
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore = "opens a device"]
    async fn consumes_prefetched_norm_and_qkv_after_mlp() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(fused.topology()).unwrap();
        let (weight, weight_scale, scales) = fixtures::weights::<Hidden, Intermediate, _, _>(
            &mut device,
            Panels::<Hidden, Intermediate>::reorder::<1, _>,
            Panels::<Hidden, Intermediate>::reorder::<16, _>,
        )
        .await;
        let gate = Panels {
            weights: BlockScaled::new(weight, Blocks::new(weight_scale), scales),
        };
        let (weight, weight_scale, mut scales) = fixtures::weights::<Hidden, Intermediate, _, _>(
            &mut device,
            Panels::<Hidden, Intermediate>::reorder::<1, _>,
            |scales| {
                // Independent gate/up block factors expose accidental sharing of paired VRF scales.
                let scales = scales
                    .iter()
                    .map(|scale| f8e4m3::from_f32(2.5 - scale.to_f32()))
                    .collect::<Vec<_>>();
                Panels::<Hidden, Intermediate>::reorder::<16, _>(&scales)
            },
        )
        .await;
        // Distinct weight tensor factors detect swapped gate/up paths.
        scales.output = fixtures::upload::<f32, m![1 # 8]>(&mut device.pdma, vec![0.25; 8]).await;
        let up = Panels {
            weights: BlockScaled::new(weight, Blocks::new(weight_scale), scales),
        };
        let (weight, weight_scale, scales) = fixtures::weights_with::<Intermediate, Hidden, _, _>(
            &mut device,
            mlp::Down::<Intermediate, Hidden>::reorder::<1, _>,
            mlp::Down::<Intermediate, Hidden>::reorder::<16, _>,
            // Select terms from both intermediate-feature clusters for each output.
            |out, term| ((out % 16 + 8 * (term % 2)) % 16) * 896 + term * 128 + out % 128,
        )
        .await;
        let mlp = mlp::Mlp {
            gate_proj: gate,
            up_proj: up,
            down_proj: mlp::Down {
                weights: BlockScaled::new(weight, Blocks::new(weight_scale), scales),
            },
        };
        let expected = (0..512).map(reference).collect::<Vec<_>>();
        {
            use crate::{
                models::decoder::attention::llama::nvfp4::{Norm, QWidth, Qkv},
                tensor::form::Weight,
            };
            let qkv = Qkv {
                query: crate::load::Named(projection(&mut device).await),
                key: crate::load::Named(projection(&mut device).await),
                value: crate::load::Named(projection(&mut device).await),
            };
            let norm = Norm::new(Weight::new(
                fixtures::upload::<bf16, m![Hidden]>(
                    &mut device.pdma,
                    vec![bf16::from_f32(1.0); 4096],
                )
                .await,
            ));
            for rows in [8, 17, 33, 257] {
                let input = fixtures::upload::<bf16, m![Rows, Hidden]>(
                    &mut device.pdma,
                    (0..512 * 4096).map(|at| {
                        bf16::from_f32(if at / 4096 < rows {
                            input(at / 4096, at % 4096)
                        } else {
                            f32::NAN
                        })
                    }),
                )
                .await;
                let count = fixtures::count(&mut device, rows).await;
                let output = launch(transition, (&mut device, &mlp, &norm, &qkv, &count, &input))
                    .await
                    .unwrap()
                    .to_host::<m![Rows, QWidth]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec();
                for row in 0..rows {
                    let scale = (expected[row].iter().map(|x| x * x).sum::<f32>() / 4096.0 + 1e-5)
                        .sqrt()
                        .recip();
                    for out in 0..4096 {
                        let feature = out * 97 % 4096;
                        let normalized = bf16::from_f32(expected[row][feature] * scale).to_f32();
                        let want =
                            bf16::from_f32(normalized * scalar_weight(out, feature) * 0.5).to_f32();
                        let got = output[row * 4096 + out].to_f32();
                        assert!(
                            got.is_finite()
                                && (got - want).abs() <= 0.02_f32.max(want.abs() * 0.015),
                            "prefetched MLP rows={rows} row={row} feature={out} got={got} want={want}"
                        );
                    }
                }
            }
        }
    }
    async fn projection<O: AxisName>(device: &mut Device) -> QkvProjection<Hidden, O> {
        let (weight, weight_scale, scales) = fixtures::weights::<Hidden, O, _, _>(
            device,
            QkvProjection::<Hidden, O>::reorder::<1, _>,
            QkvProjection::<Hidden, O>::reorder::<16, _>,
        )
        .await;
        QkvProjection {
            weights: BlockScaled::new(weight, Blocks::new(weight_scale), scales),
        }
    }

    // Sixteen projection classes let the host evaluate full dense block dots without
    // doing the same work for every output row. The class changes across 112/224-wide
    // gate/up owner boundaries, while input blocks repeat every 64 features.
    fn dense_class(output: usize) -> usize {
        (output + output / 16) % 16
    }

    fn dense_input(row: usize, feature: usize) -> f32 {
        const LANES: [f32; 16] = [
            6.0, -6.0, 0.25, 0.75, 1.25, -1.75, 2.5, -3.5, 5.0, -0.25, -0.75, 1.25, -1.75, 2.5,
            -3.5, 5.0,
        ];
        LANES[feature % 16]
            * [0.5, 1.0, 0.75, 1.25][feature / 16 % 4]
            * [0.25, 0.75, 1.0, -0.5][row % 4]
    }

    fn dense_code(projection: usize, class: usize, feature: usize) -> u8 {
        // Every weight lane is nonzero. Adjacent lanes share a code, so hidden-input
        // lanes 0/1 cancel inside each gate/up block; the remaining pairs keep a nonzero dot.
        const MAGNITUDES: [u8; 8] = [7, 1, 2, 3, 4, 5, 6, 1];
        let pair = feature % 16 / 2;
        let magnitude = MAGNITUDES[(pair + class % 8 + projection) % 8];
        magnitude | ((((pair + class / 8 + feature / 16 + projection) % 2) as u8) << 3)
    }

    fn dense_scale(projection: usize, class: usize, block: usize) -> f32 {
        // Adjacent blocks differ and include non-power-of-two E4M3 scales.
        [0.5, 1.5, 0.75, 2.0][(block + class + projection) % 4]
    }

    /// The oracle decodes canonical codes with a literal E2M1 table, sums all 16
    /// BF16-input products in each block, applies its block scale, and applies the
    /// tensor factor only after the complete projection. It shares no device layout.
    fn dense_dot(projection: usize, class: usize, input: &[f32]) -> f32 {
        const E2M1: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut sum = 0.0_f32;
        for (block, values) in input.chunks_exact(16).enumerate() {
            let mut dot = 0.0_f32;
            for (lane, value) in values.iter().enumerate() {
                dot += value * E2M1[dense_code(projection, class, block * 16 + lane) as usize];
            }
            sum += dot * dense_scale(projection, class, block);
        }
        bf16::from_f32(sum * DENSE_FACTORS[projection]).to_f32()
    }

    fn dense_reference(row: usize) -> [f32; 16] {
        let input = (0..4096)
            .map(|feature| dense_input(row, feature))
            .collect::<Vec<_>>();
        let products: [f32; 16] = std::array::from_fn(|class| {
            let gate = dense_dot(0, class, &input);
            let up = dense_dot(1, class, &input);
            let silu = bf16::from_f32(gate * (1.0 / (1.0 + (-gate).exp()))).to_f32();
            bf16::from_f32(silu * up).to_f32()
        });
        let product = (0..14336)
            .map(|feature| products[dense_class(feature)])
            .collect::<Vec<_>>();
        std::array::from_fn(|class| dense_dot(2, class, &product))
    }

    async fn dense_weights<I: AxisName, O: AxisName, W: M, B: M>(
        device: &mut Device,
        projection: usize,
        reorder_weight: impl FnOnce(&[f4e2m1]) -> Vec<f4e2m1>,
        reorder_scale: impl FnOnce(&[f8e4m3]) -> Vec<f8e4m3>,
    ) -> (
        HbmTensor<f4e2m1, m![1], W>,
        HbmTensor<f8e4m3, m![1], B>,
        crate::ops::linear::nvfp4::tiled::Scales,
    ) {
        let patterns: [[f4e2m1; 64]; 16] = std::array::from_fn(|class| {
            std::array::from_fn(|feature| f4e2m1::from_bits(dense_code(projection, class, feature)))
        });
        let mut codes = Vec::with_capacity(O::SIZE * I::SIZE);
        for output in 0..O::SIZE {
            for _ in 0..I::SIZE / 64 {
                codes.extend_from_slice(&patterns[dense_class(output)]);
            }
        }
        let scales = (0..O::SIZE * I::SIZE / 16)
            .map(|at| {
                let output = at / (I::SIZE / 16);
                let block = at % (I::SIZE / 16);
                f8e4m3::from_f32(dense_scale(projection, dense_class(output), block))
            })
            .collect::<Vec<_>>();
        (
            fixtures::upload::<f4e2m1, W>(&mut device.pdma, reorder_weight(&codes)).await,
            fixtures::upload::<f8e4m3, B>(&mut device.pdma, reorder_scale(&scales)).await,
            crate::ops::linear::nvfp4::tiled::Scales {
                output: fixtures::upload::<f32, m![1 # 8]>(
                    &mut device.pdma,
                    vec![DENSE_FACTORS[projection]; 8],
                )
                .await,
            },
        )
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore = "opens a device"]
    async fn fused_mlp_matches_dense_block_dots_with_cancellation_and_poisoned_padding() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(fused.topology()).unwrap();
        let (weight, weight_scale, scales) = dense_weights::<Hidden, Intermediate, _, _>(
            &mut device,
            0,
            Panels::<Hidden, Intermediate>::reorder::<1, _>,
            Panels::<Hidden, Intermediate>::reorder::<16, _>,
        )
        .await;
        let gate_proj = Panels {
            weights: BlockScaled::new(weight, Blocks::new(weight_scale), scales),
        };
        let (weight, weight_scale, scales) = dense_weights::<Hidden, Intermediate, _, _>(
            &mut device,
            1,
            Panels::<Hidden, Intermediate>::reorder::<1, _>,
            Panels::<Hidden, Intermediate>::reorder::<16, _>,
        )
        .await;
        let up_proj = Panels {
            weights: BlockScaled::new(weight, Blocks::new(weight_scale), scales),
        };
        let (weight, weight_scale, scales) = dense_weights::<Intermediate, Hidden, _, _>(
            &mut device,
            2,
            mlp::Down::<Intermediate, Hidden>::reorder::<1, _>,
            mlp::Down::<Intermediate, Hidden>::reorder::<16, _>,
        )
        .await;
        let mlp = mlp::Mlp {
            gate_proj,
            up_proj,
            down_proj: mlp::Down {
                weights: BlockScaled::new(weight, Blocks::new(weight_scale), scales),
            },
        };
        let expected: [[f32; 16]; 4] = std::array::from_fn(dense_reference);
        // Cover both hybrid cutovers and the large-path wave boundary with dense blocks.
        for rows in [1, 8, 9, 16, 17, 64, 65, 128, 129, 256, 257, 512] {
            let input = fixtures::upload::<bf16, m![Rows, Hidden]>(
                &mut device.pdma,
                (0..512 * 4096).map(|at| {
                    bf16::from_f32(if at / 4096 < rows {
                        dense_input(at / 4096, at % 4096)
                    } else {
                        f32::NAN
                    })
                }),
            )
            .await;
            let count = fixtures::count(&mut device, rows).await;
            let output = launch(fused, (&mut device, &mlp, &count, &input))
                .await
                .unwrap()
                .to_host::<m![Rows, Hidden]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (at, found) in output.iter().take(rows * 4096).enumerate() {
                let found = found.to_f32();
                let want = expected[at / 4096 % 4][dense_class(at % 4096)];
                let error = (found - want).abs();
                // Match the existing nonlinear oracle tolerance: BF16 SiLU/product
                // roundings precede a dense down dot and its final BF16 conversion.
                assert!(
                    found.is_finite() && error <= 0.02_f32.max(want.abs() * 0.015),
                    "dense MLP rows={rows} row={} feature={} found={found} expected={want}",
                    at / 4096,
                    at % 4096
                );
            }
        }
    }
}
