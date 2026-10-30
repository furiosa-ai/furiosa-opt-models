//! Per-expert matrices selected by device-resident routing indices.
//!
//! - A chip holds `COUNT` whole experts, the router's experts `first..first + COUNT`; a selection
//!   of another chip's expert yields a row of -0, which adds to any value without changing it, so
//!   the chips' rows add up to its owner's row.
//! - A selected row projects through its expert's matrix in FP32 and rounds once to BF16.

use crate::Op;
use crate::ops::linear::{Affine, Linear};
use crate::ops::normalization::{Chunk, FeatureSlices};
use crate::tensor::axis::SixtyFour;
use crate::tensor::mapping::BroadcastClusters as Cluster;
use crate::tensor::parallel::Chip;
use crate::tensor::scalar;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Each row's selected experts, best first, each in the first word of an eight-byte pair.
pub(crate) type Indices<R, K, C = Chip> = HbmTensor<i32, C, m![R, K, 1 # 2]>;

/// Each row's `Width` features for each of its `K` selected experts.
pub(crate) type Rows<R, K, Width, C = Chip> = HbmTensor<bf16, C, m![R, K, Width]>;

/// One value a row and selection, in the first word of an eight-byte pair.
type Slots<D, R, K, C> = HbmTensor<D, C, m![R, K, 1 # 2]>;

/// The rows a projection writes at once beside its resident weight; a selected row takes the first.
type Band = SixtyFour;

/// `output[r][k] = weights[indices[r][k] - first] @ input[r]`, or `@ input[r][k]`; -0 for an
/// index outside `first..first + COUNT`.
#[derive(DeviceSend)]
pub struct Experts<In: AxisName, Out: AxisName, const COUNT: usize, C: M = Chip> {
    /// Expert matrices in [Out, In] order.
    pub weights: [HbmTensor<bf16, C, m![Out, In]>; COUNT],
    /// The router's index of the first expert, in the first lane of eight.
    pub first: HbmTensor<i32, C, m![1 # 8]>,
}

impl<I: AxisName, Out: AxisName, const N: usize, C: M> Experts<I, Out, N, C> {
    /// Each selection's expert among `weights`, clamped into them for another chip's expert, and
    /// its mask: all ones for an expert of `weights`, zero for the others.
    fn locate<R: AxisName, K: AxisName>(
        &self,
        device: &mut Device,
        indices: &Indices<R, K, C>,
    ) -> (Slots<i32, R, K, C>, Slots<i32, R, K, C>) {
        use BitReq::{Ignore, Zero};
        type Each<D, R, K, C> = DmTensor<D, C, Cluster, m![1 # 256], m![R, K, 1 # 2]>;
        let first: DmTensor<i32, C, Cluster, m![1 # 256], m![1 # 8]> =
            self.first.to_dm(&mut device.tdma);
        let first = scalar::register(device, &first);
        let indices: Each<i32, R, K, C> = indices.to_dm(&mut device.tdma);
        let local: Each<i32, R, K, C> = device
            .main
            .begin(indices.view())
            .fetch::<m![R, K], m![1 # 2]>()
            .collect::<m![R, K], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp, &first)
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit();
        let index: Each<i32, R, K, C> = device
            .main
            .begin(local.view())
            .fetch::<m![R, K], m![1 # 2]>()
            .collect::<m![R, K], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpI32::Max, 0)
            .vector_clip(ClipBinaryOpI32::Min, N as i32 - 1)
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit();
        let mask: Each<i32, R, K, C> = device
            .main
            .begin(local.view())
            .fetch::<m![R, K], m![1 # 2]>()
            .collect::<m![R, K], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Less(0),
                Cmp::Greater(N as i32 - 1),
                Cmp::False,
                Cmp::False,
            ]))
            .vector_logic(LogicBinaryOpI32::BitAnd, 0)
            .vector_logic(
                LogicBinaryOpI32::BitOr,
                Branched::imm(TagGuard::matches([Zero, Zero, Ignore, Ignore]), -1),
            )
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit();
        (
            index.to_hbm(&mut device.tdma),
            mask.to_hbm(&mut device.tdma),
        )
    }

    /// Each selection's row, as `row(r, k)` gives it, through its expert into `output[r][k]`.
    fn evaluate<'x, R: AxisName, K: AxisName, E: M>(
        &self,
        device: &mut Device,
        indices: &Indices<R, K, C>,
        row: impl Fn(usize, usize) -> HbmTensorView<'x, bf16, C, E>,
    ) -> Rows<R, K, Out, C> {
        type Cs<W> = Symbol<Chunk<W>>;
        const { assert!(N > 0 && K::SIZE > 0) };
        let (index, mask) = self.locate(device, indices);
        let mut output = Rows::<R, K, Out, C>::new();
        let mut band: HbmTensor<bf16, C, m![Band, I]> = HbmTensor::new();
        for r in 0..R::SIZE {
            for k in 0..K::SIZE {
                row(r, k).to_hbm_view(
                    &mut device.tdma,
                    band.view_mut()
                        .tile::<m![Band], 1, m![Band = 1 #{!} Band, I]>(0),
                );
                let mut at: HbmTensor<i32, C, m![1]> = HbmTensor::new();
                index
                    .view()
                    .tile::<m![R], 1, m![R = 1 # R, K, 1 # 2]>(r)
                    .tile::<m![K], 1, m![R = 1 # R, K = 1 # K, 1 # 2]>(k)
                    .to_hbm_view(&mut device.tdma, at.view_mut());
                let expert = at.view().to_spm(&mut device.tdma) as usize;
                let linear = Linear::<I, Out, C, &HbmTensor<bf16, C, m![Out, I]>> {
                    affine: Affine::new(&self.weights[expert], ()),
                    shape: PhantomData,
                };
                let mut sums: HbmTensor<f32, C, m![Band, Out]> = HbmTensor::new();
                linear.forward(&mut *device, (&band, &mut sums));
                let mut first: HbmTensor<f32, C, m![Out]> = HbmTensor::new();
                sums.view()
                    .tile::<m![Band], 1, m![Band = 1 # Band, Out]>(0)
                    .to_hbm_view(&mut device.tdma, first.view_mut());
                let mut keep: HbmTensor<i32, C, m![1 # 2]> = HbmTensor::new();
                mask.view()
                    .tile::<m![R], 1, m![R = 1 # R, K, 1 # 2]>(r)
                    .tile::<m![K], 1, m![R = 1 # R, K = 1 # K, 1 # 2]>(k)
                    .to_hbm_view(&mut device.tdma, keep.view_mut());
                let keep: DmTensor<i32, C, Cluster, m![256], m![1 # 2]> =
                    keep.to_dm(&mut device.tdma);
                // SAFETY: every slice holds the mask; only the slice label changes.
                let keep: DmTensor<i32, C, Cluster, FeatureSlices<Out>, m![1 # 2]> =
                    unsafe { keep.reshape() };
                // A collect from one word fills every lane of the packet with it.
                let keep: VrfTensor<i32, C, Cluster, FeatureSlices<Out>, m![1 # 8]> = device
                    .sub
                    .begin(keep.view())
                    .fetch::<m![1], m![1 # 2]>()
                    .collect::<m![1], m![1 # 8]>()
                    .to_vrf();
                let first: DmTensor<f32, C, Cluster, FeatureSlices<Out>, m![Out % ({ Cs<Out> })]> =
                    first.to_dm(&mut device.tdma);
                // The mask keeps the sum or clears it between two sign flips, so a cleared row is
                // -0 whatever the stand-in matrix made, Inf and NaN included.
                let rounded: DmTensor<
                    bf16,
                    C,
                    Cluster,
                    FeatureSlices<Out>,
                    m![Out % ({ Cs<Out> })],
                > = device
                    .main
                    .begin(first.view())
                    .fetch::<m![Out % ({ Cs<Out> }) / 8], m![Out % 8]>()
                    .collect::<m![Out % ({ Cs<Out> }) / 8], m![Out % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_reinterpret::<i32>()
                    .vector_logic(LogicBinaryOpI32::BitXor, i32::MIN)
                    .vector_logic(LogicBinaryOpI32::BitAnd, &keep)
                    .vector_reinterpret::<f32>()
                    .vector_narrow_split::<m![Out % ({ Cs<Out> }) / 4], m![Out % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -1_f32)
                    .vector_widen_concat::<m![Out % ({ Cs<Out> }) / 8], m![Out % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Out % 8 # 16]>()
                    .commit_trim::<m![Out % 8]>()
                    .commit();
                rounded.view().to_hbm_view(
                    &mut device.tdma,
                    output
                        .view_mut()
                        .tile::<m![R], 1, m![R = 1 #{!} R, K, Out]>(r)
                        .tile::<m![K], 1, m![R = 1 #{!} R, K = 1 #{!} K, Out]>(k),
                );
            }
        }
        output
    }
}

impl<I: AxisName, Out: AxisName, R: AxisName, K: AxisName, const N: usize, C: M>
    Op<(&HbmTensor<bf16, C, m![R, I]>, &Indices<R, K, C>)> for Experts<I, Out, N, C>
{
    type Output = Rows<R, K, Out, C>;

    fn forward(
        &self,
        device: &mut Device,
        (input, indices): (&HbmTensor<bf16, C, m![R, I]>, &Indices<R, K, C>),
    ) -> Self::Output {
        self.evaluate(device, indices, |r, _| {
            input.view().tile::<m![R], 1, m![R = 1 # R, I]>(r)
        })
    }
}

impl<I: AxisName, Out: AxisName, R: AxisName, K: AxisName, const N: usize, C: M>
    Op<(&Rows<R, K, I, C>, &Indices<R, K, C>)> for Experts<I, Out, N, C>
{
    type Output = Rows<R, K, Out, C>;

    fn forward(
        &self,
        device: &mut Device,
        (input, indices): (&Rows<R, K, I, C>, &Indices<R, K, C>),
    ) -> Self::Output {
        self.evaluate(device, indices, |r, k| {
            input
                .view()
                .tile::<m![R], 1, m![R = 1 # R, K, I]>(r)
                .tile::<m![K], 1, m![R = 1 # R, K = 1 # K, I]>(k)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::normalization::tests::random;
    use crate::tensor::axis::Eight;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![In = 256, Out = 128, Two = 2];

    /// Eight experts on this chip, the router's experts `FIRST..FIRST + 8` of sixteen.
    const FIRST: i32 = 4;

    /// Each row's two selections: one of this chip's experts and one of another chip's, never this
    /// chip's first or last expert, the stand-ins another chip's expert clamps to.
    const SELECTED: [i32; 16] = [5, 0, 6, 12, 7, 1, 8, 13, 9, 2, 10, 14, 6, 3, 9, 15];

    /// 2^127, the stand-ins' first column: times a row's first feature, 4, it overflows FP32.
    const HUGE: f32 = 1.7014118e38;

    #[device(chip = 1)]
    fn project(
        device: &mut Device,
        experts: &Experts<In, Out, 8>,
        input: &HbmTensor<bf16, Chip, m![Eight, In]>,
        indices: &Indices<Eight, Two>,
    ) -> Rows<Eight, Two, Out> {
        experts.forward(device, (input, indices))
    }

    /// A selection of one of this chip's experts is its matrix's product with the row, rounded
    /// once; a selection of another chip's expert is a row of -0, though its stand-in's FP32
    /// product overflows.
    #[tokio::test(flavor = "current_thread")]
    async fn projects_each_selection_through_its_expert() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(project.topology()).unwrap();
        let matrices: Vec<Vec<bf16>> = (0..8)
            .map(|expert| {
                let mut matrix = random(Out::SIZE * In::SIZE, 0x1234 + expert);
                if expert == 0 || expert == 7 {
                    matrix
                        .iter_mut()
                        .step_by(In::SIZE)
                        .for_each(|v| *v = bf16::from_f32(HUGE));
                }
                matrix
            })
            .collect();
        let mut weights = Vec::new();
        for matrix in &matrices {
            let matrix = HostTensor::<bf16, m![Out, In]>::from_vec(matrix.clone());
            weights.push(matrix.to_hbm(&mut device.pdma).await.unwrap());
        }
        let experts = Experts::<In, Out, 8> {
            weights: weights.try_into().unwrap(),
            first: HostTensor::<i32, m![1]>::from_vec(vec![FIRST])
                .to_hbm::<Chip, m![1 # 8]>(&mut device.pdma)
                .await
                .unwrap(),
        };
        let mut rows = random(Eight::SIZE * In::SIZE, 0x777);
        rows.iter_mut()
            .step_by(In::SIZE)
            .for_each(|v| *v = bf16::from_f32(4.0));
        let product = |matrix: &[bf16], row: usize, o: usize| -> f64 {
            (0..In::SIZE)
                .map(|i| {
                    f64::from(matrix[o * In::SIZE + i].to_f32())
                        * f64::from(rows[row * In::SIZE + i].to_f32())
                })
                .sum()
        };
        let overflows = |e: usize| {
            (0..Eight::SIZE).all(|row| product(&matrices[e], row, 0) > f64::from(f32::MAX))
        };
        assert!(
            overflows(0) && overflows(7),
            "each stand-in's product overflows FP32"
        );
        let input = HostTensor::<bf16, m![Eight, In]>::from_vec(rows.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let indices = HostTensor::<i32, m![Eight, Two]>::from_vec(SELECTED)
            .to_hbm::<Chip, m![Eight, Two, 1 # 2]>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(project, (&mut device, &experts, &input, &indices))
            .await
            .unwrap()
            .to_host::<m![Eight, Two, Out]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, &index) in SELECTED.iter().enumerate() {
            let (row, local) = (at / Two::SIZE, index - FIRST);
            let got = &output[at * Out::SIZE..][..Out::SIZE];
            let Some(matrix) = usize::try_from(local)
                .ok()
                .and_then(|local| matrices.get(local))
            else {
                let negative = bf16::from_f32(-0.0).to_bits();
                let wrong = got.iter().find(|v| v.to_bits() != negative);
                assert!(wrong.is_none(), "selection {at}: {wrong:?} is not -0");
                continue;
            };
            for (o, got) in got.iter().enumerate() {
                // The FP32 sum's order differs from the f64 one's: allow its rounding a step.
                let want = bf16::from_f32(product(matrix, row, o) as f32).to_f32();
                let step = (want.abs() * 2f32.powi(-7)).max(f32::MIN_POSITIVE);
                assert!(
                    (got.to_f32() - want).abs() <= step,
                    "selection {at} output {o}"
                );
            }
        }
    }
}
