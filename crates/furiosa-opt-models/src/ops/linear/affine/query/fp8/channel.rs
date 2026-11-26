//! Native channel-scaled FP8 projections, with token scales retained in FP32.
//! Q, K and V share one activation TRF operand; owner names on their resident weights change,
//! while the physical four-output-owner by 64-input-slice mapping stays the same.

use super::activation::Copies;
use super::*;
use crate::ops::linear::query::group::TilePair;
use crate::ops::vector::Quantization;
use crate::tensor::form::fp8::{Channel, Rows as StoredRows};

pub type Weights<Hidden, Heads, Head> = StoredRows<
    m![Hidden],
    m![Heads, Head],
    Stored<Hidden, Heads, Head>,
    HeadScales<Heads, Head, bf16>,
    Channel,
>;

pub(crate) struct Prepared<Hidden: AxisName, Heads: AxisName, Head: AxisName> {
    weight: Weight<Hidden, Heads, Head>,
    scale: ValueScale<Hidden, Heads, Head, bf16>,
}

axes![Owner = 8];
type Cl = m![Owner / 4];
type Slices<Hidden> = m![Owner % 4, Hidden / 64];
type Operand<Hidden, G> =
    TrfTensor<f8e4m3, Chip, Cl, Slices<Hidden>, m![Tile % 8], m![G, Tile / 8, Hidden % 64]>;
type Scales<G> = DmTensor<f32, Chip, m![2], m![4, Copies], m![G, Tile]>;
type Quantizer<Hidden, G> = Quantization<
    m![G, Tile, Hidden % 64 / 8],
    m![G, Tile, Hidden % 64 / 4],
    Hidden,
    m![G, Tile],
    m![4, Copies],
    m![G, Tile / 4],
    m![Tile % 4],
>;
type ResultSlices<G, Spread> = m![Owner % 4, G, Tile / 8, { Spread }];
type RowScale<G> = DmTensor<f32, Chip, Cl, ResultSlices<G, <G as Group>::Copies>, m![Tile % 8]>;

/// A quantized period's rows and scales. Only the live groups of a single wave are initialized;
/// a paired contraction retains its static paired coverage. Consumers ignore padded rows.
pub(crate) struct Period<Hidden: AxisName, G: Group> {
    input: Input<Hidden, G>,
    scale: RowScale<G>,
    groups: usize,
}

impl<Hidden: AxisName, G: Group> Period<Hidden, G> {
    pub(crate) fn quantize(
        device: &mut Device,
        input: &activation::InDm<Hidden, G>,
        groups: usize,
    ) -> Self {
        let mut scale = Scales::<G>::new();
        let maxima = Quantizer::<Hidden, G>::scales_f32(&mut device.main, input.view());
        if const { G::SIZE == 1 } {
            maxima.commit_prefix::<m![Tile / 8], _, _>(scale.view_mut(), groups);
        } else {
            maxima.commit_view(scale.view_mut());
        }
        // SAFETY: every input-feature slice holds the same complete-token maximum.
        let broadcast = unsafe {
            scale
                .view()
                .reshape::<Chip, m![2], m![4, Hidden / 64], m![G, Tile]>()
        };
        let factors: VrfTensor<f32, Chip, m![2], m![4, Hidden / 64], m![G, Tile]> = device
            .sub
            .begin(broadcast)
            .fetch::<m![G, Tile / 8], m![Tile % 8]>()
            .collect::<m![G, Tile / 8], m![Tile % 8]>()
            .to_vrf();
        let mut codes = Input::<Hidden, G>::new();
        let coded = Quantizer::<Hidden, G>::rows(&mut device.main, input.view(), &factors);
        if const { G::SIZE == 1 } {
            coded.commit_prefix::<m![Tile / 8], _, _>(codes.view_mut(), groups);
        } else {
            coded.commit_view(codes.view_mut());
        }
        let mut rows = RowScale::<G>::new();
        scale
            .view()
            .slice_tile::<m![Copies], 1, m![4, Copies = 1 # 64]>(0)
            .to_dm_view(&mut device.tdma, rows.view_mut());
        Self {
            input: codes,
            scale: rows,
            groups,
        }
    }

    /// Fill once, then borrow the same operand for each of the three projections.
    pub(crate) fn fill(&self, device: &mut Device) -> Operand<Hidden, G> {
        // SAFETY: all eight output owners receive the same activation, with identical input strips.
        let input = unsafe {
            self.input
                .view()
                .reshape::<Chip, Cl, Slices<Hidden>, m![G, Tile, Hidden % 64]>()
        };
        let stream = device
            .sub
            .begin(input)
            .fetch::<m![Tile % 8, G, Tile / 8, Hidden % 64 / 32], m![Hidden % 32]>()
            .collect::<m![Tile % 8, G, Tile / 8, Hidden % 64 / 32], m![Hidden % 32]>();
        if const { G::SIZE == 1 } {
            stream.to_trf_prefix::<m![G, Tile / 8], _, _>(self.groups)
        } else {
            stream.to_trf()
        }
    }

    pub(crate) fn project<Heads: AxisName, Head: AxisName, Spread: M, Local: M>(
        &self,
        device: &mut Device,
        projection: &Prepared<Hidden, Heads, Head>,
        operand: &Operand<Hidden, G>,
    ) -> Output<G, Heads, Head, Spread, Local> {
        self.project_with_prefetch(device, projection, operand, |_| {})
    }

    /// Queue independent TDMA work after the channel scales land on the result slices.
    pub(crate) fn project_with_prefetch<Heads: AxisName, Head: AxisName, Spread: M, Local: M>(
        &self,
        device: &mut Device,
        projection: &Prepared<Hidden, Heads, Head>,
        operand: &Operand<Hidden, G>,
        prefetch: impl FnOnce(&mut Device),
    ) -> Output<G, Heads, Head, Spread, Local> {
        let scale = self.channels(device, projection);
        prefetch(device);
        self.project_ready(device, projection, operand, scale)
    }

    pub(crate) fn channels<Heads: AxisName, Head: AxisName, Spread: M, Local: M>(
        &self,
        device: &mut Device,
        projection: &Prepared<Hidden, Heads, Head>,
    ) -> DmTensor<bf16, Chip, Cluster<Heads>, Reduced<G, Heads, Spread>, m![{ Local }, Head]> {
        let mut scale = DmTensor::new();
        projection
            .scale
            .view()
            .slice_tile::<m![Hidden / 64], 1, m![
                Heads % (Heads / 2) / (Heads / 4),
                Heads % (Heads / 4) / (Heads / 8), Hidden / 64 = 1 # 64
            ]>(0)
            .to_dm_view(&mut device.tdma, scale.view_mut());
        scale
    }

    pub(crate) fn project_ready<Heads: AxisName, Head: AxisName, Spread: M, Local: M>(
        &self,
        device: &mut Device,
        projection: &Prepared<Hidden, Heads, Head>,
        operand: &Operand<Hidden, G>,
        scale: DmTensor<bf16, Chip, Cluster<Heads>, Reduced<G, Heads, Spread>, m![{ Local }, Head]>,
    ) -> Output<G, Heads, Head, Spread, Local> {
        // SAFETY: the two head-owner digits name the same four physical owners for Q, K and V.
        // Only their labels change; input features and the local output order are untouched.
        let weight = unsafe {
            projection.weight.view().reshape::<Chip, Cl, Slices<Hidden>,
            m![Head / 64, Heads % (Heads / 8), Head % 64, Hidden % 64]>()
        };
        let channels = unsafe {
            scale
                .view()
                .reshape::<Chip, Cl, ResultSlices<G, Spread>, m![{ Local }, Head]>()
        };
        let channel: VrfTensor<f32, Chip, Cl, ResultSlices<G, Spread>, m![{ Local }, Head]> =
            device
                .sub
                .begin(channels)
                .fetch::<m![{ Local }, Head / 8], m![Head % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Local }, Head / 8], m![Head % 8]>()
                .to_vrf();
        // SAFETY: the row digits occupy identical physical positions for Q, K and V; the
        // remaining slices duplicate the token scale, irrespective of their local head names.
        let rows = unsafe {
            self.scale
                .view()
                .reshape::<Chip, Cl, ResultSlices<G, Spread>, m![Tile % 8]>()
        };
        let row: VrfTensor<f32, Chip, Cl, ResultSlices<G, Spread>, m![Tile % 8]> = device
            .sub
            .begin(rows)
            .fetch::<m![1], m![Tile % 8]>()
            .collect::<m![1], m![Tile % 8]>()
            .to_vrf();
        let mut output: DmTensor<
            bf16,
            Chip,
            Cl,
            ResultSlices<G, Spread>,
            m![{ Local }, Tile % 8, Head],
        > = DmTensor::new();
        let products = device.main.begin(weight)
            .fetch::<Outputs<Heads, Head>, m![Hidden % 64]>()
            .collect::<m![{ Outputs<Heads, Head> }, Hidden % 64 / 32], m![Hidden % 32]>()
            .contract_outer::<m![{ Outputs<Heads, Head> }, G, Tile / 8], m![Hidden % 64], _, _, _>(operand)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Outputs<Heads, Head> }, G, Tile / 8]>()
            .contract_lane::<m![{ Outputs<Heads, Head> }, G, Tile / 8], m![Tile % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<ResultSlices<G, Spread>, m![{ Local }, Head / 64, Head % 64]>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Local }, Head / 64, Head % 64, Tile % 8 / 4], m![Tile % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &row)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &channel)
            .vector_widen_concat::<m![{ Local }, Head / 64, Head % 64], m![Tile % 8]>()
            .vector_final()
            .cast::<bf16, m![Tile % 8 # 16]>()
            .transpose::<m![{ Local }, Head / 64, Head % 64 / 4, Tile % 8], m![Head % 4 # 16]>()
            .commit_trim::<m![Head % 4]>();
        if const { G::SIZE == 1 } {
            products.commit_prefix::<m![Tile / 8], _, _>(output.view_mut(), self.groups);
        } else {
            products.commit_view(output.view_mut());
        }
        // SAFETY: the consumer's Rows names the same physical Reduced positions, as in Qwen.
        unsafe { output.reshape() }
    }
}

/// The span's pairs are quantized once in the existing normalization-owner exchange and staged
/// once. A caller copies the next pair before consuming the current one's shared TRF operand.
pub(crate) struct Staged<Hidden: AxisName, All: AxisName> {
    rows: spread::Staged<Hidden, All>,
    scales: spread::StagedScales<All, f32>,
}

impl<Hidden: AxisName, All: AxisName> Staged<Hidden, All> {
    pub(crate) fn new(
        device: &mut Device,
        normalized: &Normalized<Hidden, All>,
        end: usize,
    ) -> Self {
        let input: spread::Spread<Hidden, All> = normalized.to_dm(&mut device.tdma);
        let mut scales = spread::SpreadScales::<All, f32>::new();
        spread::Rows::<Hidden, All>::scales_f32(&mut device.main, input.view())
            .commit_prefix::<m![All / 64], _, _>(scales.view_mut(), end.div_ceil(64));
        // SAFETY: the maximum reduction copies the complete row scale to every feature slice.
        let broadcast = unsafe {
            scales.view().reshape::<Chip, m![All % 64 / 32],
            m![All % 32 / 8, Hidden / 64], m![All / 64, All % 8]>()
        };
        let factor: VrfTensor<
            f32,
            Chip,
            m![All % 64 / 32],
            m![All % 32 / 8, Hidden / 64],
            m![All / 64, All % 8],
        > = device
            .sub
            .begin(broadcast)
            .fetch::<m![All / 64], m![All % 8]>()
            .collect::<m![All / 64], m![All % 8]>()
            .to_vrf();
        let mut rows = spread::SpreadInput::<Hidden, All>::new();
        spread::Rows::<Hidden, All>::rows(&mut device.main, input.view(), &factor)
            .commit_prefix::<m![All / 64], _, _>(rows.view_mut(), end.div_ceil(64));
        let (rows, scales) = Stage.forward(device, (&rows, &scales));
        Self { rows, scales }
    }

    pub(crate) fn pair(&self, device: &mut Device, at: usize) -> Period<Hidden, TilePair> {
        let input = PairRows.forward(device, (&self.rows, at));
        let rows = PairScales::<Owner>(PhantomData).forward(device, (&self.scales, at));
        // SAFETY: combine the two adjacent owner bits; no row or scale changes physical position.
        let scale = unsafe { rows.reshape() };
        Period {
            input,
            scale,
            groups: 32,
        }
    }
}

pub(crate) fn prepare<Hidden: AxisName, Heads: AxisName, Head: AxisName>(
    device: &mut Device,
    projection: &Weights<Hidden, Heads, Head>,
) -> Prepared<Hidden, Heads, Head> {
    prepare_weight::<_, _, _, false>(device, projection, false)
}

/// A split query can defer its first half until the down exchange on single-row decode.
pub(crate) fn prepare_weight<
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
    const SPLIT: bool,
>(
    device: &mut Device,
    projection: &Weights<Hidden, Heads, Head>,
    defer_first: bool,
) -> Prepared<Hidden, Heads, Head> {
    let (mut weight, mut scale) = (DmTensor::new(), DmTensor::new());
    if const { SPLIT } {
        match defer_first {
            false => source::Half::<_, _, _, 0>(&projection.0).forward(device, &mut weight),
            true => (),
        }
    } else {
        source::Panel(&projection.0).forward(device, &mut weight);
    }
    source::Scales {
        stored: &projection.1,
        hidden: PhantomData,
    }
    .forward(device, &mut scale);
    Prepared { weight, scale }
}

pub(crate) fn load_query_half<
    const HALF: usize,
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
>(
    device: &mut Device,
    projection: &Weights<Hidden, Heads, Head>,
    query: &mut Prepared<Hidden, Heads, Head>,
) {
    source::Half::<_, _, _, HALF>(&projection.0).forward(device, &mut query.weight);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::linear::query::group::{TilePair, Tiles};
    use crate::tensor::axis;
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![H = 4096, Q = 32, K = 8, D = 128];
    type All = axis::Rows<256>;
    type Single = Tiles<1>;
    type Source = HbmTensor<bf16, Chip, m![All, H]>;
    type Query = HbmTensor<bf16, Chip, m![All, Q, D]>;
    type Key = HbmTensor<bf16, Chip, m![All, K, D]>;
    type Projections = (Weights<H, Q, D>, Weights<H, K, D>, Weights<H, K, D>);

    fn run<G: Group>(
        device: &mut Device,
        weights: &Projections,
        input: activation::InDm<H, G>,
        groups: usize,
        output: &mut (Query, Key, Key),
    ) {
        let q = prepare(device, &weights.0);
        let k = prepare(device, &weights.1);
        let v = prepare(device, &weights.2);
        let period = Period::quantize(device, &input, groups);
        let operand = period.fill(device);
        let query: Queries<G, Q, D> = period.project(device, &q, &operand);
        let key: Pairs<G, K, D> = period.project(device, &k, &operand);
        let value: Pairs<G, K, D> = period.project(device, &v, &operand);
        fn store<G: Group, Heads: AxisName, Spread: M, Local: M>(
            device: &mut Device,
            rows: &Output<G, Heads, D, Spread, Local>,
            into: &mut HbmTensor<bf16, Chip, m![All, Heads, D]>,
        ) {
            // SAFETY: restore the contraction's physical Reduced coordinates for the HBM store.
            let rows = unsafe {
                rows.view().reshape::<Chip, Cluster<Heads>, Reduced<G, Heads, Spread>, m![{Local}, Tile % 8, D]>()
            };
            rows.to_hbm_view(&mut device.tdma, G::window(into.view_mut(), 0));
        }
        store(device, &query, &mut output.0);
        store(device, &key, &mut output.1);
        store(device, &value, &mut output.2);
    }

    #[device(chip = 1)]
    fn projected(
        device: &mut Device,
        weights: &Projections,
        input: &Source,
        rows: &Value,
    ) -> (Query, Key, Key) {
        let live = crate::tensor::live::Live::<(), All>::read(rows).rows();
        let mut output = (Query::new(), Key::new(), Key::new());
        if live <= 128 {
            let rows = input
                .view()
                .tile::<m![All / 128], 1, m![All / 128 = 1 # 2, All % 128, H]>(0);
            // SAFETY: the first 128 consecutive rows are one wave.
            let rows = unsafe { rows.reshape::<Chip, m![Single # 2, Tile, H]>() };
            let input = rows.to_dm(&mut device.tdma);
            run::<Single>(device, weights, input, live.div_ceil(8), &mut output);
        } else {
            // SAFETY: the two consecutive 128-row waves constitute one pair.
            let rows = unsafe { input.view().reshape::<Chip, m![TilePair, Tile, H]>() };
            let input = rows.to_dm(&mut device.tdma);
            run::<TilePair>(device, weights, input, 32, &mut output);
        }
        output
    }

    /// Independent scales within each Q/K/V head, a maximum in the last input slice, and NaNs
    /// outside the live prefix exercise the single-wave trims and the static paired geometry.
    #[tokio::test(flavor = "current_thread")]
    async fn shared_operand_preserves_channel_scales_and_live_rows() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(projected.topology()).unwrap();
        fn code(out: usize, input: usize) -> f32 {
            ((out % 7 + input % 11) % 9) as f32 / 8.0
        }
        fn scale(out: usize, projection: usize) -> f32 {
            (out % 13 + projection + 1) as f32 / 512.0
        }
        async fn upload<Heads: AxisName>(
            device: &mut Device,
            projection: usize,
        ) -> Weights<H, Heads, D> {
            let codes: Vec<_> = (0..Heads::SIZE * D::SIZE * H::SIZE)
                .map(|at| f8e4m3::from_f32(code(at / H::SIZE, at % H::SIZE)))
                .collect();
            let codes = HostTensor::<f8e4m3, m![Heads, D, H]>::from_vec(codes)
                .to_hbm::<Chip, Layout<H, Heads, D>>(&mut device.pdma)
                .await
                .unwrap();
            let scales = HostTensor::<bf16, m![Heads, D]>::from_vec(
                (0..Heads::SIZE * D::SIZE).map(|out| bf16::from_f32(scale(out, projection))),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            Weights::new(codes, scales)
        }
        let weights = (
            upload::<Q>(&mut device, 0).await,
            upload::<K>(&mut device, 1).await,
            upload::<K>(&mut device, 2).await,
        );
        for live in [
            1, 8, 9, 16, 17, 24, 25, 32, 65, 127, 128, 129, 200, 255, 256,
        ] {
            let values: Vec<_> = (0..All::SIZE * H::SIZE)
                .map(|at| {
                    let (row, feature) = (at / H::SIZE, at % H::SIZE);
                    bf16::from_f32(if row >= live {
                        f32::NAN
                    } else if feature == H::SIZE - 1 {
                        (row % 5 + 1) as f32
                    } else {
                        (feature % 31) as f32 / 64.0
                    })
                })
                .collect();
            let input = HostTensor::<bf16, m![All, H]>::from_vec(values.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let count = crate::tensor::live::tests::count(live)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let outputs = launch(projected, (&mut device, &weights, &input, &count))
                .await
                .unwrap();
            let q = outputs
                .0
                .to_host::<m![All, Q, D]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let k = outputs
                .1
                .to_host::<m![All, K, D]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let v = outputs
                .2
                .to_host::<m![All, K, D]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (projection, result) in [&q, &k, &v].into_iter().enumerate() {
                let width = if projection == 0 { Q::SIZE } else { K::SIZE } * D::SIZE;
                for row in 0..live {
                    let xs = &values[row * H::SIZE..(row + 1) * H::SIZE];
                    let a = xs.iter().map(|v| v.to_f32().abs()).fold(0.0, f32::max) / 448.0;
                    let dots: Vec<f32> = (0..7)
                        .map(|out| {
                            xs.iter()
                                .enumerate()
                                .map(|(i, x)| {
                                    f8e4m3::from_f32(x.to_f32() / a).to_f32() * code(out, i)
                                })
                                .sum()
                        })
                        .collect();
                    for out in 0..width {
                        let expected =
                            bf16::from_f32(dots[out % 7] * a * scale(out, projection)).to_f32();
                        let got = result[row * width + out].to_f32();
                        assert!(
                            got.is_finite() && (got - expected).abs() <= expected.abs() / 128.0,
                            "live={live} projection={projection} row={row} out={out}: {got} != {expected}"
                        );
                    }
                }
            }
        }
    }
}
