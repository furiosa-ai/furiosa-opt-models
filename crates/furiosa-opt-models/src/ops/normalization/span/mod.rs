use furiosa_opt_std::prelude::*;

use crate::Op;
use crate::ops::normalization::{FeatureSlices, Offset, Part, Squares};
use crate::specialize::normalization::{capacity, last};
use crate::tensor::{axis::Rows, live::Live, parallel::Chip};

pub(crate) use residency::Selection;

pub(crate) mod kernel;
mod residency;
/// The rows one span loop trip holds: a norm's two cluster halves of 256 rows.
pub(crate) type Span = Rows<512>;
pub(crate) type Scale<Width, Rows> =
    DmTensor<bf16, Chip, m![Rows / (Rows / 2)], FeatureSlices<Width>, Offset<Width>>;
/// A span's rows, each cluster's half of them in the feature slices a chunk of features at a time.
pub(crate) type Owned<Width, All, Chips = Chip> = DmTensor<
    bf16,
    Chips,
    m![All / (All / 2)],
    FeatureSlices<Width>,
    m![All % (All / 2), { Offset<Width> }],
>;

/// A span's rows in one cluster's half.
type Half = m![Span % (Span / 2)];

/// A width whose norm runs a cluster's half of a span in tiles of `Capacity` rows, then a last one
/// of `Last` rows: the rows its vector registers hold, which the norm checks.
pub(crate) trait Tiled: AxisName {
    type Capacity;
    type Last;
}

/// A norm with its retained gain written once in DM, where its forward on a span's rows reads it.
pub(crate) struct Stage;

impl<Width: AxisName, const EPSILON: usize> Op<&super::RmsNorm<Width, EPSILON, Chip>> for Stage {
    type Output = super::RmsNorm<Width, EPSILON, Chip, Scale<Width, Span>>;

    fn forward(
        &self,
        device: &mut Device,
        norm: &super::RmsNorm<Width, EPSILON, Chip>,
    ) -> Self::Output {
        super::RmsNorm::new(norm.weight().tensor().to_dm(&mut device.tdma))
    }
}

/// A norm whose gain is in HBM writes it to DM for each span it normalizes.
impl<Width, R: AxisName, const EPSILON: usize, const CAPACITY: usize, const LAST: usize>
    Op<&Live<Owned<Width, Span>, R>> for super::RmsNorm<Width, EPSILON, Chip>
where
    Width: Tiled<Capacity = Rows<CAPACITY>, Last = Rows<LAST>>,
{
    type Output = Live<Owned<Width, Span>, R>;

    fn forward(&self, context: &mut Device, rows: &Live<Owned<Width, Span>, R>) -> Self::Output {
        let staged = Stage.forward(&mut *context, self);
        staged.forward(context, rows)
    }
}

/// The live rows of each cluster's half of a span, capped at the half; the norm keeps them.
impl<Width, R: AxisName, const EPSILON: usize, const CAPACITY: usize, const LAST: usize>
    Op<&Live<Owned<Width, Span>, R>> for super::RmsNorm<Width, EPSILON, Chip, Scale<Width, Span>>
where
    Width: Tiled<Capacity = Rows<CAPACITY>, Last = Rows<LAST>>,
{
    type Output = Live<Owned<Width, Span>, R>;

    fn forward(&self, context: &mut Device, rows: &Live<Owned<Width, Span>, R>) -> Self::Output {
        let reach = reach::<Span>(rows.rows());
        let gain: VrfTensor<f32, Chip, m![Span / (Span / 2)], FeatureSlices<Width>, Offset<Width>> =
            context
                .sub
                .begin(self.weight().view())
                .fetch::<m![{ Offset<Width> } / 8], m![{ Offset<Width> } % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Offset<Width> } / 8], m![{ Offset<Width> } % 8]>()
                .to_vrf();
        // The variances of the rows each slice holds, reduced over the slices of their features.
        let mut variance: DmTensor<f32, Chip, m![Span / (Span / 2)], m![256], m![{ Half }, 1 # 8]> =
            DmTensor::new();
        Squares::<kernel::Rows<Half, Offset<Width>>, Width, Half, Part<Width>, EPSILON>::of(
            &mut context.main,
            rows.value().view(),
        )
        .vector_inter_slice_reduce::<m![256], Half>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit_prefix::<Half, _, _>(variance.view_mut(), reach);
        // SAFETY: the reduction broadcasts each scalar to the original feature partitions.
        let variance: DmTensor<
            f32,
            Chip,
            m![Span / (Span / 2)],
            FeatureSlices<Width>,
            m![{ Half }, 1 # 8],
        > = unsafe { variance.reshape() };
        let input = (rows.value(), &variance, &gain);
        let mut output = Owned::<Width, Span>::new();
        tiles::<Width, CAPACITY, LAST>(context, input, &mut output, reach);
        rows.with(output)
    }
}

pub(crate) struct Tile<'i, 'o, Width: AxisName, Rows: M> {
    pub(crate) input: DmTensorView<
        'i,
        bf16,
        Chip,
        super::Cluster,
        super::FeatureSlices<Width>,
        m![{ Rows }, { Offset<Width> }],
    >,
    pub(crate) output: DmTensorViewMut<
        'o,
        bf16,
        Chip,
        super::Cluster,
        super::FeatureSlices<Width>,
        m![{ Rows }, { Offset<Width> }],
    >,
}

impl<'i, 'o, Width: AxisName, R: AxisName> Tile<'i, 'o, Width, m![Span % R]> {
    /// Wave `index` of `R` rows of a span, normalized into `rows`.
    pub(crate) fn of(
        span: &'i super::Spread<Width, Span>,
        index: usize,
        rows: &'o mut super::Spread<Width, R>,
    ) -> Self {
        const { assert!(R::SIZE > 0 && Span::SIZE.is_multiple_of(R::SIZE)) };
        // SAFETY: R divides Span; the view renames the same rows without moving coordinates.
        let output = unsafe {
            rows.view_mut()
                .reshape::<
                    Chip,
                    super::Cluster,
                    super::FeatureSlices<Width>,
                    m![Span % R, { Offset<Width> }],
                >()
        };
        Self {
            input: span
                .view()
                .tile::<m![Span / R], 1, m![
                    Span / R = 1 # (Span / R), Span % R, { Offset<Width> }
                ]>(index)
                .unpad::<m![Span % R, { Offset<Width> }]>(),
            output,
        }
    }
}

impl<Width: AxisName, Rows: M, const EPSILON: usize, Parameter> Op<Tile<'_, '_, Width, Rows>>
    for super::RmsNorm<Width, EPSILON, Chip, Parameter>
where
    Parameter: super::Learned<Width, Chip>,
{
    type Output = ();

    fn forward(&self, context: &mut Device, input: Tile<'_, '_, Width, Rows>) {
        self.normalized::<Rows, _>(context, input.input)
            .commit_view(input.output);
    }
}

/// A cluster's half of a span normalized into `output` in tiles of `CAPACITY` rows, then a last one
/// of `LAST` rows; each tile's rows past the live ones keep stale values.
fn tiles<Width: AxisName, const CAPACITY: usize, const LAST: usize>(
    context: &mut Device,
    input: kernel::Inputs<'_, m![Span / (Span / 2)], FeatureSlices<Width>, Half, Offset<Width>>,
    output: &mut Owned<Width, Span>,
    reach: usize,
) {
    const {
        assert!(CAPACITY == capacity::<Offset<Width>>());
        assert!(LAST == last::<Half, Offset<Width>>());
    };
    #[unroll]
    for tile in 0..const { (Half::SIZE - LAST) / CAPACITY } {
        let start = tile * CAPACITY;
        kernel::normalized::<CAPACITY, _, _, _, _>(context, input, start)
            .commit_prefix::<m![{ Half } = { CAPACITY }], _, _>(
                output.view_mut().tile::<Half, CAPACITY, m![
                    { Half } = { CAPACITY } #{!} ({ Half } / 1), { Offset<Width> }
                ]>(start),
                reach - start,
            );
    }
    let start = Half::SIZE - LAST;
    kernel::normalized::<LAST, _, _, _, _>(context, input, start)
        .commit_prefix::<m![{ Half } = { LAST }], _, _>(
            output.view_mut().tile::<Half, LAST, m![
                { Half } = { LAST } #{!} ({ Half } / 1), { Offset<Width> }
            ]>(start),
            reach - start,
        );
}

/// Rows of each cluster's half of `Rows` that the first `rows` rows fill: `rows` capped at the
/// half, spelled without a comparison since `rows` is a runtime count; no step goes negative.
pub(crate) fn reach<Rows: AxisName>(rows: usize) -> usize {
    let half = Rows::SIZE / 2;
    rows + rows / (half + 1) * half - rows / (half + 1) * rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Module;
    use crate::modules::generic::schedule::{Fill, Spill};
    use crate::ops::attention::geometry::Hidden;
    use crate::ops::normalization::tests::random;
    use crate::ops::normalization::{Blocked, EPSILON, RmsNorm};
    use crate::ops::pointwise::add::Add;
    use crate::tensor::form::Weight;
    use crate::tensor::live::tests::count;
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![Width = 4096];

    impl Tiled for Width {
        type Capacity = Rows<126>;
        type Last = Rows<4>;
    }

    impl Tiled for Hidden<5120> {
        type Capacity = Rows<125>;
        type Last = Rows<6>;
    }

    impl Tiled for Hidden<6656> {
        type Capacity = Rows<125>;
        type Last = Rows<6>;
    }

    type Stored<W> = HbmTensor<bf16, Chip, Blocked<W, Span>>;

    #[test]
    fn accepts_gain_in_dm_and_live_rows() {
        fn accepts<N>()
        where
            N: for<'d, 'a> Module<
                    &'a Live<Owned<Width, Span>, Span>,
                    (),
                    &'d mut (),
                    Output = Live<Owned<Width, Span>, Span>,
                >,
            Stage:
                for<'d, 'n> Module<&'n RmsNorm<Width, EPSILON, Chip>, (), &'d mut (), Output = N>,
        {
        }
        accepts::<RmsNorm<Width, EPSILON, Chip, Scale<Width, Span>>>();
    }

    #[test]
    fn accepts_tile_views_and_gain_in_dm() {
        fn accepts<N>()
        where
            N: for<'d, 'i, 'o> Module<Tile<'i, 'o, Width, m![128]>, (), &'d mut (), Output = ()>,
        {
        }
        accepts::<RmsNorm<Width, EPSILON, Chip, super::super::InDm<Width>>>();
    }

    /// A span's live rows normalized, then added to themselves as a layer's residual adds them.
    #[device(chip = 1)]
    fn residual<W: AxisName>(
        device: &mut Device,
        norm: &RmsNorm<W, EPSILON, Chip>,
        rows: &Value,
        input: &Stored<W>,
        output: &mut Stored<W>,
    ) where
        RmsNorm<W, EPSILON, Chip>:
            for<'t> Op<&'t Live<Owned<W, Span>, Span>, Output = Live<Owned<W, Span>, Span>>,
    {
        let live = Live::<(), Span>::read(rows);
        let tile = live.with(input.fill(&mut *device, live));
        let normalized = Op::forward(norm, &mut *device, &tile);
        let sum = Op::forward(&Add, &mut *device, (&normalized, &tile));
        output.spill(&mut *device, sum.value(), live);
    }

    /// `bf16(bf16(x * gain / sqrt(mean(x²) + epsilon)) + x)` of every row, FP32 statistics.
    fn definition(rows: &[bf16], gain: &[bf16]) -> Vec<f32> {
        let width = gain.len();
        rows.chunks_exact(width)
            .flat_map(|row| {
                let mean = row.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>() / width as f32;
                let inverse = 1.0 / (mean + 1e-6).sqrt();
                row.iter().zip(gain).map(move |(v, g)| {
                    let normalized = bf16::from_f32(v.to_f32() * inverse * g.to_f32()).to_f32();
                    bf16::from_f32(normalized + v.to_f32()).to_f32()
                })
            })
            .collect()
    }

    async fn residual_matches_the_definition<W: AxisName>(device: &mut Device)
    where
        RmsNorm<W, EPSILON, Chip>:
            for<'t> Op<&'t Live<Owned<W, Span>, Span>, Output = Live<Owned<W, Span>, Span>>,
    {
        let width = W::SIZE;
        let values = random(Span::SIZE * width, 0x9e37_79b9_7f4a_7c15 ^ width as u64);
        let gain = random(width, 0x0123_4567_89ab_cdef);
        let expected = definition(&values, &gain);
        let input = HostTensor::<bf16, m![Span, W]>::from_vec(values)
            .to_hbm::<Chip, Blocked<W, Span>>(&mut device.pdma)
            .await
            .unwrap();
        let norm = RmsNorm::new(Weight::new(
            HostTensor::<bf16, m![W]>::from_vec(gain)
                .to_hbm::<Chip, m![W]>(&mut device.pdma)
                .await
                .unwrap(),
        ));
        for live in [1, 200, 300, 512] {
            let rows = count(live).to_hbm(&mut device.pdma).await.unwrap();
            let mut output = HostTensor::<bf16, m![Span, W]>::zero()
                .to_hbm::<Chip, Blocked<W, Span>>(&mut device.pdma)
                .await
                .unwrap();
            launch(residual, (&mut *device, &norm, &rows, &input, &mut output))
                .await
                .unwrap();
            let actual = output
                .to_host::<m![Span, W]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let rows = actual.iter().zip(&expected).take(live * width);
            for (index, (got, expected)) in rows.enumerate() {
                assert!(
                    (got.to_f32() - expected).abs() <= 0.02 * expected.abs().max(1.0),
                    "width {width}, {live} live rows, row {} feature {}: {} vs {expected}",
                    index / width,
                    index % width,
                    got.to_f32(),
                );
            }
        }
    }

    /// The span's norm and residual add at every listed hidden width, padded feature slices
    /// included, match the definition on each live row.
    #[tokio::test(flavor = "current_thread")]
    async fn residual_matches_at_every_width() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        residual_matches_the_definition::<Hidden<1024>>(&mut device).await;
        residual_matches_the_definition::<Hidden<2048>>(&mut device).await;
        residual_matches_the_definition::<Hidden<2560>>(&mut device).await;
        residual_matches_the_definition::<Hidden<4096>>(&mut device).await;
        residual_matches_the_definition::<Hidden<5120>>(&mut device).await;
        residual_matches_the_definition::<Hidden<6656>>(&mut device).await;
    }
}
