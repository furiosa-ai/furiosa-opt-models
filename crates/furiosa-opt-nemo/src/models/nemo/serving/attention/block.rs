use super::*;
use crate::Module;
use crate::compose::{Tile, Tiles};
use crate::ops::attention::gqa::{PagedAttention, softmax, visibility};
use crate::ops::layout::Dm;

type Space = Tile<Jobs, Jobs>;

impl visibility::Mapping for Space {
    type Rows = m![Jobs % 8];
    type Groups = m![Groups];
    type Keys = m![Tokens];
}

impl softmax::Mapping for Space {
    type Key = Tokens;
}

type Probability = TrfTensor<bf16, Chip, Cluster, Slices, m![Groups], m![Jobs % 8, Tokens / 16]>;
type Stats<'l> = DmTensorViewMut<
    'l,
    f32,
    Chip,
    Cluster,
    Slices,
    m![Burst = 1 #{!} 8, Jobs % 8, Stat = 1 #{!} 2, Groups, 1 # 8],
>;
pub(super) const ABSENT: TagGuard =
    TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
/// The scores' scale, one over the square root of the head size.
pub(super) const SCALE: f32 = 0.08838834764831845;

/// Record `index` of wave `at`'s statistics: 0 the peak, 1 the mass.
pub(super) fn stat(records: &mut Records, at: usize, index: usize) -> Stats<'_> {
    records
        .view_mut()
        .tile::<m![Burst], 1, m![Burst = 1 #{!} 8, Jobs % 8, Stat, Groups, 1 # 8]>(at)
        .tile::<m![Stat], 1, m![Burst = 1 #{!} 8, Jobs % 8, Stat = 1 #{!} 2, Groups, 1 # 8]>(index)
}

pub(super) fn scores(context: &mut Device, key: &Decoded, query: &Lanes) -> Scores {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: (key.view(), query),
            },
        ))
        .value
}

/// Scaled scores, absent keys at the lowest bf16: a key is absent when its page index reaches the
/// job's count of pages that hold the slice's token.
pub(super) fn mask(
    context: &mut Device,
    scores: &Scores,
    (positions, counts): (&Positions, Counts<'_>),
) -> Weights {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: (scores.view(), positions.view(), counts),
            },
        ))
        .value
}

/// Each page slice's peak over its keys of every job.
pub(super) fn local_peak(context: &mut Device, scores: &Weights) -> Local {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: scores.view(),
            },
        ))
        .value
}

/// Each job's peak over the page slices' `local` peaks, at least the lowest finite score.
pub(super) fn job_peak<E: M>(
    context: &mut Device,
    local: &Local,
    peak: DmTensorViewMut<'_, f32, Chip, Cluster, Slices, E>,
) {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: (local.view(), peak),
            },
        ));
}

/// Each key's weight against its job's peak in record 0 of wave `at`, into `weights`.
pub(super) fn weights(
    context: &mut Device,
    scores: &Weights,
    records: &Records,
    at: usize,
    weights: &mut Weights,
) {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
        context,
        Dm::<Space, _> {
            tiles: Tiles::over(Jobs::SIZE),
            value: (
                scores.view(),
                records
                    .view()
                    .tile::<m![Burst], 1, m![Burst = 1 # 8, Jobs % 8, Stat, Groups, 1 # 8]>(at)
                    .tile::<m![Stat], 1, m![Burst = 1 # 8, Jobs % 8, Stat = 1 # 2, Groups, 1 # 8]>(
                        0,
                    ),
                weights.view_mut(),
            ),
        },
    ));
}

/// Each key's weight against its job's `top`, into `weights`.
pub(super) fn weigh(context: &mut Device, scores: &Weights, top: &Scalar, weights: &mut Weights) {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: (scores.view(), top, weights.view_mut()),
            },
        ));
}

/// Each page slice's weight mass over its keys of every job.
pub(super) fn local_mass(context: &mut Device, weights: &Weights) -> Local {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: (weights.view(),),
            },
        ))
        .value
}

/// Each job's mass over the page slices' `local` masses.
pub(super) fn job_mass<E: M>(
    context: &mut Device,
    local: &Local,
    mass: DmTensorViewMut<'_, f32, Chip, Cluster, Slices, E>,
) {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: ((local.view(),), mass),
            },
        ));
}

pub(super) fn fill(context: &mut Device, weights: &Weights) -> Probability {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: (weights.view(), ()),
            },
        ))
        .value
}

pub(super) fn weighted(context: &mut Device, value: &Decoded, weights: &Probability) -> Partial {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: ((value.view(), weights),),
            },
        ))
        .value
}

/// Rounds a contraction result to bf16 through Main before a DMA reads it: the context wait that
/// orders the DMA does not cover the contraction's commit, so a direct store can read stale cells.
pub(super) fn settle(context: &mut Device, partial: &Partial, rounded: &mut Rounded, at: usize) {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: (
                    partial.view(),
                    (rounded
                        .view_mut()
                        .tile::<m![Burst], 1, m![Burst = 1 #{!} 8, Jobs % 8, Groups, Head]>(at),),
                ),
            },
        ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::nemo::serving::residual::Shard;
    use furiosa_opt_std::runtime::launch;

    type Cells = m![Jobs, Heads, Groups, Page, Tokens / 16];
    type Waves = m![Parity, Burst, Jobs, Heads, Page, Lane];
    const WAVE: usize = 1;
    const PARITY: usize = 1;

    #[device(chip = 2)]
    fn masked(
        device: &mut Device,
        (scores, waves, positions): (
            &HbmTensor<f32, Chip, Cells>,
            &HbmTensor<f32, Chip, Waves>,
            &HbmTensor<f32, Chip, m![Tokens / 16]>,
        ),
    ) -> HbmTensor<bf16, Chip, Cells> {
        let scores: Scores = scores.to_dm(&mut device.tdma);
        let waves: DmTensor<f32, Chip, Cluster, Slices, m![Parity, Burst, Jobs % 8, Lane]> =
            waves.to_dm(&mut device.tdma);
        let positions: Positions = positions.to_dm(&mut device.tdma);
        // SAFETY: `Lane` lays out the eight cells of `1 # 8`.
        let waves =
            unsafe { waves.reshape::<Chip, Cluster, Slices, m![Parity, Burst, Jobs % 8, 1 # 8]>() };
        let counted = waves
            .view()
            .tile::<m![Parity], 1, m![Parity = 1 # 2, Burst, Jobs % 8, 1 # 8]>(PARITY)
            .tile::<m![Burst], 1, m![Parity = 1 # 2, Burst = 1 # 8, Jobs % 8, 1 # 8]>(WAVE);
        mask(device, &scores, (&positions, counted)).to_hbm::<Cells>(&mut device.tdma)
    }

    /// Keys at or past a job's count of pages for their token replace NaN scores; present keys
    /// scale before bf16 rounding.
    #[tokio::test]
    async fn masks_absent_keys() -> Result<(), Box<dyn std::error::Error>> {
        let mut context = Device::new(masked.topology())?;
        let values: Vec<f32> = (0..2 * Cells::SIZE)
            .map(|at| match (at + WAVE) % 4 {
                0 => f32::NAN,
                1 => 1.234375,
                2 => -2.71875,
                _ => 3.1415927,
            })
            .collect();
        // Counts of 0 to 33 pages per wave, job, head and token, so some cells of every kind; each
        // parity's half holds other counts.
        let count = |chip: usize, wave: usize, job: usize, head: usize, token: usize| {
            (chip + wave + job * 7 + head * 3 + token) % 34
        };
        let counts: Vec<f32> = (0..2 * Waves::SIZE)
            .map(|at| {
                let (chip, at) = (at / Waves::SIZE, at % Waves::SIZE / 8);
                let (parity, at) = (at / (8 * 64 * 4 * 16), at % (8 * 64 * 4 * 16));
                let (wave, job, head, token) =
                    (at / (64 * 4 * 16), at / (4 * 16) % 64, at / 16 % 4, at % 16);
                count(chip, wave + 8 * parity, job, head, token) as f32
            })
            .collect();
        let scores = HostTensor::<f32, m![Shard, { Cells }]>::from_vec(values.clone())
            .to_hbm::<Chip, Cells>(&mut context.pdma)
            .await?;
        let waves = HostTensor::<f32, m![Shard, { Waves }]>::from_vec(counts)
            .to_hbm::<Chip, Waves>(&mut context.pdma)
            .await?;
        let positions = HostTensor::<f32, m![Shard, Tokens / 16]>::from_vec(
            (0..2 * 32).map(|at| (at % 32) as f32),
        )
        .to_hbm::<Chip, m![Tokens / 16]>(&mut context.pdma)
        .await?;

        let actual = launch(masked, (&mut context, (&scores, &waves, &positions)))
            .await?
            .to_host::<m![Shard, { Cells }]>(&mut context.pdma)
            .await?
            .into_vec();
        for (at, (&actual, &score)) in actual.iter().zip(&values).enumerate() {
            let (chip, cell) = (at / Cells::SIZE, at % Cells::SIZE);
            let (job, head, token) = (cell / (4 * 4 * 512), cell / (4 * 512) % 4, cell % 512);
            let (token, page) = (token / 32, token % 32);
            let present = count(chip, WAVE + 8 * PARITY, job, head, token);
            let expected = bf16::from_f32(if page >= present {
                f32::from_bits(0xff7f_0000)
            } else {
                score * 0.08838834764831845f32
            });
            let (actual, expected) = (actual.to_f32(), expected.to_f32());
            assert!(
                actual == expected || actual.is_nan() && expected.is_nan(),
                "cell {at}: {actual} vs {expected}"
            );
        }
        Ok(())
    }
}
