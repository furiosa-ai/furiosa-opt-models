use super::HeadNorm;
use crate::Op;
use crate::load::{Context, Load, LoadError};
use crate::modules::generic::Either;
use crate::modules::generic::schedule::Begin;
use crate::ops::normalization::span::kernel;
use crate::ops::normalization::{self, Root, Scaled};
use crate::specialize::normalization::whole;
use crate::tensor::form::Weight;
use crate::tensor::parallel::Chip;
use crate::tensor::scalar::F32;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

axes![Replica = 512];
pub type Replicated<Head> = m![Replica, Head];
pub(crate) type Scale<Head> = DmTensor<bf16, Chip, m![Replica / 256], m![Replica % 256], m![Head]>;

/// A head's gain copied to every replica, in BF16 as the staged norm reads it.
impl<Head: AxisName, Chips: M + 'static, T> Load<HostTensor<f32, m![{ Chips }, { m![Head] }]>>
    for Weight<Replicated<Head>, Chips, T>
where
    T: Load<HostTensor<bf16, m![{ Chips }, { Replicated<Head> }]>>,
{
    fn load(
        context: &mut Context<'_>,
        gain: HostTensor<f32, m![{ Chips }, { m![Head] }]>,
    ) -> Result<Self, LoadError> {
        let replicated = (gain.into_vec().chunks(Head::SIZE))
            .flat_map(|head| {
                let head = head.iter().map(|&g| bf16::from_f32(g));
                head.cycle().take(Replica::SIZE * Head::SIZE)
            })
            .collect::<Vec<_>>();
        Load::load(context, HostTensor::from_vec(replicated))
    }
}

/// A head norm whose gain the checkpoint stores, and the same norm with the gain written.
pub(crate) type Stored<Head, const EPSILON: usize> =
    HeadNorm<Head, Chip, Weight<Replicated<Head>, Chip>, F32<EPSILON>>;
pub(crate) type Staged<Head, const EPSILON: usize> =
    HeadNorm<Head, Chip, Scale<Head>, F32<EPSILON>>;

/// A head norm with its replicated gain written once in DM, where its forward reads it.
pub(crate) struct Stage;

impl<Head: AxisName, const EPSILON: usize> Op<&Stored<Head, EPSILON>> for Stage {
    type Output = Staged<Head, EPSILON>;

    fn forward(&self, device: &mut Device, norm: &Stored<Head, EPSILON>) -> Self::Output {
        HeadNorm {
            weight: norm.weight.tensor().to_dm(&mut device.tdma),
            epsilon: F32,
            head: PhantomData,
        }
    }
}

/// The head norm's gain written, as a front's norm writes it before any rows arrive.
impl<Head: AxisName, const EPSILON: usize> Op<Begin<()>> for Stored<Head, EPSILON> {
    type Output = Staged<Head, EPSILON>;

    fn forward(&self, device: &mut Device, _: Begin<()>) -> Self::Output {
        Stage.forward(device, self)
    }
}

/// An absent head norm, `()`, writes nothing before the rows arrive...
impl Op<Begin<()>> for () {
    type Output = ();

    fn forward(&self, _: &mut Device, _: Begin<()>) {}
}

/// ...and leaves the head rows of a period as projected.
impl<A, B> Op<Either<A, B>> for () {
    type Output = Either<A, B>;

    fn forward(&self, _: &mut Device, rows: Either<A, B>) -> Self::Output {
        rows
    }
}

impl<Head: AxisName, const EPSILON: usize> Staged<Head, EPSILON> {
    /// Rows `u` of `T` rows of heads whose true values are `c s u`, `inverse` holding each row's
    /// `1 / (c s)²` (`c s = 1` without it), normalized as the true rows are and left in their layout.
    fn normalize<T: M, C: M, P: M, R: M>(
        &self,
        device: &mut Device,
        rows: DmTensor<bf16, Chip, C, P, m![{ T }, Head]>,
        inverse: Option<&VrfTensor<f32, Chip, C, P, R>>,
    ) -> DmTensor<bf16, Chip, C, P, m![{ T }, Head]> {
        const { assert!(whole::<T, m![Head]>() == 0) };
        let F32::<EPSILON> = self.epsilon;
        // SAFETY: the retained parameter replicates the same head gain on all clusters and slices.
        let weight = unsafe { self.weight.view().reshape::<Chip, C, P, m![Head]>() };
        let gain: VrfTensor<f32, Chip, C, P, m![Head]> = device
            .sub
            .begin(weight)
            .fetch::<m![Head / 8], m![Head % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Head / 8], m![Head % 8]>()
            .to_vrf();
        let epsilon = f32::from_bits(EPSILON as u32);
        let squares = device
            .main
            .begin(rows.view())
            .fetch::<m![{ T }, Head / 8], m![Head % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ T }, Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ T }, Head / 4], m![Head % 4]>()
            .vector_stash()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash);
        let inverse: VrfTensor<f32, Chip, C, P, m![{ T }, 1 # 8]> = match inverse {
            // Each row's `epsilon / (c s)²` joins every square, so the mean holds it once.
            Some(inverse) => {
                let variance: DmTensor<f32, Chip, C, P, m![{ T }, 1 # 8]> = squares
                    .vector_fp_ternary_with_mode(
                        FpTernaryOp::FmaF,
                        TernaryArgMode::Mode120,
                        (inverse, epsilon),
                    )
                    .vector_intra_slice_reduce::<Head, T, m![1 # 4]>(IntraSliceReduceOpF32::Add)
                    .vector_fp_div(const { Head::SIZE as f32 })
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .commit_trim::<m![1 # 2]>()
                    .commit();
                Root::<T, m![1 # 2]>::of(&mut device.main, variance.view(), |square| square)
                    .to_vrf(&mut device.sub)
            }
            None => {
                let variance: DmTensor<f32, Chip, C, P, m![{ T }, 1 # 8]> = squares
                    .vector_intra_slice_reduce::<Head, T, m![1 # 4]>(IntraSliceReduceOpF32::Add)
                    .vector_fp_div(const { Head::SIZE as f32 })
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .commit_trim::<m![1 # 2]>()
                    .commit();
                Root::<T, m![1 # 2]>::of(&mut device.main, variance.view(), |square| {
                    square.vector_fp_binary(FpBinaryOp::AddF, epsilon)
                })
                .to_vrf(&mut device.sub)
            }
        };
        normalization::Scale::<kernel::Rows<T, m![Head]>>::of(
            &mut device.main,
            rows.view(),
            &inverse,
            &gain,
        )
        .commit()
    }
}

/// Rows of `L` heads of `K` rows each, known up to each row's factor, normalized as their true
/// values are in one tile and left in their own layout.
impl<Head: AxisName, L: M, K: M, C: M, P: M, R: M, const EPSILON: usize>
    Op<Scaled<DmTensor<bf16, Chip, C, P, m![{ L }, { K }, Head]>, VrfTensor<f32, Chip, C, P, R>>>
    for Staged<Head, EPSILON>
{
    type Output = DmTensor<bf16, Chip, C, P, m![{ L }, { K }, Head]>;

    fn forward(
        &self,
        device: &mut Device,
        rows: Scaled<
            DmTensor<bf16, Chip, C, P, m![{ L }, { K }, Head]>,
            VrfTensor<f32, Chip, C, P, R>,
        >,
    ) -> Self::Output {
        let (rows, inverse) = rows.cancel();
        // SAFETY: grouping the row digits as one tuple preserves every physical coordinate.
        let rows: DmTensor<bf16, Chip, C, P, m![{ m![{ L }, { K }] }, Head]> =
            unsafe { rows.reshape() };
        let normalized = self.normalize::<m![{ L }, { K }], C, P, R>(device, rows, Some(&inverse));
        // SAFETY: as above, the other way.
        unsafe { normalized.reshape() }
    }
}

/// Rows of `L` heads of `K` rows each, their true values, normalized in one tile and left in
/// their own layout.
impl<Head: AxisName, L: M, K: M, C: M, P: M, const EPSILON: usize>
    Op<DmTensor<bf16, Chip, C, P, m![{ L }, { K }, Head]>> for Staged<Head, EPSILON>
{
    type Output = DmTensor<bf16, Chip, C, P, m![{ L }, { K }, Head]>;

    fn forward(
        &self,
        device: &mut Device,
        rows: DmTensor<bf16, Chip, C, P, m![{ L }, { K }, Head]>,
    ) -> Self::Output {
        // SAFETY: grouping the row digits as one tuple preserves every physical coordinate.
        let rows: DmTensor<bf16, Chip, C, P, m![{ m![{ L }, { K }] }, Head]> =
            unsafe { rows.reshape() };
        let normalized = self.normalize::<m![{ L }, { K }], C, P, m![1]>(device, rows, None);
        // SAFETY: as above, the other way.
        unsafe { normalized.reshape() }
    }
}

/// The head rows of a period of a pair of tiles or of a single tile.
impl<Head: AxisName, A, B, OA, OB, const EPSILON: usize> Op<Either<A, B>> for Staged<Head, EPSILON>
where
    Self: Op<A, Output = OA> + Op<B, Output = OB>,
{
    type Output = Either<OA, OB>;

    fn forward(&self, device: &mut Device, rows: Either<A, B>) -> Self::Output {
        match rows {
            Either::Left(rows) => Either::Left(self.forward(device, rows)),
            Either::Right(rows) => Either::Right(self.forward(device, rows)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![
        Clusters = 2,
        Slices = 256,
        Both = 2,
        Lone = 1,
        Group = 8,
        Width = 128
    ];

    /// A changed epsilon, as FP32 bits, large enough that it matters for near-zero rows.
    const EPSILON: usize = 1e-2_f32.to_bits() as usize;

    type Rows<Local> = HbmTensor<bf16, Chip, m![Clusters, Slices, Local, Group, Width]>;
    type Factors = HbmTensor<bf16, Chip, m![Clusters, Slices, Group]>;
    /// Each head's squared factor, for each of its rows.
    type Heads<Local> = HbmTensor<f32, Chip, m![Clusters, Slices, Local, Group]>;

    #[device(chip = 1)]
    fn normalized<Local: AxisName>(
        device: &mut Device,
        norm: &Stored<Width, EPSILON>,
        rows: &Rows<Local>,
        factors: &Factors,
        heads: &Heads<Local>,
    ) -> Rows<Local> {
        let norm = Stage.forward(&mut *device, norm);
        let rows: DmTensor<bf16, Chip, m![Clusters], m![Slices], m![Local, Group, Width]> =
            rows.to_dm(&mut device.tdma);
        let factors: DmTensor<bf16, Chip, m![Clusters], m![Slices], m![Group]> =
            factors.to_dm(&mut device.tdma);
        let heads: DmTensor<f32, Chip, m![Clusters], m![Slices], m![Local, Group]> =
            heads.to_dm(&mut device.tdma);
        let heads: VrfTensor<f32, Chip, m![Clusters], m![Slices], m![Local, Group]> = device
            .sub
            .begin(heads.view())
            .fetch::<m![Local], m![Group]>()
            .collect::<m![Local], m![Group]>()
            .to_vrf();
        let rows: Scaled<_, VrfTensor<f32, Chip, _, _, m![{ m![Local] }, { m![Group] }]>> =
            Scaled::new(&mut *device, rows, factors.view(), &heads);
        norm.forward(&mut *device, rows).to_hbm(&mut device.tdma)
    }

    /// Each row's factor: one, two distinct powers of two and one that is not.
    fn factor(index: usize, ones: bool) -> bf16 {
        let factors = [1.0, 1.0 / 64.0, 1.0 / 1024.0, 3.0 / 512.0];
        bf16::from_f32(if ones { 1.0 } else { factors[index % 4] })
    }

    /// The true value of feature `feature` of row `row`: zero, or near zero against epsilon.
    fn truth(row: usize, feature: usize) -> f32 {
        let magnitude = [0.0, 0.02, 0.1, 0.5][(row / 3) % 4];
        magnitude * (((feature * 37 + row * 11) % 101) as f32 / 25.0 - 2.0)
    }

    /// Each head's factor: one, or weight scales that are not powers of two.
    fn scale(head: usize, ones: bool) -> f32 {
        let scales = [1.37e-3, 3.1e-4, 1.0, 7.7e-4, 2.9e-3];
        if ones { 1.0 } else { scales[head % 5] }
    }

    /// Rows of `local` heads in every slice, each as stored, `true / (c s)`, its row factors `c`,
    /// each head's factor `s` for each of its rows, and its restored values.
    fn operands(local: usize, ones: bool) -> (Vec<bf16>, Vec<bf16>, Vec<f32>, Vec<f64>) {
        let slices = Clusters::SIZE * Slices::SIZE;
        let factors: Vec<bf16> = (0..slices * Group::SIZE)
            .map(|index| factor(index, ones))
            .collect();
        let mut stored = Vec::with_capacity(slices * local * Group::SIZE * Width::SIZE);
        let mut restored = Vec::with_capacity(stored.capacity());
        let mut heads = Vec::with_capacity(slices * local * Group::SIZE);
        for slice in 0..slices {
            for head in 0..local {
                let s = scale(slice * local + head, ones);
                for row in 0..Group::SIZE {
                    let c = factors[slice * Group::SIZE + row].to_f32();
                    let index = (slice * local + head) * Group::SIZE + row;
                    heads.push(s * s);
                    for feature in 0..Width::SIZE {
                        let value = bf16::from_f32(truth(index, feature) / (c * s));
                        stored.push(value);
                        restored.push(f64::from(value.to_f32()) * f64::from(c) * f64::from(s));
                    }
                }
            }
        }
        (stored, factors, heads, restored)
    }

    fn gains() -> Vec<bf16> {
        (0..Width::SIZE)
            .map(|feature| bf16::from_f32(0.5 + ((feature * 5) % 13) as f32 / 8.0))
            .collect()
    }

    /// The largest absolute error against the RMS norm of the restored values, and the largest
    /// excess over BF16 rounding of the expected value or of one, with its row.
    fn error(actual: &[bf16], restored: &[f64], gains: &[bf16]) -> (f64, f64, usize) {
        let epsilon = f64::from(f32::from_bits(EPSILON as u32));
        let mut worst = (0.0f64, f64::MIN, 0usize);
        for (row, (values, got)) in restored
            .chunks_exact(Width::SIZE)
            .zip(actual.chunks_exact(Width::SIZE))
            .enumerate()
        {
            let mean = values.iter().map(|v| v * v).sum::<f64>() / Width::SIZE as f64;
            let inverse = 1.0 / (mean + epsilon).sqrt();
            for ((value, gain), got) in values.iter().zip(gains).zip(got) {
                let expected = value * inverse * f64::from(gain.to_f32());
                let error = (f64::from(got.to_f32()) - expected).abs();
                worst.0 = worst.0.max(error);
                let excess = error - (expected.abs() / 256.0 + 1e-3);
                if excess > worst.1 {
                    worst = (worst.0, excess, row);
                }
            }
        }
        worst
    }

    /// The kernel's rows of `Local` heads for `stored`, read back to the host.
    async fn run<Local: AxisName>(
        device: &mut Device,
        norm: &Stored<Width, EPSILON>,
        stored: Vec<bf16>,
        factors: &Factors,
        heads: Vec<f32>,
    ) -> Vec<bf16> {
        let rows = HostTensor::<bf16, m![Clusters, Slices, Local, Group, Width]>::from_vec(stored)
            .to_hbm::<Chip, m![Clusters, Slices, Local, Group, Width]>(&mut device.pdma)
            .await
            .unwrap();
        let heads = HostTensor::<f32, m![Clusters, Slices, Local, Group]>::from_vec(heads)
            .to_hbm::<Chip, m![Clusters, Slices, Local, Group]>(&mut device.pdma)
            .await
            .unwrap();
        launch(normalized, (&mut *device, norm, &rows, factors, &heads))
            .await
            .unwrap()
            .to_host::<m![Clusters, Slices, Local, Group, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }

    /// Head rows known up to a positive factor per row and one per head, near zero against a
    /// changed epsilon and under a changed gain, normalize as their true values do; factors of one
    /// included.
    #[tokio::test(flavor = "current_thread")]
    async fn normalizes_as_the_true_values() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(normalized.topology()).unwrap();
        let gains = gains();
        let replicated: Vec<bf16> = (0..Replica::SIZE).flat_map(|_| gains.clone()).collect();
        let norm: Stored<Width, EPSILON> = HeadNorm {
            weight: Weight::new(
                HostTensor::<bf16, Replicated<Width>>::from_vec(replicated)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ),
            epsilon: F32,
            head: PhantomData,
        };
        let mut errors = Vec::new();
        for ones in [true, false] {
            for local in [Both::SIZE, Lone::SIZE] {
                let (stored, factors, heads, restored) = operands(local, ones);
                let factors = HostTensor::<bf16, m![Clusters, Slices, Group]>::from_vec(factors)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap();
                let actual = if local == Both::SIZE {
                    run::<Both>(&mut device, &norm, stored, &factors, heads).await
                } else {
                    run::<Lone>(&mut device, &norm, stored, &factors, heads).await
                };
                errors.push((ones, local, error(&actual, &restored, &gains)));
            }
        }
        println!("(ones, heads, (max error, excess over the bound, row)): {errors:?}");
        assert!(
            errors.iter().all(|(_, _, (_, excess, _))| *excess <= 0.0),
            "{errors:?}"
        );
    }
}
