use super::HeadNorm;
use crate::Begin;
use crate::Op;
use crate::load::{Context, Load, LoadError};
use crate::modules::generic::Either;
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

/// ...or those of a group's tiles.
impl<C: M, P: M, R: M> Op<DmTensor<bf16, Chip, C, P, R>> for () {
    type Output = DmTensor<bf16, Chip, C, P, R>;

    fn forward(&self, _: &mut Device, rows: DmTensor<bf16, Chip, C, P, R>) -> Self::Output {
        rows
    }
}

/// Rows of `L` heads of `K` rows each, normalized in one tile and left in their own layout.
impl<Head: AxisName, L: M, K: M, C: M, P: M, const EPSILON: usize>
    Op<DmTensor<bf16, Chip, C, P, m![{ L }, { K }, Head]>> for Staged<Head, EPSILON>
{
    type Output = DmTensor<bf16, Chip, C, P, m![{ L }, { K }, Head]>;

    fn forward(
        &self,
        device: &mut Device,
        rows: DmTensor<bf16, Chip, C, P, m![{ L }, { K }, Head]>,
    ) -> Self::Output {
        type T<L, K> = m![{ L }, { K }];
        const { assert!(whole::<T<L, K>, m![Head]>() == 0) };
        let F32::<EPSILON> = self.epsilon;
        // SAFETY: grouping the row digits as one tuple preserves every physical coordinate.
        let rows: DmTensor<bf16, Chip, C, P, m![{ T<L, K> }, Head]> = unsafe { rows.reshape() };
        // SAFETY: the retained parameter replicates the same head gain on all clusters and slices.
        let weight = unsafe { self.weight.view().reshape::<Chip, C, P, m![Head]>() };
        let gain: VrfTensor<f32, Chip, C, P, m![Head]> = device
            .sub
            .begin(weight)
            .fetch::<m![Head / 8], m![Head % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Head / 8], m![Head % 8]>()
            .to_vrf();
        let variance: DmTensor<f32, Chip, C, P, m![{ T<L, K> }, 1 # 8]> = device
            .main
            .begin(rows.view())
            .fetch::<m![{ T<L, K> }, Head / 8], m![Head % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ T<L, K> }, Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ T<L, K> }, Head / 4], m![Head % 4]>()
            .vector_stash()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
            .vector_intra_slice_reduce::<Head, T<L, K>, m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_fp_div(const { Head::SIZE as f32 })
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit();
        let inverse: VrfTensor<f32, Chip, C, P, m![{ T<L, K> }, 1 # 8]> =
            Root::<T<L, K>, m![1 # 2]>::of(&mut device.main, variance.view(), |square| {
                square.vector_fp_binary(FpBinaryOp::AddF, f32::from_bits(EPSILON as u32))
            })
            .to_vrf(&mut device.sub);
        let normalized: DmTensor<bf16, Chip, C, P, m![{ T<L, K> }, Head]> =
            normalization::Scale::<kernel::Rows<T<L, K>, m![Head]>>::of(
                &mut device.main,
                rows.view(),
                &inverse,
                &gain,
            )
            .commit();
        // SAFETY: as above, the other way.
        unsafe { normalized.reshape() }
    }
}

/// Rows of `L` heads of `K` rows each known up to a factor of each row and head, normalized as
/// they are stored: the factor cancels but in epsilon.
impl<Head: AxisName, L: M, K: M, C: M, P: M, const EPSILON: usize>
    Op<Scaled<DmTensor<bf16, Chip, C, P, m![{ L }, { K }, Head]>>> for Staged<Head, EPSILON>
{
    type Output = DmTensor<bf16, Chip, C, P, m![{ L }, { K }, Head]>;

    fn forward(
        &self,
        device: &mut Device,
        rows: Scaled<DmTensor<bf16, Chip, C, P, m![{ L }, { K }, Head]>>,
    ) -> Self::Output {
        self.forward(device, rows.cancel())
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

    #[device(chip = 1)]
    fn normalized<Local: AxisName>(
        device: &mut Device,
        norm: &Stored<Width, EPSILON>,
        rows: &Rows<Local>,
    ) -> Rows<Local> {
        let norm = Stage.forward(&mut *device, norm);
        let rows: DmTensor<bf16, Chip, m![Clusters], m![Slices], m![Local, Group, Width]> =
            rows.to_dm(&mut device.tdma);
        norm.forward(&mut *device, rows).to_hbm(&mut device.tdma)
    }

    /// Rows of `local` heads in every slice: zero, or near zero against epsilon, or not.
    fn values(local: usize) -> Vec<bf16> {
        let rows = Clusters::SIZE * Slices::SIZE * local * Group::SIZE;
        (0..rows * Width::SIZE)
            .map(|index| {
                let (row, feature) = (index / Width::SIZE, index % Width::SIZE);
                let magnitude = [0.0, 0.02, 0.1, 0.5][(row / 3) % 4];
                let value = ((feature * 37 + row * 11) % 101) as f32 / 25.0 - 2.0;
                bf16::from_f32(magnitude * value)
            })
            .collect()
    }

    fn gains() -> Vec<bf16> {
        (0..Width::SIZE)
            .map(|feature| bf16::from_f32(0.5 + ((feature * 5) % 13) as f32 / 8.0))
            .collect()
    }

    /// The largest absolute error against the RMS norm of the rows, and the largest excess over
    /// BF16 rounding of the expected value or of one, with its row.
    fn error(actual: &[bf16], rows: &[bf16], gains: &[bf16]) -> (f64, f64, usize) {
        let epsilon = f64::from(f32::from_bits(EPSILON as u32));
        let mut worst = (0.0f64, f64::MIN, 0usize);
        for (row, (values, got)) in rows
            .chunks_exact(Width::SIZE)
            .zip(actual.chunks_exact(Width::SIZE))
            .enumerate()
        {
            let values: Vec<f64> = values.iter().map(|v| f64::from(v.to_f32())).collect();
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

    /// The kernel's rows of `Local` heads for `values`, read back to the host.
    async fn run<Local: AxisName>(
        device: &mut Device,
        norm: &Stored<Width, EPSILON>,
        values: Vec<bf16>,
    ) -> Vec<bf16> {
        let rows = HostTensor::<bf16, m![Clusters, Slices, Local, Group, Width]>::from_vec(values)
            .to_hbm::<Chip, m![Clusters, Slices, Local, Group, Width]>(&mut device.pdma)
            .await
            .unwrap();
        launch(normalized, (&mut *device, norm, &rows))
            .await
            .unwrap()
            .to_host::<m![Clusters, Slices, Local, Group, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }

    /// Head rows of one and of two heads a slice, near zero against a changed epsilon and under a
    /// changed gain, normalize by their own mean square plus epsilon.
    #[tokio::test(flavor = "current_thread")]
    async fn normalizes_head_rows_near_zero_against_epsilon() {
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
        for local in [Both::SIZE, Lone::SIZE] {
            let values = values(local);
            let actual = if local == Both::SIZE {
                run::<Both>(&mut device, &norm, values.clone()).await
            } else {
                run::<Lone>(&mut device, &norm, values.clone()).await
            };
            errors.push((local, error(&actual, &values, &gains)));
        }
        println!("(heads, (max error, excess over the bound, row)): {errors:?}");
        assert!(
            errors.iter().all(|(_, (_, excess, _))| *excess <= 0.0),
            "{errors:?}"
        );
    }
}
