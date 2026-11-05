use furiosa_opt_std::prelude::*;

use super::vector::Argmax;
use crate::compose::Space;
use crate::{Module, ops::layout::Dm};

axes![Term = 2];
type Cluster = m![1 # 2];
type Slices<V> = m![V / 256];
type Values<D, V, R, C> = DmTensor<D, C, Cluster, Slices<V>, m![R, V % 256]>;
type Reduced<D, R, C> = DmTensor<D, C, Cluster, m![1 # 256], m![R, 1 # 8]>;
type Maximum<R, C> = DmTensor<f32, C, Cluster, m![256], m![R, 1 # 8]>;
/// A row group's logits of one vocabulary shard, one contiguous run per slice.
pub type Scores<V, R, C> = Values<bf16, V, R, C>;
/// Each row's token, the first maximum over both shards.
pub type Tokens<R, C> = DmTensor<i32, C, Cluster, m![1 # 256], m![R, 1 # 4]>;

impl<V: AxisName, C: M, R: AxisName, S: Space> Module<(&mut Device, Dm<S, &Scores<V, R, C>>)>
    for Argmax<V, C>
{
    type Output = Tokens<R, C>;

    fn forward(&self, (device, input): (&mut Device, Dm<S, &Scores<V, R, C>>)) -> Self::Output {
        let logits = input.value;
        let maximum: Reduced<f32, R, C> = self.maximum(device, logits);
        // One slice crosses the chip link; the global maximum is spread over the slices locally.
        let peer = maximum
            .chip_shuffle([1, 0])
            .to_dm::<m![R, 1 # 8]>(&mut device.tdma);
        let maximum: Reduced<f32, R, C> = device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(maximum.view(), peer.view())
            .fetch::<m![R / 2, Term, R % 2], m![1 # 8]>()
            .collect::<m![R / 2, Term, R % 2], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, m![R]>()
            .vector_clip_zip(ClipBinaryOpF32::Max)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let maximum: Maximum<R, C> = maximum.to_dm(&mut device.tdma);
        let indices = self.register(device);
        let rank = self.rank(device, (logits, &maximum, &indices));
        let peer = rank
            .chip_shuffle([1, 0])
            .to_dm::<m![R, 1 # 8]>(&mut device.tdma);
        let rank: Reduced<i32, R, C> = device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(rank.view(), peer.view())
            .fetch::<m![R / 2, Term, R % 2], m![1 # 8]>()
            .collect::<m![R / 2, Term, R % 2], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, m![R]>()
            .vector_clip_zip(ClipBinaryOpI32::Max)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        device
            .main
            .begin(rank.view())
            .fetch::<m![R], m![1 # 8]>()
            .collect::<m![R], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, Self::LAST)
            .vector_final()
            .commit_trim::<m![1 # 4]>()
            .commit()
    }
}

impl<V: AxisName, C: M> Argmax<V, C> {
    const LAST: i32 = {
        assert!(C::SIZE == 2 && V::SIZE <= i32::MAX as usize / 2);
        (2 * V::SIZE - 1) as i32
    };

    fn register(&self, device: &mut Device) -> VrfTensor<i32, C, Cluster, Slices<V>, m![V % 256]> {
        let indices: DmTensor<i32, C, Cluster, Slices<V>, m![V % 256]> =
            self.indices.to_dm(&mut device.tdma);
        device
            .sub
            .begin(indices.view())
            .fetch::<m![V % 256 / 8], m![V % 8]>()
            .collect::<m![V % 256 / 8], m![V % 8]>()
            .to_vrf::<m![V % 256]>()
    }
    /// Each row's maximum over this chip's shard, in one slice.
    fn maximum<R: AxisName>(
        &self,
        device: &mut Device,
        logits: &Scores<V, R, C>,
    ) -> Reduced<f32, R, C> {
        device
            .main
            .begin(logits.view())
            .fetch::<m![R, V % 256 / 8], m![V % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![R, V % 256 / 8], m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R, V % 256 / 4], m![V % 4]>()
            .vector_intra_slice_reduce::<V, m![R], m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<m![1 # 256], m![R]>(InterSliceReduceOpF32::Max)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit()
    }

    /// Each row's best rank over this chip's shard, in one slice: a rank orders the global maxima
    /// first and, among them, the smaller index first.
    fn rank<R: AxisName>(
        &self,
        device: &mut Device,
        (logits, maximum, indices): (
            &Scores<V, R, C>,
            &Maximum<R, C>,
            &VrfTensor<i32, C, Cluster, Slices<V>, m![V % 256]>,
        ),
    ) -> Reduced<i32, R, C> {
        // SAFETY: the vocabulary has 256 slices, each holding every row's identical global maximum.
        let maximum: DmTensorView<'_, f32, C, Cluster, Slices<V>, m![R, 1 # 8]> =
            unsafe { maximum.view().reshape() };
        let maximum = device
            .sub
            .begin(maximum)
            .fetch::<m![R], m![1 # 8]>()
            .collect::<m![R], m![1 # 8]>()
            .to_vrf::<m![R, 1 # 8]>();
        let shifted: Values<f32, V, R, C> = device
            .main
            .begin(logits.view())
            .fetch::<m![R, V % 256 / 8], m![V % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![R, V % 256 / 8], m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R, V % 256 / 4], m![V % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &maximum)
            .vector_widen_concat::<m![R, V % 256 / 8], m![V % 8]>()
            .vector_clip(ClipBinaryOpF32::Add, 0.0_f32)
            .vector_final()
            .commit_trim::<m![V % 8]>()
            .commit();
        let ranked: Values<i32, V, R, C> = device
            .main
            .begin(shifted.view())
            .fetch::<m![R, V % 256 / 8], m![V % 8]>()
            .collect::<m![R, V % 256 / 8], m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>()
            .vector_fxp(FxpBinaryOp::AddFxpSat, indices)
            .vector_final()
            .commit_trim::<m![V % 8]>()
            .commit();
        device
            .main
            .begin(ranked.view())
            .fetch::<m![R, V % 256 / 8], m![V % 8]>()
            .collect::<m![R, V % 256 / 8], m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R, V % 256 / 4], m![V % 4]>()
            .vector_intra_slice_reduce::<V, m![R], m![1 # 4]>(IntraSliceReduceOpI32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<m![1 # 256], m![R]>(InterSliceReduceOpI32::Max)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit()
    }
}

mod batch;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::{Tile, Tiles};
    use furiosa_opt_std::runtime::launch;
    axes![Vocabulary = 65536, Rows = 64];
    type Chips = m![2];
    type Greedy = Argmax<Vocabulary, Chips>;

    #[device(chip = 2)]
    fn select(
        device: &mut Device,
        greedy: &Greedy,
        logits: &HbmTensor<bf16, Chips, m![Rows, Vocabulary]>,
    ) -> HbmTensor<i32, Chips, m![Rows, 1 # 4]> {
        let logits: Scores<Vocabulary, Rows, Chips> = logits.to_dm(&mut device.tdma);
        greedy
            .forward((
                &mut *device,
                Dm {
                    value: &logits,
                    tiles: Tiles::<Tile<Rows, Rows>>::over(Rows::SIZE),
                },
            ))
            .to_hbm(&mut device.tdma)
    }

    /// Both ranks select each row's smallest global token index, maxima tied across shards or
    /// within one.
    #[tokio::test(flavor = "current_thread")]
    async fn selects_first_max() {
        // Given
        let peak = |row: usize| (row * 9973 + 17) % (Chips::SIZE * Vocabulary::SIZE);
        let tie = |row: usize| match row % 3 {
            0 => (peak(row) + Vocabulary::SIZE) % (Chips::SIZE * Vocabulary::SIZE),
            1 => (peak(row) + 1) % (Chips::SIZE * Vocabulary::SIZE),
            _ => peak(row),
        };
        let mut values = vec![bf16::from_f32(-2.0); Chips::SIZE * Rows::SIZE * Vocabulary::SIZE];
        let cell = |row: usize, token: usize| {
            (token / Vocabulary::SIZE * Rows::SIZE + row) * Vocabulary::SIZE
                + token % Vocabulary::SIZE
        };
        for row in 0..Rows::SIZE {
            let top = bf16::from_f32(1.0 + row as f32 / 16.0);
            values[cell(row, peak(row))] = top;
            values[cell(row, tie(row))] = top;
        }
        let mut device = Device::new(select.topology()).unwrap();
        let logits = HostTensor::<bf16, m![{ Chips }, Rows, Vocabulary]>::from_vec(values)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let indices = HostTensor::<i32, m![{ Chips }, Vocabulary]>::from_vec(
            (0..Chips::SIZE * Vocabulary::SIZE).map(|index| 131071 - i32::try_from(index).unwrap()),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let greedy = Greedy { indices };

        // When
        let actual = launch(select, (&mut device, &greedy, &logits))
            .await
            .unwrap()
            .to_host::<m![{ Chips }, Rows, 1 # 4]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();

        // Then
        for rank in 0..Chips::SIZE {
            for row in 0..Rows::SIZE {
                let expected = i32::try_from(peak(row).min(tie(row))).unwrap();
                assert_eq!(
                    actual[(rank * Rows::SIZE + row) * 4],
                    expected,
                    "rank={rank} row={row}"
                );
            }
        }
    }
}
