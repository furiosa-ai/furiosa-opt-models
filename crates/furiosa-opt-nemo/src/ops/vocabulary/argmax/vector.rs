use furiosa_opt_std::prelude::*;

use crate::{
    Module,
    ops::{pointwise, vector::scalar},
};

use crate::ops::layout::{BroadcastClusters as Cluster, axes::Chip};

type Segment<V> = Symbol<Part<V>>;
type Window<V> = Symbol<Extent<V>>;
type Slices<V> = m![V % ({ Window<V> }) / ({ Segment<V> }) # 256];
pub type Elements<V> = m![V / ({ Window<V> }), V % ({ Segment<V> })];
type Time<V> = m![V / ({ Window<V> }), V % ({ Segment<V> }) / 8];
type Segments<V> = m![V / ({ Window<V> }), V % ({ Segment<V> }) / 4];
type Rows<D, V, Chips = Chip> = DmTensor<D, Chips, Cluster, Slices<V>, Elements<V>>;
type Reduced<D, Chips = Chip> = DmTensor<D, Chips, Cluster, m![1 # 256], m![1 # 8]>;

/// First maximum in a finite, contiguous vocabulary vector. The index tensor is bound once.
#[derive(DeviceSend)]
pub struct Argmax<V: AxisName, Chips: M = Chip> {
    /// Reversed token IDs V::SIZE - 1 - token; larger ranks choose the first tied index.
    pub indices: HbmTensor<i32, Chips, m![V]>,
}

impl<V: AxisName, Chip: M> Module<(&mut Device, &HbmTensor<bf16, Chip, m![V]>)>
    for Argmax<V, Chip>
{
    type Output = HbmTensor<i32, Chip, m![1 # 4]>;

    fn forward(
        &self,
        (device, logits): (&mut Device, &HbmTensor<bf16, Chip, m![V]>),
    ) -> Self::Output {
        let logits: Rows<bf16, V, Chip> = logits.to_dm(&mut device.tdma);
        let maximum: Reduced<f32, Chip> = device
            .main
            .begin(logits.view())
            .fetch::<Time<V>, m![V % 8]>()
            .fetch_cast::<f32>()
            .collect::<Time<V>, m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Segments<V>, m![V % 4]>()
            .vector_intra_slice_reduce::<V, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<m![1 # 256], m![1]>(InterSliceReduceOpF32::Max)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        // Every reduced slice holds the same scalar; restore the original slice labels.
        let maximum: DmTensor<f32, Chip, Cluster, Slices<V>, m![1 # 8]> =
            unsafe { maximum.reshape() };
        let maximum = scalar::register(device, &maximum);
        let shifted: Rows<f32, V, Chip> = device
            .main
            .begin(logits.view())
            .fetch::<Time<V>, m![V % 8]>()
            .fetch_cast::<f32>()
            .collect::<Time<V>, m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Segments<V>, m![V % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &maximum)
            .vector_widen_concat::<Time<V>, m![V % 8]>()
            // Canonicalize -0 so equal signed zeros obey the same first-index rule.
            .vector_clip(ClipBinaryOpF32::Add, 0.0_f32)
            .vector_final()
            .commit_trim::<m![V % 8]>()
            .commit();
        let indices: Rows<i32, V, Chip> = self.indices.to_dm(&mut device.tdma);
        let indices: VrfTensor<i32, Chip, Cluster, Slices<V>, Elements<V>> = device
            .sub
            .begin(indices.view())
            .fetch::<Time<V>, m![V % 8]>()
            .collect::<Time<V>, m![V % 8]>()
            .to_vrf();
        // Zero differences rank by reverse index. Negative differences remain negative even
        // after adding the largest index, including a subtraction that overflows to -infinity.
        let ranked: Rows<i32, V, Chip> = device
            .main
            .begin(shifted.view())
            .fetch::<Time<V>, m![V % 8]>()
            .collect::<Time<V>, m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>()
            .vector_fxp(FxpBinaryOp::AddFxpSat, &indices)
            .vector_final()
            .commit_trim::<m![V % 8]>()
            .commit();
        let rank: Reduced<i32, Chip> = device
            .main
            .begin(ranked.view())
            .fetch::<Time<V>, m![V % 8]>()
            .collect::<Time<V>, m![V % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Segments<V>, m![V % 4]>()
            .vector_intra_slice_reduce::<V, m![1], m![1 # 4]>(IntraSliceReduceOpI32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<m![1 # 256], m![1]>(InterSliceReduceOpI32::Max)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let token: DmTensor<i32, Chip, Cluster, m![1 # 256], m![1 # 4]> = device
            .main
            .begin(rank.view())
            .fetch::<m![1], m![1 # 8]>()
            .collect::<m![1], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp_with_mode(
                FxpBinaryOp::SubFxp,
                BinaryArgMode::Mode10,
                const { V::SIZE as i32 - 1 },
            )
            .vector_final()
            .commit_trim::<m![1 # 4]>()
            .commit();
        token.to_hbm(&mut device.tdma)
    }
}

/// One slice's elements of a transfer window: the pointwise part, unless its windows fill fewer
/// than 256 slices and 256 slices of whole vectors hold the vocabulary, as one window.
#[derive(Clone, Debug)]
pub struct Part<V>(std::marker::PhantomData<V>);

impl<V: AxisName> AxisName for Part<V> {
    const NAME: Ident = Ident::new("Part");
    const SIZE: usize = if spread::<V>() {
        V::SIZE / 256
    } else {
        pointwise::Part::<m![V]>::SIZE
    };
}

/// The elements of one transfer window, which every slice shares.
#[derive(Clone, Debug)]
pub struct Extent<V>(std::marker::PhantomData<V>);

impl<V: AxisName> AxisName for Extent<V> {
    const NAME: Ident = Ident::new("Window");
    const SIZE: usize = if spread::<V>() {
        V::SIZE
    } else {
        pointwise::Window::<m![V]>::SIZE
    };
}

/// A vocabulary whose pointwise windows fill fewer than 256 slices, but which 256 slices of whole
/// eight-element vectors hold within the index operand's 1024 elements per slice.
const fn spread<V: AxisName>() -> bool {
    pointwise::Window::<m![V]>::SIZE / pointwise::Part::<m![V]>::SIZE < 256
        && V::SIZE.is_multiple_of(256 * 8)
        && V::SIZE / 256 <= 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![Vocabulary = 262144];

    #[device(chip = 1)]
    fn argmax(
        device: &mut Device,
        selector: &Argmax<Vocabulary>,
        logits: &HbmTensor<bf16, Chip, m![Vocabulary]>,
    ) -> HbmTensor<i32, Chip, m![1 # 4]> {
        selector.forward((device, logits))
    }

    /// Finite BF16 maxima select the first index across signed-zero ties and window boundaries.
    #[tokio::test(flavor = "current_thread")]
    async fn selects_first_finite_maximum() {
        let mut device = Device::new(argmax.topology()).unwrap();
        let indices = HostTensor::<i32, m![Vocabulary]>::from_vec(
            (0..Vocabulary::SIZE).map(|i| (Vocabulary::SIZE - 1 - i) as i32),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let selector = Argmax::<Vocabulary> { indices };
        let cases: &[(&[(usize, f32)], usize)] = &[
            (&[(Vocabulary::SIZE - 1, 3.0)], Vocabulary::SIZE - 1),
            (&[(7, 2.0), (131_079, 2.0)], 7),
            (&[(1, -0.0), (Vocabulary::SIZE - 1, 0.0)], 1),
            (&[(1, 0.0), (Vocabulary::SIZE - 1, -0.0)], 1),
            (
                &[(0, bf16::from_bits(0x3f7f).to_f32()), (131_072, 1.0)],
                131_072,
            ),
            (&[(0, -0.5), (Vocabulary::SIZE - 1, -0.5)], 0),
            (&[(123, bf16::from_bits(0x7f7f).to_f32())], 123),
        ];
        for &(updates, expected) in cases {
            let mut values = vec![bf16::from_bits(0xff7f); Vocabulary::SIZE];
            for &(index, value) in updates {
                values[index] = bf16::from_f32(value);
            }
            let logits = HostTensor::<bf16, m![Vocabulary]>::from_vec(values)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let selected = launch(argmax, (&mut device, &selector, &logits))
                .await
                .unwrap()
                .to_host::<m![1 # 4]>(&mut device.pdma)
                .await
                .unwrap();
            assert_eq!(
                selected.into_vec()[0],
                expected as i32,
                "updates: {updates:?}"
            );
        }
    }
}
