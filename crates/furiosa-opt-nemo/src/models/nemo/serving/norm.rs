use furiosa_opt_std::prelude::*;

use crate::Module;
use crate::models::nemo::serving::{Chips, Hidden, Rows, scale};
use crate::ops::normalization::RmsNorm;

pub type Cluster = m![Rows / 32];
/// Four rows of one 160-wide hidden group a slice.
pub type Slices = m![Rows % 32 / 4, Hidden / 160];
/// The variance's slices after the reduction: every one of a row's 32 feature partitions.
type Reduced = m![Rows % 32 / 4, 32];
const _: () = assert!(Hidden::SIZE == 32 * 160);
/// A slice's four rows, one 80-wide half of its group after the other.
pub type Values =
    DmTensor<bf16, Chips, Cluster, Slices, m![Hidden % 160 / 80, Rows % 4, Hidden % 80]>;
/// The gain landed in every row slice.
pub type Landed = DmTensor<bf16, Chips, Cluster, Slices, m![Hidden % 160]>;
type Inverse = VrfTensor<f32, Chips, Cluster, Slices, m![Rows % 4, 1 # 8]>;
type Gain = VrfTensor<f32, Chips, Cluster, Slices, m![Hidden % 160]>;
/// `Values` normalized, each 16-feature packet of a slice's four rows together, as `Input` keeps
/// them.
pub type Packets =
    DmTensor<bf16, Chips, Cluster, Slices, m![Hidden % 160 / 16, Rows % 4, Hidden % 16]>;
/// The normalized rows in `Packets`' slice order, each slice's 1,280 bytes together: the store
/// moves one run a slice and the head's reload 128-byte ones (four rows of a packet). Writes to
/// HBM cost by the element, so the store's few long runs matter more than the reload's.
pub type Input = HbmTensor<
    bf16,
    Chips,
    m![
        Rows / 32,
        Rows % 32 / 4,
        Hidden / 160,
        Hidden % 160 / 16,
        Rows % 4,
        Hidden % 16
    ],
>;

/// Each row's inverse RMS, replicated over its feature partitions; `EPSILON` holds FP32 bits.
fn inverse<const EPSILON: usize>(device: &mut Device, input: &Values) -> Inverse {
    let variance: DmTensor<f32, Chips, Cluster, Reduced, m![Rows % 4, 1 # 8]> = device
        .main
        .begin(input.view())
        .fetch::<m![Rows % 4, Hidden % 160 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows % 4, Hidden % 160 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows % 4, Hidden % 160 / 4], m![Hidden % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Hidden, m![Rows % 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(5120.0_f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, f32::from_bits(EPSILON as u32) / 32.0)
        .vector_inter_slice_reduce::<Reduced, m![Rows % 4]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // SAFETY: all feature partitions carry the same reduced row scalar in wire order.
    let variance: DmTensor<f32, Chips, Cluster, Slices, m![Rows % 4, 1 # 8]> =
        unsafe { variance.reshape() };
    device
        .main
        .begin(variance.view())
        .fetch::<m![Rows % 4], m![1 # 2]>()
        .collect::<m![Rows % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0_f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf::<m![Rows % 4, 1 # 8]>(&mut device.sub)
}

fn gain(device: &mut Device, weight: &Landed) -> Gain {
    device
        .sub
        .begin(weight.view())
        .fetch::<m![Hidden % 160 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Hidden % 160 / 8], m![Hidden % 8]>()
        .to_vrf::<m![Hidden % 160]>()
}

/// The final norm of the selected rows, four rows of a 160-wide group a slice.
impl<W: AxisName, P: M, const EPSILON: usize> Module<(&mut Device, &Values)>
    for RmsNorm<W, EPSILON, P>
{
    type Output = Packets;

    fn forward(&self, (device, input): (&mut Device, &Values)) -> Packets {
        let inverse = inverse::<EPSILON>(device, input);
        const {
            assert!(W::SIZE == Hidden::SIZE && P::SIZE == Chips::SIZE);
        }
        // SAFETY: both mappings are contiguous features on the same number of chips, checked above.
        let weight = unsafe { self.weight().tensor().view().reshape::<Chips, m![Hidden]>() };
        let weight: Landed = weight.to_dm(&mut device.tdma);
        let weight = gain(device, &weight);
        device
        .main
        .begin(input.view())
        .fetch::<m![Rows % 4 / 2, Hidden % 160 / 8, Rows % 2], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows % 4 / 2, Hidden % 160 / 8, Rows % 2], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<
            m![Rows % 4 / 2, Hidden % 160 / 8, Rows % 2, Hidden % 8 / 4],
            m![Hidden % 4],
        >()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inverse)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight)
        .vector_widen_concat::<
            m![Rows % 4 / 2, Hidden % 160 / 8, Rows % 2],
            m![Hidden % 8],
        >()
        .vector_final()
        .cast::<bf16, m![Hidden % 8 # 16]>()
        .commit_trim::<m![Hidden % 8]>()
        .commit()
    }
}

/// The post-attention norm, eight rows of one 160-wide group a slice: its copy and its activation's
/// store move 1,280-byte runs of 256-byte entries, where four rows a slice move slow 640-byte ones.
pub mod eight {
    use super::*;

    pub type Slices = m![Rows % 32 / 8 # 8, Hidden / 160];
    type Reduced = m![Rows % 32 / 8 # 8, 32];
    /// A slice's eight rows, one 80-wide half of its group after the other: each half is one
    /// residual slice's 1,280 bytes.
    pub type Values =
        DmTensor<bf16, Chips, Cluster, Slices, m![Hidden % 160 / 80, Rows % 8, Hidden % 80]>;
    /// The gain landed in every row slice, once for all the waves a layer normalizes.
    pub type Landed = DmTensor<bf16, Chips, Cluster, Slices, m![Hidden % 160]>;
    pub type Normalized<D> = DmTensor<D, Chips, Cluster, Slices, m![Rows % 8, Hidden % 160]>;

    /// RMS-normalizes one wave, applies the gain and rounds straight to FP8 by `scale`.
    pub fn quantize(
        device: &mut Device,
        input: &Values,
        (norm, scale): (
            &RmsNorm<Hidden, { 1e-5_f32.to_bits() as usize }, Chips, Landed>,
            &scale::InputScale<Chips, Cluster, Slices>,
        ),
    ) -> Normalized<f8e4m3> {
        let input = crate::ops::layout::Dm::<crate::compose::Tile<Rows, Rows>, _> {
            tiles: crate::compose::Tiles::over(Rows::SIZE),
            value: input.view(),
        };
        let norm =
            RmsNorm::<Hidden, { 1e-5_f32.to_bits() as usize }, Chips, _>::new(norm.weight().view());
        let inverse = norm
            .replicated::<_, m![Rows % 8], m![Hidden % 160], Reduced, m![1 # 8], true, _>(
                device, &input,
            )
            .inverse(device);
        let gain = norm.gain::<m![Hidden % 160 / 8], m![Hidden % 8]>(device);
        inverse.apply::<m![Rows % 8 / 2, Hidden % 160 / 8, Rows % 2], m![Hidden % 8], m![Rows % 8 / 2, Hidden % 160 / 8, Rows % 2, Hidden % 8 / 4]>(device, &gain)
            .value
            .vector_fp_div(scale.input_scale())
            .vector_widen_concat::<
                m![Rows % 8 / 2, Hidden % 160 / 8, Rows % 2],
                m![Hidden % 8],
            >()
            .vector_final()
            .cast::<f8e4m3, m![Hidden % 8 # 32]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_std::runtime::launch;

    #[device(chip = 2)]
    fn normalize(
        device: &mut Device,
        input: &Input,
        norm: &RmsNorm<Hidden, { 1e-5_f32.to_bits() as usize }, Chips>,
    ) -> Input {
        let input: Values = input.to_dm(&mut device.tdma);
        norm.forward((&mut *device, &input))
            .to_hbm(&mut device.tdma)
    }

    /// Each rank normalizes every feature with its complete row's mean square and gain.
    #[tokio::test(flavor = "current_thread")]
    async fn normalizes_each_row() {
        // Given
        let values: Vec<_> = (0..Chips::SIZE * Rows::SIZE * Hidden::SIZE)
            .map(|index| {
                let value = i16::try_from(index % 193).unwrap() - 96;
                bf16::from_f32(f32::from(value) / 4096.0)
            })
            .collect();
        let gains: Vec<_> = (0..Chips::SIZE * Hidden::SIZE)
            .map(|index| bf16::from_f32(0.5 + f32::from(u16::try_from(index % 17).unwrap()) / 16.0))
            .collect();
        let mut device = Device::new(normalize.topology()).unwrap();
        let input = HostTensor::<bf16, m![{ Chips }, Rows, Hidden]>::from_vec(values.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let weight = HostTensor::<bf16, m![{ Chips }, Hidden]>::from_vec(gains.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let norm = RmsNorm::new(crate::ops::weight::Weight::new(weight));

        // When
        let actual = launch(normalize, (&mut device, &input, &norm))
            .await
            .unwrap()
            .to_host::<m![{ Chips }, Rows, Hidden]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();

        // Then
        for (row, (values, actual)) in values
            .chunks_exact(Hidden::SIZE)
            .zip(actual.chunks_exact(Hidden::SIZE))
            .enumerate()
        {
            let gain = &gains[row / Rows::SIZE * Hidden::SIZE..][..Hidden::SIZE];
            let inverse = (values
                .iter()
                .map(|value| f64::from(value.to_f32()).powi(2))
                .sum::<f64>()
                / f64::from(u32::try_from(Hidden::SIZE).unwrap())
                + 1e-5)
                .sqrt()
                .recip();
            let (error, energy) = values.iter().zip(actual).zip(gain).fold(
                (0.0, 0.0),
                |(error, energy), ((value, actual), gain)| {
                    let expected = f64::from(value.to_f32()) * inverse * f64::from(gain.to_f32());
                    (
                        error + (f64::from(actual.to_f32()) - expected).powi(2),
                        energy + expected.powi(2),
                    )
                },
            );
            let relative = (error / energy).sqrt();
            assert!(
                relative < 1e-2,
                "rank={} row={} rel_l2={relative}",
                row / Rows::SIZE,
                row % Rows::SIZE
            );
        }
    }
}
