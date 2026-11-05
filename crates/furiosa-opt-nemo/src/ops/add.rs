use crate::Module;
use crate::compose::Space;
use crate::ops::layout::{Dm, axes::Chip};
use crate::ops::normalization::{Chunk, Cluster, Element, FeatureSlices, Spread};
use crate::ops::pointwise::{self, Float};
use furiosa_opt_std::prelude::*;

pub(crate) mod clusters;

/// Elementwise sum of two tensors in the same layout.
#[derive(Default, DeviceSend)]
pub struct Add;

impl Add {
    /// Round the resident sum while leaving its destination and commit point to the composer.
    pub(crate) fn stage<'d, 'x, S: Space, Chips: M, C: M, L: M, E: M, T: M, P: M>(
        &self,
        device: &'d mut Device,
        input: Dm<
            S,
            (
                DmTensorView<'x, bf16, Chips, C, L, E>,
                DmTensorView<'x, bf16, Chips, C, L, E>,
            ),
        >,
    ) -> Dm<S, CommitTrimTensor<'d, { Tu::Main }, bf16, Chips, C, L, T, P>>
    where
        'x: 'd,
    {
        axes![Term = 2];
        const { assert!(T::SIZE % 20 == 0) };
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin_interleaved::<Term, _, _, _, _, _>(input.value.0, input.value.1)
                .fetch::<m![{ T } / 20, Term, { T } % 20], P>()
                .fetch_cast::<f32>()
                .collect::<m![{ T } / 20, Term, { T } % 20], P>()
                .vector_init()
                .vector_intra_slice_unzip::<Term, T>()
                .vector_clip_zip(ClipBinaryOpF32::Add)
                .vector_final()
                .cast::<bf16, m![{ P } # 16]>()
                .commit_trim::<P>(),
        }
    }
}

// --8<-- [start:add]
/// HBM inputs and output retain their element type and physical layout.
impl<D: Float, S: M + 'static, Chips: M>
    Module<(
        &mut Device,
        (&HbmTensor<D, Chips, S>, &HbmTensor<D, Chips, S>),
    )> for Add
{
    type Output = HbmTensor<D, Chips, S>;

    fn forward(
        &self,
        (device, (left, right)): (
            &mut Device,
            (&HbmTensor<D, Chips, S>, &HbmTensor<D, Chips, S>),
        ),
    ) -> Self::Output {
        pointwise::Window::<S>::combine::<_, false, Chips>(device, left, right)
    }
}

/// DM rows accumulate in f32 and round the result to bf16.
impl<'d, 'x, 'y, Hidden: AxisName, Rows: AxisName>
    Module<(
        &'d mut Device,
        (&'x Spread<Hidden, Rows>, &'y Spread<Hidden, Rows>),
    )> for Add
{
    type Output = Spread<Hidden, Rows>;

    fn forward(
        &self,
        (device, (lhs, rhs)): (
            &'d mut Device,
            (&'x Spread<Hidden, Rows>, &'y Spread<Hidden, Rows>),
        ),
    ) -> Self::Output {
        let rhs: VrfTensor<f32, Chip, Cluster, FeatureSlices<Hidden>, Element<Hidden, Rows>> =
            device
                .sub
                .begin(rhs.view())
                .fetch::<m![Rows, Hidden % ({ Symbol<Chunk<Hidden>> }) / 8], m![Hidden % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Rows, Hidden % ({ Symbol<Chunk<Hidden>> }) / 8], m![Hidden % 8]>()
                .to_vrf();
        device
            .main
            .begin(lhs.view())
            .fetch::<m![Rows, Hidden % ({ Symbol<Chunk<Hidden>> }) / 8], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows, Hidden % ({ Symbol<Chunk<Hidden>> }) / 8], m![Hidden % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &rhs)
            .vector_final()
            .cast::<bf16, m![Hidden % 8 # 16]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit()
    }
}

/// Both terms are rows of one launch; the sum keeps their live tiles.
impl<'d, 'x, 'y, Hidden: AxisName, S: Space>
    Module<(
        &'d mut Device,
        (
            &'x Dm<S, Spread<Hidden, S::Envelope>>,
            &'y Dm<S, Spread<Hidden, S::Envelope>>,
        ),
    )> for Add
{
    type Output = Dm<S, Spread<Hidden, S::Envelope>>;

    fn forward(
        &self,
        (device, (lhs, rhs)): (
            &'d mut Device,
            (
                &'x Dm<S, Spread<Hidden, S::Envelope>>,
                &'y Dm<S, Spread<Hidden, S::Envelope>>,
            ),
        ),
    ) -> Self::Output {
        Dm {
            value: self.forward((device, (&lhs.value, &rhs.value))),
            tiles: lhs.tiles,
        }
    }
}

// --8<-- [end:add]

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use crate::ops::normalization::{RowMajor, Spread};

    use super::Add;
    use crate::Module;
    use crate::ops::layout::axes::Chip;
    use crate::ops::layout::axes::Eight;

    axes![Pinned = 1024];

    type Rows = HbmTensor<bf16, Chip, RowMajor<Pinned, Eight>>;

    fn terms(step: usize) -> Vec<bf16> {
        (0..Eight::SIZE * Pinned::SIZE)
            .map(|index| bf16::from_f32(((index * step) % 101) as f32 / 25.0 - 2.0))
            .collect()
    }

    /// The DM-resident add, round-tripped through HBM so the host can read it.
    #[device(chip = 1)]
    fn summed(device: &mut Device, lhs: &Rows, rhs: &Rows, output: &mut Rows) {
        let lhs: Spread<Pinned, Eight> = lhs.to_dm(&mut device.tdma);
        let rhs: Spread<Pinned, Eight> = rhs.to_dm(&mut device.tdma);
        Add.forward((&mut *device, (&lhs, &rhs)))
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn adds_elementwise() {
        let mut device = Device::new(summed.topology()).unwrap();
        let (lhs, rhs) = (terms(37), terms(53));
        let expected: Vec<f32> = lhs
            .iter()
            .zip(&rhs)
            .map(|(a, b)| a.to_f32() + b.to_f32())
            .collect();
        let lhs = HostTensor::<bf16, RowMajor<Pinned, Eight>>::from_vec(lhs)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let rhs = HostTensor::<bf16, RowMajor<Pinned, Eight>>::from_vec(rhs)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut output = HostTensor::<bf16, RowMajor<Pinned, Eight>>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        launch(summed, (&mut device, &lhs, &rhs, &mut output))
            .await
            .unwrap();
        let got = output
            .to_host::<RowMajor<Pinned, Eight>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();

        for (index, (got, expected)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (got.to_f32() - expected).abs() <= 0.02,
                "index {index}: {} vs {expected}",
                got.to_f32()
            );
        }
    }
}
