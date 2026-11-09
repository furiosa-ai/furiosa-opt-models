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
