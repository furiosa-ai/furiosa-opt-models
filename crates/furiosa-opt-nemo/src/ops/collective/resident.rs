use super::AllReduce;
use crate::{
    compose::Space,
    ops::layout::{Dm, axes::Parallel},
};
use furiosa_opt_std::prelude::*;

axes![Term = 2];

impl AllReduce<Parallel<2>> {
    /// Keep the FP32 sum unrounded for the projection's retained scale completion.
    pub(crate) fn sum<'d, 'x, S: Space, Chips: M, C: M, L: M, E: M, T: M, P: M>(
        &self,
        device: &'d mut Device,
        input: Dm<
            S,
            (
                DmTensorView<'x, bf16, Chips, C, L, E>,
                DmTensorView<'x, bf16, Chips, C, L, E>,
            ),
        >,
    ) -> Dm<
        S,
        VectorFpTensor<
            'd,
            { Tu::Main },
            f32,
            Chips,
            C,
            L,
            m![{ T }, { P } / 4],
            m![{ P } % 4],
            Fresh,
            { stage::VeOrder::IntraFirst },
            stage::Zipped,
            { stage::Way::Way4 },
        >,
    >
    where
        'x: 'd,
    {
        const { assert!(Chips::SIZE == 2 && T::SIZE % 20 == 0) };
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
                .vector_narrow_split::<m![{ T }, { P } / 4], m![{ P } % 4]>()
                .vector_fp_zip(FpBinaryOp::AddF),
        }
    }

    pub(crate) fn complete<S: Space, Chips: M, C: M, L: M, E: M, T: M, P: M>(
        &self,
        device: &mut Device,
        input: Dm<
            S,
            (
                DmTensorView<'_, bf16, Chips, C, L, E>,
                DmTensorView<'_, bf16, Chips, C, L, E>,
            ),
        >,
    ) -> Dm<S, DmTensor<bf16, Chips, C, L, m![{ T }, { P }]>> {
        const { assert!(Chips::SIZE == 2 && T::SIZE % 20 == 0) };
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
                .commit_trim::<P>()
                .commit(),
        }
    }
}

impl AllReduce<Parallel<2>> {
    /// Issue the peer transfer at the composer's point, into its retained destination.
    pub fn stage<S: Space, D: MaterializableScalar, Chips: M, C: M, L: M, E: M, OL: M, O: M>(
        &self,
        device: &mut Device,
        input: Dm<
            S,
            (
                DmTensorView<'_, D, Chips, C, L, E>,
                DmTensorViewMut<'_, D, Chips, C, OL, O>,
            ),
        >,
    ) -> Dm<S, ()> {
        const { assert!(Chips::SIZE == 2) };
        input
            .value
            .0
            .chip_shuffle([1, 0])
            .to_dm_view(&mut device.tdma, input.value.1);
        Dm {
            value: (),
            tiles: input.tiles,
        }
    }
}
