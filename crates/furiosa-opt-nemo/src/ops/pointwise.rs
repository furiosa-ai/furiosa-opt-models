//! Elementwise FP32 arithmetic with source-dtype output over bounded device-memory windows.

use furiosa_opt_std::prelude::*;

/// Flattened logical element count, independent of the input's named axes.
#[derive(Clone, Debug)]
pub(crate) struct Flat<S>(std::marker::PhantomData<S>);

impl<S: M + 'static> AxisName for Flat<S> {
    const NAME: Ident = Ident::new("Flat");
    const SIZE: usize = S::SIZE;
}

/// Elements per slice, an eight-element multiple bounded by 512.
#[derive(Clone, Debug)]
pub struct Part<S>(std::marker::PhantomData<S>);

impl<S: M + 'static> AxisName for Part<S> {
    const NAME: Ident = Ident::new("Part");

    const SIZE: usize = {
        assert!(
            S::SIZE > 0 && S::SIZE.is_multiple_of(8),
            "pointwise tensors require complete eight-element vectors"
        );
        let minimum = S::SIZE.div_ceil(256).next_multiple_of(8);
        let mut part = if minimum < 512 { minimum } else { 512 };
        while S::SIZE % part != 0 {
            part -= 8;
        }
        part
    };
}

/// One transfer window covering at most 256 equal slices and dividing the flattened input.
#[derive(Clone, Debug)]
pub struct Window<S>(std::marker::PhantomData<S>);

impl<S: M + 'static> AxisName for Window<S> {
    const NAME: Ident = Ident::new("Window");

    const SIZE: usize = {
        let parts = S::SIZE / Part::<S>::SIZE;
        let mut lanes = if parts < 256 { parts } else { 256 };
        while parts % lanes != 0 {
            lanes -= 1;
        }
        lanes * Part::<S>::SIZE
    };
}

type Width<S> = Symbol<Flat<S>>;
type Extent<S> = Symbol<Window<S>>;
type Segment<S> = Symbol<Part<S>>;
type Batch<S> = m![{ Width<S> } / ({ Extent<S> }) = 1 # ({ Width<S> } / ({ Extent<S> })), { Width<S> } % ({ Extent<S> })];
type BatchMut<S> = m![{ Width<S> } / ({ Extent<S> }) = 1 #{!} ({ Width<S> } / ({ Extent<S> })), { Width<S> } % ({ Extent<S> })];
type Slices<S> = m![{ Width<S> } % ({ Extent<S> }) / ({ Segment<S> }) # 256];
type Elements<S> = m![{ Width<S> } % ({ Segment<S> })];
type Rows<D, S, Chips = m![1]> = DmTensor<D, Chips, m![1 # 2], Slices<S>, Elements<S>>;
type Registers<S, Chips = m![1]> = VrfTensor<f32, Chips, m![1 # 2], Slices<S>, Elements<S>>;

/// Scalars widened to FP32 for arithmetic and converted back at the window boundary.
pub(crate) trait Float: MaterializableScalar + FetchCast<f32> {
    fn narrow<S: M + 'static, Chips: M>(
        device: &mut Device,
        input: Rows<f32, S, Chips>,
    ) -> Rows<Self, S, Chips>;
}

impl Float for f32 {
    fn narrow<S: M + 'static, Chips: M>(
        _: &mut Device,
        input: Rows<f32, S, Chips>,
    ) -> Rows<Self, S, Chips> {
        input
    }
}

impl Float for bf16 {
    fn narrow<S: M + 'static, Chips: M>(
        device: &mut Device,
        input: Rows<f32, S, Chips>,
    ) -> Rows<Self, S, Chips> {
        device
            .main
            .begin(input.view())
            .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .cast::<bf16, m![{ Width<S> } % 8 # 16]>()
            .commit_trim::<m![{ Width<S> } % 8]>()
            .commit()
    }
}

impl<S: M + 'static> Window<S> {
    pub(in crate::ops) fn combine<D: Float, const MULTIPLY: bool, Chips: M>(
        device: &mut Device,
        lhs: &HbmTensor<D, Chips, S>,
        rhs: &HbmTensor<D, Chips, S>,
    ) -> HbmTensor<D, Chips, S> {
        let mut result = HbmTensor::<D, Chips, m![{ Width<S> }]>::new();
        for index in 0..S::SIZE / Window::<S>::SIZE {
            let lhs: Rows<D, S, Chips> = unsafe { lhs.view().reshape::<Chips, m![{ Width<S> }]>() }
                .tile::<m![{ Width<S> } / ({ Extent<S> })], 1, Batch<S>>(index)
                .to_dm(&mut device.tdma);
            let rhs: Rows<D, S, Chips> = unsafe { rhs.view().reshape::<Chips, m![{ Width<S> }]>() }
                .tile::<m![{ Width<S> } / ({ Extent<S> })], 1, Batch<S>>(index)
                .to_dm(&mut device.tdma);
            let rhs: Registers<S, Chips> = device
                .sub
                .begin(rhs.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .to_vrf();
            let output: Rows<f32, S, Chips> = device
            .main
            .begin(lhs.view())
            .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Width<S> } % ({ Segment<S> }) / 4], m![{ Width<S> } % 4]>()
            .vector_fp_binary(
                if MULTIPLY {
                    FpBinaryOp::MulF(FpMulAlu::Mul0)
                } else {
                    FpBinaryOp::AddF
                },
                &rhs,
            )
            .vector_widen_concat::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .vector_final()
            .commit_trim::<m![{ Width<S> } % 8]>()
            .commit();
            D::narrow(device, output).view().to_hbm_view(
                &mut device.tdma,
                result
                    .view_mut()
                    .tile::<m![{ Width<S> } / ({ Extent<S> })], 1, BatchMut<S>>(index),
            );
        }
        // SAFETY: flattening preserves the contiguous element order.
        unsafe { result.reshape() }
    }
}
