//! Classifier-free guidance over complete predictions or the channels preceding learned
//! variance.

use super::{Flat, Part, Window};
use crate::Op;
use crate::tensor::scalar;
use furiosa_opt_std::prelude::*;
use std::{borrow::Borrow, marker::PhantomData};

/// FP32 values of a sample or a prediction.
type Values<S> = HbmTensor<f32, m![1], S>;

/// The guidance scale, padded to eight cells.
pub type Factor = HbmTensor<f32, m![1], m![1 # 8]>;

#[derive(DeviceSend)]
pub struct Combine<Prediction, Sample>(PhantomData<(Prediction, Sample)>);

impl<Prediction, Sample> Default for Combine<Prediction, Sample> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

#[derive(Clone, Debug)]
struct Planes<P, S>(PhantomData<(P, S)>);

impl<P: M + 'static, S: M + 'static> AxisName for Planes<P, S> {
    const NAME: Ident = Ident::new("Planes");

    const SIZE: usize = {
        assert!(S::SIZE > 0 && (P::SIZE == S::SIZE || P::SIZE == 2 * S::SIZE));
        P::SIZE / S::SIZE
    };
}

type Width<S> = Symbol<Flat<S>>;
type Plane<P, S> = Symbol<Planes<P, S>>;
type Extent<S> = Symbol<Window<S>>;
type Segment<S> = Symbol<Part<S>>;

type Batch<S> =
    m![{Width<S>} / ({Extent<S>}) = 1 # ({Width<S>} / ({Extent<S>})), {Width<S>} % ({Extent<S>})];

type BatchMut<S> = m![
    {Width<S>} / ({Extent<S>}) = 1 #{!} ({Width<S>} / ({Extent<S>})),
    {Width<S>} % ({Extent<S>})
];
type Slices<S> = m![{Width<S>} % ({Extent<S>}) / ({Segment<S>}) # 256];
type Elements<S> = m![{Width<S>} % ({Segment<S>})];
type Rows<S> = DmTensor<f32, m![1], m![1 # 2], Slices<S>, Elements<S>>;
type Registers<S> = VrfTensor<f32, m![1], m![1 # 2], Slices<S>, Elements<S>>;

impl<Prediction: M + 'static, Sample: M + 'static, G: Borrow<Factor>>
    Op<(&[Values<Prediction>; 2], G)> for Combine<Prediction, Sample>
{
    type Output = Values<Sample>;

    fn forward(
        &self,
        device: &mut Device,
        (predictions, scale): (&[Values<Prediction>; 2], G),
    ) -> Self::Output {
        let mut result = Values::<m![{Width<Sample>}]>::new();
        for index in 0..Sample::SIZE / Window::<Sample>::SIZE {
            // Indexed borrows, since `each_ref` builds raw pointers that device code cannot lower.
            let [conditional, unconditional]: [Rows<Sample>; 2] =
                [&predictions[0], &predictions[1]].map(|prediction| {
                    // SAFETY: Planes splits the same contiguous cells into whole sample planes.
                    unsafe {
                        prediction
                            .view()
                            .reshape::<m![1], m![{Plane<Prediction, Sample>}, {Width<Sample>}]>()
                    }
                    .tile::<m![{Plane<Prediction, Sample>}], 1, m![
                        {Plane<Prediction, Sample>} = 1 # ({Plane<Prediction, Sample>}),
                        {Width<Sample>}
                    ]>(0)
                    .tile::<m![{Width<Sample>} / ({Extent<Sample>})], 1, m![
                        {Plane<Prediction, Sample>} = 1 # ({Plane<Prediction, Sample>}),
                        {Batch<Sample>}
                    ]>(index)
                    .to_dm(&mut device.tdma)
                });
            let unconditional: Registers<Sample> = device
                .sub
                .begin(unconditional.view())
                .fetch::<m![{Width<Sample>} % ({Segment<Sample>}) / 8], m![{Width<Sample>} % 8]>()
                .collect::<m![{Width<Sample>} % ({Segment<Sample>}) / 8], m![{Width<Sample>} % 8]>()
                .to_vrf();
            let scale: DmTensor<f32, m![1], m![1 # 2], Slices<Sample>, m![1 # 8]> =
                scale.borrow().to_dm(&mut device.tdma);
            let scale = scalar::register(device, &scale);
            let output: Rows<Sample> = device
                .main
                .begin(conditional.view())
                .fetch::<m![{Width<Sample>} % ({Segment<Sample>}) / 8], m![{Width<Sample>} % 8]>()
                .collect::<m![{Width<Sample>} % ({Segment<Sample>}) / 8], m![{Width<Sample>} % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<
                    m![{Width<Sample>} % ({Segment<Sample>}) / 4],
                    m![{Width<Sample>} % 4],
                >()
                .vector_fp_binary(FpBinaryOp::SubF, &unconditional)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
                .vector_widen_concat::<
                    m![{Width<Sample>} % ({Segment<Sample>}) / 8],
                    m![{Width<Sample>} % 8],
                >()
                .vector_final()
                .commit_trim::<m![{Width<Sample>} % 8]>()
                .commit();
            let output: Rows<Sample> = device
                .main
                .begin(output.view())
                .fetch::<m![{Width<Sample>} % ({Segment<Sample>}) / 8], m![{Width<Sample>} % 8]>()
                .collect::<m![{Width<Sample>} % ({Segment<Sample>}) / 8], m![{Width<Sample>} % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<
                    m![{Width<Sample>} % ({Segment<Sample>}) / 4],
                    m![{Width<Sample>} % 4],
                >()
                .vector_fp_binary(FpBinaryOp::AddF, &unconditional)
                .vector_widen_concat::<
                    m![{Width<Sample>} % ({Segment<Sample>}) / 8],
                    m![{Width<Sample>} % 8],
                >()
                .vector_final()
                .commit_trim::<m![{Width<Sample>} % 8]>()
                .commit();
            output.view().to_hbm_view(
                &mut device.tdma,
                result
                    .view_mut()
                    .tile::<m![{Width<Sample>} / ({Extent<Sample>})], 1, BatchMut<Sample>>(index),
            );
        }
        // SAFETY: only the grouping changes; contiguous prediction element order is preserved.
        unsafe { result.reshape() }
    }
}

impl<Sample: M, G> Op<([Values<Sample>; 1], G)> for Combine<Sample, Sample> {
    type Output = Values<Sample>;

    fn forward(
        &self,
        _: &mut Device,
        ([prediction], _): ([Values<Sample>; 1], G),
    ) -> Values<Sample> {
        prediction
    }
}

impl<Prediction: M + 'static, Sample: M + 'static, G: Borrow<Factor>>
    Op<([Values<Prediction>; 2], G)> for Combine<Prediction, Sample>
{
    type Output = Values<Sample>;

    fn forward(
        &self,
        device: &mut Device,
        (predictions, scale): ([Values<Prediction>; 2], G),
    ) -> Values<Sample> {
        self.forward(device, (&predictions, scale))
    }
}

#[derive(DeviceSend)]
pub struct Context<P: M> {
    pub predictions: [Values<P>; 2],
    pub scale: Factor,
}

/// The predictions and the scale a launch's context holds.
impl<Prediction: M + 'static, Sample: M + 'static> Op<&mut Context<Prediction>>
    for Combine<Prediction, Sample>
{
    type Output = Values<Sample>;

    fn forward(&self, device: &mut Device, context: &mut Context<Prediction>) -> Values<Sample> {
        self.forward(device, (&context.predictions, &context.scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use crate::tests::forward;
    use furiosa_opt_std::runtime::launch;

    /// Guidance reads only the sample channels of a prediction that also holds learned variance
    /// channels.
    #[tokio::test(flavor = "current_thread")]
    async fn ignores_learned_variance_channels() {
        assert_guidance::<m![8192]>().await;
    }

    /// Guidance combines the unconditional and conditional predictions as `u + scale * (c - u)` at
    /// each scale.
    #[tokio::test(flavor = "current_thread")]
    async fn combines_complete_predictions() {
        assert_guidance::<m![4096]>().await;
    }

    async fn assert_guidance<Prediction: M + 'static>() {
        type Sample = m![4096];
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(forward.topology()).unwrap();
        let model = Combine::<Prediction, Sample>::default();
        for scale in [1.0_f32, 4.0] {
            let conditional: Vec<f32> = (0..Prediction::SIZE)
                .map(|i| {
                    if i < Sample::SIZE {
                        i as f32 / 1024.0
                    } else {
                        999.0
                    }
                })
                .collect();
            let unconditional: Vec<f32> = (0..Prediction::SIZE)
                .map(|i| if i < Sample::SIZE { -1.0 } else { -999.0 })
                .collect();
            let mut context = Context {
                predictions: [
                    HostTensor::<f32, Prediction>::from_vec(conditional)
                        .to_hbm(&mut device.pdma)
                        .await
                        .unwrap(),
                    HostTensor::<f32, Prediction>::from_vec(unconditional)
                        .to_hbm(&mut device.pdma)
                        .await
                        .unwrap(),
                ],
                scale: HostTensor::<f32, m![1 # 8]>::from_vec(vec![scale; 8])
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            };
            let output = launch(forward, (&mut device, &model, (), &mut (), &mut context))
                .await
                .unwrap();
            let actual = output.to_host::<Sample>(&mut device.pdma).await.unwrap();
            for (index, &actual) in actual.into_vec().iter().enumerate() {
                let expected = -1.0 + scale * (index as f32 / 1024.0 + 1.0);
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "{index}: {actual} != {expected}"
                );
            }
        }
    }
}
