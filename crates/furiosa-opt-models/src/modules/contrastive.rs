//! The scores of a contrastive pair of towers: each pair of embedding rows' dot product, scaled
//! by a learned factor and shifted by an optional bias.
use crate::{Module, tensor::axis::Single};
use furiosa_opt_std::prelude::*;

/// Contiguous FP32 embedding rows on one chip.
pub type Rows<Width, Count> = HbmTensor<f32, m![1], m![Count, Width]>;

/// One FP32 value on one chip.
pub type Factor = HbmTensor<f32, m![1], m![1]>;

/// The parameter is a logarithm: scores are exp(log_scale) * text @ image.T + bias.
#[derive(DeviceSend)]
pub struct DotProduct<Bias> {
    /// Logarithm of the multiplier applied to each dot product.
    pub log_scale: Factor,
    /// The scalar added to each score, or `()` for none.
    pub bias: Bias,
}

/// Compiled capacity of the text coordinate in the score matrix.
#[derive(Clone, Debug)]
pub struct Text<const N: usize>;

impl<const N: usize> AxisName for Text<N> {
    const NAME: Ident = Ident::new("Text");
    const SIZE: usize = N;
}

/// Compiled capacity of the image coordinate in the score matrix.
#[derive(Clone, Debug)]
pub struct Image<const N: usize>;

impl<const N: usize> AxisName for Image<N> {
    const NAME: Ident = Ident::new("Image");
    const SIZE: usize = N;
}

impl<W: AxisName, const T: usize, const I: usize, Bias, Env>
    Module<&([Rows<W, Single>; T], [Rows<W, Single>; I]), Env, &mut ()> for DotProduct<Bias>
{
    type Output = Rows<Image<I>, Text<T>>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: &([Rows<W, Single>; T], [Rows<W, Single>; I]),
    ) -> Self::Output {
        todo!(
            "NPU pairwise embedding dot products with learned exponential scale and optional bias"
        )
    }
}
