//! Numerical operations and their device layouts.

// Elementwise functions.
pub mod add;
pub mod cast;
mod pointwise;
pub(crate) mod vector;

// Projection and normalization.
pub mod linear;
pub mod normalization;
pub mod swiglu;

// Attention and sequence mixing.
pub mod attention;

// Layout, indexing and communication.
pub mod collective;
pub mod indexing;
pub mod layout;

// Image and embedding functions.
pub mod diffusion;
pub mod embedding;
pub mod vocabulary;

// Host storage boundary.
pub mod tensor;
pub mod weight;

pub mod multiply {
    use super::pointwise::Float;
    use crate::Module;
    use crate::ops::layout::axes::Single;
    use furiosa_opt_std::prelude::*;

    /// Multiply a single-chip tensor by an FP32 scalar; the kernel requires eight padded cells.
    #[derive(DeviceSend)]
    pub struct Scale<Factor = HbmTensor<f32, m![1], m![1]>>(pub Factor);

    impl<D: Float, S: M + 'static> Module<(&mut Device, &HbmTensor<D, m![1], S>)> for Scale {
        type Output = HbmTensor<D, m![1], S>;

        fn forward(&self, _: (&mut Device, &HbmTensor<D, m![1], S>)) -> Self::Output {
            todo!("NPU scaling by an unpadded scalar")
        }
    }

    /// Elementwise product of tensors with identical chip and element mappings.
    #[derive(Default, DeviceSend)]
    pub struct Multiply;

    impl<D: Float, S: M + 'static, Chips: M>
        Module<(
            &mut Device,
            (&HbmTensor<D, Chips, S>, &HbmTensor<D, Chips, S>),
        )> for Multiply
    {
        type Output = HbmTensor<D, Chips, S>;

        fn forward(
            &self,
            (device, (left, right)): (
                &mut Device,
                (&HbmTensor<D, Chips, S>, &HbmTensor<D, Chips, S>),
            ),
        ) -> Self::Output {
            super::pointwise::Window::<S>::combine::<_, true, Chips>(device, left, right)
        }
    }

    /// Multiply the queries of a positioned query/key pair by the step's FP32 scalar; the keys,
    /// and so the cache, keep their values.
    #[derive(Clone, Copy, Default, DeviceSend)]
    pub struct QueryScale;

    impl<'s, D: Float, S: M + 'static, K>
        Module<(
            &mut Device,
            (
                (HbmTensor<D, m![1], S>, K),
                &'s HbmTensor<f32, m![1], m![Single, 1 # 8]>,
            ),
        )> for QueryScale
    {
        type Output = (HbmTensor<D, m![1], S>, K);

        fn forward(
            &self,
            (device, ((query, key), factor)): (
                &mut Device,
                (
                    (HbmTensor<D, m![1], S>, K),
                    &'s HbmTensor<f32, m![1], m![Single, 1 # 8]>,
                ),
            ),
        ) -> Self::Output {
            (Scale::rows(device, &query, factor), key)
        }
    }

    /// The query scale `1 + beta * ln(1 + floor(position / original))` of a token, which grows
    /// once per `original` positions past the context the model was trained on.
    #[derive(Clone, Copy, Debug)]
    pub struct Temperature {
        /// Coefficient of the logarithm.
        pub beta: f32,
        /// Positions per step: the context length before extension.
        pub original: std::num::NonZeroUsize,
    }

    impl Temperature {
        /// The scale at a token's sequence position (not its cache slot), computed in FP32 and
        /// rounded to the BF16 the queries are multiplied by.
        pub fn at(&self, position: usize) -> bf16 {
            let steps = (position as f32 / self.original.get() as f32).floor();
            bf16::from_f32(1.0 + self.beta * (1.0 + steps).ln())
        }
    }

    /// Multiply each contiguous row by its corresponding scalar.
    #[derive(Default, DeviceSend)]
    pub struct RowScale;

    impl<D: Scalar, Rows: AxisName, Width: AxisName>
        Module<(
            &mut Device,
            (
                &HbmTensor<D, m![1], m![Rows, Width]>,
                &HbmTensor<D, m![1], m![Rows]>,
            ),
        )> for RowScale
    {
        type Output = HbmTensor<D, m![1], m![Rows, Width]>;

        fn forward(
            &self,
            _: (
                &mut Device,
                (
                    &HbmTensor<D, m![1], m![Rows, Width]>,
                    &HbmTensor<D, m![1], m![Rows]>,
                ),
            ),
        ) -> Self::Output {
            todo!("NPU multiplication by one scalar per row")
        }
    }

    #[cfg(test)]
    mod tests {
        use super::Temperature;

        /// The scale steps at multiples of the original context and is BF16 in between.
        #[test]
        fn steps_at_each_original_context() {
            let temperature = Temperature {
                beta: 0.1,
                original: std::num::NonZeroUsize::new(16384).unwrap(),
            };
            let scales = [0, 16383, 16384, 32767, 32768].map(|at| temperature.at(at).to_f32());
            assert_eq!(scales, [1.0, 1.0, 1.0703125, 1.0703125, 1.109375]);
        }
    }
}

pub mod activation {
    use super::pointwise::{Float, Unary};
    use crate::Module;
    use furiosa_opt_std::prelude::*;

    /// Elementwise activation selected by a model's configuration.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Activation {
        /// max(x, 0).
        Relu,
        /// x * Phi(x), using the normal cumulative distribution.
        Gelu,
        /// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
        NewGelu,
        /// x * sigmoid(x).
        Silu,
        /// x * sigmoid(1.702 * x).
        QuickGelu,
    }

    /// max(x, 0) on a single-chip tensor.
    #[derive(Default, Clone, Copy, DeviceSend)]
    pub struct Relu;

    impl<D: Scalar, S: M> Module<(&mut Device, &HbmTensor<D, m![1], S>)> for Relu {
        type Output = HbmTensor<D, m![1], S>;

        fn forward(&self, _: (&mut Device, &HbmTensor<D, m![1], S>)) -> Self::Output {
            todo!("NPU rectified linear unit")
        }
    }

    /// x * sigmoid(1.702 * x) on a single-chip tensor.
    #[derive(Default, Clone, Copy, DeviceSend)]
    pub struct QuickGelu;

    impl<D: Scalar, S: M> Module<(&mut Device, &HbmTensor<D, m![1], S>)> for QuickGelu {
        type Output = HbmTensor<D, m![1], S>;

        fn forward(&self, _: (&mut Device, &HbmTensor<D, m![1], S>)) -> Self::Output {
            todo!("NPU quick GELU activation")
        }
    }

    /// Exact x * Phi(x) on a single-chip tensor.
    #[derive(Default, Clone, Copy, DeviceSend)]
    pub struct Gelu;

    impl<D: Scalar, S: M> Module<(&mut Device, &HbmTensor<D, m![1], S>)> for Gelu {
        type Output = HbmTensor<D, m![1], S>;

        fn forward(&self, _: (&mut Device, &HbmTensor<D, m![1], S>)) -> Self::Output {
            todo!("NPU exact GELU activation")
        }
    }

    /// Tanh GELU approximation on a single-chip tensor, with coefficient 0.044715.
    #[derive(Default, Clone, Copy, DeviceSend)]
    pub struct NewGelu;

    impl<D: Float, S: M + 'static> Module<(&mut Device, &HbmTensor<D, m![1], S>)> for NewGelu {
        type Output = HbmTensor<D, m![1], S>;

        fn forward(&self, (device, input): (&mut Device, &HbmTensor<D, m![1], S>)) -> Self::Output {
            Unary::rows::<_, _, { Unary::Gelu }, _>(device, input)
        }
    }

    /// x * sigmoid(x), preserving any input chip mapping.
    #[derive(Clone, Copy, Default, DeviceSend)]
    pub struct Silu;

    impl<D: Float, S: M + 'static, Chips: M> Module<(&mut Device, &HbmTensor<D, Chips, S>)> for Silu {
        type Output = HbmTensor<D, Chips, S>;

        fn forward(&self, (device, input): (&mut Device, &HbmTensor<D, Chips, S>)) -> Self::Output {
            Unary::rows::<_, _, { Unary::Silu }, Chips>(device, input)
        }
    }

    /// 1 / (1 + exp(-x)), preserving any input chip mapping.
    #[derive(Clone, Copy, Default, DeviceSend)]
    pub struct Sigmoid;

    impl<D: Float, S: M + 'static, Chips: M> Module<(&mut Device, &HbmTensor<D, Chips, S>)>
        for Sigmoid
    {
        type Output = HbmTensor<D, Chips, S>;

        fn forward(&self, (device, input): (&mut Device, &HbmTensor<D, Chips, S>)) -> Self::Output {
            Unary::rows::<_, _, { Unary::Sigmoid }, Chips>(device, input)
        }
    }

    /// `silu(gate) * up` of rows packed as their gate features followed by their up features.
    #[derive(Clone, Copy, Default, DeviceSend)]
    pub struct SwiGlu;

    /// Hyperbolic tangent on a single-chip tensor.
    #[derive(Default, Clone, Copy, DeviceSend)]
    pub struct Tanh;

    impl<D: Float, S: M + 'static> Module<(&mut Device, &HbmTensor<D, m![1], S>)> for Tanh {
        type Output = HbmTensor<D, m![1], S>;

        fn forward(&self, (device, input): (&mut Device, &HbmTensor<D, m![1], S>)) -> Self::Output {
            Unary::rows::<_, _, { Unary::Tanh }, _>(device, input)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::Sigmoid;
        use crate::Module;
        use furiosa_opt_std::{prelude::*, runtime::launch};

        axes![Gates = 4096];

        #[device(chip = 1)]
        fn sigmoid(
            device: &mut Device,
            input: &HbmTensor<bf16, m![1], m![Gates]>,
        ) -> HbmTensor<bf16, m![1], m![Gates]> {
            Sigmoid.forward((device, input))
        }

        /// The logistic function of each BF16 gate is its FP32 value rounded once, saturating at
        /// both ends.
        #[tokio::test(flavor = "current_thread")]
        async fn sigmoid_rounds_once() {
            let _lease = crate::tests::DEVICE.lock().await;
            let mut device = Device::new(sigmoid.topology()).unwrap();
            let gates: Vec<bf16> = (0..Gates::SIZE)
                .map(|at| bf16::from_f32((at as f32 - 2048.0) / 50.0))
                .collect();
            let input = HostTensor::<bf16, m![Gates]>::from_vec(gates.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(sigmoid, (&mut device, &input))
                .await
                .unwrap()
                .to_host::<m![Gates]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (gate, output) in gates.iter().zip(output) {
                let expected = bf16::from_f32(1.0 / (1.0 + (-gate.to_f32()).exp()));
                let gate = gate.to_f32();
                assert_eq!(output.to_bits(), expected.to_bits(), "sigmoid({gate})");
            }
        }
    }
}
