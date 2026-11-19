//! H3's condition, media and timestep projections, and its final video/audio velocity heads.
//!
//! These trees load relative to the transformer's physical checkpoint prefix. The condition
//! projection runs once before the text refiner; the media and time projections run each step.
//! Packing, the refiner, blocks, output extraction and latent statistics are separate boundaries.
//! The ordinary split-projection loaders retain FP32 weights as BF16 high/low images. Their
//! existing numerical leaves preserve the reference's rounding boundaries; no kernel is defined
//! here.

use crate::Module;
use crate::load::{Load, Named};
use crate::modules::diffusion;
use crate::ops::linear::{Linear, modulation, split};
use crate::ops::normalization::adaptive;
use crate::tensor::form::split::{Sides, Stacked};
use crate::tensor::memory::residency::Residency;
use crate::tensor::memory::{Hbm, Memory};
use crate::tensor::parallel::Shard;
use furiosa_opt_std::prelude::*;

axes![
    Condition = 5120,
    Width = 5376,
    Time = 2688,
    Patch = 96,
    Latent = 32
];

/// Four live chips, with a replicated copy of each projection's coefficients on every chip.
pub type Chips = m![{ Symbol<Shard<4>> }];

/// FP32 bits of the final RMS norm's stabilizer, fixed by the H3 family configuration.
pub const EPSILON: usize = 1e-5_f32.to_bits() as usize;

/// BF16 prompt rows times a BF16 matrix, with the FP32 bias added before the BF16 rounding.
pub type Text<In = Condition, Out = Width, C = Chips, X = Hbm> = split::Biased<
    In,
    Out,
    C,
    Linear<In, Out, C, <X as Memory>::Tensor<bf16, C, m![Out, In]>>,
    <X as Memory>::Tensor<f32, C, m![Out]>,
>;

/// FP32 input and coefficient values split into BF16 parts; the result rounds once to BF16.
/// Its stored matrix is `[Out, In]`; its prepared image is `[2*Out, 2*In]`.
pub type Input<In, Out = Width, C = Chips, X = Hbm> = split::Embedding<
    In,
    Out,
    C,
    Linear<
        Sides<In>,
        Stacked<Out>,
        C,
        <X as Memory>::Tensor<bf16, C, m![{ Symbol<Stacked<Out>> }, { Symbol<Sides<In>> }]>,
    >,
    <X as Memory>::Tensor<f32, C, m![Out]>,
>;

/// Cosines then sines, an FP32 projection, FP32 SiLU, and an FP32 projection. The two projections
/// use the ordinary split loader, under the checkpoint's actual `proj_in` and `proj_out` names.
pub type Timesteps<Hidden = Width, Out = Time, C = Chips, X = Hbm> = split::TimeEmbedding<
    Hidden,
    Out,
    C,
    Named<"proj_in", Input<split::Sinusoid, Hidden, C, X>>,
    Named<"proj_out", Input<Hidden, Out, C, X>>,
    <X as Memory>::Tensor<f32, C, m![{ Symbol<split::Frequency> }]>,
>;

/// BF16 hidden rows times the FP32 matrix's BF16 high/low images, summed with FP32 bias into
/// FP32 velocities. Its prepared coefficient image is `[2*Out, In]`.
pub type Output<Out, In = Width, C = Chips, X = Hbm> = split::Precise<
    In,
    Out,
    C,
    Linear<In, Stacked<Out>, C, <X as Memory>::Tensor<bf16, C, m![{ Symbol<Stacked<Out>> }, In]>>,
    <X as Memory>::Tensor<f32, C, m![Out]>,
>;

/// One timestep's final shift/scale: FP32 SiLU, BF16 rounding, biased BF16 projection, BF16 result.
pub type Modulation<const STEP: usize, In = Time, W = Width, C = Chips, X = Hbm> =
    modulation::HeadModulation<
        In,
        W,
        STEP,
        C,
        <X as Memory>::Tensor<bf16, C, m![{ Symbol<modulation::ShiftScale<W>> }, In]>,
        <X as Memory>::Tensor<f32, C, m![{ Symbol<modulation::ShiftScale<W>> }, 1 # 8]>,
    >;

/// RMS normalization rounds to BF16 before shift/scale modulation, which rounds again.
pub type Norm<W = Width, C = Chips, X = Hbm> =
    adaptive::FinalNorm<W, EPSILON, C, <X as Memory>::Tensor<bf16, C, m![W]>>;

/// Video uses timestep 0 and projects to a 2×2 patch of the 24-channel latent.
pub type VideoHead<W = Width, T = Time, O = Patch, C = Chips, X = Hbm> = diffusion::Head<
    Named<"adaln_proj.linear", Modulation<0, T, W, C, X>>,
    Named<"norm", Norm<W, C, X>>,
    Named<"video_out", Output<O, W, C, X>>,
>;

/// Audio uses timestep 1 and projects to the stereo decoder's 32-channel latent.
pub type AudioHead<W = Width, T = Time, O = Latent, C = Chips, X = Hbm> = diffusion::Head<
    Named<"adaln_proj.linear", Modulation<1, T, W, C, X>>,
    Named<"norm", Norm<W, C, X>>,
    Named<"audio_out", Output<O, W, C, X>>,
>;

/// The transformer's input weights, with generic children so each tree can retain prepared host
/// storage and lend the same HBM weights for execution. Children remain individually callable.
#[derive(DeviceSend, Load, Residency)]
#[residency(children(Text, Video, Audio, Time))]
pub struct Embeddings<
    Text = self::Text,
    Video = Input<Patch>,
    Audio = Input<Latent>,
    Time = Timesteps,
> {
    #[load(rename = "condition_proj")]
    pub text: Text,
    #[load(rename = "video_patch_proj")]
    pub video: Video,
    #[load(rename = "audio_patch_proj")]
    pub audio: Audio,
    #[load(rename = "time_embedder")]
    pub time: Time,
}

/// Runs the step's time and media projections. The text child is deliberately called separately
/// once before refinement; this step does not rerun it or join rows into a packed buffer.
impl<T, V, A, U, I, J, E, F, H, O, Env: Copy> Module<(U, (I, J)), Env, &mut ()>
    for Embeddings<T, V, A, O>
where
    O: for<'s> Module<U, Env, &'s mut (), Output = E>,
    V: for<'s> Module<I, Env, &'s mut (), Output = F>,
    A: for<'s> Module<J, Env, &'s mut (), Output = H>,
{
    type Output = (E, (F, H));

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (time, (video, audio)): (U, (I, J)),
    ) -> Self::Output {
        let time = self.time.forward(&mut *device, env, &mut (), time);
        let video = self.video.forward(&mut *device, env, &mut (), video);
        let audio = self.audio.forward(device, env, &mut (), audio);
        (time, (video, audio))
    }
}

/// Both final heads from the same `final_layer` checkpoint namespace. As in the reference, each
/// head owns its modulation and norm child; no broad ignore hides another transformer's weights.
#[derive(DeviceSend, Load, Residency)]
#[residency(children(Video, Audio))]
pub struct FinalLayer<Video = VideoHead, Audio = AudioHead> {
    #[load(rename = "final_layer")]
    pub video: Video,
    #[load(rename = "final_layer")]
    pub audio: Audio,
}

impl<'s, V, A, E, Y, Z, VideoState, AudioState, Video, Audio, Env: Copy>
    Module<(&(V, A), &E), Env, &'s mut (VideoState, AudioState)> for FinalLayer<Video, Audio>
where
    Video: for<'a, 'b> Module<(&'a V, &'b E), Env, &'s mut VideoState, Output = Y>,
    Audio: for<'a, 'b> Module<(&'a A, &'b E), Env, &'s mut AudioState, Output = Z>,
{
    type Output = (Y, Z);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (video_state, audio_state): &'s mut (VideoState, AudioState),
        ((video, audio), condition): (&(V, A), &E),
    ) -> Self::Output {
        let video = self
            .video
            .forward(&mut *device, env, video_state, (video, condition));
        let audio = self
            .audio
            .forward(device, env, audio_state, (audio, condition));
        (video, audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::{Checkpoint, Context, LoadError};
    use crate::ops::index::packed::Timestep;
    use crate::tensor::memory::residency::{Leaves, Path, Visit};
    use safetensors::Dtype;
    use std::cell::RefCell;
    use std::convert::Infallible;
    use std::marker::PhantomData;
    use std::sync::Arc;

    axes![
        SmallCondition = 8,
        SmallWidth = 128,
        SmallTime = 16,
        SmallVideo = 8,
        SmallAudio = 4
    ];

    struct Host;
    struct Leaf<D, C, S> {
        values: Arc<[D]>,
        shape: PhantomData<(C, S)>,
    }
    struct Image {
        path: String,
        bytes: Vec<u8>,
        chips: usize,
        elements: usize,
    }

    thread_local! { static IMAGES: RefCell<Vec<Image>> = const { RefCell::new(Vec::new()) }; }

    impl Memory for Host {
        type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Leaf<D, C, S>;
    }

    impl<D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M> Load<HostTensor<D, E>>
        for Leaf<D, C, S>
    {
        fn load(cx: &mut Context<'_>, host: HostTensor<D, E>) -> Result<Self, LoadError> {
            assert_eq!(E::SIZE, C::SIZE * S::SIZE);
            let values = host.into_vec();
            IMAGES.with_borrow_mut(|images| {
                images.push(Image {
                    path: cx.path(),
                    bytes: D::to_buf(&values),
                    chips: C::SIZE,
                    elements: S::SIZE,
                })
            });
            Ok(Self {
                values: values.into(),
                shape: PhantomData,
            })
        }
    }

    impl<'a, D: MaterializableScalar, C: M + 'static, S: M + 'static> Load<&'a Checkpoint>
        for Leaf<D, C, S>
    {
        fn load(cx: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
            cx.checkpoint(from, |cx| {
                let host = from.host::<D, S, m![{ C }, { S }]>(cx)?;
                Self::load(cx, host)
            })
        }
    }

    impl<'a, D, C, S> Load<&'a Leaf<D, C, S>> for &'a Leaf<D, C, S> {
        fn load(_: &mut Context<'_>, from: &'a Leaf<D, C, S>) -> Result<Self, LoadError> {
            Ok(from)
        }
    }

    #[derive(Load, Residency)]
    #[residency(children(E, H))]
    struct Group<E, H> {
        #[load(rename = "")]
        inputs: E,
        #[load(rename = "")]
        heads: H,
    }

    type Inputs<X> = Embeddings<
        Text<SmallCondition, SmallWidth, Chips, X>,
        Input<SmallVideo, SmallWidth, Chips, X>,
        Input<SmallAudio, SmallWidth, Chips, X>,
        Timesteps<SmallWidth, SmallTime, Chips, X>,
    >;
    type Heads<X> = FinalLayer<
        VideoHead<SmallWidth, SmallTime, SmallVideo, Chips, X>,
        AudioHead<SmallWidth, SmallTime, SmallAudio, Chips, X>,
    >;
    type Canonical = Named<"transformer", Group<Inputs<Hbm>, Heads<Hbm>>>;
    type Tree = <Canonical as Residency>::In<Host>;

    struct Borrow;
    impl<'t> Leaves<'t, Host, &'t Host> for Borrow {
        type Error = Infallible;
        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &'t Leaf<D, C, S>,
            _: Path<'_>,
        ) -> Result<&'t Leaf<D, C, S>, Self::Error> {
            Ok(leaf)
        }
    }

    struct Share {
        seen: usize,
        fail_at: Option<usize>,
    }
    impl<'t> Leaves<'t, Host, Host> for Share {
        type Error = String;
        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &'t Leaf<D, C, S>,
            path: Path<'_>,
        ) -> Result<Leaf<D, C, S>, Self::Error> {
            let at = self.seen;
            self.seen += 1;
            if self.fail_at == Some(at) {
                return Err(path.to_string());
            }
            Ok(Leaf {
                values: Arc::clone(&leaf.values),
                shape: PhantomData,
            })
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Record {
        path: String,
        bytes: Vec<u8>,
        chips: usize,
        elements: usize,
        identity: usize,
        owners: usize,
    }
    #[derive(Default)]
    struct Snapshot(Vec<Record>);
    impl Snapshot {
        fn record<D: MaterializableScalar, C: M, S: M>(
            &mut self,
            leaf: &Leaf<D, C, S>,
            path: Path<'_>,
        ) {
            self.0.push(Record {
                path: path.to_string(),
                bytes: D::to_buf(&leaf.values),
                chips: C::SIZE,
                elements: S::SIZE,
                identity: Arc::as_ptr(&leaf.values).cast::<D>() as usize,
                owners: Arc::strong_count(&leaf.values),
            });
        }
    }
    impl Visit<Host> for Snapshot {
        type Error = Infallible;
        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &Leaf<D, C, S>,
            path: Path<'_>,
        ) -> Result<(), Self::Error> {
            self.record(leaf, path);
            Ok(())
        }
    }
    impl<'t> Visit<&'t Host> for Snapshot {
        type Error = Infallible;
        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &&'t Leaf<D, C, S>,
            path: Path<'_>,
        ) -> Result<(), Self::Error> {
            self.record(leaf, path);
            Ok(())
        }
    }

    struct Tensor {
        name: String,
        shape: Vec<usize>,
        values: Vec<f32>,
        dtype: Dtype,
    }

    fn snapshot<X: Memory>(tree: &<Canonical as Residency>::In<X>) -> Vec<Record>
    where
        Snapshot: Visit<X, Error = Infallible>,
    {
        let mut snapshot = Snapshot::default();
        Canonical::visit::<X, _>(tree, &mut snapshot, Path::ROOT).unwrap();
        snapshot.0
    }

    fn values(count: usize, seed: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i + seed) % 29) as f32 * 0.137 - 1.917 + 0.000_31)
            .collect()
    }

    fn fixture() -> Vec<Tensor> {
        let mut tensors = Vec::new();
        for (name, out, input) in [
            ("condition_proj", 128, 8),
            ("video_patch_proj", 128, 8),
            ("audio_patch_proj", 128, 4),
            ("time_embedder.proj_in", 128, 256),
            ("time_embedder.proj_out", 16, 128),
            ("final_layer.adaln_proj.linear", 256, 16),
            ("final_layer.video_out", 8, 128),
            ("final_layer.audio_out", 4, 128),
        ] {
            let seed = tensors.len();
            tensors.push(Tensor {
                name: format!("transformer.{name}.weight"),
                shape: vec![out, input],
                values: values(out * input, seed),
                dtype: Dtype::F32,
            });
            tensors.push(Tensor {
                name: format!("transformer.{name}.bias"),
                shape: vec![out],
                values: values(out, seed + 1),
                dtype: Dtype::F32,
            });
        }
        tensors.push(Tensor {
            name: "transformer.final_layer.norm.weight".into(),
            shape: vec![128],
            values: values(128, 19),
            dtype: Dtype::F32,
        });
        tensors
    }

    fn checkpoint(tensors: &[Tensor]) -> Checkpoint {
        let mut checkpoint = Checkpoint::default();
        for tensor in tensors {
            let bytes = match tensor.dtype {
                Dtype::BF16 => bf16::to_buf(
                    &tensor
                        .values
                        .iter()
                        .copied()
                        .map(bf16::from_f32)
                        .collect::<Vec<_>>(),
                ),
                _ => f32::to_buf(&tensor.values),
            };
            checkpoint
                .insert(&tensor.name, tensor.dtype, &tensor.shape, bytes)
                .unwrap();
        }
        checkpoint
    }

    fn load(tensors: &[Tensor]) -> Result<Tree, LoadError> {
        IMAGES.with_borrow_mut(Vec::clear);
        Tree::load(&mut Context::new(None), &checkpoint(tensors))
    }

    fn images() -> Vec<Image> {
        IMAGES.with_borrow_mut(std::mem::take)
    }

    fn floats(image: &Image) -> Vec<f32> {
        image
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn halves(image: &Image) -> Vec<bf16> {
        image
            .bytes
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
            .collect()
    }

    #[test]
    fn strict_named_tree_reads_only_the_reference_component_group() {
        let tensors = fixture();
        let _tree = load(&tensors).unwrap();
        let loaded = images();
        assert_eq!(loaded.len(), 21); // Ten input leaves, frequencies, and five leaves per head.
        assert!(loaded.iter().all(|image| image.chips == 4));
        assert_eq!(Chips::SIZE, 4);
        assert_eq!(
            (
                Condition::SIZE,
                Width::SIZE,
                Time::SIZE,
                Patch::SIZE,
                Latent::SIZE
            ),
            (5120, 5376, 2688, 96, 32)
        );
        let mut tensors = tensors;
        tensors.push(Tensor {
            name: "transformer.unclaimed.weight".into(),
            shape: vec![1],
            values: vec![1.],
            dtype: Dtype::F32,
        });
        assert!(
            matches!(load(&tensors), Err(LoadError::Unexpected(names)) if names == ["transformer.unclaimed.weight"])
        );
    }

    #[test]
    fn strict_names_collect_missing_media_and_head_weights() {
        let mut tensors = fixture();
        let missing = [
            "transformer.video_patch_proj.bias",
            "transformer.final_layer.audio_out.weight",
        ];
        tensors.retain(|tensor| !missing.contains(&tensor.name.as_str()));
        let Err(LoadError::Missing(names)) = load(&tensors) else {
            panic!("must report missing weights")
        };
        assert_eq!(names, missing);
    }

    #[test]
    fn rejects_wrong_shapes_and_nonfloating_projection_sources() {
        for name in [
            "video_patch_proj.weight",
            "time_embedder.proj_out.weight",
            "final_layer.video_out.weight",
        ] {
            let mut tensors = fixture();
            let tensor = tensors
                .iter_mut()
                .find(|tensor| tensor.name.ends_with(name))
                .unwrap();
            tensor.shape.swap(0, 1);
            let full_name = tensor.name.clone();
            assert!(
                matches!(load(&tensors), Err(LoadError::InvalidData { name, .. }) if name == full_name)
            );
        }
        let mut tensors = fixture();
        tensors
            .iter_mut()
            .find(|tensor| tensor.name.ends_with("audio_patch_proj.weight"))
            .unwrap()
            .dtype = Dtype::I32;
        assert!(
            matches!(load(&tensors), Err(LoadError::InvalidData { name, .. }) if name == "transformer.audio_patch_proj.weight")
        );
    }

    #[test]
    fn prepared_media_and_time_images_preserve_fp32_precision_parts_on_every_chip() {
        let tensors = fixture();
        let _tree = load(&tensors).unwrap();
        let loaded = images();
        for (tree_path, tensor_name) in [
            ("inputs.video.weight", "video_patch_proj.weight"),
            ("inputs.audio.weight", "audio_patch_proj.weight"),
            (
                "inputs.time.first.proj_in.weight",
                "time_embedder.proj_in.weight",
            ),
            (
                "inputs.time.second.proj_out.weight",
                "time_embedder.proj_out.weight",
            ),
        ] {
            let source = tensors
                .iter()
                .find(|tensor| tensor.name.ends_with(tensor_name))
                .unwrap();
            let image = loaded
                .iter()
                .find(|image| image.path.ends_with(tree_path))
                .unwrap();
            let [out, input] = source.shape.as_slice() else {
                unreachable!()
            };
            let mut expected = Vec::new();
            for _chip in 0..4 {
                for part in 0..2 {
                    for row in 0..*out {
                        for side in 0..2 {
                            for col in 0..*input {
                                let value = source.values[row * input + col];
                                let high = bf16::from_f32(value);
                                expected.push(match (part, side) {
                                    (0, _) => high,
                                    (1, 0) => bf16::from_f32(value - high.to_f32()),
                                    _ => bf16::from_f32(0.),
                                });
                            }
                        }
                    }
                }
            }
            assert_eq!(image.elements, 4 * out * input);
            assert_eq!(halves(image), expected, "{}", image.path);
        }
    }

    #[test]
    fn heads_keep_high_low_output_rows_and_padded_fp32_modulation_bias() {
        let tensors = fixture();
        let _tree = load(&tensors).unwrap();
        let loaded = images();
        for (path, name) in [
            (
                "heads.video.projection.video_out.weight",
                "final_layer.video_out.weight",
            ),
            (
                "heads.audio.projection.audio_out.weight",
                "final_layer.audio_out.weight",
            ),
        ] {
            let source = tensors
                .iter()
                .find(|tensor| tensor.name.ends_with(name))
                .unwrap();
            let image = loaded
                .iter()
                .find(|image| image.path.ends_with(path))
                .unwrap();
            let mut expected = Vec::new();
            for _chip in 0..4 {
                for part in 0..2 {
                    for &value in &source.values {
                        let high = bf16::from_f32(value);
                        expected.push(if part == 0 {
                            high
                        } else {
                            bf16::from_f32(value - high.to_f32())
                        });
                    }
                }
            }
            assert_eq!(halves(image), expected);
        }
        let source = tensors
            .iter()
            .find(|tensor| tensor.name.ends_with("final_layer.adaln_proj.linear.bias"))
            .unwrap();
        let biases: Vec<_> = loaded
            .iter()
            .filter(|image| image.path.ends_with("adaln_proj.linear.bias"))
            .collect();
        assert_eq!(biases.len(), 2);
        for image in biases {
            let expected: Vec<f32> = (0..4)
                .flat_map(|_| {
                    source
                        .values
                        .iter()
                        .flat_map(|&value| [value, 0., 0., 0., 0., 0., 0., 0.])
                })
                .collect();
            assert_eq!(floats(image), expected);
        }
    }

    #[test]
    fn computes_frequencies_without_a_synthetic_checkpoint_tensor() {
        let tensors = fixture();
        let _tree = load(&tensors).unwrap();
        let loaded = images();
        let image = loaded
            .iter()
            .find(|image| image.path.ends_with("time.frequencies"))
            .unwrap();
        let frequencies = floats(image);
        assert_eq!(frequencies.len(), 4 * 128);
        for chip in frequencies.chunks_exact(128) {
            for (j, &got) in chip.iter().enumerate() {
                let expected = 10_000_f64.powf(-(j as f64) / 128.);
                assert!((f64::from(got) - expected).abs() < 2e-7);
            }
        }
    }

    #[test]
    fn text_rounds_coefficients_to_bf16_but_keeps_bias_fp32() {
        let tensors = fixture();
        let _tree = load(&tensors).unwrap();
        let loaded = images();
        let weight = tensors
            .iter()
            .find(|tensor| tensor.name.ends_with("condition_proj.weight"))
            .unwrap();
        let bias = tensors
            .iter()
            .find(|tensor| tensor.name.ends_with("condition_proj.bias"))
            .unwrap();
        let image = loaded
            .iter()
            .find(|image| image.path.ends_with("inputs.text.weight.affine.weight"))
            .unwrap();
        let expected: Vec<bf16> = weight.values.iter().copied().map(bf16::from_f32).collect();
        assert_eq!(halves(image), expected.repeat(4));
        let image = loaded
            .iter()
            .find(|image| image.path.ends_with("inputs.text.bias"))
            .unwrap();
        assert_eq!(floats(image), bias.values.repeat(4));
        assert!(
            bias.values
                .iter()
                .any(|&value| bf16::from_f32(value).to_f32() != value)
        );
    }

    #[test]
    fn bf16_checkpoint_coefficients_have_zero_low_parts() {
        let mut tensors = fixture();
        tensors
            .iter_mut()
            .find(|tensor| tensor.name.ends_with("video_patch_proj.weight"))
            .unwrap()
            .dtype = Dtype::BF16;
        let _tree = load(&tensors).unwrap();
        let loaded = images();
        let image = loaded
            .iter()
            .find(|image| image.path.ends_with("inputs.video.weight"))
            .unwrap();
        let coefficients = halves(image);
        for chip in coefficients.chunks_exact(4 * 128 * 8) {
            assert!(chip[2 * 128 * 8..].iter().all(|value| value.to_f32() == 0.));
            assert!(chip[..2 * 128 * 8].iter().any(|value| value.to_f32() != 0.));
        }
    }

    #[test]
    fn borrowed_children_have_the_reference_input_and_output_scalar_contracts() {
        type R = crate::tensor::axis::Rows<64>;
        fn accepts<M, I, O>()
        where
            M: for<'s> Module<I, (), &'s mut (), Output = O>,
        {
        }
        accepts::<
            Text<SmallCondition, SmallWidth, Chips, &'static Hbm>,
            &HbmTensor<bf16, Chips, m![R, SmallCondition]>,
            HbmTensor<bf16, Chips, m![R, SmallWidth]>,
        >();
        accepts::<
            Input<SmallVideo, SmallWidth, Chips, &'static Hbm>,
            &HbmTensor<f32, Chips, m![R, SmallVideo]>,
            HbmTensor<bf16, Chips, m![R, SmallWidth]>,
        >();
        accepts::<
            Timesteps<SmallWidth, SmallTime, Chips, &'static Hbm>,
            &HbmTensor<f32, Chips, m![R,1 # 8]>,
            HbmTensor<f32, Chips, m![R, SmallTime]>,
        >();
        accepts::<
            VideoHead<SmallWidth, SmallTime, SmallVideo, Chips, &'static Hbm>,
            (
                &HbmTensor<bf16, Chips, m![R, SmallWidth]>,
                &HbmTensor<f32, Chips, m![Timestep, SmallTime]>,
            ),
            HbmTensor<f32, Chips, m![R, SmallVideo]>,
        >();
        accepts::<
            AudioHead<SmallWidth, SmallTime, SmallAudio, Chips, &'static Hbm>,
            (
                &HbmTensor<bf16, Chips, m![R, SmallWidth]>,
                &HbmTensor<f32, Chips, m![Timestep, SmallTime]>,
            ),
            HbmTensor<f32, Chips, m![R, SmallAudio]>,
        >();
        accepts::<
            Inputs<&'static Hbm>,
            (
                &HbmTensor<f32, Chips, m![R,1 # 8]>,
                (
                    &HbmTensor<f32, Chips, m![R, SmallVideo]>,
                    &HbmTensor<f32, Chips, m![R, SmallAudio]>,
                ),
            ),
            (
                HbmTensor<f32, Chips, m![R, SmallTime]>,
                (
                    HbmTensor<bf16, Chips, m![R, SmallWidth]>,
                    HbmTensor<bf16, Chips, m![R, SmallWidth]>,
                ),
            ),
        >();
        fn accepts_heads<M, I, O>()
        where
            M: for<'s> Module<I, (), &'s mut ((), ()), Output = O>,
        {
        }
        accepts_heads::<
            Heads<&'static Hbm>,
            (
                &(
                    HbmTensor<bf16, Chips, m![R, SmallWidth]>,
                    HbmTensor<bf16, Chips, m![R, SmallWidth]>,
                ),
                &HbmTensor<f32, Chips, m![Timestep, SmallTime]>,
            ),
            (
                HbmTensor<f32, Chips, m![R, SmallVideo]>,
                HbmTensor<f32, Chips, m![R, SmallAudio]>,
            ),
        >();
    }
    #[test]
    fn h3_canonical_projection_loads_visits_and_borrows_without_copying_storage() {
        fn canonical<T: Residency>()
        where
            for<'a> T::In<&'a Hbm>: Load<&'a T::In<Hbm>>,
        {
        }
        canonical::<Embeddings>();
        canonical::<FinalLayer>();

        let stored = load(&fixture()).unwrap();
        let expected = snapshot::<Host>(&stored);
        let paths: Vec<_> = expected
            .iter()
            .map(|record| record.path.strip_prefix("transformer.").unwrap())
            .collect();
        assert_eq!(
            paths,
            [
                "inputs.text.weight.affine.weight",
                "inputs.text.bias",
                "inputs.video.weight.affine.weight",
                "inputs.video.bias",
                "inputs.audio.weight.affine.weight",
                "inputs.audio.bias",
                "inputs.time.frequencies",
                "inputs.time.first.proj_in.weight.affine.weight",
                "inputs.time.first.proj_in.bias",
                "inputs.time.second.proj_out.weight.affine.weight",
                "inputs.time.second.proj_out.bias",
                "heads.video.modulation.adaln_proj.linear.weight",
                "heads.video.modulation.adaln_proj.linear.bias",
                "heads.video.norm.norm.weight.tensor",
                "heads.video.projection.video_out.weight.affine.weight",
                "heads.video.projection.video_out.bias",
                "heads.audio.modulation.adaln_proj.linear.weight",
                "heads.audio.modulation.adaln_proj.linear.bias",
                "heads.audio.norm.norm.weight.tensor",
                "heads.audio.projection.audio_out.weight.affine.weight",
                "heads.audio.projection.audio_out.bias",
            ]
        );
        assert!(
            expected
                .iter()
                .all(|record| record.chips == 4 && record.owners == 1 && !record.bytes.is_empty())
        );
        {
            let borrowed: <Canonical as Residency>::In<&Host> =
                Load::load(&mut Context::new(None), &stored).unwrap();
            assert_eq!(snapshot::<&Host>(&borrowed), expected);
            let mapped =
                Canonical::map::<Host, &Host, _>(&stored, &mut Borrow, Path::ROOT).unwrap();
            assert_eq!(snapshot::<&Host>(&mapped), expected);
        }
        let shared = Canonical::map::<Host, Host, _>(
            &stored,
            &mut Share {
                seen: 0,
                fail_at: None,
            },
            Path::ROOT,
        )
        .unwrap();
        assert!(
            snapshot::<Host>(&shared)
                .iter()
                .all(|record| record.owners == 2)
        );
        drop(stored);
        assert_eq!(snapshot::<Host>(&shared), expected);
    }

    #[test]
    fn failed_h3_projection_releases_every_partial_child() {
        let stored = load(&fixture()).unwrap();
        let expected = snapshot::<Host>(&stored);
        for at in [0, 1, 7, 11, 20] {
            let result = Canonical::map::<Host, Host, _>(
                &stored,
                &mut Share {
                    seen: 0,
                    fail_at: Some(at),
                },
                Path::ROOT,
            );
            let error = result.err().expect("mapper must stop at the rejected leaf");
            assert_eq!(error, expected[at].path);
            assert_eq!(snapshot::<Host>(&stored), expected);
        }
    }
}
