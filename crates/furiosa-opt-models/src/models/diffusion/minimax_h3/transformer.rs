//! H3's checkpoint-named refiner and joint-block weight compositions.
//!
//! Joint attention takes fourteen heads per chip, orders Q/K features for partial rotary and
//! gathers rows in nonuniform tiles. The refiner replicates all heads on four chips and keeps
//! checkpoint feature order. These owning trees reuse public loaders and numerical leaves.
//! The refiner uses a finite text envelope; joint Live execution uses [`super::joint::Context`].
//! Request state supplies visibility and runtime lengths separately from these weights.

use super::embeddings::{Chips, EPSILON, Time, Width};
use super::packing::{DEFAULT_GATHER_TILES, GATHER_BODY, GATHER_HEAD};
use crate::load::{Checkpoint, Context, Load, LoadError, Named};
use crate::modules::generic::Prefix;
use crate::modules::parallel::{Exchanged, Gathered};
use crate::modules::residual::Residual;
use crate::modules::{attention, diffusion, feedforward};
use crate::ops::attention::flash;
use crate::ops::attention::{full, geometry};
use crate::ops::collective::InTiles;
use crate::ops::linear::{self, Linear, modulation};
use crate::ops::normalization::{AdaptiveRms, RmsNorm};
use crate::ops::pointwise::{SwiGlu, add::Add, gated};
use crate::ops::swiglu::GateUp;
use crate::tensor::form::Weight;
use crate::tensor::memory::residency::Residency;
use crate::tensor::memory::{Hbm, Memory};
use crate::tensor::parallel::{Parallel, Part};
use crate::{First, Layers, Module, Sequential, Then, Unrolled};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Joint attention and feed-forward residuals, with exact checkpoint names for all children.
pub type Block<W = Width, A = Attended, F = Expanded, T = Time, X = Hbm, Tiles = GatherTiles> =
    diffusion::Adaptive<
        Named<"adaln_proj.linear", Modulation<T, W, X>>,
        Named<"attn", BlockAttention<W, A, X, Tiles>>,
        Named<"mlp", BlockFeedForward<W, F, X>>,
        Named<"norm1", AdaptiveNorm<W, X>>,
        gated::Add,
        Named<"norm2", AdaptiveNorm<W, X>>,
    >;

/// Compact joined QKV, explicit head norms, optimized attention and whole head-to-row exchange.
/// The Live call uses [`super::joint::Context`]; the adaptive block also binds request spans
/// through [`diffusion::WithSpans`]. Loading alone does not validate the numerical call.
pub type BlockAttention<W = Width, A = Attended, X = Hbm, Tiles = GatherTiles> = attention::Full<
    attention::LiveNormalizedQkv<
        attention::NormalizedQkv<
            Named<
                "qkv_proj",
                FusedQkv<
                    W,
                    A,
                    Gathered<
                        full::Joined<
                            full::Projection<
                                Grouped<W, full::Joint<Part<A, 4>>, X>,
                                flash::Head,
                                Rotated,
                            >,
                        >,
                        Parallel<4>,
                        Tiles,
                    >,
                >,
            >,
            Named<"q_norm", HeadNorm<Rotated, X>>,
            Named<"k_norm", HeadNorm<Rotated, X>>,
        >,
    >,
    flash::Compact<Heads<Part<A, 4>>, SCALE>,
    Named<"out_proj", Exchanged<Grouped<A, W, X>, Parallel<4>>>,
>;

/// Replicated K-grouped gate/up and down matrices, retaining the reference's SwiGLU rounding.
pub type BlockFeedForward<W = Width, F = Expanded, X = Hbm> = feedforward::Mlp<
    Prefix<Named<"fc1", Grouped<W, GateUp<F>, X>>>,
    Prefix<SwiGlu>,
    Prefix<Named<"fc2", Grouped<F, W, X>>>,
>;

/// A quarter of every one of the eighteen modulation chunks on each chip.
pub type Modulation<T = Time, W = Width, X = Hbm> = modulation::Projection<
    T,
    W,
    Chips,
    Part<W, 4>,
    <X as Memory>::Tensor<bf16, Chips, m![{ Symbol<modulation::Modulations<Part<W, 4>>> }, T]>,
    <X as Memory>::Tensor<f32, Chips, m![{ Symbol<modulation::Modulations<Part<W, 4>>> }, 1 # 8]>,
>;

/// Packed adaptive RMS normalization with the family's epsilon.
pub type AdaptiveNorm<W = Width, X = Hbm> =
    AdaptiveRms<W, EPSILON, Chips, <X as Memory>::Tensor<bf16, Chips, m![W]>>;

/// Two checkpoint-named pre-norm text layers, followed by the final RMS norm.
pub type Refiner<W = Width, A = Attended, F = Expanded, X = Hbm> = Then<
    Named<"blocks", Layers<[RefinerLayer<W, A, F, X>; 2], Unrolled>>,
    Named<"final_norm", Norm<W, X>>,
>;

/// Refiner layers use H3's `norm1`/`attn`/`norm2`/`mlp` names and no modulation or rotary.
pub type RefinerLayer<W = Width, A = Attended, F = Expanded, X = Hbm> = Sequential<
    Residual<Named<"norm1", Norm<W, X>>, Named<"attn", RefinerAttention<W, A, X>>, Add>,
    Residual<Named<"norm2", Norm<W, X>>, Named<"mlp", RefinerFeedForward<W, F, X>>, Add>,
>;

/// Every chip refines the same text with all heads; no inter-chip gather or exchange is needed.
pub type RefinerAttention<W = Width, A = Attended, X = Hbm> = attention::Full<
    Sequential<
        First,
        attention::NormalizedQkv<
            Named<
                "qkv_proj",
                FusedQkv<
                    W,
                    A,
                    full::Projection<Dense<W, full::Joint<A>, X>, flash::Head, Unrotated>,
                >,
            >,
            Named<"q_norm", HeadNorm<Unrotated, X>>,
            Named<"k_norm", HeadNorm<Unrotated, X>>,
        >,
    >,
    flash::Compact<Heads<A>, SCALE>,
    Named<"out_proj", Dense<A, W, X>>,
>;

/// Replicated ordinary dense matrices for the text-only refiner.
pub type RefinerFeedForward<W = Width, F = Expanded, X = Hbm> =
    feedforward::Mlp<Named<"fc1", Dense<W, GateUp<F>, X>>, SwiGlu, Named<"fc2", Dense<F, W, X>>>;

/// The gain is FP32, reordered with Q/K, and epsilon is generated on every chip during loading.
pub type HeadNorm<R = Rotated, X = Hbm> = full::HeadNorm<
    flash::Head,
    Chips,
    <X as Memory>::Tensor<f32, Chips, m![{ Symbol<flash::Head> }]>,
    <X as Memory>::Tensor<f32, Chips, m![1]>,
    EPSILON,
    R,
>;

/// H3 stores every head once in one head-major matrix. Admit that exact source shape before
/// invoking the shared conversion, whose local/replicated alternatives serve other models.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(P))]
pub struct FusedQkv<In, Attended, P> {
    #[load(rename = "")]
    #[residency(transparent)]
    pub projection: P,
    shape: PhantomData<(In, Attended)>,
}

impl<I, In, Attended, P> !crate::Op<I> for FusedQkv<In, Attended, P> {}

impl<I, E, S, In, Attended, P: Module<I, E, S>> Module<I, E, S> for FusedQkv<In, Attended, P> {
    type Output = P::Output;

    fn forward(&self, device: &mut Device, env: E, state: S, input: I) -> Self::Output {
        self.projection.forward(device, env, state, input)
    }
}

impl<'a, In: AxisName, Attended: AxisName, P: Load<&'a Checkpoint>> Load<&'a Checkpoint>
    for FusedQkv<In, Attended, P>
{
    fn load(cx: &mut Context<'_>, source: &'a Checkpoint) -> Result<Self, LoadError> {
        cx.checkpoint(source, |cx| {
            cx.field("", "weight", |cx| {
                let expected = const {
                    assert!(In::SIZE > 0 && Attended::SIZE > 0 && Attended::SIZE <= usize::MAX / 3);
                    [3 * Attended::SIZE, In::SIZE]
                };
                let (name, (_, shape)) = cx.header(source)?;
                if shape != expected {
                    return Err(LoadError::InvalidData {
                        name,
                        reason: format!("H3 head-major QKV shape {shape:?}, expected {expected:?}"),
                    });
                }
                Ok(())
            })?;
            P::load(cx, source).map(|projection| Self {
                projection,
                shape: PhantomData,
            })
        })
    }
}

pub type Norm<W = Width, X = Hbm> =
    RmsNorm<W, EPSILON, Chips, Weight<m![W], Chips, <X as Memory>::Tensor<bf16, Chips, m![W]>>>;

pub type Grouped<I, O, X = Hbm> =
    linear::Grouped<I, O, Chips, <X as Memory>::Tensor<bf16, Chips, linear::grouped::Layout<I, O>>>;

pub type Dense<I, O, X = Hbm> = Linear<I, O, Chips, <X as Memory>::Tensor<bf16, Chips, m![O, I]>>;

/// Maximum-2048 gather allocation: one 256-row first tile and ten 1024-row bodies.
pub type GatherTiles = InTiles<GATHER_BODY, DEFAULT_GATHER_TILES, GATHER_HEAD>;

/// Score multiplier fixed by the 128-feature heads.
pub const SCALE: usize = geometry::scale(flash::Head::SIZE);

axes![
    Attended = 7168,
    Expanded = 14336,
    Rotated = 96,
    Unrotated = 0
];

/// Number of 128-feature heads represented by an attended feature width.
#[derive(Clone, Debug)]
pub struct Heads<W>(PhantomData<W>);

impl<W: AxisName> AxisName for Heads<W> {
    const NAME: Ident = Ident::new("Heads");
    const SIZE: usize = {
        assert!(W::SIZE > 0 && W::SIZE.is_multiple_of(flash::Head::SIZE));
        W::SIZE / flash::Head::SIZE
    };
}

#[cfg(test)]
mod tests {
    use super::super::test_memory::{Host, IMAGES, Image, Leaf};
    use super::*;
    use crate::load::{Checkpoint, Context, Load, LoadError};
    use crate::tensor::memory::residency::{Leaves, Path, Residency, Visit};
    use safetensors::Dtype;
    use std::convert::Infallible;

    axes![W = 256, A = 512, F = 512, T = 64];
    type Joint<X> = Named<"transformer.blocks.0", Block<W, A, F, T, X>>;
    type Text<X> = Named<"transformer.token_refiner", Refiner<W, A, F, X>>;

    struct Borrow;
    impl<'a> Leaves<'a, Host, &'a Host> for Borrow {
        type Error = Infallible;
        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &'a Leaf<D, C, S>,
            _: Path<'_>,
        ) -> Result<&'a Leaf<D, C, S>, Self::Error> {
            Ok(leaf)
        }
    }
    struct Count(usize);
    impl Visit<Host> for Count {
        type Error = Infallible;
        fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
            &mut self,
            leaf: &Leaf<D, C, S>,
            _: Path<'_>,
        ) -> Result<(), Self::Error> {
            assert!(!leaf.bytes.is_empty());
            self.0 += 1;
            Ok(())
        }
    }

    fn value(row: usize, column: usize, seed: usize) -> f32 {
        ((row * 29 + column * 7 + seed * 11) % 503) as f32 / 256.0 - 251.0 / 256.0
    }

    fn tensor(cp: &mut Checkpoint, name: &str, shape: &[usize], dtype: Dtype, seed: usize) {
        let width = *shape.last().unwrap();
        let values: Vec<_> = (0..shape.iter().product())
            .map(|at| value(at / width, at % width, seed))
            .collect();
        let bytes = match dtype {
            Dtype::BF16 => {
                bf16::to_buf(&values.into_iter().map(bf16::from_f32).collect::<Vec<_>>())
            }
            Dtype::F32 => f32::to_buf(&values),
            _ => unreachable!(),
        };
        cp.insert(name, dtype, shape, bytes).unwrap();
    }

    fn layer(cp: &mut Checkpoint, prefix: &str, joint: bool) {
        for (name, shape, dtype, seed) in [
            ("attn.qkv_proj.weight", vec![1536, 256], Dtype::BF16, 1),
            ("attn.out_proj.weight", vec![256, 512], Dtype::BF16, 2),
            ("attn.q_norm.weight", vec![128], Dtype::F32, 3),
            ("attn.k_norm.weight", vec![128], Dtype::F32, 4),
            ("mlp.fc1.weight", vec![1024, 256], Dtype::BF16, 5),
            ("mlp.fc2.weight", vec![256, 512], Dtype::BF16, 6),
            ("norm1.weight", vec![256], Dtype::BF16, 7),
            ("norm2.weight", vec![256], Dtype::BF16, 8),
        ] {
            tensor(cp, &format!("{prefix}.{name}"), &shape, dtype, seed);
        }
        if joint {
            tensor(
                cp,
                &format!("{prefix}.adaln_proj.linear.weight"),
                &[4608, 64],
                Dtype::BF16,
                9,
            );
            tensor(
                cp,
                &format!("{prefix}.adaln_proj.linear.bias"),
                &[4608],
                Dtype::F32,
                10,
            );
        }
    }

    fn checkpoint(joint: bool) -> Checkpoint {
        let mut cp = Checkpoint::default();
        if joint {
            layer(&mut cp, "transformer.blocks.0", true);
        } else {
            for index in 0..2 {
                layer(
                    &mut cp,
                    &format!("transformer.token_refiner.blocks.{index}"),
                    false,
                );
            }
            tensor(
                &mut cp,
                "transformer.token_refiner.final_norm.weight",
                &[256],
                Dtype::BF16,
                11,
            );
        }
        cp
    }

    fn image(segment: &str) -> Image {
        select_image(&[segment])
    }

    fn select_image(segments: &[&str]) -> Image {
        IMAGES.with_borrow(|images| {
            let found: Vec<_> = images
                .iter()
                .filter(|image| segments.iter().all(|segment| image.path.contains(segment)))
                .collect();
            assert_eq!(
                found.len(),
                1,
                "{segments:?}: {:?}",
                images.iter().map(|i| &i.path).collect::<Vec<_>>()
            );
            found[0].clone()
        })
    }

    fn bf16_at(image: &Image, at: usize) -> bf16 {
        bf16::from_bits(u16::from_le_bytes(
            image.bytes[2 * at..2 * at + 2].try_into().unwrap(),
        ))
    }

    fn f32_at(image: &Image, at: usize) -> f32 {
        f32::from_le_bytes(image.bytes[4 * at..4 * at + 4].try_into().unwrap())
    }

    // Independent checkpoint-dimension mapping, including both unrotated gaps.
    fn paired(d: usize) -> usize {
        match d {
            0..48 => d,
            48..96 => d + 16,
            96..112 => d - 48,
            _ => d,
        }
    }

    #[test]
    fn joint_loading_shards_heads_and_each_modulation_chunk() {
        IMAGES.with_borrow_mut(Vec::clear);
        let cp = checkpoint(true);
        let _tree = Joint::<Host>::load(&mut Context::new(None), &cp).unwrap();
        let qkv = image("qkv_proj");
        let k = linear::grouped::Chunk::<W, full::Joint<Part<A, 4>>>::SIZE;
        for chip in 0..4 {
            for term in 0..3 {
                for d in 0..128 {
                    let out = term * 128 + if term == 2 { d } else { paired(d) };
                    for column in [0, 31, 127, 128, 255] {
                        let at = chip * qkv.elements
                            + ((out / 32 * (256 / k) + column / k) * 32 + out % 32) * k
                            + column % k;
                        assert_eq!(
                            bf16_at(&qkv, at),
                            bf16::from_f32(value((chip * 3 + term) * 128 + d, column, 1))
                        );
                    }
                }
            }
        }
        for (segment, seed) in [("q_norm.weight", 3), ("k_norm.weight", 4)] {
            let gain = image(segment);
            for chip in 0..4 {
                for d in 0..128 {
                    assert_eq!(f32_at(&gain, chip * 128 + paired(d)), value(0, d, seed));
                }
            }
        }
        for segment in ["q_norm.epsilon", "k_norm.epsilon"] {
            let epsilon = image(segment);
            assert_eq!(epsilon.bytes.len(), 4 * 4);
            for chip in 0..4 {
                assert_eq!(f32_at(&epsilon, chip), 1e-5);
            }
        }
        let weight = image("adaln_proj.linear.weight");
        let bias = image("adaln_proj.linear.bias");
        for chip in 0..4 {
            for chunk in 0..18 {
                for feature in 0..64 {
                    let stored = chunk * 256 + chip * 64 + feature;
                    let local = chunk * 64 + feature;
                    for column in [0, 31, 63] {
                        assert_eq!(
                            bf16_at(&weight, chip * weight.elements + local * 64 + column),
                            bf16::from_f32(value(stored, column, 9))
                        );
                    }
                    let at = chip * bias.elements + local * 8;
                    assert_eq!(f32_at(&bias, at), value(0, stored, 10));
                    for lane in 1..8 {
                        assert_eq!(f32_at(&bias, at + lane).to_bits(), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn refiner_loading_replicates_unrotated_heads_and_loads_its_final_norm() {
        IMAGES.with_borrow_mut(Vec::clear);
        let cp = checkpoint(false);
        let _tree = Text::<Host>::load(&mut Context::new(None), &cp).unwrap();
        for index in 0..2 {
            let prefix = format!("layers.{index}.");
            let qkv = select_image(&[&prefix, "qkv_proj"]);
            for chip in 0..4 {
                for head in 0..4 {
                    for term in 0..3 {
                        for d in [0, 47, 48, 95, 96, 111, 112, 127] {
                            let out = term * 512 + head * 128 + d;
                            for column in [0, 127, 255] {
                                assert_eq!(
                                    bf16_at(&qkv, chip * qkv.elements + out * 256 + column),
                                    bf16::from_f32(value((head * 3 + term) * 128 + d, column, 1))
                                );
                            }
                        }
                    }
                }
            }
            let gain = select_image(&[&prefix, "q_norm.weight"]);
            for chip in 0..4 {
                for d in 0..128 {
                    assert_eq!(f32_at(&gain, chip * 128 + d), value(0, d, 3));
                }
            }
        }
        let norm = image("final_norm.weight");
        assert_eq!(norm.elements, 256);
        for chip in 0..4 {
            for d in 0..256 {
                assert_eq!(
                    bf16_at(&norm, chip * 256 + d),
                    bf16::from_f32(value(0, d, 11))
                );
            }
        }
    }

    #[test]
    fn strict_block_loading_rejects_unexpected_bias_and_wrong_qkv_shape() {
        let mut cp = checkpoint(true);
        tensor(
            &mut cp,
            "transformer.blocks.0.attn.out_proj.bias",
            &[256],
            Dtype::F32,
            1,
        );
        let error = Joint::<Host>::load(&mut Context::new(None), &cp)
            .err()
            .unwrap();
        assert!(
            matches!(error, LoadError::Unexpected(names) if names == ["transformer.blocks.0.attn.out_proj.bias"])
        );
        let mut cp = checkpoint(true);
        assert!(cp.remove("transformer.blocks.0.attn.qkv_proj.weight"));
        cp.insert(
            "transformer.blocks.0.attn.qkv_proj.weight",
            Dtype::BF16,
            &[3, 256],
            vec![0; 3 * 256 * 2],
        )
        .unwrap();
        let error = Joint::<Host>::load(&mut Context::new(None), &cp)
            .err()
            .unwrap();
        assert!(
            matches!(error, LoadError::InvalidData { name, .. } if name == "transformer.blocks.0.attn.qkv_proj.weight")
        );
        let mut cp = checkpoint(true);
        assert!(cp.remove("transformer.blocks.0.attn.qkv_proj.weight"));
        cp.insert(
            "transformer.blocks.0.attn.qkv_proj.weight",
            Dtype::U8,
            &[1536, 256],
            vec![0; 1536 * 256],
        )
        .unwrap();
        let error = Joint::<Host>::load(&mut Context::new(None), &cp)
            .err()
            .unwrap();
        assert!(
            matches!(error, LoadError::InvalidData { name, .. } if name == "transformer.blocks.0.attn.qkv_proj.weight")
        );
        let error = Joint::<Host>::load(&mut Context::new(None), &Checkpoint::default())
            .err()
            .unwrap();
        assert!(
            matches!(error, LoadError::Missing(names) if names.iter().any(|n| n == "transformer.blocks.0.attn.qkv_proj.weight"))
        );
    }

    #[test]
    fn refuses_joint_checkpoint_with_only_one_chips_heads() {
        let mut cp = checkpoint(true);
        let name = "transformer.blocks.0.attn.qkv_proj.weight";
        assert!(cp.remove(name));
        tensor(&mut cp, name, &[384, 256], Dtype::BF16, 1);
        let error = Joint::<Host>::load(&mut Context::new(None), &cp)
            .err()
            .expect("H3 requires all heads in the stored QKV matrix");
        assert!(matches!(error, LoadError::InvalidData { name: failed, .. } if failed == name));
    }

    #[test]
    fn refuses_refiner_checkpoint_with_distinct_heads_per_chip() {
        let mut cp = checkpoint(false);
        let name = "transformer.token_refiner.blocks.0.attn.qkv_proj.weight";
        assert!(cp.remove(name));
        tensor(&mut cp, name, &[6144, 256], Dtype::BF16, 1);
        let error = Text::<Host>::load(&mut Context::new(None), &cp)
            .err()
            .expect("H3 refiner stores one matrix to replicate across chips");
        assert!(matches!(error, LoadError::InvalidData { name: failed, .. } if failed == name));
    }

    #[test]
    fn default_and_reduced_trees_support_canonical_residency_and_borrowed_loading() {
        fn canonical<M: Residency>()
        where
            for<'a> M::In<&'a Hbm>: Load<&'a M::In<Hbm>>,
        {
        }
        canonical::<Block>();
        canonical::<Refiner>();
        let cp = checkpoint(true);
        let tree = Joint::<Host>::load(&mut Context::new(None), &cp).unwrap();
        drop(cp);
        let _borrowed =
            Joint::<Hbm>::map::<Host, &Host, _>(&tree, &mut Borrow, Path::ROOT).unwrap();
        let _loaded = Joint::<&Host>::load(&mut Context::new(None), &tree).unwrap();
        let mut count = Count(0);
        Joint::<Hbm>::visit::<Host, _>(&tree, &mut count, Path::ROOT).unwrap();
        assert_eq!(count.0, 12);
        let cp = checkpoint(false);
        let tree = Text::<Host>::load(&mut Context::new(None), &cp).unwrap();
        let mut count = Count(0);
        Text::<Hbm>::visit::<Host, _>(&tree, &mut count, Path::ROOT).unwrap();
        assert_eq!(count.0, 21);
    }

    // Full's context hands (hidden, positions) to QKV. First drops the unused refiner position
    // operand without inserting another borrow, preserving this public Module boundary.
    const _: () = {
        use crate::ops::index::packed::Span;
        use crate::tensor::scalar::Value;
        type R = Span<2048, 2048>;
        type Hidden = HbmTensor<bf16, Chips, m![R, Width]>;
        type Visibility = flash::Visible<crate::tensor::axis::Rows<3>, Chips>;
        type QkvState = ((), ((), (), ()));
        type LayerState = ((QkvState, ()), ((), ((), ())));
        type State = ([LayerState; 2], ());
        fn refiner<M>()
        where
            for<'e, 's> M: Module<
                    Hidden,
                    &'e attention::Context<(), (), (&'e Visibility, &'e Value<Chips>)>,
                    &'s mut State,
                    Output = Hidden,
                >,
        {
        }
        let _ = refiner::<Refiner>;
        let _ = refiner::<Refiner<Width, Attended, Expanded, &Hbm>>;
    };
}
