//! The FP8 block contract of a [`BlockScaled`](super::BlockScaled) weight: E4M3 codes with one
//! FP32 scale per 128 x 128 block, whose inputs quantize at run time in groups of 128; and how a
//! BF16 weight quantizes at load into E4M3 codes with an FP32 scale per output row ([`Granularity`]).
//!
//! - A group's scale is its largest magnitude times the reciprocal of the largest E4M3 value, never
//!   below `1e-10` times it, so an all-zero group quantizes to zeros instead of dividing by zero.
//! - Each 128-input block contracts on its own; its FP32 sum takes the block's activation scale,
//!   then its weight scale, before the blocks add up.
//! - A checkpoint's block scales parse here once, whatever their spelling or float type
//!   ([`grid`]); a form with one scale per output row folds them under each row's largest
//!   ([`folded`]).

use core::marker::PhantomData;
use std::iter;

use furiosa_opt_std::prelude::*;
use safetensors::Dtype;

use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::tensor::parallel::Chip;

// The inputs of one quantization group, apart from the group they belong to.
axes![Group = 128];

/// Inputs that share one activation scale, and each side of a weight block.
pub const GROUP: usize = Group::SIZE;

/// Each 128-input block's FP32 sum takes its activation scale, then its weight scale, before the
/// blocks add up.
#[derive(Clone, Copy, Default, DeviceSend, Load)]
pub struct Block;

/// One FP32 weight scale per four output rows and 128 inputs, in [output, input] order, with the
/// blocks of each input half padded to an even count: a checkpoint's 128 x 128 block scale repeats
/// over the 32 four-row tiles of its block.
pub type Scales<In, Out> =
    m![Out / 4, In / (In / 2), In % (In / 2) / 128 # ({ Symbol<Blocks<In>> })];

/// The 128-input blocks of one input half, padded to an even count so that a row of FP32 scales
/// fills whole 8-byte SRAM words.
#[derive(Clone, Debug)]
pub struct Blocks<In>(PhantomData<In>);

impl<In: AxisName> AxisName for Blocks<In> {
    const NAME: Ident = Ident::new("Blocks");
    const SIZE: usize = (In::SIZE / (2 * GROUP)).next_multiple_of(2);
}

/// A block-scaled E4M3 projection of `In` inputs and `Out` outputs in its row-scaled device form:
/// its codes `W`, each row's block scales folded under its largest ([`folded`]), and those row
/// scales `S`. A form without row scales (`S = ()`) holds each row's scale in its codes, exactly.
#[derive(DeviceSend)]
pub struct Rows<In, Out, W, S = ()>(pub W, pub S, PhantomData<fn() -> (In, Out)>);

impl<In, Out, W, S> Rows<In, Out, W, S> {
    pub fn new(codes: W, scales: S) -> Self {
        Self(codes, scales, PhantomData)
    }
}

impl<In: M + 'static, Out: M + 'static, W, S> Rows<In, Out, W, S>
where
    W: Load<HostTensor<f8e4m3, m![{ Chip }, { m![{ Out }, { In }] }]>>,
{
    /// The codes `W` of `codes` on every chip, and `scales` made into `S`.
    fn place(
        codes: Vec<f8e4m3>,
        scales: impl FnOnce(&mut Context<'_>) -> Result<S, LoadError>,
        context: &mut Context<'_>,
    ) -> Result<Self, LoadError> {
        let codes = HostTensor::from_vec(codes.repeat(Chip::SIZE));
        Ok(Self::new(
            context.field("0", "", |cx| W::load(cx, codes))?,
            context.field("1", "", scales)?,
        ))
    }
}

/// The codes at the context's name with each row's scale folded in, refused unless every code
/// times its row's scale is an E4M3 value.
impl<'a, In: M + 'static, Out: M + 'static, W> Load<&'a Checkpoint> for Rows<In, Out, W, ()>
where
    W: Load<HostTensor<f8e4m3, m![{ Chip }, { m![{ Out }, { In }] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let (codes, rows) = folded::<In, Out>(from, context)?;
        let exact = |(index, code): (usize, f8e4m3)| {
            let value = f64::from(code.to_f32()) * f64::from(rows[index / In::SIZE]);
            let folded = f8e4m3::from_f32(value as f32);
            (f64::from(folded.to_f32()) == value).then_some(folded)
        };
        let codes = codes
            .into_iter()
            .enumerate()
            .map(exact)
            .collect::<Option<Vec<_>>>();
        let codes = codes.ok_or_else(|| LoadError::InvalidData {
            name: context.name(),
            reason: "a row's block scale does not fold into its E4M3 codes exactly, and the \
                form holds no row scales"
                .into(),
        })?;
        Self::place(codes, |_| Ok(()), context)
    }
}

/// The codes at the context's name and their FP32 row scales.
impl<'a, In: M + 'static, Out: M + 'static, W, L: M + 'static> Load<&'a Checkpoint>
    for Rows<In, Out, W, HbmTensor<f32, Chip, L>>
where
    W: Load<HostTensor<f8e4m3, m![{ Chip }, { m![{ Out }, { In }] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let (codes, rows) = folded::<In, Out>(from, context)?;
        let rows = HostTensor::<f32, m![{ Chip }, { Out }]>::from_vec(rows.repeat(Chip::SIZE));
        Self::place(codes, |cx| Load::load(cx, rows), context)
    }
}

/// Each head's squared row scale `s²`, `K` copies of it, for a projection whose rows of one head
/// share one row scale: a head norm of the rows its codes give restores its epsilon from it.
#[derive(DeviceSend)]
pub struct Squares<T>(pub T);

/// The codes at the context's name and each head's squared row scale; a head whose rows hold
/// different scales is refused.
impl<'a, In: M + 'static, Heads: AxisName, Head: AxisName, W, K: M + 'static> Load<&'a Checkpoint>
    for Rows<In, m![Heads, Head], W, Squares<HbmTensor<f32, Chip, m![Heads, { K }]>>>
where
    W: Load<HostTensor<f8e4m3, m![{ Chip }, { m![{ m![Heads, Head] }, { In }] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let (codes, rows) = folded::<In, m![Heads, Head]>(from, context)?;
        let squares = rows
            .chunks_exact(Head::SIZE)
            .map(|head| (head.iter().all(|&row| row == head[0])).then_some(head[0] * head[0]))
            .collect::<Option<Vec<f32>>>()
            .ok_or_else(|| LoadError::InvalidData {
                name: context.name(),
                reason: "the rows of a head hold different block scales".into(),
            })?;
        let squares = squares
            .iter()
            .flat_map(|&square| std::iter::repeat_n(square, K::SIZE))
            .collect::<Vec<_>>()
            .repeat(Chip::SIZE);
        let squares = HostTensor::<f32, m![{ Chip }, { m![Heads, { K }] }]>::from_vec(squares);
        Self::place(
            codes,
            |cx| Ok(Squares(cx.field("0", "", |cx| Load::load(cx, squares))?)),
            context,
        )
    }
}

/// A weight scale's spellings, canonical first: both multiply the E4M3 weight.
const SPELLINGS: [&str; 2] = ["weight_scale", "weight_scale_inv"];

/// Which values of a BF16 weight share an E4M3 scale when it quantizes at load: those of the
/// whole projection and of the projections fused with it, or those of one output row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Granularity {
    Tensor,
    Channel,
}

/// A weight's values in whole rows, every one finite: what a load quantizes.
pub(crate) struct Finite {
    values: Vec<f32>,
    inputs: usize,
}

impl Finite {
    /// `values` in rows of `inputs` each, unless one is not finite.
    pub(crate) fn new(values: Vec<f32>, inputs: usize) -> Option<Self> {
        assert!(
            inputs > 0 && values.len().is_multiple_of(inputs),
            "whole rows"
        );
        (values.iter().all(|value| value.is_finite())).then_some(Self { values, inputs })
    }

    /// The largest magnitude of the values.
    pub(crate) fn largest(&self) -> f32 {
        Self::magnitude(&self.values)
    }

    fn magnitude(values: &[f32]) -> f32 {
        values
            .iter()
            .fold(0.0f32, |max, value| max.max(value.abs()))
    }
}

impl Granularity {
    /// The E4M3 codes of `weight` and each row's FP32 scale, as frameworks quantize online: a
    /// part's scale is its largest magnitude over 448, a tensor's part also covering `fused`, the
    /// largest magnitude of the weights fused with it. A code is its value times the reciprocal of
    /// its scale, saturated, to nearest even; a part of zeros has scale zero and zero codes.
    pub(crate) fn quantize(self, weight: &Finite, fused: f32) -> (Vec<f8e4m3>, Vec<f32>) {
        // The largest finite E4M3 value: the conversion saturates there.
        let largest = f32::from(f8e4m3::from_f32(f32::MAX));
        let rows = weight.values.chunks_exact(weight.inputs);
        let scales: Vec<f32> = match self {
            Self::Tensor => vec![weight.largest().max(fused) / largest; rows.len()],
            Self::Channel => (rows.clone())
                .map(|row| Finite::magnitude(row) / largest)
                .collect(),
        };
        let codes = rows
            .zip(&scales)
            .flat_map(|(row, &scale)| {
                let reciprocal = if scale == 0.0 { 0.0 } else { 1.0 / scale };
                row.iter().map(move |value| {
                    f8e4m3::from_f32((value * reciprocal).min(largest).max(-largest))
                })
            })
            .collect();
        (codes, scales)
    }
}

/// A checkpoint's `[out / 128, in / 128]` block scales `grid` in the device form [`Scales`]: each
/// block row once per four output rows, each input half's blocks padded to an even count.
pub(crate) fn layout<In: AxisName, Out: AxisName>(grid: &[f32]) -> Vec<f32> {
    let (half, padded) = (In::SIZE / (2 * GROUP), Blocks::<In>::SIZE);
    (0..Out::SIZE / 4)
        .flat_map(|tile| (0..2 * padded).map(move |at| (tile, at / padded, at % padded)))
        .map(|(tile, side, block)| match block < half {
            true => grid[tile * 4 / GROUP * (2 * half) + side * half + block],
            false => 0.0,
        })
        .collect()
}

/// The `[out / 128, in / 128]` FP32 block scales of the `Out` x `In` projection, stored under
/// one of `names` (its spellings, canonical first) as FP32 or BF16, each finite and positive.
pub(crate) fn grid<In: M, Out: M>(
    from: &Checkpoint,
    context: &mut Context<'_>,
    names: [String; 2],
) -> Result<Vec<f32>, LoadError> {
    let mut stored = names.iter().filter(|name| from.get(name).is_some());
    let (name, second) = (stored.next(), stored.next());
    let name = name.ok_or_else(|| LoadError::Missing(vec![names[0].clone()]))?;
    context.read(name);
    let invalid = |reason: &str| LoadError::InvalidData {
        name: name.clone(),
        reason: reason.into(),
    };
    if second.is_some() {
        return Err(invalid("scale has two spellings"));
    }
    let view = from.get(name).expect("found above");
    let shape = [Out::SIZE.div_ceil(GROUP), In::SIZE.div_ceil(GROUP)];
    if view.shape() != shape {
        return Err(invalid(&format!(
            "expected block scales of shape {shape:?}"
        )));
    }
    let values = match view.dtype() {
        Dtype::F32 | Dtype::BF16 => Checkpoint::floats(&view),
        _ => None,
    };
    let values = values.ok_or_else(|| invalid("expected FP32 or BF16 scales"))?;
    match values.iter().all(|value| value.is_finite() && *value > 0.0) {
        true => Ok(values),
        false => Err(invalid("scale must be finite and positive")),
    }
}

/// The E4M3 codes of the `Out` x `In` projection whose `weight` is the context's name, each
/// requantized under its row's largest block scale, and those row scales.
///
/// Contract: a code times its block's scale is the folded code times its row's scale within one
/// E4M3 rounding of the product (a relative 2^-4, or the smallest E4M3 subnormal times the row
/// scale), exactly when every block-to-row scale ratio is a power of two that keeps the code normal.
pub(crate) fn folded<In: M, Out: M>(
    from: &Checkpoint,
    context: &mut Context<'_>,
) -> Result<(Vec<f8e4m3>, Vec<f32>), LoadError> {
    let shape = [Out::SIZE, In::SIZE];
    let mut children = Children::default();
    let codes =
        context
            .tensor(from)
            .and_then(|(name, view)| match (view.dtype(), view.shape() == shape) {
                (Dtype::F8_E4M3, true) => Ok(view.data().to_vec()),
                _ => Err(LoadError::InvalidData {
                    name,
                    reason: format!("expected E4M3 codes of shape {shape:?}"),
                }),
            });
    let codes = children.ask(codes)?;
    let names = SPELLINGS.map(|spelling| context.sibling(spelling));
    let blocks = children.ask(grid::<In, Out>(from, context, names))?;
    let (codes, blocks) = children.node(codes.zip(blocks))?;
    let width = In::SIZE.div_ceil(GROUP);
    let rows: Vec<f32> = blocks
        .chunks_exact(width)
        .flat_map(|row| iter::repeat_n(row.iter().copied().fold(0.0, f32::max), GROUP))
        .take(Out::SIZE)
        .collect();
    let folded = codes
        .iter()
        .enumerate()
        .map(|(index, &code)| {
            let (row, column) = (index / In::SIZE, index % In::SIZE);
            let block = blocks[row / GROUP * width + column / GROUP];
            let value = <f8e4m3 as ScalarBytes>::from_le_bytes(&[code]).to_f32();
            f8e4m3::from_f32(value * (block / rows[row]))
        })
        .collect();
    Ok((folded, rows))
}

/// The block scales of the projection at the context's name in the device form, on every chip of
/// `C`.
pub(crate) fn scales<In: AxisName, Out: AxisName, C: M + 'static>(
    from: &Checkpoint,
    context: &mut Context<'_>,
) -> Result<HostTensor<f32, m![{ C }, { Scales<In, Out> }]>, LoadError> {
    let scope = context.name();
    let names = SPELLINGS.map(|spelling| match scope.as_str() {
        "" => spelling.to_string(),
        scope => format!("{scope}.{spelling}"),
    });
    let grid = grid::<m![In], m![Out]>(from, context, names)?;
    Ok(HostTensor::from_vec(
        layout::<In, Out>(&grid).repeat(C::SIZE),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::BlockScaled;
    use super::*;

    axes![In = 256, Out = 256];

    /// A host leaf holding its values as the load built them.
    struct Host<D, C, S>(Vec<D>, PhantomData<fn() -> (C, S)>);

    impl<'a, D: MaterializableScalar, C: M + 'static, S: M + 'static> Load<&'a Checkpoint>
        for Host<D, C, S>
    {
        fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
            from.host::<D, S, m![{ C }, { S }]>(context)
                .map(|host| Host(host.into_vec(), PhantomData))
        }
    }

    impl<D: MaterializableScalar, C: M + 'static, S: M + 'static>
        Load<HostTensor<D, m![{ C }, { S }]>> for Host<D, C, S>
    {
        fn load(
            _: &mut Context<'_>,
            host: HostTensor<D, m![{ C }, { S }]>,
        ) -> Result<Self, LoadError> {
            Ok(Host(host.into_vec(), PhantomData))
        }
    }

    /// A BF16 weight, by bits, and its codes and scales under torch's `float8_e4m3fn` conversion of
    /// `value * (1 / scale)`: the whole weight's scale, then each row's.
    const WEIGHT: [[u16; 8]; 4] = [
        [48978, 16075, 16230, 49074, 48683, 16018, 48932, 48997],
        [15678, 48347, 48494, 15386, 15633, 15695, 15421, 15711],
        [49304, 49017, 16569, 49021, 16152, 16406, 16456, 49163],
        [47452, 47458, 47854, 47541, 47879, 14897, 47789, 47794],
    ];
    const TENSOR: ([[u8; 8]; 4], u32) = (
        [
            [232, 95, 105, 237, 213, 91, 228, 233],
            [70, 192, 201, 52, 67, 72, 54, 72],
            [252, 233, 126, 234, 100, 115, 119, 243],
            [136, 137, 161, 142, 162, 21, 157, 157],
        ],
        0x3c53_6db7,
    );
    const CHANNEL: ([[u8; 8]; 4], [u32; 4]) = (
        [
            [248, 112, 121, 254, 229, 107, 245, 249],
            [123, 245, 254, 105, 121, 124, 107, 125],
            [252, 233, 126, 234, 100, 115, 119, 243],
            [227, 228, 252, 233, 254, 113, 249, 249],
        ],
        [0x3b4b_6db7, 0x3908_0000, 0x3c53_6db7, 0x369a_4925],
    );

    /// E4M3 codes of an `Out` x `In` projection: normal values of both signs, a different one
    /// almost everywhere.
    fn codes() -> Vec<f8e4m3> {
        (0..Out::SIZE * In::SIZE)
            .map(|index| {
                let magnitude = 0.125 + ((index * 37) % 61) as f32 / 8.0;
                f8e4m3::from_f32(if index % 3 == 0 {
                    -magnitude
                } else {
                    magnitude
                })
            })
            .collect()
    }

    /// A checkpoint of the projection `proj` with `codes` and `scales` stored as `(name, dtype,
    /// shape, values)`.
    fn checkpoint(codes: &[f8e4m3], scales: &[(&str, Dtype, &[usize], &[f32])]) -> Checkpoint {
        let mut checkpoint = Checkpoint::default();
        let shape = [Out::SIZE, In::SIZE];
        let data = f8e4m3::to_buf(codes);
        checkpoint
            .insert("proj.weight", Dtype::F8_E4M3, &shape, data)
            .unwrap();
        for &(name, dtype, shape, values) in scales {
            let data = match dtype {
                Dtype::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                _ => values
                    .iter()
                    .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                    .collect(),
            };
            checkpoint
                .insert(&format!("proj.{name}"), dtype, shape, data)
                .unwrap();
        }
        checkpoint
    }

    fn fold(source: &Checkpoint) -> Result<(Vec<f8e4m3>, Vec<f32>), LoadError> {
        let mut context = Context::new(None);
        context.field("", "proj", |cx| {
            cx.field("weight", "weight", |cx| {
                folded::<m![In], m![Out]>(source, cx)
            })
        })
    }

    /// The value a checkpoint code means under its block's scale.
    fn stored(codes: &[f8e4m3], blocks: &[f32], index: usize) -> f32 {
        let (row, column) = (index / In::SIZE, index % In::SIZE);
        codes[index].to_f32() * blocks[row / GROUP * 2 + column / GROUP]
    }

    /// Folding block scales that differ within a row, by factors no power of two gives, keeps each
    /// value within one E4M3 rounding; a row scale of one block alone would not.
    #[test]
    fn folds_nonuniform_block_scales_within_one_rounding() {
        let (codes, blocks) = (codes(), [0.37f32, 1.9, 0.0113, 5.3]);
        let rounded = blocks.map(|scale| half::bf16::from_f32(scale).to_f32());
        let source = checkpoint(
            &codes,
            &[("weight_scale_inv", Dtype::BF16, &[2, 2], &blocks)],
        );
        let (folded, rows) = fold(&source).unwrap();
        assert_eq!(rows[0], rounded[1]);
        assert_eq!(rows[Out::SIZE - 1], rounded[3]);
        let (mut worst, mut shortcut) = (0.0f32, 0.0f32);
        for index in 0..codes.len() {
            let want = stored(&codes, &rounded, index);
            let row = rows[index / In::SIZE];
            let bound = (want.abs() / 16.0).max(row / 1024.0);
            worst = worst.max((folded[index].to_f32() * row - want).abs() / bound);
            let first = rounded[index / In::SIZE / GROUP * 2];
            shortcut = shortcut.max((codes[index].to_f32() * first - want).abs() / bound);
        }
        assert!(worst <= 1.0, "{worst}");
        assert!(
            shortcut > 1.0,
            "the fixture tells one row scale from its blocks"
        );
    }

    /// Folding block scales that are powers of two apart keeps every normal value exactly.
    #[test]
    fn folds_power_of_two_block_scales_exactly() {
        let (codes, blocks) = (codes(), [0.5f32, 2.0, 0.25, 1.0]);
        let source = checkpoint(&codes, &[("weight_scale", Dtype::F32, &[2, 2], &blocks)]);
        let (folded, rows) = fold(&source).unwrap();
        for index in 0..codes.len() {
            let row = rows[index / In::SIZE];
            assert_eq!(folded[index].to_f32() * row, stored(&codes, &blocks, index));
        }
    }

    /// Scales parse under either spelling, as FP32 or BF16; two spellings, another shape or
    /// dtype, or a scale not finite and positive are refused, each for its reason, and absent
    /// codes and scales are both missing.
    #[test]
    fn refuses_malformed_block_scales() {
        let codes = codes();
        let grid: &[usize] = &[2, 2];
        let ones = [1.0f32; 4];
        for (scales, refusal) in [
            (
                vec![("weight_scale_inv", Dtype::F32, grid, &ones[..])],
                None,
            ),
            (
                vec![
                    ("weight_scale", Dtype::F32, grid, &ones[..]),
                    ("weight_scale_inv", Dtype::F32, grid, &ones[..]),
                ],
                Some("scale has two spellings"),
            ),
            (
                vec![("weight_scale", Dtype::F32, &[4][..], &ones[..])],
                Some("expected block scales of shape [2, 2]"),
            ),
            (
                vec![("weight_scale", Dtype::F16, grid, &ones[..])],
                Some("expected FP32 or BF16 scales"),
            ),
            (
                vec![("weight_scale", Dtype::F32, grid, &[1.0, 0.0, 1.0, 1.0][..])],
                Some("scale must be finite and positive"),
            ),
            (
                vec![(
                    "weight_scale",
                    Dtype::F32,
                    grid,
                    &[1.0, f32::NAN, 1.0, 1.0][..],
                )],
                Some("scale must be finite and positive"),
            ),
        ] {
            match (fold(&checkpoint(&codes, &scales)), refusal) {
                (Ok(_), None) => {}
                (Err(LoadError::InvalidData { name, reason }), Some(refusal)) => {
                    assert_eq!(
                        (name.as_str(), reason.as_str()),
                        ("proj.weight_scale", refusal)
                    );
                }
                (result, _) => panic!("{scales:?}: {:?}", result.err()),
            }
        }
        match fold(&Checkpoint::default()).err() {
            Some(LoadError::Missing(names)) => {
                assert_eq!(names, ["proj.weight", "proj.weight_scale"])
            }
            other => panic!("{other:?} is not `Missing`"),
        }
    }

    /// A weight in 128 x 128 blocks reads its scales under the checkpoint's spelling and float
    /// type, strictly, and lays them out as the device reads them.
    #[test]
    fn block_form_reads_bf16_inverse_scales() {
        type Weight = BlockScaled<
            In,
            Out,
            f8e4m3,
            f32,
            Scales<In, Out>,
            Block,
            Chip,
            Host<f8e4m3, Chip, m![Out, In]>,
            Host<f32, Chip, Scales<In, Out>>,
        >;
        let blocks = [0.37f32, 1.9, 0.0113, 5.3];
        let source = checkpoint(
            &codes(),
            &[("weight_scale_inv", Dtype::BF16, &[2, 2], &blocks)],
        );
        let mut context = Context::new(None);
        let weight = context
            .field("", "proj", |cx| Weight::load(cx, &source))
            .unwrap();
        let rounded = blocks.map(|scale| half::bf16::from_f32(scale).to_f32());
        assert_eq!(weight.weight_scale.0, layout::<In, Out>(&rounded));
    }

    fn weight(rows: &[[u16; 8]]) -> Finite {
        let values = rows
            .as_flattened()
            .iter()
            .map(|&bits| bf16::from_bits(bits).to_f32());
        Finite::new(values.collect(), 8).unwrap()
    }

    fn bits(scales: &[f32]) -> Vec<u32> {
        scales.iter().map(|scale| scale.to_bits()).collect()
    }

    /// The codes and scales of each granularity equal the frameworks' quantizers', and a value at a
    /// midpoint rounds as the reciprocal puts it: `0.3046875` under the scale of `1.625` is 88.
    #[test]
    fn quantizes_as_the_frameworks_quantizers() {
        let (codes, scales) = Granularity::Tensor.quantize(&weight(&WEIGHT), 0.0);
        assert_eq!(f8e4m3::to_buf(&codes), TENSOR.0.as_flattened());
        assert_eq!(bits(&scales), [TENSOR.1; 4]);
        let (codes, scales) = Granularity::Channel.quantize(&weight(&WEIGHT), 0.0);
        assert_eq!(f8e4m3::to_buf(&codes), CHANNEL.0.as_flattened());
        assert_eq!(bits(&scales), CHANNEL.1);
        let pair = Finite::new(vec![1.625, 0.3046875], 2).unwrap();
        let (codes, _) = Granularity::Tensor.quantize(&pair, 0.0);
        assert_eq!(
            f8e4m3::to_buf(&codes),
            f8e4m3::to_buf(&[448.0, 88.0].map(f8e4m3::from_f32))
        );
    }

    /// A tensor scale covers the weights fused with it; a part of zeros keeps zero codes.
    #[test]
    fn shares_a_tensor_scale_with_fused_weights() {
        let (codes, scales) = Granularity::Tensor.quantize(&weight(&WEIGHT[..1]), 448.0);
        assert_eq!(scales, [1.0]);
        let expected: Vec<f8e4m3> = WEIGHT[0]
            .iter()
            .map(|&bits| f8e4m3::from_f32(bf16::from_bits(bits).to_f32()))
            .collect();
        assert_eq!(f8e4m3::to_buf(&codes), f8e4m3::to_buf(&expected));
        let zeros = Finite::new(vec![0.0; 8], 4).unwrap();
        for granularity in [Granularity::Tensor, Granularity::Channel] {
            let (codes, scales) = granularity.quantize(&zeros, 0.0);
            assert_eq!((f8e4m3::to_buf(&codes), scales), (vec![0; 8], vec![0.0; 2]));
        }
    }

    /// A weight holding a value that is not finite has no scale.
    #[test]
    fn refuses_values_that_are_not_finite() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(Finite::new(vec![1.0, value], 2).is_none(), "{value}");
        }
    }
}
