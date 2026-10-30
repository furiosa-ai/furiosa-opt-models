//! Low-rank adaptation (LoRA): a projection's output plus the update `B (A x)` of the adapter a
//! row runs with, read from adapters the caller holds.
//!
//! - One adapter's update of one projection is a [`Term`]: PEFT's `lora_A`, `[rank, inputs]`, and
//!   `lora_B`, `[outputs, rank]`, each a linear. [`load`] applies a PEFT checkpoint's configuration
//!   once, the scale into `lora_B`, builds every term on host, the rank zero-padded, then uploads.
//! - In a call that runs adapters, a projection's state is its [`Adapters`]: each adapter's term and
//!   its selection of the call's rows, one on the rows that run with it and zero on every other. The
//!   projection adds each adapter's update of the rows it selects to its own output; a call without
//!   adapters keeps the state `()`.
//! - [`Packed`] is the update of a projection that computes the query, key and value projections
//!   at once, its rows ordered as the base checkpoint's packed weight orders them.
//!
//! Numerics: `lora_A x` rounds to BF16, the selection keeps or zeroes it exactly, `lora_B` of it
//! rounds to BF16, and its sum with the projection's BF16 output rounds once.

use std::{collections::BTreeMap, marker::PhantomData};

use furiosa_opt_std::prelude::*;
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use serde_json::Value;

use crate::Module;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::linear::Affine;
use crate::ops::normalization::Spread;
use crate::ops::pointwise::{add::Add, multiply::Multiply};
use crate::tensor::{live::Live, memory::Memory};

/// The rank a term holds for each module it adapts: an adapter of a lower rank is zero-padded, and
/// a contraction part holds at least 32 inputs (`specialize::contraction`).
pub const RANK: usize = 32;

/// The rank columns of a term adapting `N` modules at once: one, or the query, key and value
/// projections a packed projection computes.
#[derive(Clone, Debug)]
pub struct Rank<const N: usize>;

impl<const N: usize> AxisName for Rank<N> {
    const NAME: Ident = Ident::new("Rank");
    const SIZE: usize = N * RANK;
}

/// One adapter's update of a projection, `lora_B (lora_A x)`: `A` holds `[Rank, In]` and `B`
/// `[Out, Rank]`, the adapter's scale folded in.
#[derive(DeviceSend)]
pub struct Term<A, B> {
    pub lora_a: Affine<A, ()>,
    pub lora_b: Affine<B, ()>,
}

/// The `N` adapters of a call as one projection's state: each adapter's term, and its selection of
/// the call's rows, `[rows, rank]` ones on the rows that run with it and zeros on every other.
#[derive(DeviceSend)]
pub struct Adapters<'a, T, S, const N: usize> {
    pub terms: [&'a T; N],
    pub selections: &'a [S; N],
}

/// The projection's output plus each adapter's update of the rows it selects: the update's
/// `lora_A` and `lora_B` run as the projection does, its selection is a product and its sum an
/// addition.
impl<'s, 'a, I, W, A, B, Out, K, R, Env, const N: usize>
    Module<I, Env, &'s mut Adapters<'a, Term<A, B>, HbmTensor<bf16, m![1], m![R, K]>, N>>
    for Affine<W, ()>
where
    I: Copy,
    Out: AxisName,
    K: AxisName,
    R: AxisName,
    Env: Copy,
    Self: for<'u> Module<I, Env, &'u mut (), Output = Live<Spread<Out, R>, R>>,
    Affine<A, ()>: for<'u> Module<I, Env, &'u mut (), Output = Live<Spread<K, R>, R>>,
    Affine<B, ()>: for<'h, 'u> Module<
            &'h Live<Spread<K, R>, R>,
            Env,
            &'u mut (),
            Output = Live<Spread<Out, R>, R>,
        >,
{
    type Output = Live<Spread<Out, R>, R>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        adapters: &'s mut Adapters<'a, Term<A, B>, HbmTensor<bf16, m![1], m![R, K]>, N>,
        input: I,
    ) -> Self::Output {
        let mut output = self.forward(&mut *device, env, &mut (), input);
        #[unroll]
        for slot in 0..N {
            let term = adapters.terms[slot];
            let shrunk = term.lora_a.forward(&mut *device, env, &mut (), input);
            let selection: Spread<K, R> = adapters.selections[slot].to_dm(&mut device.tdma);
            let selected = Multiply.forward(
                &mut *device,
                env,
                &mut (),
                (&shrunk, &shrunk.with(selection)),
            );
            let update = term.lora_b.forward(&mut *device, env, &mut (), &selected);
            output = Add.forward(&mut *device, env, &mut (), (&output, &update));
        }
        output
    }
}

/// The update of a projection computing `Queries` query rows and two equal parts of key and value
/// rows at once: its rank holds the query's, the key's and the value's in turn, and each output
/// row reads only its own part's, its outputs ordered as the base's packed weight: of each part's
/// `KvHeads` equal row blocks, block `b` of every part in turn, before block `b + 1`.
#[derive(DeviceSend)]
pub struct Packed<KvHeads, Queries, T>(T, PhantomData<fn() -> (KvHeads, Queries)>);

impl<KvHeads, Queries, T> Packed<KvHeads, Queries, T> {
    /// The term, in the packed projection's output order.
    pub fn term(&self) -> &T {
        &self.0
    }
}

/// What a PEFT LoRA checkpoint's `adapter_config.json` declares of its updates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// The rank of every update: `r`.
    pub rank: usize,
    /// The factor of every update: `lora_alpha / r`, or `lora_alpha / sqrt(r)` under rsLoRA.
    pub scale: f32,
}

/// A configuration field that changes the update unless `unset` holds of its value, and what it
/// adds.
struct Form {
    field: &'static str,
    unset: fn(&Value) -> bool,
    adds: &'static str,
}

impl Form {
    /// A field that adds nothing when absent, null, false or empty.
    const fn new(field: &'static str, adds: &'static str) -> Self {
        Self {
            field,
            unset: |value| match value {
                Value::Null | Value::Bool(false) => true,
                Value::Array(values) => values.is_empty(),
                Value::Object(entries) => entries.is_empty(),
                _ => false,
            },
            adds,
        }
    }
}

const FORMS: [Form; 15] = [
    Form::new("use_dora", "DoRA magnitudes"),
    Form {
        field: "bias",
        unset: |value| value.is_null() || value == "none",
        adds: "bias terms",
    },
    Form::new("lora_bias", "bias terms"),
    Form::new("modules_to_save", "replaced modules"),
    Form::new("rank_pattern", "per-module ranks"),
    Form::new("alpha_pattern", "per-module scales"),
    Form::new("fan_in_fan_out", "transposed factors"),
    Form::new("trainable_token_indices", "trained token embeddings"),
    Form::new("target_parameters", "adapted parameters"),
    Form::new("alora_invocation_tokens", "activated LoRA"),
    Form::new("use_qalora", "QA-LoRA pooling"),
    Form::new("layer_replication", "replicated layers"),
    Form::new("use_bdlora", "block-diagonal factors"),
    Form::new("arrow_config", "Arrow routing"),
    Form::new("kasa_config", "KaSA singular values"),
];

/// A configuration is a LoRA one of a rank the terms hold and a scale FP32 holds, none of whose
/// fields changes the update a term computes.
impl TryFrom<&Value> for Config {
    type Error = LoadError;

    fn try_from(config: &Value) -> Result<Self, LoadError> {
        let name = |field: &str| format!("adapter_config.json:{field}");
        let invalid = |field: &str, reason: String| LoadError::InvalidData {
            name: name(field),
            reason,
        };
        if let Some(kind) = config.get("peft_type").filter(|kind| *kind != "LORA") {
            return Err(invalid("peft_type", format!("{kind}, not LORA")));
        }
        if let Some(Form { field, adds, .. }) = FORMS
            .iter()
            .find(|form| (config.get(form.field)).is_some_and(|value| !(form.unset)(value)))
        {
            return Err(LoadError::Unsupported {
                name: name(field),
                reason: format!("it adds {adds}, which no term holds"),
            });
        }
        let rank = (config.get("r").and_then(Value::as_u64))
            .filter(|&rank| rank > 0)
            .ok_or_else(|| invalid("r", "not a positive integer".into()))?
            as usize;
        if rank > RANK {
            return Err(LoadError::Unsupported {
                name: name("r"),
                reason: format!("{rank}, past the rank {RANK} a term holds"),
            });
        }
        // PEFT's default numerator of the scale.
        let alpha = config.get("lora_alpha").map_or(Some(8.0), Value::as_f64);
        let alpha = alpha.ok_or_else(|| invalid("lora_alpha", "not a number".into()))?;
        let rslora = match config.get("use_rslora") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(rslora)) => *rslora,
            Some(other) => return Err(invalid("use_rslora", format!("{other}, not a boolean"))),
        };
        let denominator = match rslora {
            true => (rank as f64).sqrt(),
            false => rank as f64,
        };
        let scale = (alpha / denominator) as f32;
        if !scale.is_finite() {
            return Err(invalid(
                "lora_alpha",
                format!("{alpha} scales the update by {scale} in FP32"),
            ));
        }
        Ok(Self { rank, scale })
    }
}

/// The factors a PEFT checkpoint stores of each module it adapts, below the module's name.
const FACTORS: [&str; 2] = ["lora_A.weight", "lora_B.weight"];

/// One module's factors as a checkpoint stores them, in FP32: `a`, `lora_A` `[rank, inputs]`, and
/// `b`, `lora_B` `[outputs, rank]`.
struct Stored {
    rank: usize,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl Stored {
    /// The factors of the module at the context's name, a projection of `inputs` inputs and
    /// `outputs` outputs, of a rank up to `columns`; none when the checkpoint holds neither, and
    /// refused, naming the tensor, when it lacks one or one has another shape or no float values.
    fn read(
        from: &Checkpoint,
        context: &mut Context<'_>,
        (inputs, outputs): (usize, usize),
        columns: usize,
    ) -> Result<Option<Self>, LoadError> {
        let [a, b] = [("lora_a.weight", FACTORS[0]), ("lora_b.weight", FACTORS[1])]
            .map(|(tree, factor)| context.field(tree, factor, |cx| cx.tensor(from)));
        let ((a_name, a), (b_name, b)) = match (a, b) {
            (Err(LoadError::Missing(_)), Err(LoadError::Missing(_))) => return Ok(None),
            (a, b) => {
                let mut children = Children::default();
                let a = children.ask(a)?;
                let b = children.ask(b)?;
                children.node(a.zip(b))?
            }
        };
        let invalid = |name: String, reason: String| LoadError::InvalidData { name, reason };
        let rank = match a.shape() {
            &[rank, stored] if stored == inputs && (1..=columns).contains(&rank) => rank,
            &[rank, stored] if stored == inputs => {
                let reason = format!("of rank {rank}, not 1 to the {columns} a term holds");
                return Err(invalid(a_name, reason));
            }
            shape => {
                let reason = format!("shape {shape:?}, the module has {inputs} inputs");
                return Err(invalid(a_name, reason));
            }
        };
        if b.shape() != [outputs, rank] {
            let reason = format!(
                "shape {:?}, the module has {outputs} outputs and lora_A rank {rank}",
                b.shape()
            );
            return Err(invalid(b_name, reason));
        }
        let floats = |name: String, view: &TensorView<'_>| {
            Checkpoint::floats(view)
                .ok_or_else(|| invalid(name, format!("stored as {:?}, not a float", view.dtype())))
        };
        Ok(Some(Self {
            rank,
            a: floats(a_name, &a)?,
            b: floats(b_name, &b)?,
        }))
    }
}

/// Host memory, where a load builds an adapter's every term before it uploads any: each leaf laid
/// out as HBM lays it out.
pub struct Staged;

impl Memory for Staged {
    type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> =
        HostTensor<D, m![{ C }, { S }]>;
}

/// A one-chip leaf of BF16 values laid out as `S`, on host.
type Host<S> = HostTensor<bf16, m![{ m![1] }, { S }]>;

impl<In: AxisName, Out: AxisName, K: AxisName> Term<Host<m![K, In]>, Host<m![Out, K]>> {
    /// A term built on host from `lora_A` `[K, In]` and `lora_B` `[Out, K]` values.
    fn host(a: Vec<bf16>, b: Vec<bf16>) -> Self {
        Term {
            lora_a: Affine::new(HostTensor::from_vec(a), ()),
            lora_b: Affine::new(HostTensor::from_vec(b), ()),
        }
    }
}

/// The module at the context's name: its `lora_A` rows padded with zeros to `K` and its `lora_B`
/// rows' columns too, each value rounded once to BF16; zeros when the checkpoint holds neither
/// factor, as an adapter that leaves the module out.
impl<'c, In: AxisName, Out: AxisName, K: AxisName> Load<&'c Checkpoint>
    for Term<Host<m![K, In]>, Host<m![Out, K]>>
{
    fn load(context: &mut Context<'_>, from: &'c Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let zero = bf16::from_f32(0.0);
            let (mut a, mut b) = (
                vec![zero; K::SIZE * In::SIZE],
                vec![zero; Out::SIZE * K::SIZE],
            );
            if let Some(stored) = Stored::read(from, cx, (In::SIZE, Out::SIZE), K::SIZE)? {
                (a.iter_mut().zip(&stored.a))
                    .for_each(|(into, value)| *into = bf16::from_f32(*value));
                for (row, values) in b
                    .chunks_exact_mut(K::SIZE)
                    .zip(stored.b.chunks_exact(stored.rank))
                {
                    (row.iter_mut().zip(values))
                        .for_each(|(into, value)| *into = bf16::from_f32(*value));
                }
            }
            Ok(Self::host(a, b))
        })
    }
}

/// The query, key and value modules below the context's name, `q_proj`, `k_proj` and `v_proj`,
/// joined: each part's `lora_A` rows in turn, each padded to [`RANK`], and each output row its
/// part's `lora_B` row in the part's rank columns, zero in the others.
impl<'c, In, Out, K, KvHeads, Queries> Load<&'c Checkpoint>
    for Packed<KvHeads, Queries, Term<Host<m![K, In]>, Host<m![Out, K]>>>
where
    In: AxisName,
    Out: AxisName,
    K: AxisName,
    KvHeads: AxisName,
    Queries: AxisName,
{
    fn load(context: &mut Context<'_>, from: &'c Checkpoint) -> Result<Self, LoadError> {
        const { assert!(K::SIZE == 3 * RANK && Out::SIZE > Queries::SIZE) };
        let keys = (Out::SIZE - Queries::SIZE) / 2;
        let parts = [
            ("q_proj", Queries::SIZE),
            ("k_proj", keys),
            ("v_proj", keys),
        ];
        context.checkpoint(from, |cx| {
            let mut stored = Vec::with_capacity(parts.len());
            for (part, rows) in parts {
                stored.push(cx.field(part, part, |cx| {
                    Stored::read(from, cx, (In::SIZE, rows), RANK)
                })?);
            }
            let zero = bf16::from_f32(0.0);
            let (mut a, mut b) = (
                vec![zero; K::SIZE * In::SIZE],
                vec![zero; Out::SIZE * K::SIZE],
            );
            let mut packed = 0;
            for block in 0..KvHeads::SIZE {
                for (part, ((_, rows), stored)) in parts.iter().zip(&stored).enumerate() {
                    let block_rows = rows / KvHeads::SIZE;
                    for row in block * block_rows..(block + 1) * block_rows {
                        if let Some(Stored {
                            rank, b: values, ..
                        }) = stored
                        {
                            let into = &mut b[packed * K::SIZE + part * RANK..][..*rank];
                            for (into, value) in into.iter_mut().zip(&values[row * rank..]) {
                                *into = bf16::from_f32(*value);
                            }
                        }
                        packed += 1;
                    }
                }
            }
            assert_eq!(packed, Out::SIZE, "the parts' rows fill the packed outputs");
            for (part, stored) in stored.iter().enumerate() {
                if let Some(Stored { a: values, .. }) = stored {
                    (a[part * RANK * In::SIZE..].iter_mut().zip(values))
                        .for_each(|(into, value)| *into = bf16::from_f32(*value));
                }
            }
            Ok(Packed(Term::host(a, b), PhantomData))
        })
    }
}

/// A term built on host, in HBM of the context's device.
impl<'h, A: Load<HA>, B: Load<HB>, HA: Clone, HB: Clone> Load<&'h Term<HA, HB>> for Term<A, B> {
    fn load(context: &mut Context<'_>, host: &'h Term<HA, HB>) -> Result<Self, LoadError> {
        let a = context.field("lora_a", "", |cx| A::load(cx, host.lora_a.weight.clone()))?;
        let b = context.field("lora_b", "", |cx| B::load(cx, host.lora_b.weight.clone()))?;
        Ok(Term {
            lora_a: Affine::new(a, ()),
            lora_b: Affine::new(b, ()),
        })
    }
}

/// A packed term built on host, in HBM of the context's device.
impl<'h, KvHeads, Queries, T: Load<&'h U>, U> Load<&'h Packed<KvHeads, Queries, U>>
    for Packed<KvHeads, Queries, T>
{
    fn load(
        context: &mut Context<'_>,
        host: &'h Packed<KvHeads, Queries, U>,
    ) -> Result<Self, LoadError> {
        Ok(Packed(T::load(context, &host.0)?, PhantomData))
    }
}

/// The tree `T` of a PEFT LoRA checkpoint's terms in HBM of `device`, built as `H` in [`Staged`]
/// memory first: the checkpoint is read once with its configuration applied, and the strict
/// traversal builds every term on host and refuses a tensor no term reads
/// ([`LoadError::Unexpected`]), all before the first upload.
pub fn load<T, H>(checkpoint: &Checkpoint, device: &mut Device) -> Result<T, LoadError>
where
    H: for<'c> Load<&'c Checkpoint>,
    T: for<'h> Load<&'h H>,
{
    let staged = H::load(&mut Context::new(None), &scaled(checkpoint)?)?;
    T::load(&mut Context::new(device), &staged)
}

/// `checkpoint`, a PEFT LoRA checkpoint, read once with its configuration applied: each module's
/// `lora_A` and its `lora_B` times the scale, in FP32 and of the shapes stored, and every other
/// tensor as stored, which no term reads. A lone factor, one that is no matrix of floats of the
/// configured rank, and a configuration no term follows are refused.
fn scaled(checkpoint: &Checkpoint) -> Result<Checkpoint, LoadError> {
    let Config { rank, scale } = Config::try_from(checkpoint.config())?;
    // The module of a factor's name, and which of the two factors it names.
    let factor = |name: &str| {
        FACTORS.iter().enumerate().find_map(|(at, factor)| {
            let module = name.strip_suffix(factor)?.strip_suffix('.')?;
            Some((module.to_string(), at))
        })
    };
    let mut pairs: BTreeMap<String, [bool; 2]> = BTreeMap::new();
    for (module, at) in checkpoint.names().iter().filter_map(|name| factor(name)) {
        pairs.entry(module).or_default()[at] = true;
    }
    let missing: Vec<String> = (pairs.iter())
        .flat_map(|(module, held)| {
            (FACTORS.iter().zip(held))
                .filter(|(_, held)| !**held)
                .map(move |(factor, _)| format!("{module}.{factor}"))
        })
        .collect();
    if !missing.is_empty() {
        return Err(LoadError::missing(missing));
    }
    let mut scaled = Checkpoint::default();
    for name in checkpoint.names() {
        let view = checkpoint.get(&name).expect("a name the checkpoint lists");
        let (dtype, data) = match factor(&name) {
            None => (view.dtype(), view.data().to_vec()),
            Some((_, at)) => {
                let invalid = |reason: String| LoadError::InvalidData {
                    name: name.clone(),
                    reason,
                };
                // `lora_A` is `[rank, inputs]` and `lora_B` `[outputs, rank]`.
                let shape = view.shape();
                if shape.len() != 2 {
                    return Err(invalid(format!("shape {shape:?}, not a matrix")));
                }
                if shape[at] != rank {
                    let reason = format!(
                        "shape {shape:?}, of rank {}, not the configured {rank}",
                        shape[at]
                    );
                    return Err(invalid(reason));
                }
                let dtype = view.dtype();
                let values = Checkpoint::floats(&view)
                    .ok_or_else(|| invalid(format!("stored as {dtype:?}, not a float")))?;
                // `lora_B` carries the scale.
                let times = if at == 1 { scale } else { 1.0 };
                let bytes = values
                    .into_iter()
                    .flat_map(|value| (times * value).to_le_bytes());
                (Dtype::F32, bytes.collect())
            }
        };
        scaled
            .insert(&name, dtype, view.shape(), data)
            .expect("each name once, its bytes filling its shape");
    }
    Ok(scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::Named;
    use crate::ops::normalization::RowMajor;
    use crate::ops::normalization::tests::random;
    use crate::tensor::axis::Sixteen;
    use crate::tensor::live::tests::count;
    use crate::tensor::scalar::Value as Count;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![Hidden = 1024, Qkv = 4096, Queries = 2048];

    /// Each row's adapter: the first, none or the second.
    const ADAPTERS: [Option<usize>; 3] = [Some(0), None, Some(1)];

    /// Live rows of the sixteen: past the rows of one lane group.
    const LIVE: usize = 13;

    type Rows<W> = HbmTensor<bf16, m![1], RowMajor<W, Sixteen>>;
    type Projection<In, Out> = Affine<HbmTensor<bf16, m![1], m![Out, In]>, ()>;
    type Factors<In, Out, K> =
        Term<HbmTensor<bf16, m![1], m![K, In]>, HbmTensor<bf16, m![1], m![Out, K]>>;
    type Selection<K> = HbmTensor<bf16, m![1], m![Sixteen, K]>;
    type Normalized<W> = Live<DmTensor<bf16, m![1], m![2], m![1 # 256], m![Sixteen, W]>, Sixteen>;

    axes![Inputs = 64, Outputs = 48];

    /// The rank columns of a term adapting one module.
    type One = Rank<1>;

    /// One adapted module, `layer.proj`, of 64 inputs and 48 outputs, as PEFT names it.
    type Tree = Named<
        "base_model",
        Named<"layer", Named<"proj", Term<Host<m![One, Inputs]>, Host<m![Outputs, One]>>>>,
    >;

    /// The rows of `input`, all features a slice, as a projection reads normalized rows.
    fn rows<W: AxisName>(device: &mut Device, live: &Count, input: &Rows<W>) -> Normalized<W> {
        Live::<(), Sixteen>::read(live).with(input.to_dm(&mut device.tdma))
    }

    #[device(chip = 1)]
    fn plain<In: AxisName, Out: AxisName>(
        device: &mut Device,
        live: &Count,
        projection: &Projection<In, Out>,
        input: &Rows<In>,
    ) -> Rows<Out> {
        let rows = rows(device, live, input);
        let output = projection.forward(&mut *device, (), &mut (), &rows);
        output.value().to_hbm(&mut device.tdma)
    }

    #[device(chip = 1)]
    fn adapted<In: AxisName, Out: AxisName, K: AxisName>(
        device: &mut Device,
        live: &Count,
        projection: &Projection<In, Out>,
        terms: [&Factors<In, Out, K>; 2],
        selections: &[Selection<K>; 2],
        input: &Rows<In>,
    ) -> Rows<Out> {
        let rows = rows(device, live, input);
        let mut adapters = Adapters { terms, selections };
        let output = projection.forward(&mut *device, (), &mut adapters, &rows);
        output.value().to_hbm(&mut device.tdma)
    }

    /// `values` in HBM, laid out as `S`.
    async fn upload<S: M + 'static>(
        device: &mut Device,
        values: Vec<bf16>,
    ) -> HbmTensor<bf16, m![1], S> {
        HostTensor::<bf16, S>::from_vec(values)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap()
    }

    /// `rows x [outputs, inputs]` transposed in FP32.
    fn product(rows: &[f32], weight: &[f32], inputs: usize) -> Vec<f32> {
        rows.chunks_exact(inputs)
            .flat_map(|row| {
                weight
                    .chunks_exact(inputs)
                    .map(move |weight| row.iter().zip(weight).map(|(x, w)| x * w).sum())
            })
            .collect()
    }

    fn widened(values: &[bf16]) -> Vec<f32> {
        values.iter().map(|value| value.to_f32()).collect()
    }

    fn rounded(values: Vec<f32>) -> Vec<f32> {
        values
            .into_iter()
            .map(|value| bf16::from_f32(value).to_f32())
            .collect()
    }

    /// An adapter's factors of a projection: an eighth of pseudo-random values, the rank columns
    /// of `lora_B` past `rank` zero.
    fn factors(seed: u64, inputs: usize, outputs: usize, rank: usize) -> [Vec<bf16>; 2] {
        let columns = outputs * rank;
        let b = (random(columns, seed + 1).into_iter().enumerate()).map(|(at, value)| {
            bf16::from_f32(f32::from(u8::from(at % rank < rank / 2 + 1)) * value.to_f32() / 8.0)
        });
        let a = random(rank * inputs, seed)
            .into_iter()
            .map(|value| bf16::from_f32(value.to_f32() / 8.0));
        [a.collect(), b.collect()]
    }

    /// Mixed rows of no adapter and of each of two adapters through an adapted projection: every
    /// live row is the host's product plus its adapter's update, with BF16 roundings of `lora_A x`,
    /// of the update and of the sum; a row without an adapter is the plain projection's, bit for
    /// bit.
    async fn adds_each_rows_update<In: AxisName, Out: AxisName, K: AxisName>() {
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        // Weights of the factors' scale, so an update outweighs the tolerance of their sum.
        let weight: Vec<bf16> = (random(Out::SIZE * In::SIZE, 0x9e37).into_iter())
            .map(|value| bf16::from_f32(value.to_f32() / 8.0))
            .collect();
        let input = random(Sixteen::SIZE * In::SIZE, 0x2545);
        let projection: Projection<In, Out> =
            Affine::new(upload(&mut device, weight.clone()).await, ());
        let rows: Rows<In> = upload(&mut device, input.clone()).await;
        let live = count(LIVE).to_hbm(&mut device.pdma).await.unwrap();
        let (mut terms, mut host) = (Vec::new(), Vec::new());
        for seed in [11, 12] {
            let [a, b] = factors(seed, In::SIZE, Out::SIZE, K::SIZE);
            let term: Factors<In, Out, K> = Term {
                lora_a: Affine::new(upload(&mut device, a.clone()).await, ()),
                lora_b: Affine::new(upload(&mut device, b.clone()).await, ()),
            };
            terms.push(term);
            host.push((widened(&a), widened(&b)));
        }
        let adapter = |row: usize| ADAPTERS[row % ADAPTERS.len()].filter(|_| row < LIVE);
        let mut selections = Vec::new();
        for slot in 0..2 {
            let ones = (0..Sixteen::SIZE).flat_map(|row| {
                let one = bf16::from_f32(f32::from(u8::from(adapter(row) == Some(slot))));
                std::iter::repeat_n(one, K::SIZE)
            });
            let selection: Selection<K> = upload(&mut device, ones.collect()).await;
            selections.push(selection);
        }
        let selections: [Selection<K>; 2] = selections.try_into().ok().unwrap();
        let base = launch(plain, (&mut device, &live, &projection, &rows))
            .await
            .unwrap();
        let base = base
            .to_host::<RowMajor<Out, Sixteen>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let terms = [&terms[0], &terms[1]];
        let got = launch(
            adapted,
            (&mut device, &live, &projection, terms, &selections, &rows),
        )
        .await
        .unwrap();
        let got = got
            .to_host::<RowMajor<Out, Sixteen>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let x = widened(&input);
        let projected = rounded(product(&x, &widened(&weight), In::SIZE));
        let updates: Vec<Vec<f32>> = host
            .iter()
            .map(|(a, b)| rounded(product(&rounded(product(&x, a, In::SIZE)), b, K::SIZE)))
            .collect();
        for row in 0..LIVE {
            let Some(slot) = adapter(row) else {
                let range = row * Out::SIZE..(row + 1) * Out::SIZE;
                assert_eq!(got[range.clone()], base[range], "row {row} has no adapter");
                continue;
            };
            for column in 0..Out::SIZE {
                let at = row * Out::SIZE + column;
                let expected = projected[at] + updates[slot][at];
                let actual = got[at].to_f32();
                assert!(
                    (actual - expected).abs() <= 0.02 * expected.abs().max(1.0),
                    "row {row} column {column}: {actual}, expected {expected}"
                );
            }
        }
    }

    /// A packed query, key and value projection and an output projection, each updated by two
    /// adapters over rows of neither, one and the other.
    #[tokio::test(flavor = "current_thread")]
    async fn adds_each_adapters_update() {
        let _lease = DEVICE.lock().await;
        adds_each_rows_update::<Hidden, Qkv, Rank<3>>().await;
        adds_each_rows_update::<Queries, Hidden, Rank<1>>().await;
    }

    /// A configuration as PEFT writes it reads as its rank and its scale under each rule; each
    /// field that changes the update, a rank past the terms', a scale past FP32 and a malformed
    /// value, such as an rsLoRA flag that is no boolean, are refused, naming the field.
    #[test]
    fn reads_peft_configurations() {
        let config = |rslora: bool| {
            serde_json::json!({"peft_type": "LORA", "r": 16, "lora_alpha": 32, "use_rslora": rslora,
                "bias": "none", "use_dora": false, "rank_pattern": {}, "modules_to_save": null})
        };
        let read = |value: &Value| Config::try_from(value);
        assert_eq!(
            read(&config(false)).unwrap(),
            Config {
                rank: 16,
                scale: 2.0
            }
        );
        assert_eq!(
            read(&config(true)).unwrap(),
            Config {
                rank: 16,
                scale: 8.0
            }
        );
        let with = |field: &str, value: Value| {
            let mut config = config(false);
            config[field] = value;
            config
        };
        let refused = [
            (with("use_dora", true.into()), "use_dora", true),
            (with("bias", "all".into()), "bias", true),
            (
                with("modules_to_save", serde_json::json!(["lm_head"])),
                "modules_to_save",
                true,
            ),
            (
                with("rank_pattern", serde_json::json!({"q_proj": 8})),
                "rank_pattern",
                true,
            ),
            (with("r", 64.into()), "r", true),
            (with("r", 0.into()), "r", false),
            (with("peft_type", "IA3".into()), "peft_type", false),
            (with("lora_alpha", "x".into()), "lora_alpha", false),
            (
                with("lora_alpha", serde_json::json!(1e40)),
                "lora_alpha",
                false,
            ),
            (with("use_rslora", "true".into()), "use_rslora", false),
            (with("use_rslora", 1.into()), "use_rslora", false),
        ];
        for (config, field, unsupported) in refused {
            let field = format!("adapter_config.json:{field}");
            match (read(&config), unsupported) {
                (Err(LoadError::Unsupported { name, .. }), true)
                | (Err(LoadError::InvalidData { name, .. }), false) => assert_eq!(name, field),
                (other, _) => panic!("{field}: {other:?}"),
            }
        }
    }

    /// A checkpoint of `config` and of F32 `tensors` in a folder of its own.
    fn saved(
        config: Value,
        tensors: &[(&str, Dtype, [usize; 2])],
    ) -> (tempfile::TempDir, Checkpoint) {
        let directory = tempfile::tempdir().unwrap();
        let data: Vec<Vec<u8>> = (tensors.iter())
            .map(|(_, dtype, [rows, columns])| {
                let values = (0..rows * columns).map(|at| at as f32 / 64.0);
                match dtype {
                    Dtype::F32 => values.flat_map(f32::to_le_bytes).collect(),
                    _ => values
                        .flat_map(|value| (value as i32).to_le_bytes())
                        .collect(),
                }
            })
            .collect();
        let views = (tensors.iter().zip(&data)).map(|((name, dtype, shape), data)| {
            (
                *name,
                TensorView::new(*dtype, shape.to_vec(), data).unwrap(),
            )
        });
        let file = directory.path().join("adapter_model.safetensors");
        safetensors::serialize_to_file(views.collect::<Vec<_>>(), &None, &file).unwrap();
        let checkpoint = Checkpoint::new(directory.path(), Some(config)).unwrap();
        (directory, checkpoint)
    }

    /// Without a device, a checkpoint's terms are built on host through the strict traversal:
    /// `lora_A` zero-padded to the term's rank and `lora_B` times the scale; a tensor no term takes,
    /// a factor without its pair, one of another rank, of another shape than its module's or of no
    /// float dtype are refused, naming the tensor, so no load uploads a factor of a checkpoint it
    /// then refuses.
    #[test]
    fn refuses_unfit_tensors_before_any_upload() {
        let config = serde_json::json!({"peft_type": "LORA", "r": 4, "lora_alpha": 8});
        let (a, b) = (
            "base_model.layer.proj.lora_A.weight",
            "base_model.layer.proj.lora_B.weight",
        );
        let factors = |a_shape: [usize; 2], dtype: Dtype| {
            vec![(a, Dtype::F32, a_shape), (b, dtype, [Outputs::SIZE, 4])]
        };
        let stage = |tensors: &[(&str, Dtype, [usize; 2])]| {
            let (_directory, checkpoint) = saved(config.clone(), tensors);
            Tree::load(&mut Context::new(None), &scaled(&checkpoint)?)
        };
        let refusal = |tensors: &[(&str, Dtype, [usize; 2])]| stage(tensors).map(drop).err();
        let Named(Named(Named(term))) = stage(&factors([4, Inputs::SIZE], Dtype::F32)).unwrap();
        let (staged_a, staged_b) = (term.lora_a.weight.into_vec(), term.lora_b.weight.into_vec());
        let stored = |at: usize| at as f32 / 64.0;
        for (at, value) in staged_a.iter().enumerate() {
            let expected = if at < 4 * Inputs::SIZE {
                stored(at)
            } else {
                0.0
            };
            assert_eq!(
                value.to_f32(),
                bf16::from_f32(expected).to_f32(),
                "lora_A {at}"
            );
        }
        for (at, value) in staged_b.iter().enumerate() {
            let (row, column) = (at / RANK, at % RANK);
            let expected = if column < 4 {
                2.0 * stored(row * 4 + column)
            } else {
                0.0
            };
            assert_eq!(
                value.to_f32(),
                bf16::from_f32(expected).to_f32(),
                "lora_B {at}"
            );
        }
        let head = [
            "base_model.lm_head.lora_A.weight",
            "base_model.lm_head.lora_B.weight",
        ];
        let mut unexpected = factors([4, Inputs::SIZE], Dtype::F32);
        unexpected.extend([(head[0], Dtype::F32, [4, 8]), (head[1], Dtype::F32, [8, 4])]);
        let magnitude = "base_model.layer.proj.lora_magnitude_vector";
        let mut dora = factors([4, Inputs::SIZE], Dtype::F32);
        dora.push((magnitude, Dtype::F32, [1, Outputs::SIZE]));
        match (refusal(&unexpected), refusal(&dora)) {
            (Some(LoadError::Unexpected(head_names)), Some(LoadError::Unexpected(names))) => {
                assert_eq!(head_names, head);
                assert_eq!(names, [magnitude]);
            }
            other => panic!("tensors no term takes: {other:?}"),
        }
        match refusal(&factors([4, Inputs::SIZE], Dtype::F32)[..1]) {
            Some(LoadError::Missing(names)) => assert_eq!(names, [b]),
            other => panic!("a factor without its pair: {other:?}"),
        }
        let refused = [
            (factors([5, Inputs::SIZE], Dtype::F32), a),
            (factors([4, 32], Dtype::F32), a),
            (factors([4, Inputs::SIZE], Dtype::I32), b),
        ];
        for (tensors, tensor) in refused {
            match refusal(&tensors) {
                Some(LoadError::InvalidData { name, .. }) => assert_eq!(name, tensor),
                other => panic!("{tensor}: {other:?}"),
            }
        }
    }
}
