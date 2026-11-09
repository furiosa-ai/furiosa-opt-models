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
//! - [`GateUp`] is the update of a feed-forward's packed gate and up projections.
//! - A projection adds each adapter's update to the rows it writes in HBM ([`Adapters::add`]); a
//!   layer's [`Layer`] holds each of its projections' adapters.
//!
//! Numerics: `lora_A x` rounds to BF16, the selection keeps or zeroes it exactly, `lora_B` of it
//! rounds to BF16, and its sum with the projection's BF16 output rounds once.

use std::{collections::BTreeMap, marker::PhantomData};

use furiosa_opt_std::prelude::*;
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use serde_json::Value;

use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::linear::Affine;
use crate::ops::pointwise::{add::Add, multiply::Multiply};
use crate::tensor::parallel::Chip;
use crate::tensor::{live::Live, memory::Memory};

/// The rank a term holds for each module it adapts: an adapter of a lower rank is zero-padded, and
/// a contraction part holds at least 32 inputs (`specialize::contraction`).
pub const RANK: usize = 32;

/// The rank columns of a term adapting `N` modules at once: one, or the gate and up projections a
/// packed projection computes.
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

/// Written out, as a derive would ask the term and the selection to be `Copy` too.
impl<T, S, const N: usize> Clone for Adapters<'_, T, S, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, S, const N: usize> Copy for Adapters<'_, T, S, N> {}

/// The update of a feed-forward's projection computing its gate and up rows at once, packed as
/// the gate's rows then the up's: its rank holds the gate's, then the up's, and each output row
/// reads only its own part's.
#[derive(DeviceSend)]
pub struct GateUp<T>(T);

impl<T> GateUp<T> {
    /// The term, in the packed projection's output order.
    pub fn term(&self) -> &T {
        &self.0
    }
}

/// The adapters each of a layer's projections runs with in a call: its query, key, value and
/// output projections' and its feed-forward's packed gate/up and down projections'.
#[derive(Clone, Copy, DeviceSend)]
pub struct Layer<Q, K, V, O, G, D> {
    pub query: Q,
    pub key: K,
    pub value: V,
    pub output: O,
    pub gate_up: G,
    pub down: D,
}

/// The states of a layer's projections in a call: `()` for every one when the call runs no
/// adapter, else a [`Layer`] of their adapters.
pub trait States: Copy {
    type Query;
    type Key;
    type Value;
    type Output;
    type GateUp;
    type Down;

    fn query(&mut self) -> &mut Self::Query;
    fn key(&mut self) -> &mut Self::Key;
    fn value(&mut self) -> &mut Self::Value;
    fn output(&mut self) -> &mut Self::Output;
    fn gate_up(&mut self) -> &mut Self::GateUp;
    fn down(&mut self) -> &mut Self::Down;
}

/// A call without adapters: every projection's state is `()`.
impl States for () {
    type Query = ();
    type Key = ();
    type Value = ();
    type Output = ();
    type GateUp = ();
    type Down = ();

    fn query(&mut self) -> &mut () {
        self
    }

    fn key(&mut self) -> &mut () {
        self
    }

    fn value(&mut self) -> &mut () {
        self
    }

    fn output(&mut self) -> &mut () {
        self
    }

    fn gate_up(&mut self) -> &mut () {
        self
    }

    fn down(&mut self) -> &mut () {
        self
    }
}

impl<Q: Copy, K: Copy, V: Copy, O: Copy, G: Copy, D: Copy> States for Layer<Q, K, V, O, G, D> {
    type Query = Q;
    type Key = K;
    type Value = V;
    type Output = O;
    type GateUp = G;
    type Down = D;

    fn query(&mut self) -> &mut Q {
        &mut self.query
    }

    fn key(&mut self) -> &mut K {
        &mut self.key
    }

    fn value(&mut self) -> &mut V {
        &mut self.value
    }

    fn output(&mut self) -> &mut O {
        &mut self.output
    }

    fn gate_up(&mut self) -> &mut G {
        &mut self.gate_up
    }

    fn down(&mut self) -> &mut D {
        &mut self.down
    }
}

/// The windows of `Window` rows of `All` rows.
#[derive(Clone, Debug)]
struct Windows<All, Window>(PhantomData<(All, Window)>);

impl<All: AxisName, Window: AxisName> AxisName for Windows<All, Window> {
    const NAME: Ident = Ident::new("Windows");
    const SIZE: usize = All::SIZE / Window::SIZE;
}

impl<'a, A, B, K: AxisName, All: AxisName, const N: usize>
    Adapters<'a, Term<A, B>, HbmTensor<bf16, Chip, m![All, K]>, N>
{
    /// `lines`, a window of rows a projection wrote from launch row `start` on, the projection of
    /// the live rows `live` of `input`, plus each adapter's update of those rows; `start` is a
    /// multiple of the window's rows.
    pub(crate) fn add<'i, Rows, Window, Out, I>(
        &self,
        device: &mut Device,
        live: Live<(), Rows>,
        input: &'i I,
        start: usize,
        mut lines: HbmTensor<bf16, Chip, m![Window, Out]>,
    ) -> HbmTensor<bf16, Chip, m![Window, Out]>
    where
        Rows: AxisName,
        Window: AxisName,
        Out: AxisName,
        Affine<A, ()>: for<'o> crate::Op<
                (
                    Live<(), Rows>,
                    &'i I,
                    &'o mut HbmTensor<bf16, Chip, m![Window, K]>,
                ),
                Output = (),
            >,
        Affine<B, ()>: for<'s, 'o> crate::Op<
                (
                    Live<(), Window>,
                    &'s HbmTensor<bf16, Chip, m![Window, K]>,
                    &'o mut HbmTensor<bf16, Chip, m![Window, Out]>,
                ),
                Output = (),
            >,
    {
        type Spans<All, Window> = Symbol<Windows<All, Window>>;
        let rows = Live::<(), Window>::all().subrange(0, live.rows());
        #[unroll]
        for slot in 0..N {
            let term = self.terms[slot];
            let mut shrunk = HbmTensor::new();
            crate::Op::forward(&term.lora_a, &mut *device, (live, input, &mut shrunk));
            // SAFETY: a window's rows of a selection are contiguous, and no launch writes one.
            let selection: HbmTensor<bf16, Chip, m![Window, K]> = unsafe {
                self.selections[slot]
                    .view()
                    .reshape::<Chip, m![{ Spans<All, Window> }, Window, K]>()
                    .tile::<m![{ Spans<All, Window> }], 1, m![
                        { Spans<All, Window> } = 1 # ({ Spans<All, Window> }),
                        Window,
                        K
                    ]>(start / Window::SIZE)
                    .alias()
            };
            let selected = crate::Op::forward(&Multiply, &mut *device, (&shrunk, &selection));
            let mut update = HbmTensor::new();
            crate::Op::forward(&term.lora_b, &mut *device, (rows, &selected, &mut update));
            lines = crate::Op::forward(&Add, &mut *device, (&lines, &update));
        }
        lines
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

/// The `gate_proj` and `up_proj` modules below the context's name, joined: the gate's `lora_A`
/// rows then the up's, each padded to [`RANK`], and each output row its part's `lora_B` row in the
/// part's rank columns, zero in the other's.
impl<'c, In, Out, K> Load<&'c Checkpoint> for GateUp<Term<Host<m![K, In]>, Host<m![Out, K]>>>
where
    In: AxisName,
    Out: AxisName,
    K: AxisName,
{
    fn load(context: &mut Context<'_>, from: &'c Checkpoint) -> Result<Self, LoadError> {
        const { assert!(K::SIZE == 2 * RANK && Out::SIZE.is_multiple_of(2)) };
        let rows = Out::SIZE / 2;
        context.checkpoint(from, |cx| {
            let mut stored = Vec::with_capacity(2);
            for part in ["gate_proj", "up_proj"] {
                stored.push(cx.field(part, part, |cx| {
                    Stored::read(from, cx, (In::SIZE, rows), RANK)
                })?);
            }
            let zero = bf16::from_f32(0.0);
            let (mut a, mut b) = (
                vec![zero; K::SIZE * In::SIZE],
                vec![zero; Out::SIZE * K::SIZE],
            );
            for (part, stored) in stored.iter().enumerate() {
                let Some(Stored {
                    rank,
                    a: shrink,
                    b: expand,
                }) = stored
                else {
                    continue;
                };
                (a[part * RANK * In::SIZE..].iter_mut().zip(shrink))
                    .for_each(|(into, value)| *into = bf16::from_f32(*value));
                for row in 0..rows {
                    let into = &mut b[(part * rows + row) * K::SIZE + part * RANK..][..*rank];
                    for (into, value) in into.iter_mut().zip(&expand[row * rank..]) {
                        *into = bf16::from_f32(*value);
                    }
                }
            }
            Ok(GateUp(Term::host(a, b)))
        })
    }
}

/// A packed gate/up term built on host, in HBM of the context's device.
impl<'h, T: Load<&'h U>, U> Load<&'h GateUp<U>> for GateUp<T> {
    fn load(context: &mut Context<'_>, host: &'h GateUp<U>) -> Result<Self, LoadError> {
        Ok(GateUp(T::load(context, &host.0)?))
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

    axes![Inputs = 64, Outputs = 48];

    /// The rank columns of a term adapting one module.
    type One = Rank<1>;

    /// One adapted module, `layer.proj`, of 64 inputs and 48 outputs, as PEFT names it.
    type Tree = Named<
        "base_model",
        Named<"layer", Named<"proj", Term<Host<m![One, Inputs]>, Host<m![Outputs, One]>>>>,
    >;

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
