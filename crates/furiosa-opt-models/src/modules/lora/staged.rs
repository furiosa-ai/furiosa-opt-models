//! Host staging: a checkpoint's terms built on host as HBM lays them out, then uploaded.

use furiosa_opt_std::prelude::*;
use safetensors::tensor::TensorView;

use super::peft::{FACTORS, scaled};
use super::{GateUp, RANK, Term};
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::linear::Affine;
use crate::tensor::memory::Memory;

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

/// A packed gate/up term built on host, written into `into`'s storage.
impl<'t, 'h, T, U> Load<(&'t mut GateUp<T>, &'h GateUp<U>)> for &'t mut GateUp<T>
where
    for<'u> &'u mut T: Load<(&'u mut T, &'h U)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, host): (&'t mut GateUp<T>, &'h GateUp<U>),
    ) -> Result<Self, LoadError> {
        <&mut T as Load<_>>::load(context, (&mut into.0, &host.0))?;
        Ok(into)
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

/// A term built on host, written into `into`'s storage; the host term stays its owner's.
impl<'t, 'h, A, B, HA, HB> Load<(&'t mut Term<A, B>, &'h Term<HA, HB>)> for &'t mut Term<A, B>
where
    for<'u> &'u mut A: Load<(&'u mut A, &'h HA)>,
    for<'u> &'u mut B: Load<(&'u mut B, &'h HB)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, host): (&'t mut Term<A, B>, &'h Term<HA, HB>),
    ) -> Result<Self, LoadError> {
        context.field("lora_a", "", |cx| {
            <&mut A as Load<_>>::load(cx, (&mut into.lora_a.weight, &host.lora_a.weight))
        })?;
        context.field("lora_b", "", |cx| {
            <&mut B as Load<_>>::load(cx, (&mut into.lora_b.weight, &host.lora_b.weight))
        })?;
        Ok(into)
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

#[cfg(test)]
mod tests {
    use std::future::IntoFuture;

    use safetensors::Dtype;
    use serde_json::Value;

    use super::*;
    use crate::load::Named;
    use crate::modules::lora::Rank;
    use crate::tensor::memory::Hbm;
    use crate::tensor::parallel::Chip;

    axes![Inputs = 64, Outputs = 48];

    /// The rank columns of a term adapting one module.
    type One = Rank<1>;

    /// The rank columns of the packed gate/up term.
    type Two = Rank<2>;

    /// One adapted module, `layer.proj`, of 64 inputs and 48 outputs, as PEFT names it.
    type Tree = Named<
        "base_model",
        Named<"layer", Named<"proj", Term<Host<m![One, Inputs]>, Host<m![Outputs, One]>>>>,
    >;

    /// One layer's adapter in memory `X`, as PEFT names it: `layer.proj`, of 64 inputs and 48
    /// outputs, and the packed `layer.gate_proj` and `layer.up_proj`, of 24 outputs each.
    type Adapter<X> = Named<
        "base_model",
        Named<
            "layer",
            (
                Named<"proj", Factors<One, Outputs, X>>,
                GateUp<Factors<Two, Outputs, X>>,
            ),
        >,
    >;

    /// A term's `[K, Inputs]` and `[Out, K]` factors in memory `X`.
    type Factors<K, Out, X> = Term<
        <X as Memory>::Tensor<bf16, Chip, m![K, Inputs]>,
        <X as Memory>::Tensor<bf16, Chip, m![Out, K]>,
    >;

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

    /// An adapter written in place from a borrowed host adapter holds what an upload of that host
    /// adapter holds, in the storage its load allocated: a lower rank's columns and a module it
    /// leaves out read as zeros, and an earlier host adapter writes its own values back.
    #[test]
    fn writes_a_host_adapter_into_a_loaded_one() {
        fn values<S: M + 'static>(
            leaf: &HbmTensor<bf16, Chip, S>,
            device: &mut Device,
        ) -> Vec<bf16> {
            let read = leaf.to_host::<m![{ Chip }, { S }]>(&mut device.pdma);
            futures_executor::block_on(read.into_future())
                .unwrap()
                .into_vec()
        }
        let read = |adapter: &Adapter<Hbm>, device: &mut Device| {
            let Named(Named((Named(proj), GateUp(gate_up)))) = adapter;
            [
                values(&proj.lora_a.weight, device),
                values(&proj.lora_b.weight, device),
                values(&gate_up.lora_a.weight, device),
                values(&gate_up.lora_b.weight, device),
            ]
        };
        let leaves = [
            "proj.lora_a",
            "proj.lora_b",
            "gate_up.lora_a",
            "gate_up.lora_b",
        ];
        let rows = Outputs::SIZE / 2;
        let stage = |rank: usize, alpha: f32, modules: &[&str]| {
            let config = serde_json::json!({"peft_type": "LORA", "r": rank, "lora_alpha": alpha});
            let outputs = |module: &str| match module {
                "proj" => Outputs::SIZE,
                _ => rows,
            };
            let names: Vec<(String, [usize; 2])> = (modules.iter())
                .flat_map(|module| {
                    let name = |factor: &str| format!("base_model.layer.{module}.{factor}");
                    [
                        (name(FACTORS[0]), [rank, Inputs::SIZE]),
                        (name(FACTORS[1]), [outputs(module), rank]),
                    ]
                })
                .collect();
            let tensors: Vec<(&str, Dtype, [usize; 2])> = (names.iter())
                .map(|(name, shape)| (name.as_str(), Dtype::F32, *shape))
                .collect();
            let (_directory, checkpoint) = saved(config, &tensors);
            Adapter::<Staged>::load(&mut Context::new(None), &scaled(&checkpoint).unwrap()).unwrap()
        };
        let first = stage(8, 32.0, &["proj", "gate_proj", "up_proj"]);
        let second = stage(2, 1.0, &["gate_proj"]);

        let _lease = crate::tests::DEVICE.blocking_lock();
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mut loaded = Adapter::<Hbm>::load(&mut Context::new(&mut device), &first).unwrap();
        let uploaded = Adapter::<Hbm>::load(&mut Context::new(&mut device), &second).unwrap();
        let (first_held, second_held) = (read(&loaded, &mut device), read(&uploaded, &mut device));
        let zero = bf16::from_f32(0.0);
        let stored = |at: usize| at as f32 / 64.0;
        let gate_a = (0..Two::SIZE * Inputs::SIZE).map(|at| match at < 2 * Inputs::SIZE {
            true => bf16::from_f32(stored(at)),
            false => zero,
        });
        let gate_b = (0..Outputs::SIZE * Two::SIZE).map(|at| {
            let (row, column) = (at / Two::SIZE, at % Two::SIZE);
            match row < rows && column < 2 {
                true => bf16::from_f32(0.5 * stored(row * 2 + column)),
                false => zero,
            }
        });
        let gate_only = [
            vec![zero; One::SIZE * Inputs::SIZE],
            vec![zero; Outputs::SIZE * One::SIZE],
            gate_a.collect(),
            gate_b.collect(),
        ];
        assert_eq!(
            second_held, gate_only,
            "the second adapter adapts gate_proj alone"
        );
        for (leaf, (a, b)) in leaves.iter().zip(first_held.iter().zip(&second_held)) {
            assert_ne!(a, b, "{leaf}: both adapters hold the same values");
        }

        let free = device.memory().usage().available;
        let writes = [
            (&second, &second_held),
            (&first, &first_held),
            (&second, &second_held),
        ];
        for (from, expected) in writes {
            <&mut Adapter<Hbm> as Load<_>>::load(
                &mut Context::new(&mut device),
                (&mut loaded, from),
            )
            .unwrap();
            let now = read(&loaded, &mut device);
            for (leaf, (held, expected)) in leaves.iter().zip(now.iter().zip(expected)) {
                assert_eq!(held, expected, "{leaf}");
            }
            assert_eq!(
                device.memory().usage().available,
                free,
                "a write allocates no HBM"
            );
        }
    }
}
