use super::{Chip, HeadSlice, Heads, Single};
use crate::Op;
use crate::load::quantization::fp8::{Codes, Unquantized};
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::ops::linear;
use crate::ops::linear::kernel::Tiling;
use crate::ops::normalization::Spread;
use crate::tensor::form::fp8::Finite;
use crate::tensor::{
    live::Live,
    mapping::{Reshape, Rows as Layout},
    memory::{Memory, dtype},
};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::HostBuffer;
use safetensors::Dtype;
use std::borrow::{Borrow, Cow};
use std::marker::PhantomData;

/// Unpacks head-major QKV rows from the feature slices a projection leaves them in.
#[derive(DeviceSend, Load)]
pub struct Split<Width, KvHeads, Groups, Head>(PhantomData<(Width, KvHeads, Groups, Head)>);

/// Query, key and value weights of scalar `D` joined in memory `X` as `S`, each head's rows in that
/// order, from sibling leaves `p.{q,k,v}_proj.weight`; an update writes it once all three arrived.
/// E4M3 codes a load quantizes are those of the joined BF16 weights.
#[derive(DeviceSend)]
pub struct Packed<KvHeads: AxisName, S: M + 'static, X: Memory, D: MaterializableScalar = bf16>(
    pub X::Tensor<D, m![1], S>,
    PhantomData<KvHeads>,
);

/// The host value a joined weight of `D` loads from: its three parts joined.
pub(crate) struct Joined<D: MaterializableScalar, S: M + 'static>(
    HostTensor<D, m![{ m![1] }, { S }]>,
);

/// The three parts of `E` checked without their values, which joins them when a leaf calls it.
type Join<'a, E, S> = dyn Fn() -> Result<HostTensor<E, m![{ m![1] }, { S }]>, LoadError> + 'a;

impl<KvHeads: AxisName, S: M + 'static, X: Memory, D: MaterializableScalar>
    Packed<KvHeads, S, X, D>
{
    /// The three parts of `E` at the context's position, checked without their values; called, it
    /// joins them block by block: of each part's `KvHeads` equal row blocks, block `b` of every
    /// part in turn, before block `b + 1`.
    fn join<'a, E: MaterializableScalar>(
        source: &'a Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<Box<Join<'a, E, S>>, LoadError> {
        let name = context.name();
        let parent = name.strip_suffix("weight").unwrap_or(&name);
        let names = ["q_proj", "k_proj", "v_proj"].map(|part| format!("{parent}{part}.weight"));
        names.iter().for_each(|name| context.read(name));
        let missing: Vec<String> = names
            .iter()
            .filter(|name| source.header(name).is_none())
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(LoadError::Missing(missing));
        }
        let parts = names.map(|name| {
            let (dtype, shape) = source.header(&name).expect("checked above");
            (name, dtype, shape)
        });
        let held = dtype::<E>().map_err(|error| LoadError::Unsupported {
            name: parts[0].0.clone(),
            reason: error.to_string(),
        })?;
        let width = parts[0].2.last().copied().unwrap_or(0);
        for (name, _, shape) in &parts {
            let invalid = |reason: String| LoadError::InvalidData {
                name: name.clone(),
                reason,
            };
            match shape {
                [rows, columns] if rows % KvHeads::SIZE == 0 && *columns == width => {}
                shape => {
                    return Err(invalid(format!(
                        "shape {shape:?}, the leaf holds {} heads of rows of {width}",
                        KvHeads::SIZE
                    )));
                }
            }
        }
        let rows: usize = parts.iter().map(|(_, _, shape)| shape[0]).sum();
        if rows * width != <S as M>::SIZE {
            return Err(LoadError::InvalidData {
                name: parts[0].0.clone(),
                reason: format!(
                    "{rows} rows of {width} joined, the leaf holds {} elements",
                    <S as M>::SIZE
                ),
            });
        }
        for (name, stored, _) in &parts {
            match held {
                Dtype::BF16 => Checkpoint::admit_bf16(*stored, name)?,
                _ if *stored == held => {}
                _ => {
                    return Err(LoadError::InvalidData {
                        name: name.clone(),
                        reason: format!("stored as {stored:?}, the leaf holds {held:?}"),
                    });
                }
            }
        }
        let names = parts.map(|(name, ..)| name);
        Ok(Box::new(move || {
            let views = names
                .each_ref()
                .map(|name| source.get(name).expect("declared above"));
            let data = (views.iter().zip(&names))
                .map(|(view, name)| match held {
                    Dtype::BF16 => Checkpoint::bf16(view, name),
                    _ => Ok(Cow::Borrowed(view.data())),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let joined = (0..KvHeads::SIZE)
                .flat_map(|block| {
                    data.iter().flat_map(move |data| {
                        let bytes = data.len() / KvHeads::SIZE;
                        &data[block * bytes..][..bytes]
                    })
                })
                .copied()
                .collect();
            let joined = HostTensor::try_from(HostBuffer::from_vec(joined));
            joined.map_err(|error| LoadError::InvalidData {
                name: names[0].clone(),
                reason: error.to_string(),
            })
        }))
    }
}

/// The parts checked as [`Packed::join`] checks them; a leaf that holds values joins them.
impl<'a, KvHeads: AxisName, S: M + 'static, X: Memory, D: MaterializableScalar> Load<&'a Checkpoint>
    for Packed<KvHeads, S, X, D>
where
    X::Tensor<D, m![1], S>: for<'t> Load<&'t Join<'t, D, S>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let join = Self::join::<D>(from, cx)?;
            Ok(Packed(Load::load(cx, &*join)?, PhantomData))
        })
    }
}

impl<'t, 's, KvHeads: AxisName, S: M + 'static, X: Memory, D: MaterializableScalar>
    Load<(&'t mut Packed<KvHeads, S, X, D>, &'s Checkpoint)> for &'t mut Packed<KvHeads, S, X, D>
where
    for<'u> &'u mut X::Tensor<D, m![1], S>: Load<(
        &'u mut X::Tensor<D, m![1], S>,
        HostTensor<D, m![{ m![1] }, { S }]>,
    )>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Packed<KvHeads, S, X, D>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.rule(
                |cx| Packed::<KvHeads, S, X, D>::join::<D>(from, cx)?(),
                |cx, joined| {
                    <&mut X::Tensor<D, m![1], S> as Load<_>>::load(cx, (&mut into.0, joined))
                        .map(drop)
                },
            )
        })?;
        Ok(into)
    }
}

/// The host value of a joined weight, as a load computes it.
impl<KvHeads: AxisName, S: M + 'static, X: Memory, D: MaterializableScalar> Load<Joined<D, S>>
    for Packed<KvHeads, S, X, D>
where
    X::Tensor<D, m![1], S>: Load<HostTensor<D, m![{ m![1] }, { S }]>>,
{
    fn load(context: &mut Context<'_>, from: Joined<D, S>) -> Result<Self, LoadError> {
        Ok(Packed(Load::load(context, from.0)?, PhantomData))
    }
}

impl<'t, KvHeads: AxisName, S: M + 'static, X: Memory, D: MaterializableScalar>
    Load<(&'t mut Packed<KvHeads, S, X, D>, Joined<D, S>)> for &'t mut Packed<KvHeads, S, X, D>
where
    for<'u> &'u mut X::Tensor<D, m![1], S>: Load<(
        &'u mut X::Tensor<D, m![1], S>,
        HostTensor<D, m![{ m![1] }, { S }]>,
    )>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Packed<KvHeads, S, X, D>, Joined<D, S>),
    ) -> Result<Self, LoadError> {
        <&mut X::Tensor<D, m![1], S> as Load<_>>::load(context, (&mut into.0, from.0))?;
        Ok(into)
    }
}

/// Codes of the joined weight quantize the three BF16 parts joined.
impl<KvHeads: AxisName, In: AxisName, Out: AxisName, X: Memory> Unquantized<In, Out, m![1]>
    for Packed<KvHeads, m![Out, In], X, f8e4m3>
{
    type Host = Joined<f8e4m3, m![Out, In]>;

    fn read(from: &Checkpoint, context: &mut Context<'_>) -> Result<Finite, LoadError> {
        let joined = Packed::<KvHeads, m![Out, In], X, bf16>::join::<bf16>(from, context)?()?;
        let values = joined.into_vec().into_iter().map(bf16::to_f32).collect();
        Checkpoint::finite(values, In::SIZE, &context.name())
    }

    fn host(codes: Codes<In, Out, m![1]>) -> Joined<f8e4m3, m![Out, In]> {
        Joined(codes)
    }
}

impl<'a, KvHeads: AxisName, S: M + 'static, X: Memory, Y: Memory, D: MaterializableScalar>
    Load<&'a Packed<KvHeads, S, Y, D>> for Packed<KvHeads, S, X, D>
where
    X::Tensor<D, m![1], S>: Load<&'a Y::Tensor<D, m![1], S>>,
{
    fn load(
        context: &mut Context<'_>,
        from: &'a Packed<KvHeads, S, Y, D>,
    ) -> Result<Self, LoadError> {
        Ok(Packed(Load::load(context, &from.0)?, PhantomData))
    }
}

impl<'t, KvHeads: AxisName, S: M + 'static, X: Memory, D: MaterializableScalar>
    Load<(&'t mut Packed<KvHeads, S, X, D>, Packed<KvHeads, S, X, D>)>
    for &'t mut Packed<KvHeads, S, X, D>
where
    for<'u> &'u mut X::Tensor<D, m![1], S>:
        Load<(&'u mut X::Tensor<D, m![1], S>, X::Tensor<D, m![1], S>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Packed<KvHeads, S, X, D>, Packed<KvHeads, S, X, D>),
    ) -> Result<Self, LoadError> {
        <&mut X::Tensor<D, m![1], S> as Load<_>>::load(context, (&mut into.0, from.0))?;
        Ok(into)
    }
}

/// A projection form reads the joined weight it holds, owned or borrowed.
impl<KvHeads: AxisName, S: M + 'static, X: Memory, D: MaterializableScalar>
    Borrow<HbmTensor<D, m![1], S>> for Packed<KvHeads, S, X, D>
where
    X::Tensor<D, m![1], S>: Borrow<HbmTensor<D, m![1], S>>,
{
    fn borrow(&self) -> &HbmTensor<D, m![1], S> {
        self.0.borrow()
    }
}

/// A packed weight in row-major `[rows, inputs]` order reads as the tensor it holds.
impl<KvHeads: AxisName, S: M + 'static, X: Memory, D: MaterializableScalar> linear::Matrix
    for Packed<KvHeads, S, X, D>
where
    X::Tensor<D, m![1], S>: linear::Matrix,
{
    type In = <X::Tensor<D, m![1], S> as linear::Matrix>::In;
    type Out = <X::Tensor<D, m![1], S> as linear::Matrix>::Out;

    fn tile<T: Tiling<Self::In, Self::Out>>(
        &self,
        device: &mut Device,
        at: usize,
    ) -> DmTensor<bf16, m![1], T::Cluster, T::Slices, m![{ T::Rows }, { T::Inputs }]> {
        self.0.tile::<T>(device, at)
    }
}

impl<Width, KvHeads, Groups, Head> Split<Width, KvHeads, Groups, Head> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Width, KvHeads, Groups, Head> Default for Split<Width, KvHeads, Groups, Head> {
    fn default() -> Self {
        Self::new()
    }
}

/// A launch's query, key or value rows as the split leaves them, its live rows kept.
type Part<Rows, KvHeads, Group, Head> = Live<Heads<Rows, KvHeads, Group, Head>, Rows>;

type Unpacked<Rows, KvHeads, Groups, Head> = (
    Part<Rows, KvHeads, Groups, Head>,
    Part<Rows, KvHeads, Single, Head>,
    Part<Rows, KvHeads, Single, Head>,
);

#[derive(Clone, Debug)]
struct HeadCount<Groups>(PhantomData<Groups>);

impl<Groups: AxisName> AxisName for HeadCount<Groups> {
    const NAME: Ident = Ident::new("QkvRows");
    const SIZE: usize = Groups::SIZE + 2;
}

type Held<Groups> = Symbol<HeadCount<Groups>>;
type Lane<Rows> = m![{ Layout<Rows>::Lanes } # 256];

type Staged<Width, Rows> =
    DmTensor<bf16, Chip, m![1 # 2], Lane<Rows>, m![{ linear::Groups<Rows> }, Width]>;

type HeadMajor<Rows, KvHeads, HeadRows, Head> =
    m![{ linear::Groups<Rows> }, KvHeads, { HeadRows }, Head];

type Tiled<Rows, KvHeads, Groups, Head, Count> =
    HeadMajor<Rows, KvHeads, m![{ Held<Groups> } = ({ Count }) # ({ Held<Groups> })], Head>;

type Selected<Rows, KvHeads, Groups, Head, Count> = DmTensor<
    bf16,
    Chip,
    m![1 # 2],
    HeadSlice<Rows, KvHeads>,
    m![{ linear::Groups<Rows> }, { Held<Groups> } = ({ Count }), Head],
>;

/// The query, key and value heads of every row, in the feature slices a projection leaves them
/// in.
impl<Width: AxisName, KvHeads: AxisName, Groups: AxisName, Head: AxisName, Rows: AxisName>
    Op<Live<Spread<Width, Rows>, Rows>> for Split<Width, KvHeads, Groups, Head>
{
    type Output = Unpacked<Rows, KvHeads, Groups, Head>;

    fn forward(&self, device: &mut Device, input: Live<Spread<Width, Rows>, Rows>) -> Self::Output {
        let mut rows: Staged<Width, Rows> = DmTensor::new();
        input
            .value()
            .view()
            .to_dm_view(&mut device.tdma, rows.view_mut());
        const {
            assert!(Width::SIZE == KvHeads::SIZE * (Groups::SIZE + 2) * Head::SIZE);
        }
        // SAFETY: the fused width is head-major within every row. Relabeling preserves cells.
        let rows = unsafe {
            rows.reshape::<
                Chip,
                m![1 # 2],
                Lane<Rows>,
                HeadMajor<Rows, KvHeads, Held<Groups>, Head>,
            >()
        };
        let mut query: Selected<Rows, KvHeads, Groups, Head, Symbol<Groups>> = DmTensor::new();
        // SAFETY: query starts at offset zero. The omitted K/V rows remain padding, so each
        // head and row retains its original stride and the view stays within the allocation.
        unsafe {
            rows.view().reshape::<
                Chip,
                m![1 # 2],
                Lane<Rows>,
                Tiled<Rows, KvHeads, Groups, Head, Symbol<Groups>>,
            >()
        }
        .to_dm_view(&mut device.tdma, query.view_mut());
        let mut key: Selected<Rows, KvHeads, Groups, Head, m![1]> = DmTensor::new();
        rows.view()
            .tile::<Held<Groups>, 1, Tiled<Rows, KvHeads, Groups, Head, m![1]>>(Groups::SIZE)
            .to_dm_view(&mut device.tdma, key.view_mut());
        let mut value: Selected<Rows, KvHeads, Groups, Head, m![1]> = DmTensor::new();
        rows.view()
            .tile::<Held<Groups>, 1, Tiled<Rows, KvHeads, Groups, Head, m![1]>>(Groups::SIZE + 1)
            .to_dm_view(&mut device.tdma, value.view_mut());
        // SAFETY: each result only renames the selected head rows, preserving their order.
        let (query, key, value) = unsafe { (query.reshape(), key.reshape(), value.reshape()) };
        (input.with(query), input.with(key), input.with(value))
    }
}

/// The rows a projection of one head kind leaves in feature slices, as `Heads` heads of
/// `Groups` query heads each; each row keeps its lane and its heads their order.
impl<Width: AxisName, Heads: AxisName, Groups: AxisName, Head: AxisName, Rows: AxisName>
    Op<Live<Spread<Width, Rows>, Rows>> for Reshape<m![Heads, Groups, Head]>
{
    type Output = Live<super::Heads<Rows, Heads, Groups, Head>, Rows>;

    fn forward(&self, device: &mut Device, input: Live<Spread<Width, Rows>, Rows>) -> Self::Output {
        const { assert!(Width::SIZE == Heads::SIZE * Groups::SIZE * Head::SIZE) };
        let mut rows: Staged<Width, Rows> = DmTensor::new();
        input
            .value()
            .view()
            .to_dm_view(&mut device.tdma, rows.view_mut());
        let mut heads: super::Heads<Rows, Heads, Groups, Head> = DmTensor::new();
        // SAFETY: the features are head-major within every row. Relabeling preserves cells.
        unsafe {
            rows.view().reshape::<
                Chip,
                m![1 # 2],
                Lane<Rows>,
                m![{ linear::Groups<Rows> }, Heads, Groups, Head],
            >()
        }
        .to_dm_view(&mut device.tdma, heads.view_mut());
        input.with(heads)
    }
}

/// The rows a key or value projection leaves in feature slices, as `Heads` heads: the view above
/// with one head in each group.
impl<Width: AxisName, Heads: AxisName, Head: AxisName, Rows: AxisName>
    Op<Live<Spread<Width, Rows>, Rows>> for Reshape<m![Heads, Head]>
{
    type Output = Live<super::Heads<Rows, Heads, Single, Head>, Rows>;

    fn forward(&self, device: &mut Device, input: Live<Spread<Width, Rows>, Rows>) -> Self::Output {
        Reshape::<m![Heads, Single, Head]>::default().forward(device, input)
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;

    use crate::ops::attention::geometry::{ProjectionWidth, QueryGroups};
    use crate::ops::attention::tests::{Head, KvHeads};
    use crate::tensor::{
        axis::{Eight, Single, Sixteen},
        live::Live,
        parallel::Chip,
    };

    use super::Split;
    use crate::Op;
    use crate::ops::normalization::Spread;
    use crate::ops::pointwise::multiply::QueryScale;
    use crate::tensor;
    use crate::tensor::axis;
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;

    type Envelope = axis::Rows<8>;
    type Factors = HbmTensor<f32, Chip, m![Envelope, 1 # 8]>;

    axes![QueryWidth = 4096, KeyWidth = 1024];
    type Four = QueryGroups<4>;

    #[device(chip = 1)]
    fn split_rows<Width: AxisName, Groups: AxisName, Rows: AxisName>(
        device: &mut Device,
        input: &HbmTensor<bf16, Chip, m![Rows, Width]>,
        live: &Value,
        split: &Split<Width, KvHeads, Groups, Head>,
    ) -> (
        HbmTensor<bf16, Chip, m![Rows, KvHeads, Groups, Head]>,
        HbmTensor<bf16, Chip, m![Rows, KvHeads, Head]>,
        HbmTensor<bf16, Chip, m![Rows, KvHeads, Head]>,
    ) {
        let input: Spread<Width, Rows> = input.to_dm(&mut device.tdma);
        let value = input;
        let input = Live::<(), Rows>::read(live).with(value);
        let (query, key, value) = split.forward(&mut *device, input);
        (
            query.value().to_hbm(&mut device.tdma),
            key.value().to_hbm(&mut device.tdma),
            value.value().to_hbm(&mut device.tdma),
        )
    }

    async fn separates<Width: AxisName, Groups: AxisName, Rows: AxisName>(live: usize) {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(split_rows.topology()).unwrap();
        let values = (0..Rows::SIZE * Width::SIZE)
            .map(|index| bf16::from_f32(((index * 37) % 251) as f32))
            .collect::<Vec<_>>();
        let input = HostTensor::<bf16, m![Rows, Width]>::from_vec(values.clone())
            .to_hbm::<Chip, m![Rows, Width]>(&mut device.pdma)
            .await
            .unwrap();
        let count = tensor::live::tests::count(live)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let (query, key, value): (
            HbmTensor<bf16, Chip, m![Rows, KvHeads, Groups, Head]>,
            HbmTensor<bf16, Chip, m![Rows, KvHeads, Head]>,
            HbmTensor<bf16, Chip, m![Rows, KvHeads, Head]>,
        ) = launch(
            split_rows,
            (
                &mut device,
                &input,
                &count,
                &Split::<Width, KvHeads, Groups, Head>::new(),
            ),
        )
        .await
        .unwrap();
        let actual = [
            query
                .to_host::<m![Rows, KvHeads, Groups, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec(),
            key.to_host::<m![Rows, KvHeads, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec(),
            value
                .to_host::<m![Rows, KvHeads, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec(),
        ];
        let groups = Groups::SIZE;
        for (part, (begin, len)) in [(0, groups), (groups, 1), (groups + 1, 1)]
            .into_iter()
            .enumerate()
        {
            let width = KvHeads::SIZE * len * Head::SIZE;
            let expected = values
                .chunks_exact((groups + 2) * Head::SIZE)
                .flat_map(|head| {
                    head[begin * Head::SIZE..(begin + len) * Head::SIZE]
                        .iter()
                        .copied()
                })
                .take(live * width)
                .collect::<Vec<_>>();
            assert_eq!(actual[part][..live * width], expected[..], "part {part}");
        }
    }

    /// Envelope rows' query and key projections as heads, the queries times each row's temperature.
    #[device(chip = 1)]
    fn scaled_heads(
        device: &mut Device,
        (query, key): (
            &HbmTensor<bf16, Chip, m![Envelope, QueryWidth]>,
            &HbmTensor<bf16, Chip, m![Envelope, KeyWidth]>,
        ),
        (live, factors): (&Value, &Factors),
    ) -> (
        HbmTensor<bf16, Chip, m![Envelope, KvHeads, Four, Head]>,
        HbmTensor<bf16, Chip, m![Envelope, KvHeads, Head]>,
    ) {
        use crate::tensor::{live::Live, mapping::Reshape};
        let tiles = Live::<(), Envelope>::read(live);
        let query: Spread<QueryWidth, Envelope> = query.to_dm(&mut device.tdma);
        let key: Spread<KeyWidth, Envelope> = key.to_dm(&mut device.tdma);
        let query =
            Reshape::<m![KvHeads, Four, Head]>::default().forward(&mut *device, tiles.with(query));
        let key = Reshape::<m![KvHeads, Head]>::default().forward(&mut *device, tiles.with(key));
        let (query, key) = QueryScale.forward(&mut *device, ((query, key), factors));
        (
            query.value().to_hbm(&mut device.tdma),
            key.value().to_hbm(&mut device.tdma),
        )
    }

    /// Each live row's projected features read as heads in projection order, and its queries times
    /// its own position's temperature, rounded to BF16 once; the keys keep their values.
    #[tokio::test(flavor = "current_thread")]
    async fn scales_queries_per_row() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(scaled_heads.topology()).unwrap();
        let values = |count: usize, seed: usize| {
            (0..count)
                .map(|at| bf16::from_f32(((at * 37 + seed) % 251) as f32 / 31.0 - 4.0))
                .collect::<Vec<_>>()
        };
        let (queries, keys) = (values(8 * 4096, 3), values(8 * 1024, 7));
        let temperatures = [
            1.0, 1.0703125, 1.109375, 1.0, 1.1328125, 1.0703125, 1.0, 1.15625,
        ];
        let query = HostTensor::<bf16, m![Envelope, QueryWidth]>::from_vec(queries.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let key = HostTensor::<bf16, m![Envelope, KeyWidth]>::from_vec(keys.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let factors: Factors = HostTensor::<f32, m![Envelope, 1 # 8]>::from_vec(
            temperatures.iter().flat_map(|&t: &f32| [t; 8]),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        for live in [8, 3] {
            let count = tensor::live::tests::count(live)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let (scaled, viewed) = launch(
                scaled_heads,
                (&mut device, (&query, &key), (&count, &factors)),
            )
            .await
            .unwrap();
            let scaled = scaled
                .to_host::<m![Envelope, KvHeads, Four, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let viewed = viewed
                .to_host::<m![Envelope, KvHeads, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let bits = |values: &[bf16]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            for row in 0..live {
                let expected: Vec<bf16> = queries[row * 4096..][..4096]
                    .iter()
                    .map(|q| bf16::from_f32(q.to_f32() * temperatures[row]))
                    .collect();
                assert_eq!(
                    bits(&scaled[row * 4096..][..4096]),
                    bits(&expected),
                    "query row {row} of {live}"
                );
                assert_eq!(
                    bits(&viewed[row * 1024..][..1024]),
                    bits(&keys[row * 1024..][..1024]),
                    "key row {row} of {live}"
                );
            }
        }
    }

    /// Two and four query heads per KV head unpack at one, eight and sixteen rows, and the live
    /// rows unpack when a launch's second row group is idle.
    #[tokio::test]
    async fn splits_each_head() {
        separates::<ProjectionWidth<4096>, QueryGroups<2>, Single>(1).await;
        separates::<ProjectionWidth<4096>, QueryGroups<2>, Eight>(8).await;
        separates::<ProjectionWidth<4096>, QueryGroups<2>, Sixteen>(16).await;
        separates::<ProjectionWidth<4096>, QueryGroups<2>, Sixteen>(3).await;
        separates::<ProjectionWidth<6144>, QueryGroups<4>, Single>(1).await;
        separates::<ProjectionWidth<6144>, QueryGroups<4>, Eight>(8).await;
        separates::<ProjectionWidth<6144>, QueryGroups<4>, Sixteen>(16).await;
        separates::<ProjectionWidth<6144>, QueryGroups<4>, Sixteen>(3).await;
    }
}
