//! Weight and activation forms: how a tensor's values are stored and the numerical contract that
//! returns them to the values an op computes with.

use core::borrow::Borrow;
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use crate::load::quantization::fp8 as online;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::tensor::memory::Memory;
use crate::tensor::parallel::Chip;

pub mod fp8;
pub mod nvfp4;
pub mod split;

/// Codes of `E` in [output, input] order with one `S` scale for each block of them, the scales
/// stored in mapping `B`, which returns them to the values a linear multiplies under contract `C`;
/// the codes are held as `W` and the scales as `T`, owned or borrowed.
#[derive(DeviceSend)]
pub struct BlockScaled<
    In: AxisName,
    Out: AxisName,
    E: Scalar,
    S: Scalar,
    B: M,
    C,
    Chips: M = Chip,
    W = HbmTensor<E, Chips, m![Out, In]>,
    T = HbmTensor<S, Chips, B>,
> {
    /// Codes in [output, input] order.
    pub weight: W,
    /// The scale of each block of codes.
    pub weight_scale: T,
    /// The operands of the contract.
    pub contract: C,
    pub shape: PhantomData<fn() -> (In, Out, E, S, B, Chips)>,
}

impl<In: AxisName, Out: AxisName, E: Scalar, S: Scalar, B: M, C, Chips: M, W, T>
    BlockScaled<In, Out, E, S, B, C, Chips, W, T>
{
    pub fn codes(&self) -> &HbmTensor<E, Chips, m![Out, In]>
    where
        W: Borrow<HbmTensor<E, Chips, m![Out, In]>>,
    {
        self.weight.borrow()
    }

    pub fn scales(&self) -> &HbmTensor<S, Chips, B>
    where
        T: Borrow<HbmTensor<S, Chips, B>>,
    {
        self.weight_scale.borrow()
    }
}

/// The checkpoint projections `names` under the context's name, read as one weight whose output
/// rows are theirs in order, as a packed gated MLP holds its gate, then its up projection.
#[derive(Clone, Copy)]
pub struct Projections<'a, const N: usize> {
    pub from: &'a Checkpoint,
    pub names: [&'static str; N],
}

/// The output rows of one of the `N` projections a weight of `Out` rows holds in order.
#[derive(Clone, Debug)]
pub struct Part<Out, const N: usize>(PhantomData<Out>);

impl<Out: AxisName, const N: usize> AxisName for Part<Out, N> {
    const NAME: Ident = Ident::new("Part");
    const SIZE: usize = {
        assert!(Out::SIZE.is_multiple_of(N));
        Out::SIZE / N
    };
}

impl<const N: usize> Projections<'_, N> {
    /// What `read` reads of each projection at its name, in order, or every name they lack.
    pub(crate) fn read<T>(
        self,
        context: &mut Context<'_>,
        mut read: impl FnMut(&Checkpoint, &mut Context<'_>) -> Result<T, LoadError>,
    ) -> Result<[T; N], LoadError> {
        let mut children = Children::default();
        let built = (self.names.iter())
            .map(|name| children.ask(context.field(name, name, |cx| read(self.from, cx))))
            .collect::<Result<Vec<_>, _>>()?;
        let built = children.node(built.into_iter().collect::<Option<Vec<T>>>())?;
        Ok(built.try_into().ok().expect("one value per projection"))
    }
}

impl<In: AxisName, Out: AxisName, E: Scalar, S: Scalar, B: M, C, Chips: M, W, T>
    BlockScaled<In, Out, E, S, B, C, Chips, W, T>
{
    fn new(weight: W, weight_scale: T, contract: C) -> Self {
        Self {
            weight,
            weight_scale,
            contract,
            shape: PhantomData,
        }
    }
}

/// E4M3 codes under no contract, one FP32 scale per output row: its codes and scales as stored,
/// under `weight` and `weight_scale`; in a checkpoint a load quantizes, the codes and scales of
/// the BF16 weight stored under `weight`.
impl<'a, In: AxisName, Out: AxisName, Chips: M + 'static, W, T> Load<&'a Checkpoint>
    for BlockScaled<In, Out, f8e4m3, f32, m![Out, 1 # 8], (), Chips, W, T>
where
    W: Load<&'a Checkpoint>
        + online::Unquantized<In, Out, Chips>
        + Load<<W as online::Unquantized<In, Out, Chips>>::Host>,
    T: Load<&'a Checkpoint> + Load<online::Scales<Out, Chips>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| match from.quantization() {
            None => {
                let mut children = Children::default();
                let weight = cx.field("weight", "weight", |cx| {
                    <W as Load<&Checkpoint>>::load(cx, from)
                });
                let weight = children.ask(weight)?;
                let scale = cx.field("weight_scale", "weight_scale", |cx| {
                    <T as Load<&Checkpoint>>::load(cx, from)
                });
                let scale = children.ask(scale)?;
                let (weight, scale) = children.node(weight.zip(scale))?;
                Ok(Self::new(weight, scale, ()))
            }
            Some(_) => {
                let (codes, scales) = Self::quantized(from, cx)?;
                let weight = cx.field("weight", "weight", |cx| W::load(cx, W::host(codes)))?;
                let scale = cx.field("weight_scale", "weight_scale", |cx| T::load(cx, scales))?;
                Ok(Self::new(weight, scale, ()))
            }
        })
    }
}

impl<'t, 's, In: AxisName, Out: AxisName, Chips: M + 'static, W, T>
    Load<(
        &'t mut BlockScaled<In, Out, f8e4m3, f32, m![Out, 1 # 8], (), Chips, W, T>,
        &'s Checkpoint,
    )> for &'t mut BlockScaled<In, Out, f8e4m3, f32, m![Out, 1 # 8], (), Chips, W, T>
where
    W: online::Unquantized<In, Out, Chips>,
    for<'u> &'u mut W: Load<(&'u mut W, &'s Checkpoint)>
        + Load<(&'u mut W, <W as online::Unquantized<In, Out, Chips>>::Host)>,
    for<'u> &'u mut T:
        Load<(&'u mut T, &'s Checkpoint)> + Load<(&'u mut T, online::Scales<Out, Chips>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut BlockScaled<In, Out, f8e4m3, f32, m![Out, 1 # 8], (), Chips, W, T>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| match from.quantization() {
            None => {
                cx.field("weight", "weight", |cx| {
                    <&mut W as Load<(&mut W, &Checkpoint)>>::load(cx, (&mut into.weight, from))
                })?;
                cx.field("weight_scale", "weight_scale", |cx| {
                    <&mut T as Load<(&mut T, &Checkpoint)>>::load(cx, (&mut into.weight_scale, from))
                })
                .map(drop)
            }
            Some(_) => cx
                .rule(
                    |cx| {
                        BlockScaled::<In, Out, f8e4m3, f32, m![Out, 1 # 8], (), Chips, W, T>::quantized(
                            from, cx,
                        )
                    },
                    |cx, (codes, scales)| {
                        cx.field("weight", "weight", |cx| {
                            <&mut W as Load<_>>::load(cx, (&mut into.weight, W::host(codes)))
                        })?;
                        cx.field("weight_scale", "weight_scale", |cx| {
                            <&mut T as Load<_>>::load(cx, (&mut into.weight_scale, scales))
                        })
                        .map(drop)
                    },
                )
                .map(drop),
        })?;
        Ok(into)
    }
}

impl<In: AxisName, Out: AxisName, Chips: M + 'static, W, T>
    BlockScaled<In, Out, f8e4m3, f32, m![Out, 1 # 8], (), Chips, W, T>
{
    /// The codes and row scales the BF16 weight under `weight` quantizes to.
    fn quantized(
        from: &Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<online::Quantized<In, Out, Chips>, LoadError>
    where
        W: online::Unquantized<In, Out, Chips>,
    {
        let weight = context.field("weight", "weight", |cx| W::read(from, cx))?;
        from.channel_scaled::<In, Out, Chips>(context, weight)
    }
}

/// A weight in 128 x 128 blocks: its E4M3 codes as stored under `weight`, its FP32 block scales
/// under `weight_scale` laid out as [`fp8::Scales`].
impl<'a, In: AxisName, Out: AxisName, Chips: M + 'static, W, T> Load<&'a Checkpoint>
    for BlockScaled<In, Out, f8e4m3, f32, fp8::Scales<In, Out>, fp8::Block, Chips, W, T>
where
    W: Load<&'a Checkpoint>,
    T: Load<HostTensor<f32, m![{ Chips }, { fp8::Scales<In, Out> }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let weight = children.ask(cx.field("weight", "weight", |cx| W::load(cx, from)))?;
            let scale = cx.field("weight_scale", "", |cx| {
                let scales = fp8::scales::<In, Out, Chips>(from, cx)?;
                T::load(cx, scales)
            });
            let scale = children.ask(scale)?;
            let (weight, scale) = children.node(weight.zip(scale))?;
            Ok(Self::new(weight, scale, fp8::Block))
        })
    }
}

/// The codes written in place, and the block scales laid out again: two rules of an update.
impl<'t, 's, In: AxisName, Out: AxisName, Chips: M + 'static, W, T>
    Load<(
        &'t mut BlockScaled<In, Out, f8e4m3, f32, fp8::Scales<In, Out>, fp8::Block, Chips, W, T>,
        &'s Checkpoint,
    )> for &'t mut BlockScaled<In, Out, f8e4m3, f32, fp8::Scales<In, Out>, fp8::Block, Chips, W, T>
where
    for<'u> &'u mut W: Load<(&'u mut W, &'s Checkpoint)>,
    for<'u> &'u mut T: Load<(
        &'u mut T,
        HostTensor<f32, m![{ Chips }, { fp8::Scales<In, Out> }]>,
    )>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut BlockScaled<
                In,
                Out,
                f8e4m3,
                f32,
                fp8::Scales<In, Out>,
                fp8::Block,
                Chips,
                W,
                T,
            >,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("weight", "weight", |cx| {
                <&mut W as Load<_>>::load(cx, (&mut into.weight, from))
            })?;
            cx.field("weight_scale", "", |cx| {
                cx.rule(
                    |cx| fp8::scales::<In, Out, Chips>(from, cx),
                    |cx, host| {
                        <&mut T as Load<_>>::load(cx, (&mut into.weight_scale, host)).map(drop)
                    },
                )
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

/// An NVFP4 weight of a ModelOpt checkpoint: its packed E2M1 codes under `weight`, its E4M3 block
/// scales under `weight_scale`, and its tensor scales read by the late contract.
impl<'a, In: AxisName, Out: AxisName, A, X: Memory, W, T> Load<&'a Checkpoint>
    for Nvfp4<In, Out, nvfp4::Late<A, X>, W, T>
where
    W: Load<HostTensor<f4e2m1, m![{ Chip }, { m![Out, In] }]>>,
    T: Load<&'a Checkpoint>,
    nvfp4::Late<A, X>: Load<&'a Checkpoint>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| Self::read(from, cx, "weight"))
    }
}

/// An NVFP4 weight of a compressed-tensors checkpoint: its packed E2M1 codes under
/// `weight_packed`, its E4M3 block scales under `weight_scale`, and its tensor scale read by the
/// early contract.
impl<'a, In: AxisName, Out: AxisName, A, X: Memory, W, T> Load<&'a Checkpoint>
    for Nvfp4<In, Out, nvfp4::Early<A, X>, W, T>
where
    W: Load<HostTensor<f4e2m1, m![{ Chip }, { m![Out, In] }]>>,
    T: Load<&'a Checkpoint>,
    nvfp4::Early<A, X>: Load<&'a Checkpoint>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| Self::read(from, cx, "weight_packed"))
    }
}

/// The codes laid out again, one rule of an update; the block scales and the contract's leaves
/// written in place.
impl<'t, 's, In: AxisName, Out: AxisName, A, X: Memory, W, T>
    Load<(
        &'t mut Nvfp4<In, Out, nvfp4::Late<A, X>, W, T>,
        &'s Checkpoint,
    )> for &'t mut Nvfp4<In, Out, nvfp4::Late<A, X>, W, T>
where
    for<'u> &'u mut W: Load<(&'u mut W, Codes<In, Out>)>,
    for<'u> &'u mut T: Load<(&'u mut T, &'s Checkpoint)>,
    for<'u> &'u mut nvfp4::Late<A, X>: Load<(&'u mut nvfp4::Late<A, X>, &'s Checkpoint)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut Nvfp4<In, Out, nvfp4::Late<A, X>, W, T>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| into.update(from, cx, "weight"))?;
        Ok(into)
    }
}

impl<'t, 's, In: AxisName, Out: AxisName, A, X: Memory, W, T>
    Load<(
        &'t mut Nvfp4<In, Out, nvfp4::Early<A, X>, W, T>,
        &'s Checkpoint,
    )> for &'t mut Nvfp4<In, Out, nvfp4::Early<A, X>, W, T>
where
    for<'u> &'u mut W: Load<(&'u mut W, Codes<In, Out>)>,
    for<'u> &'u mut T: Load<(&'u mut T, &'s Checkpoint)>,
    for<'u> &'u mut nvfp4::Early<A, X>: Load<(&'u mut nvfp4::Early<A, X>, &'s Checkpoint)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut Nvfp4<In, Out, nvfp4::Early<A, X>, W, T>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| into.update(from, cx, "weight_packed"))?;
        Ok(into)
    }
}

/// An NVFP4 weight of a compressed-tensors checkpoint: its packed E2M1 codes under
/// `weight_packed`, its E4M3 block scales under `weight_scale`, and its tensor scale read by the
/// experimental summed contract.
impl<'a, In: AxisName, Out: AxisName, A, X: Memory, W, T> Load<&'a Checkpoint>
    for Nvfp4<In, Out, nvfp4::Summed<A, X>, W, T>
where
    W: Load<HostTensor<f4e2m1, m![{ Chip }, { m![Out, In] }]>>,
    T: Load<&'a Checkpoint>,
    nvfp4::Summed<A, X>: Load<&'a Checkpoint>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| Self::read(from, cx, "weight_packed"))
    }
}

impl<'t, 's, In: AxisName, Out: AxisName, A, X: Memory, W, T>
    Load<(
        &'t mut Nvfp4<In, Out, nvfp4::Summed<A, X>, W, T>,
        &'s Checkpoint,
    )> for &'t mut Nvfp4<In, Out, nvfp4::Summed<A, X>, W, T>
where
    for<'u> &'u mut W: Load<(&'u mut W, Codes<In, Out>)>,
    for<'u> &'u mut T: Load<(&'u mut T, &'s Checkpoint)>,
    for<'u> &'u mut nvfp4::Summed<A, X>: Load<(&'u mut nvfp4::Summed<A, X>, &'s Checkpoint)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut Nvfp4<In, Out, nvfp4::Summed<A, X>, W, T>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| into.update(from, cx, "weight_packed"))?;
        Ok(into)
    }
}

/// An NVFP4 [`BlockScaled`] weight under contract `C`.
type Nvfp4<In, Out, C, W, T> =
    BlockScaled<In, Out, f4e2m1, f8e4m3, nvfp4::Scales<In, Out>, C, Chip, W, T>;

/// The host value of an NVFP4 weight's codes.
type Codes<In, Out> = HostTensor<f4e2m1, m![{ Chip }, { m![Out, In] }]>;

impl<In: AxisName, Out: AxisName, C, W, T> Nvfp4<In, Out, C, W, T> {
    /// The codes under `codes`, the block scales and the contract, absent names reported
    /// together.
    fn read<'a>(
        from: &'a Checkpoint,
        context: &mut Context<'_>,
        codes: &str,
    ) -> Result<Self, LoadError>
    where
        W: Load<Codes<In, Out>>,
        T: Load<&'a Checkpoint>,
        C: Load<&'a Checkpoint>,
    {
        let mut children = Children::default();
        let weight = context.field("weight", codes, |cx| {
            let codes = nvfp4::codes::<Chip, m![Out, In]>(from, cx)?;
            W::load(cx, codes)
        });
        let weight = children.ask(weight)?;
        let scale = context.field("weight_scale", "weight_scale", |cx| T::load(cx, from));
        let scale = children.ask(scale)?;
        let contract = children.ask(context.field("contract", "", |cx| C::load(cx, from)))?;
        let parts = weight.zip(scale).zip(contract);
        children.node(parts).map(|((w, s), c)| Self::new(w, s, c))
    }

    /// The codes under `codes` written once all arrived, then the block scales and the
    /// contract's leaves in place.
    fn update<'s>(
        &mut self,
        from: &'s Checkpoint,
        context: &mut Context<'_>,
        codes: &str,
    ) -> Result<(), LoadError>
    where
        for<'u> &'u mut W: Load<(&'u mut W, Codes<In, Out>)>,
        for<'u> &'u mut T: Load<(&'u mut T, &'s Checkpoint)>,
        for<'u> &'u mut C: Load<(&'u mut C, &'s Checkpoint)>,
    {
        context.field("weight", codes, |cx| {
            cx.rule(
                |cx| nvfp4::codes::<Chip, m![Out, In]>(from, cx),
                |cx, host| <&mut W as Load<_>>::load(cx, (&mut self.weight, host)).map(drop),
            )
        })?;
        context.field("weight_scale", "weight_scale", |cx| {
            <&mut T as Load<_>>::load(cx, (&mut self.weight_scale, from))
        })?;
        context
            .field("contract", "", |cx| {
                <&mut C as Load<_>>::load(cx, (&mut self.contract, from))
            })
            .map(drop)
    }
}

impl<'a, In: AxisName, Out: AxisName, E: Scalar, S: Scalar, B: M, C, Chips: M, W, T, D, U, V>
    Load<&'a BlockScaled<In, Out, E, S, B, D, Chips, U, V>>
    for BlockScaled<In, Out, E, S, B, C, Chips, W, T>
where
    W: Load<&'a U>,
    T: Load<&'a V>,
    C: Load<&'a D>,
{
    fn load(
        context: &mut Context<'_>,
        from: &'a BlockScaled<In, Out, E, S, B, D, Chips, U, V>,
    ) -> Result<Self, LoadError> {
        let weight = context.field("weight", "weight", |cx| W::load(cx, &from.weight))?;
        let scale = context.field("weight_scale", "weight_scale", |cx| {
            T::load(cx, &from.weight_scale)
        })?;
        let contract = context.field("contract", "", |cx| C::load(cx, &from.contract))?;
        Ok(Self::new(weight, scale, contract))
    }
}

impl<'t, In: AxisName, Out: AxisName, E: Scalar, S: Scalar, B: M, C, Chips: M, W, T>
    Load<(
        &'t mut BlockScaled<In, Out, E, S, B, C, Chips, W, T>,
        BlockScaled<In, Out, E, S, B, C, Chips, W, T>,
    )> for &'t mut BlockScaled<In, Out, E, S, B, C, Chips, W, T>
where
    for<'u> &'u mut W: Load<(&'u mut W, W)>,
    for<'u> &'u mut T: Load<(&'u mut T, T)>,
    for<'u> &'u mut C: Load<(&'u mut C, C)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut BlockScaled<In, Out, E, S, B, C, Chips, W, T>,
            BlockScaled<In, Out, E, S, B, C, Chips, W, T>,
        ),
    ) -> Result<Self, LoadError> {
        context.field("weight", "weight", |cx| {
            <&mut W as Load<_>>::load(cx, (&mut into.weight, from.weight))
        })?;
        context.field("weight_scale", "weight_scale", |cx| {
            <&mut T as Load<_>>::load(cx, (&mut into.weight_scale, from.weight_scale))
        })?;
        context.field("contract", "", |cx| {
            <&mut C as Load<_>>::load(cx, (&mut into.contract, from.contract))
        })?;
        Ok(into)
    }
}

/// Retained BF16 parameter of shape `S` on chips `Chips`, held as `T`: the HBM tensor, or a
/// borrow of it in a borrowed tree.
#[derive(DeviceSend, Load)]
pub struct Weight<S: M, Chips: M = m![1], T = HbmTensor<bf16, Chips, S>> {
    #[load(rename = "")]
    tensor: T,
    shape: PhantomData<fn() -> (S, Chips)>,
}

impl<S: M, Chips: M, T> Weight<S, Chips, T> {
    /// Retain the tensor, owned or borrowed, without copying or adding layout constraints.
    pub const fn new(tensor: T) -> Self {
        Self {
            tensor,
            shape: PhantomData,
        }
    }

    /// The retained tensor itself, for a parameter form that holds it directly.
    pub fn into_tensor(self) -> T {
        self.tensor
    }

    /// Borrow the retained BF16 tensor, owned or borrowed by the tree.
    pub fn tensor(&self) -> &HbmTensor<bf16, Chips, S>
    where
        T: Borrow<HbmTensor<bf16, Chips, S>>,
    {
        self.tensor.borrow()
    }
}

/// A parameter whose value is built on host, as a computed one's is, held as its tensor is.
impl<S: M + 'static, Chips: M + 'static, T> Load<HostTensor<bf16, m![{ Chips }, { S }]>>
    for Weight<S, Chips, T>
where
    T: Load<HostTensor<bf16, m![{ Chips }, { S }]>>,
{
    fn load(
        context: &mut Context<'_>,
        host: HostTensor<bf16, m![{ Chips }, { S }]>,
    ) -> Result<Self, LoadError> {
        T::load(context, host).map(|tensor| Self {
            tensor,
            shape: PhantomData,
        })
    }
}

/// A parameter from a function of its host value, held as its tensor is: only a tensor that holds
/// values calls it.
impl<'f, S: M + 'static, Chips: M + 'static, T>
    Load<&'f (dyn Fn() -> Result<HostTensor<bf16, m![{ Chips }, { S }]>, LoadError> + 'f)>
    for Weight<S, Chips, T>
where
    T: Load<&'f (dyn Fn() -> Result<HostTensor<bf16, m![{ Chips }, { S }]>, LoadError> + 'f)>,
{
    fn load(
        context: &mut Context<'_>,
        build: &'f (dyn Fn() -> Result<HostTensor<bf16, m![{ Chips }, { S }]>, LoadError> + 'f),
    ) -> Result<Self, LoadError> {
        T::load(context, build).map(|tensor| Self {
            tensor,
            shape: PhantomData,
        })
    }
}

/// The parameter reads as the tensor it holds, as an op that takes any held form borrows it.
impl<S: M, Chips: M, T: Borrow<HbmTensor<bf16, Chips, S>>> Borrow<HbmTensor<bf16, Chips, S>>
    for Weight<S, Chips, T>
{
    fn borrow(&self) -> &HbmTensor<bf16, Chips, S> {
        self.tensor.borrow()
    }
}

/// Another handle to the same parameter, so a module's copy can outlive the borrow of the tree.
impl<S: M, Chips: M> Clone for Weight<S, Chips> {
    fn clone(&self) -> Self {
        // SAFETY: parameters are read-only; the alias names the same cells in the same order.
        Self::new(unsafe { self.tensor.view().alias() })
    }
}
