//! The preserved per-head `[query, key, value]` checkpoint split and partial-RoPE pairing.

use core::borrow::Borrow;
use core::marker::PhantomData;

use super::parameters::rounded;
use super::{AttentionWidth, Chips, HeadWidth, Heads, Hidden, Major, Normalized, Rotated};
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::modules::attention;
use crate::ops::linear::{Affine, PerHead};
use crate::ops::rotary::Rotary;
use crate::tensor::memory::residency::Residency;
use crate::tensor::memory::{Hbm, Memory};
use crate::{Module, Op};
use furiosa_opt_std::prelude::*;

/// Three projections replicated on `C`, each retained in the memory selected by `X`.
pub type Projection<C = Chips, X = Hbm> = Fused<
    C,
    <X as Memory>::Tensor<bf16, C, WeightLayout>,
    <X as Memory>::Tensor<f32, C, BiasLayout>,
>;

/// Conversion owns no alternate numerical path: the children are public `PerHead` leaves.
/// Each source tensor updates its three retained parts together; weight and bias are independent.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T, B))]
pub struct Fused<C: M, T, B> {
    pub projection: attention::Projection<Part<C, T, B>>,
}

impl<I, C: M, T, B> !Op<I> for Fused<C, T, B> {}

impl<C: M, T, B, E: Copy> Module<&Normalized<C>, E, &mut ()> for Fused<C, T, B>
where
    T: Borrow<HbmTensor<bf16, C, WeightLayout>>,
    B: Borrow<HbmTensor<f32, C, BiasLayout>>,
{
    type Output = (Major<C>, Major<C>, Major<C>);

    fn forward(
        &self,
        device: &mut Device,
        env: E,
        _: &mut (),
        input: &Normalized<C>,
    ) -> Self::Output {
        self.projection
            .forward(device, env, &mut ((), (), ()), input)
    }
}

impl<'a, C: M + 'static, T, B> Load<&'a Checkpoint> for Fused<C, T, B>
where
    T: Load<HostTensor<bf16, m![{ C }, { WeightLayout }]>>,
    B: Load<HostTensor<f32, m![{ C }, { BiasLayout }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let [query, key, value] = images::<C>(from, cx)?;
            let mut load =
                |name: &str, (weight, bias): Image<C>| -> Result<Part<C, T, B>, LoadError> {
                    cx.field("projection", "", |cx| {
                        cx.field(name, "", |cx| {
                            let weight = cx.field("0.weight", "", |cx| T::load(cx, weight))?;
                            let bias = cx.field("0.bias", "", |cx| B::load(cx, bias))?;
                            Ok(PerHead(Affine::new(weight, bias), PhantomData))
                        })
                    })
                };
            let query = load("query", query)?;
            let key = load("key", key)?;
            let value = load("value", value)?;
            Ok(Self {
                projection: attention::Projection { query, key, value },
            })
        })
    }
}

impl<'t, 's, C: M + 'static, T, B> Load<(&'t mut Fused<C, T, B>, &'s Checkpoint)>
    for &'t mut Fused<C, T, B>
where
    for<'u> &'u mut T: Load<(&'u mut T, HostTensor<bf16, m![{ C }, { WeightLayout }]>)>,
    for<'u> &'u mut B: Load<(&'u mut B, HostTensor<f32, m![{ C }, { BiasLayout }]>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Fused<C, T, B>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("weight", "", |cx| {
                cx.rule(
                    |cx| cx.field("source", "weight", |cx| weights::<C>(from, cx)),
                    |cx, images| {
                        let parts = [
                            &mut into.projection.query,
                            &mut into.projection.key,
                            &mut into.projection.value,
                        ];
                        for ((name, part), host) in
                            ["query", "key", "value"].into_iter().zip(parts).zip(images)
                        {
                            cx.field(name, "", |cx| {
                                <&mut T as Load<_>>::load(cx, (&mut part.0.weight, host)).map(drop)
                            })?;
                        }
                        Ok(())
                    },
                )
            })?;
            cx.field("bias", "", |cx| {
                cx.rule(
                    |cx| cx.field("source", "bias", |cx| biases::<C>(from, cx)),
                    |cx, images| {
                        let parts = [
                            &mut into.projection.query,
                            &mut into.projection.key,
                            &mut into.projection.value,
                        ];
                        for ((name, part), host) in
                            ["query", "key", "value"].into_iter().zip(parts).zip(images)
                        {
                            cx.field(name, "", |cx| {
                                <&mut B as Load<_>>::load(cx, (&mut part.0.bias, host)).map(drop)
                            })?;
                        }
                        Ok(())
                    },
                )
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

type Part<C, T, B> = PerHead<Hidden, AttentionWidth, HeadWidth, C, T, B>;
type Image<C> = (
    HostTensor<bf16, m![{ C }, { WeightLayout }]>,
    HostTensor<f32, m![{ C }, { BiasLayout }]>,
);

pub type WeightLayout = m![AttentionWidth, Hidden];
pub type BiasLayout = m![AttentionWidth];

/// Strict source shapes, shared source rounding, then each part's rows in head order.
fn images<C: M>(
    source: &Checkpoint,
    context: &mut Context<'_>,
) -> Result<[Image<C>; 3], LoadError> {
    let mut children = Children::default();
    let weight = children.ask(context.field("weight", "weight", |cx| weights::<C>(source, cx)))?;
    let bias = children.ask(context.field("bias", "bias", |cx| biases::<C>(source, cx)))?;
    let ([qw, kw, vw], [qb, kb, vb]) = children.node(weight.zip(bias))?;
    Ok([(qw, qb), (kw, kb), (vw, vb)])
}

fn weights<C: M>(
    source: &Checkpoint,
    context: &mut Context<'_>,
) -> Result<[HostTensor<bf16, m![{ C }, { WeightLayout }]>; 3], LoadError> {
    let weight = rounded(source, context, &[3 * AttentionWidth::SIZE, Hidden::SIZE])?;
    Ok(core::array::from_fn(|part| {
        let weights: Vec<bf16> = split(&weight, part, Hidden::SIZE)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        HostTensor::from_vec(weights.repeat(C::SIZE))
    }))
}

fn biases<C: M>(
    source: &Checkpoint,
    context: &mut Context<'_>,
) -> Result<[HostTensor<f32, m![{ C }, { BiasLayout }]>; 3], LoadError> {
    let bias = rounded(source, context, &[3 * AttentionWidth::SIZE])?;
    Ok(core::array::from_fn(|part| {
        HostTensor::from_vec(split(&bias, part, 1).repeat(C::SIZE))
    }))
}

/// Source row `(head * 3 + part) * 64 + d` becomes `head * 64 + paired(d)` for Q/K.
/// Pairing places original `(d, d + 24)` across the halves of a 64-wide head; V stays unchanged.
fn split(source: &[f32], part: usize, columns: usize) -> Vec<f32> {
    let mut rows = vec![0.0; AttentionWidth::SIZE * columns];
    for head in 0..Heads::SIZE {
        for d in 0..HeadWidth::SIZE {
            let paired = if part == 2 {
                d
            } else {
                Rotary::<HeadWidth, Rotated>::paired(d)
            };
            let from = ((head * 3 + part) * HeadWidth::SIZE + d) * columns;
            let to = (head * HeadWidth::SIZE + paired) * columns;
            rows[to..][..columns].copy_from_slice(&source[from..][..columns]);
        }
    }
    rows
}
