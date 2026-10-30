//! Heads that read a body's output: readouts, poolers, and retrieval, sparse and masked heads.

pub use readout::Selected;

use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::embedding::{Embedding, Table};
use crate::ops::index::gather::{self, Rows};
use crate::ops::index::scatter::Store;
use crate::ops::linear::affine::{self, COLUMNS, Weights};
use crate::ops::linear::stream::{Columns, ToDm};
use crate::ops::linear::{Affine, Bias, Biased};
use crate::ops::normalization::Spread;
use crate::ops::reduction::{self, Argmax};
use crate::tensor::{axis::Tile, parallel::Chip};
use crate::{Module, Then};
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;
use std::marker::PhantomData;

mod readout;

/// Applies a head to borrowed body output and returns only the head output; the body and the head
/// read under the names their family gives them.
pub type Readout<Body, Head> = Then<Body, Head>;

impl<Body, Head> Readout<Body, Head> {
    /// The body whose output the head reads, so a family names it through its catalog binding.
    pub type Body = Body;

    /// The head that reads the body's output, so a family names it through its catalog binding.
    pub type Head = Head;

    pub const fn body(&self) -> &Body {
        &self.0
    }

    pub const fn head(&self) -> &Head {
        &self.1
    }
}

/// Returns both body features and the head output computed from those features.
#[derive(Clone, DeviceSend)]
pub struct FeatureHead<Body, Head> {
    pub body: Body,
    pub head: Head,
}

/// The state holds the body's slot, then the head's.
impl<'s, I, X, O, BodyState, HeadState, Body, Head, Env: Copy>
    Module<I, Env, &'s mut (BodyState, HeadState)> for FeatureHead<Body, Head>
where
    Body: Module<I, Env, &'s mut BodyState, Output = X>,
    Head: for<'x> Module<&'x X, Env, &'s mut HeadState, Output = O>,
{
    type Output = (X, O);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (body, head): &'s mut (BodyState, HeadState),
        input: I,
    ) -> Self::Output {
        let hidden = self.body.forward(&mut *device, env, body, input);
        let output = self.head.forward(device, env, head, &hidden);
        (hidden, output)
    }
}

/// Runs a body, then passes its hidden output and the caller's pooling operand to a head.
#[derive(Clone, DeviceSend)]
pub struct Pooler<Body, Head> {
    pub body: Body,
    pub head: Head,
}

/// The pooling operand, such as the rows to pool, is the input's beside the body's; the state holds
/// the body's slot, then the head's.
impl<'s, I, C, X, O, BodyState, HeadState, Body, Head, Env: Copy>
    Module<(I, C), Env, &'s mut (BodyState, HeadState)> for Pooler<Body, Head>
where
    Body: Module<I, Env, &'s mut BodyState, Output = X>,
    Head: for<'b> Module<(&'b X, C), Env, &'s mut HeadState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (body, head): &'s mut (BodyState, HeadState),
        (input, operand): (I, C),
    ) -> Self::Output {
        let hidden = self.body.forward(&mut *device, env, body, input);
        self.head.forward(device, env, head, (&hidden, operand))
    }
}

/// A borrowed input and pooling operand.
impl<'s, I, C, X, O, BodyState, HeadState, Body, Head, Env: Copy>
    Module<&(I, C), Env, &'s mut (BodyState, HeadState)> for Pooler<Body, Head>
where
    Body: for<'x> Module<&'x I, Env, &'s mut BodyState, Output = X>,
    Head: for<'a, 'b> Module<(&'a X, &'b C), Env, &'s mut HeadState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (body, head): &'s mut (BodyState, HeadState),
        (input, operand): &(I, C),
    ) -> O {
        let hidden = self.body.forward(&mut *device, env, body, input);
        self.head.forward(device, env, head, (&hidden, operand))
    }
}

/// An exclusively borrowed input, such as one with caller-owned captures, and pooling operand.
impl<'s, I, C, X, O, BodyState, HeadState, Body, Head, Env: Copy>
    Module<&mut (I, C), Env, &'s mut (BodyState, HeadState)> for Pooler<Body, Head>
where
    Body: for<'i> Module<&'i mut I, Env, &'s mut BodyState, Output = X>,
    Head: for<'a, 'b> Module<(&'a X, &'b C), Env, &'s mut HeadState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (body, head): &'s mut (BodyState, HeadState),
        (input, operand): &mut (I, C),
    ) -> O {
        let hidden = self.body.forward(&mut *device, env, body, input);
        self.head.forward(device, env, head, (&hidden, &*operand))
    }
}

/// Pools hidden rows using the supplied operand, then computes sequence scores.
pub type SequenceHead<Pool, Score> = Then<Pool, Score>;

/// Produces dense, token and sparse retrieval outputs, each with its own operand.
#[derive(Clone, DeviceSend)]
pub struct RetrievalHead<Dense, Tokens, Sparse> {
    pub dense: Dense,
    pub tokens: Tokens,
    pub sparse: Sparse,
}

/// The state holds the dense head's slot, the token head's and the sparse head's.
impl<
    's,
    'x,
    X,
    C,
    K,
    S,
    Y,
    Z,
    O,
    DenseState,
    TokensState,
    SparseState,
    Dense,
    Tokens,
    Sparse,
    Env: Copy,
> Module<(&X, &'x (C, K, S)), Env, &'s mut (DenseState, TokensState, SparseState)>
    for RetrievalHead<Dense, Tokens, Sparse>
where
    Dense: for<'b> Module<(&'b X, &'x C), Env, &'s mut DenseState, Output = Y>,
    Tokens: for<'b> Module<(&'b X, &'x K), Env, &'s mut TokensState, Output = Z>,
    Sparse: for<'b> Module<(&'b X, &'x S), Env, &'s mut SparseState, Output = O>,
{
    type Output = (Y, Z, O);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (dense, tokens, sparse): &'s mut (DenseState, TokensState, SparseState),
        (hidden, operands): (&X, &'x (C, K, S)),
    ) -> Self::Output {
        let dense = self
            .dense
            .forward(&mut *device, env, dense, (hidden, &operands.0));
        let tokens = self
            .tokens
            .forward(&mut *device, env, tokens, (hidden, &operands.1));
        let sparse = self
            .sparse
            .forward(device, env, sparse, (hidden, &operands.2));
        (dense, tokens, sparse)
    }
}

/// Projects hidden features, applies an activation and normalization, then computes logits.
pub type MaskedHead<Projection, Activation, Norm, Head> =
    Then<Projection, Then<Activation, Then<Norm, Head>>>;

/// Projects hidden features to token logits and pools them with the supplied sparse operand.
#[derive(DeviceSend)]
pub struct SparseHead<Projection, Pool> {
    pub projection: Projection,
    pub pool: Pool,
}

/// The state is the projection's slot.
impl<X, C, Y, O, S, Projection, Pool, Env: Copy> Module<(&X, C), Env, S>
    for SparseHead<Projection, Pool>
where
    Projection: for<'x> Module<&'x X, Env, S, Output = Y>,
    Pool: for<'b, 'd> Module<(&'b Y, C), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, state: S, (hidden, operand): (&X, C)) -> O {
        let logits = self.projection.forward(&mut *device, env, state, hidden);
        self.pool.forward(device, env, &mut (), (&logits, operand))
    }
}

/// Selects hidden rows with a stored offset operand.
#[derive(DeviceSend)]
pub struct SelectRow<Selection, Offset> {
    pub select: Selection,
    pub offset: Offset,
}

/// A composition, not an op.
impl<X, Selection, Offset> !crate::Op<&X> for SelectRow<Selection, Offset> {}

impl<X, Y, Selection, Offset, Env: Copy> Module<&X, Env, &mut ()> for SelectRow<Selection, Offset>
where
    Selection: for<'a, 'b, 's> Module<(&'a X, &'b Offset), Env, &'s mut (), Output = Y>,
{
    type Output = Y;

    fn forward(&self, device: &mut Device, env: Env, _: &mut (), input: &X) -> Y {
        self.select
            .forward(device, env, &mut (), (input, &self.offset))
    }
}

/// Each row's first maximum among a padded vocabulary's tokens: the head's logits reach the argmax
/// pass by pass, as candidates, so no row's logits are stored whole.
#[derive(DeviceSend)]
pub struct Greedy<H: AxisName, V: AxisName, Head = Biased<H, V>, Select = Argmax<V>> {
    /// The projection onto the padded vocabulary; its bias is zero for the checkpoint's tokens and
    /// the lowest finite score for the padded rows.
    pub head: Head,
    /// The first maximum of each row's logits.
    pub argmax: Select,
    shape: PhantomData<(H, V)>,
}

impl<H: AxisName, V: AxisName, Head, Select> Greedy<H, V, Head, Select> {
    pub const fn new(head: Head, argmax: Select) -> Self {
        Self {
            head,
            argmax,
            shape: PhantomData,
        }
    }
}

/// A greedy head with its bias and the argmax's indices in data memory, where each call's passes
/// read them.
pub(crate) type InDm<'a, H, V> = Greedy<
    H,
    V,
    Affine<&'a HbmTensor<bf16, Chip, m![V, H]>, Columns<H, V, f32>>,
    Argmax<V, Chip, Columns<H, V, i32>>,
>;

/// A greedy head whose projection's weight, bias and argmax ranks are held as `W`, `B` and `I`:
/// HBM tensors, or borrows of them in a borrowed tree.
type Tree<H, V, W, B, I> = Greedy<H, V, Affine<W, B>, Argmax<V, Chip, I>>;

impl<H: AxisName, V: AxisName, W, B, I> Tree<H, V, W, B, I>
where
    W: Borrow<HbmTensor<bf16, Chip, m![V, H]>>,
    B: Borrow<HbmTensor<f32, Chip, m![V]>>,
    I: Borrow<HbmTensor<i32, Chip, m![V]>>,
{
    /// This head with its bias and indices written in data memory.
    pub(crate) fn to_dm(&self, device: &mut Device) -> InDm<'_, H, V> {
        let columns = ToDm::<H>(PhantomData);
        Greedy::new(
            Affine::new(
                self.head.weight.borrow(),
                columns.forward(&mut *device, (), &mut (), self.head.bias.borrow()),
            ),
            self.argmax.to_dm(device),
        )
    }
}

/// The projection's weight under `weight`, checked as [`Embedding::table`] checks a token table
/// and read by a leaf that holds values; its bias and the argmax's ranks come from the config.
impl<'a, H: AxisName, V: AxisName, W, B, I> Load<&'a Checkpoint> for Tree<H, V, W, B, I>
where
    W: for<'t> Load<&'t Table<'t, V, H, Chip>>,
    B: Load<HostTensor<f32, m![{ Chip }, { m![V] }]>>,
    Argmax<V, Chip, I>: Load<&'a Checkpoint>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let weight = cx.field("head", "", |cx| {
                cx.field("weight", "weight", |cx| {
                    cx.compare(from)?;
                    let table = Embedding::<V, H, Chip>::table(from, cx)?;
                    W::load(cx, &*table)
                })
            });
            let weight = children.ask(weight)?;
            // Zero for the `vocab_size` tokens, the lowest finite score for the padded rows: no
            // subtraction of two biases turns into a NaN.
            let bias = cx.field("head", "", |cx| {
                cx.field("bias", "bias", |cx| {
                    let tokens = Embedding::<V, H, Chip>::tokens(from, cx)?;
                    let bias = (0..V::SIZE).map(|row| if row < tokens { 0.0 } else { f32::MIN });
                    let bias = HostTensor::from_vec(bias.collect::<Vec<_>>());
                    B::load(cx, bias)
                })
            });
            let bias = children.ask(bias)?;
            let argmax = cx.field("argmax", "argmax", |cx| {
                Argmax::<V, Chip, I>::load(cx, from)
            });
            let argmax = children.ask(argmax)?;
            let parts = weight.zip(bias).zip(argmax);
            children
                .node(parts)
                .map(|((weight, bias), argmax)| Greedy::new(Affine::new(weight, bias), argmax))
        })
    }
}

/// The projection's weight written in place: one rule of an update; the bias and the ranks are
/// no update targets.
impl<'t, 's, H: AxisName, V: AxisName, W: 'static, B, I>
    Load<(&'t mut Tree<H, V, W, B, I>, &'s Checkpoint)> for &'t mut Tree<H, V, W, B, I>
where
    for<'u> &'u mut W: Load<(&'u mut W, HostTensor<bf16, m![{ Chip }, { m![V, H] }]>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Tree<H, V, W, B, I>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        let write = |cx: &mut Context<'_>| {
            let table = Embedding::<V, H, Chip>::table(from, cx)?;
            <&mut W as Load<_>>::load(cx, (&mut into.head.weight, table()?)).map(drop)
        };
        context.checkpoint(from, |cx| {
            cx.field("head", "", |cx| {
                cx.field("weight", "weight", |cx| {
                    cx.rule(|cx| cx.overwrite::<W>(from, write), |_, ()| Ok(()))
                })
            })
        })?;
        Ok(into)
    }
}

impl<'a, H: AxisName, V: AxisName, W, B, I, SourceWeight, SourceBias, SourceIndex>
    Load<&'a Tree<H, V, SourceWeight, SourceBias, SourceIndex>> for Tree<H, V, W, B, I>
where
    Affine<W, B>: Load<&'a Affine<SourceWeight, SourceBias>>,
    Argmax<V, Chip, I>: Load<&'a Argmax<V, Chip, SourceIndex>>,
{
    fn load(
        context: &mut Context<'_>,
        from: &'a Tree<H, V, SourceWeight, SourceBias, SourceIndex>,
    ) -> Result<Self, LoadError> {
        let head = context.field("head", "", |cx| Load::load(cx, &from.head))?;
        let argmax = context.field("argmax", "argmax", |cx| Load::load(cx, &from.argmax))?;
        Ok(Greedy::new(head, argmax))
    }
}

impl<'t, H: AxisName, V: AxisName, W, B, I> Load<(&'t mut Tree<H, V, W, B, I>, Tree<H, V, W, B, I>)>
    for &'t mut Tree<H, V, W, B, I>
where
    for<'u> &'u mut Affine<W, B>: Load<(&'u mut Affine<W, B>, Affine<W, B>)>,
    for<'u> &'u mut Argmax<V, Chip, I>: Load<(&'u mut Argmax<V, Chip, I>, Argmax<V, Chip, I>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Tree<H, V, W, B, I>, Tree<H, V, W, B, I>),
    ) -> Result<Self, LoadError> {
        let Greedy { head, argmax, .. } = from;
        context.field("head", "", |cx| {
            <&mut Affine<W, B> as Load<_>>::load(cx, (&mut into.head, head))
        })?;
        context.field("argmax", "argmax", |cx| {
            <&mut Argmax<V, Chip, I> as Load<_>>::load(cx, (&mut into.argmax, argmax))
        })?;
        Ok(into)
    }
}

/// Each row's token; every row of the envelope is projected.
impl<H: AxisName, V: AxisName, R: AxisName, Env: Copy> Module<&Spread<H, R>, Env, &mut ()>
    for InDm<'_, H, V>
{
    type Output = reduction::Selected<R>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        input: &Spread<H, R>,
    ) -> reduction::Selected<R> {
        let mut candidates = self.argmax.candidates::<R>();
        self.head
            .forward(&mut *device, env, &mut (), (input, &mut candidates));
        self.argmax.forward(device, env, &mut (), candidates)
    }
}

impl<H: AxisName, V: AxisName, R: AxisName, W, B, I, Env: Copy>
    Module<&mut Rows<'_, bf16, H, R>, Env, &mut ()> for Tree<H, V, W, B, I>
where
    W: Borrow<HbmTensor<bf16, Chip, m![V, H]>>,
    B: Borrow<HbmTensor<f32, Chip, m![V]>>,
    I: Borrow<HbmTensor<i32, Chip, m![V]>>,
{
    type Output = HbmTensor<i32, Chip, m![R]>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        input: &mut Rows<'_, bf16, H, R>,
    ) -> Self::Output {
        let hidden = ToDm::<H>(PhantomData).forward(&mut *device, env, &mut (), &*input);
        let tokens = self
            .to_dm(&mut *device)
            .forward(&mut *device, env, &mut (), &hidden);
        Store.forward(device, env, &mut (), &tokens)
    }
}

/// A sequence classifier's score head: one FP32 row of `H` coefficients a label, at most
/// [`COLUMNS`] labels, whose scores of a row are its dot products with them.
#[derive(DeviceSend)]
pub struct Classifier<H: AxisName>(Affine<Weights<H>, Bias>);

/// The scores of the gathered rows, label columns first.
impl<X: MaterializableScalar + FetchCast<f32>, H: AxisName, R: AxisName>
    crate::Op<&mut affine::Context<'_, X, H, R>> for Classifier<H>
{
    type Output = ();

    fn forward(&self, device: &mut Device, context: &mut affine::Context<'_, X, H, R>) {
        crate::Op::forward(&self.0, device, context);
    }
}

/// The rows of `weight`, `[labels, H]`; or, when the config names `classifier_from_token`, vLLM's
/// conversion of the vocabulary head: those tokens' rows of the sibling `lm_head.weight`, or of
/// `model.embed_tokens.weight` when tied, the second less the first when `method` is
/// `from_2_way_softmax`, which `is_original_qwen3_reranker` implies.
impl<'a, H: AxisName> Load<&'a Checkpoint> for Classifier<H> {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let config = from.config();
            let tokens = config.get("classifier_from_token");
            let name = match (tokens, config["tie_word_embeddings"].as_bool()) {
                (None, _) => cx.field("weight", "weight", |cx| cx.name()),
                (Some(_), Some(true)) => cx.sibling("model.embed_tokens.weight"),
                (Some(_), _) => cx.sibling("lm_head.weight"),
            };
            cx.read(&name);
            let view = from
                .get(&name)
                .ok_or_else(|| LoadError::Missing(vec![name.clone()]))?;
            let invalid = |name: &str, reason: String| LoadError::InvalidData {
                name: name.into(),
                reason,
            };
            let values = Checkpoint::floats(&view)
                .ok_or_else(|| invalid(&name, format!("stored as {:?}", view.dtype())))?;
            if view.shape().len() != 2 || view.shape()[1] != H::SIZE {
                return Err(invalid(
                    &name,
                    format!("shape {:?}, rows of {} features", view.shape(), H::SIZE),
                ));
            }
            let row = |index: usize| values[index * H::SIZE..][..H::SIZE].to_vec();
            let rows: Vec<Vec<f32>> = match tokens {
                None => (0..view.shape()[0]).map(row).collect(),
                Some(tokens) => {
                    let key = "config.json:classifier_from_token";
                    cx.read(key);
                    let tokens = tokens
                        .as_array()
                        .ok_or_else(|| invalid(key, format!("{tokens} is no list of tokens")))?;
                    let ids = tokens.iter().map(|token| {
                        let token = token
                            .as_str()
                            .ok_or_else(|| invalid(key, format!("{token} is no token")))?;
                        let id = from.token(token)? as usize;
                        match id < view.shape()[0] {
                            true => Ok(row(id)),
                            false => Err(invalid(key, format!("{token} has no head row"))),
                        }
                    });
                    let rows = ids.collect::<Result<Vec<_>, _>>()?;
                    let original = config["is_original_qwen3_reranker"].as_bool() == Some(true);
                    let method = match config["method"].as_str() {
                        Some(method) => method,
                        None if original => "from_2_way_softmax",
                        None => "no_post_processing",
                    };
                    match (method, rows.as_slice()) {
                        ("no_post_processing", _) => rows,
                        ("from_2_way_softmax", [no, yes]) => {
                            vec![yes.iter().zip(no).map(|(yes, no)| yes - no).collect()]
                        }
                        (method, _) => {
                            return Err(LoadError::Unsupported {
                                name: "config.json:method".into(),
                                reason: format!("{method} of {} tokens", rows.len()),
                            });
                        }
                    }
                }
            };
            if !(1..=COLUMNS).contains(&rows.len()) {
                return Err(LoadError::Unsupported {
                    name,
                    reason: format!("{} labels, a head compiles 1 to {COLUMNS}", rows.len()),
                });
            }
            let weights = rows.into_iter().enumerate().map(|(label, row)| {
                cx.field(&label.to_string(), "", |cx| {
                    cx.upload::<f32, m![1], m![H], m![H]>(&HostTensor::from_vec(row))
                })
            });
            let weights = weights.collect::<Result<Vec<_>, _>>()?;
            Ok(Self(Affine::new(
                Slice::from(weights),
                Slice::from(Vec::new()),
            )))
        })
    }
}

/// A function of each row, applied to the live rows of a call one tile of up to sixteen rows a
/// trip, for a call of at least one row; result rows past the live count are unspecified.
#[derive(DeviceSend)]
pub struct RowWise<F>(pub F);

type RowTile<R> = Symbol<Tile<R, 16>>;

impl<F, D, W, R, S, Env: core::marker::Copy> Module<&mut Rows<'_, D, W, R>, Env, &mut ()>
    for RowWise<F>
where
    D: MaterializableScalar,
    W: AxisName,
    R: AxisName,
    S: AxisName,
    F: for<'x, 's> Module<
            &'x HbmTensor<D, Chip, m![{ RowTile<R> }, W]>,
            Env,
            &'s mut (),
            Output = HbmTensor<f32, Chip, m![{ RowTile<R> }, S]>,
        >,
{
    type Output = HbmTensor<f32, Chip, m![R, S]>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        input: &mut Rows<'_, D, W, R>,
    ) -> Self::Output {
        let live = input.live();
        let tiles = live.count::<Tile<R, 16>>();
        let mut output: HbmTensor<f32, Chip, m![R, S]> = HbmTensor::new();
        let mut apply = |device: &mut Device, tile: usize| {
            let rows = gather::Copy::<W, Tile<R, 16>, D>::default().forward(
                &mut *device,
                env,
                &mut (),
                ((input.source(), live), tile),
            );
            let result = self.0.forward(&mut *device, env, &mut (), &rows);
            Store.forward(device, env, &mut (), (&result, (&mut output, tile)));
        };
        // A launch has a live row, so its first tile runs outside any runtime condition, where a
        // not yet implemented function's panic still lowers.
        apply(device, 0);
        for trip in 0..const { <m![R / ({ RowTile<R> })] as M>::SIZE - 1 } {
            if trip + 1 < tiles {
                apply(device, trip + 1);
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::axis::{Eight, Rows};
    use crate::tests::DEVICE;

    // No head tiles a vocabulary of 128256 tokens exactly; it compiles them on 129024 rows.
    axes![Hidden = 1024, Compiled = 129024, Padded = 152064];

    /// A checkpoint vocabulary below the compiled one, so the bias, not the padding, bounds the
    /// selection.
    const LIVE: usize = 151_936 - 256;

    #[device(chip = 1)]
    fn selected<V: AxisName, R: AxisName>(
        device: &mut Device,
        head: &Greedy<Hidden, V>,
        input: &HbmTensor<bf16, Chip, m![R, Hidden]>,
    ) -> HbmTensor<i32, Chip, m![R]> {
        let input: Spread<Hidden, R> = input.to_dm(&mut device.tdma);
        head.to_dm(&mut *device)
            .forward(&mut *device, (), &mut (), &input)
            .to_hbm(&mut device.tdma)
    }

    /// Greedy selects each row's first maximum among the checkpoint's tokens across ties and pass
    /// boundaries, never a padded row, at every row envelope, also when the compiled vocabulary
    /// pads a checkpoint that no head tiles exactly.
    #[tokio::test(flavor = "current_thread")]
    async fn selects_first_maximum() {
        selects_first_maximum_of::<Padded, Eight>(LIVE).await;
        selects_first_maximum_of::<Padded, Rows<32>>(LIVE).await;
        selects_first_maximum_of::<Padded, Rows<64>>(LIVE).await;
        selects_first_maximum_of::<Compiled, Rows<16>>(128_256).await;
    }

    /// Row `r` runs case `r % 9` on its own pair of input columns, of `live` checkpoint tokens
    /// compiled on `V`.
    async fn selects_first_maximum_of<V: AxisName, R: AxisName>(live: usize) {
        let cases: [(&[(usize, f32)], usize); 9] = [
            (&[(42, 5.0)], 42),
            (&[(7, 5.0), (V::SIZE / 3 + 19, 5.0)], 7),
            (
                &[(live - 1, 5.0), (live, 10.0), (V::SIZE - 1, 10.0)],
                live - 1,
            ),
            (&[(0, 5.0f32.next_down()), (live - 1, 5.0)], live - 1),
            (&[(2 * V::SIZE / 3 + 5, 3.0)], 2 * V::SIZE / 3 + 5),
            (&[(1234, -0.5), (1233, -0.5)], 1233),
            (
                &[(V::SIZE / 3 - 1, 2.0), (V::SIZE / 3, 2.0)],
                V::SIZE / 3 - 1,
            ),
            (&[(99_999, 7.0)], 99_999),
            (&[(live - 1, 3.0), (live, 3.0)], live - 1),
        ];
        let case = |row: usize| cases[row % cases.len()];
        let mut weight = vec![bf16::from_f32(0.0); V::SIZE * Hidden::SIZE];
        for token in 0..V::SIZE {
            for row in 0..R::SIZE {
                weight[token * Hidden::SIZE + 2 * row] = bf16::from_f32(-1.0);
            }
        }
        let mut input = vec![bf16::from_f32(0.0); R::SIZE * Hidden::SIZE];
        for row in 0..R::SIZE {
            input[row * Hidden::SIZE + 2 * row] = bf16::from_f32(1.0);
            input[row * Hidden::SIZE + 2 * row + 1] = bf16::from_f32(1.0);
            for &(token, value) in case(row).0 {
                // Two BF16 coefficients preserve the FP32 next_down case through the contraction.
                let high = bf16::from_f32(value);
                let low = bf16::from_f32(value - high.to_f32());
                assert_eq!(high.to_f32() + low.to_f32(), value);
                weight[token * Hidden::SIZE + 2 * row] = high;
                weight[token * Hidden::SIZE + 2 * row + 1] = low;
            }
        }
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let weight = HostTensor::<bf16, m![V, Hidden]>::from_vec(weight)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let input = HostTensor::<bf16, m![R, Hidden]>::from_vec(input)
            .to_hbm::<Chip, m![R, Hidden]>(&mut device.pdma)
            .await
            .unwrap();
        let mask = (0..V::SIZE).map(|token| if token < live { 0.0 } else { f32::MIN });
        let mask = HostTensor::<f32, m![V]>::from_vec(mask)
            .to_hbm::<Chip, m![V]>(&mut device.pdma)
            .await
            .unwrap();
        let argmax = Argmax::<V>::load(&mut Context::new(&mut device), &Checkpoint::default());
        let head = Greedy::<Hidden, V>::new(Affine::new(weight, mask), argmax.unwrap());
        let tokens = launch(selected, (&mut device, &head, &input))
            .await
            .unwrap()
            .to_host::<m![R]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (row, &token) in tokens.iter().enumerate() {
            assert_eq!(token as usize, case(row).1, "row {row}");
        }
    }
}
