//! Heads that read a body's output: readouts, poolers, and retrieval, sparse and masked heads.

pub use readout::Selected;

use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::embedding::{Embedding, Table};
use crate::ops::index::gather::{self, Rows};
use crate::ops::index::scatter::Store;
use crate::ops::linear::affine::{self, COLUMNS, Weights};
use crate::ops::linear::stream::{Columns, Logits, ToDm};
use crate::ops::linear::{self, Affine, Bias, Biased};
use crate::ops::normalization::span::Span;
use crate::ops::normalization::{Blocked, Spread};
use crate::ops::pointwise::activation::Softcap;
use crate::ops::reduction::{self, Argmax};
use crate::tensor::live::Live;
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
/// pass by pass, through its cap, as candidates, so no row's logits are stored whole.
#[derive(DeviceSend)]
pub struct Greedy<H: AxisName, V: AxisName, Head = Biased<H, V>, Select = Argmax<V>, Bound = ()> {
    /// The projection onto the padded vocabulary; its bias is zero for the checkpoint's tokens and
    /// the lowest finite score for the padded rows.
    pub head: Head,
    /// The first maximum of each row's logits.
    pub argmax: Select,
    /// The cap on every logit the argmax compares, the config's `final_logit_softcapping`: `()`
    /// for a head whose logits take none.
    pub cap: Bound,
    shape: PhantomData<(H, V)>,
}

impl<H: AxisName, V: AxisName, Head, Select, Bound> Greedy<H, V, Head, Select, Bound> {
    pub const fn new(head: Head, argmax: Select, cap: Bound) -> Self {
        Self {
            head,
            argmax,
            cap,
            shape: PhantomData,
        }
    }
}

/// A greedy head with its bias and the argmax's indices in data memory, where each call's passes
/// read them, its weight `W` and its cap borrowed.
pub(crate) type InDm<'a, H, V, W, C> =
    Greedy<H, V, Streamed<'a, H, V, W>, Argmax<V, Chip, Columns<H, V, i32>>, &'a C>;

/// A greedy head's projection as a call's passes stream it: its weight table `W` in HBM, its bias
/// in data memory.
type Streamed<'a, H, V, W> = Affine<&'a W, Columns<H, V, f32>>;

/// A greedy head whose projection's weight, bias and argmax ranks are held as `W`, `B` and `I`:
/// HBM tensors, or borrows of them in a borrowed tree; `C` caps its logits.
type Tree<H, V, W, B, I, C> = Greedy<H, V, Affine<W, B>, Argmax<V, Chip, I>, C>;

/// The config key a head's cap reads its bound under.
const CAP: &str = "final_logit_softcapping";

impl<H: AxisName, V: AxisName, W, B, I, C> Tree<H, V, W, B, I, C>
where
    W: linear::Table<Chip, Hidden = H, Vocabulary = V>,
    B: Borrow<HbmTensor<f32, Chip, m![V]>>,
    I: Borrow<HbmTensor<i32, Chip, m![V]>>,
{
    /// This head with its bias and indices written in data memory.
    pub(crate) fn to_dm(&self, device: &mut Device) -> InDm<'_, H, V, W, C> {
        let columns = ToDm::<H>(PhantomData);
        Greedy::new(
            Affine::new(
                &self.head.weight,
                columns.forward(&mut *device, (), &mut (), self.head.bias.borrow()),
            ),
            self.argmax.to_dm(device),
            &self.cap,
        )
    }
}

/// The projection's weight under `weight`, checked as [`Embedding::table`] checks a token table
/// and read by a leaf that holds values; its bias, the argmax's ranks and the cap come from the
/// config.
impl<'a, H: AxisName, V: AxisName, W, B, I, C> Load<&'a Checkpoint> for Tree<H, V, W, B, I, C>
where
    W: for<'t> Load<&'t Table<'t, V, H, Chip>>,
    B: Load<HostTensor<f32, m![{ Chip }, { m![V] }]>>,
    Argmax<V, Chip, I>: Load<&'a Checkpoint>,
    C: Load<&'a Checkpoint>,
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
            let cap = cx.field("cap", "", |cx| cx.config(CAP, |cx| C::load(cx, from)));
            let cap = children.ask(cap)?;
            let parts = weight.zip(bias).zip(argmax).zip(cap);
            children.node(parts).map(|(((weight, bias), argmax), cap)| {
                Greedy::new(Affine::new(weight, bias), argmax, cap)
            })
        })
    }
}

/// The projection's weight written in place: one rule of an update; the bias, the ranks and the
/// cap are no update targets.
impl<'t, 's, H: AxisName, V: AxisName, W: 'static, B, I, C>
    Load<(&'t mut Tree<H, V, W, B, I, C>, &'s Checkpoint)> for &'t mut Tree<H, V, W, B, I, C>
where
    for<'u> &'u mut W: Load<(&'u mut W, HostTensor<bf16, m![{ Chip }, { m![V, H] }]>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Tree<H, V, W, B, I, C>, &'s Checkpoint),
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

impl<'a, H: AxisName, V: AxisName, W, B, I, C, SourceWeight, SourceBias, SourceIndex, SourceCap>
    Load<&'a Tree<H, V, SourceWeight, SourceBias, SourceIndex, SourceCap>>
    for Tree<H, V, W, B, I, C>
where
    Affine<W, B>: Load<&'a Affine<SourceWeight, SourceBias>>,
    Argmax<V, Chip, I>: Load<&'a Argmax<V, Chip, SourceIndex>>,
    C: Load<&'a SourceCap>,
{
    fn load(
        context: &mut Context<'_>,
        from: &'a Tree<H, V, SourceWeight, SourceBias, SourceIndex, SourceCap>,
    ) -> Result<Self, LoadError> {
        let head = context.field("head", "", |cx| Load::load(cx, &from.head))?;
        let argmax = context.field("argmax", "argmax", |cx| Load::load(cx, &from.argmax))?;
        let cap = context.field("cap", "", |cx| Load::load(cx, &from.cap))?;
        Ok(Greedy::new(head, argmax, cap))
    }
}

impl<'t, H: AxisName, V: AxisName, W, B, I, C>
    Load<(&'t mut Tree<H, V, W, B, I, C>, Tree<H, V, W, B, I, C>)>
    for &'t mut Tree<H, V, W, B, I, C>
where
    for<'u> &'u mut Affine<W, B>: Load<(&'u mut Affine<W, B>, Affine<W, B>)>,
    for<'u> &'u mut Argmax<V, Chip, I>: Load<(&'u mut Argmax<V, Chip, I>, Argmax<V, Chip, I>)>,
    for<'u> &'u mut C: Load<(&'u mut C, C)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Tree<H, V, W, B, I, C>, Tree<H, V, W, B, I, C>),
    ) -> Result<Self, LoadError> {
        let Greedy {
            head, argmax, cap, ..
        } = from;
        context.field("head", "", |cx| {
            <&mut Affine<W, B> as Load<_>>::load(cx, (&mut into.head, head))
        })?;
        context.field("argmax", "argmax", |cx| {
            <&mut Argmax<V, Chip, I> as Load<_>>::load(cx, (&mut into.argmax, argmax))
        })?;
        context.field("cap", "", |cx| {
            <&mut C as Load<_>>::load(cx, (&mut into.cap, cap))
        })?;
        Ok(into)
    }
}

/// Each row's token: the projection's logits, through the cap, reach the argmax's candidates a
/// pass at a time; every row of the envelope is projected.
impl<H: AxisName, V: AxisName, R: AxisName, W, C, Env: Copy> Module<&Spread<H, R>, Env, &mut ()>
    for InDm<'_, H, V, W, C>
where
    W: linear::Table<Chip, Hidden = H, Vocabulary = V>,
    C: Cap<H, V>,
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
        self.cap.project(
            &mut *device,
            (&self.head, input),
            |device: &mut Device, logits: &Logits<H, V, R>, pass| {
                candidates.push(device, logits, pass);
            },
        );
        self.argmax.forward(device, env, &mut (), candidates)
    }
}

/// How each pass's logits reach a greedy head's argmax: as the projection leaves them for `()`, or
/// through the head's cap.
pub(crate) trait Cap<H: AxisName, V: AxisName> {
    /// Every pass's logits of `projection`'s input rows, capped, into `push` with the pass.
    fn project<R: AxisName, W: linear::Table<Chip, Hidden = H, Vocabulary = V>>(
        &self,
        device: &mut Device,
        projection: (&Streamed<'_, H, V, W>, &Spread<H, R>),
        push: impl FnMut(&mut Device, &Logits<H, V, R>, usize),
    );
}

impl<H: AxisName, V: AxisName> Cap<H, V> for () {
    fn project<R: AxisName, W: linear::Table<Chip, Hidden = H, Vocabulary = V>>(
        &self,
        device: &mut Device,
        (head, input): (&Streamed<'_, H, V, W>, &Spread<H, R>),
        push: impl FnMut(&mut Device, &Logits<H, V, R>, usize),
    ) {
        crate::Op::forward(head, device, (input, push));
    }
}

/// The softcap's bound and reciprocal reach the vector registers before the first pass, since each
/// pass's DMA queue holds the next pass's weights; each pass caps its logits before `push`.
impl<H: AxisName, V: AxisName, F> Cap<H, V> for Softcap<F>
where
    F: Borrow<HbmTensor<f32, Chip, m![1 # 8]>>,
{
    fn project<R: AxisName, W: linear::Table<Chip, Hidden = H, Vocabulary = V>>(
        &self,
        device: &mut Device,
        (head, input): (&Streamed<'_, H, V, W>, &Spread<H, R>),
        mut push: impl FnMut(&mut Device, &Logits<H, V, R>, usize),
    ) {
        let cap = self.to_vrf::<H, V, Chip>(&mut *device);
        let capped = |device: &mut Device, logits: &Logits<H, V, R>, pass| {
            let logits = crate::Op::forward(&cap, &mut *device, logits);
            push(device, &logits, pass);
        };
        crate::Op::forward(head, device, (input, capped));
    }
}

impl<H: AxisName, V: AxisName, R: AxisName, W, B, I, C, Env: Copy>
    Module<&mut Rows<'_, bf16, H, R>, Env, &mut ()> for Tree<H, V, W, B, I, C>
where
    W: linear::Table<Chip, Hidden = H, Vocabulary = V>,
    B: Borrow<HbmTensor<f32, Chip, m![V]>>,
    I: Borrow<HbmTensor<i32, Chip, m![V]>>,
    C: Cap<H, V>,
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

impl<H: AxisName> Classifier<H> {
    /// The scores of the rows `selected` names among a launch's normalized rows `hidden`, label
    /// columns first, into the first rows of `scores`; called once the rows' last effect is issued.
    pub(crate) fn read<Capacity: AxisName>(
        &self,
        context: &mut Device,
        hidden: &HbmTensor<bf16, Chip, Blocked<H, Capacity>>,
        selected: &Selected<Capacity>,
        scores: &mut HbmTensor<f32, Chip, m![Capacity, { Symbol<affine::Columns> }]>,
    ) {
        let chosen = Live::<(), Capacity>::read(&selected.live);
        for index in 0..chosen.count::<Span>() {
            let tile = chosen.tile::<Span>(index);
            let rows = selected.span(&mut *context, hidden, tile);
            let mut span = HbmTensor::<f32, Chip, m![Span, { Symbol<affine::Columns> }]>::new();
            crate::Op::forward(&self.0, &mut *context, (&rows, &mut span));
            span.view().to_hbm_view(
                &mut context.tdma,
                scores.view_mut().tile::<m![Capacity], { Span::SIZE }, m![
                    Capacity = { Span::SIZE } #{!} Capacity, { Symbol<affine::Columns> }
                ]>(tile.start()),
            );
        }
    }
}

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
