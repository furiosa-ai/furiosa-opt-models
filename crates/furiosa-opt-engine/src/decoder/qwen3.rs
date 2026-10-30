//! Qwen3's and Llama's listed roots of the packed tree as decoders: a call stages its requests,
//! streams the base model's body over every layer's own blocks into the call's rows, then keeps the
//! rows its requests keep.
//!
//! - The base model returns the kept rows' hidden values; the causal model runs its greedy head
//!   on them and returns their tokens; the sequence classifier runs its score head on them and
//!   returns their scores.
//! - A row of a BF16 or channel-scaled line attends its whole history as a work item of the size's
//!   attention, tiled as the call says.
//! - A call whose rows run with adapters passes each layer's attention and feed-forward projections
//!   the terms of the call's adapters and their selections of the rows. Only BF16 Qwen3 of 0.6B to
//!   4B runs adapters: the sizes of 8B geometry, Qwen3's, Llama's and Mistral's, lay their query,
//!   key and value rows out head-major, and a channel-scaled projection adds no update.

use super::adapters::{self, SLOTS};
use super::{Batch, CacheSpec, Decoder, Group, Kind, Limits, Output, Pool, Rows, Table, Tables};
use crate::{AsyncModule, Checkpoint, Function, LoadError, forward};
use furiosa_opt_models::load::{Context, Load};
use furiosa_opt_models::models::decoder::attention::{Geometry, llama, qwen3};
use furiosa_opt_models::models::decoder::attention::{Head, KvHeads};
use furiosa_opt_models::models::decoder::form;
use furiosa_opt_models::modules::language::Input;
use furiosa_opt_models::modules::lora::{self, Adapters, Rank};
use furiosa_opt_models::ops::attention::full::paged::{Tiling, chunks::Masks};
use furiosa_opt_models::ops::attention::gqa::Cache as Window;
use furiosa_opt_models::ops::embedding::Ids;
use furiosa_opt_models::ops::index::gather;
use furiosa_opt_models::ops::linear::affine::{self, Columns};
use furiosa_opt_models::specialize::attention::Decode;
use furiosa_opt_models::tensor::cache::{self, Block, Blocks};
use furiosa_opt_models::tensor::memory::{Hbm, alias};
use furiosa_opt_models::tensor::parallel::{Chip, Parallel};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::DeviceOutput;
use std::{convert::Infallible, io};

/// What a call of the size `S` holds: the compiled context, the size's rows, one paged group of
/// every layer's blocks, and `adapters` adapters.
fn limits<S: Geometry + Decode>(adapters: usize) -> Limits {
    Limits {
        window: S::Windows::SIZE * Window::SIZE,
        rows: S::Rows::SIZE,
        cache: CacheSpec {
            pools: vec![Pool {
                block: Block::SIZE,
                bytes: Blocks::<KvHeads, Head>::BLOCK_STRIDE as usize,
            }],
            groups: vec![Group {
                pool: 0,
                layers: 0..S::LAYERS,
                kind: Kind::Paged,
                window: None,
            }],
        },
        adapters,
    }
}

/// Each layer's blocks, viewed in the storage of a cache allocated for the decoder's spec.
fn views<L: TryFrom<Vec<Blocks<KvHeads, Head>>>>(storage: &[HbmBuffer]) -> io::Result<L> {
    let views = (storage.iter())
        .map(|layer| Blocks::from_buffer(layer.alias(..).map_err(io::Error::other)?))
        .collect::<io::Result<Vec<_>>>()?;
    Ok(views
        .try_into()
        .unwrap_or_else(|_| unreachable!("a cache of the decoder's spec holds each layer")))
}

/// The rows `kept` of a call's hidden rows, in order: a view of one ascending run, else a copy.
fn keep<W: AxisName>(
    device: &mut Device,
    hidden: &HbmBuffer,
    kept: &[usize],
) -> io::Result<Rows<bf16, m![W]>> {
    let first = kept.first().copied().unwrap_or(0);
    if (kept.iter().enumerate()).all(|(at, &row)| row == first + at) {
        return Rows::new(hidden, first..first + kept.len());
    }
    let bytes = Rows::<bf16, m![W]>::bytes();
    let target = HbmBuffer::alloc(device, kept.len() * bytes).map_err(io::Error::other)?;
    let row = |buffer: &HbmBuffer, at: usize| {
        (buffer.alias(at * bytes..(at + 1) * bytes)).map_err(io::Error::other)
    };
    let mut pairs = (kept.iter().enumerate())
        .map(|(at, &source)| Ok((row(hidden, source)?, row(&target, at)?)))
        .collect::<io::Result<Vec<_>>>()?;
    let pairs = pairs.iter_mut().map(|(from, to)| (&*from, to));
    furiosa_opt_models::ops::index::copy(&mut *device, pairs)?;
    Rows::new(&target, 0..kept.len())
}

/// The decoders of the listed sizes `$size` whose projections hold form `$form`: the base model
/// and the causal model, which run `SLOTS` adapters a call when `adapted`, and none when their
/// front is `normalized`, Qwen3's, or `projected`, Llama's.
macro_rules! decoders {
    ($(($size:ty, $form:ty) => $kind:ident),+ $(,)?) => {$(
        /// The kept rows' hidden values.
        impl Decoder for qwen3::Model<$size, $form, Parallel<1>, cache::Bf16, Hbm> {
            const OUTPUT: Output = Output::Hidden;
            const ADAPTERS: usize = decoders!(@slots $kind);
            type Output = Rows<bf16, Symbol<<$size as Geometry>::Hidden>>;
            type Layers = <$size as Geometry>::Layers<Blocks<KvHeads, Head>>;
            type Adapter = decoders!(@adapter $kind $size);

            fn limits(&self) -> Limits {
                limits::<$size>(Self::ADAPTERS)
            }

            fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
                views(storage)
            }

            decoders!(@load $kind $size);

            async fn call(
                &self,
                device: &mut Device,
                layers: &mut Self::Layers,
                batch: Batch<'_, Self::Adapter>,
                tiling: Tiling,
            ) -> io::Result<Self::Output> {
                type Size = $size;
                type R = <Size as Decode>::Rows;
                type Work = <Size as Decode>::Work;
                let (segments, ids) = batch.row_segments(0);
                let mut reads = Work::sequential::<R>(&segments, ids, tiling)?;
                if let Some(binding) = batch.adapters() {
                    reads.extend(binding.selections::<R, Rank<3>>());
                    reads.extend(binding.selections::<R, Rank<1>>());
                }
                let (staged, views) = batch
                    .stage::<R, Tables<R, <Size as Decode>::Windows>, Chip>(
                        device,
                        &reads,
                        (0, Table::Blocks),
                    )
                    .await?;
                // Placed each call: constants the calls could share, two small uploads a call.
                let masks = Masks::new(device).await?;
                let mut views = views.into_iter();
                let mut next =
                    || views.next().ok_or_else(|| io::Error::other("a read array is missing"));
                let work = Work::bind(&mut next, &masks)?;
                let tokens: Ids<R> = alias(&staged.tokens, ..)?;
                let mut call = Input::new(tokens, staged.count, ((), (staged.history, &work)));
                let model: qwen3::Model<Size, $form, Parallel<1>, cache::Bf16, &Hbm> =
                    Load::load(&mut Context::new(None), self).map_err(io::Error::other)?;
                type W = <Size as Geometry>::Hidden;
                let rows: HbmTensor<bf16, Chip, m![R, W]> =
                    DeviceOutput::alloc(device).map_err(io::Error::other)?;
                let body = Function::new(&model, forward);
                let rows = decoders!(@run $kind, batch, next, layers, rows, body, device, call);
                keep(device, &rows.into_buffer(), batch.kept())
            }
        }

        /// The kept rows' greedy tokens.
        impl Decoder
            for qwen3::ForCausalLM<$size, $form, Parallel<1>, cache::Bf16, Hbm>
        {
            const OUTPUT: Output = Output::Tokens;
            const ADAPTERS: usize = decoders!(@slots $kind);
            type Output = Rows<i32, m![1]>;
            type Layers = <$size as Geometry>::Layers<Blocks<KvHeads, Head>>;
            type Adapter = decoders!(@adapter $kind $size);

            fn limits(&self) -> Limits {
                self.model.limits()
            }

            fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
                views(storage)
            }

            decoders!(@load $kind $size);

            async fn call(
                &self,
                device: &mut Device,
                layers: &mut Self::Layers,
                batch: Batch<'_, Self::Adapter>,
                tiling: Tiling,
            ) -> io::Result<Rows<i32, m![1]>> {
                type Size = $size;
                type W = <Size as Geometry>::Hidden;
                let kept = self.model.call(device, layers, batch, tiling).await?;
                if kept.len() == 0 {
                    return Rows::new(kept.buffer(), 0..0);
                }
                let head: qwen3::LmHead<Size, &Hbm> =
                    Load::load(&mut Context::new(None), &self.lm_head).map_err(io::Error::other)?;
                let bytes = Rows::<bf16, Symbol<W>>::bytes();
                let views: gather::RowStorage<W, bf16> = (0..kept.len())
                    .map(|row| alias(kept.buffer(), row * bytes..(row + 1) * bytes))
                    .collect::<io::Result<_>>()?;
                let mut rows = gather::Rows::<bf16, W, <Size as Decode>::Rows>::new(device, &views)
                    .await?;
                let head = Function::new(&head, forward);
                let tokens = head.forward(device, (), &mut (), &mut rows).await?;
                Rows::new(&tokens.into_buffer(), 0..kept.len())
            }
        }
    )+};
    (@slots adapted) => { SLOTS };
    (@slots $kind:ident) => { 0 };
    (@adapter adapted $size:ty) => { qwen3::Adapter<$size, Hbm> };
    (@adapter $kind:ident $size:ty) => { Infallible };
    // A PEFT checkpoint's terms are built on host from the checkpoint read once, then uploaded.
    (@load adapted $size:ty) => {
        fn adapter(
            checkpoint: &Checkpoint,
            device: &mut Device,
        ) -> Result<Self::Adapter, LoadError> {
            lora::load::<qwen3::Adapter<$size, Hbm>, qwen3::Adapter<$size, lora::Staged>>(
                checkpoint, device,
            )
        }
    };
    (@load $kind:ident $size:ty) => {};
    // A layer's front slots around its query, key and value projection's `$qkv`: Qwen3's normalize
    // each query and key head after it, Llama's take the heads as projected.
    (@front projected $qkv:expr) => { $qkv };
    (@front $kind:ident $qkv:expr) => { ($qkv, (), ()) };
    // Each layer's slots: its front's, its cache blocks and its output projection's, then its
    // feed-forward's gate, up and down projections'.
    (@run adapted, $batch:ident, $next:ident, $layers:ident, $rows:ident, $body:ident, $device:ident, $call:ident) => {{
        match $batch.adapters() {
            None => decoders!(@run normalized, $batch, $next, $layers, $rows, $body, $device, $call),
            Some(binding) => {
                let packed = adapters::selected::<R, Rank<3>>(&mut $next)?;
                let output = adapters::selected::<R, Rank<1>>(&mut $next)?;
                let held = binding.held();
                let mut layer = 0;
                let slots = $layers.each_mut().map(|blocks| {
                    let terms = held.map(|adapter| &adapter.0.0.0.0[layer]);
                    layer += 1;
                    let (attention, mlp) = (terms.map(|t| &t.0.0), terms.map(|t| &t.1.0));
                    let qkv = Adapters {
                        terms: attention.map(|terms| terms.0.term()),
                        selections: &packed,
                    };
                    let o = Adapters {
                        terms: attention.map(|terms| &terms.1.0),
                        selections: &output,
                    };
                    let gate = Adapters {
                        terms: mlp.map(|terms| &terms.0.0),
                        selections: &output,
                    };
                    let up = Adapters {
                        terms: mlp.map(|terms| &terms.1.0),
                        selections: &output,
                    };
                    let down = Adapters {
                        terms: mlp.map(|terms| &terms.2.0),
                        selections: &output,
                    };
                    ((((qkv, ()), (), ()), blocks, o), (((gate, ()), up), ((), down)))
                });
                let mut state = ((((), slots), ()), $rows);
                $body.forward($device, (), &mut state, &mut $call).await?;
                state.1
            }
        }
    }};
    (@run $kind:ident, $batch:ident, $next:ident, $layers:ident, $rows:ident, $body:ident, $device:ident, $call:ident) => {{
        let slots = $layers.each_mut().map(|blocks| {
            ((decoders!(@front $kind ((), ())), blocks, ()), ((((), ()), ()), ((), ())))
        });
        let mut state = ((((), slots), ()), $rows);
        $body.forward($device, (), &mut state, &mut $call).await?;
        state.1
    }};
}

decoders!(
    (qwen3::Size<"0.6B">, form::Affine) => adapted,
    (qwen3::Size<"1.7B">, form::Affine) => adapted,
    (qwen3::Size<"4B">, form::Affine) => adapted,
    (qwen3::Size<"8B">, form::Affine) => normalized,
    (llama::Size<"8B">, form::Affine) => projected,
    (llama::Size<"Mistral-7B">, form::Affine) => projected,
    (qwen3::Size<"0.6B">, form::ChannelScaled) => normalized,
    (qwen3::Size<"1.7B">, form::ChannelScaled) => normalized,
    (qwen3::Size<"4B">, form::ChannelScaled) => normalized,
);

/// The BF16 base model of the size `S` on one chip, whose calls a classifier of `S` makes.
type Base<S> = qwen3::Model<S, form::Affine, Parallel<1>, cache::Bf16, Hbm>;

/// The sequence classifiers of the listed Qwen3 sizes `$size`.
macro_rules! classifiers {
    ($($size:ty),+ $(,)?) => {$(
        /// The kept rows' scores, label columns first.
        impl Decoder
            for qwen3::ForSequenceClassification<
                $size,
                form::Affine,
                Parallel<1>,
                cache::Bf16,
                Hbm,
            >
        {
            const OUTPUT: Output = Output::Scores;
            type Output = Rows<f32, m![Columns]>;
            type Layers = <$size as Geometry>::Layers<Blocks<KvHeads, Head>>;
            /// The base model's, of which a classifier's calls run none.
            type Adapter = <Base<$size> as Decoder>::Adapter;

            fn limits(&self) -> Limits {
                Limits {
                    adapters: Self::ADAPTERS,
                    ..self.model.limits()
                }
            }

            fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
                views(storage)
            }

            async fn call(
                &self,
                device: &mut Device,
                layers: &mut Self::Layers,
                batch: Batch<'_, Self::Adapter>,
                tiling: Tiling,
            ) -> io::Result<Rows<f32, m![Columns]>> {
                type Size = $size;
                type W = <Size as Geometry>::Hidden;
                type R = <Size as Decode>::Rows;
                let kept = self.model.call(device, layers, batch, tiling).await?;
                if kept.len() == 0 {
                    return Rows::new(kept.buffer(), 0..0);
                }
                let bytes = Rows::<bf16, Symbol<W>>::bytes();
                let views: gather::RowStorage<W, bf16> = (0..kept.len())
                    .map(|row| alias(kept.buffer(), row * bytes..(row + 1) * bytes))
                    .collect::<io::Result<_>>()?;
                let scores = HbmBuffer::alloc(device, R::SIZE * Rows::<f32, m![Columns]>::bytes())
                    .map_err(io::Error::other)?;
                let mut context = affine::Context::<bf16, W, R> {
                    input: gather::Rows::new(device, &views).await?,
                    output: alias(&scores, ..)?,
                };
                let head = Function::new(&self.score, forward);
                head.forward(device, (), &mut (), &mut context).await?;
                Rows::new(&scores, 0..kept.len())
            }
        }
    )+};
}

classifiers!(
    qwen3::Size<"0.6B">,
    qwen3::Size<"1.7B">,
    qwen3::Size<"4B">,
    qwen3::Size<"8B">,
);
