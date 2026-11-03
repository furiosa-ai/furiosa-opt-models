//! Llama's NVFP4 catalog tree in the original tiled decode order. QKV restoration is
//! overlapped with the preceding MLP, O weights with attention's last history transfer,
//! and the first MLP operands with O. The native cache stage writes fresh BF16 K/V
//! around the current layer's attention, using the same retained-row flow as Qwen FP8.
use crate::models::decoder::attention::{Head, KvHeads, llama::nvfp4::*, qwen3::launch};
use crate::modules::generic::schedule::{Fill, Spill};
use crate::modules::language::Backbone;
use crate::modules::residual::Residual;
use crate::ops::attention::full::paged::Retained;
use crate::ops::embedding;
use crate::ops::linear::nvfp4::tiled::{heads, mlp, panels::Panels};
use crate::ops::normalization::{
    RmsNorm,
    span::{self, Span, Stage},
};
use crate::ops::pointwise::add::Add;
use crate::tensor::{live::Live, parallel::Chip};
use crate::{Module, Op, Sequential, Then};
use furiosa_opt_std::prelude::*;

mod prefetch;
type Capacity = launch::Capacity<Size>;
type Reach = <Size as crate::specialize::attention::Decode>::Reach;
type Slots = <Size as crate::specialize::attention::Decode>::Slots;

impl<'i, 'a, 's, 'b, Env> Module<&'i mut launch::Input<'a, Size>, Env, launch::State<'s, 'b, Size>>
    for ForCausalLM
{
    type Output = ();
    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        state: launch::State<'s, 'b, Size>,
        input: &'i mut launch::Input<'a, Size>,
    ) {
        let causal = &self.model;
        let live = Live::<(), Capacity>::read(&input.count);
        let (context, selected) = Op::forward(
            &causal.model.rotary_emb,
            &mut *device,
            (live, &mut input.cache),
        );
        let (((hidden, slots), ()), tokens) = state;
        let Then(Backbone { embedding, body }, norm) = &causal.model.body;
        Op::forward(
            &embedding.0,
            &mut *device,
            embedding::Launch {
                ids: &input.tokens,
                count: live,
                output: &mut *hidden,
            },
        );
        let layers = &body.layers.0;
        let spans = live.count::<Span>();
        let mut query: HbmTensor<bf16, Chip, m![Reach, QWidth]> = HbmTensor::new();
        let mut retained = (
            HbmTensor::<bf16, Chip, m![Reach, KvHeads, Head]>::new(),
            HbmTensor::<bf16, Chip, m![Reach, KvHeads, Head]>::new(),
        );
        let first = layers.layer(0);
        let Sequential(Residual(first_norm, attention, _), _) = &first;
        for index in 0..spans {
            let tile = live.tile::<Span>(index);
            let rows = tile.with(hidden.fill(&mut *device, tile));
            let (gain, projection) = prefetch::Head {
                norm: &first_norm.0,
                qkv: &attention.0.qkv.projection,
                enabled: true,
            }
            .load(device);
            let rows = Op::forward(&gain, &mut *device, &rows);
            heads::project::<Hidden, QWidth, KvWidth, Head, KvHeads, QueryHeads, Reach, Slots>(
                device,
                (
                    &attention.0.qkv.projection.query.0,
                    &attention.0.qkv.projection.key.0,
                    &attention.0.qkv.projection.value.0,
                ),
                (&projection.query, &projection.key, &projection.value),
                rows.value(),
                &context.positions,
                &mut query,
                &mut retained.0,
                &mut retained.1,
                tile.start(),
                tile.rows(),
            );
        }
        for index in 0..DEPTH {
            let layer = layers.layer(index);
            let next = layers.layer((index + 1) % DEPTH);
            let Sequential(Residual(_, attention, _), Residual(ff_norm, feedforward, _)) = &layer;
            let Sequential(Residual(next_norm, next_attention, _), _) = &next;
            let attention = &attention.0;
            let pool = &mut slots[index].0.1;
            // Match Qwen FP8: item rows enter this layer's cache before attention;
            // prefill rows remain in scratch until attention has consumed them.
            let deferred = Module::forward(
                &attention.cache,
                &mut *device,
                (),
                &mut *pool,
                (&retained, (&context.write, &context.read), live),
            );
            // SAFETY: QWidth is exactly KvHeads * Groups * Head; only contiguous axes change.
            let queries = unsafe {
                query
                    .view()
                    .alias::<Chip, m![Reach, KvHeads, Groups, Head]>()
            };
            let history = Retained {
                pool: &**pool,
                key: &retained.0,
                value: &retained.1,
                read: &attention.cache.read,
            };
            let mut output_weight = (DmTensor::new(), DmTensor::new());
            let mut attended: HbmTensor<bf16, Chip, m![Reach, KvHeads, Groups, Head]> =
                HbmTensor::new();
            Module::forward(
                &attention.attend,
                &mut *device,
                (),
                &mut (),
                (
                    &queries,
                    &history,
                    &context.read,
                    (&attention.output.0, &mut output_weight),
                    &mut attended,
                ),
            );
            Module::forward(
                &attention.cache,
                &mut *device,
                (),
                &mut *pool,
                (deferred, &retained, &context.write),
            );
            // SAFETY: collapse the same head/group coordinates back into projection features.
            let attended = unsafe { attended.reshape::<Chip, m![Reach, QWidth]>() };
            for span in 0..spans {
                let tile = live.tile::<Span>(span);
                let mlp = &feedforward.0;
                let gain = Op::forward(&Stage, &mut *device, &ff_norm.0);
                let entry = mlp::stage(device, &mlp.gate_proj, &mlp.up_proj, &mlp.down_proj);
                let (output, entry) = attention.output.0.prepared_window_with_next(
                    device,
                    Live::<(), Span>::all().subrange(0, tile.rows()),
                    &attended,
                    tile.start(),
                    &output_weight,
                    entry,
                );
                let residual = tile.with(hidden.fill(&mut *device, tile));
                let residual = Op::forward(&Add, &mut *device, (&residual, &tile.with(output)));
                let normalized = Op::forward(&gain, &mut *device, &residual);
                let (output, (gain, projection)) = mlp::forward_prepared(
                    device,
                    &mlp.gate_proj,
                    &mlp.up_proj,
                    &mlp.down_proj,
                    normalized.value(),
                    tile.rows(),
                    entry,
                    prefetch::Head {
                        norm: &next_norm.0,
                        qkv: &next_attention.0.qkv.projection,
                        enabled: index + 1 < DEPTH,
                    },
                );
                let output = Op::forward(&Add, &mut *device, (&residual, &tile.with(output)));
                hidden.spill(&mut *device, output.value(), tile);
                if index + 1 < DEPTH {
                    let rows = Op::forward(&gain, &mut *device, &output);
                    let qkv = &next_attention.0.qkv.projection;
                    heads::project::<
                        Hidden,
                        QWidth,
                        KvWidth,
                        Head,
                        KvHeads,
                        QueryHeads,
                        Reach,
                        Slots,
                    >(
                        device,
                        (&qkv.query.0, &qkv.key.0, &qkv.value.0),
                        (&projection.query, &projection.key, &projection.value),
                        rows.value(),
                        &context.positions,
                        &mut query,
                        &mut retained.0,
                        &mut retained.1,
                        tile.start(),
                        tile.rows(),
                    );
                }
            }
        }
        causal
            .lm_head
            .read_normalized(device, hidden, selected, tokens, &norm.0);
    }
}
