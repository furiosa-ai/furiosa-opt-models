//! The pagings of a causal model whose layers interleave in two collections, as they bind a call:
//! each collection's layers cache in a group of their own, which the device pages, and a call is
//! one launch over both.
//!
//! - The cache holds a pool and a paged group for each collection, its layers by their position
//!   in the model; a group whose layer type bounds the positions a row attends back to reports the
//!   window its paging bound at load.
//! - A call stages both groups' histories, the byte offsets of the rows it returns and its token
//!   rows in one transfer, then launches the root over every layer's own blocks.

use super::{Batch, CacheSpec, Group, Groups, Kind, Limits, Pool, Root, Table, submit};
use furiosa_opt_models::Module;
use furiosa_opt_models::models::decoder::attention::{ForCausalLM, Geometry, LayerType, Model};
use furiosa_opt_models::models::decoder::launch::Tables;
use furiosa_opt_models::models::decoder::launch::interleaved::{self, Collections};
use furiosa_opt_models::models::decoder::{self, Attention};
use furiosa_opt_models::modules::attention::key_value::History;
use furiosa_opt_models::modules::language::Input;
use furiosa_opt_models::ops::attention::full::paged::Tiling;
use furiosa_opt_models::ops::attention::gqa::Cache;
use furiosa_opt_models::tensor::cache::{Block, Blocks};
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::DeviceOutput;
use std::convert::Infallible;
use std::io;

/// A layer's blocks of group `A`'s cache.
type Pages<A> = Blocks<<A as Attention>::Heads, <A as Attention>::Head, <A as Attention>::Cache>;

/// A launch's slots of the layers of a body `Body` whose collections attend in groups `A` and `B`.
type Slots<'b, Body, A, B> =
    <Body as Collections>::Slots<interleaved::Layer<'b, A>, interleaved::Layer<'b, B>>;

/// A causal model whose base model prepares each call's attention context with one paging a cache
/// group writes the selected rows' tokens; it runs no adapter.
impl<G: Geometry, Body, A, B, R, P, H> Root
    for ForCausalLM<G, Model<G, Body, (decoder::Paging<A, R>, decoder::Paging<B, R>), P>, H, P>
where
    A: Attention<Rotated: AxisName>,
    B: Attention<Rotated: AxisName>,
    R: AxisName,
{
    type Size = G;
    type Groups = (decoder::Paging<A, R>, decoder::Paging<B, R>);
    type Body = Body;
    type Place = HbmTensor<i32, Chip, m![R]>;
    type Row = m![1];
    type Adapting = Infallible;

    fn groups(&self) -> &Self::Groups {
        &self.model.rotary_emb
    }
}

/// One paging a group binds each collection of a body's layers to a cache group of its own, in the
/// collections' order, for calls of `R` rows: the device prepares each group's rows from the
/// history the host stages, so a call's rows attend under the adaptive tiling only.
impl<T, A, B, R> Groups<T> for (decoder::Paging<A, R>, decoder::Paging<B, R>)
where
    T: Root + DeviceSend + Sync,
    T::Body: Collections,
    A: Attention<Rotated: AxisName> + LayerType,
    B: Attention<Rotated: AxisName> + LayerType,
    R: AxisName,
    Pages<A>: Send,
    Pages<B>: Send,
    for<'b> Slots<'b, T::Body, A, B>: DeviceSend,
    for<'i, 's, 'b> T: Module<
            &'i mut interleaved::Input<R, A, B>,
            (),
            (Slots<'b, T::Body, A, B>, &'s mut T::Place),
            Output = (),
        >,
{
    const ADAPTERS: usize = 0;
    const FIXED: bool = false;
    type Adapter = Infallible;
    type Layers = (Vec<Pages<A>>, Vec<Pages<B>>);

    fn limits(&self, _: Tiling) -> Limits {
        let (first, second) = <T::Body as Collections>::positions();
        let group = |pool, layers, window: Option<usize>| Group {
            pool,
            layers,
            kind: Kind::Paged,
            window,
        };
        let pool = |bytes: i32| Pool {
            block: Block::SIZE,
            bytes: bytes as usize,
        };
        Limits {
            window: A::Windows::SIZE.max(B::Windows::SIZE) * Cache::SIZE,
            rows: R::SIZE,
            cache: CacheSpec {
                pools: vec![
                    pool(Pages::<A>::BLOCK_STRIDE),
                    pool(Pages::<B>::BLOCK_STRIDE),
                ],
                groups: vec![
                    group(0, first, A::WINDOW.map(|_| self.0.geometry.window())),
                    group(1, second, B::WINDOW.map(|_| self.1.geometry.window())),
                ],
            },
            adapters: 0,
        }
    }

    fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
        let (first, _) = <T::Body as Collections>::positions();
        let (first, second) = (storage.split_at_checked(first.len()))
            .ok_or_else(|| io::Error::other("a cache of the decoder's spec holds each layer"))?;
        Ok((pages::<A>(first)?, pages::<B>(second)?))
    }

    // The root is captured and launched in place, never held across an await: that `Send` check
    // erases lifetimes and cannot prove the root's higher-ranked bounds.
    fn launch(
        root: &T,
        device: &mut Device,
        (first, second): &mut Self::Layers,
        batch: Batch<'_, Infallible>,
        _: Tiling,
    ) -> impl Future<Output = io::Result<T::Place>> + Send {
        async move {
            // The launch's hidden rows lie one after another, a row of `Hidden` BF16 features each.
            let row = <T::Size as Geometry>::Hidden::SIZE * size_of::<bf16>();
            // The launch takes both histories and the returned rows by value, as arrays it writes.
            let mut written =
                Vec::from(batch.history::<R, Tables<R, B::Windows>>((1, Table::Blocks))?);
            written.push(batch.returned::<R>(row)?);
            let (staged, views) = batch
                .stage::<R, Tables<R, A::Windows>, Chip>(
                    device,
                    (&written, &[]),
                    (0, Table::Blocks),
                )
                .await?;
            let mut views = views.into_iter();
            let mut next =
                || (views.next()).ok_or_else(|| io::Error::other("a staged array is missing"));
            let history = History {
                segments: alias(&next()?, ..)?,
                blocks: alias(&next()?, ..)?,
                selected: alias(&next()?, ..)?,
                extents: alias(&next()?, ..)?,
            };
            let operands = (((), staged.history), ((), history), alias(&next()?, ..)?);
            let mut input: interleaved::Input<R, A, B> =
                Input::new(alias(&staged.tokens, ..)?, staged.count, operands);
            let mut place: T::Place = DeviceOutput::alloc(device).map_err(io::Error::other)?;
            let slots = <T::Body as Collections>::slots(
                first.iter_mut().map(|blocks| (blocks, ())).collect(),
                second.iter_mut().map(|blocks| (blocks, ())).collect(),
            )
            .unwrap_or_else(|| unreachable!("a cache of the decoder's spec holds each layer"));
            submit(device, root, (slots, &mut place), &mut input)?;
            Ok(place)
        }
    }
}

/// Views of `storage`, a group's layers of a cache allocated for the decoder.
fn pages<A: Attention>(storage: &[HbmBuffer]) -> io::Result<Vec<Pages<A>>> {
    (storage.iter())
        .map(|layer| Blocks::from_buffer(layer.alias(..).map_err(io::Error::other)?))
        .collect()
}
