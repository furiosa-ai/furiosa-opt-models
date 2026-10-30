//! How a call's KV cache reaches its layers: the context a call's preparation builds from its
//! history (rotations, write slots, block tables), which every layer reads beside its own blocks.

use super::Context;
use crate::Module;
use crate::load::{self, Checkpoint, Load, LoadError};
use crate::modules::language::Selected;
use crate::ops::index::table::{self, BlockIndex, Derive, Extents, Fields, Logical};
use crate::ops::linear::query::Tile;
use crate::ops::rotary::{self, SplitHalf, group};
use crate::ops::{
    attention::{
        full::paged,
        gqa::{Cache, Pages},
        latent::{self, Slot},
    },
    rotary::{Coefficients, PairFrequencies, Rotation},
};
use crate::tensor::cache::Blocks;
use crate::tensor::mapping::Packets;
use crate::tensor::memory::{Hbm, Memory};
use crate::tensor::scalar::Value;
use crate::tensor::{live::Live, parallel::Chip};
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;
use std::marker::PhantomData;

/// Logical history operands, a copy on each chip of `Chips`; extents name segments, stored blocks
/// and selected rows, not live rows.
#[derive(DeviceSend)]
pub struct History<Segments: AxisName, Blocks: AxisName, Selected: AxisName, Chips: M = Chip> {
    pub segments: HbmTensor<i32, Chips, m![Segments, Fields]>,
    pub blocks: HbmTensor<i32, Chips, m![Blocks]>,
    pub selected: HbmTensor<i32, Chips, m![Selected]>,
    pub extents: HbmTensor<i32, Chips, m![3]>,
}

/// One attention geometry's context of a one-row step, prepared from the step's compact history.
#[derive(DeviceSend, Load)]
pub struct Preparation<Rows: AxisName, Geometry, Chips: M + 'static = Chip, X: Memory = Hbm> {
    pub derive: Derive<Rows, Chips, X>,
    pub geometry: Geometry,
}

#[derive(DeviceSend)]
pub struct Batch<Capacity: AxisName, Reach: AxisName, Chips: M + 'static = Chip> {
    pub positions: HbmTensor<i32, Chips, m![Capacity]>,
    pub slot: HbmTensor<i32, Chips, m![Reach]>,
}

#[derive(DeviceSend)]
pub struct Operands<
    Capacity: AxisName,
    Reach: AxisName,
    Sequences: AxisName,
    Slots,
    Heads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    const WAVES: usize,
    Chips: M + 'static = Chip,
> {
    pub batch: Batch<Capacity, Reach, Chips>,
    pub work: paged::Work<Sequences, Heads, Groups, Head, WAVES>,
    pub selected: Selected<Capacity>,
    pub slots: PhantomData<Slots>,
}

impl<'a, R, Reach, Sequences, Slots, Heads, Groups, Head, F, const WAVES: usize>
    crate::Op<(
        Live<(), R>,
        &mut &'a Operands<R, Reach, Sequences, Slots, Heads, Groups, Head, WAVES>,
    )> for SplitHalf<Head, F>
where
    F: group::Rotations<Half = rotary::Row<Head>>,
    R: AxisName,
    Reach: AxisName,
    Sequences: AxisName,
    Slots: AxisName,
    Heads: AxisName,
    Groups: AxisName,
    Head: AxisName,
{
    type Output = (
        Context<
            SplitHalf<Head, group::Rotation<Slots, rotary::Row<Head>>>,
            &'a HbmTensor<i32, Chip, m![Reach]>,
            paged::Context<'a, Sequences, Heads, Groups, Head, WAVES, R>,
        >,
        &'a Selected<R>,
    );

    fn forward(
        &self,
        device: &mut Device,
        (live, operands): (
            Live<(), R>,
            &mut &'a Operands<R, Reach, Sequences, Slots, Heads, Groups, Head, WAVES>,
        ),
    ) -> Self::Output {
        const {
            assert!(Reach::SIZE >= R::SIZE);
            assert!(Slots::SIZE == R::SIZE.div_ceil(Tile::SIZE));
        };
        let operands: &'a Operands<_, _, _, _, _, _, _, WAVES> = operands;
        let work = operands.work.read::<R>();
        let mut rotation = SplitHalf {
            frequency: HbmTensor::new(),
            head: PhantomData,
        };
        Module::forward(
            self,
            device,
            (),
            &mut (),
            group::Gather {
                positions: &operands.batch.positions,
                coefficients: &mut rotation.frequency,
                live: live.count::<crate::ops::linear::query::Tile>(),
            },
        );
        (
            Context {
                positions: rotation,
                write: &operands.batch.slot,
                read: work,
            },
            &operands.selected,
        )
    }
}

/// The latent context of a one-row step: one derivation of the step's history, then the row's
/// logical position, slot, visible length and block IDs.
impl<Rows: AxisName, A, B: AxisName, K: AxisName, Env: Copy>
    Module<(Live<(), Rows>, &mut (A, History<Rows, B, K>)), Env, &mut ()>
    for Preparation<Rows, latent::Prepare>
{
    type Output = Context<Slot, Slot, latent::History<B>>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (tiles, cache): (Live<(), Rows>, &mut (A, History<Rows, B, K>)),
    ) -> Self::Output {
        let history = &cache.1;
        let logical = self.derive.forward(
            &mut *device,
            env,
            &mut (),
            (&history.segments, &history.blocks),
        );
        let (positions, write, read) =
            self.geometry
                .forward(device, env, &mut (), (tiles, (&logical, &history.blocks)));
        Context {
            positions,
            write,
            read,
        }
    }
}

/// Load-time constants of the multi-row paged read at `Heads` heads of `Head` elements of `D`, for
/// `Rows` rows and `Windows` attention windows, in memory `X`.
#[derive(DeviceSend)]
pub struct Paging<
    Heads: AxisName,
    Head: AxisName,
    Rows: AxisName,
    Windows: AxisName,
    D = bf16,
    X: Memory = Hbm,
    Chips: M + 'static = Chip,
    Rotated: AxisName = Head,
    Layer = (),
> {
    /// The frequency of each coordinate, on every row.
    frequencies: X::Tensor<f32, Chips, m![Rows, Rotated]>,
    /// The phases and signs that turn each row's angles into its rotation.
    coefficients: Coefficients<Rotated, Chips, X::Tensor<f32, Chips, m![Rotated]>>,
    /// Longest visible history in tokens.
    window: X::Tensor<i32, Chips, m![1]>,
    /// Window `w`'s first token, negated: `-512 w` on every row.
    starts: X::Tensor<i32, Chips, m![Windows, { Packets<Rows> }]>,
    /// Window `w`'s first table row in bytes: `128 w` on every row.
    rows: X::Tensor<i32, Chips, m![Windows, { Packets<Rows> }]>,
    /// Zero on every row: a past-live row's history length.
    zeros: X::Tensor<i32, Chips, m![Rows]>,
    /// The middle of each window slot, `t + 1/2`, on every row.
    slots: X::Tensor<f32, Chips, m![Rows, Cache]>,
    shape: PhantomData<fn() -> (Heads, Head, D, Layer)>,
}

/// The constants of rows that turn at `frequencies` and attend at most `positions` positions,
/// within the compiled `Windows`: values no checkpoint stores, alike on every row and a copy on
/// each chip of `Chips`.
impl<
    'f,
    Heads: AxisName,
    Head: AxisName,
    Rows: AxisName,
    Windows: AxisName,
    D,
    X: Memory,
    Chips: M + 'static,
    Rotated: AxisName,
    Layer,
> Load<(&'f PairFrequencies<Rotated>, usize)>
    for Paging<Heads, Head, Rows, Windows, D, X, Chips, Rotated, Layer>
where
    X::Tensor<f32, Chips, m![Rows, Rotated]>:
        Load<HostTensor<f32, m![{ Chips }, { m![Rows, Rotated] }]>>,
    X::Tensor<f32, Chips, m![Rotated]>: Load<HostTensor<f32, m![{ Chips }, { m![Rotated] }]>>,
    X::Tensor<i32, Chips, m![1]>: Load<HostTensor<i32, m![{ Chips }, { m![1] }]>>,
    X::Tensor<i32, Chips, m![Windows, { Packets<Rows> }]>:
        Load<HostTensor<i32, m![{ Chips }, { m![Windows, { Packets<Rows> }] }]>>,
    X::Tensor<i32, Chips, m![Rows]>: Load<HostTensor<i32, m![{ Chips }, { m![Rows] }]>>,
    X::Tensor<f32, Chips, m![Rows, Cache]>:
        Load<HostTensor<f32, m![{ Chips }, { m![Rows, Cache] }]>>,
{
    fn load(
        cx: &mut load::Context<'_>,
        (frequencies, positions): (&'f PairFrequencies<Rotated>, usize),
    ) -> Result<Self, LoadError> {
        let window = positions.min(Windows::SIZE * Cache::SIZE);
        if window == 0 || window >= 1 << 24 {
            return Err(LoadError::InvalidData {
                name: cx.path(),
                reason: format!("{positions} positions are no paged context extent"),
            });
        }
        let lanes = <Packets<Rows> as M>::SIZE;
        let windows = |value: &dyn Fn(usize) -> i32| -> Vec<i32> {
            let values = (0..Windows::SIZE * lanes).map(|at| {
                if at % lanes < Rows::SIZE {
                    value(at / lanes)
                } else {
                    0
                }
            });
            values.collect::<Vec<_>>().repeat(Chips::SIZE)
        };
        let coefficients = cx.field("coefficients", "coefficients", |cx| {
            Load::load(cx, frequencies)
        })?;
        let frequencies = frequencies.coordinates().repeat(Rows::SIZE * Chips::SIZE);
        let window = vec![window as i32; Chips::SIZE];
        let block = (BlockIndex::SIZE * size_of::<i32>()) as i32;
        let zeros = vec![0; Rows::SIZE * Chips::SIZE];
        let slots =
            (0..Chips::SIZE * Rows::SIZE * Cache::SIZE).map(|at| (at % Cache::SIZE) as f32 + 0.5);
        Ok(Self {
            frequencies: cx.field("frequencies", "frequencies", |cx| {
                Load::load(cx, HostTensor::from_vec(frequencies))
            })?,
            coefficients,
            window: cx.field("window", "window", |cx| {
                Load::load(cx, HostTensor::from_vec(window))
            })?,
            starts: cx.field("starts", "starts", |cx| {
                let starts = windows(&|w| -((w * Cache::SIZE) as i32));
                Load::load(cx, HostTensor::from_vec(starts))
            })?,
            rows: cx.field("rows", "rows", |cx| {
                Load::load(cx, HostTensor::from_vec(windows(&|w| w as i32 * block)))
            })?,
            zeros: cx.field("zeros", "zeros", |cx| {
                Load::load(cx, HostTensor::from_vec(zeros))
            })?,
            slots: cx.field("slots", "slots", |cx| {
                Load::load(cx, HostTensor::from_vec(slots))
            })?,
            shape: PhantomData,
        })
    }
}

/// The constants are no update target.
impl<
    't,
    's,
    Heads: AxisName,
    Head: AxisName,
    Rows: AxisName,
    Windows: AxisName,
    D,
    X: Memory,
    Chips: M + 'static,
    Rotated: AxisName,
    Layer,
>
    Load<(
        &'t mut Paging<Heads, Head, Rows, Windows, D, X, Chips, Rotated, Layer>,
        &'s Checkpoint,
    )> for &'t mut Paging<Heads, Head, Rows, Windows, D, X, Chips, Rotated, Layer>
{
    fn load(
        context: &mut load::Context<'_>,
        (into, from): (
            &'t mut Paging<Heads, Head, Rows, Windows, D, X, Chips, Rotated, Layer>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |_| Ok(()))?;
        Ok(into)
    }
}

impl<
    'a,
    Heads: AxisName,
    Head: AxisName,
    Rows: AxisName,
    Windows: AxisName,
    D,
    X: Memory,
    Y: Memory,
    Chips: M + 'static,
    Rotated: AxisName,
    Layer,
> Load<&'a Paging<Heads, Head, Rows, Windows, D, Y, Chips, Rotated, Layer>>
    for Paging<Heads, Head, Rows, Windows, D, X, Chips, Rotated, Layer>
where
    X::Tensor<f32, Chips, m![Rows, Rotated]>: Load<&'a Y::Tensor<f32, Chips, m![Rows, Rotated]>>,
    X::Tensor<f32, Chips, m![Rotated]>: Load<&'a Y::Tensor<f32, Chips, m![Rotated]>>,
    X::Tensor<i32, Chips, m![1]>: Load<&'a Y::Tensor<i32, Chips, m![1]>>,
    X::Tensor<i32, Chips, m![Windows, { Packets<Rows> }]>:
        Load<&'a Y::Tensor<i32, Chips, m![Windows, { Packets<Rows> }]>>,
    X::Tensor<i32, Chips, m![Rows]>: Load<&'a Y::Tensor<i32, Chips, m![Rows]>>,
    X::Tensor<f32, Chips, m![Rows, Cache]>: Load<&'a Y::Tensor<f32, Chips, m![Rows, Cache]>>,
{
    fn load(
        context: &mut load::Context<'_>,
        from: &'a Paging<Heads, Head, Rows, Windows, D, Y, Chips, Rotated, Layer>,
    ) -> Result<Self, LoadError> {
        let coefficients = context.field("coefficients", "coefficients", |cx| {
            Load::load(cx, &from.coefficients)
        })?;
        Ok(Self {
            frequencies: context.field("frequencies", "frequencies", |cx| {
                Load::load(cx, &from.frequencies)
            })?,
            coefficients,
            window: context.field("window", "window", |cx| Load::load(cx, &from.window))?,
            starts: context.field("starts", "starts", |cx| Load::load(cx, &from.starts))?,
            rows: context.field("rows", "rows", |cx| Load::load(cx, &from.rows))?,
            zeros: context.field("zeros", "zeros", |cx| Load::load(cx, &from.zeros))?,
            slots: context.field("slots", "slots", |cx| Load::load(cx, &from.slots))?,
            shape: PhantomData,
        })
    }
}

/// The multi-row paged context of a step: the derive recipe's logical rows, then this geometry's
/// read form; the history is the last cache operand, the others belong to the layers.
impl<
    B: AxisName,
    K: AxisName,
    A,
    Heads,
    Head,
    Rows,
    Windows,
    D,
    X: Memory,
    Env: Copy,
    Chips: M + 'static,
    Rotated: AxisName,
    Layer,
> Module<(Live<(), Rows>, &mut (A, History<Rows, B, K, Chips>)), Env, &mut ()>
    for Preparation<Rows, Paging<Heads, Head, Rows, Windows, D, X, Chips, Rotated, Layer>, Chips, X>
where
    Heads: AxisName,
    Head: AxisName,
    Rows: AxisName,
    Windows: AxisName,
    D: Scalar,
    Derive<Rows, Chips, X>: for<'d, 'h> Module<
            (
                &'h HbmTensor<i32, Chips, m![Rows, Fields]>,
                &'h HbmTensor<i32, Chips, m![B]>,
            ),
            Env,
            &'d mut (),
            Output = Logical<Rows, Chips>,
        >,
    Paging<Heads, Head, Rows, Windows, D, X, Chips, Rotated, Layer>: for<'d, 'l, 'h> Module<
            (
                Live<(), Rows>,
                (&'l Logical<Rows, Chips>, &'h History<Rows, B, K, Chips>),
            ),
            Env,
            &'d mut (),
            Output = Context<
                Rotation<Rotated, Chips, Rows>,
                HbmTensor<i32, Chips, m![Rows]>,
                Pages<Rows, Windows, Chips>,
            >,
        >,
{
    type Output = Context<
        Rotation<Rotated, Chips, Rows>,
        HbmTensor<i32, Chips, m![Rows]>,
        Pages<Rows, Windows, Chips>,
    >;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (tiles, cache): (Live<(), Rows>, &mut (A, History<Rows, B, K, Chips>)),
    ) -> Self::Output {
        let history = &cache.1;
        let logical = self.derive.forward(
            &mut *device,
            env,
            &mut (),
            (&history.segments, &history.blocks),
        );
        self.geometry
            .forward(device, env, &mut (), (tiles, (&logical, history)))
    }
}

/// The context of a step whose attention reads its host's work: each row's rotation and slot from
/// the derive recipe's logical rows, and the work as the layers borrow it.
impl<
    'x,
    B: AxisName,
    K: AxisName,
    A,
    Heads,
    Head,
    Rows,
    Windows,
    D,
    X: Memory,
    S,
    G,
    const WAVES: usize,
    Env: Copy,
    Chips: M + 'static,
    Rotated: AxisName,
    Layer,
>
    Module<
        (
            Live<(), Rows>,
            &mut (
                A,
                (
                    History<Rows, B, K, Chips>,
                    &'x paged::Work<S, Heads, G, Head, WAVES>,
                ),
            ),
        ),
        Env,
        &mut (),
    >
    for Preparation<Rows, Paging<Heads, Head, Rows, Windows, D, X, Chips, Rotated, Layer>, Chips, X>
where
    Heads: AxisName,
    Head: AxisName,
    Rows: AxisName,
    Windows: AxisName,
    D: Scalar,
    S: AxisName,
    G: AxisName,
    Derive<Rows, Chips, X>: for<'d, 'h> Module<
            (
                &'h HbmTensor<i32, Chips, m![Rows, Fields]>,
                &'h HbmTensor<i32, Chips, m![B]>,
            ),
            Env,
            &'d mut (),
            Output = Logical<Rows, Chips>,
        >,
    X::Tensor<f32, Chips, m![Rows, Rotated]>: Borrow<HbmTensor<f32, Chips, m![Rows, Rotated]>>,
    X::Tensor<f32, Chips, m![Rotated]>: Borrow<HbmTensor<f32, Chips, m![Rotated]>>,
{
    type Output = Context<
        Rotation<Rotated, Chips, Rows>,
        HbmTensor<i32, Chips, m![Rows]>,
        paged::Context<'x, S, Heads, G, Head, WAVES, Rows>,
    >;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (_, cache): (
            Live<(), Rows>,
            &mut (
                A,
                (
                    History<Rows, B, K, Chips>,
                    &'x paged::Work<S, Heads, G, Head, WAVES>,
                ),
            ),
        ),
    ) -> Self::Output {
        let (history, work) = &cache.1;
        let logical = self.derive.forward(
            &mut *device,
            env,
            &mut (),
            (&history.segments, &history.blocks),
        );
        let geometry = &self.geometry;
        let frequencies = geometry.frequencies.borrow();
        let positions =
            table::positions::<Rows, Rotated, _>(device, frequencies, &logical.positions);
        Context {
            positions: geometry.coefficients.forward(
                &mut *device,
                env,
                &mut (),
                (frequencies, &positions),
            ),
            write: table::writes::<Rows, _>(
                device,
                &logical.slots,
                const { Blocks::<Heads, Head, D, Chips>::ROW_STRIDE },
            ),
            read: work.read::<Rows>(),
        }
    }
}

impl<
    G: AxisName,
    B: AxisName,
    K: AxisName,
    Heads,
    Head,
    Rows,
    Windows,
    D,
    X: Memory,
    Env: Copy,
    Chips: M + 'static,
    Rotated: AxisName,
    Layer,
>
    Module<
        (
            Live<(), Rows>,
            (&Logical<Rows, Chips>, &History<G, B, K, Chips>),
        ),
        Env,
        &mut (),
    > for Paging<Heads, Head, Rows, Windows, D, X, Chips, Rotated, Layer>
where
    Heads: AxisName,
    Head: AxisName,
    Rows: AxisName,
    Windows: AxisName,
    D: Scalar,
    X::Tensor<f32, Chips, m![Rows, Rotated]>: Borrow<HbmTensor<f32, Chips, m![Rows, Rotated]>>,
    X::Tensor<f32, Chips, m![Rotated]>: Borrow<HbmTensor<f32, Chips, m![Rotated]>>,
    X::Tensor<i32, Chips, m![1]>: Borrow<Value<Chips>>,
    X::Tensor<i32, Chips, m![Windows, { Packets<Rows> }]>:
        Borrow<HbmTensor<i32, Chips, m![Windows, { Packets<Rows> }]>>,
    X::Tensor<i32, Chips, m![Rows]>: Borrow<HbmTensor<i32, Chips, m![Rows]>>,
    X::Tensor<f32, Chips, m![Rows, Cache]>: Borrow<HbmTensor<f32, Chips, m![Rows, Cache]>>,
{
    type Output = Context<
        Rotation<Rotated, Chips, Rows>,
        HbmTensor<i32, Chips, m![Rows]>,
        Pages<Rows, Windows, Chips>,
    >;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (tiles, (logical, history)): (
            Live<(), Rows>,
            (&Logical<Rows, Chips>, &History<G, B, K, Chips>),
        ),
    ) -> Self::Output {
        let (starts, frequencies) = (self.starts.borrow(), self.frequencies.borrow());
        let length = table::live(device, self.zeros.borrow(), tiles, &logical.lengths);
        let extents = Extents::new(device, (self.window.borrow(), starts), &length);
        let blocks = extents.table(
            device,
            (self.rows.borrow(), &history.blocks),
            &logical.tables,
            const { Blocks::<Heads, Head, D, Chips>::BLOCK_STRIDE },
        );
        let bias = extents.bias(device, (starts, self.slots.borrow()));
        let positions =
            table::positions::<Rows, Rotated, _>(device, frequencies, &logical.positions);
        Context {
            positions: self.coefficients.forward(
                &mut *device,
                env,
                &mut (),
                (frequencies, &positions),
            ),
            write: table::writes::<Rows, _>(
                device,
                &logical.slots,
                const { Blocks::<Heads, Head, D, Chips>::ROW_STRIDE },
            ),
            read: Pages {
                table: blocks,
                lengths: extents.counts,
                bias,
                rest: extents.rest,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::config::scaling::Spectrum;
    use crate::models::decoder::attention::LayerType;
    use crate::models::decoder::attention::{self, Chunks, KvHeads, qwen3};
    use crate::specialize::attention::Decode;
    use crate::tensor::axis::Eight;
    use crate::tensor::parallel::{Parallelism, Replicated};
    use crate::tests::DEVICE;

    axes![Windows = 72, Heads = 8, Head = 128];

    type Constants<C> = Paging<Heads, Head, Eight, Windows, bf16, Hbm, C>;

    /// Qwen3-0.6B's rotation and paged-read constants.
    type Table = Paging<KvHeads, attention::Head, <qwen3::Size<"0.6B"> as Decode>::Rows, Chunks>;

    struct Host;

    struct Leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
        HostTensor<D, m![{ C }, { S }]>,
    );

    impl Memory for Host {
        type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Leaf<D, C, S>;
    }

    impl<D: MaterializableScalar, C: M + 'static, S: M + 'static>
        Load<HostTensor<D, m![{ C }, { S }]>> for Leaf<D, C, S>
    {
        fn load(
            _: &mut load::Context<'_>,
            from: HostTensor<D, m![{ C }, { S }]>,
        ) -> Result<Self, LoadError> {
            Ok(Self(from))
        }
    }

    struct Layer<const NAME: &'static str>;

    impl<const NAME: &'static str> LayerType for Layer<NAME> {
        const NAME: Option<&'static str> = Some(NAME);
    }

    /// Scoped layer types bind their own window and the frequencies of their rotated coordinates.
    #[test]
    fn binds_layer_constants() {
        axes![Sliding = 256, Full = 512, Rotated = 128];
        type SlidingPaging = Paging<
            Heads,
            Sliding,
            Eight,
            Windows,
            bf16,
            Host,
            Chip,
            Sliding,
            Layer<"sliding_attention">,
        >;
        type FullPaging =
            Paging<Heads, Full, Eight, Windows, bf16, Host, Chip, Rotated, Layer<"full_attention">>;
        let sliding = serde_json::json!({"rope_type": "default", "rope_theta": 1e4});
        let full = serde_json::json!({
            "rope_type": "proportional", "rope_theta": 1e6, "partial_rotary_factor": 0.25,
        });
        let config = serde_json::json!({"text_config": {
            "rope_parameters": {"sliding_attention": sliding, "full_attention": full},
            "sliding_window": 1024, "max_position_embeddings": 262144,
            "head_dim": 256, "global_head_dim": 512,
        }});
        let empty = tempfile::tempdir().unwrap();
        let checkpoint = Checkpoint::new(empty.path(), Some(config)).unwrap();
        let mut cx = load::Context::new(None);
        let (sliding_paging, full_paging): (SlidingPaging, FullPaging) = cx
            .config("text_config", |cx| {
                cx.field("rotary_emb", "rotary_emb", |cx| {
                    Ok::<_, LoadError>((
                        <SlidingPaging as Load<_>>::load(cx, &checkpoint)?,
                        <FullPaging as Load<_>>::load(cx, &checkpoint)?,
                    ))
                })
            })
            .unwrap();
        let sliding: Spectrum =
            serde_json::from_value(serde_json::json!({"rope_parameters": sliding})).unwrap();
        let full: Spectrum =
            serde_json::from_value(serde_json::json!({"rope_parameters": full})).unwrap();
        assert_eq!(sliding_paging.window.0.into_vec(), [1024]);
        assert_eq!(full_paging.window.0.into_vec(), [36864]);
        assert_eq!(
            sliding_paging.frequencies.0.into_vec(),
            sliding
                .bind::<Sliding, Sliding>()
                .unwrap()
                .coordinates()
                .repeat(Eight::SIZE),
        );
        assert_eq!(
            full_paging.frequencies.0.into_vec(),
            full.bind::<Full, Rotated>()
                .unwrap()
                .coordinates()
                .repeat(Eight::SIZE),
        );
    }

    /// Each chip's copy of every constant of `paging`, as bytes, in field order.
    async fn copies<C: M + 'static>(
        device: &mut Device,
        paging: &Constants<C>,
    ) -> Vec<Vec<Vec<u8>>> {
        async fn chips<D: MaterializableScalar, C: M, S: M>(
            device: &mut Device,
            leaf: &HbmTensor<D, C, S>,
        ) -> Vec<Vec<u8>> {
            let values = leaf.to_host::<m![{ C }, { S }]>(&mut device.pdma).await;
            let values = values.unwrap().into_vec();
            values
                .chunks_exact(values.len() / C::SIZE)
                .map(D::to_buf)
                .collect()
        }
        let [low_phase, low_sign, high_phase, high_sign, amplitude] =
            paging.coefficients.constants();
        vec![
            chips(device, &paging.frequencies).await,
            chips(device, low_phase).await,
            chips(device, low_sign).await,
            chips(device, high_phase).await,
            chips(device, high_sign).await,
            chips(device, amplitude).await,
            chips(device, &paging.window).await,
            chips(device, &paging.starts).await,
            chips(device, &paging.rows).await,
            chips(device, &paging.zeros).await,
            chips(device, &paging.slots).await,
        ]
    }

    /// A load of the paged read form's constants from a config on two chips gives each chip the
    /// constants a one-chip load computes.
    #[tokio::test(flavor = "current_thread")]
    async fn copies_the_paged_constants_to_every_chip() {
        type Two = <Replicated<2> as Parallelism>::Chip;
        let _lease = DEVICE.lock().await;
        let empty = tempfile::tempdir().unwrap();
        let config =
            serde_json::json!({"rope_theta": 1_000_000.0, "max_position_embeddings": 40960});
        let checkpoint = Checkpoint::new(empty.path(), Some(config)).unwrap();
        let one = {
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let paging: Constants<Chip> = checkpoint.load(&mut device).unwrap();
            copies(&mut device, &paging).await
        };
        let two = {
            let mut device = Device::new(Topology { chips: 2, pes: 8 }).unwrap();
            let paging: Constants<Two> = checkpoint.load(&mut device).unwrap();
            copies(&mut device, &paging).await
        };
        for (field, (one, two)) in one.iter().zip(&two).enumerate() {
            assert_eq!(two.len(), 2, "field {field}: a copy a chip");
            assert!(two.iter().all(|copy| copy == &one[0]), "field {field}");
        }
    }

    /// The constants refuse a config whose `rope_theta` is absent, no number or not positive, or
    /// whose trained context is absent or holds no position, each under `config.json` by its reason.
    #[test]
    fn refuses_configs_without_a_rotation_or_a_context() {
        use serde_json::json;
        let directory = tempfile::tempdir().unwrap();
        let refusal = |config: &serde_json::Value| {
            let source = Checkpoint::new(directory.path(), Some(config.clone())).unwrap();
            match <Table as Load<&Checkpoint>>::load(&mut load::Context::new(None), &source) {
                Err(LoadError::InvalidData { name, reason }) => (name, reason),
                other => panic!("{config}: {:?}", other.err()),
            }
        };
        let cases = [
            (
                json!({"max_position_embeddings": 40960}),
                "missing field `rope_theta`",
            ),
            (
                json!({"rope_theta": 1e6}),
                "missing field `max_position_embeddings`",
            ),
            (
                json!({"rope_theta": "1e6", "max_position_embeddings": 40960}),
                r#"rope_theta: invalid type: string "1e6", expected f32"#,
            ),
            (
                json!({"rope_theta": 0, "max_position_embeddings": 40960}),
                "rope_theta: found 0, expected a positive finite value",
            ),
            (
                json!({"rope_theta": 1e6, "max_position_embeddings": 0}),
                "max_position_embeddings: invalid value: integer `0`, expected a nonzero usize",
            ),
        ];
        let refused: Vec<_> = cases.iter().map(|(config, _)| refusal(config)).collect();
        let named = |(_, reason): &(_, &str)| ("config.json".to_owned(), reason.to_string());
        assert_eq!(refused, cases.iter().map(named).collect::<Vec<_>>());
    }
}
