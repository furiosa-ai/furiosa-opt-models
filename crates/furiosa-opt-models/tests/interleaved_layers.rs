//! The Gemma 4 catalog's layers interleaved in two collections over a call of rows that attend a
//! history in their layers' caches: sliding layers in both collections or sliding and full layers,
//! one of each a period, and the catalog's own period of five sliding layers and a full one; the
//! full layers' attention leaf alone; and the feed-forward's NVFP4 projection leaves, under the
//! early and the summed contract.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::load::{Checkpoint, Load, Named};
use furiosa_opt_models::models::decoder::attention::gemma4::{self, full, sliding};
use furiosa_opt_models::models::decoder::form::summed;
use furiosa_opt_models::models::decoder::{self, Attention};
use furiosa_opt_models::modules::{attention, language};
use furiosa_opt_models::ops::attention::gqa::{
    Attended, Cache, LaneWindows, Pages, RowMajor, WindowRow, WindowRows, Wrap,
};
use furiosa_opt_models::ops::index::table::{BlockIndex, PADDING_BIAS};
use furiosa_opt_models::ops::normalization::Spread;
use furiosa_opt_models::ops::rotary::{PairFrequencies, Rotation};
use furiosa_opt_models::schedule::Decoder;
use furiosa_opt_models::schedule::decoder::Places;
use furiosa_opt_models::tensor::axis::Lanes;
use furiosa_opt_models::tensor::cache::{Block, Blocks, KeyValue};
use furiosa_opt_models::tensor::form::BlockScaled;
use furiosa_opt_models::tensor::form::nvfp4::{Early, Features, Scale, Scales, Summed, W4A16};
use furiosa_opt_models::tensor::live::Live;
use furiosa_opt_models::tensor::memory::Hbm;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_models::tensor::scalar::{Copies, Value};
use furiosa_opt_models::{Branch, Interleave, Module, Op};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::{HbmBuffer, launch};
use safetensors::Dtype;
use std::ops::Range;
use tokio::sync::Mutex;

/// The device tests share one chip, so they run one at a time.
static DEVICE: Mutex<()> = Mutex::const_new(());

type Rows = gemma4::Rows;
type Hidden = gemma4::Hidden;
type Intermediate = gemma4::Intermediate;

/// The layer types of the catalog's two collections.
trait Collections {
    type Sliding;
    type Full;
    type Norm;
}

impl<E, S, F, N, H> Collections
    for language::Hybrid<E, [S; gemma4::SLIDING], [F; gemma4::FULL], N, H, { gemma4::ORDER }>
{
    type Sliding = S;
    type Full = F;
    type Norm = N;
}

type Catalog = gemma4::Catalog<gemma4::Quantized, bf16, Hbm>;
type Sliding = <Catalog as Collections>::Sliding;
type Full = <Catalog as Collections>::Full;
/// The final norm after the layers.
type FinalNorm = Named<"norm", <Catalog as Collections>::Norm>;

type Group = sliding::Group;

/// One layer of each collection a period, so every other edge crosses a period.
const PAIRS: &[Branch] = &[Branch::First, Branch::Second, Branch::First, Branch::Second];

/// Two sliding layers in the first collection and two layers of group `B` in the second,
/// interleaved.
type Interleaved<B> = Interleave<([Sliding; 2], [<B as Kind>::Layer; 2]), PAIRS>;

/// Two collections of two sliding layers each, interleaved.
type Layers = Interleaved<Group>;

/// Two sliding and two full layers, interleaved: sliding to full, full to sliding across a period
/// and full to the final norm.
type Mixed = Interleaved<full::Group>;

/// The catalog's period: five sliding layers, then a full one.
const PERIOD: &[Branch] = &Branch::periodic::<6>(6);

/// One period of the catalog's order.
type Period = Interleave<([Sliding; 5], [Full; 1]), PERIOD>;

/// A call's context of group `A`: its rows' rotation, write slots and paged reads.
type Context<A> = attention::Context<
    Rotation<<A as Attention>::Rotated, Chip, Rows>,
    HbmTensor<i32, Chip, m![Rows]>,
    Pages<Rows, <A as Attention>::Windows, Chip>,
>;

type Pool<A> = Blocks<<A as Attention>::Heads, <A as Attention>::Head, <A as Attention>::Cache>;
type Pools<B> = ([Pool<Group>; 2], [Pool<B>; 2]);
type PeriodPools = ([Pool<Group>; 5], [Pool<full::Group>; 1]);

/// The call's hidden rows.
type HiddenRows = HbmTensor<bf16, Chip, m![Rows, Hidden]>;

/// A layer's state: its attention's slots, its cache blocks, its MLP's slots.
type Qkv = (((), (), ((), ())), (), ());
type Units = ((((), ()), ()), ((), ()));
type State<'b, A> = ((((Qkv, &'b mut Pool<A>, ()), ()), (Units, ())), ());

fn state<A: Attention>(pool: &mut Pool<A>) -> State<'_, A> {
    let qkv = (((), (), ((), ())), (), ());
    let units = ((((), ()), ()), ((), ()));
    ((((qkv, pool, ()), ()), (units, ())), ())
}

/// A group whose layers these tests build and run as the second collection of [`Interleaved`].
trait Kind: Attention<Rotated: AxisName, Cache = bf16> {
    /// The catalog's layer of the group.
    type Layer: for<'c> Load<&'c Checkpoint>;
    /// The attention projections a layer's checkpoint stores: a full layer's values are its
    /// keys' projection.
    const PROJECTIONS: &'static [&'static str];

    /// Runs the layers in order or by the decoder schedule, then the final norm.
    async fn forward(
        device: &mut Device,
        in_order: bool,
        model: (&Interleaved<Self>, &FinalNorm),
        contexts: (&Context<Group>, &Context<Self>),
        pools: &mut Pools<Self>,
        input: (&HiddenRows, &Value),
    ) -> HiddenRows;
}

impl Kind for sliding::Group {
    type Layer = Sliding;
    const PROJECTIONS: &'static [&'static str] = &["q_proj", "k_proj", "v_proj", "o_proj"];

    async fn forward(
        device: &mut Device,
        in_order: bool,
        model: (&Layers, &FinalNorm),
        (context, _): (&Context<Group>, &Context<Group>),
        pools: &mut Pools<Group>,
        input: (&HiddenRows, &Value),
    ) -> HiddenRows {
        let arguments = (&mut *device, model, context, pools, input);
        let output = match in_order {
            true => launch(sequential, arguments).await,
            false => launch(scheduled, arguments).await,
        };
        output.unwrap()
    }
}

impl Kind for full::Group {
    type Layer = Full;
    const PROJECTIONS: &'static [&'static str] = &["q_proj", "k_proj", "o_proj"];

    async fn forward(
        device: &mut Device,
        in_order: bool,
        model: (&Mixed, &FinalNorm),
        contexts: (&Context<Group>, &Context<full::Group>),
        pools: &mut Pools<full::Group>,
        input: (&HiddenRows, &Value),
    ) -> HiddenRows {
        let arguments = (&mut *device, model, contexts, pools, input);
        let output = match in_order {
            true => launch(sequential_mixed, arguments).await,
            false => launch(scheduled_mixed, arguments).await,
        };
        output.unwrap()
    }
}

/// The layers' sequential forward over the call's live rows, each layer on its own cache, then
/// the final norm.
#[device(chip = 1)]
fn sequential(
    device: &mut Device,
    (layers, norm): (&Layers, &FinalNorm),
    context: &Context<Group>,
    pools: &mut Pools<Group>,
    (hidden, live): (&HiddenRows, &Value),
) -> HiddenRows {
    let live = Live::<(), Rows>::read(live);
    let rows: Live<Spread<Hidden, Rows, Chip>, Rows> = live.with(hidden.to_dm(&mut device.tdma));
    let ([a, b], [c, d]) = pools;
    let mut states = (
        [state::<Group>(a), state::<Group>(b)],
        [state::<Group>(c), state::<Group>(d)],
    );
    let rows: Live<Spread<Hidden, Rows, Chip>, Rows> =
        Module::forward(layers, &mut *device, (context, context), &mut states, rows);
    let rows: Live<Spread<Hidden, Rows, Chip>, Rows> =
        Module::forward(norm, &mut *device, (), &mut (), &rows);
    rows.value().to_hbm(&mut device.tdma)
}

/// The same layers and norm run by the decoder schedule, the call's rows in place in their home.
#[device(chip = 1)]
fn scheduled(
    device: &mut Device,
    (layers, norm): (&Layers, &FinalNorm),
    context: &Context<Group>,
    pools: &mut Pools<Group>,
    (hidden, live): (&HiddenRows, &Value),
) -> HiddenRows {
    let live = Live::<(), Rows>::read(live);
    let mut home: HiddenRows = HbmTensor::new();
    hidden.view().to_hbm_view(&mut device.tdma, home.view_mut());
    let ([a, b], [c, d]) = pools;
    let mut state = (
        (Places::<Group, Rows>::new(), [(a, ()), (b, ())]),
        (Places::<Group, Rows>::new(), [(c, ()), (d, ())]),
    );
    let schedule = Decoder((layers, norm));
    Module::forward(
        &schedule,
        &mut *device,
        (context, context),
        &mut state,
        (&mut home, live),
    );
    home
}

/// The sliding and full layers' sequential forward over the call's live rows, each layer on its
/// own cache and its group's context, then the final norm.
#[device(chip = 1)]
fn sequential_mixed(
    device: &mut Device,
    (layers, norm): (&Mixed, &FinalNorm),
    contexts: (&Context<Group>, &Context<full::Group>),
    pools: &mut Pools<full::Group>,
    (hidden, live): (&HiddenRows, &Value),
) -> HiddenRows {
    let live = Live::<(), Rows>::read(live);
    let rows: Live<Spread<Hidden, Rows, Chip>, Rows> = live.with(hidden.to_dm(&mut device.tdma));
    let ([a, b], [c, d]) = pools;
    let mut states = (
        [state::<Group>(a), state::<Group>(b)],
        [state::<full::Group>(c), state::<full::Group>(d)],
    );
    let rows: Live<Spread<Hidden, Rows, Chip>, Rows> =
        Module::forward(layers, &mut *device, contexts, &mut states, rows);
    let rows: Live<Spread<Hidden, Rows, Chip>, Rows> =
        Module::forward(norm, &mut *device, (), &mut (), &rows);
    rows.value().to_hbm(&mut device.tdma)
}

/// The same sliding and full layers and norm run by the decoder schedule.
#[device(chip = 1)]
fn scheduled_mixed(
    device: &mut Device,
    (layers, norm): (&Mixed, &FinalNorm),
    contexts: (&Context<Group>, &Context<full::Group>),
    pools: &mut Pools<full::Group>,
    (hidden, live): (&HiddenRows, &Value),
) -> HiddenRows {
    let live = Live::<(), Rows>::read(live);
    let mut home: HiddenRows = HbmTensor::new();
    hidden.view().to_hbm_view(&mut device.tdma, home.view_mut());
    let ([a, b], [c, d]) = pools;
    let mut state = (
        (Places::<Group, Rows>::new(), [(a, ()), (b, ())]),
        (Places::<full::Group, Rows>::new(), [(c, ()), (d, ())]),
    );
    let schedule = Decoder((layers, norm));
    Module::forward(
        &schedule,
        &mut *device,
        contexts,
        &mut state,
        (&mut home, live),
    );
    home
}

/// The period's sequential forward over the call's live rows, then the final norm.
#[device(chip = 1)]
fn sequential_period(
    device: &mut Device,
    (layers, norm): (&Period, &FinalNorm),
    contexts: (&Context<Group>, &Context<full::Group>),
    pools: &mut PeriodPools,
    (hidden, live): (&HiddenRows, &Value),
) -> HiddenRows {
    let live = Live::<(), Rows>::read(live);
    let rows: Live<Spread<Hidden, Rows, Chip>, Rows> = live.with(hidden.to_dm(&mut device.tdma));
    let ([a, b, c, d, e], [f]) = pools;
    let mut states = (
        [
            state::<Group>(a),
            state::<Group>(b),
            state::<Group>(c),
            state::<Group>(d),
            state::<Group>(e),
        ],
        [state::<full::Group>(f)],
    );
    let rows: Live<Spread<Hidden, Rows, Chip>, Rows> =
        Module::forward(layers, &mut *device, contexts, &mut states, rows);
    let rows: Live<Spread<Hidden, Rows, Chip>, Rows> =
        Module::forward(norm, &mut *device, (), &mut (), &rows);
    rows.value().to_hbm(&mut device.tdma)
}

/// The same period and norm run by the decoder schedule.
#[device(chip = 1)]
fn scheduled_period(
    device: &mut Device,
    (layers, norm): (&Period, &FinalNorm),
    contexts: (&Context<Group>, &Context<full::Group>),
    pools: &mut PeriodPools,
    (hidden, live): (&HiddenRows, &Value),
) -> HiddenRows {
    let live = Live::<(), Rows>::read(live);
    let mut home: HiddenRows = HbmTensor::new();
    hidden.view().to_hbm_view(&mut device.tdma, home.view_mut());
    let ([a, b, c, d, e], [f]) = pools;
    let sliding = [(a, ()), (b, ()), (c, ()), (d, ()), (e, ())];
    let mut state = (
        (Places::<Group, Rows>::new(), sliding),
        (Places::<full::Group, Rows>::new(), [(f, ())]),
    );
    let schedule = Decoder((layers, norm));
    Module::forward(
        &schedule,
        &mut *device,
        contexts,
        &mut state,
        (&mut home, live),
    );
    home
}

/// The catalog's layers caching each key and value as an E4M3 code under its layer's scales.
type CodedCatalog = gemma4::Catalog<gemma4::Quantized, f8e4m3, Hbm>;
type CodedGroup = sliding::Group<f8e4m3>;
type CodedFull = full::Group<f8e4m3>;
/// The catalog's period over E4M3 caches.
type CodedPeriod = Interleave<
    (
        [<CodedCatalog as Collections>::Sliding; 5],
        [<CodedCatalog as Collections>::Full; 1],
    ),
    PERIOD,
>;
type CodedPools = ([Pool<CodedGroup>; 5], [Pool<CodedFull>; 1]);

/// The period over E4M3 caches run by the decoder schedule.
#[device(chip = 1)]
fn scheduled_coded(
    device: &mut Device,
    (layers, norm): (&CodedPeriod, &FinalNorm),
    contexts: (&Context<CodedGroup>, &Context<CodedFull>),
    pools: &mut CodedPools,
    (hidden, live): (&HiddenRows, &Value),
) -> HiddenRows {
    let live = Live::<(), Rows>::read(live);
    let mut home: HiddenRows = HbmTensor::new();
    hidden.view().to_hbm_view(&mut device.tdma, home.view_mut());
    let ([a, b, c, d, e], [f]) = pools;
    let sliding = [(a, ()), (b, ()), (c, ()), (d, ()), (e, ())];
    let mut state = (
        (Places::<CodedGroup, Rows>::new(), sliding),
        (Places::<CodedFull, Rows>::new(), [(f, ())]),
    );
    let schedule = Decoder((layers, norm));
    Module::forward(
        &schedule,
        &mut *device,
        contexts,
        &mut state,
        (&mut home, live),
    );
    home
}

/// The scale of every coded layer's keys and of its values: their codes saturate past 3.5, which
/// some of a layer's normalized keys and values reach.
const CODED: f32 = 1.0 / 128.0;

/// Layer `layer`'s checkpoint of group `A` with the static cache scales an E4M3 cache reads.
fn coded_checkpoint<A: Kind>(layer: usize) -> Checkpoint {
    let mut checkpoint = checkpoint::<A>(layer);
    for name in ["self_attn.k_scale", "self_attn.v_scale"] {
        let scale = bf16::from_f32(CODED).to_bits().to_le_bytes().to_vec();
        checkpoint.insert(name, Dtype::BF16, &[1], scale).unwrap();
    }
    checkpoint
}

/// A value as an E4M3 cache codes it under `scale`: divided, saturated, to nearest even.
fn code(value: f32, scale: f32) -> f8e4m3 {
    f8e4m3::from_f32((value / scale).clamp(-448.0, 448.0))
}

/// `values` coded in a new E4M3 cache of [`blocks`] blocks.
async fn coded_pool<A: Attention<Cache = f8e4m3>>(device: &mut Device, values: &[bf16]) -> Pool<A> {
    let bytes = blocks() * Pool::<A>::BLOCK_STRIDE as usize;
    let mut pool = Pool::<A>::from_buffer(HbmBuffer::alloc(device, bytes).unwrap()).unwrap();
    let codes: Vec<f8e4m3> = values
        .iter()
        .map(|value| code(value.to_f32(), CODED))
        .collect();
    for (block, codes) in codes.chunks_exact(codes.len() / blocks()).enumerate() {
        HostTensor::<f8e4m3, Tokens<A>>::from_vec(codes.to_vec())
            .to_hbm(&mut device.pdma)
            .output(&mut pool.tokens[block])
            .await
            .unwrap();
    }
    pool
}

/// Every code of `pool`, block by block.
async fn codes<A: Attention<Cache = f8e4m3>>(device: &mut Device, pool: &Pool<A>) -> Vec<f8e4m3> {
    let mut codes = Vec::new();
    for block in 0..blocks() {
        let tokens = pool.tokens[block]
            .to_host::<Tokens<A>>(&mut device.pdma)
            .await;
        codes.extend(tokens.unwrap().into_vec());
    }
    codes
}

/// The period's sliding layers and its full layer over E4M3 caches, each its own weights.
fn coded_period(device: &mut Device) -> CodedPeriod {
    let sliding =
        [0, 1, 2, 3, 4].map(|at| coded_checkpoint::<Group>(at).load(&mut *device).unwrap());
    let full = [coded_checkpoint::<full::Group>(5).load(device).unwrap()];
    Interleave {
        layers: (sliding, full),
    }
}

/// The call's `live` rows through the period over E4M3 caches coded from `caches`: the live rows'
/// outputs and the first layer's codes after it.
async fn run_coded(
    device: &mut Device,
    model: (&CodedPeriod, &FinalNorm),
    caches: &[Vec<bf16>; 6],
    live: usize,
) -> (Vec<bf16>, Vec<f8e4m3>) {
    let contexts = (
        context::<CodedGroup>(device, live).await,
        context::<CodedFull>(device, live).await,
    );
    let mut pools = (
        [
            coded_pool::<CodedGroup>(device, &caches[0]).await,
            coded_pool::<CodedGroup>(device, &caches[1]).await,
            coded_pool::<CodedGroup>(device, &caches[2]).await,
            coded_pool::<CodedGroup>(device, &caches[3]).await,
            coded_pool::<CodedGroup>(device, &caches[4]).await,
        ],
        [coded_pool::<CodedFull>(device, &caches[5]).await],
    );
    let (input, count) = input(device, live).await;
    let arguments = (
        &mut *device,
        model,
        (&contexts.0, &contexts.1),
        &mut pools,
        (&input, &count),
    );
    let output = launch(scheduled_coded, arguments).await.unwrap();
    let output = rows(device, &output, live).await;
    let first = codes::<CodedGroup>(device, &pools.0[0]).await;
    (output, first)
}

/// The largest and the mean distance of `found` from `want`, over the largest magnitude of `want`.
fn deviation(found: &[bf16], want: &[bf16]) -> (f64, f64) {
    let distances: Vec<f64> = (found.iter().zip(want))
        .map(|(found, want)| (f64::from(found.to_f32()) - f64::from(want.to_f32())).abs())
        .collect();
    let largest = (want.iter()).fold(0.0f64, |top, want| top.max(f64::from(want.to_f32()).abs()));
    let worst = distances.iter().copied().fold(0.0, f64::max);
    let mean = distances.iter().sum::<f64>() / distances.len() as f64;
    (worst / largest, mean / largest)
}

/// The catalog's layers in the published order, over caches of `C`.
type Order<C> = Interleave<
    (
        [<gemma4::Catalog<gemma4::Quantized, C, Hbm> as Collections>::Sliding; gemma4::SLIDING],
        [<gemma4::Catalog<gemma4::Quantized, C, Hbm> as Collections>::Full; gemma4::FULL],
    ),
    { gemma4::ORDER },
>;

/// Every layer's cache of `C`, the sliding layers' then the full ones', in their order.
type Caches<C> = (
    [Pool<sliding::Group<C>>; gemma4::SLIDING],
    [Pool<full::Group<C>>; gemma4::FULL],
);

/// The published layers and final norm over caches of `$cache`, run by the decoder schedule: the
/// call's rows after the final norm.
macro_rules! decoded {
    ($name:ident, $cache:ty) => {
        #[device(chip = 1)]
        fn $name(
            device: &mut Device,
            (layers, norm): (&Order<$cache>, &FinalNorm),
            contexts: (
                &Context<sliding::Group<$cache>>,
                &Context<full::Group<$cache>>,
            ),
            pools: &mut Caches<$cache>,
            (hidden, live): (&HiddenRows, &Value),
        ) -> HiddenRows {
            let live = Live::<(), Rows>::read(live);
            let mut home: HiddenRows = HbmTensor::new();
            hidden.view().to_hbm_view(&mut device.tdma, home.view_mut());
            let (sliding, full) = pools;
            let mut state = (
                (
                    Places::<sliding::Group<$cache>, Rows>::new(),
                    sliding.each_mut().map(|pool| (pool, ())),
                ),
                (
                    Places::<full::Group<$cache>, Rows>::new(),
                    full.each_mut().map(|pool| (pool, ())),
                ),
            );
            let schedule = Decoder((layers, norm));
            Module::forward(
                &schedule,
                &mut *device,
                contexts,
                &mut state,
                (&mut home, live),
            );
            home
        }
    };
}

decoded!(decoded_bf16, bf16);
decoded!(decoded_e4m3, f8e4m3);

/// Tokens of history before the call's first row: `PERIOD_PROFILE_HISTORY`, or 100, past a block
/// boundary.
fn history() -> usize {
    std::env::var("PERIOD_PROFILE_HISTORY").map_or(100, |tokens| tokens.parse().unwrap())
}

/// Physical blocks of each layer's cache: the history and every row of the call.
fn blocks() -> usize {
    (history() + Rows::SIZE).div_ceil(Block::SIZE) + 1
}

/// A value nothing reads: a NaN with a payload of its own.
const POISON: u16 = 0xff81;

/// Pseudo-random bytes from `state`.
fn bytes(count: usize, mut state: u64) -> Vec<u8> {
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 29) as u8
        })
        .collect()
}

/// Pseudo-random BF16 values in `[-bound, bound)`.
fn values(count: usize, state: u64, bound: f32) -> Vec<bf16> {
    let bytes = bytes(2 * count, state);
    (bytes.chunks_exact(2))
        .map(|pair| {
            let unit = u16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0 - 1.0;
            bf16::from_f32(unit * bound)
        })
        .collect()
}

/// The checkpoint of layer `layer` of group `A`: every weight its tree reads, under its names,
/// each layer's own.
fn checkpoint<A: Kind>(layer: usize) -> Checkpoint {
    let seed = |tensor: u64| 0x9e37_79b9_7f4a_7c15 ^ (layer as u64) << 40 ^ tensor << 8;
    let mut checkpoint = Checkpoint::default();
    let mut insert = |name: &str, dtype: Dtype, shape: &[usize], data: Vec<u8>| {
        checkpoint.insert(name, dtype, shape, data).unwrap();
    };
    let gains = |width: usize, tensor: u64| -> Vec<u8> {
        (0..width)
            .flat_map(|channel| {
                let gain =
                    0.5 + ((channel as u64 * 7 + tensor * 3 + layer as u64) % 9) as f32 / 8.0;
                bf16::from_f32(gain).to_bits().to_le_bytes()
            })
            .collect()
    };
    let norms = [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ];
    for (tensor, norm) in norms.into_iter().enumerate() {
        insert(
            &format!("{norm}.weight"),
            Dtype::BF16,
            &[Hidden::SIZE],
            gains(Hidden::SIZE, tensor as u64),
        );
    }
    let head = <A as Attention>::Head::SIZE;
    for (tensor, norm) in ["q_norm", "k_norm"].into_iter().enumerate() {
        let name = format!("self_attn.{norm}.weight");
        insert(&name, Dtype::BF16, &[head], gains(head, 4 + tensor as u64));
    }
    let query = <A as Attention>::QueryWidth::SIZE;
    let kv = <A as Attention>::KvWidth::SIZE;
    let channels = [
        ("q_proj", query, Hidden::SIZE),
        ("k_proj", kv, Hidden::SIZE),
        ("v_proj", kv, Hidden::SIZE),
        ("o_proj", Hidden::SIZE, query),
    ];
    let stored = channels.into_iter().enumerate();
    for (tensor, (name, rows, columns)) in
        stored.filter(|(_, (name, ..))| A::PROJECTIONS.contains(name))
    {
        // E4M3 codes below two in magnitude: the exponent field stays below eight.
        let codes = bytes(rows * columns, seed(tensor as u64))
            .into_iter()
            .map(|code| code & 0xbf);
        let name = format!("self_attn.{name}");
        insert(
            &format!("{name}.weight"),
            Dtype::F8_E4M3,
            &[rows, columns],
            codes.collect(),
        );
        let scales = (0..rows).flat_map(|row| (0.01 + (row % 17) as f32 * 0.003).to_le_bytes());
        insert(
            &format!("{name}.weight_scale"),
            Dtype::F32,
            &[rows, 1],
            scales.collect(),
        );
    }
    let blocks = [
        ("gate_proj", Intermediate::SIZE, Hidden::SIZE),
        ("up_proj", Intermediate::SIZE, Hidden::SIZE),
        ("down_proj", Hidden::SIZE, Intermediate::SIZE),
    ];
    for (tensor, (name, rows, columns)) in blocks.into_iter().enumerate() {
        let name = format!("mlp.{name}");
        let packed = bytes(rows * columns / 2, seed(8 + tensor as u64));
        insert(
            &format!("{name}.weight_packed"),
            Dtype::U8,
            &[rows, columns / 2],
            packed,
        );
        // Block scales of 0.5, 0.75, 1 and 1.5.
        let scales = bytes(rows * columns / 16, seed(12 + tensor as u64));
        let scales = scales
            .into_iter()
            .map(|at| [0x30, 0x34, 0x38, 0x3c][at as usize % 4]);
        let shape = [rows, columns / 16];
        insert(
            &format!("{name}.weight_scale"),
            Dtype::F8_E4M3,
            &shape,
            scales.collect(),
        );
        let global = 16.0_f32.to_le_bytes().to_vec();
        insert(
            &format!("{name}.weight_global_scale"),
            Dtype::F32,
            &[1],
            global,
        );
    }
    let scalar = bf16::from_f32(0.75 + 0.125 * layer as f32).to_bits();
    let scalar = scalar.to_le_bytes().to_vec();
    insert("layer_scalar", Dtype::BF16, &[1], scalar);
    checkpoint
}

/// The call's rows: rows `0..live` at the positions after the history, every other row poison.
fn hidden(live: usize) -> Vec<bf16> {
    let values = values(Rows::SIZE * Hidden::SIZE, 0x51ed_270b, 4.0);
    (values.into_iter().enumerate())
        .map(|(at, value)| match at / Hidden::SIZE < live {
            true => value,
            false => bf16::from_bits(POISON),
        })
        .collect()
}

/// Pages of `Windows` windows: row `row` sees the keys `visible(row)`, which the windows hold as a
/// ring from the block pair of its first one, the keys past the ring in the slots before it; key
/// `k` of row `row` lies at byte offset `key(row, k)` of the pool, and a row that sees none still
/// names a readable block.
async fn pages<Windows: AxisName>(
    device: &mut Device,
    visible: impl Fn(usize) -> Range<usize>,
    key: impl Fn(usize, usize) -> i32,
) -> Pages<Rows, Windows, Chip> {
    let (windows, entries) = (Windows::SIZE, BlockIndex::SIZE);
    let (ring, pair) = (windows * Cache::SIZE, 2 * Block::SIZE);
    let first = |row: usize| visible(row).start / pair * pair;
    let kept = |row: usize| (visible(row).end - first(row)).min(ring);
    let wrapped = |row: usize| visible(row).end.saturating_sub(first(row) + ring);
    let table: Vec<i32> = (0..Rows::SIZE * windows * entries)
        .map(|at| {
            let (window, row) = (at / (Rows::SIZE * entries), at / entries % Rows::SIZE);
            let entry = window * entries + at % entries;
            let used = visible(row).end.div_ceil(Block::SIZE).max(1);
            let block = (first(row) / Block::SIZE + entry).min(used - 1);
            key(row, block * Block::SIZE)
        })
        .collect();
    let counts: Vec<i32> = (0..windows * Rows::SIZE)
        .map(|at| {
            let (window, row) = (at / Rows::SIZE, at % Rows::SIZE);
            let seen = kept(row).saturating_sub(window * Cache::SIZE);
            seen.min(Cache::SIZE).div_ceil(Block::SIZE) as i32
        })
        .collect();
    let bias: Vec<f32> = (0..Rows::SIZE * windows * Cache::SIZE)
        .map(|at| {
            let (row, slot) = (at / (windows * Cache::SIZE), at % (windows * Cache::SIZE));
            let held = [slot, slot + ring].map(|key| first(row) + key);
            if held.iter().any(|key| visible(row).contains(key)) {
                0.0
            } else {
                PADDING_BIAS
            }
        })
        .collect();
    let rest = (0..Rows::SIZE)
        .filter(|&row| !visible(row).is_empty())
        .map(|row| (kept(row) - 1) / Cache::SIZE)
        .max();
    let rest = HostTensor::<i32, m![1]>::from_vec([rest.unwrap_or(0) as i32]);
    let offsets = (0..Rows::SIZE * pair).map(|at| {
        let (row, past) = (at / pair, at % pair);
        match past < wrapped(row) {
            true => key(row, first(row) + ring + past),
            false => 0,
        }
    });
    let wraps: Vec<i32> = (0..Rows::SIZE).map(|row| wrapped(row) as i32).collect();
    let any = wraps.iter().any(|&count| count > 0);
    // The paging's copy of the rows that hold the last live row, each row's windows together, of
    // the windows a cluster's lanes hold.
    let (lanes, tile) = (LaneWindows::<Windows>::SIZE, WindowRows::<Windows>::SIZE);
    let last = (0..Rows::SIZE)
        .filter(|&row| !visible(row).is_empty())
        .max();
    let first = last.unwrap_or(0) / tile * tile;
    let row_table: Vec<i32> = (0..tile * lanes * entries)
        .map(|at| {
            let (row, window) = (first + at / (lanes * entries), at / entries % lanes);
            table[(window * Rows::SIZE + row) * entries + at % entries]
        })
        .collect();
    let row_counts: Vec<i32> = (0..tile * lanes)
        .map(|at| counts[at % lanes * Rows::SIZE + first + at / lanes])
        .collect();
    Pages {
        table: HostTensor::<i32, m![Windows, Rows, BlockIndex]>::from_vec(table)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        lengths: HostTensor::<i32, m![Windows, Rows]>::from_vec(counts)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        bias: HostTensor::<f32, m![Rows, Windows, Cache]>::from_vec(bias)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        rest: rest.to_hbm(&mut device.pdma).await.unwrap(),
        rows: RowMajor {
            table: HostTensor::<
                i32,
                m![{ WindowRow<Rows, Windows> }, { Symbol<LaneWindows<Windows>> }, BlockIndex],
            >::from_vec(row_table)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
            lengths: HostTensor::<
                i32,
                m![{ WindowRow<Rows, Windows> }, { Symbol<LaneWindows<Windows>> }],
            >::from_vec(row_counts)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        },
        wrap: Wrap {
            offsets: HostTensor::<i32, m![Rows, BlockIndex % 2, Block]>::from_vec(
                offsets.collect::<Vec<_>>(),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
            counts: HostTensor::<i32, m![Rows]>::from_vec(wraps)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            any: HostTensor::<i32, m![1]>::from_vec([any as i32])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        },
    }
}

/// The call's context of group `A`: each row's rotation at its position, its slot after the
/// history and the pages of the history and the rows before it, in the blocks of a layer's cache.
async fn context<A: Attention<Rotated: AxisName>>(device: &mut Device, live: usize) -> Context<A> {
    let frequencies =
        PairFrequencies::<<A as Attention>::Rotated>::unscaled::<<A as Attention>::Head>(1e4)
            .unwrap();
    context_at::<A>(device, (history(), live), &frequencies).await
}

/// The context of a call of `live` rows after `history` positions, its rows turning at
/// `frequencies`, each row seeing the history and the rows up to its own, at most the keys its
/// group's windows hold.
async fn context_at<A: Attention<Rotated: AxisName>>(
    device: &mut Device,
    (history, live): (usize, usize),
    frequencies: &PairFrequencies<<A as Attention>::Rotated>,
) -> Context<A> {
    let rotated = <A as Attention>::Rotated::SIZE;
    let pairs = rotated / 2;
    let (mut low, mut high) = (Vec::new(), Vec::new());
    for row in 0..Rows::SIZE {
        let position = (history + row) as f32;
        for at in 0..rotated {
            let (sin, cos) = f64::from(position * frequencies.pairs()[at % pairs]).sin_cos();
            let (l, h) = match at < pairs {
                true => (cos, -sin),
                false => (sin, cos),
            };
            low.push(bf16::from_f32(l as f32));
            high.push(bf16::from_f32(h as f32));
        }
    }
    let table = |values: Vec<bf16>| {
        HostTensor::<bf16, m![Rows, { Symbol<<A as Attention>::Rotated> }]>::from_vec(values)
    };
    let rotation = Rotation {
        low: table(low).to_hbm(&mut device.pdma).await.unwrap(),
        high: table(high).to_hbm(&mut device.pdma).await.unwrap(),
    };
    let slots = (0..Rows::SIZE).map(|row| match row < live {
        true => ((history + row) as i32) * Pool::<A>::ROW_STRIDE,
        false => 0,
    });
    let write = HostTensor::<i32, m![Rows]>::from_vec(slots.collect::<Vec<_>>());
    let window = <A as Attention>::Windows::SIZE * Cache::SIZE;
    let seen = |row: usize| match row < live {
        true => (history + row + 1).saturating_sub(window)..history + row + 1,
        false => 0..0,
    };
    let read = pages::<<A as Attention>::Windows>(device, seen, |_, key| {
        let (block, slot) = (key / Block::SIZE, key % Block::SIZE);
        block as i32 * Pool::<A>::BLOCK_STRIDE + slot as i32 * Pool::<A>::ROW_STRIDE
    })
    .await;
    attention::Context {
        positions: rotation,
        write: write.to_hbm(&mut device.pdma).await.unwrap(),
        read,
    }
}

/// Values of a token of group `A`'s cache: its key and value of every head.
fn width<A: Attention>() -> usize {
    KeyValue::SIZE * <A as Attention>::Heads::SIZE * <A as Attention>::Head::SIZE
}

/// Layer `layer`'s cache: the history's keys and values, every later slot poison.
fn cache<A: Attention>(layer: usize) -> Vec<bf16> {
    let history = values(
        history() * width::<A>(),
        0x2545_f491 ^ (layer as u64) << 32,
        0.25,
    );
    let poison = bf16::from_bits(POISON);
    (0..blocks() * Block::SIZE * width::<A>())
        .map(|at| history.get(at).copied().unwrap_or(poison))
        .collect()
}

/// Tokens a block of group `A`'s cache holds, each its key and value.
type Tokens<A> = m![
    Block,
    KeyValue,
    { Symbol<<A as Attention>::Heads> },
    { Symbol<<A as Attention>::Head> }
];

/// `values` in a new cache of [`blocks`] blocks.
async fn pool<A: Attention<Cache = bf16>>(device: &mut Device, values: &[bf16]) -> Pool<A> {
    let bytes = blocks() * Pool::<A>::BLOCK_STRIDE as usize;
    let mut pool = Pool::<A>::from_buffer(HbmBuffer::alloc(device, bytes).unwrap()).unwrap();
    for (block, values) in values.chunks_exact(values.len() / blocks()).enumerate() {
        HostTensor::<bf16, Tokens<A>>::from_vec(values.to_vec())
            .to_hbm(&mut device.pdma)
            .output(&mut pool.tokens[block])
            .await
            .unwrap();
    }
    pool
}

/// Every value of `pool`, block by block.
async fn read<A: Attention<Cache = bf16>>(device: &mut Device, pool: &Pool<A>) -> Vec<bf16> {
    let mut values = Vec::new();
    for block in 0..blocks() {
        let tokens = pool.tokens[block]
            .to_host::<Tokens<A>>(&mut device.pdma)
            .await;
        values.extend(tokens.unwrap().into_vec());
    }
    values
}

/// The final norm.
fn norm(device: &mut Device) -> FinalNorm {
    let mut checkpoint = Checkpoint::default();
    let gains = (0..Hidden::SIZE).flat_map(|channel| {
        bf16::from_f32(0.75 + (channel % 5) as f32 / 8.0)
            .to_bits()
            .to_le_bytes()
    });
    let gains = gains.collect();
    (checkpoint.insert("norm.weight", Dtype::BF16, &[Hidden::SIZE], gains)).unwrap();
    checkpoint.load(device).unwrap()
}

/// Two sliding layers and two of group `B`, each its own weights, in checkpoint order of their
/// collections, and the final norm.
fn model<B: Kind>(device: &mut Device) -> (Interleaved<B>, FinalNorm) {
    let first = [0, 2].map(|at| checkpoint::<Group>(at).load(&mut *device).unwrap());
    let second = [1, 3].map(|at| checkpoint::<B>(at).load(&mut *device).unwrap());
    let layers = Interleave {
        layers: (first, second),
    };
    (layers, norm(device))
}

/// Every layer's cache before a call, in checkpoint order: the first collection's at even
/// layers, the second's of group `B` at odd ones.
fn caches<B: Kind>() -> [Vec<bf16>; 4] {
    [
        cache::<Group>(0),
        cache::<B>(1),
        cache::<Group>(2),
        cache::<B>(3),
    ]
}

/// The call's rows, `live` of them live, and their count.
async fn input(device: &mut Device, live: usize) -> (HiddenRows, Value) {
    let input = HostTensor::<bf16, m![Rows, Hidden]>::from_vec(hidden(live));
    let count = HostTensor::<i32, m![1]>::from_vec([live as i32]);
    let input = input.to_hbm(&mut device.pdma).await.unwrap();
    (input, count.to_hbm(&mut device.pdma).await.unwrap())
}

/// The `live` rows of `output`.
async fn rows(device: &mut Device, output: &HiddenRows, live: usize) -> Vec<bf16> {
    let output = output.to_host::<m![Rows, Hidden]>(&mut device.pdma).await;
    let mut output = output.unwrap().into_vec();
    output.truncate(live * Hidden::SIZE);
    output
}

/// The call's `live` rows through two sliding layers and two of group `B`, by the schedule or
/// `in_order`, from the caches `caches` hold in checkpoint order: the live rows' outputs and every
/// layer's cache after it.
async fn run<B: Kind>(
    device: &mut Device,
    in_order: bool,
    model: (&Interleaved<B>, &FinalNorm),
    caches: &[Vec<bf16>; 4],
    live: usize,
) -> (Vec<bf16>, [Vec<bf16>; 4]) {
    let contexts = (
        context::<Group>(device, live).await,
        context::<B>(device, live).await,
    );
    let mut pools = (
        [
            pool::<Group>(device, &caches[0]).await,
            pool::<Group>(device, &caches[2]).await,
        ],
        [
            pool::<B>(device, &caches[1]).await,
            pool::<B>(device, &caches[3]).await,
        ],
    );
    let (input, count) = input(device, live).await;
    let (first, second) = (&contexts.0, &contexts.1);
    let input = (&input, &count);
    let output = B::forward(device, in_order, model, (first, second), &mut pools, input).await;
    let output = rows(device, &output, live).await;
    let ([a, b], [c, d]) = &pools;
    let caches = [
        read::<Group>(device, a).await,
        read::<B>(device, c).await,
        read::<Group>(device, b).await,
        read::<B>(device, d).await,
    ];
    (output, caches)
}

/// Every layer's cache before a call of the period, in checkpoint order.
fn period_caches() -> [Vec<bf16>; 6] {
    [0, 1, 2, 3, 4, 5].map(|layer| match layer < 5 {
        true => cache::<Group>(layer),
        false => cache::<full::Group>(layer),
    })
}

/// The call's `live` rows through the period, by the schedule or `in_order`, from the caches
/// `caches` hold in checkpoint order: the live rows' outputs and every layer's cache after it.
async fn run_period(
    device: &mut Device,
    in_order: bool,
    model: (&Period, &FinalNorm),
    caches: &[Vec<bf16>; 6],
    live: usize,
) -> (Vec<bf16>, [Vec<bf16>; 6]) {
    let contexts = (
        context::<Group>(device, live).await,
        context::<full::Group>(device, live).await,
    );
    let mut pools = (
        [
            pool::<Group>(device, &caches[0]).await,
            pool::<Group>(device, &caches[1]).await,
            pool::<Group>(device, &caches[2]).await,
            pool::<Group>(device, &caches[3]).await,
            pool::<Group>(device, &caches[4]).await,
        ],
        [pool::<full::Group>(device, &caches[5]).await],
    );
    let (input, count) = input(device, live).await;
    let arguments = (
        &mut *device,
        model,
        (&contexts.0, &contexts.1),
        &mut pools,
        (&input, &count),
    );
    let output = match in_order {
        true => launch(sequential_period, arguments).await,
        false => launch(scheduled_period, arguments).await,
    };
    let output = rows(device, &output.unwrap(), live).await;
    let ([a, b, c, d, e], [f]) = &pools;
    let caches = [
        read::<Group>(device, a).await,
        read::<Group>(device, b).await,
        read::<Group>(device, c).await,
        read::<Group>(device, d).await,
        read::<Group>(device, e).await,
        read::<full::Group>(device, f).await,
    ];
    (output, caches)
}

/// The sequential forward's outputs are finite and each layer's cache, its tokens `widths` of
/// that layer's values, holds every live row's keys and values, the history and the slots past
/// the live rows as they were.
fn written(
    live: usize,
    output: &[bf16],
    (found, before): (&[Vec<bf16>], &[Vec<bf16>]),
    widths: &[usize],
) {
    for (at, value) in output.iter().enumerate() {
        let value = value.to_f32();
        assert!(value.is_finite(), "{live} rows: element {at} is {value}");
    }
    let rows = history()..history() + live;
    for (layer, ((found, before), &width)) in found.iter().zip(before).zip(widths).enumerate() {
        for (at, (found, before)) in found.iter().zip(before).enumerate() {
            let token = at / width;
            let place = format!("layer {layer}, {live} rows: token {token}");
            match rows.contains(&token) {
                true => assert!(found.to_f32().is_finite(), "{place}"),
                false => assert_eq!(found.to_bits(), before.to_bits(), "{place}"),
            }
        }
    }
}

/// The schedule's live rows and caches `found` are the sequential forward's `want` bit for bit.
fn agree(live: usize, found: (&[bf16], &[Vec<bf16>]), want: (&[bf16], &[Vec<bf16>])) {
    for (at, (found, want)) in found.0.iter().zip(want.0).enumerate() {
        let (row, feature) = (at / Hidden::SIZE, at % Hidden::SIZE);
        let place = format!("{live} rows: row {row}, feature {feature}");
        assert_eq!(found.to_bits(), want.to_bits(), "{place}");
    }
    for (layer, (held, wanted)) in found.1.iter().zip(want.1).enumerate() {
        let same = held
            .iter()
            .zip(wanted)
            .position(|(h, w)| h.to_bits() != w.to_bits());
        assert_eq!(
            same, None,
            "layer {layer}, {live} rows: first differing cache element"
        );
    }
}

/// Over E4M3 caches the period writes each live row's key and value of its first layer as the
/// host codes the BF16 period's: divided by the layer's scale, saturated, to nearest even; every
/// other slot keeps its code, and the live rows stay finite.
#[tokio::test(flavor = "current_thread")]
async fn codes_the_first_layer_as_the_host_rounds_it() {
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(scheduled_coded.topology()).unwrap();
    let sliding = [0, 1, 2, 3, 4].map(|at| checkpoint::<Group>(at).load(&mut device).unwrap());
    let full = [checkpoint::<full::Group>(5).load(&mut device).unwrap()];
    let period: Period = Interleave {
        layers: (sliding, full),
    };
    let coded = coded_period(&mut device);
    let norm = norm(&mut device);
    let caches = period_caches();
    for live in [Rows::SIZE, 9] {
        let (want, written) = run_period(&mut device, false, (&period, &norm), &caches, live).await;
        let (found, first) = run_coded(&mut device, (&coded, &norm), &caches, live).await;
        let expected: Vec<f8e4m3> = written[0]
            .iter()
            .map(|value| code(value.to_f32(), CODED))
            .collect();
        let (found_codes, expected_codes) = (f8e4m3::to_buf(&first), f8e4m3::to_buf(&expected));
        let differs =
            (found_codes.iter().zip(&expected_codes)).position(|(found, want)| found != want);
        assert_eq!(
            differs.map(|at| at / width::<Group>()),
            None,
            "{live} rows: first differing token"
        );
        assert!(
            found.iter().all(|value| value.to_f32().is_finite()),
            "{live} rows"
        );
        let (worst, mean) = deviation(&found, &want);
        println!(
            "deviation rows-{live} largest {worst:.5} mean {mean:.6} (of the largest BF16 output)"
        );
    }
}

/// The scheduled run of two sliding layers and two of group `B` gives every live row's output and
/// every slot of every layer's cache bit for bit as the sequential forward does, which itself
/// writes every live row and nothing else.
async fn schedules_as_in_order<B: Kind>() {
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(scheduled.topology()).unwrap();
    let (layers, norm) = model::<B>(&mut device);
    let caches = caches::<B>();
    let widths = [width::<Group>(), width::<B>()].repeat(2);
    for live in [Rows::SIZE, 9] {
        let model = (&layers, &norm);
        let (want, wanted) = run::<B>(&mut device, true, model, &caches, live).await;
        written(live, &want, (&wanted, &caches), &widths);
        let (found, held) = run::<B>(&mut device, false, model, &caches, live).await;
        agree(live, (&found, &held), (&want, &wanted));
    }
}

/// The sequential forward of the four layers writes every live row's keys and values into each
/// layer's cache and leaves finite rows; rows past the live ones and the history stay as they were.
#[tokio::test(flavor = "current_thread")]
async fn runs_the_layers_sequentially() {
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(sequential.topology()).unwrap();
    let (layers, norm) = model::<Group>(&mut device);
    let caches = caches::<Group>();
    for live in [Rows::SIZE, 9] {
        let (output, found) =
            run::<Group>(&mut device, true, (&layers, &norm), &caches, live).await;
        written(live, &output, (&found, &caches), &[width::<Group>(); 4]);
    }
}

/// The schedule's run of the layers and the final norm, a period seam between every other pair,
/// gives every live row's output and every slot of every layer's cache bit for bit as the
/// sequential forward does.
#[tokio::test(flavor = "current_thread")]
async fn schedules_the_layers_as_they_run_in_order() {
    schedules_as_in_order::<Group>().await;
}

/// Sliding and full layers interleaved, sliding to full, full to sliding across a period and full
/// to the final norm, run by the schedule as they run in order, bit for bit.
#[tokio::test(flavor = "current_thread")]
async fn schedules_sliding_and_full_layers_as_they_run_in_order() {
    schedules_as_in_order::<full::Group>().await;
}

/// The catalog's period, five sliding layers then a full one into the final norm, runs by the
/// schedule as it runs in order, bit for bit, the sequential forward writing every live row and
/// nothing else.
#[tokio::test(flavor = "current_thread")]
async fn schedules_the_catalog_period_as_it_runs_in_order() {
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(scheduled_period.topology()).unwrap();
    let sliding = [0, 1, 2, 3, 4].map(|at| checkpoint::<Group>(at).load(&mut device).unwrap());
    let full = [checkpoint::<full::Group>(5).load(&mut device).unwrap()];
    let layers: Period = Interleave {
        layers: (sliding, full),
    };
    let norm = norm(&mut device);
    let caches = period_caches();
    let mut widths = vec![width::<Group>(); 5];
    widths.push(width::<full::Group>());
    for live in [Rows::SIZE, 9] {
        let model = (&layers, &norm);
        let (want, wanted) = run_period(&mut device, true, model, &caches, live).await;
        written(live, &want, (&wanted, &caches), &widths);
        let (found, held) = run_period(&mut device, false, model, &caches, live).await;
        agree(live, (&found, &held), (&want, &wanted));
    }
}

type FullHeads = <full::Group as Attention>::Heads;
type FullGroups = <full::Group as Attention>::Groups;
type FullHead = <full::Group as Attention>::Head;
/// Each row's query heads of the full group, in head order.
type FullQueries = HbmTensor<bf16, Chip, m![Rows, FullHeads, FullGroups, FullHead]>;
/// The live rows' full query heads as the projection leaves them for attention.
type FullQueryHeads = Live<
    DmTensor<
        bf16,
        Chip,
        m![1 # 2],
        m![Rows % ({ Symbol<Lanes<Rows>> }) # 8, FullHeads # 8, 1 # 4],
        m![
            { m![Rows / ({ Symbol<Lanes<Rows>> }), FullGroups] },
            FullHead
        ],
    >,
    Rows,
>;

/// A score multiplier of one, as the catalog's normalized queries and keys take.
const UNIT: usize = 1.0_f32.to_bits() as usize;

/// The full group's attention leaf over each live row's pages of `pool`.
#[device(chip = 1)]
fn attend_full(
    device: &mut Device,
    (queries, live): (&FullQueries, &Value),
    pool: &Pool<full::Group>,
    read: &Pages<Rows, <full::Group as Attention>::Windows, Chip>,
    attended: &mut FullQueries,
) {
    let live = Live::<(), Rows>::read(live);
    let query: FullQueryHeads = live.with(queries.to_dm(&mut device.tdma));
    let attention = decoder::Attend::<full::Group, UNIT>::default();
    let output: Live<Attended<Rows, FullHeads, FullGroups, FullHead>, Rows> =
        Op::forward(&attention, &mut *device, (&query, pool, read));
    (output.value().view()).to_hbm_view(&mut device.tdma, attended.view_mut());
}

/// Each history a bank of the leaf's pool holds, in keys: within and across blocks, shards and
/// windows.
const LENGTHS: [usize; 16] = [
    1, 2, 15, 16, 17, 33, 100, 511, 512, 513, 777, 1024, 1025, 1536, 2047, 2048,
];

/// Keys a bank of the leaf's pool holds at most.
const BANK: usize = 2048;

/// The one KV head of a full layer attends each live row's own history, every shard and window
/// of it, as an FP64 softmax over the same BF16 operands does, to the agreement BF16 scores and
/// weights leave; the slots past a history hold NaN and infinities nothing may read.
#[tokio::test(flavor = "current_thread")]
async fn one_head_attends_each_live_row_as_the_host() {
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(attend_full.topology()).unwrap();
    let (heads, head) = (FullGroups::SIZE, FullHead::SIZE);
    let width = width::<full::Group>();
    let data = values(LENGTHS.len() * BANK * width, 0x3c6e_f372, 0.5);
    let poisons = [f32::NAN, -f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let cells = (data.iter().enumerate()).map(|(at, &value)| {
        let (bank, key) = (at / (BANK * width), at / width % BANK);
        match key < LENGTHS[bank] {
            true => value,
            false => bf16::from_f32(poisons[at % poisons.len()]),
        }
    });
    let cells: Vec<bf16> = cells.collect();
    let storage = HostBufferSlice::from(&bf16::to_buf(&cells))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let pool = Pool::<full::Group>::from_buffer(storage).unwrap();
    let bank = |row: usize| (row * 5 + row / LENGTHS.len()) % LENGTHS.len();
    let (stride, row_stride) = (
        Pool::<full::Group>::BLOCK_STRIDE,
        Pool::<full::Group>::ROW_STRIDE,
    );
    let banks = BANK / Block::SIZE;
    let read = pages::<<full::Group as Attention>::Windows>(
        &mut device,
        |row| 0..LENGTHS[bank(row)],
        |row, key| {
            let (block, slot) = (key / Block::SIZE, key % Block::SIZE);
            (bank(row) * banks + block) as i32 * stride + slot as i32 * row_stride
        },
    )
    .await;
    let queries = values(Rows::SIZE * heads * head, 0x7f4a_7c15, 0.125);
    let query: FullQueries =
        HostTensor::<bf16, m![Rows, FullHeads, FullGroups, FullHead]>::from_vec(queries.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
    let key = |row: usize, token: usize, channel: usize| {
        cells[(bank(row) * BANK + token) * width + channel].to_f32() as f64
    };
    let value = |row: usize, token: usize, channel: usize| key(row, token, head + channel);
    for live in [Rows::SIZE, 9] {
        let count: Value = HostTensor::<i32, m![1]>::from_vec([live as i32])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let poison = vec![bf16::from_bits(POISON); Rows::SIZE * heads * head];
        let mut attended =
            HostTensor::<bf16, m![Rows, FullHeads, FullGroups, FullHead]>::from_vec(poison)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let arguments = (&mut device, (&query, &count), &pool, &read, &mut attended);
        launch(attend_full, arguments).await.unwrap();
        let found = attended
            .to_host::<m![Rows, FullHeads, FullGroups, FullHead]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for row in 0..live {
            let length = LENGTHS[bank(row)];
            for group in 0..heads {
                let query = &queries[(row * heads + group) * head..][..head];
                let scores: Vec<f64> = (0..length)
                    .map(|token| {
                        let dot = query.iter().enumerate();
                        dot.map(|(c, q)| q.to_f32() as f64 * key(row, token, c))
                            .sum()
                    })
                    .collect();
                let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|score| (score - top).exp()).collect();
                let total: f64 = weights.iter().sum();
                for channel in 0..head {
                    let weighted = weights.iter().enumerate();
                    let want = weighted
                        .map(|(token, weight)| weight * value(row, token, channel))
                        .sum::<f64>()
                        / total;
                    let at = (row * heads + group) * head + channel;
                    let got = found[at].to_f32();
                    let place = format!("{live} rows: row {row} (length {length}), head {group}");
                    assert!(got.is_finite(), "{place}, channel {channel}: {got}");
                    assert!(
                        (got as f64 - want).abs() <= 1.0 / 128.0,
                        "{place}, channel {channel}: {got} != {want}"
                    );
                }
            }
        }
    }
}

/// An NVFP4 projection of `I` inputs to `O` outputs under contract `C`, its codes and block scales
/// laid out in HBM as `K` and `V`.
type Nvfp4<I, O, C, K = m![O, I], V = Scales<I, O>> = BlockScaled<
    I,
    O,
    f4e2m1,
    f8e4m3,
    Scales<I, O>,
    C,
    Chip,
    HbmTensor<f4e2m1, Chip, K>,
    HbmTensor<f8e4m3, Chip, V>,
>;

/// An NVFP4 projection under the summed contract, held as the catalog's summed form holds it.
type Summing<I, O> = Nvfp4<
    I,
    O,
    Summed<W4A16, Hbm, Features, Copies>,
    summed::Codes<I, O, Features>,
    summed::Blocks<I, O, Features>,
>;

/// The call's live rows through one NVFP4 projection, from and back to HBM rows.
#[device(chip = 1)]
fn projected<I: AxisName, O: AxisName, C, K: M, V: M>(
    device: &mut Device,
    linear: &Nvfp4<I, O, C, K, V>,
    (input, live): (&HbmTensor<bf16, Chip, m![Rows, I]>, &Value),
) -> HbmTensor<bf16, Chip, m![Rows, O]>
where
    Nvfp4<I, O, C, K, V>: DeviceSend
        + for<'x> Op<
            &'x Live<Spread<I, Rows, Chip>, Rows>,
            Output = Live<Spread<O, Rows, Chip>, Rows>,
        >,
{
    let live = Live::<(), Rows>::read(live);
    let rows: Live<Spread<I, Rows, Chip>, Rows> = live.with(input.to_dm(&mut device.tdma));
    let output = Op::forward(linear, &mut *device, &rows);
    output.value().to_hbm(&mut device.tdma)
}

/// Uniform values in `[0, 1)` from a xorshift state.
fn uniform(count: usize, mut state: u64) -> impl Iterator<Item = f64> {
    (0..count).map(move |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    })
}

/// Random NVFP4 weights of `I` inputs and `O` outputs: codes over the sixteen E2M1 values, E4M3
/// block scales from 8 to 448 and a tensor scale near 2^14, with each row of the call's inputs in
/// `[-2, 2)`.
struct Drawn {
    codes: Vec<f4e2m1>,
    scales: Vec<f8e4m3>,
    global: f32,
    inputs: Vec<bf16>,
}

impl Drawn {
    fn new<I: AxisName, O: AxisName>(seed: u64) -> Self {
        let (i, o) = (I::SIZE, O::SIZE);
        Self {
            codes: (uniform(o * i, seed).map(|u| f4e2m1::from_bits((u * 16.0) as u8))).collect(),
            scales: (uniform(o * i / 16, seed + 1))
                .map(|u| f8e4m3::from_f32((8.0 + 440.0 * u) as f32))
                .collect(),
            global: 2f32.powi(14) * 1.37,
            inputs: (uniform(Rows::SIZE * i, seed + 2))
                .map(|u| bf16::from_f32((4.0 * u - 2.0) as f32))
                .collect(),
        }
    }

    /// The weights on the device under the contract `contract` makes of the tensor scale, laid
    /// out as `G`.
    async fn coded<I: AxisName, O: AxisName, C, K: M, V: M, G: M>(
        &self,
        device: &mut Device,
        contract: impl FnOnce(Scale<G>) -> C,
    ) -> Nvfp4<I, O, C, K, V> {
        let global = HostTensor::<f32, m![1 # 8]>::from_vec([
            self.global,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);
        BlockScaled {
            weight: HostTensor::<f4e2m1, m![O, I]>::from_vec(self.codes.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            weight_scale: HostTensor::<f8e4m3, Scales<I, O>>::from_vec(self.scales.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            contract: contract(global.to_hbm(&mut device.pdma).await.unwrap()),
            shape: core::marker::PhantomData,
        }
    }

    /// The weights on the device under the summed contract, laid out as the catalog's summed form
    /// lays them out: the block scales beside the tensor scale.
    async fn summed<I: AxisName, O: AxisName>(&self, device: &mut Device) -> Summing<I, O> {
        let global = || {
            HostTensor::<f32, m![1 # 8]>::from_vec([self.global, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        };
        let scales =
            HostTensor::<f8e4m3, m![{ Chip }, { Scales<I, O> }]>::from_vec(self.scales.clone());
        BlockScaled {
            weight: HostTensor::<f4e2m1, m![O, I]>::from_vec(self.codes.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            weight_scale: summed::lead::<I, O, Features>(scales, global())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            contract: Summed {
                input: W4A16,
                global: global().to_hbm(&mut device.pdma).await.unwrap(),
                place: core::marker::PhantomData,
            },
            shape: core::marker::PhantomData,
        }
    }

    /// Every row's outputs in FP64: `sum_k code_k scale_k x_k / global`.
    fn exact<I: AxisName, O: AxisName>(&self) -> Vec<f64> {
        let (i, o) = (I::SIZE, O::SIZE);
        let inputs: Vec<f64> = (self.inputs.iter())
            .map(|x| f64::from(x.to_f32()))
            .collect();
        let mut exact = vec![0.0; Rows::SIZE * o];
        let mut weights = vec![0.0; i];
        for output in 0..o {
            for (k, weight) in weights.iter_mut().enumerate() {
                let at = output * i + k;
                *weight = f64::from(self.codes[at].to_f32())
                    * f64::from(self.scales[at / 16].to_f32())
                    / f64::from(self.global);
            }
            for (row, input) in inputs.chunks_exact(i).enumerate() {
                exact[row * o + output] = weights.iter().zip(input).map(|(w, x)| w * x).sum();
            }
        }
        exact
    }

    /// The call's rows, `live` of them live and every other row poison, and their count.
    async fn rows<I: AxisName>(
        &self,
        device: &mut Device,
        live: usize,
    ) -> (HbmTensor<bf16, Chip, m![Rows, I]>, Value) {
        let rows = (self.inputs.iter().enumerate()).map(|(at, &value)| match at / I::SIZE < live {
            true => value,
            false => bf16::from_bits(POISON),
        });
        let rows = HostTensor::<bf16, m![Rows, I]>::from_vec(rows.collect::<Vec<_>>());
        let count = HostTensor::<i32, m![1]>::from_vec([live as i32]);
        (
            rows.to_hbm(&mut device.pdma).await.unwrap(),
            count.to_hbm(&mut device.pdma).await.unwrap(),
        )
    }
}

/// The `live` rows of `linear`'s projection of `drawn`'s rows.
async fn project<I: AxisName, O: AxisName, C, K: M, V: M>(
    device: &mut Device,
    linear: &Nvfp4<I, O, C, K, V>,
    drawn: &Drawn,
    live: usize,
) -> Vec<bf16>
where
    Nvfp4<I, O, C, K, V>: DeviceSend
        + for<'x> Op<
            &'x Live<Spread<I, Rows, Chip>, Rows>,
            Output = Live<Spread<O, Rows, Chip>, Rows>,
        >,
{
    let (rows, count) = drawn.rows::<I>(device, live).await;
    let output = launch(projected, (&mut *device, linear, (&rows, &count)))
        .await
        .unwrap();
    let mut output = output
        .to_host::<m![Rows, O]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    output.truncate(live * O::SIZE);
    output
}

/// How far `found`, the first rows of `exact`, is from it: the worst output's error over one BF16
/// rounding of it plus `2^-9` of its row's RMS, and the RMS error over the RMS of the outputs.
fn error(found: &[bf16], exact: &[f64], outputs: usize) -> (f64, f64) {
    let (mut worst, mut squares, mut total) = (0.0f64, 0.0, 0.0);
    for (found, exact) in found.chunks_exact(outputs).zip(exact.chunks_exact(outputs)) {
        let rms = (exact.iter().map(|e| e * e).sum::<f64>() / outputs as f64).sqrt();
        for (got, want) in found.iter().zip(exact) {
            let got = f64::from(got.to_f32());
            worst = worst.max((got - want).abs() / (want.abs() / 256.0 + rms / 512.0));
            squares += (got - want) * (got - want);
            total += want * want;
        }
    }
    (worst, (squares / total).sqrt())
}

/// The summed contract's live rows, at one row, a lane group, one row past it, one row past two
/// and the whole envelope: each output within one BF16 rounding of the FP64 product plus `2^-9` of
/// its row's RMS for the FP32 sums, and nearer the product than the early contract's.
async fn sums_rows_nearer_than_early<I: AxisName, O: AxisName>(device: &mut Device, seed: u64)
where
    Summing<I, O>: DeviceSend
        + for<'x> Op<
            &'x Live<Spread<I, Rows, Chip>, Rows>,
            Output = Live<Spread<O, Rows, Chip>, Rows>,
        >,
    Nvfp4<I, O, Early<W4A16>>: DeviceSend
        + for<'x> Op<
            &'x Live<Spread<I, Rows, Chip>, Rows>,
            Output = Live<Spread<O, Rows, Chip>, Rows>,
        >,
{
    let drawn = Drawn::new::<I, O>(seed);
    let exact = drawn.exact::<I, O>();
    let summed: Summing<I, O> = drawn.summed(device).await;
    let early: Nvfp4<I, O, Early<W4A16>> = drawn
        .coded(device, |global| Early {
            input: W4A16,
            global,
        })
        .await;
    for live in [1, 8, 9, 17, Rows::SIZE] {
        let shape = format!("{}x{}, {live} live", I::SIZE, O::SIZE);
        let (worst, summed) = error(
            &project(device, &summed, &drawn, live).await,
            &exact,
            O::SIZE,
        );
        let (_, early) = error(
            &project(device, &early, &drawn, live).await,
            &exact,
            O::SIZE,
        );
        println!(
            "{shape}: summed worst {worst:.3} of the bound, RMS error {summed:.3e}; early {early:.3e}"
        );
        assert!(
            worst <= 1.0,
            "{shape}: an output is {worst} of the bound away"
        );
        assert!(
            summed < early,
            "{shape}: summed {summed} against early {early}"
        );
    }
}

/// The feed-forward's projections of the call's live rows under the summed contract give each
/// row's FP64 product within one rounding, nearer than the early contract: into the intermediate
/// width, each slice a row's whole input, and back, each slice a quarter of it.
#[tokio::test(flavor = "current_thread")]
async fn sums_nvfp4_rows_nearer_than_early() {
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    sums_rows_nearer_than_early::<Hidden, Intermediate>(&mut device, 1).await;
    sums_rows_nearer_than_early::<Intermediate, Hidden>(&mut device, 4).await;
}

/// The feed-forward's NVFP4 projections under the early and the summed contract, profiled at the
/// level `FURIOSA_OPT_PROFILE` asks for: for each live row count `PERIOD_PROFILE_ROWS` lists, a
/// warm-up and `PERIOD_PROFILE_STEPS` launches of each, every span written as [`Lines`] writes it.
#[tokio::test(flavor = "current_thread")]
#[ignore = "profiles launches on a device"]
async fn feed_forward_profile() {
    use std::future::IntoFuture;
    use tracing::Instrument;
    let list = |name: &str, default: &str| -> Vec<usize> {
        let list = std::env::var(name).unwrap_or_else(|_| default.into());
        list.split(',').map(|item| item.parse().unwrap()).collect()
    };
    let steps = list("PERIOD_PROFILE_STEPS", "10")[0];
    // Installed before the functions load, so that they load with the profile.
    let _spans = tracing::subscriber::set_default(Lines::new());
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let (up, down) = (
        Drawn::new::<Hidden, Intermediate>(1),
        Drawn::new::<Intermediate, Hidden>(4),
    );
    let gate: (
        Nvfp4<Hidden, Intermediate, Early<W4A16>>,
        Summing<Hidden, Intermediate>,
    ) = (
        up.coded(&mut device, |global| Early {
            input: W4A16,
            global,
        })
        .await,
        up.summed(&mut device).await,
    );
    let down_proj: (
        Nvfp4<Intermediate, Hidden, Early<W4A16>>,
        Summing<Intermediate, Hidden>,
    ) = (
        down.coded(&mut device, |global| Early {
            input: W4A16,
            global,
        })
        .await,
        down.summed(&mut device).await,
    );
    for live in list("PERIOD_PROFILE_ROWS", "1,8,9,64") {
        let rows = (
            up.rows::<Hidden>(&mut device, live).await,
            down.rows::<Intermediate>(&mut device, live).await,
        );
        for kernel in ["gate-early", "gate-summed", "down-early", "down-summed"] {
            let case = format!("{kernel}-rows-{live}");
            for sample in 0..=steps {
                let span = match sample {
                    0 => tracing::info_span!("warmup", case = case.as_str()),
                    step => tracing::info_span!("forward", case = case.as_str(), step),
                };
                let started = std::time::Instant::now();
                async {
                    let submit = tracing::info_span!("submit");
                    let ((hidden, hidden_live), (intermediate, intermediate_live)) =
                        (&rows.0, &rows.1);
                    match kernel {
                        "gate-early" => {
                            let arguments = (&mut device, &gate.0, (hidden, hidden_live));
                            let output = launch(projected, arguments)
                                .into_future()
                                .instrument(submit)
                                .await;
                            drop(output.unwrap());
                        }
                        "gate-summed" => {
                            let arguments = (&mut device, &gate.1, (hidden, hidden_live));
                            let output = launch(projected, arguments)
                                .into_future()
                                .instrument(submit)
                                .await;
                            drop(output.unwrap());
                        }
                        "down-early" => {
                            let arguments =
                                (&mut device, &down_proj.0, (intermediate, intermediate_live));
                            let output = launch(projected, arguments)
                                .into_future()
                                .instrument(submit)
                                .await;
                            drop(output.unwrap());
                        }
                        _ => {
                            let arguments =
                                (&mut device, &down_proj.1, (intermediate, intermediate_live));
                            let output = launch(projected, arguments)
                                .into_future()
                                .instrument(submit)
                                .await;
                            drop(output.unwrap());
                        }
                    }
                    (device.synchronize())
                        .instrument(tracing::info_span!("wait"))
                        .await
                        .unwrap();
                }
                .instrument(span)
                .await;
                let elapsed = started.elapsed().as_secs_f64() * 1e3;
                println!("{case} step {sample}: {elapsed:.3} ms");
            }
        }
    }
}

/// Every catalog root a registry row serves still runs by the decoder family's one loop: the
/// causal models of each Qwen3 size and form, Llama and Mistral.
#[test]
fn every_root_runs_by_the_loop() {
    use furiosa_opt_models::models::decoder::attention::{Geometry, llama, qwen3};
    use furiosa_opt_models::models::decoder::form::{Affine, BlockScaled, ChannelScaled};
    use furiosa_opt_models::models::decoder::launch;
    use furiosa_opt_models::specialize::attention::Decode;
    fn runs<R, Z: Geometry + Attention + Decode>()
    where
        for<'i, 'a, 's, 'b> R: Module<
                &'i mut launch::Input<'a, Z>,
                (),
                launch::State<'s, Z, <Z as Geometry>::Layers<launch::Layer<'b>>>,
            >,
    {
    }
    runs::<launch::Model<qwen3::Size<"0.6B">, Affine>, qwen3::Size<"0.6B">>();
    runs::<launch::Model<qwen3::Size<"8B">, Affine>, qwen3::Size<"8B">>();
    runs::<launch::Model<qwen3::Size<"4B">, ChannelScaled>, qwen3::Size<"4B">>();
    runs::<launch::Model<qwen3::Size<"8B">, BlockScaled>, qwen3::Size<"8B">>();
    runs::<launch::Model<llama::Size<"8B">, Affine>, llama::Size<"8B">>();
    runs::<launch::Model<llama::Size<"Mistral-7B">, Affine>, llama::Size<"Mistral-7B">>();
}

/// The catalog's own collections, sliding and full layers in its order, run by the schedule over
/// each group's context: every edge of the order, sliding to sliding, sliding to full, full to
/// sliding across a period and full to the final norm, is the one tokenwise work.
#[test]
fn schedules_the_catalog_order() {
    type Slot<'b, A> = (&'b mut Pool<A>, ());
    type Order = Interleave<([Sliding; gemma4::SLIDING], [Full; gemma4::FULL]), { gemma4::ORDER }>;
    fn runs<'m, 'c>()
    where
        for<'i, 's, 'b> Decoder<(&'m Order, &'m FinalNorm)>: Module<
                (&'i mut HiddenRows, Live<(), Rows>),
                (&'c Context<Group>, &'c Context<full::Group>),
                &'s mut (
                    (Places<Group, Rows>, [Slot<'b, Group>; gemma4::SLIDING]),
                    (
                        Places<full::Group, Rows>,
                        [Slot<'b, full::Group>; gemma4::FULL],
                    ),
                ),
                Output = (),
            >,
    {
    }
    runs();
}

/// The catalog's period and the mixed layers by the schedule, profiled at the level
/// `FURIOSA_OPT_PROFILE` asks for: for each live row count `PERIOD_PROFILE_ROWS` lists, a warm-up
/// and `PERIOD_PROFILE_STEPS` launches of each, every span written as [`Lines`] writes it, then the
/// hash of its last launch's live rows.
#[tokio::test(flavor = "current_thread")]
#[ignore = "profiles launches on a device"]
async fn period_profile() {
    use std::future::IntoFuture;
    use tracing::Instrument;
    let list = |name: &str, default: &str| -> Vec<usize> {
        let list = std::env::var(name).unwrap_or_else(|_| default.into());
        list.split(',').map(|item| item.parse().unwrap()).collect()
    };
    let steps = list("PERIOD_PROFILE_STEPS", "10")[0];
    // Installed before the functions load, so that they load with the profile.
    let _spans = tracing::subscriber::set_default(Lines::new());
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(scheduled_period.topology()).unwrap();
    let sliding = [0, 1, 2, 3, 4].map(|at| checkpoint::<Group>(at).load(&mut device).unwrap());
    let full = [checkpoint::<full::Group>(5).load(&mut device).unwrap()];
    let period: Period = Interleave {
        layers: (sliding, full),
    };
    let (mixed, norm) = model::<full::Group>(&mut device);
    let coded = coded_period(&mut device);
    let (held, mixes) = (period_caches(), caches::<full::Group>());
    for live in list("PERIOD_PROFILE_ROWS", "1,8,9,32,64") {
        let coded_contexts = (
            context::<CodedGroup>(&mut device, live).await,
            context::<CodedFull>(&mut device, live).await,
        );
        let mut coded_pools = (
            [
                coded_pool::<CodedGroup>(&mut device, &held[0]).await,
                coded_pool::<CodedGroup>(&mut device, &held[1]).await,
                coded_pool::<CodedGroup>(&mut device, &held[2]).await,
                coded_pool::<CodedGroup>(&mut device, &held[3]).await,
                coded_pool::<CodedGroup>(&mut device, &held[4]).await,
            ],
            [coded_pool::<CodedFull>(&mut device, &held[5]).await],
        );
        let contexts = (
            context::<Group>(&mut device, live).await,
            context::<full::Group>(&mut device, live).await,
        );
        let mut periods = (
            [
                pool::<Group>(&mut device, &held[0]).await,
                pool::<Group>(&mut device, &held[1]).await,
                pool::<Group>(&mut device, &held[2]).await,
                pool::<Group>(&mut device, &held[3]).await,
                pool::<Group>(&mut device, &held[4]).await,
            ],
            [pool::<full::Group>(&mut device, &held[5]).await],
        );
        let mut pools = (
            [
                pool::<Group>(&mut device, &mixes[0]).await,
                pool::<Group>(&mut device, &mixes[2]).await,
            ],
            [
                pool::<full::Group>(&mut device, &mixes[1]).await,
                pool::<full::Group>(&mut device, &mixes[3]).await,
            ],
        );
        let (input, count) = input(&mut device, live).await;
        let (contexts, rows) = ((&contexts.0, &contexts.1), (&input, &count));
        let mut outputs = Vec::new();
        for kernel in ["period", "mixed", "coded"] {
            let case = format!("{kernel}-rows-{live}");
            let mut last = None;
            for sample in 0..=steps {
                let span = match sample {
                    0 => tracing::info_span!("warmup", case = case.as_str()),
                    step => tracing::info_span!("forward", case = case.as_str(), step),
                };
                let started = std::time::Instant::now();
                let output = async {
                    let submit = tracing::info_span!("submit");
                    let output = match kernel {
                        "period" => {
                            let model = (&period, &norm);
                            let arguments = (&mut device, model, contexts, &mut periods, rows);
                            (launch(scheduled_period, arguments).into_future())
                                .instrument(submit)
                                .await
                        }
                        "mixed" => {
                            let model = (&mixed, &norm);
                            let arguments = (&mut device, model, contexts, &mut pools, rows);
                            (launch(scheduled_mixed, arguments).into_future())
                                .instrument(submit)
                                .await
                        }
                        _ => {
                            let model = (&coded, &norm);
                            let contexts = (&coded_contexts.0, &coded_contexts.1);
                            let arguments = (&mut device, model, contexts, &mut coded_pools, rows);
                            (launch(scheduled_coded, arguments).into_future())
                                .instrument(submit)
                                .await
                        }
                    };
                    let output = output.unwrap();
                    (device.synchronize())
                        .instrument(tracing::info_span!("wait"))
                        .await
                        .unwrap();
                    output
                }
                .instrument(span)
                .await;
                let elapsed = started.elapsed().as_secs_f64() * 1e3;
                println!("{case} step {sample}: {elapsed:.3} ms");
                last = Some(output);
            }
            let rows = crate::rows(&mut device, &last.unwrap(), live).await;
            println!("hash {case} {:016x}", fingerprint(&rows));
            outputs.push(rows);
        }
        let (worst, mean) = deviation(&outputs[2], &outputs[0]);
        println!("deviation rows-{live} largest {worst:.5} mean {mean:.6} (coded vs period)");
    }
}

/// A channel-scaled projection of `I` inputs into `O` outputs, the form of the catalog's
/// attention projections, its codes held as the form holds them.
type Coded<I, O> = furiosa_opt_models::ops::linear::Affine<
    furiosa_opt_models::tensor::form::BlockScaled<
        I,
        O,
        f8e4m3,
        f32,
        m![O, 1 # 8],
        (),
        Chip,
        HbmTensor<f8e4m3, Chip, furiosa_opt_models::models::decoder::form::channel::Codes<I, O>>,
    >,
    (),
>;

/// Rows as a norm leaves them for the query, key and value projections: by groups of eight rows,
/// sixteen features a block.
type Grouped<I> = HbmTensor<bf16, Chip, m![Rows / 8, I / 16, Rows % 8, I % 16]>;

/// Rows in order, each with its features in order, as the attended rows reach the output
/// projection.
type Ordered<I> = HbmTensor<bf16, Chip, m![Rows, I]>;

type SlidingQueryWidth = <Group as Attention>::QueryWidth;
type SlidingKeyWidth = <Group as Attention>::KvWidth;
type FullQueryWidth = <full::Group as Attention>::QueryWidth;
type FullKeyWidth = <full::Group as Attention>::KvWidth;

/// One projection of the catalog's layers alone: the live rows a count names of `input`, projected
/// by `weight` into rows in order.
macro_rules! projection {
    ($name:ident, $stored:ident, $in:ty, $out:ty) => {
        #[device(chip = 1)]
        fn $name(
            device: &mut Device,
            weight: &Coded<$in, $out>,
            rows: &Value,
            input: &$stored<$in>,
        ) -> Ordered<$out> {
            let mut output = Ordered::new();
            let live = Live::<(), Rows>::read(rows);
            Op::forward(weight, &mut *device, (live, input, &mut output));
            output
        }
    };
}

projection!(sliding_query_rows, Grouped, Hidden, SlidingQueryWidth);
projection!(sliding_key_rows, Grouped, Hidden, SlidingKeyWidth);
projection!(full_query_rows, Grouped, Hidden, FullQueryWidth);
projection!(full_key_rows, Grouped, Hidden, FullKeyWidth);
projection!(sliding_output_rows, Ordered, SlidingQueryWidth, Hidden);
projection!(full_output_rows, Ordered, FullQueryWidth, Hidden);

// The other channel-scaled decoders' widths whose projections take the same contraction: hidden
// 1024 and 2560 with their query, key, output, fused gate and up, and down widths, and a
// 4096-hidden MLP's fused gate and up.
axes![
    SmallHidden = 1024,
    SmallQueries = 2048,
    SmallKeys = 1024,
    SmallGateUp = 6144,
    MiddleHidden = 2560,
    MiddleQueries = 4096,
    MiddleKeys = 1024,
    MiddleGateUp = 19456,
    MiddleIntermediate = 9728,
    LargeHidden = 4096,
    LargeGateUp = 28672
];

projection!(small_query_rows, Grouped, SmallHidden, SmallQueries);
projection!(small_key_rows, Grouped, SmallHidden, SmallKeys);
projection!(middle_query_rows, Grouped, MiddleHidden, MiddleQueries);
projection!(middle_key_rows, Grouped, MiddleHidden, MiddleKeys);
projection!(middle_output_rows, Ordered, MiddleQueries, MiddleHidden);

/// A BF16 projection of `I` inputs into `O` outputs.
type Dense<I, O> = furiosa_opt_models::ops::linear::Affine<HbmTensor<bf16, Chip, m![O, I]>, ()>;

axes![LargeQueries = 4096, LargeGate = 14336];

/// One projection alone from rows spread over the slices as a norm leaves them, the catalog's and
/// the other decoders' path: the live rows a count names of `input`, projected by `weight`.
macro_rules! spread {
    ($name:ident, $weight:ident, $in:ty, $out:ty) => {
        #[device(chip = 1)]
        fn $name(
            device: &mut Device,
            weight: &$weight<$in, $out>,
            rows: &Value,
            input: &Ordered<$in>,
        ) -> Ordered<$out> {
            let live = Live::<(), Rows>::read(rows);
            let input: Live<Spread<$in, Rows, Chip>, Rows> =
                live.with(input.to_dm(&mut device.tdma));
            let output: Live<Spread<$out, Rows, Chip>, Rows> =
                Op::forward(weight, &mut *device, &input);
            output.value().to_hbm(&mut device.tdma)
        }
    };
}

spread!(large_gate_up_rows, Coded, LargeHidden, LargeGateUp);
spread!(small_gate_up_rows, Coded, SmallHidden, SmallGateUp);
spread!(middle_gate_up_rows, Coded, MiddleHidden, MiddleGateUp);
spread!(middle_down_rows, Coded, MiddleIntermediate, MiddleHidden);
spread!(spread_query_rows, Coded, Hidden, SlidingQueryWidth);
spread!(spread_key_rows, Coded, Hidden, SlidingKeyWidth);
spread!(dense_query_rows, Dense, LargeHidden, LargeQueries);
spread!(dense_gate_rows, Dense, LargeHidden, LargeGate);

type SlidingHeads = <Group as Attention>::Heads;
type SlidingGroups = <Group as Attention>::Groups;
type SlidingHead = <Group as Attention>::Head;

axes![
    LargeHeads = 8,
    LargeGroups = 4,
    LargeHead = 128,
    LargeKeys = 1024
];

/// One projection's rows alone split into the heads a decoder's query, key or value view reads:
/// the live rows a count names of `input`, `Heads` heads of `Groups` query heads each.
macro_rules! heads {
    ($name:ident, $width:ty, $heads:ty, $groups:ty, $head:ty) => {
        #[device(chip = 1)]
        fn $name(
            device: &mut Device,
            rows: &Value,
            input: &Ordered<$width>,
        ) -> HbmTensor<bf16, Chip, m![Rows, $heads, $groups, $head]> {
            let live = Live::<(), Rows>::read(rows);
            let input: Live<Spread<$width, Rows, Chip>, Rows> =
                live.with(input.to_dm(&mut device.tdma));
            let split =
                furiosa_opt_models::tensor::mapping::Reshape::<m![$heads, $groups, $head]>::default(
                );
            let heads = Op::forward(&split, &mut *device, input);
            heads.value().to_hbm(&mut device.tdma)
        }
    };
}

heads!(
    sliding_query_heads,
    SlidingQueryWidth,
    SlidingHeads,
    SlidingGroups,
    SlidingHead
);
heads!(
    large_query_heads,
    LargeQueries,
    LargeHeads,
    LargeGroups,
    LargeHead
);

/// A BF16 weight of the probe's values.
async fn dense<I: AxisName, O: AxisName>(device: &mut Device, seed: u64) -> Dense<I, O> {
    let weight = (HostTensor::<bf16, m![O, I]>::from_vec(values(O::SIZE * I::SIZE, seed, 0.125)))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    furiosa_opt_models::ops::linear::Affine::new(weight, ())
}

/// A channel-scaled weight of E4M3 eighths and the per-row scales of the catalog's checkpoints.
async fn coded<I: AxisName, O: AxisName>(device: &mut Device, seed: u64) -> Coded<I, O> {
    let codes = bytes(O::SIZE * I::SIZE, seed)
        .into_iter()
        .map(|byte| f8e4m3::from_f32(f32::from(byte % 31) / 8.0 - 1.875));
    let scales = (0..O::SIZE).flat_map(|row| {
        let scale = 0.01 + (row % 17) as f32 * 0.003;
        std::iter::once(scale).chain([0.0; 7])
    });
    let weight = furiosa_opt_models::tensor::form::BlockScaled {
        weight: (HostTensor::<f8e4m3, m![O, I]>::from_vec(codes.collect::<Vec<_>>()))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        weight_scale: (HostTensor::<f32, m![O, 1 # 8]>::from_vec(scales.collect::<Vec<_>>()))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        contract: (),
        shape: std::marker::PhantomData,
    };
    furiosa_opt_models::ops::linear::Affine::new(weight, ())
}

/// The FNV-1a hash of the values' bits, to compare two builds' outputs.
fn fingerprint(values: &[bf16]) -> u64 {
    (values.iter()).fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        (hash ^ u64::from(value.to_bits())).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The published checkpoint at the hub revision these probes read.
const HUB: (&str, &str) = (
    "unsloth/gemma-4-12b-it-NVFP4",
    "b1f649734b34aa5575b03d186abd1b9be3d0d5c4",
);

/// "What is 2 + 2? Answer with just the number." in Gemma 4's chat template, in its tokens.
const PROMPT: [u32; 27] = [
    2, 105, 2364, 107, 3689, 563, 236743, 236778, 900, 236743, 236778, 236881, 25685, 607, 1164,
    506, 1548, 236761, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
];

/// The hub cache's directory of the files of checkpoint `repo` at `revision` that `include`
/// matches, fetched there once.
fn snapshot((repo, revision): (&str, &str), include: &[&str]) -> std::path::PathBuf {
    let mut command = std::process::Command::new("uvx");
    command.args(["--from", "huggingface-hub", "hf", "download"]);
    command.args([repo, "--revision", revision, "--quiet"]);
    for pattern in include {
        command.args(["--include", pattern]);
    }
    let output = command
        .stderr(std::process::Stdio::inherit())
        .output()
        .expect("uvx fetches the checkpoint");
    assert!(output.status.success(), "the {repo} download failed");
    String::from_utf8(output.stdout).unwrap().trim().into()
}

/// The published checkpoint's rows of `tokens`, each its embedding times the BF16 rounding of
/// `sqrt(hidden_size)`, rounded once, as the model's scaled embedding gives them.
fn embedded(source: &Checkpoint, tokens: &[u32]) -> Vec<bf16> {
    let table = source
        .get("model.language_model.embed_tokens.weight")
        .unwrap();
    let factor = bf16::from_f32((Hidden::SIZE as f32).sqrt()).to_f32();
    let row = Hidden::SIZE * size_of::<bf16>();
    (tokens.iter())
        .flat_map(|&token| table.data()[token as usize * row..][..row].chunks_exact(2))
        .map(|pair| {
            bf16::from_f32(
                bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32() * factor,
            )
        })
        .collect()
}

/// The softcapped logits of a final-norm `row` over the tied embedding table, in FP64.
fn logits(source: &Checkpoint, row: &[bf16], cap: f64) -> Vec<f64> {
    let table = source
        .get("model.language_model.embed_tokens.weight")
        .unwrap();
    let row: Vec<f64> = row.iter().map(|value| f64::from(value.to_f32())).collect();
    (table.data().chunks_exact(Hidden::SIZE * size_of::<bf16>()))
        .map(|entry| {
            let dot = (entry.chunks_exact(2).zip(&row))
                .map(|(pair, value)| {
                    f64::from(bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32())
                        * value
                })
                .sum::<f64>();
            cap * (dot / cap).tanh()
        })
        .collect()
}

/// The final-norm rows of `tokens` through `layers` over fresh caches, a launch of at most
/// [`Rows`] of them after the ones before: the last launch's live rows.
macro_rules! prompt {
    ($device:expr, $kernel:ident, $cache:ty, ($layers:expr, $norm:expr), $tokens:expr, ($source:expr, $spectra:expr)) => {{
        let device: &mut Device = $device;
        let tokens: &[u32] = $tokens;
        let blocks = (tokens.len() + Rows::SIZE).div_ceil(Block::SIZE) + 1;
        let fresh = |device: &mut Device, bytes: i32| {
            HbmBuffer::alloc(device, blocks * bytes as usize).unwrap()
        };
        let mut pools: Caches<$cache> = (
            std::array::from_fn(|_| {
                Pool::<sliding::Group<$cache>>::from_buffer(fresh(
                    &mut *device,
                    Pool::<sliding::Group<$cache>>::BLOCK_STRIDE,
                ))
                .unwrap()
            }),
            std::array::from_fn(|_| {
                Pool::<full::Group<$cache>>::from_buffer(fresh(
                    &mut *device,
                    Pool::<full::Group<$cache>>::BLOCK_STRIDE,
                ))
                .unwrap()
            }),
        );
        let mut last = Vec::new();
        for (call, chunk) in tokens.chunks(Rows::SIZE).enumerate() {
            let (history, live) = (call * Rows::SIZE, chunk.len());
            let contexts = (
                context_at::<sliding::Group<$cache>>(&mut *device, (history, live), &$spectra.0)
                    .await,
                context_at::<full::Group<$cache>>(&mut *device, (history, live), &$spectra.1).await,
            );
            let mut hidden = embedded($source, chunk);
            hidden.resize(Rows::SIZE * Hidden::SIZE, bf16::from_bits(POISON));
            let input = HostTensor::<bf16, m![Rows, Hidden]>::from_vec(hidden)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let count = HostTensor::<i32, m![1]>::from_vec([live as i32])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let arguments = (
                &mut *device,
                ($layers, $norm),
                (&contexts.0, &contexts.1),
                &mut pools,
                (&input, &count),
            );
            let output = launch($kernel, arguments).await.unwrap();
            last = rows(&mut *device, &output, live).await;
        }
        (last, pools)
    }};
}

/// The fraction of the keys and of the values of `tokens` positions in an E4M3 `pool` that
/// saturated, coded at the largest E4M3 magnitude.
async fn saturated<A: Attention<Cache = f8e4m3>>(
    device: &mut Device,
    pool: &Pool<A>,
    tokens: usize,
) -> [f64; 2] {
    let width = <A as Attention>::Heads::SIZE * <A as Attention>::Head::SIZE;
    let mut counts = [0usize; 2];
    for block in 0..tokens.div_ceil(Block::SIZE) {
        let codes = pool.tokens[block]
            .to_host::<Tokens<A>>(&mut device.pdma)
            .await;
        let codes = f8e4m3::to_buf(&codes.unwrap().into_vec());
        for (at, code) in codes
            .iter()
            .enumerate()
            .take((tokens - block * Block::SIZE).min(Block::SIZE) * 2 * width)
        {
            counts[at / width % 2] += usize::from(code & 0x7f == 0x7e);
        }
    }
    counts.map(|count| count as f64 / (tokens * width) as f64)
}

/// The published checkpoint's logits over caches of E4M3 codes under its static scales beside
/// the BF16 caches': for the reference prompt, one of `LOGITS_PROFILE_TOKENS` pseudo-random
/// tokens (default 960) and the token IDs of the file `LOGITS_PROFILE_IDS` names, if any; every 16th
/// row of the last launch and its last, the largest and the mean distance of each logit, the KL
/// divergence and each cache's greedy token.
#[tokio::test(flavor = "current_thread")]
#[ignore = "reads the published checkpoint from the local hub cache"]
async fn logits_profile() {
    use furiosa_opt_models::Then;
    use furiosa_opt_models::models::config::scaling::Spectrum;
    use furiosa_opt_models::modules::language::Backbone;
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(decoded_bf16.topology()).unwrap();
    let source = Checkpoint::new(snapshot(HUB, &["*.json", "*.safetensors"]), None).unwrap();
    let text = &source.config()["text_config"];
    let cap = text["final_logit_softcapping"].as_f64().unwrap();
    let spectrum = |layer: &str| -> Spectrum {
        let parameters = serde_json::json!({"rope_parameters": text["rope_parameters"][layer]});
        serde_json::from_value(parameters).unwrap()
    };
    let spectra = (
        spectrum("sliding_attention")
            .bind::<sliding::Head, sliding::Head>()
            .unwrap(),
        spectrum("full_attention")
            .bind::<full::Head, full::Rotated>()
            .unwrap(),
    );
    let plain: gemma4::ForCausalLM<gemma4::Quantized, bf16> = source.load(&mut device).unwrap();
    let coded: gemma4::ForCausalLM<gemma4::Quantized, f8e4m3> = source.load(&mut device).unwrap();
    let Then(
        Backbone {
            body: plain_layers, ..
        },
        plain_norm,
    ) = &plain.model.body;
    let Then(
        Backbone {
            body: coded_layers, ..
        },
        coded_norm,
    ) = &coded.model.body;
    let length =
        std::env::var("LOGITS_PROFILE_TOKENS").map_or(960, |tokens| tokens.parse().unwrap());
    let longer: Vec<u32> = bytes(4 * length, 0x7f4a_7c15)
        .chunks_exact(4)
        .map(|word| 1000 + u32::from_le_bytes(word.try_into().unwrap()) % 199_000)
        .collect();
    // A text's token IDs, whitespace-separated, as a third prompt.
    let text: Vec<u32> = std::env::var("LOGITS_PROFILE_IDS").map_or(Vec::new(), |path| {
        let ids = std::fs::read_to_string(path).unwrap();
        ids.split_whitespace()
            .map(|id| id.parse().unwrap())
            .collect()
    });
    let prompts = [
        ("reference", &PROMPT[..]),
        ("longer", &longer[..]),
        ("text", &text[..]),
    ];
    for (name, tokens) in prompts.into_iter().filter(|(_, tokens)| !tokens.is_empty()) {
        let started = std::time::Instant::now();
        let (want, _) = prompt!(
            &mut device,
            decoded_bf16,
            bf16,
            (plain_layers, plain_norm),
            tokens,
            (&source, spectra)
        );
        let (found, pools) = prompt!(
            &mut device,
            decoded_e4m3,
            f8e4m3,
            (coded_layers, coded_norm),
            tokens,
            (&source, spectra)
        );
        let (mut sliding, mut full) = ([0.0f64; 2], [0.0f64; 2]);
        for pool in &pools.0 {
            let [keys, values] =
                saturated::<sliding::Group<f8e4m3>>(&mut device, pool, tokens.len()).await;
            sliding = [sliding[0].max(keys), sliding[1].max(values)];
        }
        for pool in &pools.1 {
            let [keys, values] =
                saturated::<full::Group<f8e4m3>>(&mut device, pool, tokens.len()).await;
            full = [full[0].max(keys), full[1].max(values)];
        }
        println!(
            "{name} saturated codes, the most of a layer: sliding keys {:.2e} values {:.2e}, full keys {:.2e} values {:.2e}",
            sliding[0], sliding[1], full[0], full[1],
        );
        let live = found.len() / Hidden::SIZE;
        let (worst, mean) = deviation(&found, &want);
        println!(
            "{name} {} tokens: final rows largest {worst:.5} mean {mean:.6} ({:.1} s)",
            tokens.len(),
            started.elapsed().as_secs_f64()
        );
        for row in (0..live).filter(|row| row % 16 == 0 || row + 1 == live) {
            let row_of = |rows: &[bf16]| rows[row * Hidden::SIZE..][..Hidden::SIZE].to_vec();
            let (plain, coded) = (
                logits(&source, &row_of(&want), cap),
                logits(&source, &row_of(&found), cap),
            );
            let distances: Vec<f64> = plain
                .iter()
                .zip(&coded)
                .map(|(p, c)| (p - c).abs())
                .collect();
            let largest = distances.iter().copied().fold(0.0, f64::max);
            let average = distances.iter().sum::<f64>() / distances.len() as f64;
            let argmax = |values: &[f64]| {
                (0..values.len())
                    .max_by(|&a, &b| values[a].total_cmp(&values[b]))
                    .unwrap()
            };
            let softmax = |values: &[f64]| {
                let top = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = values.iter().map(|value| (value - top).exp()).collect();
                let total: f64 = weights.iter().sum();
                weights
                    .into_iter()
                    .map(|weight| weight / total)
                    .collect::<Vec<_>>()
            };
            let (p, q) = (softmax(&plain), softmax(&coded));
            let kl: f64 = p
                .iter()
                .zip(&q)
                .filter(|(p, _)| **p > 0.0)
                .map(|(p, q)| p * (p / q).ln())
                .sum();
            let (a, b) = (argmax(&plain), argmax(&coded));
            let mut sorted = plain.clone();
            sorted.sort_by(|x, y| y.total_cmp(x));
            println!(
                "{name} row {}: logit largest {largest:.4} mean {average:.5}, KL {kl:.3e}, greedy {a} / {b}, BF16 top {:.3} margin {:.3}",
                tokens.len() - live + row,
                sorted[0],
                sorted[0] - sorted[1],
            );
        }
    }
}

/// The catalog's attention projections alone, profiled as [`period_profile`] profiles the period:
/// for each live row count `PERIOD_PROFILE_ROWS` lists, a warm-up and `PERIOD_PROFILE_STEPS`
/// launches of each, then the hash of its last launch's live rows.
#[tokio::test(flavor = "current_thread")]
#[ignore = "profiles launches on a device"]
async fn projection_profile() {
    use std::future::IntoFuture;
    use tracing::Instrument;
    let list = |name: &str, default: &str| -> Vec<usize> {
        let list = std::env::var(name).unwrap_or_else(|_| default.into());
        list.split(',').map(|item| item.parse().unwrap()).collect()
    };
    let steps = list("PERIOD_PROFILE_STEPS", "10")[0];
    let _spans = tracing::subscriber::set_default(Lines::new());
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(sliding_query_rows.topology()).unwrap();
    let sliding_query = coded::<Hidden, SlidingQueryWidth>(&mut device, 0x11).await;
    let sliding_key = coded::<Hidden, SlidingKeyWidth>(&mut device, 0x12).await;
    let full_query = coded::<Hidden, FullQueryWidth>(&mut device, 0x13).await;
    let full_key = coded::<Hidden, FullKeyWidth>(&mut device, 0x14).await;
    let sliding_output = coded::<SlidingQueryWidth, Hidden>(&mut device, 0x15).await;
    let small_query = coded::<SmallHidden, SmallQueries>(&mut device, 0x16).await;
    let small_key = coded::<SmallHidden, SmallKeys>(&mut device, 0x17).await;
    let middle_query = coded::<MiddleHidden, MiddleQueries>(&mut device, 0x18).await;
    let middle_key = coded::<MiddleHidden, MiddleKeys>(&mut device, 0x19).await;
    let large_gate_up = coded::<LargeHidden, LargeGateUp>(&mut device, 0x1a).await;
    let dense_query = dense::<LargeHidden, LargeQueries>(&mut device, 0x1b).await;
    let dense_gate = dense::<LargeHidden, LargeGate>(&mut device, 0x1c).await;
    let full_output = coded::<FullQueryWidth, Hidden>(&mut device, 0x1d).await;
    let middle_output = coded::<MiddleQueries, MiddleHidden>(&mut device, 0x1e).await;
    let small_gate_up = coded::<SmallHidden, SmallGateUp>(&mut device, 0x1f).await;
    let middle_gate_up = coded::<MiddleHidden, MiddleGateUp>(&mut device, 0x2b).await;
    let middle_down = coded::<MiddleIntermediate, MiddleHidden>(&mut device, 0x2c).await;
    macro_rules! profile {
        ($live:expr, $count:expr, $kernel:ident, $weight:expr, $input:expr, $out:ty) => {{
            let case = format!("{}-rows-{}", stringify!($kernel), $live);
            let mut last = None;
            for sample in 0..=steps {
                let span = match sample {
                    0 => tracing::info_span!("warmup", case = case.as_str()),
                    step => tracing::info_span!("forward", case = case.as_str(), step),
                };
                let output = async {
                    let arguments = (&mut device, $weight, $count, $input);
                    let output = (launch($kernel, arguments).into_future())
                        .instrument(tracing::info_span!("submit"))
                        .await
                        .unwrap();
                    (device.synchronize())
                        .instrument(tracing::info_span!("wait"))
                        .await
                        .unwrap();
                    output
                }
                .instrument(span)
                .await;
                last = Some(output);
            }
            let output = last
                .unwrap()
                .to_host::<m![Rows, $out]>(&mut device.pdma)
                .await;
            let output = output.unwrap().into_vec();
            let hash = fingerprint(&output[..$live * <$out>::SIZE]);
            println!("hash {case} {hash:016x}");
        }};
    }
    macro_rules! profile_heads {
        ($live:expr, $count:expr, $kernel:ident, $input:expr, $out:ty, $width:expr) => {{
            let case = format!("{}-rows-{}", stringify!($kernel), $live);
            let mut last = None;
            for sample in 0..=steps {
                let span = match sample {
                    0 => tracing::info_span!("warmup", case = case.as_str()),
                    step => tracing::info_span!("forward", case = case.as_str(), step),
                };
                let output = async {
                    let arguments = (&mut device, $count, $input);
                    let output = (launch($kernel, arguments).into_future())
                        .instrument(tracing::info_span!("submit"))
                        .await
                        .unwrap();
                    (device.synchronize())
                        .instrument(tracing::info_span!("wait"))
                        .await
                        .unwrap();
                    output
                }
                .instrument(span)
                .await;
                last = Some(output);
            }
            let output = last.unwrap().to_host::<$out>(&mut device.pdma).await;
            let output = output.unwrap().into_vec();
            let hash = fingerprint(&output[..$live * $width]);
            println!("hash {case} {hash:016x}");
        }};
    }
    for live in list("PERIOD_PROFILE_ROWS", "1,8,9,64") {
        let count: Value = (HostTensor::<i32, m![1]>::from_vec([live as i32]))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let grouped: Grouped<Hidden> =
            (HostTensor::<bf16, m![Rows, Hidden]>::from_vec(hidden(live)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let attended = (0..Rows::SIZE * SlidingQueryWidth::SIZE).map(|at| {
            let value = ((at * 29 + at / SlidingQueryWidth::SIZE * 13) % 41) as f32 / 32.0 - 0.625;
            bf16::from_f32(value)
        });
        let attended: Ordered<SlidingQueryWidth> =
            (HostTensor::<bf16, m![Rows, SlidingQueryWidth]>::from_vec(
                attended.collect::<Vec<_>>(),
            ))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let count = &count;
        profile!(
            live,
            count,
            sliding_query_rows,
            &sliding_query,
            &grouped,
            SlidingQueryWidth
        );
        profile!(
            live,
            count,
            sliding_key_rows,
            &sliding_key,
            &grouped,
            SlidingKeyWidth
        );
        profile!(
            live,
            count,
            full_query_rows,
            &full_query,
            &grouped,
            FullQueryWidth
        );
        profile!(
            live,
            count,
            full_key_rows,
            &full_key,
            &grouped,
            FullKeyWidth
        );
        profile!(
            live,
            count,
            sliding_output_rows,
            &sliding_output,
            &attended,
            Hidden
        );
        let rows = |width: usize, seed: u64| {
            (values(Rows::SIZE * width, seed, 4.0)
                .into_iter()
                .enumerate())
            .map(|(at, value)| match at / width < live {
                true => value,
                false => bf16::from_bits(POISON),
            })
            .collect::<Vec<_>>()
        };
        let small: Grouped<SmallHidden> =
            (HostTensor::<bf16, m![Rows, SmallHidden]>::from_vec(rows(SmallHidden::SIZE, 0x21)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let middle: Grouped<MiddleHidden> =
            (HostTensor::<bf16, m![Rows, MiddleHidden]>::from_vec(rows(MiddleHidden::SIZE, 0x22)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let large: Ordered<LargeHidden> =
            (HostTensor::<bf16, m![Rows, LargeHidden]>::from_vec(rows(LargeHidden::SIZE, 0x23)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        profile!(
            live,
            count,
            small_query_rows,
            &small_query,
            &small,
            SmallQueries
        );
        profile!(live, count, small_key_rows, &small_key, &small, SmallKeys);
        profile!(
            live,
            count,
            middle_query_rows,
            &middle_query,
            &middle,
            MiddleQueries
        );
        profile!(
            live,
            count,
            middle_key_rows,
            &middle_key,
            &middle,
            MiddleKeys
        );
        profile!(
            live,
            count,
            large_gate_up_rows,
            &large_gate_up,
            &large,
            LargeGateUp
        );
        let ordered: Ordered<Hidden> =
            (HostTensor::<bf16, m![Rows, Hidden]>::from_vec(hidden(live)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        profile!(
            live,
            count,
            spread_query_rows,
            &sliding_query,
            &ordered,
            SlidingQueryWidth
        );
        profile!(
            live,
            count,
            spread_key_rows,
            &sliding_key,
            &ordered,
            SlidingKeyWidth
        );
        profile!(
            live,
            count,
            dense_query_rows,
            &dense_query,
            &large,
            LargeQueries
        );
        profile!(live, count, dense_gate_rows, &dense_gate, &large, LargeGate);
        let queries: Ordered<SlidingQueryWidth> =
            (HostTensor::<bf16, m![Rows, SlidingQueryWidth]>::from_vec(rows(4096, 0x24)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let wide: Ordered<LargeQueries> =
            (HostTensor::<bf16, m![Rows, LargeQueries]>::from_vec(rows(4096, 0x25)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        type SlidingRows = m![Rows, SlidingHeads, SlidingGroups, SlidingHead];
        type LargeRows = m![Rows, LargeHeads, LargeGroups, LargeHead];
        profile_heads!(
            live,
            count,
            sliding_query_heads,
            &queries,
            SlidingRows,
            4096
        );
        profile_heads!(live, count, large_query_heads, &wide, LargeRows, 4096);
        let full_attended: Ordered<FullQueryWidth> =
            (HostTensor::<bf16, m![Rows, FullQueryWidth]>::from_vec(rows(8192, 0x26)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        profile!(
            live,
            count,
            full_output_rows,
            &full_output,
            &full_attended,
            Hidden
        );
        let middle_attended: Ordered<MiddleQueries> =
            (HostTensor::<bf16, m![Rows, MiddleQueries]>::from_vec(rows(4096, 0x27)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        profile!(
            live,
            count,
            middle_output_rows,
            &middle_output,
            &middle_attended,
            MiddleHidden
        );
        let small_ordered: Ordered<SmallHidden> =
            (HostTensor::<bf16, m![Rows, SmallHidden]>::from_vec(rows(1024, 0x28)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        profile!(
            live,
            count,
            small_gate_up_rows,
            &small_gate_up,
            &small_ordered,
            SmallGateUp
        );
        let middle_ordered: Ordered<MiddleHidden> =
            (HostTensor::<bf16, m![Rows, MiddleHidden]>::from_vec(rows(2560, 0x29)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        profile!(
            live,
            count,
            middle_gate_up_rows,
            &middle_gate_up,
            &middle_ordered,
            MiddleGateUp
        );
        let product: Ordered<MiddleIntermediate> =
            (HostTensor::<bf16, m![Rows, MiddleIntermediate]>::from_vec(rows(9728, 0x2a)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        profile!(
            live,
            count,
            middle_down_rows,
            &middle_down,
            &product,
            MiddleHidden
        );
    }
}

/// Records every span to `PERIOD_PROFILE_OUT` (default `npu.tsv`), a line each: `S id parent name
/// ns fields` as it opens (an NPU span's fields carry its cycles) and `C id ns` as it closes; a
/// kernel's name, written once on a `K label bytes name` line, is its label in the spans.
struct Lines {
    out: std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
    base: std::time::Instant,
    spans: std::sync::Mutex<Spans>,
}

/// The spans [`Lines`] tracks: the last id, each open span's handles, the spans each thread is
/// in, and the labels of the kernels named so far.
#[derive(Default)]
struct Spans {
    last: u64,
    handles: std::collections::HashMap<u64, usize>,
    entered: std::collections::HashMap<std::thread::ThreadId, Vec<u64>>,
    kernels: std::collections::HashSet<String>,
}

impl Lines {
    fn new() -> Self {
        let path = std::env::var("PERIOD_PROFILE_OUT").unwrap_or_else(|_| "npu.tsv".into());
        let out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        Self {
            out: std::sync::Mutex::new(out),
            base: std::time::Instant::now(),
            spans: std::sync::Mutex::default(),
        }
    }

    fn write(&self, line: &str) {
        use std::io::Write;
        self.out.lock().unwrap().write_all(line.as_bytes()).unwrap();
    }
}

impl tracing::Subscriber for Lines {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, attributes: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        let mut spans = self.spans.lock().unwrap();
        spans.last += 1;
        let id = spans.last;
        spans.handles.insert(id, 1);
        let thread = std::thread::current().id();
        let parent = match attributes.parent() {
            Some(parent) => parent.into_u64(),
            None if attributes.is_contextual() => (spans.entered.get(&thread))
                .and_then(|stack| stack.last().copied())
                .unwrap_or(0),
            None => 0,
        };
        let mut fields = Fields {
            text: String::new(),
            kernels: &mut spans.kernels,
            named: String::new(),
        };
        attributes.record(&mut fields);
        let line = format!(
            "{}S\t{id}\t{parent}\t{}\t{}{}\n",
            fields.named,
            attributes.metadata().name(),
            self.base.elapsed().as_nanos(),
            fields.text
        );
        drop(spans);
        self.write(&line);
        tracing::span::Id::from_u64(id)
    }

    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

    fn event(&self, _: &tracing::Event<'_>) {}

    fn enter(&self, span: &tracing::span::Id) {
        let mut spans = self.spans.lock().unwrap();
        let stack = spans.entered.entry(std::thread::current().id());
        stack.or_default().push(span.into_u64());
    }

    fn exit(&self, span: &tracing::span::Id) {
        let mut spans = self.spans.lock().unwrap();
        if let Some(stack) = spans.entered.get_mut(&std::thread::current().id())
            && let Some(at) = stack.iter().rposition(|&id| id == span.into_u64())
        {
            stack.remove(at);
        }
    }

    fn clone_span(&self, span: &tracing::span::Id) -> tracing::span::Id {
        let mut spans = self.spans.lock().unwrap();
        *spans.handles.entry(span.into_u64()).or_default() += 1;
        span.clone()
    }

    fn try_close(&self, span: tracing::span::Id) -> bool {
        let id = span.into_u64();
        let mut spans = self.spans.lock().unwrap();
        let Some(handles) = spans.handles.get_mut(&id) else {
            return false;
        };
        *handles -= 1;
        if *handles > 0 {
            return false;
        }
        spans.handles.remove(&id);
        drop(spans);
        self.write(&format!("C\t{id}\t{}\n", self.base.elapsed().as_nanos()));
        true
    }
}

/// A span's fields as `\tname=value` text, a kernel's name replaced by its label (the last path
/// segment before any generic argument), and the `K` line of a kernel it names first.
struct Fields<'a> {
    text: String,
    kernels: &'a mut std::collections::HashSet<String>,
    named: String,
}

impl tracing::field::Visit for Fields<'_> {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() != "kernel" {
            self.text.push_str(&format!("\t{}={value}", field.name()));
            return;
        }
        let path = value.split('<').next().unwrap_or(value);
        let label = path.rsplit("::").next().unwrap_or(path).to_owned();
        if self.kernels.insert(label.clone()) {
            let name: String = value.chars().take(400).collect();
            self.named = format!("K\t{label}\t{}\t{name}\n", value.len());
        }
        self.text.push_str(&format!("\tkernel={label}"));
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let text = format!("{value:?}");
        self.record_str(field, text.trim_matches('"'));
    }
}
