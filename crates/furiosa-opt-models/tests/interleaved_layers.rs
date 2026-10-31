//! The Gemma 4 catalog's sliding layers interleaved in two collections, one layer of each a
//! period, over a call of rows that attend a history in their layers' caches.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::load::Checkpoint;
use furiosa_opt_models::models::decoder::Attention;
use furiosa_opt_models::models::decoder::attention::gemma4::{self, sliding};
use furiosa_opt_models::modules::{attention, language};
use furiosa_opt_models::ops::attention::gqa::{Cache, Pages};
use furiosa_opt_models::ops::index::table::{BlockIndex, PADDING_BIAS};
use furiosa_opt_models::ops::normalization::Spread;
use furiosa_opt_models::ops::rotary::{PairFrequencies, Rotation};
use furiosa_opt_models::tensor::cache::{Block, Blocks, KeyValue};
use furiosa_opt_models::tensor::live::Live;
use furiosa_opt_models::tensor::memory::Hbm;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_models::{Branch, Interleave, Module};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::{HbmBuffer, launch};
use safetensors::Dtype;

type Rows = gemma4::Rows;
type Hidden = gemma4::Hidden;
type Intermediate = gemma4::Intermediate;

/// The layer types of the catalog's two collections.
trait Collections {
    type Sliding;
}

impl<E, S, F, N, H> Collections
    for language::Hybrid<E, [S; gemma4::SLIDING], [F; gemma4::FULL], N, H, { gemma4::ORDER }>
{
    type Sliding = S;
}

type Catalog = gemma4::Catalog<gemma4::Quantized, bf16, Hbm>;
type Sliding = <Catalog as Collections>::Sliding;

type Group = sliding::Group;
type Head = <Group as Attention>::Head;
type Heads = <Group as Attention>::Heads;
type Rotated = <Group as Attention>::Rotated;
type Windows = <Group as Attention>::Windows;

/// One sliding layer of each collection a period, so every other edge crosses a period.
const PAIRS: &[Branch] = &[Branch::First, Branch::Second, Branch::First, Branch::Second];

/// Two collections of two sliding layers each, interleaved.
type Layers = Interleave<([Sliding; 2], [Sliding; 2]), PAIRS>;

/// A call's context of the sliding group: its rows' rotation, write slots and paged reads.
type Context = attention::Context<
    Rotation<Rotated, Chip, Rows>,
    HbmTensor<i32, Chip, m![Rows]>,
    Pages<Rows, Windows, Chip>,
>;

type Pool = Blocks<Heads, Head>;
type Pools = ([Pool; 2], [Pool; 2]);

/// The call's hidden rows.
type HiddenRows = HbmTensor<bf16, Chip, m![Rows, Hidden]>;

/// A sliding layer's state: its attention's slots, its cache blocks, its MLP's slots.
type Qkv = ((((), ()), ((), ()), (((), ()), ())), (), ());
type Units = ((((), ()), ()), ((), ()));
type State<'b> = ((((Qkv, &'b mut Pool, ()), ()), (Units, ())), ());

fn state(pool: &mut Pool) -> State<'_> {
    let qkv = ((((), ()), ((), ()), (((), ()), ())), (), ());
    let units = ((((), ()), ()), ((), ()));
    ((((qkv, pool, ()), ()), (units, ())), ())
}

/// The layers' sequential forward over the call's live rows, each layer on its own cache.
#[device(chip = 1)]
fn sequential(
    device: &mut Device,
    layers: &Layers,
    context: &Context,
    pools: &mut Pools,
    (hidden, live): (&HiddenRows, &Value),
) -> HiddenRows {
    let live = Live::<(), Rows>::read(live);
    let rows: Live<Spread<Hidden, Rows, Chip>, Rows> = live.with(hidden.to_dm(&mut device.tdma));
    let ([a, b], [c, d]) = pools;
    let mut states = ([state(a), state(b)], [state(c), state(d)]);
    let rows: Live<Spread<Hidden, Rows, Chip>, Rows> =
        Module::forward(layers, &mut *device, (context, context), &mut states, rows);
    rows.value().to_hbm(&mut device.tdma)
}

/// Tokens of history before the call's first row, past a block boundary.
const HISTORY: usize = 100;

/// Physical blocks of each layer's cache: the history and every row of the call.
const BLOCKS: usize = (HISTORY + Rows::SIZE).div_ceil(Block::SIZE) + 1;

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

/// The checkpoint of sliding layer `layer`: every weight its tree reads, under its names, each
/// layer's own.
fn checkpoint(layer: usize) -> Checkpoint {
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
    for (tensor, norm) in ["q_norm", "k_norm"].into_iter().enumerate() {
        let name = format!("self_attn.{norm}.weight");
        insert(
            &name,
            Dtype::BF16,
            &[Head::SIZE],
            gains(Head::SIZE, 4 + tensor as u64),
        );
    }
    let query = <Group as Attention>::QueryWidth::SIZE;
    let kv = <Group as Attention>::KvWidth::SIZE;
    let channels = [
        ("q_proj", query, Hidden::SIZE),
        ("k_proj", kv, Hidden::SIZE),
        ("v_proj", kv, Hidden::SIZE),
        ("o_proj", Hidden::SIZE, query),
    ];
    for (tensor, (name, rows, columns)) in channels.into_iter().enumerate() {
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

/// The keys a row of the call sees: the history and every row up to its own.
fn visible(row: usize, live: usize) -> usize {
    if row < live { HISTORY + row + 1 } else { 0 }
}

/// The call's context: each row's rotation at its position, its slot after the history and the
/// pages of the history and the rows before it, in the blocks of a layer's cache.
async fn context(device: &mut Device, live: usize) -> Context {
    let (rotated, pairs) = (Rotated::SIZE, Rotated::SIZE / 2);
    let frequencies = PairFrequencies::<Rotated>::unscaled::<Head>(1e4).unwrap();
    let (mut low, mut high) = (Vec::new(), Vec::new());
    for row in 0..Rows::SIZE {
        let position = (HISTORY + row) as f32;
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
    let table = |values: Vec<bf16>| HostTensor::<bf16, m![Rows, Rotated]>::from_vec(values);
    let rotation = Rotation {
        low: table(low).to_hbm(&mut device.pdma).await.unwrap(),
        high: table(high).to_hbm(&mut device.pdma).await.unwrap(),
    };
    let slots = (0..Rows::SIZE).map(|row| match row < live {
        true => ((HISTORY + row) as i32) * Pool::ROW_STRIDE,
        false => 0,
    });
    let write = HostTensor::<i32, m![Rows]>::from_vec(slots.collect::<Vec<_>>());
    let (windows, entries) = (Windows::SIZE, BlockIndex::SIZE);
    let table: Vec<i32> = (0..Rows::SIZE * windows * entries)
        .map(|at| {
            let (row, entry) = (at / (windows * entries), at % (windows * entries));
            let used = visible(row, live).div_ceil(Block::SIZE).max(1);
            entry.min(used - 1) as i32 * Pool::BLOCK_STRIDE
        })
        .collect();
    let counts: Vec<i32> = (0..windows * Rows::SIZE)
        .map(|at| {
            let (window, row) = (at / Rows::SIZE, at % Rows::SIZE);
            let seen = visible(row, live).saturating_sub(window * Cache::SIZE);
            seen.min(Cache::SIZE).div_ceil(Block::SIZE) as i32
        })
        .collect();
    let bias: Vec<f32> = (0..Rows::SIZE * windows * Cache::SIZE)
        .map(|at| {
            let (row, key) = (at / (windows * Cache::SIZE), at % (windows * Cache::SIZE));
            if key < visible(row, live) {
                0.0
            } else {
                PADDING_BIAS
            }
        })
        .collect();
    let rest = (0..live)
        .map(|row| (visible(row, live) - 1) / Cache::SIZE)
        .max();
    let rest = HostTensor::<i32, m![1]>::from_vec([rest.unwrap_or(0) as i32]);
    let read = Pages {
        table: HostTensor::<i32, m![Rows, Windows, BlockIndex]>::from_vec(table)
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
    };
    attention::Context {
        positions: rotation,
        write: write.to_hbm(&mut device.pdma).await.unwrap(),
        read,
    }
}

/// Tokens a block holds, each its key and value.
type Tokens = m![Block, KeyValue, Heads, Head];

/// Layer `layer`'s cache: the history's keys and values, every later slot poison.
fn cache(layer: usize) -> Vec<bf16> {
    let width = KeyValue::SIZE * Heads::SIZE * Head::SIZE;
    let history = values(HISTORY * width, 0x2545_f491 ^ (layer as u64) << 32, 0.25);
    let poison = bf16::from_bits(POISON);
    (0..BLOCKS * Block::SIZE * width)
        .map(|at| history.get(at).copied().unwrap_or(poison))
        .collect()
}

/// `values` in a new cache of [`BLOCKS`] blocks.
async fn pool(device: &mut Device, values: &[bf16]) -> Pool {
    let bytes = BLOCKS * Pool::BLOCK_STRIDE as usize;
    let mut pool = Pool::from_buffer(HbmBuffer::alloc(device, bytes).unwrap()).unwrap();
    for (block, values) in values.chunks_exact(values.len() / BLOCKS).enumerate() {
        HostTensor::<bf16, Tokens>::from_vec(values.to_vec())
            .to_hbm(&mut device.pdma)
            .output(&mut pool.tokens[block])
            .await
            .unwrap();
    }
    pool
}

/// Every value of `pool`, block by block.
async fn read(device: &mut Device, pool: &Pool) -> Vec<bf16> {
    let mut values = Vec::new();
    for block in 0..BLOCKS {
        let tokens = pool.tokens[block].to_host::<Tokens>(&mut device.pdma).await;
        values.extend(tokens.unwrap().into_vec());
    }
    values
}

/// The four layers, each its own weights, in checkpoint order of their collections.
fn layers(device: &mut Device) -> Layers {
    let mut layer = |at: usize| -> Sliding { checkpoint(at).load(device).unwrap() };
    let first = [layer(0), layer(2)];
    let second = [layer(1), layer(3)];
    Interleave {
        layers: (first, second),
    }
}

/// The sequential forward of the four layers writes every live row's keys and values into each
/// layer's cache and leaves finite rows; rows past the live ones and the history stay as they were.
#[tokio::test(flavor = "current_thread")]
async fn runs_the_layers_sequentially() {
    let mut device = Device::new(sequential.topology()).unwrap();
    let layers = layers(&mut device);
    let caches: Vec<Vec<bf16>> = (0..4).map(cache).collect();
    for live in [Rows::SIZE, 9] {
        let context = context(&mut device, live).await;
        let mut pools = (
            [
                pool(&mut device, &caches[0]).await,
                pool(&mut device, &caches[2]).await,
            ],
            [
                pool(&mut device, &caches[1]).await,
                pool(&mut device, &caches[3]).await,
            ],
        );
        let input = HostTensor::<bf16, m![Rows, Hidden]>::from_vec(hidden(live));
        let input: HiddenRows = input.to_hbm(&mut device.pdma).await.unwrap();
        let count: Value = HostTensor::<i32, m![1]>::from_vec([live as i32])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let arguments = (&mut device, &layers, &context, &mut pools, (&input, &count));
        let output = launch(sequential, arguments).await.unwrap();
        let output = output
            .to_host::<m![Rows, Hidden]>(&mut device.pdma)
            .await
            .unwrap();
        let output = output.into_vec();
        for (at, value) in output[..live * Hidden::SIZE].iter().enumerate() {
            assert!(
                value.to_f32().is_finite(),
                "{live} rows: element {at} is {}",
                value.to_f32()
            );
        }
        let width = KeyValue::SIZE * Heads::SIZE * Head::SIZE;
        let ([a, b], [c, d]) = &pools;
        for (layer, pool) in [(0, a), (2, b), (1, c), (3, d)] {
            let found = read(&mut device, pool).await;
            let before = &caches[layer];
            for (at, (found, before)) in found.iter().zip(before).enumerate() {
                let token = at / width;
                let written = (HISTORY..HISTORY + live).contains(&token);
                match written {
                    true => assert!(
                        found.to_f32().is_finite(),
                        "layer {layer}, {live} rows: token {token}"
                    ),
                    false => assert_eq!(
                        found.to_bits(),
                        before.to_bits(),
                        "layer {layer}, {live} rows: token {token}"
                    ),
                }
            }
        }
    }
}
