//! The Gemma 4 catalog's full attention layers on a launch's live rows: the catalog's query and
//! key projections, head views and learned head norms, then its position op, over the rows
//! attention reads.
//!
//! - Live rows start anywhere in the envelope; each turns by its own envelope row's rotation.
//! - The projections quantize each live row to E4M3 under a scale of its own, as the checkpoint
//!   declares their inputs.
//! - The rotated coordinates of a live row are the products of its normalized heads in FP32
//!   rounded once; every row's other coordinates keep the normalized heads' bits.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::load::Checkpoint;
use furiosa_opt_models::models::decoder::Attention;
use furiosa_opt_models::models::decoder::attention::gemma4::{self, full};
use furiosa_opt_models::modules::{attention, language, transformer};
use furiosa_opt_models::ops::attention::SplitHeads;
use furiosa_opt_models::ops::normalization::Spread;
use furiosa_opt_models::ops::rotary::{PairFrequencies, Rotation};
use furiosa_opt_models::tensor::axis::{self, Single};
use furiosa_opt_models::tensor::live::Live;
use furiosa_opt_models::tensor::memory::Hbm;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_models::{Module, Op, Then};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;
use safetensors::Dtype;

type Rows = axis::Rows<64>;
type Hidden = gemma4::Hidden;
type Head = full::Head;
type Rotated = full::Rotated;
type Heads = <full::Group as Attention>::Heads;
type Groups = <full::Group as Attention>::Groups;

/// The query/key/value stage and the position op of a hybrid model's full attention layers.
trait FullLayers {
    type Qkv;
    type Position;
}

impl<E, S, Q, P, W, R, A, O, F, N, D, L, Norm, H> FullLayers
    for language::Hybrid<
        E,
        S,
        [Then<transformer::SandwichNorm<attention::Cached<Q, P, W, R, A, O>, F, N, D>, L>;
            gemma4::FULL],
        Norm,
        H,
        { gemma4::ORDER },
    >
{
    type Qkv = Q;
    type Position = P;
}

type Catalog = gemma4::Catalog<gemma4::Quantized, bf16, Hbm>;
type Qkv = <Catalog as FullLayers>::Qkv;
type Position = <Catalog as FullLayers>::Position;

/// `Group` heads of each key/value head on every live row, as the projections' split leaves them.
type HeadRows<Group> = Live<SplitHeads<Rows, Heads, Group, Head>, Rows>;

type QueryRows = m![Rows, Heads, Groups, Head];
type KeyRows = m![Rows, Heads, Head];
type QueryWidth = <full::Group as Attention>::QueryWidth;
type KvWidth = <full::Group as Attention>::KvWidth;

/// Normalized query and key heads, then the same heads positioned, each as HBM rows.
type Positioned = (
    (
        HbmTensor<bf16, Chip, QueryRows>,
        HbmTensor<bf16, Chip, KeyRows>,
    ),
    (
        HbmTensor<bf16, Chip, QueryRows>,
        HbmTensor<bf16, Chip, KeyRows>,
    ),
);

/// The catalog stage's head norms, then its position op, on query and key head rows.
fn position(
    device: &mut Device,
    (qkv, positions): (&Qkv, &Position),
    (query, key): (HeadRows<Groups>, HeadRows<Single>),
    rotation: &Rotation<Rotated, Chip, Rows>,
) -> Positioned {
    let query: HeadRows<Groups> =
        Module::forward(&qkv.query_norm, &mut *device, (), &mut (), &query);
    let key: HeadRows<Single> = Module::forward(&qkv.key_norm, &mut *device, (), &mut (), &key);
    let normalized = (
        query.value().to_hbm(&mut device.tdma),
        key.value().to_hbm(&mut device.tdma),
    );
    let (query, key) = Op::forward(positions, &mut *device, ((query, key), rotation));
    let positioned = (
        query.value().to_hbm(&mut device.tdma),
        key.value().to_hbm(&mut device.tdma),
    );
    (normalized, positioned)
}

/// The live rows `start..start + rows` positioned, the queries and keys projected by the stage
/// from `hidden`.
#[device(chip = 1)]
fn from_hidden(
    device: &mut Device,
    stage: (&Qkv, &Position),
    hidden: &HbmTensor<bf16, Chip, m![Rows, Hidden]>,
    (rotation, start, rows): (&Rotation<Rotated, Chip, Rows>, &Value, &Value),
) -> Positioned {
    let live =
        Live::<(), Rows>::all().subrange(start.extent::<m![Rows]>(), rows.extent::<m![Rows]>());
    let hidden: Live<Spread<Hidden, Rows, Chip>, Rows> = live.with(hidden.to_dm(&mut device.tdma));
    let projection = &stage.0.projection;
    let query = Module::forward(&projection.query, &mut *device, (), &mut (), &hidden);
    let key = Module::forward(&projection.key, &mut *device, (), &mut (), &hidden);
    position(device, stage, (query, key), rotation)
}

/// A value no live row reads: a NaN with a payload of its own.
const POISON: u16 = 0xff81;

/// An E4M3 code of eighths in [-1.875, 1.875], each exact.
fn code(row: usize, column: usize) -> f32 {
    ((row * 7 + column * 3 + (row * column) % 13) % 31) as f32 / 8.0 - 1.875
}

/// A projection's per-channel FP32 scale.
fn scale(row: usize) -> f32 {
    0.01 + (row % 17) as f32 * 0.003
}

/// A head norm's gain, exact in BF16.
fn gain(channel: usize) -> f32 {
    0.5 + (channel % 7) as f32 / 8.0
}

/// A hidden feature of a live row, exact in BF16.
fn feature(row: usize, column: usize) -> f32 {
    ((row * 29 + column * 13 + (column * column) % 23) % 41) as f32 / 32.0 - 0.625
}

/// The checkpoint of the full layers' query, key and head norm weights, by the names the stage
/// reads them under.
fn checkpoint() -> Checkpoint {
    let mut checkpoint = Checkpoint::default();
    let projections = [("q_proj", QueryWidth::SIZE), ("k_proj", KvWidth::SIZE)];
    for (name, rows) in projections {
        let codes: Vec<f8e4m3> = (0..rows * Hidden::SIZE)
            .map(|at| f8e4m3::from_f32(code(at / Hidden::SIZE, at % Hidden::SIZE)))
            .collect();
        let shape = [rows, Hidden::SIZE];
        (checkpoint.insert(
            &format!("{name}.weight"),
            Dtype::F8_E4M3,
            &shape,
            f8e4m3::to_buf(&codes),
        ))
        .unwrap();
        let scales = (0..rows).flat_map(|row| scale(row).to_le_bytes()).collect();
        (checkpoint.insert(
            &format!("{name}.weight_scale"),
            Dtype::F32,
            &[rows, 1],
            scales,
        ))
        .unwrap();
    }
    for name in ["q_norm", "k_norm"] {
        let gains = (0..Head::SIZE)
            .flat_map(|channel| half::bf16::from_f32(gain(channel)).to_le_bytes())
            .collect();
        (checkpoint.insert(&format!("{name}.weight"), Dtype::BF16, &[Head::SIZE], gains)).unwrap();
    }
    checkpoint
}

/// Row `row`'s first `channels` projected features on the host, as the catalog's projections
/// quantize their inputs: the row in E4M3 under its largest magnitude over 448 rounded once to
/// BF16, each sum in FP64 times that scale and the channel's, rounded to BF16.
fn project(row: usize, channels: usize) -> Vec<bf16> {
    let input: Vec<f32> = (0..Hidden::SIZE)
        .map(|column| bf16::from_f32(feature(row, column)).to_f32())
        .collect();
    let largest = input.iter().map(|x| x.abs()).fold(0.0, f32::max);
    let row_scale = bf16::from_f32(largest / 448.0).to_f32();
    let codes: Vec<f64> = (input.iter())
        .map(|x| f64::from(f8e4m3::from_f32((x / row_scale).clamp(-448.0, 448.0)).to_f32()))
        .collect();
    (0..channels)
        .map(|channel| {
            let dot: f64 = (codes.iter().enumerate())
                .map(|(column, x)| x * f64::from(code(channel, column)))
                .sum();
            bf16::from_f32((dot * f64::from(row_scale) * f64::from(scale(channel))) as f32)
        })
        .collect()
}

/// Projected heads normalized on the host, in FP32.
fn normalize(projected: &[bf16]) -> Vec<f32> {
    (projected.chunks(Head::SIZE))
        .flat_map(|head| {
            let mean = head.iter().map(|x| x.to_f32().powi(2)).sum::<f32>() / Head::SIZE as f32;
            let inverse = 1.0 / (mean + 1e-6).sqrt();
            (head.iter().enumerate()).map(move |(channel, x)| x.to_f32() * inverse * gain(channel))
        })
        .collect()
}

/// Positions the live rows `start..start + rows`, every other row poison, and checks them as the
/// module documentation states, the normalized heads within four BF16 steps of the host's.
async fn check(device: &mut Device, stage: (&Qkv, &Position), (start, rows): (usize, usize)) {
    let live = start..start + rows;
    let (head, rotated) = (Head::SIZE, Rotated::SIZE);
    let (half, pairs) = (head / 2, rotated / 2);
    let poison = bf16::from_bits(POISON);
    let frequencies = PairFrequencies::<Rotated>::unscaled::<Head>(1e6).unwrap();
    let (mut low, mut high) = (Vec::new(), Vec::new());
    for row in 0..Rows::SIZE {
        let position = (4093 + 517 * row) as f32;
        for at in 0..rotated {
            let (sin, cos) = f64::from(position * frequencies.pairs()[at % pairs]).sin_cos();
            let (l, h) = match at < pairs {
                true => (cos, -sin),
                false => (sin, cos),
            };
            let (l, h) = match live.contains(&row) {
                true => (bf16::from_f32(l as f32), bf16::from_f32(h as f32)),
                false => (poison, poison),
            };
            low.push(l);
            high.push(h);
        }
    }
    // Each live row's features, every other row poison, as rows of `width`.
    let rows_of = |width: usize, row: &dyn Fn(usize) -> Vec<bf16>| -> Vec<bf16> {
        (0..Rows::SIZE)
            .flat_map(|at| match live.contains(&at) {
                true => row(at),
                false => vec![poison; width],
            })
            .collect()
    };
    let queries: Vec<Vec<bf16>> = live
        .clone()
        .map(|row| project(row, QueryWidth::SIZE))
        .collect();
    let keys_of: Vec<Vec<bf16>> = live
        .clone()
        .map(|row| project(row, KvWidth::SIZE))
        .collect();
    let table = |values: Vec<bf16>| HostTensor::<bf16, m![Rows, Rotated]>::from_vec(values);
    let rotation = Rotation {
        low: table(low.clone()).to_hbm(&mut device.pdma).await.unwrap(),
        high: table(high.clone()).to_hbm(&mut device.pdma).await.unwrap(),
    };
    let count = |value: usize| HostTensor::<i32, m![1]>::from_vec(vec![value as i32]);
    let start_value: Value = count(start).to_hbm(&mut device.pdma).await.unwrap();
    let rows_value: Value = count(rows).to_hbm(&mut device.pdma).await.unwrap();
    let hidden = rows_of(Hidden::SIZE, &|row| {
        (0..Hidden::SIZE)
            .map(|column| bf16::from_f32(feature(row, column)))
            .collect()
    });
    let hidden: HbmTensor<bf16, Chip, m![Rows, Hidden]> =
        (HostTensor::<bf16, m![Rows, Hidden]>::from_vec(hidden).to_hbm(&mut device.pdma))
            .await
            .unwrap();
    let counts = (&rotation, &start_value, &rows_value);
    let ((query_in, key_in), (query_out, key_out)) =
        launch(from_hidden, (&mut *device, stage, &hidden, counts))
            .await
            .unwrap();
    let mut values = Vec::new();
    for tensor in [query_in, query_out] {
        values.push(
            tensor
                .to_host::<QueryRows>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec(),
        );
    }
    for tensor in [key_in, key_out] {
        values.push(
            tensor
                .to_host::<KeyRows>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec(),
        );
    }
    let [query_in, query_out, key_in, key_out] = <[_; 4]>::try_from(values).unwrap();
    for (heads, input, output, projected) in [
        (Heads::SIZE * Groups::SIZE, &query_in, &query_out, &queries),
        (Heads::SIZE, &key_in, &key_out, &keys_of),
    ] {
        for (row, projected) in live.clone().zip(projected) {
            for (channel, want) in normalize(projected).iter().enumerate() {
                let found = input[row * heads * head + channel].to_f32();
                assert!(
                    (found - want).abs() <= want.abs() / 32.0 + 1.0 / 64.0,
                    "{heads} heads at {start}+{rows}, row {row}, normalized channel {channel}: \
                     {found} vs {want}"
                );
            }
        }
        for (at, found) in output.iter().enumerate() {
            let (row, channel, first) = (at / (heads * head), at % head, at / head * head);
            let pair = channel % half;
            if pair >= pairs {
                assert_eq!(
                    found.to_bits(),
                    input[at].to_bits(),
                    "{heads} heads at {start}+{rows}, row {row}, kept channel {channel}"
                );
            } else if live.contains(&row) {
                let coefficient = row * rotated + channel / half * pairs + pair;
                let expected = input[first + pair].to_f32() * low[coefficient].to_f32()
                    + input[first + half + pair].to_f32() * high[coefficient].to_f32();
                assert_eq!(
                    found.to_f32(),
                    bf16::from_f32(expected).to_f32(),
                    "{heads} heads at {start}+{rows}, row {row}, rotated channel {channel}"
                );
            }
        }
    }
}

/// The full layers' sixteen query heads and one key head, from the envelope's first row, and from
/// rows that start inside a row group, cross one, or end at the envelope's last row.
#[tokio::test(flavor = "current_thread")]
async fn positions_the_live_rows_of_the_full_layers() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let qkv: Qkv = checkpoint().load(&mut device).unwrap();
    let stage = (&qkv, &Position::default());
    for live in [(0, 9), (0, 64), (7, 2), (13, 9), (40, 24)] {
        check(&mut device, stage, live).await;
    }
}
