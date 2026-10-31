//! A group's position op on the rows attention reads its heads from: each live row's query and
//! key heads turn their rotated coordinates by that row's rotation, and keep the others.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Op;
use furiosa_opt_models::models::decoder::attention::{gemma4, qwen3};
use furiosa_opt_models::models::decoder::{self, Attention};
use furiosa_opt_models::ops::rotary::{PairFrequencies, Rotation};
use furiosa_opt_models::tensor::axis::{self, Lanes, Single};
use furiosa_opt_models::tensor::live::Live;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;

type Rows = axis::Rows<64>;

/// `Group` query heads of each of `Heads` key/value heads, as attention reads them: a slice per
/// lane and key/value head, each holding its row groups' heads.
type HeadRows<Heads, Group, Head> = Live<
    DmTensor<
        bf16,
        Chip,
        m![1 # 2],
        m![Rows % ({ Symbol<Lanes<Rows>> }) # 8, Heads # 8, 1 # 4],
        m![{ m![Rows / ({ Symbol<Lanes<Rows>> }), Group] }, Head],
    >,
    Rows,
>;

/// Each row's query heads and key heads, as a projection writes them.
type QueryHeads<Heads, Groups, Head> = m![Rows, Heads, Groups, Head];
type KeyHeads<Heads, Head> = m![Rows, Heads, Head];
type Table<Rotated> = m![Rows, Rotated];

type Query<A> =
    QueryHeads<<A as Attention>::Heads, <A as Attention>::Groups, <A as Attention>::Head>;
type Key<A> = KeyHeads<<A as Attention>::Heads, <A as Attention>::Head>;
type Rotated<A> = <A as Attention>::Rotated;

#[device(chip = 1)]
fn rotate<Heads, Groups, Head, Rotated, P>(
    device: &mut Device,
    positions: &P,
    (query, key): (
        &HbmTensor<bf16, Chip, QueryHeads<Heads, Groups, Head>>,
        &HbmTensor<bf16, Chip, KeyHeads<Heads, Head>>,
    ),
    (rotation, live): (&Rotation<Rotated, Chip, Rows>, &Value),
) -> (
    HbmTensor<bf16, Chip, QueryHeads<Heads, Groups, Head>>,
    HbmTensor<bf16, Chip, KeyHeads<Heads, Head>>,
)
where
    Heads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Rotated: AxisName,
    P: DeviceSend
        + for<'r> Op<
            (
                (HeadRows<Heads, Groups, Head>, HeadRows<Heads, Single, Head>),
                &'r Rotation<Rotated, Chip, Rows>,
            ),
            Output = (HeadRows<Heads, Groups, Head>, HeadRows<Heads, Single, Head>),
        >,
{
    let live = Live::<(), Rows>::read(live);
    let heads = (
        live.with(query.to_dm(&mut device.tdma)),
        live.with(key.to_dm(&mut device.tdma)),
    );
    let (query, key) = positions.forward(&mut *device, (heads, rotation));
    (
        query.value().to_hbm(&mut device.tdma),
        key.value().to_hbm(&mut device.tdma),
    )
}

/// A value no rotation of a live row reads: a NaN with a payload of its own.
const POISON: u16 = 0xff81;

/// Rotates `live` of the 64 rows of group `A`'s heads at positions far from zero, every other row
/// and its rotation poison, and checks each live row's rotated coordinates against the products
/// in FP32 rounded once, and every row's other coordinates bit for bit.
async fn check<A: Attention<Rotated: AxisName>>(device: &mut Device, theta: f32, live: usize)
where
    decoder::Position<A>: Default
        + DeviceSend
        + for<'r> Op<
            (
                (
                    HeadRows<A::Heads, A::Groups, A::Head>,
                    HeadRows<A::Heads, Single, A::Head>,
                ),
                &'r Rotation<Rotated<A>, Chip, Rows>,
            ),
            Output = (
                HeadRows<A::Heads, A::Groups, A::Head>,
                HeadRows<A::Heads, Single, A::Head>,
            ),
        >,
{
    let (head, rotated) = (A::Head::SIZE, Rotated::<A>::SIZE);
    let (half, pairs) = (head / 2, rotated / 2);
    let poison = bf16::from_bits(POISON);
    let frequencies = PairFrequencies::<Rotated<A>>::unscaled::<A::Head>(theta).unwrap();
    let tables = (0..Rows::SIZE).map(|row| {
        let position = (4093 + 517 * row) as f32;
        let turn = |pair: usize| f64::from(position * frequencies.pairs()[pair]).sin_cos();
        let entry = |at: usize, low: bool| match (row < live, at < pairs, low) {
            (false, ..) => poison,
            (true, first, low) => {
                let (sin, cos) = turn(at % pairs);
                let value = match (first, low) {
                    (true, true) => cos,
                    (false, true) => sin,
                    (true, false) => -sin,
                    (false, false) => cos,
                };
                bf16::from_f32(value as f32)
            }
        };
        (
            (0..rotated).map(|at| entry(at, true)).collect::<Vec<_>>(),
            (0..rotated).map(|at| entry(at, false)).collect::<Vec<_>>(),
        )
    });
    let (low, high): (Vec<_>, Vec<_>) = tables.unzip();
    let (low, high) = (low.concat(), high.concat());
    let values = |heads: usize| {
        (0..Rows::SIZE * heads * head)
            .map(|at| match at / (heads * head) < live {
                true => bf16::from_f32(((at * 7 + at / head * 5) % 37) as f32 / 16.0 - 1.0),
                false => poison,
            })
            .collect::<Vec<_>>()
    };
    let query = values(A::Heads::SIZE * A::Groups::SIZE);
    let key = values(A::Heads::SIZE);
    let query_hbm: HbmTensor<bf16, Chip, Query<A>> =
        HostTensor::<bf16, Query<A>>::from_vec(query.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
    let key_hbm: HbmTensor<bf16, Chip, Key<A>> = HostTensor::<bf16, Key<A>>::from_vec(key.clone())
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let table =
        |values: &Vec<bf16>| HostTensor::<bf16, Table<Rotated<A>>>::from_vec(values.clone());
    let rotation = Rotation {
        low: table(&low).to_hbm(&mut device.pdma).await.unwrap(),
        high: table(&high).to_hbm(&mut device.pdma).await.unwrap(),
    };
    let count: Value = HostTensor::<i32, m![1]>::from_vec(vec![live as i32])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let positions = decoder::Position::<A>::default();
    let (query_out, key_out) = launch(
        rotate,
        (
            &mut *device,
            &positions,
            (&query_hbm, &key_hbm),
            (&rotation, &count),
        ),
    )
    .await
    .unwrap();
    let query_out = query_out
        .to_host::<Query<A>>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    let key_out = key_out
        .to_host::<Key<A>>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    for (heads, input, output) in [
        (A::Heads::SIZE * A::Groups::SIZE, &query, &query_out),
        (A::Heads::SIZE, &key, &key_out),
    ] {
        for (at, found) in output.iter().enumerate() {
            let (row, channel, start) = (at / (heads * head), at % head, at / head * head);
            let pair = channel % half;
            if pair >= pairs {
                assert_eq!(
                    found.to_bits(),
                    input[at].to_bits(),
                    "{heads} heads, row {row}, kept channel {channel}"
                );
            } else if row < live {
                let coefficient = row * rotated + channel / half * pairs + pair;
                let expected = input[start + pair].to_f32() * low[coefficient].to_f32()
                    + input[start + half + pair].to_f32() * high[coefficient].to_f32();
                assert_eq!(
                    found.to_f32(),
                    bf16::from_f32(expected).to_f32(),
                    "{heads} heads, row {row}, rotated channel {channel}"
                );
            }
        }
    }
}

/// The full group's 512-wide heads, one key/value head of sixteen query heads, turn their leading
/// 64 coordinates of each half; a whole envelope and a partial one of nine rows.
#[tokio::test(flavor = "current_thread")]
async fn rotates_the_leading_coordinates_of_wide_heads() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    for live in [9, 64] {
        check::<gemma4::full::Group>(&mut device, 1e6, live).await;
    }
}

/// Heads that rotate every coordinate turn as before: 128-wide heads of eight key/value heads of
/// four query heads, and the sliding group's 256-wide heads of eight of two.
#[tokio::test(flavor = "current_thread")]
async fn rotates_whole_heads() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    for live in [9, 64] {
        check::<qwen3::Size<"8B">>(&mut device, 1e6, live).await;
        check::<gemma4::sliding::Group>(&mut device, 1e4, live).await;
    }
}
