//! Compact Live attention and whole prefix exchange in one four-chip module call.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::ops::attention::flash::{
    Compact, Duo, Head, Queries, Span, Tokens, Visible,
};
use furiosa_opt_models::ops::collective::{Exchange, Pieces};
use furiosa_opt_models::tensor::axis::Rows;
use furiosa_opt_models::tensor::live::{BlockPrefix, Live, PrefixInputs, PrefixPlan};
use furiosa_opt_models::tensor::parallel::{Parallel, Part, Shard};
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_models::{Module, Op};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::{HbmBuffer, launch};

axes![Pairs = 7, Heads = 14, Spans = 20, Width = 7168];

type Chips = m![{ Symbol<Shard<4>> }];
type Capacity = Rows<10496>;
type QueryRows = Pieces<Capacity, 4>;
type Query = Queries<QueryRows, Pairs, Chips>;
type Kv = Tokens<Spans, Pairs, Chips>;
type Visibility = Visible<Spans, Chips>;
type Split = HbmTensor<bf16, Chips, m![{ Symbol<QueryRows> }, { Symbol<Part<Width, 4>> }]>;
type Output = HbmTensor<bf16, Chips, m![Capacity, Width]>;
type Transfers = (Rows<9472>, (Rows<9728>, (Rows<9984>,)));
type Work = PrefixInputs<Capacity, Transfers, Rows<1>, Chips>;
type Operands<'a> = (
    &'a Query,
    &'a Kv,
    &'a Kv,
    &'a Visibility,
    &'a Value<Chips>,
    &'a Work,
);

const CHIPS: usize = 4;
const SCALE: usize = (std::f32::consts::FRAC_1_SQRT_2 / 8.0).to_bits() as usize;
const POISON: u16 = 0x7fc1;
const LIMIT: f64 = 1e-2;
const GROUPS: usize = CHIPS * CHIPS * Heads::SIZE;

#[derive(DeviceSend)]
struct AttentionExchange;

impl<'a> Module<Operands<'a>, (), &'a mut Output> for AttentionExchange {
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: (),
        output: &'a mut Output,
        (query, key, value, visible, keys, work): Operands<'a>,
    ) {
        const {
            assert!(QueryRows::SIZE == 41984);
            assert!(Heads::SIZE == Pairs::SIZE * Duo::SIZE);
            assert!(Heads::SIZE * Head::SIZE == Part::<Width, 4>::SIZE);
        }
        let checked = work.read();
        let work = checked.prefix().rows();
        let interval = Live::<(), QueryRows>::all().subrange(0, CHIPS * work);
        let attended = Op::forward(
            &Compact::<Heads, SCALE>::default(),
            device,
            (interval.with(query), key, value, (visible, keys)),
        )
        .into_value();
        // SAFETY: every query row contains seven consecutive pairs of 128-wide heads. Folding
        // those axes gives the chip's 1792 columns; capacity and physical row order stay intact.
        let attended: Split = unsafe { attended.reshape() };
        let exchange = Exchange::<Parallel<4>>::default();
        // The actual owned attention result feeds one Live exchange, within one compiled entry.
        exchange.write_prefix::<bf16, Capacity, Width, Transfers, Rows<1>>(
            device,
            &checked.with(&attended),
            output,
        );
    }
}

#[device(chip = 4)]
fn attention_exchange(
    device: &mut Device,
    module: &AttentionExchange,
    inputs: Operands<'_>,
    output: &mut Output,
) {
    module.forward(device, (), output, inputs);
}

fn amplitude(row: usize, chip: usize, head: usize) -> f32 {
    ((row * 7 + chip * 5 + head * 3) % 17) as f32 - 8.0
}

fn base(chip: usize, head: usize, channel: usize) -> f32 {
    (((chip * 29 + head * 11 + channel * 3) % 61) as f32 - 30.0) / 64.0
}

/// Visible even and odd keys have scores +a and -a and values base +/- 1/4. Inputs are exactly
/// representable in BF16; no kernel output or other attention implementation supplies this oracle.
fn shifts(keys: usize) -> [f64; 17] {
    let scale = f64::from(f32::from_bits(SCALE as u32));
    std::array::from_fn(|i| {
        let score = (i as f64 - 8.0) * scale;
        let positive = keys.div_ceil(2) as f64 * score.exp();
        let negative = (keys / 2) as f64 * (-score).exp();
        0.25 * (positive - negative) / (positive + negative)
    })
}

fn expected(row: usize, chip: usize, head: usize, channel: usize, shifts: &[f64; 17]) -> f64 {
    let class = (row * 7 + chip * 5 + head * 3) % shifts.len();
    f64::from(base(chip, head, channel)) + shifts[class]
}

async fn scalar(device: &mut Device, value: usize) -> Value<Chips> {
    HostTensor::<i32, m![{ Chips }, 1]>::from_vec([i32::try_from(value).unwrap(); CHIPS])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

async fn inputs(device: &mut Device, work: usize, keys: usize) -> (Query, Kv, Kv, Visibility) {
    let query =
        HostTensor::<bf16, m![{ Chips }, { Symbol<QueryRows> }, Pairs, Duo, Head]>::from_vec(
            (0..CHIPS * QueryRows::SIZE * Heads::SIZE * Head::SIZE).map(|at| {
                let chip = at / (QueryRows::SIZE * Heads::SIZE * Head::SIZE);
                let row = at / (Heads::SIZE * Head::SIZE) % QueryRows::SIZE;
                let head = at / Head::SIZE % Heads::SIZE;
                if row >= CHIPS * work {
                    bf16::from_bits(POISON)
                } else {
                    bf16::from_f32(if at % Head::SIZE == 0 {
                        amplitude(row, chip, head)
                    } else {
                        0.0
                    })
                }
            }),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let kv_per_chip = Heads::SIZE * Spans::SIZE * Span::SIZE * Head::SIZE;
    let key = HostTensor::<bf16, m![{ Chips }, Pairs, Duo, Spans, Span, Head]>::from_vec(
        (0..CHIPS * kv_per_chip).map(|at| {
            let token = at / Head::SIZE % (Spans::SIZE * Span::SIZE);
            bf16::from_f32(if at % Head::SIZE != 0 {
                0.0
            } else if token >= keys {
                32.0
            } else if token % 2 == 0 {
                1.0
            } else {
                -1.0
            })
        }),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let value = HostTensor::<bf16, m![{ Chips }, Pairs, Duo, Spans, Span, Head]>::from_vec(
        (0..CHIPS * kv_per_chip).map(|at| {
            let chip = at / kv_per_chip;
            let head = at / (Spans::SIZE * Span::SIZE * Head::SIZE) % Heads::SIZE;
            let token = at / Head::SIZE % (Spans::SIZE * Span::SIZE);
            // Masked padding is finite and adversarial. It must not contribute to the mass or
            // values, including in the last partial span and the unvisited allocation suffix.
            bf16::from_f32(if token >= keys {
                8.0 + chip as f32
            } else {
                base(chip, head, at % Head::SIZE) + if token % 2 == 0 { 0.25 } else { -0.25 }
            })
        }),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let visible = HostTensor::<bf16, m![{ Chips }, Spans, Span]>::from_vec(
        (0..CHIPS * Spans::SIZE * Span::SIZE)
            .map(|at| bf16::from_f32(f32::from(at % (Spans::SIZE * Span::SIZE) < keys))),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    (query, key, value, visible)
}

async fn check(device: &mut Device, buffer: &HbmBuffer, work: usize, keys: usize) {
    let shifts = shifts(keys);
    let mut error = [0.0_f64; GROUPS];
    let mut norm = [0.0_f64; GROUPS];
    let mut worst = 0.0_f64;
    let row_bytes = Width::SIZE * size_of::<bf16>();
    // Bound host readback memory. This reads only the final exchanged output, after the single
    // module launch finishes; there is no host-visible attention intermediate.
    for first in (0..Capacity::SIZE).step_by(512) {
        let rows = (Capacity::SIZE - first).min(512);
        let bytes = buffer
            .slice(.., first * row_bytes..(first + rows) * row_bytes)
            .unwrap()
            .to_host(&mut device.pdma)
            .await
            .unwrap();
        assert_eq!(bytes.len(), CHIPS * rows * row_bytes);
        for destination in 0..CHIPS {
            for row in first..first + rows {
                let offset = (destination * rows + row - first) * row_bytes;
                let values = &bytes[offset..offset + row_bytes];
                if row >= work {
                    for pair in values.chunks_exact(2) {
                        assert_eq!(
                            u16::from_le_bytes([pair[0], pair[1]]),
                            POISON,
                            "tail row {row}"
                        );
                    }
                    continue;
                }
                let query_row = destination * work + row;
                for (column, pair) in values.chunks_exact(2).enumerate() {
                    let source = column / (Heads::SIZE * Head::SIZE);
                    let head = column / Head::SIZE % Heads::SIZE;
                    let channel = column % Head::SIZE;
                    let actual =
                        f64::from(bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32());
                    assert!(
                        actual.is_finite(),
                        "work {work}, destination {destination}, row {row}, column {column}"
                    );
                    let expected = expected(query_row, source, head, channel, &shifts);
                    let difference = actual - expected;
                    let group = (destination * CHIPS + source) * Heads::SIZE + head;
                    error[group] += difference * difference;
                    norm[group] += expected * expected;
                    worst = worst.max(difference.abs());
                }
            }
        }
    }
    let mut maximum = 0.0_f64;
    for group in 0..GROUPS {
        let relative = (error[group] / norm[group]).sqrt();
        assert!(
            relative < LIMIT,
            "work {work}, keys {keys}, destination/source/head group {group}: relative L2 {relative}"
        );
        maximum = maximum.max(relative);
    }
    eprintln!(
        "attention/exchange work {work}, queries {}, keys {keys}: {} checked values, max group relative L2 {maximum:.3e}, worst absolute {worst:.3e}",
        CHIPS * work,
        CHIPS * Capacity::SIZE * Width::SIZE
    );
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn compact_live_attention_feeds_whole_exchange_in_one_module_launch() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let module = AttentionExchange;
    let mut buffer = HbmBuffer::alloc_shared(&device, Capacity::SIZE * Width::SIZE * 2).unwrap();
    // One compiled entry and one destination allocation: grow, then shrink runtime work.
    for (work, keys) in [(9472, 37787), (9984, 39758), (9728, 37839)] {
        let mut output: Output = buffer.try_into().unwrap();
        let (query, key, value, visible) = inputs(&mut device, work, keys).await;
        let key_count = scalar(&mut device, keys).await;
        let plan = PrefixPlan::<(), Capacity, Transfers, Rows<1>>::new(
            BlockPrefix::try_over(work).unwrap(),
        )
        .unwrap();
        let work_count = Work::upload(&mut device, &plan).await.unwrap();
        HostTensor::<bf16, m![{ Chips }, Capacity, Width]>::from_vec(std::iter::repeat_n(
            bf16::from_bits(POISON),
            CHIPS * Capacity::SIZE * Width::SIZE,
        ))
        .to_hbm(&mut device.pdma)
        .output(&mut output)
        .await
        .unwrap();
        launch(
            attention_exchange,
            (
                &mut device,
                &module,
                (&query, &key, &value, &visible, &key_count, &work_count),
                &mut output,
            ),
        )
        .await
        .unwrap();
        buffer = output.into_buffer();
        check(&mut device, &buffer, work, keys).await;
    }
}
