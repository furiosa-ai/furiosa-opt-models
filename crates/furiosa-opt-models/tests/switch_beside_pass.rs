//! A Sub switch beside an unrelated Main contraction, in trips of a loop.
//!
//! - A trip switches every slice's packets into a staging buffer in data memory on Sub, fills the
//!   buffer into the tensor registers and contracts code rows against them on Main.
//! - The next trip's switch shares no data with the previous trip's contraction, so it may run
//!   beside it, whether the buffer is a fresh tensor every trip or one tensor for all trips.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::{prelude::*, runtime::launch};
use tokio::sync::Mutex;

/// The device tests share one chip, so they run one at a time.
static DEVICE: Mutex<()> = Mutex::const_new(());

axes![
    L = 8,
    P = 2,
    Q = 16,
    C = 4,
    V = 16,
    T = 4,
    Row = 16,
    Y = 256
];

type Chip = m![1];
type Cluster = m![1 # 2];

type Spread = HbmTensor<bf16, Chip, m![L, P, Q, T, C, V]>;
type Codes = HbmTensor<f4e2m1, Chip, m![Y, Row, C, P, Q, V]>;
type Projected = HbmTensor<bf16, Chip, m![Y, T, Row, L]>;

/// Each slice's packets of every trip, as the slice of `(L, P, Q)` holds them.
type Rows = DmTensor<bf16, Chip, Cluster, m![L, P, Q], m![T, C, V]>;
/// A trip's packets of every slice, in every slice: eight lanes of 4 KB.
type Staged = DmTensor<bf16, Chip, Cluster, m![Y], m![C, L, P, Q, V]>;
type Coded = DmTensor<f4e2m1, Chip, Cluster, m![Y], m![Row, C, P, Q, V]>;
type Out = DmTensor<bf16, Chip, Cluster, m![Y], m![T, Row, L]>;

/// The rows in data memory through a Main copy, so the trips follow Main work as a kernel's do.
fn lead(device: &mut Device, spread: &Spread) -> Rows {
    let spread: Rows = spread.to_dm(&mut device.tdma);
    (device.main.begin(spread.view()))
        .fetch::<m![T, C], m![V]>()
        .collect::<m![T, C], m![V]>()
        .commit_trim::<m![V]>()
        .commit()
}

/// A trip's packets from every slice to every slice through the switch ring, on Sub.
fn switch(device: &mut Device, rows: &Rows, trip: usize, staged: &mut Staged) {
    (device
        .sub
        .begin(rows.view().tile::<m![T], 1, m![1 # 4, C, V]>(trip)))
    .fetch::<m![C], m![V]>()
    .switch::<m![Y], m![C, L, P, Q]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
    .collect::<m![C, L, P, Q], m![V]>()
    .commit_trim::<m![V]>()
    .commit_view(staged.view_mut());
}

/// The staged packets into the tensor registers on Sub, then every code row against them on Main.
fn project(device: &mut Device, (staged, codes): (&Staged, &Coded), trip: usize, out: &mut Out) {
    let lanes: TrfTensor<bf16, Chip, Cluster, m![Y], m![L], m![C, P, Q, V]> =
        (device.sub.begin(staged.view()))
            .fetch::<m![L, C, P, Q], m![V]>()
            .collect::<m![L, C, P, Q], m![V]>()
            .to_trf();
    (device.main.begin(codes.view()))
        .fetch::<m![Row, C, P, Q / 4], m![Q % 4, V]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Row, C, P, Q], m![V]>()
        .contract_outer::<m![Row, C, P, Q / 2], m![Q % 2, V], _, _, _>(&lanes)
        .contract_packet::<m![1]>()
        .contract_time::<m![Row]>()
        .contract_lane::<m![Row], m![L]>(LaneMode::Interleaved)
        .cast::<bf16, m![L # 16]>()
        .commit_trim::<m![L]>()
        .commit_view(out.view_mut().tile::<m![T], 1, m![1 #{!} 4, Row, L]>(trip));
}

/// A fresh staging buffer every trip: dead from its fill to the next trip's switch.
#[device(chip = 1)]
fn staged_per_trip(device: &mut Device, (spread, codes): (&Spread, &Codes)) -> Projected {
    let (rows, codes): (_, Coded) = (lead(device, spread), codes.to_dm(&mut device.tdma));
    let mut out = Out::new();
    for trip in 0..T::SIZE {
        let mut staged = Staged::new();
        switch(device, &rows, trip, &mut staged);
        project(device, (&staged, &codes), trip, &mut out);
    }
    out.to_hbm(&mut device.tdma)
}

/// One staging buffer for every trip.
#[device(chip = 1)]
fn staged_once(device: &mut Device, (spread, codes): (&Spread, &Codes)) -> Projected {
    let (rows, codes): (_, Coded) = (lead(device, spread), codes.to_dm(&mut device.tdma));
    let (mut out, mut staged) = (Out::new(), Staged::new());
    for trip in 0..T::SIZE {
        switch(device, &rows, trip, &mut staged);
        project(device, (&staged, &codes), trip, &mut out);
    }
    out.to_hbm(&mut device.tdma)
}

/// Pseudo-random 64-bit words from `state`.
fn words(count: usize, mut state: u64) -> Vec<u64> {
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        })
        .collect()
}

/// The inputs, uploaded, and each output the host computes for them: every code decoded through the
/// e2m1 table, every product and sum exact in FP32, the sum rounded once to BF16.
async fn inputs(device: &mut Device) -> ((Spread, Codes), Vec<bf16>) {
    let inner = C::SIZE * P::SIZE * Q::SIZE * V::SIZE;
    let spread: Vec<bf16> = (words(L::SIZE * P::SIZE * Q::SIZE * T::SIZE * C::SIZE * V::SIZE, 1)
        .iter())
    .map(|word| bf16::from_f32((word >> 62) as f32 - 1.0))
    .collect();
    let codes: Vec<u8> = words(Y::SIZE * Row::SIZE * inner / 2, 2)
        .iter()
        .map(|word| (word >> 56) as u8)
        .collect();
    let code = |at: usize| f4e2m1::from_bits(codes[at / 2] >> (4 * (at % 2))).to_f32();
    let lane = |lane: usize, trip: usize, k: usize| {
        let (c, p, q, v) = (
            k / (P::SIZE * Q::SIZE * V::SIZE),
            k / (Q::SIZE * V::SIZE) % P::SIZE,
            k / V::SIZE % Q::SIZE,
            k % V::SIZE,
        );
        spread
            [((((lane * P::SIZE + p) * Q::SIZE + q) * T::SIZE + trip) * C::SIZE + c) * V::SIZE + v]
            .to_f32()
    };
    let decoded: Vec<f32> = (0..Y::SIZE * Row::SIZE * inner).map(code).collect();
    let lanes: Vec<f32> = (0..T::SIZE * L::SIZE * inner)
        .map(|at| lane(at / inner % L::SIZE, at / (L::SIZE * inner), at % inner))
        .collect();
    let expected = (0..Y::SIZE * T::SIZE * Row::SIZE * L::SIZE)
        .map(|at| {
            let (y, trip, row, l) = (
                at / (T::SIZE * Row::SIZE * L::SIZE),
                at / (Row::SIZE * L::SIZE) % T::SIZE,
                at / L::SIZE % Row::SIZE,
                at % L::SIZE,
            );
            let (codes, lanes) = (
                &decoded[(y * Row::SIZE + row) * inner..][..inner],
                &lanes[(trip * L::SIZE + l) * inner..][..inner],
            );
            bf16::from_f32(
                codes
                    .iter()
                    .zip(lanes)
                    .map(|(code, lane)| code * lane)
                    .sum(),
            )
        })
        .collect();
    let spread = HostTensor::<bf16, m![L, P, Q, T, C, V]>::from_vec(spread)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let codes = HostTensor::<f4e2m1, m![Y, Row, C, P, Q, V]>::from_buf(codes)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    ((spread, codes), expected)
}

/// Both buffers' forms project every trip's rows as the host does.
#[tokio::test(flavor = "current_thread")]
async fn projects_every_trip_as_the_host() {
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(staged_per_trip.topology()).unwrap();
    let ((spread, codes), expected) = inputs(&mut device).await;
    for per_trip in [true, false] {
        let arguments = (&mut device, (&spread, &codes));
        let projected = match per_trip {
            true => launch(staged_per_trip, arguments).await.unwrap(),
            false => launch(staged_once, arguments).await.unwrap(),
        };
        let projected = projected
            .to_host::<m![Y, T, Row, L]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, (got, want)) in projected.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "per trip {per_trip}: element {at}"
            );
        }
    }
}

/// The device spans profiled launches report: name, instruction, first and last cycle.
#[derive(Default)]
struct Spans(std::sync::Mutex<Vec<(String, String, u64, u64)>>);

impl tracing::Subscriber for Spans {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        #[derive(Default)]
        struct Fields(String, String, u64, u64);
        impl tracing::field::Visit for Fields {
            fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                match field.name() {
                    "begin_cycle" => self.2 = value,
                    "end_cycle" => self.3 = value,
                    _ => {}
                }
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                match field.name() {
                    "name" => self.0 = value.into(),
                    "instruction_index" => self.1 = value.into(),
                    _ => {}
                }
            }
            fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
        }
        if span.metadata().name() == "NPU" {
            let mut fields = Fields::default();
            span.record(&mut fields);
            let Fields(name, index, begin, end) = fields;
            self.0.lock().unwrap().push((name, index, begin, end));
        }
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// A fresh buffer every trip costs no more device time than one buffer: the next switch runs beside
/// the previous contraction in both. Run with `FURIOSA_OPT_PROFILE=info`, or `trace` for the spans.
#[tokio::test(flavor = "current_thread")]
#[ignore = "profiles launches on a device"]
async fn switches_beside_the_previous_trip_with_a_fresh_buffer() {
    let spans = std::sync::Arc::new(Spans::default());
    let _spans = tracing::subscriber::set_default(std::sync::Arc::clone(&spans));
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(staged_per_trip.topology()).unwrap();
    let ((spread, codes), _) = inputs(&mut device).await;
    let mut cycles = [0u64; 2];
    for (at, per_trip) in [true, false].into_iter().enumerate() {
        let mut tasks = vec![];
        for _ in 0..10 {
            spans.0.lock().unwrap().clear();
            let arguments = (&mut device, (&spread, &codes));
            match per_trip {
                true => launch(staged_per_trip, arguments).await.unwrap(),
                false => launch(staged_once, arguments).await.unwrap(),
            };
            device.synchronize().await.unwrap();
            let launched = spans.0.lock().unwrap();
            let task = launched.iter().filter(|span| span.0 == "Task");
            tasks.push(
                task.map(|span| span.3 - span.2)
                    .max()
                    .expect("run with FURIOSA_OPT_PROFILE"),
            );
        }
        tasks.sort_unstable();
        cycles[at] = tasks[tasks.len() / 2];
        println!("per trip {per_trip}: Task median {} cycles", cycles[at]);
        let mut launched = spans.0.lock().unwrap().clone();
        launched.sort_by_key(|span| span.2);
        let start = launched.first().map_or(0, |span| span.2);
        for (name, index, begin, end) in launched.iter().filter(|span| span.0 != "Task") {
            println!("  {:9} {:9} {name:20} {index}", begin - start, end - start);
        }
    }
    let [per_trip, once] = cycles;
    assert!(
        per_trip <= once + once / 100,
        "a fresh buffer every trip: {per_trip} cycles, one buffer: {once}"
    );
}
