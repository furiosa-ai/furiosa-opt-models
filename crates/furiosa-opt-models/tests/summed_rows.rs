//! A summed NVFP4 projection's live rows at the feed-forward's shapes: wherever the live rows of
//! a 64-row envelope start and however many there are, each live row's outputs are the ones the
//! whole envelope gives it, bit for bit.
//!
//! - Into the intermediate width from feature-slice rows, a lane group's switch straight into the
//!   tensor registers; back from feature-slice rows, a group switched whole into DM first; and
//!   back from the rows the first projection kept, a group's parts switched under the passes of
//!   the group before.
//! - At Gemma 4's widths and at a SwiGLU decoder's, whose owners and passes divide differently, so
//!   its kept rows stage a group at a time.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Op;
use furiosa_opt_models::models::decoder::attention::gemma4;
use furiosa_opt_models::models::decoder::form::summed::{Blocks, Codes, Ownership};
use furiosa_opt_models::ops::normalization::Spread;
use furiosa_opt_models::ops::pointwise::Kept;
use furiosa_opt_models::tensor::form::BlockScaled;
use furiosa_opt_models::tensor::form::nvfp4::{Features, Scales, Summed, W4A16};
use furiosa_opt_models::tensor::live::Live;
use furiosa_opt_models::tensor::memory::Hbm;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_std::{prelude::*, runtime::launch};
use tokio::sync::Mutex;

/// The device tests share one chip, so they run one at a time.
static DEVICE: Mutex<()> = Mutex::const_new(());

type Rows = gemma4::Rows;
type Hidden = gemma4::Hidden;
type Intermediate = gemma4::Intermediate;

/// A SwiGLU decoder's widths, as Llama 3 8B's.
mod swiglu {
    use furiosa_opt_std::prelude::*;

    axes![Hidden = 4096, Intermediate = 14336];
}

/// An NVFP4 projection of `I` inputs to `O` outputs, held as the catalog's summed form holds it,
/// its rows ending where `P` places them.
type Nvfp4<I, O, P> = BlockScaled<
    I,
    O,
    f4e2m1,
    f8e4m3,
    Scales<I, O>,
    Summed<W4A16, Hbm, P>,
    Chip,
    HbmTensor<f4e2m1, Chip, Codes<I, O, P>>,
    HbmTensor<f8e4m3, Chip, Blocks<I, O, P>>,
>;

/// The live rows `start` and `rows` name, through `linear`, from and back to HBM rows.
#[device(chip = 1)]
fn projected<I: AxisName, O: AxisName>(
    device: &mut Device,
    linear: &Nvfp4<I, O, Features>,
    (input, (start, rows)): (&HbmTensor<bf16, Chip, m![Rows, I]>, (&Value, &Value)),
) -> HbmTensor<bf16, Chip, m![Rows, O]>
where
    Nvfp4<I, O, Features>: DeviceSend
        + for<'x> Op<
            &'x Live<Spread<I, Rows, Chip>, Rows>,
            Output = Live<Spread<O, Rows, Chip>, Rows>,
        >,
{
    let live =
        Live::<(), Rows>::all().subrange(start.extent::<m![Rows]>(), rows.extent::<m![Rows]>());
    let rows: Live<Spread<I, Rows, Chip>, Rows> = live.with(input.to_dm(&mut device.tdma));
    let output = Op::forward(linear, &mut *device, &rows);
    output.value().to_hbm(&mut device.tdma)
}

/// The live rows through `gate`, kept as `K` where its passes summed them, then through `down`.
#[device(chip = 1)]
fn chained<H: AxisName, I: AxisName, K>(
    device: &mut Device,
    (gate, down): (&Nvfp4<H, I, Kept>, &Nvfp4<I, H, Features>),
    (input, (start, rows)): (&HbmTensor<bf16, Chip, m![Rows, H]>, (&Value, &Value)),
) -> HbmTensor<bf16, Chip, m![Rows, H]>
where
    Nvfp4<H, I, Kept>: DeviceSend + for<'x> Op<&'x Live<Spread<H, Rows, Chip>, Rows>, Output = K>,
    Nvfp4<I, H, Features>:
        DeviceSend + for<'k> Op<&'k K, Output = Live<Spread<H, Rows, Chip>, Rows>>,
{
    let live =
        Live::<(), Rows>::all().subrange(start.extent::<m![Rows]>(), rows.extent::<m![Rows]>());
    let rows: Live<Spread<H, Rows, Chip>, Rows> = live.with(input.to_dm(&mut device.tdma));
    let kept = Op::forward(gate, &mut *device, &rows);
    let output = Op::forward(down, &mut *device, &kept);
    output.value().to_hbm(&mut device.tdma)
}

/// Pseudo-random 64-bit words from `state`.
fn words(count: usize, mut state: u64) -> impl Iterator<Item = u64> {
    (0..count).map(move |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    })
}

/// Uniform values in `[0, 1)` from `state`.
fn uniform(count: usize, state: u64) -> impl Iterator<Item = f64> {
    words(count, state).map(|word| (word >> 11) as f64 / (1u64 << 53) as f64)
}

/// Random NVFP4 weights of `I` inputs and `O` outputs on the device: codes over the sixteen E2M1
/// values, E4M3 block scales from 8 to 448 and a tensor scale near 2^14.
async fn weight<I: AxisName, O: AxisName, P: Ownership>(
    device: &mut Device,
    seed: u64,
) -> Nvfp4<I, O, P> {
    let codes: Vec<u8> = (words(O::SIZE * I::SIZE / 2, seed))
        .map(|word| (word >> 56) as u8)
        .collect();
    let scales: Vec<f8e4m3> = (uniform(O::SIZE * I::SIZE / 16, seed + 1))
        .map(|unit| f8e4m3::from_f32((8.0 + 440.0 * unit) as f32))
        .collect();
    let global = [2f32.powi(14) * 1.37, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let pdma = &mut device.pdma;
    BlockScaled {
        weight: (HostTensor::<f4e2m1, m![O, I]>::from_buf(codes)
            .to_hbm(pdma)
            .await)
            .unwrap(),
        weight_scale: (HostTensor::<f8e4m3, Scales<I, O>>::from_vec(scales)
            .to_hbm(pdma)
            .await)
            .unwrap(),
        contract: Summed {
            input: W4A16,
            global: (HostTensor::<f32, m![1 # 8]>::from_vec(global)
                .to_hbm(pdma)
                .await)
                .unwrap(),
            place: core::marker::PhantomData,
        },
        shape: core::marker::PhantomData,
    }
}

/// Every row of the envelope in `[-2, 2)`, the rows outside `live` poison.
fn rows<I: AxisName>(state: u64, live: core::ops::Range<usize>) -> Vec<bf16> {
    (uniform(Rows::SIZE * I::SIZE, state).enumerate())
        .map(|(at, unit)| match live.contains(&(at / I::SIZE)) {
            true => bf16::from_f32((4.0 * unit - 2.0) as f32),
            false => bf16::from_bits(0x7fc1),
        })
        .collect()
}

/// A count as the device reads it.
fn count(value: usize) -> HostTensor<i32, m![1]> {
    HostTensor::from_vec([i32::try_from(value).expect("counts fit I32")])
}

/// The live rows these cases name: none, the first rows of a group up to and past its half, whole
/// groups, and rows that start inside a group, end inside one or cross groups.
const CASES: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (0, 4),
    (0, 5),
    (0, 8),
    (0, 9),
    (0, 12),
    (0, 13),
    (0, 32),
    (1, 3),
    (7, 1),
    (13, 4),
    (13, 8),
    (60, 4),
    (3, 61),
];

/// Each live row of every case at widths `$hidden` and `$intermediate` gives the outputs the whole
/// envelope gives it, in each of the three ways a projection stages its rows.
macro_rules! envelope {
    ($device:expr, $hidden:ident, $intermediate:ident) => {{
        let device: &mut Device = $device;
        let up = weight::<$hidden, $intermediate, Features>(device, 1).await;
        let gate = weight::<$hidden, $intermediate, Kept>(device, 1).await;
        let down = weight::<$intermediate, $hidden, Features>(device, 4).await;
        let mut envelope: [Vec<bf16>; 3] = Default::default();
        for (start, live) in [(0, Rows::SIZE)].iter().chain(CASES) {
            let (start, live) = (*start, *live);
            let first = count(start).to_hbm(&mut device.pdma).await.unwrap();
            let counted = count(live).to_hbm(&mut device.pdma).await.unwrap();
            let narrow = rows::<$hidden>(3, start..start + live);
            let narrow = (HostTensor::<bf16, m![Rows, $hidden]>::from_vec(narrow))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let wide = rows::<$intermediate>(5, start..start + live);
            let wide = (HostTensor::<bf16, m![Rows, $intermediate]>::from_vec(wide))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let forms: [(Vec<bf16>, usize); 3] = [
                (
                    launch(
                        projected,
                        (&mut *device, &up, (&narrow, (&first, &counted))),
                    )
                    .await
                    .unwrap()
                    .to_host::<m![Rows, $intermediate]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec(),
                    $intermediate::SIZE,
                ),
                (
                    launch(
                        projected,
                        (&mut *device, &down, (&wide, (&first, &counted))),
                    )
                    .await
                    .unwrap()
                    .to_host::<m![Rows, $hidden]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec(),
                    $hidden::SIZE,
                ),
                (
                    launch(
                        chained,
                        (&mut *device, (&gate, &down), (&narrow, (&first, &counted))),
                    )
                    .await
                    .unwrap()
                    .to_host::<m![Rows, $hidden]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec(),
                    $hidden::SIZE,
                ),
            ];
            for (form, (found, width)) in forms.into_iter().enumerate() {
                if live == Rows::SIZE {
                    envelope[form] = found;
                    continue;
                }
                let rows = start * width..(start + live) * width;
                for at in rows.clone() {
                    assert_eq!(
                        found[at].to_bits(),
                        envelope[form][at].to_bits(),
                        "{}x{} form {form}, rows {start}+{live}: output {at}",
                        $hidden::SIZE,
                        $intermediate::SIZE,
                    );
                }
            }
        }
    }};
}

/// Each live row of every case gives the outputs the whole envelope gives it, in each of the three
/// ways a projection stages its rows, at Gemma 4's widths and at a SwiGLU decoder's.
#[tokio::test(flavor = "current_thread")]
async fn projects_each_live_row_as_the_envelope() {
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    envelope!(&mut device, Hidden, Intermediate);
    {
        use swiglu::{Hidden, Intermediate};
        envelope!(&mut device, Hidden, Intermediate);
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

/// Each form's device time at each live row count `ROWS_PROFILE_ROWS` lists, over
/// `ROWS_PROFILE_STEPS` launches after a warm-up, at the level `FURIOSA_OPT_PROFILE` asks for; at
/// `trace` the commands of the last launch follow.
#[tokio::test(flavor = "current_thread")]
#[ignore = "profiles launches on a device"]
async fn rows_profile() {
    let list = |name: &str, default: &str| -> Vec<usize> {
        let list = std::env::var(name).unwrap_or_else(|_| default.into());
        list.split(',').map(|item| item.parse().unwrap()).collect()
    };
    let steps = list("ROWS_PROFILE_STEPS", "10")[0];
    let spans = std::sync::Arc::new(Spans::default());
    let _spans = tracing::subscriber::set_default(std::sync::Arc::clone(&spans));
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let up = weight::<Hidden, Intermediate, Features>(&mut device, 1).await;
    let gate = weight::<Hidden, Intermediate, Kept>(&mut device, 1).await;
    let down = weight::<Intermediate, Hidden, Features>(&mut device, 4).await;
    for live in list("ROWS_PROFILE_ROWS", "1,4,5,8,9,64") {
        let first = count(0).to_hbm(&mut device.pdma).await.unwrap();
        let counted = count(live).to_hbm(&mut device.pdma).await.unwrap();
        let narrow = rows::<Hidden>(3, 0..live);
        let narrow = (HostTensor::<bf16, m![Rows, Hidden]>::from_vec(narrow))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let wide = rows::<Intermediate>(5, 0..live);
        let wide = (HostTensor::<bf16, m![Rows, Intermediate]>::from_vec(wide))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        macro_rules! profile {
            ($form:literal, $kernel:expr, $weights:expr, $rows:expr) => {{
                let mut tasks = vec![];
                for step in 0..=steps {
                    spans.0.lock().unwrap().clear();
                    launch($kernel, (&mut device, $weights, $rows))
                        .await
                        .unwrap();
                    device.synchronize().await.unwrap();
                    let launched = spans.0.lock().unwrap();
                    let task = launched.iter().filter(|span| span.0 == "Task");
                    let task = task.map(|span| span.3 - span.2).max();
                    if step > 0 {
                        tasks.push(task.expect("run with FURIOSA_OPT_PROFILE"));
                    }
                }
                tasks.sort_unstable();
                println!(
                    "form {} rows {live} task median {} min {} max {} cycles",
                    $form,
                    tasks[tasks.len() / 2],
                    tasks[0],
                    tasks[tasks.len() - 1]
                );
                let mut launched = spans.0.lock().unwrap().clone();
                launched.sort_by_key(|span| span.2);
                let start = launched.first().map_or(0, |span| span.2);
                for (name, index, begin, end) in launched.iter().filter(|span| span.0 != "Task") {
                    println!(
                        "span {}@{live} {} {} {name} {index}",
                        $form,
                        begin - start,
                        end - start
                    );
                }
            }};
        }
        profile!("up", projected, &up, (&narrow, (&first, &counted)));
        profile!("down", projected, &down, (&wide, (&first, &counted)));
        profile!(
            "chained",
            chained,
            (&gate, &down),
            (&narrow, (&first, &counted))
        );
    }
}
