//! The GELU forms on device: exact GELU, quick GELU and the tanh approximation, also on a launch's
//! live rows in feature slices.

#![feature(float_erf)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Op;
use furiosa_opt_models::ops::index::scatter::Store;
use furiosa_opt_models::ops::normalization::Spread;
use furiosa_opt_models::ops::pointwise::activation::{Gelu, NewGelu, QuickGelu};
use furiosa_opt_models::tensor::live::Live;
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_std::{prelude::*, runtime::launch};
use tokio::sync::Mutex;

// Device tests share one chip, so they run one at a time.
static DEVICE: Mutex<()> = Mutex::const_new(());

axes![Gates = 4096, Rows = 8, Expanded = 15360];

/// A step's count, to upload.
fn count(value: usize) -> HostTensor<i32, m![1]> {
    HostTensor::from_vec([i32::try_from(value).expect("counts fit I32")])
}

#[device(chip = 1)]
fn activated<A, S: M + 'static>(
    device: &mut Device,
    activation: &A,
    input: &HbmTensor<bf16, m![1], S>,
) -> HbmTensor<bf16, m![1], S>
where
    A: DeviceSend + for<'x> Op<&'x HbmTensor<bf16, m![1], S>, Output = HbmTensor<bf16, m![1], S>>,
{
    activation.forward(device, input)
}

/// `activation` of each BF16 input of shape `S`, cycling over [-8, 8), on the device, within
/// one BF16 step of `definition` in FP64, and within 2^-16 of it near zero.
async fn matches<A, S: M + 'static>(activation: A, definition: fn(f64) -> f64)
where
    A: DeviceSend + for<'x> Op<&'x HbmTensor<bf16, m![1], S>, Output = HbmTensor<bf16, m![1], S>>,
{
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let inputs: Vec<bf16> = (0..S::SIZE)
        .map(|at| bf16::from_f32((at % 4096) as f32 / 256.0 - 8.0))
        .collect();
    let input = HostTensor::<bf16, S>::from_vec(inputs.clone())
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let output = launch(activated, (&mut device, &activation, &input))
        .await
        .unwrap()
        .to_host::<S>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    for (&x, got) in inputs.iter().zip(output) {
        assert_close(x, got, definition);
    }
}

/// `got` for input `x` is within one BF16 step of `definition` in FP64, and within 2^-16 of it
/// near zero.
fn assert_close(x: bf16, got: bf16, definition: fn(f64) -> f64) {
    let (x, got) = (f64::from(x.to_f32()), f64::from(got.to_f32()));
    let want = definition(x);
    let step = (want.abs().log2().floor() - 7.0).exp2();
    assert!(
        (got - want).abs() <= step.max(2f64.powi(-16)),
        "{x}: {got} vs {want}"
    );
}

/// `x / 2 * (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.
fn tanh_approximation(x: f64) -> f64 {
    x / 2.0 * (1.0 + ((2.0 / core::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// GELU is `x Phi(x)` through the error function, not its tanh approximation.
#[tokio::test(flavor = "current_thread")]
async fn gelu_is_exact() {
    matches::<_, m![Gates]>(Gelu, |x| x / 2.0 * (1.0 + (x / 2f64.sqrt()).erf())).await;
}

/// Quick GELU is `x sigmoid(1.702 x)`.
#[tokio::test(flavor = "current_thread")]
async fn quick_gelu_is_a_scaled_sigmoid() {
    matches::<_, m![Gates]>(QuickGelu, |x| x / (1.0 + (-1.702 * x).exp())).await;
}

/// New GELU is the tanh approximation `x / 2 * (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, also
/// over rows of a width no power of two divides.
#[tokio::test(flavor = "current_thread")]
async fn new_gelu_is_the_tanh_approximation() {
    matches::<_, m![Gates]>(NewGelu, tanh_approximation).await;
    matches::<_, m![Rows, Expanded]>(NewGelu, tanh_approximation).await;
}

#[device(chip = 1)]
fn activated_rows<A>(
    device: &mut Device,
    activation: &A,
    input: &HbmTensor<bf16, m![1], m![Rows, Expanded]>,
    (start, rows): (&Value, &Value),
    output: &mut HbmTensor<bf16, m![1], m![Rows, Expanded]>,
) where
    A: DeviceSend
        + for<'x> Op<
            &'x Live<Spread<Expanded, Rows>, Rows>,
            Output = Live<Spread<Expanded, Rows>, Rows>,
        >,
{
    let live =
        Live::<(), Rows>::all().subrange(start.extent::<m![Rows]>(), rows.extent::<m![Rows]>());
    let input: Live<Spread<Expanded, Rows>, Rows> = live.with(input.to_dm(&mut device.tdma));
    let result = activation.forward(&mut *device, &input);
    Store.forward(device, ((result.value(), result.with(())), output));
}

/// New GELU of a launch's live rows in feature slices is each row's tanh approximation, bit for
/// bit as the rows alone give it, wherever the live rows start; the other rows keep the output's.
#[tokio::test(flavor = "current_thread")]
async fn new_gelu_of_live_rows_is_the_tanh_approximation() {
    const KEPT: f32 = -3.0;
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let values: Vec<bf16> = (0..Rows::SIZE * Expanded::SIZE)
        .map(|at| bf16::from_f32((at % 4096) as f32 / 256.0 - 8.0))
        .collect();
    let upload = async |device: &mut Device, values: Vec<bf16>| {
        HostTensor::<bf16, m![Rows, Expanded]>::from_vec(values)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap()
    };
    let whole = upload(&mut device, values.clone()).await;
    let alone = launch(activated, (&mut device, &NewGelu, &whole))
        .await
        .unwrap()
        .to_host::<m![Rows, Expanded]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    for (first, rows) in [(3, 4), (0, 8), (7, 1)] {
        let live = |at: usize| (first..first + rows).contains(&(at / Expanded::SIZE));
        let input = values.iter().enumerate().map(|(at, &x)| match live(at) {
            true => x,
            false => bf16::from_f32(f32::NAN),
        });
        let input = upload(&mut device, input.collect()).await;
        let mut output = upload(
            &mut device,
            vec![bf16::from_f32(KEPT); Rows::SIZE * Expanded::SIZE],
        )
        .await;
        let (start, count) = (
            count(first).to_hbm(&mut device.pdma).await.unwrap(),
            count(rows).to_hbm(&mut device.pdma).await.unwrap(),
        );
        launch(
            activated_rows,
            (&mut device, &NewGelu, &input, (&start, &count), &mut output),
        )
        .await
        .unwrap();
        let output = output
            .to_host::<m![Rows, Expanded]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, &got) in output.iter().enumerate() {
            let want = if live(at) {
                alone[at]
            } else {
                bf16::from_f32(KEPT)
            };
            assert_eq!(got.to_bits(), want.to_bits(), "rows {first}+{rows}: {at}");
            if live(at) {
                assert_close(values[at], got, tanh_approximation);
            }
        }
    }
}
