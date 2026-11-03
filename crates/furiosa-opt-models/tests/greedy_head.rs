//! A greedy head over a list of rows: each row's token is its first maximum among the
//! checkpoint's tokens, of logits capped by the config's `final_logit_softcapping` when it has one.

#![feature(register_tool, trait_alias)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Module;
use furiosa_opt_models::load::{Checkpoint, Context, Load};
use furiosa_opt_models::modules::language::Greedy;
use furiosa_opt_models::ops::index::gather;
use furiosa_opt_models::ops::linear::{Affine, Biased, Parted};
use furiosa_opt_models::ops::pointwise::activation::Softcap;
use furiosa_opt_models::ops::reduction::Argmax;
use furiosa_opt_models::tensor::axis::{Eight, Rows};
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
use tokio::sync::Mutex;

// Device tests share one chip, so they run one at a time.
static DEVICE: Mutex<()> = Mutex::const_new(());

// No head tiles a vocabulary of 128256 tokens exactly; it compiles them on 129024 rows.
axes![Hidden = 1024, Compiled = 129024, Padded = 152064];

/// A checkpoint vocabulary below the compiled one, so the bias, not the padding, bounds the
/// selection.
const LIVE: usize = 151_936 - 256;

/// A head that selects the token of each of `R` staged rows.
trait Selects<R: AxisName> = for<'a, 'r> Module<
        &'a mut gather::Rows<'r, bf16, Hidden, R>,
        (),
        &'a mut (),
        Output = HbmTensor<i32, Chip, m![R]>,
    >;

#[device(chip = 1)]
fn selected<V: AxisName, R: AxisName, C>(
    device: &mut Device,
    head: &Greedy<Hidden, V, Biased<Hidden, V>, Argmax<V>, C>,
    input: &mut gather::Rows<'_, bf16, Hidden, R>,
) -> HbmTensor<i32, Chip, m![R]>
where
    C: DeviceSend,
    Greedy<Hidden, V, Biased<Hidden, V>, Argmax<V>, C>: Selects<R>,
{
    head.forward(device, (), &mut (), input)
}

/// A greedy head over a table that lies by parts, as the head's passes land it.
type ByParts<V> = Greedy<
    Hidden,
    V,
    Affine<HbmTensor<bf16, Chip, Parted<Hidden, V>>, HbmTensor<f32, Chip, m![V]>>,
    Argmax<V>,
>;

#[device(chip = 1)]
fn selected_by_parts<V: AxisName, R: AxisName>(
    device: &mut Device,
    head: &ByParts<V>,
    input: &mut gather::Rows<'_, bf16, Hidden, R>,
) -> HbmTensor<i32, Chip, m![R]>
where
    ByParts<V>: Selects<R>,
{
    head.forward(device, (), &mut (), input)
}

/// Input rows and coefficients giving row `r` each `(token, logit)` of `logits(r)`, else -1: two
/// BF16 coefficients on its features `2r` and `2r + 1` carry a logit exactly through FP32 sums.
fn projection<'a, V: AxisName, R: AxisName>(
    logits: impl Fn(usize) -> &'a [(usize, f32)],
) -> (Vec<bf16>, Vec<bf16>) {
    let mut weight = vec![bf16::from_f32(0.0); V::SIZE * Hidden::SIZE];
    for token in 0..V::SIZE {
        for row in 0..R::SIZE {
            weight[token * Hidden::SIZE + 2 * row] = bf16::from_f32(-1.0);
        }
    }
    let mut input = vec![bf16::from_f32(0.0); R::SIZE * Hidden::SIZE];
    for row in 0..R::SIZE {
        input[row * Hidden::SIZE + 2 * row] = bf16::from_f32(1.0);
        input[row * Hidden::SIZE + 2 * row + 1] = bf16::from_f32(1.0);
        for &(token, value) in logits(row) {
            let high = bf16::from_f32(value);
            let low = bf16::from_f32(value - high.to_f32());
            assert_eq!(high.to_f32() + low.to_f32(), value);
            weight[token * Hidden::SIZE + 2 * row] = high;
            weight[token * Hidden::SIZE + 2 * row + 1] = low;
        }
    }
    (weight, input)
}

/// A greedy head over `weight` whose bias masks the tokens from `live` on, its logits capped
/// by `cap`.
async fn greedy<V: AxisName, C>(
    device: &mut Device,
    weight: Vec<bf16>,
    live: usize,
    cap: C,
) -> Greedy<Hidden, V, Biased<Hidden, V>, Argmax<V>, C> {
    let weight = HostTensor::<bf16, m![V, Hidden]>::from_vec(weight)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let mask = (0..V::SIZE).map(|token| if token < live { 0.0 } else { f32::MIN });
    let mask = HostTensor::<f32, m![V]>::from_vec(mask)
        .to_hbm::<Chip, m![V]>(&mut device.pdma)
        .await
        .unwrap();
    let argmax = Argmax::<V>::load(&mut Context::new(&mut *device), &Checkpoint::default());
    Greedy::new(Affine::new(weight, mask), argmax.unwrap(), cap)
}

/// Each input row in HBM of its own.
async fn storage(device: &mut Device, input: Vec<bf16>) -> gather::RowStorage<Hidden, bf16> {
    let mut rows = Vec::new();
    for row in input.chunks_exact(Hidden::SIZE) {
        let row = HostTensor::<bf16, m![Hidden]>::from_vec(row.to_vec());
        rows.push(
            row.to_hbm::<Chip, m![Hidden]>(&mut device.pdma)
                .await
                .unwrap(),
        );
    }
    Slice::from(rows)
}

async fn tokens<V: AxisName, R: AxisName, C>(
    device: &mut Device,
    head: &Greedy<Hidden, V, Biased<Hidden, V>, Argmax<V>, C>,
    input: Vec<bf16>,
) -> Vec<usize>
where
    C: DeviceSend,
    Greedy<Hidden, V, Biased<Hidden, V>, Argmax<V>, C>: Selects<R>,
{
    let source = storage(device, input).await;
    let mut input = gather::Rows::<bf16, Hidden, R>::new(device, &source)
        .await
        .unwrap();
    let tokens = launch(selected, (&mut *device, head, &mut input))
        .await
        .unwrap()
        .to_host::<m![R]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    tokens.into_iter().map(|token| token as usize).collect()
}

/// Greedy selects each row's first maximum among the checkpoint's tokens across ties and pass
/// boundaries, never a padded row, at every row envelope, also when the compiled vocabulary
/// pads a checkpoint that no head tiles exactly.
#[tokio::test(flavor = "current_thread")]
async fn selects_first_maximum() {
    selects_first_maximum_of::<Padded, Eight>(LIVE).await;
    selects_first_maximum_of::<Padded, Rows<32>>(LIVE).await;
    selects_first_maximum_of::<Padded, Rows<64>>(LIVE).await;
    selects_first_maximum_of::<Compiled, Rows<16>>(128_256).await;
}

/// The logits of each case's tokens and the token it selects, of `live` checkpoint tokens
/// compiled on `V`: ties within and across passes and clusters, padded rows above the live ones.
fn cases<V: AxisName>(live: usize) -> [(Vec<(usize, f32)>, usize); 9] {
    [
        (vec![(42, 5.0)], 42),
        (vec![(7, 5.0), (V::SIZE / 3 + 19, 5.0)], 7),
        (
            vec![(live - 1, 5.0), (live, 10.0), (V::SIZE - 1, 10.0)],
            live - 1,
        ),
        (vec![(0, 5.0f32.next_down()), (live - 1, 5.0)], live - 1),
        (vec![(2 * V::SIZE / 3 + 5, 3.0)], 2 * V::SIZE / 3 + 5),
        (vec![(1234, -0.5), (1233, -0.5)], 1233),
        (
            vec![(V::SIZE / 3 - 1, 2.0), (V::SIZE / 3, 2.0)],
            V::SIZE / 3 - 1,
        ),
        (vec![(99_999, 7.0)], 99_999),
        (vec![(live - 1, 3.0), (live, 3.0)], live - 1),
    ]
}

/// Row `r` runs case `r % 9` on its own pair of input columns, of `live` checkpoint tokens
/// compiled on `V`.
async fn selects_first_maximum_of<V: AxisName, R: AxisName>(live: usize) {
    let cases = cases::<V>(live);
    let case = |row: usize| &cases[row % cases.len()];
    let (weight, input) = projection::<V, R>(|row| &case(row).0);
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let head = greedy::<V, ()>(&mut device, weight, live, ()).await;
    let tokens = tokens::<V, R, ()>(&mut device, &head, input).await;
    for (row, &token) in tokens.iter().enumerate() {
        assert_eq!(token, case(row).1, "row {row}");
    }
}

/// A head over a table that lies by parts, as its passes land it, selects each row's first
/// maximum as the row-major table's head does, across ties and pass boundaries, never a padded
/// row.
#[tokio::test(flavor = "current_thread")]
async fn selects_first_maximum_over_a_parted_table() {
    type V = Padded;
    type R = Rows<32>;
    let cases = cases::<V>(LIVE);
    let case = |row: usize| &cases[row % cases.len()];
    let (weight, input) = projection::<V, R>(|row| &case(row).0);
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let rows = greedy::<V, ()>(&mut device, weight, LIVE, ()).await;
    let Affine { weight, bias } = rows.head;
    let parted = weight
        .to_host::<m![V, Hidden]>(&mut device.pdma)
        .await
        .unwrap()
        .to_hbm::<Chip, Parted<Hidden, V>>(&mut device.pdma)
        .await
        .unwrap();
    let head: ByParts<V> = Greedy::new(Affine::new(parted, bias), rows.argmax, ());
    let source = storage(&mut device, input).await;
    let mut input = gather::Rows::<bf16, Hidden, R>::new(&mut device, &source)
        .await
        .unwrap();
    let tokens = launch(selected_by_parts, (&mut device, &head, &mut input))
        .await
        .unwrap()
        .to_host::<m![R]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    for (row, &token) in tokens.iter().enumerate() {
        assert_eq!(token as usize, case(row).1, "row {row}");
    }
}

/// The softcap a head's load places for a config whose `final_logit_softcapping` is `bound`.
fn softcap(device: &mut Device, bound: f32) -> Softcap {
    let directory = tempfile::tempdir().unwrap();
    let config = serde_json::json!({ "final_logit_softcapping": bound });
    let checkpoint = Checkpoint::new(directory.path(), Some(config)).unwrap();
    let mut context = Context::new(device);
    let loaded = context.config("final_logit_softcapping", |cx| {
        Softcap::load(cx, &checkpoint)
    });
    loaded.unwrap()
}

/// The first token of the largest `bound * tanh((z + bias) * (1 / bound))` in FP32, dividing by
/// a scalar as PyTorch's CUDA kernels do, its capped logit and the largest of the others.
fn capped_first_maximum<V: AxisName>(
    bound: f32,
    live: usize,
    logit: impl Fn(usize) -> f32,
) -> (usize, f32, f32) {
    let inverse = (1.0 / f64::from(bound)) as f32;
    (0..V::SIZE).fold((0, f32::NEG_INFINITY, f32::NEG_INFINITY), |best, token| {
        let bias = if token < live { 0.0 } else { f32::MIN };
        let capped = bound * ((logit(token) + bias) * inverse).tanh();
        match capped > best.1 {
            true => (token, capped, best.1),
            false => (best.0, best.1, best.2.max(capped)),
        }
    })
}

/// Row `row`'s FP32 logit of `token` through `weight`, as the contraction sums its two
/// coefficients.
fn logit(weight: &[bf16], row: usize, token: usize) -> f32 {
    let at = token * Hidden::SIZE + 2 * row;
    weight[at].to_f32() + weight[at + 1].to_f32()
}

/// A head capping its logits at `c` takes each row's first maximum of `c * tanh(z / c)` in FP32:
/// logits that saturate tie at `c` and choose the first, across passes and clusters too.
#[tokio::test(flavor = "current_thread")]
async fn selects_first_maximum_of_capped_logits() {
    type V = Padded;
    const BOUND: f32 = 30.0;
    let third = V::SIZE / 3;
    let cases: [(&[(usize, f32)], usize); 8] = [
        (&[(11, 480.0), (third + 19, 960.0)], 11),
        (&[(5, 3.0e4), (2 * third + 5, 1.0e6)], 5),
        (&[(42, 150.0), (99_999, 151.0)], 99_999),
        (&[(1233, 600.0), (1234, 800.0)], 1233),
        (
            &[(LIVE - 1, 2.0), (LIVE, 1.0e5), (V::SIZE - 1, 1.0e5)],
            LIVE - 1,
        ),
        (&[(777, 2.5), (third + 1, 2.5)], 777),
        (&[(2 * third + 5, 3.0)], 2 * third + 5),
        (&[(third - 1, 600.0), (third, 2.0e5)], third - 1),
    ];
    let (weight, input) = projection::<V, Eight>(|row| cases[row].0);
    let wanted: Vec<usize> = (0..Eight::SIZE)
        .map(|row| capped_first_maximum::<V>(BOUND, LIVE, |token| logit(&weight, row, token)).0)
        .collect();
    assert_eq!(wanted, cases.map(|case| case.1));
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let cap = softcap(&mut device, BOUND);
    let head = greedy::<V, _>(&mut device, weight, LIVE, cap).await;
    assert_eq!(
        tokens::<V, Eight, _>(&mut device, &head, input).await,
        wanted
    );
}

/// Rows of random logits through a head capped at 30 and at 20 take the first maximum the FP32
/// cap gives them; in half the rows some logits saturate, and the first of those wins.
#[tokio::test(flavor = "current_thread")]
async fn selects_first_maximum_of_random_capped_logits() {
    type V = Padded;
    const BOUNDS: [f32; 2] = [30.0, 20.0];
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move |below: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as usize % below
    };
    // Logits below the caps 1/32 apart at least, at random tokens, and saturating ones.
    let rows: Vec<Vec<(usize, f32)>> = (0..Eight::SIZE)
        .map(|row| {
            let below = (0..960).map(|at| {
                let logit = at as f32 / 32.0 + next(16) as f32 / 1024.0;
                (next(LIVE), logit)
            });
            let mut logits: Vec<_> = below.collect();
            logits.extend((0..50 * (row % 2)).map(|_| (next(LIVE), (400 + next(3600)) as f32)));
            logits
        })
        .collect();
    let (weight, input) = projection::<V, Eight>(|row| &rows[row][..]);
    let wanted = BOUNDS.map(|bound| {
        (0..Eight::SIZE)
            .map(|row| {
                let (token, best, next) =
                    capped_first_maximum::<V>(bound, LIVE, |token| logit(&weight, row, token));
                // A tie at the cap is exact in any FP32 tanh; another maximum leads by more than
                // two such tanh round apart, so the cap decides the token, not a rounding.
                assert!(
                    best == bound || best - next > best * 2f32.powi(-17),
                    "row {row}"
                );
                token
            })
            .collect::<Vec<_>>()
    });
    let _lease = DEVICE.lock().await;
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let cap = softcap(&mut device, BOUNDS[0]);
    let mut head = greedy::<V, _>(&mut device, weight, LIVE, cap).await;
    for (bound, wanted) in BOUNDS.into_iter().zip(wanted) {
        head.cap = softcap(&mut device, bound);
        let tokens = tokens::<V, Eight, _>(&mut device, &head, input.clone()).await;
        assert_eq!(tokens, wanted, "cap {bound}");
    }
}
