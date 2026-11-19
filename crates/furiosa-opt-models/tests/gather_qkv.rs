//! Public checkpoint loading, collective ordering, and fused projection together.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::load::{Checkpoint, Context, Load};
use furiosa_opt_models::modules::parallel::Gathered;
use furiosa_opt_models::ops::attention::full::{Joined, Joint, Projection};
use furiosa_opt_models::ops::attention::geometry::{Hidden, ProjectionWidth};
use furiosa_opt_models::ops::collective::{AllGather, GatherTile, InTiles, Pieces, Tiled};
use furiosa_opt_models::ops::linear::grouped::Layout;
use furiosa_opt_models::ops::linear::{Grouped, Linear};
use furiosa_opt_models::tensor::axis::Rows;
use furiosa_opt_models::tensor::live::{BlockPrefix, Live, Partitioned};
use furiosa_opt_models::tensor::parallel::{Parallel, Shard};
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_models::{Module, Op};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;

axes![Head = 128];

type Chips = m![{ Symbol<Shard<4>> }];
type Input<I, S> = HbmTensor<bf16, Chips, m![S, I]>;
type Output<W, S> = HbmTensor<bf16, Chips, m![{ Symbol<Pieces<S, 4>> }, W]>;
type Qkv<W, S> = (Output<W, S>, Output<W, S>, Output<W, S>);
type GatheredRows<I, S, const T: usize, const K: usize, const H: usize> =
    Tiled<GatherTile<bf16, 4, I, T>, Pieces<S, 4>, 4, T, K, H>;
type Owned<I, W> = Projection<Grouped<I, Joint<W>, Chips>, Head>;
type Borrowed<'a, I, W> =
    Projection<Grouped<I, Joint<W>, Chips, &'a HbmTensor<bf16, Chips, Layout<I, Joint<W>>>>, Head>;
type Plain<I, W> = Projection<Linear<I, Joint<W>, Chips>, Head>;

#[device(chip = 4)]
fn tiled<I: AxisName, W: AxisName, S: AxisName, const T: usize, const K: usize, const H: usize>(
    device: &mut Device,
    projection: &Borrowed<'_, I, W>,
    input: &Input<I, S>,
    _gather: &AllGather<Parallel<4>, InTiles<T, K, H>>,
) -> Qkv<W, S>
where
    for<'a> AllGather<Parallel<4>, InTiles<T, K, H>>:
        Op<&'a Input<I, S>, Output = GatheredRows<I, S, T, K, H>>,
{
    Gathered::<_, Parallel<4>, InTiles<T, K, H>>::new(projection).forward(
        device,
        (),
        &mut (),
        input,
    )
}

#[device(chip = 4)]
fn live<I: AxisName, W: AxisName, S: AxisName, const T: usize, const K: usize, const H: usize>(
    device: &mut Device,
    projection: &Borrowed<'_, I, W>,
    input: &Input<I, S>,
    start: &Value<Chips>,
    count: &Value<Chips>,
    _gather: &AllGather<Parallel<4>, InTiles<T, K, H>>,
) -> Qkv<W, S>
where
    for<'a, 'b> AllGather<Parallel<4>, InTiles<T, K, H>>:
        Op<&'a Live<&'b Input<I, S>, S>, Output = Partitioned<GatheredRows<I, S, T, K, H>, S, 4>>,
{
    let interval = Live::<(), S>::all().subrange(start.extent::<m![S]>(), count.extent::<m![S]>());
    Gathered::<_, Parallel<4>, InTiles<T, K, H>>::new(projection)
        .forward(device, (), &mut (), &interval.with(input))
        .into_value()
}

#[device(chip = 4)]
fn joined<I: AxisName, W: AxisName, S: AxisName, const T: usize, const K: usize, const H: usize>(
    device: &mut Device,
    projection: &Joined<Borrowed<'_, I, W>>,
    input: &Input<I, S>,
    blocks: &Value<Chips>,
    _gather: &AllGather<Parallel<4>, InTiles<T, K, H>>,
) -> Qkv<W, S>
where
    for<'a, 'b> AllGather<Parallel<4>, InTiles<T, K, H>>: Op<
            &'a BlockPrefix<&'b Input<I, S>, S, Rows<H>>,
            Output = BlockPrefix<GatheredRows<I, S, T, K, H>, S, Rows<H>>,
        >,
{
    let interval = BlockPrefix::<(), S, Rows<H>>::read(blocks);
    Gathered::<_, Parallel<4>, InTiles<T, K, H>>::new(projection)
        .forward(device, (), &mut (), &interval.with(input))
        .into_value()
}

#[device(chip = 4)]
fn whole<I: AxisName, W: AxisName, S: AxisName>(
    device: &mut Device,
    projection: &Borrowed<'_, I, W>,
    input: &Input<I, S>,
) -> Qkv<W, S> {
    Gathered::<_, Parallel<4>>::new(projection).forward(device, (), &mut (), input)
}

#[device(chip = 4)]
fn ordinary<I: AxisName, W: AxisName, S: AxisName>(
    device: &mut Device,
    projection: &Plain<I, W>,
    input: &Input<I, S>,
) -> Qkv<W, S> {
    Gathered::<_, Parallel<4>>::new(projection).forward(device, (), &mut (), input)
}

async fn download<W: AxisName, S: AxisName>(
    device: &mut Device,
    output: &Qkv<W, S>,
) -> [Vec<bf16>; 3] {
    let first = output
        .0
        .to_host::<m![{ Symbol<Shard<4>> }, { Symbol<Pieces<S, 4>> }, W]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    let second = output
        .1
        .to_host::<m![{ Symbol<Shard<4>> }, { Symbol<Pieces<S, 4>> }, W]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    let third = output
        .2
        .to_host::<m![{ Symbol<Shard<4>> }, { Symbol<Pieces<S, 4>> }, W]>(&mut device.pdma)
        .await
        .unwrap()
        .into_vec();
    [first, second, third]
}

fn checkpoint<I: AxisName, W: AxisName>(weights: &[bf16]) -> (tempfile::TempDir, Checkpoint) {
    let dir = tempfile::tempdir().unwrap();
    let bytes = bf16::to_buf(weights);
    let view = safetensors::tensor::TensorView::new(
        safetensors::Dtype::BF16,
        vec![4 * 3 * W::SIZE, I::SIZE],
        &bytes,
    )
    .unwrap();
    safetensors::serialize_to_file(
        [("weight", view)],
        &None,
        &dir.path().join("weights.safetensors"),
    )
    .unwrap();
    let source = Checkpoint::new(dir.path(), Some(serde_json::json!({}))).unwrap();
    (dir, source)
}

async fn check<
    I: AxisName + 'static,
    W: AxisName + 'static,
    S: AxisName + 'static,
    const T: usize,
    const K: usize,
    const H: usize,
>(
    device: &mut Device,
) where
    for<'a> AllGather<Parallel<4>, InTiles<T, K, H>>:
        Op<&'a Input<I, S>, Output = GatheredRows<I, S, T, K, H>>,
    for<'a, 'b> AllGather<Parallel<4>, InTiles<T, K, H>>:
        Op<&'a Live<&'b Input<I, S>, S>, Output = Partitioned<GatheredRows<I, S, T, K, H>, S, 4>>,
    for<'a, 'b> AllGather<Parallel<4>, InTiles<T, K, H>>: Op<
            &'a BlockPrefix<&'b Input<I, S>, S, Rows<H>>,
            Output = BlockPrefix<GatheredRows<I, S, T, K, H>, S, Rows<H>>,
        >,
{
    let weights: Vec<_> = (0..4 * 3 * W::SIZE * I::SIZE)
        .map(|i| bf16::from_f32(((i * 17 % 63) as f32 - 31.0) / 32.0))
        .collect();
    let inputs: Vec<_> = (0..4 * S::SIZE * I::SIZE)
        .map(|i| bf16::from_f32(((i * 23 % 61) as f32 - 30.0) / 32.0))
        .collect();
    let (_dir, source) = checkpoint::<I, W>(&weights);
    let mut grouped: Owned<I, W> = source.load(device).unwrap();
    let plain: Plain<I, W> = source.load(device).unwrap();
    let borrowed: Borrowed<'_, I, W> = Load::load(&mut Context::new(None), &grouped).unwrap();
    let joined_projection: Joined<Borrowed<'_, I, W>> =
        Joined(Load::load(&mut Context::new(None), &grouped).unwrap());
    let input = HostTensor::<bf16, m![{ Symbol<Shard<4>> }, S, I]>::from_vec(inputs.clone())
        .to_hbm::<Chips, m![S, I]>(&mut device.pdma)
        .await
        .unwrap();
    let gather = AllGather::<Parallel<4>, InTiles<T, K, H>>::default();
    let tiled_output = launch(tiled, (&mut *device, &borrowed, &input, &gather))
        .await
        .unwrap();
    let got = download(device, &tiled_output).await;
    drop(tiled_output);
    let whole_output = launch(whole, (&mut *device, &borrowed, &input))
        .await
        .unwrap();
    let whole_values = download(device, &whole_output).await;
    drop(whole_output);
    let plain_output = launch(ordinary, (&mut *device, &plain, &input))
        .await
        .unwrap();
    let plain_values = download(device, &plain_output).await;
    drop(plain_output);
    for part in 0..3 {
        for (at, ((got, full), plain)) in got[part]
            .iter()
            .zip(&whole_values[part])
            .zip(&plain_values[part])
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                full.to_bits(),
                "whole/tiled part {part}, element {at}"
            );
            assert_eq!(
                got.to_bits(),
                plain.to_bits(),
                "ordinary/tiled part {part}, element {at}"
            );
        }
    }
    // Poison rows outside each part's interval. Only live output slices may be downloaded;
    // other output rows are intentionally unspecified, including an entirely empty call.
    let mut ranges = vec![
        (0, 0),
        (0, 128),
        (0, H),
        (0, H + 1),
        (H, T),
        (H + 1, T - 3),
        (S::SIZE - 128, 128),
        (S::SIZE, 0),
        (0, S::SIZE),
        (0, 128),
    ];
    if T >= 1024 {
        ranges.push((H + 256, 768));
    }
    if S::SIZE >= 9984 {
        ranges.extend([(0, 9472), (0, 9728)]);
    }
    if S::SIZE > 9984 {
        ranges.extend([(0, 9984), (0, 10240)]);
    }
    if I::SIZE == 128 {
        for rows in (H..S::SIZE).step_by(H) {
            if !ranges.contains(&(0, rows)) {
                ranges.push((0, rows));
            }
        }
    }
    let mut live_elements = 0usize;
    let mut joined_elements = 0usize;
    for (origin, length) in ranges {
        let poisoned = inputs.iter().enumerate().map(|(at, &value)| {
            let row = at / I::SIZE % S::SIZE;
            if (origin..origin + length).contains(&row) {
                value
            } else {
                bf16::from_bits(0x7fc1)
            }
        });
        let input: Input<I, S> =
            HostTensor::<bf16, m![{ Symbol<Shard<4>> }, S, I]>::from_vec(poisoned)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let start: Value<Chips> = HostTensor::<i32, m![1]>::from_vec([origin as i32])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let count: Value<Chips> = HostTensor::<i32, m![1]>::from_vec([length as i32])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let result = launch(
            live,
            (&mut *device, &borrowed, &input, &start, &count, &gather),
        )
        .await
        .unwrap();
        let joined_result = if origin == 0 && length.is_multiple_of(H) {
            let units = BlockPrefix::<(), S, Rows<H>>::try_over(length)
                .unwrap()
                .blocks();
            let blocks: Value<Chips> = HostTensor::<i32, m![1]>::from_vec([units as i32])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            Some(
                launch(
                    joined,
                    (&mut *device, &joined_projection, &input, &blocks, &gather),
                )
                .await
                .unwrap(),
            )
        } else {
            None
        };
        if length == 0 {
            continue;
        }
        for (term, tensor) in [result.0, result.1, result.2].into_iter().enumerate() {
            let storage = tensor.into_buffer();
            let stride = W::SIZE * 2;
            for part in 0..4 {
                let begin = (part * S::SIZE + origin) * stride;
                let bytes = storage
                    .slice(.., begin..begin + length * stride)
                    .unwrap()
                    .to_host(&mut device.pdma)
                    .await
                    .unwrap();
                for (at, b) in bytes.chunks_exact(2).enumerate() {
                    let chip = at / (length * W::SIZE);
                    let offset = at % (length * W::SIZE);
                    let expected = got[term]
                        [(chip * 4 * S::SIZE + part * S::SIZE + origin) * W::SIZE + offset];
                    assert_eq!(
                        u16::from_le_bytes([b[0], b[1]]),
                        expected.to_bits(),
                        "live origin {origin}, length {length}, term {term}, chip {chip}, part {part}, element {offset}"
                    );
                    live_elements += 1;
                }
            }
        }
        if let Some(result) = joined_result {
            for (term, tensor) in [result.0, result.1, result.2].into_iter().enumerate() {
                let prefix = tensor
                    .into_buffer()
                    .slice(.., ..4 * length * W::SIZE * 2)
                    .unwrap()
                    .to_host(&mut device.pdma)
                    .await
                    .unwrap();
                for (at, b) in prefix.chunks_exact(2).enumerate() {
                    let chip = at / (4 * length * W::SIZE);
                    let row_part = at / (length * W::SIZE) % 4;
                    let within = at % (length * W::SIZE);
                    let expected =
                        got[term][(chip * 4 * S::SIZE + row_part * S::SIZE) * W::SIZE + within];
                    assert_eq!(
                        u16::from_le_bytes([b[0], b[1]]),
                        expected.to_bits(),
                        "joined length {length}, term {term}, chip {chip}, part {row_part}, element {within}"
                    );
                    joined_elements += 1;
                }
            }
        }
    }
    println!(
        "PASS runtime gather/QKV capacity={} input={} part_width={} live_elements={live_elements} joined_elements={joined_elements}",
        S::SIZE,
        I::SIZE,
        W::SIZE
    );
    // The device constructor bounds units before deriving addresses. Exercise that boundary
    // on real hardware as well as the host constructor's ordinary refusal contract.
    if I::SIZE == 128 && S::SIZE == 1280 {
        for units in [-1, (S::SIZE / H + 3) as i32] {
            let blocks: Value<Chips> = HostTensor::<i32, m![1]>::from_vec([units])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(
                joined,
                (&mut *device, &joined_projection, &input, &blocks, &gather),
            )
            .await
            .unwrap();
            if units < 0 {
                drop(output);
            } else {
                let bounded = download(device, &output).await;
                for (actual, expected) in bounded.iter().flatten().zip(got.iter().flatten()) {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "bounded joined prefix"
                    );
                }
            }
        }
        println!("PASS device block-count bounds");
    }

    let rows = 4 * S::SIZE;
    let heads = W::SIZE / Head::SIZE;
    let mut sampled_rows = vec![0, H - 1, H, H + T - 1, S::SIZE - 1, S::SIZE, rows - 1];
    sampled_rows.sort_unstable();
    sampled_rows.dedup();
    let columns = [0, 31, 32, Head::SIZE - 1, Head::SIZE, W::SIZE - 1];
    let mut samples = 0;
    for chip in 0..4 {
        for part in 0..3 {
            for &row in &sampled_rows {
                for &column in &columns {
                    let stored = ((chip * heads + column / Head::SIZE) * 3 + part) * Head::SIZE
                        + column % Head::SIZE;
                    let expected: f64 = inputs[row * I::SIZE..][..I::SIZE]
                        .iter()
                        .zip(&weights[stored * I::SIZE..][..I::SIZE])
                        .map(|(x, w)| f64::from(x.to_f32()) * f64::from(w.to_f32()))
                        .sum();
                    assert_eq!(
                        got[part][(chip * rows + row) * W::SIZE + column].to_bits(),
                        bf16::from_f32(expected as f32).to_bits(),
                        "FP64 chip {chip}, part {part}, row {row}, column {column}"
                    );
                    samples += 1;
                }
            }
        }
    }
    println!(
        "PASS input={} part_width={} rows_per_chip={} tile={} tiles={} head={} elements={} fp64_samples={samples}",
        I::SIZE,
        W::SIZE,
        S::SIZE,
        T,
        K,
        H,
        got.iter().map(Vec::len).sum::<usize>()
    );
    let before_update = (I::SIZE == 128).then(|| got.clone());
    drop((got, whole_values, plain_values));

    drop((borrowed, joined_projection));

    if I::SIZE == 128 {
        let replacement: Vec<_> = weights
            .iter()
            .map(|w| bf16::from_f32(-w.to_f32()))
            .collect();
        let (_dir, source) = checkpoint::<I, W>(&replacement);
        <&mut Owned<I, W> as Load<_>>::load(
            &mut Context::new(Some(&mut *device)),
            (&mut grouped, &source),
        )
        .unwrap();
        let borrowed: Borrowed<'_, I, W> = Load::load(&mut Context::new(None), &grouped).unwrap();
        let output = launch(tiled, (&mut *device, &borrowed, &input, &gather))
            .await
            .unwrap();
        let updated = download(device, &output).await;
        for (actual, prior) in updated
            .iter()
            .flatten()
            .zip(before_update.as_ref().unwrap().iter().flatten())
        {
            assert_eq!(actual.to_f32(), -prior.to_f32(), "updated QKV value");
        }
        println!("PASS grouped QKV checkpoint update");
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn tiled_gather_preserves_fused_projection() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    check::<Hidden<128>, ProjectionWidth<512>, Rows<768>, 512, 2, 256>(&mut device).await;
    check::<Hidden<128>, ProjectionWidth<512>, Rows<1280>, 512, 3, 256>(&mut device).await;
    check::<Hidden<5376>, ProjectionWidth<1792>, Rows<9472>, 1024, 10, 256>(&mut device).await;
}

/// Exercises a larger aligned sequence; full long-text model validation is separate.
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn larger_gather_preserves_fused_projection() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    check::<Hidden<5376>, ProjectionWidth<1792>, Rows<9984>, 512, 20, 256>(&mut device).await;
}

/// One first/body gather envelope serves the runtime work extents required through 2048 text
/// tokens. This checks compact projection placement; full model packing is validated separately.
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn joined_gather_prefixes_preserve_projection() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    check::<Hidden<5376>, ProjectionWidth<1792>, Rows<10496>, 1024, 11, 256>(&mut device).await;
}
