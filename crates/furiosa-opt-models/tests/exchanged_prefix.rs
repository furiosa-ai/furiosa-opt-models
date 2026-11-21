//! Compact whole exchange and bounded grouped projection through a borrowed Module view.

#![feature(register_tool, negative_impls)]
#![register_tool(furiosa_opt)]

use furiosa_opt_models::Module;
use furiosa_opt_models::load::{Checkpoint, Context, Load, Named};
use furiosa_opt_models::modules::parallel::{Exchanged, ExchangedPrefix};
use furiosa_opt_models::ops::attention::geometry::{Hidden, ProjectionWidth};
use furiosa_opt_models::ops::collective::Pieces;
use furiosa_opt_models::ops::collective::prefix;
use furiosa_opt_models::ops::linear::{Grouped, grouped::Layout};
use furiosa_opt_models::tensor::axis::Rows;
use furiosa_opt_models::tensor::live::{BlockPrefix, Live};
use furiosa_opt_models::tensor::parallel::{Parallel, Part, Shard};
use furiosa_opt_models::tensor::scalar::Value;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;

const CHIPS: usize = 4;
const ROW_CLASSES: usize = 29;
const COLUMN_CLASSES: usize = 31;
const POISON: u16 = 0x7fc1;
type Chips = m![{ Symbol<Shard<4>> }];
type B = Rows<256>;
type Split<R, I> = HbmTensor<bf16, Chips, m![{ Symbol<Pieces<R, 4>> }, { Symbol<Part<I, 4>> }]>;
type Local<R, W> = HbmTensor<bf16, Chips, m![R, W]>;
type Owned<I, O> = Exchanged<Named<"projection", Grouped<I, O, Chips>>, Parallel<4>>;
type Borrowed<'a, I, O> = Exchanged<
    Named<"projection", Grouped<I, O, Chips, &'a HbmTensor<bf16, Chips, Layout<I, O>>>>,
    Parallel<4>,
>;
type State<R, O> = (Local<R, O>, Value<Chips>);

// Deliberately not Copy: the view must pass this reader through exactly as supplied.
struct Reader<'a>(&'a Value<Chips>);

/// The real named projection writes caller-owned output. A separate per-chip scalar copy
/// witnesses that the view passes a non-Copy environment and mutable child state through.
struct Writing<L>(L);

impl<I, L> !furiosa_opt_models::Op<I> for Writing<L> {}

impl<'i, 'p, 'e, 's, R: AxisName, I: AxisName, O: AxisName, L>
    Module<&'p BlockPrefix<&'i Local<R, I>, R, B>, Reader<'e>, &'s mut State<R, O>> for Writing<L>
where
    L: for<'x, 'v, 'o, 'u> Module<
            (&'v BlockPrefix<&'x Local<R, I>, R, B>, &'o mut Local<R, O>),
            (),
            &'u mut (),
            Output = Live<(), R>,
        >,
{
    type Output = Live<(), R>;

    fn forward(
        &self,
        device: &mut Device,
        env: Reader<'e>,
        state: &'s mut State<R, O>,
        input: &'p BlockPrefix<&'i Local<R, I>, R, B>,
    ) -> Self::Output {
        env.0
            .view()
            .to_hbm_view(&mut device.tdma, state.1.view_mut());
        self.0.forward(device, (), &mut (), (input, &mut state.0))
    }
}

#[device(chip = 4)]
fn borrowed<R: AxisName, I: AxisName, O: AxisName, G: prefix::Extents>(
    device: &mut Device,
    weights: &Borrowed<'_, I, O>,
    input: &Split<R, I>,
    blocks: &prefix::Inputs<R, G, B, Chips>,
    env: &Value<Chips>,
    state: &mut State<R, O>,
) {
    let prefix = blocks.read().with(input);
    let model = Exchanged::<_, Parallel<4>>::new(Writing(&weights.operation));
    model
        .prefix::<G>()
        .forward(device, Reader(env), state, &prefix);
}

#[device(chip = 4)]
fn owned<R: AxisName, I: AxisName, O: AxisName, S: AxisName>(
    device: &mut Device,
    weights: &Owned<I, O>,
    input: &Split<R, I>,
    blocks: &prefix::Inputs<R, (S,), B, Chips>,
) -> Local<R, O> {
    let prefix = blocks.read().with(input);
    weights
        .prefix::<(S,)>()
        .forward(device, (), &mut (), &prefix)
        .into_value()
}

fn source(source_chip: usize, target_chip: usize, row: usize, column: usize) -> bf16 {
    bf16::from_f32(
        ((row * 11 + column * 5 + source_chip * 17 + target_chip * 3) % ROW_CLASSES) as f32 / 16.0
            - 14.0 / 16.0,
    )
}

fn coefficient(column: usize, k: usize) -> bf16 {
    bf16::from_f32(((column * 17 + k * 7) % COLUMN_CLASSES) as f32 / 64.0 - 15.0 / 64.0)
}

/// Desired target rows join the corresponding local row from all source chips. This computes
/// that logical operation directly, independently of TDMA layouts, aliases and Grouped packing.
/// Dyadic terms and bounded partial sums are exact in FP32, making BF16 bits a strict oracle.
fn reference<I: AxisName>() -> Vec<bf16> {
    (0..CHIPS * ROW_CLASSES * COLUMN_CLASSES)
        .map(|at| {
            let chip = at / (ROW_CLASSES * COLUMN_CLASSES);
            let row = at / COLUMN_CLASSES % ROW_CLASSES;
            let column = at % COLUMN_CLASSES;
            let sum = (0..I::SIZE)
                .map(|k| {
                    f64::from(
                        source(k / (I::SIZE / CHIPS), chip, row, k % (I::SIZE / CHIPS)).to_f32(),
                    ) * f64::from(coefficient(column, k).to_f32())
                })
                .sum::<f64>();
            bf16::from_f32(sum as f32)
        })
        .collect()
}

fn expected(reference: &[bf16], chip: usize, row: usize, column: usize) -> bf16 {
    reference[(chip * ROW_CLASSES + row % ROW_CLASSES) * COLUMN_CLASSES + column % COLUMN_CLASSES]
}

fn checkpoint<I: AxisName, O: AxisName>() -> (tempfile::TempDir, Checkpoint) {
    let directory = tempfile::tempdir().unwrap();
    let bytes = bf16::to_buf(
        &(0..O::SIZE * I::SIZE)
            .map(|at| coefficient(at / I::SIZE, at % I::SIZE))
            .collect::<Vec<_>>(),
    );
    let view = safetensors::tensor::TensorView::new(
        safetensors::Dtype::BF16,
        vec![O::SIZE, I::SIZE],
        &bytes,
    )
    .unwrap();
    safetensors::serialize_to_file(
        [("projection.weight", view)],
        &None,
        &directory.path().join("weights.safetensors"),
    )
    .unwrap();
    let checkpoint = Checkpoint::new(directory.path(), Some(serde_json::json!({}))).unwrap();
    (directory, checkpoint)
}

async fn scalar(device: &mut Device, values: [i32; CHIPS]) -> Value<Chips> {
    HostTensor::<i32, m![{ Chips }, 1]>::from_vec(values)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

async fn input<R: AxisName, I: AxisName>(device: &mut Device, work: usize) -> Split<R, I> {
    let width = I::SIZE / CHIPS;
    let per_chip = CHIPS * R::SIZE * width;
    HostTensor::<bf16, m![{ Chips }, { Symbol<Pieces<R, 4>> }, { Symbol<Part<I, 4>> }]>::from_vec(
        (0..CHIPS * per_chip).map(|at| {
            let source_chip = at / per_chip;
            let compact_row = at / width % (CHIPS * R::SIZE);
            if compact_row < CHIPS * work {
                source(
                    source_chip,
                    compact_row / work,
                    compact_row % work,
                    at % width,
                )
            } else {
                bf16::from_bits(POISON)
            }
        }),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap()
}

async fn check<R: AxisName, I: AxisName, O: AxisName, A: AxisName, D: AxisName, E: AxisName>(
    works: &[usize],
) {
    assert!(
        works
            .iter()
            .all(|&s| [A::SIZE, D::SIZE, E::SIZE].contains(&s))
    );
    let mut device = Device::new(Topology {
        chips: CHIPS as u8,
        pes: 8,
    })
    .unwrap();
    let (_directory, checkpoint) = checkpoint::<I, O>();
    let weights: Owned<I, O> = checkpoint.load(&mut device).unwrap();
    let borrowed_weights = Borrowed::load(&mut Context::new(None), &weights).unwrap();
    assert!(std::ptr::eq(
        weights.operation.0.tensor(),
        borrowed_weights.operation.0.tensor()
    ));
    let reference = reference::<I>();
    for &work in works {
        let prefix = BlockPrefix::<(), R, B>::try_over(work).unwrap();
        let count = prefix::Prefix::<(), R, (A, (D, (E,))), B>::new(prefix)
            .unwrap()
            .upload::<Chips>(&mut device)
            .await
            .unwrap();
        let input = input::<R, I>(&mut device, work).await;
        let witness = [11, 19, 23, 31];
        let env = scalar(&mut device, witness).await;
        let output = HostTensor::<bf16, m![{ Chips }, R, O]>::from_vec(std::iter::repeat_n(
            bf16::from_bits(POISON),
            CHIPS * R::SIZE * O::SIZE,
        ))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let mut state: State<R, O> = (output, scalar(&mut device, [0; CHIPS]).await);
        launch(
            borrowed,
            (
                &mut device,
                &borrowed_weights,
                &input,
                &count,
                &env,
                &mut state,
            ),
        )
        .await
        .unwrap();
        let got = state
            .0
            .to_host::<m![{ Chips }, R, O]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let marker = state
            .1
            .to_host::<m![{ Chips }, 1]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        assert_eq!(marker, witness, "environment/state routing at work {work}");
        for (at, value) in got.iter().enumerate() {
            let chip = at / (R::SIZE * O::SIZE);
            let row = at / O::SIZE % R::SIZE;
            let column = at % O::SIZE;
            let want = if row < work {
                expected(&reference, chip, row, column).to_bits()
            } else {
                POISON
            };
            assert_eq!(
                value.to_bits(),
                want,
                "chip {chip}, row {row}, column {column}, work {work}"
            );
        }
        if work == A::SIZE {
            let owned_count =
                prefix::Prefix::<(), R, (A,), B>::new(BlockPrefix::try_over(work).unwrap())
                    .unwrap()
                    .upload::<Chips>(&mut device)
                    .await
                    .unwrap();
            let output = launch(owned, (&mut device, &weights, &input, &owned_count))
                .await
                .unwrap();
            // Only the initialized prefix is read: owned allocation tails are unspecified.
            let bytes = output
                .into_buffer()
                .slice(.., ..work * O::SIZE * 2)
                .unwrap()
                .to_host(&mut device.pdma)
                .await
                .unwrap();
            for (at, bytes) in bytes.chunks_exact(2).enumerate() {
                let chip = at / (work * O::SIZE);
                let row = at / O::SIZE % work;
                let column = at % O::SIZE;
                assert_eq!(
                    u16::from_le_bytes([bytes[0], bytes[1]]),
                    expected(&reference, chip, row, column).to_bits(),
                    "owned chip {chip}, row {row}, column {column}, work {work}"
                );
            }
        }
    }
}

#[test]
fn named_owned_and_borrowed_views_keep_the_output_independent_of_exchange_storage() {
    type R = Rows<10496>;
    type I = Hidden<7168>;
    type O = ProjectionWidth<5376>;
    fn view<M>()
    where
        for<'i, 'p, 's> M: Module<
                &'p prefix::Prefix<&'i Split<R, I>, R, (Rows<9472>,), B>,
                (),
                &'s mut (),
                Output = Live<Local<R, O>, R>,
            >,
    {
    }
    view::<ExchangedPrefix<'_, Named<"projection", Grouped<I, O, Chips>>, (Rows<9472>,), 4>>();
    type Child<'a> =
        Named<"projection", Grouped<I, O, Chips, &'a HbmTensor<bf16, Chips, Layout<I, O>>>>;
    view::<ExchangedPrefix<'_, Child<'_>, (Rows<9472>,), 4>>();
}

#[test]
fn independent_oracle_distinguishes_source_chip_and_compact_stride() {
    type I = Hidden<512>;
    let reference = reference::<I>();
    assert!((0..COLUMN_CLASSES).any(
        |c| expected(&reference, 0, 7, c).to_bits() != expected(&reference, 1, 7, c).to_bits()
    ));
    let wrong = (0..I::SIZE)
        .map(|k| {
            f64::from(
                source(
                    (k / (I::SIZE / CHIPS) + 1) % CHIPS,
                    1,
                    7,
                    k % (I::SIZE / CHIPS),
                )
                .to_f32(),
            ) * f64::from(coefficient(3, k).to_f32())
        })
        .sum::<f64>();
    assert_ne!(
        expected(&reference, 1, 7, 3).to_bits(),
        bf16::from_f32(wrong as f32).to_bits()
    );
    // A capacity-strided consumer picks a different row inside the compactly packed source.
    let (work, capacity) = (9472, 10496);
    assert_ne!(
        source(2, 1, 7, 3).to_bits(),
        source(2, (capacity + 7) / work, (capacity + 7) % work, 3).to_bits()
    );
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn compact_exchange_named_projection_matches_fp64_at_all_main_work_extents() {
    check::<Rows<10496>, Hidden<512>, ProjectionWidth<256>, Rows<9472>, Rows<9728>, Rows<9984>>(&[
        9472, 9984, 9728, 9472,
    ])
    .await;
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn full_width_exchange_projection_matches_fp64_at_one_block() {
    check::<Rows<768>, Hidden<7168>, ProjectionWidth<5376>, Rows<256>, Rows<512>, Rows<768>>(&[
        256,
    ])
    .await;
}
