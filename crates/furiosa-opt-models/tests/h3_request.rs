//! One checked request supplies modulation spans, projection blocks and valid K/V packing.
//! Synthetic inputs exercise the real uploaded metadata and numerical consumers on four chips.

#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use flash::{Duo, Head, Span};
use furiosa_opt_models::Op;
use furiosa_opt_models::models::diffusion::minimax_h3::{
    embeddings::Chips,
    packing::{AUDIO_TOKENS, Capacity, Modality, Plan, VIDEO_TOKENS},
    request::{BlockInputs, BlockPlan},
};
use furiosa_opt_models::ops::attention::{flash, full};
use furiosa_opt_models::ops::collective::Pieces;
use furiosa_opt_models::ops::index::packed::{Group, Rows as PackedRows, Table};
use furiosa_opt_models::ops::normalization::AdaptiveRms;
use furiosa_opt_models::ops::pointwise::SwiGlu;
use furiosa_opt_models::ops::swiglu::{GateUp, Intermediate};
use furiosa_opt_models::tensor::{axis::Rows, form::Weight};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;

axes![Width = 256, Twenty = 20, Nineteen = 19];
type Local = Rows<10496>;
type Joined = Pieces<Local, 4>;
type Input = PackedRows<Local, Width, Chips>;
type Source = PackedRows<Joined, Width, Chips>;
type Product = PackedRows<Local, Intermediate<128>, Chips>;
type Keys<K> = flash::Tokens<K, flash::HeadPairs<Width>, Chips>;
type Modulation = (Table<Width, Chips>, Table<Width, Chips>);
type Norm = AdaptiveRms<Width, EPSILON, Chips>;
const EPSILON: usize = (1e-5f32).to_bits() as usize;
const POISON: u16 = 0x7fc1;

#[device(chip = 4)]
fn consume<K: AxisName>(
    device: &mut Device,
    inputs: &BlockInputs<Local, K>,
    norm: &Norm,
    tables: &Modulation,
    local: &Input,
    source: &Source,
    keys: &mut Keys<K>,
) -> Product {
    let call = inputs.read();
    let normalized = norm.forward(&mut *device, (local, tables, call.spans()));
    // SAFETY: this synthetic consumer interprets the same 256 features as consecutive
    // gate/up halves of 128 features, without changing physical layout or initialized rows.
    let normalized: PackedRows<Local, GateUp<Intermediate<128>>, Chips> =
        unsafe { normalized.into_value().reshape() };
    let product = SwiGlu.forward(&mut *device, &call.prefix().with(&normalized));
    full::PackKeys::<K>::default().write(
        device,
        call.keys().source_work().with(source),
        call.keys(),
        keys,
    );
    product.into_value()
}

fn payload(chip: usize, row: usize, column: usize) -> bf16 {
    bf16::from_f32(((chip * 7 + row * 11 + column * 3) % 31) as f32 / 16. - 15. / 16.)
}

fn source_row(plan: Plan, key: usize) -> usize {
    let (modality, row) = if key < VIDEO_TOKENS {
        (Modality::Video, key)
    } else if key < VIDEO_TOKENS + AUDIO_TOKENS {
        (Modality::Audio, key - VIDEO_TOKENS)
    } else {
        (Modality::Text, key - VIDEO_TOKENS - AUDIO_TOKENS)
    };
    plan.locate(modality, row).unwrap().query_row()
}

async fn run<K: AxisName>(
    device: &mut Device,
    norm: &Norm,
    tables: &Modulation,
    maximum: usize,
    lengths: &[usize],
) {
    let mut keys = Keys::<K>::alloc(device).unwrap();
    for &length in lengths {
        let plan = Capacity::new(maximum).unwrap().plan(length).unwrap();
        let prepared = BlockPlan::<Local, K>::new(plan).unwrap();
        let inputs = prepared.upload(device).await.unwrap();
        let local: Input =
            HostTensor::<bf16, m![{ Chips }, Local, Width]>::from_vec((0..4).flat_map(|_| {
                (0..Local::SIZE).flat_map(move |row| {
                    (0..Width::SIZE).map(move |_| {
                        bf16::from_bits(if row < plan.packed_rows_per_part() {
                            0x3f80
                        } else {
                            POISON
                        })
                    })
                })
            }))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let source: Source =
            HostTensor::<bf16, m![{ Chips }, { Symbol<Joined> }, Width]>::from_vec(
                (0..4).flat_map(|chip| {
                    (0..Joined::SIZE).flat_map(move |row| {
                        let semantic = row < plan.query_rows()
                            && plan
                                .token(
                                    row / plan.work_rows_per_part(),
                                    row % plan.work_rows_per_part(),
                                )
                                .is_some();
                        (0..Width::SIZE).map(move |column| {
                            if semantic {
                                payload(chip, row, column)
                            } else {
                                bf16::from_bits(POISON)
                            }
                        })
                    })
                }),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let result = launch(
            consume,
            (
                &mut *device,
                &inputs,
                norm,
                tables,
                &local,
                &source,
                &mut keys,
            ),
        )
        .await
        .unwrap();
        // The result's allocation tail is deliberately uninitialized: read only computed work.
        let bytes = result
            .into_buffer()
            .slice(.., ..plan.work_rows_per_part() * 128 * 2)
            .unwrap()
            .to_host(&mut device.pdma)
            .await
            .unwrap();
        let activated = bf16::from_f32((1_f64 / (1. + (-1_f64).exp())) as f32).to_bits();
        for chip in 0..4 {
            for row in 0..plan.work_rows_per_part() {
                let expected = if row < plan.packed_rows_per_part() {
                    activated
                } else {
                    0
                };
                for column in 0..128 {
                    let at = ((chip * plan.work_rows_per_part() + row) * 128 + column) * 2;
                    assert_eq!(
                        u16::from_le_bytes([bytes[at], bytes[at + 1]]),
                        expected,
                        "text={length} local chip={chip} row={row} column={column}"
                    );
                }
            }
        }
        let packed = keys
            .to_host::<m![{ Chips }, { Symbol<flash::HeadPairs<Width>> }, Duo, K, Span, Head]>(
                &mut device.pdma,
            )
            .await
            .unwrap()
            .into_vec();
        let capacity = K::SIZE * flash::Span::SIZE;
        for chip in 0..4 {
            for head in 0..2 {
                for key in 0..capacity {
                    let source = (key < plan.key_count()).then(|| source_row(plan, key));
                    for column in 0..128 {
                        let expected = source
                            .map_or(0, |row| payload(chip, row, head * 128 + column).to_bits());
                        let at = ((chip * 2 + head) * capacity + key) * 128 + column;
                        assert_eq!(
                            packed[at].to_bits(),
                            expected,
                            "text={length} key chip={chip} head={head} key={key} column={column}"
                        );
                    }
                }
            }
        }
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires four explicitly selected RNGD chips"]
async fn prepared_request_metadata_drives_all_three_consumers() {
    let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
    let weight = HostTensor::<bf16, m![Width]>::from_vec(vec![bf16::from_f32(1.); Width::SIZE])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let norm = Norm::new(Weight::new(weight));
    let mut table = async || {
        HostTensor::<bf16, m![Group, Width]>::from_vec(vec![
            bf16::from_f32(0.);
            Group::SIZE * Width::SIZE
        ])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
    };
    let tables: Modulation = (table().await, table().await);
    run::<Twenty>(&mut device, &norm, &tables, 2048, &[77, 2048, 129, 77]).await;
    run::<Nineteen>(&mut device, &norm, &tables, 1202, &[1202]).await;
}
