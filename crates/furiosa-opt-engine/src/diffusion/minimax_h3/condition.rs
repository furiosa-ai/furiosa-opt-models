//! Request-level conditioning with separate projection, refiner-layer and final-norm launches.
//! Runtime prompt padding and distribution supply chip-local text rows for denoising.

use super::{flight::Flight, kernel, weights::Component};
use crate::residency::{Resident, SharedDevice, block_on};
use furiosa_opt_models::models::diffusion::minimax_h3::{
    denoiser::Denoiser,
    embeddings::{self, Chips, Condition as Encoded, Width},
    packing::{Capacity, DEFAULT_MAX_TEXT, Modality, PARTS, Plan},
    transformer,
};
use furiosa_opt_models::ops::{
    attention::flash,
    index::{gather::RowStorage, packed::Span},
};
use furiosa_opt_models::tensor::{axis::Rows, memory::alias, scalar::Value};
use furiosa_opt_std::prelude::*;
use std::{io, sync::Arc};

type Projection = embeddings::Text<Encoded, Width, Chips>;
type Layer = transformer::RefinerLayer;
type Norm = transformer::Norm;
type RefinerKeys = Rows<3>;
type Hidden<const N: usize> = HbmTensor<bf16, Chips, m![{ Symbol<Span<N, N>> }, Width]>;

/// Completed chip-local text remains bound to the geometry that selected its rows.
pub(super) struct Text<T: AxisName> {
    pub(super) plan: Plan,
    pub(super) rows: HbmTensor<bf16, Chips, m![T, Width]>,
}

/// An internal view. External component views additionally retain the complete pipeline owner.
pub(super) struct Conditioner<const N: usize, T: AxisName> {
    _owner: Component<Denoiser>,
    projection: Resident<Projection>,
    layers: [Resident<Layer>; 2],
    norm: Resident<Norm>,
    device: SharedDevice,
    shape: std::marker::PhantomData<T>,
}

impl<const N: usize, T: AxisName + Send + Sync> Conditioner<N, T> {
    /// Borrow the resident stages and shared device. Prime each stage's entry before measuring
    /// the encoder's resident-prefix memory grant.
    pub(super) async fn from_resident(
        owner: Component<Denoiser>,
        device: SharedDevice,
    ) -> io::Result<Self> {
        let capacity = Capacity::new(N)?;
        if N > DEFAULT_MAX_TEXT
            || !N.is_multiple_of(64)
            || T::SIZE < capacity.plan(N)?.text_rows_per_part()
        {
            return Err(invalid(
                "compiled text conditioning shapes do not hold the configured capacity",
            ));
        }
        let resident = owner.resident();
        let projection =
            resident.part::<Projection>("embeddings.text", |tree| &tree.embeddings.text);
        let layers = std::array::from_fn(|at| {
            resident.part::<Layer>(&format!("refiner.blocks.layers.{at}"), |tree| {
                &tree.refiner.0.0.layers[at]
            })
        });
        let norm = resident.part::<Norm>("refiner.final_norm", |tree| &tree.refiner.1.0);
        device.admit(&projection)?.landed().await?;
        for layer in &layers {
            device.admit(layer)?.landed().await?;
        }
        device.admit(&norm)?.landed().await?;
        device.with_device(|device| {
            Loadable::<(
                &mut Device,
                (
                    &kernel::Condition<'_, '_>,
                    std::marker::PhantomData<[(); 4]>,
                ),
                (),
                &mut (),
                (
                    &RowStorage<Encoded, bf16, Chips>,
                    &HbmTensor<i32, Chips, m![{Symbol<Span<N,N>>}]>,
                ),
            )>::load(kernel::condition, device)
            .map_err(io::Error::other)?;
            Loadable::<(
                &mut Device,
                (
                    &kernel::RefinerLayer<'_, '_>,
                    std::marker::PhantomData<[(); 4]>,
                ),
                (&flash::Visible<RefinerKeys, Chips>, &Value<Chips>),
                &mut (),
                Hidden<N>,
            )>::load(kernel::refine, device)
            .map_err(io::Error::other)?;
            Loadable::<(
                &mut Device,
                (
                    &kernel::RefinerNorm<'_, '_>,
                    std::marker::PhantomData<[(); 4]>,
                ),
                (),
                &mut (),
                &Hidden<N>,
            )>::load(kernel::refine_norm, device)
            .map_err(io::Error::other)?;
            Ok(())
        })?;
        Ok(Self {
            _owner: owner,
            projection,
            layers,
            norm,
            device,
            shape: std::marker::PhantomData,
        })
    }

    /// Launch projection, refiner layer 0, refiner layer 1 and final normalization in order.
    /// Read the first of four output replicas, zero non-token rows and distribute text to the
    /// four parts once per request. The checked plan determines each part's valid text rows.
    pub(super) async fn prepare(&self, plan: Plan, rows: Vec<HbmBuffer>) -> io::Result<Text<T>> {
        let prepared = Prepared::<N>::new(plan, rows)?;
        let metadata = Arc::new(
            self.device
                .with_device(|device| block_on(prepared.upload(device)))?,
        );
        for row in metadata.rows.iter() {
            row.ready().await.map_err(io::Error::other)?;
        }
        let workspace = self
            .device
            .with_device(|device| Workspace::allocate(device, metadata))?;
        let mut flight = Flight::new(workspace, self.layers.len() + 2);
        let ready = self.device.admit(&self.projection)?.landed().await?;
        self.device.launch(ready, |launches, model| {
            let work = flight.held_mut();
            let mut output: Hidden<N> = alias(&work.hidden[0], ..)?;
            let accepted = launches.on::<4>().forward_into_tracked(
                kernel::Condition { model },
                kernel::condition,
                (),
                &mut (),
                (&work.metadata.rows, &work.metadata.offsets),
                &mut output,
            )?;
            flight.submitted(accepted);
            Ok(())
        })?;
        for (at, layer) in self.layers.iter().enumerate() {
            let ready = self.device.admit(layer)?.landed().await?;
            self.device.launch(ready, |launches, model| {
                let work = flight.held_mut();
                let input: Hidden<N> = alias(&work.hidden[at % 2], ..)?;
                let mut output: Hidden<N> = alias(&work.hidden[(at + 1) % 2], ..)?;
                let accepted = launches.on::<4>().forward_into_tracked(
                    kernel::RefinerLayer { model },
                    kernel::refine,
                    (&work.metadata.visible, &work.metadata.bound),
                    &mut (),
                    input,
                    &mut output,
                )?;
                flight.submitted(accepted);
                Ok(())
            })?;
        }
        let ready = self.device.admit(&self.norm)?.landed().await?;
        self.device.launch(ready, |launches, model| {
            let work = flight.held_mut();
            let input: Hidden<N> = alias(&work.hidden[self.layers.len() % 2], ..)?;
            let mut output: Hidden<N> = alias(&work.refined, ..)?;
            let accepted = launches.on::<4>().forward_into_tracked(
                kernel::RefinerNorm { model },
                kernel::refine_norm,
                (),
                &mut (),
                &input,
                &mut output,
            )?;
            flight.accepted(output.into_buffer(), accepted);
            Ok(())
        })?;
        let hidden = completed(flight).await?;
        let refined: Hidden<N> = alias(&hidden, ..)?;
        refined.ready().await.map_err(io::Error::other)?;
        let refined = self.device.with_device(|device| {
            block_on(
                refined
                    .to_host::<m![{Chips},{Symbol<Span<N,N>>},Width]>(&mut device.pdma)
                    .into_future(),
            )
            .map(HostTensor::into_vec)
            .map_err(io::Error::other)
        })?;
        // Every chip refines the same prompt. Copy valid tokens from the first replica into
        // chip-local text parts, leaving every non-token row zero.
        let mut parts = vec![bf16::from_bits(0); PARTS * T::SIZE * Width::SIZE];
        for chip in 0..PARTS {
            let span = plan
                .part(chip)
                .expect("checked four-part plan")
                .span(Modality::Text);
            for row in span.valid_rows() {
                let token = plan
                    .token(chip, row)
                    .expect("valid text rows have semantic tokens");
                let source = token.modality_row() * Width::SIZE;
                let destination = (chip * T::SIZE + row - span.rows().start) * Width::SIZE;
                parts[destination..destination + Width::SIZE]
                    .copy_from_slice(&refined[source..source + Width::SIZE]);
            }
        }
        let rows = self.device.with_device(|device| {
            block_on(
                HostTensor::<bf16, m![{ Chips }, T, Width]>::from_vec(parts)
                    .to_hbm(&mut device.pdma)
                    .into_future(),
            )
            .map_err(io::Error::other)
        })?;
        Ok(Text { plan, rows })
    }

    /// Maximum conditioning storage, including independently allocated encoder rows. That
    /// source form also covers the framework's smaller single-upload row-view representation.
    /// Metadata copies initialize their own constants; no kernel reads the sizing rows.
    pub(super) fn workspace_bytes(&self) -> io::Result<usize> {
        let plan = Capacity::new(N)?.plan(N)?;
        self.device.with_device(|device| {
            super::memory::measure(device, |device| {
                let rows = (0..N)
                    .map(|_| {
                        <HbmTensor<bf16, Chips, m![Encoded]> as furiosa_opt_std::runtime::DeviceOutput>::alloc(device)
                            .map(HbmTensor::into_buffer)
                            .map_err(io::Error::other)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                let metadata = block_on(Prepared::<N>::new(plan, rows)?.upload(device))?;
                Workspace::allocate(device, Arc::new(metadata))
            })
        })
    }
}

async fn completed<T>(flight: Flight<T>) -> io::Result<HbmBuffer> {
    let [output]: [HbmBuffer; 1] =
        flight.finish().await?.try_into().map_err(|_| {
            io::Error::other("H3 conditioning stage returned an invalid output count")
        })?;
    Ok(output)
}

/// Two alternating refinement slots and a separate final output remain owned through all
/// accepted launches, together with their input rows and metadata.
struct Workspace<const N: usize> {
    metadata: Arc<Metadata<N>>,
    hidden: [HbmBuffer; 2],
    refined: HbmBuffer,
}

impl<const N: usize> Workspace<N> {
    fn allocate(device: &mut Device, metadata: Arc<Metadata<N>>) -> io::Result<Self> {
        let row = || {
            Hidden::<N>::alloc(device)
                .map(HbmTensor::into_buffer)
                .map_err(io::Error::other)
        };
        Ok(Self {
            metadata,
            hidden: [row()?, row()?],
            refined: row()?,
        })
    }
}

struct Prepared<const N: usize> {
    rows: RowStorage<Encoded, bf16, Chips>,
    offsets: Vec<i32>,
    tokens: usize,
}

impl<const N: usize> Prepared<N> {
    fn new(plan: Plan, rows: Vec<HbmBuffer>) -> io::Result<Self> {
        if plan.capacity().max_text() != N || rows.len() != plan.text_tokens() {
            return Err(invalid(
                "encoder rows or capacity differ from the conditioning request",
            ));
        }
        let rows = rows
            .into_iter()
            .map(HbmTensor::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(io::Error::other)?
            .into();
        let offsets = (0..N)
            .map(|row| {
                row.min(plan.text_tokens())
                    .checked_mul(Encoded::SIZE)
                    .and_then(|v| v.checked_mul(2))
                    .and_then(|v| i32::try_from(v).ok())
                    .ok_or_else(|| invalid("H3 text row byte offset exceeds I32"))
            })
            .collect::<io::Result<_>>()?;
        Ok(Self {
            rows,
            offsets,
            tokens: plan.text_tokens(),
        })
    }

    async fn upload(mut self, device: &mut Device) -> io::Result<Metadata<N>> {
        let zero: HbmTensor<bf16, Chips, m![Encoded]> = HostTensor::<bf16, m![Encoded]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .map_err(io::Error::other)?;
        self.rows.extend([zero]);
        let offsets = HostTensor::<i32, m![{Symbol<Span<N,N>>}]>::from_vec(self.offsets)
            .to_hbm(&mut device.pdma)
            .await
            .map_err(io::Error::other)?;
        let keys = RefinerKeys::SIZE * flash::Span::SIZE;
        let visible =
            HostTensor::<bf16, m![{Symbol<RefinerKeys>},{Symbol<flash::Span>}]>::from_vec(
                (0..keys).map(|row| bf16::from_f32(if row < self.tokens { 1. } else { 0. })),
            )
            .to_hbm(&mut device.pdma)
            .await
            .map_err(io::Error::other)?;
        let bound = HostTensor::<i32, m![1]>::from_vec([
            i32::try_from(keys).expect("refiner bound fits I32")
        ])
        .to_hbm(&mut device.pdma)
        .await
        .map_err(io::Error::other)?;
        Ok(Metadata {
            rows: self.rows,
            offsets,
            visible,
            bound,
        })
    }
}

struct Metadata<const N: usize> {
    rows: RowStorage<Encoded, bf16, Chips>,
    offsets: HbmTensor<i32, Chips, m![{Symbol<Span<N,N>>}]>,
    visible: flash::Visible<RefinerKeys, Chips>,
    bound: Value<Chips>,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
