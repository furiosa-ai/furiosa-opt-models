//! Video decoding on the pipeline's original four chips.
//!
//! The retained compiler cannot lower the complete 36-layer entry. Separate entries retain
//! its arithmetic and order, queued through local workspace buffers. Across the 49 groups,
//! the next group's computation starts before downloading and stitching its predecessor.

use super::{flight::Flight, weights::Component};
use crate::residency::{Resident, SharedDevice, block_on};
use furiosa_opt_models::models::diffusion::minimax_h3::video::{
    Patch as PixelChannels, Rows as TileRows,
};
use furiosa_opt_models::models::diffusion::minimax_h3::{embeddings::Chips, video};
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_std::prelude::*;
use std::{io, sync::Arc};

/// Internal component view; an exposed view additionally retains the complete pipeline owner.
pub(super) struct Decoder {
    _owner: Component<video::Decoder>,
    embedding: Resident<video::Embedding>,
    layers: [Resident<video::Layer>; video::DEPTH],
    norm: Resident<video::FinalNorm>,
    head: Resident<video::Head>,
    context: Arc<video::Context>,
    device: SharedDevice,
    active: tokio::sync::Mutex<()>,
}

impl Decoder {
    /// Prime the stages and constant tile metadata before sizing the encoder's memory grant.
    /// The pipeline has already placed these weights on this same original device.
    pub(super) async fn from_resident(
        owner: Component<video::Decoder>,
        device: SharedDevice,
    ) -> io::Result<Self> {
        if device.placement().chips.len() != 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "H3 video decoding requires the pipeline's four-chip device",
            ));
        }
        device.admit(owner.resident())?.landed().await?;
        let context = device.with_device(|device| {
            kernel::embedding.load(device).map_err(io::Error::other)?;
            kernel::layer.load(device).map_err(io::Error::other)?;
            kernel::norm.load(device).map_err(io::Error::other)?;
            kernel::head.load(device).map_err(io::Error::other)?;
            video::Context::load(&mut furiosa_opt_models::load::Context::new(Some(device)))
                .map(Arc::new)
                .map_err(io::Error::other)
        })?;
        let resident = owner.resident();
        let embedding = resident.part::<video::Embedding>("embedding", |tree| &tree.embedding);
        let layers = std::array::from_fn(|at| {
            resident
                .part::<video::Layer>(&format!("body.layers.{at}"), |tree| &tree.body.layers[at])
        });
        let norm = resident.part::<video::FinalNorm>("norm", |tree| &tree.norm);
        let head = resident.part::<video::Head>("head", |tree| &tree.head);
        Ok(Self {
            _owner: owner,
            embedding,
            layers,
            norm,
            head,
            context,
            device,
            active: tokio::sync::Mutex::new(()),
        })
    }

    /// Decode and stitch the complete group sequence with the original optional tail count.
    /// Refill the original two tile slots in turn. Submit the successor before waiting for
    /// and stitching its predecessor. A flight keeps every shared workspace owner alive until
    /// all accepted readers end, including cancellation and partial submission failure.
    pub(super) async fn decode_frames(
        &self,
        values: &[f32],
        frames: usize,
    ) -> io::Result<Vec<f32>> {
        let latents = video::tiling::Latents::new(values)?;
        let _active = self.active.lock().await;
        self.context
            .rotation
            .low
            .ready()
            .await
            .map_err(io::Error::other)?;
        self.context
            .rotation
            .high
            .ready()
            .await
            .map_err(io::Error::other)?;
        self.context
            .visible
            .ready()
            .await
            .map_err(io::Error::other)?;
        let workspace = Arc::new(
            self.device
                .with_device(|device| Workspace::allocate(device, Arc::clone(&self.context)))?,
        );
        let mut stitch = video::tiling::Stitch::with_frames(frames)?;
        // The original caller pins one pixel-group target for the complete decode. A failed
        // read ends this request, so its target is dropped before any later request can reuse it.
        let mut host = HostBuffer::zero(
            Chips::SIZE * TileRows::SIZE * PixelChannels::SIZE * std::mem::size_of::<f32>(),
        );
        let mut previous = None;
        for index in 0..video::tiling::GROUPS {
            let group = video::tiling::GroupIndex::try_from(index)?;
            let slot = index % 2;
            self.upload(latents.pack(group)?, &workspace.inputs[slot])?;
            // Both input/output slots coexist, while the device's ordered launches reuse the
            // same scratch. A slot is refilled only after its earlier pixels were absorbed.
            let flight = self.submit(Arc::clone(&workspace), slot).await?;
            if let Some((group, prior)) = previous.take() {
                self.absorb(group, prior, &mut host, &mut stitch).await?;
            }
            previous = Some((group, flight));
        }
        let (group, flight) = previous.expect("the fixed profile contains groups");
        self.absorb(group, flight, &mut host, &mut stitch).await?;
        stitch.finish()
    }

    fn upload(&self, packed: video::tiling::Packed, target: &HbmBuffer) -> io::Result<()> {
        let source = packed.into_buffer();
        let mut target = target.alias(..).map_err(io::Error::other)?;
        self.device.with_device(|device| {
            block_on(
                source
                    .to_hbm(&mut device.pdma)
                    .output(&mut target)
                    .into_future(),
            )
            .map_err(io::Error::other)
        })
    }

    async fn absorb(
        &self,
        group: video::tiling::GroupIndex,
        flight: Flight<Arc<Workspace>>,
        host: &mut HostBuffer,
        stitch: &mut video::tiling::Stitch,
    ) -> io::Result<()> {
        let [pixels]: [HbmBuffer; 1] = flight
            .finish()
            .await?
            .try_into()
            .map_err(|_| io::Error::other("H3 video head returned an invalid output count"))?;
        let source = video::Pixels::<Chips>::try_from(pixels)
            .map_err(io::Error::other)?
            .into_buffer();
        self.device.with_device(|device| {
            // Own the original pixel allocation through the completed DMA. This synchronous
            // boundary cannot be cancelled between submission and completion. The next copy
            // starts only after this group's scoped unpacking and blending have finished.
            block_on(
                source
                    .to_host(&mut device.pdma)
                    .output(&mut *host)
                    .into_future(),
            )
            .map_err(io::Error::other)
        })?;
        let values = bytemuck::try_cast_slice::<u8, f32>(host)
            .map_err(|error| io::Error::other(format!("pinned H3 video pixels: {error:?}")))?;
        stitch.accept(group, values)
    }

    /// Size the actual queued workspace with both input slots and both pixel groups held.
    /// Those are the buffers that can coexist during decode's download/prefetch overlap.
    pub(super) fn workspace_bytes(&self) -> io::Result<usize> {
        self.device.with_device(|device| {
            super::memory::measure(device, |device| {
                Workspace::allocate(device, Arc::clone(&self.context))
            })
        })
    }

    async fn submit(
        &self,
        workspace: Arc<Workspace>,
        slot: usize,
    ) -> io::Result<Flight<Arc<Workspace>>> {
        let input: video::Latent = alias(&workspace.inputs[slot], ..)?;
        input.ready().await.map_err(io::Error::other)?;
        let mut flight = Flight::new(workspace, video::DEPTH + 3);
        let ready = self.device.admit(&self.embedding)?.landed().await?;
        self.device.launch(ready, |launches, model| {
            let work = flight.held_mut();
            let input: video::Latent = alias(&work.inputs[slot], ..)?;
            let mut output: video::Residual = alias(&work.rows[0], ..)?;
            let accepted = launches.forward_into_tracked(
                model,
                kernel::embedding,
                work.context.as_ref(),
                &mut ((), ()),
                &input,
                &mut output,
            )?;
            flight.submitted(accepted);
            Ok(())
        })?;
        for (at, layer) in self.layers.iter().enumerate() {
            let ready = self.device.admit(layer)?.landed().await?;
            self.device.launch(ready, |launches, model| {
                let work = flight.held_mut();
                let input: video::Residual = alias(&work.rows[at % 2], ..)?;
                let mut output: video::Residual = alias(&work.rows[(at + 1) % 2], ..)?;
                let accepted = launches.forward_into_tracked(
                    model,
                    kernel::layer,
                    work.context.as_ref(),
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
            let input: video::Residual = alias(&work.rows[video::DEPTH % 2], ..)?;
            let mut output: video::Normalized = alias(&work.normalized, ..)?;
            let accepted = launches.forward_into_tracked(
                model,
                kernel::norm,
                work.context.as_ref(),
                &mut (),
                &input,
                &mut output,
            )?;
            flight.submitted(accepted);
            Ok(())
        })?;
        let ready = self.device.admit(&self.head)?.landed().await?;
        self.device.launch(ready, |launches, model| {
            let work = flight.held_mut();
            let input: video::Normalized = alias(&work.normalized, ..)?;
            let mut output: video::Pixels = alias(&work.pixels[slot], ..)?;
            let accepted = launches.forward_into_tracked(
                model,
                kernel::head,
                work.context.as_ref(),
                &mut (),
                &input,
                &mut output,
            )?;
            flight.accepted(output.into_buffer(), accepted);
            Ok(())
        })?;
        Ok(flight)
    }
}

struct Workspace {
    inputs: [HbmBuffer; 2],
    context: Arc<video::Context>,
    rows: [HbmBuffer; 2],
    normalized: HbmBuffer,
    pixels: [HbmBuffer; 2],
}

impl Workspace {
    fn allocate(device: &mut Device, context: Arc<video::Context>) -> io::Result<Self> {
        let rows = || {
            video::Residual::<Chips>::alloc(device)
                .map(HbmTensor::into_buffer)
                .map_err(io::Error::other)
        };
        let input = || {
            video::Latent::<Chips>::alloc(device)
                .map(HbmTensor::into_buffer)
                .map_err(io::Error::other)
        };
        let pixels = || {
            video::Pixels::<Chips>::alloc(device)
                .map(HbmTensor::into_buffer)
                .map_err(io::Error::other)
        };
        Ok(Self {
            inputs: [input()?, input()?],
            context,
            rows: [rows()?, rows()?],
            normalized: video::Normalized::<Chips>::alloc(device)
                .map_err(io::Error::other)?
                .into_buffer(),
            pixels: [pixels()?, pixels()?],
        })
    }
}

/// Concrete four-chip bindings of the existing public Module implementations.
mod kernel {
    use furiosa_opt_models::models::diffusion::minimax_h3::{embeddings::Chips, video};
    use furiosa_opt_models::{Module, tensor::memory::Hbm};
    use furiosa_opt_std::prelude::*;

    #[device(chip = 4)]
    pub(super) fn embedding(
        device: &mut Device,
        module: &&video::Embedding<Chips, &Hbm>,
        context: &video::Context,
        state: &mut ((), ()),
        input: &video::Latent,
    ) -> video::Residual {
        module.forward(device, context, state, input)
    }

    #[device(chip = 4)]
    pub(super) fn layer(
        device: &mut Device,
        module: &&video::Layer<Chips, &Hbm>,
        context: &video::Context,
        state: &mut (),
        input: video::Residual,
    ) -> video::Residual {
        module.forward(device, context, state, input)
    }

    #[device(chip = 4)]
    pub(super) fn norm(
        device: &mut Device,
        module: &&video::FinalNorm<Chips, &Hbm>,
        context: &video::Context,
        state: &mut (),
        input: &video::Residual,
    ) -> video::Normalized {
        module.forward(device, context, state, input)
    }

    #[device(chip = 4)]
    pub(super) fn head(
        device: &mut Device,
        module: &&video::Head<Chips, &Hbm>,
        context: &video::Context,
        state: &mut (),
        input: &video::Normalized,
    ) -> video::Pixels {
        module.forward(device, context, state, input)
    }
}
