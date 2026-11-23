//! Video decoding on the pipeline's shared four-chip device context.
//!
//! Embedding, each of 36 transformer layers, normalization and readout launch separately
//! through the shared Module entry. Ordered launches reuse local workspace rows. Across the
//! 49 groups, submit the next group's launches before downloading and stitching its predecessor.

use super::{flight::Flight, weights::Component};
use crate::residency::{Resident, SharedDevice, block_on};
use furiosa_opt_models::models::diffusion::minimax_h3::video::{
    Patch as PixelChannels, Rows as TileRows,
};
use furiosa_opt_models::models::diffusion::minimax_h3::{embeddings::Chips, video};
use furiosa_opt_models::tensor::memory::{Hbm, alias};
use furiosa_opt_std::prelude::*;
use std::{io, sync::Arc};

/// Checked framework video options owning one fixed-profile latent payload.
pub(super) struct Input {
    values: Vec<f32>,
    frames: video::tiling::FrameCount,
}

impl Input {
    /// Validate raw layout, payload and options before admission or device work.
    pub(super) fn new(
        shape: &[usize],
        values: Vec<f32>,
        frame_num: Option<usize>,
        process_image: bool,
    ) -> io::Result<Self> {
        let invalid = || {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "decode_base requires float32 [1,24,37,48,84], frame_num in 1..=124 and process_image=False",
            )
        };
        if shape != video::tiling::LATENT_SHAPE || process_image {
            return Err(invalid());
        }
        video::tiling::Latents::new(&values).map_err(|_| invalid())?;
        Ok(Self {
            values,
            frames: video::tiling::FrameCount::try_from(
                frame_num.unwrap_or(video::tiling::OUTPUT_SHAPE[2]),
            )
            .map_err(|_| invalid())?,
        })
    }

    fn latents(&self) -> video::tiling::Latents<'_> {
        video::tiling::Latents::new(&self.values).expect("checked owned video payload")
    }

    /// The returned channel-first shape with the checked tail-frame count.
    pub(super) fn output_shape(&self) -> [usize; 5] {
        let mut shape = video::tiling::OUTPUT_SHAPE;
        shape[2] = self.frames.get();
        shape
    }
}

/// Internal component view; an exposed view additionally retains the complete pipeline owner.
pub(super) struct Decoder {
    _owner: Component<video::Decoder>,
    embedding: Resident<video::Embedding>,
    layers: [Resident<video::Layer>; video::DEPTH],
    norm: Resident<video::FinalNorm>,
    head: Resident<video::Head>,
    context: Arc<video::Context>,
    device: SharedDevice,
}

impl Decoder {
    /// Prime the stages and constant tile metadata before sizing the encoder's memory grant.
    /// The pipeline has already placed these weights on the shared device context.
    pub(super) async fn from_resident(
        owner: Component<video::Decoder>,
        device: SharedDevice,
    ) -> io::Result<Self> {
        device.admit(owner.resident())?.landed().await?;
        let context = device.with_device(|device| {
            Loadable::<(
                &mut Device,
                &&video::Embedding<Chips, &Hbm>,
                &video::Context,
                &mut ((), ()),
                &video::Latent,
            )>::load(kernel::embedding, device)
            .map_err(io::Error::other)?;
            Loadable::<(
                &mut Device,
                &&video::Layer<Chips, &Hbm>,
                &video::Context,
                &mut (),
                video::Residual,
            )>::load(kernel::layer, device)
            .map_err(io::Error::other)?;
            Loadable::<(
                &mut Device,
                &&video::FinalNorm<Chips, &Hbm>,
                &video::Context,
                &mut (),
                &video::Residual,
            )>::load(kernel::norm, device)
            .map_err(io::Error::other)?;
            Loadable::<(
                &mut Device,
                &&video::Head<Chips, &Hbm>,
                &video::Context,
                &mut (),
                &video::Normalized,
            )>::load(kernel::head, device)
            .map_err(io::Error::other)?;
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
        })
    }

    /// Allocate the original two-slot workspace and upload the checked first group. Later
    /// groups are packed and uploaded as the decoding stream consumes their predecessors.
    pub(super) async fn prepare(&self, input: Input) -> io::Result<Prepared> {
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
        let stitch = video::tiling::Stitch::with_frames(input.frames)?;
        // Reuse one host pixel-group target for the complete decode. A failed
        // read ends this request, so its target is dropped before any later request can reuse it.
        let host = HostBuffer::zero(
            Chips::SIZE * TileRows::SIZE * PixelChannels::SIZE * std::mem::size_of::<f32>(),
        );
        let first = video::tiling::GroupIndex::try_from(0)?;
        self.upload(input.latents().pack(first)?, &workspace.inputs[0])?;
        Ok(Prepared {
            input,
            workspace,
            stitch,
            host,
        })
    }

    /// Submit the prepared first group. Remaining groups are explicitly prepared and submitted
    /// as the host consumes the stream; the two reusable slots cannot hold final video frames.
    pub(super) async fn decode(&self, prepared: Prepared) -> io::Result<Decoding> {
        let first = video::tiling::GroupIndex::try_from(0)?;
        let flight = self.submit(Arc::clone(&prepared.workspace), 0).await?;
        Ok(Decoding {
            previous: Some((first, flight)),
            prepared,
        })
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
            // Retain the device pixel allocation until DMA completes. This synchronous
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

/// Checked host latents, the original workspace and the uploaded first group's operands.
pub(super) struct Prepared {
    input: Input,
    workspace: Arc<Workspace>,
    stitch: video::tiling::Stitch,
    host: HostBuffer,
}

/// A two-slot decoding stream, not a tensor of final frames. Drain accepted readers before
/// releasing workspace storage, including a conversion failure after submitting a successor.
pub(super) struct Decoding {
    previous: Option<(video::tiling::GroupIndex, Flight<Arc<Workspace>>)>,
    prepared: Prepared,
}

impl Decoding {
    /// Wait for the currently submitted group without advancing or consuming the stream.
    pub(super) async fn wait(&self) -> io::Result<()> {
        self.previous
            .as_ref()
            .expect("submitted group")
            .1
            .wait()
            .await
    }

    pub(super) async fn to_host(mut self, decoder: &Decoder) -> io::Result<Vec<f32>> {
        for index in 1..video::tiling::GROUPS {
            let group = video::tiling::GroupIndex::try_from(index)?;
            let slot = index % 2;
            let work = &mut self.prepared;
            decoder.upload(
                work.input.latents().pack(group)?,
                &work.workspace.inputs[slot],
            )?;
            let flight = decoder.submit(Arc::clone(&work.workspace), slot).await?;
            let (prior_group, prior) = self.previous.take().expect("submitted predecessor");
            decoder
                .absorb(prior_group, prior, &mut work.host, &mut work.stitch)
                .await?;
            self.previous = Some((group, flight));
        }
        let (group, flight) = self
            .previous
            .take()
            .expect("the fixed profile contains groups");
        let work = &mut self.prepared;
        decoder
            .absorb(group, flight, &mut work.host, &mut work.stitch)
            .await?;
        self.prepared.stitch.finish()
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

/// Four-chip stage bindings for the shared Module entry.
mod kernel {
    pub(super) use crate::forward_four as embedding;
    pub(super) use crate::forward_four as head;
    pub(super) use crate::forward_four as layer;
    pub(super) use crate::forward_four as norm;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_video_decode_options_before_device_work() {
        let shape = video::tiling::LATENT_SHAPE;
        let values = vec![0.0; shape.into_iter().product()];
        for frames in [None, Some(1), Some(124)] {
            let input = Input::new(&shape, values.clone(), frames, false).unwrap();
            assert_eq!(
                input.output_shape(),
                [1, 3, frames.unwrap_or(124), 768, 1344]
            );
        }
        for (shape, values, frames, process_image) in [
            (&[1, 24, 37, 84, 48][..], &values[..], None, false),
            (&shape, &values[..values.len() - 1], None, false),
            (&shape, &values[..], Some(0), false),
            (&shape, &values[..], Some(125), false),
            (&shape, &values[..], None, true),
        ] {
            let error = Input::new(shape, values.to_vec(), frames, process_image)
                .err()
                .unwrap();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(
                error.to_string(),
                "decode_base requires float32 [1,24,37,48,84], frame_num in 1..=124 and process_image=False"
            );
        }
    }
}
