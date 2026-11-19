//! One uncached step using request-owned text and metadata, with bounded accepted work.

use super::{condition, flight::Flight, kernel, request, weights::Component};
use crate::residency::{Resident, SharedDevice};
use furiosa_opt_models::models::diffusion::minimax_h3::{
    denoiser::{self, DEPTH, Denoiser},
    embeddings::{Chips, Latent, Patch, Width},
};
use furiosa_opt_models::tensor::axis::Rows;
use furiosa_opt_std::prelude::*;
use std::{io, sync::Arc};

pub(super) type Times = HbmTensor<f32, Chips, m![{ Symbol<Rows<64>> }, 1 # 8]>;
pub(super) type Video<V> = HbmTensor<f32, Chips, m![V, Patch]>;
pub(super) type Audio<A> = HbmTensor<f32, Chips, m![A, Latent]>;

/// Conditioning and numerical metadata retained together for repeated denoising steps.
/// Construction rejects different request geometry before any numerical launch.
pub(super) struct State<R: AxisName, K: AxisName, T: AxisName, A, V> {
    pub(super) request: request::Request<R, K, T, A, V>,
    pub(super) text: condition::Text<T>,
}

impl<R: AxisName, K: AxisName, T: AxisName, A, V> State<R, K, T, A, V> {
    pub(super) fn new(
        request: request::Request<R, K, T, A, V>,
        text: condition::Text<T>,
    ) -> io::Result<Self> {
        if request.plan != text.plan {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "conditioned text geometry differs from the denoising request",
            ));
        }
        Ok(Self { request, text })
    }
}

pub(super) struct Runner<R, K, T, A, V> {
    _owner: Component<Denoiser>,
    embedding: Resident<denoiser::Embeddings>,
    blocks: [Resident<denoiser::Block>; DEPTH],
    heads: Resident<denoiser::Heads>,
    device: SharedDevice,
    shape: std::marker::PhantomData<(R, K, T, A, V)>,
}

type Hidden<R> = HbmTensor<bf16, Chips, m![R, Width]>;

impl<
    R: AxisName + Send + Sync,
    K: AxisName + Send + Sync,
    T: AxisName + Send + Sync,
    A: AxisName + Send + Sync,
    V: AxisName + Send + Sync,
> Runner<R, K, T, A, V>
{
    /// Borrow stages of the already placed tree. The original execution launches embeddings,
    /// each block and heads separately; the same compiled block entry serves all 50 layers.
    /// Prime entries before sizing the encoder's resident prefix.
    pub(super) async fn from_resident(
        owner: Component<Denoiser>,
        device: SharedDevice,
    ) -> io::Result<Self> {
        if device.placement().chips.len() != 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "H3 denoising requires the pipeline's four-chip device",
            ));
        }
        device.admit(owner.resident())?.landed().await?;
        device.with_device(|device| {
            Loadable::<(
                &mut Device,
                &kernel::Embed<'_, '_>,
                &request::Inputs<R, K, T, A, V>,
                &mut (),
                (&Times, (&Video<V>, &Audio<A>), &Hidden<T>),
            )>::load(kernel::embed, device)
            .map_err(io::Error::other)?;
            Loadable::<(
                &mut Device,
                &kernel::Block<'_, '_>,
                &request::Inputs<R, K, T, A, V>,
                &mut (),
                (Hidden<R>, &kernel::EmbeddedTime),
            )>::load(kernel::block, device)
            .map_err(io::Error::other)?;
            Loadable::<(
                &mut Device,
                &kernel::Heads<'_, '_>,
                &request::Inputs<R, K, T, A, V>,
                &mut (),
                (&Hidden<R>, &kernel::EmbeddedTime),
            )>::load(kernel::heads, device)
            .map_err(io::Error::other)?;
            Ok(())
        })?;
        let resident = owner.resident();
        let embedding =
            resident.part::<denoiser::Embeddings>("embeddings", |tree| &tree.embeddings);
        let blocks = std::array::from_fn(|at| {
            resident.part::<denoiser::Block>(&format!("blocks.layers.{at}"), |tree| {
                &tree.blocks.layers[at]
            })
        });
        let heads = resident.part::<denoiser::Heads>("heads", |tree| &tree.heads);
        Ok(Self {
            _owner: owner,
            embedding,
            blocks,
            heads,
            device,
            shape: std::marker::PhantomData,
        })
    }

    /// Queue the original embeddings -> blocks -> heads sequence through two caller-owned
    /// local row slots. Ordered launches reuse those slots; no shared returned intermediate or
    /// host wait is inserted between blocks. The flight drains every accepted reader before
    /// releasing the workspace on success, partial submission failure or cancellation.
    pub(super) async fn forward(
        &self,
        state: Arc<State<R, K, T, A, V>>,
        times: Times,
        video: Video<V>,
        audio: Audio<A>,
    ) -> io::Result<(Video<V>, Audio<A>)> {
        times.ready().await.map_err(io::Error::other)?;
        video.ready().await.map_err(io::Error::other)?;
        audio.ready().await.map_err(io::Error::other)?;
        state.text.rows.ready().await.map_err(io::Error::other)?;
        let workspace = self
            .device
            .with_device(|device| Workspace::allocate(device, state, times, video, audio))?;
        // One embedding, DEPTH blocks and one completion retained with each of two head outputs.
        let mut flight = Flight::new(workspace, DEPTH + 3);
        let ready = self.device.admit(&self.embedding)?.landed().await?;
        self.device.launch(ready, |launches, model| {
            let work = flight.held_mut();
            let mut output: (Hidden<R>, kernel::EmbeddedTime) = (
                furiosa_opt_models::tensor::memory::alias(&work.hidden[0], ..)?,
                furiosa_opt_models::tensor::memory::alias(&work.time, ..)?,
            );
            let accepted = launches.forward_into_tracked(
                kernel::Embed { model },
                kernel::embed,
                &work.state.request.inputs,
                &mut (),
                (
                    &work.times,
                    (&work.video, &work.audio),
                    &work.state.text.rows,
                ),
                &mut output,
            )?;
            flight.submitted(accepted);
            Ok(())
        })?;
        for (at, block) in self.blocks.iter().enumerate() {
            let ready = self.device.admit(block)?.landed().await?;
            self.device.launch(ready, |launches, model| {
                let work = flight.held_mut();
                let input: Hidden<R> =
                    furiosa_opt_models::tensor::memory::alias(&work.hidden[at % 2], ..)?;
                let time: kernel::EmbeddedTime =
                    furiosa_opt_models::tensor::memory::alias(&work.time, ..)?;
                let mut output: Hidden<R> =
                    furiosa_opt_models::tensor::memory::alias(&work.hidden[(at + 1) % 2], ..)?;
                let accepted = launches.forward_into_tracked(
                    kernel::Block { model },
                    kernel::block,
                    &work.state.request.inputs,
                    &mut (),
                    (input, &time),
                    &mut output,
                )?;
                flight.submitted(accepted);
                Ok(())
            })?;
        }
        let ready = self.device.admit(&self.heads)?.landed().await?;
        self.device.launch(ready, |launches, model| {
            let work = flight.held_mut();
            let input: Hidden<R> =
                furiosa_opt_models::tensor::memory::alias(&work.hidden[DEPTH % 2], ..)?;
            let time: kernel::EmbeddedTime =
                furiosa_opt_models::tensor::memory::alias(&work.time, ..)?;
            let mut output: (Video<V>, Audio<A>) = (
                furiosa_opt_models::tensor::memory::alias(&work.velocities.0, ..)?,
                furiosa_opt_models::tensor::memory::alias(&work.velocities.1, ..)?,
            );
            let accepted = launches.forward_into_tracked(
                kernel::Heads { model },
                kernel::heads,
                &work.state.request.inputs,
                &mut (),
                (&input, &time),
                &mut output,
            )?;
            flight.accepted(output.0.into_buffer(), Arc::clone(&accepted));
            flight.accepted(output.1.into_buffer(), accepted);
            Ok(())
        })?;
        let [video, audio]: [HbmBuffer; 2] = flight
            .finish()
            .await?
            .try_into()
            .map_err(|_| io::Error::other("H3 heads returned an invalid media output count"))?;
        Ok((
            Video::try_from(video).map_err(io::Error::other)?,
            Audio::try_from(audio).map_err(io::Error::other)?,
        ))
    }

    /// Measure call storage beside the separately retained request state. Allocate the same
    /// sample, row ring, time embedding and output types as `forward`, without launching them.
    pub(super) fn workspace_bytes(&self, state: &Arc<State<R, K, T, A, V>>) -> io::Result<usize> {
        self.device.with_device(|device| {
            super::memory::measure(device, |device| {
                let times = Times::alloc(device).map_err(io::Error::other)?;
                let video = Video::<V>::alloc(device).map_err(io::Error::other)?;
                let audio = Audio::<A>::alloc(device).map_err(io::Error::other)?;
                Workspace::allocate(device, Arc::clone(state), times, video, audio)
            })
        })
    }
}

/// Original workspace owners stay here even when individual launches lend short-lived aliases.
struct Workspace<R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName> {
    state: Arc<State<R, K, T, A, V>>,
    times: Times,
    video: Video<V>,
    audio: Audio<A>,
    hidden: [HbmBuffer; 2],
    time: HbmBuffer,
    velocities: (HbmBuffer, HbmBuffer),
}

impl<R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName> Workspace<R, K, T, A, V> {
    fn allocate(
        device: &mut Device,
        state: Arc<State<R, K, T, A, V>>,
        times: Times,
        video: Video<V>,
        audio: Audio<A>,
    ) -> io::Result<Self> {
        let row = || {
            Hidden::<R>::alloc(device)
                .map(HbmTensor::into_buffer)
                .map_err(io::Error::other)
        };
        Ok(Self {
            state,
            times,
            video,
            audio,
            hidden: [row()?, row()?],
            time: kernel::EmbeddedTime::alloc(device)
                .map_err(io::Error::other)?
                .into_buffer(),
            velocities: (
                Video::<V>::alloc(device)
                    .map_err(io::Error::other)?
                    .into_buffer(),
                Audio::<A>::alloc(device)
                    .map_err(io::Error::other)?
                    .into_buffer(),
            ),
        })
    }
}
