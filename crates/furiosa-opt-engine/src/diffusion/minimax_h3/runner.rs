//! One denoiser runner for uncached and TeaCache steps, with bounded accepted work.

use super::{cache, condition, flight::Flight, kernel, request, weights::Component};
use crate::residency::{Launches, Resident, SharedDevice};
use furiosa_opt_models::models::diffusion::minimax_h3::{
    denoiser::{self, DEPTH, Denoiser},
    embeddings::{Chips, Latent, Patch, Width},
};
use furiosa_opt_models::tensor::{axis::Rows, memory::Hbm, memory::alias};
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

// Borrow only the selected workspace slots while the device lock is held.
type EmbeddingSlots<'a, S, A, V> = (
    &'a S,
    &'a Times,
    (&'a Video<V>, &'a Audio<A>),
    (&'a HbmBuffer, &'a HbmBuffer),
);
type BlockSlots<'a, S> = (&'a S, &'a HbmBuffer, &'a HbmBuffer, &'a HbmBuffer);
type HeadSlots<'a, S> = (
    &'a S,
    &'a HbmBuffer,
    &'a HbmBuffer,
    (&'a HbmBuffer, &'a HbmBuffer),
);

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
    /// Prime common and TeaCache entries before sizing the encoder's resident prefix.
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
            cache::prime::<R, K, T, A, V>(device)
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
        self.submit_embeddings(&mut flight, |work| {
            (
                work.state.as_ref(),
                &work.times,
                (&work.video, &work.audio),
                (&work.hidden[0], &work.time),
            )
        })
        .await?;
        self.submit_blocks(&mut flight, |at, work| {
            (
                work.state.as_ref(),
                &work.hidden[at % 2],
                &work.time,
                &work.hidden[(at + 1) % 2],
            )
        })
        .await?;
        self.submit_heads(&mut flight, |work| {
            (
                work.state.as_ref(),
                &work.hidden[DEPTH % 2],
                &work.time,
                (&work.velocities.0, &work.velocities.1),
            )
        })
        .await?;
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

    pub(super) fn device(&self) -> &SharedDevice {
        &self.device
    }

    pub(super) async fn submit_embeddings<W>(
        &self,
        flight: &mut Flight<W>,
        slots: impl for<'a> FnOnce(&'a W) -> EmbeddingSlots<'a, State<R, K, T, A, V>, A, V>,
    ) -> io::Result<()> {
        let ready = self.device.admit(&self.embedding)?.landed().await?;
        self.device.launch(ready, |launches, model| {
            let (state, times, media, (hidden, time)) = slots(flight.held_mut());
            let mut output: (Hidden<R>, kernel::EmbeddedTime) =
                (alias(hidden, ..)?, alias(time, ..)?);
            let accepted = launches.forward_into_tracked(
                kernel::Embed { model },
                kernel::embed,
                &state.request.inputs,
                &mut (),
                (times, media, &state.text.rows),
                &mut output,
            )?;
            flight.submitted(accepted);
            Ok(())
        })
    }

    pub(super) async fn submit_blocks<W>(
        &self,
        flight: &mut Flight<W>,
        mut slots: impl for<'a> FnMut(usize, &'a W) -> BlockSlots<'a, State<R, K, T, A, V>>,
    ) -> io::Result<()> {
        for (at, block) in self.blocks.iter().enumerate() {
            let ready = self.device.admit(block)?.landed().await?;
            self.device.launch(ready, |launches, model| {
                let (state, input, time, output) = slots(at, flight.held_mut());
                let input: Hidden<R> = alias(input, ..)?;
                let time: kernel::EmbeddedTime = alias(time, ..)?;
                let mut output: Hidden<R> = alias(output, ..)?;
                let accepted = launches.forward_into_tracked(
                    kernel::Block { model },
                    kernel::block,
                    &state.request.inputs,
                    &mut (),
                    (input, &time),
                    &mut output,
                )?;
                flight.submitted(accepted);
                Ok(())
            })?;
        }
        Ok(())
    }

    pub(super) async fn submit_heads<W>(
        &self,
        flight: &mut Flight<W>,
        slots: impl for<'a> FnOnce(&'a W) -> HeadSlots<'a, State<R, K, T, A, V>>,
    ) -> io::Result<()> {
        let ready = self.device.admit(&self.heads)?.landed().await?;
        self.device.launch(ready, |launches, model| {
            let (state, input, time, (video, audio)) = slots(flight.held_mut());
            let input: Hidden<R> = alias(input, ..)?;
            let time: kernel::EmbeddedTime = alias(time, ..)?;
            let mut output: (Video<V>, Audio<A>) = (alias(video, ..)?, alias(audio, ..)?);
            let accepted = launches.forward_into_tracked(
                kernel::Heads { model },
                kernel::heads,
                &state.request.inputs,
                &mut (),
                (&input, &time),
                &mut output,
            )?;
            flight.accepted(output.0.into_buffer(), Arc::clone(&accepted));
            flight.accepted(output.1.into_buffer(), accepted);
            Ok(())
        })
    }

    /// TeaCache probes the first block without exposing the runner's resident stage bindings.
    pub(super) async fn with_first_block<O>(
        &self,
        call: impl for<'m, 'w> FnOnce(&mut Launches<'_>, &'m denoiser::Block<&'w Hbm>) -> io::Result<O>,
    ) -> io::Result<O> {
        let ready = self.device.admit(&self.blocks[0])?.landed().await?;
        self.device.launch(ready, call)
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
