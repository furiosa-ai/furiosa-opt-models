//! The original cache launch sequence, with accepted readers retained through completion.

use super::{Flight, Hidden, Retained, Runner, Statistics, Workspace, kernel, rows, stage, step};
use crate::Function;
use crate::residency::block_on;
use furiosa_opt_models::models::diffusion::minimax_h3::{denoiser::DEPTH, embeddings::Chips};
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_std::prelude::*;
use std::{io, sync::Arc};

use super::step::{Audio, Times, Video};

impl<
    R: AxisName + Send + Sync,
    K: AxisName + Send + Sync,
    T: AxisName + Send + Sync,
    A: AxisName + Send + Sync,
    V: AxisName + Send + Sync,
> Runner<R, K, T, A, V>
{
    /// Invoke the host decision exactly once after the probe's readers finish. A first call
    /// always computes. The caller removes `previous` from its memo before this call: only a
    /// successful return publishes retained rows again, after every accepted reader has ended.
    pub(in crate::diffusion::minimax_h3) async fn forward<
        F: FnMut(Option<Statistics>) -> io::Result<bool> + Send,
    >(
        &self,
        state: Arc<step::State<R, K, T, A, V>>,
        previous: Option<Arc<Retained<R>>>,
        times: Times,
        video: Video<V>,
        audio: Audio<A>,
        mut decide: F,
    ) -> io::Result<((Video<V>, Audio<A>), Arc<Retained<R>>)> {
        times.ready().await.map_err(io::Error::other)?;
        video.ready().await.map_err(io::Error::other)?;
        audio.ready().await.map_err(io::Error::other)?;
        state.text.rows.ready().await.map_err(io::Error::other)?;

        let has_previous = previous.is_some();
        let cache = match previous {
            Some(cache) => cache,
            None => Arc::new(self.device.with_device(Retained::<R>::allocate)?),
        };
        let workspace = self.device.with_device(|device| {
            Workspace::allocate(device, state, Arc::clone(&cache), times, video, audio)
        })?;
        // One embedding, modulation, norm, optional L1, DEPTH blocks, residual, probe copy
        // and two head-output records. The flight owns the cache even if this future is dropped.
        let mut flight = Flight::new(workspace, DEPTH + 10);

        let ready = self.device.admit(&self.embedding)?.landed().await?;
        self.device.launch(ready, |launches, model| {
            let work = flight.held_mut();
            let mut output: (Hidden<R>, stage::EmbeddedTime) =
                (alias(&work.embedded, ..)?, alias(&work.time, ..)?);
            let accepted = launches.forward_into_tracked(
                stage::Embed { model },
                stage::embed,
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

        let ready = self.device.admit(&self.blocks[0])?.landed().await?;
        let probe_done = self.device.launch(ready, |launches, first| {
            let work = flight.held_mut();
            let time: stage::EmbeddedTime = alias(&work.time, ..)?;
            let accepted = launches.forward_into_tracked(
                kernel::Modulation {
                    model: &first.0.modulation.0,
                },
                kernel::modulate,
                (),
                &mut work.modulation,
                &time,
                &mut (),
            )?;
            flight.submitted(accepted);

            let work = flight.held_mut();
            let embedded: Hidden<R> = alias(&work.embedded, ..)?;
            let mut probe: Hidden<R> = alias(&work.probe, ..)?;
            let accepted = launches.forward_into_tracked(
                kernel::Norm {
                    model: &first.0.norm.0,
                },
                kernel::norm,
                &work.state.request.inputs,
                &mut probe,
                (&embedded, &work.modulation.0.0),
                &mut (),
            )?;
            flight.submitted(Arc::clone(&accepted));
            Ok(accepted)
        })?;

        let statistics = if has_previous {
            // These stateless modules have no weight tree. Function submits on the already
            // bound device, and Flight records their completions before the lock is released.
            let totals_done = self.device.with_device(|device| {
                let work = flight.held_mut();
                let probe: Hidden<R> = alias(&work.probe, ..)?;
                let previous: Hidden<R> = alias(&work.cache.probe, ..)?;
                let accepted =
                    Arc::new(Function::new(kernel::L1, kernel::l1).forward_into_tracked(
                        device,
                        &work.state.request.inputs,
                        &mut (),
                        (&probe, &previous),
                        &mut work.totals,
                    )?);
                flight.submitted(Arc::clone(&accepted));
                Ok(accepted)
            })?;
            // End the device reader before either downloading totals or overwriting the
            // previous probe. The flight keeps the totals and both row owners throughout DMA.
            totals_done.ended().await.map_err(io::Error::other)?;
            Some(self.device.with_device(|device| {
                let work = flight.held_mut();
                let distance = block_on(
                    work.totals
                        .0
                        .to_host::<m![{ Chips }, 1 # 8]>(&mut device.pdma)
                        .into_future(),
                )
                .map_err(io::Error::other)?
                .into_vec();
                let norm = block_on(
                    work.totals
                        .1
                        .to_host::<m![{ Chips }, 1 # 8]>(&mut device.pdma)
                        .into_future(),
                )
                .map_err(io::Error::other)?
                .into_vec();
                Ok(Statistics {
                    distance: std::array::from_fn(|chip| distance[chip * 8]),
                    norm: std::array::from_fn(|chip| norm[chip * 8]),
                })
            })?)
        } else {
            // Fresh retained storage has no readable probe or residual yet.
            probe_done.ended().await.map_err(io::Error::other)?;
            None
        };
        let requested_compute = decide(statistics)?;
        let compute = !has_previous || requested_compute;

        let final_slot = if compute {
            for (at, block) in self.blocks.iter().enumerate() {
                let ready = self.device.admit(block)?.landed().await?;
                self.device.launch(ready, |launches, model| {
                    let work = flight.held_mut();
                    let input: Hidden<R> = alias(
                        if at == 0 {
                            &work.embedded
                        } else {
                            &work.hidden[(at - 1) % 2]
                        },
                        ..,
                    )?;
                    let time: stage::EmbeddedTime = alias(&work.time, ..)?;
                    let mut output: Hidden<R> = alias(&work.hidden[at % 2], ..)?;
                    let accepted = launches.forward_into_tracked(
                        stage::Block { model },
                        stage::block,
                        &work.state.request.inputs,
                        &mut (),
                        (input, &time),
                        &mut output,
                    )?;
                    flight.submitted(accepted);
                    Ok(())
                })?;
            }
            self.device.with_device(|device| {
                let work = flight.held_mut();
                let result: Hidden<R> = alias(&work.hidden[(DEPTH - 1) % 2], ..)?;
                let embedded: Hidden<R> = alias(&work.embedded, ..)?;
                let mut residual: Hidden<R> = alias(&work.cache.residual, ..)?;
                let accepted = Arc::new(
                    Function::new(rows::Subtract, rows::subtract).forward_into_tracked(
                        device,
                        &work.state.request.inputs,
                        &mut (),
                        (&result, &embedded),
                        &mut residual,
                    )?,
                );
                flight.submitted(accepted);
                Ok(())
            })?;
            (DEPTH - 1) % 2
        } else {
            // This branch is reachable only with a completed previous residual.
            self.device.with_device(|device| {
                let work = flight.held_mut();
                let embedded: Hidden<R> = alias(&work.embedded, ..)?;
                let residual: Hidden<R> = alias(&work.cache.residual, ..)?;
                let mut output: Hidden<R> = alias(&work.hidden[0], ..)?;
                let accepted = Arc::new(Function::new(rows::Add, rows::add).forward_into_tracked(
                    device,
                    &work.state.request.inputs,
                    &mut (),
                    (&embedded, &residual),
                    &mut output,
                )?);
                flight.submitted(accepted);
                Ok(())
            })?;
            0
        };

        self.device.with_device(|device| {
            let work = flight.held_mut();
            let probe: Hidden<R> = alias(&work.probe, ..)?;
            let mut retained: Hidden<R> = alias(&work.cache.probe, ..)?;
            let accepted = Arc::new(Function::new(rows::Copy, rows::copy).forward_into_tracked(
                device,
                &work.state.request.inputs,
                &mut retained,
                &probe,
                &mut (),
            )?);
            flight.submitted(accepted);
            Ok(())
        })?;

        let ready = self.device.admit(&self.heads)?.landed().await?;
        self.device.launch(ready, |launches, model| {
            let work = flight.held_mut();
            let input: Hidden<R> = alias(&work.hidden[final_slot], ..)?;
            let time: stage::EmbeddedTime = alias(&work.time, ..)?;
            let mut output: (Video<V>, Audio<A>) = (
                alias(&work.velocities.0, ..)?,
                alias(&work.velocities.1, ..)?,
            );
            let accepted = launches.forward_into_tracked(
                stage::Heads { model },
                stage::heads,
                &work.state.request.inputs,
                &mut (),
                (&input, &time),
                &mut output,
            )?;
            flight.accepted(output.0.into_buffer(), Arc::clone(&accepted));
            flight.accepted(output.1.into_buffer(), accepted);
            Ok(())
        })?;
        let [video, audio]: [HbmBuffer; 2] = flight.finish().await?.try_into().map_err(|_| {
            io::Error::other("H3 cached heads returned an invalid media output count")
        })?;
        Ok((
            (
                Video::try_from(video).map_err(io::Error::other)?,
                Audio::try_from(audio).map_err(io::Error::other)?,
            ),
            cache,
        ))
    }
}
