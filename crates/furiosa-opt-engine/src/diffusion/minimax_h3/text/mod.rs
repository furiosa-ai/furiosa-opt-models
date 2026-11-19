//! Four-chip text encoding through independently resident layers of one retained weight tree.

use super::flight::Flight;
use crate::residency::{Cells, Resident, SharedDevice, block_on};
use furiosa_opt_models::models::decoder::attention::Head;
use furiosa_opt_models::models::diffusion::minimax_h3::{packing::DEFAULT_MAX_TEXT, text};
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_models::tensor::parallel::{Parallel, Parallelism};
use furiosa_opt_std::prelude::*;
use std::{io, sync::Arc};

use budget::Fit;
pub(crate) use budget::Limits;
use text::stage::{Cache, Hidden, Windows};
pub(crate) use weights::Weights;

mod budget;
mod kernel;
mod request;
mod weights;

type P = Parallel<4>;
type Chips = <P as Parallelism>::Chip;

/// The embedding, layers and position views retain their shared parameter cells. Construction
/// measures actual allocation costs before selecting the prefix; no payload estimate decides fit.
pub(crate) struct Encoder {
    embedding: Resident<text::Embedding>,
    positions: Resident<text::position::Positions>,
    layers: [Resident<text::Layer>; text::LAYERS],
    device: SharedDevice,
    capacity: usize,
    fit: Fit,
}

impl Encoder {
    /// Retain and pin prepared sources once. The pipeline loads its resident weights and primes
    /// its other compiled entries first; `limits.beside` reserves their transient request storage.
    /// Encoder entries are primed before measuring free memory, so their persistent allocations
    /// cannot be mistaken for space available to resident layers.
    pub(crate) async fn from_prepared(
        weights: Weights,
        device: SharedDevice,
        limits: Limits,
    ) -> io::Result<Self> {
        let Weights {
            body,
            positions,
            capacity,
        } = weights;
        let owner = Resident::<(text::Body, text::position::Positions)>::new((body, positions));
        let embedding = owner.part::<text::Embedding>("embedding", |tree| &tree.0.embedding.0);
        let positions = owner.part::<text::position::Positions>("positions", |tree| &tree.1);
        let layers: [_; text::LAYERS] = std::array::from_fn(|at| {
            owner.part::<text::Layer>(&format!("layers.{at}"), |tree| &tree.0.body.layers[at])
        });
        owner.pin()?;
        device.with_device(|device| {
            kernel::embedding.load(device).map_err(io::Error::other)?;
            kernel::layer.load(device).map_err(io::Error::other)?;
            Ok(())
        })?;
        let available = || device.with_device(|device| Ok(device.memory().usage().available));
        let start = available()?;
        device.upload(&embedding)?;
        device.upload(&positions)?;
        // Each admission is awaited outside the shared device context lock.
        device.admit(&embedding)?.landed().await?;
        device.admit(&positions)?.landed().await?;
        let embedded = available()?;
        device.upload(&layers[0])?;
        device.admit(&layers[0])?.landed().await?;
        let layered = available()?;
        // Capacity admission reserves the same request layout and indirect-address storage.
        // No fabricated prompt or numerical layer execution is needed.
        let request_bytes = device.with_device(|device| {
            request::Layout::new(capacity, capacity)?.workspace_bytes(device)
        })?;
        let fixed = start
            .checked_sub(embedded)
            .ok_or_else(|| io::Error::other("encoder embedding allocation accounting changed"))?;
        let layer = embedded
            .checked_sub(layered)
            .ok_or_else(|| io::Error::other("encoder layer allocation accounting changed"))?;
        let fit = Fit::measured(limits, start, fixed, layer, request_bytes, layers.len())?;
        for layer in &layers[..fit.resident] {
            device.upload(layer)?;
            device.admit(layer)?.landed().await?;
        }
        if fit.resident == 0 {
            layers[0].to_host()?;
        }
        Ok(Self {
            embedding,
            positions,
            layers,
            device,
            capacity,
            fit,
        })
    }

    /// Read exactly the actual formatted prompt, through layer 50 and before the final norm.
    /// The single cache is reused only after all of the preceding layer's readers have ended.
    pub(crate) async fn encode(&self, tokens: &[i32]) -> io::Result<Vec<HbmBuffer>> {
        let prepared = request::Prepared::new(tokens, self.capacity)?;
        let request = Arc::new(self.device.with_device(|device| {
            // This initial transfer has no kernel readers yet. No dependent kernel or weight
            // completion is waited under this lock.
            block_on(prepared.upload(device))
        })?);
        let mut window = Window::new(&self.layers[self.fit.resident..]);
        if self.fit.resident == 0 {
            self.device.upload(&self.layers[0])?;
        }
        let ready = self.device.admit(&self.embedding)?.landed().await?;
        let mut flight = Flight::new(Arc::clone(&request), request.rows.len());
        self.device.launch(ready, |launches, embedding| {
            for row in &request.rows {
                let (output, work) = launches.forward_tracked(
                    embedding,
                    kernel::embedding,
                    (),
                    &mut (),
                    &row.token,
                )?;
                flight.accepted(output.into_buffer(), work);
            }
            Ok(())
        })?;
        let mut rows = flight.finish().await?;
        for (at, layer) in self.layers.iter().enumerate() {
            // At most the preceding/current/next streamed layer is resident. Prefetch precedes
            // waiting for the current layer's own upload and the shared input's readiness.
            if at + 1 >= self.fit.resident && at + 1 < text::LAYERS {
                self.device.upload(&self.layers[at + 1])?;
            }
            let pair = layer.pair(&self.positions);
            let ready = self.device.admit(&pair)?.landed().await?;
            for input in &rows {
                let input: Hidden = alias(input, ..)?;
                input.ready().await.map_err(io::Error::other)?;
            }
            // Retain original input owners and all metadata/cache owners through every accepted
            // launch, including an error after a partially submitted layer and cancellation.
            let cache = request.cache()?;
            let mut flight = Flight::new((rows, Arc::clone(&request), cache), request.rows.len());
            self.device.launch(ready, |launches, (layer, positions)| {
                for (at, row) in request.rows.iter().enumerate() {
                    let (rows, _, cache) = flight.held_mut();
                    let input: Hidden = alias(&rows[at], ..)?;
                    let (output, work) = launches.forward_tracked(
                        kernel::Layer { layer, positions },
                        kernel::layer,
                        row,
                        cache,
                        input,
                    )?;
                    // Store the output and completion before the next fallible submission.
                    flight.accepted(output.into_buffer(), work);
                }
                Ok(())
            })?;
            if at > self.fit.resident {
                self.layers[at - 1].to_host()?;
            }
            rows = flight.finish().await?;
        }
        if self.fit.resident < text::LAYERS {
            self.layers[text::LAYERS - 1].to_host()?;
        }
        window.finished = true;
        Ok(rows)
    }
}

/// On cancellation or partial failure, retire all possibly prefetched copies. This guard is
/// declared before any Flight, so its blocking drain follows request-reader cleanup on unwind.
struct Window<'a> {
    layers: &'a [Resident<text::Layer>],
    finished: bool,
}

impl<'a> Window<'a> {
    fn new(layers: &'a [Resident<text::Layer>]) -> Self {
        Self {
            layers,
            finished: false,
        }
    }
}

impl Drop for Window<'_> {
    fn drop(&mut self) {
        if !self.finished {
            for layer in self.layers {
                let _ = layer.to_host();
            }
        }
    }
}
