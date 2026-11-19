//! Request-owned TeaCache rows beside the original separately launched denoiser stages.

use super::{flight::Flight, kernel as stage, request, step};
use furiosa_opt_models::models::diffusion::minimax_h3::embeddings::{Chips, Width};
use furiosa_opt_std::prelude::*;
use std::{io, marker::PhantomData, sync::Arc};

use super::step::Runner;

mod forward;
mod kernel;
mod rows;

type Hidden<R> = HbmTensor<bf16, Chips, m![R, Width]>;

/// Completed probe and residual of one request. A failed or cancelled call must not publish
/// these rows again. A flight retains their owner until every accepted reader has ended.
pub(super) struct Retained<R> {
    probe: HbmBuffer,
    residual: HbmBuffer,
    shape: PhantomData<R>,
}

impl<R: AxisName> Retained<R> {
    fn allocate(device: &mut Device) -> io::Result<Self> {
        let row = || {
            Hidden::<R>::alloc(device)
                .map(HbmTensor::into_buffer)
                .map_err(io::Error::other)
        };
        Ok(Self {
            probe: row()?,
            residual: row()?,
            shape: PhantomData,
        })
    }
}

/// Original per-chip FP32 totals. The framework applies its BF16 means and cache policy.
pub(super) struct Statistics {
    pub(super) distance: [f32; 4],
    pub(super) norm: [f32; 4],
}

impl Statistics {
    /// Preserve the repro's FP64 host sum, semantic element count, FP32 conversion and final
    /// BF16 mean. The framework performs the ratio and rescaling on these BF16-valued scalars.
    pub(super) fn means(self, elements: usize) -> (f32, f32) {
        let mean = |totals: [f32; 4]| {
            let value = totals.into_iter().map(f64::from).sum::<f64>() / elements as f64;
            bf16::from_f32(value as f32).to_f32()
        };
        (mean(self.distance), mean(self.norm))
    }
}

/// Prime the cache-specific entries alongside the runner's common denoising entries.
pub(super) fn prime<
    R: AxisName + Send + Sync,
    K: AxisName + Send + Sync,
    T: AxisName + Send + Sync,
    A: AxisName + Send + Sync,
    V: AxisName + Send + Sync,
>(
    device: &mut Device,
) -> io::Result<()> {
    Loadable::<(
        &mut Device,
        &kernel::Modulation<'_, '_>,
        (),
        &mut kernel::Modulations,
        &stage::EmbeddedTime,
    )>::load(kernel::modulate, device)
    .map_err(io::Error::other)?;
    Loadable::<(
        &mut Device,
        &kernel::Norm<'_, '_>,
        &request::Inputs<R, K, T, A, V>,
        &mut Hidden<R>,
        (&Hidden<R>, &kernel::NormTables),
    )>::load(kernel::norm, device)
    .map_err(io::Error::other)?;
    Loadable::<(
        &mut Device,
        &kernel::L1,
        &request::Inputs<R, K, T, A, V>,
        &mut (),
        (&Hidden<R>, &Hidden<R>),
    )>::load(kernel::l1, device)
    .map_err(io::Error::other)?;
    Loadable::<(
        &mut Device,
        &rows::Add,
        &request::Inputs<R, K, T, A, V>,
        &mut (),
        (&Hidden<R>, &Hidden<R>),
    )>::load(rows::add, device)
    .map_err(io::Error::other)?;
    Loadable::<(
        &mut Device,
        &rows::Subtract,
        &request::Inputs<R, K, T, A, V>,
        &mut (),
        (&Hidden<R>, &Hidden<R>),
    )>::load(rows::subtract, device)
    .map_err(io::Error::other)?;
    Loadable::<(
        &mut Device,
        &rows::Copy,
        &request::Inputs<R, K, T, A, V>,
        &mut Hidden<R>,
        &Hidden<R>,
    )>::load(rows::copy, device)
    .map_err(io::Error::other)?;
    Ok(())
}

impl<
    R: AxisName + Send + Sync,
    K: AxisName + Send + Sync,
    T: AxisName + Send + Sync,
    A: AxisName + Send + Sync,
    V: AxisName + Send + Sync,
> Runner<R, K, T, A, V>
{
    /// Measure the same six hidden-row allocations as a first cached call: four transient
    /// rows and two retained rows. No sizing buffer is read by a numerical launch.
    pub(super) fn cached_workspace_bytes(
        &self,
        state: &Arc<step::State<R, K, T, A, V>>,
    ) -> io::Result<usize> {
        self.device.with_device(|device| {
            super::memory::measure(device, |device| {
                let cache = Arc::new(Retained::allocate(device)?);
                let times = step::Times::alloc(device).map_err(io::Error::other)?;
                let video = step::Video::<V>::alloc(device).map_err(io::Error::other)?;
                let audio = step::Audio::<A>::alloc(device).map_err(io::Error::other)?;
                Workspace::allocate(device, Arc::clone(state), cache, times, video, audio)
            })
        })
    }
}

/// Preserve the embedding separately while two slots carry the block chain. Reusing that
/// embedding would destroy the original residual's right operand. Probe rows have their own
/// slot; the previous probe and residual remain in `cache` until their readers finish.
struct Workspace<R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName> {
    state: Arc<step::State<R, K, T, A, V>>,
    cache: Arc<Retained<R>>,
    times: step::Times,
    video: step::Video<V>,
    audio: step::Audio<A>,
    embedded: HbmBuffer,
    hidden: [HbmBuffer; 2],
    probe: HbmBuffer,
    time: HbmBuffer,
    modulation: kernel::Modulations,
    totals: kernel::Totals,
    velocities: (HbmBuffer, HbmBuffer),
}

impl<R: AxisName, K: AxisName, T: AxisName, A: AxisName, V: AxisName> Workspace<R, K, T, A, V> {
    fn allocate(
        device: &mut Device,
        state: Arc<step::State<R, K, T, A, V>>,
        cache: Arc<Retained<R>>,
        times: step::Times,
        video: step::Video<V>,
        audio: step::Audio<A>,
    ) -> io::Result<Self> {
        let row = || {
            Hidden::<R>::alloc(device)
                .map(HbmTensor::into_buffer)
                .map_err(io::Error::other)
        };
        let embedded = row()?;
        let hidden = [row()?, row()?];
        let probe = row()?;
        let table = || {
            furiosa_opt_models::ops::index::packed::Table::<Width, Chips>::alloc(device)
                .map_err(io::Error::other)
        };
        let modulation = (
            ((table()?, table()?), table()?),
            ((table()?, table()?), table()?),
        );
        Ok(Self {
            state,
            cache,
            times,
            video,
            audio,
            embedded,
            hidden,
            probe,
            time: stage::EmbeddedTime::alloc(device)
                .map_err(io::Error::other)?
                .into_buffer(),
            modulation,
            totals: (
                furiosa_opt_models::ops::pointwise::sum::Total::<Chips>::alloc(device)
                    .map_err(io::Error::other)?,
                furiosa_opt_models::ops::pointwise::sum::Total::<Chips>::alloc(device)
                    .map_err(io::Error::other)?,
            ),
            velocities: (
                step::Video::<V>::alloc(device)
                    .map_err(io::Error::other)?
                    .into_buffer(),
                step::Audio::<A>::alloc(device)
                    .map_err(io::Error::other)?
                    .into_buffer(),
            ),
        })
    }
}
