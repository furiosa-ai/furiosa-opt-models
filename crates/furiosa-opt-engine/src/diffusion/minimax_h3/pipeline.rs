//! One retained pipeline owner with exclusive, separately owned request state.

use super::{
    audio, cache, condition, condition::Conditioner, denoiser, memory, request, text, video,
    weights,
};
use crate::residency::{SharedDevice, block_on};
use furiosa_opt_models::models::diffusion::minimax_h3::embeddings::{Chips, Condition, Width};
use furiosa_opt_models::models::diffusion::minimax_h3::packing::{Capacity, PARTS, Plan};
use furiosa_opt_models::models::diffusion::minimax_h3::request::BlockPlan;
use furiosa_opt_std::prelude::*;
use std::{io, sync::Arc};

/// Resident denoiser and decoder weights, the streamed encoder and their shared device context.
/// Compiled axes bound storage; each request still supplies its actual valid text count.
/// Public component views must retain this owner, including when the caller drops its pipeline.
pub(super) struct Pipeline<const N: usize, R, K, T: AxisName, A, V> {
    weights: Arc<weights::Weights>,
    device: SharedDevice,
    encoder: text::Encoder,
    conditioner: Conditioner<N, T>,
    denoiser: denoiser::Denoiser<R, K, T, A, V>,
    video: video::Decoder,
    audio: audio::Decoder,
    active: Arc<tokio::sync::Mutex<()>>,
}

impl<
    const N: usize,
    R: AxisName + Send + Sync,
    K: AxisName + Send + Sync,
    T: AxisName + Send + Sync,
    A: AxisName + Send + Sync,
    V: AxisName + Send + Sync,
> Pipeline<N, R, K, T, A, V>
{
    /// Assemble prepared weights on the shared device. Prime every component's entries before
    /// selecting the encoder prefix. Measure actual call allocations and reserve the maximum
    /// completed prompt memo while sizing the encoder.
    /// `limits.beside` can additionally reserve a caller's larger external stage.
    /// Model loading and framework option parsing remain separate checked boundaries.
    pub(super) async fn from_prepared(
        prepared: weights::Prepared,
        device: SharedDevice,
        limits: text::Limits,
    ) -> io::Result<Arc<Self>> {
        if prepared.text.capacity() != N {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "prepared encoder capacity differs from the compiled pipeline",
            ));
        }
        let capacity = Capacity::new(N)?;
        // Refuse incompatible allocations before placing any weight. The maximum text plan
        // also checks the largest source spans; shorter requests derive their own work later.
        let maximum = capacity.plan(N)?;
        BlockPlan::<R, K>::new(maximum)?;
        maximum.row_plan::<R, T, A, V>()?;
        let weights::Prepared { weights, text } = prepared;
        device.upload(weights.resident())?;
        device.admit(weights.resident())?.landed().await?;
        let conditioner = Conditioner::from_resident(weights.denoiser(), device.clone()).await?;
        let denoiser =
            denoiser::Denoiser::from_resident(weights.denoiser(), device.clone()).await?;
        let video = video::Decoder::from_resident(weights.video(), device.clone()).await?;
        let audio = audio::Decoder::from_resident(weights.audio(), device.clone()).await?;
        let reserved = device.with_device(|device| {
            let prepared = request::Prepared::<R, K, T, A, V>::new(
                maximum,
                &vec![[0.0; 3]; maximum.key_count()],
                weights.inputs().transformer().rope_theta(),
            )?;
            let request = block_on(request::Request::upload(prepared, device))?;
            let rows = block_on(
                HostTensor::<bf16, m![{ Chips }, T, Width]>::zero()
                    .to_hbm(&mut device.pdma)
                    .into_future(),
            )
            .map_err(io::Error::other)?;
            denoiser::State::new(
                request,
                condition::Text {
                    plan: maximum,
                    rows,
                },
            )
            .map(Arc::new)
        })?;
        let memory = memory::Budget {
            condition: conditioner.workspace_bytes()?,
            denoising: denoiser
                .workspace_bytes(&reserved)?
                .max(denoiser.workspace_bytes_cached(&reserved)?),
            video: video.workspace_bytes()?,
            audio: audio.workspace_bytes()?,
        };
        let limits = text::Limits {
            beside: limits.beside.max(memory.call()),
            ..limits
        };
        // This real allocation set subtracts the memo's cost while the encoder chooses its
        // prefix. It is retired before returning; later framework calls retain the real memo.
        let encoder = text::Encoder::from_prepared(text, device.clone(), limits).await?;
        drop(reserved);
        Ok(Arc::new(Self {
            weights,
            device,
            encoder,
            conditioner,
            denoiser,
            video,
            audio,
            active: Arc::new(tokio::sync::Mutex::new(())),
        }))
    }

    pub(super) fn device(&self) -> &SharedDevice {
        &self.device
    }

    /// The framework's text component returns completed BF16 rows on the host, before the
    /// encoder's final norm. Keep every row owner and pipeline admission through the one batched
    /// download. Each row has four chip replicas; download only the first replica.
    pub(super) async fn encode(&self, tokens: &[i32]) -> io::Result<Vec<bf16>> {
        let _lease = self.active.lock().await;
        let rows = self.encoder.encode(tokens).await?;
        let bytes_per_row = Condition::SIZE * std::mem::size_of::<bf16>();
        let mut bytes = vec![0_u8; rows.len() * bytes_per_row];
        self.device.with_device(|device| {
            let sources = rows
                .iter()
                .map(|row| row.slice(0..1, ..).map_err(io::Error::other))
                .collect::<io::Result<Vec<_>>>()?;
            let copies = sources
                .into_iter()
                .zip(bytes.chunks_exact_mut(bytes_per_row));
            block_on(device.pdma.to_host(copies).into_future()).map_err(io::Error::other)
        })?;
        Ok(bytes
            .chunks_exact(2)
            .map(|word| bf16::from_bits(u16::from_le_bytes([word[0], word[1]])))
            .collect())
    }

    /// The framework boundary supplies one checked plan and contiguous BF16 prompt rows.
    /// Preserve that boundary without running the encoder again. One replicated upload supplies
    /// row views to the same conditioner used by token-ID preparation.
    pub(super) async fn prepare_encoded(
        self: &Arc<Self>,
        plan: Plan,
        positions: &[[f64; 3]],
        encoded: &[bf16],
    ) -> io::Result<Request<N, R, K, T, A, V>> {
        let lease = Arc::clone(&self.active).lock_owned().await;
        let prepared = request::Prepared::<R, K, T, A, V>::new(
            plan,
            positions,
            self.weights.inputs().transformer().rope_theta(),
        )?;
        let encoded = self.device.with_device(|device| {
            let source = HostBuffer::from(bf16::to_buf(encoded).repeat(PARTS));
            let buffer = block_on(source.to_hbm(&mut device.pdma).into_future())
                .map_err(io::Error::other)?;
            let bytes = Condition::SIZE * std::mem::size_of::<bf16>();
            (0..plan.text_tokens())
                .map(|row| {
                    buffer
                        .alias(row * bytes..(row + 1) * bytes)
                        .map_err(io::Error::other)
                })
                .collect::<io::Result<Vec<_>>>()
        })?;
        self.finish_preparation(lease, prepared, encoded).await
    }

    async fn finish_preparation(
        self: &Arc<Self>,
        lease: tokio::sync::OwnedMutexGuard<()>,
        prepared: request::Prepared<R, K, T, A, V>,
        encoded: Vec<HbmBuffer>,
    ) -> io::Result<Request<N, R, K, T, A, V>> {
        let plan = prepared.plan();
        // Encoder caches and streamed copies have retired. Only its completed output rows
        // coexist with conditioning metadata; joint transports never burden the encoder probe.
        let request = self
            .device
            .with_device(|device| block_on(request::Request::upload(prepared, device)))?;
        let text = self.conditioner.prepare(plan, encoded).await?;
        let state = Arc::new(denoiser::State::new(request, text)?);
        Ok(Request {
            owner: Arc::clone(self),
            state,
            _lease: lease,
        })
    }

    /// A standalone framework decoder call shares admission with encoding and denoising.
    pub(super) async fn decode_video(&self, input: video::Input<'_>) -> io::Result<Vec<f32>> {
        let _lease = self.active.lock().await;
        self.video.decode(input).await
    }

    pub(super) async fn decode_audio(&self, input: audio::Latents<'_>) -> io::Result<Vec<f32>> {
        let _lease = self.active.lock().await;
        self.audio.decode(input).await
    }
}

/// One request's text and immutable transport, with exclusive pipeline admission. Taking `&mut`
/// for a step prevents concurrent solver steps from using the same request through this API.
pub(super) struct Request<const N: usize, R: AxisName, K: AxisName, T: AxisName, A, V> {
    owner: Arc<Pipeline<N, R, K, T, A, V>>,
    state: Arc<denoiser::State<R, K, T, A, V>>,
    _lease: tokio::sync::OwnedMutexGuard<()>,
}

/// Completed preparation retained by the framework's prompt/position memo between calls.
/// It keeps the whole pipeline alive, but holds no admission while another component runs.
/// The serving adapter retains only its current memo and retires it before preparing another.
pub(super) struct Retained<const N: usize, R: AxisName, K: AxisName, T: AxisName, A, V> {
    owner: Arc<Pipeline<N, R, K, T, A, V>>,
    state: Arc<denoiser::State<R, K, T, A, V>>,
}

impl<
    const N: usize,
    R: AxisName + Send + Sync,
    K: AxisName + Send + Sync,
    T: AxisName + Send + Sync,
    A: AxisName + Send + Sync,
    V: AxisName + Send + Sync,
> Retained<N, R, K, T, A, V>
{
    /// Reacquire exclusive admission for this same owner's already completed preparation.
    pub(super) async fn resume(&self) -> Request<N, R, K, T, A, V> {
        let lease = Arc::clone(&self.owner.active).lock_owned().await;
        Request {
            owner: Arc::clone(&self.owner),
            state: Arc::clone(&self.state),
            _lease: lease,
        }
    }
}

impl<
    const N: usize,
    R: AxisName + Send + Sync,
    K: AxisName + Send + Sync,
    T: AxisName + Send + Sync,
    A: AxisName + Send + Sync,
    V: AxisName + Send + Sync,
> Request<N, R, K, T, A, V>
{
    /// Keep completed preparation across framework calls without blocking sibling components.
    /// All accepted readers from a preparation or step have ended before that call returns.
    pub(super) fn retain(self) -> Retained<N, R, K, T, A, V> {
        let Self {
            owner,
            state,
            _lease,
        } = self;
        drop(_lease);
        Retained { owner, state }
    }

    pub(super) async fn forward(
        &mut self,
        times: denoiser::Times,
        video: denoiser::Video<V>,
        audio: denoiser::Audio<A>,
    ) -> io::Result<(denoiser::Video<V>, denoiser::Audio<A>)> {
        self.owner
            .denoiser
            .forward(Arc::clone(&self.state), times, video, audio)
            .await
    }

    /// The cache owns only completed rows. The framework supplies its policy after the probe;
    /// caller ownership and admission remain identical to ordinary denoising.
    pub(super) async fn forward_cached<
        F: FnMut(Option<cache::Statistics>) -> io::Result<bool> + Send,
    >(
        &mut self,
        times: denoiser::Times,
        video: denoiser::Video<V>,
        audio: denoiser::Audio<A>,
        previous: Option<Arc<cache::Retained<R>>>,
        decide: F,
    ) -> io::Result<(
        (denoiser::Video<V>, denoiser::Audio<A>),
        Arc<cache::Retained<R>>,
    )> {
        self.owner
            .denoiser
            .forward_cached(
                Arc::clone(&self.state),
                previous,
                times,
                video,
                audio,
                decide,
            )
            .await
    }
}
