//! Framework component calls sharing one H3 pipeline, prompt memo and pinned media staging.
//!
//! Tokenization, schedules, solver steps and media packaging belong to the framework. This
//! adapter preserves the original component boundary and delegates every numerical call to the
//! existing core. Component views must retain an `Arc<Pipeline>` and use this same owner.

use super::{audio, cache, layout::Layout, pipeline, runner, text, weights};
use crate::residency::{SharedDevice, block_on};
use furiosa_opt_models::load::{Checkpoint, LoadError};
use furiosa_opt_models::models::Class;
use furiosa_opt_models::models::diffusion::minimax_h3::{
    Inputs,
    embeddings::{Chips, Condition, Latent, Patch, Width},
    packing::{Capacity, DEFAULT_MAX_TEXT, Modality},
    video::tiling,
};
use furiosa_opt_models::tensor::axis::Rows;
use furiosa_opt_std::prelude::*;
use std::{
    io,
    sync::{Arc, PoisonError},
    thread::{self, ThreadId},
};
use tokio::sync::Mutex;

/// Strictly loaded host weights, before placement on the framework's existing device.
pub struct Prepared(weights::Prepared);

impl Prepared {
    /// Load with already checked component documents, including an in-memory checkpoint.
    pub fn from_inputs(source: &Checkpoint, inputs: Inputs) -> Result<Self, LoadError> {
        weights::Prepared::load_with_inputs(source, DEFAULT_MAX_TEXT, inputs).map(Self)
    }

    /// Prime the original stages and measure their real allocations before encoder sizing.
    /// `device` is the original four-chip context; this does not open additional devices.
    pub async fn place(self, device: SharedDevice) -> io::Result<Arc<Pipeline>> {
        if device.placement().chips.len() != Chips::SIZE || device.placement().pes != (0..8) {
            return Err(invalid(
                "H3 requires four chips with all eight PEs per chip",
            ));
        }
        // Allocate staging before device placement so a page-lock refusal leaves no placed
        // core behind. It remains the same storage for every component call's lifetime.
        let state = State {
            memo: None,
            video: Staging::new::<VideoRows, Patch>()?,
            audio: Staging::new::<AudioRows, Latent>()?,
        };
        let core = Core::from_prepared(
            self.0,
            device,
            text::Limits {
                bytes: usize::MAX,
                beside: 0,
            },
        )
        .await?;
        Ok(Arc::new(Pipeline {
            core,
            state: Mutex::new(state),
            decision: Decision::default(),
        }))
    }
}

/// Original serialized framework component facade, retaining the complete four-chip core.
pub struct Pipeline {
    core: Arc<Core>,
    // Serialize every component while allowing concurrent async callers to wait without
    // blocking the executor. The DeviceContext lock is never held across core await points.
    state: Mutex<State>,
    decision: Decision,
}

// The binding drives a complete call outside the DeviceContext mutex. SharedDevice locks only
// submissions and ordinary completed DMA boundaries; completion waits release that lock.
impl Pipeline {
    pub const CHIPS: usize = Chips::SIZE;
    pub const MAX_TEXT_TOKENS: usize = DEFAULT_MAX_TEXT;

    pub const CLASS: Class = Class {
        family: "minimax_h3",
        name: "Pipeline",
    };

    /// The same shared context used by all components; a view must also retain this facade.
    pub fn device(&self) -> &SharedDevice {
        self.core.device()
    }

    /// Other threads queue normally; the synchronous policy must not wait on its own call.
    async fn state(&self) -> io::Result<tokio::sync::MutexGuard<'_, State>> {
        self.decision.check()?;
        Ok(self.state.lock().await)
    }

    /// Completed host BF16 rows from the first 50 text layers, before final normalization.
    /// Keep the previous prompt memo while encoding, as in the original streaming caller.
    pub async fn encode(&self, tokens: &[i32]) -> io::Result<Vec<bf16>> {
        let mut state = self.state().await?;
        if let Some(memo) = &mut state.memo {
            memo.cache = None;
        }
        self.core.encode(tokens).await
    }

    /// Release retained cache rows at a new generation boundary, keeping prepared prompt data.
    pub async fn reset_cache(&self) -> io::Result<()> {
        let mut state = self.state().await?;
        if let Some(memo) = &mut state.memo {
            memo.cache = None;
        }
        Ok(())
    }

    /// One uncached transformer evaluation on owned, framework-parsed host values.
    pub async fn forward(&self, input: Input) -> io::Result<Output> {
        self.forward_with::<fn(Option<(f32, f32)>) -> io::Result<bool>>(input, None)
            .await
    }

    /// Device cache work with the framework's host decision. Scalars are the original BF16
    /// means represented exactly as FP32; no GPU tensor or hidden allocation leaves the engine.
    /// A synchronous decision cannot reenter this pipeline on its own thread; such calls
    /// return an error instead of waiting for the decision's still-active evaluation.
    pub async fn forward_cached<F>(&self, input: Input, decide: F) -> io::Result<Output>
    where
        F: FnMut(Option<(f32, f32)>) -> io::Result<bool> + Send,
    {
        self.forward_with(input, Some(decide)).await
    }

    async fn forward_with<F>(&self, input: Input, mut decide: Option<F>) -> io::Result<Output>
    where
        F: FnMut(Option<(f32, f32)>) -> io::Result<bool> + Send,
    {
        let mut state = self.state().await?;
        let Checked {
            layout,
            input,
            positions,
            key,
        } = Checked::new(input)?;
        if state.memo.as_ref().is_none_or(|memo| memo.key != key) {
            // Release the prior retained region BEFORE preparing its replacement. Preparation
            // can fail or be cancelled with the memo empty; it never contains a partial entry.
            state.memo = None;
            let request = self
                .core
                .prepare_encoded(key.text, &positions, &input.prompt)
                .await?;
            state.memo = Some(Memo {
                key,
                request: request.retain(),
                cache: None,
            });
        }
        let mut request = state
            .memo
            .as_ref()
            .expect("completed memo above")
            .request
            .resume()
            .await;
        state
            .video
            .split::<VideoRows>(&layout, &input.video, Modality::Video, Patch::SIZE)?;
        state
            .audio
            .split::<AudioRows>(&layout, &input.audio, Modality::Audio, Latent::SIZE)?;
        let times = upload_times(self.device(), input.times)?;
        let video = state.video.upload::<VideoRows, Patch>(self.device())?;
        let audio = state.audio.upload::<AudioRows, Latent>(self.device())?;
        // Remove the published cache before any numerical cache launch. Failure/cancellation
        // leaves it absent; the flight drains accepted work before retiring its storage.
        let previous = state
            .memo
            .as_mut()
            .expect("completed memo above")
            .cache
            .take();
        let ((video, audio), next_cache) = if let Some(decide) = &mut decide {
            let elements = layout.sequence() * Width::SIZE;
            let (values, retained) = request
                .cached_step(times, video, audio, previous, |statistics| {
                    self.decision
                        .call(|| decide(statistics.map(|s| s.means(elements))))
                })
                .await?;
            (values, Some(retained))
        } else {
            drop(previous);
            (request.step(times, video, audio).await?, None)
        };
        state.video.download(self.device(), video)?;
        state.audio.download(self.device(), audio)?;
        let video = layout.gather::<VideoRows>(
            state.video.values()?,
            Modality::Video,
            Patch::SIZE,
            &input.video_mask,
        )?;
        let audio = layout.gather::<AudioRows>(
            state.audio.values()?,
            Modality::Audio,
            Latent::SIZE,
            &input.audio_mask,
        )?;
        let output = Output {
            video,
            video_shape: [layout.selected_video().len(), Patch::SIZE],
            audio,
            audio_shape: [layout.rows(Modality::Audio).len(), Latent::SIZE],
        };
        // Release this call's admission, leaving only a non-admitted Retained in the memo.
        drop(request);
        // Downloads and output reconstruction must succeed before publishing the next cache.
        state.memo.as_mut().expect("completed memo above").cache = next_cache;
        Ok(output)
    }

    /// Decode the fixed profile, retaining the last requested frames as in the repro.
    /// Shape and frame-count checks precede all device work.
    pub async fn decode_video(
        &self,
        shape: &[usize],
        values: &[f32],
        frame_num: Option<usize>,
        process_image: bool,
    ) -> io::Result<Decoded<5>> {
        let _state = self.state().await?;
        if shape != tiling::LATENT_SHAPE
            || values.len() != tiling::LATENT_SHAPE.into_iter().product::<usize>()
            || frame_num.is_some_and(|frames| !(1..=tiling::OUTPUT_SHAPE[2]).contains(&frames))
            || process_image
        {
            return Err(invalid(
                "decode_base requires float32 [1,24,37,48,84], frame_num in 1..=124 and process_image=False",
            ));
        }
        let frames = frame_num.unwrap_or(tiling::OUTPUT_SHAPE[2]);
        let mut shape = tiling::OUTPUT_SHAPE;
        shape[2] = frames;
        Ok(Decoded {
            values: self.core.decode_video_frames(values, frames).await?,
            shape,
        })
    }

    /// Decode channel-major stereo latents through the original audio component.
    pub async fn decode_audio(&self, shape: &[usize], values: &[f32]) -> io::Result<Decoded<3>> {
        let _state = self.state().await?;
        if shape != audio::INPUT_SHAPE
            || values.len() != audio::INPUT_SHAPE.into_iter().product::<usize>()
        {
            return Err(invalid(
                "audio decode requires contiguous float32 [2,32,207]",
            ));
        }
        Ok(Decoded {
            values: self.core.decode_audio(shape, values).await?,
            shape: audio::OUTPUT_SHAPE,
        })
    }
}

/// The policy runs synchronously while `Pipeline::state` is held. Keep its thread separately
/// so checking reentry never waits for that state. A caller on another thread still queues on
/// the ordinary async mutex. Clear the owner on success, error or unwind.
#[derive(Default)]
struct Decision {
    owner: std::sync::Mutex<Option<ThreadId>>,
}

impl Decision {
    fn check(&self) -> io::Result<()> {
        if *self.owner.lock().unwrap_or_else(PoisonError::into_inner)
            == Some(thread::current().id())
        {
            return Err(io::Error::other(
                "H3 component calls cannot reenter the same pipeline from its TeaCache decision callback",
            ));
        }
        Ok(())
    }

    fn call<T>(&self, decide: impl FnOnce() -> T) -> T {
        struct Clear<'a>(&'a std::sync::Mutex<Option<ThreadId>>);
        impl Drop for Clear<'_> {
            fn drop(&mut self) {
                *self.0.lock().unwrap_or_else(PoisonError::into_inner) = None;
            }
        }
        *self.owner.lock().unwrap_or_else(PoisonError::into_inner) = Some(thread::current().id());
        let _clear = Clear(&self.owner);
        decide()
    }
}

struct State {
    memo: Option<Memo>,
    video: Staging,
    audio: Staging,
}

struct Memo {
    key: Key,
    request: Retained,
    cache: Option<Arc<cache::Retained<JointRows>>>,
}

/// Prompt equality is BIT equality, including NaNs. Finite position equality preserves the
/// original FP64 comparison (including +0 == -0); padding positions participate in the key.
/// Canonical modality row lists are wholly determined by text count, so need no second copy.
#[derive(PartialEq)]
struct Key {
    prompt: Vec<u16>,
    positions: Vec<[f64; 3]>,
    text: usize,
}

/// Raw owned host input from the native binding. The binding additionally checks keyword
/// metadata and dtypes, derives media timesteps from tags/inverse_indices, and applies the
/// original skip-mask override before populating these effective boolean masks.
pub struct Input {
    pub text_rows: Vec<usize>,
    pub audio_rows: Vec<usize>,
    pub video_rows: Vec<usize>,
    pub selected_video: Vec<usize>,
    pub bounds: Vec<usize>,
    pub sequence: usize,
    pub prompt: Vec<bf16>,
    pub prompt_shape: [usize; 2],
    pub positions: Vec<[f64; 3]>,
    /// Video, then audio. These are the caller's effective FP32 media times.
    pub times: [f32; 2],
    /// Contiguous FP32 [sequence, Patch::SIZE], with the leading singleton batch removed.
    pub video: Vec<f32>,
    pub video_shape: [usize; 2],
    /// Contiguous FP32 [sequence, Latent::SIZE], with the leading singleton batch removed.
    pub audio: Vec<f32>,
    pub audio_shape: [usize; 2],
    pub video_mask: Vec<bool>,
    pub audio_mask: Vec<bool>,
}

/// Completed FP32 velocities in exactly the framework's selected order and explicit shapes.
pub struct Output {
    pub video: Vec<f32>,
    pub video_shape: [usize; 2],
    pub audio: Vec<f32>,
    pub audio_shape: [usize; 2],
}

/// Completed decoder values in the component's original architecture layout.
pub struct Decoded<const N: usize> {
    pub values: Vec<f32>,
    pub shape: [usize; N],
}

// One compiled profile; runtime valid counts and geometry still belong to the generic core.
// The core's checked constructor verifies these allocations against the maximum packing plan.
type JointRows = Rows<10496>;
type KeySpans = Rows<20>;
type TextRows = Rows<512>;
type AudioRows = Rows<128>;
type VideoRows = Rows<9344>;
type Core =
    pipeline::Pipeline<{ DEFAULT_MAX_TEXT }, JointRows, KeySpans, TextRows, AudioRows, VideoRows>;
type Retained =
    pipeline::Retained<{ DEFAULT_MAX_TEXT }, JointRows, KeySpans, TextRows, AudioRows, VideoRows>;

/// All host contracts are checked before preparation can upload even the prompt. The core
/// checks FP32 rotary phases with the checkpoint's theta before its first transfer as well.
struct Checked {
    layout: Layout,
    input: Input,
    positions: Vec<[f64; 3]>,
    key: Key,
}

impl Checked {
    fn new(mut input: Input) -> io::Result<Self> {
        let layout = Layout::parse(
            Capacity::default(),
            (
                std::mem::take(&mut input.text_rows),
                std::mem::take(&mut input.audio_rows),
                std::mem::take(&mut input.video_rows),
                std::mem::take(&mut input.selected_video),
            ),
            &input.bounds,
            input.sequence,
        )?;
        let text = layout.plan().text_tokens();
        if input.prompt_shape != [text, Condition::SIZE]
            || input.prompt.len() != text * Condition::SIZE
        {
            return Err(invalid(
                "prompt must contain one BF16 [5120] row per valid text token",
            ));
        }
        if input.times.iter().any(|time| !time.is_finite()) {
            return Err(invalid("video and audio timesteps must be finite FP32"));
        }
        let positions = layout.positions(&input.positions)?;
        for (name, shape, values, width) in [
            ("video", input.video_shape, &input.video, Patch::SIZE),
            ("audio", input.audio_shape, &input.audio, Latent::SIZE),
        ] {
            if shape != [layout.sequence(), width]
                || layout.sequence().checked_mul(width) != Some(values.len())
            {
                return Err(invalid(format!(
                    "{name} must be contiguous FP32 [{}, {width}]",
                    layout.sequence(),
                )));
            }
        }
        if input.video_mask.len() != layout.selected_video().len()
            || input.audio_mask.len() != layout.rows(Modality::Audio).len()
        {
            return Err(invalid(
                "effective update masks must cover exactly the selected media rows",
            ));
        }
        let key = Key {
            prompt: input.prompt.iter().map(|value| value.to_bits()).collect(),
            positions: std::mem::take(&mut input.positions),
            text,
        };
        Ok(Self {
            layout,
            input,
            positions,
            key,
        })
    }
}

/// One modality's fixed, page-locked four-part input and output storage. No full media image is
/// allocated or copied through a temporary pageable buffer on each step.
struct Staging {
    rows: HostBuffer,
    // Failed reads discard their potentially poisoned target; a later call pins a replacement.
    velocities: Option<HostBuffer>,
}

impl Staging {
    fn new<R: AxisName, W: AxisName>() -> io::Result<Self> {
        Ok(Self {
            rows: pinned::<R, W>()?,
            velocities: Some(pinned::<R, W>()?),
        })
    }

    fn split<R: AxisName>(
        &mut self,
        layout: &Layout,
        source: &[f32],
        modality: Modality,
        width: usize,
    ) -> io::Result<()> {
        let target = bytemuck::try_cast_slice_mut::<u8, f32>(&mut self.rows)
            .map_err(|error| io::Error::other(format!("pinned FP32 staging: {error:?}")))?;
        layout.split_into::<R>(source, modality, width, target)
    }

    fn upload<R: AxisName, W: AxisName>(
        &self,
        device: &SharedDevice,
    ) -> io::Result<HbmTensor<f32, Chips, m![R, W]>> {
        device.with_device(|device| {
            // Allocate ordinary caller-owned storage, matching the original step runner. DMA
            // completes synchronously here with both source and target owners still alive.
            let mut target = HbmTensor::<f32, Chips, m![R, W]>::alloc(device)
                .map_err(io::Error::other)?
                .into_buffer();
            block_on(
                self.rows
                    .to_hbm(&mut device.pdma)
                    .output(&mut target)
                    .into_future(),
            )
            .map_err(io::Error::other)?;
            HbmTensor::try_from(target).map_err(io::Error::other)
        })
    }

    fn download<R: AxisName, W: AxisName>(
        &mut self,
        device: &SharedDevice,
        tensor: HbmTensor<f32, Chips, m![R, W]>,
    ) -> io::Result<()> {
        let mut host = match self.velocities.take() {
            Some(host) => host,
            None => pinned::<R, W>()?,
        };
        let source = tensor.into_buffer();
        device.with_device(|device| {
            block_on(
                source
                    .to_host(&mut device.pdma)
                    .output(&mut host)
                    .into_future(),
            )
            .map_err(io::Error::other)
        })?;
        self.velocities = Some(host);
        Ok(())
    }

    fn values(&self) -> io::Result<&[f32]> {
        let host = self
            .velocities
            .as_ref()
            .expect("successful completed download");
        bytemuck::try_cast_slice::<u8, f32>(host)
            .map_err(|error| io::Error::other(format!("pinned FP32 velocities: {error:?}")))
    }
}

fn pinned<R: AxisName, W: AxisName>() -> io::Result<HostBuffer> {
    HostTensor::<f32, m![{ Chips }, R, W]>::zero()
        .pinned()
        .map(HostTensor::into_buffer)
        .map_err(io::Error::other)
}

fn upload_times(device: &SharedDevice, times: [f32; 2]) -> io::Result<runner::Times> {
    // Preserve the original [64,8] time table and video/audio rows 0/1; all other cells are zero.
    let mut values = vec![0.0_f32; 64 * 8];
    values[0] = times[0];
    values[8] = times[1];
    let host = HostTensor::<f32, m![{ Symbol<Rows<64>> }, 1 # 8]>::from_vec(values);
    device.with_device(|device| {
        let mut target = runner::Times::alloc(device).map_err(io::Error::other)?;
        block_on(
            host.to_hbm(&mut device.pdma)
                .output(&mut target)
                .into_future(),
        )
        .map_err(io::Error::other)?;
        Ok(target)
    })
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
