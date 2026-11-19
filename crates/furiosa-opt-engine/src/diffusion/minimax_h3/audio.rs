//! Audio decoding on the pipeline's original four chips.
//!
//! The host adapter preserves the original channel-major latent boundary, one complete decoder
//! launch and per-channel waveform trimming. Numerical work remains in the existing Module tree.

use super::{flight::Flight, weights::Component};
use crate::residency::{Resident, SharedDevice, block_on};
use furiosa_opt_models::models::diffusion::minimax_h3::{audio, embeddings::Chips, packing::PARTS};
use furiosa_opt_models::ops::convolution::{
    frames::{Stereo, Window},
    input::Latent as Channels,
};
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_std::prelude::*;
use std::io;

/// Contiguous FP32 architecture latents, before time-major packing and allocation padding.
pub(super) const INPUT_SHAPE: [usize; 3] = [Stereo::SIZE, Channels::SIZE, audio::Latent::SAMPLES];

/// Contiguous FP32 waveform layout returned by `decode`, with allocation padding removed.
pub(super) const OUTPUT_SHAPE: [usize; 3] = [Stereo::SIZE, 1, audio::Samples::SAMPLES];

/// Internal component view; an exposed view additionally retains the complete pipeline owner.
pub(super) struct Decoder {
    _owner: Component<audio::Decoder>,
    decoder: Resident<audio::Decoder>,
    device: SharedDevice,
    active: tokio::sync::Mutex<()>,
}

impl Decoder {
    /// Bind already placed weights and prime the original complete decoder entry before sizing
    /// the encoder's resident-prefix grant. The cloned resident view shares the existing cells;
    /// this adapter neither opens a device nor loads or uploads a second weight tree.
    pub(super) async fn from_resident(
        owner: Component<audio::Decoder>,
        device: SharedDevice,
    ) -> io::Result<Self> {
        if device.placement().chips.len() != PARTS {
            return Err(invalid(
                "H3 audio decoding requires the pipeline's four-chip device",
            ));
        }
        device.admit(owner.resident())?.landed().await?;
        device.with_device(|device| kernel::decode.load(device).map_err(io::Error::other))?;
        let decoder = owner.resident().clone();
        Ok(Self {
            _owner: owner,
            decoder,
            device,
            active: tokio::sync::Mutex::new(()),
        })
    }

    /// Decode contiguous FP32 `[2,32,207]` latents to channel-major `[2,1,165600]` samples.
    /// Shape and payload length are checked before any DMA or execution. Input transposition
    /// only copies FP32 values. No latent-statistics transform is introduced, and the existing
    /// Module tree retains every BF16 rounding boundary.
    /// The original caller reads the first of four identical waveform replicas and removes
    /// each channel's trailing allocation padding separately.
    pub(super) async fn decode(&self, shape: &[usize], values: &[f32]) -> io::Result<Vec<f32>> {
        let packed = Latents::new(shape, values)?.pack();
        let _active = self.active.lock().await;
        let input: audio::Input = self.device.with_device(|device| {
            // Complete the replicated upload while its host source is still owned. There is
            // no cancellation point between DMA submission and its completion.
            block_on(packed.to_hbm(&mut device.pdma).into_future()).map_err(io::Error::other)
        })?;
        input.ready().await.map_err(io::Error::other)?;
        let workspace = self
            .device
            .with_device(|device| Workspace::allocate(device, input))?;
        let mut flight = Flight::new(workspace, 1);
        let ready = self.device.admit(&self.decoder)?.landed().await?;
        self.device.launch(ready, |launches, model| {
            let work = flight.held_mut();
            let mut waveform: audio::Output = alias(&work.waveform, ..)?;
            let mut state: kernel::State = Default::default();
            let accepted = launches.forward_into_tracked(
                model,
                kernel::decode,
                (),
                &mut state,
                &work.input,
                &mut waveform,
            )?;
            // Record immediately after acceptance. Flight retains the original input and
            // output allocation through completion, cancellation and execution failure.
            flight.accepted(waveform.into_buffer(), accepted);
            Ok(())
        })?;
        let [waveform]: [HbmBuffer; 1] =
            flight.finish().await?.try_into().map_err(|_| {
                io::Error::other("H3 audio decoder returned an invalid output count")
            })?;
        let waveform = audio::Output::<Chips>::try_from(waveform).map_err(io::Error::other)?;
        let samples = self.device.with_device(|device| {
            // Keep the original output allocation until the download ends, including errors.
            // As in the repro, download all replicas, then consume only the first one below.
            block_on(
                waveform
                    .to_host::<m![{ Chips }, Stereo, { Symbol<audio::Samples> }]>(&mut device.pdma)
                    .into_future(),
            )
            .map(HostTensor::into_vec)
            .map_err(io::Error::other)
        })?;
        let mut output = Vec::with_capacity(OUTPUT_SHAPE.into_iter().product());
        for channel in 0..Stereo::SIZE {
            let start = channel * audio::Samples::SIZE;
            output.extend_from_slice(&samples[start..start + audio::Samples::SAMPLES]);
        }
        Ok(output)
    }

    /// The same typed input and output allocations as a call; the entry is already primed.
    pub(super) fn workspace_bytes(&self) -> io::Result<usize> {
        self.device.with_device(|device| {
            super::memory::measure(device, |device| {
                let input = audio::Input::<Chips>::alloc(device).map_err(io::Error::other)?;
                Workspace::allocate(device, input)
            })
        })
    }
}

/// A checked architecture-layout view. Keeping construction separate from packing prevents a
/// correctly sized payload with a different declared layout from reaching the compiled entry.
struct Latents<'a> {
    values: &'a [f32],
}

impl<'a> Latents<'a> {
    fn new(shape: &[usize], values: &'a [f32]) -> io::Result<Self> {
        if shape != INPUT_SHAPE || values.len() != INPUT_SHAPE.into_iter().product::<usize>() {
            return Err(invalid(
                "audio decode requires contiguous float32 [2,32,207] with exactly 13248 values",
            ));
        }
        Ok(Self { values })
    }

    fn pack(self) -> HostTensor<f32, m![Stereo, { Symbol<audio::Latent> }, Channels]> {
        let mut rows = vec![0.0_f32; Stereo::SIZE * audio::Latent::SIZE * Channels::SIZE];
        for batch in 0..Stereo::SIZE {
            for time in 0..audio::Latent::SAMPLES {
                for channel in 0..Channels::SIZE {
                    let source = (batch * Channels::SIZE + channel) * audio::Latent::SAMPLES + time;
                    let destination =
                        (batch * audio::Latent::SIZE + time) * Channels::SIZE + channel;
                    rows[destination] = self.values[source];
                }
            }
        }
        HostTensor::from_vec(rows)
    }
}

struct Workspace {
    input: audio::Input,
    waveform: HbmBuffer,
}

impl Workspace {
    fn allocate(device: &mut Device, input: audio::Input) -> io::Result<Self> {
        Ok(Self {
            input,
            waveform: audio::Output::<Chips>::alloc(device)
                .map_err(io::Error::other)?
                .into_buffer(),
        })
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Preserve the repro's channel-major latent boundary and zero-padded final time row.
    #[test]
    fn preserves_audio_latents() {
        let values: Vec<f32> = (0..INPUT_SHAPE.into_iter().product::<usize>())
            .map(|index| index as f32)
            .collect();
        let rows = Latents::new(&INPUT_SHAPE, &values)
            .unwrap()
            .pack()
            .into_vec();
        assert_eq!(
            [0, 1, 32, 6624, 6656, 6657].map(|index| rows[index]),
            [0., 207., 1., 0., 6624., 6831.]
        );
        for shape in [[2, 32, 208], [2, 207, 32]] {
            assert_eq!(
                Latents::new(&shape, &values).err().unwrap().kind(),
                io::ErrorKind::InvalidInput
            );
        }
        assert_eq!(
            Latents::new(&INPUT_SHAPE, &values[..values.len() - 1])
                .err()
                .unwrap()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }
}

/// Concrete binding of the unchanged complete decoder and its ordinary nested Module state.
mod kernel {
    use furiosa_opt_models::models::diffusion::minimax_h3::{audio, embeddings::Chips};
    use furiosa_opt_models::{Module, tensor::memory::Hbm};
    use furiosa_opt_std::prelude::*;

    type UnitState = ((), ((), ()));
    type BlockState = (UnitState, (UnitState, UnitState));
    type ForkState = (BlockState, (BlockState, BlockState));
    type StageState = ((), (ForkState, ()));
    type StagesState = (
        StageState,
        (
            StageState,
            (
                StageState,
                (StageState, (StageState, (StageState, StageState))),
            ),
        ),
    );
    pub(super) type State = (((), ((), StagesState)), ((), ()));

    #[device(chip = 4)]
    pub(super) fn decode(
        device: &mut Device,
        module: &&audio::Decoder<Chips, &Hbm>,
        env: (),
        state: &mut State,
        input: &audio::Input,
    ) -> audio::Output {
        module.forward(device, env, state, input)
    }
}
