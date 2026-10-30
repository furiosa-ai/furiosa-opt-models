//! A coupled video/audio layer: each stream's self-, text- and mutual cross-attention, then its
//! feed-forward.

use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Updates video and audio through self-, text- and mutual cross-attention, then feed-forward.
#[derive(DeviceSend)]
pub struct Coupled<
    VideoSelf,
    AudioSelf,
    VideoText,
    AudioText,
    VideoCross,
    AudioCross,
    VideoMlp,
    AudioMlp,
> {
    pub self_attention: (VideoSelf, AudioSelf),
    pub text_attention: (VideoText, AudioText),
    pub cross_attention: (VideoCross, AudioCross),
    pub feedforward: (VideoMlp, AudioMlp),
}

/// The env holds the condition, then the self, text and cross attentions' contexts, video's before
/// audio's; the state holds every branch's slot in that order, then the feed-forwards'.
impl<
    's,
    'e,
    X,
    Y,
    E,
    VideoSelfEnv: Copy,
    AudioSelfEnv: Copy,
    VideoTextEnv: Copy,
    AudioTextEnv: Copy,
    VideoCrossEnv: Copy,
    AudioCrossEnv: Copy,
    VideoSelfState,
    AudioSelfState,
    VideoTextState,
    AudioTextState,
    VideoCrossState,
    AudioCrossState,
    VideoMlpState,
    AudioMlpState,
    VideoSelf,
    AudioSelf,
    VideoText,
    AudioText,
    VideoCross,
    AudioCross,
    VideoMlp,
    AudioMlp,
>
    Module<
        (X, Y),
        (
            &'e E,
            (
                (VideoSelfEnv, AudioSelfEnv),
                (VideoTextEnv, AudioTextEnv),
                (VideoCrossEnv, AudioCrossEnv),
            ),
        ),
        &'s mut (
            (VideoSelfState, AudioSelfState),
            (VideoTextState, AudioTextState),
            (VideoCrossState, AudioCrossState),
            (VideoMlpState, AudioMlpState),
        ),
    >
    for Coupled<
        VideoSelf,
        AudioSelf,
        VideoText,
        AudioText,
        VideoCross,
        AudioCross,
        VideoMlp,
        AudioMlp,
    >
where
    VideoSelf: for<'a> Module<&'a X, (&'e E, VideoSelfEnv), &'s mut VideoSelfState, Output = X>,
    AudioSelf: for<'a> Module<&'a Y, (&'e E, AudioSelfEnv), &'s mut AudioSelfState, Output = Y>,
    VideoText: for<'a> Module<&'a X, (&'e E, VideoTextEnv), &'s mut VideoTextState, Output = X>,
    AudioText: for<'a> Module<&'a Y, (&'e E, AudioTextEnv), &'s mut AudioTextState, Output = Y>,
    VideoCross: for<'a, 'b> Module<
            (&'a X, &'b Y),
            (&'e E, VideoCrossEnv),
            &'s mut VideoCrossState,
            Output = X,
        >,
    AudioCross: for<'a, 'b> Module<
            (&'a Y, &'b X),
            (&'e E, AudioCrossEnv),
            &'s mut AudioCrossState,
            Output = Y,
        >,
    VideoMlp: for<'a> Module<&'a X, (&'e E, ()), &'s mut VideoMlpState, Output = X>,
    AudioMlp: for<'a> Module<&'a Y, (&'e E, ()), &'s mut AudioMlpState, Output = Y>,
{
    type Output = (X, Y);

    fn forward(
        &self,
        device: &mut Device,
        (condition, ((vs, as_), (vt, at), (vc, ac))): (
            &'e E,
            (
                (VideoSelfEnv, AudioSelfEnv),
                (VideoTextEnv, AudioTextEnv),
                (VideoCrossEnv, AudioCrossEnv),
            ),
        ),
        (
            (video_self, audio_self),
            (video_text, audio_text),
            (video_cross, audio_cross),
            (video_mlp, audio_mlp),
        ): &'s mut (
            (VideoSelfState, AudioSelfState),
            (VideoTextState, AudioTextState),
            (VideoCrossState, AudioCrossState),
            (VideoMlpState, AudioMlpState),
        ),
        (video, audio): (X, Y),
    ) -> Self::Output {
        let video =
            self.self_attention
                .0
                .forward(&mut *device, (condition, vs), video_self, &video);
        let audio =
            self.self_attention
                .1
                .forward(&mut *device, (condition, as_), audio_self, &audio);
        let video =
            self.text_attention
                .0
                .forward(&mut *device, (condition, vt), video_text, &video);
        let audio =
            self.text_attention
                .1
                .forward(&mut *device, (condition, at), audio_text, &audio);
        // Both cross-attention directions observe the same pair of pre-update hidden values.
        let next_video = self.cross_attention.0.forward(
            &mut *device,
            (condition, vc),
            video_cross,
            (&video, &audio),
        );
        let next_audio = self.cross_attention.1.forward(
            &mut *device,
            (condition, ac),
            audio_cross,
            (&audio, &video),
        );
        let video =
            self.feedforward
                .0
                .forward(&mut *device, (condition, ()), video_mlp, &next_video);
        let audio = self
            .feedforward
            .1
            .forward(device, (condition, ()), audio_mlp, &next_audio);
        (video, audio)
    }
}
