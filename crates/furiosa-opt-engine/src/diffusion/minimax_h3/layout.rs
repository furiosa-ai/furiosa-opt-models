//! Framework rows at the checked H3 packing boundary.
//!
//! The framework owns tensor shape/dtype parsing. This host adapter accepts one canonical
//! text/audio/video sequence and preserves selected video order, including repeated indices.
//! All device addresses come from `Plan`; a media axis supplies its actual allocation stride.

use furiosa_opt_models::models::diffusion::minimax_h3::host::filled;
use furiosa_opt_models::models::diffusion::minimax_h3::packing::{
    AUDIO_TOKENS, Capacity, Modality, PARTS, Plan, VIDEO_TOKENS,
};
use furiosa_opt_std::prelude::AxisName;
use std::io;

/// One request's canonical framework indices and the plan shared with request preparation.
/// No index list or count can change after parsing.
#[derive(Debug)]
pub(super) struct Layout {
    plan: Plan,
    text: Vec<usize>,
    audio: Vec<usize>,
    video: Vec<usize>,
    output: Vec<usize>,
    sequence: usize,
}

impl Layout {
    /// Accept one text -> audio -> video request, followed only by upstream padding.
    /// `bounds` is the original packed `[0, live, sequence]` contract, including its trailing
    /// entry when no padding is present. Conditioning references and batches are refused.
    /// Text length and the upstream sequence envelope come from the supplied capacity.
    pub(super) fn parse(
        capacity: Capacity,
        rows: (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>),
        bounds: &[usize],
        sequence: usize,
    ) -> io::Result<Self> {
        let (text, audio, video, output) = rows;
        let plan = capacity.plan(text.len())?;
        for (modality, rows, count) in [
            (Modality::Text, &text, plan.text_tokens()),
            (Modality::Audio, &audio, AUDIO_TOKENS),
            (Modality::Video, &video, VIDEO_TOKENS),
        ] {
            if rows.len() != count
                || rows.iter().enumerate().any(|(index, &row)| {
                    plan.locate(modality, index)
                        .is_none_or(|token| token.upstream_row() != row)
                })
            {
                return Err(invalid(format!(
                    "{modality:?} rows must describe one canonical text/audio/video request with {count} rows and no conditioning references"
                )));
            }
        }
        if sequence < plan.key_count() || sequence > capacity.key_rows() {
            return Err(invalid(format!(
                "upstream sequence: {sequence} must be within {}..={} rows",
                plan.key_count(),
                capacity.key_rows()
            )));
        }
        if bounds != [0, plan.key_count(), sequence] {
            return Err(invalid(
                "packed bounds must be [0, live rows, upstream sequence] for one request",
            ));
        }
        if output.iter().any(|&row| {
            row.checked_sub(video[0])
                .and_then(|index| plan.locate(Modality::Video, index))
                .is_none_or(|token| token.upstream_row() != row)
        }) {
            return Err(invalid("selected video output must index valid video rows"));
        }
        Ok(Self {
            plan,
            text,
            audio,
            video,
            output,
            sequence,
        })
    }

    pub(super) fn plan(&self) -> Plan {
        self.plan
    }

    pub(super) fn rows(&self, modality: Modality) -> &[usize] {
        match modality {
            Modality::Text => &self.text,
            Modality::Audio => &self.audio,
            Modality::Video => &self.video,
        }
    }

    pub(super) fn selected_video(&self) -> &[usize] {
        &self.output
    }

    pub(super) fn sequence(&self) -> usize {
        self.sequence
    }

    /// Select the actual canonical position triples for `request::Prepared::new` or
    /// `Positions::new`. Coordinates are never synthesized or reordered into attention-key
    /// order here. As in the original framework parser, even padding coordinates must be finite;
    /// `Positions` owns the additional FP32 phase/rotary validation of the valid triples.
    pub(super) fn positions(&self, upstream: &[[f64; 3]]) -> io::Result<Vec<[f64; 3]>> {
        if upstream.len() != self.sequence {
            return Err(invalid("position triples must cover the upstream sequence"));
        }
        if upstream.iter().flatten().any(|value| !value.is_finite()) {
            return Err(invalid("position triples must be finite"));
        }
        let mut positions = filled(self.plan.key_count(), [0.0; 3])?;
        positions.copy_from_slice(&upstream[..self.plan.key_count()]);
        Ok(positions)
    }

    /// Split contiguous framework FP32 `[sequence, width]` media into `[4, R, width]`, chip
    /// after chip. Pass the row axis of `denoiser::Video<V>` or `denoiser::Audio<A>` as `R`.
    /// `width` is the positive row width already established by framework shape parsing.
    /// Every padding row, including any allocation tail beyond the logical span, is exact zero.
    #[cfg(test)]
    pub(super) fn split<R: AxisName>(
        &self,
        source: &[f32],
        modality: Modality,
        width: usize,
    ) -> io::Result<Vec<f32>> {
        // Keep the allocating host adapter and reusable pinned staging on the same mapping.
        let (_, count) = self.split_geometry::<R>(source, modality, width)?;
        let mut parts = filled(count, 0.0)?;
        self.split_into::<R>(source, modality, width, &mut parts)?;
        Ok(parts)
    }

    /// Split directly into reusable host storage. Validate both lengths before writing, then
    /// clear all padding and allocation tail
    /// rows so a changed text count cannot expose a previous call's rows.
    pub(super) fn split_into<R: AxisName>(
        &self,
        source: &[f32],
        modality: Modality,
        width: usize,
        parts: &mut [f32],
    ) -> io::Result<()> {
        let (start, count) = self.split_geometry::<R>(source, modality, width)?;
        if parts.len() != count {
            return Err(invalid(
                "split destination must fill all chip media allocations",
            ));
        }
        parts.fill(0.0);
        for index in 0..self.rows(modality).len() {
            let token = self
                .plan
                .locate(modality, index)
                .expect("parsed modality rows are valid tokens");
            let source_row = token.upstream_row() * width;
            let target = (token.part() * R::SIZE + token.local_row() - start) * width;
            parts[target..target + width].copy_from_slice(&source[source_row..source_row + width]);
        }
        Ok(())
    }

    fn split_geometry<R: AxisName>(
        &self,
        source: &[f32],
        modality: Modality,
        width: usize,
    ) -> io::Result<(usize, usize)> {
        let start = self.media_start::<R>(modality)?;
        let input = elements(self.sequence, width)?;
        if source.len() != input {
            return Err(invalid(format!(
                "framework media: expected {input} FP32 elements, found {}",
                source.len()
            )));
        }
        Ok((start, self.media_elements::<R>(width)?))
    }

    /// Gather downloaded FP32 `[4, R, width]` velocities into the framework's output order.
    /// Audio selects all its canonical rows; video uses precisely `selected_video()`, including
    /// duplicates or an empty selection. `mask` covers that selected order, not the input
    /// sequence or the complete video span. False entries remain positive zero even if the
    /// corresponding device velocity is NaN. Values in true entries retain their FP32 bits.
    pub(super) fn gather<R: AxisName>(
        &self,
        parts: &[f32],
        modality: Modality,
        width: usize,
        mask: &[bool],
    ) -> io::Result<Vec<f32>> {
        let start = self.media_start::<R>(modality)?;
        let input = self.media_elements::<R>(width)?;
        if parts.len() != input {
            return Err(invalid(format!(
                "chip media: expected {input} FP32 elements, found {}",
                parts.len()
            )));
        }
        let selected = match modality {
            Modality::Audio => &self.audio,
            Modality::Video => &self.output,
            Modality::Text => unreachable!("media_start refuses text"),
        };
        if mask.len() != selected.len() {
            return Err(invalid("update mask must cover exactly the selected rows"));
        }
        let mut gathered = filled(elements(selected.len(), width)?, 0.0)?;
        let first = self.rows(modality)[0];
        for (index, (&row, &enabled)) in selected.iter().zip(mask).enumerate() {
            if enabled {
                let token = self
                    .plan
                    .locate(modality, row - first)
                    .expect("parsed selected rows are valid media tokens");
                let source = (token.part() * R::SIZE + token.local_row() - start) * width;
                let target = index * width;
                gathered[target..target + width].copy_from_slice(&parts[source..source + width]);
            }
        }
        Ok(gathered)
    }

    /// Media buffers contain a modality alone; remove only its local offset in the packed
    /// sequence. The checked logical span is a minimum extent, never an allocation stride.
    fn media_start<R: AxisName>(&self, modality: Modality) -> io::Result<usize> {
        if modality == Modality::Text {
            return Err(invalid("FP32 media conversion requires Audio or Video"));
        }
        let rows = self
            .plan
            .part(0)
            .expect("a checked plan has four parts")
            .span(modality)
            .rows();
        if R::SIZE < rows.len() {
            return Err(invalid(format!(
                "{modality:?} allocation: {} rows cannot hold the {}-row logical span",
                R::SIZE,
                rows.len()
            )));
        }
        Ok(rows.start)
    }

    fn media_elements<R: AxisName>(&self, width: usize) -> io::Result<usize> {
        let rows = PARTS
            .checked_mul(R::SIZE)
            .ok_or_else(|| invalid("chip media row count overflows usize"))?;
        elements(rows, width)
    }
}

fn elements(rows: usize, width: usize) -> io::Result<usize> {
    if width == 0 {
        return Err(invalid("media width must be positive"));
    }
    rows.checked_mul(width)
        .ok_or_else(|| invalid("media element count overflows usize"))
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_models::tensor::axis::Rows;

    type Indices = (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>);

    fn rows(text: usize) -> Indices {
        let audio = text..text + AUDIO_TOKENS;
        let video = audio.end..audio.end + VIDEO_TOKENS;
        (
            (0..text).collect(),
            audio.collect(),
            video.clone().collect(),
            vec![video.end - 1, video.start, video.end - 1, video.start + 1],
        )
    }

    fn layout(text: usize, padding: usize) -> Layout {
        let capacity = Capacity::default();
        let used = capacity.plan(text).unwrap().key_count();
        Layout::parse(
            capacity,
            rows(text),
            &[0, used, used + padding],
            used + padding,
        )
        .unwrap()
    }

    fn refused(result: io::Result<Layout>, reason: &str) {
        let error = result.unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains(reason), "{error}");
    }

    #[test]
    fn accepts_dynamic_text_and_only_capacity_bounded_upstream_padding() {
        let capacity = Capacity::new(513).unwrap();
        for text in [1, 129, capacity.max_text()] {
            let used = capacity.plan(text).unwrap().key_count();
            let sequence = capacity.key_rows();
            let layout =
                Layout::parse(capacity, rows(text), &[0, used, sequence], sequence).unwrap();
            assert_eq!(layout.plan().text_tokens(), text);
            assert_eq!(layout.rows(Modality::Audio).first(), Some(&text));
            assert_eq!(layout.sequence(), sequence);
        }
        let capacity = Capacity::default();
        let used = capacity.plan(capacity.max_text()).unwrap().key_count();
        Layout::parse(capacity, rows(capacity.max_text()), &[0, used, used], used).unwrap();
        for text in [0, capacity.max_text() + 1] {
            refused(Layout::parse(capacity, rows(text), &[], 0), "text_tokens");
        }
        let used = capacity.plan(5).unwrap().key_count();
        for sequence in [used - 1, capacity.key_rows() + 1] {
            refused(
                Layout::parse(capacity, rows(5), &[0, used, sequence], sequence),
                "upstream sequence",
            );
        }
    }

    #[test]
    fn refuses_noncanonical_modalities_batches_and_nonvideo_output_indices() {
        let capacity = Capacity::default();
        let used = capacity.plan(5).unwrap().key_count();
        for (modality, reason) in [(0, "Text rows"), (1, "Audio rows"), (2, "Video rows")] {
            let mut supplied = rows(5);
            let indices = match modality {
                0 => &mut supplied.0,
                1 => &mut supplied.1,
                _ => &mut supplied.2,
            };
            indices.swap(0, 1);
            refused(
                Layout::parse(capacity, supplied, &[0, used, used], used),
                reason,
            );
        }
        for (audio, reason) in [(true, "Audio rows"), (false, "Video rows")] {
            let mut supplied = rows(5);
            if audio {
                supplied.1.pop();
            } else {
                supplied.2.push(used);
            }
            refused(
                Layout::parse(capacity, supplied, &[0, used, used], used),
                reason,
            );
        }
        for bounds in [
            vec![0, used],
            vec![0, 5, used, used],
            vec![0, used - 1, used],
        ] {
            refused(
                Layout::parse(capacity, rows(5), &bounds, used),
                "packed bounds",
            );
        }
        for output in [0, 5, used, usize::MAX] {
            let mut supplied = rows(5);
            supplied.3 = vec![output];
            refused(
                Layout::parse(capacity, supplied, &[0, used, used], used),
                "selected video output",
            );
        }
    }

    #[test]
    fn preserves_canonical_position_values_and_drops_only_upstream_padding() {
        let layout = layout(129, 21);
        let mut positions: Vec<_> = (0..layout.sequence())
            .map(|row| [row as f64, -(row as f64), 0.25])
            .collect();
        assert_eq!(
            layout.positions(&positions).unwrap(),
            positions[..layout.plan().key_count()]
        );
        let error = layout
            .positions(&positions[..positions.len() - 1])
            .unwrap_err();
        assert!(error.to_string().contains("cover the upstream sequence"));
        positions.last_mut().unwrap()[1] = f64::NAN;
        let error = layout.positions(&positions).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("finite"));
    }

    #[test]
    fn splits_with_the_media_allocation_stride_and_gathers_original_selection_and_masks() {
        // The original Furiosa host fixture uses the final video row on chip 3, local row 9311,
        // and a padded video stride of 9344. Its audio chip boundary is row 112 in stride 128.
        // Use the same host boundary with small FP32 rows, keeping expected anchors independent
        // of Plan's address implementation. Core packing/rotary math is tested in models.
        let layout = layout(2, 21);
        let source: Vec<_> = (0..layout.sequence())
            .flat_map(|row| [row as f32 + 0.25, -0.0])
            .collect();
        let mut video = layout
            .split::<Rows<9344>>(&source, Modality::Video, 2)
            .unwrap();
        let first = 2 + AUDIO_TOKENS;
        assert_eq!(video[0], source[first * 2]);
        assert_eq!(video[9327 * 2], source[(first + 9327) * 2]);
        assert_eq!(video[9344 * 2], source[(first + 9328) * 2]);
        assert!(video[9328 * 2..9344 * 2].iter().all(|v| v.to_bits() == 0));
        assert!(
            video[(3 * 9344 + 9312) * 2..]
                .iter()
                .all(|v| v.to_bits() == 0)
        );
        assert_eq!(
            video[(3 * 9344 + 9311) * 2],
            source[(layout.plan().key_count() - 1) * 2]
        );
        video[0] = f32::NAN;
        let output = layout
            .gather::<Rows<9344>>(&video, Modality::Video, 2, &[true, false, true, true])
            .unwrap();
        let last = source[(layout.plan().key_count() - 1) * 2];
        assert_eq!(output[0], last);
        assert!(output[2..4].iter().all(|v| v.to_bits() == 0));
        assert_eq!(output[4], last);
        assert_eq!(output[6], source[(first + 1) * 2]);
        assert_eq!(output[1].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(layout.selected_video()[0], layout.selected_video()[2]);

        let audio = layout
            .split::<Rows<128>>(&source, Modality::Audio, 2)
            .unwrap();
        assert_eq!(audio[128 * 2], source[(2 + 112) * 2]);
        assert!(audio[112 * 2..128 * 2].iter().all(|v| v.to_bits() == 0));
        let mut mask = vec![true; AUDIO_TOKENS];
        mask[AUDIO_TOKENS - 1] = false;
        let output = layout
            .gather::<Rows<128>>(&audio, Modality::Audio, 2, &mask)
            .unwrap();
        assert_eq!(
            &output[..(AUDIO_TOKENS - 1) * 2],
            &source[4..(2 + AUDIO_TOKENS - 1) * 2]
        );
        assert!(
            output[(AUDIO_TOKENS - 1) * 2..]
                .iter()
                .all(|v| v.to_bits() == 0)
        );
    }

    #[test]
    fn refuses_invalid_media_shapes_strides_and_masks_and_accepts_empty_selection() {
        let layout = layout(5, 0);
        let source = vec![0.0; layout.sequence()];
        for (reason, result) in [
            (
                "Audio or Video",
                layout.split::<Rows<128>>(&source, Modality::Text, 1),
            ),
            (
                "logical span",
                layout.split::<Rows<111>>(&source, Modality::Audio, 1),
            ),
            (
                "framework media",
                layout.split::<Rows<128>>(&[], Modality::Audio, 1),
            ),
            (
                "width must be positive",
                layout.split::<Rows<128>>(&[], Modality::Audio, 0),
            ),
            (
                "overflows",
                layout.split::<Rows<128>>(&[], Modality::Audio, usize::MAX),
            ),
            (
                "chip media",
                layout.gather::<Rows<128>>(&[], Modality::Audio, 1, &[]),
            ),
            (
                "update mask",
                layout.gather::<Rows<128>>(&vec![0.0; 4 * 128], Modality::Audio, 1, &[]),
            ),
        ] {
            let error = result.unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(error.to_string().contains(reason), "{error}");
        }
        let capacity = Capacity::default();
        let used = capacity.plan(5).unwrap().key_count();
        let mut supplied = rows(5);
        supplied.3.clear();
        let empty = Layout::parse(capacity, supplied, &[0, used, used], used).unwrap();
        assert!(
            empty
                .gather::<Rows<9344>>(&vec![f32::NAN; 4 * 9344], Modality::Video, 1, &[])
                .unwrap()
                .is_empty()
        );
    }
}
