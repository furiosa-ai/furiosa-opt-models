//! Host rotation and key visibility for one checked joint text/audio/video request.
//!
//! Coordinates come from the caller in upstream text/audio/video order, with one three-axis
//! position per valid token. This module does not generate positions or infer their units.
//! Rotation rows follow compact query work, including identity padding, inside the allocation
//! envelope. Visibility instead describes the video/audio/text key prefix: K/V placement must
//! use the packed key order from [`Plan::key_runs`], not query or upstream position order.

use super::host::filled;
use super::packing::{self, Plan};
use half::bf16;
use std::{io, mem::size_of};

const AXES: usize = 3;
const FREQUENCIES: usize = 16;
const HEAD: usize = 128;
const HALF: usize = HEAD / 2;

/// Immutable host metadata for one request's compact Q/K rows and packed attention keys.
///
/// Both rotation tables are row-major `[query_capacity, 128]`, one shared image to replicate
/// across chips. `low` contains `[cos | sin]`, `high` contains `[-sin | cos]`. The first 48
/// pairs rotate, with 16 frequencies per supplied axis; the last 16 pairs remain unchanged.
/// This is the coefficient layout of `ops::rotary::Rotation<Head128>` after Q/K projection
/// weights and head-norm gains use `Rotary<Head128, Rotated96>::paired` ordering.
///
/// Semantic tokens land at [`packing::Token::query_row`]. All modality padding, work guards,
/// and unused allocation rows have finite identity coefficients. Identity does not grant
/// visibility: only the first [`Plan::key_count`] packed keys are visible.
#[derive(Debug)]
pub struct Positions {
    plan: Plan,
    low: Vec<bf16>,
    high: Vec<bf16>,
    visibility: Vec<bf16>,
}

impl Positions {
    /// Prepare supplied positions for `plan`, preserving FP32 arithmetic before BF16 rounding.
    ///
    /// `upstream` must contain exactly `plan.key_count()` triples, excluding upstream sequence
    /// padding. A padded framework input must first select its valid prefix. `theta` is the
    /// checked configuration's positive finite FP32 rotary base. Each coordinate narrows to
    /// FP32 before multiplication by `1 / theta.powf(j / 16)`; sine and cosine are evaluated
    /// separately in FP32, then rounded to BF16. The high table negates the rounded sine.
    ///
    /// Shape arithmetic, all input coordinates, frequencies and resulting phases are checked
    /// before allocating output buffers. Allocation failure is returned as an I/O error.
    pub fn new(plan: Plan, upstream: &[[f64; AXES]], theta: f32) -> io::Result<Self> {
        let rotation_len = elements(plan.capacity().query_capacity(), HEAD, "rotation")?;
        let visibility_len = elements(plan.capacity().key_rows(), 1, "visibility")?;
        if upstream.len() != plan.key_count() {
            return Err(invalid(format!(
                "positions: expected {} valid upstream triples, found {}",
                plan.key_count(),
                upstream.len()
            )));
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(invalid("theta: must be positive finite FP32"));
        }
        let frequencies: [f32; FREQUENCIES] =
            std::array::from_fn(|j| 1.0 / theta.powf(j as f32 / FREQUENCIES as f32));
        if frequencies.iter().any(|value| !value.is_finite()) {
            return Err(invalid("theta: produces a non-finite FP32 frequency"));
        }
        for (row, position) in upstream.iter().enumerate() {
            for (axis, &coordinate) in position.iter().enumerate() {
                if !coordinate.is_finite()
                    || frequencies
                        .iter()
                        .any(|&frequency| !(coordinate as f32 * frequency).is_finite())
                {
                    return Err(invalid(format!(
                        "positions[{row}][{axis}]: coordinate and all FP32 phases must be finite"
                    )));
                }
            }
        }

        let mut low = filled(rotation_len, bf16::ZERO)?;
        let mut high = filled(rotation_len, bf16::ZERO)?;
        let mut visibility = filled(visibility_len, bf16::ZERO)?;
        for (low, high) in low.chunks_exact_mut(HEAD).zip(high.chunks_exact_mut(HEAD)) {
            low[..HALF].fill(bf16::ONE);
            high[HALF..].fill(bf16::ONE);
        }
        for part in 0..packing::PARTS {
            for local_row in 0..plan.packed_rows_per_part() {
                let Some(token) = plan.token(part, local_row) else {
                    continue;
                };
                let position = upstream[token.upstream_row()];
                let at = token.query_row() * HEAD;
                for angle in 0..AXES * FREQUENCIES {
                    let phase =
                        position[angle / FREQUENCIES] as f32 * frequencies[angle % FREQUENCIES];
                    let cosine = bf16::from_f32(phase.cos());
                    let sine = bf16::from_f32(phase.sin());
                    low[at + angle] = cosine;
                    low[at + HALF + angle] = sine;
                    high[at + angle] = -sine;
                    high[at + HALF + angle] = cosine;
                }
            }
        }
        visibility[..plan.key_count()].fill(bf16::ONE);
        Ok(Self {
            plan,
            low,
            high,
            visibility,
        })
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Allocation shape for each replicated rotation table; active work is `plan.query_rows()`.
    #[cfg(test)]
    fn rotation_shape(&self) -> [usize; 2] {
        [self.plan.capacity().query_capacity(), HEAD]
    }

    #[cfg(test)]
    fn low(&self) -> &[bf16] {
        &self.low
    }

    #[cfg(test)]
    fn high(&self) -> &[bf16] {
        &self.high
    }

    /// Move the plan, `[low, high]` rotation images, and visibility image into host upload storage.
    pub fn into_parts(self) -> (Plan, [Vec<bf16>; 2], Vec<bf16>) {
        (self.plan, [self.low, self.high], self.visibility)
    }

    /// Row-major visibility image matching `ops::attention::flash::Visible`. Attention's runtime
    /// key limit is `plan.key_count()`, not this allocation's full span count or query work.
    #[cfg(test)]
    fn visibility_shape(&self) -> [usize; 2] {
        [self.plan.capacity().key_spans(), packing::KEY_SPAN]
    }

    #[cfg(test)]
    fn visibility(&self) -> &[bf16] {
        &self.visibility
    }
}

fn elements(rows: usize, width: usize, field: &str) -> io::Result<usize> {
    let count = rows
        .checked_mul(width)
        .ok_or_else(|| invalid(format!("{field}: element count overflows usize")))?;
    let bytes = count
        .checked_mul(size_of::<bf16>())
        .ok_or_else(|| invalid(format!("{field}: byte count overflows usize")))?;
    if bytes > isize::MAX as usize {
        return Err(invalid(format!("{field}: buffer exceeds isize::MAX bytes")));
    }
    Ok(count)
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use packing::Capacity;

    fn input(plan: Plan) -> Vec<[f64; AXES]> {
        vec![[0.0; AXES]; plan.key_count()]
    }

    #[test]
    fn matches_independent_trigonometry_and_preserves_unrotated_pairs() {
        let plan = Capacity::new(128).unwrap().plan(2).unwrap();
        let supplied = vec![[1.0, 2.0, 3.0]; plan.key_count()];
        let positions = Positions::new(plan, &supplied, 65_536.0).unwrap();
        // With theta = 2^16, frequency j is exactly 2^-j. Evaluate the known
        // coordinates in FP64 independently of the production FP32 frequency/trig path.
        for (axis, coordinate) in [1.0_f64, 2.0, 3.0].into_iter().enumerate() {
            for j in 0..16 {
                let phase = coordinate / f64::from(1_u32 << j);
                let cosine = bf16::from_f64(phase.cos());
                let sine = bf16::from_f64(phase.sin());
                let pair = axis * 16 + j;
                assert_eq!(positions.low()[pair].to_bits(), cosine.to_bits());
                assert_eq!(positions.low()[64 + pair].to_bits(), sine.to_bits());
                assert_eq!(positions.high()[pair].to_bits(), (-sine).to_bits());
                assert_eq!(positions.high()[64 + pair].to_bits(), cosine.to_bits());
            }
        }
        for pair in 48..64 {
            assert_eq!(positions.low()[pair], bf16::ONE);
            assert_eq!(positions.low()[64 + pair], bf16::ZERO);
            assert_eq!(positions.high()[pair], bf16::ZERO);
            assert_eq!(positions.high()[64 + pair], bf16::ONE);
        }
    }

    #[test]
    fn paired_tables_rotate_each_axis_in_the_expected_direction() {
        let plan = Capacity::default().plan(1).unwrap();
        let mut supplied = input(plan);
        supplied[0] = [
            std::f64::consts::FRAC_PI_2,
            0.0,
            -std::f64::consts::FRAC_PI_2,
        ];
        let positions = Positions::new(plan, &supplied, 1.0).unwrap();
        for (pair, expected) in [(0, [-3.0, 2.0]), (16, [2.0, 3.0]), (32, [3.0, -2.0])] {
            let rotated = [pair, pair + 64].map(|lane| {
                2.0 * positions.low()[lane].to_f32() + 3.0 * positions.high()[lane].to_f32()
            });
            for (got, want) in rotated.into_iter().zip(expected) {
                assert!((got - want).abs() < 1e-6, "{got} != {want}");
            }
        }
    }

    #[test]
    fn places_upstream_positions_at_independently_derived_compact_addresses() {
        for (text, t, g) in [(77, 32, 9472), (129, 48, 9728), (2048, 512, 9984)] {
            let plan = Capacity::default().plan(text).unwrap();
            let supplied: Vec<_> = (0..text + 414 + 37296)
                .map(|row| [row as f64 / 32.0, 0.0, 0.0])
                .collect();
            let positions = Positions::new(plan, &supplied, 1.0).unwrap();
            assert_eq!(positions.rotation_shape(), [41984, 128]);
            assert_eq!(positions.low().len(), 41984 * 128);
            assert_eq!(positions.high().len(), positions.low().len());
            assert_eq!(*positions.plan(), plan);
            assert!(
                positions.visibility()[..text + 414 + 37296]
                    .iter()
                    .all(|&value| value == bf16::ONE)
            );
            assert!(
                positions.visibility()[text + 414 + 37296..]
                    .iter()
                    .all(|&value| value == bf16::ZERO)
            );
            // Independent modality counts, starts and per-part strides; do not use locate/token
            // to choose the addresses whose coefficient placement is being checked.
            for (count, source, start, stride) in [
                (text, 0, 0, t),
                (414, text, t, 112),
                (37296, text + 414, t + 112, 9328),
            ] {
                for row in 0..count {
                    let destination = (row / stride * g + start + row % stride) * 128;
                    let angle = supplied[source + row][0] as f32;
                    assert_eq!(
                        positions.low()[destination].to_bits(),
                        bf16::from_f32(angle.cos()).to_bits(),
                        "text {text}, source {}",
                        source + row
                    );
                    assert_eq!(
                        positions.high()[destination].to_bits(),
                        (-bf16::from_f32(angle.sin())).to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn padding_guards_and_allocation_tail_are_finite_identity_and_invisible() {
        let plan = Capacity::default().plan(129).unwrap();
        let positions = Positions::new(plan, &input(plan), 10_000.0).unwrap();
        let mut semantic = vec![false; 41984];
        for (count, start, stride) in [(129, 0, 48), (414, 48, 112), (37296, 160, 9328)] {
            for row in 0..count {
                semantic[row / stride * 9728 + start + row % stride] = true;
            }
        }
        for (row, semantic) in semantic.into_iter().enumerate() {
            for lane in 0..128 {
                let at = row * 128 + lane;
                assert!(positions.low()[at].is_finite() && positions.high()[at].is_finite());
                if !semantic {
                    assert_eq!(
                        positions.low()[at].to_bits(),
                        if lane < 64 { 0x3f80 } else { 0 }
                    );
                    assert_eq!(
                        positions.high()[at].to_bits(),
                        if lane < 64 { 0 } else { 0x3f80 }
                    );
                }
            }
        }
        assert_eq!(positions.visibility_shape(), [20, 2048]);
        let visible = positions.visibility();
        assert_eq!(visible.len(), 40960);
        assert!(visible[..129 + 414 + 37296].iter().all(|&v| v == bf16::ONE));
        assert!(
            visible[129 + 414 + 37296..]
                .iter()
                .all(|&v| v == bf16::ZERO)
        );
    }

    #[test]
    fn narrows_coordinates_before_trigonometry_and_preserves_signed_zero() {
        let plan = Capacity::default().plan(1).unwrap();
        let mut supplied = input(plan);
        // A value that narrows to exactly one in FP32 must produce the same bits as one.
        supplied[0] = [1.0 + f64::from(f32::EPSILON) / 4.0, -0.0, 0.0];
        let positions = Positions::new(plan, &supplied, 1.0).unwrap();
        assert_eq!(positions.low()[0], bf16::from_f32(1.0_f32.cos()));
        assert_eq!(positions.low()[80].to_bits(), 0x8000);
        assert_eq!(positions.high()[16].to_bits(), 0);
        assert_eq!(positions.low()[96].to_bits(), 0);
        assert_eq!(positions.high()[32].to_bits(), 0x8000);
    }

    #[test]
    fn rejects_missing_extra_and_non_finite_upstream_positions() {
        let plan = Capacity::default().plan(1).unwrap();
        let supplied = input(plan);
        let error = Positions::new(plan, &supplied[..supplied.len() - 1], 1.0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        let mut extra = supplied.clone();
        extra.push([0.0; AXES]);
        assert!(
            Positions::new(plan, &extra, 1.0)
                .unwrap_err()
                .to_string()
                .contains("triples")
        );
        for coordinate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
            let mut bad = supplied.clone();
            let middle = bad.len() / 2;
            bad[middle][2] = coordinate;
            let error = Positions::new(plan, &bad, 1.0).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(error.to_string().contains("FP32 phases"));
        }
    }

    #[test]
    fn rejects_bad_bases_and_finite_inputs_with_non_finite_fp32_phases() {
        let plan = Capacity::default().plan(1).unwrap();
        let mut supplied = input(plan);
        for theta in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
            assert!(
                Positions::new(plan, &supplied, theta)
                    .unwrap_err()
                    .to_string()
                    .contains("theta")
            );
        }
        supplied[0][0] = f64::from(f32::MAX);
        let error = Positions::new(plan, &supplied, 0.25).unwrap_err();
        assert!(error.to_string().contains("FP32 phases"));
    }

    #[test]
    fn refuses_unrepresentable_buffers_before_examining_positions_or_allocating() {
        let gather =
            usize::MAX / (8 * packing::GATHER_BODY) * packing::GATHER_BODY + packing::GATHER_HEAD;
        let capacity = Capacity::with_gather_rows(1, gather).unwrap();
        let error = Positions::new(capacity.plan(1).unwrap(), &[], 1.0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(
            error
                .to_string()
                .contains("rotation: element count overflows")
        );
        assert!(
            elements(isize::MAX as usize / 2 + 1, 1, "test")
                .unwrap_err()
                .to_string()
                .contains("isize::MAX")
        );
    }
}
