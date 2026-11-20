//! One tile's request metadata, ported from the preserved Furiosa Rust video rotation loader.

use super::{Chips, HeadWidth, LIVE, Rows, TOKENS};
use crate::load::{self, Load, LoadError};
use crate::ops::attention::{flash::Span, single};
use furiosa_opt_std::prelude::*;
use half::f16;

/// Rotation and visible keys of the 2,048-row tile, separately owned from decoder weights.
/// The first 1,792 rows are latent tokens; four registers and the zero class row follow them.
/// Every layer reads these same tables. They hold no latent statistics or mutable scratch.
#[derive(DeviceSend)]
pub struct Context<C: M = Chips> {
    pub rotation: single::Rotation<C>,
    pub visible: single::Visible<C>,
}

impl<C: M + 'static> Context<C> {
    /// Prepare the reference's constant tile tables once and upload them through ordinary Load.
    /// The load context selects the device; no model weights or checkpoint tensors are read.
    pub fn load(context: &mut load::Context<'_>) -> Result<Self, LoadError> {
        let (low, high) = tables();
        let low: HostTensor<bf16, m![{ C }, { m![Rows, HeadWidth] }]> =
            HostTensor::from_vec(low.repeat(C::SIZE));
        let high: HostTensor<bf16, m![{ C }, { m![Rows, HeadWidth] }]> =
            HostTensor::from_vec(high.repeat(C::SIZE));
        let visible: Vec<bf16> = (0..Span::SIZE)
            .map(|row| bf16::from_f32(if row < LIVE { 1.0 } else { 0.0 }))
            .collect();
        let visible: HostTensor<bf16, m![{ C }, { m![Span] }]> =
            HostTensor::from_vec(visible.repeat(C::SIZE));
        let low = context.field("rotation.low", "", |cx| Load::load(cx, low))?;
        let high = context.field("rotation.high", "", |cx| Load::load(cx, high))?;
        let visible = context.field("visible", "", |cx| Load::load(cx, visible))?;
        Ok(Self {
            rotation: single::Rotation { low, high },
            visible,
        })
    }
}

/// Original FP32 inverse-frequency bits, eight frequencies for each of time, height and width.
const INVERSE: [u32; 8] = [
    0x3f80_0000,
    0x3f0f_f59a,
    0x3ea1_e89b,
    0x3e36_1887,
    0x3dcc_cccd,
    0x3d66_55c2,
    0x3d01_86e3,
    0x3c91_ad39,
];

/// F16-rounded normalized pixel centers, preserving every intermediate rounding boundary.
fn axis(size: usize) -> Vec<f16> {
    (0..size)
        .map(|index| {
            let center = f16::from_f32(index as f32 + 0.5);
            let fraction = f16::from_f32(center.to_f32() / size as f32);
            let doubled = f16::from_f32(2.0 * fraction.to_f32());
            f16::from_f32(doubled.to_f32() - 1.0)
        })
        .collect()
}

/// Full 64-feature tables: the first 24 paired coordinates rotate and the remaining eight do not.
/// Q/K projection pairing places original `(d, d + 24)` in the two 32-feature input halves.
fn tables() -> (Vec<bf16>, Vec<bf16>) {
    const {
        assert!(Rows::SIZE == Span::SIZE && Rows::SIZE == 2048);
    }
    let (t, h, w) = (axis(7), axis(16), axis(16));
    let inverse = INVERSE.map(f32::from_bits);
    let head = HeadWidth::SIZE;
    let half = head / 2;
    let mut low = vec![bf16::from_f32(0.0); Rows::SIZE * head];
    let mut high = vec![bf16::from_f32(0.0); Rows::SIZE * head];
    for row in 0..Rows::SIZE {
        low[row * head..][..half].fill(bf16::from_f32(1.0));
        high[row * head + half..][..half].fill(bf16::from_f32(1.0));
        if row >= LIVE {
            continue;
        }
        let ids = if row < TOKENS {
            [t[row / 256], h[row / 16 % 16], w[row % 16]]
        } else {
            [f16::from_f32(0.0); 3]
        };
        for (axis, id) in ids.iter().enumerate() {
            let scaled = f16::from_f32(id.to_f32() * core::f32::consts::TAU).to_f32();
            for (j, inverse) in inverse.iter().enumerate() {
                let angle = scaled * inverse;
                let pair = axis * 8 + j;
                let (cos, sin) = (angle.cos(), angle.sin());
                low[row * head + pair] = bf16::from_f32(cos);
                low[row * head + half + pair] = bf16::from_f32(sin);
                high[row * head + pair] = bf16::from_f32(-sin);
                high[row * head + half + pair] = bf16::from_f32(cos);
            }
        }
    }
    (low, high)
}
