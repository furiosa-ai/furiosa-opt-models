use furiosa_opt_std::prelude::*;

use super::Head;
use crate::Module;
use crate::models::nemo::serving::Rows;

#[derive(Clone, Debug)]
pub struct Space<C, S, T, R = T, E = Rows>(core::marker::PhantomData<(C, S, T, R, E)>);

impl<C: M + 'static, S: M + 'static, T: M + 'static, R: M + 'static, E: AxisName> AxisName
    for Space<C, S, T, R, E>
{
    const NAME: Ident = Ident::new("Rotation");
    const SIZE: usize = E::SIZE;
}

impl<C: M + 'static, S: M + 'static, T: M + 'static, R: M + 'static, E: AxisName>
    crate::compose::Space for Space<C, S, T, R, E>
{
    type Envelope = E;
}

impl<C: M + 'static, S: M + 'static, T: M + 'static, R: M + 'static, E: AxisName>
    crate::ops::attention::rope::resident::Mapping for Space<C, S, T, R, E>
{
    type Cluster = C;
    type Slices = S;
    type Input = m![{ T }, InputHalf, Half];
    type Output = m![{ T }, OutputHalf, Half];
    type Coefficients = m![{ R }, InputHalf, Half];
    type Lanes = m![OutputHalf];
    type Feed = m![{ T }, Half / 32, InputHalf];
    type Packet = m![Half % 32];
    type Flits = m![{ T }, Half / 32, InputHalf, Half % 32 / 16];
    type Flit = m![Half % 16];
    type Steps = m![{ T }, Half / 32, InputHalf];
    type Sums = m![{ T }, Half / 32];
    type Groups = m![{ T }, Half / 32, OutputHalf, Half % 32 / 8];
    type Result = m![Half % 8];
}

axes![InputHalf = 2, OutputHalf = 2, Half = 64];
pub type Stored = m![Rows, OutputHalf, InputHalf, Half];
pub type Rotated<B, C, S, T> = DmTensor<bf16, B, C, S, m![{ T }, Head]>;
pub const THETA: f32 = 1_000_000.0;

/// One wave's [`Stored`] table landed beside the rows it rotates.
pub type Landed<B, C, S, T> = DmTensor<bf16, B, C, S, m![OutputHalf, { T }, InputHalf, Half]>;

/// Landed coefficients in the register file, for [`apply`].
pub type Register<B, C, S, T> =
    TrfTensor<bf16, B, C, S, m![OutputHalf], m![{ T }, InputHalf, Half]>;

/// Rotates each head's feature pairs (f, f + 64) by its row's position; `T` is the row digit a
/// slice keeps in time.
pub fn rotate<B: M, C: M + 'static, S: M + 'static, T: M + 'static>(
    device: &mut Device,
    input: &Rotated<B, C, S, T>,
    coefficients: &Landed<B, C, S, T>,
) -> Rotated<B, C, S, T> {
    let register = fill(device, coefficients.view());
    apply(device, input.view(), &register)
}

/// `coefficients` in the register file, so one fill serves every [`apply`] of the same rows.
pub fn fill<B: M, C: M, S: M, T: M>(
    device: &mut Device,
    coefficients: DmTensorView<'_, bf16, B, C, S, m![OutputHalf, { T }, InputHalf, Half]>,
) -> Register<B, C, S, T> {
    device
        .sub
        .begin(coefficients)
        .fetch::<m![OutputHalf, { T }, InputHalf, Half / 16], m![Half % 16]>()
        .collect::<m![OutputHalf, { T }, InputHalf, Half / 16], m![Half % 16]>()
        .to_trf::<m![OutputHalf], m![{ T }, InputHalf, Half]>()
}

/// [`rotate`] against an already filled register.
pub fn apply<B: M, C: M + 'static, S: M + 'static, T: M + 'static>(
    device: &mut Device,
    input: DmTensorView<'_, bf16, B, C, S, m![{ T }, Head]>,
    coefficients: &Register<B, C, S, T>,
) -> Rotated<B, C, S, T> {
    // SAFETY: Head is the contiguous product InputHalf × Half within each row.
    let rows = unsafe { input.reshape::<B, C, S, m![{ T }, InputHalf, Half]>() };
    let rotated = crate::ops::attention::rope::Rotary::<Head, Head, bf16, B>::new().forward((
        device,
        crate::ops::layout::Dm::<Space<C, S, T>, _> {
            tiles: crate::compose::Tiles::over(Rows::SIZE),
            value: (rows, coefficients),
        },
    ));
    // SAFETY: concatenating OutputHalf × Half restores one contiguous Head per row.
    unsafe { rotated.value.reshape() }
}

/// Rotation coefficients of 64 consecutive positions starting at `position`.
pub fn coefficients(position: u16) -> HostTensor<bf16, Stored> {
    assert!(position <= u16::MAX - 63);
    HostTensor::from_vec((0u16..64).flat_map(|row| {
        (0..2).flat_map(move |output| {
            (0..2).flat_map(move |input| {
                (0u16..64).map(move |feature| {
                    let angle =
                        f32::from(position + row) / THETA.powf(f32::from(2 * feature) / 128.0);
                    bf16::from_f32(match (output, input) {
                        (0, 0) | (1, 1) => angle.cos(),
                        (0, 1) => -angle.sin(),
                        (1, 0) => angle.sin(),
                        _ => unreachable!(),
                    })
                })
            })
        })
    }))
}
