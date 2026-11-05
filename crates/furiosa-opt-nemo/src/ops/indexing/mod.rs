//! Select and store complete rows using device-resident byte offsets.

pub mod rows;
mod select;

pub use select::Select;

use crate::Module;
use crate::ops::layout::{
    Dm, Live, Rows as Layout,
    axes::{Chip, Lanes, Single},
};
use crate::ops::normalization::{RowMajor, Spread};
use crate::ops::vector::scalar;
use furiosa_opt_std::prelude::*;

/// Runtime list of individually retained contiguous rows.
pub type RowStorage<Width, D = f32> = Slice<HbmTensor<D, m![1], m![Width]>>;

type HiddenRows<Width, Rows> = HbmTensor<bf16, Chip, RowMajor<Width, Rows>>;

/// Byte offset from the source allocation or tensor base.
pub type Offset<Chips = m![1]> = HbmTensor<i32, Chips, m![Single]>;

/// Store projection feature slices as contiguous BF16 rows in HBM.
#[derive(Default, DeviceSend)]
pub struct Store;

/// The features of every row one stored slice holds: those of sixteen feature slices.
#[derive(Clone, Debug)]
struct Span<Width>(core::marker::PhantomData<Width>);

impl<Width: AxisName> AxisName for Span<Width> {
    const NAME: Ident = Ident::new("Span");
    const SIZE: usize = 16 * <crate::ops::normalization::Chunk<Width> as AxisName>::SIZE;
}

type Spanned<Width> = Symbol<Span<Width>>;

/// Rows in the slices a store transfers whole: each slice holds a span of every row.
type Stored<Width, Rows> = DmTensor<
    bf16,
    Chip,
    m![2],
    m![Width / ({ Spanned<Width> }) # 16, 1 # 16],
    m![Rows, Width % ({ Spanned<Width> })],
>;

impl Store {
    fn rows<Width: AxisName, Rows: AxisName>(
        device: &mut Device,
        input: &Spread<Width, Rows>,
    ) -> Stored<Width, Rows> {
        type Slice<Width> = Symbol<crate::ops::normalization::Chunk<Width>>;
        if const { <crate::ops::normalization::Chunk<Width> as AxisName>::SIZE == 16 } {
            return device
                .main
                .begin(input.view())
                .fetch::<m![Rows], m![Width % 16]>()
                .switch::<m![Width / ({ Spanned<Width> }) # 16, 1 # 16], m![
                    Rows,
                    Width % ({ Spanned<Width> }) / 16
                ]>(SwitchConfig::Broadcast1 {
                    slice1: 16,
                    slice0: 1,
                })
                .collect::<m![Rows, Width % ({ Spanned<Width> }) / 16], m![Width % 16]>()
                .commit_trim::<m![Width % 16]>()
                .commit();
        }
        // A slice holds several switch packets, and the switch delivers the other slices'
        // packets innermost, so the rows land in feature order through a view.
        let mut stored: Stored<Width, Rows> = DmTensor::new();
        device
            .main
            .begin(input.view())
            .fetch::<m![Rows, Width % ({ Slice<Width> }) / 16], m![Width % 16]>()
            .switch::<m![Width / ({ Spanned<Width> }) # 16, 1 # 16], m![
                Rows,
                Width % ({ Slice<Width> }) / 16,
                Width % ({ Spanned<Width> }) / ({ Slice<Width> })
            ]>(SwitchConfig::Broadcast1 {
                slice1: 16,
                slice0: 1,
            })
            .collect::<m![
                Rows,
                Width % ({ Slice<Width> }) / 16,
                Width % ({ Spanned<Width> }) / ({ Slice<Width> })
            ], m![Width % 16]>()
            .commit_trim::<m![Width % 16]>()
            .commit_view(stored.view_mut());
        stored
    }
}

impl<Width: AxisName, Rows: AxisName> Module<(&mut Device, &Spread<Width, Rows>)> for Store {
    type Output = HiddenRows<Width, Rows>;

    fn forward(&self, (device, input): (&mut Device, &Spread<Width, Rows>)) -> Self::Output {
        let mut output = HbmTensor::new();
        Self::rows(device, input)
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
        output
    }
}

/// Only the live rows: every row in one transfer when all are live, else each whole packet in
/// one, then each row of a partial last packet alone; a later row of the output is left unwritten.
impl<Width: AxisName, Rows: AxisName> Module<(&mut Device, (&Spread<Width, Rows>, Live<Rows>))>
    for Store
{
    type Output = HiddenRows<Width, Rows>;

    fn forward(
        &self,
        (device, (input, live)): (&mut Device, (&Spread<Width, Rows>, Live<Rows>)),
    ) -> Self::Output {
        let rows = Self::rows(device, input);
        let mut output: HiddenRows<Width, Rows> = HbmTensor::new();
        live.cover(
            &mut (device, &mut output),
            (
                |(device, output)| {
                    rows.view().to_hbm_view(&mut device.tdma, output.view_mut());
                },
                |(device, output), packet| {
                    rows.view()
                        .tile::<Layout<Rows>::Groups, 1, Layout<Rows>::Elements<m![Width % ({ Spanned<Width> })]>>(
                            packet,
                        )
                        .to_hbm_view(
                            &mut device.tdma,
                            output
                                .view_mut()
                                .tile::<Layout<Rows>::Groups, 1, Layout<Rows>::GroupMut<m![Width]>>(
                                    packet,
                                ),
                        );
                },
                |(device, output), row| {
                    rows.view()
                        .tile::<m![Rows], 1, m![Rows = 1 # Rows, Width % ({ Spanned<Width> })]>(row)
                        .to_hbm_view(
                            &mut device.tdma,
                            output
                                .view_mut()
                                .tile::<m![Rows], 1, m![Rows = 1 #{!} Rows, Width]>(row),
                        );
                },
            ),
        );
        output
    }
}

impl<Width: AxisName, Rows: AxisName> Module<(&mut Device, &Dm<Lanes<Rows>, Spread<Width, Rows>>)>
    for Store
{
    type Output = HiddenRows<Width, Rows>;

    fn forward(
        &self,
        (device, input): (&mut Device, &Dm<Lanes<Rows>, Spread<Width, Rows>>),
    ) -> Self::Output {
        self.forward((device, (&input.value, input.tiles)))
    }
}

/// Convert a logical element index to an exact, nonnegative I32 byte offset.
/// The caller supplies indices whose product with the element size fits I32.
#[derive(DeviceSend)]
pub struct ByteOffset<D, S>(std::marker::PhantomData<(D, S)>);

impl<D, S> Default for ByteOffset<D, S> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<D: Scalar, S: M> Module<(&mut Device, &HbmTensor<i32, Chip, m![1]>)> for ByteOffset<D, S> {
    type Output = Offset;

    fn forward(&self, (device, input): (&mut Device, &HbmTensor<i32, Chip, m![1]>)) -> Offset {
        scalar::offset::<D, S, m![Single], _>(device, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::{Count, tests::count};
    use crate::ops::attention::geometry::Hidden;
    use crate::ops::layout::axes::{Eight, Sixteen, SixtyFour};
    use crate::ops::normalization::tests::random;
    use furiosa_opt_std::runtime::launch;

    type Width = Hidden<1024>;

    #[device(chip = 1)]
    fn store_live<R: AxisName>(
        device: &mut Device,
        rows: &Count,
        input: &HbmTensor<bf16, Chip, m![R, Width]>,
    ) -> HiddenRows<Width, R> {
        let input: Spread<Width, R> = input.to_dm(&mut device.tdma);
        Store.forward((device, (&input, Live::read(rows))))
    }

    /// A store of `live` of `R` rows lands each live row as it is and leaves every later row of
    /// its output as it was.
    async fn stores_only_live_rows<R: AxisName>(device: &mut Device, live: usize) {
        let values = random(R::SIZE * Width::SIZE, 0x5851_f42d_4c95_7f2d);
        let input = HostTensor::<bf16, m![R, Width]>::from_vec(values.clone())
            .to_hbm::<Chip, m![R, Width]>(&mut device.pdma)
            .await
            .unwrap();
        let canary = bf16::from_f32(3.0);
        let mut output =
            HostTensor::<bf16, m![R, Width]>::from_vec(vec![canary; R::SIZE * Width::SIZE])
                .to_hbm::<Chip, m![R, Width]>(&mut device.pdma)
                .await
                .unwrap();
        let rows = count(live).to_hbm(&mut device.pdma).await.unwrap();
        launch(store_live, (&mut *device, &rows, &input))
            .output(&mut output)
            .await
            .unwrap();
        let landed = output
            .to_host::<m![R, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, got) in landed.iter().enumerate() {
            let row = at / Width::SIZE;
            let want = if row < live { values[at] } else { canary };
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{live} of {} rows live: row {row} column {}",
                R::SIZE,
                at % Width::SIZE
            );
        }
    }

    /// Only live rows are stored, at no row, one, a partial and a whole first packet, a partial
    /// second packet, and one short of and at a whole 64-row envelope.
    #[tokio::test(flavor = "current_thread")]
    async fn stores_only_live_rows_at_every_count() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        for live in [0, 1, 7, 8] {
            stores_only_live_rows::<Eight>(&mut device, live).await;
        }
        stores_only_live_rows::<Sixteen>(&mut device, 9).await;
        for live in [63, 64] {
            stores_only_live_rows::<SixtyFour>(&mut device, live).await;
        }
    }
}
