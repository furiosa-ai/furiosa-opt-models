use crate::{
    compose::{Tile, Tiles},
    ops::{
        add::Add,
        collective::AllReduce,
        layout::{Dm, axes::Parallel},
    },
};
use furiosa_opt_std::prelude::*;

axes![Shard = 2, Term = 2];
pub type Chips = m![Shard];
axes![Wave = 64];
type Space = Tile<Wave, Wave>;

type Values<C, S, T, P> = DmTensor<bf16, Chips, C, S, m![{ T }, { P }]>;

/// A rank's partial and the peer rank's copy of the same rows, received in its place.
pub struct Exchanged<C: M, S: M, T: M, P: M> {
    partial: Values<C, S, T, P>,
    received: Values<C, S, T, P>,
}

impl<C: M, S: M, T: M, P: M> Exchanged<C, S, T, P> {
    /// `partial` before any of it is sent, for `send` or `send_tile` to fill the peer's copy.
    pub fn new(partial: Values<C, S, T, P>) -> Self {
        Self {
            partial,
            received: Values::new(),
        }
    }

    /// Sends the whole partial to the peer rank.
    pub fn send(&mut self, device: &mut Device) {
        AllReduce::<Parallel<2>>::default().stage(
            device,
            Dm::<Space, _> {
                tiles: Tiles::over(Wave::SIZE),
                value: (self.partial.view(), self.received.view_mut()),
            },
        );
    }

    /// Sends the partial's partition tile at index 0 of axis `I` (read as `H`, written as `W`); the
    /// rest of the peer's copy stays unwritten, so rows outside the tile sum to garbage the caller
    /// never reads.
    pub fn send_tile<I: M, H: M, W: M>(&mut self, device: &mut Device) {
        AllReduce::<Parallel<2>>::default().stage(
            device,
            Dm::<Space, _> {
                tiles: Tiles::over(Wave::SIZE),
                value: (
                    self.partial.view().slice_tile::<I, 1, H>(0),
                    self.received.view_mut().slice_tile::<I, 1, W>(0),
                ),
            },
        );
    }
}

pub fn complete<C: M, S: M, T: M, P: M>(
    device: &mut Device,
    partial: Values<C, S, T, P>,
    residual: &Values<C, S, T, P>,
) -> Values<C, S, T, P> {
    let exchanged = exchange(device, partial);
    finish(device, exchanged, residual)
}

/// Sends the partial to the peer rank; the sum waits only for `finish`, so work issued between
/// the two runs while the transfer is in flight.
pub fn exchange<C: M, S: M, T: M, P: M>(
    device: &mut Device,
    partial: Values<C, S, T, P>,
) -> Exchanged<C, S, T, P> {
    let mut received = Values::new();
    AllReduce::<Parallel<2>>::default().stage(
        device,
        Dm::<Space, _> {
            tiles: Tiles::over(Wave::SIZE),
            value: (partial.view(), received.view_mut()),
        },
    );
    Exchanged { partial, received }
}

pub fn finish<C: M, S: M, T: M, P: M>(
    device: &mut Device,
    Exchanged { partial, received }: Exchanged<C, S, T, P>,
    residual: &Values<C, S, T, P>,
) -> Values<C, S, T, P> {
    const { assert!(T::SIZE % 20 == 0) };
    let sum: Values<C, S, T, P> = AllReduce::<Parallel<2>>::default()
        .complete::<Space, Chips, C, S, _, T, P>(
            &mut *device,
            Dm::<Space, _> {
                tiles: Tiles::over(Wave::SIZE),
                value: (partial.view(), received.view()),
            },
        )
        .value;
    Add.stage::<Space, Chips, C, S, _, T, P>(
        device,
        Dm::<Space, _> {
            tiles: Tiles::over(Wave::SIZE),
            value: (sum.view(), residual.view()),
        },
    )
    .value
    .commit()
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_std::runtime::launch;

    axes![Rows = 64, Hidden = 5120];
    type Input = HbmTensor<bf16, Chips, m![Rows, Hidden]>;

    #[device(chip = 2)]
    fn finish(device: &mut Device, partial: &Input, residual: &Input) -> Input {
        type Cluster = m![Hidden / 2560];
        type Slice = m![Hidden % 2560 / 80, Rows / 8];
        type Time = m![Rows % 8, Hidden % 80 / 8];
        type Packet = m![Hidden % 8];
        let partial: DmTensor<bf16, Chips, Cluster, Slice, m![{ Time }, { Packet }]> =
            partial.to_dm(&mut device.tdma);
        let residual: DmTensor<bf16, Chips, Cluster, Slice, m![{ Time }, { Packet }]> =
            residual.to_dm(&mut device.tdma);
        complete::<Cluster, Slice, Time, Packet>(device, partial, &residual)
            .to_hbm(&mut device.tdma)
    }

    /// Both ranks round the partial sum to BF16 before adding their replicated residual.
    #[tokio::test(flavor = "current_thread")]
    async fn rounds_before_residual() {
        // Given
        let mut device = Device::new(finish.topology()).unwrap();
        let count = Rows::SIZE * Hidden::SIZE;
        let cases = [
            [256.0_f32, 1.0, -256.0],
            [-256.0, -1.0, 256.0],
            [1.5, -0.5, 0.03125],
            [0.00390625, 1.0, -1.0],
        ];
        let case = |index: usize| {
            let row = index / Hidden::SIZE;
            let column = index % Hidden::SIZE;
            cases[(row * 7 + column * 3 + column / 80) % cases.len()]
        };
        let partial = HostTensor::<bf16, m![Shard, Rows, Hidden]>::from_vec(
            (0..2 * count).map(|index| bf16::from_f32(case(index % count)[index / count])),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let residual = HostTensor::<bf16, m![Shard, Rows, Hidden]>::from_vec(
            (0..2 * count).map(|index| bf16::from_f32(case(index % count)[2])),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let expected: Vec<_> = (0..count)
            .map(|index| {
                let [low, high, residual] = case(index);
                bf16::from_f32(bf16::from_f32(low + high).to_f32() + residual).to_bits()
            })
            .collect();
        // When
        let actual = launch(finish, (&mut device, &partial, &residual))
            .await
            .unwrap()
            .to_host::<m![Shard, Rows, Hidden]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        // Then
        for (rank, values) in actual.chunks_exact(count).enumerate() {
            for (index, (value, expected)) in values.iter().zip(&expected).enumerate() {
                assert_eq!(value.to_bits(), *expected, "rank={rank} index={index}");
            }
        }
    }
}
