//! A launch's ids fed on the device from the tokens the launch before selected, so a step whose
//! rows continue that launch needs no host round trip: the host names where each row's token is.
//!
//! - A step returns its selected tokens into the first half of the next input slot's [`Source`];
//!   the host writes its own token of every row it knows into the second half.
//! - [`patch`] is its own launch between two steps, made only when a row is fed: each row of the
//!   fed 128-row waves is gathered again from [`Source`], so the step reads its ids unchanged.

use super::*;

axes![Entries = 8192, Quad = 4];

/// Waves of 128 rows a launch can feed: zero through all of its rows.
type Waves = m![33];

/// A launch's candidate tokens, one per 16-byte entry as [`State::tokens`] holds them: the
/// previous launch's selected tokens from entry 0, then the host's token of row `r` at entry
/// `Capacity + r`.
pub type Source = HbmTensor<i32, Chips, m![Entries, Quad]>;

const _: () = assert!(Entries::SIZE == 2 * Capacity::SIZE && Selected::SIZE <= Capacity::SIZE);

/// The rows a launch feeds: how many waves of 128 rows from row 0, and each of their rows' token
/// as a byte offset into [`Source`].
#[derive(DeviceSend)]
pub struct Feed {
    pub waves: mlp::Count,
    pub entries: Offsets,
}

/// Overwrites the fed waves of `ids` with their rows' tokens from `source`.
#[device(chip = 2)]
pub fn patch(device: &mut Device, (source, feed): (&Source, &Feed), ids: &mut Offsets) {
    let waves = feed.waves.extent::<Waves>();
    for wave in 0..waves {
        let entries = feed
            .entries
            .view()
            .tile::<m![Capacity], 128, m![Capacity = 128 # Capacity]>(wave * 128);
        // SAFETY: the entry tile keeps its capacity stride and the wave's row order.
        let entries = unsafe { entries.reshape::<Chips, m![Tile # Capacity]>() };
        let mut indices: HbmTensor<i32, Chips, m![Tile]> = HbmTensor::new();
        entries.to_hbm_view(&mut device.tdma, indices.view_mut());
        // Whole 16-byte entries: a DMA moves at least eight bytes into data memory.
        let picked: DmTensor<i32, Chips, m![1 # 2], m![Tile / 2 # 256], m![Tile % 2, Quad]> =
            source
                .gather::<m![Entries], m![Quad]>()
                .by_byte_offsets(&indices)
                .to_dm(&mut device.tdma);
        let picked: HbmTensor<i32, Chips, m![Tile, Quad]> = picked.to_hbm(&mut device.tdma);
        let rows = ids
            .view_mut()
            .tile::<m![Capacity], 128, m![Capacity = 128 #{!} Capacity]>(wave * 128);
        // SAFETY: the 128-row tile keeps the capacity stride; only its row axis is renamed.
        let rows = unsafe { rows.reshape::<Chips, m![Tile #{!} Capacity]>() };
        picked
            .view()
            .tile::<m![Quad], 1, m![Tile, Quad = 1 # 4]>(0)
            .to_hbm_view(&mut device.tdma, rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_std::runtime::launch;

    /// Fed rows, reordered, take the previous launch's tokens; every other row keeps the host's
    /// token, inside the fed waves and past them: the ids host-fed tokens would have given.
    #[tokio::test(flavor = "current_thread")]
    async fn feeds_the_tokens_the_host_would_write() {
        for fed_rows in [1, 40, 130] {
            let mut device = Device::new(patch.topology()).unwrap();
            let rows = Capacity::SIZE;
            let previous: Vec<i32> = (0..rows as i32).map(|at| 50_000 + 7 * at).collect();
            let hosted: Vec<i32> = (0..rows as i32).map(|row| 3 * row + 1).collect();
            // Fed rows 0, 2, 4, ... continue the previous tokens in reverse order.
            let fed =
                |row: usize| (row % 2 == 0 && row / 2 < fed_rows).then(|| fed_rows - 1 - row / 2);
            let last = (0..rows).filter(|&row| fed(row).is_some()).max().unwrap();
            let entry = |at: usize| i32::try_from(at * 16).unwrap();
            let map: Vec<i32> = (0..rows)
                .map(|row| fed(row).map_or(entry(rows + row), entry))
                .collect();
            let expected: Vec<i32> = (0..rows)
                .map(|row| fed(row).map_or(hosted[row], |answer| previous[answer]))
                .collect();
            let padded = |tokens: &[i32]| {
                tokens
                    .iter()
                    .flat_map(|&token| [token, -2, -3, -4])
                    .collect::<Vec<_>>()
            };
            let both = |values: &[i32]| [values, values].concat();
            let source = HostTensor::<i32, m![{ Chips }, Entries, Quad]>::from_vec(both(
                &[padded(&previous), padded(&hosted)].concat(),
            ))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            let feed = Feed {
                waves: HostTensor::<i32, m![{ Chips }, 1]>::from_vec(both(&[i32::try_from(
                    last / 128 + 1,
                )
                .unwrap()]))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
                entries: HostTensor::<i32, m![{ Chips }, Capacity]>::from_vec(both(&map))
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            };
            let mut ids = HostTensor::<i32, m![{ Chips }, Capacity]>::from_vec(both(&hosted))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();

            launch(patch, (&mut device, (&source, &feed), &mut ids))
                .await
                .unwrap();

            let actual = ids
                .to_host::<m![{ Chips }, Capacity]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            assert_eq!(actual, both(&expected), "{fed_rows} fed rows");
        }
    }
}
