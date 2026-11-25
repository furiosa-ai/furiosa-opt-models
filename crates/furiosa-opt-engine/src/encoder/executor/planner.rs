use std::io;
use std::ops::Range;

/// A consecutive part of one input item's rows.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct ItemRows {
    pub(super) item: usize,
    pub(super) rows: Range<usize>,
}

/// One launch's consecutive item rows and their total row count.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct PackedLaunch {
    pub(super) items: Vec<ItemRows>,
    pub(super) rows: usize,
}

/// Every position is within the encoder's table, refusing the first one past it.
pub(super) fn validate_positions<'a>(
    positions: impl IntoIterator<Item = &'a [i32; 2]>,
    limit: usize,
) -> io::Result<()> {
    if positions
        .into_iter()
        .any(|xy| xy.iter().any(|&at| at as usize >= limit))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("a patch position reaches past the {limit} positions"),
        ));
    }
    Ok(())
}

/// The launches of a call of items of `rows` rows each, in order: consecutive rows packed into
/// launches of at most `long`, keeping items that fit and splitting oversized items.
pub(super) fn launches(rows: impl IntoIterator<Item = usize>, long: usize) -> Vec<PackedLaunch> {
    // Builder validates these compile-time row counts once when registering the modality.
    let mut launches = Vec::<PackedLaunch>::new();
    for (item, count) in rows.into_iter().enumerate() {
        if count == 0 {
            match launches.last_mut() {
                Some(launch) => launch.items.push(ItemRows { item, rows: 0..0 }),
                None => launches.push(PackedLaunch {
                    items: vec![ItemRows { item, rows: 0..0 }],
                    rows: 0,
                }),
            }
            continue;
        }

        if count <= long
            && launches
                .last()
                .is_some_and(|launch| count > long - launch.rows)
        {
            launches.push(PackedLaunch {
                items: Vec::new(),
                rows: 0,
            });
        }

        let mut start = 0;
        while start < count {
            if launches.last().map_or(true, |launch| launch.rows == long) {
                launches.push(PackedLaunch {
                    items: Vec::new(),
                    rows: 0,
                });
            }
            let launch = launches.last_mut().expect("a launch was just added");
            let rows = (count - start).min(long - launch.rows);
            let end = start + rows;
            launch.items.push(ItemRows {
                item,
                rows: start..end,
            });
            launch.rows += end - start;
            start = end;
        }
    }
    launches
}

/// Whether a packed launch fits the encoder's short row count.
pub(super) fn fits_short(rows: usize, short: usize) -> bool {
    rows <= short
}

#[cfg(test)]
mod tests {
    use super::{ItemRows, PackedLaunch, fits_short, launches, validate_positions};
    use std::io;

    #[test]
    fn packs_ordered_items_at_short_and_long_boundaries() {
        // Two images fit one short launch; next image (video frames) needs long launch.
        let packed = launches([256, 256, 2304], 2304);

        assert_eq!(
            packed,
            [
                PackedLaunch {
                    items: vec![
                        ItemRows {
                            item: 0,
                            rows: 0..256,
                        },
                        ItemRows {
                            item: 1,
                            rows: 0..256,
                        },
                    ],
                    rows: 512,
                },
                PackedLaunch {
                    items: vec![ItemRows {
                        item: 2,
                        rows: 0..2304,
                    }],
                    rows: 2304,
                }
            ]
        );
        assert!(fits_short(packed[0].rows, 512));
        assert!(!fits_short(packed[1].rows, 512));
    }

    #[test]
    fn starts_a_new_launch_after_filling_long_capacity() {
        let packed = launches([1800, 504, 1], 2304);

        assert_eq!(
            packed,
            [
                PackedLaunch {
                    items: vec![
                        ItemRows {
                            item: 0,
                            rows: 0..1800,
                        },
                        ItemRows {
                            item: 1,
                            rows: 0..504,
                        },
                    ],
                    rows: 2304,
                },
                PackedLaunch {
                    items: vec![ItemRows {
                        item: 2,
                        rows: 0..1,
                    }],
                    rows: 1,
                }
            ]
        );
        assert!(!fits_short(packed[0].rows, 512));
        assert!(fits_short(packed[1].rows, 512));
    }

    #[test]
    fn splits_oversized_item_and_keeps_order() {
        let packed = launches([256, 2305], 2304);

        assert_eq!(
            packed,
            [
                PackedLaunch {
                    items: vec![
                        ItemRows {
                            item: 0,
                            rows: 0..256,
                        },
                        ItemRows {
                            item: 1,
                            rows: 0..2048,
                        },
                    ],
                    rows: 2304,
                },
                PackedLaunch {
                    items: vec![ItemRows {
                        item: 1,
                        rows: 2048..2305,
                    }],
                    rows: 257,
                },
            ]
        );
    }

    #[test]
    fn preserves_zero_row_items_in_launch_order() {
        let packed = launches([0, 5], 16);

        assert_eq!(
            packed,
            [PackedLaunch {
                items: vec![
                    ItemRows {
                        item: 0,
                        rows: 0..0,
                    },
                    ItemRows {
                        item: 1,
                        rows: 0..5,
                    },
                ],
                rows: 5,
            }]
        );

        let packed = launches([16, 0], 16);

        assert_eq!(
            packed,
            [PackedLaunch {
                items: vec![
                    ItemRows {
                        item: 0,
                        rows: 0..16,
                    },
                    ItemRows {
                        item: 1,
                        rows: 0..0,
                    },
                ],
                rows: 16,
            }]
        );
    }

    #[test]
    fn validates_position_table_boundary() {
        validate_positions([&[1119, 1119]], 1120).unwrap();

        let error = validate_positions([&[1120, 0]], 1120).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "a patch position reaches past the 1120 positions"
        );
    }
}
