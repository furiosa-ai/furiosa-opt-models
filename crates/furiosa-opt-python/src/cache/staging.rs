//! Staging layout for coalesced cache copies.

use std::io;
use std::ops::Range;

use super::queue::TransferPage;

#[derive(Clone, Copy)]
/// Per-layer byte size and offset in a host cache slot.
pub(super) struct LayerLayout {
    /// Bytes in one layer page on one chip.
    pub(super) page_bytes: usize,
    /// Byte offset of this layer's page in the host slot.
    pub(super) host_offset: usize,
}

#[derive(Debug, Eq, PartialEq)]
/// One layer page's source slot and destination range in a chip's stage buffer.
pub(super) struct Item {
    /// Index of the source page in the transfer batch.
    pub(super) page: usize,
    /// Index of the layer in the copier.
    pub(super) layer: usize,
    /// Bytes copied for this page on one chip.
    pub(super) page_bytes: usize,
    /// Offset of the page in its host slot.
    pub(super) host_offset: usize,
    /// Range reserved for this page in each chip's stage buffer.
    pub(super) stage: Range<usize>,
}

impl Item {
    /// Copy this chip's bytes from a host slot into its stage range.
    pub(super) fn place(&self, chip: usize, host: &[u8], stage: &mut [u8]) -> io::Result<()> {
        let host_range = self.host_range(chip)?;
        let source = host
            .get(host_range)
            .ok_or_else(|| io::Error::other("cache staging host slot is too small"))?;
        let destination = stage
            .get_mut(self.stage.clone())
            .ok_or_else(|| io::Error::other("cache staging buffer is too small"))?;
        destination.copy_from_slice(source);
        Ok(())
    }

    /// Copy this chip's stage range back into its host slot.
    pub(super) fn take(&self, chip: usize, stage: &[u8], host: &mut [u8]) -> io::Result<()> {
        let source = stage
            .get(self.stage.clone())
            .ok_or_else(|| io::Error::other("cache staging buffer is too small"))?;
        let destination = host
            .get_mut(self.host_range(chip)?)
            .ok_or_else(|| io::Error::other("cache staging host slot is too small"))?;
        destination.copy_from_slice(source);
        Ok(())
    }

    /// Return this chip's byte range within the concatenated host page.
    fn host_range(&self, chip: usize) -> io::Result<Range<usize>> {
        let start = self
            .host_offset
            .checked_add(
                chip.checked_mul(self.page_bytes)
                    .ok_or_else(|| io::Error::other("cache staging host offset overflows"))?,
            )
            .ok_or_else(|| io::Error::other("cache staging host offset overflows"))?;
        let end = start
            .checked_add(self.page_bytes)
            .ok_or_else(|| io::Error::other("cache staging host end overflows"))?;
        Ok(start..end)
    }
}

#[derive(Debug, Eq, PartialEq)]
/// Adjacent bytes copied between one layer's stage and device buffers.
pub(super) struct Segment {
    /// Index of the layer in the copier.
    pub(super) layer: usize,
    /// Contiguous range in each chip's stage buffer.
    pub(super) stage: Range<usize>,
    /// Contiguous byte range in the layer's device buffer.
    pub(super) device: Range<usize>,
}

#[derive(Debug)]
/// Coalesced copy layout shared by every chip in the device topology.
pub(super) struct Layout {
    /// Individual pages used to gather from or scatter to host slots.
    pub(super) items: Vec<Item>,
    /// Contiguous layer ranges submitted to PDMA.
    pub(super) segments: Vec<Segment>,
    /// Total bytes required for one chip's stage buffer.
    pub(super) bytes: usize,
}

impl Layout {
    /// Build per-chip stage ranges and merge device-adjacent pages within each layer.
    pub(super) fn new(
        pages: &[TransferPage],
        groups: &[Vec<usize>],
        layers: &[LayerLayout],
    ) -> io::Result<Self> {
        let mut routes = Vec::new();
        for (page, transfer) in pages.iter().enumerate() {
            for &(group, block) in &transfer.routes {
                let group_layers = groups
                    .get(group)
                    .ok_or_else(|| io::Error::other("cache staging names an unknown group"))?;
                for &layer in group_layers {
                    let layout = layers
                        .get(layer)
                        .ok_or_else(|| io::Error::other("cache staging names an unknown layer"))?;
                    routes.push((layout.host_offset, layer, block, page, layout.page_bytes));
                }
            }
        }
        routes.sort_unstable_by_key(|&(offset, layer, block, _, _)| (offset, block, layer));
        if routes
            .windows(2)
            .any(|pair| pair[0].1 == pair[1].1 && pair[0].2 == pair[1].2)
        {
            return Err(io::Error::other(
                "cache staging routes a device layer page more than once",
            ));
        }

        let mut bytes = 0usize;
        let mut items = Vec::with_capacity(routes.len());
        let mut segments: Vec<Segment> = Vec::new();
        for (_, layer, block, page, page_bytes) in routes {
            let end = bytes
                .checked_add(page_bytes)
                .ok_or_else(|| io::Error::other("cache staging size overflows"))?;
            let device_start = usize::try_from(block)
                .map_err(io::Error::other)?
                .checked_mul(page_bytes)
                .ok_or_else(|| io::Error::other("cache staging device offset overflows"))?;
            let device_end = device_start
                .checked_add(page_bytes)
                .ok_or_else(|| io::Error::other("cache staging device end overflows"))?;
            let item = Item {
                page,
                layer,
                page_bytes,
                host_offset: layers[layer].host_offset,
                stage: bytes..end,
            };
            match segments.last_mut() {
                Some(last)
                    if last.layer == layer
                        && last.stage.end == item.stage.start
                        && last.device.end == device_start =>
                {
                    last.stage.end = end;
                    last.device.end = device_end;
                }
                _ => segments.push(Segment {
                    layer,
                    stage: item.stage.clone(),
                    device: device_start..device_end,
                }),
            }
            bytes = end;
            items.push(item);
        }
        Ok(Self {
            items,
            segments,
            bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Routes host pages into stage buffers and merges adjacent device pages.
    fn lays_out_layer_pages_and_merges_adjacent_device_ranges() {
        let pages = [
            TransferPage {
                slot: 3,
                routes: vec![(0, 4), (1, 10)],
            },
            TransferPage {
                slot: 5,
                routes: vec![(0, 5)],
            },
        ];
        let groups = [vec![0], vec![1]];
        let layers = [
            LayerLayout {
                page_bytes: 4,
                host_offset: 0,
            },
            LayerLayout {
                page_bytes: 8,
                host_offset: 16,
            },
        ];

        let layout = Layout::new(&pages, &groups, &layers).unwrap();

        assert_eq!(layout.bytes, 16);
        assert_eq!(
            layout.items,
            [
                Item {
                    page: 0,
                    layer: 0,
                    page_bytes: 4,
                    host_offset: 0,
                    stage: 0..4,
                },
                Item {
                    page: 1,
                    layer: 0,
                    page_bytes: 4,
                    host_offset: 0,
                    stage: 4..8,
                },
                Item {
                    page: 0,
                    layer: 1,
                    page_bytes: 8,
                    host_offset: 16,
                    stage: 8..16,
                },
            ]
        );
        assert_eq!(
            layout.segments,
            [
                Segment {
                    layer: 0,
                    stage: 0..8,
                    device: 16..24,
                },
                Segment {
                    layer: 1,
                    stage: 8..16,
                    device: 80..88,
                },
            ]
        );
    }

    #[test]
    /// Rejects a repeated destination page in the transfer routes.
    fn rejects_duplicate_device_layer_pages() {
        let pages = [
            TransferPage {
                slot: 0,
                routes: vec![(0, 7)],
            },
            TransferPage {
                slot: 1,
                routes: vec![(0, 7)],
            },
        ];
        let error = Layout::new(
            &pages,
            &[vec![0]],
            &[LayerLayout {
                page_bytes: 8,
                host_offset: 0,
            }],
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "cache staging routes a device layer page more than once"
        );
    }

    #[test]
    /// Gathers and restores each chip's slice of the host pages.
    fn places_and_takes_each_chip_layer_page() {
        let pages = [
            TransferPage {
                slot: 0,
                routes: vec![(0, 3)],
            },
            TransferPage {
                slot: 1,
                routes: vec![(0, 4)],
            },
        ];
        let layout = Layout::new(
            &pages,
            &[vec![0, 1]],
            &[
                LayerLayout {
                    page_bytes: 2,
                    host_offset: 0,
                },
                LayerLayout {
                    page_bytes: 2,
                    host_offset: 4,
                },
            ],
        )
        .unwrap();
        let slots = [
            [10, 11, 12, 13, 20, 21, 22, 23],
            [30, 31, 32, 33, 40, 41, 42, 43],
        ];
        let mut stages = [vec![0; layout.bytes], vec![0; layout.bytes]];

        for item in &layout.items {
            for (chip, stage) in stages.iter_mut().enumerate() {
                item.place(chip, &slots[item.page], stage).unwrap();
            }
        }

        assert_eq!(
            stages,
            [
                vec![10, 11, 30, 31, 20, 21, 40, 41],
                vec![12, 13, 32, 33, 22, 23, 42, 43]
            ]
        );

        let mut restored = [[0; 8]; 2];
        for item in &layout.items {
            for (chip, stage) in stages.iter().enumerate() {
                item.take(chip, stage, &mut restored[item.page]).unwrap();
            }
        }
        assert_eq!(restored, slots);
    }

    #[test]
    /// Orders staging pages by device block while preserving disjoint ranges.
    fn sorts_host_pages_by_device_block_and_keeps_gaps_separate() {
        let pages = [
            TransferPage {
                slot: 0,
                routes: vec![(0, 8)],
            },
            TransferPage {
                slot: 1,
                routes: vec![(0, 4)],
            },
            TransferPage {
                slot: 2,
                routes: vec![(0, 5)],
            },
            TransferPage {
                slot: 3,
                routes: vec![(0, 7)],
            },
        ];
        let layout = Layout::new(
            &pages,
            &[vec![0]],
            &[LayerLayout {
                page_bytes: 4,
                host_offset: 0,
            }],
        )
        .unwrap();

        assert_eq!(
            layout
                .items
                .iter()
                .map(|item| item.page)
                .collect::<Vec<_>>(),
            [1, 2, 3, 0]
        );
        assert_eq!(
            layout.segments,
            [
                Segment {
                    layer: 0,
                    stage: 0..8,
                    device: 16..24,
                },
                Segment {
                    layer: 0,
                    stage: 8..16,
                    device: 28..36,
                },
            ]
        );
    }
}
