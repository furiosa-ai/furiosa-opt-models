//! Packs selected decoder layers into a host slot in the copy groups' order.

use std::ops::Range;

/// Host layout shared by pools with the same byte packing.
pub(super) struct SlotLayout {
    /// Number of bytes in each slot.
    pub(super) slot_bytes: usize,
    /// Host ranges indexed by the copier's packed layer order.
    pub(super) layer_ranges: Vec<Range<usize>>,
}

/// Byte placement of one selected model layer in a packed host slot.
pub(super) struct Layer {
    /// Stable model layer ID, indexing the decoder's layer buffers.
    pub(super) id: usize,
    /// Bytes per chip for one block of the layer.
    pub(super) page_bytes: usize,
    /// Byte range occupied by all chips of the layer, in chip order.
    pub(super) host_range: Range<usize>,
}

/// Derive one packed host slot from the decoder's validated formats and the requested copy groups.
pub(super) fn pack(
    mut page_sizes: Vec<usize>,
    groups: &[Vec<usize>],
    chips: usize,
) -> Result<(Vec<Vec<Layer>>, usize), &'static str> {
    if groups.is_empty() || groups.iter().any(Vec::is_empty) || chips == 0 {
        return Err("cache copies require nonempty layer groups and a device with chips");
    }
    let mut slot_bytes: usize = 0;
    let mut packed = Vec::with_capacity(groups.len());
    for group in groups {
        let mut layers = Vec::with_capacity(group.len());
        for &id in group {
            let page_bytes = page_sizes
                .get_mut(id)
                .map(std::mem::take)
                .ok_or("cache copy group names an unknown layer")?;
            if page_bytes == 0 {
                return Err("cache copy groups must not repeat a layer");
            }
            if !page_bytes.is_multiple_of(8) {
                return Err("cache layers need eight-byte-aligned pages");
            }
            let end = page_bytes
                .checked_mul(chips)
                .and_then(|bytes| slot_bytes.checked_add(bytes))
                .ok_or("host cache slot size overflows")?;
            layers.push(Layer {
                id,
                page_bytes,
                host_range: slot_bytes..end,
            });
            slot_bytes = end;
        }
        packed.push(layers);
    }
    Ok((packed, slot_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_selected_layers_in_route_order() {
        let page_sizes = vec![8, 16, 8, 16];
        let (groups, slot_bytes) = pack(page_sizes.clone(), &[vec![3, 0], vec![1]], 2).unwrap();
        let positions = groups
            .iter()
            .flatten()
            .map(|layer| (layer.id, layer.page_bytes, layer.host_range.clone()))
            .collect::<Vec<_>>();
        assert_eq!(positions, [(3, 16, 0..32), (0, 8, 32..48), (1, 16, 48..80)]);
        assert_eq!(slot_bytes, 80);
        assert!(pack(page_sizes.clone(), &[vec![0], vec![0]], 2).is_err());
        assert!(pack(page_sizes.clone(), &[vec![4]], 2).is_err());
        assert!(pack(page_sizes, &[vec![0], vec![]], 2).is_err());
    }
}
