// The landing of a geometry whose slice holds four 16-token pages: the span's keys fill every
// slice's landing buffer.
type RawElement = m![Keys % { SLICE_KEYS } / 16, Page, Heads % 2, Kv, Head];
/// A slice's keys of its cluster's two heads, each head's key and value adjacent as the pool
/// stores them; one gather entry lands in every copy.
pub type Landed = DmTensor<f8e4m3, Chip, Cluster, m![{ Copies }, Keys / { SLICE_KEYS }], RawElement>;

/// Gathers into `landed` the keys and values of the span whose pages `pages` lists.
pub fn land(
    context: &mut Device,
    (pool, entry): (&Pool, usize),
    pages: &Pages,
    landed: &mut Landed,
) {
    // SAFETY: the host plan bounds every page by the layer's pages, and each offset starts one.
    unsafe {
        pool.gather::<m![1], m![Page, Heads, Kv, Head]>()
            .entry(entry)
            .by_byte_offsets(pages)
            .unchecked()
    }
    .to_dm_view(&mut context.tdma, landed.view_mut());
}

/// `land` into the landing buffer as the narrow geometry types it.
pub fn land_shared(
    context: &mut Device,
    (pool, entry): (&Pool, usize),
    pages: &Pages,
    landed: &mut super::narrow::Landed,
) {
    // SAFETY: every geometry's landing buffer holds 32 KiB in each of a cluster's 256 slices.
    let view = unsafe {
        landed
            .view_mut()
            .reshape::<Chip, Cluster, m![{ Copies }, Keys / { SLICE_KEYS }], RawElement>()
    };
    // SAFETY: the host plan bounds every page by the layer's pages, and each offset starts one.
    unsafe {
        pool.gather::<m![1], m![Page, Heads, Kv, Head]>()
            .entry(entry)
            .by_byte_offsets(pages)
            .unchecked()
    }
    .to_dm_view(&mut context.tdma, view);
}

/// Both heads' keys or values of every slice, in token order.
fn tokens(
    landed: &Landed,
    kv: usize,
) -> DmTensorView<'_, f8e4m3, Chip, Cluster, Slices, m![Keys % { SLICE_KEYS }, Heads % 2, Kv = 1 # 2, Head]> {
    // SAFETY: every copy of a slice's keys serves the query group that holds it, and a slice's
    // pages of sixteen tokens are its keys in order.
    unsafe {
        landed
            .view()
            .reshape::<Chip, Cluster, Slices, m![Keys % { SLICE_KEYS }, Heads % 2, Kv, Head]>()
    }
    .tile::<m![Kv], 1, m![Keys % { SLICE_KEYS }, Heads % 2, Kv = 1 # 2, Head]>(kv)
}
