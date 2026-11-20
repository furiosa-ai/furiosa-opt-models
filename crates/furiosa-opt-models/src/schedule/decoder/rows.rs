//! Transfer only the live residual rows of the FP8 layer loop.
use super::{Capacity, Chip, Decode, Geometry, Rows, Width};
use crate::ops::normalization::span::{Owned, Span};
use crate::ops::normalization::{Offset, Part};
use crate::tensor::live::Live;
use furiosa_opt_std::prelude::*;

type Cap<Z> = Symbol<Capacity<Z>>;
type Half = m![Span % (Span / 2)];
type Cluster = m![Span / (Span / 2)];

/// Each cluster owns a contiguous half-span. Decompose a partial half into aligned
/// power-of-two windows, so a one-token decode moves one hidden row.
trait Transfer {
    fn window<const OWNER: usize, const N: usize>(&mut self, device: &mut Device, at: usize);

    fn half<const OWNER: usize>(&mut self, device: &mut Device, count: usize) {
        if count == 256 {
            self.window::<OWNER, 256>(device, 0);
        } else {
            self.part::<OWNER, 128>(device, count);
            self.part::<OWNER, 64>(device, count);
            self.part::<OWNER, 32>(device, count);
            self.part::<OWNER, 16>(device, count);
            self.part::<OWNER, 8>(device, count);
            self.part::<OWNER, 4>(device, count);
            self.part::<OWNER, 2>(device, count);
            self.part::<OWNER, 1>(device, count);
        }
    }

    fn part<const OWNER: usize, const N: usize>(&mut self, device: &mut Device, count: usize) {
        if count / N % 2 == 1 {
            self.window::<OWNER, N>(device, count / (2 * N) * (2 * N));
        }
    }

    fn prefix(&mut self, device: &mut Device, count: usize) {
        self.half::<0>(device, count.min(256));
        if count > 256 {
            self.half::<1>(device, count - 256);
        }
    }
}

struct Load<'a, Z: Geometry + Decode>(&'a Rows<Z>, &'a mut Owned<Width<Z>, Span>, usize);
struct Store<'a, Z: Geometry + Decode>(&'a Owned<Width<Z>, Span>, &'a mut Rows<Z>, usize);

impl<Z: Geometry + Decode> Transfer for Load<'_, Z> {
    fn window<const OWNER: usize, const N: usize>(&mut self, device: &mut Device, at: usize) {
        let start = self.2 + OWNER * Half::SIZE + at;
        let src = self.0.view().tile::<m![{Cap<Z>}], N,
            m![{Part<Width<Z>>}, {Cap<Z>} = {N} # ({Cap<Z>} / 1), {Offset<Width<Z>>}]>(start);
        // SAFETY: split the selected rows' padding into the cluster and local-row digits.
        let src = unsafe {
            src.reshape::<Chip, m![{Part<Width<Z>>},
            {Cluster} = 1 # ({Cap<Z>} / 256), {Half} = {N} # 256, {Offset<Width<Z>>}]>(
            )
        };
        src.to_dm_view(
            &mut device.tdma,
            self.1
                .view_mut()
                .cluster_tile::<Cluster, 1, m![{Cluster} = 1 #{!} 2]>(OWNER)
                .tile::<Half, N, m![{Half} = {N} #{!} 256,{Offset<Width<Z>>}]>(at),
        );
    }
}

impl<Z: Geometry + Decode> Transfer for Store<'_, Z> {
    fn window<const OWNER: usize, const N: usize>(&mut self, device: &mut Device, at: usize) {
        let start = self.2 + OWNER * Half::SIZE + at;
        let dst = self.1.view_mut().tile::<m![{Cap<Z>}], N,
            m![{Part<Width<Z>>}, {Cap<Z>} = {N} #{!} ({Cap<Z>} / 1), {Offset<Width<Z>>}]>(start);
        // SAFETY: same strided window of N rows, at the same feature and row offsets.
        let dst = unsafe {
            dst.reshape::<Chip, m![{Part<Width<Z>>},
            {Cluster} = 1 #{!} ({Cap<Z>} / 256), {Half} = {N} #{!} 256, {Offset<Width<Z>>}]>(
            )
        };
        self.0
            .view()
            .cluster_tile::<Cluster, 1, m![{Cluster} = 1 # 2]>(OWNER)
            .tile::<Half, N, m![{Half} = {N} # 256,{Offset<Width<Z>>}]>(at)
            .to_hbm_view(&mut device.tdma, dst);
    }
}

pub(super) fn load<Z: Geometry + Decode>(
    device: &mut Device,
    buffer: &Rows<Z>,
    tile: Live<(), Capacity<Z>>,
) -> Owned<Width<Z>, Span> {
    use crate::modules::generic::schedule::Fill;
    if tile.rows() == Span::SIZE {
        return buffer.fill(device, tile);
    }
    let mut rows = Owned::new();
    Load::<Z>(buffer, &mut rows, tile.start()).prefix(device, tile.rows());
    rows
}

pub(super) fn store<Z: Geometry + Decode>(
    device: &mut Device,
    rows: &Owned<Width<Z>, Span>,
    buffer: &mut Rows<Z>,
    tile: Live<(), Capacity<Z>>,
) {
    use crate::modules::generic::schedule::Spill;
    if tile.rows() == Span::SIZE {
        buffer.spill(device, rows, tile);
    } else {
        Store::<Z>(rows, buffer, tile.start()).prefix(device, tile.rows());
    }
}
