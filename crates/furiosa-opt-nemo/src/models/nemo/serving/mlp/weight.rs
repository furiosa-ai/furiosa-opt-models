use furiosa_opt_std::prelude::*;

use super::{Branch, Hidden, Intermediate};
use crate::models::nemo::serving::{Chips, stack, stack::Depth};

type Cluster = m![Intermediate / 3584];
type Slice = m![Intermediate % 3584 / 56, Hidden % 2560 / 640];
/// A half as its DMA lands it from `Stored`: 56 outputs by 640 features a slice in one 35-KiB
/// run, its features in 80-wide groups.
type Raw = DmTensor<
    f8e4m3,
    Chips,
    Cluster,
    Slice,
    m![
        Hidden % 640 / 80,
        Intermediate % 56 / 8,
        Hidden % 80 / 8,
        Intermediate % 8,
        Hidden % 8
    ],
>;
/// A gate or up shard in its only stored form, which both routes land in long runs: a decode
/// slice's 56 outputs by 640 features one 35-KiB run, a prompt slice's 448 by 80 eight 4,480-B
/// runs.
pub type Stored = m![
    Intermediate % 3584 / 1792,
    Intermediate / 3584,
    Intermediate % 1792 / 448,
    Intermediate % 448 / 56,
    Hidden / 80,
    Intermediate % 56 / 8,
    Hidden % 80 / 8,
    Intermediate % 8,
    Hidden % 8
];

/// The position of gate / up value (`intermediate` of a chip's 7,168, `hidden`) in `Stored`.
pub fn index(intermediate: usize, hidden: usize) -> usize {
    let (i, h) = (intermediate, hidden);
    [
        (2, i % 3584 / 1792),
        (2, i / 3584),
        (4, i % 1792 / 448),
        (8, i % 448 / 56),
        (64, h / 80),
        (7, i % 56 / 8),
        (10, h % 80 / 8),
        (8, i % 8),
        (8, h % 8),
    ]
    .iter()
    .fold(0, |index, &(size, digit)| index * size + digit)
}
/// One hidden half of the gate (branch 0) and up (branch 1) weights in the contraction's layout,
/// so one command contracts both against a register.
pub type GateUp =
    DmTensor<f8e4m3, Chips, Cluster, Slice, m![Branch, Intermediate % 56, Hidden % 640]>;

/// A gate or up projection of one layer, read from its stacked slot.
pub struct Weight<'a>(pub &'a stack::Stacked<'a, f8e4m3, Stored>, pub usize);

/// One hidden half of a projection weight as its DMA lands it, before the staging pass.
pub struct Landed(Raw);

/// The gate's hidden halves landed ahead of the feed-forward: half 0 always, half 1 `count` times
/// (0 or 1), or never with `count` `None`; the feed-forward lands half 1 itself when it is not.
pub struct First {
    pub low: Landed,
    pub high: Landed,
    pub count: Option<usize>,
}

impl First {
    pub fn new(count: Option<usize>) -> Self {
        Self {
            low: Landed::new(),
            high: Landed::new(),
            count,
        }
    }
}

impl Weight<'_> {
    pub fn land_first(&self, device: &mut Device, first: &mut First) {
        self.land_into(device, (0, &mut first.low));
        if let Some(count) = first.count {
            for _ in 0..count {
                self.land_into(device, (1, &mut first.high));
            }
        }
    }

    /// `land_into` unless `count` (`First::count`) says the half landed ahead.
    pub fn land_unless(
        &self,
        device: &mut Device,
        (half, landed): (usize, &mut Landed),
        count: Option<usize>,
    ) {
        match count {
            Some(count) => {
                for _ in 0..1 - count {
                    self.land_into(device, (half, &mut *landed));
                }
            }
            None => self.land_into(device, (half, landed)),
        }
    }

    pub fn land(&self, device: &mut Device, half: usize) -> Landed {
        let mut landed = Landed(Raw::new());
        self.land_into(device, (half, &mut landed));
        landed
    }

    /// `land` into a weight allocated before the transfer is issued.
    pub fn land_into(&self, device: &mut Device, (half, landed): (usize, &mut Landed)) {
        self.0
            .at(self.1)
            .tile::<m![Hidden / 2560], 1, m![
                Depth = 1 # Depth,
                Intermediate % 3584 / 1792,
                Intermediate / 3584,
                Intermediate % 1792 / 448,
                Intermediate % 448 / 56,
                Hidden / 2560 = 1 # 2,
                Hidden % 2560 / 80,
                Intermediate % 56 / 8,
                Hidden % 80 / 8,
                Intermediate % 8,
                Hidden % 8
            ]>(half)
            .to_dm_view(&mut device.tdma, landed.0.view_mut());
    }
}

impl Landed {
    /// A weight not yet landed, for `Weight::land_into`.
    pub fn new() -> Self {
        Self(Raw::new())
    }

    /// The landed half into branch `branch` of `gate_up` in the contraction's layout: each output's
    /// 640 features contiguous, one fetch packet an output.
    pub fn stage_into(&self, device: &mut Device, (gate_up, branch): (&mut GateUp, usize)) {
        device
            .main
            .begin(self.0.view())
            .fetch::<m![Intermediate % 56], m![Hidden % 640]>()
            .collect::<m![Intermediate % 56, Hidden % 640 / 32], m![Hidden % 32]>()
            .commit_trim::<m![Hidden % 32]>()
            .commit_view(gate_up.view_mut().tile::<m![Branch], 1, m![
                Branch = 1 #{!} 2,
                Intermediate % 56,
                Hidden % 640
            ]>(branch));
    }
}
