//! Startup sizing from real allocations of the same workspace types that calls own.

use furiosa_opt_std::prelude::Device;
use std::io;

/// Per-chip storage beside the resident weight tree and already primed entries/contexts.
#[derive(Clone, Copy, Debug)]
pub(super) struct Budget {
    pub(super) condition: usize,
    pub(super) denoising: usize,
    pub(super) video: usize,
    pub(super) audio: usize,
}

impl Budget {
    /// One component runs at a time; the completed request memo can coexist with any of them.
    pub(super) fn call(self) -> usize {
        self.condition
            .max(self.denoising)
            .max(self.video)
            .max(self.audio)
    }
}

/// Hold the complete allocation set while reading its actual per-chip cost, then release it.
/// Entries and persistent component contexts must already be primed. No kernel reads these
/// sizing buffers, so their contents need no initialization or synthetic numerical execution.
pub(super) fn measure<T>(
    device: &mut Device,
    allocate: impl FnOnce(&mut Device) -> io::Result<T>,
) -> io::Result<usize> {
    let before = device.memory().usage().available;
    let held = allocate(device)?;
    let after = device.memory().usage().available;
    drop(held);
    if device.memory().usage().available != before {
        return Err(io::Error::other(
            "H3 workspace sizing did not restore its allocation baseline",
        ));
    }
    before
        .checked_sub(after)
        .ok_or_else(|| io::Error::other("H3 workspace allocation accounting changed"))
}
