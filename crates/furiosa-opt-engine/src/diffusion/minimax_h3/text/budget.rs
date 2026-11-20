//! Resident-prefix selection from measured per-chip allocations, matching the streamed repro.

use std::io;

/// A pipeline's encoder memory grant and other stages' maximum transient storage. Persistent
/// weights and compiled-entry allocations are already placed before free memory is measured.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub bytes: usize,
    pub beside: usize,
}

/// Resident prefix selected from measured per-chip allocation costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fit {
    pub resident: usize,
}

impl Fit {
    pub(super) fn measured(
        limits: Limits,
        available: usize,
        fixed: usize,
        layer: usize,
        request: usize,
        layers: usize,
    ) -> io::Result<Self> {
        let invalid = || {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid encoder allocation measurements",
            )
        };
        if layer == 0 || layers == 0 {
            return Err(invalid());
        }
        let encoding = layer
            .checked_mul(SLOTS)
            .and_then(|n| n.checked_add(request))
            .ok_or_else(invalid)?;
        let peak = encoding.max(limits.beside);
        let memory = available.min(limits.bytes);
        let spare = memory.checked_sub(fixed).and_then(|n| n.checked_sub(peak)).ok_or_else(|| {
            io::Error::new(io::ErrorKind::OutOfMemory,
                format!("{memory} bytes/chip do not hold {fixed} fixed encoder bytes beside {SLOTS} streamed layers of {layer} bytes and the request ({request} bytes), or other pipeline requests ({} bytes)", limits.beside))
        })?;
        let resident = layers.min(spare / layer);
        Ok(Self { resident })
    }
}

const SLOTS: usize = 3;

#[cfg(test)]
mod tests {
    use super::*;

    /// Original repro budget boundaries: exact fit, one byte short, the other stage dominating,
    /// and insufficient space for the streaming window. Values are bytes per chip.
    #[test]
    fn keeps_the_prefix_that_fits_beside_either_stage() {
        let (fixed, layer, request) = (30, 10, 19);
        for (memory, beside, resident) in [(109, 0, 3), (108, 0, 2), (119, 69, 2), (118, 69, 1)] {
            let fit = Fit::measured(
                Limits {
                    bytes: memory,
                    beside,
                },
                memory,
                fixed,
                layer,
                request,
                6,
            )
            .unwrap();
            assert_eq!(fit.resident, resident);
        }
        let fit = Fit::measured(
            Limits {
                bytes: 109,
                beside: 0,
            },
            108,
            fixed,
            layer,
            request,
            6,
        )
        .unwrap();
        assert_eq!(fit.resident, 2);
        let error = Fit::measured(
            Limits {
                bytes: 78,
                beside: 0,
            },
            78,
            fixed,
            layer,
            request,
            6,
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::OutOfMemory);
    }
}
