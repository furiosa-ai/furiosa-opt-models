//! Accepted row launches keep their input and request buffers until every reader ends.

use crate::residency::Completion;
use furiosa_opt_std::prelude::HbmBuffer;
use std::{io, sync::Arc};

/// One layer's inputs and accepted outputs. On cancellation or submission/execution failure,
/// Drop drains the accepted work before Rust releases any held input, cache or metadata owner.
pub(super) struct Flight<T> {
    held: T,
    rows: Vec<HbmBuffer>,
    work: Vec<Arc<Completion>>,
}

impl<T> Flight<T> {
    pub(super) fn new(held: T, rows: usize) -> Self {
        Self {
            held,
            rows: Vec::with_capacity(rows),
            work: Vec::with_capacity(rows),
        }
    }

    pub(super) fn held_mut(&mut self) -> &mut T {
        &mut self.held
    }

    /// Both vectors were reserved before submission. Call immediately after acceptance.
    pub(super) fn accepted(&mut self, row: HbmBuffer, work: Arc<Completion>) {
        self.rows.push(row);
        self.submitted(work);
    }

    /// A queued launch into workspace storage already retained in `held`. Reserve the maximum
    /// launch count before submission, as for row outputs, and record immediately on acceptance.
    pub(super) fn submitted(&mut self, work: Arc<Completion>) {
        self.work.push(work);
    }

    /// A layer's completed rows can replace the previous input set without retaining a chain
    /// of all prior layers. Cache storage may be reused only after these completions finish.
    pub(super) async fn finish(mut self) -> io::Result<Vec<HbmBuffer>> {
        for work in &self.work {
            work.ended().await.map_err(io::Error::other)?;
        }
        Ok(std::mem::take(&mut self.rows))
    }
}

impl<T> Drop for Flight<T> {
    fn drop(&mut self) {
        for work in &self.work {
            // Finish every accepted reader even when another reader failed. The normal path
            // already reports failures; cancellation cannot return one, but must drain safely.
            let _ = work.wait();
        }
    }
}
