//! Calls and accepted launches reading one device copy of a parameter.
//!
//! The owning parameter serializes access and refuses admission when its state requires it;
//! this gate counts successful admissions and retains their work until eviction takes it.

use super::Completion;
use std::sync::Arc;

/// Calls holding a copy, its distinct running readers, and the first observed reader failure.
#[derive(Default)]
pub(super) struct Gate {
    calls: usize,
    readers: Vec<Arc<Completion>>,
    failure: Option<String>,
}

impl Gate {
    /// Records one successful admission by the parameter state machine.
    pub(super) fn admit(&mut self) {
        self.calls = self.calls.checked_add(1).expect("too many admitted calls");
    }

    /// Whether an admitted call still holds the copy, independently of its launches.
    pub(super) fn held(&self) -> bool {
        self.calls > 0
    }

    /// Releases one call and records its accepted launches; returns whether calls still hold it.
    /// An admitted call that submitted no work must also release its hold.
    pub(super) fn release(&mut self, launched: &[Arc<Completion>]) -> bool {
        self.calls = self
            .calls
            .checked_sub(1)
            .expect("a release requires an admitted call");
        self.register(launched);
        self.held()
    }

    /// Keeps each running launch once, including when several calls share its completion.
    fn register(&mut self, readers: &[Arc<Completion>]) {
        self.reap();
        for reader in readers {
            if self.readers.iter().any(|kept| Arc::ptr_eq(kept, reader)) {
                continue;
            }
            match reader.result() {
                None => self.readers.push(reader.clone()),
                Some(Ok(())) => {}
                Some(Err(error)) => {
                    self.failure.get_or_insert(error);
                }
            }
        }
    }

    /// Drops finished readers, retaining only the first failure observed until it is taken.
    pub(super) fn reap(&mut self) {
        let failure = &mut self.failure;
        self.readers.retain(|reader| match reader.result() {
            None => true,
            Some(Ok(())) => false,
            Some(Err(error)) => {
                failure.get_or_insert(error);
                false
            }
        });
    }

    /// Takes running readers and the first failure after admission is closed and calls release.
    /// The caller must retain the copy and wait for every returned reader before freeing it.
    pub(super) fn take(&mut self) -> (Vec<Arc<Completion>>, Option<String>) {
        assert!(
            !self.held(),
            "cannot take readers while calls hold the copy"
        );
        self.reap();
        (std::mem::take(&mut self.readers), self.failure.take())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::Poll;

    fn failed(error: &'static str) -> Arc<Completion> {
        Arc::new(Completion::new(std::future::ready(Err(
            furiosa_opt_std::Error::Invalid(error),
        ))))
    }

    /// The test controls completion synchronously; polling never depends on time or scheduling.
    fn controlled() -> (Arc<Completion>, Arc<AtomicBool>) {
        let ready = Arc::new(AtomicBool::new(false));
        let flag = ready.clone();
        let work = Completion::new(std::future::poll_fn(move |_| {
            if flag.load(Ordering::Acquire) {
                Poll::Ready(Ok(()))
            } else {
                Poll::Pending
            }
        }));
        (Arc::new(work), ready)
    }

    #[test]
    fn holds_until_every_admitted_call_releases() {
        let mut gate = Gate::default();
        let running = Arc::new(Completion::new(std::future::pending()));
        assert!(!gate.held());
        gate.admit();
        gate.admit();
        gate.admit();

        assert!(gate.release(std::slice::from_ref(&running)));
        // A cancelled call without accepted launches still releases its admission.
        assert!(gate.release(&[]));
        gate.reap();
        assert!(gate.held());
        assert!(!gate.release(std::slice::from_ref(&running)));
        assert!(
            !gate.held(),
            "running launches are distinct from admitted calls"
        );

        let (kept, failure) = gate.take();
        assert_eq!(kept.len(), 1);
        assert!(Arc::ptr_eq(&kept[0], &running));
        assert!(failure.is_none());
        assert!(
            gate.take().0.is_empty(),
            "readers transfer to the eviction once"
        );
    }

    #[test]
    fn keeps_only_running_readers() {
        let mut gate = Gate::default();
        let (running, ready) = controlled();
        gate.admit();
        gate.release(std::slice::from_ref(&running));

        for _ in 0..1000 {
            let ended = Arc::new(Completion::new(std::future::ready(Ok(()))));
            let weak = Arc::downgrade(&ended);
            gate.admit();
            gate.release(&[ended]);
            assert!(
                weak.upgrade().is_none(),
                "finished readers are not retained"
            );
        }

        let weak = Arc::downgrade(&running);
        drop(running);
        assert!(
            weak.upgrade().is_some(),
            "running work stays owned by the gate"
        );
        ready.store(true, Ordering::Release);
        gate.reap();
        assert!(
            weak.upgrade().is_none(),
            "a finished reader is released on reaping"
        );
        let (kept, failure) = gate.take();
        assert!(kept.is_empty());
        assert!(failure.is_none());
    }

    #[test]
    fn keeps_shared_running_completions_once() {
        let mut gate = Gate::default();
        let first = Arc::new(Completion::new(std::future::pending()));
        let second = Arc::new(Completion::new(std::future::pending()));
        for _ in 0..1000 {
            gate.admit();
            gate.release(&[first.clone(), second.clone(), first.clone()]);
        }

        let (kept, failure) = gate.take();
        assert_eq!(
            kept.len(),
            2,
            "retention is bounded by distinct running work"
        );
        assert!(Arc::ptr_eq(&kept[0], &first));
        assert!(Arc::ptr_eq(&kept[1], &second));
        assert!(failure.is_none());
    }

    #[test]
    fn keeps_the_first_failure_until_taken() {
        let mut gate = Gate::default();
        let first = failed("the first launch failed");
        let first_error = first.result().unwrap().unwrap_err();
        let weak = Arc::downgrade(&first);
        gate.admit();
        gate.release(&[first]);
        assert!(weak.upgrade().is_none(), "only the failure string remains");
        for _ in 0..1000 {
            gate.admit();
            gate.release(&[
                failed("a later launch failed"),
                Arc::new(Completion::new(std::future::ready(Ok(())))),
            ]);
        }

        let running = Arc::new(Completion::new(std::future::pending()));
        gate.admit();
        gate.release(std::slice::from_ref(&running));
        let (kept, failure) = gate.take();
        assert_eq!(failure, Some(first_error));
        assert_eq!(kept.len(), 1);
        assert!(Arc::ptr_eq(&kept[0], &running));
        assert!(gate.take().1.is_none(), "eviction takes the failure once");

        let next = failed("a new copy's launch failed");
        let next_error = next.result().unwrap().unwrap_err();
        gate.admit();
        gate.release(&[next]);
        assert_eq!(gate.take().1, Some(next_error));
    }

    #[test]
    fn reaping_retains_a_failure_from_previously_running_work() {
        let ready = Arc::new(AtomicBool::new(false));
        let flag = ready.clone();
        let work = Arc::new(Completion::new(std::future::poll_fn(move |_| {
            if flag.load(Ordering::Acquire) {
                Poll::Ready(Err(furiosa_opt_std::Error::Invalid("late launch failure")))
            } else {
                Poll::Pending
            }
        })));
        let weak = Arc::downgrade(&work);
        let mut gate = Gate::default();
        gate.admit();
        gate.release(&[work]);
        ready.store(true, Ordering::Release);
        gate.reap();
        assert!(weak.upgrade().is_none());
        gate.admit();
        gate.release(&[failed("later failure")]);

        let (kept, failure) = gate.take();
        assert!(kept.is_empty());
        assert!(failure.unwrap().contains("late launch failure"));
    }

    #[test]
    fn refuses_unmatched_release_without_registering_work() {
        let mut gate = Gate::default();
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gate.release(&[failed("must not be registered")]);
        }));
        assert!(rejected.is_err());
        assert!(!gate.held());
        let (kept, failure) = gate.take();
        assert!(kept.is_empty());
        assert!(failure.is_none());

        gate.admit();
        assert!(!gate.release(&[]));
    }

    #[test]
    #[should_panic(expected = "cannot take readers while calls hold the copy")]
    fn refuses_to_take_work_while_a_call_is_admitted() {
        let mut gate = Gate::default();
        gate.admit();
        gate.take();
    }
}
