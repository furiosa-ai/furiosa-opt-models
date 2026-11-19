//! One retained host parameter and its optional device copy.
//!
//! Admission never waits; eviction closes it, drains admitted calls, and returns the copy with
//! every upload and accepted reader to wait for. Initial host pinning and device placement each
//! serialize their own callers, while transfers, allocation, and work waits run outside the
//! state lock. Failed or unwinding preparation leaves the parameter retryable.

use super::Completion;
use super::gate::Gate;
use super::source::{Image, Source};
use furiosa_opt_std::prelude::{HbmBuffer, HbmTensor, M, MaterializableScalar, Placement};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::{any::Any, io};

pub(super) struct Parameter {
    source: Source,
    state: Mutex<State>,
    changed: Condvar,
}

#[derive(Default)]
struct State {
    image: Option<Arc<Image>>,
    copy: Option<Placed>,
    evicting: bool,
    placing: bool,
    pinning: bool,
}

struct Placed {
    buffer: Arc<HbmBuffer>,
    owner: Owner,
    upload: Option<Arc<Completion>>,
    gate: Gate,
}

/// An opened device, kept alive independently of its physical placement.
#[derive(Clone)]
pub(super) struct Owner {
    pub(super) placement: Placement,
    pub(super) device: Arc<dyn Any + Send + Sync>,
}

type Issued = (HbmBuffer, Option<Arc<Completion>>);

/// Upload callbacks share the engine's real device, but do not run under a parameter lock.
/// `place` completes its write; `issue` returns only after the pinned write is accepted.
pub(super) struct Destination<'a> {
    pub(super) owner: Owner,
    pub(super) place: &'a dyn Fn(&Source) -> io::Result<HbmBuffer>,
    pub(super) issue: &'a dyn Fn(&Image) -> io::Result<Issued>,
}

/// Closing either owns this eviction, joins an earlier eviction, or finds no copy.
pub(super) enum Closing {
    Drain,
    Settle,
    Host,
}

/// The copy remains owned until the selected tree has waited for all returned work.
pub(super) struct Drained {
    _copy: Arc<HbmBuffer>,
    pub(super) work: Vec<Arc<Completion>>,
    pub(super) failure: Option<String>,
    _owner: Owner,
}

/// A placement in progress is retired on every return path, including an upload callback panic.
struct Placing<'a>(&'a Parameter);

/// First-pin preparation owns only the pinning flag, independently of device placement.
struct Pinning<'a>(&'a Parameter);

impl Parameter {
    pub(super) fn new(source: Source) -> Self {
        Self {
            source,
            state: Mutex::default(),
            changed: Condvar::new(),
        }
    }

    #[cfg(test)]
    pub(super) fn source(&self) -> &Source {
        &self.source
    }

    pub(super) fn pin(&self) -> io::Result<()> {
        self.pin_with(Source::pin, || {})
    }

    #[cfg(test)]
    pub(super) fn owner(&self) -> Option<Owner> {
        self.state().copy.as_ref().map(|copy| copy.owner.clone())
    }

    /// Admit without waiting. The holder must release even if awaiting the upload is cancelled.
    pub(super) fn admit(
        &self,
        device: &Arc<dyn Any + Send + Sync>,
        path: &str,
    ) -> io::Result<Option<Arc<Completion>>> {
        let mut state = self.state();
        let State { copy, evicting, .. } = &mut *state;
        let copy = copy
            .as_mut()
            .filter(|_| !*evicting)
            .ok_or_else(|| host(path))?;
        if !copy.owner.opened_on(device) {
            return Err(mismatch(path));
        }
        let upload = copy.landing(path)?;
        copy.gate.admit();
        Ok(upload)
    }

    /// Lend a copy to an already admitted call, including while eviction waits for that call.
    /// A caller without prior admission must not use this to enter a tree being evicted.
    pub(super) fn lend(
        &self,
        owner: &Owner,
        path: &str,
    ) -> io::Result<(Arc<HbmBuffer>, Option<Arc<Completion>>)> {
        let mut state = self.state();
        let State { copy, evicting, .. } = &mut *state;
        let copy = copy
            .as_mut()
            .filter(|copy| !*evicting || copy.gate.held())
            .ok_or_else(|| host(path))?;
        copy.owner.agrees(owner, path)?;
        let upload = copy.landing(path)?;
        copy.gate.admit();
        Ok((copy.buffer.clone(), upload))
    }

    /// Register accepted readers before releasing admission, including on later call failure.
    pub(super) fn release(&self, launched: &[Arc<Completion>]) {
        let mut state = self.state();
        let State { copy, evicting, .. } = &mut *state;
        let copy = copy.as_mut().expect("an admitted call retains its copy");
        if !copy.gate.release(launched) && *evicting {
            self.changed.notify_all();
        }
    }

    pub(super) fn close(&self) -> Closing {
        // A concurrent placement must publish or fail before eviction decides there is no copy.
        let mut state = self
            .changed
            .wait_while(self.state(), |state| state.placing)
            .unwrap_or_else(PoisonError::into_inner);
        match (&state.copy, state.evicting) {
            (_, true) => Closing::Settle,
            (None, false) => Closing::Host,
            (Some(_), false) => {
                state.evicting = true;
                Closing::Drain
            }
        }
    }

    pub(super) fn drain(&self) -> Drained {
        let mut state = self
            .changed
            .wait_while(self.state(), |state| {
                state.copy.as_ref().is_some_and(|copy| copy.gate.held())
            })
            .unwrap_or_else(PoisonError::into_inner);
        assert!(
            state.evicting,
            "only the eviction that closed admission drains"
        );
        let Placed {
            buffer,
            owner,
            upload,
            mut gate,
            ..
        } = state
            .copy
            .take()
            .expect("the closing eviction owns the copy");
        let (readers, failure) = gate.take();
        Drained {
            _copy: buffer,
            work: upload.into_iter().chain(readers).collect(),
            failure,
            _owner: owner,
        }
    }

    /// Reopen only after all copies selected by the tree eviction are safe to free and dropped.
    pub(super) fn open(&self) {
        self.state().evicting = false;
        self.changed.notify_all();
    }

    pub(super) fn settle(&self) {
        drop(self.settled());
    }

    /// Place once, serializing duplicate attempts without holding a lock across their transfer.
    pub(super) fn upload<D: MaterializableScalar, C: M, S: M>(
        &self,
        destination: &Destination<'_>,
        path: &str,
    ) -> io::Result<()> {
        if destination.owner.placement.chips.len() != C::SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{path}: the tensor's chip count differs from the destination"),
            ));
        }
        let image = {
            let mut state = self.settled();
            if let Some(copy) = &mut state.copy {
                copy.gate.reap();
                return copy.owner.agrees(&destination.owner, path);
            }
            state.placing = true;
            state.image.clone()
        };
        let placing = Placing(self);
        let (buffer, upload) = match image {
            None => ((destination.place)(&self.source)?, None),
            Some(image) => (destination.issue)(&image)?,
        };
        // Validate an alias while retaining the original allocation. On a malformed issued
        // destination, accepted work must finish before either that copy or its image is freed.
        match buffer.alias(..).and_then(HbmTensor::<D, C, S>::try_from) {
            Ok(tensor) => drop(tensor),
            Err(error) => {
                if let Some(upload) = &upload {
                    let _ = upload.wait();
                }
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{path}: {error}"),
                ));
            }
        }
        self.state().copy = Some(Placed {
            buffer: Arc::new(buffer),
            owner: destination.owner.clone(),
            upload,
            gate: Gate::default(),
        });
        drop(placing);
        Ok(())
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Only one caller prepares the initial image. The private callbacks let CPU tests replace
    /// allocation and observe a waiter entering the condition-variable wait without sleeping.
    fn pin_with(
        &self,
        prepare: impl FnOnce(&Source) -> io::Result<Image>,
        mut waiting: impl FnMut(),
    ) -> io::Result<()> {
        let mut state = self
            .changed
            .wait_while(self.state(), |state| {
                if state.pinning {
                    waiting();
                }
                state.pinning
            })
            .unwrap_or_else(PoisonError::into_inner);
        if state.image.is_some() {
            return Ok(());
        }
        state.pinning = true;
        let pinning = Pinning(self);
        drop(state);

        let image = Arc::new(prepare(&self.source)?);
        self.state().image = Some(image);
        drop(pinning);
        Ok(())
    }

    fn settled(&self) -> MutexGuard<'_, State> {
        self.changed
            .wait_while(self.state(), |state| state.evicting || state.placing)
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl Drop for Placing<'_> {
    fn drop(&mut self) {
        self.0.state().placing = false;
        self.0.changed.notify_all();
    }
}

impl Drop for Pinning<'_> {
    fn drop(&mut self) {
        self.0.state().pinning = false;
        self.0.changed.notify_all();
    }
}

impl Drop for Parameter {
    fn drop(&mut self) {
        // Admissions retain their parameter owner. The last owner can only disappear after
        // calls release, but its accepted launches may still be running.
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        if let Some(mut copy) = state.copy.take() {
            let (readers, _) = copy.gate.take();
            for work in copy.upload.iter().chain(&readers) {
                let _ = work.wait();
            }
            drop(copy);
        }
    }
}

impl Placed {
    fn landing(&mut self, path: &str) -> io::Result<Option<Arc<Completion>>> {
        match self.upload.as_ref().map(|upload| upload.result()) {
            Some(Some(Ok(()))) => {
                self.upload = None;
                Ok(None)
            }
            Some(Some(Err(error))) => Err(io::Error::other(format!(
                "{path}: its upload failed: {error}"
            ))),
            Some(None) | None => Ok(self.upload.clone()),
        }
    }
}

impl Owner {
    pub(super) fn is(&self, other: &Self) -> bool {
        self.opened_on(&other.device)
    }

    pub(super) fn opened_on(&self, device: &Arc<dyn Any + Send + Sync>) -> bool {
        std::ptr::addr_eq(Arc::as_ptr(&self.device), Arc::as_ptr(device))
    }

    fn agrees(&self, other: &Self, path: &str) -> io::Result<()> {
        if self.is(other) && self.placement == other.placement {
            Ok(())
        } else {
            Err(mismatch(path))
        }
    }
}

fn host(path: &str) -> io::Error {
    io::Error::new(io::ErrorKind::WouldBlock, format!("{path} is not resident"))
}

fn mismatch(path: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("{path} resides on another opened device"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_models::tensor::parallel::Shard;
    use furiosa_opt_std::prelude::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::task::{Poll, Waker};
    use std::thread;
    use std::time::Duration;

    axes![Width = 8];

    /// Complete controlled work on drop too, so an assertion failure cannot strand cleanup.
    struct Signal {
        ready: Arc<AtomicBool>,
        waker: Arc<Mutex<Option<Waker>>>,
    }

    impl Signal {
        fn complete(&self) {
            self.ready.store(true, Ordering::Release);
            if let Some(waker) = self.waker.lock().unwrap().take() {
                waker.wake();
            }
        }
    }

    impl Drop for Signal {
        fn drop(&mut self) {
            self.complete();
        }
    }

    fn parameter() -> Parameter {
        Parameter::new(
            Source::from_host::<f32, m![1], m![Width], m![Width]>(
                HostTensor::from_vec((0..8).map(|at| at as f32)),
                "layer.weight",
            )
            .unwrap(),
        )
    }

    fn owner() -> Owner {
        Owner {
            placement: Placement {
                chips: vec![4],
                pes: 0..8,
            },
            device: Arc::new(0u8),
        }
    }

    fn controlled() -> (Arc<Completion>, Signal) {
        let ready = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(Mutex::new(None));
        let flag = ready.clone();
        let slot = waker.clone();
        let work = Completion::new(std::future::poll_fn(move |cx| {
            let mut slot = slot.lock().unwrap();
            if flag.load(Ordering::Acquire) {
                Poll::Ready(Ok(()))
            } else {
                *slot = Some(cx.waker().clone());
                Poll::Pending
            }
        }));
        (Arc::new(work), Signal { ready, waker })
    }

    fn hardware() -> (Arc<Mutex<Device>>, Owner) {
        let device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let placement = device.placement().clone();
        let device = Arc::new(Mutex::new(device));
        let owner = Owner {
            placement,
            device: device.clone(),
        };
        (device, owner)
    }

    /// Hold one allocation open until a second caller is actually waiting. Channel handshakes
    /// establish that overlap without sleeps, pinned memory, or a device allocation.
    fn concurrent_pin_retry(unwind: bool) {
        let parameter = Arc::new(parameter());
        let calls = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        let (entered, entering) = mpsc::channel();
        let (finish, finishing) = mpsc::channel();
        let (waiting, queued) = mpsc::channel();

        struct Allocating(Arc<AtomicUsize>);
        impl Drop for Allocating {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }

        let first_parameter = parameter.clone();
        let first_calls = calls.clone();
        let first_active = active.clone();
        let first = thread::spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                first_parameter.pin_with(
                    |source| {
                        assert_eq!(source.len(), 32);
                        let previous = first_active.fetch_add(1, Ordering::SeqCst);
                        let _allocating = Allocating(first_active);
                        assert_eq!(previous, 0);
                        first_calls.fetch_add(1, Ordering::SeqCst);
                        entered.send(()).unwrap();
                        finishing.recv().unwrap();
                        if unwind {
                            panic!("first pin allocation panicked");
                        }
                        Err(io::Error::other("first pin allocation failed"))
                    },
                    || {},
                )
            }))
        });
        entering.recv().unwrap();

        let second_parameter = parameter.clone();
        let second_calls = calls.clone();
        let second_active = active.clone();
        let second = thread::spawn(move || {
            second_parameter.pin_with(
                |_| {
                    let previous = second_active.fetch_add(1, Ordering::SeqCst);
                    let _allocating = Allocating(second_active);
                    assert_eq!(previous, 0);
                    second_calls.fetch_add(1, Ordering::SeqCst);
                    Err(io::Error::other("second pin allocation failed"))
                },
                || waiting.send(()).unwrap(),
            )
        });
        queued.recv().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(active.load(Ordering::SeqCst), 1);
        {
            let state = parameter.state();
            assert!(state.pinning);
            assert!(!state.placing);
            assert!(!state.evicting);
        }

        // Pin preparation does not close admission or serialize ordinary device placement.
        // The failing callback exercises upload selection without allocating an HBM buffer.
        let place_calls = std::cell::Cell::new(0);
        let place = |_: &Source| {
            place_calls.set(place_calls.get() + 1);
            let state = parameter.state();
            assert!(state.pinning && state.placing);
            Err(io::Error::other("independent placement failed"))
        };
        let issue = |_: &Image| panic!("the pin has not published an image");
        let destination = Destination {
            owner: owner(),
            place: &place,
            issue: &issue,
        };
        let error = parameter
            .upload::<f32, m![1], m![Width]>(&destination, "weight")
            .unwrap_err();
        assert_eq!(error.to_string(), "independent placement failed");
        assert_eq!(place_calls.get(), 1);
        assert!(!parameter.state().placing);
        assert!(matches!(parameter.close(), Closing::Host));
        assert!(parameter.state().pinning);

        finish.send(()).unwrap();
        let first = first.join().unwrap();
        if unwind {
            let payload = first.expect_err("the allocation's panic is preserved");
            assert_eq!(
                payload.downcast_ref::<&str>(),
                Some(&"first pin allocation panicked")
            );
        } else {
            assert_eq!(
                first.unwrap().unwrap_err().to_string(),
                "first pin allocation failed"
            );
        }
        assert_eq!(
            second.join().unwrap().unwrap_err().to_string(),
            "second pin allocation failed"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        let state = parameter.state();
        assert!(!state.pinning);
        assert!(!state.placing);
        assert!(!state.evicting);
        assert!(state.image.is_none());
        assert!(state.copy.is_none());
    }

    #[test]
    fn concurrent_pin_serializes_allocation_and_retries_failure() {
        concurrent_pin_retry(false);
    }

    #[test]
    fn concurrent_pin_serializes_allocation_and_retries_unwind() {
        concurrent_pin_retry(true);
    }

    #[test]
    fn host_parameters_refuse_admission_without_a_device_copy() {
        let parameter = parameter();
        let owner = owner();
        let error = parameter
            .admit(&owner.device, "layer.weight")
            .err()
            .unwrap();
        assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
        assert!(error.to_string().contains("layer.weight"));
        assert!(parameter.owner().is_none());
        assert!(matches!(parameter.close(), Closing::Host));
        assert_eq!(parameter.source().len(), 32);
    }

    #[test]
    fn placement_equality_does_not_replace_opened_device_identity() {
        let first = owner();
        assert!(first.is(&first.clone()));
        let second = owner();
        assert_eq!(first.placement, second.placement);
        assert!(!first.is(&second));
        assert_eq!(
            first.agrees(&second, "layer.weight").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    #[ignore = "requires page-locked host memory from /dev/dma_heap/system"]
    fn pinning_reuses_its_image_without_placing_a_copy() {
        let parameter = parameter();
        parameter.pin().unwrap();
        let first = parameter.state().image.clone().unwrap();
        parameter.pin().unwrap();
        let again = parameter.state().image.clone().unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        assert!(parameter.owner().is_none());
        assert!(matches!(parameter.close(), Closing::Host));
    }

    /// Cancellation before submission still leaves the upload to drain; lending under prior
    /// admission remains valid while eviction closes new calls. Late failures are recoverable.
    #[test]
    #[ignore = "requires one explicitly selected RNGD chip and page-locked host memory"]
    fn retires_uploads_and_readers_before_reusing_a_parameter() {
        let (device, owner) = hardware();
        let parameter = Arc::new(parameter());
        parameter.pin().unwrap();
        let (upload, signal) = controlled();
        let place = |_: &Source| panic!("a pinned parameter issues its reusable image");
        let issue = |image: &Image| {
            let buffer =
                HbmBuffer::alloc(&device.lock().unwrap(), image.len()).map_err(io::Error::other)?;
            Ok((buffer, Some(upload.clone())))
        };
        let destination = Destination {
            owner: owner.clone(),
            place: &place,
            issue: &issue,
        };
        parameter
            .upload::<f32, m![1], m![Width]>(&destination, "layer.weight")
            .unwrap();
        assert!(
            parameter
                .admit(&owner.device, "layer.weight")
                .unwrap()
                .is_some()
        );
        parameter.release(&[]); // The call cancelled while waiting for its upload.
        assert!(matches!(parameter.close(), Closing::Drain));
        assert!(parameter.admit(&owner.device, "layer.weight").is_err());
        let drained = parameter.drain();
        assert_eq!(drained.work.len(), 1);
        assert!(Arc::ptr_eq(&drained.work[0], &upload));
        assert!(drained.failure.is_none());
        let copy = Arc::downgrade(&drained._copy);
        assert!(copy.upgrade().is_some());
        signal.complete();
        for work in &drained.work {
            work.wait().unwrap();
        }
        drop(drained);
        parameter.open();
        assert!(copy.upgrade().is_none());
        assert!(parameter.owner().is_none());

        parameter
            .upload::<f32, m![1], m![Width]>(&destination, "layer.weight")
            .unwrap();
        assert!(
            parameter
                .admit(&owner.device, "layer.weight")
                .unwrap()
                .is_none()
        );
        assert!(matches!(parameter.close(), Closing::Drain));
        let (lent, _) = parameter.lend(&owner, "layer.weight").unwrap();
        let (reader, finished) = controlled();
        parameter.release(std::slice::from_ref(&reader));
        parameter.release(&[]);
        drop(lent);
        let drained = parameter.drain();
        assert_eq!(drained.work.len(), 1);
        assert!(Arc::ptr_eq(&drained.work[0], &reader));
        finished.complete();
        for work in &drained.work {
            work.wait().unwrap();
        }
        drop(drained);
        parameter.open();

        parameter
            .upload::<f32, m![1], m![Width]>(&destination, "layer.weight")
            .unwrap();
        parameter.admit(&owner.device, "layer.weight").unwrap();
        let failure = Arc::new(Completion::new(std::future::ready(Err(
            furiosa_opt_std::Error::Invalid("accepted reader failed"),
        ))));
        parameter.release(&[failure]);
        assert!(matches!(parameter.close(), Closing::Drain));
        let drained = parameter.drain();
        assert!(
            drained
                .failure
                .as_ref()
                .unwrap()
                .contains("accepted reader failed")
        );
        drop(drained);
        parameter.open();
        parameter
            .upload::<f32, m![1], m![Width]>(&destination, "layer.weight")
            .unwrap();
        parameter.admit(&owner.device, "layer.weight").unwrap();
        parameter.release(&[]);
    }

    /// Duplicate placements serialize without holding the state lock over the transfer callback.
    #[test]
    #[ignore = "requires one explicitly selected RNGD chip"]
    fn places_once_when_uploads_race() {
        let (device, owner) = hardware();
        let parameter = Arc::new(parameter());
        let count = Arc::new(AtomicUsize::new(0));
        let (entered, entering) = mpsc::channel();
        let (go, resume) = mpsc::channel();
        let resume = Arc::new(Mutex::new(resume));
        let worker = |parameter: Arc<Parameter>| {
            let device = device.clone();
            let owner = owner.clone();
            let count = count.clone();
            let entered = entered.clone();
            let resume = resume.clone();
            thread::spawn(move || {
                let place = |source: &Source| {
                    count.fetch_add(1, Ordering::Relaxed);
                    entered.send(()).unwrap();
                    resume.lock().unwrap().recv().unwrap();
                    HbmBuffer::alloc(&device.lock().unwrap(), source.len())
                        .map_err(io::Error::other)
                };
                let issue = |_: &Image| panic!("this source was not pinned");
                parameter.upload::<f32, m![1], m![Width]>(
                    &Destination {
                        owner,
                        place: &place,
                        issue: &issue,
                    },
                    "layer.weight",
                )
            })
        };
        let first = worker(parameter.clone());
        entering.recv_timeout(Duration::from_secs(5)).unwrap();
        // This immediate refusal proves the transfer callback does not hold the state lock.
        assert_eq!(
            parameter
                .admit(&owner.device, "layer.weight")
                .err()
                .unwrap()
                .kind(),
            io::ErrorKind::WouldBlock
        );
        let second = worker(parameter.clone());
        // A second token lets a broken duplicate implementation finish rather than hang the test.
        go.send(()).unwrap();
        go.send(()).unwrap();
        first.join().unwrap().unwrap();
        second.join().unwrap().unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 1);
        parameter.admit(&owner.device, "layer.weight").unwrap();
        parameter.release(&[]);
    }

    #[test]
    fn failed_or_panicking_placement_leaves_the_parameter_retryable() {
        let parameter = parameter();
        let owner = owner();
        let failed = |_: &Source| Err(io::Error::other("placement failed"));
        let issue = |_: &Image| panic!("the source is not pinned");
        let destination = Destination {
            owner: owner.clone(),
            place: &failed,
            issue: &issue,
        };
        assert!(
            parameter
                .upload::<f32, m![1], m![Width]>(&destination, "weight")
                .is_err()
        );
        assert!(!parameter.state().placing);
        let panics = |_: &Source| panic!("placement callback panicked");
        let destination = Destination {
            owner,
            place: &panics,
            issue: &issue,
        };
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                parameter.upload::<f32, m![1], m![Width]>(&destination, "weight")
            }))
            .is_err()
        );
        assert!(!parameter.state().placing);
        assert!(matches!(parameter.close(), Closing::Host));
        assert!(parameter.owner().is_none());
    }

    /// Dropping the last tree owner after submission still waits for accepted readers.
    #[test]
    #[ignore = "requires one explicitly selected RNGD chip"]
    fn last_owner_waits_for_accepted_readers() {
        let (device, owner) = hardware();
        let parameter = Arc::new(parameter());
        let place = |source: &Source| {
            HbmBuffer::alloc(&device.lock().unwrap(), source.len()).map_err(io::Error::other)
        };
        let issue = |_: &Image| panic!("the source is not pinned");
        parameter
            .upload::<f32, m![1], m![Width]>(
                &Destination {
                    owner: owner.clone(),
                    place: &place,
                    issue: &issue,
                },
                "weight",
            )
            .unwrap();
        let (reader, finish) = controlled();
        parameter.admit(&owner.device, "weight").unwrap();
        parameter.release(&[reader]);
        let (started, starting) = mpsc::channel();
        let (done, ended) = mpsc::channel();
        let waiter = thread::spawn(move || {
            started.send(()).unwrap();
            drop(parameter);
            done.send(()).unwrap();
        });
        starting.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(matches!(
            ended.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        finish.complete();
        ended.recv_timeout(Duration::from_secs(5)).unwrap();
        waiter.join().unwrap();
    }

    /// An issued transfer's wrong destination shape is reported only after its work retires.
    #[test]
    #[ignore = "requires one explicitly selected RNGD chip and page-locked host memory"]
    fn malformed_issued_destination_waits_before_returning_failure() {
        let (device, owner) = hardware();
        let parameter = Arc::new(parameter());
        parameter.pin().unwrap();
        let (upload, finish) = controlled();
        let (issued, issuing) = mpsc::channel();
        let (done, ended) = mpsc::channel();
        let worker_parameter = parameter.clone();
        let waiter = thread::spawn(move || {
            let place = |_: &Source| panic!("the parameter was pinned");
            let issue = |image: &Image| {
                let buffer = HbmBuffer::alloc(&device.lock().unwrap(), image.len() - 8)
                    .map_err(io::Error::other)?;
                issued.send(()).unwrap();
                Ok((buffer, Some(upload.clone())))
            };
            let result = worker_parameter.upload::<f32, m![1], m![Width]>(
                &Destination {
                    owner,
                    place: &place,
                    issue: &issue,
                },
                "weight",
            );
            done.send(result.map_err(|error| error.kind())).unwrap();
        });
        issuing.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(matches!(
            ended.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        finish.complete();
        assert_eq!(
            ended.recv_timeout(Duration::from_secs(5)).unwrap(),
            Err(io::ErrorKind::InvalidData)
        );
        waiter.join().unwrap();
        assert!(parameter.owner().is_none());
        assert!(!parameter.state().placing);
        assert!(matches!(parameter.close(), Closing::Host));
    }

    /// The state machine and real uploader retain one pinned source across eviction/reload,
    /// reproducing each replicated chip's bytes without a device-to-host weight download.
    #[test]
    #[ignore = "requires four explicitly selected RNGD chips and page-locked host memory"]
    fn reloads_replicated_bytes_from_the_same_pinned_image() {
        axes![Bulk = 262144];
        type Chips = m![{ Symbol<Shard<4>> }];
        let device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
        let placement = device.placement().clone();
        let device = Arc::new(Mutex::new(device));
        let owner = Owner {
            placement,
            device: device.clone(),
        };
        let values: Vec<f32> = (0..Bulk::SIZE).map(|at| (at % 61) as f32 / 32.).collect();
        let expected: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let parameter = Parameter::new(
            Source::from_host::<f32, Chips, m![Bulk], m![Bulk]>(
                HostTensor::from_vec(values),
                "weight",
            )
            .unwrap(),
        );
        parameter.pin().unwrap();
        let image = parameter.state().image.clone().unwrap();
        let place =
            |source: &Source| super::super::upload::place(&mut device.lock().unwrap(), source);
        let issue = |image: &Image| super::super::upload::issue(&mut device.lock().unwrap(), image);
        let destination = Destination {
            owner: owner.clone(),
            place: &place,
            issue: &issue,
        };
        for _ in 0..2 {
            parameter
                .upload::<f32, Chips, m![Bulk]>(&destination, "weight")
                .unwrap();
            if let Some(upload) = parameter.admit(&owner.device, "weight").unwrap() {
                upload.wait().unwrap();
            }
            let (buffer, _) = parameter.lend(&owner, "weight").unwrap();
            let got = futures_executor::block_on(
                buffer
                    .slice(.., ..)
                    .unwrap()
                    .to_host(&mut device.lock().unwrap().pdma)
                    .submit(),
            )
            .unwrap();
            for chip in got.chunks_exact(expected.len()) {
                assert_eq!(chip, expected);
            }
            assert_eq!(got.len(), 4 * expected.len());
            parameter.release(&[]);
            parameter.release(&[]);
            drop(buffer);
            assert!(matches!(parameter.close(), Closing::Drain));
            let drained = parameter.drain();
            for work in &drained.work {
                work.wait().unwrap();
            }
            drop(drained);
            parameter.open();
            assert!(parameter.owner().is_none());
            parameter.pin().unwrap();
            assert!(Arc::ptr_eq(
                &image,
                parameter.state().image.as_ref().unwrap()
            ));
        }
    }
}
