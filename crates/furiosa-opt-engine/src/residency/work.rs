//! Shared observation of accepted device work for parameter admission and retirement.

use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::task::{self, Poll, Wake, Waker};
use std::{future::Future, panic, pin::Pin};

/// Device work on a parameter's copy, its upload or a launch that reads it: any number of waiters
/// await it, and each sees its result once it ended.
pub struct Completion {
    work: Mutex<Work>,
    waiters: Arc<Waiters>,
}

type Running = Pin<Box<dyn Future<Output = Result<(), furiosa_opt_std::Error>> + Send>>;

enum Work {
    Running(Running),
    Ended(Result<(), String>),
}

/// The tasks waiting for one piece of work, woken together whenever the work may have moved on.
#[derive(Default)]
struct Waiters(Mutex<Vec<Waker>>);

impl Completion {
    /// Track accepted work. Dropping a waiter does not cancel the submitted device work.
    pub fn new(
        work: impl Future<Output = Result<(), furiosa_opt_std::Error>> + Send + 'static,
    ) -> Self {
        Self {
            work: Mutex::new(Work::Running(Box::pin(work))),
            waiters: Arc::default(),
        }
    }

    /// Resolves with the work's result once it ended. A device-task panic is retained as a
    /// failure too, so cleanup never polls that terminal runtime future a second time.
    pub fn ended(&self) -> impl Future<Output = Result<(), String>> + Send + '_ {
        std::future::poll_fn(|context| self.poll(context))
    }

    /// Blocks this thread until the work ends; the device runtime's worker wakes it.
    pub fn wait(&self) -> Result<(), String> {
        block_on(self.ended())
    }

    /// The work's result once it ended, without waiting.
    pub fn result(&self) -> Option<Result<(), String>> {
        match self.poll(&mut task::Context::from_waker(Waker::noop())) {
            Poll::Ready(ended) => Some(ended),
            Poll::Pending => None,
        }
    }

    fn poll(&self, context: &mut task::Context<'_>) -> Poll<Result<(), String>> {
        let mut work = self.work.lock().unwrap_or_else(PoisonError::into_inner);
        if let Work::Running(running) = &mut *work {
            // Register before polling, so a wake during the poll reaches this waiter. A cached
            // result needs no registration and must not keep its caller's executor alive.
            self.waiters.register(context.waker());
            let waker = Waker::from(self.waiters.clone());
            // The SDK consumes a launch before reporting a device-task panic. Cache it as an
            // error rather than leaving a consumed future for a later waiter or eviction.
            let polled = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                running
                    .as_mut()
                    .poll(&mut task::Context::from_waker(&waker))
            }));
            let polled = match polled {
                Ok(polled) => polled.map(|result| result.map_err(|error| error.to_string())),
                Err(payload) => Poll::Ready(Err(format!(
                    "accepted work panicked: {}",
                    payload
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| payload.downcast_ref::<&str>().copied())
                        .unwrap_or("non-string panic payload")
                ))),
            };
            let Poll::Ready(ended) = polled else {
                return Poll::Pending;
            };
            *work = Work::Ended(ended.clone());
            drop(work);
            self.waiters.wake_by_ref();
            return Poll::Ready(ended);
        }
        match &*work {
            Work::Ended(ended) => Poll::Ready(ended.clone()),
            Work::Running(_) => unreachable!("a ready poll ends the work"),
        }
    }
}

/// Polls `future` on this thread to its end, parking the thread between wakes.
///
/// Synchronous callers use this outside their device-context lock. It does not enter an
/// executor, so host preparation may still drive its own synchronous source reads.
/// Each call owns its notification state: a nested wait cannot consume its caller's wake.
pub fn block_on<F: Future>(future: F) -> F::Output {
    #[derive(Default)]
    struct Notify {
        pending: Mutex<bool>,
        changed: Condvar,
    }
    impl Wake for Notify {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            *self.pending.lock().unwrap_or_else(PoisonError::into_inner) = true;
            self.changed.notify_one();
        }
    }
    let notify = Arc::new(Notify::default());
    let waker = Waker::from(Arc::clone(&notify));
    let mut context = task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        let mut pending = notify
            .pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        while !*pending {
            pending = notify
                .changed
                .wait(pending)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *pending = false;
    }
}

impl Waiters {
    fn register(&self, waker: &Waker) {
        let mut wakers = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if !wakers.iter().any(|waiting| waiting.will_wake(waker)) {
            wakers.push(waker.clone());
        }
    }
}

impl Wake for Waiters {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let wakers = std::mem::take(&mut *self.0.lock().unwrap_or_else(PoisonError::into_inner));
        wakers.into_iter().for_each(Waker::wake);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    /// An inner synchronous wait must not consume a wake already sent to the outer future.
    #[test]
    fn nested_wait_preserves_its_callers_wake() {
        let (rescue, rescue_waker) = mpsc::channel();
        let (done, result) = mpsc::channel();
        let worker = thread::spawn(move || {
            let mut first = true;
            let value = block_on(std::future::poll_fn(move |cx| {
                if !first {
                    return Poll::Ready(42);
                }
                first = false;
                rescue.send(cx.waker().clone()).unwrap();
                cx.waker().wake_by_ref();
                let mut inner_ready = false;
                block_on(std::future::poll_fn(|cx| {
                    if inner_ready {
                        Poll::Ready(())
                    } else {
                        inner_ready = true;
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                }));
                Poll::Pending
            }));
            done.send(value).unwrap();
        });
        let rescue = rescue_waker.recv_timeout(Duration::from_secs(5)).unwrap();
        let value = result.recv_timeout(Duration::from_secs(5));
        // A regression must fail without leaving a permanently parked test thread.
        rescue.wake();
        worker.join().unwrap();
        assert_eq!(value.unwrap(), 42);
    }

    /// Request cleanup may synchronously drain accepted work while its caller is being polled
    /// by an executor. A wake from the device worker must reach the nested wait without panic.
    #[test]
    fn drains_from_an_async_caller() {
        let ready = Arc::new(AtomicBool::new(false));
        let signal = ready.clone();
        let (polled, polling) = mpsc::sync_channel(1);
        let work = Completion::new(std::future::poll_fn(move |cx| {
            if ready.load(Ordering::Acquire) {
                Poll::Ready(Ok(()))
            } else {
                let _ = polled.try_send(cx.waker().clone());
                Poll::Pending
            }
        }));
        let worker = thread::spawn(move || {
            let waker = polling.recv_timeout(Duration::from_secs(5)).unwrap();
            signal.store(true, Ordering::Release);
            waker.wake();
        });
        futures_executor::block_on(async { assert_eq!(work.wait(), Ok(())) });
        worker.join().unwrap();
    }

    /// One runtime wake reaches every reader waiting on the same accepted launch.
    #[test]
    fn wakes_every_waiter() {
        let ready = Arc::new(AtomicBool::new(false));
        let flag = ready.clone();
        let (polled, polling) = mpsc::channel();
        let work = Arc::new(Completion::new(std::future::poll_fn(move |cx| {
            if flag.load(Ordering::Acquire) {
                Poll::Ready(Ok(()))
            } else {
                polled.send(cx.waker().clone()).unwrap();
                Poll::Pending
            }
        })));
        let (done, ended) = mpsc::channel();
        let waiters: Vec<_> = (0..3)
            .map(|_| {
                let work = work.clone();
                let done = done.clone();
                thread::spawn(move || done.send(work.wait()).unwrap())
            })
            .collect();
        let wakers: Vec<_> = (0..3)
            .map(|_| polling.recv_timeout(Duration::from_secs(5)).unwrap())
            .collect();
        ready.store(true, Ordering::Release);
        wakers.into_iter().next().unwrap().wake();
        for _ in 0..3 {
            assert_eq!(ended.recv_timeout(Duration::from_secs(5)).unwrap(), Ok(()));
        }
        for waiter in waiters {
            waiter.join().unwrap();
        }
    }

    /// A late device failure remains available to eviction after earlier readers observed it.
    #[test]
    fn retains_failure_for_later_readers() {
        let work = Completion::new(std::future::ready(Err(furiosa_opt_std::Error::Invalid(
            "accepted launch failed",
        ))));
        let first = work.wait().unwrap_err();
        assert!(first.contains("accepted launch failed"));
        assert_eq!(work.result(), Some(Err(first.clone())));
        assert_eq!(futures_executor::block_on(work.ended()), Err(first));
    }

    /// Cancelling a request's waiter leaves the accepted launch observable by eviction.
    #[test]
    fn dropping_a_waiter_keeps_the_work() {
        let ready = Arc::new(AtomicBool::new(false));
        let flag = ready.clone();
        let work = Completion::new(std::future::poll_fn(move |_| {
            if flag.load(Ordering::Acquire) {
                Poll::Ready(Ok(()))
            } else {
                Poll::Pending
            }
        }));
        {
            let mut waiter = std::pin::pin!(work.ended());
            let mut context = task::Context::from_waker(Waker::noop());
            assert!(waiter.as_mut().poll(&mut context).is_pending());
        }
        ready.store(true, Ordering::Release);
        assert_eq!(work.wait(), Ok(()));
    }

    /// Keeping the completion's cached result must not retain the executor of a finished waiter.
    #[test]
    fn completed_work_releases_waiter_wakers() {
        struct Executor(std::sync::atomic::AtomicBool);
        impl Wake for Executor {
            fn wake(self: Arc<Self>) {
                self.0.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let work = Completion::new(std::future::ready(Ok(())));
        for _ in 0..2 {
            let executor = Arc::new(Executor(false.into()));
            let observer = Arc::downgrade(&executor);
            let waker = Waker::from(executor);
            assert!(work.poll(&mut task::Context::from_waker(&waker)).is_ready());
            drop(waker);
            assert!(observer.upgrade().is_none());
        }
    }

    /// Like the SDK's panicked launch, this future cannot be polled after its terminal panic.
    #[test]
    fn a_panicked_work_future_becomes_one_retained_failure() {
        let mut polls = 0;
        let work = Completion::new(std::future::poll_fn(move |_| {
            polls += 1;
            assert_eq!(polls, 1, "a consumed future must never be polled again");
            panic!("controlled device-task panic");
        }));
        let failure = work.wait().unwrap_err();
        assert_eq!(
            failure,
            "accepted work panicked: controlled device-task panic"
        );
        assert_eq!(work.result(), Some(Err(failure.clone())));
        assert_eq!(work.wait(), Err(failure));
    }
}
