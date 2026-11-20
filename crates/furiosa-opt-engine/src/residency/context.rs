//! Residency bound to the framework's already opened device and its shared context owner.

use super::parameter::{Destination, Owner};
use super::resident::{Admitted, Launches, Ready, Resident};
use super::upload;
use furiosa_opt_models::tensor::memory::Hbm;
use furiosa_opt_models::tensor::memory::residency::Residency;
use furiosa_opt_std::prelude::{Device, Placement};
use std::sync::{Arc, Mutex, PoisonError};
use std::{any::Any, io};

/// A framework context containing one opened device, optionally beside its executor or state.
/// Implementations lend that same device for their lifetime; binding never opens another one.
pub trait DeviceContext: Send + 'static {
    fn device(&mut self) -> &mut Device;
}

impl DeviceContext for Device {
    fn device(&mut self) -> &mut Device {
        self
    }
}

/// Retains and serializes access to an existing framework context. Clones and independent
/// bindings of the same Arc have the same residency identity, including the original executor.
/// Pinned upload/launch submission uses the lock; their completion waits and eviction run outside
/// it. Initial pageable placement retains its existing fully copied, blocking load behavior.
#[derive(Clone)]
pub struct SharedDevice {
    shared: Arc<Mutex<dyn DeviceContext>>,
    owner: Owner,
}

impl SharedDevice {
    /// Bind the original Arc without extracting, moving or reopening its device.
    pub fn bind<C: DeviceContext>(shared: Arc<Mutex<C>>) -> Self {
        let placement = shared
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .device()
            .placement()
            .clone();
        let device: Arc<dyn Any + Send + Sync> = shared.clone();
        Self {
            shared,
            owner: Owner { placement, device },
        }
    }

    pub fn placement(&self) -> &Placement {
        &self.owner.placement
    }

    /// Submit or inspect work on the bound device. Return pending work so its completion can
    /// be awaited after releasing the lock. Wait for a dependent input before acquiring it.
    pub fn with_device<R>(&self, call: impl FnOnce(&mut Device) -> io::Result<R>) -> io::Result<R> {
        let mut context = self.shared.lock().unwrap_or_else(PoisonError::into_inner);
        let device = context.device();
        if device.placement() != self.placement() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the bound device placement changed",
            ));
        }
        call(device)
    }

    pub(crate) fn upload<T: Residency>(&self, resident: &Resident<T>) -> io::Result<()> {
        let place = |source: &_| self.with_device(|device| upload::place(device, source));
        let issue = |image: &_| self.with_device(|device| upload::issue(device, image));
        resident.to_hbm(&Destination {
            owner: self.owner.clone(),
            place: &place,
            issue: &issue,
        })
    }

    pub(crate) fn admit<'r, T: Residency>(
        &self,
        resident: &'r Resident<T>,
    ) -> io::Result<Admitted<'r, T>> {
        resident.admit(&self.owner)
    }

    pub(crate) fn launch<T: Residency, O>(
        &self,
        ready: Ready<'_, T>,
        call: impl for<'m, 'w> FnOnce(&mut Launches<'_>, &'m T::In<&'w Hbm>) -> io::Result<O>,
    ) -> io::Result<O> {
        ready.check_owner(&self.owner)?;
        self.with_device(|device| ready.launch(device, call))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residency::cell::Cell;
    use furiosa_opt_models::Op;
    use furiosa_opt_models::load::{Context, Load};
    use furiosa_opt_models::tensor::parallel::Shard;
    use furiosa_opt_std::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    type Chips = m![{ Symbol<Shard<4>> }];
    type Values = HbmTensor<f32, Chips, m![64]>;

    struct Framework {
        device: Device,
        dropped: Arc<AtomicBool>,
    }

    impl DeviceContext for Framework {
        fn device(&mut self) -> &mut Device {
            &mut self.device
        }
    }

    impl Drop for Framework {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    #[derive(DeviceSend)]
    struct Copy;

    impl Op<&Values> for Copy {
        type Output = Values;

        fn forward(&self, device: &mut Device, input: &Values) -> Values {
            input.to_hbm(&mut device.tdma)
        }
    }

    #[device(chip = 4)]
    fn copied(device: &mut Device, module: &Copy, _: (), _: &mut (), input: &Values) -> Values {
        Op::forward(module, device, input)
    }

    #[test]
    #[ignore = "requires four explicitly selected RNGD chips and pinned host memory"]
    fn residency_uses_and_retains_the_original_framework_context() {
        let dropped = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(Mutex::new(Framework {
            device: Device::new(Topology { chips: 4, pes: 8 }).unwrap(),
            dropped: Arc::clone(&dropped),
        }));
        let weak = Arc::downgrade(&shared);
        let first = SharedDevice::bind(Arc::clone(&shared));
        let second = SharedDevice::bind(Arc::clone(&shared));
        assert!(first.owner.opened_on(&second.owner.device));
        drop(shared);
        assert!(weak.upgrade().is_some());
        let expected: Vec<_> = (0..4 * 64).map(|at| at as f32 / 8.0 - 7.0).collect();
        let cell: Cell<f32, Chips, m![64]> = Load::load(
            &mut Context::new(None),
            HostTensor::<f32, m![{ Chips }, 64]>::from_vec(expected.clone()),
        )
        .unwrap();
        let resident = Resident::<Values>::new(cell);
        resident.pin().unwrap();
        for caller_output in [false, true] {
            first.upload(&resident).unwrap();
            let ready =
                futures_executor::block_on(second.admit(&resident).unwrap().landed()).unwrap();
            let other = Owner {
                placement: first.placement().clone(),
                device: Arc::new(()),
            };
            let error = ready.check_owner(&other).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            let output = second
                .launch(ready, |launches, input| {
                    if caller_output {
                        let mut output =
                            Values::alloc(launches.device()).map_err(io::Error::other)?;
                        launches.forward_into_tracked(
                            Copy,
                            copied,
                            (),
                            &mut (),
                            *input,
                            &mut output,
                        )?;
                        Ok(output)
                    } else {
                        launches.forward(Copy, copied, (), &mut (), *input)
                    }
                })
                .unwrap();
            // Retire after accepted work with no context lock held, then read the independent result.
            resident.to_host().unwrap();
            futures_executor::block_on(output.ready()).unwrap();
            let actual = first
                .with_device(|device| {
                    futures_executor::block_on(
                        output
                            .to_host::<m![{ Chips }, 64]>(&mut device.pdma)
                            .into_future(),
                    )
                    .map(HostTensor::into_vec)
                    .map_err(io::Error::other)
                })
                .unwrap();
            assert_eq!(actual, expected);
            drop(output);
        }
        drop((resident, first));
        assert!(!dropped.load(Ordering::SeqCst));
        drop(second);
        assert!(dropped.load(Ordering::SeqCst));
        assert!(weak.upgrade().is_none());
    }
}
