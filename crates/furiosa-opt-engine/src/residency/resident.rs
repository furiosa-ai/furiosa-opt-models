//! A typed tree of shared parameters, with whole-tree admission and retirement.

use super::Completion;
use super::cell::{Cell, Cells};
use super::parameter::{Closing, Destination, Owner, Parameter};
use furiosa_opt_models::tensor::memory::Hbm;
use furiosa_opt_models::tensor::memory::residency::{Leaves, Path, Residency, Visit};
use furiosa_opt_std::prelude::*;
use std::{collections::BTreeSet, convert::Infallible, io, marker::PhantomData, sync::Arc};

pub(crate) struct Resident<T: Residency> {
    tree: T::In<Cells>,
    name: Arc<str>,
}

/// Admission owns each distinct parameter while uploads are awaited outside the device lock.
pub(crate) struct Admitted<'r, T: Residency> {
    resident: &'r Resident<T>,
    owner: Owner,
    holds: Holds,
}

/// Only completed uploads can produce a launch-ready call.
pub(crate) struct Ready<'r, T: Residency> {
    resident: &'r Resident<T>,
    owner: Owner,
    holds: Holds,
}

#[derive(Default)]
struct Holds {
    parameters: Vec<Arc<Parameter>>,
    uploads: Vec<(String, Arc<Completion>)>,
    launched: Vec<Arc<Completion>>,
}

/// Extra holds while the typed alias tree is lent to a launch. The whole-call hold outlives them.
#[derive(Default)]
struct Loans(Vec<Arc<Parameter>>);

struct Parameters(Vec<(Arc<Parameter>, String)>);
struct Share;
struct Pin;
struct Upload<'a>(&'a Destination<'a>);
struct Lend<'a> {
    owner: &'a Owner,
    loans: &'a mut Loans,
}
struct Borrow;

/// The only device access a scoped reader receives: each accepted launch is recorded at once.
pub(crate) struct Launches<'a, const CHIPS: usize = 1> {
    device: &'a mut Device,
    launched: &'a mut Vec<Arc<Completion>>,
}

impl<T: Residency> Resident<T> {
    pub(crate) fn new(tree: T::In<Cells>) -> Self {
        Self {
            tree,
            name: Arc::from(""),
        }
    }

    /// Admit two shared subtrees as one launch tree, without copying or changing their weights.
    /// Retirement still targets the original views, so a streamed layer need not evict constants
    /// that its calls share. The tuple's structural paths identify its two children.
    pub(crate) fn pair<U: Residency>(&self, other: &Resident<U>) -> Resident<(T, U)> {
        Resident::new((self.clone().tree, other.clone().tree))
    }

    /// The selected subtree shares its parameter cells. A model's component view must also
    /// retain its pipeline owner when the model requires the other components to stay resident.
    pub(crate) fn part<U: Residency>(
        &self,
        name: &str,
        select: impl FnOnce(&T::In<Cells>) -> &U::In<Cells>,
    ) -> Resident<U> {
        let Ok(tree) = U::map(select(&self.tree), &mut Share, Path::ROOT);
        let name = if self.name.is_empty() {
            name.into()
        } else {
            format!("{}.{name}", self.name).into()
        };
        Resident { tree, name }
    }

    pub(crate) fn pin(&self) -> io::Result<()> {
        let root = Path::ROOT;
        T::visit(&self.tree, &mut Pin, self.path(&root))
    }

    pub(super) fn to_hbm(&self, destination: &Destination<'_>) -> io::Result<()> {
        let root = Path::ROOT;
        T::visit(&self.tree, &mut Upload(destination), self.path(&root))
    }

    /// Close admission on the whole tree first, wait for all work, release every copy, then
    /// reopen and report failure. Retained host sources and pinned images are never downloaded.
    pub(crate) fn to_host(&self) -> io::Result<()> {
        let closed = self
            .parameters()
            .into_iter()
            .map(|(parameter, _)| {
                let closing = parameter.close();
                (parameter, closing)
            })
            .collect();
        retire(closed)
    }

    pub(super) fn admit(&self, owner: &Owner) -> io::Result<Admitted<'_, T>> {
        let mut holds = Holds::default();
        for (parameter, path) in self.parameters() {
            let upload = parameter.admit(&owner.device, &path)?;
            holds.parameters.push(parameter);
            if let Some(upload) = upload
                && !holds
                    .uploads
                    .iter()
                    .any(|(_, kept)| Arc::ptr_eq(kept, &upload))
            {
                holds.uploads.push((path, upload));
            }
        }
        Ok(Admitted {
            resident: self,
            owner: owner.clone(),
            holds,
        })
    }

    fn parameters(&self) -> Vec<(Arc<Parameter>, String)> {
        let root = Path::ROOT;
        let mut parameters = Parameters(Vec::new());
        let Ok(()) = T::visit(&self.tree, &mut parameters, self.path(&root));
        let mut seen = BTreeSet::new();
        parameters
            .0
            .into_iter()
            .filter(|(parameter, _)| seen.insert(Arc::as_ptr(parameter).addr()))
            .collect()
    }

    fn path<'a>(&'a self, root: &'a Path<'a>) -> Path<'a> {
        if self.name.is_empty() {
            *root
        } else {
            root.field(&self.name)
        }
    }
}

impl<T: Residency> Clone for Resident<T> {
    fn clone(&self) -> Self {
        let Ok(tree) = T::map(&self.tree, &mut Share, Path::ROOT);
        Self {
            tree,
            name: self.name.clone(),
        }
    }
}

impl<'r, T: Residency> Admitted<'r, T> {
    /// Consuming admission makes cancellation release its holds, including a cancelled upload wait.
    pub(crate) async fn landed(self) -> io::Result<Ready<'r, T>> {
        for (path, upload) in &self.holds.uploads {
            upload
                .ended()
                .await
                .map_err(|error| io::Error::other(format!("{path}: its upload failed: {error}")))?;
        }
        Ok(Ready {
            resident: self.resident,
            owner: self.owner,
            holds: self.holds,
        })
    }
}

impl<T: Residency> Ready<'_, T> {
    pub(super) fn check_owner(&self, owner: &Owner) -> io::Result<()> {
        if !self.owner.opened_on(&owner.device) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the admitted call belongs to a different opened context",
            ));
        }
        Ok(())
    }

    /// The caller binds `device` to this call's retained owner at the engine context boundary.
    /// Scoped aliases cannot escape through borrowed outputs. Accepted work is registered even
    /// when a later submission fails or the callback unwinds.
    pub(super) fn launch<O>(
        mut self,
        device: &mut Device,
        call: impl for<'m, 'w> FnOnce(&mut Launches<'_>, &'m T::In<&'w Hbm>) -> io::Result<O>,
    ) -> io::Result<O> {
        if device.placement() != &self.owner.placement {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the admitted device placement changed",
            ));
        }
        let root = Path::ROOT;
        let mut loans = Loans::default();
        let hbm = T::map(
            &self.resident.tree,
            &mut Lend {
                owner: &self.owner,
                loans: &mut loans,
            },
            self.resident.path(&root),
        )?;
        let Ok(borrowed) = T::map(&hbm, &mut Borrow, Path::ROOT);
        let mut launches = Launches {
            device,
            launched: &mut self.holds.launched,
        };
        call(&mut launches, &borrowed)
    }
}

impl<const CHIPS: usize> Launches<'_, CHIPS> {
    /// Reborrow the submission scope for the entry's compile-time chip count.
    pub(crate) fn on<const N: usize>(&mut self) -> Launches<'_, N> {
        Launches {
            device: self.device,
            launched: self.launched,
        }
    }

    #[cfg(test)]
    pub(crate) fn device(&self) -> &Device {
        self.device
    }

    #[cfg(test)]
    pub(crate) fn forward<M, F, I, E, S, O>(
        &mut self,
        module: M,
        entry: F,
        env: E,
        state: S,
        input: I,
    ) -> io::Result<O>
    where
        M: DeviceSend + furiosa_opt_models::Module<I, E, S, Output = O> + Sync,
        I: DeviceSend + Send,
        E: DeviceSend + Send,
        S: DeviceSend + Send,
        O: furiosa_opt_std::runtime::DeviceOutput,
        F: Copy
            + Sync
            + for<'d, 'm> DeviceFn<
                (&'d mut Device, (&'m M, PhantomData<[(); CHIPS]>), E, S, I),
                Output = O,
            >,
    {
        let (output, _) = self.forward_tracked(module, entry, env, state, input)?;
        Ok(output)
    }

    /// Retain completion both for the weight readers and for the caller's input/scratch owner.
    /// Request cleanup must drain that completion before releasing or reusing those buffers.
    pub(crate) fn forward_tracked<M, F, I, E, S, O>(
        &mut self,
        module: M,
        entry: F,
        env: E,
        state: S,
        input: I,
    ) -> io::Result<(O, Arc<Completion>)>
    where
        M: DeviceSend + furiosa_opt_models::Module<I, E, S, Output = O> + Sync,
        I: DeviceSend + Send,
        E: DeviceSend + Send,
        S: DeviceSend + Send,
        O: furiosa_opt_std::runtime::DeviceOutput,
        F: Copy
            + Sync
            + for<'d, 'm> DeviceFn<
                (&'d mut Device, (&'m M, PhantomData<[(); CHIPS]>), E, S, I),
                Output = O,
            >,
    {
        let (output, work) = crate::Function::new(module, entry)
            .on::<CHIPS>()
            .forward_tracked(self.device, env, state, input)?;
        let work = Arc::new(work);
        self.launched.push(Arc::clone(&work));
        Ok((output, work))
    }

    /// Submit into a retained caller-owned workspace slot and register accepted weight readers.
    /// The caller keeps that slot and every input alive until the returned completion ends.
    pub(crate) fn forward_into_tracked<M, F, I, E, S, O>(
        &mut self,
        module: M,
        entry: F,
        env: E,
        state: S,
        input: I,
        output: &mut O,
    ) -> io::Result<Arc<Completion>>
    where
        M: DeviceSend + furiosa_opt_models::Module<I, E, S, Output = O> + Sync,
        I: DeviceSend + Send,
        E: DeviceSend + Send,
        S: DeviceSend + Send,
        O: furiosa_opt_std::runtime::DeviceOutput,
        F: Copy
            + Sync
            + for<'d, 'm> DeviceFn<
                (&'d mut Device, (&'m M, PhantomData<[(); CHIPS]>), E, S, I),
                Output = O,
            >,
    {
        let work = Arc::new(
            crate::Function::new(module, entry)
                .on::<CHIPS>()
                .forward_into_tracked(self.device, env, state, input, output)?,
        );
        self.launched.push(Arc::clone(&work));
        Ok(work)
    }

    /// Record already accepted work; submission refusals must return without adding a record.
    #[cfg(test)]
    fn accepted(&mut self, work: Arc<Completion>) {
        self.launched.push(work);
    }
}

impl Drop for Holds {
    fn drop(&mut self) {
        for parameter in &self.parameters {
            parameter.release(&self.launched);
        }
    }
}

impl Drop for Loans {
    fn drop(&mut self) {
        for parameter in &self.0 {
            parameter.release(&[]);
        }
    }
}

impl Visit<Cells> for Parameters {
    type Error = Infallible;
    fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
        &mut self,
        cell: &Cell<D, C, S>,
        path: Path<'_>,
    ) -> Result<(), Infallible> {
        self.0.push((cell.parameter().clone(), path.to_string()));
        Ok(())
    }
}

impl<'t> Leaves<'t, Cells, Cells> for Share {
    type Error = Infallible;
    fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
        &mut self,
        cell: &'t Cell<D, C, S>,
        _: Path<'_>,
    ) -> Result<Cell<D, C, S>, Infallible> {
        Ok(cell.clone())
    }
}

impl Visit<Cells> for Pin {
    type Error = io::Error;
    fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
        &mut self,
        cell: &Cell<D, C, S>,
        path: Path<'_>,
    ) -> io::Result<()> {
        cell.parameter()
            .pin()
            .map_err(|error| io::Error::new(error.kind(), format!("{path}: {error}")))
    }
}

impl Visit<Cells> for Upload<'_> {
    type Error = io::Error;
    fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
        &mut self,
        cell: &Cell<D, C, S>,
        path: Path<'_>,
    ) -> io::Result<()> {
        cell.upload(self.0, &path.to_string())
    }
}

impl<'t> Leaves<'t, Cells, Hbm> for Lend<'_> {
    type Error = io::Error;
    fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
        &mut self,
        cell: &'t Cell<D, C, S>,
        path: Path<'_>,
    ) -> io::Result<HbmTensor<D, C, S>> {
        let (buffer, upload) = cell.parameter().lend(self.owner, &path.to_string())?;
        self.loans.0.push(cell.parameter().clone());
        if upload.is_some() {
            return Err(io::Error::other(format!(
                "{path}: upload is still pending after admission waited"
            )));
        }
        buffer
            .alias(..)
            .and_then(HbmTensor::try_from)
            .map_err(io::Error::other)
    }
}

impl<'t> Leaves<'t, Hbm, &'t Hbm> for Borrow {
    type Error = Infallible;
    fn leaf<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
        &mut self,
        tensor: &'t HbmTensor<D, C, S>,
        _: Path<'_>,
    ) -> Result<&'t HbmTensor<D, C, S>, Infallible> {
        Ok(tensor)
    }
}

fn retire(closed: Vec<(Arc<Parameter>, Closing)>) -> io::Result<()> {
    let drained: Vec<_> = closed
        .iter()
        .filter(|(_, closing)| matches!(closing, Closing::Drain))
        .map(|(parameter, _)| parameter.drain())
        .collect();
    let mut failure = drained.iter().find_map(|drained| drained.failure.clone());
    let mut seen = BTreeSet::new();
    for work in drained.iter().flat_map(|drained| &drained.work) {
        if seen.insert(Arc::as_ptr(work).addr())
            && let Err(error) = work.wait()
        {
            failure.get_or_insert(error);
        }
    }
    drop(drained);
    // Reopen every copy owned by this eviction before joining shared retirements, so this call
    // holds no admission closed while waiting for another evictor.
    for (parameter, closing) in &closed {
        if matches!(closing, Closing::Drain) {
            parameter.open();
        }
    }
    for (parameter, closing) in &closed {
        if matches!(closing, Closing::Settle) {
            parameter.settle();
        }
    }
    failure.map_or(Ok(()), |error| Err(io::Error::other(error)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_models::Module;
    use furiosa_opt_models::load::{Context, Load};
    use furiosa_opt_models::tensor::memory::Memory;
    use furiosa_opt_models::tensor::parallel::Shard;
    use std::future::Future;
    use std::sync::{Mutex, mpsc};
    use std::task::{Poll, Waker};
    use std::thread;
    use std::time::Duration;

    axes![Features = 8];
    type Chips = m![{ Symbol<Shard<4>> }];
    type Rows = HbmTensor<f32, Chips, m![Features]>;

    #[derive(DeviceSend, Residency)]
    #[residency(memory(X))]
    struct Weight<X: Memory> {
        weight: X::Tensor<f32, Chips, m![Features]>,
    }

    impl<'s> Module<(), (), &'s mut ()> for Weight<&Hbm> {
        type Output = Rows;
        fn forward(&self, device: &mut Device, _: (), _: &'s mut (), _: ()) -> Rows {
            self.weight.to_hbm(&mut device.tdma)
        }
    }

    #[derive(Residency)]
    #[residency(children(A, B))]
    struct Branches<A, B> {
        first: A,
        second: B,
    }

    fn weight(value: f32) -> Weight<Cells> {
        Weight {
            weight: Load::load(
                &mut Context::new(None),
                HostTensor::<f32, m![Features]>::from_vec([value; 8]),
            )
            .unwrap(),
        }
    }

    fn hardware() -> (Arc<Mutex<Device>>, Owner) {
        let device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
        let placement = device.placement().clone();
        let device = Arc::new(Mutex::new(device));
        let owner = Owner {
            placement,
            device: device.clone(),
        };
        (device, owner)
    }

    fn finish_channel() -> (Arc<Completion>, tokio::sync::oneshot::Sender<()>) {
        let (finish, finished) = tokio::sync::oneshot::channel();
        let work = Arc::new(Completion::new(async move {
            finished.await.map_err(|_| {
                furiosa_opt_std::Error::Invalid("the test cancelled its controlled work")
            })
        }));
        (work, finish)
    }

    #[test]
    fn subtree_views_share_only_the_selected_parameter_cells() {
        type Tree = Branches<Weight<Hbm>, Weight<Hbm>>;
        let resident = Resident::<Tree>::new(Branches {
            first: weight(1.),
            second: weight(2.),
        });
        let part = resident.part::<Weight<Hbm>>("second", |tree| &tree.second);
        let shared = part.clone();
        assert!(Arc::ptr_eq(
            part.tree.weight.parameter(),
            resident.tree.second.weight.parameter()
        ));
        assert!(Arc::ptr_eq(
            part.tree.weight.parameter(),
            shared.tree.weight.parameter()
        ));
        assert!(!Arc::ptr_eq(
            resident.tree.first.weight.parameter(),
            resident.tree.second.weight.parameter()
        ));
        let parameters = part.parameters();
        assert_eq!(parameters.len(), 1);
        assert_eq!(parameters[0].1, "second.weight");
    }

    /// The exact typed tree is lent to real four-chip work; eviction through another view then
    /// frees the shared copy, and the same host source can run another request after reloading.
    #[test]
    #[ignore = "requires four explicitly selected RNGD chips and page-locked host memory"]
    fn lends_a_typed_tree_to_tracked_device_work_and_reloads() {
        let (device, owner) = hardware();
        let resident = Resident::<Weight<Hbm>>::new(weight(3.5));
        let view = resident.clone();
        resident.pin().unwrap();
        let place = |source: &super::super::source::Source| {
            super::super::upload::place(&mut device.lock().unwrap(), source)
        };
        let issue = |image: &super::super::source::Image| {
            super::super::upload::issue(&mut device.lock().unwrap(), image)
        };
        let destination = Destination {
            owner: owner.clone(),
            place: &place,
            issue: &issue,
        };
        for _ in 0..2 {
            resident.to_hbm(&destination).unwrap();
            let ready =
                futures_executor::block_on(resident.admit(&owner).unwrap().landed()).unwrap();
            let output = ready
                .launch(&mut device.lock().unwrap(), |launches, tree| {
                    launches
                        .on::<4>()
                        .forward(tree, crate::forward, (), &mut (), ())
                })
                .unwrap();
            view.to_host().unwrap();
            assert!(resident.tree.weight.parameter().owner().is_none());
            let values = futures_executor::block_on(
                output
                    .to_host::<m![{ Chips }, Features]>(&mut device.lock().unwrap().pdma)
                    .submit(),
            )
            .unwrap()
            .into_vec();
            assert_eq!(values, vec![3.5; 4 * Features::SIZE]);
        }
    }

    #[test]
    #[ignore = "requires four explicitly selected RNGD chips and page-locked host memory"]
    fn cancelling_an_upload_wait_releases_admission_without_holding_the_device() {
        let (device, owner) = hardware();
        let resident = Resident::<Weight<Hbm>>::new(weight(1.));
        resident.pin().unwrap();
        let (upload, finish) = finish_channel();
        let place = |_: &super::super::source::Source| panic!("the source was pinned");
        let issue = |image: &super::super::source::Image| {
            let buffer =
                HbmBuffer::alloc(&device.lock().unwrap(), image.len()).map_err(io::Error::other)?;
            Ok((buffer, Some(upload.clone())))
        };
        resident
            .to_hbm(&Destination {
                owner: owner.clone(),
                place: &place,
                issue: &issue,
            })
            .unwrap();
        let mut waiting = Box::pin(resident.admit(&owner).unwrap().landed());
        assert!(matches!(
            waiting
                .as_mut()
                .poll(&mut std::task::Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        assert!(
            device.try_lock().is_ok(),
            "upload waits borrow no device lock"
        );
        drop(waiting);
        finish.send(()).unwrap();
        resident.to_host().unwrap();
        assert!(resident.tree.weight.parameter().owner().is_none());
    }

    #[test]
    #[ignore = "requires four explicitly selected RNGD chips"]
    fn a_later_call_failure_or_panic_keeps_accepted_work_registered() {
        let (device, owner) = hardware();
        let resident = Resident::<Weight<Hbm>>::new(weight(1.));
        let place = |source: &super::super::source::Source| {
            super::super::upload::place(&mut device.lock().unwrap(), source)
        };
        let issue = |_: &super::super::source::Image| panic!("the source was not pinned");
        let destination = Destination {
            owner: owner.clone(),
            place: &place,
            issue: &issue,
        };
        for unwind in [false, true] {
            resident.to_hbm(&destination).unwrap();
            let (accepted, finish) = finish_channel();
            let ready =
                futures_executor::block_on(resident.admit(&owner).unwrap().landed()).unwrap();
            let mut guard = device.lock().unwrap();
            let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ready.launch(&mut guard, |launches, _| {
                    launches.accepted(accepted);
                    if unwind {
                        panic!("a later callback panicked");
                    }
                    Err::<(), _>(io::Error::other("a later submission was refused"))
                })
            }));
            drop(guard);
            if unwind {
                assert!(failed.is_err());
            } else {
                assert!(failed.unwrap().is_err());
            }
            let moved = resident.clone();
            let (done, ended) = mpsc::channel();
            let waiter = thread::spawn(move || {
                done.send(moved.to_host()).unwrap();
            });
            assert!(matches!(
                ended.recv_timeout(Duration::from_millis(100)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ));
            finish.send(()).unwrap();
            ended.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
            waiter.join().unwrap();
            assert!(resident.tree.weight.parameter().owner().is_none());
        }
    }

    #[test]
    #[ignore = "requires four explicitly selected RNGD chips"]
    fn partial_admission_refusal_releases_prior_holds() {
        type Tree = Branches<Weight<Hbm>, Weight<Hbm>>;
        let (device, owner) = hardware();
        let resident = Resident::<Tree>::new(Branches {
            first: weight(1.),
            second: weight(2.),
        });
        let first = resident.part::<Weight<Hbm>>("first", |tree| &tree.first);
        let place = |source: &super::super::source::Source| {
            super::super::upload::place(&mut device.lock().unwrap(), source)
        };
        let issue = |_: &super::super::source::Image| panic!("the source was not pinned");
        first
            .to_hbm(&Destination {
                owner: owner.clone(),
                place: &place,
                issue: &issue,
            })
            .unwrap();
        assert!(
            resident.admit(&owner).is_err(),
            "the second parameter is still on the host"
        );
        let parameter = first.tree.weight.parameter();
        assert!(matches!(parameter.close(), Closing::Drain));
        assert!(
            parameter.lend(&owner, "first.weight").is_err(),
            "no prior admission remains to lend under closed admission"
        );
        let drained = parameter.drain();
        drop(drained);
        parameter.open();
    }

    /// Every selected copy retires before a cached reader failure is returned; retry starts
    /// with new copies and no failure carried over from the retired ones.
    #[test]
    #[ignore = "requires four explicitly selected RNGD chips"]
    fn a_reader_failure_cleans_every_selected_copy_before_retry() {
        type Tree = Branches<Weight<Hbm>, Weight<Hbm>>;
        let (device, owner) = hardware();
        let resident = Resident::<Tree>::new(Branches {
            first: weight(1.),
            second: weight(2.),
        });
        let place = |source: &super::super::source::Source| {
            super::super::upload::place(&mut device.lock().unwrap(), source)
        };
        let issue = |_: &super::super::source::Image| panic!("the source was not pinned");
        let destination = Destination {
            owner: owner.clone(),
            place: &place,
            issue: &issue,
        };
        for unwind in [false, true] {
            resident.to_hbm(&destination).unwrap();
            let ready =
                futures_executor::block_on(resident.admit(&owner).unwrap().landed()).unwrap();
            ready
                .launch(&mut device.lock().unwrap(), |launches, _| {
                    launches.accepted(Arc::new(Completion::new(async move {
                        if unwind {
                            panic!("controlled accepted-reader failure");
                        }
                        Err(furiosa_opt_std::Error::Invalid(
                            "controlled accepted-reader failure",
                        ))
                    })));
                    Ok(())
                })
                .unwrap();
            let error = resident.to_host().unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("controlled accepted-reader failure")
            );
            assert!(
                resident
                    .parameters()
                    .iter()
                    .all(|(parameter, _)| parameter.owner().is_none())
            );
            resident.to_hbm(&destination).unwrap();
            drop(futures_executor::block_on(resident.admit(&owner).unwrap().landed()).unwrap());
            resident.to_host().unwrap();
        }
    }

    /// Opposite tree orders and shared parameter cells still permit concurrent eviction.
    #[test]
    #[ignore = "requires four explicitly selected RNGD chips"]
    fn concurrent_eviction_through_reordered_shared_views_finishes() {
        type Tree = Branches<Weight<Hbm>, Weight<Hbm>>;
        let (device, owner) = hardware();
        let resident = Resident::<Tree>::new(Branches {
            first: weight(1.),
            second: weight(2.),
        });
        let reversed = Resident::<Tree>::new(Branches {
            first: Weight {
                weight: resident.tree.second.weight.clone(),
            },
            second: Weight {
                weight: resident.tree.first.weight.clone(),
            },
        });
        let place = |source: &super::super::source::Source| {
            super::super::upload::place(&mut device.lock().unwrap(), source)
        };
        let issue = |_: &super::super::source::Image| panic!("the source was not pinned");
        resident
            .to_hbm(&Destination {
                owner: owner.clone(),
                place: &place,
                issue: &issue,
            })
            .unwrap();
        let (work, finish) = finish_channel();
        let ready = futures_executor::block_on(resident.admit(&owner).unwrap().landed()).unwrap();
        ready
            .launch(&mut device.lock().unwrap(), |launches, _| {
                launches.accepted(work);
                Ok(())
            })
            .unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let (started, starting) = mpsc::channel();
        let (done, ended) = mpsc::channel();
        let threads: Vec<_> = [resident.clone(), reversed]
            .into_iter()
            .map(|view| {
                let barrier = barrier.clone();
                let started = started.clone();
                let done = done.clone();
                thread::spawn(move || {
                    barrier.wait();
                    started.send(()).unwrap();
                    done.send(view.to_host()).unwrap();
                })
            })
            .collect();
        for _ in 0..2 {
            starting.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        assert!(matches!(
            ended.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        finish.send(()).unwrap();
        for _ in 0..2 {
            ended.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert!(
            resident
                .parameters()
                .iter()
                .all(|(parameter, _)| parameter.owner().is_none())
        );
    }
}
