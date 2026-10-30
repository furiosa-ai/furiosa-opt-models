//! Cache stages over a source's keys and values: projected on every call, or stored once and
//! read back.

use super::History;
use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Projects keys and values from the source on every call, without a cache.
#[derive(DeviceSend)]
pub struct Source<Projection>(pub Projection);

/// The state is the projection's slot.
impl<'x, Q, E, C, K, V, S, Projection, Env: Copy> Module<(Q, (&'x E, C)), Env, S>
    for Source<Projection>
where
    Projection: Module<&'x E, Env, S, Output = (K, V)>,
{
    type Output = (Q, History<K, V, C>);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        (query, (source, context)): (Q, (&'x E, C)),
    ) -> Self::Output {
        let (keys, values) = self.0.forward(device, env, state, source);
        (query, History(keys, values, context))
    }
}

/// Projects and stores the source's keys and values once; later calls read them back.
#[derive(DeviceSend)]
pub struct Stored<Projection, Write, Read> {
    pub projection: Projection,
    pub write: Write,
    pub read: Read,
}

/// The source's keys and values, projected and stored into the memory: the state holds the layer's
/// view of its memory, then the projection's slot.
impl<'s, 'b, 'x, E, C, K, V, ProjectionState, Projection, Write, Read, Env: Copy>
    Module<&'x E, Env, &'s mut (&'b mut C, ProjectionState)> for Stored<Projection, Write, Read>
where
    Projection: Module<&'x E, Env, &'s mut ProjectionState, Output = (K, V)>,
    Write: for<'d, 'k> Module<(&'s mut C, &'k K, &'k V), Env, &'d mut (), Output = ()>,
{
    type Output = (K, V);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (memory, projection): &'s mut (&'b mut C, ProjectionState),
        source: &'x E,
    ) -> (K, V) {
        let (keys, values) = self
            .projection
            .forward(&mut *device, env, projection, source);
        self.write
            .forward(device, env, &mut (), (&mut **memory, &keys, &values));
        (keys, values)
    }
}

/// The call that writes the memory: it projects, stores and attends the source rows' keys and
/// values; the state holds the layer's memory view, then the projection's slot.
impl<'s, 'b, 'x, Q, E, C, ReadOperand, K, V, ProjectionState, Projection, Write, Read, Env: Copy>
    Module<(Q, (ReadOperand, &'x E)), Env, &'s mut (&'b mut C, ProjectionState)>
    for Stored<Projection, Write, Read>
where
    Projection: Module<&'x E, Env, &'s mut ProjectionState, Output = (K, V)>,
    Write: for<'d, 'k> Module<(&'s mut C, &'k K, &'k V), Env, &'d mut (), Output = ()>,
{
    type Output = (Q, History<K, V, ReadOperand>);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &'s mut (&'b mut C, ProjectionState),
        (query, (context, source)): (Q, (ReadOperand, &'x E)),
    ) -> Self::Output {
        let (keys, values) = self.forward(device, env, state, source);
        (query, History(keys, values, context))
    }
}

/// A later call reads the memory the first one wrote: the input holds what attention reads beside
/// it, and no source.
impl<
    's,
    'b,
    Q,
    C,
    ReadOperand,
    HistoryKeys,
    HistoryValues,
    ProjectionState,
    Projection,
    Write,
    Read,
    Env: Copy,
> Module<(Q, (ReadOperand, ())), Env, &'s mut (&'b mut C, ProjectionState)>
    for Stored<Projection, Write, Read>
where
    Read: for<'d> Module<&'s C, Env, &'d mut (), Output = (HistoryKeys, HistoryValues)>,
{
    type Output = (Q, History<HistoryKeys, HistoryValues, ReadOperand>);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (memory, _): &'s mut (&'b mut C, ProjectionState),
        (query, (context, ())): (Q, (ReadOperand, ())),
    ) -> Self::Output {
        let (keys, values) = self.read.forward(device, env, &mut (), &**memory);
        (query, History(keys, values, context))
    }
}

/// A read of memory stored earlier: the state holds the layer's read of the memory, which attention
/// reads beside it, then the memory, whose keys and values attention reads where they are stored.
impl<'s, 'c, 'x, Q, D, K, V, R, Projection, Write, Read, Env>
    Module<(Q, &'x R), Env, &'s mut (D, &'c (K, V))> for Stored<Projection, Write, Read>
{
    type Output = (Q, History<&'c K, &'c V, &'s D>);

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        (read, memory): &'s mut (D, &'c (K, V)),
        (query, _): (Q, &'x R),
    ) -> Self::Output {
        let (keys, values): &'c (K, V) = *memory;
        (query, History(keys, values, &*read))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;
    use std::cell::RefCell;

    struct Projection;
    struct Write;
    struct Read;

    impl Module<&(i32, i32), &RefCell<Vec<&'static str>>, &mut ()> for Projection {
        type Output = (i32, i32);

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            source: &(i32, i32),
        ) -> Self::Output {
            trace.borrow_mut().push("project");
            *source
        }
    }

    impl Module<(&mut (i32, i32), &i32, &i32), &RefCell<Vec<&'static str>>, &mut ()> for Write {
        type Output = ();

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            (cache, key, value): (&mut (i32, i32), &i32, &i32),
        ) {
            trace.borrow_mut().push("write");
            *cache = (*key, *value);
        }
    }

    impl Module<&(i32, i32), &RefCell<Vec<&'static str>>, &mut ()> for Read {
        type Output = (i32, i32);

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            cache: &(i32, i32),
        ) -> Self::Output {
            trace.borrow_mut().push("read");
            *cache
        }
    }

    /// Read calls reuse encoder memory without projecting or writing the source again.
    #[test]
    fn reuses_encoder_history() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let model = Stored {
            projection: Projection,
            write: Write,
            read: Read,
        };
        let trace = &RefCell::new(Vec::new());
        let mut caches = (0, 0);
        let mut state = (&mut caches, ());
        let (query, history) = model.forward(device, trace, &mut state, (3, ((), &(7, 9))));
        assert_eq!(query, 3);
        assert!(matches!(history, History(7, 9, ())));
        assert_eq!(*trace.borrow(), ["project", "write"]);
        trace.borrow_mut().clear();
        let (query, history) = model.forward(device, trace, &mut state, (3, ((), ())));
        assert_eq!(query, 3);
        assert!(matches!(history, History(7, 9, ())));
        assert_eq!(*trace.borrow(), ["read"]);
        assert_eq!(caches, (7, 9));
    }
}
