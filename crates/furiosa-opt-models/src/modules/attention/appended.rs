//! A cache stage that appends this call's keys and values to the history, storing them only
//! when the cache is writable.

use super::{Attention, Context, History, Positioned};
use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// History with this call's K/V appended: stored before reading when the cache is writable,
/// joined to the read history when it is read-only.
#[derive(DeviceSend)]
pub struct Appended<Write, Read, Join> {
    pub write: Write,
    pub read: Read,
    pub join: Join,
}

/// A call that stores its keys and values: the state is the layer's view of its cache; the input
/// holds which history attention reads and where this call's keys and values go.
impl<'s, 'b, 'x, Q, K, V, C, W, R, HistoryKeys, HistoryValues, Write, Read, Join, Env: Copy>
    Module<((Q, K, V), (&'x R, &'x W)), Env, &'s mut &'b mut C> for Appended<Write, Read, Join>
where
    Write: for<'d, 'w, 'k> Module<(&'w mut C, &'x W, &'k K, &'k V), Env, &'d mut (), Output = ()>,
    Read:
        for<'d, 'r> Module<(&'r C, &'x R), Env, &'d mut (), Output = (HistoryKeys, HistoryValues)>,
{
    type Output = (Q, History<HistoryKeys, HistoryValues, &'x R>);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        cache: &'s mut &'b mut C,
        ((query, keys, values), (read, slots)): ((Q, K, V), (&'x R, &'x W)),
    ) -> Self::Output {
        self.write.forward(
            &mut *device,
            env,
            &mut (),
            (&mut **cache, slots, &keys, &values),
        );
        let (keys, values) = self.read.forward(device, env, &mut (), (&**cache, read));
        (query, History(keys, values, read))
    }
}

/// A call that reads the cache only: its keys and values join the read history unstored.
impl<
    's,
    'b,
    'x,
    Q,
    K,
    V,
    C,
    R,
    HistoryKeys,
    HistoryValues,
    JoinedKeys,
    JoinedValues,
    Write,
    Read,
    Join,
    Env: Copy,
> Module<((Q, K, V), (&'x R, ())), Env, &'s mut &'b mut C> for Appended<Write, Read, Join>
where
    Read:
        for<'d, 'r> Module<(&'r C, &'x R), Env, &'d mut (), Output = (HistoryKeys, HistoryValues)>,
    Join: for<'d, 'k> Module<
            (&'k HistoryKeys, &'k HistoryValues, &'k K, &'k V),
            Env,
            &'d mut (),
            Output = (JoinedKeys, JoinedValues),
        >,
{
    type Output = (Q, History<JoinedKeys, JoinedValues, &'x R>);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        cache: &'s mut &'b mut C,
        ((query, keys, values), (read, ())): ((Q, K, V), (&'x R, ())),
    ) -> Self::Output {
        let (past_keys, past_values) =
            self.read
                .forward(&mut *device, env, &mut (), (&**cache, read));
        let (keys, values) = self.join.forward(
            device,
            env,
            &mut (),
            (&past_keys, &past_values, &keys, &values),
        );
        (query, History(keys, values, read))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;
    use std::cell::RefCell;

    struct Store;
    struct Read;
    struct Join;

    impl Module<(&mut i32, &(), &i32, &i32), &RefCell<Vec<&'static str>>, &mut ()> for Store {
        type Output = ();

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            (cache, _, key, value): (&mut i32, &(), &i32, &i32),
        ) {
            trace.borrow_mut().push("write");
            *cache = key + value;
        }
    }

    impl Module<(&i32, &()), &RefCell<Vec<&'static str>>, &mut ()> for Read {
        type Output = (i32, i32);

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            (cache, _): (&i32, &()),
        ) -> Self::Output {
            trace.borrow_mut().push("read");
            (*cache, cache + 1)
        }
    }

    impl Module<(&i32, &i32, &i32, &i32), &RefCell<Vec<&'static str>>, &mut ()> for Join {
        type Output = (i32, i32);

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            (past_key, past_value, key, value): (&i32, &i32, &i32, &i32),
        ) -> Self::Output {
            trace.borrow_mut().push("join");
            (past_key + key, past_value + value)
        }
    }

    /// Writable history includes the update; read-only history includes current K/V without mutation.
    #[test]
    fn preserves_shared_history() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let model = Appended {
            write: Store,
            read: Read,
            join: Join,
        };
        let trace = &RefCell::new(Vec::new());
        let mut cache = 0;
        let (query, history) =
            model.forward(device, trace, &mut &mut cache, ((1, 2, 3), (&(), &())));
        assert_eq!(query, 1);
        assert!(matches!(history, History(5, 6, _)));
        assert_eq!(*trace.borrow(), ["write", "read"]);
        trace.borrow_mut().clear();
        let (query, history) =
            model.forward(device, trace, &mut &mut cache, ((1, 2, 3), (&(), ())));
        assert_eq!(query, 1);
        assert!(matches!(history, History(7, 9, _)));
        assert_eq!(*trace.borrow(), ["read", "join"]);
        assert_eq!(cache, 5);
    }
}

/// An appended attention reads its context from the env, whose write operand is where this call
/// stores its keys and values, or `()` for a call that only reads.
impl<
    'x,
    'e,
    X,
    PositionOperand,
    WriteOperand: Copy,
    ReadOperand,
    S,
    Projection,
    Position,
    Write,
    Read,
    Join,
    Attend,
    Project,
    Out,
> Module<&'x X, &'e Context<PositionOperand, WriteOperand, ReadOperand>, S>
    for Attention<Positioned<Projection, Position>, Appended<Write, Read, Join>, Attend, Project>
where
    Self: Module<
            (
                (
                    (&'x X, &'e PositionOperand),
                    (&'e ReadOperand, WriteOperand),
                ),
                (),
            ),
            &'e Context<PositionOperand, WriteOperand, ReadOperand>,
            S,
            Output = Out,
        >,
{
    type Output = Out;

    fn forward(
        &self,
        device: &mut Device,
        context: &'e Context<PositionOperand, WriteOperand, ReadOperand>,
        state: S,
        input: &'x X,
    ) -> Self::Output {
        self.forward(
            device,
            context,
            state,
            (
                ((input, &context.positions), (&context.read, context.write)),
                (),
            ),
        )
    }
}

/// With the routing mask its layer reads, beside the context, the projections read it too.
impl<
    'x,
    'm,
    'e,
    X,
    M,
    PositionOperand,
    WriteOperand: Copy,
    ReadOperand,
    S,
    Projection,
    Position,
    Write,
    Read,
    Join,
    Attend,
    Project,
    Out,
>
    Module<
        &'x X,
        (
            &'m M,
            &'e Context<PositionOperand, WriteOperand, ReadOperand>,
        ),
        S,
    > for Attention<Positioned<Projection, Position>, Appended<Write, Read, Join>, Attend, Project>
where
    Self: Module<
            (
                (
                    ((&'x X, &'m M), &'e PositionOperand),
                    (&'e ReadOperand, WriteOperand),
                ),
                &'m M,
            ),
            (
                &'m M,
                &'e Context<PositionOperand, WriteOperand, ReadOperand>,
            ),
            S,
            Output = Out,
        >,
{
    type Output = Out;

    fn forward(
        &self,
        device: &mut Device,
        env: (
            &'m M,
            &'e Context<PositionOperand, WriteOperand, ReadOperand>,
        ),
        state: S,
        input: &'x X,
    ) -> Self::Output {
        let (mask, context) = env;
        self.forward(
            device,
            env,
            state,
            (
                (
                    ((input, mask), &context.positions),
                    (&context.read, context.write),
                ),
                mask,
            ),
        )
    }
}
