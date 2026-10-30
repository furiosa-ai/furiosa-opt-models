//! Layer collections applied in one fixed interleaved order, each layer with its own state slot.

use super::Collection;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::{Module, Order};
use furiosa_opt_std::prelude::{Device, DeviceSend, unroll};
use std::marker::ConstParamTy;

/// Which collection supplies the next layer in a compiled composition.
#[derive(Clone, Copy, PartialEq, Eq, ConstParamTy)]
pub enum Branch {
    First,
    Second,
    Third,
}

impl Branch {
    /// The order of `N` layers whose every `period`-th layer, counting from one, is the second
    /// collection's.
    pub const fn periodic<const N: usize>(period: usize) -> [Self; N] {
        assert!(period > 0 && N.is_multiple_of(period));
        let mut order = [Self::First; N];
        let mut index = period - 1;
        while index < N {
            order[index] = Self::Second;
            index += period;
        }
        order
    }
}

/// The layers of a tuple of collections in the fixed order `ORDER`. The env and the state hold one
/// part a collection: each layer reads its collection's env and takes the next slot of its array.
#[derive(DeviceSend)]
pub struct Interleave<C, const ORDER: &'static [Branch]> {
    pub layers: C,
}

impl<A, B, const FIRST: usize, const SECOND: usize, const ORDER: &'static [Branch]>
    Interleave<([A; FIRST], [B; SECOND]), ORDER>
{
    /// Positions in the complete composition, in each collection's storage order.
    pub const POSITIONS: ([usize; FIRST], [usize; SECOND]) = {
        let positions = Order::<FIRST, SECOND>::new(ORDER).positions;
        (positions.0, positions.1)
    };
}

/// Each layer under its position in the order, as a checkpoint numbers the layers of every
/// collection together; the collections add no segment, so a family names them.
impl<'a, A, B, const FIRST: usize, const SECOND: usize, const ORDER: &'static [Branch]>
    Load<&'a Checkpoint> for Interleave<([A; FIRST], [B; SECOND]), ORDER>
where
    A: Load<&'a Checkpoint>,
    B: Load<&'a Checkpoint>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let (first, second) = Self::POSITIONS;
            let mut children = Children::default();
            let first = cx.field("0", "", |cx| placed(first, |_, cx| A::load(cx, from), cx));
            let first = children.ask(first)?;
            let second = cx.field("1", "", |cx| placed(second, |_, cx| B::load(cx, from), cx));
            let second = children.ask(second)?;
            let layers = children.node(first.zip(second))?;
            Ok(Self { layers })
        })
    }
}

/// Each layer written in place under its position in the order.
impl<'t, 's, A, B, const FIRST: usize, const SECOND: usize, const ORDER: &'static [Branch]>
    Load<(
        &'t mut Interleave<([A; FIRST], [B; SECOND]), ORDER>,
        &'s Checkpoint,
    )> for &'t mut Interleave<([A; FIRST], [B; SECOND]), ORDER>
where
    for<'u> &'u mut A: Load<(&'u mut A, &'s Checkpoint)>,
    for<'u> &'u mut B: Load<(&'u mut B, &'s Checkpoint)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut Interleave<([A; FIRST], [B; SECOND]), ORDER>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        let (first, second) = Interleave::<([A; FIRST], [B; SECOND]), ORDER>::POSITIONS;
        context.checkpoint(from, |cx| {
            cx.field("0", "", |cx| {
                into.layers
                    .0
                    .iter_mut()
                    .zip(first)
                    .enumerate()
                    .try_for_each(|(k, (layer, at))| {
                        cx.field(&k.to_string(), &at.to_string(), |cx| {
                            <&mut A as Load<_>>::load(cx, (layer, from)).map(drop)
                        })
                    })
            })?;
            cx.field("1", "", |cx| {
                into.layers
                    .1
                    .iter_mut()
                    .zip(second)
                    .enumerate()
                    .try_for_each(|(k, (layer, at))| {
                        cx.field(&k.to_string(), &at.to_string(), |cx| {
                            <&mut B as Load<_>>::load(cx, (layer, from)).map(drop)
                        })
                    })
            })
        })?;
        Ok(into)
    }
}

impl<'a, A, B, P, Q, const FIRST: usize, const SECOND: usize, const ORDER: &'static [Branch]>
    Load<&'a Interleave<([P; FIRST], [Q; SECOND]), ORDER>>
    for Interleave<([A; FIRST], [B; SECOND]), ORDER>
where
    A: Load<&'a P>,
    B: Load<&'a Q>,
{
    fn load(
        context: &mut Context<'_>,
        from: &'a Interleave<([P; FIRST], [Q; SECOND]), ORDER>,
    ) -> Result<Self, LoadError> {
        let (first, second) = Self::POSITIONS;
        let first = context.field("0", "", |cx| {
            placed(first, |k, cx| A::load(cx, &from.layers.0[k]), cx)
        })?;
        let second = context.field("1", "", |cx| {
            placed(second, |k, cx| B::load(cx, &from.layers.1[k]), cx)
        })?;
        Ok(Self {
            layers: (first, second),
        })
    }
}

impl<'t, A, B, const FIRST: usize, const SECOND: usize, const ORDER: &'static [Branch]>
    Load<(
        &'t mut Interleave<([A; FIRST], [B; SECOND]), ORDER>,
        Interleave<([A; FIRST], [B; SECOND]), ORDER>,
    )> for &'t mut Interleave<([A; FIRST], [B; SECOND]), ORDER>
where
    for<'u> &'u mut A: Load<(&'u mut A, A)>,
    for<'u> &'u mut B: Load<(&'u mut B, B)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut Interleave<([A; FIRST], [B; SECOND]), ORDER>,
            Interleave<([A; FIRST], [B; SECOND]), ORDER>,
        ),
    ) -> Result<Self, LoadError> {
        let (first, second) = Interleave::<([A; FIRST], [B; SECOND]), ORDER>::POSITIONS;
        let (left, right) = from.layers;
        for (k, ((layer, value), at)) in into.layers.0.iter_mut().zip(left).zip(first).enumerate() {
            context.field("0", "", |cx| {
                cx.field(&k.to_string(), &at.to_string(), |cx| {
                    <&mut A as Load<_>>::load(cx, (layer, value)).map(drop)
                })
            })?;
        }
        for (k, ((layer, value), at)) in into.layers.1.iter_mut().zip(right).zip(second).enumerate()
        {
            context.field("1", "", |cx| {
                cx.field(&k.to_string(), &at.to_string(), |cx| {
                    <&mut B as Load<_>>::load(cx, (layer, value)).map(drop)
                })
            })?;
        }
        Ok(into)
    }
}

// Repeated periods stay rolled; each period is unrolled for its layer types. Device code reads
// the order through constant positions; a slice constant does not lower.
impl<
    's,
    X,
    P,
    Q,
    PState,
    QState,
    PEnv: Copy,
    QEnv: Copy,
    const FIRST: usize,
    const SECOND: usize,
    const ORDER: &'static [Branch],
> Module<X, (PEnv, QEnv), &'s mut ([PState; FIRST], [QState; SECOND])> for Interleave<(P, Q), ORDER>
where
    P: Collection<X, PEnv, &'s mut PState, Output = X>,
    Q: Collection<X, QEnv, &'s mut QState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        (first_env, second_env): (PEnv, QEnv),
        (first, second): &'s mut ([PState; FIRST], [QState; SECOND]),
        mut hidden: X,
    ) -> X {
        const {
            assert!(
                P::COUNT == FIRST && Q::COUNT == SECOND,
                "one state slot for each layer"
            )
        };
        let (positions, period, groups, counts) = const {
            let order = Order::<FIRST, SECOND>::new(ORDER);
            (
                order.positions.0,
                order.period,
                ORDER.len() / order.period,
                order.counts,
            )
        };
        let (mut first, mut second) = (first.iter_mut(), second.iter_mut());
        for group in 0..groups {
            let (mut a, mut b) = (0, 0);
            #[unroll]
            for offset in 0..period {
                match a < counts[0] && positions[a] == offset {
                    true => {
                        let slot = first.next().expect("one slot for each first layer");
                        let index = group * counts[0] + a;
                        hidden = self
                            .layers
                            .0
                            .apply(&mut *device, first_env, slot, hidden, index);
                        a += 1;
                    }
                    false => {
                        let slot = second.next().expect("one slot for each second layer");
                        let index = group * counts[1] + b;
                        hidden = self
                            .layers
                            .1
                            .apply(&mut *device, second_env, slot, hidden, index);
                        b += 1;
                    }
                }
            }
        }
        hidden
    }
}

impl<
    's,
    X,
    P,
    Q,
    R,
    PState,
    QState,
    RState,
    PEnv: Copy,
    QEnv: Copy,
    REnv: Copy,
    const FIRST: usize,
    const SECOND: usize,
    const THIRD: usize,
    const ORDER: &'static [Branch],
> Module<X, (PEnv, QEnv, REnv), &'s mut ([PState; FIRST], [QState; SECOND], [RState; THIRD])>
    for Interleave<(P, Q, R), ORDER>
where
    P: Collection<X, PEnv, &'s mut PState, Output = X>,
    Q: Collection<X, QEnv, &'s mut QState, Output = X>,
    R: Collection<X, REnv, &'s mut RState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: (PEnv, QEnv, REnv),
        (first, second, third): &'s mut ([PState; FIRST], [QState; SECOND], [RState; THIRD]),
        mut hidden: X,
    ) -> X {
        const {
            assert!(
                P::COUNT == FIRST && Q::COUNT == SECOND && R::COUNT == THIRD,
                "one state slot for each layer"
            )
        };
        let (positions, period, groups, counts) = const {
            let order = Order::<FIRST, SECOND, THIRD>::new(ORDER);
            (
                order.positions,
                order.period,
                ORDER.len() / order.period,
                order.counts,
            )
        };
        let mut slots = (first.iter_mut(), second.iter_mut(), third.iter_mut());
        for group in 0..groups {
            let (mut a, mut b, mut c) = (0, 0, 0);
            #[unroll]
            for offset in 0..period {
                if a < counts[0] && positions.0[a] == offset {
                    let slot = slots.0.next().expect("one slot for each first layer");
                    let index = group * counts[0] + a;
                    hidden = self
                        .layers
                        .0
                        .apply(&mut *device, env.0, slot, hidden, index);
                    a += 1;
                } else if b < counts[1] && positions.1[b] == offset {
                    let slot = slots.1.next().expect("one slot for each second layer");
                    let index = group * counts[1] + b;
                    hidden = self
                        .layers
                        .1
                        .apply(&mut *device, env.1, slot, hidden, index);
                    b += 1;
                } else {
                    let slot = slots.2.next().expect("one slot for each third layer");
                    let index = group * counts[2] + c;
                    hidden = self
                        .layers
                        .2
                        .apply(&mut *device, env.2, slot, hidden, index);
                    c += 1;
                }
            }
        }
        hidden
    }
}

/// `N` layers, layer `k` read by `read(k)` under the checkpoint segment `positions[k]` and the tree
/// segment `k`, absent names reported together.
fn placed<L, const N: usize>(
    positions: [usize; N],
    mut read: impl FnMut(usize, &mut Context<'_>) -> Result<L, LoadError>,
    context: &mut Context<'_>,
) -> Result<[L; N], LoadError> {
    let mut children = Children::default();
    let built = positions
        .into_iter()
        .enumerate()
        .map(|(index, position)| {
            let layer = |cx: &mut Context<'_>| read(index, cx);
            children.ask(context.field(&index.to_string(), &position.to_string(), layer))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let built = children.node(built.into_iter().collect::<Option<Vec<L>>>())?;
    Ok(built.try_into().ok().expect("one layer per position"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;

    struct First;
    struct Second;
    struct Third;

    impl<Env> Module<i32, Env, &mut i32> for First {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: Env, state: &mut i32, hidden: i32) -> i32 {
            *state += hidden;
            *state
        }
    }

    impl<Env> Module<i32, Env, &mut i32> for Second {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: Env, state: &mut i32, hidden: i32) -> i32 {
            *state *= hidden;
            *state
        }
    }

    impl Module<i32, i32, &mut ()> for Third {
        type Output = i32;

        fn forward(&self, _: &mut Device, step: i32, _: &mut (), hidden: i32) -> i32 {
            hidden - step
        }
    }

    /// Interleaving gives each layer its collection's env and the next slot of its collection's
    /// state, in order.
    #[test]
    fn preserves_interleaved_states() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        const ORDER: &[Branch] = &[Branch::First, Branch::Second, Branch::First];
        type Model = Interleave<([First; 2], [Second; 1]), ORDER>;
        const { assert!(matches!(Model::POSITIONS, ([0, 2], [1]))) };
        let model = Model {
            layers: ([First, First], [Second]),
        };
        let mut state = ([1, 10], [2]);
        assert_eq!(model.forward(device, ((), ()), &mut state, 3), 18);
        assert_eq!(state, ([4, 18], [8]));
        let empty = Interleave::<([First; 0], [Second; 0]), { &[] }> { layers: ([], []) };
        assert_eq!(empty.forward(device, ((), ()), &mut ([], []), 3), 3);
        let second_only = Interleave::<([First; 0], _), { &[Branch::Second] }> {
            layers: ([], [Second]),
        };
        let mut state = ([], [2]);
        assert_eq!(second_only.forward(device, ((), ()), &mut state, 3), 6);
        assert_eq!(state, ([], [6]));
        const THREE: &[Branch] = &[Branch::First, Branch::Third, Branch::Second, Branch::First];
        let three = Interleave::<_, THREE> {
            layers: ([First, First], [Second], [Third]),
        };
        let mut state = ([1, 10], [2], [()]);
        assert_eq!(three.forward(device, ((), (), 1), &mut state, 3), 16);
        assert_eq!(state, ([4, 16], [6], [()]));
    }
}
