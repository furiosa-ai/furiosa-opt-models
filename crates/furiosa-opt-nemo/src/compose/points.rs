//! Other modules' transfers placed at a running module's points.

use furiosa_opt_std::prelude::Device;

use super::Spill;

/// Other modules' transfers that a running module, their host, issues at its points: a
/// successor's weights ahead of its loop, or a predecessor's store held back to where the DMA
/// queue idles.
///
/// The host calls [`Points::at`] at each of its points, in order, once on every path, and
/// [`Points::finish`] issues the stages of the points it did not reach as it ends, so each stage
/// issues exactly once. A stage at a point the host picks at runtime issues in place: where the
/// host calls [`Points::between`].
pub trait Points<D>: Sized {
    /// The guests past the host's next point.
    type Next: Points<D, Output = Self::Output>;
    /// What the guests leave once every stage has issued: landed weights, or `()`.
    type Output;

    /// The host's next point: issues the stages placed there.
    #[must_use = "a guest not handed on never issues"]
    fn at(self, context: &mut D) -> Self::Next;

    /// A point between the host's last point and its next, reached at most once: issues the
    /// in-place stages of the guests whose points the host has passed.
    fn between(&mut self, _: &mut D) {}

    /// The host has ended: issues every stage it did not reach, in order, and returns what the
    /// guests leave.
    fn finish(self, context: &mut D) -> Self::Output;
}

/// No guest.
impl<D> Points<D> for () {
    type Next = ();
    type Output = ();

    fn at(self, _: &mut D) {}

    fn finish(self, _: &mut D) {}
}

/// A guest whose stages have all issued, holding what they landed.
pub struct Issued<T>(pub T);

impl<D, T> Points<D> for Issued<T> {
    type Next = Self;
    type Output = T;

    fn at(self, _: &mut D) -> Self {
        self
    }

    fn finish(self, _: &mut D) -> T {
        self.0
    }
}

/// A spill deferred to a host's point: the host issues it in place, where it calls
/// [`Points::between`], so the store queues behind the host's own transfers; a host that reaches
/// no such point does not issue it.
pub struct Deferred<'a, H, R, T> {
    pub into: &'a mut H,
    pub rows: R,
    pub tile: T,
}

impl<D, H: Spill<D, R, T>, R: Copy, T: Copy> Points<D> for Deferred<'_, H, R, T> {
    type Next = Self;
    type Output = ();

    fn at(self, _: &mut D) -> Self {
        self
    }

    fn between(&mut self, context: &mut D) {
        self.into.spill(context, self.rows, self.tile);
    }

    fn finish(self, _: &mut D) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Launch([&'static str; 4]);

    impl Spill<Vec<&'static str>, &'static str, usize> for Launch {
        fn spill(&mut self, trace: &mut Vec<&'static str>, rows: &'static str, tile: usize) {
            trace.push("spill");
            self.0[tile] = rows;
        }
    }

    /// A deferred spill stores its tile's rows where the host points in place, not where it
    /// passes a point or ends.
    #[test]
    fn deferred_spills_store_in_place() {
        let mut launch = Launch([""; 4]);
        let mut trace = Vec::new();
        let points = Deferred {
            into: &mut launch,
            rows: "x",
            tile: 2,
        };
        let mut points = points.at(&mut trace);
        trace.push("host");
        points.between(&mut trace);
        points.finish(&mut trace);
        assert_eq!(
            (trace, launch.0),
            (vec!["host", "spill"], ["", "", "x", ""])
        );
    }
}

/// A guest's transfers in three stages: prepared and started at the host's next point, the rest
/// issued where the host calls [`Points::between`], and loaded as the host ends.
pub trait Prefetch {
    type Output;
    type Prepared;
    type Started;

    fn prepare(self, context: &mut Device) -> Self::Prepared;
    fn start(prepared: Self::Prepared, context: &mut Device) -> Self::Started;
    /// Issues the transfers `start` leaves for a point the caller picks; `load` reads them.
    fn rest(_: &mut Self::Started, _: &mut Device) {}
    fn load(started: Self::Started, context: &mut Device) -> Self::Output;
}

impl Prefetch for () {
    type Output = ();
    type Prepared = ();
    type Started = ();

    fn prepare(self, _: &mut Device) {}
    fn start(_: (), _: &mut Device) {}
    fn load(_: (), _: &mut Device) {}
}

/// A [`Prefetch`] as a host's guest: the host's next point prepares and starts it.
pub struct Prefetched<P>(pub P);

impl<P: Prefetch> Points<Device> for Prefetched<P> {
    type Next = Prefetching<P>;
    type Output = P::Output;

    fn at(self, context: &mut Device) -> Prefetching<P> {
        let prepared = self.0.prepare(context);
        Prefetching {
            started: P::start(prepared, context),
        }
    }

    fn finish(self, context: &mut Device) -> P::Output {
        let mut started = self.at(context);
        started.between(context);
        started.finish(context)
    }
}

/// A started [`Prefetch`]: its rest issues where the host calls [`Points::between`], which the
/// host does once on every path, and it loads as the host ends. A device-side flag cannot stand in
/// for that count: the device keeps no flag set on one branch arm past the merge.
pub struct Prefetching<P: Prefetch> {
    started: P::Started,
}

impl<P: Prefetch> Points<Device> for Prefetching<P> {
    type Next = Self;
    type Output = P::Output;

    fn at(self, _: &mut Device) -> Self {
        self
    }

    fn between(&mut self, context: &mut Device) {
        P::rest(&mut self.started, context);
    }

    fn finish(self, context: &mut Device) -> P::Output {
        P::load(self.started, context)
    }
}
