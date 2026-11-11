//! Other modules' transfers placed at a running module's points.

/// Other modules' transfers that a running module, their host, issues at its points: a
/// successor's weights ahead of its loop, or a predecessor's store held back to where the DMA
/// queue idles.
///
/// The host calls [`Points::at`] at each of its points, in order, once on every path, and
/// [`Points::finish`] issues the stages of the points it did not reach as it ends, so each stage
/// issues exactly once. A stage at a point the host picks at runtime issues in place: where the
/// host calls [`Points::between`].
pub(crate) trait Points<D>: Sized {
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
pub(crate) struct Issued<T>(pub T);

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

/// Two guests at the same points, each point issuing the first's stages ahead of the second's.
impl<D, A: Points<D>, B: Points<D>> Points<D> for (A, B) {
    type Next = (A::Next, B::Next);
    type Output = (A::Output, B::Output);

    fn at(self, context: &mut D) -> Self::Next {
        (self.0.at(context), self.1.at(context))
    }

    fn between(&mut self, context: &mut D) {
        self.0.between(context);
        self.1.between(context);
    }

    fn finish(self, context: &mut D) -> Self::Output {
        (self.0.finish(context), self.1.finish(context))
    }
}

/// A stage its host issues at its next point, holding what the stage lands.
#[derive(Clone, Copy)]
pub(crate) struct Stage<F>(pub F);

impl<D, T, F: FnOnce(&mut D) -> T> Points<D> for Stage<F> {
    type Next = Issued<T>;
    type Output = T;

    fn at(self, context: &mut D) -> Issued<T> {
        Issued((self.0)(context))
    }

    fn finish(self, context: &mut D) -> T {
        (self.0)(context)
    }
}
