//! Other modules' transfers placed at a running module's points.

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
}

/// No guest.
impl<D> Points<D> for () {
    type Next = ();
    type Output = ();

    fn at(self, _: &mut D) {}
}
