//! What a leaf's phases take: the input that begins a phase, and the places in HBM a tile's rows
//! move through.

/// A place in HBM whose rows a leaf reads on chip: [`Fill::fill`] moves the rows of one tile `T`
/// on chip.
pub(crate) trait Fill<D, T> {
    /// The tile's rows, on chip.
    type Rows;
    fn fill(&self, context: &mut D, tile: T) -> Self::Rows;
}

/// A place in HBM a leaf writes from chip: [`Spill::spill`] stores the rows `R` of one tile `T`.
pub(crate) trait Spill<D, R, T> {
    fn spill(&mut self, context: &mut D, rows: R, tile: T);
}

/// A module's input taken up to its first boundary, where it returns its state; given back, the
/// state runs the next step, and the state types refuse any other order at compile time.
pub struct Begin<T>(pub T);
