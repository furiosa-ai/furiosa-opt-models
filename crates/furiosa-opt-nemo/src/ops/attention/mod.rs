pub mod cache;
pub mod flash;
pub mod full;
pub mod gqa;
pub mod rope;

/// Width constructors a model module names its attention widths with, and the key/value pair
/// each cached token stores; head counts, head width and window envelope are the model's own.
pub mod geometry {

    use furiosa_opt_std::prelude::*;

    axes![Kv = 2];

    /// Hidden feature width of a decoder.
    #[derive(Clone, Debug)]
    pub struct Hidden<const N: usize>;

    impl<const N: usize> AxisName for Hidden<N> {
        const NAME: Ident = Ident::new("Hidden");
        const SIZE: usize = N;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use furiosa_opt_std::prelude::*;

    // The head width attention tests run at.
    axes![Head = 128];
}
