use furiosa_opt_std::prelude::*;

/// Outermost axis of a layer-major parameter tensor.
#[derive(Debug, Clone)]
pub struct Depth<const N: usize>;

impl<const N: usize> AxisName for Depth<N> {
    const NAME: Ident = Ident::new("Layers");
    const SIZE: usize = N;
}
