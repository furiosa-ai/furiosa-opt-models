use furiosa_opt_std::prelude::*;

use crate::models::nemo::serving::{Chips, Hidden, mlp, output, qkv, scale};

/// The layer axis of the catalog's layer collection, over NeMo's 40 layers.
pub type Depth = crate::modules::generic::collection::Depth<40>;

#[derive(DeviceSend)]
pub struct Stacked<'a, D: Scalar, S: M> {
    pub tensor: HbmTensorView<'a, D, Chips, m![Depth, { S }]>,
}

/// One layer's slice of a stacked resource, read where it lies.
pub type View<'a, D, S> = HbmTensorView<'a, D, Chips, m![Depth = 1 # Depth, { S }]>;

impl<D: MaterializableScalar, S: M> Stacked<'_, D, S> {
    pub fn at(&self, index: usize) -> View<'_, D, S> {
        self.tensor
            .clone()
            .tile::<m![Depth], 1, m![Depth = 1 # Depth, { S }]>(index)
    }
}

/// Every layer's weights and scales in the layouts their pieces read, packed on the host.
#[derive(DeviceSend)]
pub struct Layers<'a> {
    pub norm: Stacked<'a, bf16, m![Hidden]>,
    pub query: Stacked<'a, f8e4m3, qkv::query::Stored>,
    pub pairs: Stacked<'a, f8e4m3, qkv::contraction::Stored>,
    /// Factor 0 holds the QKV input scale and factor 1 the Q weight scale.
    pub qkv_scales: Stacked<'a, f32, scale::Packed>,
    pub pair_scale: Stacked<'a, f32, qkv::PairScale>,
    pub output: Stacked<'a, f8e4m3, output::Stored>,
    /// Factor 0 holds the O input scale and factor 1 its weight scale.
    pub output_scales: Stacked<'a, f32, scale::Packed>,
    pub post_norm: Stacked<'a, bf16, m![Hidden]>,
    pub gate: Stacked<'a, f8e4m3, mlp::weight::Stored>,
    pub up: Stacked<'a, f8e4m3, mlp::weight::Stored>,
    pub down: Stacked<'a, f8e4m3, mlp::down::Stored>,
    /// Factors 0-5: gate input and weight, up input and weight, down input and weight scales.
    pub mlp_scales: Stacked<'a, f32, scale::Packed>,
}
