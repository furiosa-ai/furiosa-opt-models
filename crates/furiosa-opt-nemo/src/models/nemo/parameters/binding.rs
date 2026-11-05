use super::*;
use crate::models::nemo::serving::{head, stack, step};

impl Model {
    /// Borrow the weights in the step's operand layouts; no cells are copied.
    pub fn recipe(&self) -> step::Recipe<'_> {
        let layers = &self.layers;
        step::Recipe {
            embedding: &self.embedding,
            layers: stack::Layers {
                norm: stack::Stacked {
                    tensor: layers.norm.view(),
                },
                query: stack::Stacked {
                    tensor: layers.query.view(),
                },
                pairs: stack::Stacked {
                    tensor: layers.pairs.view(),
                },
                qkv_scales: stack::Stacked {
                    tensor: layers.qkv_scales.view(),
                },
                pair_scale: stack::Stacked {
                    tensor: layers.pair_scale.view(),
                },
                output: stack::Stacked {
                    tensor: layers.output.view(),
                },
                output_scales: stack::Stacked {
                    tensor: layers.output_scales.view(),
                },
                post_norm: stack::Stacked {
                    tensor: layers.post_norm.view(),
                },
                gate: stack::Stacked {
                    tensor: layers.gate.view(),
                },
                up: stack::Stacked {
                    tensor: layers.up.view(),
                },
                down: stack::Stacked {
                    tensor: layers.down.view(),
                },
                mlp_scales: stack::Stacked {
                    tensor: layers.mlp_scales.view(),
                },
            },
            head: head::Projection {
                weight: self.head.view(),
            },
            greedy: &self.greedy,
            sampler: &self.sampler,
        }
    }
}

#[cfg(not(feature = "trace"))]
impl crate::Module<(&mut Device, (&step::Input, &mut step::State))> for Model {
    type Output = ();

    fn forward(&self, (device, (input, state)): (&mut Device, (&step::Input, &mut step::State))) {
        self.recipe().forward((device, (&self.norm, input, state)));
    }
}
