//! Packed gate/up weights: 1024-feature slabs staged for the MLP contraction kernels.
use crate::modules::generic::collection::View;
use crate::ops::linear::nvfp4::tiled::Weights;
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;

#[derive(DeviceSend, crate::load::Load)]
pub struct Panels<I: AxisName, O: AxisName> {
    #[load(rename = "")]
    pub weights: Weights<I, O, Stored<I, O>, StoredScales<I, O>>,
}

pub type Stored<I, O> = m![
    I / 1024,
    O / (O / 64),
    I % 1024 / 128,
    O % (O / 64),
    I % 128
];

pub type StoredScales<I, O> = m![
    I / 1024,
    O / (O / 64),
    I % 1024 / 128,
    O % (O / 64),
    I % 128 / 16
];

type Cluster<O> = m![O / (O / 2)];
type Owners<O> = m![O % (O / 2) / (O / 64)];
type Slices<I, O> = m![{ Owners<O> }, I % 1024 / 128];
type Weight<I, O, D> = DmTensor<D, Chip, Cluster<O>, Slices<I, O>, m![O % (O / 64), I % 128]>;
type Scales<I, O> =
    DmTensor<f8e4m3, Chip, Cluster<O>, Slices<I, O>, m![O % (O / 64), I % 128 / 16]>;
pub(crate) type Packed<I, O> = (Weight<I, O, f4e2m1>, Scales<I, O>);
impl<I: AxisName, O: AxisName> Panels<I, O> {
    /// Canonical [output, input / PACK] values into contiguous per-slice panels.
    /// PACK is 2 for packed codes and 16 for block scales (1 for scalar test codes).
    #[cfg(test)]
    pub fn reorder<const PACK: usize, T: Copy>(values: &[T]) -> Vec<T> {
        assert_eq!(values.len(), O::SIZE * I::SIZE / PACK);
        let mut result = Vec::with_capacity(values.len());
        for panel in 0..I::SIZE / 1024 {
            for owner in 0..64 {
                for part in 0..8 {
                    for out in owner * (O::SIZE / 64)..(owner + 1) * (O::SIZE / 64) {
                        let at = (out * I::SIZE + panel * 1024 + part * 128) / PACK;
                        result.extend_from_slice(&values[at..at + 128 / PACK]);
                    }
                }
            }
        }
        result
    }

    pub(crate) fn load(&self, device: &mut Device, panel: usize) -> Packed<I, O> {
        (
            self.weights.weight
                .view()
                .tile::<m![I / 1024], 1, m![I / 1024 = 1 # (I / 1024), O / (O / 64), I % 1024 / 128, O % (O / 64), I % 128]>(panel)
                .to_dm(&mut device.tdma),
            self.weights.weight_scale.tensor
                .view()
                .tile::<m![I / 1024], 1, m![I / 1024 = 1 # (I / 1024), O / (O / 64), I % 1024 / 128, O % (O / 64), I % 128 / 16]>(panel)
                .to_dm(&mut device.tdma),
        )
    }
}

impl<I: AxisName, O: AxisName, const N: usize> View<N> for Panels<I, O> {
    type Storage = <Weights<I, O, Stored<I, O>, StoredScales<I, O>> as View<N>>::Storage;
    type Layer = Self;
    fn view(storage: &Self::Storage, layer: usize) -> Self {
        Self {
            weights: <Weights<I, O, Stored<I, O>, StoredScales<I, O>> as View<N>>::view(
                storage, layer,
            ),
        }
    }
    fn stack(layers: [Self; N], cx: &mut crate::load::Context<'_>) -> Self::Storage {
        <Weights<I, O, Stored<I, O>, StoredScales<I, O>> as View<N>>::stack(
            layers.map(|layer| layer.weights),
            cx,
        )
    }
}
