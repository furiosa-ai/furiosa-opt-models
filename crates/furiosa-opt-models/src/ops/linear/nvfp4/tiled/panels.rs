//! Activation-stationary NVFP4 projections. A 1024-feature panel is split over eight
//! input slices and reused by the four 128-token waves of a 512-row span.
use super::block;
use crate::Op;
use crate::modules::generic::collection::View;
use crate::ops::linear::nvfp4::tiled::Weights;
use crate::tensor::{live::Live, parallel::Chip};
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
type Reduced<O, R> = m![{ Owners<O> }, R % 128 / 16];
type Weight<I, O, D> = DmTensor<D, Chip, Cluster<O>, Slices<I, O>, m![O % (O / 64), I % 128]>;
type Scales<I, O> =
    DmTensor<f8e4m3, Chip, Cluster<O>, Slices<I, O>, m![O % (O / 64), I % 128 / 16]>;
pub(crate) type Packed<I, O> = (Weight<I, O, f4e2m1>, Scales<I, O>);
pub(crate) type Prepared<I, O> =
    DmTensor<bf16, Chip, Cluster<O>, Slices<I, O>, m![I / 1024, O % (O / 64), I % 128]>;
pub(crate) type Staged<I, O> = (
    DmTensor<f4e2m1, Chip, Cluster<O>, Slices<I, O>, m![I / 1024, O % (O / 64), I % 128]>,
    DmTensor<f8e4m3, Chip, Cluster<O>, Slices<I, O>, m![I / 1024, O % (O / 64), I % 128 / 16]>,
);
type Input<I, O, R> = DmTensor<bf16, Chip, Cluster<O>, Slices<I, O>, m![R % 128, I % 128]>;
type Operand<I, O, R> =
    TrfTensor<bf16, Chip, Cluster<O>, Slices<I, O>, m![R % 8], m![R % 128 / 8, I % 128]>;
type Partial<O, R> = DmTensor<f32, Chip, Cluster<O>, Reduced<O, R>, m![R % 16, O % (O / 64)]>;
type Sums<O, R> = DmTensor<f32, Chip, Cluster<O>, Reduced<O, R>, m![R / 128, R % 16, O % (O / 64)]>;
type Rows<R, W> = HbmTensor<bf16, Chip, m![R, W]>;

pub mod resident;

axes![Terms = 2];

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

    fn input<R: AxisName>(
        &self,
        device: &mut Device,
        input: &Rows<R, I>,
        panel: usize,
        wave: usize,
    ) -> Operand<I, O, R> {
        let input: Input<I, O, R> = input.view()
            .tile::<m![R / 128], 1, m![R / 128 = 1 # (R / 128), R % 128, I]>(wave)
            .tile::<m![I / 1024], 1, m![R / 128 = 1 # (R / 128), R % 128, I / 1024 = 1 # (I / 1024), I % 1024]>(panel)
            .to_dm(&mut device.tdma);
        device
            .sub
            .begin(input.view())
            .fetch::<m![R % 8, R % 128 / 8, I % 128 / 16], m![I % 16]>()
            .collect::<m![R % 8, R % 128 / 8, I % 128 / 16], m![I % 16]>()
            .to_trf()
    }
}

impl<I: AxisName, O: AxisName> Panels<I, O> {
    /// Restore the entire Q/K/V operand before the next layer needs it.
    pub(crate) fn prepare(&self, device: &mut Device) -> Prepared<I, O> {
        let staged = self.stage(device);
        Self::restore(device, staged)
    }

    /// Stage packed operands without occupying the contraction's main context.
    pub(crate) fn stage(&self, device: &mut Device) -> Staged<I, O> {
        let codes: DmTensor<
            f4e2m1,
            Chip,
            Cluster<O>,
            Slices<I, O>,
            m![I / 1024, O % (O / 64), I % 128],
        > = self.weights.weight.to_dm(&mut device.tdma);
        let scales: DmTensor<
            f8e4m3,
            Chip,
            Cluster<O>,
            Slices<I, O>,
            m![I / 1024, O % (O / 64), I % 128 / 16],
        > = self.weights.weight_scale.tensor.to_dm(&mut device.tdma);
        (codes, scales)
    }

    pub(crate) fn restore(device: &mut Device, staged: Staged<I, O>) -> Prepared<I, O> {
        block::restore(device, staged.0, staged.1)
    }

    /// The standalone panel API also serves larger MLP projection tests.
    pub(crate) fn prepared<R: AxisName>(
        &self,
        device: &mut Device,
        live: Live<(), R>,
        input: &Rows<R, I>,
        first: Packed<I, O>,
    ) -> Rows<R, O> {
        self.project::<R, false>(device, live, input, Some(first), None)
    }

    fn project<R: AxisName, const ALL: bool>(
        &self,
        device: &mut Device,
        live: Live<(), R>,
        input: &Rows<R, I>,
        first: Option<Packed<I, O>>,
        all: Option<&Prepared<I, O>>,
    ) -> Rows<R, O> {
        const {
            assert!(I::SIZE.is_multiple_of(1024) && O::SIZE.is_multiple_of(256));
            assert!(R::SIZE.is_multiple_of(128) && R::SIZE <= 512);
            assert!(O::SIZE / 64 * 128 / 16 * 4 <= 8192);
            assert!(128 * 128 * size_of::<bf16>() == 32768);
        }
        let waves = live.rows().div_ceil(128);
        let mut sums = Sums::<O, R>::new();
        // Keep the loop-carried operand a fixed tuple: the device compiler
        // cannot select a runtime Option variant at a slab boundary.
        let mut loaded = if ALL {
            (DmTensor::new(), DmTensor::new())
        } else {
            first.unwrap()
        };
        for panel in 0..I::SIZE / 1024 {
            let weight = if ALL {
                Weight::<I, O, bf16>::new()
            } else {
                let (codes, scales) = loaded;
                loaded = if panel + 1 < I::SIZE / 1024 {
                    self.load(device, panel + 1)
                } else {
                    (DmTensor::new(), DmTensor::new())
                };
                block::restore(device, codes, scales)
            };
            let mut lanes = self.input(device, input, panel, 0);
            for wave in 0..waves {
                // Two 32-KiB operands fit the 64-KiB TRF; the final prefetch wraps within storage.
                let next = self.input(device, input, panel, (wave + 1) % (R::SIZE / 128));
                let weight = if ALL {
                    all.unwrap().view().tile::<m![I / 1024],1,
                    m![I / 1024 = 1 # (I / 1024), O % (O / 64), I % 128]>(panel)
                    .unpad::<m![O % (O / 64), I % 128]>()
                } else {
                    weight.view()
                };
                let partial: Partial<O, R> = device.main.begin(weight)
                    .fetch::<m![R % 128 / 8, O % (O / 64), I % 128 / 16], m![I % 16]>()
                    .collect::<m![R % 128 / 8, O % (O / 64), I % 128 / 16], m![I % 16]>()
                    .contract_outer::<m![R % 128 / 8, O % (O / 64), I % 128 / 32], m![I % 32], _, _, _>(&lanes)
                    .contract_packet::<m![1]>()
                    .contract_time::<m![R % 128 / 8, O % (O / 64)]>()
                    .contract_lane::<m![R % 128 / 8, O % (O / 64)], m![R % 8]>(LaneMode::Interleaved)
                    .vector_init()
                    .vector_inter_slice_reduce::<Reduced<O, R>, m![R % 16 / 8, O % (O / 64)]>(InterSliceReduceOpF32::Add)
                    .vector_final()
                    .transpose::<m![R % 16 / 8, O % (O / 64) / 2, R % 8], m![O % 2 # 8]>()
                    .commit_trim::<m![O % 2]>().commit();
                lanes = next;
                let partial = if panel == 0 {
                    partial
                } else {
                    // Copy the selected wave out of its strided span before the elementwise sum.
                    let mut previous: Partial<O, R> = DmTensor::new();
                    sums.view()
                        .tile::<m![R / 128], 1, m![R / 128 = 1 # (R / 128), R % 16, O % (O / 64)]>(
                            wave,
                        )
                        .to_dm_view(&mut device.tdma, previous.view_mut());
                    device
                        .main
                        .begin_interleaved::<Terms, _, _, _, _, _>(previous.view(), partial.view())
                        .fetch::<m![R % 16, Terms, O % (O / 64) / 8], m![O % 8]>()
                        .collect::<m![R % 16, Terms, O % (O / 64) / 8], m![O % 8]>()
                        .vector_init()
                        .vector_intra_slice_unzip::<Terms, m![R % 16, O % (O / 64) / 8]>()
                        .vector_clip_zip(ClipBinaryOpF32::Add)
                        .vector_final()
                        .commit_trim::<m![O % 8]>()
                        .commit()
                };
                partial.view().to_dm_view(&mut device.tdma, sums.view_mut()
                    .tile::<m![R / 128], 1, m![R / 128 = 1 #{!} (R / 128), R % 16, O % (O / 64)]>(wave));
            }
        }
        let mut output = Rows::<R, O>::new();
        for wave in 0..waves {
            let mut partial = Partial::<O, R>::new();
            sums.view()
                .tile::<m![R / 128], 1, m![R / 128 = 1 # (R / 128), R % 16, O % (O / 64)]>(wave)
                .to_dm_view(&mut device.tdma, partial.view_mut());
            let values = block::finish(device, &self.weights.contract, partial);
            values.view().to_hbm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![R / 128], 1, m![R / 128 = 1 #{!} (R / 128), R % 128, O]>(wave),
            );
        }
        output
    }
}

impl<I: AxisName, O: AxisName, R: AxisName> Op<(Live<(), R>, &Rows<R, I>)> for Panels<I, O> {
    type Output = Rows<R, O>;
    fn forward(
        &self,
        device: &mut Device,
        (live, input): (Live<(), R>, &Rows<R, I>),
    ) -> Self::Output {
        let first = self.load(device, 0);
        self.prepared(device, live, input, first)
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

#[cfg(test)]
mod tests {
    use crate::ops::linear::nvfp4::tiled::Blocks;
    use crate::ops::linear::nvfp4::tiled::panels::Panels;
    use crate::ops::linear::nvfp4::tiled::tests as fixtures;
    use crate::tensor::form::BlockScaled;
    use crate::{
        Op,
        tensor::{live::Live, scalar::Value as Count},
    };
    use fixtures::*;
    use furiosa_opt_std::{prelude::*, runtime::launch};

    #[device(chip = 1)]
    fn gate(
        device: &mut Device,
        projection: &Panels<Hidden, Intermediate>,
        count: &Count,
        input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
    ) -> HbmTensor<bf16, m![1], m![Rows, Intermediate]> {
        projection.forward(device, (Live::<(), Rows>::read(count), input))
    }

    /// Exercise all four hidden panels, 224-wide feature groups, tie rounding, tensor/block
    /// scales, both clusters and partial token waves. The sparse oracle is independent of tiling.
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "opens a device"]
    async fn projects_llama_gate_geometry() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(gate.topology()).unwrap();
        let (weight, weight_scale, scales) = weights::<Hidden, Intermediate, _, _>(
            &mut device,
            Panels::<Hidden, Intermediate>::reorder::<1, _>,
            Panels::<Hidden, Intermediate>::reorder::<16, _>,
        )
        .await;
        let projection = Panels {
            weights: BlockScaled::new(weight, Blocks::new(weight_scale), scales),
        };
        let input = input::<Hidden>(&mut device).await;
        for rows in [129] {
            let count = count(&mut device, rows).await;
            let output = launch(gate, (&mut device, &projection, &count, &input))
                .await
                .unwrap();
            check::<Hidden, Intermediate>(&mut device, output, rows).await;
        }
    }
}
