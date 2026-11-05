//! Single-row BF16 projections with tile-local decoding of quantized weights.

use super::Affine;
use crate::{
    Module,
    ops::{layout::axes::Chip, vector::scalar},
};
use furiosa_opt_std::prelude::*;

pub(super) use crate::ops::layout::BroadcastClusters as Cluster;

type Vector<W, Chips = Chip> = HbmTensor<bf16, Chips, m![W]>;
type Scale<O> = HbmTensor<f32, Chip, m![O, 1 # 8]>;

#[derive(Clone, Debug)]
pub(super) struct Rows<W>(std::marker::PhantomData<W>);

impl<W: AxisName> AxisName for Rows<W> {
    const NAME: Ident = Ident::new("Rows");
    const SIZE: usize = W::SIZE / 64;
}

pub(super) type Block<W> = Symbol<Rows<W>>;
pub(super) type Slices<I, O> = m![O / ({ Block<O> }), 1 # 2, I / (I / 2)];
pub(super) type Reduced<O> = m![O / ({ Block<O> }), 1 # 2, 1 # 2];
type Elements<O> = m![O % ({ Block<O> }) = 4];

type Tile<I, O, D = bf16, Chips = Chip> =
    DmTensor<D, Chips, Cluster, Slices<I, O>, m![{ Elements<O> }, I % (I / 2)]>;

type Values<O, Chips = Chip> = DmTensor<bf16, Chips, Cluster, Reduced<O>, Elements<O>>;

/// Supplies BF16 [output, input] tiles of an `In` to `Out` projection after applying the stored
/// weight scales.
trait Matrix<Chips: M = Chip> {
    type In: AxisName;
    type Out: AxisName;

    fn read(&self, device: &mut Device, index: usize) -> Tile<Self::In, Self::Out, bf16, Chips>;
}

/// Project one contiguous BF16 input row; both widths must be positive multiples of 256.
impl<I: AxisName, O: AxisName, Chip: M, W: Matrix<Chip, In = I, Out = O>>
    Module<(&mut Device, HbmTensorView<'_, bf16, Chip, m![I]>)> for Affine<W, ()>
{
    type Output = Vector<O, Chip>;

    fn forward(
        &self,
        (device, input): (&mut Device, HbmTensorView<'_, bf16, Chip, m![I]>),
    ) -> Self::Output {
        const {
            assert!(
                I::SIZE > 0
                    && O::SIZE > 0
                    && I::SIZE.is_multiple_of(256)
                    && O::SIZE.is_multiple_of(256)
            )
        };
        let input: DmTensor<bf16, Chip, Cluster, m![64, 1 # 2, I / (I / 2)], m![I % (I / 2)]> =
            input.to_dm(&mut device.tdma);
        let input: DmTensor<bf16, Chip, Cluster, Slices<I, O>, m![I % (I / 2)]> =
            unsafe { input.reshape() };
        let input: TrfTensor<bf16, Chip, Cluster, Slices<I, O>, m![1], m![I % (I / 2)]> = device
            .sub
            .begin(input.view())
            .fetch::<m![I % (I / 2) / 16], m![I % 16]>()
            .collect::<m![I % (I / 2) / 16], m![I % 16]>()
            .to_trf();
        let mut output: DmTensor<bf16, Chip, Cluster, Reduced<O>, m![O % ({ Block<O> })]> =
            DmTensor::new();
        for index in 0..(O::SIZE / 256) {
            let index = if const { O::SIZE == 256 } {
                0
            } else {
                4 * index
            };
            let tile = self.weight.read(device, index);
            let partial: DmTensor<f32, Chip, Cluster, Slices<I, O>, m![{ Elements<O> }, 1 # 8]> =
                device
                    .main
                    .begin(tile.view())
                    .fetch::<m![{ Elements<O> }, I % (I / 2) / 16], m![I % 16]>()
                    .collect::<m![{ Elements<O> }, I % (I / 2) / 16], m![I % 16]>()
                    .contract_outer::<m![{ Elements<O> }, I % (I / 2) / 32], m![I % 32], _, _, _>(
                        &input,
                    )
                    .contract_packet::<m![1]>()
                    .contract_time::<Elements<O>>()
                    .contract_lane::<Elements<O>, m![1 # 8]>(LaneMode::Interleaved)
                    .commit_trim::<m![1 # 8]>()
                    .commit();
            let value: Values<O, Chip> = device
                .main
                .begin(partial.view())
                .fetch::<Elements<O>, m![1 # 8]>()
                .collect::<Elements<O>, m![1 # 8]>()
                .vector_init()
                .vector_inter_slice_reduce::<Reduced<O>, Elements<O>>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![1 # 16]>()
                .transpose::<m![1], m![{ Elements<O> } # 16]>()
                .commit_trim::<Elements<O>>()
                .commit();
            value.view().to_dm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![O % ({ Block<O> })], 4, m![{ Elements<O> } #{!} ({ Block<O> })]>(
                        index,
                    ),
            );
        }
        output.to_hbm(&mut device.tdma)
    }
}

impl<I: AxisName, O: AxisName, Chip: M> Matrix<Chip> for HbmTensor<bf16, Chip, m![O, I]> {
    type In = I;
    type Out = O;

    fn read(&self, device: &mut Device, index: usize) -> Tile<I, O, bf16, Chip> {
        self.view().tile::<m![O % ({ Block<O> })], 4, m![O / ({ Block<O> }), { Elements<O> } # ({ Block<O> }), I]>(index).to_dm(&mut device.tdma)
    }
}

/// Every weight form projects a row through its view.
impl<I: AxisName, O: AxisName, Chip: M, W: Matrix<Chip, In = I, Out = O>>
    Module<(&mut Device, &Vector<I, Chip>)> for Affine<W, ()>
{
    type Output = Vector<O, Chip>;

    fn forward(&self, (device, input): (&mut Device, &Vector<I, Chip>)) -> Self::Output {
        self.forward((device, input.view()))
    }
}

/// A row read as its heads in order.
impl<I: AxisName, O: AxisName, Heads: AxisName, Head: AxisName, Chip: M, W>
    Module<(&mut Device, &HbmTensor<bf16, Chip, m![Heads, Head]>)> for Affine<W, ()>
where
    W: Matrix<Chip, In = I, Out = O>,
{
    type Output = Vector<O, Chip>;

    fn forward(
        &self,
        (device, input): (&mut Device, &HbmTensor<bf16, Chip, m![Heads, Head]>),
    ) -> Self::Output {
        const { assert!(I::SIZE == Heads::SIZE * Head::SIZE) };
        // SAFETY: the heads of a row are its `I` inputs in order.
        let input: HbmTensorView<'_, bf16, Chip, m![I]> = unsafe { input.view().reshape() };
        self.forward((device, input))
    }
}

impl<I: AxisName, O: AxisName> Matrix for (HbmTensor<f8e4m3, Chip, m![O, I]>, Scale<O>) {
    type In = I;
    type Out = O;

    fn read(&self, device: &mut Device, index: usize) -> Tile<I, O> {
        let packed: Tile<I, O, f8e4m3> = self.0.view().tile::<m![O % ({ Block<O> })], 4, m![O / ({ Block<O> }), { Elements<O> } # ({ Block<O> }), I]>(index).to_dm(&mut device.tdma);
        let packed: Tile<I, O> = device
            .main
            .begin(packed.view())
            .fetch::<m![{ Elements<O> }, I % (I / 2) / 16], m![I % 16]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![{ Elements<O> }, I % (I / 2) / 16], m![I % 16]>()
            .commit_trim::<m![I % 16]>()
            .commit();
        let scale: DmTensor<f32, Chip, Cluster, Slices<I, O>, m![{ Elements<O> }, 1 # 8]> = self.1.view().tile::<m![O % ({ Block<O> })], 4, m![O / ({ Block<O> }), { Elements<O> } # ({ Block<O> }), 1 # 8]>(index).to_dm(&mut device.tdma);
        let scale: VrfTensor<f32, Chip, Cluster, Slices<I, O>, m![{ Elements<O> }, 1 # 8]> = device
            .sub
            .begin(scale.view())
            .fetch::<Elements<O>, m![1 # 8]>()
            .collect::<Elements<O>, m![1 # 8]>()
            .to_vrf();
        device
            .main
            .begin(packed.view())
            .fetch::<m![{ Elements<O> }, I % (I / 2) / 16], m![I % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Elements<O> }, I % (I / 2) / 8], m![I % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Elements<O> }, I % (I / 2) / 4], m![I % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![{ Elements<O> }, I % (I / 2) / 8], m![I % 8]>()
            .vector_final()
            .cast::<bf16, m![I % 8 # 16]>()
            .commit_trim::<m![I % 8]>()
            .commit()
    }
}

impl<I: AxisName, O: AxisName> Matrix
    for (
        HbmTensor<f4e2m1, Chip, m![O, I]>,
        HbmTensor<f8e4m3, Chip, m![O, I / 16]>,
        HbmTensor<f32, Chip, m![1 # 8]>,
    )
{
    type In = I;
    type Out = O;

    fn read(&self, device: &mut Device, index: usize) -> Tile<I, O> {
        let packed: Tile<I, O, f4e2m1> = self.0.view().tile::<m![O % ({ Block<O> })], 4, m![O / ({ Block<O> }), { Elements<O> } # ({ Block<O> }), I]>(index).to_dm(&mut device.tdma);
        let packed: Tile<I, O, f8e4m3> = device
            .main
            .begin(packed.view())
            .fetch::<Elements<O>, m![I % (I / 2)]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![{ Elements<O> }, I % (I / 2) / 32], m![I % 32]>()
            .commit_trim::<m![I % 32]>()
            .commit();
        let scale: DmTensor<f8e4m3, Chip, Cluster, Slices<I, O>, m![{ Elements<O> }, I % (I / 2) / 16]> = self.1.view().tile::<m![O % ({ Block<O> })], 4, m![O / ({ Block<O> }), { Elements<O> } # ({ Block<O> }), I / 16]>(index).to_dm(&mut device.tdma);
        let scale: VrfTensor<
            f32,
            Chip,
            Cluster,
            Slices<I, O>,
            m![{ Elements<O> }, I % (I / 2) / 16],
        > = device
            .sub
            .begin(scale.view())
            .fetch::<Elements<O>, m![I % (I / 2) / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Elements<O> }, I % (I / 2) / 128], m![I / 16 % 8]>()
            .to_vrf();
        let global: DmTensor<f32, Chip, Cluster, Slices<I, O>, m![1 # 8]> =
            self.2.to_dm(&mut device.tdma);
        let global = scalar::register(device, &global);
        device
            .main
            .begin(packed.view())
            .fetch::<m![{ Elements<O> }, I % (I / 2) / 32], m![I % 32]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Elements<O> }, I % (I / 2) / 8], m![I % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Elements<O> }, I % (I / 2) / 4], m![I % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_fp_div(&global)
            .vector_widen_concat::<m![{ Elements<O> }, I % (I / 2) / 8], m![I % 8]>()
            .vector_final()
            .cast::<bf16, m![I % 8 # 16]>()
            .commit_trim::<m![I % 8]>()
            .commit()
    }
}
