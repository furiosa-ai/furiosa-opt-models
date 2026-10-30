//! Elementwise functions over bounded device-memory windows: FP32 arithmetic, rounded once to the
//! source element type.

use crate::load::Load;
use crate::ops::normalization::Chunk;
use crate::specialize::pointwise;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

pub mod activation;
pub mod add;
pub mod cast;
pub mod clamp;
pub mod gated;
pub mod guidance;
pub mod multiply;
pub mod rescale;
pub mod softplus;

/// `silu(gate) * up` of rows packed as their gate features followed by their up features.
#[derive(Clone, Copy, Default, DeviceSend, Load)]
pub struct SwiGlu;

/// Flattened logical element count, independent of the input's named axes.
#[derive(Clone, Debug)]
pub(crate) struct Flat<S>(PhantomData<S>);

impl<S: M + 'static> AxisName for Flat<S> {
    const NAME: Ident = Ident::new("Flat");
    const SIZE: usize = S::SIZE;
}

/// The flattened elements, a single one padded to a vector of eight.
#[derive(Clone, Debug)]
pub(crate) struct Vector<S>(PhantomData<S>);

impl<S: M + 'static> AxisName for Vector<S> {
    const NAME: Ident = Ident::new("Flat");
    const SIZE: usize = if S::SIZE == 1 { 8 } else { S::SIZE };
}

/// Elements per slice, an eight-element multiple.
#[derive(Clone, Debug)]
pub struct Part<S>(PhantomData<S>);

impl<S: M + 'static> AxisName for Part<S> {
    const NAME: Ident = Ident::new("Part");
    const SIZE: usize = pointwise::part(S::SIZE);
}

/// One transfer window covering equal slices and dividing the flattened input.
#[derive(Clone, Debug)]
pub struct Window<S>(PhantomData<S>);

impl<S: M + 'static> AxisName for Window<S> {
    const NAME: Ident = Ident::new("Window");
    const SIZE: usize = pointwise::window(S::SIZE);
}

type Width<S> = Symbol<Vector<S>>;
type Extent<S> = Symbol<Window<Width<S>>>;
type Segment<S> = Symbol<Part<Width<S>>>;
type Batch<S> = m![
    { Width<S> } / ({ Extent<S> }) = 1 # ({ Width<S> } / ({ Extent<S> })),
    { Width<S> } % ({ Extent<S> })
];
type BatchMut<S> = m![
    { Width<S> } / ({ Extent<S> }) = 1 #{!} ({ Width<S> } / ({ Extent<S> })),
    { Width<S> } % ({ Extent<S> })
];
type Slices<S> = m![{ Width<S> } % ({ Extent<S> }) / ({ Segment<S> }) # 256];
type Elements<S> = m![{ Width<S> } % ({ Segment<S> })];
type Rows<D, S, Chips = m![1]> = DmTensor<D, Chips, m![1 # 2], Slices<S>, Elements<S>>;
type Registers<S, Chips = m![1]> = VrfTensor<f32, Chips, m![1 # 2], Slices<S>, Elements<S>>;

/// Scalars widened to FP32 for arithmetic and converted back at the window boundary.
pub(crate) trait Float: MaterializableScalar + FetchCast<f32> {
    fn narrow<S: M + 'static, Chips: M>(
        device: &mut Device,
        input: Rows<f32, S, Chips>,
    ) -> Rows<Self, S, Chips>;
}

impl Float for f32 {
    fn narrow<S: M + 'static, Chips: M>(
        _: &mut Device,
        input: Rows<f32, S, Chips>,
    ) -> Rows<Self, S, Chips> {
        input
    }
}

impl Float for bf16 {
    fn narrow<S: M + 'static, Chips: M>(
        device: &mut Device,
        input: Rows<f32, S, Chips>,
    ) -> Rows<Self, S, Chips> {
        device
            .main
            .begin(input.view())
            .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .cast::<bf16, m![{ Width<S> } % 8 # 16]>()
            .commit_trim::<m![{ Width<S> } % 8]>()
            .commit()
    }
}

impl<S: M + 'static> Window<S> {
    /// `formula` of the inputs window by window, computed in FP32 in DM and rounded once to `D`
    /// on its way back; a single element pads to a vector of eight.
    pub(in crate::ops) fn map<D: Float, Chips: M, const N: usize>(
        device: &mut Device,
        inputs: [&HbmTensor<D, Chips, S>; N],
        formula: impl Fn(&mut Device, [Rows<D, S, Chips>; N]) -> Rows<f32, S, Chips>,
    ) -> HbmTensor<D, Chips, S> {
        if const { S::SIZE == 1 } {
            // Padded to a part (eight cells here) rather than to 8: this branch is instantiated for
            // every `S`, and the reshape below asserts equal sizes for each.
            type Padded<S> = m![{ Symbol<Flat<S>> } = 1 # ({ Segment<S> })];
            let inputs = inputs.map(|input| {
                // SAFETY: flattening preserves the contiguous element order.
                let input = unsafe { input.view().reshape::<Chips, m![{ Symbol<Flat<S>> }]>() };
                let input: DmTensor<D, Chips, m![1 # 2], m![1 # 256], Padded<S>> =
                    input.pad::<Padded<S>>().to_dm(&mut device.tdma);
                // SAFETY: both mappings cover the same eight contiguous cells; only lane 0 reaches
                // HBM.
                unsafe { input.reshape() }
            });
            let mut result = HbmTensor::<D, Chips, m![{ Symbol<Flat<S>> }]>::new();
            let output = formula(&mut *device, inputs);
            D::narrow(device, output)
                .view()
                .tile::<m![{ Width<S> }], 1, m![{ Width<S> } = 1 # 8]>(0)
                .to_hbm_view(
                    &mut device.tdma,
                    result
                        .view_mut()
                        .tile::<m![{ Symbol<Flat<S>> }], 1, m![{ Symbol<Flat<S>> } = 1]>(0),
                );
            // SAFETY: flattening preserves the contiguous element order.
            return unsafe { result.reshape() };
        }

        let mut result = HbmTensor::<D, Chips, m![{ Symbol<Flat<S>> }]>::new();
        for index in 0..S::SIZE / Window::<Width<S>>::SIZE {
            let inputs = inputs.map(|input| {
                // SAFETY: flattening preserves the contiguous element order.
                unsafe { input.view().reshape::<Chips, m![{ Symbol<Flat<S>> }]>() }
                    .tile::<m![{ Width<S> } / ({ Extent<S> })], 1, Batch<S>>(index)
                    .to_dm(&mut device.tdma)
            });
            let output = formula(&mut *device, inputs);
            D::narrow(device, output).view().to_hbm_view(
                &mut device.tdma,
                result
                    .view_mut()
                    .tile::<m![{ Width<S> } / ({ Extent<S> })], 1, BatchMut<S>>(index),
            );
        }
        // SAFETY: flattening preserves the contiguous element order.
        unsafe { result.reshape() }
    }

    /// `op` of two tensors' FP32 values in the same layout, the right one read from the vector
    /// registers.
    pub(in crate::ops) fn binary<D: Float, Chips: M>(
        device: &mut Device,
        inputs: [&HbmTensor<D, Chips, S>; 2],
        op: FpBinaryOp,
    ) -> HbmTensor<D, Chips, S> {
        Self::map(device, inputs, |device, [left, right]| {
            let right: Registers<S, Chips> = device
                .sub
                .begin(right.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .to_vrf();
            Self::stream(&mut device.main, &left)
                .vector_fp_binary(op, &right)
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        })
    }

    /// Each window's values in FP32, and the same values narrowed to four lanes, as the stream a
    /// formula computes in.
    fn stream<'l, D: Float, Chips: M>(
        main: &'l mut TuContext<{ Tu::Main }>,
        input: &'l Rows<D, S, Chips>,
    ) -> VectorNarrowTensor<
        'l,
        { Tu::Main },
        f32,
        Chips,
        m![1 # 2],
        Slices<S>,
        m![{ Width<S> } % ({ Segment<S> }) / 4],
        m![{ Width<S> } % 4],
        Fresh,
        { stage::VeOrder::IntraFirst },
    > {
        main.begin(input.view())
            .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Width<S> } % ({ Segment<S> }) / 4], m![{ Width<S> } % 4]>()
    }
}

/// One feature-slice tile of every row: its time and packet axes.
pub(super) type SpreadTime<W, R> = m![R, W % ({ Symbol<Chunk<W>> }) / 8];
pub(super) type SpreadPacket<W> = m![W % 8];

#[cfg(test)]
mod tests {
    use super::{activation::Silu, add::Add, multiply::Multiply};
    use crate::Op;
    use crate::ops::normalization::tests::random;
    use crate::ops::{index::scatter::Store, normalization::Spread};
    use crate::tensor;
    use crate::tensor::axis::Rows;
    use crate::tensor::live::tests::origins;
    use crate::tensor::scalar::Value;
    use crate::tensor::{axis::Sixteen, live::Live};
    use crate::tests::DEVICE;
    use furiosa_opt_std::{prelude::*, runtime::launch};

    type Envelope = Rows<8>;
    type Row = m![Features];
    axes![Features = 4096];

    #[device(chip = 1)]
    fn gated_rows(
        device: &mut Device,
        (gate, up): (
            &HbmTensor<bf16, m![1], m![Envelope, Features]>,
            &HbmTensor<bf16, m![1], m![Envelope, Features]>,
        ),
        live: &Value,
    ) -> HbmTensor<bf16, m![1], m![Envelope, Features]> {
        let tiles = Live::<(), Envelope>::read(live);
        let gate: Live<Spread<Features, Envelope>, Envelope> =
            tiles.with(gate.to_dm(&mut device.tdma));
        let up: Live<Spread<Features, Envelope>, Envelope> = tiles.with(up.to_dm(&mut device.tdma));
        let active = Silu.forward(&mut *device, &gate);
        let product = Multiply.forward(&mut *device, (&active, &up));
        Store.forward(device, &product)
    }

    #[device(chip = 1)]
    fn gated_row(
        device: &mut Device,
        (gate, up): (&HbmTensor<bf16, m![1], Row>, &HbmTensor<bf16, m![1], Row>),
    ) -> HbmTensor<bf16, m![1], Row> {
        let active = Silu.forward(&mut *device, gate);
        Multiply.forward(device, (&active, up))
    }

    /// A call's live rows in feature slices pass the activation and product exactly as each row
    /// alone passes them.
    #[tokio::test(flavor = "current_thread")]
    async fn gates_rows_as_alone() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(gated_rows.topology()).unwrap();
        let values = |seed| random(8 * 4096, seed);
        let (gates, ups) = (values(3), values(5));
        async fn upload<E: M>(device: &mut Device, values: &[bf16]) -> HbmTensor<bf16, m![1], E> {
            HostTensor::<bf16, E>::from_vec(values.iter().map(|v| bf16::from_f32(v.to_f32() * 4.0)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap()
        }
        let gate = upload::<m![Envelope, Features]>(&mut device, &gates).await;
        let up = upload::<m![Envelope, Features]>(&mut device, &ups).await;
        let live = tensor::live::tests::count(5)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let rows = launch(gated_rows, (&mut device, (&gate, &up), &live))
            .await
            .unwrap()
            .to_host::<m![Envelope, Features]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let bits = |values: &[bf16]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for row in 0..5 {
            let gate = upload::<Row>(&mut device, &gates[row * 4096..][..4096]).await;
            let up = upload::<Row>(&mut device, &ups[row * 4096..][..4096]).await;
            let alone = launch(gated_row, (&mut device, (&gate, &up)))
                .await
                .unwrap()
                .to_host::<Row>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            assert_eq!(bits(&rows[row * 4096..][..4096]), bits(&alone), "row {row}");
        }
    }

    #[device(chip = 1)]
    fn summed_origin_rows(
        device: &mut Device,
        (gate, up): (
            &HbmTensor<bf16, m![1], m![Sixteen, Features]>,
            &HbmTensor<bf16, m![1], m![Sixteen, Features]>,
        ),
        (start, rows): (&Value, &Value),
    ) -> HbmTensor<bf16, m![1], m![Sixteen, Features]> {
        let live = Live::<(), Sixteen>::all()
            .subrange(start.extent::<m![Sixteen]>(), rows.extent::<m![Sixteen]>());
        let gate: Live<Spread<Features, Sixteen>, Sixteen> =
            live.with(gate.to_dm(&mut device.tdma));
        let up: Live<Spread<Features, Sixteen>, Sixteen> = live.with(up.to_dm(&mut device.tdma));
        let active = Silu.forward(&mut *device, &gate);
        let product = Multiply.forward(&mut *device, (&active, &up));
        let sum = Add.forward(&mut *device, (&product, &up));
        Store.forward(device, &sum)
    }

    #[device(chip = 1)]
    fn summed_row(
        device: &mut Device,
        (gate, up): (&HbmTensor<bf16, m![1], Row>, &HbmTensor<bf16, m![1], Row>),
    ) -> HbmTensor<bf16, m![1], Row> {
        let active = Silu.forward(&mut *device, gate);
        let product = Multiply.forward(&mut *device, (&active, up));
        Add.forward(device, (&product, up))
    }

    /// Live rows that start past the envelope's first row, at a group boundary and across one,
    /// pass the activation, product and sum as each row alone does; rows outside them hold NaN.
    #[tokio::test(flavor = "current_thread")]
    async fn sums_rows_past_the_origin_as_alone() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(summed_origin_rows.topology()).unwrap();
        let values = |seed| random(16 * 4096, seed);
        let (gates, ups) = (values(7), values(11));
        let upload = async |device: &mut Device, values: Vec<bf16>| {
            HostTensor::<bf16, m![Sixteen, Features]>::from_vec(values)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap()
        };
        let bits = |values: &[bf16]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for live in origins() {
            let (first, count) = (live.start(), live.rows());
            let outside = |values: &[bf16]| -> Vec<bf16> {
                values
                    .iter()
                    .enumerate()
                    .map(|(at, v)| {
                        if (first..first + count).contains(&(at / 4096)) {
                            *v
                        } else {
                            bf16::from_f32(f32::NAN)
                        }
                    })
                    .collect()
            };
            let gate = upload(&mut device, outside(&gates)).await;
            let up = upload(&mut device, outside(&ups)).await;
            let (start, rows) = (
                tensor::live::tests::count(first)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                tensor::live::tests::count(count)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            );
            let summed = launch(
                summed_origin_rows,
                (&mut device, (&gate, &up), (&start, &rows)),
            )
            .await
            .unwrap()
            .to_host::<m![Sixteen, Features]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
            for row in first..first + count {
                let row_of = async |device: &mut Device, values: &[bf16]| {
                    HostTensor::<bf16, Row>::from_vec(values[row * 4096..][..4096].to_vec())
                        .to_hbm(&mut device.pdma)
                        .await
                        .unwrap()
                };
                let (gate, up) = (
                    row_of(&mut device, &gates).await,
                    row_of(&mut device, &ups).await,
                );
                let alone = launch(summed_row, (&mut device, (&gate, &up)))
                    .await
                    .unwrap()
                    .to_host::<Row>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec();
                assert_eq!(
                    bits(&summed[row * 4096..][..4096]),
                    bits(&alone),
                    "{first}, {count}: row {row}"
                );
            }
        }
    }
}
