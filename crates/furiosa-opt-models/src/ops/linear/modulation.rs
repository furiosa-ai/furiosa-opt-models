//! Projections of a time embedding into the modulation of packed rows: a layer's tables, a row
//! per timestep and modality, and the final norm's shift and scale of one timestep.

use crate::Op;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::index::packed::{Modality, Table, Timestep};
use crate::ops::linear::kernel;
use crate::ops::normalization::Shift;
use crate::ops::pointwise::activation::Silu;
use crate::tensor::parallel::{self, Shard};
use furiosa_opt_std::prelude::*;
use safetensors::Dtype;
use std::borrow::Borrow;
use std::marker::PhantomData;

axes![Chunks = 6, Part = 2];

/// A shift, then a scale, of each feature.
#[derive(Clone, Debug)]
pub struct ShiftScale<W>(PhantomData<W>);

impl<W: AxisName> AxisName for ShiftScale<W> {
    const NAME: Ident = Ident::new("ShiftScale");
    const SIZE: usize = 2 * W::SIZE;
}

/// The final norm's `(shift, scale)` of timestep `TIMESTEP` (video 0, audio 1): `bf16(silu(t))`
/// projected with an FP32 bias by `weight` and `bias`, rounded once, the shift features first.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct HeadModulation<
    In: AxisName,
    W: AxisName,
    const TIMESTEP: usize,
    Chips: M = m![1],
    T = HbmTensor<bf16, Chips, m![{ Symbol<ShiftScale<W>> }, In]>,
    B = HbmTensor<f32, Chips, m![{ Symbol<ShiftScale<W>> }, 1 # 8]>,
> {
    weight: T,
    bias: B,
    shape: PhantomData<fn() -> (In, W, Chips)>,
}

impl<In: AxisName, W: AxisName, const TIMESTEP: usize, Chips: M>
    HeadModulation<In, W, TIMESTEP, Chips>
{
    pub const fn new(
        weight: HbmTensor<bf16, Chips, m![{ Symbol<ShiftScale<W>> }, In]>,
        bias: HbmTensor<f32, Chips, m![{ Symbol<ShiftScale<W>> }, 1 # 8]>,
    ) -> Self {
        Self {
            weight,
            bias,
            shape: PhantomData,
        }
    }
}

impl<'x, In: AxisName, W: AxisName, const TIMESTEP: usize, Chips: M, T, B>
    Op<&'x HbmTensor<f32, Chips, m![Timestep, In]>> for HeadModulation<In, W, TIMESTEP, Chips, T, B>
where
    T: Borrow<HbmTensor<bf16, Chips, m![{ Symbol<ShiftScale<W>> }, In]>>,
    B: Borrow<HbmTensor<f32, Chips, m![{ Symbol<ShiftScale<W>> }, 1 # 8]>>,
{
    type Output = Shift<W, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        time: &'x HbmTensor<f32, Chips, m![Timestep, In]>,
    ) -> Self::Output {
        let activated = Silu.forward(&mut *device, time.view());
        let projected = kernel::Biased::<Timestep>(PhantomData).forward(
            &mut *device,
            ((self.weight.borrow(), self.bias.borrow()), activated.view()),
        );
        let mut packed: HbmTensor<bf16, Chips, m![Timestep, { Symbol<ShiftScale<W>> }]> =
            HbmTensor::new();
        projected
            .view()
            .to_hbm_view(&mut device.tdma, packed.view_mut());
        // SAFETY: a timestep's outputs are its shift, then its scale.
        let packed = unsafe { packed.view().reshape::<Chips, m![Timestep, Part, W]>() };
        let part = |device: &mut Device, part: usize| -> HbmTensor<bf16, Chips, m![W]> {
            let mut vector = HbmTensor::new();
            packed
                .tile::<m![Timestep], 1, m![Timestep = 1 # 2, Part, W]>(TIMESTEP)
                .tile::<m![Part], 1, m![Timestep = 1 # 2, Part = 1 # 2, W]>(part)
                .to_hbm_view(
                    &mut device.tdma,
                    // SAFETY: a part is its features in order.
                    unsafe {
                        vector
                            .view_mut()
                            .reshape::<Chips, m![Timestep = 1 # 2, Part = 1 # 2, W]>()
                    },
                );
            vector
        };
        (part(device, 0), part(device, 1))
    }
}

/// The weight as stored, the bias widened to its lanes.
/// By hand, with the update below: a derived `with` update copies a new leaf through host.
impl<'a, In: AxisName, W: AxisName, const TIMESTEP: usize, Chips, T, B> Load<&'a Checkpoint>
    for HeadModulation<In, W, TIMESTEP, Chips, T, B>
where
    Chips: M + 'static,
    T: Load<&'a Checkpoint>,
    B: Load<HostTensor<f32, m![{ Chips }, { m![{ Symbol<ShiftScale<W>> }, 1 # 8] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let weight = children.ask(cx.field("weight", "weight", |cx| T::load(cx, from)))?;
            let bias = children.ask(cx.field("bias", "bias", |cx| {
                let bias = widened::<ShiftScale<W>, Chips>(from, cx)?;
                B::load(cx, bias)
            }))?;
            children.node(weight.zip(bias)).map(|(weight, bias)| Self {
                weight,
                bias,
                shape: PhantomData,
            })
        })
    }
}

/// The weight and the widened bias written in place, each one rule of an update.
impl<'t, 's, In: AxisName, W: AxisName, const TIMESTEP: usize, Chips, T, B>
    Load<(
        &'t mut HeadModulation<In, W, TIMESTEP, Chips, T, B>,
        &'s Checkpoint,
    )> for &'t mut HeadModulation<In, W, TIMESTEP, Chips, T, B>
where
    Chips: M + 'static,
    for<'u> &'u mut T: Load<(&'u mut T, &'s Checkpoint)>,
    for<'u> &'u mut B: Load<(
        &'u mut B,
        HostTensor<f32, m![{ Chips }, { m![{ Symbol<ShiftScale<W>> }, 1 # 8] }]>,
    )>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut HeadModulation<In, W, TIMESTEP, Chips, T, B>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("weight", "weight", |cx| {
                <&mut T as Load<_>>::load(cx, (&mut into.weight, from))
            })?;
            cx.field("bias", "bias", |cx| {
                cx.rule(
                    |cx| widened::<ShiftScale<W>, Chips>(from, cx),
                    |cx, host| <&mut B as Load<_>>::load(cx, (&mut into.bias, host)).map(drop),
                )
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

/// Six modulation chunks of each modality: one projection output per timestep.
#[derive(Clone, Debug)]
pub struct Modulations<W>(PhantomData<W>);

impl<W: AxisName> AxisName for Modulations<W> {
    const NAME: Ident = Ident::new("Modulations");
    const SIZE: usize = Modality::SIZE * Chunks::SIZE * W::SIZE;
}

type Outputs<W> = Symbol<Modulations<W>>;

/// The chips of an `N`-way split.
type Chips<const N: usize> = m![{ Symbol<Shard<N>> }];

/// A layer's attention or feed-forward `((shift, scale), gate)` tables.
type Tables<W, Chips = m![1]> = ((Table<W, Chips>, Table<W, Chips>), Table<W, Chips>);

/// A layer's modulation tables, `bf16(silu(t))` projected by `weight` with an FP32 `bias`: feature
/// `modality * 6W + chunk * W + j` lies in table `chunk`, row `timestep * 3 + modality`.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Projection<
    In: AxisName,
    W: AxisName,
    Chips: M = m![1],
    L: AxisName = W,
    T = HbmTensor<bf16, Chips, m![{ Outputs<L> }, In]>,
    B = HbmTensor<f32, Chips, m![{ Outputs<L> }, 1 # 8]>,
> {
    weight: T,
    bias: B,
    shape: PhantomData<fn() -> (In, W, Chips, L)>,
}

impl<In: AxisName, W: AxisName, Chips: M, L: AxisName> Projection<In, W, Chips, L> {
    pub const fn new(
        weight: HbmTensor<bf16, Chips, m![{ Outputs<L> }, In]>,
        bias: HbmTensor<f32, Chips, m![{ Outputs<L> }, 1 # 8]>,
    ) -> Self {
        Self {
            weight,
            bias,
            shape: PhantomData,
        }
    }
}

impl<In: AxisName, W: AxisName, Chips: M, L: AxisName, T, B>
    Op<(HbmTensorView<'_, f32, Chips, m![Timestep, In]>,)> for Projection<In, W, Chips, L, T, B>
where
    T: Borrow<HbmTensor<bf16, Chips, m![{ Outputs<L> }, In]>>,
    B: Borrow<HbmTensor<f32, Chips, m![{ Outputs<L> }, 1 # 8]>>,
{
    type Output = HbmTensor<bf16, Chips, m![Timestep, { Outputs<L> }]>;

    fn forward(
        &self,
        device: &mut Device,
        (time,): (HbmTensorView<'_, f32, Chips, m![Timestep, In]>,),
    ) -> Self::Output {
        let activated = Silu.forward(&mut *device, time);
        let projected = kernel::Biased::<Timestep>(PhantomData).forward(
            &mut *device,
            ((self.weight.borrow(), self.bias.borrow()), activated.view()),
        );
        let mut packed = HbmTensor::new();
        projected
            .view()
            .to_hbm_view(&mut device.tdma, packed.view_mut());
        packed
    }
}

impl<'x, In: AxisName, W: AxisName, Chips: M, T, B> Op<&'x HbmTensor<f32, Chips, m![Timestep, In]>>
    for Projection<In, W, Chips, W, T, B>
where
    T: Borrow<HbmTensor<bf16, Chips, m![{ Outputs<W> }, In]>>,
    B: Borrow<HbmTensor<f32, Chips, m![{ Outputs<W> }, 1 # 8]>>,
{
    type Output = (Tables<W, Chips>, Tables<W, Chips>);

    fn forward(
        &self,
        device: &mut Device,
        time: &'x HbmTensor<f32, Chips, m![Timestep, In]>,
    ) -> Self::Output {
        let packed = self.forward(&mut *device, (time.view(),));
        // SAFETY: an output feature is its modality, its chunk and its feature, in that order.
        let packed = unsafe {
            packed
                .view()
                .reshape::<Chips, m![Timestep, Modality, Chunks, W]>()
        };
        let table = |device: &mut Device, chunk: usize| -> Table<W, Chips> {
            let mut table: Table<W, Chips> = HbmTensor::new();
            packed
                .tile::<m![Chunks], 1, m![Timestep, Modality, Chunks = 1 # 6, W]>(chunk)
                .to_hbm_view(
                    &mut device.tdma,
                    // SAFETY: a table row is `timestep * 3 + modality`.
                    unsafe {
                        table
                            .view_mut()
                            .reshape::<Chips, m![Timestep, Modality, W]>()
                    },
                );
            table
        };
        let attention = ((table(device, 0), table(device, 1)), table(device, 2));
        let feedforward = ((table(device, 3), table(device, 4)), table(device, 5));
        (attention, feedforward)
    }
}

/// A chip's projection weight: its rows of every modality's and chunk's features.
type Rows<In, L, Chips> = HostTensor<bf16, m![{ Chips }, { m![{ Outputs<L> }, In] }]>;

/// A chip's widened bias: its values of every modality's and chunk's features.
type Lanes<L, Chips> = HostTensor<f32, m![{ Chips }, { m![{ Outputs<L> }, 1 # 8] }]>;

impl<In: AxisName, W: AxisName, Chips: M + 'static, L: AxisName, T, B>
    Projection<In, W, Chips, L, T, B>
{
    /// The values of a stored tensor of every modality's and chunk's `W` features, `row` values a
    /// feature, as the chips hold them: chip `c` the `c`-th part of each chunk's features when a
    /// chip holds a part (`L` narrower than `W`), every feature on every chip otherwise.
    fn shares<V: Copy>(stored: &[V], row: usize) -> Vec<V> {
        const { assert!(L::SIZE == W::SIZE || Chips::SIZE * L::SIZE == W::SIZE) };
        let first = |chip: usize| match L::SIZE == W::SIZE {
            true => 0,
            false => chip * L::SIZE * row,
        };
        let chunks = |chip| {
            let chunks = stored.chunks_exact(W::SIZE * row);
            chunks.flat_map(move |chunk| &chunk[first(chip)..][..L::SIZE * row])
        };
        (0..Chips::SIZE).flat_map(chunks).copied().collect()
    }

    /// The BF16 weight at the context's name: its name, dtype and shape checked now, its chips'
    /// rows made by the function returned.
    fn weight<'s>(
        source: &'s Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<
        impl Fn() -> Result<Rows<In, L, Chips>, LoadError> + use<'s, In, W, Chips, L, T, B>,
        LoadError,
    > {
        let (name, view) = context.tensor(source)?;
        let invalid = |reason: String| LoadError::InvalidData {
            name: name.clone(),
            reason,
        };
        let shape = [Modulations::<W>::SIZE, In::SIZE];
        let float = matches!(view.dtype(), Dtype::BF16 | Dtype::F32);
        if !float || view.shape() != shape {
            return Err(invalid(format!(
                "{:?} of shape {:?}, the leaf holds BF16 of shape {shape:?}",
                view.dtype(),
                view.shape()
            )));
        }
        Ok(move || {
            let stored = Checkpoint::bf16(&view, &name)?;
            let rows = Self::shares(&stored, In::SIZE * size_of::<bf16>());
            let rows = furiosa_opt_std::runtime::HostBuffer::from_vec(rows);
            HostTensor::try_from(rows).map_err(|error| LoadError::InvalidData {
                name: name.clone(),
                reason: error.to_string(),
            })
        })
    }

    /// The bias at the context's name, each of the chips' values in the first of its eight lanes.
    fn bias(source: &Checkpoint, context: &mut Context<'_>) -> Result<Lanes<L, Chips>, LoadError> {
        let bias = context.floats(source, &[Modulations::<W>::SIZE])?;
        let lanes = Self::shares(&bias, 1)
            .into_iter()
            .flat_map(|value| [value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        Ok(HostTensor::from_vec(lanes.collect::<Vec<f32>>()))
    }
}

/// The weight as stored and the bias widened to its lanes, each chip its features.
/// By hand, with the update below: a derived `with` update copies a new leaf through host.
impl<'a, In: AxisName, W: AxisName, Chips, L: AxisName, T, B> Load<&'a Checkpoint>
    for Projection<In, W, Chips, L, T, B>
where
    Chips: M + 'static,
    T: for<'f> Load<&'f dyn Fn() -> Result<Rows<In, L, Chips>, LoadError>>,
    B: Load<Lanes<L, Chips>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let weight = children.ask(cx.field("weight", "weight", |cx| {
                let build = Self::weight(from, cx)?;
                T::load(cx, &build as &dyn Fn() -> _)
            }))?;
            let bias = children.ask(cx.field("bias", "bias", |cx| {
                let bias = Self::bias(from, cx)?;
                B::load(cx, bias)
            }))?;
            children.node(weight.zip(bias)).map(|(weight, bias)| Self {
                weight,
                bias,
                shape: PhantomData,
            })
        })
    }
}

/// The weight and the widened bias written in place, each one rule of an update.
impl<'t, 's, In: AxisName, W: AxisName, Chips, L: AxisName, T, B>
    Load<(&'t mut Projection<In, W, Chips, L, T, B>, &'s Checkpoint)>
    for &'t mut Projection<In, W, Chips, L, T, B>
where
    Chips: M + 'static,
    for<'u, 'f> &'u mut T: Load<(
        &'u mut T,
        &'f dyn Fn() -> Result<Rows<In, L, Chips>, LoadError>,
    )>,
    for<'u> &'u mut B: Load<(&'u mut B, Lanes<L, Chips>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Projection<In, W, Chips, L, T, B>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("weight", "weight", |cx| {
                cx.rule(
                    |cx| Projection::<In, W, Chips, L, T, B>::weight(from, cx),
                    |cx, build| {
                        <&mut T as Load<_>>::load(cx, (&mut into.weight, &build as &dyn Fn() -> _))
                            .map(drop)
                    },
                )
            })?;
            cx.field("bias", "bias", |cx| {
                cx.rule(
                    |cx| Projection::<In, W, Chips, L, T, B>::bias(from, cx),
                    |cx, host| <&mut B as Load<_>>::load(cx, (&mut into.bias, host)).map(drop),
                )
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

/// Features split over `N` chips: each chip projects its part, and every table is written on every
/// chip with the parts in chip order, one transfer from each chip to each chip.
impl<'x, In: AxisName, W: AxisName, const N: usize, T, B>
    Op<&'x HbmTensor<f32, Chips<N>, m![Timestep, In]>>
    for Projection<In, W, Chips<N>, parallel::Part<W, N>, T, B>
where
    T: Borrow<HbmTensor<bf16, Chips<N>, m![{ Outputs<parallel::Part<W, N>> }, In]>>,
    B: Borrow<HbmTensor<f32, Chips<N>, m![{ Outputs<parallel::Part<W, N>> }, 1 # 8]>>,
{
    type Output = (Tables<W, Chips<N>>, Tables<W, Chips<N>>);

    fn forward(
        &self,
        device: &mut Device,
        time: &'x HbmTensor<f32, Chips<N>, m![Timestep, In]>,
    ) -> Self::Output {
        type L<W, const N: usize> = Symbol<parallel::Part<W, N>>;
        let packed = self.forward(&mut *device, (time.view(),));
        // SAFETY: an output feature is its modality, its chunk and its part's feature, in order.
        let packed = unsafe {
            packed
                .view()
                .reshape::<Chips<N>, m![Timestep, Modality, Chunks, { L<W, N> }]>()
        };
        let table = |device: &mut Device, chunk: usize| -> Table<W, Chips<N>> {
            let mut table: HbmTensor<
                bf16,
                Broadcast<N>,
                m![Timestep, Modality, { Symbol<Shard<N>> }, { L<W, N> }],
            > = HbmTensor::new();
            packed
                .tile::<m![Chunks], 1, m![Timestep, Modality, Chunks = 1 # 6, { L<W, N> }]>(chunk)
                .to_hbm_view(&mut device.tdma, table.view_mut());
            // SAFETY: a table row is `timestep * 3 + modality`, its features the parts in chip
            // order, on every chip.
            unsafe { table.reshape() }
        };
        let attention = ((table(device, 0), table(device, 1)), table(device, 2));
        let feedforward = ((table(device, 3), table(device, 4)), table(device, 5));
        (attention, feedforward)
    }
}

/// The bias at the context's name, `R` values stored as F32 or BF16, each in the first of its eight
/// lanes, on every chip of `C`.
// TODO: read the values only for a leaf that holds bytes, as a converted leaf's build does.
fn widened<R: AxisName, C: M + 'static>(
    source: &Checkpoint,
    context: &mut Context<'_>,
) -> Result<HostTensor<f32, m![{ C }, { m![{ Symbol<R> }, 1 # 8] }]>, LoadError> {
    let bias = context.floats(source, &[R::SIZE])?;
    let lanes = bias
        .into_iter()
        .flat_map(|value| [value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let lanes: Vec<f32> = lanes.collect();
    Ok(HostTensor::from_vec(lanes.repeat(C::SIZE)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::index::packed::Group;
    use crate::ops::normalization::tests::random;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![Hidden = 5376];
    axes![Time = 2688];

    #[device(chip = 1)]
    fn projected(
        device: &mut Device,
        projection: &Projection<Time, Hidden>,
        time: &HbmTensor<f32, m![1], m![Timestep, Time]>,
    ) -> (Tables<Hidden>, Tables<Hidden>) {
        projection.forward(device, time)
    }

    /// Each table row is `bf16(bf16(silu(t)) . w + b)` of its timestep and of its modality's
    /// chunk, the sum FP64 and rounded once, within FP32 accumulation error.
    #[tokio::test(flavor = "current_thread")]
    async fn tables_take_their_modality_chunk() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let (width, inputs) = (Hidden::SIZE, Time::SIZE);
        let outputs = Modulations::<Hidden>::SIZE;
        let time: Vec<f32> = random(Timestep::SIZE * inputs, 0x2545_f491_4f6c_dd1d)
            .iter()
            .map(|v| v.to_f32() * 4.0)
            .collect();
        let weight = random(outputs * inputs, 0x9e37_79b9_7f4a_7c15);
        let bias = random(outputs, 0x0123_4567_89ab_cdef);
        let projection = Projection::new(
            HostTensor::<bf16, m![{ Outputs<Hidden> }, Time]>::from_vec(weight.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            HostTensor::<f32, m![{ Outputs<Hidden> }, 1 # 8]>::from_vec(
                bias.iter()
                    .flat_map(|b| [b.to_f32(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    .collect::<Vec<f32>>(),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        );
        let input = HostTensor::<f32, m![Timestep, Time]>::from_vec(time.clone())
            .to_hbm::<m![1], m![Timestep, Time]>(&mut device.pdma)
            .await
            .unwrap();
        let (attention, feedforward) = launch(projected, (&mut device, &projection, &input))
            .await
            .unwrap();
        let ((shift, scale), gate) = attention;
        let ((shift2, scale2), gate2) = feedforward;
        let mut tables = Vec::new();
        for table in [shift, scale, gate, shift2, scale2, gate2] {
            tables.push(
                table
                    .to_host::<m![Group, Hidden]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec(),
            );
        }
        let activated: Vec<f64> = time
            .iter()
            .map(|t| {
                let t = f64::from(*t);
                f64::from(bf16::from_f32((t / (1.0 + (-t).exp())) as f32).to_f32())
            })
            .collect();
        for (chunk, table) in tables.iter().enumerate() {
            for group in 0..Group::SIZE {
                let (timestep, modality) = (group / 3, group % 3);
                let a = &activated[timestep * inputs..][..inputs];
                for j in 0..width {
                    let output = modality * 6 * width + chunk * width + j;
                    let expected: f64 = a
                        .iter()
                        .zip(&weight[output * inputs..][..inputs])
                        .map(|(a, w)| a * f64::from(w.to_f32()))
                        .sum::<f64>()
                        + f64::from(bias[output].to_f32());
                    let got = f64::from(table[group * width + j].to_f32());
                    assert!(
                        (got - expected).abs() <= expected.abs() / 256.0 + 0.01,
                        "chunk {chunk} row {group} feature {j}: {got} vs {expected}"
                    );
                }
            }
        }
    }

    #[device(chip = 1)]
    fn shifted(
        device: &mut Device,
        modulation: &HeadModulation<Time, Hidden, 1>,
        time: &HbmTensor<f32, m![1], m![Timestep, Time]>,
    ) -> Shift<Hidden> {
        modulation.forward(device, time)
    }

    /// The head modulation is `bf16(bf16(silu(t)) . w + b)` of the audio timestep, shift then
    /// scale; the video timestep's row differs, so taking it fails.
    #[tokio::test(flavor = "current_thread")]
    async fn head_takes_its_timestep_row() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let (width, inputs) = (Hidden::SIZE, Time::SIZE);
        let time: Vec<f32> = random(Timestep::SIZE * inputs, 0x2545_f491_4f6c_dd1d)
            .iter()
            .map(|v| v.to_f32() * 4.0)
            .collect();
        let weight = random(2 * width * inputs, 0x9e37_79b9_7f4a_7c15);
        let bias = random(2 * width, 0x0123_4567_89ab_cdef);
        let modulation = HeadModulation::new(
            HostTensor::<bf16, m![{ Symbol<ShiftScale<Hidden>> }, Time]>::from_vec(weight.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            HostTensor::<f32, m![{ Symbol<ShiftScale<Hidden>> }, 1 # 8]>::from_vec(
                bias.iter()
                    .flat_map(|b| [b.to_f32(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    .collect::<Vec<f32>>(),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
        );
        let input = HostTensor::<f32, m![Timestep, Time]>::from_vec(time.clone())
            .to_hbm::<m![1], m![Timestep, Time]>(&mut device.pdma)
            .await
            .unwrap();
        let (shift, scale) = launch(shifted, (&mut device, &modulation, &input))
            .await
            .unwrap();
        let activated: Vec<f64> = time[inputs..]
            .iter()
            .map(|t| {
                let t = f64::from(*t);
                f64::from(bf16::from_f32((t / (1.0 + (-t).exp())) as f32).to_f32())
            })
            .collect();
        for (part, vector) in [shift, scale].iter().enumerate() {
            let got = vector
                .to_host::<m![Hidden]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for j in 0..width {
                let output = part * width + j;
                let expected: f64 = activated
                    .iter()
                    .zip(&weight[output * inputs..][..inputs])
                    .map(|(a, w)| a * f64::from(w.to_f32()))
                    .sum::<f64>()
                    + f64::from(bias[output].to_f32());
                let got = f64::from(got[j].to_f32());
                assert!(
                    (got - expected).abs() <= expected.abs() / 256.0 + 0.01,
                    "part {part} feature {j}: {got} vs {expected}"
                );
            }
        }
    }
}
