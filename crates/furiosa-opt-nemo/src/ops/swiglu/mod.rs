//! Packed gate/up projections compute down(silu(gate(x)) * up(x)) on BF16 rows.

use furiosa_opt_std::prelude::*;

use crate::ops::activation::SwiGlu;
use crate::ops::layout::{Dm, axes::Lanes};
use crate::ops::normalization::Spread;
use crate::{Module, ops::layout::Live, ops::linear::Groups, ops::linear::Linear};

use crate::ops::layout::axes::Chip;
use crate::ops::vector::Zip;

mod giant;
mod medium;
mod paired;
mod resident;
mod rows;
pub mod spans;
mod tiles;

/// Intermediate feature axis shared by dense feed-forward implementations.
#[derive(Clone, Debug)]
pub struct Intermediate<const N: usize, const PADDED: usize = N>;

impl<const N: usize, const PADDED: usize> AxisName for Intermediate<N, PADDED> {
    const NAME: Ident = Ident::new("Intermediate");
    const SIZE: usize = N;
}

/// The two weight matrices packed as gate followed by up.
#[derive(Debug, Clone)]
pub struct GateUp<I>(std::marker::PhantomData<I>);

impl<I: AxisName> AxisName for GateUp<I> {
    const NAME: Ident = Ident::new("GateUp");
    const SIZE: usize = I::SIZE * 2;
}

/// Storage width of the down projection, including its kernel's zero padding.
#[derive(Debug, Clone)]
pub struct Down<I>(std::marker::PhantomData<I>);

impl<const N: usize, const PADDED: usize> AxisName for Down<Intermediate<N, PADDED>> {
    const NAME: Ident = Intermediate::<N, PADDED>::NAME;
    const SIZE: usize = {
        assert!(PADDED >= N);
        PADDED
    };
}

/// SwiGLU with a packed gate/up projection and a down projection.
#[derive(DeviceSend)]
pub struct Mlp<
    Hidden: AxisName,
    Intermediate: AxisName,
    Fused = Linear<Hidden, GateUp<Intermediate>>,
    Projection = Linear<Down<Intermediate>, Hidden>,
> where
    Down<Intermediate>: AxisName,
{
    gate_up: Fused,
    down: Projection,
    features: core::marker::PhantomData<(Hidden, Intermediate)>,
}

impl<Hidden: AxisName, Intermediate: AxisName, Fused, Projection>
    Mlp<Hidden, Intermediate, Fused, Projection>
where
    Down<Intermediate>: AxisName,
{
    pub const fn new(gate_up: Fused, down: Projection) -> Self {
        Self {
            gate_up,
            down,
            features: core::marker::PhantomData,
        }
    }
}

// Gate then up in the packed projection.
axes![Term = 2];

/// One row group of an envelope of `Rows`, the envelope a live group runs alone as.
type Group<Rows> = crate::ops::layout::axes::Tile<Rows, 8>;

impl<Hidden: AxisName, Intermediate: AxisName> Mlp<Hidden, Intermediate>
where
    Down<Intermediate>: AxisName,
{
    /// `compute` of each live row group alone, run as an envelope of that one group; later rows
    /// land zeros. A group loads the weights its envelope loads, so this serves envelopes of at
    /// most two groups, where a step with a group past its live rows has one live group.
    fn each_group<Rows: AxisName>(
        &self,
        device: &mut Device,
        hidden: &Spread<Hidden, Rows>,
        live: Live<Rows>,
        compute: impl Fn(
            &Self,
            &mut Device,
            &Spread<Hidden, Group<Rows>>,
        ) -> Spread<Hidden, Group<Rows>>,
    ) -> Spread<Hidden, Rows> {
        type Chunk<Hidden> = crate::ops::normalization::Chunk<Hidden>;
        type Slices<Hidden> = crate::ops::normalization::FeatureSlices<Hidden>;
        type Cluster = crate::ops::normalization::Cluster;
        let mut output: Spread<Hidden, Rows> = DmTensor::new();
        output
            .view_mut()
            .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
        for group in 0..live.count() {
            let selected = hidden
                .view()
                .tile::<m![Rows / ({ Symbol<Group<Rows>> })], 1, m![
                    Rows / ({ Symbol<Group<Rows>> }) = 1 # (Rows / ({ Symbol<Group<Rows>> })),
                    Rows % ({ Symbol<Group<Rows>> }),
                    Hidden % ({ Symbol<Chunk<Hidden>> })
                ]>(group);
            // SAFETY: the group's rows keep their cells and strides; only the row axis is renamed.
            let selected = unsafe {
                selected.reshape::<Chip, Cluster, Slices<Hidden>, m![
                    { Symbol<Group<Rows>> } # Rows, Hidden % ({ Symbol<Chunk<Hidden>> })
                ]>()
            }
            .unpad::<m![{ Symbol<Group<Rows>> }, Hidden % ({ Symbol<Chunk<Hidden>> })]>();
            let mut rows: Spread<Hidden, Group<Rows>> = DmTensor::new();
            selected.to_dm_view(&mut device.tdma, rows.view_mut());
            let result = compute(self, device, &rows);
            // SAFETY: the destination's row tile supplies the group's outer offset.
            let result = unsafe {
                result.view().reshape::<Chip, Cluster, Slices<Hidden>, m![
                    Rows % ({ Symbol<Group<Rows>> }), Hidden % ({ Symbol<Chunk<Hidden>> })
                ]>()
            };
            result.to_dm_view(
                &mut device.tdma,
                output.view_mut().tile::<m![Rows / ({ Symbol<Group<Rows>> })], 1, m![
                    Rows / ({ Symbol<Group<Rows>> }) = 1 #{!} (Rows / ({ Symbol<Group<Rows>> })),
                    Rows % ({ Symbol<Group<Rows>> }),
                    Hidden % ({ Symbol<Chunk<Hidden>> })
                ]>(group),
            );
        }
        output
    }
}

/// A launch's live rows through the whole MLP, at the catalog's MLP position.
impl<'d, 'x, H: AxisName, I: AxisName, R: AxisName>
    Module<(&'d mut Device, &'x Dm<Lanes<R>, Spread<H, R>>)> for Mlp<H, I>
where
    Down<I>: AxisName,
    Self: for<'e, 'y> Module<(&'e mut Device, (&'y Spread<H, R>, Live<R>)), Output = Spread<H, R>>,
{
    type Output = Dm<Lanes<R>, Spread<H, R>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Dm<Lanes<R>, Spread<H, R>>),
    ) -> Self::Output {
        Dm {
            value: self.forward((device, (&input.value, input.tiles))),
            tiles: input.tiles,
        }
    }
}

/// Each row's gate half and up half land in feature slices of their own; the SiLU of the gate
/// rounds to BF16 before it multiplies the up half, as a separate activation after a packed
/// projection rounds.
impl<'d, 'x, I: AxisName, Rows: AxisName>
    Module<(&'d mut Device, &'x Dm<Lanes<Rows>, Spread<GateUp<I>, Rows>>)> for SwiGlu
{
    type Output = Dm<Lanes<Rows>, Spread<I, Rows>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Dm<Lanes<Rows>, Spread<GateUp<I>, Rows>>),
    ) -> Self::Output {
        let packed: Packed<I, Rows> = input.value.to_hbm(&mut device.tdma);
        Dm {
            value: self.forward((device, (packed, input.tiles))),
            tiles: input.tiles,
        }
    }
}

/// Rows of the packed gate/up pair in HBM, each row's gate features before its up features.
pub(crate) type Packed<I, Rows> = HbmTensor<bf16, Chip, m![Rows, { Symbol<GateUp<I>> }]>;

/// As the feature-slice form, from the pair's rows in HBM.
impl<'d, I: AxisName, Rows: AxisName> Module<(&'d mut Device, (Packed<I, Rows>, Live<Rows>))>
    for SwiGlu
{
    type Output = Spread<I, Rows>;

    fn forward(
        &self,
        (device, (packed, _)): (&'d mut Device, (Packed<I, Rows>, Live<Rows>)),
    ) -> Self::Output {
        type C<I> = Symbol<crate::ops::normalization::Chunk<I>>;
        // SAFETY: a packed row is its gate features followed by its up features.
        let packed: HbmTensor<bf16, Chip, m![Rows, Term, I]> = unsafe { packed.reshape() };
        let half = |device: &mut Device, term: usize| {
            let mut half: Spread<I, Rows> = DmTensor::new();
            packed
                .view()
                .tile::<m![Term], 1, m![Rows, Term = 1 # 2, I]>(term)
                .to_dm_view(&mut device.tdma, half.view_mut());
            half
        };
        let (gate, up) = (half(device, 0), half(device, 1));
        let activated: Spread<I, Rows> = device
            .main
            .begin(gate.view())
            .fetch::<m![Rows, I % ({ C<I> }) / 8], m![I % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows, I % ({ C<I> }) / 8], m![I % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows, I % ({ C<I> }) / 8, I % 8 / 4 % 2], m![I % 4]>()
            .vector_stash()
            .vector_fp_unary(FpUnaryOp::Sigmoid)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
            .vector_widen_concat::<m![Rows, I % ({ C<I> }) / 8], m![I % 8]>()
            .vector_final()
            .cast::<bf16, m![I % 8 # 16]>()
            .commit_trim::<m![I % 8]>()
            .commit();
        let product: Spread<I, Rows> = device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(activated.view(), up.view())
            .fetch::<m![Rows, Term, I % ({ C<I> }) / 8], m![I % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows, Term, I % ({ C<I> }) / 8], m![I % 8]>()
            .vector_init()
            .vector_zip::<
                Term,
                m![Rows, I % ({ C<I> }) / 8],
                m![Rows, I % ({ C<I> }) / 4],
                m![I % 4],
            >(FpBinaryOp::MulF(FpMulAlu::Mul0))
            .vector_final()
            .cast::<bf16, m![I % 8 # 16]>()
            .commit_trim::<m![I % 8]>()
            .commit();
        product
    }
}

/// Rows in HBM, a tile of rows at a time; as for rows on chip, the SiLU of the gate rounds to
/// BF16 before it multiplies the up half.
impl<'d, 'x, I: AxisName, R: AxisName, Chips: M>
    Module<(
        &'d mut Device,
        &'x HbmTensor<bf16, Chips, m![R, { Symbol<GateUp<I>> }]>,
    )> for SwiGlu
{
    type Output = HbmTensor<bf16, Chips, m![R, I]>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            &'x HbmTensor<bf16, Chips, m![R, { Symbol<GateUp<I>> }]>,
        ),
    ) -> HbmTensor<bf16, Chips, m![R, I]> {
        let mut output = HbmTensor::new();
        SwiGlu.forward((device, (input, &mut output)));
        output
    }
}
