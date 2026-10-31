//! SwiGLU of gate/up rows held in HBM, a row on eight slices.
//!
//! - A tile is a band of 64 rows, the rows a projection writes at a time: each cluster's 32 rows
//!   take eight slices each, each slice one contiguous eighth of a row's features.
//! - The gate's SiLU rounds to BF16 before it multiplies the up half, as for rows on chip.
//! - The next tile arrives while the current one computes, and a tile's product is stored while the
//!   next tile computes.

use crate::specialize::contraction::BAND;
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use super::{GateUp, Term};
use crate::ops::vector::Silu as _;
use crate::ops::vector::Zip;
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use crate::{Op, ops::pointwise::SwiGlu};

/// The slices a row's features spread over: the slices of a cluster over its half of the rows.
impl<R: AxisName> GateTile<R> {
    const PARTS: usize = 256 / (Self::SIZE / 2);
}

/// The rows of one tile, both clusters'.
#[derive(Clone, Debug)]
pub(crate) struct GateTile<R>(PhantomData<(R,)>);

impl<R: AxisName> AxisName for GateTile<R> {
    const NAME: Ident = Ident::new("GateTile");
    const SIZE: usize = {
        assert!(
            R::SIZE.is_multiple_of(2 * BAND),
            "rows come in pairs of bands"
        );
        BAND
    };
}

/// The tiles of the rows, in row order.
#[derive(Clone, Debug)]
pub(crate) struct GateTiles<R>(PhantomData<(R,)>);

impl<R: AxisName> AxisName for GateTiles<R> {
    const NAME: Ident = Ident::new("GateTiles");
    const SIZE: usize = R::SIZE / GateTile::<R>::SIZE;
}

/// Rows a cluster holds of a tile.
#[derive(Clone, Debug)]
pub(crate) struct GateHalf<R>(PhantomData<(R,)>);

impl<R: AxisName> AxisName for GateHalf<R> {
    const NAME: Ident = Ident::new("GateHalf");
    const SIZE: usize = GateTile::<R>::SIZE / 2;
}

/// Features of a row one slice holds.
#[derive(Clone, Debug)]
pub(crate) struct GatePart<I, R>(PhantomData<(I, R)>);

impl<I: AxisName, R: AxisName> AxisName for GatePart<I, R> {
    const NAME: Ident = Ident::new("GatePart");
    const SIZE: usize = {
        assert!(I::SIZE.is_multiple_of(64 * GateTile::<R>::PARTS));
        I::SIZE / GateTile::<R>::PARTS
    };
}

type T<R> = Symbol<GateTile<R>>;
type Ts<R> = Symbol<GateTiles<R>>;
type H<R> = Symbol<GateHalf<R>>;
type F<I, R> = Symbol<GatePart<I, R>>;

type Cluster<R> = m![{ T<R> } / ({ H<R> })];
type Slices<I, R> = m![{ T<R> } % ({ H<R> }), I / ({ F<I, R> })];
/// One part of one row's features.
type Part<I, R, Chips = Chip> =
    DmTensor<bf16, Chips, Cluster<R>, Slices<I, R>, m![I % ({ F<I, R> })]>;

/// A span's packed gate/up rows as their tiles.
type Tiled<'a, I, R, Chips> = HbmTensorView<'a, bf16, Chips, m![{ Ts<R> }, { T<R> }, Term, I]>;

/// `bf16(bf16(silu(gate)) * up)` of every row of `input` into `output`.
impl<'x, 'y, I: AxisName, R: AxisName, Chips: M>
    Op<(
        &'x HbmTensor<bf16, Chips, m![R, { Symbol<GateUp<I>> }]>,
        &'y mut HbmTensor<bf16, Chips, m![R, I]>,
    )> for SwiGlu
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, output): (
            &'x HbmTensor<bf16, Chips, m![R, { Symbol<GateUp<I>> }]>,
            &'y mut HbmTensor<bf16, Chips, m![R, I]>,
        ),
    ) {
        let tiles = <Ts<R> as M>::SIZE;
        let rows = tiled::<I, R, Chips>(input);
        // Gate and up halves of the two tiles in flight.
        let (mut gate, mut up): (Part<I, R, Chips>, Part<I, R, Chips>) =
            (DmTensor::new(), DmTensor::new());
        let (mut next_gate, mut next_up): (Part<I, R, Chips>, Part<I, R, Chips>) =
            (DmTensor::new(), DmTensor::new());
        let mut kept: Part<I, R, Chips> = DmTensor::new();
        let mut held: Part<I, R, Chips> = DmTensor::new();
        load(device, &rows, 0, (&mut gate, &mut up));
        if const { <Ts<R> as M>::SIZE > 1 } {
            load(device, &rows, 1, (&mut next_gate, &mut next_up));
        }
        // Tile t computes while tile t + 1 is written and tile t - 1 is stored.
        for pair in 0..tiles / 2 {
            let at = 2 * pair;
            multiply(device, (&gate, &up), &mut kept);
            load(device, &rows, (at + 2).min(tiles - 1), (&mut gate, &mut up));
            store(device, &kept, output, at);
            multiply(device, (&next_gate, &next_up), &mut held);
            load(
                device,
                &rows,
                (at + 3).min(tiles - 1),
                (&mut next_gate, &mut next_up),
            );
            store(device, &held, output, at + 1);
        }
        if const { <Ts<R> as M>::SIZE % 2 == 1 } {
            multiply(device, (&gate, &up), &mut kept);
            store(device, &kept, output, tiles - 1);
        }
    }
}

/// `bf16(bf16(silu(gate)) * up)` of the tiles of `input` that hold the live rows `live`, into the
/// same tiles of `output`; the later tiles are left as they were.
impl<'x, 'y, I: AxisName, R: AxisName, Chips: M, Capacity: AxisName>
    Op<(
        &'x HbmTensor<bf16, Chips, m![R, { Symbol<GateUp<I>> }]>,
        &'y mut HbmTensor<bf16, Chips, m![R, I]>,
        Live<(), Capacity>,
    )> for SwiGlu
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, output, live): (
            &'x HbmTensor<bf16, Chips, m![R, { Symbol<GateUp<I>> }]>,
            &'y mut HbmTensor<bf16, Chips, m![R, I]>,
            Live<(), Capacity>,
        ),
    ) {
        let tiles = live.rows().div_ceil(<T<R> as M>::SIZE);
        let rows = tiled::<I, R, Chips>(input);
        let (mut gate, mut up): (Part<I, R, Chips>, Part<I, R, Chips>) =
            (DmTensor::new(), DmTensor::new());
        let mut kept: Part<I, R, Chips> = DmTensor::new();
        // Tile t computes while tile t + 1 is written.
        load(device, &rows, 0, (&mut gate, &mut up));
        for at in 0..tiles {
            multiply(device, (&gate, &up), &mut kept);
            load(device, &rows, (at + 1).min(tiles - 1), (&mut gate, &mut up));
            store(device, &kept, output, at);
        }
    }
}

/// The packed gate/up rows of `input` as their tiles.
fn tiled<I: AxisName, R: AxisName, Chips: M>(
    input: &HbmTensor<bf16, Chips, m![R, { Symbol<GateUp<I>> }]>,
) -> Tiled<'_, I, R, Chips> {
    // SAFETY: a packed row is its gate features followed by its up features, and the tiles split
    // the rows in order.
    unsafe {
        input
            .view()
            .reshape::<Chips, m![{ Ts<R> }, { T<R> }, Term, I]>()
    }
}

/// Writes the gate and up halves of tile `at` of `rows` to `gate` and `up`.
fn load<I: AxisName, R: AxisName, Chips: M>(
    device: &mut Device,
    rows: &Tiled<'_, I, R, Chips>,
    at: usize,
    (gate, up): (&mut Part<I, R, Chips>, &mut Part<I, R, Chips>),
) {
    let tile =
        rows.tile::<m![{ Ts<R> }], 1, m![{ Ts<R> } = 1 # ({ Ts<R> }), { T<R> }, Term, I]>(at);
    tile.tile::<m![Term], 1, m![{ Ts<R> } = 1 # ({ Ts<R> }), { T<R> }, Term = 1 # 2, I]>(0)
        .to_dm_view(&mut device.tdma, gate.view_mut());
    tile.tile::<m![Term], 1, m![{ Ts<R> } = 1 # ({ Ts<R> }), { T<R> }, Term = 1 # 2, I]>(1)
        .to_dm_view(&mut device.tdma, up.view_mut());
}

/// `bf16(bf16(silu(gate)) * up)` of one tile's halves into `product`.
fn multiply<I: AxisName, R: AxisName, Chips: M>(
    device: &mut Device,
    (gate, up): (&Part<I, R, Chips>, &Part<I, R, Chips>),
    product: &mut Part<I, R, Chips>,
) {
    let activated: Part<I, R, Chips> = device
        .main
        .begin(gate.view())
        .fetch::<m![I % ({ F<I, R> }) / 8], m![I % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![I % ({ F<I, R> }) / 8], m![I % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![I % ({ F<I, R> }) / 8, I % 8 / 4], m![I % 4]>()
        .vector_silu()
        .vector_widen_concat::<m![I % ({ F<I, R> }) / 8], m![I % 8]>()
        .vector_final()
        .cast::<bf16, m![I % 8 # 16]>()
        .commit_trim::<m![I % 8]>()
        .commit();
    device.main.begin_interleaved::<Term, _, _, _, _, _>(activated.view(), up.view())
        .fetch::<m![I % ({ F<I, R> }) / 64, Term, I % 64 / 8], m![I % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![I % ({ F<I, R> }) / 64, Term, I % 64 / 8], m![I % 8]>()
        .vector_init()
        .vector_zip::<
            Term,
            m![I % ({ F<I, R> }) / 64, I % 64 / 8],
            m![I % ({ F<I, R> }) / 64, I % 64 / 4],
            m![I % 4],
        >(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_final()
        .cast::<bf16, m![I % 8 # 16]>()
        .commit_trim::<m![I % 8]>()
        .commit_view(product.view_mut());
}

/// Stores `product` as tile `at` of `output`.
fn store<I: AxisName, R: AxisName, Chips: M>(
    device: &mut Device,
    product: &Part<I, R, Chips>,
    output: &mut HbmTensor<bf16, Chips, m![R, I]>,
    at: usize,
) {
    // SAFETY: the tiles split the rows in order.
    let rows = unsafe {
        output
            .view_mut()
            .reshape::<Chips, m![{ Ts<R> }, { T<R> }, I]>()
    };
    product.view().to_hbm_view(
        &mut device.tdma,
        rows.tile::<m![{ Ts<R> }], 1, m![{ Ts<R> } = 1 #{!} ({ Ts<R> }), { T<R> }, I]>(at),
    );
}
