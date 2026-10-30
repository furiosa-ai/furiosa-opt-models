//! The tiling of packed rows in HBM in blocks of 512 rows (256 when the rows do not come in whole
//! 512-row blocks), eight (four) rows a slice, and its gated residual pass.
//!
//! - A block puts each cluster's half of its rows on 32 groups of slices, each slice one contiguous
//!   run of each of its rows' features in whole 256-byte bursts; per-feature operands repeat over
//!   a slice's rows, so a pass reads each operand vector for all of them in a row.
//! - A span's strips before its first whole block and after its last run as strips.

use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use super::tiling::{self, Tiling};
use crate::Op;
use crate::ops::index::packed::{Rows, Table};
use crate::ops::normalization::{self, Squares};
use crate::specialize::diffusion::Run;

/// Rows of a block of packed rows: 512 when the rows come in whole 512-row blocks, else 256.
#[derive(Clone, Debug)]
pub(crate) struct Block<R>(PhantomData<R>);

impl<R: AxisName> AxisName for Block<R> {
    const NAME: Ident = Ident::new("Block");
    const SIZE: usize = if R::SIZE.is_multiple_of(512) {
        512
    } else {
        assert!(
            R::SIZE.is_multiple_of(256),
            "packed rows come in whole 256-row blocks"
        );
        256
    };
}

/// Rows a cluster holds of a block.
#[derive(Clone, Debug)]
pub(crate) struct Halves<R>(PhantomData<R>);

impl<R: AxisName> AxisName for Halves<R> {
    const NAME: Ident = Ident::new("BlockHalves");
    const SIZE: usize = Block::<R>::SIZE / 2;
}

/// Rows a slice holds of a block.
#[derive(Clone, Debug)]
pub(crate) struct Lanes<R>(PhantomData<R>);

impl<R: AxisName> AxisName for Lanes<R> {
    const NAME: Ident = Ident::new("BlockLanes");
    const SIZE: usize = Block::<R>::SIZE / 64;
}

/// The blocks of the rows, in row order.
#[derive(Clone, Debug)]
pub(crate) struct Blocks<R>(PhantomData<R>);

impl<R: AxisName> AxisName for Blocks<R> {
    const NAME: Ident = Ident::new("Blocks");
    const SIZE: usize = R::SIZE / Block::<R>::SIZE;
}

type E<W> = Symbol<Run<W>>;
type Bk<R> = Symbol<Block<R>>;
type Hb<R> = Symbol<Halves<R>>;
type Ln<R> = Symbol<Lanes<R>>;
type Half<R> = m![{ Bk<R> } / ({ Hb<R> })];
type Slices<R, W> = m![{ Bk<R> } % ({ Hb<R> }) / ({ Ln<R> }), W / ({ E<W> }) # 8];
pub(crate) type Part<D, R, W, X> =
    DmTensor<D, X, Half<R>, Slices<R, W>, m![{ Bk<R> } % ({ Ln<R> }), W % ({ E<W> })]>;
type Operand<R, W, X> = VrfTensor<f32, X, Half<R>, Slices<R, W>, m![W % ({ E<W> })]>;
/// How a slice's rows of a block move through a norm's statistic pass.
type BlockLanes<R, W> = normalization::Lanes<
    m![{ Bk<R> } % ({ Ln<R> }), W % ({ E<W> }) / 8],
    m![W % 8],
    m![{ Bk<R> } % ({ Ln<R> }), W % ({ E<W> }) / 8, W % 8 / 4 % 2],
    m![W % 4],
    m![{ Bk<R> } % ({ Ln<R> }), W % ({ E<W> }) / 8],
    m![W % 8],
    m![W % 8],
    m![{ Bk<R> } % ({ Hb<R> }) / ({ Ln<R> }), 8],
    m![{ Bk<R> } % ({ Ln<R> }), 1 # 8],
>;

/// A span `(first strip, strips, table row)` as its strips before the first whole block, the
/// whole blocks, and the strips after, each `(first, count)`.
impl<R: AxisName> Block<R> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
    /// Every span's three regions, each `(first, count, table row)`, in packing order.
    pub(crate) const fn plan(
        spans: [(usize, usize, usize); 3],
        edge: usize,
    ) -> [[(usize, usize, usize); 3]; 3] {
        let mut plan = [[(0, 0, 0); 3]; 3];
        let mut span = 0;
        while span < 3 {
            let regions = {
                let (first, strips, _) = spans[span];
                let each = Self::SIZE / edge;
                let end = first + strips;
                let (start, stop) = (first.div_ceil(each), end / each);
                if stop <= start {
                    [(first, strips), (0, 0), (end, 0)]
                } else {
                    [
                        (first, each * start - first),
                        (start, stop - start),
                        (each * stop, end - each * stop),
                    ]
                }
            };
            let mut region = 0;
            while region < 3 {
                plan[span][region] = (regions[region].0, regions[region].1, spans[span].2);
                region += 1;
            }
            span += 1;
        }
        plan
    }

    /// Table row `index`.
    pub(crate) fn row<W: AxisName, X: M>(
        table: &Table<W, X>,
        index: usize,
    ) -> HbmTensorView<'_, bf16, X, m![W]> {
        tiling::row(table, index)
    }
}

impl<R: AxisName, W: AxisName> Tiling<W> for Block<R> {
    type Tiles<Q: AxisName> = Symbol<Blocks<Q>>;
    type Rows = Bk<R>;
    type Half = Half<R>;
    type Slices = Slices<R, W>;
    type Element = m![{ Bk<R> } % ({ Ln<R> }), W % ({ E<W> })];
    type Run = E<W>;
    type Statistic = m![{ Bk<R> } % ({ Ln<R> }), 1 # 8];
    type Modulated = m![W % ({ E<W> }) / 8, { Bk<R> } % ({ Ln<R> })];
    type Split = m![W % ({ E<W> }) / 8, { Bk<R> } % ({ Ln<R> }), W % 8 / 4 % 2];

    fn inverse<const EPSILON: usize, X: M>(
        device: &mut Device,
        part: &Part<bf16, R, W, X>,
    ) -> tiling::RowOperand<Self, W, X> {
        Squares::<
            BlockLanes<R, W>,
            W,
            m![{ Bk<R> } % ({ Ln<R> })],
            m![W / ({ E<W> })],
            EPSILON,
        >::inverse(device, part.view(), |square| square)
    }
}

/// A per-feature vector, in the vector registers of the slices that hold its features.
impl<R: AxisName, W: AxisName, X: M> Op<HbmTensorView<'_, bf16, X, m![W]>> for Block<R> {
    type Output = Operand<R, W, X>;

    fn forward(
        &self,
        device: &mut Device,
        vector: HbmTensorView<'_, bf16, X, m![W]>,
    ) -> Self::Output {
        Self::operand(device, vector)
    }
}

/// Block `at` of a residual and of its branch, each into its part.
impl<R: AxisName, W: AxisName, X: M>
    Op<(
        (&Rows<R, W, X>, &Rows<R, W, X>),
        usize,
        &mut (Part<bf16, R, W, X>, Part<bf16, R, W, X>),
    )> for Block<R>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        ((input, branch), at, (rows, update)): (
            (&Rows<R, W, X>, &Rows<R, W, X>),
            usize,
            &mut (Part<bf16, R, W, X>, Part<bf16, R, W, X>),
        ),
    ) {
        Self::fill(&mut *device, input, at, rows);
        Self::fill(device, branch, at, update);
    }
}

/// A part into block `at` of `rows`.
impl<R: AxisName, W: AxisName, X: M> Op<(&Part<bf16, R, W, X>, &mut Rows<R, W, X>, usize)>
    for Block<R>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (part, rows, at): (&Part<bf16, R, W, X>, &mut Rows<R, W, X>, usize),
    ) {
        Self::spill(device, part, rows, at);
    }
}

axes![Term = 2];

/// `bf16(x + gate * branch)` of a copied block into `output`, in one pass over both terms.
impl<R: AxisName, W: AxisName, X: M>
    Op<(
        (&Part<bf16, R, W, X>, &Part<bf16, R, W, X>),
        &Operand<R, W, X>,
        &mut Part<bf16, R, W, X>,
    )> for Block<R>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        ((input, branch), gate, output): (
            (&Part<bf16, R, W, X>, &Part<bf16, R, W, X>),
            &Operand<R, W, X>,
            &mut Part<bf16, R, W, X>,
        ),
    ) {
        device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(input.view(), branch.view())
        .fetch::<m![W % ({ E<W> }) / 64, { Bk<R> } % ({ Ln<R> }), Term, W % 64 / 8], m![W % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![W % ({ E<W> }) / 64, { Bk<R> } % ({ Ln<R> }), Term, W % 64 / 8], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<
            Term,
            m![W % ({ E<W> }) / 64, { Bk<R> } % ({ Ln<R> }), W % 64 / 8],
        >()
        .vector_narrow_split::<
            m![W % ({ E<W> }) / 64, { Bk<R> } % ({ Ln<R> }), W % 64 / 4],
            m![W % 4],
        >()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), (), gate)
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_widen_concat::<
            m![W % ({ E<W> }) / 64, { Bk<R> } % ({ Ln<R> }), W % 64 / 8],
            m![W % 8],
        >()
        .vector_final()
        .cast::<bf16, m![W % 8 # 16]>()
        .commit_trim::<m![W % 8]>()
        .commit_view(output.view_mut());
    }
}
