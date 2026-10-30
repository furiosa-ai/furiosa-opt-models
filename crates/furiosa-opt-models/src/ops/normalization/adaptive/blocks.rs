//! The modulated norm and the gated residual passes of packed rows in HBM in blocks of 512 rows
//! (256 when the rows do not come in whole 512-row blocks), eight (four) rows a slice.
//!
//! - A block puts each cluster's half of its rows on 32 groups of slices, each slice one contiguous
//!   run of each of its rows' features in whole 256-byte bursts; per-feature operands repeat over
//!   a slice's rows, so a pass reads each operand vector for all of them in a row.
//! - A span's strips before its first whole block and after its last run as strips.

use core::borrow::Borrow;
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use super::rows::Narrow;
use super::rows::Wide;
use crate::Op;
use crate::ops::index::packed::{Group, Join, Packed, Rows, Table};
use crate::ops::normalization::{self, Root, Squares, Widen};
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
type Bs<R> = Symbol<Blocks<R>>;
type Blocked<R, W> = m![{ Bs<R> }, { Bk<R> }, W];
type Half<R> = m![{ Bk<R> } / ({ Hb<R> })];
type Slices<R, W> = m![{ Bk<R> } % ({ Hb<R> }) / ({ Ln<R> }), W / ({ E<W> }) # 8];
pub(crate) type Part<D, R, W, X> =
    DmTensor<D, X, Half<R>, Slices<R, W>, m![{ Bk<R> } % ({ Ln<R> }), W % ({ E<W> })]>;
type Feature<R, W, X> = DmTensor<bf16, X, Half<R>, Slices<R, W>, m![W % ({ E<W> })]>;
type Operand<R, W, X> = VrfTensor<f32, X, Half<R>, Slices<R, W>, m![W % ({ E<W> })]>;
type RowOperand<R, W, X> =
    VrfTensor<f32, X, Half<R>, Slices<R, W>, m![{ Bk<R> } % ({ Ln<R> }), 1 # 8]>;
/// How a slice's rows of a block move through a norm's statistic pass.
type BlockLanes<R, W> = normalization::Lanes<
    m![{ Bk<R> } % ({ Ln<R> }), W % ({ E<W> }) / 8],
    m![W % 8],
    m![{ Bk<R> } % ({ Ln<R> }), W % ({ E<W> }) / 8, W % 8 / 4 % 2],
    m![W % 4],
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
}

impl<R: AxisName, W: AxisName, X: M> Op<(&Rows<R, W, X>, usize, &mut Part<bf16, R, W, X>)>
    for Block<R>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (rows, at, part): (&Rows<R, W, X>, usize, &mut Part<bf16, R, W, X>),
    ) {
        // SAFETY: the blocks split the row axis in row order; every cell keeps its address.
        unsafe { rows.view().reshape::<X, Blocked<R, W>>() }
            .tile::<m![{ Bs<R> }], 1, m![{ Bs<R> } = 1 # ({ Bs<R> }), { Bk<R> }, W]>(at)
            .to_dm_view(&mut device.tdma, part.view_mut());
    }
}
impl<R: AxisName, W: AxisName, X: M> Op<(&Part<bf16, R, W, X>, &mut Rows<R, W, X>, usize)>
    for Block<R>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (part, rows, at): (&Part<bf16, R, W, X>, &mut Rows<R, W, X>, usize),
    ) {
        part.view().to_hbm_view(
            &mut device.tdma,
            // SAFETY: the blocks split the row axis in row order; every cell keeps its address.
            unsafe { rows.view_mut().reshape::<X, Blocked<R, W>>() }
                .tile::<m![{ Bs<R> }], 1, m![{ Bs<R> } = 1 #{!} ({ Bs<R> }), { Bk<R> }, W]>(at),
        );
    }
}
impl<R: AxisName, W: AxisName, X: M> Op<HbmTensorView<'_, bf16, X, m![W]>> for Block<R> {
    type Output = Operand<R, W, X>;

    fn forward(
        &self,
        device: &mut Device,
        vector: HbmTensorView<'_, bf16, X, m![W]>,
    ) -> Self::Output {
        let in_dm: Feature<R, W, X> = vector.to_dm(&mut device.tdma);
        in_dm.widen::<W, 8>(&mut device.sub)
    }
}
impl<R: AxisName> Block<R> {
    /// Table row `index`.
    pub(crate) fn row<W: AxisName, X: M>(
        table: &Table<W, X>,
        index: usize,
    ) -> HbmTensorView<'_, bf16, X, m![W]> {
        // SAFETY: a table row is its features, contiguous.
        unsafe {
            table
                .view()
                .tile::<m![Group], 1, m![Group = 1 # Group, W]>(index)
                .reshape::<X, m![W]>()
        }
    }
}
impl<R: AxisName, W: AxisName, X: M, const EPSILON: usize, G>
    Op<(&Part<bf16, R, W, X>, &super::AdaptiveRms<W, EPSILON, X, G>)> for Block<R>
{
    type Output = RowOperand<R, W, X>;

    fn forward(
        &self,
        device: &mut Device,
        (part, _): (&Part<bf16, R, W, X>, &super::AdaptiveRms<W, EPSILON, X, G>),
    ) -> Self::Output {
        let variance: DmTensor<
            f32,
            X,
            Half<R>,
            m![{ Bk<R> } % ({ Hb<R> }) / ({ Ln<R> }), 8],
            m![{ Bk<R> } % ({ Ln<R> }), 1 # 8],
        > =
            Squares::<
                BlockLanes<R, W>,
                W,
                m![{ Bk<R> } % ({ Ln<R> })],
                m![W / ({ E<W> })],
                EPSILON,
            >::of(
                &mut device.main,
                part.view(),
            )
        .vector_inter_slice_reduce::<
            m![{ Bk<R> } % ({ Hb<R> }) / ({ Ln<R> }), 8],
            m![{ Bk<R> } % ({ Ln<R> })],
        >(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
        // SAFETY: every slice of a group holds its rows' statistics; each row keeps its address.
        let variance: DmTensor<f32, X, Half<R>, Slices<R, W>, m![{ Bk<R> } % ({ Ln<R> }), 1 # 8]> =
            unsafe { variance.reshape() };
        Root::<m![{ Bk<R> } % ({ Ln<R> })], m![1 # 8]>::of(
            &mut device.sub,
            variance.view(),
            |square| square,
        )
        .to_vrf()
    }
}
impl<R: AxisName, W: AxisName, X: M>
    Op<(
        &Part<bf16, R, W, X>,
        (&RowOperand<R, W, X>, &Operand<R, W, X>, &Operand<R, W, X>),
        &mut Part<bf16, R, W, X>,
    )> for Block<R>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (part, (inverse, gain, shift), output): (
            &Part<bf16, R, W, X>,
            (&RowOperand<R, W, X>, &Operand<R, W, X>, &Operand<R, W, X>),
            &mut Part<bf16, R, W, X>,
        ),
    ) {
        device
        .main
        .begin(part.view())
        .fetch::<m![W % ({ E<W> }) / 8, { Bk<R> } % ({ Ln<R> })], m![W % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![W % ({ E<W> }) / 8, { Bk<R> } % ({ Ln<R> })], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<
            m![W % ({ E<W> }) / 8, { Bk<R> } % ({ Ln<R> }), W % 8 / 4 % 2],
            m![W % 4],
        >()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), inverse)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), gain)
        .vector_fp_binary(FpBinaryOp::AddF, shift)
        .vector_widen_concat::<m![W % ({ E<W> }) / 8, { Bk<R> } % ({ Ln<R> })], m![W % 8]>()
        .vector_final()
        .cast::<bf16, m![W % 8 # 16]>()
        .commit_trim::<m![W % 8]>()
        .commit_view(output.view_mut());
    }
}

/// Every span of a packed sequence, RMS-normalized with `weight` and modulated by its rows of
/// `(shift, scale)`: its edge strips as strips, its whole blocks as blocks.
impl<T: AxisName, A: AxisName, V: AxisName, Width: AxisName, Chips: M, const EPSILON: usize, G>
    Op<(
        &Rows<Packed<T, A, V>, Width, Chips>,
        &mut Rows<Packed<T, A, V>, Width, Chips>,
        &(Table<Width, Chips>, Table<Width, Chips>),
    )> for super::AdaptiveRms<Width, EPSILON, Chips, G>
where
    G: Borrow<HbmTensor<bf16, Chips, m![Width]>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, output, tables): (
            &Rows<Packed<T, A, V>, Width, Chips>,
            &mut Rows<Packed<T, A, V>, Width, Chips>,
            &(Table<Width, Chips>, Table<Width, Chips>),
        ),
    ) {
        let weight = self.weight.tensor();
        // Each span's strips before its first whole block, its whole blocks, then its strips after;
        // strips are 64 rows when every span comes in them, else 16.
        let plan = const {
            Block::<Packed<T, A, V>>::plan(Join::spans::<T, A, V>(), Packed::<T, A, V>::STRIP)
        };
        #[unroll]
        for span in 0..3 {
            #[unroll]
            for region in 0..3 {
                let (first, count, index) = plan[span][region];
                if count > 0 {
                    if region == 1 {
                        type R<T, A, V> = Packed<T, A, V>;
                        type B<T, A, V> = Block<R<T, A, V>>;
                        let (shift, scale) = tables;
                        let gain = {
                            let (weight, scale) = (weight.view(), B::<T, A, V>::row(scale, index));
                            let weight = B::<T, A, V>::new().forward(&mut *device, weight);
                            let scale: Feature<R<T, A, V>, Width, Chips> =
                                scale.to_dm(&mut device.tdma);
                            device.main.begin(scale.view())
                                .fetch::<m![Width % ({ E<Width> }) / 8], m![Width % 8]>()
                                .fetch_cast::<f32>()
                                .collect::<m![Width % ({ E<Width> }) / 8], m![Width % 8]>()
                                .vector_init()
                                .vector_intra_slice_tag(TagMode::Zero)
                                .vector_narrow_split::<
                                    m![Width % ({ E<Width> }) / 8, Width % 8 / 4 % 2],
                                    m![Width % 4],
                                >()
                                .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)
                                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight)
                                .vector_widen_concat::<
                                    m![Width % ({ E<Width> }) / 8],
                                    m![Width % 8],
                                >()
                                .vector_final()
                                .to_vrf(&mut device.sub)
                        };
                        let shift = B::<T, A, V>::new()
                            .forward(&mut *device, B::<T, A, V>::row(shift, index));
                        let last = first + count - 1;
                        let (mut current, mut next): (
                            Part<bf16, R<T, A, V>, Width, Chips>,
                            Part<bf16, R<T, A, V>, Width, Chips>,
                        ) = (DmTensor::new(), DmTensor::new());
                        let (mut kept, mut held): (
                            Part<bf16, R<T, A, V>, Width, Chips>,
                            Part<bf16, R<T, A, V>, Width, Chips>,
                        ) = (DmTensor::new(), DmTensor::new());
                        B::<T, A, V>::new().forward(&mut *device, (input, first, &mut current));
                        B::<T, A, V>::new()
                            .forward(&mut *device, (input, (first + 1).min(last), &mut next));
                        for pair in 0..(count + 1) / 2 {
                            let at = first + 2 * pair;
                            let inverse =
                                B::<T, A, V>::new().forward(&mut *device, (&current, self));
                            let following =
                                B::<T, A, V>::new().forward(&mut *device, (&next, self));
                            B::<T, A, V>::new().forward(
                                &mut *device,
                                (&current, (&inverse, &gain, &shift), &mut kept),
                            );
                            B::<T, A, V>::new()
                                .forward(&mut *device, (input, (at + 2).min(last), &mut current));
                            B::<T, A, V>::new().forward(&mut *device, (&kept, &mut *output, at));
                            B::<T, A, V>::new().forward(
                                &mut *device,
                                (&next, (&following, &gain, &shift), &mut held),
                            );
                            B::<T, A, V>::new()
                                .forward(&mut *device, (input, (at + 3).min(last), &mut next));
                            B::<T, A, V>::new()
                                .forward(&mut *device, (&held, &mut *output, (at + 1).min(last)));
                        }
                    } else if const { Packed::<T, A, V>::STRIP == 64 } {
                        Wide::<Width>::new().forward(
                            &mut *device,
                            (input, &mut *output, self, tables, (first, count, index)),
                        );
                    } else {
                        Narrow::<Width>::new().forward(
                            &mut *device,
                            (input, &mut *output, self, tables, (first, count, index)),
                        );
                    }
                }
            }
        }
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
        self.forward(&mut *device, (input, at, rows));
        self.forward(device, (branch, at, update));
    }
}
