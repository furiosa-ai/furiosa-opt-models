//! The modulated norm and the gated residual passes of packed rows in HBM, a row a group of slices.
//!
//! - A strip of rows puts each cluster's half of it one row a group of slices, each slice one
//!   contiguous run of the row's features in whole 256-byte bursts, so every transfer moves
//!   aligned runs and a row's statistic reduces over the slices of its group.
//! - Strips are 64 rows (`Wide`), or 16 (`Narrow`) for a span whose edge falls inside a 64-row
//!   strip, as a sequence split over chips has.
//! - A span's per-feature operands sit in the vector registers of every slice that holds their
//!   features; the next strip arrives while the current one computes, and a strip's result is
//!   stored while the next one computes.

use core::borrow::Borrow;
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use crate::ops::index::packed::{Group, Rows, Table};
use crate::ops::normalization::{Lanes, Squares, Widen};
use crate::specialize::diffusion::{Run, Spread};

/// A strip of `2 * ROWS` rows: `ROWS` rows a cluster, `SLICES` slices a row, each `E` features
/// of the row.
#[derive(Clone, Debug)]
pub(crate) struct Lines<E, const ROWS: usize, const SLICES: usize>(PhantomData<E>);

/// 64-row strips: 32 rows a cluster, 8 slices a row, each `Run` features of the row.
pub(crate) type Wide<W> = Lines<Run<W>, 32, 8>;

/// 16-row strips: 8 rows a cluster, 32 slices a row, each `Spread` features of the row.
pub(crate) type Narrow<W> = Lines<Spread<W>, 8, 32>;

impl<E: AxisName, const ROWS: usize, const SLICES: usize> AxisName for Lines<E, ROWS, SLICES> {
    const NAME: Ident = Ident::new("Lines");
    const SIZE: usize = {
        assert!(
            ROWS * SLICES == 256,
            "the rows a cluster holds fill its 256 slices"
        );
        2 * ROWS
    };
}

/// The strips of a packed sequence.
#[derive(Clone, Debug)]
struct Bands<R, L>(PhantomData<(R, L)>);

impl<R: AxisName, L: AxisName> AxisName for Bands<R, L> {
    const NAME: Ident = Ident::new("Bands");
    const SIZE: usize = {
        assert!(R::SIZE.is_multiple_of(L::SIZE));
        R::SIZE / L::SIZE
    };
}

type Banded<R, L, W> = m![{ Symbol<Bands<R, L>> }, { Symbol<L> }, W];
type Half<E, const ROWS: usize, const SLICES: usize> =
    m![{ Symbol<Lines<E, ROWS, SLICES>> } / { ROWS }];
type Slices<W, E, const ROWS: usize, const SLICES: usize> =
    m![{ Symbol<Lines<E, ROWS, SLICES>> } % { ROWS }, W / ({ Symbol<E> }) # { SLICES }];
/// A row's part, or a per-feature vector's part in every row's slices.
type Part<D, W, X, E, const ROWS: usize, const SLICES: usize> =
    DmTensor<D, X, Half<E, ROWS, SLICES>, Slices<W, E, ROWS, SLICES>, m![W % ({ Symbol<E> })]>;
type Operand<W, X, E, const ROWS: usize, const SLICES: usize> =
    VrfTensor<f32, X, Half<E, ROWS, SLICES>, Slices<W, E, ROWS, SLICES>, m![W % ({ Symbol<E> })]>;
type RowOperand<W, X, E, const ROWS: usize, const SLICES: usize> =
    VrfTensor<f32, X, Half<E, ROWS, SLICES>, Slices<W, E, ROWS, SLICES>, m![1 # 8]>;
/// How a slice's run of a row's features moves through a norm's statistic pass.
type Quarters<W, E, const ROWS: usize, const SLICES: usize> = Lanes<
    m![W % ({ Symbol<E> }) / 8],
    m![W % 8],
    m![W % ({ Symbol<E> }) / 8, W % 8 / 4 % 2],
    m![W % 4],
    m![W % ({ Symbol<E> }) / 8],
    m![W % 8],
    m![W % 8],
    m![{ Symbol<Lines<E, ROWS, SLICES>> } % { ROWS }, { Broadcast<SLICES> }],
    m![1 # 8],
>;

impl<E, const ROWS: usize, const SLICES: usize> Lines<E, ROWS, SLICES> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }

    /// Table row `index`.
    fn table_row<W: AxisName, X: M>(
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

impl<R: AxisName, W: AxisName, X: M, E: AxisName, const ROWS: usize, const SLICES: usize>
    crate::Op<(
        &Rows<R, W, X>,
        usize,
        &mut Part<bf16, W, X, E, ROWS, SLICES>,
    )> for Lines<E, ROWS, SLICES>
{
    type Output = ();

    /// Tile `at` of `rows` into `part`.
    fn forward(
        &self,
        device: &mut Device,
        (rows, at, part): (
            &Rows<R, W, X>,
            usize,
            &mut Part<bf16, W, X, E, ROWS, SLICES>,
        ),
    ) {
        // SAFETY: the tiles split the row axis in row order; every cell keeps its address.
        unsafe { rows.view().reshape::<X, Banded<R, Self, W>>() }
            .tile::<m![{ Symbol<Bands<R, Self>> }], 1, m![
                { Symbol<Bands<R, Self>> } = 1 # ({ Symbol<Bands<R, Self>> }),
                { Symbol<Self> },
                W
            ]>(at)
            .to_dm_view(&mut device.tdma, part.view_mut());
    }
}
impl<R: AxisName, W: AxisName, X: M, E: AxisName, const ROWS: usize, const SLICES: usize>
    crate::Op<(
        &Part<bf16, W, X, E, ROWS, SLICES>,
        &mut Rows<R, W, X>,
        usize,
    )> for Lines<E, ROWS, SLICES>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (part, rows, at): (
            &Part<bf16, W, X, E, ROWS, SLICES>,
            &mut Rows<R, W, X>,
            usize,
        ),
    ) {
        part.view().to_hbm_view(
            &mut device.tdma,
            // SAFETY: the tiles split the row axis in row order; every cell keeps its address.
            unsafe { rows.view_mut().reshape::<X, Banded<R, Self, W>>() }.tile::<m![
                { Symbol<Bands<R, Self>> }
            ], 1, m![
                { Symbol<Bands<R, Self>> } = 1 #{!} ({ Symbol<Bands<R, Self>> }),
                { Symbol<Self> },
                W
            ]>(at),
        );
    }
}
impl<W: AxisName, X: M, E: AxisName, const ROWS: usize, const SLICES: usize>
    crate::Op<HbmTensorView<'_, bf16, X, m![W]>> for Lines<E, ROWS, SLICES>
{
    type Output = Operand<W, X, E, ROWS, SLICES>;

    fn forward(
        &self,
        device: &mut Device,
        vector: HbmTensorView<'_, bf16, X, m![W]>,
    ) -> Self::Output {
        let in_dm: Part<bf16, W, X, E, ROWS, SLICES> = vector.to_dm(&mut device.tdma);
        in_dm.widen::<W, 8>(&mut device.sub)
    }
}
impl<
    W: AxisName,
    X: M,
    E: AxisName,
    const ROWS: usize,
    const SLICES: usize,
    const EPSILON: usize,
    G,
>
    crate::Op<(
        &Part<bf16, W, X, E, ROWS, SLICES>,
        &super::AdaptiveRms<W, EPSILON, X, G>,
    )> for Lines<E, ROWS, SLICES>
{
    type Output = RowOperand<W, X, E, ROWS, SLICES>;

    fn forward(
        &self,
        device: &mut Device,
        (part, _): (
            &Part<bf16, W, X, E, ROWS, SLICES>,
            &super::AdaptiveRms<W, EPSILON, X, G>,
        ),
    ) -> Self::Output {
        Squares::<Quarters<W, E, ROWS, SLICES>, W, m![1], m![W / ({ Symbol<E> })], EPSILON>::inverse(
            device,
            part.view(),
            |square| square,
        )
    }
}
impl<W: AxisName, X: M, E: AxisName, const ROWS: usize, const SLICES: usize>
    crate::Op<(
        &Part<bf16, W, X, E, ROWS, SLICES>,
        (
            &RowOperand<W, X, E, ROWS, SLICES>,
            &Operand<W, X, E, ROWS, SLICES>,
            &Operand<W, X, E, ROWS, SLICES>,
        ),
        &mut Part<bf16, W, X, E, ROWS, SLICES>,
    )> for Lines<E, ROWS, SLICES>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (part, (inverse, gain, shift), output): (
            &Part<bf16, W, X, E, ROWS, SLICES>,
            (
                &RowOperand<W, X, E, ROWS, SLICES>,
                &Operand<W, X, E, ROWS, SLICES>,
                &Operand<W, X, E, ROWS, SLICES>,
            ),
            &mut Part<bf16, W, X, E, ROWS, SLICES>,
        ),
    ) {
        device
            .main
            .begin(part.view())
            .fetch::<m![W % ({ Symbol<E> }) / 8], m![W % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![W % ({ Symbol<E> }) / 8], m![W % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![W % ({ Symbol<E> }) / 8, W % 8 / 4 % 2], m![W % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), inverse)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), gain)
            .vector_fp_binary(FpBinaryOp::AddF, shift)
            .vector_widen_concat::<m![W % ({ Symbol<E> }) / 8], m![W % 8]>()
            .vector_final()
            .cast::<bf16, m![W % 8 # 16]>()
            .commit_trim::<m![W % 8]>()
            .commit_view(output.view_mut());
    }
}

impl<
    R: AxisName,
    W: AxisName,
    X: M,
    E: AxisName,
    const ROWS: usize,
    const SLICES: usize,
    const EPSILON: usize,
    G,
>
    crate::Op<(
        &Rows<R, W, X>,
        &mut Rows<R, W, X>,
        &super::AdaptiveRms<W, EPSILON, X, G>,
        &(Table<W, X>, Table<W, X>),
        (usize, usize, usize),
    )> for Lines<E, ROWS, SLICES>
where
    G: Borrow<HbmTensor<bf16, X, m![W]>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, output, norm, (shift, scale), (first, tiles, index)): (
            &Rows<R, W, X>,
            &mut Rows<R, W, X>,
            &super::AdaptiveRms<W, EPSILON, X, G>,
            &(Table<W, X>, Table<W, X>),
            (usize, usize, usize),
        ),
    ) {
        let gain = {
            let (weight, scale) = (norm.weight.tensor().view(), Self::table_row(scale, index));
            let weight = self.forward(&mut *device, weight);
            let scale: Part<bf16, W, X, E, ROWS, SLICES> = scale.to_dm(&mut device.tdma);
            device
                .main
                .begin(scale.view())
                .fetch::<m![W % ({ Symbol<E> }) / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % ({ Symbol<E> }) / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W % ({ Symbol<E> }) / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight)
                .vector_widen_concat::<m![W % ({ Symbol<E> }) / 8], m![W % 8]>()
                .vector_final()
                .to_vrf(&mut device.sub)
        };
        let shift = self.forward(&mut *device, Self::table_row(shift, index));
        let last = first + tiles - 1;
        let (mut current, mut next): (
            Part<bf16, W, X, E, ROWS, SLICES>,
            Part<bf16, W, X, E, ROWS, SLICES>,
        ) = (DmTensor::new(), DmTensor::new());
        // Two results, so that a tile computes while the tile before it is stored.
        let (mut kept, mut held): (
            Part<bf16, W, X, E, ROWS, SLICES>,
            Part<bf16, W, X, E, ROWS, SLICES>,
        ) = (DmTensor::new(), DmTensor::new());
        self.forward(&mut *device, (input, first, &mut current));
        self.forward(&mut *device, (input, (first + 1).min(last), &mut next));
        // An odd span's last pair normalizes its last tile twice.
        for pair in 0..(tiles + 1) / 2 {
            let at = first + 2 * pair;
            // Both statistics first, so Main computes the second while Sub inverts the first.
            let inverse = self.forward(&mut *device, (&current, norm));
            let following = self.forward(&mut *device, (&next, norm));
            self.forward(
                &mut *device,
                (&current, (&inverse, &gain, &shift), &mut kept),
            );
            self.forward(&mut *device, (input, (at + 2).min(last), &mut current));
            self.forward(&mut *device, (&kept, &mut *output, at));
            self.forward(
                &mut *device,
                (&next, (&following, &gain, &shift), &mut held),
            );
            self.forward(&mut *device, (input, (at + 3).min(last), &mut next));
            self.forward(&mut *device, (&held, &mut *output, (at + 1).min(last)));
        }
    }
}

impl<W: AxisName, X: M, E: AxisName, const ROWS: usize, const SLICES: usize>
    crate::Op<(
        (
            &Part<bf16, W, X, E, ROWS, SLICES>,
            &Part<bf16, W, X, E, ROWS, SLICES>,
        ),
        &Operand<W, X, E, ROWS, SLICES>,
        &mut Part<bf16, W, X, E, ROWS, SLICES>,
    )> for Lines<E, ROWS, SLICES>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        ((input, branch), gate, output): (
            (
                &Part<bf16, W, X, E, ROWS, SLICES>,
                &Part<bf16, W, X, E, ROWS, SLICES>,
            ),
            &Operand<W, X, E, ROWS, SLICES>,
            &mut Part<bf16, W, X, E, ROWS, SLICES>,
        ),
    ) {
        let scaled: Operand<W, X, E, ROWS, SLICES> = device
            .main
            .begin(branch.view())
            .fetch::<m![W % ({ Symbol<E> }) / 8], m![W % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![W % ({ Symbol<E> }) / 8], m![W % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![W % ({ Symbol<E> }) / 8, W % 8 / 4 % 2], m![W % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gate)
            .vector_widen_concat::<m![W % ({ Symbol<E> }) / 8], m![W % 8]>()
            .vector_final()
            .to_vrf(&mut device.sub);
        device
            .main
            .begin(input.view())
            .fetch::<m![W % ({ Symbol<E> }) / 8], m![W % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![W % ({ Symbol<E> }) / 8], m![W % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &scaled)
            .vector_final()
            .cast::<bf16, m![W % 8 # 16]>()
            .commit_trim::<m![W % 8]>()
            .commit_view(output.view_mut());
    }
}

impl<R: AxisName, W: AxisName, X: M, E: AxisName, const ROWS: usize, const SLICES: usize>
    crate::Op<(
        &Rows<R, W, X>,
        &Rows<R, W, X>,
        &mut Rows<R, W, X>,
        &Table<W, X>,
        (usize, usize, usize),
    )> for Lines<E, ROWS, SLICES>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, branch, output, gate, (first, tiles, index)): (
            &Rows<R, W, X>,
            &Rows<R, W, X>,
            &mut Rows<R, W, X>,
            &Table<W, X>,
            (usize, usize, usize),
        ),
    ) {
        let gate = self.forward(&mut *device, Self::table_row(gate, index));
        let last = first + tiles - 1;
        let mut current: (
            Part<bf16, W, X, E, ROWS, SLICES>,
            Part<bf16, W, X, E, ROWS, SLICES>,
        ) = (DmTensor::new(), DmTensor::new());
        let mut next: (
            Part<bf16, W, X, E, ROWS, SLICES>,
            Part<bf16, W, X, E, ROWS, SLICES>,
        ) = (DmTensor::new(), DmTensor::new());
        let (mut kept, mut held): (
            Part<bf16, W, X, E, ROWS, SLICES>,
            Part<bf16, W, X, E, ROWS, SLICES>,
        ) = (DmTensor::new(), DmTensor::new());
        self.forward(&mut *device, ((input, branch), first, &mut current));
        self.forward(
            &mut *device,
            ((input, branch), (first + 1).min(last), &mut next),
        );
        for pair in 0..(tiles + 1) / 2 {
            let at = first + 2 * pair;
            self.forward(&mut *device, ((&current.0, &current.1), &gate, &mut kept));
            self.forward(
                &mut *device,
                ((input, branch), (at + 2).min(last), &mut current),
            );
            self.forward(&mut *device, (&kept, &mut *output, at));
            self.forward(&mut *device, ((&next.0, &next.1), &gate, &mut held));
            self.forward(
                &mut *device,
                ((input, branch), (at + 3).min(last), &mut next),
            );
            self.forward(&mut *device, (&held, &mut *output, (at + 1).min(last)));
        }
    }
}

impl<R: AxisName, W: AxisName, X: M, E: AxisName, const ROWS: usize, const SLICES: usize>
    crate::Op<(
        (&Rows<R, W, X>, &Rows<R, W, X>),
        usize,
        &mut (
            Part<bf16, W, X, E, ROWS, SLICES>,
            Part<bf16, W, X, E, ROWS, SLICES>,
        ),
    )> for Lines<E, ROWS, SLICES>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        ((input, branch), at, (rows, update)): (
            (&Rows<R, W, X>, &Rows<R, W, X>),
            usize,
            &mut (
                Part<bf16, W, X, E, ROWS, SLICES>,
                Part<bf16, W, X, E, ROWS, SLICES>,
            ),
        ),
    ) {
        self.forward(&mut *device, (input, at, rows));
        self.forward(device, (branch, at, update));
    }
}
