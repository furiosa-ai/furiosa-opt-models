//! The tiling of packed rows in HBM a row a group of slices, and its gated residual pass.
//!
//! - A strip of rows puts each cluster's half of it one row a group of slices, each slice one
//!   contiguous run of the row's features in whole 256-byte bursts, so every transfer moves
//!   aligned runs and a row's statistic reduces over the slices of its group.
//! - Strips are 64 rows (`Wide`), or 16 (`Narrow`) for a span whose edge falls inside a 64-row
//!   strip, as a sequence split over chips has.

use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use super::tiling::{self, Tiling, row};
use crate::ops::index::packed::{Rows, Table};
use crate::ops::normalization::{Lanes, Squares};
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
pub(crate) struct Bands<R, L>(PhantomData<(R, L)>);

impl<R: AxisName, L: AxisName> AxisName for Bands<R, L> {
    const NAME: Ident = Ident::new("Bands");
    const SIZE: usize = {
        assert!(R::SIZE.is_multiple_of(L::SIZE));
        R::SIZE / L::SIZE
    };
}

/// A row's part, or a per-feature vector's part in every row's slices.
type Part<D, W, X, E, const ROWS: usize, const SLICES: usize> =
    tiling::Part<Lines<E, ROWS, SLICES>, W, X, D>;
type Operand<W, X, E, const ROWS: usize, const SLICES: usize> =
    tiling::Operand<Lines<E, ROWS, SLICES>, W, X>;
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
}

impl<W: AxisName, E: AxisName, const ROWS: usize, const SLICES: usize> Tiling<W>
    for Lines<E, ROWS, SLICES>
{
    type Tiles<R: AxisName> = Symbol<Bands<R, Self>>;
    type Rows = Symbol<Self>;
    type Half = m![{ Symbol<Self> } / { ROWS }];
    type Slices = m![{ Symbol<Self> } % { ROWS }, W / ({ Symbol<E> }) # { SLICES }];
    type Element = m![W % ({ Symbol<E> })];
    type Run = Symbol<E>;
    type Statistic = m![1 # 8];
    type Modulated = m![W % ({ Symbol<E> }) / 8];
    type Split = m![W % ({ Symbol<E> }) / 8, W % 8 / 4 % 2];

    fn inverse<const EPSILON: usize, X: M>(
        device: &mut Device,
        part: &Part<bf16, W, X, E, ROWS, SLICES>,
    ) -> tiling::RowOperand<Self, W, X> {
        Squares::<Quarters<W, E, ROWS, SLICES>, W, m![1], m![W / ({ Symbol<E> })], EPSILON>::inverse(
            device,
            part.view(),
            |square| square,
        )
    }
}

/// `bf16(x + gate * branch)` of a copied strip into `output`: the gated branch first, in the
/// vector registers, then added to the residual.
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

/// `bf16(x + gate * branch)` of `tiles` strips from strip `first` into `output`, the gate table row
/// `index`.
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
        let fill = |device: &mut Device, at: usize, (rows, update): &mut (_, _)| {
            Self::fill(&mut *device, input, at, rows);
            Self::fill(device, branch, at, update);
        };
        let gate = Self::operand(&mut *device, row(gate, index));
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
        fill(&mut *device, first, &mut current);
        fill(&mut *device, (first + 1).min(last), &mut next);
        for pair in 0..(tiles + 1) / 2 {
            let at = first + 2 * pair;
            self.forward(&mut *device, ((&current.0, &current.1), &gate, &mut kept));
            fill(&mut *device, (at + 2).min(last), &mut current);
            Self::spill(&mut *device, &kept, &mut *output, at);
            self.forward(&mut *device, ((&next.0, &next.1), &gate, &mut held));
            fill(&mut *device, (at + 3).min(last), &mut next);
            Self::spill(&mut *device, &held, &mut *output, (at + 1).min(last));
        }
    }
}
