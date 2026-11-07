//! The modulated norm and the gated residual of packed rows in HBM, a row a group of slices.
//!
//! - A strip of rows puts each cluster's half of it one row a group of slices, each slice one
//!   contiguous run of the row's features in whole 256-byte bursts, so every transfer moves
//!   aligned runs and a row's statistic reduces over the slices of its group.
//! - Strips are 64 rows (`wide`), or 16 (`narrow`) for a span whose edge falls inside a 64-row
//!   strip, as a sequence split over chips has.
//! - A span's per-feature operands sit in the vector registers of every slice that holds their
//!   features; the next strip lands while the current one computes, and a strip's result is
//!   stored while the next one computes.

use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

/// Features of a row one slice holds: whole 256-byte bursts over at most eight slices, and at
/// most six, so a span's two FP32 operands (6 KiB) and a pass's register cache fit 8 KiB.
#[derive(Clone, Debug)]
pub struct Run<W>(PhantomData<W>);

impl<W: AxisName> AxisName for Run<W> {
    const NAME: Ident = Ident::new("Run");
    const SIZE: usize = {
        assert!(W::SIZE.is_multiple_of(128), "rows come in 256-byte bursts");
        let bursts = W::SIZE / 128;
        let mut each = 6;
        while bursts % each != 0 || bursts / each > 8 {
            each -= 1;
            assert!(each > 0, "no run fits the slices of a row");
        }
        128 * each
    };
}

/// Features of a row one slice of a 16-row strip holds: a row spreads over up to 32 slices.
#[derive(Clone, Debug)]
pub struct Spread<W>(PhantomData<W>);

impl<W: AxisName> AxisName for Spread<W> {
    const NAME: Ident = Ident::new("Spread");
    const SIZE: usize = {
        assert!(W::SIZE.is_multiple_of(128), "rows come in 256-byte bursts");
        let bursts = W::SIZE / 128;
        let mut each = 1;
        while bursts % each != 0 || bursts / each > 32 {
            each += 1;
            assert!(each <= 6, "no run fits the slices of a row");
        }
        128 * each
    };
}

/// 64-row strips: 32 rows a cluster, 8 slices a row, each `Run` features of the row.
pub(super) mod wide {
    use core::marker::PhantomData;

    use furiosa_opt_std::prelude::*;

    use super::super::{Group, Rows, Table};
    use super::Run;
    use crate::ops::vector::MeanSquare;

    axes![Lines = 64];

    /// The strips of a packed sequence.
    #[derive(Clone, Debug)]
    pub struct Bands<R>(PhantomData<R>);

    impl<R: AxisName> AxisName for Bands<R> {
        const NAME: Ident = Ident::new("Bands");
        const SIZE: usize = {
            assert!(R::SIZE.is_multiple_of(Lines::SIZE));
            R::SIZE / Lines::SIZE
        };
    }

    type Banded<R, W> = m![{ Symbol<Bands<R>> }, Lines, W];
    type E<W> = Symbol<Run<W>>;
    type Half = m![Lines / 32];
    type Slices<W> = m![Lines % 32, W / ({ E<W> }) # 8];
    /// A row's part, or a per-feature vector's part in every row's slices.
    type Part<D, W, X> = DmTensor<D, X, Half, Slices<W>, m![W % ({ E<W> })]>;
    type Operand<W, X> = VrfTensor<f32, X, Half, Slices<W>, m![W % ({ E<W> })]>;
    type RowOperand<W, X> = VrfTensor<f32, X, Half, Slices<W>, m![1 # 8]>;

    impl<R: AxisName, W: AxisName, X: M>
        crate::Module<(&mut Device, (&Rows<R, W, X>, usize, &mut Part<bf16, W, X>))> for Lines
    {
        type Output = ();

        /// Tile `at` of `rows` into `part`.
        fn forward(
            &self,
            (device, (rows, at, part)): (
                &mut Device,
                (&Rows<R, W, X>, usize, &mut Part<bf16, W, X>),
            ),
        ) {
            // SAFETY: the tiles split the row axis in row order; every cell keeps its address.
            unsafe { rows.view().reshape::<X, Banded<R, W>>() }
                .tile::<m![{ Symbol<Bands<R>> }], 1, m![
                    { Symbol<Bands<R>> } = 1 # ({ Symbol<Bands<R>> }),
                    Lines,
                    W
                ]>(at)
                .to_dm_view(&mut device.tdma, part.view_mut());
        }
    }
    impl<R: AxisName, W: AxisName, X: M>
        crate::Module<(&mut Device, (&Part<bf16, W, X>, &mut Rows<R, W, X>, usize))> for Lines
    {
        type Output = ();

        fn forward(
            &self,
            (device, (part, rows, at)): (
                &mut Device,
                (&Part<bf16, W, X>, &mut Rows<R, W, X>, usize),
            ),
        ) {
            part.view().to_hbm_view(
                &mut device.tdma,
                // SAFETY: as for `land`.
                unsafe { rows.view_mut().reshape::<X, Banded<R, W>>() }.tile::<m![
                    { Symbol<Bands<R>> }
                ], 1, m![
                    { Symbol<Bands<R>> } = 1 #{!} ({ Symbol<Bands<R>> }),
                    Lines,
                    W
                ]>(at),
            );
        }
    }
    impl<W: AxisName, X: M> crate::Module<(&mut Device, HbmTensorView<'_, bf16, X, m![W]>)> for Lines {
        type Output = Operand<W, X>;

        fn forward(
            &self,
            (device, vector): (&mut Device, HbmTensorView<'_, bf16, X, m![W]>),
        ) -> Self::Output {
            let landed: Part<bf16, W, X> = vector.to_dm(&mut device.tdma);
            device
                .sub
                .begin(landed.view())
                .fetch::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .to_vrf()
        }
    }
    impl Lines {
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
    impl<W: AxisName, X: M, const EPSILON: usize>
        crate::Module<(
            &mut Device,
            (&Part<bf16, W, X>, &super::super::AdaptiveRms<W, EPSILON, X>),
        )> for Lines
    {
        type Output = RowOperand<W, X>;

        fn forward(
            &self,
            (device, (part, _)): (
                &mut Device,
                (&Part<bf16, W, X>, &super::super::AdaptiveRms<W, EPSILON, X>),
            ),
        ) -> Self::Output {
            let variance: DmTensor<f32, X, Half, m![Lines % 32, 8], m![1 # 8]> = device
                .main
                .begin(part.view())
                .fetch::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W % ({ E<W> }) / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_mean_square::<W, m![1], m![W / ({ E<W> })], EPSILON>()
                .vector_inter_slice_reduce::<m![Lines % 32, 8], m![1]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            // SAFETY: every slice of a group holds its row's statistic; each row keeps its address.
            let variance: DmTensor<f32, X, Half, Slices<W>, m![1 # 8]> =
                unsafe { variance.reshape() };
            device
                .sub
                .begin(variance.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_unary(FpUnaryOp::Sqrt)
                .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .to_vrf()
        }
    }
    impl<W: AxisName, X: M>
        crate::Module<(
            &mut Device,
            (
                &Part<bf16, W, X>,
                (&RowOperand<W, X>, &Operand<W, X>, &Operand<W, X>),
                &mut Part<bf16, W, X>,
            ),
        )> for Lines
    {
        type Output = ();

        fn forward(
            &self,
            (device, (part, (inverse, gain, shift), output)): (
                &mut Device,
                (
                    &Part<bf16, W, X>,
                    (&RowOperand<W, X>, &Operand<W, X>, &Operand<W, X>),
                    &mut Part<bf16, W, X>,
                ),
            ),
        ) {
            device
                .main
                .begin(part.view())
                .fetch::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W % ({ E<W> }) / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), inverse)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), gain)
                .vector_fp_binary(FpBinaryOp::AddF, shift)
                .vector_widen_concat::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .vector_final()
                .cast::<bf16, m![W % 8 # 16]>()
                .commit_trim::<m![W % 8]>()
                .commit_view(output.view_mut());
        }
    }

    impl<R: AxisName, W: AxisName, X: M, const EPSILON: usize>
        crate::Module<(
            &mut Device,
            (
                &Rows<R, W, X>,
                &mut Rows<R, W, X>,
                &super::super::AdaptiveRms<W, EPSILON, X>,
                &(Table<W, X>, Table<W, X>),
                (usize, usize, usize),
            ),
        )> for Lines
    {
        type Output = ();

        fn forward(
            &self,
            (device, (input, output, norm, (shift, scale), (first, tiles, index))): (
                &mut Device,
                (
                    &Rows<R, W, X>,
                    &mut Rows<R, W, X>,
                    &super::super::AdaptiveRms<W, EPSILON, X>,
                    &(Table<W, X>, Table<W, X>),
                    (usize, usize, usize),
                ),
            ),
        ) {
            let gain = {
                let (weight, scale) = (norm.weight.tensor().view(), Self::table_row(scale, index));
                let weight = crate::Module::forward(&Lines, (&mut *device, weight));
                let scale: Part<bf16, W, X> = scale.to_dm(&mut device.tdma);
                device
                    .main
                    .begin(scale.view())
                    .fetch::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                    .fetch_cast::<f32>()
                    .collect::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![W % ({ E<W> }) / 8, W % 8 / 4 % 2], m![W % 4]>()
                    .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight)
                    .vector_widen_concat::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                    .vector_final()
                    .to_vrf(&mut device.sub)
            };
            let shift =
                crate::Module::forward(&Lines, (&mut *device, Self::table_row(shift, index)));
            let last = first + tiles - 1;
            let (mut current, mut next): (Part<bf16, W, X>, Part<bf16, W, X>) =
                (DmTensor::new(), DmTensor::new());
            // Two results, so that a tile computes while the tile before it is stored.
            let (mut kept, mut held): (Part<bf16, W, X>, Part<bf16, W, X>) =
                (DmTensor::new(), DmTensor::new());
            crate::Module::forward(&Lines, (&mut *device, (input, first, &mut current)));
            crate::Module::forward(
                &Lines,
                (&mut *device, (input, (first + 1).min(last), &mut next)),
            );
            // An odd span's last pair normalizes its last tile twice.
            for pair in 0..(tiles + 1) / 2 {
                let at = first + 2 * pair;
                // Both statistics first, so Main computes the second while Sub inverts the first.
                let inverse = crate::Module::forward(&Lines, (&mut *device, (&current, norm)));
                let following = crate::Module::forward(&Lines, (&mut *device, (&next, norm)));
                crate::Module::forward(
                    &Lines,
                    (
                        &mut *device,
                        (&current, (&inverse, &gain, &shift), &mut kept),
                    ),
                );
                crate::Module::forward(
                    &Lines,
                    (&mut *device, (input, (at + 2).min(last), &mut current)),
                );
                crate::Module::forward(&Lines, (&mut *device, (&kept, &mut *output, at)));
                crate::Module::forward(
                    &Lines,
                    (
                        &mut *device,
                        (&next, (&following, &gain, &shift), &mut held),
                    ),
                );
                crate::Module::forward(
                    &Lines,
                    (&mut *device, (input, (at + 3).min(last), &mut next)),
                );
                crate::Module::forward(
                    &Lines,
                    (&mut *device, (&held, &mut *output, (at + 1).min(last))),
                );
            }
        }
    }

    impl<W: AxisName, X: M>
        crate::Module<(
            &mut Device,
            (
                (&Part<bf16, W, X>, &Part<bf16, W, X>),
                &Operand<W, X>,
                &mut Part<bf16, W, X>,
            ),
        )> for Lines
    {
        type Output = ();

        fn forward(
            &self,
            (device, ((input, branch), gate, output)): (
                &mut Device,
                (
                    (&Part<bf16, W, X>, &Part<bf16, W, X>),
                    &Operand<W, X>,
                    &mut Part<bf16, W, X>,
                ),
            ),
        ) {
            let scaled: Operand<W, X> = device
                .main
                .begin(branch.view())
                .fetch::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W % ({ E<W> }) / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gate)
                .vector_widen_concat::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .vector_final()
                .to_vrf(&mut device.sub);
            device
                .main
                .begin(input.view())
                .fetch::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, &scaled)
                .vector_final()
                .cast::<bf16, m![W % 8 # 16]>()
                .commit_trim::<m![W % 8]>()
                .commit_view(output.view_mut());
        }
    }

    impl<R: AxisName, W: AxisName, X: M>
        crate::Module<(
            &mut Device,
            (
                &Rows<R, W, X>,
                &Rows<R, W, X>,
                &mut Rows<R, W, X>,
                &Table<W, X>,
                (usize, usize, usize),
            ),
        )> for Lines
    {
        type Output = ();

        fn forward(
            &self,
            (device, (input, branch, output, gate, (first, tiles, index))): (
                &mut Device,
                (
                    &Rows<R, W, X>,
                    &Rows<R, W, X>,
                    &mut Rows<R, W, X>,
                    &Table<W, X>,
                    (usize, usize, usize),
                ),
            ),
        ) {
            let gate = crate::Module::forward(&Lines, (&mut *device, Self::table_row(gate, index)));
            let last = first + tiles - 1;
            let mut current: (Part<bf16, W, X>, Part<bf16, W, X>) =
                (DmTensor::new(), DmTensor::new());
            let mut next: (Part<bf16, W, X>, Part<bf16, W, X>) = (DmTensor::new(), DmTensor::new());
            let (mut kept, mut held): (Part<bf16, W, X>, Part<bf16, W, X>) =
                (DmTensor::new(), DmTensor::new());
            crate::Module::forward(
                &Lines,
                (&mut *device, ((input, branch), first, &mut current)),
            );
            crate::Module::forward(
                &Lines,
                (
                    &mut *device,
                    ((input, branch), (first + 1).min(last), &mut next),
                ),
            );
            for pair in 0..(tiles + 1) / 2 {
                let at = first + 2 * pair;
                crate::Module::forward(
                    &Lines,
                    (&mut *device, ((&current.0, &current.1), &gate, &mut kept)),
                );
                crate::Module::forward(
                    &Lines,
                    (
                        &mut *device,
                        ((input, branch), (at + 2).min(last), &mut current),
                    ),
                );
                crate::Module::forward(&Lines, (&mut *device, (&kept, &mut *output, at)));
                crate::Module::forward(
                    &Lines,
                    (&mut *device, ((&next.0, &next.1), &gate, &mut held)),
                );
                crate::Module::forward(
                    &Lines,
                    (
                        &mut *device,
                        ((input, branch), (at + 3).min(last), &mut next),
                    ),
                );
                crate::Module::forward(
                    &Lines,
                    (&mut *device, (&held, &mut *output, (at + 1).min(last))),
                );
            }
        }
    }

    impl<R: AxisName, W: AxisName, X: M>
        crate::Module<(
            &mut Device,
            (
                (&Rows<R, W, X>, &Rows<R, W, X>),
                usize,
                &mut (Part<bf16, W, X>, Part<bf16, W, X>),
            ),
        )> for Lines
    {
        type Output = ();

        fn forward(
            &self,
            (device, ((input, branch), at, (rows, update))): (
                &mut Device,
                (
                    (&Rows<R, W, X>, &Rows<R, W, X>),
                    usize,
                    &mut (Part<bf16, W, X>, Part<bf16, W, X>),
                ),
            ),
        ) {
            crate::Module::forward(self, (&mut *device, (input, at, rows)));
            crate::Module::forward(self, (device, (branch, at, update)));
        }
    }
}

/// 16-row strips: 8 rows a cluster, 32 slices a row, each `Spread` features of the row.
pub(super) mod narrow {
    use core::marker::PhantomData;

    use furiosa_opt_std::prelude::*;

    use super::super::{Group, Rows, Table};
    use super::Spread as Run;
    use crate::ops::vector::MeanSquare;

    axes![Lines = 16];

    /// The strips of a packed sequence.
    #[derive(Clone, Debug)]
    pub struct Bands<R>(PhantomData<R>);

    impl<R: AxisName> AxisName for Bands<R> {
        const NAME: Ident = Ident::new("Bands");
        const SIZE: usize = {
            assert!(R::SIZE.is_multiple_of(Lines::SIZE));
            R::SIZE / Lines::SIZE
        };
    }

    type Banded<R, W> = m![{ Symbol<Bands<R>> }, Lines, W];
    type E<W> = Symbol<Run<W>>;
    type Half = m![Lines / 8];
    type Slices<W> = m![Lines % 8, W / ({ E<W> }) # 32];
    /// A row's part, or a per-feature vector's part in every row's slices.
    type Part<D, W, X> = DmTensor<D, X, Half, Slices<W>, m![W % ({ E<W> })]>;
    type Operand<W, X> = VrfTensor<f32, X, Half, Slices<W>, m![W % ({ E<W> })]>;
    type RowOperand<W, X> = VrfTensor<f32, X, Half, Slices<W>, m![1 # 8]>;

    impl<R: AxisName, W: AxisName, X: M>
        crate::Module<(&mut Device, (&Rows<R, W, X>, usize, &mut Part<bf16, W, X>))> for Lines
    {
        type Output = ();

        /// Tile `at` of `rows` into `part`.
        fn forward(
            &self,
            (device, (rows, at, part)): (
                &mut Device,
                (&Rows<R, W, X>, usize, &mut Part<bf16, W, X>),
            ),
        ) {
            // SAFETY: the tiles split the row axis in row order; every cell keeps its address.
            unsafe { rows.view().reshape::<X, Banded<R, W>>() }
                .tile::<m![{ Symbol<Bands<R>> }], 1, m![
                    { Symbol<Bands<R>> } = 1 # ({ Symbol<Bands<R>> }),
                    Lines,
                    W
                ]>(at)
                .to_dm_view(&mut device.tdma, part.view_mut());
        }
    }
    impl<R: AxisName, W: AxisName, X: M>
        crate::Module<(&mut Device, (&Part<bf16, W, X>, &mut Rows<R, W, X>, usize))> for Lines
    {
        type Output = ();

        fn forward(
            &self,
            (device, (part, rows, at)): (
                &mut Device,
                (&Part<bf16, W, X>, &mut Rows<R, W, X>, usize),
            ),
        ) {
            part.view().to_hbm_view(
                &mut device.tdma,
                // SAFETY: as for `land`.
                unsafe { rows.view_mut().reshape::<X, Banded<R, W>>() }.tile::<m![
                    { Symbol<Bands<R>> }
                ], 1, m![
                    { Symbol<Bands<R>> } = 1 #{!} ({ Symbol<Bands<R>> }),
                    Lines,
                    W
                ]>(at),
            );
        }
    }
    impl<W: AxisName, X: M> crate::Module<(&mut Device, HbmTensorView<'_, bf16, X, m![W]>)> for Lines {
        type Output = Operand<W, X>;

        fn forward(
            &self,
            (device, vector): (&mut Device, HbmTensorView<'_, bf16, X, m![W]>),
        ) -> Self::Output {
            let landed: Part<bf16, W, X> = vector.to_dm(&mut device.tdma);
            device
                .sub
                .begin(landed.view())
                .fetch::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .to_vrf()
        }
    }
    impl Lines {
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
    impl<W: AxisName, X: M, const EPSILON: usize>
        crate::Module<(
            &mut Device,
            (&Part<bf16, W, X>, &super::super::AdaptiveRms<W, EPSILON, X>),
        )> for Lines
    {
        type Output = RowOperand<W, X>;

        fn forward(
            &self,
            (device, (part, _)): (
                &mut Device,
                (&Part<bf16, W, X>, &super::super::AdaptiveRms<W, EPSILON, X>),
            ),
        ) -> Self::Output {
            let variance: DmTensor<f32, X, Half, m![Lines % 8, 32], m![1 # 8]> = device
                .main
                .begin(part.view())
                .fetch::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W % ({ E<W> }) / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_mean_square::<W, m![1], m![W / ({ E<W> })], EPSILON>()
                .vector_inter_slice_reduce::<m![Lines % 8, 32], m![1]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            // SAFETY: every slice of a group holds its row's statistic; each row keeps its address.
            let variance: DmTensor<f32, X, Half, Slices<W>, m![1 # 8]> =
                unsafe { variance.reshape() };
            device
                .sub
                .begin(variance.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_unary(FpUnaryOp::Sqrt)
                .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .to_vrf()
        }
    }
    impl<W: AxisName, X: M>
        crate::Module<(
            &mut Device,
            (
                &Part<bf16, W, X>,
                (&RowOperand<W, X>, &Operand<W, X>, &Operand<W, X>),
                &mut Part<bf16, W, X>,
            ),
        )> for Lines
    {
        type Output = ();

        fn forward(
            &self,
            (device, (part, (inverse, gain, shift), output)): (
                &mut Device,
                (
                    &Part<bf16, W, X>,
                    (&RowOperand<W, X>, &Operand<W, X>, &Operand<W, X>),
                    &mut Part<bf16, W, X>,
                ),
            ),
        ) {
            device
                .main
                .begin(part.view())
                .fetch::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W % ({ E<W> }) / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), inverse)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), gain)
                .vector_fp_binary(FpBinaryOp::AddF, shift)
                .vector_widen_concat::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .vector_final()
                .cast::<bf16, m![W % 8 # 16]>()
                .commit_trim::<m![W % 8]>()
                .commit_view(output.view_mut());
        }
    }

    impl<R: AxisName, W: AxisName, X: M, const EPSILON: usize>
        crate::Module<(
            &mut Device,
            (
                &Rows<R, W, X>,
                &mut Rows<R, W, X>,
                &super::super::AdaptiveRms<W, EPSILON, X>,
                &(Table<W, X>, Table<W, X>),
                (usize, usize, usize),
            ),
        )> for Lines
    {
        type Output = ();

        fn forward(
            &self,
            (device, (input, output, norm, (shift, scale), (first, tiles, index))): (
                &mut Device,
                (
                    &Rows<R, W, X>,
                    &mut Rows<R, W, X>,
                    &super::super::AdaptiveRms<W, EPSILON, X>,
                    &(Table<W, X>, Table<W, X>),
                    (usize, usize, usize),
                ),
            ),
        ) {
            let gain = {
                let (weight, scale) = (norm.weight.tensor().view(), Self::table_row(scale, index));
                let weight = crate::Module::forward(&Lines, (&mut *device, weight));
                let scale: Part<bf16, W, X> = scale.to_dm(&mut device.tdma);
                device
                    .main
                    .begin(scale.view())
                    .fetch::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                    .fetch_cast::<f32>()
                    .collect::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![W % ({ E<W> }) / 8, W % 8 / 4 % 2], m![W % 4]>()
                    .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight)
                    .vector_widen_concat::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                    .vector_final()
                    .to_vrf(&mut device.sub)
            };
            let shift =
                crate::Module::forward(&Lines, (&mut *device, Self::table_row(shift, index)));
            let last = first + tiles - 1;
            let (mut current, mut next): (Part<bf16, W, X>, Part<bf16, W, X>) =
                (DmTensor::new(), DmTensor::new());
            // Two results, so that a tile computes while the tile before it is stored.
            let (mut kept, mut held): (Part<bf16, W, X>, Part<bf16, W, X>) =
                (DmTensor::new(), DmTensor::new());
            crate::Module::forward(&Lines, (&mut *device, (input, first, &mut current)));
            crate::Module::forward(
                &Lines,
                (&mut *device, (input, (first + 1).min(last), &mut next)),
            );
            // An odd span's last pair normalizes its last tile twice.
            for pair in 0..(tiles + 1) / 2 {
                let at = first + 2 * pair;
                // Both statistics first, so Main computes the second while Sub inverts the first.
                let inverse = crate::Module::forward(&Lines, (&mut *device, (&current, norm)));
                let following = crate::Module::forward(&Lines, (&mut *device, (&next, norm)));
                crate::Module::forward(
                    &Lines,
                    (
                        &mut *device,
                        (&current, (&inverse, &gain, &shift), &mut kept),
                    ),
                );
                crate::Module::forward(
                    &Lines,
                    (&mut *device, (input, (at + 2).min(last), &mut current)),
                );
                crate::Module::forward(&Lines, (&mut *device, (&kept, &mut *output, at)));
                crate::Module::forward(
                    &Lines,
                    (
                        &mut *device,
                        (&next, (&following, &gain, &shift), &mut held),
                    ),
                );
                crate::Module::forward(
                    &Lines,
                    (&mut *device, (input, (at + 3).min(last), &mut next)),
                );
                crate::Module::forward(
                    &Lines,
                    (&mut *device, (&held, &mut *output, (at + 1).min(last))),
                );
            }
        }
    }

    impl<W: AxisName, X: M>
        crate::Module<(
            &mut Device,
            (
                (&Part<bf16, W, X>, &Part<bf16, W, X>),
                &Operand<W, X>,
                &mut Part<bf16, W, X>,
            ),
        )> for Lines
    {
        type Output = ();

        fn forward(
            &self,
            (device, ((input, branch), gate, output)): (
                &mut Device,
                (
                    (&Part<bf16, W, X>, &Part<bf16, W, X>),
                    &Operand<W, X>,
                    &mut Part<bf16, W, X>,
                ),
            ),
        ) {
            let scaled: Operand<W, X> = device
                .main
                .begin(branch.view())
                .fetch::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![W % ({ E<W> }) / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gate)
                .vector_widen_concat::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .vector_final()
                .to_vrf(&mut device.sub);
            device
                .main
                .begin(input.view())
                .fetch::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![W % ({ E<W> }) / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, &scaled)
                .vector_final()
                .cast::<bf16, m![W % 8 # 16]>()
                .commit_trim::<m![W % 8]>()
                .commit_view(output.view_mut());
        }
    }

    impl<R: AxisName, W: AxisName, X: M>
        crate::Module<(
            &mut Device,
            (
                &Rows<R, W, X>,
                &Rows<R, W, X>,
                &mut Rows<R, W, X>,
                &Table<W, X>,
                (usize, usize, usize),
            ),
        )> for Lines
    {
        type Output = ();

        fn forward(
            &self,
            (device, (input, branch, output, gate, (first, tiles, index))): (
                &mut Device,
                (
                    &Rows<R, W, X>,
                    &Rows<R, W, X>,
                    &mut Rows<R, W, X>,
                    &Table<W, X>,
                    (usize, usize, usize),
                ),
            ),
        ) {
            let gate = crate::Module::forward(&Lines, (&mut *device, Self::table_row(gate, index)));
            let last = first + tiles - 1;
            let mut current: (Part<bf16, W, X>, Part<bf16, W, X>) =
                (DmTensor::new(), DmTensor::new());
            let mut next: (Part<bf16, W, X>, Part<bf16, W, X>) = (DmTensor::new(), DmTensor::new());
            let (mut kept, mut held): (Part<bf16, W, X>, Part<bf16, W, X>) =
                (DmTensor::new(), DmTensor::new());
            crate::Module::forward(
                &Lines,
                (&mut *device, ((input, branch), first, &mut current)),
            );
            crate::Module::forward(
                &Lines,
                (
                    &mut *device,
                    ((input, branch), (first + 1).min(last), &mut next),
                ),
            );
            for pair in 0..(tiles + 1) / 2 {
                let at = first + 2 * pair;
                crate::Module::forward(
                    &Lines,
                    (&mut *device, ((&current.0, &current.1), &gate, &mut kept)),
                );
                crate::Module::forward(
                    &Lines,
                    (
                        &mut *device,
                        ((input, branch), (at + 2).min(last), &mut current),
                    ),
                );
                crate::Module::forward(&Lines, (&mut *device, (&kept, &mut *output, at)));
                crate::Module::forward(
                    &Lines,
                    (&mut *device, ((&next.0, &next.1), &gate, &mut held)),
                );
                crate::Module::forward(
                    &Lines,
                    (
                        &mut *device,
                        ((input, branch), (at + 3).min(last), &mut next),
                    ),
                );
                crate::Module::forward(
                    &Lines,
                    (&mut *device, (&held, &mut *output, (at + 1).min(last))),
                );
            }
        }
    }

    impl<R: AxisName, W: AxisName, X: M>
        crate::Module<(
            &mut Device,
            (
                (&Rows<R, W, X>, &Rows<R, W, X>),
                usize,
                &mut (Part<bf16, W, X>, Part<bf16, W, X>),
            ),
        )> for Lines
    {
        type Output = ();

        fn forward(
            &self,
            (device, ((input, branch), at, (rows, update))): (
                &mut Device,
                (
                    (&Rows<R, W, X>, &Rows<R, W, X>),
                    usize,
                    &mut (Part<bf16, W, X>, Part<bf16, W, X>),
                ),
            ),
        ) {
            crate::Module::forward(self, (&mut *device, (input, at, rows)));
            crate::Module::forward(self, (device, (branch, at, update)));
        }
    }
}
