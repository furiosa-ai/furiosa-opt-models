//! The dense forms of the span MLP's weights, BF16 coefficients or E4M3 codes the contraction
//! restores by their rows' scales: the packed gate/up and the down projections run on a span's live
//! rows a band at a time, through HBM, where their SwiGLU runs over rows.

use crate::Module;
use crate::modules::generic::schedule::Begin;
use crate::ops::linear::{Linear, Matrix};
use crate::ops::normalization::span::Owned;
use crate::ops::pointwise::SwiGlu;
use crate::ops::swiglu::GateUp;
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;

/// A span's rows in HBM, row by row, as the SwiGLU over rows and the projections write them.
type Lines<Rows, Width> = HbmTensor<bf16, Chip, m![Rows, Width]>;

/// A span's rows in HBM as its slices hold them, sixteen features a block, in groups of eight rows.
type Grouped<Rows, Width> = HbmTensor<bf16, Chip, m![Rows / 8, Width / 16, Rows % 8, Width % 16]>;

/// The packed gate/up weights need nothing landed before a span's rows arrive.
impl<Hidden: AxisName, Intermediate: AxisName, T, Env> Module<Begin<()>, Env, &mut ()>
    for Linear<Hidden, GateUp<Intermediate>, Chip, T>
where
    T: Matrix<In = Hidden, Out = GateUp<Intermediate>>,
{
    type Output = ();

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) {}
}

/// The packed gate/up projection of a span's live rows, which move to HBM for it.
impl<'t, Hidden, Intermediate, Rows, Capacity, T, Env: Copy>
    Module<((), Live<&'t Owned<Hidden, Rows>, Capacity>), Env, &mut ()>
    for Linear<Hidden, GateUp<Intermediate>, Chip, T>
where
    Hidden: AxisName,
    Intermediate: AxisName,
    Rows: AxisName,
    Capacity: AxisName,
    T: Matrix<In = Hidden, Out = GateUp<Intermediate>>,
{
    type Output = Lines<Rows, GateUp<Intermediate>>;

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        ((), normalized): ((), Live<&'t Owned<Hidden, Rows>, Capacity>),
    ) -> Lines<Rows, GateUp<Intermediate>> {
        let mut grouped = Grouped::<Rows, Hidden>::new();
        normalized
            .value()
            .view()
            .to_hbm_view(&mut context.tdma, grouped.view_mut());
        let mut lines = Lines::new();
        self.affine.forward(
            &mut *context,
            env,
            &mut (),
            (
                Live::all().subrange(0, normalized.rows()),
                &grouped,
                &mut lines,
            ),
        );
        lines
    }
}

/// `bf16(bf16(silu(gate)) * up)` of a span's gate/up rows in HBM.
impl<Intermediate: AxisName, Rows: AxisName, Capacity: AxisName, Env: Copy>
    Module<Live<Lines<Rows, GateUp<Intermediate>>, Capacity>, Env, &mut ()> for SwiGlu
{
    type Output = Lines<Rows, Intermediate>;

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        projected: Live<Lines<Rows, GateUp<Intermediate>>, Capacity>,
    ) -> Self::Output {
        self.forward(context, env, &mut (), projected.value())
    }
}

/// The product, already in HBM, waits there for the down projection.
impl<Hidden, Intermediate, Rows, Capacity, T, Env>
    Module<Begin<Live<Lines<Rows, Intermediate>, Capacity>>, Env, &mut ()>
    for Linear<Intermediate, Hidden, Chip, T>
where
    Hidden: AxisName,
    Intermediate: AxisName,
    Rows: AxisName,
    Capacity: AxisName,
    T: Matrix<In = Intermediate, Out = Hidden>,
{
    type Output = Lines<Rows, Intermediate>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        Begin(product): Begin<Live<Lines<Rows, Intermediate>, Capacity>>,
    ) -> Lines<Rows, Intermediate> {
        product.into_value()
    }
}

/// The down projection of a span's live product rows, landed back on chip as the span's rows.
impl<Hidden, Intermediate, Rows, Capacity, T, Env: Copy>
    Module<Live<Lines<Rows, Intermediate>, Capacity>, Env, &mut ()>
    for Linear<Intermediate, Hidden, Chip, T>
where
    Hidden: AxisName,
    Intermediate: AxisName,
    Rows: AxisName,
    Capacity: AxisName,
    T: Matrix<In = Intermediate, Out = Hidden>,
{
    type Output = Owned<Hidden, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        product: Live<Lines<Rows, Intermediate>, Capacity>,
    ) -> Owned<Hidden, Rows> {
        let mut projected = Lines::<Rows, Hidden>::new();
        self.affine.forward(
            &mut *context,
            env,
            &mut (),
            (
                Live::all().subrange(0, product.rows()),
                product.value(),
                &mut projected,
            ),
        );
        projected.to_dm(&mut context.tdma)
    }
}
