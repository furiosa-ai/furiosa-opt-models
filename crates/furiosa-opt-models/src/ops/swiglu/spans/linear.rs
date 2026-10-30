//! The BF16 forms of the span MLP's weights: the packed gate/up and the down projections run on a
//! span's live rows a band at a time, through HBM, where their SwiGLU runs over rows.

use crate::Module;
use crate::modules::generic::schedule::Begin;
use crate::modules::lora;
use crate::ops::linear::Affine;
use crate::ops::linear::Linear;
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
impl<Hidden: AxisName, Intermediate: AxisName, Env> Module<Begin<()>, Env, &mut ()>
    for Linear<Hidden, GateUp<Intermediate>, Chip>
{
    type Output = ();

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) {}
}

/// The packed gate/up projection of a span's live rows, which move to HBM for it.
impl<'t, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName, Env: Copy>
    Module<((), Live<&'t Owned<Hidden, Rows>, Capacity>), Env, &mut ()>
    for Linear<Hidden, GateUp<Intermediate>, Chip>
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

/// The packed gate/up projection of a span's live rows plus each adapter's update of them.
impl<'s, 'a, 't, Hidden, Intermediate, Rows, Capacity, Env, A, B, K, const N: usize>
    Module<
        ((), Live<&'t Owned<Hidden, Rows>, Capacity>),
        Env,
        &'s mut lora::Adapters<'a, lora::Term<A, B>, HbmTensor<bf16, Chip, m![Capacity, K]>, N>,
    > for Linear<Hidden, GateUp<Intermediate>, Chip>
where
    Hidden: AxisName,
    Intermediate: AxisName,
    Rows: AxisName,
    Capacity: AxisName,
    Env: Copy,
    K: AxisName,
    Affine<A, ()>: for<'i, 'o> crate::Op<
            (
                Live<(), Rows>,
                &'i Grouped<Rows, Hidden>,
                &'o mut HbmTensor<bf16, Chip, m![Rows, K]>,
            ),
            Output = (),
        >,
    Affine<B, ()>: for<'r, 'o> crate::Op<
            (
                Live<(), Rows>,
                &'r HbmTensor<bf16, Chip, m![Rows, K]>,
                &'o mut Lines<Rows, GateUp<Intermediate>>,
            ),
            Output = (),
        >,
{
    type Output = Lines<Rows, GateUp<Intermediate>>;

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        adapters: &'s mut lora::Adapters<
            'a,
            lora::Term<A, B>,
            HbmTensor<bf16, Chip, m![Capacity, K]>,
            N,
        >,
        ((), normalized): ((), Live<&'t Owned<Hidden, Rows>, Capacity>),
    ) -> Lines<Rows, GateUp<Intermediate>> {
        let mut grouped = Grouped::<Rows, Hidden>::new();
        normalized
            .value()
            .view()
            .to_hbm_view(&mut context.tdma, grouped.view_mut());
        let live = Live::all().subrange(0, normalized.rows());
        let mut lines = Lines::new();
        self.affine
            .forward(&mut *context, env, &mut (), (live, &grouped, &mut lines));
        adapters.add(context, live, &grouped, normalized.start(), lines)
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
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName, Env>
    Module<Begin<Live<Lines<Rows, Intermediate>, Capacity>>, Env, &mut ()>
    for Linear<Intermediate, Hidden, Chip>
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

/// The dense down projection lands no weight panel ahead of its product: the contraction reads its
/// slabs from HBM.
impl<Hidden: AxisName, Intermediate: AxisName, Env> Module<Begin<usize>, Env, &mut ()>
    for Linear<Intermediate, Hidden, Chip>
{
    type Output = ();

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<usize>) {}
}

/// The product, already in HBM, waits there for the down projection, no panel landed for it.
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName, Env>
    Module<Begin<(Live<Lines<Rows, Intermediate>, Capacity>, ())>, Env, &mut ()>
    for Linear<Intermediate, Hidden, Chip>
{
    type Output = Lines<Rows, Intermediate>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        Begin((product, ())): Begin<(Live<Lines<Rows, Intermediate>, Capacity>, ())>,
    ) -> Lines<Rows, Intermediate> {
        product.into_value()
    }
}

/// Nor does any of its input land before it projects: the product is the down projection's input.
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName, Env>
    Module<Live<(Lines<Rows, Intermediate>, ((), ())), Capacity>, Env, &mut ()>
    for Linear<Intermediate, Hidden, Chip>
{
    type Output = Lines<Rows, Intermediate>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        staged: Live<(Lines<Rows, Intermediate>, ((), ())), Capacity>,
    ) -> Lines<Rows, Intermediate> {
        staged.into_value().0
    }
}

/// The down projection of a span's live product rows plus each adapter's update of them, landed
/// back on chip as the span's rows.
impl<'s, 'a, Hidden, Intermediate, Rows, Capacity, Env, A, B, K, const N: usize>
    Module<
        Live<Lines<Rows, Intermediate>, Capacity>,
        Env,
        &'s mut lora::Adapters<'a, lora::Term<A, B>, HbmTensor<bf16, Chip, m![Capacity, K]>, N>,
    > for Linear<Intermediate, Hidden, Chip>
where
    Hidden: AxisName,
    Intermediate: AxisName,
    Rows: AxisName,
    Capacity: AxisName,
    Env: Copy,
    K: AxisName,
    Affine<A, ()>: for<'i, 'o> crate::Op<
            (
                Live<(), Rows>,
                &'i Lines<Rows, Intermediate>,
                &'o mut HbmTensor<bf16, Chip, m![Rows, K]>,
            ),
            Output = (),
        >,
    Affine<B, ()>: for<'r, 'o> crate::Op<
            (
                Live<(), Rows>,
                &'r HbmTensor<bf16, Chip, m![Rows, K]>,
                &'o mut Lines<Rows, Hidden>,
            ),
            Output = (),
        >,
{
    type Output = Owned<Hidden, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        adapters: &'s mut lora::Adapters<
            'a,
            lora::Term<A, B>,
            HbmTensor<bf16, Chip, m![Capacity, K]>,
            N,
        >,
        product: Live<Lines<Rows, Intermediate>, Capacity>,
    ) -> Owned<Hidden, Rows> {
        let live = Live::all().subrange(0, product.rows());
        let mut projected = Lines::<Rows, Hidden>::new();
        self.affine.forward(
            &mut *context,
            env,
            &mut (),
            (live, product.value(), &mut projected),
        );
        let projected = adapters.add(context, live, product.value(), product.start(), projected);
        projected.to_dm(&mut context.tdma)
    }
}

/// The down projection of a span's live product rows, landed back on chip as the span's rows.
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName, Env: Copy>
    Module<Live<Lines<Rows, Intermediate>, Capacity>, Env, &mut ()>
    for Linear<Intermediate, Hidden, Chip>
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
