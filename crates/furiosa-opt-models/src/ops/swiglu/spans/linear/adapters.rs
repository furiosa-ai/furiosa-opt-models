use super::*;

impl<'s, 'a, 't, Hidden, Intermediate, Rows, Capacity, T, Env, A, B, K, const N: usize>
    Module<
        ((), Live<&'t Owned<Hidden, Rows>, Capacity>),
        Env,
        &'s mut lora::Adapters<'a, lora::Term<A, B>, HbmTensor<bf16, Chip, m![Capacity, K]>, N>,
    > for Linear<Hidden, GateUp<Intermediate>, Chip, T>
where
    Hidden: AxisName,
    Intermediate: AxisName,
    Rows: AxisName,
    Capacity: AxisName,
    Env: Copy,
    K: AxisName,
    T: Matrix<In = Hidden, Out = GateUp<Intermediate>>,
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

impl<'s, 'a, Hidden, Intermediate, Rows, Capacity, T, Env, A, B, K, const N: usize>
    Module<
        Live<Lines<Rows, Intermediate>, Capacity>,
        Env,
        &'s mut lora::Adapters<'a, lora::Term<A, B>, HbmTensor<bf16, Chip, m![Capacity, K]>, N>,
    > for Linear<Intermediate, Hidden, Chip, T>
where
    Hidden: AxisName,
    Intermediate: AxisName,
    Rows: AxisName,
    Capacity: AxisName,
    Env: Copy,
    K: AxisName,
    T: Matrix<In = Intermediate, Out = Hidden>,
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
