use super::*;

impl<'s, 'a, Hidden, Heads, Head, G, P, All, Env, W, A, B, K, const N: usize>
    Module<
        &Period<Hidden, G, All>,
        Env,
        &'s mut lora::Adapters<'a, Term<A, B>, Selection<All, K>, N>,
    > for Affine<W, ()>
where
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
    G: Group<Period = P>,
    P: AxisName,
    All: AxisName,
    Env: Copy,
    K: AxisName,
    W: Matrix<In = Hidden, Out = Width<Heads, Head>>,
    Affine<A, ()>: for<'i, 'o> crate::Op<
            (
                Live<(), All>,
                &'i Normalized<Hidden, All>,
                &'o mut HbmTensor<bf16, Chip, m![P, K]>,
            ),
            Output = (),
        >,
    Affine<B, ()>: for<'r, 'o> crate::Op<
            (
                Live<(), P>,
                &'r HbmTensor<bf16, Chip, m![P, K]>,
                &'o mut Lines<P, Heads, Head>,
            ),
            Output = (),
        >,
{
    type Output = Queries<G, Heads, Head>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        adapters: &'s mut lora::Adapters<'a, Term<A, B>, Selection<All, K>, N>,
        rows: &Period<Hidden, G, All>,
    ) -> Self::Output {
        let lines = rows.lines(&mut *device, env, &self.weight);
        let lines = rows.adapt(&mut *device, adapters, lines);
        Period::<Hidden, G, All>::land::<Heads, Head, QuerySpread<G, Heads>, QueryLocal<G, Heads>>(
            device, lines,
        )
    }
}

impl<'s, 'a, Hidden, Heads, Head, G, P, All, Env, W, A, B, K, const N: usize>
    Module<
        &Period<Hidden, G, All>,
        Env,
        &'s mut lora::Adapters<'a, Term<A, B>, Selection<All, K>, N>,
    > for Affine<(W, ()), ()>
where
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
    G: Group<Period = P>,
    P: AxisName,
    All: AxisName,
    Env: Copy,
    K: AxisName,
    W: Matrix<In = Hidden, Out = Width<Heads, Head>>,
    Affine<A, ()>: for<'i, 'o> crate::Op<
            (
                Live<(), All>,
                &'i Normalized<Hidden, All>,
                &'o mut HbmTensor<bf16, Chip, m![P, K]>,
            ),
            Output = (),
        >,
    Affine<B, ()>: for<'r, 'o> crate::Op<
            (
                Live<(), P>,
                &'r HbmTensor<bf16, Chip, m![P, K]>,
                &'o mut Lines<P, Heads, Head>,
            ),
            Output = (),
        >,
{
    type Output = Pairs<G, Heads, Head>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        adapters: &'s mut lora::Adapters<'a, Term<A, B>, Selection<All, K>, N>,
        rows: &Period<Hidden, G, All>,
    ) -> Self::Output {
        let lines = rows.lines(&mut *device, env, &self.weight.0);
        let lines = rows.adapt(&mut *device, adapters, lines);
        Period::<Hidden, G, All>::land::<Heads, Head, G::Copies, PairLocal<Heads>>(device, lines)
    }
}
