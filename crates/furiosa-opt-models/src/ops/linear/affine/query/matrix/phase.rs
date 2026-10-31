use super::*;

type Flat<'w, Hidden, Heads, Head> =
    HbmTensorView<'w, bf16, Chip, m![{ Symbol<Width<Heads, Head>> }, Hidden]>;

impl<'w, Hidden: AxisName, Heads: AxisName, Head: AxisName, Env> Module<Begin<()>, Env, &mut ()>
    for Affine<&'w Dense<Hidden, Heads, Head>, ()>
{
    type Output = Affine<Flat<'w, Hidden, Heads, Head>, ()>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) -> Self::Output {
        let weight: &'w Dense<Hidden, Heads, Head> = self.weight;
        // SAFETY: r=h*Head+d gives r*Hidden+i=((h*Head+d)*Hidden+i); Chip/BF16/extent stay
        // unchanged, with no padding/permutation. Consuming the read view retains weight's 'w borrow.
        Affine::new(unsafe { weight.view().reshape() }, ())
    }
}

impl<'w, Hidden: AxisName, Heads: AxisName, Head: AxisName, Env> Module<Begin<()>, Env, &mut ()>
    for Affine<(&'w Dense<Hidden, Heads, Head>, ()), ()>
{
    type Output = Affine<(Flat<'w, Hidden, Heads, Head>, ()), ()>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) -> Self::Output {
        let weight: &'w Dense<Hidden, Heads, Head> = self.weight.0;
        // SAFETY: r=h*Head+d gives r*Hidden+i=((h*Head+d)*Hidden+i); Chip/BF16/extent stay
        // unchanged, with no padding/permutation. Consuming the read view retains weight's 'w borrow.
        Affine::new((unsafe { weight.view().reshape() }, ()), ())
    }
}

impl<Hidden: AxisName, Heads: AxisName, Head: AxisName, C, S, Env> Module<Begin<()>, Env, &mut ()>
    for Weight<m![Heads, Head, Hidden], Chip, Coded<Hidden, Heads, Head, C, S>>
where
    Coded<Hidden, Heads, Head, C, S>: Clone,
{
    type Output = Affine<Coded<Hidden, Heads, Head, C, S>, ()>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) -> Self::Output {
        Affine::new(self.held().clone(), ())
    }
}

impl<Hidden: AxisName, Heads: AxisName, Head: AxisName, C, S, Env> Module<Begin<()>, Env, &mut ()>
    for (
        Weight<m![Heads, Head, Hidden], Chip, Coded<Hidden, Heads, Head, C, S>>,
        (),
    )
where
    Coded<Hidden, Heads, Head, C, S>: Clone,
{
    type Output = Affine<(Coded<Hidden, Heads, Head, C, S>, ()), ()>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: Begin<()>) -> Self::Output {
        Affine::new((self.0.held().clone(), ()), ())
    }
}

impl<Hidden: AxisName, All: AxisName, W: Matrix<In = Hidden>>
    Module<Begin<()>, Live<(), All>, &mut ()> for Affine<(W, ()), ()>
{
    type Output = Launch<Hidden, (), All>;

    fn forward(
        &self,
        _: &mut Device,
        live: Live<(), All>,
        _: &mut (),
        _: Begin<()>,
    ) -> Self::Output {
        Launch::new(live, ())
    }
}

impl<G, P, Hidden, All, W, Env> Module<Next<'_, G, Hidden, (), All>, Env, &mut ()>
    for Affine<(W, ()), ()>
where
    G: Group<Period = P>,
    P: AxisName,
    Hidden: AxisName,
    All: AxisName,
    W: Matrix<In = Hidden>,
{
    type Output = Period<Hidden, G, All>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        next: Next<'_, G, Hidden, (), All>,
    ) -> Self::Output {
        let Next { rows, live, at, .. } = next;
        Period {
            // SAFETY: the launch writes its rows only before its first period, and keeps them.
            rows: unsafe { rows.view().alias() },
            live: live.subrange(at * <TilePair as Group>::Period::SIZE, P::SIZE),
            group: PhantomData,
        }
    }
}

impl<Hidden, Heads, Head, G, P, All, W, Env> Module<&Period<Hidden, G, All>, Env, &mut ()>
    for Affine<W, ()>
where
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
    G: Group<Period = P>,
    P: AxisName,
    All: AxisName,
    W: Matrix<In = Hidden, Out = Width<Heads, Head>>,
    Env: Copy,
{
    type Output = Queries<G, Heads, Head>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        rows: &Period<Hidden, G, All>,
    ) -> Self::Output {
        let lines = rows.lines(&mut *device, env, &self.weight);
        Period::<Hidden, G, All>::land::<Heads, Head, QuerySpread<G, Heads>, QueryLocal<G, Heads>>(
            device, lines,
        )
    }
}

impl<Hidden, Heads, Head, G, P, All, W, Env> Module<&Period<Hidden, G, All>, Env, &mut ()>
    for Affine<(W, ()), ()>
where
    Hidden: AxisName,
    Heads: AxisName,
    Head: AxisName,
    G: Group<Period = P>,
    P: AxisName,
    All: AxisName,
    W: Matrix<In = Hidden, Out = Width<Heads, Head>>,
    Env: Copy,
{
    type Output = Pairs<G, Heads, Head>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        rows: &Period<Hidden, G, All>,
    ) -> Self::Output {
        let lines = rows.lines(&mut *device, env, &self.weight.0);
        Period::<Hidden, G, All>::land::<Heads, Head, G::Copies, PairLocal<Heads>>(device, lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![Hidden = 1024, Heads = 16, Head = 128];
    type View<'w> = Flat<'w, Hidden, Heads, Head>;

    /// Landing borrows the weight, not the temporary projection receiver, and supports tiling.
    #[test]
    fn borrows_dense() {
        let _: for<'w> fn(&'w Dense<Hidden, Heads, Head>, &mut Device) -> View<'w> =
            |weight, device| {
                let projection = Affine::new(weight, ());
                let landed: Affine<View<'_>, ()> =
                    projection.forward(device, (), &mut (), Begin(()));
                landed.weight
            };
        let _: for<'w> fn(&'w Dense<Hidden, Heads, Head>, &mut Device) -> View<'w> =
            |weight, device| {
                let projection = Affine::new((weight, ()), ());
                let landed: Affine<(View<'_>, ()), ()> =
                    projection.forward(device, (), &mut (), Begin(()));
                landed.weight.0
            };
        let _: for<'w> fn(&View<'w>, &mut Device) -> _ = |weight, device| {
            Matrix::tile::<crate::ops::linear::kernel::Contraction<Hidden, Width<Heads, Head>>>(
                weight, device, 0,
            )
        };
    }
}
