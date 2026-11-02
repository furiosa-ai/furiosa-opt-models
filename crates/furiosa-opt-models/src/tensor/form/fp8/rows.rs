use super::*;

/// Folded E4M3 codes and row scales, with input/output geometry independent of storage.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Rows<In: M, Out: M, W, S = ()>(pub W, pub S, PhantomData<fn() -> (In, Out)>);

impl<In: M, Out: M, W, S> Rows<In, Out, W, S> {
    pub const fn new(codes: W, scales: S) -> Self {
        Self(codes, scales, PhantomData)
    }
}

impl<In: M + 'static, Out: M + 'static, W, S> Rows<In, Out, W, S>
where
    W: Load<HostTensor<f8e4m3, m![{ Chip }, { m![{ Out }, { In }] }]>>,
{
    fn place(
        codes: Vec<f8e4m3>,
        scales: impl FnOnce(&mut Context<'_>) -> Result<S, LoadError>,
        context: &mut Context<'_>,
    ) -> Result<Self, LoadError> {
        let codes = HostTensor::from_vec(codes.repeat(Chip::SIZE));
        Ok(Self::new(
            context.field("0", "", |cx| W::load(cx, codes))?,
            context.field("1", "", scales)?,
        ))
    }
}

impl<'a, In: M + 'static, Out: M + 'static, W> Load<&'a Checkpoint> for Rows<In, Out, W, ()>
where
    W: Load<HostTensor<f8e4m3, m![{ Chip }, { m![{ Out }, { In }] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let (codes, rows) = folded::<In, Out>(from, context)?;
        let exact = |(index, code): (usize, f8e4m3)| {
            let value = f64::from(code.to_f32()) * f64::from(rows[index / In::SIZE]);
            let folded = f8e4m3::from_f32(value as f32);
            (f64::from(folded.to_f32()) == value).then_some(folded)
        };
        let codes = codes
            .into_iter()
            .enumerate()
            .map(exact)
            .collect::<Option<Vec<_>>>();
        let codes = codes.ok_or_else(|| LoadError::InvalidData {
            name: context.name(),
            reason: "a row's block scale does not fold into its E4M3 codes exactly, and the \
                form holds no row scales"
                .into(),
        })?;
        Self::place(codes, |_| Ok(()), context)
    }
}

impl<'a, In: M + 'static, Out: M + 'static, W, L: M + 'static> Load<&'a Checkpoint>
    for Rows<In, Out, W, HbmTensor<f32, Chip, L>>
where
    W: Load<HostTensor<f8e4m3, m![{ Chip }, { m![{ Out }, { In }] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let (codes, rows) = folded::<In, Out>(from, context)?;
        let rows = HostTensor::<f32, m![{ Chip }, { Out }]>::from_vec(rows.repeat(Chip::SIZE));
        Self::place(codes, |cx| Load::load(cx, rows), context)
    }
}

/// Each head's squared row scale, repeated for its output tiles.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Squares<T>(pub T);

impl<'a, In: M + 'static, Heads: AxisName, Head: AxisName, W, K: M + 'static> Load<&'a Checkpoint>
    for Rows<In, m![Heads, Head], W, Squares<HbmTensor<f32, Chip, m![Heads, { K }]>>>
where
    W: Load<HostTensor<f8e4m3, m![{ Chip }, { m![{ m![Heads, Head] }, { In }] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let (codes, rows) = folded::<In, m![Heads, Head]>(from, context)?;
        let squares = rows
            .chunks_exact(Head::SIZE)
            .map(|head| (head.iter().all(|&row| row == head[0])).then_some(head[0] * head[0]))
            .collect::<Option<Vec<f32>>>()
            .ok_or_else(|| LoadError::InvalidData {
                name: context.name(),
                reason: "the rows of a head hold different block scales".into(),
            })?;
        let squares = squares
            .iter()
            .flat_map(|&square| std::iter::repeat_n(square, K::SIZE))
            .collect::<Vec<_>>()
            .repeat(Chip::SIZE);
        let squares = HostTensor::<f32, m![{ Chip }, { m![Heads, { K }] }]>::from_vec(squares);
        Self::place(
            codes,
            |cx| Ok(Squares(cx.field("0", "", |cx| Load::load(cx, squares))?)),
            context,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::decoder::{Form, form::BlockScaled};
    use crate::modules::generic::collection::View;
    use crate::tensor::memory::Hbm;

    axes![In = 256, Out = 256];

    /// Each stored projection loads and views as the same borrowed projection, including reborrows.
    #[test]
    fn preserves_projection_residency() {
        fn contract<'a, Owned, Borrowed>()
        where
            Owned: View<2, Layer<'a> = Borrowed> + 'a,
            Borrowed: Load<&'a Owned> + View<2, Layer<'a> = Borrowed> + 'a,
        {
        }
        contract::<
            <BlockScaled as Form>::Query<In, Out, In, Hbm>,
            <BlockScaled as Form>::Query<In, Out, In, &Hbm>,
        >();
        contract::<
            <BlockScaled as Form>::Key<In, Out, In, Hbm>,
            <BlockScaled as Form>::Key<In, Out, In, &Hbm>,
        >();
        contract::<
            <BlockScaled as Form>::Value<In, Out, In, Hbm>,
            <BlockScaled as Form>::Value<In, Out, In, &Hbm>,
        >();
        contract::<
            <BlockScaled as Form>::Output<Out, In, Hbm>,
            <BlockScaled as Form>::Output<Out, In, &Hbm>,
        >();
        contract::<
            <BlockScaled as Form>::GateUp<In, Out, Hbm>,
            <BlockScaled as Form>::GateUp<In, Out, &Hbm>,
        >();
        contract::<
            <BlockScaled as Form>::Down<In, Out, Hbm>,
            <BlockScaled as Form>::Down<In, Out, &Hbm>,
        >();
    }
}
