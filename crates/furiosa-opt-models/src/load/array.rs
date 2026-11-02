//! Array elements: element `i` under segment `"i"` in both paths.

use super::{Checkpoint, Children, Context, Footprint, Load, LoadError};

impl<T: Footprint, const N: usize> Footprint for [T; N] {
    const SIZE: usize = N * T::SIZE;
}

impl<'a, T: Load<&'a Checkpoint>, const N: usize> Load<&'a Checkpoint> for [T; N] {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let built = (0..N)
                .map(|i| {
                    let element = cx.field(&i.to_string(), &i.to_string(), |cx| T::load(cx, from));
                    children.ask(element)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let built = children.node(built.into_iter().collect::<Option<Vec<T>>>())?;
            Ok(built.try_into().ok().expect("one element per index"))
        })
    }
}

impl<'t, 's, T, const N: usize> Load<(&'t mut [T; N], &'s Checkpoint)> for &'t mut [T; N]
where
    for<'u> &'u mut T: Load<(&'u mut T, &'s Checkpoint)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut [T; N], &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            into.iter_mut().enumerate().try_for_each(|(i, t)| {
                cx.field(&i.to_string(), &i.to_string(), |cx| {
                    <&mut T as Load<_>>::load(cx, (t, from))
                })
                .map(|_| ())
            })
        })?;
        Ok(into)
    }
}

impl<'a, T: Load<&'a U>, U, const N: usize> Load<&'a [U; N]> for [T; N] {
    fn load(context: &mut Context<'_>, from: &'a [U; N]) -> Result<Self, LoadError> {
        let built = from
            .iter()
            .enumerate()
            .map(|(i, u)| context.field(&i.to_string(), &i.to_string(), |cx| T::load(cx, u)))
            .collect::<Result<Vec<T>, LoadError>>()?;
        Ok(built.try_into().ok().expect("one element per element"))
    }
}

impl<'t, T, const N: usize> Load<(&'t mut [T; N], [T; N])> for &'t mut [T; N]
where
    for<'u> &'u mut T: Load<(&'u mut T, T)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut [T; N], [T; N]),
    ) -> Result<Self, LoadError> {
        into.iter_mut()
            .zip(from)
            .enumerate()
            .try_for_each(|(i, (t, value))| {
                context
                    .field(&i.to_string(), &i.to_string(), |cx| {
                        <&mut T as Load<_>>::load(cx, (t, value))
                    })
                    .map(|_| ())
            })?;
        Ok(into)
    }
}

impl<'t, 'h, T, U, const N: usize> Load<(&'t mut [T; N], &'h [U; N])> for &'t mut [T; N]
where
    for<'u> &'u mut T: Load<(&'u mut T, &'h U)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut [T; N], &'h [U; N]),
    ) -> Result<Self, LoadError> {
        into.iter_mut()
            .zip(from)
            .enumerate()
            .try_for_each(|(i, (t, u))| {
                context
                    .field(&i.to_string(), &i.to_string(), |cx| {
                        <&mut T as Load<_>>::load(cx, (t, u))
                    })
                    .map(|_| ())
            })?;
        Ok(into)
    }
}
