//! Tuple elements: element `i` under tree segment `"i"`, adding no checkpoint segment.

use super::{Checkpoint, Children, Context, Footprint, Load, LoadError};

macro_rules! tuples {
    ($(($($T:ident $U:ident $t:ident $i:tt),+);)+) => {$(
        impl<$($T: Footprint),+> Footprint for ($($T,)+) {
            const SIZE: usize = 0 $(+ $T::SIZE)+;
        }

        impl<'a, $($T: Load<&'a Checkpoint>),+> Load<&'a Checkpoint>
            for ($($T,)+)
        {
            fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
                context.checkpoint(from, |cx| {
                    let mut children = Children::default();
                    let built = ($(
                        children.ask(cx.field(stringify!($i), "", |cx| $T::load(cx, from)))?,
                    )+);
                    let built = match built {
                        ($(Some($t),)+) => Some(($($t,)+)),
                        _ => None,
                    };
                    children.node(built)
                })
            }
        }

        impl<'t, 's, $($T),+> Load<(&'t mut ($($T,)+), &'s Checkpoint)>
            for &'t mut ($($T,)+)
        where
            $(for<'u> &'u mut $T: Load<(&'u mut $T, &'s Checkpoint)>,)+
        {
            fn load(
                context: &mut Context<'_>,
                (into, from): (&'t mut ($($T,)+), &'s Checkpoint),
            ) -> Result<Self, LoadError> {
                context.checkpoint(from, |cx| {
                    $(cx.field(stringify!($i), "", |cx| {
                        <&mut $T as Load<_>>::load(cx, (&mut into.$i, from))
                    })?;)+
                    Ok(())
                })?;
                Ok(into)
            }
        }

        impl<'a, $($T: Load<&'a $U>, $U),+> Load<&'a ($($U,)+)>
            for ($($T,)+)
        {
            fn load(context: &mut Context<'_>, from: &'a ($($U,)+)) -> Result<Self, LoadError> {
                Ok(($(context.field(stringify!($i), "", |cx| $T::load(cx, &from.$i))?,)+))
            }
        }

        impl<'t, $($T),+> Load<(&'t mut ($($T,)+), ($($T,)+))> for &'t mut ($($T,)+)
        where
            $(for<'u> &'u mut $T: Load<(&'u mut $T, $T)>,)+
        {
            fn load(
                context: &mut Context<'_>,
                (into, from): (&'t mut ($($T,)+), ($($T,)+)),
            ) -> Result<Self, LoadError> {
                let ($($t,)+) = from;
                $(context.field(stringify!($i), "", |cx| {
                    <&mut $T as Load<_>>::load(cx, (&mut into.$i, $t))
                })?;)+
                Ok(into)
            }
        }
    )+};
}

tuples! {
    (A0 B0 a0 0, A1 B1 a1 1);
    (A0 B0 a0 0, A1 B1 a1 1, A2 B2 a2 2);
}
