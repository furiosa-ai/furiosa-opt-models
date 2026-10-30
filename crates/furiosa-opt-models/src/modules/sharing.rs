//! Sharing across layers: the original input later layers see.

use crate::{Branch, Module, Order, Sequential, Then};
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Joins the hidden value with the original input, projects it and runs the layer on it.
pub type Conditioned<Join, Projection, Compute> = Sequential<Then<Join, Projection>, Compute>;

/// Attention sees the original stream as well as the current hidden value.
#[derive(DeviceSend)]
pub struct ConditionedInterleave<Attention, Recurrent, const ORDER: &'static [Branch]> {
    pub attention: Attention,
    pub recurrent: Recurrent,
}

/// The env holds the attention layers' reader, then the recurrent layers'; the state one slot a
/// layer of each collection, each lent again at the layer's turn.
impl<
    X,
    C,
    D,
    Attention,
    Recurrent,
    AttentionEnv: Copy,
    RecurrentEnv: Copy,
    const A: usize,
    const R: usize,
    const ORDER: &'static [Branch],
> Module<X, (AttentionEnv, RecurrentEnv), &mut ([C; A], [D; R])>
    for ConditionedInterleave<[Attention; A], [Recurrent; R], ORDER>
where
    Attention: for<'b, 'c, 't> Module<(&'b X, &'c X), AttentionEnv, &'t mut C, Output = X>,
    Recurrent: for<'b, 't> Module<&'b X, RecurrentEnv, &'t mut D, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        (attention_env, recurrent_env): (AttentionEnv, RecurrentEnv),
        (attention, recurrent): &mut ([C; A], [D; R]),
        original: X,
    ) -> X {
        let order = const { Order::<A, R>::new(ORDER).positions.0 };
        let (mut a, mut r) = (0, 0);
        let mut hidden = None;
        if A + R > 0 {
            #[furiosa_opt_std::prelude::unroll]
            for index in 0..(A + R) {
                let input = match hidden.as_ref() {
                    Some(value) => value,
                    None => &original,
                };
                hidden = Some(if a < A && order[a] == index {
                    let next = self.attention[a].forward(
                        &mut *device,
                        attention_env,
                        &mut attention[a],
                        (input, &original),
                    );
                    a += 1;
                    next
                } else {
                    let next = self.recurrent[r].forward(
                        &mut *device,
                        recurrent_env,
                        &mut recurrent[r],
                        input,
                    );
                    r += 1;
                    next
                });
            }
        }
        match hidden {
            Some(value) => value,
            None => original,
        }
    }
}
