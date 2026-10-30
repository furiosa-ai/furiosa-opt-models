//! A cache stage that writes this call's keys and values and hands attention only the rows a
//! selector picks.

use super::{Attention, Context};
use crate::Module;
use furiosa_opt_std::prelude::Device;

/// Writes this call's keys and values; attention reads only the history rows a selector picks.
pub struct Selected<Write, Select> {
    pub write: Write,
    pub selector: Select,
}

/// The whole cache, the rows a selector picked in it and the read operand.
pub struct Selection<C, I, R>(pub C, pub I, pub R);

/// The state holds the layer's view of its cache, then the selector's slot; the input holds where
/// this call's keys and values are written and which history the selector picks from.
impl<'s, 'b, 'x, Q, K, V, C, WriteOperand, ReadOperand, I, SelectState, Write, Select, Env: Copy>
    Module<((Q, K, V), (&'x WriteOperand, &'x ReadOperand)), Env, &'s mut (&'b mut C, SelectState)>
    for Selected<Write, Select>
where
    Write: for<'w, 'k, 'u> Module<
            (&'w mut C, &'x WriteOperand, (&'k K, &'k V)),
            Env,
            &'u mut (),
            Output = (),
        >,
    Select: for<'q> Module<(&'q Q, &'s C, &'x ReadOperand), Env, &'s mut SelectState, Output = I>,
{
    type Output = (Q, Selection<&'s C, I, &'x ReadOperand>);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (cache, selector): &'s mut (&'b mut C, SelectState),
        ((query, keys, values), (write, read)): ((Q, K, V), (&'x WriteOperand, &'x ReadOperand)),
    ) -> Self::Output {
        self.write.forward(
            &mut *device,
            env,
            &mut (),
            (&mut **cache, write, (&keys, &values)),
        );
        let cache = &**cache;
        let selected = self
            .selector
            .forward(device, env, selector, (&query, cache, read));
        (query, Selection(cache, selected, read))
    }
}

impl<'c, 'r, Q, C, I, R, O, M> super::Attend<Q, Selection<&'c C, I, &'r R>> for M
where
    M: for<'q, 't> Module<(&'q Q, &'c C, &'q I, &'r R), (), &'t mut (), Output = O>,
{
    type Output = O;

    fn attend(&self, device: &mut Device, (query, history): (&Q, Selection<&'c C, I, &'r R>)) -> O {
        self.forward(
            device,
            (),
            &mut (),
            (query, history.0, &history.1, history.2),
        )
    }
}

/// A selected attention reads its call's context from the env.
impl<
    'x,
    'e,
    X,
    PositionOperand,
    WriteOperand,
    ReadOperand,
    S,
    Qkv,
    Write,
    Select,
    Attend,
    Project,
    Out,
> Module<&'x X, &'e Context<PositionOperand, WriteOperand, ReadOperand>, S>
    for Attention<Qkv, Selected<Write, Select>, Attend, Project>
where
    Self: Module<
            (
                (
                    (&'x X, &'e PositionOperand),
                    (&'e WriteOperand, &'e ReadOperand),
                ),
                (),
            ),
            &'e Context<PositionOperand, WriteOperand, ReadOperand>,
            S,
            Output = Out,
        >,
{
    type Output = Out;

    fn forward(
        &self,
        device: &mut Device,
        context: &'e Context<PositionOperand, WriteOperand, ReadOperand>,
        state: S,
        input: &'x X,
    ) -> Self::Output {
        let operands = (&context.write, &context.read);
        self.forward(
            device,
            context,
            state,
            (((input, &context.positions), operands), ()),
        )
    }
}
