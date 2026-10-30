//! Attention whose queries read the keys and values another layer wrote, without writing them.

use super::Context;
use crate::Module;
use furiosa_opt_std::prelude::Device;

/// Queries attend to K/V written by another layer; this module does not write the cache.
pub struct Shared<Query, Position, Read, Attend, Project> {
    pub query: Query,
    pub positions: Position,
    pub read: Read,
    pub attend: Attend,
    pub output: Project,
}

/// The env is the call's context; the state holds a view of the cache another layer wrote, read
/// only, then the query's, the positions', the read's and the output projection's slots.
impl<
    'e,
    's,
    'c,
    X,
    C,
    PositionOperand,
    WriteOperand,
    ReadOperand,
    ProjectedQuery,
    PositionedQuery,
    K,
    V,
    Y,
    Z,
    QueryState,
    PositionState,
    ReadState,
    ProjectState,
    Query,
    Position,
    Read,
    Attend,
    Project,
>
    Module<
        &X,
        &'e Context<PositionOperand, WriteOperand, ReadOperand>,
        &'s mut (&'c C, QueryState, PositionState, ReadState, ProjectState),
    > for Shared<Query, Position, Read, Attend, Project>
where
    Query: for<'x> Module<
            &'x X,
            &'e Context<PositionOperand, WriteOperand, ReadOperand>,
            &'s mut QueryState,
            Output = ProjectedQuery,
        >,
    Position: Module<
            (ProjectedQuery, &'e PositionOperand),
            &'e Context<PositionOperand, WriteOperand, ReadOperand>,
            &'s mut PositionState,
            Output = PositionedQuery,
        >,
    Read: Module<
            (&'c C, &'e ReadOperand),
            &'e Context<PositionOperand, WriteOperand, ReadOperand>,
            &'s mut ReadState,
            Output = (K, V),
        >,
    Attend: for<'b, 't> Module<
            (&'b PositionedQuery, &'b K, &'b V, &'e ReadOperand),
            &'e Context<PositionOperand, WriteOperand, ReadOperand>,
            &'t mut (),
            Output = Y,
        >,
    Project: for<'x> Module<
            &'x Y,
            &'e Context<PositionOperand, WriteOperand, ReadOperand>,
            &'s mut ProjectState,
            Output = Z,
        >,
{
    type Output = Z;

    fn forward(
        &self,
        device: &mut Device,
        context: &'e Context<PositionOperand, WriteOperand, ReadOperand>,
        (cache, query, positions, read, output): &'s mut (
            &'c C,
            QueryState,
            PositionState,
            ReadState,
            ProjectState,
        ),
        input: &X,
    ) -> Self::Output {
        let env = context;
        let queries = self.query.forward(&mut *device, env, query, input);
        let queries =
            self.positions
                .forward(&mut *device, env, positions, (queries, &context.positions));
        let (keys, values) = self
            .read
            .forward(&mut *device, env, read, (*cache, &context.read));
        let hidden = self.attend.forward(
            &mut *device,
            env,
            &mut (),
            (&queries, &keys, &values, &context.read),
        );
        self.output.forward(device, env, output, &hidden)
    }
}
