//! Select the caller's pooling row, then apply the optional text projection.
use super::Rows;
use crate::Module;
use crate::ops::index::{
    gather::{Gather, RowStorage},
    table::Offset,
};
use crate::tensor::axis::Single;

use furiosa_opt_std::prelude::*;

/// Select a caller-specified row, then apply the text projection.
#[derive(DeviceSend)]
pub struct Pool<Project> {
    /// Row selection by byte offset.
    pub select: Gather,
    /// Projection of the selected row, or identity for unprojected features.
    pub project: Project,
}

impl<Project> Pool<Project> {
    /// Pair the supplied projection with row selection by byte offset.
    pub fn new(project: Project) -> Self {
        Self {
            select: Gather,
            project,
        }
    }
}

impl<Width: AxisName, Output: AxisName, Project, Env: Copy> Module<&Context<Width>, Env, &mut ()>
    for Pool<Project>
where
    Project: for<'d> Module<Rows<Width, Single>, Env, &'d mut (), Output = Rows<Output, Single>>,
{
    type Output = Rows<Output, Single>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        context: &Context<Width>,
    ) -> Self::Output {
        let selected = self.select.forward(
            &mut *device,
            env,
            &mut (),
            (&context.hidden, &context.offset),
        );
        self.project.forward(device, env, &mut (), selected)
    }
}

impl<Width: AxisName, Size: AxisName, Count: AxisName, Project, Env: Copy>
    Module<(&Rows<Width, Count>, &Offset), Env, &mut ()> for Pool<Project>
where
    Project: for<'d> Module<Rows<Width, Single>, Env, &'d mut (), Output = Rows<Size, Single>>,
{
    type Output = Rows<Size, Single>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        input: (&Rows<Width, Count>, &Offset),
    ) -> Self::Output {
        let selected = self.select.forward(&mut *device, env, &mut (), input);
        self.project.forward(device, env, &mut (), selected)
    }
}

/// Opaque row storage for the device entry; compositions may pass sized rows and an offset.
#[derive(DeviceSend)]
pub struct Context<Width: AxisName> {
    /// Contiguous feature rows whose count is determined by the caller.
    pub hidden: RowStorage<Width>,
    /// Byte offset of the selected row from the start of `hidden`.
    pub offset: Offset,
}
