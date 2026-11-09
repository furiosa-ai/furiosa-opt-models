//! Readout input: the live rows of a call, borrowed from a row slice into a compiled envelope.

use crate::{
    compose::Count,
    ops::layout::{Live, axes::Chip},
};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// The rows of a slice, the live rows `0..len` of an `R`-row envelope.
#[derive(DeviceSend)]
pub struct Rows<'a, D: Scalar, W: AxisName, R: AxisName> {
    source: &'a super::RowStorage<W, D>,
    /// The source's row count, which the device reads at the launch's start.
    count: Count,
    shape: PhantomData<R>,
}

impl<D: MaterializableScalar, W: AxisName, R: AxisName> Rows<'_, D, W, R> {
    /// The live rows staged as the first rows of an envelope tensor, whose later rows are
    /// unspecified.
    pub(in crate::ops) fn stage(
        &self,
        device: &mut Device,
    ) -> (HbmTensor<D, Chip, m![R, W]>, Live<R>) {
        let live = Live::<R>::read(&self.count);
        let mut output: HbmTensor<D, Chip, m![R, W]> = HbmTensor::new();
        for row in 0..live.rows() {
            self.source[row].view().to_hbm_view(
                &mut device.tdma,
                output.view_mut().tile::<m![R], 1, m![R = 1 #{!} R, W]>(row),
            );
        }
        (output, live)
    }
}
