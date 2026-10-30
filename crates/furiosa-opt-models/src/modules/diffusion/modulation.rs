//! A gated residual whose norm the condition modulates, which each modulated block is built from,
//! and the norm and the splits of the condition it reads.

use crate::Module;
use crate::tensor::image::{Rows, Vector};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// A gated residual: normalizes with the modulation, runs the branch and gates its output in.
#[derive(DeviceSend)]
pub struct Modulated<Norm, Branch, Gate>(pub Norm, pub Branch, pub Gate);

/// The branch's slot is the state; the norm reads the modulation, the gate its gate.
impl<'m, 'g, X, Y, Z, O, M, G, S, Norm, Branch, Gate, Env: Copy>
    Module<(&X, (&'m M, &'g G)), Env, S> for Modulated<Norm, Branch, Gate>
where
    Norm: for<'d, 'x> Module<(&'x X, &'m M), Env, &'d mut (), Output = Y>,
    Branch: for<'y> Module<&'y Y, Env, S, Output = Z>,
    Gate: for<'d, 'x, 'z> Module<(&'x X, &'z Z, &'g G), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        (hidden, (modulation, gate)): (&X, (&'m M, &'g G)),
    ) -> O {
        let normalized = self
            .0
            .forward(&mut *device, env, &mut (), (hidden, modulation));
        let branch = self.1.forward(&mut *device, env, state, &normalized);
        self.2
            .forward(device, env, &mut (), (hidden, &branch, gate))
    }
}

/// Adaptive normalization shift followed by scale.
pub type Modulation<Width> = (Vector<Width>, Vector<Width>);

/// Attention and feed-forward modulation, each paired with its residual gate.
pub type Gates<Width> = (
    (Modulation<Width>, Vector<Width>),
    (Modulation<Width>, Vector<Width>),
);

/// Apply affine-free layer norm, then multiply by `1 + scale` and add shift.
#[derive(DeviceSend)]
pub struct AdaptiveNorm {
    /// Stabilizer added to the normalization variance.
    pub epsilon: HbmTensor<f32, m![1], m![1]>,
}

impl<W: AxisName, R: AxisName, Env> Module<(&Rows<W, R>, &Modulation<W>), Env, &mut ()>
    for AdaptiveNorm
{
    type Output = Rows<W, R>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: (&Rows<W, R>, &Modulation<W>),
    ) -> Self::Output {
        todo!("NPU affine-free layer norm followed by (1 + scale) multiplication and shift")
    }
}

/// Split attention shift/scale/gate followed by feed-forward shift/scale/gate.
#[derive(DeviceSend)]
pub struct SplitGates<Width>(pub PhantomData<Width>);

impl<Width> Default for SplitGates<Width> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Width: AxisName, Packed: AxisName, Env> Module<&Vector<Packed>, Env, &mut ()>
    for SplitGates<Width>
{
    type Output = Gates<Width>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: &Vector<Packed>) -> Self::Output {
        const { assert!(Packed::SIZE == 6 * Width::SIZE) };
        todo!("NPU split of attention and feedforward shift, scale, and residual gate")
    }
}

/// A conditioning payload packs a modulation's shift, then its scale.
pub struct ShiftFirst;

/// A conditioning payload packs a modulation's scale, then its shift.
pub struct ScaleFirst;

/// Split conditioning packed in `Order` into `(shift, scale)`.
#[derive(DeviceSend)]
pub struct SplitModulation<Width, Order = ShiftFirst>(pub PhantomData<(Width, Order)>);

impl<Width, Order> Default for SplitModulation<Width, Order> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Width: AxisName, Packed: AxisName, Order, Env> Module<&Vector<Packed>, Env, &mut ()>
    for SplitModulation<Width, Order>
{
    type Output = Modulation<Width>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: &Vector<Packed>) -> Self::Output {
        const { assert!(Packed::SIZE == 2 * Width::SIZE) };
        todo!(
            "NPU split of adaptive normalization into (shift, scale), respecting SCALE_FIRST input order"
        )
    }
}

/// Split conditioning into normalization shift, scale and residual gate.
#[derive(DeviceSend)]
pub struct SplitGate<Width>(pub PhantomData<Width>);

impl<Width> Default for SplitGate<Width> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Width: AxisName, Packed: AxisName, Env> Module<&Vector<Packed>, Env, &mut ()>
    for SplitGate<Width>
{
    type Output = (Modulation<Width>, Vector<Width>);

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: &Vector<Packed>) -> Self::Output {
        const { assert!(Packed::SIZE == 3 * Width::SIZE) };
        todo!("NPU split of normalization shift, scale, and residual gate")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;
    use std::cell::RefCell;

    struct Norm;

    impl Module<(&i32, &i32), &RefCell<Vec<&'static str>>, &mut ()> for Norm {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            (hidden, modulation): (&i32, &i32),
        ) -> i32 {
            trace.borrow_mut().push("norm");
            hidden + modulation
        }
    }

    struct Branch;

    impl Module<&i32, &RefCell<Vec<&'static str>>, &mut i32> for Branch {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            state: &mut i32,
            hidden: &i32,
        ) -> i32 {
            trace.borrow_mut().push("branch");
            *state += hidden;
            *state
        }
    }

    struct Gate;

    impl Module<(&i32, &i32, &i32), &RefCell<Vec<&'static str>>, &mut ()> for Gate {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            (hidden, branch, gate): (&i32, &i32, &i32),
        ) -> i32 {
            trace.borrow_mut().push("gate");
            hidden + branch * gate
        }
    }

    /// Modulation affects the branch input; gating adds that branch to the original hidden value.
    #[test]
    fn gates_conditioned_branch() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let trace = &RefCell::new(Vec::new());
        let mut state = 11;
        let output =
            Modulated(Norm, Branch, Gate).forward(device, trace, &mut state, (&2, (&3, &7)));
        assert_eq!((output, state), (114, 16));
        assert_eq!(*trace.borrow(), ["norm", "branch", "gate"]);
    }
}
