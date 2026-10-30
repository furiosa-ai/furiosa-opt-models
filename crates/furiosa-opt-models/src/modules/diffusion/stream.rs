//! Dual-stream layers: two streams share one joint attention and keep their own modulated
//! feed-forward.

use super::Modulated;
use crate::{First, Module};
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// One stream's modulation, norms, feed-forward and gate in a dual-stream layer.
#[derive(DeviceSend)]
pub struct Stream<Modulation, Norm, FeedForward, Gate> {
    pub modulation: Modulation,
    pub norms: [Norm; 2],
    pub feedforward: FeedForward,
    pub gate: Gate,
}

/// What a dual-stream layer asks of each stream, so it bounds the streams by these steps, not by
/// their children: hidden `X`, condition `E`, env `Env`, modulation and feed-forward states.
pub(crate) trait Modulate<Env, X, E, ModulationState, FeedForwardState> {
    /// The shift and scale a norm reads.
    type Norm;
    /// The gate a residual reads.
    type Gate;

    /// The attention residual's and the feed-forward residual's modulation, in that order.
    fn modulate(
        &self,
        device: &mut Device,
        env: Env,
        state: ModulationState,
        condition: &E,
    ) -> ((Self::Norm, Self::Gate), (Self::Norm, Self::Gate));

    fn normalize(&self, device: &mut Device, env: Env, hidden: &X, modulation: &Self::Norm) -> X;

    fn gate(&self, device: &mut Device, env: Env, input: &X, branch: &X, gate: &Self::Gate) -> X;

    /// The modulated, gated feed-forward residual.
    fn feedforward(
        &self,
        device: &mut Device,
        env: Env,
        state: FeedForwardState,
        hidden: &X,
        modulation: (&Self::Norm, &Self::Gate),
    ) -> X;
}

impl<X, E, M, G, ModulationState, FeedForwardState, Modulation, Norm, FeedForward, Gate, Env: Copy>
    Modulate<Env, X, E, ModulationState, FeedForwardState>
    for Stream<Modulation, Norm, FeedForward, Gate>
where
    Modulation: for<'e> Module<&'e E, Env, ModulationState, Output = ((M, G), (M, G))>,
    Norm: for<'d, 'x, 'm> Module<(&'x X, &'m M), Env, &'d mut (), Output = X>,
    FeedForward: for<'x> Module<&'x X, Env, FeedForwardState, Output = X>,
    Gate: for<'d, 'a, 'b, 'g> Module<(&'a X, &'b X, &'g G), Env, &'d mut (), Output = X>,
{
    type Norm = M;
    type Gate = G;

    fn modulate(
        &self,
        device: &mut Device,
        env: Env,
        state: ModulationState,
        condition: &E,
    ) -> ((M, G), (M, G)) {
        self.modulation.forward(device, env, state, condition)
    }

    fn normalize(&self, device: &mut Device, env: Env, hidden: &X, modulation: &M) -> X {
        self.norms[0].forward(device, env, &mut (), (hidden, modulation))
    }

    fn gate(&self, device: &mut Device, env: Env, input: &X, branch: &X, gate: &G) -> X {
        self.gate
            .forward(device, env, &mut (), (input, branch, gate))
    }

    fn feedforward(
        &self,
        device: &mut Device,
        env: Env,
        state: FeedForwardState,
        hidden: &X,
        modulation: (&M, &G),
    ) -> X {
        Modulated(&self.norms[1], &self.feedforward, &self.gate).forward(
            device,
            env,
            state,
            (hidden, modulation),
        )
    }
}

/// Updates two streams through one joint attention, then a modulated feed-forward per stream.
#[derive(DeviceSend)]
pub struct Dual<Left, Right, Attention, Extra = First> {
    pub streams: (Left, Right),
    pub attention: Attention,
    pub extra: Extra,
}

/// The env holds the condition, the joint attention's context and the extra stage's operand; each
/// stream, the attention and the extra have slots.
impl<
    's,
    'e,
    X,
    Y,
    E,
    S: Copy,
    LeftModulationState,
    LeftFeedForwardState,
    RightModulationState,
    RightFeedForwardState,
    AttentionState,
    ExtraState,
    Left,
    Right,
    Attention,
    Extra,
    Env: Copy,
>
    Module<
        (X, Y),
        (&'e E, (Env, S)),
        &'s mut (
            (
                (LeftModulationState, LeftFeedForwardState),
                (RightModulationState, RightFeedForwardState),
            ),
            AttentionState,
            ExtraState,
        ),
    > for Dual<Left, Right, Attention, Extra>
where
    Left: Modulate<Env, X, E, &'s mut LeftModulationState, &'s mut LeftFeedForwardState>,
    Right: Modulate<Env, Y, E, &'s mut RightModulationState, &'s mut RightFeedForwardState>,
    Attention: for<'a> Module<&'a (X, Y), Env, &'s mut AttentionState, Output = (X, Y)>,
    Extra: for<'a, 'b> Module<(X, (&'a X, &'b E, S)), Env, &'s mut ExtraState, Output = X>,
{
    type Output = (X, Y);

    fn forward(
        &self,
        device: &mut Device,
        (condition, (env, extra)): (&'e E, (Env, S)),
        (
            ((left_modulation, left_feedforward), (right_modulation, right_feedforward)),
            attention,
            extra_state,
        ): &'s mut (
            (
                (LeftModulationState, LeftFeedForwardState),
                (RightModulationState, RightFeedForwardState),
            ),
            AttentionState,
            ExtraState,
        ),
        (left, right): (X, Y),
    ) -> Self::Output {
        let (image, text) = &self.streams;
        let ((ln1, lg1), (ln2, lg2)) =
            image.modulate(&mut *device, env, left_modulation, condition);
        let ((rn1, rg1), (rn2, rg2)) =
            text.modulate(&mut *device, env, right_modulation, condition);
        let normalized = (
            image.normalize(&mut *device, env, &left, &ln1),
            text.normalize(&mut *device, env, &right, &rn1),
        );
        let (attention, context) =
            self.attention
                .forward(&mut *device, env, attention, &normalized);
        let updated = image.gate(&mut *device, env, &left, &attention, &lg1);
        let left = self.extra.forward(
            &mut *device,
            env,
            extra_state,
            (updated, (&left, condition, extra)),
        );
        let right = text.gate(&mut *device, env, &right, &context, &rg1);
        let left = image.feedforward(&mut *device, env, left_feedforward, &left, (&ln2, &lg2));
        let right = text.feedforward(device, env, right_feedforward, &right, (&rn2, &rg2));
        (left, right)
    }
}

/// The final joint layer reads text keys and values but does not update the text stream.
#[derive(DeviceSend)]
pub struct JointFinal<Modulation, ImageNorm, TextNorm, Attention, FeedForward, Gate, Extra = First>
{
    pub modulation: Modulation,
    pub image_norms: [ImageNorm; 2],
    pub text_norm: TextNorm,
    pub attention: Attention,
    pub feedforward: FeedForward,
    pub gate: Gate,
    pub extra: Extra,
}

/// The env holds the condition, the joint attention's context and the extra stage's operand; each
/// part but the norms and the gate has a slot.
impl<
    's,
    'e,
    X,
    Y,
    E,
    S: Copy,
    M,
    G,
    ModulationState,
    AttentionState,
    FeedForwardState,
    ExtraState,
    Modulation,
    ImageNorm,
    TextNorm,
    Attention,
    FeedForward,
    Gate,
    Extra,
    Env: Copy,
>
    Module<
        &(X, Y),
        (&'e E, (Env, S)),
        &'s mut (
            ModulationState,
            AttentionState,
            FeedForwardState,
            ExtraState,
        ),
    > for JointFinal<Modulation, ImageNorm, TextNorm, Attention, FeedForward, Gate, Extra>
where
    Modulation: Module<&'e E, Env, &'s mut ModulationState, Output = ((M, G), (M, G))>,
    ImageNorm: for<'a, 'b, 'd> Module<(&'a X, &'b M), Env, &'d mut (), Output = X>,
    TextNorm: for<'a, 'b, 'd> Module<(&'a Y, &'b E), Env, &'d mut (), Output = Y>,
    Attention: for<'a> Module<&'a (X, Y), Env, &'s mut AttentionState, Output = X>,
    FeedForward: for<'x> Module<&'x X, Env, &'s mut FeedForwardState, Output = X>,
    Gate: for<'a, 'b, 'c, 'd> Module<(&'a X, &'b X, &'c G), Env, &'d mut (), Output = X>,
    Extra: for<'a, 'b> Module<(X, (&'a X, &'b E, S)), Env, &'s mut ExtraState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        (condition, (env, extra)): (&'e E, (Env, S)),
        (modulation, attention, feedforward, extra_state): &'s mut (
            ModulationState,
            AttentionState,
            FeedForwardState,
            ExtraState,
        ),
        (image, text): &(X, Y),
    ) -> X {
        let ((attention_norm, attention_gate), (mlp_norm, mlp_gate)) =
            self.modulation
                .forward(&mut *device, env, modulation, condition);
        let normalized = (
            self.image_norms[0].forward(&mut *device, env, &mut (), (image, &attention_norm)),
            self.text_norm
                .forward(&mut *device, env, &mut (), (text, condition)),
        );
        let attention = self
            .attention
            .forward(&mut *device, env, attention, &normalized);
        let hidden = self.gate.forward(
            &mut *device,
            env,
            &mut (),
            (image, &attention, &attention_gate),
        );
        let hidden = self.extra.forward(
            &mut *device,
            env,
            extra_state,
            (hidden, (image, condition, extra)),
        );
        Modulated(&self.image_norms[1], &self.feedforward, &self.gate).forward(
            device,
            env,
            feedforward,
            (&hidden, (&mlp_norm, &mlp_gate)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;
    use std::cell::RefCell;

    struct Modulation(&'static str, ((i32, i32), (i32, i32)));

    impl Module<&i32, &RefCell<Vec<&'static str>>, &mut ()> for Modulation {
        type Output = ((i32, i32), (i32, i32));

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            _: &i32,
        ) -> Self::Output {
            trace.borrow_mut().push(self.0);
            self.1
        }
    }

    struct Norm(&'static str);

    impl Module<(&i32, &i32), &RefCell<Vec<&'static str>>, &mut ()> for Norm {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            (hidden, norm): (&i32, &i32),
        ) -> i32 {
            trace.borrow_mut().push(self.0);
            hidden + norm
        }
    }

    struct Mlp(&'static str);

    impl Module<&i32, &RefCell<Vec<&'static str>>, &mut ()> for Mlp {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            hidden: &i32,
        ) -> i32 {
            trace.borrow_mut().push(self.0);
            hidden * 2
        }
    }

    struct Gate(&'static str);

    impl Module<(&i32, &i32, &i32), &RefCell<Vec<&'static str>>, &mut ()> for Gate {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            (hidden, branch, gate): (&i32, &i32, &i32),
        ) -> i32 {
            trace.borrow_mut().push(self.0);
            hidden + branch * gate
        }
    }

    struct Attention;

    type Trace = RefCell<Vec<&'static str>>;

    impl Module<&(i32, i32), &Trace, &mut ()> for Attention {
        type Output = (i32, i32);

        fn forward(
            &self,
            _: &mut Device,
            trace: &Trace,
            _: &mut (),
            (left, right): &(i32, i32),
        ) -> Self::Output {
            trace.borrow_mut().push("attention");
            (left + right, left - right)
        }
    }

    struct Extra;

    impl Module<(i32, (&i32, &i32, i32)), &RefCell<Vec<&'static str>>, &mut ()> for Extra {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<&'static str>>,
            _: &mut (),
            (hidden, (original, condition, extra)): (i32, (&i32, &i32, i32)),
        ) -> i32 {
            trace.borrow_mut().push("extra");
            hidden + original + condition + extra
        }
    }

    /// Joint attention observes both normalized streams before either residual is updated.
    #[test]
    fn preserves_joint_updates() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let model = Dual {
            streams: (
                Stream {
                    modulation: Modulation("left modulation", ((10, 2), (20, 3))),
                    norms: [Norm("left norm"), Norm("left norm")],
                    feedforward: Mlp("left mlp"),
                    gate: Gate("left gate"),
                },
                Stream {
                    modulation: Modulation("right modulation", ((30, 5), (40, 7))),
                    norms: [Norm("right norm"), Norm("right norm")],
                    feedforward: Mlp("right mlp"),
                    gate: Gate("right gate"),
                },
            ),
            attention: Attention,
            extra: Extra,
        };
        let trace = &RefCell::new(Vec::new());
        assert_eq!(
            model.forward(device, (&1, (trace, 4)), &mut Default::default(), (2, 3)),
            (813, -970)
        );
        assert_eq!(
            *trace.borrow(),
            [
                "left modulation",
                "right modulation",
                "left norm",
                "right norm",
                "attention",
                "left gate",
                "extra",
                "right gate",
                "left norm",
                "left mlp",
                "left gate",
                "right norm",
                "right mlp",
                "right gate",
            ]
        );
    }
}
