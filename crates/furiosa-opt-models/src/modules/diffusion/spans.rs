//! Bind a checked runtime span plan to an adaptive block's normalization and residual gates.

use super::Adaptive;
use crate::Module;
use crate::load::Load;
use crate::ops::index::packed::{Rows, SpanPlan};
use crate::tensor::live::Live;
use crate::tensor::memory::residency::Residency;
use furiosa_opt_std::prelude::*;

/// An adaptive block with request-owned runtime modulation spans.
///
/// The environment is `(&condition, (&spans, env))`. Only the two norms and the gate receive
/// `spans`; modulation, attention and feed-forward receive the original `env`. The residual is
/// an owned [`Live`] tensor throughout. Request preparation must bind its initialized work and
/// the branches' work to the same `spans.work()` interval, including guards. Modality padding
/// does not acquire semantic visibility, and allocation rows beyond work remain unread.
///
/// This wrapper reuses [`Adaptive`]'s execution through borrowed children. It adds no tensor
/// allocation or copy; the numerical leaves retain their ordinary output allocations and
/// exact-zero guard policy. Loading and residency keep the wrapped tree's existing names.
#[derive(DeviceSend, Load, Residency)]
#[residency(children(M))]
pub struct WithSpans<M>(
    #[load(rename = "")]
    #[residency(transparent)]
    pub M,
);

impl<I, M> !crate::Op<I> for WithSpans<M> {}

type Hidden<R, W, C> = Live<Rows<R, W, C>, R>;

/// A numerical child's borrowed view, holding the already decoded plan for this call only.
struct BoundSpans<'m, 'p, M, R, const EDGE: usize> {
    module: &'m M,
    plan: &'p SpanPlan<R, EDGE>,
}

impl<I, M, R, const EDGE: usize> !crate::Op<I> for BoundSpans<'_, '_, M, R, EDGE> {}

impl<'p, R: AxisName, W: AxisName, C: M, N, Modulation, Env, const EDGE: usize>
    Module<(&Hidden<R, W, C>, &Modulation), Env, &mut ()> for BoundSpans<'_, 'p, N, R, EDGE>
where
    N: for<'x, 'm, 's> Module<
            (&'x Rows<R, W, C>, &'m Modulation, &'p SpanPlan<R, EDGE>),
            (),
            &'s mut (),
            Output = Hidden<R, W, C>,
        >,
{
    type Output = Hidden<R, W, C>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        state: &mut (),
        (input, modulation): (&Hidden<R, W, C>, &Modulation),
    ) -> Self::Output {
        self.module
            .forward(device, (), state, (input.value(), modulation, self.plan))
    }
}

impl<'p, R: AxisName, W: AxisName, C: M, G, Gate, Env, const EDGE: usize>
    Module<(&Hidden<R, W, C>, &Hidden<R, W, C>, &Gate), Env, &mut ()>
    for BoundSpans<'_, 'p, G, R, EDGE>
where
    G: for<'x, 'b, 'g, 's> Module<
            (
                &'x Rows<R, W, C>,
                &'b Rows<R, W, C>,
                &'g Gate,
                &'p SpanPlan<R, EDGE>,
            ),
            (),
            &'s mut (),
            Output = Hidden<R, W, C>,
        >,
{
    type Output = Hidden<R, W, C>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        state: &mut (),
        (input, branch, gate): (&Hidden<R, W, C>, &Hidden<R, W, C>, &Gate),
    ) -> Self::Output {
        self.module.forward(
            device,
            (),
            state,
            (input.value(), branch.value(), gate, self.plan),
        )
    }
}

impl<
    's,
    'e,
    'p,
    R: AxisName,
    W: AxisName,
    C: M,
    E,
    ModulationValues,
    GateValues,
    ModulationState,
    AttentionState,
    FeedForwardState,
    Modulation,
    Attention,
    FeedForward,
    Norm,
    Gate,
    FeedForwardNorm,
    Env: Copy,
    const EDGE: usize,
>
    Module<
        Hidden<R, W, C>,
        (&'e E, (&'p SpanPlan<R, EDGE>, Env)),
        &'s mut (ModulationState, AttentionState, FeedForwardState),
    > for WithSpans<Adaptive<Modulation, Attention, FeedForward, Norm, Gate, FeedForwardNorm>>
where
    Modulation: Module<
            &'e E,
            Env,
            &'s mut ModulationState,
            Output = (
                (ModulationValues, GateValues),
                (ModulationValues, GateValues),
            ),
        >,
    Norm: for<'x, 'm, 'd> Module<
            (
                &'x Rows<R, W, C>,
                &'m ModulationValues,
                &'p SpanPlan<R, EDGE>,
            ),
            (),
            &'d mut (),
            Output = Hidden<R, W, C>,
        >,
    FeedForwardNorm: for<'x, 'm, 'd> Module<
            (
                &'x Rows<R, W, C>,
                &'m ModulationValues,
                &'p SpanPlan<R, EDGE>,
            ),
            (),
            &'d mut (),
            Output = Hidden<R, W, C>,
        >,
    Gate: for<'x, 'b, 'g, 'd> Module<
            (
                &'x Rows<R, W, C>,
                &'b Rows<R, W, C>,
                &'g GateValues,
                &'p SpanPlan<R, EDGE>,
            ),
            (),
            &'d mut (),
            Output = Hidden<R, W, C>,
        >,
    Attention:
        for<'x> Module<&'x Hidden<R, W, C>, Env, &'s mut AttentionState, Output = Hidden<R, W, C>>,
    FeedForward: for<'x> Module<&'x Hidden<R, W, C>, Env, &'s mut FeedForwardState, Output = Hidden<R, W, C>>,
{
    type Output = Hidden<R, W, C>;

    fn forward(
        &self,
        device: &mut Device,
        (condition, (plan, env)): (&'e E, (&'p SpanPlan<R, EDGE>, Env)),
        state: &'s mut (ModulationState, AttentionState, FeedForwardState),
        hidden: Hidden<R, W, C>,
    ) -> Self::Output {
        Adaptive {
            modulation: &self.0.modulation,
            attention: &self.0.attention,
            feedforward: &self.0.feedforward,
            norm: BoundSpans {
                module: &self.0.norm,
                plan,
            },
            feedforward_norm: BoundSpans {
                module: &self.0.feedforward_norm,
                plan,
            },
            gate: BoundSpans {
                module: &self.0.gate,
                plan,
            },
        }
        .forward(device, (condition, env), state, hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Op;
    use crate::load::Named;
    use crate::models::diffusion::minimax_h3::{embeddings, transformer};
    use crate::ops::index::packed::{Group, Table, live::SpanInputs};
    use crate::ops::normalization::adaptive::AdaptiveRms;
    use crate::ops::pointwise::gated;
    use crate::tensor::{
        axis,
        form::Weight,
        memory::{Hbm, Memory},
        parallel::Chip,
    };
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    const EPSILON: usize = (1e-5f32).to_bits() as usize;
    type Conditions<W, C = Chip> = (
        (Table<W, C>, Table<W, C>, Table<W, C>),
        (Table<W, C>, Table<W, C>, Table<W, C>),
    );
    type BranchEnv<'p, R, W, C, const EDGE: usize> =
        (&'p SpanPlan<R, EDGE>, &'p Table<W, C>, &'p Table<W, C>);

    /// A fixture supplies already prepared tables; the production modulation child can have
    /// arbitrary state and remains responsible for computing its own tables from the condition.
    #[derive(DeviceSend, Load, Residency)]
    struct Modulation;

    impl<W: AxisName, C: M, Env> Module<&Conditions<W, C>, Env, &mut ()> for Modulation {
        type Output = (
            ((Table<W, C>, Table<W, C>), Table<W, C>),
            ((Table<W, C>, Table<W, C>), Table<W, C>),
        );

        fn forward(
            &self,
            device: &mut Device,
            _: Env,
            _: &mut (),
            condition: &Conditions<W, C>,
        ) -> Self::Output {
            let copy = |device: &mut Device, table: &Table<W, C>| {
                let mut output = Table::new();
                table
                    .view()
                    .to_hbm_view(&mut device.tdma, output.view_mut());
                output
            };
            let ((s1, a1, g1), (s2, a2, g2)) = condition;
            (
                ((copy(device, s1), copy(device, a1)), copy(device, g1)),
                ((copy(device, s2), copy(device, a2)), copy(device, g2)),
            )
        }
    }

    /// These branches read the original env, choosing distinct operands. Their real span Op
    /// preserves the input's work and exercises env routing without a second arithmetic kernel.
    #[derive(DeviceSend, Load, Residency)]
    struct Branch<const WHICH: usize>;

    impl<R: AxisName, W: AxisName, C: M, const EDGE: usize, const WHICH: usize>
        Module<&Hidden<R, W, C>, BranchEnv<'_, R, W, C, EDGE>, &mut ()> for Branch<WHICH>
    {
        type Output = Hidden<R, W, C>;

        fn forward(
            &self,
            device: &mut Device,
            (plan, attention, feedforward): BranchEnv<'_, R, W, C, EDGE>,
            _: &mut (),
            input: &Hidden<R, W, C>,
        ) -> Self::Output {
            let gate = if const { WHICH == 0 } {
                attention
            } else {
                feedforward
            };
            Op::forward(
                &gated::Add,
                device,
                (input.value(), input.value(), gate, plan),
            )
        }
    }

    type Fixture<W, C = Chip> = WithSpans<
        Adaptive<
            Named<"adaln_proj.linear", Modulation>,
            Named<"attn", Branch<0>>,
            Named<"mlp", Branch<1>>,
            Named<"norm1", AdaptiveRms<W, EPSILON, C>>,
            gated::Add,
            Named<"norm2", AdaptiveRms<W, EPSILON, C>>,
        >,
    >;

    type W = axis::Rows<256>;
    type R = axis::Rows<1280>;
    type H3Tree<X> = WithSpans<
        transformer::Block<
            embeddings::Width,
            transformer::Attended,
            transformer::Expanded,
            embeddings::Time,
            X,
        >,
    >;

    #[test]
    fn named_h3_norms_delegate_explicit_operands_and_the_wrapped_tree_projects() {
        type R = axis::Rows<10496>;
        type W = embeddings::Width;
        type C = embeddings::Chips;
        type Input = Hidden<R, W, C>;
        type Norm1 = Named<"norm1", transformer::AdaptiveNorm>;
        type Norm2 = Named<"norm2", transformer::AdaptiveNorm>;
        fn norm<N>()
        where
            for<'m, 'p, 'x, 't, 's> BoundSpans<'m, 'p, N, R, 16>:
                Module<(&'x Input, &'t (Table<W, C>, Table<W, C>)), (), &'s mut (), Output = Input>,
        {
        }
        norm::<Norm1>();
        norm::<Norm2>();
        norm::<Named<"norm1", transformer::AdaptiveNorm<W, &Hbm>>>();
        norm::<Named<"norm2", transformer::AdaptiveNorm<W, &Hbm>>>();

        fn module<T>()
        where
            for<'c, 'p, 's> T: Module<
                    Input,
                    (
                        &'c Conditions<W, C>,
                        (&'p SpanPlan<R, 16>, BranchEnv<'p, R, W, C, 16>),
                    ),
                    &'s mut ((), (), ()),
                    Output = Input,
                >,
        {
        }
        module::<Fixture<W, C>>();

        struct Shapes;
        impl Memory for Shapes {
            type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = ();
        }
        let projected: Option<<WithSpans<transformer::Block> as Residency>::In<Shapes>> = None;
        let _: Option<H3Tree<Shapes>> = projected;
        // Borrowing retains precisely the same wrapper and named child types.
        let borrowed: Option<<WithSpans<transformer::Block> as Residency>::In<&Hbm>> = None;
        let _: Option<H3Tree<&Hbm>> = borrowed;
    }

    #[device(chip = 1)]
    fn block(
        device: &mut Device,
        module: &Fixture<W>,
        condition: &Conditions<W>,
        branch: &(Table<W>, Table<W>),
        header: &SpanInputs<R, Chip, 16>,
        input: Rows<R, W>,
    ) -> Rows<R, W> {
        let plan = header.read();
        let mut state = ((), (), ());
        let output = Module::forward(
            module,
            device,
            (condition, (&plan, (&plan, &branch.0, &branch.1))),
            &mut state,
            plan.work().with(input),
        );
        output.into_value()
    }

    fn value(row: usize, column: usize) -> f64 {
        let sign = if column.is_multiple_of(2) { 1.0 } else { -1.0 };
        sign * ((row % 7 + 1) as f64 / 8.0)
    }

    fn table(which: usize, row: usize) -> f64 {
        match which {
            0 => row as f64 / 8.0 - 0.25,
            1 => row as f64 / 16.0 - 0.125,
            2 => row as f64 / 32.0 + 0.125,
            3 => row as f64 / 16.0 + 0.125,
            4 => row as f64 / 32.0 - 0.25,
            5 => row as f64 / 64.0 + 0.25,
            6 => 0.5,
            7 => -0.25,
            _ => unreachable!(),
        }
    }

    fn rounded(value: f64) -> f64 {
        f64::from(bf16::from_f32(value as f32).to_f32())
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore = "requires an explicitly selected RNGD chip"]
    async fn named_adaptive_block_routes_spans_and_env_without_exposing_guards_or_tails() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mut norm = async |gain| {
            AdaptiveRms::new(Weight::new(
                HostTensor::<bf16, m![W]>::from_vec(std::iter::repeat_n(
                    bf16::from_f32(gain),
                    W::SIZE,
                ))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            ))
        };
        let norm1 = norm(1.0).await;
        let norm2 = norm(0.75).await;
        let module = WithSpans(Adaptive {
            modulation: Named(Modulation),
            attention: Named(Branch::<0>),
            feedforward: Named(Branch::<1>),
            norm: Named(norm1),
            feedforward_norm: Named(norm2),
            gate: gated::Add,
        });
        let mut upload_table = async |which| {
            HostTensor::<bf16, m![Group, W]>::from_vec(
                (0..Group::SIZE * W::SIZE)
                    .map(|at| bf16::from_f32(table(which, at / W::SIZE) as f32)),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap()
        };
        let condition = (
            (
                upload_table(0).await,
                upload_table(1).await,
                upload_table(2).await,
            ),
            (
                upload_table(3).await,
                upload_table(4).await,
                upload_table(5).await,
            ),
        );
        let branch = (upload_table(6).await, upload_table(7).await);
        for lengths in [
            [16, 112, 720],
            [48, 112, 848],
            [256, 112, 720],
            [16, 112, 720],
        ] {
            let packed = lengths.iter().sum::<usize>();
            let plan = SpanPlan::<R, 16>::try_packed(lengths, packed.div_ceil(256) * 256).unwrap();
            let header = SpanInputs::upload(&mut device, &plan).await.unwrap();
            let input = HostTensor::<bf16, m![R, W]>::from_vec((0..R::SIZE * W::SIZE).map(|at| {
                if at / W::SIZE < packed {
                    bf16::from_f32(value(at / W::SIZE, at % W::SIZE) as f32)
                } else {
                    bf16::from_bits(0x7fc1)
                }
            }))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            let output = launch(
                block,
                (&mut device, &module, &condition, &branch, &header, input),
            )
            .await
            .unwrap();
            let bytes = output
                .into_buffer()
                .slice(.., ..plan.work_rows() * W::SIZE * 2)
                .unwrap()
                .to_host(&mut device.pdma)
                .await
                .unwrap();
            let actual: Vec<_> = bytes
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            for row in 0..plan.work_rows() {
                if row >= packed {
                    assert!(
                        actual[row * W::SIZE..(row + 1) * W::SIZE]
                            .iter()
                            .all(|v| v.to_bits() == 0),
                        "guard row {row}"
                    );
                    continue;
                }
                let modality = if row < lengths[0] {
                    1
                } else if row < lengths[0] + lengths[1] {
                    5
                } else {
                    0
                };
                let mut expected: Vec<_> = (0..W::SIZE).map(|column| value(row, column)).collect();
                for (offset, gain, branch_gate) in [(0, 1.0, 0.5), (3, 0.75, -0.25)] {
                    let mean = expected.iter().map(|x| x * x).sum::<f64>() / W::SIZE as f64;
                    let inverse = (mean + f64::from(f32::from_bits(EPSILON as u32)))
                        .sqrt()
                        .recip();
                    for x in &mut expected {
                        let normalized = rounded(
                            *x * inverse * gain * (1.0 + table(offset + 1, modality))
                                + table(offset, modality),
                        );
                        let branch = rounded(normalized + normalized * branch_gate);
                        *x = rounded(*x + branch * table(offset + 2, modality));
                    }
                }
                for (column, want) in expected.into_iter().enumerate() {
                    let got = f64::from(actual[row * W::SIZE + column].to_f32());
                    assert!(
                        got.is_finite() && (got - want).abs() <= want.abs() / 64.0 + 0.003,
                        "row {row} column {column}: {got} vs FP64/BF16 {want}"
                    );
                }
            }
        }
    }
}
