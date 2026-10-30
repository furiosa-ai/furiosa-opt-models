//! Recurrent mixers with caller-owned convolution windows and recurrence state, and the projection,
//! causal convolution and gating around their selective scan.

use core::marker::PhantomData;

use crate::Module;
use crate::ops::linear::Affine;
use crate::ops::recurrent::{Length, Rows};
use furiosa_opt_std::prelude::*;

/// Per-layer state: the convolution window and the recurrent scan's accumulated state.
#[derive(DeviceSend)]
pub struct Ssm<Convolution, Recurrence> {
    pub convolution: Convolution,
    pub recurrence: Recurrence,
}

/// Projects and convolves segmented input, selects its time-step rank and B and C, scans, then
/// gates and projects the scan output.
#[derive(DeviceSend)]
pub struct Mamba<Projection, Convolution, Selection, Time, Recurrence, Gate, Output> {
    pub projection: Projection,
    pub convolution: Convolution,
    /// The time-step rank, B and C of each convolved row.
    pub selection: Selection,
    /// The time steps from their rank.
    pub time: Time,
    pub recurrence: Recurrence,
    pub gate: Gate,
    pub output: Output,
}

/// The env is the segment lengths every recurrent layer reads; the state holds the selection's
/// slot, the time steps' and the layer's state.
impl<
    's,
    'c,
    'e,
    X,
    L,
    H,
    R,
    Y,
    G,
    Z,
    K,
    B,
    C,
    T,
    V,
    W,
    O,
    SelectionState,
    TimeState,
    Projection,
    Convolution,
    Selection,
    Time,
    Recurrence,
    Gate,
    Output,
> Module<&X, &'e L, &'s mut (SelectionState, TimeState, &'c mut Ssm<H, R>)>
    for Mamba<Projection, Convolution, Selection, Time, Recurrence, Gate, Output>
where
    Projection: for<'x, 't> Module<&'x X, &'e L, &'t mut (), Output = (Y, G)>,
    Convolution: for<'a, 'b, 't> Module<(&'a Y, &'e L, &'b mut H), &'e L, &'t mut (), Output = Z>,
    Selection: for<'z> Module<&'z Z, &'e L, &'s mut SelectionState, Output = (K, B, C)>,
    Time: for<'k> Module<&'k K, &'e L, &'s mut TimeState, Output = T>,
    Recurrence: for<'a, 'b, 't> Module<
            ((&'a Z, &'a T, &'a B, &'a C), &'e L, &'b mut R),
            &'e L,
            &'t mut (),
            Output = V,
        >,
    Gate: for<'a, 'b, 't> Module<(&'a V, &'b G), &'e L, &'t mut (), Output = W>,
    Output: for<'x, 't> Module<&'x W, &'e L, &'t mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        segments: &'e L,
        (selection, time, state): &'s mut (SelectionState, TimeState, &'c mut Ssm<H, R>),
        hidden: &X,
    ) -> O {
        let env = segments;
        let (hidden, gate) = self.projection.forward(&mut *device, env, &mut (), hidden);
        let hidden = self.convolution.forward(
            &mut *device,
            env,
            &mut (),
            (&hidden, segments, &mut state.convolution),
        );
        let (rank, b, c) = self
            .selection
            .forward(&mut *device, env, selection, &hidden);
        let time = self.time.forward(&mut *device, env, time, &rank);
        let hidden = self.recurrence.forward(
            &mut *device,
            env,
            &mut (),
            ((&hidden, &time, &b, &c), segments, &mut state.recurrence),
        );
        let hidden = self
            .gate
            .forward(&mut *device, env, &mut (), (&hidden, &gate));
        self.output.forward(device, env, &mut (), &hidden)
    }
}

/// A Mamba layer whose time steps come from its input, and B and C from the convolution.
#[derive(DeviceSend)]
pub struct Mamba2<Projection, Time, Convolution, Recurrence, Gate, Output> {
    pub projection: Projection,
    /// The time steps of each input row.
    pub time: Time,
    pub convolution: Convolution,
    pub recurrence: Recurrence,
    pub gate: Gate,
    pub output: Output,
}

/// The env is the segment lengths; the state holds the time steps' slot and the layer's state.
impl<
    's,
    'c,
    'e,
    X,
    L,
    H,
    R,
    Y,
    G,
    T,
    Z,
    B,
    C,
    V,
    W,
    O,
    TimeState,
    Projection,
    Time,
    Convolution,
    Recurrence,
    Gate,
    Output,
> Module<&X, &'e L, &'s mut (TimeState, &'c mut Ssm<H, R>)>
    for Mamba2<Projection, Time, Convolution, Recurrence, Gate, Output>
where
    Projection: for<'x, 't> Module<&'x X, &'e L, &'t mut (), Output = (Y, G)>,
    Time: for<'x> Module<&'x X, &'e L, &'s mut TimeState, Output = T>,
    Convolution:
        for<'a, 'b, 't> Module<(&'a Y, &'e L, &'b mut H), &'e L, &'t mut (), Output = (Z, B, C)>,
    Recurrence: for<'a, 'b, 't> Module<
            ((&'a Z, &'a T, &'a B, &'a C), &'e L, &'b mut R),
            &'e L,
            &'t mut (),
            Output = V,
        >,
    Gate: for<'a, 'b, 't> Module<(&'a V, &'b G), &'e L, &'t mut (), Output = W>,
    Output: for<'x, 't> Module<&'x W, &'e L, &'t mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        segments: &'e L,
        (time, state): &'s mut (TimeState, &'c mut Ssm<H, R>),
        input: &X,
    ) -> O {
        let env = segments;
        let (hidden, gate) = self.projection.forward(&mut *device, env, &mut (), input);
        let time = self.time.forward(&mut *device, env, time, input);
        let (hidden, b, c) = self.convolution.forward(
            &mut *device,
            env,
            &mut (),
            (&hidden, segments, &mut state.convolution),
        );
        let hidden = self.recurrence.forward(
            &mut *device,
            env,
            &mut (),
            ((&hidden, &time, &b, &c), segments, &mut state.recurrence),
        );
        let hidden = self
            .gate
            .forward(&mut *device, env, &mut (), (&hidden, &gate));
        self.output.forward(device, env, &mut (), &hidden)
    }
}

/// Normalizes convolution-derived Q/K, performs a gated delta recurrence and gates its output.
pub struct GatedDelta<Projection, Convolution, Normalize, Recurrence, Gate, Output> {
    pub projection: Projection,
    pub convolution: Convolution,
    pub normalize: Normalize,
    pub recurrence: Recurrence,
    pub gate: Gate,
    pub output: Output,
}

/// The env is the segment lengths; the state is the layer's state.
impl<
    's,
    'c,
    'e,
    X,
    L,
    H,
    R,
    Y,
    G,
    B,
    A,
    Q,
    K,
    V,
    U,
    Z,
    O,
    Projection,
    Convolution,
    Normalize,
    Recurrence,
    Gate,
    Output,
> Module<&X, &'e L, &'s mut &'c mut Ssm<H, R>>
    for GatedDelta<Projection, Convolution, Normalize, Recurrence, Gate, Output>
where
    Projection: for<'x, 't> Module<&'x X, &'e L, &'t mut (), Output = (Y, G, B, A)>,
    Convolution:
        for<'a, 'b, 't> Module<(&'a Y, &'e L, &'b mut H), &'e L, &'t mut (), Output = (Q, K, V)>,
    Normalize: for<'t> Module<(Q, K), &'e L, &'t mut (), Output = (Q, K)>,
    Recurrence: for<'a, 'b, 't> Module<
            (&'a Q, &'a K, &'a V, &'a B, &'a A, &'e L, &'b mut R),
            &'e L,
            &'t mut (),
            Output = U,
        >,
    Gate: for<'a, 'b, 't> Module<(&'a U, &'b G), &'e L, &'t mut (), Output = Z>,
    Output: for<'x, 't> Module<&'x Z, &'e L, &'t mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        segments: &'e L,
        state: &'s mut &'c mut Ssm<H, R>,
        hidden: &X,
    ) -> O {
        let env = segments;
        let (hidden, gate, beta, decay) =
            self.projection.forward(&mut *device, env, &mut (), hidden);
        let (query, key, value) = self.convolution.forward(
            &mut *device,
            env,
            &mut (),
            (&hidden, segments, &mut state.convolution),
        );
        let (query, key) = self
            .normalize
            .forward(&mut *device, env, &mut (), (query, key));
        let hidden = self.recurrence.forward(
            &mut *device,
            env,
            &mut (),
            (
                &query,
                &key,
                &value,
                &beta,
                &decay,
                segments,
                &mut state.recurrence,
            ),
        );
        let hidden = self
            .gate
            .forward(&mut *device, env, &mut (), (&hidden, &gate));
        self.output.forward(device, env, &mut (), &hidden)
    }
}

/// Applies independent multiplicative gates before and after a stateful short convolution.
pub struct ShortConvolution<Projection, Multiply, Convolution, Output> {
    pub projection: Projection,
    pub multiply: Multiply,
    pub convolution: Convolution,
    pub output: Output,
}

/// The env is the segment lengths; the state is the layer's convolution window.
impl<'s, 'c, 'e, X, L, H, B, Y, C, Z, U, V, O, Projection, Multiply, Convolution, Output>
    Module<&X, &'e L, &'s mut &'c mut H>
    for ShortConvolution<Projection, Multiply, Convolution, Output>
where
    Projection: for<'x, 't> Module<&'x X, &'e L, &'t mut (), Output = (B, Y, C)>,
    Multiply: for<'a, 'b, 't> Module<(&'a B, &'b Y), &'e L, &'t mut (), Output = Z>
        + for<'a, 'b, 't> Module<(&'a U, &'b C), &'e L, &'t mut (), Output = V>,
    Convolution: for<'a, 'b, 't> Module<(&'a Z, &'e L, &'b mut H), &'e L, &'t mut (), Output = U>,
    Output: for<'x, 't> Module<&'x V, &'e L, &'t mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        segments: &'e L,
        window: &'s mut &'c mut H,
        hidden: &X,
    ) -> O {
        let env = segments;
        let (before, hidden, after) = self.projection.forward(&mut *device, env, &mut (), hidden);
        let hidden = self
            .multiply
            .forward(&mut *device, env, &mut (), (&before, &hidden));
        let hidden = self.convolution.forward(
            &mut *device,
            env,
            &mut (),
            (&hidden, segments, &mut **window),
        );
        let hidden = self
            .multiply
            .forward(&mut *device, env, &mut (), (&hidden, &after));
        self.output.forward(device, env, &mut (), &hidden)
    }
}

/// Repeats low-level updates between high-level updates, coupling their hidden values by addition.
#[derive(DeviceSend)]
pub struct Hierarchy<Initial, Low, High, Add, const LOW: usize, const HIGH: usize> {
    pub initial: Initial,
    pub low: Low,
    pub high: High,
    pub add: Add,
}

/// The state holds the low module's slot, then the high module's; each update takes its slot
/// again, so neither holds borrowed state.
impl<X, LowState, HighState, Initial, Low, High, Add, const L: usize, const H: usize, Env: Copy>
    Module<X, Env, &mut (LowState, HighState)> for Hierarchy<Initial, Low, High, Add, L, H>
where
    Initial: for<'s> Module<(), Env, &'s mut (), Output = X>,
    Low: for<'s> Module<X, Env, &'s mut LowState, Output = X>,
    High: for<'s> Module<X, Env, &'s mut HighState, Output = X>,
    Add: for<'a, 'b, 's> Module<(&'a X, &'b X), Env, &'s mut (), Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (low_state, high_state): &mut (LowState, HighState),
        mut high: X,
    ) -> X {
        let mut low = self.initial.forward(&mut *device, env, &mut (), ());
        #[unroll]
        for _ in 0..H {
            #[unroll]
            for _ in 0..L {
                let input = self.add.forward(&mut *device, env, &mut (), (&low, &high));
                low = self.low.forward(&mut *device, env, &mut *low_state, input);
            }
            let input = self.add.forward(&mut *device, env, &mut (), (&high, &low));
            high = self
                .high
                .forward(&mut *device, env, &mut *high_state, input);
        }
        high
    }
}

/// Project each row into packed recurrent features, with zero or one bias tensor.
#[derive(DeviceSend)]
pub struct Projection<Weight, Bias>(pub Affine<Weight, Bias>);

/// Combined width of the input and gate projections, each Intermediate features wide.
#[derive(Clone, Debug)]
pub struct ProjectionWidth<Intermediate>(PhantomData<Intermediate>);

impl<Intermediate: AxisName> AxisName for ProjectionWidth<Intermediate> {
    const NAME: Ident = Ident::new("ProjectionWidth");
    const SIZE: usize = 2 * Intermediate::SIZE;
}

/// Packed [gate, x, B, C] projection width.
#[derive(Clone, Debug)]
pub struct Projected<Intermediate, Groups, State>(PhantomData<(Intermediate, Groups, State)>);

impl<Intermediate: AxisName, Groups: AxisName, State: AxisName> AxisName
    for Projected<Intermediate, Groups, State>
{
    const NAME: Ident = Ident::new("Projected");
    const SIZE: usize = 2 * Intermediate::SIZE + 2 * Groups::SIZE * State::SIZE;
}

/// Packed x features followed by the grouped B and C state coefficients, which one causal
/// convolution convolves.
#[derive(Clone, Debug)]
pub struct Channels<Intermediate, Groups, State>(PhantomData<(Intermediate, Groups, State)>);

impl<Intermediate: AxisName, Groups: AxisName, State: AxisName> AxisName
    for Channels<Intermediate, Groups, State>
{
    const NAME: Ident = Ident::new("Channels");
    const SIZE: usize = Intermediate::SIZE + 2 * Groups::SIZE * State::SIZE;
}

impl<Width: AxisName, Intermediate: AxisName, Count: AxisName, const BIAS: usize, Env>
    Module<&Rows<Width, Count>, Env, &mut ()>
    for Projection<
        HbmTensor<bf16, m![1], m![2, Intermediate, Width]>,
        [HbmTensor<bf16, m![1], m![2, Intermediate]>; BIAS],
    >
{
    type Output = (Rows<Intermediate, Count>, Rows<Intermediate, Count>);

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: &Rows<Width, Count>) -> Self::Output {
        const { assert!(BIAS <= 1) };
        todo!("NPU input and gate projection")
    }
}

impl<
    Width: AxisName,
    Intermediate: AxisName,
    Groups: AxisName,
    State: AxisName,
    Count: AxisName,
    const BIAS: usize,
    Env,
> Module<&Rows<Width, Count>, Env, &mut ()>
    for Projection<
        HbmTensor<bf16, m![1], m![{ Symbol<Projected<Intermediate, Groups, State>> }, Width]>,
        [HbmTensor<bf16, m![1], m![{ Symbol<Projected<Intermediate, Groups, State>> }]>; BIAS],
    >
{
    type Output = (
        Rows<Channels<Intermediate, Groups, State>, Count>,
        Rows<Intermediate, Count>,
    );

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: &Rows<Width, Count>) -> Self::Output {
        const {
            assert!(BIAS <= 1);
        }
        todo!()
    }
}

/// Causal depthwise convolution and SiLU, updating only live history rows.
#[derive(DeviceSend)]
pub struct Convolution<Weight, Bias> {
    /// Depthwise weights shaped [width, 1, kernel].
    pub weight: Weight,
    /// Zero or one per-feature bias tensor.
    pub bias: Bias,
}

/// Oldest-to-newest K-1 rows retained by a K-tap causal convolution.
#[derive(Clone, Debug)]
pub struct History<Kernel>(PhantomData<Kernel>);

impl<Kernel: AxisName> AxisName for History<Kernel> {
    const NAME: Ident = Ident::new("History");
    const SIZE: usize = Kernel::SIZE - 1;
}

impl<Width: AxisName, Kernel: AxisName, Count: AxisName, const BIAS: usize, Env>
    Module<
        (
            &Rows<Width, Count>,
            &Length,
            &mut HbmTensor<f32, m![1], m![Width, Kernel]>,
        ),
        Env,
        &mut (),
    >
    for Convolution<
        HbmTensor<bf16, m![1], m![Width, 1, Kernel]>,
        [HbmTensor<bf16, m![1], m![Width]>; BIAS],
    >
{
    type Output = Rows<Width, Count>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: (
            &Rows<Width, Count>,
            &Length,
            &mut HbmTensor<f32, m![1], m![Width, Kernel]>,
        ),
    ) -> Self::Output {
        const { assert!(BIAS <= 1) };
        todo!("NPU causal depthwise convolution and SiLU, updating only live history rows")
    }
}

/// Convolves packed x, B and C channels and returns them apart: x per feature, B and C per group.
impl<
    Intermediate: AxisName,
    Groups: AxisName,
    State: AxisName,
    Kernel: AxisName,
    Count: AxisName,
    const BIAS: usize,
    Env,
>
    Module<
        (
            &Rows<Channels<Intermediate, Groups, State>, Count>,
            &Length,
            &mut HbmTensor<
                f32,
                m![1],
                m![{ Symbol<History<Kernel>> }, { Symbol<Channels<Intermediate, Groups, State>> }],
            >,
        ),
        Env,
        &mut (),
    >
    for Convolution<
        HbmTensor<bf16, m![1], m![{ Symbol<Channels<Intermediate, Groups, State>> }, 1, Kernel]>,
        [HbmTensor<bf16, m![1], m![{ Symbol<Channels<Intermediate, Groups, State>> }]>; BIAS],
    >
{
    type Output = (
        Rows<Intermediate, Count>,
        HbmTensor<bf16, m![1], m![Count, Groups, State]>,
        HbmTensor<bf16, m![1], m![Count, Groups, State]>,
    );

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: (
            &Rows<Channels<Intermediate, Groups, State>, Count>,
            &Length,
            &mut HbmTensor<
                f32,
                m![1],
                m![{ Symbol<History<Kernel>> }, { Symbol<Channels<Intermediate, Groups, State>> }],
            >,
        ),
    ) -> Self::Output {
        const {
            assert!(BIAS <= 1);
            assert!(Kernel::SIZE > 1);
        }
        todo!()
    }
}

/// Multiply FP32 recurrence output by SiLU(gate), rounding once to BF16.
#[derive(Default, DeviceSend)]
pub struct Gate;

impl<Width: AxisName, Count: AxisName, Env>
    Module<(&Rows<Width, Count, f32>, &Rows<Width, Count>), Env, &mut ()> for Gate
{
    type Output = Rows<Width, Count>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: (&Rows<Width, Count, f32>, &Rows<Width, Count>),
    ) -> Self::Output {
        todo!("NPU FP32 SiLU gate multiplication, rounded once to BF16")
    }
}

/// Which of gating and normalization a gated RMS norm applies first; a trait, as the order is a
/// type parameter of the norm whose constant its kernel reads at compile time.
pub trait Order {
    /// Whether normalization precedes gating, as a checkpoint's `norm_before_gate` declares.
    const NORM_FIRST: bool;
}

/// Normalization, then SiLU gating.
pub struct NormFirst;

/// SiLU gating, then normalization.
pub struct GateFirst;

impl Order for NormFirst {
    const NORM_FIRST: bool = true;
}

impl Order for GateFirst {
    const NORM_FIRST: bool = false;
}

/// SiLU gating and weighted RMS normalization in `O`'s order, rounded once to BF16: HF's
/// `MambaRMSNormGated`.
#[derive(DeviceSend)]
pub struct RmsNormGated<Weight, Epsilon, O> {
    /// Per-feature normalization multiplier.
    pub weight: Weight,
    /// FP32 scalar added to the mean square before taking its square root.
    pub epsilon: Epsilon,
    /// The order of gating and normalization.
    pub order: PhantomData<O>,
}

impl<Width: AxisName, Count: AxisName, O: Order, Env>
    Module<(&Rows<Width, Count, f32>, &Rows<Width, Count>), Env, &mut ()>
    for RmsNormGated<HbmTensor<bf16, m![1], m![Width]>, HbmTensor<f32, m![1], m![1]>, O>
{
    type Output = Rows<Width, Count>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: (&Rows<Width, Count, f32>, &Rows<Width, Count>),
    ) -> Self::Output {
        todo!()
    }
}
